"""Offline unit tests for `quadvox.cloud.colab_resilience` and `runners/run_d3dv3_4view.py`."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import trimesh
from PIL import Image

from quadvox.cloud import colab_resilience
from runners.run_d3dv3_4view import main as run_d3dv3_main
from runners.run_d3dv3_4view import run_colab_d3dv3
from runners.run_s2_prop import main as run_s2_prop_main
from runners.run_s2_uv_bake import main as run_s2_uv_bake_main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ssh_key_and_openssh_config_parse() -> None:
    tmp_root = REPO_ROOT / "runs" / "_test_tmp_cloud"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as td:
        root = Path(td)
        fake_home = root / "home"
        fake_home.mkdir()
        with mock.patch("pathlib.Path.home", return_value=fake_home):
            key_path = colab_resilience.ensure_colab_ssh_key(repo_root=root)
        assert key_path.is_file()
        assert (key_path.stat().st_mode & 0o777) == 0o600

        cfg_path = colab_resilience.write_ssh_config(
            root / "staging",
            "colab",
            "quadvox-test-123",
            key_path,
            config_path=root / "staging" / "colab_sessions.json",
        )
        assert (cfg_path.stat().st_mode & 0o777) == 0o600
        if shutil.which("ssh"):
            proc = subprocess.run(
                ["ssh", "-G", "-F", str(cfg_path), "colab-runtime"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=True,
            )
            assert "user root" in proc.stdout


def test_allocate_gpu_session_fast_uses_80_95_and_cleans_orphans() -> None:
    seen_timeouts: list[float] = []
    unassigned: list[str] = []

    def fake_run_bounded(colab_bin: str, args, **kwargs):
        if args and args[0] == "new":
            seen_timeouts.append(float(kwargs["timeout_s"]))
            if "--gpu" in args and args[args.index("--gpu") + 1] == "A100":
                raise RuntimeError("ReadTimeout: Read timed out after 80s")
            if "--gpu" in args and args[args.index("--gpu") + 1] == "H100":
                return subprocess.CompletedProcess([colab_bin, *args], 0, "[colab] Session READY.\n", "")
        return subprocess.CompletedProcess([colab_bin, *args], 0, "", "")

    list_calls = iter(
        [
            [{"endpoint": "ep-keep", "accelerator": "A100", "variant": "GPU"}],
            [
                {"endpoint": "ep-keep", "accelerator": "A100", "variant": "GPU"},
                {"endpoint": "ep-orphan", "accelerator": "A100", "variant": "GPU"},
            ],
        ]
    )

    with mock.patch.object(
        colab_resilience, "list_server_assignments", side_effect=lambda _cb: next(list_calls)
    ), mock.patch.object(
        colab_resilience, "collect_known_session_endpoints", return_value={"ep-keep": "ext-1"}
    ), mock.patch.object(
        colab_resilience,
        "unassign_endpoint_direct",
        side_effect=lambda _cb, ep: unassigned.append(ep) or True,
    ), mock.patch("subprocess.run", return_value=subprocess.CompletedProcess(["colab"], 0, "", "")):
        sku, pre_eps = colab_resilience.allocate_gpu_session_fast(
            "colab", "quadvox-test", ("A100", "H100", "G4"), run_bounded=fake_run_bounded
        )

    assert sku == "H100"
    assert pre_eps == {"ep-keep"}
    assert seen_timeouts == [95.0, 95.0]
    assert unassigned == ["ep-orphan"]


def test_run_d3dv3_4view_dry_run_bundle_layer1_and_layer2() -> None:
    work = REPO_ROOT / "runs" / "_test_tmp_d3dv3_dry"
    work.mkdir(parents=True, exist_ok=True)
    coarse = work / "coarse.glb"
    sec = work / "sec_coarse.glb"
    trimesh.creation.icosphere(subdivisions=1, radius=0.4).export(str(coarse))
    trimesh.creation.icosphere(subdivisions=1, radius=0.4).export(str(sec))

    views = {}
    for vn in ("front", "left", "back", "right"):
        vp = work / f"{vn}.png"
        Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8)).save(vp)
        views[vn] = vp

    res_l1 = run_colab_d3dv3(
        out_dir=work / "out_l1",
        profile_name="object",
        coarse_path=coarse,
        secondary_coarse_path=None,
        view_paths=views,
        dry_run=True,
    )
    assert res_l1["dry_run"] is True
    assert res_l1["chunk_count"] >= 1

    res_l2 = run_colab_d3dv3(
        out_dir=work / "out_l2",
        profile_name="symmetric_humanoid",
        coarse_path=coarse,
        secondary_coarse_path=sec,
        view_paths=views,
        dry_run=True,
    )
    spec_l2 = json.loads((work / "out_l2" / "_colab_staging" / "job_spec.json").read_text(encoding="utf-8"))
    assert spec_l2["secondary_coarse_rel"] == "inputs/secondary_coarse.glb"
    assert spec_l2["profile"]["s1"]["coarse_splice"]["enabled"] is True
    assert spec_l2["profile"]["s1"]["d3dv3_fusion"]["front_roi_box"]["target_view"] == "front"
    assert res_l2["dry_run"] is True


def test_colab_d3dv3_worker_manifest_and_non_dry_run_contract() -> None:
    import tarfile
    from runners import colab_d3dv3_4view_worker, run_d3dv3_4view

    work = REPO_ROOT / "runs" / "_test_tmp_d3dv3_contract"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    try:
        # 1. Verify _package_outputs emits a manifest consumable by download_manifest_and_chunks_resilient
        out_pkg = work / "remote_outputs"
        out_pkg.mkdir(parents=True)
        (out_pkg / "fused_1536.glb").write_bytes(b"glTF-dummy-payload")
        (out_pkg / "d3dv3_report.json").write_text('{"ok": true}\n', encoding="utf-8")
        manifest = colab_d3dv3_4view_worker._package_outputs(out_pkg)
        assert manifest["archive_name"] == "outputs_bundle.tar.gz"
        assert manifest["total_size"] > 0
        assert "sha256" in manifest
        assert all("size" in ch and "sha256" in ch for ch in manifest["chunks"])

        dl_stage = work / "dl_stage"
        archive_local, loaded_manifest, transport = colab_resilience.download_manifest_and_chunks_resilient(
            "colab",
            "test-sess",
            "/remote/outputs",
            "output_manifest.json",
            dl_stage / "chunks",
            staging_dir=dl_stage,
            prefer_ssh=False,
            fallback_download_fn=lambda rf, lf: shutil.copy2(out_pkg / Path(rf).name, lf),
        )
        assert transport == "colab-download"
        assert archive_local.is_file()
        assert loaded_manifest["archive_name"] == "outputs_bundle.tar.gz"

        # 2. Verify run_colab_d3dv3(..., dry_run=False) handles upload, sliced daemon tail, download, & safe extract
        coarse = work / "coarse.glb"
        trimesh.creation.icosphere(subdivisions=1, radius=0.3).export(str(coarse))
        views = {}
        for vn in ("front", "left", "back", "right"):
            vp = work / f"{vn}.png"
            Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8)).save(vp)
            views[vn] = vp

        sample_tar = work / "sample_outputs.tar.gz"
        with tarfile.open(sample_tar, "w:gz") as tf:
            tf.add(out_pkg / "fused_1536.glb", arcname="fused_1536.glb")
            tf.add(out_pkg / "d3dv3_report.json", arcname="d3dv3_report.json")

        seen_upload_kwargs: dict[str, object] = {}
        seen_download_args: dict[str, object] = {}
        exec_calls = {"n": 0}
        cleanup_calls: list[str] = []

        def fake_upload(cb, sess, files, remote_dir, **kwargs):
            seen_upload_kwargs.update({"remote_dir": remote_dir, **kwargs})
            return "rsync"

        def fake_subproc(cmd, **kwargs):
            exec_calls["n"] += 1
            out_s = (
                "[d3dv3-tail] SLICE_CONTINUE=1\n"
                if exec_calls["n"] == 1
                else "[d3dv3-tail] WORKER_RC=0\n"
            )
            return subprocess.CompletedProcess(cmd, 0, out_s, "")

        def fake_download(cb, sess, remote_dir, manifest_name, dest_dir, **kwargs):
            seen_download_args.update(
                {
                    "remote_dir": remote_dir,
                    "manifest_name": manifest_name,
                    "dest_dir": dest_dir,
                    **kwargs,
                }
            )
            dest_dir.mkdir(parents=True, exist_ok=True)
            copied = dest_dir / "outputs_bundle.tar.gz"
            shutil.copy2(sample_tar, copied)
            return copied, {"archive_name": "outputs_bundle.tar.gz"}, "rsync"

        run_out = work / "run_out"
        with mock.patch.object(
            run_d3dv3_4view, "allocate_gpu_session_fast", return_value=("A100", set())
        ), mock.patch.object(
            run_d3dv3_4view, "upload_files_resilient", side_effect=fake_upload
        ), mock.patch.object(
            run_d3dv3_4view.subprocess, "run", side_effect=fake_subproc
        ), mock.patch.object(
            run_d3dv3_4view, "download_manifest_and_chunks_resilient", side_effect=fake_download
        ), mock.patch.object(
            run_d3dv3_4view, "cleanup_session_and_orphans", side_effect=lambda cb, s, p: cleanup_calls.append(s)
        ):
            res = run_d3dv3_4view.run_colab_d3dv3(
                out_dir=run_out,
                profile_name="object",
                coarse_path=coarse,
                secondary_coarse_path=None,
                view_paths=views,
                dry_run=False,
            )

        assert exec_calls["n"] == 2
        assert seen_upload_kwargs["staging_dir"] == run_out / "_colab_staging"
        assert seen_download_args["remote_dir"] == run_d3dv3_4view.REMOTE_OUTPUTS_DIR
        assert seen_download_args["manifest_name"] == "output_manifest.json"
        assert seen_download_args["staging_dir"] == run_out / "_colab_staging"
        assert (run_out / "colab_out" / "fused_1536.glb").is_file()
        assert (run_out / "colab_out" / "d3dv3_report.json").is_file()
        assert "fused_1536.glb" in res["extracted_files"]
        assert len(cleanup_calls) == 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_run_s1_pixal3d_cli_and_runmeta_contract() -> None:
    from quadvox.contracts import JobConfig, MeshMetrics, RunMeta, SeedCandidate
    from runners import run_s1_pixal3d

    work = REPO_ROOT / "runs" / "_test_tmp_s1_pixal_cli"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    try:
        views_dir = work / "views"
        views_dir.mkdir()
        for idx, az in enumerate((0, 90, 180, 270)):
            Image.fromarray(np.full((64, 64, 4), (200, 150, 140, 255), dtype=np.uint8)).save(
                views_dir / f"view0{idx}_azim{az:03d}.png"
            )
        out_dir = work / "out"
        rc = run_s1_pixal3d.main(
            [
                "--input",
                str(views_dir),
                "--out-dir",
                str(out_dir),
                "--cellocut-resolution",
                "768",
                "--qef",
                "--smooth-iters",
                "2",
                "--dry-run",
            ]
        )
        assert rc == 0
        spec = json.loads((out_dir / "_colab_staging" / "job_spec.json").read_text(encoding="utf-8"))
        assert spec["remesh_resolution"] == 768
        assert spec["cellocut_qef"] is True
        assert spec["smooth_iters"] == 2

        entry_src = (out_dir / "_colab_staging" / "exec_entry.py").read_text(encoding="utf-8")
        assert "colab_s1_pixal3d_worker.py" in entry_src
        assert "colab_remote_runner.py" not in entry_src

        mm = MeshMetrics(
            vertex_count=10,
            face_count=16,
            open_boundary_edges=0,
            boundary_loops=0,
            pinhole_loops=0,
            large_opening_loops=0,
            non_manifold_edges=0,
            non_manifold_vertices=0,
            degenerate_faces=0,
            duplicate_faces=0,
            connected_components=1,
            is_watertight=True,
            is_pinhole_free=True,
            bbox_min=[-0.1, 0.0, -0.1],
            bbox_max=[0.1, 1.0, 0.1],
            height=1.0,
        )
        job_cfg = JobConfig(
            mode="mv",
            input_path=str(views_dir),
            out_dir=str(out_dir),
            backend="path_a",
            repair_method="cellocut",
        )
        cand = SeedCandidate(seed=42, raw_glb="raw.glb", elapsed_s=12.0, peak_vram_gb=8.0, metrics=mm)
        meta = RunMeta(
            job=job_cfg,
            selected_seed=42,
            candidates=[cand],
            raw_metrics=mm,
            repaired_metrics=mm,
            backend_version="path_a",
            elapsed_s=15.0,
            peak_vram_gb=8.0,
            gpu_name="NVIDIA A100-SXM4-40GB",
            outputs={"raw_glb": "raw.glb", "clean_white_glb": "clean_white.glb"},
        )
        raw = meta.to_dict()
        raw["gpu_name"] = f"A100 ({raw['gpu_name']})"
        reloaded = RunMeta.from_dict(raw)
        assert reloaded.gpu_name == "A100 (NVIDIA A100-SXM4-40GB)"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_s2_prop_uv_bake_and_d3dv3_cli_dry_run_contracts(capsys) -> None:
    work = REPO_ROOT / "runs" / "_test_tmp_s2_runners"
    work.mkdir(parents=True, exist_ok=True)
    try:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
        mesh.vertices[:, 1] += 0.5
        high_glb = work / "high.glb"
        low_glb = work / "low.glb"
        mesh.export(str(high_glb))
        mesh.export(str(low_glb))

        views_dir = work / "views"
        views_dir.mkdir()
        for vn in ("front", "left", "back", "right"):
            Image.fromarray(np.full((32, 32, 3), 180, dtype=np.uint8)).save(views_dir / f"{vn}.png")

        # 1. run_s2_prop.py --dry-run and --dry-run-bundle CLI contracts
        for flag in ("--dry-run", "--dry-run-bundle"):
            out_prop = work / f"prop_{flag.lstrip('-')}"
            rc_prop = run_s2_prop_main(
                [
                    "--high-glb",
                    str(high_glb),
                    "--out-dir",
                    str(out_prop),
                    "--gpu-order",
                    "A100,H100,G4",
                    "--seeds",
                    "42,43",
                    flag,
                ]
            )
            assert rc_prop == 0
            prop_out = json.loads(capsys.readouterr().out)
            assert prop_out["gpu_order"] == ["A100", "H100", "G4"]
            assert prop_out["no_hardcoded_tmp"] is True
            assert (out_prop / "_colab_staging" / "job_spec.json").is_file()
            spec_prop = json.loads((out_prop / "_colab_staging" / "job_spec.json").read_text(encoding="utf-8"))
            assert spec_prop["lato2_seeds"] == [42, 43]
            assert spec_prop["high_glb"] == "inputs/high.glb"

        # 2. run_s2_uv_bake.py --dry-run and --dry-run-bundle CLI contracts
        for flag in ("--dry-run", "--dry-run-bundle"):
            out_uv = work / f"uv_{flag.lstrip('-')}"
            rc_uv = run_s2_uv_bake_main(
                [
                    "--high-mesh",
                    str(high_glb),
                    "--low-uv-mesh",
                    str(low_glb),
                    "--out-dir",
                    str(out_uv),
                    "--gpu-order",
                    "A100,H100,G4",
                    "--framing-mode",
                    "character",
                    flag,
                ]
            )
            assert rc_uv == 0
            uv_dry = json.loads((out_uv / "colab_uv_dry_run.json").read_text(encoding="utf-8"))
            assert uv_dry["dry_run"] is True
            assert uv_dry["dry_run_bundle"] is True
            assert uv_dry["chunk_count"] >= 1
            uv_spec = json.loads((out_uv / "_colab_staging" / "job_spec.json").read_text(encoding="utf-8"))
            assert uv_spec["jobs"][0]["framing_mode"] == "character"

        # 3. run_d3dv3_4view.py --profile symmetric_humanoid --compare-crevice-keep --dry-run
        out_d3 = work / "d3dv3_dry"
        rc_d3 = run_d3dv3_main(
            [
                "--profile",
                "symmetric_humanoid",
                "--coarse",
                str(high_glb),
                "--front",
                str(views_dir / "front.png"),
                "--left",
                str(views_dir / "left.png"),
                "--back",
                str(views_dir / "back.png"),
                "--right",
                str(views_dir / "right.png"),
                "--out-dir",
                str(out_d3),
                "--gpu-order",
                "A100,H100,G4",
                "--compare-crevice-keep",
                "--dry-run",
            ]
        )
        assert rc_d3 == 0
        d3_dry = json.loads((out_d3 / "d3dv3_dry_run.json").read_text(encoding="utf-8"))
        assert d3_dry["compare_crevice_keep"] is True
        d3_spec = json.loads((out_d3 / "_colab_staging" / "job_spec.json").read_text(encoding="utf-8"))
        assert d3_spec["compare_crevice_keep"] is True
        assert d3_spec["profile"]["s1"]["d3dv3_fusion"]["front_roi_box"]["name"] == "front_head_roi"
        assert d3_spec["profile"]["s2"]["symmetry_x"] is True
    finally:
        shutil.rmtree(work, ignore_errors=True)
