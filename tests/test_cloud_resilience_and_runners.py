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
from runners.run_d3dv3_4view import run_colab_d3dv3

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
