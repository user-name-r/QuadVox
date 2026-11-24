"""Remote Colab GPU worker for 4-View Direct3D-V3 1536^3 Voxel Fusion (Layer 1 + Layer 2).

Runs strictly inside `/content/cache` (never `/tmp`). Executes:
  1. Optional Layer-2 OpenVDB coarse mesh splicing (`coarse_splice.enabled`)
  2. 4-view Direct3D-V3 (512 -> 1024 -> 1536) lossless 8^3 voxel rotation fusion
  3. Optional S1 repair (CelloCut / inplace_clean) & normalization
  4. Chunked output packaging with SHA-256 verification
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

CACHE_ROOT = Path("/content/cache")
WORK_DIR = CACHE_ROOT / "work"
OUTPUTS_DIR = CACHE_ROOT / "outputs"
SCRATCH_DIR = CACHE_ROOT / "scratch"
LOGS_DIR = CACHE_ROOT / "logs"
D3DV3_DIR = CACHE_ROOT / "Direct3D-V3"
D3DV3_VENV = CACHE_ROOT / "d3dv3_venv"
D3DV3_REPO_URL = "https://github.com/DreamTechAI/Direct3D-V3.git"
MAX_CHUNK_BYTES = 10 * 1024 * 1024

for _p in (WORK_DIR, D3DV3_DIR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _base_colab_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(SCRATCH_DIR)
    env["TEMP"] = str(SCRATCH_DIR)
    env["TMP"] = str(SCRATCH_DIR)
    env["PYTHONUNBUFFERED"] = "1"
    env["TQDM_DISABLE"] = "1"
    pp = [str(WORK_DIR)]
    if D3DV3_DIR.exists():
        pp.append(str(D3DV3_DIR))
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pp)
    if extra:
        env.update(extra)
    return env


def _run_step(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout:
        for line in proc.stdout.splitlines()[-25:]:
            print(f"  {line}", flush=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Bootstrap step failed (rc={proc.returncode}): {' '.join(cmd)}")
    return proc


def _runtime_imports_ready(need_pymeshlab: bool = False) -> bool:
    try:
        if D3DV3_DIR.exists() and str(D3DV3_DIR) not in sys.path:
            sys.path.insert(0, str(D3DV3_DIR))
        import cudavoxelize  # noqa: F401
        import direct3d  # noqa: F401
        import fillhole_cpp  # noqa: F401
        import flash_attn  # noqa: F401
        import pyfqmr  # noqa: F401
        import scipy  # noqa: F401
        import timm  # noqa: F401
        import torch  # noqa: F401
        import trimesh  # noqa: F401
        if need_pymeshlab:
            import pymeshlab  # noqa: F401
        return True
    except Exception:
        return False


def _detect_cuda_build_env() -> tuple[dict[str, str], dict[str, Any]]:
    import torch

    cap = list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else [8, 0]
    total_gb = (
        round(float(torch.cuda.get_device_properties(0).total_memory) / (1024.0**3), 2)
        if torch.cuda.is_available()
        else 0.0
    )
    info = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda or "",
        "device_capability": cap,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_gb": total_gb,
    }
    extra = {
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""),
        "TORCH_CUDA_ARCH_LIST": f"{cap[0]}.{cap[1]}",
        "FLASH_ATTN_CUDA_ARCHS": f"{cap[0]}{cap[1]}",
        "MAX_JOBS": str(min(os.cpu_count() or 4, 4)),
        "NVCC_THREADS": "2",
        "CXXFLAGS": "-std=c++20",
    }
    return extra, info


def _find_matching_flash_attn_wheel(py_tag: str, torch_ver: str, cu_ver: str, cxx11_abi: bool) -> list[str]:
    cu_digits = cu_ver.replace(".", "")
    cu_short = "cu" + cu_digits[:3]
    cu_major = "cu" + cu_ver.split(".")[0] if cu_ver else "cu12"
    tm_parts = torch_ver.split("+")[0].split(".")
    torch_mm = f"torch{tm_parts[0]}.{tm_parts[1]}" if len(tm_parts) >= 2 else ""
    abi_str = f"cxx11abi{'TRUE' if cxx11_abi else 'FALSE'}"
    urls: list[str] = []
    for repo in ("mjun0812/flash-attention-prebuild-wheels", "Dao-AILab/flash-attention"):
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{repo}/releases?per_page=30",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                rels = json.loads(resp.read().decode("utf-8"))
            for rel in rels:
                for asset in rel.get("assets", []):
                    name = asset.get("name", "")
                    if not (name.endswith(".whl") and "linux_x86_64" in name and py_tag in name and torch_mm in name):
                        continue
                    if f"+{cu_short}{torch_mm}" not in name and f"+{cu_major}{torch_mm}" not in name:
                        continue
                    if "cxx11abi" in name and abi_str not in name:
                        continue
                    urls.append(asset["browser_download_url"])
        except Exception:
            pass
    return sorted(set(urls), reverse=True)[:6]


def _ensure_venv(venv_dir: Path, env: dict[str, str]) -> Path:
    vpy = venv_dir / "bin" / "python"
    if vpy.is_file() and (venv_dir / "bin" / "pip").is_file():
        return vpy
    if venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
    _run_step([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(venv_dir)], env=env)
    res = _run_step(
        [sys.executable, "-m", "pip", "--python", str(vpy), "install", "-q", "pip", "setuptools", "wheel"],
        env=env,
        check=False,
    )
    if res.returncode != 0:
        get_pip = SCRATCH_DIR / "get-pip.py"
        if not get_pip.is_file():
            urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", str(get_pip))
        _run_step([str(vpy), str(get_pip), "-q", "setuptools", "wheel"], env=env)
    return vpy


def _bootstrap_d3dv3_venv_and_reexec(spec_path: Path, spec: dict[str, Any]) -> None:
    s1 = spec.get("profile", {}).get("s1", {})
    repair_cfg = s1.get("repair") or {}
    need_pymeshlab = bool(repair_cfg.get("post_fusion_inplace_clean", False))
    splice_cfg = s1.get("coarse_splice") or {}
    need_splice = bool(splice_cfg.get("enabled", False) and spec.get("secondary_coarse_rel"))

    if _runtime_imports_ready(need_pymeshlab=need_pymeshlab):
        return

    t0 = time.monotonic()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    extra, gpu_info = _detect_cuda_build_env()
    env = _base_colab_env(extra)
    print(f"[d3dv3-setup] Bootstrapping Direct3D-V3 runtime for {gpu_info}", flush=True)

    if not (D3DV3_DIR / "pyproject.toml").is_file():
        print(f"[d3dv3-setup] Cloning {D3DV3_REPO_URL} -> {D3DV3_DIR}...", flush=True)
        _run_step(["git", "clone", "--depth", "1", D3DV3_REPO_URL, str(D3DV3_DIR)], env=env)

    env = _base_colab_env(extra)
    d3_py = _ensure_venv(D3DV3_VENV, env)
    chk_cmd = [
        str(d3_py),
        "-c",
        "import torch, timm, flash_attn, cudavoxelize, fillhole_cpp, pyfqmr, trimesh, direct3d",
    ]
    if _run_step(chk_cmd, env=env, check=False).returncode != 0:
        pkgs = [
            "numpy",
            "pillow",
            "scipy",
            "timm",
            "einops",
            "kornia",
            "pyfqmr",
            "pyyaml",
            "trimesh>=4",
            "transformers>=4.56,<5",
            "huggingface-hub>=0.34,<1.0",
            "safetensors",
            "scikit-image>=0.22,<0.27",
            "ninja",
            "packaging",
            "wheel",
            "setuptools",
        ]
        if need_pymeshlab:
            pkgs.append("pymeshlab")
        print("[d3dv3-setup] Installing Python runtime packages...", flush=True)
        _run_step([str(d3_py), "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed", *pkgs], env=env)
        if need_splice:
            _run_step([str(d3_py), "-m", "pip", "install", "-q", "openvdb"], env=env, check=False)

        print("[d3dv3-setup] Compiling third_party/voxelization...", flush=True)
        _run_step(
            [str(d3_py), "-m", "pip", "install", "--no-build-isolation", "--no-deps", "./third_party/voxelization"],
            cwd=D3DV3_DIR,
            env=env,
        )
        print("[d3dv3-setup] Compiling third_party/fill_holes...", flush=True)
        _run_step(
            [str(d3_py), "-m", "pip", "install", "--no-build-isolation", "--no-deps", "./third_party/fill_holes"],
            cwd=D3DV3_DIR,
            env=env,
        )
        if _run_step([str(d3_py), "-c", "import flash_attn"], env=env, check=False).returncode != 0:
            py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
            wurls = _find_matching_flash_attn_wheel(py_tag, gpu_info["torch"], str(gpu_info["torch_cuda"]), gpu_info["cxx11_abi"])
            installed = False
            for wurl in wurls:
                print(f"[d3dv3-setup] Installing prebuilt flash-attn wheel: {wurl}", flush=True)
                if _run_step([str(d3_py), "-m", "pip", "install", "--no-deps", wurl], env=env, check=False).returncode == 0:
                    installed = True
                    break
            if not installed:
                print("[d3dv3-setup] Compiling single-arch flash-attention...", flush=True)
                _run_step(
                    [
                        str(d3_py),
                        "-m",
                        "pip",
                        "install",
                        "--no-build-isolation",
                        "--no-deps",
                        "git+https://github.com/Dao-AILab/flash-attention.git@edb5c76ee329b18ed95d1f7ea9aa522a1331ab7d",
                    ],
                    env=env,
                )
        _run_step(
            [
                str(d3_py),
                "-c",
                "import torch, timm, flash_attn, cudavoxelize, fillhole_cpp, pyfqmr, direct3d; "
                "print('D3DV3_ENV_OK', torch.__version__, flash_attn.__version__)",
            ],
            env=env,
        )

    setup_elapsed = round(time.monotonic() - t0, 3)
    (LOGS_DIR / "d3dv3_setup_meta.json").write_text(
        json.dumps({"setup_elapsed_s": setup_elapsed, "gpu_info": gpu_info}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[d3dv3-setup] Ready in {setup_elapsed:.1f}s; switching into {d3_py}...", flush=True)
    os.execve(
        str(d3_py),
        [str(d3_py), "-u", str(Path(__file__).resolve()), "--spec", str(spec_path), "--skip-env-setup"],
        env,
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(1024 * 1024)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _package_outputs(outputs_dir: Path) -> dict[str, Any]:
    bundle = outputs_dir / "outputs_bundle.tar.gz"
    if bundle.exists():
        bundle.unlink()
    with tarfile.open(bundle, "w:gz") as tf:
        for p in sorted(outputs_dir.iterdir()):
            if p.name.startswith("outputs_bundle") or p.name == "output_manifest.json":
                continue
            tf.add(p, arcname=p.name, recursive=True)

    total_size = bundle.stat().st_size
    bundle_sha = _sha256_file(bundle)
    chunks: list[dict[str, Any]] = []
    if total_size <= MAX_CHUNK_BYTES:
        chunks.append(
            {
                "name": bundle.name,
                "size": total_size,
                "bytes": total_size,
                "sha256": bundle_sha,
            }
        )
    else:
        idx = 0
        with bundle.open("rb") as src:
            while True:
                data = src.read(MAX_CHUNK_BYTES)
                if not data:
                    break
                part_name = f"outputs_bundle.tar.gz.part{idx:03d}"
                part_path = outputs_dir / part_name
                part_path.write_bytes(data)
                chunks.append(
                    {
                        "name": part_name,
                        "size": len(data),
                        "bytes": len(data),
                        "sha256": _sha256_file(part_path),
                    }
                )
                idx += 1
        bundle.unlink()

    manifest = {
        "archive_name": bundle.name,
        "sha256": bundle_sha,
        "bundle_sha256": bundle_sha,
        "total_size": total_size,
        "max_chunk_bytes": MAX_CHUNK_BYTES,
        "chunks": chunks,
    }
    (outputs_dir / "output_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _compare_meshes_geom(mesh_post_path: Path, mesh_pre_path: Path) -> dict[str, Any]:
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree

    m_post = trimesh.load(str(mesh_post_path), force="mesh")
    m_pre = trimesh.load(str(mesh_pre_path), force="mesh")
    vp = np.asarray(m_post.vertices, dtype=np.float32)
    vq = np.asarray(m_pre.vertices, dtype=np.float32)

    d_pq, _ = cKDTree(vq).query(vp, k=1)
    d_qp, _ = cKDTree(vp).query(vq, k=1)
    chamfer = float(0.5 * (d_pq.mean() + d_qp.mean()))
    hausdorff_approx = float(max(d_pq.max(), d_qp.max()))

    fn_p = np.asarray(m_post.face_normals, dtype=np.float32)
    fn_q = np.asarray(m_pre.face_normals, dtype=np.float32)
    fa_p = np.asarray(m_post.area_faces, dtype=np.float32)
    fa_q = np.asarray(m_pre.area_faces, dtype=np.float32)
    side_p = (np.abs(fn_p[:, 0]) > 0.70) & (np.abs(fn_p[:, 1]) < 0.20)
    side_q = (np.abs(fn_q[:, 0]) > 0.70) & (np.abs(fn_q[:, 1]) < 0.20)

    return {
        "post_vertices": int(len(vp)),
        "post_faces": int(len(m_post.faces)),
        "pre_vertices": int(len(vq)),
        "pre_faces": int(len(m_pre.faces)),
        "symmetric_chamfer": round(chamfer, 6),
        "vertex_max_dist": round(hausdorff_approx, 6),
        "post_vertical_side_area": round(float(fa_p[side_p].sum()), 5),
        "pre_vertical_side_area": round(float(fa_q[side_q].sum()), 5),
    }


def run_worker(spec: dict[str, Any]) -> dict[str, Any]:
    if D3DV3_DIR.exists() and str(D3DV3_DIR) not in sys.path:
        sys.path.insert(0, str(D3DV3_DIR))

    import trimesh

    from quadvox.contracts import CoarseSpliceSpec, ROISovereigntyBox, ZoneSmoothingSpec
    from quadvox.s1_geometry.coarse_splice import splice_coarse_glbs
    from quadvox.s1_geometry.d3dv3_fusion import (
        D3DV3FusionConfig,
        compare_crevice_keep_preflight,
        run_d3dv3_4view_fusion,
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(SCRATCH_DIR)
    os.environ["TEMP"] = str(SCRATCH_DIR)
    os.environ["TMP"] = str(SCRATCH_DIR)

    profile = spec["profile"]
    s1 = profile.get("s1", {})
    fusion_raw = s1.get("d3dv3_fusion", {})
    target_height = float(s1.get("target_height", 1.0))

    coarse_path = WORK_DIR / spec["coarse_rel"]
    splice_cfg = CoarseSpliceSpec.from_dict(s1.get("coarse_splice") or {})
    if splice_cfg.enabled and spec.get("secondary_coarse_rel"):
        sec_path = WORK_DIR / spec["secondary_coarse_rel"]
        spliced_glb = OUTPUTS_DIR / "coarse_spliced.glb"
        splice_coarse_glbs(coarse_path, sec_path, spliced_glb, splice_cfg)
        coarse_path = spliced_glb

    roi_boxes: list[ROISovereigntyBox] = []
    if fusion_raw.get("front_roi_box"):
        roi_boxes.append(ROISovereigntyBox.from_dict(fusion_raw["front_roi_box"]))
    for rb in fusion_raw.get("roi_boxes", []):
        roi_boxes.append(ROISovereigntyBox.from_dict(rb))
    smooth_zones = [ZoneSmoothingSpec.from_dict(z) for z in fusion_raw.get("smoothing_zones", [])]

    coarse_mesh = trimesh.load(str(coarse_path), force="mesh")
    c_span_y = float(coarse_mesh.bounds[1][1] - coarse_mesh.bounds[0][1])
    crevice_cmp: dict[str, Any] = {
        "surface_preflight": compare_crevice_keep_preflight(
            coarse_mesh.vertices,
            coarse_mesh.faces,
            domain_height_full=c_span_y,
            roi_boxes=roi_boxes,
            smoothing_zones=smooth_zones,
        )
    }
    print(f"[crevice-keep] Surface comparison: {json.dumps(crevice_cmp['surface_preflight'])}", flush=True)

    raw_g = fusion_raw.get("guidance_scale")
    cfg = D3DV3FusionConfig(
        stage_resolutions=tuple(int(x) for x in fusion_raw.get("stage_resolutions", (512, 1024, 1536))),
        stage512_seeds=tuple(int(s) for s in fusion_raw.get("stage512_seeds", (2026, 2027, 2028))),
        refine_seed=int(fusion_raw.get("refine_seed", 2026)),
        guidance_scale=float(raw_g) if raw_g is not None else None,
        sharpen_t_start=float(fusion_raw.get("sharpen_t_start", 0.53)),
        cfg_decay_t_start=float(fusion_raw.get("cfg_decay_t_start", 0.50)),
        cfg_late_min=float(fusion_raw.get("cfg_late_min", 2.2)),
        max_tokens=int(fusion_raw.get("max_tokens", 200_000)),
        target_faces=(
            int(fusion_raw["target_faces"])
            if fusion_raw.get("target_faces", 2_000_000) is not None
            else None
        ),
        roi_boxes=roi_boxes,
        smoothing_zones=smooth_zones,
        crevice_keep_mode="fixed",
    )
    view_paths = {vn: WORK_DIR / rel for vn, rel in spec["view_rels"].items()}
    out_glb = OUTPUTS_DIR / "fused_1536.glb"
    res = run_d3dv3_4view_fusion(
        coarse_mesh_path=coarse_path,
        view_paths=view_paths,
        out_glb_path=out_glb,
        config=cfg,
        local_files_only=False,
        target_height=target_height,
    )

    if bool(spec.get("compare_crevice_keep", False)):
        cfg_pre = D3DV3FusionConfig(
            stage_resolutions=cfg.stage_resolutions,
            guidance_scale=cfg.guidance_scale,
            sharpen_t_start=cfg.sharpen_t_start,
            cfg_decay_t_start=cfg.cfg_decay_t_start,
            cfg_late_min=cfg.cfg_late_min,
            roi_boxes=roi_boxes,
            smoothing_zones=smooth_zones,
            crevice_keep_mode="legacy_unfixed",
        )
        pre_glb = OUTPUTS_DIR / "fused_1536_legacy_crevice.glb"
        res_pre = run_d3dv3_4view_fusion(
            coarse_mesh_path=coarse_path,
            view_paths=view_paths,
            out_glb_path=pre_glb,
            config=cfg_pre,
            local_files_only=False,
            target_height=target_height,
        )
        crevice_cmp["legacy_unfixed_run"] = res_pre
        crevice_cmp["mesh_comparison_1536"] = _compare_meshes_geom(out_glb, pre_glb)
        print(f"[crevice-keep] 1536 mesh diff: {json.dumps(crevice_cmp['mesh_comparison_1536'])}", flush=True)

    setup_meta_p = LOGS_DIR / "d3dv3_setup_meta.json"
    if setup_meta_p.is_file():
        res["setup_meta"] = json.loads(setup_meta_p.read_text(encoding="utf-8"))
    res["crevice_keep_comparison"] = crevice_cmp

    repair_cfg = s1.get("repair") or {}
    if bool(repair_cfg.get("post_fusion_inplace_clean", False)) and out_glb.is_file():
        from quadvox.s1_geometry.repair import inplace_clean_glb, normalize_glb

        norm_tmp = OUTPUTS_DIR / "._fused_1536_norm.glb"
        clean_glb = OUTPUTS_DIR / "fused_1536_clean.glb"
        normalize_glb(out_glb, norm_tmp, target_height=target_height)
        inplace_clean_glb(
            norm_tmp,
            clean_glb,
            weld_epsilon_rel=float(repair_cfg.get("weld_eps_rel", 0.0)),
            min_component_area_rel=float(repair_cfg.get("min_component_area_rel", 0.02)),
        )
        norm_tmp.unlink(missing_ok=True)
        res["cleaned_glb"] = str(clean_glb)

    (OUTPUTS_DIR / "d3dv3_report.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    manifest = _package_outputs(OUTPUTS_DIR)
    print(
        f"[d3dv3-worker] Packaged {manifest['archive_name']} bytes={manifest['total_size']} "
        f"chunks={len(manifest['chunks'])} sha256={manifest['bundle_sha256'][:12]}...",
        flush=True,
    )
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--skip-env-setup", action="store_true")
    args = ap.parse_args()
    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not args.skip_env_setup:
        _bootstrap_d3dv3_venv_and_reexec(spec_path, spec)
    run_worker(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
