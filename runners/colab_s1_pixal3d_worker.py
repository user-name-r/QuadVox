"""Remote Colab execution entrypoint for S1 Pixal3D benchmarking and inference.

Runs inside a Google Colab GPU runtime using `/content/cache` exclusively (never
system temp dirs or Google Drive mounts). Supports `--mode benchmark` and `--mode run`.

Verified upstream URLs:
- Pixal3D upstream: https://github.com/TencentARC/Pixal3D
- Pixal3D MV inference: https://github.com/TencentARC/Pixal3D/blob/master/inference_mv.py
- ComfyUI Pixal3D docs: https://github.com/Comfy-Org/docs/blob/main/tutorials/3d/pixal3d.mdx
- ComfyUI nodes_trellis2.py: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_trellis2.py
- ComfyUI nodes_mesh_postprocess.py: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_mesh_postprocess.py
- pixal3d.cpp: https://github.com/raven38/pixal3d.cpp
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
import threading
import time
from pathlib import Path
from typing import Any, Sequence


CACHE_ROOT = Path("/content/cache")
WORK_DIR = CACHE_ROOT / "work"
LOGS_DIR = CACHE_ROOT / "logs"
MODELS_DIR = CACHE_ROOT / "models"
OUTPUTS_DIR = CACHE_ROOT / "outputs"
TRANSFER_DIR = CACHE_ROOT / "transfer"
COMFY_DIR = CACHE_ROOT / "ComfyUI"
COMFY_MODELS_DIR = MODELS_DIR / "comfyui"
CPP_ROOT = CACHE_ROOT / "pixal3d_cpp"
MAX_CHUNK_BYTES = 15 * 1024 * 1024  # Strictly below Colab CLI's 20 MB limit.


class HeartbeatState:
    """Thread-safe stage tracker emitting flushed heartbeats every ~20s."""

    def __init__(self, interval_s: float = 20.0) -> None:
        self.interval_s = float(interval_s)
        self.stage = "init"
        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def set_stage(self, stage: str) -> None:
        self.stage = str(stage)
        print(
            f"[colab-remote] stage={self.stage} elapsed={time.monotonic() - self._t0:.1f}s",
            flush=True,
        )

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(timeout=self.interval_s):
            print(
                f"[colab-remote-heartbeat] stage={self.stage} "
                f"elapsed={time.monotonic() - self._t0:.1f}s",
                flush=True,
            )


def _run_logged(cmd: Sequence[str], log_file: Path, label: str) -> None:
    """Run a setup command redirecting verbose output to `log_file`."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TQDM_DISABLE"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PIP_PROGRESS_BAR"] = "off"

    with log_file.open("a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n=== [{label}] {' '.join(cmd)} ===\n")
        fh.flush()
        proc = subprocess.run(
            list(cmd),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
    if proc.returncode != 0:
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-50:])
        raise RuntimeError(f"Step {label!r} failed (rc={proc.returncode}); tail:\n{tail}")


def _query_gpu_summary() -> dict[str, Any]:
    """Query active CUDA GPU name, total VRAM, and NVIDIA driver version."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5.0,
        )
        parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
        return {
            "gpu_name": parts[0] if len(parts) > 0 else "unknown",
            "vram_total_mib": float(parts[1]) if len(parts) > 1 else 0.0,
            "driver_version": parts[2] if len(parts) > 2 else "unknown",
        }
    except Exception as exc:
        raise RuntimeError(
            f"No CUDA GPU detected on Colab runtime (refusing CPU fallback): {exc}"
        ) from exc


def _ensure_python_deps() -> None:
    """Install lightweight Python dependencies needed by S1 backend without reinstalling PyTorch."""
    pip_log = LOGS_DIR / "pip_install.log"
    pkgs = ["numpy", "scipy", "pillow", "pyyaml", "trimesh", "requests", "huggingface_hub"]
    _run_logged(
        [sys.executable, "-m", "pip", "install", "-q", *pkgs],
        pip_log,
        "pip_core_deps",
    )


def _ensure_comfyui() -> None:
    """Clone ComfyUI into `/content/cache/ComfyUI` and install non-torch requirements."""
    from quadvox.s1_geometry.generators.path_a_comfy import COMFYUI_REPO_URL

    git_log = LOGS_DIR / "comfyui_setup.log"
    if not (COMFY_DIR / "nodes.py").is_file():
        _run_logged(
            ["git", "clone", "--depth", "1", COMFYUI_REPO_URL, str(COMFY_DIR)],
            git_log,
            "clone_comfyui",
        )

    marker = COMFY_DIR / ".s1_deps_installed"
    if not marker.is_file():
        req_src = COMFY_DIR / "requirements.txt"
        req_filtered = CACHE_ROOT / "comfy_filtered_requirements.txt"
        skip_prefixes = (
            "torch",
            "torchvision",
            "torchaudio",
            "comfyui-frontend-package",
            "comfyui-workflow-templates",
            "comfyui-embedded-docs",
        )
        kept_lines = []
        for raw in req_src.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            pkg_name = line.split("=", 1)[0].split(">", 1)[0].split("<", 1)[0].split("~", 1)[0].strip().lower()
            if pkg_name in skip_prefixes:
                continue
            kept_lines.append(line)
        req_filtered.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        _run_logged(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_filtered)],
            LOGS_DIR / "pip_install.log",
            "pip_comfyui_deps",
        )
        marker.write_text("ok\n", encoding="utf-8")


def _hf_download_file(
    repo_id: str,
    filename: str,
    dest_dir: Path,
    *,
    fallback_repo: str | None = None,
) -> Path:
    """Download one file from HuggingFace into `dest_dir/filename` with optional mirror fallback."""
    target = dest_dir / filename
    if target.is_file() and target.stat().st_size > 0:
        return target

    from huggingface_hub import hf_hub_download

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    dest_dir.mkdir(parents=True, exist_ok=True)
    repos = [repo_id] + ([fallback_repo] if fallback_repo else [])
    last_exc: Exception | None = None
    for repo in repos:
        try:
            print(f"[download] {repo}/{filename} -> {target}", flush=True)
            out_path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=str(dest_dir),
            )
            resolved = Path(out_path)
            if resolved.is_file() and resolved.stat().st_size > 0:
                return resolved
        except Exception as exc:
            last_exc = exc
            print(f"[download] {repo}/{filename} failed: {exc}", flush=True)
    raise RuntimeError(
        f"Failed to download {filename} from {repos}: {last_exc}"
    )


def _download_birefnet() -> Path:
    from quadvox.s1_geometry.generators.path_a_comfy import (
        HF_COMFY_BIREFNET_REPO,
        PATH_A_BIREFNET_FILE,
    )

    return _hf_download_file(HF_COMFY_BIREFNET_REPO, PATH_A_BIREFNET_FILE, COMFY_MODELS_DIR)


def _download_path_a_weights(input_mode: str, include_moge: bool = False) -> None:
    from quadvox.s1_geometry.generators.path_a_comfy import (
        HF_COMFY_MOGE_REPO,
        HF_COMFY_PIXAL3D_REPO,
        HF_FALLBACK_PIXAL3D_REPO,
        PATH_A_MOGE_FILE,
        required_model_relpaths,
    )

    for rel in required_model_relpaths(mode=input_mode, include_moge=False):
        _hf_download_file(
            HF_COMFY_PIXAL3D_REPO,
            rel,
            COMFY_MODELS_DIR,
            fallback_repo=HF_FALLBACK_PIXAL3D_REPO,
        )
    if input_mode == "sv" and include_moge:
        _hf_download_file(HF_COMFY_MOGE_REPO, PATH_A_MOGE_FILE, COMFY_MODELS_DIR)


def _download_path_b_weights(input_mode: str) -> Path:
    from quadvox.s1_geometry.generators.path_b_cpp import (
        HF_RAVEN_MV_REPO,
        HF_RAVEN_SV_REPO,
        SHARED_GGUF_FILES,
        default_cpp_models_dir,
        required_gguf_files,
    )

    repo = HF_RAVEN_SV_REPO if input_mode == "sv" else HF_RAVEN_MV_REPO
    models_dir = default_cpp_models_dir(input_mode, cache_dir=CACHE_ROOT)
    other_dir = default_cpp_models_dir(
        "mv" if input_mode == "sv" else "sv", cache_dir=CACHE_ROOT
    )

    for fname in required_gguf_files(input_mode):
        target = models_dir / fname
        if target.is_file() and target.stat().st_size > 0:
            continue
        if fname in SHARED_GGUF_FILES:
            sibling = other_dir / fname
            if sibling.is_file() and sibling.stat().st_size > 0:
                models_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sibling, target)
                continue
        _hf_download_file(repo, fname, models_dir)
    return models_dir


def _needs_matting(path: Path) -> bool:
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        im.load()
        if im.mode not in ("RGBA", "LA", "PA"):
            return True
        alpha = np.asarray(im.convert("RGBA").getchannel(3), dtype=np.uint8)
        return bool(np.all(alpha == 255))


def _prepare_shared_views(spec: dict[str, Any], hb: HeartbeatState) -> Path:
    """Run BiRefNet matting (if needed) + `align_multiview_rgba` + `build_orbit_transforms`."""
    from PIL import Image
    from quadvox.s1_geometry.generators.path_a_comfy import run_birefnet_matte
    from quadvox.s0_views.preprocess import discover_input_views, prepare_views_dir

    hb.set_stage("preprocess_birefnet")
    raw_dir = Path(spec["raw_views_dir"])
    input_mode = str(spec["input_mode"]).lower()
    raw_paths = discover_input_views(raw_dir, mode=input_mode)

    matted_dir = CACHE_ROOT / "work" / "matted_views"
    matted_dir.mkdir(parents=True, exist_ok=True)

    need_matte_paths = [p for p in raw_paths if _needs_matting(p)]
    matted_map: dict[Path, Path] = {}
    if need_matte_paths:
        print(
            f"[preprocess] Running BiRefNet on {len(need_matte_paths)}/{len(raw_paths)} view(s)...",
            flush=True,
        )
        _download_birefnet()
        matted_outputs = run_birefnet_matte(
            need_matte_paths,
            matted_dir,
            comfy_dir=COMFY_DIR,
            models_dir=COMFY_MODELS_DIR,
            isolated=True,
            log_path=LOGS_DIR / "birefnet.log",
        )
        for src_p, out_p in zip(need_matte_paths, matted_outputs):
            matted_map[src_p] = out_p

    pre_input_paths: list[Path] = []
    for p in raw_paths:
        pre_input_paths.append(matted_map.get(p, p))

    views_prepped = OUTPUTS_DIR / "views_prepped"
    prep_info = prepare_views_dir(
        input_path=pre_input_paths,
        out_dir=views_prepped,
        mode=input_mode,
        size=1024,
        fov_deg=float(spec.get("fov_deg", 20.0)),
        pad_factor=float(spec.get("pad_factor", 1.1)),
        mesh_scale=1.0,
    )
    print(
        f"[preprocess] Prepared {len(prep_info['file_paths'])} aligned RGBA view(s) -> {views_prepped}",
        flush=True,
    )
    return views_prepped


def _run_cellocut_worker(
    in_glb: Path,
    out_glb: Path,
    meta_out: Path,
    *,
    resolution: int,
    fill: float,
    band: float,
    project_back: float,
    qef: bool = False,
    smooth_iters: int = 0,
    min_component_area_rel: float = 0.005,
    weld_eps_rel: float = 1e-10,
) -> None:
    """In-process CelloCut worker executed inside a fresh subprocess."""
    import inspect
    import torch
    from quadvox.s1_geometry.repair.cellocut import cellocut_remesh_glb

    if str(WORK_DIR) not in sys.path:
        sys.path.insert(0, str(WORK_DIR))

    sig_params = inspect.signature(cellocut_remesh_glb).parameters
    accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
    kwargs: dict[str, Any] = {
        "resolution": int(resolution),
        "fill": float(fill),
        "band": float(band),
        "project_back": float(project_back),
        "comfyui_dir": str(COMFY_DIR),
        "require_cuda": True,
    }
    if accepts_varkw or "qef" in sig_params:
        kwargs["qef"] = bool(qef)
    if accepts_varkw or "smooth_iters" in sig_params:
        kwargs["smooth_iters"] = int(smooth_iters)
    if accepts_varkw or "min_component_area_rel" in sig_params:
        kwargs["min_component_area_rel"] = float(min_component_area_rel)
    if accepts_varkw or "weld_eps_rel" in sig_params:
        kwargs["weld_eps_rel"] = float(weld_eps_rel)

    with torch.inference_mode():
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        raw_m, clean_m, info = cellocut_remesh_glb(in_glb, out_glb, **kwargs)
        elapsed = time.monotonic() - t0
        peak = float(torch.cuda.max_memory_allocated()) / (1024.0 ** 3)
        payload = {
            "elapsed_s": elapsed,
            "peak_vram_gb": peak,
            "info": info,
            "raw_metrics": raw_m.to_dict(),
            "clean_metrics": clean_m.to_dict(),
        }
        meta_out.parent.mkdir(parents=True, exist_ok=True)
        meta_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_cellocut_subprocess(
    in_glb: Path,
    out_glb: Path,
    *,
    resolution: int = 1024,
    fill: float = 20.0,
    band: float = 1.0,
    project_back: float = 0.5,
    qef: bool = False,
    smooth_iters: int = 0,
    min_component_area_rel: float = 0.005,
    weld_eps_rel: float = 1e-10,
    log_path: Path,
) -> dict[str, Any]:
    """Invoke `quadvox.s1_geometry.repair.cellocut_remesh_glb` in an isolated subprocess."""
    from quadvox.s1_geometry.generators.base import monitor_subprocess_with_vram

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    meta_out = LOGS_DIR / f"{out_glb.stem}.cellocut_meta.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cellocut-worker",
        "--in-glb",
        str(in_glb),
        "--out-glb",
        str(out_glb),
        "--meta-out",
        str(meta_out),
        "--remesh-resolution",
        str(int(resolution)),
        "--remesh-fill",
        str(float(fill)),
        "--remesh-band",
        str(float(band)),
        "--project-back",
        str(float(project_back)),
        "--qef" if qef else "--no-qef",
        "--smooth-iters",
        str(int(smooth_iters)),
        "--min-component-area-rel",
        str(float(min_component_area_rel)),
        "--weld-eps-rel",
        str(float(weld_eps_rel)),
    ]
    rc, wall_s, smi_gb, tail = monitor_subprocess_with_vram(
        cmd,
        log_path=log_path,
        heartbeat_label=f"cellocut_{in_glb.stem}_pb{project_back}",
    )
    if rc != 0:
        raise RuntimeError(f"CelloCut remesh failed (rc={rc}) for {in_glb.name}:\n{tail}")
    data = json.loads(meta_out.read_text(encoding="utf-8"))
    data["peak_vram_gb"] = max(float(data.get("peak_vram_gb", 0.0)), smi_gb)
    data["wall_elapsed_s"] = wall_s
    return data


def _normalize_for_backend(
    in_glb: Path,
    out_glb: Path,
    coordinate_frame: str,
    target_height: float,
) -> None:
    from quadvox.s1_geometry.repair.normalize import normalize_glb

    yaw_deg = 180.0 if coordinate_frame == "y_up_neg_z_front" else 0.0
    normalize_glb(
        in_glb,
        out_glb,
        target_height=float(target_height),
        yaw_deg=yaw_deg,
        up_axis="+y",
        front_axis="+z",
    )


def _run_benchmark_mode(
    spec: dict[str, Any],
    views_prepped: Path,
    gpu_info: dict[str, Any],
    hb: HeartbeatState,
) -> dict[str, Any]:
    from quadvox.s1_geometry.generators.path_a_comfy import PathAComfyGenerator
    from quadvox.s1_geometry.generators.path_b_cpp import PathBCppGenerator, ensure_trellis_cli
    from quadvox.s1_geometry.repair.inplace_clean import inplace_clean_glb
    from quadvox.s1_geometry.repair.metrics import compute_mesh_metrics

    input_mode = str(spec["input_mode"]).lower()
    resolution = int(spec.get("resolution", 1536))
    seed = int(spec.get("seed", 42))
    fov_deg = float(spec.get("fov_deg", 20.0))
    target_height = float(spec.get("target_height", 1.0))

    hb.set_stage("setup_path_a_weights")
    _download_path_a_weights(input_mode=input_mode)

    hb.set_stage("setup_path_b_cli_and_weights")
    ensure_trellis_cli(CPP_ROOT, log_path=LOGS_DIR / "path_b_build.log")
    cpp_models_dir = _download_path_b_weights(input_mode=input_mode)

    hb.set_stage("generate_path_a")
    gen_a = PathAComfyGenerator(comfy_dir=COMFY_DIR, models_dir=COMFY_MODELS_DIR)
    res_a = gen_a.generate(
        views_dir=views_prepped,
        out_glb=OUTPUTS_DIR / "path_a_raw.glb",
        mode=input_mode,
        resolution=resolution,
        seed=seed,
        fov_deg=fov_deg,
        log_path=LOGS_DIR / "path_a_generate.log",
    )

    hb.set_stage("generate_path_b")
    gen_b = PathBCppGenerator(
        cpp_root=CPP_ROOT,
        models_dir=cpp_models_dir if input_mode == "mv" else None,
        models_sv_dir=cpp_models_dir if input_mode == "sv" else None,
    )
    res_b = gen_b.generate(
        views_dir=views_prepped,
        out_glb=OUTPUTS_DIR / "path_b_raw.glb",
        mode=input_mode,
        resolution=resolution,
        seed=seed,
        fov_deg=fov_deg,
        log_path=LOGS_DIR / "path_b_generate.log",
    )

    # Compute raw metrics and run both official repair routes on both backends.
    hb.set_stage("repair_inplace_and_cellocut")
    variants: dict[str, Any] = {}
    for label, gen_res in (("path_a", res_a), ("path_b", res_b)):
        raw_path = Path(gen_res.raw_glb)
        raw_metrics = compute_mesh_metrics(raw_path)

        # Route 1: inplace_clean (O-Voxel remesh=False / WeldVertices + FillHoles)
        inplace_path = OUTPUTS_DIR / f"{label}_inplace.glb"
        print(f"[benchmark] Running inplace_clean on {raw_path.name}...", flush=True)
        t_rep0 = time.monotonic()
        _, inplace_raw_m, inplace_info = inplace_clean_glb(raw_path, inplace_path)
        _normalize_for_backend(
            inplace_path,
            inplace_path,
            gen_res.coordinate_frame,
            target_height,
        )
        inplace_elapsed = time.monotonic() - t_rep0
        inplace_clean_m = compute_mesh_metrics(inplace_path)
        print(
            f"[benchmark] {label}_inplace done in {inplace_elapsed:.1f}s "
            f"(faces={inplace_clean_m.face_count}, pinholes={inplace_clean_m.pinhole_loops})",
            flush=True,
        )

        # Route 2a: ComfyUI CelloCut solid remesh (project_back=0.5)
        bench_smooth_iters = int(
            spec.get("smooth_iters", spec.get("cellocut_smooth_iters", 0))
        )
        bench_min_area = float(spec.get("min_component_area_rel", 0.005))
        bench_weld_eps = float(spec.get("weld_eps_rel", 1e-10))
        bench_qef = bool(spec.get("cellocut_qef", spec.get("qef", False)))
        cello_05_path = OUTPUTS_DIR / f"{label}_cellocut_pb05.glb"
        cello_05_data = _run_cellocut_subprocess(
            raw_path,
            cello_05_path,
            resolution=1024,
            fill=20.0,
            band=1.0,
            project_back=0.5,
            qef=bench_qef,
            smooth_iters=bench_smooth_iters,
            min_component_area_rel=bench_min_area,
            weld_eps_rel=bench_weld_eps,
            log_path=LOGS_DIR / f"{label}_cellocut_pb05.log",
        )
        _normalize_for_backend(
            cello_05_path,
            cello_05_path,
            gen_res.coordinate_frame,
            target_height,
        )
        cello_05_m = compute_mesh_metrics(cello_05_path)

        # Route 2b: ComfyUI CelloCut solid remesh (project_back=0.0)
        cello_00_path = OUTPUTS_DIR / f"{label}_cellocut_pb00.glb"
        cello_00_data = _run_cellocut_subprocess(
            raw_path,
            cello_00_path,
            resolution=1024,
            fill=20.0,
            band=1.0,
            project_back=0.0,
            qef=bench_qef,
            smooth_iters=bench_smooth_iters,
            min_component_area_rel=bench_min_area,
            weld_eps_rel=bench_weld_eps,
            log_path=LOGS_DIR / f"{label}_cellocut_pb00.log",
        )
        _normalize_for_backend(
            cello_00_path,
            cello_00_path,
            gen_res.coordinate_frame,
            target_height,
        )
        cello_00_m = compute_mesh_metrics(cello_00_path)

        variants[label] = {
            "generation": gen_res.to_dict(),
            "raw_metrics": raw_metrics.to_dict(),
            "inplace_clean": {
                "glb": str(inplace_path.relative_to(OUTPUTS_DIR)),
                "elapsed_s": inplace_elapsed,
                "info": inplace_info,
                "pre_norm_metrics": inplace_raw_m.to_dict(),
                "clean_metrics": inplace_clean_m.to_dict(),
            },
            "cellocut_pb05": {
                "glb": str(cello_05_path.relative_to(OUTPUTS_DIR)),
                "elapsed_s": float(cello_05_data["elapsed_s"]),
                "peak_vram_gb": float(cello_05_data["peak_vram_gb"]),
                "info": cello_05_data["info"],
                "clean_metrics": cello_05_m.to_dict(),
            },
            "cellocut_pb00": {
                "glb": str(cello_00_path.relative_to(OUTPUTS_DIR)),
                "elapsed_s": float(cello_00_data["elapsed_s"]),
                "peak_vram_gb": float(cello_00_data["peak_vram_gb"]),
                "info": cello_00_data["info"],
                "clean_metrics": cello_00_m.to_dict(),
            },
        }

    report = {
        "mode": "benchmark",
        "input_mode": input_mode,
        "resolution": resolution,
        "seed": seed,
        "fov_deg": fov_deg,
        "target_height": target_height,
        "gpu_info": gpu_info,
        "variants": variants,
    }
    (OUTPUTS_DIR / "benchmark_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def _run_production_mode(
    spec: dict[str, Any],
    views_prepped: Path,
    gpu_info: dict[str, Any],
    hb: HeartbeatState,
) -> dict[str, Any]:
    from quadvox.contracts import JobConfig, RunMeta
    from quadvox.s1_geometry.generators import get_generator, normalize_backend_name
    from quadvox.s1_geometry.generators.path_b_cpp import ensure_trellis_cli
    from quadvox.s1_geometry.repair.inplace_clean import inplace_clean_glb
    from quadvox.s1_geometry.repair.metrics import compute_mesh_metrics
    from quadvox.s1_geometry.selector import candidate_sort_key, select_best_candidate

    t_total0 = time.monotonic()
    backend = normalize_backend_name(str(spec.get("backend", "path_a")))
    repair_method = str(spec.get("repair_method", "cellocut")).lower()
    input_mode = str(spec.get("input_mode", "mv")).lower()
    resolution = int(spec.get("resolution", 1536))
    fov_deg = float(spec.get("fov_deg", 20.0))
    target_height = float(spec.get("target_height", 1.0))
    remesh_resolution = int(spec.get("remesh_resolution", 1024))
    remesh_fill = float(spec.get("remesh_fill", 20.0))
    remesh_band = float(spec.get("remesh_band", 1.0))
    project_back = float(spec.get("project_back", 0.5))
    remesh_qef = bool(spec.get("cellocut_qef", spec.get("qef", False)))
    smooth_iters = int(spec.get("smooth_iters", spec.get("cellocut_smooth_iters", 0)))
    min_component_area_rel = float(spec.get("min_component_area_rel", 0.005))

    seed_list = spec.get("seed_list")
    if seed_list:
        seeds = [int(s) for s in seed_list]
    else:
        base_seed = int(spec.get("seed", 42))
        count = max(1, int(spec.get("seeds", 1)))
        seeds = [base_seed + i for i in range(count)]

    hb.set_stage(f"setup_{backend}")
    if backend == "path_a":
        _download_path_a_weights(input_mode=input_mode)
        generator = get_generator("path_a", comfy_dir=COMFY_DIR, models_dir=COMFY_MODELS_DIR)
    else:
        ensure_trellis_cli(CPP_ROOT, log_path=LOGS_DIR / "path_b_build.log")
        cpp_models_dir = _download_path_b_weights(input_mode=input_mode)
        generator = get_generator(
            "path_b",
            cpp_root=CPP_ROOT,
            models_dir=cpp_models_dir if input_mode == "mv" else None,
            models_sv_dir=cpp_models_dir if input_mode == "sv" else None,
        )

    hb.set_stage(f"generate_{backend}_{len(seeds)}seeds")
    # Keep per-seed raw GLBs outside OUTPUTS_DIR so discarded candidates are NOT packaged/downloaded.
    cand_dir = WORK_DIR / "candidates"
    if cand_dir.exists():
        shutil.rmtree(cand_dir)
    gen_results = generator.generate_seeds(
        views_dir=views_prepped,
        out_dir=cand_dir,
        seeds=seeds,
        mode=input_mode,
        resolution=resolution,
        fov_deg=fov_deg,
    )

    candidates = []
    for gr in gen_results:
        metrics = compute_mesh_metrics(gr.raw_glb)
        candidates.append(gr.to_seed_candidate(metrics))

    winner_cand, ranked_cands = select_best_candidate(candidates)
    winner_res = next(gr for gr in gen_results if gr.seed == winner_cand.seed)

    raw_glb = OUTPUTS_DIR / "raw.glb"
    shutil.copy2(winner_cand.raw_glb, raw_glb)
    raw_metrics = winner_cand.metrics

    # Preserve candidate logs in LOGS_DIR while removing discarded ~177MB GLBs.
    cand_logs_dir = LOGS_DIR / "candidates"
    cand_logs_dir.mkdir(parents=True, exist_ok=True)
    for item in cand_dir.iterdir():
        if item.suffix in (".log", ".json"):
            shutil.copy2(item, cand_logs_dir / item.name)
    for c in ranked_cands:
        c.raw_glb = "raw.glb" if c.seed == winner_cand.seed else f"discarded:seed_{c.seed}_raw.glb"
    shutil.rmtree(cand_dir, ignore_errors=True)

    wm = winner_cand.metrics
    winner_key = candidate_sort_key(winner_cand)
    selection_reason = (
        f"winner=seed_{winner_cand.seed} among {len(ranked_cands)} candidates "
        f"by key={winner_key} "
        f"(large_openings={wm.large_opening_loops}, pinholes={wm.pinhole_loops}, "
        f"nm_verts={wm.non_manifold_vertices}, open_edges={wm.open_boundary_edges}, "
        f"nm_edges={wm.non_manifold_edges}, faces={wm.face_count})"
    )
    print(f"[selector] {selection_reason}", flush=True)

    hb.set_stage(f"repair_{repair_method}")
    t_rep0 = time.monotonic()
    clean_glb = OUTPUTS_DIR / "clean_white.glb"
    repair_vram_gb = 0.0
    if repair_method in ("inplace", "inplace_clean"):
        inplace_clean_glb(raw_glb, clean_glb)
        _normalize_for_backend(
            clean_glb,
            clean_glb,
            winner_res.coordinate_frame,
            target_height,
        )
    elif repair_method in ("cellocut", "cellocut_remesh"):
        cello_data = _run_cellocut_subprocess(
            raw_glb,
            clean_glb,
            resolution=remesh_resolution,
            fill=remesh_fill,
            band=remesh_band,
            project_back=project_back,
            qef=remesh_qef,
            smooth_iters=smooth_iters,
            min_component_area_rel=min_component_area_rel,
            weld_eps_rel=float(spec.get("weld_eps_rel", 1e-10)),
            log_path=LOGS_DIR / "cellocut_repair.log",
        )
        repair_vram_gb = float(cello_data.get("peak_vram_gb", 0.0))
        _normalize_for_backend(
            clean_glb,
            clean_glb,
            winner_res.coordinate_frame,
            target_height,
        )
    elif repair_method == "none":
        _normalize_for_backend(
            raw_glb,
            clean_glb,
            winner_res.coordinate_frame,
            target_height,
        )
    else:
        raise ValueError(f"Unsupported repair_method: {repair_method!r}")

    repair_elapsed_s = time.monotonic() - t_rep0
    clean_metrics = compute_mesh_metrics(clean_glb)
    gen_elapsed_s = sum(c.elapsed_s for c in ranked_cands)
    peak_vram_gb = max(
        [c.peak_vram_gb for c in ranked_cands] + [repair_vram_gb]
    )

    rep_contract = (
        "inplace"
        if repair_method in ("inplace", "inplace_clean")
        else (
            "cellocut"
            if repair_method in ("cellocut", "cellocut_remesh")
            else ("none" if repair_method == "none" else "both")
        )
    )
    job_cfg = JobConfig(
        mode=input_mode,
        input_path=str(views_prepped),
        out_dir=str(OUTPUTS_DIR),
        backend=backend,
        repair_method=rep_contract,
        resolution=resolution,
        seed=seeds[0],
        seeds=len(seeds),
        seed_list=seeds,
        target_height=target_height,
        fov_deg=fov_deg,
    )
    meta = RunMeta(
        job=job_cfg,
        selected_seed=winner_cand.seed,
        candidates=ranked_cands,
        raw_metrics=raw_metrics,
        repaired_metrics=clean_metrics,
        backend_version=backend,
        elapsed_s=time.monotonic() - t_total0,
        peak_vram_gb=peak_vram_gb,
        gpu_name=str(gpu_info.get("gpu_name") or "unknown"),
        outputs={
            "raw_glb": "raw.glb",
            "clean_white_glb": "clean_white.glb",
            "views_prepped": "views_prepped",
            "meta_json": "meta.json",
            "selection_reason": selection_reason,
            "gen_elapsed_s": f"{gen_elapsed_s:.3f}",
            "repair_elapsed_s": f"{repair_elapsed_s:.3f}",
        },
    )
    meta.write_json(OUTPUTS_DIR / "meta.json")
    return meta.to_dict()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _package_chunked_outputs(hb: HeartbeatState) -> Path:
    """Gzip-tarball `outputs/` and `logs/`, splitting into <=15 MB chunks if needed."""
    hb.set_stage("package_outputs")
    TRANSFER_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = TRANSFER_DIR / "outputs_bundle.tar.gz"
    if archive_path.exists():
        archive_path.unlink()

    with tarfile.open(archive_path, "w:gz") as tf:
        if OUTPUTS_DIR.exists():
            for item in sorted(OUTPUTS_DIR.rglob("*")):
                # Skip temporary staging directories inside outputs.
                if "_sv_stage_" in item.parts:
                    continue
                tf.add(item, arcname=str(Path("outputs") / item.relative_to(OUTPUTS_DIR)), recursive=False)
        if LOGS_DIR.exists():
            for item in sorted(LOGS_DIR.rglob("*")):
                if item.is_file():
                    tf.add(item, arcname=str(Path("logs") / item.relative_to(LOGS_DIR)), recursive=False)

    total_size = archive_path.stat().st_size
    full_sha256 = _sha256_file(archive_path)

    chunks_meta: list[dict[str, Any]] = []
    if total_size <= MAX_CHUNK_BYTES:
        chunks_meta.append(
            {
                "name": archive_path.name,
                "size": total_size,
                "sha256": full_sha256,
            }
        )
    else:
        part_idx = 0
        with archive_path.open("rb") as src:
            while True:
                buf = src.read(MAX_CHUNK_BYTES)
                if not buf:
                    break
                part_name = f"{archive_path.name}.part{part_idx:03d}"
                part_path = TRANSFER_DIR / part_name
                part_path.write_bytes(buf)
                chunks_meta.append(
                    {
                        "name": part_name,
                        "size": len(buf),
                        "sha256": hashlib.sha256(buf).hexdigest(),
                    }
                )
                part_idx += 1
        archive_path.unlink()

    manifest = {
        "archive_name": "outputs_bundle.tar.gz",
        "total_size": total_size,
        "sha256": full_sha256,
        "max_chunk_bytes": MAX_CHUNK_BYTES,
        "chunks": chunks_meta,
    }
    manifest_path = TRANSFER_DIR / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"[package] Archive size={total_size / (1024*1024):.2f} MB "
        f"({len(chunks_meta)} chunk(s)), sha256={full_sha256[:12]}...",
        flush=True,
    )
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["S1_CACHE_DIR"] = str(CACHE_ROOT)

    parser = argparse.ArgumentParser(description="Colab Remote Runner for S1 Pixal3D")
    parser.add_argument("--spec", default=None, help="Path to job_spec.json")
    parser.add_argument("--cellocut-worker", action="store_true")
    parser.add_argument("--in-glb", default=None)
    parser.add_argument("--out-glb", default=None)
    parser.add_argument("--meta-out", default=None)
    parser.add_argument("--remesh-resolution", type=int, default=1024)
    parser.add_argument("--remesh-fill", type=float, default=20.0)
    parser.add_argument("--remesh-band", type=float, default=1.0)
    parser.add_argument("--project-back", type=float, default=0.5)
    parser.add_argument("--qef", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--smooth-iters",
        "--cellocut-smooth-iters",
        dest="smooth_iters",
        type=int,
        default=0,
    )
    parser.add_argument("--min-component-area-rel", type=float, default=0.005)
    parser.add_argument("--weld-eps-rel", type=float, default=1e-10)
    args = parser.parse_args(argv)

    if str(WORK_DIR) not in sys.path:
        sys.path.insert(0, str(WORK_DIR))
    existing_pp = os.environ.get("PYTHONPATH", "")
    if str(WORK_DIR) not in existing_pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            f"{WORK_DIR}{os.pathsep}{existing_pp}" if existing_pp else str(WORK_DIR)
        )

    if args.cellocut_worker:
        if not args.in_glb or not args.out_glb or not args.meta_out:
            raise ValueError("--cellocut-worker requires --in-glb, --out-glb, and --meta-out")
        _run_cellocut_worker(
            in_glb=Path(args.in_glb),
            out_glb=Path(args.out_glb),
            meta_out=Path(args.meta_out),
            resolution=args.remesh_resolution,
            fill=args.remesh_fill,
            band=args.remesh_band,
            project_back=args.project_back,
            qef=bool(args.qef),
            smooth_iters=args.smooth_iters,
            min_component_area_rel=args.min_component_area_rel,
            weld_eps_rel=args.weld_eps_rel,
        )
        return 0

    if not args.spec:
        raise ValueError("--spec is required unless --cellocut-worker is set")

    for d in (WORK_DIR, LOGS_DIR, MODELS_DIR, OUTPUTS_DIR, TRANSFER_DIR):
        d.mkdir(parents=True, exist_ok=True)

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    hb = HeartbeatState(interval_s=20.0)
    hb.start()
    try:
        hb.set_stage("check_gpu")
        gpu_info = _query_gpu_summary()
        print(f"[colab-remote] GPU: {gpu_info}", flush=True)

        hb.set_stage("install_python_deps")
        _ensure_python_deps()

        mode = str(spec.get("mode", "run")).lower()
        input_mode = str(spec.get("input_mode", "mv")).lower()
        backend = str(spec.get("backend", "path_a")).lower()
        repair_method = str(spec.get("repair_method", "inplace_clean")).lower()

        from quadvox.s0_views.preprocess import discover_input_views

        raw_paths = discover_input_views(Path(spec["raw_views_dir"]), mode=input_mode)
        any_needs_matting = any(_needs_matting(p) for p in raw_paths)
        need_comfy = (
            mode == "benchmark"
            or backend in ("path_a", "path_a_comfy", "comfy", "comfyui")
            or repair_method in ("cellocut", "cellocut_remesh")
            or any_needs_matting
        )
        if need_comfy:
            hb.set_stage("setup_comfyui")
            _ensure_comfyui()

        views_prepped = _prepare_shared_views(spec, hb)
        if mode == "benchmark":
            _run_benchmark_mode(spec, views_prepped, gpu_info, hb)
        elif mode == "run":
            _run_production_mode(spec, views_prepped, gpu_info, hb)
        else:
            raise ValueError(f"Unsupported remote job mode: {mode!r}")

        _package_chunked_outputs(hb)
        hb.set_stage("completed")
        print("[colab-remote] STATUS=OK", flush=True)
        return 0
    finally:
        hb.stop()


if __name__ == "__main__":
    raise SystemExit(main())
