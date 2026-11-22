"""Path B generator adapter: pure C++/GGML Q8_0 pixal3d.cpp (`--no-texture` shape-only).

Verified upstream URLs:
- Repository: https://github.com/raven38/pixal3d.cpp
- Prebuilt CUDA Linux release: https://github.com/raven38/pixal3d.cpp/releases/download/v0.10.1-desktop-alpha/trellis-cuda-linux-x64.tar.gz
- MV Q8_0 weights: https://huggingface.co/raven38/pixal3d-q8_0-v1
- SV Q8_0 weights: https://huggingface.co/raven38/pixal3d-sv-q8_0-v1

Verified in `src/trellis_cli.cpp` (`mv_check_weights`):
When `--no-texture` (`cfg.texture == false`) is passed, `mv_check_weights` only checks
`ss`, `shape512`, and `shape1024` (plus `dinov3.gguf`, `pixal3d_naf.gguf`, `ss_dec.gguf`,
and `shape_dec.gguf`), so `pixal3d_tex_flow_1024_*.gguf` and `tex_dec.gguf` do NOT need
to be downloaded.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from quadvox.s1_geometry.generators.base import (
    BaseShapeGenerator,
    GenerationResult,
    default_cache_dir,
    monitor_subprocess_with_vram,
    resolve_views_from_dir,
)
from quadvox.s0_views.preprocess import build_orbit_transforms


PIXAL3D_CPP_REPO_URL = "https://github.com/raven38/pixal3d.cpp"
PIXAL3D_CPP_GIT_URL = "https://github.com/raven38/pixal3d.cpp.git"
PIXAL3D_CPP_RELEASE_URL = (
    "https://github.com/raven38/pixal3d.cpp/releases/download/"
    "v0.10.1-desktop-alpha/trellis-cuda-linux-x64.tar.gz"
)
HF_RAVEN_MV_REPO = "raven38/pixal3d-q8_0-v1"
HF_RAVEN_SV_REPO = "raven38/pixal3d-sv-q8_0-v1"

# Prebuilt `trellis-cuda-linux-x64.tar.gz` targets CUDA 13.1 (driver >= R590).
CUDA13_MIN_DRIVER_MAJOR = 590

SHARED_GGUF_FILES = (
    "dinov3.gguf",
    "pixal3d_naf.gguf",
    "ss_dec.gguf",
    "shape_dec.gguf",
)
MV_GEOMETRY_GGUF_FILES = SHARED_GGUF_FILES + (
    "pixal3d_ss_flow_mv.gguf",
    "pixal3d_shape_flow_512_mv.gguf",
    "pixal3d_shape_flow_1024_mv.gguf",
)
SV_GEOMETRY_GGUF_FILES = SHARED_GGUF_FILES + (
    "pixal3d_ss_flow_sv.gguf",
    "pixal3d_shape_flow_512_sv.gguf",
    "pixal3d_shape_flow_1024_sv.gguf",
)


def default_cpp_root(cache_dir: str | Path | None = None) -> Path:
    """Resolve `pixal3d.cpp` cache root under `/content/cache` or local `runs/_cache`."""
    env_dir = os.environ.get("PIXAL3D_CPP_ROOT")
    if env_dir:
        return Path(env_dir)
    base = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return base / "pixal3d_cpp"


def default_cpp_models_dir(
    mode: str = "mv",
    cache_dir: str | Path | None = None,
) -> Path:
    """Resolve GGUF models directory for MV or SV (`mv` and `sv` cannot share 1 dir if flows differ)."""
    norm_mode = str(mode).lower()
    env_key = "PIXAL3D_CPP_MODELS_SV" if norm_mode == "sv" else "PIXAL3D_CPP_MODELS_MV"
    if os.environ.get(env_key):
        return Path(os.environ[env_key])
    if os.environ.get("PIXAL3D_CPP_MODELS"):
        return Path(os.environ["PIXAL3D_CPP_MODELS"])
    base = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return base / "models" / f"pixal3d_cpp_{norm_mode}"


def required_gguf_files(mode: str = "mv") -> tuple[str, ...]:
    """Return the geometry-only GGUF filenames needed for `--no-texture` runs."""
    norm_mode = str(mode).lower()
    if norm_mode == "mv":
        return MV_GEOMETRY_GGUF_FILES
    if norm_mode == "sv":
        return SV_GEOMETRY_GGUF_FILES
    raise ValueError(f"Unsupported mode {mode!r}; expected 'mv' or 'sv'")


def check_required_ggufs(models_dir: str | Path, mode: str = "mv") -> list[Path]:
    """Return missing GGUF paths under `models_dir`."""
    root = Path(models_dir)
    missing: list[Path] = []
    for fname in required_gguf_files(mode):
        p = root / fname
        if not p.is_file() or p.stat().st_size == 0:
            missing.append(p)
    return missing


def query_nvidia_driver_version() -> tuple[int | None, str]:
    """Return `(driver_major, raw_driver_version)` from `nvidia-smi`."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5.0,
        )
        raw = out.strip().splitlines()[0].strip()
        m = re.match(r"^(\d+)", raw)
        if m:
            return int(m.group(1)), raw
        return None, raw
    except Exception:
        return None, ""


def check_cuda13_driver_compat() -> tuple[bool, int | None, str]:
    """Check whether the host NVIDIA user-mode driver satisfies CUDA 13.1 (>= R590)."""
    major, raw = query_nvidia_driver_version()
    if major is None:
        return False, None, "nvidia-smi driver version unavailable"
    if major < CUDA13_MIN_DRIVER_MAJOR:
        return (
            False,
            major,
            f"driver {raw} (< R{CUDA13_MIN_DRIVER_MAJOR} required for CUDA 13.1 prebuilt binary)",
        )
    return True, major, f"driver {raw} >= R{CUDA13_MIN_DRIVER_MAJOR}"


def _find_trellis_cli_in_tree(root: Path) -> tuple[Path | None, list[Path]]:
    """Locate `trellis-cli` and shared library directories inside an extracted tree."""
    if not root.exists():
        return None, []
    cli_path: Path | None = None
    for cand in root.rglob("trellis-cli"):
        if cand.is_file():
            cli_path = cand
            break
    if cli_path is None:
        return None, []
    lib_dirs = sorted(
        {
            p.parent
            for p in root.rglob("*.so*")
            if p.is_file()
        }
    )
    if cli_path.parent not in lib_dirs:
        lib_dirs.insert(0, cli_path.parent)
    return cli_path, lib_dirs


def _find_nvcc() -> Path | None:
    found = shutil.which("nvcc")
    if found:
        return Path(found)
    for cand in (
        Path("/usr/local/cuda/bin/nvcc"),
        Path("/usr/local/cuda-12/bin/nvcc"),
        Path("/usr/local/cuda-13/bin/nvcc"),
    ):
        if cand.is_file():
            return cand
    return None


def probe_trellis_cli(bin_path: Path, lib_dirs: Sequence[Path] = ()) -> tuple[bool, str]:
    """Verify `trellis-cli` executes `--help` cleanly with `LD_LIBRARY_PATH`."""
    if not bin_path.is_file():
        return False, f"binary not found: {bin_path}"
    os.chmod(bin_path, 0o755)
    env = os.environ.copy()
    ld_parts = [str(p) for p in lib_dirs if Path(p).is_dir()]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    if ld_parts:
        env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    try:
        proc = subprocess.run(
            [str(bin_path), "--help"],
            env=env,
            capture_output=True,
            text=True,
            timeout=15.0,
        )
        combined = (proc.stdout + "\n" + proc.stderr).strip()
        if "--views" in combined and "--no-texture" in combined:
            return True, "ok"
        return False, f"unexpected --help output (rc={proc.returncode}): {combined[-400:]}"
    except Exception as exc:
        return False, str(exc)


def download_prebuilt_trellis_cli(
    cpp_root: Path,
    release_url: str = PIXAL3D_CPP_RELEASE_URL,
) -> tuple[Path, list[Path]]:
    """Download and extract the official prebuilt CUDA Linux `trellis-cli` tarball."""
    prebuilt_dir = cpp_root / "prebuilt"
    existing_bin, existing_libs = _find_trellis_cli_in_tree(prebuilt_dir)
    if existing_bin is not None:
        return existing_bin, existing_libs

    prebuilt_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cpp_root / "trellis-cuda-linux-x64.tar.gz"
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        print(f"[path_b] Downloading prebuilt tarball from {release_url}...", flush=True)
        with urllib.request.urlopen(release_url, timeout=120.0) as resp, archive_path.open(
            "wb"
        ) as dst:
            shutil.copyfileobj(resp, dst, length=1024 * 1024)

    print(f"[path_b] Extracting {archive_path.name}...", flush=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(prebuilt_dir)

    cli_path, lib_dirs = _find_trellis_cli_in_tree(prebuilt_dir)
    if cli_path is None:
        raise FileNotFoundError(
            f"trellis-cli binary not found after extracting {archive_path}"
        )
    return cli_path, lib_dirs


def build_trellis_cli_from_source(
    cpp_root: Path,
    log_path: Path | None = None,
) -> tuple[Path, list[Path]]:
    """Clone `raven38/pixal3d.cpp` and compile `trellis-cli` with local `nvcc`."""
    src_dir = cpp_root / "src"
    build_dir = cpp_root / "build"
    built_bin = build_dir / "trellis-cli"
    if built_bin.is_file():
        return built_bin, [build_dir]

    nvcc = _find_nvcc()
    if nvcc is None:
        raise RuntimeError(
            "Cannot fall back to source build for pixal3d.cpp: `nvcc` not found."
        )

    logs_root = default_cache_dir() / "logs"
    resolved_log = log_path if log_path is not None else (logs_root / "path_b_build.log")
    resolved_log.parent.mkdir(parents=True, exist_ok=True)

    if not (src_dir / "CMakeLists.txt").is_file():
        print(f"[path_b] Cloning {PIXAL3D_CPP_GIT_URL} (--recurse-submodules)...", flush=True)
        rc, _, _, tail = monitor_subprocess_with_vram(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--recurse-submodules",
                "--shallow-submodules",
                PIXAL3D_CPP_GIT_URL,
                str(src_dir),
            ],
            log_path=resolved_log,
            heartbeat_label="path_b_clone",
        )
        if rc != 0:
            raise RuntimeError(f"git clone pixal3d.cpp failed (rc={rc}):\n{tail}")

    def _configure_and_build(cuda_arch: str) -> None:
        cmake_cmd = [
            "cmake",
            "-S",
            str(src_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
            f"-DCMAKE_CUDA_COMPILER={nvcc}",
        ]
        if shutil.which("ninja"):
            cmake_cmd.extend(["-G", "Ninja"])

        print(f"[path_b] Configuring pixal3d.cpp with CUDA arch={cuda_arch}...", flush=True)
        rc, _, _, tail = monitor_subprocess_with_vram(
            cmake_cmd,
            log_path=resolved_log,
            heartbeat_label="path_b_cmake",
        )
        if rc != 0:
            raise RuntimeError(f"cmake configure for pixal3d.cpp failed (rc={rc}):\n{tail}")

        jobs = str(max(2, min(16, os.cpu_count() or 4)))
        print(f"[path_b] Building trellis-cli (-j{jobs})...", flush=True)
        rc, _, _, tail = monitor_subprocess_with_vram(
            ["cmake", "--build", str(build_dir), "-j", jobs, "--target", "trellis-cli"],
            log_path=resolved_log,
            heartbeat_label="path_b_compile",
        )
        if rc != 0:
            raise RuntimeError(f"cmake build for trellis-cli failed (rc={rc}):\n{tail}")

    try:
        _configure_and_build("native")
    except RuntimeError as exc:
        if "unsupported gpu architecture" in str(exc).lower():
            print(
                "[path_b] nvcc rejected 'native' arch; retrying with PTX fallback 89;90-virtual...",
                flush=True,
            )
            shutil.rmtree(build_dir, ignore_errors=True)
            _configure_and_build("89-real;90-virtual")
        else:
            raise

    if not built_bin.is_file():
        raise FileNotFoundError(f"Build finished but {built_bin} was not produced")
    return built_bin, [build_dir]


def ensure_trellis_cli(
    cpp_root: str | Path | None = None,
    *,
    log_path: str | Path | None = None,
    allow_source_fallback: bool = True,
) -> tuple[Path, list[Path], str]:
    """Return `(trellis_cli_path, lib_dirs, origin_tag)`, checking CUDA 13.1 compatibility."""
    root = Path(cpp_root) if cpp_root is not None else default_cpp_root()
    root.mkdir(parents=True, exist_ok=True)
    log_p = Path(log_path) if log_path is not None else None

    env_bin = os.environ.get("PIXAL3D_CPP_BIN")
    if env_bin:
        cand = Path(env_bin)
        ok, msg = probe_trellis_cli(cand, [cand.parent])
        if ok:
            return cand, [cand.parent], "env_override"
        raise RuntimeError(f"PIXAL3D_CPP_BIN={cand} failed probe: {msg}")

    built_bin = root / "build" / "trellis-cli"
    if built_bin.is_file():
        ok, _ = probe_trellis_cli(built_bin, [built_bin.parent])
        if ok:
            return built_bin, [built_bin.parent], "source_nvcc"

    driver_ok, driver_major, driver_msg = check_cuda13_driver_compat()
    if driver_ok:
        try:
            pre_bin, pre_libs = download_prebuilt_trellis_cli(root)
            ok, probe_msg = probe_trellis_cli(pre_bin, pre_libs)
            if ok:
                print(
                    f"[path_b] Using prebuilt CUDA 13.1 trellis-cli ({driver_msg})",
                    flush=True,
                )
                return pre_bin, pre_libs, "prebuilt_cuda13"
            print(
                f"[path_b] Prebuilt probe failed ({probe_msg}); falling back to nvcc build.",
                flush=True,
            )
        except Exception as exc:
            if not allow_source_fallback:
                raise
            print(
                f"[path_b] Prebuilt setup failed ({exc}); falling back to nvcc build.",
                flush=True,
            )
    else:
        print(
            f"[path_b] CUDA 13.1 driver check failed ({driver_msg}); "
            "building pixal3d.cpp from source with local nvcc...",
            flush=True,
        )

    if not allow_source_fallback:
        raise RuntimeError(
            f"Prebuilt CUDA 13.1 binary incompatible ({driver_msg}) and source fallback disabled."
        )
    src_bin, src_libs = build_trellis_cli_from_source(root, log_path=log_p)
    return src_bin, src_libs, "source_nvcc"


def _is_cuda_driver_failure(log_tail: str) -> bool:
    markers = (
        "CUDA driver version is insufficient",
        "cudaErrorInsufficientDriver",
        "ptxjitcompiler",
        "libcudart.so.13",
        "cudaGetDeviceCount",
        "no CUDA-capable device",
        "ggml_cuda",
    )
    lower = log_tail.lower()
    return any(m.lower() in lower for m in markers)


class PathBCppGenerator(BaseShapeGenerator):
    """Pure C++/GGML Q8_0 Pixal3D shape-only generator via `--no-texture`."""

    backend_name: str = "path_b"
    coordinate_frame: str = "y_up_neg_z_front"
    supported_resolutions: tuple[int, ...] = (1024, 1536)

    def __init__(
        self,
        cpp_root: str | Path | None = None,
        models_dir: str | Path | None = None,
        models_sv_dir: str | Path | None = None,
        allow_source_fallback: bool = True,
    ) -> None:
        self.cpp_root = (
            Path(cpp_root) if cpp_root is not None else default_cpp_root()
        )
        self.models_dir_mv = (
            Path(models_dir)
            if models_dir is not None
            else default_cpp_models_dir("mv")
        )
        self.models_dir_sv = (
            Path(models_sv_dir)
            if models_sv_dir is not None
            else default_cpp_models_dir("sv")
        )
        self.allow_source_fallback = bool(allow_source_fallback)

    def _prepare_sv_staging(
        self,
        views_dir: Path,
        out_glb: Path,
        resolution: int,
        fov_deg: float,
    ) -> tuple[list[str], Path | None]:
        """Prepare CLI flags for SV (`--sv-image` at 1024, or 1-frame `--views` for 1536)."""
        view_paths, _ = resolve_views_from_dir(views_dir, mode="sv")
        front_view = view_paths[0]
        fov_rad = math.radians(float(fov_deg))

        if resolution == 1024:
            return [
                "--sv-image",
                str(front_view),
                "--fov",
                f"{fov_rad:.8f}",
            ], None

        # `trellis-cli --sv-image` caps `--res` at 1024; for 1536, stage a 1-frame
        # `--views` dir with SV gauge distance (`pad_factor=1.0`) and `--pixal3d-weights sv`.
        stage_dir = out_glb.parent / f"_sv_stage_{out_glb.stem}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        staged_png = stage_dir / front_view.name
        shutil.copy2(front_view, staged_png)
        tf = build_orbit_transforms(
            file_paths=[staged_png.name],
            fov_deg=float(fov_deg),
            pad_factor=1.0,
            mesh_scale=1.0,
            azimuths_deg=[0.0],
        )
        (stage_dir / "transforms.json").write_text(
            json.dumps(tf, indent=2) + "\n", encoding="utf-8"
        )
        return [
            "--views",
            str(stage_dir),
            "--pixal3d-weights",
            "sv",
        ], stage_dir

    def generate(
        self,
        views_dir: str | Path,
        out_glb: str | Path,
        *,
        mode: str = "mv",
        resolution: int = 1536,
        seed: int = 42,
        fov_deg: float = 20.0,
        log_path: str | Path | None = None,
    ) -> GenerationResult:
        norm_mode, res = self.validate_request(mode, resolution)
        src = Path(views_dir)
        dst = Path(out_glb)
        dst.parent.mkdir(parents=True, exist_ok=True)

        models_dir = self.models_dir_sv if norm_mode == "sv" else self.models_dir_mv
        missing = check_required_ggufs(models_dir, mode=norm_mode)
        if missing:
            missing_str = ", ".join(str(p) for p in missing)
            raise FileNotFoundError(
                f"Path B required GGUF weights missing under {models_dir}: {missing_str}"
            )

        resolved_log = (
            Path(log_path)
            if log_path is not None
            else dst.with_suffix(".path_b.log")
        )
        cli_bin, lib_dirs, origin = ensure_trellis_cli(
            self.cpp_root,
            log_path=resolved_log.with_suffix(".build.log"),
            allow_source_fallback=self.allow_source_fallback,
        )

        if norm_mode == "mv":
            input_args = [
                "--views",
                str(src),
                "--pixal3d-weights",
                "mv",
            ]
        else:
            input_args, _ = self._prepare_sv_staging(
                src, dst, resolution=res, fov_deg=fov_deg
            )

        def _invoke(bin_path: Path, libs: Sequence[Path]) -> tuple[int, float, float, str]:
            cmd = [
                str(bin_path),
                *input_args,
                "--models",
                str(models_dir),
                "--res",
                str(res),
                "--seed",
                str(int(seed)),
                "--no-texture",
                "--require-gpu",
                "--output",
                str(dst),
            ]
            env: dict[str, str] = {}
            ld_parts = [str(p) for p in libs if Path(p).is_dir()]
            if os.environ.get("LD_LIBRARY_PATH"):
                ld_parts.append(os.environ["LD_LIBRARY_PATH"])
            if ld_parts:
                env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

            print(
                f"[path_b] Running trellis-cli ({origin}) res={res} mode={norm_mode} seed={seed}...",
                flush=True,
            )
            return monitor_subprocess_with_vram(
                cmd,
                env=env,
                log_path=resolved_log,
                heartbeat_label=f"path_b_{norm_mode}_s{seed}",
            )

        rc, elapsed_s, peak_vram_gb, tail = _invoke(cli_bin, lib_dirs)
        if (
            rc != 0
            and origin == "prebuilt_cuda13"
            and self.allow_source_fallback
            and _is_cuda_driver_failure(tail)
        ):
            print(
                "[path_b] Prebuilt CUDA 13.1 binary failed driver check at runtime; "
                "building from source with local nvcc and retrying...",
                flush=True,
            )
            cli_bin, lib_dirs = build_trellis_cli_from_source(
                self.cpp_root,
                log_path=resolved_log.with_suffix(".build.log"),
            )
            origin = "source_nvcc"
            rc, elapsed_s, peak_vram_gb, tail = _invoke(cli_bin, lib_dirs)

        if rc != 0:
            raise RuntimeError(
                f"Path B trellis-cli failed (rc={rc}, origin={origin}) for seed={seed}:\n{tail}"
            )
        if not dst.is_file() or dst.stat().st_size == 0:
            raise RuntimeError(f"Path B succeeded but output GLB is missing/empty: {dst}")

        ply_sidecar = dst.with_suffix(".ply")
        extra: dict[str, Any] = {
            "cli_bin": str(cli_bin),
            "cli_origin": origin,
            "models_dir": str(models_dir),
        }
        if ply_sidecar.is_file():
            extra["ply_sidecar"] = str(ply_sidecar)

        print(
            f"[path_b] Done in {elapsed_s:.1f}s -> {dst} (peak_vram={peak_vram_gb:.2f}GB)",
            flush=True,
        )
        return GenerationResult(
            backend=self.backend_name,
            mode=norm_mode,
            seed=int(seed),
            resolution=res,
            raw_glb=str(dst),
            elapsed_s=elapsed_s,
            peak_vram_gb=peak_vram_gb,
            coordinate_frame=self.coordinate_frame,
            log_path=str(resolved_log),
            extra=extra,
        )
