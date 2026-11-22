"""Portable binary/tool path resolvers with zero machine-specific hardcoding.

Resolution priority for every external tool:
1. Explicit caller argument (when provided and non-empty)
2. Environment variable (`QUADVOX_<TOOL>_BIN` or standard upstream env var)
3. `shutil.which(<name>)` on `$PATH`
4. Conventional user-local fallback (`~/.local/bin/<name>` or repo-local `tools/`)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _resolve_executable(
    explicit: str | Path | None,
    *,
    env_vars: tuple[str, ...],
    names: tuple[str, ...],
    fallback_paths: tuple[Path, ...] = (),
    default_name: str,
) -> Path:
    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser()
    for var in env_vars:
        val = os.environ.get(var, "").strip()
        if val:
            return Path(val).expanduser()
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    for fp in fallback_paths:
        if fp.is_file():
            return fp
    return Path(default_name)


def resolve_blender_bin(explicit: str | Path | None = None) -> Path:
    """Resolve Blender 5.x executable via explicit arg, `QUADVOX_BLENDER_BIN`/`BLENDER_BIN`, or PATH."""
    return _resolve_executable(
        explicit,
        env_vars=("QUADVOX_BLENDER_BIN", "BLENDER_BIN"),
        names=("blender", "blender-5.3"),
        fallback_paths=(
            Path.home() / ".local" / "bin" / "blender",
            Path("/usr/local/bin/blender"),
        ),
        default_name="blender",
    )


def resolve_colab_bin(explicit: str | Path | None = None) -> str:
    """Resolve `colab` CLI executable (`google-colab-cli`) via explicit arg, env, or PATH."""
    p = _resolve_executable(
        explicit,
        env_vars=("QUADVOX_COLAB_BIN", "COLAB_BIN"),
        names=("colab",),
        fallback_paths=(Path.home() / ".local" / "bin" / "colab",),
        default_name="colab",
    )
    return str(p)


def resolve_requad_bin(explicit: str | Path | None = None) -> Path:
    """Resolve external GPL-3.0 ReQuad engine binary/dir (installed outside git)."""
    repo_root = Path(__file__).resolve().parents[2]
    return _resolve_executable(
        explicit,
        env_vars=("QUADVOX_REQUAD_BIN", "REQUAD_BIN"),
        names=("requad",),
        fallback_paths=(
            repo_root / "tools" / "requad" / "engine" / "linux-x64" / "quadwild",
            Path.home() / ".local" / "bin" / "requad",
        ),
        default_name="requad",
    )


def resolve_quadwild_bin(explicit: str | Path | None = None) -> Path:
    """Resolve external GPL-3.0 QuadWild CLI binary (installed outside git)."""
    repo_root = Path(__file__).resolve().parents[2]
    return _resolve_executable(
        explicit,
        env_vars=("QUADVOX_QUADWILD_BIN", "QUADWILD_BIN"),
        names=("quadwild",),
        fallback_paths=(
            repo_root / "tools" / "quadwild" / "quadwild",
            Path.home() / ".local" / "bin" / "quadwild",
        ),
        default_name="quadwild",
    )


def resolve_autoremesher_bin(explicit: str | Path | None = None) -> Path:
    """Resolve AutoRemesher CLI binary."""
    return _resolve_executable(
        explicit,
        env_vars=("QUADVOX_AUTOREMESHER_BIN", "AUTOREMESHER_BIN"),
        names=("autoremesher", "AutoRemesher"),
        fallback_paths=(Path.home() / ".local" / "bin" / "autoremesher",),
        default_name="autoremesher",
    )


def resolve_comfyui_dir(explicit: str | Path | None = None) -> Path:
    """Resolve ComfyUI installation directory."""
    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser()
    for var in ("QUADVOX_COMFYUI_DIR", "COMFYUI_DIR"):
        val = os.environ.get(var, "").strip()
        if val:
            return Path(val).expanduser()
    return Path.home() / ".cache" / "quadvox" / "ComfyUI"


def resolve_pixal3d_cpp_dir(explicit: str | Path | None = None) -> Path:
    """Resolve pixal3d.cpp installation directory."""
    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser()
    for var in ("QUADVOX_PIXAL3D_CPP_DIR", "PIXAL3D_CPP_DIR"):
        val = os.environ.get(var, "").strip()
        if val:
            return Path(val).expanduser()
    return Path.home() / ".cache" / "quadvox" / "pixal3d.cpp"
