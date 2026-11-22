"""S1 Pixal3D shape-only generator backends (Path A ComfyUI & Path B pixal3d.cpp)."""

from __future__ import annotations

from typing import Any

from quadvox.s1_geometry.generators.base import (
    BaseShapeGenerator,
    GenerationResult,
    default_cache_dir,
    resolve_views_from_dir,
)
from quadvox.s1_geometry.generators.path_a_comfy import (
    PathAComfyGenerator,
    run_birefnet_matte,
    run_comfy_remesh_solid,
)
from quadvox.s1_geometry.generators.path_b_cpp import (
    PathBCppGenerator,
    ensure_trellis_cli,
)

_BACKEND_ALIASES: dict[str, type[BaseShapeGenerator]] = {
    "path_a": PathAComfyGenerator,
    "path_a_comfy": PathAComfyGenerator,
    "comfy": PathAComfyGenerator,
    "comfyui": PathAComfyGenerator,
    "path_b": PathBCppGenerator,
    "path_b_cpp": PathBCppGenerator,
    "cpp": PathBCppGenerator,
    "pixal3d.cpp": PathBCppGenerator,
}


def normalize_backend_name(backend: str) -> str:
    """Normalize backend alias to canonical `'path_a'` or `'path_b'`."""
    key = str(backend).strip().lower()
    if key in ("path_a", "path_a_comfy", "comfy", "comfyui"):
        return "path_a"
    if key in ("path_b", "path_b_cpp", "cpp", "pixal3d.cpp"):
        return "path_b"
    raise ValueError(
        f"Unsupported generator backend {backend!r}; expected 'path_a' or 'path_b'."
    )


def list_backends() -> tuple[str, ...]:
    """Return canonical backend names."""
    return ("path_a", "path_b")


def get_generator(backend: str, **kwargs: Any) -> BaseShapeGenerator:
    """Instantiate a generator backend by canonical name or alias."""
    key = str(backend).strip().lower()
    cls = _BACKEND_ALIASES.get(key)
    if cls is None:
        raise ValueError(
            f"Unsupported generator backend {backend!r}; expected one of {list_backends()}."
        )
    return cls(**kwargs)


__all__ = [
    "BaseShapeGenerator",
    "GenerationResult",
    "PathAComfyGenerator",
    "PathBCppGenerator",
    "default_cache_dir",
    "ensure_trellis_cli",
    "get_generator",
    "list_backends",
    "normalize_backend_name",
    "resolve_views_from_dir",
    "run_birefnet_matte",
    "run_comfy_remesh_solid",
]
