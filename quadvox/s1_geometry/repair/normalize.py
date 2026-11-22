"""Coordinate normalization for S1 white-mesh GLBs.

Normalizes meshes to canonical glTF right-handed `y_up_z_front`:
- Up axis: `+Y` (foot bottom at `Y = 0.0`, top at `Y = target_height`)
- Front axis: `+Z` (character facing `+Z`, character's left at `+X`)
- Horizontally centered around `X = 0.0`, `Z = 0.0` (bbox midpoints)
- Uniform scale so total height equals `target_height` (default `1.0`)

Supports configurable axis mapping (`up_axis`, `front_axis`) and `yaw_deg`
pre-rotation (e.g., `front_axis="+x"` or `yaw_deg=-90.0` for `+X`-facing exports,
or `yaw_deg=180.0` for `-Z`-facing exports).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh

from .metrics import load_mesh_raw


_AXIS_VECTORS: dict[str, np.ndarray] = {
    "+x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "+y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "-y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "+z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "-z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
}


def _parse_axis(name: str) -> np.ndarray:
    key = str(name).strip().lower()
    if key not in _AXIS_VECTORS:
        raise ValueError(f"Unsupported axis {name!r}; expected one of {sorted(_AXIS_VECTORS)}")
    return _AXIS_VECTORS[key]


def _rot_x(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_alignment_rotation(
    *,
    up_axis: str = "+y",
    front_axis: str | None = None,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """Build a proper 3x3 rotation matrix (`det == +1`) mapping source frame to `+Y` up, `+Z` front."""
    src_up = _parse_axis(up_axis)
    if front_axis is None:
        if abs(float(src_up[2])) < 0.5:
            src_front = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            # Blender/Z-up convention defaults to -Y front.
            src_front = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    else:
        src_front = _parse_axis(front_axis)

    if abs(float(np.dot(src_up, src_front))) > 1e-6:
        raise ValueError(f"up_axis={up_axis!r} and front_axis={front_axis!r} must be orthogonal")

    src_left = np.cross(src_up, src_front)
    # Rows map [src_left -> +X, src_up -> +Y, src_front -> +Z].
    r_basis = np.stack([src_left, src_up, src_front], axis=0)

    r_euler = np.eye(3, dtype=np.float64)
    if float(roll_deg) != 0.0:
        r_euler = _rot_z(math.radians(float(roll_deg))) @ r_euler
    if float(pitch_deg) != 0.0:
        r_euler = _rot_x(math.radians(float(pitch_deg))) @ r_euler
    if float(yaw_deg) != 0.0:
        r_euler = _rot_y(math.radians(float(yaw_deg))) @ r_euler

    rot = r_euler @ r_basis
    if abs(float(np.linalg.det(rot)) - 1.0) > 1e-6:
        raise RuntimeError(f"Rotation matrix determinant must be +1, got {np.linalg.det(rot)}")
    return rot


def normalize_mesh(
    mesh: trimesh.Trimesh,
    *,
    target_height: float = 1.0,
    yaw_deg: float = 0.0,
    up_axis: str = "+y",
    front_axis: str | None = None,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> trimesh.Trimesh:
    """Orient, scale, and ground a mesh to `+Y` up, `+Z` front, foot `Y=0`, centered XZ."""
    if float(target_height) <= 0.0:
        raise ValueError(f"target_height must be positive, got {target_height}")

    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    if len(verts) == 0 or len(faces) == 0:
        raise ValueError("Cannot normalize an empty mesh")

    rot = build_alignment_rotation(
        up_axis=up_axis,
        front_axis=front_axis,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
    )
    verts = verts @ rot.T

    finite = np.isfinite(verts).all(axis=1)
    if not np.any(finite):
        raise ValueError("Mesh contains no finite vertices")

    lo = verts[finite].min(axis=0)
    hi = verts[finite].max(axis=0)
    raw_height = float(hi[1] - lo[1])
    if raw_height <= 1e-12:
        raise ValueError(f"Mesh has degenerate height along +Y: {raw_height}")

    scale = float(target_height) / raw_height
    center_x = 0.5 * float(lo[0] + hi[0])
    min_y = float(lo[1])
    center_z = 0.5 * float(lo[2] + hi[2])

    verts[:, 0] = (verts[:, 0] - center_x) * scale
    verts[:, 1] = (verts[:, 1] - min_y) * scale
    verts[:, 2] = (verts[:, 2] - center_z) * scale

    out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if len(out.faces) > 0:
        out.vertex_normals
    return out


def normalize_glb(
    input_glb: str | Path,
    output_glb: str | Path,
    *,
    target_height: float = 1.0,
    yaw_deg: float = 0.0,
    up_axis: str = "+y",
    front_axis: str | None = None,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> trimesh.Trimesh:
    """Load raw GLB (`process=False`), normalize to `y_up_z_front`, and export (`process=False`)."""
    raw = load_mesh_raw(input_glb)
    norm = normalize_mesh(
        raw,
        target_height=target_height,
        yaw_deg=yaw_deg,
        up_axis=up_axis,
        front_axis=front_axis,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
    )
    out_path = Path(output_glb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm.export(str(out_path))
    return norm
