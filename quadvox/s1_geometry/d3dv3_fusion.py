"""Direct3D-V3 1536^3 4-View Lossless 8^3 Sparse-Voxel Rotation Fusion Engine.

Core mathematical insight (Layer 1 — 100% category-agnostic):
Direct3D-V3 operates on 8x8x8 sparse voxel patches (`512`, `1024`, `1536`).
Under 90-degree rotations around the Y-axis (`0 / 90 / 180 / 270 deg`, det = +1),
patch grid coordinates and internal 8^3 voxel layouts map bijectively and
losslessly with zero interpolation error (`max_abs_err == 0.0`). This enables
per-step denoising `x0` fusion across 4 orthographic views with:
  1. Orthographic Z-buffer ray occlusion culling + steep depth-cliff detection
  2. Downward-overhang crevice suppression (C1 front/back fallback in tight pockets)
  3. Timestep-scheduled temperature sharpening `beta(t)` and late CFG annealing
  4. Stage-512 multi-seed median SDF consensus & micro-shard pruning

Layer-2 configurable switches (OFF by default, supplied via YAML):
  - `roi_boxes`: `Sequence[ROISovereigntyBox]` locking designated high-priority
    frontal/hero regions (e.g. mech cockpit, chest crest, product logo, humanoid face)
    to their canonical view while suppressing orthogonal bleed.
  - `smoothing_zones`: `Sequence[ZoneSmoothingSpec]` for local geodesic transitions.

Reference URLs:
- Direct3D-V3 upstream: https://github.com/DreamTechAI/Direct3D-V3
- Official de-texture/clay prompt: https://github.com/DreamTechAI/Direct3D-V3/blob/main/assets/prompt.txt
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from quadvox.contracts import ROISovereigntyBox, ZoneSmoothingSpec

VIEW_ORDER: tuple[str, ...] = ("front", "back", "left", "right")
VIEW_DIRS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 1.0),   # front (+Z)
    (0.0, 0.0, -1.0),  # back  (-Z)
    (1.0, 0.0, 0.0),   # left  (+X)
    (-1.0, 0.0, 0.0),  # right (-X)
)
VIEW_TO_INDEX: dict[str, int] = {v: i for i, v in enumerate(VIEW_ORDER)}

STAGE_BETA_MAX: dict[int, float] = {512: 3.5, 1024: 4.5, 1536: 5.5}
STAGE512_SEEDS: tuple[int, ...] = (2025, 2026, 2027)
SHARPEN_T_START: float = 0.55
CFG_DECAY_T_START: float = 0.72
CFG_LATE_MIN: float = 3.5
TIMM_DINOV3_MODEL: str = "vit_huge_plus_patch16_dinov3.lvd1689m"


@dataclass
class D3DV3FusionConfig:
    """Configuration for 4-view Direct3D-V3 1536^3 sparse-voxel fusion."""

    stage_resolutions: tuple[int, ...] = (512, 1024, 1536)
    stage512_seeds: tuple[int, ...] = STAGE512_SEEDS
    refine_seed: int = 2026
    guidance_scale: float = 7.0
    sharpen_t_start: float = SHARPEN_T_START
    cfg_decay_t_start: float = CFG_DECAY_T_START
    cfg_late_min: float = CFG_LATE_MIN
    max_tokens: int = 120_000
    roi_boxes: list[ROISovereigntyBox] = field(default_factory=list)
    smoothing_zones: list[ZoneSmoothingSpec] = field(default_factory=list)
    domain_height_full: float | None = None
    domain_y_top_full: float = 1.0


def _sstep(e0: float, e1: float, v):
    diff = float(e1) - float(e0)
    denom = diff if abs(diff) >= 1e-8 else (1e-8 if diff >= 0.0 else -1e-8)
    t = ((v - e0) / denom).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _sstep_np(e0, e1, v):
    diff = np.asarray(e1, dtype=np.float64) - np.asarray(e0, dtype=np.float64)
    denom = np.where(diff >= 0.0, np.maximum(diff, 1e-8), np.minimum(diff, -1e-8))
    t = np.clip((v - e0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _eval_box_mask_np(
    x_full: np.ndarray,
    y_full: np.ndarray,
    z_rel: np.ndarray,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    margin: float,
) -> np.ndarray:
    """Evaluate a C1-smooth 3D box indicator `[0, 1]` using smoothstep borders."""
    m = max(float(margin), 1e-5)
    mx = _sstep_np(x_range[0] - m, x_range[0], x_full) * (1.0 - _sstep_np(x_range[1], x_range[1] + m, x_full))
    my = _sstep_np(y_range[0] - m, y_range[0], y_full) * (1.0 - _sstep_np(y_range[1], y_range[1] + m, y_full))
    mz = _sstep_np(z_range[0] - m, z_range[0], z_rel) * (1.0 - _sstep_np(z_range[1], z_range[1] + m, z_rel))
    return (mx * my * mz).astype(np.float32)


def _eval_box_mask_torch(
    x_full,
    y_full,
    z_rel,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    margin: float,
):
    m = max(float(margin), 1e-5)
    mx = _sstep(x_range[0] - m, x_range[0], x_full) * (1.0 - _sstep(x_range[1], x_range[1] + m, x_full))
    my = _sstep(y_range[0] - m, y_range[0], y_full) * (1.0 - _sstep(y_range[1], y_range[1] + m, y_full))
    mz = _sstep(z_range[0] - m, z_range[0], z_rel) * (1.0 - _sstep(z_range[1], z_range[1] + m, z_rel))
    return mx * my * mz


# ---------------------------------------------------------------------------
# Exact 4-View Discrete Rotation (`det = +1`) on Direct3D-V3 Patch Grid
# ---------------------------------------------------------------------------

def build_view_rotation(coords_front, grid: int, view: str):
    """Build exact det=+1 discrete patch rotation + lexicographic sort permutation for view."""
    is_np = isinstance(coords_front, np.ndarray)
    n = len(coords_front)
    if is_np:
        if view == "front":
            ident = np.arange(n, dtype=np.int64)
            return np.ascontiguousarray(coords_front), ident, ident
        px, py, pz = coords_front[:, 1], coords_front[:, 2], coords_front[:, 3]
        coords_rot = coords_front.copy()
        if view == "back":
            coords_rot[:, 1] = (grid - 1) - px
            coords_rot[:, 2] = py
            coords_rot[:, 3] = (grid - 1) - pz
        elif view == "left":
            coords_rot[:, 1] = (grid - 1) - pz
            coords_rot[:, 2] = py
            coords_rot[:, 3] = px
        elif view == "right":
            coords_rot[:, 1] = pz
            coords_rot[:, 2] = py
            coords_rot[:, 3] = (grid - 1) - px
        else:
            raise ValueError(f"Unsupported view {view!r}; expected one of {VIEW_ORDER}")
        if n > 0 and (int(coords_rot[:, 1:].min()) < 0 or int(coords_rot[:, 1:].max()) >= grid):
            raise RuntimeError(f"Out-of-bounds rotated patch coords for view={view} at grid={grid}")
        keys = (
            coords_rot[:, 1].astype(np.int64) * (grid * grid)
            + coords_rot[:, 2].astype(np.int64) * grid
            + coords_rot[:, 3].astype(np.int64)
        )
        perm = np.argsort(keys, kind="mergesort")
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(n, dtype=np.int64)
        return np.ascontiguousarray(coords_rot[perm]), perm, inv_perm

    import torch

    device = coords_front.device
    if view == "front":
        ident = torch.arange(n, device=device, dtype=torch.long)
        return coords_front.contiguous(), ident, ident

    px = coords_front[:, 1]
    py = coords_front[:, 2]
    pz = coords_front[:, 3]
    coords_rot = coords_front.clone()
    if view == "back":
        coords_rot[:, 1] = (grid - 1) - px
        coords_rot[:, 2] = py
        coords_rot[:, 3] = (grid - 1) - pz
    elif view == "left":
        coords_rot[:, 1] = (grid - 1) - pz
        coords_rot[:, 2] = py
        coords_rot[:, 3] = px
    elif view == "right":
        coords_rot[:, 1] = pz
        coords_rot[:, 2] = py
        coords_rot[:, 3] = (grid - 1) - px
    else:
        raise ValueError(f"Unsupported view {view!r}; expected one of {VIEW_ORDER}")

    if n > 0 and (int(coords_rot[:, 1:].min()) < 0 or int(coords_rot[:, 1:].max()) >= grid):
        raise RuntimeError(f"Out-of-bounds rotated patch coords for view={view} at grid={grid}")

    keys = (
        coords_rot[:, 1].long() * (grid * grid)
        + coords_rot[:, 2].long() * grid
        + coords_rot[:, 3].long()
    )
    perm = torch.argsort(keys)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(n, device=device, dtype=torch.long)
    return coords_rot[perm].contiguous(), perm, inv_perm


def rotate_latents_forward(latents, perm, patch_size: int, view: str):
    """Rotate canonical latents `[N, ps^3]` into camera-space view coordinates."""
    if view == "front":
        return latents
    n = latents.shape[0]
    if isinstance(latents, np.ndarray):
        t = latents.reshape(n, patch_size, patch_size, patch_size)
        if view == "back":
            t = np.flip(t, axis=(1, 3))
        elif view == "left":
            t = np.flip(np.swapaxes(t, 1, 3), axis=1)
        elif view == "right":
            t = np.flip(np.swapaxes(t, 1, 3), axis=3)
        else:
            raise ValueError(f"Unsupported view {view!r}")
        return np.ascontiguousarray(t.reshape(n, patch_size**3)[perm])

    t = latents.view(n, patch_size, patch_size, patch_size)
    if view == "back":
        t = t.flip(1, 3)
    elif view == "left":
        t = t.transpose(1, 3).flip(1)
    elif view == "right":
        t = t.transpose(1, 3).flip(3)
    else:
        raise ValueError(f"Unsupported view {view!r}")
    return t.reshape(n, patch_size**3)[perm].contiguous()


def rotate_latents_inverse(latents_rot, inv_perm, patch_size: int, view: str):
    """Invert camera-space rotated latents `[N, ps^3]` back into canonical front coordinates."""
    if view == "front":
        return latents_rot
    n = latents_rot.shape[0]
    if isinstance(latents_rot, np.ndarray):
        t = latents_rot[inv_perm].reshape(n, patch_size, patch_size, patch_size)
        if view == "back":
            t = np.flip(t, axis=(1, 3))
        elif view == "left":
            t = np.swapaxes(np.flip(t, axis=1), 1, 3)
        elif view == "right":
            t = np.swapaxes(np.flip(t, axis=3), 1, 3)
        else:
            raise ValueError(f"Unsupported view {view!r}")
        return np.ascontiguousarray(t.reshape(n, patch_size**3))

    t = latents_rot[inv_perm].view(n, patch_size, patch_size, patch_size)
    if view == "back":
        t = t.flip(1, 3)
    elif view == "left":
        t = t.flip(1).transpose(1, 3)
    elif view == "right":
        t = t.flip(3).transpose(1, 3)
    else:
        raise ValueError(f"Unsupported view {view!r}")
    return t.reshape(n, patch_size**3).contiguous()


def verify_torch_rotations_selfcheck(device) -> None:
    """Fast (<5ms) PyTorch self-check verifying all 4 discrete rotations are lossless."""
    import torch

    stage, ps, grid = 512, 8, 64
    target_w = np.array([0.12, 0.40, 0.05], dtype=np.float64)
    gx, gy, gz = [int(round((c + 0.5) * (stage - 1))) for c in target_w]
    w_exact = np.array([gx, gy, gz], dtype=np.float64) / float(stage - 1) - 0.5
    px_m, dx_m = gx // ps, gx % ps
    py_m, dy_m = gy // ps, gy % ps
    pz_m, dz_m = gz // ps, gz % ps
    patches = sorted({(0, 10, 20, 30), (0, px_m, py_m, pz_m), (0, 63 - px_m, py_m, 63 - pz_m)})
    coords = torch.tensor(patches, dtype=torch.int32, device=device)
    lat = torch.randn((len(patches), ps**3), device=device, dtype=torch.float32)
    marker = torch.zeros_like(lat)
    marker[patches.index((0, px_m, py_m, pz_m)), dx_m * ps * ps + dy_m * ps + dz_m] = 1.0
    exp_map = {
        "front": (w_exact[0], w_exact[1], w_exact[2]),
        "back": (-w_exact[0], w_exact[1], -w_exact[2]),
        "left": (-w_exact[2], w_exact[1], w_exact[0]),
        "right": (w_exact[2], w_exact[1], -w_exact[0]),
    }
    for view in VIEW_ORDER:
        cv, pv, ipv = build_view_rotation(coords, grid, view)
        rec = rotate_latents_inverse(rotate_latents_forward(lat, pv, ps, view), ipv, ps, view)
        assert float((rec - lat).abs().max().item()) == 0.0, f"Torch roundtrip failed for {view}"
        m_rot = rotate_latents_forward(marker, pv, ps, view)
        nz = torch.nonzero(m_rot == 1.0, as_tuple=False)
        assert len(nz) == 1
        rp, rv = int(nz[0, 0].item()), int(nz[0, 1].item())
        rx, ry, rz = int(cv[rp, 1].item()), int(cv[rp, 2].item()), int(cv[rp, 3].item())
        rdx, rdy, rdz = rv // (ps * ps), (rv // ps) % ps, rv % ps
        rw = ((rx * ps + rdx) / (stage - 1) - 0.5, (ry * ps + rdy) / (stage - 1) - 0.5, (rz * ps + rdz) / (stage - 1) - 0.5)
        for a, b in zip(rw, exp_map[view]):
            assert abs(a - b) < 1e-6, f"Torch marker mismatch for {view}: {rw} vs {exp_map[view]}"


# ---------------------------------------------------------------------------
# Generic Voxel World Frame & Orthographic Visibility / Crevice Weighting
# ---------------------------------------------------------------------------

def normalize_surface_to_voxel_world(verts_np: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Map raw mesh vertices into Direct3D-V3's canonical `[-0.5, 0.5]^3` voxel world frame."""
    v = np.asarray(verts_np, dtype=np.float32)
    lo = v.min(axis=0)
    hi = v.max(axis=0)
    extent = max(float((hi - lo).max()), 1e-6)
    return (v - 0.5 * (lo + hi)) * (0.95 * float(scale) / extent)


def compute_physical_frame_meta(
    v_norm: np.ndarray,
    *,
    domain_height_full: float = 1.0,
    domain_y_top_full: float = 1.0,
    anchor_z_box: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Compute invariant linear map from normalized voxel space to `[0, 1]` height-anchored coords."""
    y_min_n = float(v_norm[:, 1].min())
    y_max_n = float(v_norm[:, 1].max())
    y_span_n = max(y_max_n - y_min_n, 1e-6)
    s_to_full = float(domain_height_full) / y_span_n

    y_full = float(domain_y_top_full) - (y_max_n - v_norm[:, 1]) * s_to_full
    x_center_n = 0.5 * float(v_norm[:, 0].min() + v_norm[:, 0].max())
    x_full = (v_norm[:, 0] - x_center_n) * s_to_full

    z_mid_n = 0.5 * float(v_norm[:, 2].min() + v_norm[:, 2].max())
    if anchor_z_box is not None:
        xr = anchor_z_box.get("x_range", (-1.0, 1.0))
        yr = anchor_z_box.get("y_range", (0.0, 1.0))
        core = (y_full >= yr[0]) & (y_full <= yr[1]) & (x_full >= xr[0]) & (x_full <= xr[1])
        if int(core.sum()) >= 32:
            z_mid_n = 0.5 * float(v_norm[core, 2].min() + v_norm[core, 2].max())

    return {
        "domain_height_full": float(domain_height_full),
        "domain_y_top_full": float(domain_y_top_full),
        "y_min_n": y_min_n,
        "y_max_n": y_max_n,
        "y_span_n": y_span_n,
        "s_to_full": s_to_full,
        "x_center_n": x_center_n,
        "z_anchor_mid_n": z_mid_n,
    }


def to_physical_coords_np(
    x_n: np.ndarray,
    y_n: np.ndarray,
    z_n: np.ndarray,
    frame: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = frame["s_to_full"]
    y_full = frame["domain_y_top_full"] - (frame["y_max_n"] - y_n) * s
    x_full = (x_n - frame["x_center_n"]) * s
    z_rel = (z_n - frame["z_anchor_mid_n"]) * s
    return x_full, y_full, z_rel


def _rasterize_ortho_zbuf(
    v_norm: np.ndarray,
    faces_np: np.ndarray,
    view: str,
    zres: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hole-free orthographic Z-buffer in `[-0.5, +0.5]^2` returning max depth toward camera."""
    if view == "front":
        u, vv, depth = v_norm[:, 0], v_norm[:, 1], v_norm[:, 2]
    elif view == "back":
        u, vv, depth = -v_norm[:, 0], v_norm[:, 1], -v_norm[:, 2]
    elif view == "left":
        u, vv, depth = -v_norm[:, 2], v_norm[:, 1], v_norm[:, 0]
    elif view == "right":
        u, vv, depth = v_norm[:, 2], v_norm[:, 1], -v_norm[:, 0]
    else:
        raise ValueError(f"Unsupported view {view!r}")

    scale = float(zres - 1)
    px = np.clip((u + 0.5) * scale, 0.0, scale).astype(np.float32)
    py = np.clip((vv + 0.5) * scale, 0.0, scale).astype(np.float32)
    depth_f = depth.astype(np.float32)

    zbuf = np.full(zres * zres, -1e9, dtype=np.float32)
    f = faces_np
    px0, px1, px2 = px[f[:, 0]], px[f[:, 1]], px[f[:, 2]]
    py0, py1, py2 = py[f[:, 0]], py[f[:, 1]], py[f[:, 2]]
    d0, d1, d2 = depth_f[f[:, 0]], depth_f[f[:, 1]], depth_f[f[:, 2]]

    def _splat_bary(w0: float, w1: float, w2: float, idx_mask=None) -> None:
        if idx_mask is None:
            sx = np.rint(w0 * px0 + w1 * px1 + w2 * px2).astype(np.int32)
            sy = np.rint(w0 * py0 + w1 * py1 + w2 * py2).astype(np.int32)
            sd = w0 * d0 + w1 * d1 + w2 * d2
        else:
            sx = np.rint(w0 * px0[idx_mask] + w1 * px1[idx_mask] + w2 * px2[idx_mask]).astype(np.int32)
            sy = np.rint(w0 * py0[idx_mask] + w1 * py1[idx_mask] + w2 * py2[idx_mask]).astype(np.int32)
            sd = w0 * d0[idx_mask] + w1 * d1[idx_mask] + w2 * d2[idx_mask]
        np.clip(sx, 0, zres - 1, out=sx)
        np.clip(sy, 0, zres - 1, out=sy)
        np.maximum.at(zbuf, sy * zres + sx, sd)

    for b0, b1, b2 in (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    ):
        _splat_bary(b0, b1, b2)

    min_x = np.floor(np.minimum(np.minimum(px0, px1), px2)).astype(np.int32)
    max_x = np.ceil(np.maximum(np.maximum(px0, px1), px2)).astype(np.int32)
    min_y = np.floor(np.minimum(np.minimum(py0, py1), py2)).astype(np.int32)
    max_y = np.ceil(np.maximum(np.maximum(py0, py1), py2)).astype(np.int32)
    span_px = np.maximum(max_x - min_x, max_y - min_y)

    med = np.flatnonzero((span_px >= 2) & (span_px <= 5))
    if len(med) > 0:
        for i in range(5):
            for j in range(5 - i):
                w0 = i / 4.0
                w1 = j / 4.0
                w2 = 1.0 - w0 - w1
                if (w0, w1, w2) not in (
                    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                    (0.5, 0.5, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5),
                ):
                    _splat_bary(w0, w1, w2, med)

    big = np.flatnonzero(span_px > 5)
    for ti in big:
        x0, x1 = max(0, int(min_x[ti])), min(zres - 1, int(max_x[ti]))
        y0, y1 = max(0, int(min_y[ti])), min(zres - 1, int(max_y[ti]))
        ax, ay, ad = float(px0[ti]), float(py0[ti]), float(d0[ti])
        bx, by, bd = float(px1[ti]), float(py1[ti]), float(d1[ti])
        cx, cy, cd = float(px2[ti]), float(py2[ti]), float(d2[ti])
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denom) < 1e-8:
            continue
        inv_den = 1.0 / denom
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1, dtype=np.float32), np.arange(y0, y1 + 1, dtype=np.float32))
        w0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) * inv_den
        w1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) * inv_den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-3) & (w1 >= -1e-3) & (w2 >= -1e-3)
        if np.any(inside):
            ix = gx[inside].astype(np.int32)
            iy = gy[inside].astype(np.int32)
            zd = (w0[inside] * ad + w1[inside] * bd + w2[inside] * cd).astype(np.float32)
            np.maximum.at(zbuf, iy * zres + ix, zd)

    return zbuf.reshape(zres, zres), px, py, depth_f


def compute_surface_visibility_and_weights(
    verts_np: np.ndarray,
    faces_np: np.ndarray,
    *,
    v_scale: float = 1.0,
    domain_height_full: float = 1.0,
    domain_y_top_full: float = 1.0,
    zres: int = 512,
    tol_lo_full: float = 0.012,
    tol_hi_full: float = 0.022,
    cliff_lo_full: float = 0.014,
    cliff_hi_full: float = 0.036,
    smooth_sigma_full: float = 0.006,
    roi_boxes: Sequence[ROISovereigntyBox] = (),
    smoothing_zones: Sequence[ZoneSmoothingSpec] = (),
) -> dict[str, Any]:
    """Compute generic 4-view surface visibility, crevice suppression, and optional Layer-2 ROI locks."""
    t0 = time.monotonic()
    v_norm = normalize_surface_to_voxel_world(verts_np, scale=v_scale)
    f = np.asarray(faces_np, dtype=np.int64)
    n_v = len(v_norm)

    anchor_z_box = next((b.anchor_z_box for b in roi_boxes if b.anchor_z_box is not None), None)
    frame = compute_physical_frame_meta(
        v_norm,
        domain_height_full=domain_height_full,
        domain_y_top_full=domain_y_top_full,
        anchor_z_box=anchor_z_box,
    )
    s_to_full = frame["s_to_full"]
    x_full, y_full, z_rel = to_physical_coords_np(v_norm[:, 0], v_norm[:, 1], v_norm[:, 2], frame)

    # 1. Area-weighted outward vertex normals with signed-volume orientation guard
    v0 = v_norm[f[:, 0]].astype(np.float64)
    v1 = v_norm[f[:, 1]].astype(np.float64)
    v2 = v_norm[f[:, 2]].astype(np.float64)
    fcross = np.cross(v1 - v0, v2 - v0)
    signed_vol = float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)
    flipped_normals = False
    if signed_vol < 0.0:
        fcross = -fcross
        flipped_normals = True

    vn = np.zeros((n_v, 3), dtype=np.float64)
    for c in range(3):
        fc_c = fcross[:, c]
        vn[:, c] = (
            np.bincount(f[:, 0], weights=fc_c, minlength=n_v)
            + np.bincount(f[:, 1], weights=fc_c, minlength=n_v)
            + np.bincount(f[:, 2], weights=fc_c, minlength=n_v)
        )
    vn_len = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = (vn / np.maximum(vn_len, 1e-12)).astype(np.float32)

    e01 = np.linalg.norm(v_norm[f[:, 1]] - v_norm[f[:, 0]], axis=1)
    mean_edge = max(float(np.median(e01)) if len(e01) else 0.002, 1e-4)
    mean_edge_full = mean_edge * s_to_full

    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    adj = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_v, n_v))
    adj.sum_duplicates()
    adj.data[:] = 1.0
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv_deg = np.where(deg > 0, 1.0 / np.maximum(deg, 1.0), 0.0).astype(np.float32)
    adj_norm = adj.multiply(inv_deg[:, None]).tocsr()

    n_norm_iters = int(np.clip(round(2.0 * (0.0035 / max(mean_edge_full, 1e-4)) ** 2), 2, 10))
    for _ in range(n_norm_iters):
        vn = 0.5 * vn + 0.5 * (adj_norm @ vn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-8)

    # 2. Orthographic Z-buffer visibility + steep occlusion depth-cliff detector
    px_size_full = (1.0 / float(zres - 1)) * s_to_full
    cliff_rad_px = int(np.clip(round(0.0055 / max(px_size_full, 1e-5)), 2, 7))
    cliff_k = 2 * cliff_rad_px + 1

    vis = np.zeros((n_v, 4), dtype=np.float32)
    cliff_keep = np.ones((n_v, 4), dtype=np.float32)
    for vi, view in enumerate(VIEW_ORDER):
        zbuf, px, py, depth_f = _rasterize_ortho_zbuf(v_norm, f, view, zres=zres)
        ix = np.clip(np.rint(px).astype(np.int32), 0, zres - 1)
        iy = np.clip(np.rint(py).astype(np.int32), 0, zres - 1)
        z_front = zbuf[iy, ix]
        occl_full = (z_front - depth_f) * s_to_full
        vis[:, vi] = (1.0 - _sstep_np(tol_lo_full, tol_hi_full, occl_full)).astype(np.float32)

        if vi in (2, 3):
            zbuf_max = maximum_filter(zbuf, size=cliff_k, mode="nearest")
            z_near_max = zbuf_max[iy, ix]
            cliff_full = (z_near_max - depth_f) * s_to_full
            cliff_keep[:, vi] = (1.0 - _sstep_np(cliff_lo_full, cliff_hi_full, cliff_full)).astype(np.float32)

    # 3. Overhang crevice suppression (damps side views in tight crevices -> C1 front/back fallback)
    vis_total = vis.sum(axis=1)
    keep_vis_tot = _sstep_np(1.15, 1.85, vis_total).astype(np.float32)
    down_overhang = (
        (1.0 - _sstep_np(-0.58, -0.22, vn[:, 1])) * (1.0 - _sstep_np(1.85, 2.50, vis_total))
    ).astype(np.float32)
    crevice_keep = (keep_vis_tot * (1.0 - 0.92 * down_overhang)).astype(np.float32)

    side_keep_left = (cliff_keep[:, 2] * crevice_keep).astype(np.float32)
    side_keep_right = (cliff_keep[:, 3] * crevice_keep).astype(np.float32)

    dirs = np.asarray(VIEW_DIRS, dtype=np.float32)
    ndot = np.clip(vn @ dirs.T, 0.0, 1.0)
    vis_eff = (vis * _sstep_np(0.30, 0.60, ndot)).astype(np.float32)
    vis_eff[:, 2] *= side_keep_left
    vis_eff[:, 3] *= side_keep_right
    raw_w = vis_eff * (ndot * ndot)

    n_w_iters = int(np.clip(round(2.0 * (smooth_sigma_full / max(mean_edge_full, 1e-4)) ** 2), 4, 24))
    raw_w_smooth = raw_w.copy()
    for _ in range(n_w_iters):
        raw_w_smooth = 0.5 * raw_w_smooth + 0.5 * (adj_norm @ raw_w_smooth)

    post_keep_l = side_keep_left.copy()
    post_keep_r = side_keep_right.copy()
    for _ in range(2):
        post_keep_l = 0.5 * post_keep_l + 0.5 * (adj_norm @ post_keep_l)
        post_keep_r = 0.5 * post_keep_r + 0.5 * (adj_norm @ post_keep_r)
    raw_w_smooth[:, 2] *= post_keep_l
    raw_w_smooth[:, 3] *= post_keep_r

    # 4. Layer-2 Optional Zone Geodesic Transition Smoothing (default empty)
    for zspec in smoothing_zones:
        m_zone = _eval_box_mask_np(
            x_full, y_full, z_rel,
            x_range=zspec.x_range,
            y_range=zspec.y_range,
            z_range=zspec.z_range,
            margin=zspec.margin,
        )
        if float(m_zone.max()) > 1e-3 and zspec.view_indices:
            raw_w_extra = raw_w_smooth.copy()
            n_extra = max(8, int(round(zspec.extra_iter_mult * n_w_iters)))
            for _ in range(n_extra):
                raw_w_extra = 0.5 * raw_w_extra + 0.5 * (adj_norm @ raw_w_extra)
            blend = (zspec.blend_weight * m_zone)[:, None]
            cols_idx = list(zspec.view_indices)
            raw_w_smooth[:, cols_idx] = (1.0 - blend) * raw_w_smooth[:, cols_idx] + blend * raw_w_extra[:, cols_idx]

    # 5. Layer-2 Optional ROI Sovereignty Locks (default empty)
    roi_surf_masks: dict[str, np.ndarray] = {}
    for box in roi_boxes:
        t_vi = VIEW_TO_INDEX[box.target_view]
        t_dir = dirs[t_vi]
        ndot_t = vn @ t_dir
        m_spatial = _eval_box_mask_np(
            x_full, y_full, z_rel,
            x_range=box.x_range,
            y_range=box.y_range,
            z_range=box.z_range,
            margin=box.margin,
        )
        m_surf = (m_spatial * _sstep_np(box.normal_gate_range[0], box.normal_gate_range[1], ndot_t)).astype(np.float32)
        roi_surf_masks[box.name] = m_surf

        # Orthogonal views relative to target_view: if front/back (0,1), ortho are (2,3); else (0,1)
        ortho_cols = (2, 3) if t_vi in (0, 1) else (0, 1)
        for oc in ortho_cols:
            raw_w_smooth[:, oc] *= (1.0 - 0.99 * m_surf)
        raw_w_smooth[:, t_vi] = np.maximum(raw_w_smooth[:, t_vi], box.min_surface_weight * m_surf)

    z_mid_surf = float(0.5 * (v_norm[:, 2].min() + v_norm[:, 2].max()))
    fb_band_n = 0.018 / s_to_full
    w_fb_f = _sstep_np(z_mid_surf - fb_band_n, z_mid_surf + fb_band_n, v_norm[:, 2]).astype(np.float32)
    w_fb_surf = np.stack([w_fb_f, 1.0 - w_fb_f, np.zeros_like(w_fb_f), np.zeros_like(w_fb_f)], axis=-1)
    s_surf = raw_w_smooth.sum(axis=1, keepdims=True)
    gate_surf = _sstep_np(0.002, 0.055, s_surf).astype(np.float32)
    w_surf = gate_surf * (raw_w_smooth / np.maximum(s_surf, 1e-6)) + (1.0 - gate_surf) * w_fb_surf

    for box in roi_boxes:
        t_vi = VIEW_TO_INDEX[box.target_view]
        ortho_cols = (2, 3) if t_vi in (0, 1) else (0, 1)
        m_surf = roi_surf_masks[box.name]
        for oc in ortho_cols:
            w_surf[:, oc] *= (1.0 - box.suppress_ortho_views * m_surf)
    w_surf /= np.maximum(w_surf.sum(axis=1, keepdims=True), 1e-6)

    own_side_keep = np.where(x_full >= 0.0, side_keep_left, side_keep_right).astype(np.float32)
    crevice_damp_score = (1.0 - own_side_keep).astype(np.float32)

    return {
        "v_norm": v_norm,
        "vn": vn,
        "vis": vis,
        "raw_w": raw_w_smooth,
        "w_surf": w_surf,
        "frame": frame,
        "x_full": x_full,
        "y_full": y_full,
        "z_rel": z_rel,
        "roi_surf_masks": roi_surf_masks,
        "crevice_damp_score": crevice_damp_score,
        "meta": {
            "signed_vol": round(signed_vol, 6),
            "flipped_normals": flipped_normals,
            "mean_edge_len": round(mean_edge, 6),
            "mean_edge_full": round(mean_edge_full, 6),
            "n_norm_iters": n_norm_iters,
            "n_weight_iters": n_w_iters,
            "zres": zres,
            "cliff_k": cliff_k,
            "domain_height_full": round(float(domain_height_full), 5),
            "s_to_full": round(float(s_to_full), 5),
            "roi_boxes": [b.name for b in roi_boxes],
            "elapsed_s": round(time.monotonic() - t0, 3),
        },
    }


def _compute_centered_fb_fallback_np(p_xyz: np.ndarray, stage: int, patch_size: int):
    """Centered per-Y front/back smoothstep fallback in pure NumPy."""
    grid = stage // patch_size
    py_idx = p_xyz[:, 1].astype(np.int64)
    pz_idx = p_xyz[:, 2].astype(np.int64)

    pz_world = (pz_idx.astype(np.float32) * patch_size + 0.5 * (patch_size - 1)) / float(stage - 1) - 0.5
    z_min = np.full((grid,), np.inf, dtype=np.float32)
    z_max = np.full((grid,), -np.inf, dtype=np.float32)
    np.minimum.at(z_min, py_idx, pz_world)
    np.maximum.at(z_max, py_idx, pz_world)

    valid = np.isfinite(z_min) & np.isfinite(z_max)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) == 0:
        raise RuntimeError(f"No valid Y slices at stage {stage}")

    all_py = np.arange(grid, dtype=np.int64)
    nearest_pos = np.argmin(np.abs(all_py[:, None] - valid_idx[None, :]), axis=1)
    nearest_py = valid_idx[nearest_pos]
    z_min = np.where(valid, z_min, z_min[nearest_py])
    z_max = np.where(valid, z_max, z_max[nearest_py])

    patch_world = float(patch_size) / float(stage - 1)
    z_center = 0.5 * (z_min + z_max)
    z_span = np.maximum(z_max - z_min + patch_world, patch_world)

    rad = max(1, grid // 32)
    gx = np.arange(-rad, rad + 1, dtype=np.float32)
    sigma = max(1.0, 0.5 * rad)
    kernel = np.exp(-0.5 * (gx / sigma) ** 2)
    kernel = kernel / kernel.sum()

    def _smooth_1d(arr: np.ndarray) -> np.ndarray:
        padded = np.pad(arr, (rad, rad), mode="edge")
        return np.convolve(padded, kernel, mode="valid").astype(np.float32)

    z_mid_py = _smooth_1d(z_center)
    band_py = np.clip(0.18 * _smooth_1d(z_span), 0.014, 0.045).astype(np.float32)

    i_vox = np.arange(patch_size**3, dtype=np.int64)
    dx = (i_vox // (patch_size * patch_size)).astype(np.float32)
    dy = ((i_vox // patch_size) % patch_size).astype(np.float32)
    dz = (i_vox % patch_size).astype(np.float32)

    x_vox = p_xyz[:, 0:1].astype(np.float32) * patch_size + dx[None, :]
    y_vox = p_xyz[:, 1:2].astype(np.float32) * patch_size + dy[None, :]
    z_vox = p_xyz[:, 2:3].astype(np.float32) * patch_size + dz[None, :]

    x_world = x_vox / float(stage - 1) - 0.5
    y_world = y_vox / float(stage - 1) - 0.5
    z_world = z_vox / float(stage - 1) - 0.5

    z_mid = z_mid_py[py_idx][:, None]
    band = band_py[py_idx][:, None]
    w_front = _sstep_np(z_mid - band, z_mid + band, z_world)
    return w_front, x_world, y_world, z_world


def compute_4view_voxel_weights(
    verts,
    faces,
    coords_front,
    stage: int,
    patch_size: int,
    *,
    v_scale: float = 1.0,
    domain_height_full: float = 1.0,
    domain_y_top_full: float = 1.0,
    zres: int = 512,
    roi_boxes: Sequence[ROISovereigntyBox] = (),
    smoothing_zones: Sequence[ZoneSmoothingSpec] = (),
    surface_cache: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Interpolate surface visibility weights onto `[4, N_patches, patch_size^3]` active voxels."""
    from scipy.spatial import cKDTree

    t0 = time.monotonic()
    is_np = isinstance(coords_front, np.ndarray)
    verts_np = verts if isinstance(verts, np.ndarray) else verts.detach().cpu().numpy()
    faces_np = faces if isinstance(faces, np.ndarray) else faces.detach().cpu().numpy()
    p_xyz = coords_front[:, 1:4] if is_np else coords_front[:, 1:4].detach().cpu().numpy()
    p_xyz = p_xyz.astype(np.int64, copy=False)

    if surface_cache is None:
        surf = compute_surface_visibility_and_weights(
            verts_np,
            faces_np,
            v_scale=v_scale,
            domain_height_full=domain_height_full,
            domain_y_top_full=domain_y_top_full,
            zres=zres,
            roi_boxes=roi_boxes,
            smoothing_zones=smoothing_zones,
        )
    else:
        surf = surface_cache

    v_norm = surf["v_norm"]
    raw_w_surf = surf["raw_w"]
    frame = surf["frame"]

    sub = 2
    step_vox = patch_size // sub
    off = np.arange(sub + 1, dtype=np.int64) * step_vox
    ox, oy, oz = np.meshgrid(off, off, off, indexing="ij")
    ox_f, oy_f, oz_f = ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)

    vx_lat = (p_xyz[:, 0:1] * patch_size + ox_f[None, :]).reshape(-1)
    vy_lat = (p_xyz[:, 1:2] * patch_size + oy_f[None, :]).reshape(-1)
    vz_lat = (p_xyz[:, 2:3] * patch_size + oz_f[None, :]).reshape(-1)
    lat_keys = (vx_lat << 24) | (vy_lat << 12) | vz_lat
    uniq_keys, inv_lat = np.unique(lat_keys, return_inverse=True)

    ux = ((uniq_keys >> 24) & 0xFFF).astype(np.float32) / float(stage - 1) - 0.5
    uy = ((uniq_keys >> 12) & 0xFFF).astype(np.float32) / float(stage - 1) - 0.5
    uz = (uniq_keys & 0xFFF).astype(np.float32) / float(stage - 1) - 0.5
    lat_pts = np.stack([ux, uy, uz], axis=1)

    tree = cKDTree(v_norm.astype(np.float64))
    dists, nn_idx = tree.query(lat_pts.astype(np.float64), k=3, workers=-1)
    inv_d = 1.0 / np.maximum(dists, 1e-5)
    inv_d /= inv_d.sum(axis=1, keepdims=True)
    uniq_raw_w = np.sum(raw_w_surf[nn_idx] * inv_d[:, :, None], axis=1).astype(np.float32)

    inv_grid = inv_lat.reshape(len(p_xyz), sub + 1, sub + 1, sub + 1)

    if is_np:
        c_w = uniq_raw_w[inv_grid]
        d_lin = np.arange(patch_size, dtype=np.float32)
        gx_v, gy_v, gz_v = np.meshgrid(d_lin, d_lin, d_lin, indexing="ij")
        fx = gx_v.reshape(-1) / float(step_vox)
        fy = gy_v.reshape(-1) / float(step_vox)
        fz = gz_v.reshape(-1) / float(step_vox)
        ix0 = np.minimum(np.floor(fx).astype(np.int64), sub - 1)
        iy0 = np.minimum(np.floor(fy).astype(np.int64), sub - 1)
        iz0 = np.minimum(np.floor(fz).astype(np.int64), sub - 1)
        tx = (fx - ix0.astype(np.float32))[None, :, None]
        ty = (fy - iy0.astype(np.float32))[None, :, None]
        tz = (fz - iz0.astype(np.float32))[None, :, None]

        c00 = c_w[:, ix0, iy0, iz0] * (1.0 - tz) + c_w[:, ix0, iy0, iz0 + 1] * tz
        c01 = c_w[:, ix0, iy0 + 1, iz0] * (1.0 - tz) + c_w[:, ix0, iy0 + 1, iz0 + 1] * tz
        c10 = c_w[:, ix0 + 1, iy0, iz0] * (1.0 - tz) + c_w[:, ix0 + 1, iy0, iz0 + 1] * tz
        c11 = c_w[:, ix0 + 1, iy0 + 1, iz0] * (1.0 - tz) + c_w[:, ix0 + 1, iy0 + 1, iz0 + 1] * tz
        c0 = c00 * (1.0 - ty) + c01 * ty
        c1 = c10 * (1.0 - ty) + c11 * ty
        raw_w_vox = np.maximum(c0 * (1.0 - tx) + c1 * tx, 0.0).astype(np.float32)

        w_fb_front, x_world, y_world, z_world = _compute_centered_fb_fallback_np(p_xyz, stage, patch_size)
        zeros = np.zeros_like(w_fb_front)
        w_fb = np.stack([w_fb_front, 1.0 - w_fb_front, zeros, zeros], axis=-1)
        w_sum = raw_w_vox.sum(axis=-1, keepdims=True)
        gate = _sstep_np(0.002, 0.055, w_sum).astype(np.float32)
        w_vox = gate * (raw_w_vox / np.maximum(w_sum, 1e-6)) + (1.0 - gate) * w_fb

        xf_v, yf_v, zr_v = to_physical_coords_np(x_world, y_world, z_world, frame)
        roi_stats: dict[str, Any] = {}
        for box in roi_boxes:
            t_vi = VIEW_TO_INDEX[box.target_view]
            ortho_cols = (2, 3) if t_vi in (0, 1) else (0, 1)
            m_vox = _eval_box_mask_np(
                xf_v, yf_v, zr_v,
                x_range=box.x_range,
                y_range=box.y_range,
                z_range=box.z_range,
                margin=box.margin,
            )
            for oc in ortho_cols:
                w_vox[..., oc] *= (1.0 - box.suppress_ortho_views * m_vox)
            w_vox[..., t_vi] = np.maximum(w_vox[..., t_vi], box.min_target_weight * m_vox)
            roi_mask = m_vox > 0.5
            roi_stats[box.name] = {
                "voxel_count": int(roi_mask.sum()),
                f"w_{box.target_view}_mean": round(float(w_vox[..., t_vi][roi_mask].mean()), 4) if np.any(roi_mask) else None,
            }

        w_vox /= np.maximum(w_vox.sum(axis=-1, keepdims=True), 1e-6)
        w_views = np.ascontiguousarray(np.transpose(w_vox, (2, 0, 1)))
        sum_err = float(np.abs(w_views.sum(axis=0) - 1.0).max())
        fallback_frac = float((w_sum.squeeze(-1) < 0.002).mean())
        global_means = {vn_lbl: round(float(w_views[vi].mean()), 4) for vi, vn_lbl in enumerate(VIEW_ORDER)}
    else:
        import torch

        device = coords_front.device
        inv_t = torch.from_numpy(inv_grid).to(device=device, dtype=torch.long)
        uniq_t = torch.from_numpy(uniq_raw_w).to(device=device, dtype=torch.float32)
        c_w = uniq_t[inv_t]

        d_lin = torch.arange(patch_size, device=device, dtype=torch.float32)
        gx_v, gy_v, gz_v = torch.meshgrid(d_lin, d_lin, d_lin, indexing="ij")
        fx = gx_v.reshape(-1) / float(step_vox)
        fy = gy_v.reshape(-1) / float(step_vox)
        fz = gz_v.reshape(-1) / float(step_vox)

        ix0 = fx.floor().long().clamp_max(sub - 1)
        iy0 = fy.floor().long().clamp_max(sub - 1)
        iz0 = fz.floor().long().clamp_max(sub - 1)
        tx = (fx - ix0.float())[None, :, None]
        ty = (fy - iy0.float())[None, :, None]
        tz = (fz - iz0.float())[None, :, None]

        c00 = c_w[:, ix0, iy0, iz0] * (1.0 - tz) + c_w[:, ix0, iy0, iz0 + 1] * tz
        c01 = c_w[:, ix0, iy0 + 1, iz0] * (1.0 - tz) + c_w[:, ix0, iy0 + 1, iz0 + 1] * tz
        c10 = c_w[:, ix0 + 1, iy0, iz0] * (1.0 - tz) + c_w[:, ix0 + 1, iy0, iz0 + 1] * tz
        c11 = c_w[:, ix0 + 1, iy0 + 1, iz0] * (1.0 - tz) + c_w[:, ix0 + 1, iy0 + 1, iz0 + 1] * tz
        c0 = c00 * (1.0 - ty) + c01 * ty
        c1 = c10 * (1.0 - ty) + c11 * ty
        raw_w_vox = (c0 * (1.0 - tx) + c1 * tx).clamp_min(0.0)

        w_fb_np, xw_np, yw_np, zw_np = _compute_centered_fb_fallback_np(p_xyz, stage, patch_size)
        w_fb_front = torch.from_numpy(w_fb_np).to(device=device)
        x_world = torch.from_numpy(xw_np).to(device=device)
        y_world = torch.from_numpy(yw_np).to(device=device)
        z_world = torch.from_numpy(zw_np).to(device=device)
        zeros = torch.zeros_like(w_fb_front)
        w_fb = torch.stack([w_fb_front, 1.0 - w_fb_front, zeros, zeros], dim=-1)

        w_sum = raw_w_vox.sum(dim=-1, keepdim=True)
        gate = _sstep(0.002, 0.055, w_sum)
        w_vox = gate * (raw_w_vox / w_sum.clamp_min(1e-6)) + (1.0 - gate) * w_fb

        s_f = float(frame["s_to_full"])
        yf_v = float(frame["domain_y_top_full"]) - (float(frame["y_max_n"]) - y_world) * s_f
        xf_v = (x_world - float(frame["x_center_n"])) * s_f
        zr_v = (z_world - float(frame["z_anchor_mid_n"])) * s_f

        roi_stats = {}
        for box in roi_boxes:
            t_vi = VIEW_TO_INDEX[box.target_view]
            ortho_cols = (2, 3) if t_vi in (0, 1) else (0, 1)
            m_vox = _eval_box_mask_torch(
                xf_v, yf_v, zr_v,
                x_range=box.x_range,
                y_range=box.y_range,
                z_range=box.z_range,
                margin=box.margin,
            )
            for oc in ortho_cols:
                w_vox[..., oc] = w_vox[..., oc] * (1.0 - box.suppress_ortho_views * m_vox)
            w_vox[..., t_vi] = torch.maximum(w_vox[..., t_vi], box.min_target_weight * m_vox)
            roi_mask = m_vox > 0.5
            cnt = int(roi_mask.sum().item())
            roi_stats[box.name] = {
                "voxel_count": cnt,
                f"w_{box.target_view}_mean": round(float(w_vox[..., t_vi][roi_mask].mean().item()), 4) if cnt > 0 else None,
            }

        w_vox = w_vox / w_vox.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        w_views = w_vox.permute(2, 0, 1).contiguous()
        sum_err = float((w_views.sum(dim=0) - 1.0).abs().max().item())
        fallback_frac = float((w_sum.squeeze(-1) < 0.002).float().mean().item())
        global_means = {vn_lbl: round(float(w_views[vi].mean().item()), 4) for vi, vn_lbl in enumerate(VIEW_ORDER)}

    stats = {
        "stage": int(stage),
        "lattice_sub": int(sub),
        "unique_lattice_nodes": int(len(uniq_keys)),
        "surface_meta": surf["meta"],
        "total_weight_elapsed_s": round(time.monotonic() - t0, 3),
        "sum_w_max_abs_err": round(sum_err, 7),
        "fallback_voxel_frac": round(fallback_frac, 5),
        "global_means": global_means,
        "roi_zones": roi_stats,
    }
    return w_views, stats, surf


# ---------------------------------------------------------------------------
# Timestep-Scheduled MV Sharpening, Late CFG Decay & 4-View Denoising
# ---------------------------------------------------------------------------

def compute_step_beta_and_cfg(
    t_val: float,
    *,
    stage: int,
    guidance_scale: float,
    cfg_interval: Sequence[float],
    sharpen_t_start: float = SHARPEN_T_START,
    cfg_decay_t_start: float = CFG_DECAY_T_START,
    cfg_late_min: float = CFG_LATE_MIN,
) -> tuple[float, float]:
    """Compute per-step temperature sharpening exponent `beta(t)` and annealed `cfg(t)`."""
    beta_max = float(STAGE_BETA_MAX.get(int(stage), 4.5))
    if t_val < sharpen_t_start:
        beta = 1.0
    else:
        frac_b = min(1.0, max(0.0, (t_val - sharpen_t_start) / max(1.0 - sharpen_t_start, 1e-6)))
        beta = 1.0 + (beta_max - 1.0) * frac_b

    low, high = float(cfg_interval[0]), float(cfg_interval[1])
    in_interval = (t_val < high) and ((low == 0.0) or (t_val > low))
    if not in_interval:
        cfg_t = 1.0
    elif t_val <= cfg_decay_t_start:
        cfg_t = float(guidance_scale)
    else:
        frac_c = min(1.0, max(0.0, (t_val - cfg_decay_t_start) / max(high - cfg_decay_t_start, 1e-6)))
        cos_w = 0.5 * (1.0 - math.cos(math.pi * frac_c))
        cfg_t = float(guidance_scale) + (float(cfg_late_min) - float(guidance_scale)) * cos_w
    return float(beta), float(cfg_t)


def sharpen_view_weights(w_views, beta: float):
    """Temperature-sharpen normalized weights `[4, N, 512]` via `w^beta / sum(w^beta)`."""
    if beta <= 1.0001:
        return w_views
    if isinstance(w_views, np.ndarray):
        wp = np.power(np.maximum(w_views, 0.0), float(beta), dtype=np.float32)
        return wp / np.maximum(wp.sum(axis=0, keepdims=True), 1e-8)
    wp = w_views.clamp_min(0.0).pow(float(beta))
    return wp / wp.sum(dim=0, keepdim=True).clamp_min(1e-8)


def sample_x0_quad(
    model,
    coords_map: Mapping[str, Any],
    perms_map: Mapping[str, Any],
    inv_perms_map: Mapping[str, Any],
    w_views,
    conds_map: Mapping[str, Any],
    generator,
    *,
    stage: int,
    patch_size: int,
    steps: int,
    guidance_scale: float,
    cfg_interval: Sequence[float],
    sharpen_t_start: float = SHARPEN_T_START,
    cfg_decay_t_start: float = CFG_DECAY_T_START,
    cfg_late_min: float = CFG_LATE_MIN,
    clamp_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[Any, list[dict[str, float]]]:
    """Run per-step 4-view fused x0 denoising across rotated sparse voxel coordinates."""
    import torch

    clamp_min, clamp_max = clamp_range
    coords_front = coords_map["front"]
    device = coords_front.device
    num_patches = len(coords_front)
    assert model.out_channels == 1, f"Expected out_channels==1, got {model.out_channels}"

    latents = torch.randn(
        (num_patches, patch_size**3 * model.out_channels),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    layout = [slice(0, num_patches)]
    uncond = torch.zeros_like(conds_map["front"]) if guidance_scale > 1.0 else None
    timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)

    step_schedule_log: list[dict[str, float]] = []
    for i in range(steps):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        t_val = float(t.item())
        timestep = t.reshape(1).to(latents.dtype)

        beta_t, cfg_val = compute_step_beta_and_cfg(
            t_val,
            stage=stage,
            guidance_scale=float(guidance_scale),
            cfg_interval=cfg_interval,
            sharpen_t_start=sharpen_t_start,
            cfg_decay_t_start=cfg_decay_t_start,
            cfg_late_min=cfg_late_min,
        )
        step_schedule_log.append(
            {"step": i, "t": round(t_val, 4), "beta": round(beta_t, 3), "cfg": round(cfg_val, 3)}
        )

        w_step = sharpen_view_weights(w_views, beta_t)
        output = torch.zeros_like(latents, dtype=torch.float32)
        for vi, view in enumerate(VIEW_ORDER):
            lat_v = rotate_latents_forward(latents, perms_map[view], patch_size, view)
            x_cond = model(lat_v, coords_map[view], layout, timestep, conds_map[view])
            if uncond is not None and cfg_val != 1.0:
                x_uncond = model(lat_v, coords_map[view], layout, timestep, uncond)
                x0_rot = x_uncond + cfg_val * (x_cond - x_uncond)
                del x_uncond
            else:
                x0_rot = x_cond
            del lat_v, x_cond
            x0_v = rotate_latents_inverse(x0_rot, inv_perms_map[view], patch_size, view)
            del x0_rot
            output.addcmul_(w_step[vi], x0_v.float())
            del x0_v
        if w_step is not w_views:
            del w_step

        output.clamp_(clamp_min, clamp_max)
        noise = torch.randn(output.shape, device=device, dtype=output.dtype, generator=generator)
        latents = output * t_next + (1.0 - t_next) * noise
    return latents, step_schedule_log


def prune_stage512_micro_shards(
    verts_t: Any,
    faces_t: Any,
    *,
    min_faces: int = 500,
    min_frac_of_largest: float = 0.005,
) -> tuple[Any, Any, dict[str, Any]]:
    """Connected-component filter on stage-512 Marching Cubes mesh to strip noise bubbles."""
    import torch

    v_np = verts_t.detach().cpu().numpy().astype(np.float32, copy=False)
    f_np = faces_t.detach().cpu().numpy().astype(np.int64, copy=False)
    n_v = len(v_np)
    n_f = len(f_np)
    if n_f == 0:
        return verts_t, faces_t, {
            "raw_faces": 0, "raw_comps": 0, "largest_comp_faces": 0,
            "largest_comp_frac": 0.0, "second_comp_faces": 0, "kept_faces": 0, "kept_comps": 0,
        }
    rows = np.concatenate([f_np[:, 0], f_np[:, 1], f_np[:, 2]])
    cols = np.concatenate([f_np[:, 1], f_np[:, 2], f_np[:, 0]])
    adj = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(n_v, n_v))
    n_comp_all, v_labels = connected_components(adj, directed=False)
    f_labels = v_labels[f_np[:, 0]]
    comp_f_counts = np.bincount(f_labels, minlength=n_comp_all)
    nonzero_comps = int((comp_f_counts > 0).sum())
    order = np.argsort(comp_f_counts)[::-1]
    largest = int(comp_f_counts[order[0]])
    second = int(comp_f_counts[order[1]]) if nonzero_comps > 1 else 0
    thresh = max(min_faces, int(largest * min_frac_of_largest))
    keep_comp_mask = comp_f_counts >= thresh
    keep_comp_mask[order[0]] = True
    keep_f_mask = keep_comp_mask[f_labels]
    f_kept = f_np[keep_f_mask]
    used_v = np.unique(f_kept)
    remap = np.full(n_v, -1, dtype=np.int32)
    remap[used_v] = np.arange(len(used_v), dtype=np.int32)
    v_kept = v_np[used_v]
    f_kept_compact = remap[f_kept]
    info = {
        "raw_faces": int(n_f),
        "raw_comps": int(nonzero_comps),
        "largest_comp_faces": largest,
        "largest_comp_frac": round(float(largest) / max(1, n_f), 4),
        "second_comp_faces": second,
        "thresh_faces": int(thresh),
        "kept_faces": int(len(f_kept_compact)),
        "kept_comps": int(keep_comp_mask.sum()),
    }
    return (
        torch.tensor(v_kept, dtype=torch.float32, device=verts_t.device),
        torch.tensor(f_kept_compact, dtype=torch.int32, device=faces_t.device),
        info,
    )


def run_d3dv3_4view_fusion(
    *,
    coarse_mesh_path: str | Path,
    view_paths: Mapping[str, str | Path],
    out_glb_path: str | Path,
    config: D3DV3FusionConfig | None = None,
    pretrained_repo: str = "wushuang98/Direct3D-V3",
    local_files_only: bool = True,
    target_height: float = 1.0,
) -> dict[str, Any]:
    """Execute end-to-end 4-view Direct3D-V3 (512 -> 1024 -> 1536) fusion on CUDA."""
    import timm
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms import Normalize

    from direct3d import Direct3DV3Pipeline
    from direct3d.utils.image import prepare_image
    from direct3d.utils.mesh import load_coarse
    from direct3d.utils.sparse_mc import sparse_mc_infer
    import direct3d.utils.voxelization as d3_vox
    import trimesh

    from quadvox.s1_geometry.repair.metrics import compute_mesh_metrics
    from quadvox.s1_geometry.repair.normalize import normalize_mesh

    cfg = config or D3DV3FusionConfig()
    for vn in VIEW_ORDER:
        if vn not in view_paths or not Path(view_paths[vn]).is_file():
            raise FileNotFoundError(f"Missing required {vn!r} view for 4-view D3D-V3 fusion: {view_paths.get(vn)}")

    class _TimmDINOv3Extractor(torch.nn.Module):
        def __init__(self, dtype: torch.dtype):
            super().__init__()
            self.backbone = timm.create_model(
                TIMM_DINOV3_MODEL,
                pretrained=True,
                dynamic_img_size=True,
            ).eval()
            for mod in self.backbone.modules():
                if hasattr(mod, "periods") and isinstance(mod.periods, torch.Tensor):
                    mod.periods.copy_(mod.periods.to(torch.bfloat16).to(torch.float32))
            self.backbone.to(dtype=dtype)
            self.normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        @torch.inference_mode()
        def forward(self, image: torch.Tensor, image_size: int) -> torch.Tensor:
            weight = self.backbone.patch_embed.proj.weight
            image = image.to(device=weight.device, dtype=torch.float32)
            if image.shape[-2:] != (image_size, image_size):
                image = F.interpolate(
                    image, size=(image_size, image_size), mode="bicubic", align_corners=False, antialias=True
                )
            image = self.normalize(image).to(weight.dtype)
            x = self.backbone.patch_embed(image)
            x, rot_pos_embed = self.backbone._pos_embed(x)
            x = self.backbone.norm_pre(x)
            features = []
            for depth, blk in enumerate(self.backbone.blocks, start=1):
                x = blk(x, rope=rot_pos_embed)
                if depth in (4, 8, 16, 32):
                    features.append(self.backbone.norm(x))
            return torch.cat(features, dim=-1)

    t0 = time.monotonic()
    pipe = Direct3DV3Pipeline.from_pretrained(
        pretrained_repo,
        torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    )
    pipe._encoder = _TimmDINOv3Extractor(dtype=pipe.dtype)
    pipe.to("cuda")
    verify_torch_rotations_selfcheck(pipe.device)
    pipe._encoder.cpu()
    torch.cuda.empty_cache()

    coarse = load_coarse(str(coarse_mesh_path))
    cv = np.asarray(coarse.vertices, dtype=np.float64)
    c_span = cv.max(axis=0) - cv.min(axis=0)
    domain_height_full = float(cfg.domain_height_full if cfg.domain_height_full is not None else c_span[1])

    out_glb = Path(out_glb_path)
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    pixels_map: dict[str, torch.Tensor] = {}
    for vn in VIEW_ORDER:
        with Image.open(view_paths[vn]) as im:
            im.load()
            raw_im = im.copy()
        if raw_im.mode == "RGBA" and (np.asarray(raw_im)[..., 3] < 250).mean() > 0.05:
            fg = raw_im
        else:
            fg = pipe.remove_background(raw_im.convert("RGB"))
        pixels_map[vn] = prepare_image(fg)
    torch.cuda.empty_cache()

    verts = torch.tensor(coarse.vertices, dtype=torch.float32, device=pipe.device)
    faces = torch.tensor(coarse.faces, dtype=torch.int32, device=pipe.device)

    stage_stats: list[dict[str, Any]] = []
    final_verts_np: np.ndarray | None = None
    final_faces_np: np.ndarray | None = None

    for stage in cfg.stage_resolutions:
        cfg_stage, denoiser = pipe._denoisers[int(stage)]
        sampling = cfg_stage["sampling"].copy()
        patch_size = int(cfg_stage["denoiser"]["patch_size"])
        grid = int(stage) // patch_size

        coords_front = d3_vox.mesh_patches(verts, faces, int(stage), patch_size, cfg.max_tokens)
        coords_map: dict[str, torch.Tensor] = {}
        perms_map: dict[str, torch.Tensor] = {}
        inv_perms_map: dict[str, torch.Tensor] = {}
        for vn in VIEW_ORDER:
            cv_t, pv_t, ipv_t = build_view_rotation(coords_front, grid, vn)
            coords_map[vn] = cv_t.int()
            perms_map[vn] = pv_t
            inv_perms_map[vn] = ipv_t

        w_views, w_stat, _ = compute_4view_voxel_weights(
            verts,
            faces,
            coords_front,
            int(stage),
            patch_size,
            domain_height_full=domain_height_full,
            domain_y_top_full=cfg.domain_y_top_full,
            roi_boxes=cfg.roi_boxes,
            smoothing_zones=cfg.smoothing_zones,
        )

        pipe._encoder.to(pipe.device)
        conds_map: dict[str, torch.Tensor] = {}
        try:
            with torch.autocast("cuda", dtype=pipe.dtype):
                for vn in VIEW_ORDER:
                    conds_map[vn] = pipe._encoder(pixels_map[vn], cfg_stage["image_size"])
        finally:
            pipe._encoder.cpu()
            torch.cuda.empty_cache()

        with torch.inference_mode(), torch.autocast("cuda", dtype=pipe.dtype):
            if int(stage) == 512 and len(cfg.stage512_seeds) > 1:
                sdfs = []
                for s_val in cfg.stage512_seeds:
                    gen_s = torch.Generator(device=pipe.device).manual_seed(int(s_val))
                    sdf_s, _ = sample_x0_quad(
                        denoiser, coords_map, perms_map, inv_perms_map, w_views, conds_map, gen_s,
                        stage=int(stage), patch_size=patch_size,
                        steps=int(sampling["steps"]),
                        guidance_scale=float(cfg.guidance_scale),
                        cfg_interval=sampling["cfg_interval"],
                        sharpen_t_start=cfg.sharpen_t_start,
                        cfg_decay_t_start=cfg.cfg_decay_t_start,
                        cfg_late_min=cfg.cfg_late_min,
                    )
                    sdfs.append(sdf_s)
                sdf_fused = torch.median(torch.stack(sdfs, dim=0), dim=0).values
            else:
                gen = torch.Generator(device=pipe.device).manual_seed(int(cfg.refine_seed))
                sdf_fused, _ = sample_x0_quad(
                    denoiser, coords_map, perms_map, inv_perms_map, w_views, conds_map, gen,
                    stage=int(stage), patch_size=patch_size,
                    steps=int(sampling["steps"]),
                    guidance_scale=float(cfg.guidance_scale),
                    cfg_interval=sampling["cfg_interval"],
                    sharpen_t_start=cfg.sharpen_t_start,
                    cfg_decay_t_start=cfg.cfg_decay_t_start,
                    cfg_late_min=cfg.cfg_late_min,
                )

            mc_v, mc_f = sparse_mc_infer(coords_front[:, 1:], sdf_fused, int(stage), patch_size, 0.0)
            if int(stage) == 512:
                mc_v, mc_f, _ = prune_stage512_micro_shards(mc_v, mc_f)

        verts = mc_v.to(dtype=torch.float32, device=pipe.device)
        faces = mc_f.to(dtype=torch.int32, device=pipe.device)
        final_verts_np = verts.detach().cpu().numpy()
        final_faces_np = faces.detach().cpu().numpy()
        stage_stats.append(w_stat)
        torch.cuda.empty_cache()

    assert final_verts_np is not None and final_faces_np is not None
    raw_mesh = trimesh.Trimesh(vertices=final_verts_np, faces=final_faces_np, process=False)
    norm_mesh = normalize_mesh(raw_mesh, target_height=target_height)
    norm_mesh.export(str(out_glb))
    metrics = compute_mesh_metrics(norm_mesh)

    return {
        "out_glb": str(out_glb),
        "elapsed_s": round(time.monotonic() - t0, 3),
        "stage_stats": stage_stats,
        "metrics": metrics.to_dict(),
    }
