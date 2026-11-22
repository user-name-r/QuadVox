"""Generic color-space helpers, region-protected delighting, and multi-view exposure matching."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    return np.where(arr <= 0.04045, arr / 12.92, np.power((arr + 0.055) / 1.055, 2.4)).astype(np.float32)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    return np.where(arr <= 0.0031308, arr * 12.92, 1.055 * np.power(arr, 1.0 / 2.4) - 0.055).astype(np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB `[..., 3]` in `[0, 1]` to CIE L*a*b* (D65)."""
    lin = srgb_to_linear(rgb)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = lin @ m.T
    xyz_ref = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    t = xyz / xyz_ref
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(t > eps, np.cbrt(np.maximum(t, 0.0)), (kappa * t + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIE L*a*b* (D65) `[..., 3]` to sRGB `[0, 1]`."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    xr = np.where(fx**3 > eps, fx**3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(L > kappa * eps, ((L + 16.0) / 116.0) ** 3, L / kappa)
    zr = np.where(fz**3 > eps, fz**3, (116.0 * fz - 16.0) / kappa)
    xyz = np.stack([xr * 0.95047, yr * 1.00000, zr * 1.08883], axis=-1)
    m_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float32,
    )
    lin = xyz @ m_inv.T
    return linear_to_srgb(lin)


def apply_region_protected_delight(
    orig_rgb: np.ndarray,
    delighted_rgb: np.ndarray,
    protect_mask: np.ndarray | None,
    fg_mask: np.ndarray,
) -> np.ndarray:
    """Blend delighted image with optional user-supplied `protect_mask` `[0, 1]` preserving boundary tone."""
    delight = np.clip(delighted_rgb.astype(np.float32), 0.0, 1.0)
    if protect_mask is None:
        return delight

    orig = np.clip(orig_rgb.astype(np.float32), 0.0, 1.0)
    mp = np.clip(protect_mask.astype(np.float32), 0.0, 1.0)

    ring = fg_mask & (mp > 0.05) & (mp < 0.35)
    if int(np.sum(ring)) >= 32:
        med_orig = np.median(orig[ring], axis=0)
        med_del = np.median(delight[ring], axis=0)
        gain = np.clip(med_orig / np.maximum(med_del, 1e-4), 0.88, 1.14)
        delight = np.clip(delight * (0.5 + 0.5 * gain[None, None, :]), 0.0, 1.0)

    return np.clip(mp[:, :, None] * orig + (1.0 - mp[:, :, None]) * delight, 0.0, 1.0).astype(np.float32)


def constrained_exposure_match(
    views_rgb: Sequence[np.ndarray],
    views_fg: Sequence[np.ndarray],
    *,
    anchor_idx: int = 0,
    max_gain_delta: float = 0.15,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Match multi-view brightness/white-balance to anchor view with gain clamped to `±max_gain_delta`."""
    if not views_rgb:
        return [], []
    anchor_rgb = views_rgb[anchor_idx]
    anchor_fg = views_fg[anchor_idx]
    if not np.any(anchor_fg):
        return [v.copy() for v in views_rgb], [{"gain": [1.0, 1.0, 1.0]} for _ in views_rgb]

    anchor_med = np.median(anchor_rgb[anchor_fg], axis=0)
    lo_g = 1.0 - float(max_gain_delta)
    hi_g = 1.0 + float(max_gain_delta)

    out_views: list[np.ndarray] = []
    stats: list[dict[str, Any]] = []
    for idx, (rgb, fg) in enumerate(zip(views_rgb, views_fg)):
        if idx == anchor_idx or not np.any(fg):
            out_views.append(rgb.copy())
            stats.append({"idx": idx, "gain": [1.0, 1.0, 1.0]})
            continue
        cur_med = np.median(rgb[fg], axis=0)
        gain = np.clip(anchor_med / np.maximum(cur_med, 1e-3), lo_g, hi_g)
        matched = np.clip(rgb * gain[None, None, :], 0.0, 1.0).astype(np.float32)
        out_views.append(matched)
        stats.append({"idx": idx, "gain": [round(float(x), 4) for x in gain]})
    return out_views, stats
