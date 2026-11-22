"""Generic UV-space PBR map derivation (Curvature wear, AO crevice grime, and Marigold-IID)."""

from __future__ import annotations

import numpy as np

from quadvox.s0_views.camera import dilate_nearest_texel


def _smoothstep(e0: float, e1: float, x: np.ndarray | float) -> np.ndarray | float:
    lo, hi = float(e0), float(e1)
    t = np.clip((x - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def blend_tangent_normal(normal_01: np.ndarray, strength_map: np.ndarray | float) -> np.ndarray:
    """Scale tangent-space XY normal perturbation by `strength_map` and re-normalize to `[0, 1]`."""
    n = normal_01.astype(np.float32) * 2.0 - 1.0
    n[:, :, 0] *= strength_map
    n[:, :, 1] *= strength_map
    n[:, :, 2] = np.maximum(n[:, :, 2], 0.05)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    return np.clip(0.5 * (n + 1.0), 0.0, 1.0).astype(np.float32)


def derive_generic_pbr_maps(
    *,
    albedo_base_4k: np.ndarray,
    normal_4k: np.ndarray,
    ao_4k: np.ndarray,
    curvature_4k: np.ndarray,
    valid_mask: np.ndarray,
    iid_rough_4k: np.ndarray | None = None,
    iid_metal_4k: np.ndarray | None = None,
    normal_strength: float = 0.85,
    ao_strength: float = 0.65,
    base_roughness: float = 0.65,
    base_metallic: float = 0.0,
) -> dict[str, np.ndarray]:
    """Derive clean general-purpose PBR maps (Albedo, Normal, ORM) from S2 4K maps + optional Marigold-IID."""
    alb = np.clip(albedo_base_4k.astype(np.float32), 0.0, 1.0)
    ao = np.clip(ao_4k.astype(np.float32), 0.0, 1.0)
    curv = np.clip(curvature_4k.astype(np.float32), 0.0, 1.0)

    nrm_out = blend_tangent_normal(normal_4k, float(normal_strength))
    occ = np.clip(1.0 - float(ao_strength) * (1.0 - ao), 0.0, 1.0).astype(np.float32)

    if iid_rough_4k is not None:
        rough = np.clip(iid_rough_4k.astype(np.float32), 0.08, 0.98)
    else:
        rough = np.full(ao.shape, float(base_roughness), dtype=np.float32)
    rough = np.clip(rough - 0.06 * (curv - 0.5), 0.06, 0.98).astype(np.float32)

    if iid_metal_4k is not None:
        metal = np.clip(iid_metal_4k.astype(np.float32), 0.0, 1.0)
    else:
        metal = np.full(ao.shape, float(base_metallic), dtype=np.float32)

    orm = np.stack([occ, rough, metal], axis=-1).astype(np.float32)
    return {
        "albedo": dilate_nearest_texel(alb, valid_mask),
        "normal": dilate_nearest_texel(nrm_out, valid_mask),
        "orm": dilate_nearest_texel(orm, valid_mask),
    }


def derive_prop_pbr_maps(
    *,
    albedo_base_4k: np.ndarray,
    normal_4k: np.ndarray,
    ao_4k: np.ndarray,
    curvature_4k: np.ndarray,
    valid_mask: np.ndarray,
    iid_rough_4k: np.ndarray | None = None,
    iid_metal_4k: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Hard-surface metal PBR layering driven by 4K Curvature (>0.55 wear) and AO (<0.45 grime)."""
    alb = np.clip(albedo_base_4k.astype(np.float32), 0.0, 1.0).copy()
    ao = np.clip(ao_4k.astype(np.float32), 0.0, 1.0)
    curv = np.clip(curvature_4k.astype(np.float32), 0.0, 1.0)

    luma = 0.2126 * alb[:, :, 0] + 0.7152 * alb[:, :, 1] + 0.0722 * alb[:, :, 2]

    if iid_rough_4k is not None:
        rough = np.clip(iid_rough_4k.astype(np.float32), 0.14, 0.96)
    else:
        rough = np.where(luma < 0.42, 0.48, 0.62).astype(np.float32)

    if iid_metal_4k is not None:
        metal = np.clip(iid_metal_4k.astype(np.float32), 0.0, 1.0)
    else:
        metal = np.where(luma < 0.42, 0.55, 0.20).astype(np.float32)

    wear_mask = (_smoothstep(0.55, 0.74, curv) * (0.35 + 0.65 * ao)).astype(np.float32)
    bare_metal_rgb = np.array([0.76, 0.79, 0.82], dtype=np.float32)[None, None, :]
    alb = alb * (1.0 - 0.58 * wear_mask[:, :, None]) + bare_metal_rgb * (0.58 * wear_mask[:, :, None])
    rough = np.clip(rough - 0.24 * wear_mask, 0.14, 0.96)
    metal = np.clip(metal + 0.62 * wear_mask, 0.0, 1.0)

    grime_mask = ((1.0 - _smoothstep(0.20, 0.45, ao)) * (0.55 + 0.45 * (1.0 - _smoothstep(0.30, 0.46, curv)))).astype(np.float32)
    alb = alb * (1.0 - 0.32 * grime_mask[:, :, None])
    rough = np.clip(rough * (1.0 - 0.75 * grime_mask) + 0.90 * (0.75 * grime_mask), 0.14, 0.98)
    metal = np.clip(metal * (1.0 - 0.85 * grime_mask), 0.0, 1.0)

    occ = np.clip(0.18 + 0.82 * ao, 0.0, 1.0)
    orm_prop = np.stack([occ, rough, metal], axis=-1).astype(np.float32)

    return {
        "albedo_prop": dilate_nearest_texel(np.clip(alb, 0.0, 1.0), valid_mask),
        "orm_prop": dilate_nearest_texel(orm_prop, valid_mask),
        "normal_prop": dilate_nearest_texel(np.clip(normal_4k.astype(np.float32), 0.0, 1.0), valid_mask),
        "wear_mask": dilate_nearest_texel(wear_mask, valid_mask),
        "grime_mask": dilate_nearest_texel(grime_mask, valid_mask),
    }
