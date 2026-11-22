"""Part-aware UV unwrapping (with explicit face texel boost), 4K Normal+AO+Curvature
Cycles baking, and 4-state 6-view HD studio rendering for Track 3.

Top-level imports are strictly Python stdlib so this module runs cleanly both
inside `.venv` Python and directly inside headless Blender 5.3 via:
    blender --background --factory-startup --python-exit-code 1 --python quadvox/s2_retopo/slim_uv_bake.py -- <cmd> ...

All temporary staging files live strictly inside the caller's output directory
(ZERO hardcoded system temp paths).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from quadvox.tools.external_tools import resolve_blender_bin

DEFAULT_BLENDER_BIN = resolve_blender_bin()

VALID_TRACK3_STATES: tuple[str, ...] = (
    "wireframe_clay",
    "smooth_lowpoly",
    "normal_mapped",
    "normal_ao_composite",
    "highpoly_clay",
)

CHARACTER_VIEW_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "front",
        "label": "FRONT (+Z)",
        "location": (0.0, -3.2, 0.50),
        "target": (0.0, 0.0, 0.50),
        "ortho": True,
        "ortho_scale": 1.14,
        "width": 768,
        "height": 900,
    },
    {
        "key": "three_quarter",
        "label": "3/4 VIEW",
        "location": (1.65, -2.15, 0.72),
        "target": (0.0, 0.0, 0.50),
        "ortho": False,
        "lens": 85.0,
        "width": 768,
        "height": 900,
    },
    {
        "key": "left",
        "label": "LEFT (+X)",
        "location": (3.2, 0.0, 0.50),
        "target": (0.0, 0.0, 0.50),
        "ortho": True,
        "ortho_scale": 1.14,
        "width": 768,
        "height": 900,
    },
    {
        "key": "back",
        "label": "BACK (-Z)",
        "location": (0.0, 3.2, 0.50),
        "target": (0.0, 0.0, 0.50),
        "ortho": True,
        "ortho_scale": 1.14,
        "width": 768,
        "height": 900,
    },
    {
        "key": "right",
        "label": "RIGHT (-X)",
        "location": (-3.2, 0.0, 0.50),
        "target": (0.0, 0.0, 0.50),
        "ortho": True,
        "ortho_scale": 1.14,
        "width": 768,
        "height": 900,
    },
    {
        "key": "head",
        "label": "HEAD CLOSE-UP",
        "location": (0.25, -1.04, 0.87),
        "target": (0.0, 0.0, 0.86),
        "ortho": False,
        "lens": 90.0,
        "width": 1024,
        "height": 1024,
    },
)

OBJECT_VIEW_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "front",
        "label": "FRONT (+Z)",
        "location": (0.0, -3.0, 0.50),
        "target": (0.0, 0.0, 0.50),
        "ortho": True,
        "ortho_scale": 1.08,
        "width": 768,
        "height": 900,
    },
    {
        "key": "three_quarter",
        "label": "3/4 VIEW",
        "location": (1.65, -2.05, 0.95),
        "target": (0.0, 0.0, 0.50),
        "ortho": False,
        "lens": 80.0,
        "width": 768,
        "height": 900,
    },
    {
        "key": "left",
        "label": "LEFT (+X)",
        "location": (3.0, 0.0, 0.50),
        "target": (0.0, 0.0, 0.50),
        "ortho": True,
        "ortho_scale": 1.08,
        "width": 768,
        "height": 900,
    },
    {
        "key": "back",
        "label": "BACK (-Z)",
        "location": (0.0, 3.0, 0.50),
        "target": (0.0, 0.0, 0.50),
        "ortho": True,
        "ortho_scale": 1.08,
        "width": 768,
        "height": 900,
    },
    {
        "key": "right",
        "label": "RIGHT (-X)",
        "location": (-3.0, 0.0, 0.50),
        "target": (0.0, 0.0, 0.50),
        "ortho": True,
        "ortho_scale": 1.08,
        "width": 768,
        "height": 900,
    },
    {
        "key": "head",
        "label": "DETAIL CLOSE-UP",
        "location": (0.55, -1.35, 0.72),
        "target": (0.0, 0.0, 0.50),
        "ortho": False,
        "lens": 85.0,
        "width": 1024,
        "height": 1024,
    },
)

PART_UV_PALETTE_U8: dict[int, tuple[int, int, int]] = {
    0: (96, 102, 112),
    1: (245, 196, 174),  # head_face
    2: (186, 118, 52),   # hair
    3: (0, 212, 245),    # rigid_accessory
    4: (62, 130, 238),   # upper_garment
    5: (224, 68, 172),   # pleated_skirt
    6: (74, 196, 118),   # limbs_hands
    7: (242, 146, 38),   # socks_shoes
}

CANONICAL_MATERIAL_SLOTS: tuple[tuple[int, str], ...] = (
    (1, "Mat_01_HeadFace"),
    (2, "Mat_02_Hair"),
    (3, "Mat_03_RigidAccessory"),
    (4, "Mat_04_UpperGarment"),
    (5, "Mat_05_PleatedSkirt"),
    (6, "Mat_06_LimbsHands"),
    (7, "Mat_07_SocksShoes"),
)


@dataclass
class PartAwareUnwrapResult:
    """Result metadata and UV quality metrics from `unwrap_part_aware`."""

    input_mesh_path: str
    out_mesh_path: str
    tex_size: int
    padding_px: int
    face_scale_boost: float
    vertex_count: int
    face_count: int
    part_count: int
    island_count: int
    uv_coverage: float
    uv_overlap_count: int
    uv_area_sum: float
    head_uv_area_share: float
    head_3d_area_share: float
    head_vs_body_texel_ratio: float
    min_island_gap_px: float
    pack_scale: float
    part_uv_area_shares: dict[str, float] = field(default_factory=dict)
    uv_preview_png: str | None = None
    elapsed_s: float = 0.0
    uv_method: str = "xatlas"
    asset_type: str = "character"
    head_front_island_count: int = 0
    head_front_texel_ratio: float = 1.0
    pack_method: str = "shelf"
    min_true_island_gap_px_4k: float = 0.0
    flipped_triangles: int = 0
    zero_area_uv_triangles: int = 0
    material_slots: list[str] = field(default_factory=list)
    out_labels_npz: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        return target


@dataclass
class MultiMapBakeResult:
    """Result metadata and diagnostics from 4K Normal + AO + Curvature Cycles baking."""

    high_mesh_path: str
    low_uv_mesh_path: str
    out_normal_png: str
    out_ao_png: str
    out_curvature_png: str
    out_baked_glb: str
    out_contact_sheet_png: str | None
    tex_size: int
    padding_px: int
    normal_samples: int
    ao_samples: int
    curvature_samples: int
    use_gpu: bool
    device_used: str
    diag: float
    cage_extrusion: float
    max_ray_distance: float
    surface_dist_p95: float
    surface_dist_p99: float
    uv_island_pixels: int
    ray_miss_pixels: int
    ray_miss_ratio: float
    backface_hit_ratio: float
    backface_guarded_pixels: int
    ao_mean: float
    ao_std: float
    ao_p05: float
    ao_p95: float
    curvature_mean: float
    curvature_std: float
    curvature_p05: float
    curvature_p95: float
    curvature_dynamic_range: float
    curvature_source: str
    bake_elapsed_s: float
    blender_version: str
    isolated_part_bake: bool = False
    ring_dilation: int = 0
    head_ring_dilation: int = 0
    head_ao_exclude_hair: bool = False
    full_ao_miss_pixels: int = 0
    head_ao_miss_pixels: int = 0
    effective_padding_px: int = 0
    material_slots: list[str] = field(default_factory=list)
    per_part_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        return target


@dataclass
class Track3RenderResult:
    """Output paths and metadata from 4-state 6-view HD Cycles rendering."""

    mesh_path: str
    quad_mesh_path: str | None
    out_dir: str
    prefix: str
    states: list[str]
    framing_mode: str
    samples: int
    use_gpu: bool
    device_used: str
    blender_version: str
    elapsed_s: float
    outputs: dict[str, dict[str, str]] = field(default_factory=dict)
    ao_strength: float = 0.85

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        return target


# ---------------------------------------------------------------------------
# Pure Python/NumPy UV packing, part-aware unwrapping, and curvature helpers
# ---------------------------------------------------------------------------


def _try_pack_shelves(
    rects: Sequence[tuple[int, float, float]],
    scale: float,
    pad_uv: float,
) -> list[tuple[int, float, float, float, float, bool]] | None:
    """Greedy shelf-packer with 90-deg rotation check inside `[pad_uv, 1-pad_uv]^2`.

    Each input rect is `(island_idx, raw_w, raw_h)`. Returns a list of placements
    `(island_idx, x0, y0, placed_w, placed_h, rotated)` or `None` if it overflows.
    """
    usable = 1.0 - 2.0 * pad_uv
    if usable <= 1e-6:
        return None

    # Pre-orient each rect to prefer wider-than-tall on shelves unless tall fits better.
    scaled: list[tuple[int, float, float, bool]] = []
    for idx, rw, rh in rects:
        sw = max(1e-6, rw * scale)
        sh = max(1e-6, rh * scale)
        rot = False
        if sh > sw and sh <= usable:
            sw, sh = sh, sw
            rot = True
        if sw > usable or sh > usable:
            # Try opposite orientation before giving up.
            sw, sh = sh, sw
            rot = not rot
            if sw > usable or sh > usable:
                return None
        scaled.append((idx, sw, sh, rot))

    # Sort descending by height then width (Next-Fit / Best-Fit Decreasing Height).
    scaled.sort(key=lambda item: (item[2], item[1]), reverse=True)

    # Shelf state: [y_origin, shelf_height, cursor_x]
    shelves: list[list[float]] = []
    placements: list[tuple[int, float, float, float, float, bool]] = []
    y_top = pad_uv

    for idx, sw, sh, rot in scaled:
        placed = False
        # Try existing shelves (best remaining horizontal fit).
        best_shelf = -1
        best_rem = float("inf")
        best_orient = (sw, sh, rot)
        for s_i, shelf in enumerate(shelves):
            sy, shelf_h, cx = shelf
            for cw, ch, crot in ((sw, sh, rot), (sh, sw, not rot)):
                if ch <= shelf_h + 1e-9 and cx + cw <= 1.0 - pad_uv + 1e-9:
                    rem = (1.0 - pad_uv) - (cx + cw)
                    if rem < best_rem:
                        best_rem = rem
                        best_shelf = s_i
                        best_orient = (cw, ch, crot)
        if best_shelf >= 0:
            cw, ch, crot = best_orient
            sy, shelf_h, cx = shelves[best_shelf]
            placements.append((idx, cx, sy, cw, ch, crot))
            shelves[best_shelf][2] = cx + cw + pad_uv
            placed = True

        if not placed:
            # Open a new shelf.
            cw, ch, crot = sw, sh, rot
            if sh > sw and sw <= usable and sh <= usable:
                cw, ch, crot = sh, sw, not rot
            if y_top + ch > 1.0 - pad_uv + 1e-9:
                return None
            placements.append((idx, pad_uv, y_top, cw, ch, crot))
            shelves.append([y_top, ch, pad_uv + cw + pad_uv])
            y_top += ch + pad_uv

    return placements


def pack_uv_islands_normalized(
    raw_islands: Sequence[tuple[float, float]],
    *,
    tex_size: int = 4096,
    padding_px: int = 16,
    binary_search_steps: int = 28,
) -> tuple[float, list[tuple[float, float, float, float, bool]]]:
    """Pack `(w, h)` UV islands into `[0, 1]^2` with hard pixel padding `>= padding_px`.

    Uses binary search on a single global uniform scale `S`, preserving exact relative
    texel-density ratios across all islands (e.g. 2.25x head_face vs body boost!).
    Returns `(scale, placements_by_index)` where each placement is
    `(x0, y0, placed_w, placed_h, rotated)`.
    """
    n = len(raw_islands)
    if n == 0:
        return 1.0, []

    pad_uv = float(max(0, int(padding_px))) / float(max(16, int(tex_size)))
    rects = [(i, max(1e-6, float(w)), max(1e-6, float(h))) for i, (w, h) in enumerate(raw_islands)]

    total_raw_area = sum(w * h for _, w, h in rects)
    max_dim = max(max(w, h) for _, w, h in rects)
    usable = max(1e-4, 1.0 - 2.0 * pad_uv)

    # Upper bound: either largest island fills usable width, or total area fills usable^2.
    hi = min(usable / max(max_dim, 1e-9), math.sqrt((usable * usable) / max(total_raw_area, 1e-12)))
    lo = 0.0
    best_scale = 0.0
    best_placements: list[tuple[int, float, float, float, float, bool]] | None = None

    # Ensure hi is high enough if tiny padding makes initial guess loose, or find valid lo.
    for _ in range(int(binary_search_steps)):
        mid = 0.5 * (lo + hi)
        cand = _try_pack_shelves(rects, mid, pad_uv)
        if cand is not None:
            best_scale = mid
            best_placements = cand
            lo = mid
        else:
            hi = mid

    if best_placements is None:
        # Ultra-dense fallback: shrink scale until packing succeeds.
        trial = max(hi * 0.5, 1e-5)
        for _ in range(20):
            cand = _try_pack_shelves(rects, trial, pad_uv)
            if cand is not None:
                best_scale = trial
                best_placements = cand
                break
            trial *= 0.5

    if best_placements is None:
        raise RuntimeError(
            f"Unable to pack {n} UV islands into [0, 1]^2 with padding_px={padding_px} at tex_size={tex_size}"
        )

    by_idx: list[tuple[float, float, float, float, bool]] = [(0.0, 0.0, 0.0, 0.0, False)] * n
    for idx, x0, y0, pw, ph, rot in best_placements:
        by_idx[idx] = (x0, y0, pw, ph, rot)
    return float(best_scale), by_idx


def normalize_curvature_buffer(
    raw_channel: Any,
    uv_mask: Any,
    *,
    percentile_clip: float = 98.0,
    target_half_span: float = 0.45,
) -> tuple[Any, dict[str, float]]:
    """Normalize a float curvature map around 0.5 mid-gray (concave < 0.5, convex > 0.5).

    Preserves dilation margins cleanly by applying the same affine transform to all pixels.
    """
    import numpy as np

    arr = np.asarray(raw_channel, dtype=np.float64)
    mask = np.asarray(uv_mask, dtype=bool)
    if arr.shape != mask.shape:
        raise ValueError(f"Curvature array shape {arr.shape} != uv_mask shape {mask.shape}")

    masked_vals = arr[mask] if np.any(mask) else arr.reshape(-1)
    finite = masked_vals[np.isfinite(masked_vals)]
    if len(finite) == 0:
        out = np.full(arr.shape, 0.5, dtype=np.float32)
        return out, {"mean": 0.5, "std": 0.0, "p05": 0.5, "p95": 0.5, "dynamic_range": 0.0, "raw_scale": 0.0}

    center = float(np.median(finite))
    delta_masked = finite - center
    raw_scale = float(np.percentile(np.abs(delta_masked), float(percentile_clip)))
    if not math.isfinite(raw_scale) or raw_scale <= 1e-8:
        out = np.full(arr.shape, 0.5, dtype=np.float32)
        return out, {"mean": 0.5, "std": 0.0, "p05": 0.5, "p95": 0.5, "dynamic_range": 0.0, "raw_scale": 0.0}

    normed = np.clip(0.5 + float(target_half_span) * ((arr - center) / raw_scale), 0.0, 1.0)
    normed = np.where(np.isfinite(normed), normed, 0.5).astype(np.float32)
    m_out = normed[mask] if np.any(mask) else normed.reshape(-1)
    p05 = float(np.percentile(m_out, 5.0))
    p95 = float(np.percentile(m_out, 95.0))
    stats = {
        "mean": float(np.mean(m_out)),
        "std": float(np.std(m_out)),
        "p05": p05,
        "p95": p95,
        "dynamic_range": float(p95 - p05),
        "raw_scale": raw_scale,
    }
    return normed, stats


def _render_uv_layout_preview(
    uvs: Any,
    faces: Any,
    face_classes: Any,
    out_png: Path,
    *,
    preview_size: int = 1024,
) -> Path:
    """Render a crisp semantic-colored UV layout diagnostic PNG (head_face highlighted)."""
    import numpy as np
    from PIL import Image, ImageDraw

    res = int(preview_size)
    img = Image.new("RGB", (res, res), (20, 22, 26))
    draw = ImageDraw.Draw(img)

    uv = np.asarray(uvs, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    fc = np.asarray(face_classes, dtype=np.int32)

    px_x = np.clip(uv[:, 0] * (res - 1), 0.0, res - 1)
    px_y = np.clip((1.0 - uv[:, 1]) * (res - 1), 0.0, res - 1)

    for i in range(len(f)):
        a, b, c = int(f[i, 0]), int(f[i, 1]), int(f[i, 2])
        cid = int(fc[i])
        col = PART_UV_PALETTE_U8.get(cid, (140, 145, 155))
        pts = [(px_x[a], px_y[a]), (px_x[b], px_y[b]), (px_x[c], px_y[c])]
        draw.polygon(pts, fill=col)

    # Subsample wireframe edges if mesh is very dense so chart outlines remain readable.
    step = max(1, len(f) // 35000)
    for i in range(0, len(f), step):
        a, b, c = int(f[i, 0]), int(f[i, 1]), int(f[i, 2])
        pts = [(px_x[a], px_y[a]), (px_x[b], px_y[b]), (px_x[c], px_y[c]), (px_x[a], px_y[a])]
        draw.line(pts, fill=(18, 20, 24), width=1)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_png))
    return out_png


def _split_uv_connected_islands(
    verts_3d: Any,
    normals_3d: Any,
    uvs: Any,
    faces: Any,
    class_id: int,
    scale_factor: float,
) -> list[dict[str, Any]]:
    """Split a per-part xatlas mesh into individual connected UV islands with normalized scale."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    nv = len(verts_3d)
    nf = len(faces)
    if nv == 0 or nf == 0:
        return []

    e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    rows = np.concatenate([e0, e1])
    cols = np.concatenate([e1, e0])
    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(nv, nv))
    n_comp, v_comp = connected_components(adj, directed=False)
    f_comp = v_comp[faces[:, 0]]

    islands: list[dict[str, Any]] = []
    for comp_id in range(n_comp):
        f_mask = f_comp == comp_id
        if not np.any(f_mask):
            continue
        sub_f = faces[f_mask]
        used_v = np.unique(sub_f)
        remap = np.full(nv, -1, dtype=np.int64)
        remap[used_v] = np.arange(len(used_v), dtype=np.int64)
        local_f = remap[sub_f]
        local_v = verts_3d[used_v]
        local_n = normals_3d[used_v]
        local_uv = (uvs[used_v] * float(scale_factor)).astype(np.float64)
        uv_min = local_uv.min(axis=0)
        local_uv -= uv_min
        uv_max = local_uv.max(axis=0)
        w = max(1e-6, float(uv_max[0]))
        h = max(1e-6, float(uv_max[1]))

        v0 = local_v[local_f[:, 0]]
        v1 = local_v[local_f[:, 1]]
        v2 = local_v[local_f[:, 2]]
        a3d = 0.5 * float(np.sum(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)))
        islands.append(
            {
                "class_id": int(class_id),
                "verts": local_v,
                "normals": local_n,
                "uvs": local_uv,
                "faces": local_f,
                "raw_w": w,
                "raw_h": h,
                "area_3d": a3d,
            }
        )
    return islands


def _clean_face_labels_iterative(
    faces: Any,
    face_labels: Any,
    *,
    min_faces: int = 24,
    max_iters: int = 8,
) -> Any:
    """Iteratively absorb small semantic/chart label speckles until convergence."""
    import numpy as np

    from quadvox.s2_retopo.quad_retopo import clean_small_face_label_islands

    fl = np.asarray(face_labels, dtype=np.int32).copy()
    for _ in range(max_iters):
        nxt = clean_small_face_label_islands(faces, fl, min_faces=min_faces)
        if np.array_equal(nxt, fl):
            break
        fl = nxt
    return fl


def _build_edge_face_map(faces: Any) -> dict[tuple[int, int], list[int]]:
    """Map undirected vertex-pair edges `(min, max)` to incident face indices."""
    ef: dict[tuple[int, int], list[int]] = {}
    for fi, tri in enumerate(faces):
        for j in range(3):
            a, b = int(tri[j]), int(tri[(j + 1) % 3])
            k = (a, b) if a < b else (b, a)
            ef.setdefault(k, []).append(fi)
    return ef


def _split_pinch_vertices(
    verts: Any,
    normals: Any,
    faces: Any,
) -> tuple[Any, Any, Any]:
    """Duplicate non-manifold pinch vertices so every vertex has a single edge-connected fan."""
    import numpy as np

    v_faces: list[list[int]] = [[] for _ in range(len(verts))]
    for fi, tri in enumerate(faces):
        for vi in tri:
            v_faces[int(vi)].append(fi)
    new_verts = verts.tolist()
    new_norms = normals.tolist()
    new_faces = faces.copy()
    for vi, flist in enumerate(v_faces):
        if len(flist) <= 1:
            continue
        fsets = [{int(x) for x in faces[fi]} - {vi} for fi in flist]
        vis = [False] * len(flist)
        fans: list[list[int]] = []
        for start in range(len(flist)):
            if vis[start]:
                continue
            fan: list[int] = []
            stack = [start]
            vis[start] = True
            while stack:
                cur = stack.pop()
                fan.append(flist[cur])
                for nxt in range(len(flist)):
                    if not vis[nxt] and (fsets[cur] & fsets[nxt]):
                        vis[nxt] = True
                        stack.append(nxt)
            fans.append(fan)
        if len(fans) > 1:
            for fan in fans[1:]:
                new_vi = len(new_verts)
                new_verts.append(verts[vi].tolist())
                new_norms.append(normals[vi].tolist())
                for fi in fan:
                    new_faces[fi][new_faces[fi] == vi] = new_vi
    return (
        np.asarray(new_verts, dtype=np.float64),
        np.asarray(new_norms, dtype=np.float64),
        np.asarray(new_faces, dtype=np.int64),
    )


def _orient_sheet_faces_for_uv(faces: Any) -> tuple[Any, Any]:
    """Ensure consistent halfedge orientation per manifold sheet without mutating export winding."""
    import numpy as np

    F = len(faces)
    ef = _build_edge_face_map(faces)
    face_he: list[dict[tuple[int, int], tuple[int, int]]] = []
    for tri in faces:
        d: dict[tuple[int, int], tuple[int, int]] = {}
        for j in range(3):
            a, b = int(tri[j]), int(tri[(j + 1) % 3])
            k = (a, b) if a < b else (b, a)
            d[k] = (a, b)
        face_he.append(d)

    flip = np.zeros(F, dtype=bool)
    vis = np.zeros(F, dtype=bool)
    out = faces.copy()

    same_dir = False
    for k, fl in ef.items():
        if len(fl) == 2 and face_he[fl[0]][k] == face_he[fl[1]][k]:
            same_dir = True
            break
    if not same_dir:
        return out, flip

    adj: list[list[tuple[int, tuple[int, int]]]] = [[] for _ in range(F)]
    for k, fl in ef.items():
        if len(fl) == 2:
            f1, f2 = fl
            adj[f1].append((f2, k))
            adj[f2].append((f1, k))

    from collections import deque

    for root in range(F):
        if vis[root]:
            continue
        vis[root] = True
        flip[root] = False
        q: deque[int] = deque([root])
        comp_faces: list[int] = [root]
        while q:
            u = q.popleft()
            for v, k in adj[u]:
                if vis[v]:
                    continue
                same = face_he[u][k] == face_he[v][k]
                desired_flip = (not flip[u]) if same else flip[u]
                vis[v] = True
                flip[v] = desired_flip
                comp_faces.append(v)
                q.append(v)
        # Preserve the majority winding per connected component so 1 reversed tri never flips a whole island.
        if sum(1 for fid in comp_faces if flip[fid]) * 2 > len(comp_faces):
            for fid in comp_faces:
                flip[fid] = not flip[fid]
        for fid in comp_faces:
            if flip[fid]:
                out[fid] = out[fid][[0, 2, 1]]
    return out, flip


def _extract_manifold_uv_components(
    verts: Any,
    normals: Any,
    faces: Any,
) -> list[tuple[Any, Any, Any, Any, Any]]:
    """Split non-manifold edges and pinch vertices into 2-manifold sheets for UV cut graphs."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    ef = _build_edge_face_map(faces)
    rows, cols = [], []
    for fl in ef.values():
        if len(fl) == 2:
            rows.extend([fl[0], fl[1]])
            cols.extend([fl[1], fl[0]])
    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(faces), len(faces)))
    nc, cc = connected_components(adj, directed=False)
    out: list[tuple[Any, Any, Any, Any, Any]] = []
    for c in range(nc):
        cf_raw = faces[cc == c]
        u = np.unique(cf_raw)
        rm = np.full(len(verts), -1, dtype=np.int64)
        rm[u] = np.arange(len(u), dtype=np.int64)
        sv, sn, sf = _split_pinch_vertices(verts[u], normals[u], rm[cf_raw])
        ef2 = _build_edge_face_map(sf)
        r2, c2 = [], []
        for fl in ef2.values():
            if len(fl) == 2:
                r2.extend([fl[0], fl[1]])
                c2.extend([fl[1], fl[0]])
        adj2 = csr_matrix((np.ones(len(r2), dtype=np.int8), (r2, c2)), shape=(len(sf), len(sf)))
        nc2, cc2 = connected_components(adj2, directed=False)
        for c_sub in range(nc2):
            sub_f = sf[cc2 == c_sub]
            su = np.unique(sub_f)
            srm = np.full(len(sv), -1, dtype=np.int64)
            srm[su] = np.arange(len(su), dtype=np.int64)
            cv = np.ascontiguousarray(sv[su], dtype=np.float64)
            cn = np.ascontiguousarray(sn[su], dtype=np.float64)
            cf_orig = np.ascontiguousarray(srm[sub_f], dtype=np.int64)
            cf_ori, c_flip = _orient_sheet_faces_for_uv(cf_orig)
            out.append((cv, cn, cf_orig, cf_ori, c_flip))
    return out


def _compute_face_normals_uv(verts: Any, faces: Any) -> Any:
    import numpy as np

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(fn, axis=1, keepdims=True)
    return fn / np.maximum(norms, 1e-12)


def _compute_boundary_loops(faces: Any) -> tuple[int, list[list[int]]]:
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    ef = _build_edge_face_map(faces)
    be = [k for k, fl in ef.items() if len(fl) == 1]
    if not be:
        return 0, []
    bv = sorted({x for k in be for x in k})
    bm = {x: i for i, x in enumerate(bv)}
    adj = csr_matrix(
        (
            np.ones(len(be) * 2),
            ([bm[a] for a, _ in be] + [bm[b] for _, b in be], [bm[b] for _, b in be] + [bm[a] for a, b in be]),
        ),
        shape=(len(bv), len(bv)),
    )
    nbl, bcc = connected_components(adj, directed=False)
    loops = [[bv[i] for i in range(len(bv)) if bcc[i] == l] for l in range(nbl)]
    return int(nbl), loops


def _dijkstra_seam_path(
    verts: Any,
    edge_weights: dict[tuple[int, int], float],
    src_verts: list[int],
    dst_verts: list[int],
) -> list[tuple[int, int]]:
    import heapq

    import numpy as np

    if not src_verts or not dst_verts:
        return []
    adj: list[list[tuple[int, float]]] = [[] for _ in range(len(verts))]
    for (a, b), w in edge_weights.items():
        adj[a].append((b, w))
        adj[b].append((a, w))
    dst_set = set(dst_verts)
    dist = np.full(len(verts), float("inf"))
    prev = np.full(len(verts), -1, dtype=np.int64)
    pq: list[tuple[float, int]] = []
    for s in src_verts:
        dist[s] = 0.0
        heapq.heappush(pq, (0.0, s))
    end_v = -1
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u in dst_set:
            end_v = u
            break
        for nxt, w in adj[u]:
            nd = d + w
            if nd < dist[nxt]:
                dist[nxt] = nd
                prev[nxt] = u
                heapq.heappush(pq, (nd, nxt))
    if end_v < 0:
        return []
    path: list[tuple[int, int]] = []
    cur = end_v
    while prev[cur] >= 0:
        p = int(prev[cur])
        path.append((min(cur, p), max(cur, p)))
        cur = p
    return path


def _prune_confetti_seams(faces: Any, seams: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove any seam edges that split a dual-connected chart into disconnected confetti shards."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    if not seams:
        return []
    ef = _build_edge_face_map(faces)
    seam_set = set(seams)
    rows, cols = [], []
    for k, fl in ef.items():
        if len(fl) == 2 and k not in seam_set:
            rows.extend([fl[0], fl[1]])
            cols.extend([fl[1], fl[0]])
    F = len(faces)
    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(F, F))
    nc, cc = connected_components(adj, directed=False)
    if nc <= 1:
        return seams
    parent = list(range(nc))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    kept: list[tuple[int, int]] = []
    for k in seams:
        fl = ef.get(k)
        if fl and len(fl) == 2:
            c1, c2 = _find(int(cc[fl[0]])), _find(int(cc[fl[1]]))
            if c1 != c2:
                parent[c1] = c2
                continue
        kept.append(k)
    return kept


def _topological_cut_graph_seams(
    verts: Any,
    faces: Any,
    fn: Any,
    *,
    prefer_back: bool = True,
) -> list[tuple[int, int]]:
    """Compute minimal hidden cut-graph seams opening handles/holes/closed spheres to a single disk."""
    import heapq

    import numpy as np

    ef = _build_edge_face_map(faces)
    int_edges = [k for k, fl in ef.items() if len(fl) == 2]
    bnd_edges = [k for k, fl in ef.items() if len(fl) == 1]
    bnd_v = {x for k in bnd_edges for x in k}

    V = len(np.unique(faces))
    E = len(ef)
    F = len(faces)
    chi = V - E + F
    nb_loops, _ = _compute_boundary_loops(faces)
    if chi == 1 and nb_loops == 1:
        return []

    if chi == 2 and nb_loops == 0 and len(int_edges) >= 2:
        ew: dict[tuple[int, int], float] = {}
        for k in int_edges:
            f1, f2 = ef[k]
            dot = float(np.dot(fn[f1], fn[f2]))
            elen = float(np.linalg.norm(verts[k[1]] - verts[k[0]]))
            ew[k] = elen * (0.2 + 0.8 * max(0.0, 1.0 + dot) * 0.5)
        if prefer_back:
            s_v = int(np.argmin(verts[:, 2] - 0.3 * verts[:, 1]))
            e_v = int(np.argmin(verts[:, 2] + 0.3 * verts[:, 1]))
        else:
            centered = verts - verts.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ vt[0]
            s_v = int(np.argmin(proj))
            e_v = int(np.argmax(proj))
        if s_v == e_v:
            e_v = (s_v + 1) % len(verts)
        slit = _dijkstra_seam_path(verts, ew, [s_v], [e_v])
        if len(slit) < 2:
            slit = int_edges[: min(3, len(int_edges))]
        return _prune_confetti_seams(faces, slit)

    f_adj: list[list[tuple[float, int, tuple[int, int]]]] = [[] for _ in range(F)]
    for k in int_edges:
        f1, f2 = ef[k]
        nz = 0.5 * (float(fn[f1, 2]) + float(fn[f2, 2]))
        dot = float(np.dot(fn[f1], fn[f2]))
        z_mid = 0.5 * (float(verts[k[0], 2]) + float(verts[k[1], 2]))
        if prefer_back:
            cost = (1.0 - nz) * 10.0 + (1.0 - dot) * 5.0 - z_mid * 8.0
        else:
            cost = (1.0 - dot) * 10.0
        f_adj[f1].append((cost, f2, k))
        f_adj[f2].append((cost, f1, k))

    root_f = int(np.argmax(fn[:, 2] + 4.0 * verts[faces].mean(axis=1)[:, 2])) if prefer_back else 0
    vis = [False] * F
    vis[root_f] = True
    pq: list[tuple[float, int, tuple[int, int]]] = []
    for cst, nf, k in f_adj[root_f]:
        heapq.heappush(pq, (cst, nf, k))
    t_star: set[tuple[int, int]] = set()
    while pq:
        _, vf, k = heapq.heappop(pq)
        if vis[vf]:
            continue
        vis[vf] = True
        t_star.add(k)
        for ncst, nnf, nk in f_adj[vf]:
            if not vis[nnf]:
                heapq.heappush(pq, (ncst, nnf, nk))

    s0 = [k for k in int_edges if k not in t_star and vis[ef[k][0]] and vis[ef[k][1]]]
    seam_adj: dict[int, set[int]] = {}
    for a, b in s0:
        seam_adj.setdefault(a, set()).add(b)
        seam_adj.setdefault(b, set()).add(a)

    if bnd_v:
        q = [x for x, nbrs in seam_adj.items() if len(nbrs) == 1 and x not in bnd_v]
        while q:
            x = q.pop()
            nbrs = seam_adj.get(x)
            if not nbrs or len(nbrs) != 1:
                continue
            y = next(iter(nbrs))
            nbrs.remove(y)
            seam_adj[y].remove(x)
            if len(seam_adj[y]) == 1 and y not in bnd_v:
                q.append(y)

    raw_seams = [(min(a, b), max(a, b)) for a, nbrs in seam_adj.items() for b in nbrs if a < b]
    return _prune_confetti_seams(faces, raw_seams)


def _peel_minor_holes_to_outer(
    verts: Any,
    normals: Any,
    faces: Any,
) -> list[tuple[Any, Any, Any, Any, Any, bool]]:
    """Peel side hair-bang islands framing holes away from the main front face disk."""
    import numpy as np

    nbl, loops = _compute_boundary_loops(faces)
    if nbl <= 1:
        ori, flp = _orient_sheet_faces_for_uv(faces)
        return [(verts, normals, faces, ori, flp, True)]

    outer_l = int(np.argmax([len(l) for l in loops]))
    face_c = verts[faces].mean(axis=1)
    ctr_x = float(np.median(verts[:, 0]))
    keep_mask = np.ones(len(faces), dtype=bool)
    for l_idx, l_vs in enumerate(loops):
        if l_idx == outer_l:
            continue
        l_pts = verts[np.asarray(l_vs, dtype=np.int64)]
        if float(l_pts[:, 0].mean()) < ctr_x:
            thr_x = float(l_pts[:, 0].max()) + 0.0005
            keep_mask &= face_c[:, 0] > thr_x
        else:
            thr_x = float(l_pts[:, 0].min()) - 0.0005
            keep_mask &= face_c[:, 0] < thr_x

    zone = _clean_face_labels_iterative(faces, keep_mask.astype(np.int32), min_faces=12)
    all_c: list[tuple[Any, Any, Any, Any, Any]] = []
    for zval in sorted(np.unique(zone).tolist()):
        sub_f = faces[zone == zval]
        u = np.unique(sub_f)
        rm = np.full(len(verts), -1, dtype=np.int64)
        rm[u] = np.arange(len(u), dtype=np.int64)
        all_c.extend(_extract_manifold_uv_components(verts[u], normals[u], rm[sub_f]))
    max_len = max(len(x[2]) for x in all_c)
    return [(kv, kn, kf_orig, kf_ori, k_flp, len(kf_orig) == max_len) for kv, kn, kf_orig, kf_ori, k_flp in all_c]


def _partition_component_charts(
    verts: Any,
    faces: Any,
    fn: Any,
    *,
    target_charts: int,
    crease_deg: float = 45.0,
    min_chart_faces: int = 24,
    lloyd_iters: int = 2,
) -> Any:
    """Farthest-point + normal-aware Lloyd geodesic chart partitioning inside one manifold part."""
    import heapq

    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    F = len(faces)
    if F <= min_chart_faces or target_charts <= 1:
        return np.zeros(F, dtype=np.int32)

    ef = _build_edge_face_map(faces)
    centroids = verts[faces].mean(axis=1)
    bbox_diag = max(float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))), 1e-6)
    cos_thr = math.cos(math.radians(crease_deg))

    seed_adj: list[list[tuple[int, float]]] = [[] for _ in range(F)]
    grow_adj: list[list[tuple[int, float]]] = [[] for _ in range(F)]
    for fl in ef.values():
        if len(fl) != 2:
            continue
        f1, f2 = fl
        dot = float(np.clip(np.dot(fn[f1], fn[f2]), -1.0, 1.0))
        d3 = float(np.linalg.norm(centroids[f1] - centroids[f2])) / bbox_diag
        w_seed = max(1e-6, d3 + 0.03 * (1.0 - dot))
        crease_pen = 0.25 * max(0.0, cos_thr - dot) if dot < cos_thr else 0.0
        w_grow = max(1e-6, d3 + 0.04 * (1.0 - dot) + crease_pen)
        seed_adj[f1].append((f2, w_seed))
        seed_adj[f2].append((f1, w_seed))
        grow_adj[f1].append((f2, w_grow))
        grow_adj[f2].append((f1, w_grow))

    rows, cols = [], []
    for u, nbrs in enumerate(seed_adj):
        for v, _ in nbrs:
            rows.append(u)
            cols.append(v)
    cadj_0 = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(F, F))
    nc0, cc0 = connected_components(cadj_0, directed=False)

    final_labels = np.zeros(F, dtype=np.int32)
    next_id = 0
    for c0 in range(nc0):
        idx_c = np.flatnonzero(cc0 == c0)
        if len(idx_c) <= min_chart_faces:
            final_labels[idx_c] = next_id
            next_id += 1
            continue
        k_sub = max(1, int(round(target_charts * len(idx_c) / F)))
        k_sub = min(k_sub, max(1, len(idx_c) // max(1, min_chart_faces)))
        if k_sub <= 1:
            final_labels[idx_c] = next_id
            next_id += 1
            continue

        seeds = [int(idx_c[0])]
        min_d = np.full(F, float("inf"))
        for _ in range(k_sub):
            s = seeds[-1]
            d_cur = np.full(F, float("inf"))
            d_cur[s] = 0.0
            pq = [(0.0, s)]
            while pq:
                cd, u = heapq.heappop(pq)
                if cd > d_cur[u] or cd >= min_d[u]:
                    continue
                for v, w in seed_adj[u]:
                    nd = cd + w
                    if nd < d_cur[v] and nd < min_d[v]:
                        d_cur[v] = nd
                        heapq.heappush(pq, (nd, v))
            min_d = np.minimum(min_d, d_cur)
            if len(seeds) < k_sub:
                sub_d = min_d[idx_c]
                nxt_local = int(np.argmax(sub_d))
                if sub_d[nxt_local] <= 1e-8 or not math.isfinite(sub_d[nxt_local]):
                    break
                seeds.append(int(idx_c[nxt_local]))

        sub_chart = np.full(F, -1, dtype=np.int32)
        for it in range(max(1, lloyd_iters)):
            best_dist = np.full(F, float("inf"))
            sub_chart.fill(-1)
            pq_g: list[tuple[float, int, int]] = []
            for s_idx, s in enumerate(seeds):
                best_dist[s] = 0.0
                sub_chart[s] = s_idx
                heapq.heappush(pq_g, (0.0, s_idx, s))
            while pq_g:
                cd, cid_s, u = heapq.heappop(pq_g)
                if cd > best_dist[u]:
                    continue
                seed_n = fn[seeds[cid_s]]
                for v, w in grow_adj[u]:
                    nd = cd + w + 0.02 * (1.0 - float(np.dot(fn[v], seed_n)))
                    if nd < best_dist[v]:
                        best_dist[v] = nd
                        sub_chart[v] = cid_s
                        heapq.heappush(pq_g, (nd, cid_s, v))
            if it + 1 < lloyd_iters:
                new_seeds = []
                for s_idx in range(len(seeds)):
                    members = idx_c[sub_chart[idx_c] == s_idx]
                    if len(members) == 0:
                        new_seeds.append(seeds[s_idx])
                        continue
                    mc = centroids[members].mean(axis=0)
                    mn = fn[members].mean(axis=0)
                    mn_norm = float(np.linalg.norm(mn))
                    if mn_norm > 1e-8:
                        mn /= mn_norm
                    scores = np.linalg.norm(centroids[members] - mc, axis=1) / bbox_diag + 0.05 * (1.0 - fn[members] @ mn)
                    new_seeds.append(int(members[int(np.argmin(scores))]))
                seeds = new_seeds

        for s_idx in range(len(seeds)):
            m = idx_c[sub_chart[idx_c] == s_idx]
            if len(m) > 0:
                final_labels[m] = next_id + s_idx
        next_id += len(seeds)

    final_labels = _clean_face_labels_iterative(faces, final_labels, min_faces=min_chart_faces)
    rows, cols = [], []
    for fl in ef.values():
        if len(fl) == 2 and final_labels[fl[0]] == final_labels[fl[1]]:
            rows.extend([fl[0], fl[1]])
            cols.extend([fl[1], fl[0]])
    fadj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(F, F))
    _, final_cc = connected_components(fadj, directed=False)
    return final_cc.astype(np.int32)


def _partition_prop_planar_charts(
    verts: Any,
    faces: Any,
    fn: Any,
    *,
    crease_deg: float = 48.0,
    max_chart_faces: int = 600,
    min_chart_faces: int = 24,
) -> Any:
    """Dihedral-crease planar clustering for hard-surface props and rigid accessories."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    F = len(faces)
    ef = _build_edge_face_map(faces)
    cos_thr = math.cos(math.radians(crease_deg))
    rows, cols = [], []
    for fl in ef.values():
        if len(fl) == 2:
            f1, f2 = fl
            if float(np.dot(fn[f1], fn[f2])) >= cos_thr:
                rows.extend([f1, f2])
                cols.extend([f2, f1])
    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(F, F))
    nc, cc = connected_components(adj, directed=False)

    if nc > 12:
        cc = _clean_face_labels_iterative(faces, cc.astype(np.int32), min_faces=min_chart_faces)
        rows2, cols2 = [], []
        for fl in ef.values():
            if len(fl) == 2 and cc[fl[0]] == cc[fl[1]]:
                rows2.extend([fl[0], fl[1]])
                cols2.extend([fl[1], fl[0]])
        adj2 = csr_matrix((np.ones(len(rows2), dtype=np.int8), (rows2, cols2)), shape=(F, F))
        nc, cc = connected_components(adj2, directed=False)

    out = np.zeros(F, dtype=np.int32)
    next_id = 0
    for c in range(nc):
        mask = cc == c
        cnt = int(np.sum(mask))
        if cnt > max_chart_faces:
            sub_f_g = faces[mask]
            u = np.unique(sub_f_g)
            rm = np.full(len(verts), -1, dtype=np.int64)
            rm[u] = np.arange(len(u), dtype=np.int64)
            k = max(2, min(8, cnt // max_chart_faces))
            sub_ch = _partition_component_charts(
                verts[u],
                rm[sub_f_g],
                fn[mask],
                target_charts=k,
                crease_deg=crease_deg,
                min_chart_faces=min_chart_faces,
            )
            out[mask] = next_id + sub_ch
            next_id += int(sub_ch.max()) + 1
        else:
            out[mask] = next_id
            next_id += 1
    return out


def _align_uv_to_2d(uv: Any, target_2d: Any) -> Any:
    """2D Procrustes rotation (zero reflection) aligning UV upright onto 2D face coords."""
    import numpy as np

    u_c = uv - uv.mean(axis=0)
    t_c = target_2d - target_2d.mean(axis=0)
    h = u_c.T @ t_c
    U, _, Vt = np.linalg.svd(h)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    aligned = u_c @ R
    return aligned - aligned.min(axis=0)


def _floater_mvc_disk(verts: Any, faces: Any) -> Any | None:
    """Bijective Mean Value Coordinates harmonic map for simple single-loop disk charts."""
    import numpy as np
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    ef = _build_edge_face_map(faces)
    bnd_adj: dict[int, list[int]] = {}
    for tri in faces:
        for j in range(3):
            a, b = int(tri[j]), int(tri[(j + 1) % 3])
            k = (a, b) if a < b else (b, a)
            if len(ef[k]) == 1:
                bnd_adj.setdefault(a, []).append(b)
                bnd_adj.setdefault(b, []).append(a)
    if not bnd_adj or any(len(v) != 2 for v in bnd_adj.values()):
        return None
    start = next(iter(bnd_adj))
    loop = [start]
    prev = start
    cur = bnd_adj[start][0]
    visited = {start}
    while cur != start:
        if cur in visited:
            return None
        visited.add(cur)
        loop.append(cur)
        n0, n1 = bnd_adj[cur]
        nxt = n1 if n0 == prev else n0
        prev, cur = cur, nxt
    if len(loop) != len(bnd_adj):
        return None

    fn = _compute_face_normals_uv(verts, faces)
    mean_n = fn.mean(axis=0)
    centered = verts - verts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    is_hero_front = float(mean_n[2]) > 0.35
    proj_2d = verts[:, :2] if is_hero_front else (centered @ vt[:2].T)

    p_loop = proj_2d[loop]
    area_2d = sum(
        p_loop[i, 0] * p_loop[(i + 1) % len(loop), 1] - p_loop[(i + 1) % len(loop), 0] * p_loop[i, 1]
        for i in range(len(loop))
    )
    if area_2d < 0:
        loop = loop[::-1]
        p_loop = proj_2d[loop]

    pts_loop = verts[loop]
    cx, cy = float(proj_2d[:, 0].mean()), float(proj_2d[:, 1].mean())
    angs = np.arctan2(p_loop[:, 1] - cy, p_loop[:, 0] - cx)
    lens = np.linalg.norm(np.roll(pts_loop, -1, axis=0) - pts_loop, axis=1)
    tot = max(float(np.sum(lens)), 1e-12)
    cum = np.concatenate([[0.0], np.cumsum(lens[:-1])])
    thetas = 2.0 * math.pi * (cum / tot)
    thetas += float(np.mean(np.unwrap(angs - thetas)))
    rx = max(float(proj_2d[:, 0].max() - proj_2d[:, 0].min()), 1e-4)
    ry = max(float(proj_2d[:, 1].max() - proj_2d[:, 1].min()), 1e-4)
    asp = float(np.clip(ry / rx, 0.55, 1.80))
    if asp >= 1.0:
        ux, uy = 0.45 / asp, 0.45
    else:
        ux, uy = 0.45, 0.45 * asp
    uv_bnd = np.column_stack([0.5 + ux * np.cos(thetas), 0.5 + uy * np.sin(thetas)])

    nv = len(verts)
    A = lil_matrix((nv, nv), dtype=np.float64)
    b_rhs = np.zeros((nv, 2), dtype=np.float64)
    bnd_idx = {vi: idx for idx, vi in enumerate(loop)}
    for vi, idx in bnd_idx.items():
        A[vi, vi] = 1.0
        b_rhs[vi] = uv_bnd[idx]

    nbr_w: list[dict[int, float]] = [{} for _ in range(nv)]
    for a, b, c in faces:
        for p, q, r in ((int(a), int(b), int(c)), (int(b), int(c), int(a)), (int(c), int(a), int(b))):
            if p in bnd_idx:
                continue
            eq, er = verts[q] - verts[p], verts[r] - verts[p]
            lq = max(float(np.linalg.norm(eq)), 1e-12)
            lr = max(float(np.linalg.norm(er)), 1e-12)
            ca = float(np.clip(np.dot(eq, er) / (lq * lr), -0.999999, 0.999999))
            th = math.sqrt(max(0.0, (1.0 - ca) / (1.0 + ca)))
            nbr_w[p][q] = nbr_w[p].get(q, 0.0) + th / lq
            nbr_w[p][r] = nbr_w[p].get(r, 0.0) + th / lr

    for vi in range(nv):
        if vi in bnd_idx:
            continue
        wsum = sum(nbr_w[vi].values())
        if wsum <= 1e-12:
            A[vi, vi] = 1.0
            b_rhs[vi] = [0.5, 0.5]
        else:
            A[vi, vi] = wsum
            for nj, w in nbr_w[vi].items():
                A[vi, nj] = -w

    uv = spsolve(A.tocsr(), b_rhs)
    if not np.all(np.isfinite(uv)):
        return None
    if is_hero_front:
        uv = _align_uv_to_2d(uv, proj_2d)
    pu = uv[faces]
    a2 = (pu[:, 1, 0] - pu[:, 0, 0]) * (pu[:, 2, 1] - pu[:, 0, 1]) - (
        pu[:, 1, 1] - pu[:, 0, 1]
    ) * (pu[:, 2, 0] - pu[:, 0, 0])
    if float(np.sum(a2)) < 0.0:
        uv[:, 0] = -uv[:, 0]
    uv -= uv.min(axis=0)
    span = max(float(np.max(uv.max(axis=0))), 1e-8)
    uv = 0.02 + 0.96 * (uv / span)
    return uv


def _area_distortion_ratio(verts: Any, faces: Any, uv: Any) -> float:
    """Return P95/P10 ratio of (UV area / 3D area) to catch severe chord-streaked mappings."""
    import numpy as np

    if len(faces) < 4:
        return 1.0
    p0, p1, p2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    a3 = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    pu = uv[faces]
    a2 = 0.5 * np.abs(
        (pu[:, 1, 0] - pu[:, 0, 0]) * (pu[:, 2, 1] - pu[:, 0, 1])
        - (pu[:, 1, 1] - pu[:, 0, 1]) * (pu[:, 2, 0] - pu[:, 0, 0])
    )
    valid = a3 > 1e-12
    if not np.any(valid):
        return 1.0
    r = a2[valid] / a3[valid]
    return float(np.percentile(r, 95) / max(float(np.percentile(r, 10)), 1e-12))


def _eval_chart_uv_quality(uv: Any, faces: Any, *, tex_size: int = 512) -> tuple[int, int, float]:
    import numpy as np

    from quadvox.s2_retopo.bake_xatlas import compute_uv_raster_stats

    if len(uv) == 0 or len(faces) == 0:
        return 0, 0, 0.0
    umin, umax = uv.min(axis=0), uv.max(axis=0)
    span = max(float(np.max(umax - umin)), 1e-8)
    uv_n = 0.02 + 0.96 * ((uv - umin) / span)
    pu = uv_n[faces]
    a2 = (pu[:, 1, 0] - pu[:, 0, 0]) * (pu[:, 2, 1] - pu[:, 0, 1]) - (
        pu[:, 1, 1] - pu[:, 0, 1]
    ) * (pu[:, 2, 0] - pu[:, 0, 0])
    pos_cnt = int(np.sum(a2 > 1e-10))
    neg_cnt = int(np.sum(a2 < -1e-10))
    flips = min(pos_cnt, neg_cnt)
    st = compute_uv_raster_stats(uv_n, faces, tex_size=tex_size)
    return int(st["uv_overlap_count"]), flips, float(st["uv_coverage"])


def _eval_chart_uv_quality_safe(uv: Any, faces: Any, *, tex_size: int = 512) -> tuple[int, int, float, bool]:
    """Evaluate chart UV quality while catching zero-area/collapsed SLIM outputs."""
    import numpy as np

    if len(uv) == 0 or len(faces) == 0 or not np.all(np.isfinite(uv)):
        return 999999, max(1, len(faces)), 0.0, True
    umin, umax = uv.min(axis=0), uv.max(axis=0)
    span = float(np.max(umax - umin))
    if span < 1e-5:
        return 999999, max(1, len(faces)), 0.0, True
    pu_raw = uv[faces]
    a2_raw = (pu_raw[:, 1, 0] - pu_raw[:, 0, 0]) * (pu_raw[:, 2, 1] - pu_raw[:, 0, 1]) - (
        pu_raw[:, 1, 1] - pu_raw[:, 0, 1]
    ) * (pu_raw[:, 2, 0] - pu_raw[:, 0, 0])
    area_uv_raw = 0.5 * float(np.sum(np.abs(a2_raw)))
    if area_uv_raw <= 1e-10:
        return 999999, max(1, len(faces)), 0.0, True
    ov, neg, cov = _eval_chart_uv_quality(uv, faces, tex_size=tex_size)
    if cov < 0.002:
        return 999999, max(neg, 1), cov, True
    return ov, neg, cov, False


_BLENDER_SLIM_WORKER_SOURCE = r"""
import sys, bmesh, bpy, numpy as np

in_npz = sys.argv[sys.argv.index("--") + 1]
out_npz = sys.argv[sys.argv.index("--") + 2]
data = np.load(in_npz, allow_pickle=False)
n_charts = int(data["n_charts"])

bpy.ops.wm.read_factory_settings(use_empty=True)
out_payload = {"n_charts": n_charts}

def extract_uvs(me):
    uvl = me.uv_layers.active.data
    uv_vid_map = {}
    new_verts_idx = []
    new_uvs = []
    new_faces = []
    for p in me.polygons:
        tri = []
        for vi, li in zip(p.vertices, p.loop_indices):
            u, v = uvl[li].uv[:]
            if not (np.isfinite(u) and np.isfinite(v)):
                u, v = 0.0, 0.0
            key = (int(vi), round(float(u), 8), round(float(v), 8))
            if key not in uv_vid_map:
                uv_vid_map[key] = len(uv_vid_map)
                new_verts_idx.append(int(vi))
                new_uvs.append([float(u), float(v)])
            tri.append(uv_vid_map[key])
        new_faces.append(tri)
    return (
        np.asarray(new_verts_idx, dtype=np.int64),
        np.asarray(new_uvs, dtype=np.float64),
        np.asarray(new_faces, dtype=np.int64),
    )

for i in range(n_charts):
    verts = data[f"v_{i}"]
    faces = data[f"f_{i}"]
    seams = data[f"s_{i}"]

    for tag, method in (("slim", "MINIMUM_STRETCH"), ("abf", "ANGLE_BASED")):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        me = bpy.data.meshes.new(f"c_{i}")
        me.from_pydata(verts.tolist(), [], faces.tolist())
        me.update()
        obj = bpy.data.objects.new(f"c_{i}", me)
        bpy.context.scene.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        if len(seams) > 0:
            bm = bmesh.new()
            bm.from_mesh(me)
            bm.edges.ensure_lookup_table()
            ss = set((min(int(a), int(b)), max(int(a), int(b))) for a, b in seams)
            for e in bm.edges:
                k = (min(e.verts[0].index, e.verts[1].index), max(e.verts[0].index, e.verts[1].index))
                if k in ss:
                    e.seam = True
            bm.to_mesh(me)
            bm.free()

        me.uv_layers.new(name="UVMap")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        kw = dict(method=method, fill_holes=True, correct_aspect=True, no_flip=True, margin=0.002)
        if method == "MINIMUM_STRETCH":
            kw["iterations"] = 25
        bpy.ops.uv.unwrap(**kw)
        bpy.ops.object.mode_set(mode="OBJECT")

        vi, uv, of = extract_uvs(me)
        out_payload[f"{tag}_vi_{i}"] = vi
        out_payload[f"{tag}_uv_{i}"] = uv
        out_payload[f"{tag}_of_{i}"] = of

np.savez(out_npz, **out_payload)
"""


def _prepare_semantic_part_charts(
    verts: Any,
    normals: Any,
    faces: Any,
    cid: int,
    *,
    is_prop: bool = False,
    crease_angle_deg: float = 48.0,
) -> list[dict[str, Any]]:
    """Build low-island anatomical/crease-aware charts with cut-graph seams per class."""
    import numpy as np

    charts: list[dict[str, Any]] = []
    for cv, cn, cf_orig, cf_ori, c_flp in _extract_manifold_uv_components(verts, normals, faces):
        fn = _compute_face_normals_uv(cv, cf_ori)
        bmin, bmax = cv.min(axis=0), cv.max(axis=0)
        ext = np.maximum(bmax - bmin, 1e-6)
        ctr = 0.5 * (bmin + bmax)
        cef = _build_edge_face_map(cf_ori)
        chi = len(cv) - len(cef) + len(cf_ori)
        nbl, loops = _compute_boundary_loops(cf_ori)

        if cid == 1 and not is_prop:
            # Tiny synthetic head caps (<200 faces) retain uniform 2.25x head boost.
            if len(cf_ori) < 200:
                seams = _topological_cut_graph_seams(cv, cf_ori, fn, prefer_back=True)
                charts.append(
                    {
                        "cid": cid,
                        "is_hero": True,
                        "boost_cid": 1,
                        "verts": cv,
                        "normals": cn,
                        "orig_faces": cf_orig,
                        "faces": cf_ori,
                        "was_flipped": c_flp,
                        "seams": seams,
                        "allow_split": True,
                    }
                )
                continue

            back_frac = float(np.mean(fn[:, 2] < -0.25))
            if back_frac > 0.12 and (ctr[2] - bmin[2]) > 0.04 and len(cf_ori) > 500:
                z_cut = ctr[2] + 0.01 * ext[2]
                face_c = cv[cf_ori].mean(axis=1)
                zone = np.where(
                    face_c[:, 2] >= z_cut,
                    1,
                    np.where(face_c[:, 0] >= ctr[0], 2, 3),
                ).astype(np.int32)
                zone = _clean_face_labels_iterative(cf_ori, zone, min_faces=24)
                for sid in sorted(np.unique(zone).tolist()):
                    scf_raw = cf_orig[zone == sid]
                    su = np.unique(scf_raw)
                    srm = np.full(len(cv), -1, dtype=np.int64)
                    srm[su] = np.arange(len(su), dtype=np.int64)
                    for scv, scn, scf_orig, scf_ori, sflp in _extract_manifold_uv_components(cv[su], cn[su], srm[scf_raw]):
                        if sid == 1 and len(scf_ori) > 200:
                            for pv, pn, pf_orig, pf_ori, p_flp, is_main in _peel_minor_holes_to_outer(scv, scn, scf_orig):
                                pfn = _compute_face_normals_uv(pv, pf_ori)
                                pseams = _topological_cut_graph_seams(pv, pf_ori, pfn, prefer_back=True)
                                charts.append(
                                    {
                                        "cid": cid,
                                        "is_hero": is_main,
                                        "boost_cid": 0,
                                        "verts": pv,
                                        "normals": pn,
                                        "orig_faces": pf_orig,
                                        "faces": pf_ori,
                                        "was_flipped": p_flp,
                                        "seams": pseams,
                                        "allow_split": not is_main,
                                    }
                                )
                        else:
                            sfn = _compute_face_normals_uv(scv, scf_ori)
                            sseams = _topological_cut_graph_seams(scv, scf_ori, sfn, prefer_back=True)
                            charts.append(
                                {
                                    "cid": cid,
                                    "is_hero": False,
                                    "boost_cid": 0,
                                    "verts": scv,
                                    "normals": scn,
                                    "orig_faces": scf_orig,
                                    "faces": scf_ori,
                                    "was_flipped": sflp,
                                    "seams": sseams,
                                    "allow_split": True,
                                }
                            )
            else:
                if len(cf_ori) > 80 and float(fn[:, 2].mean()) > 0.0:
                    is_front = _clean_face_labels_iterative(cf_ori, (fn[:, 2] >= 0.05).astype(np.int32), min_faces=12)
                    for sid in sorted(np.unique(is_front).tolist()):
                        scf_raw = cf_orig[is_front == sid]
                        su = np.unique(scf_raw)
                        srm = np.full(len(cv), -1, dtype=np.int64)
                        srm[su] = np.arange(len(su), dtype=np.int64)
                        for scv, scn, scf_orig, scf_ori, sflp in _extract_manifold_uv_components(cv[su], cn[su], srm[scf_raw]):
                            if sid == 1 and len(scf_ori) > 80:
                                for pv, pn, pf_orig, pf_ori, p_flp, is_main in _peel_minor_holes_to_outer(scv, scn, scf_orig):
                                    pfn = _compute_face_normals_uv(pv, pf_ori)
                                    pseams = _topological_cut_graph_seams(pv, pf_ori, pfn, prefer_back=True)
                                    charts.append(
                                        {
                                            "cid": cid,
                                            "is_hero": is_main,
                                            "boost_cid": 0,
                                            "verts": pv,
                                            "normals": pn,
                                            "orig_faces": pf_orig,
                                            "faces": pf_ori,
                                            "was_flipped": p_flp,
                                            "seams": pseams,
                                            "allow_split": not is_main,
                                        }
                                    )
                            else:
                                sfn = _compute_face_normals_uv(scv, scf_ori)
                                sseams = _topological_cut_graph_seams(scv, scf_ori, sfn, prefer_back=True)
                                charts.append(
                                    {
                                        "cid": cid,
                                        "is_hero": False,
                                        "boost_cid": 0,
                                        "verts": scv,
                                        "normals": scn,
                                        "orig_faces": scf_orig,
                                        "faces": scf_ori,
                                        "was_flipped": sflp,
                                        "seams": sseams,
                                        "allow_split": True,
                                    }
                                )
                else:
                    seams = _topological_cut_graph_seams(cv, cf_ori, fn, prefer_back=True)
                    charts.append(
                        {
                            "cid": cid,
                            "is_hero": False,
                            "boost_cid": 0,
                            "verts": cv,
                            "normals": cn,
                            "orig_faces": cf_orig,
                            "faces": cf_ori,
                            "was_flipped": c_flp,
                            "seams": seams,
                            "allow_split": True,
                        }
                    )
            continue

        if is_prop or cid == 3:
            sub_lbl = _partition_prop_planar_charts(
                cv, cf_ori, fn, crease_deg=float(crease_angle_deg), max_chart_faces=600, min_chart_faces=24
            )
            for sid in sorted(np.unique(sub_lbl).tolist()):
                mask_s = sub_lbl == sid
                scf_raw = cf_orig[mask_s]
                su = np.unique(scf_raw)
                srm = np.full(len(cv), -1, dtype=np.int64)
                srm[su] = np.arange(len(su), dtype=np.int64)
                for scv, scn, scf_orig, scf_ori, sflp in _extract_manifold_uv_components(cv[su], cn[su], srm[scf_raw]):
                    sfn = _compute_face_normals_uv(scv, scf_ori)
                    sseams = _topological_cut_graph_seams(scv, scf_ori, sfn, prefer_back=False)
                    charts.append(
                        {
                            "cid": cid,
                            "is_hero": False,
                            "boost_cid": cid,
                            "verts": scv,
                            "normals": scn,
                            "orig_faces": scf_orig,
                            "faces": scf_ori,
                            "was_flipped": sflp,
                            "seams": sseams,
                            "allow_split": True,
                        }
                    )
            continue

        if cid == 2:
            target_k = max(2, min(14, len(cf_ori) // 900)) if len(cf_ori) > 400 else 1
            if target_k > 1 or chi < 0:
                sub_lbl = _partition_component_charts(cv, cf_ori, fn, target_charts=max(target_k, 4), crease_deg=48.0, min_chart_faces=36)
                for sid in sorted(np.unique(sub_lbl).tolist()):
                    scf_raw = cf_orig[sub_lbl == sid]
                    su = np.unique(scf_raw)
                    srm = np.full(len(cv), -1, dtype=np.int64)
                    srm[su] = np.arange(len(su), dtype=np.int64)
                    for scv, scn, scf_orig, scf_ori, sflp in _extract_manifold_uv_components(cv[su], cn[su], srm[scf_raw]):
                        sfn = _compute_face_normals_uv(scv, scf_ori)
                        sseams = _topological_cut_graph_seams(scv, scf_ori, sfn, prefer_back=False)
                        charts.append(
                            {
                                "cid": cid,
                                "is_hero": False,
                                "boost_cid": cid,
                                "verts": scv,
                                "normals": scn,
                                "orig_faces": scf_orig,
                                "faces": scf_ori,
                                "was_flipped": sflp,
                                "seams": sseams,
                                "allow_split": True,
                            }
                        )
            else:
                charts.append(
                    {
                        "cid": cid,
                        "is_hero": False,
                        "boost_cid": cid,
                        "verts": cv,
                        "normals": cn,
                        "orig_faces": cf_orig,
                        "faces": cf_ori,
                        "was_flipped": c_flp,
                        "seams": [],
                        "allow_split": True,
                    }
                )
            continue

        if cid == 4:
            target_k = max(2, min(10, len(cf_ori) // 1300)) if len(cf_ori) > 500 else 1
            if target_k > 1 or chi < 0:
                sub_lbl = _partition_component_charts(cv, cf_ori, fn, target_charts=max(target_k, 8), crease_deg=48.0, min_chart_faces=36)
                for sid in sorted(np.unique(sub_lbl).tolist()):
                    scf_raw = cf_orig[sub_lbl == sid]
                    su = np.unique(scf_raw)
                    srm = np.full(len(cv), -1, dtype=np.int64)
                    srm[su] = np.arange(len(su), dtype=np.int64)
                    for scv, scn, scf_orig, scf_ori, sflp in _extract_manifold_uv_components(cv[su], cn[su], srm[scf_raw]):
                        sfn = _compute_face_normals_uv(scv, scf_ori)
                        sseams = _topological_cut_graph_seams(scv, scf_ori, sfn, prefer_back=True)
                        charts.append(
                            {
                                "cid": cid,
                                "is_hero": False,
                                "boost_cid": cid,
                                "verts": scv,
                                "normals": scn,
                                "orig_faces": scf_orig,
                                "faces": scf_ori,
                                "was_flipped": sflp,
                                "seams": sseams,
                                "allow_split": True,
                            }
                        )
            else:
                charts.append(
                    {
                        "cid": cid,
                        "is_hero": False,
                        "boost_cid": cid,
                        "verts": cv,
                        "normals": cn,
                        "orig_faces": cf_orig,
                        "faces": cf_ori,
                        "was_flipped": c_flp,
                        "seams": [],
                        "allow_split": True,
                    }
                )
            continue

        if cid == 5:
            face_c = cv[cf_ori].mean(axis=1)
            side_lbl = _clean_face_labels_iterative(cf_ori, (face_c[:, 2] >= ctr[2]).astype(np.int32), min_faces=24)
            for sid in sorted(np.unique(side_lbl).tolist()):
                scf_raw = cf_orig[side_lbl == sid]
                su = np.unique(scf_raw)
                srm = np.full(len(cv), -1, dtype=np.int64)
                srm[su] = np.arange(len(su), dtype=np.int64)
                for scv, scn, scf_orig, scf_ori, sflp in _extract_manifold_uv_components(cv[su], cn[su], srm[scf_raw]):
                    sfn = _compute_face_normals_uv(scv, scf_ori)
                    sseams = _topological_cut_graph_seams(scv, scf_ori, sfn, prefer_back=False)
                    charts.append(
                        {
                            "cid": cid,
                            "is_hero": False,
                            "boost_cid": cid,
                            "verts": scv,
                            "normals": scn,
                            "orig_faces": scf_orig,
                            "faces": scf_ori,
                            "was_flipped": sflp,
                            "seams": sseams,
                            "allow_split": True,
                        }
                    )
            continue

        if cid == 6:
            if len(cf_ori) > 3500:
                sub_lbl = _partition_component_charts(cv, cf_ori, fn, target_charts=6, crease_deg=48.0, min_chart_faces=32)
                for sid in sorted(np.unique(sub_lbl).tolist()):
                    scf_raw = cf_orig[sub_lbl == sid]
                    su = np.unique(scf_raw)
                    srm = np.full(len(cv), -1, dtype=np.int64)
                    srm[su] = np.arange(len(su), dtype=np.int64)
                    for scv, scn, scf_orig, scf_ori, sflp in _extract_manifold_uv_components(cv[su], cn[su], srm[scf_raw]):
                        sfn = _compute_face_normals_uv(scv, scf_ori)
                        sseams = _topological_cut_graph_seams(scv, scf_ori, sfn, prefer_back=True)
                        charts.append(
                            {
                                "cid": cid,
                                "is_hero": False,
                                "boost_cid": cid,
                                "verts": scv,
                                "normals": scn,
                                "orig_faces": scf_orig,
                                "faces": scf_ori,
                                "was_flipped": sflp,
                                "seams": sseams,
                                "allow_split": True,
                            }
                        )
            else:
                ew: dict[tuple[int, int], float] = {}
                is_arm = ext[0] > 1.35 * ext[1]
                for k in cef:
                    mid = 0.5 * (cv[k[0]] + cv[k[1]])
                    elen = float(np.linalg.norm(cv[k[1]] - cv[k[0]]))
                    if is_arm:
                        ew[k] = elen * (
                            1.0
                            + 30.0 * max(0.0, mid[1] - bmin[1]) / ext[1]
                            + 5.0 * max(0.0, mid[2] - ctr[2]) / ext[2]
                        )
                    else:
                        ew[k] = elen * (
                            1.0
                            + 25.0 * abs(mid[0] - ctr[0]) / ext[0]
                            + 15.0 * max(0.0, mid[2] - ctr[2]) / ext[2]
                        )
                if chi == 0 and nbl == 2:
                    seams = _dijkstra_seam_path(cv, ew, loops[0], loops[1])
                else:
                    if is_arm:
                        s_v = min(range(len(cv)), key=lambda i: abs(cv[i, 0]) if abs(ctr[0]) > 0.05 else cv[i, 0])
                        e_v = max(range(len(cv)), key=lambda i: abs(cv[i, 0]) if abs(ctr[0]) > 0.05 else cv[i, 0])
                    else:
                        s_v = max(range(len(cv)), key=lambda i: cv[i, 1] - 0.5 * abs(cv[i, 0] - ctr[0]))
                        e_v = min(range(len(cv)), key=lambda i: cv[i, 1] + 0.5 * abs(cv[i, 0] - ctr[0]))
                    seams = _dijkstra_seam_path(cv, ew, [s_v], [e_v])
                    if chi != 2 - nbl:
                        seams = list(set(seams) | set(_topological_cut_graph_seams(cv, cf_ori, fn, prefer_back=True)))
                seams = _prune_confetti_seams(cf_ori, seams)
                charts.append(
                    {
                        "cid": cid,
                        "is_hero": False,
                        "boost_cid": cid,
                        "verts": cv,
                        "normals": cn,
                        "orig_faces": cf_orig,
                        "faces": cf_ori,
                        "was_flipped": c_flp,
                        "seams": seams,
                        "allow_split": True,
                    }
                )
            continue

        if cid == 7:
            is_sole = (cv[cf_ori].mean(axis=1)[:, 1] < (bmin[1] + 0.14 * ext[1])) & (fn[:, 1] < -0.45)
            sole_lbl = _clean_face_labels_iterative(cf_ori, is_sole.astype(np.int32), min_faces=12)
            for sid in sorted(np.unique(sole_lbl).tolist()):
                scf_raw = cf_orig[sole_lbl == sid]
                su = np.unique(scf_raw)
                srm = np.full(len(cv), -1, dtype=np.int64)
                srm[su] = np.arange(len(su), dtype=np.int64)
                for scv, scn, scf_orig, scf_ori, sflp in _extract_manifold_uv_components(cv[su], cn[su], srm[scf_raw]):
                    sfn = _compute_face_normals_uv(scv, scf_ori)
                    sbmin, sbmax = scv.min(axis=0), scv.max(axis=0)
                    sext = np.maximum(sbmax - sbmin, 1e-6)
                    sctr = 0.5 * (sbmin + sbmax)
                    scef = _build_edge_face_map(scf_ori)
                    if sid == 0 and len(scf_ori) > 100:
                        ew_s: dict[tuple[int, int], float] = {}
                        for k in scef:
                            mid = 0.5 * (scv[k[0]] + scv[k[1]])
                            elen = float(np.linalg.norm(scv[k[1]] - scv[k[0]]))
                            front_pen = 200.0 if mid[2] > sctr[2] else 0.0
                            ew_s[k] = elen * (1.0 + 25.0 * abs(mid[0] - sctr[0]) / sext[0] + front_pen)
                        back_vs = [i for i in range(len(scv)) if scv[i, 2] <= sctr[2]] or list(range(len(scv)))
                        top_v = max(back_vs, key=lambda i: scv[i, 1] - 2.0 * abs(scv[i, 0] - sctr[0]))
                        bot_v = min(back_vs, key=lambda i: scv[i, 1] + 2.0 * abs(scv[i, 0] - sctr[0]))
                        heel_seams = _dijkstra_seam_path(scv, ew_s, [top_v], [bot_v])
                    else:
                        heel_seams = []
                    topo_seams = _topological_cut_graph_seams(scv, scf_ori, sfn, prefer_back=True)
                    all_s = _prune_confetti_seams(scf_ori, list(set(heel_seams) | set(topo_seams)))
                    charts.append(
                        {
                            "cid": cid,
                            "is_hero": False,
                            "boost_cid": cid,
                            "verts": scv,
                            "normals": scn,
                            "orig_faces": scf_orig,
                            "faces": scf_ori,
                            "was_flipped": sflp,
                            "seams": all_s,
                            "allow_split": True,
                        }
                    )
            continue

        topo_seams = _topological_cut_graph_seams(cv, cf_ori, fn, prefer_back=False)
        charts.append(
            {
                "cid": cid,
                "is_hero": False,
                "boost_cid": cid,
                "verts": cv,
                "normals": cn,
                "orig_faces": cf_orig,
                "faces": cf_ori,
                "was_flipped": c_flp,
                "seams": topo_seams,
                "allow_split": True,
            }
        )

    # Ensure strictly ONE continuous front hero face chart gets boost_cid=1 when head_face is split vs back scalp.
    if cid == 1 and not is_prop and charts and sum(len(c["faces"]) for c in charts) >= 200:
        hero_cands = [ch for ch in charts if ch["is_hero"]]
        best_hero = max(hero_cands or charts, key=lambda ch: len(ch["faces"]))
        best_hero["boost_cid"] = 1
        best_hero["is_hero"] = True
        for ch in charts:
            if ch is not best_hero:
                ch["boost_cid"] = 0
                ch["is_hero"] = False
    return charts


def _run_blender_slim_batch(
    charts: list[dict[str, Any]],
    work_dir: Path,
    blender_bin: Path,
    *,
    allow_mvc_fallback: bool = False,
    check_zero_area: bool = False,
) -> list[dict[str, Any]]:
    import numpy as np

    if not charts:
        return []
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = work_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    in_npz = work_dir / "_in_charts.npz"
    out_npz = work_dir / "_out_charts.npz"
    script_py = work_dir / "_blender_uv_worker.py"
    script_py.write_text(_BLENDER_SLIM_WORKER_SOURCE, encoding="utf-8")

    in_payload: dict[str, Any] = {"n_charts": np.array(len(charts), dtype=np.int32)}
    for i, ch in enumerate(charts):
        in_payload[f"v_{i}"] = ch["verts"]
        in_payload[f"f_{i}"] = ch["faces"]
        in_payload[f"s_{i}"] = (
            np.asarray(ch["seams"], dtype=np.int64).reshape(-1, 2)
            if ch["seams"]
            else np.zeros((0, 2), dtype=np.int64)
        )
    np.savez(in_npz, **in_payload)

    cmd = [
        str(blender_bin),
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(script_py),
        "--",
        str(in_npz),
        str(out_npz),
    ]
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)
    env["TMP"] = str(tmp_dir)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    if proc.returncode != 0 or not out_npz.exists():
        raise RuntimeError(
            f"Blender SLIM UV worker failed (rc={proc.returncode}):\n"
            f"{proc.stderr[-1500:]}\n{proc.stdout[-1500:]}"
        )

    out = np.load(out_npz, allow_pickle=False)
    unwrapped: list[dict[str, Any]] = []
    for i, ch in enumerate(charts):
        best_cand = None
        best_score = (10**13, 10**13, 0.0)
        for tag in ("slim", "abf"):
            vi = out[f"{tag}_vi_{i}"]
            uv = out[f"{tag}_uv_{i}"]
            of = out[f"{tag}_of_{i}"]
            if check_zero_area:
                ov, neg, cov, z_area = _eval_chart_uv_quality_safe(uv, of)
            else:
                ov, neg, cov = _eval_chart_uv_quality(uv, of)
                z_area = False
            score = (ov + neg * 100, neg, -cov)
            if score < best_score:
                best_score = score
                best_cand = (vi, uv, of, ov, neg, z_area)

        vi, uv, of, ov, neg, z_area = best_cand  # type: ignore[misc]
        v_cut = ch["verts"][vi]
        n_cut = ch["normals"][vi]
        has_seams = bool(ch.get("seams") or ch.get("has_seams", False))
        can_mvc = (not has_seams) or check_zero_area
        if (ov > 0 or neg > 0 or z_area) and can_mvc and (ch.get("is_hero", False) or allow_mvc_fallback) and len(of) >= 3:
            uv_parts = _split_uv_connected_islands(v_cut, n_cut, uv, of, ch["cid"], 1.0)
            if len(uv_parts) == 1:
                onbl, _ = _compute_boundary_loops(of)
                ochi = len(v_cut) - len(_build_edge_face_map(of)) + len(of)
                if onbl == 1 and ochi == 1:
                    mvc_uv = _floater_mvc_disk(v_cut, of)
                    if mvc_uv is not None:
                        if check_zero_area:
                            m_ov, m_neg, _, m_z = _eval_chart_uv_quality_safe(mvc_uv, of)
                        else:
                            m_ov, m_neg, _ = _eval_chart_uv_quality(mvc_uv, of)
                            m_z = False
                        dist_lim = 65.0 if ch.get("is_hero", False) else (80.0 if check_zero_area else 10.0)
                        if (not m_z) and (m_ov + m_neg * 100) < (ov + neg * 100) and _area_distortion_ratio(v_cut, of, mvc_uv) <= dist_lim:
                            uv = mvc_uv
                            ov, neg, z_area = m_ov, m_neg, m_z

        unwrapped.append(
            {
                "cid": ch["cid"],
                "boost_cid": ch.get("boost_cid", ch["cid"]),
                "is_hero": ch.get("is_hero", False),
                "has_seams": has_seams,
                "verts": v_cut,
                "normals": n_cut,
                "faces": of,
                "uvs": uv,
                "ov512": ov,
                "neg": neg,
                "zero_area": z_area,
                "orig_verts": ch["verts"],
                "orig_normals": ch["normals"],
                "orig_faces": ch["faces"],
                "orig_unflipped_faces": ch.get("orig_faces", ch["faces"]),
                "was_flipped": ch.get("was_flipped"),
                "allow_split": ch.get("allow_split", True),
            }
        )
    return unwrapped


def _apply_safe_mvc_fallback(unwrapped: list[dict[str, Any]], *, allow_seamed: bool = False) -> list[dict[str, Any]]:
    """Use distortion-gated Floater MVC on residual overlapping disk charts (never chord-streaking)."""
    out: list[dict[str, Any]] = []
    for ch in unwrapped:
        ov, neg = ch["ov512"], ch["neg"]
        z_area = bool(ch.get("zero_area", False))
        can_mvc = allow_seamed or (not ch.get("has_seams", False))
        if (ov > 0 or neg > 0 or z_area) and can_mvc and len(ch["faces"]) >= 3:
            v, n, f, uv = ch["verts"], ch["normals"], ch["faces"], ch["uvs"]
            sub_isls = _split_uv_connected_islands(v, n, uv, f, ch["cid"], 1.0)
            if len(sub_isls) == 1 or z_area:
                onbl, _ = _compute_boundary_loops(f)
                ochi = len(v) - len(_build_edge_face_map(f)) + len(f)
                if onbl == 1 and ochi == 1:
                    mvc_uv = _floater_mvc_disk(v, f)
                    if mvc_uv is not None:
                        if allow_seamed:
                            m_ov, m_neg, _, m_z = _eval_chart_uv_quality_safe(mvc_uv, f)
                            dist_lim = 80.0 if not ch.get("is_hero", False) else 100.0
                        else:
                            m_ov, m_neg, _ = _eval_chart_uv_quality(mvc_uv, f)
                            m_z = False
                            dist_lim = 65.0 if ch.get("is_hero", False) else 10.0
                        if (not m_z) and (m_ov + m_neg * 100) < (ov + neg * 100) and _area_distortion_ratio(v, f, mvc_uv) <= dist_lim:
                            ch = dict(ch)
                            ch["uvs"] = mvc_uv
                            ch["ov512"] = m_ov
                            ch["neg"] = m_neg
                            ch["zero_area"] = m_z
        out.append(ch)
    return out


def _refine_overlapping_charts(
    unwrapped_charts: list[dict[str, Any]],
    work_dir: Path,
    blender_bin: Path,
    *,
    max_rounds: int = 3,
    max_total_charts: int = 100,
) -> list[dict[str, Any]]:
    import numpy as np

    current = list(unwrapped_charts)
    for rnd in range(max_rounds):
        good: list[dict[str, Any]] = []
        to_split: list[dict[str, Any]] = []
        ov_thr = 64 if rnd == 0 else (32 if rnd == 1 else 250)
        cands = [
            ch
            for ch in current
            if (ch["ov512"] > ov_thr or ch["neg"] > 2)
            and len(ch["orig_faces"]) >= 32
            and ch.get("allow_split", True)
        ]
        budget = max(0, max_total_charts - len(current))
        chosen_ids = set(
            id(x) for x in sorted(cands, key=lambda c: c["ov512"] + 50 * c["neg"], reverse=True)[:budget]
        )
        for ch in current:
            if id(ch) in chosen_ids:
                to_split.append(ch)
            else:
                good.append(ch)
        if not to_split:
            break
        new_subcharts: list[dict[str, Any]] = []
        for ch in to_split:
            ov, on, of = ch["orig_verts"], ch["orig_normals"], ch.get("orig_unflipped_faces", ch["orig_faces"])
            fn = _compute_face_normals_uv(ov, of)
            lbl = _partition_component_charts(
                ov,
                of,
                fn,
                target_charts=2,
                crease_deg=(44.0 if rnd == 0 else 34.0),
                min_chart_faces=16,
            )
            if len(np.unique(lbl)) <= 1:
                fc = ov[of].mean(axis=1)
                centered = fc - fc.mean(axis=0)
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                proj = centered @ vt[0]
                lbl = (proj >= np.median(proj)).astype(np.int32)
            for sid in sorted(np.unique(lbl).tolist()):
                scf_raw = of[lbl == sid]
                su = np.unique(scf_raw)
                srm = np.full(len(ov), -1, dtype=np.int64)
                srm[su] = np.arange(len(su), dtype=np.int64)
                for scv, scn, scf_orig, scf_ori, sflp in _extract_manifold_uv_components(ov[su], on[su], srm[scf_raw]):
                    sfn = _compute_face_normals_uv(scv, scf_ori)
                    sseams = _topological_cut_graph_seams(scv, scf_ori, sfn, prefer_back=False)
                    new_subcharts.append(
                        {
                            "cid": ch["cid"],
                            "boost_cid": ch["boost_cid"],
                            "is_hero": False,
                            "verts": scv,
                            "normals": scn,
                            "orig_faces": scf_orig,
                            "faces": scf_ori,
                            "was_flipped": sflp,
                            "seams": sseams,
                            "allow_split": True,
                        }
                    )
        re_unwrapped = _run_blender_slim_batch(
            new_subcharts,
            work_dir / f"refine_{rnd}",
            blender_bin,
            allow_mvc_fallback=(rnd + 1 == max_rounds),
        )
        current = good + re_unwrapped
    return _apply_safe_mvc_fallback(current)


def _refine_overlapping_charts_zero_overlap(
    unwrapped_charts: list[dict[str, Any]],
    work_dir: Path,
    blender_bin: Path,
    *,
    max_rounds: int = 6,
    max_total_charts: int = 200,
    min_chart_faces: int = 12,
) -> list[dict[str, Any]]:
    """Iteratively split only overlapping, flipped, or zero-area charts while keeping charts >= min_chart_faces."""
    import numpy as np
    import xatlas

    min_split_input = max(24, 2 * int(min_chart_faces))
    current = list(unwrapped_charts)
    for rnd in range(max_rounds):
        cands = [
            ch
            for ch in current
            if (ch["ov512"] > 0 or ch["neg"] > 0 or ch.get("zero_area", False))
            and len(ch["orig_faces"]) >= min_split_input
            and ch.get("allow_split", True)
        ]
        budget = max(0, int(max_total_charts) - len(current))
        if not cands or budget <= 0:
            break
        cands_sorted = sorted(
            cands,
            key=lambda c: (c["ov512"] + 50 * c["neg"]) * math.sqrt(max(1, len(c["orig_faces"]))),
            reverse=True,
        )
        chosen_ids = set(id(x) for x in cands_sorted[:budget])
        good = [ch for ch in current if id(ch) not in chosen_ids]
        to_split = [ch for ch in current if id(ch) in chosen_ids]
        if not to_split:
            break

        new_subcharts: list[dict[str, Any]] = []
        for ch in to_split:
            ov, on, of = ch["orig_verts"], ch["orig_normals"], ch.get("orig_unflipped_faces", ch["orig_faces"])
            fn = _compute_face_normals_uv(ov, of)
            lbl = _partition_component_charts(
                ov,
                of,
                fn,
                target_charts=2,
                crease_deg=max(18.0, 42.0 - 5.0 * rnd),
                min_chart_faces=int(min_chart_faces),
            )
            if len(np.unique(lbl)) <= 1:
                fc = ov[of].mean(axis=1)
                centered = fc - fc.mean(axis=0)
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                proj = centered @ vt[0]
                lbl = (proj >= np.median(proj)).astype(np.int32)
                if len(np.unique(lbl)) <= 1:
                    lbl = np.arange(len(of)) % 2
            for sid in sorted(np.unique(lbl).tolist()):
                scf_raw = of[lbl == sid]
                su = np.unique(scf_raw)
                srm = np.full(len(ov), -1, dtype=np.int64)
                srm[su] = np.arange(len(su), dtype=np.int64)
                for scv, scn, scf_orig, scf_ori, sflp in _extract_manifold_uv_components(ov[su], on[su], srm[scf_raw]):
                    sfn = _compute_face_normals_uv(scv, scf_ori)
                    sseams = _topological_cut_graph_seams(scv, scf_ori, sfn, prefer_back=False)
                    new_subcharts.append(
                        {
                            "cid": ch["cid"],
                            "boost_cid": ch["boost_cid"],
                            "is_hero": False,
                            "verts": scv,
                            "normals": scn,
                            "orig_faces": scf_orig,
                            "faces": scf_ori,
                            "was_flipped": sflp,
                            "seams": sseams,
                            "allow_split": True,
                        }
                    )
        re_unwrapped = _run_blender_slim_batch(
            new_subcharts,
            work_dir / f"zov_refine_{rnd}",
            blender_bin,
            allow_mvc_fallback=True,
            check_zero_area=True,
        )
        current = good + re_unwrapped

    current = _apply_safe_mvc_fallback(current, allow_seamed=True)

    # Resolve any remaining stubborn overlap/flip/zero-area charts via finer crease split or xatlas fallback.
    clean: list[dict[str, Any]] = []
    stubborn: list[dict[str, Any]] = []
    for ch in current:
        if ch["ov512"] > 0 or ch["neg"] > 0 or ch.get("zero_area", False):
            stubborn.append(ch)
        else:
            clean.append(ch)

    if stubborn:
        finer_subs: list[dict[str, Any]] = []
        direct_xatlas: list[dict[str, Any]] = []
        for ch in stubborn:
            of = ch.get("orig_unflipped_faces", ch["orig_faces"])
            if len(of) >= min_split_input:
                ov, on = ch["orig_verts"], ch["orig_normals"]
                fn = _compute_face_normals_uv(ov, of)
                lbl = _partition_component_charts(
                    ov,
                    of,
                    fn,
                    target_charts=max(2, min(8, len(of) // int(min_chart_faces))),
                    crease_deg=22.0,
                    min_chart_faces=int(min_chart_faces),
                )
                if len(np.unique(lbl)) <= 1:
                    fc = ov[of].mean(axis=1)
                    centered = fc - fc.mean(axis=0)
                    _, _, vt = np.linalg.svd(centered, full_matrices=False)
                    proj = centered @ vt[0]
                    lbl = (proj >= np.median(proj)).astype(np.int32)
                for sid in sorted(np.unique(lbl).tolist()):
                    scf_raw = of[lbl == sid]
                    su = np.unique(scf_raw)
                    srm = np.full(len(ov), -1, dtype=np.int64)
                    srm[su] = np.arange(len(su), dtype=np.int64)
                    for scv, scn, scf_orig, scf_ori, sflp in _extract_manifold_uv_components(ov[su], on[su], srm[scf_raw]):
                        sfn = _compute_face_normals_uv(scv, scf_ori)
                        sseams = _topological_cut_graph_seams(scv, scf_ori, sfn, prefer_back=False)
                        finer_subs.append(
                            {
                                "cid": ch["cid"],
                                "boost_cid": ch["boost_cid"],
                                "is_hero": False,
                                "verts": scv,
                                "normals": scn,
                                "orig_faces": scf_orig,
                                "faces": scf_ori,
                                "was_flipped": sflp,
                                "seams": sseams,
                                "allow_split": True,
                            }
                        )
            else:
                direct_xatlas.append(ch)

        if finer_subs:
            finer_unwrapped = _run_blender_slim_batch(
                finer_subs,
                work_dir / "zov_stubborn_finer",
                blender_bin,
                allow_mvc_fallback=True,
                check_zero_area=True,
            )
            finer_unwrapped = _apply_safe_mvc_fallback(finer_unwrapped, allow_seamed=True)
            for ch in finer_unwrapped:
                if ch["ov512"] > 0 or ch["neg"] > 0 or ch.get("zero_area", False):
                    direct_xatlas.append(ch)
                else:
                    clean.append(ch)

        for ch in direct_xatlas:
            ov, on, of = ch["orig_verts"], ch["orig_normals"], ch["orig_faces"]
            atlas = xatlas.Atlas()
            atlas.add_mesh(ov.astype(np.float32), of.astype(np.uint32), on.astype(np.float32))
            co = xatlas.ChartOptions()
            co.max_cost = 12.0
            co.normal_deviation_weight = 0.4
            atlas.generate(chart_options=co)
            vm, idx_f, uvs = atlas.get_mesh(0)
            vm_i = np.asarray(vm, dtype=np.int64)
            idx_i = np.asarray(idx_f, dtype=np.int64)
            uv_f = np.asarray(uvs, dtype=np.float64)
            for isl in _split_uv_connected_islands(ov[vm_i], on[vm_i], uv_f, idx_i, ch["cid"], 1.0):
                ov_s, neg_s, _, z_s = _eval_chart_uv_quality_safe(isl["uvs"], isl["faces"])
                clean.append(
                    {
                        "cid": ch["cid"],
                        "boost_cid": ch["boost_cid"],
                        "is_hero": False,
                        "has_seams": False,
                        "verts": isl["verts"],
                        "normals": isl["normals"],
                        "faces": isl["faces"],
                        "uvs": isl["uvs"],
                        "ov512": ov_s,
                        "neg": neg_s,
                        "zero_area": z_s,
                        "orig_verts": isl["verts"],
                        "orig_normals": isl["normals"],
                        "orig_faces": isl["faces"],
                        "orig_unflipped_faces": isl["faces"],
                        "was_flipped": None,
                        "allow_split": False,
                    }
                )
    return clean


def _merge_micro_islands_with_validation(
    charts: list[dict[str, Any]],
    weld_verts: Any,
    weld_normals: Any,
    weld_faces: Any,
    *,
    face_scale_boost: float,
    is_prop: bool,
    tex_size: int,
    min_island_px: int,
    work_dir: Path,
    blender_bin: Path,
    min_faces_thr: int = 12,
    max_passes: int = 2,
) -> list[dict[str, Any]]:
    """Absorb <min_faces_thr or <min_island_px^2 micro-islands into adjacent same-semantic islands via SLIM/MVC validation."""
    import numpy as np
    from scipy.spatial import cKDTree

    if len(charts) <= 1 or len(weld_faces) == 0:
        return charts

    weld_centroids = weld_verts[weld_faces].mean(axis=1)
    weld_tree = cKDTree(weld_centroids)

    # Split every chart into UV-connected units so detached micro-scraps are exposed for merging.
    units: list[dict[str, Any]] = []
    for ch in charts:
        v_p, n_p, f_p, uv_p = ch["verts"], ch["normals"], ch["faces"], ch["uvs"]
        if len(f_p) == 0 or len(v_p) == 0:
            continue
        f_out = f_p.copy()
        w_flp = ch.get("was_flipped")
        if w_flp is not None and len(w_flp) == len(f_out) and bool(np.any(w_flp)):
            f_out[w_flp] = f_out[w_flp][:, [0, 2, 1]]
        uv_parts = _split_uv_connected_islands(v_p, n_p, uv_p, f_out, ch["cid"], 1.0)
        for isl in uv_parts:
            iv, inrm, ifc, iuv = isl["verts"], isl["normals"], isl["faces"], isl["uvs"]
            icent = iv[ifc].mean(axis=1)
            _, gf_ids = weld_tree.query(icent, k=1)
            ov, neg, _, z_area = _eval_chart_uv_quality_safe(iuv, ifc)
            units.append(
                {
                    "cid": int(ch["cid"]),
                    "boost_cid": int(ch.get("boost_cid", ch["cid"])),
                    "is_hero": bool(ch.get("is_hero", False)),
                    "verts": iv,
                    "normals": inrm,
                    "faces": ifc,
                    "uvs": iuv,
                    "ov512": ov,
                    "neg": neg,
                    "zero_area": z_area,
                    "area_3d": float(isl["area_3d"]),
                    "gf_ids": np.asarray(gf_ids, dtype=np.int64),
                    "was_flipped": None,
                }
            )

    if len(units) <= 1:
        return units

    ef_weld = _build_edge_face_map(weld_faces)
    edge_pairs: list[tuple[int, int]] = []
    for fl in ef_weld.values():
        if len(fl) == 2:
            edge_pairs.append((int(fl[0]), int(fl[1])))

    min_px_area = float(max(1, int(min_island_px))) ** 2

    for pass_idx in range(max_passes):
        total_w_area = 0.0
        for u in units:
            w = (float(face_scale_boost) ** 2) if (not is_prop and u["boost_cid"] == 1) else 1.0
            total_w_area += u["area_3d"] * w
        total_w_area = max(total_w_area, 1e-12)

        is_micro: list[bool] = []
        for u in units:
            w = (float(face_scale_boost) ** 2) if (not is_prop and u["boost_cid"] == 1) else 1.0
            est_px_area = (u["area_3d"] * w / total_w_area) * (0.45 * float(tex_size) * float(tex_size))
            micro_flag = (not u["is_hero"]) and (len(u["gf_ids"]) < int(min_faces_thr) or est_px_area < min_px_area)
            is_micro.append(micro_flag)

        if not any(is_micro):
            break

        face_to_unit = np.full(len(weld_faces), -1, dtype=np.int64)
        for uid, u in enumerate(units):
            face_to_unit[u["gf_ids"]] = uid

        nbr_edges: dict[tuple[int, int], int] = {}
        for f1, f2 in edge_pairs:
            u1, u2 = int(face_to_unit[f1]), int(face_to_unit[f2])
            if u1 < 0 or u2 < 0 or u1 == u2:
                continue
            nbr_edges[(u1, u2)] = nbr_edges.get((u1, u2), 0) + 1
            nbr_edges[(u2, u1)] = nbr_edges.get((u2, u1), 0) + 1

        adj_by_unit: list[dict[int, int]] = [{} for _ in range(len(units))]
        for (u1, u2), cnt in nbr_edges.items():
            adj_by_unit[u1][u2] = cnt

        redirect: dict[int, int] = {}
        for m_idx in range(len(units)):
            if not is_micro[m_idx]:
                continue
            mu = units[m_idx]
            best_t = -1
            best_key = (-1, -1, -1)
            for t_idx, ecnt in adj_by_unit[m_idx].items():
                tu = units[t_idx]
                if tu["cid"] != mu["cid"]:
                    continue
                if mu["cid"] == 1 and tu["boost_cid"] != mu["boost_cid"] and tu["is_hero"]:
                    continue
                if is_micro[t_idx] and len(tu["gf_ids"]) <= len(mu["gf_ids"]):
                    continue
                pref_non_micro = 0 if is_micro[t_idx] else 1
                cand_key = (pref_non_micro, ecnt, len(tu["gf_ids"]))
                if cand_key > best_key:
                    best_key = cand_key
                    best_t = t_idx
            if best_t >= 0:
                redirect[m_idx] = best_t

        if not redirect:
            break

        groups: dict[int, list[int]] = {}
        for m_idx, t_idx in redirect.items():
            root = t_idx
            seen = {m_idx}
            while root in redirect and root not in seen:
                seen.add(root)
                root = redirect[root]
            if root != m_idx:
                groups.setdefault(root, []).append(m_idx)

        if not groups:
            break

        # On pass 1, if a group has multiple micros and fails, pass 2 will try one-by-one.
        slim_cands: list[dict[str, Any]] = []
        slim_meta: list[tuple[int, list[int], np.ndarray]] = []
        accepted_direct: dict[int, dict[str, Any]] = {}
        consumed_micros: set[int] = set()

        for t_idx, m_list in groups.items():
            # Pass 0 tries full group; pass 1 tries smaller half; pass 2+ tries smallest one-by-one.
            if pass_idx == 0:
                m_use = m_list
            elif pass_idx == 1 and len(m_list) > 2:
                m_sorted = sorted(m_list, key=lambda idx: len(units[idx]["gf_ids"]))
                m_use = m_sorted[: max(1, len(m_sorted) // 2)]
            else:
                m_use = [min(m_list, key=lambda idx: len(units[idx]["gf_ids"]))]
            gf_merged = np.unique(np.concatenate([units[t_idx]["gf_ids"]] + [units[m]["gf_ids"] for m in m_use]))
            su = np.unique(weld_faces[gf_merged])
            srm = np.full(len(weld_verts), -1, dtype=np.int64)
            srm[su] = np.arange(len(su), dtype=np.int64)
            comps = _extract_manifold_uv_components(weld_verts[su], weld_normals[su], srm[weld_faces[gf_merged]])
            if len(comps) != 1:
                # Fallback: try single smallest micro that preserves 1 manifold component
                picked: list[int] = []
                for m_one in sorted(m_use, key=lambda idx: len(units[idx]["gf_ids"])):
                    gf_try = np.unique(np.concatenate([units[t_idx]["gf_ids"], units[m_one]["gf_ids"]]))
                    su_t = np.unique(weld_faces[gf_try])
                    srm_t = np.full(len(weld_verts), -1, dtype=np.int64)
                    srm_t[su_t] = np.arange(len(su_t), dtype=np.int64)
                    c_try = _extract_manifold_uv_components(weld_verts[su_t], weld_normals[su_t], srm_t[weld_faces[gf_try]])
                    if len(c_try) == 1:
                        picked = [m_one]
                        gf_merged = gf_try
                        comps = c_try
                        break
                if not picked:
                    continue
                m_use = picked

            scv, scn, scf_orig, scf_ori, sflp = comps[0]
            sfn = _compute_face_normals_uv(scv, scf_ori)
            prefer_back = (units[t_idx]["cid"] in (1, 4, 6, 7)) and (not is_prop)
            sseams = _topological_cut_graph_seams(scv, scf_ori, sfn, prefer_back=prefer_back)

            # Fast pure-Python MVC path for single-boundary disk patches
            nbl, _ = _compute_boundary_loops(scf_ori)
            chi = len(scv) - len(_build_edge_face_map(scf_ori)) + len(scf_ori)
            if nbl == 1 and chi == 1 and len(scf_ori) >= 3:
                mvc_uv = _floater_mvc_disk(scv, scf_ori)
                if mvc_uv is not None:
                    mov, mneg, _, mz = _eval_chart_uv_quality_safe(mvc_uv, scf_ori)
                    if (not mz) and mov == 0 and mneg == 0 and _area_distortion_ratio(scv, scf_ori, mvc_uv) <= 80.0:
                        f_out = scf_ori.copy()
                        if sflp is not None and len(sflp) == len(f_out) and bool(np.any(sflp)):
                            f_out[sflp] = f_out[sflp][:, [0, 2, 1]]
                        if len(_split_uv_connected_islands(scv, scn, mvc_uv, f_out, units[t_idx]["cid"], 1.0)) == 1:
                            p0, p1, p2 = scv[f_out[:, 0]], scv[f_out[:, 1]], scv[f_out[:, 2]]
                            a3 = 0.5 * float(np.sum(np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)))
                            accepted_direct[t_idx] = {
                                "cid": units[t_idx]["cid"],
                                "boost_cid": units[t_idx]["boost_cid"],
                                "is_hero": units[t_idx]["is_hero"],
                                "verts": scv,
                                "normals": scn,
                                "faces": f_out,
                                "uvs": mvc_uv,
                                "ov512": mov,
                                "neg": mneg,
                                "zero_area": False,
                                "area_3d": a3,
                                "gf_ids": gf_merged,
                                "was_flipped": None,
                            }
                            consumed_micros.update(m_use)
                            continue

            slim_cands.append(
                {
                    "cid": units[t_idx]["cid"],
                    "boost_cid": units[t_idx]["boost_cid"],
                    "is_hero": units[t_idx]["is_hero"],
                    "verts": scv,
                    "normals": scn,
                    "orig_faces": scf_orig,
                    "faces": scf_ori,
                    "was_flipped": sflp,
                    "seams": sseams,
                    "allow_split": False,
                }
            )
            slim_meta.append((t_idx, m_use, gf_merged))

        if slim_cands:
            unwrapped_m = _run_blender_slim_batch(
                slim_cands,
                work_dir / f"micro_merge_{pass_idx}",
                blender_bin,
                allow_mvc_fallback=True,
                check_zero_area=True,
            )
            unwrapped_m = _apply_safe_mvc_fallback(unwrapped_m, allow_seamed=True)
            for (t_idx, m_use, gf_merged), ch_m in zip(slim_meta, unwrapped_m):
                if ch_m["ov512"] == 0 and ch_m["neg"] == 0 and not ch_m.get("zero_area", False):
                    f_out = ch_m["faces"].copy()
                    w_flp = ch_m.get("was_flipped")
                    if w_flp is not None and len(w_flp) == len(f_out) and bool(np.any(w_flp)):
                        f_out[w_flp] = f_out[w_flp][:, [0, 2, 1]]
                    uv_sub = _split_uv_connected_islands(
                        ch_m["verts"], ch_m["normals"], ch_m["uvs"], f_out, ch_m["cid"], 1.0
                    )
                    if len(uv_sub) <= 1:
                        p0, p1, p2 = ch_m["verts"][f_out[:, 0]], ch_m["verts"][f_out[:, 1]], ch_m["verts"][f_out[:, 2]]
                        a3 = 0.5 * float(np.sum(np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)))
                        accepted_direct[t_idx] = {
                            "cid": ch_m["cid"],
                            "boost_cid": ch_m["boost_cid"],
                            "is_hero": ch_m["is_hero"],
                            "verts": ch_m["verts"],
                            "normals": ch_m["normals"],
                            "faces": f_out,
                            "uvs": ch_m["uvs"],
                            "ov512": ch_m["ov512"],
                            "neg": ch_m["neg"],
                            "zero_area": False,
                            "area_3d": a3,
                            "gf_ids": gf_merged,
                            "was_flipped": None,
                        }
                        consumed_micros.update(m_use)

        if not consumed_micros:
            break

        next_units: list[dict[str, Any]] = []
        for idx, u in enumerate(units):
            if idx in consumed_micros:
                continue
            if idx in accepted_direct:
                next_units.append(accepted_direct[idx])
            else:
                next_units.append(u)
        units = next_units

    return units


def _assign_uv_island_ids(uvs: Any, faces: Any) -> Any:
    """Assign 1-based UV-connected island IDs per face."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    nv = len(uvs)
    nf = len(faces)
    if nv == 0 or nf == 0:
        return np.zeros(nf, dtype=np.int32)
    e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    rows = np.concatenate([e0, e1])
    cols = np.concatenate([e1, e0])
    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(nv, nv))
    _, v_comp = connected_components(adj, directed=False)
    f_comp = v_comp[faces[:, 0]]
    _, inv = np.unique(f_comp, return_inverse=True)
    return (inv + 1).astype(np.int32)


def _compute_min_true_island_gap_px_4k(
    uvs: Any,
    faces: Any,
    face_island_ids: Any | None = None,
    *,
    res: int = 2048,
) -> float:
    """Rasterize islands at `res` (2048 default) and measure exact min Euclidean gap in 4K pixels via EDT."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    r = max(64, int(res))
    uv = np.asarray(uvs, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(uv) == 0 or len(f) == 0:
        return 4096.0
    if face_island_ids is None:
        iids = _assign_uv_island_ids(uv, f)
    else:
        iids = np.asarray(face_island_ids, dtype=np.int32)
    unique_iids = np.unique(iids[iids > 0])
    if len(unique_iids) < 2:
        return 4096.0

    p = uv[f] * float(r)
    x0, y0 = p[:, 0, 0].copy(), p[:, 0, 1].copy()
    x1, y1 = p[:, 1, 0].copy(), p[:, 1, 1].copy()
    x2, y2 = p[:, 2, 0].copy(), p[:, 2, 1].copy()
    area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    flip = area2 < 0.0
    if np.any(flip):
        x1[flip], x2[flip] = x2[flip].copy(), x1[flip].copy()
        y1[flip], y2[flip] = y2[flip].copy(), y1[flip].copy()
        area2[flip] = -area2[flip]

    valid = np.isfinite(area2) & (area2 > 1e-9)
    p_min_x = np.maximum(0, np.floor(np.minimum(np.minimum(x0, x1), x2) - 0.5).astype(np.int32))
    p_max_x = np.minimum(r - 1, np.ceil(np.maximum(np.maximum(x0, x1), x2) - 0.5).astype(np.int32))
    p_min_y = np.maximum(0, np.floor(np.minimum(np.minimum(y0, y1), y2) - 0.5).astype(np.int32))
    p_max_y = np.minimum(r - 1, np.ceil(np.maximum(np.maximum(y0, y1), y2) - 0.5).astype(np.int32))

    island_buf = np.zeros((r, r), dtype=np.int32)

    def _is_top_left(ax: float, ay: float, bx: float, by: float) -> bool:
        return (ay < by) or (ay == by and ax > bx)

    hit_iids: set[int] = set()
    for i in np.flatnonzero(valid):
        iid = int(iids[i])
        if iid <= 0:
            continue
        ix0, ix1 = int(p_min_x[i]), int(p_max_x[i])
        iy0, iy1 = int(p_min_y[i]), int(p_max_y[i])
        if ix1 < ix0 or iy1 < iy0:
            continue
        ax, ay = float(x0[i]), float(y0[i])
        bx, by = float(x1[i]), float(y1[i])
        cx, cy = float(x2[i]), float(y2[i])
        xs = np.arange(ix0, ix1 + 1, dtype=np.float64) + 0.5
        ys = np.arange(iy0, iy1 + 1, dtype=np.float64) + 0.5
        e0 = (bx - ax) * (ys[:, None] - ay) - (by - ay) * (xs[None, :] - ax)
        e1 = (cx - bx) * (ys[:, None] - by) - (cy - by) * (xs[None, :] - bx)
        e2 = (ax - cx) * (ys[:, None] - cy) - (ay - cy) * (xs[None, :] - cx)
        m0 = (e0 >= 0.0) if _is_top_left(ax, ay, bx, by) else (e0 > 0.0)
        m1 = (e1 >= 0.0) if _is_top_left(bx, by, cx, cy) else (e1 > 0.0)
        m2 = (e2 >= 0.0) if _is_top_left(cx, cy, ax, ay) else (e2 > 0.0)
        inside = m0 & m1 & m2
        if np.any(inside):
            sub = island_buf[iy0 : iy1 + 1, ix0 : ix1 + 1]
            if np.any(inside & (sub > 0) & (sub != iid)):
                return 0.0
            sub[inside] = iid
            hit_iids.add(iid)

    # Ensure sub-pixel islands that missed pixel centers still register their centroid pixel
    if len(hit_iids) < len(unique_iids):
        for iid in unique_iids:
            ii = int(iid)
            if ii in hit_iids:
                continue
            mask_i = iids == ii
            c_uv = p[mask_i].mean(axis=(0, 1))
            cx_i = int(np.clip(round(float(c_uv[0]) - 0.5), 0, r - 1))
            cy_i = int(np.clip(round(float(c_uv[1]) - 0.5), 0, r - 1))
            if island_buf[cy_i, cx_i] > 0 and island_buf[cy_i, cx_i] != ii:
                return 0.0
            island_buf[cy_i, cx_i] = ii
            hit_iids.add(ii)

    if len(hit_iids) < 2:
        return 4096.0

    _, indices = distance_transform_edt(island_buf == 0, return_distances=True, return_indices=True)
    owner = island_buf[indices[0], indices[1]]
    sy, sx = indices[0], indices[1]
    min_d2 = float("inf")
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0a, y1a = max(0, -dy), r - max(0, dy)
        x0a, x1a = max(0, -dx), r - max(0, dx)
        y0b, y1b = max(0, dy), r - max(0, -dy)
        x0b, x1b = max(0, dx), r - max(0, -dx)
        oa = owner[y0a:y1a, x0a:x1a]
        ob = owner[y0b:y1b, x0b:x1b]
        diff = (oa != ob) & (oa > 0) & (ob > 0)
        if np.any(diff):
            dy_s = sy[y0a:y1a, x0a:x1a][diff].astype(np.float64) - sy[y0b:y1b, x0b:x1b][diff].astype(np.float64)
            dx_s = sx[y0a:y1a, x0a:x1a][diff].astype(np.float64) - sx[y0b:y1b, x0b:x1b][diff].astype(np.float64)
            cur_min = float(np.min(dy_s * dy_s + dx_s * dx_s))
            if cur_min < min_d2:
                min_d2 = cur_min

    if not math.isfinite(min_d2):
        return 4096.0
    return max(0.0, math.sqrt(min_d2) - 1.0) * (4096.0 / float(r))


def _repair_float32_loop_uvs(loop_uvs: Any, *, min_area2: float = 1e-10) -> Any:
    """Ensure every triangle has strictly positive float32 UV area (> 0) via minimal per-loop corner nudge."""
    import numpy as np

    out = np.ascontiguousarray(loop_uvs, dtype=np.float32).copy()
    if len(out) == 0:
        return out
    for _pass in range(3):
        p = out.astype(np.float64)
        a2 = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (
            p[:, 1, 1] - p[:, 0, 1]
        ) * (p[:, 2, 0] - p[:, 0, 0])
        bad = np.flatnonzero(~np.isfinite(a2) | (a2 <= min_area2))
        if len(bad) == 0:
            break
        step = 3.0e-5 * (2.0**_pass)
        for fi in bad:
            tri = p[fi]
            if not np.all(np.isfinite(tri)):
                tri = np.array([[0.5, 0.5], [0.5 + step, 0.5], [0.5, 0.5 + step]], dtype=np.float64)
                out[fi] = tri.astype(np.float32)
                continue
            edges = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
            best_a, best_b, best_c, best_len = 0, 1, 2, -1.0
            for a, b, c in edges:
                elen = float(np.linalg.norm(tri[b] - tri[a]))
                if elen > best_len:
                    best_a, best_b, best_c, best_len = a, b, c, elen
            if best_len <= 1e-6:
                base = np.clip(tri[0], 1e-4, 1.0 - 1e-4 - step)
                out[fi, 0] = base.astype(np.float32)
                out[fi, 1] = (base + np.array([step, 0.0])).astype(np.float32)
                out[fi, 2] = (base + np.array([0.0, step])).astype(np.float32)
            else:
                e = tri[best_b] - tri[best_a]
                n_perp = np.array([-e[1], e[0]], dtype=np.float64) / best_len
                delta = max(step, abs(float(a2[fi])) / best_len + step)
                new_c = np.clip(0.5 * (tri[best_a] + tri[best_b]) + delta * n_perp, 1e-5, 1.0 - 1e-5)
                out[fi, best_c] = new_c.astype(np.float32)
    return out


def _measure_glb_uv_triangle_issues(glb_path: Path) -> tuple[int, int, list[str], Any, Any]:
    """Read back an exported GLB and count negative-area & zero-area UV triangles plus material slots."""
    import numpy as np
    import trimesh

    scene_or_mesh = trimesh.load(str(glb_path), process=False)
    uv_list: list[Any] = []
    f_list: list[Any] = []
    mat_names: list[str] = []
    v_off = 0
    if isinstance(scene_or_mesh, trimesh.Scene):
        for node_name in scene_or_mesh.graph.nodes_geometry:
            _, geom_name = scene_or_mesh.graph[node_name]
            g = scene_or_mesh.geometry[geom_name]
            mname = getattr(getattr(g.visual, "material", None), "name", None)
            if mname:
                mat_names.append(str(mname))
            uv_g = np.asarray(g.visual.uv, dtype=np.float64)
            f_g = np.asarray(g.faces, dtype=np.int64)
            uv_list.append(uv_g)
            f_list.append(f_g + v_off)
            v_off += len(g.vertices)
    else:
        mname = getattr(getattr(scene_or_mesh.visual, "material", None), "name", None)
        if mname:
            mat_names.append(str(mname))
        uv_list.append(np.asarray(scene_or_mesh.visual.uv, dtype=np.float64))
        f_list.append(np.asarray(scene_or_mesh.faces, dtype=np.int64))

    uv_all = np.vstack(uv_list)
    f_all = np.vstack(f_list)
    p = uv_all[f_all]
    a2 = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (
        p[:, 1, 1] - p[:, 0, 1]
    ) * (p[:, 2, 0] - p[:, 0, 0])
    flipped = int(np.sum(a2 < -1e-12))
    zero_area = int(np.sum(np.abs(a2) <= 1e-12))
    return flipped, zero_area, mat_names, uv_all, f_all


_BLENDER_CONCAVE_PACK_SOURCE = r"""
import sys
import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1:]
in_npz = args[0]
out_npz = args[1]
margin = float(args[2])
shape_method = args[3] if len(args) > 3 else "CONCAVE"

d = np.load(in_npz, allow_pickle=False)
isl_v = d["isl_v"]
isl_f = d["isl_f"]
isl_uv = d["isl_uv"]

bpy.ops.wm.read_factory_settings(use_empty=True)
me = bpy.data.meshes.new("PackMesh")
me.from_pydata(isl_v.tolist(), [], isl_f.tolist())
me.update()
uvl = me.uv_layers.new(name="UVMap")
uvl.data.foreach_set("uv", isl_uv[isl_f.reshape(-1)].astype(np.float32).reshape(-1))

obj = bpy.data.objects.new("PackObj", me)
bpy.context.scene.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.select_all(action="SELECT")
bpy.ops.uv.pack_islands(
    shape_method=shape_method,
    rotate=True,
    scale=True,
    margin_method="FRACTION",
    margin=margin,
)
bpy.ops.object.mode_set(mode="OBJECT")

packed_loop_uv = np.empty(len(me.uv_layers.active.data) * 2, dtype=np.float64)
me.uv_layers.active.data.foreach_get("uv", packed_loop_uv)
np.savez(out_npz, packed_loop_uv=packed_loop_uv.reshape(-1, 2))
"""


_BLENDER_EXPORT_7SLOT_GLB_SOURCE = r"""
import json
import sys
import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1:]
in_npz = args[0]
out_glb = args[1]
mat_names = json.loads(args[2])

d = np.load(in_npz, allow_pickle=False)
weld_v = d["weld_v"]
weld_f = d["weld_f"]
weld_fl = d["weld_fl"]
loop_uv = d["loop_uv"]

bpy.ops.wm.read_factory_settings(use_empty=True)
me_out = bpy.data.meshes.new("PartAwareUVMesh")
me_out.from_pydata(weld_v.tolist(), [], weld_f.tolist())
me_out.update()

for mname in mat_names:
    mat = bpy.data.materials.new(name=mname)
    mat.use_nodes = True
    me_out.materials.append(mat)

mat_idx = np.clip(weld_fl - 1, 0, len(mat_names) - 1).astype(np.int32)
me_out.polygons.foreach_set("material_index", mat_idx)
me_out.polygons.foreach_set("use_smooth", np.ones(len(weld_f), dtype=bool))

uvl_out = me_out.uv_layers.new(name="UVMap")
uvl_out.data.foreach_set("uv", np.ascontiguousarray(loop_uv, dtype=np.float32).reshape(-1))

obj_out = bpy.data.objects.new("PartAwareUVMesh", me_out)
bpy.context.scene.collection.objects.link(obj_out)
bpy.context.view_layer.objects.active = obj_out
obj_out.select_set(True)

bpy.ops.export_scene.gltf(
    filepath=out_glb,
    export_format="GLB",
    use_selection=True,
    export_yup=False,
    export_normals=True,
    export_texcoords=True,
    export_materials="EXPORT",
)
"""


def _run_blender_concave_pack_loop(
    all_islands: list[dict[str, Any]],
    work_dir: Path,
    blender_bin: Path,
    *,
    tex_size: int,
    padding_px: int,
    min_true_gap_px: float,
) -> tuple[Any, float, float, float]:
    """Run Blender 5.3 CONCAVE UV pack with up to 2 margin-escalation retries to satisfy min_true_gap_px."""
    import numpy as np

    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = work_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    verts_l, faces_l, uvs_l, fid_l, off = [], [], [], [], 0
    raw_uv_area = 0.0
    for ii, isl in enumerate(all_islands):
        verts_l.append(isl["verts"])
        faces_l.append(isl["faces"] + off)
        uv_i = np.asarray(isl["uvs"], dtype=np.float64)
        pu = uv_i[isl["faces"]]
        a2 = (pu[:, 1, 0] - pu[:, 0, 0]) * (pu[:, 2, 1] - pu[:, 0, 1]) - (
            pu[:, 1, 1] - pu[:, 0, 1]
        ) * (pu[:, 2, 0] - pu[:, 0, 0])
        raw_uv_area += 0.5 * float(np.sum(np.abs(a2)))
        uvs_l.append(uv_i + np.array([5.0 * (ii % 24), 5.0 * (ii // 24)], dtype=np.float64))
        fid_l.append(np.full(len(isl["faces"]), ii + 1, dtype=np.int32))
        off += len(isl["verts"])

    isl_v = np.ascontiguousarray(np.vstack(verts_l), dtype=np.float64)
    isl_f = np.ascontiguousarray(np.vstack(faces_l), dtype=np.int64)
    isl_uv = np.ascontiguousarray(np.vstack(uvs_l), dtype=np.float64)
    isl_face_ids = np.concatenate(fid_l)

    pack_in = work_dir / "pack_in.npz"
    pack_out = work_dir / "pack_out.npz"
    pack_py = work_dir / "pack_concave.py"
    pack_py.write_text(_BLENDER_CONCAVE_PACK_SOURCE, encoding="utf-8")
    np.savez(pack_in, isl_v=isl_v, isl_f=isl_f, isl_uv=isl_uv)

    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)
    env["TMP"] = str(tmp_dir)

    gap_res = 2048 if len(isl_f) >= 400 else 512
    margin = max(0.0055, float(int(padding_px) + 8) / float(max(16, int(tex_size))))
    best_loop_uv: Any = None
    best_gap = -1.0
    best_margin = margin

    attempts = [
        ("CONCAVE", margin),
        ("CONCAVE", min(0.04, max(margin * 1.45, margin + 0.0025))),
        ("CONCAVE", min(0.04, max(margin * 2.10, margin + 0.0055))),
        ("CONVEX", min(0.04, max(margin * 2.10, 0.012))),
    ]
    for idx_att, (shape_m, cur_m) in enumerate(attempts):
        if idx_att == 3 and best_gap >= float(min_true_gap_px):
            break
        cmd = [
            str(blender_bin),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(pack_py),
            "--",
            str(pack_in),
            str(pack_out),
            f"{cur_m:.8f}",
            shape_m,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        if proc.returncode != 0 or not pack_out.exists():
            raise RuntimeError(f"Blender {shape_m} pack failed (rc={proc.returncode}):\n{proc.stderr[-1500:]}")
        packed_loop_uv = np.load(pack_out, allow_pickle=False)["packed_loop_uv"]
        loop_f = np.arange(len(isl_f) * 3, dtype=np.int64).reshape(-1, 3)
        gap_4k = _compute_min_true_island_gap_px_4k(packed_loop_uv, loop_f, isl_face_ids, res=gap_res)
        if gap_4k > best_gap:
            best_gap = gap_4k
            best_loop_uv = packed_loop_uv
            best_margin = cur_m
        if gap_4k >= float(min_true_gap_px) or len(all_islands) <= 1:
            break

    assert best_loop_uv is not None
    loop_f = np.arange(len(isl_f) * 3, dtype=np.int64).reshape(-1, 3)
    pu_p = best_loop_uv[loop_f]
    a2_p = (pu_p[:, 1, 0] - pu_p[:, 0, 0]) * (pu_p[:, 2, 1] - pu_p[:, 0, 1]) - (
        pu_p[:, 1, 1] - pu_p[:, 0, 1]
    ) * (pu_p[:, 2, 0] - pu_p[:, 0, 0])
    packed_uv_area = 0.5 * float(np.sum(np.abs(a2_p)))
    pack_scale = math.sqrt(max(packed_uv_area, 1e-12) / max(raw_uv_area, 1e-12))
    return best_loop_uv, float(best_margin), float(best_gap), float(pack_scale)


def _run_blender_export_material_slots_glb(
    weld_v: Any,
    weld_f: Any,
    weld_fl: Any,
    loop_uv_f32: Any,
    out_glb: Path,
    work_dir: Path,
    blender_bin: Path,
    *,
    material_names: Sequence[str],
) -> None:
    """Export a single continuous welded mesh with semantic material slots via Blender glTF exporter."""
    import numpy as np

    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = work_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    in_npz = work_dir / "export_7slot_in.npz"
    exp_py = work_dir / "export_7slot.py"
    exp_py.write_text(_BLENDER_EXPORT_7SLOT_GLB_SOURCE, encoding="utf-8")
    np.savez(
        in_npz,
        weld_v=np.ascontiguousarray(weld_v, dtype=np.float64),
        weld_f=np.ascontiguousarray(weld_f, dtype=np.int64),
        weld_fl=np.ascontiguousarray(weld_fl, dtype=np.int32),
        loop_uv=np.ascontiguousarray(loop_uv_f32, dtype=np.float32),
    )

    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)
    env["TMP"] = str(tmp_dir)
    cmd = [
        str(blender_bin),
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(exp_py),
        "--",
        str(in_npz),
        str(out_glb),
        json.dumps(list(material_names)),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    if proc.returncode != 0 or not out_glb.exists():
        raise RuntimeError(f"Blender 7-slot GLB export failed (rc={proc.returncode}):\n{proc.stderr[-1500:]}")


def _unwrap_via_blender_slim(
    verts: Any,
    normals_orig: Any,
    faces: Any,
    face_labels: Any,
    *,
    face_scale_boost: float,
    is_prop: bool,
    blender_bin: Path,
    scratch_parent: Path,
    max_charts: int,
    zero_overlap_refine: bool = False,
    tex_size: int = 4096,
    min_island_px: int = 16,
    crease_angle_deg: float = 48.0,
) -> list[dict[str, Any]]:
    """Run semantic/crease seam-guided Blender SLIM+ABF low-island UV pipeline."""
    import numpy as np

    raw_charts: list[dict[str, Any]] = []
    for cid in sorted(int(c) for c in np.unique(face_labels)):
        sf_g = faces[face_labels == cid]
        if len(sf_g) == 0:
            continue
        u = np.unique(sf_g)
        rm = np.full(len(verts), -1, dtype=np.int64)
        rm[u] = np.arange(len(u), dtype=np.int64)
        raw_charts.extend(
            _prepare_semantic_part_charts(
                verts[u],
                normals_orig[u],
                rm[sf_g],
                cid,
                is_prop=is_prop,
                crease_angle_deg=float(crease_angle_deg),
            )
        )

    work_dir = scratch_parent / f"._blender_uv_tmp_{os.getpid()}"
    try:
        unwrapped = _run_blender_slim_batch(
            raw_charts,
            work_dir,
            blender_bin,
            allow_mvc_fallback=False,
            check_zero_area=zero_overlap_refine,
        )
        if zero_overlap_refine:
            unwrapped = _refine_overlapping_charts_zero_overlap(
                unwrapped,
                work_dir,
                blender_bin,
                max_rounds=7,
                max_total_charts=max(int(max_charts), 260),
                min_chart_faces=12,
            )
            unwrapped = _merge_micro_islands_with_validation(
                unwrapped,
                verts,
                normals_orig,
                faces,
                face_scale_boost=float(face_scale_boost),
                is_prop=is_prop,
                tex_size=int(tex_size),
                min_island_px=int(min_island_px),
                work_dir=work_dir,
                blender_bin=blender_bin,
                min_faces_thr=12,
                max_passes=6,
            )
        else:
            unwrapped = _refine_overlapping_charts(
                unwrapped,
                work_dir,
                blender_bin,
                max_rounds=3,
                max_total_charts=max_charts,
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    all_islands: list[dict[str, Any]] = []
    for ch in unwrapped:
        v_p, n_p, f_p, uv_p = ch["verts"], ch["normals"], ch["faces"], ch["uvs"]
        if len(f_p) == 0 or len(v_p) == 0:
            continue
        p3_0, p3_1, p3_2 = v_p[f_p[:, 0]], v_p[f_p[:, 1]], v_p[f_p[:, 2]]
        area_3d = 0.5 * float(np.sum(np.linalg.norm(np.cross(p3_1 - p3_0, p3_2 - p3_0), axis=1)))
        pu = uv_p[f_p]
        area2_uv = (pu[:, 1, 0] - pu[:, 0, 0]) * (pu[:, 2, 1] - pu[:, 0, 1]) - (
            pu[:, 1, 1] - pu[:, 0, 1]
        ) * (pu[:, 2, 0] - pu[:, 0, 0])
        area_uv = 0.5 * float(np.sum(np.abs(area2_uv)))
        d_scale = math.sqrt(max(area_3d, 1e-12) / max(area_uv, 1e-12))
        part_boost = float(face_scale_boost) if (not is_prop and ch["boost_cid"] == 1) else 1.0

        # Restore original halfedge winding on output faces if sheet was re-oriented purely for UV unwrap.
        f_out = f_p.copy()
        w_flp = ch.get("was_flipped")
        if w_flp is not None and len(w_flp) == len(f_out) and bool(np.any(w_flp)):
            f_out[w_flp] = f_out[w_flp][:, [0, 2, 1]]

        part_islands = _split_uv_connected_islands(v_p, n_p, uv_p, f_out, ch["cid"], d_scale * part_boost)
        is_hero_flag = bool(ch.get("boost_cid", 0) == 1)
        for isl in part_islands:
            if zero_overlap_refine:
                ipu = isl["uvs"][isl["faces"]]
                ia2 = (ipu[:, 1, 0] - ipu[:, 0, 0]) * (ipu[:, 2, 1] - ipu[:, 0, 1]) - (
                    ipu[:, 1, 1] - ipu[:, 0, 1]
                ) * (ipu[:, 2, 0] - ipu[:, 0, 0])
                if float(np.sum(ia2)) < 0.0:
                    uv_m = isl["uvs"].copy()
                    uv_m[:, 0] = -uv_m[:, 0]
                    uv_m -= uv_m.min(axis=0)
                    isl["uvs"] = uv_m
            isl["is_hero_front"] = is_hero_flag
        all_islands.extend(part_islands)
    return all_islands


def unwrap_part_aware(
    mesh_path: str | Path,
    out_mesh_path: str | Path,
    *,
    labels_npz_path: str | Path | None = None,
    labels_source_mesh_path: str | Path | None = None,
    tex_size: int = 4096,
    padding_px: int = 16,
    face_scale_boost: float = 2.25,
    raster_verify_size: int = 1024,
    uv_preview_png: str | Path | None = None,
    unwrap_json_path: str | Path | None = None,
    uv_method: str = "auto",
    asset_type: str = "auto",
    blender_bin: str | Path = DEFAULT_BLENDER_BIN,
    max_charts: int | None = None,
    zero_overlap_refine: bool = True,
    pack_method: str = "blender_concave",
    assign_material_slots: bool = True,
    min_island_px: int = 16,
    min_true_gap_px: float = 6.0,
    out_labels_npz: str | Path | None = None,
    crease_angle_deg: float = 48.0,
) -> PartAwareUnwrapResult:
    """Part-aware low-island UV unwrap preserving head/face texel priority and >= padding_px gaps.

    By default (`uv_method='auto'`), meshes with `>= 400` faces use the S2.5 semantic-seam
    Blender 5.3 SLIM+ABF pipeline (`<= 80-120` islands, upright single-island front face,
    back scalp un-boosted) when Blender is available; small synthetic meshes or `uv_method='xatlas'`
    fall back cleanly to per-part `xatlas`.
    """
    import numpy as np
    import trimesh
    import xatlas
    from scipy.spatial import cKDTree

    from quadvox.s2_retopo.bake_xatlas import _compute_smooth_vertex_normals, compute_uv_raster_stats
    from quadvox.s2_retopo.metrics import extract_vf
    from quadvox.s2_retopo.quad_retopo import (
        get_canonical_class_names,
        load_semantic_npz,
        parse_obj_polygons,
        transfer_face_labels_kdtree,
        vertex_labels_to_face_labels,
    )

    t0 = time.perf_counter()
    method_req = str(uv_method).strip().lower()
    if method_req not in ("auto", "blender_slim", "xatlas"):
        raise ValueError(f"Unsupported uv_method {uv_method!r}; expected 'auto', 'blender_slim', or 'xatlas'.")
    atype_req = str(asset_type).strip().lower()
    if atype_req not in ("auto", "character", "prop"):
        raise ValueError(f"Unsupported asset_type {asset_type!r}; expected 'auto', 'character', or 'prop'.")
    pack_req = str(pack_method).strip().lower()
    if pack_req not in ("shelf", "blender_concave"):
        raise ValueError(f"Unsupported pack_method {pack_method!r}; expected 'shelf' or 'blender_concave'.")

    in_p = Path(mesh_path)
    out_p = Path(out_mesh_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    blender_p = Path(blender_bin)

    if in_p.suffix.lower() == ".obj":
        verts, _, faces, _ = parse_obj_polygons(in_p)
    else:
        verts, faces = extract_vf(in_p)

    if len(verts) == 0 or len(faces) == 0:
        raise ValueError(f"Cannot part-aware unwrap empty mesh: {in_p}")

    normals_orig = _compute_smooth_vertex_normals(verts, faces)
    face_centroids = (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) / 3.0
    has_labels_file = (labels_npz_path is not None) and Path(labels_npz_path).exists()
    is_prop = (atype_req == "prop") or (atype_req == "auto" and not has_labels_file)
    effective_asset_type = "prop" if is_prop else "character"
    has_valid_labels_npz = (not is_prop) and has_labels_file
    effective_assign_slots = bool(assign_material_slots) and has_valid_labels_npz

    # Resolve per-face semantic labels.
    face_labels: np.ndarray | None = None
    class_names_json_str: str | None = None
    if not is_prop and labels_npz_path is not None and Path(labels_npz_path).exists():
        sem = load_semantic_npz(labels_npz_path)
        fl = sem["face_labels"]
        vl = sem["vertex_labels"]
        with np.load(labels_npz_path, allow_pickle=False) as raw_lbl_npz:
            if "class_names_json" in raw_lbl_npz:
                class_names_json_str = str(raw_lbl_npz["class_names_json"])
        lbl_src_explicit = (
            labels_source_mesh_path is not None
            and Path(labels_source_mesh_path).exists()
            and Path(labels_source_mesh_path).resolve() != in_p.resolve()
        )
        if not lbl_src_explicit and fl is not None and len(fl) == len(faces):
            face_labels = np.asarray(fl, dtype=np.int32)
        elif not lbl_src_explicit and vl is not None and len(vl) == len(verts):
            face_labels = vertex_labels_to_face_labels(faces, vl)
        else:
            lbl_src = Path(labels_source_mesh_path) if labels_source_mesh_path is not None else in_p
            if lbl_src.exists():
                sv, sf = extract_vf(lbl_src)
                src_fl = fl if (fl is not None and len(fl) == len(sf)) else (
                    vertex_labels_to_face_labels(sf, vl) if (vl is not None and len(vl) == len(sv)) else None
                )
                if src_fl is not None:
                    sc = (sv[sf[:, 0]] + sv[sf[:, 1]] + sv[sf[:, 2]]) / 3.0
                    face_labels = transfer_face_labels_kdtree(sc, src_fl, face_centroids)

    if face_labels is None:
        if is_prop:
            face_labels = np.zeros(len(faces), dtype=np.int32)
        else:
            # Geometric head-zone fallback (y_norm > 0.72 and |x| <= 0.14) when no semantic NPZ is supplied.
            y_lo = float(np.min(verts[:, 1]))
            y_hi = float(np.max(verts[:, 1]))
            h = max(y_hi - y_lo, 1e-8)
            y_norm = (face_centroids[:, 1] - y_lo) / h
            is_head = (y_norm > 0.72) & (np.abs(face_centroids[:, 0]) <= 0.14)
            face_labels = np.where(is_head, 1, 4).astype(np.int32)

    if len(faces) >= 1000 and not is_prop:
        clean_min = 30 if zero_overlap_refine else 24
        face_labels = _clean_face_labels_iterative(faces, face_labels, min_faces=clean_min)

    if method_req == "auto":
        effective_uv_method = "blender_slim" if (len(faces) >= 400 and blender_p.is_file()) else "xatlas"
    else:
        effective_uv_method = method_req
    if (effective_uv_method == "blender_slim" or pack_req == "blender_concave" or effective_assign_slots) and not blender_p.is_file():
        raise FileNotFoundError(f"Blender binary required for requested UV/pack/export options not found: {blender_p}")

    present_classes = sorted(int(c) for c in np.unique(face_labels))
    resolved_max_charts = (
        int(max_charts) if max_charts is not None else (90 if is_prop else (180 if zero_overlap_refine else 112))
    )

    if effective_uv_method == "blender_slim":
        all_islands = _unwrap_via_blender_slim(
            verts,
            normals_orig,
            faces,
            face_labels,
            face_scale_boost=float(face_scale_boost),
            is_prop=is_prop,
            blender_bin=blender_p,
            scratch_parent=out_p.parent,
            max_charts=resolved_max_charts,
            zero_overlap_refine=bool(zero_overlap_refine),
            tex_size=int(tex_size),
            min_island_px=int(min_island_px),
            crease_angle_deg=float(crease_angle_deg),
        )
    else:
        all_islands = []
        for cid in present_classes:
            f_mask = face_labels == cid
            sub_f_global = faces[f_mask]
            if len(sub_f_global) == 0:
                continue
            used_v = np.unique(sub_f_global)
            remap = np.full(len(verts), -1, dtype=np.int64)
            remap[used_v] = np.arange(len(used_v), dtype=np.int64)
            sub_v = np.ascontiguousarray(verts[used_v], dtype=np.float64)
            sub_n = np.ascontiguousarray(normals_orig[used_v], dtype=np.float64)
            sub_f = np.ascontiguousarray(remap[sub_f_global], dtype=np.int64)

            atlas = xatlas.Atlas()
            atlas.add_mesh(
                np.ascontiguousarray(sub_v, dtype=np.float32),
                np.ascontiguousarray(sub_f, dtype=np.uint32),
                np.ascontiguousarray(sub_n, dtype=np.float32),
            )
            co = xatlas.ChartOptions()
            po = xatlas.PackOptions()
            po.resolution = 1024
            po.padding = 2
            po.bilinear = True
            atlas.generate(chart_options=co, pack_options=po)

            vmapping, indices, uvs_raw = atlas.get_mesh(0)
            vm = np.asarray(vmapping, dtype=np.int64)
            idx_f = np.ascontiguousarray(indices, dtype=np.int64)
            uv_p = np.ascontiguousarray(uvs_raw, dtype=np.float64)
            v_p = sub_v[vm]
            n_p = sub_n[vm]

            p3_0, p3_1, p3_2 = v_p[idx_f[:, 0]], v_p[idx_f[:, 1]], v_p[idx_f[:, 2]]
            area_3d_part = 0.5 * float(np.sum(np.linalg.norm(np.cross(p3_1 - p3_0, p3_2 - p3_0), axis=1)))
            pu = uv_p[idx_f]
            area2_uv = (pu[:, 1, 0] - pu[:, 0, 0]) * (pu[:, 2, 1] - pu[:, 0, 1]) - (
                pu[:, 1, 1] - pu[:, 0, 1]
            ) * (pu[:, 2, 0] - pu[:, 0, 0])
            area_uv_part = 0.5 * float(np.sum(np.abs(area2_uv)))

            density_norm_scale = math.sqrt(max(area_3d_part, 1e-12) / max(area_uv_part, 1e-12))
            part_boost = float(face_scale_boost) if (not is_prop and cid == 1) else 1.0
            part_scale = density_norm_scale * part_boost

            part_islands = _split_uv_connected_islands(v_p, n_p, uv_p, idx_f, cid, part_scale)
            for isl in part_islands:
                isl["is_hero_front"] = bool(not is_prop and cid == 1)
            all_islands.extend(part_islands)

    if zero_overlap_refine or pack_req == "blender_concave" or effective_assign_slots:
        for isl in all_islands:
            ipu = isl["uvs"][isl["faces"]]
            ia2 = (ipu[:, 1, 0] - ipu[:, 0, 0]) * (ipu[:, 2, 1] - ipu[:, 0, 1]) - (
                ipu[:, 1, 1] - ipu[:, 0, 1]
            ) * (ipu[:, 2, 0] - ipu[:, 0, 0])
            if float(np.sum(ia2)) < 0.0:
                uv_m = isl["uvs"].copy()
                uv_m[:, 0] = -uv_m[:, 0]
                uv_m -= uv_m.min(axis=0)
                isl["uvs"] = uv_m

    out_verts_list: list[np.ndarray] = []
    out_norms_list: list[np.ndarray] = []
    out_uvs_list: list[np.ndarray] = []
    out_faces_list: list[np.ndarray] = []
    out_face_cls_list: list[np.ndarray] = []
    out_face_iid_list: list[np.ndarray] = []
    v_offset = 0

    placed_boxes: list[tuple[float, float, float, float]] = []
    part_uv_areas: dict[int, float] = {}
    part_3d_areas: dict[int, float] = {}
    hero_front_uv_area = 0.0
    hero_front_3d_area = 0.0
    head_front_island_count = 0

    pack_work_dir = out_p.parent / f"._blender_pack_tmp_{os.getpid()}"
    try:
        if pack_req == "blender_concave":
            packed_loop_uv, _used_margin, pre_export_gap_4k, pack_scale = _run_blender_concave_pack_loop(
                all_islands,
                pack_work_dir,
                blender_p,
                tex_size=int(tex_size),
                padding_px=int(padding_px),
                min_true_gap_px=float(min_true_gap_px),
            )
            loop_cursor = 0
            for iid_idx, isl in enumerate(all_islands, start=1):
                f_local = isl["faces"]
                n_f = len(f_local)
                tri_uv = packed_loop_uv[loop_cursor : loop_cursor + n_f * 3].reshape(n_f, 3, 2)
                loop_cursor += n_f * 3

                uv_placed_v = np.zeros((len(isl["verts"]), 2), dtype=np.float64)
                uv_placed_v[f_local.reshape(-1)] = tri_uv.reshape(-1, 2)
                bmin_i = uv_placed_v.min(axis=0)
                bmax_i = uv_placed_v.max(axis=0)
                placed_boxes.append((float(bmin_i[0]), float(bmin_i[1]), float(bmax_i[0]), float(bmax_i[1])))

                area2 = (tri_uv[:, 1, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 2, 1] - tri_uv[:, 0, 1]) - (
                    tri_uv[:, 1, 1] - tri_uv[:, 0, 1]
                ) * (tri_uv[:, 2, 0] - tri_uv[:, 0, 0])
                uv_a = 0.5 * float(np.sum(np.abs(area2)))
                cid = int(isl["class_id"])
                part_uv_areas[cid] = part_uv_areas.get(cid, 0.0) + uv_a
                part_3d_areas[cid] = part_3d_areas.get(cid, 0.0) + float(isl["area_3d"])
                if isl.get("is_hero_front", False):
                    head_front_island_count += 1
                    hero_front_uv_area += uv_a
                    hero_front_3d_area += float(isl["area_3d"])

                out_verts_list.append(isl["verts"])
                out_norms_list.append(isl["normals"])
                out_uvs_list.append(uv_placed_v)
                out_faces_list.append(f_local + v_offset)
                out_face_cls_list.append(np.full(n_f, cid, dtype=np.int32))
                out_face_iid_list.append(np.full(n_f, iid_idx, dtype=np.int32))
                v_offset += len(isl["verts"])
        else:
            raw_dims = [(isl["raw_w"], isl["raw_h"]) for isl in all_islands]
            pack_scale, placements = pack_uv_islands_normalized(
                raw_dims,
                tex_size=int(tex_size),
                padding_px=int(padding_px),
            )
            for iid_idx, (isl, (x0, y0, pw, ph, rot)) in enumerate(zip(all_islands, placements), start=1):
                uv_local = isl["uvs"] * pack_scale
                if rot:
                    uv_rot = np.empty_like(uv_local)
                    uv_rot[:, 0] = uv_local[:, 1]
                    uv_rot[:, 1] = -uv_local[:, 0]
                    uv_rot -= uv_rot.min(axis=0)
                    uv_placed = uv_rot + np.array([x0, y0], dtype=np.float64)
                else:
                    uv_placed = uv_local + np.array([x0, y0], dtype=np.float64)

                placed_boxes.append((x0, y0, x0 + pw, y0 + ph))
                f_local = isl["faces"]
                pu = uv_placed[f_local]
                area2 = (pu[:, 1, 0] - pu[:, 0, 0]) * (pu[:, 2, 1] - pu[:, 0, 1]) - (
                    pu[:, 1, 1] - pu[:, 0, 1]
                ) * (pu[:, 2, 0] - pu[:, 0, 0])
                uv_a = 0.5 * float(np.sum(np.abs(area2)))
                cid = int(isl["class_id"])
                part_uv_areas[cid] = part_uv_areas.get(cid, 0.0) + uv_a
                part_3d_areas[cid] = part_3d_areas.get(cid, 0.0) + float(isl["area_3d"])
                if isl.get("is_hero_front", False):
                    head_front_island_count += 1
                    hero_front_uv_area += uv_a
                    hero_front_3d_area += float(isl["area_3d"])

                out_verts_list.append(isl["verts"])
                out_norms_list.append(isl["normals"])
                out_uvs_list.append(uv_placed)
                out_faces_list.append(f_local + v_offset)
                out_face_cls_list.append(np.full(len(f_local), cid, dtype=np.int32))
                out_face_iid_list.append(np.full(len(f_local), iid_idx, dtype=np.int32))
                v_offset += len(isl["verts"])

        v_all = np.ascontiguousarray(np.vstack(out_verts_list), dtype=np.float64)
        n_all = np.ascontiguousarray(np.vstack(out_norms_list), dtype=np.float64)
        uv_all = np.ascontiguousarray(np.vstack(out_uvs_list), dtype=np.float32)
        f_all = np.ascontiguousarray(np.vstack(out_faces_list), dtype=np.int64)
        fc_all = np.ascontiguousarray(np.concatenate(out_face_cls_list), dtype=np.int32)
        fiid_all = np.ascontiguousarray(np.concatenate(out_face_iid_list), dtype=np.int32)

        if zero_overlap_refine or pack_req == "blender_concave" or effective_assign_slots:
            isl_loop_uv = _repair_float32_loop_uvs(uv_all[f_all])
            if zero_overlap_refine or effective_assign_slots or pack_req == "blender_concave":
                # Rebuild per-loop vertex arrays when any sliver triangle corner was nudged
                orig_tri_uv = uv_all[f_all]
                diff_mask = np.any(np.abs(isl_loop_uv - orig_tri_uv) > 0.0, axis=(1, 2))
                if np.any(diff_mask) and not effective_assign_slots:
                    v_all = np.ascontiguousarray(v_all[f_all.reshape(-1)], dtype=np.float64)
                    n_all = np.ascontiguousarray(n_all[f_all.reshape(-1)], dtype=np.float64)
                    uv_all = np.ascontiguousarray(isl_loop_uv.reshape(-1, 2), dtype=np.float32)
                    f_all = np.arange(len(f_all) * 3, dtype=np.int64).reshape(-1, 3)

        if effective_assign_slots:
            isl_centroids = v_all[f_all].mean(axis=1)
            map_dists, isl_to_weld_face = cKDTree(face_centroids).query(isl_centroids, k=1)
            if float(np.max(map_dists)) > 1e-5 or len(np.unique(isl_to_weld_face)) != len(faces):
                raise RuntimeError(
                    f"Island-to-weld face mapping mismatch: max_dist={np.max(map_dists)}, "
                    f"unique={len(np.unique(isl_to_weld_face))}/{len(faces)}"
                )
            isl_loop_uv = _repair_float32_loop_uvs(uv_all[f_all])
            weld_loop_uv = np.zeros((len(faces), 3, 2), dtype=np.float32)
            for isl_fi, w_fi in enumerate(isl_to_weld_face):
                w_pos = verts[faces[w_fi]]
                i_pos = v_all[f_all[isl_fi]]
                i_uvs = isl_loop_uv[isl_fi]
                for k in range(3):
                    best_j = int(np.argmin(np.linalg.norm(i_pos - w_pos[k], axis=1)))
                    weld_loop_uv[w_fi, k] = i_uvs[best_j]
            weld_loop_uv = _repair_float32_loop_uvs(weld_loop_uv)
            mat_names = [name for _, name in CANONICAL_MATERIAL_SLOTS]
            _run_blender_export_material_slots_glb(
                verts,
                faces,
                face_labels,
                weld_loop_uv,
                out_p,
                pack_work_dir,
                blender_p,
                material_names=mat_names,
            )
        else:
            out_mesh = trimesh.Trimesh(
                vertices=v_all,
                faces=f_all,
                vertex_normals=n_all,
                process=False,
            )
            out_mesh.visual = trimesh.visual.TextureVisuals(uv=uv_all)
            out_mesh.export(str(out_p))
    finally:
        shutil.rmtree(pack_work_dir, ignore_errors=True)

    # Final verification directly on the exported GLB on disk
    if out_p.suffix.lower() in (".glb", ".gltf"):
        flipped_tris, zero_uv_tris, exported_mat_names, glb_uv_all, glb_f_all = _measure_glb_uv_triangle_issues(out_p)
    else:
        p_chk = uv_all.astype(np.float64)[f_all]
        a2_chk = (p_chk[:, 1, 0] - p_chk[:, 0, 0]) * (p_chk[:, 2, 1] - p_chk[:, 0, 1]) - (
            p_chk[:, 1, 1] - p_chk[:, 0, 1]
        ) * (p_chk[:, 2, 0] - p_chk[:, 0, 0])
        flipped_tris = int(np.sum(a2_chk < -1e-12))
        zero_uv_tris = int(np.sum(np.abs(a2_chk) <= 1e-12))
        exported_mat_names = []
        glb_uv_all, glb_f_all = uv_all.astype(np.float64), f_all

    verify_res = min(int(tex_size), max(64, int(raster_verify_size)))
    raster = compute_uv_raster_stats(glb_uv_all, glb_f_all, tex_size=verify_res)

    gap_res = 2048 if len(glb_f_all) >= 400 else 512
    if pack_req == "blender_concave" or effective_assign_slots:
        min_true_gap_4k = _compute_min_true_island_gap_px_4k(glb_uv_all, glb_f_all, res=gap_res)
    else:
        min_true_gap_4k = _compute_min_true_island_gap_px_4k(uv_all, f_all, fiid_all, res=gap_res)

    min_gap_uv = float("inf")
    if len(placed_boxes) >= 2:
        boxes = np.asarray(placed_boxes, dtype=np.float64)
        for i in range(len(boxes)):
            dx = np.maximum(0.0, np.maximum(boxes[i, 0] - boxes[i + 1 :, 2], boxes[i + 1 :, 0] - boxes[i, 2]))
            dy = np.maximum(0.0, np.maximum(boxes[i, 1] - boxes[i + 1 :, 3], boxes[i + 1 :, 1] - boxes[i, 3]))
            gap = np.maximum(dx, dy)
            if len(gap) > 0:
                min_gap_uv = min(min_gap_uv, float(np.min(gap)))
    min_island_gap_px = (
        float(min_true_gap_4k)
        if pack_req == "blender_concave"
        else (float(min_gap_uv * tex_size) if math.isfinite(min_gap_uv) else float(padding_px))
    )

    out_labels_str: str | None = None
    if out_labels_npz is not None:
        exp_v, exp_f = extract_vf(out_p)
        exp_centroids = exp_v[exp_f].mean(axis=1)
        exp_dists, exp_to_weld = cKDTree(face_centroids).query(exp_centroids, k=1)
        if float(np.max(exp_dists)) > 1e-5 or len(np.unique(exp_to_weld)) != len(faces):
            raise RuntimeError(f"Exported GLB label alignment mismatch: max_dist={np.max(exp_dists)}")
        aligned_fl = face_labels[exp_to_weld].astype(np.int32)
        out_lbl_p = Path(out_labels_npz)
        out_lbl_p.parent.mkdir(parents=True, exist_ok=True)
        if class_names_json_str is None:
            class_names_json_str = json.dumps({str(k): v for k, v in get_canonical_class_names().items()})
        np.savez(out_lbl_p, face_labels=aligned_fl, class_names_json=class_names_json_str)
        out_labels_str = str(out_lbl_p)

    total_uv_area = max(1e-12, sum(part_uv_areas.values()))
    total_3d_area = max(1e-12, sum(part_3d_areas.values()))
    head_uv_area = part_uv_areas.get(1, 0.0)
    head_3d_area = part_3d_areas.get(1, 0.0)
    body_uv_area = max(0.0, total_uv_area - head_uv_area)
    body_3d_area = max(0.0, total_3d_area - head_3d_area)

    head_uv_share = float(head_uv_area / total_uv_area)
    head_3d_share = float(head_3d_area / total_3d_area)
    if head_3d_area > 1e-12 and body_3d_area > 1e-12 and body_uv_area > 1e-12:
        head_lin_td = math.sqrt(head_uv_area / head_3d_area)
        body_lin_td = math.sqrt(body_uv_area / body_3d_area)
        head_vs_body_ratio = float(head_lin_td / max(body_lin_td, 1e-12))
    else:
        head_vs_body_ratio = 1.0

    non_hero_uv_area = max(0.0, total_uv_area - hero_front_uv_area)
    non_hero_3d_area = max(0.0, total_3d_area - hero_front_3d_area)
    if hero_front_3d_area > 1e-12 and non_hero_3d_area > 1e-12 and non_hero_uv_area > 1e-12:
        hero_lin_td = math.sqrt(hero_front_uv_area / hero_front_3d_area)
        non_hero_lin_td = math.sqrt(non_hero_uv_area / non_hero_3d_area)
        head_front_texel_ratio = float(hero_lin_td / max(non_hero_lin_td, 1e-12))
    else:
        head_front_texel_ratio = 1.0

    class_names = get_canonical_class_names()
    part_shares_named = {
        ("prop_body" if (is_prop and cid == 0) else class_names.get(cid, f"class_{cid}")): round(area / total_uv_area, 6)
        for cid, area in sorted(part_uv_areas.items())
    }

    preview_str: str | None = None
    if uv_preview_png is not None:
        if effective_assign_slots:
            exp_v_p, exp_f_p = extract_vf(out_p)
            exp_c_p = exp_v_p[exp_f_p].mean(axis=1)
            _, exp_idx_p = cKDTree(face_centroids).query(exp_c_p, k=1)
            prev_fc = face_labels[exp_idx_p]
            prev_p = _render_uv_layout_preview(glb_uv_all, glb_f_all, prev_fc, Path(uv_preview_png), preview_size=1024)
        else:
            prev_p = _render_uv_layout_preview(uv_all, f_all, fc_all, Path(uv_preview_png), preview_size=1024)
        preview_str = str(prev_p)

    result = PartAwareUnwrapResult(
        input_mesh_path=str(in_p),
        out_mesh_path=str(out_p),
        tex_size=int(tex_size),
        padding_px=int(padding_px),
        face_scale_boost=float(face_scale_boost),
        vertex_count=int(len(glb_uv_all)),
        face_count=int(len(glb_f_all)),
        part_count=int(len(present_classes)),
        island_count=int(len(all_islands)),
        uv_coverage=float(raster["uv_coverage"]),
        uv_overlap_count=int(raster["uv_overlap_count"]),
        uv_area_sum=float(raster["uv_area_sum"]),
        head_uv_area_share=head_uv_share,
        head_3d_area_share=head_3d_share,
        head_vs_body_texel_ratio=head_vs_body_ratio,
        min_island_gap_px=min_island_gap_px,
        pack_scale=float(pack_scale),
        part_uv_area_shares=part_shares_named,
        uv_preview_png=preview_str,
        elapsed_s=time.perf_counter() - t0,
        uv_method=effective_uv_method,
        asset_type=effective_asset_type,
        head_front_island_count=int(head_front_island_count),
        head_front_texel_ratio=float(head_front_texel_ratio),
        pack_method=pack_req,
        min_true_island_gap_px_4k=float(min_true_gap_4k),
        flipped_triangles=int(flipped_tris),
        zero_area_uv_triangles=int(zero_uv_tris),
        material_slots=exported_mat_names,
        out_labels_npz=out_labels_str,
    )
    if unwrap_json_path is not None:
        result.write_json(unwrap_json_path)
    return result


# ---------------------------------------------------------------------------
# Headless Blender Cycles 4K Multi-Map Baking (Normal + AO + Curvature)
# ---------------------------------------------------------------------------


def _blender_configure_cycles(scene: Any, *, use_gpu: bool, samples: int, denoise: bool = False) -> str:
    import bpy

    scene.render.engine = "CYCLES"
    scene.cycles.samples = max(1, int(samples))
    scene.cycles.use_denoising = bool(denoise)
    if denoise:
        try:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
        except Exception:
            pass

    if not use_gpu:
        scene.cycles.device = "CPU"
        return "CPU"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    cuda_devs = [d for d in prefs.devices if d.type == "CUDA"]
    if not cuda_devs:
        raise RuntimeError("CUDA GPU requested for Cycles, but no CUDA devices were detected.")
    for dev in prefs.devices:
        dev.use = dev.type == "CUDA"
    scene.cycles.device = "GPU"
    return f"CUDA:{','.join(d.name for d in cuda_devs)}"


def _blender_import_meshes(mesh_path: Path) -> list[Any]:
    import bpy

    before = set(bpy.data.objects)
    ext = mesh_path.suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(
            filepath=str(mesh_path),
            merge_vertices=False,
            import_shading="NORMALS",
        )
    elif ext == ".obj":
        bpy.ops.wm.obj_import(
            filepath=str(mesh_path),
            forward_axis="NEGATIVE_Z",
            up_axis="Y",
        )
    elif ext == ".ply":
        bpy.ops.wm.ply_import(
            filepath=str(mesh_path),
            forward_axis="NEGATIVE_Z",
            up_axis="Y",
        )
    else:
        raise ValueError(f"Unsupported mesh format for Blender import: {mesh_path}")

    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No MESH objects imported from {mesh_path}")
    return meshes


def _blender_estimate_surface_dist(
    high_meshes: Sequence[Any],
    low_obj: Any,
    *,
    max_query_points: int = 40000,
) -> tuple[float, float]:
    import bpy
    import numpy as np
    from mathutils.bvhtree import BVHTree

    depsgraph = bpy.context.evaluated_depsgraph_get()
    high_bvhs = [(hobj.matrix_world.inverted(), BVHTree.FromObject(hobj, depsgraph)) for hobj in high_meshes]
    if not high_bvhs:
        return 0.0, 0.0

    mw_low = low_obj.matrix_world
    raw_pts = [mw_low @ v.co for v in low_obj.data.vertices]
    raw_pts.extend(mw_low @ p.center for p in low_obj.data.polygons)
    if not raw_pts:
        return 0.0, 0.0

    step = max(1, len(raw_pts) // int(max_query_points))
    q_pts = raw_pts[::step][: int(max_query_points)]
    dists = np.full(len(q_pts), float("inf"), dtype=np.float64)
    for mw_inv, bvh in high_bvhs:
        for idx, pt in enumerate(q_pts):
            _, _, _, d = bvh.find_nearest(mw_inv @ pt)
            if d is not None and d < dists[idx]:
                dists[idx] = float(d)
    valid = dists[np.isfinite(dists)]
    if len(valid) == 0:
        return 0.0, 0.0
    return float(np.percentile(valid, 95.0)), float(np.percentile(valid, 99.0))


def _blender_assign_discrete_curvature_attribute(mesh_obj: Any, attr_name: str = "DiscreteCurvature") -> None:
    """Compute umbrella discrete mean curvature `(v_i - mean(nbrs)) . n_i` as a FLOAT point attribute."""
    import numpy as np

    me = mesh_obj.data
    n_v = len(me.vertices)
    if n_v == 0:
        return
    co = np.empty(n_v * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)

    vn = np.empty(n_v * 3, dtype=np.float64)
    me.vertices.foreach_get("normal", vn)
    vn = vn.reshape(-1, 3)

    n_e = len(me.edges)
    if n_e == 0:
        curv = np.zeros(n_v, dtype=np.float32)
    else:
        ev = np.empty(n_e * 2, dtype=np.int64)
        me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        nbr_sum = np.zeros((n_v, 3), dtype=np.float64)
        nbr_cnt = np.zeros(n_v, dtype=np.float64)
        np.add.at(nbr_sum, ev[:, 0], co[ev[:, 1]])
        np.add.at(nbr_sum, ev[:, 1], co[ev[:, 0]])
        np.add.at(nbr_cnt, ev[:, 0], 1.0)
        np.add.at(nbr_cnt, ev[:, 1], 1.0)
        mean_nbr = np.where(nbr_cnt[:, None] > 0.0, nbr_sum / np.maximum(nbr_cnt[:, None], 1.0), co)
        # Positive where vertex sticks outward along normal (convex), negative where concave.
        lap_vec = co - mean_nbr
        raw_curv = np.sum(lap_vec * vn, axis=1)
        scale = float(np.percentile(np.abs(raw_curv), 98.0)) if len(raw_curv) else 0.0
        if not math.isfinite(scale) or scale <= 1e-12:
            curv = np.full(n_v, 0.5, dtype=np.float32)
        else:
            # Map strictly into [0.05, 0.95] around 0.5 so Cycles Emission never clips negative concavity.
            curv = np.clip(0.5 + 0.45 * (raw_curv / scale), 0.05, 0.95).astype(np.float32)

    if attr_name in me.attributes:
        me.attributes.remove(me.attributes[attr_name])
    attr = me.attributes.new(name=attr_name, type="FLOAT", domain="POINT")
    attr.data.foreach_set("value", curv)


def _write_4k_contact_sheet_in_blender(
    *,
    normal_png: Path,
    ao_png: Path,
    curvature_png: Path,
    uv_preview_png: Path | None,
    out_sheet_png: Path,
    labels: Sequence[str],
    tile_size: int = 1024,
) -> Path:
    """Build a 2x2 labeled diagnostic contact sheet of 4K Normal, AO, Curvature, and UV Layout."""
    import numpy as np

    from quadvox.s2_retopo.render import (
        _draw_banner_label,
        _read_png_rgb_in_blender,
        _resize_rgb_bilinear,
        _write_png_rgb_in_blender,
    )

    paths = [normal_png, ao_png, curvature_png, uv_preview_png if (uv_preview_png and uv_preview_png.exists()) else normal_png]
    pad = 8
    ts = int(tile_size)
    sheet = np.full((2 * ts + pad, 2 * ts + pad, 3), (18, 20, 24), dtype=np.uint8)

    for idx, (p, lbl) in enumerate(zip(paths, labels)):
        arr = _read_png_rgb_in_blender(p)
        tile = _resize_rgb_bilinear(arr, ts, ts).copy()
        _draw_banner_label(tile, lbl, scale=3)
        r, c = divmod(idx, 2)
        y0 = r * (ts + pad)
        x0 = c * (ts + pad)
        sheet[y0 : y0 + ts, x0 : x0 + ts] = tile

    _write_png_rgb_in_blender(out_sheet_png, sheet)
    return out_sheet_png


def _erode_bool_8(mask: Any, radius: int) -> Any:
    """8-connected binary erosion treating image edges as interior."""
    import numpy as np

    cur = np.asarray(mask, dtype=bool).copy()
    steps = max(0, int(radius))
    h, w = cur.shape
    for _ in range(steps):
        padded = np.pad(cur, 1, mode="constant", constant_values=True)
        eroded = np.ones_like(cur, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                eroded &= padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        cur = eroded
    return cur


def _gather_window_stats(
    work_pad: Any,
    ref_pad: Any,
    by: Any,
    bx: Any,
    max_r: int,
    win_r: int,
    *,
    exclude_center: bool = False,
) -> tuple[Any, Any, Any, Any]:
    """Gather neighbor count, min, max, and 1/dist weighted mean over `[-win_r, win_r]`."""
    import numpy as np

    n = int(len(by))
    c = int(work_pad.shape[2])
    count = np.zeros(n, dtype=np.int32)
    wsum = np.zeros(n, dtype=np.float32)
    acc = np.zeros((n, c), dtype=np.float32)
    vmin = np.full((n, c), np.inf, dtype=np.float32)
    vmax = np.full((n, c), -np.inf, dtype=np.float32)

    py = by + max_r
    px = bx + max_r
    for dy in range(-win_r, win_r + 1):
        for dx in range(-win_r, win_r + 1):
            if exclude_center and dy == 0 and dx == 0:
                continue
            valid = ref_pad[py + dy, px + dx]
            if not np.any(valid):
                continue
            vals = work_pad[py + dy, px + dx]
            dist = float(np.hypot(dy, dx))
            w = 1.0 if dist == 0.0 else (1.0 / dist)
            count[valid] += 1
            wsum[valid] += w
            acc[valid] += w * vals[valid]
            vmin[valid] = np.minimum(vmin[valid], vals[valid])
            vmax[valid] = np.maximum(vmax[valid], vals[valid])

    has_ref = count > 0
    mean = np.zeros((n, c), dtype=np.float32)
    mean[has_ref] = acc[has_ref] / np.maximum(wsum[has_ref, None], 1e-8)
    return has_ref, vmin, vmax, mean


def smooth_clamp_uv_island_border(
    arr: Any,
    mask: Any,
    *,
    border_px: int = 2,
    smooth_alpha: float = 0.8,
    renormalize_normals: bool = False,
    min_tangent_nz: float = 0.25,
) -> Any:
    """Clamp and smooth outer `1-2 px` UV island border texels before EDT gutter dilation.

    Uses a tiered local clean reference lookup (`erode_N` interior -> `erode_1` -> non-self mask)
    so grazing rim spikes on UV island boundaries are clamped to local interior bounds and
    smoothed toward clean neighbors without altering pixels deep inside the UV island.
    """
    import numpy as np

    orig = np.asarray(arr)
    mask_b = np.asarray(mask, dtype=bool)
    out = orig.copy()
    bpx = max(0, int(border_px))
    if bpx == 0 or not np.any(mask_b):
        return out

    erode_1 = _erode_bool_8(mask_b, 1)
    interior = erode_1 if bpx == 1 else _erode_bool_8(mask_b, bpx)
    border_mask = mask_b & ~interior
    if not np.any(border_mask):
        return out

    was_2d = out.ndim == 2
    work = np.asarray(out, dtype=np.float32)
    if was_2d:
        work = work[:, :, None]

    by, bx = np.where(border_mask)
    max_r = 3
    work_pad = np.pad(work, ((max_r, max_r), (max_r, max_r), (0, 0)), mode="edge")
    int_pad = np.pad(interior, ((max_r, max_r), (max_r, max_r)), mode="constant", constant_values=False)
    e1_pad = np.pad(erode_1, ((max_r, max_r), (max_r, max_r)), mode="constant", constant_values=False)
    m_pad = np.pad(mask_b, ((max_r, max_r), (max_r, max_r)), mode="constant", constant_values=False)

    orig_vals = work[by, bx]
    c = int(work.shape[2])
    n = int(len(by))
    resolved = np.zeros(n, dtype=bool)
    ref_min = orig_vals.copy()
    ref_max = orig_vals.copy()
    ref_mean = orig_vals.copy()

    tiers = [
        (int_pad, 2, False),
        (int_pad, 3, False),
        (e1_pad, 2, False),
        (m_pad, 2, True),
    ]
    for ref_pad, win_r, excl_self in tiers:
        need = ~resolved
        if not np.any(need):
            break
        has_ref, vmin, vmax, mean = _gather_window_stats(
            work_pad, ref_pad, by[need], bx[need], max_r, win_r, exclude_center=excl_self
        )
        if np.any(has_ref):
            idx = np.flatnonzero(need)[has_ref]
            ref_min[idx] = vmin[has_ref]
            ref_max[idx] = vmax[has_ref]
            ref_mean[idx] = mean[has_ref]
            resolved[idx] = True

    clamped = orig_vals.copy()
    if np.any(resolved):
        clamped[resolved] = np.clip(orig_vals[resolved], ref_min[resolved], ref_max[resolved])

    alpha_base = float(np.clip(smooth_alpha, 0.0, 1.0))
    outer_rim = ~erode_1[by, bx]
    alpha = np.where(outer_rim, alpha_base, 0.5 * alpha_base)[:, None].astype(np.float32)
    blended = clamped.copy()
    if np.any(resolved):
        blended[resolved] = (1.0 - alpha[resolved]) * clamped[resolved] + alpha[resolved] * ref_mean[resolved]
        blended[resolved] = np.clip(blended[resolved], ref_min[resolved], ref_max[resolved])

    if renormalize_normals and c == 3:
        scale = 255.0 if np.issubdtype(orig.dtype, np.integer) else 1.0
        n_xyz = (blended / scale) * 2.0 - 1.0
        n_xyz[:, 2] = np.maximum(n_xyz[:, 2], float(min_tangent_nz))
        norms = np.linalg.norm(n_xyz, axis=1, keepdims=True)
        bad = norms[:, 0] < 1e-6
        if np.any(bad):
            n_xyz[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            norms[bad] = 1.0
        blended = np.clip(0.5 + 0.5 * (n_xyz / norms), 0.0, 1.0) * scale

    work_out = work.copy()
    work_out[by, bx] = blended
    if np.issubdtype(orig.dtype, np.integer):
        work_out = np.rint(work_out)
    res = work_out[:, :, 0] if was_2d else work_out
    return res.astype(orig.dtype, copy=False)


def _blender_dilate_2d_margin(
    arr: Any,
    mask: Any,
    padding_px: int,
    bg_value: Sequence[float],
    *,
    border_smooth_px: int = 2,
    renormalize_normals: bool = False,
) -> Any:
    """Smooth/clamp outer `1-2 px` island borders, then extend valid pixels outward by `padding_px`."""
    import numpy as np

    pad = max(0, int(padding_px))
    out = arr.copy()
    if int(border_smooth_px) > 0 and np.any(mask):
        out = smooth_clamp_uv_island_border(
            out,
            mask,
            border_px=int(border_smooth_px),
            renormalize_normals=bool(renormalize_normals),
        )
    bg = np.asarray(bg_value, dtype=out.dtype)
    out[~mask] = bg
    if pad == 0 or not np.any(mask):
        return out

    try:
        from scipy.ndimage import distance_transform_edt

        dist, indices = distance_transform_edt(~mask, return_distances=True, return_indices=True)
        fill = (~mask) & (dist <= float(pad))
        out[fill] = out[indices[0][fill], indices[1][fill]]
        return out
    except ImportError:
        pass

    filled = mask.copy()
    h, w = filled.shape
    c = out.shape[2]
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for _ in range(pad):
        frontier = ~filled
        if not np.any(frontier):
            break
        acc = np.zeros((h, w, c), dtype=np.float64)
        cnt = np.zeros((h, w), dtype=np.int32)
        for dy, dx in neighbors:
            y_src0, y_src1 = max(0, -dy), h - max(0, dy)
            x_src0, x_src1 = max(0, -dx), w - max(0, dx)
            y_dst0, y_dst1 = max(0, dy), h - max(0, -dy)
            x_dst0, x_dst1 = max(0, dx), w - max(0, -dx)
            valid_nbr = filled[y_src0:y_src1, x_src0:x_src1]
            acc[y_dst0:y_dst1, x_dst0:x_dst1] += np.where(
                valid_nbr[:, :, None],
                out[y_src0:y_src1, x_src0:x_src1],
                0.0,
            )
            cnt[y_dst0:y_dst1, x_dst0:x_dst1] += valid_nbr.astype(np.int32)
        upd = frontier & (cnt > 0)
        if not np.any(upd):
            break
        out[upd] = (acc[upd] / cnt[upd, None]).astype(out.dtype)
        filled |= upd
    return out


def _blender_inpaint_masked_2d(
    arr: Any,
    valid_mask: Any,
    target_mask: Any,
    fallback_value: Sequence[float],
    max_dist_px: float = 8.0,
) -> Any:
    """Inpaint pixels in `target_mask & (~valid_mask)` from nearest `valid_mask` pixels."""
    import numpy as np

    arr_np = np.asarray(arr)
    squeeze_2d = arr_np.ndim == 2
    out = (arr_np[:, :, None] if squeeze_2d else arr_np).copy()
    valid = np.asarray(valid_mask, dtype=bool)
    fill_mask = np.asarray(target_mask, dtype=bool) & (~valid)
    if not np.any(fill_mask):
        return out[:, :, 0] if squeeze_2d else out

    fb = np.asarray(fallback_value, dtype=out.dtype)
    if fb.ndim == 0 or len(fb) == 1:
        fb_vec = np.full(out.shape[2], float(fb.ravel()[0]), dtype=out.dtype)
    else:
        fb_vec = fb.astype(out.dtype)

    if not np.any(valid):
        out[fill_mask] = fb_vec
        return out[:, :, 0] if squeeze_2d else out

    try:
        from scipy.ndimage import distance_transform_edt

        dist, indices = distance_transform_edt(~valid, return_distances=True, return_indices=True)
        near = fill_mask & (dist <= float(max_dist_px))
        far = fill_mask & (~near)
        if np.any(near):
            out[near] = out[indices[0][near], indices[1][near]]
        if np.any(far):
            out[far] = fb_vec
        return out[:, :, 0] if squeeze_2d else out
    except ImportError:
        pass

    steps = max(1, int(np.ceil(float(max_dist_px))))
    filled = valid.copy()
    h, w, c = out.shape
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for _ in range(steps):
        frontier = fill_mask & (~filled)
        if not np.any(frontier):
            break
        acc = np.zeros((h, w, c), dtype=np.float64)
        cnt = np.zeros((h, w), dtype=np.int32)
        for dy, dx in neighbors:
            y_src0, y_src1 = max(0, -dy), h - max(0, dy)
            x_src0, x_src1 = max(0, -dx), w - max(0, dx)
            y_dst0, y_dst1 = max(0, dy), h - max(0, -dy)
            x_dst0, x_dst1 = max(0, dx), w - max(0, -dx)
            valid_nbr = filled[y_src0:y_src1, x_src0:x_src1]
            acc[y_dst0:y_dst1, x_dst0:x_dst1] += np.where(
                valid_nbr[:, :, None],
                out[y_src0:y_src1, x_src0:x_src1],
                0.0,
            )
            cnt[y_dst0:y_dst1, x_dst0:x_dst1] += valid_nbr.astype(np.int32)
        upd = frontier & (cnt > 0)
        if not np.any(upd):
            break
        out[upd] = (acc[upd] / cnt[upd, None]).astype(out.dtype)
        filled |= upd
    still_missing = fill_mask & (~filled)
    if np.any(still_missing):
        out[still_missing] = fb_vec
    return out[:, :, 0] if squeeze_2d else out


def _blender_build_mesh_obj(
    bpy: Any,
    name: str,
    verts_co: Any,
    faces: Any,
    *,
    matrix_world: Any | None = None,
    uvs_per_loop: Any | None = None,
    custom_loop_normals: Any | None = None,
    point_attr: tuple[str, Any] | None = None,
) -> Any:
    """Fast vectorized Blender mesh construction using `foreach_set`."""
    import numpy as np

    nv = len(verts_co)
    nf = len(faces)
    me = bpy.data.meshes.new(name)
    me.vertices.add(nv)
    me.vertices.foreach_set("co", np.ascontiguousarray(verts_co, dtype=np.float32).ravel())

    n_loops = nf * 3
    me.loops.add(n_loops)
    me.loops.foreach_set("vertex_index", np.ascontiguousarray(faces, dtype=np.int32).ravel())

    me.polygons.add(nf)
    me.polygons.foreach_set("loop_start", np.arange(0, n_loops, 3, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(nf, 3, dtype=np.int32))
    me.polygons.foreach_set("use_smooth", np.ones(nf, dtype=bool))
    me.update(calc_edges=True)

    if uvs_per_loop is not None:
        uv_layer = me.uv_layers.new(name="UVMap")
        uv_layer.data.foreach_set("uv", np.ascontiguousarray(uvs_per_loop, dtype=np.float32).ravel())

    if custom_loop_normals is not None and len(custom_loop_normals) == n_loops:
        me.normals_split_custom_set(np.ascontiguousarray(custom_loop_normals, dtype=np.float32))

    if point_attr is not None:
        attr_name, attr_vals = point_attr
        attr = me.attributes.new(name=attr_name, type="FLOAT", domain="POINT")
        attr.data.foreach_set("value", np.ascontiguousarray(attr_vals, dtype=np.float32).ravel())

    obj = bpy.data.objects.new(name, me)
    if matrix_world is not None:
        obj.matrix_world = matrix_world.copy()
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _blender_compute_discrete_curv_np(verts: Any, faces: Any) -> Any:
    """Vectorized umbrella discrete mean curvature mapped into [0.05, 0.95] around 0.5."""
    import numpy as np

    nv = len(verts)
    if nv == 0 or len(faces) == 0:
        return np.full(nv, 0.5, dtype=np.float32)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    vn = np.zeros((nv, 3), dtype=np.float64)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    vnorm = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = vn / np.maximum(vnorm, 1e-12)

    all_e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    u = np.minimum(all_e[:, 0], all_e[:, 1])
    v = np.maximum(all_e[:, 0], all_e[:, 1])
    edges = np.unique(np.stack([u, v], axis=1), axis=0)

    nbr_sum = np.zeros((nv, 3), dtype=np.float64)
    nbr_cnt = np.zeros(nv, dtype=np.float64)
    np.add.at(nbr_sum, edges[:, 0], verts[edges[:, 1]])
    np.add.at(nbr_sum, edges[:, 1], verts[edges[:, 0]])
    np.add.at(nbr_cnt, edges[:, 0], 1.0)
    np.add.at(nbr_cnt, edges[:, 1], 1.0)
    mean_nbr = np.where(nbr_cnt[:, None] > 0.0, nbr_sum / np.maximum(nbr_cnt[:, None], 1.0), verts)
    raw = np.sum((verts - mean_nbr) * vn, axis=1)
    scale = float(np.percentile(np.abs(raw), 98.0)) if len(raw) else 0.0
    if not math.isfinite(scale) or scale <= 1e-12:
        return np.full(nv, 0.5, dtype=np.float32)
    return np.clip(0.5 + 0.45 * (raw / scale), 0.05, 0.95).astype(np.float32)


def _blender_measure_uv_gap_px_4k(loop_uvs: Any, *, res: int = 1024, max_pad_check: int = 20) -> float:
    """Estimate minimum inter-island UV gap in 4K pixels inside Blender Python (no scipy required)."""
    import numpy as np

    nf = len(loop_uvs)
    if nf == 0:
        return 4096.0
    tri_uv = np.asarray(loop_uvs, dtype=np.float64).reshape(nf, 3, 2)

    # Pure-Python Union-Find over quantized UV vertices to group UV-connected islands
    uv_vid: dict[tuple[int, int], int] = {}
    f_uv_idx = np.empty((nf, 3), dtype=np.int64)
    for fi in range(nf):
        for k in range(3):
            key = (int(round(float(tri_uv[fi, k, 0]) * 100000.0)), int(round(float(tri_uv[fi, k, 1]) * 100000.0)))
            vid = uv_vid.get(key)
            if vid is None:
                vid = len(uv_vid)
                uv_vid[key] = vid
            f_uv_idx[fi, k] = vid

    nv = len(uv_vid)
    parent = list(range(nv))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for fi in range(nf):
        a, b, c = int(f_uv_idx[fi, 0]), int(f_uv_idx[fi, 1]), int(f_uv_idx[fi, 2])
        _union(a, b)
        _union(b, c)

    roots = [_find(int(f_uv_idx[fi, 0])) for fi in range(nf)]
    root_map: dict[int, int] = {}
    iids = np.empty(nf, dtype=np.int32)
    for fi, rt in enumerate(roots):
        iid = root_map.get(rt)
        if iid is None:
            iid = len(root_map) + 1
            root_map[rt] = iid
        iids[fi] = iid

    if len(root_map) < 2:
        return 4096.0

    try:
        flat_uv = tri_uv.reshape(-1, 2)
        flat_f = np.arange(nf * 3, dtype=np.int64).reshape(nf, 3)
        return _compute_min_true_island_gap_px_4k(flat_uv, flat_f, iids, res=2048)
    except ImportError:
        pass

    r = max(256, int(res))
    px = tri_uv * float(r)
    x0, y0 = px[:, 0, 0].copy(), px[:, 0, 1].copy()
    x1, y1 = px[:, 1, 0].copy(), px[:, 1, 1].copy()
    x2, y2 = px[:, 2, 0].copy(), px[:, 2, 1].copy()
    area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    flip = area2 < 0.0
    if np.any(flip):
        x1[flip], x2[flip] = x2[flip].copy(), x1[flip].copy()
        y1[flip], y2[flip] = y2[flip].copy(), y1[flip].copy()
        area2[flip] = -area2[flip]

    island_buf = np.zeros((r, r), dtype=np.int32)
    valid = np.isfinite(area2) & (area2 > 1e-9)
    p_min_x = np.maximum(0, np.floor(np.minimum(np.minimum(x0, x1), x2) - 0.5).astype(np.int32))
    p_max_x = np.minimum(r - 1, np.ceil(np.maximum(np.maximum(x0, x1), x2) - 0.5).astype(np.int32))
    p_min_y = np.maximum(0, np.floor(np.minimum(np.minimum(y0, y1), y2) - 0.5).astype(np.int32))
    p_max_y = np.minimum(r - 1, np.ceil(np.maximum(np.maximum(y0, y1), y2) - 0.5).astype(np.int32))

    for i in np.flatnonzero(valid):
        iid = int(iids[i])
        ix0, ix1 = int(p_min_x[i]), int(p_max_x[i])
        iy0, iy1 = int(p_min_y[i]), int(p_max_y[i])
        if ix1 < ix0 or iy1 < iy0:
            continue
        ax, ay = float(x0[i]), float(y0[i])
        bx, by = float(x1[i]), float(y1[i])
        cx, cy = float(x2[i]), float(y2[i])
        xs = np.arange(ix0, ix1 + 1, dtype=np.float64) + 0.5
        ys = np.arange(iy0, iy1 + 1, dtype=np.float64) + 0.5
        e0 = (bx - ax) * (ys[:, None] - ay) - (by - ay) * (xs[None, :] - ax)
        e1 = (cx - bx) * (ys[:, None] - by) - (cy - by) * (xs[None, :] - bx)
        e2 = (ax - cx) * (ys[:, None] - cy) - (ay - cy) * (xs[None, :] - cx)
        inside = (e0 >= 0.0) & (e1 >= 0.0) & (e2 >= 0.0)
        if np.any(inside):
            sub = island_buf[iy0 : iy1 + 1, ix0 : ix1 + 1]
            if np.any(inside & (sub > 0) & (sub != iid)):
                return 0.0
            sub[inside] = iid

    scale_4k = 4096.0 / float(r)
    cur = island_buf.copy()
    max_steps = max(2, int(math.ceil(float(max_pad_check) / scale_4k)) + 2)
    nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for step in range(1, max_steps + 1):
        nxt = cur.copy()
        for dy, dx in nbrs:
            y_s0, y_s1 = max(0, -dy), r - max(0, dy)
            x_s0, x_s1 = max(0, -dx), r - max(0, dx)
            y_d0, y_d1 = max(0, dy), r - max(0, -dy)
            x_d0, x_d1 = max(0, dx), r - max(0, -dx)
            src = cur[y_s0:y_s1, x_s0:x_s1]
            dst = nxt[y_d0:y_d1, x_d0:x_d1]
            collide = (src > 0) & (dst > 0) & (src != dst)
            if np.any(collide):
                return max(0.0, float(2 * step - 2) * scale_4k)
            fill = (dst == 0) & (src > 0)
            dst[fill] = src[fill]
        cur = nxt
    return float(2 * max_steps) * scale_4k


def _run_bake_4k_in_blender(
    *,
    high_mesh_path: Path,
    low_uv_mesh_path: Path,
    out_normal_png: Path,
    out_ao_png: Path,
    out_curvature_png: Path,
    out_baked_glb: Path,
    out_contact_sheet_png: Path | None,
    uv_preview_png: Path | None,
    tex_size: int,
    padding_px: int,
    normal_samples: int,
    ao_samples: int,
    ao_distance_rel: float,
    curvature_samples: int,
    use_gpu: bool,
    ray_distance: float | None,
    cage_extrusion: float | None,
    meta_json_path: Path,
    isolated_part_bake: bool | None = None,
    high_labels_npz: Path | None = None,
    low_labels_npz: Path | None = None,
    ring_dilation: int = 3,
    head_ring_dilation: int = 16,
    head_ao_exclude_hair: bool = True,
    preserve_material_slots: bool = True,
) -> MultiMapBakeResult:
    import bpy
    import mathutils
    import numpy as np

    resolved_isolated = (high_labels_npz is not None) if isolated_part_bake is None else bool(isolated_part_bake)
    t0 = time.perf_counter()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    device_used = _blender_configure_cycles(scene, use_gpu=use_gpu, samples=normal_samples, denoise=False)
    scene.view_settings.view_transform = "Standard"

    high_meshes = _blender_import_meshes(high_mesh_path)
    low_meshes = _blender_import_meshes(low_uv_mesh_path)

    if len(low_meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in low_meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = low_meshes[0]
        bpy.ops.object.join()
        low_obj = bpy.context.view_layer.objects.active
    else:
        low_obj = low_meshes[0]

    if not low_obj.data.uv_layers:
        raise ValueError(f"Lowpoly mesh has no UV layer for 4K baking: {low_uv_mesh_path}")

    bpy.ops.object.select_all(action="DESELECT")
    low_obj.select_set(True)
    bpy.context.view_layer.objects.active = low_obj
    if not low_obj.data.has_custom_normals:
        bpy.ops.object.shade_smooth()

    bpy.context.view_layer.update()
    mw = low_obj.matrix_world
    pts = [mw @ mathutils.Vector(c) for c in low_obj.bound_box]
    bmin = mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    bmax = mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    diag = max(float((bmax - bmin).length), 1e-6)

    d_p95, d_p99 = _blender_estimate_surface_dist(high_meshes, low_obj)
    if cage_extrusion is not None:
        resolved_cage = float(cage_extrusion)
    elif d_p99 > 0.0:
        resolved_cage = float(np.clip(2.0 * d_p99, 0.0012 * diag, 0.015 * diag))
    else:
        resolved_cage = 0.002 * diag

    if ray_distance is not None:
        resolved_ray = float(ray_distance)
    else:
        resolved_ray = float(max(2.5 * resolved_cage, 0.003 * diag))

    # Build a topology-identical welded smooth-normal cage so split UV seams & >40 deg
    # sharp hard-surface prop edges project continuous averaged rays with zero dark gaps.
    mw3 = mw.to_3x3()
    mw3_inv = mw3.inverted_safe()
    low_verts = low_obj.data.vertices
    pos_normals: dict[tuple[float, float, float], Any] = {}
    for poly in low_obj.data.polygons:
        wn = mw3 @ (poly.normal * max(float(poly.area), 1e-12))
        for vi in poly.vertices:
            wco = mw @ low_verts[vi].co
            key = (round(float(wco.x), 5), round(float(wco.y), 5), round(float(wco.z), 5))
            if key in pos_normals:
                pos_normals[key] += wn
            else:
                pos_normals[key] = wn.copy()

    res = int(tex_size)
    pad = max(0, int(padding_px))

    def _ensure_nodes(mat: Any) -> Any:
        if mat.node_tree is None:
            mat.use_nodes = True
        return mat.node_tree

    # ------------------------------------------------------------------
    # Part-Isolated 4K Baking Path (auto-enabled when high_labels_npz provided unless opted out)
    # ------------------------------------------------------------------
    if resolved_isolated:
        from quadvox.s2_retopo.render import _write_png_rgb_in_blender

        if len(high_meshes) > 1:
            bpy.ops.object.select_all(action="DESELECT")
            for obj in high_meshes:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = high_meshes[0]
            bpy.ops.object.join()
            high_obj = bpy.context.view_layer.objects.active
        else:
            high_obj = high_meshes[0]

        if high_labels_npz is None or not Path(high_labels_npz).is_file():
            raise FileNotFoundError(f"isolated_part_bake=True requires valid high_labels_npz: {high_labels_npz}")
        with np.load(str(high_labels_npz), allow_pickle=False) as hd:
            high_face_labels = np.asarray(hd["face_labels"], dtype=np.int32).reshape(-1)
        n_high_faces = len(high_obj.data.polygons)
        if len(high_face_labels) != n_high_faces:
            raise ValueError(f"high_face_labels length {len(high_face_labels)} != highpoly polygons {n_high_faces}")

        n_low_faces = len(low_obj.data.polygons)
        slot_to_cid: dict[int, int] = {}
        for s_idx, slot in enumerate(low_obj.material_slots):
            mname = slot.material.name if slot.material is not None else ""
            for cid, cname in CANONICAL_MATERIAL_SLOTS:
                prefix = cname[:6]
                if mname.startswith(prefix) or cname.lower() in mname.lower():
                    slot_to_cid[s_idx] = cid
                    break

        if len(slot_to_cid) >= 2:
            mat_indices = np.empty(n_low_faces, dtype=np.int32)
            low_obj.data.polygons.foreach_get("material_index", mat_indices)
            low_face_cids = np.array([slot_to_cid.get(int(mi), 0) for mi in mat_indices], dtype=np.int32)
        elif low_labels_npz is not None and Path(low_labels_npz).is_file():
            with np.load(str(low_labels_npz), allow_pickle=False) as ld:
                low_face_cids = np.asarray(ld["face_labels"], dtype=np.int32).reshape(-1)
            if len(low_face_cids) != n_low_faces:
                raise ValueError(f"low_face_cids length {len(low_face_cids)} != lowpoly polygons {n_low_faces}")
        elif len(low_obj.material_slots) >= 1:
            mat_indices = np.empty(n_low_faces, dtype=np.int32)
            low_obj.data.polygons.foreach_get("material_index", mat_indices)
            low_face_cids = mat_indices.astype(np.int32)
        else:
            low_face_cids = np.zeros(n_low_faces, dtype=np.int32)

        n_high_verts = len(high_obj.data.vertices)
        high_co = np.empty(n_high_verts * 3, dtype=np.float32)
        high_obj.data.vertices.foreach_get("co", high_co)
        high_co = high_co.reshape(-1, 3)

        high_loop_total = np.empty(n_high_faces, dtype=np.int32)
        high_obj.data.polygons.foreach_get("loop_total", high_loop_total)
        if np.any(high_loop_total != 3):
            raise RuntimeError(f"Highpoly mesh contains non-triangle polygons: {np.unique(high_loop_total)}")
        high_faces = np.empty(n_high_faces * 3, dtype=np.int32)
        high_obj.data.loops.foreach_get("vertex_index", high_faces)
        high_faces = high_faces.reshape(-1, 3)
        high_discrete_curv = _blender_compute_discrete_curv_np(high_co.astype(np.float64), high_faces)

        n_low_verts = len(low_obj.data.vertices)
        low_co = np.empty(n_low_verts * 3, dtype=np.float32)
        low_obj.data.vertices.foreach_get("co", low_co)
        low_co = low_co.reshape(-1, 3)

        low_loop_total = np.empty(n_low_faces, dtype=np.int32)
        low_obj.data.polygons.foreach_get("loop_total", low_loop_total)
        if np.any(low_loop_total != 3):
            raise RuntimeError(f"Lowpoly mesh contains non-triangle polygons: {np.unique(low_loop_total)}")
        low_faces = np.empty(n_low_faces * 3, dtype=np.int32)
        low_obj.data.loops.foreach_get("vertex_index", low_faces)
        low_faces = low_faces.reshape(-1, 3)

        n_low_loops = n_low_faces * 3
        low_uvs = np.empty(n_low_loops * 2, dtype=np.float32)
        low_obj.data.uv_layers.active.data.foreach_get("uv", low_uvs)
        low_uvs = low_uvs.reshape(n_low_faces, 3, 2)

        low_corner_normals = np.empty(n_low_loops * 3, dtype=np.float32)
        low_obj.data.corner_normals.foreach_get("vector", low_corner_normals)
        low_corner_normals = low_corner_normals.reshape(n_low_faces, 3, 3)

        measured_gap_px = _blender_measure_uv_gap_px_4k(low_uvs, max_pad_check=max(pad * 2 + 4, 24))
        effective_pad = min(pad, max(4, int(measured_gap_px // 2) - 1)) if pad > 0 else 0

        low_cage_co = low_co.copy()
        for vi in range(n_low_verts):
            wco = mw @ low_obj.data.vertices[vi].co
            key = (round(float(wco.x), 5), round(float(wco.y), 5), round(float(wco.z), 5))
            wn = pos_normals.get(key)
            if wn is not None and wn.length > 1e-12:
                disp = mw3_inv @ (wn.normalized() * resolved_cage)
                low_cage_co[vi] += np.array([disp.x, disp.y, disp.z], dtype=np.float32)

        full_cage_data = low_obj.data.copy()
        full_cage_data.vertices.foreach_set("co", np.ascontiguousarray(low_cage_co, dtype=np.float32).ravel())
        full_cage_obj = bpy.data.objects.new("FullWeldedSmoothBakeCage", full_cage_data)
        full_cage_obj.matrix_world = mw.copy()
        bpy.context.scene.collection.objects.link(full_cage_obj)
        full_cage_obj.hide_render = True
        bpy.context.view_layer.update()

        high_obj.hide_render = True
        low_obj.hide_render = True

        mask_mat = bpy.data.materials.new("PartMaskMat")
        mnt = _ensure_nodes(mask_mat)
        mnt.nodes.clear()
        em = mnt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        em.inputs["Strength"].default_value = 1.0
        out_m = mnt.nodes.new("ShaderNodeOutputMaterial")
        mnt.links.new(em.outputs["Emission"], out_m.inputs["Surface"])
        mtex = mnt.nodes.new("ShaderNodeTexImage")
        mnt.nodes.active = mtex

        bake_mat = bpy.data.materials.new("PartBakeTargetMat")
        bnt = _ensure_nodes(bake_mat)
        btex = bnt.nodes.new("ShaderNodeTexImage")
        bnt.nodes.active = btex

        curv_mat = bpy.data.materials.new("PartCurvPointinessMat")
        cnt = _ensure_nodes(curv_mat)
        cnt.nodes.clear()
        geom_node = cnt.nodes.new("ShaderNodeNewGeometry")
        curv_em = cnt.nodes.new("ShaderNodeEmission")
        curv_em.inputs["Strength"].default_value = 1.0
        curv_out = cnt.nodes.new("ShaderNodeOutputMaterial")
        cnt.links.new(geom_node.outputs["Pointiness"], curv_em.inputs["Color"])
        cnt.links.new(curv_em.outputs["Emission"], curv_out.inputs["Surface"])

        comp_uv_mask = np.zeros((res, res), dtype=bool)
        comp_miss_mask = np.zeros((res, res), dtype=bool)
        comp_part_id = np.zeros((res, res), dtype=np.int16)
        comp_normal_rgb = np.full((res, res, 3), np.array([0.5, 0.5, 1.0], dtype=np.float32), dtype=np.float32)
        comp_curv_raw = np.full((res, res), 0.5, dtype=np.float32)
        uv_preview_rgba = np.full((res, res, 3), np.array([20, 22, 26], dtype=np.uint8), dtype=np.uint8)

        total_island_px = 0
        total_miss_px = 0
        total_hit_px = 0
        total_backface_px = 0
        per_part_stats: dict[str, Any] = {}
        high_mw = high_obj.matrix_world
        low_mw = low_obj.matrix_world
        rings = max(0, int(ring_dilation))
        head_rings = max(rings, int(head_ring_dilation))
        hair_head_rings = min(head_rings, max(rings, 8))
        head_hf_dilated_mask: Any = None

        def _dilate_high_face_rings(seed_mask: Any, n_steps: int, *, restrict_mask: Any = None) -> Any:
            mask_cur = seed_mask.copy()
            for _ in range(max(0, int(n_steps))):
                vm = np.zeros(n_high_verts, dtype=bool)
                vm[high_faces[mask_cur].ravel()] = True
                step_hits = np.any(vm[high_faces], axis=1)
                if restrict_mask is not None:
                    mask_cur = mask_cur | (step_hits & restrict_mask)
                else:
                    mask_cur = step_hits
            return mask_cur

        def _inpaint_part_buffer_on_mask(
            buf: Any,
            bad_mask: Any,
            fallback: Sequence[float],
            *,
            max_dist_px: float = 8.0,
            renormalize_normals: bool = False,
        ) -> Any:
            if not np.any(bad_mask):
                return buf
            out = buf.copy()
            for cid_val in np.unique(comp_part_id[bad_mask]):
                cid_i = int(cid_val)
                if cid_i <= 0:
                    continue
                part_m = comp_part_id == cid_i
                target_m = part_m & bad_mask
                valid_m = part_m & (~bad_mask)
                out = _blender_inpaint_masked_2d(
                    out,
                    valid_m,
                    target_m,
                    fallback,
                    max_dist_px=max_dist_px,
                )
            if renormalize_normals and out.ndim == 3 and out.shape[2] == 3:
                vec = out[bad_mask].astype(np.float64) * 2.0 - 1.0
                nrm_len = np.linalg.norm(vec, axis=-1, keepdims=True)
                good_v = nrm_len[:, 0] > 1e-8
                unit_v = np.where(
                    good_v[:, None],
                    vec / np.maximum(nrm_len, 1e-8),
                    np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
                )
                out[bad_mask] = np.clip(unit_v * 0.5 + 0.5, 0.0, 1.0).astype(out.dtype)
            return out

        def _bake_one_part(cid: int, cname: str, *, use_discrete_curv: bool = False) -> None:
            nonlocal total_island_px, total_miss_px, total_hit_px, total_backface_px, head_hf_dilated_mask

            low_f_mask = low_face_cids == cid
            n_lf = int(np.sum(low_f_mask))
            high_f_mask0 = high_face_labels == cid
            n_hf0 = int(np.sum(high_f_mask0))
            if n_lf == 0 or n_hf0 == 0:
                per_part_stats[cname] = {
                    "class_id": cid,
                    "low_faces": n_lf,
                    "high_faces_raw": n_hf0,
                    "high_faces_dilated": 0,
                    "uv_island_pixels": 0,
                    "ray_miss_pixels": 0,
                    "ray_miss_ratio": 0.0,
                    "backface_hit_ratio": 0.0,
                }
                return

            if cid == 1:
                hf_mask = _dilate_high_face_rings(high_f_mask0, head_rings)
                head_hf_dilated_mask = hf_mask.copy()
                part_rings_used = head_rings
            elif cid == 2:
                hf_mask = _dilate_high_face_rings(high_f_mask0, rings)
                extra_hair_head = max(0, hair_head_rings - rings)
                if extra_hair_head > 0:
                    hf_mask = _dilate_high_face_rings(
                        hf_mask,
                        extra_hair_head,
                        restrict_mask=(high_face_labels == 1),
                    )
                part_rings_used = hair_head_rings
            else:
                hf_mask = _dilate_high_face_rings(high_f_mask0, rings)
                part_rings_used = rings
            n_hf_dil = int(np.sum(hf_mask))

            lf_sub = low_faces[low_f_mask]
            used_lv, inv_lf = np.unique(lf_sub, return_inverse=True)
            local_lf = inv_lf.reshape(-1, 3).astype(np.int32)
            local_lv = low_co[used_lv]
            local_cage_lv = low_cage_co[used_lv]
            local_uv = low_uvs[low_f_mask].reshape(-1, 2)
            local_cn = low_corner_normals[low_f_mask].reshape(-1, 3)

            low_part_obj = _blender_build_mesh_obj(
                bpy,
                f"LowPart_{cid}",
                local_lv,
                local_lf,
                matrix_world=low_mw,
                uvs_per_loop=local_uv,
                custom_loop_normals=local_cn,
            )
            cage_part_obj = _blender_build_mesh_obj(
                bpy,
                f"CagePart_{cid}",
                local_cage_lv,
                local_lf,
                matrix_world=low_mw,
            )
            cage_part_obj.hide_render = True

            hf_sub = high_faces[hf_mask]
            used_hv, inv_hf = np.unique(hf_sub, return_inverse=True)
            local_hf = inv_hf.reshape(-1, 3).astype(np.int32)
            local_hv = high_co[used_hv]
            pt_attr = ("DiscreteCurvature", high_discrete_curv[used_hv]) if use_discrete_curv else None
            high_part_obj = _blender_build_mesh_obj(
                bpy,
                f"HighPart_{cid}",
                local_hv,
                local_hf,
                matrix_world=high_mw,
                point_attr=pt_attr,
            )
            bpy.context.view_layer.update()

            part_mask_img = bpy.data.images.new(f"PartMask_{cid}", width=res, height=res, alpha=True, float_buffer=False)
            part_mask_img.colorspace_settings.name = "Non-Color"
            mtex.image = part_mask_img
            mnt.nodes.active = mtex
            low_part_obj.data.materials.clear()
            low_part_obj.data.materials.append(mask_mat)
            bpy.ops.object.select_all(action="DESELECT")
            low_part_obj.select_set(True)
            bpy.context.view_layer.objects.active = low_part_obj
            scene.cycles.samples = 1
            bpy.ops.object.bake(type="EMIT", use_selected_to_active=False, margin=0, use_clear=True)

            pm_buf = np.empty(res * res * 4, dtype=np.float32)
            part_mask_img.pixels.foreach_get(pm_buf)
            part_uv_mask = pm_buf.reshape(res, res, 4)[:, :, 0] > 0.5
            bpy.data.images.remove(part_mask_img)

            part_px = int(np.sum(part_uv_mask))
            comp_uv_mask[:] |= part_uv_mask
            comp_part_id[part_uv_mask] = int(cid)
            col_u8 = np.array(PART_UV_PALETTE_U8.get(cid, (140, 145, 155)), dtype=np.uint8)
            uv_preview_rgba[part_uv_mask] = col_u8

            low_part_obj.data.materials[0] = bake_mat

            def _sel_part_h2l() -> None:
                bpy.ops.object.select_all(action="DESELECT")
                high_part_obj.select_set(True)
                low_part_obj.select_set(True)
                bpy.context.view_layer.objects.active = low_part_obj

            cage_kw = {"use_cage": True, "cage_object": cage_part_obj.name, "cage_extrusion": 0.0}

            scene.cycles.samples = max(1, int(normal_samples))
            sent_img = bpy.data.images.new(f"PartSent_{cid}", width=res, height=res, alpha=True, float_buffer=False)
            sent_img.colorspace_settings.name = "Non-Color"
            sent_img.pixels.foreach_set(np.zeros(res * res * 4, dtype=np.float32))
            btex.image = sent_img
            bnt.nodes.active = btex
            _sel_part_h2l()
            bpy.ops.object.bake(
                type="NORMAL",
                normal_space="TANGENT",
                normal_r="POS_X",
                normal_g="POS_Y",
                normal_b="POS_Z",
                use_selected_to_active=True,
                max_ray_distance=resolved_ray,
                margin=0,
                use_clear=False,
                **cage_kw,
            )
            sb = np.empty(res * res * 4, dtype=np.float32)
            sent_img.pixels.foreach_get(sb)
            sa = sb.reshape(res, res, 4)
            bpy.data.images.remove(sent_img)

            miss_m = part_uv_mask & np.all(sa[:, :, :3] <= 1e-4, axis=-1)
            hit_m = part_uv_mask & (~miss_m)
            hit_n = int(np.sum(hit_m))
            bf_m = hit_m & (sa[:, :, 2] <= 0.505)
            bf_n = int(np.sum(bf_m))
            miss_n = int(np.sum(miss_m))
            comp_miss_mask[part_uv_mask] = miss_m[part_uv_mask]

            total_island_px += part_px
            total_miss_px += miss_n
            total_hit_px += hit_n
            total_backface_px += bf_n

            pnrm_img = bpy.data.images.new(f"PartNrm_{cid}", width=res, height=res, alpha=False, float_buffer=False)
            pnrm_img.colorspace_settings.name = "Non-Color"
            btex.image = pnrm_img
            bnt.nodes.active = btex
            _sel_part_h2l()
            bpy.ops.object.bake(
                type="NORMAL",
                normal_space="TANGENT",
                normal_r="POS_X",
                normal_g="POS_Y",
                normal_b="POS_Z",
                use_selected_to_active=True,
                max_ray_distance=resolved_ray,
                margin=0,
                use_clear=True,
                **cage_kw,
            )
            nb = np.empty(res * res * 4, dtype=np.float32)
            pnrm_img.pixels.foreach_get(nb)
            na = nb.reshape(res, res, 4)[:, :, :3]
            comp_normal_rgb[part_uv_mask] = na[part_uv_mask]
            bpy.data.images.remove(pnrm_img)

            if not use_discrete_curv:
                high_part_obj.data.materials.clear()
                high_part_obj.data.materials.append(curv_mat)
            else:
                dc_mat = bpy.data.materials.new(f"PartDiscCurv_{cid}")
                dnt = _ensure_nodes(dc_mat)
                dnt.nodes.clear()
                an = dnt.nodes.new("ShaderNodeAttribute")
                an.attribute_name = "DiscreteCurvature"
                de = dnt.nodes.new("ShaderNodeEmission")
                de.inputs["Strength"].default_value = 1.0
                do = dnt.nodes.new("ShaderNodeOutputMaterial")
                dnt.links.new(an.outputs["Fac"], de.inputs["Color"])
                dnt.links.new(de.outputs["Emission"], do.inputs["Surface"])
                high_part_obj.data.materials.clear()
                high_part_obj.data.materials.append(dc_mat)

            scene.cycles.samples = max(1, int(curvature_samples))
            pcurv_img = bpy.data.images.new(f"PartCurv_{cid}", width=res, height=res, alpha=False, float_buffer=True)
            pcurv_img.colorspace_settings.name = "Non-Color"
            btex.image = pcurv_img
            bnt.nodes.active = btex
            _sel_part_h2l()
            bpy.ops.object.bake(
                type="EMIT",
                use_selected_to_active=True,
                max_ray_distance=resolved_ray,
                margin=0,
                use_clear=True,
                **cage_kw,
            )
            cb = np.empty(res * res * 4, dtype=np.float32)
            pcurv_img.pixels.foreach_get(cb)
            ca = cb.reshape(res, res, 4)[:, :, 0]
            comp_curv_raw[part_uv_mask] = ca[part_uv_mask]
            bpy.data.images.remove(pcurv_img)

            per_part_stats[cname] = {
                "class_id": cid,
                "rings_used": int(part_rings_used),
                "low_faces": n_lf,
                "high_faces_raw": n_hf0,
                "high_faces_dilated": n_hf_dil,
                "uv_island_pixels": part_px,
                "ray_miss_pixels": miss_n,
                "ray_miss_ratio": float(miss_n / part_px) if part_px > 0 else 0.0,
                "backface_hit_pixels": bf_n,
                "backface_hit_ratio": float(bf_n / hit_n) if hit_n > 0 else 0.0,
            }

            for tmp_obj in (low_part_obj, cage_part_obj, high_part_obj):
                mdata = tmp_obj.data
                bpy.data.objects.remove(tmp_obj, do_unlink=True)
                if mdata.users == 0:
                    bpy.data.meshes.remove(mdata)

        slot_name_map = dict(CANONICAL_MATERIAL_SLOTS)
        present_part_cids = sorted(int(c) for c in np.unique(low_face_cids))
        for cid in present_part_cids:
            cname = slot_name_map.get(cid, f"Mat_{cid:02d}_Part")
            _bake_one_part(cid, cname, use_discrete_curv=False)

        if np.any(comp_miss_mask):
            comp_curv_raw = _inpaint_part_buffer_on_mask(
                comp_curv_raw,
                comp_miss_mask,
                (0.5,),
                max_dist_px=8.0,
            )
        curv_norm, curv_stats = normalize_curvature_buffer(comp_curv_raw, comp_uv_mask)
        curvature_source = f"blender_pointiness_isolated_{rings}ring_h{head_rings}"
        if curv_stats["dynamic_range"] < 0.02:
            total_island_px = 0
            total_miss_px = 0
            total_hit_px = 0
            total_backface_px = 0
            comp_miss_mask.fill(False)
            for cid, cname in CANONICAL_MATERIAL_SLOTS:
                _bake_one_part(cid, cname, use_discrete_curv=True)
            if np.any(comp_miss_mask):
                comp_curv_raw = _inpaint_part_buffer_on_mask(
                    comp_curv_raw,
                    comp_miss_mask,
                    (0.5,),
                    max_dist_px=8.0,
                )
            curv_norm, curv_stats = normalize_curvature_buffer(comp_curv_raw, comp_uv_mask)
            curvature_source = f"discrete_mean_curvature_isolated_{rings}ring_h{head_rings}"

        uv_island_pixels = int(np.sum(comp_uv_mask))
        ray_miss_pixels = int(total_miss_px)
        ray_miss_ratio = float(total_miss_px / total_island_px) if total_island_px > 0 else 1.0
        backface_hit_ratio = float(total_backface_px / total_hit_px) if total_hit_px > 0 else 0.0

        guard_mask = comp_uv_mask & (comp_miss_mask | (comp_normal_rgb[:, :, 2] <= 0.58))
        backface_guarded_pixels = int(np.sum(guard_mask))
        if backface_guarded_pixels > 0:
            comp_normal_rgb = _inpaint_part_buffer_on_mask(
                comp_normal_rgb,
                guard_mask,
                (0.5, 0.5, 1.0),
                max_dist_px=8.0,
                renormalize_normals=True,
            )

        dilated_normal = _blender_dilate_2d_margin(
            comp_normal_rgb,
            comp_uv_mask,
            effective_pad,
            (0.5, 0.5, 1.0),
            renormalize_normals=True,
        )
        out_normal_png.parent.mkdir(parents=True, exist_ok=True)
        nrm_out_img = bpy.data.images.new("BakedNormal4K", width=res, height=res, alpha=False, float_buffer=False)
        nrm_out_img.colorspace_settings.name = "Non-Color"
        nrm_rgba = np.ones((res, res, 4), dtype=np.float32)
        nrm_rgba[:, :, :3] = np.clip(dilated_normal, 0.0, 1.0)
        nrm_out_img.pixels.foreach_set(nrm_rgba.ravel())
        nrm_out_img.filepath_raw = str(out_normal_png)
        nrm_out_img.file_format = "PNG"
        nrm_out_img.save()
        bpy.data.images.remove(nrm_out_img)

        dilated_curv = _blender_dilate_2d_margin(curv_norm[:, :, None], comp_uv_mask, effective_pad, (0.5,))[:, :, 0]
        out_curvature_png.parent.mkdir(parents=True, exist_ok=True)
        curv_out_img = bpy.data.images.new("BakedCurv4K", width=res, height=res, alpha=False, float_buffer=False)
        curv_out_img.colorspace_settings.name = "Non-Color"
        curv_rgba = np.ones((res, res, 4), dtype=np.float32)
        curv_rgba[:, :, 0] = dilated_curv
        curv_rgba[:, :, 1] = dilated_curv
        curv_rgba[:, :, 2] = dilated_curv
        curv_out_img.pixels.foreach_set(curv_rgba.ravel())
        curv_out_img.filepath_raw = str(out_curvature_png)
        curv_out_img.file_format = "PNG"
        curv_out_img.save()
        bpy.data.images.remove(curv_out_img)

        # AO Bake: full highpoly 24-sample AO for slots 2..7; optional hair-excluded highpoly AO for Mat_01_HeadFace
        high_obj.hide_render = False
        low_obj.hide_render = False

        ao_mat = bpy.data.materials.new("HighpolyFullAOMat")
        ant = _ensure_nodes(ao_mat)
        ant.nodes.clear()
        ao_node = ant.nodes.new("ShaderNodeAmbientOcclusion")
        ao_node.samples = 8
        ao_node.only_local = False
        ao_node.inside = False
        ao_node.inputs["Distance"].default_value = float(ao_distance_rel) * diag
        ao_em = ant.nodes.new("ShaderNodeEmission")
        ao_em.inputs["Strength"].default_value = 1.0
        ao_out = ant.nodes.new("ShaderNodeOutputMaterial")
        ant.links.new(ao_node.outputs["AO"], ao_em.inputs["Color"])
        ant.links.new(ao_em.outputs["Emission"], ao_out.inputs["Surface"])

        high_obj.data.materials.clear()
        high_obj.data.materials.append(ao_mat)
        low_obj.data.materials.clear()
        low_obj.data.materials.append(bake_mat)

        scene.cycles.samples = max(1, int(ao_samples))
        full_ao_miss_pixels = 0
        head_ao_miss_pixels = 0
        ao_img = bpy.data.images.new("BakedFullAO4K", width=res, height=res, alpha=True, float_buffer=True)
        ao_img.colorspace_settings.name = "Non-Color"
        ao_img.pixels.foreach_set(np.full(res * res * 4, -1.0, dtype=np.float32))
        btex.image = ao_img
        bnt.nodes.active = btex
        bpy.ops.object.select_all(action="DESELECT")
        high_obj.select_set(True)
        low_obj.select_set(True)
        bpy.context.view_layer.objects.active = low_obj
        bpy.ops.object.bake(
            type="EMIT",
            use_selected_to_active=True,
            max_ray_distance=resolved_ray,
            margin=0,
            use_clear=False,
            use_cage=True,
            cage_object=full_cage_obj.name,
            cage_extrusion=0.0,
        )
        ao_buf = np.empty(res * res * 4, dtype=np.float32)
        ao_img.pixels.foreach_get(ao_buf)
        comp_ao_r = ao_buf.reshape(res, res, 4)[:, :, 0].copy()
        bpy.data.images.remove(ao_img)

        full_ao_miss_mask = comp_uv_mask & (comp_ao_r < -0.5)
        full_ao_miss_pixels = int(np.sum(full_ao_miss_mask))
        comp_ao_r = np.where(comp_ao_r >= -0.5, np.clip(comp_ao_r, 0.0, 1.0), -1.0).astype(np.float32)
        if full_ao_miss_pixels > 0:
            comp_ao_r = _inpaint_part_buffer_on_mask(
                comp_ao_r,
                full_ao_miss_mask,
                (1.0,),
                max_dist_px=12.0,
            )
        comp_ao_r = np.where(comp_uv_mask, np.clip(comp_ao_r, 0.0, 1.0), 1.0).astype(np.float32)

        head_low_mask = low_face_cids == 1
        nohair_high_mask = high_face_labels != 2
        if (
            head_ao_exclude_hair
            and np.any(head_low_mask)
            and np.any(high_face_labels == 2)
            and np.any(nohair_high_mask)
        ):
            high_obj.hide_render = True
            low_obj.hide_render = True

            if head_hf_dilated_mask is None:
                head_hf_dilated_mask = _dilate_high_face_rings(high_face_labels == 1, head_rings)
            head_hair_collar_mask = head_hf_dilated_mask & (high_face_labels == 2)

            hf_nh = high_faces[nohair_high_mask]
            u_nh, inv_nh = np.unique(hf_nh, return_inverse=True)
            high_nohair_obj = _blender_build_mesh_obj(
                bpy,
                "HighNoHairAO",
                high_co[u_nh],
                inv_nh.reshape(-1, 3).astype(np.int32),
                matrix_world=high_mw,
            )
            high_nohair_obj.data.materials.append(ao_mat)
            high_nohair_obj.visible_shadow = True

            high_head_hair_collar_obj = None
            if np.any(head_hair_collar_mask):
                hf_hc = high_faces[head_hair_collar_mask]
                u_hc, inv_hc = np.unique(hf_hc, return_inverse=True)
                high_head_hair_collar_obj = _blender_build_mesh_obj(
                    bpy,
                    "HighHeadHairCollarNoShadow",
                    high_co[u_hc],
                    inv_hc.reshape(-1, 3).astype(np.int32),
                    matrix_world=high_mw,
                )
                high_head_hair_collar_obj.data.materials.append(ao_mat)
                high_head_hair_collar_obj.visible_shadow = False

            lf_hd = low_faces[head_low_mask]
            u_lh, inv_lh = np.unique(lf_hd, return_inverse=True)
            local_lh_f = inv_lh.reshape(-1, 3).astype(np.int32)
            low_head_obj = _blender_build_mesh_obj(
                bpy,
                "LowHeadAO",
                low_co[u_lh],
                local_lh_f,
                matrix_world=low_mw,
                uvs_per_loop=low_uvs[head_low_mask].reshape(-1, 2),
                custom_loop_normals=low_corner_normals[head_low_mask].reshape(-1, 3),
            )
            cage_head_obj = _blender_build_mesh_obj(
                bpy,
                "CageHeadAO",
                low_cage_co[u_lh],
                local_lh_f,
                matrix_world=low_mw,
            )
            cage_head_obj.hide_render = True
            bpy.context.view_layer.update()

            hmask_img = bpy.data.images.new("HeadAOMask", width=res, height=res, alpha=True, float_buffer=False)
            hmask_img.colorspace_settings.name = "Non-Color"
            mtex.image = hmask_img
            mnt.nodes.active = mtex
            low_head_obj.data.materials.clear()
            low_head_obj.data.materials.append(mask_mat)
            bpy.ops.object.select_all(action="DESELECT")
            low_head_obj.select_set(True)
            bpy.context.view_layer.objects.active = low_head_obj
            scene.cycles.samples = 1
            bpy.ops.object.bake(type="EMIT", use_selected_to_active=False, margin=0, use_clear=True)
            hmb = np.empty(res * res * 4, dtype=np.float32)
            hmask_img.pixels.foreach_get(hmb)
            head_uv_mask = hmb.reshape(res, res, 4)[:, :, 0] > 0.5
            bpy.data.images.remove(hmask_img)

            low_head_obj.data.materials[0] = bake_mat
            hao_img = bpy.data.images.new("HeadNoHairAO4K", width=res, height=res, alpha=True, float_buffer=True)
            hao_img.colorspace_settings.name = "Non-Color"
            hao_img.pixels.foreach_set(np.full(res * res * 4, -1.0, dtype=np.float32))
            btex.image = hao_img
            bnt.nodes.active = btex
            scene.cycles.samples = max(1, int(ao_samples))
            bpy.ops.object.select_all(action="DESELECT")
            high_nohair_obj.select_set(True)
            if high_head_hair_collar_obj is not None:
                high_head_hair_collar_obj.select_set(True)
            low_head_obj.select_set(True)
            bpy.context.view_layer.objects.active = low_head_obj
            bpy.ops.object.bake(
                type="EMIT",
                use_selected_to_active=True,
                max_ray_distance=resolved_ray,
                margin=0,
                use_clear=False,
                use_cage=True,
                cage_object=cage_head_obj.name,
                cage_extrusion=0.0,
            )
            haob = np.empty(res * res * 4, dtype=np.float32)
            hao_img.pixels.foreach_get(haob)
            head_ao_r = haob.reshape(res, res, 4)[:, :, 0].copy()
            bpy.data.images.remove(hao_img)

            head_ao_miss_mask = head_uv_mask & (head_ao_r < -0.5)
            head_ao_miss_pixels = int(np.sum(head_ao_miss_mask))
            head_valid = head_uv_mask & (~head_ao_miss_mask)
            comp_ao_r[head_valid] = np.clip(head_ao_r[head_valid], 0.0, 1.0)
            if head_ao_miss_pixels > 0:
                comp_ao_r = _blender_inpaint_masked_2d(
                    comp_ao_r,
                    head_valid,
                    head_ao_miss_mask,
                    (1.0,),
                    max_dist_px=12.0,
                )

            if "Mat_01_HeadFace" in per_part_stats:
                per_part_stats["Mat_01_HeadFace"].update(
                    {
                        "head_ring_dilation": int(head_rings),
                        "head_ao_hair_collar_faces": int(np.sum(head_hair_collar_mask)),
                        "head_ao_miss_pixels": int(head_ao_miss_pixels),
                        "head_ao_zero_pixels": int(np.sum(head_uv_mask & (comp_ao_r < 0.02))),
                    }
                )

            cleanup_head_objs = [high_nohair_obj, low_head_obj, cage_head_obj]
            if high_head_hair_collar_obj is not None:
                cleanup_head_objs.append(high_head_hair_collar_obj)
            for tmp_obj in cleanup_head_objs:
                mdata = tmp_obj.data
                bpy.data.objects.remove(tmp_obj, do_unlink=True)
                if mdata.users == 0:
                    bpy.data.meshes.remove(mdata)

            high_obj.hide_render = False
            low_obj.hide_render = False

        ao_masked = comp_ao_r[comp_uv_mask] if np.any(comp_uv_mask) else comp_ao_r.reshape(-1)
        ao_mean = float(np.mean(ao_masked))
        ao_std = float(np.std(ao_masked))
        ao_p05 = float(np.percentile(ao_masked, 5.0))
        ao_p95 = float(np.percentile(ao_masked, 95.0))

        dilated_ao = _blender_dilate_2d_margin(comp_ao_r[:, :, None], comp_uv_mask, effective_pad, (1.0,))[:, :, 0]
        out_ao_png.parent.mkdir(parents=True, exist_ok=True)
        ao_out_img = bpy.data.images.new("BakedAO4K", width=res, height=res, alpha=False, float_buffer=False)
        ao_out_img.colorspace_settings.name = "Non-Color"
        ao_rgba = np.ones((res, res, 4), dtype=np.float32)
        ao_rgba[:, :, 0] = dilated_ao
        ao_rgba[:, :, 1] = dilated_ao
        ao_rgba[:, :, 2] = dilated_ao
        ao_out_img.pixels.foreach_set(ao_rgba.ravel())
        ao_out_img.filepath_raw = str(out_ao_png)
        ao_out_img.file_format = "PNG"
        ao_out_img.save()
        bpy.data.images.remove(ao_out_img)

        preview_for_sheet = uv_preview_png
        if out_contact_sheet_png is not None and (preview_for_sheet is None or not preview_for_sheet.exists()):
            fallback_prev = out_baked_glb.parent / f"._{out_baked_glb.stem}_uv_mask_preview.png"
            _write_png_rgb_in_blender(fallback_prev, uv_preview_rgba[::-1, :, :])
            preview_for_sheet = fallback_prev
    else:
        effective_pad = pad
        low_face_cids = None
        per_part_stats = {}
        head_rings = 0
        full_ao_miss_pixels = 0
        head_ao_miss_pixels = 0
        cage_obj = None
        if len(low_verts) > 0 and len(low_obj.data.polygons) > 0:
            cage_data = low_obj.data.copy()
            cage_obj = bpy.data.objects.new("WeldedSmoothBakeCage", cage_data)
            cage_obj.matrix_world = mw.copy()
            bpy.context.scene.collection.objects.link(cage_obj)
            cage_obj.hide_render = True
            for cv in cage_data.vertices:
                wco = mw @ cv.co
                key = (round(float(wco.x), 5), round(float(wco.y), 5), round(float(wco.z), 5))
                wn = pos_normals.get(key)
                if wn is not None and wn.length > 1e-12:
                    cv.co = cv.co + (mw3_inv @ (wn.normalized() * resolved_cage))
            bpy.context.view_layer.update()

        bake_cage_kwargs: dict[str, Any] = (
            {"use_cage": True, "cage_object": cage_obj.name, "cage_extrusion": 0.0}
            if cage_obj is not None
            else {"cage_extrusion": resolved_cage}
        )

        # Step 1: Exact Cycles UV island mask (margin=0 EMIT white)
        mask_img = bpy.data.images.new("UVMask4K", width=res, height=res, alpha=True, float_buffer=False)
        mask_img.colorspace_settings.name = "Non-Color"
        mask_mat = bpy.data.materials.new("BakeMaskMat")
        mnt = _ensure_nodes(mask_mat)
        mnt.nodes.clear()
        em = mnt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        em.inputs["Strength"].default_value = 1.0
        out_m = mnt.nodes.new("ShaderNodeOutputMaterial")
        mnt.links.new(em.outputs["Emission"], out_m.inputs["Surface"])
        mtex = mnt.nodes.new("ShaderNodeTexImage")
        mtex.image = mask_img
        mnt.nodes.active = mtex

        low_obj.data.materials.clear()
        low_obj.data.materials.append(mask_mat)
        bpy.ops.object.select_all(action="DESELECT")
        low_obj.select_set(True)
        bpy.context.view_layer.objects.active = low_obj
        scene.cycles.samples = 1
        bpy.ops.object.bake(type="EMIT", use_selected_to_active=False, margin=0, use_clear=True)

        mask_buf = np.empty(res * res * 4, dtype=np.float32)
        mask_img.pixels.foreach_get(mask_buf)
        uv_mask = mask_buf.reshape(res, res, 4)[:, :, 0] > 0.5
        uv_island_pixels = int(np.sum(uv_mask))
        bpy.data.images.remove(mask_img)

        bake_mat = bpy.data.materials.new("BakeTargetMat4K")
        bnt = _ensure_nodes(bake_mat)
        btex = bnt.nodes.new("ShaderNodeTexImage")
        bnt.nodes.active = btex
        low_obj.data.materials[0] = bake_mat

        def _select_high_to_low() -> None:
            bpy.ops.object.select_all(action="DESELECT")
            for hobj in high_meshes:
                hobj.select_set(True)
            low_obj.select_set(True)
            bpy.context.view_layer.objects.active = low_obj

        # Step 2: 4K Tangent-Space Normal Map (sentinel + final + backface guard)
        scene.cycles.samples = max(1, int(normal_samples))
        sentinel_img = bpy.data.images.new("SentinelNormal4K", width=res, height=res, alpha=True, float_buffer=False)
        sentinel_img.colorspace_settings.name = "Non-Color"
        sentinel_img.pixels.foreach_set(np.zeros(res * res * 4, dtype=np.float32))
        btex.image = sentinel_img
        bnt.nodes.active = btex

        _select_high_to_low()
        bpy.ops.object.bake(
            type="NORMAL",
            normal_space="TANGENT",
            normal_r="POS_X",
            normal_g="POS_Y",
            normal_b="POS_Z",
            use_selected_to_active=True,
            max_ray_distance=resolved_ray,
            margin=0,
            use_clear=False,
            **bake_cage_kwargs,
        )
        sent_buf = np.empty(res * res * 4, dtype=np.float32)
        sentinel_img.pixels.foreach_get(sent_buf)
        sent_arr = sent_buf.reshape(res, res, 4)
        miss_mask = uv_mask & np.all(sent_arr[:, :, :3] <= 1e-4, axis=-1)
        hit_mask = uv_mask & (~miss_mask)
        hit_count = int(np.sum(hit_mask))
        backface_mask = hit_mask & (sent_arr[:, :, 2] <= 0.505)
        backface_hit_ratio = float(np.sum(backface_mask) / hit_count) if hit_count > 0 else 0.0
        ray_miss_pixels = int(np.sum(miss_mask))
        ray_miss_ratio = float(ray_miss_pixels / uv_island_pixels) if uv_island_pixels > 0 else 1.0
        bpy.data.images.remove(sentinel_img)

        nrm_img = bpy.data.images.new("BakedNormal4K", width=res, height=res, alpha=False, float_buffer=False)
        nrm_img.colorspace_settings.name = "Non-Color"
        btex.image = nrm_img
        bnt.nodes.active = btex
        _select_high_to_low()
        bpy.ops.object.bake(
            type="NORMAL",
            normal_space="TANGENT",
            normal_r="POS_X",
            normal_g="POS_Y",
            normal_b="POS_Z",
            use_selected_to_active=True,
            max_ray_distance=resolved_ray,
            margin=pad,
            margin_type="EXTEND",
            use_clear=True,
            **bake_cage_kwargs,
        )

        nrm_buf = np.empty(res * res * 4, dtype=np.float32)
        nrm_img.pixels.foreach_get(nrm_buf)
        nrm_arr = nrm_buf.reshape(res, res, 4)
        guard_mask = nrm_arr[:, :, 2] <= 0.58
        backface_guarded_pixels = int(np.sum(guard_mask))
        if backface_guarded_pixels > 0:
            nrm_arr[guard_mask] = np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float32)
            nrm_img.pixels.foreach_set(nrm_arr.ravel())
            nrm_img.update()

        out_normal_png.parent.mkdir(parents=True, exist_ok=True)
        nrm_img.filepath_raw = str(out_normal_png)
        nrm_img.file_format = "PNG"
        nrm_img.save()
        bpy.data.images.remove(nrm_img)

        # Step 3: 4K Ambient Occlusion Map (ShaderNodeAmbientOcclusion -> EMIT)
        ao_mat = bpy.data.materials.new("HighpolyAOMat")
        ant = _ensure_nodes(ao_mat)
        ant.nodes.clear()
        ao_node = ant.nodes.new("ShaderNodeAmbientOcclusion")
        ao_node.samples = 8
        ao_node.only_local = False
        ao_node.inside = False
        ao_node.inputs["Distance"].default_value = float(ao_distance_rel) * diag
        ao_em = ant.nodes.new("ShaderNodeEmission")
        ao_em.inputs["Strength"].default_value = 1.0
        ao_out = ant.nodes.new("ShaderNodeOutputMaterial")
        ant.links.new(ao_node.outputs["AO"], ao_em.inputs["Color"])
        ant.links.new(ao_em.outputs["Emission"], ao_out.inputs["Surface"])

        for hobj in high_meshes:
            hobj.data.materials.clear()
            hobj.data.materials.append(ao_mat)

        scene.cycles.samples = max(1, int(ao_samples))
        ao_img = bpy.data.images.new("BakedAO4K", width=res, height=res, alpha=False, float_buffer=False)
        ao_img.colorspace_settings.name = "Non-Color"
        btex.image = ao_img
        bnt.nodes.active = btex
        _select_high_to_low()
        bpy.ops.object.bake(
            type="EMIT",
            use_selected_to_active=True,
            max_ray_distance=resolved_ray,
            margin=pad,
            margin_type="EXTEND",
            use_clear=True,
            **bake_cage_kwargs,
        )

        ao_buf = np.empty(res * res * 4, dtype=np.float32)
        ao_img.pixels.foreach_get(ao_buf)
        ao_r = ao_buf.reshape(res, res, 4)[:, :, 0]
        ao_masked = ao_r[uv_mask] if np.any(uv_mask) else ao_r.reshape(-1)
        ao_mean = float(np.mean(ao_masked))
        ao_std = float(np.std(ao_masked))
        ao_p05 = float(np.percentile(ao_masked, 5.0))
        ao_p95 = float(np.percentile(ao_masked, 95.0))

        out_ao_png.parent.mkdir(parents=True, exist_ok=True)
        ao_img.filepath_raw = str(out_ao_png)
        ao_img.file_format = "PNG"
        ao_img.save()
        bpy.data.images.remove(ao_img)

        # Step 4: 4K Curvature Map (Pointiness float bake + discrete fallback)
        curv_mat = bpy.data.materials.new("HighpolyCurvatureMat")
        cnt = _ensure_nodes(curv_mat)
        cnt.nodes.clear()
        geom_node = cnt.nodes.new("ShaderNodeNewGeometry")
        curv_em = cnt.nodes.new("ShaderNodeEmission")
        curv_em.inputs["Strength"].default_value = 1.0
        curv_out = cnt.nodes.new("ShaderNodeOutputMaterial")
        cnt.links.new(geom_node.outputs["Pointiness"], curv_em.inputs["Color"])
        cnt.links.new(curv_em.outputs["Emission"], curv_out.inputs["Surface"])

        for hobj in high_meshes:
            hobj.data.materials.clear()
            hobj.data.materials.append(curv_mat)

        scene.cycles.samples = max(1, int(curvature_samples))
        curv_raw_img = bpy.data.images.new("BakedCurvFloat4K", width=res, height=res, alpha=False, float_buffer=True)
        curv_raw_img.colorspace_settings.name = "Non-Color"
        btex.image = curv_raw_img
        bnt.nodes.active = btex
        _select_high_to_low()
        bpy.ops.object.bake(
            type="EMIT",
            use_selected_to_active=True,
            max_ray_distance=resolved_ray,
            margin=pad,
            margin_type="EXTEND",
            use_clear=True,
            **bake_cage_kwargs,
        )

        curv_buf = np.empty(res * res * 4, dtype=np.float32)
        curv_raw_img.pixels.foreach_get(curv_buf)
        curv_r = curv_buf.reshape(res, res, 4)[:, :, 0]
        curv_norm, curv_stats = normalize_curvature_buffer(curv_r, uv_mask)
        curvature_source = "blender_pointiness"

        if curv_stats["dynamic_range"] < 0.02:
            for hobj in high_meshes:
                _blender_assign_discrete_curvature_attribute(hobj, "DiscreteCurvature")
            cnt.nodes.clear()
            attr_node = cnt.nodes.new("ShaderNodeAttribute")
            attr_node.attribute_name = "DiscreteCurvature"
            curv_em2 = cnt.nodes.new("ShaderNodeEmission")
            curv_em2.inputs["Strength"].default_value = 1.0
            curv_out2 = cnt.nodes.new("ShaderNodeOutputMaterial")
            cnt.links.new(attr_node.outputs["Fac"], curv_em2.inputs["Color"])
            cnt.links.new(curv_em2.outputs["Emission"], curv_out2.inputs["Surface"])
            _select_high_to_low()
            bpy.ops.object.bake(
                type="EMIT",
                use_selected_to_active=True,
                max_ray_distance=resolved_ray,
                margin=pad,
                margin_type="EXTEND",
                use_clear=True,
                **bake_cage_kwargs,
            )
            curv_raw_img.pixels.foreach_get(curv_buf)
            curv_r = curv_buf.reshape(res, res, 4)[:, :, 0]
            curv_norm, curv_stats = normalize_curvature_buffer(curv_r, uv_mask)
            curvature_source = "discrete_mean_curvature"

        bpy.data.images.remove(curv_raw_img)

        curv_out_img = bpy.data.images.new("BakedCurv4K", width=res, height=res, alpha=False, float_buffer=False)
        curv_out_img.colorspace_settings.name = "Non-Color"
        curv_rgba = np.ones((res, res, 4), dtype=np.float32)
        curv_rgba[:, :, 0] = curv_norm
        curv_rgba[:, :, 1] = curv_norm
        curv_rgba[:, :, 2] = curv_norm
        curv_out_img.pixels.foreach_set(curv_rgba.ravel())
        out_curvature_png.parent.mkdir(parents=True, exist_ok=True)
        curv_out_img.filepath_raw = str(out_curvature_png)
        curv_out_img.file_format = "PNG"
        curv_out_img.save()
        bpy.data.images.remove(curv_out_img)
        preview_for_sheet = uv_preview_png

    # ------------------------------------------------------------------
    # Step 5: Export self-contained baked GLB (packed 4K Normal + AO)
    # ------------------------------------------------------------------
    if low_obj.parent is not None:
        mw_keep = low_obj.matrix_world.copy()
        low_obj.parent = None
        low_obj.matrix_world = mw_keep
    for obj in list(bpy.data.objects):
        if obj != low_obj:
            mdata = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mdata is not None and mdata.users == 0:
                bpy.data.meshes.remove(mdata)

    packed_nrm = bpy.data.images.load(str(out_normal_png))
    packed_nrm.colorspace_settings.name = "Non-Color"
    packed_nrm.pack()

    packed_ao = bpy.data.images.load(str(out_ao_png))
    packed_ao.colorspace_settings.name = "Non-Color"
    packed_ao.pack()

    def _build_clay_slot_mat(mat_name: str) -> Any:
        clay_mat = bpy.data.materials.new(mat_name)
        lnt = _ensure_nodes(clay_mat)
        bsdf = lnt.nodes.get("Principled BSDF")
        if bsdf is None:
            bsdf = lnt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 0.62 if not preserve_material_slots else 0.60
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.35

        ao_tex = lnt.nodes.new("ShaderNodeTexImage")
        ao_tex.image = packed_ao
        ao_tex.interpolation = "Linear"
        ao_tex.extension = "EXTEND"

        mix_ao = lnt.nodes.new("ShaderNodeMix")
        mix_ao.data_type = "RGBA"
        mix_ao.blend_type = "MULTIPLY"
        mix_ao.inputs[0].default_value = 0.75
        mix_ao.inputs[6].default_value = (0.60, 0.60, 0.61, 1.0)
        lnt.links.new(ao_tex.outputs["Color"], mix_ao.inputs[7])
        lnt.links.new(mix_ao.outputs[2], bsdf.inputs["Base Color"])

        nrm_tex = lnt.nodes.new("ShaderNodeTexImage")
        nrm_tex.image = packed_nrm
        nrm_tex.interpolation = "Linear"
        nrm_tex.extension = "EXTEND"
        nrm_node = lnt.nodes.new("ShaderNodeNormalMap")
        nrm_node.space = "TANGENT"
        nrm_node.inputs["Strength"].default_value = 1.0
        lnt.links.new(nrm_tex.outputs["Color"], nrm_node.inputs["Color"])
        lnt.links.new(nrm_node.outputs["Normal"], bsdf.inputs["Normal"])
        return clay_mat

    exported_mat_slots: list[str] = []
    if preserve_material_slots and low_face_cids is not None:
        low_obj.data.materials.clear()
        cid_to_slot: dict[int, int] = {}
        for slot_idx, (cid, cname) in enumerate(CANONICAL_MATERIAL_SLOTS):
            low_obj.data.materials.append(_build_clay_slot_mat(cname))
            cid_to_slot[cid] = slot_idx
            exported_mat_slots.append(cname)
        mat_idx_arr = np.array([cid_to_slot.get(int(c), 0) for c in low_face_cids], dtype=np.int32)
        low_obj.data.polygons.foreach_set("material_index", mat_idx_arr)
        low_obj.data.update()
    else:
        low_obj.data.materials.clear()
        low_obj.data.materials.append(_build_clay_slot_mat("NeutralClayBaked4KMat"))
        exported_mat_slots = ["NeutralClayBaked4KMat"]

    out_baked_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    low_obj.select_set(True)
    bpy.context.view_layer.objects.active = low_obj
    bpy.ops.export_scene.gltf(
        filepath=str(out_baked_glb),
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_tangents=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
    )

    sheet_str: str | None = None
    if out_contact_sheet_png is not None:
        sheet_labels = [
            f"NORMAL {res} | MISS {ray_miss_ratio * 100.0:.2f}% | BF {backface_hit_ratio * 100.0:.2f}%",
            f"AO {res} | MEAN {ao_mean:.2f} | STD {ao_std:.2f}",
            f"CURVATURE {res} | P05 {curv_stats['p05']:.2f} P95 {curv_stats['p95']:.2f}",
            "PART-AWARE UV LAYOUT (7 SLOTS)"
            if (resolved_isolated or (preview_for_sheet and preview_for_sheet.exists()))
            else f"NORMAL {res}",
        ]
        _write_4k_contact_sheet_in_blender(
            normal_png=out_normal_png,
            ao_png=out_ao_png,
            curvature_png=out_curvature_png,
            uv_preview_png=preview_for_sheet,
            out_sheet_png=out_contact_sheet_png,
            labels=sheet_labels,
            tile_size=1024,
        )
        sheet_str = str(out_contact_sheet_png)

    res_obj = MultiMapBakeResult(
        high_mesh_path=str(high_mesh_path),
        low_uv_mesh_path=str(low_uv_mesh_path),
        out_normal_png=str(out_normal_png),
        out_ao_png=str(out_ao_png),
        out_curvature_png=str(out_curvature_png),
        out_baked_glb=str(out_baked_glb),
        out_contact_sheet_png=sheet_str,
        tex_size=res,
        padding_px=pad,
        normal_samples=max(1, int(normal_samples)),
        ao_samples=max(1, int(ao_samples)),
        curvature_samples=max(1, int(curvature_samples)),
        use_gpu=bool(use_gpu),
        device_used=device_used,
        diag=diag,
        cage_extrusion=resolved_cage,
        max_ray_distance=resolved_ray,
        surface_dist_p95=d_p95,
        surface_dist_p99=d_p99,
        uv_island_pixels=uv_island_pixels,
        ray_miss_pixels=ray_miss_pixels,
        ray_miss_ratio=ray_miss_ratio,
        backface_hit_ratio=backface_hit_ratio,
        backface_guarded_pixels=backface_guarded_pixels,
        ao_mean=ao_mean,
        ao_std=ao_std,
        ao_p05=ao_p05,
        ao_p95=ao_p95,
        curvature_mean=float(curv_stats["mean"]),
        curvature_std=float(curv_stats["std"]),
        curvature_p05=float(curv_stats["p05"]),
        curvature_p95=float(curv_stats["p95"]),
        curvature_dynamic_range=float(curv_stats["dynamic_range"]),
        curvature_source=curvature_source,
        bake_elapsed_s=time.perf_counter() - t0,
        blender_version=".".join(str(x) for x in bpy.app.version),
        isolated_part_bake=bool(resolved_isolated),
        ring_dilation=int(ring_dilation) if resolved_isolated else 0,
        head_ring_dilation=int(head_rings) if resolved_isolated else 0,
        head_ao_exclude_hair=bool(head_ao_exclude_hair) if resolved_isolated else False,
        full_ao_miss_pixels=int(full_ao_miss_pixels) if resolved_isolated else 0,
        head_ao_miss_pixels=int(head_ao_miss_pixels) if resolved_isolated else 0,
        effective_padding_px=int(effective_pad),
        material_slots=exported_mat_slots,
        per_part_stats=per_part_stats,
    )
    res_obj.write_json(meta_json_path)
    return res_obj


def build_bake_4k_blender_cmd(
    *,
    blender_bin: str | Path,
    high_mesh_path: str | Path,
    low_uv_mesh_path: str | Path,
    out_normal_png: str | Path,
    out_ao_png: str | Path,
    out_curvature_png: str | Path,
    out_baked_glb: str | Path,
    meta_json_path: str | Path,
    out_contact_sheet_png: str | Path | None = None,
    uv_preview_png: str | Path | None = None,
    tex_size: int = 4096,
    padding_px: int = 16,
    normal_samples: int = 1,
    ao_samples: int = 24,
    ao_distance_rel: float = 0.12,
    curvature_samples: int = 1,
    use_gpu: bool = False,
    ray_distance: float | None = None,
    cage_extrusion: float | None = None,
    isolated_part_bake: bool | None = None,
    high_labels_npz: str | Path | None = None,
    low_labels_npz: str | Path | None = None,
    ring_dilation: int = 3,
    head_ring_dilation: int = 16,
    head_ao_exclude_hair: bool = True,
    preserve_material_slots: bool = True,
) -> list[str]:
    """Construct headless Blender CLI command for 4K Normal + AO + Curvature baking."""
    resolved_isolated = (high_labels_npz is not None) if isolated_part_bake is None else bool(isolated_part_bake)
    cmd = [
        str(Path(blender_bin)),
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "bake-blender",
        "--high",
        str(Path(high_mesh_path)),
        "--low",
        str(Path(low_uv_mesh_path)),
        "--out-normal",
        str(Path(out_normal_png)),
        "--out-ao",
        str(Path(out_ao_png)),
        "--out-curvature",
        str(Path(out_curvature_png)),
        "--out-glb",
        str(Path(out_baked_glb)),
        "--meta-json",
        str(Path(meta_json_path)),
        "--tex-size",
        str(int(tex_size)),
        "--padding-px",
        str(int(padding_px)),
        "--normal-samples",
        str(int(normal_samples)),
        "--ao-samples",
        str(int(ao_samples)),
        "--ao-distance-rel",
        str(float(ao_distance_rel)),
        "--curvature-samples",
        str(int(curvature_samples)),
    ]
    if out_contact_sheet_png is not None:
        cmd.extend(["--out-sheet", str(Path(out_contact_sheet_png))])
    if uv_preview_png is not None:
        cmd.extend(["--uv-preview-png", str(Path(uv_preview_png))])
    if use_gpu:
        cmd.append("--use-gpu")
    if ray_distance is not None:
        cmd.extend(["--ray-distance", str(float(ray_distance))])
    if cage_extrusion is not None:
        cmd.extend(["--cage-extrusion", str(float(cage_extrusion))])
    if resolved_isolated:
        cmd.append("--isolated-part-bake")
        cmd.extend(["--ring-dilation", str(int(ring_dilation))])
        cmd.extend(["--head-ring-dilation", str(int(head_ring_dilation))])
        cmd.append("--head-ao-exclude-hair" if head_ao_exclude_hair else "--no-head-ao-exclude-hair")
    elif isolated_part_bake is False:
        cmd.append("--no-isolated-part-bake")
    if high_labels_npz is not None:
        cmd.extend(["--high-labels-npz", str(Path(high_labels_npz))])
    if low_labels_npz is not None:
        cmd.extend(["--low-labels-npz", str(Path(low_labels_npz))])
    cmd.append("--preserve-material-slots" if preserve_material_slots else "--no-preserve-material-slots")
    return cmd


def bake_4k_maps_blender(
    high_mesh_path: str | Path,
    low_uv_mesh_path: str | Path,
    out_normal_png: str | Path,
    out_ao_png: str | Path,
    out_curvature_png: str | Path,
    out_baked_glb: str | Path,
    *,
    out_contact_sheet_png: str | Path | None = None,
    uv_preview_png: str | Path | None = None,
    tex_size: int = 4096,
    padding_px: int = 16,
    normal_samples: int = 1,
    ao_samples: int = 24,
    ao_distance_rel: float = 0.12,
    curvature_samples: int = 1,
    use_gpu: bool = False,
    ray_distance: float | None = None,
    cage_extrusion: float | None = None,
    blender_bin: str | Path = DEFAULT_BLENDER_BIN,
    timeout_s: float = 2400.0,
    bake_json_path: str | Path | None = None,
    isolated_part_bake: bool | None = None,
    high_labels_npz: str | Path | None = None,
    low_labels_npz: str | Path | None = None,
    ring_dilation: int = 3,
    head_ring_dilation: int = 16,
    head_ao_exclude_hair: bool = True,
    preserve_material_slots: bool = True,
) -> MultiMapBakeResult:
    """Bake 4K Normal + AO + Curvature maps in an isolated headless Blender Cycles subprocess."""
    hp = Path(high_mesh_path)
    lp = Path(low_uv_mesh_path)
    n_p = Path(out_normal_png)
    a_p = Path(out_ao_png)
    c_p = Path(out_curvature_png)
    g_p = Path(out_baked_glb)
    blender_p = Path(blender_bin)

    if not hp.exists():
        raise FileNotFoundError(f"Highpoly mesh not found: {hp}")
    if not lp.exists():
        raise FileNotFoundError(f"Lowpoly UV mesh not found: {lp}")
    if not blender_p.exists():
        raise FileNotFoundError(f"Blender binary not found: {blender_p}")

    n_p.parent.mkdir(parents=True, exist_ok=True)
    cleanup_meta = bake_json_path is None
    meta_path = (
        Path(bake_json_path)
        if bake_json_path is not None
        else (n_p.parent / f"._{n_p.stem}_bake4k_meta_{os.getpid()}.json")
    )
    work_tmp = n_p.parent / f"._{n_p.stem}_bake4k_tmp_{os.getpid()}"
    work_tmp.mkdir(parents=True, exist_ok=True)

    cmd = build_bake_4k_blender_cmd(
        blender_bin=blender_p,
        high_mesh_path=hp,
        low_uv_mesh_path=lp,
        out_normal_png=n_p,
        out_ao_png=a_p,
        out_curvature_png=c_p,
        out_baked_glb=g_p,
        meta_json_path=meta_path,
        out_contact_sheet_png=out_contact_sheet_png,
        uv_preview_png=uv_preview_png,
        tex_size=int(tex_size),
        padding_px=int(padding_px),
        normal_samples=int(normal_samples),
        ao_samples=int(ao_samples),
        ao_distance_rel=float(ao_distance_rel),
        curvature_samples=int(curvature_samples),
        use_gpu=bool(use_gpu),
        ray_distance=ray_distance,
        cage_extrusion=cage_extrusion,
        isolated_part_bake=isolated_part_bake,
        high_labels_npz=high_labels_npz,
        low_labels_npz=low_labels_npz,
        ring_dilation=int(ring_dilation),
        head_ring_dilation=int(head_ring_dilation),
        head_ao_exclude_hair=bool(head_ao_exclude_hair),
        preserve_material_slots=bool(preserve_material_slots),
    )

    env = os.environ.copy()
    env["TMPDIR"] = str(work_tmp)
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            env=env,
        )
        if (
            proc.returncode != 0
            or not n_p.exists()
            or not a_p.exists()
            or not c_p.exists()
            or not g_p.exists()
            or not meta_path.exists()
        ):
            raise RuntimeError(
                f"Blender 4K multi-map bake failed (code={proc.returncode}):\n"
                f"CMD: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
            )
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        return MultiMapBakeResult(**payload)
    finally:
        if cleanup_meta and meta_path.exists():
            meta_path.unlink(missing_ok=True)
        shutil.rmtree(work_tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Headless Blender 6-View HD Studio Renderer (True Quad Wireframe + Normal+AO)
# ---------------------------------------------------------------------------


def _run_render_track3_in_blender(
    *,
    mesh_path: Path,
    quad_mesh_path: Path | None,
    out_dir: Path,
    prefix: str,
    states: Sequence[str],
    normal_map_path: Path | None,
    ao_map_path: Path | None,
    framing_mode: str,
    samples: int,
    use_gpu: bool,
    meta_json_path: Path,
    ao_strength: float = 0.45,
) -> Track3RenderResult:
    import bpy
    import mathutils
    import numpy as np

    from quadvox.s2_retopo.render import (
        _draw_banner_label,
        _read_png_rgb_in_blender,
        _resize_rgb_bilinear,
        _write_png_rgb_in_blender,
    )

    t0 = time.perf_counter()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    device_used = _blender_configure_cycles(scene, use_gpu=use_gpu, samples=samples, denoise=True)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.15

    world = bpy.data.worlds.new("Track3StudioWorld")
    scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node is not None:
        bg_node.inputs["Color"].default_value = (0.16, 0.17, 0.19, 1.0)
        bg_node.inputs["Strength"].default_value = 0.65

    def _add_area_light(
        name: str,
        location: tuple[float, float, float],
        target: tuple[float, float, float],
        energy: float,
        size: float,
        color: tuple[float, float, float],
    ) -> None:
        ldata = bpy.data.lights.new(name, type="AREA")
        ldata.energy = energy
        ldata.size = size
        ldata.color = color
        lobj = bpy.data.objects.new(name, ldata)
        scene.collection.objects.link(lobj)
        lobj.location = location
        direction = mathutils.Vector(target) - mathutils.Vector(location)
        lobj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    _add_area_light("KeyLight", (1.85, -2.35, 2.25), (0.0, 0.0, 0.55), 145.0, 1.5, (1.0, 0.98, 0.95))
    _add_area_light("FillLight", (-2.15, -1.75, 1.05), (0.0, 0.0, 0.50), 48.0, 2.3, (0.90, 0.94, 1.0))
    _add_area_light("RimLight", (-0.55, 2.35, 1.90), (0.0, 0.0, 0.65), 110.0, 1.2, (1.0, 1.0, 1.0))
    _add_area_light("LowBounce", (0.0, -0.85, -0.40), (0.0, 0.0, 0.44), 22.0, 1.8, (0.95, 0.96, 1.0))

    cam_data = bpy.data.cameras.new("Track3Cam")
    cam_obj = bpy.data.objects.new("Track3Cam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    def _normalize_imported_group(meshes: Sequence[Any], mode: str) -> tuple[Any, float]:
        root = bpy.data.objects.new("NormRoot", None)
        scene.collection.objects.link(root)
        for obj in meshes:
            if obj.parent is None:
                obj.parent = root
        bpy.context.view_layer.update()

        min_x = min_y = min_z = float("inf")
        max_x = max_y = max_z = float("-inf")
        for mobj in meshes:
            mw = mobj.matrix_world
            for corner in mobj.bound_box:
                p = mw @ mathutils.Vector(corner)
                min_x, min_y, min_z = min(min_x, p.x), min(min_y, p.y), min(min_z, p.z)
                max_x, max_y, max_z = max(max_x, p.x), max(max_y, p.y), max(max_z, p.z)

        sx, sy, sz = max_x - min_x, max_y - min_y, max_z - min_z
        cx, cy, cz = 0.5 * (min_x + max_x), 0.5 * (min_y + max_y), 0.5 * (min_z + max_z)
        if mode == "object":
            extent = max(sx, sy, sz, 1e-6)
            scale = 0.85 / extent
            root.scale = (scale, scale, scale)
            root.location = (-cx * scale, -cy * scale, 0.50 - cz * scale)
        else:
            height = max(sz, 1e-6)
            scale = 1.0 / height
            root.scale = (scale, scale, scale)
            root.location = (-cx * scale, -cy * scale, -min_z * scale)
        bpy.context.view_layer.update()
        diag = math.sqrt(sx * sx + sy * sy + sz * sz) * scale
        return root, diag

    def _aim_camera(spec: dict[str, Any]) -> None:
        cam_obj.location = spec["location"]
        direction = mathutils.Vector(spec["target"]) - mathutils.Vector(spec["location"])
        cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        if spec["ortho"]:
            cam_data.type = "ORTHO"
            cam_data.ortho_scale = float(spec["ortho_scale"])
        else:
            cam_data.type = "PERSP"
            cam_data.lens = float(spec["lens"])
        cam_data.clip_start = 0.01
        cam_data.clip_end = 50.0
        scene.render.resolution_x = int(spec["width"])
        scene.render.resolution_y = int(spec["height"])
        scene.render.resolution_percentage = 100

    def _make_mat(name: str) -> tuple[Any, Any, Any]:
        mat = bpy.data.materials.new(name)
        if mat.node_tree is None:
            mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes.get("Principled BSDF") or nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Metallic"].default_value = 0.0
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.35
        return mat, nt, bsdf

    view_specs = OBJECT_VIEW_SPECS if framing_mode == "object" else CHARACTER_VIEW_SPECS
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, str]] = {}

    for state in states:
        if state not in VALID_TRACK3_STATES:
            raise ValueError(f"Unsupported Track 3 render state '{state}'")

        # For wireframe_clay, prefer native quad .obj so TRUE QUAD LOOPS render without diagonals!
        use_quad_src = (
            state == "wireframe_clay"
            and quad_mesh_path is not None
            and quad_mesh_path.exists()
        )
        active_src = quad_mesh_path if use_quad_src else mesh_path
        meshes = _blender_import_meshes(active_src)
        root_empty, norm_diag = _normalize_imported_group(meshes, framing_mode)

        if state == "wireframe_clay":
            clay_mat, _, clay_bsdf = _make_mat("T3_WireClay")
            clay_bsdf.inputs["Base Color"].default_value = (0.64, 0.64, 0.65, 1.0)
            clay_bsdf.inputs["Roughness"].default_value = 0.68
            line_mat, _, line_bsdf = _make_mat("T3_WireLines")
            line_bsdf.inputs["Base Color"].default_value = (0.04, 0.05, 0.07, 1.0)
            line_bsdf.inputs["Roughness"].default_value = 0.85

            for mobj in meshes:
                mobj.data.polygons.foreach_set("use_smooth", [False] * len(mobj.data.polygons))
                mobj.data.materials.clear()
                mobj.data.materials.append(clay_mat)
                mobj.data.materials.append(line_mat)
                # Compute adaptive wireframe thickness in local units matching object scale.
                local_scale = max(float(abs(root_empty.scale.x)), 1e-6)
                poly_n = max(1, len(mobj.data.polygons))
                # Thinner wires on denser meshes (35k), slightly bolder on coarse 2k accessories.
                world_thick = float(np.clip(0.00055 * (35000.0 / max(poly_n, 500)) ** 0.28, 0.00028, 0.0014))
                wf_mod = mobj.modifiers.new("TruePolyWire", "WIREFRAME")
                wf_mod.use_replace = False
                wf_mod.material_offset = 1
                wf_mod.use_even_offset = False
                wf_mod.use_relative_offset = False
                wf_mod.use_boundary = True
                wf_mod.thickness = world_thick / local_scale
                mobj.data.update()

        elif state in ("smooth_lowpoly", "highpoly_clay"):
            mat, _, bsdf = _make_mat(f"T3_{state}")
            bsdf.inputs["Base Color"].default_value = (0.58, 0.58, 0.59, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.56 if state == "smooth_lowpoly" else 0.64
            for mobj in meshes:
                mobj.data.polygons.foreach_set("use_smooth", [True] * len(mobj.data.polygons))
                mobj.data.materials.clear()
                mobj.data.materials.append(mat)
                mobj.data.update()

        elif state in ("normal_mapped", "normal_ao_composite"):
            if normal_map_path is None or not normal_map_path.exists():
                raise FileNotFoundError(f"State '{state}' requires valid --normal-map: {normal_map_path}")
            mat, nt, bsdf = _make_mat(f"T3_{state}")
            bsdf.inputs["Roughness"].default_value = 0.60
            nrm_img = bpy.data.images.load(str(normal_map_path), check_existing=False)
            nrm_img.colorspace_settings.name = "Non-Color"
            ntex = nt.nodes.new("ShaderNodeTexImage")
            ntex.image = nrm_img
            ntex.interpolation = "Linear"
            ntex.extension = "EXTEND"
            nrm_node = nt.nodes.new("ShaderNodeNormalMap")
            nrm_node.space = "TANGENT"
            nrm_node.inputs["Strength"].default_value = 1.0
            nt.links.new(ntex.outputs["Color"], nrm_node.inputs["Color"])
            nt.links.new(nrm_node.outputs["Normal"], bsdf.inputs["Normal"])

            if state == "normal_ao_composite":
                if ao_map_path is None or not ao_map_path.exists():
                    raise FileNotFoundError(f"State 'normal_ao_composite' requires valid --ao-map: {ao_map_path}")
                ao_img = bpy.data.images.load(str(ao_map_path), check_existing=False)
                ao_img.colorspace_settings.name = "Non-Color"
                atex = nt.nodes.new("ShaderNodeTexImage")
                atex.image = ao_img
                atex.interpolation = "Linear"
                atex.extension = "EXTEND"
                mix = nt.nodes.new("ShaderNodeMix")
                mix.data_type = "RGBA"
                mix.blend_type = "MULTIPLY"
                mix.inputs[0].default_value = float(ao_strength)
                mix.inputs[6].default_value = (0.60, 0.60, 0.61, 1.0)
                nt.links.new(atex.outputs["Color"], mix.inputs[7])
                nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
            else:
                bsdf.inputs["Base Color"].default_value = (0.58, 0.58, 0.59, 1.0)

            for mobj in meshes:
                mobj.data.polygons.foreach_set("use_smooth", [True] * len(mobj.data.polygons))
                slot_count = max(1, len(mobj.data.materials))
                mobj.data.materials.clear()
                for _ in range(slot_count):
                    mobj.data.materials.append(mat)
                mobj.data.update()

        state_paths: dict[str, str] = {}
        panels: list[np.ndarray] = []
        panel_w, panel_h = 768, 900

        for spec in view_specs:
            _aim_camera(spec)
            vk = spec["key"]
            img_path = out_dir / f"{prefix}_{state}_{vk}.png"
            scene.render.filepath = str(img_path)
            bpy.ops.render.render(write_still=True)
            state_paths[vk] = str(img_path)

            arr = _read_png_rgb_in_blender(img_path)
            if arr.shape[0] != panel_h or arr.shape[1] != panel_w:
                panel = np.full((panel_h, panel_w, 3), (18, 20, 24), dtype=np.uint8)
                fit = min(panel_w / arr.shape[1], (panel_h - 60) / arr.shape[0])
                rw = max(1, int(round(arr.shape[1] * fit)))
                rh = max(1, int(round(arr.shape[0] * fit)))
                resized = _resize_rgb_bilinear(arr, rw, rh)
                y_off = 52 + max(0, (panel_h - 52 - rh) // 2)
                x_off = max(0, (panel_w - rw) // 2)
                panel[y_off : y_off + rh, x_off : x_off + rw] = resized
            else:
                panel = arr.copy()
            _draw_banner_label(panel, spec["label"], scale=3)
            panels.append(panel)

        pad = 6
        sheet = np.full((2 * panel_h + pad, 3 * panel_w + 2 * pad, 3), (18, 20, 24), dtype=np.uint8)
        for idx, pan in enumerate(panels):
            r, c = divmod(idx, 3)
            y0 = r * (panel_h + pad)
            x0 = c * (panel_w + pad)
            sheet[y0 : y0 + panel_h, x0 : x0 + panel_w] = pan

        sheet_path = out_dir / f"{prefix}_{state}_sheet.png"
        _write_png_rgb_in_blender(sheet_path, sheet)
        state_paths["sheet"] = str(sheet_path)
        outputs[state] = state_paths

        # Cleanly remove state meshes before next state so memory stays minimal.
        for obj in list(meshes) + [root_empty]:
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)

    res = Track3RenderResult(
        mesh_path=str(mesh_path),
        quad_mesh_path=str(quad_mesh_path) if quad_mesh_path is not None else None,
        out_dir=str(out_dir),
        prefix=str(prefix),
        states=list(states),
        framing_mode=str(framing_mode),
        samples=max(1, int(samples)),
        use_gpu=bool(use_gpu),
        device_used=device_used,
        blender_version=".".join(str(x) for x in bpy.app.version),
        elapsed_s=time.perf_counter() - t0,
        outputs=outputs,
        ao_strength=float(ao_strength),
    )
    res.write_json(meta_json_path)
    return res


def build_render_track3_blender_cmd(
    *,
    blender_bin: str | Path,
    mesh_path: str | Path,
    out_dir: str | Path,
    prefix: str,
    states: Sequence[str],
    meta_json_path: str | Path,
    quad_mesh_path: str | Path | None = None,
    normal_map_path: str | Path | None = None,
    ao_map_path: str | Path | None = None,
    framing_mode: str = "character",
    samples: int = 32,
    use_gpu: bool = False,
    ao_strength: float = 0.45,
) -> list[str]:
    """Construct headless Blender CLI command for Track 3 4-state 6-view HD rendering."""
    cmd = [
        str(Path(blender_bin)),
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "render-blender",
        "--mesh",
        str(Path(mesh_path)),
        "--out-dir",
        str(Path(out_dir)),
        "--prefix",
        str(prefix),
        "--states",
        *[str(s) for s in states],
        "--framing-mode",
        str(framing_mode),
        "--samples",
        str(int(samples)),
        "--meta-json",
        str(Path(meta_json_path)),
    ]
    if quad_mesh_path is not None:
        cmd.extend(["--quad-mesh", str(Path(quad_mesh_path))])
    if normal_map_path is not None:
        cmd.extend(["--normal-map", str(Path(normal_map_path))])
    if ao_map_path is not None:
        cmd.extend(["--ao-map", str(Path(ao_map_path))])
    if use_gpu:
        cmd.append("--use-gpu")
    cmd.extend(["--ao-strength", str(float(ao_strength))])
    return cmd


def render_track3_views_blender(
    mesh_path: str | Path,
    out_dir: str | Path,
    prefix: str,
    *,
    states: Sequence[str] = ("wireframe_clay", "smooth_lowpoly", "normal_mapped", "normal_ao_composite"),
    quad_mesh_path: str | Path | None = None,
    normal_map_path: str | Path | None = None,
    ao_map_path: str | Path | None = None,
    framing_mode: str = "character",
    samples: int = 32,
    use_gpu: bool = False,
    blender_bin: str | Path = DEFAULT_BLENDER_BIN,
    timeout_s: float = 2400.0,
    report_json_path: str | Path | None = None,
    ao_strength: float = 0.45,
) -> Track3RenderResult:
    """Render 6-view HD PNGs + 3x2 labeled sheets in an isolated headless Blender Cycles subprocess."""
    m_p = Path(mesh_path)
    o_d = Path(out_dir)
    blender_p = Path(blender_bin)

    if not m_p.exists():
        raise FileNotFoundError(f"Mesh not found: {m_p}")
    if not blender_p.exists():
        raise FileNotFoundError(f"Blender binary not found: {blender_p}")

    o_d.mkdir(parents=True, exist_ok=True)
    cleanup_meta = report_json_path is None
    meta_path = (
        Path(report_json_path)
        if report_json_path is not None
        else (o_d / f"._{prefix}_render_t3_meta_{os.getpid()}.json")
    )
    work_tmp = o_d / f"._{prefix}_render_t3_tmp_{os.getpid()}"
    work_tmp.mkdir(parents=True, exist_ok=True)

    cmd = build_render_track3_blender_cmd(
        blender_bin=blender_p,
        mesh_path=m_p,
        out_dir=o_d,
        prefix=prefix,
        states=states,
        meta_json_path=meta_path,
        quad_mesh_path=quad_mesh_path,
        normal_map_path=normal_map_path,
        ao_map_path=ao_map_path,
        framing_mode=framing_mode,
        samples=int(samples),
        use_gpu=bool(use_gpu),
        ao_strength=float(ao_strength),
    )

    env = os.environ.copy()
    env["TMPDIR"] = str(work_tmp)
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            env=env,
        )
        if proc.returncode != 0 or not meta_path.exists():
            raise RuntimeError(
                f"Blender Track 3 render failed (code={proc.returncode}):\n"
                f"CMD: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
            )
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        return Track3RenderResult(**payload)
    finally:
        if cleanup_meta and meta_path.exists():
            meta_path.unlink(missing_ok=True)
        shutil.rmtree(work_tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track 3 Part-Aware UV, 4K Multi-Map Baking & HD Rendering.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_unw = sub.add_parser("unwrap", help="Part-aware UV unwrap with explicit face texel boost.")
    p_unw.add_argument("--mesh", required=True)
    p_unw.add_argument("--out", required=True)
    p_unw.add_argument("--labels-npz", default=None)
    p_unw.add_argument("--labels-source-mesh", default=None)
    p_unw.add_argument("--tex-size", type=int, default=4096)
    p_unw.add_argument("--padding-px", type=int, default=16)
    p_unw.add_argument("--face-scale-boost", type=float, default=2.25)
    p_unw.add_argument("--uv-preview-png", default=None)
    p_unw.add_argument("--report-json", default=None)
    p_unw.add_argument("--uv-method", default="auto", choices=["auto", "blender_slim", "xatlas"])
    p_unw.add_argument("--asset-type", default="auto", choices=["auto", "character", "prop"])
    p_unw.add_argument("--blender-bin", default=str(DEFAULT_BLENDER_BIN))
    p_unw.add_argument("--max-charts", type=int, default=None)
    p_unw.add_argument("--zero-overlap-refine", action=argparse.BooleanOptionalAction, default=True)
    p_unw.add_argument("--pack-method", default="blender_concave", choices=["shelf", "blender_concave"])
    p_unw.add_argument("--assign-material-slots", action=argparse.BooleanOptionalAction, default=True)
    p_unw.add_argument("--min-island-px", type=int, default=16)
    p_unw.add_argument("--min-true-gap-px", type=float, default=6.0)
    p_unw.add_argument("--out-labels-npz", default=None)

    for cmd_name in ("bake", "bake-blender"):
        p_b = sub.add_parser(cmd_name, help="Bake 4K Normal + AO + Curvature maps in Cycles.")
        p_b.add_argument("--high", required=True)
        p_b.add_argument("--low", required=True)
        p_b.add_argument("--out-normal", required=True)
        p_b.add_argument("--out-ao", required=True)
        p_b.add_argument("--out-curvature", required=True)
        p_b.add_argument("--out-glb", required=True)
        p_b.add_argument("--out-sheet", default=None)
        p_b.add_argument("--uv-preview-png", default=None)
        p_b.add_argument("--tex-size", type=int, default=4096)
        p_b.add_argument("--padding-px", type=int, default=16)
        p_b.add_argument("--normal-samples", type=int, default=1)
        p_b.add_argument("--ao-samples", type=int, default=24)
        p_b.add_argument("--ao-distance-rel", type=float, default=0.12)
        p_b.add_argument("--curvature-samples", type=int, default=1)
        p_b.add_argument("--use-gpu", action="store_true")
        p_b.add_argument("--ray-distance", type=float, default=None)
        p_b.add_argument("--cage-extrusion", type=float, default=None)
        p_b.add_argument("--blender-bin", default=str(DEFAULT_BLENDER_BIN))
        p_b.add_argument("--meta-json", default=None)
        p_b.add_argument("--report-json", default=None)
        p_b.add_argument("--isolated-part-bake", action=argparse.BooleanOptionalAction, default=None)
        p_b.add_argument("--high-labels-npz", default=None)
        p_b.add_argument("--low-labels-npz", default=None)
        p_b.add_argument("--ring-dilation", type=int, default=3)
        p_b.add_argument("--head-ring-dilation", type=int, default=16)
        p_b.add_argument("--head-ao-exclude-hair", action=argparse.BooleanOptionalAction, default=True)
        p_b.add_argument("--preserve-material-slots", action=argparse.BooleanOptionalAction, default=True)

    for cmd_name in ("render", "render-blender"):
        p_r = sub.add_parser(cmd_name, help="Render 4-state 6-view HD studio PNGs.")
        p_r.add_argument("--mesh", required=True)
        p_r.add_argument("--quad-mesh", default=None)
        p_r.add_argument("--out-dir", required=True)
        p_r.add_argument("--prefix", required=True)
        p_r.add_argument(
            "--states",
            nargs="+",
            default=["wireframe_clay", "smooth_lowpoly", "normal_mapped", "normal_ao_composite"],
            choices=list(VALID_TRACK3_STATES),
        )
        p_r.add_argument("--normal-map", default=None)
        p_r.add_argument("--ao-map", default=None)
        p_r.add_argument("--framing-mode", default="character", choices=["character", "object"])
        p_r.add_argument("--samples", type=int, default=32)
        p_r.add_argument("--use-gpu", action="store_true")
        p_r.add_argument("--blender-bin", default=str(DEFAULT_BLENDER_BIN))
        p_r.add_argument("--meta-json", default=None)
        p_r.add_argument("--report-json", default=None)
        p_r.add_argument("--ao-strength", type=float, default=0.45)

    return parser


def _parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if argv is None:
        raw = sys.argv[1:]
        if "--" in raw:
            raw = raw[raw.index("--") + 1 :]
    else:
        raw = list(argv)
    return _build_arg_parser().parse_args(raw)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli_args(argv)
    if args.command == "unwrap":
        res = unwrap_part_aware(
            mesh_path=args.mesh,
            out_mesh_path=args.out,
            labels_npz_path=args.labels_npz,
            labels_source_mesh_path=args.labels_source_mesh,
            tex_size=int(args.tex_size),
            padding_px=int(args.padding_px),
            face_scale_boost=float(args.face_scale_boost),
            uv_preview_png=args.uv_preview_png,
            unwrap_json_path=args.report_json,
            uv_method=args.uv_method,
            asset_type=args.asset_type,
            blender_bin=args.blender_bin,
            max_charts=args.max_charts,
            zero_overlap_refine=bool(args.zero_overlap_refine),
            pack_method=str(args.pack_method),
            assign_material_slots=bool(args.assign_material_slots),
            min_island_px=int(args.min_island_px),
            min_true_gap_px=float(args.min_true_gap_px),
            out_labels_npz=args.out_labels_npz,
        )
        print(res.to_json(indent=2))
        return 0

    if args.command == "bake-blender":
        meta_p = Path(args.meta_json or args.report_json)
        _run_bake_4k_in_blender(
            high_mesh_path=Path(args.high),
            low_uv_mesh_path=Path(args.low),
            out_normal_png=Path(args.out_normal),
            out_ao_png=Path(args.out_ao),
            out_curvature_png=Path(args.out_curvature),
            out_baked_glb=Path(args.out_glb),
            out_contact_sheet_png=Path(args.out_sheet) if args.out_sheet else None,
            uv_preview_png=Path(args.uv_preview_png) if args.uv_preview_png else None,
            tex_size=int(args.tex_size),
            padding_px=int(args.padding_px),
            normal_samples=int(args.normal_samples),
            ao_samples=int(args.ao_samples),
            ao_distance_rel=float(args.ao_distance_rel),
            curvature_samples=int(args.curvature_samples),
            use_gpu=bool(args.use_gpu),
            ray_distance=args.ray_distance,
            cage_extrusion=args.cage_extrusion,
            meta_json_path=meta_p,
            isolated_part_bake=args.isolated_part_bake,
            high_labels_npz=Path(args.high_labels_npz) if args.high_labels_npz else None,
            low_labels_npz=Path(args.low_labels_npz) if args.low_labels_npz else None,
            ring_dilation=int(args.ring_dilation),
            head_ring_dilation=int(args.head_ring_dilation),
            head_ao_exclude_hair=bool(args.head_ao_exclude_hair),
            preserve_material_slots=bool(args.preserve_material_slots),
        )
        return 0

    if args.command == "bake":
        res = bake_4k_maps_blender(
            high_mesh_path=args.high,
            low_uv_mesh_path=args.low,
            out_normal_png=args.out_normal,
            out_ao_png=args.out_ao,
            out_curvature_png=args.out_curvature,
            out_baked_glb=args.out_glb,
            out_contact_sheet_png=args.out_sheet,
            uv_preview_png=args.uv_preview_png,
            tex_size=int(args.tex_size),
            padding_px=int(args.padding_px),
            normal_samples=int(args.normal_samples),
            ao_samples=int(args.ao_samples),
            ao_distance_rel=float(args.ao_distance_rel),
            curvature_samples=int(args.curvature_samples),
            use_gpu=bool(args.use_gpu),
            ray_distance=args.ray_distance,
            cage_extrusion=args.cage_extrusion,
            blender_bin=args.blender_bin,
            bake_json_path=args.report_json or args.meta_json,
            isolated_part_bake=args.isolated_part_bake,
            high_labels_npz=args.high_labels_npz,
            low_labels_npz=args.low_labels_npz,
            ring_dilation=int(args.ring_dilation),
            head_ring_dilation=int(args.head_ring_dilation),
            head_ao_exclude_hair=bool(args.head_ao_exclude_hair),
            preserve_material_slots=bool(args.preserve_material_slots),
        )
        print(res.to_json(indent=2))
        return 0

    if args.command == "render-blender":
        meta_p = Path(args.meta_json or args.report_json)
        _run_render_track3_in_blender(
            mesh_path=Path(args.mesh),
            quad_mesh_path=Path(args.quad_mesh) if args.quad_mesh else None,
            out_dir=Path(args.out_dir),
            prefix=str(args.prefix),
            states=list(args.states),
            normal_map_path=Path(args.normal_map) if args.normal_map else None,
            ao_map_path=Path(args.ao_map) if args.ao_map else None,
            framing_mode=str(args.framing_mode),
            samples=int(args.samples),
            use_gpu=bool(args.use_gpu),
            meta_json_path=meta_p,
            ao_strength=float(args.ao_strength),
        )
        return 0

    if args.command == "render":
        res = render_track3_views_blender(
            mesh_path=args.mesh,
            quad_mesh_path=args.quad_mesh,
            out_dir=args.out_dir,
            prefix=args.prefix,
            states=args.states,
            normal_map_path=args.normal_map,
            ao_map_path=args.ao_map,
            framing_mode=args.framing_mode,
            samples=int(args.samples),
            use_gpu=bool(args.use_gpu),
            blender_bin=args.blender_bin,
            report_json_path=args.report_json or args.meta_json,
            ao_strength=float(args.ao_strength),
        )
        print(res.to_json(indent=2))
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
