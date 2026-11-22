"""Quad retopology runners for Track 3: QuadriFlow, ReQuad v0.11.3 (semantic-density), and QuadWild Bi-MDF.

Top-level imports are strictly Python stdlib so this module executes cleanly both
inside `.venv` Python and directly inside headless Blender 5.3 via:
    blender --background --factory-startup --python-exit-code 1 --python quadvox/s2_retopo/quad_retopo.py -- <cmd> ...

All intermediate work directories and Blender extension staging folders live inside
the job output directory (ZERO hardcoded system temp paths).
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
DEFAULT_REQUAD_DIR = Path(os.environ.get("QUADVOX_REQUAD_DIR") or (Path(_REPO_ROOT) / "tools" / "requad"))
DEFAULT_QUADWILD_DIR = Path(os.environ.get("QUADVOX_QUADWILD_DIR") or (Path(_REPO_ROOT) / "tools" / "quadwild"))

CANONICAL_CLASS_NAMES: dict[int, str] = {
    0: "unknown",
    1: "head_face",
    2: "hair",
    3: "rigid_accessory",
    4: "upper_garment",
    5: "pleated_skirt",
    6: "limbs_hands",
    7: "socks_shoes",
}

# Default semantic quad budgets summing to 35,000 quads.
# Bald body: head_face ~8k, upper_garment ~10k, pleated_skirt ~5k, limbs_hands + socks_shoes ~12k.
DEFAULT_BALD_PART_TARGETS: dict[int, int] = {
    1: 8000,   # head_face
    4: 10000,  # upper_garment
    5: 5000,   # pleated_skirt
    6: 7500,   # limbs_hands
    7: 4500,   # socks_shoes
}

# Full a1 character: proportional ~35k with hair and rigid accessories budgeted cleanly.
DEFAULT_FULL_PART_TARGETS: dict[int, int] = {
    1: 7500,   # head_face
    2: 6500,   # hair
    3: 1000,   # rigid_accessory
    4: 8000,   # upper_garment
    5: 4000,   # pleated_skirt
    6: 5000,   # limbs_hands
    7: 3000,   # socks_shoes
}

QUADWILD_PREP_TEMPLATE = """\
do_remesh {do_remesh}
sharp_feature_thr {sharp_angle}
alpha 0.01
scaleFact 1
"""

QUADWILD_MAIN_TEMPLATE = """\
alpha {alpha}
ilpMethod 1
timeLimit 200
gapLimit 0.0
callbackTimeLimit 8 3.00 5.000 10.0 20.0 30.0 60.0 90.0 120.0
callbackGapLimit 8 0.005 0.02 0.05 0.10 0.15 0.20 0.25 0.3
minimumGap 0.4
isometry 1
regularityQuadrilaterals 1
regularityNonQuadrilaterals 1
regularityNonQuadrilateralsWeight 0.9
alignSingularities {align}
alignSingularitiesWeight 0.1
repeatLosingConstraintsIterations 1
repeatLosingConstraintsQuads 0
repeatLosingConstraintsNonQuads 0
repeatLosingConstraintsAlign 0
hardParityConstraint 1
scaleFact {scale}
fixedChartClusters 0
useFlowSolver 1
flow_config_filename "config/main_config/flow_virtual_simple.json"
satsuma_config_filename "config/satsuma/lemon.json"
"""


def get_canonical_class_names() -> dict[int, str]:
    """Return canonical 7-class mapping, preferring `topo_exp.seg_contract` if present."""
    try:
        from topo_exp import seg_contract  # type: ignore[import-not-found]

        for attr in ("CLASS_NAMES", "CANONICAL_CLASSES", "ID_TO_NAME", "SEMANTIC_CLASSES"):
            mapping = getattr(seg_contract, attr, None)
            if isinstance(mapping, Mapping) and len(mapping) >= 7:
                return {int(k): str(v) for k, v in mapping.items()}
            if isinstance(mapping, (list, tuple)) and len(mapping) >= 7:
                return {idx: str(name) for idx, name in enumerate(mapping)}
    except Exception:
        pass
    return dict(CANONICAL_CLASS_NAMES)


@dataclass
class QuadRetopoResult:
    """Comprehensive topology, symmetry, fidelity, and per-part quad allocation result."""

    engine: str
    input_mesh_path: str
    highpoly_mesh_path: str
    out_quad_obj: str
    out_tri_glb: str
    target_quads: int
    vertex_count: int = 0
    poly_count: int = 0
    quad_count: int = 0
    tri_count: int = 0
    ngon_count: int = 0
    equiv_tri_count: int = 0
    quad_ratio: float = 0.0
    open_boundary_edges: int = 0
    non_manifold_edges: int = 0
    non_manifold_vertices: int = 0
    connected_components: int = 0
    is_watertight: bool = False
    valence_4_ratio: float = 0.0
    valence_3_ratio: float = 0.0
    valence_5_ratio: float = 0.0
    valence_ge8_ratio: float = 0.0
    x_symmetry_error_mean: float = 0.0
    x_symmetry_error_p95: float = 0.0
    x_symmetry_error_p99: float = 0.0
    chamfer_distance: float | None = None
    chamfer_high_to_low: float | None = None
    chamfer_low_to_high: float | None = None
    hausdorff_p95: float | None = None
    hausdorff_p99: float | None = None
    per_part_quad_counts: dict[str, int] = field(default_factory=dict)
    per_part_quad_shares: dict[str, float] = field(default_factory=dict)
    per_part_target_quads: dict[str, int] = field(default_factory=dict)
    density_diagnostics: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    blender_version: str = ""
    notes: str = ""

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
# Pure NumPy geometry, semantic density, boundary-guide, and quad-metric math
# ---------------------------------------------------------------------------


def parse_obj_polygons(obj_path: str | Path) -> tuple[Any, list[list[int]], Any, dict[str, int]]:
    """Parse raw vertices and variable-length polygons (quads/tris/ngons) from an OBJ file."""
    import numpy as np

    src = Path(obj_path)
    if not src.exists():
        raise FileNotFoundError(f"OBJ mesh not found: {src}")

    verts: list[tuple[float, float, float]] = []
    polygons: list[list[int]] = []
    tris: list[tuple[int, int, int]] = []
    quad_count = 0
    tri_count = 0
    ngon_count = 0
    equiv_tri_count = 0

    with src.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v ") or line.startswith("v\t"):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f ") or line.startswith("f\t"):
                tokens = line.split()[1:]
                idx_list: list[int] = []
                n_v = len(verts)
                for tok in tokens:
                    raw_idx = int(tok.split("/")[0])
                    idx_list.append(raw_idx - 1 if raw_idx > 0 else n_v + raw_idx)
                k = len(idx_list)
                if k < 3:
                    continue
                polygons.append(idx_list)
                if k == 3:
                    tri_count += 1
                    equiv_tri_count += 1
                    tris.append((idx_list[0], idx_list[1], idx_list[2]))
                elif k == 4:
                    quad_count += 1
                    equiv_tri_count += 2
                    tris.append((idx_list[0], idx_list[1], idx_list[2]))
                    tris.append((idx_list[0], idx_list[2], idx_list[3]))
                else:
                    ngon_count += 1
                    equiv_tri_count += k - 2
                    for j in range(1, k - 1):
                        tris.append((idx_list[0], idx_list[j], idx_list[j + 1]))

    v_arr = np.asarray(verts, dtype=np.float64) if verts else np.zeros((0, 3), dtype=np.float64)
    t_arr = np.asarray(tris, dtype=np.int64) if tris else np.zeros((0, 3), dtype=np.int64)
    counts = {
        "poly_count": len(polygons),
        "quad_count": quad_count,
        "tri_count": tri_count,
        "ngon_count": ngon_count,
        "equiv_tri_count": equiv_tri_count,
    }
    return v_arr, polygons, t_arr, counts


def load_semantic_npz(npz_path: str | Path) -> dict[str, Any]:
    """Load canonical 7-class semantic segmentation labels matching `seg_contract.py`."""
    import numpy as np

    src = Path(npz_path)
    if not src.exists():
        raise FileNotFoundError(f"Semantic labels NPZ not found: {src}")

    with np.load(str(src), allow_pickle=False) as data:
        keys = set(data.files)
        face_labels = (
            np.asarray(data["face_labels"], dtype=np.int32).reshape(-1)
            if "face_labels" in keys
            else None
        )
        vertex_labels = (
            np.asarray(data["vertex_labels"], dtype=np.int32).reshape(-1)
            if "vertex_labels" in keys
            else None
        )
        part_ids_raw = (
            np.asarray(data["part_ids_raw"], dtype=np.int32).reshape(-1)
            if "part_ids_raw" in keys
            else None
        )
        confidence = (
            np.asarray(data["confidence"], dtype=np.float32).reshape(-1)
            if "confidence" in keys
            else None
        )
        class_names = get_canonical_class_names()
        if "class_names_json" in keys:
            raw_cn = data["class_names_json"]
            cn_str = str(raw_cn.item() if hasattr(raw_cn, "item") else raw_cn)
            try:
                parsed = json.loads(cn_str)
                if isinstance(parsed, dict):
                    class_names = {int(k): str(v) for k, v in parsed.items()}
                elif isinstance(parsed, list):
                    class_names = {i: str(v) for i, v in enumerate(parsed)}
            except Exception:
                pass
        method_name = ""
        if "method_name" in keys:
            raw_m = data["method_name"]
            method_name = str(raw_m.item() if hasattr(raw_m, "item") else raw_m)
        elapsed_s = 0.0
        if "elapsed_s" in keys:
            raw_e = data["elapsed_s"]
            elapsed_s = float(raw_e.item() if hasattr(raw_e, "item") else raw_e)

    if face_labels is None and vertex_labels is None:
        raise KeyError(
            f"Semantic NPZ {src} contains neither 'face_labels' nor 'vertex_labels' (keys={sorted(keys)})"
        )
    return {
        "face_labels": face_labels,
        "vertex_labels": vertex_labels,
        "part_ids_raw": part_ids_raw,
        "confidence": confidence,
        "class_names": class_names,
        "method_name": method_name,
        "elapsed_s": elapsed_s,
    }


def vertex_labels_to_face_labels(faces: Any, vertex_labels: Any) -> Any:
    """Derive per-face integer labels from per-vertex labels by majority vote (ties -> lowest vertex)."""
    import numpy as np

    f = np.asarray(faces, dtype=np.int64)
    vl = np.asarray(vertex_labels, dtype=np.int32).reshape(-1)
    if len(f) == 0:
        return np.zeros((0,), dtype=np.int32)
    l0 = vl[f[:, 0]]
    l1 = vl[f[:, 1]]
    l2 = vl[f[:, 2]]
    out = np.where((l1 == l2) & (l1 != l0), l1, l0)
    return out.astype(np.int32)


def transfer_face_labels_kdtree(
    src_centroids: Any,
    src_face_labels: Any,
    dst_centroids: Any,
) -> Any:
    """Transfer per-face semantic labels between meshes via nearest face centroid KDTree."""
    import numpy as np
    from scipy.spatial import cKDTree

    sc = np.asarray(src_centroids, dtype=np.float64)
    sl = np.asarray(src_face_labels, dtype=np.int32).reshape(-1)
    dc = np.asarray(dst_centroids, dtype=np.float64)
    if len(dc) == 0:
        return np.zeros((0,), dtype=np.int32)
    if len(sc) == 0 or len(sl) == 0:
        return np.zeros((len(dc),), dtype=np.int32)
    tree = cKDTree(sc)
    _, nn = tree.query(dc, workers=-1)
    return sl[nn].astype(np.int32)


def compute_part_targets(
    face_labels: Any,
    face_areas: Any,
    target_quads: int = 35000,
    part_targets: Mapping[int, int] | None = None,
) -> dict[int, int]:
    """Resolve per-part target quad counts for present classes summing to `target_quads`.

    Auto-selects bald vs full character budget profiles when `part_targets` is omitted:
    - Bald (`hair` area share < 1.5%): head_face ~8k, upper_garment ~10k, pleated_skirt ~5k,
      limbs_hands + socks_shoes ~12k (35k total).
    - Full (`hair` present): proportional ~35k with hair (~6.5k) and rigid_accessory (~1k) budgeted.
    """
    import numpy as np

    fl = np.asarray(face_labels, dtype=np.int32).reshape(-1)
    fa = np.asarray(face_areas, dtype=np.float64).reshape(-1)
    total_area = float(np.sum(fa)) if len(fa) else 0.0
    if total_area <= 1e-12 or len(fl) == 0:
        return {1: int(target_quads)}

    present_classes = sorted(int(c) for c in np.unique(fl))
    class_areas = {c: float(np.sum(fa[fl == c])) for c in present_classes}
    class_area_shares = {c: class_areas[c] / total_area for c in present_classes}

    if part_targets is not None:
        base_profile = {int(k): max(0, int(v)) for k, v in part_targets.items()}
    else:
        hair_share = class_area_shares.get(2, 0.0)
        base_profile = dict(DEFAULT_BALD_PART_TARGETS if hair_share < 0.015 else DEFAULT_FULL_PART_TARGETS)

    # Assign raw weights for every present class; classes omitted from profile get area-proportional weight.
    raw_weights: dict[int, float] = {}
    specified_sum = sum(base_profile.get(c, 0) for c in present_classes if c in base_profile and base_profile[c] > 0)
    specified_area = sum(class_areas[c] for c in present_classes if c in base_profile and base_profile[c] > 0)
    fallback_density = (specified_sum / specified_area) if (specified_sum > 0 and specified_area > 1e-12) else (target_quads / total_area)

    for c in present_classes:
        if c in base_profile and base_profile[c] > 0:
            raw_weights[c] = float(base_profile[c])
        else:
            raw_weights[c] = max(1.0, fallback_density * class_areas[c])

    w_sum = sum(raw_weights.values())
    if w_sum <= 1e-12:
        return {c: max(1, int(round(target_quads * class_area_shares[c]))) for c in present_classes}

    # Proportional integer allocation summing strictly to target_quads.
    exact = {c: float(target_quads) * (raw_weights[c] / w_sum) for c in present_classes}
    alloc = {c: max(1, int(math.floor(exact[c]))) for c in present_classes}
    remainder = int(target_quads) - sum(alloc.values())
    frac_order = sorted(present_classes, key=lambda c: exact[c] - math.floor(exact[c]), reverse=True)
    idx = 0
    while remainder > 0 and frac_order:
        alloc[frac_order[idx % len(frac_order)]] += 1
        remainder -= 1
        idx += 1
    while remainder < 0 and frac_order:
        cand = frac_order[-(idx % len(frac_order)) - 1]
        if alloc[cand] > 1:
            alloc[cand] -= 1
            remainder += 1
        idx += 1
        if idx > len(frac_order) * 4:
            break
    return alloc


def requad_signal_to_rgba(signal: Any) -> Any:
    """Convert ReQuad paint signal `s in [-1, 1]` to exact RGBA colors `(R, G, B, 1)`.

    Matches `tools/requad/operator.py` line 288:
        signal = R - (G + B) / 2.0
        edge_length_multiplier = 4.0 ** (-signal)
    So `R = 0.5 + 0.5*s`, `G = B = 0.5 - 0.5*s` recovers `signal` identically:
    - s = +1 -> (1, 0, 0) pure red   -> 0.25x edge length (16x denser quads)
    - s =  0 -> (0.5, 0.5, 0.5) gray -> 1.00x edge length (neutral)
    - s = -1 -> (0, 1, 1) pure cyan  -> 4.00x edge length (1/16x sparser quads)
    """
    import numpy as np

    s = np.clip(np.asarray(signal, dtype=np.float64), -1.0, 1.0)
    rgba = np.empty((len(s), 4), dtype=np.float32)
    rgba[:, 0] = (0.5 + 0.5 * s).astype(np.float32)
    rgba[:, 1] = (0.5 - 0.5 * s).astype(np.float32)
    rgba[:, 2] = (0.5 - 0.5 * s).astype(np.float32)
    rgba[:, 3] = 1.0
    return rgba


def compute_requad_paint_density_colors(
    verts: Any,
    faces: Any,
    face_labels: Any,
    *,
    target_quads: int = 35000,
    part_targets: Mapping[int, int] | None = None,
    max_signal_abs: float = 0.98,
) -> tuple[Any, dict[str, Any]]:
    """Compute per-vertex RGBA paint-density colors for ReQuad v0.11.3 from 7-class labels.

    Returns:
        vertex_rgba: `(V, 4)` float32 RGBA array for Blender `POINT` color attribute.
        diagnostics: JSON-safe dict with per-part area shares, target quad counts,
            edge multipliers, and paint signals.
    """
    import numpy as np

    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    fl = np.asarray(face_labels, dtype=np.int32).reshape(-1)
    if len(f) != len(fl):
        raise ValueError(f"face_labels length ({len(fl)}) != face count ({len(f)})")

    if len(v) == 0 or len(f) == 0:
        return np.zeros((len(v), 4), dtype=np.float32), {"parts": {}}

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    face_areas = np.where(np.isfinite(face_areas) & (face_areas > 0.0), face_areas, 1e-12)
    total_area = float(np.sum(face_areas))

    resolved_targets = compute_part_targets(
        fl, face_areas, target_quads=int(target_quads), part_targets=part_targets
    )
    target_sum = float(sum(resolved_targets.values()))
    class_names = get_canonical_class_names()

    class_signals: dict[int, float] = {}
    parts_diag: dict[str, dict[str, Any]] = {}
    for c, tgt in resolved_targets.items():
        mask = fl == c
        area_c = float(np.sum(face_areas[mask]))
        area_share = area_c / max(total_area, 1e-12)
        target_share = float(tgt) / max(target_sum, 1.0)
        density_ratio = target_share / max(area_share, 1e-6)
        # Quads ~ Area / edge^2 => edge_mult = 1 / sqrt(density_ratio) = 4^(-signal)
        # => signal = 0.5 * log(density_ratio) / log(4) = 0.25 * log2(density_ratio).
        raw_signal = 0.25 * math.log2(max(density_ratio, 1e-6))
        clamped_signal = float(np.clip(raw_signal, -float(max_signal_abs), float(max_signal_abs)))
        class_signals[c] = clamped_signal
        cname = class_names.get(c, f"class_{c}")
        parts_diag[cname] = {
            "class_id": int(c),
            "target_quads": int(tgt),
            "target_share": round(target_share, 6),
            "area_share": round(area_share, 6),
            "density_ratio": round(density_ratio, 4),
            "edge_multiplier": round(4.0 ** (-clamped_signal), 4),
            "paint_signal": round(clamped_signal, 4),
        }

    # Map per-face signal onto vertices via area-weighted average.
    face_sig = np.zeros(len(f), dtype=np.float64)
    for c, sig in class_signals.items():
        face_sig[fl == c] = sig

    vert_sig_num = np.zeros(len(v), dtype=np.float64)
    vert_sig_den = np.zeros(len(v), dtype=np.float64)
    for col in range(3):
        np.add.at(vert_sig_num, f[:, col], face_sig * face_areas)
        np.add.at(vert_sig_den, f[:, col], face_areas)
    vert_sig = np.where(vert_sig_den > 0.0, vert_sig_num / np.maximum(vert_sig_den, 1e-12), 0.0)
    vertex_rgba = requad_signal_to_rgba(vert_sig)

    diagnostics = {
        "target_quads": int(target_quads),
        "resolved_target_sum": int(sum(resolved_targets.values())),
        "total_surface_area": total_area,
        "parts": parts_diag,
    }
    return vertex_rgba, diagnostics


def clean_small_face_label_islands(
    faces: Any,
    face_labels: Any,
    *,
    min_faces: int = 24,
) -> Any:
    """Absorb tiny disconnected same-class face islands (< min_faces) into their majority neighbor class.

    Prevents 1-10 face segmentation speckles from shattering ReQuad material Flow Guides
    and part-aware UV islands while preserving all genuine semantic part boundaries.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    f = np.asarray(faces, dtype=np.int64)
    fl = np.asarray(face_labels, dtype=np.int32).reshape(-1).copy()
    nf = len(f)
    if nf == 0 or len(fl) != nf or int(min_faces) <= 1:
        return fl

    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    u = np.minimum(edges[:, 0], edges[:, 1])
    v = np.maximum(edges[:, 0], edges[:, 1])
    f_ids = np.tile(np.arange(nf, dtype=np.int64), 3)
    order = np.lexsort((v, u))
    us, vs, fs = u[order], v[order], f_ids[order]
    same_edge = (us[1:] == us[:-1]) & (vs[1:] == vs[:-1])
    fa = fs[:-1][same_edge]
    fb = fs[1:][same_edge]
    if len(fa) == 0:
        return fl

    # Build same-label face adjacency for connected components.
    same_lbl = fl[fa] == fl[fb]
    s_fa = fa[same_lbl]
    s_fb = fb[same_lbl]
    rows = np.concatenate([s_fa, s_fb])
    cols = np.concatenate([s_fb, s_fa])
    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(nf, nf))
    n_comp, comp_ids = connected_components(adj, directed=False)
    comp_sizes = np.bincount(comp_ids, minlength=n_comp)

    small_comps = np.flatnonzero(comp_sizes < int(min_faces))
    if len(small_comps) == 0:
        return fl

    # Map border adjacencies from small components to larger/different-label neighbors.
    diff_lbl = ~same_lbl
    d_fa = fa[diff_lbl]
    d_fb = fb[diff_lbl]
    small_set = set(int(c) for c in small_comps)
    nbr_votes: dict[int, dict[int, int]] = {}
    for a, b in zip(d_fa, d_fb):
        ca, cb = int(comp_ids[a]), int(comp_ids[b])
        if ca in small_set and comp_sizes[cb] >= comp_sizes[ca]:
            d = nbr_votes.setdefault(ca, {})
            lb = int(fl[b])
            d[lb] = d.get(lb, 0) + 1
        if cb in small_set and comp_sizes[ca] >= comp_sizes[cb]:
            d = nbr_votes.setdefault(cb, {})
            la = int(fl[a])
            d[la] = d.get(la, 0) + 1

    # Reassign each small component to its highest-vote bordering class.
    remap_lbl: dict[int, int] = {}
    for cid in small_comps:
        votes = nbr_votes.get(int(cid))
        if votes:
            best_cls = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            remap_lbl[int(cid)] = int(best_cls)

    if remap_lbl:
        for cid, new_cls in remap_lbl.items():
            fl[comp_ids == cid] = new_cls
    return fl


def extract_part_boundary_edges(
    faces: Any,
    face_labels: Any,
    *,
    ignore_unknown: bool = False,
) -> Any:
    """Return sorted unique undirected edges `(B, 2)` separating faces of different semantic classes."""
    import numpy as np

    f = np.asarray(faces, dtype=np.int64)
    fl = np.asarray(face_labels, dtype=np.int32).reshape(-1)
    if len(f) == 0 or len(fl) != len(f):
        return np.zeros((0, 2), dtype=np.int64)

    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    u = np.minimum(edges[:, 0], edges[:, 1])
    v = np.maximum(edges[:, 0], edges[:, 1])
    f_ids = np.tile(np.arange(len(f), dtype=np.int64), 3)

    order = np.lexsort((v, u))
    u_s = u[order]
    v_s = v[order]
    f_s = f_ids[order]

    same = (u_s[1:] == u_s[:-1]) & (v_s[1:] == v_s[:-1])
    fa = f_s[:-1][same]
    fb = f_s[1:][same]
    la = fl[fa]
    lb = fl[fb]
    diff_cls = la != lb
    if ignore_unknown:
        diff_cls &= (la > 0) & (lb > 0)

    if not np.any(diff_cls):
        return np.zeros((0, 2), dtype=np.int64)

    bu = u_s[:-1][same][diff_cls]
    bv = v_s[:-1][same][diff_cls]
    pairs = np.column_stack([bu, bv])
    # Deduplicate in case of non-manifold edges.
    uniq = np.unique(pairs, axis=0)
    return np.ascontiguousarray(uniq, dtype=np.int64)


def compute_x_symmetry_error(verts: Any) -> dict[str, float]:
    """Measure X=0 mirror symmetry error normalized by mesh bounding-box height (Y or Z extent)."""
    import numpy as np
    from scipy.spatial import cKDTree

    v = np.asarray(verts, dtype=np.float64)
    if len(v) == 0:
        return {"mean": 0.0, "p95": 0.0, "p99": 0.0}

    extents = np.ptp(v, axis=0)
    norm_scale = float(max(extents[1], extents[2], np.linalg.norm(extents), 1e-8))
    mirrored = v.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    tree = cKDTree(v)
    dists, _ = tree.query(mirrored, workers=-1)
    rel = dists / norm_scale
    return {
        "mean": float(np.mean(rel)),
        "p95": float(np.percentile(rel, 95.0)),
        "p99": float(np.percentile(rel, 99.0)),
    }


def compute_polygon_edge_stats(num_verts: int, polygons: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Compute true polygon-mesh edge manifoldness, boundary count, and vertex valence stats."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    if num_verts <= 0 or not polygons:
        return {
            "open_boundary_edges": 0,
            "non_manifold_edges": 0,
            "connected_components": 0,
            "valence_4_ratio": 0.0,
            "valence_3_ratio": 0.0,
            "valence_5_ratio": 0.0,
            "valence_ge8_ratio": 0.0,
        }

    eu_list: list[int] = []
    ev_list: list[int] = []
    ref = np.zeros(num_verts, dtype=bool)
    for poly in polygons:
        k = len(poly)
        if k < 3:
            continue
        for i in range(k):
            a = int(poly[i])
            b = int(poly[(i + 1) % k])
            if a == b:
                continue
            ref[a] = True
            ref[b] = True
            if a < b:
                eu_list.append(a)
                ev_list.append(b)
            else:
                eu_list.append(b)
                ev_list.append(a)

    if not eu_list:
        return {
            "open_boundary_edges": 0,
            "non_manifold_edges": 0,
            "connected_components": 0,
            "valence_4_ratio": 0.0,
            "valence_3_ratio": 0.0,
            "valence_5_ratio": 0.0,
            "valence_ge8_ratio": 0.0,
        }

    u = np.asarray(eu_list, dtype=np.int64)
    v = np.asarray(ev_list, dtype=np.int64)
    order = np.lexsort((v, u))
    us = u[order]
    vs = v[order]
    diff = np.empty(len(us), dtype=bool)
    diff[0] = True
    if len(us) > 1:
        diff[1:] = (us[1:] != us[:-1]) | (vs[1:] != vs[:-1])
    starts = np.flatnonzero(diff)
    uniq_u = us[starts]
    uniq_v = vs[starts]
    counts = np.diff(np.append(starts, len(us)))

    open_boundary_edges = int(np.sum(counts == 1))
    non_manifold_edges = int(np.sum(counts > 2))

    deg = np.bincount(np.concatenate([uniq_u, uniq_v]), minlength=num_verts)
    ref_deg = deg[ref]
    denom = max(1, len(ref_deg))

    rows = np.concatenate([uniq_u, uniq_v])
    cols = np.concatenate([uniq_v, uniq_u])
    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num_verts, num_verts))
    _, comp_lbls = connected_components(adj, directed=False)
    conn_comps = int(np.unique(comp_lbls[ref]).size)

    return {
        "open_boundary_edges": open_boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "connected_components": conn_comps,
        "valence_4_ratio": float(np.sum(ref_deg == 4) / denom),
        "valence_3_ratio": float(np.sum(ref_deg == 3) / denom),
        "valence_5_ratio": float(np.sum(ref_deg == 5) / denom),
        "valence_ge8_ratio": float(np.sum(ref_deg >= 8) / denom),
    }


def compute_quad_mesh_metrics(
    mesh_path: str | Path,
    *,
    highpoly_mesh_path: str | Path | None = None,
    highpoly_points: Any | None = None,
    highpoly_tree: Any | None = None,
    label_centroids: Any | None = None,
    label_ids: Any | None = None,
    sample_count: int = 100000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate true quad/polygon topology, X=0 symmetry, per-part counts, and fidelity vs highpoly."""
    import numpy as np

    from quadvox.s2_retopo.metrics import (
        compute_corner_fan_components,
        compute_surface_distance_metrics,
        extract_vf,
    )

    p = Path(mesh_path)
    if p.suffix.lower() == ".obj":
        verts, polygons, tri_faces, pcounts = parse_obj_polygons(p)
    else:
        verts, tri_faces = extract_vf(p)
        polygons = tri_faces.tolist()
        pcounts = {
            "poly_count": int(len(tri_faces)),
            "quad_count": 0,
            "tri_count": int(len(tri_faces)),
            "ngon_count": 0,
            "equiv_tri_count": int(len(tri_faces)),
        }

    edge_stats = compute_polygon_edge_stats(len(verts), polygons)
    _, fan_counts = compute_corner_fan_components(len(verts), tri_faces)
    non_manifold_vertices = int(np.sum(fan_counts > 1))
    is_watertight = (
        pcounts["poly_count"] > 0
        and edge_stats["open_boundary_edges"] == 0
        and edge_stats["non_manifold_edges"] == 0
        and non_manifold_vertices == 0
    )
    quad_ratio = (
        float(pcounts["quad_count"]) / float(pcounts["poly_count"])
        if pcounts["poly_count"] > 0
        else 0.0
    )
    sym_stats = compute_x_symmetry_error(verts)

    dist_stats: dict[str, float | None] = {
        "chamfer_distance": None,
        "chamfer_high_to_low": None,
        "chamfer_low_to_high": None,
        "hausdorff_p95": None,
        "hausdorff_p99": None,
    }
    if highpoly_points is not None:
        dist_stats = compute_surface_distance_metrics(
            verts,
            tri_faces,
            high_points=highpoly_points,
            high_tree=highpoly_tree,
            sample_count=int(sample_count),
            seed=int(seed),
        )
    elif highpoly_mesh_path is not None and Path(highpoly_mesh_path).exists():
        hv, hf = extract_vf(highpoly_mesh_path)
        dist_stats = compute_surface_distance_metrics(
            verts,
            tri_faces,
            high_verts=hv,
            high_faces=hf,
            sample_count=int(sample_count),
            seed=int(seed),
        )

    per_part_counts: dict[str, int] = {}
    per_part_shares: dict[str, float] = {}
    if label_centroids is not None and label_ids is not None and polygons and len(verts) > 0:
        poly_centers = np.array(
            [verts[np.asarray(poly, dtype=np.int64)].mean(axis=0) for poly in polygons],
            dtype=np.float64,
        )
        poly_labels = transfer_face_labels_kdtree(label_centroids, label_ids, poly_centers)
        class_names = get_canonical_class_names()
        total_polys = float(len(polygons))
        for cid in sorted(int(c) for c in np.unique(poly_labels)):
            cnt = int(np.sum(poly_labels == cid))
            cname = class_names.get(cid, f"class_{cid}")
            per_part_counts[cname] = cnt
            per_part_shares[cname] = round(cnt / max(total_polys, 1.0), 6)

    return {
        "vertex_count": int(len(verts)),
        "poly_count": int(pcounts["poly_count"]),
        "quad_count": int(pcounts["quad_count"]),
        "tri_count": int(pcounts["tri_count"]),
        "ngon_count": int(pcounts["ngon_count"]),
        "equiv_tri_count": int(pcounts["equiv_tri_count"]),
        "quad_ratio": quad_ratio,
        "open_boundary_edges": int(edge_stats["open_boundary_edges"]),
        "non_manifold_edges": int(edge_stats["non_manifold_edges"]),
        "non_manifold_vertices": non_manifold_vertices,
        "connected_components": int(edge_stats["connected_components"]),
        "is_watertight": bool(is_watertight),
        "valence_4_ratio": float(edge_stats["valence_4_ratio"]),
        "valence_3_ratio": float(edge_stats["valence_3_ratio"]),
        "valence_5_ratio": float(edge_stats["valence_5_ratio"]),
        "valence_ge8_ratio": float(edge_stats["valence_ge8_ratio"]),
        "x_symmetry_error_mean": float(sym_stats["mean"]),
        "x_symmetry_error_p95": float(sym_stats["p95"]),
        "x_symmetry_error_p99": float(sym_stats["p99"]),
        "chamfer_distance": dist_stats["chamfer_distance"],
        "chamfer_high_to_low": dist_stats["chamfer_high_to_low"],
        "chamfer_low_to_high": dist_stats["chamfer_low_to_high"],
        "hausdorff_p95": dist_stats["hausdorff_p95"],
        "hausdorff_p99": dist_stats["hausdorff_p99"],
        "per_part_quad_counts": per_part_counts,
        "per_part_quad_shares": per_part_shares,
    }


def build_clean_proxy_open3d(
    input_mesh_path: str | Path,
    out_proxy_path: str | Path,
    *,
    target_faces: int = 120000,
) -> Path:
    """Build a clean manifold proxy via Open3D QEM when an input exceeds remesher limits."""
    import trimesh

    from quadvox.s2_retopo.metrics import extract_vf
    from quadvox.s2_retopo.preclean import preclean_mesh
    from quadvox.s2_retopo.qem import decimate_open3d

    in_p = Path(input_mesh_path)
    out_p = Path(out_proxy_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    verts, faces = extract_vf(in_p)
    if len(faces) <= int(target_faces):
        preclean_mesh(in_p, out_p, min_component_area_rel=0.0, dedup_faces=True)
        return out_p

    v_dec, f_dec = decimate_open3d(verts, faces, target_faces=int(target_faces))
    raw_tmp = out_p.parent / f"._{out_p.stem}_raw_proxy{out_p.suffix}"
    try:
        trimesh.Trimesh(vertices=v_dec, faces=f_dec, process=False).export(str(raw_tmp))
        preclean_mesh(raw_tmp, out_p, min_component_area_rel=0.0, dedup_faces=True)
    finally:
        raw_tmp.unlink(missing_ok=True)
    return out_p


# ---------------------------------------------------------------------------
# Blender-side headless execution helpers (only imported inside Blender)
# ---------------------------------------------------------------------------


def _blender_import_single_mesh(mesh_path: Path, name: str) -> Any:
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

    # Clear parent links while keeping world transforms, then join & apply transforms so
    # local space == world space (guaranteeing X=0 is the exact sagittal symmetry plane).
    for obj in meshes:
        if obj.parent is not None:
            mw = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = mw

    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    active = bpy.context.view_layer.objects.active
    for obj in imported:
        if obj != active and obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.select_all(action="DESELECT")
    active.select_set(True)
    bpy.context.view_layer.objects.active = active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    active.name = name
    active.data.name = name
    return active


def _blender_shrinkwrap_to_target(
    low_obj: Any,
    target_obj: Any,
    *,
    snap_x_symmetry_seam: bool = False,
) -> None:
    import bpy
    import numpy as np

    bpy.ops.object.select_all(action="DESELECT")
    low_obj.select_set(True)
    bpy.context.view_layer.objects.active = low_obj

    pre_seam_mask = None
    if snap_x_symmetry_seam and len(low_obj.data.vertices) > 0:
        coords = np.empty(len(low_obj.data.vertices) * 3, dtype=np.float32)
        low_obj.data.vertices.foreach_get("co", coords)
        coords = coords.reshape(-1, 3)
        pre_seam_mask = np.abs(coords[:, 0]) < 1e-4

    mod = low_obj.modifiers.new(name="HighpolyShrinkwrap", type="SHRINKWRAP")
    mod.target = target_obj
    mod.wrap_method = "NEAREST_SURFACEPOINT"
    mod.wrap_mode = "ON_SURFACE"
    mod.offset = 0.0
    bpy.ops.object.modifier_apply(modifier=mod.name)

    if pre_seam_mask is not None and np.any(pre_seam_mask) and len(low_obj.data.vertices) == len(pre_seam_mask):
        coords = np.empty(len(low_obj.data.vertices) * 3, dtype=np.float32)
        low_obj.data.vertices.foreach_get("co", coords)
        coords = coords.reshape(-1, 3)
        coords[pre_seam_mask, 0] = 0.0
        low_obj.data.vertices.foreach_set("co", coords.ravel())
        low_obj.data.update()


def _blender_export_quad_obj_and_glb(obj: Any, out_quad_obj: Path, out_tri_glb: Path) -> None:
    import bpy

    out_quad_obj.parent.mkdir(parents=True, exist_ok=True)
    out_tri_glb.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    obj.data.polygons.foreach_set("use_smooth", [True] * len(obj.data.polygons))
    obj.data.update()

    # Export native quad OBJ in y_up_z_front axes matching GLB / trimesh / render.py.
    bpy.ops.wm.obj_export(
        filepath=str(out_quad_obj),
        export_selected_objects=True,
        export_materials=False,
        export_uv=False,
        export_normals=True,
        export_triangulated_mesh=False,
        forward_axis="NEGATIVE_Z",
        up_axis="Y",
    )

    bpy.ops.export_scene.gltf(
        filepath=str(out_tri_glb),
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_normals=True,
        export_materials="NONE",
    )


def _blender_clean_manifold_normals(obj: Any) -> None:
    """Clean degenerate/non-manifold elements and enforce consistent outward face normals."""
    import bmesh

    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-6)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-7, edges=list(bm.edges))
    # Strip faces incident to non-manifold edges (>2 adjacent faces) so QuadriFlow accepts the mesh.
    for _ in range(3):
        bad_faces = set()
        for e in bm.edges:
            if len(e.link_faces) > 2:
                for f in e.link_faces:
                    bad_faces.add(f)
        if not bad_faces:
            break
        bmesh.ops.delete(bm, geom=list(bad_faces), context="FACES")
    loose_edges = [e for e in bm.edges if len(e.link_faces) == 0]
    if loose_edges:
        bmesh.ops.delete(bm, geom=loose_edges, context="EDGES")
    loose_verts = [v for v in bm.verts if len(v.link_faces) == 0]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(me)
    bm.free()
    me.update()


def _blender_voxel_manifold_fallback(obj: Any, voxel_rel: float = 0.0022) -> None:
    """Apply `bpy.ops.object.voxel_remesh()` with `use_remesh_fix_poles=True` for QuadriFlow.

    Note: Blender's REMESH modifier (`mode='VOXEL'`) omits pole fixing and can emit
    manifold-check failures inside QuadriFlow; `object.voxel_remesh` with
    `use_remesh_fix_poles=True` guarantees a clean watertight manifold input.
    """
    import bpy
    import mathutils

    bpy.context.view_layer.update()
    mw = obj.matrix_world
    pts = [mw @ mathutils.Vector(c) for c in obj.bound_box]
    bmin = mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    bmax = mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    diag = max(float((bmax - bmin).length), 1e-4)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    obj.data.remesh_voxel_size = max(1e-4, float(voxel_rel) * diag)
    obj.data.remesh_voxel_adaptivity = 0.0
    obj.data.use_remesh_fix_poles = True
    bpy.ops.object.voxel_remesh()


def _run_quadriflow_in_blender(
    *,
    input_mesh_path: Path,
    highpoly_mesh_path: Path | None,
    out_quad_obj: Path,
    out_tri_glb: Path,
    target_faces: int,
    use_mesh_symmetry: bool,
    use_preserve_sharp: bool,
    use_preserve_boundary: bool,
    smooth_normals: bool,
    seed: int,
    meta_json_path: Path,
) -> None:
    import bpy

    t0 = time.perf_counter()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    src_obj = _blender_import_single_mesh(input_mesh_path, "QuadriFlowSource")

    # Duplicate source as shrinkwrap fallback if no separate highpoly path was supplied.
    shrink_target = None
    if highpoly_mesh_path is not None and highpoly_mesh_path.resolve() != input_mesh_path.resolve():
        shrink_target = _blender_import_single_mesh(highpoly_mesh_path, "HighpolyRef")
    else:
        shrink_copy = src_obj.copy()
        shrink_copy.data = src_obj.data.copy()
        shrink_copy.name = "HighpolyRef"
        bpy.context.scene.collection.objects.link(shrink_copy)
        shrink_target = shrink_copy

    # Ensure manifoldness & consistent outward normals before calling QuadriFlow.
    _blender_clean_manifold_normals(src_obj)

    def _try_quadriflow(sym_flag: bool) -> set[str]:
        bpy.ops.object.select_all(action="DESELECT")
        src_obj.select_set(True)
        bpy.context.view_layer.objects.active = src_obj
        return bpy.ops.object.quadriflow_remesh(
            use_mesh_symmetry=bool(sym_flag),
            use_preserve_sharp=bool(use_preserve_sharp),
            use_preserve_boundary=bool(use_preserve_boundary),
            smooth_normals=bool(smooth_normals),
            mode="FACES",
            target_faces=int(target_faces),
            seed=int(seed),
        )

    used_voxel_fallback = False
    used_symmetry = bool(use_mesh_symmetry)
    res = _try_quadriflow(used_symmetry)

    if res != {"FINISHED"}:
        # Non-manifold pinches or multi-shell hair intersections: guarantee watertight manifold via
        # `object.voxel_remesh(use_remesh_fix_poles=True)`, then re-run QuadriFlow + highpoly shrinkwrap.
        _blender_voxel_manifold_fallback(src_obj, voxel_rel=0.0022)
        used_voxel_fallback = True
        res = _try_quadriflow(used_symmetry)

    if res != {"FINISHED"} and used_symmetry:
        # Final fallback if asymmetric hair accessories trip symmetry bisection.
        used_symmetry = False
        res = _try_quadriflow(False)

    if res != {"FINISHED"}:
        _blender_voxel_manifold_fallback(src_obj, voxel_rel=0.0038)
        used_symmetry = False
        res = _try_quadriflow(False)

    if res != {"FINISHED"}:
        raise RuntimeError(f"bpy.ops.object.quadriflow_remesh failed with result {res}")

    if shrink_target is not None:
        _blender_shrinkwrap_to_target(
            src_obj, shrink_target, snap_x_symmetry_seam=bool(used_symmetry)
        )
        bpy.data.objects.remove(shrink_target, do_unlink=True)

    _blender_export_quad_obj_and_glb(src_obj, out_quad_obj, out_tri_glb)

    meta = {
        "elapsed_s": time.perf_counter() - t0,
        "blender_version": ".".join(str(x) for x in bpy.app.version),
        "poly_count": len(src_obj.data.polygons),
        "quad_count": sum(1 for p in src_obj.data.polygons if len(p.vertices) == 4),
        "used_voxel_fallback": used_voxel_fallback,
        "used_symmetry": used_symmetry,
    }
    meta_json_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def yup_to_blender_zup(points: Any) -> Any:
    """Map `y_up_z_front` `(x, y, z)` coordinates to Blender Z-up `(x, -z, y)`."""
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) coordinate array, got shape {pts.shape}")
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0]
    out[:, 1] = -pts[:, 2]
    out[:, 2] = pts[:, 1]
    return out


def _run_requad_in_blender(
    *,
    proxy_mesh_path: Path,
    highpoly_mesh_path: Path | None,
    guide_npz_path: Path,
    out_quad_obj: Path,
    out_tri_glb: Path,
    target_quads: int,
    sym_x: bool,
    preset: str,
    adaptive_size: float,
    relax_iterations: int,
    meta_json_path: Path,
) -> None:
    import bpy
    import numpy as np
    from mathutils import kdtree

    t0 = time.perf_counter()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="bl_ext.user_default.requad")
    if not hasattr(bpy.context.scene, "requad"):
        raise RuntimeError(
            "ReQuad extension 'bl_ext.user_default.requad' enabled but Scene.requad not registered."
        )

    src_obj = _blender_import_single_mesh(proxy_mesh_path, "ReQuadSource")
    me = src_obj.data

    with np.load(str(guide_npz_path), allow_pickle=False) as gdata:
        gkeys = set(gdata.files)
        vert_rgba = np.asarray(gdata["vertex_rgba"], dtype=np.float32)
        face_labels = np.asarray(gdata["face_labels"], dtype=np.int32).reshape(-1)
        boundary_edges = np.asarray(gdata["boundary_edges"], dtype=np.int64)
        proxy_verts = np.asarray(gdata["proxy_verts"], dtype=np.float64) if "proxy_verts" in gkeys else None
        proxy_centroids = (
            np.asarray(gdata["proxy_centroids"], dtype=np.float64) if "proxy_centroids" in gkeys else None
        )

    verts_exact = len(vert_rgba) == len(me.vertices)
    faces_exact = len(face_labels) == len(me.polygons)

    # 1. Paint Density vertex color attribute (exact ReQuad convention: red=finer, cyan=coarser).
    # Falls back to Blender Z-up KDTree lookup if glTF import split/reindexed vertices.
    if verts_exact:
        aligned_rgba = np.ascontiguousarray(vert_rgba, dtype=np.float32)
    else:
        if proxy_verts is None or len(proxy_verts) != len(vert_rgba):
            raise ValueError(
                f"ReQuad guide vertex_rgba length ({len(vert_rgba)}) != Blender vertex count ({len(me.vertices)}) "
                "and proxy_verts missing for KDTree fallback."
            )
        src_v_zup = yup_to_blender_zup(proxy_verts)
        kd_v = kdtree.KDTree(len(src_v_zup))
        for idx, pt in enumerate(src_v_zup):
            kd_v.insert((float(pt[0]), float(pt[1]), float(pt[2])), idx)
        kd_v.balance()
        aligned_rgba = np.empty((len(me.vertices), 4), dtype=np.float32)
        for vi, v in enumerate(me.vertices):
            _, nn_idx, _ = kd_v.find(v.co)
            aligned_rgba[vi] = vert_rgba[int(nn_idx)]

    cattr = me.color_attributes.new(name="ReQuadDensity", type="FLOAT_COLOR", domain="POINT")
    cattr.data.foreach_set("color", aligned_rgba.ravel())
    me.color_attributes.active_color = cattr

    # 2. Material boundaries as Flow Guides (one material slot per semantic class).
    # Falls back to Blender Z-up KDTree lookup if glTF import altered polygon count/order.
    if faces_exact:
        aligned_fl = face_labels
    elif proxy_centroids is not None and len(proxy_centroids) == len(face_labels) and len(me.polygons) > 0:
        src_c_zup = yup_to_blender_zup(proxy_centroids)
        kd_f = kdtree.KDTree(len(src_c_zup))
        for idx, pt in enumerate(src_c_zup):
            kd_f.insert((float(pt[0]), float(pt[1]), float(pt[2])), idx)
        kd_f.balance()
        aligned_fl = np.empty(len(me.polygons), dtype=np.int32)
        for pi, poly in enumerate(me.polygons):
            _, nn_idx, _ = kd_f.find(poly.center)
            aligned_fl[pi] = face_labels[int(nn_idx)]
    else:
        aligned_fl = None

    if aligned_fl is not None and len(aligned_fl) == len(me.polygons):
        uniq_cls = sorted(int(c) for c in np.unique(aligned_fl))
        cls_to_slot: dict[int, int] = {}
        me.materials.clear()
        for slot_idx, cid in enumerate(uniq_cls):
            mat = bpy.data.materials.new(name=f"SemPart_{cid}")
            me.materials.append(mat)
            cls_to_slot[cid] = slot_idx
        mat_indices = np.array([cls_to_slot[int(c)] for c in aligned_fl], dtype=np.int32)
        me.polygons.foreach_set("material_index", mat_indices)

    # 3. Mark semantic part-boundary edges as sharp & UV seam Flow Guides.
    if len(me.edges) > 0:
        bset: set[tuple[int, int]] = set()
        if verts_exact and faces_exact and len(boundary_edges) > 0:
            bset = {(int(min(a, b)), int(max(a, b))) for a, b in boundary_edges}
        elif aligned_fl is not None and len(aligned_fl) == len(me.polygons):
            edge_cls: dict[tuple[int, int], int] = {}
            for pi, poly in enumerate(me.polygons):
                lbl = int(aligned_fl[pi])
                for a, b in poly.edge_keys:
                    ekey = (a, b) if a < b else (b, a)
                    prev = edge_cls.get(ekey)
                    if prev is None:
                        edge_cls[ekey] = lbl
                    elif prev != lbl:
                        bset.add(ekey)
        if bset:
            n_e = len(me.edges)
            ev = np.empty(n_e * 2, dtype=np.int64)
            me.edges.foreach_get("vertices", ev)
            ev = ev.reshape(-1, 2)
            flags = np.array(
                [(int(min(a, b)), int(max(a, b))) in bset for a, b in ev],
                dtype=bool,
            )
            me.edges.foreach_set("use_edge_sharp", flags)
            me.edges.foreach_set("use_seam", flags)
    me.update()

    # 4. Configure & invoke ReQuad headlessly (`bpy.app.background == True` runs `_run_blocking`).
    rq = bpy.context.scene.requad
    rq.preset = str(preset).upper()
    rq.target_count = int(target_quads)
    rq.count_mode = "QUADS"
    rq.sym_x = bool(sym_x)
    rq.sym_y = False
    rq.sym_z = False
    rq.use_paint_density = True
    rq.adaptive_size = float(adaptive_size)
    rq.adaptive_count = False
    rq.material_guides = True
    rq.guide_sharp = True
    rq.guide_seams = True
    rq.pre_remesh = True
    rq.align_singularities = True
    rq.relax_iterations = int(relax_iterations)
    rq.keep_materials = False
    rq.smooth_result = True
    rq.hide_original = False

    bpy.ops.object.select_all(action="DESELECT")
    src_obj.select_set(True)
    bpy.context.view_layer.objects.active = src_obj

    before_objs = set(bpy.data.objects)
    res = bpy.ops.requad.remesh()
    if res != {"FINISHED"}:
        raise RuntimeError(f"bpy.ops.requad.remesh failed with status {res}")

    new_meshes = [o for o in bpy.data.objects if o not in before_objs and o.type == "MESH"]
    if not new_meshes:
        cand = bpy.data.objects.get(src_obj.name + "_requad")
        if cand is not None and cand.type == "MESH":
            new_meshes = [cand]
    if not new_meshes:
        raise RuntimeError("ReQuad finished without producing a result mesh object.")
    rq_obj = new_meshes[0]

    # 5. Final Relax & Shrinkwrap projection onto original 1.99M / 1.56M highpoly.
    if highpoly_mesh_path is not None and highpoly_mesh_path.exists():
        hp_obj = _blender_import_single_mesh(highpoly_mesh_path, "HighpolyRef")
        _blender_shrinkwrap_to_target(rq_obj, hp_obj, snap_x_symmetry_seam=bool(sym_x))
        bpy.data.objects.remove(hp_obj, do_unlink=True)
    else:
        _blender_shrinkwrap_to_target(rq_obj, src_obj, snap_x_symmetry_seam=bool(sym_x))

    _blender_export_quad_obj_and_glb(rq_obj, out_quad_obj, out_tri_glb)

    meta = {
        "elapsed_s": time.perf_counter() - t0,
        "blender_version": ".".join(str(x) for x in bpy.app.version),
        "poly_count": len(rq_obj.data.polygons),
        "quad_count": sum(1 for p in rq_obj.data.polygons if len(p.vertices) == 4),
    }
    meta_json_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Public Python API: QuadriFlow, ReQuad Semantic, and QuadWild Bi-MDF CLI
# ---------------------------------------------------------------------------


def build_quadriflow_blender_cmd(
    *,
    blender_bin: str | Path,
    input_mesh_path: str | Path,
    highpoly_mesh_path: str | Path | None,
    out_quad_obj: str | Path,
    out_tri_glb: str | Path,
    meta_json_path: str | Path,
    target_faces: int = 35000,
    use_mesh_symmetry: bool = False,
    use_preserve_sharp: bool = True,
    use_preserve_boundary: bool = True,
    smooth_normals: bool = True,
    seed: int = 0,
) -> list[str]:
    """Construct the headless Blender CLI command for QuadriFlow."""
    cmd = [
        str(Path(blender_bin)),
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "quadriflow-blender",
        "--input",
        str(Path(input_mesh_path)),
        "--out-quad-obj",
        str(Path(out_quad_obj)),
        "--out-tri-glb",
        str(Path(out_tri_glb)),
        "--meta-json",
        str(Path(meta_json_path)),
        "--target-faces",
        str(int(target_faces)),
        "--seed",
        str(int(seed)),
    ]
    if highpoly_mesh_path is not None:
        cmd.extend(["--highpoly", str(Path(highpoly_mesh_path))])
    if use_mesh_symmetry:
        cmd.append("--use-mesh-symmetry")
    if use_preserve_sharp:
        cmd.append("--use-preserve-sharp")
    if use_preserve_boundary:
        cmd.append("--use-preserve-boundary")
    if smooth_normals:
        cmd.append("--smooth-normals")
    return cmd


def run_quadriflow_blender(
    input_mesh_path: str | Path,
    out_quad_obj: str | Path,
    out_tri_glb: str | Path,
    *,
    highpoly_mesh_path: str | Path | None = None,
    labels_npz_path: str | Path | None = None,
    labels_source_mesh_path: str | Path | None = None,
    target_faces: int = 35000,
    use_mesh_symmetry: bool = False,
    use_preserve_sharp: bool = True,
    use_preserve_boundary: bool = True,
    smooth_normals: bool = True,
    seed: int = 0,
    max_input_faces: int = 200000,
    blender_bin: str | Path = DEFAULT_BLENDER_BIN,
    timeout_s: float = 1800.0,
    sample_count: int = 100000,
    report_json_path: str | Path | None = None,
) -> QuadRetopoResult:
    """Run Blender 5.3 headless QuadriFlow with X-symmetry and post-shrinkwrap onto highpoly."""
    import numpy as np

    from quadvox.s2_retopo.metrics import extract_vf

    in_p = Path(input_mesh_path)
    hp_p = Path(highpoly_mesh_path) if highpoly_mesh_path is not None else in_p
    quad_p = Path(out_quad_obj)
    glb_p = Path(out_tri_glb)
    blender_p = Path(blender_bin)

    if not in_p.exists():
        raise FileNotFoundError(f"Input mesh not found: {in_p}")
    if not blender_p.exists():
        raise FileNotFoundError(f"Blender binary not found: {blender_p}")

    quad_p.parent.mkdir(parents=True, exist_ok=True)
    glb_p.parent.mkdir(parents=True, exist_ok=True)

    # Optionally build a clean manifold proxy if raw highpoly is huge (>max_input_faces).
    remesh_in_p = in_p
    ephemeral_proxy: Path | None = None
    if max_input_faces > 0:
        _, f_check = extract_vf(in_p)
        if len(f_check) > int(max_input_faces):
            ephemeral_proxy = quad_p.parent / f"._{quad_p.stem}_qf_proxy_{os.getpid()}.glb"
            build_clean_proxy_open3d(in_p, ephemeral_proxy, target_faces=int(max_input_faces))
            remesh_in_p = ephemeral_proxy

    meta_path = quad_p.parent / f"._{quad_p.stem}_qf_meta_{os.getpid()}.json"
    work_tmp = quad_p.parent / f"._{quad_p.stem}_qf_tmp_{os.getpid()}"
    work_tmp.mkdir(parents=True, exist_ok=True)

    cmd = build_quadriflow_blender_cmd(
        blender_bin=blender_p,
        input_mesh_path=remesh_in_p,
        highpoly_mesh_path=hp_p,
        out_quad_obj=quad_p,
        out_tri_glb=glb_p,
        meta_json_path=meta_path,
        target_faces=int(target_faces),
        use_mesh_symmetry=bool(use_mesh_symmetry),
        use_preserve_sharp=bool(use_preserve_sharp),
        use_preserve_boundary=bool(use_preserve_boundary),
        smooth_normals=bool(smooth_normals),
        seed=int(seed),
    )

    env = os.environ.copy()
    env["TMPDIR"] = str(work_tmp)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            env=env,
        )
        if proc.returncode != 0 or not quad_p.exists() or not glb_p.exists() or not meta_path.exists():
            raise RuntimeError(
                f"Blender QuadriFlow failed (code={proc.returncode}):\n"
                f"CMD: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    finally:
        meta_path.unlink(missing_ok=True)
        shutil.rmtree(work_tmp, ignore_errors=True)
        if ephemeral_proxy is not None:
            ephemeral_proxy.unlink(missing_ok=True)

    # Optional per-part quad measurement if semantic labels were provided.
    label_centroids = None
    label_ids = None
    if labels_npz_path is not None and Path(labels_npz_path).exists():
        sem = load_semantic_npz(labels_npz_path)
        lbl_src = Path(labels_source_mesh_path) if labels_source_mesh_path is not None else hp_p
        sv, sf = extract_vf(lbl_src)
        fl = sem["face_labels"]
        if fl is None and sem["vertex_labels"] is not None and len(sem["vertex_labels"]) == len(sv):
            fl = vertex_labels_to_face_labels(sf, sem["vertex_labels"])
        if fl is not None and len(fl) == len(sf):
            label_centroids = (sv[sf[:, 0]] + sv[sf[:, 1]] + sv[sf[:, 2]]) / 3.0
            label_ids = np.asarray(fl, dtype=np.int32)

    metrics = compute_quad_mesh_metrics(
        quad_p,
        highpoly_mesh_path=hp_p,
        label_centroids=label_centroids,
        label_ids=label_ids,
        sample_count=int(sample_count),
        seed=42,
    )
    result = QuadRetopoResult(
        engine="quadriflow",
        input_mesh_path=str(in_p),
        highpoly_mesh_path=str(hp_p),
        out_quad_obj=str(quad_p),
        out_tri_glb=str(glb_p),
        target_quads=int(target_faces),
        elapsed_s=time.perf_counter() - t0,
        blender_version=str(meta.get("blender_version", "")),
        **metrics,
    )
    if report_json_path is not None:
        result.write_json(report_json_path)
    return result


def build_requad_blender_cmd(
    *,
    blender_bin: str | Path,
    proxy_mesh_path: str | Path,
    highpoly_mesh_path: str | Path | None,
    guide_npz_path: str | Path,
    out_quad_obj: str | Path,
    out_tri_glb: str | Path,
    meta_json_path: str | Path,
    target_quads: int = 35000,
    sym_x: bool = False,
    preset: str = "ORGANIC",
    adaptive_size: float = 20.0,
    relax_iterations: int = 30,
) -> list[str]:
    """Construct the headless Blender CLI command for ReQuad semantic-density remeshing."""
    cmd = [
        str(Path(blender_bin)),
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "requad-blender",
        "--proxy",
        str(Path(proxy_mesh_path)),
        "--guide-npz",
        str(Path(guide_npz_path)),
        "--out-quad-obj",
        str(Path(out_quad_obj)),
        "--out-tri-glb",
        str(Path(out_tri_glb)),
        "--meta-json",
        str(Path(meta_json_path)),
        "--target-quads",
        str(int(target_quads)),
        "--preset",
        str(preset),
        "--adaptive-size",
        str(float(adaptive_size)),
        "--relax-iterations",
        str(int(relax_iterations)),
    ]
    if highpoly_mesh_path is not None:
        cmd.extend(["--highpoly", str(Path(highpoly_mesh_path))])
    if sym_x:
        cmd.append("--sym-x")
    return cmd


def run_requad_semantic_blender(
    proxy_mesh_path: str | Path,
    labels_npz_path: str | Path,
    out_quad_obj: str | Path,
    out_tri_glb: str | Path,
    *,
    highpoly_mesh_path: str | Path | None = None,
    labels_source_mesh_path: str | Path | None = None,
    target_quads: int = 35000,
    part_targets: Mapping[int, int] | None = None,
    sym_x: bool = False,
    preset: str = "ORGANIC",
    adaptive_size: float = 20.0,
    relax_iterations: int = 30,
    requad_dir: str | Path = DEFAULT_REQUAD_DIR,
    blender_bin: str | Path = DEFAULT_BLENDER_BIN,
    timeout_s: float = 2400.0,
    sample_count: int = 100000,
    report_json_path: str | Path | None = None,
) -> QuadRetopoResult:
    """Run ReQuad v0.11.3 headlessly in Blender 5.3 with optional Paint Density, Flow Guides, and Symmetry X."""
    import numpy as np

    from quadvox.s2_retopo.metrics import extract_vf

    proxy_p = Path(proxy_mesh_path)
    hp_p = Path(highpoly_mesh_path) if highpoly_mesh_path is not None else proxy_p
    labels_p = Path(labels_npz_path) if labels_npz_path is not None else None
    quad_p = Path(out_quad_obj)
    glb_p = Path(out_tri_glb)
    rq_dir = Path(requad_dir).resolve()
    blender_p = Path(blender_bin)

    if not proxy_p.exists():
        raise FileNotFoundError(f"Proxy mesh not found: {proxy_p}")
    if labels_p is not None and not labels_p.exists():
        raise FileNotFoundError(f"Semantic labels NPZ not found: {labels_p}")
    if not (rq_dir / "__init__.py").exists():
        raise FileNotFoundError(f"ReQuad addon directory invalid (missing __init__.py): {rq_dir}")
    if not blender_p.exists():
        raise FileNotFoundError(f"Blender binary not found: {blender_p}")

    quad_p.parent.mkdir(parents=True, exist_ok=True)
    glb_p.parent.mkdir(parents=True, exist_ok=True)

    # Ensure bundled engine binaries inside requad_dir are executable.
    eng_bin_dir = rq_dir / "engine" / "linux-x64"
    for bname in ("quadwild", "quad_from_patches"):
        bpath = eng_bin_dir / bname
        if bpath.exists():
            os.chmod(bpath, 0o755)

    # Load proxy mesh and optionally align semantic labels (uniform class 0 if omitted).
    pv, pf = extract_vf(proxy_p)
    proxy_centroids = (pv[pf[:, 0]] + pv[pf[:, 1]] + pv[pf[:, 2]]) / 3.0

    if labels_p is not None:
        sem = load_semantic_npz(labels_p)
        face_labels = sem["face_labels"]
        vert_labels = sem["vertex_labels"]

        if face_labels is not None and len(face_labels) == len(pf):
            proxy_fl = np.asarray(face_labels, dtype=np.int32)
        elif vert_labels is not None and len(vert_labels) == len(pv):
            proxy_fl = vertex_labels_to_face_labels(pf, vert_labels)
        else:
            lbl_src = Path(labels_source_mesh_path) if labels_source_mesh_path is not None else hp_p
            sv, sf = extract_vf(lbl_src)
            if face_labels is not None and len(face_labels) == len(sf):
                src_fl = np.asarray(face_labels, dtype=np.int32)
            elif vert_labels is not None and len(vert_labels) == len(sv):
                src_fl = vertex_labels_to_face_labels(sf, vert_labels)
            else:
                raise ValueError(
                    f"Cannot match semantic labels in {labels_p} to proxy ({len(pf)} faces, {len(pv)} verts) "
                    f"or source {lbl_src} ({len(sf)} faces, {len(sv)} verts)."
                )
            src_centroids = (sv[sf[:, 0]] + sv[sf[:, 1]] + sv[sf[:, 2]]) / 3.0
            proxy_fl = transfer_face_labels_kdtree(src_centroids, src_fl, proxy_centroids)

        # Absorb tiny <32-face semantic speckles so material/sharp Flow Guides stay crisp and manifold.
        proxy_fl = clean_small_face_label_islands(pf, proxy_fl, min_faces=32)
    else:
        proxy_fl = np.zeros(len(pf), dtype=np.int32)

    vert_rgba, density_diag = compute_requad_paint_density_colors(
        pv, pf, proxy_fl, target_quads=int(target_quads), part_targets=part_targets
    )
    boundary_edges = extract_part_boundary_edges(pf, proxy_fl, ignore_unknown=False)

    pid = os.getpid()
    guide_npz = quad_p.parent / f"._{quad_p.stem}_requad_guides_{pid}.npz"
    meta_path = quad_p.parent / f"._{quad_p.stem}_requad_meta_{pid}.json"
    ext_root = quad_p.parent / f"._{quad_p.stem}_requad_ext_{pid}"
    work_tmp = quad_p.parent / f"._{quad_p.stem}_requad_tmp_{pid}"

    np.savez_compressed(
        str(guide_npz),
        vertex_rgba=vert_rgba,
        face_labels=proxy_fl,
        boundary_edges=boundary_edges,
        proxy_verts=np.asarray(pv, dtype=np.float32),
        proxy_centroids=np.asarray(proxy_centroids, dtype=np.float32),
    )

    ext_user_default = ext_root / "user_default"
    ext_user_default.mkdir(parents=True, exist_ok=True)
    work_tmp.mkdir(parents=True, exist_ok=True)
    ext_link = ext_user_default / "requad"
    if ext_link.is_symlink() or ext_link.exists():
        ext_link.unlink()
    ext_link.symlink_to(rq_dir)

    cmd = build_requad_blender_cmd(
        blender_bin=blender_p,
        proxy_mesh_path=proxy_p,
        highpoly_mesh_path=hp_p,
        guide_npz_path=guide_npz,
        out_quad_obj=quad_p,
        out_tri_glb=glb_p,
        meta_json_path=meta_path,
        target_quads=int(target_quads),
        sym_x=bool(sym_x),
        preset=str(preset),
        adaptive_size=float(adaptive_size),
        relax_iterations=int(relax_iterations),
    )

    env = os.environ.copy()
    env["BLENDER_USER_EXTENSIONS"] = str(ext_root)
    env["TMPDIR"] = str(work_tmp)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            env=env,
        )
        if proc.returncode != 0 or not quad_p.exists() or not glb_p.exists() or not meta_path.exists():
            # Surface any ReQuad engine.log found in work_tmp for immediate diagnosis.
            eng_logs = ""
            for lf in sorted(work_tmp.rglob("engine.log")):
                try:
                    eng_logs += f"\n--- {lf.name} ---\n{lf.read_text(encoding='utf-8', errors='replace')[-1500:]}"
                except OSError:
                    pass
            raise RuntimeError(
                f"Blender ReQuad v0.11.3 failed (code={proc.returncode}):\n"
                f"CMD: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}{eng_logs}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    finally:
        guide_npz.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        if ext_link.is_symlink():
            ext_link.unlink(missing_ok=True)
        shutil.rmtree(ext_root, ignore_errors=True)
        shutil.rmtree(work_tmp, ignore_errors=True)

    metrics = compute_quad_mesh_metrics(
        quad_p,
        highpoly_mesh_path=hp_p,
        label_centroids=proxy_centroids,
        label_ids=proxy_fl,
        sample_count=int(sample_count),
        seed=42,
    )
    target_by_name = {
        cname: int(info["target_quads"])
        for cname, info in density_diag.get("parts", {}).items()
    }
    result = QuadRetopoResult(
        engine="requad_semantic",
        input_mesh_path=str(proxy_p),
        highpoly_mesh_path=str(hp_p),
        out_quad_obj=str(quad_p),
        out_tri_glb=str(glb_p),
        target_quads=int(target_quads),
        per_part_target_quads=target_by_name,
        density_diagnostics=density_diag,
        elapsed_s=time.perf_counter() - t0,
        blender_version=str(meta.get("blender_version", "")),
        notes=f"boundary_guide_edges={len(boundary_edges)};method={sem.get('method_name', '')}",
        **metrics,
    )
    if report_json_path is not None:
        result.write_json(report_json_path)
    return result


def _estimate_quadwild_scale_fact(patches_obj: Path, target_quads: int) -> float:
    """Compute calibrated QuadWild `scaleFact` matching ReQuad's area/edge formula."""
    import numpy as np

    verts, _, tris, _ = parse_obj_polygons(patches_obj)
    if len(verts) == 0 or len(tris) == 0:
        return 1.0
    e0 = verts[tris[:, 1]] - verts[tris[:, 0]]
    e1 = verts[tris[:, 2]] - verts[tris[:, 0]]
    e2 = verts[tris[:, 2]] - verts[tris[:, 1]]
    area_total = 0.5 * float(np.sum(np.linalg.norm(np.cross(e0, e1), axis=1)))
    avg_edge = float(
        np.mean(np.concatenate([np.linalg.norm(e0, axis=1), np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1)]))
    )
    if area_total <= 1e-12 or avg_edge <= 1e-12:
        return 1.0
    return float(1.033 * math.sqrt(area_total / max(1, int(target_quads))) / avg_edge)


def build_quadwild_cli_commands(
    *,
    quadwild_dir: str | Path,
    input_obj: str | Path,
    prep_config: str | Path,
    patches_obj: str | Path,
    main_config: str | Path,
    qfp_iter: int = 1,
) -> tuple[list[str], list[str]]:
    """Construct exact QuadWild Bi-MDF step-1 and step-2 CLI command lines."""
    qw_root = Path(quadwild_dir).resolve()
    cmd_step1 = [str(qw_root / "quadwild"), str(Path(input_obj).resolve()), "2", str(Path(prep_config).resolve())]
    cmd_step2 = [
        str(qw_root / "quad_from_patches"),
        str(Path(patches_obj).resolve()),
        str(int(qfp_iter)),
        str(Path(main_config).resolve()),
    ]
    return cmd_step1, cmd_step2


def find_quadwild_output_obj(workdir: Path, patches_obj: Path, run_idx: int) -> Path | None:
    """Locate the quadrangulation OBJ produced by either upstream QuadWild v0.0.2 or ReQuad's qfp."""
    explicit = [
        patches_obj.with_name(f"{patches_obj.stem}_{run_idx}_quadrangulation_smooth.obj"),
        patches_obj.with_name(f"{patches_obj.stem}_quadrangulation_smooth.obj"),
        patches_obj.with_name(f"{patches_obj.stem}_{run_idx}_quadrangulation.obj"),
        patches_obj.with_name(f"{patches_obj.stem}_quadrangulation.obj"),
    ]
    for cand in explicit:
        if cand.exists() and cand.stat().st_size > 0:
            return cand
    # Fallback glob matching any newly generated quadrangulation OBJ in workdir.
    for pattern in ("*quadrangulation_smooth.obj", "*quadrangulation*.obj"):
        matches = [p for p in sorted(workdir.glob(pattern)) if p.stat().st_size > 0]
        if matches:
            return matches[-1]
    return None


def is_quadwild_step2_output_valid(
    returncode: int,
    stdout: str,
    stderr: str,
    cand_path: Path | None,
) -> tuple[bool, int]:
    """Validate QuadWild `quad_from_patches` step-2 output without masking real crashes.

    Upstream statically-linked QuadWild Bi-MDF v0.0.2 can emit SIGSEGV (`returncode == -11`)
    during C++ static destructor teardown AFTER printing `*** Done ***` and saving a valid
    quadrangulation OBJ. Non-zero exit codes are accepted ONLY when `returncode == -11`,
    stdout/stderr explicitly confirms `*** Done ***`, and `cand_path` parses with `quad_count > 0`.
    """
    if cand_path is None or not cand_path.exists() or cand_path.stat().st_size <= 0:
        return False, 0

    try:
        _, _, _, pcounts = parse_obj_polygons(cand_path)
    except Exception:
        return False, 0

    quad_count = int(pcounts.get("quad_count", 0))
    poly_count = int(pcounts.get("poly_count", 0))
    if quad_count <= 0 or poly_count <= 0:
        return False, 0

    if int(returncode) == 0:
        return True, poly_count

    combined = f"{stdout}\n{stderr}"
    if int(returncode) == -11 and "*** Done ***" in combined:
        return True, poly_count

    return False, 0


def _write_rescaled_obj(src_obj: Path, dst_obj: Path, scale: float, center: Sequence[float]) -> None:
    """Copy a quad/poly OBJ while mapping normalized vertices back to original world coordinates."""
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    s = float(scale)
    out_lines: list[str] = []
    with src_obj.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            if raw_line.startswith("v ") or raw_line.startswith("v\t"):
                parts = raw_line.split()
                if len(parts) >= 4:
                    x = float(parts[1]) * s + cx
                    y = float(parts[2]) * s + cy
                    z = float(parts[3]) * s + cz
                    out_lines.append(f"v {x:.8f} {y:.8f} {z:.8f}\n")
                    continue
            out_lines.append(raw_line)
    dst_obj.parent.mkdir(parents=True, exist_ok=True)
    dst_obj.write_text("".join(out_lines), encoding="utf-8")


def run_quadwild_cli(
    input_mesh_path: str | Path,
    out_quad_obj: str | Path,
    out_tri_glb: str | Path,
    *,
    highpoly_mesh_path: str | Path | None = None,
    target_quads: int = 2500,
    sharp_angle: float = 35.0,
    do_remesh: bool = True,
    align_singularities: bool = True,
    max_input_faces: int = 45000,
    quadwild_dir: str | Path = DEFAULT_QUADWILD_DIR,
    timeout_s: float = 1200.0,
    sample_count: int = 100000,
    report_json_path: str | Path | None = None,
) -> QuadRetopoResult:
    """Run QuadWild Bi-MDF v0.0.2 directly via CLI for rigid accessories and hard-surface props."""
    import numpy as np
    import trimesh

    from quadvox.s2_retopo.metrics import extract_vf

    in_p = Path(input_mesh_path)
    hp_p = Path(highpoly_mesh_path) if highpoly_mesh_path is not None else in_p
    quad_p = Path(out_quad_obj)
    glb_p = Path(out_tri_glb)
    qw_root = Path(quadwild_dir).resolve()

    if not in_p.exists():
        raise FileNotFoundError(f"Input mesh not found: {in_p}")
    for bname in ("quadwild", "quad_from_patches"):
        bpath = qw_root / bname
        if not bpath.exists():
            raise FileNotFoundError(f"QuadWild binary missing: {bpath}")
        os.chmod(bpath, 0o755)

    quad_p.parent.mkdir(parents=True, exist_ok=True)
    glb_p.parent.mkdir(parents=True, exist_ok=True)

    workdir = quad_p.parent / f"._{quad_p.stem}_quadwild_{os.getpid()}"
    workdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        # Step 0: Cap input density proportionally to target_quads (~14 tris/quad, bounded by max_input_faces)
        # so QuadWild v0.0.2 finishes in seconds instead of blowing up on 80k-120k+ inputs.
        effective_max_faces = (
            min(int(max_input_faces), max(12000, 14 * int(target_quads)))
            if int(max_input_faces) > 0
            else 0
        )
        verts_in, faces_in = extract_vf(in_p)
        if effective_max_faces > 0 and len(faces_in) > effective_max_faces:
            proxy_glb = workdir / "input_proxy.glb"
            build_clean_proxy_open3d(in_p, proxy_glb, target_faces=effective_max_faces)
            verts_in, faces_in = extract_vf(proxy_glb)

        # Normalize to unit bounding-box diagonal centered at origin so tiny props (e.g. H=0.025)
        # do not suffer numerical isotropic-remesh blow-up inside QuadWild v0.0.2.
        bmin = verts_in.min(axis=0)
        bmax = verts_in.max(axis=0)
        orig_center = 0.5 * (bmin + bmax)
        orig_diag = float(np.linalg.norm(bmax - bmin))
        if not math.isfinite(orig_diag) or orig_diag <= 1e-9:
            orig_diag = 1.0
        verts_norm = (verts_in - orig_center) / orig_diag

        input_obj = workdir / "input.obj"
        with input_obj.open("w", encoding="utf-8") as fh:
            np.savetxt(fh, verts_norm, fmt="v %.8f %.8f %.8f")
            np.savetxt(fh, faces_in + 1, fmt="f %d %d %d")

        prep_cfg = workdir / "prep.txt"
        main_cfg = workdir / "main.txt"

        # Try fast direct field/tracing (do_remesh=0) when input is already a clean <=45k manifold mesh;
        # automatically fall back to do_remesh=1 if step 1 fails or produces no patches.
        remesh_attempts = [False, True] if (not do_remesh or len(faces_in) <= 45000) else [True]
        patches_obj: Path | None = None
        last_err_step1 = ""

        for try_remesh in remesh_attempts:
            for stale in workdir.glob("input*_p0.*"):
                stale.unlink(missing_ok=True)
            prep_cfg.write_text(
                QUADWILD_PREP_TEMPLATE.format(
                    do_remesh=int(bool(try_remesh)),
                    sharp_angle=float(sharp_angle),
                ),
                encoding="utf-8",
            )
            expected_patch = workdir / ("input_rem_p0.obj" if try_remesh else "input_p0.obj")
            cmd1, _ = build_quadwild_cli_commands(
                quadwild_dir=qw_root,
                input_obj=input_obj,
                prep_config=prep_cfg,
                patches_obj=expected_patch,
                main_config=main_cfg,
                qfp_iter=1,
            )
            proc1 = subprocess.run(
                cmd1,
                cwd=str(qw_root),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
            )
            cand_patches = [expected_patch, workdir / "input_rem_p0.obj", workdir / "input_p0.obj"]
            found_p = next((p for p in cand_patches if p.exists() and p.stat().st_size > 0), None)
            if proc1.returncode == 0 and found_p is not None:
                patches_obj = found_p
                break
            last_err_step1 = (
                f"QuadWild step 1 (do_remesh={int(try_remesh)}) failed (code={proc1.returncode}):\n"
                f"CMD: {' '.join(cmd1)}\nSTDOUT:\n{proc1.stdout[-1500:]}\nSTDERR:\n{proc1.stderr[-1500:]}"
            )

        if patches_obj is None:
            raise RuntimeError(last_err_step1 or "QuadWild step 1 produced no patch OBJ.")

        # Step 2: Bi-MDF quantization + quadrangulation with optional 2nd calibration pass.
        scale = _estimate_quadwild_scale_fact(patches_obj, int(target_quads))
        best_obj: Path | None = None
        best_err = float("inf")
        best_got = 0

        for run_idx in range(1, 3):
            for stale in workdir.glob("*quadrangulation*.obj"):
                if best_obj is None or stale.resolve() != best_obj.resolve():
                    stale.unlink(missing_ok=True)
            main_cfg.write_text(
                QUADWILD_MAIN_TEMPLATE.format(
                    alpha=0.02,
                    align=int(bool(align_singularities)),
                    scale=scale,
                ),
                encoding="utf-8",
            )
            _, cmd2 = build_quadwild_cli_commands(
                quadwild_dir=qw_root,
                input_obj=input_obj,
                prep_config=prep_cfg,
                patches_obj=patches_obj,
                main_config=main_cfg,
                qfp_iter=run_idx,
            )
            t_qfp = time.perf_counter()
            proc2 = subprocess.run(
                cmd2,
                cwd=str(qw_root),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
            )
            qfp_elapsed = time.perf_counter() - t_qfp
            cand = find_quadwild_output_obj(workdir, patches_obj, run_idx)
            valid_out, got = is_quadwild_step2_output_valid(
                proc2.returncode, proc2.stdout, proc2.stderr, cand
            )
            if not valid_out or cand is None:
                if best_obj is not None:
                    break
                produced = [p.name for p in sorted(workdir.glob("*.obj"))]
                raise RuntimeError(
                    f"QuadWild step 2 (quad_from_patches) failed (code={proc2.returncode}, objs={produced}):\n"
                    f"CMD: {' '.join(cmd2)}\nSTDOUT:\n{proc2.stdout[-1500:]}\nSTDERR:\n{proc2.stderr[-1500:]}"
                )

            # Save snapshot of this pass so a subsequent pass cannot overwrite it (v0.0.2 reuses filename!).
            snap_cand = workdir / f"_pass_{run_idx}_quad.obj"
            shutil.copy2(cand, snap_cand)

            err = abs(got - int(target_quads))
            if got > 0 and err < best_err:
                best_err = err
                best_obj = snap_cand
                best_got = got

            # Avoid spending a second multi-minute Bi-MDF pass if pass 1 already converged or was heavy.
            if got <= 0 or err <= 0.25 * float(target_quads) or qfp_elapsed > 120.0:
                break
            scale *= math.sqrt(float(got) / float(max(1, int(target_quads))))

        if best_obj is None:
            raise RuntimeError("QuadWild produced no valid quadrangulation OBJ.")

        # Un-normalize output quad vertices back to exact original world space.
        _write_rescaled_obj(best_obj, quad_p, orig_diag, orig_center.tolist())
        qv, _, qtris, _ = parse_obj_polygons(quad_p)
        trimesh.Trimesh(vertices=qv, faces=qtris, process=False).export(str(glb_p))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    metrics = compute_quad_mesh_metrics(
        quad_p,
        highpoly_mesh_path=hp_p,
        sample_count=int(sample_count),
        seed=42,
    )
    result = QuadRetopoResult(
        engine="quadwild_bimdf",
        input_mesh_path=str(in_p),
        highpoly_mesh_path=str(hp_p),
        out_quad_obj=str(quad_p),
        out_tri_glb=str(glb_p),
        target_quads=int(target_quads),
        elapsed_s=time.perf_counter() - t0,
        notes=f"sharp_angle={sharp_angle};got_polys={best_got}",
        **metrics,
    )
    if report_json_path is not None:
        result.write_json(report_json_path)
    return result


# ---------------------------------------------------------------------------
# CLI entrypoint (both host/Colab Python and Blender `--python` subcommands)
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track 3 Quad Retopology CLI (QuadriFlow, ReQuad, QuadWild).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_qfb = sub.add_parser("quadriflow-blender", help="Internal Blender entrypoint for QuadriFlow.")
    p_qfb.add_argument("--input", required=True)
    p_qfb.add_argument("--highpoly", default=None)
    p_qfb.add_argument("--out-quad-obj", required=True)
    p_qfb.add_argument("--out-tri-glb", required=True)
    p_qfb.add_argument("--meta-json", required=True)
    p_qfb.add_argument("--target-faces", type=int, default=35000)
    p_qfb.add_argument("--use-mesh-symmetry", action="store_true")
    p_qfb.add_argument("--use-preserve-sharp", action="store_true")
    p_qfb.add_argument("--use-preserve-boundary", action="store_true")
    p_qfb.add_argument("--smooth-normals", action="store_true")
    p_qfb.add_argument("--seed", type=int, default=0)

    p_rqb = sub.add_parser("requad-blender", help="Internal Blender entrypoint for ReQuad v0.11.3.")
    p_rqb.add_argument("--proxy", required=True)
    p_rqb.add_argument("--highpoly", default=None)
    p_rqb.add_argument("--guide-npz", required=True)
    p_rqb.add_argument("--out-quad-obj", required=True)
    p_rqb.add_argument("--out-tri-glb", required=True)
    p_rqb.add_argument("--meta-json", required=True)
    p_rqb.add_argument("--target-quads", type=int, default=35000)
    p_rqb.add_argument("--sym-x", action="store_true")
    p_rqb.add_argument("--preset", default="ORGANIC")
    p_rqb.add_argument("--adaptive-size", type=float, default=20.0)
    p_rqb.add_argument("--relax-iterations", type=int, default=30)

    p_qf = sub.add_parser("quadriflow", help="Run QuadriFlow baseline.")
    p_qf.add_argument("--input", required=True)
    p_qf.add_argument("--highpoly", default=None)
    p_qf.add_argument("--labels-npz", default=None)
    p_qf.add_argument("--labels-source-mesh", default=None)
    p_qf.add_argument("--out-quad-obj", required=True)
    p_qf.add_argument("--out-tri-glb", required=True)
    p_qf.add_argument("--target-faces", type=int, default=35000)
    p_qf.add_argument("--symmetry", action="store_true", help="Enable Layer-2 X-axis symmetry.")
    p_qf.add_argument("--no-symmetry", action="store_true", help="Explicitly disable X-axis symmetry.")
    p_qf.add_argument("--blender-bin", default=str(DEFAULT_BLENDER_BIN))
    p_qf.add_argument("--report-json", default=None)

    p_rq = sub.add_parser("requad", help="Run ReQuad quad remesh (with optional semantic-density & X=0 symmetry).")
    p_rq.add_argument("--proxy", required=True)
    p_rq.add_argument("--labels-npz", default=None)
    p_rq.add_argument("--highpoly", default=None)
    p_rq.add_argument("--labels-source-mesh", default=None)
    p_rq.add_argument("--out-quad-obj", required=True)
    p_rq.add_argument("--out-tri-glb", required=True)
    p_rq.add_argument("--target-quads", type=int, default=35000)
    p_rq.add_argument("--sym-x", action="store_true", help="Enable Layer-2 ReQuad X=0 mirror symmetry.")
    p_rq.add_argument("--no-sym-x", action="store_true", help="Explicitly disable ReQuad X=0 symmetry.")
    p_rq.add_argument("--preset", default="ORGANIC")
    p_rq.add_argument("--adaptive-size", type=float, default=20.0)
    p_rq.add_argument("--relax-iterations", type=int, default=30)
    p_rq.add_argument("--requad-dir", default=str(DEFAULT_REQUAD_DIR))
    p_rq.add_argument("--blender-bin", default=str(DEFAULT_BLENDER_BIN))
    p_rq.add_argument("--report-json", default=None)

    p_qw = sub.add_parser("quadwild", help="Run QuadWild Bi-MDF directly via CLI.")
    p_qw.add_argument("--input", required=True)
    p_qw.add_argument("--highpoly", default=None)
    p_qw.add_argument("--out-quad-obj", required=True)
    p_qw.add_argument("--out-tri-glb", required=True)
    p_qw.add_argument("--target-quads", type=int, default=2500)
    p_qw.add_argument("--sharp-angle", type=float, default=35.0)
    p_qw.add_argument("--max-input-faces", type=int, default=120000)
    p_qw.add_argument("--quadwild-dir", default=str(DEFAULT_QUADWILD_DIR))
    p_qw.add_argument("--report-json", default=None)

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
    if args.command == "quadriflow-blender":
        _run_quadriflow_in_blender(
            input_mesh_path=Path(args.input),
            highpoly_mesh_path=Path(args.highpoly) if args.highpoly else None,
            out_quad_obj=Path(args.out_quad_obj),
            out_tri_glb=Path(args.out_tri_glb),
            target_faces=int(args.target_faces),
            use_mesh_symmetry=bool(args.use_mesh_symmetry),
            use_preserve_sharp=bool(args.use_preserve_sharp),
            use_preserve_boundary=bool(args.use_preserve_boundary),
            smooth_normals=bool(args.smooth_normals),
            seed=int(args.seed),
            meta_json_path=Path(args.meta_json),
        )
        return 0

    if args.command == "requad-blender":
        _run_requad_in_blender(
            proxy_mesh_path=Path(args.proxy),
            highpoly_mesh_path=Path(args.highpoly) if args.highpoly else None,
            guide_npz_path=Path(args.guide_npz),
            out_quad_obj=Path(args.out_quad_obj),
            out_tri_glb=Path(args.out_tri_glb),
            target_quads=int(args.target_quads),
            sym_x=bool(args.sym_x),
            preset=str(args.preset),
            adaptive_size=float(args.adaptive_size),
            relax_iterations=int(args.relax_iterations),
            meta_json_path=Path(args.meta_json),
        )
        return 0

    if args.command == "quadriflow":
        res = run_quadriflow_blender(
            input_mesh_path=args.input,
            highpoly_mesh_path=args.highpoly,
            labels_npz_path=args.labels_npz,
            labels_source_mesh_path=args.labels_source_mesh,
            out_quad_obj=args.out_quad_obj,
            out_tri_glb=args.out_tri_glb,
            target_faces=int(args.target_faces),
            use_mesh_symmetry=bool(args.symmetry) and not bool(args.no_symmetry),
            blender_bin=args.blender_bin,
            report_json_path=args.report_json,
        )
        print(res.to_json(indent=2))
        return 0

    if args.command == "requad":
        res = run_requad_semantic_blender(
            proxy_mesh_path=args.proxy,
            labels_npz_path=args.labels_npz,
            highpoly_mesh_path=args.highpoly,
            labels_source_mesh_path=args.labels_source_mesh,
            out_quad_obj=args.out_quad_obj,
            out_tri_glb=args.out_tri_glb,
            target_quads=int(args.target_quads),
            sym_x=bool(args.sym_x) and not bool(args.no_sym_x),
            preset=str(args.preset),
            adaptive_size=float(args.adaptive_size),
            relax_iterations=int(args.relax_iterations),
            requad_dir=args.requad_dir,
            blender_bin=args.blender_bin,
            report_json_path=args.report_json,
        )
        print(res.to_json(indent=2))
        return 0

    if args.command == "quadwild":
        res = run_quadwild_cli(
            input_mesh_path=args.input,
            highpoly_mesh_path=args.highpoly,
            out_quad_obj=args.out_quad_obj,
            out_tri_glb=args.out_tri_glb,
            target_quads=int(args.target_quads),
            sharp_angle=float(args.sharp_angle),
            max_input_faces=int(args.max_input_faces),
            quadwild_dir=args.quadwild_dir,
            report_json_path=args.report_json,
        )
        print(res.to_json(indent=2))
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
