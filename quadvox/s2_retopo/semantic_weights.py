"""3D semantic proxy construction, chunked KDTree back-projection, and Laplacian weight smoothing.

Orthogonal design (per S2 experiment architecture):
- Semantic inference runs ONLY on a lightweight uniform QEM proxy (~150k-200k faces)
  to avoid OOM on 3.13M / 10.51M highpoly assets.
- Labels and continuous features are projected back onto EVERY vertex of the original
  cleaned highpoly mesh via chunked `scipy.spatial.cKDTree` without modifying topology
  or splitting parts.
- Weights are diffused via sparse CSR vertex-adjacency Laplacian smoothing (min >= 1.0)
  before passing to PyMeshLab `qualityweight=True`.
"""

from __future__ import annotations

import colorsys
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from quadvox.s2_retopo.metrics import compute_triangle_areas, extract_vf


def build_semantic_proxy(
    input_mesh_or_path: str | Path | trimesh.Trimesh,
    out_proxy_path: str | Path | None = None,
    *,
    target_faces: int = 150000,
    engine: str = "pyfqmr",
    preserve_border: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a uniform QEM proxy mesh purely for 3D semantic inference (never final retopo)."""
    from quadvox.s2_retopo.qem import decimate_pyfqmr, decimate_pymeshlab

    t0 = time.perf_counter()
    verts, faces = extract_vf(input_mesh_or_path)
    orig_faces = int(len(faces))

    if orig_faces <= int(target_faces) or orig_faces == 0:
        v_proxy = np.ascontiguousarray(verts, dtype=np.float64)
        f_proxy = np.ascontiguousarray(faces, dtype=np.int64)
    elif engine.strip().lower() == "pymeshlab":
        v_proxy, f_proxy = decimate_pymeshlab(
            verts,
            faces,
            target_faces=int(target_faces),
            preserve_boundary=preserve_border,
        )
    else:
        v_proxy, f_proxy = decimate_pyfqmr(
            verts,
            faces,
            target_faces=int(target_faces),
            preserve_border=preserve_border,
        )

    if out_proxy_path is not None:
        out_p = Path(out_proxy_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=v_proxy, faces=f_proxy, process=False).export(str(out_p))

    info = {
        "proxy_engine": engine,
        "source_faces": orig_faces,
        "proxy_vertices": int(len(v_proxy)),
        "proxy_faces": int(len(f_proxy)),
        "target_faces": int(target_faces),
        "elapsed_s": time.perf_counter() - t0,
        "out_proxy_path": str(out_proxy_path) if out_proxy_path is not None else None,
    }
    return v_proxy, f_proxy, info


def face_values_to_vertex_values(
    num_verts: int,
    faces: np.ndarray,
    face_values: np.ndarray,
    *,
    mode: str = "majority",
) -> np.ndarray:
    """Convert per-face labels or scalars on a mesh to per-vertex values."""
    vals = np.asarray(face_values)
    if len(vals) != len(faces):
        raise ValueError(f"face_values len {len(vals)} != face count {len(faces)}")
    if num_verts == 0 or len(faces) == 0:
        return np.zeros((num_verts,), dtype=vals.dtype)

    flat_verts = faces.reshape(-1)
    repeated = np.repeat(vals, 3, axis=0)

    if mode == "mean":
        if vals.ndim == 1:
            sums = np.bincount(flat_verts, weights=repeated.astype(np.float64), minlength=num_verts)
            counts = np.bincount(flat_verts, minlength=num_verts)
            return sums / np.maximum(counts, 1)
        out = np.zeros((num_verts, vals.shape[1]), dtype=np.float64)
        counts = np.bincount(flat_verts, minlength=num_verts)
        np.add.at(out, flat_verts, repeated)
        return out / np.maximum(counts[:, None], 1)

    # Majority vote for discrete integer cluster IDs.
    ids = repeated.astype(np.int64)
    order = np.lexsort((ids, flat_verts))
    sv = flat_verts[order]
    si = ids[order]

    # Group by (vertex, part_id) and count occurrences.
    diff = np.empty(len(sv), dtype=bool)
    diff[0] = True
    if len(sv) > 1:
        diff[1:] = (sv[1:] != sv[:-1]) | (si[1:] != si[:-1])
    starts = np.flatnonzero(diff)
    counts = np.diff(np.append(starts, len(sv)))
    u_v = sv[starts]
    u_i = si[starts]

    # Pick highest-count part_id per vertex.
    best_order = np.lexsort((-counts, u_v))
    bv = u_v[best_order]
    bi = u_i[best_order]
    first_v = np.empty(len(bv), dtype=bool)
    first_v[0] = True
    if len(bv) > 1:
        first_v[1:] = bv[1:] != bv[:-1]

    out = np.full((num_verts,), -1, dtype=np.int32)
    out[bv[first_v]] = bi[first_v].astype(np.int32)
    return out


def project_proxy_labels_and_features_to_highpoly(
    high_verts: np.ndarray,
    proxy_points: np.ndarray,
    *,
    proxy_part_ids: np.ndarray | None = None,
    proxy_vertex_scores: np.ndarray | None = None,
    proxy_vertex_features: np.ndarray | None = None,
    chunk_size: int = 250000,
) -> dict[str, np.ndarray]:
    """Back-project proxy labels, scores, or features onto highpoly vertices via chunked KDTree.

    Chunks queries so 5.4M+ vertex meshes stay well within bounded RAM.
    """
    hv = np.asarray(high_verts, dtype=np.float64)
    pv = np.asarray(proxy_points, dtype=np.float64)
    n_high = len(hv)
    if n_high == 0:
        res: dict[str, np.ndarray] = {
            "nn_indices": np.zeros((0,), dtype=np.int64),
            "nn_distances": np.zeros((0,), dtype=np.float32),
        }
        if proxy_part_ids is not None:
            res["high_part_ids"] = np.zeros((0,), dtype=np.int32)
        if proxy_vertex_scores is not None:
            res["high_scores"] = np.zeros((0,), dtype=np.float32)
        if proxy_vertex_features is not None:
            d = np.asarray(proxy_vertex_features).shape[1]
            res["high_features"] = np.zeros((0, d), dtype=np.float32)
        return res

    if len(pv) == 0:
        raise ValueError("proxy_points cannot be empty when high_verts is non-empty.")

    if proxy_part_ids is not None and len(proxy_part_ids) != len(pv):
        raise ValueError(f"proxy_part_ids len {len(proxy_part_ids)} != proxy_points len {len(pv)}")
    if proxy_vertex_scores is not None and len(proxy_vertex_scores) != len(pv):
        raise ValueError(
            f"proxy_vertex_scores len {len(proxy_vertex_scores)} != proxy_points len {len(pv)}"
        )
    if proxy_vertex_features is not None and len(proxy_vertex_features) != len(pv):
        raise ValueError(
            f"proxy_vertex_features len {len(proxy_vertex_features)} != proxy_points len {len(pv)}"
        )

    tree = cKDTree(pv)
    nn_indices = np.empty(n_high, dtype=np.int64)
    nn_distances = np.empty(n_high, dtype=np.float32)

    step = max(1, int(chunk_size))
    for start in range(0, n_high, step):
        end = min(n_high, start + step)
        dists, idxs = tree.query(hv[start:end], workers=-1)
        nn_indices[start:end] = idxs
        nn_distances[start:end] = dists.astype(np.float32)

    out: dict[str, np.ndarray] = {
        "nn_indices": nn_indices,
        "nn_distances": nn_distances,
    }
    if proxy_part_ids is not None:
        pids = np.asarray(proxy_part_ids, dtype=np.int32)
        out["high_part_ids"] = pids[nn_indices]
    if proxy_vertex_scores is not None:
        scores = np.asarray(proxy_vertex_scores, dtype=np.float32)
        out["high_scores"] = scores[nn_indices]
    if proxy_vertex_features is not None:
        feats = np.asarray(proxy_vertex_features, dtype=np.float32)
        out["high_features"] = feats[nn_indices]
    return out


def _build_vertex_adjacency_csr(num_verts: int, faces: np.ndarray) -> csr_matrix:
    """Build unweighted symmetric 1-ring vertex-adjacency CSR matrix (`A_ij in {0, 1}`)."""
    if num_verts == 0 or len(faces) == 0:
        return csr_matrix((num_verts, num_verts), dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)
    e0 = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    e1 = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    valid = e0 != e1
    e0 = e0[valid]
    e1 = e1[valid]
    rows = np.concatenate([e0, e1])
    cols = np.concatenate([e1, e0])
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(num_verts, num_verts), dtype=np.float32)
    adj.sum_duplicates()
    adj.data[:] = 1.0
    return adj


def smooth_vertex_weights(
    verts: np.ndarray,
    faces: np.ndarray,
    weights: np.ndarray,
    *,
    iterations: int = 2,
    alpha: float = 0.5,
    min_weight: float = 1.0,
) -> np.ndarray:
    """Diffuse per-vertex weights along mesh edges via sparse CSR Laplacian smoothing."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    n_verts = len(verts)
    if len(w) != n_verts:
        raise ValueError(f"weights length {len(w)} != vertex count {n_verts}")
    w = np.where(np.isfinite(w), w, float(min_weight))
    w = np.maximum(w, float(min_weight))
    if iterations <= 0 or alpha <= 0.0 or len(faces) == 0 or n_verts == 0:
        return w

    adj = _build_vertex_adjacency_csr(n_verts, faces)
    deg = np.maximum(np.asarray(adj.sum(axis=1), dtype=np.float64).reshape(-1), 1.0)
    a = float(np.clip(alpha, 0.0, 1.0))

    cur = w.astype(np.float32, copy=True)
    deg32 = deg.astype(np.float32)
    for _ in range(int(iterations)):
        nbr_mean = (adj @ cur) / deg32
        cur = (1.0 - a) * cur + a * nbr_mean
        cur = np.maximum(cur, float(min_weight))
    return cur.astype(np.float64)


def compute_cluster_bbox_focus_weights(
    verts: np.ndarray,
    faces: np.ndarray,
    part_ids: np.ndarray,
    *,
    focus_weight_multiplier: float = 6.0,
    head_y_thresh: float = 0.72,
    min_centroid_y: float = 0.68,
    min_head_overlap: float = 0.50,
    max_bbox_height: float = 0.55,
    max_cluster_vert_ratio: float = 0.60,
    smooth_iters: int = 2,
    smooth_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Minimal 2-tier semantic focus weighting on whole 3D head/facial cluster(s).

    Selects complete semantic clusters whose normalized `y_up_z_front` centroid and
    vertical bounding box lie in the head/facial zone, boosts ONLY those clusters to
    `focus_weight_multiplier` (rest `1.0`), then applies sparse Laplacian smoothing.

    Returns:
        weights: `(V,)` float64 smoothed per-vertex weights (`min >= 1.0`)
        focus_mask: `(V,)` bool pre-smoothing binary mask of selected cluster vertices
        diagnostics: dict with cluster inventory, selected IDs, ratios, and fallback status
    """
    v = np.asarray(verts, dtype=np.float64)
    pids = np.asarray(part_ids, dtype=np.int64).reshape(-1)
    n_verts = len(v)
    if len(pids) != n_verts:
        raise ValueError(f"part_ids length {len(pids)} != vertex count {n_verts}")
    if n_verts == 0:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=bool),
            {"part_cluster_count": 0, "selected_cluster_ids": [], "focus_vertex_ratio": 0.0},
        )

    finite = np.isfinite(v[:, 1])
    y_vals = v[finite, 1] if np.any(finite) else np.zeros(1, dtype=np.float64)
    y_min = float(np.min(y_vals))
    y_max = float(np.max(y_vals))
    height = max(y_max - y_min, 1e-12)
    y_norm = (v[:, 1] - y_min) / height

    unique_ids = [int(c) for c in np.unique(pids) if int(c) >= 0]
    cluster_stats: list[dict[str, Any]] = []
    selected_ids: list[int] = []

    for cid in unique_ids:
        mask = pids == cid
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        v_ratio = float(cnt / n_verts)
        yn_c = y_norm[mask]
        c_y = float(np.mean(yn_c))
        b_min_y = float(np.min(yn_c))
        b_max_y = float(np.max(yn_c))
        b_h = b_max_y - b_min_y
        head_overlap = float(np.mean(yn_c > float(head_y_thresh)))
        is_selected = (
            c_y >= float(min_centroid_y)
            and head_overlap >= float(min_head_overlap)
            and b_h <= float(max_bbox_height)
            and v_ratio <= float(max_cluster_vert_ratio)
        )
        if is_selected:
            selected_ids.append(cid)
        cluster_stats.append(
            {
                "cluster_id": cid,
                "vertex_count": cnt,
                "vertex_ratio": v_ratio,
                "centroid_y_norm": c_y,
                "bbox_min_y_norm": b_min_y,
                "bbox_max_y_norm": b_max_y,
                "bbox_height_norm": b_h,
                "head_overlap_ratio": head_overlap,
                "selected": is_selected,
            }
        )

    fallback_used = False
    if not selected_ids and cluster_stats:
        fallback_used = True
        candidates = [
            st
            for st in cluster_stats
            if st["head_overlap_ratio"] > 0.25 and st["vertex_ratio"] <= float(max_cluster_vert_ratio)
        ]
        if not candidates:
            candidates = cluster_stats
        best = max(candidates, key=lambda st: (st["head_overlap_ratio"], st["centroid_y_norm"]))
        selected_ids = [int(best["cluster_id"])]
        for st in cluster_stats:
            st["selected"] = int(st["cluster_id"]) in selected_ids

    focus_mask = np.isin(pids, np.asarray(selected_ids, dtype=np.int64)) if selected_ids else np.zeros(n_verts, dtype=bool)
    raw_weights = np.where(focus_mask, max(1.0, float(focus_weight_multiplier)), 1.0)
    weights = smooth_vertex_weights(
        v,
        faces,
        raw_weights,
        iterations=smooth_iters,
        alpha=smooth_alpha,
        min_weight=1.0,
    )

    diagnostics = {
        "mode": "cluster_bbox_focus",
        "part_cluster_count": len(unique_ids),
        "selected_cluster_ids": selected_ids,
        "focus_vertex_ratio": float(np.mean(focus_mask)),
        "weighted_vertex_ratio": float(np.mean(weights > 1.05)),
        "selection_fallback_used": fallback_used,
        "focus_weight_multiplier": float(focus_weight_multiplier),
        "smooth_iters": int(smooth_iters),
        "smooth_alpha": float(smooth_alpha),
        "clusters": cluster_stats,
    }
    return weights, focus_mask, diagnostics


def _proxy_discrete_mask_fallback_scores(
    proxy_verts: np.ndarray,
    proxy_faces: np.ndarray,
    proxy_part_ids: np.ndarray,
) -> np.ndarray:
    """Fallback proxy vertex score when a semantic backend only exposes discrete masks.

    Combines:
    1. Inverse-part-area score (`1 / sqrt(rel_area_c)`), giving higher weight to
       small intricate semantic parts (eyes, ears, hair accessories, hands).
    2. Semantic boundary proximity (diffusion from edges where `part_id[u] != part_id[v]`).
    """
    n_verts = len(proxy_verts)
    pids = np.asarray(proxy_part_ids, dtype=np.int64).reshape(-1)
    if n_verts == 0 or len(proxy_faces) == 0:
        return np.zeros((n_verts,), dtype=np.float32)

    areas = compute_triangle_areas(proxy_verts, proxy_faces)
    vert_area = np.zeros(n_verts, dtype=np.float64)
    third_area = areas / 3.0
    for col in range(3):
        np.add.at(vert_area, proxy_faces[:, col], third_area)

    valid = pids >= 0
    unique_ids = np.unique(pids[valid])
    part_area_map: dict[int, float] = {}
    total_area = max(float(np.sum(vert_area)), 1e-12)
    for cid in unique_ids:
        part_area_map[int(cid)] = max(float(np.sum(vert_area[pids == cid])) / total_area, 5e-4)

    inv_area = np.ones(n_verts, dtype=np.float64)
    for cid, rel_a in part_area_map.items():
        inv_area[pids == cid] = 1.0 / math.sqrt(rel_a)
    lo = float(np.min(inv_area))
    hi = float(np.max(inv_area))
    if hi > lo + 1e-12:
        norm_inv_area = (inv_area - lo) / (hi - lo)
    else:
        norm_inv_area = np.zeros(n_verts, dtype=np.float64)

    # Boundary proximity via 2-step diffusion from inter-part edges.
    f = np.asarray(proxy_faces, dtype=np.int64)
    e0 = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    e1 = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    bnd_edges = pids[e0] != pids[e1]
    bnd_seed = np.zeros(n_verts, dtype=np.float64)
    if np.any(bnd_edges):
        bnd_verts = np.unique(np.concatenate([e0[bnd_edges], e1[bnd_edges]]))
        bnd_seed[bnd_verts] = 1.0
    adj = _build_vertex_adjacency_csr(n_verts, f)
    deg = np.maximum(np.asarray(adj.sum(axis=1), dtype=np.float64).reshape(-1), 1.0)
    bnd_prox = bnd_seed.astype(np.float32)
    deg32 = deg.astype(np.float32)
    for _ in range(2):
        bnd_prox = np.maximum(bnd_seed.astype(np.float32), 0.75 * ((adj @ bnd_prox) / deg32))

    combined = 0.5 * norm_inv_area + 0.5 * bnd_prox.astype(np.float64)
    return combined.astype(np.float32)


def compute_proxy_feature_grad_scores(
    proxy_verts: np.ndarray,
    proxy_faces: np.ndarray,
    proxy_features: np.ndarray,
) -> np.ndarray:
    """Compute per-vertex spatial feature gradient magnitude on a proxy mesh.

    Evaluates `||f[v] - f[u]|| / max(||x[v] - x[u]||, 0.1 * median_edge_len)`
    across unique undirected proxy edges and averages back to incident vertices.
    Returns `(V_proxy,)` float32 gradient magnitude scores.
    """
    pv = np.asarray(proxy_verts, dtype=np.float64)
    pf = np.asarray(proxy_faces, dtype=np.int64)
    feats = np.asarray(proxy_features, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats[:, None]
    n_proxy = len(pv)
    if len(feats) != n_proxy:
        raise ValueError(f"proxy_features length {len(feats)} != proxy_verts len {n_proxy}")
    if len(pf) == 0 or n_proxy == 0:
        return np.zeros(n_proxy, dtype=np.float32)

    e0 = np.concatenate([pf[:, 0], pf[:, 1], pf[:, 2]])
    e1 = np.concatenate([pf[:, 1], pf[:, 2], pf[:, 0]])
    u = np.minimum(e0, e1)
    v = np.maximum(e0, e1)
    valid = u != v
    u = u[valid]
    v = v[valid]
    if len(u) == 0:
        return np.zeros(n_proxy, dtype=np.float32)

    order = np.lexsort((v, u))
    u_s = u[order]
    v_s = v[order]
    uniq = np.empty(len(u_s), dtype=bool)
    uniq[0] = True
    if len(u_s) > 1:
        uniq[1:] = (u_s[1:] != u_s[:-1]) | (v_s[1:] != v_s[:-1])
    eu = u_s[uniq]
    ev = v_s[uniq]

    dx = np.linalg.norm(pv[ev] - pv[eu], axis=1)
    med_len = float(np.median(dx)) if len(dx) > 0 else 1.0
    edge_eps = max(1e-8, 0.1 * med_len)
    df = np.linalg.norm(feats[ev] - feats[eu], axis=1)
    grad = df / np.maximum(dx, edge_eps)

    sum_g = np.zeros(n_proxy, dtype=np.float64)
    cnt_g = np.zeros(n_proxy, dtype=np.int64)
    np.add.at(sum_g, eu, grad)
    np.add.at(sum_g, ev, grad)
    np.add.at(cnt_g, eu, 1)
    np.add.at(cnt_g, ev, 1)
    return (sum_g / np.maximum(cnt_g, 1)).astype(np.float32)


def compute_feature_grad_soft_weights(
    high_verts: np.ndarray,
    high_faces: np.ndarray,
    proxy_verts: np.ndarray,
    proxy_faces: np.ndarray,
    *,
    proxy_features: np.ndarray | None = None,
    proxy_vertex_scores: np.ndarray | None = None,
    proxy_part_ids: np.ndarray | None = None,
    focus_weight_multiplier: float = 6.0,
    p_low: float = 50.0,
    p_high: float = 95.0,
    smooth_iters: int = 2,
    smooth_alpha: float = 0.5,
    chunk_size: int = 250000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute continuous soft QEM weights from 3D feature gradients (or discrete-mask fallback).

    When precomputed `proxy_vertex_scores` `(V_proxy,)` or continuous `proxy_features`
    `(V_proxy, D)` are provided:
    1. Obtains per-proxy-vertex spatial gradient magnitude scores via
       `compute_proxy_feature_grad_scores(...)` (or directly uses `proxy_vertex_scores`).
    2. Projects proxy vertex gradient scores to `high_verts` via chunked `cKDTree`.
    3. Maps `[p50 .. p95]` percentiles to `[1.0 .. focus_weight_multiplier]`.
    4. Applies sparse CSR Laplacian smoothing (`min >= 1.0`).

    When only discrete `proxy_part_ids` are provided, uses the documented inverse-part-area +
    semantic-boundary-proximity fallback.
    """
    hv = np.asarray(high_verts, dtype=np.float64)
    pv = np.asarray(proxy_verts, dtype=np.float64)
    pf = np.asarray(proxy_faces, dtype=np.int64)
    n_high = len(hv)
    n_proxy = len(pv)

    if n_high == 0:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float32),
            {"mode": "empty", "part_cluster_count": 0, "focus_vertex_ratio": 0.0},
        )

    if proxy_vertex_scores is not None:
        scores_in = np.asarray(proxy_vertex_scores, dtype=np.float32)
        if scores_in.ndim == 2 and scores_in.shape[1] == 1:
            scores_in = scores_in[:, 0]
        if scores_in.ndim != 1 or len(scores_in) != n_proxy:
            raise ValueError(
                f"proxy_vertex_scores shape {scores_in.shape} does not match proxy_verts len {n_proxy}"
            )
        proxy_scores = np.where(np.isfinite(scores_in), scores_in, 0.0).astype(np.float32)
        score_mode = "continuous_feature_gradient"
    elif proxy_features is not None:
        proxy_scores = compute_proxy_feature_grad_scores(pv, pf, proxy_features)
        score_mode = "continuous_feature_gradient"
    elif proxy_part_ids is not None:
        proxy_scores = _proxy_discrete_mask_fallback_scores(pv, pf, proxy_part_ids)
        score_mode = "discrete_mask_fallback"
    else:
        raise ValueError("Provide proxy_vertex_scores, proxy_features, or proxy_part_ids.")

    proj = project_proxy_labels_and_features_to_highpoly(
        hv,
        pv,
        proxy_part_ids=proxy_part_ids,
        proxy_vertex_scores=proxy_scores,
        chunk_size=chunk_size,
    )
    high_scores = proj["high_scores"].astype(np.float64)
    lo_val = float(np.percentile(high_scores, float(p_low)))
    hi_val = float(np.percentile(high_scores, float(p_high)))

    if hi_val <= lo_val + 1e-12:
        soft_score = np.zeros(n_high, dtype=np.float64)
    else:
        soft_score = np.clip((high_scores - lo_val) / (hi_val - lo_val), 0.0, 1.0)

    mult = max(1.0, float(focus_weight_multiplier))
    raw_weights = 1.0 + (mult - 1.0) * soft_score
    weights = smooth_vertex_weights(
        hv,
        high_faces,
        raw_weights,
        iterations=smooth_iters,
        alpha=smooth_alpha,
        min_weight=1.0,
    )

    n_clusters = 0
    if proxy_part_ids is not None:
        n_clusters = int(len([c for c in np.unique(proxy_part_ids) if int(c) >= 0]))

    diagnostics = {
        "mode": score_mode,
        "part_cluster_count": n_clusters,
        "p_low": float(p_low),
        "p_high": float(p_high),
        "score_p_low": lo_val,
        "score_p_high": hi_val,
        "focus_weight_multiplier": mult,
        "focus_vertex_ratio": float(np.mean(soft_score > 0.25)),
        "weighted_vertex_ratio": float(np.mean(weights > 1.05)),
        "smooth_iters": int(smooth_iters),
        "smooth_alpha": float(smooth_alpha),
    }
    return weights, soft_score.astype(np.float32), diagnostics


def save_weights_npz(
    path: str | Path,
    *,
    weights: np.ndarray,
    part_ids: np.ndarray | None = None,
    soft_score: np.ndarray | None = None,
    focus_mask: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Persist per-vertex weights, cluster IDs, scores, and metadata to compressed `.npz`."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "weights": np.asarray(weights, dtype=np.float32),
        "metadata_json": np.array(json.dumps(dict(metadata or {}), ensure_ascii=False)),
    }
    if part_ids is not None:
        arrays["part_ids"] = np.asarray(part_ids, dtype=np.int32)
    if soft_score is not None:
        arrays["soft_score"] = np.asarray(soft_score, dtype=np.float32)
    if focus_mask is not None:
        arrays["focus_mask"] = np.asarray(focus_mask, dtype=np.uint8)
    np.savez_compressed(str(out_path), **arrays)
    return out_path


def load_weights_npz(path: str | Path) -> dict[str, Any]:
    """Load saved per-vertex weights `.npz` and decode metadata JSON."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Weights file not found: {src}")
    payload: dict[str, Any] = {}
    with np.load(str(src), allow_pickle=False) as data:
        for key in data.files:
            if key == "metadata_json":
                raw_meta = str(data[key].item())
                meta = json.loads(raw_meta) if raw_meta else {}
                payload["metadata"] = meta
                for mk, mv in meta.items():
                    payload.setdefault(mk, mv)
            elif key == "weights":
                payload[key] = np.asarray(data[key], dtype=np.float64)
            else:
                payload[key] = np.asarray(data[key])
    return payload


def generate_cluster_palette(cluster_ids: Sequence[int]) -> dict[int, tuple[int, int, int, int]]:
    """Generate deterministic high-contrast RGBA uint8 colors per cluster ID."""
    palette: dict[int, tuple[int, int, int, int]] = {-1: (128, 128, 128, 255)}
    sorted_ids = sorted({int(c) for c in cluster_ids if int(c) >= 0})
    golden = 0.6180339887498949
    for idx, cid in enumerate(sorted_ids):
        hue = (idx * golden) % 1.0
        sat = 0.72 if (idx % 2 == 0) else 0.88
        val = 0.92 if (idx % 3 != 2) else 0.78
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette[cid] = (
            int(round(r * 255)),
            int(round(g * 255)),
            int(round(b * 255)),
            255,
        )
    return palette


def export_colored_cluster_preview(
    verts: np.ndarray,
    faces: np.ndarray,
    part_ids: np.ndarray,
    out_glb_path: str | Path,
    *,
    focus_cluster_ids: Sequence[int] | None = None,
    summary_json_path: str | Path | None = None,
) -> Path:
    """Export a vertex-colored `.glb` mesh for Blender `semantic_clusters` visual review."""
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    pids = np.asarray(part_ids, dtype=np.int64).reshape(-1)
    if len(pids) != len(v):
        raise ValueError(f"part_ids length {len(pids)} != vertex count {len(v)}")

    unique_ids = sorted({int(c) for c in pids})
    palette = generate_cluster_palette(unique_ids)
    focus_set = {int(c) for c in (focus_cluster_ids or ())}

    rgba = np.zeros((len(v), 4), dtype=np.uint8)
    for cid in unique_ids:
        color = palette.get(cid, (128, 128, 128, 255))
        if focus_set and cid in focus_set:
            # Bright warm gold/coral boost on selected head/focus clusters for instant clarity.
            color = (255, 195, 45, 255) if len(focus_set) == 1 else color
        rgba[pids == cid] = np.asarray(color, dtype=np.uint8)

    out_path = Path(out_glb_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)
    mesh.export(str(out_path))

    if summary_json_path is not None:
        sum_path = Path(summary_json_path)
        sum_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "out_glb_path": str(out_path),
            "vertex_count": int(len(v)),
            "face_count": int(len(f)),
            "cluster_count": len([c for c in unique_ids if c >= 0]),
            "focus_cluster_ids": sorted(focus_set),
            "palette": {str(k): list(val) for k, val in palette.items()},
        }
        sum_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return out_path
