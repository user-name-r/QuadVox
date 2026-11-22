"""Deterministic topology, geometry fidelity, and mesh quality metrics for S2 experiments.

ALWAYS loads GLB/mesh assets with `trimesh.load(path, force="mesh", process=False)`
so raw vertices/edges are inspected honestly without silent welding or merging.
For `.obj` inputs (e.g. AutoRemesher quad meshes), polygon face lines are also parsed
explicitly to preserve `quad_count`, `tri_count`, and `equiv_tri_count`.

Verified reference:
- S1 `s1_backend/repair/metrics.py` (exact edge, corner-fan, and boundary-loop algorithms).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from quadvox.s2_retopo.contracts import MeshMetrics, RetopoMetrics


@dataclass(frozen=True)
class BoundaryLoop:
    """Single closed boundary edge cycle traced on a triangle mesh."""

    vertices: list[int]
    directed_edges: list[tuple[int, int]]
    perimeter: float
    is_pinhole: bool


@dataclass(frozen=True)
class PolygonCounts:
    """Raw polygon decomposition counts extracted from an OBJ or triangle mesh."""

    quad_count: int
    tri_count: int
    ngon_count: int
    equiv_tri_count: int


def parse_obj_polygon_counts(path: str | Path) -> PolygonCounts:
    """Parse `.obj` face declarations (`f ...`) to count quads, triangles, and ngons."""
    src = Path(path)
    quad_count = 0
    tri_count = 0
    ngon_count = 0
    equiv_tri_count = 0
    with src.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("f ") or line.startswith("f\t"):
                tokens = line.split()[1:]
                n_verts = len(tokens)
                if n_verts == 3:
                    tri_count += 1
                    equiv_tri_count += 1
                elif n_verts == 4:
                    quad_count += 1
                    equiv_tri_count += 2
                elif n_verts > 4:
                    ngon_count += 1
                    equiv_tri_count += n_verts - 2
    return PolygonCounts(
        quad_count=quad_count,
        tri_count=tri_count,
        ngon_count=ngon_count,
        equiv_tri_count=equiv_tri_count,
    )


def load_mesh_raw(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh without implicit vertex welding, face deduplication, or reordering."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Mesh file not found: {src}")

    loaded = trimesh.load(str(src), force="mesh", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    if isinstance(loaded, trimesh.Scene):
        verts_parts: list[np.ndarray] = []
        faces_parts: list[np.ndarray] = []
        offset = 0
        for node_name in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph[node_name]
            geom = loaded.geometry.get(geom_name)
            if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
                continue
            v = np.asarray(geom.vertices, dtype=np.float64)
            f = np.asarray(geom.faces, dtype=np.int64)
            if transform is not None:
                v = trimesh.transform_points(v, transform)
            verts_parts.append(v)
            faces_parts.append(f + offset)
            offset += len(v)
        if not verts_parts:
            return trimesh.Trimesh(
                vertices=np.zeros((0, 3), dtype=np.float64),
                faces=np.zeros((0, 3), dtype=np.int64),
                process=False,
            )
        return trimesh.Trimesh(
            vertices=np.vstack(verts_parts),
            faces=np.vstack(faces_parts),
            process=False,
        )

    raise TypeError(f"Unsupported mesh type loaded from {src}: {type(loaded).__name__}")


def extract_vf(mesh_or_path: str | Path | trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Extract float64 (V, 3) vertices and int64 (F, 3) triangle faces without processing."""
    mesh = load_mesh_raw(mesh_or_path) if isinstance(mesh_or_path, (str, Path)) else mesh_or_path
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[0] == 0:
        verts = np.zeros((0, 3), dtype=np.float64)
    if faces.ndim != 2 or faces.shape[0] == 0:
        faces = np.zeros((0, 3), dtype=np.int64)
    return verts, faces


def _undirected_edge_stats(
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group all undirected face edges.

    Returns:
        unique_u: (E,) int64 smaller vertex endpoint per unique undirected edge
        unique_v: (E,) int64 larger vertex endpoint per unique undirected edge
        counts: (E,) int64 number of incident half-edges per unique undirected edge
        half_edges_dir: (3F_valid, 2) int64 directed half-edges (filtered to u != v)
        edge_inverse: (3F_valid,) int64 mapping each valid half-edge to unique edge index [0, E)
    """
    if len(faces) == 0:
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_e = np.zeros((0, 2), dtype=np.int64)
        return empty_i, empty_i, empty_i, empty_e, empty_i

    half_edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    valid = half_edges[:, 0] != half_edges[:, 1]
    half_edges = half_edges[valid]
    if len(half_edges) == 0:
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_e = np.zeros((0, 2), dtype=np.int64)
        return empty_i, empty_i, empty_i, empty_e, empty_i

    u = np.minimum(half_edges[:, 0], half_edges[:, 1])
    v = np.maximum(half_edges[:, 0], half_edges[:, 1])

    order = np.lexsort((v, u))
    u_s = u[order]
    v_s = v[order]

    diff = np.empty(len(u_s), dtype=bool)
    diff[0] = True
    if len(u_s) > 1:
        diff[1:] = (u_s[1:] != u_s[:-1]) | (v_s[1:] != v_s[:-1])

    group_starts = np.flatnonzero(diff)
    unique_u = u_s[group_starts]
    unique_v = v_s[group_starts]
    counts = np.diff(np.append(group_starts, len(u_s)))

    sorted_group_id = np.cumsum(diff) - 1
    edge_inverse = np.empty(len(u_s), dtype=np.int64)
    edge_inverse[order] = sorted_group_id
    return unique_u, unique_v, counts, half_edges, edge_inverse


def _count_duplicate_faces(faces: np.ndarray) -> int:
    if len(faces) <= 1:
        return 0
    sf = np.sort(faces, axis=1)
    order = np.lexsort((sf[:, 2], sf[:, 1], sf[:, 0]))
    s = sf[order]
    same = (s[1:, 0] == s[:-1, 0]) & (s[1:, 1] == s[:-1, 1]) & (s[1:, 2] == s[:-1, 2])
    return int(np.sum(same))


def _count_connected_components(num_verts: int, faces: np.ndarray) -> int:
    if len(faces) == 0 or num_verts == 0:
        return 0
    e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    rows = np.concatenate([e0, e1])
    cols = np.concatenate([e1, e0])
    data = np.ones(len(rows), dtype=np.int8)
    adj = csr_matrix((data, (rows, cols)), shape=(num_verts, num_verts))
    _, labels = connected_components(adj, directed=False)
    return int(np.unique(labels[faces[:, 0]]).size)


def compute_corner_fan_components(
    num_verts: int, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Partition all 3*F face corners into edge-connected vertex-fan components.

    Two corners belonging to the same vertex are connected iff their faces share
    an edge incident to that vertex. A vertex with >1 fan component is non-manifold.
    """
    num_faces = len(faces)
    num_corners = 3 * num_faces
    if num_faces == 0 or num_verts == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((num_verts,), dtype=np.int64)

    f_idx = np.arange(num_faces, dtype=np.int64)
    c0 = np.concatenate([3 * f_idx + 0, 3 * f_idx + 1, 3 * f_idx + 2])
    c1 = np.concatenate([3 * f_idx + 1, 3 * f_idx + 2, 3 * f_idx + 0])
    va = faces.reshape(-1)[c0]
    vb = faces.reshape(-1)[c1]

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []

    self_mask = va == vb
    if np.any(self_mask):
        row_parts.append(c0[self_mask])
        col_parts.append(c1[self_mask])

    valid_mask = ~self_mask
    if np.any(valid_mask):
        va_v = va[valid_mask]
        vb_v = vb[valid_mask]
        c0_v = c0[valid_mask]
        c1_v = c1[valid_mask]

        swap = va_v > vb_v
        u = np.where(swap, vb_v, va_v)
        v = np.where(swap, va_v, vb_v)
        cu = np.where(swap, c1_v, c0_v)
        cv = np.where(swap, c0_v, c1_v)

        order = np.lexsort((v, u))
        u_s = u[order]
        v_s = v[order]
        cu_s = cu[order]
        cv_s = cv[order]

        if len(u_s) > 1:
            same = (u_s[1:] == u_s[:-1]) & (v_s[1:] == v_s[:-1])
            if np.any(same):
                row_parts.append(cu_s[:-1][same])
                col_parts.append(cu_s[1:][same])
                row_parts.append(cv_s[:-1][same])
                col_parts.append(cv_s[1:][same])

    if row_parts:
        r_all = np.concatenate(row_parts)
        c_all = np.concatenate(col_parts)
        rows = np.concatenate([r_all, c_all])
        cols = np.concatenate([c_all, r_all])
        data = np.ones(len(rows), dtype=np.int8)
        adj = csr_matrix((data, (rows, cols)), shape=(num_corners, num_corners))
    else:
        adj = csr_matrix((num_corners, num_corners), dtype=np.int8)

    _, corner_labels = connected_components(adj, directed=False)

    verts_flat = faces.reshape(-1)
    order_vc = np.lexsort((corner_labels, verts_flat))
    vs = verts_flat[order_vc]
    ls = corner_labels[order_vc]
    uniq_mask = np.empty(len(vs), dtype=bool)
    uniq_mask[0] = True
    if len(vs) > 1:
        uniq_mask[1:] = (vs[1:] != vs[:-1]) | (ls[1:] != ls[:-1])
    fan_counts = np.bincount(vs[uniq_mask], minlength=num_verts)
    return corner_labels, fan_counts


def trace_boundary_loops(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    pinhole_max_perimeter_ratio: float = 0.03,
    pinhole_max_vertices: int = 16,
    diag: float | None = None,
) -> list[BoundaryLoop]:
    """Trace closed boundary edge cycles and classify pinhole vs large-opening loops."""
    if len(faces) == 0 or len(verts) == 0:
        return []

    _, _, counts, half_edges, edge_inverse = _undirected_edge_stats(faces)
    if len(counts) == 0:
        return []

    boundary_edge_mask = counts == 1
    if not np.any(boundary_edge_mask):
        return []

    if diag is None:
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        diag = float(np.linalg.norm(bbox_max - bbox_min))
    max_perim = float(pinhole_max_perimeter_ratio) * float(diag)

    he_is_boundary = boundary_edge_mask[edge_inverse]
    b_directed = half_edges[he_is_boundary]

    num_b = len(b_directed)
    adj_undirected: dict[int, list[tuple[int, int]]] = {}
    out_directed: dict[int, list[tuple[int, int]]] = {}

    for eid in range(num_b):
        s = int(b_directed[eid, 0])
        t = int(b_directed[eid, 1])
        adj_undirected.setdefault(s, []).append((t, eid))
        adj_undirected.setdefault(t, []).append((s, eid))
        out_directed.setdefault(s, []).append((t, eid))

    visited_edge = [False] * num_b
    raw_cycles: list[list[tuple[int, tuple[int, int]]]] = []

    def _pick_next_edge(curr: int) -> tuple[int, tuple[int, int]] | None:
        for nxt, eid in out_directed.get(curr, ()):
            if not visited_edge[eid]:
                visited_edge[eid] = True
                return nxt, (curr, nxt)
        for nxt, eid in adj_undirected.get(curr, ()):
            if not visited_edge[eid]:
                visited_edge[eid] = True
                return nxt, (curr, nxt)
        return None

    degree_one_verts = [v for v, nbrs in adj_undirected.items() if len(nbrs) == 1]
    start_order = degree_one_verts + sorted(
        v for v in adj_undirected if len(adj_undirected[v]) != 1
    )

    for start_v in start_order:
        while True:
            step0 = _pick_next_edge(start_v)
            if step0 is None:
                break
            nxt0, de0 = step0
            path_verts = [start_v]
            path_edges = [de0]
            pos_in_path: dict[int, int] = {start_v: 0}
            curr = nxt0

            while True:
                if curr in pos_in_path:
                    idx0 = pos_in_path[curr]
                    cyc_verts = path_verts[idx0:]
                    cyc_edges = path_edges[idx0:]
                    if len(cyc_verts) >= 3:
                        raw_cycles.append(list(zip(cyc_verts, cyc_edges)))
                    path_verts = path_verts[:idx0]
                    path_edges = path_edges[:idx0]
                    pos_in_path = {v: i for i, v in enumerate(path_verts)}
                pos_in_path[curr] = len(path_verts)
                path_verts.append(curr)

                step = _pick_next_edge(curr)
                if step is None:
                    break
                nxt, de = step
                path_edges.append(de)
                curr = nxt

    loops: list[BoundaryLoop] = []
    for cyc in raw_cycles:
        cyc_verts = [item[0] for item in cyc]
        cyc_edges = [item[1] for item in cyc]
        pts = verts[np.asarray(cyc_verts, dtype=np.int64)]
        pts_next = np.roll(pts, -1, axis=0)
        perimeter = float(np.sum(np.linalg.norm(pts_next - pts, axis=1)))
        is_pinhole = (
            len(cyc_verts) <= int(pinhole_max_vertices)
            and diag > 0.0
            and perimeter <= max_perim
        )
        loops.append(
            BoundaryLoop(
                vertices=cyc_verts,
                directed_edges=cyc_edges,
                perimeter=perimeter,
                is_pinhole=bool(is_pinhole),
            )
        )
    return loops


def compute_triangle_areas(
    verts: np.ndarray, faces: np.ndarray, *, chunk_size: int = 1_000_000
) -> np.ndarray:
    """Compute per-face triangle areas in chunks to keep memory bounded on 10M+ meshes."""
    num_faces = len(faces)
    if num_faces == 0:
        return np.zeros((0,), dtype=np.float64)
    areas = np.empty(num_faces, dtype=np.float64)
    for start in range(0, num_faces, chunk_size):
        end = min(num_faces, start + chunk_size)
        f_chunk = faces[start:end]
        v0 = verts[f_chunk[:, 0]]
        v1 = verts[f_chunk[:, 1]]
        v2 = verts[f_chunk[:, 2]]
        areas[start:end] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    return areas


def compute_mesh_metrics_from_vf(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    pinhole_max_perimeter_ratio: float = 0.03,
    pinhole_max_vertices: int = 16,
    area_eps_rel: float = 1e-12,
) -> MeshMetrics:
    """Compute deterministic topology and bounding-box metrics from (V, F) arrays."""
    vertex_count = int(len(verts))
    face_count = int(len(faces))

    if vertex_count == 0:
        return MeshMetrics(
            vertex_count=0,
            face_count=0,
            open_boundary_edges=0,
            boundary_loops=0,
            pinhole_loops=0,
            large_opening_loops=0,
            non_manifold_edges=0,
            non_manifold_vertices=0,
            degenerate_faces=0,
            duplicate_faces=0,
            connected_components=0,
            is_watertight=False,
            is_pinhole_free=False,
            bbox_min=[0.0, 0.0, 0.0],
            bbox_max=[0.0, 0.0, 0.0],
            height=0.0,
        )

    finite_mask = np.isfinite(verts).all(axis=1)
    if np.any(finite_mask):
        bbox_min = verts[finite_mask].min(axis=0)
        bbox_max = verts[finite_mask].max(axis=0)
    else:
        bbox_min = np.zeros(3, dtype=np.float64)
        bbox_max = np.zeros(3, dtype=np.float64)
    height = float(bbox_max[1] - bbox_min[1])
    diag = float(np.linalg.norm(bbox_max - bbox_min))

    if face_count == 0:
        return MeshMetrics(
            vertex_count=vertex_count,
            face_count=0,
            open_boundary_edges=0,
            boundary_loops=0,
            pinhole_loops=0,
            large_opening_loops=0,
            non_manifold_edges=0,
            non_manifold_vertices=0,
            degenerate_faces=0,
            duplicate_faces=0,
            connected_components=0,
            is_watertight=False,
            is_pinhole_free=False,
            bbox_min=bbox_min.tolist(),
            bbox_max=bbox_max.tolist(),
            height=height,
        )

    dup_idx = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    areas = compute_triangle_areas(verts, faces)
    area_thresh = max(1e-14, float(area_eps_rel) * (diag * diag))
    degen_mask = dup_idx | ~np.isfinite(areas) | (areas <= area_thresh)
    degenerate_faces = int(np.sum(degen_mask))

    duplicate_faces = _count_duplicate_faces(faces)

    _, _, edge_counts, _, _ = _undirected_edge_stats(faces)
    open_boundary_edges = int(np.sum(edge_counts == 1))
    non_manifold_edges = int(np.sum(edge_counts > 2))

    _, fan_counts = compute_corner_fan_components(vertex_count, faces)
    non_manifold_vertices = int(np.sum(fan_counts > 1))

    conn_comps = _count_connected_components(vertex_count, faces)

    loops = trace_boundary_loops(
        verts,
        faces,
        pinhole_max_perimeter_ratio=pinhole_max_perimeter_ratio,
        pinhole_max_vertices=pinhole_max_vertices,
        diag=diag,
    )
    pinhole_loops = sum(1 for lp in loops if lp.is_pinhole)
    large_opening_loops = len(loops) - pinhole_loops
    boundary_loops = len(loops)

    is_watertight = (
        face_count > 0
        and open_boundary_edges == 0
        and non_manifold_edges == 0
        and non_manifold_vertices == 0
        and degenerate_faces == 0
        and duplicate_faces == 0
    )
    is_pinhole_free = (
        face_count > 0
        and pinhole_loops == 0
        and non_manifold_edges == 0
        and non_manifold_vertices == 0
        and degenerate_faces == 0
    )

    return MeshMetrics(
        vertex_count=vertex_count,
        face_count=face_count,
        open_boundary_edges=open_boundary_edges,
        boundary_loops=boundary_loops,
        pinhole_loops=pinhole_loops,
        large_opening_loops=large_opening_loops,
        non_manifold_edges=non_manifold_edges,
        non_manifold_vertices=non_manifold_vertices,
        degenerate_faces=degenerate_faces,
        duplicate_faces=duplicate_faces,
        connected_components=conn_comps,
        is_watertight=is_watertight,
        is_pinhole_free=is_pinhole_free,
        bbox_min=bbox_min.tolist(),
        bbox_max=bbox_max.tolist(),
        height=height,
    )


def compute_mesh_metrics(
    mesh_or_path: str | Path | trimesh.Trimesh,
    *,
    pinhole_max_perimeter_ratio: float = 0.03,
    pinhole_max_vertices: int = 16,
    area_eps_rel: float = 1e-12,
) -> MeshMetrics:
    """Compute deterministic topology and bounding-box metrics for a mesh or GLB path."""
    verts, faces = extract_vf(mesh_or_path)
    return compute_mesh_metrics_from_vf(
        verts,
        faces,
        pinhole_max_perimeter_ratio=pinhole_max_perimeter_ratio,
        pinhole_max_vertices=pinhole_max_vertices,
        area_eps_rel=area_eps_rel,
    )


def sample_surface_points(
    verts: np.ndarray,
    faces: np.ndarray,
    count: int = 100000,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Deterministically sample `count` area-weighted points on a triangle surface."""
    if len(verts) == 0 or count <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    if len(faces) == 0:
        idx = rng.integers(0, len(verts), size=int(count))
        return np.asarray(verts[idx], dtype=np.float64)

    areas = compute_triangle_areas(verts, faces)
    areas = np.where(np.isfinite(areas) & (areas > 0.0), areas, 0.0)
    total_area = float(np.sum(areas))
    if total_area <= 0.0:
        idx = rng.integers(0, len(verts), size=int(count))
        return np.asarray(verts[idx], dtype=np.float64)

    cdf = np.cumsum(areas, dtype=np.float64) / total_area
    cdf[-1] = 1.0
    u = rng.random(int(count))
    face_idx = np.searchsorted(cdf, u, side="right")
    face_idx = np.clip(face_idx, 0, len(faces) - 1)

    r1 = np.sqrt(rng.random(int(count)))
    r2 = rng.random(int(count))
    w0 = (1.0 - r1)[:, None]
    w1 = (r1 * (1.0 - r2))[:, None]
    w2 = (r1 * r2)[:, None]

    f_sel = faces[face_idx]
    return w0 * verts[f_sel[:, 0]] + w1 * verts[f_sel[:, 1]] + w2 * verts[f_sel[:, 2]]


def prepare_highpoly_reference(
    high_mesh_or_path: str | Path | trimesh.Trimesh,
    *,
    sample_count: int = 100000,
    seed: int = 42,
) -> tuple[np.ndarray, cKDTree, MeshMetrics]:
    """Sample a highpoly mesh once and build a reusable cKDTree + MeshMetrics."""
    verts, faces = extract_vf(high_mesh_or_path)
    metrics = compute_mesh_metrics_from_vf(verts, faces)
    pts = sample_surface_points(verts, faces, count=sample_count, seed=seed)
    tree = cKDTree(pts) if len(pts) > 0 else cKDTree(np.zeros((1, 3), dtype=np.float64))
    return pts, tree, metrics


def compute_surface_distance_metrics(
    low_verts: np.ndarray,
    low_faces: np.ndarray,
    *,
    high_verts: np.ndarray | None = None,
    high_faces: np.ndarray | None = None,
    high_points: np.ndarray | None = None,
    high_tree: cKDTree | None = None,
    sample_count: int = 100000,
    seed: int = 42,
) -> dict[str, float | None]:
    """Compute bidirectional Chamfer distance and Hausdorff p95/p99 against highpoly."""
    if high_points is None:
        if high_verts is None or high_faces is None:
            raise ValueError("Provide either (high_verts, high_faces) or high_points.")
        high_points = sample_surface_points(high_verts, high_faces, count=sample_count, seed=seed)

    low_points = sample_surface_points(low_verts, low_faces, count=sample_count, seed=seed + 1)
    if len(high_points) == 0 or len(low_points) == 0:
        return {
            "chamfer_distance": None,
            "chamfer_high_to_low": None,
            "chamfer_low_to_high": None,
            "hausdorff_p95": None,
            "hausdorff_p99": None,
        }

    tree_high = high_tree if high_tree is not None else cKDTree(high_points)
    tree_low = cKDTree(low_points)

    d_h2l, _ = tree_low.query(high_points, workers=-1)
    d_l2h, _ = tree_high.query(low_points, workers=-1)

    c_h2l = float(np.mean(d_h2l))
    c_l2h = float(np.mean(d_l2h))
    chamfer = 0.5 * (c_h2l + c_l2h)
    hp95 = float(max(np.percentile(d_h2l, 95.0), np.percentile(d_l2h, 95.0)))
    hp99 = float(max(np.percentile(d_h2l, 99.0), np.percentile(d_l2h, 99.0)))
    return {
        "chamfer_distance": chamfer,
        "chamfer_high_to_low": c_h2l,
        "chamfer_low_to_high": c_l2h,
        "hausdorff_p95": hp95,
        "hausdorff_p99": hp99,
    }


def compute_head_face_ratios(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    head_thresh: float = 0.72,
    head_core_thresh: float = 0.76,
    y_min: float | None = None,
    height: float | None = None,
) -> tuple[float, float]:
    """Compute fraction of faces with normalized centroid height y_norm > 0.72 and > 0.76."""
    if len(faces) == 0 or len(verts) == 0:
        return 0.0, 0.0

    if y_min is None or height is None:
        finite = np.isfinite(verts[:, 1])
        if not np.any(finite):
            return 0.0, 0.0
        y_vals = verts[finite, 1]
        y_lo = float(np.min(y_vals))
        y_hi = float(np.max(y_vals))
        h = y_hi - y_lo
    else:
        y_lo = float(y_min)
        h = float(height)

    if not math.isfinite(h) or h <= 1e-12:
        return 0.0, 0.0

    cy = (verts[faces[:, 0], 1] + verts[faces[:, 1], 1] + verts[faces[:, 2], 1]) / 3.0
    y_norm = (cy - y_lo) / h
    head_ratio = float(np.mean(y_norm > float(head_thresh)))
    head_core_ratio = float(np.mean(y_norm > float(head_core_thresh)))
    return head_ratio, head_core_ratio


def compute_valence_ge8_ratio(num_verts: int, faces: np.ndarray) -> float:
    """Fraction of referenced vertices with undirected edge degree >= 8."""
    if num_verts <= 0 or len(faces) == 0:
        return 0.0
    unique_u, unique_v, _, _, _ = _undirected_edge_stats(faces)
    if len(unique_u) == 0:
        return 0.0
    deg = np.bincount(np.concatenate([unique_u, unique_v]), minlength=num_verts)
    referenced = np.zeros(num_verts, dtype=bool)
    referenced[faces.reshape(-1)] = True
    denom = int(np.sum(referenced))
    if denom == 0:
        return 0.0
    return float(np.sum(deg[referenced] >= 8) / denom)


def compute_aspect_ratio_p95(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    area_eps_rel: float = 1e-14,
) -> float:
    """Compute 95th-percentile normalized triangle aspect ratio (equilateral = 1.0).

    Uses `L_max * (a + b + c) / (4 * sqrt(3) * area)` and cleanly skips zero-area
    degenerate faces (counted separately by `degenerate_faces`).
    """
    if len(faces) == 0 or len(verts) == 0:
        return 0.0

    finite_mask = np.isfinite(verts).all(axis=1)
    if np.any(finite_mask):
        v_fin = verts[finite_mask]
        diag = float(np.linalg.norm(v_fin.max(axis=0) - v_fin.min(axis=0)))
    else:
        diag = 0.0
    area_thresh = max(1e-16, float(area_eps_rel) * (diag * diag))

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    valid = np.isfinite(areas) & (areas > area_thresh) & np.isfinite(e0) & np.isfinite(e1) & np.isfinite(e2)
    if not np.any(valid):
        return 0.0

    a = e0[valid]
    b = e1[valid]
    c = e2[valid]
    ar_area = areas[valid]
    l_max = np.maximum(np.maximum(a, b), c)
    perim = a + b + c
    aspect = (l_max * perim) / (4.0 * math.sqrt(3.0) * ar_area)
    aspect = np.maximum(aspect, 1.0)
    return float(np.percentile(aspect, 95.0))


def compute_retopo_metrics(
    mesh_or_path: str | Path | trimesh.Trimesh,
    *,
    highpoly_mesh_or_path: str | Path | trimesh.Trimesh | None = None,
    highpoly_points: np.ndarray | None = None,
    highpoly_tree: cKDTree | None = None,
    sample_count: int = 100000,
    seed: int = 42,
    semantic_extract_s: float = 0.0,
    part_cluster_count: int = 0,
    focus_vertex_ratio: float = 0.0,
    uv_overlap_count: int = 0,
    uv_coverage: float = 0.0,
    bake_elapsed_s: float = 0.0,
    ray_miss_ratio: float = 0.0,
    cage_extrusion: float | None = None,
    max_ray_distance: float | None = None,
    decimate_elapsed_s: float = 0.0,
    error: str | None = None,
    notes: str = "",
) -> RetopoMetrics:
    """Compute complete topology, fidelity, semantic allocation, and UV/bake metrics."""
    quad_count = 0
    tri_count = 0
    equiv_tri_count = 0

    if isinstance(mesh_or_path, (str, Path)):
        src_path = Path(mesh_or_path)
        if src_path.suffix.lower() == ".obj":
            poly_counts = parse_obj_polygon_counts(src_path)
            quad_count = poly_counts.quad_count
            tri_count = poly_counts.tri_count
            equiv_tri_count = poly_counts.equiv_tri_count

    verts, faces = extract_vf(mesh_or_path)
    topo = compute_mesh_metrics_from_vf(verts, faces)

    if equiv_tri_count == 0:
        tri_count = topo.face_count
        equiv_tri_count = topo.face_count

    head_face_ratio, head_core_ratio = compute_head_face_ratios(verts, faces)
    valence_ge8_ratio = compute_valence_ge8_ratio(len(verts), faces)
    aspect_ratio_p95 = compute_aspect_ratio_p95(verts, faces)

    dist_stats: dict[str, float | None] = {
        "chamfer_distance": None,
        "chamfer_high_to_low": None,
        "chamfer_low_to_high": None,
        "hausdorff_p95": None,
        "hausdorff_p99": None,
    }
    if highpoly_points is not None or highpoly_mesh_or_path is not None:
        if highpoly_points is None and highpoly_mesh_or_path is not None:
            hv, hf = extract_vf(highpoly_mesh_or_path)
            dist_stats = compute_surface_distance_metrics(
                verts,
                faces,
                high_verts=hv,
                high_faces=hf,
                sample_count=sample_count,
                seed=seed,
            )
        else:
            dist_stats = compute_surface_distance_metrics(
                verts,
                faces,
                high_points=highpoly_points,
                high_tree=highpoly_tree,
                sample_count=sample_count,
                seed=seed,
            )

    return RetopoMetrics(
        vertex_count=topo.vertex_count,
        face_count=topo.face_count,
        open_boundary_edges=topo.open_boundary_edges,
        boundary_loops=topo.boundary_loops,
        pinhole_loops=topo.pinhole_loops,
        large_opening_loops=topo.large_opening_loops,
        non_manifold_edges=topo.non_manifold_edges,
        non_manifold_vertices=topo.non_manifold_vertices,
        degenerate_faces=topo.degenerate_faces,
        duplicate_faces=topo.duplicate_faces,
        connected_components=topo.connected_components,
        is_watertight=topo.is_watertight,
        is_pinhole_free=topo.is_pinhole_free,
        bbox_min=topo.bbox_min,
        bbox_max=topo.bbox_max,
        height=topo.height,
        quad_count=quad_count,
        tri_count=tri_count,
        equiv_tri_count=equiv_tri_count,
        chamfer_distance=dist_stats["chamfer_distance"],
        chamfer_high_to_low=dist_stats["chamfer_high_to_low"],
        chamfer_low_to_high=dist_stats["chamfer_low_to_high"],
        hausdorff_p95=dist_stats["hausdorff_p95"],
        hausdorff_p99=dist_stats["hausdorff_p99"],
        sample_count=int(sample_count),
        semantic_extract_s=float(semantic_extract_s),
        part_cluster_count=int(part_cluster_count),
        focus_vertex_ratio=float(focus_vertex_ratio),
        head_face_ratio=head_face_ratio,
        head_core_ratio=head_core_ratio,
        valence_ge8_ratio=valence_ge8_ratio,
        aspect_ratio_p95=aspect_ratio_p95,
        uv_overlap_count=int(uv_overlap_count),
        uv_coverage=float(uv_coverage),
        bake_elapsed_s=float(bake_elapsed_s),
        ray_miss_ratio=float(ray_miss_ratio),
        cage_extrusion=cage_extrusion,
        max_ray_distance=max_ray_distance,
        decimate_elapsed_s=float(decimate_elapsed_s),
        error=error,
        notes=notes,
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute deterministic S2 RetopoMetrics.")
    parser.add_argument("--mesh", required=True, help="Target mesh path (.glb/.obj/.ply).")
    parser.add_argument("--highpoly", default=None, help="Optional reference highpoly mesh.")
    parser.add_argument("--samples", type=int, default=100000, help="Surface sample count.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    metrics = compute_retopo_metrics(
        args.mesh,
        highpoly_mesh_or_path=args.highpoly,
        sample_count=args.samples,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - t0
    if not metrics.notes:
        metrics.notes = f"evaluated_in={elapsed:.3f}s"

    if args.out:
        metrics.write_json(args.out)
    else:
        print(metrics.to_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
