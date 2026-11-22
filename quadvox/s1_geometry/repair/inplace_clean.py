"""Official Route 1: general-purpose topology repair matching O-Voxel `remesh=False`
and ComfyUI `WeldVertices` + `FillHoles` standards.

Zero character/clothing semantic rules: performs pure general geometry repair
(sub-voxel seam welding, duplicate/degenerate face cleanup, non-manifold edge/vertex
repair, tiny floater speck removal, small pinhole filling, and orientation unification).

Verified upstream URLs:
- O-Voxel standard clean branch (`remesh=False`):
  https://github.com/microsoft/TRELLIS.2/blob/main/o-voxel/o_voxel/postprocess.py
- ComfyUI WeldVertices & FillHoles:
  https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_mesh_postprocess.py
"""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from quadvox.contracts import MeshMetrics

from .metrics import (
    _undirected_edge_stats,
    compute_corner_fan_components,
    compute_mesh_metrics,
    load_mesh_raw,
    trace_boundary_loops,
)


def _bbox_diag(verts: np.ndarray) -> float:
    if len(verts) == 0:
        return 0.0
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    return float(np.linalg.norm(hi - lo))


def _compact_unreferenced_vertices(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    used = np.zeros(len(verts), dtype=bool)
    used[faces.reshape(-1)] = True
    if np.all(used):
        return verts, faces
    remap = np.full(len(verts), -1, dtype=np.int64)
    kept_indices = np.flatnonzero(used)
    remap[kept_indices] = np.arange(len(kept_indices), dtype=np.int64)
    return verts[kept_indices], remap[faces]


def _weld_vertices(
    verts: np.ndarray,
    faces: np.ndarray,
    epsilon_rel: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Merge coincident / sub-voxel seam vertices via radius clustering."""
    if len(verts) == 0 or len(faces) == 0 or epsilon_rel <= 0.0:
        return verts, faces, 0

    diag = _bbox_diag(verts)
    if diag <= 0.0:
        return verts, faces, 0

    eps = max(float(epsilon_rel) * diag, 1e-12)
    pairs = cKDTree(verts).query_pairs(r=eps, output_type="ndarray")
    if len(pairs) == 0:
        return verts, faces, 0

    num_v = len(verts)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num_v, num_v))
    n_unique, labels = connected_components(adj, directed=False)
    if n_unique >= num_v:
        return verts, faces, 0

    new_verts = np.zeros((n_unique, 3), dtype=np.float64)
    counts = np.bincount(labels, minlength=n_unique).astype(np.float64)
    np.add.at(new_verts, labels, verts)
    new_verts /= np.maximum(counts[:, None], 1.0)
    new_faces = labels[faces]
    return new_verts, new_faces, int(num_v - n_unique)


def _drop_bad_and_duplicate_faces(
    verts: np.ndarray,
    faces: np.ndarray,
    area_eps_rel: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0 or len(verts) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)

    num_v = len(verts)
    finite_v = np.isfinite(verts).all(axis=1)
    valid_idx = (faces >= 0).all(axis=1) & (faces < num_v).all(axis=1)
    faces = faces[valid_idx]
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)

    faces = faces[finite_v[faces].all(axis=1)]
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)

    non_dup_idx = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    faces = faces[non_dup_idx]
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)

    diag = _bbox_diag(verts)
    area_thresh = max(1e-14, float(area_eps_rel) * (diag * diag))
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    faces = faces[np.isfinite(areas) & (areas > area_thresh)]
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)

    # Deduplicate unordered vertex triples, keeping first occurrence in deterministic order.
    sf = np.sort(faces, axis=1)
    order = np.lexsort((sf[:, 2], sf[:, 1], sf[:, 0]))
    s = sf[order]
    keep_sorted = np.empty(len(s), dtype=bool)
    keep_sorted[0] = True
    if len(s) > 1:
        keep_sorted[1:] = (s[1:, 0] != s[:-1, 0]) | (s[1:, 1] != s[:-1, 1]) | (s[1:, 2] != s[:-1, 2])
    keep_orig = np.zeros(len(faces), dtype=bool)
    keep_orig[order[keep_sorted]] = True
    faces = faces[keep_orig]
    return _compact_unreferenced_vertices(verts, faces)


def _build_face_edge_lookup(faces: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], list[tuple[int, int]]]]:
    """Return per-face undirected edge keys and map `(u, v) -> [(face_id, dir_sign), ...]`."""
    num_f = len(faces)
    edge_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    face_edges = np.empty((num_f, 3, 2), dtype=np.int64)
    for fi in range(num_f):
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        for ei, (p, q) in enumerate(((a, b), (b, c), (c, a))):
            if p < q:
                key = (p, q)
                sign = 1
            else:
                key = (q, p)
                sign = -1
            face_edges[fi, ei, 0] = key[0]
            face_edges[fi, ei, 1] = key[1]
            edge_map.setdefault(key, []).append((fi, sign))
    return face_edges, edge_map


def _remove_topological_fins_and_bridges(faces: np.ndarray) -> np.ndarray:
    """Remove unambiguous 1-face internal bridges (3 edges >= 3) and dangling 1-tri fins (1 edge >= 3, 2 edges == 1)."""
    if len(faces) == 0:
        return faces

    unique_u, unique_v, counts, _, edge_inverse = _undirected_edge_stats(faces)
    if not np.any(counts > 2):
        return faces

    num_f = len(faces)
    # Map each face's 3 half-edges to its undirected edge index.
    # _undirected_edge_stats builds half_edges via vstack([f[:,[0,1]], f[:,[1,2]], f[:,[2,0]]])
    # and filters u != v. Since degenerate u == v faces were already removed, len(edge_inverse) == 3*num_f.
    if len(edge_inverse) != 3 * num_f:
        return faces

    face_edge_ids = edge_inverse.reshape(3, num_f).T  # (F, 3)
    cur_counts = counts.copy()
    f_edge_counts = cur_counts[face_edge_ids]

    is_bridge = np.all(f_edge_counts >= 3, axis=1)
    is_fin = (np.sum(f_edge_counts >= 3, axis=1) == 1) & (np.sum(f_edge_counts == 1, axis=1) == 2)
    candidates = np.flatnonzero(is_bridge | is_fin)
    if len(candidates) == 0:
        return faces

    keep = np.ones(num_f, dtype=bool)
    for fi in candidates:
        e0, e1, e2 = int(face_edge_ids[fi, 0]), int(face_edge_ids[fi, 1]), int(face_edge_ids[fi, 2])
        c0, c1, c2 = int(cur_counts[e0]), int(cur_counts[e1]), int(cur_counts[e2])
        if c0 >= 3 and c1 >= 3 and c2 >= 3:
            keep[fi] = False
            cur_counts[e0] -= 1
            cur_counts[e1] -= 1
            cur_counts[e2] -= 1
        else:
            # Verify it is still a dangling fin (1 non-manifold >= 3 edge, 2 boundary == 1 edges).
            cs = (c0, c1, c2)
            if sum(1 for c in cs if c >= 3) == 1 and sum(1 for c in cs if c == 1) == 2:
                keep[fi] = False
                cur_counts[e0] -= 1
                cur_counts[e1] -= 1
                cur_counts[e2] -= 1

    return faces[keep]


def _pair_faces_around_nonmanifold_edge(
    u: int,
    v: int,
    inc_faces: list[tuple[int, int]],
    verts: np.ndarray,
    faces: np.ndarray,
) -> list[tuple[int, int]]:
    """Pair incident faces around a non-manifold edge `(u, v)` by cyclic azimuth around the edge axis."""
    if len(inc_faces) < 2:
        return []

    axis = verts[v] - verts[u]
    axis_len = float(np.linalg.norm(axis))
    if axis_len <= 1e-15:
        return []
    axis_unit = axis / axis_len

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(axis_unit, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    t1 = np.cross(axis_unit, ref)
    t1 /= max(float(np.linalg.norm(t1)), 1e-15)
    t2 = np.cross(axis_unit, t1)

    angles: list[tuple[float, int, int]] = []
    for fi, sign in inc_faces:
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        other = a if (a != u and a != v) else (b if (b != u and b != v) else c)
        vec = verts[other] - verts[u]
        perp = vec - float(np.dot(vec, axis_unit)) * axis_unit
        ang = float(np.arctan2(float(np.dot(perp, t2)), float(np.dot(perp, t1))))
        angles.append((ang, fi, sign))

    angles.sort(key=lambda item: item[0])
    m = len(angles)
    if m == 3:
        # Pick the adjacent pair with largest angular separation (widest surface sheet),
        # preferring opposite half-edge directions when available.
        best_score = (-1, -1.0)
        best_pair = (angles[0][1], angles[1][1])
        for i in range(3):
            a0, f0, s0 = angles[i]
            a1, f1, s1 = angles[(i + 1) % 3]
            gap = (a1 - a0) % (2.0 * np.pi)
            opp = 1 if s0 != s1 else 0
            score = (opp, min(gap, 2.0 * np.pi - gap))
            if score > best_score:
                best_score = score
                best_pair = (f0, f1)
        return [best_pair]

    # Even or >3: pair consecutive faces around the edge starting from the best opposite-sign pairing.
    pairs_even = [(angles[i][1], angles[i + 1][1]) for i in range(0, m - 1, 2)]
    pairs_odd = [(angles[i][1], angles[(i + 1) % m][1]) for i in range(1, m, 2)]
    opp_even = sum(1 for i in range(0, m - 1, 2) if angles[i][2] != angles[i + 1][2])
    opp_odd = sum(1 for i in range(1, m, 2) if angles[i][2] != angles[(i + 1) % m][2])
    return pairs_even if opp_even >= opp_odd else pairs_odd


def _split_corners_by_edges(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    pair_non_manifold: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Split non-manifold edges and vertex fans cleanly via 3*F-corner connected components.

    Follows CuMesh::repair_non_manifold_edges by connecting face corners across
    manifold (degree-2) edges (and optional disjoint sheet pairs on degree->2 edges),
    then promoting each corner component to its own vertex.
    """
    num_f = len(faces)
    num_corners = 3 * num_f
    if num_f == 0:
        return verts, faces

    # Build lookup from (face_id, vertex_id) -> corner_id (3*face_id + slot).
    # For valid non-degenerate faces, all 3 vertices in a face are distinct.
    row_links: list[int] = []
    col_links: list[int] = []

    def _connect_faces_on_edge(f1: int, f2: int, u: int, v: int) -> None:
        c1_u = c1_v = c2_u = c2_v = -1
        for slot in range(3):
            if int(faces[f1, slot]) == u:
                c1_u = 3 * f1 + slot
            elif int(faces[f1, slot]) == v:
                c1_v = 3 * f1 + slot
            if int(faces[f2, slot]) == u:
                c2_u = 3 * f2 + slot
            elif int(faces[f2, slot]) == v:
                c2_v = 3 * f2 + slot
        if c1_u >= 0 and c2_u >= 0:
            row_links.append(c1_u)
            col_links.append(c2_u)
        if c1_v >= 0 and c2_v >= 0:
            row_links.append(c1_v)
            col_links.append(c2_v)

    # Fast vectorized link generation for degree-2 edges; handle degree > 2 separately if needed.
    f_idx = np.arange(num_f, dtype=np.int64)
    c0 = np.concatenate([3 * f_idx + 0, 3 * f_idx + 1, 3 * f_idx + 2])
    c1 = np.concatenate([3 * f_idx + 1, 3 * f_idx + 2, 3 * f_idx + 0])
    va = faces.reshape(-1)[c0]
    vb = faces.reshape(-1)[c1]
    swap = va > vb
    eu = np.where(swap, vb, va)
    ev = np.where(swap, va, vb)
    cu = np.where(swap, c1, c0)
    cv = np.where(swap, c0, c1)

    order = np.lexsort((ev, eu))
    eu_s = eu[order]
    ev_s = ev[order]
    cu_s = cu[order]
    cv_s = cv[order]

    diff = np.empty(len(eu_s), dtype=bool)
    diff[0] = True
    if len(eu_s) > 1:
        diff[1:] = (eu_s[1:] != eu_s[:-1]) | (ev_s[1:] != ev_s[:-1])
    starts = np.flatnonzero(diff)
    counts = np.diff(np.append(starts, len(eu_s)))

    # Connect all manifold degree-2 edges in vectorized slices.
    deg2_starts = starts[counts == 2]
    r_vec = np.concatenate([cu_s[deg2_starts], cv_s[deg2_starts]])
    c_vec = np.concatenate([cu_s[deg2_starts + 1], cv_s[deg2_starts + 1]])

    if pair_non_manifold:
        nm_indices = np.flatnonzero(counts > 2)
        for idx in nm_indices:
            st = int(starts[idx])
            cnt = int(counts[idx])
            u_val = int(eu_s[st])
            v_val = int(ev_s[st])
            inc: list[tuple[int, int]] = []
            for k in range(st, st + cnt):
                corner_u = int(cu_s[k])
                fi = corner_u // 3
                # Determine original half-edge sign wrt (u_val, v_val)
                s_u = corner_u % 3
                s_v = int(cv_s[k]) % 3
                sign = 1 if (s_u + 1) % 3 == s_v else -1
                inc.append((fi, sign))
            for f1, f2 in _pair_faces_around_nonmanifold_edge(u_val, v_val, inc, verts, faces):
                _connect_faces_on_edge(f1, f2, u_val, v_val)

    r_all = [r_vec]
    c_all = [c_vec]
    if row_links:
        r_all.append(np.asarray(row_links, dtype=np.int64))
        c_all.append(np.asarray(col_links, dtype=np.int64))
    r_cat = np.concatenate(r_all)
    c_cat = np.concatenate(c_all)
    rows = np.concatenate([r_cat, c_cat])
    cols = np.concatenate([c_cat, r_cat])
    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num_corners, num_corners))
    n_new_v, corner_labels = connected_components(adj, directed=False)

    # First corner representative for each component determines its 3D vertex position.
    first_corner = np.empty(n_new_v, dtype=np.int64)
    first_corner[corner_labels] = np.arange(num_corners, dtype=np.int64)
    old_v_for_new = faces.reshape(-1)[first_corner]
    new_verts = verts[old_v_for_new]
    new_faces = corner_labels.reshape(num_f, 3)
    return new_verts, new_faces


def _drop_excess_nonmanifold_faces(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Last-resort edge cleanup if circular loops keep an edge >2 faces after splitting."""
    if len(faces) == 0:
        return faces
    _, edge_map = _build_face_edge_lookup(faces)
    drop_faces: set[int] = set()
    for (u, v), inc in edge_map.items():
        if len(inc) <= 2:
            continue
        pairs = _pair_faces_around_nonmanifold_edge(u, v, inc, verts, faces)
        kept = {pairs[0][0], pairs[0][1]} if pairs else {inc[0][0], inc[1][0]}
        for fi, _ in inc:
            if fi not in kept:
                drop_faces.add(fi)
    if not drop_faces:
        return faces
    keep_mask = np.ones(len(faces), dtype=bool)
    keep_mask[list(drop_faces)] = False
    return faces[keep_mask]


def _repair_non_manifold(
    verts: np.ndarray,
    faces: np.ndarray,
    area_eps_rel: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return verts, faces

    faces = _remove_topological_fins_and_bridges(faces)
    verts, faces = _drop_bad_and_duplicate_faces(verts, faces, area_eps_rel=area_eps_rel)
    if len(faces) == 0:
        return verts, faces

    for step in range(6):
        _, _, counts, _, _ = _undirected_edge_stats(faces)
        has_nm_edge = bool(np.any(counts > 2))
        _, fan_counts = compute_corner_fan_components(len(verts), faces)
        has_nm_vert = bool(np.any(fan_counts > 1))
        if not has_nm_edge and not has_nm_vert:
            break

        if step == 0:
            verts, faces = _split_corners_by_edges(verts, faces, pair_non_manifold=True)
        elif step <= 2:
            verts, faces = _split_corners_by_edges(verts, faces, pair_non_manifold=False)
        else:
            faces = _drop_excess_nonmanifold_faces(verts, faces)
            verts, faces = _split_corners_by_edges(verts, faces, pair_non_manifold=False)

        verts, faces = _drop_bad_and_duplicate_faces(verts, faces, area_eps_rel=area_eps_rel)
        if len(faces) == 0:
            break

    return verts, faces


def _remove_small_components(
    verts: np.ndarray,
    faces: np.ndarray,
    min_area_rel: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove tiny isolated floater specks < min_area_rel of total surface area."""
    if len(faces) == 0 or min_area_rel <= 0.0:
        return verts, faces

    num_v = len(verts)
    e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    rows = np.concatenate([e0, e1])
    cols = np.concatenate([e1, e0])
    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num_v, num_v))
    _, v_labels = connected_components(adj, directed=False)
    face_comp = v_labels[faces[:, 0]]

    uniq_c, inv = np.unique(face_comp, return_inverse=True)
    if len(uniq_c) <= 1:
        return verts, faces

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    comp_areas = np.bincount(inv, weights=face_areas, minlength=len(uniq_c))
    total_area = float(np.sum(comp_areas))
    if total_area <= 0.0:
        return verts, faces

    largest = int(np.argmax(comp_areas))
    keep_comp = comp_areas >= (float(min_area_rel) * total_area)
    keep_comp[largest] = True
    if np.all(keep_comp):
        return verts, faces

    faces = faces[keep_comp[inv]]
    return _compact_unreferenced_vertices(verts, faces)


def _unify_face_orientations(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Propagate consistent winding per edge-connected component and orient outward."""
    num_f = len(faces)
    if num_f == 0:
        return faces

    faces = faces.copy()
    f_idx = np.arange(num_f, dtype=np.int64)
    va = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    vb = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    f_id = np.concatenate([f_idx, f_idx, f_idx])

    swap = va > vb
    eu = np.where(swap, vb, va)
    ev = np.where(swap, va, vb)
    # dir_sign == +1 if face half-edge goes eu -> ev, else -1
    dsign = np.where(swap, -1, 1)

    order = np.lexsort((ev, eu))
    eu_s = eu[order]
    ev_s = ev[order]
    f_s = f_id[order]
    d_s = dsign[order]

    diff = np.empty(len(eu_s), dtype=bool)
    diff[0] = True
    if len(eu_s) > 1:
        diff[1:] = (eu_s[1:] != eu_s[:-1]) | (ev_s[1:] != ev_s[:-1])
    starts = np.flatnonzero(diff)
    counts = np.diff(np.append(starts, len(eu_s)))
    deg2 = starts[counts == 2]

    adj_faces: list[list[tuple[int, int]]] = [[] for _ in range(num_f)]
    for st in deg2:
        fa, fb = int(f_s[st]), int(f_s[st + 1])
        # Same sign wrt undirected edge means one face must flip relative to the other.
        same_dir = 1 if d_s[st] == d_s[st + 1] else 0
        adj_faces[fa].append((fb, same_dir))
        adj_faces[fb].append((fa, same_dir))

    visited = np.zeros(num_f, dtype=bool)
    flip_flag = np.zeros(num_f, dtype=bool)
    diag = _bbox_diag(verts)
    vol_eps = 1e-12 * (diag ** 3) if diag > 0.0 else 1e-15

    for seed in range(num_f):
        if visited[seed]:
            continue
        comp: list[int] = []
        queue: deque[int] = deque([seed])
        visited[seed] = True
        flip_flag[seed] = False
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            cur_flip = bool(flip_flag[cur])
            for nb, need_rel_flip in adj_faces[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    flip_flag[nb] = cur_flip ^ bool(need_rel_flip)
                    queue.append(nb)

        comp_arr = np.asarray(comp, dtype=np.int64)
        comp_flips = flip_flag[comp_arr]
        if np.any(comp_flips):
            to_flip = comp_arr[comp_flips]
            faces[to_flip] = faces[to_flip][:, [0, 2, 1]]

        # Check outward orientation via centered signed volume of the component.
        comp_faces = faces[comp_arr]
        used_v = np.unique(comp_faces.reshape(-1))
        center = verts[used_v].mean(axis=0)
        p0 = verts[comp_faces[:, 0]] - center
        p1 = verts[comp_faces[:, 1]] - center
        p2 = verts[comp_faces[:, 2]] - center
        signed_vol = float(np.sum(np.einsum("ij,ij->i", p0, np.cross(p1 - p0, p2 - p0))) / 6.0)
        if signed_vol < -vol_eps:
            faces[comp_arr] = comp_faces[:, [0, 2, 1]]

    return faces


def _split_collinear_t_junction(
    verts: np.ndarray,
    faces: np.ndarray,
    vs: list[int],
    area_thresh: float,
) -> tuple[np.ndarray, bool]:
    """Resolve a collinear pinhole/T-junction loop by splitting the spanning face edge."""
    pts = verts[np.asarray(vs, dtype=np.int64)]
    # Find the two extreme endpoints (p, q) of the collinear segment.
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    i_max, j_max = np.unravel_index(int(np.argmax(dists)), dists.shape)
    p, q = vs[int(i_max)], vs[int(j_max)]
    if p == q:
        return faces, False

    pq_vec = verts[q] - verts[p]
    pq_len2 = float(np.dot(pq_vec, pq_vec))
    if pq_len2 <= 1e-24:
        return faces, False

    # Verify all vertices in `vs` lie collinearly on segment (p, q) and pick an interior vertex `mid`.
    best_mid = -1
    best_dist = -1.0
    for idx, vid in enumerate(vs):
        if vid == p or vid == q:
            continue
        rel = verts[vid] - verts[p]
        t = float(np.dot(rel, pq_vec)) / pq_len2
        perp = float(np.linalg.norm(np.cross(rel, pq_vec))) / math.sqrt(pq_len2)
        if perp > max(1e-7, 1e-4 * math.sqrt(pq_len2)):
            return faces, False
        if 0.0 < t < 1.0:
            score = min(t, 1.0 - t)
            if score > best_dist:
                best_dist = score
                best_mid = vid

    if best_mid < 0:
        return faces, False

    # Locate the unique boundary face containing edge (p, q) and split it at `best_mid`.
    cand_faces = np.flatnonzero(np.any(faces == p, axis=1) & np.any(faces == q, axis=1))
    for fi in cand_faces:
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        tri = (a, b, c)
        # Preserve cyclic winding order of face `fi`.
        for k in range(3):
            u, v, opp = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            if (u == p and v == q) or (u == q and v == p):
                if opp == best_mid:
                    return faces, False
                # Verify resulting split triangles have non-degenerate area.
                a1 = 0.5 * float(np.linalg.norm(np.cross(verts[best_mid] - verts[u], verts[opp] - verts[u])))
                a2 = 0.5 * float(np.linalg.norm(np.cross(verts[v] - verts[best_mid], verts[opp] - verts[best_mid])))
                if a1 <= area_thresh or a2 <= area_thresh:
                    return faces, False
                faces[fi] = [u, best_mid, opp]
                faces = np.vstack([faces, np.array([[best_mid, v, opp]], dtype=np.int64)])
                return faces, True
    return faces, False


def _fill_pinholes(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    max_perimeter_ratio: float = 0.03,
    max_vertices: int = 16,
    area_eps_rel: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fill small O-Voxel residual pinhole cycles while preserving large cavity openings."""
    if len(faces) == 0 or max_perimeter_ratio <= 0.0 or max_vertices < 3:
        return verts, faces, 0

    diag = _bbox_diag(verts)
    area_thresh = max(1e-14, float(area_eps_rel) * (diag * diag))

    # First resolve any collinear T-junction pinholes by splitting the spanning face edge.
    t_junctions_fixed = 0
    for _ in range(128):
        loops = trace_boundary_loops(
            verts,
            faces,
            pinhole_max_perimeter_ratio=max_perimeter_ratio,
            pinhole_max_vertices=max_vertices,
            diag=diag,
        )
        collinear_fixed_any = False
        for lp in loops:
            if not lp.is_pinhole:
                continue
            vs = lp.vertices
            if len(vs) == 3:
                tri_area = 0.5 * float(
                    np.linalg.norm(np.cross(verts[vs[1]] - verts[vs[0]], verts[vs[2]] - verts[vs[0]]))
                )
                if tri_area <= area_thresh:
                    faces, did_split = _split_collinear_t_junction(verts, faces, vs, area_thresh)
                    if did_split:
                        t_junctions_fixed += 1
                        collinear_fixed_any = True
                        break
            else:
                faces, did_split = _split_collinear_t_junction(verts, faces, vs, area_thresh)
                if did_split:
                    t_junctions_fixed += 1
                    collinear_fixed_any = True
                    break
        if not collinear_fixed_any:
            break

    loops = trace_boundary_loops(
        verts,
        faces,
        pinhole_max_perimeter_ratio=max_perimeter_ratio,
        pinhole_max_vertices=max_vertices,
        diag=diag,
    )
    pinholes = [lp for lp in loops if lp.is_pinhole]
    if not pinholes:
        return verts, faces, t_junctions_fixed

    # Collect original directed boundary half-edges so fill triangles oppose existing winding.
    _, _, counts, half_edges, edge_inverse = _undirected_edge_stats(faces)
    boundary_he = half_edges[(counts == 1)[edge_inverse]]
    directed_set = {(int(s), int(t)) for s, t in boundary_he}

    new_verts_list: list[np.ndarray] = []
    new_faces_list: list[list[int]] = []
    next_v_idx = len(verts)
    filled = t_junctions_fixed

    for lp in pinholes:
        vs = lp.vertices
        k = len(vs)
        if k < 3:
            continue
        fwd_hits = sum(1 for i in range(k) if (vs[i], vs[(i + 1) % k]) in directed_set)
        rev_hits = sum(1 for i in range(k) if (vs[(i + 1) % k], vs[i]) in directed_set)
        # If existing half-edges go vs[i] -> vs[i+1], fill triangles must traverse vs[i+1] -> vs[i].
        existing_forward = fwd_hits >= rev_hits

        if k == 3:
            a, b, c = vs[0], vs[1], vs[2]
            tri = [c, b, a] if existing_forward else [a, b, c]
            new_faces_list.append(tri)
            filled += 1
        else:
            centroid = verts[np.asarray(vs, dtype=np.int64)].mean(axis=0)
            c_idx = next_v_idx
            next_v_idx += 1
            new_verts_list.append(centroid)
            for i in range(k):
                u = vs[i]
                v = vs[(i + 1) % k]
                tri = [v, u, c_idx] if existing_forward else [u, v, c_idx]
                new_faces_list.append(tri)
            filled += 1

    if not new_faces_list:
        return verts, faces, filled

    if new_verts_list:
        verts = np.vstack([verts, np.asarray(new_verts_list, dtype=np.float64)])
    faces = np.vstack([faces, np.asarray(new_faces_list, dtype=np.int64)])
    return verts, faces, filled


def inplace_clean_mesh(
    mesh: trimesh.Trimesh,
    *,
    weld_epsilon_rel: float = 1e-5,
    area_eps_rel: float = 1e-12,
    min_component_area_rel: float = 1e-5,
    max_hole_perimeter_ratio: float = 0.03,
    max_hole_vertices: int = 16,
    unify_normals: bool = True,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Run Official Route 1 topology repair (`o_voxel remesh=False` / WeldVertices + FillHoles)."""
    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()

    verts, faces, n_welded = _weld_vertices(verts, faces, epsilon_rel=weld_epsilon_rel)
    verts, faces = _drop_bad_and_duplicate_faces(verts, faces, area_eps_rel=area_eps_rel)
    verts, faces = _repair_non_manifold(verts, faces, area_eps_rel=area_eps_rel)
    verts, faces = _remove_small_components(verts, faces, min_area_rel=min_component_area_rel)

    if unify_normals and len(faces) > 0:
        faces = _unify_face_orientations(verts, faces)

    verts, faces, n_filled = _fill_pinholes(
        verts,
        faces,
        max_perimeter_ratio=max_hole_perimeter_ratio,
        max_vertices=max_hole_vertices,
        area_eps_rel=area_eps_rel,
    )

    verts, faces = _drop_bad_and_duplicate_faces(verts, faces, area_eps_rel=area_eps_rel)
    if unify_normals and len(faces) > 0:
        faces = _unify_face_orientations(verts, faces)

    clean_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if len(clean_mesh.faces) > 0:
        clean_mesh.vertex_normals  # Recompute clean area-weighted vertex normals without welding.

    info = {
        "method": "inplace_clean",
        "welded_vertices": int(n_welded),
        "filled_pinholes": int(n_filled),
        "weld_epsilon_rel": float(weld_epsilon_rel),
        "min_component_area_rel": float(min_component_area_rel),
        "max_hole_perimeter_ratio": float(max_hole_perimeter_ratio),
        "max_hole_vertices": int(max_hole_vertices),
    }
    return clean_mesh, info


def inplace_clean_glb(
    input_glb: str | Path,
    output_glb: str | Path,
    *,
    weld_epsilon_rel: float = 1e-5,
    area_eps_rel: float = 1e-12,
    min_component_area_rel: float = 1e-5,
    max_hole_perimeter_ratio: float = 0.03,
    max_hole_vertices: int = 16,
    unify_normals: bool = True,
) -> tuple[MeshMetrics, MeshMetrics, dict[str, Any]]:
    """Load raw GLB (`process=False`), run inplace repair, export repaired GLB (`process=False`)."""
    raw_mesh = load_mesh_raw(input_glb)
    raw_metrics = compute_mesh_metrics(
        raw_mesh,
        pinhole_max_perimeter_ratio=max_hole_perimeter_ratio,
        pinhole_max_vertices=max_hole_vertices,
        area_eps_rel=area_eps_rel,
    )

    clean_mesh, info = inplace_clean_mesh(
        raw_mesh,
        weld_epsilon_rel=weld_epsilon_rel,
        area_eps_rel=area_eps_rel,
        min_component_area_rel=min_component_area_rel,
        max_hole_perimeter_ratio=max_hole_perimeter_ratio,
        max_hole_vertices=max_hole_vertices,
        unify_normals=unify_normals,
    )

    out_path = Path(output_glb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_mesh.export(str(out_path))

    clean_metrics = compute_mesh_metrics(
        out_path,
        pinhole_max_perimeter_ratio=max_hole_perimeter_ratio,
        pinhole_max_vertices=max_hole_vertices,
        area_eps_rel=area_eps_rel,
    )
    return raw_metrics, clean_metrics, info
