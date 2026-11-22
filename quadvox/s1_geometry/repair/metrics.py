"""Deterministic mesh topology and boundary-loop metrics.

ALWAYS loads GLB/mesh assets with `trimesh.load(path, force="mesh", process=False)`
so raw vertices/edges are inspected honestly without silent welding or merging.

Verified upstream URLs:
- O-Voxel standard clean branch (`remesh=False`):
  https://github.com/microsoft/TRELLIS.2/blob/main/o-voxel/o_voxel/postprocess.py
- ComfyUI WeldVertices, FillHoles, RemeshMesh (`sign_mode="solid"`):
  https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_mesh_postprocess.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from quadvox.contracts import MeshMetrics


@dataclass(frozen=True)
class BoundaryLoop:
    """Single closed boundary edge cycle traced on a triangle mesh."""

    vertices: list[int]
    directed_edges: list[tuple[int, int]]
    perimeter: float
    is_pinhole: bool


def load_mesh_raw(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh without any implicit vertex welding, face deduplication, or reordering."""
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
            return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
        return trimesh.Trimesh(
            vertices=np.vstack(verts_parts),
            faces=np.vstack(faces_parts),
            process=False,
        )

    raise TypeError(f"Unsupported mesh type loaded from {src}: {type(loaded).__name__}")


def _extract_vf(mesh_or_path: str | Path | trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    mesh = load_mesh_raw(mesh_or_path) if isinstance(mesh_or_path, (str, Path)) else mesh_or_path
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[0] == 0:
        verts = np.zeros((0, 3), dtype=np.float64)
    if faces.ndim != 2 or faces.shape[0] == 0:
        faces = np.zeros((0, 3), dtype=np.int64)
    return verts, faces


def _undirected_edge_stats(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group all undirected face edges.

    Returns:
        unique_u: (E,) int64 smaller vertex endpoint per unique undirected edge
        unique_v: (E,) int64 larger vertex endpoint per unique undirected edge
        counts: (E,) int64 number of incident half-edges per unique undirected edge
        half_edges_dir: (3F, 2) int64 original directed half-edges (filtered to u != v)
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


def compute_corner_fan_components(num_verts: int, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Partition all 3*F face corners into edge-connected vertex-fan components.

    Two corners belonging to the same vertex are connected iff their faces share
    an edge incident to that vertex. A vertex with >1 fan component is non-manifold.

    Returns:
        corner_labels: (3F,) int64 component label for each corner in `faces.reshape(-1)`
        fan_counts: (V,) int64 number of disconnected fan components per vertex
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

    # Keep degenerate self-edges inside a face from creating artificial multi-fans.
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
    b_directed = half_edges[he_is_boundary]  # (B, 2) each row is (src, tgt) from its single face

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
        # Prefer continuing along the face half-edge direction when unvisited.
        for nxt, eid in out_directed.get(curr, ()):
            if not visited_edge[eid]:
                visited_edge[eid] = True
                return nxt, (curr, nxt)
        for nxt, eid in adj_undirected.get(curr, ()):
            if not visited_edge[eid]:
                visited_edge[eid] = True
                return nxt, (curr, nxt)
        return None

    # Walk open-degree-1 chains first so figure-8 or branched edges don't pollute closed cycles.
    degree_one_verts = [v for v, nbrs in adj_undirected.items() if len(nbrs) == 1]
    start_order = degree_one_verts + sorted(v for v in adj_undirected if len(adj_undirected[v]) != 1)

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


def compute_mesh_metrics(
    mesh_or_path: str | Path | trimesh.Trimesh,
    *,
    pinhole_max_perimeter_ratio: float = 0.03,
    pinhole_max_vertices: int = 16,
    area_eps_rel: float = 1e-12,
) -> MeshMetrics:
    """Compute deterministic topology and bounding-box metrics for a mesh or GLB path."""
    verts, faces = _extract_vf(mesh_or_path)
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

    dup_idx = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
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


def _project_mesh_silhouette_masks(
    mesh: trimesh.Trimesh,
    transforms_meta: dict[str, Any],
    *,
    mask_size: int = 256,
    n_samples: int = 400_000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Project a `y_up_z_front` mesh into Z-up NeRF `transforms.json` camera silhouettes."""
    import math
    from scipy import ndimage

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(verts) == 0 or len(faces) == 0:
        raise ValueError("Cannot compute silhouette IoU on empty mesh")

    rng = np.random.default_rng(seed)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total_area = float(areas.sum())
    if total_area > 0.0:
        probs = areas / total_area
        f_ids = rng.choice(len(faces), size=int(n_samples), replace=True, p=probs)
        r1 = np.sqrt(rng.random(int(n_samples)))
        r2 = rng.random(int(n_samples))
        w0 = 1.0 - r1
        w1 = r1 * (1.0 - r2)
        w2 = r1 * r2
        pts_yup = w0[:, None] * v0[f_ids] + w1[:, None] * v1[f_ids] + w2[:, None] * v2[f_ids]
        pts_yup = np.vstack([pts_yup, verts[:: max(1, len(verts) // 100_000)]])
    else:
        pts_yup = verts

    pts_zup = np.empty_like(pts_yup)
    pts_zup[:, 0] = pts_yup[:, 0]
    pts_zup[:, 1] = -pts_yup[:, 2]
    pts_zup[:, 2] = pts_yup[:, 1]

    top_fov = float(transforms_meta.get("camera_angle_x", math.radians(20.0)))
    struct = np.ones((3, 3), dtype=bool)
    masks: dict[str, np.ndarray] = {}

    for fr in transforms_meta["frames"]:
        c2w = np.asarray(fr["transform_matrix"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        fov_x = float(fr.get("camera_angle_x", top_fov))
        focal = (0.5 * float(mask_size)) / math.tan(0.5 * fov_x)

        cam_pts = pts_zup @ w2c[:3, :3].T + w2c[:3, 3]
        depth = -cam_pts[:, 2]
        valid = depth > 1e-4
        xc = cam_pts[valid, 0]
        yc = cam_pts[valid, 1]
        zc = depth[valid]

        us = np.rint(0.5 * float(mask_size) + focal * (xc / zc)).astype(np.int64)
        vs = np.rint(0.5 * float(mask_size) - focal * (yc / zc)).astype(np.int64)
        inside = (us >= 0) & (us < mask_size) & (vs >= 0) & (vs < mask_size)

        grid = np.zeros((mask_size, mask_size), dtype=bool)
        grid[vs[inside], us[inside]] = True
        grid = ndimage.binary_closing(grid, structure=struct)
        name = str(fr.get("name") or Path(fr["file_path"]).stem)
        masks[name] = grid

    return masks


def _tight_bbox_iou(m_a: np.ndarray, m_b: np.ndarray, norm_size: int = 256) -> float:
    from PIL import Image

    def _norm(m: np.ndarray) -> np.ndarray:
        ys, xs = np.where(m)
        if len(xs) == 0:
            return np.zeros((norm_size, norm_size), dtype=bool)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop = m[y0:y1, x0:x1]
        h, w = crop.shape
        scale = float(norm_size - 16) / max(h, w)
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        pil = Image.fromarray((crop.astype(np.uint8) * 255)).resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = np.zeros((norm_size, norm_size), dtype=bool)
        oy = (norm_size - nh) // 2
        ox = (norm_size - nw) // 2
        canvas[oy : oy + nh, ox : ox + nw] = np.asarray(pil) > 127
        return canvas

    ca = _norm(m_a)
    cb = _norm(m_b)
    inter = int(np.logical_and(ca, cb).sum())
    union = int(np.logical_or(ca, cb).sum())
    return float(inter) / max(union, 1)


def compute_multiview_silhouette_iou(
    mesh_or_path: str | Path | trimesh.Trimesh,
    views_dir: str | Path,
    *,
    mask_size: int = 256,
) -> dict[str, Any]:
    """Compute per-view & mean 4-view Silhouette IoU (calibrated + tight-bbox-normalized)."""
    import json
    from PIL import Image

    vdir = Path(views_dir)
    transforms_path = vdir / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Missing transforms.json in {vdir}")
    transforms_meta = json.loads(transforms_path.read_text(encoding="utf-8"))

    mesh = load_mesh_raw(mesh_or_path) if isinstance(mesh_or_path, (str, Path)) else mesh_or_path
    proj_masks = _project_mesh_silhouette_masks(mesh, transforms_meta, mask_size=mask_size)

    per_view: dict[str, dict[str, float]] = {}
    calib_ious: list[float] = []
    tight_ious: list[float] = []
    for fr in transforms_meta["frames"]:
        name = str(fr.get("name") or Path(fr["file_path"]).stem)
        img_path = vdir / fr["file_path"]
        with Image.open(img_path) as im:
            im.load()
            alpha = im.convert("RGBA").getchannel(3).resize((mask_size, mask_size), Image.Resampling.BILINEAR)
            ref_mask = np.asarray(alpha, dtype=np.uint8) >= 128

        pred_mask = proj_masks[name]
        inter = int(np.logical_and(ref_mask, pred_mask).sum())
        union = int(np.logical_or(ref_mask, pred_mask).sum())
        calib_iou = float(inter) / max(union, 1)
        tight_iou = _tight_bbox_iou(ref_mask, pred_mask, norm_size=mask_size)
        per_view[name] = {
            "calibrated_iou": round(calib_iou, 4),
            "tight_bbox_iou": round(tight_iou, 4),
        }
        calib_ious.append(calib_iou)
        tight_ious.append(tight_iou)

    return {
        "mask_size": int(mask_size),
        "mean_calibrated_iou": round(float(np.mean(calib_ious)), 4),
        "min_calibrated_iou": round(float(np.min(calib_ious)), 4),
        "mean_tight_bbox_iou": round(float(np.mean(tight_ious)), 4),
        "min_tight_bbox_iou": round(float(np.min(tight_ious)), 4),
        "per_view": per_view,
    }
