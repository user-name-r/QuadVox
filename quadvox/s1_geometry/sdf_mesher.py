"""Generic Kuhn 6-tetrahedra Marching Tetrahedra SDF mesher for thin watertight geometry."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import trimesh

TETS = np.array(
    [
        [0, 1, 3, 7],
        [0, 1, 5, 7],
        [0, 4, 5, 7],
        [0, 4, 6, 7],
        [0, 2, 6, 7],
        [0, 2, 3, 7],
    ],
    dtype=np.int64,
)

CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.int64,
)

CASE_EDGES: dict[int, list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]] = {
    1: [((0, 1), (0, 2), (0, 3))],
    14: [((0, 1), (0, 3), (0, 2))],
    2: [((1, 0), (1, 3), (1, 2))],
    13: [((1, 0), (1, 2), (1, 3))],
    4: [((2, 0), (2, 1), (2, 3))],
    11: [((2, 0), (2, 3), (2, 1))],
    8: [((3, 0), (3, 2), (3, 1))],
    7: [((3, 0), (3, 1), (3, 2))],
    3: [((0, 2), (0, 3), (1, 3)), ((0, 2), (1, 3), (1, 2))],
    12: [((0, 2), (1, 3), (0, 3)), ((0, 2), (1, 2), (1, 3))],
    5: [((0, 1), (2, 3), (0, 3)), ((0, 1), (2, 1), (2, 3))],
    10: [((0, 1), (0, 3), (2, 3)), ((0, 1), (2, 3), (2, 1))],
    6: [((1, 0), (2, 0), (2, 3)), ((1, 0), (2, 3), (1, 3))],
    9: [((1, 0), (2, 3), (2, 0)), ((1, 0), (1, 3), (2, 3))],
}


def extract_marching_tetrahedra(
    sdf_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    grid_shape: tuple[int, int, int] = (96, 96, 36),
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a watertight, manifold triangle mesh from an arbitrary SDF using Kuhn 6-tets."""
    nx, ny, nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
    xs = np.linspace(float(bounds_min[0]), float(bounds_max[0]), nx)
    ys = np.linspace(float(bounds_min[1]), float(bounds_max[1]), ny)
    zs = np.linspace(float(bounds_min[2]), float(bounds_max[2]), nz)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    F = np.asarray(sdf_fn(X, Y, Z), dtype=np.float64)

    coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    f_flat = F.reshape(-1)
    cx, cy, cz = nx - 1, ny - 1, nz - 1
    ix = np.arange(cx)[:, None, None]
    iy = np.arange(cy)[None, :, None]
    iz = np.arange(cz)[None, None, :]
    stride_x = ny * nz
    stride_y = nz

    c_idx = np.empty((cx, cy, cz, 8), dtype=np.int64)
    for k, (dx, dy, dz) in enumerate(CORNERS):
        c_idx[..., k] = (ix + dx) * stride_x + (iy + dy) * stride_y + (iz + dz)
    c_idx = c_idx.reshape(-1, 8)
    tets = c_idx[:, TETS].reshape(-1, 4)

    ft = f_flat[tets]
    inside = ft <= 0.0
    mask = (
        inside[:, 0].astype(np.int64) * 1
        + inside[:, 1].astype(np.int64) * 2
        + inside[:, 2].astype(np.int64) * 4
        + inside[:, 3].astype(np.int64) * 8
    )

    edge_to_vid: dict[tuple[int, int], int] = {}
    verts_list: list[np.ndarray] = []
    faces_list: list[tuple[int, int, int]] = []

    def get_edge_vid(u: int, v: int) -> int:
        key = (u, v) if u < v else (v, u)
        vid = edge_to_vid.get(key)
        if vid is not None:
            return vid
        fu = float(f_flat[u])
        fv = float(f_flat[v])
        denom = fu - fv
        t = 0.5 if abs(denom) < 1e-14 else float(np.clip(fu / denom, 1e-4, 1.0 - 1e-4))
        p = coords[u] * (1.0 - t) + coords[v] * t
        vid = len(verts_list)
        verts_list.append(p)
        edge_to_vid[key] = vid
        return vid

    active_ids = np.flatnonzero((mask != 0) & (mask != 15))
    for tid in active_ids:
        t_verts = tets[tid]
        m = int(mask[tid])
        for (a0, a1), (b0, b1), (c0, c1) in CASE_EDGES[m]:
            va = get_edge_vid(int(t_verts[a0]), int(t_verts[a1]))
            vb = get_edge_vid(int(t_verts[b0]), int(t_verts[b1]))
            vc = get_edge_vid(int(t_verts[c0]), int(t_verts[c1]))
            if va != vb and vb != vc and va != vc:
                faces_list.append((va, vb, vc))

    verts = np.asarray(verts_list, dtype=np.float64)
    faces = np.asarray(faces_list, dtype=np.int64)
    if len(faces) == 0:
        return verts, faces

    # Ensure outward winding matches positive SDF gradient.
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cen = tri.mean(axis=1)
    eps = 1e-4
    gx = (sdf_fn(cen[:, 0] + eps, cen[:, 1], cen[:, 2]) - sdf_fn(cen[:, 0] - eps, cen[:, 1], cen[:, 2])) / (2.0 * eps)
    gy = (sdf_fn(cen[:, 0], cen[:, 1] + eps, cen[:, 2]) - sdf_fn(cen[:, 0], cen[:, 1] - eps, cen[:, 2])) / (2.0 * eps)
    gz = (sdf_fn(cen[:, 0], cen[:, 1], cen[:, 2] + eps) - sdf_fn(cen[:, 0], cen[:, 1], cen[:, 2] - eps)) / (2.0 * eps)
    grad = np.stack([gx, gy, gz], axis=1)
    if float(np.sum(fn * grad)) < 0.0:
        faces = faces[:, [0, 2, 1]]
    return verts, faces


def build_sdf_mesh(
    sdf_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    *,
    grid_shape: tuple[int, int, int] = (96, 96, 36),
    target_faces: int | None = None,
    taubin_iters: int = 8,
    newton_iters: int = 3,
) -> trimesh.Trimesh:
    """Extract, optionally decimate/smooth, and Newton-project a watertight SDF mesh."""
    verts, faces = extract_marching_tetrahedra(sdf_fn, bounds_min, bounds_max, grid_shape=grid_shape)
    if target_faces is not None and len(faces) > int(target_faces):
        try:
            import open3d as o3d  # type: ignore[import-not-found]

            o3_m = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts),
                o3d.utility.Vector3iVector(faces),
            )
            o3_m = o3_m.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
            if taubin_iters > 0:
                o3_m = o3_m.filter_smooth_taubin(number_of_iterations=int(taubin_iters))
            verts = np.asarray(o3_m.vertices, dtype=np.float64)
            faces = np.asarray(o3_m.triangles, dtype=np.int64)
        except ImportError:
            from quadvox.s2_retopo.qem import decimate_pyfqmr

            verts, faces = decimate_pyfqmr(verts, faces, int(target_faces))

    if newton_iters > 0 and len(verts) > 0:
        eps = 1e-4
        for _ in range(int(newton_iters)):
            val = np.asarray(sdf_fn(verts[:, 0], verts[:, 1], verts[:, 2]), dtype=np.float64)
            gx = (sdf_fn(verts[:, 0] + eps, verts[:, 1], verts[:, 2]) - sdf_fn(verts[:, 0] - eps, verts[:, 1], verts[:, 2])) / (2.0 * eps)
            gy = (sdf_fn(verts[:, 0], verts[:, 1] + eps, verts[:, 2]) - sdf_fn(verts[:, 0], verts[:, 1] - eps, verts[:, 2])) / (2.0 * eps)
            gz = (sdf_fn(verts[:, 0], verts[:, 1], verts[:, 2] + eps) - sdf_fn(verts[:, 0], verts[:, 1], verts[:, 2] - eps)) / (2.0 * eps)
            g = np.stack([gx, gy, gz], axis=1)
            g2 = np.sum(g * g, axis=1)
            step = np.clip(val / np.maximum(g2, 1e-8), -0.02, 0.02)
            verts = verts - step[:, None] * g

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    trimesh.repair.fix_normals(mesh)
    return mesh
