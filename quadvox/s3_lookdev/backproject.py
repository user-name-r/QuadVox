"""Generic Two-Band multi-view orthographic backprojection, 3D harmonic hole fill, and UV rasterization.

Supports:
- Layer 1 (Universal Core): arbitrary UV-mapped GLB/OBJ meshes with or without semantic labels,
  dual-frequency Two-Band blending, E2 blind-spot auxiliary view gating, 3D welded surface
  harmonic hole filling across UV seams, and exact EDT gutter padding.
- Layer 2 (Configurable ROI Lock): optional `roi_boxes` (`ROISovereigntyBox`) or explicit
  `sovereignty_mask` locking high-priority front regions (e.g., cockpit, logo, face) to their
  canonical view without hardcoding any category-specific coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from quadvox.contracts import ROISovereigntyBox
from quadvox.s0_views.camera import (
    MattedView,
    OrthoCameraSpec,
    dilate_nearest_texel,
    render_mesh_ortho_zbuf,
)

_VIEW_AZIM_MAP: dict[str, float] = {
    "front": 0.0,
    "left": 90.0,
    "back": 180.0,
    "right": 270.0,
}

_VIEW_DIR_MAP: dict[str, tuple[float, float, float]] = {
    "front": (0.0, 0.0, 1.0),
    "back": (0.0, 0.0, -1.0),
    "left": (1.0, 0.0, 0.0),
    "right": (-1.0, 0.0, 0.0),
}


def _smoothstep(e0: float, e1: float, x: np.ndarray | float) -> np.ndarray | float:
    lo, hi = float(e0), float(e1)
    if lo > hi:
        t = np.clip((x - hi) / max(lo - hi, 1e-8), 0.0, 1.0)
        return 1.0 - (t * t * (3.0 - 2.0 * t))
    t = np.clip((x - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    vn = np.zeros_like(verts, dtype=np.float64)
    for c in range(3):
        np.add.at(vn[:, c], faces[:, 0], fn[:, c])
        np.add.at(vn[:, c], faces[:, 1], fn[:, c])
        np.add.at(vn[:, c], faces[:, 2], fn[:, c])
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    return (vn / np.maximum(norms, 1e-12)).astype(np.float32)


def load_uv_mesh_and_labels(
    mesh_path: str | Path,
    *,
    labels_npz_path: str | Path | None = None,
    labels_source_mesh_path: str | Path | None = None,
    is_prop: bool = False,
) -> dict[str, Any]:
    """Load an unwrapped GLB/OBJ mesh and optional per-face semantic IDs."""
    import trimesh

    mp = Path(mesh_path)
    loaded = trimesh.load(str(mp), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"Scene contains no meshes: {mp}")
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    else:
        mesh = loaded

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if getattr(mesh.visual, "uv", None) is None:
        raise ValueError(f"Mesh has no UV coordinates (TEXCOORD_0): {mp}")
    uvs = np.asarray(mesh.visual.uv, dtype=np.float32)
    if len(uvs) != len(verts):
        raise ValueError(f"UV count {len(uvs)} != vertex count {len(verts)} in {mp}")

    try:
        vnorms = np.asarray(mesh.vertex_normals, dtype=np.float32)
        if vnorms.shape != verts.shape or not np.all(np.isfinite(vnorms)):
            vnorms = _compute_vertex_normals(verts.astype(np.float64), faces)
    except Exception:
        vnorms = _compute_vertex_normals(verts.astype(np.float64), faces)
    vnorms /= np.maximum(np.linalg.norm(vnorms, axis=1, keepdims=True), 1e-8)

    face_centroids = (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) / 3.0
    face_labels: np.ndarray | None = None

    if not is_prop and labels_npz_path is not None and Path(labels_npz_path).is_file():
        npz = np.load(str(labels_npz_path), allow_pickle=True)
        fl = npz["face_labels"] if "face_labels" in npz else None
        vl = npz["vertex_labels"] if "vertex_labels" in npz else (npz["labels"] if "labels" in npz else None)
        if fl is not None and len(fl) == len(faces):
            face_labels = np.asarray(fl, dtype=np.int32)
        elif vl is not None and len(vl) == len(verts):
            f_v = np.asarray(vl, dtype=np.int32)[faces]
            out_fl = np.zeros(len(faces), dtype=np.int32)
            for i in range(len(faces)):
                vals, cnts = np.unique(f_v[i], return_counts=True)
                out_fl[i] = int(vals[np.argmax(cnts)])
            face_labels = out_fl
        elif labels_source_mesh_path is not None and Path(labels_source_mesh_path).is_file():
            src = trimesh.load(str(labels_source_mesh_path), force="mesh", process=False)
            sv = np.asarray(src.vertices, dtype=np.float32)
            sf = np.asarray(src.faces, dtype=np.int64)
            if fl is not None and len(fl) == len(sf):
                sc = (sv[sf[:, 0]] + sv[sf[:, 1]] + sv[sf[:, 2]]) / 3.0
                _, idx = cKDTree(sc).query(face_centroids, k=1)
                face_labels = np.asarray(fl, dtype=np.int32)[idx]
            elif vl is not None and len(vl) == len(sv):
                _, idx = cKDTree(sv).query(face_centroids, k=1)
                face_labels = np.asarray(vl, dtype=np.int32)[idx]

    if face_labels is None:
        face_labels = np.zeros(len(faces), dtype=np.int32)

    return {
        "mesh": mesh,
        "verts": verts,
        "faces": faces,
        "uvs": uvs,
        "vnorms": vnorms,
        "face_labels": face_labels,
        "face_centroids": face_centroids,
        "is_prop": bool(is_prop),
    }


def rasterize_uv_attributes(
    verts: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    vnorms: np.ndarray,
    face_labels: np.ndarray,
    *,
    tex_size: int = 1024,
) -> dict[str, np.ndarray]:
    """Rasterize 3D position, normal, semantic class ID, and face index into UV space (V-up -> row-down)."""
    S = int(tex_size)
    valid_mask = np.zeros((S, S), dtype=bool)
    texel_pos = np.zeros((S, S, 3), dtype=np.float32)
    texel_nrm = np.zeros((S, S, 3), dtype=np.float32)
    texel_cls = np.zeros((S, S), dtype=np.uint8)
    texel_face_idx = np.full((S, S), -1, dtype=np.int32)

    u_px = uvs[:, 0] * float(S) - 0.5
    v_px = (1.0 - uvs[:, 1]) * float(S) - 0.5

    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
    ax, bx, cx = u_px[f0], u_px[f1], u_px[f2]
    ay, by, cy = v_px[f0], v_px[f1], v_px[f2]

    denoms = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    min_x = np.clip(np.floor(np.minimum(np.minimum(ax, bx), cx)).astype(np.int32), 0, S - 1)
    max_x = np.clip(np.ceil(np.maximum(np.maximum(ax, bx), cx)).astype(np.int32), 0, S - 1)
    min_y = np.clip(np.floor(np.minimum(np.minimum(ay, by), cy)).astype(np.int32), 0, S - 1)
    max_y = np.clip(np.ceil(np.maximum(np.maximum(ay, by), cy)).astype(np.int32), 0, S - 1)

    eps = -1e-4
    active = np.flatnonzero(np.abs(denoms) > 1e-8)
    for fi in active:
        x0, x1 = int(min_x[fi]), int(max_x[fi])
        y0, y1 = int(min_y[fi]), int(max_y[fi])
        if x1 < x0 or y1 < y0:
            continue
        ax_i, ay_i = float(ax[fi]), float(ay[fi])
        bx_i, by_i = float(bx[fi]), float(by[fi])
        cx_i, cy_i = float(cx[fi]), float(cy[fi])
        inv_d = 1.0 / float(denoms[fi])

        xs = np.arange(x0, x1 + 1, dtype=np.float64)[None, :]
        ys = np.arange(y0, y1 + 1, dtype=np.float64)[:, None]
        w0 = ((by_i - cy_i) * (xs - cx_i) + (cx_i - bx_i) * (ys - cy_i)) * inv_d
        w1 = ((cy_i - ay_i) * (xs - cx_i) + (ax_i - cx_i) * (ys - cy_i)) * inv_d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not np.any(inside):
            continue

        ry, rx = np.where(inside)
        gy, gx = y0 + ry, x0 + rx
        bw0 = w0[ry, rx][:, None].astype(np.float32)
        bw1 = w1[ry, rx][:, None].astype(np.float32)
        bw2 = w2[ry, rx][:, None].astype(np.float32)

        i0, i1, i2 = int(f0[fi]), int(f1[fi]), int(f2[fi])
        texel_pos[gy, gx] = bw0 * verts[i0] + bw1 * verts[i1] + bw2 * verts[i2]
        nrm_i = bw0 * vnorms[i0] + bw1 * vnorms[i1] + bw2 * vnorms[i2]
        nrm_i /= np.maximum(np.linalg.norm(nrm_i, axis=1, keepdims=True), 1e-8)
        texel_nrm[gy, gx] = nrm_i
        texel_cls[gy, gx] = int(face_labels[fi])
        texel_face_idx[gy, gx] = fi
        valid_mask[gy, gx] = True

    return {
        "valid_mask": valid_mask,
        "texel_pos": texel_pos,
        "texel_nrm": texel_nrm,
        "texel_cls": texel_cls,
        "texel_face_idx": texel_face_idx,
    }


def _solve_harmonic_subset(
    verts: np.ndarray,
    sub_faces: np.ndarray,
    sub_face_rgb_sum: np.ndarray,
    sub_face_cnt: np.ndarray,
    texel_pos_sub: np.ndarray,
    uf_sub_local: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Solve welded-vertex Dirichlet harmonic interpolation on a single face subset."""
    used_v, inv_sub_f = np.unique(sub_faces.ravel(), return_inverse=True)
    sub_faces_local = inv_sub_f.reshape(-1, 3)
    sub_verts = verts[used_v]
    n_sv = len(sub_verts)

    vert_rgb_sum = np.zeros((n_sv, 3), dtype=np.float64)
    vert_cnt = np.zeros(n_sv, dtype=np.float64)
    has_f = sub_face_cnt > 0
    f_obs = sub_faces_local[has_f]
    fc_obs = sub_face_rgb_sum[has_f] / sub_face_cnt[has_f, None]
    for corner in range(3):
        for c in range(3):
            np.add.at(vert_rgb_sum[:, c], f_obs[:, corner], fc_obs[:, c])
        np.add.at(vert_cnt, f_obs[:, corner], 1.0)

    quant = np.rint(sub_verts.astype(np.float64) * 1e5).astype(np.int64)
    _, inv_weld = np.unique(quant, axis=0, return_inverse=True)
    n_w = int(inv_weld.max()) + 1
    w_rgb_sum = np.zeros((n_w, 3), dtype=np.float64)
    for c in range(3):
        w_rgb_sum[:, c] = np.bincount(inv_weld, weights=vert_rgb_sum[:, c], minlength=n_w)
    w_cnt = np.bincount(inv_weld, weights=vert_cnt, minlength=n_w)
    w_known = w_cnt > 0
    w_val = np.zeros((n_w, 3), dtype=np.float64)
    w_val[w_known] = w_rgb_sum[w_known] / w_cnt[w_known, None]

    wf = inv_weld[sub_faces_local]
    edges = np.vstack([
        np.stack([wf[:, 0], wf[:, 1]], axis=1),
        np.stack([wf[:, 1], wf[:, 2]], axis=1),
        np.stack([wf[:, 2], wf[:, 0]], axis=1),
    ])
    edges = np.sort(edges, axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(edges, axis=0)

    unknown_idx = np.flatnonzero(~w_known)
    if len(unknown_idx) > 0:
        w_pos = np.zeros((n_w, 3), dtype=np.float64)
        w_pos_cnt = np.bincount(inv_weld, minlength=n_w)
        for c in range(3):
            w_pos[:, c] = np.bincount(inv_weld, weights=sub_verts[:, c], minlength=n_w) / np.maximum(w_pos_cnt, 1)

        if len(edges) > 0:
            elen = np.linalg.norm(w_pos[edges[:, 0]] - w_pos[edges[:, 1]], axis=1)
            ew = 1.0 / np.maximum(elen, 1e-4)
            rows = np.concatenate([edges[:, 0], edges[:, 1]])
            cols = np.concatenate([edges[:, 1], edges[:, 0]])
            vals = np.concatenate([ew, ew])
            adj = coo_matrix((vals, (rows, cols)), shape=(n_w, n_w)).tocsr()
        else:
            adj = coo_matrix((n_w, n_w), dtype=np.float64).tocsr()

        n_comp, comp_ids = connected_components(adj, directed=False)
        comp_has_known = np.zeros(n_comp, dtype=bool)
        if np.any(w_known):
            np.logical_or.at(comp_has_known, comp_ids[w_known], True)
        unseeded_mask = (~w_known) & (~comp_has_known[comp_ids])
        if np.any(unseeded_mask):
            w_val[unseeded_mask] = w_val[w_known].mean(axis=0)
        w_fixed = w_known | unseeded_mask
        is_unk = ~w_fixed

        if np.any(is_unk):
            deg = np.asarray(adj.sum(axis=1)).ravel()
            L_full = coo_matrix((deg, (np.arange(n_w), np.arange(n_w))), shape=(n_w, n_w)).tocsr() - adj
            mask_unk_diag = coo_matrix((is_unk.astype(np.float64), (np.arange(n_w), np.arange(n_w))), shape=(n_w, n_w))
            mask_kno_diag = coo_matrix((w_fixed.astype(np.float64), (np.arange(n_w), np.arange(n_w))), shape=(n_w, n_w))
            A_sys = (mask_unk_diag @ L_full + mask_kno_diag).tocsc()
            rhs = np.where(w_fixed[:, None], w_val, 0.0)
            solved = spsolve(A_sys, rhs)
            if solved.ndim == 1:
                solved = solved[:, None]
            w_solved = np.clip(solved, 0.0, 1.0).astype(np.float32)
        else:
            w_solved = np.clip(w_val, 0.0, 1.0).astype(np.float32)
    else:
        w_solved = np.clip(w_val, 0.0, 1.0).astype(np.float32)

    v_solved = w_solved[inv_weld]
    f_verts = sub_faces_local[uf_sub_local]
    d0 = 1.0 / np.maximum(np.linalg.norm(texel_pos_sub - sub_verts[f_verts[:, 0]], axis=1), 1e-5)
    d1 = 1.0 / np.maximum(np.linalg.norm(texel_pos_sub - sub_verts[f_verts[:, 1]], axis=1), 1e-5)
    d2 = 1.0 / np.maximum(np.linalg.norm(texel_pos_sub - sub_verts[f_verts[:, 2]], axis=1), 1e-5)
    ds = d0 + d1 + d2
    col_fill = (
        (d0 / ds)[:, None] * v_solved[f_verts[:, 0]]
        + (d1 / ds)[:, None] * v_solved[f_verts[:, 1]]
        + (d2 / ds)[:, None] * v_solved[f_verts[:, 2]]
    )
    return np.clip(col_fill, 0.0, 1.0).astype(np.float32), int(len(unknown_idx))


def mesh_harmonic_fill(
    verts: np.ndarray,
    faces: np.ndarray,
    texel_pos: np.ndarray,
    texel_face_idx: np.ndarray,
    valid_mask: np.ndarray,
    albedo_uv: np.ndarray,
    observed_mask: np.ndarray,
    face_labels: np.ndarray | None = None,
    empty_label_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fill unseen UV texels via 3D welded mesh surface harmonic interpolation.

    When `face_labels` is provided (optional per-face semantic ID array of length `len(faces)`),
    harmonic solves and vertex welding are isolated strictly within each semantic label so
    Laplace edges never cross semantic boundaries and empty-observed classes fall back cleanly
    to `empty_label_rgb` without cross-region color bleed.
    """
    unseen_uv = valid_mask & (~observed_mask)
    if not np.any(unseen_uv):
        return albedo_uv, {"unseen_texels": 0, "method": "none"}

    n_f = len(faces)
    fl: np.ndarray | None = None
    if face_labels is not None:
        fl = np.asarray(face_labels)
        if fl.shape != (n_f,):
            raise ValueError(f"face_labels shape {fl.shape} does not match faces count ({n_f},)")

    obs_texel_frac = float(np.sum(observed_mask)) / max(float(np.sum(valid_mask)), 1.0)

    f_idx_obs = texel_face_idx[observed_mask]
    rgb_obs = albedo_uv[observed_mask]
    valid_obs = (f_idx_obs >= 0) & (f_idx_obs < n_f)
    f_idx_obs = f_idx_obs[valid_obs]
    rgb_obs = rgb_obs[valid_obs]

    face_rgb_sum = np.zeros((n_f, 3), dtype=np.float64)
    for c in range(3):
        face_rgb_sum[:, c] = np.bincount(f_idx_obs, weights=rgb_obs[:, c], minlength=n_f)
    face_cnt = np.bincount(f_idx_obs, minlength=n_f).astype(np.float64)

    if fl is None and not np.any(face_cnt > 0):
        return albedo_uv, {"unseen_texels": int(np.sum(unseen_uv)), "method": "fallback_empty"}

    out = albedo_uv.copy()
    uy, ux = np.where(unseen_uv)
    uf = texel_face_idx[uy, ux]
    valid_f = (uf >= 0) & (uf < n_f)
    uy, ux, uf = uy[valid_f], ux[valid_f], uf[valid_f]

    total_unknown_verts = 0
    empty_observed_labels: list[int] = []
    if len(uf) > 0:
        tp = texel_pos[uy, ux]
        if fl is None:
            col_fill, n_unk = _solve_harmonic_subset(verts, faces, face_rgb_sum, face_cnt, tp, uf)
            out[uy, ux] = col_fill
            total_unknown_verts = n_unk
        else:
            unseen_labels = fl[uf]
            neutral = np.asarray(empty_label_rgb, dtype=np.float32)
            for lbl in np.unique(unseen_labels):
                u_mask = unseen_labels == lbl
                lbl_faces = np.flatnonzero(fl == lbl)
                sub_face_cnt = face_cnt[lbl_faces]
                if not np.any(sub_face_cnt > 0):
                    out[uy[u_mask], ux[u_mask]] = neutral
                    empty_observed_labels.append(int(lbl))
                    continue
                uf_local = np.searchsorted(lbl_faces, uf[u_mask])
                col_fill, n_unk = _solve_harmonic_subset(
                    verts,
                    faces[lbl_faces],
                    face_rgb_sum[lbl_faces],
                    sub_face_cnt,
                    tp[u_mask],
                    uf_local,
                )
                out[uy[u_mask], ux[u_mask]] = col_fill
                total_unknown_verts += n_unk

    return out, {
        "unseen_texels": int(np.sum(unseen_uv)),
        "observed_texel_frac": round(obs_texel_frac, 4),
        "unknown_welded_verts": int(total_unknown_verts),
        "label_isolated": fl is not None,
        "empty_observed_labels": empty_observed_labels,
        "method": "3d_welded_harmonic",
    }


def _bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    x_cl = np.clip(xs, 0.0, float(W - 1))
    y_cl = np.clip(ys, 0.0, float(H - 1))
    x0 = np.floor(x_cl).astype(np.int32)
    y0 = np.floor(y_cl).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    fx = (x_cl - x0)[:, None].astype(np.float32)
    fy = (y_cl - y0)[:, None].astype(np.float32)

    c00 = img[y0, x0]
    c10 = img[y0, x1]
    c01 = img[y1, x0]
    c11 = img[y1, x1]
    return ((1.0 - fy) * ((1.0 - fx) * c00 + fx * c10) + fy * ((1.0 - fx) * c01 + fx * c11)).astype(np.float32)


def _compute_texel_roi_masks(
    verts: np.ndarray,
    pts: np.ndarray,
    nrms: np.ndarray,
    roi_boxes: Sequence[ROISovereigntyBox],
) -> list[tuple[ROISovereigntyBox, np.ndarray]]:
    """Map Layer-2 `ROISovereigntyBox` configs into `[0, 1]` UV texel masks."""
    if not roi_boxes:
        return []
    y_min = float(verts[:, 1].min())
    y_max = float(verts[:, 1].max())
    y_span = max(y_max - y_min, 1e-6)
    s_to_full = 1.0 / y_span
    y_full = 1.0 - (y_max - pts[:, 1]) * s_to_full
    x_center = 0.5 * float(verts[:, 0].min() + verts[:, 0].max())
    x_full = (pts[:, 0] - x_center) * s_to_full

    out: list[tuple[ROISovereigntyBox, np.ndarray]] = []
    for box in roi_boxes:
        z_mid = 0.5 * float(verts[:, 2].min() + verts[:, 2].max())
        if box.anchor_z_box is not None:
            xr = box.anchor_z_box.get("x_range", (-1.0, 1.0))
            yr = box.anchor_z_box.get("y_range", (0.0, 1.0))
            v_yf = 1.0 - (y_max - verts[:, 1]) * s_to_full
            v_xf = (verts[:, 0] - x_center) * s_to_full
            core = (v_yf >= yr[0]) & (v_yf <= yr[1]) & (v_xf >= xr[0]) & (v_xf <= xr[1])
            if int(core.sum()) >= 32:
                z_mid = 0.5 * float(verts[core, 2].min() + verts[core, 2].max())
        z_rel = (pts[:, 2] - z_mid) * s_to_full

        m = max(float(box.margin), 1e-5)
        mx = _smoothstep(box.x_range[0] - m, box.x_range[0], x_full) * (
            1.0 - _smoothstep(box.x_range[1], box.x_range[1] + m, x_full)
        )
        my = _smoothstep(box.y_range[0] - m, box.y_range[0], y_full) * (
            1.0 - _smoothstep(box.y_range[1], box.y_range[1] + m, y_full)
        )
        mz = _smoothstep(box.z_range[0] - m, box.z_range[0], z_rel) * (
            1.0 - _smoothstep(box.z_range[1], box.z_range[1] + m, z_rel)
        )

        t_dir = np.asarray(_VIEW_DIR_MAP[box.target_view], dtype=np.float32)
        ndot_t = nrms @ t_dir
        n_gate = _smoothstep(box.normal_gate_range[0], box.normal_gate_range[1], ndot_t)
        mask = np.clip(mx * my * mz * n_gate, 0.0, 1.0).astype(np.float32)
        out.append((box, mask))
    return out


def bake_two_band_albedo(
    *,
    mesh_data: dict[str, Any],
    uv_attrs: dict[str, np.ndarray],
    views: Sequence[MattedView],
    cameras: Sequence[OrthoCameraSpec],
    detail_power: float = 32.0,
    low_power: float = 4.0,
    blur_sigma_px: float = 6.0,
    occl_tol_rel: float = 0.015,
    roi_boxes: Sequence[ROISovereigntyBox] = (),
    sovereignty_texel_mask: np.ndarray | None = None,
    sovereignty_view: str = "front",
    e2_aux_threshold: float = 0.50,
    run_mesh_fill: bool = True,
    run_gutter_pad: bool = True,
    face_labels: np.ndarray | None = None,
    isolate_harmonic_by_label: bool = True,
) -> dict[str, Any]:
    """Two-Band dual-frequency backprojection with optional Layer-2 ROI sovereignty lock,
    E2 blind-spot gating, 3D harmonic hole fill, and exact EDT gutter dilation.
    """
    if len(views) != len(cameras) or not views:
        raise ValueError("views and cameras must be non-empty and equal length")

    verts = mesh_data["verts"]
    faces = mesh_data["faces"]

    valid_mask = uv_attrs["valid_mask"]
    texel_pos = uv_attrs["texel_pos"]
    texel_nrm = uv_attrs["texel_nrm"]
    texel_face_idx = uv_attrs["texel_face_idx"]
    H, W = valid_mask.shape

    span = np.max(verts.max(axis=0) - verts.min(axis=0))
    occl_tol = float(occl_tol_rel) * max(float(span), 1e-4)

    vy, vx = np.where(valid_mask)
    pts = texel_pos[vy, vx]
    nrms = texel_nrm[vy, vx]
    n_tex = len(pts)

    roi_masks = _compute_texel_roi_masks(verts, pts, nrms, roi_boxes)
    explicit_sov = (
        np.clip(sovereignty_texel_mask[vy, vx].astype(np.float32), 0.0, 1.0)
        if sovereignty_texel_mask is not None
        else None
    )

    n_views = len(views)
    q_all = np.zeros((n_views, n_tex), dtype=np.float32)
    low_samples = np.zeros((n_views, n_tex, 3), dtype=np.float32)
    high_samples = np.zeros((n_views, n_tex, 3), dtype=np.float32)

    for k, (view, cam) in enumerate(zip(views, cameras)):
        zbuf_info = render_mesh_ortho_zbuf(verts, faces, cam)
        zbuf = zbuf_info["zbuf"]
        hit_mask = zbuf_info["hit_mask"]

        dist_edge = ndimage.distance_transform_edt(view.fg_mask & hit_mask).astype(np.float32)
        edge_fade = np.clip((dist_edge - 1.0) / 3.5, 0.0, 1.0)

        px_x, px_y, depth = (
            cam.cx_px + (pts @ cam.cam_right) * cam.sx_px,
            cam.cy_px - (pts @ cam.cam_up) * cam.sy_px,
            pts @ cam.cam_pos_dir,
        )
        in_bounds = (px_x >= 0.0) & (px_x <= cam.width - 1.0) & (px_y >= 0.0) & (px_y <= cam.height - 1.0)
        ix = np.clip(np.rint(px_x).astype(np.int32), 0, cam.width - 1)
        iy = np.clip(np.rint(px_y).astype(np.int32), 0, cam.height - 1)

        z_front = zbuf[iy, ix]
        occl = z_front - depth
        vis = np.where(in_bounds & (z_front > -1e8), 1.0 - _smoothstep(0.5 * occl_tol, occl_tol, occl), 0.0).astype(np.float32)

        ndot = np.clip(nrms @ cam.cam_pos_dir, 0.0, 1.0).astype(np.float32)
        ndot_eff = (ndot * _smoothstep(0.08, 0.35, ndot)).astype(np.float32)

        fg_val = view.alpha_soft[iy, ix]
        ef = edge_fade[iy, ix]
        q = vis * ndot_eff * fg_val * ef * float(cam.trust)

        # Layer-2 Configurable ROI Sovereignty Lock
        for box, r_mask in roi_masks:
            target_az = _VIEW_AZIM_MAP[box.target_view]
            az_diff = abs(((float(cam.azim_deg) - target_az + 180.0) % 360.0) - 180.0)
            is_target = bool(cam.is_primary and az_diff < 5.0 and abs(cam.elev_deg) < 5.0)
            if is_target:
                q = np.where(r_mask > 0.1, np.maximum(q, r_mask * vis), q)
            else:
                q = q * (1.0 - box.suppress_ortho_views * r_mask)

        if explicit_sov is not None:
            target_az = _VIEW_AZIM_MAP.get(sovereignty_view, 0.0)
            az_diff = abs(((float(cam.azim_deg) - target_az + 180.0) % 360.0) - 180.0)
            is_target = bool(cam.is_primary and az_diff < 5.0 and abs(cam.elev_deg) < 5.0)
            if is_target:
                q = np.where(explicit_sov > 0.1, np.maximum(q, explicit_sov * vis), q)
            else:
                q = q * (1.0 - 0.995 * explicit_sov)

        q_all[k] = np.clip(q, 0.0, 1.0)

        img_dil = view.rgb_dilated
        low_img = ndimage.gaussian_filter(img_dil, sigma=(float(blur_sigma_px), float(blur_sigma_px), 0.0))
        high_img = img_dil - low_img
        low_samples[k] = _bilinear_sample(low_img, px_x, px_y)
        high_samples[k] = _bilinear_sample(high_img, px_x, px_y)

    primary_idx = [k for k, c in enumerate(cameras) if c.is_primary]
    aux_idx = [k for k, c in enumerate(cameras) if not c.is_primary]

    if primary_idx:
        q_s0_max = np.max(q_all[primary_idx], axis=0)
    else:
        q_s0_max = np.zeros(n_tex, dtype=np.float32)

    if aux_idx:
        aux_gate = (1.0 - _smoothstep(0.12, float(e2_aux_threshold), q_s0_max)).astype(np.float32)
        for k in aux_idx:
            q_all[k] *= aux_gate

    q_max = np.max(q_all, axis=0)
    observed = q_max >= 0.03

    q_norm = q_all / np.maximum(q_max[None, :], 1e-8)
    w_low = np.power(np.maximum(q_norm, 0.0), float(low_power)) * (q_all > 1e-5)
    w_high = np.power(np.maximum(q_norm, 0.0), float(detail_power)) * (q_all > 1e-5)

    w_low_sum = np.sum(w_low, axis=0)
    w_high_sum = np.sum(w_high, axis=0)

    low_blend = np.sum(w_low[:, :, None] * low_samples, axis=0) / np.maximum(w_low_sum[:, None], 1e-8)
    high_blend = np.sum(w_high[:, :, None] * high_samples, axis=0) / np.maximum(w_high_sum[:, None], 1e-8)
    tex_rgb = np.clip(low_blend + high_blend, 0.0, 1.0).astype(np.float32)

    albedo_uv = np.zeros((H, W, 3), dtype=np.float32)
    albedo_uv[vy[observed], vx[observed]] = tex_rgb[observed]

    observed_mask = np.zeros((H, W), dtype=bool)
    observed_mask[vy[observed], vx[observed]] = True

    s0_conf_uv = np.zeros((H, W), dtype=np.float32)
    s0_conf_uv[vy, vx] = np.clip(q_s0_max, 0.0, 1.0)

    fill_stats: dict[str, Any] = {"unseen_texels": int(np.sum(valid_mask & ~observed_mask)), "method": "skipped"}
    if run_mesh_fill:
        resolved_face_labels: np.ndarray | None = None
        if isolate_harmonic_by_label:
            candidate_labels = face_labels if face_labels is not None else mesh_data.get("face_labels")
            if candidate_labels is not None:
                arr_labels = np.asarray(candidate_labels)
                if arr_labels.shape == (len(faces),):
                    resolved_face_labels = arr_labels
        albedo_uv, fill_stats = mesh_harmonic_fill(
            verts,
            faces,
            texel_pos,
            texel_face_idx,
            valid_mask,
            albedo_uv,
            observed_mask,
            face_labels=resolved_face_labels,
        )

    if run_gutter_pad:
        albedo_uv = dilate_nearest_texel(albedo_uv, valid_mask)

    return {
        "albedo": np.clip(albedo_uv, 0.0, 1.0).astype(np.float32),
        "observed_mask": observed_mask,
        "s0_confidence": s0_conf_uv,
        "observed_fraction": round(float(np.sum(observed_mask)) / max(float(np.sum(valid_mask)), 1.0), 4),
        "fill_stats": fill_stats,
    }
