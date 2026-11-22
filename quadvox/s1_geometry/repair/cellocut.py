"""Official Route 2 adapter: ComfyUI `RemeshMesh` (`sign_mode="solid"`, CelloCut).

Wraps ComfyUI's `remesh_narrow_band_dc` kernel with exact bounding-box fitting
from `RemeshMesh.execute` so Colab GPU Phase 2 benchmarking can compare Route 1
(`inplace_clean`) vs Route 2 (`cellocut` solid remesh) on identical raw GLBs.

Fails cleanly with `CelloCutNotAvailableError` when ComfyUI or CUDA GPU is not
available locally (NO silent fallback or degradation).

Verified upstream URLs:
- ComfyUI `RemeshMesh` node:
  https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_mesh_postprocess.py
- ComfyUI `remesh_narrow_band_dc` (`sign_mode="solid"` / CelloCut):
  https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/mesh3d/postprocess/remesh.py
- ComfyUI official Pixal3D workflow template:
  https://github.com/Comfy-Org/workflow_templates/blob/main/templates/3d_pixal3d_multi_views.json
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from quadvox.contracts import MeshMetrics

from .inplace_clean import (
    _bbox_diag,
    _compact_unreferenced_vertices,
    _drop_bad_and_duplicate_faces,
    _fill_pinholes,
    _remove_small_components,
    _repair_non_manifold,
    _unify_face_orientations,
)
from .metrics import (
    _undirected_edge_stats,
    compute_corner_fan_components,
    compute_mesh_metrics,
    load_mesh_raw,
)


class CelloCutNotAvailableError(RuntimeError):
    """Raised when ComfyUI RemeshMesh (`sign_mode='solid'`) or CUDA is unavailable."""


def find_comfyui_root(comfyui_dir: str | Path | None = None) -> Path | None:
    """Locate a ComfyUI checkout containing `comfy_extras/mesh3d/postprocess/remesh.py`."""
    candidates: list[Path] = []
    if comfyui_dir is not None:
        candidates.append(Path(comfyui_dir))
    env_dir = os.environ.get("COMFYUI_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            Path("/content/ComfyUI"),
            Path("/content/cache/ComfyUI"),
            Path.cwd() / "ComfyUI",
        ]
    )
    for cand in candidates:
        remesh_file = cand / "comfy_extras" / "mesh3d" / "postprocess" / "remesh.py"
        if remesh_file.is_file():
            return cand.resolve()
    return None


def _import_remesh_backend(
    comfyui_dir: str | Path | None = None,
    *,
    require_cuda: bool = True,
) -> tuple[Any, Any, Any, str]:
    try:
        import torch
    except ImportError as exc:
        raise CelloCutNotAvailableError(
            "PyTorch is not installed in the current environment; CelloCut solid remesh "
            "requires PyTorch + CUDA + ComfyUI (run on Colab GPU)."
        ) from exc

    if require_cuda and not torch.cuda.is_available():
        raise CelloCutNotAvailableError(
            "CUDA GPU is not available locally; CelloCut solid remesh requires a CUDA GPU "
            "(no silent CPU fallback)."
        )

    root = find_comfyui_root(comfyui_dir)
    if root is None:
        raise CelloCutNotAvailableError(
            "ComfyUI checkout not found (checked explicit comfyui_dir, COMFYUI_DIR, "
            "/content/ComfyUI, /content/cache/ComfyUI, ./ComfyUI)."
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        from comfy_extras.mesh3d.postprocess.qem_decimate import qem_cluster_decimate
        from comfy_extras.mesh3d.postprocess.remesh import remesh_narrow_band_dc
    except Exception as exc:
        raise CelloCutNotAvailableError(
            f"Failed to import ComfyUI CelloCut remesh kernel from {root}: {exc}"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch, remesh_narrow_band_dc, qem_cluster_decimate, device


def is_cellocut_available(
    comfyui_dir: str | Path | None = None,
    *,
    require_cuda: bool = True,
) -> bool:
    """Return True iff ComfyUI `remesh_narrow_band_dc` and required device are available."""
    try:
        _import_remesh_backend(comfyui_dir=comfyui_dir, require_cuda=require_cuda)
        return True
    except CelloCutNotAvailableError:
        return False


def _dedup_face_indices_only(faces: np.ndarray) -> np.ndarray:
    """Drop duplicate-index and duplicate faces without discarding sliver triangles yet."""
    if len(faces) == 0:
        return faces
    non_dup_idx = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    faces = faces[non_dup_idx]
    if len(faces) <= 1:
        return faces
    sf = np.sort(faces, axis=1)
    order = np.lexsort((sf[:, 2], sf[:, 1], sf[:, 0]))
    s = sf[order]
    keep_sorted = np.empty(len(s), dtype=bool)
    keep_sorted[0] = True
    keep_sorted[1:] = (s[1:, 0] != s[:-1, 0]) | (s[1:, 1] != s[:-1, 1]) | (s[1:, 2] != s[:-1, 2])
    keep_orig = np.zeros(len(faces), dtype=bool)
    keep_orig[order[keep_sorted]] = True
    return faces[keep_orig]


def _collapse_edge_pairs(
    verts: np.ndarray,
    faces: np.ndarray,
    u: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Merge vertex pairs `(u, w)` and remap faces without touching unrelated split sheets."""
    num_v = len(verts)
    if len(u) == 0 or num_v == 0:
        return verts, faces, 0
    rows = np.concatenate([u, w])
    cols = np.concatenate([w, u])
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


def postclean_cellocut_mesh(
    mesh: trimesh.Trimesh,
    *,
    weld_eps_rel: float = 1e-10,
    min_component_area_rel: float = 0.005,
    area_eps_rel: float = 1e-12,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Post-clean CelloCut Dual-Contouring output without merging `_split_touching_sheets` vertices.

    1. Remove tiny isolated DC voxel specks (`area < min_component_area_rel * total_area`).
    2. Weld zero-length/snapped degenerate edges (`<= weld_eps_rel * diag`) and collapse
       microscopic degenerate-face shortest edges (`<= 1e-2 * diag`) BEFORE dropping any faces,
       preserving watertightness across `project_back` vertex snaps.
    3. Guard manifoldness & orientation without re-merging unrelated touching sheets.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    if len(verts) == 0 or len(faces) == 0:
        empty = trimesh.Trimesh(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )
        return empty, {
            "postclean": True,
            "removed_speck_faces": 0,
            "collapsed_vertices": 0,
            "repaired_non_manifold": False,
        }

    # Drop faces touching non-finite vertices if any.
    finite_v = np.isfinite(verts).all(axis=1)
    if not np.all(finite_v):
        faces = faces[finite_v[faces].all(axis=1)]

    faces_before = len(faces)
    verts, faces = _remove_small_components(verts, faces, min_area_rel=min_component_area_rel)
    removed_speck_faces = max(0, faces_before - len(faces))

    diag = _bbox_diag(verts)
    area_thresh = max(1e-14, float(area_eps_rel) * (diag * diag))
    weld_eps = max(float(weld_eps_rel) * diag, 1e-12) if float(weld_eps_rel) > 0.0 else 0.0
    max_collapse_len = max(weld_eps, 1e-2 * diag)

    collapsed_vertices = 0

    # Pass A: weld edges of degenerate faces whose endpoints were snapped within weld_eps.
    if weld_eps > 0.0 and len(faces) > 0:
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        dup_idx = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
        degen_faces = faces[(areas <= area_thresh) & ~dup_idx]
        if len(degen_faces) > 0:
            e01 = np.linalg.norm(verts[degen_faces[:, 1]] - verts[degen_faces[:, 0]], axis=1)
            e12 = np.linalg.norm(verts[degen_faces[:, 2]] - verts[degen_faces[:, 1]], axis=1)
            e20 = np.linalg.norm(verts[degen_faces[:, 0]] - verts[degen_faces[:, 2]], axis=1)
            u_list: list[np.ndarray] = []
            w_list: list[np.ndarray] = []
            for mask, a_col, b_col in (
                (e01 <= weld_eps, degen_faces[:, 0], degen_faces[:, 1]),
                (e12 <= weld_eps, degen_faces[:, 1], degen_faces[:, 2]),
                (e20 <= weld_eps, degen_faces[:, 2], degen_faces[:, 0]),
            ):
                if np.any(mask):
                    u_list.append(a_col[mask])
                    w_list.append(b_col[mask])
            if u_list:
                verts, faces, n_merged = _collapse_edge_pairs(
                    verts, faces, np.concatenate(u_list), np.concatenate(w_list)
                )
                collapsed_vertices += n_merged
                faces = _dedup_face_indices_only(faces)

    # Pass B: iteratively collapse shortest edge of any residual microscopic degenerate faces.
    for _ in range(8):
        if len(faces) == 0:
            break
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        dup_idx = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
        degen_idx = np.flatnonzero((areas <= area_thresh) & ~dup_idx)
        if len(degen_idx) == 0:
            break

        df = faces[degen_idx]
        e01 = np.linalg.norm(verts[df[:, 1]] - verts[df[:, 0]], axis=1)
        e12 = np.linalg.norm(verts[df[:, 2]] - verts[df[:, 1]], axis=1)
        e20 = np.linalg.norm(verts[df[:, 0]] - verts[df[:, 2]], axis=1)
        elens = np.stack([e01, e12, e20], axis=1)
        best = np.argmin(elens, axis=1)
        best_len = elens[np.arange(len(df)), best]
        valid = best_len <= max_collapse_len
        if not np.any(valid):
            break

        df_v = df[valid]
        best_v = best[valid]
        u = np.where(best_v == 0, df_v[:, 0], np.where(best_v == 1, df_v[:, 1], df_v[:, 2]))
        w = np.where(best_v == 0, df_v[:, 1], np.where(best_v == 1, df_v[:, 2], df_v[:, 0]))
        verts, faces, n_merged = _collapse_edge_pairs(verts, faces, u, w)
        if n_merged == 0:
            break
        collapsed_vertices += n_merged
        faces = _dedup_face_indices_only(faces)

    verts, faces = _drop_bad_and_duplicate_faces(verts, faces, area_eps_rel=area_eps_rel)
    verts, faces = _compact_unreferenced_vertices(verts, faces)

    repaired_nm = False
    if len(faces) > 0:
        _, _, counts, _, _ = _undirected_edge_stats(faces)
        has_nm_edge = bool(np.any(counts > 2))
        _, fan_counts = compute_corner_fan_components(len(verts), faces)
        has_nm_vert = bool(np.any(fan_counts > 1))
        if has_nm_edge or has_nm_vert:
            repaired_nm = True
            verts, faces = _repair_non_manifold(verts, faces, area_eps_rel=area_eps_rel)
            verts, faces, _ = _fill_pinholes(verts, faces, area_eps_rel=area_eps_rel)
            verts, faces = _drop_bad_and_duplicate_faces(verts, faces, area_eps_rel=area_eps_rel)
            if len(faces) > 0:
                faces = _unify_face_orientations(verts, faces)

    clean = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if len(clean.faces) > 0:
        clean.vertex_normals

    info = {
        "postclean": True,
        "weld_eps_rel": float(weld_eps_rel),
        "min_component_area_rel": float(min_component_area_rel),
        "area_eps_rel": float(area_eps_rel),
        "removed_speck_faces": int(removed_speck_faces),
        "collapsed_vertices": int(collapsed_vertices),
        "repaired_non_manifold": bool(repaired_nm),
    }
    return clean, info


def cellocut_remesh_mesh(
    mesh: trimesh.Trimesh,
    *,
    resolution: int = 1024,
    fill: float = 20.0,
    band: float = 1.0,
    project_back: float = 0.5,
    qef: bool = False,
    fix_poles: bool = False,
    smooth_iters: int = 0,
    drop_small_components: float | None = None,
    precluster_max_verts: int = 0,
    weld_eps_rel: float = 1e-10,
    min_component_area_rel: float = 0.005,
    area_eps_rel: float = 1e-12,
    comfyui_dir: str | Path | None = None,
    require_cuda: bool = True,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Run ComfyUI RemeshMesh (`sign_mode='solid'`) on a Trimesh object.

    Follows exact bounding-box fit and dual-contouring invocation from ComfyUI's
    `RemeshMesh.execute` (`comfy_extras/nodes_mesh_postprocess.py`).
    """
    torch, remesh_narrow_band_dc, qem_cluster_decimate, device = _import_remesh_backend(
        comfyui_dir=comfyui_dir,
        require_cuda=require_cuda,
    )

    verts_np = np.asarray(mesh.vertices, dtype=np.float32)
    faces_np = np.asarray(mesh.faces, dtype=np.int64)
    if len(verts_np) == 0 or len(faces_np) == 0:
        raise ValueError("Cannot run CelloCut remesh on an empty mesh")

    compute_device = torch.device(device)
    vv = torch.from_numpy(np.ascontiguousarray(verts_np)).to(compute_device).float()
    ff = torch.from_numpy(np.ascontiguousarray(faces_np)).to(compute_device).to(torch.int64)

    bad = ~(vv.abs() < 1e38).all(dim=1)
    if bool(bad.any()):
        ff = ff[~bad[ff].any(dim=1)]
    if ff.numel() == 0:
        raise ValueError("Mesh contains no finite faces for CelloCut remesh")

    used = torch.zeros(vv.shape[0], dtype=torch.bool, device=compute_device)
    used[ff.reshape(-1)] = True
    vv, ff = vv[used], (used.cumsum(0) - 1)[ff]

    if precluster_max_verts > 0 and vv.shape[0] > int(precluster_max_verts):
        vv, ff, _ = qem_cluster_decimate(vv, ff, target_verts=int(precluster_max_verts), colors=None)

    rs_scale = (int(resolution) + 3.0 * float(band)) / int(resolution)
    rs_center = torch.zeros(3, dtype=vv.dtype, device=compute_device)

    lo, hi = vv.min(dim=0)[0], vv.max(dim=0)[0]
    extent = float((hi - lo).max())
    reach = float(torch.max(-lo, hi).max())
    fit = extent > 0.0 and (reach > 0.51 or extent < 0.5)
    mesh_center = None
    span = 1.0
    if fit:
        mesh_center = 0.5 * (lo + hi)
        cell = rs_scale / int(resolution)
        margin = math.ceil(float(band)) + 1
        span = (int(resolution) - 2 * margin) * cell
        vv = (vv - mesh_center) / extent * span
        rs_center = rs_center + ((int(resolution) / 2.0 - 0.25) % 1.0) * cell

    resolved_drop_small = (
        float(min_component_area_rel)
        if drop_small_components is None
        else float(drop_small_components)
    )

    rv, rf, _ = remesh_narrow_band_dc(
        vv,
        ff,
        resolution=int(resolution),
        band=float(band),
        project_back=float(project_back),
        qef=bool(qef),
        sign_mode="solid",
        manifold=False,
        fill=float(fill),
        fix_poles=bool(fix_poles),
        smooth_iters=int(smooth_iters),
        drop_small_components=resolved_drop_small,
        drop_inverted_components=False,
        drop_enclosed_components=False,
        scale=rs_scale,
        center=rs_center,
        colors=None,
    )

    if fit and mesh_center is not None:
        rv = rv / span * extent + mesh_center

    out_verts = rv.detach().cpu().numpy().astype(np.float64)
    out_faces = rf.detach().cpu().numpy().astype(np.int64)

    del vv, ff, rv, rf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    remeshed = trimesh.Trimesh(vertices=out_verts, faces=out_faces, process=False)
    remeshed, post_info = postclean_cellocut_mesh(
        remeshed,
        weld_eps_rel=weld_eps_rel,
        min_component_area_rel=min_component_area_rel,
        area_eps_rel=area_eps_rel,
    )

    info = {
        "method": "cellocut_remesh",
        "sign_mode": "solid",
        "resolution": int(resolution),
        "fill": float(fill),
        "band": float(band),
        "project_back": float(project_back),
        "qef": bool(qef),
        "fix_poles": bool(fix_poles),
        "smooth_iters": int(smooth_iters),
        "drop_small_components": float(resolved_drop_small),
        "device": device,
        **post_info,
    }
    return remeshed, info


def cellocut_remesh_glb(
    input_glb: str | Path,
    output_glb: str | Path,
    *,
    resolution: int = 1024,
    fill: float = 20.0,
    band: float = 1.0,
    project_back: float = 0.5,
    qef: bool = False,
    fix_poles: bool = False,
    smooth_iters: int = 0,
    drop_small_components: float | None = None,
    precluster_max_verts: int = 0,
    weld_eps_rel: float = 1e-10,
    min_component_area_rel: float = 0.005,
    area_eps_rel: float = 1e-12,
    comfyui_dir: str | Path | None = None,
    require_cuda: bool = True,
) -> tuple[MeshMetrics, MeshMetrics, dict[str, Any]]:
    """Load raw GLB (`process=False`), run CelloCut solid remesh, and export (`process=False`)."""
    raw_mesh = load_mesh_raw(input_glb)
    raw_metrics = compute_mesh_metrics(raw_mesh)

    remeshed, info = cellocut_remesh_mesh(
        raw_mesh,
        resolution=resolution,
        fill=fill,
        band=band,
        project_back=project_back,
        qef=qef,
        fix_poles=fix_poles,
        smooth_iters=smooth_iters,
        drop_small_components=drop_small_components,
        precluster_max_verts=precluster_max_verts,
        weld_eps_rel=weld_eps_rel,
        min_component_area_rel=min_component_area_rel,
        area_eps_rel=area_eps_rel,
        comfyui_dir=comfyui_dir,
        require_cuda=require_cuda,
    )

    out_path = Path(output_glb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    remeshed.export(str(out_path))

    clean_metrics = compute_mesh_metrics(out_path)
    return raw_metrics, clean_metrics, info
