"""Lossless mesh precleaning: strip zero-area faces, duplicates, micro-components, and orphan vertices.

Preserves all remaining vertex positions and manifold topology without silent welding.
Provides both `preclean_mesh(...)` Python API and `python -m topo_exp.preclean` CLI.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from quadvox.s2_retopo.contracts import PrecleanDiff
from quadvox.s2_retopo.metrics import (
    compute_mesh_metrics_from_vf,
    compute_triangle_areas,
    extract_vf,
)


def preclean_mesh(
    input_path: str | Path,
    output_path: str | Path,
    *,
    min_component_area_rel: float = 0.0,
    area_eps_rel: float = 1e-12,
    dedup_faces: bool = True,
    diff_json_path: str | Path | None = None,
) -> PrecleanDiff:
    """Clean degenerate faces, optional small disconnected shells, and unreferenced vertices.

    Args:
        input_path: Input mesh path (`.glb`/`.obj`/`.ply`).
        output_path: Destination cleaned mesh path.
        min_component_area_rel: Drop connected components whose surface area is below
            `min_component_area_rel * total_surface_area` (e.g. `0.001` for cellocut_pb05,
            `0.0` for inplace to keep all parts).
        area_eps_rel: Relative triangle-area threshold (`area <= max(1e-14, area_eps_rel * diag^2)`).
        dedup_faces: Whether to strip exact duplicate triangle vertex triples.
        diff_json_path: Optional path to serialize `PrecleanDiff` JSON.
    """
    t0 = time.perf_counter()
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    verts, faces = extract_vf(in_path)
    before = compute_mesh_metrics_from_vf(verts, faces, area_eps_rel=area_eps_rel)

    if len(verts) == 0 or len(faces) == 0:
        empty = trimesh.Trimesh(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )
        empty.export(str(out_path))
        after = compute_mesh_metrics_from_vf(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
        diff = PrecleanDiff(
            input_path=str(in_path),
            output_path=str(out_path),
            before=before,
            after=after,
            min_component_area_rel=float(min_component_area_rel),
            elapsed_s=time.perf_counter() - t0,
        )
        if diff_json_path is not None:
            diff.write_json(diff_json_path)
        return diff

    finite_mask = np.isfinite(verts).all(axis=1)
    if np.any(finite_mask):
        v_fin = verts[finite_mask]
        diag = float(np.linalg.norm(v_fin.max(axis=0) - v_fin.min(axis=0)))
    else:
        diag = 0.0
    area_thresh = max(1e-14, float(area_eps_rel) * (diag * diag))

    # 1. Strip degenerate faces (repeated vertex indices or zero/non-finite area).
    dup_idx = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    areas = compute_triangle_areas(verts, faces)
    valid_face_mask = (~dup_idx) & np.isfinite(areas) & (areas > area_thresh)
    removed_degen = int(len(faces) - int(np.sum(valid_face_mask)))
    faces = faces[valid_face_mask]
    areas = areas[valid_face_mask]

    # 2. Strip exact duplicate faces (preserving first deterministic occurrence).
    removed_dup = 0
    if dedup_faces and len(faces) > 1:
        sf = np.sort(faces, axis=1)
        order = np.lexsort((sf[:, 2], sf[:, 1], sf[:, 0]))
        s = sf[order]
        is_first_sorted = np.empty(len(s), dtype=bool)
        is_first_sorted[0] = True
        is_first_sorted[1:] = (
            (s[1:, 0] != s[:-1, 0])
            | (s[1:, 1] != s[:-1, 1])
            | (s[1:, 2] != s[:-1, 2])
        )
        keep_mask = np.zeros(len(faces), dtype=bool)
        keep_mask[order[is_first_sorted]] = True
        removed_dup = int(len(faces) - int(np.sum(keep_mask)))
        faces = faces[keep_mask]
        areas = areas[keep_mask]

    # 3. Drop tiny disconnected micro-shells if requested.
    removed_small_comps = 0
    if float(min_component_area_rel) > 0.0 and len(faces) > 0:
        n_verts = len(verts)
        e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        rows = np.concatenate([e0, e1])
        cols = np.concatenate([e1, e0])
        data = np.ones(len(rows), dtype=np.int8)
        adj = csr_matrix((data, (rows, cols)), shape=(n_verts, n_verts))
        n_comps, vert_labels = connected_components(adj, directed=False)
        face_comp = vert_labels[faces[:, 0]]
        comp_areas = np.bincount(face_comp, weights=areas, minlength=n_comps)
        active_comps = np.unique(face_comp)
        total_area = float(np.sum(comp_areas[active_comps]))
        min_area = float(min_component_area_rel) * total_area
        keep_comp = comp_areas >= min_area
        # Never wipe out the entire mesh if min_component_area_rel exceeds max component area.
        if not np.any(keep_comp[active_comps]) and len(active_comps) > 0:
            best_comp = active_comps[int(np.argmax(comp_areas[active_comps]))]
            keep_comp[best_comp] = True
        keep_face_mask = keep_comp[face_comp]
        kept_active = int(np.sum(keep_comp[active_comps]))
        removed_small_comps = int(len(active_comps) - kept_active)
        faces = faces[keep_face_mask]

    # 4. Compact unreferenced vertices without altering vertex positions.
    if len(faces) > 0:
        used_mask = np.zeros(len(verts), dtype=bool)
        used_mask[faces.reshape(-1)] = True
        kept_vert_idx = np.flatnonzero(used_mask)
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[kept_vert_idx] = np.arange(len(kept_vert_idx), dtype=np.int64)
        verts_clean = np.ascontiguousarray(verts[kept_vert_idx], dtype=np.float64)
        faces_clean = np.ascontiguousarray(remap[faces], dtype=np.int64)
        removed_unreferenced = int(len(verts) - len(verts_clean))
    else:
        verts_clean = np.zeros((0, 3), dtype=np.float64)
        faces_clean = np.zeros((0, 3), dtype=np.int64)
        removed_unreferenced = int(len(verts))

    cleaned = trimesh.Trimesh(vertices=verts_clean, faces=faces_clean, process=False)
    cleaned.export(str(out_path))

    after = compute_mesh_metrics_from_vf(verts_clean, faces_clean, area_eps_rel=area_eps_rel)
    diff = PrecleanDiff(
        input_path=str(in_path),
        output_path=str(out_path),
        before=before,
        after=after,
        removed_degenerate_faces=removed_degen,
        removed_duplicate_faces=removed_dup,
        removed_small_components=removed_small_comps,
        removed_unreferenced_vertices=removed_unreferenced,
        min_component_area_rel=float(min_component_area_rel),
        elapsed_s=time.perf_counter() - t0,
    )
    if diff_json_path is not None:
        diff.write_json(diff_json_path)
    return diff


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lossless mesh precleaning for S2 retopology.")
    parser.add_argument("--input", required=True, help="Input mesh path (.glb/.obj/.ply).")
    parser.add_argument("--out", required=True, help="Output cleaned mesh path.")
    parser.add_argument(
        "--min-component-area-rel",
        type=float,
        default=0.0,
        help="Drop connected components with relative area < threshold (e.g. 0.001 for cellocut).",
    )
    parser.add_argument(
        "--area-eps-rel",
        type=float,
        default=1e-12,
        help="Relative zero-area degenerate face threshold.",
    )
    parser.add_argument(
        "--no-dedup-faces",
        action="store_true",
        help="Keep exact duplicate faces instead of stripping them.",
    )
    parser.add_argument(
        "--diff-json",
        default=None,
        help="Optional output path for PrecleanDiff JSON.",
    )
    args = parser.parse_args(argv)

    diff = preclean_mesh(
        args.input,
        args.out,
        min_component_area_rel=args.min_component_area_rel,
        area_eps_rel=args.area_eps_rel,
        dedup_faces=not args.no_dedup_faces,
        diff_json_path=args.diff_json,
    )
    print(diff.to_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
