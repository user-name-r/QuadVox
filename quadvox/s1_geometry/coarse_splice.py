"""Layer-2 configurable OpenVDB dual-coarse level-set splicer.

Smoothly blends two coarse meshes (e.g. a full-body silhouette-accurate coarse
mesh and a front-proportion-accurate coarse mesh) via C1 smoothstep weights
across a configurable spatial axis and optional lateral protection guard.
All coordinates live in YAML (`CoarseSpliceSpec`), never hardcoded in engine code.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from quadvox.contracts import CoarseSpliceSpec
from quadvox.tools.external_tools import resolve_blender_bin


def _smoothstep_np(e0: float, e1: float, v: np.ndarray) -> np.ndarray:
    t = np.clip((v - e0) / max(e1 - e0, 1e-8), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def compute_splice_blend_weights(
    grid_coords_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    spec: CoarseSpliceSpec,
) -> np.ndarray:
    """Compute 3D C1 blend weight `w in [0, 1]` (`0` = primary, `1` = secondary)."""
    xw, yw, zw = grid_coords_xyz
    nx, ny, nz = len(xw), len(yw), len(zw)
    e0, e1 = spec.transition_range

    if spec.split_axis == "y":
        s_axis = _smoothstep_np(e0, e1, yw)[None, :, None]
    elif spec.split_axis == "x":
        s_axis = _smoothstep_np(e0, e1, xw)[:, None, None]
    else:
        s_axis = _smoothstep_np(e0, e1, zw)[None, None, :]

    w = np.broadcast_to(s_axis, (nx, ny, nz)).copy()

    # Optional lateral guard (e.g., keep outstretched horizontal limbs from primary coarse mesh).
    if spec.lateral_guard_x is not None:
        gx0, gx1 = spec.lateral_guard_x
        guard_x = _smoothstep_np(gx0, gx1, np.abs(xw))[:, None, None]
        if spec.lateral_guard_y_max is not None:
            gy0, gy1 = spec.lateral_guard_y_max
            guard_y = (1.0 - _smoothstep_np(gy0, gy1, yw))[None, :, None]
        else:
            guard_y = 1.0
        w *= 1.0 - guard_x * guard_y

    return w.astype(np.float32)


def splice_coarse_meshes_openvdb(
    primary_verts: np.ndarray,
    primary_faces: np.ndarray,
    secondary_verts: np.ndarray,
    secondary_faces: np.ndarray,
    spec: CoarseSpliceSpec,
    *,
    openvdb_module: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Blend two coarse meshes in OpenVDB level-set space according to `spec`."""
    if openvdb_module is None:
        import openvdb as openvdb_module  # type: ignore[import-not-found]

    t0 = time.monotonic()
    vox = 1.0 / float(spec.voxel_res)
    hw = float(spec.half_width)
    xform = openvdb_module.createLinearTransform(voxelSize=vox)

    v1 = np.asarray(primary_verts, dtype=np.float32)
    f1 = np.asarray(primary_faces, dtype=np.int32)
    v2 = np.asarray(secondary_verts, dtype=np.float32)
    f2 = np.asarray(secondary_faces, dtype=np.int32)

    g1 = openvdb_module.FloatGrid.createLevelSetFromPolygons(points=v1, triangles=f1, transform=xform, halfWidth=hw)
    g2 = openvdb_module.FloatGrid.createLevelSetFromPolygons(points=v2, triangles=f2, transform=xform, halfWidth=hw)

    bb1 = g1.evalActiveVoxelBoundingBox()
    bb2 = g2.evalActiveVoxelBoundingBox()
    pad = 3
    ijk_min = tuple(min(bb1[0][i], bb2[0][i]) - pad for i in range(3))
    ijk_max = tuple(max(bb1[1][i], bb2[1][i]) + pad for i in range(3))
    shape = tuple(ijk_max[i] - ijk_min[i] + 1 for i in range(3))

    arr1 = np.empty(shape, dtype=np.float32)
    arr2 = np.empty(shape, dtype=np.float32)
    g1.copyToArray(arr1, ijk=ijk_min)
    g2.copyToArray(arr2, ijk=ijk_min)

    xw = (ijk_min[0] + np.arange(shape[0], dtype=np.float32)) * vox
    yw = (ijk_min[1] + np.arange(shape[1], dtype=np.float32)) * vox
    zw = (ijk_min[2] + np.arange(shape[2], dtype=np.float32)) * vox

    w = compute_splice_blend_weights((xw, yw, zw), spec)
    arr_hyb = ((1.0 - w) * arr1 + w * arr2).astype(np.float32)

    g_hyb = openvdb_module.FloatGrid(float(g1.background))
    g_hyb.transform = xform
    g_hyb.gridClass = openvdb_module.GridClass.LEVEL_SET
    g_hyb.copyFromArray(arr_hyb, ijk=ijk_min, tolerance=1e-6)

    pts, tris, quads = g_hyb.convertToPolygons(isovalue=0.0, adaptivity=0.0)
    face_parts = []
    if tris is not None and len(tris) > 0:
        face_parts.append(tris.astype(np.int64))
    if quads is not None and len(quads) > 0:
        q = quads.astype(np.int64)
        face_parts.append(q[:, [0, 1, 2]])
        face_parts.append(q[:, [0, 2, 3]])
    faces = np.vstack(face_parts).astype(np.int32) if face_parts else np.zeros((0, 3), dtype=np.int32)

    return pts.astype(np.float32), faces, {
        "voxel_res": int(spec.voxel_res),
        "grid_shape": list(shape),
        "vertex_count": int(len(pts)),
        "face_count": int(len(faces)),
        "elapsed_s": round(time.monotonic() - t0, 3),
    }


def splice_coarse_glbs(
    primary_glb: str | Path,
    secondary_glb: str | Path,
    out_glb: str | Path,
    spec: CoarseSpliceSpec,
    *,
    blender_bin: str | Path | None = None,
) -> dict[str, Any]:
    """Splice two normalized coarse GLB meshes using OpenVDB (direct or via Blender 5.3)."""
    p1 = Path(primary_glb)
    p2 = Path(secondary_glb)
    out_p = Path(out_glb)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    m1 = trimesh.load(str(p1), force="mesh", process=False)
    m2 = trimesh.load(str(p2), force="mesh", process=False)
    v1, f1 = np.asarray(m1.vertices, dtype=np.float32), np.asarray(m1.faces, dtype=np.int32)
    v2, f2 = np.asarray(m2.vertices, dtype=np.float32), np.asarray(m2.faces, dtype=np.int32)

    try:
        import openvdb  # type: ignore[import-not-found]

        pts, faces, stats = splice_coarse_meshes_openvdb(v1, f1, v2, f2, spec, openvdb_module=openvdb)
        trimesh.Trimesh(vertices=pts, faces=faces, process=False).export(str(out_p))
        return stats
    except ImportError:
        pass

    # Fallback: run inside Blender 5.3's bundled `openvdb` Python module, staged inside out_p.parent.
    b_bin = resolve_blender_bin(blender_bin)
    stage_dir = out_p.parent / "_splice_staging"
    stage_dir.mkdir(parents=True, exist_ok=True)
    in_npz = stage_dir / "inputs.npz"
    out_npz = stage_dir / "output.npz"
    spec_json = stage_dir / "spec.json"
    worker_py = stage_dir / "blender_splice_worker.py"

    np.savez(in_npz, v1=v1, f1=f1, v2=v2, f2=f2)
    spec_json.write_text(
        json.dumps(
            {
                "voxel_res": spec.voxel_res,
                "half_width": spec.half_width,
                "split_axis": spec.split_axis,
                "transition_range": list(spec.transition_range),
                "lateral_guard_x": list(spec.lateral_guard_x) if spec.lateral_guard_x else None,
                "lateral_guard_y_max": list(spec.lateral_guard_y_max) if spec.lateral_guard_y_max else None,
            }
        ),
        encoding="utf-8",
    )
    repo_root = str(Path(__file__).resolve().parents[2])
    worker_py.write_text(
        "\n".join(
            [
                "import json, sys, numpy as np",
                f"sys.path.insert(0, {repo_root!r})",
                "from quadvox.contracts import CoarseSpliceSpec",
                "from quadvox.s1_geometry.coarse_splice import splice_coarse_meshes_openvdb",
                "import openvdb",
                f"d = np.load({str(in_npz)!r})",
                f"spec = CoarseSpliceSpec.from_dict(json.loads(open({str(spec_json)!r}).read()))",
                "pts, faces, stats = splice_coarse_meshes_openvdb(d['v1'], d['f1'], d['v2'], d['f2'], spec, openvdb_module=openvdb)",
                f"np.savez({str(out_npz)!r}, verts=pts, faces=faces, stats_json=json.dumps(stats))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [str(b_bin), "--background", "--factory-startup", "--python-exit-code", "1", "--python", str(worker_py)],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    res = np.load(out_npz, allow_pickle=False)
    trimesh.Trimesh(vertices=res["verts"], faces=res["faces"], process=False).export(str(out_p))
    return json.loads(str(res["stats_json"]))
