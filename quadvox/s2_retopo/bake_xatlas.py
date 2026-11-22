"""Decoupled xatlas UV parameterization and headless Blender Cycles normal baking.

Upstream references:
- xatlas-python: https://github.com/mworchel/xatlas-python
- Blender Cycles Bake API: https://docs.blender.org/manual/en/latest/render/cycles/baking.html

Top-level imports are restricted to Python stdlib so this module can be executed
both inside the project `.venv` (for `unwrap_xatlas`) and directly as a
headless Blender `--python` script (for `bake_normal_map_blender`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from quadvox.tools.external_tools import resolve_blender_bin

DEFAULT_BLENDER_BIN = resolve_blender_bin()


@dataclass
class UnwrapResult:
    """Result metadata and UV quality metrics from `unwrap_xatlas`."""

    input_mesh_path: str
    out_mesh_path: str
    tex_size: int
    padding_px: int
    vertex_count: int
    face_count: int
    atlas_width: int
    atlas_height: int
    atlas_count: int
    chart_count: int
    atlas_utilization: float
    uv_coverage: float
    uv_overlap_count: int
    uv_area_sum: float
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        return target


@dataclass
class BakeResult:
    """Result metadata and ray-miss diagnostics from `bake_normal_map_blender`."""

    high_mesh_path: str
    low_uv_mesh_path: str
    out_normal_png: str
    out_baked_glb: str
    tex_size: int
    padding_px: int
    samples: int
    use_gpu: bool
    device_used: str
    diag: float
    cage_extrusion: float
    max_ray_distance: float
    uv_island_pixels: int
    ray_miss_pixels: int
    ray_miss_ratio: float
    bake_elapsed_s: float
    blender_version: str
    surface_dist_p95: float = 0.0
    surface_dist_p99: float = 0.0
    backface_hit_ratio: float = 0.0
    backface_guarded_pixels: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        return target


def _compute_smooth_vertex_normals(verts: Any, faces: Any) -> Any:
    """Compute area-weighted unit vertex normals on the pre-split `(V, F)` topology."""
    import numpy as np

    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    vn = np.zeros((len(v), 3), dtype=np.float64)
    if len(v) == 0 or len(f) == 0:
        return vn

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)
    face_n = np.where(np.isfinite(face_n), face_n, 0.0)
    for col in range(3):
        np.add.at(vn, f[:, col], face_n)

    norms = np.linalg.norm(vn, axis=1)
    good = np.isfinite(norms) & (norms > 1e-12)
    out = np.zeros((len(v), 3), dtype=np.float64)
    out[good] = vn[good] / norms[good, None]
    out[~good] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return out


def compute_uv_raster_stats(
    uvs: Any,
    faces: Any,
    tex_size: int = 2048,
) -> dict[str, Any]:
    """Rasterize UV triangles into a pixel hit-count buffer using the top-left fill rule.

    Uses exact pixel centers `(x + 0.5, y + 0.5)` on a `(tex_size, tex_size)` grid
    with CCW orientation normalization and top-left edge ownership so adjacent
    triangles sharing an edge inside a UV chart are never double-counted.
    """
    import numpy as np

    res = max(16, int(tex_size))
    uv = np.asarray(uvs, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(uv) == 0 or len(f) == 0:
        return {
            "uv_coverage": 0.0,
            "uv_overlap_count": 0,
            "uv_area_sum": 0.0,
            "covered_pixels": 0,
        }

    p = uv[f]  # (F, 3, 2)
    # Exact UV area sum in normalized [0, 1]^2 space.
    area2_uv = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (
        p[:, 1, 1] - p[:, 0, 1]
    ) * (p[:, 2, 0] - p[:, 0, 0])
    uv_area_sum = float(0.5 * np.sum(np.abs(area2_uv)))

    px = p * float(res)
    x0 = px[:, 0, 0].copy()
    y0 = px[:, 0, 1].copy()
    x1 = px[:, 1, 0].copy()
    y1 = px[:, 1, 1].copy()
    x2 = px[:, 2, 0].copy()
    y2 = px[:, 2, 1].copy()

    area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    flip = area2 < 0.0
    if np.any(flip):
        tx, ty = x1[flip].copy(), y1[flip].copy()
        x1[flip], y1[flip] = x2[flip], y2[flip]
        x2[flip], y2[flip] = tx, ty
        area2[flip] = -area2[flip]

    valid = np.isfinite(area2) & (area2 > 1e-9)
    p_min_x = np.maximum(0, np.floor(np.minimum(np.minimum(x0, x1), x2) - 0.5).astype(np.int32))
    p_max_x = np.minimum(res - 1, np.ceil(np.maximum(np.maximum(x0, x1), x2) - 0.5).astype(np.int32))
    p_min_y = np.maximum(0, np.floor(np.minimum(np.minimum(y0, y1), y2) - 0.5).astype(np.int32))
    p_max_y = np.minimum(res - 1, np.ceil(np.maximum(np.maximum(y0, y1), y2) - 0.5).astype(np.int32))

    hit = np.zeros((res, res), dtype=np.uint16)

    def _is_top_left(ax: float, ay: float, bx: float, by: float) -> bool:
        return (ay < by) or (ay == by and ax > bx)

    for i in np.flatnonzero(valid):
        ix0 = int(p_min_x[i])
        ix1 = int(p_max_x[i])
        iy0 = int(p_min_y[i])
        iy1 = int(p_max_y[i])
        if ix1 < ix0 or iy1 < iy0:
            continue

        ax, ay = float(x0[i]), float(y0[i])
        bx, by = float(x1[i]), float(y1[i])
        cx, cy = float(x2[i]), float(y2[i])

        xs = np.arange(ix0, ix1 + 1, dtype=np.float64) + 0.5
        ys = np.arange(iy0, iy1 + 1, dtype=np.float64) + 0.5

        e0 = (bx - ax) * (ys[:, None] - ay) - (by - ay) * (xs[None, :] - ax)
        e1 = (cx - bx) * (ys[:, None] - by) - (cy - by) * (xs[None, :] - bx)
        e2 = (ax - cx) * (ys[:, None] - cy) - (ay - cy) * (xs[None, :] - cx)

        m0 = (e0 >= 0.0) if _is_top_left(ax, ay, bx, by) else (e0 > 0.0)
        m1 = (e1 >= 0.0) if _is_top_left(bx, by, cx, cy) else (e1 > 0.0)
        m2 = (e2 >= 0.0) if _is_top_left(cx, cy, ax, ay) else (e2 > 0.0)
        inside = m0 & m1 & m2
        if np.any(inside):
            hit[iy0 : iy1 + 1, ix0 : ix1 + 1] += inside

    covered = int(np.sum(hit >= 1))
    overlap_count = int(np.sum(hit > 1))
    uv_coverage = float(covered / float(res * res))
    return {
        "uv_coverage": uv_coverage,
        "uv_overlap_count": overlap_count,
        "uv_area_sum": uv_area_sum,
        "covered_pixels": covered,
    }


def unwrap_xatlas(
    mesh_path: str | Path,
    out_mesh_path: str | Path,
    tex_size: int = 2048,
    padding_px: int = 8,
    *,
    unwrap_json_path: str | Path | None = None,
) -> UnwrapResult:
    """Unwrap and pack UVs via `xatlas`, keeping pre-split smooth normals and topology intact."""
    import numpy as np
    import trimesh
    import xatlas

    from quadvox.s2_retopo.metrics import extract_vf

    t0 = time.perf_counter()
    in_path = Path(mesh_path)
    out_path = Path(out_mesh_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    verts, faces = extract_vf(in_path)
    if len(verts) == 0 or len(faces) == 0:
        raise ValueError(f"Cannot xatlas-unwrap empty mesh: {in_path}")

    normals_orig = _compute_smooth_vertex_normals(verts, faces)

    v_f32 = np.ascontiguousarray(verts, dtype=np.float32)
    f_u32 = np.ascontiguousarray(faces, dtype=np.uint32)
    n_f32 = np.ascontiguousarray(normals_orig, dtype=np.float32)

    atlas = xatlas.Atlas()
    atlas.add_mesh(v_f32, f_u32, n_f32)

    co = xatlas.ChartOptions()
    po = xatlas.PackOptions()
    po.resolution = int(tex_size)
    po.padding = int(padding_px)
    po.bilinear = True
    atlas.generate(chart_options=co, pack_options=po)

    vmapping, indices, uvs = atlas.get_mesh(0)
    vm = np.asarray(vmapping, dtype=np.int64)
    f_uv = np.ascontiguousarray(indices, dtype=np.int64)
    uv_f32 = np.ascontiguousarray(uvs, dtype=np.float32)

    v_uv = np.ascontiguousarray(verts[vm], dtype=np.float64)
    n_uv = np.ascontiguousarray(normals_orig[vm], dtype=np.float64)

    raster = compute_uv_raster_stats(uv_f32, f_uv, tex_size=int(tex_size))

    out_mesh = trimesh.Trimesh(
        vertices=v_uv,
        faces=f_uv,
        vertex_normals=n_uv,
        process=False,
    )
    out_mesh.visual = trimesh.visual.TextureVisuals(uv=uv_f32)
    out_mesh.export(str(out_path))

    result = UnwrapResult(
        input_mesh_path=str(in_path),
        out_mesh_path=str(out_path),
        tex_size=int(tex_size),
        padding_px=int(padding_px),
        vertex_count=int(len(v_uv)),
        face_count=int(len(f_uv)),
        atlas_width=int(atlas.width),
        atlas_height=int(atlas.height),
        atlas_count=int(atlas.atlas_count),
        chart_count=int(atlas.chart_count),
        atlas_utilization=float(atlas.utilization),
        uv_coverage=float(raster["uv_coverage"]),
        uv_overlap_count=int(raster["uv_overlap_count"]),
        uv_area_sum=float(raster["uv_area_sum"]),
        elapsed_s=time.perf_counter() - t0,
    )
    if unwrap_json_path is not None:
        result.write_json(unwrap_json_path)
    return result


def _import_mesh_in_blender(mesh_path: Path) -> list[Any]:
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
    return meshes


def _configure_cycles_device(scene: Any, use_gpu: bool) -> str:
    import bpy

    scene.render.engine = "CYCLES"
    if not use_gpu:
        scene.cycles.device = "CPU"
        return "CPU"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    cuda_devs = [d for d in prefs.devices if d.type == "CUDA"]
    if not cuda_devs:
        raise RuntimeError("CUDA GPU requested for Cycles baking, but no CUDA devices were detected.")
    for dev in prefs.devices:
        dev.use = dev.type == "CUDA"
    scene.cycles.device = "GPU"
    return f"CUDA:{','.join(d.name for d in cuda_devs)}"


def _set_material_nodes(mat: Any) -> Any:
    if mat.node_tree is None:
        mat.use_nodes = True
    return mat.node_tree


def _estimate_low_to_high_surface_dist(
    high_meshes: Sequence[Any],
    low_obj: Any,
    *,
    max_query_points: int = 40000,
) -> tuple[float, float]:
    """Measure p95 and p99 low-to-high surface distance in Blender world space via BVHTree."""
    import bpy
    import numpy as np
    from mathutils.bvhtree import BVHTree

    depsgraph = bpy.context.evaluated_depsgraph_get()
    high_bvhs = [(hobj.matrix_world.inverted(), BVHTree.FromObject(hobj, depsgraph)) for hobj in high_meshes]
    if not high_bvhs:
        return 0.0, 0.0

    mw_low = low_obj.matrix_world
    raw_pts = [mw_low @ v.co for v in low_obj.data.vertices]
    raw_pts.extend(mw_low @ p.center for p in low_obj.data.polygons)
    if not raw_pts:
        return 0.0, 0.0

    if len(raw_pts) > int(max_query_points):
        step = max(1, len(raw_pts) // int(max_query_points))
        q_pts = raw_pts[::step][: int(max_query_points)]
    else:
        q_pts = raw_pts

    dists = np.full(len(q_pts), float("inf"), dtype=np.float64)
    for mw_inv, bvh in high_bvhs:
        for idx, pt in enumerate(q_pts):
            _, _, _, d = bvh.find_nearest(mw_inv @ pt)
            if d is not None and d < dists[idx]:
                dists[idx] = float(d)

    finite = np.isfinite(dists)
    if not np.any(finite):
        return 0.0, 0.0
    valid = dists[finite]
    return float(np.percentile(valid, 95.0)), float(np.percentile(valid, 99.0))


def _run_bake_in_blender(
    *,
    high_mesh_path: Path,
    low_uv_mesh_path: Path,
    out_normal_png: Path,
    out_baked_glb: Path,
    tex_size: int,
    padding_px: int,
    samples: int,
    use_gpu: bool,
    ray_distance: float | None,
    cage_extrusion: float | None,
    meta_json_path: Path | None,
) -> BakeResult:
    import bpy
    import mathutils
    import numpy as np

    if not high_mesh_path.exists():
        raise FileNotFoundError(f"Highpoly mesh not found: {high_mesh_path}")
    if not low_uv_mesh_path.exists():
        raise FileNotFoundError(f"Lowpoly UV mesh not found: {low_uv_mesh_path}")

    t0 = time.perf_counter()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    device_used = _configure_cycles_device(scene, use_gpu=use_gpu)
    scene.cycles.samples = max(1, int(samples))
    scene.cycles.use_denoising = False
    scene.view_settings.view_transform = "Standard"

    high_meshes = _import_mesh_in_blender(high_mesh_path)
    low_meshes = _import_mesh_in_blender(low_uv_mesh_path)

    # Ensure single active lowpoly object (unwrap_xatlas outputs 1 mesh).
    if len(low_meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in low_meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = low_meshes[0]
        bpy.ops.object.join()
        low_obj = bpy.context.view_layer.objects.active
    else:
        low_obj = low_meshes[0]

    if not low_obj.data.uv_layers:
        raise ValueError(f"Lowpoly mesh has no UV layer for normal baking: {low_uv_mesh_path}")

    # Preserve pre-split smooth custom normals if present; otherwise apply smooth shading.
    bpy.ops.object.select_all(action="DESELECT")
    low_obj.select_set(True)
    bpy.context.view_layer.objects.active = low_obj
    if not low_obj.data.has_custom_normals:
        bpy.ops.object.shade_smooth()

    # World-space bounding box diagonal of lowpoly mesh for adaptive ray bounds.
    bpy.context.view_layer.update()
    mw = low_obj.matrix_world
    pts = [mw @ mathutils.Vector(c) for c in low_obj.bound_box]
    bmin = mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    bmax = mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    diag = float((bmax - bmin).length)
    if diag <= 1e-8:
        diag = 1.0

    d_p95, d_p99 = _estimate_low_to_high_surface_dist(high_meshes, low_obj)
    if cage_extrusion is not None:
        resolved_cage = float(cage_extrusion)
    elif d_p99 > 0.0:
        resolved_cage = float(np.clip(2.0 * d_p99, 0.0012 * diag, 0.015 * diag))
    else:
        resolved_cage = 0.002 * diag

    if ray_distance is not None:
        resolved_ray = float(ray_distance)
    else:
        resolved_ray = float(max(2.5 * resolved_cage, 0.003 * diag))

    res = int(tex_size)

    # Step 1: Bake pure-white EMIT pass on low_obj alone (margin=0) for exact Cycles UV island mask.
    mask_img = bpy.data.images.new("UVMask", width=res, height=res, alpha=True, float_buffer=False)
    mask_img.colorspace_settings.name = "Non-Color"
    mask_mat = bpy.data.materials.new("BakeUVMaskMat")
    mnt = _set_material_nodes(mask_mat)
    mnt.nodes.clear()
    em_node = mnt.nodes.new("ShaderNodeEmission")
    em_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    em_node.inputs["Strength"].default_value = 1.0
    out_node = mnt.nodes.new("ShaderNodeOutputMaterial")
    mnt.links.new(em_node.outputs["Emission"], out_node.inputs["Surface"])
    mtex_node = mnt.nodes.new("ShaderNodeTexImage")
    mtex_node.image = mask_img
    mnt.nodes.active = mtex_node

    low_obj.data.materials.clear()
    low_obj.data.materials.append(mask_mat)
    bpy.ops.object.select_all(action="DESELECT")
    low_obj.select_set(True)
    bpy.context.view_layer.objects.active = low_obj
    bpy.ops.object.bake(
        type="EMIT",
        use_selected_to_active=False,
        margin=0,
        use_clear=True,
    )
    mask_buf = np.empty(res * res * 4, dtype=np.float32)
    mask_img.pixels.foreach_get(mask_buf)
    mask_arr = mask_buf.reshape(res, res, 4)
    uv_mask = mask_arr[:, :, 0] > 0.5
    uv_island_pixels = int(np.sum(uv_mask))

    # Step 2: Bake TANGENT NORMAL with margin=0 onto sentinel (0, 0, 0, 0) image for ray_miss_ratio.
    sentinel_img = bpy.data.images.new(
        "SentinelNormal", width=res, height=res, alpha=True, float_buffer=False
    )
    sentinel_img.colorspace_settings.name = "Non-Color"
    sentinel_img.pixels.foreach_set(np.zeros(res * res * 4, dtype=np.float32))

    bake_mat = bpy.data.materials.new("BakeTargetMat")
    bnt = _set_material_nodes(bake_mat)
    btex = bnt.nodes.new("ShaderNodeTexImage")
    btex.image = sentinel_img
    bnt.nodes.active = btex
    low_obj.data.materials[0] = bake_mat

    bpy.ops.object.select_all(action="DESELECT")
    for hobj in high_meshes:
        hobj.select_set(True)
    low_obj.select_set(True)
    bpy.context.view_layer.objects.active = low_obj

    bpy.ops.object.bake(
        type="NORMAL",
        normal_space="TANGENT",
        normal_r="POS_X",
        normal_g="POS_Y",
        normal_b="POS_Z",
        use_selected_to_active=True,
        cage_extrusion=resolved_cage,
        max_ray_distance=resolved_ray,
        margin=0,
        use_clear=False,
    )
    sent_buf = np.empty(res * res * 4, dtype=np.float32)
    sentinel_img.pixels.foreach_get(sent_buf)
    sent_arr = sent_buf.reshape(res, res, 4)
    miss_mask = uv_mask & np.all(sent_arr[:, :, :3] <= 1e-4, axis=-1)
    hit_mask = uv_mask & (~miss_mask)
    hit_count = int(np.sum(hit_mask))
    backface_mask = hit_mask & (sent_arr[:, :, 2] <= 0.505)
    backface_hit_ratio = float(np.sum(backface_mask) / hit_count) if hit_count > 0 else 0.0
    ray_miss_pixels = int(np.sum(miss_mask))
    ray_miss_ratio = float(ray_miss_pixels / uv_island_pixels) if uv_island_pixels > 0 else 1.0

    # Step 3: Final tangent-space normal bake with margin=padding_px and neutral clear.
    final_img = bpy.data.images.new(
        "BakedNormal", width=res, height=res, alpha=False, float_buffer=False
    )
    final_img.colorspace_settings.name = "Non-Color"
    btex.image = final_img
    bnt.nodes.active = btex

    bpy.ops.object.bake(
        type="NORMAL",
        normal_space="TANGENT",
        normal_r="POS_X",
        normal_g="POS_Y",
        normal_b="POS_Z",
        use_selected_to_active=True,
        cage_extrusion=resolved_cage,
        max_ray_distance=resolved_ray,
        margin=max(0, int(padding_px)),
        margin_type="EXTEND",
        use_clear=True,
    )

    # Guard residual backface / extreme grazing hits (N_z <= 0.16, i.e. B <= 0.58)
    # by reverting to neutral tangent normal (0.5, 0.5, 1.0) so thin hair/collar
    # edges do not produce Cycles shadow-terminator black spots.
    final_buf = np.empty(res * res * 4, dtype=np.float32)
    final_img.pixels.foreach_get(final_buf)
    final_arr = final_buf.reshape(res, res, 4)
    guard_mask = final_arr[:, :, 2] <= 0.58
    backface_guarded_pixels = int(np.sum(guard_mask))
    if backface_guarded_pixels > 0:
        final_arr[guard_mask] = np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float32)
        final_img.pixels.foreach_set(final_arr.ravel())
        final_img.update()

    out_normal_png.parent.mkdir(parents=True, exist_ok=True)
    final_img.filepath_raw = str(out_normal_png)
    final_img.file_format = "PNG"
    final_img.save()

    # Step 4: Build neutral clay Principled BSDF with packed normal map and export self-contained GLB.
    if low_obj.parent is not None:
        mw_keep = low_obj.matrix_world.copy()
        low_obj.parent = None
        low_obj.matrix_world = mw_keep
    for obj in list(bpy.data.objects):
        if obj != low_obj:
            bpy.data.objects.remove(obj, do_unlink=True)

    packed_nrm = bpy.data.images.load(str(out_normal_png))
    packed_nrm.colorspace_settings.name = "Non-Color"
    packed_nrm.pack()

    clay_mat = bpy.data.materials.new("NeutralClayBakedMat")
    cnt = _set_material_nodes(clay_mat)
    bsdf = cnt.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = cnt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.58, 0.58, 0.59, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.64
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.35

    tex_node = cnt.nodes.new("ShaderNodeTexImage")
    tex_node.image = packed_nrm
    tex_node.interpolation = "Linear"
    tex_node.extension = "EXTEND"

    nrm_node = cnt.nodes.new("ShaderNodeNormalMap")
    nrm_node.space = "TANGENT"
    nrm_node.inputs["Strength"].default_value = 1.0

    cnt.links.new(tex_node.outputs["Color"], nrm_node.inputs["Color"])
    cnt.links.new(nrm_node.outputs["Normal"], bsdf.inputs["Normal"])

    low_obj.data.materials.clear()
    low_obj.data.materials.append(clay_mat)

    out_baked_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    low_obj.select_set(True)
    bpy.context.view_layer.objects.active = low_obj
    bpy.ops.export_scene.gltf(
        filepath=str(out_baked_glb),
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_tangents=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
    )

    elapsed_s = time.perf_counter() - t0
    blender_ver = ".".join(str(x) for x in bpy.app.version)

    res_obj = BakeResult(
        high_mesh_path=str(high_mesh_path),
        low_uv_mesh_path=str(low_uv_mesh_path),
        out_normal_png=str(out_normal_png),
        out_baked_glb=str(out_baked_glb),
        tex_size=res,
        padding_px=int(padding_px),
        samples=max(1, int(samples)),
        use_gpu=bool(use_gpu),
        device_used=device_used,
        diag=diag,
        cage_extrusion=resolved_cage,
        max_ray_distance=resolved_ray,
        uv_island_pixels=uv_island_pixels,
        ray_miss_pixels=ray_miss_pixels,
        ray_miss_ratio=ray_miss_ratio,
        bake_elapsed_s=elapsed_s,
        blender_version=blender_ver,
        surface_dist_p95=d_p95,
        surface_dist_p99=d_p99,
        backface_hit_ratio=backface_hit_ratio,
        backface_guarded_pixels=backface_guarded_pixels,
    )
    if meta_json_path is not None:
        res_obj.write_json(meta_json_path)
    return res_obj


def bake_normal_map_blender(
    high_mesh_path: str | Path,
    low_uv_mesh_path: str | Path,
    out_normal_png: str | Path,
    out_baked_glb: str | Path,
    tex_size: int = 2048,
    blender_bin: str | Path = DEFAULT_BLENDER_BIN,
    use_gpu: bool = False,
    ray_distance: float | None = None,
    cage_extrusion: float | None = None,
    *,
    padding_px: int = 8,
    samples: int = 1,
    bake_json_path: str | Path | None = None,
) -> BakeResult:
    """Bake tangent-space normal map in headless Blender Cycles and export self-contained GLB."""
    high_p = Path(high_mesh_path)
    low_p = Path(low_uv_mesh_path)
    png_p = Path(out_normal_png)
    glb_p = Path(out_baked_glb)

    if not high_p.exists():
        raise FileNotFoundError(f"Highpoly mesh not found: {high_p}")
    if not low_p.exists():
        raise FileNotFoundError(f"Lowpoly UV mesh not found: {low_p}")

    if "bpy" in sys.modules:
        return _run_bake_in_blender(
            high_mesh_path=high_p,
            low_uv_mesh_path=low_p,
            out_normal_png=png_p,
            out_baked_glb=glb_p,
            tex_size=int(tex_size),
            padding_px=int(padding_px),
            samples=int(samples),
            use_gpu=bool(use_gpu),
            ray_distance=ray_distance,
            cage_extrusion=cage_extrusion,
            meta_json_path=Path(bake_json_path) if bake_json_path is not None else None,
        )

    blender_path = Path(blender_bin)
    if not blender_path.exists():
        raise FileNotFoundError(f"Blender binary not found: {blender_path}")

    png_p.parent.mkdir(parents=True, exist_ok=True)
    glb_p.parent.mkdir(parents=True, exist_ok=True)

    cleanup_meta = bake_json_path is None
    meta_path = (
        Path(bake_json_path)
        if bake_json_path is not None
        else (png_p.parent / f"._{png_p.stem}_bake_meta_{os.getpid()}.json")
    )

    cmd: list[str] = [
        str(blender_path),
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "bake",
        "--high",
        str(high_p),
        "--low",
        str(low_p),
        "--out-png",
        str(png_p),
        "--out-glb",
        str(glb_p),
        "--tex-size",
        str(int(tex_size)),
        "--padding-px",
        str(int(padding_px)),
        "--samples",
        str(int(samples)),
        "--report-json",
        str(meta_path),
    ]
    if use_gpu:
        cmd.append("--use-gpu")
    if ray_distance is not None:
        cmd.extend(["--ray-distance", str(float(ray_distance))])
    if cage_extrusion is not None:
        cmd.extend(["--cage-extrusion", str(float(cage_extrusion))])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not png_p.exists() or not glb_p.exists() or not meta_path.exists():
            raise RuntimeError(
                f"Blender normal bake failed (code={proc.returncode}):\n"
                f"CMD: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
            )
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        return BakeResult(**payload)
    finally:
        if cleanup_meta and meta_path.exists():
            meta_path.unlink(missing_ok=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="xatlas UV unwrapping and Blender Cycles tangent-space normal baking."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_unwrap = sub.add_parser("unwrap", help="Unwrap UV coordinates with xatlas.")
    p_unwrap.add_argument("--mesh", required=True, help="Input mesh path (.glb/.obj/.ply).")
    p_unwrap.add_argument("--out", required=True, help="Output UV-mapped .glb path.")
    p_unwrap.add_argument("--tex-size", type=int, default=2048, help="Target atlas texture size.")
    p_unwrap.add_argument("--padding-px", type=int, default=8, help="Chart packing padding in pixels.")
    p_unwrap.add_argument("--report-json", default=None, help="Optional UnwrapResult JSON output.")

    p_bake = sub.add_parser("bake", help="Bake tangent normal map in headless Blender Cycles.")
    p_bake.add_argument("--high", required=True, help="Highpoly reference mesh.")
    p_bake.add_argument("--low", required=True, help="Lowpoly UV-mapped mesh.")
    p_bake.add_argument("--out-png", required=True, help="Output 2048x2048 tangent normal PNG.")
    p_bake.add_argument("--out-glb", required=True, help="Output self-contained baked .glb.")
    p_bake.add_argument("--tex-size", type=int, default=2048, help="Bake texture resolution.")
    p_bake.add_argument("--padding-px", type=int, default=8, help="Bake dilation margin in pixels.")
    p_bake.add_argument("--samples", type=int, default=1, help="Cycles bake samples per texel.")
    p_bake.add_argument("--blender-bin", default=str(DEFAULT_BLENDER_BIN), help="Blender executable.")
    p_bake.add_argument("--use-gpu", action="store_true", help="Enable Cycles CUDA GPU baking.")
    p_bake.add_argument("--ray-distance", type=float, default=None, help="Override max_ray_distance.")
    p_bake.add_argument("--cage-extrusion", type=float, default=None, help="Override cage_extrusion.")
    p_bake.add_argument("--report-json", default=None, help="Optional BakeResult JSON output.")

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
    if args.command == "unwrap":
        res = unwrap_xatlas(
            mesh_path=args.mesh,
            out_mesh_path=args.out,
            tex_size=int(args.tex_size),
            padding_px=int(args.padding_px),
            unwrap_json_path=args.report_json,
        )
        print(res.to_json(indent=2))
        return 0

    if args.command == "bake":
        res = bake_normal_map_blender(
            high_mesh_path=args.high,
            low_uv_mesh_path=args.low,
            out_normal_png=args.out_png,
            out_baked_glb=args.out_baked_glb if hasattr(args, "out_baked_glb") else args.out_glb,
            tex_size=int(args.tex_size),
            blender_bin=args.blender_bin,
            use_gpu=bool(args.use_gpu),
            ray_distance=args.ray_distance,
            cage_extrusion=args.cage_extrusion,
            padding_px=int(args.padding_px),
            samples=int(args.samples),
            bake_json_path=args.report_json,
        )
        if "bpy" not in sys.modules:
            print(res.to_json(indent=2))
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
