"""Headless Blender 5.3 studio renderer & glTF 2.0 / GLB PBR exporter (Layer 1 Universal Core).

Top-level imports are strictly Python stdlib so this runs inside Blender 5.3 via:
    blender --background --factory-startup --python-exit-code 1 --python quadvox/s3_lookdev/studio_export.py -- ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

OBJECT_VIEW_SPECS: tuple[dict[str, Any], ...] = (
    {"key": "front", "label": "FRONT (+Z)", "location": (0.0, -3.0, 0.50), "target": (0.0, 0.0, 0.50), "ortho": True, "ortho_scale": 1.08, "width": 768, "height": 768},
    {"key": "three_quarter", "label": "3/4 VIEW", "location": (1.65, -2.05, 1.05), "target": (0.0, 0.0, 0.50), "ortho": False, "lens": 80.0, "width": 768, "height": 768},
    {"key": "top_34", "label": "TOP 3/4", "location": (1.20, -1.75, 2.05), "target": (0.0, 0.0, 0.50), "ortho": False, "lens": 75.0, "width": 768, "height": 768},
    {"key": "back", "label": "BACK (-Z)", "location": (0.0, 3.0, 0.50), "target": (0.0, 0.0, 0.50), "ortho": True, "ortho_scale": 1.08, "width": 768, "height": 768},
    {"key": "left", "label": "LEFT (+X)", "location": (3.0, 0.0, 0.50), "target": (0.0, 0.0, 0.50), "ortho": True, "ortho_scale": 1.08, "width": 768, "height": 768},
    {"key": "closeup", "label": "DETAIL CLOSE-UP", "location": (0.48, -1.25, 0.70), "target": (0.10, -0.15, 0.52), "ortho": False, "lens": 90.0, "width": 1024, "height": 1024},
)

_FONT_5X7: dict[str, tuple[str, ...]] = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "/": ("00001", "00010", "00100", "00100", "01000", "10000", "00000"),
    "|": ("00100", "00100", "00100", "00100", "00100", "00100", "00100"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00110", "01000", "10000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11100", "10010", "10001", "10001", "10001", "10010", "11100"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


def _draw_banner_label(img_rgb: Any, text: str, scale: int = 3) -> None:
    banner_h = 7 * scale + 14
    img_rgb[:banner_h, :, :] = (24, 26, 30)
    x0, y0 = 10, 7
    fg = (238, 240, 244)
    h, w, _ = img_rgb.shape
    for ch in text.upper():
        glyph = _FONT_5X7.get(ch, _FONT_5X7[" "])
        for gy, row_str in enumerate(glyph):
            for gx, bit in enumerate(row_str):
                if bit == "1":
                    py0, px0 = y0 + gy * scale, x0 + gx * scale
                    if py0 + scale <= h and px0 + scale <= w:
                        img_rgb[py0 : py0 + scale, px0 : px0 + scale] = fg
        x0 += 6 * scale


def _resize_rgb_bilinear(img_rgb: Any, target_w: int, target_h: int) -> Any:
    import numpy as np

    src_h, src_w, _ = img_rgb.shape
    if src_h == target_h and src_w == target_w:
        return img_rgb
    ys = np.linspace(0.0, src_h - 1.0, target_h, dtype=np.float32)
    xs = np.linspace(0.0, src_w - 1.0, target_w, dtype=np.float32)
    y0, x0 = np.floor(ys).astype(np.int32), np.floor(xs).astype(np.int32)
    y1, x1 = np.minimum(y0 + 1, src_h - 1), np.minimum(x0 + 1, src_w - 1)
    wy, wx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    src = img_rgb.astype(np.float32)
    top = (1.0 - wx) * src[y0][:, x0] + wx * src[y0][:, x1]
    bot = (1.0 - wx) * src[y1][:, x0] + wx * src[y1][:, x1]
    return np.clip(np.rint((1.0 - wy) * top + wy * bot), 0, 255).astype(np.uint8)


def _read_png_rgb_in_blender(fpath: Path) -> Any:
    import bpy
    import numpy as np

    img = bpy.data.images.load(str(fpath), check_existing=False)
    try:
        w, h = int(img.size[0]), int(img.size[1])
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        arr = buf.reshape(h, w, 4)[::-1, :, :3]
        return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    finally:
        bpy.data.images.remove(img)


def _write_png_rgb_in_blender(fpath: Path, arr_rgb: Any) -> None:
    import bpy
    import numpy as np

    fpath.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = arr_rgb.shape
    img = bpy.data.images.new("_SheetOut", width=w, height=h, alpha=False, float_buffer=False)
    try:
        img.colorspace_settings.name = "sRGB"
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = arr_rgb[::-1, :, :3].astype(np.float32) / 255.0
        img.pixels.foreach_set(rgba.ravel())
        img.filepath_raw = str(fpath)
        img.file_format = "PNG"
        img.save()
    finally:
        bpy.data.images.remove(img)


def _fit_panel(arr: Any, panel_w: int, panel_h: int, label: str) -> Any:
    import numpy as np

    if arr.shape[0] == panel_h and arr.shape[1] == panel_w:
        panel = arr.copy()
    else:
        panel = np.full((panel_h, panel_w, 3), (22, 24, 28), dtype=np.uint8)
        fit = min(panel_w / arr.shape[1], (panel_h - 48) / arr.shape[0])
        rw = max(1, int(round(arr.shape[1] * fit)))
        rh = max(1, int(round(arr.shape[0] * fit)))
        resized = _resize_rgb_bilinear(arr, rw, rh)
        y_off = 44 + max(0, (panel_h - 44 - rh) // 2)
        x_off = max(0, (panel_w - rw) // 2)
        panel[y_off : y_off + rh, x_off : x_off + rw] = resized
    _draw_banner_label(panel, label, scale=3)
    return panel


def _configure_render_engine(scene: Any, *, engine: str, use_gpu: bool, samples: int, view_transform: str) -> str:
    import bpy

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = view_transform
    scene.view_settings.look = "None"
    scene.view_settings.exposure = -0.18 if view_transform == "AgX" else 0.0

    if engine == "BLENDER_EEVEE":
        scene.render.engine = "BLENDER_EEVEE"
        try:
            scene.eevee.taa_render_samples = max(16, int(samples))
        except Exception:
            pass
        return "BLENDER_EEVEE"

    scene.render.engine = "CYCLES"
    scene.cycles.samples = max(1, int(samples))
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    if not use_gpu:
        scene.cycles.device = "CPU"
        return "CYCLES:CPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    cuda_devs = [d for d in prefs.devices if d.type == "CUDA"]
    if not cuda_devs:
        scene.cycles.device = "CPU"
        return "CYCLES:CPU"
    for d in prefs.devices:
        d.use = d.type == "CUDA"
    scene.cycles.device = "GPU"
    return f"CYCLES:CUDA:{','.join(d.name for d in cuda_devs)}"


def _setup_studio_world_and_lights(scene: Any) -> None:
    import bpy
    import mathutils

    world = bpy.data.worlds.new("S3StudioWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.18, 0.20, 0.23, 1.0)
        bg.inputs["Strength"].default_value = 0.56

    def _add_area(name: str, loc: tuple[float, float, float], tgt: tuple[float, float, float], eng: float, sz: float, col: tuple[float, float, float]) -> None:
        ld = bpy.data.lights.new(name, type="AREA")
        ld.energy = eng
        ld.size = sz
        ld.color = col
        lo = bpy.data.objects.new(name, ld)
        scene.collection.objects.link(lo)
        lo.location = loc
        lo.rotation_euler = (mathutils.Vector(tgt) - mathutils.Vector(loc)).to_track_quat("-Z", "Y").to_euler()

    _add_area("KeyLight", (1.90, -2.35, 2.20), (0.0, 0.0, 0.55), 112.0, 1.5, (1.0, 0.98, 0.95))
    _add_area("FillLight", (-2.15, -1.80, 1.10), (0.0, 0.0, 0.50), 46.0, 2.5, (0.93, 0.96, 1.0))
    _add_area("RimLight", (-0.50, 2.35, 1.95), (0.0, 0.0, 0.65), 88.0, 1.2, (1.0, 1.0, 1.0))
    _add_area("LowBounce", (0.0, -0.90, -0.35), (0.0, 0.0, 0.45), 14.0, 1.9, (0.96, 0.97, 1.0))


def _import_and_normalize(mesh_path: Path, framing_mode: str = "object") -> tuple[Any, list[Any]]:
    import bpy
    import mathutils

    before = set(bpy.data.objects)
    ext = mesh_path.suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(mesh_path), merge_vertices=False, import_shading="NORMALS")
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(mesh_path), forward_axis="NEGATIVE_Z", up_axis="Y")
    else:
        raise ValueError(f"Unsupported mesh extension: {mesh_path}")

    meshes = [o for o in set(bpy.data.objects) - before if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No MESH objects imported from {mesh_path}")
    if len(meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
        meshes = [bpy.context.view_layer.objects.active]

    obj = meshes[0]
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.data.polygons.foreach_set("use_smooth", [True] * len(obj.data.polygons))
    obj.data.update()

    pts = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    min_x, max_x = min(p.x for p in pts), max(p.x for p in pts)
    min_y, max_y = min(p.y for p in pts), max(p.y for p in pts)
    min_z, max_z = min(p.z for p in pts), max(p.z for p in pts)
    sx, sy, sz = max_x - min_x, max_y - min_y, max_z - min_z
    cx, cy, cz = 0.5 * (min_x + max_x), 0.5 * (min_y + max_y), 0.5 * (min_z + max_z)

    root = bpy.data.objects.new("StudioNormRoot", None)
    bpy.context.scene.collection.objects.link(root)
    obj.parent = root
    if framing_mode == "object":
        extent = max(sx, sy, sz, 1e-6)
        sc = 0.85 / extent
        root.scale = (sc, sc, sc)
        root.location = (-cx * sc, -cy * sc, 0.50 - cz * sc)
    else:
        sc = 1.0 / max(sz, 1e-6)
        root.scale = (sc, sc, sc)
        root.location = (-cx * sc, -cy * sc, -min_z * sc)
    bpy.context.view_layer.update()

    mw_norm = obj.matrix_world.copy()
    obj.parent = None
    obj.matrix_world = mw_norm
    bpy.data.objects.remove(root, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj, [obj]


def _ensure_gltf_occlusion_group() -> Any:
    import bpy

    grp_name = "glTF Material Output"
    if grp_name in bpy.data.node_groups:
        return bpy.data.node_groups[grp_name]
    ng = bpy.data.node_groups.new(grp_name, "ShaderNodeTree")
    ng.interface.new_socket(name="Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
    ng.nodes.new("NodeGroupInput")
    return ng


def _build_pbr_material(
    name: str,
    *,
    albedo_path: Path,
    normal_path: Path,
    orm_path: Path,
    render_ao_mul: float = 0.0,
    specular_ior_level: float = 0.50,
) -> Any:
    """Build glTF-2.0-compliant Principled BSDF + `glTF Material Output` Occlusion material."""
    import bpy

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = float(specular_ior_level)
    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    alb_img = bpy.data.images.load(str(albedo_path), check_existing=True)
    alb_img.colorspace_settings.name = "sRGB"
    alb_tex = nt.nodes.new("ShaderNodeTexImage")
    alb_tex.image = alb_img
    alb_tex.extension = "EXTEND"

    orm_img = bpy.data.images.load(str(orm_path), check_existing=True)
    orm_img.colorspace_settings.name = "Non-Color"
    orm_tex = nt.nodes.new("ShaderNodeTexImage")
    orm_tex.image = orm_img
    orm_tex.extension = "EXTEND"

    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(orm_tex.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])

    gltf_out = nt.nodes.new("ShaderNodeGroup")
    gltf_out.node_tree = _ensure_gltf_occlusion_group()
    nt.links.new(sep.outputs["Red"], gltf_out.inputs["Occlusion"])

    if render_ao_mul > 1e-4:
        mix_ao = nt.nodes.new("ShaderNodeMix")
        mix_ao.data_type = "RGBA"
        mix_ao.blend_type = "MULTIPLY"
        mix_ao.inputs[0].default_value = float(render_ao_mul)
        nt.links.new(alb_tex.outputs["Color"], mix_ao.inputs[6])
        nt.links.new(sep.outputs["Red"], mix_ao.inputs[7])
        nt.links.new(mix_ao.outputs[2], bsdf.inputs["Base Color"])
    else:
        nt.links.new(alb_tex.outputs["Color"], bsdf.inputs["Base Color"])

    nrm_img = bpy.data.images.load(str(normal_path), check_existing=True)
    nrm_img.colorspace_settings.name = "Non-Color"
    nrm_tex = nt.nodes.new("ShaderNodeTexImage")
    nrm_tex.image = nrm_img
    nrm_tex.extension = "EXTEND"
    nrm_node = nt.nodes.new("ShaderNodeNormalMap")
    nrm_node.space = "TANGENT"
    nrm_node.inputs["Strength"].default_value = 1.0
    nt.links.new(nrm_tex.outputs["Color"], nrm_node.inputs["Color"])
    nt.links.new(nrm_node.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def _export_clean_glb(obj: Any, out_glb: Path) -> None:
    import bpy

    out_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=str(out_glb),
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_tangents=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
    )


def _render_views(
    scene: Any,
    cam_obj: Any,
    cam_data: Any,
    view_specs: Sequence[dict[str, Any]],
    out_dir: Path,
    prefix: str,
) -> dict[str, str]:
    import bpy
    import mathutils

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for spec in view_specs:
        cam_obj.location = spec["location"]
        direction = mathutils.Vector(spec["target"]) - mathutils.Vector(spec["location"])
        cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        if spec["ortho"]:
            cam_data.type = "ORTHO"
            cam_data.ortho_scale = float(spec["ortho_scale"])
        else:
            cam_data.type = "PERSP"
            cam_data.lens = float(spec["lens"])
        cam_data.clip_start = 0.01
        cam_data.clip_end = 50.0
        scene.render.resolution_x = int(spec["width"])
        scene.render.resolution_y = int(spec["height"])
        scene.render.resolution_percentage = 100
        out_p = out_dir / f"{prefix}_{spec['key']}.png"
        scene.render.filepath = str(out_p)
        bpy.ops.render.render(write_still=True)
        paths[spec["key"]] = str(out_p)
    return paths


def run_pbr_studio_export(args: argparse.Namespace) -> dict[str, Any]:
    import bpy
    import numpy as np

    t0 = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    _configure_render_engine(scene, engine="CYCLES", use_gpu=bool(args.use_gpu), samples=int(args.samples), view_transform="AgX")
    _setup_studio_world_and_lights(scene)
    cam_d = bpy.data.cameras.new("StudioCam")
    cam_o = bpy.data.objects.new("StudioCam", cam_d)
    scene.collection.objects.link(cam_o)
    scene.camera = cam_o

    _, meshes = _import_and_normalize(Path(args.mesh), str(args.framing_mode))
    obj = meshes[0]

    mat_export = _build_pbr_material(
        "AssetPBR_GLTF",
        albedo_path=Path(args.albedo),
        normal_path=Path(args.normal_4k),
        orm_path=Path(args.orm),
        render_ao_mul=0.0,
    )
    obj.data.materials.clear()
    obj.data.materials.append(mat_export)
    out_glb = out_dir / args.glb_name
    _export_clean_glb(obj, out_glb)

    mat_rnd = _build_pbr_material(
        "AssetPBR_Render",
        albedo_path=Path(args.albedo),
        normal_path=Path(args.normal_4k),
        orm_path=Path(args.orm),
        render_ao_mul=0.55,
    )
    obj.data.materials[0] = mat_rnd
    views = _render_views(scene, cam_o, cam_d, OBJECT_VIEW_SPECS, out_dir / "pbr_views", "pbr")

    pw, ph, pad = 768, 768, 6
    sheet = np.full((2 * ph + pad, 3 * pw + 2 * pad, 3), (18, 20, 24), dtype=np.uint8)
    for idx, spec in enumerate(OBJECT_VIEW_SPECS):
        arr = _read_png_rgb_in_blender(Path(views[spec["key"]]))
        pan = _fit_panel(arr, pw, ph, f"PBR | {spec['label']}")
        r, c = divmod(idx, 3)
        sheet[r * (ph + pad) : r * (ph + pad) + ph, c * (pw + pad) : c * (pw + pad) + pw] = pan
    sheet_path = out_dir / "asset_pbr_6view.png"
    _write_png_rgb_in_blender(sheet_path, sheet)

    res = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "glb_path": str(out_glb),
        "sheet_6view": str(sheet_path),
        "views": views,
    }
    if args.meta_json:
        Path(args.meta_json).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return res


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Headless Blender 5.3 S3 Studio Renderer & glTF Exporter.")
    p.add_argument("--mesh", required=True)
    p.add_argument("--albedo", required=True)
    p.add_argument("--normal-4k", required=True)
    p.add_argument("--orm", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--glb-name", default="asset_pbr.glb")
    p.add_argument("--framing-mode", choices=("object", "standing"), default="object")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--meta-json", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        raw = sys.argv[1:]
        if "--" in raw:
            raw = raw[raw.index("--") + 1 :]
    else:
        raw = list(argv)
    args = _build_parser().parse_args(raw)
    run_pbr_studio_export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
