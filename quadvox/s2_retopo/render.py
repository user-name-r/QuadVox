"""Headless Blender Cycles studio renderer for S2 retopology visual inspection.

Supports 5 rendering states matching `s1_backend/render_preview.py` camera framing
(normalized character height = 1.0, `y_up_z_front` mapped to Blender Z-up):
- `wireframe_clay`: neutral clay + crisp Cycles `ShaderNodeWireframe` overlay
- `smooth_lowpoly`: `shade_smooth` without normal map (exposes poles/pinches)
- `normal_mapped`: `shade_smooth` with Non-Color tangent-space normal map
- `highpoly_clay`: original highpoly neutral clay reference
- `semantic_clusters`: colored 3D semantic cluster / weight inspection

For every requested state, writes:
1. 6 independent full-resolution PNGs (`{prefix}_{state}_{view}.png`:
   768x900 for `front`, `three_quarter`, `left`, `back`, `right`; 1024x1024 for `head`).
2. 1 labeled 3x2 sheet (`{prefix}_{state}_sheet.png`) for fast comparison.

Top-level imports are strictly Python stdlib so this script can be run both from
`.venv` Python (spawning headless Blender) and directly via `blender --python`.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from quadvox.tools.external_tools import resolve_blender_bin

DEFAULT_BLENDER_BIN = resolve_blender_bin()

VALID_STATES = (
    "wireframe_clay",
    "smooth_lowpoly",
    "normal_mapped",
    "highpoly_clay",
    "semantic_clusters",
)

VIEW_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "front",
        "label": "FRONT (+Z)",
        "location": (0.0, -3.2, 0.50),
        "target": (0.0, 0.0, 0.50),
        "ortho": True,
        "ortho_scale": 1.14,
        "width": 768,
        "height": 900,
    },
    {
        "key": "three_quarter",
        "label": "3/4 VIEW",
        "location": (1.65, -2.15, 0.72),
        "target": (0.0, 0.0, 0.50),
        "ortho": False,
        "lens": 85.0,
        "width": 768,
        "height": 900,
    },
    {
        "key": "left",
        "label": "LEFT (+X)",
        "location": (3.2, 0.0, 0.50),
        "target": (0.0, 0.0, 0.50),
        "ortho": True,
        "ortho_scale": 1.14,
        "width": 768,
        "height": 900,
    },
    {
        "key": "back",
        "label": "BACK (-Z)",
        "location": (0.0, 3.2, 0.50),
        "target": (0.0, 0.0, 0.50),
        "ortho": True,
        "ortho_scale": 1.14,
        "width": 768,
        "height": 900,
    },
    {
        "key": "right",
        "label": "RIGHT (-X)",
        "location": (-3.2, 0.0, 0.50),
        "target": (0.0, 0.0, 0.50),
        "ortho": True,
        "ortho_scale": 1.14,
        "width": 768,
        "height": 900,
    },
    {
        "key": "head",
        "label": "HEAD CLOSE-UP",
        "location": (0.25, -1.04, 0.87),
        "target": (0.0, 0.0, 0.86),
        "ortho": False,
        "lens": 90.0,
        "width": 1024,
        "height": 1024,
    },
)


@dataclass
class RenderResult:
    """Output paths and runtime metadata produced by `render_mesh_states`."""

    mesh_path: str
    out_dir: str
    prefix: str
    states: list[str]
    samples: int
    use_gpu: bool
    device_used: str
    blender_version: str
    elapsed_s: float
    outputs: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        return target


_FONT_5X7: dict[str, tuple[str, ...]] = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "/": ("00001", "00010", "00100", "00100", "01000", "10000", "00000"),
    "|": ("00100", "00100", "00100", "00100", "00100", "00100", "00100"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00110", "01000", "10000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11100", "10010", "10001", "10001", "10001", "10010", "11100"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
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
    x0 = 10
    y0 = 7
    fg = (238, 240, 244)
    h, w, _ = img_rgb.shape
    for ch in text.upper():
        glyph = _FONT_5X7.get(ch, _FONT_5X7[" "])
        for gy, row_str in enumerate(glyph):
            for gx, bit in enumerate(row_str):
                if bit == "1":
                    py0 = y0 + gy * scale
                    px0 = x0 + gx * scale
                    if py0 + scale <= h and px0 + scale <= w:
                        img_rgb[py0 : py0 + scale, px0 : px0 + scale] = fg
        x0 += 6 * scale


def _resize_rgb_bilinear(img_rgb: Any, target_w: int, target_h: int) -> Any:
    """Pure-NumPy bilinear resize for `(H, W, 3)` uint8 arrays (no PIL/OIIO dependency)."""
    import numpy as np

    src_h, src_w, _ = img_rgb.shape
    if src_h == target_h and src_w == target_w:
        return img_rgb

    ys = np.linspace(0.0, src_h - 1.0, target_h, dtype=np.float32)
    xs = np.linspace(0.0, src_w - 1.0, target_w, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]

    src = img_rgb.astype(np.float32)
    top = (1.0 - wx) * src[y0][:, x0] + wx * src[y0][:, x1]
    bot = (1.0 - wx) * src[y1][:, x0] + wx * src[y1][:, x1]
    out = (1.0 - wy) * top + wy * bot
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _read_png_rgb_in_blender(fpath: Path) -> Any:
    import numpy as np

    try:
        import OpenImageIO as oiio

        inp = oiio.ImageInput.open(str(fpath))
        if inp is None:
            raise RuntimeError(f"OpenImageIO failed to open {fpath}")
        try:
            arr = inp.read_image(format=oiio.UINT8)
        finally:
            inp.close()
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]
        return np.ascontiguousarray(arr, dtype=np.uint8)
    except ImportError:
        import bpy

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
    import numpy as np

    fpath.parent.mkdir(parents=True, exist_ok=True)
    h, w, c = arr_rgb.shape
    try:
        import OpenImageIO as oiio

        out = oiio.ImageOutput.create(str(fpath))
        if out is None:
            raise RuntimeError(f"OpenImageIO failed to create output for {fpath}")
        try:
            spec = oiio.ImageSpec(w, h, c, oiio.UINT8)
            if not out.open(str(fpath), spec):
                raise RuntimeError(f"Failed to open {fpath}: {out.geterror()}")
            if not out.write_image(np.ascontiguousarray(arr_rgb, dtype=np.uint8)):
                raise RuntimeError(f"Failed to write {fpath}: {out.geterror()}")
        finally:
            out.close()
    except ImportError:
        import bpy

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


def _set_material_nodes(mat: Any) -> Any:
    if mat.node_tree is None:
        mat.use_nodes = True
    return mat.node_tree


def _configure_cycles_render(scene: Any, *, samples: int, use_gpu: bool) -> str:
    import bpy

    scene.render.engine = "CYCLES"
    scene.cycles.samples = max(1, int(samples))
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass

    if not use_gpu:
        scene.cycles.device = "CPU"
        return "CPU"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    cuda_devs = [d for d in prefs.devices if d.type == "CUDA"]
    if not cuda_devs:
        raise RuntimeError("CUDA GPU requested for Cycles rendering, but no CUDA devices were detected.")
    for dev in prefs.devices:
        dev.use = dev.type == "CUDA"
    scene.cycles.device = "GPU"
    return f"CUDA:{','.join(d.name for d in cuda_devs)}"


def _find_embedded_normal_image(meshes: Sequence[Any]) -> Any | None:
    """Locate an existing normal-map Image linked inside imported GLB materials."""
    for obj in meshes:
        for mat in obj.data.materials:
            if mat is None or mat.node_tree is None:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "NORMAL_MAP":
                    col_in = node.inputs.get("Color")
                    if col_in is not None and col_in.is_linked:
                        src_node = col_in.links[0].from_node
                        if src_node.type == "TEX_IMAGE" and src_node.image is not None:
                            return src_node.image
    # Fallback: check any image attached to materials if named like normal.
    for obj in meshes:
        for mat in obj.data.materials:
            if mat is None or mat.node_tree is None:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    nrm_links = [
                        lk
                        for lk in node.outputs.get("Color", []).links
                        if lk.to_node.type == "NORMAL_MAP"
                    ] if "Color" in node.outputs else []
                    if nrm_links or "norm" in node.image.name.lower():
                        return node.image
    return None


def _apply_labels_npz_to_meshes(meshes: Sequence[Any], labels_npz: Path) -> str:
    """Load per-vertex cluster IDs or weights from `.npz` and write a Blender color attribute."""
    import numpy as np

    if not labels_npz.exists():
        raise FileNotFoundError(f"Labels/weights npz not found: {labels_npz}")

    with np.load(str(labels_npz), allow_pickle=False) as data:
        keys = set(data.files)
        if "part_ids" in keys:
            arr = np.asarray(data["part_ids"], dtype=np.int64).reshape(-1)
            kind = "clusters"
        elif "cluster_ids" in keys:
            arr = np.asarray(data["cluster_ids"], dtype=np.int64).reshape(-1)
            kind = "clusters"
        elif "labels" in keys:
            arr = np.asarray(data["labels"], dtype=np.int64).reshape(-1)
            kind = "clusters"
        elif "weights" in keys:
            arr = np.asarray(data["weights"], dtype=np.float64).reshape(-1)
            kind = "weights"
        else:
            raise KeyError(
                f"No supported cluster/weight key (part_ids/cluster_ids/labels/weights) in {labels_npz}: {sorted(keys)}"
            )

    total_verts = sum(len(obj.data.vertices) for obj in meshes)
    if len(arr) != total_verts:
        raise ValueError(
            f"Labels array length ({len(arr)}) does not match mesh vertex count ({total_verts})"
        )

    rgba_all = np.zeros((total_verts, 4), dtype=np.float32)
    rgba_all[:, 3] = 1.0
    if kind == "clusters":
        unique_ids = sorted({int(c) for c in arr})
        golden = 0.6180339887498949
        palette: dict[int, tuple[float, float, float]] = {-1: (0.45, 0.45, 0.45)}
        for idx, cid in enumerate([c for c in unique_ids if c >= 0]):
            hue = (idx * golden) % 1.0
            sat = 0.72 if (idx % 2 == 0) else 0.88
            val = 0.92 if (idx % 3 != 2) else 0.78
            palette[cid] = colorsys.hsv_to_rgb(hue, sat, val)
        for cid in unique_ids:
            rgb = palette.get(cid, (0.5, 0.5, 0.5))
            rgba_all[arr == cid, 0] = rgb[0]
            rgba_all[arr == cid, 1] = rgb[1]
            rgba_all[arr == cid, 2] = rgb[2]
    else:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        t = np.zeros_like(arr, dtype=np.float64) if hi <= lo + 1e-12 else np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        # Cool blue -> teal -> warm amber -> hot red heatmap.
        rgba_all[:, 0] = np.clip(1.6 * t - 0.1, 0.12, 1.0)
        rgba_all[:, 1] = np.clip(0.85 - np.abs(t - 0.45) * 1.4, 0.18, 0.92)
        rgba_all[:, 2] = np.clip(0.95 - 1.4 * t, 0.08, 0.95)

    offset = 0
    attr_name = "S2SemanticColor"
    for obj in meshes:
        mesh = obj.data
        nv = len(mesh.vertices)
        sub = rgba_all[offset : offset + nv]
        offset += nv
        if attr_name in mesh.color_attributes:
            mesh.color_attributes.remove(mesh.color_attributes[attr_name])
        cattr = mesh.color_attributes.new(name=attr_name, type="FLOAT_COLOR", domain="POINT")
        cattr.data.foreach_set("color", np.ascontiguousarray(sub, dtype=np.float32).ravel())
        mesh.color_attributes.active_color = cattr
    return attr_name


def _build_state_material(
    *,
    state: str,
    meshes: Sequence[Any],
    normal_map_path: Path | None,
    labels_npz: Path | None,
    embedded_normal_img: Any | None,
) -> Any:
    import bpy

    mat = bpy.data.materials.new(f"S2_{state}_Mat")
    nt = _set_material_nodes(mat)
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

    bsdf.inputs["Metallic"].default_value = 0.0
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.35

    if state == "wireframe_clay":
        bsdf.inputs["Roughness"].default_value = 0.68
        wf = nt.nodes.new("ShaderNodeWireframe")
        wf.use_pixel_size = True
        wf.inputs["Size"].default_value = 1.05
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        # ShaderNodeMix RGBA sockets: 0=Factor(VALUE), 6=A(RGBA), 7=B(RGBA), output 2=Result(RGBA).
        mix.inputs[6].default_value = (0.62, 0.62, 0.63, 1.0)
        mix.inputs[7].default_value = (0.05, 0.06, 0.08, 1.0)
        nt.links.new(wf.outputs["Factor"], mix.inputs[0])
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        return mat

    if state == "smooth_lowpoly":
        bsdf.inputs["Base Color"].default_value = (0.58, 0.58, 0.59, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.56
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.40
        return mat

    if state == "highpoly_clay":
        bsdf.inputs["Base Color"].default_value = (0.58, 0.58, 0.59, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.64
        return mat

    if state == "normal_mapped":
        bsdf.inputs["Base Color"].default_value = (0.58, 0.58, 0.59, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.60
        if normal_map_path is not None:
            if not normal_map_path.exists():
                raise FileNotFoundError(f"Normal map PNG not found: {normal_map_path}")
            nrm_img = bpy.data.images.load(str(normal_map_path), check_existing=True)
        elif embedded_normal_img is not None:
            nrm_img = embedded_normal_img
        else:
            raise ValueError(
                "normal_mapped render state requires either --normal-map or an embedded normal map in the GLB."
            )
        nrm_img.colorspace_settings.name = "Non-Color"
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = nrm_img
        tex.interpolation = "Linear"
        tex.extension = "EXTEND"
        nrm_node = nt.nodes.new("ShaderNodeNormalMap")
        nrm_node.space = "TANGENT"
        nrm_node.inputs["Strength"].default_value = 1.0
        nt.links.new(tex.outputs["Color"], nrm_node.inputs["Color"])
        nt.links.new(nrm_node.outputs["Normal"], bsdf.inputs["Normal"])
        return mat

    if state == "semantic_clusters":
        bsdf.inputs["Roughness"].default_value = 0.68
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.25
        if labels_npz is not None:
            attr_name = _apply_labels_npz_to_meshes(meshes, labels_npz)
        else:
            active_attr = None
            for obj in meshes:
                cattrs = obj.data.color_attributes
                if len(cattrs) > 0:
                    active_attr = (
                        cattrs.active_color.name
                        if cattrs.active_color is not None
                        else cattrs[0].name
                    )
                    break
            if active_attr is None:
                raise ValueError(
                    "semantic_clusters render state requires vertex color attributes in the mesh or --labels-npz."
                )
            attr_name = active_attr

        attr_node = nt.nodes.new("ShaderNodeAttribute")
        attr_node.attribute_name = attr_name
        nt.links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])
        return mat

    raise ValueError(f"Unsupported render state '{state}'. Valid: {VALID_STATES}")


def _run_render_in_blender(
    *,
    mesh_path: Path,
    out_dir: Path,
    prefix: str,
    states: Sequence[str],
    normal_map_path: Path | None,
    labels_npz: Path | None,
    samples: int,
    use_gpu: bool,
    align_tripo: bool,
    yaw_deg: float,
    meta_json_path: Path | None,
) -> RenderResult:
    import bpy
    import mathutils
    import numpy as np

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    t0 = time.perf_counter()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    device_used = _configure_cycles_render(scene, samples=samples, use_gpu=use_gpu)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.15

    # Studio background world matching s1_backend/render_preview.py.
    world = bpy.data.worlds.new("StudioWorld")
    scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node is not None:
        bg_node.inputs["Color"].default_value = (0.16, 0.17, 0.19, 1.0)
        bg_node.inputs["Strength"].default_value = 0.65

    # Import mesh and detect any embedded normal map BEFORE replacing materials.
    before_objs = set(bpy.data.objects)
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
        raise ValueError(f"Unsupported mesh extension for Blender import: {mesh_path}")

    imported = [obj for obj in bpy.data.objects if obj not in before_objs]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No MESH objects imported from {mesh_path}")

    embedded_normal_img = _find_embedded_normal_image(meshes)
    has_custom_normals_flags = {obj.name: bool(obj.data.has_custom_normals) for obj in meshes}

    # Normalize character in memory to height=1.0 standing on z=0..1 facing -Y.
    root_empty = bpy.data.objects.new("RenderRoot", None)
    scene.collection.objects.link(root_empty)
    for obj in imported:
        if obj.parent is None:
            obj.parent = root_empty

    extra_yaw = (-90.0 if align_tripo else 0.0) + float(yaw_deg)
    if extra_yaw != 0.0:
        root_empty.rotation_euler = (0.0, 0.0, math.radians(extra_yaw))
    bpy.context.view_layer.update()

    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")
    for mobj in meshes:
        mw = mobj.matrix_world
        for corner in mobj.bound_box:
            p = mw @ mathutils.Vector(corner)
            min_x = min(min_x, p.x)
            min_y = min(min_y, p.y)
            min_z = min(min_z, p.z)
            max_x = max(max_x, p.x)
            max_y = max(max_y, p.y)
            max_z = max(max_z, p.z)

    height = max_z - min_z
    if height <= 1e-8:
        raise ValueError(f"Imported mesh has degenerate height {height}: {mesh_path}")
    scale = 1.0 / height
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    root_empty.scale = (scale, scale, scale)
    root_empty.location = (-cx * scale, -cy * scale, -min_z * scale)
    bpy.context.view_layer.update()

    # Clean 3-point studio lighting + gentle lower bounce matching render_preview.py.
    def _add_area_light(
        name: str,
        location: tuple[float, float, float],
        target: tuple[float, float, float],
        energy: float,
        size: float,
        color: tuple[float, float, float],
    ) -> None:
        ldata = bpy.data.lights.new(name, type="AREA")
        ldata.energy = energy
        ldata.size = size
        ldata.color = color
        lobj = bpy.data.objects.new(name, ldata)
        scene.collection.objects.link(lobj)
        lobj.location = location
        direction = mathutils.Vector(target) - mathutils.Vector(location)
        lobj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    _add_area_light("KeyLight", (1.85, -2.35, 2.25), (0.0, 0.0, 0.55), 145.0, 1.5, (1.0, 0.98, 0.95))
    _add_area_light("FillLight", (-2.15, -1.75, 1.05), (0.0, 0.0, 0.50), 48.0, 2.3, (0.90, 0.94, 1.0))
    _add_area_light("RimLight", (-0.55, 2.35, 1.90), (0.0, 0.0, 0.65), 110.0, 1.2, (1.0, 1.0, 1.0))
    _add_area_light("LowBounce", (0.0, -0.85, -0.40), (0.0, 0.0, 0.44), 22.0, 1.8, (0.95, 0.96, 1.0))

    cam_data = bpy.data.cameras.new("StudioCam")
    cam_obj = bpy.data.objects.new("StudioCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    def _aim_camera(spec: dict[str, Any]) -> None:
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

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, str]] = {}

    for state in states:
        if state not in VALID_STATES:
            raise ValueError(f"Unsupported render state '{state}'. Valid: {VALID_STATES}")

        # Configure polygon shading per state without destroying custom normals on normal_mapped.
        for mobj in meshes:
            if state == "wireframe_clay":
                mobj.data.polygons.foreach_set("use_smooth", [False] * len(mobj.data.polygons))
                mobj.data.update()
            elif state in ("smooth_lowpoly", "normal_mapped"):
                if not has_custom_normals_flags.get(mobj.name, False):
                    mobj.data.polygons.foreach_set("use_smooth", [True] * len(mobj.data.polygons))
                    mobj.data.update()
                else:
                    mobj.data.polygons.foreach_set("use_smooth", [True] * len(mobj.data.polygons))
                    mobj.data.update()
            elif state == "semantic_clusters":
                mobj.data.polygons.foreach_set("use_smooth", [True] * len(mobj.data.polygons))
                mobj.data.update()

        mat = _build_state_material(
            state=state,
            meshes=meshes,
            normal_map_path=normal_map_path,
            labels_npz=labels_npz,
            embedded_normal_img=embedded_normal_img,
        )
        for mobj in meshes:
            mobj.data.materials.clear()
            mobj.data.materials.append(mat)

        state_paths: dict[str, str] = {}
        panels: list[np.ndarray] = []
        panel_w, panel_h = 768, 900

        for spec in VIEW_SPECS:
            _aim_camera(spec)
            view_key = spec["key"]
            img_path = out_dir / f"{prefix}_{state}_{view_key}.png"
            scene.render.filepath = str(img_path)
            bpy.ops.render.render(write_still=True)
            state_paths[view_key] = str(img_path)

            arr = _read_png_rgb_in_blender(img_path)
            if arr.shape[0] != panel_h or arr.shape[1] != panel_w:
                # Fit square 1024x1024 HEAD view into 768x900 panel cleanly with zero aspect distortion.
                panel = np.full((panel_h, panel_w, 3), (18, 20, 24), dtype=np.uint8)
                fit_scale = min(panel_w / arr.shape[1], (panel_h - 60) / arr.shape[0])
                rw = max(1, int(round(arr.shape[1] * fit_scale)))
                rh = max(1, int(round(arr.shape[0] * fit_scale)))
                resized = _resize_rgb_bilinear(arr, rw, rh)
                y_off = 52 + max(0, (panel_h - 52 - rh) // 2)
                x_off = max(0, (panel_w - rw) // 2)
                panel[y_off : y_off + rh, x_off : x_off + rw] = resized
            else:
                panel = arr.copy()
            _draw_banner_label(panel, spec["label"], scale=3)
            panels.append(panel)

        cols, rows = 3, 2
        pad = 6
        sheet_w = cols * panel_w + (cols - 1) * pad
        sheet_h = rows * panel_h + (rows - 1) * pad
        sheet = np.full((sheet_h, sheet_w, 3), (18, 20, 24), dtype=np.uint8)
        for idx, pan in enumerate(panels):
            r, c = divmod(idx, cols)
            y0 = r * (panel_h + pad)
            x0 = c * (panel_w + pad)
            sheet[y0 : y0 + panel_h, x0 : x0 + panel_w] = pan

        sheet_path = out_dir / f"{prefix}_{state}_sheet.png"
        _write_png_rgb_in_blender(sheet_path, sheet)
        state_paths["sheet"] = str(sheet_path)
        outputs[state] = state_paths

    res = RenderResult(
        mesh_path=str(mesh_path),
        out_dir=str(out_dir),
        prefix=str(prefix),
        states=list(states),
        samples=max(1, int(samples)),
        use_gpu=bool(use_gpu),
        device_used=device_used,
        blender_version=".".join(str(x) for x in bpy.app.version),
        elapsed_s=time.perf_counter() - t0,
        outputs=outputs,
    )
    if meta_json_path is not None:
        res.write_json(meta_json_path)
    return res


def render_mesh_states(
    mesh_path: str | Path,
    out_dir: str | Path,
    prefix: str,
    states: Sequence[str] = ("wireframe_clay", "smooth_lowpoly", "normal_mapped"),
    *,
    normal_map_path: str | Path | None = None,
    labels_npz: str | Path | None = None,
    samples: int = 32,
    use_gpu: bool = False,
    align_tripo: bool = False,
    yaw_deg: float = 0.0,
    blender_bin: str | Path = DEFAULT_BLENDER_BIN,
    report_json_path: str | Path | None = None,
) -> RenderResult:
    """Render 6 full-resolution views + 3x2 sheet per state via headless Blender Cycles."""
    m_path = Path(mesh_path)
    o_dir = Path(out_dir)
    if not m_path.exists():
        raise FileNotFoundError(f"Mesh not found: {m_path}")

    state_list = [str(s) for s in states]
    for st in state_list:
        if st not in VALID_STATES:
            raise ValueError(f"Unsupported render state '{st}'. Valid: {VALID_STATES}")

    nrm_p = Path(normal_map_path) if normal_map_path is not None else None
    lbl_p = Path(labels_npz) if labels_npz is not None else None
    rep_p = Path(report_json_path) if report_json_path is not None else None

    if "bpy" in sys.modules:
        return _run_render_in_blender(
            mesh_path=m_path,
            out_dir=o_dir,
            prefix=prefix,
            states=state_list,
            normal_map_path=nrm_p,
            labels_npz=lbl_p,
            samples=int(samples),
            use_gpu=bool(use_gpu),
            align_tripo=bool(align_tripo),
            yaw_deg=float(yaw_deg),
            meta_json_path=rep_p,
        )

    blender_path = Path(blender_bin)
    if not blender_path.exists():
        raise FileNotFoundError(f"Blender binary not found: {blender_path}")

    o_dir.mkdir(parents=True, exist_ok=True)
    cleanup_meta = rep_p is None
    meta_path = (
        rep_p
        if rep_p is not None
        else (o_dir / f"._{prefix}_render_meta_{os.getpid()}.json")
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
        "--mesh",
        str(m_path),
        "--out-dir",
        str(o_dir),
        "--prefix",
        str(prefix),
        "--states",
        *state_list,
        "--samples",
        str(int(samples)),
        "--yaw-deg",
        str(float(yaw_deg)),
        "--report-json",
        str(meta_path),
    ]
    if nrm_p is not None:
        cmd.extend(["--normal-map", str(nrm_p)])
    if lbl_p is not None:
        cmd.extend(["--labels-npz", str(lbl_p)])
    if use_gpu:
        cmd.append("--use-gpu")
    if align_tripo:
        cmd.append("--align-tripo")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not meta_path.exists():
            raise RuntimeError(
                f"Blender Cycles render failed (code={proc.returncode}):\n"
                f"CMD: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
            )
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        return RenderResult(**payload)
    finally:
        if cleanup_meta and meta_path.exists():
            meta_path.unlink(missing_ok=True)


def render_state(
    mesh_path: str | Path,
    out_dir: str | Path,
    prefix: str,
    state: str = "highpoly_clay",
    *,
    normal_map_path: str | Path | None = None,
    labels_npz: str | Path | None = None,
    samples: int = 32,
    use_gpu: bool = False,
    align_tripo: bool = False,
    yaw_deg: float = 0.0,
    blender_bin: str | Path = DEFAULT_BLENDER_BIN,
    report_json_path: str | Path | None = None,
) -> dict[str, str]:
    """Convenience single-state wrapper returning `{view: png_path, 'sheet': sheet_path}`."""
    res = render_mesh_states(
        mesh_path=mesh_path,
        out_dir=out_dir,
        prefix=prefix,
        states=[state],
        normal_map_path=normal_map_path,
        labels_npz=labels_npz,
        samples=samples,
        use_gpu=use_gpu,
        align_tripo=align_tripo,
        yaw_deg=yaw_deg,
        blender_bin=blender_bin,
        report_json_path=report_json_path,
    )
    return dict(res.outputs[state])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless Blender Cycles multi-state S2 renderer.")
    parser.add_argument("--mesh", required=True, help="Target mesh path (.glb/.obj/.ply).")
    parser.add_argument("--out-dir", required=True, help="Output directory for view PNGs and 3x2 sheets.")
    parser.add_argument("--prefix", required=True, help="Filename prefix `{prefix}_{state}_{view}.png`.")
    parser.add_argument(
        "--states",
        nargs="+",
        default=["wireframe_clay", "smooth_lowpoly", "normal_mapped"],
        choices=list(VALID_STATES),
        help="One or more render states to produce.",
    )
    parser.add_argument("--normal-map", default=None, help="Optional tangent-space normal map PNG.")
    parser.add_argument("--labels-npz", default=None, help="Optional semantic cluster/weight .npz.")
    parser.add_argument("--samples", type=int, default=32, help="Cycles samples per pixel.")
    parser.add_argument("--use-gpu", action="store_true", help="Enable Cycles CUDA GPU rendering.")
    parser.add_argument("--align-tripo", action="store_true", help="Rotate mesh -90 deg around Z-up.")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="Additional yaw rotation in degrees.")
    parser.add_argument("--blender-bin", default=str(DEFAULT_BLENDER_BIN), help="Blender executable.")
    parser.add_argument("--report-json", default=None, help="Optional RenderResult JSON output.")
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
    res = render_mesh_states(
        mesh_path=args.mesh,
        out_dir=args.out_dir,
        prefix=args.prefix,
        states=args.states,
        normal_map_path=args.normal_map,
        labels_npz=args.labels_npz,
        samples=int(args.samples),
        use_gpu=bool(args.use_gpu),
        align_tripo=bool(args.align_tripo),
        yaw_deg=float(args.yaw_deg),
        blender_bin=args.blender_bin,
        report_json_path=args.report_json,
    )
    if "bpy" not in sys.modules:
        print(res.to_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
