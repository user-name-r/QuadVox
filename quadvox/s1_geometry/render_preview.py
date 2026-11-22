"""Headless Blender 5.3 studio clay renderer for S1 white-mesh inspection & reference comparison.

Can be executed in two ways:
1. Directly via Blender 5.3:
   `blender --background --python quadvox/s1_geometry/render_preview.py -- \\
       --mesh <glb> [--compare-ref <ref.glb>] \\
       [--align-ref] [--yaw-deg <deg>] --out <png>`
2. Via Python CLI or function import:
   `from quadvox.s1_geometry.render_preview import render_preview_sheet, render_reference_comparison`
   (automatically invokes `blender` as a subprocess).
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from quadvox.tools.external_tools import resolve_blender_bin

DEFAULT_BLENDER_BIN = resolve_blender_bin()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render studio clay preview / reference comparison sheet.")
    parser.add_argument("--mesh", required=True, help="Path to target .glb/.obj/.ply mesh.")
    parser.add_argument(
        "--compare-ref",
        "--compare-tripo",
        dest="compare_tripo",
        default=None,
        help="Optional path to reference GLB for side-by-side comparison sheet.",
    )
    parser.add_argument(
        "--align-ref",
        "--align-tripo",
        dest="align_tripo",
        action="store_true",
        help="Auto-rotate target mesh -90 deg yaw (for +X-facing reference GLB) in memory before rendering.",
    )
    parser.add_argument(
        "--yaw-deg",
        type=float,
        default=0.0,
        help="Additional rotation around +Y up-axis in degrees (in memory only).",
    )
    parser.add_argument("--out", required=True, help="Output PNG sheet path.")
    parser.add_argument(
        "--blender-bin",
        default=str(DEFAULT_BLENDER_BIN),
        help="Path to Blender 5.3 binary when invoked from standard Python.",
    )
    return parser


def _parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if argv is None:
        raw = sys.argv[1:]
        if "--" in raw:
            raw = raw[raw.index("--") + 1 :]
    else:
        raw = list(argv)
    return _build_arg_parser().parse_args(raw)


def _invoke_blender_subprocess(
    *,
    mesh: str | Path,
    out: str | Path,
    compare_tripo: str | Path | None = None,
    align_tripo: bool = False,
    yaw_deg: float = 0.0,
    blender_bin: str | Path = DEFAULT_BLENDER_BIN,
) -> Path:
    blender_path = Path(blender_bin)
    if not blender_path.exists():
        raise FileNotFoundError(f"Blender binary not found: {blender_path}")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
        str(mesh),
        "--out",
        str(out_path),
        "--yaw-deg",
        str(float(yaw_deg)),
    ]
    if compare_tripo is not None:
        cmd.extend(["--compare-tripo", str(compare_tripo)])
    if align_tripo:
        cmd.append("--align-tripo")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"Blender render failed (code={proc.returncode}):\n"
            f"CMD: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
        )
    return out_path


def render_preview_sheet(
    mesh: str | Path,
    out: str | Path,
    *,
    align_tripo: bool = False,
    yaw_deg: float = 0.0,
    blender_bin: str | Path = DEFAULT_BLENDER_BIN,
) -> Path:
    """Render 6-panel studio clay preview sheet (`FRONT`, `3/4`, `LEFT`, `BACK`, `RIGHT`, `HEAD`)."""
    if "bpy" in sys.modules:
        _run_in_blender(
            mesh_path=Path(mesh),
            out_path=Path(out),
            compare_tripo=None,
            align_tripo=align_tripo,
            yaw_deg=yaw_deg,
        )
        return Path(out)
    return _invoke_blender_subprocess(
        mesh=mesh,
        out=out,
        compare_tripo=None,
        align_tripo=align_tripo,
        yaw_deg=yaw_deg,
        blender_bin=blender_bin,
    )


def render_reference_comparison(
    mesh: str | Path,
    out: str | Path,
    *,
    ref_glb: str | Path,
    align_ref: bool = False,
    yaw_deg: float = 0.0,
    blender_bin: str | Path = DEFAULT_BLENDER_BIN,
) -> Path:
    """Render side-by-side 6-angle comparison sheet against a reference GLB."""
    if "bpy" in sys.modules:
        _run_in_blender(
            mesh_path=Path(mesh),
            out_path=Path(out),
            compare_tripo=Path(ref_glb),
            align_tripo=align_ref,
            yaw_deg=yaw_deg,
        )
        return Path(out)
    return _invoke_blender_subprocess(
        mesh=mesh,
        out=out,
        compare_tripo=ref_glb,
        align_tripo=align_ref,
        yaw_deg=yaw_deg,
        blender_bin=blender_bin,
    )


def render_tripo_comparison(
    mesh: str | Path,
    out: str | Path,
    *,
    tripo_glb: str | Path,
    align_tripo: bool = False,
    yaw_deg: float = 0.0,
    blender_bin: str | Path = DEFAULT_BLENDER_BIN,
) -> Path:
    """Compatibility alias for `render_reference_comparison`."""
    return render_reference_comparison(
        mesh,
        out,
        ref_glb=tripo_glb,
        align_ref=align_tripo,
        yaw_deg=yaw_deg,
        blender_bin=blender_bin,
    )


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


def _draw_banner_label(img_rgb, text: str, scale: int = 2) -> None:
    """Draw crisp header bar + 5x7 bitmap font label into a (H, W, 3) uint8 array in-place."""
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


def _run_in_blender(
    *,
    mesh_path: Path,
    out_path: Path,
    compare_tripo: Path | None,
    align_tripo: bool,
    yaw_deg: float,
) -> None:
    import bpy
    import mathutils
    import numpy as np
    import OpenImageIO as oiio

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    if compare_tripo is not None and not compare_tripo.exists():
        raise FileNotFoundError(f"Tripo reference GLB not found: {compare_tripo}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.15

    # Studio background world
    world = bpy.data.worlds.new("StudioWorld")
    scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node is not None:
        bg_node.inputs["Color"].default_value = (0.16, 0.17, 0.19, 1.0)
        bg_node.inputs["Strength"].default_value = 0.65

    # Neutral matte clay material (preserves geometry facets cleanly, readable specular highlights).
    clay_mat = bpy.data.materials.new("NeutralClayMat")
    if clay_mat.node_tree is None:
        clay_mat.use_nodes = True
    bsdf = clay_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.58, 0.58, 0.59, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 0.64
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.35

    def _import_and_normalize_in_memory(glb_path: Path, name_prefix: str, extra_yaw_deg: float) -> list:
        before_objs = set(bpy.data.objects)
        ext = glb_path.suffix.lower()
        if ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=str(glb_path))
        elif ext == ".obj":
            bpy.ops.wm.obj_import(filepath=str(glb_path))
        elif ext == ".ply":
            bpy.ops.wm.ply_import(filepath=str(glb_path))
        else:
            raise ValueError(f"Unsupported mesh extension for Blender import: {glb_path}")

        imported = [obj for obj in bpy.data.objects if obj not in before_objs]
        meshes = [obj for obj in imported if obj.type == "MESH"]
        if not meshes:
            raise RuntimeError(f"No MESH objects imported from {glb_path}")

        for mobj in meshes:
            mobj.data.materials.clear()
            mobj.data.materials.append(clay_mat)

        root_empty = bpy.data.objects.new(f"{name_prefix}_Root", None)
        scene.collection.objects.link(root_empty)
        for obj in imported:
            if obj.parent is None:
                obj.parent = root_empty

        if extra_yaw_deg != 0.0:
            root_empty.rotation_euler = (0.0, 0.0, math.radians(float(extra_yaw_deg)))
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
            raise ValueError(f"Imported mesh has degenerate height {height}: {glb_path}")
        scale = 1.0 / height
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)

        root_empty.scale = (scale, scale, scale)
        root_empty.location = (-cx * scale, -cy * scale, -min_z * scale)
        bpy.context.view_layer.update()
        return meshes

    def _set_visible(mesh_list: list, visible: bool) -> None:
        for obj in mesh_list:
            obj.hide_render = not visible
            obj.hide_viewport = not visible

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

    # Clean 3-point studio lighting + gentle lower bounce for skirt-bottom visibility.
    _add_area_light("KeyLight", (1.85, -2.35, 2.25), (0.0, 0.0, 0.55), 145.0, 1.5, (1.0, 0.98, 0.95))
    _add_area_light("FillLight", (-2.15, -1.75, 1.05), (0.0, 0.0, 0.50), 48.0, 2.3, (0.90, 0.94, 1.0))
    _add_area_light("RimLight", (-0.55, 2.35, 1.90), (0.0, 0.0, 0.65), 110.0, 1.2, (1.0, 1.0, 1.0))
    _add_area_light("LowBounce", (0.0, -0.85, -0.40), (0.0, 0.0, 0.44), 22.0, 1.8, (0.95, 0.96, 1.0))

    cam_data = bpy.data.cameras.new("StudioCam")
    cam_obj = bpy.data.objects.new("StudioCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    def _aim_camera(
        *,
        location: tuple[float, float, float],
        target: tuple[float, float, float],
        ortho: bool,
        ortho_scale: float = 1.14,
        lens: float = 85.0,
    ) -> None:
        cam_obj.location = location
        direction = mathutils.Vector(target) - mathutils.Vector(location)
        cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        if ortho:
            cam_data.type = "ORTHO"
            cam_data.ortho_scale = ortho_scale
        else:
            cam_data.type = "PERSP"
            cam_data.lens = lens
        cam_data.clip_start = 0.01
        cam_data.clip_end = 50.0

    # Camera presets in Blender Z-up where normalized asset faces -Y, up is +Z, left is +X.
    cam_presets = {
        "FRONT (+Z)": dict(location=(0.0, -3.2, 0.50), target=(0.0, 0.0, 0.50), ortho=True, ortho_scale=1.14),
        "3/4 VIEW": dict(location=(1.65, -2.15, 0.72), target=(0.0, 0.0, 0.50), ortho=False, lens=85.0),
        "LEFT (+X)": dict(location=(3.2, 0.0, 0.50), target=(0.0, 0.0, 0.50), ortho=True, ortho_scale=1.14),
        "BACK (-Z)": dict(location=(0.0, 3.2, 0.50), target=(0.0, 0.0, 0.50), ortho=True, ortho_scale=1.14),
        "RIGHT (-X)": dict(location=(-3.2, 0.0, 0.50), target=(0.0, 0.0, 0.50), ortho=True, ortho_scale=1.14),
        "UPPER CLOSE-UP": dict(location=(0.25, -1.04, 0.87), target=(0.0, 0.0, 0.86), ortho=False, lens=90.0),
        "MID DETAIL": dict(location=(0.08, -0.82, 0.72), target=(0.0, 0.0, 0.70), ortho=False, lens=90.0),
        "LOWER BASE": dict(location=(0.10, -0.44, 0.14), target=(0.0, 0.0, 0.44), ortho=False, lens=45.0),
    }

    target_yaw = (-90.0 if align_tripo else 0.0) + float(yaw_deg)
    target_meshes = _import_and_normalize_in_memory(mesh_path, "Target", target_yaw)

    tripo_meshes: list = []
    if compare_tripo is not None:
        tripo_meshes = _import_and_normalize_in_memory(compare_tripo, "TripoRef", -90.0)
        _set_visible(tripo_meshes, False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / f"._render_tmp_{out_path.stem}_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    panel_w, panel_h = (720, 840) if compare_tripo is not None else (768, 900)
    scene.render.resolution_x = panel_w
    scene.render.resolution_y = panel_h
    scene.render.resolution_percentage = 100

    try:
        def _read_png_rgb(fpath: Path) -> np.ndarray:
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

        def _write_png_rgb(fpath: Path, arr: np.ndarray) -> None:
            h, w, c = arr.shape
            out = oiio.ImageOutput.create(str(fpath))
            if out is None:
                raise RuntimeError(f"OpenImageIO failed to create output for {fpath}")
            try:
                spec = oiio.ImageSpec(w, h, c, oiio.UINT8)
                if not out.open(str(fpath), spec):
                    raise RuntimeError(f"Failed to open {fpath}: {out.geterror()}")
                if not out.write_image(np.ascontiguousarray(arr, dtype=np.uint8)):
                    raise RuntimeError(f"Failed to write {fpath}: {out.geterror()}")
            finally:
                out.close()

        def _render_panel(preset_key: str, tag: str, prefix: str) -> np.ndarray:
            _aim_camera(**cam_presets[preset_key])
            fpath = tmp_dir / f"{prefix}.png"
            scene.render.filepath = str(fpath)
            bpy.ops.render.render(write_still=True)
            panel = _read_png_rgb(fpath)
            _draw_banner_label(panel, tag, scale=3)
            return panel

        if compare_tripo is None:
            view_order = [
                "FRONT (+Z)",
                "3/4 VIEW",
                "LEFT (+X)",
                "BACK (-Z)",
                "RIGHT (-X)",
                "UPPER CLOSE-UP",
            ]
            panels = [_render_panel(k, k, f"p{i}") for i, k in enumerate(view_order)]
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
            _write_png_rgb(out_path, sheet)
        else:
            compare_views = [
                "FRONT (+Z)",
                "3/4 VIEW",
                "LEFT (+X)",
                "BACK (-Z)",
                "RIGHT (-X)",
                "UPPER CLOSE-UP",
                "MID DETAIL",
                "LOWER BASE",
            ]
            # 2 view-pairs per row x 4 rows -> 4 panels wide x 4 rows tall (side-by-side OURS | TRIPO).
            pair_gap = 12
            inner_gap = 2
            row_gap = 6
            cols_pairs = 2
            rows_pairs = 4
            pair_w = 2 * panel_w + inner_gap
            sheet_w = cols_pairs * pair_w + (cols_pairs - 1) * pair_gap
            sheet_h = rows_pairs * panel_h + (rows_pairs - 1) * row_gap
            sheet = np.full((sheet_h, sheet_w, 3), (18, 20, 24), dtype=np.uint8)

            for idx, vname in enumerate(compare_views):
                _set_visible(target_meshes, True)
                _set_visible(tripo_meshes, False)
                p_ours = _render_panel(vname, f"OURS | {vname}", f"ours_{idx}")

                _set_visible(target_meshes, False)
                _set_visible(tripo_meshes, True)
                p_tripo = _render_panel(vname, f"TRIPO | {vname}", f"tripo_{idx}")

                r, c = divmod(idx, cols_pairs)
                x0 = c * (pair_w + pair_gap)
                y0 = r * (panel_h + row_gap)
                sheet[y0 : y0 + panel_h, x0 : x0 + panel_w] = p_ours
                sheet[y0 : y0 + panel_h, x0 + panel_w + inner_gap : x0 + 2 * panel_w + inner_gap] = p_tripo

            _write_png_rgb(out_path, sheet)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli_args(argv)
    if args.compare_tripo:
        render_tripo_comparison(
            mesh=args.mesh,
            out=args.out,
            tripo_glb=args.compare_tripo,
            align_tripo=bool(args.align_tripo),
            yaw_deg=float(args.yaw_deg),
            blender_bin=args.blender_bin,
        )
    else:
        render_preview_sheet(
            mesh=args.mesh,
            out=args.out,
            align_tripo=bool(args.align_tripo),
            yaw_deg=float(args.yaw_deg),
            blender_bin=args.blender_bin,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
