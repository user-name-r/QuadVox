"""Unified CLI entrypoint (`quadvox`) for the Universal Multi-View 3D Pipeline (Layer 1 + Layer 2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from quadvox.contracts import CoarseSpliceSpec, ROISovereigntyBox, load_profile_config
from quadvox.tools.external_tools import resolve_blender_bin


def _cmd_profile_show(args: argparse.Namespace) -> int:
    prof = load_profile_config(args.profile)
    print(json.dumps(prof.to_dict(), indent=2, ensure_ascii=False))
    return 0


def _cmd_s0_expand(args: argparse.Namespace) -> int:
    from quadvox.s0_views.turnaround import generate_turnaround_views

    views = [v.strip() for v in args.views.split(",") if v.strip()] if args.views else None
    report = generate_turnaround_views(
        front_image=args.front,
        sheet_image=args.sheet,
        out_dir=args.out_dir,
        subject_description=args.subject,
        views=views,
        config_path=args.providers_config,
        align_to_front=not bool(args.no_align),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_s0_preprocess(args: argparse.Namespace) -> int:
    from quadvox.s0_views.preprocess import prepare_views_dir

    prof = load_profile_config(args.profile)
    s0_cfg = prof.s0
    manifest = prepare_views_dir(
        input_path=args.input,
        out_dir=args.out_dir,
        mode=args.mode,
        size=int(args.view_size or s0_cfg.get("view_size", 1024)),
        fov_deg=float(args.fov_deg or s0_cfg.get("fov_deg", 20.0)),
        pad_factor=float(args.pad_factor or s0_cfg.get("pad_factor", 1.1)),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def _cmd_s1_repair(args: argparse.Namespace) -> int:
    from quadvox.s1_geometry.repair import (
        cellocut_remesh_glb,
        compute_mesh_metrics,
        inplace_clean_glb,
        normalize_glb,
    )

    prof = load_profile_config(args.profile)
    rep = prof.s1.get("repair", {})
    method = str(args.method or rep.get("default_method", "cellocut")).lower()
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    work_p = out_p.with_name(f"._{out_p.stem}_pre_norm.glb")

    if method == "cellocut":
        cellocut_remesh_glb(
            args.input,
            work_p,
            resolution=int(args.cellocut_res or rep.get("cellocut_resolution", 1024)),
            fill_hole_perimeter=float(rep.get("cellocut_fill", 20.0)),
            band=float(args.cellocut_band if args.cellocut_band is not None else rep.get("cellocut_band", 1.0)),
            project_back=float(args.cellocut_pb if args.cellocut_pb is not None else rep.get("cellocut_project_back", 0.5)),
            qef=bool(args.cellocut_qef if args.cellocut_qef is not None else rep.get("cellocut_qef", False)),
            smooth_iters=int(rep.get("cellocut_smooth_iters", 0)),
        )
    elif method == "inplace":
        inplace_clean_glb(args.input, work_p)
    else:
        raise ValueError(f"Unsupported repair method {method!r}")

    normalize_glb(work_p, out_p, target_height=float(args.target_height))
    if work_p.exists():
        work_p.unlink()

    metrics = compute_mesh_metrics(out_p)
    print(metrics.to_json(indent=2))
    return 0


def _cmd_s1_splice(args: argparse.Namespace) -> int:
    from quadvox.s1_geometry.coarse_splice import splice_coarse_glbs

    prof = load_profile_config(args.profile)
    spec = prof.coarse_splice()
    if args.transition:
        lo, hi = [float(x) for x in args.transition.split(",")]
        spec = CoarseSpliceSpec(
            enabled=True,
            voxel_res=int(args.voxel_res or spec.voxel_res),
            half_width=spec.half_width,
            split_axis=str(args.axis or spec.split_axis),
            transition_range=(lo, hi),
            lateral_guard_x=spec.lateral_guard_x,
            lateral_guard_y_max=spec.lateral_guard_y_max,
        )
    stats = splice_coarse_glbs(
        args.primary,
        args.secondary,
        args.output,
        spec,
        blender_bin=args.blender_bin,
    )
    print(json.dumps(stats, indent=2))
    return 0


def _cmd_s1_render(args: argparse.Namespace) -> int:
    from quadvox.s1_geometry.render_preview import render_preview_sheet, render_reference_comparison

    if args.compare_ref:
        out_p = render_reference_comparison(
            args.mesh,
            args.out,
            ref_glb=args.compare_ref,
            align_ref=bool(args.align_ref),
            yaw_deg=float(args.yaw_deg),
            blender_bin=args.blender_bin,
        )
    else:
        out_p = render_preview_sheet(
            args.mesh,
            args.out,
            align_tripo=bool(args.align_ref),
            yaw_deg=float(args.yaw_deg),
            blender_bin=args.blender_bin,
        )
    print(json.dumps({"rendered_sheet": str(out_p)}, indent=2))
    return 0


def _cmd_s2_unwrap_bake(args: argparse.Namespace) -> int:
    from quadvox.s2_retopo.slim_uv_bake import bake_4k_maps_blender, unwrap_part_aware

    prof = load_profile_config(args.profile)
    s2 = prof.s2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    uv_glb = out_dir / "mesh_uv.glb"

    unwrap_res = unwrap_part_aware(
        args.low_mesh,
        uv_glb,
        labels_npz_path=args.labels_npz,
        tex_size=int(args.tex_size or s2.get("tex_size", 4096)),
        padding_px=int(s2.get("padding_px", 16)),
        uv_method=str(args.uv_method or s2.get("uv_method", "blender_slim")),
        asset_type=str(args.asset_type or prof.asset_type),
        blender_bin=args.blender_bin,
    )
    bake_res = bake_4k_maps_blender(
        args.high_mesh,
        uv_glb,
        out_dir / "normal_4k.png",
        out_dir / "ao_4k.png",
        out_dir / "curvature_4k.png",
        out_dir / "baked_preview.glb",
        tex_size=int(args.tex_size or s2.get("tex_size", 4096)),
        padding_px=int(s2.get("padding_px", 16)),
        use_gpu=bool(args.use_gpu),
        blender_bin=args.blender_bin,
    )
    print(json.dumps({"unwrap": unwrap_res.to_dict(), "bake": bake_res.to_dict()}, indent=2))
    return 0


def _cmd_s2_lato2_repair(args: argparse.Namespace) -> int:
    from quadvox.s2_retopo.lato2_repair import repair_prop_lato2_mesh

    rep = repair_prop_lato2_mesh(
        args.input,
        args.output,
        ref_mesh=args.ref_mesh,
    )
    print(json.dumps(rep.to_dict(), indent=2))
    return 0


def _cmd_s3_lookdev(args: argparse.Namespace) -> int:
    from PIL import Image
    import numpy as np

    from quadvox.s0_views.camera import calibrate_s0_ortho_camera, matte_s0_view
    from quadvox.s3_lookdev.backproject import (
        bake_two_band_albedo,
        load_uv_mesh_and_labels,
        rasterize_uv_attributes,
    )
    from quadvox.s3_lookdev.delight import constrained_exposure_match
    from quadvox.s3_lookdev.materials import derive_generic_pbr_maps, derive_prop_pbr_maps

    prof = load_profile_config(args.profile)
    s3 = prof.s3
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_data = load_uv_mesh_and_labels(
        args.mesh,
        labels_npz_path=args.labels_npz,
        is_prop=(prof.asset_type == "prop"),
    )
    tex_size = int(args.tex_size or prof.s2.get("tex_size", 4096))
    uv_attrs = rasterize_uv_attributes(
        mesh_data["verts"],
        mesh_data["faces"],
        mesh_data["uvs"],
        mesh_data["vnorms"],
        mesh_data["face_labels"],
        tex_size=tex_size,
    )

    view_specs = (
        ("front", 0.0, args.front),
        ("left", 90.0, args.left),
        ("back", 180.0, args.back),
        ("right", 270.0, args.right),
    )
    views = [matte_s0_view(p, name=n) for n, _, p in view_specs if p]
    azims = [az for _, az, p in view_specs if p]
    if not views:
        raise ValueError("At least --front view is required for S3 lookdev")

    matched_rgbs, _ = constrained_exposure_match(
        [v.rgb_dilated for v in views],
        [v.fg_mask for v in views],
    )
    for v, m_rgb in zip(views, matched_rgbs):
        v.rgb_dilated = m_rgb

    cameras = [
        calibrate_s0_ortho_camera(
            mesh_data["verts"],
            mesh_data["faces"],
            v.fg_mask,
            name=v.name,
            azim_deg=az,
            is_primary=True,
        )[0]
        for v, az in zip(views, azims)
    ]

    roi_boxes: list[ROISovereigntyBox] = prof.roi_boxes() if s3.get("roi_sovereignty", False) else []
    baked = bake_two_band_albedo(
        mesh_data=mesh_data,
        uv_attrs=uv_attrs,
        views=views,
        cameras=cameras,
        detail_power=float(s3.get("detail_power", 32.0)),
        low_power=float(s3.get("low_power", 4.0)),
        blur_sigma_px=float(s3.get("blur_sigma_px", 6.0)),
        occl_tol_rel=float(s3.get("occl_tol_rel", 0.015)),
        roi_boxes=roi_boxes,
    )
    alb = baked["albedo"]

    def _save_img(path: Path, arr: np.ndarray) -> None:
        Image.fromarray(np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)).save(path)

    alb_path = out_dir / "albedo_4k.png"
    _save_img(alb_path, alb)

    report: dict[str, Any] = {
        "profile": prof.name,
        "albedo_4k": str(alb_path),
        "observed_fraction": baked["observed_fraction"],
        "fill_stats": baked["fill_stats"],
        "roi_boxes_active": [b.name for b in roi_boxes],
    }

    if args.normal_4k and args.ao_4k and args.curvature_4k:
        nrm = np.asarray(Image.open(args.normal_4k).convert("RGB"), dtype=np.float32) / 255.0
        ao = np.asarray(Image.open(args.ao_4k).convert("L"), dtype=np.float32) / 255.0
        curv = np.asarray(Image.open(args.curvature_4k).convert("L"), dtype=np.float32) / 255.0
        recipe = str(s3.get("pbr_recipe", "generic_pbr"))
        if recipe == "hard_surface_metal":
            pbr = derive_prop_pbr_maps(
                albedo_base_4k=alb,
                normal_4k=nrm,
                ao_4k=ao,
                curvature_4k=curv,
                valid_mask=uv_attrs["valid_mask"],
            )
            _save_img(out_dir / "albedo_prop_4k.png", pbr["albedo_prop"])
            _save_img(out_dir / "orm_prop_4k.png", pbr["orm_prop"])
            _save_img(out_dir / "normal_prop_4k.png", pbr["normal_prop"])
        else:
            pbr = derive_generic_pbr_maps(
                albedo_base_4k=alb,
                normal_4k=nrm,
                ao_4k=ao,
                curvature_4k=curv,
                valid_mask=uv_attrs["valid_mask"],
            )
            _save_img(out_dir / "orm_4k.png", pbr["orm"])
            _save_img(out_dir / "normal_out_4k.png", pbr["normal"])
        report["pbr_recipe"] = recipe

    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quadvox",
        description="Universal Multi-View 3D Pipeline (4-view Direct3D-V3 + Pixal3D + SLIM UV + Two-Band PBR).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prof = sub.add_parser("profile-show", help="Inspect merged Layer-1/Layer-2 profile configuration.")
    p_prof.add_argument("--profile", default="object")

    p_s0e = sub.add_parser("s0-expand", help="Generate aligned 90/180/270 orthographic views.")
    p_s0e.add_argument("--front", required=True)
    p_s0e.add_argument("--sheet", default=None)
    p_s0e.add_argument("--out-dir", required=True)
    p_s0e.add_argument("--subject", default="the 3D asset")
    p_s0e.add_argument("--views", default=None)
    p_s0e.add_argument("--providers-config", default=None)
    p_s0e.add_argument("--no-align", action="store_true")

    p_s0p = sub.add_parser("s0-preprocess", help="Normalize RGBA views and generate orbit transforms.json.")
    p_s0p.add_argument("--profile", default="object")
    p_s0p.add_argument("--input", required=True)
    p_s0p.add_argument("--out-dir", required=True)
    p_s0p.add_argument("--mode", choices=("mv", "sv"), default="mv")
    p_s0p.add_argument("--view-size", type=int, default=None)
    p_s0p.add_argument("--fov-deg", type=float, default=None)
    p_s0p.add_argument("--pad-factor", type=float, default=None)
    p_s0p.add_argument("--matte-bg", action="store_true")

    p_s1r = sub.add_parser("s1-repair", help="Repair and normalize raw S1 mesh (CelloCut / inplace).")
    p_s1r.add_argument("--profile", default="object")
    p_s1r.add_argument("--input", required=True)
    p_s1r.add_argument("--output", required=True)
    p_s1r.add_argument("--method", choices=("cellocut", "inplace"), default=None)
    p_s1r.add_argument("--cellocut-res", type=int, default=None)
    p_s1r.add_argument("--cellocut-band", type=float, default=None)
    p_s1r.add_argument("--cellocut-pb", type=float, default=None)
    p_s1r.add_argument("--cellocut-qef", action="store_true", default=None)
    p_s1r.add_argument("--target-height", type=float, default=1.0)

    p_s1sp = sub.add_parser("s1-splice", help="Layer-2 OpenVDB dual-coarse mesh splice.")
    p_s1sp.add_argument("--profile", default="symmetric_humanoid")
    p_s1sp.add_argument("--primary", required=True)
    p_s1sp.add_argument("--secondary", required=True)
    p_s1sp.add_argument("--output", required=True)
    p_s1sp.add_argument("--axis", choices=("x", "y", "z"), default=None)
    p_s1sp.add_argument("--transition", default=None, help="Comma-separated transition range, e.g. '0.69,0.75'")
    p_s1sp.add_argument("--voxel-res", type=int, default=None)
    p_s1sp.add_argument("--blender-bin", default=str(resolve_blender_bin()))

    p_s1ren = sub.add_parser("s1-render", help="Render 6-view studio clay preview in Blender 5.3.")
    p_s1ren.add_argument("--mesh", required=True)
    p_s1ren.add_argument("--out", required=True)
    p_s1ren.add_argument("--compare-ref", default=None)
    p_s1ren.add_argument("--align-ref", action="store_true")
    p_s1ren.add_argument("--yaw-deg", type=float, default=0.0)
    p_s1ren.add_argument("--blender-bin", default=str(resolve_blender_bin()))

    p_s2u = sub.add_parser("s2-unwrap-bake", help="Run Blender SLIM low-island UV + 4K Cycles Cage bake.")
    p_s2u.add_argument("--profile", default="object")
    p_s2u.add_argument("--high-mesh", required=True)
    p_s2u.add_argument("--low-mesh", required=True)
    p_s2u.add_argument("--out-dir", required=True)
    p_s2u.add_argument("--labels-npz", default=None)
    p_s2u.add_argument("--asset-type", default=None)
    p_s2u.add_argument("--uv-method", default=None)
    p_s2u.add_argument("--tex-size", type=int, default=None)
    p_s2u.add_argument("--use-gpu", action="store_true")
    p_s2u.add_argument("--blender-bin", default=str(resolve_blender_bin()))

    p_s2l = sub.add_parser("s2-lato2-repair", help="Repair LATO.2 prop mesh cracks/coplanars/winding.")
    p_s2l.add_argument("--input", required=True)
    p_s2l.add_argument("--output", required=True)
    p_s2l.add_argument("--ref-mesh", default=None)

    p_s3 = sub.add_parser("s3-lookdev", help="Run Two-Band backprojection, 3D harmonic fill & PBR map derivation.")
    p_s3.add_argument("--profile", default="object")
    p_s3.add_argument("--mesh", required=True)
    p_s3.add_argument("--front", required=True)
    p_s3.add_argument("--left", default=None)
    p_s3.add_argument("--back", default=None)
    p_s3.add_argument("--right", default=None)
    p_s3.add_argument("--normal-4k", default=None)
    p_s3.add_argument("--ao-4k", default=None)
    p_s3.add_argument("--curvature-4k", default=None)
    p_s3.add_argument("--labels-npz", default=None)
    p_s3.add_argument("--tex-size", type=int, default=None)
    p_s3.add_argument("--out-dir", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dispatch = {
        "profile-show": _cmd_profile_show,
        "s0-expand": _cmd_s0_expand,
        "s0-preprocess": _cmd_s0_preprocess,
        "s1-repair": _cmd_s1_repair,
        "s1-splice": _cmd_s1_splice,
        "s1-render": _cmd_s1_render,
        "s2-unwrap-bake": _cmd_s2_unwrap_bake,
        "s2-lato2-repair": _cmd_s2_lato2_repair,
        "s3-lookdev": _cmd_s3_lookdev,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
