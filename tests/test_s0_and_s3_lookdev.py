"""Offline unit tests for S0 preprocessing, camera calibration, and S3 Two-Band lookdev & PBR."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from quadvox.contracts import ROISovereigntyBox
from quadvox.s0_views.camera import (
    calibrate_s0_ortho_camera,
    compute_silhouette_alignment_warp,
    matte_s0_view,
    render_mesh_ortho_zbuf,
)
from quadvox.s0_views.preprocess import prepare_views_dir
from quadvox.s3_lookdev.backproject import (
    bake_two_band_albedo,
    load_uv_mesh_and_labels,
    rasterize_uv_attributes,
)
from quadvox.s3_lookdev.delight import (
    apply_region_protected_delight,
    constrained_exposure_match,
)
from quadvox.s3_lookdev.materials import derive_generic_pbr_maps, derive_prop_pbr_maps

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_uv_cube(path: Path) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=(0.5, 0.8, 0.5))
    box.vertices[:, 1] += 0.5
    u = np.clip(box.vertices[:, 0] + 0.5, 0.05, 0.95)
    v = np.clip(box.vertices[:, 1], 0.05, 0.95)
    box.visual = trimesh.visual.TextureVisuals(uv=np.stack([u, v], axis=1))
    box.export(str(path))
    return box


def test_s0_preprocess_and_camera_calibration(tmp_path: Path | None = None) -> None:
    work = REPO_ROOT / "runs" / "_test_tmp_s0"
    work.mkdir(parents=True, exist_ok=True)
    raw_dir = work / "raw"
    raw_dir.mkdir(exist_ok=True)
    for idx in range(4):
        arr = np.full((128, 128, 4), (220, 220, 220, 0), dtype=np.uint8)
        arr[20:108, 36:92] = (180, 90, 60, 255)
        Image.fromarray(arr).save(raw_dir / f"view0{idx}_azim{idx*90:03d}.png")

    prepped = work / "prepped"
    manifest = prepare_views_dir(raw_dir, prepped, mode="mv", size=128)
    assert len(manifest["transforms"]["frames"]) == 4
    assert (prepped / "transforms.json").is_file()

    matted = matte_s0_view(raw_dir / "view00_azim000.png", name="front")
    assert matted.fg_mask.sum() > 2000

    glb = work / "cube.glb"
    _make_uv_cube(glb)
    mdata = load_uv_mesh_and_labels(glb, is_prop=True)
    cam, cal_stats = calibrate_s0_ortho_camera(
        mdata["verts"], mdata["faces"], matted.fg_mask, name="front", azim_deg=0.0
    )
    assert cal_stats["calibrated_iou"] > 0.70

    zbuf = render_mesh_ortho_zbuf(mdata["verts"], mdata["faces"], cam)
    warped, warp_stats = compute_silhouette_alignment_warp(matted, zbuf["hit_mask"])
    assert warped.fg_mask.shape == matted.fg_mask.shape
    assert "iou_before" in warp_stats


def test_s3_delight_two_band_and_pbr_derivation() -> None:
    work = REPO_ROOT / "runs" / "_test_tmp_s3"
    work.mkdir(parents=True, exist_ok=True)
    glb = work / "cube_uv.glb"
    _make_uv_cube(glb)
    mdata = load_uv_mesh_and_labels(glb, is_prop=True)
    uv_attrs = rasterize_uv_attributes(
        mdata["verts"], mdata["faces"], mdata["uvs"], mdata["vnorms"], mdata["face_labels"], tex_size=64
    )
    assert uv_attrs["valid_mask"].sum() > 100

    # Synthetic 4-view images: front=red, left=green, back=blue, right=yellow
    colors = {
        "front": (0.9, 0.15, 0.15),
        "left": (0.15, 0.85, 0.15),
        "back": (0.15, 0.20, 0.90),
        "right": (0.90, 0.85, 0.15),
    }
    azims = {"front": 0.0, "left": 90.0, "back": 180.0, "right": 270.0}
    views, cams = [], []
    for name, col in colors.items():
        img = np.full((64, 64, 3), 0.85, dtype=np.float32)
        img[12:56, 18:46] = np.array(col, dtype=np.float32)
        mv = matte_s0_view((img * 255).astype(np.uint8), name=name)
        views.append(mv)
        cam, _ = calibrate_s0_ortho_camera(mdata["verts"], mdata["faces"], mv.fg_mask, name=name, azim_deg=azims[name])
        cams.append(cam)

    # Layer-1 Two-Band bake
    baked_l1 = bake_two_band_albedo(mesh_data=mdata, uv_attrs=uv_attrs, views=views, cameras=cams)
    assert baked_l1["albedo"].shape == (64, 64, 3)
    assert baked_l1["observed_fraction"] > 0.3

    # Layer-2 ROI sovereignty lock on front
    roi = ROISovereigntyBox(name="front_logo", x_range=(-0.3, 0.3), y_range=(0.2, 0.8), z_range=(0.05, 0.4))
    baked_l2 = bake_two_band_albedo(
        mesh_data=mdata, uv_attrs=uv_attrs, views=views, cameras=cams, roi_boxes=[roi]
    )
    assert baked_l2["albedo"].shape == (64, 64, 3)

    # Region-protected delight & exposure match
    orig = np.full((32, 32, 3), 0.8, dtype=np.float32)
    delighted = np.full((32, 32, 3), 0.6, dtype=np.float32)
    pmask = np.zeros((32, 32), dtype=np.float32)
    pmask[10:20, 10:20] = 1.0
    blended = apply_region_protected_delight(orig, delighted, pmask, np.ones((32, 32), dtype=bool))
    assert float(blended[15, 15, 0]) > 0.75

    matched, _ = constrained_exposure_match([orig, delighted], [np.ones((32, 32), dtype=bool)] * 2)
    assert len(matched) == 2

    # Generic & Prop metal PBR maps
    nrm = np.full((64, 64, 3), (0.5, 0.5, 1.0), dtype=np.float32)
    ao = np.full((64, 64), 0.9, dtype=np.float32)
    curv = np.full((64, 64), 0.5, dtype=np.float32)
    curv[10:20, 10:20] = 0.85  # edge wear
    ao[30:40, 30:40] = 0.15    # crevice grime

    pbr_prop = derive_prop_pbr_maps(
        albedo_base_4k=baked_l1["albedo"], normal_4k=nrm, ao_4k=ao, curvature_4k=curv, valid_mask=uv_attrs["valid_mask"]
    )
    assert float(pbr_prop["wear_mask"][15, 15]) > 0.5
    assert float(pbr_prop["grime_mask"][35, 35]) > 0.5

    pbr_gen = derive_generic_pbr_maps(
        albedo_base_4k=baked_l1["albedo"], normal_4k=nrm, ao_4k=ao, curvature_4k=curv, valid_mask=uv_attrs["valid_mask"]
    )
    assert pbr_gen["orm"].shape == (64, 64, 3)
