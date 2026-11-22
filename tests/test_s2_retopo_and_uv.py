"""Offline unit tests for S2 retopology, LATO.2 prop repair, ReQuad Layer-2 symmetry, and SLIM UV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from quadvox.s2_retopo.lato2_repair import repair_prop_lato2_mesh
from quadvox.s2_retopo.metrics import compute_mesh_metrics
from quadvox.s2_retopo.preclean import preclean_mesh
from quadvox.s2_retopo.qem import decimate_mesh_file
from quadvox.s2_retopo.quad_retopo import build_requad_blender_cmd
from quadvox.s2_retopo.slim_uv_bake import (
    _blender_dilate_2d_margin,
    bake_4k_maps_blender,
    build_bake_4k_blender_cmd,
    smooth_clamp_uv_island_border,
    unwrap_part_aware,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_s2_preclean_qem_lato2_and_unwrap() -> None:
    work = REPO_ROOT / "runs" / "_test_tmp_s2"
    work.mkdir(parents=True, exist_ok=True)
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.45)
    sphere.vertices[:, 1] += 0.5
    hp_path = work / "high.glb"
    sphere.export(str(hp_path))

    cleaned_path = work / "cleaned.glb"
    diff = preclean_mesh(hp_path, cleaned_path)
    assert diff.after.face_count > 0

    m = compute_mesh_metrics(cleaned_path)
    assert m.is_watertight

    dec_fq_path = work / "dec_fq.glb"
    decimate_mesh_file(cleaned_path, dec_fq_path, engine="pyfqmr", target_faces=300)
    assert compute_mesh_metrics(dec_fq_path).face_count <= 320

    dec_pml_path = work / "dec_pml.glb"
    decimate_mesh_file(
        cleaned_path, dec_pml_path, engine="pymeshlab", target_faces=300, preserve_topology=True, planar_quadric=False
    )
    assert compute_mesh_metrics(dec_pml_path).face_count <= 350

    lato_rep_path = work / "lato_rep.glb"
    rep = repair_prop_lato2_mesh(cleaned_path, lato_rep_path, ref_mesh=hp_path)
    assert rep.is_watertight is True

    uv_path = work / "uv_out.glb"
    unwrap_res = unwrap_part_aware(
        dec_pml_path,
        uv_path,
        uv_method="xatlas",
        asset_type="auto",
        tex_size=512,
        padding_px=4,
    )
    assert unwrap_res.asset_type == "prop"
    assert unwrap_res.uv_overlap_count == 0


def test_s2_layer2_symmetry_and_bake_cli_builders() -> None:
    cmd_no_sym = build_requad_blender_cmd(
        blender_bin="blender",
        proxy_mesh_path="proxy.glb",
        highpoly_mesh_path="high.glb",
        guide_npz_path="guide.npz",
        out_quad_obj="out.obj",
        out_tri_glb="out.glb",
        meta_json_path="meta.json",
        sym_x=False,
    )
    assert "--sym-x" not in cmd_no_sym

    cmd_sym = build_requad_blender_cmd(
        blender_bin="blender",
        proxy_mesh_path="proxy.glb",
        highpoly_mesh_path="high.glb",
        guide_npz_path="guide.npz",
        out_quad_obj="out.obj",
        out_tri_glb="out.glb",
        meta_json_path="meta.json",
        sym_x=True,
    )
    assert "--sym-x" in cmd_sym

    bake_cmd = build_bake_4k_blender_cmd(
        blender_bin="blender",
        high_mesh_path="high.glb",
        low_uv_mesh_path="low.glb",
        out_normal_png="n.png",
        out_ao_png="a.png",
        out_curvature_png="c.png",
        out_baked_glb="b.glb",
        meta_json_path="m.json",
    )
    assert "quadvox/s2_retopo/slim_uv_bake.py" in " ".join(bake_cmd)


def test_uv_border_smoothing_clamp_before_edt_and_defaults() -> None:
    import inspect

    sig_unwrap = inspect.signature(unwrap_part_aware)
    assert sig_unwrap.parameters["zero_overlap_refine"].default is True
    assert sig_unwrap.parameters["pack_method"].default == "blender_concave"

    sig_bake = inspect.signature(bake_4k_maps_blender)
    assert sig_bake.parameters["head_ring_dilation"].default == 16
    sig_cmd = inspect.signature(build_bake_4k_blender_cmd)
    assert sig_cmd.parameters["head_ring_dilation"].default == 16

    # Synthetic UV island [12:52, 12:52] inside 64x64 atlas with outer 1-2 px rim spikes.
    mask = np.zeros((64, 64), dtype=bool)
    mask[12:52, 12:52] = True

    ao = np.ones((64, 64), dtype=np.float32)
    ao[mask] = 0.94
    ao[30:34, 30:34] = 0.42  # legitimate interior detail untouched
    # Inject 0.0 barcode spikes on outermost 1-2 px island perimeter
    ao[12, 12:52] = 0.0
    ao[13, 20:30] = 0.05
    ao[12:52, 51] = 0.0

    smoothed_ao = smooth_clamp_uv_island_border(ao, mask, border_px=2)
    # Interior (>= 3 px inside border) must remain bit-exact
    assert np.array_equal(smoothed_ao[14:50, 14:50], ao[14:50, 14:50])
    # Outer 1-2 px rim spikes must be eliminated before EDT dilation
    border_ring = mask.copy()
    border_ring[14:50, 14:50] = False
    assert float(smoothed_ao[border_ring].min()) >= 0.90

    dilated_ao = _blender_dilate_2d_margin(ao, mask, padding_px=16, bg_value=(1.0,))
    assert float(dilated_ao[6:12, 12:52].min()) >= 0.90  # no 0.0 gutter barcode streaks
    assert np.array_equal(dilated_ao[14:50, 14:50], ao[14:50, 14:50])

    # Tangent-space normal map outer-rim spike must be clamped + renormalized with z >= 0.60
    nrm = np.full((64, 64, 3), (0.5, 0.5, 1.0), dtype=np.float32)
    nrm[12, 18:45] = (0.05, 0.5, 0.05)  # grazing/inverted normal spike on outer 1px rim
    dilated_nrm = _blender_dilate_2d_margin(
        nrm, mask, padding_px=12, bg_value=(0.5, 0.5, 1.0), renormalize_normals=True
    )
    assert float(dilated_nrm[mask, 2].min()) >= 0.60
    assert float(dilated_nrm[8:12, 18:45, 2].min()) >= 0.60
    vec = dilated_nrm[12, 24] * 2.0 - 1.0
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-4

    # Thin 2-px strip fallback (interior erode_2 empty) must remain finite without NaNs
    thin_mask = np.zeros((16, 16), dtype=bool)
    thin_mask[7:9, 3:13] = True
    thin_arr = np.full((16, 16), 0.8, dtype=np.float32)
    thin_arr[7, 7] = 0.0
    thin_out = smooth_clamp_uv_island_border(thin_arr, thin_mask, border_px=2)
    assert np.all(np.isfinite(thin_out))
    assert float(thin_out[7, 7]) > 0.4
