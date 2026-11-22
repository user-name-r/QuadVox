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
from quadvox.s2_retopo.slim_uv_bake import build_bake_4k_blender_cmd, unwrap_part_aware

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
