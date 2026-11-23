"""Unit tests for 4-view Direct3D-V3 lossless 8^3 voxel rotation fusion,
Layer-2 configurable ROI sovereignty, OpenVDB splice weights, and SDF meshing.
"""

from __future__ import annotations

import numpy as np
import trimesh

from quadvox.contracts import CoarseSpliceSpec, ROISovereigntyBox
from quadvox.s1_geometry.coarse_splice import compute_splice_blend_weights
from quadvox.s1_geometry.d3dv3_fusion import (
    VIEW_ORDER,
    _sstep,
    _sstep_np,
    build_view_rotation,
    compare_crevice_keep_preflight,
    compute_4view_voxel_weights,
    compute_step_beta_and_cfg,
    compute_surface_visibility_and_weights,
    rotate_latents_forward,
    rotate_latents_inverse,
    sharpen_view_weights,
)
from quadvox.s1_geometry.sdf_mesher import build_sdf_mesh


def test_4view_discrete_rotation_lossless_roundtrip_numpy() -> None:
    stage, ps = 512, 8
    grid = stage // ps
    rng = np.random.default_rng(7)
    patches = sorted({(0, 10, 20, 30), (0, 35, 22, 41), (0, 63 - 35, 22, 63 - 41)})
    coords = np.array(patches, dtype=np.int32)
    lat = rng.normal(size=(len(patches), ps**3)).astype(np.float32)

    gx, gy, gz = 310, 185, 268
    px_m, dx_m = gx // ps, gx % ps
    py_m, dy_m = gy // ps, gy % ps
    pz_m, dz_m = gz // ps, gz % ps
    coords = np.vstack([coords, np.array([[0, px_m, py_m, pz_m]], dtype=np.int32)])
    lat = np.vstack([lat, rng.normal(size=(1, ps**3)).astype(np.float32)])
    marker = np.zeros_like(lat)
    marker[-1, dx_m * ps * ps + dy_m * ps + dz_m] = 1.0

    w_exact = np.array([gx, gy, gz], dtype=np.float64) / float(stage - 1) - 0.5
    exp_map = {
        "front": (w_exact[0], w_exact[1], w_exact[2]),
        "back": (-w_exact[0], w_exact[1], -w_exact[2]),
        "left": (-w_exact[2], w_exact[1], w_exact[0]),
        "right": (w_exact[2], w_exact[1], -w_exact[0]),
    }

    for view in VIEW_ORDER:
        cv, pv, ipv = build_view_rotation(coords, grid, view)
        rec = rotate_latents_inverse(rotate_latents_forward(lat, pv, ps, view), ipv, ps, view)
        assert float(np.abs(rec - lat).max()) == 0.0

        m_rot = rotate_latents_forward(marker, pv, ps, view)
        idx = np.argwhere(m_rot == 1.0)
        assert len(idx) == 1
        rp, rv = int(idx[0, 0]), int(idx[0, 1])
        rx, ry, rz = int(cv[rp, 1]), int(cv[rp, 2]), int(cv[rp, 3])
        rdx, rdy, rdz = rv // (ps * ps), (rv // ps) % ps, rv % ps
        rw = ((rx * ps + rdx) / (stage - 1) - 0.5, (ry * ps + rdy) / (stage - 1) - 0.5, (rz * ps + rdz) / (stage - 1) - 0.5)
        for a, b in zip(rw, exp_map[view]):
            assert abs(a - b) < 1e-6


def test_surface_and_voxel_weights_layer1_vs_layer2_roi() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.45)
    verts = np.asarray(sphere.vertices, dtype=np.float32)
    verts[:, 1] += 0.50  # [0.05, 0.95]
    faces = np.asarray(sphere.faces, dtype=np.int32)

    # Layer 1: zero ROI boxes
    surf_l1 = compute_surface_visibility_and_weights(verts, faces, zres=128, roi_boxes=())
    w_surf_l1 = surf_l1["w_surf"]
    assert np.allclose(w_surf_l1.sum(axis=1), 1.0, atol=1e-5)
    assert surf_l1["meta"]["roi_boxes"] == []

    # Build sparse patch coords overlapping front & side vertices
    stage, ps = 512, 8
    coords = np.array(
        [
            [0, 32, 32, 48],  # front (+Z)
            [0, 48, 32, 32],  # left (+X)
            [0, 32, 32, 16],  # back (-Z)
        ],
        dtype=np.int32,
    )
    w_vox_l1, stats_l1, _ = compute_4view_voxel_weights(verts, faces, coords, stage, ps, zres=128)
    assert w_vox_l1.shape == (4, len(coords), ps**3)
    assert stats_l1["sum_w_max_abs_err"] < 1e-5

    # Layer 2: front ROI sovereignty box locking front patch [0, 32, 32, 48]
    roi = ROISovereigntyBox(
        name="hero_cockpit",
        x_range=(-0.25, 0.25),
        y_range=(0.30, 0.70),
        z_range=(0.05, 0.60),
        margin=0.04,
        target_view="front",
        min_target_weight=0.25,
        min_surface_weight=0.90,
        suppress_ortho_views=0.98,
    )
    w_vox_l2, stats_l2, _ = compute_4view_voxel_weights(
        verts, faces, coords, stage, ps, zres=128, roi_boxes=[roi]
    )
    assert "hero_cockpit" in stats_l2["roi_zones"]
    assert stats_l2["roi_zones"]["hero_cockpit"]["voxel_count"] > 0
    # Inside front patch (idx 0), front weight dominates and side weights are suppressed.
    assert float(w_vox_l2[0, 0].mean()) >= 0.85
    assert float(w_vox_l2[2, 0].mean()) <= 0.05
    assert float(w_vox_l2[3, 0].mean()) <= 0.05


def test_step_beta_cfg_schedule_and_sharpening() -> None:
    from quadvox.s1_geometry.d3dv3_fusion import _sstep_np

    # Verify _sstep_np works for both ascending (e0 < e1) and descending (e0 > e1) overhang curves.
    vals = np.array([0.5, -0.10, -0.22, -0.40, -0.58, -0.90], dtype=np.float32)
    down = _sstep_np(-0.22, -0.58, vals)
    assert float(down[0]) == 0.0 and float(down[1]) == 0.0 and float(down[2]) == 0.0
    assert 0.45 <= float(down[3]) <= 0.55
    assert abs(float(down[4]) - 1.0) < 1e-6 and float(down[5]) == 1.0

    b0, c0 = compute_step_beta_and_cfg(0.2, stage=1536, guidance_scale=4.0, cfg_interval=(0.1, 0.9))
    b_mid, c_mid = compute_step_beta_and_cfg(0.70, stage=1536, guidance_scale=4.0, cfg_interval=(0.1, 0.9))
    b1, c1 = compute_step_beta_and_cfg(0.899, stage=1536, guidance_scale=4.0, cfg_interval=(0.1, 0.9))
    assert b0 == 1.0 and c0 == 4.0
    assert b_mid > 2.0 and abs(c_mid - 3.1) < 0.05  # Linear midpoint at t = 0.70 -> (4.0 + 2.2) / 2 = 3.1
    assert b1 > 3.5 and abs(c1 - 2.2) < 0.05

    w = np.array([[[0.7, 0.3], [0.2, 0.8]], [[0.0, 0.0], [0.5, 0.5]], [[0.3, 0.7], [0.3, 0.2]], [[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    w /= w.sum(axis=0, keepdims=True)
    ws = sharpen_view_weights(w, beta=4.0)
    assert np.allclose(ws.sum(axis=0), 1.0, atol=1e-6)
    assert float(ws[1, 0, 0]) == 0.0  # 0-preserving


def test_symmetric_humanoid_roi_bounds_and_inplace_clean_f32_watertight(tmp_path) -> None:
    from quadvox.contracts import load_profile_config
    from quadvox.s1_geometry.d3dv3_fusion import _eval_box_mask_np
    from quadvox.s1_geometry.repair import compute_mesh_metrics, inplace_clean_glb

    prof = load_profile_config("symmetric_humanoid")
    fusion = prof.s1["d3dv3_fusion"]
    assert float(fusion["guidance_scale"]) == 4.0
    assert int(fusion["max_tokens"]) == 200_000
    assert int(fusion["target_faces"]) == 2_000_000

    box = prof.roi_boxes()[0]
    m_back_of_head = _eval_box_mask_np(
        np.array([0.0]), np.array([0.875]), np.array([0.0]),
        x_range=box.x_range, y_range=box.y_range, z_range=box.z_range, margin=box.margin,
    )
    m_face_center = _eval_box_mask_np(
        np.array([0.0]), np.array([0.875]), np.array([0.04]),
        x_range=box.x_range, y_range=box.y_range, z_range=box.z_range, margin=box.margin,
    )
    assert float(m_back_of_head[0]) == 0.0
    assert float(m_face_center[0]) == 1.0

    # Inject a float32-sliver degenerate triangle onto a closed icosahedron by pinching two vertices within 1e-7
    ico = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    v = np.asarray(ico.vertices, dtype=np.float64).copy()
    f = np.asarray(ico.faces, dtype=np.int64).copy()
    # Collapse face 0's edge (v[f[0,0]] and v[f[0,1]]) within float32 rounding epsilon (3e-8)
    v[f[0, 1]] = v[f[0, 0]] + np.array([3e-8, 0.0, 0.0])
    sliver_glb = tmp_path / "sliver.glb"
    clean_glb = tmp_path / "clean.glb"
    trimesh.Trimesh(vertices=v, faces=f, process=False).export(str(sliver_glb))
    _, clean_m, _ = inplace_clean_glb(sliver_glb, clean_glb)
    recheck = compute_mesh_metrics(clean_glb)
    assert clean_m.is_watertight and recheck.is_watertight
    assert recheck.degenerate_faces == 0
    assert recheck.open_boundary_edges == 0


def test_coarse_splice_blend_weights_and_sdf_mesher() -> None:
    spec = CoarseSpliceSpec(
        enabled=True,
        split_axis="y",
        transition_range=(0.68, 0.76),
        lateral_guard_x=(0.15, 0.21),
        lateral_guard_y_max=(0.78, 0.81),
    )
    xw = np.linspace(-0.3, 0.3, 13, dtype=np.float32)
    yw = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    zw = np.linspace(-0.1, 0.1, 5, dtype=np.float32)
    w = compute_splice_blend_weights((xw, yw, zw), spec)
    # Bottom (y=0) uses primary (w=0); top center (x=0, y=1) uses secondary (w=1);
    # lateral arm region (|x|=0.3, y=0.75) stays protected (w near 0).
    assert float(w[6, 0, 2]) == 0.0
    assert float(w[6, -1, 2]) == 1.0
    assert float(w[0, 15, 2]) < 0.05

    mesh = build_sdf_mesh(
        lambda x, y, z: np.sqrt(x * x + y * y + z * z) - 0.35,
        bounds_min=(-0.5, -0.5, -0.5),
        bounds_max=(0.5, 0.5, 0.5),
        grid_shape=(24, 24, 24),
        taubin_iters=0,
        newton_iters=2,
    )
    assert mesh.is_watertight
    assert len(mesh.faces) > 200


def test_sstep_descending_and_crevice_keep_preserves_vertical_sides() -> None:
    vals = np.array([0.5, 0.0, -0.22, -0.40, -0.58, -0.9], dtype=np.float32)
    desc = _sstep_np(-0.22, -0.58, vals)
    expected = 1.0 - _sstep_np(-0.58, -0.22, vals)
    assert np.allclose(desc, expected, atol=1e-6)
    assert abs(float(desc[1]) - 0.0) < 1e-6  # vn_y = 0.0 (vertical side) -> not downward overhang
    assert abs(float(desc[3]) - 0.5) < 1e-6  # midpoint -0.40
    assert abs(float(desc[4]) - 1.0) < 1e-6  # vn_y <= -0.58 (true downward overhang)

    try:
        import torch  # type: ignore

        t_desc = _sstep(-0.22, -0.58, torch.tensor(vals))
        assert np.allclose(t_desc.numpy(), expected, atol=1e-6)
    except ImportError:
        pass

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.45)
    verts = np.asarray(sphere.vertices, dtype=np.float32)
    verts[:, 1] += 0.50
    faces = np.asarray(sphere.faces, dtype=np.int32)
    res = compute_surface_visibility_and_weights(verts, faces, zres=128, roi_boxes=())
    w_surf = res["w_surf"]
    vnorms = np.asarray(sphere.vertex_normals, dtype=np.float32)
    vertical_left = (vnorms[:, 0] > 0.85) & (np.abs(vnorms[:, 1]) < 0.15)
    assert int(vertical_left.sum()) >= 5
    assert float(w_surf[vertical_left, 2].mean()) >= 0.70


def test_crevice_keep_preflight_comparison_fixed_vs_legacy() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.42)
    verts = np.asarray(sphere.vertices, dtype=np.float32)
    verts[:, 1] += 0.50
    faces = np.asarray(sphere.faces, dtype=np.int32)

    pre = compute_surface_visibility_and_weights(
        verts, faces, zres=128, roi_boxes=(), crevice_keep_mode="legacy_unfixed"
    )
    post = compute_surface_visibility_and_weights(
        verts, faces, zres=128, roi_boxes=(), crevice_keep_mode="fixed"
    )
    vn = post["vn"]
    vis_tot = post["vis"].sum(axis=1)
    vert_partial = (np.abs(vn[:, 0]) > 0.70) & (np.abs(vn[:, 1]) < 0.20) & (vis_tot < 2.35)
    overhang = (vn[:, 1] < -0.58) & (vis_tot < 2.20)
    assert np.any(vert_partial) and np.any(overhang)
    assert float(post["crevice_keep"][vert_partial].mean()) > float(pre["crevice_keep"][vert_partial].mean()) + 0.50
    assert float(post["crevice_keep"][overhang].mean()) < float(pre["crevice_keep"][overhang].mean()) - 0.30

    cmp_rep = compare_crevice_keep_preflight(verts, faces, zres=128)
    assert cmp_rep["crevice_keep_max_abs_diff"] > 0.5
    assert cmp_rep["vertical_partial_crevice_keep_postfix"] > cmp_rep["vertical_partial_crevice_keep_preflight"]
