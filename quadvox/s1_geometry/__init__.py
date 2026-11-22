"""S1 3D Geometry Generation, 4-View Direct3D-V3 Voxel Fusion, Repair, and Preview."""

from quadvox.s1_geometry.coarse_splice import (
    compute_splice_blend_weights,
    splice_coarse_glbs,
    splice_coarse_meshes_openvdb,
)
from quadvox.s1_geometry.d3dv3_fusion import (
    D3DV3FusionConfig,
    VIEW_DIRS,
    VIEW_ORDER,
    build_view_rotation,
    compute_4view_voxel_weights,
    compute_physical_frame_meta,
    compute_step_beta_and_cfg,
    compute_surface_visibility_and_weights,
    normalize_surface_to_voxel_world,
    prune_stage512_micro_shards,
    rotate_latents_forward,
    rotate_latents_inverse,
    run_d3dv3_4view_fusion,
    sample_x0_quad,
    sharpen_view_weights,
    verify_torch_rotations_selfcheck,
)
from quadvox.s1_geometry.sdf_mesher import build_sdf_mesh, extract_marching_tetrahedra
from quadvox.s1_geometry.selector import candidate_sort_key, rank_candidates, select_best_candidate

__all__ = [
    "D3DV3FusionConfig",
    "VIEW_DIRS",
    "VIEW_ORDER",
    "build_sdf_mesh",
    "build_view_rotation",
    "candidate_sort_key",
    "compute_4view_voxel_weights",
    "compute_physical_frame_meta",
    "compute_splice_blend_weights",
    "compute_step_beta_and_cfg",
    "compute_surface_visibility_and_weights",
    "extract_marching_tetrahedra",
    "normalize_surface_to_voxel_world",
    "prune_stage512_micro_shards",
    "rank_candidates",
    "rotate_latents_forward",
    "rotate_latents_inverse",
    "run_d3dv3_4view_fusion",
    "sample_x0_quad",
    "select_best_candidate",
    "sharpen_view_weights",
    "splice_coarse_glbs",
    "splice_coarse_meshes_openvdb",
    "verify_torch_rotations_selfcheck",
]
