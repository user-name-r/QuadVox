"""S3 Lookdev: Region-protected delighting, Two-Band backprojection, 3D harmonic fill, and PBR export."""

from quadvox.s3_lookdev.backproject import (
    bake_two_band_albedo,
    load_uv_mesh_and_labels,
    mesh_harmonic_fill,
    rasterize_uv_attributes,
)
from quadvox.s3_lookdev.delight import (
    apply_region_protected_delight,
    constrained_exposure_match,
    lab_to_rgb,
    linear_to_srgb,
    rgb_to_lab,
    srgb_to_linear,
)
from quadvox.s3_lookdev.materials import (
    blend_tangent_normal,
    derive_generic_pbr_maps,
    derive_prop_pbr_maps,
)

__all__ = [
    "apply_region_protected_delight",
    "bake_two_band_albedo",
    "blend_tangent_normal",
    "constrained_exposure_match",
    "derive_generic_pbr_maps",
    "derive_prop_pbr_maps",
    "lab_to_rgb",
    "linear_to_srgb",
    "load_uv_mesh_and_labels",
    "mesh_harmonic_fill",
    "rasterize_uv_attributes",
    "rgb_to_lab",
    "srgb_to_linear",
]
