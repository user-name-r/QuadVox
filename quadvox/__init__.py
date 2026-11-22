"""Universal Multi-View 3D Production Engine (S0 -> S1 -> S2 -> S3).

Layer 1 (Universal Core): 4-view Direct3D-V3 1536^3 lossless 8^3 voxel fusion,
Pixal3D MV + CelloCut (QEF profiles), S2 SLIM low-island UV + 4K Cage baking + LATO.2,
and S3 region-protected delighting + Two-Band backprojection + 3D harmonic fill + PBR.

Layer 2 (Configurable Humanoid / Symmetric Switches): YAML-driven front_roi_box,
OpenVDB coarse_splice, and ReQuad symmetry_x toggles (all disabled by default).
"""

from quadvox.contracts import (
    AssetProfile,
    CoarseSpliceSpec,
    JobConfig,
    MeshMetrics,
    ROISovereigntyBox,
    RunMeta,
    SeedCandidate,
    SemanticSchema,
    ZoneSmoothingSpec,
    load_profile_config,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "AssetProfile",
    "CoarseSpliceSpec",
    "JobConfig",
    "MeshMetrics",
    "ROISovereigntyBox",
    "RunMeta",
    "SeedCandidate",
    "SemanticSchema",
    "ZoneSmoothingSpec",
    "load_profile_config",
]
