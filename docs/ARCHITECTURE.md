# Architecture & Algorithm Specification: Universal Core (Layer 1) & Configurable Switches (Layer 2)

**English** | [简体中文](ARCHITECTURE.zh-CN.md)

---

## 1. Architectural Principles

1. **Category-Agnostic Universal Core (Layer 1)**: `quadvox/` contains zero character-specific coordinates, hardcoded RGB color thresholds, or format-specific assumptions. All geometry reconstruction, retopology, SLIM UV unwrapping, 4K cage baking, and Two-Band PBR lookdev kernels operate on arbitrary 3D meshes.
2. **Opt-In Configurable Extensions via YAML (Layer 2)**: Capabilities tailored for humanoid or symmetric assets—**Front ROI View-Priority Lock (`front_roi_box`)**, **OpenVDB Level-Set Spatial Coarse Mesh Splicing (`coarse_splice`)**, and **`X=0` Mirror-Symmetric Quad Retopology (`symmetry_x`)**—are strictly parameterized via `configs/presets/*.yaml` and default to `false` / `null` in `configs/engine_defaults.yaml`.
3. **Host Safety & Process Isolation**:
   - Thin-client cloud orchestrators (`runners/`) never load multi-million-face meshes or run heavy Blender jobs on the local control machine; compute-intensive stages execute on remote GPU workers managed by `quadvox/cloud/colab_resilience.py`.
   - External GPL-3.0 tools (`Blender`, `ReQuad`, `QuadWild`, `ComfyUI`) are never bundled in the repository and are invoked strictly as standalone CLI subprocesses resolved via `quadvox/tools/external_tools.py`.
   - Zero hardcoded `/tmp` or machine-specific home directory paths anywhere in the codebase.

---

## 2. Stage 0 (`quadvox/s0_views/`): Multi-View Alignment & Orthographic Calibration

- **Geometry-First Matte Clay Prompting (`turnaround.py`)**: constructs geometry-focused prompts (aligned with Direct3D-V3 upstream `assets/prompt.txt`) that suppress baked lighting, specular reflections, and surface texturing prior to S1 geometry inference.
- **RGBA Normalization & Orbit Manifest (`preprocess.py`)**: extracts clean alpha mattes, centers and scales `0° / 90° / 180° / 270°` views with uniform padding (`pad_factor=1.1`), and writes canonical camera transforms (`transforms.json`).
- **Differentiable Orthographic Camera Calibration (`camera.py`)**: calibrates per-view scale and 2D translation (`calibrate_s0_ortho_camera`) against the target 3D mesh silhouette, paired with smooth silhouette alignment warps (`compute_silhouette_alignment_warp`).

---

## 3. Stage 1 (`quadvox/s1_geometry/`): 4-View Direct3D-V3 1536³ Lossless Sparse-Voxel Fusion

Implemented in `quadvox/s1_geometry/d3dv3_fusion.py`:

1. **Exact Discrete `8³` Sparse-Block Rotation (`build_view_rotation`, `rotate_latents_forward`, `rotate_latents_inverse`)**:
   - Permutes sparse block coordinates `(batch, px, py, pz)` and intra-block `8×8×8` voxel tensors by `0° / 90° / 180° / 270°` around `+Y` (`det = +1`).
   - Verified with a `<5ms` zero-error roundtrip self-check (`max_abs_err == 0.0`) before inference begins.
2. **Orthographic Occlusion Culling & Overhang Concavity Suppression (`compute_surface_visibility_and_weights`)**:
   - Rasterizes hole-free `512×512` orthographic Z-buffers for `front (+Z)`, `back (-Z)`, `left (+X)`, and `right (-X)`.
   - Evaluates steep occlusion depth discontinuities (`cliff_keep`) and downward-facing overhang concavities (`crevice_keep`) so tight concavities fall back smoothly to C1 front/back weights instead of suffering grazing-angle side-view tearing.
   - Projects surface visibility weights onto sparse `8³` lattice nodes (`compute_4view_voxel_weights`) with centered front/back fallback for interior voxels.
3. **Layer-2 Front ROI View-Priority Lock (`ROISovereigntyBox`)**:
   - Evaluates a C1-smooth 3D bounding box in normalized `[0, 1]` height-anchored coordinates (with optional `anchor_z_box` local depth centering), boosting `target_view` and suppressing orthogonal views inside the ROI.
4. **Timestep Directional Sharpening & Late CFG Annealing (`compute_step_beta_and_cfg`, `sharpen_view_weights`)**:
   - Temperature-sharpens normalized 4-view weights `w^beta / sum(w^beta)` for `t >= 0.55` (`beta_max = 3.5 / 4.5 / 5.5` at `512 / 1024 / 1536`).
   - Cosine-anneals CFG guidance scale after `t >= 0.72` down toward `cfg_late_min = 3.5`.

---

## 4. Layer-2 Configurable OpenVDB Level-Set Coarse Splice (`quadvox/s1_geometry/coarse_splice.py`)

When `s1.coarse_splice.enabled: true`, two complementary coarse meshes (for example, one with cleaner overall silhouette and another with sharper frontal details) are converted to OpenVDB `FloatGrid` signed-distance level sets at `voxel_res=384`, blended via a C1 smoothstep weight field along `split_axis` across `transition_range` (with optional `lateral_guard_x` / `lateral_guard_y_max` spatial protection), and polygonized back into a single watertight coarse mesh.

---

## 5. Stage 2 (`quadvox/s2_retopo/`) & Stage 3 (`quadvox/s3_lookdev/`)

- **Stage 2 — Retopology, Low-Island SLIM UV & 4K Welded-Cage Baking**:
  - `preclean.py` & `qem.py`: topology-preserving quadric edge collapse (`preserve_topology=True, planar_quadric=False` via PyMeshLab / pyfqmr / Open3D).
  - `lato2_repair.py`: repairs hard-surface `LATO.2` low-poly outputs (coplanar overlap removal, pinhole sealing, and coherent outward normal winding).
  - `quad_retopo.py`: subprocess wrappers for `ReQuad`, `QuadWild`, and `QuadriFlow` with configurable Layer-2 `symmetry_x` (`False` by default).
  - `slim_uv_bake.py`: crease-guided Blender 5.3 `MINIMUM_STRETCH` (SLIM) low-island UV unwrapping and 4K Cycles baking (`normal_4k.png`, `ao_4k.png`, `curvature_4k.png`) using a welded displaced cage (`WeldedSmoothBakeCage`).
- **Stage 3 — Delighting, Two-Band Backprojection & 3D Manifold Harmonic Inpainting**:
  - `delight.py`: region-protected delighting with masked exposure and chroma preservation.
  - `backproject.py`: frequency-separated Two-Band multi-view blending (`low_power=4.0`, `detail_power=32.0`), optional Layer-2 `ROISovereigntyBox` priority locking, seam-welded 3D Dirichlet harmonic hole filling across occluded surface regions, and Euclidean distance transform (EDT) UV gutter padding.
  - `materials.py` & `studio_export.py`: curvature/AO metallic PBR or generic PBR ORM map derivation and Blender 5.3 Cycles glTF 2.0 `.glb` export.
