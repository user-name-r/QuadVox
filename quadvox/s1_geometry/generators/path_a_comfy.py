"""Path A generator adapter: ComfyUI Pixal3D int8_convrot (shape-only) + BiRefNet + RemeshMesh.

Verified upstream URLs:
- ComfyUI Pixal3D docs: https://github.com/Comfy-Org/docs/blob/main/tutorials/3d/pixal3d.mdx
- ComfyUI Trellis2/Pixal3D nodes: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_trellis2.py
- ComfyUI mesh postprocess nodes: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_mesh_postprocess.py
- ComfyUI official MV workflow: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/3d_pixal3d_multi_views.json
- ComfyUI Pixal3D weights: https://huggingface.co/Comfy-Org/Pixal3D
- ComfyUI Pixal3D fallback mirror: https://huggingface.co/chibifire/Pixal3D-Comfy
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from quadvox.s1_geometry.generators.base import (
    BaseShapeGenerator,
    GenerationResult,
    default_cache_dir,
    monitor_subprocess_with_vram,
    resolve_views_from_dir,
)


COMFYUI_REPO_URL = "https://github.com/Comfy-Org/ComfyUI.git"
COMFY_PIXAL3D_DOCS_URL = "https://github.com/Comfy-Org/docs/blob/main/tutorials/3d/pixal3d.mdx"
COMFY_NODES_TRELLIS2_URL = (
    "https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_trellis2.py"
)
COMFY_NODES_POSTPROCESS_URL = (
    "https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_mesh_postprocess.py"
)
COMFY_MV_WORKFLOW_URL = (
    "https://github.com/Comfy-Org/workflow_templates/blob/main/templates/3d_pixal3d_multi_views.json"
)

HF_COMFY_PIXAL3D_REPO = "Comfy-Org/Pixal3D"
HF_FALLBACK_PIXAL3D_REPO = "chibifire/Pixal3D-Comfy"
HF_COMFY_BIREFNET_REPO = "Comfy-Org/BiRefNet"
HF_COMFY_MOGE_REPO = "Comfy-Org/MoGe"

# Geometry-only subset (deliberately excludes trellis_2_texture_vae_bf16.safetensors).
PATH_A_SHARED_FILES = (
    "vae/trellis_2_shape_vae_bf16.safetensors",
    "clip_vision/dino_v3_L_naf_fp32.safetensors",
)
PATH_A_MV_UNET = "diffusion_models/pixal3d_multiview_int8_convrot.safetensors"
PATH_A_SV_UNET = "diffusion_models/pixal3d_int8_convrot.safetensors"
PATH_A_BIREFNET_FILE = "background_removal/birefnet.safetensors"
PATH_A_MOGE_FILE = "geometry_estimation/moge_2_vitl_normal_fp16.safetensors"


def default_comfy_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve ComfyUI checkout directory under `/content/cache` or local `runs/_cache`."""
    env_dir = os.environ.get("COMFYUI_DIR")
    if env_dir:
        return Path(env_dir)
    base = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return base / "ComfyUI"


def default_comfy_models_dir(
    comfy_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Resolve ComfyUI models directory."""
    env_dir = os.environ.get("COMFYUI_MODELS_DIR")
    if env_dir:
        return Path(env_dir)
    if cache_dir is not None:
        return Path(cache_dir) / "models" / "comfyui"
    cdir = default_comfy_dir()
    return cdir / "models"


def required_model_relpaths(mode: str = "mv", include_moge: bool = False) -> list[str]:
    """Return required relative model paths for Path A shape-only generation."""
    norm_mode = str(mode).lower()
    if norm_mode not in ("mv", "sv"):
        raise ValueError(f"Unsupported mode {mode!r}; expected 'mv' or 'sv'")
    files = list(PATH_A_SHARED_FILES)
    files.append(PATH_A_MV_UNET if norm_mode == "mv" else PATH_A_SV_UNET)
    if norm_mode == "sv" and include_moge:
        files.append(PATH_A_MOGE_FILE)
    return files


def check_required_models(
    models_dir: str | Path,
    mode: str = "mv",
    include_moge: bool = False,
) -> list[Path]:
    """Return missing required model paths under `models_dir`."""
    root = Path(models_dir)
    missing: list[Path] = []
    for rel in required_model_relpaths(mode=mode, include_moge=include_moge):
        p = root / rel
        if not p.is_file() or p.stat().st_size == 0:
            missing.append(p)
    return missing


def _init_headless_comfy(comfy_dir: Path, models_dir: Path, out_dir: Path) -> None:
    """Initialize ComfyUI modules for headless Python execution without CLI arg parsing."""
    if not comfy_dir.is_dir():
        raise FileNotFoundError(
            f"ComfyUI directory not found at {comfy_dir}. "
            f"Clone {COMFYUI_REPO_URL} or set COMFYUI_DIR."
        )

    comfy_str = str(comfy_dir.resolve())
    if comfy_str not in sys.path:
        sys.path.insert(0, comfy_str)

    # Prevent ComfyUI from parsing outer script flags.
    sys.argv = [sys.argv[0]]
    import comfy.options  # type: ignore

    comfy.options.enable_args_parsing(False)

    import comfy.utils  # type: ignore
    import folder_paths  # type: ignore

    comfy.utils.PROGRESS_BAR_ENABLED = False
    out_dir.mkdir(parents=True, exist_ok=True)
    folder_paths.set_output_directory(str(out_dir))

    for sub in (
        "diffusion_models",
        "vae",
        "clip_vision",
        "background_removal",
        "geometry_estimation",
    ):
        sub_path = models_dir / sub
        if sub_path.exists():
            folder_paths.add_model_folder_path(sub, str(sub_path), is_default=True)


def _load_rgba_tensor(path: Path) -> "Any":
    """Load an image as `[1, H, W, 4]` float32 tensor and verify non-trivial alpha."""
    import torch

    with Image.open(path) as im:
        im.load()
        if im.mode not in ("RGBA", "LA", "PA"):
            raise ValueError(
                f"View {path} has mode={im.mode}; Path A expects pre-matted RGBA input."
            )
        rgba = im.convert("RGBA")
        arr = np.asarray(rgba, dtype=np.uint8)
    alpha = arr[:, :, 3]
    if np.all(alpha == 255) or not np.any(alpha > 0):
        raise ValueError(f"View {path} lacks a valid foreground alpha channel.")
    tensor = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)
    return tensor


def _is_canonical_four_view_orbit(
    view_paths: Sequence[Path],
    transforms_meta: dict[str, Any] | None,
) -> bool:
    if len(view_paths) != 4:
        return False
    if transforms_meta is None:
        return True
    frames = transforms_meta.get("frames", [])
    if len(frames) != 4:
        return False
    mesh_scale = float(transforms_meta.get("mesh_scale", 1.0))
    if abs(mesh_scale - 1.0) > 1e-4:
        return False
    expected_names = ("azim000", "azim090", "azim180", "azim270")
    names = [str(fr.get("name", "")) for fr in frames]
    if all(bool(n) for n in names) and tuple(names) == expected_names:
        return True
    return all("transform_matrix" not in fr for fr in frames)


def _build_mv_conditioning(
    clip_vision: Any,
    view_paths: Sequence[Path],
    transforms_meta: dict[str, Any] | None,
    fov_deg: float,
) -> tuple[Any, Any]:
    """Build Pixal3D MV conditioning matching ComfyUI `Pixal3DMultiViewConditioning`."""
    import torch
    from comfy_extras.nodes_trellis2 import (  # type: ignore
        Pixal3DMultiViewConditioning,
        _build_pixal3d_conditioning,
    )

    view_tensors = [_load_rgba_tensor(p) for p in view_paths]

    if _is_canonical_four_view_orbit(view_paths, transforms_meta):
        resolved_fov = float(fov_deg)
        if transforms_meta is not None and "camera_angle_x" in transforms_meta:
            resolved_fov = math.degrees(float(transforms_meta["camera_angle_x"]))
        node_cls = Pixal3DMultiViewConditioning.PREPARE_CLASS_CLONE(None)
        out = node_cls.execute(
            clip_vision_model=clip_vision,
            fov=resolved_fov,
            front=view_tensors[0],
            left=view_tensors[1],
            back=view_tensors[2],
            right=view_tensors[3],
        )
        return out[0], out[1]

    if transforms_meta is None or not transforms_meta.get("frames"):
        raise ValueError(
            "Non-canonical multi-view input requires transforms.json with camera poses."
        )

    frames = transforms_meta["frames"]
    top_fov = transforms_meta.get("camera_angle_x", math.radians(float(fov_deg)))
    mesh_scale = float(transforms_meta.get("mesh_scale", 1.0))

    premult_views = []
    matrices = []
    fovs = []
    for tensor, fr in zip(view_tensors, frames):
        rgb = tensor[..., :3] * tensor[..., 3:4]
        if rgb.shape[1:3] != (1024, 1024):
            import comfy.utils  # type: ignore

            rgb = (
                comfy.utils.common_upscale(
                    rgb.movedim(-1, 1), 1024, 1024, "lanczos", "disabled"
                )
                .movedim(1, -1)
            )
        premult_views.append(rgb)
        matrices.append(fr["transform_matrix"])
        fovs.append(float(fr.get("camera_angle_x", top_fov)))

    images_cat = torch.cat(premult_views, dim=0)
    c2w = torch.tensor(matrices, dtype=torch.float32)
    cam_angles = torch.tensor(fovs, dtype=torch.float32)
    mesh_scale_t = torch.tensor([mesh_scale], dtype=torch.float32)
    out = _build_pixal3d_conditioning(
        clip_vision,
        images_cat,
        c2w,
        cam_angles,
        mesh_scale_t,
        num_views=len(premult_views),
    )
    return out[0], out[1]


def _build_sv_conditioning(
    clip_vision: Any,
    view_path: Path,
    fov_deg: float,
    use_moge: bool = False,
) -> tuple[Any, Any, float]:
    """Build Pixal3D SV conditioning matching ComfyUI `Pixal3DConditioning`."""
    from comfy_extras.nodes_trellis2 import Pixal3DConditioning  # type: ignore

    rgba_t = _load_rgba_tensor(view_path)
    rgb_premult = rgba_t[..., :3] * rgba_t[..., 3:4]

    resolved_fov = float(fov_deg)
    if use_moge and resolved_fov <= 0.0:
        from comfy_extras.nodes_moge import (  # type: ignore
            LoadMoGeModel,
            MoGeGeometryToFOV,
            MoGeInference,
        )

        moge_model = LoadMoGeModel.PREPARE_CLASS_CLONE(None).execute(
            "moge_2_vitl_normal_fp16.safetensors"
        )[0]
        moge_geom = MoGeInference.PREPARE_CLASS_CLONE(None).execute(
            moge_model=moge_model,
            image=rgb_premult,
            resolution_level=9,
            fov_x_degrees=0.0,
            batch_size=4,
            force_projection=True,
            apply_mask=True,
            refine_steps=3,
        )[0]
        resolved_fov = float(
            MoGeGeometryToFOV.PREPARE_CLASS_CLONE(None).execute(
                moge_geometry=moge_geom,
                axis="horizontal",
                unit="degrees",
            )[0]
        )
    elif resolved_fov <= 0.0:
        resolved_fov = 49.13

    out = Pixal3DConditioning.PREPARE_CLASS_CLONE(None).execute(
        clip_vision_model=clip_vision,
        image=rgb_premult,
        camera_angle_x=resolved_fov,
    )
    return out[0], out[1], resolved_fov


def _run_comfy_generate_inprocess(
    views_dir: Path,
    out_glb: Path,
    *,
    mode: str,
    resolution: int,
    seed: int,
    fov_deg: float,
    comfy_dir: Path,
    models_dir: Path,
    use_moge: bool = False,
) -> dict[str, Any]:
    """Execute ComfyUI Pixal3D shape-only graph in the current process."""
    t0 = time.monotonic()
    _init_headless_comfy(comfy_dir, models_dir, out_glb.parent)

    import torch
    import nodes  # type: ignore
    from comfy_extras import nodes_custom_sampler, nodes_model_advanced  # type: ignore
    from comfy_extras.nodes_save_3d import mesh_item_to_glb_bytes  # type: ignore
    from comfy_extras.nodes_trellis2 import (  # type: ignore
        EmptyTrellis2LatentStructure,
        Trellis2ShapeStage,
        Trellis2UpsampleStage,
        VaeDecodeShapeTrellis,
        VaeDecodeStructureTrellis2,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device is not available for Path A ComfyUI generation; refusing CPU fallback."
        )

    with torch.inference_mode():
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(int(seed))

        view_paths, transforms_meta = resolve_views_from_dir(views_dir, mode=mode)
        unet_rel = PATH_A_MV_UNET if mode == "mv" else PATH_A_SV_UNET
        unet_name = Path(unet_rel).name

        print(
            f"[path_a] Loading models: unet={unet_name} res={resolution} mode={mode} seed={seed}",
            flush=True,
        )
        model = nodes.UNETLoader().load_unet(unet_name, "default")[0]
        clip_vision = nodes.CLIPVisionLoader().load_clip("dino_v3_L_naf_fp32.safetensors")[0]
        vae = nodes.VAELoader().load_vae("trellis_2_shape_vae_bf16.safetensors")[0]

        if mode == "mv":
            pos, neg = _build_mv_conditioning(clip_vision, view_paths, transforms_meta, fov_deg)
            used_fov = fov_deg
        else:
            pos, neg, used_fov = _build_sv_conditioning(
                clip_vision, view_paths[0], fov_deg=fov_deg, use_moge=use_moge
            )

        print("[path_a] Stage 1/3: Sparse structure sampling (12 steps, shift=5.0)...", flush=True)
        latent_ss = EmptyTrellis2LatentStructure.PREPARE_CLASS_CLONE(None).execute(batch_size=1)[0]
        m_ss = nodes_custom_sampler.CFGOverride.PREPARE_CLASS_CLONE(None).execute(
            model=model, cfg=1.0, start_percent=0.667, end_percent=1.0
        )[0]
        m_ss = nodes_model_advanced.RescaleCFG().patch(m_ss, multiplier=0.7)[0]
        m_ss = nodes_model_advanced.ModelSamplingSD3().patch(m_ss, shift=5.0)[0]

        ksampler = nodes.KSampler()
        ss_samples = ksampler.sample(
            model=m_ss,
            seed=int(seed),
            steps=12,
            cfg=7.5,
            sampler_name="euler",
            scheduler="normal",
            positive=pos,
            negative=neg,
            latent_image=latent_ss,
            denoise=1.0,
        )[0]
        voxel = VaeDecodeStructureTrellis2.PREPARE_CLASS_CLONE(None).execute(
            samples=ss_samples, vae=vae, resolution="32"
        )[0]

        print("[path_a] Stage 2/3: Shape LR-512 sampling (20 steps)...", flush=True)
        pos_s1, neg_s1, lat_s1 = Trellis2ShapeStage.PREPARE_CLASS_CLONE(None).execute(
            positive=pos, negative=neg, voxel=voxel
        )
        m_sh = nodes_custom_sampler.CFGOverride.PREPARE_CLASS_CLONE(None).execute(
            model=model, cfg=1.0, start_percent=0.769, end_percent=1.0
        )[0]
        m_sh = nodes_model_advanced.RescaleCFG().patch(m_sh, multiplier=0.5)[0]
        s1_samples = ksampler.sample(
            model=m_sh,
            seed=int(seed),
            steps=20,
            cfg=7.5,
            sampler_name="euler",
            scheduler="normal",
            positive=pos_s1,
            negative=neg_s1,
            latent_image=lat_s1,
            denoise=1.0,
        )[0]

        if int(resolution) >= 1024:
            print(
                f"[path_a] Stage 3/3: Shape HR-{resolution} cascade sampling (12 steps, simple)...",
                flush=True,
            )
            pos_s2, neg_s2, lat_s2 = Trellis2UpsampleStage.PREPARE_CLASS_CLONE(None).execute(
                positive=pos_s1,
                negative=neg_s1,
                shape_latent=s1_samples,
                vae=vae,
                target_resolution=int(resolution),
            )
            final_samples = ksampler.sample(
                model=m_sh,
                seed=int(seed),
                steps=12,
                cfg=7.5,
                sampler_name="euler",
                scheduler="simple",
                positive=pos_s2,
                negative=neg_s2,
                latent_image=lat_s2,
                denoise=1.0,
            )[0]
        else:
            final_samples = s1_samples

        print("[path_a] Decoding shape latent via VaeDecodeShapeTrellis...", flush=True)
        mesh_out, _subs = VaeDecodeShapeTrellis.PREPARE_CLASS_CLONE(None).execute(
            samples=final_samples, vae=vae
        )
        glb_bytes = mesh_item_to_glb_bytes(mesh_out, 0)
        if not glb_bytes:
            raise RuntimeError("VaeDecodeShapeTrellis produced an empty mesh.")

    out_glb.parent.mkdir(parents=True, exist_ok=True)
    out_glb.write_bytes(glb_bytes)

    elapsed_s = time.monotonic() - t0
    peak_alloc_gb = float(torch.cuda.max_memory_allocated()) / (1024.0 ** 3)
    peak_reserved_gb = float(torch.cuda.max_memory_reserved()) / (1024.0 ** 3)
    model_frame = str(final_samples.get("model_frame", "y_up"))
    coord_frame = "y_up_z_front" if model_frame == "y_up" else model_frame
    verts_shape = tuple(int(x) for x in mesh_out.vertices.shape)
    faces_shape = tuple(int(x) for x in mesh_out.faces.shape)

    print(
        f"[path_a] Done in {elapsed_s:.1f}s -> {out_glb} "
        f"(peak_alloc={peak_alloc_gb:.2f}GB, model_frame={model_frame}, faces={faces_shape})",
        flush=True,
    )
    return {
        "backend": "path_a",
        "mode": mode,
        "seed": int(seed),
        "resolution": int(resolution),
        "raw_glb": str(out_glb),
        "elapsed_s": elapsed_s,
        "peak_vram_gb": peak_alloc_gb,
        "peak_reserved_vram_gb": peak_reserved_gb,
        "coordinate_frame": coord_frame,
        "model_frame": model_frame,
        "fov_deg": float(used_fov),
        "vertices_shape": list(verts_shape),
        "faces_shape": list(faces_shape),
    }


def run_birefnet_matte_inprocess(
    image_paths: Sequence[str | Path],
    out_dir: str | Path,
    *,
    comfy_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
) -> list[Path]:
    """Run ComfyUI BiRefNet (`birefnet.safetensors`) on RGB images, emitting RGBA PNGs."""
    cdir = Path(comfy_dir) if comfy_dir is not None else default_comfy_dir()
    mdir = Path(models_dir) if models_dir is not None else default_comfy_models_dir(cdir)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    biref_path = mdir / PATH_A_BIREFNET_FILE
    if not biref_path.is_file():
        raise FileNotFoundError(
            f"BiRefNet checkpoint missing at {biref_path} "
            f"(from https://huggingface.co/{HF_COMFY_BIREFNET_REPO})"
        )

    _init_headless_comfy(cdir, mdir, dest)
    import torch
    from comfy_extras.nodes_bg_removal import (  # type: ignore
        LoadBackgroundRemovalModel,
        RemoveBackground,
    )

    with torch.inference_mode():
        bg_model = LoadBackgroundRemovalModel.PREPARE_CLASS_CLONE(None).execute(
            "birefnet.safetensors"
        )[0]
        rmbg_cls = RemoveBackground.PREPARE_CLASS_CLONE(None)

        outputs: list[Path] = []
        for idx, src_item in enumerate(image_paths):
            src = Path(src_item)
            with Image.open(src) as im:
                im.load()
                rgb_pil = im.convert("RGB")
            rgb_np = np.asarray(rgb_pil, dtype=np.float32) / 255.0
            img_t = torch.from_numpy(rgb_np).unsqueeze(0)
            mask_t = rmbg_cls.execute(bg_removal_model=bg_model, image=img_t)[0]
            mask_np = np.clip(mask_t[0].detach().cpu().numpy() * 255.0, 0.0, 255.0).astype(
                np.uint8
            )
            rgba_np = np.dstack([np.asarray(rgb_pil, dtype=np.uint8), mask_np])
            rgba_np[rgba_np[:, :, 3] == 0, :3] = 0
            out_path = dest / f"{src.stem}_rgba.png"
            if out_path.exists():
                out_path = dest / f"{idx:02d}_{src.stem}_rgba.png"
            Image.fromarray(rgba_np, mode="RGBA").save(out_path, format="PNG")
            outputs.append(out_path)
    return outputs


def run_birefnet_matte(
    image_paths: Sequence[str | Path],
    out_dir: str | Path,
    *,
    comfy_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    isolated: bool = True,
    log_path: str | Path | None = None,
) -> list[Path]:
    """Extract RGBA mattes via ComfyUI BiRefNet, defaulting to an isolated subprocess."""
    cdir = Path(comfy_dir) if comfy_dir is not None else default_comfy_dir()
    mdir = Path(models_dir) if models_dir is not None else default_comfy_models_dir(cdir)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if not isolated:
        return run_birefnet_matte_inprocess(
            image_paths, dest, comfy_dir=cdir, models_dir=mdir
        )

    manifest_path = dest / "_birefnet_io.json"
    manifest_path.write_text(
        json.dumps({"inputs": [str(Path(p)) for p in image_paths]}, indent=2),
        encoding="utf-8",
    )
    cmd = [
        sys.executable,
        "-m",
        "quadvox.s1_geometry.generators.path_a_comfy",
        "birefnet",
        "--manifest",
        str(manifest_path),
        "--out-dir",
        str(dest),
        "--comfy-dir",
        str(cdir),
        "--models-dir",
        str(mdir),
    ]
    rc, _, _, tail = monitor_subprocess_with_vram(
        cmd,
        log_path=log_path,
        heartbeat_label="birefnet",
    )
    if rc != 0:
        raise RuntimeError(f"BiRefNet subprocess failed (rc={rc}):\n{tail}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [Path(p) for p in data["outputs"]]


def run_comfy_remesh_solid_inprocess(
    in_glb: str | Path,
    out_glb: str | Path,
    *,
    resolution: int = 1024,
    fill: float = 20.0,
    band: float = 1.0,
    project_back: float = 0.5,
    qef: bool = False,
    drop_small_components: float = 0.01,
    smooth_iters: int = 0,
    fix_poles: bool = False,
    precluster_max_verts: int = 20_000_000,
    comfy_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run ComfyUI `RemeshMesh` (`sign_mode='solid'`, CelloCut) in-process on `in_glb`."""
    import torch
    import trimesh

    t0 = time.monotonic()
    src = Path(in_glb)
    dst = Path(out_glb)
    cdir = Path(comfy_dir) if comfy_dir is not None else default_comfy_dir()
    mdir = Path(models_dir) if models_dir is not None else default_comfy_models_dir(cdir)
    _init_headless_comfy(cdir, mdir, dst.parent)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device is not available for ComfyUI RemeshMesh solid (CelloCut); "
            "refusing silent local fallback."
        )

    from comfy_api.latest import Types  # type: ignore
    from comfy_extras.nodes_mesh_postprocess import RemeshMesh  # type: ignore
    from comfy_extras.nodes_save_3d import mesh_item_to_glb_bytes  # type: ignore

    # Always disable automatic vertex welding/processing on read.
    loaded = trimesh.load(str(src), force="mesh", process=False)
    verts_np = np.asarray(loaded.vertices, dtype=np.float32)
    faces_np = np.asarray(loaded.faces, dtype=np.int32)
    if verts_np.size == 0 or faces_np.size == 0:
        raise ValueError(f"Input mesh {src} has no geometry")

    mesh_obj = Types.MESH(
        vertices=torch.from_numpy(verts_np).unsqueeze(0),
        faces=torch.from_numpy(faces_np).unsqueeze(0),
    )

    with torch.inference_mode():
        torch.cuda.reset_peak_memory_stats()
        print(
            f"[remesh_solid] Running ComfyUI RemeshMesh(solid, res={resolution}, fill={fill}, "
            f"band={band}, project_back={project_back}, qef={bool(qef)}) on {src.name}...",
            flush=True,
        )
        remeshed = RemeshMesh.PREPARE_CLASS_CLONE(None).execute(
            mesh=mesh_obj,
            resolution=int(resolution),
            sign_mode={"sign_mode": "solid", "fill": float(fill), "qef": bool(qef)},
            band=float(band),
            project_back=float(project_back),
            fix_poles=bool(fix_poles),
            smooth_iters=int(smooth_iters),
            drop_small_components=float(drop_small_components),
            precluster_max_verts=int(precluster_max_verts),
        )[0]

        glb_bytes = mesh_item_to_glb_bytes(remeshed, 0)
        if not glb_bytes:
            raise RuntimeError(f"ComfyUI RemeshMesh produced an empty mesh for {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(glb_bytes)

    elapsed_s = time.monotonic() - t0
    peak_vram_gb = float(torch.cuda.max_memory_allocated()) / (1024.0 ** 3)
    return {
        "in_glb": str(src),
        "out_glb": str(dst),
        "resolution": int(resolution),
        "sign_mode": "solid",
        "fill": float(fill),
        "band": float(band),
        "project_back": float(project_back),
        "qef": bool(qef),
        "elapsed_s": elapsed_s,
        "peak_vram_gb": peak_vram_gb,
        "out_vertices": int(remeshed.vertices.shape[1]),
        "out_faces": int(remeshed.faces.shape[1]),
    }


def run_comfy_remesh_solid(
    in_glb: str | Path,
    out_glb: str | Path,
    *,
    resolution: int = 1024,
    fill: float = 20.0,
    band: float = 1.0,
    project_back: float = 0.5,
    qef: bool = False,
    drop_small_components: float = 0.01,
    smooth_iters: int = 0,
    fix_poles: bool = False,
    precluster_max_verts: int = 20_000_000,
    comfy_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    isolated: bool = True,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run ComfyUI `RemeshMesh` (`sign_mode='solid'`), isolated in a subprocess by default."""
    cdir = Path(comfy_dir) if comfy_dir is not None else default_comfy_dir()
    mdir = Path(models_dir) if models_dir is not None else default_comfy_models_dir(cdir)
    dst = Path(out_glb)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not isolated:
        return run_comfy_remesh_solid_inprocess(
            in_glb=in_glb,
            out_glb=dst,
            resolution=resolution,
            fill=fill,
            band=band,
            project_back=project_back,
            qef=qef,
            drop_small_components=drop_small_components,
            smooth_iters=smooth_iters,
            fix_poles=fix_poles,
            precluster_max_verts=precluster_max_verts,
            comfy_dir=cdir,
            models_dir=mdir,
        )

    meta_json = dst.with_suffix(".remesh_meta.json")
    cmd = [
        sys.executable,
        "-m",
        "quadvox.s1_geometry.generators.path_a_comfy",
        "remesh",
        "--in-glb",
        str(Path(in_glb)),
        "--out-glb",
        str(dst),
        "--resolution",
        str(int(resolution)),
        "--fill",
        str(float(fill)),
        "--band",
        str(float(band)),
        "--project-back",
        str(float(project_back)),
        "--qef" if qef else "--no-qef",
        "--drop-small",
        str(float(drop_small_components)),
        "--smooth-iters",
        str(int(smooth_iters)),
        "--precluster-max-verts",
        str(int(precluster_max_verts)),
        "--comfy-dir",
        str(cdir),
        "--models-dir",
        str(mdir),
        "--meta-out",
        str(meta_json),
    ]
    if fix_poles:
        cmd.append("--fix-poles")

    rc, wall_s, smi_peak_gb, tail = monitor_subprocess_with_vram(
        cmd,
        log_path=log_path,
        heartbeat_label=f"remesh_solid_pb{project_back}",
    )
    if rc != 0:
        raise RuntimeError(f"ComfyUI RemeshMesh subprocess failed (rc={rc}):\n{tail}")
    info = json.loads(meta_json.read_text(encoding="utf-8"))
    info["wall_elapsed_s"] = wall_s
    info["peak_vram_gb"] = max(float(info.get("peak_vram_gb", 0.0)), smi_peak_gb)
    return info


class PathAComfyGenerator(BaseShapeGenerator):
    """ComfyUI Pixal3D int8_convrot shape-only generator (isolated subprocess per seed)."""

    backend_name: str = "path_a"
    coordinate_frame: str = "y_up_z_front"
    supported_resolutions: tuple[int, ...] = (512, 1024, 1536)

    def __init__(
        self,
        comfy_dir: str | Path | None = None,
        models_dir: str | Path | None = None,
        use_moge: bool = False,
    ) -> None:
        self.comfy_dir = (
            Path(comfy_dir) if comfy_dir is not None else default_comfy_dir()
        )
        self.models_dir = (
            Path(models_dir)
            if models_dir is not None
            else default_comfy_models_dir(self.comfy_dir)
        )
        self.use_moge = bool(use_moge)

    def generate(
        self,
        views_dir: str | Path,
        out_glb: str | Path,
        *,
        mode: str = "mv",
        resolution: int = 1536,
        seed: int = 42,
        fov_deg: float = 20.0,
        log_path: str | Path | None = None,
    ) -> GenerationResult:
        norm_mode, res = self.validate_request(mode, resolution)
        src = Path(views_dir)
        dst = Path(out_glb)
        dst.parent.mkdir(parents=True, exist_ok=True)

        missing = check_required_models(
            self.models_dir,
            mode=norm_mode,
            include_moge=(norm_mode == "sv" and self.use_moge),
        )
        if missing:
            missing_str = ", ".join(str(p) for p in missing)
            raise FileNotFoundError(
                f"Path A required geometry weights missing under {self.models_dir}: {missing_str}"
            )
        if not self.comfy_dir.is_dir():
            raise FileNotFoundError(f"ComfyUI checkout missing at {self.comfy_dir}")

        meta_json = dst.with_suffix(".path_a_meta.json")
        resolved_log = (
            Path(log_path)
            if log_path is not None
            else dst.with_suffix(".path_a.log")
        )
        cmd = [
            sys.executable,
            "-m",
            "quadvox.s1_geometry.generators.path_a_comfy",
            "generate",
            "--views-dir",
            str(src),
            "--out-glb",
            str(dst),
            "--mode",
            norm_mode,
            "--resolution",
            str(res),
            "--seed",
            str(int(seed)),
            "--fov-deg",
            str(float(fov_deg)),
            "--comfy-dir",
            str(self.comfy_dir),
            "--models-dir",
            str(self.models_dir),
            "--meta-out",
            str(meta_json),
        ]
        if self.use_moge:
            cmd.append("--use-moge")

        rc, wall_s, smi_peak_gb, tail = monitor_subprocess_with_vram(
            cmd,
            log_path=resolved_log,
            heartbeat_label=f"path_a_{norm_mode}_s{seed}",
        )
        if rc != 0:
            raise RuntimeError(
                f"Path A ComfyUI generation failed (rc={rc}) for seed={seed}:\n{tail}"
            )
        if not dst.is_file() or dst.stat().st_size == 0:
            raise RuntimeError(f"Path A succeeded but output GLB is missing/empty: {dst}")

        info = json.loads(meta_json.read_text(encoding="utf-8"))
        peak_vram = max(float(info.get("peak_vram_gb", 0.0)), smi_peak_gb)
        return GenerationResult(
            backend=self.backend_name,
            mode=norm_mode,
            seed=int(seed),
            resolution=res,
            raw_glb=str(dst),
            elapsed_s=float(info.get("elapsed_s", wall_s)),
            peak_vram_gb=peak_vram,
            coordinate_frame=str(info.get("coordinate_frame", self.coordinate_frame)),
            log_path=str(resolved_log),
            extra={
                "wall_elapsed_s": wall_s,
                "peak_reserved_vram_gb": float(info.get("peak_reserved_vram_gb", 0.0)),
                "model_frame": info.get("model_frame", "y_up"),
                "fov_deg": float(info.get("fov_deg", fov_deg)),
                "vertices_shape": info.get("vertices_shape"),
                "faces_shape": info.get("faces_shape"),
            },
        )


def _cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Path A ComfyUI isolated worker")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--views-dir", required=True)
    p_gen.add_argument("--out-glb", required=True)
    p_gen.add_argument("--mode", default="mv", choices=("mv", "sv"))
    p_gen.add_argument("--resolution", type=int, default=1536)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--fov-deg", type=float, default=20.0)
    p_gen.add_argument("--comfy-dir", required=True)
    p_gen.add_argument("--models-dir", required=True)
    p_gen.add_argument("--meta-out", required=True)
    p_gen.add_argument("--use-moge", action="store_true")

    p_bg = sub.add_parser("birefnet")
    p_bg.add_argument("--manifest", required=True)
    p_bg.add_argument("--out-dir", required=True)
    p_bg.add_argument("--comfy-dir", required=True)
    p_bg.add_argument("--models-dir", required=True)

    p_rm = sub.add_parser("remesh")
    p_rm.add_argument("--in-glb", required=True)
    p_rm.add_argument("--out-glb", required=True)
    p_rm.add_argument("--resolution", type=int, default=1024)
    p_rm.add_argument("--fill", type=float, default=20.0)
    p_rm.add_argument("--band", type=float, default=1.0)
    p_rm.add_argument("--project-back", type=float, default=0.5)
    p_rm.add_argument("--qef", action=argparse.BooleanOptionalAction, default=False)
    p_rm.add_argument("--drop-small", type=float, default=0.01)
    p_rm.add_argument("--smooth-iters", type=int, default=0)
    p_rm.add_argument("--precluster-max-verts", type=int, default=20_000_000)
    p_rm.add_argument("--fix-poles", action="store_true")
    p_rm.add_argument("--comfy-dir", required=True)
    p_rm.add_argument("--models-dir", required=True)
    p_rm.add_argument("--meta-out", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "generate":
        meta = _run_comfy_generate_inprocess(
            views_dir=Path(args.views_dir),
            out_glb=Path(args.out_glb),
            mode=args.mode,
            resolution=args.resolution,
            seed=args.seed,
            fov_deg=args.fov_deg,
            comfy_dir=Path(args.comfy_dir),
            models_dir=Path(args.models_dir),
            use_moge=args.use_moge,
        )
        Path(args.meta_out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return 0

    if args.cmd == "birefnet":
        manifest_path = Path(args.manifest)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = run_birefnet_matte_inprocess(
            payload["inputs"],
            out_dir=Path(args.out_dir),
            comfy_dir=Path(args.comfy_dir),
            models_dir=Path(args.models_dir),
        )
        payload["outputs"] = [str(p) for p in outputs]
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0

    if args.cmd == "remesh":
        meta = run_comfy_remesh_solid_inprocess(
            in_glb=Path(args.in_glb),
            out_glb=Path(args.out_glb),
            resolution=args.resolution,
            fill=args.fill,
            band=args.band,
            project_back=args.project_back,
            qef=bool(args.qef),
            drop_small_components=args.drop_small,
            smooth_iters=args.smooth_iters,
            fix_poles=args.fix_poles,
            precluster_max_verts=args.precluster_max_verts,
            comfy_dir=Path(args.comfy_dir),
            models_dir=Path(args.models_dir),
        )
        Path(args.meta_out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(_cli_main())
