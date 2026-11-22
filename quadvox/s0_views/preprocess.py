"""Deterministic S1 single-view and multi-view preprocessing for Pixal3D.

Produces Z-up Blender/NeRF `transforms.json` and uniformly scaled, foot/head-aligned
1024x1024 RGBA views matching upstream Pixal3D and ComfyUI conventions.

Reference URLs:
- Upstream Pixal3D: https://github.com/TencentARC/Pixal3D
- Upstream inference_mv.py: https://github.com/TencentARC/Pixal3D/blob/master/inference_mv.py
- ComfyUI Pixal3D docs: https://github.com/Comfy-Org/docs/blob/main/tutorials/3d/pixal3d.mdx
- ComfyUI nodes_trellis2.py: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_trellis2.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image


DEFAULT_VIEW_SIZE = 1024
DEFAULT_FOV_DEG = 20.0
DEFAULT_PAD_FACTOR = 1.1
DEFAULT_AZIMUTHS_DEG = (0.0, 90.0, 180.0, 270.0)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ForegroundBBox:
    """Inclusive-exclusive foreground bounding box within an image."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin


def _snap_zero(value: float, eps: float = 1e-12) -> float:
    return 0.0 if abs(value) < eps else float(value)


def orbit_distance_from_fov(fov_deg: float = DEFAULT_FOV_DEG, pad_factor: float = DEFAULT_PAD_FACTOR) -> float:
    """Compute canonical orbit camera distance matching ComfyUI Pixal3DMultiViewConditioning.

    Formula: distance = pad_factor * 0.5 / tan(radians(fov_deg) / 2.0)
    At fov_deg=20.0 and pad_factor=1.1, distance == 3.1192050007897403 (~3.1192).
    """
    fov_rad = math.radians(float(fov_deg))
    if not (0.0 < fov_rad < math.pi):
        raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")
    if float(pad_factor) <= 0.0:
        raise ValueError(f"pad_factor must be positive, got {pad_factor}")
    return float(pad_factor) * 0.5 / math.tan(fov_rad / 2.0)


def orbit_c2w_matrix(azimuth_deg: float, elevation_deg: float, distance: float) -> list[list[float]]:
    """Build a 4x4 Blender/NeRF Z-up camera-to-world matrix looking at the origin.

    Matches `_orbit_camera_to_world` in ComfyUI `nodes_trellis2.py` and
    `TencentARC/Pixal3D/inference_mv.py`:
    - Frame 0 (azimuth=0, elevation=0) sits at (0, -distance, 0) looking +Y with world +Z up.
    - Azimuth +90 sits at (+distance, 0, 0) looking -X (character's left profile).
    - Azimuth 180 sits at (0, +distance, 0) looking -Y (back view).
    - Azimuth 270 sits at (-distance, 0, 0) looking +X (character's right profile).
    """
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))

    back = np.array(
        [
            math.sin(az) * math.cos(el),
            -math.cos(az) * math.cos(el),
            math.sin(el),
        ],
        dtype=np.float64,
    )
    right = np.array(
        [
            math.cos(az),
            math.sin(az),
            0.0,
        ],
        dtype=np.float64,
    )
    up = np.cross(back, right)

    mat = np.eye(4, dtype=np.float64)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = back
    mat[:3, 3] = back * float(distance)

    return [[_snap_zero(float(mat[r, c])) for c in range(4)] for r in range(4)]


def build_orbit_transforms(
    file_paths: Sequence[str | Path],
    fov_deg: float = DEFAULT_FOV_DEG,
    pad_factor: float = DEFAULT_PAD_FACTOR,
    mesh_scale: float = 1.0,
    azimuths_deg: Sequence[float] | None = None,
    elevations_deg: Sequence[float] | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build exact Z-up Blender/NeRF `transforms.json` dict for Pixal3D MV/SV.

    Follows https://github.com/TencentARC/Pixal3D/blob/master/inference_mv.py and
    `Pixal3DMultiViewConditioning` in https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_trellis2.py.
    """
    if not file_paths:
        raise ValueError("file_paths must contain at least one view path")

    num_views = len(file_paths)
    if azimuths_deg is None:
        if num_views <= len(DEFAULT_AZIMUTHS_DEG):
            resolved_az = list(DEFAULT_AZIMUTHS_DEG[:num_views])
        else:
            resolved_az = [i * (360.0 / num_views) for i in range(num_views)]
    else:
        resolved_az = [float(a) for a in azimuths_deg]
        if len(resolved_az) != num_views:
            raise ValueError(
                f"azimuths_deg length ({len(resolved_az)}) must match file_paths length ({num_views})"
            )

    if elevations_deg is None:
        resolved_el = [0.0] * num_views
    else:
        resolved_el = [float(e) for e in elevations_deg]
        if len(resolved_el) != num_views:
            raise ValueError(
                f"elevations_deg length ({len(resolved_el)}) must match file_paths length ({num_views})"
            )

    camera_angle_x = math.radians(float(fov_deg))
    distance = orbit_distance_from_fov(fov_deg=fov_deg, pad_factor=pad_factor)

    frames: list[dict[str, Any]] = []
    base_path = Path(base_dir).resolve() if base_dir is not None else None
    for raw_path, az, el in zip(file_paths, resolved_az, resolved_el):
        p = Path(raw_path)
        if base_path is not None:
            try:
                rel_str = p.resolve().relative_to(base_path).as_posix()
            except ValueError:
                rel_str = p.name
        elif p.is_absolute():
            rel_str = p.name
        else:
            rel_str = p.as_posix()

        az_label = int(round(az)) % 360
        frames.append(
            {
                "file_path": rel_str,
                "name": f"azim{az_label:03d}",
                "transform_matrix": orbit_c2w_matrix(az, el, distance),
            }
        )

    return {
        "camera_angle_x": float(camera_angle_x),
        "mesh_scale": float(mesh_scale),
        "frames": frames,
    }


def _open_image(item: Image.Image | str | Path) -> tuple[Image.Image, str]:
    if isinstance(item, Image.Image):
        return item, "<Image>"
    path = Path(item)
    if not path.exists():
        raise FileNotFoundError(f"View image not found: {path}")
    with Image.open(path) as im:
        im.load()
        return im.copy(), str(path)


def validate_rgba_alpha(
    image: Image.Image | str | Path,
    matte_fn: Callable[[Image.Image], Image.Image] | None = None,
    alpha_threshold: int = 8,
) -> Image.Image:
    """Validate that an input view carries non-trivial alpha, or apply `matte_fn`.

    Matches `inference_mv.py:load_rgba`: a fully opaque alpha channel (all 255)
    or an RGB image without alpha fails explicitly unless `matte_fn` is provided.
    """
    im, hint = _open_image(image)
    if im.mode in ("RGBA", "LA", "PA"):
        rgba = im.convert("RGBA")
        alpha = np.asarray(rgba.getchannel(3), dtype=np.uint8)
        if not np.all(alpha == 255):
            if not np.any(alpha > alpha_threshold) and not np.any(alpha > 0):
                raise ValueError(f"View {hint} has an empty alpha mask (all zeros)")
            return rgba

    if matte_fn is None:
        raise ValueError(
            f"View {hint} lacks a non-trivial alpha mask (mode={im.mode}); "
            "provide pre-matted RGBA images or a matte_fn (e.g. BiRefNet)."
        )

    matted = matte_fn(im.convert("RGB")).convert("RGBA")
    matted_alpha = np.asarray(matted.getchannel(3), dtype=np.uint8)
    if np.all(matted_alpha == 255) or not np.any(matted_alpha > 0):
        raise ValueError(f"matte_fn failed to produce a valid foreground alpha mask for {hint}")
    return matted


def foreground_bbox(rgba: Image.Image, alpha_threshold: int = 8) -> ForegroundBBox:
    """Compute tight foreground bounding box `[xmin, ymin, xmax, ymax]` from RGBA alpha."""
    if rgba.mode != "RGBA":
        raise ValueError(f"Expected RGBA image, got {rgba.mode}")
    alpha = np.asarray(rgba.getchannel(3), dtype=np.uint8)
    mask = alpha >= alpha_threshold
    if not np.any(mask):
        mask = alpha > 0
    if not np.any(mask):
        raise ValueError("Cannot compute foreground bbox on empty alpha mask")
    ys, xs = np.where(mask)
    return ForegroundBBox(
        xmin=int(xs.min()),
        ymin=int(ys.min()),
        xmax=int(xs.max()) + 1,
        ymax=int(ys.max()) + 1,
    )


def _zero_transparent_rgb(rgba: Image.Image) -> Image.Image:
    arr = np.asarray(rgba, dtype=np.uint8).copy()
    arr[arr[:, :, 3] == 0, :3] = 0
    return Image.fromarray(arr, mode="RGBA")


def align_multiview_rgba(
    rgba_images: Sequence[Image.Image | str | Path],
    size: int = DEFAULT_VIEW_SIZE,
    pad_factor: float = DEFAULT_PAD_FACTOR,
    matte_fn: Callable[[Image.Image], Image.Image] | None = None,
    alpha_threshold: int = 8,
) -> list[Image.Image]:
    """Normalize single-view or multi-view RGBA frames onto `size x size` canvases.

    Computes a SINGLE shared uniform scale factor from global/reference bounds
    anchored to Frame 0's head-to-foot height and global max span (`target_span = size / pad_factor`).
    Never rescales individual views independently; aligns foot bottom and head top
    horizontally across all views on transparent RGBA.
    """
    if not rgba_images:
        raise ValueError("rgba_images must not be empty")
    if int(size) <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if float(pad_factor) <= 1.0:
        raise ValueError(f"pad_factor must be > 1.0, got {pad_factor}")

    validated = [
        validate_rgba_alpha(item, matte_fn=matte_fn, alpha_threshold=alpha_threshold)
        for item in rgba_images
    ]
    bboxes = [foreground_bbox(im, alpha_threshold=alpha_threshold) for im in validated]

    ref_h = float(bboxes[0].height)
    max_w = max(float(b.width) for b in bboxes)
    max_h = max(float(b.height) for b in bboxes)
    global_span = max(max_w, max_h, ref_h)
    target_span = float(size) / float(pad_factor)
    scale = target_span / global_span

    # Anchor shared foot-bottom baseline so Frame 0 (and any equal-height views)
    # sit centered vertically while guaranteeing no view overflows top/bottom.
    ref_scaled_h = int(round(max_h * scale))
    shared_foot_y = int(round((float(size) + float(ref_scaled_h)) / 2.0))

    aligned: list[Image.Image] = []
    for im, bbox in zip(validated, bboxes):
        crop = _zero_transparent_rgb(im.crop((bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)))
        new_w = max(1, int(round(float(bbox.width) * scale)))
        new_h = max(1, int(round(float(bbox.height) * scale)))
        resized = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

        paste_x = int(round((float(size) - float(new_w)) / 2.0))
        paste_y = shared_foot_y - new_h
        paste_x = max(0, min(int(size) - new_w, paste_x))
        paste_y = max(0, min(int(size) - new_h, paste_y))

        canvas = Image.new("RGBA", (int(size), int(size)), (0, 0, 0, 0))
        canvas.paste(resized, (paste_x, paste_y), resized)
        aligned.append(_zero_transparent_rgb(canvas))

    return aligned


def align_single_view_rgba(
    rgba_image: Image.Image | str | Path,
    size: int = DEFAULT_VIEW_SIZE,
    pad_factor: float = DEFAULT_PAD_FACTOR,
    matte_fn: Callable[[Image.Image], Image.Image] | None = None,
    alpha_threshold: int = 8,
) -> Image.Image:
    """Normalize a single RGBA image onto a `size x size` transparent canvas."""
    return align_multiview_rgba(
        [rgba_image],
        size=size,
        pad_factor=pad_factor,
        matte_fn=matte_fn,
        alpha_threshold=alpha_threshold,
    )[0]


def discover_input_views(input_path: str | Path, mode: str = "mv") -> list[Path]:
    """Discover ordered input view files from a file path or directory."""
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input path does not exist: {src}")

    if src.is_file():
        return [src]

    transforms_file = src / "transforms.json"
    if transforms_file.exists():
        meta = json.loads(transforms_file.read_text(encoding="utf-8"))
        frames = meta.get("frames", [])
        if frames:
            resolved = [src / fr["file_path"] for fr in frames]
            missing = [str(p) for p in resolved if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing frame(s) referenced in {transforms_file}: {missing}")
            return resolved[:1] if mode == "sv" else resolved

    canonical_names = [
        "view00_azim000.png",
        "view01_azim090.png",
        "view02_azim180.png",
        "view03_azim270.png",
    ]
    if all((src / name).exists() for name in canonical_names):
        views = [src / name for name in canonical_names]
        return views[:1] if mode == "sv" else views

    candidates = sorted(
        p for p in src.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"No view images found in {src}")
    return candidates[:1] if mode == "sv" else candidates


def prepare_views_dir(
    input_path: str | Path | Sequence[str | Path],
    out_dir: str | Path,
    mode: str = "mv",
    size: int = DEFAULT_VIEW_SIZE,
    fov_deg: float = DEFAULT_FOV_DEG,
    pad_factor: float = DEFAULT_PAD_FACTOR,
    mesh_scale: float = 1.0,
    matte_fn: Callable[[Image.Image], Image.Image] | None = None,
) -> dict[str, Any]:
    """Preprocess SV or MV input views into aligned 1024x1024 RGBA PNGs + transforms.json."""
    norm_mode = str(mode).lower()
    if norm_mode not in ("sv", "mv"):
        raise ValueError(f"Unsupported mode {mode!r}; expected 'sv' or 'mv'")

    if isinstance(input_path, (str, Path)):
        raw_paths = discover_input_views(input_path, mode=norm_mode)
    else:
        raw_paths = [Path(p) for p in input_path]
        if not raw_paths:
            raise ValueError("input_path sequence must not be empty")
        if norm_mode == "sv":
            raw_paths = raw_paths[:1]

    aligned = align_multiview_rgba(
        raw_paths,
        size=size,
        pad_factor=pad_factor,
        matte_fn=matte_fn,
    )

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    azimuths = (
        list(DEFAULT_AZIMUTHS_DEG[: len(aligned)])
        if len(aligned) <= len(DEFAULT_AZIMUTHS_DEG)
        else [i * (360.0 / len(aligned)) for i in range(len(aligned))]
    )
    out_files: list[Path] = []
    for idx, (im, az) in enumerate(zip(aligned, azimuths)):
        az_label = int(round(az)) % 360
        out_file = dest / f"view{idx:02d}_azim{az_label:03d}.png"
        im.save(out_file, format="PNG")
        out_files.append(out_file)

    transforms = build_orbit_transforms(
        file_paths=[f.name for f in out_files],
        fov_deg=fov_deg,
        pad_factor=pad_factor,
        mesh_scale=mesh_scale,
        azimuths_deg=azimuths,
    )
    transforms_path = dest / "transforms.json"
    transforms_path.write_text(json.dumps(transforms, indent=2) + "\n", encoding="utf-8")

    return {
        "mode": norm_mode,
        "out_dir": str(dest),
        "transforms_path": str(transforms_path),
        "file_paths": [str(f) for f in out_files],
        "transforms": transforms,
    }
