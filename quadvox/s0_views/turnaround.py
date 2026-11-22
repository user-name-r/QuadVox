"""Generic S0 turnaround & smooth-porcelain conversion via OpenAI-compatible `/v1/images/edits`.

Keeps the user's primary front reference unmodified as Frame 0 (`view00_azim000.png`),
generates aligned 90/180/270-degree orthographic views for arbitrary objects/props/assets,
and provides a Direct3D-V3-aligned smooth matte porcelain conversion prompt builder.

Reference URLs:
- Direct3D-V3 official de-texture / clay prompt:
  https://github.com/DreamTechAI/Direct3D-V3/blob/main/assets/prompt.txt
- Upstream Pixal3D MV convention:
  https://github.com/TencentARC/Pixal3D/blob/master/inference_mv.py
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import requests
from PIL import Image
from scipy import ndimage

from quadvox.s0_views.preprocess import ForegroundBBox

DEFAULT_PROVIDERS_CONFIG = Path.home() / ".config" / "3dpipeline" / "image_providers.json"

# Incorporates Direct3D-V3 official `assets/prompt.txt` cues:
# smooth polished grain-free resin/porcelain + deeply recessed narrow crevices with near-black shadow floor
# + continuous shading across flat/curved surfaces.
D3DV3_PORCELAIN_PROMPT_BASE = (
    "Convert Image 1 into an unpainted, grain-free, pure monochrome neutral off-white "
    "smooth polished bisque porcelain / matte resin master sculpt (#DCDCE0 albedo, strict R=G=B grayscale) "
    "against a flat uniform mid-dark slate gray (#50545C) background. "
    "STRICT 1:1 ORTHOGRAPHIC SILHOUETTE LOCK: preserve exact pixel-aligned silhouette, proportions, "
    "and centered framing of Image 1 with zero perspective drift. "
    "SMOOTH CONTINUOUS SHADING & DEEP RECESS CONTRAST: flat and curved panels must have perfectly clean, "
    "continuous Lambertian shading free of clay grain, noise, tool marks, or specular hotspots; "
    "deeply recessed narrow channels, seams, and bevels must read with crisp dark-floor AO shadow contrast. "
    "Strip all 2D color, paint, decals, and ink outlines (represent surface seams only as clean low-relief bevels)."
)


def build_generic_turnaround_specs(subject_description: str = "the 3D asset") -> dict[str, dict[str, str]]:
    """Build category-agnostic 90/180/270 orthographic turnaround prompt specs."""
    subj = subject_description.strip() or "the 3D asset"
    return {
        "azim090": {
            "filename": "view01_azim090.png",
            "prompt": (
                f"Strictly straight-on 90-degree ORTHOGRAPHIC LEFT SIDE PROFILE studio render of {subj} "
                "from Image 1 on a solid flat uniform light gray (#DDDDDF) background. "
                "Camera is at +X looking strictly at the LEFT profile (front of subject faces LEFT). "
                "Match exact vertical top-to-bottom height, baseline alignment, silhouette proportions, "
                "and flat diffuse studio lighting of Image 1; zero floor shadow, zero text, zero extra props."
            ),
        },
        "azim180": {
            "filename": "view02_azim180.png",
            "prompt": (
                f"Strictly straight-on 180-degree ORTHOGRAPHIC BACK ELEVATION studio render of {subj} "
                "from Image 1 on a solid flat uniform light gray (#DDDDDF) background. "
                "Camera looks strictly at the back (-Z -> +Z). Match exact silhouette scale, "
                "top-to-bottom vertical height, symmetry/asymmetry from reference sheets, "
                "and flat diffuse studio lighting of Image 1; zero floor shadow, zero text, zero extra props."
            ),
        },
        "azim270": {
            "filename": "view03_azim270.png",
            "prompt": (
                f"Strictly straight-on 270-degree ORTHOGRAPHIC RIGHT SIDE PROFILE studio render of {subj} "
                "from Image 1 on a solid flat uniform light gray (#DDDDDF) background. "
                "Camera is at -X looking strictly at the RIGHT profile (front of subject faces RIGHT). "
                "Match exact vertical top-to-bottom height, baseline alignment, silhouette proportions, "
                "and flat diffuse studio lighting of Image 1; zero floor shadow, zero text, zero extra props."
            ),
        },
    }


def build_porcelain_prompt(
    view_label: str,
    *,
    subject_details: str = "",
    extra_directives: str = "",
) -> str:
    """Compose a generic D3D-V3-aligned smooth porcelain sculpt prompt for any view."""
    parts = [D3DV3_PORCELAIN_PROMPT_BASE, f"VIEWPOINT: PURE ORTHOGRAPHIC {view_label.upper()}."]
    if subject_details.strip():
        parts.append(subject_details.strip())
    if extra_directives.strip():
        parts.append(extra_directives.strip())
    return " ".join(parts)


def load_image_provider_routes(config_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load and priority-sort image provider routes without logging or leaking API keys."""
    env_cfg = os.environ.get("QUADVOX_IMAGE_PROVIDERS_CONFIG", "").strip()
    if config_path is not None:
        path = Path(config_path)
    elif env_cfg:
        path = Path(env_cfg).expanduser()
    else:
        path = DEFAULT_PROVIDERS_CONFIG

    if not path.exists():
        raise FileNotFoundError(f"Image providers config not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    routes = raw.get("routes", [])
    if not isinstance(routes, list) or not routes:
        raise ValueError(f"No routes configured in {path}")

    valid_routes: list[dict[str, Any]] = []
    for r in routes:
        if not isinstance(r, dict):
            continue
        if all(r.get(k) for k in ("base_url", "api_key", "model_name")):
            valid_routes.append(dict(r))

    if not valid_routes:
        raise ValueError(f"No valid routes with base_url/api_key/model_name in {path}")

    valid_routes.sort(key=lambda r: int(r.get("priority", 999)))
    return valid_routes


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _download_image_bytes(url: str, timeout: int = 120) -> bytes:
    # Do NOT forward any provider API key when fetching a signed output URL.
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def call_images_edits(
    prompt: str,
    ref_image_paths: Sequence[str | Path],
    routes: Sequence[dict[str, Any]],
    size: str = "1024x1024",
    quality: str = "high",
    timeout: int = 180,
) -> tuple[Image.Image, dict[str, str]]:
    """Call OpenAI-compatible POST `{base_url}/v1/images/edits` across priority routes.

    Never prints, logs, or includes `api_key` in returned metadata or exception text.
    """
    paths = [Path(p) for p in ref_image_paths]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Reference image not found: {p}")

    file_payloads = [(p.name, p.read_bytes(), _guess_mime(p)) for p in paths]
    errors: list[str] = []

    for route in routes:
        route_id = str(route.get("id") or route.get("name") or "unknown")
        base_url = str(route["base_url"]).rstrip("/")
        model_name = str(route["model_name"])
        api_key = str(route["api_key"])
        endpoint = f"{base_url}/v1/images/edits"

        files = [("image[]", payload) for payload in file_payloads]
        data = {
            "model": model_name,
            "prompt": prompt,
            "size": size,
            "quality": quality,
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            resp = requests.post(
                endpoint,
                headers=headers,
                data=data,
                files=files,
                timeout=timeout,
            )
            if resp.status_code >= 400:
                snippet = resp.text[:240].replace(api_key, "***")
                errors.append(f"{route_id}: HTTP {resp.status_code} ({snippet})")
                continue

            payload = resp.json()
            items = payload.get("data") or []
            if not items:
                errors.append(f"{route_id}: empty data field in JSON response")
                continue

            item = items[0]
            if item.get("b64_json"):
                img_bytes = base64.b64decode(item["b64_json"])
            elif item.get("url"):
                img_bytes = _download_image_bytes(str(item["url"]), timeout=timeout)
            else:
                errors.append(f"{route_id}: missing b64_json/url in data[0]")
                continue

            with Image.open(io.BytesIO(img_bytes)) as im:
                im.load()
                rgb = im.convert("RGB")
            return rgb, {"route_id": route_id, "model_name": model_name, "base_url": base_url}

        except Exception as exc:
            safe_msg = str(exc).replace(api_key, "***")
            errors.append(f"{route_id}: {type(exc).__name__}: {safe_msg}")

    raise RuntimeError("All image edit routes failed: " + "; ".join(errors))


def estimate_border_bg_rgb(image: Image.Image, border_px: int = 12) -> tuple[int, int, int]:
    """Estimate flat background RGB from the outer border strip of an RGB image."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w, _ = arr.shape
    b = max(1, min(border_px, h // 8, w // 8))
    samples = np.concatenate(
        [
            arr[:b, :, :].reshape(-1, 3),
            arr[-b:, :, :].reshape(-1, 3),
            arr[:, :b, :].reshape(-1, 3),
            arr[:, -b:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    med = np.median(samples, axis=0)
    return (int(round(med[0])), int(round(med[1])), int(round(med[2])))


def measure_foreground_bbox(
    image: Image.Image,
    threshold: int = 16,
    min_component_pixels: int = 64,
) -> tuple[ForegroundBBox, tuple[int, int, int]]:
    """Measure foreground bbox on a flat-background RGB image."""
    rgb = image.convert("RGB")
    bg_rgb = estimate_border_bg_rgb(rgb)
    arr = np.asarray(rgb, dtype=np.int16)
    bg_vec = np.asarray(bg_rgb, dtype=np.int16)
    diff = np.abs(arr - bg_vec).max(axis=-1)
    mask = diff > int(threshold)

    labeled, num_features = ndimage.label(mask)
    if num_features > 0:
        counts = np.bincount(labeled.ravel())
        keep = counts >= int(min_component_pixels)
        keep[0] = False
        cleaned = keep[labeled]
        if np.any(cleaned):
            mask = cleaned

    if not np.any(mask):
        raise ValueError("Failed to detect foreground subject against background")

    ys, xs = np.where(mask)
    bbox = ForegroundBBox(
        xmin=int(xs.min()),
        ymin=int(ys.min()),
        xmax=int(xs.max()) + 1,
        ymax=int(ys.max()) + 1,
    )
    return bbox, bg_rgb


def align_generated_view_to_reference(
    generated: Image.Image,
    ref_bbox: ForegroundBBox,
    canvas_size: tuple[int, int] = (1024, 1024),
    threshold: int = 16,
) -> tuple[Image.Image, dict[str, Any]]:
    """Uniformly scale and align a generated RGB view so vertical span and baseline match Frame 0."""
    gen_rgb = generated.convert("RGB")
    gen_bbox, bg_rgb = measure_foreground_bbox(gen_rgb, threshold=threshold)

    scale = float(ref_bbox.height) / float(gen_bbox.height)
    new_w = max(1, int(round(float(gen_bbox.width) * scale)))
    new_h = int(ref_bbox.height)

    target_w, target_h = int(canvas_size[0]), int(canvas_size[1])
    if new_w > target_w or new_h > target_h:
        clamp_scale = min(float(target_w) / float(gen_bbox.width), float(target_h) / float(gen_bbox.height))
        scale = min(scale, clamp_scale)
        new_w = max(1, int(round(float(gen_bbox.width) * scale)))
        new_h = max(1, int(round(float(gen_bbox.height) * scale)))

    crop = gen_rgb.crop((gen_bbox.xmin, gen_bbox.ymin, gen_bbox.xmax, gen_bbox.ymax))
    resized = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

    ref_cx = (float(ref_bbox.xmin) + float(ref_bbox.xmax)) / 2.0
    paste_x = int(round(ref_cx - float(new_w) / 2.0))
    paste_y = int(ref_bbox.ymax) - new_h
    paste_x = max(0, min(target_w - new_w, paste_x))
    paste_y = max(0, min(target_h - new_h, paste_y))

    canvas = Image.new("RGB", (target_w, target_h), bg_rgb)
    canvas.paste(resized, (paste_x, paste_y))

    aligned_bbox, _ = measure_foreground_bbox(canvas, threshold=threshold)
    stats = {
        "raw_bbox": [gen_bbox.xmin, gen_bbox.ymin, gen_bbox.xmax, gen_bbox.ymax],
        "aligned_bbox": [aligned_bbox.xmin, aligned_bbox.ymin, aligned_bbox.xmax, aligned_bbox.ymax],
        "scale_applied": round(scale, 5),
        "bg_rgb": list(bg_rgb),
    }
    return canvas, stats


def generate_turnaround_views(
    front_image: str | Path,
    out_dir: str | Path,
    *,
    sheet_image: str | Path | None = None,
    subject_description: str = "the 3D asset",
    view_specs: Mapping[str, Mapping[str, str]] | None = None,
    views: Sequence[str] | None = None,
    config_path: str | Path | None = None,
    align_to_front: bool = True,
) -> dict[str, Any]:
    """Run generic S0 turnaround generation into `out_dir`, keeping `front_image` unmodified as Frame 0."""
    front_path = Path(front_image)
    if not front_path.exists():
        raise FileNotFoundError(f"Front reference image not found: {front_path}")

    ref_images: list[Path] = [front_path]
    if sheet_image is not None:
        sp = Path(sheet_image)
        if not sp.exists():
            raise FileNotFoundError(f"Reference sheet image not found: {sp}")
        ref_images.append(sp)

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    frame0_path = dest / "view00_azim000.png"
    shutil.copy2(front_path, frame0_path)

    with Image.open(frame0_path) as im0:
        im0.load()
        frame0_size = im0.size
        ref_bbox, ref_bg = measure_foreground_bbox(im0)

    routes = load_image_provider_routes(config_path)
    specs = view_specs if view_specs is not None else build_generic_turnaround_specs(subject_description)
    target_views = list(views) if views is not None else ["azim090", "azim180", "azim270"]

    report_path = dest / "s0_report.json"
    existing_generated: dict[str, Any] = {}
    if report_path.exists():
        try:
            existing_generated = dict(json.loads(report_path.read_text(encoding="utf-8")).get("generated", {}))
        except Exception:
            existing_generated = {}

    report: dict[str, Any] = {
        "front_source": str(front_path),
        "sheet_source": str(sheet_image) if sheet_image is not None else None,
        "out_dir": str(dest),
        "frame0_byte_identical": frame0_path.read_bytes() == front_path.read_bytes(),
        "frame0_bbox": [ref_bbox.xmin, ref_bbox.ymin, ref_bbox.xmax, ref_bbox.ymax],
        "frame0_bg_rgb": list(ref_bg),
        "generated": existing_generated,
    }

    for view_key in target_views:
        if view_key not in specs:
            raise ValueError(f"Unknown S0 view key {view_key!r}; expected one of {list(specs)}")
        spec = specs[view_key]
        out_path = dest / spec["filename"]

        raw_img, route_meta = call_images_edits(
            prompt=spec["prompt"],
            ref_image_paths=ref_images,
            routes=routes,
            size="1024x1024",
            quality="high",
        )

        if align_to_front:
            final_img, align_stats = align_generated_view_to_reference(
                raw_img,
                ref_bbox=ref_bbox,
                canvas_size=frame0_size,
            )
        else:
            final_img = raw_img.resize(frame0_size, Image.Resampling.LANCZOS) if raw_img.size != frame0_size else raw_img
            bbox_only, bg_only = measure_foreground_bbox(final_img)
            align_stats = {
                "raw_bbox": [bbox_only.xmin, bbox_only.ymin, bbox_only.xmax, bbox_only.ymax],
                "aligned_bbox": [bbox_only.xmin, bbox_only.ymin, bbox_only.xmax, bbox_only.ymax],
                "scale_applied": 1.0,
                "bg_rgb": list(bg_only),
            }

        final_img.save(out_path, format="PNG")
        report["generated"][view_key] = {
            "path": str(out_path),
            "route": route_meta,
            **align_stats,
        }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
