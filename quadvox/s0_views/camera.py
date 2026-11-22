"""Generic S0 foreground matting, orthographic camera calibration, Z-buffer, and silhouette alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class MattedView:
    """Preprocessed S0 or auxiliary view with clean dilated foreground RGB and soft alpha."""

    name: str
    rgb_orig: np.ndarray        # float32 (H, W, 3) in [0, 1]
    rgb_dilated: np.ndarray     # float32 (H, W, 3) in [0, 1], rim-eroded & EDT-dilated
    alpha_soft: np.ndarray      # float32 (H, W) in [0, 1]
    fg_mask: np.ndarray         # bool (H, W)
    bg_rgb: tuple[float, float, float]


@dataclass
class OrthoCameraSpec:
    """Calibrated orthographic camera in glTF (+Y up, +Z front, +X right) coordinates."""

    name: str
    azim_deg: float
    elev_deg: float
    width: int
    height: int
    cam_pos_dir: np.ndarray     # (3,) unit vector from origin toward camera
    cam_right: np.ndarray       # (3,) unit vector (+u image col)
    cam_up: np.ndarray          # (3,) unit vector (-v image row)
    cx_px: float
    cy_px: float
    sx_px: float                # pixels per world unit horizontally
    sy_px: float                # pixels per world unit vertically
    trust: float = 1.0          # 1.0 for primary S0 views; lower for auxiliary views
    is_primary: bool = True


def dilate_nearest_texel(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Fill every invalid pixel/texel with the nearest valid pixel/texel via exact EDT."""
    if np.all(valid_mask) or not np.any(valid_mask):
        return arr.copy()
    _, (iy, ix) = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
    return arr[iy, ix]


def matte_s0_view(
    image: Image.Image | np.ndarray | str | Path,
    *,
    name: str = "view",
    bg_tol: float = 14.0,
    feather_tol: float = 26.0,
    rim_erode_px: int = 3,
) -> MattedView:
    """Extract clean foreground alpha from RGB/RGBA view, drop outer rim halo, and EDT-dilate RGB."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            im.load()
            arr = np.asarray(im)
    elif isinstance(image, Image.Image):
        arr = np.asarray(image)
    else:
        arr = np.asarray(image)

    has_alpha = arr.ndim == 3 and arr.shape[2] == 4
    if has_alpha:
        alpha_u8 = arr[:, :, 3]
        if np.any(alpha_u8 < 16) and np.any(alpha_u8 > 240):
            rgb_f = arr[:, :, :3].astype(np.float32) / 255.0
            alpha_soft = alpha_u8.astype(np.float32) / 255.0
            fg_mask = alpha_soft >= 0.5
            bg_rgb = (0.0, 0.0, 0.0)
            return _finalize_matted_view(name, rgb_f, alpha_soft, fg_mask, bg_rgb, rim_erode_px)

    rgb_u8 = arr[:, :, :3] if arr.ndim == 3 else np.repeat(arr[:, :, None], 3, axis=2)
    rgb_f = rgb_u8.astype(np.float32) / 255.0
    rgb_255 = rgb_u8.astype(np.float32)

    border = np.concatenate([rgb_255[0, :], rgb_255[-1, :], rgb_255[:, 0], rgb_255[:, -1]], axis=0)
    bg_255 = np.median(border, axis=0)
    dist = np.linalg.norm(rgb_255 - bg_255[None, None, :], axis=-1)

    cand_bg = dist <= float(bg_tol)
    comp, _ = ndimage.label(cand_bg)
    border_labels = set(np.unique(np.concatenate([comp[0, :], comp[-1, :], comp[:, 0], comp[:, -1]]))) - {0}
    bg_mask = np.isin(comp, list(border_labels))

    # Also detect enclosed studio backdrop pockets (e.g. narrow interior gaps).
    pocket_cand = (~bg_mask) & (dist <= min(6.5, 0.55 * float(bg_tol)))
    pockets, npk = ndimage.label(pocket_cand)
    for pk in range(1, npk + 1):
        pm = pockets == pk
        if int(np.sum(pm)) >= 24 and float(np.mean(dist[pm])) <= 4.2:
            if float(np.max(np.std(rgb_255[pm], axis=0))) <= 3.5:
                bg_mask |= pm

    fg_raw = ~bg_mask
    fg_lbl, n_fg = ndimage.label(fg_raw)
    if n_fg > 1:
        counts = np.bincount(fg_lbl.ravel())
        counts[0] = 0
        largest = int(counts.max())
        keep = counts >= max(32, int(0.001 * largest))
        keep[0] = False
        fg_raw = keep[fg_lbl]
        bg_mask = ~fg_raw

    edge_band = ndimage.binary_dilation(bg_mask, iterations=2) & ~bg_mask & (dist <= float(feather_tol))
    alpha_soft = np.where(bg_mask, 0.0, 1.0).astype(np.float32)
    if np.any(edge_band):
        alpha_soft[edge_band] = np.clip(
            (dist[edge_band] - float(bg_tol)) / max(float(feather_tol - bg_tol), 1e-3), 0.0, 1.0
        )
    fg_mask = alpha_soft >= 0.5
    bg_rgb = tuple(round(float(x / 255.0), 4) for x in bg_255)
    return _finalize_matted_view(name, rgb_f, alpha_soft, fg_mask, bg_rgb, rim_erode_px)


def _finalize_matted_view(
    name: str,
    rgb_f: np.ndarray,
    alpha_soft: np.ndarray,
    fg_mask: np.ndarray,
    bg_rgb: tuple[float, float, float],
    rim_erode_px: int,
) -> MattedView:
    """Drop outer 2-3px silhouette rim before EDT foreground color dilation."""
    if not np.any(fg_mask):
        return MattedView(name, rgb_f, rgb_f.copy(), alpha_soft, fg_mask, bg_rgb)

    if int(rim_erode_px) > 0:
        core_fg = ndimage.binary_erosion(fg_mask, iterations=int(rim_erode_px))
        if not np.any(core_fg):
            core_fg = fg_mask
    else:
        core_fg = fg_mask

    rgb_dilated = dilate_nearest_texel(rgb_f, core_fg)
    return MattedView(
        name=name,
        rgb_orig=rgb_f,
        rgb_dilated=rgb_dilated,
        alpha_soft=alpha_soft,
        fg_mask=fg_mask,
        bg_rgb=bg_rgb,
    )


def build_view_basis(azim_deg: float, elev_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(cam_pos_dir, cam_right, cam_up)` in glTF +Y-up/+Z-front coords."""
    az = math.radians(float(azim_deg))
    el = math.radians(float(elev_deg))
    cam_pos_dir = np.array(
        [math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)],
        dtype=np.float64,
    )
    cam_pos_dir /= max(float(np.linalg.norm(cam_pos_dir)), 1e-12)
    cam_forward = -cam_pos_dir
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(cam_forward, world_up))) > 0.99:
        world_up = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    cam_right = np.cross(cam_forward, world_up)
    cam_right /= max(float(np.linalg.norm(cam_right)), 1e-12)
    cam_up = np.cross(cam_right, cam_forward)
    cam_up /= max(float(np.linalg.norm(cam_up)), 1e-12)
    return cam_pos_dir.astype(np.float32), cam_right.astype(np.float32), cam_up.astype(np.float32)


def project_points_ortho(pts: np.ndarray, cam: OrthoCameraSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D points `[..., 3]` to `(px_x, px_y, depth)` under calibrated ortho camera."""
    u_world = pts @ cam.cam_right
    v_world = pts @ cam.cam_up
    depth = pts @ cam.cam_pos_dir
    px_x = cam.cx_px + u_world * cam.sx_px
    px_y = cam.cy_px - v_world * cam.sy_px
    return px_x, px_y, depth


def render_mesh_ortho_zbuf(
    verts: np.ndarray,
    faces: np.ndarray,
    cam: OrthoCameraSpec,
    *,
    face_labels: np.ndarray | None = None,
    vnorms: np.ndarray | None = None,
    render_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Rasterize exact orthographic depth, hit mask, normals, and optional semantic class ID."""
    W = max(16, int(round(cam.width * render_scale)))
    H = max(16, int(round(cam.height * render_scale)))
    px_x, px_y, depth = project_points_ortho(verts, cam)
    px_x = px_x * render_scale
    px_y = px_y * render_scale

    p0x, p1x, p2x = px_x[faces[:, 0]], px_x[faces[:, 1]], px_x[faces[:, 2]]
    p0y, p1y, p2y = px_y[faces[:, 0]], px_y[faces[:, 1]], px_y[faces[:, 2]]
    d0, d1, d2 = depth[faces[:, 0]], depth[faces[:, 1]], depth[faces[:, 2]]

    denoms = (p1y - p2y) * (p0x - p2x) + (p2x - p1x) * (p0y - p2y)
    min_x = np.clip(np.floor(np.minimum(np.minimum(p0x, p1x), p2x)).astype(np.int32), 0, W - 1)
    max_x = np.clip(np.ceil(np.maximum(np.maximum(p0x, p1x), p2x)).astype(np.int32), 0, W - 1)
    min_y = np.clip(np.floor(np.minimum(np.minimum(p0y, p1y), p2y)).astype(np.int32), 0, H - 1)
    max_y = np.clip(np.ceil(np.maximum(np.maximum(p0y, p1y), p2y)).astype(np.int32), 0, H - 1)

    zbuf = np.full((H, W), -1e9, dtype=np.float32)
    hit_face = np.full((H, W), -1, dtype=np.int32)
    hit_cls = np.zeros((H, W), dtype=np.uint8)
    hit_nrm = np.zeros((H, W, 3), dtype=np.float32)

    eps = -1e-4
    active = np.flatnonzero(np.abs(denoms) > 1e-6)
    for fi in active:
        x0, x1 = int(min_x[fi]), int(max_x[fi])
        y0, y1 = int(min_y[fi]), int(max_y[fi])
        if x1 < x0 or y1 < y0:
            continue
        ax, ay = float(p0x[fi]), float(p0y[fi])
        bx, by = float(p1x[fi]), float(p1y[fi])
        cx, cy = float(p2x[fi]), float(p2y[fi])
        inv_d = 1.0 / float(denoms[fi])

        xs = np.arange(x0, x1 + 1, dtype=np.float64)[None, :]
        ys = np.arange(y0, y1 + 1, dtype=np.float64)[:, None]
        w0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) * inv_d
        w1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) * inv_d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not np.any(inside):
            continue

        ry, rx = np.where(inside)
        gy = y0 + ry
        gx = x0 + rx
        bw0 = w0[ry, rx]
        bw1 = w1[ry, rx]
        bw2 = w2[ry, rx]
        zd = (bw0 * float(d0[fi]) + bw1 * float(d1[fi]) + bw2 * float(d2[fi])).astype(np.float32)

        better = zd > zbuf[gy, gx]
        if not np.any(better):
            continue
        gy = gy[better]
        gx = gx[better]
        zbuf[gy, gx] = zd[better]
        hit_face[gy, gx] = fi
        if face_labels is not None:
            hit_cls[gy, gx] = int(face_labels[fi])
        if vnorms is not None:
            i0, i1, i2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            n_i = (
                bw0[better, None].astype(np.float32) * vnorms[i0]
                + bw1[better, None].astype(np.float32) * vnorms[i1]
                + bw2[better, None].astype(np.float32) * vnorms[i2]
            )
            n_i /= np.maximum(np.linalg.norm(n_i, axis=1, keepdims=True), 1e-8)
            hit_nrm[gy, gx] = n_i

    hit_mask = hit_face >= 0
    return {
        "zbuf": zbuf,
        "hit_mask": hit_mask,
        "hit_face": hit_face,
        "hit_cls": hit_cls,
        "hit_nrm": hit_nrm,
    }


def calibrate_s0_ortho_camera(
    verts: np.ndarray,
    faces: np.ndarray,
    fg_mask: np.ndarray,
    *,
    name: str,
    azim_deg: float,
    elev_deg: float = 0.0,
    trust: float = 1.0,
    is_primary: bool = True,
    allow_anisotropic_xy: bool = True,
) -> tuple[OrthoCameraSpec, dict[str, float]]:
    """Fit orthographic camera `(cx, cy, sx, sy)` aligning mesh silhouette onto S0 `fg_mask`."""
    H, W = fg_mask.shape
    ys, xs = np.where(fg_mask)
    if len(xs) == 0:
        raise ValueError(f"Empty foreground mask when calibrating camera {name}")

    img_x0, img_x1 = float(xs.min()), float(xs.max())
    img_y0, img_y1 = float(ys.min()), float(ys.max())
    img_w = max(img_x1 - img_x0, 1.0)
    img_h = max(img_y1 - img_y0, 1.0)
    img_cx = 0.5 * (img_x0 + img_x1)
    img_cy = 0.5 * (img_y0 + img_y1)

    cam_pos_dir, cam_right, cam_up = build_view_basis(azim_deg, elev_deg)
    u_w = verts @ cam_right
    v_w = verts @ cam_up
    u_min, u_max = float(u_w.min()), float(u_w.max())
    v_min, v_max = float(v_w.min()), float(v_w.max())
    u_mid = 0.5 * (u_min + u_max)
    v_mid = 0.5 * (v_min + v_max)
    mesh_w = max(u_max - u_min, 1e-6)
    mesh_h = max(v_max - v_min, 1e-6)

    sy_init = img_h / mesh_h
    sx_init = (img_w / mesh_w) if allow_anisotropic_xy else sy_init
    cx_init = img_cx - u_mid * sx_init
    cy_init = img_cy + v_mid * sy_init

    best = OrthoCameraSpec(
        name=name,
        azim_deg=float(azim_deg),
        elev_deg=float(elev_deg),
        width=W,
        height=H,
        cam_pos_dir=cam_pos_dir,
        cam_right=cam_right,
        cam_up=cam_up,
        cx_px=cx_init,
        cy_px=cy_init,
        sx_px=sx_init,
        sy_px=sy_init,
        trust=float(trust),
        is_primary=bool(is_primary),
    )

    r_scale = min(1.0, 192.0 / float(max(H, W)))
    rw, rh = max(32, int(round(W * r_scale))), max(32, int(round(H * r_scale)))
    fg_ds = np.asarray(
        Image.fromarray((fg_mask.astype(np.uint8) * 255), mode="L").resize((rw, rh), Image.Resampling.NEAREST)
    ) > 127

    def _eval_iou(spec: OrthoCameraSpec) -> float:
        buf = render_mesh_ortho_zbuf(verts, faces, spec, render_scale=rw / float(W))
        m_mask = buf["hit_mask"]
        inter = float(np.sum(m_mask & fg_ds))
        union = float(np.sum(m_mask | fg_ds))
        return inter / max(union, 1.0)

    best_iou = _eval_iou(best)
    init_iou = best_iou

    for step_px, step_sc in ((4.0, 0.025), (1.5, 0.010)):
        improved = True
        iters = 0
        while improved and iters < 6:
            improved = False
            iters += 1
            for dcx, dcy, dsx, dsy in (
                (-step_px, 0.0, 1.0, 1.0),
                (step_px, 0.0, 1.0, 1.0),
                (0.0, -step_px, 1.0, 1.0),
                (0.0, step_px, 1.0, 1.0),
                (0.0, 0.0, 1.0 - step_sc, 1.0),
                (0.0, 0.0, 1.0 + step_sc, 1.0),
                (0.0, 0.0, 1.0, 1.0 - step_sc),
                (0.0, 0.0, 1.0, 1.0 + step_sc),
            ):
                if not allow_anisotropic_xy and abs(dsx - dsy) > 1e-6:
                    continue
                cand = OrthoCameraSpec(
                    name=best.name,
                    azim_deg=best.azim_deg,
                    elev_deg=best.elev_deg,
                    width=best.width,
                    height=best.height,
                    cam_pos_dir=best.cam_pos_dir,
                    cam_right=best.cam_right,
                    cam_up=best.cam_up,
                    cx_px=best.cx_px + dcx,
                    cy_px=best.cy_px + dcy,
                    sx_px=best.sx_px * dsx,
                    sy_px=best.sy_px * dsy,
                    trust=best.trust,
                    is_primary=best.is_primary,
                )
                iou = _eval_iou(cand)
                if iou > best_iou + 1e-4:
                    best = cand
                    best_iou = iou
                    improved = True

    return best, {
        "init_iou": round(init_iou, 4),
        "calibrated_iou": round(best_iou, 4),
        "cx_px": round(best.cx_px, 2),
        "cy_px": round(best.cy_px, 2),
        "sx_px": round(best.sx_px, 2),
        "sy_px": round(best.sy_px, 2),
    }


def compute_silhouette_alignment_warp(
    view: MattedView,
    mesh_hit_mask: np.ndarray,
    *,
    protect_mask_2d: np.ndarray | None = None,
    max_shift_px: float = 10.0,
) -> tuple[MattedView, dict[str, Any]]:
    """Gently align 2D S0 silhouette boundaries to the 3D mesh silhouette without warping protected regions."""
    H, W = view.fg_mask.shape
    if mesh_hit_mask.shape != (H, W):
        raise ValueError(f"Shape mismatch: mesh_hit_mask {mesh_hit_mask.shape} vs {(H, W)}")

    dx_field = np.zeros((H, W), dtype=np.float32)
    row_has_both = np.any(mesh_hit_mask, axis=1) & np.any(view.fg_mask, axis=1)
    xs_grid = np.arange(W, dtype=np.float32)[None, :]

    for y in np.flatnonzero(row_has_both):
        mc = np.flatnonzero(mesh_hit_mask[y])
        ic = np.flatnonzero(view.fg_mask[y])
        mx0, mx1 = float(mc[0]), float(mc[-1])
        ix0, ix1 = float(ic[0]), float(ic[-1])
        mw = max(mx1 - mx0, 8.0)
        iw = max(ix1 - ix0, 8.0)
        if abs(mw - iw) > 2.5 * float(max_shift_px) and mw > 40.0:
            continue
        x_src = ix0 + (xs_grid[0] - mx0) * (iw / mw)
        dx_row = np.clip(x_src - xs_grid[0], -float(max_shift_px), float(max_shift_px))
        dx_field[y, :] = dx_row

    dx_smooth = ndimage.gaussian_filter(dx_field, sigma=(14.0, 18.0))
    dx_smooth = np.clip(dx_smooth, -float(max_shift_px), float(max_shift_px))
    if protect_mask_2d is not None:
        p_blur = ndimage.gaussian_filter(protect_mask_2d.astype(np.float32), sigma=10.0)
        dx_smooth *= np.clip(1.0 - p_blur, 0.0, 1.0)

    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    sample_x = np.clip(xx + dx_smooth, 0.0, float(W - 1))

    warped_fg = ndimage.map_coordinates(view.fg_mask.astype(np.float32), [yy, sample_x], order=1, mode="nearest") > 0.5
    iou_before = float(np.sum(mesh_hit_mask & view.fg_mask)) / max(float(np.sum(mesh_hit_mask | view.fg_mask)), 1.0)
    iou_after = float(np.sum(mesh_hit_mask & warped_fg)) / max(float(np.sum(mesh_hit_mask | warped_fg)), 1.0)

    if iou_after <= iou_before + 0.001:
        return view, {"applied": False, "iou_before": round(iou_before, 4), "iou_after": round(iou_after, 4)}

    warped_orig = np.stack(
        [ndimage.map_coordinates(view.rgb_orig[:, :, c], [yy, sample_x], order=1, mode="nearest") for c in range(3)],
        axis=-1,
    ).astype(np.float32)
    warped_dil = np.stack(
        [ndimage.map_coordinates(view.rgb_dilated[:, :, c], [yy, sample_x], order=1, mode="nearest") for c in range(3)],
        axis=-1,
    ).astype(np.float32)
    warped_alpha = ndimage.map_coordinates(view.alpha_soft, [yy, sample_x], order=1, mode="nearest").astype(np.float32)

    out_view = MattedView(
        name=view.name,
        rgb_orig=np.clip(warped_orig, 0.0, 1.0),
        rgb_dilated=np.clip(warped_dil, 0.0, 1.0),
        alpha_soft=np.clip(warped_alpha, 0.0, 1.0),
        fg_mask=warped_fg,
        bg_rgb=view.bg_rgb,
    )
    return out_view, {
        "applied": True,
        "iou_before": round(iou_before, 4),
        "iou_after": round(iou_after, 4),
        "max_dx_px": round(float(np.max(np.abs(dx_smooth))), 2),
    }
