"""Unified quadric edge collapse (QEM) and AutoRemesher runners for S2 experiments.

Supported engines and verified upstream repositories:
- pyfqmr: https://github.com/Kramer84/pyfqmr-Fast-Quadric-Mesh-Reduction
- PyMeshLab: https://github.com/cnr-isti-vclab/PyMeshLab
  (Linux libOpenGL.so.0 note: https://github.com/cnr-isti-vclab/PyMeshLab/issues/450)
- Open3D: https://github.com/isl-org/Open3D
- AutoRemesher: https://github.com/huxingyi/autoremesher

CRITICAL PyMeshLab detail:
When `vertex_weights` are provided, construct
`pymeshlab.Mesh(vertex_matrix=V, face_matrix=F, v_scalar_array=weights)` directly
rather than calling `m.vertex_scalar_array()` before vertex quality exists.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import trimesh

from quadvox.s2_retopo.contracts import RetopoMetrics
from quadvox.s2_retopo.metrics import (
    compute_retopo_metrics,
    extract_vf,
    load_mesh_raw,
    parse_obj_polygon_counts,
)


def decimate_pyfqmr(
    verts: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    *,
    preserve_border: bool = True,
    aggressiveness: float = 7.0,
    max_iterations: int = 100,
    update_rate: int = 5,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Fast Quadric Mesh Reduction (`pyfqmr`) on `(V, F)` arrays."""
    import pyfqmr

    v_in = np.ascontiguousarray(verts, dtype=np.float64)
    f_in = np.ascontiguousarray(faces, dtype=np.int32)
    if len(f_in) <= int(target_faces) or len(f_in) == 0:
        return v_in.copy(), f_in.astype(np.int64, copy=True)

    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(v_in, f_in)
    simplifier.simplify_mesh(
        target_count=max(4, int(target_faces)),
        update_rate=int(update_rate),
        aggressiveness=float(aggressiveness),
        max_iterations=int(max_iterations),
        verbose=bool(verbose),
        preserve_border=bool(preserve_border),
    )
    v_out, f_out, _ = simplifier.getMesh()
    return (
        np.ascontiguousarray(v_out, dtype=np.float64),
        np.ascontiguousarray(f_out, dtype=np.int64),
    )


def should_compare_unpreserved_border(
    actual_faces: int,
    target_faces: int,
    *,
    tolerance_ratio: float = 0.10,
) -> bool:
    """Return True if `preserve_border=True` stalled >10% above `target_faces`."""
    if target_faces <= 0:
        return False
    return int(actual_faces) > int(round(float(target_faces) * (1.0 + float(tolerance_ratio))))


def sanitize_vertex_weights(
    vertex_weights: np.ndarray,
    expected_num_verts: int,
    *,
    min_weight: float = 1e-3,
) -> np.ndarray:
    """Validate vertex weights length and clamp finite values >= `min_weight`."""
    w = np.asarray(vertex_weights, dtype=np.float64).reshape(-1)
    if len(w) != int(expected_num_verts):
        raise ValueError(
            f"vertex_weights length ({len(w)}) does not match vertex count ({expected_num_verts})"
        )
    w = np.where(np.isfinite(w), w, 1.0)
    w = np.maximum(w, float(min_weight))
    return np.ascontiguousarray(w, dtype=np.float64)


def compute_true_head_face_ratio(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    head_y_norm_thresh: float = 0.72,
    head_abs_x_max: float = 0.14,
    y_min: float | None = None,
    height: float | None = None,
) -> tuple[float, int]:
    """Compute fraction and count of faces with y_norm > 0.72 AND |x| <= 0.14.

    Distinguishes real head faces from T-pose arms/hands that also sit above
    `y_norm > 0.72` at larger `|x|`.
    """
    if len(faces) == 0 or len(verts) == 0:
        return 0.0, 0

    if y_min is None or height is None:
        finite = np.isfinite(verts[:, 1])
        if not np.any(finite):
            return 0.0, 0
        y_vals = verts[finite, 1]
        y_lo = float(np.min(y_vals))
        h = float(np.max(y_vals)) - y_lo
    else:
        y_lo = float(y_min)
        h = float(height)

    if not math.isfinite(h) or h <= 1e-12:
        return 0.0, 0

    cx = (verts[faces[:, 0], 0] + verts[faces[:, 1], 0] + verts[faces[:, 2], 0]) / 3.0
    cy = (verts[faces[:, 0], 1] + verts[faces[:, 1], 1] + verts[faces[:, 2], 1]) / 3.0
    y_norm = (cy - y_lo) / h
    mask = (y_norm > float(head_y_norm_thresh)) & (np.abs(cx) <= float(head_abs_x_max))
    count = int(np.sum(mask))
    return float(count / len(faces)), count


def decimate_pymeshlab(
    verts: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    *,
    vertex_weights: np.ndarray | None = None,
    preserve_boundary: bool = True,
    boundary_weight: float = 1.5,
    preserve_normal: bool = True,
    preserve_topology: bool = False,
    planar_quadric: bool = True,
    quality_thr: float = 0.3,
    optimal_placement: bool = True,
    autoclean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run PyMeshLab `meshing_decimation_quadric_edge_collapse` with optional vertex weights."""
    import pymeshlab

    v_in = np.ascontiguousarray(verts, dtype=np.float64)
    f_in = np.ascontiguousarray(faces, dtype=np.int32)
    if len(f_in) <= int(target_faces) or len(f_in) == 0:
        return v_in.copy(), f_in.astype(np.int64, copy=True)

    use_quality_weight = vertex_weights is not None
    if use_quality_weight:
        w = sanitize_vertex_weights(vertex_weights, len(v_in), min_weight=1e-3)
        pm_mesh = pymeshlab.Mesh(
            vertex_matrix=v_in,
            face_matrix=f_in,
            v_scalar_array=w,
        )
    else:
        pm_mesh = pymeshlab.Mesh(
            vertex_matrix=v_in,
            face_matrix=f_in,
        )

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pm_mesh, "input")
    if not hasattr(ms, "meshing_decimation_quadric_edge_collapse"):
        raise RuntimeError(
            "PyMeshLab missing meshing_decimation_quadric_edge_collapse "
            "(see https://github.com/cnr-isti-vclab/PyMeshLab/issues/450 for libOpenGL.so.0)"
        )

    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=max(4, int(target_faces)),
        qualitythr=float(quality_thr),
        preserveboundary=bool(preserve_boundary),
        boundaryweight=float(boundary_weight),
        preservenormal=bool(preserve_normal),
        preservetopology=bool(preserve_topology),
        optimalplacement=bool(optimal_placement),
        planarquadric=bool(planar_quadric),
        qualityweight=bool(use_quality_weight),
        autoclean=bool(autoclean),
    )

    out_mesh = ms.current_mesh()
    v_out = np.ascontiguousarray(out_mesh.vertex_matrix(), dtype=np.float64)
    f_out = np.ascontiguousarray(out_mesh.face_matrix(), dtype=np.int64)
    return v_out, f_out


def decimate_open3d(
    verts: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    *,
    preserve_boundary: bool = True,
    boundary_weight: float = 1.5,
    maximum_error: float = float("inf"),
) -> tuple[np.ndarray, np.ndarray]:
    """Run Open3D `simplify_quadric_decimation` on `(V, F)` arrays."""
    import open3d as o3d

    v_in = np.ascontiguousarray(verts, dtype=np.float64)
    f_in = np.ascontiguousarray(faces, dtype=np.int32)
    if len(f_in) <= int(target_faces) or len(f_in) == 0:
        return v_in.copy(), f_in.astype(np.int64, copy=True)

    o3_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(v_in),
        triangles=o3d.utility.Vector3iVector(f_in),
    )
    bw = float(boundary_weight) if preserve_boundary else 0.0
    out_mesh = o3_mesh.simplify_quadric_decimation(
        target_number_of_triangles=max(4, int(target_faces)),
        maximum_error=float(maximum_error),
        boundary_weight=bw,
    )
    out_mesh.remove_unreferenced_vertices()
    v_out = np.ascontiguousarray(np.asarray(out_mesh.vertices), dtype=np.float64)
    f_out = np.ascontiguousarray(np.asarray(out_mesh.triangles), dtype=np.int64)
    return v_out, f_out


def resolve_autoremesher_bin(autoremesher_bin: str | Path | None = None) -> Path:
    """Resolve AutoRemesher executable path from argument, env, or PATH."""
    candidates: list[str] = []
    if autoremesher_bin is not None:
        candidates.append(str(autoremesher_bin))
    env_bin = os.environ.get("AUTOREMESHER_BIN")
    if env_bin:
        candidates.append(env_bin)
    which_bin = shutil.which("autoremesher")
    if which_bin:
        candidates.append(which_bin)

    for cand in candidates:
        p = Path(cand)
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(
        "AutoRemesher binary not found. Pass --autoremesher-bin or set AUTOREMESHER_BIN."
    )


def run_autoremesher_cli(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_faces: int = 35000,
    target_quads: int | None = None,
    autoremesher_bin: str | Path | None = None,
    report_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Invoke headless AutoRemesher CLI with `.glb <-> .obj` conversion and quad tracking."""
    t0 = time.perf_counter()
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path = resolve_autoremesher_bin(autoremesher_bin)

    work_root = Path(work_dir) if work_dir is not None else (out_path.parent / f"._{out_path.stem}_ar_work")
    work_root.mkdir(parents=True, exist_ok=True)

    # Convert non-OBJ input (e.g. .glb) to .obj inside project work directory.
    if in_path.suffix.lower() == ".obj":
        in_obj = in_path
    else:
        in_obj = work_root / f"{in_path.stem}_input.obj"
        raw_mesh = load_mesh_raw(in_path)
        raw_mesh.export(str(in_obj))

    if out_path.suffix.lower() == ".obj":
        out_obj = out_path
    else:
        out_obj = out_path.with_suffix(".obj")

    rep_path = Path(report_path) if report_path is not None else (work_root / f"{out_path.stem}_ar_report.json")
    rep_path.parent.mkdir(parents=True, exist_ok=True)

    tq = int(target_quads) if target_quads is not None else max(1, int(round(float(target_faces) / 2.0)))

    cmd = [
        str(bin_path),
        "-i",
        str(in_obj),
        "-o",
        str(out_obj),
        "--target-quads",
        str(tq),
        "--report",
        str(rep_path),
    ]
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"

    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=float(timeout),
    )
    elapsed_s = time.perf_counter() - t0
    if proc.returncode != 0 or not out_obj.exists():
        raise RuntimeError(
            f"AutoRemesher failed (code={proc.returncode}, elapsed={elapsed_s:.2f}s):\n"
            f"CMD: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
        )

    poly_counts = parse_obj_polygon_counts(out_obj)
    if out_path.suffix.lower() != ".obj":
        tri_mesh = load_mesh_raw(out_obj)
        tri_mesh.export(str(out_path))

    report_payload: dict[str, Any] = {}
    if rep_path.exists():
        try:
            report_payload = json.loads(rep_path.read_text(encoding="utf-8"))
        except Exception:
            report_payload = {"raw_report_text": rep_path.read_text(encoding="utf-8", errors="replace")}

    return {
        "input_obj": str(in_obj),
        "output_obj": str(out_obj),
        "output_path": str(out_path),
        "report_path": str(rep_path),
        "target_quads": tq,
        "quad_count": poly_counts.quad_count,
        "tri_count": poly_counts.tri_count,
        "ngon_count": poly_counts.ngon_count,
        "equiv_tri_count": poly_counts.equiv_tri_count,
        "elapsed_s": elapsed_s,
        "report": report_payload,
    }


def _load_weights_array(
    weights_npz: str | Path,
    *,
    weights_key: str = "weights",
    weight_multiplier: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    from quadvox.s2_retopo.semantic_weights import load_weights_npz

    payload = load_weights_npz(weights_npz)
    if weights_key not in payload:
        raise KeyError(f"Key '{weights_key}' not found in weights npz: {list(payload.keys())}")
    w = np.asarray(payload[weights_key], dtype=np.float64)
    if weight_multiplier is not None:
        target_mult = max(1e-3, float(weight_multiplier))
        orig_mult = payload.get("focus_weight_multiplier")
        if orig_mult is not None and float(orig_mult) > 1.0 + 1e-6:
            norm_field = np.clip((w - 1.0) / (float(orig_mult) - 1.0), 0.0, 1.0)
            w = 1.0 + (target_mult - 1.0) * norm_field
        elif float(np.max(w)) > 1.0 + 1e-6:
            norm_field = np.clip((w - 1.0) / (float(np.max(w)) - 1.0), 0.0, 1.0)
            w = 1.0 + (target_mult - 1.0) * norm_field
        elif "soft_score" in payload:
            score = np.clip(np.asarray(payload["soft_score"], dtype=np.float64), 0.0, 1.0)
            w = 1.0 + (target_mult - 1.0) * score
        elif "focus_mask" in payload:
            mask = np.asarray(payload["focus_mask"], dtype=np.float64) > 0.5
            w = np.where(mask, target_mult, 1.0)
    return w, payload


def decimate_mesh_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    engine: str,
    target_faces: int = 35000,
    preserve_boundary: bool = True,
    boundary_weight: float = 1.5,
    preserve_normal: bool = True,
    preserve_topology: bool = False,
    planar_quadric: bool = True,
    quality_thr: float = 0.3,
    optimal_placement: bool = True,
    autoclean: bool = True,
    vertex_weights: np.ndarray | None = None,
    weights_npz: str | Path | None = None,
    weights_key: str = "weights",
    weight_multiplier: float | None = None,
    highpoly_path: str | Path | None = None,
    sample_count: int = 100000,
    seed: int = 42,
    fallback_unpreserved_out: str | Path | None = None,
    autoremesher_bin: str | Path | None = None,
    target_quads: int | None = None,
    autoremesher_timeout: float = 600.0,
    metrics_json_path: str | Path | None = None,
) -> tuple[RetopoMetrics, dict[str, Any]]:
    """Decimate a mesh file with the selected engine, save output, and compute RetopoMetrics."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine_norm = engine.strip().lower()

    extra_info: dict[str, Any] = {"engine": engine_norm}
    weights_meta: dict[str, Any] = {}
    if vertex_weights is None and weights_npz is not None:
        vertex_weights, weights_meta = _load_weights_array(
            weights_npz,
            weights_key=weights_key,
            weight_multiplier=weight_multiplier,
        )
        extra_info["weights_npz"] = str(weights_npz)

    t0 = time.perf_counter()
    if engine_norm == "autoremesher":
        ar_info = run_autoremesher_cli(
            in_path,
            out_path,
            target_faces=target_faces,
            target_quads=target_quads,
            autoremesher_bin=autoremesher_bin,
            timeout=autoremesher_timeout,
        )
        decimate_elapsed_s = float(ar_info["elapsed_s"])
        extra_info["autoremesher"] = ar_info
        metric_target_path: str | Path = ar_info["output_obj"]
    else:
        verts, faces = extract_vf(in_path)
        if engine_norm == "pyfqmr":
            v_out, f_out = decimate_pyfqmr(
                verts,
                faces,
                target_faces=target_faces,
                preserve_border=preserve_boundary,
            )
            if (
                preserve_boundary
                and fallback_unpreserved_out is not None
                and should_compare_unpreserved_border(len(f_out), target_faces)
            ):
                fb_path = Path(fallback_unpreserved_out)
                fb_path.parent.mkdir(parents=True, exist_ok=True)
                t_fb = time.perf_counter()
                v_fb, f_fb = decimate_pyfqmr(
                    verts,
                    faces,
                    target_faces=target_faces,
                    preserve_border=False,
                )
                fb_elapsed = time.perf_counter() - t_fb
                trimesh.Trimesh(vertices=v_fb, faces=f_fb, process=False).export(str(fb_path))
                extra_info["pyfqmr_unpreserved_fallback"] = {
                    "path": str(fb_path),
                    "face_count": int(len(f_fb)),
                    "vertex_count": int(len(v_fb)),
                    "elapsed_s": fb_elapsed,
                }
        elif engine_norm == "pymeshlab":
            v_out, f_out = decimate_pymeshlab(
                verts,
                faces,
                target_faces=target_faces,
                vertex_weights=vertex_weights,
                preserve_boundary=preserve_boundary,
                boundary_weight=boundary_weight,
                preserve_normal=preserve_normal,
                preserve_topology=preserve_topology,
                planar_quadric=planar_quadric,
                quality_thr=quality_thr,
                optimal_placement=optimal_placement,
                autoclean=autoclean,
            )
            extra_info["pymeshlab_options"] = {
                "preserve_boundary": bool(preserve_boundary),
                "boundary_weight": float(boundary_weight),
                "preserve_normal": bool(preserve_normal),
                "preserve_topology": bool(preserve_topology),
                "planar_quadric": bool(planar_quadric),
                "quality_thr": float(quality_thr),
                "optimal_placement": bool(optimal_placement),
                "autoclean": bool(autoclean),
            }
        elif engine_norm == "open3d":
            v_out, f_out = decimate_open3d(
                verts,
                faces,
                target_faces=target_faces,
                preserve_boundary=preserve_boundary,
                boundary_weight=boundary_weight,
            )
        else:
            raise ValueError(
                f"Unsupported QEM/retopo engine '{engine}'. "
                "Expected one of: pyfqmr, pymeshlab, open3d, autoremesher."
            )

        decimate_elapsed_s = time.perf_counter() - t0
        out_mesh = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
        out_mesh.export(str(out_path))
        metric_target_path = out_path
        th_ratio, th_count = compute_true_head_face_ratio(v_out, f_out)
        extra_info["true_head_face_ratio"] = th_ratio
        extra_info["true_head_face_count"] = th_count

    focus_ratio = 0.0
    part_clusters = 0
    semantic_extract_s = 0.0
    if vertex_weights is not None:
        focus_ratio = float(np.mean(np.asarray(vertex_weights, dtype=np.float64) > 1.05))
    if weights_meta:
        part_clusters = int(weights_meta.get("part_cluster_count", 0))
        semantic_extract_s = float(weights_meta.get("semantic_extract_s", 0.0))
        if "focus_vertex_ratio" in weights_meta:
            focus_ratio = float(weights_meta["focus_vertex_ratio"])

    metrics = compute_retopo_metrics(
        metric_target_path,
        highpoly_mesh_or_path=highpoly_path,
        sample_count=sample_count,
        seed=seed,
        semantic_extract_s=semantic_extract_s,
        part_cluster_count=part_clusters,
        focus_vertex_ratio=focus_ratio,
        decimate_elapsed_s=decimate_elapsed_s,
    )
    if metrics_json_path is not None:
        metrics.write_json(metrics_json_path)
    return metrics, extra_info


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-subprocess S2 mesh decimation.")
    parser.add_argument("--input", required=True, help="Input cleaned highpoly mesh (.glb/.obj/.ply).")
    parser.add_argument("--out", required=True, help="Output lowpoly mesh path.")
    parser.add_argument(
        "--engine",
        required=True,
        choices=["pyfqmr", "pymeshlab", "open3d", "autoremesher"],
        help="Decimation / retopology engine.",
    )
    parser.add_argument("--target-faces", type=int, default=35000, help="Target triangle count.")
    parser.add_argument("--target-quads", type=int, default=None, help="AutoRemesher target quad count.")
    parser.add_argument(
        "--preserve-boundary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve open boundary loops during decimation.",
    )
    parser.add_argument("--boundary-weight", type=float, default=1.5, help="Boundary preservation weight.")
    parser.add_argument("--quality-thr", type=float, default=0.3, help="PyMeshLab qualitythr.")
    parser.add_argument(
        "--preserve-normal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PyMeshLab preservenormal flag.",
    )
    parser.add_argument(
        "--preserve-topology",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="PyMeshLab preservetopology flag.",
    )
    parser.add_argument(
        "--planar-quadric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PyMeshLab planarquadric flag.",
    )
    parser.add_argument("--weights-npz", default=None, help="Optional per-vertex weights .npz path.")
    parser.add_argument("--weights-key", default="weights", help="Array key inside weights .npz.")
    parser.add_argument(
        "--weight-multiplier",
        type=float,
        default=None,
        help="Optional multiplier rescale when .npz contains soft_score/focus_mask.",
    )
    parser.add_argument("--highpoly", default=None, help="Optional highpoly path for Chamfer/Hausdorff.")
    parser.add_argument("--samples", type=int, default=100000, help="Surface sample count for distance metrics.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    parser.add_argument(
        "--fallback-unpreserved-out",
        default=None,
        help="Optional path to emit preserve_border=False pyfqmr comparison if stalled >10%% above target.",
    )
    parser.add_argument("--autoremesher-bin", default=None, help="Path to AutoRemesher binary.")
    parser.add_argument("--timeout", type=float, default=600.0, help="AutoRemesher CLI timeout in seconds.")
    parser.add_argument("--metrics-json", default=None, help="Optional output RetopoMetrics JSON path.")
    parser.add_argument("--payload-json", default=None, help="Optional full {metrics, extra} JSON path.")
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    metrics, extra = decimate_mesh_file(
        args.input,
        args.out,
        engine=args.engine,
        target_faces=args.target_faces,
        preserve_boundary=args.preserve_boundary,
        boundary_weight=args.boundary_weight,
        preserve_normal=args.preserve_normal,
        preserve_topology=args.preserve_topology,
        planar_quadric=args.planar_quadric,
        quality_thr=args.quality_thr,
        weights_npz=args.weights_npz,
        weights_key=args.weights_key,
        weight_multiplier=args.weight_multiplier,
        highpoly_path=args.highpoly,
        sample_count=args.samples,
        seed=args.seed,
        fallback_unpreserved_out=args.fallback_unpreserved_out,
        autoremesher_bin=args.autoremesher_bin,
        target_quads=args.target_quads,
        autoremesher_timeout=args.timeout,
        metrics_json_path=args.metrics_json,
    )
    payload = {"metrics": metrics.to_dict(), "extra": extra}
    if args.payload_json:
        pj = Path(args.payload_json)
        pj.parent.mkdir(parents=True, exist_ok=True)
        pj.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
