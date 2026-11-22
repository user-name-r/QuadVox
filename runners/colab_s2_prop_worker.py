"""Colab GPU runner for Step 2C: Prop LATO.2 v3000 + Crack/Manifold Repair + S2.5 Low-UV + 4K Bake.

Runs strictly inside `/content/cache` (never `/tmp`), leaves `quadvox/s2_retopo/slim_uv_bake.py`
unmodified, enforces `0/0/0/1` watertight manifold closure, evaluates `39°` vs
`48°` prop seam thresholds per Gate 3/#158, bakes 4K Normal+AO+Curvature via
`WeldedSmoothBakeCage`, and renders 4-state + front-lower-right closeup sheets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

CACHE_ROOT = Path("/content/cache")
WORK_DIR = CACHE_ROOT / "work"
LOGS_DIR = CACHE_ROOT / "logs"
OUTPUTS_DIR = CACHE_ROOT / "outputs"
TRANSFER_DIR = CACHE_ROOT / "transfer"
SCRATCH_DIR = CACHE_ROOT / "scratch"
REPOS_DIR = CACHE_ROOT / "repos"
MODELS_DIR = CACHE_ROOT / "models"
TOOLS_DIR = CACHE_ROOT / "tools"

MAX_CHUNK_BYTES = 5 * 1024 * 1024
DEFAULT_BLENDER_URL = "https://download.blender.org/release/Blender5.3/blender-5.3.0-linux-x64.tar.xz"
DEFAULT_LATO2_REPO = "https://github.com/sakalond/ComfyUI-LATO2.git"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(1024 * 1024)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    prev_pp = env.get("PYTHONPATH", "")
    parts = [str(WORK_DIR), str(WORK_DIR / "scripts")]
    if prev_pp:
        parts.append(prev_pp)
    env["PYTHONPATH"] = ":".join(parts)
    env["TMPDIR"] = str(SCRATCH_DIR)
    env["TEMP"] = str(SCRATCH_DIR)
    env["TMP"] = str(SCRATCH_DIR)
    env["PIP_CACHE_DIR"] = str(CACHE_ROOT / "pip_cache")
    env["HF_HOME"] = str(CACHE_ROOT / "hf_home")
    env["TORCH_HOME"] = str(CACHE_ROOT / "torch_home")
    env["TQDM_DISABLE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_logged(
    cmd: list[str],
    log_path: Path,
    stage: str,
    *,
    cwd: Path | None = None,
    timeout: float = 1800.0,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{stage}] Running: {' '.join(cmd)}", flush=True)
    t0 = time.monotonic()
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd or WORK_DIR),
            env=_base_env(),
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:])
        raise RuntimeError(f"[{stage}] failed rc={proc.returncode} ({elapsed:.1f}s):\n{tail}")
    print(f"[{stage}] OK ({elapsed:.1f}s)", flush=True)
    return round(elapsed, 2)


def _setup_sys_and_python_deps() -> None:
    _run_logged(
        [
            "bash",
            "-lc",
            "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
            "libopengl0 libglu1-mesa libxrender1 libx11-6 libgl1 libxi6 libxkbcommon0 "
            "libxkbcommon-x11-0 libegl1 libsm6 libice6 libxfixes3 libfontconfig1 libdbus-1-3 "
            "xz-utils unzip curl git > /dev/null",
        ],
        LOGS_DIR / "apt_setup.log",
        "setup_apt",
        timeout=600.0,
    )
    _run_logged(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "pymeshlab",
            "trimesh",
            "scipy",
            "pillow",
            "xatlas",
        ],
        LOGS_DIR / "pip_setup.log",
        "setup_pip",
        timeout=600.0,
    )
    # Open3D optional (cp313 fallback uses PyMeshLab QEM cleanly if Open3D wheel is unavailable)
    try:
        _run_logged(
            [sys.executable, "-m", "pip", "install", "--quiet", "open3d"],
            LOGS_DIR / "pip_open3d.log",
            "setup_pip_open3d",
            timeout=300.0,
        )
    except Exception as exc:
        print(f"[setup_pip_open3d] Open3D unavailable ({exc}); will use PyMeshLab QEM for 120k proxy.", flush=True)


def _ensure_blender_linux(blender_url: str = DEFAULT_BLENDER_URL) -> Path:
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    cached = TOOLS_DIR / "blender-5.3.0-linux-x64" / "blender"
    if cached.is_file():
        return cached
    for p in sorted(TOOLS_DIR.glob("blender-*/blender")):
        if p.is_file():
            return p
    arc = TOOLS_DIR / "blender-5.3.0-linux-x64.tar.xz"
    if not arc.is_file():
        print(f"[setup_blender] Downloading {blender_url}...", flush=True)
        urllib.request.urlretrieve(blender_url, str(arc))
    with tarfile.open(arc, "r:xz") as tf:
        tf.extractall(TOOLS_DIR, filter="data") if hasattr(tarfile, "data_filter") else tf.extractall(TOOLS_DIR)
    arc.unlink(missing_ok=True)
    for p in sorted(TOOLS_DIR.glob("blender-*/blender")):
        if p.is_file():
            return p
    raise FileNotFoundError(f"Blender binary missing after extracting {blender_url}")


def _ensure_lato2_env(repo_url: str = DEFAULT_LATO2_REPO) -> tuple[Path, Path, Path, Path, Path]:
    repo_dir = REPOS_DIR / "ComfyUI-LATO2"
    if not (repo_dir / "install_lato2.py").is_file():
        _run_logged(
            ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
            LOGS_DIR / "lato2_clone.log",
            "lato2_clone",
            cwd=REPOS_DIR,
            timeout=300.0,
        )
    models_dir = MODELS_DIR / "lato2"
    models_dir.mkdir(parents=True, exist_ok=True)
    lato_py = repo_dir / "envs" / "lato2" / "bin" / "python"
    if not lato_py.is_file() or not (models_dir / "vflow.pt").is_file():
        _run_logged(
            [
                sys.executable,
                str(repo_dir / "install_lato2.py"),
                "--models-dir",
                str(models_dir),
                "--only",
                "repo,env,models",
            ],
            LOGS_DIR / "lato2_install.log",
            "lato2_install",
            cwd=repo_dir,
            timeout=1500.0,
        )
    runner_py = repo_dir / "runners" / "lato2_runner.py"
    vendor_repo = repo_dir / "vendor" / "LATO.2"
    dino_dir = models_dir / "dinov2"
    return lato_py, runner_py, vendor_repo, models_dir, dino_dir


def _build_proxy_open3d(src_glb: Path, out_glb: Path, target_faces: int = 120_000) -> dict[str, Any]:
    import numpy as np
    import trimesh

    t0 = time.monotonic()
    loaded = trimesh.load(str(src_glb), force="mesh", process=False)
    v = np.asarray(loaded.vertices, dtype=np.float64)
    f = np.asarray(loaded.faces, dtype=np.int64)
    orig_f = int(len(f))
    backend = "identity"
    if orig_f <= int(target_faces):
        v_out, f_out = v, f
    else:
        try:
            import open3d as o3d

            om = o3d.geometry.TriangleMesh()
            om.vertices = o3d.utility.Vector3dVector(v)
            om.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))
            om.remove_duplicated_vertices()
            om.remove_degenerate_triangles()
            om.remove_duplicated_triangles()
            dec = om.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
            dec.remove_unreferenced_vertices()
            dec.remove_degenerate_triangles()
            dec.remove_duplicated_triangles()
            v_out = np.asarray(dec.vertices, dtype=np.float64)
            f_out = np.asarray(dec.triangles, dtype=np.int64)
            backend = "open3d_qem"
        except Exception as exc:
            import pymeshlab

            print(f"[proxy] Open3D QEM unavailable ({exc}); using PyMeshLab QEM fallback...", flush=True)
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f.astype(np.int32)))
            ms.apply_filter(
                "meshing_decimation_quadric_edge_collapse",
                targetfacenum=int(target_faces),
                preservenormal=True,
                preservetopology=True,
                optimalplacement=True,
                planarquadric=False,
            )
            cm = ms.current_mesh()
            v_out = np.asarray(cm.vertex_matrix(), dtype=np.float64)
            f_out = np.asarray(cm.face_matrix(), dtype=np.int64)
            backend = "pymeshlab_qem"
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=v_out, faces=f_out, process=False).export(str(out_glb))
    return {
        "backend": backend,
        "source_faces": orig_f,
        "proxy_vertices": int(len(v_out)),
        "proxy_faces": int(len(f_out)),
        "elapsed_s": round(time.monotonic() - t0, 2),
    }


def _compute_proxy_chamfer(cand_glb: Path, ref_glb: Path, n_samples: int = 40_000) -> dict[str, float]:
    import numpy as np
    from scipy.spatial import cKDTree
    import trimesh

    mc = trimesh.load(str(cand_glb), force="mesh", process=False)
    mr = trimesh.load(str(ref_glb), force="mesh", process=False)
    rng = np.random.default_rng(42)
    pts_c, _ = trimesh.sample.sample_surface(mc, n_samples, seed=rng)
    pts_r, _ = trimesh.sample.sample_surface(mr, n_samples, seed=rng)
    d_c2r, _ = cKDTree(pts_r).query(pts_c)
    d_r2c, _ = cKDTree(pts_c).query(pts_r)
    return {
        "chamfer_c2r_l1": round(float(np.mean(d_c2r)), 6),
        "chamfer_r2c_l1": round(float(np.mean(d_r2c)), 6),
        "chamfer_sym_l1": round(float(0.5 * (np.mean(d_c2r) + np.mean(d_r2c))), 6),
    }


def _unwrap_prop_with_crease_sweep(
    clean_glb: Path,
    uv_dir: Path,
    blender_bin: Path,
    crease_candidates: Sequence[float] = (39.0, 48.0),
) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Run `quadvox.s2_retopo.slim_uv_bake.unwrap_part_aware(..., asset_type='prop')` cleanly without altering `slim_uv_bake.py`."""
    from quadvox.s2_retopo import slim_uv_bake as bake_4k  # type: ignore

    results: list[dict[str, Any]] = []
    uv_dir.mkdir(parents=True, exist_ok=True)

    for deg in crease_candidates:
        tag = f"deg{int(round(deg))}"
        out_glb = uv_dir / f"prop_v3000_{tag}_lowuv.glb"
        out_png = uv_dir / f"prop_v3000_{tag}_uv_layout.png"
        out_json = uv_dir / f"prop_v3000_{tag}_unwrap.json"

        res = bake_4k.unwrap_part_aware(
            clean_glb,
            out_glb,
            tex_size=4096,
            padding_px=16,
            face_scale_boost=1.0,
            uv_preview_png=out_png,
            unwrap_json_path=out_json,
            uv_method="blender_slim",
            asset_type="prop",
            blender_bin=blender_bin,
            crease_angle_deg=float(deg),
        )

        meta = res.to_dict()
        ov_pct = 100.0 * float(meta["uv_overlap_count"]) / float(1024 * 1024)
        results.append(
            {
                "crease_deg": float(deg),
                "out_glb": str(out_glb),
                "uv_preview_png": str(out_png),
                "unwrap_json": str(out_json),
                "island_count": int(meta["island_count"]),
                "uv_coverage": float(meta["uv_coverage"]),
                "uv_overlap_count": int(meta["uv_overlap_count"]),
                "uv_overlap_pct_1024": round(ov_pct, 5),
                "min_island_gap_px": float(meta["min_island_gap_px"]),
                "elapsed_s": float(meta["elapsed_s"]),
            }
        )

    # Prefer low overlap (<0.1%), then highest UV coverage, breaking ties in favor of <= 40 deg seam
    def _rank(item: dict[str, Any]) -> tuple[int, float, float]:
        ov_ok = 1 if item["uv_overlap_pct_1024"] < 0.1 else 0
        return (ov_ok, round(item["uv_coverage"], 5), 1.0 if item["crease_deg"] <= 40.5 else 0.0)

    best = max(results, key=_rank)
    return (
        Path(best["out_glb"]),
        Path(best["uv_preview_png"]),
        Path(best["unwrap_json"]),
        {"candidates": results, "selected": best},
    )


def _render_front_crack_comparison(
    blender_bin: Path,
    baseline_glb: Path | None,
    repaired_glb: Path,
    baked_glb: Path,
    normal_png: Path,
    ao_png: Path,
    out_png: Path,
) -> None:
    """Render tight Cycles GPU closeup of the front-lower-right area comparing baseline vs repaired vs baked."""
    from PIL import Image, ImageDraw

    helper_py = SCRATCH_DIR / "render_prop_crack_cmp.py"
    panels: list[tuple[str, Path, bool]] = []
    if baseline_glb is not None and baseline_glb.is_file():
        panels.append(("RAW_OLD_LATO2_V3000", baseline_glb, False))
    panels.append(("REPAIRED_CLAY_V3000", repaired_glb, False))
    panels.append(("FINAL_4K_NORMAL_AO", baked_glb, True))

    panel_paths = [(lbl, str(SCRATCH_DIR / f"crack_cmp_{idx}.png")) for idx, (lbl, _, _) in enumerate(panels)]

    code = f'''
import bpy, os
from mathutils import Vector

panels = {[(lbl, str(p), textured) for lbl, p, textured in panels]!r}
normal_path = {str(normal_png)!r}
ao_path = {str(ao_png)!r}
scratch_dir = {str(SCRATCH_DIR)!r}

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def setup_cycles():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = (d.type == "CUDA")
    sc.cycles.device = "GPU"
    sc.cycles.samples = 48
    sc.cycles.use_denoising = True
    sc.render.resolution_x = 960
    sc.render.resolution_y = 720
    sc.render.image_settings.file_format = "PNG"
    world = bpy.data.worlds.new("W")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.24, 0.26, 0.29, 1.0)
        bg.inputs[1].default_value = 0.7

for idx, (lbl, glb_p, use_maps) in enumerate(panels):
    clear_scene()
    setup_cycles()
    bpy.ops.import_scene.gltf(filepath=glb_p)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for o in meshes:
        if not use_maps:
            mat = bpy.data.materials.new("ClayMat")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            bsdf.inputs["Base Color"].default_value = (0.78, 0.79, 0.81, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.52
            o.data.materials.clear()
            o.data.materials.append(mat)
        else:
            mat = bpy.data.materials.new("BakedMat")
            mat.use_nodes = True
            nt = mat.node_tree
            bsdf = nt.nodes.get("Principled BSDF")
            bsdf.inputs["Roughness"].default_value = 0.45
            if os.path.isfile(ao_path):
                img_ao = bpy.data.images.load(ao_path, check_existing=True)
                img_ao.colorspace_settings.name = "Non-Color"
                tex_ao = nt.nodes.new("ShaderNodeTexImage")
                tex_ao.image = img_ao
                mix = nt.nodes.new("ShaderNodeMix")
                mix.data_type = "RGBA"
                mix.blend_type = "MULTIPLY"
                mix.inputs[0].default_value = 0.75
                mix.inputs[6].default_value = (0.78, 0.79, 0.81, 1.0)
                nt.links.new(tex_ao.outputs["Color"], mix.inputs[7])
                nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
            if os.path.isfile(normal_path):
                img_n = bpy.data.images.load(normal_path, check_existing=True)
                img_n.colorspace_settings.name = "Non-Color"
                tex_n = nt.nodes.new("ShaderNodeTexImage")
                tex_n.image = img_n
                nmap = nt.nodes.new("ShaderNodeNormalMap")
                nmap.space = "TANGENT"
                nmap.inputs["Strength"].default_value = 1.0
                nt.links.new(tex_n.outputs["Color"], nmap.inputs["Color"])
                nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
            o.data.materials.clear()
            o.data.materials.append(mat)

    for lpos, energy in [((2.0, -2.5, 2.2), 160.0), ((-2.0, -2.0, 1.4), 90.0), ((0.5, 2.5, 1.8), 65.0)]:
        ldata = bpy.data.lights.new("L", type="AREA")
        ldata.energy = energy
        ldata.size = 2.0
        lo = bpy.data.objects.new("L", ldata)
        lo.location = lpos
        d = Vector((0.35, 0.0, 0.25)) - Vector(lpos)
        lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        bpy.context.collection.objects.link(lo)

    all_corners = [m.matrix_world @ Vector(c) for m in meshes for c in m.bound_box]
    bmin = Vector((min(v.x for v in all_corners), min(v.y for v in all_corners), min(v.z for v in all_corners)))
    bmax = Vector((max(v.x for v in all_corners), max(v.y for v in all_corners), max(v.z for v in all_corners)))
    center = 0.5 * (bmin + bmax)
    span = max((bmax - bmin).length, 1e-3)

    cam_d = bpy.data.cameras.new("Cam")
    cam_d.lens = 48.0
    cam_o = bpy.data.objects.new("Cam", cam_d)
    cam_o.location = (center.x + 0.32 * span, center.y - 1.35 * span, center.z - 0.12 * span)
    target = Vector((center.x + 0.15 * span, center.y - 0.10 * span, center.z - 0.15 * span))
    cam_o.rotation_euler = (target - Vector(cam_o.location)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(cam_o)
    bpy.context.scene.camera = cam_o

    p_out = os.path.join(scratch_dir, f"crack_cmp_{{idx}}.png")
    bpy.context.scene.render.filepath = p_out
    bpy.ops.render.render(write_still=True)
'''
    helper_py.write_text(code, encoding="utf-8")
    _run_logged(
        [
            str(blender_bin),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(helper_py),
        ],
        LOGS_DIR / "render_crack_cmp.log",
        "render_crack_closeup",
        timeout=600.0,
    )

    imgs = [Image.open(p).convert("RGB") for _, p in panel_paths]
    w, h = imgs[0].size
    bar_h = 36
    sheet = Image.new("RGB", (w * len(imgs), h + bar_h), (28, 30, 34))
    draw = ImageDraw.Draw(sheet)
    for i, ((lbl, _), im) in enumerate(zip(panel_paths, imgs)):
        sheet.paste(im, (i * w, bar_h))
        draw.text((i * w + 16, 10), lbl, fill=(238, 240, 245))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


def _package_outputs(outputs_dir: Path, logs_dir: Path, max_chunk_bytes: int = MAX_CHUNK_BYTES) -> dict[str, Any]:
    TRANSFER_DIR.mkdir(parents=True, exist_ok=True)
    arc_path = SCRATCH_DIR / "prop_s2_outputs.tar.gz"
    files: list[tuple[Path, str]] = []
    for fp in sorted(outputs_dir.rglob("*")):
        if not fp.is_file():
            continue
        rel = fp.relative_to(outputs_dir)
        # Skip heavy intermediate 120k proxy / unused candidate UV glbs
        if rel.name.endswith("_proxy_120k.glb"):
            continue
        files.append((fp, f"outputs/{rel.as_posix()}"))
    for lp in sorted(logs_dir.glob("*.log")):
        if lp.is_file():
            files.append((lp, f"logs/{lp.name}"))
    with tarfile.open(arc_path, "w:gz") as tf:
        for src, arcname in files:
            tf.add(src, arcname=arcname, recursive=False)

    total_size = arc_path.stat().st_size
    full_sha = _sha256_file(arc_path)
    chunks: list[dict[str, Any]] = []
    idx = 0
    with arc_path.open("rb") as fh:
        while True:
            buf = fh.read(max_chunk_bytes)
            if not buf:
                break
            cname = f"{arc_path.name}.part{idx:03d}"
            cpath = TRANSFER_DIR / cname
            cpath.write_bytes(buf)
            chunks.append(
                {
                    "name": cname,
                    "size": len(buf),
                    "sha256": hashlib.sha256(buf).hexdigest(),
                }
            )
            idx += 1
    arc_path.unlink(missing_ok=True)
    manifest = {
        "archive_name": arc_path.name,
        "total_size": total_size,
        "sha256": full_sha,
        "file_count": len(files),
        "chunks": chunks,
    }
    (TRANSFER_DIR / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def run_prop_s2_pipeline(spec: dict[str, Any]) -> dict[str, Any]:
    t0 = time.monotonic()
    for d in (WORK_DIR, LOGS_DIR, OUTPUTS_DIR, TRANSFER_DIR, SCRATCH_DIR, REPOS_DIR, MODELS_DIR, TOOLS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(WORK_DIR))
    sys.path.insert(0, str(WORK_DIR / "scripts"))

    report: dict[str, Any] = {
        "track": "Step 2C Prop LATO.2 v3000 + Repair + S2.5 Low-UV + 4K Bake",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": {},
        "errors": [],
    }
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        report["gpu"] = gpu_info
        print(f"[prop-s2] GPU: {gpu_info}", flush=True)

        _setup_sys_and_python_deps()
        from quadvox.s2_retopo.lato2_repair import (  # type: ignore
            fix_glb_outward_winding_and_strip_normals,
            repair_prop_lato2_mesh,
        )
        from quadvox.s2_retopo.metrics import compute_mesh_metrics  # type: ignore

        blender_bin = _ensure_blender_linux(spec.get("blender_url", DEFAULT_BLENDER_URL))
        high_glb = WORK_DIR / spec["high_glb"]
        baseline_old_glb = (WORK_DIR / spec["baseline_old_glb"]) if spec.get("baseline_old_glb") else None
        raw_precomputed = (WORK_DIR / spec["raw_lato2_glb"]) if spec.get("raw_lato2_glb") else None

        lato_dir = OUTPUTS_DIR / "lato2_candidates"
        lato_dir.mkdir(parents=True, exist_ok=True)
        seeds = [int(s) for s in spec.get("lato2_seeds", [42, 43])]

        if raw_precomputed is not None:
            if not raw_precomputed.is_file():
                raise FileNotFoundError(f"Missing precomputed raw LATO.2 GLB: {raw_precomputed}")
            proxy_or_ref_glb = high_glb
            report["proxy"] = {
                "backend": "skipped_reused_raw_lato2",
                "raw_lato2_glb": spec["raw_lato2_glb"],
            }
            seed_sources: list[tuple[int, Path, float]] = []
            seed_val = seeds[0] if seeds else 42
            raw_out = lato_dir / f"prop_b085_pb070_lato2_v3000_seed{seed_val}_raw.glb"
            shutil.copy2(raw_precomputed, raw_out)
            seed_sources.append((seed_val, raw_out, 0.0))
        else:
            lato_py, runner_py, vendor_repo, lato_models, dino_dir = _ensure_lato2_env(
                spec.get("lato2_repo", DEFAULT_LATO2_REPO)
            )
            proxy_glb = SCRATCH_DIR / "prop_qef1_b085_pb070_proxy_120k.glb"
            report["proxy"] = _build_proxy_open3d(high_glb, proxy_glb, target_faces=int(spec.get("proxy_faces", 120_000)))
            print(f"[prop-s2] Proxy built: {report['proxy']}", flush=True)
            proxy_or_ref_glb = proxy_glb
            seed_sources = []
            for seed in seeds:
                raw_out = lato_dir / f"prop_b085_pb070_lato2_v3000_seed{seed}_raw.glb"
                elapsed_l = _run_logged(
                    [
                        str(lato_py),
                        str(runner_py),
                        "--repo",
                        str(vendor_repo),
                        "--ckpt",
                        str(lato_models),
                        "--dino_hub_dir",
                        str(dino_dir),
                        "--input",
                        str(proxy_glb),
                        "--output",
                        str(raw_out),
                        "--vert_num",
                        str(int(spec.get("vert_num", 3000))),
                        "--seed",
                        str(seed),
                        "--renderer",
                        "numpy",
                        "--solidify",
                        "--surface",
                    ],
                    LOGS_DIR / f"lato2_seed{seed}.log",
                    f"lato2_v3000_seed{seed}",
                    cwd=REPOS_DIR / "ComfyUI-LATO2",
                    timeout=900.0,
                )
                seed_sources.append((seed, raw_out, elapsed_l))

        best_cand: dict[str, Any] | None = None
        for seed, raw_out, elapsed_l in seed_sources:
            clean_out = lato_dir / f"prop_b085_pb070_lato2_v3000_seed{seed}_clean.glb"
            rep = repair_prop_lato2_mesh(raw_out, clean_out, ref_mesh=proxy_or_ref_glb)
            mets = compute_mesh_metrics(clean_out).to_dict()
            chf = _compute_proxy_chamfer(clean_out, proxy_or_ref_glb)
            cand_entry = {
                "seed": seed,
                "lato2_elapsed_s": elapsed_l,
                "raw_glb": str(raw_out.relative_to(OUTPUTS_DIR)),
                "clean_glb": str(clean_out.relative_to(OUTPUTS_DIR)),
                "repair_report": rep.to_dict(),
                "topo_metrics": mets,
                "chamfer_vs_proxy": chf,
            }
            report["candidates"][f"seed_{seed}"] = cand_entry
            print(
                f"[prop-s2] seed={seed}: WT={mets['is_watertight']} open={mets['open_boundary_edges']} "
                f"nm={mets['non_manifold_vertices']} comp={mets['connected_components']} "
                f"sym_chf={chf['chamfer_sym_l1']} si_fold={rep.remaining_si_fold_faces}",
                flush=True,
            )
            wt_ok = bool(mets["is_watertight"] and mets["connected_components"] == 1 and mets["non_manifold_vertices"] == 0)
            score = (1 if wt_ok else 0, -rep.remaining_si_fold_faces, -chf["chamfer_sym_l1"])
            if best_cand is None or score > best_cand["_score"]:
                best_cand = {**cand_entry, "_score": score, "_clean_path": clean_out}

        assert best_cand is not None
        final_clean_glb = OUTPUTS_DIR / "prop_lato2_v3000_clean.glb"
        shutil.copy2(best_cand["_clean_path"], final_clean_glb)
        final_mets = compute_mesh_metrics(final_clean_glb).to_dict()
        if not (final_mets["is_watertight"] and final_mets["is_pinhole_free"] and final_mets["connected_components"] == 1):
            raise RuntimeError(f"Repaired prop mesh failed strict manifold gate: {final_mets}")
        report["selected_candidate"] = {k: v for k, v in best_cand.items() if not k.startswith("_")}

        uv_dir = OUTPUTS_DIR / "uv_sweep"
        lowuv_glb, uv_preview_png, unwrap_json, uv_report = _unwrap_prop_with_crease_sweep(
            final_clean_glb,
            uv_dir,
            blender_bin,
            crease_candidates=(39.0, 48.0),
        )
        report["uv_unwrap"] = uv_report

        final_lowuv = OUTPUTS_DIR / "prop_lato2_v3000_lowuv.glb"
        final_uv_png = OUTPUTS_DIR / "prop_lato2_v3000_uv_layout.png"
        final_uv_json = OUTPUTS_DIR / "prop_lato2_v3000_unwrap.json"
        shutil.copy2(lowuv_glb, final_lowuv)
        shutil.copy2(uv_preview_png, final_uv_png)
        shutil.copy2(unwrap_json, final_uv_json)

        lowuv_fix_stats = fix_glb_outward_winding_and_strip_normals(final_lowuv, high_glb)
        report["lowuv_orientation_fix"] = lowuv_fix_stats
        print(f"[prop-s2] Low-UV winding/normal fix: {lowuv_fix_stats}", flush=True)

        out_normal = OUTPUTS_DIR / "prop_lato2_v3000_normal_4k.png"
        out_ao = OUTPUTS_DIR / "prop_lato2_v3000_ao_4k.png"
        out_curv = OUTPUTS_DIR / "prop_lato2_v3000_curvature_4k.png"
        out_baked_glb = OUTPUTS_DIR / "prop_lato2_v3000_baked_4k.glb"
        out_sheet = OUTPUTS_DIR / "prop_lato2_v3000_maps_4096_sheet.png"
        bake_meta_json = OUTPUTS_DIR / "prop_lato2_v3000_bake_meta.json"
        bake_py = WORK_DIR / "quadvox" / "s2_retopo" / "slim_uv_bake.py"

        _run_logged(
            [
                str(blender_bin),
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(bake_py),
                "--",
                "bake-blender",
                "--high",
                str(high_glb),
                "--low",
                str(final_lowuv),
                "--out-normal",
                str(out_normal),
                "--out-ao",
                str(out_ao),
                "--out-curvature",
                str(out_curv),
                "--out-glb",
                str(out_baked_glb),
                "--out-sheet",
                str(out_sheet),
                "--uv-preview-png",
                str(final_uv_png),
                "--tex-size",
                "4096",
                "--padding-px",
                "16",
                "--normal-samples",
                "1",
                "--ao-samples",
                "24",
                "--ao-distance-rel",
                "0.12",
                "--curvature-samples",
                "1",
                "--use-gpu",
                "--meta-json",
                str(bake_meta_json),
            ],
            LOGS_DIR / "prop_bake_4k.log",
            "prop_bake_4k",
            timeout=1200.0,
        )
        report["bake_meta"] = json.loads(bake_meta_json.read_text(encoding="utf-8"))

        renders_dir = OUTPUTS_DIR / "renders"
        render_meta_json = OUTPUTS_DIR / "prop_lato2_v3000_render_meta.json"
        _run_logged(
            [
                str(blender_bin),
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(bake_py),
                "--",
                "render-blender",
                "--mesh",
                str(out_baked_glb),
                "--out-dir",
                str(renders_dir),
                "--prefix",
                "prop_lato2_v3000",
                "--states",
                "wireframe_clay",
                "smooth_lowpoly",
                "normal_mapped",
                "normal_ao_composite",
                "--normal-map",
                str(out_normal),
                "--ao-map",
                str(out_ao),
                "--framing-mode",
                "object",
                "--samples",
                "32",
                "--use-gpu",
                "--meta-json",
                str(render_meta_json),
            ],
            LOGS_DIR / "prop_render_4state.log",
            "prop_render_4state",
            timeout=1200.0,
        )

        crack_cmp_png = renders_dir / "prop_lato2_crack_fix_closeup.png"
        _render_front_crack_comparison(
            blender_bin,
            baseline_old_glb,
            final_clean_glb,
            out_baked_glb,
            out_normal,
            out_ao,
            crack_cmp_png,
        )

        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["total_elapsed_s"] = round(time.monotonic() - t0, 2)
        report["status"] = "ok"
        (OUTPUTS_DIR / "prop_lato2_final_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        dl_manifest = _package_outputs(OUTPUTS_DIR, LOGS_DIR, max_chunk_bytes=int(spec.get("max_chunk_bytes", MAX_CHUNK_BYTES)))
        report["download_manifest"] = dl_manifest
        (OUTPUTS_DIR / "prop_lato2_final_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return report
    except Exception as exc:
        report["status"] = "error"
        report["fatal_error"] = str(exc)
        report["fatal_traceback"] = traceback.format_exc()
        (OUTPUTS_DIR / "prop_lato2_final_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        try:
            _package_outputs(OUTPUTS_DIR, LOGS_DIR)
        except Exception:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Colab GPU Step 2C prop LATO.2 + 4K bake runner")
    ap.add_argument("--spec", required=True)
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    run_prop_s2_pipeline(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
