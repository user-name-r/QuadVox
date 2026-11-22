"""Single-session Colab GPU runner for S2.5 4K Normal+AO+Curvature Cycles baking & 4-state HD rendering.

Strictly executes inside `/content/cache` on a Colab GPU VM (`A100` / `H100` / `G4`).
Zero system temp paths; chunks downloadable outputs into <= 6 MiB SHA256-verified bundles:
- Bundle 1: JSON reports, 4K 2x2 contact sheets, rendered 6-view sheets & head/close-up previews.
- Bundle 2: Full-resolution 4096x4096 Normal + AO + Curvature PNGs and self-contained baked 4K GLBs.
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
TOOLS_DIR = CACHE_ROOT / "tools"

MAX_CHUNK_BYTES = 6 * 1024 * 1024
DEFAULT_BLENDER_URL = "https://download.blender.org/release/Blender5.3/blender-5.3.0-linux-x64.tar.xz"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(1024 * 1024)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def ensure_blender_linux(blender_url: str = DEFAULT_BLENDER_URL) -> Path:
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    cached = TOOLS_DIR / "blender-5.3.0-linux-x64" / "blender"
    if cached.is_file():
        return cached
    for p in sorted(TOOLS_DIR.glob("blender-*/blender")):
        if p.is_file():
            return p

    arc = TOOLS_DIR / "blender-5.3.0-linux-x64.tar.xz"
    if not arc.is_file():
        print(f"[setup] Downloading Blender 5.3.0 from {blender_url}...", flush=True)
        urllib.request.urlretrieve(blender_url, str(arc))
    print(f"[setup] Extracting {arc.name}...", flush=True)
    with tarfile.open(arc, "r:xz") as tf:
        tf.extractall(TOOLS_DIR)
    arc.unlink(missing_ok=True)
    for p in sorted(TOOLS_DIR.glob("blender-*/blender")):
        if p.is_file():
            return p
    raise FileNotFoundError(f"Blender binary missing after extracting {blender_url}")


def _run_logged(
    cmd: list[str],
    log_path: Path,
    stage: str,
    *,
    cwd: Path,
    timeout: float = 1800.0,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    prev_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{WORK_DIR}:{prev_pp}" if prev_pp else str(WORK_DIR)
    env["TMPDIR"] = str(SCRATCH_DIR)
    env["TEMP"] = str(SCRATCH_DIR)
    env["TMP"] = str(SCRATCH_DIR)
    env["TQDM_DISABLE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[{stage}] Running: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
        )
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"[{stage}] failed with rc={proc.returncode}:\n{tail}")
    print(f"[{stage}] OK in {time.perf_counter() - t0:.2f}s", flush=True)


def split_file_chunks(
    archive_path: Path,
    out_dir: Path,
    *,
    max_chunk_bytes: int,
    manifest_name: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_size = archive_path.stat().st_size
    full_hasher = hashlib.sha256()
    chunks_meta: list[dict[str, Any]] = []
    if total_size <= max_chunk_bytes:
        full_sha = _sha256_file(archive_path)
        dst = out_dir / archive_path.name
        if dst.resolve() != archive_path.resolve():
            shutil.copy2(archive_path, dst)
        chunks_meta.append({"name": dst.name, "size": total_size, "sha256": full_sha})
    else:
        idx = 0
        io_block = 1024 * 1024
        with archive_path.open("rb") as src:
            while True:
                remaining = int(max_chunk_bytes)
                part_name = f"{archive_path.name}.part{idx:03d}"
                part_path = out_dir / part_name
                ph = hashlib.sha256()
                part_size = 0
                with part_path.open("wb") as dst:
                    while remaining > 0:
                        buf = src.read(min(io_block, remaining))
                        if not buf:
                            break
                        dst.write(buf)
                        ph.update(buf)
                        full_hasher.update(buf)
                        part_size += len(buf)
                        remaining -= len(buf)
                if part_size == 0:
                    part_path.unlink(missing_ok=True)
                    break
                chunks_meta.append({"name": part_name, "size": part_size, "sha256": ph.hexdigest()})
                idx += 1
        full_sha = full_hasher.hexdigest()

    manifest = {
        "archive_name": archive_path.name,
        "total_size": total_size,
        "sha256": full_sha,
        "max_chunk_bytes": int(max_chunk_bytes),
        "chunks": chunks_meta,
    }
    (out_dir / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def package_dual_bundles(outputs_dir: Path, logs_dir: Path, *, max_chunk_bytes: int) -> dict[str, Any]:
    TRANSFER_DIR.mkdir(parents=True, exist_ok=True)
    b1_files: list[tuple[Path, str]] = []
    b2_files: list[tuple[Path, str]] = []

    for fpath in sorted(outputs_dir.rglob("*")):
        if not fpath.is_file():
            continue
        rel = fpath.relative_to(outputs_dir)
        name = fpath.name.lower()
        is_heavy = (
            name.endswith("_normal_4k.png")
            or name.endswith("_ao_4k.png")
            or name.endswith("_curvature_4k.png")
            or name.endswith("_baked_4k.glb")
            or name.endswith(".glb")
        )
        # Keep baked_4k.glb in Bundle 2; keep compact sheets/head previews/JSONs in Bundle 1
        if is_heavy:
            b2_files.append((fpath, f"outputs/{rel.as_posix()}"))
        else:
            b1_files.append((fpath, f"outputs/{rel.as_posix()}"))

    for lpath in sorted(logs_dir.glob("*.log")):
        if lpath.is_file():
            b1_files.append((lpath, f"logs/{lpath.name}"))

    b1_arc = SCRATCH_DIR / "bundle1_reports_previews.tar.gz"
    with tarfile.open(b1_arc, "w:gz") as tf:
        for src, arcname in b1_files:
            tf.add(src, arcname=arcname, recursive=False)
    b1_manifest = split_file_chunks(b1_arc, TRANSFER_DIR, max_chunk_bytes=max_chunk_bytes, manifest_name="bundle1_manifest.json")
    b1_arc.unlink(missing_ok=True)

    b2_arc = SCRATCH_DIR / "bundle2_4k_assets.tar.gz"
    with tarfile.open(b2_arc, "w:gz") as tf:
        for src, arcname in b2_files:
            tf.add(src, arcname=arcname, recursive=False)
    b2_manifest = split_file_chunks(b2_arc, TRANSFER_DIR, max_chunk_bytes=max_chunk_bytes, manifest_name="bundle2_manifest.json")
    b2_arc.unlink(missing_ok=True)

    index = {
        "bundle1": {"manifest_name": "bundle1_manifest.json", "file_count": len(b1_files), **b1_manifest},
        "bundle2": {"manifest_name": "bundle2_manifest.json", "file_count": len(b2_files), **b2_manifest},
    }
    (TRANSFER_DIR / "download_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def run_uv_bake_pipeline(spec: dict[str, Any]) -> dict[str, Any]:
    t0 = time.monotonic()
    for d in (WORK_DIR, LOGS_DIR, OUTPUTS_DIR, TRANSFER_DIR, SCRATCH_DIR, TOOLS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    os.environ["TMPDIR"] = str(SCRATCH_DIR)
    os.environ["TEMP"] = str(SCRATCH_DIR)
    os.environ["TMP"] = str(SCRATCH_DIR)

    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    print(f"[colab] GPU: {gpu_info}", flush=True)

    blender_bin = ensure_blender_linux(spec.get("blender_url", DEFAULT_BLENDER_URL))
    print(f"[colab] Blender: {blender_bin}", flush=True)
    bake_py = WORK_DIR / "quadvox" / "s2_retopo" / "slim_uv_bake.py"

    report: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_info,
        "blender_bin": str(blender_bin),
        "jobs": {},
        "errors": [],
    }

    try:
        for job in spec.get("jobs", []):
            name = str(job["name"])
            j_out = OUTPUTS_DIR / name
            j_out.mkdir(parents=True, exist_ok=True)
            high_mesh = WORK_DIR / job["high_mesh"]
            low_mesh = WORK_DIR / job["low_uv_mesh"]
            uv_preview = (WORK_DIR / job["uv_preview_png"]) if job.get("uv_preview_png") else None
            tex_size = int(job.get("tex_size", 4096))
            padding_px = int(job.get("padding_px", 16))
            ao_samples = int(job.get("ao_samples", 24))
            render_samples = int(job.get("render_samples", 32))
            framing_mode = str(job.get("framing_mode", "character"))

            out_normal = j_out / f"{name}_normal_4k.png"
            out_ao = j_out / f"{name}_ao_4k.png"
            out_curv = j_out / f"{name}_curvature_4k.png"
            out_glb = j_out / f"{name}_baked_4k.glb"
            out_sheet = j_out / f"{name}_maps_4096_sheet.png"
            meta_json = j_out / f"{name}_bake_meta.json"
            render_dir = j_out / "renders"
            render_meta = j_out / f"{name}_render_meta.json"

            # Copy UV preview & unwrap JSON into output folder so everything sits cleanly together
            if uv_preview is not None and uv_preview.is_file():
                shutil.copy2(uv_preview, j_out / f"{name}_uv_layout.png")
            if job.get("unwrap_json"):
                uj = WORK_DIR / job["unwrap_json"]
                if uj.is_file():
                    shutil.copy2(uj, j_out / f"{name}_unwrap.json")

            bake_cmd = [
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
                str(high_mesh),
                "--low",
                str(low_mesh),
                "--out-normal",
                str(out_normal),
                "--out-ao",
                str(out_ao),
                "--out-curvature",
                str(out_curv),
                "--out-glb",
                str(out_glb),
                "--out-sheet",
                str(out_sheet),
                "--tex-size",
                str(tex_size),
                "--padding-px",
                str(padding_px),
                "--normal-samples",
                "1",
                "--ao-samples",
                str(ao_samples),
                "--ao-distance-rel",
                "0.12",
                "--curvature-samples",
                "1",
                "--use-gpu",
                "--meta-json",
                str(meta_json),
            ]
            if uv_preview is not None and uv_preview.is_file():
                bake_cmd.extend(["--uv-preview-png", str(uv_preview)])

            _run_logged(bake_cmd, LOGS_DIR / f"{name}_bake.log", f"{name}:bake_4k", cwd=WORK_DIR, timeout=1200.0)

            render_cmd = [
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
                str(out_glb),
                "--out-dir",
                str(render_dir),
                "--prefix",
                name,
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
                framing_mode,
                "--samples",
                str(render_samples),
                "--use-gpu",
                "--meta-json",
                str(render_meta),
            ]
            _run_logged(render_cmd, LOGS_DIR / f"{name}_render.log", f"{name}:render_4state", cwd=WORK_DIR, timeout=1200.0)

            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            report["jobs"][name] = {
                "status": "ok",
                "ray_miss_ratio": meta.get("ray_miss_ratio"),
                "backface_hit_ratio": meta.get("backface_hit_ratio"),
                "ao_mean": meta.get("ao_mean"),
                "curvature_source": meta.get("curvature_source"),
                "curvature_dynamic_range": meta.get("curvature_dynamic_range"),
                "bake_elapsed_s": meta.get("bake_elapsed_s"),
                "device_used": meta.get("device_used"),
            }
            print(
                f"[{name}] COMPLETE: ray_miss={meta.get('ray_miss_ratio', -1.0)*100:.3f}%, "
                f"backface={meta.get('backface_hit_ratio', -1.0)*100:.3f}%, "
                f"device={meta.get('device_used')}",
                flush=True,
            )

        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["total_elapsed_s"] = round(time.monotonic() - t0, 2)
        (OUTPUTS_DIR / "colab_uv_bake_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report["download_bundles"] = package_dual_bundles(
            OUTPUTS_DIR, LOGS_DIR, max_chunk_bytes=int(spec.get("max_chunk_bytes", MAX_CHUNK_BYTES))
        )
        (OUTPUTS_DIR / "colab_uv_bake_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return report
    except Exception as exc:
        report["fatal_error"] = str(exc)
        report["fatal_traceback"] = traceback.format_exc()
        (OUTPUTS_DIR / "colab_uv_bake_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        try:
            package_dual_bundles(OUTPUTS_DIR, LOGS_DIR, max_chunk_bytes=MAX_CHUNK_BYTES)
        except Exception:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Colab GPU 4K Bake & Render runner for S2.5 low-island UV.")
    ap.add_argument("--spec", required=True)
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    run_uv_bake_pipeline(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
