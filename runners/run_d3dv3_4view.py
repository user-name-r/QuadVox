"""Host thin-client Colab orchestrator for the 4-View Direct3D-V3 1536^3 Voxel Fusion Pipeline.

Supports both:
- Layer 1 (Universal Core): arbitrary 4-view orthographic objects/props/organic assets
- Layer 2 (Configurable Switches): YAML-driven `front_roi_box`, `coarse_splice`, and `smoothing_zones`

Uses `quadvox.cloud.colab_resilience` (80s/95s aligned assign, orphan `unassign`, SSH/rsync binary transfer)
and never hardcodes `/tmp` or machine-specific home paths.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import py_compile
import shutil
import signal
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Sequence

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from quadvox.cloud.colab_resilience import (
    allocate_gpu_session_fast,
    cleanup_session_and_orphans,
    colab_assign_env,
    download_manifest_and_chunks_resilient,
    sha256_file,
    upload_files_resilient,
)
from quadvox.contracts import load_profile_config
from quadvox.tools.external_tools import resolve_colab_bin

DEFAULT_GPU_ORDER: tuple[str, ...] = ("A100", "H100", "G4")
REMOTE_CACHE_ROOT = "/content/cache"
REMOTE_WORK_DIR = f"{REMOTE_CACHE_ROOT}/work"
REMOTE_TRANSFER_DIR = f"{REMOTE_CACHE_ROOT}/transfer"
REMOTE_LOGS_DIR = f"{REMOTE_CACHE_ROOT}/logs"
REMOTE_SPEC_PATH = f"{REMOTE_TRANSFER_DIR}/job_spec.json"
MAX_CHUNK_BYTES = 10 * 1024 * 1024
_SYS_TEMP_ROOT = os.path.sep + "tmp"


def _assert_not_system_temp(path: Path | str) -> None:
    raw = os.path.normpath(str(path).strip())
    if raw == _SYS_TEMP_ROOT or raw.startswith(_SYS_TEMP_ROOT + os.path.sep):
        raise ValueError(f"Refusing system temp path {raw!r}; use workspace runs/ instead.")


def scan_no_hardcoded_tmp(paths: Sequence[Path]) -> list[str]:
    """AST-scan Python files to verify zero hardcoded system temp path literals."""
    violations: list[str] = []
    for p in paths:
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value.strip()
                if val == _SYS_TEMP_ROOT or val.startswith(_SYS_TEMP_ROOT + os.path.sep):
                    violations.append(f"{p}:{getattr(node, 'lineno', 0)}: {val!r}")
    return violations


def build_d3dv3_upload_bundle(
    *,
    staging_dir: Path,
    profile_name: str,
    coarse_path: Path,
    secondary_coarse_path: Path | None,
    view_paths: dict[str, Path],
    extra_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Package `quadvox/`, `configs/`, worker script, and inputs into SHA256-verified tar chunks."""
    _assert_not_system_temp(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    prof = load_profile_config(profile_name)

    quadvox_dir = WORKSPACE_ROOT / "quadvox"
    configs_dir = WORKSPACE_ROOT / "configs"
    worker_py = WORKSPACE_ROOT / "runners" / "colab_d3dv3_4view_worker.py"
    for req in (quadvox_dir, configs_dir, worker_py, coarse_path):
        if not req.exists():
            raise FileNotFoundError(f"Required bundle source missing: {req}")
    for vn, vp in view_paths.items():
        if not vp.is_file():
            raise FileNotFoundError(f"Missing {vn} view image: {vp}")

    bundle_path = staging_dir / "d3dv3_upload.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tf:
        for item in sorted(quadvox_dir.rglob("*")):
            if "__pycache__" in item.parts or item.suffix == ".pyc" or not item.is_file():
                continue
            tf.add(item, arcname=item.relative_to(WORKSPACE_ROOT).as_posix(), recursive=False)
        for item in sorted(configs_dir.rglob("*")):
            if item.is_file():
                tf.add(item, arcname=item.relative_to(WORKSPACE_ROOT).as_posix(), recursive=False)
        tf.add(worker_py, arcname="runners/colab_d3dv3_4view_worker.py", recursive=False)
        tf.add(coarse_path, arcname=f"inputs/coarse{coarse_path.suffix}", recursive=False)
        if secondary_coarse_path is not None:
            tf.add(secondary_coarse_path, arcname=f"inputs/secondary_coarse{secondary_coarse_path.suffix}", recursive=False)
        for vn, vp in view_paths.items():
            tf.add(vp, arcname=f"inputs/{vn}{vp.suffix}", recursive=False)

    # Split into <= MAX_CHUNK_BYTES parts
    chunks: list[dict[str, Any]] = []
    idx = 0
    with bundle_path.open("rb") as src:
        while True:
            data = src.read(MAX_CHUNK_BYTES)
            if not data:
                break
            part_name = f"d3dv3_upload.tar.gz.part{idx:03d}"
            part_path = staging_dir / part_name
            part_path.write_bytes(data)
            chunks.append({"name": part_name, "bytes": len(data), "sha256": sha256_file(part_path)})
            idx += 1

    spec: dict[str, Any] = {
        "profile": prof.to_dict(),
        "coarse_rel": f"inputs/coarse{coarse_path.suffix}",
        "secondary_coarse_rel": f"inputs/secondary_coarse{secondary_coarse_path.suffix}" if secondary_coarse_path else None,
        "view_rels": {vn: f"inputs/{vn}{vp.suffix}" for vn, vp in view_paths.items()},
        "chunks": chunks,
        "bundle_sha256": sha256_file(bundle_path),
    }
    if extra_spec:
        spec.update(extra_spec)
    (staging_dir / "job_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return {"bundle_path": bundle_path, "spec": spec, "chunks": chunks}


def run_colab_d3dv3(
    *,
    out_dir: Path,
    profile_name: str,
    coarse_path: Path,
    secondary_coarse_path: Path | None,
    view_paths: dict[str, Path],
    gpu_order: Sequence[str] = DEFAULT_GPU_ORDER,
    colab_bin: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or dry-run-package 4-view Direct3D-V3 1536^3 fusion on Google Colab."""
    _assert_not_system_temp(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = out_dir / "_colab_staging"

    worker_py = WORKSPACE_ROOT / "runners" / "colab_d3dv3_4view_worker.py"
    fusion_py = WORKSPACE_ROOT / "quadvox" / "s1_geometry" / "d3dv3_fusion.py"
    for f in (Path(__file__), worker_py, fusion_py):
        py_compile.compile(str(f), doraise=True)
    viol = scan_no_hardcoded_tmp([Path(__file__), worker_py, fusion_py])
    if viol:
        raise ValueError("Hardcoded temp paths detected:\n" + "\n".join(viol))

    bundle = build_d3dv3_upload_bundle(
        staging_dir=staging_dir,
        profile_name=profile_name,
        coarse_path=coarse_path,
        secondary_coarse_path=secondary_coarse_path,
        view_paths=view_paths,
    )

    if dry_run:
        summary = {
            "dry_run": True,
            "profile": profile_name,
            "bundle_bytes": bundle["bundle_path"].stat().st_size,
            "chunk_count": len(bundle["chunks"]),
            "staging_dir": str(staging_dir),
        }
        (out_dir / "d3dv3_dry_run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    cb = resolve_colab_bin(colab_bin)
    session_name = f"quadvox-d3dv3-{int(time.time())}"
    sku, pre_eps = allocate_gpu_session_fast(cb, session_name, gpu_order)
    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        upload_paths = [staging_dir / "job_spec.json"] + [staging_dir / c["name"] for c in bundle["chunks"]]
        upload_files_resilient(cb, session_name, upload_paths, REMOTE_TRANSFER_DIR, repo_root=WORKSPACE_ROOT)

        import subprocess

        remote_cmd = (
            f"mkdir -p {REMOTE_WORK_DIR} {REMOTE_LOGS_DIR} && "
            f"cat {REMOTE_TRANSFER_DIR}/d3dv3_upload.tar.gz.part* > {REMOTE_TRANSFER_DIR}/d3dv3_upload.tar.gz && "
            f"tar -xzf {REMOTE_TRANSFER_DIR}/d3dv3_upload.tar.gz -C {REMOTE_WORK_DIR} && "
            f"cd {REMOTE_WORK_DIR} && PYTHONPATH={REMOTE_WORK_DIR} "
            f"python3 runners/colab_d3dv3_4view_worker.py --spec {REMOTE_SPEC_PATH}"
        )
        subprocess.run(
            [cb, "exec", "-s", session_name, "--timeout", "300", "--", "bash", "-lc", remote_cmd],
            check=True,
            stdin=subprocess.DEVNULL,
            env=colab_assign_env(),
        )
        dl_dir = out_dir / "colab_out"
        download_manifest_and_chunks_resilient(
            cb,
            session_name,
            f"{REMOTE_CACHE_ROOT}/outputs/output_manifest.json",
            dl_dir,
            repo_root=WORKSPACE_ROOT,
        )
        return {"gpu_sku": sku, "session": session_name, "out_dir": str(dl_dir)}
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        cleanup_session_and_orphans(cb, session_name, pre_eps)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Colab 4-View Direct3D-V3 1536^3 Voxel Fusion Orchestrator.")
    ap.add_argument("--profile", default="object", help="Built-in profile, Layer-2 preset, or YAML path.")
    ap.add_argument("--coarse", required=True, help="Primary coarse mesh (.glb/.obj/.ply).")
    ap.add_argument("--secondary-coarse", default=None, help="Optional Layer-2 secondary coarse mesh for OpenVDB splice.")
    ap.add_argument("--front", required=True)
    ap.add_argument("--left", required=True)
    ap.add_argument("--back", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--out-dir", default="runs/d3dv3_4view")
    ap.add_argument("--gpu-order", default="A100,H100,G4")
    ap.add_argument("--colab-bin", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    res = run_colab_d3dv3(
        out_dir=Path(args.out_dir),
        profile_name=args.profile,
        coarse_path=Path(args.coarse),
        secondary_coarse_path=Path(args.secondary_coarse) if args.secondary_coarse else None,
        view_paths={
            "front": Path(args.front),
            "left": Path(args.left),
            "back": Path(args.back),
            "right": Path(args.right),
        },
        gpu_order=[g.strip().upper() for g in args.gpu_order.split(",") if g.strip()],
        colab_bin=args.colab_bin,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
