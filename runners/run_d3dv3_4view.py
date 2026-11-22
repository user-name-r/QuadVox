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
import re
import shutil
import signal
import subprocess
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
    is_transient_error,
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
REMOTE_SCRATCH_DIR = f"{REMOTE_CACHE_ROOT}/scratch"
REMOTE_OUTPUTS_DIR = f"{REMOTE_CACHE_ROOT}/outputs"
REMOTE_SPEC_PATH = f"{REMOTE_TRANSFER_DIR}/job_spec.json"
MAX_CHUNK_BYTES = 10 * 1024 * 1024
DEFAULT_SLICE_S = 180.0
DEFAULT_EXEC_TIMEOUT_S = 240.0
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


def _safe_extract_tar(archive_path: Path, dest_dir: Path) -> list[str]:
    """Safely extract `archive_path` into `dest_dir` (blocking path traversal)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()
    extracted: list[str] = []
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            rel = Path(member.name)
            if rel.is_absolute() or ".." in rel.parts:
                raise RuntimeError(f"Unsafe path in downloaded archive: {member.name}")
            target = (dest_resolved / rel).resolve()
            if not target.is_relative_to(dest_resolved):
                raise RuntimeError(f"Path traversal blocked: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                src_f = tf.extractfile(member)
                if src_f is None:
                    continue
                with src_f, target.open("wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
                extracted.append(rel.as_posix())
    return extracted


def _write_remote_exec_entry(staging_dir: Path, bundle_sha256: str, *, slice_s: float = DEFAULT_SLICE_S) -> Path:
    """Write idempotent remote daemon launcher + sliced log-tail entrypoint."""
    entry_path = staging_dir / "exec_d3dv3_entry.py"
    lines = [
        "import hashlib, os, subprocess, sys, tarfile, time",
        "from pathlib import Path",
        f"transfer = Path({REMOTE_TRANSFER_DIR!r})",
        f"work = Path({REMOTE_WORK_DIR!r})",
        f"logs = Path({REMOTE_LOGS_DIR!r})",
        f"outputs = Path({REMOTE_OUTPUTS_DIR!r})",
        f"scratch = Path({REMOTE_SCRATCH_DIR!r})",
        f"spec_path = Path({REMOTE_SPEC_PATH!r})",
        "for d in (transfer, work, logs, outputs, scratch):",
        "    d.mkdir(parents=True, exist_ok=True)",
        "pid_file = logs / 'd3dv3_worker.pid'",
        "log_file = logs / 'd3dv3_worker.log'",
        "rc_file = logs / 'd3dv3_worker.rc'",
        "pos_file = logs / 'd3dv3_tail.offset'",
        "if not pid_file.is_file() and not rc_file.is_file():",
        "    arc = transfer / 'd3dv3_upload.tar.gz'",
        "    parts = sorted(transfer.glob('d3dv3_upload.tar.gz.part*'))",
        "    if parts:",
        "        with arc.open('wb') as dst:",
        "            for pt in parts:",
        "                dst.write(pt.read_bytes())",
        "    elif not arc.is_file():",
        "        raise FileNotFoundError(f'Missing upload bundle under {transfer}')",
        "    h = hashlib.sha256()",
        "    with arc.open('rb') as fh:",
        "        while True:",
        "            buf = fh.read(1024 * 1024)",
        "            if not buf:",
        "                break",
        "            h.update(buf)",
        f"    if h.hexdigest() != {bundle_sha256!r}:",
        "        raise RuntimeError('Reassembled upload bundle SHA-256 mismatch on remote')",
        "    with tarfile.open(arc, 'r:gz') as tf:",
        "        for m in tf.getmembers():",
        "            rel = Path(m.name)",
        "            if rel.is_absolute() or '..' in rel.parts:",
        "                raise RuntimeError(f'Unsafe archive member: {m.name}')",
        "        tf.extractall(work, filter='data') if hasattr(tarfile, 'data_filter') else tf.extractall(work)",
        "    arc.unlink(missing_ok=True)",
        "    for pt in parts:",
        "        pt.unlink(missing_ok=True)",
        "    launcher = work / '_launch_d3dv3_worker.py'",
        "    launcher_src = (",
        "        'import os, subprocess, sys\\n'",
        "        'from pathlib import Path\\n'",
        "        f'work = Path({str(work)!r})\\n'",
        "        f'log_f = Path({str(log_file)!r})\\n'",
        "        f'rc_f = Path({str(rc_file)!r})\\n'",
        "        f'spec_s = {str(spec_path)!r}\\n'",
        "        'env = os.environ.copy()\\n'",
        "        'prev_pp = env.get(\"PYTHONPATH\", \"\")\\n'",
        "        'env[\"PYTHONPATH\"] = f\"{work}:{prev_pp}\" if prev_pp else str(work)\\n'",
        "        f'env[\"TMPDIR\"] = {str(scratch)!r}\\n'",
        "        f'env[\"TEMP\"] = {str(scratch)!r}\\n'",
        "        f'env[\"TMP\"] = {str(scratch)!r}\\n'",
        "        'env[\"PYTHONUNBUFFERED\"] = \"1\"\\n'",
        "        'env[\"TQDM_DISABLE\"] = \"1\"\\n'",
        "        'cmd = [sys.executable, \"-u\", str(work / \"runners\" / \"colab_d3dv3_4view_worker.py\"), \"--spec\", spec_s]\\n'",
        "        'with log_f.open(\"wb\") as lf:\\n'",
        "        '    p = subprocess.run(cmd, cwd=str(work), env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT)\\n'",
        "        'tmp = rc_f.with_suffix(\".tmp\")\\n'",
        "        'tmp.write_text(str(p.returncode), encoding=\"utf-8\")\\n'",
        "        'tmp.replace(rc_f)\\n'",
        "    )",
        "    launcher.write_text(launcher_src, encoding='utf-8')",
        "    proc = subprocess.Popen(",
        "        [sys.executable, '-u', str(launcher)],",
        "        cwd=str(work),",
        "        stdin=subprocess.DEVNULL,",
        "        stdout=subprocess.DEVNULL,",
        "        stderr=subprocess.DEVNULL,",
        "        start_new_session=True,",
        "    )",
        "    pid_file.write_text(str(proc.pid), encoding='utf-8')",
        "offset = int(pos_file.read_text(encoding='utf-8').strip()) if pos_file.is_file() else 0",
        "t0 = time.monotonic()",
        "last_emit = t0",
        "while True:",
        "    if log_file.is_file() and log_file.stat().st_size > offset:",
        "        with log_file.open('rb') as lf:",
        "            lf.seek(offset)",
        "            chunk = lf.read()",
        "        if chunk:",
        "            offset += len(chunk)",
        "            pos_file.write_text(str(offset), encoding='utf-8')",
        "            sys.stdout.write(chunk.decode('utf-8', errors='replace'))",
        "            sys.stdout.flush()",
        "            last_emit = time.monotonic()",
        "    if rc_file.is_file():",
        "        rc_val = int(rc_file.read_text(encoding='utf-8').strip())",
        "        print(f'[d3dv3-tail] WORKER_RC={rc_val}', flush=True)",
        "        sys.exit(0)",
        f"    if time.monotonic() - t0 >= {float(slice_s)!r}:",
        "        print('[d3dv3-tail] SLICE_CONTINUE=1', flush=True)",
        "        sys.exit(0)",
        "    if time.monotonic() - last_emit >= 25.0:",
        "        print('[d3dv3-tail] heartbeat...', flush=True)",
        "        last_emit = time.monotonic()",
        "    time.sleep(2.0)",
    ]
    entry_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return entry_path


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
    remote_timeout_s: float = 3600.0,
    compare_crevice_keep: bool = False,
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
        extra_spec={"compare_crevice_keep": bool(compare_crevice_keep)},
    )
    entry_py = _write_remote_exec_entry(staging_dir, bundle["spec"]["bundle_sha256"])

    if dry_run:
        summary = {
            "dry_run": True,
            "profile": profile_name,
            "compare_crevice_keep": bool(compare_crevice_keep),
            "bundle_bytes": bundle["bundle_path"].stat().st_size,
            "chunk_count": len(bundle["chunks"]),
            "staging_dir": str(staging_dir),
            "entry_script": str(entry_py),
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
        upload_files_resilient(
            cb,
            session_name,
            upload_paths,
            REMOTE_TRANSFER_DIR,
            staging_dir=staging_dir,
            repo_root=WORKSPACE_ROOT,
        )

        t_exec0 = time.monotonic()
        transient_failures = 0
        while True:
            elapsed = time.monotonic() - t_exec0
            if elapsed > float(remote_timeout_s):
                raise TimeoutError(
                    f"Colab D3D-V3 worker exceeded outer wall-clock limit ({remote_timeout_s:.0f}s)"
                )
            cmd = [
                cb,
                "exec",
                "-s",
                session_name,
                "--timeout",
                str(int(DEFAULT_EXEC_TIMEOUT_S)),
                "-f",
                str(entry_py),
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=DEFAULT_EXEC_TIMEOUT_S + 30.0,
                    env=colab_assign_env(),
                )
            except subprocess.TimeoutExpired as exc:
                transient_failures += 1
                if transient_failures > 6:
                    raise TimeoutError("Colab exec slice timed out repeatedly while tailing daemon") from exc
                time.sleep(min(10.0, 2.0 * transient_failures))
                continue

            if proc.stdout:
                print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
            if proc.returncode != 0:
                combined = f"{proc.stdout}\n{proc.stderr}"
                if is_transient_error(combined) and transient_failures < 8:
                    transient_failures += 1
                    time.sleep(min(15.0, 2.0 * transient_failures))
                    continue
                raise RuntimeError(
                    f"Colab D3D-V3 tail failed (rc={proc.returncode}):\n{combined[-1500:]}"
                )

            m_rc = re.search(r"\[d3dv3-tail\]\s+WORKER_RC=(-?\d+)", proc.stdout or "")
            if m_rc:
                worker_rc = int(m_rc.group(1))
                if worker_rc != 0:
                    raise RuntimeError(f"Remote colab_d3dv3_4view_worker failed with rc={worker_rc}")
                break
            if "[d3dv3-tail] SLICE_CONTINUE=1" in (proc.stdout or ""):
                transient_failures = 0
                continue
            transient_failures += 1
            if transient_failures > 8:
                raise RuntimeError(
                    "Colab D3D-V3 exec exited rc=0 without [d3dv3-tail] sentinel after retries"
                )
            time.sleep(2.0)

        dl_chunks_dir = staging_dir / "download_chunks"
        archive_local, _manifest, _transport = download_manifest_and_chunks_resilient(
            cb,
            session_name,
            REMOTE_OUTPUTS_DIR,
            "output_manifest.json",
            dl_chunks_dir,
            staging_dir=staging_dir,
            repo_root=WORKSPACE_ROOT,
        )
        dl_dir = out_dir / "colab_out"
        extracted_files = _safe_extract_tar(archive_local, dl_dir)
        report_path = dl_dir / "d3dv3_report.json"
        report_data = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        host_summary = {
            "gpu_sku": sku,
            "session": session_name,
            "profile": profile_name,
            "compare_crevice_keep": bool(compare_crevice_keep),
            "out_dir": str(dl_dir),
            "extracted_files": extracted_files,
            "download_transport": _transport,
            "download_chunk_count": len(_manifest.get("chunks", [])),
            "download_total_bytes": int(_manifest.get("total_size", 0)),
            "download_bundle_sha256": str(_manifest.get("bundle_sha256", _manifest.get("sha256", ""))),
            "report": report_data,
        }
        (out_dir / "d3dv3_host_summary.json").write_text(
            json.dumps(host_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        return host_summary
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
    ap.add_argument("--remote-timeout-s", type=float, default=3600.0)
    ap.add_argument("--compare-crevice-keep", action="store_true")
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
        remote_timeout_s=float(args.remote_timeout_s),
        compare_crevice_keep=bool(args.compare_crevice_keep),
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
