"""Local Colab orchestrator for S1 Pixal3D benchmarking and production runs.

Strictly follows `colab-operator` rules:
- Uses `colab` (never mounts Google Drive, never hardcodes `/tmp`).
- Single-session batching with `try/finally` + signal cleanup guaranteeing `colab stop`,
  empty `colab sessions`, and `Usage rate: 0.00/hr` on exit.
- GPU fallback chain `G4 -> A100 -> H100` (refuses silent CPU fallback).
- Finite `--timeout 300` (never `0`) with `TQDM_DISABLE=1` and `HF_HUB_DISABLE_PROGRESS_BARS=1`.
- 15 MB chunked gzip upload/download guard with SHA256 verification on reassembly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Sequence

from quadvox.contracts import JobConfig, RunMeta
from quadvox.s0_views.preprocess import discover_input_views

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from quadvox.cloud.colab_resilience import (  # noqa: E402
    allocate_gpu_session_fast,
    cleanup_session_and_orphans,
    colab_assign_env,
    download_manifest_and_chunks_resilient,
    upload_files_resilient,
)


from quadvox.tools.external_tools import resolve_colab_bin
DEFAULT_COLAB_BIN = resolve_colab_bin()
REMOTE_CACHE_ROOT = "/content/cache"
REMOTE_WORK_DIR = f"{REMOTE_CACHE_ROOT}/work"
REMOTE_TRANSFER_DIR = f"{REMOTE_CACHE_ROOT}/transfer"
REMOTE_SPEC_PATH = f"{REMOTE_TRANSFER_DIR}/job_spec.json"
MAX_CHUNK_BYTES = 15 * 1024 * 1024  # Strictly under Colab CLI's 20 MB limit.


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(1024 * 1024)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


_SYS_TEMP_ROOT = os.path.sep + "tmp"


def _assert_not_system_temp(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved == _SYS_TEMP_ROOT or resolved.startswith(_SYS_TEMP_ROOT + os.path.sep):
        raise ValueError(f"Refusing system temp path {resolved}; use workspace runs/ instead.")


def _collect_input_files(input_path: Path, input_mode: str) -> tuple[list[Path], Path | None]:
    """Collect ordered input image files and optional `transforms.json` for upload."""
    views = discover_input_views(input_path, mode=input_mode)
    transforms_file = (input_path / "transforms.json") if input_path.is_dir() else None
    if transforms_file is not None and not transforms_file.is_file():
        transforms_file = None
    return views, transforms_file


def _split_file_chunks(
    archive_path: Path,
    staging_dir: Path,
    max_chunk_bytes: int = MAX_CHUNK_BYTES,
) -> dict[str, Any]:
    """Split `archive_path` into <=`max_chunk_bytes` chunks with SHA256 manifest."""
    total_size = archive_path.stat().st_size
    full_sha = _sha256_file(archive_path)
    chunks_meta: list[dict[str, Any]] = []

    if total_size <= max_chunk_bytes:
        chunks_meta.append(
            {
                "name": archive_path.name,
                "size": total_size,
                "sha256": full_sha,
                "local_path": str(archive_path),
            }
        )
    else:
        idx = 0
        with archive_path.open("rb") as src:
            while True:
                buf = src.read(max_chunk_bytes)
                if not buf:
                    break
                part_name = f"{archive_path.name}.part{idx:03d}"
                part_path = staging_dir / part_name
                part_path.write_bytes(buf)
                chunks_meta.append(
                    {
                        "name": part_name,
                        "size": len(buf),
                        "sha256": hashlib.sha256(buf).hexdigest(),
                        "local_path": str(part_path),
                    }
                )
                idx += 1

    manifest = {
        "archive_name": archive_path.name,
        "total_size": total_size,
        "sha256": full_sha,
        "max_chunk_bytes": max_chunk_bytes,
        "chunks": [
            {"name": c["name"], "size": c["size"], "sha256": c["sha256"]}
            for c in chunks_meta
        ],
    }
    manifest_path = staging_dir / "upload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "chunk_paths": [Path(c["local_path"]) for c in chunks_meta],
    }


def _build_upload_bundle(
    job: JobConfig,
    staging_dir: Path,
    *,
    is_benchmark: bool,
    project_back: float,
    cellocut_resolution: int = 1024,
    cellocut_fill: float = 20.0,
    cellocut_band: float = 1.0,
    cellocut_qef: bool = False,
    cellocut_smooth_iters: int = 0,
    smooth_iters: int | None = None,
    min_component_area_rel: float = 0.005,
    weld_eps_rel: float = 1e-10,
    asset_type: str = "character",
) -> dict[str, Any]:
    """Package `quadvox/`, `runners/colab_s1_pixal3d_worker.py`, and input views <=15 MB/chunk."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = staging_dir / "upload_bundle.tar.gz"

    quadvox_dir = WORKSPACE_ROOT / "quadvox"
    runner_script = WORKSPACE_ROOT / "runners" / "colab_s1_pixal3d_worker.py"
    if not quadvox_dir.is_dir():
        raise FileNotFoundError(f"Missing quadvox package at {quadvox_dir}")
    if not runner_script.is_file():
        raise FileNotFoundError(f"Missing remote runner at {runner_script}")

    input_path = Path(job.input_path)
    views, transforms_file = _collect_input_files(input_path, job.mode)

    with tarfile.open(bundle_path, "w:gz") as tf:
        for py_item in sorted(quadvox_dir.rglob("*")):
            if "__pycache__" in py_item.parts or py_item.suffix == ".pyc":
                continue
            tf.add(
                py_item,
                arcname=str(Path("quadvox") / py_item.relative_to(quadvox_dir)),
                recursive=False,
            )

        tf.add(runner_script, arcname="runners/colab_s1_pixal3d_worker.py", recursive=False)
        for vpath in views:
            tf.add(vpath, arcname=f"inputs/{vpath.name}", recursive=False)
        if transforms_file is not None:
            tf.add(transforms_file, arcname="inputs/transforms.json", recursive=False)

    split_info = _split_file_chunks(bundle_path, staging_dir)
    resolved_smooth_iters = int(
        smooth_iters if smooth_iters is not None else cellocut_smooth_iters
    )

    spec = {
        "mode": "benchmark" if is_benchmark else "run",
        "asset_type": str(asset_type),
        "input_mode": str(job.mode).lower(),
        "backend": str(job.backend),
        "repair_method": str(job.repair_method),
        "remesh_resolution": int(cellocut_resolution),
        "remesh_fill": float(cellocut_fill),
        "remesh_band": float(cellocut_band),
        "project_back": float(project_back),
        "cellocut_qef": bool(cellocut_qef),
        "smooth_iters": resolved_smooth_iters,
        "min_component_area_rel": float(min_component_area_rel),
        "weld_eps_rel": float(weld_eps_rel),
        "resolution": int(job.resolution),
        "seed": int(job.seed),
        "seeds": int(job.seeds),
        "seed_list": list(job.seed_list) if job.seed_list is not None else None,
        "fov_deg": float(job.fov_deg),
        "pad_factor": float(job.pad_factor),
        "target_height": float(job.target_height),
        "raw_views_dir": f"{REMOTE_WORK_DIR}/inputs",
    }
    spec_path = staging_dir / "job_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    init_dirs_script = staging_dir / "init_dirs.py"
    init_dirs_script.write_text(
        "import os, shutil, subprocess\n"
        f"for sub in ('work', 'transfer', 'logs', 'models', 'outputs'):\n"
        f"    os.makedirs(f'{REMOTE_CACHE_ROOT}/{{sub}}', exist_ok=True)\n"
        "out = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True)\n"
        "print('[colab-init] GPU:', out.strip(), flush=True)\n"
        "print('[colab-init] RSYNC=' + ('1' if shutil.which('rsync') else '0'), flush=True)\n",
        encoding="utf-8",
    )

    exec_entry_script = staging_dir / "exec_entry.py"
    exec_entry_script.write_text(
        "import hashlib, json, os, subprocess, sys, tarfile\n"
        "from pathlib import Path\n"
        f"transfer = Path({REMOTE_TRANSFER_DIR!r})\n"
        f"work = Path({REMOTE_WORK_DIR!r})\n"
        "work.mkdir(parents=True, exist_ok=True)\n"
        "manifest = json.loads((transfer / 'upload_manifest.json').read_text(encoding='utf-8'))\n"
        "archive_path = transfer / manifest['archive_name']\n"
        "chunks = manifest['chunks']\n"
        "if len(chunks) > 1:\n"
        "    with archive_path.open('wb') as dst:\n"
        "        for c in chunks:\n"
        "            part = transfer / c['name']\n"
        "            raw = part.read_bytes()\n"
        "            if hashlib.sha256(raw).hexdigest() != c['sha256']:\n"
        "                raise RuntimeError(f\"SHA256 mismatch on upload chunk {c['name']}\")\n"
        "            dst.write(raw)\n"
        "actual_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()\n"
        "if actual_sha != manifest['sha256']:\n"
        "    raise RuntimeError('Upload archive SHA256 mismatch after reassembly')\n"
        "with tarfile.open(archive_path, 'r:gz') as tf:\n"
        "    if hasattr(tarfile, 'data_filter'):\n"
        "        tf.extractall(work, filter='data')\n"
        "    else:\n"
        "        tf.extractall(work)\n"
        "runner = work / 'runners' / 'colab_s1_pixal3d_worker.py'\n"
        "env = os.environ.copy()\n"
        "prev_pp = env.get('PYTHONPATH', '')\n"
        "env['PYTHONPATH'] = f'{work}:{prev_pp}' if prev_pp else str(work)\n"
        f"cmd = [sys.executable, '-u', str(runner), '--spec', {REMOTE_SPEC_PATH!r}]\n"
        "proc = subprocess.Popen(cmd, cwd=str(work), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
        "assert proc.stdout is not None\n"
        "for line in proc.stdout:\n"
        "    print(line, end='', flush=True)\n"
        "rc = proc.wait()\n"
        "if rc != 0:\n"
        "    raise RuntimeError(f'colab_s1_pixal3d_worker failed with exit code {rc}')\n"
        "print('[colab-exec-entry] ' + 'STATUS=OK', flush=True)\n",
        encoding="utf-8",
    )

    return {
        "spec": spec,
        "spec_path": spec_path,
        "init_dirs_script": init_dirs_script,
        "exec_entry_script": exec_entry_script,
        **split_info,
    }


def _run_colab(
    colab_bin: str,
    args: Sequence[str],
    *,
    stream: bool = False,
    check: bool = True,
    timeout_s: float | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [colab_bin, *args]
    env = colab_assign_env()
    if not stream:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        return proc

    lines: list[str] = []
    proc_p = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc_p.stdout is not None
    for line in proc_p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    rc = proc_p.wait(timeout=timeout_s)
    out_text = "".join(lines)
    if check and rc != 0:
        raise RuntimeError(
            f"Command failed (rc={rc}): {' '.join(cmd)}\noutput_tail:\n"
            + "\n".join(lines[-50:])
        )
    return subprocess.CompletedProcess(cmd, rc, out_text, "")


def _allocate_gpu_session(
    colab_bin: str,
    session_name: str,
    gpu_order: Sequence[str],
) -> tuple[str, set[str]]:
    """Allocate a Colab GPU VM with 80s/95s aligned fast-fail (no CPU fallback)."""
    return allocate_gpu_session_fast(
        colab_bin,
        session_name,
        gpu_order,
        repo_root=WORKSPACE_ROOT,
    )


def _verify_zero_residual(
    colab_bin: str,
    session_name: str,
    *,
    pre_existing_endpoints: set[str] | None = None,
    alloc_uncertain: bool = False,
) -> dict[str, str]:
    """Stop `session_name`, unassign any `[?]` orphans, and verify zero residual."""
    res = cleanup_session_and_orphans(
        colab_bin,
        session_name,
        repo_root=WORKSPACE_ROOT,
        pre_existing_endpoints=pre_existing_endpoints,
        alloc_uncertain=alloc_uncertain,
    )
    sess_out = str(res["sessions"])
    usage_out = str(res["usage"])
    print(f"[colab-cleanup] sessions:\n{sess_out}", flush=True)
    print(f"[colab-cleanup] usage:\n{usage_out}", flush=True)

    rate_match = re.search(r"Usage rate:\s*([0-9.]+)/hr", usage_out)
    if (
        not pre_existing_endpoints
        and rate_match
        and float(rate_match.group(1)) > 0.0
    ):
        raise RuntimeError(
            f"Colab usage rate is non-zero after stop ({usage_out}); manual check required!"
        )
    if f"[{session_name}]" in sess_out:
        raise RuntimeError(
            f"Colab session {session_name!r} still listed active after stop:\n{sess_out}"
        )
    return {"sessions": sess_out, "usage": usage_out}


def _safe_extract_outputs(archive_path: Path, out_dir: Path) -> None:
    """Safely extract `outputs/*` -> `out_dir/*` and `logs/*` -> `out_dir/logs/*`."""
    out_resolved = out_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            rel = Path(member.name)
            if rel.is_absolute() or ".." in rel.parts:
                raise RuntimeError(f"Unsafe path in downloaded archive: {member.name}")
            if rel.parts and rel.parts[0] == "outputs":
                sub_parts = rel.parts[1:]
                if not sub_parts:
                    continue
                target = (out_resolved / Path(*sub_parts)).resolve()
            elif rel.parts and rel.parts[0] == "logs":
                target = (out_resolved / rel).resolve()
            else:
                target = (out_resolved / rel).resolve()

            if not str(target).startswith(str(out_resolved)):
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


def _rewrite_remote_paths(obj: Any, out_dir: Path) -> Any:
    """Rewrite `/content/cache/{outputs,logs}/...` strings inside downloaded metadata to local paths."""
    out_prefix = "/content/cache/outputs/"
    log_prefix = "/content/cache/logs/"
    if isinstance(obj, str):
        if obj.startswith(out_prefix):
            return str(out_dir / obj[len(out_prefix) :])
        if obj.startswith(log_prefix):
            return str(out_dir / "logs" / obj[len(log_prefix) :])
        return obj
    if isinstance(obj, list):
        return [_rewrite_remote_paths(x, out_dir) for x in obj]
    if isinstance(obj, dict):
        return {k: _rewrite_remote_paths(v, out_dir) for k, v in obj.items()}
    return obj


def _download_and_reassemble(
    colab_bin: str,
    session_name: str,
    staging_dir: Path,
    out_dir: Path,
    *,
    remote_has_rsync: bool | None = None,
) -> Path:
    """Download chunked output tarball via rsync/scp (or fallback), verify SHA256s, and extract."""
    bundle_local, _manifest, _transport = download_manifest_and_chunks_resilient(
        colab_bin,
        session_name,
        REMOTE_TRANSFER_DIR,
        "download_manifest.json",
        staging_dir,
        staging_dir=staging_dir,
        repo_root=WORKSPACE_ROOT,
        remote_has_rsync_hint=remote_has_rsync,
        fallback_download_fn=lambda rf, lf: _run_colab(
            colab_bin, ["download", "-s", session_name, rf, str(lf)], timeout_s=120.0
        ),
    )
    _safe_extract_outputs(bundle_local, out_dir)
    return bundle_local


def execute_job(
    job: JobConfig,
    *,
    benchmark: bool = False,
    dry_run: bool = False,
    project_back: float = 0.5,
    cellocut_resolution: int = 1024,
    cellocut_fill: float = 20.0,
    cellocut_band: float = 1.0,
    cellocut_qef: bool = False,
    cellocut_smooth_iters: int = 0,
    smooth_iters: int | None = None,
    min_component_area_rel: float = 0.005,
    weld_eps_rel: float = 1e-10,
    asset_type: str = "character",
    colab_bin: str = DEFAULT_COLAB_BIN,
) -> RunMeta | dict[str, Any]:
    """Execute an S1 benchmark or production run on a single Colab GPU session."""
    out_dir = Path(job.out_dir)
    _assert_not_system_temp(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = out_dir / "_colab_staging"

    bundle = _build_upload_bundle(
        job,
        staging_dir,
        is_benchmark=benchmark,
        project_back=project_back,
        cellocut_resolution=cellocut_resolution,
        cellocut_fill=cellocut_fill,
        cellocut_band=cellocut_band,
        cellocut_qef=cellocut_qef,
        cellocut_smooth_iters=cellocut_smooth_iters,
        smooth_iters=smooth_iters,
        min_component_area_rel=min_component_area_rel,
        weld_eps_rel=weld_eps_rel,
        asset_type=asset_type,
    )
    if dry_run:
        return {
            "dry_run": True,
            "mode": "benchmark" if benchmark else "run",
            "out_dir": str(out_dir),
            "staging_dir": str(staging_dir),
            "upload_manifest": bundle["manifest"],
            "spec": bundle["spec"],
        }

    session_name = f"s1-{'bench' if benchmark else 'run'}-{int(time.time())}"
    cleanup_done = False
    pre_existing_endpoints: set[str] | None = None
    allocated_gpu: str | None = None

    def _cleanup() -> None:
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        _verify_zero_residual(
            colab_bin,
            session_name,
            pre_existing_endpoints=pre_existing_endpoints,
            alloc_uncertain=(allocated_gpu is None),
        )

    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _on_signal(signum: int, _frame: Any) -> None:
        try:
            _cleanup()
        finally:
            raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        allocated_gpu, pre_existing_endpoints = _allocate_gpu_session(
            colab_bin, session_name, job.gpu_order
        )

        init_proc = _run_colab(
            colab_bin,
            [
                "exec",
                "-s",
                session_name,
                "--timeout",
                "180",
                "-f",
                str(bundle["init_dirs_script"]),
            ],
            stream=True,
        )
        remote_has_rsync: bool | None = None
        if "[colab-init] RSYNC=1" in init_proc.stdout:
            remote_has_rsync = True
        elif "[colab-init] RSYNC=0" in init_proc.stdout:
            remote_has_rsync = False

        # Upload spec, manifest, and bundle chunk(s) via rsync/scp over SSH (with REST fallback).
        upload_files_resilient(
            colab_bin,
            session_name,
            [
                bundle["spec_path"],
                bundle["manifest_path"],
                *bundle["chunk_paths"],
            ],
            REMOTE_TRANSFER_DIR,
            staging_dir=staging_dir,
            repo_root=WORKSPACE_ROOT,
            remote_has_rsync_hint=remote_has_rsync,
            fallback_upload_fn=lambda lf, rf: _run_colab(
                colab_bin, ["upload", "-s", session_name, str(lf), rf], timeout_s=120.0
            ),
        )

        print("[colab-exec] Starting remote S1 pipeline...", flush=True)
        exec_proc = _run_colab(
            colab_bin,
            [
                "exec",
                "-s",
                session_name,
                "--timeout",
                "300",
                "--env",
                "TQDM_DISABLE=1",
                "--env",
                "HF_HUB_DISABLE_PROGRESS_BARS=1",
                "-f",
                str(bundle["exec_entry_script"]),
            ],
            stream=True,
        )
        if not any(
            line.strip() == "[colab-exec-entry] STATUS=OK"
            for line in exec_proc.stdout.splitlines()
        ):
            raise RuntimeError(
                "Remote Colab execution did not emit completion sentinel '[colab-exec-entry] STATUS=OK'."
            )

        _download_and_reassemble(
            colab_bin,
            session_name,
            staging_dir,
            out_dir,
            remote_has_rsync=remote_has_rsync,
        )
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        _cleanup()

    if benchmark:
        report_path = out_dir / "benchmark_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report = _rewrite_remote_paths(report, out_dir)
        report["colab_sku"] = allocated_gpu
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return report

    meta_path = out_dir / "meta.json"
    raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw_meta = _rewrite_remote_paths(raw_meta, out_dir)
    if allocated_gpu and not raw_meta.get("gpu_name"):
        raw_meta["gpu_name"] = allocated_gpu
    elif allocated_gpu and allocated_gpu not in str(raw_meta.get("gpu_name", "")):
        raw_meta["gpu_name"] = f"{allocated_gpu} ({raw_meta['gpu_name']})"
    meta = RunMeta.from_dict(raw_meta)
    meta.write_json(meta_path)
    return meta


def run_benchmark(
    job: JobConfig,
    *,
    dry_run: bool = False,
    project_back: float = 0.5,
    cellocut_resolution: int = 1024,
    cellocut_fill: float = 20.0,
    cellocut_band: float = 1.0,
    cellocut_qef: bool = False,
    cellocut_smooth_iters: int = 0,
    smooth_iters: int | None = None,
    min_component_area_rel: float = 0.005,
    weld_eps_rel: float = 1e-10,
    asset_type: str = "character",
    colab_bin: str = DEFAULT_COLAB_BIN,
) -> dict[str, Any]:
    """Run Phase 2 Colab benchmark comparing Path A & Path B x inplace_clean & CelloCut."""
    result = execute_job(
        job,
        benchmark=True,
        dry_run=dry_run,
        project_back=project_back,
        cellocut_resolution=cellocut_resolution,
        cellocut_fill=cellocut_fill,
        cellocut_band=cellocut_band,
        cellocut_qef=cellocut_qef,
        cellocut_smooth_iters=cellocut_smooth_iters,
        smooth_iters=smooth_iters,
        min_component_area_rel=min_component_area_rel,
        weld_eps_rel=weld_eps_rel,
        asset_type=asset_type,
        colab_bin=colab_bin,
    )
    assert isinstance(result, dict)
    return result


def run_s1(
    job: JobConfig,
    *,
    dry_run: bool = False,
    project_back: float = 0.5,
    cellocut_resolution: int = 1024,
    cellocut_fill: float = 20.0,
    cellocut_band: float = 1.0,
    cellocut_qef: bool = False,
    cellocut_smooth_iters: int = 0,
    smooth_iters: int | None = None,
    min_component_area_rel: float = 0.005,
    weld_eps_rel: float = 1e-10,
    asset_type: str = "character",
    colab_bin: str = DEFAULT_COLAB_BIN,
) -> RunMeta | dict[str, Any]:
    """Run S1 production generation + repair + normalization on Colab."""
    return execute_job(
        job,
        benchmark=False,
        dry_run=dry_run,
        project_back=project_back,
        cellocut_resolution=cellocut_resolution,
        cellocut_fill=cellocut_fill,
        cellocut_band=cellocut_band,
        cellocut_qef=cellocut_qef,
        cellocut_smooth_iters=cellocut_smooth_iters,
        smooth_iters=smooth_iters,
        min_component_area_rel=min_component_area_rel,
        weld_eps_rel=weld_eps_rel,
        asset_type=asset_type,
        colab_bin=colab_bin,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local Colab orchestrator for S1 Pixal3D")
    p.add_argument("--mode", choices=("benchmark", "run"), default="run")
    p.add_argument("--asset-type", choices=("character", "prop"), default="character")
    p.add_argument("--input", required=True, help="Input views directory or single image")
    p.add_argument("--out-dir", required=True, help="Local output directory under runs/")
    p.add_argument("--input-mode", choices=("mv", "sv"), default="mv")
    p.add_argument("--backend", choices=("path_a", "path_b"), default="path_a")
    p.add_argument(
        "--repair",
        "--repair-method",
        dest="repair_method",
        choices=("inplace_clean", "inplace", "cellocut", "none"),
        default=None,
    )
    p.add_argument("--project-back", type=float, default=None)
    p.add_argument("--cellocut-resolution", "--remesh-resolution", dest="remesh_resolution", type=int, default=None)
    p.add_argument("--cellocut-fill", "--remesh-fill", dest="remesh_fill", type=float, default=None)
    p.add_argument("--cellocut-band", "--remesh-band", dest="remesh_band", type=float, default=None)
    p.add_argument("--qef", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--smooth-iters",
        "--cellocut-smooth-iters",
        dest="smooth_iters",
        type=int,
        default=None,
    )
    p.add_argument("--min-component-area-rel", type=float, default=None)
    p.add_argument("--weld-eps-rel", type=float, default=None)
    p.add_argument("--resolution", type=int, default=1536, choices=(1024, 1536))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed-list", default=None)
    p.add_argument("--fov-deg", type=float, default=20.0)
    p.add_argument("--pad-factor", type=float, default=1.1)
    p.add_argument("--target-height", type=float, default=1.0)
    p.add_argument("--gpu-order", default="G4,A100,H100")
    p.add_argument("--colab-bin", default=DEFAULT_COLAB_BIN)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    from quadvox.contracts import load_profile_config

    args = _build_parser().parse_args(argv)
    prof = load_profile_config("hard_surface_prop" if args.asset_type == "prop" else "object")
    rep = prof.s1.get("repair", {})
    seed_list = (
        [int(x.strip()) for x in args.seed_list.split(",") if x.strip()]
        if args.seed_list
        else None
    )
    gpu_order = tuple(
        g.strip().upper() for g in str(args.gpu_order).split(",") if g.strip()
    )
    is_bench = args.mode == "benchmark"
    repair_method = "both" if is_bench else str(args.repair_method or rep.get("default_method", "cellocut"))
    job = JobConfig(
        mode=args.input_mode,
        input_path=args.input,
        out_dir=args.out_dir,
        backend="both" if is_bench else args.backend,
        repair_method=repair_method,
        resolution=args.resolution,
        seed=args.seed,
        seeds=args.seeds,
        seed_list=seed_list,
        target_height=args.target_height,
        fov_deg=args.fov_deg,
        gpu_order=gpu_order,
        pad_factor=args.pad_factor,
    )
    project_back = float(args.project_back if args.project_back is not None else rep.get("cellocut_project_back", 0.5))
    remesh_res = int(args.remesh_resolution if args.remesh_resolution is not None else rep.get("cellocut_resolution", 1024))
    remesh_fill = float(args.remesh_fill if args.remesh_fill is not None else rep.get("cellocut_fill", 20.0))
    remesh_band = float(args.remesh_band if args.remesh_band is not None else rep.get("cellocut_band", 1.0))
    c_qef = bool(args.qef if args.qef is not None else rep.get("cellocut_qef", False))
    s_iters = int(args.smooth_iters if args.smooth_iters is not None else rep.get("cellocut_smooth_iters", 0))
    min_comp = float(args.min_component_area_rel if args.min_component_area_rel is not None else rep.get("min_component_area_rel", 0.005))
    weld_eps = float(args.weld_eps_rel if args.weld_eps_rel is not None else rep.get("weld_eps_rel", 1e-10))
    if is_bench:
        res = run_benchmark(
            job,
            dry_run=args.dry_run,
            project_back=project_back,
            cellocut_resolution=remesh_res,
            cellocut_fill=remesh_fill,
            cellocut_band=remesh_band,
            cellocut_qef=c_qef,
            cellocut_smooth_iters=s_iters,
            min_component_area_rel=min_comp,
            weld_eps_rel=weld_eps,
            asset_type=args.asset_type,
            colab_bin=args.colab_bin,
        )
    else:
        res = run_s1(
            job,
            dry_run=args.dry_run,
            project_back=project_back,
            cellocut_resolution=remesh_res,
            cellocut_fill=remesh_fill,
            cellocut_band=remesh_band,
            cellocut_qef=c_qef,
            cellocut_smooth_iters=s_iters,
            min_component_area_rel=min_comp,
            weld_eps_rel=weld_eps,
            asset_type=args.asset_type,
            colab_bin=args.colab_bin,
        )
    payload = res.to_dict() if hasattr(res, "to_dict") else res
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
