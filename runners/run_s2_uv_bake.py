"""Zero-RAM host thin-client Colab orchestrator for S2.5 4K Cycles baking & HD renders.

Obeys `colab-operator` & Repo Host Safety Guardrails:
- Never imports trimesh/open3d/pymeshlab/blender on host; streams tar archives from disk.
- Splits uploads and downloads into <= 6 MiB SHA256-verified chunks with 12x transient retry.
- Uses GPU order strictly `("A100", "H100", "G4")` with finite `--timeout 300` (never CPU).
- Snapshots pre-existing Colab sessions before `colab new`, stopping ONLY our `s25-uv-*` session.
- Zero hardcoded system temp paths (`_assert_not_system_temp` + AST scan).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import py_compile
import re
import selectors
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Sequence

from quadvox.tools.external_tools import resolve_colab_bin
DEFAULT_COLAB_BIN = resolve_colab_bin()
DEFAULT_GPU_ORDER: tuple[str, ...] = ("A100", "H100", "G4")
DEFAULT_TIMEOUT = 300
MAX_CHUNK_BYTES = 6 * 1024 * 1024
DEFAULT_MAX_RETRIES = 12
_SYS_TEMP_ROOT = os.path.sep + "tmp"

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
HERE_DIR = Path(__file__).resolve().parent
REMOTE_RUNNER_PATH = HERE_DIR / "colab_s2_uv_bake_worker.py"

REMOTE_CACHE_ROOT = "/content/cache"
REMOTE_WORK_DIR = f"{REMOTE_CACHE_ROOT}/work"
REMOTE_TRANSFER_DIR = f"{REMOTE_CACHE_ROOT}/transfer"
REMOTE_LOGS_DIR = f"{REMOTE_CACHE_ROOT}/logs"
REMOTE_SCRATCH_DIR = f"{REMOTE_CACHE_ROOT}/scratch"
REMOTE_SPEC_PATH = f"{REMOTE_TRANSFER_DIR}/job_spec.json"
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


def _assert_not_system_temp(path: Path | str) -> None:
    raw = str(path)
    norm = os.path.normpath(raw.strip())
    if norm == _SYS_TEMP_ROOT or norm.startswith(_SYS_TEMP_ROOT + os.path.sep):
        raise ValueError(f"Refusing system temp path {raw!r}; keep staging inside workspace.")
    resolved = str(Path(path).resolve())
    if resolved == _SYS_TEMP_ROOT or resolved.startswith(_SYS_TEMP_ROOT + os.path.sep):
        raise ValueError(f"Refusing resolved system temp path {resolved!r}; keep staging inside workspace.")


def validate_gpu_order(gpu_order: Sequence[str] | str) -> tuple[str, ...]:
    items = [x.strip() for x in gpu_order.split(",") if x.strip()] if isinstance(gpu_order, str) else [str(x).strip() for x in gpu_order if str(x).strip()]
    if not items:
        raise ValueError("gpu_order must not be empty; CPU fallback is prohibited.")
    out: list[str] = []
    for item in items:
        sku = item.upper()
        if sku in {"CPU", "NONE", "AUTO", "DEFAULT", ""}:
            raise ValueError(f"Refusing non-GPU SKU {item!r}; CPU fallback is prohibited.")
        if sku not in out:
            out.append(sku)
    return tuple(out)


def parse_colab_session_names(sessions_stdout: str) -> set[str]:
    names: set[str] = set()
    for line in sessions_stdout.splitlines():
        m = re.match(r"^\s*\[([A-Za-z0-9._-]+)\]\s+\S+", line)
        if m and m.group(1).lower() != "colab":
            names.add(m.group(1))
    return names


def scan_no_hardcoded_tmp(paths: Sequence[Path]) -> list[str]:
    violations: list[str] = []
    token = _SYS_TEMP_ROOT
    for py_file in paths:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value.strip()
                if val == token or val.startswith(token + "/"):
                    violations.append(f"{py_file}:{node.lineno}:{val!r}")
    return violations


def split_file_chunks_streaming(
    archive_path: Path,
    staging_dir: Path,
    *,
    max_chunk_bytes: int = MAX_CHUNK_BYTES,
    manifest_name: str = "upload_manifest.json",
) -> dict[str, Any]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    total_size = archive_path.stat().st_size
    full_hasher = hashlib.sha256()
    chunks_meta: list[dict[str, Any]] = []

    if total_size <= max_chunk_bytes:
        full_sha = _sha256_file(archive_path)
        chunks_meta.append({"name": archive_path.name, "size": total_size, "sha256": full_sha, "local_path": str(archive_path)})
    else:
        idx = 0
        io_block = 1024 * 1024
        with archive_path.open("rb") as src:
            while True:
                remaining = int(max_chunk_bytes)
                part_name = f"{archive_path.name}.part{idx:03d}"
                part_path = staging_dir / part_name
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
                chunks_meta.append({"name": part_name, "size": part_size, "sha256": ph.hexdigest(), "local_path": str(part_path)})
                idx += 1
        full_sha = full_hasher.hexdigest()

    manifest = {
        "archive_name": archive_path.name,
        "total_size": total_size,
        "sha256": full_sha,
        "max_chunk_bytes": int(max_chunk_bytes),
        "chunks": [{"name": c["name"], "size": c["size"], "sha256": c["sha256"]} for c in chunks_meta],
    }
    manifest_path = staging_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"manifest": manifest, "manifest_path": manifest_path, "chunk_paths": [Path(c["local_path"]) for c in chunks_meta]}


def reassemble_and_verify_chunks(manifest: dict[str, Any], chunk_dir: Path, out_archive_path: Path) -> Path:
    chunks = manifest["chunks"]
    out_archive_path.parent.mkdir(parents=True, exist_ok=True)
    if len(chunks) == 1:
        c = chunks[0]
        src = chunk_dir / c["name"]
        if not src.is_file() or _sha256_file(src) != c["sha256"]:
            raise RuntimeError(f"Chunk mismatch or missing: {src}")
        if src.resolve() != out_archive_path.resolve():
            shutil.copy2(src, out_archive_path)
    else:
        with out_archive_path.open("wb") as out_fh:
            for c in chunks:
                part = chunk_dir / c["name"]
                if not part.is_file() or _sha256_file(part) != c["sha256"]:
                    raise RuntimeError(f"SHA256 mismatch on chunk {c['name']}")
                with part.open("rb") as pf:
                    shutil.copyfileobj(pf, out_fh, length=1024 * 1024)
    if _sha256_file(out_archive_path) != manifest["sha256"]:
        raise RuntimeError(f"Final SHA256 mismatch on {out_archive_path.name}")
    return out_archive_path


def safe_extract_archive(archive_path: Path, target_dir: Path) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            rel = member.name.lstrip("./")
            if not rel:
                continue
            parts = rel.split("/", 1)
            if parts[0] == "outputs" and len(parts) == 2:
                dst = target_dir / parts[1]
            elif parts[0] == "logs":
                dst = target_dir / "logs" / (parts[1] if len(parts) == 2 else "")
            else:
                dst = target_dir / rel
            if member.isdir():
                dst.mkdir(parents=True, exist_ok=True)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            src_fh = tf.extractfile(member)
            if src_fh is None:
                continue
            with src_fh, dst.open("wb") as out_fh:
                shutil.copyfileobj(src_fh, out_fh, length=1024 * 1024)
            extracted.append(str(dst))
    return extracted


def build_upload_bundle(
    *,
    staging_dir: Path,
    jobs: list[dict[str, Any]],
    gpu_order: tuple[str, ...],
    timeout_s: int,
    max_chunk_bytes: int = MAX_CHUNK_BYTES,
) -> dict[str, Any]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = staging_dir / "upload_payload.tar.gz"
    files_to_pack: list[tuple[Path, str]] = []

    for py_path in sorted((WORKSPACE_ROOT / "quadvox").rglob("*.py")):
        if "__pycache__" not in py_path.parts:
            files_to_pack.append((py_path, py_path.relative_to(WORKSPACE_ROOT).as_posix()))
    files_to_pack.append((REMOTE_RUNNER_PATH, f"runner/{REMOTE_RUNNER_PATH.name}"))

    remote_jobs: list[dict[str, Any]] = []
    seen_arc: set[str] = set()
    for job in jobs:
        name = str(job["name"])
        r_job = dict(job)
        for key in ("high_mesh", "low_uv_mesh", "uv_preview_png", "unwrap_json"):
            val = job.get(key)
            if not val:
                continue
            src = Path(val).resolve()
            if not src.is_file():
                raise FileNotFoundError(f"Missing required job input for {name}.{key}: {src}")
            arc = f"inputs/{name}/{src.name}"
            if arc not in seen_arc:
                files_to_pack.append((src, arc))
                seen_arc.add(arc)
            r_job[key] = arc
        remote_jobs.append(r_job)

    with tarfile.open(bundle_path, "w:gz", dereference=True) as tf:
        for src, arc in files_to_pack:
            tf.add(src, arcname=arc, recursive=False)

    split_info = split_file_chunks_streaming(bundle_path, staging_dir, max_chunk_bytes=max_chunk_bytes, manifest_name="upload_manifest.json")
    if len(split_info["manifest"]["chunks"]) > 1 and bundle_path.exists():
        bundle_path.unlink()

    spec = {
        "gpu_order": list(gpu_order),
        "timeout_s": int(timeout_s),
        "max_chunk_bytes": int(max_chunk_bytes),
        "blender_url": DEFAULT_BLENDER_URL,
        "jobs": remote_jobs,
    }
    spec_path = staging_dir / "job_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    init_dirs_script = staging_dir / "init_dirs.py"
    init_dirs_script.write_text(
        "import os, subprocess\n"
        f"for sub in ('work', 'transfer', 'logs', 'outputs', 'scratch', 'tools'):\n"
        f"    os.makedirs(f'{REMOTE_CACHE_ROOT}/{{sub}}', exist_ok=True)\n"
        f"os.environ['TMPDIR'] = {REMOTE_SCRATCH_DIR!r}\n"
        "print('[colab-init]', subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True).strip(), flush=True)\n",
        encoding="utf-8",
    )

    keepalive_script = staging_dir / "keepalive.py"
    keepalive_script.write_text("import time\nprint(f'[colab-keepalive] {time.time():.0f}', flush=True)\n", encoding="utf-8")

    exec_entry_script = staging_dir / "exec_entry.py"
    exec_entry_code = (
        "import hashlib, json, os, subprocess, sys, tarfile, time\n"
        "from pathlib import Path\n"
        f"transfer = Path({REMOTE_TRANSFER_DIR!r})\n"
        f"work = Path({REMOTE_WORK_DIR!r})\n"
        f"scratch = Path({REMOTE_SCRATCH_DIR!r})\n"
        f"logs_dir = Path({REMOTE_LOGS_DIR!r})\n"
        "for d in (work, scratch, logs_dir): d.mkdir(parents=True, exist_ok=True)\n"
        "pid_file = work / 'remote_uv.pid'\n"
        "rc_file = work / 'remote_uv.rc'\n"
        "log_file = logs_dir / 'remote_uv.log'\n"
        "pos_file = work / 'remote_uv.pos'\n"
        "if not pid_file.is_file():\n"
        "    for f in (rc_file, pos_file):\n"
        "        if f.exists(): f.unlink()\n"
        "    manifest = json.loads((transfer / 'upload_manifest.json').read_text(encoding='utf-8'))\n"
        "    archive_path = transfer / manifest['archive_name']\n"
        "    chunks = manifest['chunks']\n"
        "    if len(chunks) > 1:\n"
        "        with archive_path.open('wb') as dst:\n"
        "            for c in chunks:\n"
        "                part = transfer / c['name']\n"
        "                h = hashlib.sha256()\n"
        "                with part.open('rb') as src:\n"
        "                    while True:\n"
        "                        buf = src.read(1024 * 1024)\n"
        "                        if not buf: break\n"
        "                        h.update(buf)\n"
        "                        dst.write(buf)\n"
        "                if h.hexdigest() != c['sha256']: raise RuntimeError(f'SHA mismatch {c[\"name\"]}')\n"
        "    h_full = hashlib.sha256()\n"
        "    with archive_path.open('rb') as fh:\n"
        "        while True:\n"
        "            buf = fh.read(1024 * 1024)\n"
        "            if not buf: break\n"
        "            h_full.update(buf)\n"
        "    if h_full.hexdigest() != manifest['sha256']: raise RuntimeError('Archive SHA mismatch')\n"
        "    with tarfile.open(archive_path, 'r:gz') as tf:\n"
        "        tf.extractall(work, filter='data') if hasattr(tarfile, 'data_filter') else tf.extractall(work)\n"
        "    archive_path.unlink(missing_ok=True)\n"
        f"    runner = work / 'runner' / {REMOTE_RUNNER_PATH.name!r}\n"
        f"    spec_str = {REMOTE_SPEC_PATH!r}\n"
        "    wrapper = work / '_launch_uv.py'\n"
        "    wrapper_lines = [\n"
        "        'import os, subprocess, sys',\n"
        "        'from pathlib import Path',\n"
        "        f'work = Path({str(work)!r})',\n"
        "        f'log_file = Path({str(log_file)!r})',\n"
        "        f'rc_file = Path({str(rc_file)!r})',\n"
        "        'env = os.environ.copy()',\n"
        "        'env[\"PYTHONPATH\"] = str(work)',\n"
        "        f'env[\"TMPDIR\"] = {str(scratch)!r}',\n"
        "        f'env[\"TEMP\"] = {str(scratch)!r}',\n"
        "        f'env[\"TMP\"] = {str(scratch)!r}',\n"
        "        'env[\"TQDM_DISABLE\"] = \"1\"',\n"
        "        'env[\"PYTHONUNBUFFERED\"] = \"1\"',\n"
        "        f'cmd = {[sys.executable, \"-u\", str(runner), \"--spec\", spec_str]!r}',\n"
        "        'with log_file.open(\"wb\") as lf:',\n"
        "        '    proc = subprocess.run(cmd, cwd=str(work), env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT)',\n"
        "        'tmp_rc = rc_file.with_suffix(\".tmp\")',\n"
        "        'tmp_rc.write_text(str(proc.returncode), encoding=\"utf-8\")',\n"
        "        'tmp_rc.replace(rc_file)',\n"
        "    ]\n"
        "    wrapper.write_text(chr(10).join(wrapper_lines) + chr(10), encoding='utf-8')\n"
        "    p = subprocess.Popen([sys.executable, '-u', str(wrapper)], cwd=str(work), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)\n"
        "    pid_file.write_text(str(p.pid), encoding='utf-8')\n"
        "offset = int(pos_file.read_text(encoding='utf-8').strip()) if pos_file.is_file() else 0\n"
        f"slice_max_s = {min(180.0, max(30.0, float(timeout_s) * 0.6))!r}\n"
        "t0 = time.monotonic()\n"
        "last_emit = t0\n"
        "status = 'OK'\n"
        "while True:\n"
        "    if log_file.is_file() and log_file.stat().st_size > offset:\n"
        "        with log_file.open('rb') as lf:\n"
        "            lf.seek(offset)\n"
        "            chunk = lf.read()\n"
        "        if chunk:\n"
        "            offset += len(chunk)\n"
        "            pos_file.write_text(str(offset), encoding='utf-8')\n"
        "            sys.stdout.write(chunk.decode('utf-8', errors='replace'))\n"
        "            sys.stdout.flush()\n"
        "            last_emit = time.monotonic()\n"
        "            continue\n"
        "    if rc_file.is_file():\n"
        "        if log_file.is_file() and log_file.stat().st_size > offset:\n"
        "            continue\n"
        "        rc = int(rc_file.read_text(encoding='utf-8').strip())\n"
        "        if rc != 0: raise RuntimeError(f'colab_remote_uv_bake failed rc={rc}')\n"
        "        status = 'OK'\n"
        "        break\n"
        "    now = time.monotonic()\n"
        "    if now - t0 >= slice_max_s:\n"
        "        status = 'RUNNING'\n"
        "        break\n"
        "    if now - last_emit >= 20.0:\n"
        "        print(f'[colab-uv-keepalive] elapsed={now - t0:.1f}s', flush=True)\n"
        "        last_emit = now\n"
        "    time.sleep(0.5)\n"
        "print('[colab-exec-entry] STATUS=' + status, flush=True)\n"
    )
    exec_entry_script.write_text(exec_entry_code, encoding="utf-8")
    for gp in (init_dirs_script, keepalive_script, exec_entry_script):
        py_compile.compile(str(gp), doraise=True)

    return {
        "spec": spec,
        "spec_path": spec_path,
        "files_packed": [arc for _, arc in files_to_pack],
        "init_dirs_script": init_dirs_script,
        "keepalive_script": keepalive_script,
        "exec_entry_script": exec_entry_script,
        **split_info,
    }


def _run_colab(colab_bin: str, args: Sequence[str], *, stream: bool = False, check: bool = True, timeout_s: float | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [colab_bin, *args]
    if not stream:
        op = str(args[0]) if args else ""
        t_o = timeout_s or (600.0 if op in ("upload", "download") else (240.0 if op == "new" else 90.0))
        tries = 6 if op in ("usage", "sessions", "stop") else 1
        proc: subprocess.CompletedProcess[str] | None = None
        for att in range(1, tries + 1):
            proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=t_o)
            if proc.returncode == 0:
                break
            err = proc.stderr + proc.stdout
            if att >= tries or not any(k in err for k in ("SSLEOFError", "SSLError", "RemoteDisconnected", "ConnectionResetError")):
                break
            time.sleep(min(6.0, 2.0 * att))
        assert proc is not None
        if check and proc.returncode != 0:
            raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
        return proc

    lines: list[str] = []
    idle_lim = float(timeout_s or 360.0)
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    assert p.stdout is not None
    fd = p.stdout.fileno()
    os.set_blocking(fd, False)
    sel = selectors.DefaultSelector()
    sel.register(fd, selectors.EVENT_READ)
    buf = b""
    last_act = time.monotonic()
    saw_conn_err_at: float | None = None
    eof = False
    killed: str | None = None
    try:
        while not eof:
            ev = sel.select(timeout=5.0)
            now = time.monotonic()
            if not ev:
                if saw_conn_err_at is not None and (now - saw_conn_err_at) >= 15.0:
                    killed = "colab exec stalled after connection error (15s grace)"
                    break
                if (now - last_act) >= idle_lim:
                    killed = f"colab exec idle timeout {idle_lim:.0f}s"
                    break
                if p.poll() is not None:
                    while True:
                        try:
                            c = os.read(fd, 8192)
                        except BlockingIOError:
                            break
                        if not c:
                            eof = True
                            break
                        buf += c
                    break
                continue
            while True:
                try:
                    c = os.read(fd, 8192)
                except BlockingIOError:
                    break
                if not c:
                    eof = True
                    break
                buf += c
                last_act = time.monotonic()
                while b"\n" in buf:
                    rl, buf = buf.split(b"\n", 1)
                    ln = rl.decode("utf-8", errors="replace") + "\n"
                    print(ln, end="", flush=True)
                    lines.append(ln)
                    if any(k in ln for k in ("RuntimeError: Connection was lost", "WebSocketConnectionClosedException", "SSLError:", "MaxRetryError:")):
                        if saw_conn_err_at is None:
                            saw_conn_err_at = time.monotonic()
        if buf:
            tl = buf.decode("utf-8", errors="replace")
            print(tl, end="", flush=True)
            lines.append(tl)
    finally:
        sel.close()
        if killed is not None and p.poll() is None:
            p.kill()
        if p.poll() is None:
            p.wait(timeout=10.0)
    rc = int(p.returncode if p.returncode is not None else -1)
    if check and (rc != 0 or killed is not None):
        raise RuntimeError(f"Stream cmd failed (rc={rc}, killed={killed}): {' '.join(cmd)}\n" + "\n".join(lines[-40:]))
    return subprocess.CompletedProcess(cmd, rc, "".join(lines), "")


def _transfer_retry(colab_bin: str, session_name: str, args: Sequence[str], *, cleanup_path: Path | None = None) -> None:
    last_exc: Exception | None = None
    for att in range(1, DEFAULT_MAX_RETRIES + 1):
        try:
            _run_colab(colab_bin, args, check=True)
            return
        except Exception as exc:
            last_exc = exc
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)
            if "File or directory not found" in str(exc):
                raise
            sess = _run_colab(colab_bin, ["sessions"], check=False)
            if sess.returncode == 0 and f"[{session_name}]" not in (sess.stdout + sess.stderr):
                raise RuntimeError(f"Colab session {session_name!r} gone during transfer: {exc}") from exc
            if att < DEFAULT_MAX_RETRIES:
                time.sleep(min(20.0, 2.5 * att))
    assert last_exc is not None
    raise last_exc


def _download_bundle(colab_bin: str, session_name: str, manifest_name: str, staging_dir: Path, out_dir: Path) -> dict[str, Any]:
    man_local = staging_dir / manifest_name
    _transfer_retry(colab_bin, session_name, ["download", "-s", session_name, f"{REMOTE_TRANSFER_DIR}/{manifest_name}", str(man_local)], cleanup_path=man_local)
    manifest = json.loads(man_local.read_text(encoding="utf-8"))
    dl_dir = staging_dir / f"dl_{Path(manifest_name).stem}"
    dl_dir.mkdir(parents=True, exist_ok=True)
    for idx, c in enumerate(manifest["chunks"], start=1):
        part_local = dl_dir / c["name"]
        if part_local.is_file() and part_local.stat().st_size == int(c["size"]) and _sha256_file(part_local) == str(c["sha256"]):
            continue
        print(f"[colab-dl] {manifest_name} chunk {idx}/{len(manifest['chunks'])}: {c['name']} ({c['size']/(1024*1024):.2f} MB)", flush=True)
        for _ in range(2):
            part_local.unlink(missing_ok=True)
            _transfer_retry(colab_bin, session_name, ["download", "-s", session_name, f"{REMOTE_TRANSFER_DIR}/{c['name']}", str(part_local)], cleanup_path=part_local)
            if _sha256_file(part_local) == str(c["sha256"]):
                break
    bundle_local = staging_dir / manifest["archive_name"]
    reassemble_and_verify_chunks(manifest, dl_dir, bundle_local)
    extracted = safe_extract_archive(bundle_local, out_dir)
    bundle_local.unlink(missing_ok=True)
    return {"manifest_name": manifest_name, "chunk_count": len(manifest["chunks"]), "extracted_count": len(extracted)}


def run_colab_uv_bake(
    *,
    jobs: list[dict[str, Any]],
    out_dir: Path,
    gpu_order: Sequence[str] | str = DEFAULT_GPU_ORDER,
    timeout: int = DEFAULT_TIMEOUT,
    colab_bin: str = DEFAULT_COLAB_BIN,
    dry_run_bundle: bool = False,
) -> dict[str, Any]:
    _assert_not_system_temp(out_dir)
    validated_gpus = validate_gpu_order(gpu_order)
    if int(timeout) <= 0:
        raise ValueError(f"timeout must be > 0, got {timeout}")

    py_compile.compile(str(WORKSPACE_ROOT / "quadvox" / "s2_retopo" / "slim_uv_bake.py"), doraise=True)
    py_compile.compile(str(REMOTE_RUNNER_PATH), doraise=True)
    py_compile.compile(str(Path(__file__)), doraise=True)
    viol = scan_no_hardcoded_tmp([WORKSPACE_ROOT / "quadvox" / "s2_retopo" / "slim_uv_bake.py", REMOTE_RUNNER_PATH, Path(__file__)])
    if viol:
        raise ValueError("Hardcoded temp paths detected:\n" + "\n".join(viol))

    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = out_dir / "_colab_staging"
    bundle = build_upload_bundle(staging_dir=staging_dir, jobs=jobs, gpu_order=validated_gpus, timeout_s=int(timeout))

    if dry_run_bundle:
        dry_rep = {
            "dry_run_bundle": True,
            "total_size_mb": round(bundle["manifest"]["total_size"] / (1024 * 1024), 3),
            "chunk_count": len(bundle["manifest"]["chunks"]),
            "max_chunk_mb": MAX_CHUNK_BYTES / (1024 * 1024),
            "files_packed_count": len(bundle["files_packed"]),
            "jobs": [j["name"] for j in jobs],
            "staging_dir": str(staging_dir),
        }
        (out_dir / "colab_uv_dry_run.json").write_text(json.dumps(dry_rep, indent=2) + "\n", encoding="utf-8")
        print("DRY_RUN_BUNDLE=OK", json.dumps(dry_rep, indent=2), flush=True)
        return dry_rep

    pre_sess = _run_colab(colab_bin, ["sessions"], check=False)
    pre_existing = parse_colab_session_names(pre_sess.stdout + pre_sess.stderr) if pre_sess.returncode == 0 else set()
    print(f"[colab] Pre-existing external sessions (untouched): {sorted(pre_existing)}", flush=True)

    session_name = f"s25-uv-{int(time.time())}"
    cleanup_done = False

    def _cleanup() -> None:
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        for _ in range(3):
            _run_colab(colab_bin, ["stop", "-s", session_name], check=False)
            s_chk = _run_colab(colab_bin, ["sessions"], check=False)
            cur = parse_colab_session_names(s_chk.stdout + s_chk.stderr)
            if session_name not in cur:
                print(f"[colab-cleanup] Stopped {session_name}; remaining sessions: {sorted(cur)}", flush=True)
                break
            time.sleep(2.0)

    prev_int, prev_term = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)

    def _on_sig(signum: int, _frame: Any) -> None:
        try:
            _cleanup()
        finally:
            raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    dl_results: list[dict[str, Any]] = []
    allocated_gpu: str | None = None
    try:
        for sku in validated_gpus:
            p = _run_colab(colab_bin, ["new", "-s", session_name, "--gpu", sku], check=False)
            if p.returncode == 0:
                allocated_gpu = sku
                break
            _run_colab(colab_bin, ["stop", "-s", session_name], check=False)
        if allocated_gpu is None:
            raise RuntimeError(f"Unable to allocate GPU from {validated_gpus}")
        print(f"[colab] Allocated {session_name} ({allocated_gpu})", flush=True)
        time.sleep(4.0)

        for att in range(1, 8):
            try:
                _run_colab(colab_bin, ["exec", "-s", session_name, "--timeout", "180", "-f", str(bundle["init_dirs_script"])], stream=True, timeout_s=90.0)
                break
            except Exception:
                if att >= 7:
                    raise
                time.sleep(min(10.0, 2.0 * att))

        up_pairs = [
            (bundle["spec_path"], REMOTE_SPEC_PATH),
            (bundle["manifest_path"], f"{REMOTE_TRANSFER_DIR}/upload_manifest.json"),
            *((cp, f"{REMOTE_TRANSFER_DIR}/{cp.name}") for cp in bundle["chunk_paths"]),
        ]
        for idx, (lf, rf) in enumerate(up_pairs, start=1):
            print(f"[colab-up] ({idx}/{len(up_pairs)}) {lf.name} ({lf.stat().st_size/(1024*1024):.2f} MB)", flush=True)
            _transfer_retry(colab_bin, session_name, ["upload", "-s", session_name, str(lf), rf])

        exec_cmd = ["exec", "-s", session_name, "--timeout", str(int(timeout)), "--env", "TQDM_DISABLE=1", "-f", str(bundle["exec_entry_script"])]
        completed = False
        for sl in range(1, 61):
            ep = _run_colab(colab_bin, exec_cmd, stream=True, timeout_s=float(timeout) + 60.0, check=False)
            lines = [ln.strip() for ln in ep.stdout.splitlines()]
            if "[colab-exec-entry] STATUS=OK" in lines:
                completed = True
                break
            if ep.returncode != 0 and "[colab-exec-entry] STATUS=RUNNING" not in lines:
                if "colab_remote_uv_bake failed" in ep.stdout:
                    raise RuntimeError(ep.stdout[-2000:])
                time.sleep(5.0)
        if not completed:
            raise RuntimeError("Colab UV bake exceeded max exec slices")

        for man_name in ("bundle1_manifest.json", "bundle2_manifest.json"):
            dl_results.append(_download_bundle(colab_bin, session_name, man_name, staging_dir, out_dir))
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
        _cleanup()

    return {"session": session_name, "gpu": allocated_gpu, "downloads": dl_results, "out_dir": str(out_dir)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Colab GPU 4K bake + render orchestrator for S2.5 low-island UV.")
    ap.add_argument("--out-dir", default="runs/s2_uv_bake")
    ap.add_argument("--gpu-order", default="A100,H100,G4")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--dry-run-bundle", action="store_true")
    ap.add_argument("--high-mesh", required=True, help="Highpoly reference GLB/OBJ.")
    ap.add_argument("--low-uv-mesh", required=True, help="Lowpoly UV-unwrapped GLB.")
    ap.add_argument("--uv-preview-png", default=None)
    ap.add_argument("--unwrap-json", default=None)
    ap.add_argument("--framing-mode", choices=("object", "character"), default="object")
    args = ap.parse_args(argv)

    jobs = [
        {
            "name": Path(args.low_uv_mesh).stem,
            "high_mesh": args.high_mesh,
            "low_uv_mesh": args.low_uv_mesh,
            "uv_preview_png": args.uv_preview_png,
            "unwrap_json": args.unwrap_json,
            "tex_size": 4096,
            "padding_px": 16,
            "ao_samples": 24,
            "render_samples": 32,
            "framing_mode": args.framing_mode,
        }
    ]
    res = run_colab_uv_bake(
        jobs=jobs,
        out_dir=Path(args.out_dir),
        gpu_order=args.gpu_order,
        timeout=int(args.timeout),
        dry_run_bundle=bool(args.dry_run_bundle),
    )
    print(json.dumps(res, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
