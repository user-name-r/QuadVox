"""Host thin-client Colab orchestrator for Step 2C (Prop LATO.2 v3000 + Repair + S2.5 Low-UV + 4K Bake).

Strictly complies with `colab-operator`:
- Zero heavy 3D imports on host (tarballs staged straight from disk).
- Chunks uploads (<= 10 MiB) and downloads (<= 5 MiB) with SHA-256 verification.
- Snapshots pre-existing Colab sessions before `colab new`; `try/finally` stops
  and deletes ONLY our own `s2-prop-*` session and verifies `Usage rate: 0.00/hr`.
- Zero hardcoded `/tmp` paths (`_assert_not_system_temp` + AST scan).
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
DEFAULT_GPU_ORDER: tuple[str, ...] = ("G4", "A100", "H100")
DEFAULT_TIMEOUT = 300
MAX_UPLOAD_CHUNK_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_RETRIES = 12
_SYS_TEMP_ROOT = os.path.sep + "tmp"

HERE_DIR = Path(__file__).resolve().parent
TOPO_ROOT = HERE_DIR.parent
FINAL_OUT_DIR = TOPO_ROOT / "runs" / "s2_prop"

DEFAULT_HIGH_GLB = None
DEFAULT_OLD_LATO2_GLB = None

REMOTE_RUNNER_PATH = HERE_DIR / "colab_s2_prop_worker.py"
REMOTE_REPAIR_PATH = TOPO_ROOT / "quadvox" / "s2_retopo" / "lato2_repair.py"

REMOTE_CACHE_ROOT = "/content/cache"
REMOTE_WORK_DIR = f"{REMOTE_CACHE_ROOT}/work"
REMOTE_TRANSFER_DIR = f"{REMOTE_CACHE_ROOT}/transfer"
REMOTE_LOGS_DIR = f"{REMOTE_CACHE_ROOT}/logs"
REMOTE_SCRATCH_DIR = f"{REMOTE_CACHE_ROOT}/scratch"
REMOTE_SPEC_PATH = f"{REMOTE_TRANSFER_DIR}/job_spec.json"


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
    items = (
        [x.strip() for x in gpu_order.split(",") if x.strip()]
        if isinstance(gpu_order, str)
        else [str(x).strip() for x in gpu_order if str(x).strip()]
    )
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
    for p in paths:
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value.strip()
                if s == _SYS_TEMP_ROOT or s.startswith(_SYS_TEMP_ROOT + os.path.sep):
                    violations.append(f"{p.name}:{getattr(node, 'lineno', '?')}:{s!r}")
    return violations


def _split_file_chunks(
    src_path: Path,
    chunks_dir: Path,
    *,
    max_chunk_bytes: int = MAX_UPLOAD_CHUNK_BYTES,
) -> dict[str, Any]:
    total_size = src_path.stat().st_size
    full_sha = _sha256_file(src_path)
    chunks: list[dict[str, Any]] = []
    if total_size <= max_chunk_bytes:
        chunks.append({"name": src_path.name, "size": total_size, "sha256": full_sha})
    else:
        idx = 0
        with src_path.open("rb") as fh:
            while True:
                buf = fh.read(max_chunk_bytes)
                if not buf:
                    break
                part_name = f"{src_path.name}.part{idx:03d}"
                part_path = chunks_dir / part_name
                part_path.write_bytes(buf)
                chunks.append(
                    {"name": part_name, "size": len(buf), "sha256": hashlib.sha256(buf).hexdigest()}
                )
                idx += 1
        src_path.unlink(missing_ok=True)
    return {
        "archive_name": src_path.name,
        "total_size": total_size,
        "sha256": full_sha,
        "max_chunk_bytes": max_chunk_bytes,
        "chunks": chunks,
    }


def _run_cmd(
    cmd: list[str],
    *,
    timeout: int = DEFAULT_TIMEOUT,
    stream: bool = False,
) -> subprocess.CompletedProcess[str]:
    if not stream:
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            stdout, stderr = proc.communicate(timeout=timeout)
            return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            proc.wait()
            raise TimeoutError(f"Timed out after {timeout}s: {' '.join(cmd)}") from exc

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    lines: list[str] = []
    sel = selectors.DefaultSelector()
    assert proc.stdout is not None
    sel.register(proc.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + float(timeout)
    saw_conn_err_at: float | None = None
    try:
        while True:
            now = time.monotonic()
            rem = deadline - now
            if rem <= 0.0:
                raise TimeoutError(f"Timed out after {timeout}s: {' '.join(cmd)}")
            if saw_conn_err_at is not None and (now - saw_conn_err_at) >= 15.0:
                raise TimeoutError(f"Colab exec stalled after connection error: {' '.join(cmd)}")
            events = sel.select(timeout=min(1.0, rem))
            if events:
                line = proc.stdout.readline()
                if line:
                    lines.append(line)
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    if any(k in line for k in ("Connection was lost", "WebSocketConnectionClosedException", "MaxRetryError:")):
                        if saw_conn_err_at is None:
                            saw_conn_err_at = time.monotonic()
                    continue
                if proc.poll() is not None:
                    break
            elif proc.poll() is not None:
                for rem_line in proc.stdout:
                    lines.append(rem_line)
                    sys.stdout.write(rem_line)
                    sys.stdout.flush()
                break
    except BaseException:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        proc.wait()
        raise
    finally:
        sel.close()
    rc = proc.wait()
    return subprocess.CompletedProcess(cmd, rc, "".join(lines), "")


def _retry_colab_cmd(
    cmd: list[str],
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    label: str = "colab",
    stream: bool = False,
) -> subprocess.CompletedProcess[str]:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            res = _run_cmd(cmd, timeout=timeout, stream=stream)
            if res.returncode == 0:
                return res
            msg = ((res.stderr or "") + "\n" + (res.stdout or "")).strip()
            if res.returncode == 2 and "No such option" in msg or "No such command" in msg:
                raise ValueError(f"{label} invalid CLI invocation: {msg[-400:]}")
            last_err = RuntimeError(f"{label} failed (rc={res.returncode}): {msg[-400:]}")
        except ValueError:
            raise
        except (TimeoutError, subprocess.SubprocessError, OSError) as exc:
            last_err = exc
        if attempt < max_retries:
            wait_s = min(15.0, float(attempt) * 2.0)
            print(f"[{label}] transient failure ({attempt}/{max_retries}): {last_err}; retrying in {wait_s:.0f}s...", flush=True)
            time.sleep(wait_s)
    raise RuntimeError(f"{label} exhausted {max_retries} attempts: {last_err}")


def _colab_cmd(colab_bin: str, *args: str) -> list[str]:
    return [colab_bin, *args]


def _build_staging_bundle(
    *,
    high_glb: Path,
    old_lato2_glb: Path | None,
    raw_lato2_glb: Path | None = None,
    out_dir: Path,
    lato2_seeds: Sequence[int],
    build_tarball: bool = True,
) -> dict[str, Any]:
    _assert_not_system_temp(out_dir)
    if not high_glb.is_file():
        raise FileNotFoundError(f"Missing prop highpoly GLB: {high_glb}")
    if not REMOTE_RUNNER_PATH.is_file() or not REMOTE_REPAIR_PATH.is_file():
        raise FileNotFoundError("Missing Step 2C remote runner or repair helper script.")
    quadvox_dir = TOPO_ROOT / "quadvox"
    if not (quadvox_dir / "s2_retopo" / "slim_uv_bake.py").is_file():
        raise FileNotFoundError(f"Missing quadvox/s2_retopo/slim_uv_bake.py under {TOPO_ROOT}")

    py_compile.compile(str(REMOTE_RUNNER_PATH), doraise=True)
    py_compile.compile(str(REMOTE_REPAIR_PATH), doraise=True)
    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    violations = scan_no_hardcoded_tmp([REMOTE_RUNNER_PATH, REMOTE_REPAIR_PATH, Path(__file__).resolve()])
    if violations:
        raise ValueError(f"Detected forbidden system temp paths: {violations}")

    staging_dir = out_dir / "_colab_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    if raw_lato2_glb is not None and not raw_lato2_glb.is_file():
        raise FileNotFoundError(f"Missing precomputed raw LATO.2 GLB: {raw_lato2_glb}")
    spec = {
        "high_glb": f"inputs/{high_glb.name}",
        "baseline_old_glb": f"inputs/{old_lato2_glb.name}" if (old_lato2_glb and old_lato2_glb.is_file()) else None,
        "raw_lato2_glb": f"inputs/{raw_lato2_glb.name}" if raw_lato2_glb is not None else None,
        "proxy_faces": 120_000,
        "vert_num": 3000,
        "lato2_seeds": [int(s) for s in lato2_seeds],
        "max_chunk_bytes": 5 * 1024 * 1024,
    }
    (staging_dir / "job_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    init_dirs_py = staging_dir / "init_dirs.py"
    init_dirs_py.write_text(
        "import os, subprocess\n"
        f"for sub in ('work', 'transfer', 'logs', 'outputs', 'scratch', 'repos', 'models', 'tools', 'pip_cache', 'hf_home', 'torch_home'):\n"
        f"    os.makedirs(f'{REMOTE_CACHE_ROOT}/{{sub}}', exist_ok=True)\n"
        f"os.environ['TMPDIR'] = {REMOTE_SCRATCH_DIR!r}\n"
        "print('[colab-init] GPU:', subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip(), flush=True)\n",
        encoding="utf-8",
    )
    keepalive_py = staging_dir / "keepalive.py"
    keepalive_py.write_text("import time\nprint(f'[colab-keepalive] {time.time():.0f}', flush=True)\n", encoding="utf-8")

    exec_entry_py = staging_dir / "exec_entry.py"
    exec_entry_py.write_text(
        "import hashlib, json, os, subprocess, sys, tarfile, time\n"
        "from pathlib import Path\n"
        f"transfer = Path({REMOTE_TRANSFER_DIR!r})\n"
        f"work = Path({REMOTE_WORK_DIR!r})\n"
        f"scratch = Path({REMOTE_SCRATCH_DIR!r})\n"
        f"logs_dir = Path({REMOTE_LOGS_DIR!r})\n"
        "for d in (work, scratch, logs_dir): d.mkdir(parents=True, exist_ok=True)\n"
        "pid_file = work / 'prop_s2.pid'\n"
        "rc_file = work / 'prop_s2.rc'\n"
        "log_file = logs_dir / 'prop_s2.log'\n"
        "pos_file = work / 'prop_s2.pos'\n"
        "if not pid_file.is_file():\n"
        "    for f in (rc_file, pos_file):\n"
        "        if f.exists(): f.unlink()\n"
        "    manifest = json.loads((transfer / 'upload_manifest.json').read_text(encoding='utf-8'))\n"
        "    arc = transfer / manifest['archive_name']\n"
        "    chunks = manifest['chunks']\n"
        "    if len(chunks) > 1:\n"
        "        with arc.open('wb') as dst:\n"
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
        "    if hashlib.sha256(arc.read_bytes() if arc.stat().st_size < 16*1024*1024 else b'').hexdigest() == manifest['sha256']:\n"
        "        pass\n"
        "    else:\n"
        "        hf = hashlib.sha256()\n"
        "        with arc.open('rb') as af:\n"
        "            while True:\n"
        "                b = af.read(1024 * 1024)\n"
        "                if not b: break\n"
        "                hf.update(b)\n"
        "        if hf.hexdigest() != manifest['sha256']: raise RuntimeError('Archive SHA mismatch')\n"
        "    with tarfile.open(arc, 'r:gz') as tf:\n"
        "        tf.extractall(work, filter='data') if hasattr(tarfile, 'data_filter') else tf.extractall(work)\n"
        "    arc.unlink(missing_ok=True)\n"
        "    for part_f in transfer.glob('*.part*'):\n"
        "        part_f.unlink(missing_ok=True)\n"
        f"    runner = work / 'scripts' / {REMOTE_RUNNER_PATH.name!r}\n"
        f"    spec_str = {REMOTE_SPEC_PATH!r}\n"
        "    wrapper = work / '_launch_prop_s2.py'\n"
        "    wrapper_lines = [\n"
        "        'import os, subprocess, sys',\n"
        "        'from pathlib import Path',\n"
        "        f'work = Path({str(work)!r})',\n"
        "        f'log_file = Path({str(log_file)!r})',\n"
        "        f'rc_file = Path({str(rc_file)!r})',\n"
        "        'env = os.environ.copy()',\n"
        "        'env[\"PYTHONPATH\"] = str(work) + \":\" + str(work / \"scripts\")',\n"
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
        "t0 = time.monotonic()\n"
        "last_emit = t0\n"
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
        "    if rc_file.is_file():\n"
        "        rc_val = int(rc_file.read_text(encoding='utf-8').strip())\n"
        "        print(f'[batch-tail] COMPLETE rc={rc_val}', flush=True)\n"
        "        sys.exit(rc_val)\n"
        "    if time.monotonic() - t0 >= 180.0:\n"
        "        print('[batch-tail] BATCH_YIELD=REATTACH', flush=True)\n"
        "        sys.exit(0)\n"
        "    if time.monotonic() - last_emit >= 25.0:\n"
        "        print('[batch-tail] heartbeat...', flush=True)\n"
        "        last_emit = time.monotonic()\n"
        "    time.sleep(2.0)\n",
        encoding="utf-8",
    )

    manifest_info: dict[str, Any] = {}
    if build_tarball:
        arc_path = staging_dir / "prop_s2_upload.tar.gz"
        with tarfile.open(arc_path, "w:gz") as tf:
            tf.add(REMOTE_RUNNER_PATH, arcname=f"scripts/{REMOTE_RUNNER_PATH.name}", recursive=False)
            tf.add(REMOTE_REPAIR_PATH, arcname=f"scripts/{REMOTE_REPAIR_PATH.name}", recursive=False)
            for item in sorted(quadvox_dir.rglob("*")):
                if "__pycache__" in item.parts or item.suffix == ".pyc":
                    continue
                if item.is_file():
                    tf.add(item, arcname=f"quadvox/{item.relative_to(quadvox_dir).as_posix()}", recursive=False)
            tf.add(high_glb, arcname=f"inputs/{high_glb.name}", recursive=False)
            if old_lato2_glb and old_lato2_glb.is_file():
                tf.add(old_lato2_glb, arcname=f"inputs/{old_lato2_glb.name}", recursive=False)
            if raw_lato2_glb and raw_lato2_glb.is_file():
                tf.add(raw_lato2_glb, arcname=f"inputs/{raw_lato2_glb.name}", recursive=False)
        manifest_info = _split_file_chunks(arc_path, staging_dir, max_chunk_bytes=MAX_UPLOAD_CHUNK_BYTES)
        (staging_dir / "upload_manifest.json").write_text(json.dumps(manifest_info, indent=2) + "\n", encoding="utf-8")

    return {
        "staging_dir": staging_dir,
        "spec": spec,
        "upload_manifest": manifest_info,
        "init_dirs_py": init_dirs_py,
        "keepalive_py": keepalive_py,
        "exec_entry_py": exec_entry_py,
    }


def _safe_extract_archive(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            target = (dest_dir / member.name).resolve()
            if dest_dir.resolve() not in target.parents and target != dest_dir.resolve():
                raise ValueError(f"Unsafe tar member path: {member.name}")
        tf.extractall(dest_dir, filter="data") if hasattr(tarfile, "data_filter") else tf.extractall(dest_dir)


def _reassemble_and_extract_download(
    dl_manifest: dict[str, Any],
    dl_dir: Path,
    out_dir: Path,
    *,
    fetch_chunk_fn: Any,
) -> dict[str, Any]:
    """Fetch SHA-256 verified download chunks, reassemble archive, verify full SHA-256, and extract."""
    dl_dir.mkdir(parents=True, exist_ok=True)
    dl_archive = dl_dir / str(dl_manifest["archive_name"])
    chunks = dl_manifest["chunks"]
    with dl_archive.open("wb") as dst:
        for i, ch in enumerate(chunks, 1):
            part_loc = dl_dir / ch["name"]
            print(f"[prop-s2-dl] ({i}/{len(chunks)}) {ch['name']} ({ch['size']/1048576:.2f} MB)...", flush=True)
            fetch_chunk_fn(str(ch["name"]), part_loc)
            if _sha256_file(part_loc) != ch["sha256"]:
                raise RuntimeError(f"Chunk SHA mismatch: {ch['name']}")
            dst.write(part_loc.read_bytes())
            part_loc.unlink(missing_ok=True)
    if _sha256_file(dl_archive) != dl_manifest["sha256"]:
        raise RuntimeError("Reassembled download archive failed full SHA-256 verification.")

    extract_tmp = dl_dir / "_extracted"
    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    _safe_extract_archive(dl_archive, extract_tmp)
    for sub in ("outputs", "logs"):
        src_sub = extract_tmp / sub
        if src_sub.is_dir():
            target_sub = out_dir if sub == "outputs" else (out_dir / "logs")
            target_sub.mkdir(parents=True, exist_ok=True)
            for item in src_sub.rglob("*"):
                if item.is_file():
                    rel = item.relative_to(src_sub)
                    dst_p = target_sub / rel
                    dst_p.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dst_p)
    shutil.rmtree(extract_tmp, ignore_errors=True)
    dl_archive.unlink(missing_ok=True)
    report_p = out_dir / "prop_lato2_final_report.json"
    if report_p.is_file():
        return json.loads(report_p.read_text(encoding="utf-8"))
    return {"status": "ok", "out_dir": str(out_dir)}


def run_colab_prop_s2(
    *,
    high_glb: Path = DEFAULT_HIGH_GLB,
    old_lato2_glb: Path | None = DEFAULT_OLD_LATO2_GLB,
    raw_lato2_glb: Path | None = None,
    out_dir: Path = FINAL_OUT_DIR,
    gpu_order: Sequence[str] | str = DEFAULT_GPU_ORDER,
    lato2_seeds: Sequence[int] = (42, 43),
    colab_bin: str = DEFAULT_COLAB_BIN,
    dry_run: bool = False,
    dry_run_bundle: bool = False,
) -> dict[str, Any]:
    resolved_gpus = validate_gpu_order(gpu_order)
    _assert_not_system_temp(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = _build_staging_bundle(
        high_glb=high_glb,
        old_lato2_glb=old_lato2_glb,
        raw_lato2_glb=raw_lato2_glb,
        out_dir=out_dir,
        lato2_seeds=lato2_seeds,
        build_tarball=(dry_run_bundle or not dry_run),
    )
    if dry_run or dry_run_bundle:
        return {
            "dry_run": True,
            "dry_run_bundle": dry_run_bundle,
            "out_dir": str(out_dir),
            "gpu_order": list(resolved_gpus),
            "spec": bundle["spec"],
            "upload_manifest": bundle["upload_manifest"],
            "no_hardcoded_tmp": True,
        }

    if not Path(colab_bin).is_file():
        raise FileNotFoundError(f"Colab CLI not found at {colab_bin}")

    pre_sessions_res = _run_cmd([colab_bin, "sessions"], timeout=60)
    pre_existing = parse_colab_session_names(pre_sessions_res.stdout or "")
    print(f"[prop-s2-orch] Pre-existing sessions (protected): {sorted(pre_existing)}", flush=True)

    session_name = f"s2-prop-{int(time.time())}"
    allocated_gpu: str | None = None
    staging_dir: Path = bundle["staging_dir"]
    try:
        for sku in resolved_gpus:
            try:
                _retry_colab_cmd(
                    _colab_cmd(colab_bin, "new", "-s", session_name, "--gpu", sku, "--high-mem"),
                    timeout=90,
                    max_retries=6,
                    label=f"colab_new_{sku}",
                )
                allocated_gpu = sku
                break
            except Exception as exc:
                print(f"[prop-s2-orch] GPU {sku} unavailable: {exc}", flush=True)
        if allocated_gpu is None:
            raise RuntimeError(f"All requested GPU SKUs unavailable: {resolved_gpus}")

        _retry_colab_cmd(
            _colab_cmd(colab_bin, "exec", "-s", session_name, "--timeout", str(DEFAULT_TIMEOUT), "--file", str(bundle["init_dirs_py"])),
            timeout=DEFAULT_TIMEOUT,
            label="init_dirs",
            stream=True,
        )

        # Upload manifest, spec, and <=10 MiB archive chunks
        upload_items: list[tuple[Path, str]] = [
            (staging_dir / "job_spec.json", REMOTE_SPEC_PATH),
            (staging_dir / "upload_manifest.json", f"{REMOTE_TRANSFER_DIR}/upload_manifest.json"),
        ]
        for ch in bundle["upload_manifest"]["chunks"]:
            upload_items.append((staging_dir / ch["name"], f"{REMOTE_TRANSFER_DIR}/{ch['name']}"))

        for idx, (loc, rem) in enumerate(upload_items, 1):
            print(f"[prop-s2-upload] ({idx}/{len(upload_items)}) {loc.name} -> {rem}", flush=True)
            _retry_colab_cmd(
                _colab_cmd(colab_bin, "upload", "-s", session_name, str(loc), rem),
                timeout=DEFAULT_TIMEOUT,
                label=f"upload_{loc.name}",
            )

        for sl in range(1, 30):
            res = _retry_colab_cmd(
                _colab_cmd(colab_bin, "exec", "-s", session_name, "--timeout", str(DEFAULT_TIMEOUT), "--file", str(bundle["exec_entry_py"])),
                timeout=DEFAULT_TIMEOUT,
                label=f"remote_slice_{sl}",
                stream=True,
            )
            out_txt = res.stdout or ""
            m_rc = re.search(r"\[batch-tail\] COMPLETE rc=(-?\d+)", out_txt)
            if m_rc:
                rc_val = int(m_rc.group(1))
                if rc_val != 0:
                    raise RuntimeError(f"Remote pipeline exited rc={rc_val}; see output above.")
                break
            if "[batch-tail] BATCH_YIELD=REATTACH" not in out_txt:
                break
            _retry_colab_cmd(
                _colab_cmd(colab_bin, "exec", "-s", session_name, "--timeout", "120", "--file", str(bundle["keepalive_py"])),
                timeout=120,
                label=f"keepalive_{sl}",
            )

        dl_dir = staging_dir / "downloads"
        dl_dir.mkdir(parents=True, exist_ok=True)
        dl_manifest_local = dl_dir / "download_manifest.json"
        _retry_colab_cmd(
            _colab_cmd(colab_bin, "download", "-s", session_name, f"{REMOTE_TRANSFER_DIR}/download_manifest.json", str(dl_manifest_local)),
            timeout=DEFAULT_TIMEOUT,
            label="dl_manifest",
        )
        dl_manifest = json.loads(dl_manifest_local.read_text(encoding="utf-8"))

        def _fetch_chunk(chunk_name: str, local_path: Path) -> None:
            _retry_colab_cmd(
                _colab_cmd(colab_bin, "download", "-s", session_name, f"{REMOTE_TRANSFER_DIR}/{chunk_name}", str(local_path)),
                timeout=DEFAULT_TIMEOUT,
                label=f"dl_{chunk_name}",
            )

        return _reassemble_and_extract_download(
            dl_manifest, dl_dir, out_dir, fetch_chunk_fn=_fetch_chunk
        )
    finally:
        try:
            _run_cmd([colab_bin, "stop", "-s", session_name], timeout=60)
        except Exception:
            pass
        sess_res = _run_cmd([colab_bin, "sessions"], timeout=60)
        usage_res = _run_cmd([colab_bin, "usage"], timeout=60)
        after = parse_colab_session_names(sess_res.stdout or "")
        ours_left = [s for s in sorted(after - pre_existing) if s == session_name]
        print("[colab-cleanup] sessions:\n" + (sess_res.stdout or "").strip(), flush=True)
        print("[colab-cleanup] usage:\n" + (usage_res.stdout or "").strip(), flush=True)
        if ours_left:
            print(f"[colab-cleanup][WARN] Our session {ours_left} still listed!", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Colab GPU orchestrator for Hard-Surface Prop LATO.2 + 4K Bake")
    p.add_argument("--high-glb", required=True)
    p.add_argument("--old-lato2-glb", default="")
    p.add_argument("--raw-lato2-glb", default="")
    p.add_argument("--out-dir", default=str(FINAL_OUT_DIR))
    p.add_argument("--gpu-order", default="G4,A100,H100")
    p.add_argument("--seeds", default="42,43")
    p.add_argument("--colab-bin", default=DEFAULT_COLAB_BIN)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dry-run-bundle", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    res = run_colab_prop_s2(
        high_glb=Path(args.high_glb),
        old_lato2_glb=Path(args.old_lato2_glb) if args.old_lato2_glb else None,
        raw_lato2_glb=Path(args.raw_lato2_glb) if args.raw_lato2_glb else None,
        out_dir=Path(args.out_dir),
        gpu_order=args.gpu_order,
        lato2_seeds=seeds,
        colab_bin=args.colab_bin,
        dry_run=bool(args.dry_run),
        dry_run_bundle=bool(args.dry_run_bundle),
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
