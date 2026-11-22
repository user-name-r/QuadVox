"""Shared Colab session, SSH/rsync transport, and orphan cleanup utilities.

Provides three reliability upgrades across Colab orchestrators:
1. 80s/95s aligned fast-fail GPU allocation (`COLAB_CLI_REQUEST_TIMEOUT=80`,
   `COLAB_CLI_ASSIGN_RETRIES=0`, outer timeout `95.0s`) so 30.5s gateway 503s
   fail over rapidly while slow legitimate allocations up to 80s are not killed
   mid-flight.
2. Binary `rsync` / `scp` file transfer over `colab ssh --proxy-mode` using an
   auto-provisioned Ed25519 key, avoiding slow Base64 JSON chunk loops while
   preserving automatic fallback to chunked `colab upload`/`download`.
3. True server-side `[?]` orphan `unassign` cleanup that snapshots pre-existing
   endpoints and inspects all known local/isolated `sessions.json` stores so
   concurrent lanes and external sessions (e.g. `s2-topo-*`) are preserved.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

COLAB_ASSIGN_REQUEST_TIMEOUT = 80
COLAB_ASSIGN_RETRIES = 0
COLAB_ASSIGN_RETRY_BACKOFF = 5
COLAB_ALLOC_TIMEOUT_S = 95.0

TRANSIENT_MARKERS = (
    "connection was lost",
    "connection reset",
    "connection refused",
    "connection aborted",
    "websocketconnectionclosedexception",
    "hung after websocket disconnect",
    "remotedisconnected",
    "broken pipe",
    "timed out",
    "timeout",
    "idle timeout",
    "eof",
    "502",
    "503",
    "504",
    "unavailable",
    "temporary failure",
    "try again",
    "429",
    "already-active-session",
)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in 1 MiB streaming blocks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(1024 * 1024)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def is_transient_error(exc: Exception | str) -> bool:
    """Return True if an exception or message looks like a transient Colab error."""
    msg = str(exc).lower()
    return any(marker in msg for marker in TRANSIENT_MARKERS)


def retry_transient(label: str, attempts: int, fn: Callable[[], Any]) -> Any:
    """Retry `fn` with bounded backoff when `is_transient_error` matches."""
    last_exc: Exception | None = None
    for idx in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if not is_transient_error(exc) or idx >= attempts:
                raise
            delay = min(15.0, 2.0 * idx)
            print(
                f"[retry] {label} attempt {idx}/{attempts} ({exc}); sleeping {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def colab_assign_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return subprocess environment enforcing 80s request timeout & 0 internal retries."""
    env = dict(base_env) if base_env is not None else os.environ.copy()
    env["COLAB_CLI_REQUEST_TIMEOUT"] = str(COLAB_ASSIGN_REQUEST_TIMEOUT)
    env["COLAB_CLI_ASSIGN_RETRIES"] = str(COLAB_ASSIGN_RETRIES)
    env["COLAB_CLI_RETRY_BACKOFF"] = str(COLAB_ASSIGN_RETRY_BACKOFF)
    return env


def resolve_colab_python(colab_bin: str) -> str | None:
    """Return the Python interpreter from `colab_bin`'s shebang if available."""
    try:
        bin_path = Path(colab_bin)
        if not bin_path.is_file():
            resolved = shutil.which(colab_bin)
            if not resolved:
                return None
            bin_path = Path(resolved)
        first_line = bin_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        if first_line.startswith("#!"):
            interp = first_line[2:].strip().split()[0]
            if Path(interp).is_file():
                return interp
    except Exception:
        return None
    return None


def _run_colab_python_snippet(
    colab_bin: str,
    code: str,
    *,
    args: Sequence[str] = (),
    timeout_s: float = 25.0,
) -> subprocess.CompletedProcess[str] | None:
    interp = resolve_colab_python(colab_bin)
    if not interp:
        return None
    env = colab_assign_env()
    return subprocess.run(
        [interp, "-c", code, *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )


def list_server_assignments(colab_bin: str) -> list[dict[str, str]]:
    """Query server-side Colab assignments via `colab_cli`'s own Python client.

    Returns a list of dicts: `{"endpoint": ..., "accelerator": ..., "variant": ...}`.
    Falls back to parsing `colab sessions` if the interpreter cannot be invoked.
    """
    snippet = (
        "import json\n"
        "from colab_cli.common import state\n"
        "rows = []\n"
        "for a in state.client.list_assignments():\n"
        "    rows.append({\n"
        "        'endpoint': str(a.endpoint),\n"
        "        'accelerator': str(a.accelerator.value),\n"
        "        'variant': str(a.variant.name),\n"
        "    })\n"
        "print(json.dumps(rows))\n"
    )
    try:
        proc = _run_colab_python_snippet(colab_bin, snippet, timeout_s=25.0)
        if proc is not None and proc.returncode == 0 and proc.stdout.strip():
            parsed = json.loads(proc.stdout.strip().splitlines()[-1])
            if isinstance(parsed, list):
                return [
                    {
                        "endpoint": str(item.get("endpoint", "")),
                        "accelerator": str(item.get("accelerator", "")),
                        "variant": str(item.get("variant", "")),
                    }
                    for item in parsed
                    if item.get("endpoint")
                ]
    except Exception:
        pass

    # Fallback: parse `colab sessions` stdout.
    try:
        proc = subprocess.run(
            [colab_bin, "sessions"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=25.0,
            env=colab_assign_env(),
        )
        rows: list[dict[str, str]] = []
        for line in (proc.stdout or "").splitlines():
            m = re.match(r"^\s*\[([^\]]+)\]\s+(\S+)", line)
            if m and m.group(1).lower() != "colab":
                hw = re.search(r"Hardware:\s*(\S+)", line)
                rows.append(
                    {
                        "endpoint": m.group(2),
                        "accelerator": hw.group(1) if hw else "",
                        "variant": "",
                    }
                )
        return rows
    except Exception:
        return []


def snapshot_assignment_endpoints(colab_bin: str) -> set[str]:
    """Snapshot currently active server-side assignment endpoints."""
    return {row["endpoint"] for row in list_server_assignments(colab_bin) if row.get("endpoint")}


def collect_known_session_endpoints(
    repo_root: Path | None = None,
    extra_configs: Sequence[Path | None] = (),
) -> dict[str, str]:
    """Collect `{endpoint: session_name}` across global and workspace session stores."""
    store_paths: list[Path] = [Path.home() / ".config" / "colab-cli" / "sessions.json"]
    for cfg in extra_configs:
        if cfg is not None:
            store_paths.append(Path(cfg))
    if repo_root is not None and (repo_root / "runs").is_dir():
        store_paths.extend((repo_root / "runs").glob("**/_colab_staging/colab_sessions.json"))

    seen: set[Path] = set()
    mapping: dict[str, str] = {}
    for sp in store_paths:
        try:
            resolved = sp.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for name, item in data.items():
            if isinstance(item, dict) and item.get("endpoint"):
                mapping[str(item["endpoint"])] = str(name)
    return mapping


def unassign_endpoint_direct(colab_bin: str, endpoint: str) -> bool:
    """Directly unassign a server-side endpoint (including `[?]` orphans)."""
    snippet = (
        "import sys\n"
        "from colab_cli.common import state\n"
        "state.client.unassign(sys.argv[1])\n"
    )
    try:
        proc = _run_colab_python_snippet(colab_bin, snippet, args=[endpoint], timeout_s=25.0)
        return proc is not None and proc.returncode == 0
    except Exception:
        return False


def cleanup_session_and_orphans(
    colab_bin: str,
    session_name: str,
    *,
    config_path: Path | None = None,
    repo_root: Path | None = None,
    pre_existing_endpoints: set[str] | None = None,
    alloc_uncertain: bool = False,
    run_bounded: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Stop `session_name`, unassign any newly-created `[?]` orphan endpoints, and verify."""
    def _exec_cli(args: Sequence[str], timeout_s: float = 30.0) -> subprocess.CompletedProcess[str]:
        if run_bounded is not None:
            return run_bounded(
                colab_bin,
                list(args),
                timeout_s=timeout_s,
                check=False,
                config_path=config_path,
            )
        prefix = ["--config", str(config_path)] if config_path is not None else []
        return subprocess.run(
            [colab_bin, *prefix, *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=colab_assign_env(),
        )

    cleared = False
    for attempt in range(1, 5):
        _exec_cli(["stop", "-s", session_name], timeout_s=35.0)
        sess_res = _exec_cli(["sessions"], timeout_s=25.0)
        out_txt = (sess_res.stdout or "") + (sess_res.stderr or "")
        if sess_res.returncode == 0 and f"[{session_name}]" not in out_txt:
            cleared = True
            break
        time.sleep(min(4.0, float(attempt)))

    known_endpoints = collect_known_session_endpoints(repo_root, extra_configs=[config_path])
    unassigned_orphans: list[str] = []
    server_rows = list_server_assignments(colab_bin)
    pre_eps = pre_existing_endpoints if pre_existing_endpoints is not None else set()

    for row in server_rows:
        ep = row["endpoint"]
        if ep in pre_eps:
            continue
        owner_name = known_endpoints.get(ep)
        # Clean up if it is either our own session stuck on server, or a true `[?]` orphan
        # (not registered in any local/isolated sessions.json) created after launch.
        if owner_name == session_name or (owner_name is None and (alloc_uncertain or not cleared)):
            if unassign_endpoint_direct(colab_bin, ep):
                unassigned_orphans.append(ep)
                print(
                    f"[colab-cleanup] Unassigned orphan/lingering endpoint {ep} "
                    f"(hw={row.get('accelerator') or '?'})",
                    flush=True,
                )

    global_sess = subprocess.run(
        [colab_bin, "sessions"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=25.0,
        env=colab_assign_env(),
    )
    global_usage = subprocess.run(
        [colab_bin, "usage"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=25.0,
        env=colab_assign_env(),
    )
    sess_out = (global_sess.stdout or "").strip()
    usage_out = (global_usage.stdout or "").strip()
    if f"[{session_name}]" not in sess_out:
        cleared = True

    return {
        "cleared": cleared,
        "unassigned_orphans": unassigned_orphans,
        "sessions": sess_out,
        "usage": usage_out,
    }


def allocate_gpu_session_fast(
    colab_bin: str,
    session_name: str,
    gpu_order: Sequence[str],
    *,
    config_path: Path | None = None,
    repo_root: Path | None = None,
    alloc_timeout_s: float = COLAB_ALLOC_TIMEOUT_S,
    run_bounded: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> tuple[str, set[str]]:
    """Allocate a Colab GPU VM with 80s/95s aligned fast-fail and orphan guard.

    Returns `(allocated_sku, pre_existing_endpoints)`.
    """
    if not gpu_order:
        raise ValueError("gpu_order must not be empty; CPU fallback is prohibited.")

    pre_existing = snapshot_assignment_endpoints(colab_bin)
    failures: list[str] = []

    for raw_sku in gpu_order:
        sku = str(raw_sku).strip().upper()
        if sku in ("CPU", "NONE", "AUTO", "DEFAULT", ""):
            raise ValueError(f"Unsupported non-GPU SKU {raw_sku!r} in gpu_order.")

        print(
            f"[colab-alloc] Requesting {session_name!r} --gpu {sku} "
            f"(req_timeout={COLAB_ASSIGN_REQUEST_TIMEOUT}s, retries={COLAB_ASSIGN_RETRIES}, "
            f"outer_bound={alloc_timeout_s:.0f}s)...",
            flush=True,
        )
        t0 = time.monotonic()
        try:
            if run_bounded is not None:
                proc = run_bounded(
                    colab_bin,
                    ["new", "-s", session_name, "--gpu", sku],
                    timeout_s=alloc_timeout_s,
                    idle_timeout_s=alloc_timeout_s,
                    stream=True,
                    check=True,
                    config_path=config_path,
                )
                elapsed = time.monotonic() - t0
                print(
                    f"[colab-alloc] Allocated {sku} for {session_name!r} in {elapsed:.1f}s.",
                    flush=True,
                )
                return sku, pre_existing

            prefix = ["--config", str(config_path)] if config_path is not None else []
            proc = subprocess.run(
                [colab_bin, *prefix, "new", "-s", session_name, "--gpu", sku],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=alloc_timeout_s,
                env=colab_assign_env(),
            )
            elapsed = time.monotonic() - t0
            if proc.returncode == 0:
                print(
                    f"[colab-alloc] Allocated {sku} for {session_name!r} in {elapsed:.1f}s.",
                    flush=True,
                )
                return sku, pre_existing
            err_msg = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            raise RuntimeError(err_msg[-1500:])
        except Exception as exc:
            elapsed = time.monotonic() - t0
            err_str = str(exc)
            uncertain = any(
                tok in err_str.lower()
                for tok in ("readtimeout", "timed out", "timeout", "read timed out")
            )
            failures.append(f"{sku} ({elapsed:.1f}s): {err_str[-400:]}")
            print(
                f"[colab-alloc] GPU {sku} failed in {elapsed:.1f}s; cleaning up before next SKU...",
                flush=True,
            )
            cleanup_session_and_orphans(
                colab_bin,
                session_name,
                config_path=config_path,
                repo_root=repo_root,
                pre_existing_endpoints=pre_existing,
                alloc_uncertain=uncertain,
                run_bounded=run_bounded,
            )

    raise RuntimeError(
        f"Failed to allocate any GPU in gpu_order {list(gpu_order)}:\n" + "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# SSH / rsync transport over `colab ssh --proxy-mode`
# ---------------------------------------------------------------------------

def ensure_colab_ssh_key(repo_root: Path | None = None) -> Path:
    """Return path to an Ed25519 private key (reusing `~/.ssh/id_ed25519` or generating one)."""
    default_priv = Path.home() / ".ssh" / "id_ed25519"
    default_pub = Path.home() / ".ssh" / "id_ed25519.pub"
    if default_priv.is_file() and default_pub.is_file():
        return default_priv

    if repo_root is not None:
        key_dir = repo_root / "runs" / "_cache" / "ssh"
    else:
        key_dir = Path.home() / ".config" / "colab-cli" / "ssh"
    key_dir.mkdir(parents=True, exist_ok=True)
    priv_path = key_dir / "colab_ed25519"
    pub_path = key_dir / "colab_ed25519.pub"

    if not priv_path.is_file() or not pub_path.is_file():
        priv_path.unlink(missing_ok=True)
        pub_path.unlink(missing_ok=True)
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(priv_path), "-q"],
            stdin=subprocess.DEVNULL,
            check=True,
        )
    os.chmod(priv_path, 0o600)
    return priv_path


def write_ssh_config(
    staging_dir: Path,
    colab_bin: str,
    session_name: str,
    key_path: Path,
    *,
    config_path: Path | None = None,
) -> Path:
    """Write a dedicated OpenSSH config bridging `colab-runtime` via `colab ssh --proxy-mode`."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = staging_dir / "colab_ssh_config"

    proxy_argv = [str(colab_bin)]
    if config_path is not None:
        proxy_argv.extend(["--config", str(Path(config_path).resolve())])
    proxy_argv.extend(
        ["ssh", "--proxy-mode", "-s", session_name, "-i", str(key_path.resolve())]
    )
    proxy_cmd = shlex.join(proxy_argv).replace("%", "%%")

    lines = [
        "Host colab-runtime",
        "  User root",
        f"  IdentityFile {key_path.resolve()}",
        "  IdentitiesOnly yes",
        "  StrictHostKeyChecking no",
        f"  UserKnownHostsFile {os.devnull}",
        "  LogLevel ERROR",
        "  BatchMode yes",
        "  ConnectTimeout 25",
        "  ServerAliveInterval 15",
        "  ServerAliveCountMax 3",
        f"  ProxyCommand {proxy_cmd}",
        "",
    ]
    cfg_path.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(cfg_path, 0o600)
    return cfg_path


def _ssh_transport_cmd(ssh_config_path: Path) -> str:
    return shlex.join(["ssh", "-F", str(ssh_config_path.resolve())])


def _run_ssh_command(
    ssh_config_path: Path,
    remote_cmd: str,
    *,
    timeout_s: float = 45.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-F", str(ssh_config_path.resolve()), "colab-runtime", remote_cmd],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _ensure_remote_dir_ssh(ssh_config_path: Path, remote_dir: str) -> None:
    proc = _run_ssh_command(
        ssh_config_path,
        f"mkdir -p {shlex.quote(remote_dir)}",
        timeout_s=45.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ssh mkdir failed (rc={proc.returncode}): {(proc.stdout + proc.stderr).strip()}"
        )


def _remote_has_rsync(ssh_config_path: Path) -> bool:
    if not shutil.which("rsync"):
        return False
    proc = _run_ssh_command(ssh_config_path, "command -v rsync >/dev/null 2>&1", timeout_s=35.0)
    return proc.returncode == 0


def upload_files_resilient(
    colab_bin: str,
    session_name: str,
    local_files: Sequence[Path],
    remote_dir: str,
    *,
    staging_dir: Path,
    config_path: Path | None = None,
    repo_root: Path | None = None,
    prefer_ssh: bool = True,
    remote_has_rsync_hint: bool | None = None,
    fallback_upload_fn: Callable[[Path, str], Any] | None = None,
) -> str:
    """Upload `local_files` into `remote_dir` via rsync/scp over SSH, falling back to `colab upload`.

    Returns the transport method used (`"rsync"`, `"scp"`, or `"colab-upload"`).
    """
    files = [Path(f) for f in local_files]
    if not files:
        return "none"
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f"Missing upload file: {f}")

    remote_base = remote_dir.rstrip("/")
    if prefer_ssh and shutil.which("ssh") and shutil.which("ssh-keygen"):
        try:
            key_path = ensure_colab_ssh_key(repo_root)
            ssh_cfg = write_ssh_config(
                staging_dir, colab_bin, session_name, key_path, config_path=config_path
            )
            _ensure_remote_dir_ssh(ssh_cfg, remote_base)
            use_rsync = (
                remote_has_rsync_hint
                if remote_has_rsync_hint is not None
                else _remote_has_rsync(ssh_cfg)
            )
            if use_rsync and shutil.which("rsync"):
                def _do_rsync() -> None:
                    cmd = [
                        "rsync",
                        "-av",
                        "--partial",
                        "--timeout=60",
                        "-e",
                        _ssh_transport_cmd(ssh_cfg),
                        *[str(f) for f in files],
                        f"colab-runtime:{remote_base}/",
                    ]
                    proc = subprocess.run(
                        cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600.0
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"rsync upload failed (rc={proc.returncode}): "
                            f"{(proc.stdout + proc.stderr)[-1200:]}"
                        )

                retry_transient("rsync upload", 4, _do_rsync)
                print(
                    f"[upload-rsync] Uploaded {len(files)} file(s) -> {remote_base}/",
                    flush=True,
                )
                return "rsync"

            if shutil.which("scp"):
                def _do_scp() -> None:
                    cmd = [
                        "scp",
                        "-F",
                        str(ssh_cfg.resolve()),
                        *[str(f) for f in files],
                        f"colab-runtime:{remote_base}/",
                    ]
                    proc = subprocess.run(
                        cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600.0
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"scp upload failed (rc={proc.returncode}): "
                            f"{(proc.stdout + proc.stderr)[-1200:]}"
                        )

                retry_transient("scp upload", 4, _do_scp)
                print(
                    f"[upload-scp] Uploaded {len(files)} file(s) -> {remote_base}/",
                    flush=True,
                )
                return "scp"
        except Exception as exc:
            print(
                f"[upload-fallback] SSH/rsync upload unavailable ({exc}); "
                "falling back to colab upload...",
                flush=True,
            )

    for idx, lf in enumerate(files, start=1):
        rf = f"{remote_base}/{lf.name}"
        print(f"[upload-rest] ({idx}/{len(files)}) {lf.name} -> {rf}", flush=True)
        if fallback_upload_fn is not None:
            retry_transient(f"upload {lf.name}", 8, lambda lpath=lf, rpath=rf: fallback_upload_fn(lpath, rpath))
        else:
            prefix = ["--config", str(config_path)] if config_path is not None else []
            retry_transient(
                f"upload {lf.name}",
                8,
                lambda lpath=lf, rpath=rf: subprocess.run(
                    [colab_bin, *prefix, "upload", "-s", session_name, str(lpath), rpath],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=120.0,
                    check=True,
                    env=colab_assign_env(),
                ),
            )
    return "colab-upload"


def download_manifest_and_chunks_resilient(
    colab_bin: str,
    session_name: str,
    remote_dir: str,
    manifest_name: str,
    dest_dir: Path,
    *,
    staging_dir: Path,
    config_path: Path | None = None,
    repo_root: Path | None = None,
    prefer_ssh: bool = True,
    remote_has_rsync_hint: bool | None = None,
    fallback_download_fn: Callable[[str, Path], Any] | None = None,
) -> tuple[Path, dict[str, Any], str]:
    """Download manifest + chunked/single archive via rsync/scp over SSH, falling back to `colab download`.

    Verifies every chunk SHA-256 and reassembles `archive_name` with full SHA-256 verification.
    Returns `(reassembled_archive_path, manifest_dict, transport_used)`.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    remote_base = remote_dir.rstrip("/")
    manifest_local = dest_dir / manifest_name
    transport_used = "colab-upload"

    def _rest_download(remote_file: str, local_file: Path) -> None:
        if fallback_download_fn is not None:
            fallback_download_fn(remote_file, local_file)
            return
        prefix = ["--config", str(config_path)] if config_path is not None else []
        proc = subprocess.run(
            [colab_bin, *prefix, "download", "-s", session_name, remote_file, str(local_file)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120.0,
            env=colab_assign_env(),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"colab download failed (rc={proc.returncode}): "
                f"{(proc.stdout + proc.stderr)[-1200:]}"
            )

    ssh_cfg: Path | None = None
    use_rsync = False
    if prefer_ssh and shutil.which("ssh") and shutil.which("ssh-keygen"):
        try:
            key_path = ensure_colab_ssh_key(repo_root)
            ssh_cfg = write_ssh_config(
                staging_dir, colab_bin, session_name, key_path, config_path=config_path
            )
            use_rsync = (
                remote_has_rsync_hint
                if remote_has_rsync_hint is not None
                else _remote_has_rsync(ssh_cfg)
            )
            if use_rsync and shutil.which("rsync"):
                def _fetch_man_rsync() -> None:
                    cmd = [
                        "rsync",
                        "-av",
                        "--partial",
                        "--timeout=60",
                        "-e",
                        _ssh_transport_cmd(ssh_cfg),
                        f"colab-runtime:{remote_base}/{manifest_name}",
                        str(manifest_local),
                    ]
                    proc = subprocess.run(
                        cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90.0
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"rsync manifest failed (rc={proc.returncode}): "
                            f"{(proc.stdout + proc.stderr)[-800:]}"
                        )

                retry_transient("rsync manifest", 4, _fetch_man_rsync)
                transport_used = "rsync"
            elif shutil.which("scp"):
                def _fetch_man_scp() -> None:
                    cmd = [
                        "scp",
                        "-F",
                        str(ssh_cfg.resolve()),
                        f"colab-runtime:{remote_base}/{manifest_name}",
                        str(manifest_local),
                    ]
                    proc = subprocess.run(
                        cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90.0
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"scp manifest failed (rc={proc.returncode}): "
                            f"{(proc.stdout + proc.stderr)[-800:]}"
                        )

                retry_transient("scp manifest", 4, _fetch_man_scp)
                transport_used = "scp"
            else:
                ssh_cfg = None
        except Exception as exc:
            print(
                f"[download-fallback] SSH/rsync manifest fetch failed ({exc}); "
                "using colab download...",
                flush=True,
            )
            ssh_cfg = None

    if not manifest_local.is_file():
        retry_transient(
            f"download {manifest_name}",
            8,
            lambda: _rest_download(f"{remote_base}/{manifest_name}", manifest_local),
        )
        transport_used = "colab-download"

    manifest = json.loads(manifest_local.read_text(encoding="utf-8"))
    chunks = manifest["chunks"]
    chunk_names = [str(c["name"]) for c in chunks]

    def _all_chunks_valid() -> bool:
        for c in chunks:
            lp = dest_dir / c["name"]
            if not lp.is_file() or sha256_file(lp) != c["sha256"]:
                return False
        return True

    if ssh_cfg is not None and transport_used == "rsync":
        files_list = staging_dir / "_rsync_download_files.txt"
        files_list.write_text("\n".join(chunk_names) + "\n", encoding="utf-8")

        def _fetch_chunks_rsync() -> None:
            cmd = [
                "rsync",
                "-av",
                "--partial",
                "--timeout=60",
                f"--files-from={files_list}",
                "-e",
                _ssh_transport_cmd(ssh_cfg),
                f"colab-runtime:{remote_base}/",
                str(dest_dir) + "/",
            ]
            proc = subprocess.run(
                cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900.0
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"rsync download failed (rc={proc.returncode}): "
                    f"{(proc.stdout + proc.stderr)[-1200:]}"
                )
            if not _all_chunks_valid():
                raise TimeoutError("SHA256 mismatch after rsync download; retrying partial transfer")

        try:
            retry_transient("rsync chunks", 4, _fetch_chunks_rsync)
            print(
                f"[download-rsync] Downloaded {len(chunks)} chunk(s) "
                f"({manifest['total_size'] / (1024 * 1024):.2f} MiB)",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[download-fallback] rsync chunks failed ({exc}); falling back to colab download...",
                flush=True,
            )
            ssh_cfg = None
    elif ssh_cfg is not None and transport_used == "scp":
        def _fetch_chunks_scp() -> None:
            cmd = [
                "scp",
                "-F",
                str(ssh_cfg.resolve()),
                *[f"colab-runtime:{remote_base}/{cn}" for cn in chunk_names],
                str(dest_dir) + "/",
            ]
            proc = subprocess.run(
                cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900.0
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"scp download failed (rc={proc.returncode}): "
                    f"{(proc.stdout + proc.stderr)[-1200:]}"
                )
            if not _all_chunks_valid():
                raise TimeoutError("SHA256 mismatch after scp download")

        try:
            retry_transient("scp chunks", 4, _fetch_chunks_scp)
            print(
                f"[download-scp] Downloaded {len(chunks)} chunk(s) "
                f"({manifest['total_size'] / (1024 * 1024):.2f} MiB)",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[download-fallback] scp chunks failed ({exc}); falling back to colab download...",
                flush=True,
            )
            ssh_cfg = None

    if not _all_chunks_valid():
        transport_used = "colab-download"
        for idx, c in enumerate(chunks, start=1):
            lp = dest_dir / c["name"]
            if lp.is_file() and sha256_file(lp) == c["sha256"]:
                continue
            print(
                f"[download-rest] ({idx}/{len(chunks)}) {c['name']} ({c['size']} B)",
                flush=True,
            )

            def _fetch_one(cname: str = c["name"], csha: str = c["sha256"], target: Path = lp) -> None:
                _rest_download(f"{remote_base}/{cname}", target)
                if not target.is_file() or sha256_file(target) != csha:
                    target.unlink(missing_ok=True)
                    raise TimeoutError(f"Download chunk SHA256 mismatch: {cname}")

            retry_transient(f"download {c['name']}", 8, _fetch_one)

    archive_local = dest_dir / manifest["archive_name"]
    if len(chunks) > 1 or chunks[0]["name"] != manifest["archive_name"]:
        with archive_local.open("wb") as dst:
            for c in chunks:
                part_path = dest_dir / c["name"]
                with part_path.open("rb") as src:
                    shutil.copyfileobj(src, dst)

    if sha256_file(archive_local) != manifest["sha256"]:
        raise RuntimeError(f"Reassembled archive SHA256 mismatch: {manifest['archive_name']}")

    return archive_local, manifest, transport_used
