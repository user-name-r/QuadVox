"""Base generator contracts and isolated subprocess / VRAM instrumentation helpers.

Reference URLs:
- Pixal3D upstream: https://github.com/TencentARC/Pixal3D
- Pixal3D MV inference: https://github.com/TencentARC/Pixal3D/blob/master/inference_mv.py
- Pixal3D weights: https://huggingface.co/TencentARC/Pixal3D
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from quadvox.contracts import MeshMetrics, SeedCandidate


UPSTREAM_PIXAL3D_URL = "https://github.com/TencentARC/Pixal3D"
UPSTREAM_INFERENCE_MV_URL = "https://github.com/TencentARC/Pixal3D/blob/master/inference_mv.py"
UPSTREAM_HF_PIXAL3D_URL = "https://huggingface.co/TencentARC/Pixal3D"

DEFAULT_COLAB_CACHE_DIR = "/content/cache"
DEFAULT_LOCAL_CACHE_DIR = "runs/_cache"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def is_colab_runtime() -> bool:
    """Return True when running inside a Google Colab VM with `/content`."""
    content_dir = Path("/content")
    return content_dir.is_dir() and os.access(content_dir, os.W_OK)


def default_cache_dir(workspace_root: str | Path | None = None) -> Path:
    """Resolve persistent working cache directory without touching system temp dirs."""
    env_cache = os.environ.get("S1_CACHE_DIR")
    if env_cache:
        return Path(env_cache)
    if is_colab_runtime():
        return Path(DEFAULT_COLAB_CACHE_DIR)
    root = Path(workspace_root) if workspace_root is not None else Path.cwd()
    return root / DEFAULT_LOCAL_CACHE_DIR


def _natural_sort_key(name: str) -> list[int | str]:
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def resolve_views_from_dir(
    views_dir: str | Path,
    mode: str = "mv",
) -> tuple[list[Path], dict[str, Any] | None]:
    """Resolve ordered view image paths (and optional transforms.json) from `views_dir`."""
    norm_mode = str(mode).lower()
    if norm_mode not in ("mv", "sv"):
        raise ValueError(f"Unsupported mode {mode!r}; expected 'mv' or 'sv'")

    src = Path(views_dir)
    if src.is_file():
        if norm_mode != "sv":
            raise ValueError(f"MV mode requires a directory of views, got file: {src}")
        return [src], None

    if not src.is_dir():
        raise FileNotFoundError(f"Views directory not found: {src}")

    transforms_path = src / "transforms.json"
    transforms_meta: dict[str, Any] | None = None
    if transforms_path.exists():
        transforms_meta = json.loads(transforms_path.read_text(encoding="utf-8"))
        frames = transforms_meta.get("frames", [])
        if not frames:
            raise ValueError(f"{transforms_path} contains no frames")
        resolved = [src / fr["file_path"] for fr in frames]
        missing = [str(p) for p in resolved if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing frames referenced by {transforms_path}: {missing}")
        return (resolved[:1] if norm_mode == "sv" else resolved), transforms_meta

    canonical = [
        "view00_azim000.png",
        "view01_azim090.png",
        "view02_azim180.png",
        "view03_azim270.png",
    ]
    if all((src / name).exists() for name in canonical):
        paths = [src / name for name in canonical]
        return (paths[:1] if norm_mode == "sv" else paths), None

    images = sorted(
        (p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: _natural_sort_key(p.name),
    )
    if not images:
        raise FileNotFoundError(f"No view images found in {src}")
    return (images[:1] if norm_mode == "sv" else images), None


def _collect_descendant_pids(root_pid: int) -> set[int]:
    """Collect `root_pid` and all descendant PIDs via `/proc` (Linux best-effort)."""
    pids = {root_pid}
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return pids

    children_map: dict[int, list[int]] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        stat_path = entry / "stat"
        try:
            stat_text = stat_path.read_text(encoding="utf-8", errors="ignore")
            rparen = stat_text.rfind(")")
            if rparen < 0:
                continue
            fields = stat_text[rparen + 2 :].split()
            if len(fields) >= 2:
                ppid = int(fields[1])
                children_map.setdefault(ppid, []).append(pid)
        except OSError:
            continue

    stack = [root_pid]
    while stack:
        cur = stack.pop()
        for child in children_map.get(cur, ()):
            if child not in pids:
                pids.add(child)
                stack.append(child)
    return pids


def query_nvidia_smi_vram_gb(pids: set[int] | None = None, gpu_index: int = 0) -> float:
    """Query used GPU memory in GiB for specific PIDs, falling back to device memory.used."""
    if pids:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3.0,
            )
            total_mib = 0.0
            matched = False
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 2:
                    continue
                try:
                    proc_pid = int(parts[0])
                    mem_mib = float(parts[1])
                except ValueError:
                    continue
                if proc_pid in pids:
                    total_mib += mem_mib
                    matched = True
            if matched and total_mib > 0.0:
                return total_mib / 1024.0
        except Exception:
            pass

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3.0,
        )
        first = out.strip().splitlines()[0].strip()
        return float(first) / 1024.0
    except Exception:
        return 0.0


def tail_file(path: Path, lines: int = 50) -> str:
    """Return the last `lines` lines of `path` safely."""
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def monitor_subprocess_with_vram(
    cmd: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    heartbeat_label: str = "worker",
    heartbeat_interval_s: float = 20.0,
    vram_poll_interval_s: float = 0.4,
) -> tuple[int, float, float, str]:
    """Run `cmd` in an isolated subprocess, polling peak VRAM and emitting heartbeats.

    Heavy subprocess output is written to `log_path` so callers never flood stdout,
    while flushed heartbeat lines keep Colab's idle watchdog happy.
    Returns `(returncode, elapsed_s, peak_vram_gb, log_tail)`.
    """
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})
    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    existing_pp = merged_env.get("PYTHONPATH", "")
    if repo_root not in existing_pp.split(os.pathsep):
        merged_env["PYTHONPATH"] = (
            f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else repo_root
        )
    merged_env.setdefault("TQDM_DISABLE", "1")
    merged_env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    merged_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    resolved_log = Path(log_path) if log_path is not None else None
    if resolved_log is not None:
        resolved_log.parent.mkdir(parents=True, exist_ok=True)
        log_fh = resolved_log.open("w", encoding="utf-8", errors="replace")
    else:
        log_fh = None

    baseline_vram_gb = query_nvidia_smi_vram_gb(pids=None)
    peak_vram_gb = 0.0
    stop_event = threading.Event()
    t0 = time.monotonic()

    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        env=merged_env,
        stdout=log_fh if log_fh is not None else subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def _vram_and_heartbeat_loop() -> None:
        nonlocal peak_vram_gb
        last_hb = t0
        while not stop_event.wait(timeout=vram_poll_interval_s):
            pids = _collect_descendant_pids(proc.pid)
            sample_pid_gb = query_nvidia_smi_vram_gb(pids=pids)
            if sample_pid_gb <= 0.0:
                device_gb = query_nvidia_smi_vram_gb(pids=None)
                sample_pid_gb = max(0.0, device_gb - baseline_vram_gb)
            if sample_pid_gb > peak_vram_gb:
                peak_vram_gb = sample_pid_gb

            now = time.monotonic()
            if now - last_hb >= heartbeat_interval_s:
                print(
                    f"[{heartbeat_label}] running... elapsed={now - t0:.1f}s "
                    f"peak_vram={peak_vram_gb:.2f}GB",
                    flush=True,
                )
                last_hb = now

    monitor_thread = threading.Thread(target=_vram_and_heartbeat_loop, daemon=True)
    monitor_thread.start()

    captured_tail = ""
    try:
        if log_fh is None and proc.stdout is not None:
            stdout_data, _ = proc.communicate()
            captured_tail = "\n".join(stdout_data.splitlines()[-60:])
        else:
            proc.wait()
    finally:
        stop_event.set()
        monitor_thread.join(timeout=2.0)
        if log_fh is not None:
            log_fh.flush()
            log_fh.close()

    elapsed_s = time.monotonic() - t0
    if resolved_log is not None:
        captured_tail = tail_file(resolved_log, lines=60)
    return int(proc.returncode), float(elapsed_s), float(peak_vram_gb), captured_tail


@dataclass
class GenerationResult:
    """Result emitted by a shape-only S1 generator backend."""

    backend: str
    mode: str
    seed: int
    resolution: int
    raw_glb: str
    elapsed_s: float
    peak_vram_gb: float
    coordinate_frame: str
    log_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_seed_candidate(self, metrics: MeshMetrics) -> SeedCandidate:
        """Convert this raw generation result + topology metrics to `SeedCandidate`."""
        return SeedCandidate(
            seed=self.seed,
            raw_glb=self.raw_glb,
            elapsed_s=self.elapsed_s,
            peak_vram_gb=self.peak_vram_gb,
            metrics=metrics,
        )


class BaseShapeGenerator(ABC):
    """Abstract interface for shape-only Pixal3D generator backends (Path A / Path B)."""

    backend_name: str = "base"
    coordinate_frame: str = "y_up_z_front"
    supported_resolutions: tuple[int, ...] = (1024, 1536)

    def validate_request(self, mode: str, resolution: int) -> tuple[str, int]:
        norm_mode = str(mode).lower()
        if norm_mode not in ("mv", "sv"):
            raise ValueError(f"Unsupported input mode {mode!r}; expected 'mv' or 'sv'")
        res = int(resolution)
        if res not in self.supported_resolutions:
            raise ValueError(
                f"{self.backend_name} supports resolutions {self.supported_resolutions}, got {res}"
            )
        return norm_mode, res

    @abstractmethod
    def generate(
        self,
        views_dir: str | Path,
        out_glb: str | Path,
        *,
        mode: str = "mv",
        resolution: int = 1536,
        seed: int = 42,
        fov_deg: float = 20.0,
        log_path: str | Path | None = None,
    ) -> GenerationResult:
        """Run isolated shape-only inference and write raw mesh to `out_glb`."""

    def generate_seeds(
        self,
        views_dir: str | Path,
        out_dir: str | Path,
        seeds: Sequence[int],
        *,
        mode: str = "mv",
        resolution: int = 1536,
        fov_deg: float = 20.0,
    ) -> list[GenerationResult]:
        """Run multiple seeds sequentially, each in an isolated subprocess freeing 100% VRAM."""
        dest = Path(out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        results: list[GenerationResult] = []
        for seed in seeds:
            seed_val = int(seed)
            raw_glb = dest / f"seed_{seed_val}_raw.glb"
            log_file = dest / f"seed_{seed_val}_{self.backend_name}.log"
            res = self.generate(
                views_dir=views_dir,
                out_glb=raw_glb,
                mode=mode,
                resolution=resolution,
                seed=seed_val,
                fov_deg=fov_deg,
                log_path=log_file,
            )
            results.append(res)
        return results
