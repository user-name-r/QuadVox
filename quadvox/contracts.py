"""Shared JSON-serializable data contracts and generic engine extension schemas.

All domain-specific behavior (such as face ROI sovereignty boxes, semantic
part schemas, or custom material slots) is injected through these generic
contracts rather than hardcoded into the engine.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


def _to_json_value(value: Any) -> Any:
    """Convert dataclasses, Paths, tuples, and NumPy scalars/arrays to JSON-safe Python primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        return {k: _to_json_value(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _to_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        return _to_json_value(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        return _to_json_value(value.item())
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _coerce_float3(values: Sequence[float] | Any, name: str) -> list[float]:
    if hasattr(values, "tolist") and callable(values.tolist):
        values = values.tolist()
    if len(values) != 3:
        raise ValueError(f"{name} must have length 3, got {len(values)}")
    return [float(values[0]), float(values[1]), float(values[2])]


def _coerce_float2(values: Sequence[float] | Any, name: str) -> tuple[float, float]:
    if hasattr(values, "tolist") and callable(values.tolist):
        values = values.tolist()
    if len(values) != 2:
        raise ValueError(f"{name} must have length 2, got {len(values)}")
    return (float(values[0]), float(values[1]))


@dataclass
class ROISovereigntyBox:
    """Generic 3D spatial Region-of-Interest (ROI) view-sovereignty lock.

    Enforces priority for a designated canonical view (e.g. 'front' for a mech
    cockpit, product logo, or character face) while suppressing orthogonal side-view
    bleed inside a C1-smooth box in physical [0, 1] height-normalized coordinates.
    """

    name: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    margin: float = 0.025
    target_view: str = "front"
    min_target_weight: float = 0.15
    min_surface_weight: float = 0.90
    suppress_ortho_views: float = 0.985
    normal_gate_range: tuple[float, float] = (0.10, 0.40)
    anchor_z_box: dict[str, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.x_range = _coerce_float2(self.x_range, "x_range")
        self.y_range = _coerce_float2(self.y_range, "y_range")
        self.z_range = _coerce_float2(self.z_range, "z_range")
        self.margin = float(self.margin)
        self.target_view = str(self.target_view)
        self.min_target_weight = float(self.min_target_weight)
        self.min_surface_weight = float(self.min_surface_weight)
        self.suppress_ortho_views = float(self.suppress_ortho_views)
        self.normal_gate_range = _coerce_float2(self.normal_gate_range, "normal_gate_range")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_value(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ROISovereigntyBox:
        anchor = data.get("anchor_z_box")
        anchor_parsed: dict[str, tuple[float, float]] | None = None
        if isinstance(anchor, Mapping):
            anchor_parsed = {
                str(k): _coerce_float2(v, f"anchor_z_box.{k}") for k, v in anchor.items()
            }
        return cls(
            name=str(data.get("name", "roi")),
            x_range=data["x_range"],
            y_range=data["y_range"],
            z_range=data["z_range"],
            margin=float(data.get("margin", 0.025)),
            target_view=str(data.get("target_view", "front")),
            min_target_weight=float(data.get("min_target_weight", 0.15)),
            min_surface_weight=float(data.get("min_surface_weight", 0.90)),
            suppress_ortho_views=float(data.get("suppress_ortho_views", 0.985)),
            normal_gate_range=data.get("normal_gate_range", (0.10, 0.40)),
            anchor_z_box=anchor_parsed,
        )


@dataclass
class ZoneSmoothingSpec:
    """Generic 3D spatial zone for extra geodesic view-weight transition smoothing."""

    name: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    margin: float = 0.025
    blend_weight: float = 0.85
    extra_iter_mult: float = 1.8
    view_indices: tuple[int, ...] = (1, 2)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.x_range = _coerce_float2(self.x_range, "x_range")
        self.y_range = _coerce_float2(self.y_range, "y_range")
        self.z_range = _coerce_float2(self.z_range, "z_range")
        self.margin = float(self.margin)
        self.blend_weight = float(self.blend_weight)
        self.extra_iter_mult = float(self.extra_iter_mult)
        self.view_indices = tuple(int(v) for v in self.view_indices)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ZoneSmoothingSpec:
        return cls(
            name=str(data.get("name", "smooth_zone")),
            x_range=data["x_range"],
            y_range=data["y_range"],
            z_range=data["z_range"],
            margin=float(data.get("margin", 0.025)),
            blend_weight=float(data.get("blend_weight", 0.85)),
            extra_iter_mult=float(data.get("extra_iter_mult", 1.8)),
            view_indices=tuple(int(i) for i in data.get("view_indices", (1, 2))),
        )


@dataclass
class SemanticSchema:
    """Generic semantic part schema injected into S2 UV/retopo and S3 lookdev."""

    name: str
    class_names: dict[int, str]
    class_colors_u8: dict[int, tuple[int, int, int]]
    material_slots: tuple[tuple[int, str], ...]
    priority_boost_cid: int | None = None
    priority_scale_boost: float = 1.0
    requad_density_weights: dict[int, float] = field(default_factory=dict)
    symmetric_class_ids: tuple[int, ...] = ()

    @classmethod
    def default_generic(cls) -> SemanticSchema:
        """Return a minimal single-class schema for general objects and props."""
        return cls(
            name="generic_object",
            class_names={0: "object"},
            class_colors_u8={0: (140, 152, 168)},
            material_slots=((0, "Mat_00_Object"),),
            priority_boost_cid=None,
            priority_scale_boost=1.0,
            requad_density_weights={0: 1.0},
            symmetric_class_ids=(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class_names": {str(k): v for k, v in self.class_names.items()},
            "class_colors_u8": {str(k): list(v) for k, v in self.class_colors_u8.items()},
            "material_slots": [[int(cid), name] for cid, name in self.material_slots],
            "priority_boost_cid": self.priority_boost_cid,
            "priority_scale_boost": self.priority_scale_boost,
            "requad_density_weights": {str(k): float(v) for k, v in self.requad_density_weights.items()},
            "symmetric_class_ids": list(self.symmetric_class_ids),
        }


@dataclass
class CoarseSpliceSpec:
    """Layer-2 configurable OpenVDB level-set splicing spec for dual coarse meshes."""

    enabled: bool = False
    voxel_res: int = 384
    half_width: float = 14.0
    split_axis: str = "y"
    transition_range: tuple[float, float] = (0.69, 0.75)
    lateral_guard_x: tuple[float, float] | None = None
    lateral_guard_y_max: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.voxel_res = int(self.voxel_res)
        self.half_width = float(self.half_width)
        self.split_axis = str(self.split_axis).lower()
        if self.split_axis not in ("x", "y", "z"):
            raise ValueError(f"split_axis must be 'x', 'y', or 'z', got {self.split_axis!r}")
        self.transition_range = _coerce_float2(self.transition_range, "transition_range")
        if self.lateral_guard_x is not None:
            self.lateral_guard_x = _coerce_float2(self.lateral_guard_x, "lateral_guard_x")
        if self.lateral_guard_y_max is not None:
            self.lateral_guard_y_max = _coerce_float2(self.lateral_guard_y_max, "lateral_guard_y_max")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CoarseSpliceSpec:
        return cls(
            enabled=bool(data.get("enabled", False)),
            voxel_res=int(data.get("voxel_res", 384)),
            half_width=float(data.get("half_width", 14.0)),
            split_axis=str(data.get("split_axis", "y")),
            transition_range=data.get("transition_range", (0.69, 0.75)),
            lateral_guard_x=data.get("lateral_guard_x"),
            lateral_guard_y_max=data.get("lateral_guard_y_max"),
        )


@dataclass
class AssetProfile:
    """Unified S0 -> S3 pipeline configuration profile for an asset category or preset."""

    name: str
    asset_type: str = "prop"  # "prop" | "character" | "object"
    plugin: str | None = None
    s0: dict[str, Any] = field(default_factory=dict)
    s1: dict[str, Any] = field(default_factory=dict)
    s2: dict[str, Any] = field(default_factory=dict)
    s3: dict[str, Any] = field(default_factory=dict)
    cloud: dict[str, Any] = field(default_factory=dict)

    def roi_boxes(self) -> list[ROISovereigntyBox]:
        fusion = self.s1.get("d3dv3_fusion", {})
        boxes: list[ROISovereigntyBox] = []
        front_box = fusion.get("front_roi_box")
        if front_box:
            boxes.append(front_box if isinstance(front_box, ROISovereigntyBox) else ROISovereigntyBox.from_dict(front_box))
        for r in fusion.get("roi_boxes", []):
            boxes.append(r if isinstance(r, ROISovereigntyBox) else ROISovereigntyBox.from_dict(r))
        return boxes

    def smoothing_zones(self) -> list[ZoneSmoothingSpec]:
        raw = self.s1.get("d3dv3_fusion", {}).get("smoothing_zones", [])
        return [
            z if isinstance(z, ZoneSmoothingSpec) else ZoneSmoothingSpec.from_dict(z)
            for z in raw
        ]

    def coarse_splice(self) -> CoarseSpliceSpec:
        raw = self.s1.get("coarse_splice") or {}
        return raw if isinstance(raw, CoarseSpliceSpec) else CoarseSpliceSpec.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return _to_json_value(asdict(self))


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


def load_profile_config(
    profile_or_path: str | Path = "object",
    *,
    defaults_path: str | Path | None = None,
) -> AssetProfile:
    """Load an `AssetProfile` merged over `configs/engine_defaults.yaml`.

    `profile_or_path` may be a built-in profile name (`object`, `hard_surface_prop`,
    `organic`), a preset name under `configs/presets/<name>.yaml` (e.g.
    `symmetric_humanoid`, `hard_surface_roi`), or an explicit path to a YAML file.
    """
    repo_root = Path(__file__).resolve().parent.parent
    def_path = Path(defaults_path) if defaults_path is not None else repo_root / "configs" / "engine_defaults.yaml"
    base_cfg: dict[str, Any] = {}
    if def_path.is_file():
        base_cfg = yaml.safe_load(def_path.read_text(encoding="utf-8")) or {}

    base_stages = {
        "s0": dict(base_cfg.get("s0", {})),
        "s1": dict(base_cfg.get("s1", {})),
        "s2": dict(base_cfg.get("s2", {})),
        "s3": dict(base_cfg.get("s3", {})),
        "cloud": dict(base_cfg.get("cloud", {})),
    }

    cand = Path(str(profile_or_path))
    override: dict[str, Any] = {}
    profile_name = str(profile_or_path)
    if cand.is_file():
        override = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
        profile_name = str(override.get("profile_name") or cand.stem)
    else:
        preset_file = repo_root / "configs" / "presets" / f"{profile_name}.yaml"
        if preset_file.is_file():
            override = yaml.safe_load(preset_file.read_text(encoding="utf-8")) or {}
        elif profile_name in (base_cfg.get("profiles") or {}):
            override = dict(base_cfg["profiles"][profile_name])
        else:
            raise FileNotFoundError(
                f"Unknown profile/preset {profile_or_path!r} (checked engine_defaults.yaml and {preset_file})"
            )

    merged = _deep_merge(base_stages, override)
    return AssetProfile(
        name=profile_name,
        asset_type=str(override.get("asset_type", "prop")),
        plugin=override.get("plugin"),
        s0=dict(merged.get("s0", {})),
        s1=dict(merged.get("s1", {})),
        s2=dict(merged.get("s2", {})),
        s3=dict(merged.get("s3", {})),
        cloud=dict(merged.get("cloud", {})),
    )


# ---------------------------------------------------------------------------
# S1 Core Job & Mesh Topology Contracts
# ---------------------------------------------------------------------------

@dataclass
class JobConfig:
    """Configuration for a single S1 generation/repair/benchmark job."""

    mode: str
    input_path: str
    out_dir: str
    backend: str
    repair_method: str
    resolution: int = 1536
    seed: int = 42
    seeds: int = 1
    seed_list: list[int] | tuple[int, ...] | None = None
    target_height: float = 1.0
    fov_deg: float = 20.0
    gpu_order: tuple[str, ...] = ("G4", "A100", "H100")
    pad_factor: float = 1.1

    def __post_init__(self) -> None:
        self.mode = str(self.mode)
        self.input_path = str(self.input_path)
        self.out_dir = str(self.out_dir)
        self.backend = str(self.backend)
        self.repair_method = str(self.repair_method)
        self.resolution = int(self.resolution)
        self.seed = int(self.seed)
        self.seeds = int(self.seeds)
        if self.seed_list is not None:
            self.seed_list = [int(s) for s in self.seed_list]
        self.target_height = float(self.target_height)
        self.fov_deg = float(self.fov_deg)
        self.gpu_order = tuple(str(g) for g in self.gpu_order)
        self.pad_factor = float(self.pad_factor)

    def resolved_seeds(self) -> list[int]:
        """Return the concrete list of integer seeds to evaluate."""
        if self.seed_list is not None and len(self.seed_list) > 0:
            return [int(s) for s in self.seed_list]
        count = max(1, int(self.seeds))
        return [int(self.seed) + idx for idx in range(count)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "input_path": self.input_path,
            "out_dir": self.out_dir,
            "backend": self.backend,
            "repair_method": self.repair_method,
            "resolution": self.resolution,
            "seed": self.seed,
            "seeds": self.seeds,
            "seed_list": list(self.seed_list) if self.seed_list is not None else None,
            "target_height": self.target_height,
            "fov_deg": self.fov_deg,
            "gpu_order": list(self.gpu_order),
            "pad_factor": self.pad_factor,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobConfig:
        gpu_order_raw = data.get("gpu_order", ("G4", "A100", "H100"))
        seed_list_raw = data.get("seed_list")
        return cls(
            mode=data["mode"],
            input_path=data["input_path"],
            out_dir=data["out_dir"],
            backend=data["backend"],
            repair_method=data["repair_method"],
            resolution=int(data.get("resolution", 1536)),
            seed=int(data.get("seed", 42)),
            seeds=int(data.get("seeds", 1)),
            seed_list=[int(s) for s in seed_list_raw] if seed_list_raw is not None else None,
            target_height=float(data.get("target_height", 1.0)),
            fov_deg=float(data.get("fov_deg", 20.0)),
            gpu_order=tuple(str(g) for g in gpu_order_raw),
            pad_factor=float(data.get("pad_factor", 1.1)),
        )


@dataclass
class MeshMetrics:
    """Deterministic topological and bounding-box metrics for a triangle mesh."""

    vertex_count: int
    face_count: int
    open_boundary_edges: int
    boundary_loops: int
    pinhole_loops: int
    large_opening_loops: int
    non_manifold_edges: int
    non_manifold_vertices: int
    degenerate_faces: int
    duplicate_faces: int
    connected_components: int
    is_watertight: bool
    is_pinhole_free: bool
    bbox_min: list[float]
    bbox_max: list[float]
    height: float

    def __post_init__(self) -> None:
        self.vertex_count = int(self.vertex_count)
        self.face_count = int(self.face_count)
        self.open_boundary_edges = int(self.open_boundary_edges)
        self.boundary_loops = int(self.boundary_loops)
        self.pinhole_loops = int(self.pinhole_loops)
        self.large_opening_loops = int(self.large_opening_loops)
        self.non_manifold_edges = int(self.non_manifold_edges)
        self.non_manifold_vertices = int(self.non_manifold_vertices)
        self.degenerate_faces = int(self.degenerate_faces)
        self.duplicate_faces = int(self.duplicate_faces)
        self.connected_components = int(self.connected_components)
        self.is_watertight = bool(self.is_watertight)
        self.is_pinhole_free = bool(self.is_pinhole_free)
        self.bbox_min = _coerce_float3(self.bbox_min, "bbox_min")
        self.bbox_max = _coerce_float3(self.bbox_max, "bbox_max")
        self.height = float(self.height)

    def to_dict(self) -> dict[str, Any]:
        return _to_json_value(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MeshMetrics:
        return cls(
            vertex_count=int(data["vertex_count"]),
            face_count=int(data["face_count"]),
            open_boundary_edges=int(data["open_boundary_edges"]),
            boundary_loops=int(data["boundary_loops"]),
            pinhole_loops=int(data["pinhole_loops"]),
            large_opening_loops=int(data["large_opening_loops"]),
            non_manifold_edges=int(data["non_manifold_edges"]),
            non_manifold_vertices=int(data["non_manifold_vertices"]),
            degenerate_faces=int(data["degenerate_faces"]),
            duplicate_faces=int(data["duplicate_faces"]),
            connected_components=int(data["connected_components"]),
            is_watertight=bool(data["is_watertight"]),
            is_pinhole_free=bool(data["is_pinhole_free"]),
            bbox_min=list(data["bbox_min"]),
            bbox_max=list(data["bbox_max"]),
            height=float(data["height"]),
        )


@dataclass
class SeedCandidate:
    """Raw generation candidate produced for a single seed."""

    seed: int
    raw_glb: str
    elapsed_s: float
    peak_vram_gb: float
    metrics: MeshMetrics

    def __post_init__(self) -> None:
        self.seed = int(self.seed)
        self.raw_glb = str(self.raw_glb)
        self.elapsed_s = float(self.elapsed_s)
        self.peak_vram_gb = float(self.peak_vram_gb)
        if isinstance(self.metrics, Mapping):
            self.metrics = MeshMetrics.from_dict(self.metrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "raw_glb": self.raw_glb,
            "elapsed_s": self.elapsed_s,
            "peak_vram_gb": self.peak_vram_gb,
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SeedCandidate:
        metrics_raw = data["metrics"]
        metrics = (
            metrics_raw
            if isinstance(metrics_raw, MeshMetrics)
            else MeshMetrics.from_dict(metrics_raw)
        )
        return cls(
            seed=int(data["seed"]),
            raw_glb=str(data["raw_glb"]),
            elapsed_s=float(data["elapsed_s"]),
            peak_vram_gb=float(data["peak_vram_gb"]),
            metrics=metrics,
        )


@dataclass
class RunMeta:
    """End-to-end metadata emitted for an S1 backend run."""

    job: JobConfig
    selected_seed: int
    candidates: list[SeedCandidate]
    raw_metrics: MeshMetrics
    repaired_metrics: MeshMetrics
    backend_version: str
    elapsed_s: float
    peak_vram_gb: float
    gpu_name: str
    outputs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.job, Mapping):
            self.job = JobConfig.from_dict(self.job)
        self.selected_seed = int(self.selected_seed)
        self.candidates = [
            c if isinstance(c, SeedCandidate) else SeedCandidate.from_dict(c)
            for c in self.candidates
        ]
        if isinstance(self.raw_metrics, Mapping):
            self.raw_metrics = MeshMetrics.from_dict(self.raw_metrics)
        if isinstance(self.repaired_metrics, Mapping):
            self.repaired_metrics = MeshMetrics.from_dict(self.repaired_metrics)
        self.backend_version = str(self.backend_version)
        self.elapsed_s = float(self.elapsed_s)
        self.peak_vram_gb = float(self.peak_vram_gb)
        self.gpu_name = str(self.gpu_name)
        self.outputs = {str(k): str(v) for k, v in dict(self.outputs).items()}

    def to_dict(self) -> dict[str, Any]:
        return _to_json_value(
            {
                "job": self.job.to_dict(),
                "selected_seed": self.selected_seed,
                "candidates": [c.to_dict() for c in self.candidates],
                "raw_metrics": self.raw_metrics.to_dict(),
                "repaired_metrics": self.repaired_metrics.to_dict(),
                "backend_version": self.backend_version,
                "elapsed_s": self.elapsed_s,
                "peak_vram_gb": self.peak_vram_gb,
                "gpu_name": self.gpu_name,
                "outputs": dict(self.outputs),
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")
        return out_path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunMeta:
        job_raw = data["job"]
        raw_m = data["raw_metrics"]
        rep_m = data["repaired_metrics"]
        return cls(
            job=job_raw if isinstance(job_raw, JobConfig) else JobConfig.from_dict(job_raw),
            selected_seed=int(data["selected_seed"]),
            candidates=[
                c if isinstance(c, SeedCandidate) else SeedCandidate.from_dict(c)
                for c in data.get("candidates", [])
            ],
            raw_metrics=raw_m if isinstance(raw_m, MeshMetrics) else MeshMetrics.from_dict(raw_m),
            repaired_metrics=rep_m if isinstance(rep_m, MeshMetrics) else MeshMetrics.from_dict(rep_m),
            backend_version=str(data["backend_version"]),
            elapsed_s=float(data["elapsed_s"]),
            peak_vram_gb=float(data["peak_vram_gb"]),
            gpu_name=str(data["gpu_name"]),
            outputs={str(k): str(v) for k, v in dict(data.get("outputs", {})).items()},
        )
