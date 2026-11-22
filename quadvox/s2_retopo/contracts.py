"""Shared JSON-serializable data contracts for the S2 semantic-weighted retopology experiments.

Matches the orthogonal dataclass + JSON serialization style of `s1_backend/contracts.py`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


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


def _coerce_opt_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _write_json_file(payload: Mapping[str, Any], path: str | Path, default_name: str) -> Path:
    target = Path(path)
    if target.exists() and target.is_dir():
        target = target / default_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


@dataclass
class RetopoJobConfig:
    """Configuration for a single S2 retopology / bake / evaluation job."""

    input_mesh_path: str
    engine: str
    target_faces: int = 35000
    preserve_boundary: bool = True
    boundary_weight: float = 1.5
    semantic_backend: str = "none"
    semantic_proxy_faces: int = 150000
    weight_mapping_mode: str = "none"
    focus_weight_multiplier: float = 1.0
    enable_bake: bool = True
    uv_tex_size: int = 2048
    uv_padding_px: int = 8
    # Extended execution metadata
    job_id: str = ""
    stage: str = "stage1"
    input_name: str = ""
    out_dir: str = ""
    target_height: float = 1.0
    coordinate_frame: str = "y_up_z_front"
    preserve_normal: bool = True
    planar_quadric: bool = True
    quality_thr: float = 0.3
    optimal_placement: bool = True
    autoclean: bool = True
    smooth_iters: int = 2
    smooth_alpha: float = 0.5
    use_gpu_bake: bool = False
    seed: int = 42
    notes: str = ""

    def __post_init__(self) -> None:
        self.input_mesh_path = str(self.input_mesh_path)
        self.engine = str(self.engine)
        self.target_faces = int(self.target_faces)
        self.preserve_boundary = bool(self.preserve_boundary)
        self.boundary_weight = float(self.boundary_weight)
        self.semantic_backend = str(self.semantic_backend)
        self.semantic_proxy_faces = int(self.semantic_proxy_faces)
        self.weight_mapping_mode = str(self.weight_mapping_mode)
        self.focus_weight_multiplier = float(self.focus_weight_multiplier)
        self.enable_bake = bool(self.enable_bake)
        self.uv_tex_size = int(self.uv_tex_size)
        self.uv_padding_px = int(self.uv_padding_px)
        self.job_id = str(self.job_id)
        self.stage = str(self.stage)
        self.input_name = str(self.input_name)
        self.out_dir = str(self.out_dir)
        self.target_height = float(self.target_height)
        self.coordinate_frame = str(self.coordinate_frame)
        self.preserve_normal = bool(self.preserve_normal)
        self.planar_quadric = bool(self.planar_quadric)
        self.quality_thr = float(self.quality_thr)
        self.optimal_placement = bool(self.optimal_placement)
        self.autoclean = bool(self.autoclean)
        self.smooth_iters = int(self.smooth_iters)
        self.smooth_alpha = float(self.smooth_alpha)
        self.use_gpu_bake = bool(self.use_gpu_bake)
        self.seed = int(self.seed)
        self.notes = str(self.notes)

    def to_dict(self) -> dict[str, Any]:
        return _to_json_value(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: str | Path) -> Path:
        return _write_json_file(self.to_dict(), path, "job_config.json")

    def save(self, path: str | Path) -> Path:
        return self.write_json(path)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetopoJobConfig:
        return cls(
            input_mesh_path=str(data["input_mesh_path"]),
            engine=str(data["engine"]),
            target_faces=int(data.get("target_faces", 35000)),
            preserve_boundary=bool(data.get("preserve_boundary", True)),
            boundary_weight=float(data.get("boundary_weight", 1.5)),
            semantic_backend=str(data.get("semantic_backend", "none")),
            semantic_proxy_faces=int(data.get("semantic_proxy_faces", 150000)),
            weight_mapping_mode=str(data.get("weight_mapping_mode", "none")),
            focus_weight_multiplier=float(data.get("focus_weight_multiplier", 1.0)),
            enable_bake=bool(data.get("enable_bake", True)),
            uv_tex_size=int(data.get("uv_tex_size", 2048)),
            uv_padding_px=int(data.get("uv_padding_px", 8)),
            job_id=str(data.get("job_id", "")),
            stage=str(data.get("stage", "stage1")),
            input_name=str(data.get("input_name", "")),
            out_dir=str(data.get("out_dir", "")),
            target_height=float(data.get("target_height", 1.0)),
            coordinate_frame=str(data.get("coordinate_frame", "y_up_z_front")),
            preserve_normal=bool(data.get("preserve_normal", True)),
            planar_quadric=bool(data.get("planar_quadric", True)),
            quality_thr=float(data.get("quality_thr", 0.3)),
            optimal_placement=bool(data.get("optimal_placement", True)),
            autoclean=bool(data.get("autoclean", True)),
            smooth_iters=int(data.get("smooth_iters", 2)),
            smooth_alpha=float(data.get("smooth_alpha", 0.5)),
            use_gpu_bake=bool(data.get("use_gpu_bake", False)),
            seed=int(data.get("seed", 42)),
            notes=str(data.get("notes", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> RetopoJobConfig:
        source = Path(path)
        if source.is_dir():
            source = source / "job_config.json"
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))


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
        return _to_json_value(
            {
                "vertex_count": self.vertex_count,
                "face_count": self.face_count,
                "open_boundary_edges": self.open_boundary_edges,
                "boundary_loops": self.boundary_loops,
                "pinhole_loops": self.pinhole_loops,
                "large_opening_loops": self.large_opening_loops,
                "non_manifold_edges": self.non_manifold_edges,
                "non_manifold_vertices": self.non_manifold_vertices,
                "degenerate_faces": self.degenerate_faces,
                "duplicate_faces": self.duplicate_faces,
                "connected_components": self.connected_components,
                "is_watertight": self.is_watertight,
                "is_pinhole_free": self.is_pinhole_free,
                "bbox_min": self.bbox_min,
                "bbox_max": self.bbox_max,
                "height": self.height,
            }
        )

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
class PrecleanDiff:
    """Before/after MeshMetrics comparison for lossless preclean stripping."""

    input_path: str
    output_path: str
    before: MeshMetrics
    after: MeshMetrics
    removed_degenerate_faces: int = 0
    removed_duplicate_faces: int = 0
    removed_small_components: int = 0
    removed_unreferenced_vertices: int = 0
    min_component_area_rel: float = 0.0
    elapsed_s: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        self.input_path = str(self.input_path)
        self.output_path = str(self.output_path)
        if isinstance(self.before, Mapping):
            self.before = MeshMetrics.from_dict(self.before)
        if isinstance(self.after, Mapping):
            self.after = MeshMetrics.from_dict(self.after)
        self.removed_degenerate_faces = int(self.removed_degenerate_faces)
        self.removed_duplicate_faces = int(self.removed_duplicate_faces)
        self.removed_small_components = int(self.removed_small_components)
        self.removed_unreferenced_vertices = int(self.removed_unreferenced_vertices)
        self.min_component_area_rel = float(self.min_component_area_rel)
        self.elapsed_s = float(self.elapsed_s)
        self.notes = str(self.notes)

    def to_dict(self) -> dict[str, Any]:
        return _to_json_value(
            {
                "input_path": self.input_path,
                "output_path": self.output_path,
                "before": self.before.to_dict(),
                "after": self.after.to_dict(),
                "removed_degenerate_faces": self.removed_degenerate_faces,
                "removed_duplicate_faces": self.removed_duplicate_faces,
                "removed_small_components": self.removed_small_components,
                "removed_unreferenced_vertices": self.removed_unreferenced_vertices,
                "min_component_area_rel": self.min_component_area_rel,
                "elapsed_s": self.elapsed_s,
                "notes": self.notes,
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: str | Path) -> Path:
        return _write_json_file(self.to_dict(), path, "preclean_diff.json")

    def save(self, path: str | Path) -> Path:
        return self.write_json(path)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PrecleanDiff:
        before_raw = data["before"]
        after_raw = data["after"]
        return cls(
            input_path=str(data["input_path"]),
            output_path=str(data["output_path"]),
            before=before_raw if isinstance(before_raw, MeshMetrics) else MeshMetrics.from_dict(before_raw),
            after=after_raw if isinstance(after_raw, MeshMetrics) else MeshMetrics.from_dict(after_raw),
            removed_degenerate_faces=int(data.get("removed_degenerate_faces", 0)),
            removed_duplicate_faces=int(data.get("removed_duplicate_faces", 0)),
            removed_small_components=int(data.get("removed_small_components", 0)),
            removed_unreferenced_vertices=int(data.get("removed_unreferenced_vertices", 0)),
            min_component_area_rel=float(data.get("min_component_area_rel", 0.0)),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
            notes=str(data.get("notes", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> PrecleanDiff:
        source = Path(path)
        if source.is_dir():
            source = source / "preclean_diff.json"
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))


@dataclass
class RetopoMetrics:
    """Comprehensive topology, geometric fidelity, semantic allocation, and UV/bake metrics."""

    # Topology (aligned with MeshMetrics)
    vertex_count: int = 0
    face_count: int = 0
    open_boundary_edges: int = 0
    boundary_loops: int = 0
    pinhole_loops: int = 0
    large_opening_loops: int = 0
    non_manifold_edges: int = 0
    non_manifold_vertices: int = 0
    degenerate_faces: int = 0
    duplicate_faces: int = 0
    connected_components: int = 0
    is_watertight: bool = False
    is_pinhole_free: bool = False
    bbox_min: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bbox_max: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    height: float = 0.0
    # Quad vs triangle decomposition (e.g. AutoRemesher / Tripo reference)
    quad_count: int = 0
    tri_count: int = 0
    equiv_tri_count: int = 0
    # Bidirectional geometric fidelity vs highpoly (normalized space)
    chamfer_distance: float | None = None
    chamfer_high_to_low: float | None = None
    chamfer_low_to_high: float | None = None
    hausdorff_p95: float | None = None
    hausdorff_p99: float | None = None
    sample_count: int = 100000
    # Semantic weighting & mesh distribution diagnostics
    semantic_extract_s: float = 0.0
    part_cluster_count: int = 0
    focus_vertex_ratio: float = 0.0
    head_face_ratio: float = 0.0
    head_core_ratio: float = 0.0
    valence_ge8_ratio: float = 0.0
    aspect_ratio_p95: float = 0.0
    # UV parameterization & Cycles normal bake
    uv_overlap_count: int = 0
    uv_coverage: float = 0.0
    bake_elapsed_s: float = 0.0
    ray_miss_ratio: float = 0.0
    cage_extrusion: float | None = None
    max_ray_distance: float | None = None
    # Timing & status
    decimate_elapsed_s: float = 0.0
    error: str | None = None
    notes: str = ""

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
        self.quad_count = int(self.quad_count)
        self.tri_count = int(self.tri_count)
        self.equiv_tri_count = int(self.equiv_tri_count)
        self.chamfer_distance = _coerce_opt_float(self.chamfer_distance)
        self.chamfer_high_to_low = _coerce_opt_float(self.chamfer_high_to_low)
        self.chamfer_low_to_high = _coerce_opt_float(self.chamfer_low_to_high)
        self.hausdorff_p95 = _coerce_opt_float(self.hausdorff_p95)
        self.hausdorff_p99 = _coerce_opt_float(self.hausdorff_p99)
        self.sample_count = int(self.sample_count)
        self.semantic_extract_s = float(self.semantic_extract_s)
        self.part_cluster_count = int(self.part_cluster_count)
        self.focus_vertex_ratio = float(self.focus_vertex_ratio)
        self.head_face_ratio = float(self.head_face_ratio)
        self.head_core_ratio = float(self.head_core_ratio)
        self.valence_ge8_ratio = float(self.valence_ge8_ratio)
        self.aspect_ratio_p95 = float(self.aspect_ratio_p95)
        self.uv_overlap_count = int(self.uv_overlap_count)
        self.uv_coverage = float(self.uv_coverage)
        self.bake_elapsed_s = float(self.bake_elapsed_s)
        self.ray_miss_ratio = float(self.ray_miss_ratio)
        self.cage_extrusion = _coerce_opt_float(self.cage_extrusion)
        self.max_ray_distance = _coerce_opt_float(self.max_ray_distance)
        self.decimate_elapsed_s = float(self.decimate_elapsed_s)
        self.error = str(self.error) if self.error is not None else None
        self.notes = str(self.notes)

    def to_dict(self) -> dict[str, Any]:
        return _to_json_value(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: str | Path) -> Path:
        return _write_json_file(self.to_dict(), path, "retopo_metrics.json")

    def save(self, path: str | Path) -> Path:
        return self.write_json(path)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetopoMetrics:
        return cls(
            vertex_count=int(data.get("vertex_count", 0)),
            face_count=int(data.get("face_count", 0)),
            open_boundary_edges=int(data.get("open_boundary_edges", 0)),
            boundary_loops=int(data.get("boundary_loops", 0)),
            pinhole_loops=int(data.get("pinhole_loops", 0)),
            large_opening_loops=int(data.get("large_opening_loops", 0)),
            non_manifold_edges=int(data.get("non_manifold_edges", 0)),
            non_manifold_vertices=int(data.get("non_manifold_vertices", 0)),
            degenerate_faces=int(data.get("degenerate_faces", 0)),
            duplicate_faces=int(data.get("duplicate_faces", 0)),
            connected_components=int(data.get("connected_components", 0)),
            is_watertight=bool(data.get("is_watertight", False)),
            is_pinhole_free=bool(data.get("is_pinhole_free", False)),
            bbox_min=list(data.get("bbox_min", [0.0, 0.0, 0.0])),
            bbox_max=list(data.get("bbox_max", [0.0, 0.0, 0.0])),
            height=float(data.get("height", 0.0)),
            quad_count=int(data.get("quad_count", 0)),
            tri_count=int(data.get("tri_count", 0)),
            equiv_tri_count=int(data.get("equiv_tri_count", 0)),
            chamfer_distance=_coerce_opt_float(data.get("chamfer_distance")),
            chamfer_high_to_low=_coerce_opt_float(data.get("chamfer_high_to_low")),
            chamfer_low_to_high=_coerce_opt_float(data.get("chamfer_low_to_high")),
            hausdorff_p95=_coerce_opt_float(data.get("hausdorff_p95")),
            hausdorff_p99=_coerce_opt_float(data.get("hausdorff_p99")),
            sample_count=int(data.get("sample_count", 100000)),
            semantic_extract_s=float(data.get("semantic_extract_s", 0.0)),
            part_cluster_count=int(data.get("part_cluster_count", 0)),
            focus_vertex_ratio=float(data.get("focus_vertex_ratio", 0.0)),
            head_face_ratio=float(data.get("head_face_ratio", 0.0)),
            head_core_ratio=float(data.get("head_core_ratio", 0.0)),
            valence_ge8_ratio=float(data.get("valence_ge8_ratio", 0.0)),
            aspect_ratio_p95=float(data.get("aspect_ratio_p95", 0.0)),
            uv_overlap_count=int(data.get("uv_overlap_count", 0)),
            uv_coverage=float(data.get("uv_coverage", 0.0)),
            bake_elapsed_s=float(data.get("bake_elapsed_s", 0.0)),
            ray_miss_ratio=float(data.get("ray_miss_ratio", 0.0)),
            cage_extrusion=_coerce_opt_float(data.get("cage_extrusion")),
            max_ray_distance=_coerce_opt_float(data.get("max_ray_distance")),
            decimate_elapsed_s=float(data.get("decimate_elapsed_s", 0.0)),
            error=str(data["error"]) if data.get("error") is not None else None,
            notes=str(data.get("notes", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> RetopoMetrics:
        source = Path(path)
        if source.is_dir():
            source = source / "retopo_metrics.json"
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))


@dataclass
class RetopoRunResult:
    """Result record for a single engine/budget/weighting run."""

    config: RetopoJobConfig
    metrics: RetopoMetrics
    status: str = "ok"
    mesh_path: str = ""
    uv_mesh_path: str = ""
    normal_map_path: str = ""
    baked_glb_path: str = ""
    weights_path: str = ""
    render_paths: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.config, Mapping):
            self.config = RetopoJobConfig.from_dict(self.config)
        if isinstance(self.metrics, Mapping):
            self.metrics = RetopoMetrics.from_dict(self.metrics)
        self.status = str(self.status)
        self.mesh_path = str(self.mesh_path)
        self.uv_mesh_path = str(self.uv_mesh_path)
        self.normal_map_path = str(self.normal_map_path)
        self.baked_glb_path = str(self.baked_glb_path)
        self.weights_path = str(self.weights_path)
        self.render_paths = _to_json_value(dict(self.render_paths))
        self.artifacts = _to_json_value(dict(self.artifacts))
        self.elapsed_s = float(self.elapsed_s)
        self.error = str(self.error) if self.error is not None else None
        self.extra = _to_json_value(dict(self.extra))

    def to_dict(self) -> dict[str, Any]:
        return _to_json_value(
            {
                "config": self.config.to_dict(),
                "metrics": self.metrics.to_dict(),
                "status": self.status,
                "mesh_path": self.mesh_path,
                "uv_mesh_path": self.uv_mesh_path,
                "normal_map_path": self.normal_map_path,
                "baked_glb_path": self.baked_glb_path,
                "weights_path": self.weights_path,
                "render_paths": self.render_paths,
                "artifacts": self.artifacts,
                "elapsed_s": self.elapsed_s,
                "error": self.error,
                "extra": self.extra,
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetopoRunResult:
        cfg = data["config"]
        met = data["metrics"]
        return cls(
            config=cfg if isinstance(cfg, RetopoJobConfig) else RetopoJobConfig.from_dict(cfg),
            metrics=met if isinstance(met, RetopoMetrics) else RetopoMetrics.from_dict(met),
            status=str(data.get("status", "ok")),
            mesh_path=str(data.get("mesh_path", "")),
            uv_mesh_path=str(data.get("uv_mesh_path", "")),
            normal_map_path=str(data.get("normal_map_path", "")),
            baked_glb_path=str(data.get("baked_glb_path", "")),
            weights_path=str(data.get("weights_path", "")),
            render_paths=dict(data.get("render_paths", {})),
            artifacts=dict(data.get("artifacts", {})),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
            error=str(data["error"]) if data.get("error") is not None else None,
            extra=dict(data.get("extra", {})),
        )


@dataclass
class StageReport:
    """Aggregated report for a pipeline stage (Stage 1, Stage 2, Stage 3, or combined)."""

    stage: str
    status: str = "ok"
    input_name: str = ""
    input_mesh_path: str = ""
    preclean: PrecleanDiff | None = None
    results: list[RetopoRunResult] = field(default_factory=list)
    winners: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    error: str | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.stage = str(self.stage)
        self.status = str(self.status)
        self.input_name = str(self.input_name)
        self.input_mesh_path = str(self.input_mesh_path)
        if isinstance(self.preclean, Mapping):
            self.preclean = PrecleanDiff.from_dict(self.preclean)
        normalized_results: list[RetopoRunResult] = []
        for item in self.results:
            if isinstance(item, RetopoRunResult):
                normalized_results.append(item)
            elif isinstance(item, Mapping):
                normalized_results.append(RetopoRunResult.from_dict(item))
            else:
                raise TypeError(f"Unsupported result entry in StageReport: {type(item).__name__}")
        self.results = normalized_results
        self.winners = _to_json_value(dict(self.winners))
        self.artifacts = _to_json_value(dict(self.artifacts))
        self.elapsed_s = float(self.elapsed_s)
        self.error = str(self.error) if self.error is not None else None
        self.notes = str(self.notes)
        self.extra = _to_json_value(dict(self.extra))

    def to_dict(self) -> dict[str, Any]:
        return _to_json_value(
            {
                "stage": self.stage,
                "status": self.status,
                "input_name": self.input_name,
                "input_mesh_path": self.input_mesh_path,
                "preclean": self.preclean.to_dict() if self.preclean is not None else None,
                "results": [r.to_dict() for r in self.results],
                "winners": self.winners,
                "artifacts": self.artifacts,
                "elapsed_s": self.elapsed_s,
                "error": self.error,
                "notes": self.notes,
                "extra": self.extra,
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: str | Path) -> Path:
        return _write_json_file(self.to_dict(), path, f"{self.stage}_report.json")

    def save(self, path: str | Path) -> Path:
        return self.write_json(path)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StageReport:
        preclean_raw = data.get("preclean")
        return cls(
            stage=str(data["stage"]),
            status=str(data.get("status", "ok")),
            input_name=str(data.get("input_name", "")),
            input_mesh_path=str(data.get("input_mesh_path", "")),
            preclean=(
                preclean_raw
                if preclean_raw is None or isinstance(preclean_raw, PrecleanDiff)
                else PrecleanDiff.from_dict(preclean_raw)
            ),
            results=[
                item if isinstance(item, RetopoRunResult) else RetopoRunResult.from_dict(item)
                for item in data.get("results", [])
            ],
            winners=dict(data.get("winners", {})),
            artifacts=dict(data.get("artifacts", {})),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
            error=str(data["error"]) if data.get("error") is not None else None,
            notes=str(data.get("notes", "")),
            extra=dict(data.get("extra", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> StageReport:
        source = Path(path)
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))


@dataclass
class ExperimentManifest:
    """Top-level manifest written for an entire local or Colab S2 experiment session."""

    experiment_name: str = "s2_retopo"
    created_at: str = ""
    environment: str = "local"
    gpu_info: dict[str, Any] = field(default_factory=dict)
    blender_version: str = ""
    coordinate_frame: str = "y_up_z_front"
    target_height: float = 1.0
    inputs: dict[str, str] = field(default_factory=dict)
    precleans: dict[str, PrecleanDiff] = field(default_factory=dict)
    stages: dict[str, StageReport] = field(default_factory=dict)
    all_runs: list[RetopoRunResult] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.experiment_name = str(self.experiment_name)
        self.created_at = str(self.created_at)
        self.environment = str(self.environment)
        self.gpu_info = _to_json_value(dict(self.gpu_info))
        self.blender_version = str(self.blender_version)
        self.coordinate_frame = str(self.coordinate_frame)
        self.target_height = float(self.target_height)
        self.inputs = {str(k): str(v) for k, v in self.inputs.items()}
        self.precleans = {
            str(k): (v if isinstance(v, PrecleanDiff) else PrecleanDiff.from_dict(v))
            for k, v in self.precleans.items()
        }
        self.stages = {
            str(k): (v if isinstance(v, StageReport) else StageReport.from_dict(v))
            for k, v in self.stages.items()
        }
        self.all_runs = [
            item if isinstance(item, RetopoRunResult) else RetopoRunResult.from_dict(item)
            for item in self.all_runs
        ]
        self.artifacts = _to_json_value(dict(self.artifacts))
        self.errors = [_to_json_value(dict(e)) for e in self.errors]
        self.elapsed_s = float(self.elapsed_s)
        self.notes = str(self.notes)
        self.extra = _to_json_value(dict(self.extra))

    def to_dict(self) -> dict[str, Any]:
        return _to_json_value(
            {
                "experiment_name": self.experiment_name,
                "created_at": self.created_at,
                "environment": self.environment,
                "gpu_info": self.gpu_info,
                "blender_version": self.blender_version,
                "coordinate_frame": self.coordinate_frame,
                "target_height": self.target_height,
                "inputs": self.inputs,
                "precleans": {k: v.to_dict() for k, v in self.precleans.items()},
                "stages": {k: v.to_dict() for k, v in self.stages.items()},
                "all_runs": [r.to_dict() for r in self.all_runs],
                "artifacts": self.artifacts,
                "errors": self.errors,
                "elapsed_s": self.elapsed_s,
                "notes": self.notes,
                "extra": self.extra,
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write_json(self, path: str | Path) -> Path:
        return _write_json_file(self.to_dict(), path, "experiment_manifest.json")

    def save(self, path: str | Path) -> Path:
        return self.write_json(path)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentManifest:
        return cls(
            experiment_name=str(data.get("experiment_name", "s2_retopo")),
            created_at=str(data.get("created_at", "")),
            environment=str(data.get("environment", "local")),
            gpu_info=dict(data.get("gpu_info", {})),
            blender_version=str(data.get("blender_version", "")),
            coordinate_frame=str(data.get("coordinate_frame", "y_up_z_front")),
            target_height=float(data.get("target_height", 1.0)),
            inputs=dict(data.get("inputs", {})),
            precleans={
                str(k): (v if isinstance(v, PrecleanDiff) else PrecleanDiff.from_dict(v))
                for k, v in data.get("precleans", {}).items()
            },
            stages={
                str(k): (v if isinstance(v, StageReport) else StageReport.from_dict(v))
                for k, v in data.get("stages", {}).items()
            },
            all_runs=[
                item if isinstance(item, RetopoRunResult) else RetopoRunResult.from_dict(item)
                for item in data.get("all_runs", [])
            ],
            artifacts=dict(data.get("artifacts", {})),
            errors=[dict(e) for e in data.get("errors", [])],
            elapsed_s=float(data.get("elapsed_s", 0.0)),
            notes=str(data.get("notes", "")),
            extra=dict(data.get("extra", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> ExperimentManifest:
        source = Path(path)
        if source.is_dir():
            source = source / "experiment_manifest.json"
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))
