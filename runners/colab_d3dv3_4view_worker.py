"""Remote Colab GPU worker for 4-View Direct3D-V3 1536^3 Voxel Fusion (Layer 1 + Layer 2).

Runs strictly inside `/content/cache` (never `/tmp`). Executes:
  1. Optional Layer-2 OpenVDB coarse mesh splicing (`coarse_splice.enabled`)
  2. 4-view Direct3D-V3 (512 -> 1024 -> 1536) lossless 8^3 voxel rotation fusion
  3. Optional S1 repair (CelloCut / inplace_clean) & normalization
  4. Chunked output packaging with SHA-256 verification
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

CACHE_ROOT = Path("/content/cache")
WORK_DIR = CACHE_ROOT / "work"
OUTPUTS_DIR = CACHE_ROOT / "outputs"
SCRATCH_DIR = CACHE_ROOT / "scratch"
MAX_CHUNK_BYTES = 10 * 1024 * 1024

if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

from quadvox.contracts import CoarseSpliceSpec, ROISovereigntyBox, ZoneSmoothingSpec
from quadvox.s1_geometry.coarse_splice import splice_coarse_glbs
from quadvox.s1_geometry.d3dv3_fusion import D3DV3FusionConfig, run_d3dv3_4view_fusion


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(1024 * 1024)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _package_outputs(outputs_dir: Path) -> dict[str, Any]:
    bundle = outputs_dir / "outputs_bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as tf:
        for p in sorted(outputs_dir.iterdir()):
            if p.name.startswith("outputs_bundle") or p.name == "output_manifest.json":
                continue
            tf.add(p, arcname=p.name, recursive=True)

    chunks: list[dict[str, Any]] = []
    idx = 0
    with bundle.open("rb") as src:
        while True:
            data = src.read(MAX_CHUNK_BYTES)
            if not data:
                break
            part_name = f"outputs_bundle.tar.gz.part{idx:03d}"
            part_path = outputs_dir / part_name
            part_path.write_bytes(data)
            chunks.append({"name": part_name, "bytes": len(data), "sha256": _sha256_file(part_path)})
            idx += 1

    manifest = {"bundle_sha256": _sha256_file(bundle), "chunks": chunks}
    (outputs_dir / "output_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_worker(spec: dict[str, Any]) -> dict[str, Any]:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(SCRATCH_DIR)
    os.environ["TEMP"] = str(SCRATCH_DIR)
    os.environ["TMP"] = str(SCRATCH_DIR)

    profile = spec["profile"]
    s1 = profile.get("s1", {})
    fusion_raw = s1.get("d3dv3_fusion", {})

    coarse_path = WORK_DIR / spec["coarse_rel"]
    splice_cfg = CoarseSpliceSpec.from_dict(s1.get("coarse_splice") or {})
    if splice_cfg.enabled and spec.get("secondary_coarse_rel"):
        sec_path = WORK_DIR / spec["secondary_coarse_rel"]
        spliced_glb = OUTPUTS_DIR / "coarse_spliced.glb"
        splice_coarse_glbs(coarse_path, sec_path, spliced_glb, splice_cfg)
        coarse_path = spliced_glb

    roi_boxes: list[ROISovereigntyBox] = []
    if fusion_raw.get("front_roi_box"):
        roi_boxes.append(ROISovereigntyBox.from_dict(fusion_raw["front_roi_box"]))
    for rb in fusion_raw.get("roi_boxes", []):
        roi_boxes.append(ROISovereigntyBox.from_dict(rb))
    smooth_zones = [ZoneSmoothingSpec.from_dict(z) for z in fusion_raw.get("smoothing_zones", [])]

    cfg = D3DV3FusionConfig(
        stage_resolutions=tuple(int(x) for x in fusion_raw.get("stage_resolutions", (512, 1024, 1536))),
        guidance_scale=float(fusion_raw.get("guidance_scale", 7.0)),
        sharpen_t_start=float(fusion_raw.get("sharpen_t_start", 0.55)),
        cfg_decay_t_start=float(fusion_raw.get("cfg_decay_t_start", 0.72)),
        cfg_late_min=float(fusion_raw.get("cfg_late_min", 3.5)),
        roi_boxes=roi_boxes,
        smoothing_zones=smooth_zones,
    )
    view_paths = {vn: WORK_DIR / rel for vn, rel in spec["view_rels"].items()}
    out_glb = OUTPUTS_DIR / "fused_1536.glb"
    res = run_d3dv3_4view_fusion(
        coarse_mesh_path=coarse_path,
        view_paths=view_paths,
        out_glb_path=out_glb,
        config=cfg,
        local_files_only=False,
        target_height=float(s1.get("target_height", 1.0)),
    )
    (OUTPUTS_DIR / "d3dv3_report.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    _package_outputs(OUTPUTS_DIR)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    run_worker(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
