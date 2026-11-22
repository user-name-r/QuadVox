"""Manifold, crack/Z-fighting, pinhole, and orientation repair for LATO.2 prop meshes.

Targets common ComfyUI-LATO2 (`--solidify --surface`) artifacts observed on
hard-surface props (`prop_upgr_1536_qef1_lato2_v3000.glb`):
1. Tiny disconnected floating boxes/tetrahedra (`connected_components > 1`).
2. Non-manifold pinch vertices and non-manifold edges.
3. Self-intersecting / folded slit patches & inverted winding clusters that
   produce visible dark diagonal cracks during Cycles rendering/baking.
4. Open boundaries or degenerate/duplicate faces left after patch cleanup.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np
import pymeshlab
import trimesh


@dataclass(frozen=True)
class PropRepairReport:
    input_vertices: int
    input_faces: int
    output_vertices: int
    output_faces: int
    dropped_components: int
    dropped_fold_slit_faces: int
    open_boundary_edges: int
    non_manifold_edges: int
    non_manifold_vertices: int
    degenerate_faces: int
    duplicate_faces: int
    connected_components: int
    is_watertight: bool
    is_pinhole_free: bool
    signed_volume: float
    remaining_si_fold_faces: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mesh_metrics_basic(mesh: trimesh.Trimesh) -> dict[str, Any]:
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    ef: dict[tuple[int, int], int] = defaultdict(int)
    sorted_faces: set[tuple[int, int, int]] = set()
    dup_faces = 0
    for tri in f:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        key = tuple(sorted((a, b, c)))
        if key in sorted_faces:
            dup_faces += 1
        else:
            sorted_faces.add(key)  # type: ignore[arg-type]
        for u, w in ((a, b), (b, c), (c, a)):
            ef[(min(u, w), max(u, w))] += 1

    open_e = sum(1 for c in ef.values() if c == 1)
    nm_e = sum(1 for c in ef.values() if c > 2)

    v_faces: dict[int, list[int]] = defaultdict(list)
    for fi, tri in enumerate(f):
        for vi in tri:
            v_faces[int(vi)].append(fi)
    nm_v = 0
    for vi, fl in v_faces.items():
        if len(fl) < 2:
            continue
        adj: dict[int, set[int]] = defaultdict(set)
        for fi in fl:
            other = [int(x) for x in f[fi] if int(x) != vi]
            if len(other) == 2:
                adj[other[0]].add(other[1])
                adj[other[1]].add(other[0])
        if not adj:
            continue
        start = next(iter(adj))
        seen = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if len(seen) != len(adj):
            nm_v += 1

    deg_faces = int(np.sum(np.asarray(mesh.area_faces, dtype=np.float64) <= 1e-12)) if len(f) else 0
    comps = len(mesh.split(only_watertight=False)) if len(f) else 0
    is_wt = open_e == 0 and nm_e == 0 and nm_v == 0 and deg_faces == 0 and dup_faces == 0
    return {
        "vertex_count": int(len(v)),
        "face_count": int(len(f)),
        "open_boundary_edges": int(open_e),
        "non_manifold_edges": int(nm_e),
        "non_manifold_vertices": int(nm_v),
        "degenerate_faces": int(deg_faces),
        "duplicate_faces": int(dup_faces),
        "connected_components": int(comps),
        "is_watertight": bool(is_wt),
        "is_pinhole_free": bool(is_wt),
    }


def _dedup_and_clean_faces(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return verts, faces
    seen: set[tuple[int, int, int]] = set()
    keep: list[int] = []
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    for fi, tri in enumerate(faces):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a == b or b == c or a == c or areas[fi] <= 1e-10:
            continue
        key = tuple(sorted((a, b, c)))
        if key in seen:
            continue
        seen.add(key)  # type: ignore[arg-type]
        keep.append(fi)
    tm = trimesh.Trimesh(vertices=verts, faces=faces[keep], process=False)
    tm.remove_unreferenced_vertices()
    return np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int32)


def _collapse_zero_area_faces(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    area_thresh: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Eliminate collinear/zero-area triangles by collapsing their shortest edge cleanly."""
    v = verts.copy()
    f = faces.copy()
    for _ in range(32):
        if len(f) == 0:
            break
        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        deg = np.flatnonzero(~np.isfinite(areas) | (areas <= area_thresh))
        if len(deg) == 0:
            break
        fi = int(deg[0])
        a, b, c = int(f[fi, 0]), int(f[fi, 1]), int(f[fi, 2])
        if a == b or b == c or a == c:
            f = np.delete(f, fi, axis=0)
            continue
        edges = (
            (float(np.linalg.norm(v[a] - v[b])), a, b),
            (float(np.linalg.norm(v[b] - v[c])), b, c),
            (float(np.linalg.norm(v[c] - v[a])), c, a),
        )
        _, u, w = min(edges)
        f[f == w] = u
        keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        f = f[keep]
    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    tm.remove_unreferenced_vertices()
    return np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int32)


def _tri_overlap_2d(p1: np.ndarray, p2: np.ndarray) -> bool:
    def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def _seg_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        o1, o2 = _orient(a, b, c), _orient(a, b, d)
        o3, o4 = _orient(c, d, a), _orient(c, d, b)
        return (o1 * o2 < -1e-10) and (o3 * o4 < -1e-10)

    def _pt_in(pt: np.ndarray, tri: np.ndarray) -> bool:
        o = [_orient(tri[k], tri[(k + 1) % 3], pt) for k in range(3)]
        return (min(o) > 1e-8) or (max(o) < -1e-8)

    for i in range(3):
        a, b = p1[i], p1[(i + 1) % 3]
        for j in range(3):
            if _seg_cross(a, b, p2[j], p2[(j + 1) % 3]):
                return True
    return (
        any(_pt_in(pt, p2) for pt in p1)
        or any(_pt_in(pt, p1) for pt in p2)
        or _pt_in(p1.mean(axis=0), p2)
        or _pt_in(p2.mean(axis=0), p1)
    )


def _find_tight_coplanar_overlaps(
    mesh: trimesh.Trimesh,
    *,
    plane_tol: float = 5e-4,
) -> list[tuple[int, int]]:
    """Find pairs of non-adjacent coplanar triangles colliding within `plane_tol`."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)
    if len(f) < 2:
        return []
    fn = np.asarray(mesh.face_normals, dtype=np.float64)
    fc = v[f].mean(axis=1)
    fmin = v[f].min(axis=1)
    fmax = v[f].max(axis=1)
    pairs: list[tuple[int, int]] = []
    for i in range(len(f)):
        ni = fn[i]
        dots = np.abs(fn @ ni)
        cands = np.where(
            (dots > 0.995)
            & np.all(fmax >= fmin[i] - plane_tol, axis=1)
            & np.all(fmin[i] <= fmax + plane_tol, axis=1)
        )[0]
        for j in cands:
            if j <= i or len(set(f[i]) & set(f[j])) >= 2:
                continue
            if abs(float(np.dot(fc[j] - v[f[i][0]], ni))) > plane_tol:
                continue
            ref = np.array([1.0, 0.0, 0.0]) if abs(ni[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
            u = np.cross(ni, ref)
            norm_u = float(np.linalg.norm(u))
            if norm_u < 1e-12:
                continue
            u /= norm_u
            w = np.cross(ni, u)
            p1 = np.column_stack(((v[f[i]] - v[f[i][0]]) @ u, (v[f[i]] - v[f[i][0]]) @ w))
            p2 = np.column_stack(((v[f[j]] - v[f[i][0]]) @ u, (v[f[j]] - v[f[i][0]]) @ w))
            if _tri_overlap_2d(p1, p2):
                pairs.append((int(i), int(j)))
    return pairs


def _find_si_fold_mask(mesh: trimesh.Trimesh) -> np.ndarray:
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)
    if len(f) == 0:
        return np.zeros(0, dtype=bool)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f))
    ms.apply_filter("compute_selection_by_self_intersections_per_face")
    si = np.asarray(ms.current_mesh().face_selection_array(), dtype=bool)

    dir_edges: dict[tuple[int, int], list[tuple[int, bool]]] = defaultdict(list)
    for fi, tri in enumerate(f):
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            dir_edges[(min(a, b), max(a, b))].append((fi, a < b))
    fold_faces: set[int] = {
        item[0]
        for lst in dir_edges.values()
        if len(lst) == 2 and lst[0][1] == lst[1][1]
        for item in lst
    }
    adj_f = mesh.face_adjacency
    fn = np.asarray(mesh.face_normals, dtype=np.float64)
    if len(adj_f):
        dots = np.clip((fn[adj_f[:, 0]] * fn[adj_f[:, 1]]).sum(axis=1), -1.0, 1.0)
        high_fold = np.where(np.degrees(np.arccos(dots)) > 170.0)[0]
        for idx in high_fold:
            fold_faces.add(int(adj_f[idx, 0]))
            fold_faces.add(int(adj_f[idx, 1]))

    return np.array([bool(si[i]) and (i in fold_faces) for i in range(len(f))], dtype=bool)


def compute_outward_flip_mask(
    centers: np.ndarray,
    face_normals: np.ndarray,
    ref_mesh: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute per-face flip mask aligning triangle orientation outward vs `ref_mesh`."""
    n_faces = len(centers)
    if n_faces == 0:
        return np.zeros(0, dtype=bool), {"used_open3d": False, "flipped_faces": 0, "ambig_after_occ": 0}

    ref = trimesh.load(str(ref_mesh), force="mesh", process=False)
    if not isinstance(ref, trimesh.Trimesh) or len(ref.faces) == 0:
        return np.zeros(n_faces, dtype=bool), {"used_open3d": False, "flipped_faces": 0, "ambig_after_occ": n_faces}

    c_l = np.asarray(centers, dtype=np.float64)
    n_l = np.asarray(face_normals, dtype=np.float64)
    ref_verts = np.asarray(ref.vertices, dtype=np.float64)
    ref_faces = np.asarray(ref.faces, dtype=np.int32)
    ref_fn = np.asarray(ref.face_normals, dtype=np.float64)

    try:
        import open3d as o3d  # type: ignore
    except Exception:
        o3d = None

    if o3d is not None:
        tmesh = o3d.t.geometry.TriangleMesh()
        tmesh.vertex.positions = o3d.core.Tensor(ref_verts.astype(np.float32))
        tmesh.triangle.indices = o3d.core.Tensor(ref_faces.astype(np.int32))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)

        bbox_diag = float(np.linalg.norm(ref_verts.max(axis=0) - ref_verts.min(axis=0)))
        scale = max(0.5 * bbox_diag, 1e-4)
        flip = np.zeros(n_faces, dtype=bool)
        ambig = np.ones(n_faces, dtype=bool)

        for rel_eps in (0.002, 0.005, 0.010, 0.020, 0.045):
            if not np.any(ambig):
                break
            eps = float(rel_eps) * scale
            idx_a = np.flatnonzero(ambig)
            c_sub = c_l[idx_a]
            n_sub = n_l[idx_a]
            oo = scene.compute_occupancy(
                o3d.core.Tensor((c_sub + eps * n_sub).astype(np.float32))
            ).numpy()
            oi = scene.compute_occupancy(
                o3d.core.Tensor((c_sub - eps * n_sub).astype(np.float32))
            ).numpy()
            si = (oo > 0.5) & (oi < 0.5)
            so = (oo < 0.5) & (oi > 0.5)
            flip[idx_a[si]] = True
            ambig[idx_a[si | so]] = False

        rem_ambig = int(ambig.sum())
        if rem_ambig > 0:
            idx_a = np.flatnonzero(ambig)
            cp = scene.compute_closest_points(o3d.core.Tensor(c_l[idx_a].astype(np.float32)))
            tids = cp["primitive_ids"].numpy()
            dots = np.einsum("ij,ij->i", n_l[idx_a], ref_fn[tids])
            flip[idx_a[dots < 0.0]] = True

        return flip, {
            "used_open3d": True,
            "flipped_faces": int(flip.sum()),
            "ambig_after_occ": rem_ambig,
        }

    from scipy.spatial import cKDTree

    c_r = np.asarray(ref.triangles_center, dtype=np.float64)
    _, idx = cKDTree(c_r).query(c_l)
    dots = (n_l * ref_fn[idx]).sum(axis=1)
    flip = dots < -0.3
    return flip, {
        "used_open3d": False,
        "flipped_faces": int(flip.sum()),
        "ambig_after_occ": n_faces,
    }


def _align_normals_to_ref(mesh: trimesh.Trimesh, ref_mesh: str | Path) -> trimesh.Trimesh:
    """Flip individual inverted faces whose normals oppose reference exterior."""
    if len(mesh.faces) == 0:
        return mesh
    flip, _ = compute_outward_flip_mask(mesh.triangles_center, mesh.face_normals, ref_mesh)
    if not np.any(flip):
        return mesh
    f = np.asarray(mesh.faces, dtype=np.int32).copy()
    f[flip] = f[flip][:, [0, 2, 1]]
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices, dtype=np.float64), faces=f, process=False)


def _node_world_matrices(gltf: dict[str, Any]) -> dict[int, np.ndarray]:
    """Compute 4x4 world transforms keyed by mesh index from glTF scene nodes."""
    nodes = gltf.get("nodes", [])
    if not nodes:
        return {}
    children_set = {ch for nd in nodes for ch in nd.get("children", [])}
    roots = [i for i in range(len(nodes)) if i not in children_set]

    def _local_mat(nd: dict[str, Any]) -> np.ndarray:
        if "matrix" in nd:
            return np.asarray(nd["matrix"], dtype=np.float64).reshape(4, 4).T
        mat = np.eye(4, dtype=np.float64)
        if "scale" in nd:
            sx, sy, sz = nd["scale"]
            mat = mat @ np.diag([sx, sy, sz, 1.0])
        if "rotation" in nd:
            qx, qy, qz, qw = nd["rotation"]
            rot = trimesh.transformations.quaternion_matrix([qw, qx, qy, qz])
            mat = rot @ mat
        if "translation" in nd:
            tx, ty, tz = nd["translation"]
            tmat = np.eye(4, dtype=np.float64)
            tmat[:3, 3] = [tx, ty, tz]
            mat = tmat @ mat
        return mat

    out: dict[int, np.ndarray] = {}

    def _walk(idx: int, parent_m: np.ndarray) -> None:
        nd = nodes[idx]
        cur_m = parent_m @ _local_mat(nd)
        if "mesh" in nd and nd["mesh"] not in out:
            out[int(nd["mesh"])] = cur_m
        for ch in nd.get("children", []):
            _walk(int(ch), cur_m)

    for r in roots:
        _walk(r, np.eye(4, dtype=np.float64))
    return out


def fix_glb_outward_winding_and_strip_normals(
    glb_path: str | Path,
    ref_mesh: str | Path,
    *,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Flip inverted triangles in-place inside a GLB while preserving UV/vertex layout verbatim.

    Also strips `NORMAL` and `TANGENT` attributes so Blender recomputes clean smooth shading
    (`has_custom_normals == False`) from the repaired outward triangle windings.
    """
    src = Path(glb_path)
    dst = Path(out_path) if out_path is not None else src
    raw = src.read_bytes()
    if len(raw) < 28:
        raise ValueError(f"Invalid GLB file (too small): {src}")
    magic, version, total_len = struct.unpack("<III", raw[:12])
    if magic != 0x46546C67 or version != 2:
        raise ValueError(f"Unsupported GLB header in {src}: magic={hex(magic)} ver={version}")

    c0_len, c0_type = struct.unpack("<II", raw[12:20])
    if c0_type != 0x4E4F534A:
        raise ValueError(f"Expected JSON chunk first in {src}")
    gltf = json.loads(raw[20 : 20 + c0_len].decode("utf-8"))

    bin_off = 20 + c0_len
    if bin_off + 8 > len(raw):
        raise ValueError(f"Missing BIN chunk in {src}")
    c1_len, c1_type = struct.unpack("<II", raw[bin_off : bin_off + 8])
    if c1_type != 0x004E4942:
        raise ValueError(f"Expected BIN chunk second in {src}")
    bin_buf = bytearray(raw[bin_off + 8 : bin_off + 8 + c1_len])

    accessors = gltf.get("accessors", [])
    buffer_views = gltf.get("bufferViews", [])
    mesh_mats = _node_world_matrices(gltf)

    total_faces = 0
    flipped_faces = 0
    stripped_normals = 0
    ambig_total = 0
    used_o3d = False

    dtype_map = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32}
    for mi, mesh_obj in enumerate(gltf.get("meshes", [])):
        m4 = mesh_mats.get(mi, np.eye(4, dtype=np.float64))
        for prim in mesh_obj.get("primitives", []):
            if prim.get("mode", 4) != 4:
                continue
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs or "indices" not in prim:
                continue

            if "NORMAL" in attrs:
                del attrs["NORMAL"]
                stripped_normals += 1
            attrs.pop("TANGENT", None)

            pos_acc = accessors[attrs["POSITION"]]
            pos_bv = buffer_views[pos_acc["bufferView"]]
            pos_start = pos_bv.get("byteOffset", 0) + pos_acc.get("byteOffset", 0)
            pos_stride = pos_bv.get("byteStride") or 12
            v_cnt = int(pos_acc["count"])
            if pos_stride == 12:
                verts = np.frombuffer(bin_buf, dtype=np.float32, count=v_cnt * 3, offset=pos_start).reshape(v_cnt, 3).astype(np.float64)
            else:
                raw_slice = np.frombuffer(bin_buf, dtype=np.uint8, count=v_cnt * pos_stride, offset=pos_start)
                verts = np.ndarray(shape=(v_cnt, 3), dtype=np.float32, buffer=raw_slice, strides=(pos_stride, 4)).astype(np.float64)

            verts_w = (m4[:3, :3] @ verts.T).T + m4[:3, 3]

            idx_acc = accessors[prim["indices"]]
            idx_bv = buffer_views[idx_acc["bufferView"]]
            idx_start = idx_bv.get("byteOffset", 0) + idx_acc.get("byteOffset", 0)
            idx_dt = dtype_map.get(int(idx_acc["componentType"]))
            if idx_dt is None:
                raise ValueError(f"Unsupported glTF index componentType={idx_acc['componentType']}")
            i_cnt = int(idx_acc["count"])
            if i_cnt % 3 != 0:
                raise ValueError(f"Non-triangle index count {i_cnt} in {src}")
            n_tris = i_cnt // 3
            idx_arr = np.frombuffer(bin_buf, dtype=idx_dt, count=i_cnt, offset=idx_start).reshape(n_tris, 3).copy()

            tri_pts = verts_w[idx_arr]
            centers = tri_pts.mean(axis=1)
            cross = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
            norms = np.linalg.norm(cross, axis=1, keepdims=True)
            face_normals = cross / np.maximum(norms, 1e-12)

            flip, stats = compute_outward_flip_mask(centers, face_normals, ref_mesh)
            used_o3d = used_o3d or bool(stats["used_open3d"])
            ambig_total += int(stats["ambig_after_occ"])
            total_faces += n_tris
            if np.any(flip):
                idx_arr[flip] = idx_arr[flip][:, [0, 2, 1]]
                flipped_faces += int(flip.sum())
                idx_bytes = idx_arr.astype(idx_dt, copy=False).tobytes()
                bin_buf[idx_start : idx_start + len(idx_bytes)] = idx_bytes

    json_bytes = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_pad = (-len(json_bytes)) % 4
    json_bytes += b" " * json_pad
    bin_bytes = bytes(bin_buf)
    bin_pad = (-len(bin_bytes)) % 4
    bin_bytes += b"\x00" * bin_pad

    out_total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    packed = (
        struct.pack("<III", 0x46546C67, 2, out_total_len)
        + struct.pack("<II", len(json_bytes), 0x4E4F534A)
        + json_bytes
        + struct.pack("<II", len(bin_bytes), 0x004E4942)
        + bin_bytes
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(packed)
    return {
        "total_faces": int(total_faces),
        "flipped_faces": int(flipped_faces),
        "stripped_normal_attrs": int(stripped_normals),
        "ambig_after_occ": int(ambig_total),
        "used_open3d": bool(used_o3d),
    }


def repair_prop_lato2_mesh(
    in_mesh: str | Path,
    out_mesh: str | Path,
    *,
    ref_mesh: str | Path | None = None,
    max_hole_edges: int = 300,
    strip_si_fold_slits: bool = True,
    strip_coplanar_zfight: bool = True,
    max_iters: int = 10,
) -> PropRepairReport:
    """Repair LATO.2 prop mesh into a single watertight, zero-NM manifold shell."""
    loaded = trimesh.load(str(in_mesh), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Expected single Trimesh from {in_mesh}")

    in_v, in_f = int(len(loaded.vertices)), int(len(loaded.faces))
    comps = loaded.split(only_watertight=False)
    if not comps:
        raise ValueError(f"Empty mesh in {in_mesh}")
    dropped_components = max(0, len(comps) - 1)
    tm = max(comps, key=lambda c: float(c.area))
    trimesh.repair.fix_normals(tm)

    dropped_fold_slit_faces = 0
    if strip_coplanar_zfight:
        pairs = _find_tight_coplanar_overlaps(tm, plane_tol=5e-4)
        if pairs:
            areas = np.asarray(tm.area_faces, dtype=np.float64)
            active = list(pairs)
            drop_set: set[int] = set()
            while active:
                cnt = Counter(x for a, b in active for x in (a, b))
                victim = max(cnt.keys(), key=lambda idx: (cnt[idx], -float(areas[idx]), idx))
                drop_set.add(victim)
                active = [(a, b) for a, b in active if a != victim and b != victim]
            if len(drop_set) < len(tm.faces) // 3:
                keep_m = np.ones(len(tm.faces), dtype=bool)
                for idx in drop_set:
                    keep_m[idx] = False
                dropped_fold_slit_faces += int(len(drop_set))
                v_kept, f_kept = _dedup_and_clean_faces(
                    np.asarray(tm.vertices, dtype=np.float64),
                    np.asarray(tm.faces[keep_m], dtype=np.int32),
                )
                tm = trimesh.Trimesh(vertices=v_kept, faces=f_kept, process=False)

    if strip_si_fold_slits:
        fold_mask = _find_si_fold_mask(tm)
        n_fold = int(fold_mask.sum())
        dropped_fold_slit_faces += n_fold
        if n_fold > 0 and n_fold < len(tm.faces) // 3:
            v_kept, f_kept = _dedup_and_clean_faces(
                np.asarray(tm.vertices, dtype=np.float64),
                np.asarray(tm.faces[~fold_mask], dtype=np.int32),
            )
            tm = trimesh.Trimesh(vertices=v_kept, faces=f_kept, process=False)

    for _ in range(max_iters):
        v_cur, f_cur = _collapse_zero_area_faces(
            np.asarray(tm.vertices, dtype=np.float64),
            np.asarray(tm.faces, dtype=np.int32),
        )
        v_cur, f_cur = _dedup_and_clean_faces(v_cur, f_cur)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v_cur, face_matrix=f_cur))
        ms.apply_filter("meshing_remove_unreferenced_vertices")
        ms.apply_filter("meshing_remove_duplicate_faces")
        ms.apply_filter("meshing_repair_non_manifold_vertices")
        ms.apply_filter("meshing_repair_non_manifold_edges")

        cm = ms.current_mesh()
        v_cm, f_cm = _collapse_zero_area_faces(
            np.asarray(cm.vertex_matrix(), dtype=np.float64),
            np.asarray(cm.face_matrix(), dtype=np.int32),
        )
        tm = trimesh.Trimesh(vertices=v_cm, faces=f_cm, process=False)
        sub_comps = tm.split(only_watertight=False)
        if len(sub_comps) > 1:
            tm = max(sub_comps, key=lambda c: float(c.area))

        mets = _mesh_metrics_basic(tm)
        if (
            mets["is_watertight"]
            and mets["is_pinhole_free"]
            and mets["connected_components"] == 1
        ):
            break

        if mets["open_boundary_edges"] > 0:
            ms_close = pymeshlab.MeshSet()
            ms_close.add_mesh(
                pymeshlab.Mesh(
                    vertex_matrix=np.asarray(tm.vertices, dtype=np.float64),
                    face_matrix=np.asarray(tm.faces, dtype=np.int32),
                )
            )
            ms_close.apply_filter("meshing_repair_non_manifold_vertices")
            ms_close.apply_filter("meshing_repair_non_manifold_edges")
            ms_close.apply_filter(
                "meshing_close_holes",
                maxholesize=int(max_hole_edges),
                newfaceselected=False,
                selfintersection=False,
            )
            ms_close.apply_filter("meshing_repair_non_manifold_vertices")
            ms_close.apply_filter("meshing_repair_non_manifold_edges")
            cm2 = ms_close.current_mesh()
            v_c2, f_c2 = _collapse_zero_area_faces(
                np.asarray(cm2.vertex_matrix(), dtype=np.float64),
                np.asarray(cm2.face_matrix(), dtype=np.int32),
            )
            tm = trimesh.Trimesh(vertices=v_c2, faces=f_c2, process=False)

    trimesh.repair.fix_normals(tm)
    if float(tm.volume) < 0.0:
        tm.invert()
    if ref_mesh is not None and Path(ref_mesh).is_file():
        tm = _align_normals_to_ref(tm, ref_mesh)

    final_mets = _mesh_metrics_basic(tm)
    rem_si_fold = int(_find_si_fold_mask(tm).sum())

    out_path = Path(out_mesh)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tm.export(str(out_path))

    return PropRepairReport(
        input_vertices=in_v,
        input_faces=in_f,
        output_vertices=final_mets["vertex_count"],
        output_faces=final_mets["face_count"],
        dropped_components=dropped_components,
        dropped_fold_slit_faces=dropped_fold_slit_faces,
        open_boundary_edges=final_mets["open_boundary_edges"],
        non_manifold_edges=final_mets["non_manifold_edges"],
        non_manifold_vertices=final_mets["non_manifold_vertices"],
        degenerate_faces=final_mets["degenerate_faces"],
        duplicate_faces=final_mets["duplicate_faces"],
        connected_components=final_mets["connected_components"],
        is_watertight=final_mets["is_watertight"],
        is_pinhole_free=final_mets["is_pinhole_free"],
        signed_volume=round(float(tm.volume), 6),
        remaining_si_fold_faces=rem_si_fold,
    )
