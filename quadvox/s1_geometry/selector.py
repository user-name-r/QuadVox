"""Deterministic multi-seed candidate selector for S1 Pixal3D runs.

Supports two explicit ranking modes:
- `post_repair=False` (default, raw O-Voxel candidates prior to CelloCut):
  ranks by `(large_opening_loops, pinhole_loops, non_manifold_vertices,
  round(open_boundary_edges / max(face_count, 1), 6), non_manifold_edges,
  -face_count, seed)` ascending so raw tears/structural openings and vertex
  pinches dominate before topology cleanup.
- `post_repair=True` (repaired meshes):
  ranks by `(0 if is_watertight else 1, non_manifold_edges, pinhole_loops,
  open_boundary_edges, degenerate_faces, connected_components, -face_count,
  seed)` ascending.
"""

from __future__ import annotations

from typing import Any, Sequence

from quadvox.contracts import SeedCandidate


def candidate_sort_key(
    candidate: SeedCandidate,
    *,
    post_repair: bool = False,
) -> tuple[Any, ...]:
    """Return deterministic ascending sort key (smaller is better)."""
    m = candidate.metrics
    if post_repair:
        return (
            0 if m.is_watertight else 1,
            int(m.non_manifold_edges),
            int(m.pinhole_loops),
            int(m.open_boundary_edges),
            int(m.degenerate_faces),
            int(m.connected_components),
            -int(m.face_count),
            int(candidate.seed),
        )
    return (
        int(m.large_opening_loops),
        int(m.pinhole_loops),
        int(m.non_manifold_vertices),
        round(int(m.open_boundary_edges) / max(int(m.face_count), 1), 6),
        int(m.non_manifold_edges),
        -int(m.face_count),
        int(candidate.seed),
    )


def rank_candidates(
    candidates: Sequence[SeedCandidate],
    *,
    post_repair: bool = False,
) -> list[SeedCandidate]:
    """Return candidates sorted from best to worst."""
    if not candidates:
        raise ValueError("candidates must contain at least one SeedCandidate")
    return sorted(candidates, key=lambda c: candidate_sort_key(c, post_repair=post_repair))


def select_best_candidate(
    candidates: Sequence[SeedCandidate],
    *,
    post_repair: bool = False,
) -> tuple[SeedCandidate, list[SeedCandidate]]:
    """Return `(winner, ranked_candidates)` for `RunMeta.seeds_evaluated`."""
    ranked = rank_candidates(candidates, post_repair=post_repair)
    return ranked[0], ranked
