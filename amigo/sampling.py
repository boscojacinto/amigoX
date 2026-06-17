"""
Sample the crochet graph vertices by tracing isolines of f.

For each row level f_i = i * w we:
  1. Find all mesh-edge crossings of the isoline f = f_i.
  2. Trace the isoline face-by-face, starting from the seam crossing,
     to get an ordered sequence of crossing points.
  3. Compute cumulative arc length along that ordered sequence.
  4. Sample at equal arc-length intervals of w to get the crochet graph
     vertices X_G for this row.

The first row (f = 0) is the seed vertex.
The last row  (f = f_max) is the vertex with maximum geodesic distance.

Returns
-------
rows_pos : list of (n_i, 3) ndarrays — 3-D positions of graph vertices
row_f    : list of floats           — f level for each row
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict


def sample_crochet_graph(
    V, F, f: np.ndarray,
    stitch_width: float,
    f_max_idx: int,
    seed_idx: int,
    seam_vertices: list[int],
):
    """
    Sample crochet graph vertices for a single (possibly cut) mesh segment.

    Parameters
    ----------
    V, F          : mesh geometry
    f             : geodesic distance values at vertices
    stitch_width  : w — sampling interval
    f_max_idx     : vertex index of the isoline maximum
    seed_idx      : vertex index of the seed (first row)
    seam_vertices : ordered vertex list of the cut seam (seed → tip)

    Returns
    -------
    rows_pos : list of ndarray (n_i, 3)
    row_f    : list of float
    """
    f_max = float(f[f_max_idx])
    n_rows = max(int(np.floor(f_max / stitch_width)), 1)

    # Pre-build adjacency structures once
    edge_faces = _build_edge_face_map(F)
    seam_edge_set = _seam_to_edge_set(seam_vertices)

    # Pole-to-pole axis (fallback for angle-sort if traversal fails)
    axis = V[f_max_idx] - V[seed_idx]
    axis_len = np.linalg.norm(axis)
    if axis_len > 1e-10:
        axis = axis / axis_len
    else:
        axis = np.array([0.0, 0.0, 1.0])

    seam_pts = V[seam_vertices] if len(seam_vertices) else None

    rows_pos: list[np.ndarray] = []
    row_f: list[float] = []

    # Row 0: seed
    rows_pos.append(V[seed_idx: seed_idx + 1].copy())
    row_f.append(0.0)

    for i in range(1, n_rows):
        f_level = i * stitch_width
        if f_level >= f_max:
            break

        pts = _trace_isoline(V, F, f, f_level, edge_faces, seam_edge_set, axis)
        if len(pts) < 2:
            continue

        pts_arr = np.asarray(pts)
        arc = _arc_lengths(pts_arr)
        total = arc[-1]
        if total < 1e-10:
            continue

        n_stitches = max(int(np.floor(total / stitch_width)), 1)
        sample_s = np.arange(n_stitches) * stitch_width
        sampled = _interp_arc(pts_arr, arc, sample_s)

        # Consistent winding + a common start (the seam) so consecutive loops
        # line up — otherwise DTW couples points across the loop (long edges).
        sampled = _roll_to_seam(_orient_row(sampled, axis), seam_pts)

        rows_pos.append(sampled)
        row_f.append(f_level)

    # Last row: tip
    rows_pos.append(V[f_max_idx: f_max_idx + 1].copy())
    row_f.append(f_max)

    return rows_pos, row_f


# ---------------------------------------------------------------------------
# Isoline tracing
# ---------------------------------------------------------------------------

def _build_edge_face_map(F):
    """Map each undirected edge (u,v), u<v → list of incident face indices."""
    ef: dict[tuple, list] = defaultdict(list)
    for fi, face in enumerate(F):
        for a, b in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            ef[(min(a, b), max(a, b))].append(fi)
    return ef


def _seam_to_edge_set(seam_vertices: list[int]) -> set[tuple]:
    """Convert ordered seam vertex list to set of undirected edges."""
    s = set()
    for i in range(len(seam_vertices) - 1):
        a, b = seam_vertices[i], seam_vertices[i + 1]
        s.add((min(a, b), max(a, b)))
    return s


def _crossing_points(V, F_arr, f: np.ndarray, f_level: float, edge_faces):
    """
    Compute all edge crossing points for the isoline f = f_level.

    Returns dict: (u, v) → 3-D crossing point  (u < v).
    """
    crossings: dict[tuple, np.ndarray] = {}
    for (u, v) in edge_faces:
        fu, fv = f[u], f[v]
        if (fu < f_level < fv) or (fv < f_level < fu):
            t = (f_level - fu) / (fv - fu)
            crossings[(u, v)] = (1 - t) * V[u] + t * V[v]
    return crossings


def _face_crossing_edges(face, crossings):
    """Return the (at most 2) crossing edges in a face."""
    edges = [
        (min(face[0], face[1]), max(face[0], face[1])),
        (min(face[1], face[2]), max(face[1], face[2])),
        (min(face[2], face[0]), max(face[2], face[0])),
    ]
    return [e for e in edges if e in crossings]


def _walk_loop(F, crossings, edge_faces, start_edge):
    """Trace one connected isoline loop from start_edge. Returns (points, visited)."""
    start_faces = edge_faces.get(start_edge, [])
    if not start_faces:
        return [crossings[start_edge]], {start_edge}
    path = [crossings[start_edge]]
    visited = {start_edge}
    current_face_idx = start_faces[0]
    for _ in range(len(crossings) + 1):
        ce = _face_crossing_edges(F[current_face_idx], crossings)
        next_edge = next((e for e in ce if e not in visited), None)
        if next_edge is None:
            break
        path.append(crossings[next_edge])
        visited.add(next_edge)
        nb = [fi for fi in edge_faces.get(next_edge, []) if fi != current_face_idx]
        if not nb:
            break
        current_face_idx = nb[0]
    return path, visited


def _trace_isoline(V, F, f, f_level, edge_faces, seam_edge_set, axis):
    """
    Trace the isoline at f = f_level as a single connected loop.

    An f-level can cross several disconnected loops (e.g. two limbs of one
    segment). Tracing one loop and then appending the other loops' points in
    arbitrary order produces a self-crossing "row" and a tangled column-edge
    mess downstream. Instead we extract every connected loop and return a single
    clean one — the loop the seam passes through (for consistency across rows),
    or the largest loop if the seam does not cross this level.
    """
    crossings = _crossing_points(V, F, f, f_level, edge_faces)
    if not crossings:
        return []

    seam_starts = [e for e in seam_edge_set if e in crossings]
    remaining = set(crossings)
    loops = []  # (points, touches_seam)
    while remaining:
        start = next((e for e in seam_starts if e in remaining), None)
        if start is None:
            start = next(iter(remaining))
        pts, visited = _walk_loop(F, crossings, edge_faces, start)
        if not visited:
            remaining.discard(start)
            continue
        loops.append((pts, bool(visited & seam_edge_set)))
        remaining -= visited

    if not loops:
        return []
    seam_loops = [p for p, touches in loops if touches]
    candidates = seam_loops if seam_loops else [p for p, _ in loops]
    return max(candidates, key=len)


# ---------------------------------------------------------------------------
# Arc-length helpers
# ---------------------------------------------------------------------------

def _orient_row(pts: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Give a loop a consistent winding direction (positive around `axis`)."""
    if len(pts) < 3:
        return pts
    center = pts.mean(axis=0)
    rel = pts - center
    signed = np.dot(np.cross(rel, np.roll(rel, -1, axis=0)).sum(axis=0), axis)
    return pts[::-1].copy() if signed < 0 else pts


def _roll_to_seam(pts: np.ndarray, seam_pts: np.ndarray | None) -> np.ndarray:
    """Rotate a loop so its start is the sample nearest the seam (consistent
    start point across rows ⇒ short, aligned DTW couplings)."""
    if seam_pts is None or len(pts) < 3:
        return pts
    d = np.linalg.norm(pts[:, None, :] - seam_pts[None, :, :], axis=2).min(axis=1)
    return np.roll(pts, -int(np.argmin(d)), axis=0)


def _arc_lengths(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc lengths along a polyline, shape (n,)."""
    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(dists)])


def _interp_arc(pts: np.ndarray, arc: np.ndarray, query_s: np.ndarray) -> np.ndarray:
    """Linearly interpolate positions at query arc-length values."""
    result = np.empty((len(query_s), 3))
    for idx, s in enumerate(query_s):
        j = np.searchsorted(arc, s, side="right") - 1
        j = int(np.clip(j, 0, len(pts) - 2))
        ds = arc[j + 1] - arc[j]
        t = (s - arc[j]) / (ds + 1e-300)
        result[idx] = (1 - t) * pts[j] + t * pts[j + 1]
    return result
