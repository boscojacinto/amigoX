"""
Geodesic distance function f and topological analysis of its level sets.

f(v) = geodesic distance from seed s to v.
Rows of the crochet graph are isolines of f.
Saddle points of f indicate where the isoline topology changes
(branching) — each saddle causes the level set to split.
"""

import numpy as np
import potpourri3d as pp3d

from .mesh_ops import vertex_adjacency


def compute_geodesic_distance(V, F, seed_idx: int, t_scale: float = 1.0):
    """
    Heat-method geodesic distance from seed_idx to all vertices.

    t_scale multiplies the default time parameter (mean-edge-length²).
    The paper recommends increasing t_scale (doubling repeatedly) until
    no neighbouring saddles/extrema remain.
    """
    solver = pp3d.MeshHeatMethodDistanceSolver(V, F, t_coef=t_scale)
    return solver.compute_distance(seed_idx)


def find_maximum(f):
    """Index of the vertex with the largest geodesic distance."""
    return int(np.argmax(f))


def find_saddle_points(V, F, f):
    """
    Detect saddle points of f on the mesh.

    A vertex v is a saddle if the sign pattern of (f[neighbour] - f[v])
    changes sign more than twice as you go around the one-ring,
    i.e. the number of sign alternations in the cyclic neighbour sequence
    is ≥ 4 (two descending and two ascending arcs).

    Returns list of vertex indices sorted by f value.
    """
    adj = vertex_adjacency(F, len(V))
    vf_map = _vertex_to_ordered_faces(V, F)

    saddles = []
    for v in range(len(V)):
        if not adj[v]:
            continue
        ordered_nb = _ordered_one_ring(v, adj[v], vf_map, V, F)
        if ordered_nb is None or len(ordered_nb) < 3:
            continue

        signs = np.sign(f[ordered_nb] - f[v])
        # Remove zeros (flat neighbours — treat as positive for counting)
        signs[signs == 0] = 1
        # Count sign changes in the cyclic sequence
        changes = int(np.sum(np.abs(np.diff(np.append(signs, signs[0]))) > 0))
        if changes >= 4:
            saddles.append(v)

    saddles.sort(key=lambda v: f[v])
    return saddles


def geodesic_path(V, F, src: int, tgt: int):
    """
    Return the sequence of vertex indices along the geodesic from src to tgt
    using the edge-flip method.
    """
    solver = pp3d.EdgeFlipGeodesicSolver(V, F)
    pts = solver.find_geodesic_path(src, tgt)
    # pts is an (n, 3) array of 3-D positions — snap each to nearest vertex
    return _snap_to_vertices(V, pts)


def auto_smooth_geodesic(V, F, seed_idx: int, max_doublings: int = 6):
    """
    Find a t_scale for the heat method such that f has no neighbouring
    saddles/extrema (the paper's robustness step).

    Returns (f, t_scale).
    """
    t = 1.0
    for _ in range(max_doublings):
        f = compute_geodesic_distance(V, F, seed_idx, t_scale=t)
        saddles = find_saddle_points(V, F, f)
        adj = vertex_adjacency(F, len(V))
        # Check for neighbouring saddles or saddle adjacent to extremum
        if not _has_neighbouring_critical(f, saddles, adj):
            return f, t
        t *= 2.0
    return f, t


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vertex_to_ordered_faces(V, F):
    """Map each vertex to its incident face list (unordered)."""
    vf = [[] for _ in range(len(V))]
    for fi, face in enumerate(F):
        for v in face:
            vf[v].append(fi)
    return vf


def _ordered_one_ring(v, neighbours, vf_map, V, F):
    """
    Order the one-ring neighbours of v angularly around v.
    Returns ordered list of vertex indices, or None if mesh is not manifold
    around v.
    """
    # Build a map from unordered neighbour set to ordered ring by traversing
    # adjacent faces around v.
    incident_faces = vf_map[v]
    if not incident_faces:
        return None

    # For each face incident to v, record its two "other" vertices
    face_others = {}
    for fi in incident_faces:
        f = F[fi]
        others = [u for u in f if u != v]
        if len(others) == 2:
            face_others[fi] = tuple(others)

    # Build edge graph: for each pair (a, b) in a face (a, b adjacent to v),
    # there is a "next" relation a -> b in the ring (or b -> a).
    next_in_ring = {}
    for fi, (a, b) in face_others.items():
        # In the face, a and b are consecutive neighbours of v.
        # We don't know the orientation yet; store both.
        next_in_ring.setdefault(a, []).append(b)
        next_in_ring.setdefault(b, []).append(a)

    # Traverse the ring
    start = next(iter(face_others.values()))[0]
    ring = [start]
    prev = v  # pretend we came from v
    curr = start
    for _ in range(len(incident_faces)):
        options = [u for u in next_in_ring.get(curr, []) if u != prev]
        if not options:
            break
        prev = curr
        curr = options[0]
        if curr == start:
            break
        ring.append(curr)

    return ring if len(ring) >= 3 else None


def _has_neighbouring_critical(f, saddles, adj):
    """True if any saddle is adjacent to another critical point."""
    saddle_set = set(saddles)
    max_v = int(np.argmax(f))
    min_v = int(np.argmin(f))
    critical = saddle_set | {max_v, min_v}
    for s in saddles:
        for nb in adj[s]:
            if nb in critical and nb != s:
                return True
    return False


def _snap_to_vertices(V, pts):
    """Snap each point in pts to the nearest mesh vertex. Returns vertex indices."""
    from scipy.spatial import cKDTree
    tree = cKDTree(V)
    _, idx = tree.query(pts)
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for i in idx:
        if i not in seen:
            seen.add(i)
            result.append(int(i))
    return result
