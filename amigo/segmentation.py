"""
Mesh segmentation for branching shapes (Section 7.2 of the paper).

When the geodesic distance function f has saddle points, the isolines
of f become multiply-connected.  The algorithm slices the mesh along the
isoline at each saddle (in order of increasing f value), yielding segments
that are each either a half-sphere or a cylinder topology.  Segments are
crocheted in topological-sort order and joined with the join-as-you-go
method (no separate sewing).
"""

from __future__ import annotations

import numpy as np
from collections import deque


def detect_branching(f: np.ndarray, saddles: list[int]) -> bool:
    """True if the mesh has at least one saddle point (→ needs segmentation)."""
    return len(saddles) > 0


def _face_components(F, face_ids: np.ndarray) -> list[np.ndarray]:
    """Split a set of faces into connected components (faces sharing an edge)."""
    face_ids = np.asarray(face_ids)
    if len(face_ids) <= 1:
        return [face_ids]
    # Map each undirected edge to the band-local faces touching it.
    from collections import defaultdict
    edge_faces = defaultdict(list)
    for local, fi in enumerate(face_ids):
        a, b, c = F[fi]
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces[(min(u, v), max(u, v))].append(local)
    parent = list(range(len(face_ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for locals_on_edge in edge_faces.values():
        for k in range(1, len(locals_on_edge)):
            ra, rb = find(locals_on_edge[0]), find(locals_on_edge[k])
            if ra != rb:
                parent[ra] = rb
    groups = defaultdict(list)
    for local in range(len(face_ids)):
        groups[find(local)].append(local)
    return [face_ids[np.asarray(g)] for g in groups.values()]


def segment_by_saddles(V, F, f: np.ndarray, saddles: list[int]):
    """
    Partition mesh faces into segments by slicing at saddle isolines.

    For each saddle σ_i (sorted by f value), we assign each face to the
    segment corresponding to which f-band [f_{i-1}, f_i] it falls in.

    Parameters
    ----------
    V, F     : mesh geometry
    f        : geodesic distance at vertices
    saddles  : list of saddle vertex indices sorted by f value

    Returns
    -------
    segments : list of dicts, each with keys
        'faces'     : (k,) int array of face indices in this segment
        'f_lo'      : lower f bound of the segment
        'f_hi'      : upper f bound of the segment
        'is_first'  : True if this is the initial (seed-side) segment
        'is_last'   : True if this is the terminal (tip) segment
    """
    f_max = float(f.max())
    # Merge saddles that sit at nearly the same f value — clustered saddles
    # (common near a concave junction) otherwise create zero-width / razor-thin
    # bands that shatter into sliver components.
    eps = 0.05 * (f_max - float(f.min()))
    f_vals_at_saddles = []
    for fv in sorted(float(f[s]) for s in saddles):
        if not f_vals_at_saddles or fv - f_vals_at_saddles[-1] > eps:
            f_vals_at_saddles.append(fv)

    # f boundaries between segments
    f_lo_list = [0.0] + f_vals_at_saddles
    f_hi_list = f_vals_at_saddles + [f_max]

    # Face centroid f-values
    f_face = f[F].mean(axis=1)

    # A band-component must be big enough to crochet; tinier ones are slivers
    # from the banding and are skipped (their faces aren't crocheted separately).
    min_faces = max(8, int(0.004 * len(F)))

    segments = []
    n_bands = len(f_lo_list)
    for idx, (f_lo, f_hi) in enumerate(zip(f_lo_list, f_hi_list)):
        mask = (f_face >= f_lo) & (f_face <= f_hi)
        face_ids = np.where(mask)[0]
        if len(face_ids) == 0:
            continue
        # A band above a saddle can contain several disconnected limbs; each is
        # its own segment so every isoline is a single simple loop (otherwise the
        # tracer mixes loops and the crochet graph tangles).
        comps = _face_components(F, face_ids)
        for comp in comps:
            # Keep the band whole if it is a single component (don't drop a
            # small but legitimate base/tip band); only filter slivers when a
            # band actually fragments into several components.
            if len(comp) == 0 or (len(comps) > 1 and len(comp) < min_faces):
                continue
            segments.append({
                "faces": comp,
                "f_lo": f_lo,
                "f_hi": f_hi,
                "is_first": idx == 0,
                "is_last": idx == n_bands - 1,
            })

    return segments


def segment_meshes(V, F, segments: list[dict]):
    """
    For each segment, extract the sub-mesh (V_seg, F_seg) and the local
    vertex index map.

    Returns list of (V_seg, F_seg, global_to_local) dicts.
    """
    result = []
    for seg in segments:
        face_ids = seg["faces"]
        F_seg_global = F[face_ids]
        local_verts = np.unique(F_seg_global)
        g2l = {gv: lv for lv, gv in enumerate(local_verts)}
        F_seg = np.vectorize(g2l.__getitem__)(F_seg_global)
        V_seg = V[local_verts]
        result.append({
            **seg,
            "V": V_seg,
            "F": F_seg,
            "local_verts": local_verts,
            "global_to_local": g2l,
        })
    return result


def find_segment_boundaries(seg_data: list[dict], V, F, f: np.ndarray):
    """
    For each consecutive pair of segments, find the shared boundary vertices
    (the isoline at the saddle between them).

    Returns list of lists of global vertex indices (one per segment boundary).
    """
    boundaries = []
    for i in range(len(seg_data) - 1):
        seg_curr = seg_data[i]
        seg_next = seg_data[i + 1]
        shared = set(seg_curr["local_verts"]).intersection(seg_next["local_verts"])
        boundaries.append(sorted(shared, key=lambda v: f[v]))
    return boundaries


def topological_sort_segments(seg_data: list[dict]) -> list[int]:
    """
    Return segment indices in crochet order (already sorted by f_lo,
    which corresponds to a valid topological sort of G_sigma).
    """
    return list(range(len(seg_data)))
