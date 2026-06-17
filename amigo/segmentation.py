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
    if not saddles:
        # No branching — the whole mesh is one segment
        f_vals_at_saddles = []
    else:
        f_vals_at_saddles = [float(f[s]) for s in saddles]

    # f boundaries between segments
    f_lo_list = [0.0] + f_vals_at_saddles
    f_hi_list = f_vals_at_saddles + [float(f.max())]

    # Face centroid f-values
    f_face = f[F].mean(axis=1)

    segments = []
    for idx, (f_lo, f_hi) in enumerate(zip(f_lo_list, f_hi_list)):
        mask = (f_face >= f_lo) & (f_face <= f_hi)
        face_ids = np.where(mask)[0]
        if len(face_ids) == 0:
            continue
        segments.append({
            "faces": face_ids,
            "f_lo": f_lo,
            "f_hi": f_hi,
            "is_first": idx == 0,
            "is_last": idx == len(f_lo_list) - 1,
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
