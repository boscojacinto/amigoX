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


def _split_branches(F, f_face, faces, f_lo, f_hi, is_first, min_faces, depth=0):
    """
    Recursively split a segment where its cross-section bifurcates.

    Saddle detection can miss the level at which a bulbous shape's limbs
    actually separate (e.g. a heart's two lobes). Here we find the lowest
    f-level above which the segment's faces form ≥2 connected components and
    split there: a lower ring segment plus one child segment per limb. Each
    child recurses, so nested branching is handled.
    """
    faces = np.asarray(faces)
    # A real limb must be a substantial fraction of the segment, not a surface
    # bump — this keeps the split count to genuine branches (e.g. a heart's two
    # lobes) instead of shattering bulbous tips into many tiny segments.
    branch_min = max(min_faces, int(0.15 * len(faces)))
    if depth > 6 or len(faces) < 2 * branch_min:
        return [{"faces": faces, "f_lo": f_lo, "f_hi": f_hi, "is_first": is_first}]

    fb = f_face[faces]
    split_L = None
    for L in np.linspace(f_lo, f_hi, 40)[1:-1]:
        upper = faces[fb >= L]
        if len(upper) < 2 * branch_min:
            break
        comps = [c for c in _face_components(F, upper) if len(c) >= branch_min]
        if len(comps) >= 2:
            split_L = L
            break

    if split_L is None:
        return [{"faces": faces, "f_lo": f_lo, "f_hi": f_hi, "is_first": is_first}]

    lower = faces[fb < split_L]
    upper_comps = [c for c in _face_components(F, faces[fb >= split_L])
                   if len(c) >= min_faces]
    out = []
    if len(lower) >= min_faces:
        out.append({"faces": lower, "f_lo": f_lo, "f_hi": split_L,
                    "is_first": is_first})
        child_first = False
    else:
        child_first = is_first  # no lower ring — children inherit root-ness
    for c in upper_comps:
        out.extend(_split_branches(F, f_face, c, split_L, f_hi, child_first,
                                   min_faces, depth + 1))
    return out


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
    eps = 0.006 * (f_max - float(f.min()))
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
            # A component may still bifurcate higher up (bulbous limbs that the
            # saddle pass missed) — split it where its cross-section divides.
            segments.extend(_split_branches(
                F, f_face, comp, f_lo, f_hi, idx == 0, min_faces))

    # is_last: a segment with no higher-f segment sharing its boundary is a tip.
    for s in segments:
        s.setdefault("is_last", False)
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


def build_segment_tree(seg_data: list[dict], seed_idx: int):
    """
    Build the parent/child tree of segments for join-as-you-go crocheting.

    A segment's parent is the lower-f segment it shares the most boundary
    vertices with (the saddle isoline between them). Roots have no parent.
    Returns (parent, children, order) where ``order`` is a DFS preorder
    (parents before children, each limb contiguous) starting at the segment
    that contains the seed.
    """
    n = len(seg_data)
    vsets = [set(int(v) for v in s["local_verts"]) for s in seg_data]
    parent = [-1] * n
    for t in range(n):
        best, best_ov = -1, 0
        for s in range(n):
            if s == t or seg_data[s]["f_lo"] >= seg_data[t]["f_lo"]:
                continue
            ov = len(vsets[s] & vsets[t])
            if ov > best_ov:
                best, best_ov = s, ov
        parent[t] = best

    children = {i: [] for i in range(n)}
    for t in range(n):
        if parent[t] >= 0:
            children[parent[t]].append(t)

    seed_roots = [i for i in range(n) if parent[i] < 0 and seed_idx in vsets[i]]
    other_roots = [i for i in range(n) if parent[i] < 0 and i not in seed_roots]

    order, seen = [], set()

    def dfs(u):
        if u in seen:
            return
        seen.add(u)
        order.append(u)
        for c in sorted(children[u], key=lambda c: len(seg_data[c]["faces"]),
                        reverse=True):
            dfs(c)

    for r in seed_roots + other_roots:
        dfs(r)
    for i in range(n):              # safety: any stragglers
        dfs(i)
    return parent, children, order
