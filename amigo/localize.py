"""
Problem localizers for the Phase-3 crochetability editor.

``localize(V, F)`` turns the crochetability diagnostics into a live red/green
checklist plus a list of *localized* problems — each carrying the specific
geometry responsible (vertex / face / loop / component ids) and the editor tool
that fixes it. This is the deterministic bridge that lets the editor
auto-highlight culprits and arm the matching tool (no LLM needed).

Problem schema:
    {
      "type": "open_boundary" | "multiple_components" | "thin_region"
            | "noisy_saddles" | "handle",
      "severity": "high" | "medium",
      "region": {"kind": "loop"|"component"|"vertices"|"faces", "ids": [...]},
      "suggested_tool": "fill_loop" | "delete_component" | "inflate"
            | "local_smooth" | "cut_handle",
      "detail": "...",
    }
"""

from __future__ import annotations

import numpy as np
import trimesh

from .diagnostics import _principal_pole_seed
from .geodesics import auto_smooth_geodesic, find_saddle_points
from .mesh_ops import normalize_to_unit_area
from .pipeline import amigo_pipeline_data


def _edge_loops(open_edges: np.ndarray) -> list[list[int]]:
    """Order a set of boundary edges into vertex cycles."""
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in open_edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    loops, seen_edges = [], set()
    for start in list(adj.keys()):
        for nxt in adj[start]:
            if (start, nxt) in seen_edges or (nxt, start) in seen_edges:
                continue
            loop = [start]
            prev, cur = start, nxt
            seen_edges.add((start, nxt))
            while cur != start and len(loop) < len(adj) + 1:
                loop.append(cur)
                nbrs = [v for v in adj[cur] if v != prev]
                if not nbrs:
                    break
                prev, nxt2 = cur, nbrs[0]
                seen_edges.add((cur, nxt2))
                cur = nxt2
            if len(loop) >= 3:
                loops.append(loop)
    return loops


def boundary_loops(V, F) -> list[list[int]]:
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    idx = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    if len(idx) == 0:
        return []
    return _edge_loops(mesh.edges_sorted[idx])


def components(V, F) -> list[list[int]]:
    """Return per-component face-id lists (global indices), largest first."""
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return []
    # Map each split part's faces back to global face ids via face centroids.
    centroids = mesh.triangles_center
    from scipy.spatial import cKDTree
    tree = cKDTree(centroids)
    out = []
    for p in parts:
        _, ids = tree.query(p.triangles_center)
        out.append([int(i) for i in np.unique(ids)])
    out.sort(key=len, reverse=True)
    return out


def thin_regions(V, F, seed: int, stitch_width: float) -> list[int]:
    """Vertex ids near stitch rows that fall below the crochetable thresholds."""
    Vn, _ = normalize_to_unit_area(V, F)
    try:
        data = amigo_pipeline_data(Vn, F, seed_idx=seed,
                                   stitch_width=stitch_width, verbose=False)
    except Exception:  # noqa: BLE001
        return []
    thin_pts = []
    for s in data["segments"]:
        rows = s["rows"]
        counts = [len(r) for r in rows]
        interior = counts[1:-1] if len(counts) > 2 else counts
        if (len(rows) < 2) or (interior and min(interior) < 3):
            for r in rows:
                thin_pts.extend(r)
    if not thin_pts:
        return []
    from scipy.spatial import cKDTree
    tree = cKDTree(V)
    _, ids = tree.query(np.asarray(thin_pts))
    return [int(i) for i in np.unique(ids)]


def _fundamental_cycle(u, v, parent, depth):
    """Cycle = tree path u→v (via LCA) plus the edge (u,v)."""
    pu, pv = [], []
    a, b = u, v
    while depth[a] > depth[b]:
        pu.append(a); a = parent[a]
    while depth[b] > depth[a]:
        pv.append(b); b = parent[b]
    while a != b:
        pu.append(a); a = parent[a]
        pv.append(b); b = parent[b]
    return pu + [a] + pv[::-1]


def handle_loops(V, F, genus) -> list[list[int]]:
    """Non-contractible (tunnel) loops when genus > 0, via tree–cotree.

    Builds a spanning tree of the vertex graph and a spanning tree of the dual
    (face) graph from the remaining edges; the edges in neither tree are the
    homology generators. Each generator's fundamental cycle is a tunnel loop.
    The editor also offers manual loop-pick as a fallback.
    """
    if not genus or genus <= 0:
        return []
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)

    # Primal spanning tree (BFS) over the vertex graph.
    from collections import deque, defaultdict
    adj = defaultdict(list)
    for a, b in mesh.edges_unique:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    n = len(V)
    parent = [-1] * n
    depth = [0] * n
    seen = [False] * n
    tree_edges = set()
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = True
        q = deque([s])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    parent[y] = x
                    depth[y] = depth[x] + 1
                    tree_edges.add(frozenset((x, y)))
                    q.append(y)

    # Dual spanning tree over faces, using only primal edges NOT in the tree.
    fa = mesh.face_adjacency
    fae = mesh.face_adjacency_edges
    uf = list(range(len(mesh.faces)))

    def find(i):
        while uf[i] != i:
            uf[i] = uf[uf[i]]
            i = uf[i]
        return i

    generators = []
    for (f0, f1), (eu, ev) in zip(fa, fae):
        if frozenset((int(eu), int(ev))) in tree_edges:
            continue
        r0, r1 = find(int(f0)), find(int(f1))
        if r0 != r1:
            uf[r0] = r1            # add dual edge to dual tree
        else:
            generators.append((int(eu), int(ev)))  # creates dual cycle → generator

    loops = [_fundamental_cycle(u, v, parent, depth) for u, v in generators]
    loops = [lp for lp in loops if len(lp) >= 3]
    loops.sort(key=len)
    return loops[:max(genus, 1)]


def localize(V, F, seed: int | None = None, stitch_width: float = 0.05) -> dict:
    """Compute the crochetability checklist and a list of localized problems."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    if seed is None:
        seed = _principal_pole_seed(V)

    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    watertight = bool(mesh.is_watertight)
    euler = int(mesh.euler_number)
    genus = (2 - euler) // 2 if watertight else None
    try:
        n_comp = len(mesh.split(only_watertight=False))
    except Exception:  # noqa: BLE001
        n_comp = 1

    f, _ = auto_smooth_geodesic(V, F, seed)
    saddles = find_saddle_points(V, F, f)

    problems: list[dict] = []

    comp = components(V, F)
    if len(comp) > 1:
        # smaller components are the ones to remove
        for cids in comp[1:]:
            problems.append({
                "type": "multiple_components", "severity": "high",
                "region": {"kind": "faces", "ids": cids},
                "suggested_tool": "delete_component",
                "detail": "Disconnected piece — crochet needs one component.",
            })

    for loop in boundary_loops(V, F):
        problems.append({
            "type": "open_boundary", "severity": "high",
            "region": {"kind": "loop", "ids": loop},
            "suggested_tool": "fill_loop",
            "detail": "Open boundary — mesh must be watertight.",
        })

    for loop in handle_loops(V, F, genus):
        problems.append({
            "type": "handle", "severity": "high",
            "region": {"kind": "loop", "ids": loop},
            "suggested_tool": "cut_handle",
            "detail": "Handle/tunnel (genus > 0) — isolines stop being simple loops.",
        })

    thin = thin_regions(V, F, seed, stitch_width)
    if thin:
        problems.append({
            "type": "thin_region", "severity": "medium",
            "region": {"kind": "vertices", "ids": thin},
            "suggested_tool": "inflate",
            "detail": "Feature too thin for stitches (rows < 2 or < 3 stitches/row).",
        })

    if len(saddles) > 6:
        problems.append({
            "type": "noisy_saddles", "severity": "medium",
            "region": {"kind": "vertices", "ids": [int(s) for s in saddles]},
            "suggested_tool": "local_smooth",
            "detail": f"{len(saddles)} saddles — surface noise fragments the field.",
        })

    checklist = {
        "watertight": {"ok": watertight, "value": watertight},
        "single_component": {"ok": n_comp == 1, "value": n_comp},
        "genus_zero": {"ok": genus == 0, "value": genus},
        "few_saddles": {"ok": len(saddles) <= 6, "value": len(saddles)},
        "thick_enough": {"ok": not thin, "value": len(thin)},
    }
    return {"seed": int(seed), "checklist": checklist, "problems": problems}
