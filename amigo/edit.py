"""
Mesh-editing operations for the Phase-3 crochetability editor.

Each op takes ``(V, F, **params)`` and returns ``(V2, F2, report)``. Reshape
tools (move / scale / inflate) are previewed on the client and committed through
``set_vertex_positions`` so the preview matches the result exactly; topological
tools (fill / delete / smooth / cut) run here authoritatively.

``apply(op, V, F, params)`` dispatches and attaches before/after diagnostics so
the caller can detect a regression and offer undo.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .diagnostics import analyze_mesh
from .localize import boundary_loops
from .mesh_ops import vertex_adjacency


def _mesh(V, F):
    return trimesh.Trimesh(vertices=np.asarray(V, dtype=np.float64),
                           faces=np.asarray(F, dtype=np.int64), process=False)


def _out(m):
    return (np.asarray(m.vertices, dtype=np.float64),
            np.asarray(m.faces, dtype=np.int64))


def set_vertex_positions(V, F, ids, positions, **_):
    """Move a subset of vertices to client-computed positions (move/scale/inflate)."""
    V2 = np.asarray(V, dtype=np.float64).copy()
    ids = np.asarray(ids, dtype=np.int64)
    V2[ids] = np.asarray(positions, dtype=np.float64)
    return V2, np.asarray(F, dtype=np.int64), {
        "op": "set_vertex_positions", "n_moved": int(len(ids))}


def inflate(V, F, ids, distance: float = 0.02, **_):
    """Offset selected vertices along their outward normals (server-side thicken)."""
    m = _mesh(V, F)
    n = np.asarray(m.vertex_normals)
    V2 = np.asarray(V, dtype=np.float64).copy()
    ids = np.asarray(ids, dtype=np.int64)
    V2[ids] = V2[ids] + n[ids] * float(distance)
    return V2, np.asarray(F, dtype=np.int64), {
        "op": "inflate", "n_moved": int(len(ids)), "distance": float(distance)}


def local_smooth(V, F, ids, iterations: int = 8, lam: float = 0.5, **_):
    """Laplacian smoothing restricted to a vertex subset (the rest are pinned)."""
    V2 = np.asarray(V, dtype=np.float64).copy()
    adj = vertex_adjacency(F, len(V2))
    ids = [int(i) for i in ids]
    for _ in range(int(iterations)):
        new = V2.copy()
        for i in ids:
            nb = adj[i]
            if nb:
                new[i] = (1 - lam) * V2[i] + lam * V2[list(nb)].mean(axis=0)
        V2 = new
    return V2, np.asarray(F, dtype=np.int64), {
        "op": "local_smooth", "n_smoothed": len(ids), "iterations": int(iterations)}


def _fill_one_loop(V, F, loop):
    """Cap one boundary loop by centroid-fan triangulation."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    loop = [int(i) for i in loop]
    c_idx = len(V)
    centroid = V[loop].mean(axis=0)
    V2 = np.vstack([V, centroid[None, :]])
    fan = [[loop[i], loop[(i + 1) % len(loop)], c_idx] for i in range(len(loop))]
    F2 = np.vstack([F, np.asarray(fan, dtype=np.int64)])
    return V2, F2


def fill_loop(V, F, loop_ids, **_):
    """Close a single boundary loop and fix winding so the mesh is watertight."""
    before = bool(_mesh(V, F).is_watertight)
    V2, F2 = _fill_one_loop(V, F, loop_ids)
    m = _mesh(V2, F2)
    trimesh.repair.fix_normals(m)
    V2, F2 = _out(m)
    return V2, F2, {"op": "fill_loop", "watertight_before": before,
                    "watertight_after": bool(m.is_watertight)}


def _delete_faces(V, F, face_ids):
    m = _mesh(V, F)
    mask = np.ones(len(m.faces), dtype=bool)
    mask[np.asarray(face_ids, dtype=np.int64)] = False
    m.update_faces(mask)
    m.remove_unreferenced_vertices()
    return _out(m)


def delete_faces(V, F, face_ids, **_):
    before = int(len(F))
    V2, F2 = _delete_faces(V, F, face_ids)
    return V2, F2, {"op": "delete_faces", "faces_before": before,
                    "faces_after": int(len(F2))}


def delete_component(V, F, face_ids, **_):
    """Alias for delete_faces — the localizer supplies a component's face ids."""
    V2, F2, rep = delete_faces(V, F, face_ids)
    rep["op"] = "delete_component"
    return V2, F2, rep


def cut_handle(V, F, loop_ids, **_):
    """
    Reduce genus by cutting a tunnel loop and capping the openings.

    Pragmatic + robust: delete the band of faces touching the loop (opens the
    tunnel into two boundaries), then cap every resulting boundary loop. For a
    genus-g handle this drops the genus by one.
    """
    g_before = analyze_mesh(V, F)["topology"]["estimated_genus"]
    loopset = set(int(i) for i in loop_ids)
    F_arr = np.asarray(F, dtype=np.int64)
    band = [i for i, f in enumerate(F_arr) if loopset & set(int(x) for x in f)]
    if not band:
        return V, F, {"op": "cut_handle", "changed": False,
                      "note": "loop not on the surface"}
    V2, F2 = _delete_faces(V, F, band)
    # Cap all newly-opened boundary loops.
    for _ in range(8):
        loops = boundary_loops(V2, F2)
        if not loops:
            break
        V2, F2 = _fill_one_loop(V2, F2, loops[0])
    m = _mesh(V2, F2)
    trimesh.repair.fix_normals(m)
    V2, F2 = _out(m)
    g_after = analyze_mesh(V2, F2)["topology"]["estimated_genus"]
    return V2, F2, {"op": "cut_handle", "genus_before": g_before,
                    "genus_after": g_after}


EDIT_OPS = {
    "set_vertex_positions": set_vertex_positions,
    "inflate": inflate,
    "local_smooth": local_smooth,
    "fill_loop": fill_loop,
    "delete_faces": delete_faces,
    "delete_component": delete_component,
    "cut_handle": cut_handle,
}


def apply(op: str, V, F, params: dict | None = None) -> dict:
    """Dispatch an edit op; return {V, F, report} with before/after diagnostics."""
    if op not in EDIT_OPS:
        raise KeyError(f"Unknown edit op '{op}'")
    before = analyze_mesh(V, F)["topology"]
    V2, F2, report = EDIT_OPS[op](V, F, **(params or {}))
    after = analyze_mesh(V2, F2)["topology"]
    report["topology_before"] = before
    report["topology_after"] = after
    return {"V": V2, "F": F2, "report": report}
