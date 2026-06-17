"""
Mesh diagnostics for crochetability assessment.

``analyze_mesh(V, F)`` computes the signals the Phase-2 agent reasons over to
decide whether a mesh is crochetable by the AmiGo technique:

  * Topology — is it a closed, orientable, single-component 2-manifold? What
    genus? Any holes / non-manifold edges?
  * Geometry — size, aspect ratio, degenerate faces, edge-length spread.
  * AmiGo-front — run the actual pipeline front (geodesic field → saddles →
    segments → stitch rows) and report how it behaves. This is the real
    "is the technique applicable" signal; a failure here is itself reported.

Everything here is deterministic (no LLM) and reuses the existing pipeline
modules rather than reimplementing them.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .mesh_ops import face_areas, normalize_to_unit_area
from .pipeline import amigo_pipeline_data


def _principal_pole_seed(V: np.ndarray) -> int:
    """Heuristic seed: the vertex farthest from the centroid (a natural tip)."""
    c = V.mean(axis=0)
    return int(np.argmax(np.linalg.norm(V - c, axis=1)))


def _topology(V: np.ndarray, F: np.ndarray) -> dict:
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    # Open (boundary) edges appear in exactly one face.
    open_edges = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    try:
        n_components = len(mesh.split(only_watertight=False))
    except Exception:
        n_components = 1
    euler = int(mesh.euler_number)
    watertight = bool(mesh.is_watertight)
    genus = (2 - euler) // 2 if watertight else None
    return {
        "is_watertight": watertight,
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": euler,
        "estimated_genus": genus,
        "n_components": int(n_components),
        "n_open_edges": int(len(open_edges)),
        "has_holes": bool(len(open_edges) > 0),
    }


def _geometry(V: np.ndarray, F: np.ndarray) -> dict:
    areas = face_areas(V, F)
    extent = V.max(axis=0) - V.min(axis=0)
    longest = float(extent.max())
    shortest = float(extent[extent > 0].min()) if np.any(extent > 0) else 0.0
    # Edge lengths
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    el = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    return {
        "n_vertices": int(len(V)),
        "n_faces": int(len(F)),
        "surface_area": float(areas.sum()),
        "bbox_extent": [float(x) for x in extent],
        "aspect_ratio": float(longest / shortest) if shortest > 0 else float("inf"),
        "n_degenerate_faces": int(np.count_nonzero(areas < 1e-12)),
        "edge_length_min": float(el.min()),
        "edge_length_median": float(np.median(el)),
        "edge_length_max": float(el.max()),
    }


def _amigo_front(V: np.ndarray, F: np.ndarray, seed: int, stitch_width: float) -> dict:
    """Run the real pipeline front and summarise its behaviour (or its failure)."""
    Vn, _ = normalize_to_unit_area(V, F)
    try:
        data = amigo_pipeline_data(Vn, F, seed_idx=seed,
                                   stitch_width=stitch_width, verbose=False)
    except Exception as exc:  # noqa: BLE001
        return {"ran": False, "error": f"{type(exc).__name__}: {exc}"}

    segs = data["segments"]
    seg_summ = []
    for s in segs:
        counts = [len(r) for r in s["rows"]]
        # The magic-ring seed row and the tip row are single points by
        # construction (the poles); judge thinness on the interior rows.
        interior = counts[1:-1] if len(counts) > 2 else counts
        seg_summ.append({
            "rows": len(s["rows"]),
            "min_stitches_per_row": min(interior) if interior else 0,
            "max_stitches_per_row": max(counts) if counts else 0,
        })
    thin = [s for s in seg_summ if s["rows"] < 2 or s["min_stitches_per_row"] < 3]
    return {
        "ran": True,
        "seed": int(seed),
        "stitch_width": stitch_width,
        "n_saddles": len(data["saddles"]),
        "n_segments": len(segs),
        "n_thin_segments": len(thin),
        "segments": seg_summ,
        "pattern_lines": data["pattern"].count("\n") + 1 if segs else 0,
    }


def analyze_mesh(V: np.ndarray, F: np.ndarray,
                 seed: int | None = None,
                 stitch_width: float = 0.05) -> dict:
    """
    Compute crochetability diagnostics for a mesh.

    Parameters
    ----------
    V, F          : mesh geometry (any scale; the AmiGo-front normalises internally)
    seed          : seed vertex for the geodesic field; default = farthest-from-centroid
    stitch_width  : stitch size used for the AmiGo-front probe

    Returns
    -------
    dict with "topology", "geometry", and "amigo_front" sub-dicts.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    if seed is None:
        seed = _principal_pole_seed(V)
    return {
        "topology": _topology(V, F),
        "geometry": _geometry(V, F),
        "amigo_front": _amigo_front(V, F, seed, stitch_width),
    }
