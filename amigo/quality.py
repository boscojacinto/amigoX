"""
Pattern-quality evaluation for the closed-loop seed optimizer.

Given a generated crochet graph (the output of ``amigo_pipeline_data``) and the
mesh it was generated from, score how well the stitches *represent the surface*:

  * coverage      — fraction of mesh surface area within reach of a stitch
                    (the complement is "uncovered" surface no stitch represents).
  * floating      — column/row edges whose Euclidean length is far larger than a
                    stitch (edges that span empty space across a concave or
                    branching gap instead of hugging the surface).
  * thin segments — segments too small to crochet (reuses the diagnostics rule).

These feed a single scalar ``score`` in [0, 1] that the seed optimizer maximizes.
Pure numpy + scipy.cKDTree; no LLM, no new dependencies.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .mesh_ops import face_areas, normalize_to_unit_area
from .pipeline import amigo_pipeline_data

# How far from a stitch a face centroid may sit and still count as covered.
# Rows sit ~stitch_width apart geodesically, so 1.5x gives tolerance on curvature.
COVER_RADIUS_FACTOR = 1.5
# An edge longer than this many stitch-widths (or median edge lengths) is "floating".
FLOAT_FACTOR = 2.5
# Cap how many culprit ids/positions we return (keeps web payloads small).
_MAX_REPORT = 300


def _stitch_points(data: dict) -> np.ndarray:
    """All crochet-graph vertex positions, flattened across every segment/row."""
    pts: list = []
    for seg in data["segments"]:
        for row in seg["rows"]:
            pts.extend(row)
    if not pts:
        return np.empty((0, 3), dtype=float)
    return np.asarray(pts, dtype=float)


def _edge_lengths(data: dict):
    """Yield (length, midpoint) for every row edge and column edge in the graph."""
    for seg in data["segments"]:
        rows = [np.asarray(r, dtype=float) for r in seg["rows"]]
        # Row edges: consecutive stitches within a row (rows are open polylines
        # here; the closing edge is implicit and not part of col/row connectivity).
        for row in rows:
            for a in range(len(row) - 1):
                p, q = row[a], row[a + 1]
                yield float(np.linalg.norm(p - q)), 0.5 * (p + q)
        # Column edges: DTW couplings between consecutive rows.
        for i, edges in enumerate(seg["col_edges"]):
            for (j, k) in edges:
                p, q = rows[i][j], rows[i + 1][k]
                yield float(np.linalg.norm(p - q)), 0.5 * (p + q)


def _thin_segment_count(data: dict) -> int:
    """
    Count segments too thin to crochet.

    Like diagnostics._amigo_front this ignores the single-point pole rows, but it
    is more forgiving of the natural 1-2 stitch ramp rows right after a magic
    ring: a segment is thin only when it has <2 rows, or when *several* interior
    rows (more than a quarter, and more than one) fall below 3 stitches — the
    signature of a genuine thin spike/sheet rather than a clean cap.
    """
    thin = 0
    for seg in data["segments"]:
        counts = [len(r) for r in seg["rows"]]
        if len(counts) < 2:
            thin += 1
            continue
        interior = counts[1:-1] if len(counts) > 2 else counts
        if not interior:
            thin += 1
            continue
        n_below = sum(1 for c in interior if c < 3)
        if n_below > max(1, 0.25 * len(interior)):
            thin += 1
    return thin


def evaluate_pattern(V, F, data: dict, stitch_width: float, *,
                     cover_radius_factor: float = COVER_RADIUS_FACTOR,
                     float_factor: float = FLOAT_FACTOR) -> dict:
    """
    Score a generated pattern against the mesh it came from.

    Parameters
    ----------
    V, F          : the *normalized* mesh (same arrays passed to the pipeline).
    data          : the dict returned by ``amigo_pipeline_data``.
    stitch_width  : the stitch width used to generate ``data``.

    Returns a dict with the scalar ``score`` plus the sub-metrics and the
    culprit ids/positions (capped) for highlighting.
    """
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=np.int64)
    pts = _stitch_points(data)

    n_segments = len(data["segments"])
    n_saddles = len(data.get("saddles", []))
    n_thin = _thin_segment_count(data)
    thin_fraction = n_thin / n_segments if n_segments else 1.0

    # ---- Coverage: which faces are within reach of a stitch? ------------------
    areas = face_areas(V, F)
    total_area = float(areas.sum())
    if len(pts) == 0 or total_area <= 0:
        return {
            "ran": True, "score": 0.0, "coverage": 0.0,
            "uncovered_area_fraction": 1.0, "n_uncovered_faces": int(len(F)),
            "uncovered_face_ids": list(range(min(len(F), _MAX_REPORT))),
            "uncovered_centroids": [],
            "n_floating": 0, "floating_fraction": 0.0, "floating_edges": [],
            "n_segments": n_segments, "n_thin_segments": n_thin,
            "n_saddles": n_saddles, "median_edge_length": 0.0,
            "n_stitches": int(len(pts)),
        }

    centroids = V[F].mean(axis=1)
    tree = cKDTree(pts)
    dist, _ = tree.query(centroids)
    cover_radius = cover_radius_factor * stitch_width
    covered = dist <= cover_radius
    coverage = float(areas[covered].sum() / total_area)
    uncovered_ids = np.nonzero(~covered)[0]
    # Report the worst (farthest-from-any-stitch) uncovered faces first.
    order = uncovered_ids[np.argsort(-dist[uncovered_ids])][:_MAX_REPORT]

    # ---- Floating edges: far longer than a stitch -----------------------------
    lengths, mids = [], []
    for L, mid in _edge_lengths(data):
        lengths.append(L)
        mids.append(mid)
    lengths = np.asarray(lengths, dtype=float)
    mids = np.asarray(mids, dtype=float) if len(mids) else np.empty((0, 3))
    median_len = float(np.median(lengths)) if len(lengths) else 0.0
    threshold = float_factor * max(stitch_width, median_len)
    float_mask = lengths > threshold
    n_floating = int(float_mask.sum())
    total_edges = int(len(lengths))
    floating_fraction = n_floating / total_edges if total_edges else 0.0
    float_idx = np.nonzero(float_mask)[0]
    float_idx = float_idx[np.argsort(-lengths[float_idx])][:_MAX_REPORT]
    floating_edges = [{"midpoint": mids[i].tolist(), "length": float(lengths[i])}
                      for i in float_idx]

    # ---- Scalar score ---------------------------------------------------------
    score = coverage * (1.0 - 0.5 * floating_fraction) - 0.1 * thin_fraction
    score = float(max(0.0, min(1.0, score)))

    return {
        "ran": True,
        "score": score,
        "coverage": coverage,
        "uncovered_area_fraction": float(1.0 - coverage),
        "n_uncovered_faces": int(uncovered_ids.size),
        "uncovered_face_ids": [int(i) for i in order],
        "uncovered_centroids": centroids[order].tolist(),
        "n_floating": n_floating,
        "floating_fraction": float(floating_fraction),
        "floating_edges": floating_edges,
        "n_segments": n_segments,
        "n_thin_segments": n_thin,
        "n_saddles": n_saddles,
        "median_edge_length": median_len,
        "n_stitches": int(len(pts)),
    }


def evaluate_seed(V, F, seed_idx: int, stitch_width: float = 0.05) -> dict:
    """
    Normalize, run the pipeline at ``seed_idx``, and score the result.

    Robust to pipeline failure: returns ``{"ran": False, "score": 0.0, ...}``.
    The mesh is normalized internally so coverage compares like-with-like.
    """
    Vn, _ = normalize_to_unit_area(np.asarray(V, dtype=float),
                                   np.asarray(F, dtype=np.int64))
    try:
        data = amigo_pipeline_data(Vn, F, seed_idx=int(seed_idx),
                                   stitch_width=stitch_width, verbose=False)
    except Exception as exc:  # noqa: BLE001
        return {"ran": False, "score": 0.0, "seed": int(seed_idx),
                "stitch_width": stitch_width,
                "error": f"{type(exc).__name__}: {exc}"}
    metrics = evaluate_pattern(Vn, F, data, stitch_width)
    metrics["seed"] = int(seed_idx)
    metrics["stitch_width"] = stitch_width
    return metrics


def summarize_metrics(m: dict) -> dict:
    """Compact, array-free view of evaluate_seed output for the LLM agent."""
    if not m.get("ran", False):
        return {"ran": False, "seed": m.get("seed"),
                "error": m.get("error", "pipeline failed")}
    return {
        "ran": True,
        "seed": m.get("seed"),
        "stitch_width": m.get("stitch_width"),
        "score": round(m["score"], 4),
        "coverage": round(m["coverage"], 4),
        "uncovered_area_fraction": round(m["uncovered_area_fraction"], 4),
        "n_uncovered_faces": m["n_uncovered_faces"],
        "n_floating": m["n_floating"],
        "floating_fraction": round(m["floating_fraction"], 4),
        "n_segments": m["n_segments"],
        "n_thin_segments": m["n_thin_segments"],
        "n_saddles": m["n_saddles"],
    }
