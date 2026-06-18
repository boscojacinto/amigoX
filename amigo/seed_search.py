"""
Seed-candidate generation for the closed-loop optimizer.

A good crochet seed is usually a *pole* of the shape — a natural tip or limb end
— so the geodesic field flows cleanly over the surface. We generate a diverse,
well-separated set of candidates by geodesic farthest-point sampling (FPS),
seeded from the principal pole, plus a couple of bounding-box extrema for
variety. Each candidate carries a short human descriptor for the LLM agent.

``rank_candidates`` scores each candidate with ``quality.evaluate_seed`` and
sorts them — a deterministic fallback used by the CLI when no API key is present.
"""

from __future__ import annotations

import numpy as np

from .diagnostics import _principal_pole_seed
from .geodesics import compute_geodesic_distance
from .quality import evaluate_seed


def seed_candidates(V, F, k: int = 6) -> list[dict]:
    """
    Return up to ``k`` diverse candidate seeds, each as
    ``{"seed": int, "descriptor": str}``.

    Strategy: principal pole, then geodesic farthest-point sampling (each new
    seed is the vertex farthest — geodesically — from all chosen seeds so far),
    then bounding-box axis extrema to fill out variety.
    """
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=np.int64)
    n = len(V)
    k = max(1, min(k, n))

    chosen: list[int] = []
    descriptors: dict[int, str] = {}

    pole = _principal_pole_seed(V)
    chosen.append(pole)
    descriptors[pole] = "principal pole (farthest vertex from centroid)"

    # Geodesic farthest-point sampling: maintain the min geodesic distance to the
    # set of chosen seeds and repeatedly pick its argmax.
    min_dist = None
    while len(chosen) < k:
        try:
            d = np.asarray(compute_geodesic_distance(V, F, chosen[-1]), dtype=float)
        except Exception:  # noqa: BLE001 — heat solver can fail on bad meshes
            break
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        min_dist = d if min_dist is None else np.minimum(min_dist, d)
        nxt = int(np.argmax(min_dist))
        if nxt in descriptors or min_dist[nxt] <= 0:
            break
        chosen.append(nxt)
        descriptors[nxt] = f"geodesic farthest point #{len(chosen) - 1}"

    # Fill remaining slots with bounding-box axis extrema (cheap, geometry-based).
    if len(chosen) < k:
        for axis, lbl in ((0, "min-x"), (0, "max-x"), (1, "min-y"), (1, "max-y"),
                          (2, "min-z"), (2, "max-z")):
            if len(chosen) >= k:
                break
            idx = int(np.argmin(V[:, axis])) if lbl.startswith("min") \
                else int(np.argmax(V[:, axis]))
            if idx not in descriptors:
                chosen.append(idx)
                descriptors[idx] = f"bounding-box {lbl} extremum"

    return [{"seed": int(s), "descriptor": descriptors[s]} for s in chosen]


def rank_candidates(V, F, candidates: list[int] | None = None,
                    stitch_width: float = 0.05, k: int = 6) -> list[dict]:
    """
    Score each candidate seed with the deterministic quality metric and return
    them sorted best-first. Each entry is the full ``evaluate_seed`` dict plus
    the candidate's ``descriptor`` (when available).

    If ``candidates`` is None, generate them with ``seed_candidates``.
    """
    if candidates is None:
        cands = seed_candidates(V, F, k=k)
    else:
        cands = [{"seed": int(s), "descriptor": ""} for s in candidates]

    ranked = []
    for c in cands:
        m = evaluate_seed(V, F, c["seed"], stitch_width)
        m["descriptor"] = c.get("descriptor", "")
        ranked.append(m)
    ranked.sort(key=lambda m: m.get("score", 0.0), reverse=True)
    return ranked
