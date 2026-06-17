"""
Deterministic mesh-simplification transforms for crochetability.

Each transform takes ``(V, F, **params)`` and returns ``(V2, F2, report)`` where
``report`` is a short dict describing what changed. These are the levers the
Phase-2 agent proposes (and the user approves) to make a mesh easier to crochet:

  * keep_largest_component / remove_small_components — fix multi-component meshes
  * fill_holes                                       — close boundaries (open mesh)
  * smooth                                           — shed high-frequency detail
                                                       that spawns spurious saddles
  * decimate                                         — reduce face count for
                                                       robustness / speed

All are pure NumPy/trimesh and deterministic.
"""

from __future__ import annotations

import numpy as np
import trimesh


def _mesh(V, F):
    return trimesh.Trimesh(vertices=np.asarray(V, dtype=np.float64),
                           faces=np.asarray(F, dtype=np.int64), process=False)


def _out(mesh):
    return (np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64))


def keep_largest_component(V, F, **_):
    """Discard all but the largest connected component (by face count)."""
    m = _mesh(V, F)
    parts = m.split(only_watertight=False)
    if len(parts) <= 1:
        return V, F, {"technique": "keep_largest_component", "changed": False,
                      "n_components": len(parts)}
    largest = max(parts, key=lambda p: len(p.faces))
    V2, F2 = _out(largest)
    return V2, F2, {
        "technique": "keep_largest_component", "changed": True,
        "n_components_before": len(parts),
        "faces_before": int(len(F)), "faces_after": int(len(F2)),
    }


def remove_small_components(V, F, min_face_fraction: float = 0.05, **_):
    """Drop components smaller than ``min_face_fraction`` of the total faces."""
    m = _mesh(V, F)
    parts = m.split(only_watertight=False)
    if len(parts) <= 1:
        return V, F, {"technique": "remove_small_components", "changed": False,
                      "n_components": len(parts)}
    total = sum(len(p.faces) for p in parts)
    keep = [p for p in parts if len(p.faces) >= min_face_fraction * total]
    if not keep:
        keep = [max(parts, key=lambda p: len(p.faces))]
    merged = trimesh.util.concatenate(keep)
    V2, F2 = _out(merged)
    return V2, F2, {
        "technique": "remove_small_components", "changed": True,
        "n_components_before": len(parts), "n_components_after": len(keep),
        "faces_before": int(len(F)), "faces_after": int(len(F2)),
    }


def fill_holes(V, F, **_):
    """Close boundary loops so the mesh becomes watertight where possible."""
    m = _mesh(V, F)
    before = bool(m.is_watertight)
    m.fill_holes()
    m.remove_unreferenced_vertices()
    V2, F2 = _out(m)
    return V2, F2, {
        "technique": "fill_holes", "changed": not before,
        "watertight_before": before, "watertight_after": bool(m.is_watertight),
        "faces_before": int(len(F)), "faces_after": int(len(F2)),
    }


def smooth(V, F, iterations: int = 10, **_):
    """Taubin smoothing — reduces high-frequency detail without shrinking."""
    m = _mesh(V, F)
    trimesh.smoothing.filter_taubin(m, iterations=int(iterations))
    V2, F2 = _out(m)
    return V2, F2, {
        "technique": "smooth", "changed": True, "iterations": int(iterations),
    }


def decimate(V, F, target_faces: int | None = None, target_fraction: float = 0.5, **_):
    """
    Reduce face count via quadric decimation.

    Needs ``fast-simplification`` (or open3d) behind trimesh's
    ``simplify_quadric_decimation``. Falls back to Taubin smoothing if no
    decimation backend is installed — reported in the result.
    """
    m = _mesh(V, F)
    if target_faces is None:
        target_faces = max(int(len(F) * target_fraction), 50)
    try:
        d = m.simplify_quadric_decimation(face_count=int(target_faces))
        V2, F2 = _out(d)
        return V2, F2, {
            "technique": "decimate", "changed": True, "backend": "quadric",
            "faces_before": int(len(F)), "faces_after": int(len(F2)),
            "target_faces": int(target_faces),
        }
    except Exception as exc:  # noqa: BLE001 — backend missing/failed
        trimesh.smoothing.filter_taubin(m, iterations=8)
        V2, F2 = _out(m)
        return V2, F2, {
            "technique": "decimate", "changed": True, "backend": "fallback_smooth",
            "note": f"quadric decimation unavailable ({type(exc).__name__}); "
                    f"applied Taubin smoothing instead",
            "faces_before": int(len(F)), "faces_after": int(len(F2)),
        }


# Registry the agent's apply_simplification tool dispatches on.
TRANSFORMS = {
    "keep_largest_component": keep_largest_component,
    "remove_small_components": remove_small_components,
    "fill_holes": fill_holes,
    "smooth": smooth,
    "decimate": decimate,
}


def apply(technique: str, V, F, params: dict | None = None):
    """Dispatch to a named transform. Raises KeyError for unknown techniques."""
    fn = TRANSFORMS[technique]
    return fn(V, F, **(params or {}))
