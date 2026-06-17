"""
Main AmiGo pipeline.

Given a closed triangle mesh, a seed vertex, and a stitch width, produces
human-readable crochet instructions (a pattern).

Algorithm outline (Algorithm 1 in the paper):
  1. Load and normalise mesh.
  2. Compute f = geodesic distance from seed (with automatic t-smoothing).
  3. Detect saddle points → determine segment decomposition.
  4. For each segment (in topological order):
     a. Find the seam (geodesic cut from seed/boundary to local tip).
     b. Trace isolines of f, sample at equal arc-length intervals → X_G.
     c. Build DTW coupling → column edges C.
     d. Run transducer → raw stitch sequence.
  5. Apply loop folding → human-readable pattern.

Two entry points share the same core:
  * ``amigo_pipeline``      — path in, pattern string out (used by the CLI).
  * ``amigo_pipeline_data`` — in-memory (V, F) in, structured result out,
                              exposing every intermediate (geodesic field,
                              saddles, per-segment 3-D stitch rows, pattern).
                              Used by the web UI.
"""

from __future__ import annotations

import numpy as np

from .mesh_ops import load_mesh, normalize_to_unit_area
from .geodesics import (
    auto_smooth_geodesic,
    find_maximum,
    find_saddle_points,
    geodesic_path,
)
from .sampling import sample_crochet_graph
from .connectivity import build_connectivity
from .instructions import generate_all_instructions
from .loop_folding import format_pattern
from .export_crochetparade import to_crochetparade
from .segmentation import (
    segment_by_saddles,
    segment_meshes,
    find_segment_boundaries,
    topological_sort_segments,
)


def _stitch_types_per_vertex(rows_pos: list, all_instr: list) -> list[list[str]]:
    """
    Map every crochet-graph vertex (point in ``rows_pos``) to the stitch type
    that produces it, by replaying the transducer's fan-out.

    ``all_instr[i]`` is the stitch sequence that *creates* row ``i`` (the
    transition from row ``i-1``). For a single transition the stitch fan-out
    sums exactly to ``len(rows_pos[i])``:
        sc -> 1 vertex,  inc(x) -> x vertices,  dec(x) -> 1 vertex.

    Returns one list of type strings per row, aligned point-for-point.
    """
    types: list[list[str]] = []
    for i, row in enumerate(rows_pos):
        n = len(row)
        if i == 0:
            # Row 0 is the seed / magic-ring centre.
            types.append(["magic"] * n)
            continue
        row_types: list[str] = []
        for st in all_instr[i]:
            if st.type == "inc":
                row_types.extend(["inc"] * st.x)
            elif st.type == "dec":
                row_types.append("dec")
            elif st.type == "magic":
                row_types.extend(["magic"] * st.x)
            else:  # sc
                row_types.append("sc")
        # Guard against any off-by-one from degenerate rows.
        if len(row_types) < n:
            row_types.extend(["sc"] * (n - len(row_types)))
        types.append(row_types[:n])
    return types


def amigo_pipeline_data(
    V: np.ndarray,
    F: np.ndarray,
    seed_idx: int = 0,
    stitch_width: float = 0.05,
    verbose: bool = True,
) -> dict:
    """
    Run the full AmiGo pipeline on already-loaded, already-normalised geometry
    and return all intermediates needed for visualisation.

    Parameters
    ----------
    V, F          : normalised mesh geometry (V from ``normalize_to_unit_area``)
    seed_idx      : vertex index to start crocheting from
    stitch_width  : stitch size in normalised units (~0.05 ≈ 20 rows)
    verbose       : print progress messages

    Returns
    -------
    dict with keys:
        "field"      : (n_verts,) geodesic distance from seed
        "field_max"  : float, max field value
        "seed"       : int, seed vertex index
        "tip"        : int, vertex index of the global maximum
        "saddles"    : list[int] saddle vertex indices
        "segments"   : list of {"rows": list[list[[x,y,z]]], "is_first": bool}
                       where each row is a list of 3-D stitch positions
        "instructions": flat list of stitch instructions
        "pattern"    : human-readable pattern string
        "crochetparade": crochetparade (.cp) string
    """
    # ------------------------------------------------------------------
    # 2. Geodesic distance f
    # ------------------------------------------------------------------
    if verbose:
        print("Computing geodesic distances …")
    f, t_scale = auto_smooth_geodesic(V, F, seed_idx)
    if verbose:
        print(f"  t_scale = {t_scale}")

    f_max_idx = find_maximum(f)
    saddles = find_saddle_points(V, F, f)
    if verbose:
        print(f"  max-f vertex = {f_max_idx},  saddle count = {len(saddles)}")

    # ------------------------------------------------------------------
    # 3. Segment decomposition
    # ------------------------------------------------------------------
    segments = segment_by_saddles(V, F, f, saddles)
    seg_data = segment_meshes(V, F, segments)
    boundaries = find_segment_boundaries(seg_data, V, F, f)
    order = topological_sort_segments(seg_data)
    if verbose:
        print(f"  {len(seg_data)} segment(s)")

    # ------------------------------------------------------------------
    # 4. Per-segment processing
    # ------------------------------------------------------------------
    all_instructions = []
    viz_segments = []

    for seg_idx in order:
        seg = seg_data[seg_idx]
        V_s = seg["V"]
        F_s = seg["F"]
        lv  = seg["local_verts"]
        g2l = seg["global_to_local"]
        f_s = f[lv]

        # Local seed index
        if seg["is_first"]:
            local_seed = g2l.get(seed_idx, int(np.argmin(f_s)))
        else:
            prev_bnd = boundaries[seg_idx - 1] if seg_idx > 0 else []
            cands = [g2l[v] for v in prev_bnd if v in g2l]
            local_seed = cands[0] if cands else int(np.argmin(f_s))

        local_max = int(np.argmax(f_s))

        # Seam: geodesic path from local seed to local tip
        if verbose:
            print(f"  Segment {seg_idx}: tracing seam …")
        try:
            seam_verts = geodesic_path(V_s, F_s, local_seed, local_max)
        except Exception:
            seam_verts = [local_seed, local_max]

        if verbose:
            print(f"  Segment {seg_idx}: sampling isoline grid …")

        rows_pos, row_f = sample_crochet_graph(
            V_s, F_s, f_s,
            stitch_width=stitch_width,
            f_max_idx=local_max,
            seed_idx=local_seed,
            seam_vertices=seam_verts,
        )

        if len(rows_pos) < 2:
            if verbose:
                print(f"  Segment {seg_idx}: too thin, skipping.")
            continue

        if verbose:
            counts = [len(r) for r in rows_pos]
            print(f"  Segment {seg_idx}: {len(rows_pos)} rows, "
                  f"stitches/row: {min(counts)}–{max(counts)}")

        # Connectivity (DTW)
        _, col_edges = build_connectivity(rows_pos)

        # Stitch instructions
        seg_instr = generate_all_instructions(rows_pos, col_edges)

        # Capture the crochet graph for visualisation: 3-D stitch rows,
        # per-vertex stitch types, and the column-edge coupling between
        # consecutive rows (as [j, k] index pairs).
        viz_segments.append({
            "is_first": bool(seg["is_first"]),
            "rows": [np.asarray(r, dtype=float).tolist() for r in rows_pos],
            "types": _stitch_types_per_vertex(rows_pos, seg_instr),
            "col_edges": [[[int(j), int(k)] for (j, k) in edges]
                          for edges in col_edges],
        })

        # For non-first segments skip the duplicated magic-circle row
        if not seg["is_first"] and seg_instr:
            seg_instr = seg_instr[1:]

        all_instructions.extend(seg_instr)

    if not all_instructions:
        pattern = "(no instructions generated — mesh may be too small or degenerate)"
        cp = ""
    else:
        pattern = format_pattern(all_instructions)
        cp = to_crochetparade(all_instructions)

    return {
        "field": f.astype(float).tolist(),
        "field_max": float(f[f_max_idx]),
        "seed": int(seed_idx),
        "tip": int(f_max_idx),
        "saddles": [int(s) for s in saddles],
        "segments": viz_segments,
        "instructions": all_instructions,
        "pattern": pattern,
        "crochetparade": cp,
    }


def amigo_pipeline(
    mesh_path: str,
    seed_idx: int = 0,
    stitch_width: float = 0.05,
    eps: float = 1e-3,          # kept for API compatibility, unused
    verbose: bool = True,
    output_format: str = "text",   # "text" | "crochetparade"
) -> str:
    """
    Run the full AmiGo pipeline from a mesh file and return a pattern string.

    Parameters
    ----------
    mesh_path     : path to a closed triangle mesh (.obj, .ply, etc.)
    seed_idx      : vertex index to start crocheting from
    stitch_width  : stitch size in the same units as the normalised mesh
                    (mesh is normalised to unit surface area, so 0.05 ≈ 20 rows)
    verbose       : print progress messages

    Returns
    -------
    pattern : multi-line string with the complete crochet pattern
    """
    # ------------------------------------------------------------------
    # 1. Load and normalise
    # ------------------------------------------------------------------
    V, F = load_mesh(mesh_path)
    V, scale = normalize_to_unit_area(V, F)
    if verbose:
        print(f"Mesh: {len(V)} vertices, {len(F)} faces  (scale={scale:.4f})")

    data = amigo_pipeline_data(
        V, F,
        seed_idx=seed_idx,
        stitch_width=stitch_width,
        verbose=verbose,
    )

    if output_format == "crochetparade":
        return data["crochetparade"]
    return data["pattern"]
