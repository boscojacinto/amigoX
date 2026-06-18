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

from .mesh_ops import load_mesh, normalize_to_unit_area, weld_vertices
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
    build_segment_tree,
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
            # Row 0 is the magic-ring centre (root) or the inherited join loop.
            kind = (all_instr[0][0].type
                    if all_instr and all_instr[0] else "magic")
            types.append([kind if kind in ("magic", "join") else "magic"] * n)
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


def _align_loop(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Roll/reverse loop ``src`` to best match ``ref`` (same loop, different
    sampling) so the join coupling between them stays local."""
    src = np.asarray(src, dtype=float)
    if len(src) < 3 or len(ref) < 1:
        return src

    def aligned(a):
        k = int(np.argmin(np.linalg.norm(a - ref[0], axis=1)))
        return np.roll(a, -k, axis=0)

    def cost(a):
        m = min(len(a), len(ref))
        return float(np.linalg.norm(a[:m] - ref[:m], axis=1).sum())

    fwd = aligned(src)
    rev = aligned(src[::-1].copy())
    return fwd if cost(fwd) <= cost(rev) else rev


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
    # 1b. Weld coincident vertices. Split seams (duplicated verts) disconnect
    #     the surface, so the geodesic can't cross them and whole patches get a
    #     constant f → no isolines → uncovered regions. We compute everything on
    #     the welded mesh and map the field/indices back to the caller's original
    #     vertex indexing on the way out (the web UI keys off the original mesh).
    # ------------------------------------------------------------------
    V_orig_n = len(V)
    Vw, Fw, old2new = weld_vertices(V, F)
    if verbose and len(Vw) < V_orig_n:
        print(f"  welded {V_orig_n} → {len(Vw)} vertices "
              f"({V_orig_n - len(Vw)} coincident)")
    # A representative original vertex for each welded vertex (for output remap).
    new2old = np.zeros(len(Vw), dtype=int)
    new2old[old2new] = np.arange(V_orig_n)
    seed_orig = int(seed_idx)
    V, F = Vw, Fw
    seed_idx = int(old2new[seed_orig])

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
    # 3. Segment decomposition + parent/child tree (join-as-you-go)
    # ------------------------------------------------------------------
    segments = segment_by_saddles(V, F, f, saddles, stitch_width=stitch_width)
    seg_data = segment_meshes(V, F, segments)
    parent, children, order = build_segment_tree(seg_data, seed_idx)
    if verbose:
        print(f"  {len(seg_data)} segment(s)")

    # ------------------------------------------------------------------
    # 4a. Sample each segment's isoline rows (parents before children). A root
    #     starts at a magic ring; an internal segment ends at its top boundary
    #     loop; a leaf closes to a tip point.
    # ------------------------------------------------------------------
    seg_rows: dict[int, list] = {}
    for seg_idx in order:
        seg = seg_data[seg_idx]
        V_s, F_s = seg["V"], seg["F"]
        lv, g2l = seg["local_verts"], seg["global_to_local"]
        f_s = f[lv]
        is_root = parent[seg_idx] < 0
        is_leaf = len(children[seg_idx]) == 0

        if is_root:
            local_seed = g2l.get(seed_idx, int(np.argmin(f_s)))
        else:
            pv = set(int(v) for v in seg_data[parent[seg_idx]]["local_verts"])
            shared = [g2l[int(v)] for v in lv if int(v) in pv]
            local_seed = (min(shared, key=lambda l: f_s[l])
                          if shared else int(np.argmin(f_s)))
        local_max = int(np.argmax(f_s))

        try:
            seam_verts = geodesic_path(V_s, F_s, local_seed, local_max)
        except Exception:
            seam_verts = [local_seed, local_max]

        rows_pos, _ = sample_crochet_graph(
            V_s, F_s, f_s, stitch_width=stitch_width,
            f_max_idx=local_max, seed_idx=local_seed, seam_vertices=seam_verts,
            cap_seed=is_root, cap_tip=is_leaf,
        )
        if len(rows_pos) >= 2:
            seg_rows[seg_idx] = [np.asarray(r, dtype=float) for r in rows_pos]

    # ------------------------------------------------------------------
    # 4b. Join — prepend each child's portion of its parent's top boundary
    #     loop (partitioned among siblings by nearest first row), so the child
    #     is worked into the existing boundary, not a fresh magic ring.
    # ------------------------------------------------------------------
    for p in order:
        if p not in seg_rows:
            continue
        kids = [c for c in children[p] if c in seg_rows]
        if not kids:
            continue
        boundary = seg_rows[p][-1]
        if len(boundary) < 2:
            continue
        kid_firsts = [seg_rows[c][0] for c in kids]
        assign = np.empty(len(boundary), dtype=int)
        for bi, bp in enumerate(boundary):
            assign[bi] = int(np.argmin(
                [float(np.linalg.norm(kf - bp, axis=1).min()) for kf in kid_firsts]))
        for ki, c in enumerate(kids):
            idxs = np.where(assign == ki)[0]
            if len(idxs) >= 2:
                join_row = _align_loop(boundary[idxs], seg_rows[c][0])
                seg_rows[c] = [join_row] + seg_rows[c]

    # ------------------------------------------------------------------
    # 4c. Connectivity, instructions and viz — in crochet (DFS) order.
    # ------------------------------------------------------------------
    all_instructions = []
    viz_segments = []
    for seg_idx in order:
        rows_pos = seg_rows.get(seg_idx)
        if not rows_pos or len(rows_pos) < 2:
            continue
        is_root = parent[seg_idx] < 0
        _, col_edges = build_connectivity(rows_pos)
        seg_instr = generate_all_instructions(rows_pos, col_edges,
                                              magic_start=is_root)
        if verbose:
            counts = [len(r) for r in rows_pos]
            print(f"  Segment {seg_idx}: {len(rows_pos)} rows, "
                  f"stitches/row: {min(counts)}–{max(counts)}"
                  f"{'' if is_root else '  (joined)'}")

        viz_segments.append({
            "is_first": bool(is_root),
            "rows": [r.tolist() for r in rows_pos],
            "types": _stitch_types_per_vertex(rows_pos, seg_instr),
            "col_edges": [[[int(j), int(k)] for (j, k) in edges]
                          for edges in col_edges],
        })
        all_instructions.extend(seg_instr)

    if not all_instructions:
        pattern = "(no instructions generated — mesh may be too small or degenerate)"
        cp = ""
    else:
        pattern = format_pattern(all_instructions)
        cp = to_crochetparade(all_instructions)

    # Map field (per welded vertex) and indices back to the caller's original
    # vertex space so the web UI can colour the un-welded mesh it holds.
    field_orig = f[old2new]
    return {
        "field": field_orig.astype(float).tolist(),
        "field_max": float(f[f_max_idx]),
        "seed": seed_orig,
        "tip": int(new2old[f_max_idx]),
        "saddles": [int(new2old[s]) for s in saddles],
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
