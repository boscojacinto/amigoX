"""
Transducer: convert crochet-graph connectivity into stitch sequences.

The transducer walks two consecutive rows simultaneously, advancing
pointers in each row based on the coupling degree:

  * One current-row vertex → one next-row vertex :  sc  (single crochet)
  * x current-row vertices → one next-row vertex :  dec(x)  (decrease)
  * one current-row vertex → x next-row vertices :  inc(x)  (increase)

This is a linear-time operation (Section 5.1 of the paper).

A Stitch is a named tuple: (type, x)
  type ∈ {'sc', 'inc', 'dec', 'magic'}
  x    = multiplicity (x=1 for sc, x>1 for inc/dec)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class Stitch:
    type: str   # 'sc', 'inc', 'dec', 'magic'
    x: int = 1  # fan-out / fan-in for inc/dec


def generate_row_instructions(
    n_curr: int,
    n_next: int,
    coupling: list[tuple[int, int]],
) -> list[Stitch]:
    """
    Generate stitches for a single row transition.

    Parameters
    ----------
    n_curr   : number of vertices in current row S_i
    n_next   : number of vertices in next row S_{i+1}
    coupling : list of (j, k) pairs from col_edges[i]

    Returns
    -------
    stitches : list of Stitch objects in crochet order
    """
    # Build mapping: for each vertex in row_curr, which row_next vertices?
    conn_fwd: dict[int, list[int]] = {j: [] for j in range(n_curr)}
    conn_bwd: dict[int, list[int]] = {k: [] for k in range(n_next)}
    for j, k in coupling:
        conn_fwd[j].append(k)
        conn_bwd[k].append(j)

    # Remove duplicates while preserving order
    for j in conn_fwd:
        seen: set = set()
        conn_fwd[j] = [k for k in conn_fwd[j] if not (k in seen or seen.add(k))]
    for k in conn_bwd:
        seen = set()
        conn_bwd[k] = [j for j in conn_bwd[k] if not (j in seen or seen.add(j))]

    stitches: list[Stitch] = []
    pi = 0  # pointer in current row
    pj = 0  # pointer in next row

    while pi < n_curr or pj < n_next:
        if pi >= n_curr and pj < n_next:
            # Remaining next-row vertices with no current-row partner → sc
            stitches.append(Stitch("sc"))
            pj += 1
            continue
        if pj >= n_next and pi < n_curr:
            # Remaining current-row vertices → absorb into last dec
            pi += 1
            continue

        j = pi
        k = pj
        n_fwd = len(conn_fwd.get(j, []))
        n_bwd = len(conn_bwd.get(k, []))

        if n_bwd > 1:
            # dec: multiple current-row vertices → one next-row vertex
            x = n_bwd
            stitches.append(Stitch("dec", x))
            pi += x
            pj += 1
        elif n_fwd > 1:
            # inc: one current-row vertex → multiple next-row vertices
            x = n_fwd
            stitches.append(Stitch("inc", x))
            pi += 1
            pj += x
        else:
            stitches.append(Stitch("sc"))
            pi += 1
            pj += 1

    return stitches


def generate_all_instructions(rows_pos: list, col_edges: list) -> list[list[Stitch]]:
    """
    Generate stitch sequences for all rows.

    The first row is always the magic circle (or slip-stitch ring).
    """
    all_rows: list[list[Stitch]] = []
    n_rows = len(rows_pos)

    # Row 0: magic circle with as many sc as the first real row has vertices
    n_first = len(rows_pos[1]) if n_rows > 1 else 1
    all_rows.append([Stitch("magic", n_first)])

    for i in range(n_rows - 1):
        n_curr = len(rows_pos[i])
        n_next = len(rows_pos[i + 1])
        coupling = col_edges[i]
        stitches = generate_row_instructions(n_curr, n_next, coupling)
        all_rows.append(stitches)

    return all_rows
