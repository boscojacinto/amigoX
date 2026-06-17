"""
Build crochet-graph connectivity: row edges R and column edges C.

Column edges connect consecutive rows via the minimum-cost coupling found
by Dynamic Time Warping (DTW) on the 3-D embedded positions.

A coupling C between rows S_i (length n) and S_{i+1} (length m) is
a monotone path from (0,0) to (n-1, m-1) in index space.
DTW finds the coupling that minimises the total Euclidean distance
between coupled positions.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# DTW coupling
# ---------------------------------------------------------------------------

def dtw_coupling(pos_a: np.ndarray, pos_b: np.ndarray):
    """
    Minimum-cost monotone coupling between two sequences of 3-D points.

    Parameters
    ----------
    pos_a : (n, 3)
    pos_b : (m, 3)

    Returns
    -------
    path : list of (i, j) index pairs  (monotone, length >= max(n,m))
    """
    n, m = len(pos_a), len(pos_b)

    # Pairwise Euclidean cost matrix
    diff = pos_a[:, None, :] - pos_b[None, :, :]   # (n, m, 3)
    cost = np.sqrt(np.sum(diff ** 2, axis=2))       # (n, m)

    # Sakoe–Chiba band: the rows are seam-aligned and similar length, so the
    # correct coupling runs near the diagonal. Constraining DP to a band keeps
    # the coupling local — without it, DTW takes long "shortcut" matches across
    # concave loops (e.g. a heart's heart-shaped cross-section), which show up
    # as tangled column edges.
    band = max(abs(n - m) + 2, int(0.2 * max(n, m)))

    def in_band(i, j):
        center = i * (m - 1) / (n - 1) if n > 1 else 0
        return abs(j - center) <= band

    # DP table
    dp = np.full((n, m), np.inf)
    dp[0, 0] = cost[0, 0]

    for i in range(1, n):
        if in_band(i, 0):
            dp[i, 0] = dp[i - 1, 0] + cost[i, 0]
    for j in range(1, m):
        if in_band(0, j):
            dp[0, j] = dp[0, j - 1] + cost[0, j]
    for i in range(1, n):
        for j in range(1, m):
            if not in_band(i, j):
                continue
            dp[i, j] = cost[i, j] + min(dp[i - 1, j - 1],
                                         dp[i - 1, j],
                                         dp[i, j - 1])

    # Backtrack
    path = []
    i, j = n - 1, m - 1
    path.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            best = np.argmin([dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1]])
            if best == 0:
                i, j = i - 1, j - 1
            elif best == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))

    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Full graph connectivity
# ---------------------------------------------------------------------------

def build_connectivity(rows_pos: list):
    """
    Build row edges R and column edges C for the crochet graph.

    Parameters
    ----------
    rows_pos : list of (n_i, 3) arrays — one per row

    Returns
    -------
    row_edges : list of lists; row_edges[i] = list of (i, j) pairs
        within row i (consecutive vertex pairs)
    col_edges : list of lists; col_edges[i] = DTW coupling between
        row i and row i+1 as (j_in_row_i, k_in_row_{i+1}) index pairs
    """
    n_rows = len(rows_pos)

    # Row edges: consecutive pairs within each row
    row_edges = []
    for i, pos in enumerate(rows_pos):
        n = len(pos)
        row_edges.append([(j, j + 1) for j in range(n - 1)])

    # Column edges: DTW coupling between consecutive rows
    col_edges = []
    for i in range(n_rows - 1):
        pa = rows_pos[i]
        pb = rows_pos[i + 1]
        if len(pa) == 1 and len(pb) == 1:
            col_edges.append([(0, 0)])
        elif len(pa) == 1:
            col_edges.append([(0, j) for j in range(len(pb))])
        elif len(pb) == 1:
            col_edges.append([(j, 0) for j in range(len(pa))])
        else:
            col_edges.append(dtw_coupling(pa, pb))

    return row_edges, col_edges
