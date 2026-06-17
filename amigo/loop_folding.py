"""
Loop folding: convert raw stitch sequences into compact, human-readable
crochet patterns (Section 5.2 of the paper).

Folding happens at three levels, in order:
  1. Sequences: find maximal repeating subsequences within a row.
  2. Stitches: collapse runs of identical stitches (e.g. sc,sc,sc → 3sc).
  3. Rows: merge consecutive identical rows into a range.

The output is a list of strings, one per row (or row range).
"""

from __future__ import annotations
from .instructions import Stitch


# ---------------------------------------------------------------------------
# Stitch formatting
# ---------------------------------------------------------------------------

def _stitch_str(s: Stitch) -> str:
    if s.type == "magic":
        return f"magic ring, {s.x}sc"
    if s.type == "sc":
        return "sc"
    if s.type == "inc":
        return "inc" if s.x == 2 else f"inc({s.x})"
    if s.type == "dec":
        return "dec" if s.x == 2 else f"dec({s.x})"
    return s.type


def _row_total(stitches: list[Stitch]) -> int:
    """Total stitch count in the NEXT row after executing these stitches."""
    total = 0
    for s in stitches:
        if s.type == "sc":
            total += 1
        elif s.type == "inc":
            total += s.x
        elif s.type == "dec":
            total += 1
        elif s.type == "magic":
            total += s.x
        elif s.type == "join":
            total += s.x
    return total


# ---------------------------------------------------------------------------
# Level 2: collapse runs of identical stitches
# ---------------------------------------------------------------------------

def _fold_runs(stitches: list[Stitch]) -> list:
    """
    Replace runs of identical adjacent stitches with (count, stitch) tuples.
    E.g. [sc, sc, sc] → [(3, sc)]
    Single stitches become (1, stitch).
    """
    if not stitches:
        return []
    result = []
    run_type = stitches[0].type
    run_x = stitches[0].x
    count = 1
    for s in stitches[1:]:
        if s.type == run_type and s.x == run_x:
            count += 1
        else:
            result.append((count, Stitch(run_type, run_x)))
            run_type, run_x, count = s.type, s.x, 1
    result.append((count, Stitch(run_type, run_x)))
    return result


def _run_str(count: int, s: Stitch) -> str:
    if count == 1:
        return _stitch_str(s)
    if s.type == "sc":
        return f"{count}sc"
    base = _stitch_str(s)
    return f"{count}{base}"


# ---------------------------------------------------------------------------
# Level 1: find maximal repeating subsequence within a run-folded row
# ---------------------------------------------------------------------------

def _find_period(seq: list) -> int | None:
    """
    Return smallest period p such that seq is a perfect repetition of seq[:p],
    or None if no such p < len(seq) exists.
    """
    n = len(seq)
    for p in range(1, n):
        if n % p == 0 and seq == seq[:p] * (n // p):
            return p
    return None


def _fold_sequences(runs: list) -> str:
    """
    Given a run-folded row (list of (count, Stitch)), try to find a repeating
    subsequence and format accordingly.
    """
    if not runs:
        return ""

    p = _find_period(runs)
    if p is not None:
        unit = runs[:p]
        reps = len(runs) // p
        unit_str = ", ".join(_run_str(c, s) for c, s in unit)
        if reps == 1:
            return unit_str
        return f"({unit_str})*{reps}"

    return ", ".join(_run_str(c, s) for c, s in runs)


# ---------------------------------------------------------------------------
# Row formatting
# ---------------------------------------------------------------------------

def format_row(row_idx: int, stitches: list[Stitch]) -> str:
    """
    Format a single row as a human-readable string.
    """
    if not stitches:
        return f"Row {row_idx}: —"

    total = _row_total(stitches)

    if len(stitches) == 1 and stitches[0].type == "magic":
        return f"Row {row_idx}: {_stitch_str(stitches[0])} ({total})"

    runs = _fold_runs(stitches)
    body = _fold_sequences(runs)
    return f"Row {row_idx}: {body} ({total})"


# ---------------------------------------------------------------------------
# Level 3: merge identical consecutive rows
# ---------------------------------------------------------------------------

def _rows_equal(a: list[Stitch], b: list[Stitch]) -> bool:
    if len(a) != len(b):
        return False
    return all(sa.type == sb.type and sa.x == sb.x for sa, sb in zip(a, b))


def format_pattern(all_instructions: list[list[Stitch]]) -> str:
    """
    Format the complete pattern as a multi-line string, merging identical rows.
    """
    lines = []
    n = len(all_instructions)
    i = 0
    display_row = 1

    while i < n:
        # A join row starts a new limb worked into a shared boundary loop.
        block0 = all_instructions[i]
        if len(block0) == 1 and block0[0].type == "join":
            lines.append("")
            lines.append(f"— New limb: work {block0[0].x}sc into the boundary —")
            i += 1
            continue

        start = i
        j = i + 1
        while j < n and _rows_equal(all_instructions[i], all_instructions[j]):
            j += 1

        block = all_instructions[i]
        end = j - 1
        total = _row_total(block)
        runs = _fold_runs(block)
        body = _fold_sequences(runs)

        if len(block) == 1 and block[0].type == "magic":
            body = _stitch_str(block[0])

        if start == end:
            label = f"Row {display_row}"
        else:
            span = end - start + 1
            label = f"Rows {display_row}-{display_row + span - 1}"
            display_row += span - 1

        lines.append(f"{label}: {body} ({total})")
        display_row += 1
        i = j

    return "\n".join(lines)
