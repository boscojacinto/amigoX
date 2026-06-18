#!/usr/bin/env python3
"""
AmiGoX — Closed-loop seed optimization.

Finds the seed vertex that yields the best crochet pattern for a mesh — one that
covers the whole surface with no floating stitches — then writes the pattern.

By default the Claude Opus 4.8 agent drives the loop: it lists candidate seeds,
generates and analyzes the pattern each produces, and iterates to the best.
Use --deterministic (or run without an API key) to rank candidates with the
quality metric directly, no LLM.

Usage:
    python optimize.py mesh.obj
    python optimize.py mesh.obj --stitch-width 0.04
    python optimize.py mesh.obj --deterministic --out pattern.txt
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="Optimize the crochet seed (AmiGoX).")
    ap.add_argument("mesh", help="Path to a triangle mesh (.obj, .ply, .stl, …)")
    ap.add_argument("--stitch-width", type=float, default=0.05,
                    help="Stitch size relative to the normalised mesh (default 0.05).")
    ap.add_argument("--deterministic", action="store_true",
                    help="Skip the LLM; rank candidate seeds with the quality metric.")
    ap.add_argument("--out", default=None,
                    help="Write the winning pattern here (default <mesh>.pattern.txt).")
    args = ap.parse_args()

    from amigo.mesh_ops import load_mesh
    from amigo import amigo_pipeline

    V, F = load_mesh(args.mesh)

    choice = None
    if args.deterministic:
        choice = _deterministic(V, F, args.stitch_width)
    else:
        choice = _agentic(V, F, args.stitch_width)
        if choice is None:
            print("\nAgent did not converge; falling back to deterministic ranking.")
            choice = _deterministic(V, F, args.stitch_width)

    if choice is None:
        print("No usable seed found.", file=sys.stderr)
        sys.exit(1)

    seed = int(choice["seed"])
    print("\n" + "=" * 60)
    print(f"  BEST SEED: {seed}   (score {choice.get('score', float('nan')):.4f})")
    print("=" * 60)
    if "coverage" in choice:
        print(f"  coverage={choice['coverage']:.4f}  "
              f"floating={choice.get('n_floating', '?')}  "
              f"thin_segments={choice.get('n_thin_segments', '?')}")
    for r in choice.get("reasons", []):
        print(f"  • {r}")

    pattern = amigo_pipeline(mesh_path=args.mesh, seed_idx=seed,
                             stitch_width=args.stitch_width, verbose=False)
    out = args.out or (os.path.splitext(args.mesh)[0] + ".pattern.txt")
    header = "=" * 60 + "\n  AmiGoX — Crochet Pattern (auto-seed)\n" + "=" * 60
    with open(out, "w") as fh:
        fh.write(header + "\n\n" + pattern + "\n")
    print(f"\nPattern written to {out}")
    print(f"  (regenerate any time:  python main.py {args.mesh} --seed {seed} "
          f"--stitch-width {args.stitch_width})")


def _deterministic(V, F, stitch_width):
    from amigo import rank_candidates
    print("Ranking candidate seeds (deterministic)…")
    ranked = rank_candidates(V, F, stitch_width=stitch_width)
    for m in ranked:
        tag = "ok " if m.get("ran") else "err"
        print(f"  [{tag}] seed {m.get('seed'):>6}  score={m.get('score', 0):.4f}"
              f"  ({m.get('descriptor', '')})")
    best = next((m for m in ranked if m.get("ran")), None)
    if best is None:
        return None
    best.setdefault("reasons", [f"highest deterministic score among "
                                f"{len(ranked)} candidates"])
    return best


def _agentic(V, F, stitch_width):
    from amigo import run_seed_optimization
    state = {"current": (V, F), "applied": []}

    def emit(ev):
        t = ev["type"]
        if t in ("thinking", "text"):
            sys.stdout.write(ev["text"])
        elif t == "tool_call":
            print(f"\n  → {ev['name']}({_short(ev['input'])})")
        elif t == "tool_result":
            print(f"    ✓ {ev['name']}" + (" [error]" if ev["is_error"] else ""))
        elif t == "seed_eval":
            s = ev["summary"]
            if s.get("ran"):
                print(f"      · seed {s['seed']}: score={s['score']} "
                      f"coverage={s['coverage']} floating={s['n_floating']} "
                      f"segments={s['n_segments']}")
            else:
                print(f"      · seed {s.get('seed')}: failed ({s.get('error', '')})")
        elif t == "seed_choice":
            pass  # printed by the caller
        elif t == "error":
            print(f"\n[error] {ev['message']}")
        sys.stdout.flush()

    try:
        choice = run_seed_optimization(state, emit=emit, stitch_width=stitch_width)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return None
    if not choice:
        return None
    # Merge the agent's choice with the full metrics it recorded for that seed.
    metrics = state.get("metrics", {}).get(int(choice["seed"]), {})
    merged = {**metrics, **choice}
    return merged


def _short(d, n=80):
    s = str(d)
    return s if len(s) <= n else s[:n] + "…"


if __name__ == "__main__":
    main()
