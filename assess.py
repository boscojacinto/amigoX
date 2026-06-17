#!/usr/bin/env python3
"""
AmiGoX — Crochetability assessment (Phase 2).

Runs the Claude Opus 4.8 agent to judge whether a 3D mesh is crochetable by the
AmiGo technique and, with your per-step approval, apply simplifications to make
it so.

Usage:
    export ANTHROPIC_API_KEY=...
    python assess.py mesh.obj
    python assess.py mesh.obj --stitch-width 0.04

Approved transforms are applied to a working copy; if the mesh changed, the
result is written to <mesh>.simplified.obj.
"""

import argparse
import os
import sys

import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser(description="Assess mesh crochetability (AmiGoX agent).")
    ap.add_argument("mesh", help="Path to a triangle mesh (.obj, .ply, .stl, …)")
    ap.add_argument("--yes", action="store_true",
                    help="Auto-approve every proposed simplification.")
    args = ap.parse_args()

    from amigo.mesh_ops import load_mesh
    from amigo import run_assessment

    V, F = load_mesh(args.mesh)
    state = {"current": (V, F), "applied": []}

    def emit(ev):
        t = ev["type"]
        if t == "thinking":
            sys.stdout.write(ev["text"])
        elif t == "text":
            sys.stdout.write(ev["text"])
        elif t == "tool_call":
            print(f"\n  → {ev['name']}({_short(ev['input'])})")
        elif t == "tool_result":
            print(f"    ✓ {ev['name']}" + (" [error]" if ev["is_error"] else ""))
        elif t == "applied":
            print(f"    ✎ applied: {ev['report']}")
        elif t == "verdict":
            _print_verdict(ev["verdict"])
        elif t == "error":
            print(f"\n[error] {ev['message']}")
        sys.stdout.flush()

    def get_decision(proposal):
        print(f"\n  ⚙ Proposed: {proposal['technique']}  "
              f"params={proposal['params']}\n    rationale: {proposal['rationale']}")
        if args.yes:
            print("    auto-approved (--yes)")
            return {"approve": True, "note": ""}
        ans = input("    Apply this? [y/N] ").strip().lower()
        return {"approve": ans in ("y", "yes"), "note": ""}

    try:
        run_assessment(state, emit=emit, get_decision=get_decision)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(1)

    if state["applied"]:
        out = os.path.splitext(args.mesh)[0] + ".simplified.obj"
        V2, F2 = state["current"]
        trimesh.Trimesh(vertices=V2, faces=F2, process=False).export(out)
        print(f"\nSimplified mesh written to {out}")
        print(f"  Run it through the pattern generator:  python main.py {out}")


def _short(d, n=80):
    s = str(d)
    return s if len(s) <= n else s[:n] + "…"


def _print_verdict(v):
    print("\n" + "=" * 60)
    print(f"  CROCHETABLE: {v['crochetable'].upper()}  "
          f"(confidence: {v['confidence']})")
    print("=" * 60)
    print(v.get("summary", ""))
    if v.get("reasons"):
        print("\nReasons:")
        for r in v["reasons"]:
            print(f"  • {r}")
    if v.get("applied_steps"):
        print("\nApplied:")
        for s in v["applied_steps"]:
            print(f"  ✎ {s}")
    if v.get("recommended_manual_steps"):
        print("\nRecommended manual steps:")
        for s in v["recommended_manual_steps"]:
            print(f"  → {s}")


if __name__ == "__main__":
    main()
