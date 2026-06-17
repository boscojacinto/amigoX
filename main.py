#!/usr/bin/env python3
"""
AmiGo — Computational Design of Amigurumi Crochet Patterns
CLI entry point.

Usage:
    python main.py mesh.obj
    python main.py mesh.obj --seed 42 --stitch-width 0.04
    python main.py mesh.obj --seed 0 --stitch-width 0.05 --out pattern.txt
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate amigurumi crochet patterns from a 3D mesh."
    )
    parser.add_argument("mesh", help="Path to a closed triangle mesh (.obj, .ply, …)")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Vertex index to start crocheting from (default: 0)"
    )
    parser.add_argument(
        "--stitch-width", type=float, default=0.05,
        help="Stitch size relative to normalised mesh (default: 0.05 ≈ 20 rows)"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-3,
        help="Laplacian regularisation weight for the column-order solve (default: 1e-3)"
    )
    parser.add_argument(
        "--format", choices=["text", "crochetparade"], default="text",
        help="Output format: 'text' (default) or 'crochetparade' (.cp file)"
    )
    parser.add_argument(
        "--out", default=None,
        help="Write pattern to this file instead of stdout"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress messages"
    )
    args = parser.parse_args()

    from amigo import amigo_pipeline

    pattern = amigo_pipeline(
        mesh_path=args.mesh,
        seed_idx=args.seed,
        stitch_width=args.stitch_width,
        eps=args.eps,
        verbose=not args.quiet,
        output_format=args.format,
    )

    if args.format == "crochetparade":
        output = pattern
    else:
        header = (
            "=" * 60 + "\n"
            + "  AmiGo — Crochet Pattern\n"
            + "=" * 60
        )
        output = header + "\n\n" + pattern + "\n"

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(output)
        print(f"Pattern written to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
