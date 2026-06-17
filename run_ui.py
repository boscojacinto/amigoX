#!/usr/bin/env python3
"""
Launch the AmiGoX interactive web UI.

    python run_ui.py            # serves on http://127.0.0.1:8000
    python run_ui.py --port 9000

Then open the printed URL, upload a mesh, click to pick a seed vertex,
choose a stitch width, and generate the crochet pattern.
"""
import argparse

import uvicorn


def main():
    ap = argparse.ArgumentParser(description="AmiGoX web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    args = ap.parse_args()

    print(f"\n  AmiGoX UI  →  http://{args.host}:{args.port}\n")
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
