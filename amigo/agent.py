"""
Crochetability agent (Claude Opus 4.8).

A manual, streaming agentic loop that judges whether a mesh is crochetable by
the AmiGo technique and, with the user's per-step approval, applies
simplification transforms to make it so.

The loop is shared between the CLI and the web UI through two callbacks:

    run_assessment(state, *, emit, get_decision) -> dict | None

  * ``state``        : {"current": (V, F), "applied": [...]}  (mutated in place)
  * ``emit(event)``  : receives progress events (thinking / text / tool_call /
                       tool_result / verdict / error)
  * ``get_decision`` : called for the *gated* mutation tool only; returns
                       {"approve": bool, "note": str}. The CLI prompts the
                       terminal; the web pushes an approval request and waits.

Returns the structured verdict dict (or None if the model ended without one).
Mesh mutations land in ``state["current"]``.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .diagnostics import analyze_mesh
from .simplify import apply as apply_transform, TRANSFORMS
from .edit import apply as apply_edit, EDIT_OPS
from .localize import localize
from .mesh_ops import normalize_to_unit_area
from .pipeline import amigo_pipeline_data

MODEL = "claude-opus-4-8"


def _load_dotenv():
    """Load KEY=VALUE pairs from a nearby .env into os.environ (no override).

    The Anthropic SDK reads credentials from the environment but does not parse
    .env files, so we do it here. Searches the package dir, the project root,
    and the current working directory.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, ".env"),                       # amigo/.env
        os.path.join(os.path.dirname(here), ".env"),      # project-root/.env
        os.path.join(os.getcwd(), ".env"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except OSError:
            pass

SYSTEM_PROMPT = """\
You assess whether a 3D triangle mesh can be turned into a single-piece \
amigurumi crochet pattern by the AmiGo technique, and if not, help make it so.

How the AmiGo technique works: it computes a geodesic distance field from a \
seed vertex, treats the field's isolines as crochet rows, segments the mesh at \
saddle points of the field to handle limbs/branches, and walks consecutive rows \
to emit single-crochet / increase / decrease stitches.

A mesh is cleanly crochetable when ALL of these hold:
- It is a closed, orientable 2-manifold (watertight, consistent winding).
- It is a SINGLE connected component.
- It is ~genus 0. Saddle segmentation handles limbs/branches, but handles or \
tunnels (genus > 0) are NOT supported — the isolines stop being simple loops.
- The geodesic field is a reasonable Morse function: not fragmented into many \
tiny segments by high-frequency surface noise (a huge saddle count is a red flag).
- Features are thick enough for stitches: each segment needs at least ~2 rows \
and interior rows need at least ~3 stitches. Thin spikes/sheets fail.

You can inspect the mesh and fix it two ways. Each fix is shown to the user for \
approval before it is applied; if denied, adapt.

GLOBAL simplifications (apply_simplification) — for whole-mesh issues:
- keep_largest_component / remove_small_components : multiple disconnected pieces
- fill_holes      : boundaries / open (non-watertight) mesh
- smooth          : high-frequency detail that spawns spurious saddles
- decimate        : excessive face count (robustness / speed)

LOCALIZED edits (propose_edit) — for problems tied to specific geometry. First \
call get_problems to get the localized problem list (each carries the exact \
region ids and a suggested_tool); then propose_edit with that op and region:
- fill_loop       : close a specific open boundary loop (region.ids = loop)
- delete_component: remove a disconnected piece (region.ids = faces)
- cut_handle      : reduce genus by cutting a tunnel loop and capping (region.ids = loop)
- inflate         : thicken a thin region (region.ids = vertices; params {distance})
- local_smooth    : smooth a noisy patch (region.ids = vertices)

cut_handle and inflate mean genus reduction and limb-thickening ARE now \
achievable — use them when get_problems reports a handle or thin_region. Only \
fall back to recommended_manual_steps for things no tool covers.

Process:
1. get_diagnostics, then get_problems, to see topology, the pipeline front, and \
the localized problem list.
2. Optionally try_pipeline (different seeds) to test empirically.
3. Fix the MINIMAL set of problems: apply_simplification for global issues, \
propose_edit (with the region from get_problems) for localized ones. Re-check \
with get_problems / get_diagnostics after each.
4. Call submit_verdict exactly once. Be honest if it still cannot be made crochetable.

Keep narration brief. Prefer the fewest fixes that make the mesh crochetable.\
"""


def _tools() -> list[dict]:
    techniques = list(TRANSFORMS.keys())
    return [
        {
            "name": "get_diagnostics",
            "description": "Compute crochetability diagnostics (topology, geometry, "
                           "and AmiGo pipeline-front behaviour) for the current mesh.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "try_pipeline",
            "description": "Run the full AmiGo pipeline on the current mesh and report a "
                           "summary (segments, rows, pattern size) or the error it raised. "
                           "Use to empirically test crochetability and seed choices.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "seed": {"type": "integer",
                             "description": "Seed vertex index; omit for the default pole."},
                    "stitch_width": {"type": "number",
                                     "description": "Stitch size (default 0.05)."},
                },
            },
        },
        {
            "name": "get_problems",
            "description": "Return the localized crochetability problems for the current "
                           "mesh — each with the exact region (vertex/face/loop ids) and a "
                           "suggested_tool — plus the red/green checklist. Use the region "
                           "ids in a propose_edit call.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "propose_edit",
            "description": "Propose a localized edit using a region from get_problems. The "
                           "user must approve it before it is applied. On approval the mesh "
                           "is replaced and fresh diagnostics are returned.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": list(EDIT_OPS.keys())},
                    "params": {"type": "object",
                               "description": "Op params, e.g. {\"loop_ids\":[...]}, "
                                              "{\"face_ids\":[...]}, or "
                                              "{\"ids\":[...],\"distance\":0.03}."},
                    "rationale": {"type": "string"},
                },
                "required": ["op", "params", "rationale"],
            },
        },
        {
            "name": "apply_simplification",
            "description": "Propose a simplification transform. The user must approve it "
                           "before it is applied to the mesh. On approval, the mesh is "
                           "replaced and fresh diagnostics are returned.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "technique": {"type": "string", "enum": techniques},
                    "params": {"type": "object",
                               "description": "Optional technique parameters "
                                              "(e.g. {\"iterations\": 15} for smooth)."},
                    "rationale": {"type": "string",
                                  "description": "Why this helps crochetability."},
                },
                "required": ["technique", "rationale"],
            },
        },
        {
            "name": "submit_verdict",
            "description": "Submit the final crochetability verdict. Call exactly once.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "crochetable": {"type": "string",
                                    "enum": ["yes", "no", "with_modifications"]},
                    "confidence": {"type": "string",
                                   "enum": ["low", "medium", "high"]},
                    "reasons": {"type": "array", "items": {"type": "string"}},
                    "applied_steps": {"type": "array", "items": {"type": "string"}},
                    "recommended_manual_steps": {"type": "array",
                                                 "items": {"type": "string"}},
                    "summary": {"type": "string"},
                },
                "required": ["crochetable", "confidence", "reasons",
                             "applied_steps", "recommended_manual_steps", "summary"],
                "additionalProperties": False,
            },
        },
    ]


def _run_tool(name, tool_input, state, emit, get_decision):
    """Execute a tool. Returns (result_str, is_error, verdict_or_None)."""
    if name == "get_diagnostics":
        V, F = state["current"]
        return json.dumps(analyze_mesh(V, F)), False, None

    if name == "try_pipeline":
        V, F = state["current"]
        Vn, _ = normalize_to_unit_area(V, F)
        seed = int(tool_input["seed"]) if tool_input.get("seed") is not None else None
        if seed is None:
            from .diagnostics import _principal_pole_seed
            seed = _principal_pole_seed(Vn)
        sw = float(tool_input.get("stitch_width", 0.05))
        try:
            data = amigo_pipeline_data(Vn, F, seed_idx=seed,
                                       stitch_width=sw, verbose=False)
            summ = {
                "ran": True, "seed": seed, "stitch_width": sw,
                "n_saddles": len(data["saddles"]), "n_segments": len(data["segments"]),
                "pattern_lines": data["pattern"].count("\n") + 1,
                "rows_per_segment": [len(s["rows"]) for s in data["segments"]],
            }
        except Exception as exc:  # noqa: BLE001
            summ = {"ran": False, "error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(summ), not summ["ran"], None

    if name == "get_problems":
        V, F = state["current"]
        return json.dumps(localize(V, F)), False, None

    if name in ("apply_simplification", "propose_edit"):
        is_edit = name == "propose_edit"
        technique = tool_input["op"] if is_edit else tool_input["technique"]
        params = tool_input.get("params") or {}
        rationale = tool_input.get("rationale", "")
        registry = EDIT_OPS if is_edit else TRANSFORMS
        if technique not in registry:
            return f"Unknown op '{technique}'.", True, None
        decision = get_decision({"technique": technique, "params": params,
                                 "rationale": rationale})
        if not decision.get("approve"):
            note = decision.get("note", "")
            return (f"User declined to apply '{technique}'."
                    + (f" Note: {note}" if note else "")), False, None
        V, F = state["current"]
        try:
            if is_edit:
                r = apply_edit(technique, V, F, params)
                V2, F2, report = r["V"], r["F"], r["report"]
            else:
                V2, F2, report = apply_transform(technique, V, F, params)
        except Exception as exc:  # noqa: BLE001
            return f"'{technique}' failed: {type(exc).__name__}: {exc}", True, None
        state["current"] = (np.asarray(V2, dtype=np.float64),
                            np.asarray(F2, dtype=np.int64))
        state["applied"].append({"technique": technique, "params": params,
                                 "rationale": rationale, "report": report})
        emit({"type": "applied", "report": report})
        result = {"report": report, "diagnostics": analyze_mesh(V2, F2),
                  "checklist": localize(V2, F2)["checklist"]}
        return json.dumps(result), False, None

    if name == "submit_verdict":
        return None, False, dict(tool_input)

    return f"Unknown tool '{name}'.", True, None


def run_assessment(state, *, emit, get_decision, max_steps: int = 16):
    """
    Drive the crochetability agent. See module docstring.

    Requires ANTHROPIC_API_KEY in the environment (or a nearby .env file).
    """
    _load_dotenv()
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set — the crochetability agent needs an "
            "Anthropic API key (claude-opus-4-8).")

    import anthropic
    client = anthropic.Anthropic()
    tools = _tools()
    messages = [{
        "role": "user",
        "content": "Assess whether this mesh is crochetable by the AmiGo technique. "
                   "Start by calling get_diagnostics.",
    }]
    verdict = None

    for _ in range(max_steps):
        with client.messages.stream(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": "high"},
            tools=tools,
            messages=messages,
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        emit({"type": "text", "text": event.delta.text})
                    elif event.delta.type == "thinking_delta":
                        emit({"type": "thinking", "text": event.delta.thinking})
            final = stream.get_final_message()

        messages.append({"role": "assistant", "content": final.content})

        tool_uses = [b for b in final.content if b.type == "tool_use"]
        if not tool_uses:
            break  # model ended its turn without (more) tools

        tool_results = []
        stop = False
        for tu in tool_uses:
            emit({"type": "tool_call", "name": tu.name, "input": tu.input})
            result, is_error, maybe_verdict = _run_tool(
                tu.name, tu.input, state, emit, get_decision)
            if maybe_verdict is not None:
                verdict = maybe_verdict
                emit({"type": "verdict", "verdict": verdict,
                      "mesh_changed": len(state["applied"]) > 0})
                stop = True
                break
            emit({"type": "tool_result", "name": tu.name, "is_error": is_error})
            tool_results.append({
                "type": "tool_result", "tool_use_id": tu.id,
                "content": result, "is_error": is_error,
            })
        if stop:
            break
        messages.append({"role": "user", "content": tool_results})

    return verdict
