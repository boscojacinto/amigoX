"""
AmiGoX web UI backend (FastAPI).

Endpoints
---------
GET  /                -> serves the single-page frontend
POST /api/upload      -> upload a mesh, returns normalised geometry for rendering
POST /api/run         -> run the AmiGo pipeline on a previously uploaded mesh,
                         returns the geodesic field, saddles, 3-D stitch rows
                         and the human-readable pattern.

The uploaded, *normalised* (V, F) arrays are cached server-side keyed by an id.
The pipeline runs on those exact arrays so that a vertex index picked in the
browser (by clicking the mesh) matches the pipeline's ``seed_idx``.
"""

from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import traceback
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from amigo import amigo_pipeline_data
from amigo.mesh_ops import load_mesh, normalize_to_unit_area

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")

app = FastAPI(title="AmiGoX")


@app.middleware("http")
async def no_cache(request, call_next):
    """Prevent the browser from caching the frontend during development."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


# In-memory cache of uploaded meshes: id -> {"V": ndarray, "F": ndarray}
_MESHES: dict[str, dict] = {}
# In-memory crochetability-assessment sessions: sid -> {events, decisions}
_SESSIONS: dict[str, dict] = {}


def _register_mesh(V, F) -> dict:
    """Cache a mesh and return the geometry payload the frontend renders."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    mesh_id = uuid.uuid4().hex
    _MESHES[mesh_id] = {"V": V, "F": F}
    center = V.mean(axis=0)
    radius = float(np.linalg.norm(V - center, axis=1).max())
    return {
        "id": mesh_id,
        "n_verts": int(len(V)),
        "n_faces": int(len(F)),
        "vertices": V.astype(float).flatten().tolist(),
        "faces": F.astype(int).flatten().tolist(),
        "center": center.astype(float).tolist(),
        "radius": radius,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "mesh.obj")[1] or ".obj"
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        V, F = load_mesh(tmp_path)
        V, scale = normalize_to_unit_area(V, F)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to load mesh: {exc}")
    finally:
        os.unlink(tmp_path)
    return _register_mesh(V, F)


class RunRequest(BaseModel):
    id: str
    seed: int = 0
    stitch_width: float = 0.05


@app.post("/api/run")
async def run(req: RunRequest):
    mesh = _MESHES.get(req.id)
    if mesh is None:
        raise HTTPException(status_code=404, detail="Mesh not found — upload it first.")

    V, F = mesh["V"], mesh["F"]
    if not (0 <= req.seed < len(V)):
        raise HTTPException(status_code=400, detail="Seed index out of range.")
    if req.stitch_width <= 0:
        raise HTTPException(status_code=400, detail="Stitch width must be positive.")

    try:
        result = amigo_pipeline_data(
            V, F,
            seed_idx=req.seed,
            stitch_width=req.stitch_width,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    # Drop the heavyweight instruction objects; the UI only needs render data.
    result.pop("instructions", None)
    return result


# ----------------------------------------------------------------------------
# Crochetability agent (Phase 2)
# ----------------------------------------------------------------------------
class AssessRequest(BaseModel):
    id: str


class DecisionRequest(BaseModel):
    approve: bool
    note: str = ""


def _run_session(sid: str, V, F):
    """Background thread: drive the agent, bridging callbacks to the queues."""
    from amigo import run_assessment

    sess = _SESSIONS[sid]
    state = {"current": (np.asarray(V, dtype=np.float64),
                         np.asarray(F, dtype=np.int64)), "applied": []}

    def emit(ev):
        sess["events"].put(ev)

    def get_decision(proposal):
        sess["events"].put({"type": "approval_request", "proposal": proposal})
        return sess["decisions"].get()  # blocks until /decision arrives

    try:
        run_assessment(state, emit=emit, get_decision=get_decision)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        sess["events"].put({"type": "error", "message": str(exc)})

    if state["applied"]:
        V2, F2 = state["current"]
        sess["events"].put({"type": "simplified", "mesh": _register_mesh(V2, F2)})
    sess["events"].put({"type": "_end"})


@app.post("/api/assess")
async def assess_start(req: AssessRequest):
    mesh = _MESHES.get(req.id)
    if mesh is None:
        raise HTTPException(status_code=404, detail="Mesh not found — upload it first.")
    sid = uuid.uuid4().hex
    _SESSIONS[sid] = {"events": queue.Queue(), "decisions": queue.Queue()}
    threading.Thread(target=_run_session, args=(sid, mesh["V"], mesh["F"]),
                     daemon=True).start()
    return {"session_id": sid}


@app.get("/api/assess/{sid}/stream")
def assess_stream(sid: str):
    sess = _SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="Assessment session not found.")

    def gen():
        while True:
            ev = sess["events"].get()
            if ev.get("type") == "_end":
                yield "data: " + json.dumps({"type": "end"}) + "\n\n"
                break
            yield "data: " + json.dumps(ev) + "\n\n"
        _SESSIONS.pop(sid, None)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no"})


@app.post("/api/assess/{sid}/decision")
async def assess_decision(sid: str, req: DecisionRequest):
    sess = _SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="Assessment session not found.")
    sess["decisions"].put({"approve": req.approve, "note": req.note})
    return {"ok": True}


# ----------------------------------------------------------------------------
# Crochetability editor (Phase 3)
# ----------------------------------------------------------------------------
class EditRequest(BaseModel):
    op: str
    params: dict = {}


@app.get("/api/localize/{mesh_id}")
async def localize_mesh(mesh_id: str):
    mesh = _MESHES.get(mesh_id)
    if mesh is None:
        raise HTTPException(status_code=404, detail="Mesh not found — upload it first.")
    from amigo.localize import localize
    try:
        return localize(mesh["V"], mesh["F"])
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Localize error: {exc}")


@app.post("/api/edit/{mesh_id}")
async def edit_mesh(mesh_id: str, req: EditRequest):
    mesh = _MESHES.get(mesh_id)
    if mesh is None:
        raise HTTPException(status_code=404, detail="Mesh not found — upload it first.")
    from amigo import edit as edit_ops
    from amigo.localize import localize
    try:
        result = edit_ops.apply(req.op, mesh["V"], mesh["F"], req.params)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Edit error: {exc}")
    payload = _register_mesh(result["V"], result["F"])
    new_mesh = _MESHES[payload["id"]]
    return {
        "mesh": payload,
        "report": result["report"],
        "checklist": localize(new_mesh["V"], new_mesh["F"])["checklist"],
    }


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
