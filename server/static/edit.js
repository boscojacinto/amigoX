// Crochetability-aware mesh editor (Phase 3).
// initEditor(ctx) wires a floating editor panel onto the shared three.js scene.
// ctx: { THREE, scene, camera, renderer, controls, canvas, getMesh, getState,
//        loadMeshPayload, setStatus, markerSize }
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { computeBoundsTree, disposeBoundsTree, acceleratedRaycast }
  from "three-mesh-bvh";

export function initEditor(ctx) {
  const { THREE, scene, camera, renderer, controls, canvas } = ctx;

  // Accelerated raycasting for vertex picking on large meshes.
  THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
  THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
  THREE.Mesh.prototype.raycast = acceleratedRaycast;

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();

  let active = false;
  let selection = new Set();        // selected vertex ids
  let problems = [];                // from /api/localize
  let armed = null;                 // a problem armed by the checklist
  let brushRadius = 0.06;
  let originalPos = null;           // Float32Array snapshot for preview restore
  let undoStack = [];               // payloads
  let highlight = null;             // THREE.Points overlay
  let gizmoProxy = null, gizmo = null, gizmoStart = null;

  // ---- panel -------------------------------------------------------------
  const panel = document.createElement("div");
  panel.id = "editor-panel";
  panel.style.display = "none";
  panel.innerHTML = `
    <div class="ep-sec"><b>Crochetability</b><div id="ep-check"></div></div>
    <div class="ep-sec"><b>Selection</b>
      <div id="ep-selinfo" class="muted">click the mesh to select · shift-click adds</div>
      <label>brush <input id="ep-brush" type="range" min="0.02" max="0.2" step="0.01" value="0.06"></label>
      <div class="ep-row">
        <button id="ep-clear" class="ep-mini">clear</button>
        <button id="ep-move" class="ep-mini">move gizmo</button>
      </div>
    </div>
    <div class="ep-sec"><b>Tools</b>
      <label>inflate <input id="ep-inflate" type="range" min="-0.1" max="0.1" step="0.005" value="0"></label>
      <div class="ep-row">
        <button id="ep-inflate-go" class="ep-mini">apply inflate</button>
        <button id="ep-smooth" class="ep-mini">smooth</button>
      </div>
      <div id="ep-fix"></div>
    </div>
    <div class="ep-sec ep-row">
      <button id="ep-undo" class="ep-mini">↶ undo</button>
      <button id="ep-done" class="ep-mini">done</button>
    </div>`;
  document.getElementById("viewport").appendChild(panel);

  const $ = (id) => panel.querySelector(id);

  // ---- helpers -----------------------------------------------------------
  function geom() { return ctx.getMesh()?.geometry; }
  function posAttr() { return geom()?.getAttribute("position"); }

  function snapshot() {
    const p = posAttr();
    originalPos = new Float32Array(p.array);
  }
  function restore() {
    if (!originalPos) return;
    posAttr().array.set(originalPos);
    posAttr().needsUpdate = true;
    geom().computeVertexNormals();
  }

  function refreshHighlight() {
    if (highlight) { scene.remove(highlight); highlight.geometry.dispose(); highlight.material.dispose(); highlight = null; }
    if (!selection.size) { $("#ep-selinfo").textContent = "click the mesh to select · shift-click adds"; return; }
    const p = posAttr();
    const pts = [];
    for (const i of selection) pts.push(p.getX(i), p.getY(i), p.getZ(i));
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    highlight = new THREE.Points(g, new THREE.PointsMaterial({
      color: 0xffd24d, size: ctx.markerSize() * 2.2, sizeAttenuation: true, depthTest: false }));
    highlight.renderOrder = 999;
    scene.add(highlight);
    $("#ep-selinfo").textContent = `${selection.size} vertices selected`;
  }

  function selectIds(ids, additive) {
    if (!additive) selection.clear();
    for (const i of ids) selection.add(i);
    refreshHighlight();
  }

  // ---- picking -----------------------------------------------------------
  let dragged = false;
  canvas.addEventListener("pointerdown", () => { dragged = false; });
  canvas.addEventListener("pointermove", () => { dragged = true; });
  canvas.addEventListener("pointerup", (ev) => {
    if (!active || dragged || gizmo?.dragging) return;
    const mesh = ctx.getMesh();
    if (!mesh) return;
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObject(mesh, false);
    if (!hits.length) return;
    const hit = hits[0].point;
    const p = posAttr();
    const r2 = brushRadius * brushRadius;
    const ids = [];
    const v = new THREE.Vector3();
    for (let i = 0; i < p.count; i++) {
      v.set(p.getX(i), p.getY(i), p.getZ(i));
      if (v.distanceToSquared(hit) <= r2) ids.push(i);
    }
    selectIds(ids, ev.shiftKey);
  });

  // ---- live preview: inflate (along normals) -----------------------------
  function previewInflate(dist) {
    if (!selection.size) return;
    restore();
    const p = posAttr();
    const n = geom().getAttribute("normal");
    for (const i of selection) {
      p.setXYZ(i,
        originalPos[i * 3] + n.getX(i) * dist,
        originalPos[i * 3 + 1] + n.getY(i) * dist,
        originalPos[i * 3 + 2] + n.getZ(i) * dist);
    }
    p.needsUpdate = true;
    refreshHighlight();
  }

  // ---- gizmo move (soft falloff) -----------------------------------------
  function enableGizmo() {
    if (!selection.size) { ctx.setStatus("Select vertices first."); return; }
    disableGizmo();
    const p = posAttr();
    const c = new THREE.Vector3();
    for (const i of selection) c.add(new THREE.Vector3(p.getX(i), p.getY(i), p.getZ(i)));
    c.multiplyScalar(1 / selection.size);
    gizmoProxy = new THREE.Object3D(); gizmoProxy.position.copy(c); scene.add(gizmoProxy);
    gizmoStart = c.clone();
    snapshot();
    gizmo = new TransformControls(camera, renderer.domElement);
    gizmo.attach(gizmoProxy);
    gizmo.addEventListener("dragging-changed", (e) => { controls.enabled = !e.value; });
    gizmo.addEventListener("objectChange", () => {
      const d = gizmoProxy.position.clone().sub(gizmoStart);
      const pa = posAttr();
      for (const i of selection) {
        pa.setXYZ(i, originalPos[i * 3] + d.x, originalPos[i * 3 + 1] + d.y, originalPos[i * 3 + 2] + d.z);
      }
      pa.needsUpdate = true; refreshHighlight();
    });
    scene.add(gizmo);
  }
  function disableGizmo() {
    if (gizmo) { gizmo.detach(); scene.remove(gizmo); gizmo.dispose?.(); gizmo = null; }
    if (gizmoProxy) { scene.remove(gizmoProxy); gizmoProxy = null; }
    controls.enabled = true;
  }

  // ---- commit edits ------------------------------------------------------
  async function commit(op, params) {
    const state = ctx.getState();
    if (!state.meshId) return;
    ctx.setStatus(`Applying ${op}…`);
    try {
      const res = await fetch(`/api/edit/${state.meshId}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ op, params }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const data = await res.json();
      if (state.currentPayload) undoStack.push(state.currentPayload);
      disableGizmo();
      ctx.loadMeshPayload(data.mesh);     // re-render (also refreshes editor)
      renderChecklist(data.checklist);
      ctx.setStatus(`${op}: ${describe(data.report)}`);
    } catch (e) {
      ctx.setStatus("Error: " + e.message);
      restore();
    }
  }

  function describe(r) {
    if (r.genus_before !== undefined) return `genus ${r.genus_before}→${r.genus_after}`;
    if (r.watertight_after !== undefined) return `watertight: ${r.watertight_after}`;
    if (r.faces_after !== undefined) return `faces ${r.faces_before}→${r.faces_after}`;
    if (r.n_moved !== undefined) return `${r.n_moved} verts moved`;
    return "done";
  }

  function commitSelectionPositions() {
    if (!selection.size) return;
    const p = posAttr();
    const ids = [...selection];
    const positions = ids.map((i) => [p.getX(i), p.getY(i), p.getZ(i)]);
    commit("set_vertex_positions", { ids, positions });
  }

  // ---- checklist + armed fixes ------------------------------------------
  const CHECK_LABELS = {
    watertight: "Watertight", single_component: "1 component",
    genus_zero: "Genus 0", few_saddles: "Few saddles", thick_enough: "Thick enough",
  };
  function renderChecklist(checklist) {
    const el = $("#ep-check");
    el.innerHTML = "";
    for (const [k, v] of Object.entries(checklist)) {
      const chip = document.createElement("span");
      chip.className = "chip " + (v.ok ? "ok" : "bad");
      chip.textContent = `${CHECK_LABELS[k] || k}: ${v.value}`;
      el.appendChild(chip);
    }
  }

  function renderFixes() {
    const el = $("#ep-fix");
    el.innerHTML = "";
    if (!problems.length) { el.innerHTML = '<div class="muted">No problems detected ✓</div>'; return; }
    for (const pr of problems) {
      const btn = document.createElement("button");
      btn.className = "ep-mini ep-fix";
      btn.textContent = `${pr.suggested_tool.replace("_", " ")} — ${pr.type.replace("_", " ")}`;
      btn.title = pr.detail || "";
      btn.addEventListener("click", () => armFix(pr));
      el.appendChild(btn);
    }
  }

  function armFix(pr) {
    armed = pr;
    const ids = pr.region.ids;
    if (pr.region.kind === "faces") {
      // map faces → vertices for highlighting
      const idx = geom().index;
      const vset = new Set();
      for (const f of ids) { vset.add(idx.getX(f * 3)); vset.add(idx.getX(f * 3 + 1)); vset.add(idx.getX(f * 3 + 2)); }
      selectIds([...vset], false);
    } else {
      selectIds(ids, false);
    }
    // run the suggested tool directly (it already knows its region)
    const tool = pr.suggested_tool;
    if (tool === "fill_loop") commit("fill_loop", { loop_ids: ids });
    else if (tool === "delete_component") commit("delete_component", { face_ids: ids });
    else if (tool === "cut_handle") commit("cut_handle", { loop_ids: ids });
    else if (tool === "local_smooth") commit("local_smooth", { ids, iterations: 8 });
    else if (tool === "inflate") ctx.setStatus("Thin region selected — drag inflate or move gizmo, then apply.");
  }

  async function refreshLocalize() {
    const state = ctx.getState();
    if (!state.meshId) return;
    try {
      const res = await fetch(`/api/localize/${state.meshId}`);
      if (!res.ok) return;
      const data = await res.json();
      problems = data.problems;
      renderChecklist(data.checklist);
      renderFixes();
    } catch { /* ignore */ }
  }

  // ---- wiring ------------------------------------------------------------
  $("#ep-brush").addEventListener("input", (e) => { brushRadius = +e.target.value; });
  $("#ep-clear").addEventListener("click", () => { selection.clear(); disableGizmo(); refreshHighlight(); });
  $("#ep-move").addEventListener("click", enableGizmo);
  $("#ep-inflate").addEventListener("input", (e) => previewInflate(+e.target.value));
  $("#ep-inflate-go").addEventListener("click", () => { commitSelectionPositions(); $("#ep-inflate").value = 0; });
  $("#ep-smooth").addEventListener("click", () => {
    if (selection.size) commit("local_smooth", { ids: [...selection], iterations: 8 });
  });
  $("#ep-undo").addEventListener("click", () => {
    if (!undoStack.length) { ctx.setStatus("Nothing to undo."); return; }
    ctx.loadMeshPayload(undoStack.pop());
    ctx.setStatus("Undone.");
  });
  $("#ep-done").addEventListener("click", () => api.toggle(false));

  // ---- public api --------------------------------------------------------
  const api = {
    toggle(on) {
      active = on === undefined ? !active : on;
      panel.style.display = active ? "block" : "none";
      if (active) {
        snapshot();
        if (geom() && !geom().boundsTree) geom().computeBoundsTree();
        refreshLocalize();
        ctx.setStatus("Edit mode — click problems or the mesh, edit, then Generate.");
      } else {
        disableGizmo();
        selection.clear(); refreshHighlight();
      }
      return active;
    },
    isActive() { return active; },
    onMeshLoaded() {
      selection.clear(); disableGizmo();
      if (highlight) { scene.remove(highlight); highlight = null; }
      const g = geom();
      if (g) { g.disposeBoundsTree?.(); if (active) g.computeBoundsTree(); }
      if (active) { snapshot(); refreshLocalize(); }
    },
  };
  return api;
}
