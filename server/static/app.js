import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { initEditor } from "./edit.js?v=1";

let editor = null;  // Phase-3 mesh editor (created after scene setup)

// ----------------------------------------------------------------------------
// State
// ----------------------------------------------------------------------------
const state = {
  meshId: null,
  vertices: null,   // Float32Array (flat xyz)
  faces: null,      // Int32Array (flat)
  nVerts: 0,
  seed: null,       // picked vertex index
  center: new THREE.Vector3(),
  radius: 1,
  roiVertices: new Set(),  // brushed region-of-interest vertex indices
};

const els = {
  file: document.getElementById("file"),
  sw: document.getElementById("sw"),
  swVal: document.getElementById("sw-val"),
  autoWidth: document.getElementById("auto-width"),
  run: document.getElementById("run"),
  assess: document.getElementById("assess"),
  autoSeed: document.getElementById("auto-seed"),
  roiBrush: document.getElementById("roi-brush"),
  roiClear: document.getElementById("roi-clear"),
  roiRelay: document.getElementById("roi-relay"),
  editToggle: document.getElementById("edit-toggle"),
  seedPill: document.getElementById("seed-pill"),
  seedHint: document.getElementById("seed-hint"),
  pattern: document.getElementById("pattern"),
  panelTitle: document.getElementById("panel-title"),
  panelToggle: document.getElementById("panel-toggle"),
  drawer: document.getElementById("drawer"),
  drawerTab: document.getElementById("drawer-tab"),
  drawerClose: document.getElementById("drawer-close"),
  agent: document.getElementById("agent"),
  agentLog: document.getElementById("agent-log"),
  agentApprove: document.getElementById("agent-approve"),
  agentVerdict: document.getElementById("agent-verdict"),
  copy: document.getElementById("copy"),
  meshInfo: document.getElementById("mesh-info"),
  legend: document.getElementById("legend"),
  status: document.getElementById("status"),
};

// ----------------------------------------------------------------------------
// three.js scene
// ----------------------------------------------------------------------------
const canvas = document.getElementById("canvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161a);

const camera = new THREE.PerspectiveCamera(50, 1, 0.001, 1000);
camera.position.set(0, 0, 3);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 0.9);
key.position.set(2, 3, 4);
scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.4);
fill.position.set(-3, -1, -2);
scene.add(fill);

// Groups that get rebuilt each run
let meshObj = null;     // the surface
const overlay = new THREE.Group();   // stitch rows, saddles, markers
scene.add(overlay);
const optOverlay = new THREE.Group();  // optimizer culprits (uncovered / floating)
scene.add(optOverlay);
const roiOverlay = new THREE.Group();  // brushed region-of-interest (green)
scene.add(roiOverlay);
let seedMarker = null;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

// Fat-line materials need their screen resolution kept in sync.
const lineMaterials = [];

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    for (const m of lineMaterials) m.resolution.set(w, h);
  }
}
function animate() {
  requestAnimationFrame(animate);
  resize();
  controls.update();
  renderer.render(scene, camera);
}
animate();

// ----------------------------------------------------------------------------
// Viridis-ish colour ramp for the geodesic field
// ----------------------------------------------------------------------------
const VIRIDIS = [
  [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530],
  [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
  [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
  [0.741, 0.873, 0.150], [0.993, 0.906, 0.144],
];
function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (VIRIDIS.length - 1);
  const i = Math.floor(x), frac = x - i;
  const a = VIRIDIS[i], b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
  return [a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac, a[2] + (b[2] - a[2]) * frac];
}

// ----------------------------------------------------------------------------
// Mesh loading / rendering
// ----------------------------------------------------------------------------
function buildMeshGeometry() {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(state.vertices, 3));
  geo.setIndex(new THREE.BufferAttribute(state.faces, 1));
  geo.computeVertexNormals();
  // neutral grey vertex colours until a field is computed
  const colors = new Float32Array(state.nVerts * 3).fill(0.55);
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return geo;
}

function showMesh() {
  if (meshObj) { scene.remove(meshObj); meshObj.geometry.dispose(); meshObj.material.dispose(); }
  clearOverlay();
  const geo = buildMeshGeometry();
  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.85, metalness: 0.0,
    flatShading: false, side: THREE.DoubleSide,
  });
  meshObj = new THREE.Mesh(geo, mat);
  scene.add(meshObj);

  // frame the camera
  state.center.fromArray(state.centerArr);
  controls.target.copy(state.center);
  const r = state.radius;
  camera.position.set(state.center.x, state.center.y, state.center.z + r * 2.6);
  camera.near = r * 0.01; camera.far = r * 100; camera.updateProjectionMatrix();
  controls.update();
}

function clearOverlay() {
  while (overlay.children.length) {
    const c = overlay.children.pop();
    c.geometry?.dispose();
    c.material?.dispose();
  }
  seedMarker = null;
}

function vertexPos(idx) {
  return new THREE.Vector3(
    state.vertices[idx * 3], state.vertices[idx * 3 + 1], state.vertices[idx * 3 + 2]
  );
}

function markerSize() { return state.radius * 0.025; }

function placeSeedMarker(idx) {
  if (seedMarker) { overlay.remove(seedMarker); seedMarker.geometry.dispose(); seedMarker.material.dispose(); }
  const g = new THREE.SphereGeometry(markerSize() * 1.4, 16, 16);
  const m = new THREE.MeshBasicMaterial({ color: 0xff2d78 });
  seedMarker = new THREE.Mesh(g, m);
  seedMarker.position.copy(vertexPos(idx));
  overlay.add(seedMarker);
}

// ----------------------------------------------------------------------------
// Pick a vertex by clicking the mesh
// ----------------------------------------------------------------------------
let dragged = false;
canvas.addEventListener("pointerdown", () => { dragged = false; });
canvas.addEventListener("pointermove", () => { dragged = true; });
canvas.addEventListener("pointerup", (ev) => {
  if (dragged || !meshObj || els.autoSeed.checked || els.roiBrush.checked) return;  // auto/brush mode
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(meshObj, false);
  if (!hits.length) return;
  const hit = hits[0];
  // nearest of the face's three vertices to the hit point
  const f = hit.face;
  let best = f.a, bestD = Infinity;
  for (const vi of [f.a, f.b, f.c]) {
    const d = vertexPos(vi).distanceToSquared(hit.point);
    if (d < bestD) { bestD = d; best = vi; }
  }
  state.seed = best;
  placeSeedMarker(best);
  refreshSeedMode();
});

// ----------------------------------------------------------------------------
// ROI brush — paint trouble spots; the auto-optimizer weights covering them.
// Mirrors the editor's BVH-accelerated radius pick (edit.js installs the
// accelerated raycast prototype at startup), but paints continuously on drag.
// ----------------------------------------------------------------------------
function roiBrushRadius2() {
  const r = state.radius * 0.09;
  return r * r;
}

function paintRoiAt(clientX, clientY) {
  if (!meshObj) return;
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(meshObj, false);
  if (!hits.length) return;
  const hit = hits[0].point;
  const r2 = roiBrushRadius2();
  const before = state.roiVertices.size;
  const v = new THREE.Vector3();
  for (let i = 0; i < state.nVerts; i++) {
    v.set(state.vertices[i * 3], state.vertices[i * 3 + 1], state.vertices[i * 3 + 2]);
    if (v.distanceToSquared(hit) <= r2) state.roiVertices.add(i);
  }
  if (state.roiVertices.size !== before) refreshRoiOverlay();
}

function refreshRoiOverlay() {
  while (roiOverlay.children.length) {
    const c = roiOverlay.children.pop();
    c.geometry?.dispose();
    c.material?.dispose();
  }
  els.roiClear.disabled = !state.roiVertices.size;
  refreshRoiRelay();
  if (!state.roiVertices.size || !state.vertices) return;
  const pos = [];
  for (const i of state.roiVertices)
    pos.push(state.vertices[i * 3], state.vertices[i * 3 + 1], state.vertices[i * 3 + 2]);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  roiOverlay.add(new THREE.Points(g, new THREE.PointsMaterial({
    color: 0x46e08a, size: markerSize() * 1.8, sizeAttenuation: true,
  })));
}

let painting = false;
canvas.addEventListener("pointerdown", (ev) => {
  if (!els.roiBrush.checked || !meshObj || ev.button !== 0) return;
  painting = true;
  paintRoiAt(ev.clientX, ev.clientY);
});
canvas.addEventListener("pointermove", (ev) => {
  if (painting) paintRoiAt(ev.clientX, ev.clientY);
});
window.addEventListener("pointerup", () => { painting = false; });

els.roiBrush.addEventListener("change", () => {
  // While brushing, drags paint instead of orbiting the camera.
  controls.enabled = !els.roiBrush.checked;
  els.roiClear.disabled = !state.roiVertices.size;
  setStatus(els.roiBrush.checked
    ? "Brush on — drag to paint trouble spots; toggle off to orbit."
    : "");
});

els.roiClear.addEventListener("click", () => {
  state.roiVertices.clear();
  refreshRoiOverlay();
  els.roiClear.disabled = true;
});

// The local re-lay needs a brushed region AND a seed (manually picked or chosen
// by the auto-optimizer). It densifies the brushed segment(s) and gives a merged
// protrusion its own spiral — the local levers that actually add stitches there.
function refreshRoiRelay() {
  els.roiRelay.disabled =
    !(state.meshId && state.roiVertices.size && state.seed != null);
}

els.roiRelay.addEventListener("click", runRelayout);

async function runRelayout() {
  if (!state.meshId || state.seed == null || !state.roiVertices.size) return;
  els.roiRelay.disabled = true;
  els.panelToggle.style.display = "none";
  showPanel("pattern");
  setStatus("Re-laying brushed area (finer rows + its own spiral)…");
  try {
    const res = await fetch("/api/relayout", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: state.meshId, seed: state.seed,
        stitch_width: Number(els.sw.value),
        roi_vertices: [...state.roiVertices],
        densify: true, force_segment: true,
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    renderResult(data);
    showOptimizeCulprits(data.metrics);
    placeSeedMarker(state.seed);
    els.pattern.textContent = data.pattern;
    els.copy.style.display = "inline-block";
    setDrawer(true);
    const b = data.roi_before || {}, a = data.roi_after || {};
    const pct = (x) => (x == null ? "?" : (x * 100).toFixed(1) + "%");
    setStatus(`Brushed area: ${b.roi_stitches ?? "?"} → ${a.roi_stitches ?? "?"} stitches · `
      + `coverage ${pct(b.roi_coverage)} → ${pct(a.roi_coverage)} · `
      + `segments ${b.n_segments ?? "?"} → ${a.n_segments ?? "?"}.`);
  } catch (e) {
    setStatus("Re-lay failed: " + e.message);
  } finally {
    refreshRoiRelay();
  }
}

// ----------------------------------------------------------------------------
// API
// ----------------------------------------------------------------------------
function setStatus(msg) { els.status.textContent = msg || ""; }

// Left slide-out drawer holding the pattern / agent panel.
function setDrawer(open) {
  els.drawer.classList.toggle("open", open);
  els.drawerTab.firstChild.nodeValue = open ? "◂" : "▸";
}

// Flip the drawer between the generated pattern and the agent run log.
function showPanel(which) {
  const pat = which === "pattern";
  els.pattern.style.display = pat ? "block" : "none";
  els.agent.style.display = pat ? "none" : "block";
  els.panelTitle.textContent = pat ? "Pattern" : "Auto-seed log";
  els.copy.style.display = (pat && els.pattern.textContent.trim()) ? "inline-block" : "none";
  els.panelToggle.textContent = pat ? "Log" : "Pattern";
}
els.panelToggle.addEventListener("click", () =>
  showPanel(els.pattern.style.display === "none" ? "pattern" : "agent"));
els.drawerTab.addEventListener("click", () =>
  setDrawer(!els.drawer.classList.contains("open")));
els.drawerClose.addEventListener("click", () => setDrawer(false));

// Load a mesh payload (from /api/upload or an agent-produced simplified mesh)
// into the viewer and reset the per-mesh UI state.
function loadMeshPayload(data) {
  state.meshId = data.id;
  state.currentPayload = data;
  state.vertices = new Float32Array(data.vertices);
  state.faces = new Uint32Array(data.faces);
  state.nVerts = data.n_verts;
  state.centerArr = data.center;
  state.radius = data.radius;
  state.seed = null;
  showMesh();
  els.meshInfo.innerHTML = `<b>${data.n_verts}</b> verts · <b>${data.n_faces}</b> faces`;
  els.legend.style.display = "none";
  els.copy.style.display = "none";
  els.assess.disabled = false;
  els.autoSeed.disabled = false;
  els.autoSeed.parentElement.classList.remove("disabled");
  els.roiBrush.disabled = false;
  els.roiBrush.parentElement.classList.remove("disabled");
  els.editToggle.disabled = false;
  state.roiVertices.clear();
  refreshRoiOverlay();
  controls.enabled = !els.roiBrush.checked;
  clearOptimizeOverlay();
  refreshSeedMode();
  editor?.onMeshLoaded();
  // The drawer holds the generated pattern — keep it closed until there is one.
  setDrawer(false);
  els.panelToggle.style.display = "none";
  showPanel("pattern");
  els.copy.style.display = "none";   // no real pattern yet
}

els.file.addEventListener("change", async () => {
  const file = els.file.files[0];
  if (!file) return;
  setStatus("Uploading & loading mesh…");
  els.run.disabled = true;
  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    loadMeshPayload(await res.json());
    els.pattern.textContent = "Pick a seed vertex on the mesh, then generate.";
    setStatus("Click the mesh to choose where to start crocheting.");
  } catch (e) {
    setStatus("Error: " + e.message);
  }
});

els.sw.addEventListener("input", () => {
  els.swVal.textContent = Number(els.sw.value).toFixed(3);
});

// Reflect the current seed mode (auto vs manual) in the pill, hint and the
// Generate button — and gate Generate accordingly.
function refreshSeedMode() {
  const auto = els.autoSeed.checked;
  const hasMesh = !!state.meshId;
  if (auto) {
    els.seedPill.textContent =
      state.seed == null ? "auto (Claude picks)" : `auto → vertex #${state.seed}`;
    els.seedHint.textContent =
      "Generate runs Claude's optimizer — it picks the seed and stitch width.";
    els.run.textContent = "Generate (auto-seed) 🎯";
    els.run.disabled = !hasMesh;
  } else {
    els.seedPill.textContent =
      state.seed == null ? "click the mesh to pick" : `vertex #${state.seed}`;
    els.seedHint.textContent =
      "Click a point on the mesh — crocheting starts here (the magic circle).";
    els.run.textContent = "Generate pattern";
    els.run.disabled = !(hasMesh && state.seed != null);
  }
  // Manual "Auto" width needs a chosen seed; in auto mode the optimizer does it.
  els.autoWidth.disabled = auto || !(hasMesh && state.seed != null);
  refreshRoiRelay();
}

// Set the stitch-width slider + label (clamped to the slider's range).
function setStitchWidth(w) {
  const min = Number(els.sw.min), max = Number(els.sw.max);
  const v = Math.max(min, Math.min(max, Number(w)));
  els.sw.value = v;
  els.swVal.textContent = Number(els.sw.value).toFixed(3);
}

els.autoWidth.addEventListener("click", async () => {
  if (state.seed == null || !state.meshId) return;
  els.autoWidth.disabled = true;
  const prev = els.autoWidth.textContent;
  els.autoWidth.textContent = "…";
  try {
    const res = await fetch("/api/suggest_width", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: state.meshId, seed: state.seed }),
    });
    const info = await res.json();
    if (!res.ok || !info.ok) throw new Error(info.detail || info.reason || "no estimate");
    setStitchWidth(info.stitch_width);
    setStatus(`Auto width ${info.stitch_width} — narrowest feature girth `
      + `${info.min_segment_girth}${info.limited_by_narrowness ? " (limited by a narrow feature)" : ""}.`);
  } catch (e) {
    setStatus("Auto width failed: " + e.message);
  } finally {
    els.autoWidth.textContent = prev;
    els.autoWidth.disabled = false;
  }
});

els.autoSeed.addEventListener("change", () => {
  if (els.autoSeed.checked && seedMarker) {
    overlay.remove(seedMarker);
    seedMarker.geometry.dispose();
    seedMarker.material.dispose();
    seedMarker = null;
  }
  clearOptimizeOverlay();
  refreshSeedMode();
});

els.run.addEventListener("click", () => {
  if (!state.meshId) return;
  if (els.autoSeed.checked) runAuto();
  else if (state.seed != null) runManual();
});

async function runManual() {
  els.run.disabled = true;
  els.panelToggle.style.display = "none";
  showPanel("pattern");
  clearOptimizeOverlay();
  setStatus("Generating pattern… (geodesics, segmentation, sampling)");
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: state.meshId, seed: state.seed,
        stitch_width: Number(els.sw.value),
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    renderResult(data);
    els.pattern.textContent = data.pattern;
    els.copy.style.display = "inline-block";
    setDrawer(true);
    const nRows = data.segments.reduce((s, seg) => s + seg.rows.length, 0);
    setStatus(`Done — ${data.segments.length} segment(s), ${nRows} rows, ${state.lastStitchCount} stitches.`);
  } catch (e) {
    setStatus("Error: " + e.message);
  } finally {
    els.run.disabled = false;
  }
}

els.copy.addEventListener("click", () => {
  navigator.clipboard.writeText(els.pattern.textContent);
  els.copy.textContent = "Copied!";
  setTimeout(() => (els.copy.textContent = "Copy"), 1200);
});

// ----------------------------------------------------------------------------
// Phase-3 editor instance (shares this scene)
// ----------------------------------------------------------------------------
editor = initEditor({
  THREE, scene, camera, renderer, controls, canvas,
  getMesh: () => meshObj,
  getState: () => state,
  loadMeshPayload,
  setStatus,
  markerSize,
});

els.editToggle.addEventListener("click", () => {
  const on = editor.toggle();
  els.editToggle.classList.toggle("active", on);
  els.editToggle.textContent = on ? "Close editor" : "Edit mesh ✎";
});

// ----------------------------------------------------------------------------
// Crochetability agent (Phase 2): stream the agent, gate each simplification
// ----------------------------------------------------------------------------
let assessSession = null;
let assessSource = null;

function logLine(html, cls) {
  const div = document.createElement("div");
  if (cls) div.className = cls;
  div.innerHTML = html;
  els.agentLog.appendChild(div);
  els.agent.scrollTop = els.agent.scrollHeight;
}

function logText(text, cls) {
  // append into a trailing same-class node so streamed deltas stay on one line
  let last = els.agentLog.lastElementChild;
  if (!last || last.dataset.kind !== (cls || "text")) {
    last = document.createElement("div");
    last.dataset.kind = cls || "text";
    if (cls) last.className = cls;
    els.agentLog.appendChild(last);
  }
  last.textContent += text;
  els.agent.scrollTop = els.agent.scrollHeight;
}

const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

els.assess.addEventListener("click", async () => {
  if (!state.meshId) return;
  els.assess.disabled = true;
  els.panelTitle.textContent = "Crochetability";
  els.pattern.style.display = "none";
  els.agent.style.display = "block";
  setDrawer(true);
  els.agentLog.innerHTML = "";
  els.agentApprove.innerHTML = "";
  els.agentVerdict.innerHTML = "";
  setStatus("Assessing crochetability with Claude Opus 4.8…");
  try {
    const res = await fetch("/api/assess", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: state.meshId }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    assessSession = (await res.json()).session_id;
    startAssessStream(assessSession);
  } catch (e) {
    setStatus("Error: " + e.message);
    els.assess.disabled = false;
  }
});

function startAssessStream(sid) {
  assessSource = new EventSource(`/api/assess/${sid}/stream`);
  assessSource.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    switch (ev.type) {
      case "thinking": logText(ev.text, "think"); break;
      case "text": logText(ev.text, "text"); break;
      case "tool_call":
        logLine(`→ <span class="tool">${esc(ev.name)}</span>` +
          (ev.name === "apply_simplification" ? ` <em>${esc(ev.input.technique)}</em>` : ""));
        break;
      case "tool_result":
        logLine(`&nbsp;&nbsp;✓ ${esc(ev.name)}${ev.is_error ? ' <span class="err">[error]</span>' : ""}`);
        break;
      case "applied":
        logLine(`&nbsp;&nbsp;✎ applied ${esc(ev.report.technique)}`, "applied");
        break;
      case "approval_request": renderApproval(ev.proposal); break;
      case "verdict": renderVerdict(ev.verdict); break;
      case "simplified": addLoadSimplified(ev.mesh); break;
      case "error": logLine(`⚠ ${esc(ev.message)}`, "err"); break;
      case "end":
        assessSource.close();
        els.assess.disabled = false;
        setStatus("Assessment complete.");
        break;
    }
  };
  assessSource.onerror = () => {
    assessSource.close();
    els.assess.disabled = false;
    setStatus("Assessment stream closed.");
  };
}

function renderApproval(p) {
  els.agentApprove.innerHTML = "";
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML =
    `<h4>Proposed: ${esc(p.technique)}</h4>` +
    `<div class="muted">${esc(p.rationale || "")}</div>` +
    (Object.keys(p.params || {}).length
      ? `<div class="muted">params: ${esc(JSON.stringify(p.params))}</div>` : "") +
    `<div class="actions">
       <button class="apply-yes">Approve &amp; apply</button>
       <button class="btn-deny apply-no">Deny</button>
     </div>`;
  els.agentApprove.appendChild(card);
  const decide = async (approve) => {
    card.querySelectorAll("button").forEach((b) => (b.disabled = true));
    await fetch(`/api/assess/${assessSession}/decision`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approve, note: "" }),
    });
    els.agentApprove.innerHTML = "";
  };
  card.querySelector(".apply-yes").addEventListener("click", () => decide(true));
  card.querySelector(".apply-no").addEventListener("click", () => decide(false));
}

function renderVerdict(v) {
  const cls = v.crochetable === "yes" ? "v-yes"
    : v.crochetable === "no" ? "v-no" : "v-mod";
  const list = (arr) => (arr && arr.length)
    ? "<ul>" + arr.map((x) => `<li>${esc(x)}</li>`).join("") + "</ul>" : "";
  els.agentVerdict.innerHTML =
    `<div class="card">
       <h4><span class="verdict-badge ${cls}">${esc(v.crochetable.replace("_", " "))}</span>
           <span class="muted">· confidence: ${esc(v.confidence)}</span></h4>
       <div>${esc(v.summary || "")}</div>
       ${v.reasons && v.reasons.length ? "<div class='muted' style='margin-top:8px'>Reasons:</div>" + list(v.reasons) : ""}
       ${v.applied_steps && v.applied_steps.length ? "<div class='muted' style='margin-top:8px'>Applied:</div>" + list(v.applied_steps) : ""}
       ${v.recommended_manual_steps && v.recommended_manual_steps.length ? "<div class='muted' style='margin-top:8px'>Recommended manual steps:</div>" + list(v.recommended_manual_steps) : ""}
     </div>`;
}

// ----------------------------------------------------------------------------
// Closed-loop seed optimization: Claude tries seeds, scores each for surface
// coverage and floating stitches, then we render the winning pattern.
// ----------------------------------------------------------------------------
let optimizeSource = null;

function clearOptimizeOverlay() {
  while (optOverlay.children.length) {
    const c = optOverlay.children.pop();
    c.geometry?.dispose();
    c.material?.dispose();
  }
}

function showOptimizeCulprits(metrics) {
  clearOptimizeOverlay();
  if (!metrics) return;
  // uncovered face centroids → red dots
  const uc = metrics.uncovered_centroids || [];
  if (uc.length) {
    const pos = [];
    for (const p of uc) pos.push(p[0], p[1], p[2]);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    optOverlay.add(new THREE.Points(g, new THREE.PointsMaterial({
      color: 0xff3b3b, size: markerSize() * 2.4, sizeAttenuation: true,
    })));
  }
  // floating-edge midpoints → yellow dots
  const fl = metrics.floating_edges || [];
  if (fl.length) {
    const pos = [];
    for (const e of fl) pos.push(e.midpoint[0], e.midpoint[1], e.midpoint[2]);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    optOverlay.add(new THREE.Points(g, new THREE.PointsMaterial({
      color: 0xffd23b, size: markerSize() * 2.4, sizeAttenuation: true,
    })));
  }
}

function renderSeedChoice(choice, metrics) {
  const m = metrics || {};
  const pct = (x) => (x == null ? "?" : (x * 100).toFixed(1) + "%");
  els.agentVerdict.innerHTML =
    `<div class="card">
       <h4><span class="verdict-badge v-yes">best seed #${esc(choice.seed)}</span>
           <span class="muted">· score ${esc((choice.score ?? 0).toFixed(3))}</span></h4>
       <div>${esc(choice.summary || "")}</div>
       <div class="muted" style="margin-top:8px">
         coverage ${pct(m.coverage)} · floating stitches ${esc(m.n_floating ?? "?")}
         · thin segments ${esc(m.n_thin_segments ?? "?")}${
           m.roi_coverage != null ? ` · brushed area ${pct(m.roi_coverage)}` : ""}</div>
       ${choice.reasons && choice.reasons.length
         ? "<div class='muted' style='margin-top:8px'>Reasons:</div><ul>"
           + choice.reasons.map((r) => `<li>${esc(r)}</li>`).join("") + "</ul>"
         : ""}
     </div>`;
}

async function runAuto() {
  if (!state.meshId) return;
  els.run.disabled = true;
  els.assess.disabled = true;
  els.panelTitle.textContent = "Auto-seed";
  els.pattern.style.display = "none";
  els.agent.style.display = "block";
  els.panelToggle.style.display = "none";
  setDrawer(true);
  els.agentLog.innerHTML = "";
  els.agentApprove.innerHTML = "";
  els.agentVerdict.innerHTML = "";
  clearOptimizeOverlay();
  setStatus("Optimizing the seed with Claude Opus 4.8…");
  try {
    const res = await fetch("/api/optimize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: state.meshId, stitch_width: Number(els.sw.value),
        roi_vertices: [...state.roiVertices],
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    startOptimizeStream((await res.json()).session_id);
  } catch (e) {
    setStatus("Error: " + e.message);
    els.assess.disabled = false;
    refreshSeedMode();
  }
}

function startOptimizeStream(sid) {
  optimizeSource = new EventSource(`/api/optimize/${sid}/stream`);
  optimizeSource.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    switch (ev.type) {
      case "thinking": logText(ev.text, "think"); break;
      case "text": logText(ev.text, "text"); break;
      case "tool_call":
        logLine(`→ <span class="tool">${esc(ev.name)}</span>` +
          (ev.name === "evaluate_seed" ? ` <em>seed ${esc(ev.input.seed)}</em>` : ""));
        break;
      case "tool_result":
        logLine(`&nbsp;&nbsp;✓ ${esc(ev.name)}${ev.is_error ? ' <span class="err">[error]</span>' : ""}`);
        break;
      case "seed_eval": {
        const s = ev.summary || {};
        if (s.ran) {
          const roi = s.roi_coverage != null
            ? `, brushed ${esc((s.roi_coverage * 100).toFixed(1))}%` : "";
          logLine(`&nbsp;&nbsp;<span class="seed-score">seed ${esc(s.seed)} @ w=${esc(s.stitch_width)} → score `
            + `<b>${esc(s.score)}</b>, coverage ${esc((s.coverage * 100).toFixed(1))}%, `
            + `floating ${esc(s.n_floating)}, thin ${esc(s.n_thin_segments)}${roi}</span>`, "applied");
        } else {
          logLine(`&nbsp;&nbsp;<span class="err">seed ${esc(s.seed)} → failed`
            + (s.error ? `: ${esc(s.error)}` : "") + `</span>`);
        }
        break;
      }
      case "seed_choice":
        renderSeedChoice(ev.choice, ev.metrics);
        applySeedChoice(ev.choice, ev.metrics);
        break;
      case "error": logLine(`⚠ ${esc(ev.message)}`, "err"); break;
      case "end":
        optimizeSource.close();
        els.assess.disabled = false;
        refreshSeedMode();
        break;
    }
  };
  optimizeSource.onerror = () => {
    optimizeSource.close();
    els.assess.disabled = false;
    refreshSeedMode();
    setStatus("Optimization stream closed.");
  };
}

// Adopt the winning seed + width, generate its pattern, and overlay the culprits.
async function applySeedChoice(choice, metrics) {
  state.seed = Number(choice.seed);
  refreshRoiRelay();
  const width = choice.stitch_width ? Number(choice.stitch_width) : Number(els.sw.value);
  if (choice.stitch_width) setStitchWidth(width);   // reflect the chosen width
  setStatus(`Best seed #${state.seed} @ width ${width} — generating its pattern…`);
  try {
    const res = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: state.meshId, seed: state.seed, stitch_width: width,
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    renderResult(data);
    showOptimizeCulprits(metrics);
    placeSeedMarker(state.seed);
    els.pattern.textContent = data.pattern;
    // The agent log streamed live; now surface the final pattern (the deliverable).
    // The Log ⇄ Pattern toggle lets the user flip back to review the run.
    showPanel("pattern");
    els.panelToggle.style.display = "inline-block";
    const nRows = data.segments.reduce((s, seg) => s + seg.rows.length, 0);
    setStatus(`Auto-seed #${state.seed}: ${data.segments.length} segment(s), `
      + `${nRows} rows. Red = uncovered, yellow = floating stitches.`);
  } catch (e) {
    setStatus("Error generating pattern: " + e.message);
  }
}

function addLoadSimplified(mesh) {
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `<h4>Simplified mesh ready</h4>
    <div class="muted">${mesh.n_verts} verts · ${mesh.n_faces} faces</div>
    <div class="actions"><button class="load-simpl">Load into viewer</button></div>`;
  els.agentVerdict.appendChild(card);
  card.querySelector(".load-simpl").addEventListener("click", () => {
    loadMeshPayload(mesh);
    els.agent.style.display = "none";
    els.pattern.style.display = "block";
    els.panelTitle.textContent = "Pattern";
    els.pattern.textContent = "Simplified mesh loaded. Pick a seed and generate.";
    setStatus("Loaded simplified mesh — click to pick a seed.");
  });
}

// ----------------------------------------------------------------------------
// Render pipeline result: colour by field, mark saddles, overlay stitch rows
// ----------------------------------------------------------------------------
function renderResult(data) {
  clearOverlay();
  placeSeedMarker(state.seed);

  // 1. colour the surface by geodesic field
  const field = data.field;
  const fmax = data.field_max || 1;
  const colors = meshObj.geometry.getAttribute("color");
  const MIX = 0.58;  // lerp toward white so the graph overlay stays legible
  for (let i = 0; i < field.length; i++) {
    const c = viridis(field[i] / fmax);
    colors.setXYZ(i,
      c[0] + (1 - c[0]) * MIX,
      c[1] + (1 - c[1]) * MIX,
      c[2] + (1 - c[2]) * MIX);
  }
  colors.needsUpdate = true;
  els.legend.style.display = "flex";

  // 2. saddle markers (orange) + tip marker (cyan)
  const saddleGeo = new THREE.SphereGeometry(markerSize(), 12, 12);
  const saddleMat = new THREE.MeshBasicMaterial({ color: 0xffa733 });
  for (const s of data.saddles) {
    const m = new THREE.Mesh(saddleGeo, saddleMat);
    m.position.copy(vertexPos(s));
    overlay.add(m);
  }
  const tip = new THREE.Mesh(
    new THREE.SphereGeometry(markerSize() * 1.3, 16, 16),
    new THREE.MeshBasicMaterial({ color: 0x51d1c8 })
  );
  tip.position.copy(vertexPos(data.tip));
  overlay.add(tip);

  // ---- the crochet graph: red row edges, blue column edges, stitch dots ----
  // Lift everything slightly off the surface (outward from the mesh centre)
  // so the graph sits on top of the mesh instead of z-fighting / hiding in it.
  const lift = state.radius * 0.012;
  const off = (p) => {
    const dx = p[0] - state.center.x, dy = p[1] - state.center.y, dz = p[2] - state.center.z;
    const len = Math.hypot(dx, dy, dz) || 1;
    const s = lift / len;
    return [p[0] + dx * s, p[1] + dy * s, p[2] + dz * s];
  };

  const rowSeg = [];     // red: consecutive stitches within a row (closed loop)
  const colSeg = [];     // blue: DTW coupling between consecutive rows
  const positions = [];  // stitch vertices
  let nStitches = 0;

  for (const seg of data.segments) {
    const rows = seg.rows.map((row) => row.map(off));   // pre-lift the points

    for (const row of rows)
      for (const p of row) { positions.push(p[0], p[1], p[2]); nStitches++; }

    // red row edges (closed loops around the surface)
    for (const row of rows) {
      if (row.length < 2) continue;
      for (let j = 0; j < row.length; j++) {
        const a = row[j], b = row[(j + 1) % row.length];
        rowSeg.push(a[0], a[1], a[2], b[0], b[1], b[2]);
      }
    }

    // blue column edges from the DTW coupling
    const ce = seg.col_edges || [];
    for (let i = 0; i < ce.length; i++) {
      const ra = rows[i], rb = rows[i + 1];
      if (!ra || !rb) continue;
      for (const [j, k] of ce[i]) {
        if (j >= ra.length || k >= rb.length) continue;
        const a = ra[j], b = rb[k];
        colSeg.push(a[0], a[1], a[2], b[0], b[1], b[2]);
      }
    }
  }

  // Fat lines: linewidth is in *world units*, so edges thicken as you zoom in.
  const mkFatSegments = (arr, color, width, opacity) => {
    const g = new LineSegmentsGeometry();
    g.setPositions(arr);
    const m = new LineMaterial({
      color, linewidth: width, worldUnits: true,
      transparent: opacity < 1, opacity,
      resolution: new THREE.Vector2(canvas.clientWidth, canvas.clientHeight),
    });
    lineMaterials.push(m);
    const seg = new LineSegments2(g, m);
    seg.computeLineDistances();
    return seg;
  };
  const wRow = state.radius * 0.012;
  const wCol = state.radius * 0.007;
  overlay.add(mkFatSegments(colSeg, 0x3b4bd8, wCol, 0.85));  // column edges (blue)
  overlay.add(mkFatSegments(rowSeg, 0xe6271f, wRow, 1.0));   // row edges (red)

  // stitch vertices as dark dots (grow with zoom via size attenuation)
  const pgeo = new THREE.BufferGeometry();
  pgeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  overlay.add(new THREE.Points(pgeo, new THREE.PointsMaterial({
    color: 0x101216, size: markerSize() * 2.0, sizeAttenuation: true,
  })));
  state.lastStitchCount = nStitches;

  document.getElementById("stitch-legend").style.display = "flex";
}
