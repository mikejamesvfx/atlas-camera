/**
 * Atlas Viewport — ComfyUI frontend extension
 *
 * Embeds a Three.js 3D scene inside the AtlasBlockoutViewport node.
 * On node execution the recovered camera is fetched from /atlas/camera_data/{nodeId}
 * and applied to the Three.js camera so the scene is pre-aligned to the source photo.
 *
 * The viewport renders the solve's Python-derived geometry (relief meshes,
 * fitted primitives, patch/clean-plate sources) with 📽 camera projection;
 * "Render Proxy Passes" produces shaded / depth / normal / mask images that
 * are base64-encoded into the client_data STRING widget and sent to Python.
 * (The old in-browser primitive/OBJ-proxy placement toolbar was removed
 * 2026-07-09 — see the note above the projection-material section.)
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ---------------------------------------------------------------------------
// Three.js — vendored local bundle (lib/atlas-three.bundle.js): three core
// r185 + OBJLoader + FBXLoader in one self-contained ESM file, built by
// `npm run build:comfy-three` in ui/ (entry: ui/bundle/atlas-three-entry.js)
// and committed so users never need npm or a network connection.
//
// This replaced a CDN-based chain that was quietly broken: ComfyUI does NOT
// expose its internal three build at ../../lib/three.module.js (it's a hashed
// Vite chunk with no import surface), so the old first-choice import always
// failed over to unpkg; and the unpkg examples/jsm loaders use a bare
// `import "three"` specifier that never resolves without an import map, so
// OBJLoader/FBXLoader silently never loaded at all (verified live 2026-07-07).
// ---------------------------------------------------------------------------
let THREE;
let OBJLoader;
let FBXLoader;

async function loadThree() {
  if (THREE) return;
  try {
    // The bundle re-exports the full three namespace at the top level, so the
    // module object itself serves as THREE; the loaders ride along as extra
    // named exports (they aren't part of the core namespace — no collisions).
    const mod = await import("./lib/atlas-three.bundle.js");
    THREE = mod;
    OBJLoader = mod.OBJLoader;
    FBXLoader = mod.FBXLoader;
  } catch (e) {
    console.error("[AtlasBlockout] Failed to load Three.js bundle:", e);
  }
}

// ---------------------------------------------------------------------------
// Scale-reference proxy meshes (examples/models/*.obj, served by Python).
// Files are authored in centimetres, so we scale by 0.01 into the metric world
// that the recovered camera + ground plane live in — a correctly-sized human or
// car is the fastest visual check that the solve and camera height are right.
// ---------------------------------------------------------------------------
const ATLAS_VIEWPORT_PREVIEW_MAX_LONG_EDGE = 1280;

// Default node size for FRESHLY ADDED viewport nodes only (saved workflows
// keep whatever size they stored — see the onConfigure tracker in
// nodeCreated). LiteGraph's own computed default is a cramped ~270×438;
// this is double the 460px the example workflows historically shipped at,
// so the 3D preview is usable without an immediate manual resize.
// Display-only: render resolution is still governed solely by the
// `resolution` widget (a 768 render shown at 960 wide is a mild CSS upscale —
// bump `resolution` for a sharper image at this size).
const ATLAS_VIEWPORT_DEFAULT_WIDTH = 960;
const ATLAS_VIEWPORT_DEFAULT_HEIGHT = 720;

// Pin a DOM widget's `width` property to permanently-undefined. ComfyUI's
// per-frame DOM-widget layout (DomWidgets.vue updateWidgets, read from the
// frontend 1.45.20 sourcemaps) sizes the widget's host element as
// `widget.width ?? node.width` — for a full-width widget like ours the
// correct steady state is `undefined` (fall through to the live node width).
// Observed live (2026-07-07): something writes a one-shot stale pixel width
// onto the widget object (~394 = this node type's pre-configure computed
// width), after which the 3D canvas box permanently collapses to that width
// on the next interaction that re-syncs widget style (mouseup/click →
// DomWidget.vue's selectOn:['focus','click'] listener → selectNode →
// style recompute) while the node itself stays big. The writer was never
// caught in the act (it is sporadic — likely a frontend transient or another
// extension), so instead of chasing it, make the property unwritable-by-
// anyone: reads always yield undefined, writes are swallowed. LiteGraph's
// own uses (`widget.width || nodeWidth` in hit-testing/drawing) fall through
// to node width identically, so this is behavior-neutral everywhere else.
function pinDomWidgetFullWidth(domWidget) {
  try {
    Object.defineProperty(domWidget, "width", {
      configurable: true,
      get() { return undefined; },
      set() {},
    });
  } catch (e) {
    console.warn("[AtlasBlockout] could not pin DOM widget width:", e);
  }
}

function atlasFitLongEdge(width, height, maxLongEdge) {
  const w = Math.max(16, Math.round(width || maxLongEdge || 1280));
  const h = Math.max(16, Math.round(height || maxLongEdge || 720));
  const longEdge = Math.max(w, h);
  const limit = Math.max(16, Math.round(maxLongEdge || longEdge));
  if (longEdge <= limit) return { width: w, height: h };
  const scale = limit / longEdge;
  return {
    width: Math.max(16, Math.round(w * scale)),
    height: Math.max(16, Math.round(h * scale)),
  };
}

function atlasViewportPreviewSize(width, height) {
  return atlasFitLongEdge(width, height, ATLAS_VIEWPORT_PREVIEW_MAX_LONG_EDGE);
}

function atlasViewportSizeTraceEnabled() {
  try {
    return typeof localStorage !== "undefined" &&
      localStorage.getItem("ATLAS_VIEWPORT_SIZE_TRACE") === "1";
  } catch (_) {
    return false;
  }
}

function installViewportSizeTrace(node, domWidget, element) {
  if (!atlasViewportSizeTraceEnabled() || node._atlasSizeTraceInstalled) return;
  node._atlasSizeTraceInstalled = true;

  const label = `[AtlasBlockout size trace node ${node.id ?? "?"}]`;
  const rawToProxy = new WeakMap();
  const proxies = new WeakSet();
  const observers = [];

  function canvasState() {
    const resizingNode = app?.canvas?.resizing_node;
    return {
      node_size: Array.isArray(node.size) ? [node.size[0], node.size[1]] : node.size,
      resizing_node: resizingNode?.id ?? null,
      resizing_this_node: resizingNode === node,
      pointer_down: Boolean(app?.canvas?.pointer_is_down || app?.canvas?.pointerDown),
    };
  }

  function trace(kind, detail) {
    console.groupCollapsed?.(`${label} ${kind}`);
    console.log(`${label} ${kind}`, detail, canvasState());
    console.trace?.(`${label} ${kind}`);
    console.groupEnd?.();
  }

  function suspiciousSizeWrite(prop, prev, next) {
    const from = Number(prev);
    const to = Number(next);
    if (!Number.isFinite(from) || !Number.isFinite(to) || Math.abs(to - from) < 1) {
      return false;
    }
    const resizingThisNode = app?.canvas?.resizing_node === node;
    return !resizingThisNode || to < from - 8 || (prop === "0" && to < 320);
  }

  function wrapSizeArray(size) {
    if (!size || typeof size !== "object") return size;
    if (proxies.has(size)) return size;
    if (rawToProxy.has(size)) return rawToProxy.get(size);
    const proxy = new Proxy(size, {
      set(target, prop, value, receiver) {
        const prev = target[prop];
        const ok = Reflect.set(target, prop, value, receiver);
        if ((prop === "0" || prop === "1") && suspiciousSizeWrite(prop, prev, value)) {
          trace(`node.size[${prop}] write`, {
            axis: prop === "0" ? "width" : "height",
            from: prev,
            to: value,
            raw_size: [target[0], target[1]],
          });
        }
        return ok;
      },
    });
    rawToProxy.set(size, proxy);
    proxies.add(proxy);
    return proxy;
  }

  try {
    let currentSize = wrapSizeArray(node.size);
    Object.defineProperty(node, "size", {
      configurable: true,
      enumerable: true,
      get() { return currentSize; },
      set(next) {
        const prev = currentSize;
        currentSize = wrapSizeArray(next);
        trace("node.size replace", {
          from: Array.isArray(prev) ? [prev[0], prev[1]] : prev,
          to: Array.isArray(currentSize) ? [currentSize[0], currentSize[1]] : currentSize,
        });
      },
    });
  } catch (e) {
    console.warn(`${label} could not install node.size proxy`, e);
  }

  function observeResize(target, name) {
    if (!target || typeof ResizeObserver === "undefined") return;
    let last = null;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const rect = entry.contentRect;
        const next = { width: Math.round(rect.width), height: Math.round(rect.height) };
        if (last) {
          const dw = next.width - last.width;
          const dh = next.height - last.height;
          const resizingThisNode = app?.canvas?.resizing_node === node;
          const suspicious = !resizingThisNode || dw < -8 || next.width < 320 || Math.abs(dh) > 8;
          if (suspicious && (dw || dh)) {
            trace("DOM-widget resize delta", { target: name, from: last, to: next, delta: { width: dw, height: dh } });
          }
        }
        last = next;
      }
    });
    observer.observe(target);
    observers.push(observer);
  }

  observeResize(element, "widget-element");
  if (domWidget?.element && domWidget.element !== element) {
    observeResize(domWidget.element, "dom-widget-element");
  }
  requestAnimationFrame(() => observeResize(element?.parentElement, "widget-host"));

  node._atlasSizeTraceCleanup = () => observers.forEach((observer) => observer.disconnect());
  console.info(`${label} tracing enabled. Set localStorage.ATLAS_VIEWPORT_SIZE_TRACE = "0" to disable.`);
}

// Ground point under the camera's view centre, so the proxy/orbit-pivot lands
// where the camera is looking rather than at an arbitrary spot.
//
// lookAheadDist caps the ground-ray intersection distance and (for the
// looking-level/up case below) sets how far along the view ray the pivot
// sits. Near-horizontal shots (dir.y close to 0 — common for ordinary
// eye-level photography) make -p.y/dir.y blow up to hundreds or thousands of
// metres; when this feeds createOrbitControls' syncFromCamera, that huge
// distance becomes the orbit sphere's radius, so even a single pixel of drag
// swings the camera sideways by metres and the recovered geometry (which
// only spans tens of metres) leaves frame instantly. Capping keeps the pivot
// (and thus the orbit radius) within the scene's actual scale; callers pass
// the solved scene depth when known.
function groundPointInView(camera, lookAheadDist = 30) {
  const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  const p = camera.position;
  if (dir.y < -1e-3) {
    const t = Math.min(-p.y / dir.y, lookAheadDist);
    return new THREE.Vector3(p.x + t * dir.x, p.y + t * dir.y, p.z + t * dir.z);
  }
  // Looking level or upward — e.g. a tall building/facade shot, where the
  // view ray never crosses the ground plane in front of the camera. The
  // pivot used to be hardcoded to (p.x, 0, p.z - 3): a fixed point 3 units
  // along WORLD -Z, completely ignoring which way the camera actually faced
  // and how far away the subject really was. For an upward-looking shot of a
  // building tens of metres away, that pivot could be both the wrong
  // direction and absurdly close — orbiting around it swings the camera
  // instantly off into empty space, which is exactly the "mesh disappears
  // the moment you click to rotate" bug (confirmed live: the relief mesh —
  // a tall building facade — only reappeared after manually zooming the
  // orbit radius way out). Anchor along the camera's ACTUAL view direction
  // at the same scene-depth-aware distance instead.
  return new THREE.Vector3(p.x + lookAheadDist * dir.x, p.y + lookAheadDist * dir.y, p.z + lookAheadDist * dir.z);
}

// ---------------------------------------------------------------------------
// Self-contained orbit controller.
//
// The three.js examples/jsm OrbitControls uses a bare `import ... from "three"`
// specifier that browsers can't resolve without an import map, so it silently
// fails to load — which is why the viewport had no orbit. This minimal controller
// depends only on the already-loaded THREE module. It is initialised *from* the
// recovered camera (syncFromCamera) so the default view is the camera's own
// perspective; the first drag then orbits around the look-at target.
// ---------------------------------------------------------------------------
function createOrbitControls(camera, dom) {
  const target = new THREE.Vector3(0, 1, 0);
  const sph = { radius: 5, theta: 0, phi: Math.PI / 3 };
  let dragging = false, panning = false, lx = 0, ly = 0;
  // Disabled while Camera Path fly-mode is active (createFlyControls) so the
  // two controllers never fight over the same pointer/wheel events.
  let enabled = true;
  // UE-style tracking keys, scoped to THIS element's focus (no global
  // listeners — clicking the viewport focuses it; unrelated keys pass
  // through untouched so ComfyUI hotkeys keep working). tabIndex -1 =
  // focusable by click/JS but never in the tab order. outline suppressed —
  // the grab cursor already signals interactivity.
  dom.tabIndex = -1;
  dom.style.outline = "none";
  const NAV_KEYS = new Set(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
                            "KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE"]);
  const pressed = new Set();
  let navShift = false, lastNavT = 0;
  function onKeyDown(e) {
    if (!enabled || !NAV_KEYS.has(e.code)) return;
    pressed.add(e.code);
    navShift = e.shiftKey;
    e.preventDefault();
    e.stopPropagation();
  }
  function onKeyUp(e) {
    if (!NAV_KEYS.has(e.code)) return;
    pressed.delete(e.code);
    navShift = e.shiftKey;
    e.stopPropagation();
  }
  function onBlur() { pressed.clear(); }

  // Derived geometry (relief mesh, backdrop, fitted primitives) only ever
  // covers what the RECOVERED camera could see — a forward-facing cone — since
  // it's reconstructed from one photo. The orbit pivot is a nearby ground
  // point while the scene can extend many times farther, so an unconstrained
  // drag swings the viewing DIRECTION far more than it swings the camera
  // position: a modest-looking rotate can easily point past that cone into
  // space nothing was ever built for, which reads as the mesh/projection
  // "disappearing". Clamp yaw/pitch to an arc around the recovered direction
  // (theta0/phi0, re-anchored by syncFromCamera on every camera apply) so
  // orbiting always keeps something in view, while still allowing enough
  // sweep to inspect parallax and occlusion.
  let theta0 = 0, phi0 = Math.PI / 3;
  const MAX_YAW = THREE.MathUtils.degToRad(80);
  const MAX_PITCH = THREE.MathUtils.degToRad(55);
  // Asymmetric, per-scene clamp limits (radians, relative to theta0/phi0).
  // Defaults reproduce the historical ±80°/±55° arc; 🧭 Safe Zone replaces
  // them with MEASURED limits (see findSafeEnvelope) so the artist can't
  // orbit into holes at all.
  let limits = { thetaMin: -MAX_YAW, thetaMax: MAX_YAW,
                 phiMin: -MAX_PITCH, phiMax: MAX_PITCH };
  const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));

  // Recovered-camera ROLL about the view axis, captured at syncFromCamera and
  // re-applied after every lookAt. GeoCalib solves include roll (tilted
  // gravity — measured live at 28.4° on a hazy ridge photo with no true
  // horizon), and applyRecoveredView poses the camera with it; without this,
  // the first drag's apply() snapped the camera level and the whole projected
  // scene visibly rotated by the discarded roll (artist-reported as "the
  // orbit camera rotates anticlockwise when I click").
  let rollAngle = 0;

  function syncFromCamera() {
    const off = camera.position.clone().sub(target);
    sph.radius = Math.max(0.01, off.length());
    sph.theta = Math.atan2(off.x, off.z);
    sph.phi = Math.acos(Math.min(1, Math.max(-1, off.y / sph.radius)));
    theta0 = sph.theta;
    phi0 = sph.phi;
    // Signed roll = angle from the LEVEL up (world-up projected perpendicular
    // to the actual view direction — what lookAt would produce) to the
    // camera's ACTUAL up, about the view axis. Both from the quaternion, not
    // from position-target, so this measures the real orientation. Stable
    // under repeated syncs: apply() reproduces exactly this roll, so
    // re-measuring returns the same value.
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    const actualUp = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
    const lvl = new THREE.Vector3(0, 1, 0).addScaledVector(fwd, -fwd.y);
    if (lvl.lengthSq() > 1e-8) {
      lvl.normalize();
      rollAngle = Math.atan2(lvl.clone().cross(actualUp).dot(fwd), lvl.dot(actualUp));
    } else {
      rollAngle = 0;  // looking straight up/down — roll is undefined, go level
    }
  }
  function apply() {
    const sp = Math.sin(sph.phi), cp = Math.cos(sph.phi);
    camera.position.set(
      target.x + sph.radius * sp * Math.sin(sph.theta),
      target.y + sph.radius * cp,
      target.z + sph.radius * sp * Math.cos(sph.theta)
    );
    camera.up.set(0, 1, 0);
    camera.lookAt(target);
    // rotateZ spins about local +z = the BACKWARD axis, so it applies -angle
    // about the view direction; negate to reproduce the measured roll.
    if (Math.abs(rollAngle) > 1e-6) camera.rotateZ(-rollAngle);
  }
  function onDown(e) {
    if (!enabled) return;
    dragging = true;
    panning = e.button === 2 || e.shiftKey;
    lx = e.clientX; ly = e.clientY;
    dom.style.cursor = "grabbing";
    dom.focus({ preventScroll: true });  // arm the tracking keys
    // stopPropagation on POINTERDOWN specifically (not mousedown) is required:
    // LiteGraph's LGraphCanvas binds its node-drag/selection handling via
    // canvas.addEventListener('pointerdown', ...) — Pointer Events, which fire
    // BEFORE the corresponding legacy mousedown in the same click. Stopping
    // propagation on mousedown (as this used to) is too late; the interception
    // already happened on pointerdown. ComfyUI's own first-party Load3D widget
    // guards the identical case with `@pointerdown.stop` in Load3D.vue — this
    // mirrors that. Without it, a pointerdown here starts BOTH our orbit drag
    // and LiteGraph's own drag/selection handling on the same motion, which is
    // what reads as "the mesh disappears the moment you click to rotate."
    e.preventDefault();
    e.stopPropagation();
  }
  function onUp() { dragging = false; dom.style.cursor = "grab"; }
  function onMove(e) {
    if (!enabled || !dragging) return;
    const dx = e.clientX - lx, dy = e.clientY - ly;
    lx = e.clientX; ly = e.clientY;
    if (panning) {
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
      const k = sph.radius * 0.0015;
      target.addScaledVector(right, -dx * k).addScaledVector(up, dy * k);
    } else {
      const deltaTheta = wrapAngle(sph.theta - dx * 0.005 - theta0);
      sph.theta = theta0 + Math.min(limits.thetaMax, Math.max(limits.thetaMin, deltaTheta));

      const rawPhi = Math.min(Math.PI - 0.05, Math.max(0.05, sph.phi - dy * 0.005));
      sph.phi = Math.min(phi0 + limits.phiMax, Math.max(phi0 + limits.phiMin, rawPhi));
    }
    e.stopPropagation();
    apply();
  }
  function onWheel(e) {
    if (!enabled) return;
    sph.radius = Math.max(0.05, sph.radius * (1 + Math.sign(e.deltaY) * 0.1));
    apply();
    e.preventDefault();
    e.stopPropagation(); // don't let the graph canvas zoom underneath the widget
  }
  dom.addEventListener("pointerdown", onDown);
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
  dom.addEventListener("wheel", onWheel, { passive: false });
  dom.addEventListener("keydown", onKeyDown);
  dom.addEventListener("keyup", onKeyUp);
  dom.addEventListener("blur", onBlur);
  dom.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); });
  return {
    target,
    setTarget(v) { target.copy(v); },
    syncFromCamera,
    // TRUE tracking (translate target; apply() repositions the camera from
    // the sphere, so camera + target move together) — the exact mechanic the
    // Shift-drag pan above uses, exposed for the tracking keys.
    pan(v) { target.add(v); apply(); },
    // Per-frame keyboard integration (called from the animate() loop).
    // Mapping per the user's spec: ↑/↓ track in/out, ←/→ track left/right,
    // A/D track up/down — with W/S (in/out) and Q/E (up/down) as the UE
    // muscle-memory aliases. Self-timed; scene-scaled step. Deliberately
    // SLOW by default (real-camera tracking feel, user-tuned 2026-07-12,
    // twice): base 0.15·radius/s; Shift = 4× -> 0.6·radius/s.
    updateKeys() {
      const now = performance.now();
      const dt = lastNavT ? Math.min(0.1, (now - lastNavT) / 1000) : 0;
      lastNavT = now;
      if (!enabled || dragging || pressed.size === 0) return;
      const step = sph.radius * 0.15 * dt * (navShift ? 4 : 1);
      if (!(step > 0)) return;
      const forward = target.clone().sub(camera.position).normalize();
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
      const move = new THREE.Vector3();
      if (pressed.has("ArrowUp") || pressed.has("KeyW")) move.add(forward);
      if (pressed.has("ArrowDown") || pressed.has("KeyS")) move.sub(forward);
      if (pressed.has("ArrowRight")) move.add(right);
      if (pressed.has("ArrowLeft")) move.sub(right);
      if (pressed.has("KeyA") || pressed.has("KeyE")) move.y += 1;
      if (pressed.has("KeyD") || pressed.has("KeyQ")) move.y -= 1;
      if (move.lengthSq() === 0) return;
      this.pan(move.normalize().multiplyScalar(step));
    },
    // Measured (or default) orbit limits — pass null to restore defaults.
    setLimits(l) {
      limits = l ? { ...l }
        : { thetaMin: -MAX_YAW, thetaMax: MAX_YAW,
            phiMin: -MAX_PITCH, phiMax: MAX_PITCH };
    },
    // Recovered-pose anchors + orbit sphere, for the 🧭 probe renders.
    getFrame() {
      return { theta0, phi0, radius: sph.radius, target: target.clone() };
    },
    setEnabled(v) { enabled = v; if (!v) dragging = false; dom.style.cursor = v ? "grab" : "default"; },
    dispose() {
      dom.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      dom.removeEventListener("wheel", onWheel);
      dom.removeEventListener("keydown", onKeyDown);
      dom.removeEventListener("keyup", onKeyUp);
      dom.removeEventListener("blur", onBlur);
    },
  };
}

// ---------------------------------------------------------------------------
// Fly-mode controller for Camera Path authoring.
//
// The orbit controller above is deliberately clamped to a small arc around
// the recovered camera's own direction (see its comment) because derived
// geometry only covers what that camera actually photographed. Authoring a
// dolly/pan camera move is the opposite case — leaving that cone is the
// whole point of testing how projection degrades under motion — so this is
// a separate, UNCLAMPED free-fly controller (RMB-hold + WASD/QE, mirroring
// spiritform/comfyblockout's fly nav), only active while Camera Path mode is on.
// ---------------------------------------------------------------------------
function createFlyControls(camera, dom) {
  let enabled = false;
  let dragging = false, lx = 0, ly = 0;
  let yaw = 0, pitch = 0;
  const keys = new Set();
  const MOVE_UNITS_PER_SEC = 4;

  function syncFromCamera() {
    const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, "YXZ");
    yaw = euler.y;
    pitch = euler.x;
  }
  function applyLook() {
    camera.quaternion.setFromEuler(new THREE.Euler(pitch, yaw, 0, "YXZ"));
  }
  function onDown(e) {
    if (!enabled || e.button !== 2) return;
    dragging = true;
    lx = e.clientX; ly = e.clientY;
    dom.style.cursor = "grabbing";
    e.preventDefault();
    e.stopPropagation();
  }
  function onUp() {
    dragging = false;
    keys.clear();
    if (enabled) dom.style.cursor = "crosshair";
  }
  function onMove(e) {
    if (!enabled || !dragging) return;
    const dx = e.clientX - lx, dy = e.clientY - ly;
    lx = e.clientX; ly = e.clientY;
    yaw -= dx * 0.004;
    pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, pitch - dy * 0.004));
    applyLook();
    e.stopPropagation();
  }
  function onKeyDown(e) { if (enabled && dragging) keys.add(e.key.toLowerCase()); }
  function onKeyUp(e) { keys.delete(e.key.toLowerCase()); }
  function onContext(e) { if (enabled) { e.preventDefault(); e.stopPropagation(); } }

  function tick(dt) {
    if (!enabled || !dragging || keys.size === 0) return;
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
    const step = MOVE_UNITS_PER_SEC * dt;
    if (keys.has("w")) camera.position.addScaledVector(forward, step);
    if (keys.has("s")) camera.position.addScaledVector(forward, -step);
    if (keys.has("d")) camera.position.addScaledVector(right, step);
    if (keys.has("a")) camera.position.addScaledVector(right, -step);
    if (keys.has("e")) camera.position.y += step;
    if (keys.has("q")) camera.position.y -= step;
  }

  dom.addEventListener("pointerdown", onDown);
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  dom.addEventListener("contextmenu", onContext);

  return {
    tick,
    setEnabled(v) {
      enabled = v;
      dragging = false;
      keys.clear();
      dom.style.cursor = v ? "crosshair" : "grab";
      if (v) syncFromCamera();
    },
    dispose() {
      dom.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      dom.removeEventListener("contextmenu", onContext);
    },
  };
}

// NOTE (2026-07-09): the primitive toolbar (Box/Plane/Cylinder/Person) and the
// 🧍/🚗 OBJ scale-proxy buttons were removed — browser-only meshes that never
// persisted to the solve or any export ("void functions", artist-confirmed
// unused; scale checks are covered by the tiered cascade + ℹ Info HUD). The
// /atlas/proxy_model route and examples/models/*.obj remain server-side; if
// hand-placed proxy geometry is ever wanted again, build it as a NODE writing
// a real PROXY_ROLE primitive, not ephemeral viewport meshes (see git history
// for the removed loadProxyModel/createPrimitive implementations).

// ---------------------------------------------------------------------------
// Camera-projection material (matte-painting projection).
//
// Ported from ui/src/ProjectionMaterial.ts: project each fragment's world
// position through the RECOVERED camera (uAtlasViewMatrix + fx/fy/cx/cy) to an
// image pixel and sample the source photo there. Because texels are assigned by
// ray, geometry at slightly wrong depth still receives exactly the pixels its
// silhouette subtends — the image reassembles perfectly from Camera View.
// Deviation from the ui version: depthWrite/depthTest ON so multiple proxies
// occlude each other correctly (the ui version was a single-ground overlay).
//
// Conflicts with AtlasBlockoutViewport's preview_expand: dilated geometry is,
// by construction, surface the recovered camera never actually photographed,
// so its projected pixel always falls outside the source frame and gets
// discarded below -> renders empty/black. This is why preview_expand now
// defaults to 1.0 (off) node-side; raising it helps the undressed grey
// preview orbit further, but guarantees black gaps the moment Project is on
// and you orbit even slightly off the exact recovered viewpoint.
// ---------------------------------------------------------------------------
const PROJECTION_VERTEX_SHADER = `
  uniform mat4 uAtlasViewMatrix;
  uniform float uFx;
  uniform float uFy;
  uniform float uCx;
  uniform float uCy;
  varying vec2 vImagePx;
  varying float vCamZ;
  varying vec3 vWorldPos;
  varying vec3 vWorldNormal;
  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    vWorldNormal = normalize(mat3(modelMatrix) * normal);
    vec4 cam = uAtlasViewMatrix * worldPos;
    vCamZ = cam.z;
    float depth = -cam.z;   // Atlas camera looks along -Z
    if (depth > 1e-5) {
      vImagePx = vec2(uCx + uFx * cam.x / depth, uCy - uFy * cam.y / depth);
    } else {
      vImagePx = vec2(-1.0, -1.0);
    }
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

// uFacingThreshold: discard fragments whose surface is more grazing to THIS
// projector than the threshold (|normal . dir-to-camera| < threshold). This is
// the dot-product occlusion / facing-ratio mask from gs_mptk — patch cameras
// use a positive threshold so they only paint surfaces they see reasonably
// head-on (letting the primary / other patches fill grazing areas). The primary
// passes a negative threshold so it never facing-discards (it always has
// priority where it can see). |.| is used so mesh winding / DoubleSide is
// irrelevant.
// uLight{1,2}Intensity default to 0 (movableLights start off — see "Movable
// point lights" below), which keeps this a strict no-op: relight == vec3(1.0)
// == the original texture-only output, so every workflow authored before this
// feature renders pixel-identical unless an artist explicitly dials a light
// up. This is a stylized dodge-and-burn multiply, NOT physically-correct
// relighting — the source photo already carries its own real-world lighting;
// there is no normal-lighting term here to "correct", only to bias by eye.
const PROJECTION_FRAGMENT_SHADER = `
  uniform sampler2D uTexture;
  uniform sampler2D uMatte;
  uniform float uHasMatte;
  uniform sampler2D uHiddenMask;
  uniform float uHasHiddenMask;
  uniform float uDebugHidden;
  uniform vec3 uHiddenTint;
  uniform float uLayerDebug;
  uniform vec3 uLayerTint;
  uniform vec2 uImageSize;
  uniform float uOpacity;
  uniform vec3 uCamPos;
  uniform float uFacingThreshold;
  uniform vec3 uLight1Pos;
  uniform vec3 uLight1Color;
  uniform float uLight1Intensity;
  uniform vec3 uLight2Pos;
  uniform vec3 uLight2Color;
  uniform float uLight2Intensity;
  uniform vec3 uLight3Pos;
  uniform vec3 uLight3Color;
  uniform float uLight3Intensity;
  uniform float uSceneScale;
  uniform float uBumpStrength;
  uniform float uBumpScale;
  uniform sampler2D uNormalMap;   // predicted WORLD normals (MoGe *-normal), (n+1)/2 in RGB
  uniform float uHasNormalMap;
  varying vec2 vImagePx;
  varying float vCamZ;
  varying vec3 vWorldPos;
  varying vec3 vWorldNormal;
  float atlasRelightTerm(vec3 lightPos, vec3 lightColor, float intensity, vec3 worldPos, vec3 worldNormal) {
    if (intensity <= 0.0) return 0.0;
    vec3 toLight = lightPos - worldPos;
    float dist = length(toLight);
    float ndotl = max(dot(normalize(worldNormal), normalize(toLight)), 0.0);
    // Scale-aware falloff: distance is measured relative to the scene's metric
    // scale (uSceneScale = recovered camera height / 1.6 m default eye height),
    // so a light placed proportionally to the scene gives the same relight at
    // any AtlasScaleOverride. uSceneScale=1 (the ~1.6 m default) reproduces the
    // original 1/(1+0.05·dist²) exactly — backward-compatible.
    float ds = dist / max(uSceneScale, 1e-3);
    float atten = 1.0 / (1.0 + 0.05 * ds * ds);
    return intensity * ndotl * atten;
  }
  // Detail relight: perturb the surface normal using the PHOTO's own luminance
  // as a heightfield, so the lights sculpt fine surface detail (brick, foliage,
  // rock) the coarse projection geometry lacks. The height gradient is sampled
  // in TEXEL space (zoom-stable — the detail scale doesn't change as you orbit),
  // then mapped tangent→world by a cotangent frame built from screen-space
  // derivatives of world position + uv (no precomputed tangents needed). Feeds
  // the relight ONLY — the base texture is never altered. Brighter = higher.
  vec3 atlasBumpNormal(vec3 N, vec3 p, vec2 uv, float strength) {
    // Sampling offset in texels (uBumpScale) sets the detail scale: 1 texel is
    // too fine to register on a big plate (adjacent-pixel luminance is near-
    // identical), so the default samples several texels apart for real
    // meso-detail (brick/foliage). Larger = coarser/stronger.
    vec2 texel = max(uBumpScale, 1.0) / uImageSize;
    vec3 lw = vec3(0.299, 0.587, 0.114);
    float hL = dot(texture2D(uTexture, uv - vec2(texel.x, 0.0)).rgb, lw);
    float hR = dot(texture2D(uTexture, uv + vec2(texel.x, 0.0)).rgb, lw);
    float hD = dot(texture2D(uTexture, uv - vec2(0.0, texel.y)).rgb, lw);
    float hU = dot(texture2D(uTexture, uv + vec2(0.0, texel.y)).rgb, lw);
    vec3 tn = normalize(vec3((hL - hR) * strength, (hD - hU) * strength, 1.0));
    vec3 dp1 = dFdx(p), dp2 = dFdy(p);
    vec2 duv1 = dFdx(uv), duv2 = dFdy(uv);
    vec3 dp2perp = cross(dp2, N);
    vec3 dp1perp = cross(N, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(dot(T, T), dot(B, B)));
    mat3 tbn = mat3(T * invmax, B * invmax, N);
    return normalize(tbn * tn);
  }
  // Matches THREE.ShaderChunk's own LinearTosRGB (r0.41666 ~= 1/2.4). uTexture
  // is tagged colorSpace=SRGBColorSpace, so the GPU already decodes it to
  // LINEAR on sample — texture2D() below returns linear, not display sRGB.
  // Built-in materials (MeshStandardMaterial etc.) get Three's own
  // colorspace_fragment chunk auto-appended for the reverse encode before
  // output; a raw ShaderMaterial like this one never does, so without this
  // explicit encode the whole projected photo silently renders too dark/
  // desaturated (linear values written straight into an sRGB framebuffer).
  vec3 atlasLinearToSRGB(vec3 value) {
    return mix(pow(value, vec3(0.41666)) * 1.055 - vec3(0.055), value * 12.92, vec3(lessThanEqual(value, vec3(0.0031308))));
  }
  void main() {
    if (vCamZ >= 0.0) discard;                    // behind the projector camera
    vec2 uv = vImagePx / uImageSize;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) discard;
    // Per-pixel edge matte (ProjectionSource.mask_b64) — the classic DMP move:
    // geometry silhouettes tear at grid-quad resolution (blocky staircases),
    // so the full-resolution matte cuts the TRUE edge instead. Sampled at the
    // same projected pixel as the photo itself, so it needs no separate UVs.
    if (uHasMatte > 0.5 && texture2D(uMatte, uv).r < 0.5) discard;
    vec3 toCam = normalize(uCamPos - vWorldPos);
    float facing = abs(dot(normalize(vWorldNormal), toCam));
    if (facing < uFacingThreshold) discard;       // too grazing for this projector
    vec4 col = texture2D(uTexture, uv);
    // Relight normal: the model's predicted WORLD normal (uNormalMap, already
    // aligned to the recovered frame — image-resolution, cleaner than the coarse
    // mesh normal) when present, else the geometry normal; then optionally
    // perturbed with photo-luminance micro-detail (uBumpStrength > 0). Only the
    // LIGHTS read this — the facing discard above stays on the true geometry normal.
    vec3 N = normalize(vWorldNormal);
    if (uHasNormalMap > 0.5) {
      vec3 mn = texture2D(uNormalMap, uv).rgb * 2.0 - 1.0;
      if (dot(mn, mn) > 0.25) N = normalize(mn);
    }
    if (uBumpStrength > 0.0) N = atlasBumpNormal(N, vWorldPos, uv, uBumpStrength);
    vec3 relight = vec3(1.0)
      + uLight1Color * atlasRelightTerm(uLight1Pos, uLight1Color, uLight1Intensity, vWorldPos, N)
      + uLight2Color * atlasRelightTerm(uLight2Pos, uLight2Color, uLight2Intensity, vWorldPos, N)
      + uLight3Color * atlasRelightTerm(uLight3Pos, uLight3Color, uLight3Intensity, vWorldPos, N);
    vec3 outColor = atlasLinearToSRGB(clamp(col.rgb * relight, 0.0, 1.0));
    // 🩻 hidden-geometry provenance overlay (debug): tint the surface region
    // whose depth was SUBSTITUTED by AtlasPredictHiddenGeometry (the node's
    // hidden_mask, threaded through the ProjectionSource) — red = LaRI,
    // blue = World Tracing (uHiddenTint set per-source at material build).
    // Sampled at the same projected uv as the photo/matte; applied after the
    // sRGB encode since it's a display-space annotation, not scene color.
    if (uDebugHidden > 0.5 && uHasHiddenMask > 0.5 && texture2D(uHiddenMask, uv).r > 0.5) {
      outColor = mix(outColor, uHiddenTint, 0.5);
    }
    // 🎨 layer-debug overlay: tint EVERYTHING this projection source paints
    // with its own identifying color (base/primary + each ProjectionSource
    // get distinct palette entries at material build; legend in the toolbar).
    // Strong mix so layer coverage reads at a glance; display-space like 🩻.
    if (uLayerDebug > 0.5) {
      outColor = mix(outColor, uLayerTint, 0.65);
    }
    gl_FragColor = vec4(outColor, col.a * uOpacity);
  }
`;

// ---------------------------------------------------------------------------
// Patch-priority ordering (ProjectionSource.priority — "higher wins; the
// primary is implicitly highest", core/schema.py). Real z-buffering already
// resolves most overlap between the primary and patch geometry (or between
// two patches) by actual depth; these two mechanisms only disambiguate the
// band where depth is coincident or near-coincident (independently-derived
// meshes rarely align exactly):
//   - renderOrder makes EXACT depth ties deterministic (Three sorts opaque
//     renderables by renderOrder before the per-object depth test) instead
//     of scene-graph/load-order-dependent.
//   - polygonOffsetUnits biases the effective depth-buffer value by a small,
//     priority-scaled amount so a higher-priority mesh wins within that
//     epsilon window, while a genuinely-nearer mesh (real gap larger than the
//     bias) still wins the normal z-test.
// The primary is never a ProjectionSource (no priority field) and is given a
// sentinel renderOrder above any patch, satisfying "implicitly highest"
// without a synthetic number. Discards (behind-camera / out-of-UV / facing-
// threshold, in the fragment shader below) happen before any depth write, so
// this is independent of the separate preview_expand/Project dilation
// tradeoff documented above.
// 🎨 layer-debug identity palette: primary/base gets its own fixed color;
// each ProjectionSource takes palette[index % length]. Chosen for mutual
// distinguishability at the shader's 0.65 mix over arbitrary photos.
const LAYER_DEBUG_PRIMARY = 0x2fd6c3;               // teal — base mesh + backdrop
const LAYER_DEBUG_PALETTE = [
  0xff6a3d, // orange — typically the fg layer
  0x3d8bff, // blue   — typically the X-ray/bg layer
  0xffd23d, // yellow
  0xc95aff, // violet
  0x6aff5a, // green
  0xff5aa8, // pink
];

const PATCH_PRIORITY_CEILING = 100; // matches nodes.py AtlasAddPatchView widget max
const PATCH_OFFSET_STEP = 4;        // depth-bias units; tuned visually in-viewport
function priorityToRenderOrder(p) {
  return 1 + Math.round(Math.max(0, p || 0));
}
function priorityToOffsetUnits(p) {
  const c = Math.min(PATCH_PRIORITY_CEILING, Math.max(0, p || 0));
  return PATCH_OFFSET_STEP * (1 - c / PATCH_PRIORITY_CEILING) + 0.5; // always > 0
}

function makeProjectionMaterial(data, texture, opts) {
  const options = opts || {};
  const flat = data.view_matrix.flat();
  const vm = new THREE.Matrix4();
  vm.set(
    flat[0], flat[1], flat[2], flat[3],
    flat[4], flat[5], flat[6], flat[7],
    flat[8], flat[9], flat[10], flat[11],
    flat[12], flat[13], flat[14], flat[15]
  );
  const camPos = data.camera_position || [0, 0, 0];
  // Scale-aware relight falloff (see PROJECTION_FRAGMENT_SHADER): the light
  // attenuation distance scales with the scene's metric scale, proxied by the
  // recovered camera height vs the 1.6 m default eye height — so a large
  // AtlasScaleOverride (geometry 100 m+) no longer starves the lights. Exactly
  // 1 at the default height, so existing ~1.6 m-camera looks are unchanged.
  const sceneScale = Math.max(Math.abs(camPos[1]) / 1.6, 0.1);
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uAtlasViewMatrix: { value: vm },
      uFx: { value: data.fx || 1 },
      uFy: { value: data.fy || data.fx || 1 },
      uCx: { value: data.cx ?? (data.image_width || 1) / 2 },
      uCy: { value: data.cy ?? (data.image_height || 1) / 2 },
      uTexture: { value: texture },
      uImageSize: { value: new THREE.Vector2(data.image_width || 1, data.image_height || 1) },
      uOpacity: { value: 1.0 },
      uCamPos: { value: new THREE.Vector3(camPos[0], camPos[1], camPos[2]) },
      // Optional per-pixel edge matte (see PROJECTION_FRAGMENT_SHADER). The
      // uniform-gated branch means a null sampler is never actually read.
      uMatte: { value: options.matteTexture || null },
      uHasMatte: { value: options.matteTexture ? 1.0 : 0.0 },
      // 🩻 hidden-geometry provenance mask + tint (debug overlay; uDebugHidden
      // is synced live by syncProjectionLightUniforms like the light uniforms,
      // since projection materials are rebuilt on every execution).
      uHiddenMask: { value: options.hiddenMaskTexture || null },
      uHasHiddenMask: { value: options.hiddenMaskTexture ? 1.0 : 0.0 },
      uDebugHidden: { value: 0 },
      uHiddenTint: { value: options.hiddenTint || new THREE.Color(1.0, 0.15, 0.15) },
      // 🎨 layer-debug identity color (fixed per source at build; toggle is
      // uLayerDebug, live-synced like uDebugHidden/the light uniforms).
      uLayerDebug: { value: 0 },
      uLayerTint: { value: options.layerTint || new THREE.Color(LAYER_DEBUG_PRIMARY) },
      // Primary: -1 (never facing-discards). Patches: positive (fill head-on only).
      uFacingThreshold: { value: options.facingThreshold ?? -1.0 },
      // Movable point lights (💡) — kept at intensity 0 here; synced live each
      // frame from the shared `movableLights` rig by syncProjectionLightUniforms()
      // so every projection material (primary + every patch/clean-plate source)
      // stays in lockstep without needing to be rebuilt when a light moves.
      uLight1Pos: { value: new THREE.Vector3() },
      uLight1Color: { value: new THREE.Color(0xffffff) },
      uLight1Intensity: { value: 0 },
      uLight2Pos: { value: new THREE.Vector3() },
      uLight2Color: { value: new THREE.Color(0xffffff) },
      uLight2Intensity: { value: 0 },
      uLight3Pos: { value: new THREE.Vector3() },
      uLight3Color: { value: new THREE.Color(0xffffff) },
      uLight3Intensity: { value: 0 },
      uSceneScale: { value: sceneScale },   // scale-aware relight falloff (cam height / 1.6m)
      uNormalMap: { value: null },          // predicted world-normal relight map (loaded below if present)
      uHasNormalMap: { value: 0 },
      // Detail-relight bump strength (💡 Lights panel "Detail" slider); 0 = off
      // = the geometry normal, so backward-compatible. Live-synced like lights.
      uBumpStrength: { value: 0 },
      uBumpScale: { value: 8.0 },   // luminance-gradient sampling offset in texels ("Scale")
    },
    vertexShader: PROJECTION_VERTEX_SHADER,
    fragmentShader: PROJECTION_FRAGMENT_SHADER,
    side: THREE.DoubleSide,
    transparent: false,
    depthWrite: true,
    depthTest: true,
  });
  // Priority-driven depth bias (patches only — options.priority is unset for
  // the primary, which relies solely on its renderOrder sentinel instead).
  if (options.priority !== undefined) {
    mat.polygonOffset = true;
    mat.polygonOffsetFactor = 0;
    mat.polygonOffsetUnits = priorityToOffsetUnits(options.priority);
  }
  // Predicted world-normal relight map (MoGe *-normal), loaded async and gated by
  // uHasNormalMap. NoColorSpace (raw data, never gamma-decoded) + flipY:false so
  // it samples at the same projected uv as the photo.
  if (data.normal_map_b64) {
    new THREE.TextureLoader().load(data.normal_map_b64, (tex) => {
      tex.colorSpace = THREE.NoColorSpace;
      tex.flipY = false;
      tex.needsUpdate = true;
      mat.uniforms.uNormalMap.value = tex;
      mat.uniforms.uHasNormalMap.value = 1;
    });
  }
  return mat;
}

function loadTextureFromB64(b64, cb) {
  if (!b64) return;
  const loader = new THREE.TextureLoader();
  loader.load(b64, (tex) => {
    tex.flipY = false;                // shader UV origin is top-left
    tex.colorSpace = THREE.SRGBColorSpace;
    cb(tex);
  }, undefined, (err) => {
    // A layer whose plate never loads stays grey in 📽 Project — make that
    // diagnosable instead of silent.
    console.warn("[AtlasBlockout] projection texture failed to load (layer stays grey):", err);
  });
}

// Edge mattes are DATA, not color: tagging them SRGBColorSpace would make the
// GPU sRGB-decode on sample (a gray 128 would read ~0.216 linear, silently
// shifting the 0.5 threshold). NoColorSpace keeps stored bytes = sampled
// values; the default linear mag filter gives a soft half-pixel edge.
// ALWAYS calls cb — with null on missing/failed matte — so a broken matte
// degrades to an unmatted projection instead of leaving the layer grey
// forever (the projection material only builds inside this callback).
function loadMatteFromB64(b64, cb) {
  if (!b64) { cb(null); return; }
  const loader = new THREE.TextureLoader();
  loader.load(b64, (tex) => {
    tex.flipY = false;
    tex.colorSpace = THREE.NoColorSpace;
    cb(tex);
  }, undefined, (err) => {
    console.warn("[AtlasBlockout] edge matte failed to load — projecting unmatted:", err);
    cb(null);
  });
}

function loadProjectionTexture(data, cb) {
  if (!data.source_image_b64) return;
  const loader = new THREE.TextureLoader();
  loader.load(data.source_image_b64, (tex) => {
    // The shader computes UV with a top-left pixel origin; do NOT share the
    // background texture, which keeps three.js's default flipY=true.
    tex.flipY = false;
    tex.colorSpace = THREE.SRGBColorSpace;
    cb(tex);
  });
}

// Build meshes for the Python-derived projection proxies (ground/walls/boxes/
// cylinders/backdrop). Transforms arrive as row-major 16-float arrays — the
// same convention THREE.Matrix4.set() takes.
function buildDerivedProxies(scene, data) {
  const old = scene.getObjectByName("atlas_derived_proxies");
  if (old) {
    old.traverse((m) => {
      m.geometry?.dispose?.();
      // Dispose only per-mesh grey materials — never the shared projection
      // ShaderMaterial a projected mesh may currently hold.
      if (m.material?.isMeshStandardMaterial) m.material.dispose();
      if (m.userData?._prevMaterial?.isMeshStandardMaterial) {
        m.userData._prevMaterial.dispose();
      }
    });
    scene.remove(old);
  }
  const entries = data.proxy_geometry || [];
  const group = new THREE.Group();
  group.name = "atlas_derived_proxies";
  group.userData.atlasDerivedGroup = true;
  for (const e of entries) {
    let geo;
    const d = e.dimensions || [1, 1, 1];
    if (e.type === "mesh") {
      // Relief mesh: world-space vertices/faces/uvs shipped flat in the payload.
      if (!e.vertices?.length || !e.faces?.length) continue;
      geo = new THREE.BufferGeometry();
      geo.setAttribute("position",
        new THREE.BufferAttribute(new Float32Array(e.vertices), 3));
      if (e.uvs?.length) {
        geo.setAttribute("uv", new THREE.BufferAttribute(new Float32Array(e.uvs), 2));
      }
      geo.setIndex(new THREE.BufferAttribute(new Uint32Array(e.faces), 1));
      geo.computeVertexNormals();
    } else if (e.type === "box") {
      geo = new THREE.BoxGeometry(d[0], d[1], d[2]);
    } else if (e.type === "cylinder") {
      geo = new THREE.CylinderGeometry(d[0] / 2, d[0] / 2, d[1], 24);
    } else {
      geo = new THREE.PlaneGeometry(d[0], d[1]);
    }
    const mat = new THREE.MeshStandardMaterial({
      color: 0x9a9a9a, roughness: 0.85, side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.matrixAutoUpdate = false;
    mesh.matrix.set(...e.transform);
    mesh.userData.atlasDerived = true;
    mesh.name = e.name || "derived_proxy";
    // Sentinel above any patch renderOrder (see priorityToRenderOrder) — the
    // primary is implicitly highest priority per ProjectionSource's contract,
    // with no synthetic priority number needed.
    mesh.renderOrder = 100000;
    group.add(mesh);
  }
  scene.add(group);
  return group;
}

// Build one proxy entry's THREE geometry (relief mesh / box / cylinder / plane).
// Shared by the primary derived proxies and the multi-angle patch sources.
function proxyEntryToGeometry(e) {
  const d = e.dimensions || [1, 1, 1];
  if (e.type === "mesh") {
    if (!e.vertices?.length || !e.faces?.length) return null;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(e.vertices), 3));
    if (e.uvs?.length) {
      geo.setAttribute("uv", new THREE.BufferAttribute(new Float32Array(e.uvs), 2));
    }
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(e.faces), 1));
    geo.computeVertexNormals();
    return geo;
  }
  if (e.type === "box") return new THREE.BoxGeometry(d[0], d[1], d[2]);
  if (e.type === "cylinder") return new THREE.CylinderGeometry(d[0] / 2, d[0] / 2, d[1], 24);
  return new THREE.PlaneGeometry(d[0], d[1]);
}

// Build the multi-angle patch sources (AtlasAddPatchView). Each source is its
// own camera + AI novel-view image + geometry; each mesh carries its OWN
// projection material (bound to that source's camera+image, with a facing-ratio
// mask) in userData._projMaterial, so applyProjection layers it over the
// primary. Patch geometry is Python-owned (regenerated each execution), so —
// like the derived group — Clear leaves it alone.
function buildPatchSources(scene, data, onSourceReady) {
  const stale = [];
  scene.traverse((c) => { if (c.userData?.atlasPatchGroup) stale.push(c); });
  for (const g of stale) {
    g.traverse((m) => {
      m.geometry?.dispose?.();
      if (m.material?.isMeshStandardMaterial) m.material.dispose();
      if (m.userData?._prevMaterial?.isMeshStandardMaterial) m.userData._prevMaterial.dispose();
      const pm = m.userData?._projMaterial;
      if (pm) {
        pm.uniforms?.uTexture?.value?.dispose?.();
        pm.uniforms?.uMatte?.value?.dispose?.();
        pm.uniforms?.uHiddenMask?.value?.dispose?.();
        pm.dispose?.();
      }
    });
    scene.remove(g);
  }

  const sources = data.projection_sources || [];
  sources.forEach((src, idx) => {
    const group = new THREE.Group();
    group.name = `atlas_patch_${idx}`;
    group.userData.atlasPatchGroup = true;
    // Band metrics for the 📏 Band Box overlay: a finite far_m on a clean-plate
    // layer is the AtlasBoundedBand cutoff (the foreground's back edge).
    group.userData.sourceName = src.name;
    group.userData.near_m = src.near_m;
    group.userData.far_m = src.far_m;
    group.userData.band_geometry = src.band_geometry;
    group.userData.projection_mode = src.projection_mode;
    const meshes = [];
    for (const e of (src.proxy_geometry || [])) {
      const geo = proxyEntryToGeometry(e);
      if (!geo) continue;
      const mat = new THREE.MeshStandardMaterial({ color: 0x8a9a80, roughness: 0.85, side: THREE.DoubleSide });
      mat.polygonOffset = true;
      mat.polygonOffsetFactor = 0;
      mat.polygonOffsetUnits = priorityToOffsetUnits(src.priority);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.matrixAutoUpdate = false;
      mesh.matrix.set(...e.transform);
      mesh.userData.atlasPatch = true;
      mesh.name = e.name || `patch_${idx}`;
      // Deterministic overlap ordering from ProjectionSource.priority — see
      // priorityToRenderOrder/priorityToOffsetUnits above makeProjectionMaterial.
      mesh.renderOrder = priorityToRenderOrder(src.priority);
      group.add(mesh);
      meshes.push(mesh);
    }
    scene.add(group);
    // Load this patch's novel view and build its projection material. Patches
    // only paint surfaces they see reasonably head-on (facingThreshold > 0), so
    // grazing/occluded areas fall through to the primary or other patches.
    loadTextureFromB64(src.image_b64, (tex) => {
      // Clean-plate layers (AtlasCleanPlateLayer) are same-camera plates, not
      // novel angles — they must paint head-on AND grazing surfaces, exactly
      // like the primary (facingThreshold -1 = never facing-discards), relying
      // on depth + priority alone to order overlapping layers. Ordinary
      // multi-angle patches keep the grazing-discard behavior so they only
      // fill surfaces they see reasonably head-on.
      const facingThreshold = src.projection_mode === "clean_plate" ? -1 : 0.2;
      const build = (matteTexture, hiddenMaskTexture) => {
        // 🩻 provenance tint per backend: red = LaRI, blue = World Tracing.
        const hiddenTint = src.hidden_backend === "world-tracing"
          ? new THREE.Color(0.2, 0.4, 1.0)
          : new THREE.Color(1.0, 0.15, 0.15);
        const patchMat = makeProjectionMaterial(src, tex,
          { facingThreshold, priority: src.priority, matteTexture,
            hiddenMaskTexture, hiddenTint,
            layerTint: new THREE.Color(
              LAYER_DEBUG_PALETTE[idx % LAYER_DEBUG_PALETTE.length]) });
        for (const m of meshes) {
          const prev = m.userData._projMaterial;
          if (prev && prev !== patchMat) {
            prev.uniforms?.uTexture?.value?.dispose?.();
            prev.uniforms?.uMatte?.value?.dispose?.();
            prev.uniforms?.uHiddenMask?.value?.dispose?.();
            prev.dispose?.();
          }
          m.userData._projMaterial = patchMat;
        }
        if (typeof onSourceReady === "function") onSourceReady();
      };
      // Per-pixel edge matte (ProjectionSource.mask_b64): geometry stays
      // coarse; the matte cuts the true silhouette in the shader.
      // loadMatteFromB64 always calls back (null on missing/failed matte).
      loadMatteFromB64(src.mask_b64, (matteTexture) => {
        if (src.hidden_mask_b64) {
          loadMatteFromB64(src.hidden_mask_b64,
            (hm) => build(matteTexture, hm));
        } else {
          build(matteTexture, null);
        }
      });
    });
  });
}

// ---------------------------------------------------------------------------
// Camera data cache (per node id)
// ---------------------------------------------------------------------------
const _cameraDataCache = new Map(); // nodeId → camera dict

async function fetchCameraData(nodeId) {
  try {
    const resp = await fetch(`/atlas/camera_data/${nodeId}`);
    if (!resp.ok) return null;
    const data = await resp.json();
    if (data && data.view_matrix) {
      _cameraDataCache.set(nodeId, data);
      return data;
    }
  } catch (e) {
    console.warn("[AtlasBlockout] Could not fetch camera data:", e);
  }
  return null;
}

// ---------------------------------------------------------------------------
// Apply recovered Atlas camera to Three.js PerspectiveCamera
// Atlas convention: row-major 4×4 view matrix, camera looks along -Z.
// ---------------------------------------------------------------------------
function applyRecoveredCamera(threeCamera, data) {
  if (!data || !data.view_matrix || !THREE) return;

  const flat = data.view_matrix.flat();
  // THREE.Matrix4.set() takes column-major order, but Atlas stores rows.
  // We set via elements array (column-major) by transposing:
  const vm = new THREE.Matrix4();
  vm.set(
    flat[0],  flat[1],  flat[2],  flat[3],
    flat[4],  flat[5],  flat[6],  flat[7],
    flat[8],  flat[9],  flat[10], flat[11],
    flat[12], flat[13], flat[14], flat[15]
  );

  // camToWorld = inverse of view matrix
  const camToWorld = vm.clone().invert();
  threeCamera.matrix.copy(camToWorld);
  threeCamera.matrix.decompose(
    threeCamera.position,
    threeCamera.quaternion,
    threeCamera.scale
  );

  // FOV from fy and image height. Deliberately NOT data.fy/data.image_height
  // directly — those are also read by makeProjectionMaterial() for the
  // PRIMARY source's own texture-sampling uniforms (this same `data` object
  // feeds both applyCamera() and setProxies() from the same execution), so
  // overriding them for a project-level ShotCam would corrupt how the photo
  // gets projected onto geometry. render_fy/render_image_height are a
  // separate pair the Python side always sets — equal to fy/image_height
  // when no ShotCam is wired in (so this is a no-op then), or the shot
  // format's own values when one is (AtlasDefineShotCam + AtlasBlockoutViewport's
  // shot_cam input / a solve with .shot_cam attached by AtlasMergeGeometry).
  const imageH = data.render_image_height ?? data.image_height ?? 1080;
  const fy = data.render_fy ?? data.fy ?? 1;
  const fovYRad = 2 * Math.atan(imageH / (2 * fy));
  threeCamera.fov = fovYRad * (180 / Math.PI);
  const aspect = (data.target_width || 512) / (data.target_height || 512);
  threeCamera.aspect = aspect;
  threeCamera.updateProjectionMatrix();
}

// ---------------------------------------------------------------------------
// Primitive helper
// ---------------------------------------------------------------------------
function atlasReadRenderTargetAsBase64(renderer, renderTarget, width, height, mime = "image/png", quality) {
  const buffer = new Uint8Array(width * height * 4);
  renderer.readRenderTargetPixels(renderTarget, 0, 0, width, height, buffer);

  // Flip Y (WebGL origin is bottom-left, canvas is top-left).
  const flipped = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    const srcRow = (height - 1 - y) * width * 4;
    const dstRow = y * width * 4;
    flipped.set(buffer.subarray(srcRow, srcRow + width * 4), dstRow);
  }

  const offscreen = document.createElement("canvas");
  offscreen.width = width;
  offscreen.height = height;
  const ctx = offscreen.getContext("2d");
  const imageData = ctx.createImageData(width, height);
  imageData.data.set(flipped);
  ctx.putImageData(imageData, 0, 0);
  return offscreen.toDataURL(mime, quality).split(",")[1];
}

function atlasRenderSceneToBase64(renderer, scene, camera, width, height, options = {}) {
  if (!THREE) return null;
  const renderTarget = options.renderTarget || new THREE.WebGLRenderTarget(width, height);
  const ownsRenderTarget = !options.renderTarget;
  const hasOverrideMaterial = Object.prototype.hasOwnProperty.call(options, "overrideMaterial");
  const prevOverrideMaterial = scene.overrideMaterial;

  try {
    if (hasOverrideMaterial) scene.overrideMaterial = options.overrideMaterial;
    renderer.setRenderTarget(renderTarget);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    return atlasReadRenderTargetAsBase64(renderer, renderTarget, width, height, options.mime, options.quality);
  } finally {
    renderer.setRenderTarget(null);
    if (hasOverrideMaterial) scene.overrideMaterial = prevOverrideMaterial;
    if (ownsRenderTarget) renderTarget.dispose();
  }
}

// ---------------------------------------------------------------------------
// Render all passes to base64-encoded PNG strings
// ---------------------------------------------------------------------------
async function renderAllPasses(renderer, scene, camera, width, height, exclude = []) {
  if (!THREE) return null;

  // The passes must contain geometry only: hide the background photo plane and
  // viewport helpers (grid) for every pass, restore after.
  const hidden = [];
  const hideList = [...exclude];
  scene.traverse((c) => { if (c.userData?.atlasHelper) hideList.push(c); });
  for (const obj of hideList) {
    if (obj && obj.visible) { obj.visible = false; hidden.push(obj); }
  }

  const rt = new THREE.WebGLRenderTarget(width, height);

  function renderToBase64(overrideMat) {
    const options = { renderTarget: rt };
    if (arguments.length) options.overrideMaterial = overrideMat;
    return atlasRenderSceneToBase64(renderer, scene, camera, width, height, options);
  }

  try {
    // Shaded: standard PBR render (or the projection material if 📽 is on)
    const shadedB64 = renderToBase64();

    // Depth: linear view-space depth normalised to the visible scene extent —
    // MeshDepthMaterial over the default 0.01..1000 range has no usable contrast.
    let far = 20;
    const tmpV = new THREE.Vector3();
    scene.traverse((c) => {
      if (c.isMesh && c.visible) {
        c.getWorldPosition(tmpV);
        far = Math.max(far, tmpV.distanceTo(camera.position) * 1.5);
      }
    });
    const depthMat = new THREE.ShaderMaterial({
      uniforms: { uFar: { value: far } },
      vertexShader: `
        varying float vViewZ;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          vViewZ = -mv.z;
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: `
        uniform float uFar;
        varying float vViewZ;
        void main() {
          float d = clamp(1.0 - vViewZ / uFar, 0.0, 1.0);
          gl_FragColor = vec4(d, d, d, 1.0);
        }`,
      side: THREE.DoubleSide,
    });
    const depthBg = scene.background;
    scene.background = new THREE.Color(0x000000);
    const depthB64 = renderToBase64(depthMat);
    scene.background = depthBg;
    depthMat.dispose();

    // Normal: custom ShaderMaterial. toneMapped:false — the exposure slider
    // must never alter these deterministic RGB-encoded normal values (the
    // custom depthMat above is unaffected regardless: it writes gl_FragColor
    // directly with no <tonemapping_fragment> chunk, so tone mapping never
    // applies to it in the first place).
    const normalMat = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide, toneMapped: false });
    const normalB64 = renderToBase64(normalMat);
    normalMat.dispose();

    // Mask: white geometry, black background. Also exposure-immune.
    const bg = scene.background;
    scene.background = new THREE.Color(0x000000);
    const maskMat = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, toneMapped: false });
    const maskB64 = renderToBase64(maskMat);
    scene.background = bg;
    maskMat.dispose();

    return { shaded: shadedB64, depth: depthB64, normal: normalB64, mask: maskB64 };
  } finally {
    hidden.forEach((o) => { o.visible = true; });
    rt.dispose();
  }
}

// ---------------------------------------------------------------------------
// Build the in-node UI (canvas + toolbar)
// ---------------------------------------------------------------------------
function buildNodeUI(node, containerEl) {
  if (!THREE) {
    containerEl.innerHTML = "<p style='color:#f88;padding:8px'>Three.js not available</p>";
    return;
  }

  // Output dimensions. These start square and are resized on execution to the
  // source image / ShotCam aspect. The visible canvas uses a capped preview
  // backbuffer with the same aspect, so UI responsiveness does not limit final
  // Render Proxy Passes or Camera Path proxy frames.
  node._atlasW = node._atlasW || node._atlasResolution || 768;
  node._atlasH = node._atlasH || node._atlasResolution || 768;
  let W = node._atlasW, H = node._atlasH;
  let previewSize = atlasViewportPreviewSize(W, H);
  let previewW = previewSize.width, previewH = previewSize.height;
  node._atlasPreviewW = previewW; node._atlasPreviewH = previewH;

  // Toolbar
  const toolbar = document.createElement("div");
  toolbar.style.cssText = "display:flex;gap:4px;padding:4px;background:#1a1a1a;flex-wrap:wrap";

  // Canvas, wrapped so the diagram SVG and metadata HUD can sit on top of it
  // without blocking orbit dragging (pointer-events:none on the overlays).
  //
  // flex:1;min-height:0 (canvasWrap, a flex child of `container` below) +
  // height:100% (canvas, of canvasWrap) deliberately does NOT derive layout
  // height from the canvas's own intrinsic width/height attributes (no
  // `height:auto`) — dragging the node's corner just gives canvasWrap more
  // flex space, which the canvas fills via a plain CSS/browser rescale of
  // whatever's already in its WebGL buffer. No JS resize hook is involved:
  // an earlier attempt hooked node.onResize to call resizeViewport (which
  // sets canvas.width/height) to re-render at the new size, but that fed
  // back into ComfyUI's own DOM-widget layout math (which WAS keyed off the
  // canvas's auto-derived height) and froze the tab. This CSS-only approach
  // can't create that loop since resizing never touches canvas.width/height.
  const canvasWrap = document.createElement("div");
  // min-width:0 overrides flexbox's default min-width:auto — without it, a
  // flex item's floor is its content's min-content size, and for a <canvas>
  // (a "replaced element") that's its INTRINSIC width (the `width` ATTRIBUTE,
  // e.g. 768px — `width:100%` in CSS only affects the USED size, not this
  // floor). That silently forced canvasWrap, and the node containing it, to
  // never shrink below the canvas's intrinsic pixel width regardless of the
  // node's actual size — surfacing as the node snapping/stretching wider on
  // the first interaction that triggered a relayout (e.g. mousedown to orbit).
  canvasWrap.style.cssText = "position:relative;width:100%;max-width:100%;align-self:stretch;flex:1;min-height:0;min-width:0;line-height:0;background:#111;overflow:hidden;";

  const canvas = document.createElement("canvas");
  canvas.width = previewW;
  canvas.height = previewH;
  // object-fit:contain letterboxes/pillarboxes the canvas's intrinsic
  // width/height (the capped preview backbuffer, with the same aspect a
  // ShotCam/source resolves to) within whatever box width:100%/height:100% gives it,
  // instead of stretching/squashing the WebGL content to fill a mismatched
  // container shape. `object-fit` applies to <canvas> like any other
  // replaced element and needs no JS — same CSS-only, no-new-resize-hook
  // constraint as the rest of this block (see the comment above canvasWrap).
  // KNOWN LIMITATION, not fixed here: the diagram/HUD SVG overlays below
  // are absolutely positioned to the full canvasWrap box (inset:0;100%),
  // so they'll misalign with the now-letterboxed canvas content whenever
  // its aspect doesn't match the container's — narrow (only visible with
  // 📊 Diagram/ℹ Info toggled on AND a significant aspect mismatch, e.g.
  // from AtlasDefineShotCam), left for a follow-up rather than risking a
  // flexbox+aspect-ratio rewrite in a spot with 3 prior documented bugs.
  canvas.style.cssText = "display:block;width:100%;height:100%;object-fit:contain;background:#111;cursor:grab";

  // Diagram overlay: layered VP / horizon / ground SVG, image-pixel-space
  // viewBox so it aligns with the source photo regardless of canvas size.
  const svgNS = "http://www.w3.org/2000/svg";
  const diagramSvg = document.createElementNS(svgNS, "svg");
  diagramSvg.setAttribute("viewBox", "0 0 1 1");
  diagramSvg.style.cssText = "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;display:none;";
  const gVpLines = document.createElementNS(svgNS, "g");
  const gHorizon = document.createElementNS(svgNS, "g");
  const gGround = document.createElementNS(svgNS, "g");
  gGround.style.opacity = "0.35"; gHorizon.style.opacity = "0.85"; gVpLines.style.opacity = "0.7";
  diagramSvg.append(gGround, gVpLines, gHorizon); // ground under, horizon on top

  // Metadata HUD: solved lens/distance/confidence readout.
  const metaHud = document.createElement("div");
  metaHud.style.cssText = "position:absolute;top:6px;left:6px;padding:6px 8px;background:rgba(10,10,14,0.72);" +
    "color:#cde;font:10px/1.5 monospace;border-radius:4px;pointer-events:none;white-space:pre;display:none;";

  const localControlsLayer = document.createElement("div");
  localControlsLayer.style.cssText =
    "position:absolute;left:0;right:0;bottom:0;z-index:8;display:flex;flex-direction:column;align-items:stretch;gap:0;" +
    "background:rgba(16,16,22,0.88);pointer-events:auto;line-height:normal;";

  canvasWrap.append(canvas, diagramSvg, metaHud, localControlsLayer);

  // Three.js setup
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setSize(previewW, previewH, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  // Exposure only has a visible effect with a tone-mapping operator active.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;

  function applyOutputProfilePreview(profile = {}) {
    const exposureStops = Number(profile.exposure ?? 0) || 0;
    const trim = Math.max(0, Number(profile.display_trim ?? 1) || 1);
    const gamma = Math.max(0.1, Number(profile.gamma ?? 1) || 1);
    renderer.toneMappingExposure = Math.pow(2, exposureStops) * trim;
    canvas.style.filter = gamma !== 1 ? `brightness(${trim}) contrast(${Math.max(0.1, 1 / gamma)})` : `brightness(${trim})`;
  }

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a1a);

  const camera = new THREE.PerspectiveCamera(60, W / H, 0.01, 1000);
  camera.position.set(0, 1.6, 5);
  camera.lookAt(0, 1, 0);

  // Lighting
  scene.add(new THREE.HemisphereLight(0xf5f0e8, 0x201810, 2.2));
  const key = new THREE.DirectionalLight(0xffffff, 1.4);
  key.position.set(4, 6, 3);
  scene.add(key);

  // Movable point lights (💡 Lights panel) — added alongside the fixed hemi/key
  // lights above, never replacing them. Default intensity 0 so no existing
  // workflow's look changes until an artist explicitly raises one; real
  // THREE.PointLights so they light the grey/shaded MeshStandardMaterial
  // preview and the "shaded" render pass exactly like any other scene light,
  // with zero extra wiring needed there. Their position/color/intensity also
  // drive a stylized multiply-only "relight" term in the projection shader
  // (see PROJECTION_FRAGMENT_SHADER) — kept in sync every frame by
  // syncProjectionLightUniforms() below rather than at material-creation time,
  // since projection materials are frequently rebuilt (every execution, every
  // patch/clean-plate source) and must not go stale.
  const movableLights = [
    new THREE.PointLight(0xffffff, 0, 0, 2),
    new THREE.PointLight(0xffffff, 0, 0, 2),
    new THREE.PointLight(0xffffff, 0, 0, 2),
  ];
  movableLights[0].position.set(2, 3, 2);
  movableLights[1].position.set(-2, 3, -2);
  movableLights[2].position.set(0, 4, 3);
  movableLights.forEach((l) => scene.add(l));
  // Place the (unmoved) relight lights NEAR the recovered geometry, scaled to
  // the scene, on each execution. The fixed near-origin defaults sit ~scene-
  // depth away at a large AtlasScaleOverride (geometry 100 m+), so the lights
  // never reach it and raising them does nothing (user-reported). We put each
  // light in front of + above the geometry pivot at ~0.36× the camera→pivot
  // distance (→ a strong-but-not-saturating relight through the scale-aware
  // atten). Respects manual placement: a light the artist has dragged (its
  // panel X/Y/Z edited → `atlasMoved`) is never repositioned.
  function placeDefaultLights() {
    // Pivot + scale from ALL projected geometry (derived proxies AND patch/clean-
    // plate meshes) — computeGeometryPivot deliberately excludes patch sources
    // and runs before they're built, so it can't be reused for this.
    const box = new THREE.Box3();
    let any = false;
    scene.traverse((o) => {
      if (o.isMesh && (o.userData.atlasPatch || o.userData.atlasDerived)) { box.expandByObject(o); any = true; }
    });
    if (!any || box.isEmpty()) return;
    const pivot = box.getCenter(new THREE.Vector3());
    // Scene scale → the 🎯 pivot-offset step, so a nudge stays proportional at
    // any AtlasScaleOverride (a 1m step is useless when geometry sits at 150m).
    lastSceneRadius = box.getSize(new THREE.Vector3()).length() * 0.5 || 10;
    if (pivotInputs) {
      const step = Math.max(lastSceneRadius / 40, 0.05).toPrecision(2);
      pivotInputs.forEach((inp) => { inp.step = step; });
    }
    const camPos = (recoveredData && recoveredData.camera_position)
      ? new THREE.Vector3(recoveredData.camera_position[0], recoveredData.camera_position[1], recoveredData.camera_position[2])
      : camera.position.clone();
    const toCam = camPos.clone().sub(pivot);
    const D = toCam.length() || 10;
    toCam.normalize();
    const up = new THREE.Vector3(0, 1, 0);
    let right = new THREE.Vector3().crossVectors(toCam, up);
    if (right.lengthSq() < 1e-6) right.set(1, 0, 0);
    right.normalize();
    const offs = [[0.18, 0.22, 0.22], [-0.18, 0.22, 0.22], [0.0, 0.28, 0.28]];
    movableLights.forEach((l, i) => {
      if (l.userData.atlasMoved) return;
      const o = offs[i] || offs[0];
      l.position.copy(pivot)
        .addScaledVector(right, o[0] * D)
        .addScaledVector(up, o[1] * D)
        .addScaledVector(toCam, o[2] * D);
      if (l._atlasInputs) {
        l._atlasInputs[0].value = l.position.x.toFixed(1);
        l._atlasInputs[1].value = l.position.y.toFixed(1);
        l._atlasInputs[2].value = l.position.z.toFixed(1);
      }
    });
  }
  let _lightsWereActive = false;
  // 🩻 hidden-geometry provenance overlay toggle — synced into every
  // projection material by the same live mechanism as the lights (materials
  // are rebuilt on every execution, so a set-once approach would go stale).
  let debugHiddenOn = false;
  let layerDebugOn = false; // 🎨 per-layer identity tint toggle
  let bumpStrength = 0;     // 💡 Lights panel "Detail" — photo-luminance relight bump
  let bumpScale = 8;        // 💡 Lights panel "Scale" — bump sampling offset (texels)
  function syncProjectionLightUniforms() {
    const active = movableLights.some((l) => l.intensity > 0) || debugHiddenOn
      || layerDebugOn || bumpStrength > 0;
    // Skip the traverse entirely while both lights have always been off (the
    // default), but still run once on the on->off transition so any material
    // that previously picked up a nonzero uLightNIntensity gets zeroed out.
    if (!active && !_lightsWereActive) return;
    _lightsWereActive = active;
    scene.traverse((obj) => {
      const mat = obj.material;
      if (!mat?.isShaderMaterial || !mat.uniforms?.uLight1Pos) return;
      movableLights.forEach((l, i) => {
        const n = i + 1;
        if (!mat.uniforms[`uLight${n}Pos`]) return; // material predates this light count
        mat.uniforms[`uLight${n}Pos`].value.copy(l.position);
        mat.uniforms[`uLight${n}Color`].value.copy(l.color);
        mat.uniforms[`uLight${n}Intensity`].value = l.intensity;
      });
      if (mat.uniforms.uDebugHidden) {
        mat.uniforms.uDebugHidden.value = debugHiddenOn ? 1 : 0;
      }
      if (mat.uniforms.uLayerDebug) {
        mat.uniforms.uLayerDebug.value = layerDebugOn ? 1 : 0;
      }
      if (mat.uniforms.uBumpStrength) {
        mat.uniforms.uBumpStrength.value = bumpStrength;
      }
      if (mat.uniforms.uBumpScale) {
        mat.uniforms.uBumpScale.value = bumpScale;
      }
    });
  }

  // Ground grid (viewport helper — excluded from render passes)
  const grid = new THREE.GridHelper(20, 20, 0x444444, 0x333333);
  grid.userData.atlasHelper = true;
  scene.add(grid);

  // Orbit controls (self-contained; see createOrbitControls).
  const controls = createOrbitControls(camera, canvas);
  controls.setTarget(new THREE.Vector3(0, 1, 0));
  controls.syncFromCamera();

  // Fly controls for Camera Path authoring (disabled until that mode is toggled on).
  const fly = createFlyControls(camera, canvas);

  // Background reference image (loaded after camera data is set)
  let bgMesh = null;
  // The exact recovered camera pose, stored so "Camera View" can snap back to it.
  let recoveredData = null;
  // Last geometry-derived orbit pivot (median-depth, from setProxies) — reused
  // by 📷 Camera View so a reset never regresses to the ground-point heuristic.
  let lastGeometryPivot = null;

  // 🎯 Manual orbit-pivot offset (world metres, session-only). Added on top of
  // whatever base pivot the auto-logic picks (geometry median-depth or the
  // ground-point fallback), so the artist can nudge the point the orbit swings
  // around — useful once AtlasScaleOverride pushes geometry out to 100m+ and the
  // auto centroid isn't where you want to look. `pivotBase` is the un-offset
  // pivot the auto-logic last set; `applyPivotOffset` re-targets base+offset live.
  const pivotOffset = new THREE.Vector3(0, 0, 0);
  let pivotBase = null;              // un-offset world pivot (set at every setTarget site)
  let lastSceneRadius = 10;          // geometry bounding radius — scales the panel step
  let pivotInputs = null;            // [x,y,z] <input> refs, for step rescaling on execute
  function targetWithOffset(base) {  // base (Vector3) + the manual offset
    pivotBase = base.clone();
    return base.clone().add(pivotOffset);
  }
  function applyPivotOffset() {      // live re-target when the offset changes
    if (!pivotBase) return;
    controls.setTarget(pivotBase.clone().add(pivotOffset));
    controls.syncFromCamera();
  }

  // Animation loop — assign to node._atlasRafId each frame so cancelAnimationFrame works.
  // The orbit/fly controllers update the camera on input events; pathPlayback
  // (set by the Camera Path "Play" button, below) drives it during path preview.
  let pathPlayback = null; // { startTime, durationSec, onDone }
  let applyPathPoseAtT = null; // assigned once the Camera Path block below runs
  let lastTickTime = performance.now();
  function animate() {
    node._atlasRafId = requestAnimationFrame(animate);
    const now = performance.now();
    const dt = Math.min(0.1, (now - lastTickTime) / 1000);
    lastTickTime = now;
    fly.tick(dt);
    controls.updateKeys();  // UE-style tracking keys (self-timed; no-op when idle)
    if (pathPlayback) {
      const t = Math.min(1, (now - pathPlayback.startTime) / 1000 / pathPlayback.durationSec);
      applyPathPoseAtT(t);
      if (t >= 1) { const done = pathPlayback.onDone; pathPlayback = null; done?.(); }
    }
    syncProjectionLightUniforms();
    // Deferred aspect snap: execution can finish while the node is scrolled
    // off-screen, where ComfyUI hides the DOM widget and every rect measures
    // 0 — snapNodeHeightToRenderAspect stashes the aspect instead, and this
    // retries it once the widget is visible/laid out again. No-op (one null
    // check) on every other frame.
    if (pendingSnapAspect != null) snapNodeHeightToRenderAspect(pendingSnapAspect);
    renderer.render(scene, camera);
  }
  node._atlasRafId = requestAnimationFrame(animate);

  // Camera View button — snap the orbit camera back to the recovered perspective.
  const camBtn = document.createElement("button");
  camBtn.textContent = "📷 Camera View";
  camBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2f3a;color:#cde;border:1px solid #456;border-radius:3px";
  camBtn.onclick = () => { if (recoveredData) applyRecoveredView(recoveredData, { force: true }); };
  toolbar.appendChild(camBtn);

  // 📽 Project toggle — camera-project the source photo onto ALL geometry
  // (derived proxies + patch/clean-plate sources) from the recovered camera.
  // Defaults ON (beta UX request): the projected photo is the product; the
  // grey mesh is the diagnostic view, reached by toggling 📽 OFF. Textures
  // load async, and every load-completion path already re-applies via
  // `if (projectionOn) applyProjection(true)` — starting true just makes
  // those paths fire when the first texture lands.
  let projectionOn = true;
  // See-through backdrop on/off (🕳 toggle below). When on (default) the
  // enlarged background-photo plane fills any pixel the projection discards
  // (matte silhouettes, tears, out-of-frame) so holes read as the photo, not
  // black; off restores the plain diagnostic view where discarded pixels are
  // black (and the enlarged plane's edge-smear is gone).
  let seeThroughOn = true;
  let projMaterial = null;
  const projBtn = document.createElement("button");
  projBtn.textContent = "📽 Project";
  projBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#3a2a5a;color:#dcf;border:1px solid #546;border-radius:3px";

  function isProjectable(c) {
    if (!c.isMesh || c === bgMesh) return false;
    // atlasUserGeo/atlasProxy branches removed with the primitive/proxy
    // buttons (2026-07-09) — only Python-derived geometry and patch sources
    // exist in the scene now.
    return !!(c.userData?.atlasDerived || c.userData?.atlasPatch);
  }

  function applyProjection(on) {
    scene.traverse((c) => {
      if (!isProjectable(c)) return;
      // Patch meshes carry their OWN projection material (their source's
      // camera+image+facing mask); everything else uses the shared primary one.
      const mat = c.userData._projMaterial || projMaterial;
      if (on && mat) {
        // Stash the ORIGINAL material only once — re-applying with a rebuilt
        // projection material must not overwrite it with the stale one.
        if (!c.userData._prevMaterial) c.userData._prevMaterial = c.material;
        c.material = mat;
      } else if (c.userData._prevMaterial) {
        c.material = c.userData._prevMaterial;
        delete c.userData._prevMaterial;
      }
    });
    // See-through backdrop: the background photo plane STAYS visible under
    // 📽 Project (renderOrder -100000, depthTest false) so it fills any pixel the
    // projection discards (matte silhouettes, tears, out-of-frame) with the photo
    // instead of black — the projected geometry draws on top of it everywhere it
    // actually paints, so it only shows through in the holes. Gated by the 🕳
    // See-through toggle under Project; in the grey (Project OFF) view the plane
    // is the plain photo backdrop and always shows. Hidden only during the
    // deterministic export passes (renderAllPasses / Safe Zone probe).
    if (bgMesh) bgMesh.visible = on ? seeThroughOn : true;
  }

  projBtn.onclick = () => {
    if (!projMaterial) return; // no solve/texture yet
    projectionOn = !projectionOn;
    projBtn.style.background = projectionOn ? "#3a2a5a" : "#2a2a3a";
    applyProjection(projectionOn);
  };
  toolbar.appendChild(projBtn);

  // 🎬 Backdrop toggle — every primitive-fitting derivation strategy
  // (azimuth_walls, vertical_extrusion, ransac_planes, room_cuboid — never
  // relief_mesh) always emits one extra flat "projection_backdrop" plane
  // (proxy_geometry.py / depth_geometry.build_backdrop_primitive) sized to
  // cover the whole frustum at the far-depth percentile, as a catch-all so
  // 📽 Project never shows raw background behind the fitted primitives. When
  // geometry_mode is "both" (relief_mesh + primitives) that backdrop plane
  // is also projectable and sits behind/around the actual relief mesh,
  // receiving its own copy of the projected texture — this hides it (a
  // plain visibility toggle handles both the grey preview AND 📽 Project,
  // since an invisible mesh never renders regardless of material) so
  // Project only paints the generated mesh. Re-applied in setProxies()
  // below since buildDerivedProxies rebuilds fresh mesh objects (default
  // visible=true) on every execution.
  let backdropVisible = true;
  const backdropBtn = document.createElement("button");
  backdropBtn.textContent = "🎬 Backdrop";
  backdropBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  function setBackdropVisible(v) {
    backdropVisible = v;
    backdropBtn.style.background = v ? "#2a2a2a" : "#3a1a1a";
    backdropBtn.style.color = v ? "#ddd" : "#faa";
    scene.traverse((c) => { if (c.name === "projection_backdrop") c.visible = v; });
  }
  backdropBtn.onclick = () => setBackdropVisible(!backdropVisible);
  toolbar.appendChild(backdropBtn);

  // 🕳 See-through toggle — governs the enlarged background-photo plane (bgMesh)
  // that fills discarded projection pixels under 📽 Project (see applyProjection).
  // ON (default) = holes show the softened photo backdrop instead of black; OFF
  // = plain view, discarded pixels are black and the enlarged plane's edge-smear
  // is gone. Distinct from 🎬 Backdrop (that governs the flat projection_backdrop
  // primitive). Display-only, session state, like every other viewport toggle.
  const seeThruBtn = document.createElement("button");
  seeThruBtn.textContent = "🕳 See-through";
  seeThruBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a3a2a;color:#cfc;border:1px solid #465;border-radius:3px";
  function setSeeThrough(v) {
    seeThroughOn = v;
    seeThruBtn.style.background = v ? "#2a3a2a" : "#3a1a1a";
    seeThruBtn.style.color = v ? "#cfc" : "#faa";
    if (bgMesh) bgMesh.visible = projectionOn ? seeThroughOn : true;
  }
  seeThruBtn.onclick = () => setSeeThrough(!seeThroughOn);
  toolbar.appendChild(seeThruBtn);

  // 📏 Band Box overlay — a translucent red box around the AtlasBoundedBand
  // FOREGROUND: the clean-plate layer whose far_m is FINITE is the one the
  // bounded band clipped at the cutoff (near + N·W); its axis-aligned bounds
  // show exactly where the foreground relief is capped and where the sky card
  // falls back behind it. Session-only display state, rebuilt each execution.
  let bandBoxOn = false;
  let bandBox = null;
  function disposeBandBox() {
    if (!bandBox) return;
    scene.remove(bandBox);
    bandBox.traverse((o) => { o.geometry?.dispose?.(); o.material?.map?.dispose?.(); o.material?.dispose?.(); });
    bandBox = null;
  }
  // A camera-facing text sprite (canvas texture — self-contained, no font/CSS2D
  // loader needed) used to label the cutoff distance on the 📏 Band Box.
  function makeBandLabel(text, worldHeight = 0.9, color = 0xff2020) {
    const r = (color >> 16) & 255, g = (color >> 8) & 255, b = color & 255;
    const canvas = document.createElement("canvas");
    let ctx = canvas.getContext("2d");
    const fontPx = 48;
    ctx.font = `bold ${fontPx}px sans-serif`;
    const w = Math.ceil(ctx.measureText(text).width) + 40;
    const h = fontPx + 28;
    canvas.width = w; canvas.height = h;
    ctx = canvas.getContext("2d");         // resizing the canvas clears state
    ctx.font = `bold ${fontPx}px sans-serif`;
    // Darkened box color as the background, bright box color as the border, white
    // text — so any palette color stays legible and matches its box.
    ctx.fillStyle = `rgba(${(r * 0.42) | 0},${(g * 0.42) | 0},${(b * 0.42) | 0},0.9)`;
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = `rgba(${r},${g},${b},0.95)`; ctx.lineWidth = 4;
    ctx.strokeRect(2, 2, w - 4, h - 4);
    ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(text, w / 2, h / 2 + 2);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace; tex.needsUpdate = true;
    const spr = new THREE.Sprite(new THREE.SpriteMaterial({
      map: tex, depthTest: false, transparent: true }));
    spr.scale.set(worldHeight * (w / h), worldHeight, 1);
    spr.renderOrder = 100003;
    return spr;
  }
  // Build ONE bounded-band box (cage + cutoff plane + distance label) for a
  // patch group, into `parent`. Geometry is emitted in `M`'s VIEW space (so the
  // back face lands exactly on the cutoff plane at any camera pitch) when M is
  // given, else in world space; the caller applies cam->world once to `parent`.
  function addBandBoxFor(fg, parent, M, fillOp, planeOp, color) {
    const cutoff = Math.abs(fg.userData.far_m);
    const wbox = new THREE.Box3().setFromObject(fg);
    if (wbox.isEmpty()) return;
    let boxGeo = null, cutGeo = null, labelPos = null;
    if (M) {
      const vb = new THREE.Box3();           // fg AABB corners in view space
      const mn = wbox.min, mx = wbox.max;
      for (let i = 0; i < 8; i++) {
        vb.expandByPoint(new THREE.Vector3(
          (i & 1) ? mx.x : mn.x, (i & 2) ? mx.y : mn.y, (i & 4) ? mx.z : mn.z).applyMatrix4(M));
      }
      const nearZ = vb.max.z, farZ = -cutoff;              // camera looks -Z
      const zLo = Math.min(nearZ, farZ), zHi = Math.max(nearZ, farZ);
      const cx = (vb.min.x + vb.max.x) / 2, cy = (vb.min.y + vb.max.y) / 2, cz = (zLo + zHi) / 2;
      const sx = Math.max(vb.max.x - vb.min.x, 1e-3), sy = Math.max(vb.max.y - vb.min.y, 1e-3);
      boxGeo = new THREE.BoxGeometry(sx, sy, Math.max(zHi - zLo, 1e-3)); boxGeo.translate(cx, cy, cz);
      cutGeo = new THREE.PlaneGeometry(sx, sy); cutGeo.translate(cx, cy, farZ);
      labelPos = new THREE.Vector3(cx, vb.max.y, farZ);    // top of the cutoff plane
    } else {
      const size = new THREE.Vector3(); wbox.getSize(size);
      const center = new THREE.Vector3(); wbox.getCenter(center);
      boxGeo = new THREE.BoxGeometry(Math.max(size.x, 1e-3), Math.max(size.y, 1e-3), Math.max(size.z, 1e-3));
      boxGeo.translate(center.x, center.y, center.z);
      labelPos = new THREE.Vector3(center.x, center.y + size.y / 2, center.z);
    }
    const fill = new THREE.Mesh(boxGeo, new THREE.MeshBasicMaterial({
      color: color, transparent: true, opacity: fillOp, side: THREE.DoubleSide, depthWrite: false }));
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(boxGeo),
      new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.95, depthTest: false }));
    fill.renderOrder = 100001; edges.renderOrder = 100002;
    parent.add(fill); parent.add(edges);
    if (cutGeo) {
      const cut = new THREE.Mesh(cutGeo, new THREE.MeshBasicMaterial({
        color: color, transparent: true, opacity: planeOp, side: THREE.DoubleSide, depthWrite: false }));
      cut.renderOrder = 100001; parent.add(cut);
    }
    const label = makeBandLabel(`cutoff ${cutoff.toFixed(1)} m`, 0.9, color);
    label.position.copy(labelPos); parent.add(label);
  }
  function buildBandBox() {
    disposeBandBox();
    if (!bandBoxOn || !THREE) return;
    // EVERY bounded foreground layer = a patch group with a FINITE far_m (a
    // clean-plate layer the bounded band clipped at its own cutoff). Box EACH,
    // so a multi-plane matte (one fg layer per building/object) shows one red
    // cage + cutoff label per layer. The background card's far_m is null/+inf.
    const bounded = [];
    scene.traverse((c) => {
      if (c.userData?.atlasPatchGroup && typeof c.userData.far_m === "number" && isFinite(c.userData.far_m)) bounded.push(c);
    });
    if (!bounded.length) return; // no bounded band in this scene — nothing to box
    bounded.sort((a, b) => a.userData.far_m - b.userData.far_m); // near -> far, for stable colors
    scene.updateMatrixWorld(true);
    // Build every box in the RECOVERED camera's frame so each back face lands on
    // its own cutoff plane regardless of camera pitch; one cam->world applied to
    // the shared parent. Falls back to world-space AABBs if no view matrix.
    const vm = recoveredData && recoveredData.view_matrix;
    let M = null, place = null;
    if (vm && vm.length === 4) {
      M = new THREE.Matrix4().set(
        vm[0][0], vm[0][1], vm[0][2], vm[0][3],
        vm[1][0], vm[1][1], vm[1][2], vm[1][3],
        vm[2][0], vm[2][1], vm[2][2], vm[2][3],
        vm[3][0], vm[3][1], vm[3][2], vm[3][3]);
      place = M.clone().invert();
    }
    bandBox = new THREE.Group();
    bandBox.name = "atlas_band_box";
    // Frame-spanning band boxes stack, so scale the fill/plane opacity down with
    // the count — one box stays bold, three read light so the scene shows through
    // (the always-visible edges + cutoff plane + label still define each).
    const N = bounded.length;
    const fillOp = Math.min(0.13, 0.16 / N), planeOp = Math.min(0.28, 0.42 / N);
    // Distinct color per box (by depth: near -> far). A single box is red, matching
    // the original; multiple bands get their own hue so they're tellable apart.
    const PALETTE = [0xff3838, 0xffb020, 0x30c8ff, 0x44e05a, 0xb060ff, 0xf5e030];
    bounded.forEach((fg, i) => addBandBoxFor(fg, bandBox, M, fillOp, planeOp, PALETTE[i % PALETTE.length]));
    if (place) bandBox.applyMatrix4(place);
    scene.add(bandBox);
  }
  const bandBoxBtn = document.createElement("button");
  bandBoxBtn.textContent = "📏 Band Box";
  bandBoxBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  bandBoxBtn.onclick = () => {
    bandBoxOn = !bandBoxOn;
    bandBoxBtn.style.background = bandBoxOn ? "#3a1a1a" : "#2a2a2a";
    bandBoxBtn.style.color = bandBoxOn ? "#f88" : "#ddd";
    buildBandBox();
  };
  toolbar.appendChild(bandBoxBtn);

  // 🩻 X-ray provenance overlay — tints the surface region whose depth was
  // SUBSTITUTED by AtlasPredictHiddenGeometry (red = LaRI, blue = World
  // Tracing) at 50% over the projected photo. Only visible under 📽 Project
  // (the tint lives in the projection shader) and only when a hidden-geometry
  // workflow threaded a hidden_mask into a ProjectionSource.
  const dbgHiddenBtn = document.createElement("button");
  dbgHiddenBtn.textContent = "🩻 X-ray";
  dbgHiddenBtn.title = "Highlight predicted hidden geometry (red = LaRI, blue = World Tracing)";
  dbgHiddenBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  dbgHiddenBtn.onclick = () => {
    debugHiddenOn = !debugHiddenOn;
    dbgHiddenBtn.style.background = debugHiddenOn ? "#4a1a2a" : "#2a2a2a";
    dbgHiddenBtn.style.color = debugHiddenOn ? "#fac" : "#ddd";
  };
  toolbar.appendChild(dbgHiddenBtn);

  // 🎨 Layers — per-layer identity overlay: tints EVERYTHING each projection
  // source paints with its own color (base/primary teal; each
  // ProjectionSource takes the module palette by index), with an on-canvas
  // legend of layer names. Generalizes "show fg / mid / bg" to any layer
  // stack. Projection-mode only, like 🩻 — same live uniform sync.
  const layerLegend = document.createElement("div");
  layerLegend.style.cssText = "position:absolute;left:6px;bottom:6px;padding:6px 8px;" +
    "background:rgba(10,10,14,0.78);color:#cde;font:10px/1.6 monospace;" +
    "border-radius:4px;pointer-events:none;display:none;z-index:7;";
  canvasWrap.appendChild(layerLegend);
  function refreshLayerLegend() {
    const hex = (c) => "#" + c.toString(16).padStart(6, "0");
    const rows = [[hex(LAYER_DEBUG_PRIMARY), "base mesh + backdrop (primary)"]];
    (recoveredData?.projection_sources || []).forEach((s, i) => {
      rows.push([hex(LAYER_DEBUG_PALETTE[i % LAYER_DEBUG_PALETTE.length]),
                 s.name || `layer ${i}`]);
    });
    layerLegend.replaceChildren(...rows.map(([c, label]) => {
      const row = document.createElement("div");
      const sw = document.createElement("span");
      sw.style.cssText = `display:inline-block;width:10px;height:10px;border-radius:2px;` +
        `background:${c};margin-right:6px;vertical-align:middle;`;
      row.append(sw, document.createTextNode(label));
      return row;
    }));
  }
  const layerBtn = document.createElement("button");
  layerBtn.textContent = "🎨 Layers";
  layerBtn.title = "Tint each projection layer a distinct color (with legend)";
  layerBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  layerBtn.onclick = () => {
    layerDebugOn = !layerDebugOn;
    layerBtn.style.background = layerDebugOn ? "#2a3a1a" : "#2a2a2a";
    layerBtn.style.color = layerDebugOn ? "#cfa" : "#ddd";
    if (layerDebugOn) refreshLayerLegend();
    layerLegend.style.display = layerDebugOn ? "block" : "none";
  };
  toolbar.appendChild(layerBtn);

  // ---------------------------------------------------------------------------
  // 📐 Extract Angle — orbit/fly to the view you want a patch generated at
  // (e.g. the last frame of an intended camera move, MPTK style), click, and
  // the orbit delta from the RECOVERED camera is measured about the payload's
  // `orbit_pivot` (camera_math.ground_lookat_pivot — the SAME pivot
  // orbit_camera uses backend-side, NOT this viewport's own geometry-centroid
  // orbit pivot, so the result round-trips exactly through
  // AtlasAddPatchView/AtlasOcclusionMask's camera construction), snapped to
  // the Qwen Multiple-Angles LoRA's nearest named views, written into
  // client_data.patch_angle, and re-queued so the node's four STRING outputs
  // (patch_azimuth_view/patch_elevation_view/patch_distance/patch_prompt) go
  // live. Assumes the source photo is "front view"/"eye-level shot" (set
  // source_* downstream accordingly) and measures in the true world frame —
  // leave flip_azimuth OFF downstream for extracted angles.
  //
  // ATLAS_NAMED_VIEWS mirrors nodes.py's _AZIMUTH_VIEWS/_ELEVATION_VIEWS/
  // _DISTANCE_VIEWS — same accepted hand-sync duplication as
  // SCENE_TYPE_PRESETS in atlas_derive_geometry.js and catmullRom3JS here;
  // keep all three tables in sync with nodes.py by hand.
  // ---------------------------------------------------------------------------
  const ATLAS_AZIMUTH_VIEWS = [
    ["front view", 0], ["front-right quarter view", 45], ["right side view", 90],
    ["back-right quarter view", 135], ["back view", 180], ["back-left quarter view", 225],
    ["left side view", 270], ["front-left quarter view", 315],
  ];
  const ATLAS_ELEVATION_VIEWS = [
    ["low-angle shot", -30], ["eye-level shot", 0], ["elevated shot", 30], ["high-angle shot", 60],
  ];
  const ATLAS_DISTANCE_VIEWS = [["close-up", 0.6], ["medium shot", 1.0], ["wide shot", 1.8]];

  const angleHud = document.createElement("div");
  angleHud.style.cssText = "position:absolute;top:6px;right:6px;padding:6px 8px;background:rgba(10,10,14,0.82);" +
    "color:#dec;font:10px/1.5 monospace;border-radius:4px;pointer-events:auto;white-space:pre;display:none;z-index:9;";
  canvasWrap.appendChild(angleHud);

  function extractPatchAngle() {
    if (!recoveredData?.camera_position) return null;
    const pv = recoveredData.orbit_pivot;
    if (!pv) return { error: "no orbit_pivot in payload — re-queue the graph once to refresh" };
    const pivot = new THREE.Vector3(pv[0], pv[1], pv[2]);
    const p0 = recoveredData.camera_position;
    const o0 = new THREE.Vector3(p0[0], p0[1], p0[2]).sub(pivot);
    const o1 = camera.position.clone().sub(pivot);
    const r0 = Math.max(o0.length(), 1e-9);
    const r1 = Math.max(o1.length(), 1e-9);
    // Mirrors camera_math.orbit_camera exactly: azimuth = atan2(x, z) about
    // world +Y, elevation = asin(y / r), radius scaled by distance_scale.
    const az0 = Math.atan2(o0.x, o0.z), az1 = Math.atan2(o1.x, o1.z);
    const el0 = Math.asin(Math.max(-1, Math.min(1, o0.y / r0)));
    const el1 = Math.asin(Math.max(-1, Math.min(1, o1.y / r1)));
    const wrapDeg = (d) => ((d + 180) % 360 + 360) % 360 - 180;
    const dAz = wrapDeg((az1 - az0) * 180 / Math.PI);
    const dEl = (el1 - el0) * 180 / Math.PI;
    const distScale = r1 / r0;

    // Snap to the LoRA's absolute named views, assuming source = front view /
    // eye-level shot (patch = source + delta). DIRECTIONAL snapping: beyond a
    // small deadband, always advance at least one named view IN THE DIRECTION
    // of the orbit — never collapse back to the source view. Nearest-snap
    // rounded a deliberate 15° orbit back to "front view" (the azimuth grid
    // is 45°), generating a patch identical to the source photo (found live).
    const AZ_DEADBAND = 5, EL_DEADBAND = 10;
    let azTargetDeg = 0;
    if (Math.abs(dAz) >= AZ_DEADBAND) {
      azTargetDeg = Math.sign(dAz) * 45 * Math.max(1, Math.round(Math.abs(dAz) / 45));
    }
    const patchAzAbs = ((azTargetDeg % 360) + 360) % 360;
    let azName = ATLAS_AZIMUTH_VIEWS[0];
    for (const [name, deg] of ATLAS_AZIMUTH_VIEWS) {
      if (deg === patchAzAbs) { azName = [name, deg]; break; }
    }
    const azErr = Math.abs(wrapDeg(azTargetDeg - dAz));

    // Elevation views sit at -30/0/30/60: same outward rule (one negative
    // step available, two positive).
    let elTargetDeg = 0;
    if (Math.abs(dEl) >= EL_DEADBAND) {
      elTargetDeg = dEl > 0 ? (dEl < 45 ? 30 : 60) : -30;
    }
    let elName = ATLAS_ELEVATION_VIEWS[1];
    for (const [name, deg] of ATLAS_ELEVATION_VIEWS) {
      if (deg === elTargetDeg) { elName = [name, deg]; break; }
    }
    const elErr = Math.abs(dEl - elTargetDeg);
    let distName = ATLAS_DISTANCE_VIEWS[1], distErr = 1e9;
    for (const [name, s] of ATLAS_DISTANCE_VIEWS) {
      const err = Math.abs(Math.log(distScale / s)); // nearest in log space
      if (err < distErr) { distErr = err; distName = [name, s]; }
    }
    const prompt = `<sks> ${azName[0]} ${elName[0]} ${distName[0]}`;
    return {
      dAz, dEl, distScale,
      azimuth_view: azName[0], azSnapDeg: azName[1], azErr,
      elevation_view: elName[0], elSnapDeg: elName[1], elErr,
      distance_view: distName[0], distSnapScale: distName[1],
      prompt,
    };
  }

  function persistPatchAngleToClientData(r) {
    const widget = node.widgets?.find((w) => w.name === "client_data");
    if (!widget) return;
    let existing = {};
    try { existing = widget.value ? JSON.parse(widget.value) : {}; } catch (_) { existing = {}; }
    // Merge (like camera_path / render passes) so the buttons never clobber
    // each other's results.
    existing.patch_angle = {
      azimuth_view: r.azimuth_view,
      elevation_view: r.elevation_view,
      distance_view: r.distance_view,
      prompt: r.prompt,
      raw: { d_azimuth_deg: r.dAz, d_elevation_deg: r.dEl, distance_scale: r.distScale },
      // Identity of the solve+image this was extracted FROM — the backend
      // re-arms the patch-branch pause when it no longer matches (e.g. the
      // artist swapped the input photo), instead of running a stale angle.
      fingerprint: recoveredData?.solve_fingerprint || "",
    };
    widget.value = JSON.stringify(existing);
    widget.callback?.(widget.value);
  }

  const angleBtn = document.createElement("button");
  angleBtn.textContent = "📐 Extract Angle";
  angleBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  angleBtn.title = "Orbit/fly to the view you want a patch at, then click: measures the orbit " +
    "delta from the recovered camera, snaps it to the Qwen Multiple-Angles named views, and " +
    "re-queues so the patch_* STRING outputs go live.";
  angleBtn.onclick = () => {
    const r = extractPatchAngle();
    if (!r) { angleHud.textContent = "(no solve yet — queue the graph first)"; angleHud.style.display = "block"; return; }
    if (r.error) { angleHud.textContent = r.error; angleHud.style.display = "block"; return; }
    const f1 = (v) => (v >= 0 ? "+" : "") + v.toFixed(1);
    // Zero-orbit extraction = the patch will just reproduce the source photo.
    // The LoRA's named views snap on a 45° azimuth grid, so any orbit under
    // ±22.5° lands back on "front view" — and an execution used to reset the
    // camera to the recovered pose, making accidental zero-orbit extractions
    // easy (found live). Warn loudly instead of silently generating a no-op.
    const zeroOrbit = r.azimuth_view === "front view"
      && r.elevation_view === "eye-level shot" && r.distance_view === "medium shot";
    const warn = zeroOrbit
      ? `\n⚠ ZERO-ORBIT: the camera is within the snap deadband of the\n` +
        `source view — the generated patch would just match the photo.\n` +
        `Orbit deliberately (any move past ~5° advances to the next\n` +
        `named view in that direction) and click 📐 again.`
      : "";
    angleHud.textContent =
      `📐 Patch angle (source = front view)\n` +
      `Δaz  ${f1(r.dAz)}°  → ${r.azimuth_view} (${r.azErr.toFixed(0)}° off)\n` +
      `Δel  ${f1(r.dEl)}°  → ${r.elevation_view} (${r.elErr.toFixed(0)}° off)\n` +
      `dist ×${r.distScale.toFixed(2)} → ${r.distance_view}\n` +
      `${r.prompt}${warn}\n` +
      `(re-queued — patch_* outputs are live)      [✕]`;
    angleHud.style.display = "block";
    angleHud.onclick = (e) => { angleHud.style.display = "none"; e.stopPropagation(); };
    persistPatchAngleToClientData(r);
    app.queuePrompt(0, 1);
  };
  toolbar.appendChild(angleBtn);

  // ---------------------------------------------------------------------------
  // 🧭 Safe Zone — MEASURE the scene's actual safe camera envelope and clamp
  // the orbit to it, so the artist cannot move into holes at all. This is the
  // no-diffusion MVP answer to coverage: instead of generating patches for
  // unseen areas, restrict the move to what the projection actually covers.
  // Method: probe renders with the projection materials active into a small
  // offscreen target whose clear color is a pure-magenta sentinel — every
  // pixel the projection discards (out-of-frame, matte, facing, tears) shows
  // the sentinel, so counting magenta pixels IS the exact per-pose hole
  // fraction as the real renderer sees it, every shader rule included. Scan
  // each direction from the recovered pose in 2.5° steps until the hole
  // fraction exceeds baseline + 0.4%, then clamp the orbit controller to the
  // measured arc. Envelope persists in client_data with the solve
  // fingerprint (same staleness rule as 📐 extractions).
  function renderProbe(probeCam) {
    const W = 160;
    const H = Math.max(8, Math.round(W / (camera.aspect || 1.7778)));
    if (!node._atlasProbeRT || node._atlasProbeRT.width !== W || node._atlasProbeRT.height !== H) {
      node._atlasProbeRT?.dispose();
      node._atlasProbeRT = new THREE.WebGLRenderTarget(W, H);
    }
    const rt = node._atlasProbeRT;
    const prevTarget = renderer.getRenderTarget();
    const prevColor = new THREE.Color();
    renderer.getClearColor(prevColor);
    const prevAlpha = renderer.getClearAlpha();
    const gridWas = grid.visible;
    const bgWas = bgMesh ? bgMesh.visible : false;
    // scene.background overrides the clear color at render() time — it was
    // silently repainting every probe frame #1a1a1a, burying the sentinel
    // (found live: baseline read 0 holes at every angle, so the scan always
    // ran to the hard max and the clamp never changed). Null it for the probe.
    const sceneBgWas = scene.background;
    scene.background = null;
    grid.visible = false;
    if (bgMesh) bgMesh.visible = false;
    try {
      probeCam.aspect = W / H;
      probeCam.updateProjectionMatrix();
      renderer.setRenderTarget(rt);
      renderer.setClearColor(0xff00ff, 1);
      renderer.clear();
      renderer.render(scene, probeCam);
      const buf = new Uint8Array(W * H * 4);
      renderer.readRenderTargetPixels(rt, 0, 0, W, H, buf);
      let holes = 0;
      for (let i = 0; i < buf.length; i += 4) {
        if (buf[i] > 240 && buf[i + 1] < 16 && buf[i + 2] > 240) holes++;
      }
      return holes / (W * H);
    } finally {
      renderer.setRenderTarget(prevTarget);
      renderer.setClearColor(prevColor, prevAlpha);
      grid.visible = gridWas;
      if (bgMesh) bgMesh.visible = bgWas;
      scene.background = sceneBgWas;
    }
  }

  function measureHoleFractionAt(dTheta, dPhi) {
    const f = controls.getFrame();
    const th = f.theta0 + dTheta;
    const ph = Math.min(Math.PI - 0.05, Math.max(0.05, f.phi0 + dPhi));
    const probeCam = camera.clone();
    probeCam.position.set(
      f.target.x + f.radius * Math.sin(ph) * Math.sin(th),
      f.target.y + f.radius * Math.cos(ph),
      f.target.z + f.radius * Math.sin(ph) * Math.cos(th));
    probeCam.up.set(0, 1, 0);
    probeCam.lookAt(f.target);
    probeCam.updateMatrixWorld(true);
    return renderProbe(probeCam);
  }

  function scanDirection(fn, hardMaxDeg, tol) {
    // Linear 1° scan (not binary search): hole fraction need not be
    // monotonic in angle, probes are ~7ms at 160px, and the coarser 2.5°
    // step measurably undersold real limits (a 4.3° true limit read as
    // 2.5° — verified live against the fine-grained hole curve).
    let lastGood = 0;
    for (let a = 1; a <= hardMaxDeg + 1e-6; a += 1) {
      if (fn(THREE.MathUtils.degToRad(a)) > tol) break;
      lastGood = a;
    }
    return lastGood;
  }

  function findSafeEnvelope() {
    if (!projMaterial) return null;
    const wasOn = projectionOn;
    if (!wasOn) applyProjection(true);
    try {
      const baseline = measureHoleFractionAt(0, 0);
      // Allow 2% of the frame beyond baseline before calling a pose unsafe —
      // the baseline itself is nonzero on torn meshes (~4% on the hangar),
      // and sub-2% hole slivers read as minor edge artifacts, not failures.
      const tol = baseline + 0.02;
      return {
        baseline,
        yawPlusDeg: scanDirection((r) => measureHoleFractionAt(+r, 0), 80, tol),
        yawMinusDeg: scanDirection((r) => measureHoleFractionAt(-r, 0), 80, tol),
        phiPlusDeg: scanDirection((r) => measureHoleFractionAt(0, +r), 55, tol),
        phiMinusDeg: scanDirection((r) => measureHoleFractionAt(0, -r), 55, tol),
      };
    } finally {
      if (!wasOn) applyProjection(false);
    }
  }

  function applyEnvelopeLimits(env) {
    controls.setLimits({
      thetaMin: -THREE.MathUtils.degToRad(env.yawMinusDeg),
      thetaMax: THREE.MathUtils.degToRad(env.yawPlusDeg),
      phiMin: -THREE.MathUtils.degToRad(env.phiMinusDeg),
      phiMax: THREE.MathUtils.degToRad(env.phiPlusDeg),
    });
  }

  function persistEnvelopeToClientData(env) {
    const widget = node.widgets?.find((w) => w.name === "client_data");
    if (!widget) return;
    let existing = {};
    try { existing = widget.value ? JSON.parse(widget.value) : {}; } catch (_) { existing = {}; }
    existing.envelope = { ...env, fingerprint: recoveredData?.solve_fingerprint || "" };
    widget.value = JSON.stringify(existing);
    widget.callback?.(widget.value);
  }

  // Debug surface (console): node._atlasProbe(dThetaRad, dPhiRad) -> hole
  // fraction; node._atlasScene/_atlasCamera for inspection.
  node._atlasProbe = measureHoleFractionAt;
  node._atlasScene = scene;
  node._atlasCamera = camera;

  const envBtn = document.createElement("button");
  envBtn.textContent = "🧭 Safe Zone";
  envBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  envBtn.title = "Measure this scene's actual safe camera envelope (probe renders count " +
    "projection holes per pose) and clamp orbiting to it — the no-patch way to guarantee " +
    "a hole-free camera move.";
  envBtn.onclick = () => {
    const env = findSafeEnvelope();
    if (!env) {
      angleHud.textContent = "(no solve yet — queue the graph first)";
      angleHud.style.display = "block";
      return;
    }
    applyEnvelopeLimits(env);
    persistEnvelopeToClientData(env);
    angleHud.textContent =
      `🧭 Safe camera envelope (measured, holes ≤ ${(env.baseline * 100 + 2).toFixed(1)}%)
` +
      `yaw   +${env.yawPlusDeg.toFixed(1)}° / −${env.yawMinusDeg.toFixed(1)}°
` +
      `pitch +${env.phiMinusDeg.toFixed(1)}° up / −${env.phiPlusDeg.toFixed(1)}° down
` +
      `Orbit is now clamped to this zone. Keep camera-path
` +
      `moves inside these angles for a hole-free shot.      [✕]`;
    angleHud.style.display = "block";
    angleHud.onclick = (e) => { angleHud.style.display = "none"; e.stopPropagation(); };
  };
  toolbar.appendChild(envBtn);

  // ---------------------------------------------------------------------------
  // 🎥 Camera Path — author a keyframed camera move (fly nav, unclamped) to
  // test how 📽 Project holds up while the camera moves, then bake it to an
  // IMAGE batch (path_frames) for a core Video Combine node, or hand the raw
  // keyframes (camera_path) to AtlasExportCameraPathUSD for a DCC-facing
  // animated camera. See camera_path.py's sample_camera_path — the functions
  // below (catmullRom3JS/applyEasingJS/sampleKeyframePoseAtFrame) MUST stay in
  // sync with it; they exist here (rather than round-tripping to Python) so
  // Play can scrub live at 60fps.
  // ---------------------------------------------------------------------------
  let pathMode = false;
  let pathKeyframes = []; // [{frame_index, position:{x,y,z}, target:{x,y,z}, up:{x,y,z}, easing}]
  let pathFrameCount = 48;
  let pathFps = 24;
  const pathGroup = new THREE.Group();
  pathGroup.userData.atlasHelper = true; // excluded from render passes like the grid
  pathGroup.visible = false;
  scene.add(pathGroup);

  function catmullRom3JS(p0, p1, p2, p3, t) {
    const t2 = t * t, t3 = t2 * t;
    const f = (a, b, c, d) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
    return { x: f(p0.x, p1.x, p2.x, p3.x), y: f(p0.y, p1.y, p2.y, p3.y), z: f(p0.z, p1.z, p2.z, p3.z) };
  }
  function applyEasingJS(t, easing) {
    if (easing === "ease_in") return t * t;
    if (easing === "ease_out") return 1 - (1 - t) * (1 - t);
    if (easing === "ease_in_out") return 3 * t * t - 2 * t * t * t;
    return t;
  }
  function sampleKeyframePoseAtFrame(frame) {
    const kfs = pathKeyframes;
    if (kfs.length === 0) return null;
    if (kfs.length === 1) return { position: kfs[0].position, target: kfs[0].target };
    const positions = [kfs[0].position, ...kfs.map((k) => k.position), kfs[kfs.length - 1].position];
    const targets = [kfs[0].target, ...kfs.map((k) => k.target), kfs[kfs.length - 1].target];
    const frameIdx = kfs.map((k) => k.frame_index);
    const easings = kfs.map((k) => k.easing);
    let seg, localT;
    if (frame <= frameIdx[0]) { seg = 0; localT = 0; }
    else if (frame >= frameIdx[frameIdx.length - 1]) { seg = frameIdx.length - 2; localT = 1; }
    else {
      seg = 0;
      for (let i = 0; i < frameIdx.length - 1; i++) {
        if (frameIdx[i] <= frame && frame <= frameIdx[i + 1]) { seg = i; break; }
      }
      const span = frameIdx[seg + 1] - frameIdx[seg];
      localT = span ? (frame - frameIdx[seg]) / span : 0;
    }
    const easedT = applyEasingJS(localT, easings[seg]);
    return {
      position: catmullRom3JS(positions[seg], positions[seg + 1], positions[seg + 2], positions[seg + 3], easedT),
      target: catmullRom3JS(targets[seg], targets[seg + 1], targets[seg + 2], targets[seg + 3], easedT),
    };
  }
  // Exposed to the shared animate() loop above via the outer `applyPathPoseAtT` name.
  applyPathPoseAtT = function (t) {
    const frame = t * Math.max(0, pathFrameCount - 1);
    const pose = sampleKeyframePoseAtFrame(frame);
    if (!pose) return;
    camera.position.set(pose.position.x, pose.position.y, pose.position.z);
    camera.up.set(0, 1, 0);
    camera.lookAt(pose.target.x, pose.target.y, pose.target.z);
  };

  function rebuildPathVisualization() {
    pathGroup.children.forEach((c) => { c.geometry?.dispose?.(); c.material?.dispose?.(); });
    pathGroup.clear();
    if (pathKeyframes.length === 0) return;
    const markerGeo = new THREE.SphereGeometry(0.08, 12, 8);
    pathKeyframes.forEach((kf) => {
      const marker = new THREE.Mesh(markerGeo, new THREE.MeshBasicMaterial({ color: 0xffaa33 }));
      marker.position.set(kf.position.x, kf.position.y, kf.position.z);
      pathGroup.add(marker);
    });
    if (pathKeyframes.length >= 2) {
      // Built-in CatmullRomCurve3 for the visual line only — a close-enough
      // preview of the route; the eased/phantom-endpoint math above (which
      // mirrors camera_path.py exactly) is what actually drives Play/Bake.
      const curve = new THREE.CatmullRomCurve3(
        pathKeyframes.map((k) => new THREE.Vector3(k.position.x, k.position.y, k.position.z))
      );
      const pts = curve.getPoints(Math.max(2, pathKeyframes.length * 16));
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: 0xffaa33 })
      );
      pathGroup.add(line);
    }
  }

  // Vec3-object <-> array boundary conversion: pathKeyframes keeps {x,y,z}
  // objects internally (convenient for camera.position.set(...) etc.), but
  // schema.py's AtlasCameraKeyframe.from_dict iterates position/target/up as
  // plain [x,y,z] arrays (matching every other Point3D in this codebase) —
  // must convert both ways at the JSON boundary or Python's float(v) blows up
  // trying to convert the dict keys "x"/"y"/"z" themselves.
  function kfToJSON(kf) {
    const v3 = (v) => [v.x, v.y, v.z];
    return { frame_index: kf.frame_index, position: v3(kf.position), target: v3(kf.target), up: v3(kf.up), easing: kf.easing };
  }
  function kfFromJSON(data) {
    const obj = (a) => ({ x: a[0], y: a[1], z: a[2] });
    return { frame_index: data.frame_index, position: obj(data.position), target: obj(data.target), up: obj(data.up || [0, 1, 0]), easing: data.easing || "linear" };
  }

  function persistPathToClientData() {
    const widget = node.widgets?.find((w) => w.name === "client_data");
    if (!widget) return;
    let existing = {};
    try { existing = widget.value ? JSON.parse(widget.value) : {}; } catch (_) { existing = {}; }
    existing.camera_path = { keyframes: pathKeyframes.map(kfToJSON), fps: pathFps, frame_count: pathFrameCount };
    widget.value = JSON.stringify(existing);
    widget.callback?.(widget.value);
  }

  function restorePathFromClientData() {
    const widget = node.widgets?.find((w) => w.name === "client_data");
    if (!widget?.value) return;
    try {
      const existing = JSON.parse(widget.value);
      const cp = existing.camera_path;
      if (cp?.keyframes) {
        pathKeyframes = cp.keyframes.map(kfFromJSON);
        pathFps = cp.fps || 24;
        pathFrameCount = cp.frame_count || 48;
      }
    } catch (_) { /* no persisted path yet */ }
  }
  restorePathFromClientData();

  const pathBtn = document.createElement("button");
  pathBtn.textContent = "🎥 Camera Path";
  pathBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  pathBtn.onclick = () => {
    pathMode = !pathMode;
    pathBtn.style.background = pathMode ? "#3a2a1a" : "#2a2a2a";
    pathGroup.visible = pathMode;
    pathPanel.style.display = pathMode ? "flex" : "none";
    controls.setEnabled(!pathMode);
    fly.setEnabled(pathMode);
    if (!pathMode) pathPlayback = null;
  };
  toolbar.appendChild(pathBtn);

  // Camera Path panel — keyframe list + timeline controls. Its own row below
  // the toolbar (see the "Assemble" section), hidden until 🎥 Camera Path is on.
  const pathPanel = document.createElement("div");
  pathPanel.style.cssText = "display:none;flex-wrap:wrap;align-items:center;gap:6px;padding:4px 6px;background:#181818;border-top:1px solid #333;font-size:11px;color:#ccc";

  const kfListEl = document.createElement("div");
  kfListEl.style.cssText = "display:flex;flex-wrap:wrap;gap:4px;";

  function renderKeyframeList() {
    kfListEl.replaceChildren();
    pathKeyframes
      .slice()
      .sort((a, b) => a.frame_index - b.frame_index)
      .forEach((kf) => {
        const row = document.createElement("span");
        row.style.cssText = "display:inline-flex;align-items:center;gap:3px;background:#242424;border:1px solid #3a3a3a;border-radius:3px;padding:1px 4px;";
        const label = document.createElement("span");
        label.textContent = `#${kf.frame_index}`;
        const easingSel = document.createElement("select");
        easingSel.style.cssText = "font-size:10px;background:#1e1e1e;color:#ccc;border:1px solid #444;";
        ["linear", "ease_in", "ease_out", "ease_in_out"].forEach((opt) => {
          const o = document.createElement("option");
          o.value = opt; o.textContent = opt;
          if (opt === kf.easing) o.selected = true;
          easingSel.appendChild(o);
        });
        easingSel.onchange = () => { kf.easing = easingSel.value; persistPathToClientData(); };
        const delBtn = document.createElement("button");
        delBtn.textContent = "✕";
        delBtn.style.cssText = "font-size:10px;cursor:pointer;background:none;color:#f88;border:none;";
        delBtn.onclick = () => {
          pathKeyframes = pathKeyframes.filter((k) => k !== kf);
          renderKeyframeList();
          rebuildPathVisualization();
          persistPathToClientData();
        };
        row.append(label, easingSel, delBtn);
        kfListEl.appendChild(row);
      });
  }
  renderKeyframeList();
  rebuildPathVisualization();

  // Current camera pose as a {position, target} pair (target = a point straight
  // ahead at the solved scene depth, or a 10m fallback) — shared by "+ Keyframe"
  // and the presets below so both capture the camera identically.
  function captureCurrentPose() {
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    const captureDist = recoveredData?.camera_meta?.scene_depth_m || 10;
    const target = camera.position.clone().addScaledVector(forward, captureDist);
    return {
      position: { x: camera.position.x, y: camera.position.y, z: camera.position.z },
      target: { x: target.x, y: target.y, z: target.z },
    };
  }

  const addKfBtn = document.createElement("button");
  addKfBtn.textContent = "+ Keyframe";
  addKfBtn.style.cssText = "padding:2px 6px;font-size:11px;cursor:pointer;background:#2a3a2a;color:#cfc;border:1px solid #464;border-radius:3px";
  addKfBtn.onclick = () => {
    const pose = captureCurrentPose();
    const nextFrame = pathKeyframes.length
      ? Math.max(...pathKeyframes.map((k) => k.frame_index)) + Math.max(1, Math.round(pathFrameCount / 4))
      : 0;
    pathKeyframes.push({
      frame_index: Math.min(nextFrame, Math.max(0, pathFrameCount - 1)),
      position: pose.position,
      target: pose.target,
      up: { x: 0, y: 1, z: 0 },
      easing: "linear",
    });
    renderKeyframeList();
    rebuildPathVisualization();
    persistPathToClientData();
  };

  // ---------------------------------------------------------------------------
  // Presets — quick-start 2-keyframe moves (start = current pose, end = the
  // same pose transformed) instead of hand-placing keyframes with fly nav
  // every time. Pan rotates the TARGET around the (fixed) camera position —
  // the camera swivels in place, like a real pan. Orbit moves the POSITION
  // around the (fixed) target — the camera arcs around the subject. Dolly
  // moves the POSITION toward/away from the (fixed) target along the view
  // axis. All three are plain vector math (no Euler/yaw sign ambiguity):
  // "right"/"left" and "in"/"out" are derived directly from the camera's own
  // forward/right vectors at the moment the preset is applied, so they're
  // unambiguous regardless of world orientation.
  // ---------------------------------------------------------------------------
  function computePresetEndPose(basePose, presetKey, angleDeg, distanceFrac) {
    const E = basePose.position, T = basePose.target;
    const fwd = { x: T.x - E.x, y: T.y - E.y, z: T.z - E.z };
    const dist = Math.hypot(fwd.x, fwd.y, fwd.z) || 1;
    const fwdN = { x: fwd.x / dist, y: fwd.y / dist, z: fwd.z / dist };
    // right = normalize(cross(forward, world-up)) — matches THREE's camera-right
    // convention; cross(v, (0,1,0)) simplifies to (-v.z, 0, v.x).
    const right = { x: -fwdN.z, y: 0, z: fwdN.x };
    const rightLen = Math.hypot(right.x, right.y, right.z) || 1;
    const rightN = { x: right.x / rightLen, y: right.y / rightLen, z: right.z / rightLen };
    const a = THREE.MathUtils.degToRad(angleDeg) * (presetKey.endsWith("_left") ? -1 : 1);

    if (presetKey === "pan_left" || presetKey === "pan_right") {
      const newFwd = {
        x: fwdN.x * Math.cos(a) + rightN.x * Math.sin(a),
        y: fwdN.y * Math.cos(a) + rightN.y * Math.sin(a),
        z: fwdN.z * Math.cos(a) + rightN.z * Math.sin(a),
      };
      return { position: { ...E }, target: { x: E.x + newFwd.x * dist, y: E.y + newFwd.y * dist, z: E.z + newFwd.z * dist } };
    }
    if (presetKey === "orbit_left" || presetKey === "orbit_right") {
      const off = { x: E.x - T.x, y: E.y - T.y, z: E.z - T.z };
      const cos = Math.cos(a), sin = Math.sin(a);
      const rotated = { x: off.x * cos + off.z * sin, y: off.y, z: -off.x * sin + off.z * cos };
      return { position: { x: T.x + rotated.x, y: T.y + rotated.y, z: T.z + rotated.z }, target: { ...T } };
    }
    // dolly_in / dolly_out — move along the view axis toward/away from the fixed target.
    const scale = presetKey === "dolly_in" ? Math.max(0.05, 1 - distanceFrac) : 1 + distanceFrac;
    const newDist = dist * scale;
    return {
      position: { x: T.x - fwdN.x * newDist, y: T.y - fwdN.y * newDist, z: T.z - fwdN.z * newDist },
      target: { ...T },
    };
  }

  // Directional presets (Pan/Orbit) are framed as "arrive at my current
  // vantage from the other side" rather than "leave from here" — Pan Left
  // starts with the geo/subject swung to the right of frame and sweeps to
  // center (camera position never moves for a true pan, only the view
  // direction); Orbit Left starts with the camera itself physically
  // positioned to the right of the pivot and arcs to center (position DOES
  // move for orbit). Both are built the same way: compute the START pose by
  // running the OPPOSITE preset (e.g. pan_right for a pan_left request) on
  // the current pose, then set the END to the current pose unchanged — so
  // applying a preset always finishes on whatever you're currently looking
  // at, arriving from the named side. Dolly has no left/right, so it keeps
  // the original current-pose-is-the-start shape.
  const PRESET_OPPOSITE = {
    pan_left: "pan_right", pan_right: "pan_left",
    orbit_left: "orbit_right", orbit_right: "orbit_left",
  };

  const PRESET_OPTIONS = [
    { key: "", label: "— Preset —" },
    { key: "pan_left", label: "Pan Left" },
    { key: "pan_right", label: "Pan Right" },
    { key: "orbit_left", label: "Orbit Left" },
    { key: "orbit_right", label: "Orbit Right" },
    { key: "dolly_in", label: "Dolly In" },
    { key: "dolly_out", label: "Dolly Out" },
  ];
  const presetSelect = document.createElement("select");
  presetSelect.style.cssText = "font-size:11px;background:#1e1e1e;color:#ccc;border:1px solid #444;";
  PRESET_OPTIONS.forEach(({ key, label }) => {
    const o = document.createElement("option");
    o.value = key; o.textContent = label;
    presetSelect.appendChild(o);
  });
  const presetAngleInput = document.createElement("input");
  presetAngleInput.type = "number"; presetAngleInput.min = "1"; presetAngleInput.max = "180"; presetAngleInput.value = "30";
  presetAngleInput.title = "angle (degrees) — used by Pan/Orbit presets";
  presetAngleInput.style.cssText = "width:38px;font-size:11px;background:#1e1e1e;color:#ccc;border:1px solid #444;";
  const presetAmountInput = document.createElement("input");
  presetAmountInput.type = "number"; presetAmountInput.min = "0.05"; presetAmountInput.max = "0.9"; presetAmountInput.step = "0.05"; presetAmountInput.value = "0.4";
  presetAmountInput.title = "distance fraction — used by Dolly In/Out presets";
  presetAmountInput.style.cssText = "width:42px;font-size:11px;background:#1e1e1e;color:#ccc;border:1px solid #444;";
  const applyPresetBtn = document.createElement("button");
  applyPresetBtn.textContent = "Apply Preset";
  applyPresetBtn.style.cssText = "padding:2px 6px;font-size:11px;cursor:pointer;background:#2a2a3a;color:#dcf;border:1px solid #546;border-radius:3px";
  applyPresetBtn.onclick = () => {
    const presetKey = presetSelect.value;
    if (!presetKey) return;
    const basePose = captureCurrentPose();
    const angleDeg = parseFloat(presetAngleInput.value) || 30;
    const amountFrac = parseFloat(presetAmountInput.value) || 0.4;

    let startPose, endPose;
    const opposite = PRESET_OPPOSITE[presetKey];
    if (opposite) {
      startPose = computePresetEndPose(basePose, opposite, angleDeg, amountFrac);
      endPose = basePose;
    } else {
      startPose = basePose;
      endPose = computePresetEndPose(basePose, presetKey, angleDeg, amountFrac);
    }

    const endFrame = Math.max(1, pathFrameCount - 1);
    // Replaces any existing keyframes — presets are a fresh quick-start, not
    // an addition to hand-placed ones (which would rarely combine sensibly).
    pathKeyframes = [
      { frame_index: 0, position: startPose.position, target: startPose.target, up: { x: 0, y: 1, z: 0 }, easing: "ease_in_out" },
      { frame_index: endFrame, position: endPose.position, target: endPose.target, up: { x: 0, y: 1, z: 0 }, easing: "linear" },
    ];
    renderKeyframeList();
    rebuildPathVisualization();
    persistPathToClientData();
  };

  const frameCountInput = document.createElement("input");
  frameCountInput.type = "number"; frameCountInput.min = "1"; frameCountInput.max = "2000";
  frameCountInput.value = String(pathFrameCount);
  frameCountInput.style.cssText = "width:48px;font-size:11px;background:#1e1e1e;color:#ccc;border:1px solid #444;";
  frameCountInput.title = "frame_count";
  frameCountInput.onchange = () => { pathFrameCount = Math.max(1, parseInt(frameCountInput.value, 10) || 48); persistPathToClientData(); };

  const fpsInput = document.createElement("input");
  fpsInput.type = "number"; fpsInput.min = "1"; fpsInput.max = "240";
  fpsInput.value = String(pathFps);
  fpsInput.style.cssText = "width:40px;font-size:11px;background:#1e1e1e;color:#ccc;border:1px solid #444;";
  fpsInput.title = "fps";
  fpsInput.onchange = () => { pathFps = Math.max(1, parseFloat(fpsInput.value) || 24); persistPathToClientData(); };

  const playBtn = document.createElement("button");
  playBtn.textContent = "▶ Play";
  playBtn.style.cssText = "padding:2px 6px;font-size:11px;cursor:pointer;background:#2a2a3a;color:#dcf;border:1px solid #546;border-radius:3px";
  playBtn.onclick = () => {
    if (pathKeyframes.length === 0) return;
    pathPlayback = {
      startTime: performance.now(),
      durationSec: Math.max(0.2, pathFrameCount / pathFps),
      onDone: () => { if (recoveredData) applyRecoveredView(recoveredData, { force: true }); },
    };
  };

  const bakeBtn = document.createElement("button");
  bakeBtn.textContent = "⏺ Bake Proxy Path";
  bakeBtn.style.cssText = "padding:2px 8px;font-size:11px;cursor:pointer;background:#3a1a2a;color:#fac;border:1px solid #645;border-radius:3px";
  bakeBtn.onclick = async () => {
    if (pathKeyframes.length === 0) return;
    bakeBtn.disabled = true;
    bakeBtn.textContent = "Baking Proxy...";
    const savedPos = camera.position.clone();
    const savedQuat = camera.quaternion.clone();
    const savedAspect = camera.aspect;
    const wasPlaying = !!pathPlayback;
    let outputRt = null;
    pathPlayback = null;
    pathGroup.visible = false; // exclude keyframe markers/line from baked frames
    try {
      const frames = [];
      outputRt = new THREE.WebGLRenderTarget(W, H);
      camera.aspect = W / H;
      camera.updateProjectionMatrix();
      for (let frame = 0; frame < pathFrameCount; frame++) {
        const pose = sampleKeyframePoseAtFrame(frame);
        if (!pose) break;
        camera.position.set(pose.position.x, pose.position.y, pose.position.z);
        camera.up.set(0, 1, 0);
        camera.lookAt(pose.target.x, pose.target.y, pose.target.z);
        // JPEG, not PNG: baked frames feed a video encoder (h264, lossy), so
        // lossless PNG is pure waste — JPEG is ~5–10× smaller and stops the
        // whole clip's base64 from OOM-ing the JS heap when it's all stringified
        // into one client_data blob (the reported bake OOM at 1280×100 frames).
        frames.push(atlasRenderSceneToBase64(renderer, scene, camera, W, H,
          { renderTarget: outputRt, mime: "image/jpeg", quality: 0.9 }));
      }
      const widget = node.widgets?.find((w) => w.name === "client_data");
      let existing = {};
      try { existing = widget?.value ? JSON.parse(widget.value) : {}; } catch (_) { existing = {}; }
      existing.path_frames = frames;
      existing.camera_path = { keyframes: pathKeyframes.map(kfToJSON), fps: pathFps, frame_count: pathFrameCount };
      existing.atlas_proxy_path = {
        transport: "jpeg_base64_proxy_ldr",
        width: W,
        height: H,
        fps: pathFps,
        frame_count: pathFrameCount,
      };
      if (widget) {
        widget.value = JSON.stringify(existing);
        widget.callback?.(widget.value);
      }
      app.queuePrompt(0, 1);
    } finally {
      outputRt?.dispose();
      camera.position.copy(savedPos);
      camera.quaternion.copy(savedQuat);
      camera.aspect = savedAspect;
      camera.updateProjectionMatrix();
      pathGroup.visible = pathMode;
      bakeBtn.disabled = false;
      bakeBtn.textContent = "⏺ Bake Proxy Path";
      if (wasPlaying) playBtn.onclick();
    }
  };

  const fcWrap = document.createElement("span");
  fcWrap.style.cssText = "display:inline-flex;align-items:center;gap:2px;";
  fcWrap.append(document.createTextNode("frames"), frameCountInput);
  const fpsWrap = document.createElement("span");
  fpsWrap.style.cssText = "display:inline-flex;align-items:center;gap:2px;";
  fpsWrap.append(document.createTextNode("fps"), fpsInput);

  const presetWrap = document.createElement("span");
  presetWrap.style.cssText = "display:inline-flex;align-items:center;gap:2px;padding-left:6px;border-left:1px solid #333;";
  presetWrap.append(presetSelect, presetAngleInput, presetAmountInput, applyPresetBtn);

  // ---------------------------------------------------------------------------
  // Import Camera FBX (Phase B) — a DCC-authored camera move (Blender/Maya
  // export) sampled client-side via FBXLoader + AnimationMixer, no Python FBX
  // parsing (same "Three.js is frontend-only" rule OBJLoader already follows).
  //
  // An FBX export has no ground-truth relationship to Atlas's world frame —
  // same problem AtlasAddPatchView solves for a single static offset via a
  // constructed (not solved) patch camera. This applies the same principle to
  // a full animation curve: treat it as a RELATIVE move from wherever the
  // viewport camera currently is, aligning the FBX camera's own frame-0
  // forward vector to the current base forward (captureCurrentPose()) so
  // "dolly 2m and pan 15°" transfers even though the FBX's absolute axes
  // don't correspond to Atlas's scene at all. Verified numerically for pure
  // translation and pure level-pan (both reproduce exactly — see git history).
  // Known limitation: alignQuat is a single minimal rotation (its axis is
  // whatever cross(fbxForward0, baseForward) happens to be), which only
  // commutes with the FBX clip's OWN rotation when both cameras are
  // reasonably level (near-zero pitch/roll, the same assumption
  // horizon_row_from_extrinsics already makes elsewhere) — a level FBX pan
  // transfers its exact angle regardless of the two cameras' absolute yaw
  // offset, but a steeply pitched FBX camera aligned to a very differently-
  // pitched base view can pick up some rotational "swim" beyond a pure
  // yaw/pitch transfer. Acceptable for a first pass; recalibrate by eye
  // (same principle as AtlasAddPatchView's flip_azimuth) if it looks off.
  // ---------------------------------------------------------------------------
  const importFbxInput = document.createElement("input");
  importFbxInput.type = "file";
  importFbxInput.accept = ".fbx";
  importFbxInput.style.display = "none";

  const importStatusEl = document.createElement("span");
  importStatusEl.style.cssText = "font-size:10px;color:#9ab;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";

  const importSamplesInput = document.createElement("input");
  importSamplesInput.type = "number"; importSamplesInput.min = "2"; importSamplesInput.max = "300"; importSamplesInput.value = "30";
  importSamplesInput.title = "samples to take across the FBX clip's duration";
  importSamplesInput.style.cssText = "width:42px;font-size:11px;background:#1e1e1e;color:#ccc;border:1px solid #444;";

  const importScaleInput = document.createElement("input");
  importScaleInput.type = "number"; importScaleInput.min = "0.001"; importScaleInput.step = "0.01"; importScaleInput.value = "1.0";
  importScaleInput.title = "position scale — FBX units (often cm) vs. the solved metric scene rarely match; adjust by eye if the imported move looks too big/small";
  importScaleInput.style.cssText = "width:46px;font-size:11px;background:#1e1e1e;color:#ccc;border:1px solid #444;";

  async function importCameraFBX(file) {
    if (!FBXLoader) { importStatusEl.textContent = "FBXLoader unavailable"; return; }
    importStatusEl.textContent = "Parsing...";
    try {
      const buffer = await file.arrayBuffer();
      const group = new FBXLoader().parse(buffer, "");
      let camObj = null;
      group.traverse((o) => { if (o.isCamera && !camObj) camObj = o; });
      if (!camObj) { importStatusEl.textContent = "No camera found in FBX"; return; }
      const clip = group.animations?.[0];
      if (!clip) { importStatusEl.textContent = "No animation clip on the FBX camera"; return; }

      const sampleCount = Math.max(2, parseInt(importSamplesInput.value, 10) || 30);
      const scale = parseFloat(importScaleInput.value) || 1.0;
      const mixer = new THREE.AnimationMixer(group);
      mixer.clipAction(clip, camObj).play();

      const basePose = captureCurrentPose();
      const baseForward = new THREE.Vector3(
        basePose.target.x - basePose.position.x,
        basePose.target.y - basePose.position.y,
        basePose.target.z - basePose.position.z
      );
      const baseCaptureDist = baseForward.length() || 10;
      baseForward.normalize();

      const samples = [];
      const pos = new THREE.Vector3(), quat = new THREE.Quaternion(), scl = new THREE.Vector3();
      for (let s = 0; s < sampleCount; s++) {
        mixer.setTime((clip.duration * s) / (sampleCount - 1));
        group.updateMatrixWorld(true);
        camObj.matrixWorld.decompose(pos, quat, scl);
        samples.push({
          position: pos.clone(),
          forward: new THREE.Vector3(0, 0, -1).applyQuaternion(quat),
        });
      }

      const alignQuat = new THREE.Quaternion().setFromUnitVectors(samples[0].forward, baseForward);
      const pos0 = samples[0].position;
      const basePos = new THREE.Vector3(basePose.position.x, basePose.position.y, basePose.position.z);

      pathKeyframes = samples.map((sample, i) => {
        const alignedForward = sample.forward.clone().applyQuaternion(alignQuat);
        const posDelta = sample.position.clone().sub(pos0).applyQuaternion(alignQuat).multiplyScalar(scale);
        const newPos = basePos.clone().add(posDelta);
        const newTarget = newPos.clone().addScaledVector(alignedForward, baseCaptureDist);
        return {
          frame_index: i,
          position: { x: newPos.x, y: newPos.y, z: newPos.z },
          target: { x: newTarget.x, y: newTarget.y, z: newTarget.z },
          up: { x: 0, y: 1, z: 0 },
          easing: "linear",
        };
      });

      pathFrameCount = sampleCount;
      frameCountInput.value = String(pathFrameCount);
      if (clip.duration > 0) {
        pathFps = Math.max(1, Math.round(sampleCount / clip.duration));
        fpsInput.value = String(pathFps);
      }
      renderKeyframeList();
      rebuildPathVisualization();
      persistPathToClientData();
      importStatusEl.textContent = `Imported ${sampleCount} kf from "${clip.name || "FBX clip"}"`;
    } catch (e) {
      console.error("[AtlasBlockout] FBX camera import failed:", e);
      importStatusEl.textContent = "Import failed — see console";
    }
  }
  importFbxInput.onchange = () => {
    const file = importFbxInput.files?.[0];
    if (file) importCameraFBX(file);
    importFbxInput.value = "";
  };

  const importBtn = document.createElement("button");
  importBtn.textContent = "📥 Import Camera FBX";
  importBtn.disabled = !FBXLoader;
  importBtn.title = FBXLoader ? "Import a camera animation from an FBX file (Blender/Maya export)" : "FBXLoader failed to load in this browser";
  importBtn.style.cssText = "padding:2px 6px;font-size:11px;cursor:pointer;background:#2a2a3a;color:#dcf;border:1px solid #546;border-radius:3px" + (importBtn.disabled ? ";opacity:0.5;cursor:not-allowed" : "");
  importBtn.onclick = () => importFbxInput.click();

  const importWrap = document.createElement("span");
  importWrap.style.cssText = "display:inline-flex;align-items:center;gap:2px;padding-left:6px;border-left:1px solid #333;";
  importWrap.append(importBtn, importFbxInput, importSamplesInput, importScaleInput, importStatusEl);

  pathPanel.append(addKfBtn, presetWrap, importWrap, kfListEl, fcWrap, fpsWrap, playBtn, bakeBtn);

  // 📊 Diagram toggle — layered VP / horizon / ground SVG overlay, each layer
  // independently dimmable. Vanishing points are populated only by the
  // classical (non-learned) solve path — the VP layer is simply empty when
  // using AtlasLearnedSolveFromImage, which predicts focal+gravity directly
  // rather than via vanishing points; horizon/ground still work either way.
  let diagramOn = false;
  const diagBtn = document.createElement("button");
  diagBtn.textContent = "📊 Diagram";
  diagBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  const layerSliders = [
    { g: gVpLines, label: "VP", init: 0.7 },
    { g: gHorizon, label: "Hz", init: 0.85 },
    { g: gGround, label: "Gnd", init: 0.35 },
  ].map(({ g, label, init }) => {
    const wrap = document.createElement("span");
    wrap.style.cssText = "display:inline-flex;align-items:center;gap:2px;font-size:10px;color:#9ab;margin-left:4px;";
    const lab = document.createElement("span"); lab.textContent = label;
    const slider = document.createElement("input");
    slider.type = "range"; slider.min = "0"; slider.max = "1"; slider.step = "0.05"; slider.value = String(init);
    slider.style.cssText = "width:44px;vertical-align:middle;";
    slider.disabled = true;
    slider.oninput = () => { g.style.opacity = slider.value; };
    wrap.append(lab, slider);
    return wrap;
  });
  diagBtn.onclick = () => {
    diagramOn = !diagramOn;
    diagramSvg.style.display = diagramOn ? "block" : "none";
    diagBtn.style.background = diagramOn ? "#2a3a3a" : "#2a2a2a";
    layerSliders.forEach((w) => { w.querySelector("input").disabled = !diagramOn; });
  };
  toolbar.append(diagBtn, ...layerSliders);

  // ℹ Info toggle — solved latent-camera metadata (lens, distance, confidence).
  let infoOn = false;
  const infoBtn = document.createElement("button");
  infoBtn.textContent = "ℹ Info";
  infoBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  infoBtn.onclick = () => {
    infoOn = !infoOn;
    metaHud.style.display = infoOn ? "block" : "none";
    infoBtn.style.background = infoOn ? "#2a3a3a" : "#2a2a2a";
  };
  toolbar.appendChild(infoBtn);

  // ☀ Exposure — tone-mapped brightness preview of the LIT (grey/shaded)
  // geometry. Never affects the projected photo texture (the projection
  // shader writes gl_FragColor directly with no tone-mapping chunk) or the
  // depth/normal/mask render passes (explicitly toneMapped:false above).
  const expWrap = document.createElement("span");
  expWrap.style.cssText = "display:inline-flex;align-items:center;gap:3px;font-size:11px;color:#ddd;margin-left:4px;";
  const expLabel = document.createElement("span"); expLabel.textContent = "☀";
  const expSlider = document.createElement("input");
  expSlider.type = "range"; expSlider.min = "0.1"; expSlider.max = "3"; expSlider.step = "0.05"; expSlider.value = "1";
  expSlider.style.cssText = "width:70px;vertical-align:middle;";
  expSlider.oninput = () => { renderer.toneMappingExposure = parseFloat(expSlider.value); };
  expWrap.append(expLabel, expSlider);
  toolbar.appendChild(expWrap);

  // 💡 Lights — up to 2 movable THREE.PointLights. Unlike ☀ Exposure (which is
  // genuinely immune to the projection shader by construction), a light's
  // intensity IS wired into the shader's relight term above — but only once an
  // artist raises it off its default-0, so today's Project output is unaffected
  // until this panel is actually used.
  let lightsOn = false;
  const lightBtn = document.createElement("button");
  lightBtn.textContent = "💡 Lights";
  lightBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  lightBtn.onclick = () => {
    lightsOn = !lightsOn;
    lightBtn.style.background = lightsOn ? "#3a2a1a" : "#2a2a2a";
    lightPanel.style.display = lightsOn ? "flex" : "none";
  };
  toolbar.appendChild(lightBtn);

  // 🎯 Orbit pivot offset — nudge the point the orbit swings around (world
  // metres, ADDED on top of the auto pivot). Session-only; default (0,0,0) = the
  // auto pivot exactly, so nothing changes until dialled. Useful once
  // AtlasScaleOverride pushes geometry to 100m+ and the auto centroid isn't
  // where you want to look. The step auto-scales with the scene (placeDefaultLights).
  let pivotOn = false;
  const pivotBtn = document.createElement("button");
  pivotBtn.textContent = "🎯 Pivot";
  pivotBtn.title = "Manually offset the orbit pivot (world metres)";
  pivotBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  pivotBtn.onclick = () => {
    pivotOn = !pivotOn;
    pivotBtn.style.background = pivotOn ? "#1a2a3a" : "#2a2a2a";
    pivotPanel.style.display = pivotOn ? "flex" : "none";
  };
  toolbar.appendChild(pivotBtn);

  const pivotPanel = document.createElement("div");
  pivotPanel.style.cssText = "display:none;flex-wrap:wrap;align-items:center;gap:8px;padding:4px 6px;background:#181818;border-top:1px solid #333;font-size:11px;color:#ccc";
  const pivotLabel = document.createElement("span");
  pivotLabel.textContent = "Orbit pivot offset (m):";
  pivotPanel.appendChild(pivotLabel);
  pivotInputs = ["x", "y", "z"].map((axis) => {
    const wrap = document.createElement("span");
    wrap.style.cssText = "display:inline-flex;align-items:center;gap:3px;";
    const lab = document.createElement("span");
    lab.textContent = axis.toUpperCase();
    const inp = document.createElement("input");
    inp.type = "number";
    inp.value = "0";
    inp.step = "0.25";
    inp.style.cssText = "width:60px;background:#111;color:#ddd;border:1px solid #444;border-radius:3px;padding:2px 4px;font-size:11px";
    inp.onchange = () => {
      const v = parseFloat(inp.value);
      pivotOffset[axis] = Number.isFinite(v) ? v : 0;
      applyPivotOffset();
    };
    wrap.appendChild(lab);
    wrap.appendChild(inp);
    pivotPanel.appendChild(wrap);
    return inp;
  });
  const pivotReset = document.createElement("button");
  pivotReset.textContent = "Reset";
  pivotReset.title = "Recentre the orbit pivot on the auto (geometry) point";
  pivotReset.style.cssText = "padding:2px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  pivotReset.onclick = () => {
    pivotOffset.set(0, 0, 0);
    pivotInputs.forEach((inp) => { inp.value = "0"; });
    applyPivotOffset();
  };
  pivotPanel.appendChild(pivotReset);

  // ⛶ Fullscreen — the browser Fullscreen API on canvasWrap (canvas + all
  // HUD/diagram/legend overlays; NOT the container, whose toolbar may live in
  // a detached Output Desk — canvasWrap behaves identically in both modes).
  // Pure display change: no node sizing, no widget layout, no canvas
  // attribute writes — the render RESOLUTION stays governed by the
  // `resolution` widget (CSS object-fit:contain letterboxes, exactly like
  // dragging the node large). Esc exits natively; entering focuses the
  // canvas so the tracking keys (↑↓ in/out · ←→ left/right · A/D up/down)
  // work immediately.
  const fsBtn = document.createElement("button");
  fsBtn.textContent = "⛶ Fullscreen";
  fsBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  fsBtn.onclick = () => {
    if (document.fullscreenElement === canvasWrap) {
      document.exitFullscreen?.();
    } else {
      canvasWrap.requestFullscreen?.().catch(() => {});
    }
  };
  toolbar.appendChild(fsBtn);
  const onFsChange = () => {
    const active = document.fullscreenElement === canvasWrap;
    fsBtn.textContent = active ? "⛶ Exit" : "⛶ Fullscreen";
    fsBtn.style.background = active ? "#2a3a3a" : "#2a2a2a";
    if (active) canvas.focus({ preventScroll: true });
  };
  document.addEventListener("fullscreenchange", onFsChange);
  // Removed via the CHAINED onRemoved cleanup (never assign onRemoved —
  // see the orphaned-DOM lineage entry).
  node._atlasFsCleanup = () => document.removeEventListener("fullscreenchange", onFsChange);

  const lightPanel = document.createElement("div");
  lightPanel.style.cssText = "display:none;flex-wrap:wrap;align-items:center;gap:10px;padding:4px 6px;background:#181818;border-top:1px solid #333;font-size:11px;color:#ccc";
  movableLights.forEach((light, idx) => {
    const group = document.createElement("span");
    group.style.cssText = "display:inline-flex;align-items:center;gap:4px;";
    const label = document.createElement("span");
    label.textContent = `Light ${idx + 1}`;
    label.style.cssText = "color:#ddd;font-weight:600;";
    group.appendChild(label);
    light._atlasInputs = [];
    ["x", "y", "z"].forEach((axis) => {
      const axisLabel = document.createElement("span");
      axisLabel.textContent = axis.toUpperCase();
      axisLabel.style.cssText = "color:#888;";
      const input = document.createElement("input");
      input.type = "number";
      input.step = "0.1";
      input.value = light.position[axis].toFixed(1);
      input.style.cssText = "width:52px;background:#1e1e1e;color:#ddd;border:1px solid #444;border-radius:3px;padding:1px 3px;";
      // Editing a position pins the light — placeDefaultLights won't move it again.
      input.oninput = () => { light.position[axis] = parseFloat(input.value) || 0; light.userData.atlasMoved = true; };
      group.append(axisLabel, input);
      light._atlasInputs.push(input);
    });
    const intLabel = document.createElement("span");
    intLabel.textContent = "Intensity";
    intLabel.style.cssText = "color:#888;margin-left:4px;";
    const intSlider = document.createElement("input");
    intSlider.type = "range"; intSlider.min = "0"; intSlider.max = "10"; intSlider.step = "0.05"; intSlider.value = "0";
    intSlider.style.cssText = "width:70px;vertical-align:middle;";
    intSlider.oninput = () => { light.intensity = parseFloat(intSlider.value) || 0; };
    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.value = `#${light.color.getHexString()}`;
    colorInput.style.cssText = "width:22px;height:18px;padding:0;border:1px solid #444;background:none;cursor:pointer;";
    colorInput.oninput = () => { light.color.set(colorInput.value); };
    group.append(intLabel, intSlider, colorInput);
    lightPanel.appendChild(group);
  });

  // Detail relight — photo-luminance bump strength. Perturbs the normal the
  // lights read (uBumpStrength), so they sculpt fine surface detail the coarse
  // geometry lacks. 0 = off (geometry normal only). Needs a light raised above 0.
  {
    const group = document.createElement("span");
    group.style.cssText = "display:inline-flex;align-items:center;gap:4px;";
    const label = document.createElement("span");
    label.textContent = "Detail";
    label.style.cssText = "color:#ddd;font-weight:600;";
    label.title = "Photo-luminance surface detail for the lights (raise a light too).";
    const slider = document.createElement("input");
    slider.type = "range"; slider.min = "0"; slider.max = "6"; slider.step = "0.05"; slider.value = "0";
    slider.style.cssText = "width:90px;vertical-align:middle;";
    const val = document.createElement("span");
    val.textContent = "0.00"; val.style.cssText = "color:#888;width:28px;";
    slider.oninput = () => { bumpStrength = parseFloat(slider.value) || 0; val.textContent = bumpStrength.toFixed(2); };
    group.append(label, slider, val);
    // Scale = luminance-gradient sampling offset in texels (detail coarseness).
    const sLabel = document.createElement("span");
    sLabel.textContent = "Scale"; sLabel.style.cssText = "color:#888;margin-left:4px;";
    const sSlider = document.createElement("input");
    sSlider.type = "range"; sSlider.min = "1"; sSlider.max = "32"; sSlider.step = "1"; sSlider.value = String(bumpScale);
    sSlider.style.cssText = "width:70px;vertical-align:middle;";
    const sVal = document.createElement("span");
    sVal.textContent = String(bumpScale); sVal.style.cssText = "color:#888;width:20px;";
    sSlider.oninput = () => { bumpScale = parseFloat(sSlider.value) || 1; sVal.textContent = String(bumpScale); };
    group.append(sLabel, sSlider, sVal);
    lightPanel.appendChild(group);
  }

  // (Clear button removed 2026-07-09 along with the primitive/proxy buttons —
  // its only job was removing the user meshes those buttons created.)

  // Render Proxy Passes button
  const renderBtn = document.createElement("button");
  renderBtn.textContent = "⬛ Render Proxy Passes";
  renderBtn.style.cssText = "padding:3px 10px;font-size:11px;cursor:pointer;background:#1a3a1a;color:#afa;border:1px solid #464;border-radius:3px;margin-left:auto";
  renderBtn.onclick = async () => {
    renderBtn.disabled = true;
    renderBtn.textContent = "Rendering Proxy...";
    const savedAspect = camera.aspect;
    try {
      camera.aspect = W / H;
      camera.updateProjectionMatrix();
      const passes = await renderAllPasses(renderer, scene, camera, W, H, [bgMesh]);
      if (!passes) return;
      // Merge into client_data rather than overwrite — preserves a previously
      // baked camera_path/path_frames (same widget, see ⏺ Bake Proxy Path) instead
      // of wiping it out.
      const widget = node.widgets?.find((w) => w.name === "client_data");
      if (widget) {
        let existing = {};
        try { existing = widget.value ? JSON.parse(widget.value) : {}; } catch (_) { existing = {}; }
        widget.value = JSON.stringify({
          ...existing,
          ...passes,
          atlas_proxy_passes: {
            transport: "png_base64_proxy_ldr",
            width: W,
            height: H,
            passes: ["shaded", "depth", "normal", "mask"],
          },
        });
        widget.callback?.(widget.value);
      }
      // Re-queue the prompt so Python receives the frames
      app.queuePrompt(0, 1);
    } finally {
      camera.aspect = savedAspect;
      camera.updateProjectionMatrix();
      renderBtn.disabled = false;
      renderBtn.textContent = "⬛ Render Proxy Passes";
    }
  };
  toolbar.appendChild(renderBtn);

  // Assemble. The DOM widget's normal flow must remain canvas-only: ComfyUI's
  // DOMWidget layout currently reports minWidth:0, so putting toolbar/pathPanel
  // beside the canvas in flex flow can collapse the widget width on relayout.
  // With no AtlasViewportControls node connected, controls live in an absolute
  // overlay inside canvasWrap. With a controls node connected, the same DOM
  // elements are reparented there and the local overlay is hidden/empty.
  containerEl.appendChild(canvasWrap);
  let _atlasToolbarMountTarget = null;
  let _atlasPathMountTarget = null;
  let _atlasLightMountTarget = null;
  function mountControls() {
    const controlsNode = getLinkedControlsNode(node);
    const externalTarget = controlsNode?._atlasControlsContainer || null;
    const toolbarTarget = controlsNode?._atlasToolbarContainer || externalTarget || localControlsLayer;
    const pathTarget = controlsNode?._atlasPathContainer || toolbarTarget;
    const lightTarget = controlsNode?._atlasLightContainer || toolbarTarget;
    localControlsLayer.style.display = externalTarget ? "none" : "flex";
    if (toolbarTarget !== _atlasToolbarMountTarget) {
      _atlasToolbarMountTarget = toolbarTarget;
      toolbarTarget.appendChild(toolbar);
    }
    if (pathTarget !== _atlasPathMountTarget) {
      _atlasPathMountTarget = pathTarget;
      pathTarget.appendChild(pathPanel);
    }
    if (lightTarget !== _atlasLightMountTarget) {
      _atlasLightMountTarget = lightTarget;
      lightTarget.appendChild(lightPanel);
      lightTarget.appendChild(pivotPanel);
    }
    if (recoveredData) updateLinkedOutputDesk(recoveredData);
  }
  mountControls();

  // Store refs for cleanup and camera application
  node._atlasRenderer = renderer;
  node._atlasScene = scene;
  node._atlasCamera = camera;
  node._atlasControls = controls;
  node._atlasFly = fly;
  node._atlasBgMesh = null;
  node._atlasW = W;
  node._atlasH = H;
  node._atlasApplyOutputProfilePreview = applyOutputProfilePreview;

  // Resize the render target + canvas so the viewport matches the source image
  // aspect (target_width/target_height come from the Python node, derived from the
  // incoming image). Keeps the camera aspect and canvas aspect in sync.
  function resizeViewport(w, h) {
    w = Math.max(16, Math.round(w || W));
    h = Math.max(16, Math.round(h || H));
    W = w; H = h;
    node._atlasW = w; node._atlasH = h;
    previewSize = atlasViewportPreviewSize(w, h);
    previewW = previewSize.width; previewH = previewSize.height;
    node._atlasPreviewW = previewW; node._atlasPreviewH = previewH;
    canvas.width = previewW; canvas.height = previewH;
    renderer.setSize(previewW, previewH, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    snapNodeHeightToRenderAspect(w / h);
  }

  // Snap the NODE height so the canvas box's shape matches the render aspect —
  // then object-fit:contain has nothing to letterbox and the preview fills the
  // full node width edge-to-edge (artist request 2026-07-07: previously a node
  // dragged wide showed the render pillarboxed in #111 dead space, which reads
  // as a "small preview" no matter how big the node is). Runs only from
  // resizeViewport (i.e. on execution, when the authoritative render dims
  // arrive) — deliberately NOT from a node.onResize hook, preserving the
  // "no JS resize hooks" rule this node earned the hard way (see the resize
  // history in CLAUDE.md); between executions a hand-dragged shape may
  // letterbox, and the next Queue snaps it back. Chrome height (title bar +
  // widget rows + any locally-mounted toolbar) is measured from the live
  // layout rather than hardcoded, so the detached-Output-Desk and local-
  // toolbar cases both come out exact.
  let pendingSnapAspect = null;
  function snapNodeHeightToRenderAspect(renderAspect) {
    if (!renderAspect || !isFinite(renderAspect)) return;
    if (document.fullscreenElement === canvasWrap) {
      // ⛶ fullscreen: rects are SCREEN-sized — snapping now would persist a
      // garbage node height behind the fullscreen view. Defer; the animate()
      // retry re-enters this guard each frame (one check) and the snap
      // applies correctly the moment fullscreen exits.
      pendingSnapAspect = renderAspect;
      return;
    }
    const scale = app.canvas?.ds?.scale || 1;
    const wrapRect = canvasWrap.getBoundingClientRect();
    if (!(wrapRect.width > 0) || !(wrapRect.height > 0)) {
      // Node is off-screen/hidden — rects are unmeasurable. Defer to the
      // animate() loop, which retries until the widget is laid out again.
      pendingSnapAspect = renderAspect;
      return;
    }
    pendingSnapAspect = null;
    const wrapW = wrapRect.width / scale;   // node units (host style px track node size 1:1)
    const wrapH = wrapRect.height / scale;
    const chrome = node.size[1] - wrapH;    // everything above/around the canvas box
    const desiredH = Math.min(4096, Math.max(120, chrome + wrapW / renderAspect));
    if (Math.abs(desiredH - node.size[1]) > 4) {
      node.setSize([node.size[0], desiredH]);
      node.graph?.setDirtyCanvas(true, true);
    }
  }

  function updateLinkedOutputDesk(data = {}) {
    const controlsNode = getLinkedControlsNode(node);
    controlsNode?._atlasOutputDeskUpdate?.({
      ...data,
      target_width: W,
      target_height: H,
      preview_width: previewW,
      preview_height: previewH,
    });
  }

  // Apply the recovered camera and initialise the orbit controller *from* it, so
  // the default view is the camera's own perspective (matching the source photo).
  function applyRecoveredView(data, opts = {}) {
    if (data.target_width && data.target_height) {
      resizeViewport(data.target_width, data.target_height);
    }
    // Only RESET the viewing camera when the solve/image actually changed.
    // Every execution used to snap the camera back to the recovered pose,
    // which became load-bearing-bad once 📐 Extract Angle re-queues the
    // graph: the artist's orbited view was wiped mid-flow, and a second 📐
    // click from the reset pose silently overwrote their real extraction
    // with a zero-orbit "front view" (found live). Same-solve re-executions
    // now preserve navigation; 📷 Camera View remains the explicit reset.
    // EXPLICIT resets (📷 button, ▶ Play's end-of-playback snap-back) pass
    // { force: true } — without it this guard silently swallowed the click,
    // because on an unchanged solve sameSolve is always true (reported live:
    // "Camera View doesn't work anymore").
    const sameSolve = !!(node._atlasLastSolveFp && data.solve_fingerprint
      && node._atlasLastSolveFp === data.solve_fingerprint);
    node._atlasLastSolveFp = data.solve_fingerprint || null;
    if (opts.force || !sameSolve) {
      applyRecoveredCamera(camera, data);
      if (!sameSolve) lastGeometryPivot = null; // new scene — stale pivot invalid
      // Reuse the geometry pivot when we have one (median-depth pivot from
      // setProxies) — 📷 Camera View used to fall back to the ground-point
      // heuristic here, which stomped the good pivot with one capped at
      // 1.5× scene depth, so the FIRST re-orbit after a reset swung around a
      // point way behind the subject (artist-reported 2026-07-09). The
      // heuristic remains only the no-geometry-yet fallback.
      if (lastGeometryPivot) {
        controls.setTarget(targetWithOffset(lastGeometryPivot));
      } else {
        // Prefer the solved scene depth (when a derive-geometry node ran) over
        // the generic 30m default so the orbit radius matches this scene.
        const sceneDepth = data.camera_meta?.scene_depth_m;
        const pivotMax = sceneDepth ? sceneDepth * 1.5 : 30;
        controls.setTarget(targetWithOffset(groundPointInView(camera, pivotMax)));
      }
      controls.syncFromCamera();                     // init orbit state from recovered pose
    }
    recoveredData = data;
    // Stale-extraction cleanup + pause visibility: if the persisted
    // patch_angle was extracted from a DIFFERENT solve/image than the one
    // that just executed, clear it from the widget (the backend already
    // refuses it — this keeps the UI honest). And whenever the patch branch
    // is paused (no valid extraction) while patch_* outputs are actually
    // wired, SAY SO — a silently-skipped branch otherwise reads as "the
    // workflow ran and produced nothing" (reported live).
    try {
      const widget = node.widgets?.find((w) => w.name === "client_data");
      let pa = null;
      let cleared = false;
      if (widget?.value) {
        const existing = JSON.parse(widget.value);
        pa = existing.patch_angle || null;
        if (pa && data.solve_fingerprint && pa.fingerprint !== data.solve_fingerprint) {
          delete existing.patch_angle;
          widget.value = JSON.stringify(existing);
          widget.callback?.(widget.value);
          pa = null;
          cleared = true;
        }
        // 🧭 Safe-zone envelope follows the same staleness rule: re-apply a
        // matching measurement on every execution (the clamp lives on the
        // controller instance and dies with the page otherwise); clear it and
        // restore the default clamp when the solve/image changed.
        const env = existing.envelope || null;
        if (env) {
          if (data.solve_fingerprint && env.fingerprint !== data.solve_fingerprint) {
            delete existing.envelope;
            widget.value = JSON.stringify(existing);
            widget.callback?.(widget.value);
            controls.setLimits(null);
          } else if (typeof env.yawPlusDeg === "number") {
            controls.setLimits({
              thetaMin: -THREE.MathUtils.degToRad(env.yawMinusDeg),
              thetaMax: THREE.MathUtils.degToRad(env.yawPlusDeg),
              phiMin: -THREE.MathUtils.degToRad(env.phiMinusDeg),
              phiMax: THREE.MathUtils.degToRad(env.phiPlusDeg),
            });
          }
        }
      }
      const patchWired = (node.outputs || []).slice(6, 10)
        .some((o) => (o.links || []).length > 0);
      if (!pa && patchWired) {
        angleHud.textContent =
          (cleared
            ? "📐 Patch angle cleared — the source image/solve changed.\n"
            : "📐 No patch angle extracted for this image.\n") +
          "The patch branch (Qwen generation / AddPatchView / exports)\n" +
          "is PAUSED — orbit to your target view and click\n" +
          "📐 Extract Angle to run it.      [✕]";
        angleHud.style.display = "block";
        angleHud.onclick = (e) => { angleHud.style.display = "none"; e.stopPropagation(); };
      }
    } catch (_) { /* malformed client_data — leave it to the backend guard */ }
    applyOutputProfilePreview(data.output_profile || atlasOutputProfileFromWidgets(getLinkedControlsNode(node) || {}));
    updateLinkedOutputDesk(data);
  }

  // Orbit pivot from the DERIVED geometry: the recovered camera's central
  // view ray at the MEDIAN sampled vertex depth (was a Box3 bounding-box
  // center until 2026-07-09 — see the inline comment for why that parked the
  // pivot deep behind the subject on full-scene relief meshes). Excludes
  // "projection_backdrop" (the always-emitted flat catch-all far plane, same
  // one 🎬 Backdrop toggles); patch/clean-plate sources live in their own
  // atlas_patch_N groups and are never included. Called once
  // buildDerivedProxies has real geometry to measure (setProxies, below) to
  // REPLACE applyRecoveredView's ground-point-in-view fallback. Returns null
  // when there's no derived geometry yet, in which case that fallback stays.
  function computeGeometryPivot(data) {
    const group = scene.getObjectByName("atlas_derived_proxies");
    if (!group?.children?.length) return null;
    // Recovered camera origin + forward from the payload — NOT the live
    // camera: the pivot is recomputed on every execution and must not depend
    // on wherever the user has orbited to.
    let origin, forward;
    if (data?.view_matrix) {
      const flat = data.view_matrix.flat();
      const vm = new THREE.Matrix4();
      vm.set(flat[0], flat[1], flat[2], flat[3], flat[4], flat[5], flat[6],
             flat[7], flat[8], flat[9], flat[10], flat[11], flat[12], flat[13],
             flat[14], flat[15]);
      const c2w = vm.clone().invert();
      origin = new THREE.Vector3().setFromMatrixPosition(c2w);
      // Camera looks down -Z in camera space -> world forward = -(3rd column).
      forward = new THREE.Vector3(
        -c2w.elements[8], -c2w.elements[9], -c2w.elements[10]).normalize();
    } else {
      origin = camera.position.clone();
      forward = camera.getWorldDirection(new THREE.Vector3());
    }
    // Pivot = the central view ray at the MEDIAN sampled vertex depth — not
    // the bounding-box center. A Box3 center is (min+max)/2, i.e. dominated
    // by tails: a single full-scene relief mesh (the hidden-geometry
    // workflows' base geometry) spans near-foreground to the far clip plus
    // fill/outpaint skirts, which parked the pivot deep behind the subject
    // (artist-reported 2026-07-09). The median vertex depth is the depth of
    // the middle of the visible surface AREA (relief grids sample the image
    // uniformly), which matches "the middle of what the photo shows".
    const depths = [];
    group.updateMatrixWorld(true);
    const v = new THREE.Vector3();
    group.children.forEach((mesh) => {
      if (mesh.name === "projection_backdrop") return;
      const pos = mesh.geometry?.attributes?.position;
      if (!pos?.count) return;
      const stride = Math.max(1, Math.floor(pos.count / 800));
      for (let i = 0; i < pos.count; i += stride) {
        v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld);
        const d = v.sub(origin).dot(forward);
        if (d > 0 && Number.isFinite(d)) depths.push(d);
      }
    });
    if (!depths.length) return null;
    depths.sort((a, b) => a - b);
    const median = depths[Math.floor(depths.length / 2)];
    return origin.clone().addScaledVector(forward, median);
  }

  // Layered VP / horizon / ground diagnostic diagram. viewBox uses the
  // SOLVE's native image pixel space (not the canvas render resolution) so
  // vanishing-point/horizon positions need no rescaling — the SVG's own
  // aspect-preserving scaling maps it onto the canvas automatically. VP
  // marker fan-lines run from the image corners to each VP position; a VP
  // far outside the frame is simply clipped by the SVG's default
  // overflow:hidden, leaving just the converging lines visible at the edge.
  function updateDiagramOverlay(data) {
    const iw = data.image_width || 1;
    const ih = data.image_height || 1;
    diagramSvg.setAttribute("viewBox", `0 0 ${iw} ${ih}`);
    gVpLines.replaceChildren();
    gHorizon.replaceChildren();
    gGround.replaceChildren();

    const VP_COLORS = { left: "#ff7832", right: "#32a0ff", vertical: "#50dc64" };
    const corners = [[0, 0], [iw, 0], [iw, ih], [0, ih]];
    const fontPx = Math.max(10, iw * 0.014);

    let hzY = ih * 0.45; // fallback split if no horizon was solved
    const hz = data.horizon_line;
    if (hz && hz.endpoints_px) {
      const [p0, p1] = hz.endpoints_px;
      const line = document.createElementNS(svgNS, "line");
      line.setAttribute("x1", p0[0]); line.setAttribute("y1", p0[1]);
      line.setAttribute("x2", p1[0]); line.setAttribute("y2", p1[1]);
      line.setAttribute("stroke", "#ffe050");
      line.setAttribute("stroke-width", String(Math.max(1, iw * 0.0015)));
      gHorizon.appendChild(line);
      hzY = (p0[1] + p1[1]) / 2;

      const label = document.createElementNS(svgNS, "text");
      label.setAttribute("x", String(iw * 0.02));
      label.setAttribute("y", String(Math.max(fontPx + 2, hzY - 6)));
      label.setAttribute("fill", "#ffe050");
      label.setAttribute("font-size", String(fontPx));
      label.textContent = `Horizon (${Math.round((hz.confidence || 0) * 100)}%)`;
      gHorizon.appendChild(label);
    }

    // Ground: shaded region below the horizon.
    const groundRect = document.createElementNS(svgNS, "rect");
    groundRect.setAttribute("x", "0"); groundRect.setAttribute("y", String(hzY));
    groundRect.setAttribute("width", String(iw));
    groundRect.setAttribute("height", String(Math.max(0, ih - hzY)));
    groundRect.setAttribute("fill", "#3caa50");
    gGround.appendChild(groundRect);

    // Vanishing points. Empty on the learned (GeoCalib) solve path — it
    // predicts focal+gravity directly rather than via classical VP detection —
    // so this layer only populates when the solve used detect_vanishing_points.
    (data.vanishing_points || []).forEach((vp) => {
      const [vx, vy] = vp.position_px;
      const color = VP_COLORS[vp.direction_label] || "#cccccc";
      corners.forEach(([cx, cy]) => {
        const ln = document.createElementNS(svgNS, "line");
        ln.setAttribute("x1", String(cx)); ln.setAttribute("y1", String(cy));
        ln.setAttribute("x2", String(vx)); ln.setAttribute("y2", String(vy));
        ln.setAttribute("stroke", color);
        ln.setAttribute("stroke-width", String(Math.max(0.75, iw * 0.0008)));
        ln.setAttribute("opacity", "0.55");
        gVpLines.appendChild(ln);
      });
      const dot = document.createElementNS(svgNS, "circle");
      dot.setAttribute("cx", String(vx)); dot.setAttribute("cy", String(vy));
      dot.setAttribute("r", String(Math.max(4, iw * 0.006)));
      dot.setAttribute("fill", color);
      gVpLines.appendChild(dot);
      const lbl = document.createElementNS(svgNS, "text");
      lbl.setAttribute("x", String(vx + iw * 0.01)); lbl.setAttribute("y", String(vy - iw * 0.01));
      lbl.setAttribute("fill", color); lbl.setAttribute("font-size", String(fontPx));
      lbl.textContent = `${vp.direction_label || "vp"} (${Math.round((vp.confidence || 0) * 100)}%)`;
      gVpLines.appendChild(lbl);
    });
  }

  // Solved latent-camera metadata HUD: lens (focal/sensor/FOV), distance
  // (camera height, scene depth), and solve provenance/confidence.
  function updateMetaHud(data) {
    const m = data.camera_meta || {};
    const lines = [];
    if (m.focal_mm != null) {
      const fov = m.fov_h_deg != null ? `  (FOV ${m.fov_h_deg.toFixed(1)}°)` : "";
      lines.push(`Lens      ${m.focal_mm.toFixed(1)}mm${fov}`);
    }
    if (m.sensor_mm != null) lines.push(`Sensor    ${m.sensor_mm.toFixed(1)}mm`);
    if (m.camera_height_m != null) lines.push(`Height    ${m.camera_height_m.toFixed(2)}m`);
    if (m.scene_depth_m != null) lines.push(`Scene depth ~${m.scene_depth_m.toFixed(1)}m`);
    if (m.confidence != null) lines.push(`Confidence  ${Math.round(m.confidence * 100)}%`);
    if (m.source_method) lines.push(`Method    ${m.source_method}`);
    if (m.scale_source) lines.push(`Scale     ${m.scale_source}`);
    metaHud.textContent = lines.join("\n") || "(no camera metadata)";
  }

  // Return setter so caller can apply camera and background image
  return {
    mountControls,
    applyCamera(data) {
      applyRecoveredView(data);
    },
    setDiagnostics(data) {
      updateDiagramOverlay(data);
      updateMetaHud(data);
    },
    setProxies(data) {
      // Build the Python-derived projection proxies and (re)create the shared
      // projection material from the recovered camera + source photo.
      buildDerivedProxies(scene, data);
      setBackdropVisible(backdropVisible); // reapply — fresh meshes default to visible
      // Recentre the orbit pivot on the actual generated geometry now that it
      // exists (replaces applyRecoveredView's ground-point fallback). Only
      // re-syncs the orbit SPHERE PARAMETERS from wherever the camera already
      // is — never moves the camera itself, so this can't disrupt an
      // in-progress inspection even on a re-execution (e.g. ⏺ Bake Proxy Path).
      const geometryPivot = computeGeometryPivot(data);
      if (geometryPivot) {
        lastGeometryPivot = geometryPivot;
        controls.setTarget(targetWithOffset(geometryPivot));
        controls.syncFromCamera();
      }
      loadProjectionTexture(data, (tex) => {
        const old = projMaterial;
        projMaterial = makeProjectionMaterial(data, tex);
        if (projectionOn) applyProjection(true);
        if (old) { old.uniforms?.uTexture?.value?.dispose?.(); old.dispose(); }
      });
      // Multi-angle patch sources: each builds its own geometry + a projection
      // material (bound to its camera+image, facing-masked) that layers over
      // the primary to fill areas the primary camera couldn't see.
      buildPatchSources(scene, data, () => { if (projectionOn) applyProjection(true); });
      if (projectionOn) applyProjection(true); // grey until textures arrive
      buildBandBox(); // rebuild the 📏 overlay against this execution's geometry
      placeDefaultLights(); // relight lights follow the (now-built) geometry + scale
    },
    setBackground(imgBase64) {
      if (!imgBase64 || !THREE) return;
      const loader = new THREE.TextureLoader();
      loader.load(imgBase64, (tex) => {
        // Swap the OLD bgMesh out here, at callback time — reading the live
        // `bgMesh` closure variable right before reassigning it — rather than
        // at call time (before this async load even started). refreshFromSolve
        // deliberately fires twice per execution (node.onExecuted + the
        // api "executed" listener, for cross-version robustness), so two
        // overlapping setBackground calls are the normal case, not an edge
        // case. Checking/disposing at call time meant neither of two
        // in-flight loads ever saw the other's finished mesh, so each left
        // its predecessor orphaned in the scene — permanently, since nothing
        // still referenced it — frozen at that execution's old camera pose.
        // Reading the current value inside this (synchronous, non-interleaved)
        // callback body instead means each completed load always tears down
        // whatever is currently in the scene, regardless of firing order.
        if (bgMesh) { scene.remove(bgMesh); bgMesh.geometry.dispose(); bgMesh.material.map?.dispose(); bgMesh.material.dispose(); }
        tex.colorSpace = THREE.SRGBColorSpace;
        // Size a plane to exactly fill the recovered camera's frustum at distance D
        // and place it along the view axis, so the photo aligns with the 3D scene
        // from the recovered ("Camera View") perspective. depthTest:false keeps it a
        // backdrop behind any placed geometry.
        const D = 12;
        const fovRad = (camera.fov * Math.PI) / 180;
        const ph = 2 * D * Math.tan(fovRad / 2);
        const pw = ph * (camera.aspect || 1);
        // As a see-through backdrop the plane must cover well PAST the recovered
        // frustum so orbiting off-axis doesn't run the view off its edge into
        // black. Enlarge the plane by K but keep the photo itself frustum-sized
        // (UVs scaled about centre by 1/K) and clamp the border, so the photo
        // stays aligned with the geometry at Camera View while the outer ring is
        // the edge pixels stretched outward — a soft fill, never black.
        const K = 3.0;
        const geo = new THREE.PlaneGeometry(pw * K, ph * K);
        const uvA = geo.attributes.uv;
        for (let i = 0; i < uvA.count; i++) {
          uvA.setXY(i, 0.5 + (uvA.getX(i) - 0.5) * K, 0.5 + (uvA.getY(i) - 0.5) * K);
        }
        uvA.needsUpdate = true;
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
        tex.needsUpdate = true;
        // In the outer ring (UV outside [0,1]) don't STREAK the clamped edge
        // pixels — softly fade toward the photo's own average colour, so the
        // backdrop dissolves into a soft ambient instead of stretched streaks
        // (still never black). Average = a 1x1 downscale of the photo.
        let ambient = new THREE.Color(0.02, 0.02, 0.03);
        try {
          const cnv = document.createElement("canvas"); cnv.width = cnv.height = 1;
          const cx = cnv.getContext("2d"); cx.drawImage(tex.image, 0, 0, 1, 1);
          const px = cx.getImageData(0, 0, 1, 1).data;
          ambient = new THREE.Color(px[0] / 255, px[1] / 255, px[2] / 255).convertSRGBToLinear();
        } catch (e) { /* tainted canvas -> keep the dark default */ }
        const mat = new THREE.ShaderMaterial({
          uniforms: { map: { value: tex }, uAmbient: { value: ambient } },
          depthWrite: false, depthTest: false,
          vertexShader:
            "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
          fragmentShader:
            "uniform sampler2D map; uniform vec3 uAmbient; varying vec2 vUv;" +
            "vec3 l2s(vec3 c){ return mix(pow(c,vec3(0.41666))*1.055-0.055, c*12.92, vec3(lessThanEqual(c,vec3(0.0031308)))); }" +
            "void main(){ vec3 p = texture2D(map, clamp(vUv,0.0,1.0)).rgb;" +
            " vec2 d = max(vec2(0.0), max(-vUv, vUv-1.0)); float f = smoothstep(0.0, 0.45, length(d));" +
            " gl_FragColor = vec4(l2s(mix(p, uAmbient, f)), 1.0); }",
        });
        bgMesh = new THREE.Mesh(geo, mat);
        // Deepest renderOrder so it draws FIRST as a pure background canvas: every
        // projected layer (renderOrder >= 1 via priorityToRenderOrder, primary
        // 100000) draws on top and overwrites it where it paints, while any pixel
        // the projection DISCARDS (matte-cut silhouette, torn quad, out-of-frame)
        // reveals this backdrop photo instead of the black clear colour. depthTest
        // false means it can never occlude geometry regardless of its D distance.
        bgMesh.renderOrder = -100000;
        const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
        bgMesh.position.copy(camera.position).addScaledVector(fwd, D);
        bgMesh.quaternion.copy(camera.quaternion);
        // The "see-through to backdrop": stays visible UNDER 📽 Project so the
        // matte/tear outliers see through to the photo, not black — unless the 🕳
        // See-through toggle is off. Independent of the 🎬 Backdrop toggle (which
        // only governs the projection_backdrop plane).
        bgMesh.visible = projectionOn ? seeThroughOn : true;
        scene.add(bgMesh);
        node._atlasBgMesh = bgMesh;
      });
    },
  };
}

// ---------------------------------------------------------------------------
// Cross-node linking: AtlasBlockoutViewport <-> AtlasViewportControls.
//
// The `controls` input/output carries no real data — its only job is to let
// a graph LINK exist between the two nodes so each side's frontend JS can
// find the other's live node instance (via node.graph, not app.graph, so
// this keeps working inside subgraphs) and either reparent DOM into it
// (viewport -> controls) or trigger a reparent on it (controls -> viewport).
// This is a normal graph connection for wiring purposes only; nothing about
// it depends on ComfyUI ever actually executing/transmitting a value.
// ---------------------------------------------------------------------------
function getLinkedControlsNode(viewportNode) {
  const idx = viewportNode.findInputSlot?.("controls") ?? -1;
  const linkId = idx >= 0 ? viewportNode.inputs?.[idx]?.link : null;
  if (linkId == null) return null;
  const graph = viewportNode.graph;
  const link = graph?.links?.[linkId];
  return link ? graph.getNodeById(link.origin_id) : null;
}

function getLinkedViewportNodes(controlsNode) {
  const linkIds = controlsNode.outputs?.[0]?.links;
  if (!linkIds?.length) return [];
  const graph = controlsNode.graph;
  return linkIds
    .map((id) => graph?.links?.[id])
    .filter(Boolean)
    .map((link) => graph.getNodeById(link.target_id))
    .filter(Boolean);
}

function atlasWidget(node, name) {
  return node.widgets?.find((w) => w.name === name) || null;
}

function atlasWidgetValue(node, name, fallback = "") {
  const widget = atlasWidget(node, name);
  return widget?.value ?? fallback;
}

function atlasSetWidgetValue(node, name, value) {
  const widget = atlasWidget(node, name);
  if (!widget) return;
  widget.value = value;
  widget.callback?.(widget.value);
}

function atlasOutputProfileFromWidgets(node) {
  return {
    config_label: atlasWidgetValue(node, "config_label", "ACES 2.0 / Studio"),
    config_path: atlasWidgetValue(node, "config_path", ""),
    working_colorspace: atlasWidgetValue(node, "working_colorspace", "ACEScg"),
    output_colorspace: atlasWidgetValue(node, "output_colorspace", "ACES - ACEScg"),
    display: atlasWidgetValue(node, "display", "sRGB - Display"),
    view: atlasWidgetValue(node, "view", "ACES 2.0 SDR-video"),
    // look/lut_path/exposure/gamma widgets removed 2026-07-10 (redundant on
    // the node — exposure duplicated the viewport's own ☀ control); the
    // profile keys stay at neutral defaults so applyOutputProfilePreview and
    // downstream consumers keep their contract unchanged.
    look: "None",
    lut_path: "",
    exposure: 0,
    gamma: 1,
    display_trim: Number(atlasWidgetValue(node, "display_trim", 1)) || 1,
    preview_only: true,
  };
}

function buildAtlasOutputDesk(node, container) {
  container.innerHTML = "";
  container.style.cssText =
    "width:100%;display:flex;flex-direction:column;gap:0;background:#111318;color:#d7dce5;" +
    "border:1px solid #333846;border-radius:6px;overflow:hidden;font:11px/1.35 system-ui,sans-serif;";

  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;gap:8px;padding:6px 8px;background:#191d25;border-bottom:1px solid #303642;";
  const title = document.createElement("strong");
  title.textContent = "Atlas Output Desk";
  title.style.cssText = "font-size:12px;color:#f0f4ff;";
  const badges = document.createElement("div");
  badges.style.cssText = "display:flex;gap:5px;flex-wrap:wrap;margin-left:auto;";
  function badge(text, tone = "neutral") {
    const el = document.createElement("span");
    const colors = {
      neutral: ["#242a34", "#8b96a8"],
      proxy: ["#332716", "#f0b65a"],
      shot: ["#153024", "#6ee7a8"],
      ocio: ["#222545", "#aeb8ff"],
    }[tone] || ["#242a34", "#8b96a8"];
    el.textContent = text;
    el.style.cssText = `padding:2px 6px;border-radius:999px;background:${colors[0]};color:${colors[1]};border:1px solid rgba(255,255,255,.08);`;
    return el;
  }
  const proxyBadge = badge("Proxy/LDR", "proxy");
  const resBadge = badge("Output --", "neutral");
  const shotBadge = badge("ShotCam --", "shot");
  const ocioBadge = badge("OCIO preview", "ocio");
  badges.append(proxyBadge, resBadge, shotBadge, ocioBadge);
  header.append(title, badges);

  const tabBar = document.createElement("div");
  tabBar.style.cssText = "display:flex;gap:1px;background:#0d0f14;border-bottom:1px solid #2e3440;";
  const panels = {};
  const panelWrap = document.createElement("div");
  panelWrap.style.cssText = "min-height:96px;background:#151820;";

  function makePanel(name) {
    const panel = document.createElement("div");
    panel.style.cssText = "display:none;padding:6px;gap:6px;flex-wrap:wrap;align-items:center;";
    panelWrap.appendChild(panel);
    panels[name] = panel;
    const btn = document.createElement("button");
    btn.textContent = name;
    btn.style.cssText = "flex:1;padding:5px 6px;border:0;background:#171b23;color:#aeb6c5;font-size:11px;cursor:pointer;";
    btn.onclick = () => {
      for (const [key, p] of Object.entries(panels)) p.style.display = key === name ? "flex" : "none";
      [...tabBar.children].forEach((child) => {
        child.style.background = child === btn ? "#252b37" : "#171b23";
        child.style.color = child === btn ? "#f4f7ff" : "#aeb6c5";
      });
    };
    tabBar.appendChild(btn);
    return { panel, btn };
  }

  const view = makePanel("View");
  makePanel("Plates");
  const color = makePanel("Color");
  const passes = makePanel("Passes");
  const path = makePanel("Path");
  const lights = makePanel("Lights");

  const toolbarSlot = document.createElement("div");
  toolbarSlot.style.cssText = "display:flex;flex-wrap:wrap;align-items:center;gap:4px;width:100%;";
  view.panel.appendChild(toolbarSlot);

  const plateInfo = document.createElement("div");
  plateInfo.style.cssText = "display:grid;grid-template-columns:auto 1fr;gap:4px 8px;width:100%;color:#b8c0cf;";
  panels.Plates.appendChild(plateInfo);

  function addColorField(label, widgetName, type = "text", attrs = {}) {
    const wrap = document.createElement("label");
    wrap.style.cssText = "display:grid;grid-template-columns:92px minmax(120px,1fr);align-items:center;gap:6px;width:100%;";
    const lab = document.createElement("span");
    lab.textContent = label;
    lab.style.cssText = "color:#9aa5b8;";
    const input = document.createElement("input");
    input.type = type;
    input.value = atlasWidgetValue(node, widgetName, attrs.defaultValue ?? "");
    input.style.cssText = "min-width:0;background:#0d1016;color:#edf2ff;border:1px solid #343b4a;border-radius:4px;padding:3px 5px;font-size:11px;";
    Object.assign(input, attrs);
    input.onchange = input.oninput = () => {
      atlasSetWidgetValue(node, widgetName, type === "number" ? Number(input.value) : input.value);
      const profile = atlasOutputProfileFromWidgets(node);
      getLinkedViewportNodes(node).forEach((vp) => vp._atlasApplyOutputProfilePreview?.(profile));
      node._atlasOutputDeskUpdate?.({ output_profile: profile });
    };
    wrap.append(lab, input);
    color.panel.appendChild(wrap);
    return input;
  }
  addColorField("Config", "config_label");
  addColorField("Config path", "config_path");
  addColorField("Working", "working_colorspace");
  addColorField("Output", "output_colorspace");
  addColorField("Display", "display");
  addColorField("View", "view");
  addColorField("Trim", "display_trim", "number", { step: "0.05", min: "0" });
  const previewNote = document.createElement("div");
  previewNote.textContent = "Display-inferred preview only. Final OCIO/LUT fidelity belongs to OCIO Write, Nuke, Maya, or Resolve.";
  previewNote.style.cssText = "width:100%;padding:5px 6px;border-radius:4px;background:#1d2130;color:#b8c4ff;";
  color.panel.appendChild(previewNote);

  const passInfo = document.createElement("div");
  passInfo.textContent = "Proxy/LDR passes: shaded, depth, normal, mask. Use OCIO/DCC for final float EXR renders.";
  passInfo.style.cssText = "width:100%;padding:5px 6px;border-radius:4px;background:#211d14;color:#f0c177;";
  passes.panel.appendChild(passInfo);

  const pathSlot = document.createElement("div");
  pathSlot.style.cssText = "display:flex;flex-wrap:wrap;align-items:center;gap:4px;width:100%;";
  path.panel.appendChild(pathSlot);

  const lightSlot = document.createElement("div");
  lightSlot.style.cssText = "display:flex;flex-wrap:wrap;align-items:center;gap:4px;width:100%;";
  lights.panel.appendChild(lightSlot);
  const lightInfo = document.createElement("div");
  lightInfo.textContent = "Movable point lights: always relight the grey/shaded preview; only affect 📽 Project once a light's intensity is raised above 0.";
  lightInfo.style.cssText = "width:100%;padding:5px 6px;border-radius:4px;background:#1d2130;color:#b8c4ff;";
  lights.panel.appendChild(lightInfo);

  container.append(header, tabBar, panelWrap);
  view.btn.click();

  node._atlasToolbarContainer = toolbarSlot;
  node._atlasPathContainer = pathSlot;
  node._atlasLightContainer = lightSlot;
  node._atlasControlsContainer = toolbarSlot;
  node._atlasOutputDeskUpdate = (data = {}) => {
    const width = data.target_width || data.width || data.output_width;
    const height = data.target_height || data.height || data.output_height;
    resBadge.textContent = width && height ? `Output ${Math.round(width)}x${Math.round(height)}` : "Output --";
    shotBadge.textContent = data.shot_cam ? "ShotCam on" : "ShotCam/profile";
    const profile = data.output_profile || {};
    ocioBadge.textContent = profile.output_colorspace ? `OCIO ${profile.output_colorspace}` : "OCIO preview";
    const plate = data.source_plate || {};
    plateInfo.innerHTML = "";
    const rows = [
      ["Plate", plate.image_path || "Proxy preview only"],
      ["Colorspace", plate.colorspace || "unspecified"],
      ["Bit depth", plate.bit_depth || "unknown"],
      ["Role", plate.role || "source"],
      ["Status", plate.is_proxy === false ? "File-backed final plate" : "Proxy/LDR preview"],
    ];
    for (const [k, v] of rows) {
      const key = document.createElement("span"); key.textContent = k; key.style.cssText = "color:#7f8a9c;";
      const val = document.createElement("span"); val.textContent = String(v); val.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
      plateInfo.append(key, val);
    }
  };
  node._atlasOutputDeskUpdate();
}

// ---------------------------------------------------------------------------
// ComfyUI extension registration
// ---------------------------------------------------------------------------
app.registerExtension({
  name: "AtlasCamera.Blockout",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AtlasBlockoutViewport") return;
    await loadThree();
  },

  // Migration shim: workflows saved before 2026-07-10 serialized 11 (or 12,
  // with a trailing DOM-widget placeholder) widgets_values on
  // AtlasViewportControls — look/lut_path/exposure/gamma sat at indices 6-9
  // ahead of display_trim. After those widgets were removed, a stale array
  // feeds the old `look` string ("None") into display_trim and the prompt
  // fails FLOAT validation. widgets_values is positional, so heal it here,
  // BEFORE litegraph assigns widget values (onConfigure fires too late for
  // that). New-layout arrays (length 7/8) pass through untouched.
  beforeConfigureGraph(graphData) {
    for (const n of graphData?.nodes ?? []) {
      if (n.type !== "AtlasViewportControls") continue;
      const wv = n.widgets_values;
      if (Array.isArray(wv) && wv.length >= 11) {
        wv.splice(6, 4);
        console.log("[AtlasCamera] migrated stale AtlasViewportControls widgets_values (node", n.id, ")");
      }
    }
  },

  async nodeCreated(node) {
    if (node.comfyClass === "AtlasViewportControls") {
      // Second half of the widgets_values migration: a node PASTED from an
      // old clipboard bypasses beforeConfigureGraph, and litegraph assigns
      // widget values before onConfigure — so sanitize AFTER configure runs.
      // display_trim is the only numeric widget; a stale array shifts the
      // old `look` string into it (NaN). Install synchronously, before the
      // await below, or configure fires first and we miss it.
      const prevControlsConfigure = node.onConfigure;
      node.onConfigure = function (...args) {
        const out = prevControlsConfigure?.apply(this, args);
        const dt = this.widgets?.find((w) => w.name === "display_trim");
        if (dt && !Number.isFinite(Number(dt.value))) dt.value = 1;
        return out;
      };
      // Wait one tick for ComfyUI to finish building the node DOM.
      await new Promise((r) => setTimeout(r, 0));
      const container = document.createElement("div");
      container.style.cssText = "width:100%;display:flex;flex-direction:column;gap:0;";
      pinDomWidgetFullWidth(node.addDOMWidget("atlas_viewport_controls", "div", container, {
        serialize: false,
        getValue() { return null; },
        setValue() {},
      }));
      buildAtlasOutputDesk(node, container);
      // Nudge any already-created, already-linked viewport(s) to reparent
      // into us now that our container exists — covers the case where the
      // viewport node's own nodeCreated ran first (creation order isn't
      // guaranteed when a saved workflow loads both nodes at once).
      getLinkedViewportNodes(node).forEach((vp) => vp._atlasRemount?.());
      const prevOnConnectionsChange = node.onConnectionsChange;
      node.onConnectionsChange = function (...args) {
        prevOnConnectionsChange?.apply(this, args);
        getLinkedViewportNodes(node).forEach((vp) => vp._atlasRemount?.());
      };
      return;
    }

    if (node.comfyClass !== "AtlasBlockoutViewport") return;

    // Track whether this node is being restored from a saved workflow.
    // onConfigure fires only for deserialized nodes, and it fires during
    // graph.configure — i.e. after this handler's first await suspends — so
    // the hook MUST be installed here, synchronously, to catch it. This is
    // what lets the default-size bump below apply only to fresh nodes.
    let restoredFromSave = false;
    const prevOnConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      restoredFromSave = true;
      return prevOnConfigure?.apply(this, args);
    };

    await loadThree();
    if (!THREE) return;

    // Wait one tick for ComfyUI to finish building the node DOM
    await new Promise((r) => setTimeout(r, 0));

    // Read the long-edge resolution widget (W×H is derived from the source image
    // aspect on execution, so the viewport inherits the image's aspect).
    const resWidget = node.widgets?.find((w) => w.name === "resolution");
    node._atlasResolution = resWidget?.value ?? 768;

    // Create a DOM container widget. height:100% (not the default natural-
    // content-height sizing) is what actually lets the canvas inside grow
    // when the node is resized — see canvasWrap's comment above for why.
    // min-width:0 for the same reason as canvasWrap's — defense in depth in
    // case ComfyUI's own widget-hosting layout is flex too.
    const container = document.createElement("div");
    container.style.cssText = "width:100%;height:100%;min-width:0;display:flex;flex-direction:column;gap:0;overflow:hidden;";

    const domWidget = node.addDOMWidget("atlas_viewport", "div", container, {
      serialize: false,
      getValue() { return null; },
      setValue() {},
      // Sanctioned sizing hooks (DOMWidgetOptions.getMinHeight/getMaxHeight,
      // scripts/domWidget.ts) instead of leaving LiteGraph's own layout math
      // to fall back to its hardcoded 50px default — gives it an accurate
      // floor and a practical ceiling, so dragging larger is never fought.
      getMinHeight() { return 240; },
      getMaxHeight() { return 8192; },
    });
    pinDomWidgetFullWidth(domWidget);

    installViewportSizeTrace(node, domWidget, container);

    const ui = buildNodeUI(node, container);

    // Reparent the toolbar/panel into a connected AtlasViewportControls node
    // (leaving this node perspective-only), or fall back to appending them
    // locally when nothing is connected — fully backward-compatible with
    // workflows saved before AtlasViewportControls existed.
    node._atlasRemount = ui?.mountControls;
    ui?.mountControls();

    // Freshly added nodes default to a large preview (see the constants'
    // comment); nodes restored from a save keep their stored size. Math.max
    // so a future larger computed default is never shrunk.
    if (!restoredFromSave) {
      node.setSize([
        Math.max(node.size[0], ATLAS_VIEWPORT_DEFAULT_WIDTH),
        Math.max(node.size[1], ATLAS_VIEWPORT_DEFAULT_HEIGHT),
      ]);
      node.graph?.setDirtyCanvas(true, true);
    }
    const prevOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
      prevOnConnectionsChange?.apply(this, args);
      ui?.mountControls();
    };

    // On node execution complete: apply recovered camera + source image +
    // derived projection proxies.
    const refreshFromSolve = async () => {
      const cameraData = await fetchCameraData(String(node.id));
      if (!cameraData) return;
      ui?.applyCamera(cameraData);
      if (cameraData.source_image_b64) {
        ui?.setBackground(cameraData.source_image_b64);
      }
      ui?.setProxies(cameraData);
      ui?.setDiagnostics(cameraData);
    };
    node.onExecuted = refreshFromSolve;

    // Restore from the SERVER's payload cache on creation: after a page
    // reload (or when ComfyUI serves this node from its execution cache and
    // never emits "executed"), the viewport would otherwise sit on an empty
    // grid even though a perfectly good solve exists — chronic in the staged
    // master workflow, whose whole rhythm is re-queues with an unchanged
    // stage 0. /atlas/camera_data/{id} is LRU-kept server-side across
    // queues, so a miss is harmless and a hit repopulates instantly.
    setTimeout(() => { refreshFromSolve(); }, 300);

    // node.onExecuted only fires when ComfyUI delivers a "ui" payload for this
    // node — subscribe to the api-level executed event too, so the viewport
    // refreshes regardless of frontend version quirks.
    const onApiExecuted = (event) => {
      const d = event?.detail;
      const executedId = d?.node ?? d?.display_node;
      if (String(executedId) === String(node.id)) refreshFromSolve();
    };
    api.addEventListener("executed", onApiExecuted);

    // Track resolution widget changes (applied on the next execution's resize).
    if (resWidget) resWidget.callback = (v) => { node._atlasResolution = v; };

    // Cleanup on node removal. MUST CHAIN, never assign: addDOMWidget has
    // already installed ComfyUI's own onRemoved (useChainCallback in
    // domWidget.ts) which detaches the widget's DOM from the document —
    // clobbering it left every replaced viewport's container + WebGL canvas
    // + overlays ORPHANED in the page, where they rendered in normal
    // document flow (floating slider stubs near the top, a body-wide canvas
    // sheet at the bottom — found live on the AtlasInput quickstart after a
    // workflow switch, confirmed by a 0×0-rect orphan canvas in the DOM).
    const prevOnRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
      prevOnRemoved?.apply(this, args);
      api.removeEventListener("executed", onApiExecuted);
      node._atlasSizeTraceCleanup?.();
      node._atlasFsCleanup?.();
      cancelAnimationFrame(node._atlasRafId);
      node._atlasRenderer?.dispose();
      node._atlasControls?.dispose();
      node._atlasFly?.dispose();
      // Belt-and-braces for frontends whose addDOMWidget cleanup semantics
      // differ: removing an already-detached element is a no-op.
      container.remove();
    };
  },
});
