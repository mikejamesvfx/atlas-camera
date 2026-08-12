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

// The preview BACKBUFFER may exceed the logical preview size, because the canvas
// is displayed CSS-scaled (`object-fit:contain`) — rendering 1:1 into it wastes
// the display's real pixels and leaves every silhouette a screen pixel coarser
// than it needs to be. This is supersampling for the DISPLAY only: it is
// unrelated to the `resolution` widget, which sets node._atlasW/_atlasH, the
// dimensions Render Proxy Passes and baked Camera Path frames actually use.
// The two were conflated once and the supersample was wrongly refused as a
// violation of "render resolution is governed solely by the resolution widget".
//
// Bounded rather than devicePixelRatio-driven: an unbounded DPR would quadruple
// the fill cost of a dense relief mesh on a high-DPR laptop for no visible gain.
//
// Raised from 2048/2x (2026-08-11) because that bound was set against a canvas
// displayed at roughly preview size, and it silently inverts on a big one: an
// 8K display draws this canvas across far more than 2048 device pixels, so the
// buffer stops being a SUPERsample and becomes an UNDERsample stretched to fit,
// which is exactly when a silhouette looks worst. 3x from the 1280 logical
// preview reaches 3840 — about 1:1 on an 8K panel — at 2.25x the fill of the
// old cap. Still bounded, and still display-only: it cannot touch
// node._atlasW/_atlasH, which is what Render Proxy Passes and baked frames use.
const ATLAS_VIEWPORT_BACKBUFFER_MAX_LONG_EDGE = 3840;
const ATLAS_VIEWPORT_BACKBUFFER_MAX_SCALE = 3;

function atlasBackbufferScale(previewW, previewH) {
  const longEdge = Math.max(previewW || 0, previewH || 0, 1);
  return Math.max(1, Math.min(ATLAS_VIEWPORT_BACKBUFFER_MAX_SCALE,
                              ATLAS_VIEWPORT_BACKBUFFER_MAX_LONG_EDGE / longEdge));
}

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
  // setEnabled(false) is available to callers, but since the fly controller's
  // removal (2026-07-16) the orbit controller stays enabled even in Camera
  // Path mode — path playback re-poses the camera per-frame after input.
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
  let navShift = false, lastNavT = 0, navShakeEnv = 0;
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
      let moved = false;
      if (enabled && !dragging && pressed.size > 0) {
        const step = sph.radius * 0.15 * dt * (navShift ? 4 : 1);
        if (step > 0) {
          const forward = target.clone().sub(camera.position).normalize();
          const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
          const move = new THREE.Vector3();
          if (pressed.has("ArrowUp") || pressed.has("KeyW")) move.add(forward);
          if (pressed.has("ArrowDown") || pressed.has("KeyS")) move.sub(forward);
          if (pressed.has("ArrowRight")) move.add(right);
          if (pressed.has("ArrowLeft")) move.sub(right);
          if (pressed.has("KeyA") || pressed.has("KeyE")) move.y += 1;
          if (pressed.has("KeyD") || pressed.has("KeyQ")) move.y -= 1;
          if (move.lengthSq() > 0) {
            this.pan(move.normalize().multiplyScalar(step));
            moved = true;
          }
        }
      }
      // 🎬 handheld envelope: ramps toward 1 while the keys actually move the
      // camera, decays to 0 at rest — animate() scales the live nav shake by
      // it so navigation shake fades in/out instead of popping.
      navShakeEnv += ((moved ? 1 : 0) - navShakeEnv) * Math.min(1, dt * 5);
      if (!moved && navShakeEnv < 0.001) navShakeEnv = 0;
    },
    getNavShakeEnv() { return navShakeEnv; },
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

// NOTE (2026-07-16): the free-fly controller (createFlyControls, RMB+WASD/QE)
// that powered manual Camera Path keyframing was removed along with the
// keyframe editor — the panel is now five deterministic one-click moves (see
// the Camera Path block below). The orbit controller stays ENABLED in path
// mode; recover the fly controller from git history if free navigation is
// ever wanted again.

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

// Where the transition ribbon's fade begins, in ribbon_t. MIRRORS
// atlas_camera/core/transition_ribbon.py RIBBON_FADE_START — the same curve is
// baked into the exported GLB vertex alpha, so the viewport and a DCC show the
// same skirt. Pinned by tests/test_frontend_mirrors.py. Starting above 0 keeps
// the ribbon opaque where it meets the rim; a fade beginning AT the rim would
// reintroduce the soft-edged hole the tear exists to avoid.
const RIBBON_FADE_START = 0.15;

// Along-rim averaging applied to the skirt, in SOURCE-PLATE texels, reached at
// its outer edge and ramped in with ribbon_t. Each column is frozen to one
// texel, so unsmudged the skirt is a fan of flat radial streaks that band
// against each other; averaging across neighbouring columns bleeds the
// subject's own edge colour outward instead. VIEWPORT-ONLY — a DCC samples the
// same frozen UV and gets the unsmudged streak, so this softens the preview and
// does not change exported appearance.
const RIBBON_SMUDGE_TEXELS = 12.0;

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
  attribute float atlasEdgeRisk;
  varying float vAtlasEdgeRisk;
  attribute float atlasRibbonT;
  varying float vAtlasRibbonT;
  varying vec2 vAtlasBakedUv;
  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    vWorldNormal = normalize(mat3(modelMatrix) * normal);
    vAtlasEdgeRisk = atlasEdgeRisk;
    vAtlasRibbonT = atlasRibbonT;
    // The mesh's BAKED uv, which for a transition-ribbon vertex is the frozen
    // silhouette texel. Everything else here re-derives its texel by projecting
    // the world position (vImagePx), and that is exactly wrong for a skirt: it
    // extends outward in image space, so its projected pixel lands OUTSIDE the
    // subject and it samples the backdrop. Frozen UVs were only ever reaching
    // the exporters; the viewport threw them away.
    vAtlasBakedUv = uv;
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
  uniform float uMatteSoft;   // 1 = continuous visibility field, do not threshold
  uniform float uSoftStretch; // 0 = off; soft-layering smear fade strength
  uniform float uRibbonSmudge; // texels of along-rim averaging at the skirt's outer edge
  uniform float uLayerDebug;
  uniform vec3 uLayerTint;
  uniform sampler2D uPatchMask;
  uniform float uHasPatchMask;
  uniform vec3 uPatchTint;
  uniform mat4 uPatchViewMatrix;
  uniform float uPatchFx;
  uniform float uPatchFy;
  uniform float uPatchCx;
  uniform float uPatchCy;
  uniform vec2 uPatchImageSize;
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
  uniform sampler2D uPrimaryDepth;
  uniform float uHasPrimaryDepth;
  uniform vec2 uPrimaryDepthSize;
  uniform float uOccludePrimary;
  uniform float uOccludeBias;
  uniform float uOccludeFeather;
  uniform float uStretchStart;
  uniform float uStretchEnd;
  // 🎭 debug-matte isolate: a GLOBAL mask sampled at the fragment's PRIMARY-
  // camera projected uv (uDbg* = the recovered camera, NOT this source's own
  // projector — patches/outpainted skies project through different cameras, so
  // the isolate must be evaluated in one shared image space). Outside the
  // matte the display-space color is dimmed by uDebugMatteDim (0 = hard cull).
  uniform sampler2D uDebugMatte;
  uniform float uHasDebugMatte;
  uniform float uDebugMatteOn;
  uniform float uDebugMatteDim;
  uniform mat4 uDbgViewMatrix;
  uniform float uDbgFx;
  uniform float uDbgFy;
  uniform float uDbgCx;
  uniform float uDbgCy;
  uniform vec2 uDbgImageSize;
  varying vec2 vImagePx;
  varying float vCamZ;
  varying vec3 vWorldPos;
  varying vec3 vWorldNormal;
  varying float vAtlasEdgeRisk;
  varying float vAtlasRibbonT;
  varying vec2 vAtlasBakedUv;
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
  float atlasUnpackMetricDepth(vec2 depthUv) {
    vec3 pCol = texture2D(uPrimaryDepth, depthUv).rgb;
    float r = floor(pCol.r * 255.0 + 0.5);
    float g = floor(pCol.g * 255.0 + 0.5);
    float b = floor(pCol.b * 255.0 + 0.5);
    return (r * 65536.0 + g * 256.0 + b) / 1000.0;
  }
  float atlasRelativeDepthJump(float centerZ, float sampleZ) {
    if (sampleZ <= 0.0005) return 0.0;
    return abs(sampleZ - centerZ) / max(min(sampleZ, centerZ), 0.001);
  }
  void main() {
    if (vCamZ >= 0.0) discard;                    // behind the projector camera
    vec2 uv = vImagePx / uImageSize;
    // A transition-ribbon fragment samples the SILHOUETTE texel it was frozen
    // to, not the pixel it happens to project to. The skirt exists outside the
    // subject's outline, so re-projecting drags in whatever is behind it —
    // sky, backdrop, the white surround of a product plate — and paints the
    // transition with it (seen live as a white halo around a machine). Reading
    // the baked uv turns the skirt into what it was always specified to be: the
    // subject's own edge colour, extended outward. The y flip is the OBJ
    // bottom-left convention the mesh stores against this shader's top-left.
    bool isRibbon = vAtlasRibbonT > 0.0;
    // Taken here, in UNIFORM control flow: a derivative computed inside the
    // branch below would be undefined for the fragments that skip it.
    vec2 bakedGx = dFdx(vAtlasBakedUv);
    vec2 bakedGy = dFdy(vAtlasBakedUv);
    if (isRibbon) uv = vec2(vAtlasBakedUv.x, 1.0 - vAtlasBakedUv.y);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) discard;
    float coverage = 1.0;
    float depthEdge = 0.0;
    vec2 texelDx = dFdx(uv) * uImageSize;
    vec2 texelDy = dFdy(uv) * uImageSize;
    float majorFootprint = max(length(texelDx), length(texelDy));
    // A ribbon fragment is EXEMPT from the primary-depth shadow test. Its uv is
    // pinned to the rim texel, so the stored depth it would read is the rim's —
    // and the skirt deliberately recedes behind the rim, so the comparison says
    // "occluded" for every fragment and deletes the whole skirt.
    if (uOccludePrimary > 0.5 && uHasPrimaryDepth > 0.5 && !isRibbon) {
      float storedZ = atlasUnpackMetricDepth(uv);
      // Only trust a shadow comparison beside a REAL discontinuity in the
      // primary depth map.  A separately retopologized/exported mesh may have
      // been regenerated with another depth model; treating that broad model
      // disagreement as occlusion erased entire front-facing facades.  The
      // multi-scale neighbour jump turns this into the intended edge operation.
      // Zero is the packer's invalid/no-depth sentinel and never occludes.
      if (storedZ > 0.0005) {
        vec2 depthTexel = 1.0 / max(uPrimaryDepthSize, vec2(1.0));
        vec2 depthDx = dFdx(uv) * uPrimaryDepthSize;
        vec2 depthDy = dFdy(uv) * uPrimaryDepthSize;
        float depthProbeRadius = clamp(
          max(length(depthDx), length(depthDy)), 1.0, 4.0);
        vec2 wideDepthTexel = depthTexel * depthProbeRadius;
        float zL = atlasUnpackMetricDepth(uv - vec2(depthTexel.x, 0.0));
        float zR = atlasUnpackMetricDepth(uv + vec2(depthTexel.x, 0.0));
        float zU = atlasUnpackMetricDepth(uv - vec2(0.0, depthTexel.y));
        float zD = atlasUnpackMetricDepth(uv + vec2(0.0, depthTexel.y));
        float zWL = atlasUnpackMetricDepth(uv - vec2(wideDepthTexel.x, 0.0));
        float zWR = atlasUnpackMetricDepth(uv + vec2(wideDepthTexel.x, 0.0));
        float zWU = atlasUnpackMetricDepth(uv - vec2(0.0, wideDepthTexel.y));
        float zWD = atlasUnpackMetricDepth(uv + vec2(0.0, wideDepthTexel.y));
        float maxDepthJump = 0.0;
        maxDepthJump = max(maxDepthJump, atlasRelativeDepthJump(storedZ, zL));
        maxDepthJump = max(maxDepthJump, atlasRelativeDepthJump(storedZ, zR));
        maxDepthJump = max(maxDepthJump, atlasRelativeDepthJump(storedZ, zU));
        maxDepthJump = max(maxDepthJump, atlasRelativeDepthJump(storedZ, zD));
        maxDepthJump = max(maxDepthJump, atlasRelativeDepthJump(storedZ, zWL));
        maxDepthJump = max(maxDepthJump, atlasRelativeDepthJump(storedZ, zWR));
        maxDepthJump = max(maxDepthJump, atlasRelativeDepthJump(storedZ, zWU));
        maxDepthJump = max(maxDepthJump, atlasRelativeDepthJump(storedZ, zWD));
        float relativeDepthJump = maxDepthJump;
        depthEdge = smoothstep(0.015, 0.08, relativeDepthJump);

        // A curtain triangle straddles the depth step. On the foreground side
        // its interpolated depth lies BEHIND the stored surface; on the
        // background side it lies IN FRONT. A one-sided "behind" test therefore
        // always leaves half of the rubber sheet. Compare the absolute relative
        // mismatch, still gated to real multi-scale depth discontinuities, and
        // resolve it through derivative-filtered linear coverage.
        float relativeDepthMismatch = abs(-vCamZ - storedZ) / max(storedZ, 0.001);
        float compareFeather = max(
          uOccludeFeather, 1.5 * fwidth(relativeDepthMismatch));
        float depthMismatch = smoothstep(
          uOccludeBias, uOccludeBias + compareFeather, relativeDepthMismatch);
        coverage *= 1.0 - depthEdge * depthMismatch;
      }
    }
    // Per-pixel edge matte (ProjectionSource.mask_b64) — the classic DMP move:
    // geometry silhouettes tear at grid-quad resolution (blocky staircases),
    // so the full-resolution matte cuts the TRUE edge instead. Sampled at the
    // same projected pixel as the photo itself, so it needs no separate UVs.
    if (uHasMatte > 0.5) {
      float matte = texture2D(uMatte, uv).r;
      // Mattes are linear DATA. Filter only their coverage; never pass them
      // through the RGB display transform below. fwidth gives a roughly
      // one-pixel transition at any viewport scale.
      //
      // Feathered UNCONDITIONALLY. This used to sit inside uOccludePrimary and
      // fall back to a hard discard at 0.5 with the toggle off, which put a
      // binary cut back on the exact edge the matte exists to soften — and the
      // toggle is off by default. Coverage from a silhouette matte says nothing
      // about depth occlusion; the two were never related.
      if (uMatteSoft > 0.5) {
        // SOFT LAYERING: the matte is a CONTINUOUS visibility field
        // A = exp(-beta*|grad disparity|^2), not a cut-here mask. Multiply it
        // in directly — thresholding at 0.5 would re-binarize the exact
        // gradient it exists to carry and put the hard edge straight back.
        coverage *= clamp(matte, 0.0, 1.0);
      } else {
        float matteFeather = clamp(0.5 * fwidth(matte), 0.04, 0.25);
        coverage *= smoothstep(0.5 - matteFeather, 0.5 + matteFeather, matte);
      }
    }
    // Transition ribbon: a bounded edge-extension skirt hanging off a torn
    // silhouette. Its UV is FROZEN at the rim texel, so it has no texture
    // derivative for the footprint term below to see and no depth gradient for
    // a matte to see — the fade has to come from the per-vertex parameter the
    // geometry was built with. Unconditional, like the matte: a skirt whose
    // fade depends on a viewport toggle is just a visible slab with the toggle
    // off. RIBBON_FADE_START mirrors core/transition_ribbon.py and is the same
    // curve baked into the exported vertex alpha, so viewport and DCC agree.
    if (vAtlasRibbonT > 0.0) {
      coverage *= 1.0 - smoothstep(${RIBBON_FADE_START.toFixed(4)}, 1.0,
                                   clamp(vAtlasRibbonT, 0.0, 1.0));
      // Kill the faded tail EARLY. The material writes depth, so a ribbon
      // fragment at 0.5% alpha is invisible yet still lays down an occluding
      // depth value that hides the inpainted band the skirt exists to reveal.
      if (coverage < 0.01) discard;
    }
    // Soft layering keeps the mesh UNTORN, so a rubber-band triangle spanning a
    // depth cliff survives to be shaded. Its texel footprint explodes (one
    // screen pixel covers many source texels), which is an independent detector
    // of exactly those fragments — and it has to act ALONE here. The stretch
    // term further down multiplies by edgeRisk, so a camera-facing smear in a
    // smooth depth region fades by nothing at all.
    if (uSoftStretch > 0.0) {
      coverage *= 1.0 - uSoftStretch * smoothstep(
        uStretchStart, uStretchEnd, majorFootprint);
    }
    vec3 toCam = normalize(uCamPos - vWorldPos);
    float facing = abs(dot(normalize(vWorldNormal), toCam));
    if (uOccludePrimary > 0.5) {
      // The backend marks the kept side of every deliberately torn quad and
      // supplies two diminishing inward rings. Interpolation turns that exact
      // topology boundary into a several-pixel alpha roll-off even without a
      // primary depth texture (the reference workflow case). This erodes into
      // existing geometry only; it never invents pixels outside the mesh.
      // Ribbon fragments carry their own fade (vAtlasRibbonT) and must not also
      // take the topology feather: welding ring 0 onto the rim means they
      // interpolate the rim's edge_risk of 1.0, which would erode the skirt
      // inward from the very seam it exists to cover.
      float topologyRisk = isRibbon ? 0.0 : clamp(vAtlasEdgeRisk, 0.0, 1.0);
      // Coverage dilation: at/near Camera View the source texel footprint is
      // compact, so keep the existing STRAIGHT RGB opaque farther toward the
      // true mesh edge before beginning the inward feather. This cancels the
      // apparent boundary growth caused by a symmetric fade. As orbiting makes
      // the footprint large, relax the dilation so the wider tear-suppression
      // feather returns. A fragment shader cannot paint outside missing
      // triangles; this deliberately expands coverage only over fragments that
      // already exist and therefore never samples/bleeds neighbouring RGB.
      float topologyStretch = smoothstep(2.0, 8.0, majorFootprint);
      float topologyDilate = mix(0.38, 0.08, topologyStretch);
      float topologyCoverage = 1.0 - smoothstep(
        topologyDilate, 1.0, topologyRisk);
      coverage *= topologyCoverage;

      float facingFeather = clamp(1.5 * fwidth(facing), 0.01, 0.12);
      coverage *= smoothstep(uFacingThreshold - facingFeather,
                             uFacingThreshold + facingFeather, facing);

      // Fade only a LARGE source-texel footprint that is also grazing or sits
      // on a true depth edge.  The old major/minor anisotropy ratio classified
      // ordinary perspective foreshortening on broad building faces as a tear,
      // producing the large triangular holes seen in the regression captures.
      float footprintFeather = clamp(1.5 * fwidth(majorFootprint), 0.5, 4.0);
      float footprintRisk = smoothstep(uStretchStart - footprintFeather,
                                       uStretchEnd + footprintFeather,
                                       majorFootprint);
      float grazingRisk = 1.0 - smoothstep(0.06, 0.30, facing);
      float edgeRisk = max(depthEdge, grazingRisk);
      coverage *= 1.0 - footprintRisk * edgeRisk;

      // A projected plate edge is coverage too. Resolve it over about one
      // screen pixel instead of exposing a hard staircase at the UV border.
      float frameEdge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
      float frameFeather = max(1.5 * fwidth(frameEdge), 1.0 / max(uImageSize.x, uImageSize.y));
      coverage *= smoothstep(0.0, frameFeather, frameEdge);

    } else if (facing < uFacingThreshold) {
      discard;                                    // too grazing for this projector
    }
    vec4 col = texture2D(uTexture, uv);
    // Smudge the skirt ALONG the silhouette, widening with distance from the
    // rim. Each column is frozen to a single texel, so without this the skirt
    // is a fan of hard radial streaks — one flat colour per column, banding
    // against its neighbours. Averaging across neighbouring columns turns that
    // into a smooth outward bleed of the subject's own edge colour.
    //
    // The direction is free: vAtlasBakedUv is CONSTANT within a column and
    // changes only between columns, so its screen-space gradient already points
    // along the rim. Derivatives are taken in uniform control flow above; taking
    // them inside the branch would be undefined.
    if (isRibbon && uRibbonSmudge > 0.0) {
      vec2 along = vec2(bakedGx.x + bakedGy.x, -(bakedGx.y + bakedGy.y));
      vec2 alongTexels = along * uImageSize;
      float alongLen = length(alongTexels);
      if (alongLen > 1e-6) {
        vec2 stepUv = (alongTexels / alongLen)
                    * (uRibbonSmudge * clamp(vAtlasRibbonT, 0.0, 1.0)) / uImageSize;
        col = 0.2 * (texture2D(uTexture, uv - 2.0 * stepUv)
                   + texture2D(uTexture, uv - stepUv)
                   + col
                   + texture2D(uTexture, uv + stepUv)
                   + texture2D(uTexture, uv + 2.0 * stepUv));
      }
    }
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
    // OCIO/associated-alpha rule: colour transforms operate on STRAIGHT RGB;
    // alpha/coverage is linear data and is not transformed. If RGB ever arrives
    // premultiplied it must be unpremultiplied before this line. The current
    // projection textures are straight; Three.js uses straight-alpha blending
    // below (premultipliedAlpha:false) and multiplies RGB exactly once at the
    // blend boundary. Multiplying RGB by coverage here would double-premultiply.
    vec3 outColor = atlasLinearToSRGB(clamp(col.rgb * relight, 0.0, 1.0));
    // 🎨 layer-debug overlay: tint EVERYTHING this projection source paints
    // with its own identifying color (base/primary + each ProjectionSource
    // get distinct palette entries at material build; legend in the toolbar).
    // Strong mix so layer coverage reads at a glance; display-space like 🩻.
    if (uLayerDebug > 0.5) {
      outColor = mix(outColor, uLayerTint, 0.65);
      // created_islands is source-image-space, so reproject this fragment
      // through the original solved camera. The identity therefore survives
      // merging and retopology instead of depending on transient face IDs.
      if (uHasPatchMask > 0.5) {
        vec4 pcam = uPatchViewMatrix * vec4(vWorldPos, 1.0);
        float pdepth = -pcam.z;
        if (pdepth > 1e-5) {
          vec2 ppx = vec2(uPatchCx + uPatchFx * pcam.x / pdepth,
                          uPatchCy - uPatchFy * pcam.y / pdepth);
          vec2 puv = ppx / uPatchImageSize;
          if (puv.x >= 0.0 && puv.x <= 1.0 && puv.y >= 0.0 && puv.y <= 1.0
              && texture2D(uPatchMask, puv).r > 0.5) {
            outColor = mix(outColor, uPatchTint, 0.90);
          }
        }
      }
    }
    // 🎭 debug-matte isolate (display-space, like the overlays above): project
    // this fragment through the PRIMARY/recovered camera (same math as the
    // vertex shader's vImagePx, evaluated per-fragment against uDbg*) and dim
    // or cull everything outside the wired matte. Out-of-frame / behind-camera
    // fragments count as outside — they can't be inside a source-image matte.
    if (uDebugMatteOn > 0.5 && uHasDebugMatte > 0.5) {
      float dbgIn = 0.0;
      vec4 dcam = uDbgViewMatrix * vec4(vWorldPos, 1.0);
      float ddepth = -dcam.z;
      if (ddepth > 1e-5) {
        vec2 dpx = vec2(uDbgCx + uDbgFx * dcam.x / ddepth,
                        uDbgCy - uDbgFy * dcam.y / ddepth);
        vec2 duv = dpx / uDbgImageSize;
        if (duv.x >= 0.0 && duv.x <= 1.0 && duv.y >= 0.0 && duv.y <= 1.0) {
          dbgIn = texture2D(uDebugMatte, duv).r;
        }
      }
      if (dbgIn < 0.5) {
        if (uDebugMatteDim <= 0.001) discard;
        outColor *= uDebugMatteDim;
      }
    }
    // WebGL's sRGB texture decode and atlasLinearToSRGB affect RGB only. Combine
    // source alpha with the independently filtered linear coverage, then hand
    // straight RGB + straight alpha to the fixed-function blend stage. A tiny
    // fully-hidden cutoff avoids writing depth for numerically-zero fragments.
    float finalAlpha = clamp(col.a * uOpacity * coverage, 0.0, 1.0);
    if (finalAlpha <= (1.0 / 255.0)) discard;
    gl_FragColor = vec4(outColor, finalAlpha);
  }
`;

// ---------------------------------------------------------------------------
// Patch-priority ordering (ProjectionSource.priority — "higher wins; the
// primary is implicitly highest", core/schema.py). Real z-buffering already
// resolves most overlap between the primary and patch geometry (or between
// two patches) by actual depth; these two mechanisms only disambiguate the
// band where depth is coincident or near-coincident (independently-derived
// meshes rarely align exactly):
//   - renderOrder makes EXACT depth ties deterministic (Three sorts
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
const PLANAR_PATCH_DEBUG = 0xff2fd6;                 // magenta — generated hole islands
const LAYER_DEBUG_PALETTE = [
  0xff6a3d, // orange — typically the fg layer
  0x3d8bff, // blue   — typically the background layer
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
      uMatteSoft: { value: options.matteSoft ? 1.0 : 0.0 },
      uSoftStretch: { value: options.matteSoft ? 1.0 : 0.0 },
      // Per-mesh: the value the relief-mesh node recorded on the primitive, so
      // the widget, the viewport and the GLB bake all read ONE number. Falls
      // back to the constant for geometry built before it existed.
      uRibbonSmudge: {
        value: (options.ribbonSmudgePx === undefined || options.ribbonSmudgePx === null)
          ? RIBBON_SMUDGE_TEXELS : Number(options.ribbonSmudgePx),
      },
      uPrimaryDepth: { value: options.primaryDepthTexture || null },
      uHasPrimaryDepth: { value: options.primaryDepthTexture ? 1.0 : 0.0 },
      uPrimaryDepthSize: { value: new THREE.Vector2(
        data.primary_depth_width || data.image_width || 1,
        data.primary_depth_height || data.image_height || 1,
      ) },
      uOccludePrimary: { value: 0 },
      // Relative-depth units: reject only after 1.5% behind the stored surface,
      // feathering over another 2%.  The depth-edge gate in the shader is the
      // primary safety boundary; these values tune its soft alpha roll-off.
      uOccludeBias: { value: 0.015 },
      uOccludeFeather: { value: 0.02 },
      uStretchStart: { value: 4.0 },
      uStretchEnd: { value: 16.0 },
      // 🎭 debug-matte isolate — all values pushed per-frame by
      // syncProjectionLightUniforms (materials are rebuilt every execution,
      // so build-time defaults here are just inert placeholders).
      uDebugMatte: { value: null },
      uHasDebugMatte: { value: 0 },
      uDebugMatteOn: { value: 0 },
      uDebugMatteDim: { value: 0.15 },
      uDbgViewMatrix: { value: new THREE.Matrix4() },
      uDbgFx: { value: 1 },
      uDbgFy: { value: 1 },
      uDbgCx: { value: 0.5 },
      uDbgCy: { value: 0.5 },
      uDbgImageSize: { value: new THREE.Vector2(1, 1) },
      // 🎨 layer-debug identity color (fixed per source at build; toggle is
      // uLayerDebug, live-synced like the light uniforms).
      uLayerDebug: { value: 0 },
      uLayerTint: { value: options.layerTint || new THREE.Color(LAYER_DEBUG_PRIMARY) },
      // ◩ Planar-hole-patch identity, live-populated from the viewport's
      // source-space patch_mask input. Projector coordinates make this safe
      // across a downstream AtlasRetopologizeLayer.
      uPatchMask: { value: null },
      uHasPatchMask: { value: 0 },
      uPatchTint: { value: new THREE.Color(PLANAR_PATCH_DEBUG) },
      uPatchViewMatrix: { value: new THREE.Matrix4() },
      uPatchFx: { value: 1 },
      uPatchFy: { value: 1 },
      uPatchCx: { value: 0.5 },
      uPatchCy: { value: 0.5 },
      uPatchImageSize: { value: new THREE.Vector2(1, 1) },
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
    transparent: true,
    premultipliedAlpha: false,
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

// Every projection geometry supplies the custom shader attribute. Relief
// meshes receive Python's compact per-vertex torn-boundary field; ordinary
// proxy primitives receive zeros and remain completely unaffected.
function attachAtlasEdgeRisk(geo, entry) {
  const count = geo?.attributes?.position?.count || 0;
  const source = entry?.edge_risk;
  const values = Array.isArray(source) && source.length === count
    ? new Float32Array(source)
    : new Float32Array(count);
  geo.setAttribute("atlasEdgeRisk", new THREE.BufferAttribute(values, 1));
  return attachAtlasRibbonT(geo, entry);
}

// Per-vertex transition-ribbon parameter, uploaded on the SAME path as
// edge_risk. A zero-filled fallback is required, not merely tidy: the vertex
// shader declares the attribute unconditionally, and a missing buffer leaves it
// undefined, which would fade arbitrary fragments of an ordinary mesh.
function attachAtlasRibbonT(geo, entry) {
  const count = geo?.attributes?.position?.count || 0;
  const source = entry?.ribbon_t;
  const values = Array.isArray(source) && source.length === count
    ? new Float32Array(source)
    : new Float32Array(count);
  geo.setAttribute("atlasRibbonT", new THREE.BufferAttribute(values, 1));
  return geo;
}

// Build meshes for the Python-derived projection proxies (ground/walls/boxes/
// cylinders/backdrop). Transforms arrive as row-major 16-float arrays — the
// same convention THREE.Matrix4.set() takes.
// metadata.source values written by the viewport's own draw tools — kept in
// step with nodes_viewport.DRAWN_KINDS' emitted sources.
const DRAWN_PROXY_SOURCES = new Set([
  "viewport_polygon", "viewport_box", "viewport_sphere",
]);

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
    attachAtlasEdgeRisk(geo, e);
    const mat = new THREE.MeshStandardMaterial({
      color: 0x9a9a9a, roughness: 0.85, side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.matrixAutoUpdate = false;
    mesh.matrix.set(...e.transform);
    mesh.userData.atlasDerived = true;
    // Hand-drawn surfaces stand where the camera never saw, so they get the
    // SMEARED plate rather than the raw one (see drawn_plate_b64 below).
    mesh.userData.atlasDrawn = DRAWN_PROXY_SOURCES.has(e.metadata?.source);
    // solve_b geometry from AtlasMergeGeometry is the clean-background layer
    // of a layered solve — with a clean_plate connected it projects THAT.
    mesh.userData.atlasCleanSource = e.metadata?.merged_from === "solve_b";
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
  let geo;
  if (e.type === "mesh") {
    if (!e.vertices?.length || !e.faces?.length) return null;
    geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(e.vertices), 3));
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
  return attachAtlasEdgeRisk(geo, e);
}

function projectionEvidenceLabel(evidenceType) {
  return evidenceType === "photographed" ? "PHOTO"
    : evidenceType === "generated" ? "GENERATED" : "SOURCE";
}

function projectionGeometryEntries(src, data) {
  return src.proxy_geometry?.length ? src.proxy_geometry
    : src.evidence_type === "photographed" ? (data.proxy_geometry || []) : [];
}

// Build the multi-angle patch sources (AtlasAddPatchView). Each source is its
// own camera + AI novel-view image + geometry; each mesh carries its OWN
// projection material (bound to that source's camera+image, with a facing-ratio
// mask) in userData._projMaterial, so applyProjection layers it over the
// primary. Patch geometry is Python-owned (regenerated each execution), so —
// like the derived group — Clear leaves it alone.
// --- Dynamic-plate frame streams (AtlasLoadDynamicPlate) ------------------
// A source whose payload carries `dynamic_plate: {key, frame_count, fps}`
// plays its generated sequence in the viewport: frames stream lazily from
// /atlas/dynamic_plate/{key}/{index} and swap into the projection material's
// uTexture on the render ticker. Rebuilt alongside the materials on every
// execution (buildPatchSources clears the list), so no state survives a
// graph re-run — same lifecycle rule as every other projection uniform.
const _dynamicPlateStreams = [];

function syncDynamicPlateFrames(now) {
  if (!_dynamicPlateStreams.length) return;
  for (const stream of _dynamicPlateStreams) {
    const index = Math.floor((now / 1000) * stream.fps) % stream.frameCount;
    if (index === stream.lastIndex) continue;
    let tex = stream.textures.get(index);
    if (tex === undefined) {
      stream.textures.set(index, null);  // in flight — never refetch
      new THREE.TextureLoader().load(
        `/atlas/dynamic_plate/${stream.key}/${index}`,
        (loaded) => {
          loaded.flipY = false;                // shader UV origin is top-left
          loaded.colorSpace = THREE.SRGBColorSpace;
          stream.textures.set(index, loaded);
        },
        undefined,
        () => stream.textures.delete(index));  // transient failure: retry later
      continue;                                // hold current frame meanwhile
    }
    if (tex === null) continue;                // still loading
    stream.lastIndex = index;
    for (const mat of stream.materials) {
      if (mat.uniforms?.uTexture) mat.uniforms.uTexture.value = tex;
    }
  }
}

function buildPatchSources(scene, data, onSourceReady) {
  // materials are about to be rebuilt — drop every stream with them (their
  // cached textures belong to the outgoing material generation)
  for (const stream of _dynamicPlateStreams) {
    for (const tex of stream.textures.values()) tex?.dispose?.();
  }
  _dynamicPlateStreams.length = 0;
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
    for (const e of projectionGeometryEntries(src, data)) {
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
      const build = (matteTexture, matteSoft) => {
        const patchMat = makeProjectionMaterial(src, tex,
          { facingThreshold, priority: src.priority, matteTexture, matteSoft,
            ribbonSmudgePx: (src.proxy_geometry || []).find(
              (g) => g.type === "mesh"
                && g.metadata?.ribbon_smudge_px !== undefined
            )?.metadata?.ribbon_smudge_px,
            layerTint: new THREE.Color(
              LAYER_DEBUG_PALETTE[idx % LAYER_DEBUG_PALETTE.length]) });
        for (const m of meshes) {
          const prev = m.userData._projMaterial;
          if (prev && prev !== patchMat) {
            prev.uniforms?.uTexture?.value?.dispose?.();
            prev.uniforms?.uMatte?.value?.dispose?.();
            prev.dispose?.();
          }
          m.userData._projMaterial = patchMat;
        }
        const dyn = src.dynamic_plate;
        if (dyn?.key && dyn.frame_count > 0) {
          _dynamicPlateStreams.push({
            key: dyn.key,
            frameCount: dyn.frame_count,
            fps: dyn.fps > 0 ? dyn.fps : 24,
            materials: [patchMat],
            textures: new Map(),
            lastIndex: -1,
          });
        }
        if (typeof onSourceReady === "function") onSourceReady();
      };
      // Per-pixel edge matte: geometry stays coarse; the matte cuts the true
      // silhouette in the shader. `mask_b64` is the source's own hand-authored
      // or unseen-areas matte and wins; failing that, a band layer built with
      // `silhouette_matte` carries one on its relief primitive, which is how the
      // silhouette work reaches the layer stack at all.
      // loadMatteFromB64 always calls back (null on missing/failed matte).
      const bandEntry = (src.proxy_geometry || []).find(
        (e) => e.type === "mesh" && e.silhouette_matte_b64);
      const bandMatte = bandEntry?.silhouette_matte_b64 || "";
      // A hand-authored mask_b64 is a CUT; only the band's own field can
      // be soft, so the mode follows whichever matte actually won.
      const bandSoft = !src.mask_b64
        && bandEntry?.silhouette_matte_mode === "soft";
      loadMatteFromB64(src.mask_b64 || bandMatte,
        (matteTexture) => build(matteTexture, bandSoft));
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
  // `samples` matters here and was never set: the visible canvas is created with
  // antialias:true, but a WebGLRenderTarget defaults to samples:0, so every
  // OFFSCREEN render — previews, baked path frames, the output desk — came back
  // strictly aliased while the viewport beside it looked smooth. Silhouettes are
  // exactly where that shows. Ignored on WebGL1.
  const renderTarget = options.renderTarget
    || new THREE.WebGLRenderTarget(width, height, { samples: 4 });
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
async function renderAllPasses(
  renderer, scene, camera, width, height, exclude = [], patchSelection = null
) {
  if (!THREE) return null;

  // The passes must contain geometry only: hide the background photo plane and
  // viewport helpers (grid) for every pass, restore after.
  const hidden = [];
  const hideList = [...exclude];
  scene.traverse((c) => { if (c.userData?.atlasHelper) hideList.push(c); });
  for (const obj of hideList) {
    if (obj && obj.visible) { obj.visible = false; hidden.push(obj); }
  }

  const rt = new THREE.WebGLRenderTarget(width, height, { samples: 4 });

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

    // Select generated geometry in original-camera image space, but rasterise
    // it from the CURRENT orbit camera. This is the reprojected inpaint mask.
    const patch = patchSelection || {};
    const patchCam = patch.camera || {};
    const patchMat = new THREE.ShaderMaterial({
      uniforms: {
        uPatchMask: { value: patch.texture || null },
        uHasPatchMask: { value: patch.texture ? 1 : 0 },
        uPatchViewMatrix: { value: patchCam.vm || new THREE.Matrix4() },
        uPatchFx: { value: patchCam.fx || 1 },
        uPatchFy: { value: patchCam.fy || patchCam.fx || 1 },
        uPatchCx: { value: patchCam.cx ?? 0.5 },
        uPatchCy: { value: patchCam.cy ?? 0.5 },
        uPatchImageSize: {
          value: new THREE.Vector2(patchCam.w || 1, patchCam.h || 1),
        },
      },
      vertexShader: `
        varying vec3 vPatchWorldPos;
        void main() {
          vec4 world = modelMatrix * vec4(position, 1.0);
          vPatchWorldPos = world.xyz;
          gl_Position = projectionMatrix * viewMatrix * world;
        }`,
      fragmentShader: `
        uniform sampler2D uPatchMask;
        uniform float uHasPatchMask;
        uniform mat4 uPatchViewMatrix;
        uniform float uPatchFx;
        uniform float uPatchFy;
        uniform float uPatchCx;
        uniform float uPatchCy;
        uniform vec2 uPatchImageSize;
        varying vec3 vPatchWorldPos;
        void main() {
          if (uHasPatchMask < 0.5) discard;
          vec4 pcam = uPatchViewMatrix * vec4(vPatchWorldPos, 1.0);
          float depth = -pcam.z;
          if (depth <= 1e-5) discard;
          vec2 px = vec2(uPatchCx + uPatchFx * pcam.x / depth,
                         uPatchCy - uPatchFy * pcam.y / depth);
          vec2 uv = px / uPatchImageSize;
          if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) discard;
          if (texture2D(uPatchMask, uv).r < 0.5) discard;
          gl_FragColor = vec4(1.0);
        }`,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    scene.background = new THREE.Color(0x000000);
    const patchMaskB64 = renderToBase64(patchMat);
    scene.background = bg;
    patchMat.dispose();

    return {
      shaded: shadedB64,
      depth: depthB64,
      normal: normalB64,
      mask: maskB64,
      patch_render_mask: patchMaskB64,
    };
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
  // LOGICAL preview size stays what it was — reported to Python, used for
  // node layout. Only the drawing buffer is supersampled.
  let previewScale = atlasBackbufferScale(previewW, previewH);
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
  canvas.width = Math.round(previewW * previewScale);
  canvas.height = Math.round(previewH * previewScale);
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
  // left:66px clears the tool rail; top:46px sits under its status chip.
  metaHud.style.cssText = "position:absolute;top:46px;left:66px;padding:6px 8px;background:rgba(10,10,14,0.72);" +
    "color:#cde;font:10px/1.5 monospace;border-radius:4px;pointer-events:none;white-space:pre;display:none;";

  const localControlsLayer = document.createElement("div");
  localControlsLayer.style.cssText =
    "position:absolute;left:0;right:0;bottom:0;z-index:8;display:flex;flex-direction:column;align-items:stretch;gap:0;" +
    "background:rgba(16,16,22,0.88);pointer-events:auto;line-height:normal;";

  canvasWrap.append(canvas, diagramSvg, metaHud, localControlsLayer);

  // DCC-style vertical tool rail for the blockout draw tools (Draw / Box /
  // Sphere / Edit / Snap / Apply). Lives on canvasWrap and is NEVER reparented
  // by mountControls: these tools act on the canvas under the cursor, so they
  // stay with the viewport even when the rest of the toolbar moves to an
  // AtlasViewportControls node. Vertically centred so it clears metaHud
  // (top-left) and drawStatus / the layer legend (bottom-left).
  const drawRail = document.createElement("div");
  drawRail.style.cssText =
    "position:absolute;left:10px;top:12px;z-index:10;" +
    "display:flex;flex-direction:column;gap:4px;padding:8px 6px;" +
    "background:rgba(18,18,24,0.95);border:1px solid #2c2c34;border-radius:10px;" +
    "pointer-events:auto;line-height:normal;";
  canvasWrap.appendChild(drawRail);

  // Status chip beside the rail — the horizontal bar naming the active tool
  // and the snap state ("Box · Snap on"), mirroring the rail's own icons.
  const railStatus = document.createElement("div");
  railStatus.style.cssText =
    "position:absolute;left:66px;top:12px;z-index:10;display:flex;align-items:center;" +
    "gap:7px;padding:6px 12px;background:rgba(24,24,30,0.92);border:1px solid #2c2c34;" +
    "border-radius:9px;color:#ddd;font:12px/1 sans-serif;pointer-events:none;" +
    "line-height:normal;white-space:nowrap;";
  canvasWrap.appendChild(railStatus);

  // Three.js setup
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setSize(Math.round(previewW * previewScale),
                   Math.round(previewH * previewScale), false);
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
  let occludePrimaryOn = false;
  // 🎭 debug-matte isolate (node `debug_matte` input): ON by default — wiring
  // a matte means you want the isolate; the toolbar 🎭 button toggles it and
  // its slider sets the outside-matte dim (0 = hard cull). Session-only.
  let debugMatteOn = true;
  let debugMatteDim = 0.15;
  let debugMatteTex = null;   // loaded per execution from data.debug_matte_b64
  let debugMatteCam = null;   // {vm, fx, fy, cx, cy, w, h} — the PRIMARY camera
  let patchMaskTex = null;     // AtlasPlanarHolePatch created_islands
  let patchMaskCam = null;     // original solved camera for reprojection
  let occludeBias = 0.015;
  let layerDebugOn = false; // 🎨 per-layer identity tint toggle
  let bumpStrength = 0;     // 💡 Lights panel "Detail" — photo-luminance relight bump
  let bumpScale = 8;        // 💡 Lights panel "Scale" — bump sampling offset (texels)
  function syncProjectionLightUniforms() {
    const active = movableLights.some((l) => l.intensity > 0)
      || layerDebugOn || bumpStrength > 0 || occludePrimaryOn
      || (debugMatteOn && !!debugMatteTex) || (layerDebugOn && !!patchMaskTex);
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
      if (mat.uniforms.uOccludePrimary) {
        mat.uniforms.uOccludePrimary.value = occludePrimaryOn ? 1 : 0;
      }
      if (mat.uniforms.uOccludeBias) {
        mat.uniforms.uOccludeBias.value = occludeBias;
      }
      if (mat.uniforms.uLayerDebug) {
        mat.uniforms.uLayerDebug.value = layerDebugOn ? 1 : 0;
      }
      if (mat.uniforms.uPatchMask) {
        const hasPatch = !!patchMaskTex && !!patchMaskCam;
        mat.uniforms.uPatchMask.value = patchMaskTex;
        mat.uniforms.uHasPatchMask.value = hasPatch ? 1 : 0;
        if (hasPatch) {
          mat.uniforms.uPatchViewMatrix.value.copy(patchMaskCam.vm);
          mat.uniforms.uPatchFx.value = patchMaskCam.fx;
          mat.uniforms.uPatchFy.value = patchMaskCam.fy;
          mat.uniforms.uPatchCx.value = patchMaskCam.cx;
          mat.uniforms.uPatchCy.value = patchMaskCam.cy;
          mat.uniforms.uPatchImageSize.value.set(patchMaskCam.w, patchMaskCam.h);
        }
      }
      if (mat.uniforms.uBumpStrength) {
        mat.uniforms.uBumpStrength.value = bumpStrength;
      }
      if (mat.uniforms.uBumpScale) {
        mat.uniforms.uBumpScale.value = bumpScale;
      }
      // 🎭 debug-matte isolate: texture + toggle + dim + the PRIMARY camera
      // (materials are rebuilt every execution, so everything is pushed here
      // rather than at build — the same reason the light uniforms sync live).
      if (mat.uniforms.uDebugMatte) {
        const on = debugMatteOn && !!debugMatteTex && !!debugMatteCam;
        mat.uniforms.uDebugMatte.value = debugMatteTex;
        mat.uniforms.uHasDebugMatte.value = debugMatteTex ? 1 : 0;
        mat.uniforms.uDebugMatteOn.value = on ? 1 : 0;
        mat.uniforms.uDebugMatteDim.value = debugMatteDim;
        if (on) {
          mat.uniforms.uDbgViewMatrix.value.copy(debugMatteCam.vm);
          mat.uniforms.uDbgFx.value = debugMatteCam.fx;
          mat.uniforms.uDbgFy.value = debugMatteCam.fy;
          mat.uniforms.uDbgCx.value = debugMatteCam.cx;
          mat.uniforms.uDbgCy.value = debugMatteCam.cy;
          mat.uniforms.uDbgImageSize.value.set(debugMatteCam.w, debugMatteCam.h);
        }
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
    updatePivotGizmo();              // move the marker immediately (don't wait a frame)
  }

  // 🎯 Pivot gizmo — an always-on-top marker at the orbit target so the artist
  // can SEE the point the orbit rotates around (a small sphere + short RGB axis
  // lines). depthTest:false / high renderOrder so it reads through geometry.
  // Sized to the scene each frame; visible only while the 🎯 Pivot panel is
  // open. NOT tagged atlasDerived/atlasPatch, so it never enters the pivot /
  // light-placement / band-box / projection logic. Hidden during the
  // deterministic export/Safe-Zone passes (they stash + restore its visibility).
  let pivotGizmo = null;
  function ensurePivotGizmo() {
    if (pivotGizmo || !THREE) return;
    pivotGizmo = new THREE.Group();
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(1, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xffcc33, depthTest: false, transparent: true, opacity: 0.95 }));
    dot.renderOrder = 100003;
    pivotGizmo.add(dot);
    [[[3.5, 0, 0], 0xff5555], [[0, 3.5, 0], 0x55ff55], [[0, 0, 3.5], 0x5599ff]].forEach(([v, col]) => {
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-v[0], -v[1], -v[2]), new THREE.Vector3(v[0], v[1], v[2])]);
      const ln = new THREE.Line(g, new THREE.LineBasicMaterial({ color: col, depthTest: false, transparent: true, opacity: 0.95 }));
      ln.renderOrder = 100002;
      pivotGizmo.add(ln);
    });
    pivotGizmo.visible = false;
    scene.add(pivotGizmo);
  }
  function updatePivotGizmo() {      // called per-frame from animate()
    if (!pivotGizmo || !pivotGizmo.visible) return;
    const t = controls.target || (controls.getTarget && controls.getTarget());
    if (t) pivotGizmo.position.set(t.x, t.y, t.z);
    // Size by camera→pivot distance, NOT the scene bounds (which include the far
    // backdrop card and would balloon the marker) — so it stays a small,
    // ~constant on-screen size as you dolly/orbit at any AtlasScaleOverride.
    const dist = camera.position.distanceTo(pivotGizmo.position) || 10;
    pivotGizmo.scale.setScalar(Math.max(dist * 0.02, 0.02));
  }

  // Animation loop — assign to node._atlasRafId each frame so cancelAnimationFrame works.
  // The orbit controller updates the camera on input events; pathPlayback
  // (set by the Camera Path "Play" button, below) drives it during path preview.
  let pathPlayback = null; // { startTime, durationSec, onDone }
  let applyPathPoseAtT = null; // assigned once the Camera Path block below runs
  let bakeInProgress = false; // set by bakeProxyPathFrames; gates the nav shake
  // 🎬 live handheld shake for the tracking keys: subtract-prev/add-new so the
  // orbit controller's state is never touched and the camera lands back on its
  // exact base pose once the envelope decays. Never during path playback or a
  // bake — those own the camera, and baked pixels must stay deterministic.
  const navShakeOffset = new THREE.Vector3();
  function animate() {
    node._atlasRafId = requestAnimationFrame(animate);
    const now = performance.now();
    camera.position.sub(navShakeOffset); // restore the base pose from last frame
    navShakeOffset.set(0, 0, 0);
    controls.updateKeys();  // UE-style tracking keys (self-timed; no-op when idle)
    if (pathPlayback) {
      const t = Math.min(1, (now - pathPlayback.startTime) / 1000 / pathPlayback.durationSec);
      applyPathPoseAtT(t);
      if (t >= 1) {
        const done = pathPlayback.onDone;
        pathPlayback = null;
        camera.up.set(0, 1, 0); // undo any 🎬 shake roll before restoring
        done?.();
      }
    } else if (!bakeInProgress && pathShakeEnabled && pathShakeIntensity > 0) {
      const env = controls.getNavShakeEnv();
      if (env > 0.001) {
        // Wall-clock time base (fed in as frame@24fps) — frame-rate
        // independent; deterministic per timestamp, though nav shake is a
        // live-preview feel only and is never baked or exported.
        const off = atlasShakeOffsetsJS((now / 1000) * 24, 24, pathShakeIntensity * env, pathShakeSeed);
        const radius = controls.getFrame().radius || 1;
        navShakeOffset
          .copy(new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0).multiplyScalar(off[0] * radius))
          .addScaledVector(new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1), off[1] * radius)
          .addScaledVector(new THREE.Vector3().setFromMatrixColumn(camera.matrix, 2), -off[2] * radius);
        camera.position.add(navShakeOffset);
      }
    }
    syncProjectionLightUniforms();
    syncDynamicPlateFrames(now);  // animated dynamic-plate projections
    updatePivotGizmo();   // track the orbit target + rescale to the scene
    updateEditGizmo();    // pin the ✎ translate gizmo to the edit selection
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
    // The background photo plane is the grey (Project OFF) backdrop only. Under
    // 📽 Project it is HIDDEN, so any pixel the projection discards (matte
    // silhouette, torn quad, out-of-frame) reads as the black clear colour. (The
    // former 🕳 See-through hole-fill — keeping this plane visible under Project —
    // was removed as too buggy.) Also hidden during the deterministic export
    // passes (renderAllPasses / Safe Zone probe), handled separately.
    if (bgMesh) bgMesh.visible = !on;
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


  // ✂ Occlude ray-traced shadow map + filtered projection-edge cull
  const dbgOccludeBtn = document.createElement("button");
  dbgOccludeBtn.textContent = "✂ Occlude";
  dbgOccludeBtn.title = "Filter occluded, grazing and matte-edge projection texels using the primary depth map";
  dbgOccludeBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  dbgOccludeBtn.onclick = () => {
    occludePrimaryOn = !occludePrimaryOn;
    dbgOccludeBtn.style.background = occludePrimaryOn ? "#4a3a1a" : "#2a2a2a";
    dbgOccludeBtn.style.color = occludePrimaryOn ? "#fd4" : "#ddd";
  };
  toolbar.appendChild(dbgOccludeBtn);

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
    if (recoveredData?.patch_mask_b64) {
      rows.push([hex(PLANAR_PATCH_DEBUG), "generated planar hole islands"]);
    }
    (recoveredData?.projection_sources || []).forEach((s, i) => {
      rows.push([hex(LAYER_DEBUG_PALETTE[i % LAYER_DEBUG_PALETTE.length]),
                 `${s.name || `layer ${i}`} · ${projectionEvidenceLabel(s.evidence_type)}`]);
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

  // 🎭 Matte — debug-matte isolate (node `debug_matte` input, e.g. a layer's
  // SAM3 mask): under 📽 Project, everything whose PRIMARY-camera projection
  // falls outside the wired matte is dimmed to the slider level (0 = hard
  // cull), so one layer's region can be inspected/orbited in isolation.
  // Projection-mode only, live-synced like 🩻/🎨. ON by default when a matte
  // is wired; inert (and visually dimmed) when none is.
  const matteBtn = document.createElement("button");
  matteBtn.textContent = "🎭 Matte";
  matteBtn.title = "Isolate the region inside the wired debug_matte (dim/cull outside)";
  matteBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  const matteDimSlider = document.createElement("input");
  matteDimSlider.type = "range";
  matteDimSlider.min = "0"; matteDimSlider.max = "0.6"; matteDimSlider.step = "0.05";
  matteDimSlider.value = String(debugMatteDim);
  matteDimSlider.title = "Outside-matte brightness (0 = hard cull)";
  matteDimSlider.style.cssText = "width:70px;vertical-align:middle;display:none;";
  matteDimSlider.oninput = () => { debugMatteDim = parseFloat(matteDimSlider.value); };
  function refreshMatteBtn() {
    const has = !!debugMatteTex || !!(recoveredData && recoveredData.debug_matte_b64);
    matteBtn.style.opacity = has ? "1" : "0.45";
    matteBtn.style.background = (debugMatteOn && has) ? "#1a2a3a" : "#2a2a2a";
    matteBtn.style.color = (debugMatteOn && has) ? "#8cf" : "#ddd";
    matteDimSlider.style.display = (debugMatteOn && has) ? "inline-block" : "none";
  }
  matteBtn.onclick = () => {
    debugMatteOn = !debugMatteOn;
    refreshMatteBtn();
  };
  node._atlasRefreshMatteBtn = refreshMatteBtn;
  toolbar.appendChild(matteBtn);
  toolbar.appendChild(matteDimSlider);

  // ---------------------------------------------------------------------------
  // 📐 Extract Angle — orbit to the view you want a patch generated at
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
  angleBtn.title = "Orbit to the view you want a patch at, then click: measures the orbit " +
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
  // ✏️ Draw — author N-gons directly in 3D, then Apply.
  //
  // The reason this lives in the VIEWPORT and not on the flat plate: an
  // occluded hole only exists once the model is turned. Drawing on the plate
  // cannot see it, and depth inside such an outline belongs to the occluder,
  // not to the surface the artist means.
  //
  // A click is a ray. The clicks that HIT geometry establish the plane; every
  // click after that is a ray x plane intersection, so the outline can be
  // drawn straight across a hole that has nothing to hit.
  //
  // atlasEstablishPlaneFromHits / atlasIntersectRayWithPlane below MIRROR
  // core/polygon_planes.py by hand (the repo's accepted-duplication pattern,
  // same as the Catmull-Rom path math). They are deliberately plain-array
  // maths with no THREE dependency so tests/test_frontend_mirrors.py can
  // execute them under node and pin them numerically against Python.
  // ---------------------------------------------------------------------------
  function atlasEstablishPlaneFromHits(hits, up) {
    const U = up || [0, 1, 0];
    const n = hits.length;
    if (n < 2) return null;
    const ulen = Math.hypot(U[0], U[1], U[2]) || 1;
    const u = [U[0] / ulen, U[1] / ulen, U[2] / ulen];

    if (n >= 3) {
      // Newell's method — exact for 3 points and for any coplanar set, and
      // it collapses to zero on collinear input.
      let nx = 0, ny = 0, nz = 0;
      for (let i = 0; i < n; i += 1) {
        const a = hits[i], b = hits[(i + 1) % n];
        nx += (a[1] - b[1]) * (a[2] + b[2]);
        ny += (a[2] - b[2]) * (a[0] + b[0]);
        nz += (a[0] - b[0]) * (a[1] + b[1]);
      }
      let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (const p of hits) {
        for (let k = 0; k < 3; k += 1) {
          if (p[k] < lo[k]) lo[k] = p[k];
          if (p[k] > hi[k]) hi[k] = p[k];
        }
      }
      const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
      const mag = Math.hypot(nx, ny, nz);
      if (mag > 1e-8 * Math.max(span * span, 1e-12)) {
        const nrm = [nx / mag, ny / mag, nz / mag];
        const c = [0, 0, 0];
        for (const p of hits) { c[0] += p[0] / n; c[1] += p[1] / n; c[2] += p[2] / n; }
        return { normal: nrm, offset: nrm[0] * c[0] + nrm[1] * c[1] + nrm[2] * c[2] };
      }
    }

    // Two hits (or collinear ones): raise a GRAVITY-ALIGNED plane through
    // them. Yaw is the only free choice left and vertical is the facade
    // answer — the case where a hole runs off the frame edge and only two of
    // its edges are visible to click.
    const a = hits[0], b = hits[n - 1];
    const span = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const nx = u[1] * span[2] - u[2] * span[1];
    const ny = u[2] * span[0] - u[0] * span[2];
    const nz = u[0] * span[1] - u[1] * span[0];
    const mag = Math.hypot(nx, ny, nz);
    // The hits differ only in height: every vertical plane contains them.
    if (mag < 1e-9) return null;
    const nrm = [nx / mag, ny / mag, nz / mag];
    return { normal: nrm, offset: nrm[0] * a[0] + nrm[1] * a[1] + nrm[2] * a[2] };
  }

  // The canvas is object-fit:contain (see its cssText above), so the drawing
  // buffer is LETTERBOXED inside the element box: the rendered image does not
  // fill the client rect, and naive rect-relative NDC puts every picking ray
  // off by the bar size. Found live — clicks did not land where they were made,
  // and on a near-edge-on plane the small angular error became a huge world
  // displacement. Pure math, pinned by tests/test_frontend_mirrors.py.
  function atlasContainNdc(clientX, clientY, rect, bufW, bufH) {
    const scale = Math.min(rect.width / bufW, rect.height / bufH);
    const dispW = bufW * scale, dispH = bufH * scale;
    const originX = rect.left + (rect.width - dispW) / 2;
    const originY = rect.top + (rect.height - dispH) / 2;
    const x = ((clientX - originX) / dispW) * 2 - 1;
    const y = -(((clientY - originY) / dispH) * 2 - 1);
    return { x, y, inside: x >= -1 && x <= 1 && y >= -1 && y <= 1 };
  }

  function atlasIntersectRayWithPlane(origin, direction, plane) {
    const dlen = Math.hypot(direction[0], direction[1], direction[2]) || 1;
    const d = [direction[0] / dlen, direction[1] / dlen, direction[2] / dlen];
    const n = plane.normal;
    const denom = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
    if (Math.abs(denom) < 1e-9) return null;
    const t = (plane.offset
      - (n[0] * origin[0] + n[1] * origin[1] + n[2] * origin[2])) / denom;
    if (!(t > 1e-6) || !Number.isFinite(t)) return null;
    return [origin[0] + t * d[0], origin[1] + t * d[1], origin[2] + t * d[2]];
  }

  // -- draw-mode state --------------------------------------------------------
  let drawOn = false;
  let drawHits = [];        // world points that hit geometry (establish the plane)
  let drawPoints = [];      // the outline being drawn, world space
  let drawRays = [];        // the click ray behind each point, for re-projection
  let editOn = false;       // ✎ Edit: move points of ALREADY-drawn outlines
  let editDrag = null;      // { poly, index } while a handle is being dragged
  let editSel = null;       // { poly, indices } — persistent selection after a
                            // grab, so the translate gizmo has something to sit on
  let editWeld = null;      // { poly, index } — the ADJOINING vertex the dragged
                            // one is about to weld onto (both draw red)
  let drawDirty = false;    // outlines changed since the last ✅ Apply
  let editSnap = true;      // snap clicks/drags to mesh edges (Shift bypasses)
  let drawPlane = null;     // { normal, offset }
  let drawnPolygons = [];   // committed outlines, awaiting Apply
  let drawTilt = 0;         // radians, applied about the first-two-hits axis
  let drawPush = 0;         // metres along the plane normal
  const drawRaycaster = new THREE.Raycaster();
  const drawGroup = new THREE.Group();
  drawGroup.name = "atlas_draw_overlay";
  drawGroup.userData.atlasHelper = true;   // excluded from render passes
  scene.add(drawGroup);

  function drawTargets() {
    const targets = [];
    scene.traverse((o) => {
      if (!o.isMesh) return;
      if (o.userData?.atlasDerived || o.userData?.atlasPatch) targets.push(o);
    });
    return targets;
  }

  function drawPlaneAdjusted() {
    if (!drawPlane) return null;
    if (!drawTilt && !drawPush) return drawPlane;
    let normal = drawPlane.normal.slice();
    let offset = drawPlane.offset;
    if (drawTilt && drawHits.length >= 2) {
      // Rotate the normal about the axis through the first two hits, so the
      // plane pivots on the edge the artist actually clicked.
      const a = drawHits[0], b = drawHits[drawHits.length - 1];
      const ax = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const alen = Math.hypot(ax[0], ax[1], ax[2]) || 1;
      const k = [ax[0] / alen, ax[1] / alen, ax[2] / alen];
      const c = Math.cos(drawTilt), s = Math.sin(drawTilt);
      const kdotn = k[0] * normal[0] + k[1] * normal[1] + k[2] * normal[2];
      const kxn = [
        k[1] * normal[2] - k[2] * normal[1],
        k[2] * normal[0] - k[0] * normal[2],
        k[0] * normal[1] - k[1] * normal[0],
      ];
      normal = [
        normal[0] * c + kxn[0] * s + k[0] * kdotn * (1 - c),
        normal[1] * c + kxn[1] * s + k[1] * kdotn * (1 - c),
        normal[2] * c + kxn[2] * s + k[2] * kdotn * (1 - c),
      ];
      // Keep the plane on the clicked edge through the rotation.
      offset = normal[0] * a[0] + normal[1] * a[1] + normal[2] * a[2];
    }
    return { normal, offset: offset + drawPush };
  }

  function clearDrawOverlay() {
    for (const child of [...drawGroup.children]) {
      child.geometry?.dispose?.();
      child.material?.dispose?.();
      drawGroup.remove(child);
    }
  }

  function refreshDrawOverlay() {
    clearDrawOverlay();
    const outline = (pts, color, closed) => {
      if (pts.length < 2) return;
      const flat = [];
      for (const p of pts) flat.push(p[0], p[1], p[2]);
      if (closed) flat.push(pts[0][0], pts[0][1], pts[0][2]);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position",
        new THREE.BufferAttribute(new Float32Array(flat), 3));
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({
        color, depthTest: false, transparent: true,
      }));
      line.renderOrder = 200000;
      line.userData.atlasHelper = true;
      drawGroup.add(line);
    };
    const BOX_EDGES = [[0, 1], [1, 2], [2, 3], [3, 0],
                       [4, 5], [5, 6], [6, 7], [7, 4],
                       [0, 4], [1, 5], [2, 6], [3, 7]];
    for (const poly of drawnPolygons) {
      if (poly.kind === "sphere" && poly.points_world.length === 2) {
        const [c, surf] = poly.points_world;
        sphereWireframe(c, Math.hypot(surf[0] - c[0], surf[1] - c[1], surf[2] - c[2]),
                        0x7ddc86);
      } else if (poly.kind === "box" && poly.points_world.length === 8) {
        // An 8-corner solid is not a closed loop — drawing it as one would
        // trace a nonsense zigzag through the middle of the box.
        const flat = [];
        for (const [a, b] of BOX_EDGES) {
          flat.push(...poly.points_world[a], ...poly.points_world[b]);
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position",
          new THREE.BufferAttribute(new Float32Array(flat), 3));
        const seg = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({
          color: 0x7ddc86, depthTest: false, transparent: true,
        }));
        seg.renderOrder = 200000;
        seg.userData.atlasHelper = true;
        drawGroup.add(seg);
      } else {
        outline(poly.points_world, 0x7ddc86, true);
      }
    }
    if (boxOn && boxStage > 0) refreshBoxPreview();
    if (sphereOn && sphereStage === 1 && sphereRadius > 0) {
      sphereWireframe(sphereCentreNow(), sphereRadius, 0xffcc44);
    }
    if (editOn && drawnPolygons.length) {
      // Selected handles are drawn separately and hot, so a face grab is
      // visibly a FACE and not a single corner that happens to be under the
      // cursor.
      // Weld pair draws RED (both the dragged vertex and its target), hot
      // selection orange, the rest blue.
      const plain = [], hot = [], weld = [];
      for (const poly of drawnPolygons) {
        const selected = (editDrag && editDrag.poly === poly)
          ? new Set(editDrag.indices) : null;
        const weldIdx = (editWeld && editWeld.poly === poly) ? editWeld.index : -1;
        poly.points_world.forEach((q, i) => {
          if (i === weldIdx || (editWeld && selected && selected.has(i))) {
            weld.push(q[0], q[1], q[2]);
          } else if (selected && selected.has(i)) {
            hot.push(q[0], q[1], q[2]);
          } else {
            plain.push(q[0], q[1], q[2]);
          }
        });
      }
      const addPoints = (arr, color, size) => {
        if (!arr.length) return;
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position",
          new THREE.BufferAttribute(new Float32Array(arr), 3));
        const pts = new THREE.Points(geo, new THREE.PointsMaterial({
          color, size, sizeAttenuation: false, depthTest: false,
        }));
        pts.renderOrder = 200002;
        pts.userData.atlasHelper = true;
        drawGroup.add(pts);
      };
      addPoints(plain, 0x50d0ff, 10);
      addPoints(hot, 0xff9040, 14);
      addPoints(weld, 0xff3030, 16);
    }
    if (extrudeDrag && extrudeDrag.delta) {
      const { a, b, delta: d } = extrudeDrag;
      outline([a, b,
               [b[0] + d[0], b[1] + d[1], b[2] + d[2]],
               [a[0] + d[0], a[1] + d[1], a[2] + d[2]]], 0xffe066, true);
    }
    if (quadOn && quadPoints.length) {
      outline(quadPoints, 0xffe066, quadPoints.length >= 3);
      const flat = [];
      for (const p of quadPoints) flat.push(p[0], p[1], p[2]);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position",
        new THREE.BufferAttribute(new Float32Array(flat), 3));
      const dots = new THREE.Points(geo, new THREE.PointsMaterial({
        color: 0xffe066, size: 9, sizeAttenuation: false, depthTest: false,
      }));
      dots.renderOrder = 200001;
      dots.userData.atlasHelper = true;
      drawGroup.add(dots);
    }
    outline(drawPoints, 0xffffff, false);
    if (drawPoints.length) {
      const flat = [];
      for (const p of drawPoints) flat.push(p[0], p[1], p[2]);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position",
        new THREE.BufferAttribute(new Float32Array(flat), 3));
      const dots = new THREE.Points(geo, new THREE.PointsMaterial({
        color: 0xffcc44, size: 8, sizeAttenuation: false, depthTest: false,
      }));
      dots.renderOrder = 200001;
      dots.userData.atlasHelper = true;
      drawGroup.add(dots);
    }
  }

  // Re-intersect every stored click ray with the CURRENT (tilted/pushed) plane.
  // The outline keeps its on-screen shape; only its depth/orientation moves.
  function recomputeDrawPoints() {
    const plane = drawPlaneAdjusted();
    if (!plane) return;
    for (let i = 0; i < drawPoints.length; i += 1) {
      const r = drawRays[i];
      if (!r) continue;
      const landed = atlasIntersectRayWithPlane(r[0], r[1], plane);
      if (landed) drawPoints[i] = landed;
    }
    refreshDrawOverlay();
  }

  // Nearest committed-outline vertex to a click, in NDC (so the hit radius is
  // a constant ~14 px on screen whatever the preview resolution).
  // Closest point on an infinite AXIS line to a cursor ray (Ericson's
  // line-line closest approach). This is what makes a Shift-constrained drag
  // track the cursor along one world axis instead of sliding freely in depth:
  // without it a "drag upwards" also moved X and Z, and a pass over distant
  // geometry threw the corner across the scene (reported live). Plain-array
  // maths so tests/test_frontend_mirrors.py can execute it under node.
  function atlasClosestPointOnAxis(rayOrigin, rayDir, axisOrigin, axisDir) {
    const dl = Math.hypot(rayDir[0], rayDir[1], rayDir[2]) || 1;
    const d1 = [rayDir[0] / dl, rayDir[1] / dl, rayDir[2] / dl];
    const al = Math.hypot(axisDir[0], axisDir[1], axisDir[2]) || 1;
    const d2 = [axisDir[0] / al, axisDir[1] / al, axisDir[2] / al];
    const w0 = [rayOrigin[0] - axisOrigin[0],
                rayOrigin[1] - axisOrigin[1],
                rayOrigin[2] - axisOrigin[2]];
    const b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2];
    const denom = 1 - b * b;
    if (Math.abs(denom) < 1e-9) return null;      // sighting along the axis
    const d = d1[0] * w0[0] + d1[1] * w0[1] + d1[2] * w0[2];
    const e = d2[0] * w0[0] + d2[1] * w0[1] + d2[2] * w0[2];
    const t = (e - b * d) / denom;
    return [axisOrigin[0] + t * d2[0],
            axisOrigin[1] + t * d2[1],
            axisOrigin[2] + t * d2[2]];
  }

  const EDIT_AXES = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };

  // Which world axis the artist means, from the direction they actually
  // dragged on screen: project each axis at the grabbed corner and take the
  // one whose screen direction best matches the cursor delta. Locked for the
  // rest of the gesture so it cannot flip mid-drag.
  function pickConstraintAxis(origin, deltaNdc) {
    const len = Math.hypot(deltaNdc.x, deltaNdc.y);
    if (len < 1e-4) return null;
    const dir = { x: deltaNdc.x / len, y: deltaNdc.y / len };
    const base = new THREE.Vector3(...origin).project(camera);
    let best = null, bestScore = 0;
    for (const [name, axis] of Object.entries(EDIT_AXES)) {
      const tip = new THREE.Vector3(
        origin[0] + axis[0], origin[1] + axis[1], origin[2] + axis[2]
      ).project(camera);
      const sx = tip.x - base.x, sy = tip.y - base.y;
      const slen = Math.hypot(sx, sy);
      if (slen < 1e-6) continue;                  // axis points at the camera
      const score = Math.abs((sx / slen) * dir.x + (sy / slen) * dir.y);
      if (score > bestScore) { bestScore = score; best = name; }
    }
    return best;
  }

  function closestPointOnSegment(point, a, b) {
    const ab = b.clone().sub(a);
    const len2 = ab.lengthSq();
    if (len2 < 1e-12) return a.clone();
    let t = point.clone().sub(a).dot(ab) / len2;
    t = Math.max(0, Math.min(1, t));
    return a.clone().add(ab.multiplyScalar(t));
  }

  // Snap a surface hit onto the nearest EDGE of the triangle it landed on --
  // and, since the closest point on a segment collapses to an endpoint near a
  // corner, onto the nearest vertex too. A torn hole's rim IS mesh edges, so
  // this is what makes a drawn patch actually meet the geometry instead of
  // floating near it. Falls back to the raw surface hit when the cursor is not
  // close to any edge, so a point can still be placed mid-face.
  function snapHitToEdge(hit, mapped) {
    const pos = hit.object?.geometry?.attributes?.position;
    if (!hit.face || !pos) return hit.point.clone();
    const vertex = (i) => new THREE.Vector3()
      .fromBufferAttribute(pos, i).applyMatrix4(hit.object.matrixWorld);
    const tri = [vertex(hit.face.a), vertex(hit.face.b), vertex(hit.face.c)];
    const tol = 2 * 12 / Math.max(canvas.height / previewScale, 1); // ~12 px on SCREEN, not backbuffer
    let best = null, bestD = Infinity;
    for (let e = 0; e < 3; e += 1) {
      const cand = closestPointOnSegment(hit.point, tri[e], tri[(e + 1) % 3]);
      const proj = cand.clone().project(camera);
      const d = Math.hypot(proj.x - mapped.x, proj.y - mapped.y);
      if (d < bestD) { bestD = d; best = cand; }
    }
    return (best && bestD <= tol) ? best : hit.point.clone();
  }

  // The six quads of a blockout box over the canonical corner order. MIRRORS
  // core/polygon_planes.BOX_QUADS by hand (pinned in test_frontend_mirrors.py):
  // the two must agree or a face selected here would not be the face Python
  // closes into triangles.
  const EDIT_BOX_QUADS = [
    [0, 1, 2, 3], [4, 5, 6, 7],
    [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
  ];

  function projectToNdc(p) {
    return new THREE.Vector3(p[0], p[1], p[2]).project(camera);
  }

  function pointInScreenPolygon(mapped, pts) {
    const proj = pts.map(projectToNdc);
    let inside = false;
    for (let i = 0, j = proj.length - 1; i < proj.length; j = i, i += 1) {
      const a = proj[i], b = proj[j];
      if ((a.y > mapped.y) !== (b.y > mapped.y)
          && mapped.x < ((b.x - a.x) * (mapped.y - a.y)) / (b.y - a.y) + a.x) {
        inside = !inside;
      }
    }
    return inside;
  }

  function cameraDistanceTo(pts) {
    const eye = camera.getWorldPosition(new THREE.Vector3());
    let cx = 0, cy = 0, cz = 0;
    for (const p of pts) { cx += p[0]; cy += p[1]; cz += p[2]; }
    const n = pts.length || 1;
    return Math.hypot(cx / n - eye.x, cy / n - eye.y, cz / n - eye.z);
  }

  // A whole FACE under the cursor, so a cube's lid can be raised in one drag
  // rather than four. For a box that is one of its six quads; for a polygon it
  // is the outline itself, which moves the whole shape. Nearest face wins when
  // several overlap on screen.
  function findFaceUnder(mapped) {
    let best = null, bestDepth = Infinity;
    for (const poly of drawnPolygons) {
      if (poly.enabled === false) continue;
      const groups = (poly.kind === "box" && poly.points_world.length === 8)
        ? EDIT_BOX_QUADS
        : [poly.points_world.map((_, i) => i)];
      for (const indices of groups) {
        if (indices.length < 3) continue;
        const pts = indices.map((i) => poly.points_world[i]);
        if (!pointInScreenPolygon(mapped, pts)) continue;
        const depth = cameraDistanceTo(pts);
        if (depth < bestDepth) { bestDepth = depth; best = { poly, indices: [...indices] }; }
      }
    }
    return best;
  }

  function selectionCentroid(points) {
    const c = [0, 0, 0];
    for (const p of points) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
    const n = points.length || 1;
    return [c[0] / n, c[1] / n, c[2] / n];
  }

  // ✎ translate gizmo — three screen-constant coloured axis arrows at the
  // persistent selection (editSel), Maya-style. Grabbing an arrow tip starts an
  // axis-locked drag of that selection: it only PRE-SETS editDrag.axis — all
  // the movement maths is the existing atlasClosestPointOnAxis path in
  // onEditPointerMove, unchanged. Lives directly in `scene` (drawGroup is
  // cleared by every refreshDrawOverlay, which runs per drag frame) and is
  // tagged atlasHelper on the group and every child so render/export passes
  // skip it — same discipline as the pivot gizmo and the ground grid.
  const GIZMO_AXIS_COLORS = { x: 0xff5555, y: 0x55ff55, z: 0x5599ff };
  const GIZMO_SHAFT_LEN = 4;      // in gizmo-local units, scaled per frame
  const GIZMO_TIP_AT = 4.6;
  let editGizmo = null;
  function ensureEditGizmo() {
    if (editGizmo) return;
    editGizmo = new THREE.Group();
    editGizmo.name = "atlas_edit_gizmo";
    editGizmo.userData.atlasHelper = true;
    for (const [name, axis] of Object.entries(EDIT_AXES)) {
      const col = GIZMO_AXIS_COLORS[name];
      const shaft = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(0, 0, 0),
          new THREE.Vector3(axis[0] * GIZMO_SHAFT_LEN, axis[1] * GIZMO_SHAFT_LEN,
                            axis[2] * GIZMO_SHAFT_LEN)]),
        new THREE.LineBasicMaterial({
          color: col, depthTest: false, transparent: true, opacity: 0.9 }));
      shaft.renderOrder = 200003;
      const tip = new THREE.Mesh(
        new THREE.ConeGeometry(0.45, 1.2, 10),
        new THREE.MeshBasicMaterial({
          color: col, depthTest: false, transparent: true, opacity: 0.9 }));
      tip.position.set(axis[0] * GIZMO_TIP_AT, axis[1] * GIZMO_TIP_AT,
                       axis[2] * GIZMO_TIP_AT);
      // ConeGeometry points +Y; orient it along its axis.
      if (name === "x") tip.rotation.z = -Math.PI / 2;
      if (name === "z") tip.rotation.x = Math.PI / 2;
      tip.renderOrder = 200004;
      shaft.userData.atlasHelper = true; shaft.userData.atlasGizmoAxis = name;
      tip.userData.atlasHelper = true; tip.userData.atlasGizmoAxis = name;
      editGizmo.add(shaft, tip);
    }
    editGizmo.visible = false;
    scene.add(editGizmo);
  }
  function gizmoTipWorld(name) {
    const a = EDIT_AXES[name];
    const s = editGizmo.scale.x;
    const p = editGizmo.position;
    return [p.x + a[0] * GIZMO_TIP_AT * s,
            p.y + a[1] * GIZMO_TIP_AT * s,
            p.z + a[2] * GIZMO_TIP_AT * s];
  }
  // Called per-frame from animate() beside updatePivotGizmo, so it tracks the
  // selection through drags and stays a constant on-screen size while orbiting.
  function updateEditGizmo() {
    if (!editSel || !editOn || drawOn || boxOn || sphereOn || quadOn
        || extrudeOn || wandOn
        || !drawnPolygons.includes(editSel.poly)) {
      if (editGizmo) editGizmo.visible = false;
      return;
    }
    ensureEditGizmo();
    const pts = editSel.indices
      .map((i) => editSel.poly.points_world[i]).filter(Boolean);
    if (!pts.length) { editGizmo.visible = false; return; }
    const c = selectionCentroid(pts);
    editGizmo.position.set(c[0], c[1], c[2]);
    const dist = camera.position.distanceTo(editGizmo.position) || 10;
    editGizmo.scale.setScalar(Math.max(dist * 0.02, 0.02));
    editGizmo.visible = true;
  }
  // Nearest arrow tip to the cursor in NDC — the same constant-pixel tolerance
  // findEditPointNear uses, so grabbing a gizmo feels like grabbing a handle.
  function pickGizmoAxis(mapped) {
    if (!editGizmo || !editGizmo.visible) return null;
    const tol = 2 * 14 / Math.max(canvas.height / previewScale, 1);
    let best = null, bestD = Infinity;
    for (const name of Object.keys(EDIT_AXES)) {
      const t = gizmoTipWorld(name);
      const proj = new THREE.Vector3(t[0], t[1], t[2]).project(camera);
      const d = Math.hypot(proj.x - mapped.x, proj.y - mapped.y);
      if (d < bestD) { bestD = d; best = name; }
    }
    return bestD <= tol ? best : null;
  }
  // axis = "x"|"y"|"z" brightens that arrow and dims the others (the visual
  // "axis locked" feedback, shared by gizmo drags AND Shift / X/Y/Z locks);
  // null restores the resting look.
  function setGizmoAxisEmphasis(axis) {
    if (!editGizmo) return;
    for (const child of editGizmo.children) {
      const mine = child.userData.atlasGizmoAxis === axis;
      child.material.opacity = axis ? (mine ? 1.0 : 0.25) : 0.9;
    }
  }

  function applyEditDelta(delta) {
    const d = delta;
    editDrag.indices.forEach((idx, k) => {
      const o = editDrag.origins[k];
      editDrag.poly.points_world[idx] = [o[0] + d[0], o[1] + d[1], o[2] + d[2]];
    });
  }

  function findEditPointNear(mapped) {
    const tol = 2 * 14 / Math.max(canvas.height / previewScale, 1);
    let best = null, bestD = Infinity;
    drawnPolygons.forEach((poly, pi) => {
      poly.points_world.forEach((q, vi) => {
        const proj = new THREE.Vector3(q[0], q[1], q[2]).project(camera);
        const d = Math.hypot(proj.x - mapped.x, proj.y - mapped.y);
        if (d < bestD) { bestD = d; best = { poly, pi, index: vi }; }
      });
    });
    return bestD <= tol ? best : null;
  }

  // Nearest vertex of ANOTHER shape to the cursor — the weld target. Welding
  // is what actually closes the hairline between two quads that copied a
  // shared edge: on release the dragged vertex takes the target's exact
  // coordinates, so the two outlines coincide instead of nearly-meeting.
  // Same-shape vertices are excluded: a weld inside one outline would leave a
  // duplicate point its own triangulation chokes on.
  function findWeldTarget(mapped) {
    const tol = 2 * 14 / Math.max(canvas.height / previewScale, 1);
    let best = null, bestD = Infinity;
    for (const poly of drawnPolygons) {
      if (editDrag && poly === editDrag.poly) continue;
      poly.points_world.forEach((q, vi) => {
        const proj = new THREE.Vector3(q[0], q[1], q[2]).project(camera);
        const d = Math.hypot(proj.x - mapped.x, proj.y - mapped.y);
        if (d < bestD) { bestD = d; best = { poly, index: vi, point: q }; }
      });
    }
    return bestD <= tol ? best : null;
  }

  function editRayToPolygonPlane(mapped, poly, index) {
    // Drag WITHIN the outline's own plane: an N-gon that stops being planar
    // cannot be triangulated or projected sanely. A BOX has no single plane,
    // so its corners fall back to a camera-facing plane through the corner
    // itself — a free drag in screen space at that corner's depth.
    drawRaycaster.setFromCamera(new THREE.Vector2(mapped.x, mapped.y), camera);
    const o = camera.getWorldPosition(new THREE.Vector3());
    const d = drawRaycaster.ray.direction;
    let plane = poly.plane;
    if (!plane || !plane.normal) {
      const q = (editDrag && editDrag.poly === poly && editDrag.origin)
        ? editDrag.origin
        : (poly.points_world[index] || poly.points_world[0]);
      const fwd = camera.getWorldDirection(new THREE.Vector3());
      const n = [fwd.x, fwd.y, fwd.z];
      plane = { normal: n, offset: n[0] * q[0] + n[1] * q[1] + n[2] * q[2] };
    }
    return atlasIntersectRayWithPlane([o.x, o.y, o.z], [d.x, d.y, d.z], plane);
  }

  function mappedFromEvent(ev) {
    return atlasContainNdc(ev.clientX, ev.clientY, canvas.getBoundingClientRect(),
                           canvas.width, canvas.height);
  }

  // CAPTURE phase: createOrbitControls binds pointerdown on the bubble phase,
  // so grabbing a handle here (and stopping propagation) suppresses the orbit
  // drag for that gesture only — orbiting anywhere else still works while
  // editing, which is how you reach a point on the far side.
  function onEditPointerDown(ev) {
    if (!editOn) return;
    const mapped = mappedFromEvent(ev);
    if (!mapped.inside) return;
    // A gizmo arrow under the cursor wins over everything (Maya order): it
    // starts an axis-locked drag of the persistent selection with the axis
    // already chosen — the move handler's existing axis branch does the rest.
    if (editSel && editGizmo?.visible && !ev.ctrlKey && !ev.metaKey) {
      const axis = pickGizmoAxis(mapped);
      if (axis) {
        const origins = editSel.indices.map((i) => [...editSel.poly.points_world[i]]);
        editDrag = {
          poly: editSel.poly,
          indices: [...editSel.indices],
          index: editSel.indices[0],
          origins,
          origin: selectionCentroid(origins),
          startNdc: { x: mapped.x, y: mapped.y },
          axis,
        };
        setGizmoAxisEmphasis(axis);
        drawHud(`✎ ${axis.toUpperCase()} axis drag`
                + (editDrag.indices.length > 1
                   ? ` — moving ${editDrag.indices.length} points` : ""));
        canvas.setPointerCapture?.(ev.pointerId);
        refreshDrawOverlay();
        ev.stopPropagation();
        ev.preventDefault();
        return;
      }
    }
    const found = findEditPointNear(mapped);
    // A vertex under the cursor wins over the face behind it; otherwise fall
    // through to face selection, so clicking a cube's lid grabs all four of
    // its corners at once.
    const sel = found
      ? { poly: found.poly, indices: [found.index], pi: found.pi }
      : findFaceUnder(mapped);
    if (!sel) {
      // Empty click deselects (gizmo hides) but must NOT eat the event —
      // orbiting on empty space still has to work while editing.
      editSel = null;
      return;
    }

    if (ev.ctrlKey || ev.metaKey) {
      const pi = drawnPolygons.indexOf(sel.poly);
      if (sel.poly.kind === "box" || sel.poly.kind === "sphere"
          || sel.indices.length > 1 || sel.poly.points_world.length <= 3) {
        // Deleting one corner of a solid would leave a shape its builder
        // cannot close, so the whole thing goes.
        drawnPolygons.splice(pi, 1);
        drawHud("✎ shape deleted");
      } else {
        sel.poly.points_world.splice(sel.indices[0], 1);
        drawHud("✎ point deleted");
      }
      // Stored indices are stale after any delete on that shape.
      if (editSel && editSel.poly === sel.poly) editSel = null;
      drawDirty = true;
      refreshDrawOverlay();
      ev.stopPropagation();
      ev.preventDefault();
      return;
    }

    const origins = sel.indices.map((i) => [...sel.poly.points_world[i]]);
    editDrag = {
      poly: sel.poly,
      indices: sel.indices,
      index: sel.indices[0],
      origins,
      origin: selectionCentroid(origins),   // the reference the drag moves
      startNdc: { x: mapped.x, y: mapped.y },
      axis: null,
    };
    // The grab becomes the persistent selection: on release the translate
    // gizmo appears here for axis-clean follow-up nudges.
    editSel = { poly: sel.poly, indices: [...sel.indices] };
    editWeld = null;
    drawHud(sel.indices.length > 1
      ? `✎ face selected (${sel.indices.length} points) — `
        + "SHIFT locks an axis"
      : "✎ point grabbed — SHIFT locks an axis");
    canvas.setPointerCapture?.(ev.pointerId);
    refreshDrawOverlay();
    ev.stopPropagation();
    ev.preventDefault();
  }

  function onEditPointerMove(ev) {
    if (!editDrag) {
      // Idle hover: light up the arrow tip under the cursor so the gizmo
      // reads as grabbable before it is grabbed.
      if (editOn && editSel && editGizmo?.visible) {
        const hoverMapped = mappedFromEvent(ev);
        setGizmoAxisEmphasis(hoverMapped.inside ? pickGizmoAxis(hoverMapped) : null);
      }
      return;
    }
    const mapped = mappedFromEvent(ev);
    if (!mapped.inside) return;

    // Shift locks the drag to ONE world axis. Without it a "drag upwards" also
    // moved X and Z, and passing the cursor over distant geometry snapped the
    // corner to that surface, throwing it across the scene (reported live).
    // The axis is chosen from the direction actually dragged and then held for
    // the rest of the gesture; X / Y / Z force it explicitly.
    if (ev.shiftKey || editDrag.axis) {
      editWeld = null;
      if (!editDrag.axis) {
        editDrag.axis = pickConstraintAxis(editDrag.origin, {
          x: mapped.x - editDrag.startNdc.x,
          y: mapped.y - editDrag.startNdc.y,
        });
      }
      if (editDrag.axis) {
        drawRaycaster.setFromCamera(new THREE.Vector2(mapped.x, mapped.y), camera);
        const o = camera.getWorldPosition(new THREE.Vector3());
        const rd = drawRaycaster.ray.direction;
        const onAxis = atlasClosestPointOnAxis(
          [o.x, o.y, o.z], [rd.x, rd.y, rd.z],
          editDrag.origin, EDIT_AXES[editDrag.axis]);
        if (onAxis) {
          applyEditDelta([onAxis[0] - editDrag.origin[0],
                          onAxis[1] - editDrag.origin[1],
                          onAxis[2] - editDrag.origin[2]]);
          setGizmoAxisEmphasis(editDrag.axis);
          drawDirty = true;
          refreshDrawOverlay();
          drawHud(`✎ ${editDrag.axis.toUpperCase()} axis locked`
                  + (editDrag.indices.length > 1
                     ? ` — moving ${editDrag.indices.length} points`
                     : " — release Shift for a free drag"));
          ev.stopPropagation();
          return;
        }
      }
    }

    // A multi-point (face) selection TRANSLATES: snapping a face by its
    // centroid onto a mesh edge would mean nothing, so only a lone vertex
    // takes the edge snap below.
    if (editDrag.indices.length > 1) {
      editWeld = null;
      const target = editRayToPolygonPlane(mapped, editDrag.poly, editDrag.index);
      if (!target) return;
      applyEditDelta([target[0] - editDrag.origin[0],
                      target[1] - editDrag.origin[1],
                      target[2] - editDrag.origin[2]]);
      drawDirty = true;
      refreshDrawOverlay();
      ev.stopPropagation();
      return;
    }

    // Otherwise snap to geometry FIRST: constraining the drag to the outline's
    // own plane put the point where the PLANE is, not under the cursor.
    // An adjoining shape's VERTEX outranks a mesh edge: near one, both draw
    // red and the drag locks onto its exact coordinates — release welds.
    let landed = null;
    editWeld = editSnap ? findWeldTarget(mapped) : null;
    if (editWeld) {
      landed = [...editWeld.point];
    } else if (editSnap) {
      drawRaycaster.setFromCamera(new THREE.Vector2(mapped.x, mapped.y), camera);
      const hits = drawRaycaster.intersectObjects(drawTargets(), false);
      if (hits.length) {
        const snapped = snapHitToEdge(hits[0], mapped);
        landed = [snapped.x, snapped.y, snapped.z];
      }
    }
    if (!landed) landed = editRayToPolygonPlane(mapped, editDrag.poly, editDrag.index);
    if (!landed) return;
    editDrag.poly.points_world[editDrag.index] = landed;
    if (editWeld) drawHud("✎ on an adjoining vertex — release to weld");
    drawDirty = true;
    refreshDrawOverlay();
    ev.stopPropagation();
  }

  function onEditPointerUp(ev) {
    if (!editDrag) return;
    // A single-vertex drag welds ON RELEASE too, when it ends on another
    // shape's vertex — this covers gizmo / Shift / X-Y-Z axis drags, which
    // bypass the live weld preview to keep the axis pure but should still
    // weld when they land on a corner.
    if (editDrag.indices.length === 1 && editSnap && !editWeld) {
      const upMapped = mappedFromEvent(ev);
      const t = upMapped.inside ? findWeldTarget(upMapped) : null;
      if (t) {
        editDrag.poly.points_world[editDrag.index] = [...t.point];
        editWeld = t;
        drawDirty = true;
      }
    }
    // Snapped points can leave the original plane, so re-fit it from what the
    // outline now IS. The stored plane drives the emitted normal and the UV
    // basis, so a stale one would misorient the projection.
    if (editDrag.poly.kind !== "box" && editDrag.poly.kind !== "sphere") {
      const refit = atlasEstablishPlaneFromHits(editDrag.poly.points_world);
      if (refit) editDrag.poly.plane = refit;
    }
    const welded = !!editWeld;
    editWeld = null;
    editDrag = null;
    setGizmoAxisEmphasis(null);
    canvas.releasePointerCapture?.(ev.pointerId);
    refreshDrawOverlay();
    drawHud(welded
      ? "✎ welded — the two outlines now share that corner exactly · ✅ Apply rebuilds"
      : "✎ moved — drag a gizmo arrow for an axis-clean nudge, "
        + "then ✅ Apply to rebuild");
    ev.stopPropagation();
  }

  function drawHud(message) {
    drawStatus.textContent = message;
    drawStatus.style.display = message ? "block" : "none";
  }

  function drawStatusLine() {
    if (!drawPlane) {
      return `✏️ click surfaces to set the plane — ${drawHits.length} hit(s), ` +
        "2 raises a vertical plane, 3+ best-fit";
    }
    const rule = drawHits.length >= 3 ? "best-fit" : "vertical through 2 hits";
    return `✏️ plane set (${rule}) · ${drawPoints.length} point(s) · ` +
      `tilt ${(drawTilt * 180 / Math.PI).toFixed(0)}° push ${drawPush.toFixed(2)}m\n` +
      "click to add · ctrl-click deletes · Enter closes · Esc discards";
  }

  function onDrawClick(ev) {
    if (!drawOn) return;
    const box = canvas.getBoundingClientRect();
    const mapped = atlasContainNdc(ev.clientX, ev.clientY, box,
                                   canvas.width, canvas.height);
    if (!mapped.inside) {
      drawHud("✏️ that click is in the letterbox bar, not the image");
      return;
    }
    const ndc = new THREE.Vector2(mapped.x, mapped.y);

    // Ctrl/Cmd-click removes the point under the cursor. Compared in NDC with
    // a tolerance derived from the drawing buffer, so the hit radius is a
    // constant ~14 px on screen whatever the preview resolution.
    if (ev.ctrlKey || ev.metaKey) {
      const tol = 2 * 14 / Math.max(canvas.height / previewScale, 1);
      let best = -1, bestD = Infinity;
      for (let i = 0; i < drawPoints.length; i += 1) {
        const q = new THREE.Vector3(...drawPoints[i]).project(camera);
        const d = Math.hypot(q.x - mapped.x, q.y - mapped.y);
        if (d < bestD) { bestD = d; best = i; }
      }
      if (best >= 0 && bestD <= tol) {
        drawPoints.splice(best, 1);
        drawRays.splice(best, 1);
        // A removed point may also have been one of the plane-defining hits.
        if (drawHits.length > drawPoints.length) {
          drawHits = drawHits.slice(0, drawPoints.length);
          drawPlane = drawHits.length >= 2
            ? atlasEstablishPlaneFromHits(drawHits) : null;
        }
        refreshDrawOverlay();
        drawHud(drawStatusLine());
      } else {
        drawHud("✏️ ctrl-click a point to delete it (none under the cursor)");
      }
      return;
    }

    drawRaycaster.setFromCamera(ndc, camera);
    const rayOrigin = camera.getWorldPosition(new THREE.Vector3());
    const rayDir = drawRaycaster.ray.direction.clone();
    // Remembered alongside the point (only once the point is ACCEPTED, so a
    // rejected click leaves no orphan): adjusting tilt/push re-intersects every
    // stored ray, keeping the outline where it was drawn ON SCREEN while the
    // plane moves under it.
    const ray = [[rayOrigin.x, rayOrigin.y, rayOrigin.z],
                 [rayDir.x, rayDir.y, rayDir.z]];

    // An existing drawn corner under the cursor outranks the mesh: the point
    // takes its exact coordinates (born welded), and counts as a plane-
    // establishing hit like any surface click.
    if (editSnap && !ev.shiftKey) {
      const t = findWeldTarget(mapped);
      if (t) {
        const world = [...t.point];
        drawPoints.push(world);
        drawRays.push(ray);
        if (!drawPlane) {
          drawHits.push(world);
          drawPlane = atlasEstablishPlaneFromHits(drawHits);
        }
        refreshDrawOverlay();
        drawHud(drawStatusLine());
        return;
      }
    }

    const hits = drawRaycaster.intersectObjects(drawTargets(), false);
    if (hits.length) {
      const p = editSnap ? snapHitToEdge(hits[0], mapped) : hits[0].point;
      const world = [p.x, p.y, p.z];
      drawPoints.push(world);
      drawRays.push(ray);
      // Only the clicks BEFORE the plane exists define it; afterwards a hit is
      // just another outline point, so the plane cannot drift mid-outline.
      if (!drawPlane) {
        drawHits.push(world);
        drawPlane = atlasEstablishPlaneFromHits(drawHits);
      }
    } else {
      const plane = drawPlaneAdjusted();
      if (!plane) {
        drawHud("✏️ nothing hit yet — the first clicks must land on geometry " +
                "to establish the plane");
        return;
      }
      const landed = atlasIntersectRayWithPlane(
        [rayOrigin.x, rayOrigin.y, rayOrigin.z],
        [rayDir.x, rayDir.y, rayDir.z], plane);
      if (!landed) {
        drawHud("✏️ that ray does not meet the plane — orbit and try again");
        return;
      }
      drawPoints.push(landed);
      drawRays.push(ray);
    }
    refreshDrawOverlay();
    drawHud(drawStatusLine());
  }

  function closeDrawnOutline() {
    if (drawPoints.length < 3) {
      drawHud("✏️ an outline needs at least 3 points");
      return;
    }
    const plane = drawPlaneAdjusted() || drawPlane;
    drawnPolygons.push({
      id: `d${drawnPolygons.length + 1}`,
      label: `drawn plane ${drawnPolygons.length + 1}`,
      enabled: true,
      points_world: drawPoints,
      plane: { normal: plane.normal, offset: plane.offset },
      established_from: {
        hits: drawHits.length,
        rule: drawHits.length >= 3 ? "best_fit_newell" : "vertical_through_two_hits",
      },
    });
    drawPoints = [];
    drawRays = [];
    drawHits = [];
    drawPlane = null;
    drawTilt = 0;
    drawPush = 0;
    if (drawTiltInput) drawTiltInput.value = "0";
    if (drawPushInput) drawPushInput.value = "0";
    refreshDrawOverlay();
    drawHud(`✏️ ${drawnPolygons.length} outline(s) ready — click ✅ Apply to build them`);
  }

  function onDrawKey(ev) {
    if (editDrag && "xyz".includes(ev.key.toLowerCase())) {
      editDrag.axis = ev.key.toLowerCase();
      setGizmoAxisEmphasis(editDrag.axis);
      drawHud(`✎ ${editDrag.axis.toUpperCase()} axis locked`);
      ev.preventDefault();
      return;
    }
    if (wandOn) {
      if (ev.key === "Enter" || ev.key === "Escape") {
        wandBtn.onclick();
        drawHud("🪄 done — orbit freely, ✅ Apply builds the fills");
        ev.preventDefault();
      } else if ((ev.key === "Backspace" || ev.key === "Delete")
                 && drawnPolygons.length) {
        drawnPolygons.pop();
        drawDirty = true;
        refreshDrawOverlay();
        drawHud("🪄 last fill removed");
        ev.preventDefault();
      }
      return;
    }
    if (extrudeOn) {
      if (ev.key === "Enter" || ev.key === "Escape") {
        extrudeBtn.onclick();
        drawHud("➬ done — orbit freely, ✅ Apply builds the extrusions");
        ev.preventDefault();
      } else if ((ev.key === "Backspace" || ev.key === "Delete")
                 && drawnPolygons.length) {
        drawnPolygons.pop();
        drawDirty = true;
        refreshDrawOverlay();
        drawHud("➬ last shape removed");
        ev.preventDefault();
      }
      return;
    }
    if (quadOn) {
      if (ev.key === "Enter") {
        // Enter CONFIRMS AND EXITS: the artist's next move is orbiting to the
        // next hole, so drop straight back to the orbit cursor. An unfinished
        // quad is discarded; committed ones stay.
        quadBtn.onclick();
        drawHud("⬜ done — orbit to the next hole, ✅ Apply builds the fills");
        ev.preventDefault();
      } else if (ev.key === "Escape") {
        // Esc stays IN the tool: discard the in-progress quad / end the strip.
        quadPoints = [];
        quadPrev = null;
        refreshDrawOverlay();
        drawHud(quadStatusLine());
        ev.preventDefault();
      } else if (ev.key === "Backspace" || ev.key === "Delete") {
        if (quadPoints.length) {
          quadPoints.pop();
        } else if (drawnPolygons.length) {
          // commitQuad stores the SAME array in the shape and in quadPrev, so
          // identity tells us whether the popped shape was this strip's last.
          const popped = drawnPolygons.pop();
          if (quadPrev && popped.points_world === quadPrev) quadPrev = null;
          drawDirty = true;
        }
        refreshDrawOverlay();
        drawHud(quadStatusLine());
        ev.preventDefault();
      }
      return;
    }
    if (sphereOn) {
      if (ev.key === "Enter") {
        // Enter ALWAYS exits to orbit (any tool, any state — the artist's
        // next move is orbiting): commit the in-progress sphere when it has a
        // radius, discard a half-placed one.
        if (sphereStage === 1 && sphereRadius > 0.05) finishSphere();
        sphereBtn.onclick();
        drawHud("● done — orbit freely, ✅ Apply builds");
        ev.preventDefault();
      }
      else if (ev.key === "Escape") {
        sphereStage = 0; sphereContact = null; sphereRadius = 0;
        refreshDrawOverlay(); drawHud(sphereStatusLine()); ev.preventDefault();
      }
      return;
    }
    if (boxOn) {
      if (ev.key === "Enter") {
        // Enter ALWAYS exits to orbit: commit the box when it has height,
        // discard a half-built one.
        if (boxStage >= 2 && boxHeight > 0.05) finishBox();
        boxBtn.onclick();
        drawHud("▣ done — orbit freely, ✅ Apply builds");
        ev.preventDefault();
      }
      else if (ev.key === "Escape") {
        boxStage = 0; boxBase = null; boxOpposite = null; boxHeight = 0;
        refreshDrawOverlay(); drawHud(boxStatusLine()); ev.preventDefault();
      }
      return;
    }
    if (!drawOn) return;
    if (ev.key === "Enter") {
      // Enter ALWAYS exits to orbit: close the outline when it has 3+ points,
      // discard a shorter one.
      if (drawPoints.length >= 3) closeDrawnOutline();
      drawBtn.onclick();
      drawHud("✏️ done — orbit freely, ✅ Apply builds");
      ev.preventDefault();
    }
    else if (ev.key === "Escape") {
      drawPoints = []; drawRays = []; drawHits = []; drawPlane = null;
      refreshDrawOverlay(); drawHud(drawStatusLine()); ev.preventDefault();
    } else if (ev.key === "Backspace" || ev.key === "Delete") {
      if (drawPoints.length) {
        drawPoints.pop();
        drawRays.pop();
        if (drawHits.length > drawPoints.length) {
          drawHits.pop();
          drawPlane = atlasEstablishPlaneFromHits(drawHits);
        }
      } else if (drawnPolygons.length) {
        drawnPolygons.pop();
      }
      refreshDrawOverlay(); drawHud(drawStatusLine()); ev.preventDefault();
    }
  }

  function persistDrawnPolygonsToClientData() {
    const widget = node.widgets?.find((w) => w.name === "client_data");
    if (!widget) return;
    let existing = {};
    try { existing = widget.value ? JSON.parse(widget.value) : {}; } catch (_) { existing = {}; }
    // Merge, like every other viewport button, so Apply never clobbers a
    // baked path or a stored render pass.
    existing.drawn_polygons = drawnPolygons.map((p) => ({
      ...p,
      // Identity of the solve+image these were drawn against: the backend
      // drops them rather than re-applying an outline authored on another
      // photo (the same discipline 📐 Extract Angle uses).
      fingerprint: recoveredData?.solve_fingerprint || "",
    }));
    widget.value = JSON.stringify(existing);
    widget.callback?.(widget.value);
  }

  const drawStatus = document.createElement("div");
  drawStatus.style.cssText =
    "display:none;position:absolute;left:8px;bottom:8px;z-index:12;padding:6px 8px;" +
    "font:11px/1.45 monospace;white-space:pre;color:#dfe;background:rgba(20,26,20,.86);" +
    "border:1px solid #4a6;border-radius:4px;pointer-events:none;";
  canvasWrap.appendChild(drawStatus);

  // Scoped to the canvas, never the document: unrelated ComfyUI hotkeys must
  // keep working (the same rule createOrbitControls follows).
  canvas.addEventListener("click", onDrawClick);
  canvas.addEventListener("click", onQuadClick);
  canvas.addEventListener("click", onWandClick);
  canvas.addEventListener("click", onBoxClick);
  canvas.addEventListener("pointermove", onBoxMove);
  canvas.addEventListener("click", onSphereClick);
  canvas.addEventListener("pointermove", onSphereMove);
  canvas.addEventListener("keydown", onDrawKey);
  canvas.addEventListener("pointerdown", onEditPointerDown, true);
  canvas.addEventListener("pointermove", onEditPointerMove, true);
  canvas.addEventListener("pointerup", onEditPointerUp, true);
  canvas.addEventListener("pointerdown", onExtrudePointerDown, true);
  canvas.addEventListener("pointermove", onExtrudePointerMove, true);
  canvas.addEventListener("pointerup", onExtrudePointerUp, true);

  // Tilt / push: the "adjustable after" half of the plane rules. Both
  // re-intersect the STORED click rays, so the outline keeps the shape drawn
  // on screen while the plane rotates about the clicked edge or slides along
  // its own normal.
  // Shown as a contextual flyout beside the rail only while ✏️ Draw is active —
  // tilt/push belong to the draw gesture, not the main toolbar.
  const drawAdjust = document.createElement("div");
  drawAdjust.style.cssText =
    "display:none;align-items:center;gap:4px;font-size:10px;color:#9c9;" +
    "padding:2px 6px;background:#1a201a;border:1px solid #3a4a3a;border-radius:3px;" +
    "position:absolute;left:66px;top:46px;z-index:10;" +
    "pointer-events:auto;line-height:normal;white-space:nowrap;";
  const drawTiltInput = document.createElement("input");
  drawTiltInput.type = "range";
  drawTiltInput.min = "-60"; drawTiltInput.max = "60"; drawTiltInput.step = "1";
  drawTiltInput.value = "0";
  drawTiltInput.title = "Tilt the plane about the axis through the clicked edge";
  drawTiltInput.style.cssText = "width:70px;";
  const drawPushInput = document.createElement("input");
  drawPushInput.type = "range";
  drawPushInput.min = "-20"; drawPushInput.max = "20"; drawPushInput.step = "0.1";
  drawPushInput.value = "0";
  drawPushInput.title = "Push the plane along its own normal (metres)";
  drawPushInput.style.cssText = "width:70px;";
  const drawAdjustLabel = document.createElement("span");
  drawAdjustLabel.textContent = "tilt 0° push 0.0m";
  const onAdjust = () => {
    drawTilt = Number(drawTiltInput.value) * Math.PI / 180;
    drawPush = Number(drawPushInput.value);
    drawAdjustLabel.textContent =
      `tilt ${Number(drawTiltInput.value).toFixed(0)}° ` +
      `push ${drawPush.toFixed(1)}m`;
    recomputeDrawPoints();
    drawHud(drawStatusLine());
  };
  drawTiltInput.oninput = onAdjust;
  drawPushInput.oninput = onAdjust;
  drawAdjust.append(drawTiltInput, drawPushInput, drawAdjustLabel);
  canvasWrap.appendChild(drawAdjust);

  const drawBtn = document.createElement("button");
  drawBtn.textContent = "✏️ Draw";
  drawBtn.title = "Draw an N-gon in 3D to fill an occluded hole. Orbit until the "
    + "hole's surviving edges are visible, click them to establish the plane, then "
    + "click across the hole freely. Enter closes the outline, ✅ Apply builds them.";
  drawBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  drawBtn.onclick = () => {
    drawOn = !drawOn;
    if (drawOn && editOn) editBtn.onclick();
    if (drawOn && boxOn) boxBtn.onclick();
    if (drawOn && sphereOn) sphereBtn.onclick();
    if (drawOn && quadOn) quadBtn.onclick();
    if (drawOn && extrudeOn) extrudeBtn.onclick();
    if (drawOn && wandOn) wandBtn.onclick();
    drawBtn.style.background = drawOn ? "#1a3a1a" : "#2a2a2a";
    drawBtn.style.color = drawOn ? "#8f8" : "#ddd";
    // Orbiting while drawing would fight the click; the artist turns the
    // model first, then draws.
    controls.setEnabled(!drawOn);
    canvas.style.cursor = drawOn ? "crosshair" : "grab";
    drawHud(drawOn ? drawStatusLine() : "");
    drawAdjust.style.display = drawOn ? "flex" : "none";
    if (!drawOn) {
      drawPoints = []; drawRays = []; drawHits = []; drawPlane = null;
      refreshDrawOverlay();
    }
  };

  const editBtn = document.createElement("button");
  editBtn.textContent = "✎ Edit";
  editBtn.title = "Move points of outlines you have already drawn: drag a handle to "
    + "slide it WITHIN its own plane, ctrl-click a handle to delete it. Orbiting "
    + "still works — only the grab itself suppresses it. Click ✅ Apply to rebuild.";
  editBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  editBtn.onclick = () => {
    editOn = !editOn;
    editBtn.style.background = editOn ? "#1a2a3a" : "#2a2a2a";
    editBtn.style.color = editOn ? "#8cf" : "#ddd";
    if (editOn && drawOn) drawBtn.onclick();
    if (editOn && boxOn) boxBtn.onclick();
    if (editOn && sphereOn) sphereBtn.onclick();
    if (editOn && quadOn) quadBtn.onclick();
    if (editOn && extrudeOn) extrudeBtn.onclick();
    if (editOn && wandOn) wandBtn.onclick();
    editDrag = null;
    editSel = null;
    refreshDrawOverlay();
    drawHud(editOn
      ? (drawnPolygons.length
          ? "✎ drag a handle · SHIFT locks one axis (or press X/Y/Z) · "
            + "ctrl-click deletes · then ✅ Apply"
          : "✎ nothing drawn yet — use ✏️ Draw first")
      : "");
  };

  const snapBtn = document.createElement("button");
  snapBtn.textContent = "Snap";
  snapBtn.title = "Snap drawn and dragged points onto the nearest mesh edge/vertex "
    + "under the cursor. A torn hole's rim IS mesh edges, so this is what makes a "
    + "patch meet the geometry. Toggle off for free placement; in Edit, Shift constrains a drag to one axis instead.";
  snapBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#1a2a1a;color:#8f8;border:1px solid #444;border-radius:3px";
  snapBtn.onclick = () => {
    editSnap = !editSnap;
    snapBtn.style.background = editSnap ? "#1a2a1a" : "#2a2a2a";
    snapBtn.style.color = editSnap ? "#8f8" : "#ddd";
  };

  const drawApplyBtn = document.createElement("button");
  drawApplyBtn.textContent = "✅ Apply";
  drawApplyBtn.title = "Build the drawn outlines into the solve's geometry and re-queue, "
    + "so the `solve` output carries them downstream (retopo / export).";
  drawApplyBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  drawApplyBtn.onclick = () => {
    if (drawPoints.length >= 3) closeDrawnOutline();
    // Zero shapes is still applyable when dirty: deleting the LAST shape has
    // to persist the now-empty list, or the baked geometry could never be
    // removed again.
    if (!drawnPolygons.length && !drawDirty) {
      drawHud("✏️ nothing to apply — draw an outline and press Enter first");
      return;
    }
    drawDirty = false;
    persistDrawnPolygonsToClientData();
    drawHud(drawnPolygons.length
      ? `✅ applying ${drawnPolygons.length} outline(s) — re-queued`
      : "✅ removing the deleted shapes — re-queued");
    app.queuePrompt(0, 1);
  };

  const deleteBtn = document.createElement("button");
  deleteBtn.textContent = "🗑";
  deleteBtn.title = "Delete the selected shape (grab one in ✎ Edit first); with no "
    + "selection, deletes the most recent shape. Ctrl-click a handle in Edit still "
    + "deletes a single point. Click ✅ Apply to rebuild without it.";
  deleteBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  deleteBtn.onclick = () => {
    if (editSel && drawnPolygons.includes(editSel.poly)) {
      drawnPolygons.splice(drawnPolygons.indexOf(editSel.poly), 1);
      drawHud("🗑 selected shape deleted — ✅ Apply to rebuild without it");
    } else if (drawnPolygons.length) {
      drawnPolygons.pop();
      drawHud("🗑 last shape deleted — ✅ Apply to rebuild without it");
    } else {
      drawHud("🗑 nothing to delete");
      return;
    }
    editSel = null;
    editDrag = null;
    drawDirty = true;
    refreshDrawOverlay();
  };

  // Clear ALL drawn/filled shapes in one go. 🗑 deletes one at a time, which is
  // the wrong tool after a wand session leaves thirty fills on a torn mesh —
  // thirty clicks to start over (asked for live 2026-08-08).
  //
  // ARMED IN TWO CLICKS, never one: there is no undo in the viewport, and this
  // is the only control that can destroy an entire session's drawing. The first
  // click arms and says how many shapes are at stake; a second click within
  // CLEAR_ARM_MS commits; doing nothing disarms. Deliberately NOT a window
  // confirm() — a modal dialog blocks the whole ComfyUI page (and any automation
  // driving it) until dismissed, where an armed button costs nothing to ignore.
  const CLEAR_ARM_MS = 4000;
  let clearArmed = false;
  let clearArmTimer = null;
  const clearAllBtn = document.createElement("button");
  clearAllBtn.textContent = "🧹";
  clearAllBtn.title = "Clear ALL drawn and wand-filled shapes at once, plus any "
    + "outline still in progress. Click once to arm, again to confirm (there is "
    + "no undo). Click ✅ Apply afterwards to rebuild without them — an empty "
    + "list applies, so this really does unbake the geometry.";
  clearAllBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  function disarmClearAll() {
    clearArmed = false;
    if (clearArmTimer !== null) { clearTimeout(clearArmTimer); clearArmTimer = null; }
    clearAllBtn.style.background = "transparent";
    clearAllBtn.style.color = "#c9c9d1";
  }
  clearAllBtn.onclick = () => {
    const pending = drawPoints.length;
    if (!drawnPolygons.length && !pending) {
      disarmClearAll();
      drawHud("🧹 nothing to clear");
      return;
    }
    if (!clearArmed) {
      clearArmed = true;
      clearAllBtn.style.background = "#4a1d1d";
      clearAllBtn.style.color = "#ff9b9b";
      drawHud(`🧹 clear ALL ${drawnPolygons.length} shape(s)`
        + (pending ? ` + the outline in progress` : "")
        + ` — click 🧹 again to confirm, there is no undo`);
      clearArmTimer = setTimeout(() => {
        if (!clearArmed) return;
        disarmClearAll();
        drawHud("🧹 clear-all disarmed");
      }, CLEAR_ARM_MS);
      return;
    }
    const n = drawnPolygons.length;
    disarmClearAll();
    drawnPolygons.length = 0;          // same array — the overlay/payload hold it
    drawPoints = []; drawRays = []; drawHits = []; drawPlane = null;
    editSel = null;
    editDrag = null;
    drawDirty = true;                  // an EMPTY list still applies when dirty
    refreshDrawOverlay();
    drawHud(`🧹 cleared ${n} shape(s) — ✅ Apply to rebuild without them`);
  };

  // ---------------------------------------------------------------------------
  // Box — three-stage blockout solid: footprint on the ground, then extrude up.
  //
  // A plane fills a hole you can see through; a box fills a MASS the camera
  // never saw round the back of. On an XYZ perspective plate the ground plane
  // is already known (scale is reconciled so ground sits at Y=0), which is what
  // makes a footprint-then-extrude gesture well-defined from a single view.
  //
  // Stages: 1 = pick the base corner, 2 = drag the footprint, 3 = drag the
  // height. Enter or a third click finishes; Esc cancels. The result is stored
  // as 8 corners in the SAME points_world array a polygon uses, so Edit's
  // handles, snapping and deletion all work on a box without a second code
  // path (core/polygon_planes.box_mesh_from_corners closes them into 12
  // triangles, deciding winding per face so dragged corners cannot invert it).
  // ---------------------------------------------------------------------------
  let boxOn = false;
  let boxStage = 0;         // 0 idle, 1 footprint, 2 height
  let boxBase = null;       // [x, y, z] first corner
  let boxOpposite = null;   // [x, y, z] opposite footprint corner
  let boxHeight = 0;

  // Ground contact is a Y-only affair: the first click RESTS the shape on the
  // ground plane (or on geometry via ctrl-click) but X/Z stay exactly where
  // the cursor ray landed. An earlier build also quantised X/Z (and the
  // footprint/height/radius) to the 1 m grid cells — reverted live: the grid
  // jumps fought the artist when a blockout had to hug a torn edge.
  function boxCornersNow() {
    if (!boxBase || !boxOpposite) return null;
    const y = boxBase[1];
    const x0 = boxBase[0], z0 = boxBase[2];
    const x1 = boxOpposite[0], z1 = boxOpposite[2];
    const h = boxHeight;
    return [
      [x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1],
      [x0, y + h, z0], [x1, y + h, z0], [x1, y + h, z1], [x0, y + h, z1],
    ];
  }

  function refreshBoxPreview() {
    const corners = boxCornersNow();
    if (!corners) return;
    const EDGES = [[0, 1], [1, 2], [2, 3], [3, 0],
                   [4, 5], [5, 6], [6, 7], [7, 4],
                   [0, 4], [1, 5], [2, 6], [3, 7]];
    const flat = [];
    for (const [a, b] of EDGES) {
      if (boxStage === 1 && a >= 4) continue;      // no lid until extruding
      if (boxStage === 1 && b >= 4) continue;
      flat.push(...corners[a], ...corners[b]);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position",
      new THREE.BufferAttribute(new Float32Array(flat), 3));
    const seg = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({
      color: 0xffcc44, depthTest: false, transparent: true,
    }));
    seg.renderOrder = 200003;
    seg.userData.atlasHelper = true;
    drawGroup.add(seg);
  }

  // Where a cursor ray meets the horizontal plane the footprint lives on.
  function boxGroundPoint(mapped, baseY) {
    drawRaycaster.setFromCamera(new THREE.Vector2(mapped.x, mapped.y), camera);
    const o = camera.getWorldPosition(new THREE.Vector3());
    const d = drawRaycaster.ray.direction;
    return atlasIntersectRayWithPlane(
      [o.x, o.y, o.z], [d.x, d.y, d.z],
      { normal: [0, 1, 0], offset: baseY });
  }

  // Height comes from a VERTICAL plane through the footprint centre that faces
  // the camera — the standard way to read an up-drag off a 2D cursor.
  function boxHeightAt(mapped) {
    const corners = boxCornersNow();
    if (!corners) return 0;
    const cx = (boxBase[0] + boxOpposite[0]) / 2;
    const cz = (boxBase[2] + boxOpposite[2]) / 2;
    const fwd = camera.getWorldDirection(new THREE.Vector3());
    let n = [fwd.x, 0, fwd.z];
    const len = Math.hypot(n[0], n[2]);
    if (len < 1e-6) return boxHeight;               // looking straight down
    n = [n[0] / len, 0, n[2] / len];
    drawRaycaster.setFromCamera(new THREE.Vector2(mapped.x, mapped.y), camera);
    const o = camera.getWorldPosition(new THREE.Vector3());
    const d = drawRaycaster.ray.direction;
    const landed = atlasIntersectRayWithPlane(
      [o.x, o.y, o.z], [d.x, d.y, d.z],
      { normal: n, offset: n[0] * cx + n[2] * cz });
    if (!landed) return boxHeight;
    return Math.max(0.05, landed[1] - boxBase[1]);
  }

  function boxStatusLine() {
    if (boxStage === 0) {
      return (editSnap
        ? "▣ click the base corner — it rests on the ground (ctrl-click: on geometry)"
        : "▣ click the base corner — on the ground plane (grid snap off)")
        + "\nctrl-click to start on geometry instead (a roof, a ledge)";
    }
    if (boxStage === 1) return "▣ drag the footprint · click to fix it · Esc cancels";
    return `▣ drag the height (${boxHeight.toFixed(2)}m) · click or Enter finishes`;
  }

  function onBoxMove(ev) {
    if (!boxOn || boxStage === 0) return;
    const mapped = mappedFromEvent(ev);
    if (!mapped.inside) return;
    if (boxStage === 1) {
      const p = boxGroundPoint(mapped, boxBase[1]);
      if (p) boxOpposite = p;
    } else {
      boxHeight = boxHeightAt(mapped);
    }
    refreshDrawOverlay();
    drawHud(boxStatusLine());
  }

  function finishBox() {
    const corners = boxCornersNow();
    if (!corners || boxHeight <= 0.05) {
      drawHud("▣ give the box some height first");
      return;
    }
    drawnPolygons.push({
      id: `b${drawnPolygons.length + 1}`,
      label: `blockout box ${drawnPolygons.length + 1}`,
      enabled: true,
      kind: "box",
      points_world: corners,
    });
    boxStage = 0; boxBase = null; boxOpposite = null; boxHeight = 0;
    drawDirty = true;
    refreshDrawOverlay();
    drawHud(`▣ box added — ${drawnPolygons.length} shape(s) ready, click Apply`);
  }

  function onBoxClick(ev) {
    if (!boxOn) return;
    const mapped = mappedFromEvent(ev);
    if (!mapped.inside) return;

    if (boxStage === 0) {
      // Default: the solved ground plane (Y=0). Taking the base from geometry
      // by default put it wherever the relief mesh happened to be — and that
      // mesh droops below the grid at torn edges, so boxes began underground
      // (reported live). Ctrl/Cmd-click asks for the geometry height
      // explicitly, for a mass that starts on a roof or a ledge; X/Z still
      // snap to the grid, only the height comes from the surface.
      let base = null;
      if (ev.ctrlKey || ev.metaKey) {
        drawRaycaster.setFromCamera(new THREE.Vector2(mapped.x, mapped.y), camera);
        const hits = drawRaycaster.intersectObjects(drawTargets(), false);
        if (hits.length) {
          const hp = editSnap ? snapHitToEdge(hits[0], mapped) : hits[0].point;
          base = [hp.x, hp.y, hp.z];
        } else {
          drawHud("▣ ctrl-click found no geometry — starting on the ground");
        }
      }
      if (!base) {
        base = boxGroundPoint(mapped, 0);
        if (!base) { drawHud("▣ that ray misses the ground plane"); return; }
      }
      boxBase = base;   // Y already rests on ground/geometry; X/Z stay free
      boxOpposite = [...boxBase];
      boxHeight = 0;
      boxStage = 1;
    } else if (boxStage === 1) {
      boxStage = 2;
    } else {
      finishBox();
    }
    refreshDrawOverlay();
    drawHud(boxStatusLine());
  }

  const boxBtn = document.createElement("button");
  boxBtn.textContent = "▣ Box";
  boxBtn.title = "Blockout solid: click the base corner, drag the footprint, click, "
    + "drag the height, Enter to finish. The base always sits on the ground plane "
    + "(Y=0) unless you ctrl-click, which starts it at the height of the geometry "
    + "under the cursor instead; the footprint and height follow the cursor freely. "
    + "The 8 corners are then editable one by one in Edit, exactly like polygon "
    + "points — raise it onto a roof there if that is what you want.";
  boxBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  boxBtn.onclick = () => {
    boxOn = !boxOn;
    boxBtn.style.background = boxOn ? "#3a2a1a" : "#2a2a2a";
    boxBtn.style.color = boxOn ? "#fc8" : "#ddd";
    if (boxOn) {
      if (drawOn) drawBtn.onclick();
      if (editOn) editBtn.onclick();
      if (sphereOn) sphereBtn.onclick();
      if (quadOn) quadBtn.onclick();
      if (extrudeOn) extrudeBtn.onclick();
      if (wandOn) wandBtn.onclick();
    }
    controls.setEnabled(!boxOn);
    canvas.style.cursor = boxOn ? "crosshair" : "grab";
    boxStage = 0; boxBase = null; boxOpposite = null; boxHeight = 0;
    refreshDrawOverlay();
    drawHud(boxOn ? boxStatusLine() : "");
  };

  // ---------------------------------------------------------------------------
  // Sphere — two stages: click where it touches down, drag the radius.
  //
  // Stored as two CONTROL points (centre, and a point on the surface) rather
  // than its mesh, so it has exactly two Edit handles and needs no
  // sphere-specific editing code: drag the centre to move it, drag the surface
  // handle to resize. Python rebuilds the mesh from those two
  // (core/polygon_planes.sphere_mesh_from_control_points).
  //
  // Like a box it TOUCHES DOWN on the ground rather than being centred on the
  // click — a blockout mass sits on the ground, so the contact point is what
  // the artist is actually pointing at.
  // ---------------------------------------------------------------------------
  let sphereOn = false;
  let sphereStage = 0;      // 0 idle, 1 dragging the radius
  let sphereContact = null; // [x, y, z] where it rests
  let sphereRadius = 0;

  function sphereCentreNow() {
    if (!sphereContact) return null;
    return [sphereContact[0], sphereContact[1] + sphereRadius, sphereContact[2]];
  }

  function sphereControlPoints() {
    const c = sphereCentreNow();
    if (!c || sphereRadius <= 0) return null;
    return [c, [c[0] + sphereRadius, c[1], c[2]]];
  }

  // Three great circles read as a sphere in wireframe without shipping a mesh
  // to the overlay.
  function sphereWireframe(centre, radius, color) {
    const STEPS = 48;
    const flat = [];
    const axes = [[0, 1], [0, 2], [1, 2]];
    for (const [a, b] of axes) {
      for (let i = 0; i < STEPS; i += 1) {
        for (const t of [i, i + 1]) {
          const ang = (2 * Math.PI * t) / STEPS;
          const p = [centre[0], centre[1], centre[2]];
          p[a] += radius * Math.cos(ang);
          p[b] += radius * Math.sin(ang);
          flat.push(p[0], p[1], p[2]);
        }
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position",
      new THREE.BufferAttribute(new Float32Array(flat), 3));
    const seg = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({
      color, depthTest: false, transparent: true,
    }));
    seg.renderOrder = 200003;
    seg.userData.atlasHelper = true;
    drawGroup.add(seg);
  }

  function sphereStatusLine() {
    if (sphereStage === 0) {
      return "● click where the sphere touches down (ctrl-click for geometry)";
    }
    return `● drag the radius (${sphereRadius.toFixed(2)}m) · click or Enter finishes`;
  }

  function onSphereMove(ev) {
    if (!sphereOn || sphereStage !== 1) return;
    const mapped = mappedFromEvent(ev);
    if (!mapped.inside) return;
    const p = boxGroundPoint(mapped, sphereContact[1]);
    if (!p) return;
    const r = Math.hypot(p[0] - sphereContact[0], p[2] - sphereContact[2]);
    sphereRadius = Math.max(0.05, r);
    refreshDrawOverlay();
    drawHud(sphereStatusLine());
  }

  function finishSphere() {
    const control = sphereControlPoints();
    if (!control || sphereRadius <= 0.05) {
      drawHud("● give the sphere a radius first");
      return;
    }
    drawnPolygons.push({
      id: `s${drawnPolygons.length + 1}`,
      label: `blockout sphere ${drawnPolygons.length + 1}`,
      enabled: true,
      kind: "sphere",
      points_world: control,
    });
    sphereStage = 0; sphereContact = null; sphereRadius = 0;
    drawDirty = true;
    refreshDrawOverlay();
    drawHud(`● sphere added — ${drawnPolygons.length} shape(s) ready, click Apply`);
  }

  function onSphereClick(ev) {
    if (!sphereOn) return;
    const mapped = mappedFromEvent(ev);
    if (!mapped.inside) return;

    if (sphereStage === 0) {
      let contact = null;
      if (ev.ctrlKey || ev.metaKey) {
        drawRaycaster.setFromCamera(new THREE.Vector2(mapped.x, mapped.y), camera);
        const hits = drawRaycaster.intersectObjects(drawTargets(), false);
        if (hits.length) {
          const hp = editSnap ? snapHitToEdge(hits[0], mapped) : hits[0].point;
          contact = [hp.x, hp.y, hp.z];
        } else {
          drawHud("● ctrl-click found no geometry — resting on the ground");
        }
      }
      if (!contact) {
        contact = boxGroundPoint(mapped, 0);
        if (!contact) { drawHud("● that ray misses the ground plane"); return; }
      }
      sphereContact = contact;
      sphereRadius = 0;
      sphereStage = 1;
    } else {
      finishSphere();
    }
    refreshDrawOverlay();
    drawHud(sphereStatusLine());
  }

  const sphereBtn = document.createElement("button");
  sphereBtn.textContent = "● Sphere";
  sphereBtn.title = "Blockout sphere: click where it touches down (ctrl-click to rest it "
    + "on geometry), drag the radius, Enter to finish. It is stored as a centre and a "
    + "surface point, so Edit gives it exactly two handles — drag the centre to move "
    + "it, the outer handle to resize.";
  sphereBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  sphereBtn.onclick = () => {
    sphereOn = !sphereOn;
    sphereBtn.style.background = sphereOn ? "#2a1a3a" : "#2a2a2a";
    sphereBtn.style.color = sphereOn ? "#c8f" : "#ddd";
    if (sphereOn) {
      if (drawOn) drawBtn.onclick();
      if (editOn) editBtn.onclick();
      if (boxOn) boxBtn.onclick();
      if (quadOn) quadBtn.onclick();
      if (extrudeOn) extrudeBtn.onclick();
      if (wandOn) wandBtn.onclick();
    }
    controls.setEnabled(!sphereOn);
    canvas.style.cursor = sphereOn ? "crosshair" : "grab";
    sphereStage = 0; sphereContact = null; sphereRadius = 0;
    refreshDrawOverlay();
    drawHud(sphereOn ? sphereStatusLine() : "");
  };

  // ---------------------------------------------------------------------------
  // ⬜ Quad — Maya-style live quad draw, for filling ENCLOSED tears fast.
  //
  // The rail's tear-filler: 4 clicks make the first quad (any click order —
  // the points are re-ordered into the non-crossing loop by min perimeter,
  // and each click edge/vertex-snaps to the tear rim exactly like ✏️ Draw),
  // then each FOLLOWING quad costs 2 clicks: the first of them seeds the new
  // quad from the nearest edge of the previous one, so strips grow in any
  // direction like Maya's quad-draw. Esc/Enter ends the strip; Backspace pops
  // the last point, then the last quad.
  //
  // Deliberately NOT a new kind: every committed quad is an ordinary
  // kind-less 4-point polygon ({points_world, plane}) — meshing, ✎ Edit,
  // the gizmo, 🗑 and ✅ Apply all work on it with zero new code paths.
  // Adjacent quads COPY their shared edge's values (JSON shapes cannot share
  // references); they coincide exactly at commit, and edge-snap re-meets them
  // if one is edited later.
  // ---------------------------------------------------------------------------
  let quadOn = false;
  let quadPoints = [];      // the quad being built (seeded with 2 pts mid-strip)
  let quadPrev = null;      // points_world of the last committed quad, for seeding
  let quadCount = 0;

  const quadDist3 = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
  function quadPerimeter(pts) {
    let s = 0;
    for (let i = 0; i < pts.length; i += 1) s += quadDist3(pts[i], pts[(i + 1) % pts.length]);
    return s;
  }
  // The simple (non-self-intersecting) loop through 4 points is the cyclic
  // order with the smallest perimeter — a bowtie always pays for its crossing
  // with two longer diagonally-swapped edges. lockFirstEdge keeps a seeded
  // strip edge adjacent (only the last two points may swap).
  function orderQuad(pts, lockFirstEdge) {
    const orders = lockFirstEdge
      ? [[0, 1, 2, 3], [0, 1, 3, 2]]
      : [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3]];
    let best = null, bestP = Infinity;
    for (const o of orders) {
      const cand = o.map((i) => pts[i]);
      const p = quadPerimeter(cand);
      if (p < bestP) { bestP = p; best = cand; }
    }
    return best;
  }
  // Nearest edge of the previous quad to a clicked point — the edge the new
  // quad grows from, so a strip can turn any direction mid-flow.
  function nearestQuadEdge(pts, p) {
    let best = null, bestD = Infinity;
    for (let i = 0; i < 4; i += 1) {
      const a = new THREE.Vector3(...pts[i]);
      const b = new THREE.Vector3(...pts[(i + 1) % 4]);
      const on = closestPointOnSegment(new THREE.Vector3(...p), a, b);
      const d = on.distanceTo(new THREE.Vector3(...p));
      if (d < bestD) { bestD = d; best = [pts[i], pts[(i + 1) % 4]]; }
    }
    return best;
  }

  function quadStatusLine() {
    const seeded = quadPrev && quadPoints.length >= 2;
    const need = 4 - quadPoints.length;
    if (!quadPoints.length && !quadPrev) {
      return "⬜ click 4 points around a tear (snap grabs the rim) — the 4th commits";
    }
    if (!quadPoints.length && quadPrev) {
      return "⬜ strip: click near an edge of the last quad to grow from it · Esc/Enter ends";
    }
    return `⬜ ${quadPoints.length} point(s)${seeded ? " (2 seeded from the last quad)" : ""}`
      + ` — ${need} more commit${need === 1 ? "s" : ""} the quad`
      + " · Backspace undoes · Esc/Enter ends";
  }

  function commitQuad() {
    const seeded = !!quadPrev;
    const pts = orderQuad(quadPoints.map((p) => [...p]), seeded);
    const plane = atlasEstablishPlaneFromHits(pts);
    if (!plane) {
      drawHud("⬜ those 4 points are degenerate — Backspace and re-click");
      return;
    }
    quadCount += 1;
    drawnPolygons.push({
      id: `q${quadCount}`,
      label: `quad ${quadCount}`,
      enabled: true,
      points_world: pts,
      plane: { normal: plane.normal, offset: plane.offset },
      established_from: { hits: 4, rule: "quad_draw" },
    });
    quadPrev = pts;
    quadPoints = [];
    drawDirty = true;
    refreshDrawOverlay();
    drawHud("⬜ quad committed — click to grow the strip, Esc/Enter to end, ✅ Apply builds");
  }

  function onQuadClick(ev) {
    if (!quadOn) return;
    const mapped = mappedFromEvent(ev);
    if (!mapped.inside) {
      drawHud("⬜ that click is in the letterbox bar, not the image");
      return;
    }
    // Ctrl-click removes the in-progress point under the cursor (same NDC
    // tolerance as ✏️ Draw).
    if (ev.ctrlKey || ev.metaKey) {
      const tol = 2 * 14 / Math.max(canvas.height / previewScale, 1);
      let best = -1, bestD = Infinity;
      for (let i = 0; i < quadPoints.length; i += 1) {
        const q = new THREE.Vector3(...quadPoints[i]).project(camera);
        const d = Math.hypot(q.x - mapped.x, q.y - mapped.y);
        if (d < bestD) { bestD = d; best = i; }
      }
      if (best >= 0 && bestD <= tol) quadPoints.splice(best, 1);
      refreshDrawOverlay();
      drawHud(quadStatusLine());
      return;
    }

    const ndc = new THREE.Vector2(mapped.x, mapped.y);
    drawRaycaster.setFromCamera(ndc, camera);
    let landed = null;
    // A click on an EXISTING drawn corner takes its exact coordinates first —
    // new quads are born already welded to their neighbours instead of
    // nearly-meeting them.
    if (editSnap && !ev.shiftKey) {
      const t = findWeldTarget(mapped);
      if (t) landed = [...t.point];
    }
    const hits = landed ? [] : drawRaycaster.intersectObjects(drawTargets(), false);
    if (landed) {
      // welded to a drawn vertex — nothing more to resolve
    } else if (hits.length) {
      const p = (editSnap && !ev.shiftKey) ? snapHitToEdge(hits[0], mapped) : hits[0].point;
      landed = [p.x, p.y, p.z];
    } else {
      // Off-mesh (mid-tear, where the hole IS): land on the plane fit from the
      // points placed so far — mid-tear clicks stay coplanar with the rim.
      const fitFrom = quadPoints.length >= 2 ? quadPoints : quadPrev;
      const plane = fitFrom ? atlasEstablishPlaneFromHits(fitFrom) : null;
      const o = camera.getWorldPosition(new THREE.Vector3());
      const d = drawRaycaster.ray.direction;
      if (plane) {
        landed = atlasIntersectRayWithPlane([o.x, o.y, o.z], [d.x, d.y, d.z], plane);
      }
      // Grazing-ray guard (found live): orbiting between clicks can leave the
      // strip's plane nearly edge-on to the camera, and the intersection then
      // lands absurdly far away — a sliver quad shooting across the scene.
      // A landing further from the existing points than 4× their own spread
      // is rejected and re-landed on a CAMERA-FACING plane through the last
      // point instead, which is always well-conditioned.
      if (fitFrom && fitFrom.length) {
        let spread = 0;
        for (let i = 1; i < fitFrom.length; i += 1) {
          spread = Math.max(spread, quadDist3(fitFrom[i - 1], fitFrom[i]));
        }
        const limit = 4 * Math.max(spread, 1);
        const near = (p) => fitFrom.some((q) => quadDist3(p, q) <= limit);
        if (!landed || !near(landed)) {
          const anchor = fitFrom[fitFrom.length - 1];
          const fwd = camera.getWorldDirection(new THREE.Vector3());
          const n = [fwd.x, fwd.y, fwd.z];
          const facing = {
            normal: n,
            offset: n[0] * anchor[0] + n[1] * anchor[1] + n[2] * anchor[2],
          };
          landed = atlasIntersectRayWithPlane([o.x, o.y, o.z], [d.x, d.y, d.z], facing);
        }
      }
      if (!landed) {
        drawHud("⬜ nothing hit — the first clicks must land on geometry (the tear rim)");
        return;
      }
    }

    // First click of a strip continuation: seed the new quad with the nearest
    // edge of the previous one, so this click is already point 3 of 4.
    if (!quadPoints.length && quadPrev) {
      const edge = nearestQuadEdge(quadPrev, landed);
      quadPoints = [[...edge[1]], [...edge[0]]];
    }
    quadPoints.push(landed);
    if (quadPoints.length >= 4) {
      commitQuad();
      return;
    }
    refreshDrawOverlay();
    drawHud(quadStatusLine());
  }

  const quadBtn = document.createElement("button");
  quadBtn.textContent = "⬜ Quad";
  quadBtn.title = "Maya-style quad draw for filling enclosed tears: click 4 points "
    + "around the hole (snap grabs the rim edges; any click order) and the 4th "
    + "commits the quad. After that each quad costs 2 clicks — the strip grows "
    + "from whichever edge of the last quad you click nearest. Esc/Enter ends "
    + "the strip, Backspace undoes, ✅ Apply builds. Use ✏️ Draw / ▣ Box for "
    + "boundary edges and large-scale mass instead.";
  quadBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  quadBtn.onclick = () => {
    quadOn = !quadOn;
    quadBtn.style.background = quadOn ? "#2a2a1a" : "#2a2a2a";
    quadBtn.style.color = quadOn ? "#ff8" : "#ddd";
    if (quadOn) {
      if (drawOn) drawBtn.onclick();
      if (editOn) editBtn.onclick();
      if (boxOn) boxBtn.onclick();
      if (sphereOn) sphereBtn.onclick();
      if (extrudeOn) extrudeBtn.onclick();
      if (wandOn) wandBtn.onclick();
    }
    controls.setEnabled(!quadOn);
    canvas.style.cursor = quadOn ? "crosshair" : "grab";
    quadPoints = [];
    quadPrev = null;
    refreshDrawOverlay();
    drawHud(quadOn ? quadStatusLine() : "");
  };

  // ---------------------------------------------------------------------------
  // ➬ Extrude — pull a NEW quad out of any existing drawn edge, Maya-style.
  //
  // The complement to ⬜ Quad: instead of clicking 4 fresh points, grab an edge
  // of a shape that already exists (quad, drawn n-gon, or box wire) and DRAG —
  // the edge's two endpoints are copied and follow the cursor on a
  // camera-facing plane through the grab point (always well-conditioned, no
  // grazing-plane blowups), release commits the quad. Each committed extrusion
  // is an ordinary kind-less 4-point polygon like ⬜ Quad's output, so Edit /
  // weld / 🗑 / Apply all just work; the source edge's values are COPIED, and
  // the red weld snap re-closes the seam if either side is edited later.
  // ---------------------------------------------------------------------------
  let extrudeOn = false;
  let extrudeDrag = null;   // { a, b, grab, delta } while pulling
  let extrudeCount = 0;

  function ndcSegDist(m, a, b) {
    const abx = b.x - a.x, aby = b.y - a.y;
    const len2 = abx * abx + aby * aby;
    let t = len2 > 1e-12 ? ((m.x - a.x) * abx + (m.y - a.y) * aby) / len2 : 0;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(m.x - (a.x + t * abx), m.y - (a.y + t * aby));
  }

  // Nearest edge of ANY drawn shape to the cursor, in screen space (constant
  // ~14 px tolerance like every other pick). Spheres have no edges to pull.
  function nearestDrawnEdge(mapped) {
    const tol = 2 * 14 / Math.max(canvas.height / previewScale, 1);
    const EX_BOX_EDGES = [[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]];
    let best = null, bestD = Infinity;
    for (const poly of drawnPolygons) {
      if (poly.kind === "sphere") continue;
      const pts = poly.points_world;
      const pairs = (poly.kind === "box" && pts.length === 8)
        ? EX_BOX_EDGES
        : pts.map((_, i) => [i, (i + 1) % pts.length]);
      for (const [i0, i1] of pairs) {
        const a = projectToNdc(pts[i0]);
        const b = projectToNdc(pts[i1]);
        const d = ndcSegDist(mapped, a, b);
        if (d < bestD) { bestD = d; best = { a: pts[i0], b: pts[i1] }; }
      }
    }
    return bestD <= tol ? best : null;
  }

  function onExtrudePointerDown(ev) {
    if (!extrudeOn) return;
    const mapped = mappedFromEvent(ev);
    if (!mapped.inside) return;
    const edge = nearestDrawnEdge(mapped);
    if (!edge) {
      drawHud("➬ no edge under the cursor — grab an edge of a drawn shape");
      return;
    }
    const grab = [(edge.a[0] + edge.b[0]) / 2,
                  (edge.a[1] + edge.b[1]) / 2,
                  (edge.a[2] + edge.b[2]) / 2];
    extrudeDrag = { a: [...edge.a], b: [...edge.b], grab, delta: null };
    canvas.setPointerCapture?.(ev.pointerId);
    drawHud("➬ pull the new edge out — release to commit");
    ev.stopPropagation();
    ev.preventDefault();
  }

  function onExtrudePointerMove(ev) {
    if (!extrudeDrag) return;
    const mapped = mappedFromEvent(ev);
    if (!mapped.inside) return;
    drawRaycaster.setFromCamera(new THREE.Vector2(mapped.x, mapped.y), camera);
    const o = camera.getWorldPosition(new THREE.Vector3());
    const d = drawRaycaster.ray.direction;
    const fwd = camera.getWorldDirection(new THREE.Vector3());
    const n = [fwd.x, fwd.y, fwd.z];
    const g = extrudeDrag.grab;
    const facing = { normal: n, offset: n[0] * g[0] + n[1] * g[1] + n[2] * g[2] };
    const landed = atlasIntersectRayWithPlane([o.x, o.y, o.z], [d.x, d.y, d.z], facing);
    if (!landed) return;
    extrudeDrag.delta = [landed[0] - g[0], landed[1] - g[1], landed[2] - g[2]];
    refreshDrawOverlay();
    ev.stopPropagation();
  }

  function onExtrudePointerUp(ev) {
    if (!extrudeDrag) return;
    const { a, b, delta } = extrudeDrag;
    extrudeDrag = null;
    canvas.releasePointerCapture?.(ev.pointerId);
    const len = delta ? Math.hypot(delta[0], delta[1], delta[2]) : 0;
    if (len < 0.02) {
      refreshDrawOverlay();
      drawHud("➬ barely moved — drag further to pull a quad out");
      ev.stopPropagation();
      return;
    }
    const pts = orderQuad([
      [...a], [...b],
      [b[0] + delta[0], b[1] + delta[1], b[2] + delta[2]],
      [a[0] + delta[0], a[1] + delta[1], a[2] + delta[2]],
    ], true);
    const plane = atlasEstablishPlaneFromHits(pts);
    if (!plane) {
      refreshDrawOverlay();
      drawHud("➬ that pull is degenerate — try a different direction");
      ev.stopPropagation();
      return;
    }
    extrudeCount += 1;
    drawnPolygons.push({
      id: `x${extrudeCount}`,
      label: `extrude ${extrudeCount}`,
      enabled: true,
      points_world: pts,
      plane: { normal: plane.normal, offset: plane.offset },
      established_from: { hits: 4, rule: "edge_extrude" },
    });
    drawDirty = true;
    refreshDrawOverlay();
    drawHud("➬ quad extruded — pull another edge, Enter exits, ✅ Apply builds");
    ev.stopPropagation();
  }

  const extrudeBtn = document.createElement("button");
  extrudeBtn.textContent = "➬ Extrude";
  extrudeBtn.title = "Pull a new quad out of an existing edge: grab any edge of a "
    + "drawn shape (quad, n-gon or box) and drag — the edge is copied and follows "
    + "the cursor, release commits the quad. Perfect for extending a strip's "
    + "bottom/side edges over a tear instead of clicking new quads. Enter exits, "
    + "✅ Apply builds; the red weld snap in ✎ Edit re-closes any seam later.";
  extrudeBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  extrudeBtn.onclick = () => {
    extrudeOn = !extrudeOn;
    extrudeBtn.style.background = extrudeOn ? "#2a221a" : "#2a2a2a";
    extrudeBtn.style.color = extrudeOn ? "#fc6" : "#ddd";
    if (extrudeOn) {
      if (drawOn) drawBtn.onclick();
      if (editOn) editBtn.onclick();
      if (boxOn) boxBtn.onclick();
      if (sphereOn) sphereBtn.onclick();
      if (quadOn) quadBtn.onclick();
      if (wandOn) wandBtn.onclick();
    }
    controls.setEnabled(!extrudeOn);
    canvas.style.cursor = extrudeOn ? "crosshair" : "grab";
    extrudeDrag = null;
    refreshDrawOverlay();
    drawHud(extrudeOn
      ? "➬ grab an edge of a drawn shape and pull a quad out of it"
      : "");
  };

  // ---------------------------------------------------------------------------
  // 🪄 Wand — one-click hole fill: click INSIDE any enclosed tear and its
  // boundary rim becomes the fill.
  //
  // Boundary loops are extracted from each drawTargets() mesh (edges owned by
  // exactly ONE triangle are hole rims; vertices deduped by position so seams
  // in the buffer don't break loops), cached per geometry. A click picks the
  // INNERMOST loop containing the cursor (smallest projected area — the mesh's
  // outer border always contains the click too, but never wins, and a hard
  // area cap rejects it outright). The fill commits the loop's EXACT rim
  // vertices as an ordinary drawn polygon — born welded to the mesh at every
  // vertex, and Edit / 🗑 / ✅ Apply work on it like anything else. Degenerate
  // rims are safe: the backend wraps each polygon in try/except and reports
  // "skipped(...)" instead of failing the node.
  // ---------------------------------------------------------------------------
  let wandOn = false;
  let wandCount = 0;
  const WAND_MAX_RIM = 600;      // rim verts; bigger holes → use Quad/Draw
  const WAND_MAX_NDC_AREA = 0.8; // full viewport is 4.0 — outer borders lose
  // Bay mouth tolerance is WORLD-space, relative to the rim's own edge
  // length: a mouth up to this many median rim edges bridges, wider refuses.
  // Tears jag at relief-grid resolution, so "median rim edge" ≈ one grid
  // cell — this stays stable across zoom (the old NDC tolerance shrank and
  // grew with the camera) and across scene scale (metric vs assumed).
  const WAND_GAP_EDGE_FACTOR = 8;
  const WAND_BAY_LOCAL_R = 0.7;  // only rim verts this close to the click count (NDC — interaction locality)

  function medianRimEdge(path) {
    const n = Math.min(path.length - 1, 200);
    if (n < 1) return 0;
    const step = Math.max(1, Math.floor((path.length - 1) / n));
    const lens = [];
    for (let i = 0; i + step < path.length; i += step) {
      lens.push(quadDist3(path[i], path[i + step]) / step);
    }
    lens.sort((x, y) => x - y);
    return lens[Math.floor(lens.length / 2)] || 0;
  }

  // A closed boundary walk that returns through a vertex it already used is a
  // figure-8 — a PINCHED rim, which is what the walk produces where two tears
  // of a torn relief mesh meet at a single shared vertex. Packaged whole, the
  // backend rightly refuses it as self-intersecting ("vertex N repeats vertex
  // M", docs/dev/wand_self_intersecting_rims.md). Each lobe is a fillable hole
  // in its own right, so split at every repeated id into simple sub-loops —
  // the innermost-containing-loop pick then fills whichever lobe was clicked.
  // A rim that merely TOUCHES itself splits into fills the same way. No-op on
  // healthy rims; sub-loops under 3 vertices are not polygons and are dropped.
  function splitLoopAtRepeats(path) {
    const loops = [];
    const stack = [];
    const at = new Map();         // id -> index in stack
    for (const id of path) {
      const j = at.get(id);
      if (j === undefined) {
        at.set(id, stack.length);
        stack.push(id);
        continue;
      }
      // The cycle from the earlier occurrence back to here is one lobe; the
      // pinch vertex stays on the stack — the walk continues through it.
      const lobe = stack.slice(j);
      for (const v of lobe.slice(1)) at.delete(v);
      stack.length = j + 1;
      if (lobe.length >= 3) loops.push(lobe);
    }
    if (stack.length >= 3) loops.push(stack);
    return loops;
  }

  function meshBoundaryLoops(mesh) {
    const geo = mesh.geometry;
    if (!geo?.attributes?.position) return { loops: [], chains: [] };
    const cached = mesh.userData._atlasWandLoops;
    if (cached && cached.uuid === geo.uuid) return cached;
    const pos = geo.attributes.position;
    const count = pos.count;
    // Dedup vertices by rounded position — torn meshes often duplicate rim
    // vertices across triangles, which would break edge counting.
    const canon = new Map();      // posKey -> canonical id
    const ids = new Array(count);
    const canonPos = [];
    for (let i = 0; i < count; i += 1) {
      const k = pos.getX(i).toFixed(5) + "," + pos.getY(i).toFixed(5) + ","
        + pos.getZ(i).toFixed(5);
      let id = canon.get(k);
      if (id === undefined) {
        id = canonPos.length;
        canon.set(k, id);
        canonPos.push([pos.getX(i), pos.getY(i), pos.getZ(i)]);
      }
      ids[i] = id;
    }
    const idx = geo.index ? geo.index.array : null;
    const triCount = (idx ? idx.length : count) / 3;
    const edges = new Map();      // "a_b" (a<b) -> count
    for (let t = 0; t < triCount; t += 1) {
      const a = ids[idx ? idx[t * 3] : t * 3];
      const b = ids[idx ? idx[t * 3 + 1] : t * 3 + 1];
      const c = ids[idx ? idx[t * 3 + 2] : t * 3 + 2];
      for (const [u, v] of [[a, b], [b, c], [c, a]]) {
        if (u === v) continue;
        const key = u < v ? u + "_" + v : v + "_" + u;
        edges.set(key, (edges.get(key) || 0) + 1);
      }
    }
    const adj = new Map();        // boundary adjacency: id -> [ids]
    for (const [key, n] of edges) {
      if (n !== 1) continue;
      const [u, v] = key.split("_").map(Number);
      if (!adj.has(u)) adj.set(u, []);
      if (!adj.has(v)) adj.set(v, []);
      adj.get(u).push(v);
      adj.get(v).push(u);
    }
    // Walk the boundary graph. CLOSED loops are enclosed holes; walks that
    // dead-end (junction vertices where tears pinch the border, degree > 2)
    // are kept as OPEN CHAINS — the bay fallback needs exactly those rim
    // runs, and an early build that discarded them (and truncated any walk
    // over the rim cap) is why boundary bays never filled (found live).
    // No length cap here: the cap applies to what a FILL may use, not to
    // extraction — capping the walk silently threw away the outer border.
    const ekey = (u, v) => (u < v ? u + "_" + v : v + "_" + u);
    const usedEdge = new Set();
    const loops = [], chains = [];
    for (const start of adj.keys()) {
      for (const first of adj.get(start)) {
        const k0 = ekey(start, first);
        if (usedEdge.has(k0)) continue;
        usedEdge.add(k0);
        const path = [start, first];
        let prev = start, cur = first, closed = false;
        while (path.length < 100000) {
          const next = (adj.get(cur) || []).find(
            (n) => n !== prev && !usedEdge.has(ekey(cur, n)));
          if (next === undefined) break;
          usedEdge.add(ekey(cur, next));
          prev = cur;
          cur = next;
          if (cur === start) { closed = true; break; }
          path.push(cur);
        }
        if (path.length < 3) continue;
        const toWorld = (sub) => sub.map((id) => {
          const p = new THREE.Vector3(...canonPos[id])
            .applyMatrix4(mesh.matrixWorld);
          return [p.x, p.y, p.z];
        });
        if (closed) {
          // Split pinched walks HERE, on exact integer ids — after world
          // mapping it would take an epsilon to see the repeat.
          for (const sub of splitLoopAtRepeats(path)) loops.push(toWorld(sub));
        } else {
          // Open chains stay whole: the bay fallback needs the full rim run.
          chains.push(toWorld(path));
        }
      }
    }
    mesh.userData._atlasWandLoops = { uuid: geo.uuid, loops, chains };
    return mesh.userData._atlasWandLoops;
  }

  function ndcLoopArea(pts) {
    let area = 0;
    const proj = pts.map(projectToNdc);
    for (let i = 0, j = proj.length - 1; i < proj.length; j = i, i += 1) {
      area += (proj[j].x + proj[i].x) * (proj[j].y - proj[i].y);
    }
    return Math.abs(area / 2);
  }

  // Fallback for holes that OPEN onto the mesh's outer border (no closed
  // interior loop): find two rim vertices near the click whose projections
  // nearly touch — the bay's mouth — bridge them, and fill the enclosed arc.
  // Pairs are only searched among rim verts local to the click, which keeps
  // the O(n²) pair scan tiny even on a many-thousand-vertex border loop.
  function wandBayFromPath(path, mapped, closed) {
    const n = path.length;
    if (n < 6) return null;
    const proj = path.map(projectToNdc);
    const local = [];
    for (let i = 0; i < n; i += 1) {
      if (Math.hypot(proj[i].x - mapped.x, proj[i].y - mapped.y) <= WAND_BAY_LOCAL_R) {
        local.push(i);
      }
    }
    const gapTol = WAND_GAP_EDGE_FACTOR * medianRimEdge(path);
    if (!gapTol) return null;
    let best = null, bestArea = Infinity;
    for (let ai = 0; ai < local.length; ai += 1) {
      for (let bi = ai + 1; bi < local.length; bi += 1) {
        const a = local[ai], b = local[bi];
        const gap = quadDist3(path[a], path[b]);
        if (gap > gapTol) continue;                     // mouth too wide
        const arcs = [];
        if (closed) {
          const stride = Math.min(b - a, n - (b - a));
          if (stride < 3) continue;                     // adjacent rim verts
          // Two arcs close a..b; the bay is whichever encloses the click.
          const fwd = [], back = [];
          for (let i = a; i !== b; i = (i + 1) % n) fwd.push(path[i]);
          fwd.push(path[b]);
          for (let i = b; i !== a; i = (i + 1) % n) back.push(path[i]);
          back.push(path[a]);
          arcs.push(fwd, back);
        } else {
          if (b - a < 3) continue;
          arcs.push(path.slice(a, b + 1));              // sub-run + bridge
        }
        for (const arc of arcs) {
          if (arc.length < 3 || arc.length > WAND_MAX_RIM) continue;
          if (!pointInScreenPolygon(mapped, arc)) continue;
          const area = ndcLoopArea(arc);
          if (area > WAND_MAX_NDC_AREA) continue;
          if (area < bestArea) { bestArea = area; best = arc; }
        }
      }
    }
    return best ? { arc: best, area: bestArea } : null;
  }

  function onWandClick(ev) {
    if (!wandOn) return;
    // A thrown exception would silently kill every later click — surface it.
    try {
      onWandClickInner(ev);
    } catch (err) {
      console.error("[atlas wand]", err);
      drawHud("🪄 internal error — see the browser console: " + err.message);
    }
  }

  function onWandClickInner(ev) {
    const mapped = mappedFromEvent(ev);
    if (!mapped.inside) return;
    // A rim already claimed by an earlier wand fill is skipped, so re-clicking
    // a filled hole doesn't stack duplicates (the derived mesh still reports
    // the boundary — the fill is a separate polygon on top of it).
    // Two vertices, not one: sibling lobes of a split pinched rim all START
    // at the shared pinch vertex, so equal-length lobes match on the first
    // point alone and the second lobe would wrongly read as already filled.
    const alreadyFilled = (pts) => drawnPolygons.some((p) =>
      (p.established_from?.rule === "wand_fill"
       || p.established_from?.rule === "wand_bay_fill")
      && p.points_world.length === pts.length
      && quadDist3(p.points_world[0], pts[0]) < 1e-6
      && quadDist3(p.points_world[1], pts[1]) < 1e-6);
    let best = null, bestArea = Infinity, bestRule = "wand_fill";
    const allPaths = [];    // [points, closed] — loops AND open chains
    for (const mesh of drawTargets()) {
      const { loops, chains } = meshBoundaryLoops(mesh);
      for (const chain of chains) allPaths.push([chain, false]);
      for (const loop of loops) {
        allPaths.push([loop, true]);
        if (loop.length > WAND_MAX_RIM) continue;
        if (alreadyFilled(loop)) continue;
        if (!pointInScreenPolygon(mapped, loop)) continue;
        const area = ndcLoopArea(loop);
        if (area > WAND_MAX_NDC_AREA) continue;   // outer border / huge rim
        if (area < bestArea) { bestArea = area; best = loop; }
      }
    }
    if (!best) {
      // No closed interior loop — try the boundary-bay fallback on every
      // path: closed loops the first pass rejected AND the open chains the
      // border's junction points break its rim into.
      for (const [path, closed] of allPaths) {
        const bay = wandBayFromPath(path, mapped, closed);
        if (bay && !alreadyFilled(bay.arc) && bay.area < bestArea) {
          bestArea = bay.area;
          best = bay.arc;
          bestRule = "wand_bay_fill";
        }
      }
    }
    if (!best) {
      // Diagnostic HUD: say WHY nothing matched, so tolerance tuning has
      // numbers instead of guesses.
      let loops = 0, chains = 0, localVerts = 0;
      let minGap = Infinity, minGapTol = 0;
      for (const [path, closed] of allPaths) {
        if (closed) loops += 1; else chains += 1;
        const proj = path.map(projectToNdc);
        const local = [];
        for (let i = 0; i < path.length; i += 1) {
          if (Math.hypot(proj[i].x - mapped.x, proj[i].y - mapped.y)
              <= WAND_BAY_LOCAL_R) local.push(i);
        }
        localVerts += local.length;
        const tol = WAND_GAP_EDGE_FACTOR * medianRimEdge(path);
        for (let ai = 0; ai < local.length; ai += 1) {
          for (let bi = ai + 1; bi < local.length; bi += 1) {
            const a = local[ai], b = local[bi];
            if (Math.abs(b - a) < 3) continue;
            const g = quadDist3(path[a], path[b]);
            if (g < minGap) { minGap = g; minGapTol = tol; }
          }
        }
      }
      console.warn("[atlas wand] no fill:",
                   { loops, chains, localVerts, minGap, minGapTol });
      drawHud("🪄 no fill — " + loops + " loop(s), " + chains + " chain(s), "
              + localVerts + " rim vert(s) near click, closest mouth "
              + (minGap === Infinity ? "n/a"
                 : minGap.toFixed(2) + "m (limit " + minGapTol.toFixed(2) + "m)"));
      return;
    }
    const pts = best.map((p) => [...p]);
    const plane = atlasEstablishPlaneFromHits(pts);
    if (!plane) {
      drawHud("🪄 that rim is degenerate — fill it with ⬜ Quad instead");
      return;
    }
    wandCount += 1;
    drawnPolygons.push({
      id: `w${wandCount}`,
      label: `wand fill ${wandCount}`,
      enabled: true,
      points_world: pts,
      plane: { normal: plane.normal, offset: plane.offset },
      established_from: { hits: pts.length, rule: bestRule },
    });
    drawDirty = true;
    refreshDrawOverlay();
    drawHud(`🪄 ${bestRule === "wand_bay_fill" ? "boundary notch" : "hole"} `
            + `filled (${pts.length} rim verts, born welded) — keep clicking, `
            + "Enter exits, ✅ Apply builds");
  }

  const wandBtn = document.createElement("button");
  wandBtn.textContent = "🪄 Wand";
  wandBtn.title = "Magic-wand hole fill: click INSIDE any enclosed tear and its "
    + "boundary rim becomes the fill — every rim vertex is used exactly, so the "
    + "patch is born welded to the mesh. Keep clicking holes, Enter exits, "
    + "✅ Apply builds. Very large or open-edged holes still need ⬜ Quad / ✏️ Draw.";
  wandBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  wandBtn.onclick = () => {
    wandOn = !wandOn;
    if (wandOn) {
      if (drawOn) drawBtn.onclick();
      if (editOn) editBtn.onclick();
      if (boxOn) boxBtn.onclick();
      if (sphereOn) sphereBtn.onclick();
      if (quadOn) quadBtn.onclick();
      if (extrudeOn) extrudeBtn.onclick();
    }
    controls.setEnabled(!wandOn);
    canvas.style.cursor = wandOn ? "crosshair" : "grab";
    refreshDrawOverlay();
    drawHud(wandOn
      ? "🪄 click inside an enclosed tear to fill it — Enter exits"
      : "");
  };

  // Assemble the rail in DCC order — create tools, edit tools, apply — with
  // thin separators between the groups. Buttons carry monochrome line-art SVG
  // icons (stroke:currentColor, so each button's existing colour toggling
  // recolours the icon for free); the full wording stays in each tooltip.
  const railSvg = (inner) =>
    `<svg width="26" height="26" viewBox="0 0 24 24" fill="none" ` +
    `stroke="currentColor" stroke-width="1.7" stroke-linecap="round" ` +
    `stroke-linejoin="round">${inner}</svg>`;
  const RAIL_ICONS = {
    draw: railSvg('<path d="M17 3.5l3.5 3.5L8 19.5 3.5 20.5 4.5 16 17 3.5z"/>'
      + '<path d="M14.5 6l3.5 3.5"/>'),
    quad: railSvg('<rect x="5.5" y="5.5" width="13" height="13" rx="1"/>'
      + '<rect x="3.6" y="3.6" width="3.8" height="3.8" fill="currentColor" stroke="none"/>'
      + '<rect x="16.6" y="3.6" width="3.8" height="3.8" fill="currentColor" stroke="none"/>'
      + '<rect x="3.6" y="16.6" width="3.8" height="3.8" fill="currentColor" stroke="none"/>'
      + '<rect x="16.6" y="16.6" width="3.8" height="3.8" fill="currentColor" stroke="none"/>'),
    extrude: railSvg('<path d="M4 19.5h16"/><path d="M12 16V5.5"/>'
      + '<path d="M8 9.5L12 5.5l4 4"/>'),
    box: railSvg('<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/>'
      + '<path d="M12 12l8-4.5M12 12L4 7.5M12 12v9"/>'),
    sphere: railSvg('<circle cx="12" cy="12" r="8.5"/>'
      + '<path d="M3.5 12c2.6 3.2 14.4 3.2 17 0" opacity="0.6"/>'),
    edit: railSvg('<path d="M6 3l12 9.5-5.5 1 2.8 6-3 1.3-2.7-6L6 18V3z"/>'),
    snap: railSvg('<path d="M7 3.5v8a5 5 0 0 0 10 0v-8"/>'
      + '<path d="M7 8h4M13 8h4"/>'),
    trash: railSvg('<path d="M4 7h16M10 7V4h4v3M6 7l1 13h10l1-13"/>'
      + '<path d="M10 11v5.5M14 11v5.5"/>'),
    // Clear-all: the same bin, shifted right to make room for sweep strokes —
    // "everything goes in", distinct from the single-shape 🗑 at a glance.
    trashAll: railSvg('<path d="M7.5 7.5h13M13 7.5V5h4v2.5M9.5 7.5l1 12.5h8l1-12.5"/>'
      + '<path d="M1.5 7h4M1.5 11h4M1.5 15h4" opacity="0.75"/>'),
    apply: railSvg('<path d="M4.5 12.5l5.5 5.5L19.5 6.5"/>'),
    wand: railSvg('<path d="M4 20l9-9"/>'
      + '<path d="M15.5 3.5v4M13.5 5.5h4"/>'
      + '<path d="M20 9.5v3M18.5 11h3"/>'
      + '<path d="M17 16.5v2.5M15.8 17.8h2.5" opacity="0.7"/>'),
    // Chevrons for the rail collapse toggle: up = "fold the tools away",
    // down = "bring them back".
    collapse: railSvg('<path d="M6 14.5l6-6 6 6"/>'),
    expand: railSvg('<path d="M6 9.5l6 6 6-6"/>'),
  };
  const styleRailBtn = (btn, icon) => {
    btn.innerHTML = RAIL_ICONS[icon];
    btn.style.width = "44px";
    btn.style.height = "44px";
    btn.style.padding = "0";
    btn.style.display = "flex";
    btn.style.alignItems = "center";
    btn.style.justifyContent = "center";
    btn.style.borderRadius = "8px";
    btn.style.borderColor = "transparent";
    btn.style.background = "transparent";
  };
  const railSeparator = () => {
    const s = document.createElement("div");
    s.style.cssText = "height:1px;margin:3px 6px;background:#34343e;";
    return s;
  };
  styleRailBtn(wandBtn, "wand");
  styleRailBtn(drawBtn, "draw");
  styleRailBtn(quadBtn, "quad");
  styleRailBtn(extrudeBtn, "extrude");
  styleRailBtn(boxBtn, "box");
  styleRailBtn(sphereBtn, "sphere");
  styleRailBtn(editBtn, "edit");
  styleRailBtn(snapBtn, "snap");
  styleRailBtn(deleteBtn, "trash");
  styleRailBtn(clearAllBtn, "trashAll");
  styleRailBtn(drawApplyBtn, "apply");
  // Collapse toggle. The tool buttons live in their own container so hiding
  // them leaves the toggle itself on screen — a control that can hide its own
  // only affordance for coming back is a trap. Purely presentational: it never
  // changes the active tool, never touches drawnPolygons and never sets
  // drawDirty, so folding the rail away mid-draw cannot lose work. The status
  // chip stays visible for exactly that reason — with the rail hidden it is
  // the only thing still reporting that Draw or Snap is live.
  //
  // Session-only, and deliberately NOT persisted into client_data: this is
  // presentation state, not solve evidence, and a saved workflow that opened
  // with its tools already hidden would read as a broken viewport.
  const railTools = document.createElement("div");
  railTools.style.cssText = "display:flex;flex-direction:column;gap:4px;";
  railTools.append(wandBtn, drawBtn, quadBtn, extrudeBtn, boxBtn, sphereBtn,
                   railSeparator(),
                   editBtn, snapBtn, deleteBtn, clearAllBtn,
                   railSeparator(), drawApplyBtn);

  let railToolsVisible = true;   // tools ON by default
  const railToggleBtn = document.createElement("button");
  railToggleBtn.style.cssText =
    "width:44px;height:24px;padding:0;display:flex;align-items:center;" +
    "justify-content:center;border:1px solid transparent;border-radius:8px;" +
    "background:transparent;color:#8a8a94;cursor:pointer;";
  function syncRailCollapsed() {
    railTools.style.display = railToolsVisible ? "flex" : "none";
    railToggleBtn.innerHTML = RAIL_ICONS[railToolsVisible ? "collapse" : "expand"]
      .replace('width="26" height="26"', 'width="20" height="20"');
    railToggleBtn.title = railToolsVisible
      ? "Hide the blockout tool rail (the active tool and snap state stay as "
        + "they are — nothing is discarded)"
      : "Show the blockout tool rail";
  }
  railToggleBtn.onclick = () => {
    railToolsVisible = !railToolsVisible;
    syncRailCollapsed();
  };
  drawRail.append(railToggleBtn, railTools);
  syncRailCollapsed();

  // The chip mirrors whichever tool is live plus the snap state. Every rail
  // button's onclick is wrapped (not replaced — mutual-exclusion cross-calls
  // still work) so the chip refreshes after any toggle.
  const chipIcon = (name) => {
    const s = document.createElement("span");
    s.style.cssText = "display:inline-flex;width:15px;height:15px;";
    s.innerHTML = RAIL_ICONS[name]
      .replace('width="26" height="26"', 'width="15" height="15"');
    return s;
  };
  function updateRailStatus() {
    railStatus.textContent = "";
    const active =
      wandOn ? ["wand", "Wand"] :
      drawOn ? ["draw", "Draw"] :
      quadOn ? ["quad", "Quad"] :
      extrudeOn ? ["extrude", "Extrude"] :
      boxOn ? ["box", "Box"] :
      sphereOn ? ["sphere", "Sphere"] :
      editOn ? ["edit", "Edit"] : null;
    if (active) {
      railStatus.appendChild(chipIcon(active[0]));
      railStatus.appendChild(document.createTextNode(active[1]));
    } else {
      railStatus.appendChild(document.createTextNode("Orbit"));
    }
    const dot = document.createElement("span");
    dot.textContent = "·";
    dot.style.color = "#666";
    railStatus.appendChild(dot);
    const snapWrap = document.createElement("span");
    snapWrap.style.cssText = "display:inline-flex;align-items:center;gap:5px;"
      + (editSnap ? "color:#7dd87d;" : "color:#888;");
    snapWrap.appendChild(chipIcon("snap"));
    snapWrap.appendChild(document.createTextNode(editSnap ? "Snap on" : "Snap off"));
    railStatus.appendChild(snapWrap);
  }
  // Uniform active styling (the buttons' own onclicks still set their legacy
  // toolbar tints — this runs after them and wins): active tool = blue fill,
  // snap = green, idle = bare icon.
  function syncRailActive() {
    const set = (btn, on) => {
      btn.style.background = on ? "#1d3a5c" : "transparent";
      btn.style.color = on ? "#8ec2ff" : "#c9c9d1";
    };
    set(wandBtn, wandOn);
    set(drawBtn, drawOn);
    set(quadBtn, quadOn);
    set(extrudeBtn, extrudeOn);
    set(boxBtn, boxOn);
    set(sphereBtn, sphereOn);
    set(editBtn, editOn);
    snapBtn.style.background = editSnap ? "#1e3a24" : "transparent";
    snapBtn.style.color = editSnap ? "#7dd87d" : "#c9c9d1";
  }
  deleteBtn.style.color = "#c9c9d1";
  clearAllBtn.style.color = "#c9c9d1";
  drawApplyBtn.style.background = "#1e3a24";
  drawApplyBtn.style.color = "#7dd87d";
  // Any OTHER rail action disarms a pending clear-all: the artist has moved on,
  // and an arm that outlived the moment would turn the next 🧹 click into a
  // one-click wipe — exactly what the two-click gate exists to prevent.
  for (const b of [wandBtn, drawBtn, quadBtn, extrudeBtn, boxBtn, sphereBtn,
                   editBtn, snapBtn]) {
    const orig = b.onclick;
    b.onclick = () => { disarmClearAll(); orig(); syncRailActive(); updateRailStatus(); };
  }
  for (const b of [deleteBtn, drawApplyBtn]) {
    const orig = b.onclick;
    b.onclick = () => { disarmClearAll(); orig(); };
  }
  syncRailActive();
  updateRailStatus();

  // Restore previously-applied outlines from the persisted widget, so ✎ Edit
  // works after a reload / workflow reopen and not only in the session that
  // drew them.
  try {
    const cdWidget = node.widgets?.find((w) => w.name === "client_data");
    const stored = cdWidget?.value ? JSON.parse(cdWidget.value).drawn_polygons : null;
    if (Array.isArray(stored) && stored.length) {
      drawnPolygons = stored
        .filter((poly) => Array.isArray(poly?.points_world) && poly.plane)
        .map((poly) => ({ ...poly, points_world: poly.points_world.map((q) => [...q]) }));
      refreshDrawOverlay();
    }
  } catch (_) { /* malformed client_data — the backend guard reports it */ }

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
      // Deliberately NOT multisampled, unlike every other target here: this
      // 160px probe is MEASURED for coverage, not looked at, and resolving
      // samples would blend partial coverage into the very counts it reports.
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
    const gizmoWas = pivotGizmo ? pivotGizmo.visible : false;
    scene.background = null;
    grid.visible = false;
    if (bgMesh) bgMesh.visible = false;
    if (pivotGizmo) pivotGizmo.visible = false;
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
      if (pivotGizmo) pivotGizmo.visible = gizmoWas;
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
  // 🎥 Camera Path — six deterministic one-click moves (Orbit L/R, Pan L/R,
  // Dolly In, 🌀 Vertigo) to test how 📽 Project holds up while the camera moves, then
  // bake to an IMAGE batch (path_frames) for a core Video Combine node, or
  // hand the raw keyframes (camera_path) to AtlasExportCameraPathUSD for a
  // DCC-facing animated camera. Every move is computed from the RECOVERED
  // camera pose + the geometry pivot (never the live orbited camera), always
  // 24 fps / 100 frames, ease_in_out — 📥 FBX import is the one exception
  // that may override fps/frame_count (a DCC clip defines its own timing).
  // The manual keyframe editor + free-fly controller were removed 2026-07-16
  // (git history has them). See camera_path.py's sample_camera_path — the
  // functions below (catmullRom3JS/applyEasingJS/sampleKeyframePoseAtFrame)
  // MUST stay in sync with it; they exist here (rather than round-tripping
  // to Python) so Play can scrub live at 60fps.
  // ---------------------------------------------------------------------------
  const PATH_FPS = 24;          // film — fixed; FBX import may override
  const PATH_FRAME_COUNT = 100; // always 100 frames; FBX import may override
  let pathMode = false;
  let pathKeyframes = []; // [{frame_index, position:{x,y,z}, target:{x,y,z}, up:{x,y,z}, easing}]
  let pathFrameCount = PATH_FRAME_COUNT;
  let pathFps = PATH_FPS;
  let pathLensScale = 1.2;
  // 🎬 Cinematic rig-noise state — persisted in client_data.camera_path and
  // read by Python's sample_camera_path (USD export opts in; analysis
  // consumers always get the clean move).
  let pathShakeEnabled = false;
  let pathShakeIntensity = 1.0;
  let pathShakeSeed = 1;
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
  function catmullRom1JS(p0, p1, p2, p3, t) {
    const t2 = t * t, t3 = t2 * t;
    return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
  }
  // 🎬 Cinematic rig-noise (track chatter / jib bounce / mechanical resonance)
  // — mirrors _hash01 / shake_offsets / apply_shake_to_pose in camera_path.py
  // EXACTLY (the same accepted hand-sync duplication as catmullRom3JS above;
  // pinned by tests/test_frontend_mirrors.py). Determinism doctrine: phases
  // come from a 32-bit INTEGER hash of (seed, band) — never Math.random() at
  // sample time and never a float sin-fract hash — so live preview, bake and
  // the Python USD export sample bit-identical curves.
  function atlasHash01(n) {
    n = n >>> 0;
    n = (((n ^ 61) >>> 0) ^ (n >>> 16)) >>> 0;
    n = Math.imul(n, 9) >>> 0;
    n = (n ^ (n >>> 4)) >>> 0;
    n = Math.imul(n, 0x27d4eb2d) >>> 0;
    n = (n ^ (n >>> 15)) >>> 0;
    return n / 4294967296;
  }
  function atlasShakeOffsetsJS(frame, fps, intensity, seed) {
    if (!(intensity > 0)) return [0, 0, 0, 0, 0, 0];
    const t = fps > 0 ? frame / fps : 0;
    const TWO_PI = 6.283185307179586;
    const ph = (k) => atlasHash01(((Math.imul(seed, 1013) >>> 0) + k) >>> 0) * TWO_PI;
    const sin = Math.sin;
    // Jib bounce (low frequency): vertical sway + slight pitch/roll.
    let dy = 0.0040 * (sin(TWO_PI * 0.35 * t + ph(0)) + 0.6 * sin(TWO_PI * 0.65 * t + ph(1)));
    let dx = 0.0012 * sin(TWO_PI * 0.45 * t + ph(2));
    let dz = 0;
    let rx = 0.12 * sin(TWO_PI * 0.35 * t + ph(3));
    let ry = 0;
    let rz = 0.07 * sin(TWO_PI * 0.55 * t + ph(4));
    // Track chatter (high frequency): small lateral/axial buzz + yaw.
    dx += 0.0007 * (sin(TWO_PI * 9.1 * t + ph(5)) + sin(TWO_PI * 11.7 * t + ph(6)) + sin(TWO_PI * 13.9 * t + ph(7)));
    dz += 0.0007 * (sin(TWO_PI * 9.1 * t + ph(8)) + sin(TWO_PI * 11.7 * t + ph(9)) + sin(TWO_PI * 13.9 * t + ph(10)));
    ry += 0.03 * sin(TWO_PI * 11.7 * t + ph(11));
    // Mechanical resonance (beat-modulated mid frequency).
    dy += 0.0015 * sin(TWO_PI * 4.3 * t + ph(12)) * (0.5 + 0.5 * sin(TWO_PI * 0.18 * t + ph(13)));
    rx += 0.02 * sin(TWO_PI * 4.3 * t + ph(14)) * (0.5 + 0.5 * sin(TWO_PI * 0.18 * t + ph(15)));
    return [dx * intensity, dy * intensity, dz * intensity, rx * intensity, ry * intensity, rz * intensity];
  }
  function atlasApplyShakeToPoseJS(pos, tgt, up, off) {
    const dx = off[0], dy = off[1], dz = off[2];
    const rxDeg = off[3], ryDeg = off[4], rzDeg = off[5];
    if (dx === 0 && dy === 0 && dz === 0 && rxDeg === 0 && ryDeg === 0 && rzDeg === 0) {
      return { position: pos, target: tgt, up: up };
    }
    const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const rot = (v, axis, ang) => { // Rodrigues about UNIT axis
      const c = Math.cos(ang), s = Math.sin(ang);
      const cr = cross(axis, v), d = dot(axis, v);
      return [
        v[0] * c + cr[0] * s + axis[0] * d * (1 - c),
        v[1] * c + cr[1] * s + axis[1] * d * (1 - c),
        v[2] * c + cr[2] * s + axis[2] * d * (1 - c),
      ];
    };
    const fwd = sub(tgt, pos);
    let dist = Math.sqrt(dot(fwd, fwd));
    if (dist <= 1e-12) dist = 1;
    const f = scl(fwd, 1 / dist);
    let r = cross(f, [0, 1, 0]);
    const rlen = Math.sqrt(dot(r, r));
    r = rlen > 1e-6 ? scl(r, 1 / rlen) : [1, 0, 0];
    const u = cross(r, f);
    const trans = [
      (r[0] * dx + u[0] * dy + f[0] * dz) * dist,
      (r[1] * dx + u[1] * dy + f[1] * dz) * dist,
      (r[2] * dx + u[2] * dy + f[2] * dz) * dist,
    ];
    const pos2 = add(pos, trans);
    const deg = Math.PI / 180;
    let f2 = rot(f, u, ryDeg * deg);
    f2 = rot(f2, r, rxDeg * deg);
    const tgt2 = add(pos2, scl(f2, dist));
    const up2 = rot(up, f2, rzDeg * deg);
    return { position: pos2, target: tgt2, up: up2 };
  }
  // Keyframed VERTICAL fov channel (🌀 Vertigo) — mirrors camera_path.py's
  // sample_camera_path_fov_deg exactly (fill-forward missing fovs, phantom
  // endpoints, same easing); returns null when no keyframe carries fov_deg
  // so static-lens paths keep the solved intrinsics. Pure (takes kfs) so
  // tests/test_frontend_mirrors.py can execute it against the Python twin.
  function sampleFovChannel(kfs, frame) {
    if (kfs.length === 0 || !kfs.some((k) => k.fov_deg != null)) return null;
    let prev = kfs.find((k) => k.fov_deg != null).fov_deg;
    const fovs = kfs.map((k) => (k.fov_deg != null ? (prev = k.fov_deg) : prev));
    if (kfs.length === 1) return fovs[0];
    const padded = [fovs[0], ...fovs, fovs[fovs.length - 1]];
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
    return catmullRom1JS(padded[seg], padded[seg + 1], padded[seg + 2], padded[seg + 3], easedT);
  }
  function sampleKeyframePoseAtFrame(frame) {
    const kfs = pathKeyframes;
    if (kfs.length === 0) return null;
    if (kfs.length === 1) return { position: kfs[0].position, target: kfs[0].target, fovDeg: sampleFovChannel(kfs, frame) };
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
      fovDeg: sampleFovChannel(kfs, frame),
    };
  }
  // Keyframe pose with the 🎬 Cinematic rig noise layered on — the ONE
  // application point shared by live playback and the bake, so baked pixels
  // match the preview exactly. `up` is [0,1,0] unless shake roll tilts it.
  function shakenPoseAtFrame(frame) {
    const pose = sampleKeyframePoseAtFrame(frame);
    if (!pose) return null;
    if (!pathShakeEnabled || !(pathShakeIntensity > 0)) {
      return { position: pose.position, target: pose.target, up: [0, 1, 0], fovDeg: pose.fovDeg };
    }
    const off = atlasShakeOffsetsJS(frame, pathFps, pathShakeIntensity, pathShakeSeed);
    const p = pose.position, tg = pose.target;
    const shaken = atlasApplyShakeToPoseJS([p.x, p.y, p.z], [tg.x, tg.y, tg.z], [0, 1, 0], off);
    return {
      position: { x: shaken.position[0], y: shaken.position[1], z: shaken.position[2] },
      target: { x: shaken.target[0], y: shaken.target[1], z: shaken.target[2] },
      up: shaken.up,
      fovDeg: pose.fovDeg,
    };
  }
  // Exposed to the shared animate() loop above via the outer `applyPathPoseAtT` name.
  applyPathPoseAtT = function (t) {
    const frame = t * Math.max(0, pathFrameCount - 1);
    const pose = shakenPoseAtFrame(frame);
    if (!pose) return;
    camera.position.set(pose.position.x, pose.position.y, pose.position.z);
    camera.up.set(pose.up[0], pose.up[1], pose.up[2]);
    camera.lookAt(pose.target.x, pose.target.y, pose.target.z);
    // 🔭 playback lens (display/bake-only FOV multiplier; slider below). Read
    // per-frame so dragging the slider mid-preview applies live. A keyframed
    // fov (🌀 Vertigo) replaces the solved fov as the BASE the multiplier
    // composes onto (tan-space divide in playbackLensFovDeg) — the two never
    // fight: the keyframes carry the dolly-zoom ramp, the slider stays a
    // uniform zoom on top. Playback end restores the solved FOV via
    // applyRecoveredView; cancel restores it in the 🎥 toggle.
    const baseFov = pose.fovDeg != null ? pose.fovDeg : (recoveredData ? solvedFovDeg() : null);
    if (baseFov != null) {
      camera.fov = playbackLensFovDeg(baseFov);
      camera.updateProjectionMatrix();
    }
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
    return { frame_index: kf.frame_index, position: v3(kf.position), target: v3(kf.target), up: v3(kf.up), fov_deg: kf.fov_deg ?? null, easing: kf.easing };
  }
  function kfFromJSON(data) {
    const obj = (a) => ({ x: a[0], y: a[1], z: a[2] });
    return { frame_index: data.frame_index, position: obj(data.position), target: obj(data.target), up: obj(data.up || [0, 1, 0]), fov_deg: data.fov_deg ?? null, easing: data.easing || "linear" };
  }

  function persistPathToClientData() {
    const widget = node.widgets?.find((w) => w.name === "client_data");
    if (!widget) return;
    let existing = {};
    try { existing = widget.value ? JSON.parse(widget.value) : {}; } catch (_) { existing = {}; }
    // Any authored-pose or lens edit invalidates previously rendered frames.
    // Keep the tiny parametric path, but never leave a stale image batch that
    // could be mistaken for the new camera/lens in a repair preview.
    delete existing.path_frames;
    delete existing.atlas_proxy_path;
    existing.camera_path = {
      keyframes: pathKeyframes.map(kfToJSON), fps: pathFps,
      frame_count: pathFrameCount, lens_scale: pathLensScale,
      baked_frame_indices: [],
      shake_enabled: pathShakeEnabled, shake_intensity: pathShakeIntensity,
      shake_seed: pathShakeSeed,
    };
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
        pathFps = cp.fps || PATH_FPS;
        pathFrameCount = cp.frame_count || PATH_FRAME_COUNT;
        pathLensScale = Number(cp.lens_scale ?? existing.atlas_proxy_path?.lens_scale
          ?? pathLensScale) || pathLensScale;
        pathShakeEnabled = !!cp.shake_enabled;
        const shakeIntensity = Number(cp.shake_intensity);
        pathShakeIntensity = Number.isFinite(shakeIntensity) ? shakeIntensity : 1.0;
        pathShakeSeed = Math.max(1, Math.floor(Number(cp.shake_seed) || 1));
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
    // Orbit controls stay ENABLED in path mode (the fly controller they used
    // to yield to is gone) — playback re-poses the camera per-frame anyway.
    if (!pathMode) {
      pathPlayback = null; // cancelling mid-play skips onDone —
      camera.up.set(0, 1, 0); // — so undo any 🎬 shake roll here too
      if (pivotGizmo) pivotGizmo.visible = pivotOn; // — so restore the gizmo here
      grid.visible = true; // — and the floor grid
      if (recoveredData) { // — and undo the 🔭 playback lens
        camera.fov = solvedFovDeg();
        camera.updateProjectionMatrix();
      }
    }
  };
  toolbar.appendChild(pathBtn);

  // Camera Path panel — keyframe list + timeline controls. Its own row below
  // the toolbar (see the "Assemble" section), hidden until 🎥 Camera Path is on.
  const pathPanel = document.createElement("div");
  pathPanel.style.cssText = "display:none;flex-wrap:wrap;align-items:center;gap:6px;padding:4px 6px;background:#181818;border-top:1px solid #333;font-size:11px;color:#ccc";

  rebuildPathVisualization();

  // Current camera pose as a {position, target} pair (target = a point straight
  // ahead at the solved scene depth, or a 10m fallback). Now used only by the
  // 📥 FBX import below as the base pose the imported clip is applied onto.
  function captureCurrentPose() {
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    const captureDist = recoveredData?.camera_meta?.scene_depth_m || 10;
    const target = camera.position.clone().addScaledVector(forward, captureDist);
    return {
      position: { x: camera.position.x, y: camera.position.y, z: camera.position.z },
      target: { x: target.x, y: target.y, z: target.z },
    };
  }

  // ---------------------------------------------------------------------------
  // Shared move math for the one-click buttons. Pan rotates the TARGET around
  // the (fixed) camera position — the camera swivels in place, like a real
  // pan. Orbit moves the POSITION around the (fixed) target — the camera arcs
  // around the subject. Dolly moves the POSITION toward the (fixed) target
  // along the view axis. All three are plain vector math (no Euler/yaw sign
  // ambiguity): "right"/"left" and "in" are derived directly from the base
  // pose's own forward/right vectors, so they're unambiguous regardless of
  // world orientation. (Numerically verified — see git history.)
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

  // Deterministic basis for the one-click moves: the RECOVERED camera pose
  // (never the live orbited camera — "full frame" means the photo's own
  // framing, and re-clicking a move after hand-orbiting must produce the
  // identical path) + the geometry pivot (median-depth mesh centre, same
  // cascade as applyRecoveredView's reset) + the artist's 🎯 offset.
  // Deliberately NOT targetWithOffset() — that helper re-baselines pivotBase
  // as a side effect, which would silently move what applyPivotOffset()
  // re-targets later.
  function recoveredMoveBasis() {
    if (!recoveredData?.view_matrix || !THREE) return null;
    const tmp = new THREE.PerspectiveCamera();
    applyRecoveredCamera(tmp, recoveredData); // pose only; throwaway camera
    let base = lastGeometryPivot ? lastGeometryPivot.clone() : null;
    if (!base) {
      const sceneDepth = recoveredData.camera_meta?.scene_depth_m;
      base = groundPointInView(tmp, sceneDepth ? sceneDepth * 1.5 : 30);
    }
    const pivot = base.add(pivotOffset);
    return { position: tmp.position.clone(), quaternion: tmp.quaternion.clone(), pivot };
  }

  // -------------------------------------------------------------------------
  // 🔭 Playback lens — a focal MULTIPLIER applied to the viewing camera's FOV
  // during path playback and bake ONLY (never the projection shader's own
  // camera/uniforms, never the solve, never the USD export — those keep the
  // recovered intrinsics). Slider left = wider (the left end is recomputed at
  // every move click so full-wide frames the projection geometry's edges),
  // right = zoomed in. Default 1.2 = the gentle push-in the moves used to get
  // by physically lerping toward the pivot (which raised the eye — see the
  // move comment below).
  // -------------------------------------------------------------------------
  const LENS_MAX = 2.5;
  function solvedFovDeg() {
    const imageH = recoveredData?.render_image_height ?? recoveredData?.image_height ?? 1080;
    const fy = recoveredData?.render_fy ?? recoveredData?.fy ?? 1;
    return 2 * Math.atan(imageH / (2 * fy)) * (180 / Math.PI);
  }
  function playbackLensFovDeg(baseFovDeg) {
    // Base defaults to the solved fov; 🌀 Vertigo playback/bake passes the
    // keyframed per-frame fov instead so the slider composes with (never
    // overrides) a keyframed lens ramp.
    const fov0 = baseFovDeg != null ? baseFovDeg : solvedFovDeg();
    const m = Math.max(0.05, pathLensScale);
    return 2 * Math.atan(Math.tan((fov0 * Math.PI) / 360) / m) * (180 / Math.PI);
  }
  // Bounding radius of the ACTUAL projection geometry about the pivot —
  // derived proxies + patch/clean-plate groups, excluding the far catch-all
  // "projection_backdrop" plane (it would balloon the wide end out to the
  // backdrop distance instead of the photographed geometry's edges).
  function geometryBoundingRadius(center) {
    let r = 0;
    const box = new THREE.Box3();
    const groups = scene.children.filter(
      (c) => c.name === "atlas_derived_proxies" || /^atlas_patch_/.test(c.name || ""));
    groups.forEach((g) => g.traverse((o) => {
      if (!o.isMesh || o.name === "projection_backdrop") return;
      box.setFromObject(o);
      if (box.isEmpty()) return;
      for (const cx of [box.min.x, box.max.x])
        for (const cy of [box.min.y, box.max.y])
          for (const cz of [box.min.z, box.max.z])
            r = Math.max(r, center.distanceTo(new THREE.Vector3(cx, cy, cz)));
    }));
    return r;
  }
  function updateLensWideLimit(E, P) {
    const d = E.distanceTo(P) || 1;
    const R = geometryBoundingRadius(P);
    let min = 0.5; // fallback wide end when there is no geometry to measure
    if (R > 0) {
      // FOV that fits a sphere of radius R at distance d (capped: past 120°
      // the perspective distortion stops being useful for a marketing move).
      const fovFit = Math.min(120, 2 * Math.asin(Math.min(0.999, R / d)) * (180 / Math.PI));
      const fov0 = solvedFovDeg();
      min = Math.tan((fov0 * Math.PI) / 360) / Math.tan((fovFit * Math.PI) / 360);
      min = Math.min(1.0, Math.max(0.2, min)); // wide end is never a zoom-in
    }
    lensSlider.min = String(Math.round(min * 100) / 100);
    if (pathLensScale < min) { pathLensScale = min; lensSlider.value = String(min); }
    lensReadout.textContent = `${Number(pathLensScale).toFixed(2)}×`;
  }

  // One-click moves. Fixed grammar (user-specified, 2026-07-16): slow filmic
  // moves, 24 fps / 100 frames, ease_in_out. EVERY move's frame 0 sits at the
  // EXACT recovered camera position (the earlier 20% positional pre-zoom
  // lerped toward the pivot, which sits higher than the eye on most plates —
  // it visibly RAISED frame 0; artist-reported, removed 2026-07-16). Orbit/Pan
  // look at the pivot (which lies on the recovered central view ray, so
  // composition holds) — Orbit arcs ±15° about world-Y through the pivot, Pan
  // swivels ±15° in place. Dolly In pushes 20% of the camera→pivot distance
  // along the recovered view axis (target ON that axis — framing preserved).
  // The "zoom in a little" now comes from the playback LENS (slider below),
  // never from moving the camera.
  const MOVE_ANGLE_DEG = 15;
  const MOVE_DOLLY_FRAC = 0.2;  // dolly-in travel as a fraction of cam→pivot
  const PUSH_IN_FRAC = 0.35;    // ⭆ Push In — a stronger, ease-shaped dolly
  const ARC_DOLLY_FRAC = 0.15;  // ⤴/⤵ Arc — dolly component of the combined move
  // Easing applied to a move's FIRST keyframe (the whole 2-keyframe move, or
  // the opening segment of a 3-keyframe arc). Values must be exactly the four
  // curves camera_path.py understands — this selector adds no new easing
  // functions, so the JS/Python mirror pin is untouched.
  let moveEasing = "ease_in_out";
  function applyMovePreset(kind) {
    const basis = recoveredMoveBasis();
    if (!basis) return; // no solve yet — nothing to move around
    const { position: E, quaternion, pivot: P } = basis;
    const v3o = (v) => ({ x: v.x, y: v.y, z: v.z });
    const kfAt = (frame, pose, easing) => ({
      frame_index: frame, position: pose.position, target: pose.target,
      up: { x: 0, y: 1, z: 0 }, easing,
    });
    let newKeyframes;
    if (kind === "dolly_in" || kind === "push_in" || kind === "vertigo") {
      const d = E.distanceTo(P) || 1;
      const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion);
      const T = E.clone().addScaledVector(fwd, d); // target ON the view axis
      const startPose = { position: v3o(E), target: v3o(T) };
      const frac = kind === "push_in" ? PUSH_IN_FRAC : MOVE_DOLLY_FRAC;
      const endPose = computePresetEndPose(startPose, "dolly_in", 0, frac);
      newKeyframes = [kfAt(0, startPose, moveEasing),
                      kfAt(PATH_FRAME_COUNT - 1, endPose, "linear")];
    } else if (kind === "arc_left" || kind === "arc_right") {
      // Combined orbit + dolly-in. THREE keyframes (0 / 60 / 99) so the
      // Catmull-Rom actually curves through the combined move instead of
      // cutting the chord straight to the end pose.
      const startPose = { position: v3o(E), target: v3o(P) };
      const orbitKind = kind === "arc_left" ? "orbit_left" : "orbit_right";
      const arcPose = (angleDeg, frac) => {
        const o = computePresetEndPose(startPose, orbitKind, angleDeg, 0);
        const T = o.target, Ep = o.position, k = 1 - frac; // dolly toward the fixed target
        return {
          position: { x: T.x + (Ep.x - T.x) * k, y: T.y + (Ep.y - T.y) * k, z: T.z + (Ep.z - T.z) * k },
          target: { ...T },
        };
      };
      newKeyframes = [
        kfAt(0, startPose, moveEasing),
        kfAt(60, arcPose(MOVE_ANGLE_DEG / 2, ARC_DOLLY_FRAC / 2), "linear"),
        kfAt(PATH_FRAME_COUNT - 1, arcPose(MOVE_ANGLE_DEG, ARC_DOLLY_FRAC), "linear"),
      ];
    } else {
      const startPose = { position: v3o(E), target: v3o(P) }; // locked at the recovered eye
      const endPose = computePresetEndPose(startPose, kind, MOVE_ANGLE_DEG, 0);
      newKeyframes = [kfAt(0, startPose, moveEasing),
                      kfAt(PATH_FRAME_COUNT - 1, endPose, "linear")];
    }
    updateLensWideLimit(E, P); // widen the slider's left end to fit this scene
    // A move click always restores the fixed film timing (an earlier FBX
    // import may have overridden it) and replaces any existing keyframes.
    pathFps = PATH_FPS;
    pathFrameCount = PATH_FRAME_COUNT;
    pathKeyframes = newKeyframes;
    if (kind === "vertigo") {
      // 🌀 Dolly-zoom: same 20% push-in as Dolly In, but the lens counter-
      // animates so the pivot-plane framing holds — the target sits at the
      // camera→pivot distance d, the end distance is d·(1−frac), and the
      // pivot plane's framed width ∝ d·tan(fov/2), so holding it needs
      // tan(fov_end/2) = tan(fov_start/2) / (1−frac) (wider lens as the
      // camera closes in; the background visibly recedes). fov keyframes are
      // the SOLVED vertical fov pre-🔭-lens: the playback slider composes on
      // top as a uniform zoom, exactly like every other move.
      const fov0 = solvedFovDeg();
      const fov1 = 2 * Math.atan(Math.tan((fov0 * Math.PI) / 360) / (1 - MOVE_DOLLY_FRAC)) * (180 / Math.PI);
      pathKeyframes[0].fov_deg = fov0;
      pathKeyframes[pathKeyframes.length - 1].fov_deg = fov1;
    }
    rebuildPathVisualization();
    persistPathToClientData();
    playBtn.onclick(); // auto-preview once; snaps back to the recovered view on done
  }

  // APPEND-ONLY: the kind strings serialize into client_data camera_path (and
  // into muscle memory and docs) — never rename an existing one.
  const MOVES = [
    ["orbit_left", "⟲ Orbit L", "Arc 15° left around the mesh centre from the exact recovered eye"],
    ["orbit_right", "⟳ Orbit R", "Arc 15° right around the mesh centre from the exact recovered eye"],
    ["pan_left", "⇠ Pan L", "Swivel 15° left in place at the exact recovered eye"],
    ["pan_right", "⇢ Pan R", "Swivel 15° right in place at the exact recovered eye"],
    ["dolly_in", "⭢ Dolly In", "Push in 20% of the distance to the mesh centre from the full recovered framing"],
    ["arc_left", "⤴ Arc L", "Combined move: orbit 15° left WHILE pushing in 15% — 3 keyframes so the path genuinely curves"],
    ["arc_right", "⤵ Arc R", "Combined move: orbit 15° right WHILE pushing in 15% — 3 keyframes so the path genuinely curves"],
    ["push_in", "⭆ Push In", "Stronger 35% push toward the mesh centre, shaped by the easing selector"],
    ["vertigo", "🌀 Vertigo", "Dolly-zoom: push in 20% while the lens widens to hold the pivot-plane framing — background recedes"],
  ];
  const moveWrap = document.createElement("span");
  moveWrap.style.cssText = "display:inline-flex;align-items:center;gap:3px;";
  MOVES.forEach(([kind, label, tip]) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.title = tip + " — 100 frames @ 24 fps; computed from the recovered camera, not the current view";
    b.style.cssText = "padding:2px 6px;font-size:11px;cursor:pointer;background:#2a2a3a;color:#dcf;border:1px solid #546;border-radius:3px";
    b.onclick = () => applyMovePreset(kind);
    moveWrap.appendChild(b);
  });
  // Easing selector for the moves above — the four curves camera_path.py
  // already implements; picking one changes the NEXT move click.
  const easeSel = document.createElement("select");
  easeSel.title = "Easing for the one-click moves (applies on the next move click)";
  easeSel.style.cssText = "padding:1px 2px;font-size:11px;background:#2a2a3a;color:#dcf;border:1px solid #546;border-radius:3px";
  ["ease_in_out", "ease_in", "ease_out", "linear"].forEach((v) => {
    const o = document.createElement("option");
    o.value = v; o.textContent = v.replace(/_/g, " ");
    easeSel.appendChild(o);
  });
  easeSel.value = moveEasing;
  easeSel.onchange = () => { moveEasing = easeSel.value; };
  moveWrap.appendChild(easeSel);

  // 🔭 Lens slider — see the playback-lens comment block above.
  const lensWrap = document.createElement("span");
  lensWrap.style.cssText = "display:inline-flex;align-items:center;gap:3px;padding-left:6px;border-left:1px solid #333;";
  lensWrap.title = "Playback lens: slide left = wider (full left frames the projection geometry's edges), " +
    "right = zoomed in. Applies to path playback + baked frames ONLY — the solve, the projection, " +
    "and the USD camera keep the recovered lens. Live while a preview plays.";
  const lensLabel = document.createElement("span");
  lensLabel.textContent = "🔭";
  const lensSlider = document.createElement("input");
  lensSlider.type = "range";
  lensSlider.min = "0.5"; // widened per-scene at each move click (updateLensWideLimit)
  lensSlider.max = String(LENS_MAX);
  lensSlider.step = "0.01";
  lensSlider.value = String(pathLensScale);
  lensSlider.style.cssText = "width:90px;";
  const lensReadout = document.createElement("span");
  lensReadout.textContent = `${pathLensScale.toFixed(2)}×`;
  lensReadout.style.cssText = "min-width:38px;color:#9ad;";
  lensSlider.oninput = () => {
    pathLensScale = parseFloat(lensSlider.value) || 1.0;
    lensReadout.textContent = `${pathLensScale.toFixed(2)}×`;
    persistPathToClientData();
    // applyPathPoseAtT reads pathLensScale per frame — a mid-preview drag
    // applies live with no extra wiring.
  };
  lensWrap.append(lensLabel, lensSlider, lensReadout);

  // 🎬 Cinematic rig-noise controls. Unlike the 🔭 lens (display/bake-only),
  // shake ALSO reaches the exported USD camera — sample_camera_path applies it
  // when the path enables it, so the DCC camera matches the baked pixels.
  // The same intensity drives a live-only handheld layer on the WASD/arrow
  // tracking keys (see animate()). All changes persist via
  // persistPathToClientData, which also invalidates stale baked frames.
  const shakeWrap = document.createElement("span");
  shakeWrap.style.cssText = "display:inline-flex;align-items:center;gap:3px;padding-left:6px;border-left:1px solid #333;";
  shakeWrap.title = "Cinematic rig noise: deterministic track chatter + jib bounce + mechanical resonance " +
    "on path playback, baked frames AND the exported USD camera (same seed everywhere). " +
    "Also adds a live handheld feel to the tracking keys. 🎲 rerolls the seed.";
  const shakeBtn = document.createElement("button");
  shakeBtn.textContent = "🎬";
  shakeBtn.style.cssText = "padding:2px 6px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  const syncShakeBtn = () => { shakeBtn.style.background = pathShakeEnabled ? "#3a2a1a" : "#2a2a2a"; };
  const shakeSlider = document.createElement("input");
  shakeSlider.type = "range";
  shakeSlider.min = "0";
  shakeSlider.max = "2";
  shakeSlider.step = "0.05";
  shakeSlider.value = String(pathShakeIntensity);
  shakeSlider.style.cssText = "width:70px;";
  const shakeReadout = document.createElement("span");
  shakeReadout.style.cssText = "min-width:38px;color:#da9;";
  const syncShakeReadout = () => { shakeReadout.textContent = `${pathShakeIntensity.toFixed(2)}×`; };
  shakeBtn.onclick = () => {
    pathShakeEnabled = !pathShakeEnabled;
    syncShakeBtn();
    persistPathToClientData();
  };
  shakeSlider.oninput = () => {
    pathShakeIntensity = parseFloat(shakeSlider.value) || 0;
    syncShakeReadout();
    persistPathToClientData();
    // shakenPoseAtFrame reads the vars per frame — a mid-preview drag applies
    // live, same as the 🔭 lens.
  };
  const reseedBtn = document.createElement("button");
  reseedBtn.textContent = "🎲";
  reseedBtn.title = "Reroll the shake seed (a different but equally deterministic take)";
  reseedBtn.style.cssText = "padding:2px 5px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
  reseedBtn.onclick = () => {
    // Math.random at CLICK time only — sampling stays fully deterministic.
    pathShakeSeed = 1 + Math.floor(Math.random() * 0x7fffffff);
    persistPathToClientData();
  };
  syncShakeBtn();
  syncShakeReadout();
  shakeWrap.append(shakeBtn, shakeSlider, shakeReadout, reseedBtn);

  const playBtn = document.createElement("button");
  playBtn.textContent = "▶ Play";
  playBtn.style.cssText = "padding:2px 6px;font-size:11px;cursor:pointer;background:#2a2a3a;color:#dcf;border:1px solid #546;border-radius:3px";
  playBtn.onclick = () => {
    if (pathKeyframes.length === 0) return;
    // Clean playback: hide the 🎯 pivot gizmo + the orange path markers + the
    // floor grid for the duration of ANY preview (artist request — playback
    // is what gets screen-recorded for marketing). Restore is derived from
    // the helpers' OWNER states (pivotOn / pathMode; the grid is always-on
    // outside the deterministic passes), never a stashed .visible — a
    // mid-play re-click would stash the already-hidden value and restore to
    // hidden forever.
    if (pivotGizmo) pivotGizmo.visible = false;
    pathGroup.visible = false;
    grid.visible = false;
    pathPlayback = {
      startTime: performance.now(),
      durationSec: Math.max(0.2, pathFrameCount / pathFps),
      onDone: () => {
        if (pivotGizmo) pivotGizmo.visible = pivotOn;
        pathGroup.visible = pathMode;
        grid.visible = true;
        if (recoveredData) applyRecoveredView(recoveredData, { force: true });
      },
    };
  };

  let repairBakeBtn = null;
  let bakeBtn = null;

  async function bakeProxyPathFrames(frameIndices, triggerBtn, busyLabel) {
    if (pathKeyframes.length === 0) return;
    repairBakeBtn.disabled = true;
    bakeBtn.disabled = true;
    const idleLabel = triggerBtn.textContent;
    triggerBtn.textContent = busyLabel;
    const savedPos = camera.position.clone();
    const savedQuat = camera.quaternion.clone();
    const savedAspect = camera.aspect;
    const savedFov = camera.fov;
    const wasPlaying = !!pathPlayback;
    let outputRt = null;
    pathPlayback = null;
    bakeInProgress = true; // keeps the live nav handheld shake out of baked renders
    pathGroup.visible = false; // exclude keyframe markers/line from baked frames
    grid.visible = false; // keep the floor grid out of baked frames
    if (pivotGizmo) pivotGizmo.visible = false; // keep the 🎯 marker out of baked frames
    try {
      const frames = [];
      const bakedFrameIndices = [];
      // Baked frames are a deliverable, not a preview — MSAA here is the
      // difference between a clean silhouette and a stair-stepped one.
      outputRt = new THREE.WebGLRenderTarget(W, H, { samples: 4 });
      camera.aspect = W / H;
      // Baked frames honor the 🔭 playback lens so the recording matches the
      // preview exactly (the USD camera ships the solved intrinsics — plus
      // the keyframed fov ramp when 🌀 Vertigo keyed one, never the lens).
      if (recoveredData) camera.fov = playbackLensFovDeg();
      camera.updateProjectionMatrix();
      for (const frame of frameIndices) {
        // shakenPoseAtFrame (not sampleKeyframePoseAtFrame): the 🎬 rig noise
        // bakes into the pixels through the SAME helper playback uses.
        const pose = shakenPoseAtFrame(frame);
        if (!pose) continue;
        camera.position.set(pose.position.x, pose.position.y, pose.position.z);
        camera.up.set(pose.up[0], pose.up[1], pose.up[2]);
        camera.lookAt(pose.target.x, pose.target.y, pose.target.z);
        // Keyframed fov (🌀 Vertigo) — same base-composes-with-🔭-lens rule
        // as applyPathPoseAtT, re-set per frame since the ramp animates.
        if (pose.fovDeg != null) {
          camera.fov = playbackLensFovDeg(pose.fovDeg);
          camera.updateProjectionMatrix();
        }
        // JPEG, not PNG: baked frames feed a video encoder (h264, lossy), so
        // lossless PNG is pure waste — JPEG is ~5–10× smaller and stops the
        // whole clip's base64 from OOM-ing the JS heap when it's all stringified
        // into one client_data blob (the reported bake OOM at 1280×100 frames).
        frames.push(atlasRenderSceneToBase64(renderer, scene, camera, W, H,
          { renderTarget: outputRt, mime: "image/jpeg", quality: 0.9 }));
        bakedFrameIndices.push(frame);
      }
      const widget = node.widgets?.find((w) => w.name === "client_data");
      let existing = {};
      try { existing = widget?.value ? JSON.parse(widget.value) : {}; } catch (_) { existing = {}; }
      existing.path_frames = frames;
      existing.camera_path = {
        keyframes: pathKeyframes.map(kfToJSON), fps: pathFps,
        frame_count: pathFrameCount, lens_scale: pathLensScale,
        baked_frame_indices: bakedFrameIndices,
        shake_enabled: pathShakeEnabled, shake_intensity: pathShakeIntensity,
        shake_seed: pathShakeSeed,
      };
      existing.atlas_proxy_path = {
        transport: "jpeg_base64_proxy_ldr",
        width: W,
        height: H,
        fps: pathFps,
        frame_count: pathFrameCount,
        stored_frame_count: frames.length,
        frame_indices: bakedFrameIndices,
        lens_scale: pathLensScale, // 🔭 display lens baked into the frames (provenance)
      };
      if (widget) {
        widget.value = JSON.stringify(existing);
        widget.callback?.(widget.value);
      }
      app.queuePrompt(0, 1);
    } finally {
      bakeInProgress = false;
      outputRt?.dispose();
      camera.position.copy(savedPos);
      camera.quaternion.copy(savedQuat);
      camera.up.set(0, 1, 0); // undo any 🎬 shake roll left on the up vector
      camera.aspect = savedAspect;
      camera.fov = savedFov;
      camera.updateProjectionMatrix();
      pathGroup.visible = pathMode;
      // Owner-state restores (grid is always-on outside the deterministic
      // passes; the gizmo follows the 🎯 panel) — a stash here would capture
      // "already hidden" when Bake is clicked mid-preview (Bake cancels the
      // preview, so its onDone restore never fires) and leave them off forever.
      grid.visible = true;
      if (pivotGizmo) pivotGizmo.visible = pivotOn;
      repairBakeBtn.disabled = false;
      bakeBtn.disabled = false;
      triggerBtn.textContent = idleLabel;
      if (wasPlaying) playBtn.onclick();
    }
  }

  repairBakeBtn = document.createElement("button");
  repairBakeBtn.textContent = "📷 Bake Repair Frame";
  repairBakeBtn.title = "Store only the final path frame for path-guided repair (~99% smaller than a full 100-frame bake)";
  repairBakeBtn.style.cssText = "padding:2px 8px;font-size:11px;cursor:pointer;background:#1a3a2a;color:#bfe;border:1px solid #465;border-radius:3px";
  repairBakeBtn.onclick = () => bakeProxyPathFrames(
    [Math.max(0, pathFrameCount - 1)],
    repairBakeBtn,
    "Baking Repair Frame...",
  );

  bakeBtn = document.createElement("button");
  bakeBtn.textContent = "⏺ Bake Full Path";
  bakeBtn.title = "Store every rendered frame for video/clip workflows";
  bakeBtn.style.cssText = "padding:2px 8px;font-size:11px;cursor:pointer;background:#3a1a2a;color:#fac;border:1px solid #645;border-radius:3px";
  bakeBtn.onclick = () => {
    const frameIndices = Array.from(
      { length: Math.max(0, pathFrameCount) }, (_, index) => index);
    return bakeProxyPathFrames(
      frameIndices,
      bakeBtn,
      "Baking Full Path...",
    );
  };


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

      // The one sanctioned override of the fixed 24/100 timing: a DCC clip
      // defines its own duration. Any move-button click restores the defaults.
      pathFrameCount = sampleCount;
      if (clip.duration > 0) {
        pathFps = Math.max(1, Math.round(sampleCount / clip.duration));
      }
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

  pathPanel.append(
    moveWrap, lensWrap, shakeWrap, playBtn, repairBakeBtn, bakeBtn, importWrap);

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
    ensurePivotGizmo();               // show the on-screen marker while adjusting
    if (pivotGizmo) { pivotGizmo.visible = pivotOn; updatePivotGizmo(); }
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
    node._atlasLastRenderError = null;
    const savedAspect = camera.aspect;
    try {
      camera.aspect = W / H;
      camera.updateProjectionMatrix();
      const passes = await renderAllPasses(
        renderer, scene, camera, W, H,
        [bgMesh, pivotGizmo].filter(Boolean),
        { texture: patchMaskTex, camera: patchMaskCam },
      );
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
            passes: ["shaded", "depth", "normal", "mask", "patch_render_mask"],
          },
        });
        widget.callback?.(widget.value);
      }
      // Re-queue the prompt so Python receives the frames
      app.queuePrompt(0, 1);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      node._atlasLastRenderError = message;
      renderBtn.title = `Proxy render failed: ${message}`;
      console.error("[Atlas] Render Proxy Passes failed:", error);
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
    previewScale = atlasBackbufferScale(previewW, previewH);
    node._atlasPreviewW = previewW; node._atlasPreviewH = previewH;
    canvas.width = Math.round(previewW * previewScale);
    canvas.height = Math.round(previewH * previewScale);
    renderer.setSize(canvas.width, canvas.height, false);
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
    // 🎭 debug-matte isolate: (re)load the matte texture + capture the PRIMARY
    // camera for the shader's per-fragment projection. NoColorSpace (a matte is
    // data, not color) + flipY:false (top-left uv origin, like uMatte). The old
    // texture is disposed; absent matte clears the isolate entirely.
    try {
      if (debugMatteTex) { debugMatteTex.dispose(); debugMatteTex = null; }
      debugMatteCam = null;
      if (data.debug_matte_b64) {
        const flat = data.view_matrix.flat();
        const dvm = new THREE.Matrix4();
        dvm.set(flat[0], flat[1], flat[2], flat[3],
                flat[4], flat[5], flat[6], flat[7],
                flat[8], flat[9], flat[10], flat[11],
                flat[12], flat[13], flat[14], flat[15]);
        debugMatteCam = {
          vm: dvm,
          fx: data.fx || 1, fy: data.fy || data.fx || 1,
          cx: data.cx ?? (data.image_width || 1) / 2,
          cy: data.cy ?? (data.image_height || 1) / 2,
          w: data.image_width || 1, h: data.image_height || 1,
        };
        new THREE.TextureLoader().load(data.debug_matte_b64, (tex) => {
          tex.colorSpace = THREE.NoColorSpace;
          tex.flipY = false;
          tex.needsUpdate = true;
          if (debugMatteTex) debugMatteTex.dispose();
          debugMatteTex = tex;
          node._atlasRefreshMatteBtn?.();
        });
      }
      node._atlasRefreshMatteBtn?.();
    } catch (e) { /* a bad matte must never break the viewport refresh */ }
    // ◩ Planar patch identity: source-space data sampled through the original
    // recovered camera. It drives both the 🎨 Layers magenta overlay and the
    // current-camera patch_render_mask proxy pass.
    try {
      if (patchMaskTex) { patchMaskTex.dispose(); patchMaskTex = null; }
      patchMaskCam = null;
      if (data.patch_mask_b64) {
        const flat = data.view_matrix.flat();
        const pvm = new THREE.Matrix4();
        pvm.set(flat[0], flat[1], flat[2], flat[3],
                flat[4], flat[5], flat[6], flat[7],
                flat[8], flat[9], flat[10], flat[11],
                flat[12], flat[13], flat[14], flat[15]);
        patchMaskCam = {
          vm: pvm,
          fx: data.fx || 1, fy: data.fy || data.fx || 1,
          cx: data.cx ?? (data.image_width || 1) / 2,
          cy: data.cy ?? (data.image_height || 1) / 2,
          w: data.image_width || 1, h: data.image_height || 1,
        };
        new THREE.TextureLoader().load(data.patch_mask_b64, (tex) => {
          tex.colorSpace = THREE.NoColorSpace;
          tex.flipY = false;
          tex.magFilter = THREE.NearestFilter;
          tex.minFilter = THREE.NearestFilter;
          tex.needsUpdate = true;
          if (patchMaskTex) patchMaskTex.dispose();
          patchMaskTex = tex;
          if (layerDebugOn) refreshLayerLegend();
        });
      }
      if (layerDebugOn) refreshLayerLegend();
    } catch (e) { /* a bad patch mask must never break the viewport refresh */ }
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
    // textContent clears children, so the warning render below stays
    // idempotent across repeated updates.
    metaHud.textContent = lines.join("\n") || "(no camera metadata)";
    const sh = m.scale_health;
    if (sh && !sh.safe_to_export) {
      const warn = document.createElement("div");
      warn.textContent = `⚠ SCALE ${String(sh.status || "").toUpperCase()} — ${sh.detail || "not verified"}`;
      warn.style.color = "#ff6a3d";
      warn.style.marginTop = "3px";
      warn.style.maxWidth = "340px";
      warn.style.whiteSpace = "normal";
      metaHud.appendChild(warn);
    }
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
      
      const buildDrawnMat = (dTex) => {
        // Drawn surfaces project the SMEARED plate: the raw one would paint
        // them with whatever occluded them, since they stand exactly where the
        // camera has no data. Same projector, different texture — so they stay
        // registered with everything else.
        if (!data.drawn_plate_b64) {
          scene.traverse((c) => {
            if (c.userData?.atlasDrawn) delete c.userData._projMaterial;
          });
          return;
        }
        new THREE.TextureLoader().load(data.drawn_plate_b64, (tex) => {
          // Same top-left-origin convention as loadProjectionTexture: the
          // projection shader flips V itself, so three.js must not.
          tex.flipY = false;
          tex.colorSpace = THREE.SRGBColorSpace;
          const mat = makeProjectionMaterial(data, tex, { primaryDepthTexture: dTex });
          let used = false;
          scene.traverse((c) => {
            if (!c.userData?.atlasDrawn) return;
            const stale = c.userData._projMaterial;
            c.userData._projMaterial = mat;
            used = true;
            if (stale && stale !== mat) {
              stale.uniforms?.uTexture?.value?.dispose?.();
              stale.dispose?.();
            }
          });
          if (!used) { tex.dispose(); mat.dispose?.(); return; }
          if (projectionOn) applyProjection(true);
        });
      };

      // A skipped outline has to SAY so. The per-outline report used to be a
      // STRING output only, so a fill that was dropped — stale fingerprint,
      // unknown shape kind, self-intersecting outline — just failed to appear:
      // you drew three, got two, and nothing anywhere told you why. Silence on
      // a branch skip is what the gate doctrine exists to prevent. Only
      // surfaced when something actually skipped; an all-ok report stays quiet
      // so the HUD does not nag on every successful Apply.
      if (typeof data.draw_report === "string" && data.draw_report.includes("skipped(")) {
        const skipped = data.draw_report
          .split("\n")
          .filter((line) => line.includes("skipped("));
        if (skipped.length) {
          drawHud(`⚠️ ${skipped.length} outline(s) skipped — ${skipped.join(" · ")}`);
        }
      }

      const buildBackdropMat = (dTex) => {
        // With a clean plate connected, the projection_backdrop plane AND any
        // solve_b geometry merged in by AtlasMergeGeometry (atlasCleanSource —
        // the clean-background layer of a layered solve) project IT instead
        // of the source photo: tears in the primary mesh then reveal the
        // clean background behind the foreground — the simple composite.
        // Same projector, so it stays registered with the plate.
        const wantsClean = (c) =>
          c.name === "projection_backdrop" || c.userData?.atlasCleanSource;
        if (!data.clean_plate_b64) {
          scene.traverse((c) => {
            if (wantsClean(c)) delete c.userData._projMaterial;
          });
          return;
        }
        new THREE.TextureLoader().load(data.clean_plate_b64, (tex) => {
          // Same top-left-origin convention as loadProjectionTexture: the
          // projection shader flips V itself, so three.js must not.
          tex.flipY = false;
          tex.colorSpace = THREE.SRGBColorSpace;
          // NO depth cull on clean-plate surfaces: primary_depth marks
          // everything behind the foreground as occluded, which is EXACTLY
          // where the clean plate must show — culling it painted those
          // regions black (found live, first clean-plate composite).
          const mat = makeProjectionMaterial(data, tex, { primaryDepthTexture: null });
          let used = false;
          scene.traverse((c) => {
            if (!wantsClean(c)) return;
            const stale = c.userData._projMaterial;
            c.userData._projMaterial = mat;
            used = true;
            if (stale && stale !== mat) {
              stale.uniforms?.uTexture?.value?.dispose?.();
              stale.dispose?.();
            }
          });
          if (!used) { tex.dispose(); mat.dispose?.(); return; }
          if (projectionOn) applyProjection(true);
        });
      };

      // The primary relief mesh's OWN silhouette matte, when AtlasDeriveReliefMesh
      // shipped one. Patch layers have always had `mask_b64`; the primary — the
      // layer whose grid-quantized skyline is the visible staircase — never did,
      // so uHasMatte was 0 on every path through this file.
      const primaryMatteEntry = (data.proxy_geometry || []).find(
        (e) => e.type === "mesh" && e.metadata?.source === "depth_relief_mesh"
          && e.silhouette_matte_b64);
      const primaryMatteB64 = primaryMatteEntry?.silhouette_matte_b64 || "";
      const primaryMatteSoft =
        primaryMatteEntry?.silhouette_matte_mode === "soft";
      // The relief mesh records its own ribbon smudge width; read it from the
      // relief entry rather than the matte entry, which only exists when a
      // matte was requested.
      const primaryReliefEntry = (data.proxy_geometry || []).find(
        (e) => e.type === "mesh" && e.metadata?.source === "depth_relief_mesh");
      const primaryRibbonSmudge = primaryReliefEntry?.metadata?.ribbon_smudge_px;

      const buildPrimaryMat = (dTex) => {
        loadProjectionTexture(data, (tex) => {
          loadMatteFromB64(primaryMatteB64, (matteTexture) => {
            const old = projMaterial;
            projMaterial = makeProjectionMaterial(data, tex, {
              primaryDepthTexture: dTex, matteTexture,
              matteSoft: primaryMatteSoft,
              ribbonSmudgePx: primaryRibbonSmudge,
            });
            if (projectionOn) applyProjection(true);
            if (old) { old.uniforms?.uTexture?.value?.dispose?.(); old.dispose(); }
          });
        });
        buildDrawnMat(dTex);
        buildBackdropMat(dTex);
      };
      
      if (data.primary_depth_b64) {
        new THREE.TextureLoader().load(data.primary_depth_b64, (dTex) => {
          // Packed metric depth is DATA, not colour. It shares the projection
          // shader's top-left UV convention and must never be sRGB/OCIO decoded.
          dTex.flipY = false;
          dTex.colorSpace = THREE.NoColorSpace;
          dTex.magFilter = THREE.NearestFilter;
          dTex.minFilter = THREE.NearestFilter;
          dTex.needsUpdate = true;
          buildPrimaryMat(dTex);
        });
      } else {
        buildPrimaryMat(null);
      }
      
      // Multi-angle patch sources: each builds its own geometry + a projection
      // material (bound to its camera+image, facing-masked) that layers over
      // the primary to fill areas the primary camera couldn't see.
      buildPatchSources(scene, data, () => { if (projectionOn) applyProjection(true); });
      if (projectionOn) applyProjection(true); // grey until textures arrive
      buildBandBox(); // rebuild the 📏 overlay against this execution's geometry
      placeDefaultLights(); // relight lights follow the (now-built) geometry + scale
    },
    setBackground(imgBase64) {
      // Background source-photo backplate REMOVED (user request 2026-07-23):
      // the enlarged edge-smeared photo plane only ever showed in grey
      // (Project OFF) mode and was redundant against the layer meshes + grid
      // (and not what the 🎬 Backdrop button toggles — that is the
      // projection_backdrop geometry plane). Tear down any existing plane and
      // never build one; bgMesh stays null so every `if (bgMesh)` guard elsewhere
      // no-ops. Project mode is unaffected (the plane was already hidden there).
      if (bgMesh) {
        scene.remove(bgMesh);
        bgMesh.geometry.dispose();
        bgMesh.material.map?.dispose();
        bgMesh.material.dispose();
        bgMesh = null;
      }
      node._atlasBgMesh = null;
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
      // Belt-and-braces for frontends whose addDOMWidget cleanup semantics
      // differ: removing an already-detached element is a no-op.
      container.remove();
    };
  },
});
