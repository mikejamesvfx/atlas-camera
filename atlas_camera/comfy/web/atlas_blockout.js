/**
 * Atlas Viewport — ComfyUI frontend extension
 *
 * Embeds a Three.js 3D scene inside the AtlasBlockoutViewport node.
 * On node execution the recovered camera is fetched from /atlas/camera_data/{nodeId}
 * and applied to the Three.js camera so the scene is pre-aligned to the source photo.
 *
 * The user places primitive geometry (Box, Plane, Cylinder, Person Card), then
 * clicks "Render Proxy Passes" to produce proxy/LDR shaded / depth / normal /
 * mask images that are base64-encoded into the client_data STRING widget and
 * sent back to Python.
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
const PROXY_COLORS = { woman: 0xffddbb, sedan: 0x6688aa };
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
  const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));

  function syncFromCamera() {
    const off = camera.position.clone().sub(target);
    sph.radius = Math.max(0.01, off.length());
    sph.theta = Math.atan2(off.x, off.z);
    sph.phi = Math.acos(Math.min(1, Math.max(-1, off.y / sph.radius)));
    theta0 = sph.theta;
    phi0 = sph.phi;
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
  }
  function onDown(e) {
    if (!enabled) return;
    dragging = true;
    panning = e.button === 2 || e.shiftKey;
    lx = e.clientX; ly = e.clientY;
    dom.style.cursor = "grabbing";
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
      sph.theta = theta0 + Math.min(MAX_YAW, Math.max(-MAX_YAW, deltaTheta));

      const rawPhi = Math.min(Math.PI - 0.05, Math.max(0.05, sph.phi - dy * 0.005));
      sph.phi = Math.min(phi0 + MAX_PITCH, Math.max(phi0 - MAX_PITCH, rawPhi));
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
  dom.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); });
  return {
    target,
    setTarget(v) { target.copy(v); },
    syncFromCamera,
    setEnabled(v) { enabled = v; if (!v) dragging = false; dom.style.cursor = v ? "grab" : "default"; },
    dispose() {
      dom.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      dom.removeEventListener("wheel", onWheel);
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

function loadProxyModel(scene, modelName, camera) {
  if (!THREE || !OBJLoader) {
    console.warn("[AtlasBlockout] OBJLoader not ready; cannot add proxy:", modelName);
    return;
  }
  const loader = new OBJLoader();
  loader.load(
    `/atlas/proxy_model/${modelName}.obj`,
    (obj) => {
      obj.scale.setScalar(0.01); // centimetres -> metres
      const mat = new THREE.MeshStandardMaterial({
        color: PROXY_COLORS[modelName] ?? 0xaaaaaa,
        roughness: 0.75,
      });
      obj.traverse((c) => { if (c.isMesh) c.material = mat; });
      const g = groundPointInView(camera);
      obj.position.set(g.x, 0, g.z);
      // Yaw the proxy to face the camera (cosmetic; keeps it upright).
      obj.rotation.y = Math.atan2(camera.position.x - g.x, camera.position.z - g.z);
      obj.userData.atlasProxy = true;
      scene.add(obj);
    },
    undefined,
    (err) => console.error("[AtlasBlockout] Failed to load proxy", modelName, err)
  );
}

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
  varying vec2 vImagePx;
  varying float vCamZ;
  varying vec3 vWorldPos;
  varying vec3 vWorldNormal;
  float atlasRelightTerm(vec3 lightPos, vec3 lightColor, float intensity, vec3 worldPos, vec3 worldNormal) {
    if (intensity <= 0.0) return 0.0;
    vec3 toLight = lightPos - worldPos;
    float dist = length(toLight);
    float ndotl = max(dot(normalize(worldNormal), normalize(toLight)), 0.0);
    float atten = 1.0 / (1.0 + 0.05 * dist * dist);
    return intensity * ndotl * atten;
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
    vec3 toCam = normalize(uCamPos - vWorldPos);
    float facing = abs(dot(normalize(vWorldNormal), toCam));
    if (facing < uFacingThreshold) discard;       // too grazing for this projector
    vec4 col = texture2D(uTexture, uv);
    vec3 relight = vec3(1.0)
      + uLight1Color * atlasRelightTerm(uLight1Pos, uLight1Color, uLight1Intensity, vWorldPos, vWorldNormal)
      + uLight2Color * atlasRelightTerm(uLight2Pos, uLight2Color, uLight2Intensity, vWorldPos, vWorldNormal);
    vec3 outColor = atlasLinearToSRGB(clamp(col.rgb * relight, 0.0, 1.0));
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
  return mat;
}

function loadTextureFromB64(b64, cb) {
  if (!b64) return;
  const loader = new THREE.TextureLoader();
  loader.load(b64, (tex) => {
    tex.flipY = false;                // shader UV origin is top-left
    tex.colorSpace = THREE.SRGBColorSpace;
    cb(tex);
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
      if (pm) { pm.uniforms?.uTexture?.value?.dispose?.(); pm.dispose?.(); }
    });
    scene.remove(g);
  }

  const sources = data.projection_sources || [];
  sources.forEach((src, idx) => {
    const group = new THREE.Group();
    group.name = `atlas_patch_${idx}`;
    group.userData.atlasPatchGroup = true;
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
      const patchMat = makeProjectionMaterial(src, tex, { facingThreshold, priority: src.priority });
      for (const m of meshes) {
        const prev = m.userData._projMaterial;
        if (prev && prev !== patchMat) {
          prev.uniforms?.uTexture?.value?.dispose?.();
          prev.dispose?.();
        }
        m.userData._projMaterial = patchMat;
      }
      if (typeof onSourceReady === "function") onSourceReady();
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
function createPrimitive(type) {
  if (!THREE) return null;
  let geometry, material;
  const mat = new THREE.MeshStandardMaterial({ color: 0xaaaaaa, roughness: 0.8 });

  switch (type) {
    case "box":
      geometry = new THREE.BoxGeometry(1, 1, 1);
      material = new THREE.MeshStandardMaterial({ color: 0x8899bb, roughness: 0.7 });
      break;
    case "plane":
      geometry = new THREE.PlaneGeometry(4, 4);
      material = new THREE.MeshStandardMaterial({ color: 0x88aa88, roughness: 0.9, side: THREE.DoubleSide });
      break;
    case "cylinder":
      geometry = new THREE.CylinderGeometry(0.5, 0.5, 1, 16);
      material = new THREE.MeshStandardMaterial({ color: 0xbbaa88, roughness: 0.7 });
      break;
    case "person":
      // 0.55×1.75×0.02 card (matches Atlas proxy defaults)
      geometry = new THREE.BoxGeometry(0.55, 1.75, 0.02);
      material = new THREE.MeshStandardMaterial({ color: 0xffddbb, roughness: 0.6 });
      break;
    default:
      geometry = new THREE.BoxGeometry(1, 1, 1);
      material = mat;
  }

  const mesh = new THREE.Mesh(geometry, material);
  // Position in front of camera at ground level
  mesh.position.set(0, (type === "plane" ? 0 : 0.5), -3);
  if (type === "plane") mesh.rotation.x = -Math.PI / 2;
  return mesh;
}

function atlasReadRenderTargetAsBase64(renderer, renderTarget, width, height) {
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
  return offscreen.toDataURL("image/png").split(",")[1];
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
    return atlasReadRenderTargetAsBase64(renderer, renderTarget, width, height);
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

  const primitives = [
    { label: "Box", type: "box" },
    { label: "Plane", type: "plane" },
    { label: "Cylinder", type: "cylinder" },
    { label: "Person", type: "person" },
  ];

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
  ];
  movableLights[0].position.set(2, 3, 2);
  movableLights[1].position.set(-2, 3, -2);
  movableLights.forEach((l) => scene.add(l));
  let _lightsWereActive = false;
  function syncProjectionLightUniforms() {
    const active = movableLights.some((l) => l.intensity > 0);
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
        mat.uniforms[`uLight${n}Pos`].value.copy(l.position);
        mat.uniforms[`uLight${n}Color`].value.copy(l.color);
        mat.uniforms[`uLight${n}Intensity`].value = l.intensity;
      });
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

  // Primitive buttons
  primitives.forEach(({ label, type }) => {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:3px";
    btn.onclick = () => {
      const mesh = createPrimitive(type);
      if (mesh) {
        mesh.userData.atlasUserGeo = true;
        scene.add(mesh);
        if (projectionOn) applyProjection(true); // new geometry catches the projection
      }
    };
    toolbar.appendChild(btn);
  });

  // Scale-reference proxy buttons (real, correctly-sized meshes on the ground).
  const proxies = [
    { label: "🧍 Woman", model: "woman" },
    { label: "🚗 Sedan", model: "sedan" },
  ];
  proxies.forEach(({ label, model }) => {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#22322a;color:#cfd;border:1px solid #465;border-radius:3px";
    btn.onclick = () => loadProxyModel(scene, model, camera);
    toolbar.appendChild(btn);
  });

  // Camera View button — snap the orbit camera back to the recovered perspective.
  const camBtn = document.createElement("button");
  camBtn.textContent = "📷 Camera View";
  camBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2f3a;color:#cde;border:1px solid #456;border-radius:3px";
  camBtn.onclick = () => { if (recoveredData) applyRecoveredView(recoveredData); };
  toolbar.appendChild(camBtn);

  // 📽 Project toggle — camera-project the source photo onto ALL geometry
  // (derived proxies, user primitives, OBJ proxies) from the recovered camera.
  let projectionOn = false;
  let projMaterial = null;
  const projBtn = document.createElement("button");
  projBtn.textContent = "📽 Project";
  projBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#2a2a3a;color:#dcf;border:1px solid #546;border-radius:3px";

  function isProjectable(c) {
    if (!c.isMesh || c === bgMesh) return false;
    if (c.userData?.atlasDerived || c.userData?.atlasUserGeo || c.userData?.atlasPatch) return true;
    // OBJ-proxy children: any ancestor tagged atlasProxy.
    let p = c.parent;
    while (p) {
      if (p.userData?.atlasProxy) return true;
      p = p.parent;
    }
    return false;
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
    // The projection IS the image now — the floating background photo plane
    // only duplicates/confuses projected views.
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
      onDone: () => { if (recoveredData) applyRecoveredView(recoveredData); },
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
        frames.push(atlasRenderSceneToBase64(renderer, scene, camera, W, H, { renderTarget: outputRt }));
      }
      const widget = node.widgets?.find((w) => w.name === "client_data");
      let existing = {};
      try { existing = widget?.value ? JSON.parse(widget.value) : {}; } catch (_) { existing = {}; }
      existing.path_frames = frames;
      existing.camera_path = { keyframes: pathKeyframes.map(kfToJSON), fps: pathFps, frame_count: pathFrameCount };
      existing.atlas_proxy_path = {
        transport: "png_base64_proxy_ldr",
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

  const lightPanel = document.createElement("div");
  lightPanel.style.cssText = "display:none;flex-wrap:wrap;align-items:center;gap:10px;padding:4px 6px;background:#181818;border-top:1px solid #333;font-size:11px;color:#ccc";
  movableLights.forEach((light, idx) => {
    const group = document.createElement("span");
    group.style.cssText = "display:inline-flex;align-items:center;gap:4px;";
    const label = document.createElement("span");
    label.textContent = `Light ${idx + 1}`;
    label.style.cssText = "color:#ddd;font-weight:600;";
    group.appendChild(label);
    ["x", "y", "z"].forEach((axis) => {
      const axisLabel = document.createElement("span");
      axisLabel.textContent = axis.toUpperCase();
      axisLabel.style.cssText = "color:#888;";
      const input = document.createElement("input");
      input.type = "number";
      input.step = "0.1";
      input.value = light.position[axis].toFixed(1);
      input.style.cssText = "width:52px;background:#1e1e1e;color:#ddd;border:1px solid #444;border-radius:3px;padding:1px 3px;";
      input.oninput = () => { light.position[axis] = parseFloat(input.value) || 0; };
      group.append(axisLabel, input);
    });
    const intLabel = document.createElement("span");
    intLabel.textContent = "Intensity";
    intLabel.style.cssText = "color:#888;margin-left:4px;";
    const intSlider = document.createElement("input");
    intSlider.type = "range"; intSlider.min = "0"; intSlider.max = "5"; intSlider.step = "0.05"; intSlider.value = "0";
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

  // Clear button
  const clearBtn = document.createElement("button");
  clearBtn.textContent = "Clear";
  clearBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#3a1a1a;color:#faa;border:1px solid #644;border-radius:3px";
  clearBtn.onclick = () => {
    // Remove user geometry and OBJ proxies; derived projection proxies are
    // Python-owned and regenerate on execution, so Clear leaves them alone.
    const toRemove = scene.children.filter(
      (c) => (c.isMesh || c.userData?.atlasProxy) && c !== bgMesh
        && !c.userData?.atlasDerivedGroup
    );
    toRemove.forEach((c) => {
      scene.remove(c);
      // Dispose meshes (including those inside a loaded OBJ group). Never
      // dispose the shared projection material.
      c.traverse?.((m) => {
        m.geometry?.dispose?.();
        if (m.material !== projMaterial) m.material?.dispose?.();
      });
      c.geometry?.dispose?.();
      if (c.material !== projMaterial) c.material?.dispose?.();
    });
  };
  toolbar.appendChild(clearBtn);

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
  function applyRecoveredView(data) {
    if (data.target_width && data.target_height) {
      resizeViewport(data.target_width, data.target_height);
    }
    applyRecoveredCamera(camera, data);
    // Prefer the solved scene depth (when a derive-geometry node ran) over the
    // generic 30m default so the orbit radius matches this scene's actual scale.
    const sceneDepth = data.camera_meta?.scene_depth_m;
    const pivotMax = sceneDepth ? sceneDepth * 1.5 : 30;
    controls.setTarget(groundPointInView(camera, pivotMax)); // pivot on the looked-at ground point
    controls.syncFromCamera();                     // init orbit state from recovered pose
    recoveredData = data;
    applyOutputProfilePreview(data.output_profile || atlasOutputProfileFromWidgets(getLinkedControlsNode(node) || {}));
    updateLinkedOutputDesk(data);
  }

  // Bounding-box centroid of the DERIVED geometry (relief mesh and/or fitted
  // primitives) — excludes "projection_backdrop" (the always-emitted flat
  // catch-all far plane, same one 🎬 Backdrop toggles; see its comment
  // above), since including that would drag the centroid out toward the
  // frustum's far edge instead of the actual reconstructed subject. Called
  // once buildDerivedProxies has real geometry to measure (setProxies,
  // below) to REPLACE applyRecoveredView's ground-point-in-view fallback —
  // that fallback is a generic heuristic (where the camera's forward ray
  // crosses Y=0) that only coincidentally matches the geometry's actual
  // centre; this is exact. Returns null when there's no derived geometry yet
  // (e.g. the workflow never ran AtlasDeriveProjectionGeometry), in which
  // case the ground-point fallback is left in place.
  function computeGeometryPivot() {
    const group = scene.getObjectByName("atlas_derived_proxies");
    if (!group?.children?.length) return null;
    const box = new THREE.Box3();
    let any = false;
    group.children.forEach((mesh) => {
      if (mesh.name === "projection_backdrop") return;
      box.expandByObject(mesh);
      any = true;
    });
    return any && !box.isEmpty() ? box.getCenter(new THREE.Vector3()) : null;
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
      const geometryPivot = computeGeometryPivot();
      if (geometryPivot) {
        controls.setTarget(geometryPivot);
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
        const geo = new THREE.PlaneGeometry(pw, ph);
        const mat = new THREE.MeshBasicMaterial({ map: tex, depthWrite: false, depthTest: false });
        bgMesh = new THREE.Mesh(geo, mat);
        bgMesh.renderOrder = -1;
        const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
        bgMesh.position.copy(camera.position).addScaledVector(fwd, D);
        bgMesh.quaternion.copy(camera.quaternion);
        bgMesh.visible = !projectionOn; // hidden while 📽 Project is active
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
    look: atlasWidgetValue(node, "look", "None"),
    lut_path: atlasWidgetValue(node, "lut_path", ""),
    exposure: Number(atlasWidgetValue(node, "exposure", 0)) || 0,
    gamma: Number(atlasWidgetValue(node, "gamma", 1)) || 1,
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
  addColorField("Look", "look");
  addColorField("LUT", "lut_path");
  addColorField("Exposure", "exposure", "number", { step: "0.1" });
  addColorField("Gamma", "gamma", "number", { step: "0.05", min: "0.1" });
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

  async nodeCreated(node) {
    if (node.comfyClass === "AtlasViewportControls") {
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

    // Cleanup on node removal
    node.onRemoved = () => {
      api.removeEventListener("executed", onApiExecuted);
      node._atlasSizeTraceCleanup?.();
      cancelAnimationFrame(node._atlasRafId);
      node._atlasRenderer?.dispose();
      node._atlasControls?.dispose();
      node._atlasFly?.dispose();
    };
  },
});
