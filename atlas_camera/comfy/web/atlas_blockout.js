/**
 * Atlas Blockout Viewport — ComfyUI frontend extension
 *
 * Embeds a Three.js 3D scene inside the AtlasBlockoutViewport node.
 * On node execution the recovered camera is fetched from /atlas/camera_data/{nodeId}
 * and applied to the Three.js camera so the scene is pre-aligned to the source photo.
 *
 * The user places primitive geometry (Box, Plane, Cylinder, Person Card), then
 * clicks "Render Passes" to produce shaded / depth / normal / mask images that
 * are base64-encoded into the client_data STRING widget and sent back to Python.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ---------------------------------------------------------------------------
// Three.js — loaded from CDN so there is no build step needed.
// ComfyUI ships its own Three.js; we try to import it first.
// ---------------------------------------------------------------------------
let THREE;
let OBJLoader;

async function loadThree() {
  if (THREE) return;
  try {
    // ComfyUI bundles Three.js at this path in recent versions
    const mod = await import("../../lib/three.module.js");
    THREE = mod;
  } catch (_) {
    try {
      const mod = await import(
        "https://unpkg.com/three@0.163.0/build/three.module.js"
      );
      THREE = mod;
    } catch (e) {
      console.error("[AtlasBlockout] Failed to load Three.js:", e);
    }
  }
  try {
    const objMod = await import(
      "https://unpkg.com/three@0.163.0/examples/jsm/loaders/OBJLoader.js"
    );
    OBJLoader = objMod.OBJLoader;
  } catch (e) {
    console.warn("[AtlasBlockout] OBJLoader unavailable; proxy models disabled:", e);
    OBJLoader = null;
  }
}

// ---------------------------------------------------------------------------
// Scale-reference proxy meshes (examples/models/*.obj, served by Python).
// Files are authored in centimetres, so we scale by 0.01 into the metric world
// that the recovered camera + ground plane live in — a correctly-sized human or
// car is the fastest visual check that the solve and camera height are right.
// ---------------------------------------------------------------------------
const PROXY_COLORS = { woman: 0xffddbb, sedan: 0x6688aa };

// Ground point (Y=0) under the camera's view centre, so the proxy lands where the
// camera is looking rather than at an arbitrary spot.
function groundPointInView(camera) {
  const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  const p = camera.position;
  if (dir.y < -1e-3) {
    const t = -p.y / dir.y;
    return new THREE.Vector3(p.x + t * dir.x, 0, p.z + t * dir.z);
  }
  return new THREE.Vector3(p.x, 0, p.z - 3);
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
    dragging = true;
    panning = e.button === 2 || e.shiftKey;
    lx = e.clientX; ly = e.clientY;
    dom.style.cursor = "grabbing";
    e.preventDefault();
  }
  function onUp() { dragging = false; dom.style.cursor = "grab"; }
  function onMove(e) {
    if (!dragging) return;
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
    apply();
  }
  function onWheel(e) {
    sph.radius = Math.max(0.05, sph.radius * (1 + Math.sign(e.deltaY) * 0.1));
    apply();
    e.preventDefault();
  }
  dom.addEventListener("mousedown", onDown);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  dom.addEventListener("wheel", onWheel, { passive: false });
  dom.addEventListener("contextmenu", (e) => e.preventDefault());
  return {
    target,
    setTarget(v) { target.copy(v); },
    syncFromCamera,
    dispose() {
      dom.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      dom.removeEventListener("wheel", onWheel);
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
// ---------------------------------------------------------------------------
const PROJECTION_VERTEX_SHADER = `
  uniform mat4 uAtlasViewMatrix;
  uniform float uFx;
  uniform float uFy;
  uniform float uCx;
  uniform float uCy;
  varying vec2 vImagePx;
  varying float vCamZ;
  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
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

const PROJECTION_FRAGMENT_SHADER = `
  uniform sampler2D uTexture;
  uniform vec2 uImageSize;
  uniform float uOpacity;
  varying vec2 vImagePx;
  varying float vCamZ;
  void main() {
    if (vCamZ >= 0.0) discard;                    // behind the recovered camera
    vec2 uv = vImagePx / uImageSize;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) discard;
    vec4 col = texture2D(uTexture, uv);
    gl_FragColor = vec4(col.rgb, col.a * uOpacity);
  }
`;

function makeProjectionMaterial(data, texture) {
  const flat = data.view_matrix.flat();
  const vm = new THREE.Matrix4();
  vm.set(
    flat[0], flat[1], flat[2], flat[3],
    flat[4], flat[5], flat[6], flat[7],
    flat[8], flat[9], flat[10], flat[11],
    flat[12], flat[13], flat[14], flat[15]
  );
  return new THREE.ShaderMaterial({
    uniforms: {
      uAtlasViewMatrix: { value: vm },
      uFx: { value: data.fx || 1 },
      uFy: { value: data.fy || data.fx || 1 },
      uCx: { value: data.cx ?? (data.image_width || 1) / 2 },
      uCy: { value: data.cy ?? (data.image_height || 1) / 2 },
      uTexture: { value: texture },
      uImageSize: { value: new THREE.Vector2(data.image_width || 1, data.image_height || 1) },
      uOpacity: { value: 1.0 },
    },
    vertexShader: PROJECTION_VERTEX_SHADER,
    fragmentShader: PROJECTION_FRAGMENT_SHADER,
    side: THREE.DoubleSide,
    transparent: false,
    depthWrite: true,
    depthTest: true,
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
    group.add(mesh);
  }
  scene.add(group);
  return group;
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

  // FOV from fy and image height
  const imageH = data.image_height || 1080;
  const fy = data.fy || 1;
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

  // Helper: render scene to RT, read pixels, return base64
  function renderToBase64(overrideMat) {
    if (overrideMat) scene.overrideMaterial = overrideMat;
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    scene.overrideMaterial = null;

    const buffer = new Uint8Array(width * height * 4);
    renderer.readRenderTargetPixels(rt, 0, 0, width, height, buffer);

    // Flip Y (WebGL origin is bottom-left, canvas is top-left)
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

  try {
    // Shaded: standard PBR render (or the projection material if 📽 is on)
    const shadedB64 = renderToBase64(null);

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

  // Render dimensions. These start square and are resized to the source image's
  // aspect on execution (resizeViewport), so the viewport inherits the image aspect.
  node._atlasW = node._atlasW || node._atlasResolution || 768;
  node._atlasH = node._atlasH || node._atlasResolution || 768;
  let W = node._atlasW, H = node._atlasH;

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
  const canvasWrap = document.createElement("div");
  canvasWrap.style.cssText = "position:relative;width:100%;line-height:0;";

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  canvas.style.cssText = "display:block;width:100%;height:auto;background:#111;cursor:grab";

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

  canvasWrap.append(canvas, diagramSvg, metaHud);

  // Three.js setup
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setSize(W, H, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  // Exposure only has a visible effect with a tone-mapping operator active.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;

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

  // Animation loop — assign to node._atlasRafId each frame so cancelAnimationFrame works.
  // The orbit controller updates the camera on input events, so we only render here.
  function animate() {
    node._atlasRafId = requestAnimationFrame(animate);
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
    if (c.userData?.atlasDerived || c.userData?.atlasUserGeo) return true;
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
      if (on && projMaterial) {
        // Stash the ORIGINAL material only once — re-applying with a rebuilt
        // projection material must not overwrite it with the stale one.
        if (!c.userData._prevMaterial) c.userData._prevMaterial = c.material;
        c.material = projMaterial;
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

  // Render Passes button
  const renderBtn = document.createElement("button");
  renderBtn.textContent = "⬛ Render Passes";
  renderBtn.style.cssText = "padding:3px 10px;font-size:11px;cursor:pointer;background:#1a3a1a;color:#afa;border:1px solid #464;border-radius:3px;margin-left:auto";
  renderBtn.onclick = async () => {
    renderBtn.disabled = true;
    renderBtn.textContent = "Rendering...";
    try {
      const passes = await renderAllPasses(renderer, scene, camera, W, H, [bgMesh]);
      if (!passes) return;
      // Write to client_data widget
      const widget = node.widgets?.find((w) => w.name === "client_data");
      if (widget) {
        widget.value = JSON.stringify(passes);
        widget.callback?.(widget.value);
      }
      // Re-queue the prompt so Python receives the frames
      app.queuePrompt(0, 1);
    } finally {
      renderBtn.disabled = false;
      renderBtn.textContent = "⬛ Render Passes";
    }
  };
  toolbar.appendChild(renderBtn);

  // Assemble
  containerEl.appendChild(toolbar);
  containerEl.appendChild(canvasWrap);

  // Store refs for cleanup and camera application
  node._atlasRenderer = renderer;
  node._atlasScene = scene;
  node._atlasCamera = camera;
  node._atlasControls = controls;
  node._atlasBgMesh = null;
  node._atlasW = W;
  node._atlasH = H;

  // Resize the render target + canvas so the viewport matches the source image
  // aspect (target_width/target_height come from the Python node, derived from the
  // incoming image). Keeps the camera aspect and canvas aspect in sync.
  function resizeViewport(w, h) {
    w = Math.max(16, Math.round(w || W));
    h = Math.max(16, Math.round(h || H));
    W = w; H = h;
    node._atlasW = w; node._atlasH = h;
    canvas.width = w; canvas.height = h;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  // Apply the recovered camera and initialise the orbit controller *from* it, so
  // the default view is the camera's own perspective (matching the source photo).
  function applyRecoveredView(data) {
    if (data.target_width && data.target_height) {
      resizeViewport(data.target_width, data.target_height);
    }
    applyRecoveredCamera(camera, data);
    controls.setTarget(groundPointInView(camera)); // pivot on the looked-at ground point
    controls.syncFromCamera();                     // init orbit state from recovered pose
    recoveredData = data;
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
      loadProjectionTexture(data, (tex) => {
        const old = projMaterial;
        projMaterial = makeProjectionMaterial(data, tex);
        if (projectionOn) applyProjection(true);
        if (old) { old.uniforms?.uTexture?.value?.dispose?.(); old.dispose(); }
      });
      if (projectionOn) applyProjection(true); // grey until texture arrives
    },
    setBackground(imgBase64) {
      if (!imgBase64 || !THREE) return;
      if (bgMesh) { scene.remove(bgMesh); bgMesh.geometry.dispose(); bgMesh.material.dispose(); }
      const loader = new THREE.TextureLoader();
      loader.load(imgBase64, (tex) => {
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
// ComfyUI extension registration
// ---------------------------------------------------------------------------
app.registerExtension({
  name: "AtlasCamera.Blockout",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AtlasBlockoutViewport") return;
    await loadThree();
  },

  async nodeCreated(node) {
    if (node.comfyClass !== "AtlasBlockoutViewport") return;
    await loadThree();
    if (!THREE) return;

    // Wait one tick for ComfyUI to finish building the node DOM
    await new Promise((r) => setTimeout(r, 0));

    // Read the long-edge resolution widget (W×H is derived from the source image
    // aspect on execution, so the viewport inherits the image's aspect).
    const resWidget = node.widgets?.find((w) => w.name === "resolution");
    node._atlasResolution = resWidget?.value ?? 768;

    // Create a DOM container widget
    const container = document.createElement("div");
    container.style.cssText = "width:100%;display:flex;flex-direction:column;gap:0;";

    const domWidget = node.addDOMWidget("atlas_viewport", "div", container, {
      serialize: false,
      getValue() { return null; },
      setValue() {},
    });

    const ui = buildNodeUI(node, container);

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
      cancelAnimationFrame(node._atlasRafId);
      node._atlasRenderer?.dispose();
      node._atlasControls?.dispose();
    };
  },
});
