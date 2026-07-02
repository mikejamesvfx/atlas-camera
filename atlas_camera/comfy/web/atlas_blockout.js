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

// ---------------------------------------------------------------------------
// Three.js — loaded from CDN so there is no build step needed.
// ComfyUI ships its own Three.js; we try to import it first.
// ---------------------------------------------------------------------------
let THREE;
let OrbitControls;

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
    const ocMod = await import(
      "https://unpkg.com/three@0.163.0/examples/jsm/controls/OrbitControls.js"
    );
    OrbitControls = ocMod.OrbitControls;
  } catch (_) {
    OrbitControls = null;
  }
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
async function renderAllPasses(renderer, scene, camera, width, height) {
  if (!THREE) return null;
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

  // Shaded: standard PBR render
  const shadedB64 = renderToBase64(null);

  // Depth: MeshDepthMaterial
  const depthMat = new THREE.MeshDepthMaterial({ depthPacking: THREE.BasicDepthPacking });
  const depthB64 = renderToBase64(depthMat);
  depthMat.dispose();

  // Normal: custom ShaderMaterial
  const normalMat = new THREE.MeshNormalMaterial();
  const normalB64 = renderToBase64(normalMat);
  normalMat.dispose();

  // Mask: white geometry, black background
  const bg = scene.background;
  scene.background = new THREE.Color(0x000000);
  const maskMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const maskB64 = renderToBase64(maskMat);
  scene.background = bg;
  maskMat.dispose();

  rt.dispose();
  return { shaded: shadedB64, depth: depthB64, normal: normalB64, mask: maskB64 };
}

// ---------------------------------------------------------------------------
// Build the in-node UI (canvas + toolbar)
// ---------------------------------------------------------------------------
function buildNodeUI(node, containerEl) {
  if (!THREE) {
    containerEl.innerHTML = "<p style='color:#f88;padding:8px'>Three.js not available</p>";
    return;
  }

  const W = node._atlasWidth || 512;
  const H = node._atlasHeight || 512;

  // Toolbar
  const toolbar = document.createElement("div");
  toolbar.style.cssText = "display:flex;gap:4px;padding:4px;background:#1a1a1a;flex-wrap:wrap";

  const primitives = [
    { label: "Box", type: "box" },
    { label: "Plane", type: "plane" },
    { label: "Cylinder", type: "cylinder" },
    { label: "Person", type: "person" },
  ];

  // Canvas
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  canvas.style.cssText = `display:block;width:100%;max-height:${H}px;object-fit:contain;background:#111;cursor:grab`;

  // Three.js setup
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setSize(W, H, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

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

  // Ground grid
  const grid = new THREE.GridHelper(20, 20, 0x444444, 0x333333);
  scene.add(grid);

  // OrbitControls
  let controls = null;
  if (OrbitControls) {
    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 1, 0);
  }

  // Background reference image (loaded after camera data is set)
  let bgMesh = null;

  // Animation loop — assign to node._atlasRafId each frame so cancelAnimationFrame works
  function animate() {
    node._atlasRafId = requestAnimationFrame(animate);
    controls?.update();
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
      if (mesh) scene.add(mesh);
    };
    toolbar.appendChild(btn);
  });

  // Clear button
  const clearBtn = document.createElement("button");
  clearBtn.textContent = "Clear";
  clearBtn.style.cssText = "padding:3px 8px;font-size:11px;cursor:pointer;background:#3a1a1a;color:#faa;border:1px solid #644;border-radius:3px";
  clearBtn.onclick = () => {
    const toRemove = scene.children.filter(
      (c) => c.isMesh && c !== bgMesh
    );
    toRemove.forEach((c) => { scene.remove(c); c.geometry?.dispose(); c.material?.dispose(); });
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
      const passes = await renderAllPasses(renderer, scene, camera, W, H);
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
  containerEl.appendChild(canvas);

  // Store refs for cleanup and camera application
  node._atlasRenderer = renderer;
  node._atlasScene = scene;
  node._atlasCamera = camera;
  node._atlasControls = controls;
  node._atlasBgMesh = null;
  node._atlasW = W;
  node._atlasH = H;

  // Return setter so caller can apply camera and background image
  return {
    applyCamera(data) {
      applyRecoveredCamera(camera, data);
      controls?.target.set(
        data.camera_position?.[0] ?? 0,
        0,
        data.camera_position?.[2] ?? 0
      );
      controls?.update();
    },
    setBackground(imgBase64) {
      if (!imgBase64 || !THREE) return;
      if (bgMesh) { scene.remove(bgMesh); bgMesh.geometry.dispose(); bgMesh.material.dispose(); }
      const loader = new THREE.TextureLoader();
      loader.load(imgBase64, (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        const iw = tex.image.width || W;
        const ih = tex.image.height || H;
        const aspect = iw / ih;
        const pw = 4.6;
        const ph = pw / aspect;
        const geo = new THREE.PlaneGeometry(pw, ph);
        const mat = new THREE.MeshBasicMaterial({ map: tex, depthWrite: false, depthTest: false });
        bgMesh = new THREE.Mesh(geo, mat);
        bgMesh.renderOrder = -1;
        bgMesh.position.set(0, ph * 0.46 + 0.18, -3.85);
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

    // Read width/height from widgets
    const wWidget = node.widgets?.find((w) => w.name === "width");
    const hWidget = node.widgets?.find((w) => w.name === "height");
    node._atlasWidth = wWidget?.value ?? 512;
    node._atlasHeight = hWidget?.value ?? 512;

    // Create a DOM container widget
    const container = document.createElement("div");
    container.style.cssText = "width:100%;display:flex;flex-direction:column;gap:0;";

    const domWidget = node.addDOMWidget("atlas_viewport", "div", container, {
      serialize: false,
      getValue() { return null; },
      setValue() {},
    });

    const ui = buildNodeUI(node, container);

    // On node execution complete: apply recovered camera + source image
    node.onExecuted = async (_outputData) => {
      const cameraData = await fetchCameraData(String(node.id));
      if (!cameraData) return;
      ui?.applyCamera(cameraData);
      if (cameraData.source_image_b64) {
        ui?.setBackground(cameraData.source_image_b64);
      }
    };

    // Sync width/height widget changes to Three.js
    if (wWidget) wWidget.callback = (v) => { node._atlasWidth = v; };
    if (hWidget) hWidget.callback = (v) => { node._atlasHeight = v; };

    // Cleanup on node removal
    node.onRemoved = () => {
      cancelAnimationFrame(node._atlasRafId);
      node._atlasRenderer?.dispose();
      node._atlasControls?.dispose();
    };
  },
});
