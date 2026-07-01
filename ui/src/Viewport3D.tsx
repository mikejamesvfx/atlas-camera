import {
  Box,
  Camera,
  Eye,
  Film,
  Grid3X3,
  Image as ImageIcon,
  Layers3,
  Lock,
  ScanLine,
  Unlock
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import * as THREE from "three";
import { addProjectionGround } from "./ProjectionMaterial";
import type {
  CameraAnalysis,
  Constraints,
  Viewport3DMode,
  Viewport3DProxyObject,
  Viewport3DState
} from "./types";

type Viewport3DProps = {
  analysis: CameraAnalysis | null;
  constraints: Constraints;
  sourceUrl: string;
  solvePayload: any;
  state: Viewport3DState;
  selectedProxy: Viewport3DProxyObject | null;
  onDisplayChange: (display: Viewport3DState["display"]) => void;
  onSelectProxy: (id: string | null) => void;
};

type SceneReadout = {
  focalLength?: number;
  fovHorizontal?: number;
  cameraPosition: [number, number, number];
  proxyCount: number;
};

const modeOptions: Array<{ id: Viewport3DMode; label: string }> = [
  { id: "image_match", label: "Image" },
  { id: "perspective", label: "Orbit" },
  { id: "top", label: "Top" },
  { id: "front", label: "Front" },
  { id: "side", label: "Side" }
];

export function Viewport3D({
  analysis,
  constraints,
  sourceUrl,
  solvePayload,
  state,
  selectedProxy,
  onDisplayChange,
  onSelectProxy
}: Viewport3DProps) {
  return (
    <Viewport3DStage
      constraints={constraints}
      sourceUrl={sourceUrl}
      analysis={analysis}
      solvePayload={solvePayload}
      viewport={state}
      selectedProxyId={selectedProxy?.id ?? state.selected_proxy_id ?? null}
      onDisplayChange={onDisplayChange}
      onSelectProxy={onSelectProxy}
    />
  );
}

function Viewport3DStage({
  constraints,
  sourceUrl,
  analysis,
  solvePayload,
  viewport,
  selectedProxyId,
  onDisplayChange,
  onSelectProxy
}: {
  constraints: Constraints;
  sourceUrl: string;
  analysis: CameraAnalysis | null;
  solvePayload: any;
  viewport: Viewport3DState;
  selectedProxyId: string | null;
  onDisplayChange: (display: Viewport3DState["display"]) => void;
  onSelectProxy: (id: string | null) => void;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [rendererStatus, setRendererStatus] = useState("3D viewport ready");
  const readout = useMemo(
    () => buildSceneReadout(viewport, analysis, solvePayload),
    [analysis, solvePayload, viewport]
  );

  const updateDisplay = (display: Partial<Viewport3DState["display"]>) => {
    onDisplayChange({ ...viewport.display, ...display });
  };

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;
    setRendererStatus("3D viewport ready");

    let renderer: any;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    } catch {
      setRendererStatus("WebGL is unavailable in this view");
      return undefined;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#171411");
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.domElement.className = "viewport3d-canvas";
    mount.appendChild(renderer.domElement);

    const renderCamera = new THREE.PerspectiveCamera(46, 1, 0.05, 500);
    const target = new THREE.Vector3(0, 0.85, -2.2);
    const orbit = {
      yaw: viewport.display.active_mode === "side" ? Math.PI / 2 : 0.62,
      pitch: viewport.display.active_mode === "top" ? 1.34 : 0.48,
      distance: viewport.display.active_mode === "image_match" ? 7.1 : 6.2
    };

    const root = new THREE.Group();
    scene.add(root);
    scene.add(new THREE.HemisphereLight(0xf6eee4, 0x211a16, 2.4));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.4);
    keyLight.position.set(4, 6, 3);
    scene.add(keyLight);

    let projectionCleanup: (() => void) | null = null;
    if (viewport.display.show_projection && analysis && sourceUrl && constraints.image_width > 0 && constraints.image_height > 0) {
      projectionCleanup = addProjectionGround(
        root,
        analysis,
        constraints.image_width,
        constraints.image_height,
        sourceUrl,
        () => { renderer.render(scene, renderCamera); },
        (message) => { setRendererStatus(`Projection error: ${message}`); }
      );
    }

    if (viewport.display.show_image && sourceUrl) {
      addImagePlane(root, sourceUrl, constraints, viewport.display.image_opacity, () => {
        renderer.render(scene, renderCamera);
      });
    }
    if (viewport.display.show_grid) {
      root.add(createGroundGrid(viewport.display.grid_scale));
    }
    if (viewport.display.show_axes) {
      root.add(createAxisRig(2.1));
    }
    if (viewport.display.show_frustum) {
      root.add(createCameraFrustum(readout.cameraPosition, readout.fovHorizontal, constraints));
    }
    if (viewport.display.show_guides) {
      root.add(createGuideFamilies(constraints));
    }
    if (viewport.display.show_horizon) {
      root.add(createHorizonBand(solvePayload, constraints));
    }
    if (viewport.display.show_proxies) {
      for (const proxy of viewport.proxy_objects) {
        root.add(createProxyObject(proxy, proxy.id === selectedProxyId));
      }
    }

    const resize = () => {
      const bounds = mount.getBoundingClientRect();
      const width = Math.max(1, Math.floor(bounds.width));
      const height = Math.max(1, Math.floor(bounds.height));
      renderer.setSize(width, height, false);
      renderCamera.aspect = width / height;
      renderCamera.updateProjectionMatrix();
    };

    let frameId = 0;
    let dragging = false;
    let lastPointer: { x: number; y: number } | null = null;
    const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

    const applyView = () => {
      applyCameraMode(renderCamera, viewport.display.active_mode, viewport.display.lock_camera_to_view, target, orbit);
      renderer.render(scene, renderCamera);
    };

    const animate = () => {
      if (!reducedMotion && !dragging && viewport.display.active_mode === "perspective") {
        orbit.yaw += 0.0012;
      }
      applyView();
      frameId = window.requestAnimationFrame(animate);
    };

    const handlePointerDown = (event: PointerEvent) => {
      if (viewport.display.active_mode !== "perspective" && viewport.display.active_mode !== "image_match") return;
      dragging = true;
      lastPointer = { x: event.clientX, y: event.clientY };
      renderer.domElement.setPointerCapture(event.pointerId);
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (!dragging || !lastPointer) return;
      const dx = event.clientX - lastPointer.x;
      const dy = event.clientY - lastPointer.y;
      orbit.yaw -= dx * 0.006;
      orbit.pitch = clamp(orbit.pitch + dy * 0.004, -0.15, 1.42);
      lastPointer = { x: event.clientX, y: event.clientY };
    };

    const handlePointerUp = (event: PointerEvent) => {
      dragging = false;
      lastPointer = null;
      if (renderer.domElement.hasPointerCapture(event.pointerId)) {
        renderer.domElement.releasePointerCapture(event.pointerId);
      }
    };

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      orbit.distance = clamp(orbit.distance + event.deltaY * 0.006, 2.6, 18);
    };

    resize();
    renderer.domElement.addEventListener("pointerdown", handlePointerDown);
    renderer.domElement.addEventListener("pointermove", handlePointerMove);
    renderer.domElement.addEventListener("pointerup", handlePointerUp);
    renderer.domElement.addEventListener("pointercancel", handlePointerUp);
    renderer.domElement.addEventListener("wheel", handleWheel, { passive: false });
    const observer = new ResizeObserver(resize);
    observer.observe(mount);
    animate();

    return () => {
      projectionCleanup?.();
      window.cancelAnimationFrame(frameId);
      observer.disconnect();
      renderer.domElement.removeEventListener("pointerdown", handlePointerDown);
      renderer.domElement.removeEventListener("pointermove", handlePointerMove);
      renderer.domElement.removeEventListener("pointerup", handlePointerUp);
      renderer.domElement.removeEventListener("pointercancel", handlePointerUp);
      renderer.domElement.removeEventListener("wheel", handleWheel);
      disposeObject(root);
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [analysis, constraints, readout, selectedProxyId, solvePayload, sourceUrl, viewport]);

  return (
    <section className="viewport3d-panel" aria-label="3D camera viewport">
      <div className="viewport3d-toolbar">
        <div className="viewport3d-modes" aria-label="3D view modes">
          {modeOptions.map((mode) => (
            <button
              key={mode.id}
              className={viewport.display.active_mode === mode.id ? "active" : ""}
              type="button"
              onClick={() => updateDisplay({ active_mode: mode.id })}
            >
              {mode.label}
            </button>
          ))}
        </div>
        <div className="viewport3d-toggles" aria-label="3D display toggles">
          <ViewportToggle
            label="Image plate"
            active={viewport.display.show_image}
            onClick={() => updateDisplay({ show_image: !viewport.display.show_image })}
            icon={<ImageIcon size={15} />}
          />
          <ViewportToggle
            label="Grid"
            active={viewport.display.show_grid}
            onClick={() => updateDisplay({ show_grid: !viewport.display.show_grid })}
            icon={<Grid3X3 size={15} />}
          />
          <ViewportToggle
            label="Axes"
            active={viewport.display.show_axes}
            onClick={() => updateDisplay({ show_axes: !viewport.display.show_axes })}
            icon={<ScanLine size={15} />}
          />
          <ViewportToggle
            label="Camera frustum"
            active={viewport.display.show_frustum}
            onClick={() => updateDisplay({ show_frustum: !viewport.display.show_frustum })}
            icon={<Camera size={15} />}
          />
          <ViewportToggle
            label="Proxy objects"
            active={viewport.display.show_proxies}
            onClick={() => updateDisplay({ show_proxies: !viewport.display.show_proxies })}
            icon={<Box size={15} />}
          />
          <ViewportToggle
            label={analysis ? "Project image on ground" : "Project image on ground (run Analyze first)"}
            active={viewport.display.show_projection}
            disabled={!analysis}
            onClick={() => updateDisplay({ show_projection: !viewport.display.show_projection })}
            icon={<Film size={15} />}
          />
          <ViewportToggle
            label={viewport.display.lock_camera_to_view ? "Unlock view" : "Lock view"}
            active={viewport.display.lock_camera_to_view}
            onClick={() => updateDisplay({ lock_camera_to_view: !viewport.display.lock_camera_to_view })}
            icon={viewport.display.lock_camera_to_view ? <Lock size={15} /> : <Unlock size={15} />}
          />
        </div>
      </div>
      <div ref={mountRef} className="viewport3d-mount">
        <div className="viewport3d-view-label">{formatModeName(viewport.display.active_mode)}</div>
        <div className="viewport3d-fallback">
          <Layers3 size={20} />
          <span>{rendererStatus}</span>
        </div>
      </div>
      <div className="viewport3d-actionbar" aria-label="3D navigation actions">
        <button type="button" onClick={() => updateDisplay({ active_mode: "perspective", lock_camera_to_view: false })}>
          Frame All
        </button>
        <button type="button" onClick={() => updateDisplay({ active_mode: "perspective", lock_camera_to_view: false })} disabled={!selectedProxyId}>
          Frame Selected
        </button>
        <button type="button" onClick={() => updateDisplay({ active_mode: "image_match", lock_camera_to_view: true, image_opacity: 0.9 })}>
          Camera 1:1
        </button>
        <button
          type="button"
          onClick={() => updateDisplay({
            active_mode: "image_match",
            show_image: true,
            show_grid: true,
            show_axes: true,
            show_frustum: true,
            show_guides: true,
            show_proxies: true,
            show_horizon: true,
            lock_camera_to_view: true
          })}
        >
          Reset View
        </button>
      </div>
      <div className="viewport3d-readout">
        <span>
          <Eye size={13} />
          {formatVec3(readout.cameraPosition)}
        </span>
        <span>{formatOptional(readout.focalLength, "mm")}</span>
        <span>{readout.proxyCount} proxies</span>
      </div>
      {!!viewport.proxy_objects.length && (
        <div className="viewport3d-proxy-strip" aria-label="3D proxy selection">
          {viewport.proxy_objects.map((proxy) => (
            <button
              key={proxy.id}
              type="button"
              className={proxy.id === selectedProxyId ? "active" : ""}
              onClick={() => onSelectProxy(proxy.id === selectedProxyId ? null : proxy.id)}
            >
              {proxy.label}
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

function ViewportToggle({
  label,
  active,
  icon,
  onClick,
  disabled = false
}: {
  label: string;
  active: boolean;
  icon: ReactNode;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      className={active ? "viewport3d-toggle active" : "viewport3d-toggle"}
      type="button"
      title={label}
      aria-label={label}
      aria-pressed={active}
      disabled={disabled}
      onClick={onClick}
    >
      {icon}
    </button>
  );
}

function buildSceneReadout(
  viewport: Viewport3DState,
  analysis: CameraAnalysis | null,
  solvePayload: any
): SceneReadout {
  const intrinsics = solvePayload?.camera?.intrinsics ?? {};
  const estimation = solvePayload?.debug_metadata?.camera_estimation ?? {};
  const cameraPosition = tuple3(
    viewport.camera_overrides.camera_position ??
      analysis?.camera_position ??
      solvePayload?.camera?.extrinsics?.camera_position,
    [0, 1.6, 5.5]
  );
  return {
    focalLength: viewport.camera_overrides.focal_length_mm ?? intrinsics.focal_length_mm ?? estimation.focal_length_mm,
    fovHorizontal: analysis?.fov_deg.horizontal ?? estimation.fov_horizontal_deg,
    cameraPosition,
    proxyCount: viewport.proxy_objects.length
  };
}

function addImagePlane(
  root: any,
  sourceUrl: string,
  constraints: Constraints,
  opacity: number,
  onLoad: () => void
) {
  const aspect = constraints.image_width > 0 && constraints.image_height > 0
    ? constraints.image_width / constraints.image_height
    : 16 / 9;
  const width = 4.6;
  const height = width / aspect;
  const geometry = new THREE.PlaneGeometry(width, height);
  const material = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    opacity: clamp(opacity, 0.1, 1),
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false
  });
  const plane = new THREE.Mesh(geometry, material);
  plane.position.set(0, height * 0.46 + 0.18, -3.85);
  root.add(plane);

  const loader = new THREE.TextureLoader();
  loader.load(sourceUrl, (texture: any) => {
    texture.colorSpace = THREE.SRGBColorSpace;
    material.map = texture;
    material.needsUpdate = true;
    onLoad();
  });
}

function createGroundGrid(gridScale: number) {
  const size = clamp(gridScale, 0.4, 4) * 8;
  const grid = new THREE.GridHelper(size, 16, 0x6a5b4c, 0x3f3731);
  grid.position.y = 0;
  return grid;
}

function createAxisRig(size: number) {
  const group = new THREE.Group();
  group.add(createLine([[0, 0.02, 0], [size, 0.02, 0]], 0xd42b2b));
  group.add(createLine([[0, 0.02, 0], [0, size, 0]], 0x66a96c));
  group.add(createLine([[0, 0.02, 0], [0, 0.02, -size]], 0x5b86b2));
  return group;
}

function createCameraFrustum(
  position: [number, number, number],
  fovHorizontal: number | undefined,
  constraints: Constraints
) {
  const group = new THREE.Group();
  const cameraBody = new THREE.Mesh(
    new THREE.BoxGeometry(0.28, 0.2, 0.18),
    new THREE.MeshStandardMaterial({ color: 0xd42b2b, roughness: 0.65 })
  );
  cameraBody.position.set(...position);
  group.add(cameraBody);

  const aspect = constraints.image_width > 0 && constraints.image_height > 0
    ? constraints.image_width / constraints.image_height
    : 16 / 9;
  const depth = 1.25;
  const halfWidth = Math.tan(((fovHorizontal ?? 52) * Math.PI) / 360) * depth;
  const halfHeight = halfWidth / aspect;
  const corners = [
    new THREE.Vector3(position[0] - halfWidth, position[1] + halfHeight, position[2] - depth),
    new THREE.Vector3(position[0] + halfWidth, position[1] + halfHeight, position[2] - depth),
    new THREE.Vector3(position[0] + halfWidth, position[1] - halfHeight, position[2] - depth),
    new THREE.Vector3(position[0] - halfWidth, position[1] - halfHeight, position[2] - depth)
  ];
  const origin = new THREE.Vector3(...position);
  for (const corner of corners) {
    group.add(createLine([origin.toArray(), corner.toArray()], 0xd42b2b, 0.7));
  }
  for (let index = 0; index < corners.length; index += 1) {
    group.add(createLine([corners[index].toArray(), corners[(index + 1) % corners.length].toArray()], 0xd42b2b, 0.7));
  }
  return group;
}

function createGuideFamilies(constraints: Constraints) {
  const group = new THREE.Group();
  const imageWidth = constraints.image_width || 1;
  const imageHeight = constraints.image_height || 1;
  const families = [
    { lines: constraints.line_groups.left, color: 0xf5f2ed, z: -2.65 },
    { lines: constraints.line_groups.right, color: 0xd42b2b, z: -2.5 },
    { lines: constraints.line_groups.vertical, color: 0xf0c66b, z: -2.35 }
  ];
  for (const family of families) {
    for (const line of family.lines.slice(-8)) {
      const first = imagePointToScene(line[0], imageWidth, imageHeight, family.z);
      const second = imagePointToScene(line[1], imageWidth, imageHeight, family.z);
      group.add(createLine([first, second], family.color, 0.78));
    }
  }
  return group;
}

function createHorizonBand(solvePayload: any, constraints: Constraints) {
  const endpoints = solvePayload?.horizon_line?.endpoints_px;
  const group = new THREE.Group();
  if (!Array.isArray(endpoints) || endpoints.length < 2) return group;
  const imageWidth = constraints.image_width || 1;
  const imageHeight = constraints.image_height || 1;
  group.add(createLine(
    [
      imagePointToScene(endpoints[0], imageWidth, imageHeight, -2.82),
      imagePointToScene(endpoints[1], imageWidth, imageHeight, -2.82)
    ],
    0xd42b2b,
    0.92
  ));
  return group;
}

function createProxyObject(proxy: Viewport3DProxyObject, selected: boolean) {
  const group = new THREE.Group();
  group.name = proxy.id;
  group.position.set(...proxy.position);
  group.rotation.set(...proxy.rotation);
  group.scale.set(...proxy.scale);

  const accent = selected ? 0xf0c66b : 0xc9bfae;
  const material = new THREE.MeshStandardMaterial({
    color: proxy.type === "person_card" ? 0x8f332e : 0x75685a,
    roughness: 0.8,
    metalness: 0.05,
    transparent: proxy.type === "person_card",
    opacity: proxy.type === "person_card" ? 0.88 : 1
  });

  if (proxy.type === "floor_plane") {
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshStandardMaterial({
      color: 0x3f3731,
      roughness: 0.9,
      side: THREE.DoubleSide
    }));
    plane.rotation.x = -Math.PI / 2;
    group.add(plane);
  } else if (proxy.type === "wall_plane") {
    group.add(new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material));
  } else if (proxy.type === "person_card") {
    const card = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material);
    card.position.y = 0;
    group.add(card);
  } else {
    group.add(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material));
  }

  const box = new THREE.Box3().setFromObject(group);
  const helper = new THREE.Box3Helper(box, accent);
  helper.name = `${proxy.id}-outline`;
  group.add(helper);
  return group;
}

function createLine(points: number[][], color: number, opacity = 1) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map((point) => new THREE.Vector3(point[0], point[1], point[2])));
  const material = new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity });
  return new THREE.Line(geometry, material);
}

function applyCameraMode(
  camera: any,
  mode: Viewport3DMode,
  lockCameraToView: boolean,
  target: any,
  orbit: { yaw: number; pitch: number; distance: number }
) {
  if (mode === "top") {
    camera.position.set(0, 8.2, -2.2);
    camera.up.set(0, 0, -1);
  } else if (mode === "front") {
    camera.position.set(0, 1.25, 5.4);
    camera.up.set(0, 1, 0);
  } else if (mode === "side") {
    camera.position.set(6.2, 1.55, -2.2);
    camera.up.set(0, 1, 0);
  } else if (mode === "image_match" && lockCameraToView) {
    camera.position.set(0, 1.55, 4.8);
    camera.up.set(0, 1, 0);
  } else {
    const radius = orbit.distance;
    camera.position.set(
      target.x + Math.sin(orbit.yaw) * Math.cos(orbit.pitch) * radius,
      target.y + Math.sin(orbit.pitch) * radius,
      target.z + Math.cos(orbit.yaw) * Math.cos(orbit.pitch) * radius
    );
    camera.up.set(0, 1, 0);
  }
  camera.lookAt(target);
}

function imagePointToScene(point: number[], width: number, height: number, z: number): [number, number, number] {
  const aspect = width / height;
  const frameWidth = 4.6;
  const frameHeight = frameWidth / aspect;
  return [
    ((Number(point[0]) / width) - 0.5) * frameWidth,
    (0.5 - (Number(point[1]) / height)) * frameHeight + frameHeight * 0.46 + 0.18,
    z
  ];
}

function tuple3(value: unknown, fallback: [number, number, number]): [number, number, number] {
  if (!Array.isArray(value) || value.length < 3) return fallback;
  const next = [Number(value[0]), Number(value[1]), Number(value[2])] as [number, number, number];
  return next.every(Number.isFinite) ? next : fallback;
}

function disposeObject(object: any) {
  object.traverse((child: any) => {
    const mesh = child;
    if (mesh.geometry) mesh.geometry.dispose();
    const material = mesh.material;
    if (Array.isArray(material)) {
      for (const item of material) item.dispose();
    } else if (material) {
      material.dispose();
    }
  });
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function formatVec3(value: [number, number, number]) {
  return value.map((item) => item.toFixed(2)).join(", ");
}

function formatOptional(value: number | undefined, unit: string) {
  return typeof value === "number" ? `${value.toFixed(2)} ${unit}` : `-- ${unit}`;
}

function formatModeName(mode: Viewport3DMode) {
  if (mode === "image_match") return "Image Match";
  return mode.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
