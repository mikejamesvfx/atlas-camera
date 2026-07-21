import * as THREE from "three";
import type { CameraAnalysis } from "./types";

// Mirrors projectPointToImage from viewport3dMath.ts in GLSL.
// Atlas view matrix convention: camera looks toward -Z (cam.z < 0 for in-front points).
// Image convention: origin top-left, Y increases downward.
// Texture is loaded with flipY=false so UV (0,0) = top-left, matching pixel coords.
const VERTEX_SHADER = `
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

    float depth = -cam.z;
    if (depth > 1e-5) {
      vImagePx = vec2(
        uCx + uFx * cam.x / depth,
        uCy - uFy * cam.y / depth
      );
    } else {
      vImagePx = vec2(-1.0, -1.0);
    }

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const FRAGMENT_SHADER = `
  uniform sampler2D uTexture;
  uniform vec2 uImageSize;
  uniform float uOpacity;

  varying vec2 vImagePx;
  varying float vCamZ;

  void main() {
    if (vCamZ >= 0.0) discard;
    vec2 uv = vImagePx / uImageSize;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) discard;
    vec4 col = texture2D(uTexture, uv);
    // Texture colour management decodes/encodes RGB only. Alpha remains
    // unassociated linear data and is not passed through a colour transform.
    gl_FragColor = vec4(col.rgb, col.a * uOpacity);
  }
`;

const DEPTH_VERTEX_SHADER = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const DEPTH_FRAGMENT_SHADER = `
  precision highp float;

  uniform mat4  uCamToWorld;
  uniform vec3  uCameraPos;
  uniform float uFx;
  uniform float uFy;
  uniform float uCx;
  uniform float uCy;
  uniform vec2  uImageSize;
  uniform float uDepthNear;
  uniform float uDepthFar;
  uniform float uOpacity;

  varying vec2 vUv;

  vec3 depthHeatmap(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.90, 0.12, 0.04);
    vec3 c1 = vec3(0.96, 0.72, 0.08);
    vec3 c2 = vec3(0.20, 0.84, 0.60);
    vec3 c3 = vec3(0.08, 0.22, 0.86);
    if (t < 0.333) return mix(c0, c1, t * 3.0);
    if (t < 0.667) return mix(c1, c2, (t - 0.333) * 3.0);
    return mix(c2, c3, (t - 0.667) * 3.0);
  }

  void main() {
    if (uCameraPos.y <= 0.0) discard;

    // Three.js PlaneGeometry UV: (0,0)=bottom-left; image: (0,0)=top-left
    vec2 px = vec2(vUv.x * uImageSize.x, (1.0 - vUv.y) * uImageSize.y);

    // Unproject pixel to camera-space ray (camera looks along -Z)
    vec3 rayCam = normalize(vec3(
      (px.x - uCx) / uFx,
      -(px.y - uCy) / uFy,
      -1.0
    ));

    // Rotate to world space (direction only, w=0)
    vec3 rayWorld = normalize((uCamToWorld * vec4(rayCam, 0.0)).xyz);

    // Intersect with ground plane Y=0: cameraPos.y + t * rayWorld.y = 0
    if (abs(rayWorld.y) < 1.0e-5) discard;
    float t = -uCameraPos.y / rayWorld.y;
    if (t < 0.001) discard;

    float normalized = clamp((t - uDepthNear) / (uDepthFar - uDepthNear), 0.0, 1.0);
    gl_FragColor = vec4(depthHeatmap(normalized), uOpacity);
  }
`;

function buildAtlasViewUniform(analysis: CameraAnalysis): any {
  const vm = analysis.view_matrix;
  return new THREE.Matrix4().set(
    vm[0][0], vm[0][1], vm[0][2], vm[0][3],
    vm[1][0], vm[1][1], vm[1][2], vm[1][3],
    vm[2][0], vm[2][1], vm[2][2], vm[2][3],
    vm[3][0], vm[3][1], vm[3][2], vm[3][3]
  );
}

function createProjectionMaterial(
  analysis: CameraAnalysis,
  imageWidth: number,
  imageHeight: number,
  texture: any
): any {
  return new THREE.ShaderMaterial({
    uniforms: {
      uAtlasViewMatrix: { value: buildAtlasViewUniform(analysis) },
      uFx: { value: analysis.focal_px.fx },
      uFy: { value: analysis.focal_px.fy },
      uCx: { value: analysis.principal_point_px.cx },
      uCy: { value: analysis.principal_point_px.cy },
      uTexture: { value: texture },
      uImageSize: { value: new THREE.Vector2(imageWidth, imageHeight) },
      uOpacity: { value: 1.0 }
    },
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    transparent: true,
    // NormalBlending expects straight RGB here. If an associated plate is ever
    // accepted, unpremultiply before its RGB transform and re-premultiply only
    // at the blend boundary; never colour-transform alpha.
    premultipliedAlpha: false,
    side: THREE.DoubleSide,
    depthWrite: false,
    depthTest: true
  });
}

export function addProjectionGround(
  root: any,
  analysis: CameraAnalysis,
  imageWidth: number,
  imageHeight: number,
  sourceUrl: string,
  onLoad: () => void,
  onError?: (message: string) => void
): () => void {
  const geometry = new THREE.PlaneGeometry(40, 40, 64, 64);
  const placeholder = new THREE.MeshBasicMaterial({
    transparent: true,
    opacity: 0,
    depthWrite: false,
    depthTest: false
  });
  const mesh = new THREE.Mesh(geometry, placeholder);
  mesh.name = "projection_ground";
  mesh.rotation.x = -Math.PI / 2;
  mesh.renderOrder = 1;
  root.add(mesh);

  let aborted = false;
  let activeMaterial: any = null;
  let activeTexture: any = null;

  const loader = new THREE.TextureLoader();
  loader.load(
    sourceUrl,
    (texture: any) => {
      if (aborted) {
        texture.dispose();
        return;
      }
      texture.flipY = false;
      texture.colorSpace = THREE.SRGBColorSpace;
      const mat = createProjectionMaterial(analysis, imageWidth, imageHeight, texture);
      activeMaterial = mat;
      activeTexture = texture;
      mesh.material = mat;
      onLoad();
    },
    undefined,
    () => {
      if (!aborted) onError?.(`Projection texture failed to load: ${sourceUrl}`);
    }
  );

  return () => {
    aborted = true;
    geometry.dispose();
    placeholder.dispose();
    activeMaterial?.dispose();
    activeTexture?.dispose();
  };
}

export function addDepthOverlay(
  root: any,
  analysis: CameraAnalysis,
  imageWidth: number,
  imageHeight: number,
  onReady: () => void
): () => void {
  const aspect = imageWidth > 0 && imageHeight > 0 ? imageWidth / imageHeight : 16 / 9;
  const planeWidth = 4.6;
  const planeHeight = planeWidth / aspect;

  const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight);

  const viewMatrix = buildAtlasViewUniform(analysis);
  const camToWorld = viewMatrix.clone().invert();

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uCamToWorld: { value: camToWorld },
      uCameraPos:  { value: new THREE.Vector3(
        analysis.camera_position[0],
        analysis.camera_position[1],
        analysis.camera_position[2]
      )},
      uFx:        { value: analysis.focal_px.fx },
      uFy:        { value: analysis.focal_px.fy },
      uCx:        { value: analysis.principal_point_px.cx },
      uCy:        { value: analysis.principal_point_px.cy },
      uImageSize: { value: new THREE.Vector2(imageWidth, imageHeight) },
      uDepthNear: { value: 1.0 },
      uDepthFar:  { value: 50.0 },
      uOpacity:   { value: 0.65 }
    },
    vertexShader: DEPTH_VERTEX_SHADER,
    fragmentShader: DEPTH_FRAGMENT_SHADER,
    transparent: true,
    // Procedural heatmap RGB + independent linear opacity (straight alpha).
    premultipliedAlpha: false,
    side: THREE.DoubleSide,
    depthWrite: false,
    depthTest: true
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = "depth_overlay";
  mesh.position.set(0, planeHeight * 0.46 + 0.18, -3.85);
  mesh.renderOrder = 2;
  root.add(mesh);

  onReady();

  return () => {
    root.remove(mesh);
    geometry.dispose();
    material.dispose();
  };
}
