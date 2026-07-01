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
    gl_FragColor = vec4(col.rgb, col.a * uOpacity);
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
