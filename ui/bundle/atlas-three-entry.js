/**
 * Entry point for the vendored Three.js bundle used by the ComfyUI frontend
 * extension (atlas_camera/comfy/web/atlas_blockout.js).
 *
 * Built with `npm run build:comfy-three` into
 * atlas_camera/comfy/web/lib/atlas-three.bundle.js — a single self-contained
 * ESM file (three core + the two loaders the viewport needs), committed to the
 * repo so ComfyUI users never need npm or a network connection.
 *
 * Re-exports the full three namespace at the top level so the extension can
 * treat the imported module object as THREE directly, plus the addon loaders
 * (which are not part of the core namespace, so no name collisions).
 */
export * from "three";
export { OBJLoader } from "three/addons/loaders/OBJLoader.js";
export { FBXLoader } from "three/addons/loaders/FBXLoader.js";
