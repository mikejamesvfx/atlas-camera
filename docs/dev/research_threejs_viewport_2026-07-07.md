# Research: Viability of a Larger three.js Implementation for the Atlas Viewport

**Date:** 2026-07-07 · **Status:** Researched, implemented, and verified live
**Question:** Is a fuller/properly-managed three.js integration viable for addressing ongoing `AtlasBlockoutViewport` issues?

## Executive Summary

**Yes — viable, and implemented same-day as a vendored-bundle upgrade, not a rewrite.** The viewport's architecture (custom orbit controller, projection `ShaderMaterial`, DOM-widget mounting) is sound; the weak layer was purely *how three.js got onto the page*. The old CDN-based loading chain was quietly broken in two ways (confirmed live, high confidence), and the fix — a committed, self-contained three.js r185 bundle — restored two dead features and removed the internet dependency.

## Findings (all verified against the live install unless noted)

### 1. The old loading chain was broken in production — CONFIRMED

Live browser probes against ComfyUI 0.27.0 / frontend 1.45.20:

| Import | Result |
|---|---|
| `../../lib/three.module.js` (first choice, "ComfyUI ships its own Three.js") | **Fails — path does not exist.** The frontend bundles three r180 only as a hashed internal Vite chunk (`vendor-three-*.js`), no import map anywhere. |
| unpkg CDN `three@0.163.0` (fallback) | Loads — so the viewport worked, but internet-dependent and ~2 years / 22 revisions stale. |
| unpkg `examples/jsm` `OBJLoader`/`FBXLoader` | **Fail** — `Failed to resolve module specifier "three"` (bare specifier, no import map). Both are try/caught → **🧍/🚗 scale proxies and 📥 FBX camera import were silently dead**. |

### 2. Repo maintained two divergent three.js integrations

`ui/` (React workbench) pins `three@^0.185.0` under Vite — the current release (r185, June 2026). The ComfyUI extension ran CDN r163. The projection shader exists in both (`ui/src/ProjectionMaterial.ts` + hand-synced GLSL port).

### 3. Ecosystem practice agrees

comfyui-3d-viewer-pro bundles three r170 fully locally; ComfyUI-3D-Pack vendors three + gsplat.js locally. Nobody consumes ComfyUI's internal chunk (no export contract). Import-map injection is fragile (must precede first module resolution; ComfyUI owns index.html).

### 4. Migration r163→r185 was effectively free for this codebase

No deprecated APIs in use (grep-verified: no `.encoding`, `useLegacyLights`, `WebGLMultipleRenderTargets`, etc.). Custom GLSL `ShaderMaterial` fully supported. One watch-item: FBXLoader r184+ auto-converts Z-up→Y-up; the FBX import's frame-0 `alignQuat` normalization should absorb it, recalibrate by eye when next used.

## What was implemented

1. `ui/bundle/atlas-three-entry.js` — bundle entry: `export * from "three"` + the two loaders.
2. `ui/package.json` — `esbuild ^0.28.0` devDep + `npm run build:comfy-three` script.
3. `atlas_camera/comfy/web/lib/atlas-three.bundle.js` — committed 770KB minified ESM (three r185 + OBJLoader + FBXLoader).
4. `loadThree()` in `atlas_blockout.js` — single local import, **no CDN fallback** (broken bundle should fail loudly, not degrade into version skew).
5. Docs: CLAUDE.md frontend section, `docs/COMFY_WORKFLOW.md` (loading + troubleshooting).

## Live verification results (r185 bundle)

- Bundle serves at `/extensions/AtlasCamera/lib/atlas-three.bundle.js` (200), imports in-browser: `REVISION 185`, both loaders present.
- Core projection workflow (`atlas_camera_core_projection_workflow.json`) executed end-to-end; viewport rendered grey relief-mesh preview at 768×432.
- 📽 Project: hangar photo projected with correct contrast/saturation (`atlasLinearToSRGB` encode intact).
- 🧍 Woman OBJ proxy: loaded a 50,454-vertex mesh at 0.01 scale onto the ground plane — **restored from dead**.
- Render Proxy Passes round-trip: non-black 768×432 shaded output through `client_data` → Python.
- **Zero unpkg/CDN network requests** — fully offline-capable.
- Console clean of `[AtlasBlockout]` errors/warnings.

## What this does NOT fix (out of scope, unchanged)

Overlay letterbox misalignment (CSS follow-up), orbit-coverage black regions (inherent to single-photo projection), Python↔JS Catmull-Rom duplication.

## Deferred follow-ups

- Share projection shader source between `ui/` and the comfy extension (now on the same three version — prerequisite met).
- `TransformControls` for interactive primitive placement (now bundleable).
- TSL/NodeMaterial exploration so custom projection materials inherit three's own colorspace/tonemapping chunks.
- Live FBX camera import test with a real DCC-authored FBX (loader restored; r184 Y-up auto-conversion to be eyeballed).

## Sources

- Live install inspection: ComfyUI_V91 venv, `comfyui_frontend_package` 1.45.20; live browser import probes 2026-07-07
- [three.js releases](https://github.com/mrdoob/three.js/releases) (r185, June 2026) · [Migration Guide](https://github.com/mrdoob/three.js/wiki/Migration-Guide)
- [comfyui-3d-viewer-pro](https://github.com/brandondunwell/comfyui-3d-viewer-pro) · [ComfyUI-3D-Pack](https://github.com/MrForExample/ComfyUI-3D-Pack) · [comfy-3d-viewers](https://github.com/PozzettiAndrea/comfy-3d-viewers)
- [Import maps browser support](https://caniuse.com/import-maps) · [web.dev: import maps](https://web.dev/blog/import-maps-in-all-modern-browsers)
