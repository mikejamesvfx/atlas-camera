# Third-Party Notices & License Boundaries

Atlas Camera itself is **MIT** (see LICENSE). Nothing below is vendored into
this repository — every third-party model or package is installed or cloned by
the user, and the nodes that depend on one fail soft with an informative error
when it's absent. This page is the honest map of what each optional capability
pulls in and what its terms allow.

## Summary table

| Dependency | License | How it's obtained | Used by | Commercial use |
|---|---|---|---|---|
| GeoCalib | Apache 2.0 | `pip install` from GitHub (`[neural]` docs) | Learned camera solve | ✅ |
| Depth Anything V2 | Apache 2.0 (small/base weights; large is CC BY-NC 4.0 — check the variant you install) | HuggingFace via transformers | Depth estimation (legacy default) | ✅ small/base · ⚠ large |
| Depth Anything 3 (`DA3METRIC-LARGE`) | Apache 2.0 | user-installed `depth_anything_3` package (`--no-deps`, see INSTALL.md) | **Default** depth backend | ✅ |
| LaRI | **No upstream license** (all rights reserved by default) | user clones github.com/ruili3/lari → `lari_path`/`ATLAS_LARI_PATH` | `AtlasPredictHiddenGeometry` (lari-scene) | ❌ research only |
| World Tracing (WT-DiT r69l) | **CC BY-NC-ND 4.0**, checkpoint HF-gated | user clones repo + requests checkpoint access → `wt_path`/`ATLAS_WT_PATH` | `AtlasPredictHiddenGeometry` (world-tracing-scene) | ❌ non-commercial |
| SAM 3 (via ComfyUI-RMBG) | per Meta's SAM license / pack's terms | ComfyUI Manager | Sky + foreground mattes in the hero workflows | check pack |
| comfyui-inpaint-nodes (LaMa/MAT) | **GPL-3.0** | ComfyUI Manager | X-ray clean plates, inpaint-layer track | graph-level only — see boundary below |
| big-lama.pt weights | Apache 2.0 (LaMa) | pack's model download | same | ✅ |
| ComfyUI-OCIO | per pack | ComfyUI Manager | ACEScg full-float examples | check pack |
| Qwen-Image-Edit-2511 + Multiple-Angles LoRA | per model card | user-installed models | Master-DMP patch generation | check card |
| VideoCombinePlus (+ ffmpeg) | per pack / LGPL-GPL ffmpeg build | ComfyUI Manager | Hero dolly bakes | check pack |
| three.js r185 | MIT | **vendored** (`atlas_camera/comfy/web/lib/atlas-three.bundle.js`, built from `ui/`'s pinned dependency) | Blockout viewport | ✅ |

## The two research-only backends, stated plainly

The **hidden-geometry track is research-only** in any deployment that includes
its backends:

- **LaRI** ships with no license file, which legally defaults to
  all-rights-reserved — stricter than any non-commercial license. Atlas never
  vendors or redistributes its code or weights; the node requires the user's
  own clone and warns in its report output. If the track matures, the right
  move is asking the authors for a license (issue planned).
- **World Tracing**'s scene checkpoint is gated on HuggingFace and licensed
  CC BY-NC-ND 4.0: non-commercial, no derivatives of the weights. The same
  user-clone pattern applies.

Everything else in Atlas — the solve, geometry derivation, layer stack,
viewport, and the whole professional/OCIO output path — carries **no
non-commercial dependency**. Removing the 🔬 node from a graph removes the
restriction.

## The GPL boundary

Masking/inpainting is never implemented inside `atlas_camera`. GPL-licensed
ComfyUI packs (comfyui-inpaint-nodes) participate only as **separate nodes
wired into a graph** — graph-level composition, not linking — so Atlas's MIT
license is unaffected. This boundary is deliberate and documented in
INSTALL.md's "Optional Inpaint Integration"; keep it: any future inpaint
capability belongs in the graph, not in this package.

## Weights are not code

Model weights downloaded at runtime (HuggingFace, pack model folders) are
governed by their own model cards/licenses regardless of the wrapper code's
license. When in doubt about a deployment, check the **weights'** terms — the
table above lists them where known, but model cards change; the card is
authoritative.
