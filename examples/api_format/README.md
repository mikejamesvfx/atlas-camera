# API-format workflows (validated 2026-07-04)

These are ComfyUI's *API* format (node-id -> {class_type, inputs}), not the
UI/litegraph format the other files in `examples/` use. They have no layout
data and **cannot be opened in ComfyUI's browser canvas** — POST them
directly to `/prompt`, or run them with the `comfyui` skill's `comfy_client.py`.
For interactive, click-around browser testing, use the `.json` files one
level up in `examples/` instead.

All three were run end-to-end against a live ComfyUI instance on 2026-07-04
and completed successfully:

- `atlas_camera_core_projection.api.json` — 6-node minimal pipeline (solve →
  derive → viewport → export relief mesh). Source image swapped to
  `example.png` (ships with every ComfyUI install) for portability; the
  exact same graph structure was validated against other source images
  repeatedly this session.
- `atlas_camera_learned_full_pipeline.api.json` — the full 26-node learned
  pipeline (VLM scale cues, derive, decompose, all analysis nodes, viewport,
  and all 5 DCC export formats: JSON/Blender/Nuke/USD/relief-mesh/review
  package). Requires the `usd-core` pip package in ComfyUI's venv for the
  USD export node (`pip install usd-core`) — confirmed installed and working
  as of this validation run.
- `atlas_camera_multiangle_patch_selfcontained.api.json` — full self-contained
  multi-angle patch demo: generates its own source photo and novel-view patch
  via Qwen-Image-2512 + Qwen-Image-Edit-2511 + the Multiple-Angles LoRA, no
  external image needed. Requires those specific model files in ComfyUI's
  model paths (see `docs/comfyui_agent_kit/atlas-camera.md` for the pack's
  model dependencies). Validated with an exact orbit-angle check (45.0°) on
  five different generated scenes.
