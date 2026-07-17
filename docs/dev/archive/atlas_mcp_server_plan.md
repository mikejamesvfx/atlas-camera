> ARCHIVED 2026-07-18 — superseded by the shipped atlas_camera/mcp server + docs/MCP_SERVER.md

# Atlas v1 MCP Server — design plan

**Status:** v1 IMPLEMENTED (2026-07-16, same day) — `atlas_camera/mcp/`
(`python -m atlas_camera.mcp`, `[mcp]` extra). Eight tools + three resources,
live-verified against a running ComfyUI (health, validate, catalog, viewport
census, and the full solve round-trip: upload → VP solve → JSON read-back).
`atlas_bake_camera_move` is the one deferred tool (browser-bound, per below).
The UI→API flattening/validation moved to `atlas_camera/mcp/comfy_http.py`
as the canonical implementation; the `tools/` CLIs are now thin wrappers.
Every tool is grounded in an operation that was *actually needed and
performed* while driving Atlas headlessly through the 2026-07-15/16
verification sessions — this is the operational surface an AI assistant
provably needs, not a speculative one.

## Why an MCP server

Driving Atlas today from an AI assistant means: hand-flattening UI workflows
to API format, knowing which gates block execution and which widget opens
them, polling `/history`, decoding viewport payloads to check layer health,
and knowing ~60 nodes' calibration doctrine (which depth model per scene
type, which grid/edge values per band, seam priorities). All of that
knowledge exists in this repo (CLAUDE.md, docs, the debug JSON) but none of
it is machine-callable. An MCP server makes Atlas operable by any
MCP-capable assistant without shipping them the whole repo as prompt.

## Architecture

A thin Python MCP server (stdio; `mcp` reference SDK) that talks to a running
ComfyUI over HTTP — the same boundary `tools/run_ui_workflow.py` and
`tools/validate_ui_workflow.py` already use. It does NOT import torch or run
models itself; ComfyUI stays the execution engine. Configuration: `COMFY_HOST`
(default `127.0.0.1:8188`), optional `ATLAS_REPO` for docs/schema resources.

## Proposed tools (v1)

| Tool | Backing implementation (exists today) | Notes |
|---|---|---|
| `atlas_health` | `GET /system_stats` + `/object_info` probe of Atlas classes | Reports server, VRAM, which Atlas nodes are registered (catches a missing `ATLAS_EXPERIMENTAL`), missing third-party packs, and the opencv/EXR codec check that bit this session (cv2 5.x cannot read EXR). |
| `atlas_run_workflow` | `tools/run_ui_workflow.py` (UI→API flatten off live `/object_info`, KJ rail resolution, muted/bypassed handling, gate override via `--set`, queue + poll with verbatim tracebacks) | The single most-used operation of these sessions. Args: workflow path or JSON, overrides dict, `open_gates: bool` (auto-finds `AtlasSolveGate`/`proceed`). |
| `atlas_validate_workflow` | `tools/validate_ui_workflow.py` | Positional-widget drift, link integrity, widget ranges, rail resolution — returns the error list. |
| `atlas_solve_image` | a 3-node generated graph (LoadImage → AtlasLearnedSolveFromImage/AtlasSolveFromImage → AtlasExportSolveJSON), or direct `atlas.recover()` in-process for CPU-light VP solves | Returns the solve JSON summary (focal, FOV, height, pitch/roll, confidence, scale_source). |
| `atlas_read_debug_report` | the 🔍 `AtlasDebugReport` stable-path JSON (`atlas_debug/*.json`, schema-versioned) | Purpose-built for this: one file-read instead of autopsying live payloads. The server should also be able to REQUEST one (inject the node + re-run). |
| `atlas_inspect_viewport` | `GET /atlas/camera_data/{node_id}` | Layer census used constantly this session: projection_sources names/priorities/band ranges/vert counts/matte presence, camera meta. |
| `atlas_export_scene` | a generated export graph fanning one solve into the chosen exporters (`AtlasExportNuke[Layers]`, `AtlasExportMayaLayers`/`ReviewScene`, `AtlasExportBlender`, `AtlasExportUSD`, `AtlasExportReliefMesh` with retopo knobs) | Args: solve JSON path or upstream workflow, formats list, output_dir. Returns written file paths. |
| `atlas_bake_camera_move` | headless browser (Playwright) driving the viewport's one-click moves + ⏺ Bake, then extracting the baked JPEG frames from `client_data` (they are already JPEG bytes) | The one tool that needs a browser; validated end-to-end 2026-07-16. Args: workflow, move (`orbit_left`…`dolly_in`), lens scale. Returns frames dir + fps. |
| `atlas_node_catalog` | `/object_info` filtered to Atlas + the CLAUDE.md catalog rows | Machine-readable: inputs/outputs/defaults + the doctrine line per node. |

## Proposed resources (v1)

- `atlas://catalog` — the node catalog table (generated, versioned with the repo).
- `atlas://calibration` — the per-scene-type doctrine distilled from CLAUDE.md:
  exterior→V2-Outdoor / interior→MoGe-or-V2-Indoor; band relief 384/1.5;
  seam priorities farthest-highest; edge_extend on behind-layers only;
  AI-vista scale is assumed_default and needs the 📐/height dial;
  interior `sky_heuristic=False`, `max_edge_factor` 40–80; SAM3 prompts joined
  with "and", never commas; comma prompts silently return empty masks.
- `atlas://schemas/solve` and `atlas://schemas/debug_report` — the JSON shapes
  (`AtlasSolve.to_json`, the 🔍 schema-versioned debug JSON).
- `atlas://gates` — the gate state table (`docs/dev/gate_state_table.md`):
  which persisted widgets block execution and how approval fingerprints re-arm.

## Operational knowledge the server must encode (learned this session)

1. **Gates**: shipped workflows close `AtlasSolveGate` (`proceed=False`);
   headless runs open it per-run. `AtlasAssessImage.auto_continue=True` is
   advisory-flow; a stale `approved_for` fingerprint re-arms on new images.
2. **UI→API flattening**: KJ Set/Get rails are frontend-only; muted (mode 2)
   terminal nodes drop; bypassed (mode 4) nodes forward their first same-type
   input; `seed` widgets carry a phantom `control_after_generate` slot and
   `image_upload` widgets a phantom upload slot.
3. **Environment traps**: `ATLAS_EXPERIMENTAL=1` must be set before python
   for the X-ray/RenderFix nodes; `OPENCV_IO_ENABLE_OPENEXR=1` + opencv 4.x
   for the OCIO/EXR path; `[moge]`/`[neural-da3]`/`[usd]` extras are runtime
   requirements the health tool should probe.
4. **The viewport is browser-side**: anything involving Bake/Render Passes/📐
   needs a real rendering browser context (hidden tabs freeze rAF — measured);
   everything else runs fully headless.
5. **Editable-install reality**: node code loads at server start — a code
   change needs a restart, and on this machine the venv resolves `atlas_camera`
   through a finder-style editable install (PYTHONPATH cannot shadow it).

## Not in v1

- Writing/committing workflow JSONs (authoring stays with the generators).
- Driving DCC apps (Nuke/Maya) — the exports are the handoff boundary.
- Cloud execution.

## Suggested repo layout when implemented

```
atlas_camera/mcp/__init__.py      # server entry: python -m atlas_camera.mcp
atlas_camera/mcp/comfy_http.py    # the run/validate/flatten logic, lifted from tools/
atlas_camera/mcp/resources.py     # catalog/calibration/schema resource builders
pyproject: [project.optional-dependencies] mcp = ["mcp"]
```
