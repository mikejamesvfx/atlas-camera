# Agent handoff — pause a graph, brief an agent, resume on its reply

`AtlasAgentHandoff` 🤝🔬 (experimental tier, `ATLAS_EXPERIMENTAL=1`) is the
contract between a ComfyUI graph and an **external agent** — Claude Code through
the atlas MCP, Hermes, OpenClaw, or a person with `curl`. The node holds no
model and no MCP client: it publishes a *brief*, **blocks** the queue until a
*resume* arrives (bounded by `timeout_s`), then optionally brings the agent's
Blender work back into the solve. ComfyUI stays the engine; the agent stays
pluggable. Live-verified 2026-08-16: brief → Claude operating Blender 5.2 via
blender-mcp → resume → 2 meshes appended, 78 s wall-clock.

## Where things are

```
<ComfyUI output>/atlas_agent/<node_id>/
  brief.json          written by the node when it pauses
  resume.json         written by the agent; consumed by the node
  history/            every brief/resume (and stale resumes) with timestamps
```

## The brief

| key | meaning |
|---|---|
| `task` | the artist's instruction, verbatim from the node's `task` widget |
| `token` | echo it in the resume — an old reply can never release a new pause |
| `deadline` | epoch seconds; the graph continues (or fails, `on_timeout`) after it |
| `exchange_dir`, `seed_json`, `scene_blend`, `out_meshes_npz` | the Blender massing exchange folder (from `AtlasBlenderMassing.exchange_dir`) |
| `collections` | model under `atlas_out`; `atlas_reference` is never exported |
| `measured` | MoGe-measured scene: `camera_height_m`, `ground_y_m`, `extent_m`, `median_depth_m`, planes … |
| `camera` | recovered intrinsics/pose (Atlas Y-up; the Blender seed is Z-up) |
| `snapshots` | the automatic viewport PNGs (📽 on/off from the recovered camera) — LOOK at these |
| `tools_allowed` | what the artist permits: `blender_mcp`, `blender_headless`, `atlas_mcp`, `comfy_mcp`, `filesystem` |
| `return_contract`, `resume.how` | exactly how to hand back |

## The resume

```json
{"token": "<from brief>", "status": "done" | "skip" | "fail",
 "reply": "one or two lines of what you did", "blend_file": "<saved .blend or empty>", "notes": ""}
```

Three ways to send it (all write the same file):

- **MCP** (atlas server): `atlas_agent_brief(node_id)` → work → `atlas_agent_resume(node_id, token, reply, status, blend_file)`
- **HTTP**: `GET /atlas/agent/brief/{node_id}`, `POST /atlas/agent/resume/{node_id}` (JSON body above)
- **file**: write `resume.json` next to the brief

With `auto_import` (default on) and `status=done`, the node runs
`atlas_camera/blender/recipes/export_meshes.py` headless on `blend_file` (or the brief's
`scene_blend`), reads the `atlas_out` meshes back, refuses a seed built for a
different solve (`expect_fingerprint`), and appends them as PROXY_ROLE
`blender_import` primitives with projective UVs regenerated for the recovered
camera — the same path as `AtlasBlenderImportMeshes`. `skip`/`fail`/timeout pass
the solve through and say so; nothing raises unless `on_timeout=fail`.

## A ready-made operator loop (Claude Code / Hermes)

1. `atlas_agent_brief(<node id>)` — read `task`, `measured`, `snapshots`, `scene_blend`.
2. Look at the snapshot PNGs. Decide what geometry the photo needs.
3. Blender: `bpy.ops.wm.open_mainfile(filepath=scene_blend)`; measure the
   `atlas_reference` objects (`atlas_cloud`, `ref_projection_plane_*`,
   `ref_projection_ground`) for placement; model under `atlas_out` in metres
   (ground is at Blender Z = `ground_y_m`); `bpy.ops.wm.save_as_mainfile(...)`.
4. `atlas_agent_resume(<node id>, token, reply="…", status="done", blend_file=scene_blend)`.

Rules the agent should respect: never edit `atlas_reference`; keep the camera
object; metres; short reply. **Fit, don't eyeball:** streets rarely run along
the camera axis — build volumes from the `ref_projection_plane_*` orientations
(fit each facade as a plane / a line in top view) rather than axis-aligned
boxes; `footprint_source=measured_planes` on the massing node hands you these
slabs pre-oriented. Distrust the cloud beyond ~3× `median_depth_m` (monocular
far-field compression makes facades converge); model near geometry from the
cloud, far geometry from the photo. Doctrine unchanged: the atlas MCP server never
executes — it reads/writes these JSON files, ComfyUI does the work.

## What paints the agent's meshes

The viewport's `clean_plate` input paints the `projection_backdrop` and any
primitive tagged `paint_with: clean_plate`. Meshes coming back through
`AtlasAgentHandoff` default to **`clean_plate`** (they are the OCCLUDED
surfaces — water and hillside behind a foreground object), while
`AtlasBlenderMassing` / `AtlasBlenderImportMeshes` default to `source_photo`
(facades the plate shows). Both have a `paint_with` widget; a Blender custom
property `atlas_paint = "clean_plate" | "source_photo"` on a mesh overrides
per object. So: wire your clean plate into the viewport, keep the agent's
water/hill on `clean_plate`, and orbiting reveals the clean background on them.

For more than one plate, or geometry from several sources, use **`AtlasPlateLayer`** 🎞
after the handoff: `geometry_filter="agent_"` (or `blender_import`) + your clean plate →
a projection layer on exactly those meshes; chain another for the next plate.

## The override switch

`mode` on the node (appended 2026-08-16): `wait` (default — pause for an agent),
`import_only` (no pause; act as if the agent said `done` and import whatever
`scene.blend` / `out_meshes.npz` already hold — the re-run case after the agent
has done its work), `skip` (no pause, no import). The `ATLAS_AGENT_MODE`
environment variable overrides the widget for the whole server, so a headless
benchmark or an unattended queue never blocks on a brief nobody will answer.

## Later

`operator=local_vlm` — the same node running the loop in-process against its
own brief (LM Studio / Ollama vision model + MCP clients). Same contract, no
change to briefs or resumes.
