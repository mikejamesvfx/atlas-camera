# Test drive: driving Atlas from OpenAI Codex via the Atlas MCP server

A hands-on test case: use **Codex** (a non-Claude agent) to operate Atlas in
ComfyUI purely through the `atlas` MCP server — proving the server is
genuinely agent-agnostic. Written 2026-07-17 against v0.6.0 beta on the
authoring machine; adapt the paths for yours.

## Already set up on this machine

- `~/.codex/config.toml` carries the server (added 2026-07-17):

  ```toml
  [mcp_servers.atlas]
  command = "python"
  args = ["-m", "atlas_camera.mcp"]
  startup_timeout_sec = 60

  [mcp_servers.atlas.env]
  COMFY_HOST = "127.0.0.1:8188"
  COMFY_DIR = 'C:\Users\miike\ComfyUI_V91\ComfyUI'
  ```

  (`COMFY_DIR` lets the read-back tools resolve server-side files —
  solve JSONs, debug reports.)

- The global `python` resolves `atlas_camera` (editable install → the dev
  checkout) and `mcp` — verified. If that ever breaks, set `command` to the
  ComfyUI venv python: `C:\Users\miike\ComfyUI_V91\ComfyUI\venv\Scripts\python.exe`.

## Preflight (30 seconds)

1. ComfyUI must be up: launch `Windows_Run_GPU_EXR_fixed.bat` (it now sets
   `ATLAS_EXPERIMENTAL=1` + `ATLAS_LARI_PATH` too). Confirm:
   `curl http://127.0.0.1:8188/system_stats` → 200.
2. Start Codex **in the repo** so relative workflow paths resolve:

   ```
   cd C:\Users\miike\Desktop\AtlasCamera_Claude
   codex
   ```

3. In Codex, check the server attached (`/mcp` in the TUI, or just ask it
   to list its atlas tools). Approve the server/tool prompts when Codex
   asks — every test below is read-only against your filesystem; the only
   writes land in ComfyUI's `atlas_exports/`.

## The test script — six prompts, escalating

Paste these one at a time. Each has a pass condition; if a step fails, the
troubleshooting table below covers the likely cause.

**T1 — connectivity + health**

> Using the atlas MCP server, call `atlas_health` and summarize the result.

✅ Pass: `ok: true`, ~60 atlas nodes, `experimental_registered: true`,
VRAM figure, missing third-party list (may be empty).

**T2 — the knowledge layer**

> Read the `atlas://calibration` resource and tell me: which depth model
> should I use for an interior, and what is the sky-rise scale doctrine?

✅ Pass: it answers MoGe/V2-Indoor for interiors, and describes counting a
building's storeys × 3.5 m into `AtlasReferenceScaleSolve`. This proves a
foreign agent inherits Atlas doctrine it was never trained on.

**T3 — a real camera solve**

> Solve the camera for
> `C:\Users\miike\Desktop\AtlasCamera_Claude\examples\images\ghosttown.jpg`
> with `atlas_solve_image` (learned method). Report focal, FOV, camera
> height, confidence, and — most importantly — interpret `scale_source`.

✅ Pass: ~21 mm / ~81°, height ≈ 2.5 m with `scale_source:
"depth_ground_plane"` (measured, tier 2), confidence ≈ 0.88. Bonus points
if the agent explains why a *measured* source beats an assumed one.

**T4 — run a shipped showcase workflow**

> Validate `examples/showcase/atlas_city_blocks_newyork_workflow.json`
> with `atlas_validate_workflow`, then run it with `atlas_run_workflow`.
> Report whether gates were opened, how many nodes executed, and any node
> errors verbatim.

✅ Pass: validation clean; run completes (warm cache: well under a minute)
with the counted-storey reference scale in play. This is the step a
generic ComfyUI driver *cannot* do — the JSON has KJ rails and would not
queue raw.

**T5 — the debug autopsy**

> Read the debug report at `atlas_debug/showcase_ghosttown.json` with
> `atlas_read_debug_report`. How many projection layers are there, what
> geometry type does each band use, and are there any red flags?

✅ Pass: 5 sources (sky + 4 bands), per-band geometry (`card`/`relief`/
`ground`), vertex counts, empty flags list. (Path resolves because
`COMFY_DIR` is set.)

**T6 — DCC export round-trip**

> Export the saved solve `atlas_exports/showcase_alley/alley_solve.json`
> to Nuke and USD with `atlas_export_scene`, output_dir
> `atlas_exports/codex_test`. Then tell me exactly which files were written.

✅ Pass: completed run; files under
`C:\Users\miike\ComfyUI_V91\ComfyUI\atlas_exports\codex_test\`
(`nuke_projection.py` + `.nk`, `camera.usda`).

## What this test case demonstrates

If all six pass, you've shown: MCP handshake from a non-Claude agent →
knowledge-resource transfer → upload+solve round-trip → the UI-format
flatten/gate machinery → server-side file read-back → multi-format DCC
export. That is the entire v1 surface except camera-move baking (browser-
bound by design — do that in the ComfyUI viewport).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Codex shows no `atlas` server | Config typo, or Codex started before the edit — restart Codex. `python -m atlas_camera.mcp` in a terminal should sit silently waiting on stdin (Ctrl-C to exit); if it errors, fix that first. |
| `atlas_health` → not reachable | ComfyUI isn't running on :8188. |
| `experimental_registered: false` | Server launched without `ATLAS_EXPERIMENTAL=1` — use the fixed bat. Only matters for the X-ray workflows. |
| T4 fails at validation | You're not running Codex from the repo root (relative path), or the workflow was hand-edited. |
| T5 "not found" | `COMFY_DIR` missing/wrong in the config's env block, or the ghosttown showcase hasn't been run on this server since its last restart (run it once — T4 style — and retry). |
| Codex re-asks approval on every call | Expected on first contact per tool; approve-for-session (or mark the project trusted) to quiet it. |

## Known asymmetries vs Claude Code

- Codex approval UX differs but the tools are identical — the server is a
  plain stdio MCP process with no client-specific behavior.
- Claude Code sessions in the repo pick the server up automatically from
  the committed `.mcp.json`; Codex needs the `config.toml` entry above
  (its per-project MCP support doesn't read `.mcp.json`).
