# The Atlas MCP Server — driving Atlas from any AI agent

Atlas ships an [MCP](https://modelcontextprotocol.io) server that lets any
MCP-capable agent — Claude Code, Claude Desktop, Cursor, Windsurf, a custom
Agent-SDK app — operate a running ComfyUI + Atlas install: solve cameras,
run and validate workflows, read debug reports, and export to Nuke, Maya,
Blender, and USD. The agent gets the *operational knowledge* too: the server
publishes Atlas's calibration doctrine and gate semantics as resources, so
an assistant that has never seen this repo still knows which depth model
fits an interior, why its first Queue stopped at a gate, and how to fix a
wrong metric scale.

The server is a thin translator, not an engine: it talks to ComfyUI over
HTTP and never loads torch or runs models itself. ComfyUI stays the
execution engine; kill the MCP process at any time and nothing is lost.

---

## 1 · Prerequisites

1. **A running ComfyUI with the Atlas node pack loaded** (see
   [INSTALL.md](../INSTALL.md)). The MCP server is a client of that server —
   it does nothing useful while ComfyUI is down (`atlas_health` will tell
   you exactly that).
2. **The `[mcp]` extra** in whichever Python environment will launch the
   server:

   ```bash
   pip install -e ".[mcp]"        # from a repo checkout
   # or, on an existing install:
   pip install "mcp>=1.2"
   ```

3. Optional, for the experimental X-ray/RenderFix workflows: launch ComfyUI
   with `ATLAS_EXPERIMENTAL=1` (and `ATLAS_LARI_PATH` for LaRI).
   `atlas_health` reports whether those nodes are registered.

**Smoke test** (no agent needed — should print a JSON-RPC banner and wait
on stdin; Ctrl-C to exit):

```bash
python -m atlas_camera.mcp
```

## 2 · Configuration (environment variables)

| Variable | Default | What it does |
|---|---|---|
| `COMFY_HOST` | `127.0.0.1:8188` | The ComfyUI server to drive. |
| `COMFY_DIR` | *(unset)* | ComfyUI's root directory on disk. Optional — set it and the server can read back solve JSONs and debug reports that nodes write server-side (they're written relative to ComfyUI's cwd). |
| `ATLAS_REPO` | *(unset)* | A repo checkout path, for doc-backed resources. |

## 3 · Registering with your agent

### Claude Code

A checkout of this repo already contains `.mcp.json` at the root — open
Claude Code inside the repo and approve the server when prompted. Done.

To register it user-wide (any project) instead:

```bash
claude mcp add atlas --scope user -e COMFY_HOST=127.0.0.1:8188 -- python -m atlas_camera.mcp
```

### Claude Desktop

Add to `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "atlas": {
      "command": "python",
      "args": ["-m", "atlas_camera.mcp"],
      "env": { "COMFY_HOST": "127.0.0.1:8188" }
    }
  }
}
```

### Cursor / Windsurf / anything with an `mcpServers` block

Same JSON shape as above, in the client's MCP config file
(`.cursor/mcp.json`, `~/.codeium/windsurf/mcp_config.json`, …).

### A custom agent (Claude Agent SDK, LangChain, etc.)

It's a standard stdio MCP server: spawn `python -m atlas_camera.mcp` with
the env vars set and speak MCP over stdin/stdout. With the Claude Agent
SDK, pass it in `mcp_servers` exactly as you would any stdio server.

> **Which `python`?** The one you type in the config must resolve
> `atlas_camera` and `mcp`. If Atlas lives in a venv (e.g. ComfyUI's), use
> that venv's full interpreter path as `command` instead of `python`.

## 4 · The tools

Everything returns compact JSON. Paths passed **to** tools are paths on
*your* machine (the agent's filesystem); paths that come back marked
"server-relative" are relative to ComfyUI's working directory — set
`COMFY_DIR` and the reading tools resolve them for you.

| Tool | What it does | The call you'll actually make |
|---|---|---|
| `atlas_health` | Is ComfyUI up? Version, free VRAM, how many Atlas nodes registered, whether the experimental nodes loaded, which third-party packs are missing. | *Always call this first.* No arguments. |
| `atlas_solve_image` | Upload a photo, recover its camera, return the solve summary (focal, FOV, height, confidence, `scale_source`). | `image_path` (local file); `method` `"learned"` (GeoCalib, default) or `"vp"` (classical vanishing points); `camera_height_m > 0` adds a scale override for elevated vantages. |
| `atlas_run_workflow` | Flatten a UI-format workflow JSON (official v1 subgraphs expanded, proxy widgets applied, KJ Set/Get rails resolved, muted/bypassed nodes handled) and run it to completion. | `workflow_path`; `open_gates=True` (default) opens shipped-closed `AtlasSolveGate`s; `overrides={"12.image": "my.png"}` retargets any widget by node id. |
| `atlas_validate_workflow` | Lint a workflow—including nodes inside official subgraphs—against the *live* server definitions: positional-widget drift, broken links, out-of-range widgets, dangling rails. | `workflow_path`. Run before `atlas_run_workflow` on hand-edited JSON. |
| `atlas_read_debug_report` | Read the 🔍 `AtlasDebugReport` JSON — per-layer vertex counts, band ranges, matte coverage, red flags. The one-file autopsy for "why is this layer empty". | `json_path` (default `atlas_debug/master_debug.json`, server-relative). |
| `atlas_inspect_viewport` | Layer census of a viewport's last execution: projection sources, priorities, band ranges, geometry type, vertex counts, synthesized fill cells, tear/stretch QA, and camera meta. | `node_id` of the `AtlasBlockoutViewport` in the workflow you just ran. |
| `atlas_export_scene` | Fan a saved solve into DCC exporters. | `solve_json_path` (server-relative, from `AtlasExportSolveJSON`); `formats` ⊆ `nuke · nuke_layers · maya_layers · maya_review · blender · usd · review_package`. Layer formats need a solve carrying projection sources. |
| `atlas_node_catalog` | Machine-readable list of every Atlas node on the server with inputs (widget vs link) and outputs. | Optional `name_filter` substring. |

**Deliberately absent:** camera-move *baking* (`⏺ Bake Path`) is
browser-side WebGL and can't run over HTTP — author and bake moves in the
ComfyUI viewport, or drive a real browser. Everything else here is fully
headless.

## 5 · The resources (the knowledge layer)

Agents should read these before making Atlas decisions — they exist so the
doctrine travels with the tools:

- **`atlas://calibration`** — the per-scene playbook: which depth model for
  exteriors vs interiors, relief grid/edge values per band, seam-priority
  rules, the cleanplate-derived hidden-support doctrine, and the metric-scale
  doctrine (including the sky-rise rule: on any
  plate with buildings, *count a visible building's storeys × 3.5 m* into
  `AtlasReferenceScaleSolve` rather than eyeballing a height — a
  plausible-looking guess was measured ~2.5× off on a real plate).
- **`atlas://gates`** — why a first Queue "finishes" in seconds: the
  ExecutionBlocker gate family, which widget opens each gate, and how
  fingerprinted approvals re-arm on new images.
- **`atlas://schemas/solve`** — the shape of the solve JSON the tools
  return and consume.

## 6 · A typical session (what the agent actually does)

```
1. atlas_health
     → {"ok": true, "atlas_nodes": 71, "experimental_registered": true, ...}
        (67 standard + 4 experimental when ATLAS_EXPERIMENTAL=1; 67 without)

2. read atlas://calibration        (agent now knows the doctrine)

3. atlas_solve_image  image_path="C:/plates/street.jpg"
     → focal 21.5mm, height 1.6m, scale_source "assumed_default"   ← flag!

4. agent sees assumed_default on an elevated shot → counts storeys on a
   building in the plate → re-solves or runs a workflow with
   AtlasReferenceScaleSolve / camera_height_m

5. atlas_run_workflow  workflow_path="examples/atlas_camera_staged_master_workflow.json"
     → completed, node errors verbatim if any

6. atlas_read_debug_report         (layer census, red flags)

7. atlas_export_scene  solve_json_path="atlas_exports/.../solve.json"
                       formats=["nuke_layers", "usd"]
     → written file paths under output_dir
```

For subject-removal workflows, inspect the layer census before export. A
full-frame background cleanplate should normally report zero synthesized fill
cells and carry relief derived from a second depth solve of the approved
cleanplate; the original-depth foreground should be explicitly matted. A large
`n_filled_cells` count beneath a removed subject is a warning that the graph is
using far-band diffusion instead of continuous support geometry.

## 7 · Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `atlas_health` → `ComfyUI not reachable` | ComfyUI isn't running, or `COMFY_HOST` points at the wrong port. |
| `experimental_registered: false` | ComfyUI was launched without `ATLAS_EXPERIMENTAL=1`. Restart it with the env var **set before python starts**; workflows using 🩻 X-ray / 🔬 RenderFix will otherwise fail with `missing_node_type`. |
| A run "succeeds" in ~2 s with almost no outputs | A gate is closed. `atlas_run_workflow(open_gates=True)` handles `AtlasSolveGate`; a VLM image gate needs `auto_continue` or a `{"<id>.proceed": true}` override. See `atlas://gates`. |
| `missing_third_party` lists packs | A workflow explicitly references those external nodes. The three shipping workflows do not require KJ/rgthree/ComfyUI-RMBG; AtlasInput's optional legacy LaMa mode self-skips when its pack is absent, while the staged master uses native SDXL nodes. |
| Solve JSON / debug report "not found" | Those files are written server-side, relative to ComfyUI's cwd. Set `COMFY_DIR` so the MCP server can resolve and read them. |
| Metric numbers look ~10× small | `scale_source: "assumed_default"` — the elevated-vantage trap. Count storeys (see `atlas://calibration`) or pass `camera_height_m`. |

## 8 · Design notes

The v1 tool surface is not speculative: every tool maps to an operation
that was actually needed while driving Atlas headlessly during the
showcase-verification sessions. The
UI→API flattening the tools share lives in
`atlas_camera/mcp/comfy_http.py` (stdlib-only) and is unit-tested offline
in `tests/test_mcp_comfy_http.py`; the CLI twins
(`tools/run_ui_workflow.py`, `tools/validate_ui_workflow.py`) are thin
wrappers over the same module, so the terminal and the agent can never
drift apart.
