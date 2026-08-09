<p align="center">
  <img src="assets/atlas_camera_icon.png" width="132" alt="Atlas Camera">
</p>

<h1 align="center">Atlas Camera</h1>

<p align="center">
  <b>One photograph in. A metric pinhole camera and a colour-managed projection<br>
  setup out — for Nuke, Maya, USD and Blender.</b>
</p>

<p align="center">
  <a href="https://github.com/mikejamesvfx/atlas-camera/actions/workflows/tests.yml"><img src="https://github.com/mikejamesvfx/atlas-camera/actions/workflows/tests.yml/badge.svg?branch=main" alt="tests"></a>
  <a href="https://registry.comfy.org/nodes/atlas-camera"><img src="https://img.shields.io/badge/ComfyUI_Registry-atlas--camera-eaa03a" alt="ComfyUI Registry"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2fb7a6" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/python-3.10+-c4b29a" alt="Python 3.10+">
  <a href="https://mikejamesvfx.com"><img src="https://img.shields.io/badge/a-mikejamesvfx_tool-c4b29a" alt="a mikejamesvfx tool"></a>
</p>

---

## Start here

Install into ComfyUI, then load **[`examples/atlas_input_quickstart_workflow.json`](examples/atlas_input_quickstart_workflow.json)**.

```bash
cd <ComfyUI>/custom_nodes
git clone https://github.com/mikejamesvfx/atlas-camera.git
```

Or install from the [ComfyUI Registry](https://registry.comfy.org/nodes/atlas-camera).

That workflow is the whole tool in one node. Point its `LoadImage` at any
photograph and queue:

```
LoadImage → AtlasInput → solved camera + projection geometry + viewport
```

**`AtlasInput` is the front door.** It runs the solve, derives projection
geometry, and hands you a scene you can look through — one node, sensible
defaults, no wiring. Everything else in the pack exists to take over a stage of
that chain once you need to.

In the ComfyUI Add-Node menu, **`Atlas` holds ten numbered folders in pipeline
order** — `01 · Input & Camera` through `10 · Export` — so the menu reads as the
job rather than as an alphabet. **`Atlas/advanced`** holds every gated node
(experimental, legacy, iOS): still supported, still tested, just not where you
start.

## What comes out

1. **Solve** — the camera from one photograph: focal length, orientation and
   horizon, with a confidence value. Deterministic geometric solve, or a learned
   prior (GeoCalib) for harder frames.
2. **Project** — projection geometry (relief mesh or fitted primitives), with
   the plate cast back through the recovered camera.
3. **Review** — a real-time fullscreen viewport with simple camera moves —
   dolly, orbit, pan — at your delivery resolution.
4. **Export** — a native setup for Nuke (`.nk` + Python), Maya (`.ma`), USD,
   Blender, and a relief mesh (OBJ/GLB with the projection baked into UVs).
   Verified in the real applications.

It **solves a camera, not a mesh.** Where most 3D nodes generate geometry from an
image, Atlas does the inverse-problem job a projection pipeline actually needs:
recover a real metric pinhole camera, then project the photograph onto derived
geometry. From the recovered viewpoint the plate reassembles exactly; scale error
shows only as parallax on a move — never as smeared texture.

Colour-managed and float-safe throughout: plates are tracked by reference in
their working colourspace (ACEScg) and bit depth (EXR 16/32-bit float), the
projection path stays floating-point, and it hands off to OpenColorIO, Nuke,
Maya and Resolve. Render format is a project-level camera up to **8192 px**.

## Install tiers

The dependency contract has three tiers: the **schema, solve JSON and DCC
exporters are dependency-free**; numerical camera recovery needs **NumPy**;
automatic line detection needs **NumPy + OpenCV** (`[vision]`). Every node
registers without heavy dependencies; a GPU is only needed for the optional
neural features.

| Tier | Install | Adds |
|---|---|---|
| **Core** | *(nothing)* | Schema, solve JSON, masks, DCC export — pure Python, no GPU |
| **`[vision]`** | numpy + opencv | Vanishing-point camera solve with line detection + debug overlays |
| **`[neural]`** | torch + GeoCalib | Learned solve, monocular depth, depth-driven geometry, patches |

Depth Anything V2 (`V2-Metric-Outdoor`) is the default depth backend — Apache-
licensed and transformers-only, so `[neural]` needs no extra install. MoGe-2
(`[moge]`, interior specialist) and Depth Anything 3 (`[neural-da3]`) are
selectable alternatives. Full setup is in **[INSTALL.md](INSTALL.md)**.

## Beyond the front door

Once `AtlasInput` isn't enough, the pack opens up stage by stage: a tiered
confirm-to-adopt metric scale cascade, composable geometry strategies combined
Nuke-Merge-style, the 2.5D matte-painting layer stack (sky dome, depth-band
clean plates, edge mattes, hole masks), camera-path authoring with baked frames,
and per-layer exports.

Rather than list them here, the full catalogue — every node, its inputs, outputs
and the rule it obeys — lives in **[docs/NODE_CATALOG.md](docs/NODE_CATALOG.md)**.
For how the pieces fit together, see the
**[ecosystem guide](docs/ECOSYSTEM_GUIDE.md)**.

Ten more workflows ship alongside the quickstart in
[`examples/`](examples/) — metric scale from references, layered projection,
camera moves and patches, plate finishing, export fan-out, occlusion analysis,
and an agentic variant with a terminal VLM/solve report for headless
automation. Point any of their `LoadImage` nodes at a photograph of your own.

Experimental nodes stay hidden unless you set `ATLAS_EXPERIMENTAL=1` before
launching ComfyUI. Nothing in the pack needs Docker, a Blender install, or a
user-cloned research model.

## Documentation

- [Install guide](INSTALL.md) — dependency tiers and optional backends
- [Technical brief](docs/TECH_AND_DIFFERENTIATION.md) — camera solve + projection vs mesh generation
- [User guide](docs/USER_GUIDE.md) · [Node catalog](docs/NODE_CATALOG.md) · [Ecosystem guide](docs/ECOSYSTEM_GUIDE.md)
- [Camera moves & marketing renders](docs/CAMERA_MOVES.md) — single photo → Nuke dolly
- [DCC exports](docs/DCC_EXPORTS.md) · [Third-party & licenses](THIRD_PARTY.md)
- **MCP server** — `pip install atlas-camera[mcp]` then `python -m atlas_camera.mcp`: drive Atlas from any MCP-capable assistant — [usage guide](docs/MCP_SERVER.md). A repo checkout auto-registers it for Claude Code via `.mcp.json`
- [Changelog](CHANGELOG.md) · [Roadmap](docs/ROADMAP.md)

## License

Atlas Camera is **[MIT](LICENSE)** — free for commercial use. It vendors nothing
restrictive; every optional model or package is installed by the user, and its
node fails soft with an informative message when absent.

**No node Atlas registers depends on a non-commercial model.** One caveat
remains and it is yours to choose: Depth Anything V2's **large** weights are
CC BY-NC 4.0, while its small/base weights are Apache 2.0 — pick the variant
that suits your use. Everything else — the solve, geometry, layer stack,
viewport, and the full OpenColorIO output path — is unrestricted. Full map in
**[THIRD_PARTY.md](THIRD_PARTY.md)**.

---

<p align="center"><sub>A <a href="https://mikejamesvfx.com">mikejamesvfx</a> tool · MIT · built for matte painters and environment artists.</sub></p>
