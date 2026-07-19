# `tools/generate_companion_workflows.py`

Generator for the two **companion showcase workflows** — the deeper, narrated
builds that sit alongside the simple quickstart graphs:

| Output file | What it demonstrates |
|---|---|
| `atlas_jungle_xray_cameramove.json` 🩻 | Single photo → predicted hidden geometry (LaRI X-ray) → LaMa clean plate → a bakeable **camera move** exported to Nuke/USD |
| `atlas_castle_dcc_handoff.json` 🎞 | Plate registration → solve → sky + body layers → **Nuke `.nk` / Maya `.ma` / USD / OBJ+GLB** DCC handoff |

They are "companions" to the castle DMP marketing set and share its builder, so
this script never hand-writes a widget list.

## Why it exists (and why it's drift-proof)

ComfyUI serializes a node's widgets into a **positional** `widgets_values`
array. Hand-saved workflows silently fall out of sync when a node gains an
appended widget (see `tests/test_shipping_workflow_widgets.py` and the drift we
had to backfill in the quickstart graphs). This generator sidesteps that
entirely: **every widget list is derived from a live `/object_info` snapshot**,
so the emitted arrays always match the current node definitions. Regenerate
after node changes and the companions stay correct by construction.

## Running it

```bash
# 1. Snapshot the live node definitions from a running ComfyUI
curl -s http://127.0.0.1:8188/object_info -o object_info.json

# 2. Generate both companions into a directory
python tools/generate_companion_workflows.py \
    object_info.json \
    examples/showcase \
    tools/generate_castle_dmp_workflow.py
```

Positional arguments (the script reads `sys.argv` directly, no argparse):

1. **`object_info.json`** — a `/object_info` dump from a ComfyUI that has the
   Atlas pack (plus the third-party packs the graphs use) installed.
2. **output dir** — where `atlas_jungle_xray_cameramove.json` and
   `atlas_castle_dcc_handoff.json` are written.
3. **path to `tools/generate_castle_dmp_workflow.py`** — the shared builder
   (see below).

It prints a one-line summary per file (`nodes= links= groups= rails=`).

## How it's built

### The shared `WF` builder
The heavy lifting lives in `generate_castle_dmp_workflow.py`, which exposes a
`WF` class that assembles a UI-format ComfyUI workflow and, crucially, looks up
each node's widget order/defaults from the `object_info` snapshot. This script
imports that module dynamically:

```python
sys.argv = ["gen", OI_PATH, str(_tmp)]      # spoof its argv…
cg = <import generate_castle_dmp_workflow.py>
_tmp.unlink(missing_ok=True)                # …and discard what it writes on import
WF = cg.WF
```

The spoof is necessary because the castle generator reads `argv` and writes a
file **at import time**; pointing it at a throwaway path lets us borrow its `WF`
class without producing a stray file.

`WF` methods used here:

- `w.node(type, [x,y], [w,h], title, {widget: value, …})` — an Atlas/registered
  node; unspecified widgets take their `object_info` defaults.
- `w.raw(type, pos, size, title, widgets, inputs, outputs, …)` — a fully
  hand-specified node, used for core/third-party nodes (`LoadImage`, KJ
  `SetNode`/`GetNode`).
- `w.link(src, out_slot, dst, input_name)` — wire an output to a named input.
- `w.group(title, [x,y,w,h], color)` / `w.note([x,y], [w,h], text)` — the
  titled boxes and the long explanatory sticky-notes.
- `w.dump()` — the final workflow dict.

### `Builder` + KJ rails
`Builder` wraps a `WF` and adds two helpers that keep the graphs readable by
routing shared signals through **KJNodes Set/Get rails** instead of long
cross-screen wires:

- `rset(name, src, slot, pos)` — drop a `SetNode` (`kijai/ComfyUI-KJNodes`) that
  publishes `src`'s output slot onto a named rail (`plate`, `solve`, `depth`,
  `hidden_mask`, …), remembering the rail's type in `self.sets`.
- `rget(name, pos)` — drop a matching `GetNode` that reads that rail.

So a typical stanza is "get the rails I need → make a node → wire them → set the
node's output onto a new rail":

```python
g1 = b.rget("plate", [360, 40])
solve = w.node("AtlasLearnedSolveFromImage", …)
w.link(g1, 0, solve, "image")
b.rset("solve", solve, 0, [800, 520])
```

### The two `build_*` functions
`build_xray()` and `build_dcc()` each construct one graph stage-by-stage
(numbered groups 0→4 that read left-to-right) and return `(builder, name)`. The
module-level loop at the bottom runs both, stamps the workflow `id`, and writes
the JSON.

**`build_xray` — jungle X-ray camera move** (experimental): plate → learned
solve → ✅ Solve Gate (approve before the heavy model runs) → MoGe depth →
`AtlasPredictHiddenGeometry` (LaRI) → LaMa clean plate via ✂ crop/stitch → a
base relief that projects the *original* photo plus an X-ray `AtlasCleanPlateLayer`
that uses **mask membership** (`hidden_mask`→grow→invert→`exclude_mask`,
`paint_matte`→`layer_matte`), not a depth band → viewport to author a camera
move + Nuke/USD export.

**`build_dcc` — castle DCC handoff**: `AtlasRegisterPlate` (Output Desk) →
learned solve → ✅ gate → `AtlasAttachSourcePlate` → depth + `AtlasSkyDomeLayer`
+ `AtlasCleanPlateLayer` → `AtlasExportNukeLayers` / `AtlasExportMayaLayers` /
`AtlasExportUSD` → a separate `AtlasDeriveReliefMesh` → `AtlasExportReliefMesh`
(OBJ/GLB + interior hole fill) with a viewport fill-preview.

The generous `w.note(...)` blocks in each build are the on-canvas documentation
that ships inside the workflow — they explain the calibration and the design
decisions to whoever opens the graph.

## Gotchas

- **The USD camera-path export in the X-ray graph is MUTED by design**
  (`cpu["mode"] = 4`). `AtlasExportCameraPathUSD` errors until the viewport's
  ⏺ Bake Path has produced a camera path — un-mute it *after* baking.
- **Seeds are pinned** (LaRI and LaMa at `seed=0`). ComfyUI auto-adds a
  "randomize" control to any widget named `seed`; leaving it would silently
  re-roll the generative geometry every queue.
- **The X-ray graph is experimental / research-only.** It needs
  `ATLAS_EXPERIMENTAL=1` set *before* Python starts and a user-cloned LaRI repo
  pointed at by `ATLAS_LARI_PATH` (LaRI has no upstream license, so nothing is
  vendored).
- **Third-party packs the graphs reference:** KJNodes (the Set/Get rails),
  comfyui-inpaint-nodes (LaMa), ComfyUI-RMBG (SAM3). Your `object_info` snapshot
  must include them or the corresponding `w.raw`/`w.node` widget lookups won't
  match a real install.

## Related

- `tools/generate_castle_dmp_workflow.py` — the `WF` builder this script reuses.
- `tools/generate_canonical_ocio_dcc_workflows.py` — the portable canonical
  OCIO/DCC generator (same `object_info`-driven, drift-proof pattern).
- `tests/test_shipping_workflow_widgets.py` — the guard that catches positional
  widget drift in the hand-saved shipping workflows this generator avoids.
