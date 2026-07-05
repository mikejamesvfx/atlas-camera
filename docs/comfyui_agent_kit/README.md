# Atlas Camera skill for ComfyUI-Agent-Kit

A drop-in **Node Library entry** that teaches a
[ComfyUI-Agent-Kit](https://github.com/SlavaSexton/ComfyUI-Agent-Kit) agent (Claude Code, Codex, Gemini CLI, or
Qwen Code) how to drive the **AtlasCamera** ComfyUI node pack — recover a camera from one photo, derive
projectable 3D geometry, project the photo onto it, and export to Maya/Nuke/Blender/USD.

The kit stores custom-node-pack knowledge as `NODE_LIBRARY/<pack>.md` files (see its `NODE_LIBRARY/_SCHEMA.md`
and the `ocio.md` example). [`atlas-camera.md`](atlas-camera.md) here is authored in exactly that format:
durable semantics (what each node/wire is FOR, how to wire the pipeline, gotchas, placement) on top of the live
`get_node_info` I/O the agent confirms at runtime.

## Prerequisite

The AtlasCamera pack must be installed in the target ComfyUI so the nodes actually exist:

```bash
# symlink atlas_camera/comfy -> <COMFYUI>/custom_nodes/AtlasCamera, then in ComfyUI's venv:
pip install -e ".[neural]"
pip install "git+https://github.com/cvg/GeoCalib.git"
# restart ComfyUI
```

## Install the skill knowledge

Copy `atlas-camera.md` into the kit's Node Library for whichever agent(s) you use, then register it in the index.

**Claude Code** (skill or plugin install of the kit):
```bash
cp atlas-camera.md ~/.claude/skills/comfyui/NODE_LIBRARY/atlas-camera.md
```
**Codex:** `~/.agents/skills/comfyui/NODE_LIBRARY/`  ·  **Gemini CLI:** `~/.gemini/extensions/comfyui/NODE_LIBRARY/`
·  **Qwen Code:** `~/.qwen/extensions/comfyui/NODE_LIBRARY/`

Then add a row to that same folder's `_INDEX.md` (the kit's "one front door" — the agent routes here for any node
question), under **## Documented nodes**:

```markdown
### `atlas-camera.md` - single-image camera recovery + matte-painting projection (AtlasCamera, Miike Burns)
| Entry | Purpose |
|-------|---------|
| `atlas-camera.md` - the AtlasCamera pack | recover a camera from one still (real/AI) -> derive projectable relief-mesh / fitted-primitive geometry -> project the photo onto it live in the Three.js viewport (matte-painting) -> export to Maya / Nuke / Blender / USD; incl. multi-angle AI-patch fill for occluded areas. Two custom wire types: ATLAS_SOLVE / ATLAS_CAMERA (Atlas-only). Solve/derive/patch/relief need the [neural] extra. |
```

## Keeping it honest (the kit's two laws)

- **Live vs curated:** the agent should confirm exact inputs/outputs/defaults with `get_node_info <ClassType>`
  (`/object_info`) — those change between versions. `atlas-camera.md` deliberately does NOT freeze full I/O; it
  holds the semantics `/object_info` can't give.
- **Confirmed vs inferred + date:** the entry's I/O is marked *confirmed from the pack source `INPUT_TYPES`
  2026-07-03*. When AtlasCamera ships new nodes/fields, update `atlas-camera.md` (and re-sync the copies above),
  matching the `_SCHEMA.md` entry template.
