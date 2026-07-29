# Shoot project format (`atlas_shoot.json`)

The contract between Atlas and the iPhone app. Atlas writes it; the app reads
it, guides a photograph per entry, and sends the results back.

Written by `AtlasShootList 📸`. Worked example, generated rather than
hand-written so it cannot drift: [`shoot_project.example.json`](shoot_project.example.json)
(regenerate with `python tools/build_shoot_example.py`; a test compares them).

**This lives in `docs/`, not `docs/dev/`, deliberately** — `docs/dev/` is
gitignored, so anything in it does not survive a clone, and this file has to
reach the machine building the app.

---

## Why the file exists

Every hole Atlas fills today is invented: inpainting, predicted hidden geometry,
edge-dilated smear. A matte painter with a camera can often just go and
photograph the missing material — the paving, the brick, the render — not the
specific occluded scene, which may be unreachable, but the same KIND of surface
at the same angle and light.

Atlas already knows what is missing. This states it in terms a photographer can
act on.

## Shape

```json
{
  "version": 1,
  "plate_size": [4000, 3000],
  "shots": [ { ...one photograph to take... } ],
  "guidance": ["one human-readable line per shot, same order as `shots`"],
  "lighting": { "measured": false, "note": "..." },
  "notes": []
}
```

### `shots[]`

| field | meaning |
|---|---|
| `node_id` | stable id of the occluded surface in the occlusion graph |
| `subject` | what to point the camera at, e.g. `"pavement, kerb"` |
| `hidden_by` | what is covering it, e.g. `["bollard", "parked_car"]` |
| `incidence_deg` | **0 = square to the surface, 90 = edge-on** |
| `distance_m` | how far the plate sees this surface |
| `depth_range_m` | `[near, far]` across the visible extent |
| `px_per_m` | plate pixels per metre of surface — match or beat it |
| `tear_px` | how much is missing; the sort key |
| `priority` | 1 = shoot first (worst hole) |
| `surface_normal` | world-space unit normal, Y-up right-handed |
| `kind` | `ground` / `surface` / `object` / `backdrop` |
| `volumetric` | **true = no plane; do not shoot a flat texture** |
| `warnings` | per-shot cautions, safe to surface verbatim |
| `metadata` | completion/texture policy and fit confidence |

## The three things a client must get right

### 1. `incidence_deg` is the load-bearing number, and it is range-dependent

The same floor is steep underfoot and almost edge-on at the horizon, so "the
angle to the ground" is meaningless without a distance — the value here is
computed at that surface's own range. A paving slab photographed square will not
sit into a plate that sees it at 85°, however good the texture.

Guide the user to roughly this angle. It does not need to be exact: if the phone
records its own pose, Atlas can correct the residual.

### 2. `volumetric: true` means there is no surface to photograph

An alleyway, a doorway, a recess. No plane was fitted, so no angle describes it,
and `incidence_deg` is meaningless — it is `0.0` as a placeholder, not a
measurement. These need the user aligned against ghosted geometry rather than
shooting a flat texture. Branch on this flag; do not treat it as an angle of
zero.

### 3. Lighting is NOT specified, and that is a statement

`lighting.measured` is `false`. Atlas cannot measure sun direction or hardness
from a single plate, and rather than emit a confident-looking azimuth that is
really a guess, the project ships the plate itself (`reference_plate.png`,
written alongside) to match by eye.

**Do not read the absence of lighting fields as "lighting does not matter".** It
is the single thing most likely to make a patch read as a sticker. Flat or
overcast capture is safest, because Atlas can relight from surface normals it
already has.

If `lighting.measured` ever becomes `true`, fields will be ADDED alongside it;
clients should switch on the flag rather than on field presence.

## Versioning

`version` is an integer and increments on any breaking change. Fields are
**added**, never renamed or repurposed — the same append-only discipline Atlas
applies to node widgets, for the same reason: a client in the field cannot be
updated in lockstep.

A client should refuse a `version` it does not know rather than guess.

## Returning a shot

Send the photograph back as a **one-frame `.r3d`** — see
`docs/dev/atlas_iphone_app_spec.md` if you have it, or the summary below.

That container needs no new format work on the Atlas side: `Record3DCapture.open()`
requires only `w`/`h` and a pose, so a single-frame capture is a first-class
input rather than a special case.

Two conventions produce well-formed but WRONG geometry if reversed, and nothing
downstream can detect either:

* `K` is **column-major**: `[fx, 0, 0, 0, fy, 0, cx, cy, 1]`
* the pose quaternion is **scalar-last**: `[qx, qy, qz, qw, tx, ty, tz]`
  (Swift's `simd_quatf` exposes `.real` = w and `.imag` = xyz)

ARKit's basis is already Atlas's — x-right / y-up / z-back, metres. **No axis
remap, and the −Z canonicalization must not be applied.**

Atlas ingests the returned image through `AtlasAddPatchView`, which takes the
solve plus the patch image and its angle.
