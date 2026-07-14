# Experimental / demonstration workflows

These are **not** the shipping catalog. The three pinned catalog workflows
(`atlas_input_quickstart`, `atlas_camera_staged_master`,
`atlas_input_ocio_quickstart`) live in the parent `examples/` directory and are
locked by `tests/test_example_workflows.py`. Everything here is a hand-built
demonstration of a specific technique — useful to open, learn from, and adapt,
but not part of the supported/tested surface. Image paths point at the author's
local ComfyUI `input/` (e.g. `moge_tests/…`); repoint `LoadImage` to your own.

## Scope layers (hero object at MoGe detail, backdrop as a clean flat plane)
- `atlas_scope_monument_valley.json` — sky dome + **V2 terrain relief** (receding
  desert + distant buttes) + **MoGe rock relief** (matted buttes/spires, priority 20).
- `atlas_scope_seacliff_castle.json` — sky dome + **flat sea card** backdrop
  (`band_geometry=card`) + **MoGe castle relief** (matted, priority 20).

  Recipe: a flat `card` backdrop when the far field is featureless (sea, haze);
  a V2 `relief` backdrop when it has real receding structure you want parallax on.
  Both use the original photo directly — Camera View is pixel-clean; orbiting
  reveals the hole behind the foreground (add a clean plate / inpaint for
  orbit-clean holes).

## Hand-plate DMP (castle removal → ocean plane + sky dome + MoGe foreground)
- `atlas_castle_dmp.json` — inpaint the castle/ground away, project ocean on a
  ground plane and sky on a dome, merge with a MoGe relief solve (band box + cutout).
- `atlas_castle_dmp_experimental.json` — same, plus a LaRI X-ray hidden-geometry
  layer where the castle needs occluded continuation.

## Multi-layer 2.5D + split depth (MoGe foreground / V2 backdrop)
- `atlas_00022_multilayer.json` — sky + banded clean-plate layers (Photoshop plate),
  Nuke + Maya layer exports.
- `atlas_00022_multilayer_moGe.json` — the split-depth variant: MoGe for the
  detailed machine + hillside layer, V2 for the flat backdrop/terrain + sky.
- `atlas_00022_templecity.json` — temple-city plate + clean plate (SAM confidence
  0.3 for the large smooth central dome).

## MoGe-only (single depth, single relief — interiors & object scenes)
- `atlas_moge_only.json` — the minimal MoGe path: LoadImage → learned solve →
  `AtlasDepthMap` (MoGe) → `AtlasDeriveReliefMesh` → viewport.
- `atlas_moge_atlas_*.json` — the same graph across the `moge_tests` batch
  (hangar/cathedral interiors are ideal; big-sky exteriors auto-cull sky to the
  backdrop — for those, prefer a scope-layer split above).
