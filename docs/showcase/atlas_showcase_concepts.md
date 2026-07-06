# Atlas Camera — 10 Showcase Concepts

For the user guide + showcase site. Each concept is a recognizable VFX / matte-painting
shot chosen to spotlight a *different* cluster of Atlas features, so the reel doubles as
a complete capability tour. The through-line USP on every one: **a real parallax camera
move from a single still, no video model, with a clean DCC handoff.**

Legend — feature tags map to real nodes/modes:
`solve` = AtlasLearnedSolveFromImage · `relief` = relief mesh · `walls/towers/roofs/room`
= derive strategies · `scene_type` = preset · `patch` = AtlasAddPatchView · `plates` =
inpaint clean-plate layers · `path` = Camera Path mode · `shotcam` = AtlasDefineShotCam ·
`scale` = reference/VLM scale · `export` = DCC/USD/relief-mesh export.

---

## 1. "The Approach" — aerial reveal over a forest to a mystical castle
**The shot:** high drone push over a rolling pine canopy; a valley opens and a spired
castle rises out of the mist in the distance. Slow crane-down as we near it.
**Atlas spotlight:** `scene_type=forests` (high-density relief for canopy) merged with
`towers_spires` for the castle silhouette (`AtlasMergeGeometry`) · `path` crane/dolly ·
atmospheric far `backdrop`.
**Why it sells:** the hero example. Canopy parallax + a distinct architectural subject in
one frame proves the merge-two-strategies story. The mist hides the 2.5D flatness at
distance — a real matte-painter trick, built in.
**Source still:** a Midjourney/Flux aerial "epic fantasy castle above a pine forest, low
morning fog, cinematic" plate.

## 2. "Cathedral of Light" — interior dolly through a vast nave
**The shot:** locked dolly gliding forward down a cathedral aisle, columns sweeping past
in strong parallax, god-rays from clerestory windows.
**Atlas spotlight:** `scene_type=indoor` / `room_cuboid` (Manhattan floor+walls+ceiling) ·
`plates` for the column layers so the dolly reveals aisle behind each pillar · `path`
straight dolly-in.
**Why it sells:** the clearest demonstration of clean-plate inpaint layers — every column
you pass *should* black out on a naive projection; instead the aisle behind it is
inpainted and holds. Interiors are also the case people assume single-image projection
can't do.

## 3. "The Siege View" — orbit around a clifftop fortress
**The shot:** 180° orbit around a fortress perched on a sea cliff; towers and sloped roofs
reveal their far sides as we come around.
**Atlas spotlight:** `towers_spires` (silhouette-accurate extrusion) + `roofs` (RANSAC
sloped planes) · `patch` — a Qwen multi-angle novel view fills the occluded back of the
keep · `path` orbit preset.
**Why it sells:** the multi-angle patch pipeline's headline use — an orbit is exactly
where single-photo projection fails, and the patch view visibly rescues the far side.
Great side-by-side: "orbit without patch (black)" vs "with patch."

## 4. "Storm Peaks" — slow push into a mountain range
**The shot:** near-static, achingly slow push toward layered ridgelines fading into
haze; foreground rocks slide in parallax.
**Atlas spotlight:** `scene_type=mountains` (relief at high quality) · depth-graded
`plates` for 2–3 ridge bands · subtle `path` push.
**Why it sells:** pure terrain / organic relief mesh with no primitives at all — shows
Atlas handles free-form nature, not just architecture. The atmospheric depth banding is a
textbook matte technique and reads as "expensive."

## 5. "Ghost Town" — parallax dolly down an abandoned main street
**The shot:** low dolly down a derelict western/urban street; wrecked cars and a fallen
sign in the foreground rake past, storefronts recede.
**Atlas spotlight:** `walls` (azimuth walls + foreground boxes/cylinders for cars) ·
`plates` behind the foreground wrecks · `scale` (car proxy pins metric camera height) ·
`path` dolly.
**Why it sells:** the 2.5D urban parallax classic, and the scale-reference feature gets a
natural home (a known-size sedan sets the camera height, so the person/car proxies sit
correctly). Strong "before/after depth-layer" breakdown.

## 6. "Hangar Bay" — anamorphic reveal in a sci-fi interior
**The shot:** 2.39:1 anamorphic slow truck across a starship hangar; gantries and a
docked ship in mid-ground, blast doors beyond.
**Atlas spotlight:** `room_cuboid` + primitive `walls` · `shotcam` (defines a 2.39
anamorphic project format so the viewport + export conform) · `export` to Nuke projection.
**Why it sells:** the only concept that leads with **ShotCam** — shows Atlas respects a
real project format (sensor/lens/aspect) independent of the source plate's dimensions.
Speaks directly to compositors: "this drops into my Nuke script at the right format."

## 7. "First Light" — establishing character shot with a boom-up
**The shot:** a lone figure on a ridge, back to camera; a gentle boom-up and slight push
expands the vista behind them.
**Atlas spotlight:** `scale` (VLM/reference — the human sets metric camera height) ·
person proxy for instant sanity-check · `relief` background · `path` boom preset.
**Why it sells:** foregrounds Atlas's **measured, not assumed** metric grounding — the
1.8 m figure proves the camera height and lens are physically right, which is what makes
the parallax feel real rather than warped. Good "the math is correct" beat for the guide.

## 8. "Lost Temple" — crane reveal through jungle to a ruined temple
**The shot:** crane up and over dense foliage to reveal a moss-covered temple in a
clearing; vines and leaves in the extreme foreground.
**Atlas spotlight:** `scene_type=organic` / `forests` relief for the jungle · `patch` to
see around a temple pillar as we crane past · `plates` for the foreground foliage layer ·
`path` crane.
**Why it sells:** combines organic relief + patch view + foreground plates in one
crowd-pleaser — the "everything at once" shot for the sizzle reel finale. Foreground
foliage parallax is the single most convincing 2.5D cue.

## 9. "Monolith" — wide orbit around a desert monument at dusk
**The shot:** ultrawide slow orbit around towering mesa/monolith forms, long shadows,
dusk gradient sky.
**Atlas spotlight:** `scene_type=outdoor` + `roofs` (RANSAC any-orientation planes for
the angular rock faces) · `shotcam` wide format · `path` orbit · `export` relief mesh
(OBJ/GLB) to ZBrush/Blender.
**Why it sells:** big, simple, geometric forms make the recovered geometry legible — ideal
for the guide's "here's the derived mesh" diagram. Ends on the **export** story: the same
mesh opens textured in Blender/ZBrush.

## 10. "Winter's Eve" — cozy village push with glowing windows
**The shot:** gentle push toward a snowy alpine village at blue hour; a church spire, warm
window lights, snow-laden foreground firs sliding past.
**Atlas spotlight:** `simple_walls` + `towers_spires` (church spire) · `plates` for the
foreground firs · viewport `☀ Exposure` + `📊 Diagram`/`ℹ Info` diagnostics on screen ·
gentle `path` push.
**Why it sells:** the "diagnostics + polish" showcase — a chance to show the VP/horizon
overlay, the camera HUD, and exposure control while still delivering a warm, sellable
matte. Proves Atlas is a *working* tool, not just a demo.

---

## How to use this set across the site

- **Sizzle reel order:** 1 → 8 → 3 → 5 → 6 (open epic, peak with the "everything" shot,
  land on the pro/compositor note). Save 4 and 10 as the quieter beats.
- **Feature-coverage matrix** (so nothing's un-demoed):

| Feature | Concepts |
|---|---|
| relief mesh (organic/terrain) | 1, 4, 8 |
| primitive walls / boxes | 5, 6 |
| towers/spires (silhouette) | 1, 3, 10 |
| RANSAC roofs/facades | 3, 9 |
| room cuboid (interior) | 2, 6 |
| merge strategies | 1 |
| multi-angle patch view | 3, 8 |
| clean-plate inpaint layers | 2, 4, 5, 8, 10 |
| metric scale (reference/VLM) | 5, 7 |
| ShotCam project format | 6, 9 |
| Camera Path (crane/orbit/dolly/boom) | all |
| DCC / mesh export | 6, 9 |
| viewport diagnostics | 10 |

- **Per-concept page template for the guide:** source still → recovered camera + derived
  geometry (grey) → 📽 Project (Camera View) → the move (baked video) → optional DCC
  handoff frame. That five-panel arc *is* the Atlas pitch, repeated ten ways.

## Asset-generation note

Every source plate can be a single AI still (Flux / Midjourney / SDXL) — which is itself
on-message, since Atlas's learned solve is robust on AI-generated imagery. Keep a
consistent art direction (one lens feel, one grade) across all ten so the reel reads as a
body of work rather than ten unrelated tests.
