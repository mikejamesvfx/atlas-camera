# Atlas Showcase — Text-to-Image Prompts (source plates)

Ten prompts, one per concept. Written to produce **good Atlas source plates**, not just
pretty pictures. Every prompt is engineered for the three things the solver + projection
need:

1. **Readable camera cues** — a visible horizon and/or strong converging perspective lines
   so the learned solve locks focal + tilt.
2. **Layered depth** — a distinct foreground element, a mid-ground subject, and a receding
   background, so the clean-plate parallax has something to separate.
3. **Consistent art direction** — a shared style anchor so the ten read as one body of work.

Works in Midjourney, Flux, or SDXL. Aspect ratios given per shot; Midjourney uses `--ar`,
Flux/SDXL set the dimensions. Generate 4+ per concept and pick the frame with the clearest
depth separation and the least lens distortion.

---

## Shared style anchor (append to every prompt for consistency)
> cinematic matte painting, shot on 35mm, subtle anamorphic character, muted filmic color
> grade, volumetric atmosphere, painterly photoreal detail, natural depth haze, no text, no
> watermark

## Shared negative prompt (SDXL/Flux)
> fisheye, extreme wide-angle distortion, warped horizon, tilted dutch angle, flat frontal
> lighting, collage, split screen, border, frame, text, watermark, low detail, oversaturated

## Composition rules baked into the prompts (why they're worded this way)
- "clear horizon line" / "strong linear perspective" → gives the solver its anchor.
- "foreground … mid-ground … distant" phrasing → forces the depth layering.
- "eye-level" or a named camera height → keeps tilt sane and the ground plane readable.
- Avoid "top-down / bird's-eye vertical" — a fully vertical aerial has no horizon to solve.
  The aerials below are **high oblique** (horizon still in frame) on purpose.

---

### 01 · The Approach — aerial forest to castle
> A high oblique aerial view over an endless rolling pine forest at dawn, low morning fog
> filling the valleys, a distant fantasy castle with tall spires rising from the mist on a
> far ridge, clear horizon line high in the frame, layered depth from near canopy to distant
> peaks, golden backlight. **[+ style anchor]**
`--ar 16:9` · *Composition: keep the near canopy detailed in the lower third, castle small on
the horizon — the distance sells the reveal.*

### 02 · Cathedral of Light — interior nave
> Interior of a vast gothic cathedral nave, symmetrical rows of towering stone columns
> receding to a distant altar, strong one-point linear perspective, god-rays streaming from
> high clerestory windows, dust in the light, eye-level camera on the central aisle, polished
> stone floor reflecting the light. **[+ style anchor]**
`--ar 2:1` · *Composition: dead-center vanishing point, columns clearly separated in depth for
the clean-plate layers.*

### 03 · The Siege View — clifftop fortress
> A medieval stone fortress with multiple towers and steep slate roofs perched on a dramatic
> sea cliff, three-quarter view showing two sides of the keep, crashing ocean far below, clear
> horizon over the sea, dramatic side lighting at late afternoon, foreground rocky outcrop.
> **[+ style anchor]**
`--ar 16:9` · *Composition: three-quarter angle (not straight-on) so an orbit has a genuine
occluded far side for the patch view to fill.*

### 04 · Storm Peaks — mountain range
> A range of towering snow-capped mountain ridgelines receding into atmospheric haze, layered
> silhouettes fading from dark foreground rock to pale distant peaks, dramatic storm light
> breaking through clouds, sharp foreground boulders, clear high horizon. **[+ style anchor]**
`--ar 21:9` · *Composition: at least three clearly separated depth layers of ridgeline — the
banding is the whole shot.*

### 05 · Ghost Town — abandoned street
> An abandoned dusty main street of a derelict western town at golden hour, weathered wooden
> storefronts receding down both sides in strong perspective, a rusted broken-down car and a
> fallen sign in the foreground, long shadows, clear horizon at the end of the street, eye-level.
> **[+ style anchor]**
`--ar 16:9` · *Composition: put the wrecked car + sign clearly camera-forward and to one side —
they're the parallax foreground the plates reveal behind.*

### 06 · Hangar Bay — sci-fi interior
> Interior of a colossal starship hangar bay, a docked spacecraft mid-ground, industrial
> gantries and walkways in strong linear perspective, massive blast doors in the distance,
> volumetric light beams, cool teal and orange practical lighting, eye-level wide view.
> **[+ style anchor]**
`--ar 2.39:1` · *Composition: anamorphic wide — strong orthogonal architecture gives the
room-cuboid solve clean walls; frame it for a 2.39 crop.*

### 07 · First Light — character on a ridge
> A lone cloaked figure standing on a rocky ridge seen from behind, facing a vast valley
> landscape opening below, mountains in the far distance, dawn light, the figure sharp in the
> mid-foreground for scale, clear horizon, natural eye-level camera. **[+ style anchor]**
`--ar 16:9` · *Composition: the human must be full-height and unobstructed — Atlas uses it as a
known-size scale reference to fix camera height.*

### 08 · Lost Temple — jungle reveal
> An ancient moss-covered stone temple half-swallowed by dense jungle, revealed in a clearing,
> massive carved pillars, thick foreground vines and giant leaves framing the shot, shafts of
> sunlight through the canopy, humid haze, layered depth from foreground foliage to temple to
> distant trees. **[+ style anchor]**
`--ar 16:9` · *Composition: heavy, distinct foreground foliage camera-close — the single most
convincing parallax cue when it rakes past.*

### 09 · Monolith — desert monument at dusk
> Towering angular sandstone mesa and monolith rock formations rising from a flat desert plain
> at dusk, long dramatic shadows, deep orange-to-violet dusk sky gradient, a small foreground
> rock for scale, crystal-clear low horizon, ultrawide vista. **[+ style anchor]**
`--ar 21:9` · *Composition: big simple angular forms read cleanly as derived geometry — ideal
for the "here's the mesh" breakdown and the RANSAC-plane pass.*

### 10 · Winter's Eve — alpine village at blue hour
> A cozy snow-covered alpine village at blue hour, a church with a tall spire at its center,
> warm glowing windows, snow-laden pine trees in the foreground, gentle chimney smoke, distant
> mountains, soft twilight, clear horizon, eye-level view down the main lane. **[+ style anchor]**
`--ar 16:9` · *Composition: foreground firs for parallax, spire as the vertical hero for the
towers pass, warm windows for the exposure-diagnostics demo.*

---

## Working tips
- **Generate wide, then don't crop away the horizon.** The solver wants it. If a favourite
  frame has the horizon cut off, out-paint a sliver of sky back rather than re-rolling.
- **Prefer moderate lenses.** Reject frames that look fisheye/ultra-wide-distorted — they
  fight the pinhole solve. The style anchor's "35mm" nudges the model this way.
- **Keep the grade consistent.** Once you find a look you like (contrast, color temp), reuse
  the same seed/style reference across all ten so the reel feels authored, not sampled.
- **One hero, then the rest.** Fully produce Concept 01 end-to-end first (still → solve →
  project → move → export) to validate the pipeline, then batch the other nine stills.
