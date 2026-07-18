# Atlas v1 edge/outlier cleanup plan

The attached city result shows two different failure classes that should not be
treated with one global threshold:

1. **Stretched shards** — a triangle survives a tear test and its baked UVs
   smear across a depth discontinuity.
2. **Black tears** — the mesh correctly refuses a bad triangle, but there is no
   replacement surface for the novel view.

## Landed in this pass

`build_relief_mesh` now computes the world-edge budget from the triangle's
**median** depth rather than `dmax`. One hallucinated far-depth corner can no
longer inflate the budget enough to preserve a frame-spanning shard. The
existing depth-ratio and normal-bend tests still decide whether the triangle is
valid.

The new `quad_coherence` guard addresses a subtler case: when one triangle of a
grid quad fails and the other survives, the surviving diagonal can still
interpolate a long UV wedge. Final relief-node output rejects both halves of
that quad, producing a clean matte-compatible hole. The low-level mesh API
keeps partial quads available for callers that explicitly prefer coverage.

Relief metadata now also records `stretch_ratio_p95` and
`stretch_fraction_gt12`. These are QA signals for triangles that technically
pass validity but are highly anisotropic in world space; the debug report can
recommend a card or segmented inpaint fallback before an artist sees a
frame-spanning smear.

## Recommended v1 stack

### 1. Detect, don't smooth, isolated depth outliers

Add a confidence mask from a local median/MAD test and a normal-consistency
test. Feed it to `exclude_mask` so bad cells become explicit holes. Smoothing a
large outlier moves the error into neighbouring geometry and creates a softer,
harder-to-debug stretch.

### 2. Use instance mattes for clean-plate inference

The new `AtlasSegmentedSDXLInpaint` path uses SAM3 `Separate` building masks,
intersects each instance with LaRI's `paint_matte`, and inpaints each building
crop independently. This prevents one city-wide SDXL crop from inventing a
single connected mega-structure.

### 3. Separate silhouette coverage from texture coverage

For a torn boundary, use a tiny receding geometry skirt only outside the
trusted silhouette and cut it with the full-resolution matte. Never extend the
textured mesh by copying UVs without a matte. This hides grid stepping while
preventing the stretched-texel ribbons visible in the screenshot.

### 4. Add a view-dependent confidence fallback

Every projection source should carry a confidence alpha made from:

- hole mask,
- synthesized-depth mask,
- distance from the recovered camera,
- facing ratio,
- edge/outlier distance transform.

At the recovered camera, the original plate wins. As the camera moves, trusted
relief wins first, then segmented SDXL plates, then a sky/backdrop card. This
turns hard black pops into a controlled degradation rather than exposing a
single questionable mesh triangle.

### 5. Keep the sky out of relief entirely

The temple/city workflows should use `AtlasSkyDomeLayer` for sky pixels. A
relief mesh should never be asked to explain cloud or glass reflections; those
regions are the highest-leverage source of giant outliers.

### 6. v1 acceptance tests

- no triangle edge exceeds the local median-depth edge budget;
- no projected pixel uses a synthesized-depth texel at the recovered camera;
- no instance crop spans more than one disconnected building unless explicitly
  selected;
- orbit at ±3° and ±6° has no frame-spanning UV streaks;
- black coverage is reported as a mask/percentage, never silently hidden;
- Nuke/Maya exports receive the same masks and confidence metadata as the
  viewport.

The goal is not to force every pixel to remain textured at every orbit angle.
The v1 goal is an honest, layered result whose failures are local, soft, and
artist-controllable instead of global stretched shards.
