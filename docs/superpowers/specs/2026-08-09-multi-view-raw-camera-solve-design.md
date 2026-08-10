# Deterministic Multi-View RAW Camera Solve

**Date:** 2026-08-09  
**Status:** Approved design; spec-panel reviewed 2026-08-09 (thresholds inlined, dependency surface named, failure diagnostics routed to atlas_debug)

## Purpose

Atlas must combine two or three overlapping photographs of one static scene into a single reproducible camera rig. The primary use case is a photographer moving sideways between street-scene photographs. Rotation-only capture is also supported when the environment prevents translation.

The same solve may later receive Qwen Multiple-Angles LoRA views. Photographed views and generated views remain different evidence classes: photographed pixels determine the measured camera rig, while Qwen cameras come only from their declared view angles. Generated pixels never influence measured camera registration.

## Capture Contract

**2026-08-10 revision — trusted-EXIF and burst capture.** Camera-processed
JPEGs are accepted alongside RAW: their EXIF carries the same body/lens/focal
evidence, the camera's own engine already applied lens correction, and they
import at a stamped lower trust tier (`metadata_source="jpeg_exif"`,
`undistort_status="camera_processed"`). Sets may also be BURSTS of up to 16
frames via `AtlasMultiViewSolveBurst`: beyond three frames the solver fits an
anchor-star pair topology — (photo 1, i) pairs only — so every frame must
share overlap with photo 1; the three-frame closure constraint is unchanged.

Version one accepts an ordered set of two or three photographs (or an
anchor-star burst, above) that:

- show a static scene with substantial overlap;
- come from the same camera body and lens at one focal length;
- retain their RAW metadata;
- use the same geometric development and undistortion settings; and
- preferably contain textured ground, architectural edges, and features at several depths.

Photo 1 is the explicit anchor. It defines the primary camera and Atlas world frame. Reordering the inputs deliberately changes the anchor, while rerunning an unchanged ordered set must not change the result.

Translated captures should use roughly 70% overlap and a lateral movement of approximately 0.5-1.5 metres. A measured lens-centre height is required for deterministic metric scale. Rotation-only captures may share one optical centre and contribute photographed projection coverage, but they do not provide baseline-derived geometry.

## Public Node Contract

Add an `AtlasMultiViewSolve` ComfyUI node with these link inputs:

- required `image_1` and `image_2`;
- optional `image_3`;
- matching optional `raw_meta_1`, `raw_meta_2`, and `raw_meta_3`;
- matching optional `plate_ref_1`, `plate_ref_2`, and `plate_ref_3`.

Its widgets include:

- `capture_mode`: `auto`, `translated`, or `rotation_only`;
- `camera_height_m`: measured from the ground to photo 1's lens centre, with
  `0.0` meaning unset; translated mode fails until a positive value is entered;
- `match_quality`: `balanced`, `conservative`, or `permissive`, with
  `balanced` as the default; and
- `seed`: an integer mixed into the content-derived deterministic sample
  schedule, defaulting to `0`. Every seed value is equally deterministic; it
  exists as an escape hatch when the content-derived schedule happens to land
  on degenerate samples, letting an artist re-roll the candidate order without
  touching the images. Leave it at `0` unless a solve fails on evidence that
  looks sufficient.

All widgets are appended in their final positional order. Combo values are append-only after release.

The node returns:

1. one combined `ATLAS_SOLVE`;
2. an artist-facing report;
3. structured registration diagnostics as JSON; and
4. match-overlay previews for the evaluated photo pairs.

## Scene Representation

Photo 1 becomes the solve's primary `camera` and `source_plate`. Photos 2 and 3 become `ProjectionSource` entries containing their recovered cameras, photographed plate references, and evidence metadata.

The current `ProjectionSource` schema is retained rather than adding a competing rig structure. Its documentation is broadened so it represents any extra projection camera, not only an AI novel view. Each source records at least:

- `evidence_type`: `photographed` or `generated`;
- the registration method and selected capture mode;
- reprojection error and confidence;
- scale provenance;
- source ordering and anchor identity.

Photographed sources must remain distinguishable from Qwen sources throughout viewport display, JSON export, health reporting, and DCC export.

## Registration Pipeline

### 1. Validation

Before feature extraction, validate camera body, lens, focal length, developed dimensions, orientation, and undistortion status. Incompatible metadata fails the set with a specific diagnostic. Missing trusted RAW metadata is an error for this version rather than permission to guess intrinsics.

### 2. Deterministic correspondences

Extract stable local features from every developed display plate. Sort features and descriptors by an explicit stable key. Match every secondary photo against photo 1; when photo 3 exists, also match photos 2 and 3 so the rig has a closure constraint.

Use mutual matching, ratio filtering, spatial distribution checks, and deterministic ordering. Moving people, foliage, reflections, and vehicles are handled as geometric outliers rather than segmented by a learned model.

### 3. Competing motion models

Evaluate three interpretations from the same ordered evidence:

- a calibrated essential-matrix model for translated capture;
- a planar-translated model — Faugeras homography decomposition — for
  translated capture of a near-planar scene (a street facade), where the
  essential matrix is degenerate. It requires the homography to fit well AND
  to be inconsistent with a pure rotation, its inliers to dominate the raw
  matches (>= 50%), and the decomposed pose to clear the same cheirality and
  parallax bars as the essential path. Its spatial-coverage bar is two grid
  cells below the profile minimum, because one plane legitimately
  concentrates its consensus — the inlier-dominance requirement is the
  compensating anti-contamination guard (added 2026-08-09, validated on the
  sh001 X-H2 street set); and
- a homography/rotation model for a shared optical centre.

In auto the essential model outranks planar-translated, which outranks
rotation-only. A rig may mix essential and planar pairs — both are translated
capture; each pair's `pose_source` is recorded in diagnostics.

Sampling uses a fixed schedule derived from the input fingerprint and exposed seed, with fixed iteration budgets and stable tie-breaking. It must not depend on ambient random state, GPU inference, or thread completion order.

In `auto`, choose translated only when parallax, inlier distribution, triangulation, and residual checks support observable translation. Choose rotation-only only when its independent quality checks pass. Ambiguous evidence fails rather than silently selecting a plausible model.

Forced modes evaluate only the requested interpretation and fail if its checks do not pass.

### 4. Joint refinement

For translated sets, triangulate shared tracks and jointly refine the secondary camera poses and sparse landmarks. Photo 1 and trusted RAW intrinsics remain locked. The optimiser uses a fixed initialization, fixed iteration limit, deterministic damping schedule, robust loss, and stable summation/order rules.

For three photographs, tracks and poses must close across all three pairings. A third photograph that cannot agree with the photo 1-2 rig fails specifically as an inconsistent third view.

For rotation-only sets, refine camera orientations with every camera position fixed to photo 1's optical centre. Do not invent translation.

### 5. Canonical world and metric scale

Canonicalize the rig into photo 1's Atlas world frame. Preserve Atlas's right-handed, Y-up convention and recovered-camera-facing-negative-Z convention at the core boundary.

Translated reconstruction initially has arbitrary scale. Three metric anchors
resolve it, in strict priority order (revised 2026-08-09 after the first real
captures showed textureless asphalt/grass rarely supports the ground fit):

1. **Measured baseline** (`baseline_m` widget, 0.0 = unset): the measured
   distance between photo 1 and photo 2 optical centres, applied directly to
   the recovered baseline. Needs no ground plane. When `camera_height_m` is
   also entered the rig is seated so photo 1 sits at that height above Y=0
   (flat-ground assumption, stated in the scale notes); otherwise photo 1
   stays at the vertical origin.
2. **Fitted ground plane + `camera_height_m`**: the original anchor —
   requires the plane-fit criteria tabled below.
3. **Learned metric depth prior** (`learned_scale_fallback` widget, opt-in,
   default off): the outdoor metric depth model predicts photo 1's depth map
   at the adapter boundary; core takes the median ratio between predicted and
   recovered landmark depths (>= 24 samples, median-absolute-deviation under
   50% of the median, else rejected). Recorded with a diagnostics warning
   naming its roughly 20-30% absolute-scale uncertainty. Learned depth still
   never determines relative pose.

If no anchor succeeds, report `scale_unavailable` naming every remedy and do
not emit a metric solve.

If the ground plane or height is unusable, report `scale unavailable` and do not emit a metric solve. Rotation-only sets have no translation scale to resolve.

### Fixed quality thresholds

These are the released numbers behind `match_quality` (append-only once
shipped; changing them is a behaviour change to every saved workflow):

| Profile | Ratio | Min inliers | Reprojection (px) | Min triangulation angle | Min grid cells | Max features |
|---|---|---|---|---|---|---|
| `conservative` | 0.70 | 64 | 1.0 | 1.5° | 8 | 8000 |
| `balanced` | 0.75 | 48 | 1.5 | 1.0° | 6 | 8000 |
| `permissive` | 0.80 | 32 | 2.5 | 0.5° | 4 | 10000 |
| `salvage` | 0.90 | 24 | 3.0 | 0.3° | 2 | 12000 |

`salvage` (appended 2026-08-09) is the explicit last resort for captures where
the stricter profiles detect too little calibration evidence — repetitive
texture strangled by the Lowe ratio, thin overlap. The geometric checks
remain the guards; the diagnostics, not the picture, carry the trust verdict.
Feature DETECTION additionally runs on a deterministically bounded copy of
each plate (long side ≤ 4000 px, keypoints scaled back to full-resolution
pixels): full-resolution SIFT on 40MP plates finds fine detail that does not
repeat between frames and measurably halves usable inliers.

All profiles additionally require a positive-depth fraction of at least 0.75
for a translated model. Three-view closure fails as `inconsistent_third_view`
beyond `0.5°` rotation, `1.5°` translation direction, or `2.0 px` median
closed-track reprojection under `balanced`; the limits scale linearly with the
selected profile's reprojection threshold relative to `1.5 px`. The ground
plane used for metric scale must have its normal within `20°` of the recovered
up direction, at least 24 inlier landmarks covering at least 15% of valid
landmarks, and a positive anchor-to-plane distance; its RANSAC tolerance is 5%
of the candidate spread (roughly 8 cm at street scale — real road crown and
triangulation noise, measured live 2026-08-09). Ground candidacy itself is a
world-space test — landmarks below the anchor camera in the recovered Y-up
frame — not an image-space horizon test. The metric anchor itself
is exact by construction: one uniform scale sets photo 1's height to the
entered `camera_height_m` with no fitted residual.

## Qwen Multiple-Angles Integration

`AtlasAddPatchView` remains downstream of `AtlasMultiViewSolve`.

- Qwen cameras are created from the declared Multiple-Angles LoRA view vocabulary and existing exact/named-view contracts.
- Qwen-generated pixels never enter feature matching, bundle refinement, ground fitting, or confidence for the photographed rig.
- Generated sources record `evidence_type=generated`; recovered RAW sources record `evidence_type=photographed`.
- A solve may contain both, and reports and viewport legends must make the distinction visible.

This preserves the useful deterministic camera relationship declared by the Qwen workflow without treating synthetic content as photogrammetric measurement.

## Determinism Contract

For identical ordered image content, RAW metadata, settings, Atlas version, and an unchanged dependency environment — the exact installed numpy and opencv-python builds, since OpenCV releases may legitimately alter feature extraction output — repeated runs must produce identical:

- feature and match selections;
- candidate sampling order;
- motion-model choice;
- accepted tracks and inliers;
- camera matrices and metric scale;
- confidence values and diagnostic fields; and
- serialized solve JSON.

The node's content fingerprint includes every image, metadata field, and widget that can alter execution. Changing input order changes the anchor by design and therefore changes the fingerprint.

## Outcomes and Failure Taxonomy

The node fails closed and names the actionable cause:

- `metadata_mismatch`: camera, lens, focal, dimensions, orientation, or undistortion conflict;
- `insufficient_overlap`: too few well-distributed shared features;
- `dynamic_scene_contamination`: consensus is dominated by inconsistent moving regions;
- `degenerate_geometry`: translated pose is unsupported by any model — a planar scene alone no longer lands here (the planar-translated model covers it), but a scene that defeats both decompositions still does;
- `scale_unavailable`: relative translated rig is valid but metric scale cannot be anchored;
- `inconsistent_third_view`: photo 3 does not close against the photo 1-2 rig;
- `ambiguous_motion_model`: neither translated nor rotation-only interpretation passes its checks.

`auto` may fall back from translated to rotation-only only when the rotation-only model passes its own strict checks. It never falls back to independent single-image solves.

`rotation_only` is a successful, explicitly degraded outcome rather than a
failure. Its solve records that all cameras share one optical centre and that
no translation geometry was recovered.

The core registration API always returns a structured outcome, including
diagnostics and overlays for failed attempts. The ComfyUI adapter returns all
four node outputs only for successful translated or rotation-only outcomes. On
any failure it raises an actionable error containing the outcome code and
summary; ComfyUI cannot return output links and raise in the same execution.
It must never emit a plausible but untrustworthy `ATLAS_SOLVE`.

Because a raise strands the node outputs, the adapter first writes the failed
run's structured diagnostics to `atlas_debug/multiview_failure.json` and its
pair overlays to `atlas_debug/multiview_failure_pair_N.png` under ComfyUI's
working directory (the AtlasDebugReport convention), and names that path in
the raised error. A debug-write failure never masks the registration error.

## Diagnostics and Confidence

The artist report identifies:

- the anchor photo and selected capture mode;
- metadata compatibility;
- feature, match, track, and inlier counts;
- spatial coverage of accepted evidence;
- per-pair and per-camera reprojection error;
- three-view closure error when applicable;
- fitted ground support and applied camera-height scale;
- scale provenance and confidence;
- photographed and generated source counts;
- warnings and the remedy for every degraded or failed state.

Structured diagnostics expose the same evidence without requiring consumers to parse prose. Pairwise overlays show accepted and rejected correspondences in stable colours and ordering.

## Testing

### Automated fixtures

Use compact synthetic image sets and generated feature correspondences with known cameras. Cover:

- two- and three-camera translated solves;
- rotation-only auto-detection and forced overrides;
- metric scaling from entered camera height and a fitted ground plane;
- three-view pose-graph closure;
- RAW metadata mismatches;
- insufficient overlap, planar degeneracy, and dynamic outliers;
- photographed sources followed by Qwen patch sources;
- solve JSON round trips, viewport payloads, and exporter compatibility;
- input permutation, proving that anchor changes are deliberate while relative geometry remains equivalent.

### Determinism gate

Run identical cases repeatedly and, where practical, in fresh processes. Require exact equality for selected evidence, model choice, matrices, reports, diagnostics, and serialized JSON. Tests must perturb ambient random state and thread settings to prove the solver does not inherit them.

### Fuji X-H2 acceptance set

Keep the full RAF files local because of their size. Retain a compact local manifest containing capture order, camera/lens/focal metadata, lens-centre height, approximate lateral spacing, and expected mode.

The acceptance run writes overlays and a registration report. It passes when:

- all expected cameras register in the correct mode;
- all photographed plates align in the Atlas viewport without obvious camera-to-camera sliding;
- reprojection and closure errors are under the fixed quality thresholds tabled above;
- the recorded scale provenance carries the entered anchor height exactly (the uniform scale is exact by construction);
- repeated runs are byte-equivalent under the determinism contract; and
- adding Qwen sources does not change any photographed camera value.

## Out of Scope for Version One

- More than three photographs.
- Mixed cameras, lenses, focal lengths, or inconsistent development settings.
- Non-static scene reconstruction.
- Metric scale inferred from learned monocular depth WITHOUT the artist
  opting in (the `learned_scale_fallback` widget is the explicit opt-in).
- Treating Qwen pixels as measured registration evidence.
- Dense multi-view stereo or texture seam optimisation.
- AUTOMATIC baseline detection; the `baseline_m` widget is a manual measured
  value in the same trust class as `camera_height_m`.
