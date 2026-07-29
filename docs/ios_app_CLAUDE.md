# TEMPLATE — copy this to the root of the `atlas-ios` repo as `CLAUDE.md`

It is stored here so it travels in the Atlas clone. It is NOT instructions for
the Atlas repo; it is written for the iOS project on the Mac.

---

# CLAUDE.md — Atlas iPhone app

## What this is

Atlas (a ComfyUI camera-solve and matte-painting-projection toolkit, Python,
separate repo) recovers a 3D camera from a single photograph and projects the
plate onto derived geometry. Anything the original camera could not see is a
hole, and today every hole is **invented** — inpainting, predicted hidden
geometry, edge-dilated smear.

This app replaces invention with photography. Atlas writes a shooting brief;
the user photographs the missing material — the paving, the brick, the render —
and sends it back. Not the specific occluded scene, which may be unreachable,
but the same KIND of surface at the same angle and light.

The author is a matte painter and photographer first. Guidance in this app
should read like a shot brief, not like telemetry.

## The contract is a JSON file, and it is the only coupling

`atlas_shoot.json`, specified in `SHOOT_PROJECT_FORMAT.md` (copy it into this
repo). `shoot_project.example.json` is a generated fixture — **develop against
it**. No Windows box, no ComfyUI, no network required to build or test.

- `version` is an integer. **Refuse a version you do not know rather than
  guessing.**
- Fields are added, never renamed or repurposed. Do not reorder or "tidy" them.
- The Atlas side has a test that regenerates the fixture and compares, so if it
  changes here without changing there, that is a bug in this repo.

## Three rules a client must get right

**1. `incidence_deg` is range-dependent.** 0 is square to the surface, 90 is
edge-on. The same floor is steep underfoot and almost edge-on at the horizon, so
there is no single "angle to the ground" — the value is computed at that
surface's own distance. A slab photographed square will not sit into a plate
that sees it at 85°.

Guide roughly to this angle. It need not be exact: if the capture records its
own ARKit pose, Atlas corrects the residual. A measured pose beats a matched one.

**2. `volumetric: true` means there is NO surface to photograph.** An alleyway,
a doorway, a recess — no plane was fitted, so no angle describes it, and
`incidence_deg` is a placeholder of `0.0`, not a measurement. **Branch on the
flag.** Treating it as "square on" is the most likely serious bug in this app.

**3. `lighting.measured` is `false`, and that is a statement.** Atlas cannot
measure sun direction or hardness from one plate, so rather than emit a
confident-looking azimuth that is really a guess, the project ships the plate
(`reference_plate.png`) to match by eye.

Do not read missing lighting fields as "lighting does not matter" — it is the
single thing most likely to make a patch read as a sticker. Prefer flat or
overcast capture: Atlas can relight from surface normals it already has, but it
cannot remove baked-in hard shadows.

If `measured` ever becomes `true`, fields will be ADDED alongside it. Switch on
the flag, not on field presence.

## Returning a capture

Package as a **one-frame `.r3d`** — a ZIP containing `metadata` (JSON),
`rgbd/0.jpg`, `rgbd/0.depth` (float32 metres, LZFSE or raw), optional
`rgbd/0.conf`.

This needs no new format work on the Atlas side: its importer requires only
`w`/`h` and a pose, so a single-frame capture is a first-class input.

### Two conventions that fail SILENTLY

Both produce well-formed, wrong geometry that nothing downstream can detect.
Both are already pinned by tests on the Atlas side; this app must match.

- **`K` is COLUMN-major**: `[fx, 0, 0, 0, fy, 0, cx, cy, 1]`. Written
  row-major, the principal point reads as 0 and lands in the image corner.
- **The pose quaternion is SCALAR-LAST**: `[qx, qy, qz, qw, tx, ty, tz]`.
  Swift's `simd_quatf` exposes `.real` (w) and `.imag` (xyz), so write
  `[q.imag.x, q.imag.y, q.imag.z, q.real, ...]`. Scalar-first yields a valid
  rotation describing a different rotation.

### ARKit's basis is already Atlas's

x-right / y-up / z-back, right-handed, metres. So:

- **No axis remap. None.**
- **Do not apply any −Z canonicalisation.** Atlas applies one to *inferred*
  solves because yaw is unobservable from a single still; here yaw is measured,
  and flipping it turns a measurement into fiction.

The app does not have to do anything to honour these — it must simply not
"helpfully" convert coordinates before writing them.

## Capture specifics that bite

- Enable `isDepthDataDeliveryEnabled` and
  `isCameraCalibrationDataDeliveryEnabled` on the **session configuration
  before** the photo settings, or depth silently comes back nil.
- Convert with `AVDepthData.converting(toDepthDataType:
  kCVPixelFormatType_DepthFloat32)` — **depth, not disparity**. Disparity is
  1/metres and would invert the scene while looking entirely plausible.
- **Rescale intrinsics.** `intrinsicMatrixReferenceDimensions` is the resolution
  the matrix was measured at and usually differs from the delivered photo. Skip
  this and the principal point is wrong, subtly, everywhere.
- Only LiDAR devices give usable `AVDepthData`. On others, degrade to a plain
  photo and **say so in the UI** — a capture without measured depth looks
  identical to one with it, and that difference is the entire point of the app.

## Build order — do not start with AR

1. Parse the project, list shots by `priority`, show `subject`, `guidance` and
   `warnings` verbatim.
2. Capture, package as one-frame `.r3d`.
3. Send back via the Tailscale share sheet.
4. Confirm Atlas ingests it.

That is a complete round trip with zero AR. Only once it works is the AR overlay
(for `volumetric` shots) worth building — and when it comes, overlay the hole's
**geometry**, not the plate's pixels: off-location the plate shows a building
that is not in front of the user. Show the plate crop alongside as the
appearance reference instead.

## House style

- Guidance strings from the project are written for a photographer. Surface them
  as-is; do not paraphrase into UI-speak.
- Prefer refusing with a clear message over guessing — an unknown `version`, a
  missing plane, a device without depth. A wrong capture costs a trip.
- No fabricated numbers in the UI. If Atlas did not measure it, do not display
  it as if it did.
