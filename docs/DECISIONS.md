# Atlas Architecture Decisions

Status: locked for v0.1 compatibility. These decisions adapt the spec-panel
review to the current `atlas_camera` codebase. Code that violates one of these
contracts is a bug, not a style preference.

## 1. Canonical Coordinate Convention

Atlas core stores scene and camera transforms in right-handed, Y-up
coordinates. Image coordinates use origin top-left, x right, y down.

OpenCV/CV-native estimates are allowed inside recovery code, but conversion to
the Atlas core convention must happen at solver/import boundaries. Exporters own
the final conversion from Atlas core into target host conventions such as Maya,
Blender, USD, or Nuke.

## 2. Camera Math Lives In Core

Physical camera conversions live in `atlas_camera.core.camera_math` and are
tested once:

- millimeters to inches and back
- focal length to field of view and back
- sensor pixel offsets to normalized film offsets
- explicit focal fallback from FOV plus an assumed full-frame sensor

Exporters call these helpers and pass values to host-specific field names. They
do not keep local copies of conversion formulas.

## 3. Missing Focal Length Is Never Silent

When focal length cannot be directly recovered, Atlas may infer one from a field
of view estimate and an explicit full-frame sensor fallback. That value must be
marked as inferred, the focal confidence metric must be lowered, and notes or
warnings must say which assumption was used.

Atlas must not invent a focal length and present it as recovered fact.

## 4. Maya Names Are A Frozen Interface

The canonical Maya node names are:

| Node | Name |
|---|---|
| Camera transform/shape | `atlas_CAMERA` |
| Projection frustum group | `atlas_PROJECTION_GRP` |
| Recovered geometry group | `atlas_GEOMETRY_GRP` |
| Debug/diagnostic group | `atlas_DEBUG_GRP` |
| Reference image group | `atlas_REFERENCE_GRP` |

Downstream TD tools may string-match these names. Future renames require a
schema/version migration note.

## 5. Determinism

Atlas is deterministic under a fixed, surfaced seed. Vanishing-point detection
uses seeded RANSAC; callers can pass a seed, and solves record the seed used in
debug metadata. Unconditional bit-for-bit GPU determinism is not promised.

## 6. Confidence Metrics

Structured confidence uses a global score and per-parameter metrics. Scores are
relative heuristics, not calibrated probabilities.

`LatentCamera` metrics use this fixed key set:

```text
horizon, vp1, vp2, vp3, focal, extrinsics, sensor
```

The legacy `LatentScene.confidence` float remains as a compatibility mirror for
existing callers.

## 7. Recovered Object Surface

Recovered objects share confidence and serialization contracts. They do not all
share a forced generic `.value`; a camera is already a structured object. The
existing `LatentComponent.value` placeholders remain for depth, geometry,
lighting, and semantics until those become concrete recovered objects.

## 8. Schema Versioning

Serialized scene and recovered camera data include `schema_version`. Any future
breaking schema change requires an explicit version bump and migration note.

## 9. UI Viewport State Is Not Solver Evidence

The optional React workbench may store display preferences, camera-preview
overrides, and editable proxy objects under `constraints.viewport3d`. This state
is a local inspection aid for the artist workspace.

The deterministic solver must not treat `viewport3d` as camera evidence. If a
proxy or visual guide should influence solving, it must be promoted into an
explicit supported constraint such as `scale_constraints` or a future geometry
constraint with documented semantics.
