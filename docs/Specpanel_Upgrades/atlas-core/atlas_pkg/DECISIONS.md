# Atlas — Architecture Decisions

Status: **LOCKED for v0.1**. These eight decisions were made by senior panel
review before any core code was written. Each is a contract: code that
violates one of these is a bug, not a style choice. Changing a decision
after v0.1 ships requires a version bump and a migration note, not a quiet
edit to this file.

---

## 1. Canonical coordinate convention

**Decision:** Core stores all camera math in **OpenCV / CV-native** convention:
right-handed, +Y down, camera looks down +Z into the scene. This is what
vanishing-point and horizon inference naturally produce, so it is the
convention with zero translation cost at the point of recovery.

**Contract:**
- `LatentCamera.world_matrix`, `.view_matrix`, `.projection_matrix` are
  *always* OpenCV-convention. This is documented on the class itself, not
  left to tribal knowledge.
- Handedness flips, axis remaps, and basis changes happen **only** inside
  exporters (`export/maya.py`, `export/blender.py`, etc.). Core never flips
  a sign for a downstream DCC's benefit.
- Every exporter states its target convention in a module docstring and the
  exact remap it applies, so the diff between "OpenCV" and "target" is
  inspectable in one place.

**Why:** if conversion logic is allowed to leak into individual exporters,
each one reinvents it slightly differently, they silently disagree, and the
bug (a mirrored Blender camera relative to Maya) doesn't surface until
someone diffs two DCCs by eye.

---

## 2. Unit conversions live in core, tested once

**Decision:** All physical-unit math — mm↔inches, px↔normalized film offset,
FOV↔focal length — lives in `core/camera_math.py`. No exporter performs its
own unit math; exporters call core functions and pass through DCC-specific
field names only.

**Known landmines this closes:**
- Maya's `horizontalFilmAperture` / `verticalFilmAperture` are in **inches**,
  not mm.
- Maya's `horizontalFilmOffset` / `verticalFilmOffset` are a **normalized
  fraction of film aperture**, not a pixel value.
- `focal_length_mm → camera.focalLength` is a straight pass-through *only*
  if both sides agree on what "mm" means for that sensor size — verified by
  the golden-camera round-trip test (§8 / `test_exports.py`).

**Why:** a conversion is either right or wrong; there is no reason to give
it more than one chance to be wrong.

---

## 3. Missing focal length: defined fallback, never silent invention

**Decision:** If focal length cannot be recovered directly (only FOV is
observable from vanishing points, and focal length requires an assumed
sensor size), Atlas:

1. Assumes a stated default sensor (`sensor_width_mm = 36.0`,
   full-frame equivalent) **only as a last resort**, and only to make the
   field exportable.
2. Marks `focal_length_mm` with `inferred = True`.
3. **Lowers the confidence score** for that parameter specifically
   (`confidence.individual_metrics['focal']`), it does not stay at whatever
   the FOV-only estimate produced.
4. Writes a human-readable line to `notes` explaining exactly what was
   assumed and why (see `LatentCamera.notes`).

**Contract:** Atlas never invents a focal length and presents it as
recovered fact. "Recover, Don't Invent" means an assumption is always
visible, never laundered into a confident-looking number.

---

## 4. Maya node names are a frozen interface

**Decision:** The two source documents disagreed on naming
(`atlas_CAMERA` / `atlas_PROJECTION_GRP` / `atlas_GEOMETRY_GRP` /
`atlas_DEBUG_GRP` / `atlas_REFERENCE_GRP` vs.
`atlas_latentCamera_CAM` / `atlas_projection_GRP` / `atlas_debug_GRP`).

Per the general conflict-resolution rule (§ below — vision doc owns naming
and positioning), the **canonical names are**:

| Node | Name |
|---|---|
| Camera transform/shape | `atlas_CAMERA` |
| Projection frustum group | `atlas_PROJECTION_GRP` |
| Recovered geometry group | `atlas_GEOMETRY_GRP` |
| Debug/diagnostic group | `atlas_DEBUG_GRP` |
| Reference image group | `atlas_REFERENCE_GRP` |

**Contract:** these strings are a stable public interface. TDs write
downstream tools that string-match them. Any future rename requires a
`schema_version` bump and a documented migration, never a silent edit.
The instruction-doc names above are recorded here as historical context
only — they were never shipped, so no alias/back-compat shim is needed for
them.

---

## 5. Determinism, reworded honestly

**Decision:** The vision document's "identical inputs → identical outputs"
is replaced with: **deterministic under a fixed, surfaced seed.**

**Why:** vanishing-point detection uses RANSAC (seeded randomness); depth or
other neural-recovered properties run on GPU kernels that are not
bit-reproducible without specific (and costly) determinism flags. True
unconditional bit-stability isn't free and isn't always worth the
performance cost.

**Contract:** every recovery call accepts and records a `seed`. Given the
same seed and the same inputs, Atlas is deterministic. The seed used is
always surfaced back to the artist (in `notes` or a dedicated field), never
hidden as an internal default.

---

## 6. `individual_metrics` schema, defined now

**Decision:** `ConfidenceModel` has exactly two parts:

```python
global_score: float                    # 0.0–1.0, overall recovery confidence
individual_metrics: dict[str, float]    # 0.0–1.0 each, fixed key set below
```

Fixed keys for `LatentCamera` (other `RecoveredObject` subclasses define
their own key sets, but every subclass must define its set explicitly — no
ad hoc keys invented per-module):

```
horizon, vp1, vp2, vp3, focal, extrinsics, sensor
```

**Calibration honesty:** these are **relative heuristics**, not calibrated
probabilities. A `focal` score of 0.82 means "more reliable than a 0.6
focal estimate in this same recovery," not "82% likely to be within X mm of
ground truth." This is stated once here and is not allowed to be implied
otherwise anywhere in UI copy or docs.

**Why now, not later:** a uniform schema is what lets a workspace render
confidence consistently across every recovered parameter. Inventing keys
per-module breaks that the first time two modules disagree on what
"extrinsics confidence" means.

---

## 7. `RecoveredObject` base class

**Decision:** Every recoverable thing (`LatentCamera` today; `LatentDepth`,
`LatentGeometry`, etc. later) shares exactly this surface:

```python
class RecoveredObject(ABC):
    confidence: ConfidenceModel
    schema_version: str

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "RecoveredObject": ...
```

**What it deliberately does NOT force:** a generic `.value` property. A
scalar depth sample's "value" is a clean concept; a camera's "value" *is*
the object itself, and forcing `.value` to return `self` is a tell that the
abstraction was stretched past where it's useful. The real shared surface —
the thing that actually generalizes — is confidence + serialization + export
dispatch.

**Export dispatch resolves the API conflict between the two docs:**
- Vision doc: `scene.export.maya()` (scene-level orchestration)
- Instruction doc: `camera.to_maya()` (object-level method)

Both are correct, at different layers. **Objects own `to_<format>()`.**
`Scene.export.maya()` is a thin orchestrator that walks scene components and
calls each one's own `to_maya()`, then assembles the result. This gets both
API shapes from one mechanism, and means a third-party module that adds a
new `RecoveredObject` subclass with its own `to_maya()` is picked up by
scene-level export automatically — no registry to keep in sync by hand.

---

## 8. `schema_version` on every serialized object

**Decision:** `to_dict()` / `to_json()` always include `schema_version`
(currently `"0.1.0"`), set on the class, not computed.

**Why:** adding it after the first camera has been saved to disk means every
existing file becomes unversioned legacy data. Adding it before the first
file is written costs one field and a comment.

---

## Scope ruling for v0.1

Ships **exactly**: `LatentCamera`, perspective/vanishing-point/horizon
inference, Maya export, JSON export. Everything else in either source
document (Blender, Nuke, USD, COLMAP, `LatentDepth`, `LatentGeometry`,
multi-image fusion, Gaussian splats) is roadmap, not backlog. Scaffolding
that those features will eventually need (the `RecoveredObject` base, the
export-dispatch pattern) is built now because it's cheap now; the features
themselves are not.

## Conflict-resolution rule

Where the vision document and the instruction document disagree: **the
vision document wins on naming, positioning, and API surface** (it is the
later, platform-level document). **The instruction document fills
field-level gaps** where the vision document is silent (e.g. specific dict
keys, specific test cases). Every individual resolution is recorded once,
here, rather than re-litigated in code comments scattered across the
package.
