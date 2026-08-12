# Atlas Dynamic Plates v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Represent a dynamic region (water first) of a solved still as a `DynamicPlate` — crop-adjusted camera + matte + receiver plane + optional temporal frames — separable from the Atlas camera move.

**Architecture:** Three additive layers. (1) `core/camera_crop.py` + `core/dynamic_plate.py`: pure-math crop/resize intrinsics, plate schema, matte/ROI utilities, receiver-plane builder, validator — repo-conventional dataclasses, string constants, `_json_ready` serialization, numpy behind `_require_numpy`. (2) `atlas_camera/dynamic/`: temporal-generator abstraction + ComfyUI/LTX adapter over the existing stdlib `mcp.comfy_http` bridge (capability-gated, template-driven), plus `python -m atlas_camera.dynamic` CLI. (3) `exporters/dynamic_plate_package.py` + `exporters/dynamic_plate_blender.py`: artifact package on disk and a Blender script with projector camera ≠ artist camera and an image-SEQUENCE projected texture. No ComfyUI nodes in v0.1 (pin-test cascade; deferred).

**Tech Stack:** stdlib + optional numpy (`[vision]`), optional Pillow (`[image]`), stdlib urllib to ComfyUI for LTX. No new required deps.

## Global Constraints

- Core stays host-agnostic; nothing outside `comfy/` imports `comfy/` (importing `atlas_camera.mcp.comfy_http` from `dynamic/` is fine — stdlib-only module, not comfy).
- No Enum classes — module-level string constants (repo has zero Enums).
- Schema style: `@dataclass(slots=True)`, `to_dict()` = `_json_ready(self)`, classmethod `from_dict` tolerant of missing keys.
- Optional deps via `_require_numpy()`-style RuntimeError naming the pip extra; capability probes are network-free bools; LTX/Comfy absence must never break normal imports.
- DCC conversions only at adapter boundaries (`dcc_transform.blender_matrix_from_atlas`).
- Pixel convention: origin top-left, `u = cx + fx*x/w`, `v = cy - fy*y/w`, `w = -z_cam`; ray dirs cam-frame `[(u-cx)/fx, -(v-cy)/fy, -1]`.
- 4x4 `camera_view_matrix` is the world-math source; never build world math from the 3x3.
- Provenance strings lowercase: `"observed"`, `"derived_from_solve"`, `"generated"`, `"human_verified"`, `"inferred"`.
- Failure statuses lowercase snake: `region_invalid`, `camera_crop_failure`, `receiver_geometry_unavailable`, `generator_unavailable`, `generation_failure`, `frame_sequence_incomplete`, `projection_setup_failure`, `export_failure`.
- Image sequence is authoritative; MP4 preview optional; never claim scene-linear EXR; record color-space strings explicitly.
- Run `python -m pytest -q` before first change and after last.

---

### Task 1: Crop-adjusted camera intrinsics (`core/camera_crop.py`)

**Files:**
- Create: `atlas_camera/core/camera_crop.py`
- Test: `tests/test_camera_crop.py`

**Interfaces (produces):**
```python
@dataclass(slots=True)
class RegionROI:
    x: int; y: int; width: int; height: int
    def to_dict(self) -> dict; @classmethod def from_dict(cls, d) -> RegionROI | None
    def clamped(self, image_width: int, image_height: int) -> RegionROI
    def expanded(self, *, pad_px: int = 0, pad_frac: float = 0.0,
                 image_width: int, image_height: int) -> RegionROI  # pad_frac of max(w,h), then clamp

def crop_intrinsics(intrinsics: AtlasIntrinsics, roi: RegionROI) -> AtlasIntrinsics
    # deepcopy; image_width/height = roi.w/h; cx' = cx - roi.x; cy' = cy - roi.y
    # (cx/cy resolved via CameraSpec fallback ladder first: fx_px/cx_px -> principal_point -> centre;
    #  fx/fy need focal: resolve via CameraSpec.for_image); principal_point_px updated;
    # sensor_height_mm recomputed = sensor_width_mm * h/w (widen_intrinsics precedent)

def scale_intrinsics(intrinsics: AtlasIntrinsics, out_width: int, out_height: int) -> AtlasIntrinsics
    # sx = out_w/w, sy = out_h/h; fx*=sx, fy*=sy, cx*=sx, cy*=sy

@dataclass(slots=True)
class CropTransform:      # persisted, exactly invertible
    source_width: int; source_height: int
    roi: RegionROI
    output_width: int; output_height: int   # == roi.w/h when unscaled
    def full_to_crop(self, px: float, py: float) -> tuple[float, float]
    def crop_to_full(self, px: float, py: float) -> tuple[float, float]
    to_dict / from_dict
```

- [ ] Step 1: failing tests — full-frame crop identity; offset crop shifts principal point exactly (`cx' == cx - x`); crop+resize scales fx/cx by sx and fy/cy by sy; round-trip full→crop→resized→full returns original pixel to 1e-9; ROI clamp/expand (pad_px, pad_frac, clamped at borders); degenerate ROI (w<=0) raises `ValueError`.
- [ ] Step 2: run, verify fail. Step 3: implement. Step 4: pass. Step 5: commit `feat(core): crop-adjusted camera intrinsics + invertible CropTransform`.

### Task 2: DynamicPlate schema (`core/dynamic_plate.py`)

**Files:**
- Create: `atlas_camera/core/dynamic_plate.py`
- Test: `tests/test_dynamic_plate_schema.py`

**Interfaces (produces):**
```python
DYNAMIC_REGION_TYPES = ("water","cloud","smoke","fire","foliage","cloth","actor","generic")
# statuses
PLATE_STATUS_DRAFT="draft"; PLATE_STATUS_READY="ready"; PLATE_STATUS_GENERATED="generated"; PLATE_STATUS_FAILED="failed"
GENERATOR_NOT_AVAILABLE = "not_available"
FAILURE_* constants (Global Constraints list)

@dataclass(slots=True)
class ReceiverGeometry:
    kind: str = "plane"                    # v0.1: plane only
    primitive: AtlasProxyPrimitive | None  # plane transform + dims, Atlas Y-up world
    path: str | None = None                # exported OBJ, package-relative
    coordinate_system: str = "right_handed"; up_axis: str = "Y"
    provenance: str = "derived_from_solve"
    to_dict / from_dict

@dataclass(slots=True)
class DynamicPlate:
    plate_id: str; semantic_type: str      # from DYNAMIC_REGION_TYPES
    source_image: str; source_width: int; source_height: int
    matte_path: str | None
    matte_bbox: RegionROI | None           # tight bbox pre-overscan
    source_roi: RegionROI | None           # inference ROI (overscanned, clamped)
    crop_transform: CropTransform | None
    source_camera: LatentCamera | None; crop_camera: LatentCamera | None
    receiver: ReceiverGeometry | None
    frame_rate: float = 24.0; frame_start: int = 0; frame_end: int = 0
    generator: str = ""; generator_config: dict; prompt: str = ""; seed: int | None
    projection_mode: str = "camera_projection"
    matte_feather_px: float = 0.0
    color_metadata: dict                   # input_color_space / generator_* / atlas_working
    provenance: dict                       # {"source_region":"observed","crop_camera":"derived_from_solve",...}
    status: str = PLATE_STATUS_DRAFT
    warnings: list[str]; metadata: dict
    schema_version: ClassVar[str] = "0.1"
    to_dict / to_json / from_dict
```
Prompt preset: `WATER_PROMPT_DEFAULT` module constant (editable default, spec §18 wording, not magic).

- [ ] Step 1: failing tests — serialization round-trip via to_json/from_dict preserves ROI/cameras/receiver; unknown semantic_type raises; from_dict of {}/None tolerated for optional subobjects; provenance defaults present; generated frames never marked observed.
- [ ] Steps 2-5: fail → implement → pass → commit `feat(core): DynamicPlate schema`.

### Task 3: Matte utilities

**Files:**
- Modify: `atlas_camera/core/dynamic_plate.py` (same module; numpy behind `_require_numpy`)
- Test: `tests/test_dynamic_plate_matte.py`

**Interfaces (produces):**
```python
def matte_bbox(matte, *, threshold: float = 0.5) -> RegionROI | None   # HxW array (uint8/float); None if empty
def validate_matte_dimensions(matte_shape, image_width, image_height) -> None  # raises ValueError
def feather_matte(matte, radius_px: float)  # separable box blur x3 ≈ gaussian; pure numpy
def crop_image_region(image, roi: RegionROI)  # HxW[xC] slice
```

- [ ] Step 1: failing tests — bbox of synthetic blob exact; empty matte → None; dim mismatch raises; overscan expand clips at borders (uses RegionROI.expanded); feather preserves range [0,1] and widens support; crop slice shape == roi.
- [ ] Steps 2-5, commit `feat(core): dynamic-plate matte utilities`.

### Task 4: Receiver plane builder + registration math

**Files:**
- Modify: `atlas_camera/core/dynamic_plate.py`
- Test: `tests/test_dynamic_plate_receiver.py`

**Interfaces (produces):**
```python
def pixel_ray_world(camera: LatentCamera, px: float, py: float) -> tuple[origin3, dir3]
    # from camera_view_matrix inverse; dir_cam = [(u-cx)/fx, -(v-cy)/fy, -1] (CameraSpec.for_image ladder)
def build_receiver_plane(solve_or_camera, roi: RegionROI, *, plane_height: float = 0.0,
                         max_distance: float = 500.0, margin: float = 1.1) -> ReceiverGeometry
    # sample ROI edge/corner pixels, intersect with world plane y=plane_height
    # (polygon_planes.intersect_ray_with_plane); rays missing/behind → clamp at max_distance along ray
    # projected to the plane; extents+centroid → AtlasProxyPrimitive("plane",
    # transform=depth_geometry.plane_transform(u=[1,0,0], v=[0,0,-1], n=[0,1,0], c), dims=(ex,ez,0));
    # metadata={"role":"dynamic_plate_receiver","plane_height":...}
def write_plane_obj(receiver: ReceiverGeometry, path) -> Path   # single quad, Y-up, projective UVs left to DCC
```

- [ ] Step 1: failing tests — analytic camera (known height/pitch, via conftest `make_atlas_solve` or `look_at_view_matrix`): center-bottom pixel ray hits y=0 at expected distance (hand-computed); plane primitive dims/centroid enclose all sampled hits; **registration gate (release-blocking)**: for several pixels, full-image pixel → CropTransform.full_to_crop → crop-camera `pixel_ray_world` → plane intersection == full-camera ray plane intersection to 1e-6.
- [ ] Steps 2-5, commit `feat(core): dynamic-plate receiver plane + registration-verified crop rays`.

### Task 5: Validator + frame-sequence checks

**Files:**
- Modify: `atlas_camera/core/dynamic_plate.py`
- Test: `tests/test_dynamic_plate_validator.py`

**Interfaces (produces):**
```python
@dataclass(slots=True)
class PlateValidationIssue: severity: str; code: str; message: str   # severity fail|warn
def validate_dynamic_plate(plate, *, package_dir=None, matte_shape=None,
                           frame_paths=None) -> list[PlateValidationIssue]
    # checks (spec §30): source exists (if package_dir), matte dims match, ROI within image,
    # crop intrinsics finite/positive, receiver present, frame sequence complete+contiguous,
    # frame dims match metadata, fps>0, frame count matches range, status explicit,
    # color metadata present, crop registration self-consistent (spot-check via Task 4 math)
def frame_sequence_report(frame_paths, *, expected_count, expected_size=None) -> list[PlateValidationIssue]
```

- [ ] Step 1: failing tests — valid plate → []; each broken field yields its coded issue (`region_invalid`, `camera_crop_failure`, `receiver_geometry_unavailable`, `frame_sequence_incomplete`); missing middle frame detected; dim mismatch detected.
- [ ] Steps 2-5, commit `feat(core): DynamicPlate validator`.

### Task 6: Artifact package writer (`exporters/dynamic_plate_package.py`)

**Files:**
- Create: `atlas_camera/exporters/dynamic_plate_package.py`
- Test: `tests/test_dynamic_plate_package.py`

**Interfaces (produces):**
```python
def build_dynamic_plate_package(plate: DynamicPlate, output_dir, *, source_image_path,
                                matte=None, context_pad_frac=0.15) -> DynamicPlateResult
# Layout (spec §24): dynamic/<TYPE>_<id>/manifest.json, source/{crop.png,matte.png,context.png},
# camera/{source_camera.json,crop_camera.json}, geometry/receiver.obj, generated/, preview/
# Pillow behind _require_pil ([image] extra). manifest.json = plate.to_dict() + created_at
# + atlas_version + package paths. Manifest IS the artifact here — it must succeed
# (unlike atlas_project.json side-manifest). atlas_project.json side-write stays try/except.
@dataclass(slots=True)
class DynamicPlateResult: package_dir: Path; files: dict[str, Path]; warnings: list[str]
def load_dynamic_plate(package_dir) -> DynamicPlate
```

- [ ] Step 1: failing tests — package tree matches layout; manifest round-trips through `load_dynamic_plate`; crop.png dims == roi; matte written when supplied; result.files keys pinned.
- [ ] Steps 2-5, commit `feat(exporters): DynamicPlate artifact package`.

### Task 7: Temporal generator abstraction (`atlas_camera/dynamic/`)

**Files:**
- Create: `atlas_camera/dynamic/__init__.py` (light: re-export names below)
- Create: `atlas_camera/dynamic/generators.py`
- Test: `tests/test_dynamic_generators.py`

**Interfaces (produces):**
```python
@dataclass(slots=True)
class TemporalGenerationConfig:
    prompt: str = ""; seed: int | None = None; fps: float = 24.0; frame_count: int = 96
    width: int | None = None; height: int | None = None    # inference resize; None = crop size
    mode: str = "image_to_video"    # shipped mode v0.1 (doc §19); "video_to_video" reserved
    extra: dict = field(default_factory=dict)

@dataclass(slots=True)
class TemporalGenerationResult:
    status: str                      # "ok" | "not_available" | "failed"
    frame_paths: list[str]; width: int = 0; height: int = 0
    fps: float = 0.0; frame_count: int = 0
    generator: str = ""; model: str = ""; method: str = ""; seed: int | None = None
    source_roi: RegionROI | None = None; crop_camera: LatentCamera | None = None
    metadata: dict; warnings: list[str]
    to_dict / from_dict

class TemporalGenerator(Protocol):
    name: str
    def available(self) -> tuple[bool, str]: ...          # (ok, reason)
    def generate(self, plate: DynamicPlate, package_dir, config: TemporalGenerationConfig) -> TemporalGenerationResult: ...

def resolve_generator(name: str) -> TemporalGenerator     # "ltx" -> LTXComfyGenerator, "none" -> NullGenerator
class NullGenerator: available -> (False, "no generator selected"); generate -> not_available result
```

- [ ] Step 1: failing tests — resolve_generator("none") works with zero optional deps; unknown name raises ValueError listing choices; NullGenerator.generate returns status `not_available` with warning, never touches disk; `import atlas_camera` and `import atlas_camera.core.dynamic_plate` succeed with no torch/comfy present (subprocess `-c` import check, pattern from existing dependency-isolation tests).
- [ ] Steps 2-5, commit `feat(dynamic): temporal generator abstraction`.

### Task 8: LTX-via-ComfyUI adapter (`dynamic/ltx_comfy.py`)

**Files:**
- Create: `atlas_camera/dynamic/ltx_comfy.py`
- Test: `tests/test_dynamic_ltx_comfy.py` (fully offline, monkeypatched `http_json`)

**Interfaces (produces):**
```python
class LTXComfyGenerator:
    name = "ltx"
    def __init__(self, *, host=None, template_path=None, timeout=1800): ...
        # host: arg -> $COMFY_HOST -> 127.0.0.1:8188 ; template: arg -> $ATLAS_LTX_TEMPLATE
    def available(self) -> tuple[bool, str]
        # (a) template resolves; (b) GET /object_info reachable; (c) required LTX class_types
        #     from the template all present in object_info. Never raises.
    def generate(self, plate, package_dir, config) -> TemporalGenerationResult
```
Mechanics: reuse `atlas_camera.mcp.comfy_http` (`upload_image`, `ui_to_api`, `queue_and_wait`, `http_json`). Template is a ComfyUI workflow JSON (UI or API format; detect by "nodes" key). Overrides located by class_type scan: LoadImage→uploaded crop.png; positive CLIPTextEncode→prompt; sampler seed→seed; length/frames widget→frame_count; fps where present. Outputs: read `/history` image outputs, download each via `/view?filename=&subfolder=&type=` (stdlib urlretrieve in this module — comfy_http contract stays no-blob), write `generated/frame_%04d.png`. Missing template/host → `not_available` (never raises); Comfy error → `failed` with error text in warnings. Metadata records `camera_preservation: "unverified_i2v"` diagnostic (spec §20) + color note `generator_output_color_space: "sRGB"`.

- [ ] Step 1: failing offline tests — available() False+reason when template missing / host down (http_json raises) / class_type absent; generate() with fake template + monkeypatched http_json/upload/urlretrieve produces frames + `ok` result with correct fps/frame_count/seed; queue error → `failed`, warnings carry message.
- [ ] Steps 2-5, commit `feat(dynamic): LTX ComfyUI adapter (template-driven, capability-gated)`.

### Task 9: Blender export (`exporters/dynamic_plate_blender.py`)

**Files:**
- Create: `atlas_camera/exporters/dynamic_plate_blender.py`
- Test: `tests/test_dynamic_plate_blender.py`

**Interfaces (produces):**
```python
def write_dynamic_plate_blender_script(plate: DynamicPlate, package_dir, output_path) -> Path
```
Generated bpy script (string-template style of `blender_exporter.py`, reuse `blender_matrix_from_atlas`):
- receiver plane mesh from `ReceiverGeometry.primitive` (from_pydata quad, converted points);
- **projection camera** `atlas_projection_camera` from `crop_camera` (fixed, NOT scene.camera);
- **artist camera** `atlas_render_camera` duplicate of source camera, set as `scene.camera` (spec §29 split);
- projection material: same TexCoord→Camera node chain as blender_exporter.py:158-237 but bound to the projection-camera object (`ShaderNodeTexCoord.object = proj_cam`) with crop-camera scale/offset (`scale_u = fx'/w'... offset_u = cx'/w'`);
- `img.source='SEQUENCE'`; `image_user.frame_start/frame_duration/frame_offset/use_auto_refresh=True` from plate frame range; path = first generated frame (fallback: source/crop.png as still with warning comment);
- `scene.frame_start/end`, fps from plate.
- [ ] Step 1: failing tests — script text contains both camera objects, SEQUENCE source, frame_duration == frame count, projection cam transform from crop_camera via blender matrix, and `scene.camera = ` artist camera only; no-frames fallback path present.
- [ ] Steps 2-5, commit `feat(exporters): Blender dynamic-plate script (projector/artist camera split, image sequence)`.

### Task 10: CLI (`atlas_camera/dynamic/__main__.py` + `cli.py`)

**Files:**
- Create: `atlas_camera/dynamic/cli.py`, `atlas_camera/dynamic/__main__.py` (3-line delegate)
- Test: `tests/test_dynamic_plate_cli.py`

Interface (argparse style of tools/solve_image.py):
```text
python -m atlas_camera.dynamic create --image X --matte M --type water
    [--solve atlas_solve.json | solve inline via atlas.recover]
    --out DIR [--generator none|ltx] [--fps 24] [--frames 96] [--seed N]
    [--prompt "..."] [--overscan-frac 0.1] [--plane-height 0.0] [--feather-px 8]
    [--blender] [--host H] [--template T]
python -m atlas_camera.dynamic validate --package DIR
```
create: load/solve camera → matte bbox → overscan ROI → crop/scale intrinsics → receiver plane → package → optional generate → optional blender script → validate → print report + exit code. Generator unavailable is exit 0 with `generator status = not_available` printed (spec §32).

- [ ] Step 1: failing tests — run `main([...])` in-process with tmp image+matte+saved solve, generator none: package exists, manifest valid, exit 0; validate subcommand on that package exits 0; broken matte dims exits nonzero with `region_invalid` in output.
- [ ] Steps 2-5, commit `feat(dynamic): headless CLI (create/validate)`.

### Task 11: SAM3 mask assist (optional, thin)

**Files:**
- Modify: `atlas_camera/dynamic/cli.py` (flag `--auto-matte "ocean, sea water"`)
- Test: extend `tests/test_dynamic_plate_cli.py` (monkeypatch `sam3_concept_mask`)

`--auto-matte CONCEPTS` (mutually exclusive with `--matte`): call `inference.sam3_segmenter.sam3_concept_mask`; unavailability → actionable error telling user to pass `--matte` (artist path never blocked, spec §5).
- [ ] Steps: failing test (mocked mask feeds pipeline; unavailable → clean error) → implement → pass → commit `feat(dynamic): optional SAM3 auto-matte assist`.

### Task 12: Docs + release-gate run + report

**Files:**
- Create: `docs/DYNAMIC_PLATES.md` (workflow, shipped mode = image-to-video, ocean-plane limitation §10, provenance, CLI)
- Modify: `CLAUDE.md` (one routing line), `docs/DESIGN_RULES.md` (dynamic-plate rules: camera move never baked; sequence authoritative; plane receiver limitation)

- [ ] Full suite `python -m pytest -q` green.
- [ ] Real run (spec §34/§35/§36): pick a real repo/example still with water-like lower region (or user plate from V91 input if reachable; else synthetic-matte over example.png). Solve → matte → CLI create → inspect crop intrinsics numerically → registration spot-check printed → Blender script emitted → dolly documented (artist camera translated ~0.5–1 m in script; projection camera fixed by construction). LTX: probe live ComfyUI; if not reachable, package reports `not_available` — do NOT fabricate frames.
- [ ] Commit docs + gate artifacts note. Final report: shipped / reused / added / generator-dependent / unsupported / next extension.

## Self-Review notes

- Spec coverage: §3→T2, §4→T2, §5→T11+T10, §6→T1/T3, §7-8→T1, §9-10→T4, §11-12→T4+T9, §13→T9 (depth testing native in Blender; occlusion via existing static scene geometry — documented, not re-implemented), §14-15→T3/T1, §16→T7, §17-21→T8, §22-23→T2/T8 color_metadata, §24-25→T6, §26→T2 provenance, §27→T10, §28-29→T9 (decision: Blender, reported), §30-31→T5, §32→T7/T8 gating, §33→all tests, §34-36→T12, §37-38 scope respected (no nodes, no sim, no v2v).
- Type consistency: RegionROI/CropTransform defined T1, consumed T2-T10 by those names. TemporalGenerationResult consumed by CLI T10.
- Known risk: LTX template availability on this machine — adapter is template-driven so absence degrades to `not_available` honestly.
