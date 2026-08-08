# Deterministic Multi-View RAW Solve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic two- or three-photo RAW camera solve that produces one anchored Atlas rig, automatically distinguishes translated from rotation-only capture, and keeps Qwen-generated views outside measured registration.

**Architecture:** Four focused core modules separate typed outcomes, feature evidence, calibrated geometry, and orchestration. A thin ComfyUI node converts tensors to those core contracts and assembles photographed `ProjectionSource` entries; existing Qwen patch nodes remain downstream and receive explicit generated-evidence provenance.

**Tech Stack:** Python 3.10+, NumPy 1.24+, OpenCV 4.8+ SIFT/BF matching, Atlas core schema, ComfyUI tensors, pytest 8.

## Global Constraints

- Run `python -m pytest -q` before the first implementation task and after the final task.
- `atlas_camera.core` keeps zero required runtime dependencies; NumPy and OpenCV imports in this feature must be guarded with informative `[vision]` installation errors.
- Do not add SciPy or a learned model to the multi-view registration path.
- Photo 1 is always the anchor; input reordering deliberately changes the world frame.
- Trusted RAW intrinsics remain fixed during pose refinement.
- Generated Qwen pixels never enter matching, motion fitting, triangulation, ground fitting, confidence, or scale.
- Coordinate conversion occurs only at adapter boundaries; core remains right-handed, Y-up, with the anchor facing world `-Z`.
- Existing registered node keys, display names, combo values, output slots, and widget positions are immutable.
- New `AtlasMultiViewSolve` widgets are ordered `capture_mode`, `camera_height_m`, `match_quality`, `seed`; later additions append only.
- `camera_height_m=0.0` means unset and makes a translated solve fail with `scale_unavailable`.
- The same ordered inputs, settings, Atlas version, NumPy version, and OpenCV version must serialize byte-identical solve JSON across repeated fresh-process runs.
- After code changes, run `graphify update .`.

## File Structure

- Create `atlas_camera/core/multiview_types.py`: immutable settings, frame/evidence records, diagnostics, outcome codes, and JSON conversion.
- Create `atlas_camera/core/multiview_features.py`: guarded image normalization, SIFT extraction, stable feature ordering, pair matching, fingerprints, and overlays.
- Create `atlas_camera/core/multiview_geometry.py`: deterministic sample schedules, essential/homography fitting, mode selection, tracks, triangulation, refinement, ground fitting, and frame canonicalization.
- Create `atlas_camera/core/multiview_solver.py`: validation, pair orchestration, primary VP anchor, scale application, `AtlasSolve`/`ProjectionSource` assembly, and failure outcomes.
- Create `atlas_camera/comfy/nodes_multiview.py`: ComfyUI tensor/RAW-link adapter and public node contract.
- Create focused tests: `tests/test_multiview_types.py`, `tests/test_multiview_features.py`, `tests/test_multiview_geometry.py`, `tests/test_multiview_solver.py`, and `tests/test_multiview_node.py`.
- Modify integration surfaces only where required: `schema.py`, `nodes.py`, `node_registry.py`, `viewport_payload.py`, `scene_health.py`, `nodes_geometry.py`, `atlas_blockout.js`, catalog/docs, and pinned tests.

---

### Task 1: Typed contracts, validation vocabulary, and deterministic fingerprints

**Files:**
- Create: `atlas_camera/core/multiview_types.py`
- Modify: `atlas_camera/raw/pipeline.py`
- Test: `tests/test_multiview_types.py`
- Modify: `tests/test_raw_integration.py`
- Modify: `tests/test_load_raw_node.py`

**Interfaces:**
- Produces: `MultiViewFrame`, `MultiViewSettings`, `FeatureSet`, `PairMatches`, `PairModelEvidence`, `RegistrationDiagnostics`, `RegistrationOutcome`, `QUALITY_PROFILES`, and `registration_fingerprint()`.
- Consumed by: every later task.

- [ ] **Step 1: Write failing contract tests**

```python
def test_settings_reject_unknown_values():
    with pytest.raises(ValueError, match="capture_mode"):
        MultiViewSettings(capture_mode="guess")
    with pytest.raises(ValueError, match="match_quality"):
        MultiViewSettings(match_quality="reckless")

def test_fingerprint_changes_with_order_and_seed_but_not_ambient_rng():
    a, b = _frames()
    first = registration_fingerprint([a, b], MultiViewSettings(seed=7))
    random.seed(999)
    np.random.seed(999)
    assert registration_fingerprint([a, b], MultiViewSettings(seed=7)) == first
    assert registration_fingerprint([b, a], MultiViewSettings(seed=7)) != first
    assert registration_fingerprint([a, b], MultiViewSettings(seed=8)) != first

def test_failed_outcome_serializes_without_a_solve():
    out = RegistrationOutcome.failed("insufficient_overlap", "12 matches")
    assert out.solve is None
    assert out.diagnostics.to_dict()["outcome_code"] == "insufficient_overlap"

def test_raw_import_result_preserves_exif_orientation():
    result = _raw_import_result(orientation=6)
    assert result.orientation == 6
```

- [ ] **Step 2: Run the focused tests and confirm the import failure**

Run: `python -m pytest tests/test_multiview_types.py -v`

Expected: collection fails because `atlas_camera.core.multiview_types` does not exist.

- [ ] **Step 3: Implement exact data contracts and profiles**

```python
CaptureMode = Literal["auto", "translated", "rotation_only"]
MatchQuality = Literal["balanced", "conservative", "permissive"]
OutcomeCode = Literal[
    "translated", "rotation_only", "metadata_mismatch",
    "insufficient_overlap", "dynamic_scene_contamination",
    "degenerate_geometry", "scale_unavailable",
    "inconsistent_third_view", "ambiguous_motion_model",
]

@dataclass(frozen=True)
class QualityProfile:
    ratio: float
    min_inliers: int
    reprojection_threshold_px: float
    min_triangulation_angle_deg: float
    min_grid_cells: int
    max_features: int

QUALITY_PROFILES = {
    "conservative": QualityProfile(0.70, 64, 1.0, 1.5, 8, 8000),
    "balanced": QualityProfile(0.75, 48, 1.5, 1.0, 6, 8000),
    "permissive": QualityProfile(0.80, 32, 2.5, 0.5, 4, 10000),
}

@dataclass(frozen=True)
class MultiViewSettings:
    capture_mode: CaptureMode = "auto"
    camera_height_m: float = 0.0
    match_quality: MatchQuality = "balanced"
    seed: int = 0

@dataclass(frozen=True)
class MultiViewFrame:
    image: Any
    raw_meta: Any
    plate_ref: AtlasPlateRef | None = None
    label: str = ""

@dataclass(frozen=True)
class FeatureSet:
    points_xy: Any
    descriptors: Any
    responses: Any
    stable_indices: Any

@dataclass(frozen=True)
class PairMatches:
    frame_a: int
    frame_b: int
    points_a: Any
    points_b: Any
    indices: Any
    distances: Any
    occupied_grid_cells: int

@dataclass(frozen=True)
class PairModelEvidence:
    frame_a: int
    frame_b: int
    essential_matrix: Any | None
    homography: Any | None
    relative_rotation: Any | None
    translation_direction: Any | None
    essential_inliers: Any
    homography_inliers: Any
    essential_inlier_count: int
    homography_inlier_count: int
    median_essential_error_px: float
    median_homography_error_px: float
    median_triangulation_angle_deg: float
    positive_depth_fraction: float

@dataclass
class RegistrationDiagnostics:
    outcome_code: OutcomeCode
    summary: str
    selected_mode: str | None = None
    metadata_checks: list[dict[str, Any]] = field(default_factory=list)
    pair_metrics: list[dict[str, Any]] = field(default_factory=list)
    camera_metrics: list[dict[str, Any]] = field(default_factory=list)
    scale: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

@dataclass
class RegistrationOutcome:
    solve: AtlasSolve | None
    diagnostics: RegistrationDiagnostics
    overlays: tuple[Any, ...] = ()

    @classmethod
    def failed(cls, code: OutcomeCode, summary: str) -> "RegistrationOutcome":
        return cls(None, RegistrationDiagnostics(code, summary))
```

Define all array-bearing fields as `Any` so importing the module does not import NumPy. `registration_fingerprint()` hashes contiguous image bytes, shape/dtype, the ordered RAW fields used by validation, plate source path, Atlas settings, and seed with SHA-256. `RegistrationDiagnostics.to_dict()` must use `_json_ready`-compatible builtins only.

Append `orientation: int | None = None` to `RawImportResult` and populate it from `RawMetadata.orientation` in `import_raw()`. This is an additive Python contract and gives validation the EXIF orientation that the approved capture contract requires. Extend the existing RAW fixture factories with a default of `1`.

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/test_multiview_types.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the contracts**

```powershell
git add atlas_camera/core/multiview_types.py atlas_camera/raw/pipeline.py tests/test_multiview_types.py tests/test_raw_integration.py tests/test_load_raw_node.py
git commit -m "feat(multiview): define deterministic registration contracts"
```

---

### Task 2: Stable SIFT evidence, pair matching, and overlays

**Files:**
- Create: `atlas_camera/core/multiview_features.py`
- Test: `tests/test_multiview_features.py`

**Interfaces:**
- Consumes: `FeatureSet`, `PairMatches`, `QualityProfile`.
- Produces: `extract_features(image, profile) -> FeatureSet`, `match_features(a, b, profile, frame_a, frame_b) -> PairMatches`, and `render_match_overlay(image_a, image_b, matches, inlier_mask=None) -> ndarray`.

- [ ] **Step 1: Write failing deterministic feature tests**

```python
def test_feature_and_match_order_is_exactly_repeatable():
    left, right = _shifted_checkerboard()
    profile = QUALITY_PROFILES["balanced"]
    a1 = extract_features(left, profile)
    b1 = extract_features(right, profile)
    m1 = match_features(a1, b1, profile, 0, 1)
    random.seed(41); np.random.seed(41); cv2.setRNGSeed(41)
    a2 = extract_features(left, profile)
    b2 = extract_features(right, profile)
    m2 = match_features(a2, b2, profile, 0, 1)
    np.testing.assert_array_equal(a1.points_xy, a2.points_xy)
    np.testing.assert_array_equal(m1.indices, m2.indices)
    np.testing.assert_array_equal(m1.distances, m2.distances)

def test_matches_are_mutual_spatially_distributed_and_overlay_is_stable():
    # Assert unique query/train indices, occupied 4x4 cells, expected shape,
    # and identical SHA-256 for two rendered overlays.
```

- [ ] **Step 2: Run the test and verify missing functions**

Run: `python -m pytest tests/test_multiview_features.py -v`

Expected: import failure for `extract_features`.

- [ ] **Step 3: Implement guarded image normalization and feature extraction**

Use `_require_numpy()` and `_require_cv2()` helpers whose messages name `pip install -e .[vision]`. Convert float RGB to clipped, rounded `uint8` grayscale explicitly. Run `cv2.SIFT_create(nfeatures=profile.max_features)`, then sort keypoints by:

```python
order = sorted(range(len(keypoints)), key=lambda i: (
    round(keypoints[i].pt[1], 6), round(keypoints[i].pt[0], 6),
    -round(keypoints[i].response, 9), keypoints[i].octave,
    round(keypoints[i].size, 6), round(keypoints[i].angle, 6), i,
))
```

Reorder descriptors with the same indices and persist the stable feature index.

- [ ] **Step 4: Implement mutual ratio matching and stable overlays**

Create `matcher = cv2.BFMatcher(cv2.NORM_L2)` and run `matcher.knnMatch(desc_a, desc_b, k=2)` plus `matcher.knnMatch(desc_b, desc_a, k=2)`. Apply the selected ratio, keep only mutual pairs, sort by `(query_index, train_index, distance)`, and reject duplicate indices. Compute occupied 4x4 grid cells for diagnostics. Render accepted matches green and rejected matches red using a fixed palette and integer-rounded coordinates.

- [ ] **Step 5: Run feature tests twice**

Run: `python -m pytest tests/test_multiview_features.py -v && python -m pytest tests/test_multiview_features.py -v`

Expected: both runs pass with identical digest assertions.

- [ ] **Step 6: Commit feature evidence**

```powershell
git add atlas_camera/core/multiview_features.py tests/test_multiview_features.py
git commit -m "feat(multiview): add stable image correspondences"
```

---

### Task 3: Deterministic essential/homography fitting and mode selection

**Files:**
- Create: `atlas_camera/core/multiview_geometry.py`
- Test: `tests/test_multiview_geometry.py`

**Interfaces:**
- Consumes: `PairMatches`, trusted `AtlasIntrinsics`, `MultiViewSettings`, registration fingerprint.
- Produces: `fit_pair_models(matches, intr_a, intr_b, settings, fingerprint) -> PairModelEvidence` and `select_capture_mode(evidence, requested_mode, profile) -> Literal["translated", "rotation_only"]`.

- [ ] **Step 1: Write synthetic calibrated-pair tests**

```python
def test_translated_pair_selects_essential_model_exactly():
    matches, intr_a, intr_b = _project_known_scene(
        rotation_y_deg=7.0, translation=(0.8, 0.0, 0.1), outliers=30)
    evidence = fit_pair_models(matches, intr_a, intr_b,
                               MultiViewSettings(), "ab" * 32)
    assert select_capture_mode(evidence, "auto", QUALITY_PROFILES["balanced"]) == "translated"
    assert evidence.essential_inlier_count >= 80
    assert evidence.median_triangulation_angle_deg >= 1.0

def test_shared_centre_pair_selects_rotation_only():
    matches, intr_a, intr_b = _project_known_scene(
        rotation_y_deg=24.0, translation=(0.0, 0.0, 0.0))
    evidence = fit_pair_models(matches, intr_a, intr_b,
                               MultiViewSettings(), "cd" * 32)
    assert select_capture_mode(evidence, "auto", QUALITY_PROFILES["balanced"]) == "rotation_only"

def test_forced_translated_rejects_rotation_only_evidence():
    profile = QUALITY_PROFILES["balanced"]
    with pytest.raises(MotionModelError, match="degenerate_geometry"):
        select_capture_mode(evidence, "translated", profile)
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `python -m pytest tests/test_multiview_geometry.py -v`

Expected: missing geometry functions.

- [ ] **Step 3: Implement deterministic sample schedules and normalized solvers**

Derive a local `numpy.random.Generator(PCG64)` seed from `SHA256(fingerprint + settings.seed + model_name)`. Precompute all sample index arrays before scoring. Use 2,048 eight-point essential samples and 1,024 four-point homography samples, capped by the available unique combinations.

Implement Hartley normalization, eight-point essential fitting, SVD enforcement to singular values `(s, s, 0)`, four-point normalized DLT homography, Sampson essential error, and symmetric homography transfer error. Stable winner ordering is:

```python
score = (-inlier_count, median_error, tuple(sample_indices), model_bytes)
```

Decompose the winning essential matrix into four poses, triangulate its inliers, and select by maximum positive-depth count, then median reprojection error, then candidate index.

Define `MotionModelError(ValueError)` with an `outcome_code` attribute so lower-level geometric failures map to the exact public failure vocabulary without parsing exception text.

- [ ] **Step 4: Implement explicit auto/forced classification**

Translated passes when it meets the quality profile's minimum inliers, spatial cells, reprojection threshold, positive-depth fraction `>=0.75`, and minimum median triangulation angle. Rotation-only passes when its homography rotation residual is within the profile threshold and its inlier support is at least the profile minimum. If both pass, translated wins only when its triangulation angle passes; otherwise rotation-only wins. If neither passes, raise `ambiguous_motion_model` with both score summaries.

- [ ] **Step 5: Run geometry tests including random-state perturbation**

Run: `python -m pytest tests/test_multiview_geometry.py -v`

Expected: all model and mode-selection tests pass.

- [ ] **Step 6: Commit pair geometry**

```powershell
git add atlas_camera/core/multiview_geometry.py tests/test_multiview_geometry.py
git commit -m "feat(multiview): fit deterministic camera motion models"
```

---

### Task 4: Three-view tracks, triangulation, closure, and fixed-order refinement

**Files:**
- Modify: `atlas_camera/core/multiview_geometry.py`
- Modify: `tests/test_multiview_geometry.py`

**Interfaces:**
- Produces: `build_tracks(pair_matches, n_frames) -> tuple[FeatureTrack, ...]`, `initialise_rig(pair_evidence, mode) -> CameraRig`, `refine_rig(rig, tracks, intrinsics, mode) -> RefinedRig`, and `measure_three_view_closure(refined) -> ClosureMetrics`.
- Consumed by: `multiview_solver.solve_multiview()`.

- [ ] **Step 1: Add failing two/three-view refinement tests**

```python
def test_three_view_tracks_close_and_refinement_reduces_error():
    fixture = _three_camera_fixture(noise_px=0.35, outliers=45)
    tracks = build_tracks(fixture.pairs, n_frames=3)
    initial = initialise_rig(fixture.evidence, "translated")
    refined = refine_rig(initial, tracks, fixture.intrinsics, "translated")
    assert refined.reprojection_rmse_px < initial.reprojection_rmse_px
    assert refined.closure.rotation_error_deg < 0.15
    assert refined.closure.translation_direction_error_deg < 0.5

def test_inconsistent_third_view_has_a_distinct_error():
    fixture = _three_camera_fixture(scramble_pair_1_2=True)
    with pytest.raises(MotionModelError, match="inconsistent_third_view"):
        refine_rig(fixture.initial_rig, fixture.tracks,
                   fixture.intrinsics, "translated")
```

- [ ] **Step 2: Run the new tests and verify missing APIs**

Run: `python -m pytest tests/test_multiview_geometry.py -k "three_view or inconsistent" -v`

Expected: missing track/refinement functions.

- [ ] **Step 3: Implement stable tracks and triangulation**

Join pair matches by `(frame_index, stable_feature_index)`. Reject tracks that contain two features from one frame or fail pairwise closure. Sort tracks by the complete tuple of observations. Triangulate with normalized linear DLT, normalize homogeneous coordinates, require positive depth in every observing camera, and calculate reprojection errors in original pixels.

Define the geometry records in this module:

```python
@dataclass(frozen=True)
class FeatureObservation:
    frame_index: int
    feature_index: int
    point_xy: tuple[float, float]

@dataclass(frozen=True)
class FeatureTrack:
    track_id: int
    observations: tuple[FeatureObservation, ...]

@dataclass(frozen=True)
class CameraRig:
    rotations: tuple[Any, ...]
    translations: tuple[Any, ...]
    landmarks: Any
    reprojection_rmse_px: float

@dataclass(frozen=True)
class ClosureMetrics:
    rotation_error_deg: float
    translation_direction_error_deg: float
    median_reprojection_px: float

@dataclass(frozen=True)
class RefinedRig(CameraRig):
    accepted_track_ids: tuple[int, ...]
    closure: ClosureMetrics
```

- [ ] **Step 4: Implement deterministic alternating refinement without SciPy**

Use eight fixed outer iterations:

1. retriangulate every accepted landmark in stable track order;
2. update each secondary camera in index order with a six-parameter SE(3) pose-only Gauss-Newton step;
3. use a Huber loss with delta equal to the selected profile's reprojection threshold;
4. build each 6x6 normal equation in float64 and stable observation order;
5. use damping schedule `(1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 1e-6)`;
6. accept a step only when total robust error decreases.

Photo 1's pose and every intrinsic remain fixed. Rotation-only refinement exposes only three rotation parameters and pins all translations to zero.

- [ ] **Step 5: Add and pass closure thresholds**

For three-view translated rigs, fail with `inconsistent_third_view` when closure exceeds `0.5°` rotation, `1.5°` translation direction, or `2.0 px` median closed-track reprojection under the balanced profile. Scale these limits by the quality profile's reprojection threshold relative to `1.5 px`.

Run: `python -m pytest tests/test_multiview_geometry.py -v`

Expected: all pair, track, and refinement tests pass.

- [ ] **Step 6: Commit joint refinement**

```powershell
git add atlas_camera/core/multiview_geometry.py tests/test_multiview_geometry.py
git commit -m "feat(multiview): refine two and three camera rigs"
```

---

### Task 5: Anchor orientation, ground scale, solve assembly, and failure outcomes

**Files:**
- Create: `atlas_camera/core/multiview_solver.py`
- Modify: `atlas_camera/core/schema.py`
- Create: `tests/test_multiview_solver.py`

**Interfaces:**
- Produces: `validate_multiview_frames(frames, settings)`, `solve_multiview(frames, settings) -> RegistrationOutcome`.
- `solve_multiview` is the only public orchestration entry point used by ComfyUI.

- [ ] **Step 1: Write failing orchestration and validation tests**

```python
def test_metadata_mismatch_fails_before_feature_extraction(monkeypatch):
    frames = [_frame(focal=23.0), _frame(focal=35.0)]
    monkeypatch.setattr(features, "extract_features",
                        lambda *_: pytest.fail("must not extract"))
    out = solve_multiview(frames, MultiViewSettings())
    assert out.solve is None
    assert out.diagnostics.outcome_code == "metadata_mismatch"

def test_translated_rig_is_scaled_to_measured_anchor_height():
    out = solve_multiview(_translated_frames(),
                          MultiViewSettings(camera_height_m=1.43))
    assert out.diagnostics.outcome_code == "translated"
    assert out.solve.camera.extrinsics.camera_position[1] == pytest.approx(1.43)
    assert out.solve.debug_metadata["scale_source"] == "measured_camera_height"
    assert all(s.metadata["evidence_type"] == "photographed"
               for s in out.solve.projection_sources)

def test_qwen_pixels_are_not_an_input_to_the_solver_signature():
    assert list(inspect.signature(solve_multiview).parameters) == ["frames", "settings"]

def test_clustered_consensus_reports_dynamic_scene_contamination():
    out = solve_multiview(_frames_with_many_matches_on_one_moving_car(),
                          MultiViewSettings(camera_height_m=1.43))
    assert out.solve is None
    assert out.diagnostics.outcome_code == "dynamic_scene_contamination"
```

- [ ] **Step 2: Run the solver tests and confirm failure**

Run: `python -m pytest tests/test_multiview_solver.py -v`

Expected: missing `multiview_solver`.

- [ ] **Step 3: Implement strict RAW validation**

Require two or three frames. Compare normalized camera make/model, lens model, focal length within `0.05 mm`, sensor dimensions within `0.05 mm`, image orientation, developed dimensions, and `undistort_status`. Collect every mismatch into `RegistrationDiagnostics.metadata_checks`; return `metadata_mismatch` without extracting features. Require non-null, positive focal and sensor width on every frame.

- [ ] **Step 4: Implement anchor orientation and rig composition**

Run the existing classical vanishing-point detector on photo 1 with `random_seed=settings.seed` and trusted RAW intrinsics. Require two valid orthogonal VPs and use them for the anchor's pitch/roll. If they are absent, return `degenerate_geometry` with a remedy that asks for clearer architectural lines or artist constraints; sparse correspondence alone cannot distinguish a ground plane from a façade, so neither translated nor rotation-only mode may guess world-up. Compose every relative camera pose into photo 1's world frame, then apply `_face_camera_toward_negative_z` once to the whole rig so relative transforms remain unchanged.

- [ ] **Step 5: Implement deterministic ground fitting and metric scale**

Select triangulated landmarks below the anchor horizon and run a fixed-schedule three-point plane fit. Require a plane normal within `20°` of the recovered up direction, at least 24 inliers, at least 15% of valid landmarks, and positive anchor-to-plane distance. Orient the normal toward `+Y`, rotate the complete rig and landmarks so the plane is horizontal, and apply:

```python
scale = settings.camera_height_m / anchor_plane_distance
world_point = scale * rotated_point - (0.0, scaled_plane_y, 0.0)
```

Set photo 1's Y to exactly `camera_height_m`. A translated solve with height `<=0` or no valid ground plane returns `scale_unavailable`. Learned depth is never called.

Before model fitting, return `insufficient_overlap` when any required pair has fewer than the profile's minimum mutual matches. When raw matches are at least twice that minimum but every geometric consensus occupies fewer than the profile's minimum 4x4 grid cells, return `dynamic_scene_contamination` and report the consensus bounding box; this distinguishes a moving-object-dominated cluster from a genuinely match-poor pair.

- [ ] **Step 6: Assemble one Atlas solve and broaden schema documentation**

Build the primary `AtlasCamera`, `source_plate`, horizon, confidence, and debug metadata. Add one `ProjectionSource` for each secondary photographed frame with its preview, `plate_ref`, recovered camera, empty `proxy_geometry`, and metadata keys from the spec. Change only the `ProjectionSource` docstring in `schema.py`; serialization already preserves metadata.

- [ ] **Step 7: Prove deterministic serialization and failure outcomes**

Run: `python -m pytest tests/test_multiview_solver.py -v`

Expected: successful translated/rotation-only cases pass, every failure returns a structured outcome, and two runs produce identical `solve.to_json()`.

- [ ] **Step 8: Commit the core solver**

```powershell
git add atlas_camera/core/multiview_solver.py atlas_camera/core/schema.py tests/test_multiview_solver.py
git commit -m "feat(multiview): assemble anchored photographed camera rigs"
```

---

### Task 6: ComfyUI node, registry, façade, and cache contract

**Files:**
- Create: `atlas_camera/comfy/nodes_multiview.py`
- Modify: `atlas_camera/comfy/node_registry.py`
- Modify: `atlas_camera/comfy/nodes.py`
- Create: `tests/test_multiview_node.py`
- Modify: `tests/test_comfy_node_registry.py`
- Modify: `tests/test_facade_surface.py`

**Interfaces:**
- Produces registered node key `AtlasMultiViewSolve`, display name `Atlas Multi-View RAW Solve 📷📷`, outputs `("ATLAS_SOLVE", "STRING", "STRING", "IMAGE")`, and output names `("solve", "report", "registration_json", "match_overlays")`.

- [ ] **Step 1: Write failing public-contract tests**

```python
def test_node_contract_and_widget_order():
    cls = NODE_CLASS_MAPPINGS["AtlasMultiViewSolve"]
    assert NODE_DISPLAY_NAME_MAPPINGS["AtlasMultiViewSolve"] == "Atlas Multi-View RAW Solve 📷📷"
    assert cls.RETURN_NAMES == ("solve", "report", "registration_json", "match_overlays")
    spec = cls.INPUT_TYPES()
    assert list(spec["required"]) == ["image_1", "image_2"]
    widgets = [k for k, v in spec["optional"].items()
               if not v[1].get("forceInput", False)]
    assert widgets == ["capture_mode", "camera_height_m", "match_quality", "seed"]

def test_node_fingerprint_changes_when_any_link_or_widget_changes():
    # Assert image_1, image_2, image_3, raw_meta links, plate refs, order,
    # mode, height, quality, and seed each alter IS_CHANGED.
```

- [ ] **Step 2: Run node and registry tests to verify failure**

Run: `python -m pytest tests/test_multiview_node.py tests/test_comfy_node_registry.py tests/test_facade_surface.py -v`

Expected: new key/class absent.

- [ ] **Step 3: Implement the thin adapter**

`AtlasMultiViewSolve.solve()` converts each `IMAGE` batch element to contiguous HWC float32 NumPy, builds ordered `MultiViewFrame` records, calls `solve_multiview`, converts overlay arrays to one torch `IMAGE` batch, and serializes diagnostics with `sort_keys=True`. It does no geometry itself.

Use required image links for photos 1 and 2. Put `image_3`, `raw_meta_1..3`, and `plate_ref_1..3` in `optional` with `forceInput=True`, followed by the four widgets in the exact approved order. Validate that image 3 cannot be supplied without its RAW metadata.

When the outcome has no solve, raise:

```python
raise RuntimeError(
    f"AtlasMultiViewSolve [{code}]: {summary}\n"
    f"registration diagnostics: {json.dumps(details, sort_keys=True)}"
)
```

- [ ] **Step 4: Register and re-export the node**

Add the new import/module entry and display name without reordering existing entries. Add `AtlasMultiViewSolve` to the exact normal-key and façade sets; increment the standard count from 90 to 91. Do not change experimental, legacy, or iOS gates.

- [ ] **Step 5: Run focused public-contract tests**

Run: `python -m pytest tests/test_multiview_node.py tests/test_comfy_node_registry.py tests/test_facade_surface.py -v`

Expected: all pass.

- [ ] **Step 6: Commit the node surface**

```powershell
git add atlas_camera/comfy/nodes_multiview.py atlas_camera/comfy/node_registry.py atlas_camera/comfy/nodes.py tests/test_multiview_node.py tests/test_comfy_node_registry.py tests/test_facade_surface.py
git commit -m "feat(comfy): expose deterministic multi-view RAW solve"
```

---

### Task 7: Photographed/generated provenance in Qwen, viewport, health, and exports

**Files:**
- Modify: `atlas_camera/comfy/nodes_geometry.py`
- Modify: `atlas_camera/comfy/viewport_payload.py`
- Modify: `atlas_camera/comfy/web/atlas_blockout.js`
- Modify: `atlas_camera/core/scene_health.py`
- Modify: `atlas_camera/exporters/_layers.py`
- Modify: `tests/test_add_patch_view.py`
- Create: `tests/test_multiview_provenance.py`
- Modify: `tests/test_scene_health_gate.py`
- Modify: `tests/test_nuke_layers_export.py`

**Interfaces:**
- Adds `evidence_type` to serialized projection-source payloads and reports.
- Keeps exporter file formats unchanged; provenance rides existing metadata/manifests.
- Photographed sources with no private mesh reuse the primary scene's derived geometry; generated sources never receive that fallback.

- [ ] **Step 1: Add failing provenance tests**

```python
def test_qwen_patch_is_explicitly_generated():
    out = _add_patch(_base_solve(), _patch_image())
    assert out.projection_sources[-1].metadata["evidence_type"] == "generated"

def test_viewport_and_health_distinguish_evidence_types():
    solve = _mixed_photographed_and_generated_solve()
    sources = _serialize_projection_sources(solve)
    assert [s["evidence_type"] for s in sources] == [
        "photographed", "generated"]
    health = evaluate_scene_health(solve).to_dict()
    assert health["projection_evidence_counts"] == {
        "photographed": 1, "generated": 1, "unknown": 0}

def test_photographed_source_reuses_primary_geometry_but_generated_does_not():
    solve = _mixed_solve_with_primary_geometry_and_empty_source_geometry()
    photo_geometry, photo_origin = _projection_geometry_for_source(
        solve, solve.projection_sources[0])
    generated_geometry, generated_origin = _projection_geometry_for_source(
        solve, solve.projection_sources[1])
    assert photo_origin == "primary_scene"
    assert len(photo_geometry) > 0
    assert generated_origin == "source"
    assert generated_geometry == []
```

- [ ] **Step 2: Run provenance tests and verify failure**

Run: `python -m pytest tests/test_add_patch_view.py tests/test_multiview_provenance.py tests/test_scene_health_gate.py tests/test_nuke_layers_export.py -k "evidence or generated or photographed" -v`

Expected: missing `evidence_type` fields/assertions fail.

- [ ] **Step 3: Stamp and transport provenance**

Add `"evidence_type": "generated"` in `AtlasAddPatchView._finish_patch`. Preserve `photographed` from the multi-view solver. In `viewport_payload._serialize_projection_sources`, emit `evidence_type = metadata.get("evidence_type", "unknown")`. Add `projection_evidence_counts: dict[str, int]` to `HealthReport`, include it in `to_dict()`, and populate photographed/generated/unknown counts in `evaluate_scene_health` without converting unknown legacy layers into either trusted class.

When both photographed and generated counts are non-zero, add a `mixed_projection_evidence` warning that says generated cameras did not influence the photographed registration. Do not lower photographed camera confidence because a generated layer is present.

- [ ] **Step 4: Make the viewport distinction visible**

In the existing projection-source layer legend, append `PHOTO`, `GENERATED`, or `SOURCE` to the label and use the current palette with a small icon/text distinction; do not change projection shader behavior, priority, masking, or colorspace. In `buildPatchSources`, choose `src.proxy_geometry` when present; only when it is empty and `src.evidence_type === "photographed"`, clone `data.proxy_geometry` as the projection surface. An empty generated source remains empty. Pin the emitted payload, evidence label formatter, and fallback selection in the focused provenance test.

- [ ] **Step 5: Prove exporters preserve mixed layers without camera mutation**

In `exporters/_layers.py`, add `_projection_geometry_for_source(solve, src) -> tuple[list[AtlasProxyPrimitive], str]`. Return `(src.proxy_geometry, "source")` when private geometry exists; return `(solve.projection_scene.proxy_geometry, "primary_scene")` only for an empty photographed source; otherwise return `([], "source")`. Make `collect_projection_layers` use it and include `geometry_source` in each returned layer dict. Extend the Nuke layer test to serialize one photographed and one generated source, verify both layer cameras and evidence metadata in the manifest, verify the photographed geometry fallback, and assert that adding the Qwen source leaves every photographed camera matrix byte-identical.

- [ ] **Step 6: Run integration tests**

Run: `python -m pytest tests/test_add_patch_view.py tests/test_multiview_provenance.py tests/test_scene_health_gate.py tests/test_nuke_layers_export.py tests/test_debug_report.py -v`

Expected: all pass.

- [ ] **Step 7: Commit provenance integration**

```powershell
git add atlas_camera/comfy/nodes_geometry.py atlas_camera/comfy/viewport_payload.py atlas_camera/comfy/web/atlas_blockout.js atlas_camera/core/scene_health.py atlas_camera/exporters/_layers.py tests/test_add_patch_view.py tests/test_multiview_provenance.py tests/test_scene_health_gate.py tests/test_nuke_layers_export.py
git commit -m "feat(multiview): distinguish photographed and generated sources"
```

---

### Task 8: User workflow, catalog, capture guide, and X-H2 acceptance runner

**Files:**
- Create: `examples/atlas_multiview_raw_qwen_workflow.json`
- Create: `tools/validate_multiview_capture.py`
- Create: `tests/test_multiview_capture_tool.py`
- Modify: `docs/NODE_CATALOG.md`
- Modify: `docs/USER_GUIDE.md`
- Modify: `docs/DESIGN_RULES.md`
- Modify: `docs/ECOSYSTEM_GUIDE.md`
- Modify: `tests/test_example_workflows.py`
- Modify: `tests/test_shipping_workflow_paths.py`

**Interfaces:**
- Documents `AtlasLoadRAW x2/x3 -> AtlasMultiViewSolve -> geometry/viewport -> optional AtlasAddPatchView`.
- Acceptance runner consumes two or three RAW paths plus measured camera height and writes a JSON report and pair overlays.

- [ ] **Step 1: Write failing acceptance-tool tests**

```python
def test_manifest_rejects_missing_height_for_translated_capture(tmp_path):
    manifest = _manifest(tmp_path, camera_height_m=0.0)
    result = run_manifest(manifest, solve_fn=_translated_stub)
    assert result["outcome_code"] == "scale_unavailable"

def test_acceptance_report_hash_is_stable(tmp_path):
    first = run_manifest(_manifest(tmp_path), solve_fn=_stable_stub)
    second = run_manifest(_manifest(tmp_path), solve_fn=_stable_stub)
    assert canonical_json(first) == canonical_json(second)
```

- [ ] **Step 2: Run the tool tests and confirm failure**

Run: `python -m pytest tests/test_multiview_capture_tool.py -v`

Expected: tool module absent.

- [ ] **Step 3: Implement the local RAF acceptance runner**

Accept a manifest with ordered `raw_paths`, `camera_height_m`, `capture_mode`, `match_quality`, and `seed`. Use `atlas_camera.raw.import_raw`, build frames, call `solve_multiview`, and write canonical `registration.json` plus `pair_01.png`, `pair_02.png`, and `pair_12.png` when present. Do not commit RAF files or machine-absolute paths.

Document this manifest shape:

```json
{
  "raw_paths": ["left.raf", "centre.raf", "right.raf"],
  "camera_height_m": 1.43,
  "capture_mode": "auto",
  "match_quality": "balanced",
  "seed": 0
}
```

- [ ] **Step 4: Add and pin the shipping workflow**

Build a portable workflow using placeholder input-relative RAF paths, three `AtlasLoadRAW` nodes, `AtlasMultiViewSolve`, viewport/report nodes, and one downstream Qwen patch slot wired through `AtlasAddPatchView`. Preserve positional widget ordering, UUID node ids, and no absolute paths. Add the example filename to the pin tests.

- [ ] **Step 5: Update user and developer documentation**

Add the exact node row to `NODE_CATALOG.md`; add the static-scene/same-lens/70%-overlap/lateral-baseline/lens-height capture recipe and rotation-only limitations to `USER_GUIDE.md`; record the measured-vs-generated evidence rule and deterministic anchor rule in `DESIGN_RULES.md`; add the new solve source to `ECOSYSTEM_GUIDE.md`.

- [ ] **Step 6: Run workflow, documentation-adjacent, and tool tests**

Run: `python -m pytest tests/test_multiview_capture_tool.py tests/test_example_workflows.py tests/test_shipping_workflow_paths.py tests/test_frontend_mirrors.py -v`

Expected: all pass and the workflow contains no machine path.

- [ ] **Step 7: Commit workflow and documentation**

```powershell
git add examples/atlas_multiview_raw_qwen_workflow.json tools/validate_multiview_capture.py tests/test_multiview_capture_tool.py docs/NODE_CATALOG.md docs/USER_GUIDE.md docs/DESIGN_RULES.md docs/ECOSYSTEM_GUIDE.md tests/test_example_workflows.py tests/test_shipping_workflow_paths.py
git commit -m "docs(multiview): ship RAW capture and Qwen workflow"
```

---

### Task 9: Fresh-process determinism gate and full repository verification

**Files:**
- Create: `tests/test_multiview_determinism.py`
- Modify: `atlas_camera/core/multiview_types.py`
- Modify: `atlas_camera/core/multiview_features.py`
- Modify: `atlas_camera/core/multiview_geometry.py`
- Update generated graph files through `graphify update .`.

**Interfaces:**
- Produces the release gate proving repeated-process equality.

- [ ] **Step 1: Write the failing subprocess determinism test**

```python
def test_fresh_processes_emit_identical_multiview_json(tmp_path):
    outputs = []
    for ambient_seed in (1, 97, 2026):
        proc = subprocess.run(
            [sys.executable, "tests/multiview_subprocess_fixture.py",
             str(ambient_seed)], check=True, capture_output=True, text=True)
        outputs.append(proc.stdout)
    assert outputs[0] == outputs[1] == outputs[2]
```

The subprocess fixture sets different Python, NumPy, and OpenCV ambient seeds, solves the same committed synthetic three-view case, and prints canonical diagnostics plus `solve.to_json()`.

- [ ] **Step 2: Run the gate and confirm it fails before fixture support exists**

Run: `python -m pytest tests/test_multiview_determinism.py -v`

Expected: failure because the subprocess fixture is absent or output differs.

- [ ] **Step 3: Add the deterministic subprocess fixture and remove every unstable ordering found**

Create `tests/multiview_subprocess_fixture.py`. Fix only root causes: unsorted dict/set iteration, unstable equal-score ties, non-canonical float coercion, or inherited RNG use. Do not weaken exact equality to tolerances.

- [ ] **Step 4: Run all multi-view tests**

Run: `python -m pytest tests/test_multiview_types.py tests/test_multiview_features.py tests/test_multiview_geometry.py tests/test_multiview_solver.py tests/test_multiview_node.py tests/test_multiview_capture_tool.py tests/test_multiview_determinism.py -v`

Expected: all pass.

- [ ] **Step 5: Run the complete repository suite**

Run: `python -m pytest -q`

Expected: zero failures; optional heavy tests may skip through existing `importorskip` guards.

- [ ] **Step 6: Refresh and verify the knowledge graph**

Run: `graphify update .`

Expected: graph update succeeds. If the Windows `uv` trampoline cannot launch, use the interpreter recorded in `graphify-out/.graphify_python`; do not edit graph files manually.

- [ ] **Step 7: Inspect the final diff and commit the release gate**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors and only intended files present.

```powershell
git add tests/test_multiview_determinism.py tests/multiview_subprocess_fixture.py graphify-out
git commit -m "test(multiview): enforce fresh-process determinism"
```

## Completion Evidence

Before declaring implementation complete, capture:

- the baseline and final `python -m pytest -q` summaries;
- the exact fresh-process determinism test output;
- one successful translated two-photo synthetic report;
- one successful translated three-photo synthetic report;
- one successful rotation-only report;
- one expected failure for each documented failure code;
- the shipping workflow pin-test result;
- the X-H2 local acceptance report when RAF files are available; and
- `graphify update .` output.
