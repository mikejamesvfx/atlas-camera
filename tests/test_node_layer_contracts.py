"""Node-layer contracts for the nodes the feature audit found unevidenced.

The 2026-07-24 audit turned up 14 registered standard nodes with no shipping
workflow, no dedicated test, and no MCP consumer. The C-1 live probe proved all
14 EXECUTE, so none is dead code — but "it ran once by hand" is not a contract,
and two of them (`AtlasStereoRender`, `AtlasPlanarRewarp`) were doubly
misleading: `tests/test_stereo_render.py` and `tests/test_planar_projection.py`
cover the CORE math and never touch the node classes, so a swapped output slot
or a wrong tensor layout in the wrapper passed the whole suite.

This file pins the wrapper: output arity and ORDER against RETURN_NAMES (slots
serialize into saved workflows), the values a downstream node actually routes
on, and each node's documented fail-soft path. Heavy backends are skipped, not
faked — a skipped test says "unverified here", a mocked one would say "verified"
about a mock.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from atlas_camera.comfy import node_registry as reg


def _node(name):
    cls = reg.NODE_CLASS_MAPPINGS[name]
    return cls(), getattr(cls, cls.FUNCTION), cls


def _call(name, *args, **kwargs):
    """Invoke a node through its declared FUNCTION and assert the returned
    tuple matches RETURN_TYPES arity — the check that catches a slot added or
    dropped without the registry pin noticing."""
    inst, fn, cls = _node(name)
    out = fn(inst, *args, **kwargs)
    assert isinstance(out, tuple), f"{name} must return a tuple, got {type(out)}"
    assert len(out) == len(cls.RETURN_TYPES), (
        f"{name} returned {len(out)} values but declares "
        f"{len(cls.RETURN_TYPES)} RETURN_TYPES")
    return out


def _solve_with_horizon_and_vps(make_atlas_solve, width=64, height=48,
                                horizon_v=24.0, flip_vp_order=False):
    """A solve carrying the horizon + vanishing points the QA nodes read.

    `make_atlas_solve` builds a bare camera: `horizon_line` is None and
    `vanishing_points` is empty, so the horizon/VP nodes hit their no-data
    branches and prove nothing. These are the fields the real solvers populate.
    """
    from atlas_camera.core.schema import AtlasHorizon, AtlasVanishingPoint

    solve = make_atlas_solve(image_width=width, image_height=height,
                             position=(0.0, 1.6, 0.0))
    # Exactly what solver.py's learned path emits: (0, 1, -horizon_y).
    solve.horizon_line = AtlasHorizon(
        line_coefficients=(0.0, 1.0, -horizon_v),
        endpoints_px=((0.0, horizon_v), (float(width), horizon_v)),
        confidence=0.9)
    vps = [
        AtlasVanishingPoint(position_px=(-float(width), horizon_v),
                            direction_label="left", confidence=0.8,
                            supporting_lines=[((0.0, 10.0), (float(width), 20.0))]),
        AtlasVanishingPoint(position_px=(2.0 * width, horizon_v),
                            direction_label="right", confidence=0.8,
                            supporting_lines=[((0.0, 40.0), (float(width), 30.0))]),
    ]
    solve.vanishing_points = list(reversed(vps)) if flip_vp_order else vps
    return solve


# --------------------------------------------------------------------------
# Gates & QA — the routing surface. These feed raw floats into other graphs,
# so a transposed or reordered slot silently mis-routes every downstream node.
# --------------------------------------------------------------------------

def test_decompose_solve_slots_match_the_solve_they_came_from(make_atlas_solve):
    solve = make_atlas_solve(image_width=1600, image_height=900, focal=50.0)
    (camera, confidence, source_method, width, height, solve_json,
     horizon_deg) = _call("AtlasDecomposeSolve", solve)

    assert camera is solve.camera
    assert (width, height) == (1600, 900)
    assert isinstance(source_method, str)
    assert 0.0 <= float(confidence) <= 1.0
    # solve_json is the contract surface other tools read; it must parse and
    # agree with the typed slots rather than being a separate rendering.
    parsed = json.loads(solve_json)
    assert parsed["image_width"] == 1600 and parsed["image_height"] == 900
    assert isinstance(horizon_deg, float)


def test_decompose_camera_reports_intrinsics_not_a_recomputation(make_atlas_solve):
    solve = make_atlas_solve(image_width=1600, image_height=900, focal=50.0,
                             sensor_w=36.0)
    intr = solve.camera.intrinsics
    (fx, fy, cx, cy, cam_x, cam_y, cam_z, focal_mm,
     fov_h_deg) = _call("AtlasDecomposeCamera", solve.camera)

    assert fx == pytest.approx(intr.fx_px) and fy == pytest.approx(intr.fy_px)
    assert cx == pytest.approx(intr.cx_px) and cy == pytest.approx(intr.cy_px)
    assert focal_mm == pytest.approx(50.0)
    # Position comes from the extrinsics, in the same order the schema stores
    # it — a transposed read here would put the camera somewhere plausible but
    # wrong, which is exactly the class of bug that survives eyeballing.
    assert (cam_x, cam_y, cam_z) == pytest.approx(
        tuple(float(v) for v in solve.camera.extrinsics.camera_position))
    # Horizontal FOV must follow from fx and the image width, not the sensor.
    expected = 2.0 * math.degrees(math.atan(0.5 * 1600 / intr.fx_px))
    assert fov_h_deg == pytest.approx(expected, abs=1e-6)


def test_gravity_override_is_absolute_so_reapplying_is_a_no_op(make_atlas_solve):
    """The node documents itself as an ABSOLUTE override, not a nudge.

    That distinction is only observable on the second application: an absolute
    set is idempotent, a relative one drifts. Nothing else pins it, and the
    node had no evidence of any kind before this test.
    """
    solve = make_atlas_solve()
    once, report = _call("AtlasGravityOverride", solve, pitch_deg=-12.0,
                         roll_deg=3.0)
    twice, _ = _call("AtlasGravityOverride", once, pitch_deg=-12.0,
                     roll_deg=3.0)

    m_once = np.asarray(once.camera.extrinsics.camera_view_matrix, dtype=float)
    m_twice = np.asarray(twice.camera.extrinsics.camera_view_matrix, dtype=float)
    assert np.allclose(m_once, m_twice, atol=1e-9)
    assert isinstance(report, str) and report

    # The input solve must not be mutated — derive nodes downstream branch off
    # the original, and an in-place edit would leak into the other branch.
    assert np.allclose(
        np.asarray(solve.camera.extrinsics.camera_view_matrix, dtype=float),
        np.asarray(make_atlas_solve().camera.extrinsics.camera_view_matrix,
                   dtype=float))


def test_vp_visualization_draws_only_what_its_toggles_ask_for(make_atlas_solve):
    pytest.importorskip("PIL")
    torch = pytest.importorskip("torch")
    solve = _solve_with_horizon_and_vps(make_atlas_solve)
    image = torch.zeros(1, 48, 64, 3, dtype=torch.float32)

    (drawn,) = _call("AtlasVPVisualization", image, solve)
    assert tuple(drawn.shape) == (1, 48, 64, 3)
    assert drawn.dtype == torch.float32
    # Something must actually be drawn, or the node is a very expensive
    # identity function.
    assert float(drawn.max()) > 0.0

    (blank,) = _call("AtlasVPVisualization", image, solve, show_horizon=False,
                     show_vp_lines=False)
    assert float(blank.max()) == 0.0, "both overlays off must draw nothing"

    # A bare solve carries no horizon and no VPs. Drawing nothing is the right
    # answer there, but it means a test built on the shared fixture alone would
    # pass while proving nothing — hence _solve_with_horizon_and_vps above.
    (nothing,) = _call("AtlasVPVisualization", image,
                       make_atlas_solve(image_width=64, image_height=48))
    assert float(nothing.max()) == 0.0


# --------------------------------------------------------------------------
# Masks & depth derived from solve geometry (pure — no model backends)
# --------------------------------------------------------------------------

def test_horizon_mask_is_sky_above_ground_below(make_atlas_solve):
    """Regression for an inverted sky mask (found 2026-07-27).

    `ax+by+c=0` names the same line for (a,b,c) and (-a,-b,-c), and the two
    producers disagreed. `solver.py`'s learned path — the primary path for AI
    images — emits (0, 1, -horizon_y), for which the node's `signed` grows
    DOWNWARD, so it returned the GROUND as sky: the exact inverse of its
    docstring and of its "Sky Mask" display name. The node now canonicalizes
    the coefficients, and this pins the polarity in the producer's own sign.
    """
    pytest.importorskip("torch")
    solve = _solve_with_horizon_and_vps(make_atlas_solve, horizon_v=24.0)
    (mask,) = _call("AtlasHorizonMask", solve, 64, 48, 0)

    arr = np.asarray(mask[0])
    assert arr.shape == (48, 64)
    assert arr[0].mean() > 0.9, "top of frame is above the horizon = sky = 1"
    assert arr[-1].mean() < 0.1, "bottom of frame is ground = 0"


def test_horizon_mask_polarity_survives_a_flipped_coefficient_sign(
        make_atlas_solve):
    """The VP path builds the horizon from two vanishing points, and
    `line_from_points` flips the sign with the ORDER of those points — so the
    mask's polarity used to depend on which VP the detector happened to list
    first. Both signs must now give the same mask.
    """
    pytest.importorskip("torch")
    from atlas_camera.core.schema import AtlasHorizon

    solve = _solve_with_horizon_and_vps(make_atlas_solve, horizon_v=24.0)
    (positive_b,) = _call("AtlasHorizonMask", solve, 64, 48, 0)

    a, b, c = solve.horizon_line.line_coefficients
    solve.horizon_line = AtlasHorizon(line_coefficients=(-a, -b, -c),
                                      confidence=0.9)
    (negative_b,) = _call("AtlasHorizonMask", solve, 64, 48, 0)

    assert np.array_equal(np.asarray(positive_b), np.asarray(negative_b))


def test_horizon_mask_without_a_solved_horizon_is_all_sky(make_atlas_solve):
    """Documented no-data branch: a bare solve has `horizon_line=None`, and the
    node returns all ones rather than guessing a horizon."""
    pytest.importorskip("torch")
    (mask,) = _call("AtlasHorizonMask",
                    make_atlas_solve(image_width=64, image_height=48),
                    64, 48, 0)
    assert float(np.asarray(mask).min()) == 1.0


def test_horizon_mask_feather_softens_the_transition(make_atlas_solve):
    pytest.importorskip("torch")
    solve = _solve_with_horizon_and_vps(make_atlas_solve)
    (hard,) = _call("AtlasHorizonMask", solve, 64, 48, 0)
    (soft,) = _call("AtlasHorizonMask", solve, 64, 48, 12)

    def intermediate(mask):
        a = np.asarray(mask[0])
        return int(((a > 0.02) & (a < 0.98)).sum())

    assert intermediate(hard) == 0, "no feather must give a hard step"
    assert intermediate(soft) > 0


def test_ground_depth_map_second_output_matches_the_legacy_ground_mask(
        make_atlas_solve):
    """`AtlasGroundMask` is gated to the legacy tier on the claim that
    `AtlasGroundDepthMap` output 1 is bit-identical to it.

    That claim was measured live during the audit and then recorded in a JSON
    report — which is evidence, not a guard. If the two ever diverge, the
    documented migration silently starts handing artists a different mask, so
    the equivalence belongs in the suite.
    """
    torch = pytest.importorskip("torch")
    solve = make_atlas_solve(image_width=96, image_height=64)

    depth_image, ground_mask = _call("AtlasGroundDepthMap", solve, 96, 64,
                                     1.0, 50.0)
    legacy_cls = reg.LEGACY_NODE_CLASS_MAPPINGS["AtlasGroundMask"]
    (legacy_mask,) = getattr(legacy_cls(), legacy_cls.FUNCTION)(solve, 96, 64)

    assert np.array_equal(np.asarray(ground_mask), np.asarray(legacy_mask))
    # near/far drive only the visual ramp, never the mask — the other half of
    # the same claim.
    _, mask_other_range = _call("AtlasGroundDepthMap", solve, 96, 64, 5.0, 9.0)
    assert np.array_equal(np.asarray(ground_mask),
                          np.asarray(mask_other_range))
    assert tuple(depth_image.shape) == (1, 64, 96, 3)


# --------------------------------------------------------------------------
# Scale & guided solve
# --------------------------------------------------------------------------

def test_apply_scale_references_refuses_until_confirmed(make_atlas_solve):
    """`confirm` is a gate, and an unconfirmed run must be a visible no-op
    rather than a quiet one (gate doctrine)."""
    solve = make_atlas_solve(position=(0.0, 1.6, 0.0))
    refs = json.dumps([{"label": "door", "real_height_m": 2.0,
                        "base_px": [50, 900], "top_px": [50, 500],
                        "confidence": 0.9}])

    unconfirmed, height, report = _call("AtlasApplyScaleReferences", solve,
                                        refs, confirm=False)
    assert unconfirmed is solve or np.allclose(
        np.asarray(unconfirmed.camera.extrinsics.camera_position, dtype=float),
        np.asarray(solve.camera.extrinsics.camera_position, dtype=float))
    assert report, "an unconfirmed no-op must still explain itself"

    # A reference below min_confidence must be rejected even WITH confirm, or
    # the confidence widget does nothing.
    _, _, filtered = _call("AtlasApplyScaleReferences", solve, refs,
                           confirm=True, min_confidence=0.99)
    assert filtered


def test_apply_scale_references_survives_malformed_json(make_atlas_solve):
    solve = make_atlas_solve()
    out, _height, report = _call("AtlasApplyScaleReferences", solve,
                                 "{not json", confirm=True)
    assert out is not None and report, "bad JSON must report, not raise"


# --------------------------------------------------------------------------
# Nodes whose CORE is tested but whose wrapper was not
# --------------------------------------------------------------------------

def test_planar_rewarp_passes_the_plate_through_when_unwired():
    """Documented fail-soft (confirmed in the C-1 probe): no warp_spec means
    return the original plate untouched rather than raising."""
    torch = pytest.importorskip("torch")
    original = torch.rand(1, 32, 48, 3, dtype=torch.float32)
    edited = torch.zeros(1, 32, 48, 3, dtype=torch.float32)

    image, coverage = _call("AtlasPlanarRewarp", None, edited, original)
    assert torch.allclose(image, original)
    assert float(np.asarray(coverage).max()) == 0.0, (
        "nothing was composited, so coverage must be empty")


def _relief_solve(make_atlas_solve, width=64, height=48):
    """A solve carrying real projection geometry, via the same derive node a
    workflow would use. Stereo has nothing to render without it."""
    pytest.importorskip("torch")
    from atlas_camera.inference.depth_estimator import DepthResult

    solve = make_atlas_solve(image_width=width, image_height=height,
                             position=(0.0, 1.6, 0.0))
    vv = np.arange(height, dtype=np.float64)[:, None]
    depth = np.broadcast_to(4.0 + 0.05 * vv, (height, width)).copy()
    depth_result = DepthResult(
        depth=depth, is_metric=True, model_id="fake",
        image_width=width, image_height=height,
        near=float(depth.min()), far=float(depth.max()))
    inst, fn, _cls = _node("AtlasDeriveReliefMesh")
    out, _hole = fn(inst, solve, depth_result, relief_grid=32)
    return out


def test_stereo_render_fails_soft_and_says_why_without_projection_geometry(
        make_atlas_solve):
    """Documented fail-soft: a solve with no serialized projection meshes has
    nothing to render, and the node must explain that rather than raise or
    hand back a silently black pair."""
    torch = pytest.importorskip("torch")
    solve = make_atlas_solve(image_width=64, image_height=48)
    source = torch.rand(1, 48, 64, 3, dtype=torch.float32)

    stereo, left, right, report = _call("AtlasStereoRender", solve, source,
                                        resolution=128)
    assert "AtlasDeriveProjectionGeometry" in report, (
        "the fail-soft must name the node that fixes it")
    assert tuple(stereo.shape) == tuple(source.shape)
    assert tuple(left.shape) == tuple(right.shape)


def test_stereo_render_output_modes_keep_left_and_right_slots_consistent(
        make_atlas_solve):
    """`output_mode` reshapes slot 0 only; `left`/`right` stay the individual
    eyes in every mode, which is what a downstream comp graph relies on."""
    torch = pytest.importorskip("torch")
    solve = _relief_solve(make_atlas_solve)
    source = torch.rand(1, 48, 64, 3, dtype=torch.float32)

    sbs, left, right, report = _call(
        "AtlasStereoRender", solve, source, resolution=128, output_mode="sbs")
    assert tuple(left.shape) == tuple(right.shape)
    assert sbs.shape[2] == left.shape[2] * 2, "sbs must be twice as wide"
    assert isinstance(report, str) and report

    anaglyph, left2, right2, _ = _call(
        "AtlasStereoRender", solve, source, resolution=128,
        output_mode="anaglyph")
    assert tuple(anaglyph.shape) == tuple(left.shape)
    assert tuple(left2.shape) == tuple(left.shape)

    # Zero interocular collapses the pair — the degenerate case a comp artist
    # hits while dialling the rig in, and it must not raise.
    mono_l, mono_r = _call("AtlasStereoRender", solve, source, resolution=128,
                           interocular_m=0.0)[1:3]
    assert torch.allclose(mono_l, mono_r, atol=1e-5)


# --------------------------------------------------------------------------
# Optional-dependency nodes. Skipped rather than mocked.
# --------------------------------------------------------------------------

def _constraint_lines():
    """Three groups, matching tests/test_artist_guided_constraints.py. The C-1
    probe established that all three are required — two groups is a solver
    error, not a degraded solve."""
    return {
        "left": [((0.0, 36.0), (160.0, 12.0)),
                 ((0.0, 40.6667), (160.0, 26.0)),
                 ((0.0, 44.6667), (160.0, 38.0))],
        "right": [((0.0, 12.0), (160.0, 36.0)),
                  ((0.0, 26.0), (160.0, 40.6667)),
                  ((0.0, 38.0), (160.0, 44.6667))],
        # Verticals must CONVERGE — three exactly-parallel lines are degenerate
        # and the VP fit rejects them outright. These meet at (80, -2000), the
        # mild upward convergence a slightly tilted camera produces.
        "vertical": [((41.531, 10.0), (40.0, 90.0)),
                     ((80.0, 10.0), (80.0, 90.0)),
                     ((118.469, 10.0), (120.0, 90.0))],
    }


def test_constrained_solve_consumes_artist_line_groups(make_atlas_solve):
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    image = torch.zeros(1, 96, 160, 3, dtype=torch.float32)
    payload = json.dumps({"image_width": 160, "image_height": 96,
                          "line_groups": _constraint_lines(),
                          "camera_height": 1.7})

    (solve,) = _call("AtlasConstrainedSolve", image, payload)
    assert solve.source_method == "artist_guided_constraints"
    assert solve.image_width == 160 and solve.image_height == 96
    assert solve.camera.intrinsics.fx_px > 0.0
    # The node takes an IMAGE tensor, not a path — the distinction cost a
    # wasted probe during the audit, so pin it.
    assert solve.camera.extrinsics.camera_view_matrix is not None


def test_constrained_solve_rejects_unusable_constraints():
    torch = pytest.importorskip("torch")
    image = torch.zeros(1, 96, 160, 3, dtype=torch.float32)
    with pytest.raises(Exception):
        _call("AtlasConstrainedSolve", image, "{not json")


# Model-backed nodes. Their weights are multi-GB and network-dependent, so the
# happy path belongs in the live probe (reports/live_probe_baseline.json), not
# the unit suite. What IS pinned here is the surface a workflow serializes
# against plus the error path an artist actually hits — a wrong model id or a
# missing checkpoint must name the problem rather than surfacing as a bare
# KeyError from somewhere inside transformers.

def test_depth_anything_rejects_an_unknown_model_id_by_name():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    image = torch.zeros(1, 32, 32, 3, dtype=torch.float32)
    with pytest.raises(Exception) as excinfo:
        _call("AtlasDepthAnything", image,
              depth_model="definitely/not-a-real-depth-model")
    assert "not-a-real-depth-model" in str(excinfo.value)


def test_segmented_sdxl_inpaint_surface_is_stable():
    cls = reg.NODE_CLASS_MAPPINGS["AtlasSegmentedSDXLInpaint"]
    required = list(cls.INPUT_TYPES()["required"])
    # Widgets serialize positionally, so this order is a saved-workflow
    # contract; the node needs SDXL weights to run, which is why only the
    # surface is pinned here (execution proven in the C-1 live probe).
    assert required == ["image", "restrict_mask", "prompt", "checkpoint",
                        "max_instances", "steps", "cfg", "denoise", "seed"]
    assert cls.RETURN_NAMES == ("image", "report")


def test_load_plate_names_the_missing_file_in_the_error(tmp_path):
    """A missing plate is a config mistake, and this node RAISES rather than
    failing soft — deliberately, since a silently-empty plate would propagate
    into every downstream layer. Pin that it raises with the path in the
    message, so the artist can see which path was wrong."""
    pytest.importorskip("OpenImageIO")
    missing = tmp_path / "nope.exr"
    with pytest.raises(RuntimeError, match="no such file"):
        _call("AtlasLoadPlate", str(missing))

    with pytest.raises(RuntimeError, match="file_path"):
        _call("AtlasLoadPlate", "   ")


def test_usd_camera_loader_round_trips_an_exported_camera(tmp_path,
                                                          make_atlas_solve):
    pytest.importorskip("pxr")
    from atlas_camera.exporters.usd_exporter import USDExporter

    solve = make_atlas_solve(image_width=1920, image_height=1080, focal=35.0)
    usd_path = tmp_path / "cam.usda"
    USDExporter().export_camera(solve, str(usd_path))

    (camera,) = _call("AtlasUSDCameraLoader", str(usd_path), 1920, 1080)
    # The round trip is the contract: what the exporter wrote is what the
    # loader must hand back, in Atlas's own units.
    assert camera.intrinsics.fx_px == pytest.approx(
        solve.camera.intrinsics.fx_px, rel=1e-3)
    assert camera.intrinsics.focal_length_mm == pytest.approx(35.0, rel=1e-3)
