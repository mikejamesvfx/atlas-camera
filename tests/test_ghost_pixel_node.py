"""Contract for AtlasGhostPixelMap 👻 — the geometry half.

tests/test_ghost_pixels.py pins the classification algebra on hand-built
fields. This file pins what the NODE measures off a real z-buffered render:
that a lateral move behind a depth cliff produces GHOST, that a plate that
never had geometry produces INVALID instead, and that a zoom cannot disocclude
because it does not move the eye.

The fixture mirrors tests/test_disocclusion_guide.py deliberately — same
relief mesh, same step cliff — because the headline invariant is that this
node reclassifies the guide's hole rather than finding a different one.
"""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from atlas_camera.comfy.node_registry import NODE_CLASS_MAPPINGS  # noqa: E402
from atlas_camera.core.ghost_pixels import GhostClass  # noqa: E402

GHOST = NODE_CLASS_MAPPINGS["AtlasGhostPixelMap"]
GUIDE = NODE_CLASS_MAPPINGS["AtlasDisocclusionGuide"]
W = H = 96
RES = 128


def _solve(step: bool = True, punch: bool = False):
    """A solve carrying a real relief mesh.

    `step` adds a 4x depth cliff — the occluder whose edge a lateral move
    reveals geometry behind. `punch` removes depth from a square INSIDE the
    near block, so the mesh never covers it and the PLATE itself is a hole
    there. It has to sit somewhere the control solidly covers, or the test is
    vacuous: the relief mesh already leaves the frame edges uncovered, so a
    hole punched out there would classify INVALID with or without the punch.
    """
    from atlas_camera.comfy.nodes_geometry import AtlasDeriveReliefMesh
    from atlas_camera.core.intrinsics import build_intrinsics
    from atlas_camera.core.schema import AtlasCamera, AtlasExtrinsics, AtlasSolve
    from atlas_camera.inference.depth_estimator import DepthResult

    d = np.full((H, W), 12.0, dtype=np.float32)
    if step:
        d[30:70, 30:70] = 3.0
    if punch:
        d[40:56, 40:56] = np.nan     # never-derived geometry inside the block
    depth = DepthResult(depth=d, is_metric=True, model_id="t",
                        image_width=W, image_height=H)
    intr = build_intrinsics(image_width=W, image_height=H,
                            focal_length_mm=35.0, sensor_width_mm=36.0)
    cam = AtlasCamera(intrinsics=intr, extrinsics=AtlasExtrinsics(
        camera_position=(0.0, 0.0, 0.0),
        camera_world_matrix=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),
                             (0, 0, 0, 1))))
    base = AtlasSolve(camera=cam, image_width=W, image_height=H)
    return AtlasDeriveReliefMesh().derive(base, depth, relief_grid=96,
                                          depth_edge_rel=0.5)[0]


def _path(position=(2.0, 0.0, 0.0), frames=3, fov_deg=None):
    from atlas_camera.core.schema import AtlasCameraKeyframe, AtlasCameraPath
    return AtlasCameraPath(keyframes=[
        AtlasCameraKeyframe(frame_index=0, position=(0.0, 0.0, 0.0),
                            target=(0.0, 0.0, -10.0), fov_deg=fov_deg),
        AtlasCameraKeyframe(frame_index=frames - 1, position=position,
                            target=(0.0, 0.0, -10.0), fov_deg=fov_deg),
    ], frame_count=frames)


def _img():
    return torch.full((1, H, W, 3), 0.5, dtype=torch.float32)


def _run(**kw):
    kw.setdefault("resolution", RES)
    out = GHOST().classify(_solve(kw.pop("step", True),
                                  kw.pop("punch", False)),
                           _img(), **kw)
    guide, valid, ghost, invalid, overlay, report = out
    return {
        "guide": guide, "valid": valid[0].numpy(), "ghost": ghost[0].numpy(),
        "invalid": invalid[0].numpy(), "overlay": overlay,
        "report": json.loads(report),
    }


# ------------------------------------------------------------ A: identity

def test_identity_reprojection_produces_almost_no_ghosts():
    """Target == source. Nothing moved, so nothing can have been revealed."""
    r = _run()                                    # no camera_path
    stats = r["report"]["worst_frame"]
    assert stats["ghost_fraction"] < 0.01, (
        f"identity produced {stats['ghost_fraction']:.2%} ghost — the "
        "classifier is finding disocclusion where the camera did not move")
    # A partial relief mesh legitimately leaves a large part of the frame
    # uncovered — measured ~44% on this fixture, and AtlasDisocclusionGuide
    # documents the same effect ("a FLAT depth map with zero occlusion still
    # renders ~50% uncovered"). The contract is NOT that identity is fully
    # covered; it is that everything uncovered lands in INVALID, because
    # nothing was ever occluding it.
    assert stats["invalid_fraction"] > 0.0
    assert (stats["valid_fraction"] + stats["invalid_fraction"]
            == pytest.approx(1.0, abs=1e-6))


def test_identity_report_names_the_solved_camera_as_the_source():
    assert "solved camera" in _run()["report"]["view_source"]


# ------------------------------------- B: flat plane, small lateral shift

def test_a_flat_scene_shifted_sideways_has_no_interior_ghost():
    """No depth discontinuity means nothing can occlude anything."""
    r = _run(step=False, camera_path=_path((0.5, 0.0, 0.0)))
    interior = r["ghost"][16:-16, 16:-16]
    assert interior.mean() < 0.02, (
        "a fronto-parallel plane produced interior ghosts — those can only be "
        "rasterisation holes being misread as disocclusion")


# ------------------------------------ C: foreground occluder — the headline

def test_a_lateral_move_past_a_depth_cliff_reveals_ghost_pixels():
    r = _run(camera_path=_path((2.0, 0.0, 0.0)))
    stats = r["report"]["worst_frame"]
    assert stats["ghost_pixel_count"] > 0, (
        "the move revealed nothing behind a 4x depth cliff — the fixture or "
        "the classifier is broken")
    assert stats["ghost_region_count"] >= 1
    assert stats["largest_ghost_region_px"] > 1


def test_ghost_appears_beside_the_occluder_not_scattered_over_the_frame():
    """Disocclusion is a BAND at a depth edge, not speckle."""
    r = _run(camera_path=_path((2.0, 0.0, 0.0)))
    stats = r["report"]["worst_frame"]
    # the largest connected region must dominate: a real reveal is coherent
    assert (stats["largest_ghost_region_px"]
            >= 0.30 * stats["ghost_pixel_count"])


def test_a_bigger_move_reveals_more_than_a_smaller_one():
    """Monotonicity — the single cheapest check that this tracks geometry."""
    small = _run(camera_path=_path((0.5, 0.0, 0.0)))["report"]["worst_frame"]
    big = _run(camera_path=_path((3.0, 0.0, 0.0)))["report"]["worst_frame"]
    assert big["ghost_fraction"] > small["ghost_fraction"]


# ------------------------------------------- D: missing depth is NOT ghost

def test_frame_edge_the_mesh_never_reached_is_invalid_not_ghost():
    """The relief mesh does not reach the frame edge, and that is not a reveal.

    This is the most common real case by area — a partial mesh — and the one
    that made the guide's report separate its causes in the first place: a
    flat depth map with zero occlusion still renders ~50% uncovered. Calling
    it disocclusion would aim a filler at the whole border.
    """
    r = _run(camera_path=_path((1.0, 0.0, 0.0)))
    top = slice(0, 8)
    assert r["invalid"][top, :].mean() > 0.5, (
        "the uncovered frame edge did not classify INVALID")
    assert r["ghost"][top, :].mean() < 0.05, (
        "the uncovered frame edge leaked into GHOST — this is the sky-fill "
        "bug the classifier exists to prevent")


def test_punching_out_geometry_converts_valid_into_invalid_never_ghost():
    """The control covers this square. Removing its depth must not invent one.

    Deleting geometry can only ever make Atlas know LESS. If a filler were
    pointed at the result it would be inventing over an upstream gap, which
    is the distinction between INVALID and GHOST stated as a delta.
    """
    region = (slice(42, 54), slice(42, 54))
    control = _run(camera_path=_path((1.0, 0.0, 0.0)))
    punched = _run(punch=True, camera_path=_path((1.0, 0.0, 0.0)))

    assert control["valid"][region].mean() > 0.9, (
        "fixture drift: the control no longer covers the punched square, so "
        "this test would pass without measuring anything")
    assert punched["invalid"][region].mean() > 0.9
    assert punched["ghost"][region].mean() < 0.05


# --------------------------------------------------- E: large translation

def test_a_large_move_stays_stable_and_does_not_ghost_the_whole_frame():
    r = _run(camera_path=_path((12.0, 0.0, 0.0)))
    stats = r["report"]["worst_frame"]
    assert stats["ghost_fraction"] < 0.95, (
        "ghost swallowed the frame — out-of-frame pixels are being counted as "
        "disocclusion instead of OUT_OF_BOUNDS")
    assert np.isfinite(stats["ghost_fraction"])
    assert stats["out_of_bounds_pixel_count"] > 0, (
        "a 12-unit swing left the plate frame but nothing classified "
        "OUT_OF_BOUNDS")


def test_fractions_stay_normalised_on_a_large_move():
    stats = _run(camera_path=_path((12.0, 0.0, 0.0)))["report"]["worst_frame"]
    total = sum(stats[f"{n}_fraction"] for n in
                ("valid", "ghost", "invalid", "out_of_bounds"))
    assert total == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------ F: zoom vs dolly

def test_a_zoom_cannot_disocclude_but_a_dolly_can():
    """A focal change moves every pixel without moving the EYE.

    Parallax is what reveals, and a zoom has none. If these two produced the
    same ghost map the node would be measuring image scale rather than camera
    geometry.
    """
    zoom = _run(camera_path=_path((0.0, 0.0, 0.0), fov_deg=20.0))
    dolly = _run(camera_path=_path((0.0, 0.0, 4.0)))
    z, d = zoom["report"]["worst_frame"], dolly["report"]["worst_frame"]
    assert z["ghost_fraction"] < 0.01, (
        f"a pure zoom produced {z['ghost_fraction']:.2%} ghost; a focal "
        "change reveals nothing because the eye never moves")
    assert d["ghost_fraction"] > z["ghost_fraction"]


def test_zoom_actually_changes_the_render():
    """Guards the test above from passing because fov was silently ignored.

    `sample_camera_path` returns poses only — the fov channel has its own
    sampler, which AtlasDisocclusionGuide never calls. Without this check,
    "a zoom produces no ghosts" would pass just as well on a node that
    rendered the solved focal and ignored the zoom entirely.
    """
    wide = _run(camera_path=_path((0.0, 0.0, 0.0), fov_deg=60.0))["guide"]
    tight = _run(camera_path=_path((0.0, 0.0, 0.0), fov_deg=15.0))["guide"]
    assert not np.allclose(wide.numpy(), tight.numpy()), (
        "the fov channel was ignored — zoom is not being applied at all")


# ------------------------------------------------------- G: exclude mask

def test_excluded_pixels_leave_the_fractions_entirely():
    mask = torch.zeros(1, H, W, dtype=torch.float32)
    mask[:, 0:24, :] = 1.0
    stats = _run(camera_path=_path((2.0, 0.0, 0.0)),
                 exclude_mask=mask)["report"]["worst_frame"]
    assert stats["excluded_fraction"] > 0.0
    assert stats["scored_px"] < stats["total_px"]
    assert stats["low_confidence_pixel_count"] > 0


# ------------------------------------------------- the headline invariant

def test_the_classes_partition_exactly_the_guides_hole_mask():
    """This node reclassifies the guide's hole — it must not find another.

    Both run the same rasteriser at the same raster, so ghost | invalid |
    out_of_bounds has to equal the guide's hole_mask pixel for pixel. If it
    ever does not, one of the two is measuring something else.
    """
    solve, img, path = _solve(), _img(), _path((2.0, 0.0, 0.0))
    _, hole_mask, _ = GUIDE().guide(solve, img, camera_path=path,
                                    resolution=RES)
    out = GHOST().classify(solve, img, camera_path=path, resolution=RES)
    _, valid, ghost, invalid, _, _ = out

    hole = hole_mask.numpy() > 0.5
    classified_hole = (ghost.numpy() > 0.5) | (invalid.numpy() > 0.5)
    # out_of_bounds is the remaining hole class; reconstruct it as the
    # complement of valid so the assertion needs no extra output.
    classified_hole |= ~(valid.numpy() > 0.5) & ~classified_hole
    np.testing.assert_array_equal(classified_hole, hole)


# ------------------------------------------------------------- contracts

def test_a_solve_without_meshes_refuses_instead_of_claiming_total_ghost():
    """Claiming the whole frame is ghost would send a model off inventing."""
    from atlas_camera.core.intrinsics import build_intrinsics
    from atlas_camera.core.schema import AtlasCamera, AtlasExtrinsics, AtlasSolve

    intr = build_intrinsics(image_width=W, image_height=H,
                            focal_length_mm=35.0, sensor_width_mm=36.0)
    bare = AtlasSolve(camera=AtlasCamera(intrinsics=intr,
                                         extrinsics=AtlasExtrinsics()),
                      image_width=W, image_height=H)
    _, _, ghost, _, _, report = GHOST().classify(bare, _img())
    assert ghost.numpy().sum() == 0
    assert "error" in json.loads(report)


def test_the_report_is_json_and_publishes_the_class_numbering():
    report = _run()["report"]
    assert report["class_values"]["GHOST"] == int(GhostClass.GHOST)
    assert report["class_values"]["INVALID"] == int(GhostClass.INVALID)
    assert set(report) >= {"frames", "raster", "worst_frame", "per_frame"}


def test_every_frame_of_a_path_is_classified():
    r = GHOST().classify(_solve(), _img(), camera_path=_path(frames=4),
                         resolution=RES)
    assert r[0].shape[0] == 4
    assert len(json.loads(r[5])["per_frame"]) == 4


def test_masks_and_overlay_share_the_render_raster():
    r = _run(camera_path=_path((1.0, 0.0, 0.0)))
    assert r["ghost"].shape == r["valid"].shape == r["invalid"].shape
    assert r["overlay"].shape[1:3] == r["guide"].shape[1:3]


def test_classification_is_deterministic():
    first = _run(camera_path=_path((2.0, 0.0, 0.0)))["ghost"]
    second = _run(camera_path=_path((2.0, 0.0, 0.0)))["ghost"]
    np.testing.assert_array_equal(first, second)
