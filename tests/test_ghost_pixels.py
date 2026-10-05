"""GHOST, INVALID and OUT_OF_BOUNDS must never collapse into each other.

The contract pinned here is the one the node exists for: a hole in the target
view has at least three unrelated causes, and only ONE of them is a surface a
generator should be asked to invent. The occlusion-fill doctrine records what
happens when the distinction is left to prose — a raw "peak hole" figure that
climbed 64.8% -> 86.1% on an IMPROVING run, because it counted sky.

These tests cover the classification algebra with hand-built boolean fields,
so they run without a rasteriser and without torch (CI installs neither).
The geometry that produces those fields is pinned by
tests/test_ghost_pixel_node.py.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.ghost_pixels import (  # noqa: E402
    CLASS_COLOURS,
    GhostClass,
    build_debug_overlay,
    class_masks,
    classify_target_coverage,
    ghost_stats,
)


def _fields(shape=(8, 8)):
    """All-covered, all-inside, no-plate-hole: the identity case."""
    return {
        "covered": np.ones(shape, dtype=bool),
        "inside_plate_frame": np.ones(shape, dtype=bool),
        "plate_hole_at_target": np.zeros(shape, dtype=bool),
    }


# --------------------------------------------------------------- classes

def test_fully_covered_view_is_all_valid_and_has_no_ghosts():
    cm = classify_target_coverage(**_fields())
    assert (cm == GhostClass.VALID).all()
    assert ghost_stats(cm)["ghost_fraction"] == 0.0


def test_hole_behind_an_occluder_is_ghost():
    """Inside the plate frame, not a hole there — something stood in front."""
    f = _fields()
    f["covered"][2:5, 2:5] = False
    cm = classify_target_coverage(**f)
    assert (cm[2:5, 2:5] == GhostClass.GHOST).all()
    assert ghost_stats(cm)["ghost_pixel_count"] == 9


def test_hole_that_is_also_a_plate_hole_is_invalid_not_ghost():
    """Sky / never-derived geometry. Nothing occluded it, so nothing to fill.

    This is the property that stops a generator being aimed at the sky — the
    G5 field run measured the auto-ROI ranking a sky cluster first without it.
    """
    f = _fields()
    f["covered"][0:3, :] = False
    f["plate_hole_at_target"][0:3, :] = True
    cm = classify_target_coverage(**f)
    assert (cm[0:3, :] == GhostClass.INVALID).all()
    stats = ghost_stats(cm)
    assert stats["ghost_pixel_count"] == 0
    assert stats["invalid_pixel_count"] == 24


def test_hole_outside_the_plate_frame_is_out_of_bounds_not_ghost():
    """The camera swung past the edge of the photograph. That is outpainting."""
    f = _fields()
    f["covered"][:, 0:2] = False
    f["inside_plate_frame"][:, 0:2] = False
    cm = classify_target_coverage(**f)
    assert (cm[:, 0:2] == GhostClass.OUT_OF_BOUNDS).all()
    assert ghost_stats(cm)["ghost_pixel_count"] == 0


def test_the_three_hole_causes_coexist_without_bleeding():
    """One frame, three causes, disjoint regions — the headline contract."""
    f = _fields()
    f["covered"][0:2, :] = False          # sky band
    f["plate_hole_at_target"][0:2, :] = True
    f["covered"][:, 6:8] = False          # swung off the plate
    f["inside_plate_frame"][:, 6:8] = False
    f["covered"][4:6, 1:3] = False        # genuine disocclusion

    cm = classify_target_coverage(**f)
    masks = class_masks(cm)

    assert masks["invalid"].any() and masks["out_of_bounds"].any()
    assert (cm[4:6, 1:3] == GhostClass.GHOST).all()
    # every hole is classified exactly once, and only holes are classified
    holes = ~f["covered"]
    assert (masks["valid"] == f["covered"]).all()
    assert (masks["ghost"] | masks["invalid"] | masks["out_of_bounds"]
            == holes).all()


def test_out_of_bounds_wins_over_plate_hole():
    """A pixel with no plate counterpart cannot be tested against the plate."""
    f = _fields()
    f["covered"][:] = False
    f["inside_plate_frame"][:] = False
    f["plate_hole_at_target"][:] = True   # meaningless here; must not win
    cm = classify_target_coverage(**f)
    assert (cm == GhostClass.OUT_OF_BOUNDS).all()


def test_coverage_wins_over_every_hole_class():
    """A pixel the rasteriser covered is VALID whatever the plate says."""
    f = _fields()
    f["plate_hole_at_target"][:] = True
    cm = classify_target_coverage(**f)
    assert (cm == GhostClass.VALID).all()


# --------------------------------------------------------------- excludes

def test_exclude_overrides_geometry_and_never_reports_as_ghost():
    """An artist declaration is intent, not evidence."""
    f = _fields()
    f["covered"][2:5, 2:5] = False        # would otherwise be GHOST
    exclude = np.zeros_like(f["covered"])
    exclude[2:5, 2:5] = True
    cm = classify_target_coverage(**f, exclude=exclude)
    assert (cm[2:5, 2:5] == GhostClass.LOW_CONFIDENCE).all()
    assert ghost_stats(cm)["ghost_pixel_count"] == 0


def test_excluded_pixels_leave_both_numerator_and_denominator():
    """`move_budget.disocclusion_fraction`'s ignore rule, restated.

    Without it sky reads as a permanent hole and every fraction collapses.
    """
    f = _fields(shape=(10, 10))
    f["covered"][0:5, :] = False          # half the frame excluded away
    exclude = np.zeros_like(f["covered"])
    exclude[0:5, :] = True
    cm = classify_target_coverage(**f, exclude=exclude)

    stats = ghost_stats(cm)
    assert stats["total_px"] == 100
    assert stats["scored_px"] == 50
    # the surviving half is fully covered, so VALID is 100% of what is scored
    assert stats["valid_fraction"] == 1.0
    assert stats["excluded_fraction"] == 0.5


# --------------------------------------------------------------- stats

def test_ghost_regions_are_counted_and_the_largest_measured():
    f = _fields(shape=(12, 12))
    f["covered"][1:3, 1:3] = False        # 4 px
    f["covered"][6:10, 6:10] = False      # 16 px
    stats = ghost_stats(classify_target_coverage(**f))
    assert stats["ghost_region_count"] == 2
    assert stats["largest_ghost_region_px"] == 16


def test_fractions_sum_to_one_when_nothing_is_excluded():
    f = _fields(shape=(10, 10))
    f["covered"][0:2, :] = False
    f["plate_hole_at_target"][0:2, :] = True
    f["covered"][:, 9] = False
    f["inside_plate_frame"][:, 9] = False
    f["covered"][5, 5] = False
    stats = ghost_stats(classify_target_coverage(**f))
    total = sum(stats[f"{n}_fraction"]
                for n in ("valid", "ghost", "invalid", "out_of_bounds"))
    assert total == pytest.approx(1.0)


def test_stats_are_json_safe_scalars():
    """The report is JSON — numpy scalars would not serialise."""
    import json

    stats = ghost_stats(classify_target_coverage(**_fields()))
    json.dumps(stats)                      # must not raise
    for key, value in stats.items():
        assert isinstance(value, (int, float)), f"{key} is {type(value)}"


# --------------------------------------------------------------- overlay

def test_overlay_keeps_the_underlying_image_visible():
    cm = classify_target_coverage(**_fields())
    rgb = np.full((8, 8, 3), 0.5, dtype=np.float32)
    out = build_debug_overlay(rgb, cm)
    assert out.shape == (8, 8, 3)
    # green-tinted, but the 0.5 grey still contributes
    assert out[..., 1].mean() > 0.5
    assert out[..., 0].mean() == pytest.approx(0.5 * (1 - 0.55))


def test_overlay_distinguishes_every_class_by_colour():
    f = _fields()
    f["covered"][0:2, :] = False
    f["plate_hole_at_target"][0:2, :] = True
    f["covered"][:, 7] = False
    f["inside_plate_frame"][:, 7] = False
    f["covered"][4, 4] = False
    cm = classify_target_coverage(**f)
    out = build_debug_overlay(np.zeros((8, 8, 3), np.float32), cm, alpha=1.0)

    for cls in (GhostClass.VALID, GhostClass.GHOST, GhostClass.INVALID,
                GhostClass.OUT_OF_BOUNDS):
        sel = cm == int(cls)
        assert sel.any(), f"fixture did not produce {cls.name}"
        assert np.allclose(out[sel], np.asarray(CLASS_COLOURS[cls]))


def test_overlay_rejects_a_mismatched_class_map():
    cm = classify_target_coverage(**_fields(shape=(8, 8)))
    with pytest.raises(ValueError, match="does not match"):
        build_debug_overlay(np.zeros((4, 4, 3), np.float32), cm)


# --------------------------------------------------------------- contracts

def test_class_values_are_the_published_numbering():
    """These serialise into reports and downstream consumers. Append only."""
    assert (int(GhostClass.INVALID), int(GhostClass.VALID),
            int(GhostClass.GHOST), int(GhostClass.OUT_OF_BOUNDS),
            int(GhostClass.LOW_CONFIDENCE),
            int(GhostClass.VISIBILITY_CONFLICT)) == (0, 1, 2, 3, 4, 5)


def test_mismatched_field_shapes_are_refused_not_broadcast():
    f = _fields()
    f["inside_plate_frame"] = np.ones((4, 4), dtype=bool)
    with pytest.raises(ValueError, match="does not match covered"):
        classify_target_coverage(**f)


def test_classification_is_deterministic():
    f = _fields()
    f["covered"][2:5, 2:5] = False
    first = classify_target_coverage(**f)
    for _ in range(3):
        assert (classify_target_coverage(**f) == first).all()
