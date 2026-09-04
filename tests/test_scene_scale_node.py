"""AtlasSceneScale — support-geometry sizes measured from the depth map.

The node exists because a ground plane and a sky card are sized in METRES, so a
set of numbers tuned on one plate is wrong on the next: across a 31-plate sweep
median scene distance ran 0.95 to 586. What is pinned here is that the sizes
scale WITH the scene (the whole point), that the artist-derived ratios are
reproduced exactly, and that the two refusals are refusals rather than confident
numbers describing nothing.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.comfy.nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    AtlasSceneScale,
)


def _depth(values, *, is_metric=True):
    return SimpleNamespace(depth=np.asarray(values, dtype=np.float32),
                           is_metric=is_metric)


def _ramp(near=5.0, far=40.0, h=64, w=64):
    """A scene with real depth spread — a ground receding from the camera."""
    return np.linspace(near, far, h)[:, None] * np.ones((1, w))


def test_registered_and_output_names():
    assert NODE_CLASS_MAPPINGS["AtlasSceneScale"] is AtlasSceneScale
    assert AtlasSceneScale.RETURN_NAMES == (
        "ground_width_m", "ground_depth_m", "ground_offset_z",
        "sky_distance_m", "median_m", "p99_m", "report")


def test_sizes_are_the_measured_ratios_of_the_scene():
    d = _ramp()
    gw, gd, off, sky, median, p99, report = AtlasSceneScale().measure(_depth(d))

    assert median == pytest.approx(float(np.median(d)))
    assert p99 == pytest.approx(float(np.percentile(d, 99)))
    # Defaults are the artist's hand-tuned judgement expressed as ratios.
    assert gw == pytest.approx(median * 3.0)
    assert gd == pytest.approx(median * 6.0)
    assert off == pytest.approx(median * -1.5)
    assert sky == pytest.approx(p99 * 1.8)
    # Deeper than wide on purpose: the plane exists to be travelled INTO.
    assert gd > gw
    # And the plane is pushed AWAY from camera, not pulled in front of it.
    assert off < 0
    assert "metric" in report


def test_the_defaults_reproduce_the_plate_they_were_tuned_on():
    """The ratios are not invented: an artist tuned 50x100 pushed 25 back with a
    sky card at 300, on a plate whose median distance was 17.0 m."""
    d = np.full((64, 64), 17.0)
    d[0, 0] = 1.0          # a little spread so min_spread is satisfied
    d[-1, -1] = 60.0
    gw, gd, off, sky, median, _p99, _r = AtlasSceneScale().measure(
        _depth(d), min_spread=0.0)

    assert median == pytest.approx(17.0)
    assert gw == pytest.approx(51.0, abs=1.0)      # ~50
    assert gd == pytest.approx(102.0, abs=2.0)     # ~100
    assert off == pytest.approx(-25.5, abs=1.0)    # ~-25


def test_sizes_track_the_scene_rather_than_being_fixed():
    """The same 50x100 m ground is a wall in front of one subject and a postage
    stamp under another — so a 10x deeper scene must get a 10x bigger plane."""
    small = AtlasSceneScale().measure(_depth(_ramp(1.0, 8.0)))
    large = AtlasSceneScale().measure(_depth(_ramp(100.0, 800.0)))

    assert large[0] / small[0] == pytest.approx(100.0, rel=0.05)
    assert large[3] / small[3] == pytest.approx(100.0, rel=0.05)


def test_relative_depth_is_reported_and_still_used():
    """A relative model's metres are arbitrary but the RATIOS stay right, and a
    plane in the right proportion to a scene of unknown scale is exactly as
    useful as the scene is. Silently refusing it would throw that away."""
    out = AtlasSceneScale().measure(_depth(_ramp(), is_metric=False))

    assert out[0] > 0 and out[3] > 0
    assert "RELATIVE" in out[6] and "arbitrary" in out[6]
    assert "PROPORTIONS still hold" in out[6]


def test_a_flat_solve_is_refused_not_averaged():
    """Everything at one distance is a FAILED solve, and a median taken from it
    is a confident number describing nothing."""
    flat = np.full((64, 64), 12.0)
    with pytest.raises(ValueError, match="min_spread"):
        AtlasSceneScale().measure(_depth(flat))

    # ...and the refusal is overridable, deliberately: 0 = never refuse.
    out = AtlasSceneScale().measure(_depth(flat), min_spread=0.0)
    assert out[4] == pytest.approx(12.0)


def test_an_empty_depth_map_names_the_real_problem():
    with pytest.raises(ValueError, match="no finite positive samples"):
        AtlasSceneScale().measure(_depth(np.full((32, 32), np.nan)))


def test_non_positive_samples_are_dropped_not_counted():
    """0 and negative depths are 'no measurement here', not 'zero metres away'
    — counting them would drag the median toward the camera."""
    d = _ramp()
    poisoned = d.copy()
    poisoned[:8] = 0.0
    poisoned[8:12] = -3.0

    clean_median = AtlasSceneScale().measure(_depth(d[12:]))[4]
    got_median = AtlasSceneScale().measure(_depth(poisoned))[4]

    assert got_median == pytest.approx(clean_median, rel=0.02)
