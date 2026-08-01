"""The patch-view registration maths, outside the node that used to own it.

`AtlasAddPatchView.add` carried ~118 lines of host-agnostic geometry: a forward
splat rasterizer, a pinhole projection, and the closed-form inversion of the
affine-in-s depth model z(s) = z_cam + s*(z_p - z_cam) with a robust median and
a 500-sample support floor. None of it mentioned torch or a ComfyUI type, and
the only interface it had was a node class — so the one assertion covering it
was `meta["scale_source"] == "primary_registration"`, and the derivation itself
was never exercised.
"""
import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.patch_registration import (  # noqa: E402
    solve_scale_from_primary,
    splat_coverage,
)

W = H = 64
FX = FY = 80.0
CX = CY = 32.0
IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _camera(**over):
    spec = dict(view_matrix=IDENTITY, fx=FX, fy=FY, cx=CX, cy=CY,
                width=W, height=H)
    spec.update(over)
    return spec


# ------------------------------------------------------------------- scale

def test_a_known_scale_is_recovered_exactly():
    """Patch and primary share a pose, so z_cam is 0 and s collapses to the
    ratio of the metric map to the patch depth. A relative depth that is 4x
    too small must report scale 4."""
    patch_depth = np.full((H, W), 2.5, dtype=np.float64)
    primary_metric = np.full((H, W), 10.0, dtype=np.float64)

    scale, info = solve_scale_from_primary(
        patch_depth,
        patch_camera=_camera(),
        patch_camera_position=(0.0, 0.0, 0.0),
        primary_metric_map=primary_metric,
        primary_camera=_camera(),
    )
    assert scale == pytest.approx(4.0, rel=1e-9)
    assert info["n_samples"] >= 500
    assert info["accepted"] is True


def test_too_few_samples_refuses_rather_than_guessing():
    """The 500-sample floor is the difference between a measured registration
    and a median of noise."""
    patch_depth = np.full((H, W), 2.5, dtype=np.float64)
    primary_metric = np.full((H, W), np.nan, dtype=np.float64)
    primary_metric[:4, :4] = 10.0                      # 16 usable samples

    scale, info = solve_scale_from_primary(
        patch_depth,
        patch_camera=_camera(),
        patch_camera_position=(0.0, 0.0, 0.0),
        primary_metric_map=primary_metric,
        primary_camera=_camera(),
    )
    assert scale is None
    assert info["accepted"] is False
    assert info["n_samples"] < 500


def test_the_sample_floor_is_tunable_and_then_it_accepts():
    patch_depth = np.full((H, W), 2.5, dtype=np.float64)
    primary_metric = np.full((H, W), np.nan, dtype=np.float64)
    primary_metric[:8, :8] = 10.0

    scale, info = solve_scale_from_primary(
        patch_depth,
        patch_camera=_camera(),
        patch_camera_position=(0.0, 0.0, 0.0),
        primary_metric_map=primary_metric,
        primary_camera=_camera(),
        min_samples=16,
    )
    assert scale == pytest.approx(4.0, rel=1e-9)
    assert info["n_samples"] == 64


def test_excluded_pixels_do_not_vote():
    """Sky is noise for registration. Poisoning every non-excluded pixel with a
    different ratio proves the exclusion is actually applied."""
    patch_depth = np.full((H, W), 2.5, dtype=np.float64)
    primary_metric = np.full((H, W), 25.0, dtype=np.float64)   # ratio 10
    primary_metric[:32, :] = 10.0                               # ratio 4
    exclude = np.zeros((H, W), dtype=bool)
    exclude[32:, :] = True                                      # drop ratio 10

    scale, _info = solve_scale_from_primary(
        patch_depth,
        patch_camera=_camera(),
        patch_camera_position=(0.0, 0.0, 0.0),
        primary_metric_map=primary_metric,
        primary_camera=_camera(),
        exclude_mask=exclude,
    )
    assert scale == pytest.approx(4.0, rel=1e-9)


def test_absurd_ratios_are_rejected_before_the_median():
    """s outside (1e-3, 1e3) is not a scale, it is a broken sample."""
    patch_depth = np.full((H, W), 2.5, dtype=np.float64)
    primary_metric = np.full((H, W), 10.0, dtype=np.float64)
    primary_metric[:2, :] = 1e9                       # s = 4e8, must not count

    scale, info = solve_scale_from_primary(
        patch_depth,
        patch_camera=_camera(),
        patch_camera_position=(0.0, 0.0, 0.0),
        primary_metric_map=primary_metric,
        primary_camera=_camera(),
    )
    assert scale == pytest.approx(4.0, rel=1e-9)
    assert info["n_samples"] == H * W - 2 * W


def test_a_median_ignores_a_minority_of_outliers():
    patch_depth = np.full((H, W), 2.5, dtype=np.float64)
    primary_metric = np.full((H, W), 10.0, dtype=np.float64)
    primary_metric[:20, :] = 30.0                     # 31% of pixels say 12
    scale, _ = solve_scale_from_primary(
        patch_depth,
        patch_camera=_camera(),
        patch_camera_position=(0.0, 0.0, 0.0),
        primary_metric_map=primary_metric,
        primary_camera=_camera(),
    )
    assert scale == pytest.approx(4.0, rel=1e-9)


# ---------------------------------------------------------------- coverage

def test_points_in_front_of_the_camera_mark_coverage():
    pts = np.array([[0.0, 0.0, -10.0]], dtype=np.float64)
    cover = splat_coverage(pts, camera=_camera(), close_px=0)
    assert cover.shape == (H, W)
    assert cover[int(CY), int(CX)]
    assert cover.sum() == 1


def test_points_behind_the_camera_are_dropped():
    pts = np.array([[0.0, 0.0, 10.0]], dtype=np.float64)
    assert not splat_coverage(pts, camera=_camera(), close_px=0).any()


def test_points_outside_the_frame_are_dropped():
    pts = np.array([[1000.0, 0.0, -10.0]], dtype=np.float64)
    assert not splat_coverage(pts, camera=_camera(), close_px=0).any()


def test_closing_fills_the_gap_between_sparse_samples():
    """Splat sparsity undercounts 'seen', and an undercounted coverage lets an
    AI patch overwrite real pixels — so the closing is load-bearing."""
    pts = np.array([[0.0, 0.0, -10.0], [0.5, 0.0, -10.0]], dtype=np.float64)
    open_cover = splat_coverage(pts, camera=_camera(), close_px=0)
    closed = splat_coverage(pts, camera=_camera(), close_px=3)
    assert closed.sum() > open_cover.sum()
    assert closed[int(CY), int(CX)]


def test_an_empty_point_set_covers_nothing():
    pts = np.zeros((0, 3), dtype=np.float64)
    assert not splat_coverage(pts, camera=_camera(), close_px=2).any()
