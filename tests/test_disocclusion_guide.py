"""Contract for AtlasDisocclusionGuide 🟣.

The node renders a move and marks pixels no projection mesh covers, as
conditioning for a generative filler. See docs/dev/crossviewwarp_analysis.md for
where the sentinel-colour convention comes from.

The subtle thing pinned here is the distinction the node exists to keep honest:
an uncovered pixel is NOT necessarily a disocclusion. It is either geometry the
move revealed (invent it) or geometry that was never derived (fix it upstream).
Found live — a FLAT depth map with zero occlusion still renders ~50% uncovered
on a partial relief mesh, so a report that called all of it "disocclusion" was
wrong on the most common case.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from atlas_camera.comfy.node_registry import NODE_CLASS_MAPPINGS  # noqa: E402

GUIDE = NODE_CLASS_MAPPINGS["AtlasDisocclusionGuide"]
W = H = 96


def _solve(step: bool = True):
    """A solve carrying a real relief mesh; `step` adds a 4x depth cliff."""
    from atlas_camera.comfy.nodes_geometry import AtlasDeriveReliefMesh
    from atlas_camera.core.intrinsics import build_intrinsics
    from atlas_camera.core.schema import AtlasCamera, AtlasExtrinsics, AtlasSolve
    from atlas_camera.inference.depth_estimator import DepthResult

    d = np.full((H, W), 12.0, dtype=np.float32)
    if step:
        d[30:70, 30:70] = 3.0
    depth = DepthResult(depth=d, is_metric=True, model_id="t",
                        image_width=W, image_height=H)
    intr = build_intrinsics(image_width=W, image_height=H,
                            focal_length_mm=35.0, sensor_width_mm=36.0)
    cam = AtlasCamera(intrinsics=intr, extrinsics=AtlasExtrinsics(
        camera_position=(0.0, 0.0, 0.0),
        camera_world_matrix=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))))
    base = AtlasSolve(camera=cam, image_width=W, image_height=H)
    return AtlasDeriveReliefMesh().derive(base, depth, relief_grid=96,
                                          depth_edge_rel=0.5)[0]


def _path(dx: float = 2.0, frames: int = 5):
    from atlas_camera.core.schema import AtlasCameraKeyframe, AtlasCameraPath
    return AtlasCameraPath(keyframes=[
        AtlasCameraKeyframe(frame_index=0, position=(0.0, 0.0, 0.0),
                            target=(0.0, 0.0, -10.0)),
        AtlasCameraKeyframe(frame_index=frames - 1, position=(dx, 0.5, 0.0),
                            target=(0.0, 0.0, -10.0)),
    ], frame_count=frames)


def _img():
    # Deterministic, and mid-grey so it can never coincide with a sentinel.
    return torch.full((1, H, W, 3), 0.5, dtype=torch.float32)


class TestSentinelMarking:
    def test_marked_pixels_are_exactly_the_sentinel_colour(self):
        guide, mask, _ = GUIDE().guide(_solve(), _img(), resolution=256)
        rgb = guide[0].numpy()
        is_magenta = np.abs(rgb - np.array([1.0, 0.0, 1.0])).max(-1) < 1e-6
        assert is_magenta.sum() > 0, "fixture produced no holes — test is vacuous"
        np.testing.assert_array_equal(is_magenta, mask[0].numpy() > 0.5)

    def test_the_mask_is_returned_separately_so_no_colour_keying_is_needed(self):
        guide, mask, _ = GUIDE().guide(_solve(), _img(), resolution=256)
        assert mask.shape == (guide.shape[0], guide.shape[1], guide.shape[2])

    @pytest.mark.parametrize("name,rgb", [
        ("magenta", (1.0, 0.0, 1.0)),
        ("chroma_green", (0.0, 1.0, 0.0)),
        ("black", (0.0, 0.0, 0.0)),
    ])
    def test_each_sentinel_paints_its_own_colour(self, name, rgb):
        guide, mask, _ = GUIDE().guide(_solve(), _img(), sentinel=name,
                                       resolution=256)
        painted = guide[0].numpy()[mask[0].numpy() > 0.5]
        assert painted.size, "no holes to check"
        np.testing.assert_allclose(painted, np.tile(rgb, (len(painted), 1)),
                                   atol=1e-6)

    def test_sentinel_values_are_append_only(self):
        """Combo values serialize into saved workflows — never reorder or rename.

        The ORDER matters as much as membership: ComfyUI stores the chosen value
        by string, but a reordered list changes which one a fresh node defaults
        to in the UI.
        """
        assert list(GUIDE.SENTINELS) == ["magenta", "chroma_green", "black"]
        assert GUIDE.INPUT_TYPES()["optional"]["sentinel"][0] == \
            ["magenta", "chroma_green", "black"]


class TestUncoveredIsNotDisocclusion:
    """The distinction the node exists to keep honest."""

    def test_flat_depth_still_reports_uncovered_frame(self):
        """Zero occlusion, yet not fully covered — so "uncovered" alone is not
        evidence of disocclusion, and the report must not call it that."""
        _, mask, report = GUIDE().guide(_solve(step=False), _img(), resolution=256)
        assert float(mask.mean()) > 0.05, "fixture no longer partial — revisit"
        assert "never covered at all" in report

    def test_the_report_splits_move_revealed_from_never_covered(self):
        _, _, report = GUIDE().guide(_solve(), _img(), camera_path=_path(),
                                     resolution=256)
        assert "the MOVE revealed" in report
        assert "never covered at all" in report
        assert "the solved camera itself covers" in report

    def test_incomplete_geometry_is_called_an_upstream_problem(self):
        _, _, report = GUIDE().guide(_solve(), _img(), resolution=256)
        assert "projection geometry is incomplete" in report, (
            "a filler cannot fix missing geometry; the report must say so "
            "rather than implying more inventing is the answer")

    def test_a_bigger_move_reveals_more(self):
        near = GUIDE().guide(_solve(), _img(), camera_path=_path(dx=0.2),
                             resolution=256)[1]
        far = GUIDE().guide(_solve(), _img(), camera_path=_path(dx=4.0),
                            resolution=256)[1]
        assert float(far.mean()) > float(near.mean())


class TestCameraPath:
    def test_a_path_produces_one_frame_per_sample(self):
        guide, mask, report = GUIDE().guide(_solve(), _img(), camera_path=_path(frames=5),
                                            resolution=256)
        assert guide.shape[0] == 5 and mask.shape[0] == 5
        assert "camera path (5 frames)" in report

    def test_without_a_path_it_renders_the_solved_camera_once(self):
        guide, _, report = GUIDE().guide(_solve(), _img(), resolution=256)
        assert guide.shape[0] == 1
        assert "no path connected" in report


class TestDilation:
    def test_dilation_only_ever_grows_the_marked_region(self):
        """It must never be able to claim coverage it does not have."""
        base = GUIDE().guide(_solve(), _img(), hole_dilate_px=0,
                             resolution=256)[1][0].numpy() > 0.5
        for r in (1, 3, 6):
            grown = GUIDE().guide(_solve(), _img(), hole_dilate_px=r,
                                  resolution=256)[1][0].numpy() > 0.5
            assert grown[base].all(), f"dilate {r}px un-marked a hole pixel"
            assert grown.sum() >= base.sum()

    def test_dilation_does_not_wrap_around_the_frame_edge(self):
        """np.roll wraps; rolling the UNPADDED mask leaks holes from one edge to
        the opposite one, which would show up as phantom disocclusion along a
        border that is actually fully covered."""
        mask = np.zeros((16, 16), dtype=bool)
        mask[0, 0] = True
        out = GUIDE._dilate(mask, 2, np)
        assert out[0:3, 0:3].sum() == 9          # the corner grew
        assert not out[-3:, :].any(), "wrapped to the bottom edge"
        assert not out[:, -3:].any(), "wrapped to the right edge"
        assert out.sum() == 9

    def test_zero_radius_is_the_identity(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[3, 3] = True
        np.testing.assert_array_equal(GUIDE._dilate(mask, 0, np), mask)

    def test_reported_coverage_ignores_dilation(self):
        """Dilation is a hint for the consumer. If it moved the coverage number,
        a cosmetic widget would silently change the measurement."""
        import re
        pct = lambda r: re.search(  # noqa: E731
            r"worst frame ([\d.]+)%",
            GUIDE().guide(_solve(), _img(), hole_dilate_px=r, resolution=256)[2]).group(1)
        assert pct(0) == pct(6)


class TestNoGeometry:
    def test_a_solve_with_no_meshes_marks_nothing_and_says_so(self):
        """Returning a FULLY marked frame would send a downstream model off
        inventing an entire image from nothing."""
        from atlas_camera.core.intrinsics import build_intrinsics
        from atlas_camera.core.schema import AtlasCamera, AtlasExtrinsics, AtlasSolve
        intr = build_intrinsics(image_width=W, image_height=H,
                                focal_length_mm=35.0, sensor_width_mm=36.0)
        bare = AtlasSolve(camera=AtlasCamera(
            intrinsics=intr,
            extrinsics=AtlasExtrinsics(
                camera_position=(0.0, 0.0, 0.0),
                camera_world_matrix=((1, 0, 0, 0), (0, 1, 0, 0),
                                     (0, 0, 1, 0), (0, 0, 0, 1)))),
            image_width=W, image_height=H)
        img = _img()
        guide, mask, report = GUIDE().guide(bare, img)
        assert float(mask.sum()) == 0.0
        assert torch.equal(guide, img), "source must pass through untouched"
        assert "no serialized projection meshes" in report
        assert "do not read the empty hole_mask as 'fully covered'" in report


class TestAgreesWithTheEstablishedRenderer:
    def test_coverage_matches_atlas_stereo_render(self):
        """Both read alpha from the same z-buffered render_scene, so a drift
        here means one of them grew its own private notion of coverage."""
        import re
        solve, img = _solve(), _img()
        stereo_report = NODE_CLASS_MAPPINGS["AtlasStereoRender"]().render(
            solve, img, interocular_m=0.001, resolution=256)[-1]
        stereo_cov = float(re.search(r"coverage L ([\d.]+)%", stereo_report).group(1))
        guide_report = GUIDE().guide(solve, img, resolution=256)[2]
        guide_cov = float(re.search(r"worst frame ([\d.]+)%", guide_report).group(1))
        assert abs(stereo_cov - guide_cov) < 0.15
