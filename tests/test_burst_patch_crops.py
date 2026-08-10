"""Contract for burst patch crops — "which photograph saw into this hole".

The middle-anchor case: a burst walks past a scene, the middle frame is the shot,
and the frames either side stood somewhere else and photographed the surfaces the
anchor could not see. `rank_burst_frames` ranks those ALREADY-SOLVED cameras, and
`AtlasSolveBurstPatchCrops` returns the crop out of the winning photograph.

The fixture is the same occluded-slab geometry tests/test_path_hole_repair.py and
tests/test_view_solver.py use, so all three agree on what an island is. The
flanking cameras are built with `orbit_camera` — the same helper `rank_views`
places its candidates with — which is what makes "this angle sees the hole" mean
the same thing in both solvers.

These tests are written against the REAL schema. An earlier draft invented
`solve.burst_cameras` and `solve.source_image`, neither of which exists on an
AtlasSolve: the node passed its tests and returned a black patch in ComfyUI,
because the assertions only ever exercised an early return. Per-frame cameras
live on `solve.projection_sources[i].camera`, and their pixels on `.plate_ref`.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas_camera.core.camera_math import orbit_camera  # noqa: E402
from atlas_camera.core.path_hole_repair import PathHoleRepairConfig  # noqa: E402
from atlas_camera.core.schema import (  # noqa: E402
    AtlasPlateRef,
    AtlasProjectionScene,
    AtlasSolve,
    LatentCamera,
    ProjectionSource,
)
from atlas_camera.core.view_solver import (  # noqa: E402
    BurstFrameScore,
    rank_burst_frames,
)


FIT = PathHoleRepairConfig(
    normal_tolerance_deg=15.0, max_plane_error_m=0.02, max_hole_fraction=0.20,
)


@pytest.fixture(scope="module")
def scene():
    from test_path_hole_repair import _fixture

    mesh, camera, _path = _fixture()
    hole = np.asarray(mesh.hole_mask, dtype=bool)
    assert hole.any(), "fixture produced no holes — every test would be vacuous"
    return mesh, camera, hole


def _flank(camera, d_azimuth_deg: float) -> LatentCamera:
    """A camera standing beside the anchor, aimed at the same subject."""
    return LatentCamera(
        intrinsics=camera.intrinsics,
        extrinsics=orbit_camera(
            camera.extrinsics, (0.0, 0.0, -5.0),
            d_azimuth_deg=float(d_azimuth_deg), d_elevation_deg=0.0,
        ),
    )


def _preview_b64(pixels) -> str:
    """A browser-preview data URI, the way the burst solve writes one."""
    Image = pytest.importorskip("PIL.Image")
    arr = (np.clip(pixels, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _photographed_source(camera, name: str, frame_index: int, pixels=None):
    intr = camera.intrinsics
    if pixels is None:
        # A deliberately non-uniform plate: a crop of a flat grey would pass an
        # "is it black?" assertion no matter which region it came from.
        ramp = np.linspace(0.0, 1.0, int(intr.image_width), dtype=np.float32)
        pixels = np.repeat(ramp[None, :, None], int(intr.image_height), axis=0)
        pixels = np.repeat(pixels, 3, axis=2)
    return ProjectionSource(
        camera=camera,
        name=name,
        image_b64=None,
        plate_ref=AtlasPlateRef(
            image_path=None,
            preview_b64=_preview_b64(pixels),
            role="source",
            is_proxy=False,
        ),
        metadata={"evidence_type": "photographed", "frame_index": frame_index},
    )


def _solve_with(mesh, camera, sources) -> AtlasSolve:
    from atlas_camera.core.proxy_geometry import relief_mesh_primitive

    return AtlasSolve(
        camera=camera,
        projection_scene=AtlasProjectionScene(
            proxy_geometry=[relief_mesh_primitive(mesh)],
        ),
        projection_sources=list(sources),
    )


class TestRankBurstFrames:
    def test_each_camera_is_rasterized_in_its_own_frame(self, scene):
        """Every camera gets its OWN visibility and its OWN ROI.

        The failure this guards is the cheap one: ranking with the source
        camera's pose or intrinsics for every entry, which returns identical
        scores and an ROI that means nothing in the frame it names. Cameras
        standing at ±40° must disagree with the anchor and with each other.
        """
        mesh, camera, hole = scene
        cams = [camera, _flank(camera, 40.0), _flank(camera, -40.0)]
        scores = rank_burst_frames(
            mesh, hole, source_camera=camera, burst_cameras=cams,
            resolution=192, config=FIT,
        )
        assert len(scores) == 3
        assert {type(s) for s in scores} == {BurstFrameScore}
        by_index = {s.frame_index: s for s in scores}
        assert all(s.visible_px > 0 for s in scores)
        assert by_index[1].visible_px != by_index[0].visible_px
        assert by_index[2].visible_px != by_index[0].visible_px
        assert by_index[1].crop_roi != by_index[0].crop_roi
        assert by_index[2].crop_roi != by_index[0].crop_roi
        # The ±40° flanks are mirror images of each other about a centred
        # fixture, so THEY may legitimately agree — only the anchor must differ.

    def test_asymmetric_cameras_do_not_share_a_score(self, scene):
        mesh, camera, hole = scene
        scores = rank_burst_frames(
            mesh, hole, source_camera=camera, resolution=192, config=FIT,
            burst_cameras=[_flank(camera, 15.0), _flank(camera, 55.0)],
        )
        assert len({s.visible_px for s in scores}) == 2

    def test_results_are_sorted_best_first(self, scene):
        mesh, camera, hole = scene
        scores = rank_burst_frames(
            mesh, hole, source_camera=camera, resolution=192, config=FIT,
            burst_cameras=[camera, _flank(camera, 25.0), _flank(camera, 50.0)],
        )
        assert [s.visible_px for s in scores] == sorted(
            (s.visible_px for s in scores), reverse=True)

    def test_crop_roi_bounds_the_visible_geometry_not_the_whole_frame(self, scene):
        mesh, camera, hole = scene
        scores = rank_burst_frames(
            mesh, hole, source_camera=camera, resolution=192, config=FIT,
            burst_cameras=[_flank(camera, 40.0)], margin_px=4,
        )
        best = scores[0]
        assert best.visible_px > 0
        u_min, v_min, u_max, v_max = best.crop_roi
        intr = camera.intrinsics
        assert 0 <= u_min < u_max <= int(intr.image_width)
        assert 0 <= v_min < v_max <= int(intr.image_height)
        area = (u_max - u_min) * (v_max - v_min)
        assert area < int(intr.image_width) * int(intr.image_height)

    def test_a_frame_scoring_zero_reports_no_narrowed_crop(self, scene):
        """A sub-threshold speckle must not hand back a confident-looking box.

        Boxing every non-zero id — rather than only the islands that survived
        min_visible_pixels — lets a frame report visible_px = 0 alongside a tight
        ROI, which downstream reads as "here is your patch".
        """
        mesh, camera, hole = scene
        scores = rank_burst_frames(
            mesh, hole, source_camera=camera, resolution=192, config=FIT,
            burst_cameras=[_flank(camera, 40.0)], min_visible_pixels=10 ** 6,
        )
        intr = camera.intrinsics
        assert scores[0].visible_px == 0
        assert scores[0].crop_roi == (0, 0, int(intr.image_width), int(intr.image_height))

    def test_duplicate_camera_objects_do_not_collapse(self, scene):
        """Cameras repeat in a burst; an id()-keyed tie-break silently loses one."""
        mesh, camera, hole = scene
        flank = _flank(camera, 40.0)
        scores = rank_burst_frames(
            mesh, hole, source_camera=camera, resolution=192, config=FIT,
            burst_cameras=[flank, flank, camera],
        )
        assert len(scores) == 3
        assert sorted(s.frame_index for s in scores) == [0, 1, 2]


class TestPatchCropNode:
    def _node(self):
        from atlas_camera.comfy.nodes_multiview import AtlasSolveBurstPatchCrops

        return AtlasSolveBurstPatchCrops()

    def _mask(self, hole):
        torch = pytest.importorskip("torch")
        return torch.from_numpy(hole.astype(np.float32))[None, ...]

    def test_it_crops_the_winning_photograph_not_the_anchor(self, scene):
        mesh, camera, hole = scene
        sources = [
            _photographed_source(_flank(camera, 40.0), "photo_2", 1),
            _photographed_source(_flank(camera, -40.0), "photo_3", 2),
        ]
        solve = _solve_with(mesh, camera, sources)

        patch, frame_index, roi, report = self._node().solve_crops(
            solve, self._mask(hole), margin_px=4, resolution=192,
        )

        assert frame_index in (1, 2), "must name a registered flanking frame"
        u_min, v_min, u_max, v_max = (int(v) for v in roi.split(","))
        assert (u_max - u_min, v_max - v_min) == (patch.shape[2], patch.shape[1])
        assert float(patch.max()) > 0.0, "a black patch means no plate was read"
        # The plate is a horizontal ramp, so a crop that really came from the ROI
        # carries the ramp values at those columns — not the full 0..1 sweep a
        # whole-frame or wrong-frame crop would show.
        assert float(patch.max()) - float(patch.min()) < 0.999
        assert "photo_" in report and "px of hole geometry" in report

    def test_the_anchor_is_never_offered_as_its_own_patch(self, scene):
        """Refuse honestly rather than return the plate that owns the hole."""
        mesh, camera, hole = scene
        solve = _solve_with(mesh, camera, [])
        patch, frame_index, _roi, report = self._node().solve_crops(
            solve, self._mask(hole), resolution=192,
        )
        assert frame_index == -1
        assert float(patch.abs().sum()) == 0.0
        assert "no registered flanking photographs" in report

    def test_no_relief_mesh_is_reported_not_guessed(self, scene):
        _mesh, camera, hole = scene
        solve = AtlasSolve(camera=camera, projection_scene=AtlasProjectionScene())
        _patch, frame_index, _roi, report = self._node().solve_crops(
            solve, self._mask(hole), resolution=192,
        )
        assert frame_index == -1
        assert "AtlasDeriveReliefMesh" in report

    def test_an_unreadable_plate_refuses_instead_of_returning_black(self, scene):
        mesh, camera, hole = scene
        blind = ProjectionSource(
            camera=_flank(camera, 40.0),
            name="photo_2",
            plate_ref=AtlasPlateRef(image_path=None, preview_b64=None),
            metadata={"evidence_type": "photographed", "frame_index": 1},
        )
        solve = _solve_with(mesh, camera, [blind])
        _patch, frame_index, _roi, report = self._node().solve_crops(
            solve, self._mask(hole), resolution=192,
        )
        assert frame_index == -1
        assert "could not be read" in report

    def test_a_mask_at_the_wrong_resolution_is_resampled(self, scene):
        mesh, camera, hole = scene
        torch = pytest.importorskip("torch")
        half = hole[::2, ::2].astype(np.float32)
        solve = _solve_with(
            mesh, camera, [_photographed_source(_flank(camera, 40.0), "photo_2", 1)],
        )
        _patch, frame_index, _roi, report = self._node().solve_crops(
            solve, torch.from_numpy(half)[None, ...], resolution=192,
        )
        assert frame_index == 1, report
