"""AtlasPlaneMattes — the missing link between a fitted plane and a layer.

`core.plane_masks` computes which pixels a plane actually explains, and until
this node existed it had NO caller anywhere in atlas-camera: the module was
written for atlas-world's private roundtrip and nothing public ever ran it. The
visible consequence was a `.atlas` package with planes and zero layers, which
opens in the editor showing the relief mesh and nothing else — reported live.
"""
from __future__ import annotations

import base64
import math

import numpy as np
import pytest

from atlas_camera.comfy.nodes_geometry import AtlasPlaneMattes
from atlas_camera.inference.depth_estimator import DepthResult
from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasProjectionScene,
    AtlasProxyPrimitive,
    AtlasSolve,
    LatentCamera,
)

WIDTH, HEIGHT = 64, 48
FX = FY = 60.0


def _view_matrix():
    """Camera at the origin looking down -Z, which is the Atlas convention."""
    return [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]


def _plane(name, *, distance, width=1000.0, height=1000.0):
    """A fronto-parallel plane at `distance` in front of the camera.

    Columns are (u, v, n, c) — `plane_transform` writes the local axes as
    COLUMNS, and reading them as rows yields a plausible transposed basis that
    crops along the wrong axes.
    """
    return AtlasProxyPrimitive(
        name=name,
        primitive_type="plane",
        transform_matrix=[[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, -float(distance)],
                          [0.0, 0.0, 0.0, 1.0]],
        dimensions=(width, height),
        metadata={"role": "projection_proxy"},
    )


def _solve(primitives):
    camera = LatentCamera(
        intrinsics=AtlasIntrinsics(
            image_width=WIDTH, image_height=HEIGHT,
            fx_px=FX, fy_px=FY,
            cx_px=WIDTH / 2.0, cy_px=HEIGHT / 2.0,
            principal_point_px=(WIDTH / 2.0, HEIGHT / 2.0)),
        extrinsics=AtlasExtrinsics(camera_view_matrix=_view_matrix()),
    )
    return AtlasSolve(
        camera=camera,
        image_width=WIDTH,
        image_height=HEIGHT,
        projection_scene=AtlasProjectionScene(proxy_geometry=list(primitives)),
    )


def _depth_two_slabs():
    """Near slab on the left half, far slab on the right. Two planes explain it.

    An ATLAS_DEPTH_MAP is a DepthResult, not a bare array — it carries its own
    resolution, which is what the node falls back to when the solve and the
    depth estimate disagree about raster size.
    """
    depth = np.empty((HEIGHT, WIDTH), dtype=np.float32)
    depth[:, : WIDTH // 2] = 5.0
    depth[:, WIDTH // 2 :] = 20.0
    return DepthResult(depth=depth, is_metric=True, model_id="test",
                       image_width=WIDTH, image_height=HEIGHT,
                       near=5.0, far=20.0)


def _sources(solve):
    return list(getattr(solve, "projection_sources", None) or [])


def _decode(source):
    payload = source.mask_b64.split(",", 1)[-1]
    return base64.b64decode(payload)


def test_each_plane_gets_a_layer_carrying_its_own_matte():
    solve = _solve([_plane("near", distance=5.0), _plane("far", distance=20.0)])

    out, report = AtlasPlaneMattes().mattes(solve, _depth_two_slabs())

    sources = _sources(out)
    assert [s.name for s in sources] == ["near", "far"]
    for source in sources:
        assert source.mask_b64, "a layer without a matte explains nothing"
        assert _decode(source)[:8] == b"\x89PNG\r\n\x1a\n"
        # The source carries its OWN plane, so the editor has a surface to put
        # the matte on rather than a record pointing at nothing.
        assert len(source.proxy_geometry) == 1
    assert "2 plane" in report


def test_the_matte_is_exclusive_so_the_plate_lands_on_one_surface():
    """Two planes both claiming a pixel puts the photograph on two surfaces at
    different depths, which an orbit shows as a doubled sliding ghost."""

    solve = _solve([_plane("near", distance=5.0), _plane("far", distance=20.0)])

    out, _ = AtlasPlaneMattes().mattes(solve, _depth_two_slabs())

    from atlas_camera.comfy.node_helpers import _b64_png_to_mask

    masks = [_b64_png_to_mask(s.mask_b64) for s in _sources(out)]
    overlap = np.logical_and(*masks)
    assert not overlap.any(), "no pixel may belong to two planes"
    assert masks[0][:, : WIDTH // 2].all(), "the near plane owns the near slab"
    assert masks[1][:, WIDTH // 2 :].all(), "the far plane owns the far slab"


def test_the_farthest_plane_takes_the_highest_priority():
    """Seam doctrine: band priorities are FARTHEST-highest, so the near surface
    draws over the far one rather than being buried by it."""

    solve = _solve([_plane("near", distance=5.0), _plane("far", distance=20.0)])

    out, _ = AtlasPlaneMattes().mattes(solve, _depth_two_slabs())

    near, far = _sources(out)
    assert far.priority > near.priority


def test_a_plane_that_explains_nothing_is_reported_not_written():
    """A plane fitted against a different depth map than the one being assigned
    is a finding, not a crash — and a layer with an empty matte is a layer that
    mattes nothing, so it must not be written at all."""

    solve = _solve([_plane("real", distance=5.0), _plane("nowhere", distance=900.0)])

    out, report = AtlasPlaneMattes().mattes(solve, _depth_two_slabs())

    assert [s.name for s in _sources(out)] == ["real"]
    assert "nowhere" in report and "explain" in report.lower()


def test_running_twice_replaces_its_own_layers_rather_than_doubling_them():
    """Re-queue is the normal case in ComfyUI. Appending again would give the
    scene two layers per plane, each claiming the same pixels."""

    solve = _solve([_plane("near", distance=5.0), _plane("far", distance=20.0)])

    once, _ = AtlasPlaneMattes().mattes(solve, _depth_two_slabs())
    twice, _ = AtlasPlaneMattes().mattes(once, _depth_two_slabs())

    assert len(_sources(twice)) == len(_sources(once)) == 2


def test_layers_from_another_node_survive():
    """It replaces ITS OWN layers only. A patch view or a clean plate is another
    node's evidence and this node has no business dropping it."""

    solve = _solve([_plane("near", distance=5.0)])
    from atlas_camera.core.schema import ProjectionSource

    solve.projection_sources = [
        ProjectionSource(camera=solve.camera, name="patch_01", priority=9.0)
    ]

    out, _ = AtlasPlaneMattes().mattes(solve, _depth_two_slabs())

    names = [s.name for s in _sources(out)]
    assert "patch_01" in names and "near" in names


def test_a_solve_with_no_planes_says_so_rather_than_silently_doing_nothing():
    solve = _solve([])

    out, report = AtlasPlaneMattes().mattes(solve, _depth_two_slabs())

    assert not _sources(out)
    assert "no plane" in report.lower()


def test_the_source_declares_clean_plate_projection():
    """A source with geometry but no image of its own projects the PRIMARY
    plate, and must say `clean_plate`: it shares the primary's camera, so it has
    to paint grazing surfaces as well as head-on ones, and the default patch
    facing threshold would drop a ground plane out of its own projection almost
    everywhere."""

    solve = _solve([_plane("near", distance=5.0)])

    out, _ = AtlasPlaneMattes().mattes(solve, _depth_two_slabs())

    source = _sources(out)[0]
    assert source.metadata.get("projection_mode") == "clean_plate"
    assert source.image_b64 is None


def test_a_plane_layer_is_not_reported_as_an_empty_mesh():
    """`zero_vertex_layer` asks a MESH question of an analytic surface.

    n_vertices is summed off each geometry's metadata, and a plane has no mesh
    to count — so every layer AtlasPlaneMattes produced read as ZERO vertices
    and the scene-health gate FAILED a scene whose layers were all correct.
    Seen live the first time the node ran end to end: four plane layers, four
    mattes on disk, and five red flags.
    """
    from atlas_camera.core.scene_health import evaluate_scene_health

    solve = _solve([_plane("near", distance=5.0), _plane("far", distance=20.0)])
    out, _ = AtlasPlaneMattes().mattes(solve, _depth_two_slabs())

    health = evaluate_scene_health(out)

    offenders = [f for f in health.flags if f.code == "zero_vertex_layer"]
    assert not offenders, [f.message for f in offenders]
