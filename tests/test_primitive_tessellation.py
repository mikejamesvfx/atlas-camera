"""Solid proxy primitives must reach the headless rasterizer.

`AtlasBlockoutMassing` emits `primitive_type="box"`, and the viewport draws
boxes/cylinders/planes as projectable geometry (`atlas_blockout.js` builds the
THREE geometry and stamps `atlasDerived = true`). `gather_scene_meshes` used to
skip every non-"mesh" primitive, so AtlasDisocclusionGuide, the occlusion-fill
path and move-budget measurements were all blind to placeholder mass — a
measured 2x camera-envelope win was invisible to the pipeline that needed it.

The centring convention is the subtle part and has its own test: THREE's
BoxGeometry is centred, and `block_massing.box_transform` matches it by putting
`ground_y + 0.5 * height` in the transform. A unit cube spanning y in [0, 1]
would float every mass half its own height off the ground.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.projection_render import (  # noqa: E402
    _TESSELLATED_PRIMITIVES, gather_scene_meshes, tessellate_primitive)
from atlas_camera.core.schema import AtlasProxyPrimitive  # noqa: E402


def _prim(kind, dims=(2.0, 4.0, 6.0), translate=(0.0, 0.0, 0.0), **meta):
    m = [[1.0, 0.0, 0.0, translate[0]],
         [0.0, 1.0, 0.0, translate[1]],
         [0.0, 0.0, 1.0, translate[2]],
         [0.0, 0.0, 0.0, 1.0]]
    return AtlasProxyPrimitive(name=f"p_{kind}", primitive_type=kind,
                               transform_matrix=m, dimensions=dims,
                               metadata=meta or {})


def test_box_is_centred_on_its_transform_not_sitting_on_it():
    """The centring convention box_transform relies on."""
    verts, faces = tessellate_primitive(_prim("box", dims=(2.0, 4.0, 6.0)))
    assert len(verts) == 8 and len(faces) == 12
    assert np.allclose(verts.min(axis=0), [-1.0, -2.0, -3.0])
    assert np.allclose(verts.max(axis=0), [1.0, 2.0, 3.0])
    # centre of the box is the transform's translation
    assert np.allclose(verts.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-12)


def test_box_transform_puts_a_massing_box_on_the_ground():
    """End-to-end with the real massing transform: base sits at ground_y."""
    bm = pytest.importorskip("atlas_camera.core.block_massing")
    box = bm.MassingBox(u0=0.0, u1=10.0, v0=0.0, v1=20.0, height_m=30.0,
                        corners=(), zone="test")
    matrix, dims = bm.box_transform(box, azimuth_deg=0.0, ground_y=0.0)
    verts, _ = tessellate_primitive(
        AtlasProxyPrimitive(name="m", primitive_type="box",
                            transform_matrix=matrix, dimensions=dims))
    assert verts[:, 1].min() == pytest.approx(0.0, abs=1e-9)
    assert verts[:, 1].max() == pytest.approx(30.0, abs=1e-9)


def test_translation_is_applied():
    verts, _ = tessellate_primitive(
        _prim("box", dims=(2.0, 2.0, 2.0), translate=(5.0, 7.0, -3.0)))
    assert np.allclose(verts.mean(axis=0), [5.0, 7.0, -3.0], atol=1e-12)


@pytest.mark.parametrize("kind", _TESSELLATED_PRIMITIVES)
def test_every_allowlisted_primitive_tessellates(kind):
    out = tessellate_primitive(_prim(kind))
    assert out is not None, kind
    verts, faces = out
    assert len(verts) >= 3 and len(faces) >= 1
    assert faces.max() < len(verts)


@pytest.mark.parametrize("kind", ["height_guide", "axis_guide", "mesh", "wat"])
def test_guides_and_unknown_types_are_never_tessellated(kind):
    """Guides are UI furniture; they must never occlude a render."""
    assert tessellate_primitive(_prim(kind)) is None


def test_planes_are_excluded_because_they_are_coverage_cards():
    """projection_ground / backdrop / dynamic-plate receivers are planes whose
    job is to GUARANTEE coverage. Rasterising them headless would report zero
    disocclusion everywhere and blind the guide to the holes it exists to find.
    """
    assert "plane" not in _TESSELLATED_PRIMITIVES
    assert tessellate_primitive(_prim("plane")) is None


def test_gather_scene_meshes_now_sees_boxes(monkeypatch):
    """The regression this whole change exists to fix."""
    from atlas_camera.core.schema import (AtlasCamera, AtlasIntrinsics,
                                          AtlasProjectionScene, AtlasSolve)
    camera = AtlasCamera(intrinsics=AtlasIntrinsics(
        image_width=100, image_height=100, fx_px=100.0, fy_px=100.0,
        cx_px=50.0, cy_px=50.0))
    solve = AtlasSolve(camera=camera)
    solve.projection_scene = AtlasProjectionScene(
        proxy_geometry=[_prim("box"), _prim("height_guide")])

    meshes = gather_scene_meshes(solve, with_uvs=True)
    labels = [m[0] for m in meshes]
    assert any("p_box" in a for a in labels), labels
    assert not any("guide" in a for a in labels), labels
    # UVs must exist or render_scene silently skips the mesh
    box = next(m for m in meshes if "p_box" in m[0])
    uvs = box[3]
    assert uvs is not None and len(uvs) == len(box[1])
    assert box[5]["tessellated_from"] == "box"
