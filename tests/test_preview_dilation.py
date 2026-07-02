"""Tests for preview-only geometry dilation (blockout viewport orbit coverage).

For any point p with local normal n̂, radiating from a pivot:
    p' = pivot + ((p-pivot)·n̂)n̂ + scale * [(p-pivot) - ((p-pivot)·n̂)n̂]
Only the component perpendicular to n̂ scales; the normal-aligned component
(depth from pivot) is preserved. Verified: closed-form on planes/volumes, and
that the per-vertex mesh formula reduces to the SAME closed-form as the plane
case on a flat quad (a plane is a mesh where every vertex shares one normal).
"""

import numpy as np
import pytest

from atlas_camera.core.proxy_geometry import (
    dilate_proxy_geometry_for_preview,
    serialize_proxy_geometry,
)
from atlas_camera.core.schema import AtlasProjectionScene, AtlasProxyPrimitive

# proxy_geometry.py's plane-matrix builder is private (_plane_transform);
# import via the module for test fixture construction.
from atlas_camera.core import proxy_geometry as pg


def _plane_prim(center, u, v, n, w, h, name="p"):
    return AtlasProxyPrimitive(
        name=name, primitive_type="plane",
        transform_matrix=pg._plane_transform(np.array(u), np.array(v), np.array(n), np.array(center)),
        dimensions=(w, h, 0.0), material="atlas_projection_proxy",
        metadata={"role": pg.PROXY_ROLE, "source": "test"},
    )


def test_plane_dilation_preserves_normal_depth_and_scales_extent():
    pivot = np.array([0.0, 1.6, 0.0])
    center = np.array([0.0, 1.5, -10.0])  # a wall facing the camera
    n = np.array([0.0, 0.0, 1.0])
    u, v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    prim = _plane_prim(center, u, v, n, 4.0, 3.0)

    scale = 2.0
    out = dilate_proxy_geometry_for_preview([prim], pivot=pivot, scale=scale)[0]

    # Dimensions scale exactly.
    assert out.dimensions[0] == pytest.approx(8.0)
    assert out.dimensions[1] == pytest.approx(6.0)

    # Depth from pivot along the normal is UNCHANGED (plane doesn't drift
    # toward/away from the camera).
    new_center = np.array(out.transform_matrix)[:3, 3]
    old_depth = float(np.dot(center - pivot, n))
    new_depth = float(np.dot(new_center - pivot, n))
    assert new_depth == pytest.approx(old_depth, abs=1e-9)

    # Tangential offset from pivot scales by `scale`.
    old_tang = (center - pivot) - old_depth * n
    new_tang = (new_center - pivot) - new_depth * n
    assert np.allclose(new_tang, scale * old_tang)


def test_scale_one_is_a_no_op():
    pivot = np.array([0.0, 1.6, 0.0])
    prim = _plane_prim([0, 1.5, -10], [1, 0, 0], [0, 1, 0], [0, 0, 1], 4.0, 3.0)
    out = dilate_proxy_geometry_for_preview([prim], pivot=pivot, scale=1.0)
    assert out == [prim]  # returns the original list unchanged (identity op)


def test_volume_dilation_scales_uniformly_from_pivot():
    pivot = np.array([0.0, 0.0, 0.0])
    center = np.array([2.0, 0.0, -6.0])
    box = AtlasProxyPrimitive(
        name="box", primitive_type="box",
        transform_matrix=pg._plane_transform(
            np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), center),
        dimensions=(1.0, 2.0, 1.5), material="atlas_projection_proxy",
        metadata={"role": pg.PROXY_ROLE, "source": "test"},
    )
    scale = 1.5
    out = dilate_proxy_geometry_for_preview([box], pivot=pivot, scale=scale)[0]
    new_center = np.array(out.transform_matrix)[:3, 3]
    assert np.allclose(new_center, scale * center)  # pure radial scale (pivot at origin)
    assert out.dimensions == pytest.approx((1.5, 3.0, 2.25))


def test_mesh_dilation_agrees_with_plane_closed_form_on_a_flat_quad():
    # A flat quad (two triangles) all sharing the SAME normal must dilate
    # identically to the closed-form plane result at each corner.
    pivot = np.array([0.5, 1.0, 2.0])
    n = np.array([0.0, 0.0, 1.0])
    corners = np.array([
        [-2.0, -1.0, -10.0], [2.0, -1.0, -10.0],
        [2.0, 1.0, -10.0], [-2.0, 1.0, -10.0],
    ])
    faces = [0, 1, 2, 0, 2, 3]
    mesh_prim = AtlasProxyPrimitive(
        name="mesh", primitive_type="mesh",
        dimensions=(0.0, 0.0, 0.0), material="atlas_projection_proxy",
        metadata={
            "role": pg.PROXY_ROLE, "source": "test",
            "vertices": [round(float(x), 6) for x in corners.reshape(-1)],
            "faces": faces,
            "uvs": [0, 0] * 4,
        },
    )
    scale = 1.7
    out = dilate_proxy_geometry_for_preview([mesh_prim], pivot=pivot, scale=scale)[0]
    new_verts = np.array(out.metadata["vertices"]).reshape(-1, 3)

    for i, corner in enumerate(corners):
        d = corner - pivot
        d_n = float(np.dot(d, n)) * n
        d_t = d - d_n
        expected = pivot + d_n + scale * d_t
        assert np.allclose(new_verts[i], expected, atol=1e-4), f"corner {i}"


def test_serialize_proxy_geometry_applies_preview_expand():
    scene = AtlasProjectionScene()
    scene.proxy_geometry.append(_plane_prim([0, 1.5, -10], [1, 0, 0], [0, 1, 0], [0, 0, 1], 4.0, 3.0))
    pivot = [0.0, 1.6, 0.0]

    plain = serialize_proxy_geometry(scene)
    expanded = serialize_proxy_geometry(scene, preview_expand=2.0, preview_pivot=pivot)

    assert plain[0]["dimensions"][0] == pytest.approx(4.0)
    assert expanded[0]["dimensions"][0] == pytest.approx(8.0)
    assert expanded[0]["metadata"]["preview_dilated"] is True

    # Underlying solve data is untouched — export/measurement geometry unaffected.
    assert scene.proxy_geometry[0].dimensions[0] == pytest.approx(4.0)


def test_serialize_without_pivot_or_scale_is_unchanged():
    scene = AtlasProjectionScene()
    scene.proxy_geometry.append(_plane_prim([0, 1.5, -10], [1, 0, 0], [0, 1, 0], [0, 0, 1], 4.0, 3.0))
    # preview_expand > 1 but no pivot supplied -> no dilation (matches
    # _extract_blockout_camera always passing a pivot; this guards the default).
    out = serialize_proxy_geometry(scene, preview_expand=2.0, preview_pivot=None)
    assert out[0]["dimensions"][0] == pytest.approx(4.0)
