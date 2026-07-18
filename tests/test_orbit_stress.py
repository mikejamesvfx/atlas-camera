"""tools/orbit_stress_test.py — headless orbit coverage/stretch scoring."""

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasProxyPrimitive,
    AtlasSolve,
    LatentCamera,
)

_SPEC = importlib.util.spec_from_file_location(
    "orbit_stress_test",
    Path(__file__).resolve().parents[1] / "tools" / "orbit_stress_test.py")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def _wall_solve(stretched=False):
    """Camera at (0,1.6,0) looking at a huge wall z=-10 that fills the frame."""
    eye, target = (0.0, 1.6, 0.0), (0.0, 1.6, -10.0)
    view, world, rot3 = look_at_view_matrix(eye, target)
    extr = AtlasExtrinsics(camera_position=eye, camera_rotation_matrix=rot3,
                           camera_world_matrix=world, camera_view_matrix=view)
    intr = AtlasIntrinsics(image_width=640, image_height=480, fx_px=400.0,
                           fy_px=400.0, cx_px=320.0, cy_px=240.0)
    solve = AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))
    # Wall comfortably wider than the frustum (~4x) but realistically bounded
    # — relief meshes are frustum-derived, and the tool's conservative
    # whole-face near-plane cull treats any face touching the camera plane
    # as a hole (a +/-200 wall at 10 deep wraps behind the camera on a 3deg
    # orbit and would cull entirely; found while writing this test).
    s = 30.0
    verts = [(-s, 1.6 - s, -10.0), (s, 1.6 - s, -10.0),
             (s, 1.6 + s, -10.0), (-s, 1.6 + s, -10.0)]
    faces = [(0, 1, 2), (0, 2, 3)]
    if stretched:
        # A sliver triangle (edge ratio >> 12) pasted over the wall centre.
        verts += [(-5.0, 1.6, -9.9), (5.0, 1.6, -9.9), (5.0, 1.7, -9.9)]
        faces += [(4, 5, 6)]
    prim = AtlasProxyPrimitive(
        name="wall", primitive_type="mesh",
        metadata={"vertices": [c for v in verts for c in v],
                  "faces": [i for f in faces for i in f],
                  "uvs": [0.0] * (2 * len(verts)),
                  "n_vertices": len(verts), "n_faces": len(faces),
                  "role": "projection_proxy"})
    solve.projection_scene.proxy_geometry.append(prim)
    return solve


def test_full_wall_has_no_holes_at_any_small_orbit():
    report = _MOD.run_stress(_wall_solve(), res=128, az_steps=(3.0, 6.0),
                             el_steps=(3.0,))
    assert len(report["poses"]) == 7  # 0 + ±3/±6 az + ±3 el
    for row in report["poses"]:
        assert row["hole_pct"] < 1.0, row
        assert row["stretch_pct"] == 0.0


def test_stretched_sliver_is_scored():
    report = _MOD.run_stress(_wall_solve(stretched=True), res=128,
                             az_steps=(3.0,), el_steps=())
    zero = report["poses"][0]
    assert zero["stretch_pct"] > 0.0
    assert report["layers"][0]["n_faces"] == 3


def test_no_geometry_exits_clearly():
    solve = _wall_solve()
    solve.projection_scene.proxy_geometry.clear()
    with pytest.raises(SystemExit, match="no mesh geometry"):
        _MOD.run_stress(solve)
