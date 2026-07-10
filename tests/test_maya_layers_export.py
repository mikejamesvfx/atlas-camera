"""Tests for the all-in-one Maya layers export (write_maya_layers_scene +
AtlasExportMayaLayers) — the Maya twin of the Nuke layers exporter: native
per-layer cameras in one .ma + an on-open scriptNode building the projection
networks. Reuses the analytic layered-solve fixture pattern from
tests/test_nuke_layers_export.py.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from atlas_camera.comfy.nodes import (
    NODE_CLASS_MAPPINGS,
    AtlasCleanPlateLayer,
    AtlasExportMayaLayers,
    AtlasSkyDomeLayer,
)
from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.schema import AtlasExtrinsics, AtlasIntrinsics, AtlasSolve, LatentCamera
from atlas_camera.exporters.maya_exporter import _matrix_to_maya_trs, write_maya_layers_scene
from atlas_camera.inference.depth_estimator import DepthResult

W = H = 256
FX = FY = 250.0
CX = CY = 128.0
SKY = 60.0
CAM_HEIGHT = 1.6


def _view_matrix(h):
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, -h),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _solve(h=CAM_HEIGHT):
    intr = AtlasIntrinsics(
        image_width=W, image_height=H, focal_length_mm=35.0, sensor_width_mm=36.0,
        fx_px=FX, fy_px=FY, cx_px=CX, cy_px=CY,
    )
    extr = AtlasExtrinsics(camera_view_matrix=_view_matrix(h),
                           camera_position=(0.0, h, 0.0),
                           camera_world_matrix=(
                               (1.0, 0.0, 0.0, 0.0),
                               (0.0, 1.0, 0.0, h),
                               (0.0, 0.0, 1.0, 0.0),
                               (0.0, 0.0, 0.0, 1.0)))
    return AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))


def _scene_depth(h=CAM_HEIGHT, wall_z=-10.0, wall_h=3.0):
    _, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dy = -(vv - CY) / FY
    depth = np.full((H, W), SKY)
    tg = np.full((H, W), np.inf)
    ld = dy < -1e-6
    tg[ld] = -h / dy[ld]
    t = -wall_z
    y_at = h + dy * t
    vis = (y_at >= 0.0) & (y_at <= wall_h)
    return np.stack([depth, np.where(np.isfinite(tg), tg, SKY),
                     np.where(vis, t, SKY)]).min(axis=0).astype(np.float32)


def _depth_result(depth_map):
    return DepthResult(depth=depth_map, is_metric=True, model_id="fake",
                       image_width=W, image_height=H,
                       near=float(depth_map.min()), far=float(depth_map.max()))


def _layered_solve():
    solve = _solve()
    depth_map = _scene_depth()
    depth = _depth_result(depth_map)
    plate = torch.rand(1, H, W, 3, dtype=torch.float32)
    sky = torch.from_numpy((depth_map >= SKY - 1e-6).astype(np.float32)).unsqueeze(0)
    out, _, _ext = AtlasSkyDomeLayer().add_layer(solve, depth, sky, plate, relief_grid=32,
                                           name="sky", edge_extend_px=0, frame_outpaint_px=0)
    out, _, _ext = AtlasCleanPlateLayer().add_layer(out, depth, plate, near_m=5.0, far_m=12.0,
                                              relief_grid=32, name="bg")
    return out


def test_node_registered():
    assert NODE_CLASS_MAPPINGS["AtlasExportMayaLayers"] is AtlasExportMayaLayers
    assert AtlasExportMayaLayers.RETURN_TYPES == ("STRING", "STRING")


@pytest.mark.parametrize("eye,target", [
    ((0.0, 2.0, 5.0), (0.0, 0.0, 0.0)),
    ((3.0, 1.5, -4.0), (-1.0, 0.5, 2.0)),
    ((-2.0, 6.0, 1.0), (0.0, 0.0, -8.0)),
])
def test_euler_decomposition_recomposes_the_rotation(eye, target):
    """_matrix_to_maya_trs must be an exact inverse under Maya's default
    'xyz' rotate order (column-vector composition C = Rz @ Ry @ Rx) — this
    guards the only genuinely risky math in the .ma writer."""
    _view, world, _rot = look_at_view_matrix(eye, target)
    t, (rx, ry, rz) = _matrix_to_maya_trs(world)
    assert t == pytest.approx(eye)

    a, b, c = math.radians(rx), math.radians(ry), math.radians(rz)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    recomposed = Rz @ Ry @ Rx
    original = np.array([[world[i][j] for j in range(3)] for i in range(3)])
    np.testing.assert_allclose(recomposed, original, atol=1e-9)


def test_layers_export_writes_ma_with_cameras_and_scriptnode(tmp_path):
    solve = _layered_solve()
    result = write_maya_layers_scene(solve, tmp_path)

    assert result["layers"] == ["sky", "bg"]
    ma = (tmp_path / "maya_layers.ma").read_text(encoding="utf-8")
    # Native per-layer cameras + the render camera.
    assert 'createNode transform -n "atlas_RenderCam"' in ma
    for layer in ("sky", "bg"):
        assert f'createNode transform -n "atlas_{layer}_ProjCam"' in ma
        assert (tmp_path / f"{layer}_plate.png").exists()
        assert (tmp_path / f"{layer}_mesh.obj").exists()
    # On-open scriptNode (python, execute-on-open) carrying the builder.
    assert 'createNode script -n "atlasLayersOnOpen"' in ma
    assert 'setAttr ".scriptType" 1;' in ma
    assert 'setAttr ".sourceType" 1;' in ma
    # Perspective projection frustum comes from linkedCamera (the projection
    # node has no focal/aperture attrs — confirmed live in Maya 2027), and
    # imported OBJs get the cm->m x100 compensation.
    assert "projType" in ma and "linkedCamera" in ma
    assert "100, 100, 100" in ma
    # Units declared metric; paths forward-slashed inside the script string.
    assert "currentUnit -l meter" in ma
    assert ":\\\\" not in ma.replace("\\\\n", "")  # no unescaped drive-letter backslash paths


def test_sky_matte_reaches_transparency_wiring(tmp_path):
    solve = _layered_solve()
    write_maya_layers_scene(solve, tmp_path)
    ma = (tmp_path / "maya_layers.ma").read_text(encoding="utf-8")
    # The sky layer embeds a matte -> its script entry must carry
    # has_matte True (which wires file.outTransparency -> lambert.transparency).
    assert "outTransparency" in ma
    assert (tmp_path / "sky_matte.png").exists()


def test_export_errors_loudly_without_layers(tmp_path):
    with pytest.raises(ValueError, match="No exportable projection layers"):
        write_maya_layers_scene(_solve(), tmp_path)


def test_node_wrapper_returns_paths_and_summary(tmp_path):
    solve = _layered_solve()
    ma_path, summary = AtlasExportMayaLayers().export(solve, str(tmp_path))
    assert ma_path.endswith("maya_layers.ma")
    assert "2 layer(s): sky, bg" in summary
