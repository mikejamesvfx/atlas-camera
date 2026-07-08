"""Tests for the multi-layer Nuke export (write_nuke_layers_script +
AtlasExportNukeLayers) — every ProjectionSource on a solve becomes its own
Read + Camera2 + Project3D2 + ReadGeo2 chain, merged through one Scene into
one ScanlineRender rendered from the primary camera.

Builds the layers through the REAL nodes (AtlasSkyDomeLayer +
AtlasCleanPlateLayer) on the same analytic occluder fixture the inpaint-layer
tests use, so this also covers the primitive->ReliefMesh round-trip
(_mesh_from_primitive is relief_mesh_primitive's exact inverse).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from atlas_camera.comfy.nodes import (
    NODE_CLASS_MAPPINGS,
    AtlasCleanPlateLayer,
    AtlasExportNukeLayers,
    AtlasSkyDomeLayer,
)
from atlas_camera.core.schema import AtlasExtrinsics, AtlasIntrinsics, AtlasSolve, LatentCamera
from atlas_camera.exporters.nuke_exporter import write_nuke_layers_script
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
    """Solve with two real layers: a sky dome + one clean-plate band."""
    solve = _solve()
    depth_map = _scene_depth()
    depth = _depth_result(depth_map)
    plate = torch.rand(1, H, W, 3, dtype=torch.float32)

    sky = torch.from_numpy((depth_map >= SKY - 1e-6).astype(np.float32)).unsqueeze(0)
    out, _ = AtlasSkyDomeLayer().add_layer(solve, depth, sky, plate, relief_grid=32, name="sky")
    out, _ = AtlasCleanPlateLayer().add_layer(out, depth, plate, near_m=5.0, far_m=12.0,
                                              relief_grid=32, name="bg")
    return out


def test_node_registered():
    assert NODE_CLASS_MAPPINGS["AtlasExportNukeLayers"] is AtlasExportNukeLayers
    assert AtlasExportNukeLayers.RETURN_TYPES == ("STRING", "STRING")


def test_layers_export_writes_nk_plates_and_objs(tmp_path):
    solve = _layered_solve()
    result = write_nuke_layers_script(solve, tmp_path)

    assert result["layers"] == ["sky", "bg"]
    nk = (tmp_path / "nuke_layers.nk").read_text(encoding="utf-8")
    # One Read/ProjCam/Project3D/Geo chain per layer, plus Scene + render.
    for layer in ("sky", "bg"):
        assert f"Read_{layer}" in nk
        assert f"ProjCam_{layer}" in nk
        assert f"Project3D_{layer}" in nk
        assert f"Geo_{layer}" in nk
        assert (tmp_path / f"{layer}_plate.png").exists()
        assert (tmp_path / f"{layer}_mesh.obj").exists()
    assert "Scene {" in nk and "inputs 2" in nk
    assert "ScanlineRender1" in nk
    # Render camera wired via the proven onScriptLoad callback, never pushed.
    assert "onScriptLoad" in nk and "RenderCam1" in nk
    # All embedded paths are forward-slashed (TCL escaping eats backslashes).
    assert "\\\\" not in nk.replace("\\n", "")


def test_layer_cameras_carry_their_own_pose(tmp_path):
    solve = _layered_solve()
    # Give the second layer's ProjectionSource a distinct camera position to
    # prove per-layer cameras are honored (patch sources orbit for real).
    moved = solve.projection_sources[1]
    intr = moved.camera.intrinsics
    extr = AtlasExtrinsics(camera_view_matrix=_view_matrix(2.5),
                           camera_position=(3.0, 2.5, 1.0),
                           camera_world_matrix=(
                               (1.0, 0.0, 0.0, 3.0),
                               (0.0, 1.0, 0.0, 2.5),
                               (0.0, 0.0, 1.0, 1.0),
                               (0.0, 0.0, 0.0, 1.0)))
    moved.camera = LatentCamera(intrinsics=intr, extrinsics=extr)

    write_nuke_layers_script(solve, tmp_path)
    nk = (tmp_path / "nuke_layers.nk").read_text(encoding="utf-8")
    assert "translate {3.0 2.5 1.0}" in nk


def test_export_errors_loudly_without_layers(tmp_path):
    with pytest.raises(ValueError, match="No exportable projection layers"):
        write_nuke_layers_script(_solve(), tmp_path)


def test_node_wrapper_returns_paths_and_summary(tmp_path):
    solve = _layered_solve()
    nk_path, summary = AtlasExportNukeLayers().export(solve, str(tmp_path))
    assert nk_path.endswith("nuke_layers.nk")
    assert "2 layer(s): sky, bg" in summary


def test_matte_lands_in_plate_alpha_and_standalone_file(tmp_path):
    from PIL import Image

    solve = _layered_solve()
    # The sky dome embeds its segmentation as mask_b64 automatically.
    assert solve.projection_sources[0].mask_b64

    write_nuke_layers_script(solve, tmp_path)
    sky_plate = Image.open(tmp_path / "sky_plate.png")
    assert sky_plate.mode == "RGBA"          # matte embedded in alpha
    assert (tmp_path / "sky_matte.png").exists()  # and standalone
    # The bg layer had no matte -> plain RGB plate, no matte file.
    bg_plate = Image.open(tmp_path / "bg_plate.png")
    assert bg_plate.mode == "RGB"
    assert not (tmp_path / "bg_matte.png").exists()
