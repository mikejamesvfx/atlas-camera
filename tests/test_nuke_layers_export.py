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
    out, _, _ext = AtlasSkyDomeLayer().add_layer(solve, depth, sky, plate, relief_grid=32, name="sky")
    out, _, _ext = AtlasCleanPlateLayer().add_layer(out, depth, plate, near_m=5.0, far_m=12.0,
                                              relief_grid=32, name="bg")
    return out


def test_node_registered():
    assert NODE_CLASS_MAPPINGS["AtlasExportNukeLayers"] is AtlasExportNukeLayers
    assert AtlasExportNukeLayers.RETURN_TYPES == ("STRING", "STRING")
    assert list(AtlasExportNukeLayers.INPUT_TYPES()["optional"])[0] == "output_profile"


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
    assert nk.count("colorspace sRGB - Display") == 2
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


def test_node_wrapper_does_not_mask_retopology_value_errors(monkeypatch, tmp_path):
    import atlas_camera.exporters.nuke_exporter as exporter

    def fail(*args, **kwargs):
        raise ValueError("retopology backend failed")

    monkeypatch.setattr(exporter, "write_nuke_layers_script", fail)
    with pytest.raises(ValueError, match="retopology backend failed"):
        AtlasExportNukeLayers().export(_layered_solve(), str(tmp_path))


def test_node_wrapper_returns_paths_and_summary(tmp_path):
    solve = _layered_solve()
    nk_path, summary = AtlasExportNukeLayers().export(solve, str(tmp_path))
    assert nk_path.endswith("nuke_layers.nk")
    assert "2 layer(s): sky, bg" in summary


def test_node_wrapper_retopologizes_every_layer(tmp_path):
    solve = _layered_solve()
    nk_path, summary = AtlasExportNukeLayers().export(
        solve, str(tmp_path), retopo_method="smooth",
        retopo_smooth_iterations=1,
    )
    assert nk_path.endswith("nuke_layers.nk")
    assert "smooth retopo" in summary
    assert (tmp_path / "sky_mesh.obj").exists()
    assert (tmp_path / "bg_mesh.obj").exists()


def test_shared_collector_reuses_byte_identical_retopology(monkeypatch, tmp_path):
    """Nuke and Maya must not get two slightly different remeshes."""
    from pathlib import Path

    from atlas_camera.exporters import _layers
    import atlas_camera.core.mesh_retopo as retopo_mod

    _layers._RETOPO_CACHE.clear()
    calls = {"n": 0}

    def deliberately_drifting_retopo(mesh, **kwargs):
        calls["n"] += 1
        mesh.vertices = mesh.vertices.copy()
        mesh.vertices[:, 0] += calls["n"] * 0.001
        return {"method": kwargs["method"], "changed": True, "note": "test"}

    monkeypatch.setattr(retopo_mod, "apply_retopo", deliberately_drifting_retopo)
    solve = _layered_solve()
    a, _ = _layers.collect_projection_layers(
        solve, tmp_path / "nuke", retopo_method="smooth",
        retopo_smooth_iterations=17,
    )
    b, _ = _layers.collect_projection_layers(
        solve, tmp_path / "maya", retopo_method="smooth",
        retopo_smooth_iterations=17,
    )
    assert calls["n"] == len(solve.projection_sources)
    for left, right in zip(a, b):
        assert Path(left["obj_path"]).read_bytes() == Path(right["obj_path"]).read_bytes()


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


def test_render_camera_is_animatable_and_bands_reformat(tmp_path):
    """RenderCam must use TRS (translate + rotate + rot_order), NOT useMatrix —
    Nuke greys out the transform channels when useMatrix is on, so a
    matrix-driven camera can't be keyframed for a move. Projection cameras keep
    useMatrix (static). And each band gets a 'resize none' Reformat so a swapped
    original-size EXR conforms to the band's (outpainted) projection format."""
    solve = _layered_solve()
    nk = open(write_nuke_layers_script(solve, tmp_path)["nk_path"]).read()

    start = nk.rfind("Camera2 {", 0, nk.index("name RenderCam1"))
    rc = nk[start:nk.index("}", nk.index("name RenderCam1"))]
    assert "rot_order XYZ" in rc and "rotate {" in rc
    assert "useMatrix" not in rc  # channels stay unlocked/animatable

    # every PROJECTION camera stays matrix-driven (static); the render cam does
    # NOT — so useMatrix count == number of projection cameras.
    n_proj = nk.count("ProjCam_")
    assert n_proj >= 1
    assert nk.count("useMatrix true") == n_proj

    # one 'resize none' conform Reformat per layer (name-agnostic Fit_<layer>)
    assert nk.count(" name Fit_") == n_proj
    assert nk.count("resize none") == n_proj
    assert "black_outside true" in nk


def test_render_camera_trs_round_trips_the_world_matrix():
    """The exported XYZ Euler + translate must reconstruct the camera world
    matrix exactly, so the animatable render camera still lines up with the
    matrix-driven projection cameras at frame 0."""
    import math
    from atlas_camera.exporters.nuke_exporter import _matrix_to_nuke_euler_xyz
    solve = _layered_solve()
    W = [list(r) for r in solve.camera.extrinsics.camera_world_matrix]
    a, b, c = (math.radians(v) for v in _matrix_to_nuke_euler_xyz(W))

    def rx(t): return np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    def ry(t): return np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
    def rz(t): return np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    R = np.array([row[:3] for row in W[:3]])
    assert np.abs(rx(a) @ ry(b) @ rz(c) - R).max() < 1e-9
