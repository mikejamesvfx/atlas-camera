"""The plate's SDR->HDR tone curve, carried onto generated vertex colour."""

import numpy as np
import pytest

from atlas_camera.core.hdr_transfer import apply_curve, fit_sdr_to_hdr_curve, srgb_to_acescg


def _model(x_acescg):
    """A stand-in conversion: linear below 0.4, highlights expanded above."""
    return np.where(x_acescg < 0.4, x_acescg, 0.4 + (x_acescg - 0.4) * 6.0)


def _plate_pair(h=300, w=400):
    rng = np.random.default_rng(0)
    sdr = rng.random((h, w, 3)).astype(np.float32)
    return sdr, _model(srgb_to_acescg(sdr)).astype(np.float32)


def test_curve_recovers_the_conversion_and_applies_to_unseen_colours():
    sdr, hdr = _plate_pair()
    curve = fit_sdr_to_hdr_curve(sdr, hdr, bins=96)
    assert curve["residual_stops"] < 0.02
    vc = np.array([[0.2, 0.3, 0.4], [0.95, 0.9, 0.85], [0.5, 0.5, 0.5]], np.float32)
    got = apply_curve(curve, vc)
    want = _model(srgb_to_acescg(vc))
    assert np.max(np.abs(np.log2(got / want))) < 0.08
    assert got.max() > 1.0                       # highlights really expanded


def test_curve_is_monotone_and_extrapolates_upward():
    sdr, hdr = _plate_pair()
    curve = fit_sdr_to_hdr_curve(sdr * 0.8, hdr)  # plate never reaches white
    ys = np.asarray(curve["log2_y"])
    assert (np.diff(ys, axis=1) >= -1e-9).all()
    lo, hi = apply_curve(curve, np.array([[0.8, 0.8, 0.8], [1.0, 1.0, 1.0]], np.float32))
    assert (hi >= lo).all()


def test_misaligned_plates_refused():
    with pytest.raises(ValueError, match="pixel-aligned"):
        fit_sdr_to_hdr_curve(np.zeros((4, 4, 3)), np.zeros((4, 5, 3)))


def test_node_stores_hdr_vertex_colours_and_scene_writes_ply(tmp_path):
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes_object_mesh import AtlasHDRVertexTransfer
    from atlas_camera.core.proxy_geometry import PROXY_ROLE
    from atlas_camera.core.schema import (AtlasExtrinsics, AtlasIntrinsics,
                                          AtlasProxyPrimitive, AtlasSolve, LatentCamera)
    from atlas_camera.exporters.scene_glb import build_scene_layers

    solve = AtlasSolve(camera=LatentCamera(
        intrinsics=AtlasIntrinsics(image_width=400, image_height=300, focal_length_mm=35.0,
                                   sensor_width_mm=36.0, fx_px=300.0, fy_px=300.0),
        extrinsics=AtlasExtrinsics(camera_view_matrix=tuple(map(tuple, np.eye(4))))))
    verts = [[0, 0, -5], [1, 0, -5], [0, 1, -5]]
    solve.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
        name="pixal3d_object", primitive_type="mesh", dimensions=(0.0, 0.0, 0.0),
        material="m", metadata={"role": PROXY_ROLE, "source": "pixal3d",
                                "vertices": np.ravel(verts).tolist(), "faces": [0, 1, 2],
                                "uvs": [0, 0, 1, 0, 0, 1],
                                "vertex_colors": [0.95, 0.9, 0.85] * 3,
                                "photo_weight": [0.0, 0.0, 0.0]}))
    sdr, hdr = _plate_pair()
    out, report = AtlasHDRVertexTransfer().transfer(
        solve, torch.from_numpy(sdr)[None], hdr_image=torch.from_numpy(hdr)[None])
    meta = out.projection_scene.proxy_geometry[0].metadata
    assert len(meta["vertex_colors_hdr"]) == 9 and max(meta["vertex_colors_hdr"]) > 1.0
    assert meta["vertex_colors_hdr_space"] == "ACEScg" and "reconstruction" in report
    assert "vertex_colors_hdr" not in solve.projection_scene.proxy_geometry[0].metadata

    from PIL import Image
    layers, sidecars, _ = build_scene_layers(out, Image.new("RGB", (400, 300)), exr_dir=tmp_path,
                                             exr_prefix="t")
    ply = tmp_path / "t_pixal3d_object_vertex_hdr.ply"
    assert ply.is_file() and ply.read_bytes().startswith(b"ply\nformat binary_little_endian")
    assert layers[0].extras["vertex_colors_hdr_ply"] == ply.name


def test_node_passes_through_without_hdr():
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes_object_mesh import AtlasHDRVertexTransfer
    from atlas_camera.core.schema import AtlasSolve, LatentCamera, AtlasIntrinsics
    solve = AtlasSolve(camera=LatentCamera(intrinsics=AtlasIntrinsics(
        image_width=4, image_height=4, focal_length_mm=35.0, sensor_width_mm=36.0)))
    _, report = AtlasHDRVertexTransfer().transfer(solve, torch.zeros(1, 4, 4, 3))
    assert "passed through" in report
