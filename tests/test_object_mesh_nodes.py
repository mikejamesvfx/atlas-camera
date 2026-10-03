"""AtlasObjectCrop + AtlasImportGeneratedMesh on an analytic ground+box scene.

The "model output" is the truth box expressed in Pixal3D's frame at an
arbitrary scale, so a correct node chain must put it back where it came from.
"""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas_camera.comfy.nodes import NODE_CLASS_MAPPINGS  # noqa: E402
from atlas_camera.comfy.nodes_object_mesh import (  # noqa: E402
    AtlasImportGeneratedMesh,
    AtlasObjectCrop,
)
from atlas_camera.core.generated_mesh import pixal_camera_distance, world_to_source_camera  # noqa: E402
from atlas_camera.core.move_budget import rasterize_coverage  # noqa: E402
from atlas_camera.core.object_crop import ObjectCropCamera  # noqa: E402
from atlas_camera.core.scene_health import evaluate_scene_health  # noqa: E402
from atlas_camera.core.schema import (  # noqa: E402
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasSolve,
    LatentCamera,
)
from atlas_camera.inference.depth_estimator import DepthResult  # noqa: E402

from test_generated_mesh_import import _subdivided_box  # noqa: E402

W, H = 320, 200
FX = FY = 260.0
CX, CY = W / 2.0, H / 2.0
CAM_HEIGHT = 1.6
SKY = 80.0
BOX_CENTER = (1.4, 0.5, -5.0)
BOX_SIZE = (1.0, 1.0, 1.0)
S0 = 6.0


def _view():
    return np.array([[1.0, 0, 0, 0], [0, 1.0, 0, -CAM_HEIGHT],
                     [0, 0, 1.0, 0], [0, 0, 0, 1.0]])


def _solve():
    intr = AtlasIntrinsics(image_width=W, image_height=H, focal_length_mm=35.0,
                           sensor_width_mm=36.0, fx_px=FX, fy_px=FY, cx_px=CX, cy_px=CY)
    extr = AtlasExtrinsics(camera_view_matrix=tuple(map(tuple, _view())))
    return AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))


def _raster(v, f):
    return rasterize_coverage(v, f, view_matrix=_view(), fx=FX, fy=FY, cx=CX, cy=CY,
                              width=W, height=H, backend="numpy")


@pytest.fixture()
def scene():
    box_v, box_f = _subdivided_box(BOX_CENTER, BOX_SIZE, n=8)
    ground_v = np.array([[-60, 0, 2], [60, 0, 2], [60, 0, -60], [-60, 0, -60]], dtype=float)
    ground_f = np.array([[0, 1, 2], [0, 2, 3]])
    box_alpha, box_z = _raster(box_v, box_f)
    _, ground_z = _raster(ground_v, ground_f)
    depth = np.minimum(box_z, ground_z)
    sky = ~np.isfinite(depth)
    depth = np.where(sky, SKY, depth).astype(np.float32)
    dr = DepthResult(depth=depth, is_metric=True, model_id="fake", image_width=W,
                     image_height=H, near=float(depth.min()), far=float(depth.max()))
    image = torch.rand(1, H, W, 3)
    obj = torch.from_numpy(box_alpha.astype(np.float32))[None]
    sky_t = torch.from_numpy(sky.astype(np.float32))[None]
    return dict(solve=_solve(), depth=dr, image=image, obj=obj, sky=sky_t,
                box_v=box_v, box_f=box_f)


def _pixal_mesh(scene, handle, *, mirror_x=False, colours=True):
    crop = ObjectCropCamera.from_dict(handle)
    p_s = world_to_source_camera(scene["box_v"], view_matrix=_view())
    p_v = (p_s / S0) @ crop.rotation.T
    v = p_v + np.array([0.0, 0.0, pixal_camera_distance(crop.fov_deg)])
    if mirror_x:
        v[:, 0] = -v[:, 0]
    cols = torch.full((1, len(v), 3), 0.6) if colours else None
    return SimpleNamespace(vertices=torch.from_numpy(v).float()[None],
                           faces=torch.from_numpy(scene["box_f"]).int()[None],
                           vertex_colors=cols)


def test_registered_with_stable_sockets():
    assert NODE_CLASS_MAPPINGS["AtlasObjectCrop"] is AtlasObjectCrop
    assert AtlasObjectCrop.RETURN_TYPES == ("IMAGE", "MASK", "FLOAT", "ATLAS_OBJECT_CROP",
                                            "STRING")
    assert AtlasImportGeneratedMesh.RETURN_TYPES == ("ATLAS_SOLVE", "STRING", "MASK")


def test_crop_outputs(scene):
    crop, cmask, fov, handle, report = AtlasObjectCrop().crop(
        scene["solve"], scene["image"], scene["obj"], size=256)
    assert crop.shape == (1, 256, 256, 3) and cmask.shape == (1, 256, 256)
    assert 0.0 < fov < 90.0 and handle["kind"] == "atlas_object_crop"
    # Object fills ~1/pad of the frame, centred.
    ys, xs = np.nonzero(cmask[0].numpy() > 0.5)
    assert abs((xs.min() + xs.max()) / 2 - 127.5) < 8
    assert "from the solve" in report


def test_import_puts_the_object_back(scene):
    _, _, _, handle, _ = AtlasObjectCrop().crop(scene["solve"], scene["image"],
                                                 scene["obj"], size=256)
    solve_out, report, alpha = AtlasImportGeneratedMesh().import_mesh(
        scene["solve"], _pixal_mesh(scene, handle), handle, scene["depth"],
        scene["image"], scene["obj"], sky_mask=scene["sky"], rests_on_ground=True)
    assert "APPENDED" in report, report
    prims = [p for p in solve_out.projection_scene.proxy_geometry
             if (p.metadata or {}).get("source") == "pixal3d"]
    assert len(prims) == 1
    meta = prims[0].metadata
    assert meta["generated_grade"] == "ok", report
    v = np.asarray(meta["vertices"]).reshape(-1, 3)
    assert np.max(np.abs(v - scene["box_v"])) < 0.05
    assert len(meta["vertex_colors"]) == len(meta["vertices"])
    assert len(meta["photo_weight"]) == len(v)
    w = np.asarray(meta["photo_weight"])
    assert 0.1 < (w >= 0.5).mean() < 0.9   # some seen, some hidden
    assert alpha.shape == (1, H, W) and float(alpha.sum()) > 100
    # Input solve untouched (deepcopy) and the health engine stays quiet.
    assert not scene["solve"].projection_scene.proxy_geometry
    codes = [f.code for f in evaluate_scene_health(solve_out).flags]
    assert "generated_object_unverified" not in codes


def test_mirrored_mesh_is_not_ok(scene):
    _, _, _, handle, _ = AtlasObjectCrop().crop(scene["solve"], scene["image"],
                                                 scene["obj"], size=256)
    solve_out, report, _ = AtlasImportGeneratedMesh().import_mesh(
        scene["solve"], _pixal_mesh(scene, handle, mirror_x=True), handle,
        scene["depth"], scene["image"], scene["obj"], sky_mask=scene["sky"],
        on_gate_fail="inspect")
    prims = [p for p in solve_out.projection_scene.proxy_geometry
             if (p.metadata or {}).get("source") == "pixal3d"]
    assert prims and prims[0].metadata["generated_grade"] != "ok", report
    codes = [f.code for f in evaluate_scene_health(solve_out).flags]
    assert "generated_object_unverified" in codes


def test_refuse_passes_solve_through(scene):
    _, _, _, handle, _ = AtlasObjectCrop().crop(scene["solve"], scene["image"],
                                                 scene["obj"], size=256)
    bad_depth = DepthResult(depth=np.full((H, W), np.nan, dtype=np.float32), is_metric=True,
                            model_id="fake", image_width=W, image_height=H, near=0.0, far=1.0)
    solve_out, report, _ = AtlasImportGeneratedMesh().import_mesh(
        scene["solve"], _pixal_mesh(scene, handle), handle, bad_depth,
        scene["image"], scene["obj"])
    assert "REFUSED" in report
    assert not solve_out.projection_scene.proxy_geometry


def test_missing_vertex_colours_go_grey(scene):
    _, _, _, handle, _ = AtlasObjectCrop().crop(scene["solve"], scene["image"],
                                                 scene["obj"], size=256)
    solve_out, report, _ = AtlasImportGeneratedMesh().import_mesh(
        scene["solve"], _pixal_mesh(scene, handle, colours=False), handle,
        scene["depth"], scene["image"], scene["obj"], sky_mask=scene["sky"])
    assert "neutral grey" in report
    meta = solve_out.projection_scene.proxy_geometry[-1].metadata
    assert set(meta["vertex_colors"]) == {0.5}


def test_empty_mask_raises(scene):
    with pytest.raises(ValueError, match="empty"):
        AtlasObjectCrop().crop(scene["solve"], scene["image"], torch.zeros(1, H, W))


# --- refusals are reports, never crashes ---------------------------------------

def _nan_depth():
    return DepthResult(depth=np.full((H, W), np.nan, dtype=np.float32), is_metric=True,
                       model_id="fake", image_width=W, image_height=H, near=0.0, far=1.0)


def _handle(scene):
    return AtlasObjectCrop().crop(scene["solve"], scene["image"], scene["obj"], size=256)[3]


def test_unmeasurable_scale_refuses_even_under_inspect(scene):
    # on_gate_fail="inspect" used to reach float(None) and raise TypeError.
    handle = _handle(scene)
    solve_out, report, alpha = AtlasImportGeneratedMesh().import_mesh(
        scene["solve"], _pixal_mesh(scene, handle), handle, _nan_depth(),
        scene["image"], scene["obj"], on_gate_fail="inspect")
    assert "REFUSED — scale unmeasurable" in report
    assert "regardless of on_gate_fail" in report
    assert not solve_out.projection_scene.proxy_geometry
    assert float(alpha.sum()) == 0.0


def test_out_of_range_faces_refuse_before_decimation(scene):
    handle = _handle(scene)
    mesh = _pixal_mesh(scene, handle)
    bad = mesh.faces.clone()
    bad[0, 0, 0] = int(mesh.vertices.shape[1]) + 5
    mesh.faces = bad
    # A budget below the face count would decimate first and hit IndexError.
    solve_out, report, _ = AtlasImportGeneratedMesh().import_mesh(
        scene["solve"], mesh, handle, scene["depth"], scene["image"], scene["obj"],
        max_faces=100)
    assert "REFUSED — malformed MESH" in report and "face indices" in report
    assert not solve_out.projection_scene.proxy_geometry


def test_negative_face_index_refuses(scene):
    handle = _handle(scene)
    mesh = _pixal_mesh(scene, handle)
    mesh.faces[0, 3, 1] = -1
    _, report, _ = AtlasImportGeneratedMesh().import_mesh(
        scene["solve"], mesh, handle, scene["depth"], scene["image"], scene["obj"])
    assert "REFUSED — malformed MESH" in report


def test_nan_vertices_refuse(scene):
    handle = _handle(scene)
    mesh = _pixal_mesh(scene, handle)
    mesh.vertices[0, 2, 1] = float("nan")
    solve_out, report, _ = AtlasImportGeneratedMesh().import_mesh(
        scene["solve"], mesh, handle, scene["depth"], scene["image"], scene["obj"],
        on_gate_fail="inspect")
    assert "REFUSED — malformed MESH" in report and "non-finite" in report
    assert not solve_out.projection_scene.proxy_geometry


def test_missed_face_budget_is_reported(scene, monkeypatch):
    from atlas_camera.core import generated_mesh

    real = generated_mesh.cluster_decimate
    monkeypatch.setattr(generated_mesh, "cluster_decimate",
                        lambda *a, **k: real(*a, **{**k, "max_rounds": 0}))
    handle = _handle(scene)
    _, report, _ = AtlasImportGeneratedMesh().import_mesh(
        scene["solve"], _pixal_mesh(scene, handle), handle, scene["depth"],
        scene["image"], scene["obj"], sky_mask=scene["sky"], max_faces=100)
    assert "decimation MISSED the face budget" in report


# --- the crop matte is zero where the crop sees outside the photo ----------

def test_frame_edge_crop_matte_is_zero_outside_the_photo(scene):
    from atlas_camera.core.object_crop import crop_sample_grid

    obj = torch.zeros(1, H, W)
    obj[:, 60:140, W - 30:] = 1.0          # an object cut by the right frame edge
    for background in ("black", "photo"):
        crop, cmask, _, handle, report = AtlasObjectCrop().crop(
            scene["solve"], scene["image"] + 0.5, obj, size=128, background=background)
        _, _, valid = crop_sample_grid(ObjectCropCamera.from_dict(handle))
        assert (~valid).any(), "fixture must put part of the crop outside the photo"
        assert float(cmask[0].numpy()[~valid].max()) == 0.0
        assert float(cmask[0].numpy()[valid].max()) > 0.5
        if background == "photo":
            assert float(crop[0].numpy()[~valid].max()) == 0.0
        assert "outside the photo" in report
