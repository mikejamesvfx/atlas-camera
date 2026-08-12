"""Headless CLI (create/validate) for Dynamic Plates."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image

from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.io import save_solve_json
from atlas_camera.core.schema import AtlasExtrinsics, AtlasSolve, LatentCamera
from atlas_camera.dynamic.cli import main


@pytest.fixture
def shot(tmp_path):
    """A tiny castle-by-sea stand-in: image, water matte, saved solve."""
    rng = np.random.default_rng(0)
    img_path = tmp_path / "castle.png"
    Image.fromarray((rng.random((360, 640, 3)) * 255).astype(np.uint8)
                    ).save(img_path)
    matte = np.zeros((360, 640), dtype=np.uint8)
    matte[250:360, :] = 255
    matte_path = tmp_path / "ocean_mask.png"
    Image.fromarray(matte, mode="L").save(matte_path)

    view, world, rot3 = look_at_view_matrix((0.0, 12.0, 0.0), (0.0, 0.0, -50.0))
    solve = AtlasSolve(
        camera=LatentCamera(
            intrinsics=build_intrinsics(image_width=640, image_height=360,
                                        focal_length_mm=32.0),
            extrinsics=AtlasExtrinsics(camera_position=(0.0, 12.0, 0.0),
                                       camera_rotation_matrix=rot3,
                                       camera_world_matrix=world,
                                       camera_view_matrix=view)),
        image_width=640, image_height=360)
    solve_path = tmp_path / "atlas_solve.json"
    save_solve_json(solve, solve_path)
    return {"image": img_path, "matte": matte_path, "solve": solve_path,
            "out": tmp_path / "shot"}


def test_create_and_validate_generator_none(shot, capsys):
    rc = main(["create", "--image", str(shot["image"]),
               "--matte", str(shot["matte"]), "--type", "water",
               "--solve", str(shot["solve"]), "--out", str(shot["out"]),
               "--generator", "none", "--blender"])
    out = capsys.readouterr().out
    assert rc == 0, out
    pkg = shot["out"] / "dynamic" / "WATER_0001"
    assert (pkg / "manifest.json").exists()
    assert (pkg / "source" / "crop.png").exists()
    assert (pkg / "geometry" / "receiver.obj").exists()
    assert (pkg / "blender_open_scene.py").exists()
    assert "crop camera:" in out
    assert "dynamic plate OK" in out

    rc = main(["validate", "--package", str(pkg)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "valid" in out


def test_create_second_plate_increments_id(shot, capsys):
    for _ in range(2):
        rc = main(["create", "--image", str(shot["image"]),
                   "--matte", str(shot["matte"]),
                   "--solve", str(shot["solve"]), "--out", str(shot["out"])])
        assert rc == 0
    capsys.readouterr()
    assert (shot["out"] / "dynamic" / "WATER_0002").exists()


def test_create_rejects_bad_matte_dims(shot, tmp_path, capsys):
    bad = tmp_path / "bad_matte.png"
    Image.fromarray(np.full((100, 100), 255, dtype=np.uint8), mode="L"
                    ).save(bad)
    rc = main(["create", "--image", str(shot["image"]), "--matte", str(bad),
               "--solve", str(shot["solve"]), "--out", str(shot["out"])])
    out = capsys.readouterr().out
    assert rc == 1
    assert "region_invalid" in out


def test_create_rejects_empty_matte(shot, tmp_path, capsys):
    empty = tmp_path / "empty.png"
    Image.fromarray(np.zeros((360, 640), dtype=np.uint8), mode="L").save(empty)
    rc = main(["create", "--image", str(shot["image"]), "--matte", str(empty),
               "--solve", str(shot["solve"]), "--out", str(shot["out"])])
    out = capsys.readouterr().out
    assert rc == 1
    assert "matte is empty" in out


def test_generator_ltx_unavailable_still_succeeds(shot, capsys, monkeypatch):
    monkeypatch.delenv("ATLAS_LTX_TEMPLATE", raising=False)
    rc = main(["create", "--image", str(shot["image"]),
               "--matte", str(shot["matte"]),
               "--solve", str(shot["solve"]), "--out", str(shot["out"]),
               "--generator", "ltx"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "status = not_available" in out
    pkg = shot["out"] / "dynamic" / "WATER_0001"
    assert (pkg / "generated" / "generation_result.json").exists()


def test_auto_matte_uses_sam3(shot, capsys, monkeypatch):
    def fake_mask(image, concepts, **kwargs):
        m = np.zeros((360, 640), dtype=bool)
        m[250:360, :] = True
        return m, ["ocean"], 0.3

    import atlas_camera.inference.sam3_segmenter as seg
    monkeypatch.setattr(seg, "sam3_concept_mask", fake_mask)
    rc = main(["create", "--image", str(shot["image"]),
               "--auto-matte", "ocean, sea water",
               "--solve", str(shot["solve"]), "--out", str(shot["out"])])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "auto-matte: matched=['ocean']" in out
