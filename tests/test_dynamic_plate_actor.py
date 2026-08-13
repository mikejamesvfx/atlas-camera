"""Track B: card receivers + chroma-key mattes + actor plates."""
from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image

from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.dynamic_plate import (
    ACTOR_PROMPT_DEFAULT,
    build_receiver_card,
    chroma_key_mattes,
    pixel_ray_world,
)
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasExtrinsics, LatentCamera


def _camera(width=640, height=360):
    view, world, rot3 = look_at_view_matrix((0.0, 10.0, 0.0), (0.0, 0.0, -40.0))
    return LatentCamera(
        intrinsics=build_intrinsics(image_width=width, image_height=height,
                                    focal_length_mm=32.0),
        extrinsics=AtlasExtrinsics(camera_position=(0.0, 10.0, 0.0),
                                   camera_rotation_matrix=rot3,
                                   camera_world_matrix=world,
                                   camera_view_matrix=view))


def test_card_sits_on_anchor_ray():
    cam = _camera()
    anchor = (480.0, 120.0)
    rec = build_receiver_card(cam, anchor_px=anchor, distance_m=25.0,
                              width_m=8.0)
    assert rec.kind == "card"
    tf = rec.primitive.transform_matrix
    centre = (tf[0][3], tf[1][3], tf[2][3])
    origin, direction = pixel_ray_world(cam, *anchor)
    expected = tuple(origin[i] + 25.0 * direction[i] for i in range(3))
    assert centre == pytest.approx(expected, abs=1e-9)
    # normal points back toward the camera
    n = (tf[0][2], tf[1][2], tf[2][2])
    to_cam = tuple(origin[i] - centre[i] for i in range(3))
    dot = sum(n[i] * to_cam[i] for i in range(3))
    assert dot > 0
    assert rec.primitive.dimensions[0] == 8.0
    assert rec.primitive.metadata["kind"] == "card"


def test_card_axes_are_camera_frame():
    cam = _camera()
    rec = build_receiver_card(cam, anchor_px=(320.0, 180.0), distance_m=10.0,
                              width_m=4.0, height_m=3.0)
    tf = rec.primitive.transform_matrix
    world = cam.extrinsics.camera_world_matrix
    for col in range(3):
        axis = (tf[0][col], tf[1][col], tf[2][col])
        cam_axis = (world[0][col], world[1][col], world[2][col])
        assert axis == pytest.approx(cam_axis)
    assert rec.primitive.dimensions[:2] == (4.0, 3.0)


def test_chroma_key_extracts_subject(tmp_path):
    frames = []
    for i in range(3):
        img = np.full((60, 80, 3), (200, 200, 200), dtype=np.uint8)  # backdrop
        img[20:45, 30 + i:55 + i] = (60, 20, 20)                     # subject
        p = tmp_path / f"frame_{i:04d}.png"
        Image.fromarray(img).save(p)
        frames.append(p)
    mattes = chroma_key_mattes(frames, tmp_path / "out", feather_px=0.0)
    assert len(mattes) == 3
    m = np.asarray(Image.open(mattes[0]))
    assert m[30, 40] > 200          # subject opaque
    assert m[5, 5] < 30             # backdrop keyed out
    assert m[30, 40 + 0] >= m[5, 5]


def test_actor_prompt_default_mentions_backdrop():
    assert "backdrop" in ACTOR_PROMPT_DEFAULT
    assert "no camera movement" in ACTOR_PROMPT_DEFAULT
