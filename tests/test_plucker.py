import math

import numpy as np
import pytest

from atlas_camera.comfy.plucker import plucker_embedding, ray_map


def sample(position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0), focal=35.0):
    return {
        "position": list(position),
        "rotation": list(rotation),
        "focalLengthMm": focal,
        "focusDistanceM": 4.0,
        "tStop": 2.8,
        "filmback": {"name": "Super 35", "widthMm": 24.89, "heightMm": 18.66},
    }


def test_shape_is_frames_height_width_six():
    rays = ray_map([sample(), sample()], width=16, height=9)
    assert rays.shape == (2, 9, 16, 6)


def test_identity_camera_centre_pixel_looks_down_minus_z():
    # An odd resolution puts a pixel exactly on the optical axis.
    rays = ray_map([sample()], width=9, height=9)
    centre = rays[0, 4, 4, 3:]
    assert np.allclose(centre, [0.0, 0.0, -1.0], atol=1e-6)


def test_all_directions_are_unit_length():
    rays = ray_map([sample()], width=16, height=9)
    lengths = np.linalg.norm(rays[..., 3:], axis=-1)
    assert np.allclose(lengths, 1.0, atol=1e-6)


def test_origin_channel_is_the_sample_position_untouched():
    # Rule 5's guard: this module converts nothing. What went in comes out.
    rays = ray_map([sample(position=(1.5, -2.25, 3.125))], width=4, height=4)
    assert np.allclose(rays[0, :, :, :3], [1.5, -2.25, 3.125], atol=0.0)


def test_horizontal_extent_matches_the_lens_fov():
    # Half the horizontal FOV, measured off the outermost pixel centre, must
    # agree with 2*atan(w/2f) scaled by how far that pixel centre sits from
    # the frame edge. Checked against app/director/camera/lens.ts::horizontalFov.
    width, height, focal = 8, 8, 35.0
    sensor = 24.89
    rays = ray_map([sample(focal=focal)], width=width, height=height)
    rightmost = rays[0, 4, width - 1, 3:]
    x_ndc = ((width - 1) + 0.5) / width * 2.0 - 1.0
    expected = math.atan2(x_ndc * sensor / 2.0, focal)
    measured = math.atan2(rightmost[0], -rightmost[2])
    assert measured == pytest.approx(expected, abs=1e-6)


def test_yaw_ninety_degrees_about_y_points_down_minus_x():
    # Right-handed +90 deg about Y maps (0,0,-1) to (-1,0,0).
    half = math.radians(90.0) / 2.0
    rotation = (0.0, math.sin(half), 0.0, math.cos(half))
    rays = ray_map([sample(rotation=rotation)], width=9, height=9)
    assert np.allclose(rays[0, 4, 4, 3:], [-1.0, 0.0, 0.0], atol=1e-6)


def test_moment_is_zero_for_a_camera_at_the_origin():
    rays = ray_map([sample()], width=8, height=8)
    embedding = plucker_embedding(rays)
    assert np.allclose(embedding[..., :3], 0.0, atol=1e-6)
    assert np.allclose(embedding[..., 3:], rays[..., 3:], atol=0.0)


def test_moment_is_perpendicular_to_direction():
    rays = ray_map([sample(position=(2.0, 1.0, -3.0))], width=8, height=8)
    embedding = plucker_embedding(rays)
    dots = np.einsum("...i,...i->...", embedding[..., :3], embedding[..., 3:])
    assert np.allclose(dots, 0.0, atol=1e-6)


def test_rejects_a_sample_missing_its_filmback():
    bad = sample()
    del bad["filmback"]
    with pytest.raises(KeyError):
        ray_map([bad], width=4, height=4)
