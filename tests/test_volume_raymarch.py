"""Ray-marching a distance field into Atlas's layered-ray representation.

The properties pinned here are the ones that make the adapter worth having over
marching cubes: crossings PAIR into single surfaces at the right depth, layers
come out ordered front-to-back, and the output plugs straight into
``core.hidden_geometry`` without translation.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlas_camera.core.hidden_geometry import select_hidden_surface
from atlas_camera.core.volume_raymarch import march_layers

RES = 64
EXTENT = 8.0          # metres, cube
BBOX_MIN = np.array([-4.0, -4.0, 2.0])     # centred laterally, 2..10 m deep
VOXEL = EXTENT / RES


def _empty():
    return np.full((RES, RES, RES), 3.0, dtype=np.float32)


def _add_slab(field, z_metres, half_thickness_voxels=1.0):
    """A wall at a known depth, as an UNSIGNED distance band — i.e. the double
    wall a TUDF really contains, not an idealised single sheet."""
    iz = (z_metres - BBOX_MIN[2]) / VOXEL
    zz = np.arange(RES)[:, None, None]
    d = np.abs(zz - iz)
    band = np.clip(d, 0.0, 3.0).astype(np.float32)
    return np.minimum(field, np.broadcast_to(band, field.shape))


def _march(field, **kw):
    opts = dict(fx=64.0, fy=64.0, cx=32.0, cy=32.0, width=64, height=64,
                threshold=0.5, max_layers=6)
    opts.update(kw)
    return march_layers(field, BBOX_MIN, np.full(3, EXTENT), **opts)


def test_single_slab_yields_one_layer_at_the_right_depth():
    """The whole point: entry and exit of the shell collapse to ONE surface at
    the true depth, not two walls offset by +/- the threshold."""
    field = _add_slab(_empty(), 5.0)
    layers, stats = _march(field)

    centre = layers[32, 32]
    assert centre[0] > 0, "no surface found on the centre ray"
    assert centre[0] == pytest.approx(5.0, abs=2 * VOXEL)
    assert centre[1] == 0.0, "a single slab must not produce a second layer"
    assert stats["rays_with_surface"] > 0


def test_midpoint_removes_the_double_wall_bias():
    """Meshing the level set directly puts samples ~1 voxel off the surface.
    Pairing must land within a fraction of that."""
    field = _add_slab(_empty(), 6.0)
    layers, _ = _march(field)
    hit = layers[..., 0]
    hit = hit[hit > 0]
    err = np.abs(hit.mean() - 6.0)
    assert err < 0.5 * VOXEL, f"midpoint biased by {err / VOXEL:.2f} voxels"


def test_two_slabs_come_back_ordered_front_to_back():
    field = _add_slab(_add_slab(_empty(), 4.0), 8.0)
    layers, stats = _march(field)
    centre = layers[32, 32]
    assert centre[0] == pytest.approx(4.0, abs=2 * VOXEL)
    assert centre[1] == pytest.approx(8.0, abs=2 * VOXEL)
    assert centre[0] < centre[1], "layers must be front-to-back"
    assert stats["mean_layers_per_hit"] >= 1.5


def test_crossings_are_even_which_is_the_double_wall_signature():
    field = _add_slab(_add_slab(_empty(), 4.0), 8.0)
    _, stats = _march(field)
    # Both slabs sit well inside the slab range, so every wall is entered AND
    # left: an odd count would mean a wall was clipped or stepped over.
    assert stats["odd_crossing_fraction"] < 0.05


def test_empty_field_yields_no_layers():
    layers, stats = _march(_empty())
    assert not layers.any()
    assert stats["rays_with_surface"] == 0


def test_output_feeds_select_hidden_surface_unchanged():
    """The adapter's reason to exist: the stack must be consumable by the
    existing calibrated selector with no translation step."""
    field = _add_slab(_add_slab(_empty(), 4.0), 8.0)
    layers, _ = _march(field)
    visible = np.full((64, 64), 4.0)          # the camera sees the near slab

    hidden, valid, stats = select_hidden_surface(layers, visible)

    assert valid.any(), "no hidden surface selected from a two-slab scene"
    assert hidden[valid].mean() == pytest.approx(8.0, abs=0.5)
    assert stats["coverage"] > 0.5


def test_step_must_be_finer_than_a_voxel():
    """A 2-voxel wall can be stepped clean over by a >= 1 voxel step, which
    silently deletes a surface rather than failing."""
    with pytest.raises(ValueError, match="step_voxels"):
        _march(_empty(), step_voxels=1.0)


def test_layer_budget_is_respected():
    field = _empty()
    for z in (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0):
        field = _add_slab(field, z)
    layers, stats = _march(field, max_layers=3)
    assert layers.shape[2] == 3
    assert stats["max_layers_reached"] > 0


def test_bad_field_shape_rejected():
    with pytest.raises(ValueError):
        _march(np.zeros((8, 8), dtype=np.float32))
