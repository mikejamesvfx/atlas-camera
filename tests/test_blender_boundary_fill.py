"""Contract tests for the scoped, topology-preserving Blender fill path."""
from __future__ import annotations

import json

import numpy as np
import pytest


def _grid_with_centre_hole():
    """A 3 x 3 quad sheet whose centre cell is deliberately missing."""
    vertices = np.array(
        [[x, y, 0.0] for y in range(4) for x in range(4)], dtype=np.float64)
    uvs = np.array(
        [[x / 3.0, y / 3.0] for y in range(4) for x in range(4)],
        dtype=np.float64)
    faces = []
    for y in range(3):
        for x in range(3):
            if (x, y) == (1, 1):
                continue
            a = y * 4 + x
            b, c, d = a + 1, a + 4, a + 5
            faces.extend(((a, b, d), (a, d, c)))
    return vertices, np.asarray(faces, dtype=np.int64), uvs


def test_mask_selects_only_the_interior_loop_not_the_image_perimeter():
    from atlas_camera.blender.boundary_fill import select_masked_interior_loops

    _, faces, uvs = _grid_with_centre_hole()
    mask = np.zeros((12, 12), dtype=np.float32)
    mask[4:8, 4:8] = 1.0

    loops, report = select_masked_interior_loops(
        faces, uvs, mask, image_width=12, image_height=12,
        max_hole_edges=8,
    )

    assert len(loops) == 1
    assert set(loops[0]) == {5, 6, 9, 10}
    assert report["perimeter_loops"] == 1
    assert report["matched_loops"] == 1


def test_empty_mask_does_not_fall_back_to_filling_every_interior_loop():
    from atlas_camera.blender.boundary_fill import select_masked_interior_loops

    _, faces, uvs = _grid_with_centre_hole()
    loops, report = select_masked_interior_loops(
        faces, uvs, np.zeros((12, 12), dtype=np.float32),
        image_width=12, image_height=12, max_hole_edges=8,
    )

    assert loops == []
    assert report["masked_pixels"] == 0
    assert report["matched_loops"] == 0


def test_driver_accepts_added_faces_but_refuses_to_move_existing_vertices(
        monkeypatch, tmp_path):
    import atlas_camera.blender.boundary_fill as boundary_fill

    vertices, faces, _ = _grid_with_centre_hole()

    def fake_run_recipe(recipe_name, exchange_dir, **_kwargs):
        assert recipe_name == "boundary_fill.py"
        params = json.loads((exchange_dir / "params.json").read_text())
        assert params["backend"] == "native"
        assert params["selected_loops"] == [[5, 6, 10, 9]]
        with np.load(exchange_dir / "in.npz") as data:
            out_v = data["patch_vertices"]
            out_f = np.vstack((data["patch_faces"], [[5, 6, 10]]))
        np.savez(exchange_dir / "out.npz", vertices=out_v, faces=out_f)
        return {"faces_created": 1}

    monkeypatch.setattr(boundary_fill, "run_recipe", fake_run_recipe)
    got = boundary_fill.fill_selected_boundary_loops(
        vertices, faces, [[5, 6, 10, 9]], backend="native",
        exchange_dir=tmp_path,
    )

    assert np.array_equal(got["vertices"], vertices)
    assert len(got["faces"]) == len(faces) + 1
    assert got["report"]["faces_created"] == 1

    def fake_moved_recipe(_recipe_name, exchange_dir, **_kwargs):
        with np.load(exchange_dir / "in.npz") as data:
            out_v = data["patch_vertices"].copy()
            out_f = data["patch_faces"]
        out_v[0, 0] += 0.01
        np.savez(exchange_dir / "out.npz", vertices=out_v, faces=out_f)
        return {}

    monkeypatch.setattr(boundary_fill, "run_recipe", fake_moved_recipe)
    with pytest.raises(RuntimeError, match="altered 1 existing vertex"):
        boundary_fill.fill_selected_boundary_loops(
            vertices, faces, [[5, 6, 10, 9]], backend="native",
            exchange_dir=tmp_path / "moved",
        )
