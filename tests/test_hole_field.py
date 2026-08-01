"""One module owns "a hole": the lattice, the components, and island identity.

Before this module there were two lattice recoveries with two refusal protocols,
two component labellers, and an island identity that `path_hole_repair` joined
back to `planar_hole_patch`'s rejections through a coincidence of anchor cells —
documented in a comment reading "both modules label the same lattice, so anchors
coincide". These tests make that invariant something the code holds, not
something a comment claims.
"""
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.hole_field import (  # noqa: E402
    HoleField,
    LatticeError,
    components,
    recover_lattice,
)


def _lattice_mesh(n=4):
    """A perfect n x n relief lattice: every cell present, two triangles each."""
    verts, uvs = [], []
    for r in range(n):
        for c in range(n):
            verts.append([float(c), float(-r), 10.0])
            uvs.append([c / (n - 1), 1.0 - r / (n - 1)])
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            a, b = r * n + c, r * n + c + 1
            d, e = (r + 1) * n + c, (r + 1) * n + c + 1
            faces.append([a, b, d])
            faces.append([b, e, d])
    return SimpleNamespace(
        vertices=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        uvs=np.asarray(uvs, dtype=np.float64),
    )


# ---------------------------------------------------------------- components

def test_four_connectivity_keeps_diagonal_touches_apart():
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    assert len(components(mask)) == 2


def test_eight_connectivity_joins_diagonal_touches():
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    assert len(components(mask, connectivity=8)) == 1


def test_an_empty_mask_has_no_components():
    assert components(np.zeros((4, 4), dtype=bool)) == []


def test_a_component_holds_every_cell_of_its_region():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = mask[1, 2] = mask[2, 1] = True
    (only,) = components(mask)
    assert only == {(1, 1), (1, 2), (2, 1)}


# ---------------------------------------------------------------- HoleField

def _two_island_mask():
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 0] = True                                   # 1 cell,  anchor (0,0)
    mask[3, 3] = mask[3, 4] = mask[4, 3] = True         # 3 cells, anchor (3,3)
    return mask


def test_island_ids_start_at_one_and_run_smallest_first():
    """The order is the one `path_hole_repair` already assigned — by size, then
    by top-left cell. Changing it would renumber every reported island."""
    field = HoleField.from_cell_mask(_two_island_mask())
    assert [i.id for i in field.islands] == [1, 2]
    assert [i.size for i in field.islands] == [1, 3]


def test_the_anchor_is_the_top_left_cell_of_the_island():
    field = HoleField.from_cell_mask(_two_island_mask())
    assert [i.anchor for i in field.islands] == [(0, 0), (3, 3)]


def test_the_id_map_is_zero_off_the_islands():
    field = HoleField.from_cell_mask(_two_island_mask())
    assert field.id_map[5, 5] == 0
    assert field.id_map[0, 0] == 1
    assert field.id_map[4, 3] == 2


def test_id_at_reads_the_map_and_is_zero_outside():
    field = HoleField.from_cell_mask(_two_island_mask())
    assert field.id_at((3, 4)) == 2
    assert field.id_at((5, 5)) == 0
    assert field.id_at((99, 99)) == 0


def test_id_at_an_anchor_recovers_that_island():
    """This is exactly the join `path_hole_repair` performs against a fit
    rejection's anchor cell."""
    field = HoleField.from_cell_mask(_two_island_mask())
    for island in field.islands:
        assert field.id_at(island.anchor) == island.id


def test_labelling_is_stable_across_rebuilds():
    a = HoleField.from_cell_mask(_two_island_mask())
    b = HoleField.from_cell_mask(_two_island_mask())
    assert [(i.id, i.anchor) for i in a.islands] == [(i.id, i.anchor) for i in b.islands]


def test_island_lookup_by_id():
    field = HoleField.from_cell_mask(_two_island_mask())
    assert field.island(2).size == 3
    assert field.island(99) is None


def test_an_empty_mask_yields_no_islands():
    field = HoleField.from_cell_mask(np.zeros((4, 4), dtype=bool))
    assert field.islands == []
    assert int(field.id_map.max()) == 0


# ------------------------------------------------------------------ lattice

def test_recover_lattice_maps_a_perfect_grid():
    lat = recover_lattice(_lattice_mesh(4), 4, 4)
    assert lat["coverage"].shape == (3, 3)
    assert int(lat["coverage"].min()) == 2          # two triangles per cell


def test_a_lattice_error_is_still_a_value_error():
    """Callers catch ValueError today; the named class must not escape them."""
    assert issubclass(LatticeError, ValueError)


def test_off_lattice_uvs_are_refused_by_name():
    mesh = _lattice_mesh(4)
    mesh.uvs = np.asarray(
        np.random.default_rng(0).random((len(mesh.vertices), 2)), dtype=np.float64
    )
    with pytest.raises(LatticeError):
        recover_lattice(mesh, 4, 4)


def test_an_empty_mesh_is_refused_by_name():
    empty = SimpleNamespace(
        vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64),
        uvs=np.zeros((0, 2)),
    )
    with pytest.raises(LatticeError):
        recover_lattice(empty, 4, 4)


# ------------------------------------------------- shared identity contract

def _torn_mesh():
    """A relief mesh with one enclosed hole, plus its camera numbers."""
    from atlas_camera.core.relief_mesh import build_relief_mesh

    w = h = 65
    view = (
        (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0),
    )
    depth = np.full((h, w), 5.0, dtype=np.float32)
    exclusion = np.zeros((h, w), dtype=bool)
    exclusion[27:38, 27:38] = True
    mesh = build_relief_mesh(
        depth, view_matrix=view, fx=80.0, fy=80.0, cx=32.0, cy=32.0,
        grid_long_edge=w, depth_edge_rel=5.0, max_edge_factor=0.0,
        floor_clamp=None, exclude_mask=exclusion, apply_sky_heuristic=False,
        quad_coherence=True,
    )
    return mesh, dict(view_matrix=view, fx=80.0, fy=80.0, cx=32.0, cy=32.0,
                      image_width=w, image_height=h)


def test_patch_records_name_the_island_they_belong_to():
    """`path_hole_repair` used to join a fit rejection back to its island by
    looking its anchor cell up in a separately-built label map, relying on the
    two labellings agreeing. The record now carries the id itself."""
    from atlas_camera.core.planar_hole_patch import (
        PlanarHolePatchConfig,
        patch_planar_holes,
    )

    mesh, cam = _torn_mesh()
    _patched, _remaining, report = patch_planar_holes(
        mesh, mesh.hole_mask,
        config=PlanarHolePatchConfig(max_hole_fraction=0.001), **cam
    )
    records = list(report.get("rejected") or []) + list(report.get("filled") or [])
    assert records, "the fixture must produce at least one island record"
    for record in records:
        assert record["island_id"] > 0
        assert tuple(record["anchor_cell"]) is not None


def test_a_shared_hole_field_gives_the_same_ids_to_both_engines():
    from atlas_camera.core.planar_hole_patch import (
        PlanarHolePatchConfig,
        patch_planar_holes,
    )

    mesh, cam = _torn_mesh()
    _p, _r, report = patch_planar_holes(
        mesh, mesh.hole_mask,
        config=PlanarHolePatchConfig(max_hole_fraction=0.001), **cam
    )
    field = report["hole_field"]
    for record in list(report.get("rejected") or []):
        anchor = tuple(int(v) for v in record["anchor_cell"])
        assert field.id_at(anchor) == record["island_id"]


def test_a_hole_field_of_the_wrong_shape_is_refused_loudly():
    """Silently mislabelling is the failure this whole module exists to stop."""
    from atlas_camera.core.planar_hole_patch import patch_planar_holes

    mesh, cam = _torn_mesh()
    wrong = HoleField.from_cell_mask(np.zeros((3, 3), dtype=bool))
    with pytest.raises(ValueError, match="hole field"):
        patch_planar_holes(mesh, mesh.hole_mask, hole_field=wrong, **cam)


def test_planar_hole_patch_keeps_its_private_names_working():
    """`path_hole_repair` imported these two privates across a module boundary;
    they must keep resolving while callers migrate to the public names."""
    from atlas_camera.core import planar_hole_patch as php

    assert php._components is components
    assert php._recover_lattice is recover_lattice
