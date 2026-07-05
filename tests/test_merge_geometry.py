"""Tests for AtlasMergeGeometry — the explicit combinator for two
independently-derived solves' geometry (the Nuke-Merge-node equivalent for
AtlasDeriveWalls/AtlasDeriveReliefMesh/etc). Pure Python, no depth/geometry
extraction involved — just hand-built AtlasProxyPrimitive lists, so this
needs only the core schema.
"""

from atlas_camera.comfy.nodes import AtlasMergeGeometry, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from atlas_camera.core.proxy_geometry import PROXY_ROLE
from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasProxyPrimitive,
    AtlasSolve,
    LatentCamera,
)


def _prim(name, primitive_type="plane"):
    return AtlasProxyPrimitive(
        name=name, primitive_type=primitive_type,
        metadata={"role": PROXY_ROLE, "source": "test"},
    )


def _solve(camera_x, *prims):
    intr = AtlasIntrinsics(image_width=512, image_height=512, focal_length_mm=35.0)
    extr = AtlasExtrinsics(camera_position=(camera_x, 0.0, 0.0))
    solve = AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))
    solve.projection_scene.proxy_geometry = list(prims)
    return solve


def test_node_registered_and_return_types():
    assert NODE_CLASS_MAPPINGS["AtlasMergeGeometry"] is AtlasMergeGeometry
    assert "AtlasMergeGeometry" in NODE_DISPLAY_NAME_MAPPINGS
    assert AtlasMergeGeometry.RETURN_TYPES == ("ATLAS_SOLVE",)


def test_merge_combines_both_solves_geometry():
    solve_a = _solve(0.0, _prim("projection_wall_01"), _prim("projection_backdrop"))
    solve_b = _solve(0.0, _prim("projection_relief_mesh", "mesh"), _prim("projection_backdrop"))

    (out,) = AtlasMergeGeometry().merge(solve_a, solve_b)

    names = [p.name for p in out.projection_scene.proxy_geometry]
    assert "projection_wall_01" in names
    assert "projection_relief_mesh" in names


def test_merge_deduplicates_backdrop_keeping_solve_a():
    solve_a = _solve(0.0, _prim("projection_wall_01"), _prim("projection_backdrop"))
    solve_b = _solve(0.0, _prim("projection_relief_mesh", "mesh"), _prim("projection_backdrop"))

    (out,) = AtlasMergeGeometry().merge(solve_a, solve_b)

    backdrops = [p for p in out.projection_scene.proxy_geometry if p.name == "projection_backdrop"]
    assert len(backdrops) == 1


def test_merge_keeps_only_backdrop_when_only_one_side_has_it():
    solve_a = _solve(0.0, _prim("projection_wall_01"))  # no backdrop
    solve_b = _solve(0.0, _prim("projection_relief_mesh", "mesh"), _prim("projection_backdrop"))

    (out,) = AtlasMergeGeometry().merge(solve_a, solve_b)

    backdrops = [p for p in out.projection_scene.proxy_geometry if p.name == "projection_backdrop"]
    assert len(backdrops) == 1


def test_merge_uses_solve_a_camera():
    solve_a = _solve(5.0, _prim("projection_wall_01"))
    solve_b = _solve(99.0, _prim("projection_relief_mesh", "mesh"))

    (out,) = AtlasMergeGeometry().merge(solve_a, solve_b)

    assert out.camera.extrinsics.camera_position[0] == 5.0


def test_merge_does_not_mutate_either_input_solve():
    solve_a = _solve(0.0, _prim("projection_wall_01"))
    solve_b = _solve(0.0, _prim("projection_relief_mesh", "mesh"))
    a_before = len(solve_a.projection_scene.proxy_geometry)
    b_before = len(solve_b.projection_scene.proxy_geometry)

    AtlasMergeGeometry().merge(solve_a, solve_b)

    assert len(solve_a.projection_scene.proxy_geometry) == a_before
    assert len(solve_b.projection_scene.proxy_geometry) == b_before


def test_merge_does_not_duplicate_pass_through_non_proxy_role_geometry():
    # Regression test for a bug found via live end-to-end verification (not
    # unit tests): every solve used to start with a placeholder "ground_plane"
    # entry (role="ground", not PROXY_ROLE) from
    # projection_scene.create_default_projection_scene() — every derive node
    # passed it through untouched, so BOTH solve_a and solve_b inherited
    # their own copy of the exact same pre-existing entry. Merging must only
    # take solve_b's PROXY_ROLE geometry (what its own derive node actually
    # added), never solve_b's full list, or a shared entry like this gets
    # duplicated. The placeholder itself has since been removed (it had no
    # consumer and confusingly collided in name with the real, rendered
    # "projection_ground" primitive) — kept as a synthetic, hand-built
    # fixture here so this guard stays covered regardless.
    ground = AtlasProxyPrimitive(
        name="ground_plane", primitive_type="plane", metadata={"role": "ground", "up_axis": "Y"})
    solve_a = _solve(0.0, ground, _prim("projection_wall_01"))
    solve_b = _solve(0.0, ground, _prim("projection_relief_mesh", "mesh"))

    (out,) = AtlasMergeGeometry().merge(solve_a, solve_b)

    ground_entries = [p for p in out.projection_scene.proxy_geometry if p.name == "ground_plane"]
    assert len(ground_entries) == 1


def test_merge_records_debug_metadata():
    solve_a = _solve(0.0, _prim("projection_wall_01"))
    solve_b = _solve(0.0, _prim("projection_relief_mesh", "mesh"), _prim("projection_backdrop"))

    (out,) = AtlasMergeGeometry().merge(solve_a, solve_b)

    meta = out.projection_scene.debug_metadata["proxy_derivation_merge"]
    assert meta["solve_a_prims"] == 1
    assert meta["solve_b_prims_merged"] == 2
    assert meta["merged_prims_total"] == 3
