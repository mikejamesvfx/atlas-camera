"""AtlasPlateLayer — any plate on any PROXY_ROLE geometry as a ProjectionSource."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from atlas_camera.comfy.nodes_inpaint import AtlasPlateLayer  # noqa: E402
from atlas_camera.core.proxy_geometry import PROXY_ROLE  # noqa: E402
from atlas_camera.core.schema import AtlasProxyPrimitive  # noqa: E402

from test_blender_measured_bridge import _box, _solve  # noqa: E402


def _with_geo():
    s = _solve()                       # has drawn_plane_01 (viewport_polygon)
    v, f = _box()
    for name, src in (("agent_water", "blender_import"), ("massing_ground_plane", "blender_massing"),
                      ("projection_relief_mesh", "depth_relief_mesh")):
        s.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
            name=name, primitive_type="mesh", material="atlas_projection_proxy",
            metadata={"role": PROXY_ROLE, "source": src, "n_vertices": 8, "n_faces": 12,
                      "vertices": v.reshape(-1).tolist(), "faces": f.reshape(-1).tolist(),
                      "uvs": [0.0] * 16, "edge_risk": [], "ribbon_t": []}))
    return s


def test_default_filter_takes_blender_and_drawn_geometry_and_moves_it():
    s = _with_geo()
    plate = torch.rand(1, 60, 80, 3)
    out, report = AtlasPlateLayer().add_layer(s, plate, name="cp")
    assert len(out.projection_sources) == 1
    src = out.projection_sources[0]
    names = sorted(p.name for p in src.proxy_geometry)
    assert names == ["agent_water", "drawn_plane_01", "massing_ground_plane"]
    assert src.metadata["projection_mode"] == "clean_plate"
    assert src.image_b64.startswith("data:image/jpeg")
    # moved out of the primary scene; the relief mesh stays
    left = [p.name for p in out.projection_scene.proxy_geometry]
    assert left == ["projection_relief_mesh"]
    assert "3 removed" in report
    # same camera as the primary
    assert src.camera.extrinsics.camera_view_matrix == out.camera.extrinsics.camera_view_matrix
    # input untouched
    assert len(s.projection_scene.proxy_geometry) == 4 and not s.projection_sources


def test_filter_by_name_prefix_and_geometry_from_and_no_move():
    s = _with_geo()
    other = _with_geo()
    plate = torch.rand(1, 60, 80, 3)
    out, report = AtlasPlateLayer().add_layer(
        s, plate, geometry_from=other, geometry_filter="agent_", move_from_primary=False, name="w")
    src = out.projection_sources[0]
    assert [p.name for p in src.proxy_geometry] == ["agent_water"]
    assert len(out.projection_scene.proxy_geometry) == 4        # nothing removed
    assert src.metadata["geometry_from"] == "geometry_from"


def test_star_takes_everything_and_empty_selection_passes_through():
    s = _with_geo()
    plate = torch.rand(1, 60, 80, 3)
    out, _ = AtlasPlateLayer().add_layer(s, plate, geometry_filter="*")
    assert len(out.projection_sources[0].proxy_geometry) == 4
    out2, report = AtlasPlateLayer().add_layer(s, plate, geometry_filter="nothing_here")
    assert not out2.projection_sources and "no PROXY_ROLE primitive matches" in report


def test_two_layers_two_plates():
    s = _with_geo()
    p1, p2 = torch.rand(1, 60, 80, 3), torch.rand(1, 60, 80, 3)
    out, _ = AtlasPlateLayer().add_layer(s, p1, geometry_filter="agent_", name="water_plate")
    out, _ = AtlasPlateLayer().add_layer(out, p2, geometry_filter="drawn_plane_", name="wall_plate", priority=7)
    assert [x.name for x in out.projection_sources] == ["water_plate", "wall_plate"]
    assert out.projection_sources[1].priority == 7.0


def test_moving_a_drawn_plane_changes_the_blender_seed_fingerprint():
    """CROSS-MODULE: solve_seed_fingerprint hashes the primary primitive
    roster, so AtlasPlateLayer(move_from_primary=True) on a DRAWN plane
    invalidates a Blender seed built earlier in the graph. Blender-sourced
    meshes are already exempt, which is why the water/hillside case is safe.

    Pinned because the refusal surfaces at AtlasBlenderImportMeshes and reads
    as if the massing node were at fault; both refusal messages now name this.
    """
    from atlas_camera.blender.measured import solve_seed_fingerprint

    before = solve_seed_fingerprint(_with_geo())
    plate = torch.rand(1, 60, 80, 3)

    only_blender, _ = AtlasPlateLayer().add_layer(
        _with_geo(), plate, geometry_filter="agent_water", name="w")
    assert solve_seed_fingerprint(only_blender) == before,         "a Blender-sourced mesh is exempt — the round-trip must stay valid"

    with_drawn, _ = AtlasPlateLayer().add_layer(
        _with_geo(), plate, geometry_filter="drawn_plane_01", name="d")
    assert solve_seed_fingerprint(with_drawn) != before,         "a drawn plane is part of the seeded roster — moving it IS a change"


def test_both_refusals_name_the_plate_layer_cause():
    """The message is the fix: without it the artist re-runs massing forever."""
    import inspect
    from atlas_camera.comfy import nodes_agent, nodes_geometry

    for mod in (nodes_geometry, nodes_agent):
        src = inspect.getsource(mod)
        assert "move_from_primary" in src, (
            f"{mod.__name__} lost the stale-seed hint naming AtlasPlateLayer")
