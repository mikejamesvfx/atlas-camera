"""Tests for the composable geometry-derivation nodes (AtlasDeriveReliefMesh,
AtlasDeriveWalls, AtlasDeriveTowersSpires, AtlasDeriveRoofsFacades,
AtlasDeriveInteriorRoom) — each a thin wrapper around an already-tested core
extraction function, taking a pre-computed ATLAS_DEPTH_MAP instead of running
its own depth estimation (see AtlasDepthMap). Uses a self-contained analytic
ground+wall depth map (same convention as test_proxy_geometry.py: level
camera at (0, h, 0), identity rotation) rather than a real photo, so these
run with only numpy — no [neural] extra or model download needed.
"""

import numpy as np
import pytest

from atlas_camera.comfy.nodes import (
    NODE_CLASS_MAPPINGS,
    AtlasDeriveInteriorRoom,
    AtlasDeriveProjectionGeometry,
    AtlasDeriveReliefMesh,
    AtlasDeriveRoofsFacades,
    AtlasDeriveTowersSpires,
    AtlasDeriveWalls,
)
from atlas_camera.core.proxy_geometry import PROXY_ROLE
from atlas_camera.core.schema import AtlasExtrinsics, AtlasIntrinsics, AtlasSolve, LatentCamera
from atlas_camera.inference.depth_estimator import DepthResult

W = H = 256
FX = FY = 250.0
CX = CY = 128.0
SKY = 60.0
CAM_HEIGHT = 1.6


def _view_matrix(h):
    """Level camera at (0, h, 0), identity rotation — world->cam translation only."""
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, -h),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _room_depth(h=CAM_HEIGHT, wall_z=-8.0, wall_h=3.0):
    """Ground plane (Y=0) + one fronto-parallel wall at world z=wall_z, height wall_h."""
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dx = (uu - CX) / FX
    dy = -(vv - CY) / FY  # cam y-up; image v grows downward

    depth = np.full((H, W), SKY)

    # Ground: t where h + dy*t == 0 (looking downward only).
    t_ground = np.full((H, W), np.inf)
    looking_down = dy < -1e-6
    t_ground[looking_down] = -h / dy[looking_down]

    # Wall: world z = -t == wall_z -> t = -wall_z; visible where 0 <= y <= wall_h.
    t_wall = np.full((H, W), np.inf)
    t = -wall_z
    y_at = h + dy * t
    visible = (y_at >= 0.0) & (y_at <= wall_h)
    t_wall[visible] = t

    stacked = np.stack([
        depth,
        np.where(np.isfinite(t_ground), t_ground, SKY),
        np.where(np.isfinite(t_wall), t_wall, SKY),
    ])
    return stacked.min(axis=0).astype(np.float32)


def _solve(h=CAM_HEIGHT):
    intr = AtlasIntrinsics(
        image_width=W, image_height=H, focal_length_mm=35.0, sensor_width_mm=36.0,
        fx_px=FX, fy_px=FY, cx_px=CX, cy_px=CY,
    )
    extr = AtlasExtrinsics(camera_view_matrix=_view_matrix(h))
    return AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))


def _depth_result(depth_map):
    return DepthResult(
        depth=depth_map, is_metric=True, model_id="fake",
        image_width=W, image_height=H, near=float(depth_map.min()), far=float(depth_map.max()),
    )


def _proxy_names(out_solve):
    return [p.name for p in out_solve.projection_scene.proxy_geometry]


def _all_tagged(out_solve):
    return all((p.metadata or {}).get("role") == PROXY_ROLE
               for p in out_solve.projection_scene.proxy_geometry)


def test_all_five_nodes_registered():
    # AtlasDeriveReliefMesh additionally exposes the relief mesh's own
    # hole_mask (see relief_mesh.ReliefMesh.hole_mask); the other four are
    # primitive-only derivers and have no mesh to have holes in.
    for name in ["AtlasDeriveWalls", "AtlasDeriveTowersSpires",
                 "AtlasDeriveRoofsFacades", "AtlasDeriveInteriorRoom"]:
        assert name in NODE_CLASS_MAPPINGS
        # `report` was APPENDED (slot 1) so these nodes can explain a dropped
        # or invented backdrop; slot 0 is unchanged, so saved links survive.
        assert NODE_CLASS_MAPPINGS[name].RETURN_TYPES == ("ATLAS_SOLVE", "STRING")
        assert NODE_CLASS_MAPPINGS[name].RETURN_NAMES == ("solve", "report")
    assert "AtlasDeriveReliefMesh" in NODE_CLASS_MAPPINGS
    # Same append, 2026-08-17: the relief pair were left behind when the four
    # above grew a report, and their no-focal path returned an all-ZERO
    # hole_mask — full coverage, the answer a PERFECT mesh gives.
    for name in ("AtlasDeriveReliefMesh", "AtlasDeriveProjectionGeometry"):
        assert NODE_CLASS_MAPPINGS[name].RETURN_TYPES == ("ATLAS_SOLVE", "MASK", "STRING")
        assert NODE_CLASS_MAPPINGS[name].RETURN_NAMES == ("solve", "hole_mask", "report")


def test_relief_mesh_produces_mesh_and_backdrop():
    pytest.importorskip("torch")  # hole_mask output tensor needs torch
    solve = _solve()
    depth = _depth_result(_room_depth())
    out, hole_mask, _rep = AtlasDeriveReliefMesh().derive(solve, depth, relief_grid=32)

    names = _proxy_names(out)
    assert "projection_relief_mesh" in names
    assert "projection_backdrop" in names
    assert _all_tagged(out)
    assert tuple(hole_mask.shape) == (1, H, W)


def test_relief_quality_overrides_relief_grid():
    pytest.importorskip("torch")  # hole_mask output tensor needs torch
    solve = _solve()
    depth = _depth_result(_room_depth())
    out, _hole_mask, _rep = AtlasDeriveReliefMesh().derive(solve, depth, relief_grid=32, relief_quality="low")
    assert out.projection_scene.debug_metadata["proxy_derivation"]["relief_grid"] == 64


def test_walls_node_finds_ground_and_wall():
    solve = _solve()
    depth = _depth_result(_room_depth())
    out, _rep = AtlasDeriveWalls().derive(solve, depth, max_walls=4, max_objects=0)

    names = _proxy_names(out)
    assert "projection_ground" in names
    assert any(n.startswith("projection_wall_") for n in names)
    assert "projection_backdrop" in names
    assert out.projection_scene.debug_metadata["proxy_derivation"]["primitive_method"] == "azimuth_walls"
    assert _all_tagged(out)


def test_towers_spires_node_finds_ground_and_wall():
    solve = _solve()
    depth = _depth_result(_room_depth())
    out, _rep = AtlasDeriveTowersSpires().derive(solve, depth, max_walls=4, max_objects=0)

    names = _proxy_names(out)
    assert "projection_ground" in names
    assert any(n.startswith("projection_wall_") for n in names)
    assert out.projection_scene.debug_metadata["proxy_derivation"]["primitive_method"] == "vertical_extrusion"


def test_no_focal_length_returns_solve_unchanged():
    intr = AtlasIntrinsics(image_width=W, image_height=H)  # no fx_px
    solve = AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=AtlasExtrinsics()))
    depth = _depth_result(_room_depth())

    out, _rep = AtlasDeriveWalls().derive(solve, depth)
    assert out is solve  # returned unchanged, per the fx<=0 guard


def test_roofs_facades_node_runs_and_tags_output():
    solve = _solve()
    depth = _depth_result(_room_depth())
    out, _rep = AtlasDeriveRoofsFacades().derive(solve, depth, max_planes=8)

    assert out.projection_scene.debug_metadata["proxy_derivation"]["primitive_method"] == "ransac_planes"
    assert _all_tagged(out)


def test_interior_room_node_runs_and_tags_output():
    solve = _solve()
    depth = _depth_result(_room_depth())
    out, _rep = AtlasDeriveInteriorRoom().derive(solve, depth)

    assert out.projection_scene.debug_metadata["proxy_derivation"]["primitive_method"] == "room_cuboid"
    assert _all_tagged(out)


def test_derive_node_does_not_mutate_input_solve():
    solve = _solve()
    depth = _depth_result(_room_depth())
    before = len(solve.projection_scene.proxy_geometry)

    AtlasDeriveWalls().derive(solve, depth)

    assert len(solve.projection_scene.proxy_geometry) == before


def test_layer_nodes_expose_shared_quad_coherence_guard():
    from atlas_camera.comfy.nodes import AtlasCleanPlateLayer, AtlasDepthLayerMask

    assert "quad_coherence" in AtlasCleanPlateLayer.INPUT_TYPES()["optional"]
    assert "quad_coherence" in AtlasDepthLayerMask.INPUT_TYPES()["optional"]


def test_derive_nodes_do_not_expose_redundant_live_fill_widgets():
    """Live repair is a separate legacy-node concern, not a mesh-build dial."""
    widget_names = {
        "live_fill_holes",
        "live_fill_distance_m",
        "live_fill_max_hole_edges",
        "live_fill_edge_sawteeth",
    }
    for node_class in (AtlasDeriveProjectionGeometry, AtlasDeriveReliefMesh):
        assert widget_names.isdisjoint(node_class.INPUT_TYPES()["optional"])


def test_exclude_mask_records_coverage_and_sky_suppression(capsys):
    """An explicit exclude_mask REPLACES the internal sky heuristic rather than
    adding to it, and until now nothing said so.

    A mask wired with the wrong polarity therefore meshes the sky and still
    reports success — measured live on an 8K plate 2026-07-27, where a
    subject-only exclude produced a fat pass-1 that was largely sky. Record the
    mask's frame coverage (so the polarity is readable afterwards) and say on
    the console when the heuristic is the thing being silenced.
    """
    torch = pytest.importorskip("torch")
    solve = _solve()
    depth = _depth_result(_room_depth())
    h, w = _room_depth().shape

    band = torch.zeros(1, h, w, dtype=torch.float32)
    band[:, : h // 2, :] = 1.0
    out, _, _rep = AtlasDeriveReliefMesh().derive(
        solve, depth, relief_grid=32, exclude_mask=band, sky_heuristic=True)
    debug = out.projection_scene.debug_metadata["proxy_derivation"]
    stats = debug["exclude_mask"]
    assert stats["frame_fraction"] == pytest.approx(0.5, abs=0.02)
    assert stats["sky_heuristic_suppressed"] is True
    assert "REPLACES the internal sky heuristic" in capsys.readouterr().out

    # sky_heuristic already off: the mask replaces nothing, so no console noise.
    out_off, _, _rep = AtlasDeriveReliefMesh().derive(
        solve, depth, relief_grid=32, exclude_mask=band, sky_heuristic=False)
    off = out_off.projection_scene.debug_metadata[
        "proxy_derivation"]["exclude_mask"]
    assert off["sky_heuristic_suppressed"] is False
    assert "REPLACES" not in capsys.readouterr().out

    # No mask wired: the block is absent entirely, so "unwired" stays
    # distinguishable from "wired but empty".
    plain, _, _rep = AtlasDeriveReliefMesh().derive(solve, depth, relief_grid=32)
    assert "exclude_mask" not in plain.projection_scene.debug_metadata[
        "proxy_derivation"]


def test_live_hole_fill_disabled_by_default():
    pytest.importorskip("torch")
    solve = _solve()
    depth = _depth_result(_room_depth())
    out, _, _rep = AtlasDeriveReliefMesh().derive(solve, depth, relief_grid=32)
    meta = out.projection_scene.debug_metadata["proxy_derivation"]["relief_mesh"]
    assert "live_hole_fill" not in meta


def test_atlas_derive_projection_geometry_signature_matches_input_types():
    """ComfyUI passes inputs in ``INPUT_TYPES`` order.  A drift between the
    function signature and the declared order makes saved workflows load
    values into the wrong parameters, exactly the bug fixed for ``AtlasInput``
    in the same commit series.  This pins the full contract (widgets + sockets)."""
    import inspect
    sig = inspect.signature(AtlasDeriveProjectionGeometry.derive)
    all_params = [n for n in list(sig.parameters.keys())[1:]
                 if not n.startswith("_")]
    it = AtlasDeriveProjectionGeometry.INPUT_TYPES()
    input_names = []
    for sec in ("required", "optional"):
        input_names.extend(it.get(sec, {}).keys())
    assert all_params == input_names, f"params {all_params} inputs {input_names}"


def test_atlas_derive_relief_mesh_signature_matches_input_types():
    """``exclude_mask`` and ``outlier_mask`` are MASK sockets, but they still
    participate in ComfyUI's positional input order and must align."""
    import inspect
    sig = inspect.signature(AtlasDeriveReliefMesh.derive)
    all_params = [n for n in list(sig.parameters.keys())[1:]
                 if not n.startswith("_")]
    it = AtlasDeriveReliefMesh.INPUT_TYPES()
    input_names = []
    for sec in ("required", "optional"):
        input_names.extend(it.get(sec, {}).keys())
    assert all_params == input_names, f"params {all_params} inputs {input_names}"


# ---------------------------------------------------------------------------
# backdrop provenance — the "a plane appeared out of nowhere" fix
# ---------------------------------------------------------------------------

def _all_invalid_depth():
    return _depth_result(np.zeros((H, W), dtype=np.float32))


def _prim_names(out):
    return [p.name for p in out.projection_scene.proxy_geometry]


def test_assumed_backdrop_is_dropped_by_default():
    """With no valid depth there is nothing to place a backdrop against, so
    the old code emitted a plane at a hardcoded 60 m with invented extents.
    That invented geometry is what shows up as an unexplained plane."""
    pytest.importorskip("torch")
    out, report = AtlasDeriveWalls().derive(_solve(), _all_invalid_depth())
    assert "projection_backdrop" not in _prim_names(out)
    assert "ASSUMED" in report and "60 m" in report


def test_assumed_backdrop_can_be_restored_explicitly():
    """backdrop='always' is the pre-2026-07-27 behaviour — still available,
    but it must SAY the plane is invented rather than implying it was measured."""
    pytest.importorskip("torch")
    out, report = AtlasDeriveWalls().derive(
        _solve(), _all_invalid_depth(), backdrop="always")
    assert "projection_backdrop" in _prim_names(out)
    assert "KEPT an ASSUMED backdrop" in report


def test_measured_backdrop_survives_the_default():
    """The gate must only drop INVENTED backdrops — a measured one is real
    geometry and has to come through untouched."""
    pytest.importorskip("torch")
    out, report = AtlasDeriveWalls().derive(_solve(), _depth_result(_room_depth()))
    assert "projection_backdrop" in _prim_names(out)
    assert "ASSUMED" not in report


def test_backdrop_never_drops_even_a_measured_one():
    pytest.importorskip("torch")
    out, report = AtlasDeriveWalls().derive(
        _solve(), _depth_result(_room_depth()), backdrop="never")
    assert "projection_backdrop" not in _prim_names(out)
    assert "backdrop=never" in report


def test_backdrop_primitive_records_its_provenance():
    """The metadata is what makes the decision auditable downstream."""
    pytest.importorskip("torch")
    out, _ = AtlasDeriveWalls().derive(_solve(), _depth_result(_room_depth()))
    backdrop = next(p for p in out.projection_scene.proxy_geometry
                    if p.name == "projection_backdrop")
    assert backdrop.metadata["backdrop_depth_source"] == "measured"
    assert backdrop.metadata["backdrop_extents_source"] in ("frustum", "assumed")


def test_backdrop_widget_is_appended_last():
    for cls in (AtlasDeriveWalls, AtlasDeriveTowersSpires,
                AtlasDeriveRoofsFacades, AtlasDeriveInteriorRoom):
        assert list(cls.INPUT_TYPES()["optional"])[-1] == "backdrop"


def test_every_derive_node_refuses_a_focal_less_solve_the_same_way():
    """Characterization pin, written before the four derive bodies were
    collapsed onto one shared implementation. The refusal text is identical in
    all four today only because it was copy-pasted four times; after the
    collapse it is identical because there is one of it."""
    pytest.importorskip("torch")
    from atlas_camera.comfy.nodes_geometry import (
        AtlasDeriveInteriorRoom,
        AtlasDeriveRoofsFacades,
        AtlasDeriveTowersSpires,
    )

    solve = _solve()
    solve.camera.intrinsics.fx_px = 0.0
    depth = _depth_result(_room_depth())
    for node in (AtlasDeriveWalls(), AtlasDeriveTowersSpires(),
                 AtlasDeriveRoofsFacades(), AtlasDeriveInteriorRoom()):
        out, report = node.derive(solve, depth)
        assert out is solve, f"{type(node).__name__} must pass the solve through"
        assert report == ("SKIPPED — solve has no usable focal (fx <= 0); "
                          "geometry unchanged"), type(node).__name__


def test_no_usable_focal_reports_instead_of_silently_passing_through():
    """The old guard was a silent no-op; gate doctrine wants the explanation."""
    pytest.importorskip("torch")
    solve = _solve()
    solve.camera.intrinsics.fx_px = 0.0
    out, report = AtlasDeriveWalls().derive(solve, _depth_result(_room_depth()))
    assert out is solve
    assert "SKIPPED" in report and "focal" in report


def test_relief_derive_without_focal_says_so_and_marks_everything_uncovered():
    """A derive that never ran must not look like a perfect one.

    hole_mask is "where will Project show black". All-ZERO asserts full
    coverage — exactly what a flawless mesh returns — so the old no-focal
    guard was indistinguishable downstream from the best possible success:
    AtlasPlanarHolePatch saw nothing to patch, the inpaint router saw no tears.
    """
    pytest.importorskip("torch")
    solve = _solve()
    solve.camera.intrinsics.fx_px = 0.0
    out, holes, report = AtlasDeriveReliefMesh().derive(
        solve, _depth_result(_room_depth()))
    assert out is solve
    assert "SKIPPED" in report and "focal" in report
    assert float(holes.min()) == 1.0, "every pixel must read as uncovered"


def test_relief_derive_reports_coverage_on_the_happy_path():
    pytest.importorskip("torch")
    out, holes, report = AtlasDeriveReliefMesh().derive(
        _solve(), _depth_result(_room_depth()))
    assert "AtlasDeriveReliefMesh" in report and "covers" in report
    assert out is not None
