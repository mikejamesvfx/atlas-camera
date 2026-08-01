"""Occluded surfaces -> a shooting brief a photographer can act on.

Atlas already knows what is missing; the occlusion graph enumerates it. What
never existed was a way to state it in terms someone holding a camera can use:
what to point at, at what angle, from how far, and how badly it is needed.

The angle is the load-bearing number. A paving slab photographed square looks
nothing like the same slab raking away at 80 degrees, however good the texture —
so `surface_incidence_deg` gets the most attention here.
"""
from __future__ import annotations

import pytest

from atlas_camera.core.occlusion_graph import (
    AtlasOcclusionGraph,
    OcclusionEdge,
    OcclusionNode,
)
from atlas_camera.core.shoot_list import (
    build_shoot_list,
    sampling_px_per_m,
    shoot_project,
    surface_incidence_deg,
)

np = pytest.importorskip("numpy")


class _Intr:
    fx_px = 1000.0


class _Extr:
    camera_position = (0.0, 1.6, 0.0)


class _Cam:
    intrinsics = _Intr()
    extrinsics = _Extr()


class _Solve:
    camera = _Cam()


def _graph(nodes, edges):
    return AtlasOcclusionGraph(version=1, nodes=nodes, edges=edges, notes=[])


def _ground(node_id="ground", d=0.0, depth=(4.0, 12.0)):
    return OcclusionNode(id=node_id, kind="ground", plane={"normal": [0, 1, 0], "d": d},
                         depth_range_m=depth, completion_policy="extend_plane")


# ------------------------------------------------------------ incidence


def test_looking_straight_down_is_zero_incidence():
    """Camera above a floor, looking at the point directly below it."""
    assert surface_incidence_deg((0, 1, 0), (0.0, 5.0, 0.0), (0.0, 0.0, 0.0)) \
        == pytest.approx(0.0, abs=1e-6)


def test_skimming_along_a_surface_is_ninety():
    """A floor viewed from floor level, straight ahead — pure grazing."""
    assert surface_incidence_deg((0, 1, 0), (0.0, 0.0, 0.0), (0.0, 0.0, -10.0)) \
        == pytest.approx(90.0, abs=1e-6)


def test_forty_five_degrees_reads_as_forty_five():
    assert surface_incidence_deg((0, 1, 0), (0.0, 10.0, 0.0), (0.0, 0.0, -10.0)) \
        == pytest.approx(45.0, abs=1e-6)


def test_a_flipped_normal_describes_the_same_surface():
    """A plane fitted 'backwards' must not read as 170 degrees.

    Both orientations describe the same physical surface, and a sign convention
    slipping through would send the photographer to the wrong angle entirely.
    """
    up = surface_incidence_deg((0, 1, 0), (0.0, 10.0, 0.0), (0.0, 0.0, -10.0))
    down = surface_incidence_deg((0, -1, 0), (0.0, 10.0, 0.0), (0.0, 0.0, -10.0))
    assert up == pytest.approx(down, abs=1e-6)


def test_degenerate_inputs_do_not_raise():
    assert surface_incidence_deg((0, 0, 0), (0, 1, 0), (0, 0, 0)) == 0.0
    assert surface_incidence_deg((0, 1, 0), (1, 1, 1), (1, 1, 1)) == 0.0


# ------------------------------------------------------------ sampling


def test_sampling_density_falls_with_distance():
    near = sampling_px_per_m(1000.0, 5.0, 0.0)
    far = sampling_px_per_m(1000.0, 50.0, 0.0)
    assert near == pytest.approx(200.0)
    assert far == pytest.approx(20.0)


def test_grazing_destroys_sampling_density():
    """Why a road running to the horizon carries so little real detail."""
    square = sampling_px_per_m(1000.0, 10.0, 0.0)
    raking = sampling_px_per_m(1000.0, 10.0, 80.0)
    assert raking < square * 0.2


def test_sampling_is_zero_for_nonsense_input():
    assert sampling_px_per_m(1000.0, 0.0, 0.0) == 0.0
    assert sampling_px_per_m(0.0, 10.0, 0.0) == 0.0


# ------------------------------------------------------------ the brief


def test_a_torn_surface_becomes_a_shot():
    g = _graph([_ground()], [OcclusionEdge(occluder="car", occludee="ground",
                                           tear_pixels=5000)])
    shots = build_shoot_list(_Solve(), g)
    assert len(shots) == 1
    s = shots[0]
    assert s.node_id == "ground" and s.tear_px == 5000
    assert s.priority == 1
    assert 0.0 < s.incidence_deg < 90.0
    assert s.px_per_m > 0
    assert "car" in s.hidden_by


def test_trivial_holes_are_not_worth_a_trip():
    g = _graph([_ground()], [OcclusionEdge(occluder="post", occludee="ground",
                                           tear_pixels=12)])
    assert build_shoot_list(_Solve(), g) == []


def test_the_worst_hole_is_shot_first():
    nodes = [_ground("a"), _ground("b"), _ground("c")]
    edges = [OcclusionEdge(occluder="x", occludee="a", tear_pixels=1000),
             OcclusionEdge(occluder="x", occludee="b", tear_pixels=9000),
             OcclusionEdge(occluder="x", occludee="c", tear_pixels=4000)]
    order = [s.node_id for s in build_shoot_list(_Solve(), _graph(nodes, edges))]
    assert order == ["b", "c", "a"]
    assert [s.priority for s in build_shoot_list(_Solve(), _graph(nodes, edges))] == [1, 2, 3]


def test_tears_from_several_occluders_accumulate():
    """One surface hidden by three things needs all of it photographed."""
    g = _graph([_ground()], [
        OcclusionEdge(occluder="car", occludee="ground", tear_pixels=2000),
        OcclusionEdge(occluder="bin", occludee="ground", tear_pixels=1500),
        OcclusionEdge(occluder="tree", occludee="ground", tear_pixels=800)])
    s = build_shoot_list(_Solve(), g)[0]
    assert s.tear_px == 4300
    assert s.hidden_by == ["bin", "car", "tree"]


def test_a_surface_with_no_plane_is_flagged_volumetric():
    """An alleyway is not a texture.

    Without a fitted plane there is no single surface to photograph, so the app
    must fall back to aligning against ghosted geometry — and saying so is more
    useful than inventing an angle for something that has none.
    """
    node = OcclusionNode(id="alley", kind="object", plane=None, depth_range_m=(6.0, 20.0))
    s = build_shoot_list(_Solve(), _graph([node], [
        OcclusionEdge(occluder="wall", occludee="alley", tear_pixels=8000)]))[0]
    assert s.volumetric is True
    assert "ghosted geometry" in " ".join(s.warnings)
    assert "ghost" in s.guidance


def test_an_almost_edge_on_surface_warns_rather_than_pretending():
    node = OcclusionNode(id="road", kind="ground",
                         plane={"normal": [0, 1, 0], "d": 0.0},
                         depth_range_m=(200.0, 400.0))
    s = build_shoot_list(_Solve(), _graph([node], [
        OcclusionEdge(occluder="x", occludee="road", tear_pixels=6000)]))[0]
    assert s.incidence_deg > 85.0
    assert any("edge-on" in w for w in s.warnings)


def test_layer_plan_supplies_the_subject_and_can_veto():
    class _Plan:
        def __init__(self, nid, concepts, needs):
            self.node_id, self.concepts, self.needs_clean_plate = nid, concepts, needs

    g = _graph([_ground("g1"), _ground("g2")], [
        OcclusionEdge(occluder="x", occludee="g1", tear_pixels=5000),
        OcclusionEdge(occluder="x", occludee="g2", tear_pixels=5000)])
    plans = [_Plan("g1", ["pavement", "kerb"], True),
             _Plan("g2", ["sky"], False)]          # already handled elsewhere
    shots = build_shoot_list(_Solve(), g, layer_plan=plans)
    assert [s.node_id for s in shots] == ["g1"]
    assert "pavement" in shots[0].subject


def test_guidance_is_a_sentence_a_person_can_follow():
    g = _graph([_ground()], [OcclusionEdge(occluder="car", occludee="ground",
                                           tear_pixels=5000)])
    text = build_shoot_list(_Solve(), g)[0].guidance
    assert "shoot" in text.lower()
    assert "px per metre" in text
    assert any(w in text for w in ("face-on", "slight angle", "raking"))


# ------------------------------------------------------------ project


def test_project_states_that_lighting_is_not_measured():
    """The absence of lighting fields must not read as "lighting does not matter".

    Sun direction and hardness decide whether a patch sits or reads as a sticker,
    and Atlas cannot measure them from one plate — so the payload says so rather
    than emitting a confident-looking angle that is really a guess.
    """
    g = _graph([_ground()], [OcclusionEdge(occluder="car", occludee="ground",
                                           tear_pixels=5000)])
    proj = shoot_project(build_shoot_list(_Solve(), g), plate_size=(4000, 3000))
    assert proj["lighting"]["measured"] is False
    assert "match the reference crop" in proj["lighting"]["note"].lower()
    assert proj["plate_size"] == [4000, 3000]


def test_project_round_trips_through_json():
    import json

    g = _graph([_ground()], [OcclusionEdge(occluder="car", occludee="ground",
                                           tear_pixels=5000)])
    proj = shoot_project(build_shoot_list(_Solve(), g))
    assert json.loads(json.dumps(proj)) == proj


def test_empty_graph_yields_an_empty_project():
    proj = shoot_project(build_shoot_list(_Solve(), _graph([], [])))
    assert proj["shots"] == [] and proj["version"] == 1


# --------------------------------------------------- incidence at a range


def test_incidence_grows_with_distance_along_a_floor():
    """The same floor is steep underfoot and almost edge-on at the horizon.

    Regression: the first implementation evaluated the plane at its closest
    point to the world origin, which for a ground plane sits directly beneath
    the camera — so every floor reported as face-on and distance had no effect
    at all.
    """
    from atlas_camera.core.shoot_list import incidence_at_range

    eye = (0.0, 1.6, 0.0)
    near = incidence_at_range((0, 1, 0), 0.0, eye, 4.0)
    mid = incidence_at_range((0, 1, 0), 0.0, eye, 20.0)
    far = incidence_at_range((0, 1, 0), 0.0, eye, 200.0)

    assert near == pytest.approx(66.42, abs=0.1)     # acos(1.6/4)
    assert near < mid < far
    assert far > 89.0, "distant ground must read as grazing"


def test_incidence_is_zero_when_the_plane_is_nearer_than_the_range():
    """No ray of that length reaches the plane — must not produce NaN."""
    from atlas_camera.core.shoot_list import incidence_at_range
    assert incidence_at_range((0, 1, 0), 0.0, (0.0, 10.0, 0.0), 2.0) == 0.0


def test_incidence_at_range_ignores_normal_sign():
    from atlas_camera.core.shoot_list import incidence_at_range
    up = incidence_at_range((0, 1, 0), 0.0, (0.0, 1.6, 0.0), 10.0)
    down = incidence_at_range((0, -1, 0), 0.0, (0.0, 1.6, 0.0), 10.0)
    assert up == pytest.approx(down, abs=1e-9)


# ----------------------------------------------------------- AtlasShootList


def _solve_with_graph(tmp_path, nodes, edges):
    import atlas_camera.core.schema as sch
    from atlas_camera.core.camera_math import look_at_view_matrix
    from atlas_camera.core.intrinsics import build_intrinsics
    from atlas_camera.core.occlusion_graph import attach_occlusion_graph

    view, world, rot = look_at_view_matrix(eye=(0, 1.6, 0), target=(0, 1.6, -1),
                                           up=(0, 1, 0))
    intr = build_intrinsics(image_width=4000, image_height=3000,
                            focal_length_mm=35.0, sensor_width_mm=36.0)
    extr = sch.AtlasExtrinsics(camera_position=(0.0, 1.6, 0.0),
                               camera_rotation_matrix=rot,
                               camera_world_matrix=world, camera_view_matrix=view)
    solve = sch.AtlasSolve(camera=sch.AtlasCamera(intrinsics=intr, extrinsics=extr),
                           image_path="x.jpg", image_width=4000, image_height=3000)
    attach_occlusion_graph(solve, _graph(nodes, edges))
    return solve


def _node():
    from atlas_camera.comfy import node_registry as reg
    return reg.NODE_CLASS_MAPPINGS["AtlasShootList"]()


def test_node_writes_a_readable_project(tmp_path):
    import json

    solve = _solve_with_graph(tmp_path, [_ground("pavement")],
                              [OcclusionEdge(occluder="car", occludee="pavement",
                                             tear_pixels=21000)])
    _s, path, report = _node().build(solve, output_dir=str(tmp_path),
                                     project_name="brief")
    data = json.loads(open(path, encoding="utf-8").read())
    assert data["version"] == 1 and len(data["shots"]) == 1
    assert data["shots"][0]["tear_px"] == 21000
    assert "px per metre" in report


def test_node_refuses_a_solve_with_no_graph(tmp_path):
    """Running this before AtlasOcclusionGraph must say so, not return nothing."""
    import atlas_camera.core.schema as sch
    from atlas_camera.core.camera_math import look_at_view_matrix
    from atlas_camera.core.intrinsics import build_intrinsics

    view, world, rot = look_at_view_matrix(eye=(0, 1.6, 0), target=(0, 1.6, -1),
                                           up=(0, 1, 0))
    intr = build_intrinsics(image_width=100, image_height=100,
                            focal_length_mm=35.0, sensor_width_mm=36.0)
    extr = sch.AtlasExtrinsics(camera_position=(0.0, 1.6, 0.0),
                               camera_rotation_matrix=rot,
                               camera_world_matrix=world, camera_view_matrix=view)
    bare = sch.AtlasSolve(camera=sch.AtlasCamera(intrinsics=intr, extrinsics=extr),
                          image_path="x.jpg", image_width=100, image_height=100)
    with pytest.raises(ValueError, match="no occlusion graph"):
        _node().build(bare, output_dir=str(tmp_path))


def test_node_says_so_when_nothing_is_worth_shooting(tmp_path):
    solve = _solve_with_graph(tmp_path, [_ground()],
                              [OcclusionEdge(occluder="post", occludee="ground",
                                             tear_pixels=10)])
    _s, _path, report = _node().build(solve, output_dir=str(tmp_path))
    assert "nothing worth photographing" in report


def test_node_report_always_carries_the_lighting_caveat(tmp_path):
    """A brief that lists angles but not lighting invites the reader to assume
    lighting is handled. It is not, and the report says so every time."""
    solve = _solve_with_graph(tmp_path, [_ground()],
                              [OcclusionEdge(occluder="car", occludee="ground",
                                             tear_pixels=9000)])
    _s, _path, report = _node().build(solve, output_dir=str(tmp_path))
    assert "Lighting is NOT specified" in report


def test_node_writes_the_reference_plate_it_was_given(tmp_path):
    """The brief is for someone standing in the location holding a camera, and
    the reference plate is the only thing in the package that shows them what
    the shot they are matching looks like. It was silently never written: the
    save was wrapped in `except Exception`, so a NameError on the (unimported)
    tensor helper degraded into a "reference plate could not be written"
    warning that read like a bad image rather than a missing import."""
    torch = pytest.importorskip("torch")

    solve = _solve_with_graph(tmp_path, [_ground("pavement")],
                              [OcclusionEdge(occluder="car", occludee="pavement",
                                             tear_pixels=21000)])
    image = torch.zeros((1, 12, 16, 3), dtype=torch.float32)
    _s, path, report = _node().build(solve, output_dir=str(tmp_path),
                                     project_name="brief", reference_image=image)

    from pathlib import Path

    plate = Path(path).parent / "reference_plate.png"
    assert plate.is_file(), "the reference plate was not written"
    assert "reference plate could not be written" not in report


def test_node_output_contract():
    from atlas_camera.comfy import node_registry as reg
    cls = reg.NODE_CLASS_MAPPINGS["AtlasShootList"]
    assert cls.RETURN_TYPES == ("ATLAS_SOLVE", "STRING", "STRING")
    assert cls.RETURN_NAMES == ("solve", "project_path", "report")


# ------------------------------------------------- the iOS contract fixture


def _repo_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[1]


def _load_example_builder():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_shoot_example", _repo_root() / "tools" / "build_shoot_example.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_ios_example_fixture_matches_what_atlas_emits():
    """The iOS app is built against this file, on another machine, in another
    language. If it drifts from reality that is discovered on the Mac — late,
    and confusingly. Same discipline as the feature audit.
    """
    import json

    committed = json.loads(
        (_repo_root() / "docs" / "shoot_project.example.json").read_text(encoding="utf-8"))
    fresh = _load_example_builder().build()
    assert committed == fresh, (
        "docs/shoot_project.example.json is stale — "
        "run python tools/build_shoot_example.py")


def test_the_example_covers_the_volumetric_case():
    """The fixture must exercise the branch a client is most likely to get wrong.

    `volumetric: true` means no plane was fitted and `incidence_deg` is a
    placeholder rather than a measurement. An example containing only planar
    shots would let a client ship treating it as an angle of zero.
    """
    import json
    data = json.loads(
        (_repo_root() / "docs" / "shoot_project.example.json").read_text(encoding="utf-8"))
    kinds = {s["volumetric"] for s in data["shots"]}
    assert kinds == {True, False}, "the fixture must show both planar and volumetric"


def test_the_example_states_lighting_is_unmeasured():
    import json
    data = json.loads(
        (_repo_root() / "docs" / "shoot_project.example.json").read_text(encoding="utf-8"))
    assert data["lighting"]["measured"] is False


def test_the_contract_doc_is_tracked_not_in_gitignored_dev():
    """docs/dev/ does not survive a clone, and the Mac needs this file.

    Guards the specific mistake of "improving" the docs by moving the contract
    into docs/dev/ with the other design notes.
    """
    root = _repo_root()
    assert (root / "docs" / "SHOOT_PROJECT_FORMAT.md").exists()
    assert (root / "docs" / "IOS_APP_BOOTSTRAP.md").exists()
    assert not (root / "docs" / "dev" / "SHOOT_PROJECT_FORMAT.md").exists()
