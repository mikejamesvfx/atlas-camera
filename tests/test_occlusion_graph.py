"""Tests for the occlusion graph — the contract between measurement and construction.

The graph's job is not to be clever. It is to state, per region, what may be
built there, and to refuse to license anything it cannot justify. So most of
these tests are about the REFUSALS: an unclassified tear must stay open, an
unrecognised class must be rejected rather than trusted, and a solve with no
depth must produce a graph that says so instead of one that quietly licenses
completion on no evidence.
"""

import numpy as np
import pytest

from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.occlusion_graph import (
    TEAR_WALL_CONTINUATION,
    POLICY_BACKDROP,
    POLICY_CONSERVATIVE_PROXY,
    POLICY_EXTEND_PLANE,
    POLICY_NONE,
    TEAR_OBJECT_COMPLETION,
    TEAR_UNKNOWN,
    AtlasOcclusionGraph,
    attach_occlusion_graph,
    build_occlusion_graph,
)
from atlas_camera.core.schema import (
    AtlasCamera,
    AtlasExtrinsics,
    AtlasProxyPrimitive,
    AtlasSolve,
)

W, H, F = 128, 96, 128.0


def _identity_view():
    from atlas_camera.core.camera_math import look_at_view_matrix
    view, _, _ = look_at_view_matrix((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
    return np.asarray(view, dtype=np.float64)


def _plane(name, *, normal=(0.0, 0.0, 1.0), centre=(0.0, 0.0, -8.0), source="ransac_planes"):
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    # Local X=u, Y=v, Z=normal — the THREE.PlaneGeometry frame Atlas uses.
    u = np.cross((0.0, 1.0, 0.0), n)
    u = u / max(np.linalg.norm(u), 1e-9) if np.linalg.norm(u) > 1e-9 else np.array([1.0, 0, 0])
    v = np.cross(n, u)
    mat = tuple(
        (float(u[i]), float(v[i]), float(n[i]), float(centre[i])) for i in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)
    return AtlasProxyPrimitive(name=name, primitive_type="plane", transform_matrix=mat,
                               dimensions=(20.0, 20.0, 0.0),
                               metadata={"role": "projection_proxy", "source": source})


def _box(name):
    return AtlasProxyPrimitive(name=name, primitive_type="box", dimensions=(1.0, 2.0, 1.0),
                               metadata={"role": "projection_proxy", "source": "depth_derivation"})


def _solve(prims=()):
    intr = build_intrinsics(image_width=W, image_height=H, focal_length_mm=35.0)
    intr.fx_px = intr.fy_px = F
    intr.cx_px, intr.cy_px = W / 2.0, H / 2.0
    solve = AtlasSolve(
        camera=AtlasCamera(intrinsics=intr,
                           extrinsics=AtlasExtrinsics(camera_view_matrix=_identity_view())),
        image_width=W, image_height=H,
    )
    solve.projection_scene.proxy_geometry = list(prims)
    return solve


def _gentle_step_depth(near=8.0, far=9.0):
    """A 1 m step against an 8 m reference — a 12.5% relative discontinuity.

    That is a tear at the 0.05 threshold this module hardcoded, and NOT a tear
    at build_relief_mesh's actual `depth_edge_rel` default of 0.5. The gap
    between those two numbers is the point.
    """
    depth = np.full((H, W), float(far))
    depth[:, W // 3: 2 * W // 3] = float(near)
    return depth


def test_the_tear_threshold_is_a_parameter_not_a_hardcoded_constant():
    """This module called 0.05 "the same silhouette test build_relief_mesh
    tears on". build_relief_mesh's `depth_edge_rel` defaults to 0.5 and artists
    relax it to 1.0 for forests, so the two tear sets disagreed by 10-20x on
    stock settings while the comment claimed they matched. The threshold is now
    something a caller can pass, so a caller holding the mesh's real value can
    make them agree.
    """
    solve = _solve()
    depth = _gentle_step_depth()

    tight = build_occlusion_graph(solve, depth=depth)
    assert not any("no silhouette discontinuities" in n for n in tight.notes), (
        "the default threshold must still tear on a 12.5% step")

    relaxed = build_occlusion_graph(solve, depth=depth, tear_edge_rel=0.5)
    assert any("no silhouette discontinuities" in n for n in relaxed.notes), (
        "at build_relief_mesh's own default this step is not a tear")


def test_the_default_tear_threshold_is_unchanged():
    """Behaviour pin: the parameter was added without moving the default."""
    import inspect

    from atlas_camera.core.occlusion_graph import build_occlusion_graph as fn

    assert inspect.signature(fn).parameters["tear_edge_rel"].default == 0.05


def _two_layer_depth(near=2.0, far=9.0):
    """A near slab in front of a far plane — one unambiguous occlusion boundary."""
    depth = np.full((H, W), float(far))
    depth[:, W // 3: 2 * W // 3] = float(near)
    return depth


# --- scene decomposition ---

def test_graph_lists_one_node_per_proxy_and_skips_the_relief_mesh():
    from atlas_camera.core.proxy_geometry import relief_mesh_primitive
    from atlas_camera.core.relief_mesh import ReliefMesh

    mesh = ReliefMesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]),
                      uvs=np.zeros((3, 2)))
    solve = _solve([relief_mesh_primitive(mesh), _plane("wall_01"), _box("chair_01")])
    graph = build_occlusion_graph(solve, depth=_two_layer_depth())

    ids = [n.id for n in graph.nodes]
    assert ids == ["wall_01", "chair_01"], "the relief mesh is not a scene part"
    assert graph.node("chair_01").kind == "object"
    assert graph.node("wall_01").kind == "surface"


def test_backdrop_and_ground_are_recognised_by_name():
    solve = _solve([_plane("projection_backdrop"), _plane("projection_ground")])
    graph = build_occlusion_graph(solve, depth=_two_layer_depth())
    assert graph.node("projection_backdrop").kind == "backdrop"
    assert graph.node("projection_backdrop").completion_policy == POLICY_BACKDROP
    assert graph.node("projection_ground").kind == "ground"


def test_plane_normal_is_read_from_the_third_column_not_the_second():
    """Atlas planes are local X=u, Y=v, Z=normal.

    Reading the wrong column yields a plausible-looking unit vector pointing
    90 degrees away, which would silently mis-license every plane continuation.
    """
    solve = _solve([_plane("wall_01", normal=(0.0, 0.0, 1.0), centre=(0.0, 0.0, -8.0))])
    graph = build_occlusion_graph(solve, depth=_two_layer_depth())
    plane = graph.node("wall_01").plane
    assert plane is not None
    assert plane["normal"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    assert plane["d"] == pytest.approx(-8.0, abs=1e-6)


def test_duplicate_primitive_names_do_not_collapse_into_one_node():
    solve = _solve([_plane("wall"), _plane("wall"), _plane("wall")])
    graph = build_occlusion_graph(solve, depth=_two_layer_depth())
    assert len({n.id for n in graph.nodes}) == 3


# --- tears ---

def test_depth_discontinuity_becomes_an_occlusion_edge_nearer_side_occluding():
    solve = _solve([_plane("wall_near", centre=(0.0, 0.0, -2.0)),
                    _plane("wall_far", centre=(0.0, 0.0, -9.0))])
    graph = build_occlusion_graph(solve, depth=_two_layer_depth(near=2.0, far=9.0))

    assert graph.edges, "a two-layer scene must produce at least one tear"
    edge = graph.edges[0]
    assert edge.near_depth_m < edge.far_depth_m, "occluder must be the nearer side"
    assert edge.tear_pixels > 0


def test_a_scene_with_no_discontinuity_produces_no_tears_and_says_so():
    solve = _solve([_plane("wall_01", centre=(0.0, 0.0, -5.0)),
                    _plane("wall_02", centre=(0.0, 0.0, -5.0))])
    flat = np.full((H, W), 5.0)
    graph = build_occlusion_graph(solve, depth=flat)
    assert graph.edges == []
    assert any("no silhouette discontinuities" in n for n in graph.notes)


# --- the refusals ---

def test_missing_depth_licenses_nothing_and_records_why():
    solve = _solve([_plane("wall_01")])
    graph = build_occlusion_graph(solve, depth=None)
    assert graph.edges == []
    assert any("no depth supplied" in n for n in graph.notes)


def test_unknown_tear_class_leaves_the_occludee_unfilled():
    """The central refusal: a tear Atlas cannot classify stays a tear."""
    solve = _solve([_plane("wall_near", centre=(0.0, 0.0, -2.0)),
                    _plane("wall_far", centre=(0.0, 0.0, -9.0))])
    graph = build_occlusion_graph(solve, depth=_two_layer_depth())
    key = f"{graph.edges[0].occluder}|{graph.edges[0].occludee}"

    graph = build_occlusion_graph(solve, depth=_two_layer_depth(),
                                  tear_classes={key: TEAR_UNKNOWN})
    occludee = graph.node(graph.edges[0].occludee)
    assert occludee.completion_policy == POLICY_NONE
    assert any("could not be classified" in n for n in occludee.notes)


def test_one_unclassifiable_tear_blocks_completion_of_the_whole_surface():
    """POLICY_NONE dominates when several tears expose the same surface.

    If part of a surface is genuinely unknown, building over the rest of it
    hides that fact behind geometry that looks finished — the precise failure
    this design exists to prevent.
    """
    solve = _solve([_plane("a", centre=(0.0, 0.0, -2.0)),
                    _plane("b", centre=(0.0, 0.0, -5.0)),
                    _plane("c", centre=(0.0, 0.0, -9.0))])
    depth = np.full((H, W), 9.0)
    depth[:, 40:70] = 5.0
    depth[:, 80:110] = 2.0

    graph = build_occlusion_graph(solve, depth=depth)
    exposing_c = [e for e in graph.edges if e.occludee == "c"]
    assert len(exposing_c) >= 2, "scene must expose 'c' through more than one tear"

    classes = {f"{e.occluder}|{e.occludee}": TEAR_WALL_CONTINUATION
               for e in graph.edges}
    classes[f"{exposing_c[0].occluder}|c"] = TEAR_UNKNOWN

    graph = build_occlusion_graph(solve, depth=depth, tear_classes=classes)
    assert graph.node("c").completion_policy == POLICY_NONE


def test_unrecognised_tear_class_is_rejected_rather_than_trusted():
    """A bad class silently licenses the wrong construction, so it must not pass."""
    solve = _solve([_plane("wall_near", centre=(0.0, 0.0, -2.0)),
                    _plane("wall_far", centre=(0.0, 0.0, -9.0))])
    graph = build_occlusion_graph(solve, depth=_two_layer_depth())
    key = f"{graph.edges[0].occluder}|{graph.edges[0].occludee}"

    graph = build_occlusion_graph(solve, depth=_two_layer_depth(),
                                  tear_classes={key: "demolish_everything"})
    assert graph.edges[0].tear_class != "demolish_everything"
    assert graph.edges[0].classified_by == "depth_heuristic"
    assert any("unrecognised tear class" in n for n in graph.notes)


def test_a_solve_with_no_proxies_says_so_instead_of_returning_an_empty_graph():
    graph = build_occlusion_graph(_solve(), depth=_two_layer_depth())
    assert graph.nodes == []
    assert any("no proxy primitives" in n for n in graph.notes)


def test_vlm_classification_is_recorded_as_such():
    solve = _solve([_plane("wall_near", centre=(0.0, 0.0, -2.0)),
                    _plane("wall_far", centre=(0.0, 0.0, -9.0))])
    graph = build_occlusion_graph(solve, depth=_two_layer_depth())
    key = f"{graph.edges[0].occluder}|{graph.edges[0].occludee}"

    graph = build_occlusion_graph(solve, depth=_two_layer_depth(),
                                  tear_classes={key: TEAR_OBJECT_COMPLETION})
    assert graph.edges[0].tear_class == TEAR_OBJECT_COMPLETION
    assert graph.edges[0].classified_by == "vlm"
    assert graph.node(graph.edges[0].occludee).completion_policy == POLICY_CONSERVATIVE_PROXY


# --- confidence + persistence ---

def test_confidence_comes_from_scene_health_not_from_this_module():
    from atlas_camera.core.scene_health import scale_health

    solve = _solve([_plane("wall_01")])
    graph = build_occlusion_graph(solve, depth=_two_layer_depth())
    assert graph.node("wall_01").confidence == pytest.approx(
        float(scale_health(solve).confidence or 0.0))


def test_graph_round_trips_through_dict():
    solve = _solve([_plane("wall_near", centre=(0.0, 0.0, -2.0)),
                    _plane("wall_far", centre=(0.0, 0.0, -9.0)), _box("chair")])
    graph = build_occlusion_graph(solve, depth=_two_layer_depth())
    restored = AtlasOcclusionGraph.from_dict(graph.to_dict())

    assert [n.id for n in restored.nodes] == [n.id for n in graph.nodes]
    assert [n.completion_policy for n in restored.nodes] == \
        [n.completion_policy for n in graph.nodes]
    assert len(restored.edges) == len(graph.edges)
    assert restored.version == graph.version


def test_attach_writes_to_the_reserved_semantics_slot():
    solve = _solve([_plane("wall_01")])
    graph = build_occlusion_graph(solve, depth=_two_layer_depth())
    attach_occlusion_graph(solve, graph)

    assert solve.semantics.value["occlusion_graph"]["version"] == graph.version
    assert solve.semantics.exportable is True


def test_attached_graph_survives_solve_serialization():
    """The solve JSON is a contract; the graph must ride along in it."""
    solve = _solve([_plane("wall_near", centre=(0.0, 0.0, -2.0)),
                    _plane("wall_far", centre=(0.0, 0.0, -9.0))])
    graph = build_occlusion_graph(solve, depth=_two_layer_depth())
    attach_occlusion_graph(solve, graph)

    payload = solve.to_dict()["semantics"]["value"]["occlusion_graph"]
    assert len(payload["nodes"]) == len(graph.nodes)
    assert payload["nodes"][0]["completion_policy"] in {
        POLICY_EXTEND_PLANE, POLICY_NONE, POLICY_CONSERVATIVE_PROXY, POLICY_BACKDROP}


def test_describe_names_every_node_that_will_build_nothing():
    solve = _solve([_plane("wall_near", centre=(0.0, 0.0, -2.0)),
                    _plane("wall_far", centre=(0.0, 0.0, -9.0))])
    graph = build_occlusion_graph(solve, depth=_two_layer_depth())
    graph.node(graph.edges[0].occludee).completion_policy = POLICY_NONE
    text = graph.describe()
    assert "nothing will be built" in text


# --- layer plan ---

def _plan_graph(policy=POLICY_EXTEND_PLANE):
    from atlas_camera.core.occlusion_graph import OcclusionEdge, OcclusionNode
    return AtlasOcclusionGraph(
        nodes=[
            OcclusionNode(id="projection_box_01", kind="object",
                          completion_policy=policy, depth_range_m=(3.0, 4.0)),
            OcclusionNode(id="projection_wall_01", kind="surface",
                          completion_policy=policy, depth_range_m=(8.0, 9.0)),
            OcclusionNode(id="projection_wall_02", kind="surface",
                          completion_policy=policy, depth_range_m=(20.0, 22.0)),
            OcclusionNode(id="projection_backdrop", kind="backdrop"),
        ],
        edges=[OcclusionEdge(occluder="projection_box_01",
                             occludee="projection_wall_01")],
    )


def test_layer_plan_splits_occluder_from_what_it_hides():
    from atlas_camera.core.occlusion_graph import layer_plan

    plan = layer_plan(_plan_graph())
    assert [l.node_id for l in plan] == ["projection_box_01", "projection_wall_01"]

    fg, bg = plan
    assert fg.role == "foreground" and not fg.needs_clean_plate
    assert bg.role == "background" and bg.needs_clean_plate
    # The clean plate is a DIFFERENT image, so its depth must be solved on it —
    # inheriting the original's depth is the far-band cliff failure.
    assert bg.needs_own_depth_solve


def test_layer_plan_orders_occluders_in_front():
    from atlas_camera.core.occlusion_graph import layer_plan

    plan = layer_plan(_plan_graph())
    assert plan[0].order < plan[1].order
    assert plan[0].exposes == ["projection_wall_01"]
    assert plan[1].hidden_by == ["projection_box_01"]


def test_layer_plan_excludes_the_backdrop():
    """The cyclorama is not a layer — it is what layers sit in front of."""
    from atlas_camera.core.occlusion_graph import layer_plan

    assert all(l.node_id != "projection_backdrop" for l in layer_plan(_plan_graph()))


def test_unoccluded_surfaces_are_omitted_unless_asked_for():
    from atlas_camera.core.occlusion_graph import layer_plan

    assert all(l.node_id != "projection_wall_02" for l in layer_plan(_plan_graph()))
    wide = layer_plan(_plan_graph(), include_unoccluded=True)
    assert any(l.node_id == "projection_wall_02" for l in wide)
    # Nothing hides it, so it needs no plate even when included.
    assert not next(l for l in wide if l.node_id == "projection_wall_02").needs_clean_plate


def test_a_refused_tear_does_not_quietly_acquire_a_clean_plate():
    """The graph's refusal has to survive into the layer manifest.

    POLICY_NONE means Atlas could not classify what is behind the occluder.
    Generating a plate there would invent content for a region it just said it
    could not reason about.
    """
    from atlas_camera.core.occlusion_graph import layer_plan

    plan = layer_plan(_plan_graph(policy=POLICY_NONE))
    bg = next(l for l in plan if l.role == "background")
    assert not bg.needs_clean_plate
    assert not bg.needs_own_depth_solve


def test_concepts_are_derived_for_sam3_but_are_only_placeholders():
    """SAM3 segments concepts; a fitter id is not one.

    The VLM pass exists to replace these with what the thing actually is.
    """
    from atlas_camera.core.occlusion_graph import layer_plan

    assert [l.concepts for l in layer_plan(_plan_graph())] == ["box", "wall"]
