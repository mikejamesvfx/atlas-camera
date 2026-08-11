"""Contract for the camera-space transition ribbon.

The defect it answers is measurable: with `soft_visibility` the mesh is never
torn, so a depth cliff stretches into a fin whose world length is set by the
depth jump and by nothing else. Tearing instead leaves a hard open rim. The
ribbon keeps the tear and hangs a skirt off it whose width is a fixed number of
SCREEN pixels — bounded by construction, at any scene depth.

That last clause is the whole contract, and it is why the skirt is not an
extrusion along the view ray: under a pinhole camera every point on a ray
through the camera centre projects to the same pixel, so a ray extrusion has
exactly zero screen width. The tests below measure the finished geometry back
through the recovered camera rather than trusting the requested width.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlas_camera.core.relief_mesh import build_relief_mesh
from atlas_camera.core.transition_ribbon import (
    RIBBON_BEND_MAX,
    RIBBON_FADE_START,
    ribbon_alpha,
)

H = W = 512
FX = FY = 450.0
GRID = 128
SLOPE, INTERCEPT = 0.6, 90.0

BUILD = dict(view_matrix=np.eye(4), fx=FX, fy=FY, cx=W / 2.0, cy=H / 2.0,
             grid_long_edge=GRID, depth_edge_rel=0.5, max_edge_factor=12.0,
             floor_clamp=None, apply_sky_heuristic=False, quad_coherence=True)

RIBBON_PX = 48.0

#: One lattice cell in plate pixels. Relaxation slides each column's base from
#: the true staircase rim toward its smoothed version, so a column's raw
#: ring0->ringN distance picks up a LATERAL component on top of its exact radial
#: extent. That component is bounded by the staircase amplitude, i.e. one cell,
#: which is the tolerance below. Measuring radially instead would be
#: tautological — the offset IS applied along the normal — so the honest check
#: is the raw distance with the bound stated.
STEP_PX = max(H, W) / GRID


def _cliff(near_m: float, far_m: float) -> np.ndarray:
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    return np.where(ys > (xs * SLOPE + INTERCEPT), near_m, far_m).astype(np.float32)


@pytest.fixture(scope="module")
def cliff():
    return _cliff(4.0, 12.0)


def _project(mesh):
    """Vertices back through the recovered camera -> (pixels (N,2), forward m).

    The view matrix is the identity here, so this is the plain pinhole map with
    Atlas's image convention (origin top-left, y down).
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    fwd = -v[:, 2]
    u = BUILD["cx"] + FX * v[:, 0] / fwd
    p = BUILD["cy"] - FY * v[:, 1] / fwd
    return np.stack([u, p], axis=1), fwd


def _columns(mesh):
    """Group ribbon vertices into their columns by ribbon_t == 0 starts.

    Ring order is contiguous per column by construction, so a column is a run of
    ``rings + 1`` consecutive ribbon vertices — but compaction may drop unused
    ones, so recover the runs from ribbon_t rather than assuming a stride.

    Ring 0 carries ``ribbon_t == 0``, indistinguishable by value from the mesh,
    and where it is depends on welding: unwelded it is the index directly before
    the run; welded it IS a rim vertex, at an arbitrary index. It is recovered
    through the frozen UV, which is the one thing every ring of a column shares
    and no neighbouring column does.
    """
    t = np.asarray(mesh.ribbon_t, dtype=np.float64)
    ribbon_idx = np.nonzero(t > 0.0)[0]
    if not len(ribbon_idx):
        return []
    faces = np.asarray(mesh.faces)
    uvs = np.asarray(mesh.uvs, dtype=np.float64)
    # A column's t rises 1/R .. 1. Split where it STOPS rising (or the indices
    # jump). Splitting on t == 0 separators only works unwelded — welded there
    # are none, and the whole ribbon block reads as one column.
    tv = t[ribbon_idx]
    starts = np.nonzero((np.diff(ribbon_idx) > 1) | (np.diff(tv) <= 0.0))[0] + 1
    runs = [g for g in np.split(ribbon_idx, starts) if len(g) > 1]

    welded = bool(mesh.stats["transition_ribbon"].get("weld_ring0"))
    if not welded:
        return [np.concatenate([[g[0] - 1], g]) for g in runs]

    # Ring 0 is whichever t == 0 vertex shares a ribbon face AND the frozen UV.
    touches = {}
    is_rib = t > 0.0
    for tri in faces[is_rib[faces].any(axis=1)]:
        base_v = [int(v) for v in tri if not is_rib[v]]
        for v in tri:
            if is_rib[v]:
                touches.setdefault(int(v), set()).update(base_v)
    out = []
    for g in runs:
        cands = touches.get(int(g[0]), set())
        match = [c for c in cands if np.allclose(uvs[c], uvs[g[0]], atol=1e-6)]
        if len(match) == 1:
            out.append(np.concatenate([[match[0]], g]))
    return out


# --------------------------------------------------------------------------
# The default must not move


def test_off_by_default_is_byte_identical(cliff):
    plain = build_relief_mesh(cliff, **BUILD)
    again = build_relief_mesh(cliff, transition_ribbon=False, **BUILD)
    assert plain.ribbon_t is None
    assert "transition_ribbon" not in plain.stats
    assert np.array_equal(plain.faces, again.faces)
    assert np.array_equal(plain.vertices, again.vertices)
    assert np.array_equal(plain.uvs, again.uvs)


def test_the_fixture_actually_tears(cliff):
    """Everything below is meaningless if there is no rim to skirt."""
    assert build_relief_mesh(cliff, **BUILD).stats["torn_fraction"] > 0.0


def test_ribbon_adds_geometry_and_reports_it(cliff):
    mesh = build_relief_mesh(cliff, transition_ribbon=True,
                             ribbon_px=RIBBON_PX, **BUILD)
    plain = build_relief_mesh(cliff, **BUILD)
    info = mesh.stats["transition_ribbon"]
    assert info["n_faces"] > 0
    assert info["n_columns"] > 0
    assert len(mesh.faces) == len(plain.faces) + info["n_faces"]
    assert mesh.ribbon_t is not None
    assert len(mesh.ribbon_t) == len(mesh.vertices)


# --------------------------------------------------------------------------
# Screen-space width — the whole point


@pytest.mark.parametrize("near_m,far_m", [(2.0, 6.0), (8.0, 24.0), (40.0, 120.0)])
def test_apparent_width_is_the_requested_pixels_at_any_scene_depth(near_m, far_m):
    """Near, mid and far geometry, one requested width, one measured answer.

    Adaptive width is off: it deliberately varies the request with the size of
    the discontinuity, which would confound the invariance being measured here.
    """
    mesh = build_relief_mesh(_cliff(near_m, far_m), transition_ribbon=True,
                             ribbon_px=RIBBON_PX, ribbon_adaptive=False, **BUILD)
    info = mesh.stats["transition_ribbon"]
    assert info["n_columns"] - info["n_width_clamped"] > 10
    assert abs(info["measured_px_p50"] - RIBBON_PX) <= 1.5
    assert abs(info["measured_px_p95"] - RIBBON_PX) <= 3.0

    # And independently, from the geometry: a fold clamp may make a column
    # narrower, and relaxation adds at most one cell of lateral slide, but
    # nothing may make a column meaningfully WIDER than asked.
    px, _ = _project(mesh)
    spans = np.asarray(
        [float(np.hypot(*(px[c[-1]] - px[c[0]]))) for c in _columns(mesh)])
    assert float(spans.max()) <= RIBBON_PX + STEP_PX


def test_reported_width_matches_the_measured_width(cliff):
    """The stat is a measurement of the finished ribbon, not an echo of the
    request — a drift here means the construction stopped being screen-space."""
    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             ribbon_adaptive=False, **BUILD)
    px, _ = _project(mesh)
    spans = [float(np.hypot(*(px[c[-1]] - px[c[0]]))) for c in _columns(mesh)]
    reported = mesh.stats["transition_ribbon"]["measured_px_p50"]
    assert reported == pytest.approx(float(np.percentile(spans, 50)), abs=1.0)


def test_width_scales_with_the_request(cliff):
    narrow = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=24.0,
                               ribbon_adaptive=False, **BUILD)
    wide = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=96.0,
                             ribbon_adaptive=False, **BUILD)
    assert (wide.stats["transition_ribbon"]["measured_px_p50"]
            > 3.5 * narrow.stats["transition_ribbon"]["measured_px_p50"])


def test_both_sides_of_a_cliff_get_a_skirt(cliff):
    """The FAR sheet has an open rim too, and its outward direction runs UNDER
    the near object — exactly where a disocclusion needs continuation. Its
    background probe points at the nearer sheet and is correctly rejected, so it
    falls back to the tear margin and stays narrow while the near rim widens."""
    info = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=24.0,
                             **BUILD).stats["transition_ribbon"]
    assert info["n_columns_measured_bg"] > 0, "the near rim must find background"
    assert info["n_columns_measured_bg"] < info["n_columns"], (
        "the far rim must NOT accept the nearer sheet as its background")


# --------------------------------------------------------------------------
# Depth profile


def test_world_length_is_bounded_by_the_skirt_width_not_by_the_depth_jump():
    """Screen width bounds only what the RECOVERED camera sees.

    Found live on a 7680px plate: the depth ramp ran all the way to the fallback
    background (`d0 * (1 + depth_edge_rel)`, +50% of depth), so a ~1 m-wide
    skirt at 30 m was 15 m deep — invisible edge-on and an enormous tube under
    orbit. `ribbon_px` therefore appeared not to control length at all. The
    depth run is now capped to a multiple of the skirt's own world width, which
    makes ribbon_px the single length control from ANY viewpoint.
    """
    from atlas_camera.core.transition_ribbon import RIBBON_MAX_DEPTH_SLOPE

    def _lengths(px_width):
        mesh = build_relief_mesh(_cliff(20.0, 60.0), transition_ribbon=True,
                                 ribbon_px=px_width, ribbon_adaptive=False,
                                 **BUILD)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        out = [float(np.linalg.norm(verts[c[-1]] - verts[c[0]]))
               for c in _columns(mesh)]
        return np.asarray(out), mesh.stats["transition_ribbon"]

    narrow, n_info = _lengths(16.0)
    wide, w_info = _lengths(64.0)

    # The cap is an aspect ratio, so world length scales with the REQUESTED
    # width rather than with the (identical) depth jump of the fixture.
    assert float(np.median(wide)) > 3.0 * float(np.median(narrow))

    # And it really is bounded. A column is `W = px * d / f` metres wide and at
    # most `slope * W` deep, so its length is at most hypot of the two — taken
    # at the FAR sheet's depth, which is what dominates p95 (both sheets of the
    # cliff carry a skirt). The margin covers the lateral relaxation slide.
    world_width = 64.0 * 60.0 / FX
    bound = 1.4 * float(np.hypot(RIBBON_MAX_DEPTH_SLOPE * world_width, world_width))
    assert float(np.percentile(wide, 95)) <= bound, (
        f"{np.percentile(wide, 95):.1f} m against a {bound:.1f} m bound — the "
        f"depth run is not tracking the skirt width")
    assert w_info["world_len_p95_m"] > 0.0
    assert w_info["n_depth_capped"] > 0, "the fixture must exercise the cap"


def test_columns_per_ribbon_width_reports_the_scallop_risk(cliff):
    """A skirt narrower than the spacing between its own columns is a FRINGE.

    Reported live as "ribbon_bend bending on the wrong axis" — a row of
    U-shaped tongues. The bend only shapes depth along t and cannot vary
    between columns; what actually happened is that ribbon_px (24) sat below
    the rim's column spacing, so each column's strip was a thin finger with a
    wide quad sagging between it and its neighbour, and depth_slope 5 stretched
    those fingers five times their own width. The stat exists so the ratio is
    visible instead of being diagnosed from a screenshot.
    """
    narrow = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=2.0,
                               ribbon_adaptive=False, **BUILD)
    wide = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=64.0,
                             ribbon_adaptive=False, **BUILD)
    n = narrow.stats["transition_ribbon"]
    w = wide.stats["transition_ribbon"]
    assert n["column_spacing_px"] == pytest.approx(w["column_spacing_px"], rel=0.05), (
        "spacing is a property of the rim, not of the requested width")
    assert n["columns_per_ribbon_width"] < 1.0 < w["columns_per_ribbon_width"]


def test_depth_slope_is_the_orbit_length_control():
    """The exposed knob has to actually move world length, monotonically, and
    stop capping once it exceeds the discontinuity it is bounding."""
    def _info(slope):
        return build_relief_mesh(
            _cliff(20.0, 60.0), transition_ribbon=True, ribbon_px=48.0,
            ribbon_adaptive=False, ribbon_depth_slope=slope,
            **BUILD).stats["transition_ribbon"]

    tight, mid, loose = _info(0.5), _info(2.0), _info(8.0)
    assert tight["world_len_p50_m"] < mid["world_len_p50_m"] < loose["world_len_p50_m"]
    assert tight["depth_slope_max"] == 0.5
    # A generous slope stops binding: the real jump becomes the limit again.
    assert loose["n_depth_capped"] < tight["n_depth_capped"]


def test_ribbon_depth_recedes_monotonically(cliff):
    """Once the ribbon moves behind the foreground it must never come back.

    This is what protects against Bézier overshoot, a malformed background
    sample, and the self-intersection either would produce.
    """
    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             ribbon_bend=RIBBON_BEND_MAX, **BUILD)
    _, fwd = _project(mesh)
    for col in _columns(mesh):
        d = fwd[col]
        assert np.all(np.diff(d) >= -1e-4), "ribbon curled back toward the camera"


def test_ribbon_never_comes_forward_of_its_own_silhouette(cliff):
    """No ribbon vertex may sit on the foreground-facing side of the tear."""
    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             **BUILD)
    _, fwd = _project(mesh)
    for col in _columns(mesh):
        assert float(fwd[col].min()) >= float(fwd[col[0]]) - 1e-4


def test_relaxation_smooths_the_staircase_without_shrinking_the_skirt():
    """A torn lattice rim is a staircase even where the silhouette is straight.
    Built straight out of it, every step becomes a terrace at its own depth —
    seen live as slits between neighbouring strips from the front and separated
    slats from behind. Relaxation pulls the outer rings onto their neighbours'
    average; renormalizing the displacement afterwards is what keeps the width
    contract intact, and is the whole reason smoothing is allowed here at all.
    """
    mesh = build_relief_mesh(_cliff(4.0, 12.0), transition_ribbon=True,
                             ribbon_px=RIBBON_PX, ribbon_adaptive=False, **BUILD)
    px, _ = _project(mesh)
    cols = _columns(mesh)

    # Width survives (this is the contract relaxation could most easily break).
    spans = np.asarray([float(np.hypot(*(px[c[-1]] - px[c[0]]))) for c in cols])
    assert abs(float(np.percentile(spans, 50)) - RIBBON_PX) <= 1.5
    assert float(spans.max()) <= RIBBON_PX + STEP_PX

    # The outer edge must be SMOOTHER than the rim it came from. Measured with
    # the rim adjacency inside the builder — walking columns in vertex-index
    # order compares points that are not neighbours and reports nonsense.
    info = mesh.stats["transition_ribbon"]
    assert info["outer_roughness_px"] < 0.5 * info["rim_roughness_px"], (
        f"outer {info['outer_roughness_px']:.2f}px vs rim "
        f"{info['rim_roughness_px']:.2f}px — relaxation is not doing anything")


def test_bend_is_a_dwell_not_a_bulge(cliff):
    """`ribbon_bend` says how long the ribbon holds near foreground depth before
    falling toward the background — so a larger bend must leave the MIDPOINT
    nearer the camera, with both endpoints unchanged."""
    def _mid_fraction(bend):
        mesh = build_relief_mesh(cliff, transition_ribbon=True,
                                 ribbon_px=RIBBON_PX, ribbon_bend=bend, **BUILD)
        _, fwd = _project(mesh)
        fracs = []
        for col in _columns(mesh):
            d = fwd[col]
            span = float(d[-1] - d[0])
            if span > 1e-6:
                fracs.append((float(d[len(d) // 2]) - float(d[0])) / span)
        return float(np.median(fracs))

    assert _mid_fraction(0.0) == pytest.approx(0.5, abs=0.05), "bend=0 is linear"
    assert _mid_fraction(RIBBON_BEND_MAX) < _mid_fraction(0.0) - 0.1


def test_bend_is_clamped_to_the_monotonic_range_both_ways(cliff):
    """Outside +/-0.5 the ramp stops being monotonic at one end or the other.
    Asking for it must be clamped, not honoured."""
    for asked, want in ((5.0, RIBBON_BEND_MAX), (-5.0, -RIBBON_BEND_MAX)):
        mesh = build_relief_mesh(cliff, transition_ribbon=True,
                                 ribbon_px=RIBBON_PX, ribbon_bend=asked, **BUILD)
        assert mesh.stats["transition_ribbon"]["bend"] == want


def test_negative_bend_curls_away_fast_and_stays_monotonic(cliff):
    """The sign is the artist's control, and both signs must recede.

    Negative is the tight inward lip of the Maya sculpt: leave the rim fast,
    then level off. Positive dwells then dives. Neither may come back toward
    the camera at any point, which is what bounds the range at +/-0.5.
    """
    def _mid_fraction(bend):
        mesh = build_relief_mesh(cliff, transition_ribbon=True,
                                 ribbon_px=RIBBON_PX, ribbon_bend=bend, **BUILD)
        _, fwd = _project(mesh)
        fracs = []
        for col in _columns(mesh):
            d = fwd[col]
            assert np.all(np.diff(d) >= -1e-4), f"bend={bend} curled back"
            span = float(d[-1] - d[0])
            if span > 1e-6:
                fracs.append((float(d[len(d) // 2]) - float(d[0])) / span)
        return float(np.median(fracs))

    tight = _mid_fraction(-RIBBON_BEND_MAX)
    linear = _mid_fraction(0.0)
    flange = _mid_fraction(RIBBON_BEND_MAX)
    assert tight > linear + 0.1, "negative bend must recede FASTER than linear"
    assert flange < linear - 0.1, "positive bend must dwell"


# --------------------------------------------------------------------------
# Topology and UVs


def _component_count(mesh) -> int:
    """Connected components of the mesh graph, by union-find over face edges."""
    parent = np.arange(len(mesh.vertices))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for tri in np.asarray(mesh.faces):
        a = _find(int(tri[0]))
        for other in tri[1:]:
            b = _find(int(other))
            if a != b:
                parent[b] = a
    used = np.unique(np.asarray(mesh.faces).reshape(-1))
    return len({_find(int(v)) for v in used})


def test_welding_ring0_does_not_bridge_the_tear(cliff):
    """The rule is about the NEAR sheet against the FAR sheet, not about a skirt
    against its own rim.

    Ring 0 shares the rim vertex, which gives continuous normals and makes a
    crack between mesh and skirt impossible. What must NOT happen is the two
    sides of a depth cliff becoming one surface — so the component count is the
    real invariant, and it has to match the ribbon-less mesh exactly.
    """
    plain = build_relief_mesh(cliff, **BUILD)
    welded = build_relief_mesh(cliff, transition_ribbon=True,
                               ribbon_px=RIBBON_PX, **BUILD)
    assert welded.stats["transition_ribbon"]["weld_ring0"] is True
    assert _component_count(welded) == _component_count(plain), (
        "the skirt joined two sheets that the tear separates")
    _assert_manifold_winding(welded)


def test_welding_does_not_let_hole_fill_seal_the_disocclusion(cliff):
    """The interaction welding could plausibly break, checked rather than assumed.

    Unwelded, the tear rim is its own open loop. Welded, the only open loop is
    the skirt's OUTER edge — a larger loop that encircles the disocclusion. If
    the interior hole filler swallowed that, it would close the very gap the
    tear exists to create, which is the "never fix a black tear by raising a
    threshold" failure arriving through a side door.
    """
    from atlas_camera.core.mesh_repair import fill_interior_holes

    plain = build_relief_mesh(cliff, **BUILD)
    welded = build_relief_mesh(cliff, transition_ribbon=True,
                               ribbon_px=RIBBON_PX, **BUILD)

    plain_new, _ = fill_interior_holes(
        np.asarray(plain.faces, dtype=np.int64), max_hole_edges=64)
    weld_new, weld_counts = fill_interior_holes(
        np.asarray(welded.faces, dtype=np.int64), max_hole_edges=64)

    # The cliff's disocclusion is a long loop; nothing near its size may be
    # filled in either build.
    assert not [c for c in weld_counts if c > 64]

    # And the fill must not fuse the two sheets in the welded build when it did
    # not in the plain one.
    class _M:  # ReliefMesh is slots=True, so build a throwaway carrier
        def __init__(self, mesh, extra):
            self.vertices = mesh.vertices
            self.faces = (np.concatenate([np.asarray(mesh.faces), extra], axis=0)
                          if len(extra) else np.asarray(mesh.faces))

    assert (_component_count(_M(welded, weld_new))
            == _component_count(_M(plain, plain_new))), (
        "hole fill bridged the tear once the skirt was welded on")


def test_welded_ring0_is_a_real_mesh_vertex_not_a_duplicate(cliff):
    """Welded, ring 0 is not merely coincident — it IS the rim vertex, so the
    skirt is referenced by faces that also belong to the surface."""
    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             **BUILD)
    t = np.asarray(mesh.ribbon_t)
    faces = np.asarray(mesh.faces)
    rib_face = (t > 0.0)[faces].any(axis=1)
    ring0 = np.unique(faces[rib_face][(t > 0.0)[faces[rib_face]] == False])  # noqa: E712
    base_verts = np.unique(faces[~rib_face].reshape(-1))
    assert len(ring0) > 0
    assert np.isin(ring0, base_verts).all(), (
        "a welded ring 0 must be a vertex the surface already uses")


def test_unwelded_ring0_is_coincident_but_separate(cliff):
    """The free-floating mode is still supported and still shares no vertex."""
    from atlas_camera.core.relief_mesh import build_relief_mesh as _build

    mesh = _build(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX, **BUILD)
    # Rebuild through the core builder with welding off.
    from atlas_camera.core.transition_ribbon import build_transition_ribbon
    plain = _build(cliff, **BUILD)
    res = build_transition_ribbon(
        vertices=np.asarray(plain.vertices, dtype=np.float64),
        faces=np.asarray(plain.faces, dtype=np.int64),
        view_matrix=np.eye(4), fx=FX, fy=FY, cx=W / 2.0, cy=H / 2.0, scale=1.0,
        unproject=__import__(
            "atlas_camera.core.transition_ribbon", fromlist=["x"]
        ).plain_unprojector(np.eye(4), FX, FY, W / 2.0, H / 2.0),
        depth_edge_rel=0.5, image_width=W, image_height=H,
        ribbon_px=RIBBON_PX, weld_ring0=False)
    assert res["stats"]["weld_ring0"] is False
    # Every face references only NEW vertices when unwelded.
    assert int(res["faces"].min()) >= len(plain.vertices)
    assert mesh.stats["transition_ribbon"]["weld_ring0"] is True


def test_every_ring_inherits_the_silhouette_uv_unchanged(cliff):
    """Frozen UV is what makes this an edge-extend clamp instead of a smear."""
    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             **BUILD)
    uvs = np.asarray(mesh.uvs, dtype=np.float64)
    for col in _columns(mesh):
        column_uvs = uvs[col]
        assert np.allclose(column_uvs, column_uvs[0], atol=1e-7), (
            "a ribbon ring drifted off the silhouette texel")


def test_ribbon_faces_wind_toward_the_camera(cliff):
    """Same winding as the sheet they hang off, or a DCC backface-culls them."""
    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             **BUILD)
    px, _ = _project(mesh)
    faces = np.asarray(mesh.faces)
    a, b, c = px[faces[:, 0]], px[faces[:, 1]], px[faces[:, 2]]
    area = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    touch = (np.asarray(mesh.ribbon_t) > 0.0)[faces].any(axis=1)
    fg_sign = np.sign(np.median(area[~touch]))
    ribbon = area[touch]
    ribbon = ribbon[np.abs(ribbon) > 1e-9]
    assert float(np.mean(np.sign(ribbon) == fg_sign)) > 0.99


def _assert_manifold_winding(mesh):
    """No directed edge may appear twice — the condition `mesh_repair`'s
    face-fan walk needs, and the one AtlasExportReliefMesh enforces."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    directed = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    _, counts = np.unique(directed, axis=0, return_counts=True)
    assert int(counts.max()) == 1, (
        f"{int((counts > 1).sum())} directed edge(s) duplicated — inconsistent "
        f"winding or a column shared by more than two quads")


def test_ribbon_winding_stays_manifold_on_a_branching_silhouette(cliff):
    """Found live: AtlasExportReliefMesh raised NonManifoldWindingError on a
    castle silhouette ("face fan around vertex 26771 did not close after 72461
    rotations"). Two causes, both here — a rim vertex where 3+ edges meet shared
    ONE column between 3+ quads, and per-triangle winding correction let
    neighbouring quads disagree."""
    _assert_manifold_winding(
        build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                          **BUILD))

    # A silhouette with real junctions and pinches: separated blocks at three
    # depths plus a thin spire, which is where rim degree exceeds two.
    depth = np.full((H, W), 40.0, dtype=np.float32)
    depth[120:380, 120:200] = 5.0
    depth[120:380, 210:290] = 9.0
    depth[60:380, 240:255] = 4.0        # spire crossing the block below it
    depth[300:460, 150:420] = 14.0
    depth[200:260, 330:400] = 7.0
    for grid in (64, 128, 256):
        mesh = build_relief_mesh(
            depth, transition_ribbon=True, ribbon_px=RIBBON_PX,
            **{**BUILD, "grid_long_edge": grid})
        assert mesh.stats["transition_ribbon"]["n_faces"] > 0
        _assert_manifold_winding(mesh)


def test_sub_quad_rim_is_welded_into_a_connected_curve(cliff):
    """`sub_quad_boundary` emits each torn cell's polygons independently, so its
    crossing vertices are duplicated per cell and the raw open rim is thousands
    of ISOLATED one-edge fragments, not a silhouette curve. Unwelded, no column
    can be shared and normal smoothing has no neighbours — found live as a dense
    spray of loose blades instead of a skirt.

    A connected chain has about as many columns as edges; a fully fragmented rim
    has twice as many. That ratio is the test.
    """
    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             sub_quad_boundary=True, **BUILD)
    assert mesh.stats["sub_quad_cut"]["n_cut_cells"] > 0, "fixture must cut"
    info = mesh.stats["transition_ribbon"]
    assert info["n_columns"] < 1.2 * info["n_rim_edges"], (
        f"{info['n_columns']} columns for {info['n_rim_edges']} rim edges — the "
        f"rim is fragmented, so nothing is shared and nothing is smoothed")
    # And the consequence that made it visible: blades fold, a welded rim does not.
    assert info["n_folded_quads"] == 0
    assert info["n_width_clamped"] == 0


def test_retopology_removes_the_ribbon_instead_of_remeshing_it(cliff):
    """A remesh treats every triangle as surface, and the skirt is not surface.

    Found live: `AtlasRetopologizeLayer` smoothed the ribbon along with the
    mesh, welding the two sheets it exists to keep apart and moving the rim out
    from under it — detached slabs with hard dark rims. The stale per-vertex
    array then failed the length guard downstream and zero-filled, so the skirt
    rendered fully OPAQUE, which is the opposite of its purpose.
    """
    from atlas_camera.comfy.nodes_geometry import _strip_transition_ribbon

    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             **BUILD)
    n_faces_before = len(mesh.faces)
    ribbon_faces = mesh.stats["transition_ribbon"]["n_faces"]

    dropped = _strip_transition_ribbon(mesh, np)
    assert dropped == ribbon_faces
    assert len(mesh.faces) == n_faces_before - ribbon_faces
    assert mesh.ribbon_t is None
    # What is left must be the plain torn mesh — same faces a ribbon-less build
    # produces, and still valid for a remesh to consume.
    plain = build_relief_mesh(cliff, **BUILD)
    assert len(mesh.faces) == len(plain.faces)
    assert len(mesh.vertices) == len(plain.vertices)
    assert np.allclose(np.sort(mesh.vertices, axis=0),
                       np.sort(plain.vertices, axis=0), atol=1e-4)
    # And the per-vertex fields stay aligned after the compaction.
    assert len(mesh.uvs) == len(mesh.vertices)
    if mesh.edge_risk is not None:
        assert len(mesh.edge_risk) == len(mesh.vertices)


def test_retopology_rebuilds_the_skirt_on_the_new_rim(cliff):
    """Stripping alone is a capability loss. The skirt is derived from the rim,
    so after the rim moves it can simply be derived again — with the settings
    the relief-mesh node recorded, not invented ones. The background probe
    cannot run without a depth map, and measurement says that costs almost
    nothing: it succeeded for 4-6% of columns on a castle and 0% on a machine
    plate; everything else already used the tear-margin fallback.
    """
    from atlas_camera.comfy.nodes_geometry import (
        _rebuild_transition_ribbon,
        _strip_transition_ribbon,
    )
    from atlas_camera.core.proxy_geometry import relief_mesh_primitive

    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             ribbon_adaptive=False, **BUILD)
    meta = relief_mesh_primitive(mesh).metadata
    # The settings must ride the primitive as SCALARS or the metadata filter
    # drops them and a rebuild has nothing to replay.
    assert meta["ribbon_px"] == RIBBON_PX
    assert meta["ribbon_adaptive"] is False

    class _Intr:
        fx_px = fy_px = FX
        cx_px, cy_px = W / 2.0, H / 2.0
        image_width, image_height = W, H

    class _Extr:
        camera_view_matrix = np.eye(4)

    class _Cam:
        intrinsics, extrinsics = _Intr(), _Extr()

    _strip_transition_ribbon(mesh, np)
    assert mesh.ribbon_t is None
    added = _rebuild_transition_ribbon(mesh, _Cam(), meta, np)
    assert added > 0

    assert mesh.ribbon_t is not None
    assert len(mesh.ribbon_t) == len(mesh.vertices)
    assert len(mesh.uvs) == len(mesh.vertices)
    _assert_manifold_winding(mesh)

    # Same requested width, achieved on the rebuilt rim.
    px, _ = _project(mesh)
    spans = np.asarray(
        [float(np.hypot(*(px[c[-1]] - px[c[0]]))) for c in _columns(mesh)])
    assert abs(float(np.percentile(spans, 50)) - RIBBON_PX) <= 1.5


def test_rebuilt_ribbon_survives_the_node_not_just_the_helper():
    """End-to-end through AtlasRetopologizeLayer, because the helper passing in
    isolation is exactly what hid the real bug: the rebuilt array was written
    into the primitive and then wiped by the stale-array invalidation that ran
    after it. Two rules whose ORDER decided the outcome.
    """
    pytest.importorskip("torch")
    pytest.importorskip("trimesh")
    from atlas_camera.comfy.nodes_geometry import (
        AtlasDeriveReliefMesh,
        AtlasRetopologizeLayer,
    )
    from atlas_camera.core.intrinsics import build_intrinsics
    from atlas_camera.core.schema import AtlasCamera, AtlasExtrinsics, AtlasSolve
    from atlas_camera.inference.depth_estimator import DepthResult

    n = 256
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
    arr = np.where(ys > (xs * 0.6 + 40.0), 4.0, 12.0).astype(np.float32)
    depth = DepthResult(depth=arr, is_metric=True, model_id="test",
                        image_width=n, image_height=n)
    cam = AtlasCamera(
        intrinsics=build_intrinsics(image_width=n, image_height=n,
                                    focal_length_mm=35.0, sensor_width_mm=36.0),
        extrinsics=AtlasExtrinsics(
            camera_position=(0.0, 0.0, 0.0),
            camera_world_matrix=((1, 0, 0, 0), (0, 1, 0, 0),
                                 (0, 0, 1, 0), (0, 0, 0, 1))))
    solve = AtlasSolve(camera=cam, image_width=n, image_height=n)

    built = AtlasDeriveReliefMesh().derive(
        solve, depth, relief_grid=64, depth_edge_rel=0.5, max_edge_factor=4.0,
        sky_heuristic=False, transition_ribbon=True, ribbon_px=24.0,
        ribbon_bend=-0.3, ribbon_adaptive=False)[0]

    def _meta(s):
        for p in s.projection_scene.proxy_geometry:
            md = p.metadata or {}
            if md.get("source") == "depth_relief_mesh":
                return md
        return {}

    assert len(_meta(built)["ribbon_t"]) == _meta(built)["n_vertices"]

    out, report = AtlasRetopologizeLayer().retopo(
        built, layer="", method="decimate", target_vertex_count=2000,
        rebuild_transition_ribbon=True)
    md = _meta(out)
    assert "re-derived on the new rim" in report
    assert len(md["ribbon_t"]) == md["n_vertices"], (
        "the rebuilt ribbon_t did not survive into the primitive")
    assert any(v > 0.0 for v in md["ribbon_t"])

    off, report_off = AtlasRetopologizeLayer().retopo(
        built, layer="", method="decimate", target_vertex_count=2000,
        rebuild_transition_ribbon=False)
    assert "NOT rebuilt" in report_off
    assert _meta(off)["ribbon_t"] == []


def test_rebuild_declines_without_usable_intrinsics(cliff):
    """Never invent a camera: a layer whose intrinsics cannot place a pixel gets
    a torn mesh and a report line, not a guessed skirt."""
    from atlas_camera.comfy.nodes_geometry import (
        _rebuild_transition_ribbon,
        _strip_transition_ribbon,
    )
    from atlas_camera.core.proxy_geometry import relief_mesh_primitive

    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             **BUILD)
    meta = relief_mesh_primitive(mesh).metadata
    _strip_transition_ribbon(mesh, np)

    class _Cam:
        intrinsics = None
        extrinsics = None

    assert _rebuild_transition_ribbon(mesh, _Cam(), meta, np) == 0
    assert mesh.ribbon_t is None


def test_stripping_is_a_no_op_without_a_ribbon(cliff):
    from atlas_camera.comfy.nodes_geometry import _strip_transition_ribbon

    mesh = build_relief_mesh(cliff, **BUILD)
    before = len(mesh.faces)
    assert _strip_transition_ribbon(mesh, np) == 0
    assert len(mesh.faces) == before


def test_ribbon_survives_the_exporter_manifold_check(cliff):
    """The end-to-end guard: this is the call that actually failed."""
    from atlas_camera.core.mesh_repair import boundary_edges, walk_loops

    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             sub_quad_boundary=True, **BUILD)
    _assert_manifold_winding(mesh)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    # The pivot walk is what raised NonManifoldWindingError in the field.
    walk_loops(boundary_edges(faces), faces)


def test_concave_corners_do_not_invert_adjacent_strips():
    """A tight concave notch crosses neighbouring outward normals. v1 detects
    the fold and shrinks or drops the quad — it must never emit inverted
    geometry, because that is what reaches a DCC."""
    depth = np.full((H, W), 20.0, dtype=np.float32)
    # A deep, narrow notch cut into a near block: two concave corners.
    depth[150:400, 150:360] = 5.0
    depth[150:330, 230:280] = 20.0
    mesh = build_relief_mesh(depth, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             **BUILD)
    px, _ = _project(mesh)
    faces = np.asarray(mesh.faces)
    touch = (np.asarray(mesh.ribbon_t) > 0.0)[faces].any(axis=1)
    a, b, c = px[faces[touch][:, 0]], px[faces[touch][:, 1]], px[faces[touch][:, 2]]
    area = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    live = area[np.abs(area) > 1e-9]
    assert len(live) > 0
    assert float(np.mean(np.sign(live) == np.sign(np.median(live)))) > 0.99


# --------------------------------------------------------------------------
# Background sampling and adaptivity


def test_background_depth_must_be_genuinely_behind():
    """The outward normal can point at sky, or at a NEARER neighbour. A sample
    that is not behind by the same relative margin that tore the mesh is
    rejected, or the ribbon curls forward into an unrelated object."""
    depth = np.full((H, W), 30.0, dtype=np.float32)
    depth[100:400, 100:250] = 6.0      # near block
    depth[100:400, 250:400] = 3.0      # a NEARER neighbour, right beside it
    mesh = build_relief_mesh(depth, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             **BUILD)
    _, fwd = _project(mesh)
    for col in _columns(mesh):
        assert float(fwd[col].min()) >= float(fwd[col[0]]) - 1e-4


def test_adaptive_width_tracks_the_relative_jump_not_the_metre_jump():
    """Scene scale must not change the ribbon. The same relative discontinuity
    at 10x the distance is the same discontinuity."""
    small = build_relief_mesh(_cliff(4.0, 12.0), transition_ribbon=True,
                              ribbon_px=RIBBON_PX, **BUILD)
    large = build_relief_mesh(_cliff(40.0, 120.0), transition_ribbon=True,
                              ribbon_px=RIBBON_PX, **BUILD)
    a = small.stats["transition_ribbon"]["measured_px_p50"]
    b = large.stats["transition_ribbon"]["measured_px_p50"]
    assert a == pytest.approx(b, rel=0.1)


# --------------------------------------------------------------------------
# Gates and the shared fade


def test_soft_visibility_and_the_ribbon_refuse_to_run_together(cliff):
    """Two answers to one question: soft visibility deletes the rim the ribbon
    hangs off. Skipping must be visible, never silent."""
    mesh = build_relief_mesh(cliff, transition_ribbon=True, soft_visibility=True,
                             ribbon_px=RIBBON_PX, **BUILD)
    info = mesh.stats["transition_ribbon"]
    assert info["skipped"] == "soft_visibility"
    assert "soft_visibility" in info["reason"]
    assert mesh.ribbon_t is None


def test_ribbon_t_survives_the_proxy_round_trip(cliff):
    """Every mutating node rebuilds the mesh from primitive metadata. A mesh
    that lost ribbon_t there exports the skirt as an opaque lip."""
    from atlas_camera.core.proxy_geometry import (
        AtlasProjectionScene,
        relief_mesh_primitive,
        serialize_proxy_geometry,
    )
    from atlas_camera.exporters._layers import mesh_from_primitive

    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             **BUILD)
    prim = relief_mesh_primitive(mesh)
    entry = serialize_proxy_geometry(AtlasProjectionScene(proxy_geometry=[prim]))[0]
    assert len(entry["ribbon_t"]) == len(mesh.vertices), (
        "ribbon_t must be lifted to the top level — the metadata scalar filter "
        "drops arrays left behind")

    again = mesh_from_primitive(prim)
    assert again.ribbon_t is not None
    assert np.allclose(again.ribbon_t, mesh.ribbon_t, atol=1e-4)


def test_glb_carries_the_evaluated_fade_as_vertex_alpha(cliff, tmp_path):
    """Not `1 - ribbon_t`: the viewport applies a smoothstep, so exporting the
    raw parameter would hand a DCC a linear ramp where the viewport shows an
    S-curve, and the same mesh would look different in the two places."""
    import json
    import struct

    from atlas_camera.exporters.relief_mesh_exporter import export_relief_mesh_glb

    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             **BUILD)
    path = export_relief_mesh_glb(mesh, tmp_path)["glb"]
    with open(path, "rb") as fh:
        blob = fh.read()
    json_len = struct.unpack("<I", blob[12:16])[0]
    gltf = json.loads(blob[20:20 + json_len].decode("utf-8"))

    prim = gltf["meshes"][0]["primitives"][0]
    assert "COLOR_0" in prim["attributes"]
    assert gltf["materials"][0]["alphaMode"] == "BLEND", (
        "without BLEND the alpha is ignored and the skirt is an opaque lip")

    acc = gltf["accessors"][prim["attributes"]["COLOR_0"]]
    assert acc["type"] == "VEC4" and acc["count"] == len(mesh.vertices)
    view = gltf["bufferViews"][acc["bufferView"]]
    bin_start = 20 + json_len + 8
    raw = blob[bin_start + view["byteOffset"]:
               bin_start + view["byteOffset"] + view["byteLength"]]
    alpha = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)[:, 3]
    assert np.allclose(alpha, ribbon_alpha(mesh.ribbon_t), atol=1e-6)
    assert float(alpha[np.asarray(mesh.ribbon_t) == 0.0].min()) == pytest.approx(1.0)
    assert float(alpha.min()) < 0.01, "the outer edge must reach transparent"


def test_glb_bakes_the_smudge_as_vertex_colour_on_its_own_material(cliff, tmp_path):
    """The smudge cannot live in the texture — every ring of a column shares one
    frozen texel but wants a different blur width, and one texel cannot hold a
    t-dependent value. So it is baked per-vertex, and the skirt needs its OWN
    untextured material: COLOR_0 multiplies the base colour in glTF, so leaving
    it on the textured material would multiply the bake by the plate and darken
    it rather than replacing it.
    """
    import json
    import struct

    from PIL import Image

    from atlas_camera.exporters.relief_mesh_exporter import export_relief_mesh_glb

    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             ribbon_smudge_px=10.0, **BUILD)
    # A plate with strong horizontal colour variation, so an along-rim average
    # is measurably different from a point sample.
    grad = np.zeros((H, W, 3), dtype=np.uint8)
    grad[:, :, 0] = (np.arange(W) % 32 < 16) * 255
    grad[:, :, 2] = 255 - grad[:, :, 0]
    tex = Image.fromarray(grad)

    path = export_relief_mesh_glb(mesh, tmp_path, texture=tex)["glb"]
    blob = open(path, "rb").read()
    json_len = struct.unpack("<I", blob[12:16])[0]
    gltf = json.loads(blob[20:20 + json_len].decode("utf-8"))

    prims = gltf["meshes"][0]["primitives"]
    assert len(prims) == 2, "the skirt must be its own primitive"
    ribbon_mat = gltf["materials"][prims[1]["material"]]
    assert "transition_ribbon" in ribbon_mat["name"]
    assert "baseColorTexture" not in ribbon_mat["pbrMetallicRoughness"], (
        "an untextured material is what makes COLOR_0 the colour, not a tint")
    assert ribbon_mat["alphaMode"] == "BLEND"

    # Every triangle is still exported exactly once, across both primitives.
    total = sum(gltf["accessors"][p["indices"]]["count"] for p in prims)
    assert total == len(mesh.faces) * 3

    # The baked colour is an AVERAGE, so on this deliberately banded plate the
    # outer rings must not equal any single source texel.
    acc = gltf["accessors"][prims[0]["attributes"]["COLOR_0"]]
    view = gltf["bufferViews"][acc["bufferView"]]
    start = 20 + json_len + 8 + view["byteOffset"]
    rgba = np.frombuffer(blob[start:start + view["byteLength"]],
                         dtype=np.float32).reshape(-1, 4)
    t = np.asarray(mesh.ribbon_t)
    outer = rgba[t > 0.75][:, :3]
    assert len(outer) > 0
    pure = np.abs(outer - np.round(outer)).max(axis=1)
    assert float(np.mean(pure > 0.02)) > 0.3, (
        "outer rings look like point samples, not an along-rim average")
    assert np.allclose(rgba[t == 0.0][:, :3], 1.0), "surface must stay white"


def test_glb_without_a_ribbon_is_unchanged(cliff, tmp_path):
    from atlas_camera.exporters.relief_mesh_exporter import export_relief_mesh_glb

    mesh = build_relief_mesh(cliff, **BUILD)
    path = export_relief_mesh_glb(mesh, tmp_path)["glb"]
    blob = open(path, "rb").read()
    json_len = __import__("struct").unpack("<I", blob[12:16])[0]
    gltf = __import__("json").loads(blob[20:20 + json_len].decode("utf-8"))
    assert "COLOR_0" not in gltf["meshes"][0]["primitives"][0]["attributes"]
    assert "alphaMode" not in gltf["materials"][0]


def test_obj_puts_the_ribbon_in_its_own_material(cliff, tmp_path):
    """OBJ has no per-vertex alpha, so the gradient cannot survive — but a
    separate material lets a TD isolate, dial or delete the skirt in one click
    instead of hunting loose triangles."""
    from atlas_camera.exporters.relief_mesh_exporter import export_relief_mesh

    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             **BUILD)
    out = export_relief_mesh(mesh, tmp_path)
    obj = open(out["obj"], encoding="utf-8").read()
    mtl = open(out["mtl"], encoding="utf-8").read()
    assert obj.count("usemtl ") == 2
    assert "usemtl atlas_relief_projection_transition_ribbon" in obj
    assert "newmtl atlas_relief_projection_transition_ribbon" in mtl
    # Every face still written exactly once, and v/vt still 1:1.
    assert sum(1 for line in obj.splitlines() if line.startswith("f ")) == len(mesh.faces)
    assert (sum(1 for line in obj.splitlines() if line.startswith("v "))
            == sum(1 for line in obj.splitlines() if line.startswith("vt ")))


def test_obj_bakes_the_smudge_as_vertex_colours(cliff, tmp_path):
    """OBJ carries one UV set and the skirt's UVs must keep pointing at the
    plate (the viewport samples them there), so a strip-atlas remap is out and
    the `v x y z r g b` extension is what is left.

    The skirt material must then drop map_Kd: OBJ does not define whether a
    reader multiplies vertex colour by the map or replaces it, so shipping both
    makes the result importer-dependent. Kd carries the mean skirt colour so a
    reader that ignores vertex colours still gets a plausible flat tone.
    """
    from PIL import Image

    from atlas_camera.exporters.relief_mesh_exporter import export_relief_mesh

    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             ribbon_smudge_px=10.0, **BUILD)
    grad = np.zeros((H, W, 3), dtype=np.uint8)
    grad[:, :, 0] = (np.arange(W) % 32 < 16) * 255
    grad[:, :, 2] = 255 - grad[:, :, 0]
    out = export_relief_mesh(mesh, tmp_path, texture=Image.fromarray(grad))

    obj = open(out["obj"], encoding="utf-8").read().splitlines()
    mtl = open(out["mtl"], encoding="utf-8").read()

    v_lines = [ln for ln in obj if ln.startswith("v ")]
    assert len(v_lines) == len(mesh.vertices)
    assert all(len(ln.split()) == 7 for ln in v_lines), "expected v x y z r g b"

    # v/vt stay 1:1 — the invariant the face writer depends on.
    assert len([ln for ln in obj if ln.startswith("vt ")]) == len(v_lines)
    assert sum(1 for ln in obj if ln.startswith("f ")) == len(mesh.faces)

    ribbon_block = mtl[mtl.index("newmtl atlas_relief_projection_transition_ribbon"):]
    assert "map_Kd" not in ribbon_block, (
        "vertex colours and map_Kd together are importer-dependent")
    assert "map_Kd" in mtl[:mtl.index("newmtl atlas_relief_projection_transition")], (
        "the SURFACE material must keep its texture")

    # Surface vertices stay white; ribbon vertices carry real colour.
    cols = np.asarray([[float(x) for x in ln.split()[4:7]] for ln in v_lines])
    t = np.asarray(mesh.ribbon_t)
    assert np.allclose(cols[t == 0.0], 1.0)
    assert not np.allclose(cols[t > 0.0], 1.0)


def test_obj_keeps_the_plate_on_the_skirt_when_nothing_is_baked(cliff, tmp_path):
    """Without a smudge there is no vertex colour, so the skirt keeps map_Kd and
    samples its frozen rim texel — the hard edge-extend, which is correct."""
    from atlas_camera.exporters.relief_mesh_exporter import export_relief_mesh

    mesh = build_relief_mesh(cliff, transition_ribbon=True, ribbon_px=RIBBON_PX,
                             ribbon_smudge_px=0.0, **BUILD)
    out = export_relief_mesh(mesh, tmp_path, texture_path="plate.exr")
    obj = open(out["obj"], encoding="utf-8").read().splitlines()
    assert all(len(ln.split()) == 4 for ln in obj if ln.startswith("v "))
    mtl = open(out["mtl"], encoding="utf-8").read()
    assert mtl.count("map_Kd") == 2, "both materials keep the plate"


def test_obj_without_a_ribbon_keeps_one_material(cliff, tmp_path):
    from atlas_camera.exporters.relief_mesh_exporter import export_relief_mesh

    out = export_relief_mesh(build_relief_mesh(cliff, **BUILD), tmp_path)
    assert open(out["obj"], encoding="utf-8").read().count("usemtl ") == 1


def test_the_fade_is_opaque_at_the_rim_and_gone_by_the_outer_edge():
    """One curve, three consumers — shader, GLB vertex colour, and these tests.
    Opaque where it meets the rim, or the fade reintroduces the soft-edged hole
    the tear exists to avoid."""
    assert float(ribbon_alpha(0.0)) == pytest.approx(1.0)
    assert float(ribbon_alpha(RIBBON_FADE_START)) == pytest.approx(1.0)
    assert float(ribbon_alpha(1.0)) == pytest.approx(0.0)
    t = np.linspace(0.0, 1.0, 64)
    assert np.all(np.diff(ribbon_alpha(t)) <= 1e-6), "the fade must be monotonic"
