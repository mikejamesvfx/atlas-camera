"""A fill patch is the BACKMOST layer, so it must not tear.

`build_relief_mesh` tears a cell whose depth jumps more than `depth_edge_rel`
across one grid step. On a PRIMARY relief that is load-bearing: the tear is a
real silhouette and the layer behind reveals through it. A patch generated to
fill a disocclusion is itself that behind-layer -- nothing is further back --
so every torn cell re-opens a hole in exactly the region the patch exists to
close.

Measured live 2026-09-04 on the sea-cliff castle before this was fixed:
`fill_roi1`'s mesh came back with `torn_fraction = 0.404`, and re-surveying the
filled solve left 6,301 px of the ROI still holed -- the interior core of the
fill, ringed by the mesh that did survive, with a patch vertex within 2 px of
88% of it and ZERO orphaned vertices. The geometry was there; its faces were
not.

Not tearing means a stretched triangle bridges a genuine depth cliff inside the
fill instead. That is the correct trade for this layer and it is the project's
own seam doctrine: the smear lives on the layers BEHIND, and only the frontmost
band keeps a clean cut.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from atlas_camera.comfy.nodes import AtlasAddPatchView  # noqa: E402

from test_add_patch_view import _synthetic_primary  # noqa: E402


def _patch_cliff_depth(monkeypatch, near=3.0, far=45.0):
    """Depth the way a GENERATED fill's depth actually looks.

    One straight cliff tears a single column of cells (measured: 2.8%) and
    would not exercise this at all. A monocular estimate over inpainted content
    is blocky at cell scale -- rock against water against distance -- so the
    fixture is a coarse random field of near/far blocks, which reproduces the
    live 40% order of magnitude.
    """
    from dataclasses import dataclass

    @dataclass
    class _FakeDepth:
        depth: object
        is_metric: bool = True
        model_id: str = "fake"

    def fake(image_path, *, model_id=None, device=None, focal_px=None):
        h = w = 256
        rng = np.random.default_rng(7)
        blocks = rng.uniform(0.0, 1.0, size=(16, 16)) < 0.45
        d = np.where(blocks, near, far).astype(np.float32)
        d = np.repeat(np.repeat(d, h // 16, 0), w // 16, 1)
        d += np.linspace(0.0, 2.0, h)[:, None]        # not two flat cards
        return _FakeDepth(depth=d)

    import atlas_camera.inference.depth_estimator as de
    monkeypatch.setattr(de, "estimate_depth", fake)


def _patch(solve, **kw):
    img = torch.rand(1, 256, 256, 3, dtype=torch.float32)
    out, _report = AtlasAddPatchView().add_patch(
        solve, img, geometry_source="own_depth", relief_grid=64,
        mask_unseen_only=False, **kw)
    prim = out.projection_sources[-1].proxy_geometry[0]
    return (prim.metadata or {})


def test_a_hand_placed_patch_keeps_the_shipped_tearing(monkeypatch):
    """The artist node is unchanged: a considered patch of a real novel view
    still tears at its silhouettes, which is what makes it composite."""
    _patch_cliff_depth(monkeypatch)
    solve, _pivot, _eye = _synthetic_primary()

    meta = _patch(solve)

    assert meta["torn_fraction"] > 0.05, (
        "the cliff fixture must actually tear at the shipped thresholds, or "
        "this suite is not testing anything")


def test_tearing_thresholds_are_settable_on_the_patch(monkeypatch):
    """The fill path needs to say 'this layer has nothing behind it'."""
    _patch_cliff_depth(monkeypatch)
    solve, _pivot, _eye = _synthetic_primary()

    torn = _patch(solve)
    intact = _patch(solve, depth_edge_rel=1e9, max_edge_factor=0.0)

    assert intact["torn_fraction"] < 0.5 * torn["torn_fraction"]
    # More faces over the SAME vertices: the cells came back, the grid did not
    # change. A drop in vertices would mean something was excluded instead.
    assert intact["n_faces"] > torn["n_faces"]
    assert intact["n_vertices"] >= torn["n_vertices"]


def test_the_residual_floor_is_invalid_depth_not_the_silhouette_tests(monkeypatch):
    """With both silhouette tests off, what remains torn must be INDEPENDENT of
    the depth range -- which is how we know it is a different mechanism.

    Measured: a constant 0.114 across a 15x depth ratio, a 1.8x one and a 1.4x
    one. It is the ~120 vertices the sky heuristic invalidates, each killing the
    quads around it. That floor is NOT what this fix addresses and is left
    alone: a fill patch can legitimately contain sky, and inventing geometry
    where there is no depth is a different decision from refusing to tear.
    """
    solve, _pivot, _eye = _synthetic_primary()
    no_tear = dict(depth_edge_rel=1e9, max_edge_factor=0.0)

    floors = []
    for near, far in ((3.0, 45.0), (10.0, 18.0), (10.0, 14.0)):
        _patch_cliff_depth(monkeypatch, near=near, far=far)
        floors.append(_patch(solve, **no_tear)["torn_fraction"])

    assert max(floors) - min(floors) < 1e-6, floors
    # And the shipped thresholds DO track the depth range, which is the
    # contrast that makes the point.
    _patch_cliff_depth(monkeypatch, near=3.0, far=45.0)
    wide = _patch(solve)["torn_fraction"]
    _patch_cliff_depth(monkeypatch, near=10.0, far=14.0)
    narrow = _patch(solve)["torn_fraction"]
    assert wide > narrow


def test_defaults_are_the_relief_mesh_defaults(monkeypatch):
    """Appended widgets must not move any existing patch. Passing the defaults
    explicitly has to be identical to not passing them at all."""
    _patch_cliff_depth(monkeypatch)
    solve, _pivot, _eye = _synthetic_primary()

    implicit = _patch(solve)
    explicit = _patch(solve, depth_edge_rel=0.5, max_edge_factor=12.0)

    assert implicit["n_faces"] == explicit["n_faces"]
    assert implicit["torn_fraction"] == pytest.approx(explicit["torn_fraction"])


def test_fill_occluded_wires_the_no_tear_thresholds_into_every_patch():
    """The automated loop must not need an artist to know about this."""
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        from test_fill_nodes import (_fill_node, _img,
                                     _wall_solve_with_two_occluders,
                                     _FakeDepth as _FD)
        from atlas_camera.comfy.nodes_fill import AtlasCameraMovePreset

        node = _fill_node(mp)
        solve = _wall_solve_with_two_occluders()
        source = _img(np.full((96, 128, 3), 120, np.uint8))
        path, _exact, _r = AtlasCameraMovePreset().build(solve, "arc_left",
                                                         angle_deg=35.0)
        out = node.fill(solve, source, model="M", clip="C", vae="V",
                        camera_path=path, primary_depth=_FD(),
                        min_area_px=16, snap=16, max_rois=4)
    finally:
        mp.undo()

    patches = [n for n in out["expand"].values()
               if n["class_type"] == "AtlasAddPatchView"]
    assert patches, "the fixture must expand at least one patch"
    for p in patches:
        # The RATIO test is relaxed: that is the one that tears at a legitimate
        # cliff inside a fill, and a fill has nothing behind it.
        assert p["inputs"]["depth_edge_rel"] >= 1e6
        # The EDGE-LENGTH test is NOT disabled, it is loosened. It is the only
        # thing standing between a hallucinated far-depth corner and a
        # metres-long shard, and a shard exports to the DCC as real geometry.
        assert 0.0 < p["inputs"]["max_edge_factor"] < 12.0 * 10


def test_the_edge_length_guard_is_loosened_but_never_disabled():
    """Measured 2026-09-04 on the sea-cliff castle by dropping exactly the faces
    each candidate would tear and re-rendering against the FILLABLE hole:

        max_edge_factor   fillable closed   worst edge
        0 (disabled)          72%            11.13 m
        100                   69%             5.56 m
        40                    66%             2.39 m
        12 (relief default)   61%             0.72 m

    The coverage curve is nearly flat and the shard curve is not, so disabling
    it bought 6 points over 40 and licensed an 11 m triangle in a scene ~20 m
    deep. A hole is honest; an 11 m smear across a depth gap is geometry an
    artist takes into Maya believing it.
    """
    from atlas_camera.comfy.nodes_fill import (_FILL_NO_TEAR_DEPTH_EDGE_REL,
                                               _FILL_MAX_EDGE_FACTOR)

    assert _FILL_NO_TEAR_DEPTH_EDGE_REL >= 1e6
    assert 12.0 < _FILL_MAX_EDGE_FACTOR <= 100.0


# ------------------------------------------------ the sky heuristic on a fill

def test_torn_fraction_separates_the_mask_bound_from_real_tearing():
    """`torn_fraction` is 1 - faces/FULL grid, and a fill patch's mesh is
    deliberately bounded to the hole -- so most of that number is the bound,
    not tearing.

    Measured on a 34% hole: mask-bounded with every silhouette test off still
    reports 0.658, which is just 1 - 0.342. Quoting it as a tearing figure (as
    every note about these patches did until 2026-09-04) overstates tearing by
    whatever the mask excluded. `excluded_fraction` and
    `torn_fraction_eligible` split the two.
    """
    from atlas_camera.core.relief_mesh import build_relief_mesh

    H = W = 128
    depth = np.full((H, W), 10.0)
    hole = np.zeros((H, W), bool)
    hole[20:100, 25:95] = True                    # 34.2% of the rect
    K = dict(view_matrix=((1, 0, 0, 0), (0, 1, 0, -1.6), (0, 0, 1, 0),
                          (0, 0, 0, 1)),
             fx=200.0, fy=200.0, cx=W / 2, cy=H / 2, grid_long_edge=64,
             scale=1.0, horizon_y=H * 0.45, apply_sky_heuristic=False,
             depth_edge_rel=1e9, max_edge_factor=0.0)

    unbounded = build_relief_mesh(depth, **K).stats
    bounded = build_relief_mesh(depth, exclude_mask=~hole, **K).stats

    # Nothing tears in either: the silhouette tests are off and depth is flat.
    assert unbounded["torn_fraction"] < 0.01
    assert bounded["torn_fraction"] > 0.5, "the bound must dominate the old figure"
    assert bounded["excluded_fraction"] > 0.5
    assert bounded["torn_fraction_eligible"] < 0.05, (
        "with the bound accounted for, nothing was actually torn")


def test_the_sky_heuristic_costs_a_fill_patch_a_third_of_its_geometry():
    """Why a fill turns it off, in the same measurement that justifies it.

    The heuristic kills pixels above the horizon whose depth is far or ROUGH,
    and a fill's depth is a monocular estimate over INVENTED content, which is
    rough by construction. Measured on a fill-shaped mesh it removed 35% of the
    faces inside the hole -- geometry the patch exists to supply, in a layer
    with nothing behind it, so what it leaves is a hole rather than slightly
    wrong depth. Same trade as the tearing fix.
    """
    from atlas_camera.core.relief_mesh import build_relief_mesh

    H = W = 128
    rng = np.random.default_rng(5)
    blocks = rng.uniform(0, 1, size=(8, 8)) < 0.45
    f = np.repeat(np.repeat(np.where(blocks, 1.0, 1.9), H // 8, 0), W // 8, 1)
    depth = np.full((H, W), 10.0) * f
    hole = np.zeros((H, W), bool)
    hole[20:100, 25:95] = True
    K = dict(view_matrix=((1, 0, 0, 0), (0, 1, 0, -1.6), (0, 0, 1, 0),
                          (0, 0, 0, 1)),
             fx=200.0, fy=200.0, cx=W / 2, cy=H / 2, grid_long_edge=64,
             scale=1.0, horizon_y=H * 0.45, exclude_mask=~hole,
             depth_edge_rel=1e9, max_edge_factor=0.0)

    on = build_relief_mesh(depth, apply_sky_heuristic=True, **K).stats
    off = build_relief_mesh(depth, apply_sky_heuristic=False, **K).stats

    assert off["n_faces"] > on["n_faces"] * 1.3
    # It EXCLUDES cells rather than tearing them, so the eligible-tear figure
    # stays ~0 either way and the cost shows up as exclusion. Asserting it the
    # other way round would be asserting the wrong mechanism.
    assert on["excluded_fraction"] > off["excluded_fraction"] + 0.05
    assert on["torn_fraction_eligible"] < 0.05


def test_fill_occluded_turns_the_sky_heuristic_off_on_its_patches():
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        from test_fill_nodes import (_fill_node, _img,
                                     _wall_solve_with_two_occluders,
                                     _FakeDepth as _FD)
        from atlas_camera.comfy.nodes_fill import AtlasCameraMovePreset

        node = _fill_node(mp)
        solve = _wall_solve_with_two_occluders()
        source = _img(np.full((96, 128, 3), 120, np.uint8))
        path, _e, _r = AtlasCameraMovePreset().build(solve, "arc_left",
                                                     angle_deg=35.0)
        out = node.fill(solve, source, model="M", clip="C", vae="V",
                        camera_path=path, primary_depth=_FD(),
                        min_area_px=16, snap=16, max_rois=4)
    finally:
        mp.undo()

    patches = [n for n in out["expand"].values()
               if n["class_type"] == "AtlasAddPatchView"]
    assert patches
    for p in patches:
        assert p["inputs"]["sky_heuristic"] is False
