"""core.conditioning: every signal checked against a CLOSED FORM, not a golden.

The scenes here are fronto-parallel quads at known depth under a known camera
translation, because that configuration has an exact analytic answer -- a
plane's flow is ``-fx*t/Z`` in u and exactly zero in v. A golden-image pin would
catch a regression but would not catch a plausible-looking wrong sign, wrong
axis, or off-by-half-pixel convention; the closed form catches all three, which
is the whole reason flow is derived here instead of estimated.

numpy only: no torch, no ComfyUI. This is the layer that must stay testable
without a host.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlas_camera.core.conditioning import (
    PARALLAX_FLOOR_PX, render_conditioning_sequence,
)
from atlas_camera.core.ghost_pixels import GhostClass
from atlas_camera.core.projection_render import project_points

W, H = 160, 120
FX = FY = 200.0
CX, CY = W / 2.0, H / 2.0
K = [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]]
WALL_Z = 10.0


def view_at(x: float):
    """Camera at (x, 0, 0) looking down world -Z, Y up. World->camera 4x4."""
    vm = np.eye(4)
    vm[0, 3] = -x
    return vm


def quad(z: float, half_w: float, half_h: float, label: str):
    verts = np.array([[-half_w, -half_h, -z], [half_w, -half_h, -z],
                      [half_w, half_h, -z], [-half_w, half_h, -z]],
                     dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                   dtype=np.float64)
    return (label, verts, faces, uvs, "primary", {})


@pytest.fixture
def textures():
    return {"primary": np.ones((8, 8, 4), dtype=np.float64)}


@pytest.fixture
def wall():
    return [quad(WALL_Z, 40.0, 30.0, "wall")]


def render(meshes, textures, views, intrinsics=None, **kw):
    return render_conditioning_sequence(
        meshes, textures, views=views,
        intrinsics=intrinsics or [K] * len(views),
        plate_view=views[0], plate_k=K, width=W, height=H, **kw)


def pixel_grid():
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    return np.stack([uu, vv], axis=-1)


def test_depth_is_the_z_buffer_not_a_normalised_preview(wall, textures):
    """Metric metres, straight off the render. A 10 m wall reads 10.0."""
    seq = render(wall, textures, [view_at(0.0)])
    assert np.allclose(seq.depth_m[0], WALL_Z, atol=1e-6)


def test_holes_are_nan_never_zero_and_never_a_sentinel_distance(textures):
    """Zero is a legal coordinate and 1e6 is a legal depth; a hole is neither."""
    small = [quad(WALL_Z, 1.0, 1.0, "small")]
    seq = render(small, textures, [view_at(0.0)])
    hole = seq.alpha[0] <= 0.0
    assert hole.any(), "the scene must not fill the frame for this test"
    assert np.isnan(seq.depth_m[0][hole]).all()
    assert np.isnan(seq.position_world[0][hole]).all()


def test_lateral_flow_equals_the_closed_form(wall, textures):
    """A fronto-parallel plane: flow_u == -fx*t/Z exactly, flow_v == 0."""
    t = 0.5
    seq = render(wall, textures, [view_at(0.0), view_at(t)])
    sel = seq.flow_valid_fwd[0] & ~seq.flow_occluded_fwd[0]
    assert sel.any()
    du = seq.flow_fwd[0][..., 0][sel]
    dv = seq.flow_fwd[0][..., 1][sel]
    assert np.abs(du - (-FX * t / WALL_Z)).max() < 1e-9
    assert np.abs(dv).max() < 1e-9


def test_flow_scales_with_inverse_depth(textures):
    """Twice as far, half the parallax. The defining property of a motion field."""
    t = 0.5
    near = render([quad(5.0, 40.0, 30.0, "near")], textures,
                  [view_at(0.0), view_at(t)])
    far = render([quad(10.0, 40.0, 30.0, "far")], textures,
                 [view_at(0.0), view_at(t)])
    du_near = np.nanmedian(near.flow_fwd[0][..., 0])
    du_far = np.nanmedian(far.flow_fwd[0][..., 0])
    assert du_near == pytest.approx(2.0 * du_far, rel=1e-9)


def test_a_static_camera_has_zero_flow(wall, textures):
    seq = render(wall, textures, [view_at(0.0), view_at(0.0)])
    sel = seq.flow_valid_fwd[0]
    assert np.abs(seq.flow_fwd[0][sel]).max() < 1e-9


def test_a_zoom_flows_radially_with_a_stationary_eye(wall, textures):
    """A focal change moves every pixel ALONG its radius from the principal
    point. Zero tangential component is the signature; anything else means the
    eye moved, which a zoom must not do."""
    k_zoomed = [[FX * 1.25, 0.0, CX], [0.0, FY * 1.25, CY], [0.0, 0.0, 1.0]]
    seq = render(wall, textures, [view_at(0.0)] * 2, intrinsics=[K, k_zoomed])
    flow = seq.flow_fwd[0]
    radius = pixel_grid() - np.array([CX, CY])
    tangential = flow[..., 0] * radius[..., 1] - flow[..., 1] * radius[..., 0]
    sel = seq.flow_valid_fwd[0]
    assert np.abs(tangential[sel]).max() < 1e-9
    assert np.abs(flow[sel]).max() > 1.0, "the zoom must actually move pixels"


def test_a_zoom_cannot_disocclude(wall, textures):
    """No eye motion, no parallax, so no newly-visible surface. Mirrors the
    same assertion AtlasGhostPixelMap's suite makes about the class map."""
    k_zoomed = [[FX * 1.25, 0.0, CX], [0.0, FY * 1.25, CY], [0.0, 0.0, 1.0]]
    seq = render(wall, textures, [view_at(0.0)] * 2, intrinsics=[K, k_zoomed])
    assert (seq.class_map[1] == int(GhostClass.GHOST)).sum() == 0


def test_the_occlusion_flag_fires_where_a_blocker_sweeps_across(textures):
    """The blocker subtends 40 px at z=3 and sweeps 20 px against the wall's 6
    for a 0.3 m move, so it crosses 14 px of previously-visible wall over its
    40 px height. 560 px is that rectangle, exactly -- the flag is not merely
    non-empty, it is the right size."""
    meshes = [quad(WALL_Z, 40.0, 30.0, "wall"), quad(3.0, 0.3, 0.3, "blocker")]
    seq = render(meshes, textures, [view_at(0.0), view_at(0.3)])
    sweep_px = FX * 0.3 / 3.0 - FX * 0.3 / WALL_Z
    blocker_h = 2 * FY * 0.3 / 3.0
    assert int(seq.flow_occluded_fwd[0].sum()) == pytest.approx(
        sweep_px * blocker_h, rel=0.05)


def test_backward_flow_is_measured_not_negated(textures):
    """Where a pixel is visible in both frames the two agree; the point of
    computing backward flow separately is the pixels where they cannot."""
    meshes = [quad(WALL_Z, 40.0, 30.0, "wall")]
    seq = render(meshes, textures, [view_at(0.0), view_at(0.4)])
    both = seq.flow_valid_bwd[1] & ~seq.flow_occluded_bwd[1]
    assert both.any()
    # frame 1 -> frame 0 is the reverse translation, so +fx*t/Z.
    du = seq.flow_bwd[1][..., 0][both]
    assert np.abs(du - (FX * 0.4 / WALL_Z)).max() < 1e-9


def test_edge_frames_have_nan_flow_not_zero_flow(wall, textures):
    """The first frame has no previous and the last no next. A zero vector is
    what a genuinely static pixel reads, so the two must not look alike."""
    seq = render(wall, textures, [view_at(0.0), view_at(0.3)])
    assert np.isnan(seq.flow_bwd[0]).all()
    assert not seq.flow_valid_bwd[0].any()
    assert np.isnan(seq.flow_fwd[-1]).all()
    assert not seq.flow_valid_fwd[-1].any()


def test_position_reprojects_to_its_own_pixel(wall, textures):
    """The XYZ pass and the depth pass describe the same surface. Tolerance is
    float32 storage (what EXR and torch want), not the float64 math, which the
    flow assertions above land at exactly 0.0."""
    seq = render(wall, textures, [view_at(0.0)])
    pts = seq.position_world[0].reshape(-1, 3).astype(np.float64)
    px, _ = project_points(pts, view_at(0.0), FX, FY, CX, CY)
    ok = np.isfinite(px).all(axis=1)
    assert ok.any()
    assert np.abs(px[ok] - pixel_grid().reshape(-1, 2)[ok]).max() < 1e-4


def test_normals_are_unit_and_face_the_camera(wall, textures):
    """A wall facing down -Z has a +Z world normal; only the magnitude is a
    hard invariant, so both are asserted rather than only the cheap one."""
    seq = render(wall, textures, [view_at(0.0)])
    n = seq.normal_world[0][seq.normal_valid[0]]
    assert np.abs(np.linalg.norm(n, axis=-1) - 1.0).max() < 1e-9
    assert np.abs(n[:, 2]).min() > 0.99


def test_parallax_is_reported_and_flat_frames_are_named(wall, textures):
    """A move's last frame has no forward neighbour, so it reads 0 parallax and
    must appear in frames_without_parallax -- the scorer excludes exactly these
    from its aggregate rather than scoring a frame with nothing to get right."""
    seq = render(wall, textures, [view_at(0.0), view_at(0.5), view_at(1.0)])
    parallax = seq.meta["parallax_px"]
    assert parallax[0] == pytest.approx(FX * 0.5 / WALL_Z)
    assert parallax[-1] == 0.0
    assert seq.meta["frames_without_parallax"] == [2]
    assert PARALLAX_FLOOR_PX > 0.0


def test_a_mismatched_intrinsics_list_refuses(wall, textures):
    """A silent fallback to the solved focal renders a zoom as no change at
    all, which is precisely the failure this refusal exists to prevent."""
    with pytest.raises(ValueError, match="one K per view"):
        render_conditioning_sequence(
            wall, textures, views=[view_at(0.0), view_at(0.3)],
            intrinsics=[K], plate_view=view_at(0.0), plate_k=K,
            width=W, height=H)


def test_an_exclude_mask_off_the_render_raster_refuses(wall, textures):
    with pytest.raises(ValueError, match="render raster"):
        render(wall, textures, [view_at(0.0)],
               exclude=np.zeros((H + 1, W), dtype=bool))


def test_opting_out_drops_the_field_entirely(wall, textures):
    """None, not a zero array: an all-zero flow field is indistinguishable from
    a measured static scene."""
    seq = render(wall, textures, [view_at(0.0), view_at(0.3)],
                 want_flow=False, want_normals=False, want_position=False)
    assert seq.flow_fwd is None
    assert seq.normal_world is None
    assert seq.position_world is None
    assert seq.depth_m is not None


def test_metadata_states_what_the_path_cannot_express(wall, textures):
    """A reader must not infer a shifted-sensor channel from silence."""
    seq = render(wall, textures, [view_at(0.0)])
    assert seq.meta["principal_point"] == "static"
    assert "metres" in seq.meta["depth_units"]
    assert "pixels" in seq.meta["flow_units"]


def test_dilation_matches_the_ghost_nodes_own_helper(textures):
    """core uses mask_ops.dilate; AtlasGhostPixelMap uses its own square-roll
    _dilate. They must be the same operation or the two classify differently
    for the same inputs."""
    from atlas_camera.core.mask_ops import dilate
    rng = np.random.default_rng(7)
    node_dilate = _node_dilate()
    for radius in (1, 2, 3, 5):
        mask = rng.random((37, 53)) < 0.04
        assert np.array_equal(node_dilate(mask.copy(), radius, np),
                              dilate(mask, iterations=radius, connectivity=8))


def _node_dilate():
    guide = pytest.importorskip(
        "atlas_camera.comfy.nodes_viewport",
        reason="the comparison needs the comfy wrapper; core does not")
    return guide.AtlasDisocclusionGuide._dilate
