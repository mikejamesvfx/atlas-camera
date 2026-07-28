"""Tests for the camera move budget — the forward disocclusion measurement.

The headline test is ANALYTIC, and deliberately needs no photograph, no depth
model and no GPU: a fronto-parallel occluder slab at depth ``d1`` in front of a
background plane at ``d2``, viewed by a pinhole camera that then dollies
laterally by ``t``.

There are two closed forms, one per measure, and both are exercised.

**Raw coverage** (``disocclusion_fraction``, the rasterizer's own correctness):
the newly-exposed strip behind the slab is ``f*t*(1/d1 - 1/d2)`` pixels wide and
the band entering at frame edge is ``f*t/d2``. Those sum to ``f*t/d1`` — total
uncovered area depends only on the NEAREST surface, independent of the
background distance. That independence makes it a strong oracle rather than a
tautology: a rasterizer that mishandled tears or the frame boundary would fail
it while still matching at a single distance.

**Tear disocclusion** (``tear_disocclusion_fraction``, what the budget reports):
sealed-minus-covered counts only the strip, ``f*t*(1/d1 - 1/d2)``. The
frame-edge band drops out because the sealed surface does not extend past the
plate either — nothing photographed is being torn away there.

Two results worth recognising as correct rather than suspicious: pan and tilt
come back unbounded, because rotation about the optical centre produces no
parallax and so cannot open a tear; and ``dolly_y`` is unbounded because the
slab spans full frame height, so vertical motion reveals nothing behind it.

The mesh here is built by the test itself rather than by
``relief_mesh.build_relief_mesh`` — deliberate test hygiene, so a bug in
production mesh construction cannot hide inside the oracle.
"""

import numpy as np
import pytest

from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.move_budget import (
    AtlasMoveBudget,
    coverage_backends,
    disocclusion_fraction,
    estimate_move_budget,
    offset_view_matrix,
    rasterize_coverage,
)
from atlas_camera.core.relief_mesh import ReliefMesh
from atlas_camera.core.schema import (
    AtlasCamera,
    AtlasCameraKeyframe,
    AtlasCameraPath,
    AtlasExtrinsics,
    AtlasSolve,
)


# --- test-local scene construction (independent of production mesh code) ---

def _slab_depth(width=400, height=200, d_near=2.0, d_far=8.0,
                slab_frac=0.25) -> np.ndarray:
    """Depth map: a full-height central slab at ``d_near``, background ``d_far``."""
    depth = np.full((height, width), float(d_far))
    half = int(width * slab_frac / 2)
    depth[:, width // 2 - half: width // 2 + half] = float(d_near)
    return depth


def _torn_grid_mesh(depth, *, fx, fy, cx, cy, edge_rel=0.1):
    """Triangulate a depth grid at pixel centres, tearing quads that span a
    depth discontinuity — the same silhouette-tear rule the relief mesh uses,
    reimplemented minimally so this test does not depend on it.

    The camera is at the origin looking down -Z with identity rotation, so
    camera space and world space coincide.
    """
    height, width = depth.shape
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    x = (uu - cx) / fx * depth
    y = -(vv - cy) / fy * depth
    z = -depth
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    idx = np.arange(height * width).reshape(height, width)
    tl, tr = idx[:-1, :-1], idx[:-1, 1:]
    bl, br = idx[1:, :-1], idx[1:, 1:]

    quad = np.stack([depth[:-1, :-1], depth[:-1, 1:],
                     depth[1:, :-1], depth[1:, 1:]], axis=-1)
    span = quad.max(axis=-1) - quad.min(axis=-1)
    keep = span <= edge_rel * quad.min(axis=-1)

    faces = np.concatenate([
        np.stack([tl[keep], bl[keep], br[keep]], axis=-1),
        np.stack([tl[keep], br[keep], tr[keep]], axis=-1),
    ], axis=0).astype(np.int32)
    return vertices, faces


def _lateral_view_matrix(t: float) -> np.ndarray:
    """Camera dollied to x=t, orientation unchanged (still looking down -Z)."""
    view, _, _ = look_at_view_matrix((t, 0.0, 0.0), (t, 0.0, -1.0), (0.0, 1.0, 0.0))
    return np.asarray(view, dtype=np.float64)


def _measure(t, *, backend, width=400, height=200, focal=400.0,
             d_near=2.0, d_far=8.0):
    depth = _slab_depth(width, height, d_near, d_far)
    cx, cy = width / 2.0, height / 2.0
    verts, faces = _torn_grid_mesh(depth, fx=focal, fy=focal, cx=cx, cy=cy)
    return disocclusion_fraction(
        verts, faces,
        view_matrix=_lateral_view_matrix(t),
        fx=focal, fy=focal, cx=cx, cy=cy,
        width=width, height=height, backend=backend,
    )


def _predicted(t, *, focal=400.0, d_near=2.0, width=400):
    """f*t/d_near pixels of hole, as a fraction of frame width."""
    return focal * abs(t) / d_near / width


ALL_BACKENDS = coverage_backends()


# --- the analytic oracle ---

@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_zero_motion_has_no_disocclusion(backend):
    # The original camera sees exactly what it recorded — every pixel covered.
    assert _measure(0.0, backend=backend) < 1e-3


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("t", [0.025, 0.05, 0.1])
def test_disocclusion_matches_closed_form_parallax(backend, t):
    measured = _measure(t, backend=backend)
    predicted = _predicted(t)
    # Tolerance is 2 pixels of strip width, the discretization floor.
    assert abs(measured - predicted) < 2.0 / 400


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_disocclusion_is_independent_of_background_distance(backend):
    """The strong form: total hole depends only on the NEAREST surface.

    Doubling and quadrupling the background distance redistributes the hole
    between the behind-slab strip and the frame-edge band, but must not change
    the total. A rasterizer that mishandled tears or the frame boundary would
    fail this while still passing the single-distance case.
    """
    near = _measure(0.05, backend=backend, d_far=8.0)
    far = _measure(0.05, backend=backend, d_far=20.0)
    farther = _measure(0.05, backend=backend, d_far=64.0)
    for value in (near, far, farther):
        assert abs(value - _predicted(0.05)) < 2.0 / 400
    assert abs(near - farther) < 2.0 / 400


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_disocclusion_grows_monotonically_with_motion(backend):
    fractions = [_measure(t, backend=backend) for t in (0.0, 0.02, 0.05, 0.1)]
    assert all(b > a for a, b in zip(fractions, fractions[1:]))


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_nearer_occluder_costs_more_budget(backend):
    """Halving the occluder distance doubles the disocclusion."""
    near = _measure(0.05, backend=backend, d_near=1.0)
    far = _measure(0.05, backend=backend, d_near=2.0)
    assert abs(near - 2.0 * far) < 4.0 / 400


# --- coverage raster shape/semantics ---

@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_rasterize_coverage_returns_full_frame_bool_mask(backend):
    depth = _slab_depth()
    verts, faces = _torn_grid_mesh(depth, fx=400.0, fy=400.0, cx=200.0, cy=100.0)
    coverage, zbuf = rasterize_coverage(
        verts, faces, view_matrix=_lateral_view_matrix(0.0),
        fx=400.0, fy=400.0, cx=200.0, cy=100.0,
        width=400, height=200, backend=backend,
    )
    assert coverage.shape == (200, 400)
    assert coverage.dtype == bool
    assert coverage.all()
    # Depth buffer carries forward distance where covered; slab reads nearer.
    assert np.isfinite(zbuf[coverage]).all()
    assert zbuf[100, 200] == pytest.approx(2.0, abs=0.05)
    assert zbuf[100, 5] == pytest.approx(8.0, abs=0.05)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_hole_appears_on_the_trailing_side_of_the_occluder(backend):
    """Dollying right (+x) exposes background on the occluder's right side.

    Guards against a sign error that would still pass the fraction tests.
    """
    depth = _slab_depth()
    verts, faces = _torn_grid_mesh(depth, fx=400.0, fy=400.0, cx=200.0, cy=100.0)
    coverage, _ = rasterize_coverage(
        verts, faces, view_matrix=_lateral_view_matrix(0.05),
        fx=400.0, fy=400.0, cx=200.0, cy=100.0,
        width=400, height=200, backend=backend,
    )
    row = ~coverage[100]
    # Ignore the frame-edge band; look only at the interior hole.
    interior = row[20:380]
    hole_cols = np.flatnonzero(interior) + 20
    assert hole_cols.size > 0
    slab_right_edge = 200 + int(400 * 0.25 / 2)  # 250
    assert np.all(hole_cols > 200), "hole must be right of frame centre"
    assert abs(hole_cols.min() - (slab_right_edge - 400 * 0.05 / 2.0)) < 4


# --- candidate camera construction ---

def test_offset_view_matrix_dolly_moves_the_eye_along_camera_axes():
    base = _lateral_view_matrix(0.0)
    moved = offset_view_matrix(base, dolly_x=0.5, dolly_z=0.25)
    eye = np.linalg.inv(moved)[:3, 3]
    # Camera looks down -Z, so a positive dolly_z moves the eye to -Z.
    assert eye == pytest.approx([0.5, 0.0, -0.25], abs=1e-9)


def test_offset_view_matrix_tilt_is_camera_frame_not_transposed():
    """A positive tilt must raise the view direction, not lower it.

    This is the transpose trap the coordinate rules warn about: rotating on the
    wrong side inverts pitch while leaving every magnitude identical, so only a
    directional assertion catches it.
    """
    base = _lateral_view_matrix(0.0)
    tilted = offset_view_matrix(base, tilt_deg=10.0)
    forward = -np.linalg.inv(tilted)[:3, 2]   # camera forward = -Z column
    assert forward[1] > 0.05, "positive tilt should aim the camera upward"
    assert forward[2] < 0.0, "camera must still face predominantly -Z"


def test_offset_view_matrix_pan_keeps_the_horizon_level():
    """Pan is a world-side yaw, so the camera's up vector stays world-up."""
    base = _lateral_view_matrix(0.0)
    panned = offset_view_matrix(base, pan_deg=25.0)
    up = np.linalg.inv(panned)[:3, 1]
    assert up == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


# --- envelope + path ---

def _slab_solve_and_mesh(width=400, height=200, focal=400.0, d_near=2.0, d_far=8.0):
    cx, cy = width / 2.0, height / 2.0
    depth = _slab_depth(width, height, d_near, d_far)
    verts, faces = _torn_grid_mesh(depth, fx=focal, fy=focal, cx=cx, cy=cy)
    mesh = ReliefMesh(vertices=verts, faces=faces,
                      uvs=np.zeros((len(verts), 2), dtype=np.float64))
    intr = build_intrinsics(image_width=width, image_height=height,
                            focal_length_mm=35.0, sensor_width_mm=36.0)
    intr.fx_px = intr.fy_px = focal
    intr.cx_px, intr.cy_px = cx, cy
    solve = AtlasSolve(
        camera=AtlasCamera(
            intrinsics=intr,
            extrinsics=AtlasExtrinsics(camera_view_matrix=_lateral_view_matrix(0.0)),
        ),
        image_width=width, image_height=height,
    )
    return solve, mesh


def _sealed_mesh(width=400, height=200, focal=400.0, d_near=2.0, d_far=8.0):
    """The same grid with tearing disabled — the surface envelope the depth map
    described before silhouette tears removed part of it."""
    cx, cy = width / 2.0, height / 2.0
    depth = _slab_depth(width, height, d_near, d_far)
    verts, faces = _torn_grid_mesh(depth, fx=focal, fy=focal, cx=cx, cy=cy,
                                   edge_rel=1e9)
    return ReliefMesh(vertices=verts, faces=faces,
                      uvs=np.zeros((len(verts), 2), dtype=np.float64))


def _solve_with_backdrop(solve, mesh, *, distance=200.0):
    """The solve with its relief mesh AND a far backdrop plane — the shape a
    real Atlas projection scene has, since proxy derivation always emits a
    cyclorama behind everything.

    Atlas planes live in their local XY plane with Z as the normal (the
    THREE.PlaneGeometry frame proxy_geometry._plane_transform builds), so an
    identity-rotation plane already faces the camera down -Z.
    """
    from atlas_camera.core.proxy_geometry import relief_mesh_primitive
    from atlas_camera.core.schema import AtlasProxyPrimitive

    backdrop = AtlasProxyPrimitive(
        name="projection_backdrop",
        primitive_type="plane",
        transform_matrix=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, -distance),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dimensions=(4.0 * distance, 4.0 * distance, 0.0),
        metadata={"role": "atlas_proxy"},
    )
    solve.projection_scene.proxy_geometry = [relief_mesh_primitive(mesh), backdrop]
    return solve


def test_tear_disocclusion_matches_closed_form_strip_width():
    """Sealed-minus-covered isolates the strip behind the occluder.

    Unlike the raw-coverage measure, the frame-edge band does NOT count: the
    sealed surface does not extend there either, so there is no photographed
    surface being torn away. What remains is exactly the parallax strip,
    ``f*t*(1/d_near - 1/d_far)`` pixels wide.
    """
    from atlas_camera.core.move_budget import tear_disocclusion_fraction

    focal, width, height = 400.0, 400, 200
    d_near, d_far, t = 2.0, 8.0, 0.05
    depth = _slab_depth(width, height, d_near, d_far)
    torn = _torn_grid_mesh(depth, fx=focal, fy=focal, cx=200.0, cy=100.0)
    sealed = _torn_grid_mesh(depth, fx=focal, fy=focal, cx=200.0, cy=100.0,
                             edge_rel=1e9)

    fraction, _, _ = tear_disocclusion_fraction(
        torn, sealed, view_matrix=_lateral_view_matrix(t),
        fx=focal, fy=focal, cx=200.0, cy=100.0,
        width=width, height=height, backend="numpy",
    )
    predicted = focal * t * (1.0 / d_near - 1.0 / d_far) / width
    assert abs(fraction - predicted) < 2.0 / width


def test_tear_disocclusion_ignores_a_backdrop_that_covers_everything():
    """A cyclorama behind the tear must not read as the tear being filled."""
    from atlas_camera.core.move_budget import tear_disocclusion_fraction

    solve, mesh = _slab_solve_and_mesh()
    scene = _solve_with_backdrop(solve, mesh)
    budget = estimate_move_budget(
        scene, sealed_mesh=_sealed_mesh(), threshold=0.02, backend="numpy",
        axes=("dolly_x",), bisect_steps=10, max_dolly_m=1.0,
    )
    # With the backdrop naively counted as coverage this saturates at the cap;
    # excluded, the real tear binds it near the closed-form limit.
    assert budget.dolly_x_m < 0.5
    assert "projection_backdrop" not in budget.geometry_sources


def test_estimate_move_budget_dolly_limit_matches_closed_form():
    """The lateral limit is where the parallax strip crosses the threshold."""
    solve, mesh = _slab_solve_and_mesh()
    budget = estimate_move_budget(
        solve, mesh=mesh, sealed_mesh=_sealed_mesh(), threshold=0.02,
        backend="numpy", axes=("dolly_x",), bisect_steps=10, max_dolly_m=1.0,
    )
    # threshold*W / (f*(1/d_near - 1/d_far)) = 0.02*400 / (400*0.375)
    expected = 0.02 * 400 / (400.0 * (1 / 2.0 - 1 / 8.0))
    assert budget.dolly_x_m == pytest.approx(expected, rel=0.2)
    assert budget.threshold == 0.02
    assert budget.samples, "probes must be recorded for plotting"


def test_estimate_move_budget_requires_geometry():
    solve, _ = _slab_solve_and_mesh()
    with pytest.raises(ValueError, match="no geometry"):
        estimate_move_budget(solve, backend="numpy", axes=("dolly_x",))


def test_rotation_is_not_penalised_for_merely_leaving_frame():
    """Panning off the edge of the plate is not a tear.

    Under a raw-coverage measure a pan of half a degree already pushed frame
    edge past the mesh boundary and read as disocclusion, collapsing rotation
    budgets by two orders of magnitude for a reason unrelated to occlusion.
    Sealed-minus-covered is immune by construction: the sealed surface does not
    extend past the plate either, so there is nothing there to be torn away.
    """
    solve, mesh = _slab_solve_and_mesh()
    budget = estimate_move_budget(
        solve, mesh=mesh, sealed_mesh=_sealed_mesh(), threshold=0.02,
        backend="numpy", axes=("pan",), bisect_steps=8, max_angle_deg=30.0,
    )
    assert budget.pan_deg > 5.0


def test_unsealed_mesh_is_reported_rather_than_read_as_an_infinite_budget():
    """A sealing pass that silently does nothing must not look like good news.

    The mesh here carries placeholder UVs, so the grid lattice cannot be
    recovered and nothing gets sealed — which would otherwise make every
    measurement read zero disocclusion and hand back an unbounded budget.

    It must also have a genuine INTERIOR hole. A relief mesh whose only open
    boundary is the plate perimeter has nothing to seal, and warning there
    would fire on essentially every real scene.
    """
    from atlas_camera.core.move_budget import _unsealed_warning

    depth = np.full((200, 400), 6.0)
    depth[90:110, 150:180] = np.nan          # an enclosed gap, not the perimeter
    verts, faces = _torn_grid_mesh(depth, fx=400.0, fy=400.0, cx=200.0, cy=100.0)
    uvs = np.stack([np.tile(np.linspace(0, 1, 400), 200),
                    np.repeat(np.linspace(1, 0, 200), 400)], axis=1)
    mesh = ReliefMesh(vertices=verts, faces=faces, uvs=uvs)

    # sealed is the SAME mesh: the sealing pass added nothing.
    warning = _unsealed_warning(mesh, mesh, image_width=400, image_height=200)
    assert warning is not None
    assert "NOT trustworthy" in warning


def test_a_mesh_whose_only_boundary_is_the_plate_perimeter_is_not_flagged():
    """The false positive that made the warning useless on real scenes.

    A relief mesh built from a frame-filling depth map is a complete lattice
    bounded only by the plate edge. Flagging that as an unsealed tear fired on
    every real solve and would have trained the warning to be ignored.
    """
    from atlas_camera.core.move_budget import _unsealed_warning

    depth = np.full((200, 400), 6.0)
    verts, faces = _torn_grid_mesh(depth, fx=400.0, fy=400.0, cx=200.0, cy=100.0)
    uvs = np.stack([np.tile(np.linspace(0, 1, 400), 200),
                    np.repeat(np.linspace(1, 0, 200), 400)], axis=1)
    mesh = ReliefMesh(vertices=verts, faces=faces, uvs=uvs)

    assert _unsealed_warning(mesh, mesh, image_width=400, image_height=200) is None


def test_estimate_move_budget_reports_saturation_rather_than_a_false_limit():
    """A threshold nothing can exceed must be flagged, not reported as a limit."""
    solve, mesh = _slab_solve_and_mesh()
    budget = estimate_move_budget(
        solve, mesh=mesh, threshold=0.95, backend="numpy",
        axes=("dolly_x",), bisect_steps=4, max_dolly_m=0.5,
    )
    assert "dolly_x" in budget.saturated
    assert budget.dolly_x_m == pytest.approx(0.5)


def test_estimate_move_budget_is_tighter_for_a_nearer_occluder():
    near, near_mesh = _slab_solve_and_mesh(d_near=1.0)
    far, far_mesh = _slab_solve_and_mesh(d_near=4.0)
    kwargs = dict(threshold=0.02, backend="numpy", axes=("dolly_x",),
                  bisect_steps=10, max_dolly_m=2.0)
    near_budget = estimate_move_budget(near, mesh=near_mesh,
                                       sealed_mesh=_sealed_mesh(d_near=1.0), **kwargs)
    far_budget = estimate_move_budget(far, mesh=far_mesh,
                                      sealed_mesh=_sealed_mesh(d_near=4.0), **kwargs)
    assert near_budget.dolly_x_m < far_budget.dolly_x_m


def _straight_dolly_path(distance: float, frames: int = 5) -> AtlasCameraPath:
    return AtlasCameraPath(
        keyframes=[
            AtlasCameraKeyframe(frame_index=0, position=(0.0, 0.0, 0.0),
                                target=(0.0, 0.0, -1.0)),
            AtlasCameraKeyframe(frame_index=frames - 1, position=(distance, 0.0, 0.0),
                                target=(distance, 0.0, -1.0)),
        ],
        frame_count=frames,
    )


def test_camera_path_within_budget_is_accepted():
    solve, mesh = _slab_solve_and_mesh()
    budget = estimate_move_budget(
        solve, mesh=mesh, sealed_mesh=_sealed_mesh(), threshold=0.05,
        backend="numpy", axes=(),
        camera_path=_straight_dolly_path(0.02),
    )
    assert budget.path_within_budget is True
    assert len(budget.path_frames) == 5
    assert budget.path_worst_fraction <= 0.05


def test_camera_path_beyond_budget_is_rejected_and_names_the_worst_frame():
    solve, mesh = _slab_solve_and_mesh()
    budget = estimate_move_budget(
        solve, mesh=mesh, sealed_mesh=_sealed_mesh(), threshold=0.02,
        backend="numpy", axes=(),
        camera_path=_straight_dolly_path(0.5),
    )
    assert budget.path_within_budget is False
    # Disocclusion grows with distance, so the last frame is the worst.
    assert budget.path_worst_frame == 4
    assert budget.path_worst_fraction > 0.02


def test_move_budget_round_trips_through_dict():
    solve, mesh = _slab_solve_and_mesh()
    budget = estimate_move_budget(
        solve, mesh=mesh, sealed_mesh=_sealed_mesh(), threshold=0.02,
        backend="numpy", axes=("dolly_x",), bisect_steps=4, max_dolly_m=0.5,
        camera_path=_straight_dolly_path(0.1, frames=3),
    )
    restored = AtlasMoveBudget.from_dict(budget.to_dict())
    assert restored.dolly_x_m == pytest.approx(budget.dolly_x_m, abs=1e-5)
    assert restored.threshold == budget.threshold
    assert restored.path_within_budget == budget.path_within_budget
    assert restored.path_worst_frame == budget.path_worst_frame
    assert len(restored.path_frames) == len(budget.path_frames)


def test_describe_is_human_readable_and_names_the_path_verdict():
    solve, mesh = _slab_solve_and_mesh()
    budget = estimate_move_budget(
        solve, mesh=mesh, sealed_mesh=_sealed_mesh(), threshold=0.02,
        backend="numpy", axes=("dolly_x",), bisect_steps=4, max_dolly_m=0.5,
        camera_path=_straight_dolly_path(0.5),
    )
    text = budget.describe()
    assert "Safe camera envelope" in text
    assert "EXCEEDS budget" in text


def test_scene_triangles_include_projection_source_layer_meshes():
    """Clean-plate layers live on ProjectionSource, not in proxy_geometry.

    Without collecting them, a coverage measurement is blind to exactly the
    geometry the layered workflow adds — found live when a clean-plate layer
    visibly filling a tear changed nothing in the move budget. The budget's
    covered set must count layer meshes, and the report must name them.
    """
    from atlas_camera.core.primitive_mesh import collect_scene_triangles
    from atlas_camera.core.proxy_geometry import relief_mesh_primitive
    from atlas_camera.core.schema import ProjectionSource

    solve, mesh = _slab_solve_and_mesh()
    solve.projection_scene.proxy_geometry = [relief_mesh_primitive(mesh)]

    layer_mesh = ReliefMesh(
        vertices=np.array([[0.0, 0.0, -30.0], [1.0, 0.0, -30.0], [0.0, 1.0, -30.0]]),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        uvs=np.zeros((3, 2)))
    solve.projection_sources.append(ProjectionSource(
        camera=solve.camera, name="cleanplate_background",
        proxy_geometry=[relief_mesh_primitive(
            layer_mesh, name="cleanplate_background_relief_mesh")]))

    verts, faces, sources = collect_scene_triangles(solve)
    assert any(s.startswith("layer:cleanplate_background/") for s in sources)
    n_base = len(mesh.faces)
    assert len(faces) == n_base + 1

    # And the opt-out still exists for callers that want the scene alone.
    _, faces_scene_only, src2 = collect_scene_triangles(
        solve, include_projection_sources=False)
    assert len(faces_scene_only) == n_base
    assert not any(s.startswith("layer:") for s in src2)
