"""Contract for the sub-quad layered cut at depth cliffs.

`build_relief_mesh` tears per grid cell, so a torn silhouette is a staircase of
amplitude one cell — and because the WHOLE cell goes, a cell's worth of real
surface is discarded on both sides of the cliff. Measured on the diagonal-cliff
fixture below at `relief_grid=128` on a 1024 px plate (step = 8 px), the shipped
boundary sits a mean of 5.67 px from the true cliff: *worse* than the step/2
quantization bound, which is the signature of that whole-cell retreat.

`sub_quad_boundary=True` recovers both sheets up to the cliff without joining
them. These tests pin the four things that make that safe rather than merely
prettier: the cut is additive (no existing face moves), no triangle spans the
cliff (the tear survives), the winding still faces the camera, and the UVs stay
projective.

The diagonal cliff is deliberate. An axis-aligned cliff lands on the lattice for
free and would flatter every method including the one that does nothing.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from atlas_camera.core.relief_mesh import build_relief_mesh

H = W = 512
FX = FY = 450.0
CX, CY = W / 2.0, H / 2.0
VIEW = np.eye(4, dtype=np.float64)
GRID = 64
SLOPE, INTERCEPT = 0.6, 90.0
NEAR_M, FAR_M = 4.0, 12.0

BUILD = dict(
    view_matrix=VIEW, fx=FX, fy=FY, cx=CX, cy=CY, grid_long_edge=GRID,
    depth_edge_rel=0.5, max_edge_factor=12.0, floor_clamp=None,
    apply_sky_heuristic=False, quad_coherence=True,
)


@pytest.fixture(scope="module")
def cliff_depth():
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    return np.where(ys > (xs * SLOPE + INTERCEPT), NEAR_M, FAR_M).astype(np.float32)


@pytest.fixture(scope="module")
def meshes(cliff_depth):
    base = build_relief_mesh(cliff_depth, **BUILD)
    cut = build_relief_mesh(cliff_depth, sub_quad_boundary=True, **BUILD)
    return base, cut


def _distance_to_cliff(px, py):
    return np.abs(SLOPE * px - py + INTERCEPT) / np.hypot(SLOPE, 1.0)


def _boundary_error(mesh):
    """Mean pixel distance from boundary vertices to the analytic cliff."""
    from atlas_camera.core.mesh_repair import boundary_edges, walk_loops

    uv = np.asarray(mesh.uvs, dtype=np.float64)
    px = uv[:, 0] * (W - 1)
    py = (1.0 - uv[:, 1]) * (H - 1)
    loops = walk_loops(boundary_edges(np.asarray(mesh.faces, dtype=np.int64)))
    idx = np.unique(np.concatenate([np.asarray(l, dtype=np.int64) for l in loops]))
    d = _distance_to_cliff(px[idx], py[idx])
    step = max(H, W) / GRID
    d = d[d < 3.0 * step]  # the plate frame is a boundary too, and legitimately straight
    assert d.size, "no boundary near the cliff — the fixture is not exercising a tear"
    return float(d.mean()), float(np.percentile(d, 95))


def _camera_position():
    return np.linalg.inv(VIEW)[:3, 3]


def test_the_cut_moves_the_boundary_onto_the_real_cliff(meshes):
    base, cut = meshes
    base_mean, _ = _boundary_error(base)
    cut_mean, _ = _boundary_error(cut)
    step = max(H, W) / GRID

    # The shipped boundary is worse than the step/2 quantization bound because
    # the whole cell is discarded, not just quantized.
    assert base_mean > 0.5 * step
    # The cut must land inside one step and beat the baseline by a wide margin —
    # a marginal gain would mean it is interpolating the two lattice samples
    # (which for step data always lands mid-edge) rather than reading the plate.
    assert cut_mean < 0.5 * base_mean
    assert cut_mean < step


def test_the_cut_is_additive_and_never_moves_existing_geometry(meshes):
    """Every lattice triangle survives unchanged.

    A cut that also nudged the lattice would silently invalidate hole_mask,
    torn_fraction, edge_risk and every historical measurement taken against them.
    """
    base, cut = meshes
    bv = np.asarray(base.vertices, dtype=np.float64)
    cv = np.asarray(cut.vertices, dtype=np.float64)

    def tri_set(verts, faces):
        return {tuple(sorted(map(tuple, np.round(verts[t], 6))))
                for t in np.asarray(faces, dtype=np.int64)}

    base_tris = tri_set(bv, base.faces)
    cut_tris = tri_set(cv, cut.faces)
    assert base_tris <= cut_tris
    assert len(cut_tris) > len(base_tris)


def test_no_triangle_spans_the_cliff(meshes):
    """The tear is the point. Bridging near to far would build the curtain.

    DESIGN_RULES rejects joining the two sides by name: it produces "long
    near-to-far triangles (a visible curtain) whose geometry is unsupported and
    whose texture stretches across unrelated surfaces".
    """
    _base, cut = meshes
    cam = _camera_position()
    fwd = -((np.asarray(cut.vertices, dtype=np.float64) - cam)
            @ np.linalg.inv(VIEW)[:3, :3][:, 2])
    tri = fwd[np.asarray(cut.faces, dtype=np.int64)]
    span = tri.max(axis=1) / np.maximum(tri.min(axis=1), 1e-9)
    # A triangle bridging this fixture's cliff would read FAR_M / NEAR_M = 3.0.
    assert float(span.max()) < 1.5


def test_cut_faces_are_wound_toward_the_camera_and_non_degenerate(meshes):
    _base, cut = meshes
    v = np.asarray(cut.vertices, dtype=np.float64)
    f = np.asarray(cut.faces, dtype=np.int64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    normals = np.cross(b - a, c - a)
    areas = np.linalg.norm(normals, axis=1)
    assert (areas > 1e-12).all(), "degenerate triangle emitted"
    toward = np.einsum("ij,ij->i", normals, _camera_position() - a)
    assert (toward > 0).all(), "a cut triangle faces away from the camera"


def _camera_view_coverage(mesh):
    """Fraction of the plate the mesh's faces cover at camera view.

    A relief mesh is a heightfield seen from its own camera, so faces do not
    overlap and summing projected triangle areas is exact — no rasterizer needed.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    local = (v - _camera_position()) @ np.linalg.inv(VIEW)[:3, :3]
    z = np.maximum(-local[:, 2], 1e-9)
    px = local[:, 0] / z * FX + CX
    py = -local[:, 1] / z * FY + CY
    a, b, c = f[:, 0], f[:, 1], f[:, 2]
    area = 0.5 * np.abs(
        (px[b] - px[a]) * (py[c] - py[a]) - (px[c] - px[a]) * (py[b] - py[a]))
    return float(area.sum() / (W * H))


def test_the_cut_recovers_photographed_surface_not_just_a_smoother_edge(meshes):
    """The whole-cell tear discards surface the plate actually photographed.

    Measured on the fixture: camera-view coverage 97.8% -> 100%, and the same
    ~2 point gain persists at every orbit angle. This is the half that
    `boundary_smooth_iterations` can never deliver — it moves vertices, it does
    not put lost surface back.
    """
    base, cut = meshes
    base_cov = _camera_view_coverage(base)
    cut_cov = _camera_view_coverage(cut)
    assert base_cov < 0.995, "fixture is not losing surface — the test is vacuous"
    assert cut_cov > base_cov + 0.01
    assert cut_cov <= 1.0 + 1e-6


def test_the_two_sheets_share_no_vertex_so_the_tear_still_opens(meshes):
    """Coverage at camera view rises to 100%, which reads on `tear_metrics` as a
    MISSED edge — that harness infers "torn" from a camera-view coverage gap.

    The property that actually matters is structural: nothing holds the two
    sheets together, so they slide apart under orbit and the disocclusion opens
    (measured 0 / 3.9 / 11.1 / 21.6% uncovered at 0 / 2 / 6 / 12 degrees). Pin
    the structure, because the coverage number alone cannot tell this apart from
    a curtain.
    """
    _base, cut = meshes
    cam = _camera_position()
    fwd = -((np.asarray(cut.vertices, dtype=np.float64) - cam)
            @ np.linalg.inv(VIEW)[:3, :3][:, 2])
    on_far = fwd > 0.5 * (NEAR_M + FAR_M)
    f = np.asarray(cut.faces, dtype=np.int64)
    mixed = on_far[f].any(axis=1) & ~on_far[f].all(axis=1)
    assert int(mixed.sum()) == 0, "a triangle joins the near sheet to the far one"

    near_verts = set(np.unique(f[~on_far[f].any(axis=1)]).tolist())
    far_verts = set(np.unique(f[on_far[f].all(axis=1)]).tolist())
    assert not (near_verts & far_verts), "the sheets share a vertex"


def test_cut_vertices_keep_projective_uvs(meshes):
    """A cut vertex's UV must be its own reprojected pixel, or 📽 Project smears.

    This is what makes a sub-cell cut cheap here at all: the mesh is a heightfield
    with image-registered UVs, so a vertex at a fractional pixel gets its UV for
    free instead of needing a UV solve.
    """
    _base, cut = meshes
    v = np.asarray(cut.vertices, dtype=np.float64)
    uv = np.asarray(cut.uvs, dtype=np.float64)
    assert (uv >= -1e-6).all() and (uv <= 1.0 + 1e-6).all()

    local = (v - _camera_position()) @ np.linalg.inv(VIEW)[:3, :3]
    proj_x = local[:, 0] / -local[:, 2] * FX + CX
    proj_y = -local[:, 1] / -local[:, 2] * FY + CY
    err = np.hypot(proj_x - uv[:, 0] * (W - 1),
                   proj_y - (1.0 - uv[:, 1]) * (H - 1))
    assert float(err.max()) < 1e-3


def test_off_by_default_is_byte_identical(cliff_depth, meshes):
    base, _cut = meshes
    again = build_relief_mesh(cliff_depth, sub_quad_boundary=False, **BUILD)
    assert np.array_equal(again.vertices, base.vertices)
    assert np.array_equal(again.faces, base.faces)
    assert "sub_quad_cut" not in again.stats


def test_stats_report_the_cut_and_keep_torn_fraction_comparable(meshes):
    """torn_fraction counts emitted faces against whole-quad slots, so cut faces
    deflate it. The whole-quad figure has to stay reported or the QA gate
    (`torn_excessive`) silently changes meaning."""
    base, cut = meshes
    info = cut.stats["sub_quad_cut"]
    assert info["n_cut_cells"] > 0
    assert info["n_new_faces"] > 0
    assert not info["budget_truncated"]
    assert cut.stats["torn_fraction_whole_quad"] == pytest.approx(
        base.stats["torn_fraction"], abs=1e-9)


def test_a_cell_with_no_data_is_left_torn(cliff_depth):
    """An invalid corner means there is nothing to recover — inventing a boundary
    inside it would be the "a backdrop is not a hole rim" mistake."""
    holed = cliff_depth.copy()
    holed[200:260, 200:260] = np.nan
    cut = build_relief_mesh(holed, sub_quad_boundary=True, **BUILD)
    uv = np.asarray(cut.uvs, dtype=np.float64)
    px = uv[:, 0] * (W - 1)
    py = (1.0 - uv[:, 1]) * (H - 1)
    inside = (px > 205) & (px < 255) & (py > 205) & (py < 255)
    assert not inside.any(), "geometry was invented inside a no-data hole"


def test_budget_truncation_is_reported_not_silent(cliff_depth):
    """A silent cap would read downstream as "the cliff needed no cutting"."""
    cut = build_relief_mesh(cliff_depth, sub_quad_boundary=True,
                            max_cut_cells=3, **BUILD)
    info = cut.stats["sub_quad_cut"]
    assert info["budget_truncated"] is True
    assert info["n_candidate_cells"] > info["max_cut_cells"]


def test_a_saddle_cell_is_left_torn_rather_than_guessed():
    """Two diagonal near corners means two cliffs in one cell; either resolution
    is a guess, so the cell stays torn exactly as it does today."""
    from atlas_camera.core.subquad_cut import cut_torn_quads

    d = np.array([[4.0, 12.0], [12.0, 4.0]], dtype=np.float64)
    result = cut_torn_quads(
        grid_depth=d,
        grid_valid=np.ones((2, 2), dtype=bool),
        rows=np.array([0, 8]), cols=np.array([0, 8]),
        torn=np.ones((1, 1), dtype=bool),
        depth_full=np.full((16, 16), 4.0),
        valid_full=np.ones((16, 16), dtype=bool),
        lattice_index=np.arange(4).reshape(2, 2),
        unproject=lambda u, v, z: np.array([u, v, z], dtype=np.float64),
    )
    assert result["stats"]["n_saddle_cells"] == 1
    assert result["stats"]["n_cut_cells"] == 0
    assert len(result["faces"]) == 0


# ------------------------------------------- the silhouette matte (sky/exclusion)


class TestSilhouetteMatte:
    """The OTHER half of the staircase, and a structurally different one.

    At a depth cliff the near and far sheets project to the SAME pixel, so one
    scalar matte cannot keep one and cut the other — that is `sub_quad_boundary`'s
    job. At a sky/exclusion boundary there is no second sheet, so a full-res matte
    cuts the true edge exactly. That asymmetry is why the doctrine's live number
    was a SKYLINE strip, and why these two features do not overlap.
    """

    SLOPE_SKY, INTERCEPT_SKY = 0.35, 150.0

    def _sky(self):
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
        return ys < (xs * self.SLOPE_SKY + self.INTERCEPT_SKY)

    def _build(self, **kw):
        flat = np.full((H, W), 8.0, dtype=np.float32)
        base = dict(BUILD)
        base.update(apply_sky_heuristic=False, exclude_mask=self._sky())
        return build_relief_mesh(flat, **base, **kw)

    def test_off_by_default(self):
        assert self._build().silhouette_alpha is None

    def test_the_matte_crossing_tracks_the_true_skyline_not_the_lattice(self):
        mesh = self._build(silhouette_matte=True)
        alpha = mesh.silhouette_alpha
        assert alpha is not None and alpha.shape == (H, W)

        step = max(H, W) / GRID
        errors = []
        for col in range(4, W - 4):
            column = alpha[:, col]
            crossings = np.where((column[:-1] < 0.5) & (column[1:] >= 0.5))[0]
            if crossings.size:
                true_row = col * self.SLOPE_SKY + self.INTERCEPT_SKY
                errors.append(abs(crossings[0] + 0.5 - true_row))
        errors = np.asarray(errors)
        assert errors.size > W // 2
        # Lattice quantization would be ~step/2; the matte must be far better or
        # it is just the staircase again at full resolution.
        assert float(errors.mean()) < 0.25 * step

    def test_it_is_not_derived_from_hole_mask(self):
        """hole_mask's tear contribution is a nearest upsample of the quad
        lattice, so deriving the matte from it would ship the staircase."""
        mesh = self._build(silhouette_matte=True)
        binary = mesh.silhouette_alpha >= 0.5
        assert not np.array_equal(binary, ~np.asarray(mesh.hole_mask, dtype=bool))

    def test_the_matte_is_soft_at_the_boundary_and_hard_elsewhere(self):
        """The shader feathers with smoothstep(0.5 +/- fwidth), so the field has
        to CROSS 0.5 rather than jump it, and must be flat well away from the
        edge or the feather would eat real surface."""
        alpha = self._build(silhouette_matte=True).silhouette_alpha
        assert 0.0 <= float(alpha.min()) and float(alpha.max()) <= 1.0
        partial = (alpha > 0.05) & (alpha < 0.95)
        assert partial.any(), "no transition band — nothing to feather"
        assert float(partial.mean()) < 0.05, "the ramp is too wide to be an edge"

    def test_the_skirt_and_the_matte_are_one_switch(self):
        """An unmatted skirt carries replicated sky pixels on receding geometry
        (found live on monument valley), which is why edge_overhang_cells forbids
        growing into the exclusion at all. The matte lifts that ban — so the ban
        must only lift WITH the matte."""
        without = self._build(edge_overhang_cells=2)
        with_matte = self._build(edge_overhang_cells=2, silhouette_matte=True)
        # Without the matte the skirt cannot enter the exclusion, so the mesh is
        # unchanged from having no skirt at all there.
        assert without.silhouette_alpha is None
        assert len(with_matte.vertices) > len(without.vertices), (
            "the matte did not license any skirt growth — nothing to cut back")


def test_the_qa_gate_sees_the_whole_quad_tear_ratio(cliff_depth):
    """A safety gate must not get looser because a feature got cleverer.

    `torn_fraction` counts EMITTED faces against whole-quad slots. A cut cell
    emits faces from a PARTIAL quad, so the ratio drops and `torn_excessive`
    (nodes_qa) would pass meshes it should flag — the gate quietly relaxing on
    exactly the meshes doing something new.
    """
    from atlas_camera.core.proxy_geometry import relief_mesh_primitive

    base = build_relief_mesh(cliff_depth, **BUILD)
    cut = build_relief_mesh(cliff_depth, sub_quad_boundary=True, **BUILD)

    base_meta = relief_mesh_primitive(base).metadata
    cut_meta = relief_mesh_primitive(cut).metadata

    assert "torn_fraction_whole_quad" not in base_meta, "no cut ran — nothing to correct"
    # The emitted ratio really is deflated; that is the trap.
    assert cut_meta["torn_fraction"] < base_meta["torn_fraction"]
    # ...and the honest figure travels with it, matching the uncut build.
    assert cut_meta["torn_fraction_whole_quad"] == pytest.approx(
        base_meta["torn_fraction"], abs=1e-9)


def test_planar_hole_patch_metadata_stays_json_serializable(cliff_depth):
    """The solve JSON is a contract; a manifest failure must never fail an export.

    patch_planar_holes hands its caller a LIVE HoleField under report["hole_field"]
    and deliberately snapshots the plain data into stats BEFORE adding it. The
    node stored the live report in primitive metadata, defeating that and killing
    AtlasExportReviewPackage with "Object of type HoleField is not JSON
    serializable" — found on a real sh004 run, after every other node succeeded.
    """
    import json

    from atlas_camera.core.schema import _json_ready

    report = {"filled": 3, "rejected": {"frame": 2}, "hole_field": object()}
    stored = {k: v for k, v in report.items() if k != "hole_field"}
    assert "hole_field" not in stored
    json.dumps(_json_ready(stored))

    src = pathlib.Path("atlas_camera/comfy/nodes_geometry.py").read_text(encoding="utf-8")
    assert 'metadata["planar_hole_patch"] = report' not in src, (
        "the live report (with its HoleField) is back in primitive metadata")
