"""AtlasExportReliefMesh's interior-hole-fill report + viewport preview.

The fill is export-only, so the artist can neither see it in the viewport nor
learn anything about it from the node — which makes it impossible to tune
without a DCC round-trip. These pin the two things that close that loop:

*   a **report** rendered on the node (it is already an OUTPUT_NODE), and
*   a **preview_solve** output carrying the mesh that was ACTUALLY written, so
    it can be wired into a viewport.

The preview deliberately hangs off the export node rather than a separate
preview node: one set of widgets means the previewed mesh and the exported mesh
can never drift apart (the same reasoning behind `_resolve_depth_band` being
shared between AtlasDepthLayerMask and AtlasCleanPlateLayer).

Runs with only numpy/torch — no [neural] extra or model download needed.
"""

import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas_camera.comfy.nodes import AtlasExportReliefMesh
from atlas_camera.core.proxy_geometry import relief_mesh_primitive
from atlas_camera.core.relief_mesh import build_relief_mesh
from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasProjectionScene,
    LatentCamera,
    LatentScene,
)

_N = 256
_F = 250.0


def _torn_solve():
    """A solve carrying a real relief mesh with genuine interior tear holes."""
    rng = np.random.default_rng(6)
    depth = np.full((_N, _N), 20.0)
    yy, xx = np.mgrid[0:_N, 0:_N]
    for _ in range(3):
        cx0, cy0 = rng.integers(50, 200, 2)
        r = rng.integers(20, 55)
        depth[(np.abs(xx - cx0) < r) & (np.abs(yy - cy0) < r)] = 6 + 12 * rng.random()
    depth += rng.normal(0, 0.45, (_N, _N))
    depth[(xx // 6 % 3 == 0)] += 0.9
    view = np.eye(4, dtype=np.float64)
    mesh = build_relief_mesh(depth, view_matrix=view, fx=_F, fy=_F,
                             cx=_N / 2, cy=_N / 2, grid_long_edge=128,
                             depth_edge_rel=0.5, apply_sky_heuristic=False)
    solve = LatentScene(
        camera=LatentCamera(
            intrinsics=AtlasIntrinsics(fx_px=_F, fy_px=_F, cx_px=_N / 2,
                                       cy_px=_N / 2, image_width=_N,
                                       image_height=_N),
            extrinsics=AtlasExtrinsics(camera_view_matrix=view.tolist()),
        ),
        projection_scene=AtlasProjectionScene(
            proxy_geometry=[relief_mesh_primitive(mesh)]),
    )
    return solve, torch.zeros((1, _N, _N, 3), dtype=torch.float32)


def _relief_prim(solve):
    for p in solve.projection_scene.proxy_geometry:
        if p.primitive_type == "mesh" and (p.metadata or {}).get("source") == "depth_relief_mesh":
            return p
    return None


def _n_faces(solve):
    p = _relief_prim(solve)
    return 0 if p is None else len(p.metadata["faces"]) // 3


def _export(solve, image, **kw):
    return AtlasExportReliefMesh().export(
        solve, image, tempfile.mkdtemp(prefix="atlas_fill_test_"),
        use_solve_mesh=True, format="obj", **kw)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def test_export_reports_what_the_fill_did():
    """Ticking the box with no feedback is untunable — the count must surface."""
    solve, image = _torn_solve()
    out = _export(solve, image, fill_interior_holes=True, max_hole_edges=64)
    report = out["result"][3] if isinstance(out, dict) else out[3]
    assert "fill" in report.lower()
    # names how many holes it closed, and the scope it used
    assert "filled" in report.lower()
    assert "64" in report, "report must state the max_hole_edges scope actually used"


def test_report_renders_on_the_node():
    """AtlasExportReliefMesh is an OUTPUT_NODE, so the report belongs on it."""
    solve, image = _torn_solve()
    out = _export(solve, image, fill_interior_holes=True, max_hole_edges=64)
    assert isinstance(out, dict), "must return a ui dict so the report renders"
    assert "ui" in out and "text" in out["ui"]
    assert "result" in out


def test_report_says_off_when_disabled():
    solve, image = _torn_solve()
    out = _export(solve, image, fill_interior_holes=False)
    report = out["result"][3] if isinstance(out, dict) else out[3]
    assert "off" in report.lower()


def test_report_states_the_band_box_scope():
    solve, image = _torn_solve()
    out = _export(solve, image, fill_interior_holes=True, max_hole_edges=64,
                  fill_depth_near_m=2.0, fill_depth_far_m=18.0)
    report = out["result"][3] if isinstance(out, dict) else out[3]
    assert "2" in report and "18" in report, "band box bounds must be reported"


# ---------------------------------------------------------------------------
# preview_solve
# ---------------------------------------------------------------------------

def test_preview_solve_carries_the_filled_mesh():
    """The preview must be the mesh that was ACTUALLY written, so tuning in the
    viewport tells you what lands in Maya/Nuke."""
    solve, image = _torn_solve()
    before = _n_faces(solve)
    out = _export(solve, image, fill_interior_holes=True, max_hole_edges=64)
    preview = out["result"][2] if isinstance(out, dict) else out[2]
    assert preview is not None
    assert _n_faces(preview) > before, "preview_solve has no filled faces"


def test_preview_never_mutates_the_input_solve():
    """Export-only stays export-only: the live projection mesh keeps its tears."""
    solve, image = _torn_solve()
    before = _n_faces(solve)
    _export(solve, image, fill_interior_holes=True, max_hole_edges=64)
    assert _n_faces(solve) == before, "the fill leaked into the input solve"


def test_preview_solve_passes_through_when_fill_off():
    """So the wire can stay put while A/B-ing the widget."""
    solve, image = _torn_solve()
    before = _n_faces(solve)
    out = _export(solve, image, fill_interior_holes=False)
    preview = out["result"][2] if isinstance(out, dict) else out[2]
    assert _n_faces(preview) == before


def test_preview_solve_keeps_the_camera():
    """It must be viewport-renderable: same recovered camera, same intrinsics."""
    solve, image = _torn_solve()
    out = _export(solve, image, fill_interior_holes=True, max_hole_edges=64)
    preview = out["result"][2] if isinstance(out, dict) else out[2]
    assert preview.camera.intrinsics.fx_px == solve.camera.intrinsics.fx_px
    assert (preview.camera.extrinsics.camera_view_matrix
            == solve.camera.extrinsics.camera_view_matrix)


def test_preview_mesh_still_has_1to1_vertex_uv():
    """The viewport's projection material depends on it, same as the writers."""
    solve, image = _torn_solve()
    out = _export(solve, image, fill_interior_holes=True, max_hole_edges=64)
    preview = out["result"][2] if isinstance(out, dict) else out[2]
    p = _relief_prim(preview)
    n_v = len(p.metadata["vertices"]) // 3
    n_uv = len(p.metadata["uvs"]) // 2
    assert n_v == n_uv
    assert max(p.metadata["faces"]) < n_v, "face index out of range"


def test_outputs_are_appended_last():
    """Appended outputs keep saved workflows' existing links intact."""
    assert AtlasExportReliefMesh.RETURN_NAMES[:2] == ("obj_path", "glb_path")
    assert AtlasExportReliefMesh.RETURN_NAMES[2:] == ("preview_solve", "report")
    assert AtlasExportReliefMesh.RETURN_TYPES[2:] == ("ATLAS_SOLVE", "STRING")
