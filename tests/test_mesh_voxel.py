"""core.mesh_voxel — surface-nets watertightness, orientation, pocket fill,
and the voxel_remesh end-to-end contract through apply_retopo (export-only).
"""

import numpy as np
import pytest

from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.mesh_repair import boundary_edges
from atlas_camera.core.mesh_voxel import (
    fill_interior_invalid,
    render_depth_grid,
    surface_nets,
    taubin_smooth,
    voxel_remesh,
)

FX = FY = 200.0
W, H = 160, 120
CX, CY = W / 2, H / 2


def _view():
    view, _w, _r = look_at_view_matrix((0.0, 0.0, 0.0), (0.0, 0.0, -10.0))
    return np.asarray(view, dtype=np.float64)


def _signed_volume(verts, faces):
    v = verts[faces]
    return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)


def _euler(verts, faces):
    e = np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]],
                                     faces[:, [2, 0]]]), axis=1), axis=0)
    return len(verts) - len(e) + len(faces)


def _quad_mesh(z=-6.0, half=3.0):
    """Fronto-parallel quad centred on the view axis at depth |z|."""
    verts = np.array([[-half, -half, z], [half, -half, z],
                      [half, half, z], [-half, half, z]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return verts, faces


# ---------------------------------------------------------------------------
# surface nets
# ---------------------------------------------------------------------------

def test_surface_nets_cube_is_watertight_outward_sphere_topology():
    occ = np.zeros((6, 6, 6), dtype=bool)
    occ[1:5, 1:5, 1:5] = True
    verts, faces = surface_nets(occ)
    assert len(faces) > 0
    assert len(boundary_edges(faces)) == 0          # watertight
    assert _euler(verts, faces) == 2                # topological sphere
    assert _signed_volume(verts, faces) > 0         # outward winding
    # A 4-voxel cube's dual surface encloses roughly the voxel volume.
    assert _signed_volume(verts, faces) == pytest.approx(4 ** 3, rel=0.35)


def test_surface_nets_empty_grid_yields_nothing():
    verts, faces = surface_nets(np.zeros((4, 4, 4), dtype=bool))
    assert len(verts) == 0 and len(faces) == 0


def test_surface_nets_torus_topology():
    # Solid ring (torus): euler characteristic 0, still watertight.
    occ = np.zeros((5, 12, 12), dtype=bool)
    yy, xx = np.mgrid[0:12, 0:12]
    r = np.hypot(yy - 5.5, xx - 5.5)
    ring = (r >= 2.0) & (r <= 4.5)
    occ[1:4] = ring[None]
    verts, faces = surface_nets(occ)
    assert len(boundary_edges(faces)) == 0
    assert _euler(verts, faces) == 0


def test_taubin_smooth_keeps_counts_and_shrinks_blockiness():
    occ = np.zeros((6, 6, 6), dtype=bool)
    occ[1:5, 1:5, 1:5] = True
    verts, faces = surface_nets(occ)
    sm = taubin_smooth(verts, faces, iterations=8)
    assert sm.shape == verts.shape
    # Taubin must not collapse the shape: volume stays in the same ballpark.
    assert _signed_volume(sm, faces) == pytest.approx(
        _signed_volume(verts, faces), rel=0.35)


# ---------------------------------------------------------------------------
# depth raster + pocket fill
# ---------------------------------------------------------------------------

def test_render_depth_grid_flat_quad():
    # half=1.0 keeps the quad well inside the frame (u spread ±33 px) so the
    # corners are genuinely uncovered; half=3 would cover the whole raster.
    verts, faces = _quad_mesh(z=-6.0, half=1.0)
    depth = render_depth_grid(verts, faces, _view(), FX, FY, CX, CY, W, H)
    centre = depth[H // 2 - 5:H // 2 + 5, W // 2 - 5:W // 2 + 5]
    assert np.isfinite(centre).all()
    assert centre == pytest.approx(6.0, abs=1e-6)
    assert not np.isfinite(depth[0, 0])             # off-quad = uncovered


def test_fill_interior_invalid_fills_pockets_keeps_border_open():
    d = np.full((32, 32), 5.0)
    d[10:14, 10:14] = np.nan     # enclosed pocket -> filled
    d[0:6, 20:24] = np.nan       # touches the border -> stays open
    out = fill_interior_invalid(d)
    assert np.isfinite(out[10:14, 10:14]).all()
    assert out[10:14, 10:14] == pytest.approx(5.0, abs=1e-3)
    assert not np.isfinite(out[0, 21])


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------

def test_voxel_remesh_quad_becomes_watertight_world_solid():
    verts, faces = _quad_mesh(z=-6.0, half=3.0)
    out_v, out_f = voxel_remesh(
        verts, faces, view_matrix=_view(),
        fx=FX, fy=FY, cx=CX, cy=CY, image_width=W, image_height=H,
        grid=48, smooth_iterations=4)
    assert len(out_f) > 0
    assert np.isfinite(out_v).all()
    assert len(boundary_edges(out_f)) == 0          # watertight in world space
    # The shell hugs the source plane: all depths within ~1 slab of 6 m.
    z = -out_v[:, 2]
    assert 5.0 < float(z.min()) and float(z.max()) < 8.0
    # And its footprint stays near the quad's world extent.
    assert float(np.abs(out_v[:, 0]).max()) < 4.5


def test_voxel_remesh_raises_on_empty_coverage():
    verts, faces = _quad_mesh(z=+6.0)  # behind the camera
    with pytest.raises(ValueError):
        voxel_remesh(verts, faces, view_matrix=_view(),
                     fx=FX, fy=FY, cx=CX, cy=CY, image_width=W, image_height=H)


def test_apply_retopo_voxel_remesh_regenerates_uvs():
    from atlas_camera.core.mesh_retopo import _RETOPO_METHODS, apply_retopo

    assert _RETOPO_METHODS == ("off", "quad", "decimate", "smooth", "voxel_remesh")

    class M:
        pass

    m = M()
    m.vertices, m.faces = _quad_mesh(z=-6.0)
    m.uvs = np.zeros((4, 2))
    rep = apply_retopo(
        m, method="voxel_remesh", target_vertex_count=2000,
        view_matrix=_view(), fx=FX, fy=FY, cx=CX, cy=CY,
        image_width=W, image_height=H)
    assert rep["changed"] is True
    assert "watertight" in rep["note"]
    assert len(m.vertices) == len(m.uvs)            # 1:1 vertex-UV regenerated
    assert len(boundary_edges(np.asarray(m.faces))) == 0
    ok = np.isfinite(m.uvs)
    assert ok.all()


def test_apply_retopo_voxel_remesh_requires_intrinsics():
    from atlas_camera.core.mesh_retopo import apply_retopo

    class M:
        pass

    m = M()
    m.vertices, m.faces = _quad_mesh()
    with pytest.raises(ValueError):
        apply_retopo(m, method="voxel_remesh")
