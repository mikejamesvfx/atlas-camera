"""Measured-primitives Blender bridge — Atlas side, NO Blender needed.

Covers the seed writer, the multi-mesh reader, the import gate + projective-UV
regeneration, and both nodes with `run_recipe` faked (the pattern
test_blender_organic_fill uses). A live-Blender smoke test at the bottom is
skipped when no Blender is installed.
"""
from __future__ import annotations

import json
import shutil

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.blender import (  # noqa: E402
    blender_to_atlas, build_blender_command, read_meshes, write_scene_seed,
)
from atlas_camera.blender.exchange import OUT_MESHES_JSON, OUT_MESHES_NPZ, SEED_JSON, SEED_NPZ  # noqa: E402
from atlas_camera.blender.measured import (  # noqa: E402
    MASSING_SOURCE, meshes_to_primitives, seed_from_solve, solve_seed_fingerprint,
)
from atlas_camera.core.camera_math import look_at_view_matrix  # noqa: E402
from atlas_camera.core.proxy_geometry import PROXY_ROLE, serialize_proxy_geometry  # noqa: E402
from atlas_camera.core.schema import (  # noqa: E402
    AtlasExtrinsics, AtlasIntrinsics, AtlasProjectionScene, AtlasProxyPrimitive,
    AtlasSolve, LatentCamera,
)

W, H, FX = 800, 600, 700.0


def _solve():
    eye, target = (0.0, 1.6, 0.0), (0.0, 0.5, -10.0)
    view, world, rot3 = look_at_view_matrix(eye, target)
    extr = AtlasExtrinsics(camera_position=eye, camera_rotation_matrix=rot3,
                           camera_world_matrix=world, camera_view_matrix=view)
    intr = AtlasIntrinsics(image_width=W, image_height=H, fx_px=FX, fy_px=FX,
                           cx_px=W / 2, cy_px=H / 2)
    s = AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))
    s.projection_scene = AtlasProjectionScene()
    s.debug_metadata["scale_source"] = "measured_baseline"
    s.debug_metadata["baseline_m"] = 14.6
    # A drawn footprint on the ground (source viewport_polygon), 4x3 m, 8 m out.
    fv = np.array([[-2, 0, -8], [2, 0, -8], [2, 0, -11], [-2, 0, -11]], float)
    ff = np.array([[0, 1, 2], [0, 2, 3]])
    s.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
        name="drawn_plane_01", primitive_type="mesh", material="atlas_projection_proxy",
        metadata={"role": PROXY_ROLE, "source": "viewport_polygon", "label": "footprint",
                  "vertices": fv.reshape(-1).tolist(), "faces": ff.reshape(-1).tolist(),
                  "uvs": [0.0] * 8, "edge_risk": [], "ribbon_t": []}))
    return s


def _box(cx=0.0, cz=-9.0, h=3.0, hw=1.0):
    v = np.array([[cx - hw, 0, cz - hw], [cx + hw, 0, cz - hw], [cx + hw, 0, cz + hw],
                  [cx - hw, 0, cz + hw],
                  [cx - hw, h, cz - hw], [cx + hw, h, cz - hw], [cx + hw, h, cz + hw],
                  [cx - hw, h, cz + hw]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, f


def _write_out(exdir, meshes):
    """Emulate what the recipe writes: Blender-space arrays + tags."""
    from atlas_camera.blender import atlas_to_blender
    arrays, meta = {}, []
    for i, (name, v, f, tags) in enumerate(meshes):
        arrays[f"mesh_{i}_vertices"] = atlas_to_blender(v)
        arrays[f"mesh_{i}_faces"] = np.asarray(f, np.int32)
        meta.append({"index": i, "name": name, **tags})
    np.savez(exdir / OUT_MESHES_NPZ, **arrays)
    (exdir / OUT_MESHES_JSON).write_text(json.dumps(
        {"recipe": "fake", "meshes": meta, "blender_version": "5.2.0"}), encoding="utf-8")


# ---------------------------------------------------------------------------

class TestSeed:
    def test_seed_from_solve_carries_camera_reference_and_measured(self, tmp_path):
        seed = seed_from_solve(_solve())
        assert seed["camera"]["fx"] == FX and seed["camera"]["image_width"] == W
        assert [p["source"] for p in seed["primitives"]] == ["viewport_polygon"]
        assert seed["measured"]["baseline_m"] == 14.6
        assert seed["measured"]["camera_height_m"] == pytest.approx(1.6)
        write_scene_seed(tmp_path, camera=seed["camera"], primitives=seed["primitives"],
                         drawn_shapes=[{"id": "a", "kind": "polygon", "points_world": [[0, 0, -5]]}],
                         params={"solve_fingerprint": seed["fingerprint"]})
        js = json.loads((tmp_path / SEED_JSON).read_text(encoding="utf-8"))
        assert js["params"]["solve_fingerprint"] == seed["fingerprint"]
        assert js["primitives"][0]["n_faces"] == 2
        # Drawn shape converted to Blender axes: Atlas (0,0,-5) -> Blender (0,5,0).
        assert js["drawn_shapes"][0]["points_blender"] == [[0.0, 5.0, 0.0]]
        with np.load(tmp_path / SEED_NPZ) as data:
            v = data["prim_0_vertices"]
            # Atlas footprint at Y=0 lands on Blender Z=0.
            assert np.allclose(v[:, 2], 0.0)
            assert np.allclose(blender_to_atlas(v)[:, 1], 0.0)

    def test_camera_world_matrix_is_rotated_into_blender_axes(self, tmp_path):
        seed = seed_from_solve(_solve())
        write_scene_seed(tmp_path, camera=seed["camera"], primitives=[], params={})
        js = json.loads((tmp_path / SEED_JSON).read_text(encoding="utf-8"))
        m = np.asarray(js["camera"]["matrix_world_blender"])
        # Atlas camera at (0,1.6,0) -> Blender (0,0,1.6); it looks toward Atlas -Z
        # = Blender +Y, so the camera's local -Z axis (third column, negated)
        # points along +Y in Blender.
        assert np.allclose(m[:3, 3], [0.0, 0.0, 1.6], atol=1e-9)
        fwd = -m[:3, 2]
        assert fwd[1] > 0.99 * np.linalg.norm(fwd)
        assert abs(np.linalg.det(m[:3, :3]) - 1.0) < 1e-9

    def test_fingerprint_tracks_camera_and_roster(self):
        a, b = _solve(), _solve()
        assert solve_seed_fingerprint(a) == solve_seed_fingerprint(b)
        b.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
            name="x", primitive_type="plane", metadata={"role": PROXY_ROLE}))
        assert solve_seed_fingerprint(a) != solve_seed_fingerprint(b)


class TestReadMeshes:
    def test_reads_back_into_atlas_space_with_tags(self, tmp_path):
        v, f = _box()
        _write_out(tmp_path, [("ground_plane", v, f, {"kind": "ground_plane"})])
        got = read_meshes(tmp_path)
        assert got["rejected"] == []
        m = got["meshes"][0]
        assert m["name"] == "ground_plane" and m["kind"] == "ground_plane"
        assert np.allclose(m["vertices"], v)

    def test_bad_meshes_are_rejected_individually(self, tmp_path):
        v, f = _box()
        bad_v = v.copy(); bad_v[0, 0] = np.nan
        _write_out(tmp_path, [("ok", v, f, {}), ("nan", bad_v, f, {}),
                              ("range", v, f + 100, {})])
        got = read_meshes(tmp_path)
        assert [m["name"] for m in got["meshes"]] == ["ok"]
        assert sorted(r["name"] for r in got["rejected"]) == ["nan", "range"]

    def test_missing_archive_raises_quotably(self, tmp_path):
        with pytest.raises(RuntimeError, match="out_meshes.npz"):
            read_meshes(tmp_path)


class TestMeshesToPrimitives:
    def test_projective_uvs_regenerated_and_scalars_serialize(self):
        s = _solve()
        v, f = _box()
        prims, rejected = meshes_to_primitives(
            s, [{"name": "box", "vertices": v, "faces": f, "kind": "massing_box"}],
            source=MASSING_SOURCE, name_prefix="massing")
        assert rejected == []
        p = prims[0]
        assert p.metadata["role"] == PROXY_ROLE and p.metadata["source"] == MASSING_SOURCE
        assert len(p.metadata["uvs"]) == 2 * len(v)
        uv = np.asarray(p.metadata["uvs"]).reshape(-1, 2)
        assert (uv >= -0.01).all() and (uv <= 1.01).all()
        assert p.metadata["blender_kind"] == "massing_box"
        # Nested-free metadata survives the scalar-only viewport serializer.
        s.projection_scene.proxy_geometry.append(p)
        ser = serialize_proxy_geometry(s.projection_scene)
        assert any(e.get("name") == p.name for e in ser)

    def test_below_ground_and_far_meshes_are_rejected_not_raised(self):
        s = _solve()
        v, f = _box()
        low = v.copy(); low[:, 1] -= 1.0
        far = v.copy(); far[:, 2] -= 500.0
        prims, rejected = meshes_to_primitives(
            s, [{"name": "low", "vertices": low, "faces": f},
                {"name": "far", "vertices": far, "faces": f},
                {"name": "ok", "vertices": v, "faces": f}],
            min_y_m=-0.05, max_radius_m=100.0)
        assert [p.name for p in prims] == ["blender_ok"]
        assert sorted(r["name"] for r in rejected) == ["far", "low"]


class TestNodes:
    def _fake_run(self, tmp_path, monkeypatch, *, meshes):
        import atlas_camera.blender as B
        calls = []

        def fake(recipe_name, exdir, *, blender_path="", timeout_s=300, blend_file=""):
            calls.append((recipe_name, str(exdir), str(blend_file)))
            _write_out(exdir, meshes)
            return {"blender_version": "5.2.0", "meshes_out": len(meshes),
                    "footprints": 1, "facades": 0, "massing_boxes": 0,
                    "ground_plane": True, "skipped_polygons": 0,
                    "selection_rule": "atlas_out"}
        monkeypatch.setattr(B, "run_recipe", fake)
        return calls

    def test_massing_node_appends_and_keeps_primary_untouched(self, tmp_path, monkeypatch):
        from atlas_camera.comfy.nodes_geometry import AtlasBlenderMassing
        v, f = _box()
        gv = np.array([[-30, 0, 30], [30, 0, 30], [30, 0, -30], [-30, 0, -30]], float)
        gf = np.array([[0, 1, 2], [0, 2, 3]])
        calls = self._fake_run(tmp_path, monkeypatch, meshes=[
            ("ground_plane", gv, gf, {"kind": "ground_plane", "source": "blender_massing"}),
            ("mass_drawn_plane_01", v, f, {"kind": "footprint_extrusion", "height_m": 3.0,
                                           "source": "blender_massing"}),
        ])
        s = _solve()
        before = json.dumps(serialize_proxy_geometry(s.projection_scene))
        out, report, exdir = AtlasBlenderMassing().massing(
            s, exchange_dir=str(tmp_path / "ex"), default_height_m=3.0)
        assert calls and calls[0][0] == "massing.py"
        assert (tmp_path / "ex" / SEED_JSON).is_file()
        names = [p.name for p in out.projection_scene.proxy_geometry]
        assert names[0] == "drawn_plane_01"                       # original kept
        assert "massing_ground_plane" in names and "massing_mass_drawn_plane_01" in names
        assert all(p.metadata["source"] == MASSING_SOURCE
                   for p in out.projection_scene.proxy_geometry[1:])
        # Input solve untouched (deepcopy), primary camera identical.
        assert json.dumps(serialize_proxy_geometry(s.projection_scene)) == before
        assert out.camera.extrinsics.camera_view_matrix == s.camera.extrinsics.camera_view_matrix
        assert "2 measured primitive(s) appended" in report
        assert exdir == str(tmp_path / "ex")

    def test_massing_node_reports_blender_failure_and_passes_through(self, tmp_path, monkeypatch):
        import atlas_camera.blender as B
        from atlas_camera.comfy.nodes_geometry import AtlasBlenderMassing

        def boom(*a, **k):
            raise RuntimeError("Blender not found: set ATLAS_BLENDER_PATH")
        monkeypatch.setattr(B, "run_recipe", boom)
        s = _solve()
        out, report, _ = AtlasBlenderMassing().massing(s, exchange_dir=str(tmp_path))
        assert "FAILED" in report and "ATLAS_BLENDER_PATH" in report
        assert len(out.projection_scene.proxy_geometry) == 1

    def test_massing_seed_only_mode_writes_no_geometry(self, tmp_path, monkeypatch):
        from atlas_camera.comfy.nodes_geometry import AtlasBlenderMassing
        calls = self._fake_run(tmp_path, monkeypatch, meshes=[])
        out, report, _ = AtlasBlenderMassing().massing(
            _solve(), exchange_dir=str(tmp_path), run_recipe=False)
        assert calls == [] and "seed written only" in report
        assert len(out.projection_scene.proxy_geometry) == 1

    def test_import_node_refuses_stale_seed_unless_told_otherwise(self, tmp_path, monkeypatch):
        from atlas_camera.comfy.nodes_geometry import AtlasBlenderImportMeshes
        v, f = _box()
        _write_out(tmp_path, [("box", v, f, {})])
        (tmp_path / SEED_JSON).write_text(json.dumps(
            {"params": {"solve_fingerprint": "deadbeefdeadbeef"}}), encoding="utf-8")
        s = _solve()
        out, report = AtlasBlenderImportMeshes().import_meshes(s, exchange_dir=str(tmp_path))
        assert report.startswith("REFUSED") and len(out.projection_scene.proxy_geometry) == 1
        out, report = AtlasBlenderImportMeshes().import_meshes(
            s, exchange_dir=str(tmp_path), expect_fingerprint=False)
        assert "warning" in report and len(out.projection_scene.proxy_geometry) == 2
        assert out.projection_scene.proxy_geometry[1].metadata["source"] == "blender_import"

    def test_import_node_runs_export_recipe_when_blend_file_given(self, tmp_path, monkeypatch):
        from atlas_camera.comfy.nodes_geometry import AtlasBlenderImportMeshes
        v, f = _box()
        calls = self._fake_run(tmp_path, monkeypatch, meshes=[("edited", v, f, {})])
        blend = tmp_path / "scene.blend"; blend.write_bytes(b"BLENDER")
        out, report = AtlasBlenderImportMeshes().import_meshes(
            _solve(), exchange_dir=str(tmp_path), blend_file=str(blend))
        assert calls[0][0] == "export_meshes.py" and calls[0][2] == str(blend)
        assert "1 mesh(es) appended" in report
        assert out.projection_scene.proxy_geometry[-1].name == "blender_edited"

    def test_import_node_is_changed_tracks_the_archive(self, tmp_path):
        from atlas_camera.comfy.nodes_geometry import AtlasBlenderImportMeshes
        v, f = _box()
        _write_out(tmp_path, [("a", v, f, {})])
        a = AtlasBlenderImportMeshes.IS_CHANGED(None, exchange_dir=str(tmp_path))
        _write_out(tmp_path, [("a", v, f, {}), ("b", v, f, {})])
        b = AtlasBlenderImportMeshes.IS_CHANGED(None, exchange_dir=str(tmp_path))
        assert a != b


class TestRunnerBlendFile:
    def test_blend_file_precedes_background(self, tmp_path):
        from pathlib import Path
        cmd = build_blender_command(Path("blender"), Path("r.py"), tmp_path,
                                    blend_file=Path("s.blend"))
        assert cmd[:3] == ["blender", "s.blend", "--background"]
        cmd = build_blender_command(Path("blender"), Path("r.py"), tmp_path)
        assert cmd[:2] == ["blender", "--background"]


# ---------------------------------------------------------------------------
# Live smoke (needs Blender >= 4.2 on this machine)

def _blender_available():
    from atlas_camera.blender import resolve_blender_exe
    try:
        return resolve_blender_exe("") is not None
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _blender_available(), reason="no Blender install found")
def test_live_massing_extrudes_footprint_and_roundtrips(tmp_path):
    from atlas_camera.comfy.nodes_geometry import AtlasBlenderImportMeshes, AtlasBlenderMassing
    s = _solve()
    out, report, exdir = AtlasBlenderMassing().massing(
        s, exchange_dir=str(tmp_path), default_height_m=4.0, ground_extent_m=40.0,
        timeout_s=240)
    assert "FAILED" not in report, report
    prims = {p.name: p for p in out.projection_scene.proxy_geometry}
    assert "massing_ground_plane" in prims
    mass = [p for n, p in prims.items() if n.startswith("massing_mass_")]
    assert mass, report
    v = np.asarray(mass[0].metadata["vertices"]).reshape(-1, 3)
    assert v[:, 1].max() == pytest.approx(4.0, abs=1e-3)
    assert v[:, 1].min() == pytest.approx(0.0, abs=1e-3)
    assert (tmp_path / "scene.blend").is_file()
    # GUI round-trip: export_meshes.py on the saved .blend re-imports the same set.
    out2, rep2 = AtlasBlenderImportMeshes().import_meshes(
        s, exchange_dir=str(tmp_path / "rt"), blend_file=str(tmp_path / "scene.blend"),
        expect_fingerprint=False, timeout_s=240)
    assert "FAILED" not in rep2, rep2
    assert len(out2.projection_scene.proxy_geometry) == 1 + len(mass) + 1
    shutil.rmtree(tmp_path / "rt", ignore_errors=True)


def test_fingerprint_ignores_blender_returned_meshes():
    """Import fed by the massing OUTPUT must not read as stale (found live)."""
    s = _solve()
    fp0 = solve_seed_fingerprint(s)
    v, f = _box()
    prims, _ = meshes_to_primitives(s, [{"name": "g", "vertices": v, "faces": f}],
                                    source=MASSING_SOURCE, name_prefix="massing")
    s.projection_scene.proxy_geometry.extend(prims)
    assert solve_seed_fingerprint(s) == fp0


def test_resolver_strips_quotes_and_accepts_a_directory(tmp_path):
    from atlas_camera.blender import resolve_blender_exe
    import sys
    exe = tmp_path / ("blender.exe" if sys.platform.startswith("win") else "blender")
    exe.write_bytes(b"x")
    assert resolve_blender_exe(f'"{exe}"') == exe
    assert resolve_blender_exe(str(tmp_path)) == exe


class TestMeasuredSeed:
    """MoGe pointmap → sky-free cloud + measured ground/height/extents/planes."""

    def _depth_result(self, s):
        """Synthetic OpenCV pointmap: ground at camera height 1.6 m + a wall at
        8 m + sky (NaN) above the horizon."""
        from types import SimpleNamespace
        intr = s.camera.intrinsics
        w, h = 160, 120
        fx = intr.fx_px * w / intr.image_width; fy = intr.fy_px * h / intr.image_height
        cx, cy = w / 2, h / 2
        uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        # Rays in the ATLAS camera frame, rotated to WORLD by the solve pose,
        # intersected with the true ground (y=0) and a wall (z=-8), then
        # written back as an OpenCV pointmap (y down, z forward).
        view = np.asarray(s.camera.extrinsics.camera_view_matrix, float)
        c2w = np.linalg.inv(view); R = c2w[:3, :3]; pos = c2w[:3, 3]
        d_cam = np.stack([(uu - cx) / fx, -(vv - cy) / fy, -np.ones_like(uu)], -1)
        d = d_cam @ R.T
        with np.errstate(all="ignore"):
            t_ground = np.where(d[..., 1] < -0.02, -pos[1] / d[..., 1], np.inf)
            t_wall = np.where(d[..., 2] < 0, (-8.0 - pos[2]) / d[..., 2], np.inf)
        t = np.minimum(t_ground, t_wall)
        cam_pts = d_cam * t[..., None]                    # Atlas cam frame
        pts = np.stack([cam_pts[..., 0], -cam_pts[..., 1], -cam_pts[..., 2]], -1)
        z = pts[..., 2]
        sky = ~np.isfinite(t) | (d[..., 1] > 0.15)         # up-going rays = sky
        pts[sky] = np.nan
        z = np.where(sky, np.nan, z)
        return SimpleNamespace(points=pts.astype(np.float32), depth=z.astype(np.float32),
                               model_id="Ruicheng/moge-2-vitl-normal", is_metric=True,
                               image_width=w, image_height=h,
                               metadata={"predicted_focal_px": fx})

    def test_measured_seed_excludes_sky_and_reports_ground(self, tmp_path):
        from atlas_camera.blender.measured import measure_scene_from_pointmap
        s = _solve()
        dr = self._depth_result(s)
        m = measure_scene_from_pointmap(s, dr, max_points=5000, seed=1)
        assert m is not None
        meas = m["measured"]
        assert meas["scale_source"] == "moge_pointmap"
        assert meas["n_cloud_points"] == 5000
        assert meas["excluded_fraction"] > 0.05
        assert not np.isnan(m["cloud"]).any()
        # camera height measured from the pointmap ~1.6 (MoGe scale), so the
        # ground lands ~Y=0 here because the synthetic solve also sits at 1.6.
        assert meas["camera_height_m"] == pytest.approx(1.6, abs=0.15)
        assert m["ground_y"] == pytest.approx(0.0, abs=0.15)
        assert meas["extent_m"][2] > 3.0

    def test_seed_from_solve_measured_mode_leaves_relief_home(self, tmp_path):
        s = _solve()
        # add a heavy 'relief' primitive that must NOT ride the measured seed
        v, f = _box()
        s.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
            name="projection_relief_mesh", primitive_type="mesh",
            metadata={"role": PROXY_ROLE, "source": "depth_relief_mesh",
                      "vertices": v.reshape(-1).tolist(), "faces": f.reshape(-1).tolist(),
                      "uvs": [0.0] * 16, "edge_risk": [], "ribbon_t": []}))
        seed = seed_from_solve(s, depth_result=self._depth_result(s), max_points=2000)
        srcs = [p["source"] for p in seed["primitives"]]
        assert "depth_relief_mesh" not in srcs
        assert "viewport_polygon" in srcs                     # artist intent rides
        assert seed["cloud"] is not None and len(seed["cloud"]) == 2000
        assert seed["measured"]["seed_mode"] == "measured_pointmap"
        write_scene_seed(tmp_path, camera=seed["camera"], primitives=seed["primitives"],
                         params={"ground_y_m": seed["ground_y"]}, cloud=seed["cloud"])
        with np.load(tmp_path / SEED_NPZ) as data:
            assert data["cloud_points"].shape == (2000, 3)
        js = json.loads((tmp_path / SEED_JSON).read_text(encoding="utf-8"))
        assert js["n_cloud_points"] == 2000
        # relief mode when no pointmap:
        seed2 = seed_from_solve(s)
        assert "depth_relief_mesh" in [p["source"] for p in seed2["primitives"]]
        assert seed2["measured"]["seed_mode"] == "relief_reference"


def test_project_routes_the_exchange_dir_into_the_blender_lane(tmp_path, monkeypatch):
    """AtlasProject wired: exchange folder = <shot>/blender/<basename> for massing
    AND import (same project → same folder), superseding an absolute path."""
    from atlas_camera.comfy.nodes_geometry import _blender_exchange_dir

    class _Proj:
        def subdir(self, lane, create=True):
            p = tmp_path / "PRJ" / "sh010" / lane
            if create:
                p.mkdir(parents=True, exist_ok=True)
            return p

    m = _blender_exchange_dir("atlas_exports/blender_massing", tag="massing", project=_Proj())
    i = _blender_exchange_dir("atlas_exports/blender_massing", tag="import", project=_Proj(), create=False)
    assert m == i == tmp_path / "PRJ" / "sh010" / "blender" / "blender_massing"
    assert m.is_dir()
    d = _blender_exchange_dir("", tag="massing", project=_Proj())
    assert d.name == "massing"
    assert _blender_exchange_dir("", tag="import", create=False) is None
    assert _blender_exchange_dir(str(tmp_path / "x"), tag="import", create=False) == tmp_path / "x"


def test_reimport_of_the_same_exchange_mesh_is_skipped(tmp_path, monkeypatch):
    from atlas_camera.comfy.nodes_geometry import AtlasBlenderImportMeshes
    v, f = _box()
    _write_out(tmp_path, [("box", v, f, {})])
    s = _solve()
    out1, rep1 = AtlasBlenderImportMeshes().import_meshes(s, exchange_dir=str(tmp_path))
    out2, rep2 = AtlasBlenderImportMeshes().import_meshes(out1, exchange_dir=str(tmp_path),
                                                          name_prefix="again")
    names = [p.name for p in out2.projection_scene.proxy_geometry]
    assert names.count("blender_box") == 1 and "again_box" not in names
    assert "already imported" in rep2


@pytest.mark.skipif(not _blender_available(), reason="no Blender install found")
def test_live_massing_rerun_preserves_agent_objects(tmp_path):
    """A re-queue re-runs massing; artist/agent meshes under atlas_out survive
    and are exported (found live 2026-08-16 — they were being clobbered)."""
    from atlas_camera.blender import run_recipe, read_meshes, write_scene_seed
    s = _solve()
    seed = seed_from_solve(s)
    params = {"default_height_m": 3.0, "ground_extent_m": 20.0, "ground_plane": True,
              "footprint_source": "both", "save_blend": True, "solve_fingerprint": seed["fingerprint"]}
    write_scene_seed(tmp_path, camera=seed["camera"], primitives=seed["primitives"], params=params)
    run_recipe("massing.py", tmp_path, timeout_s=240)
    # an "agent" adds a mesh under atlas_out via a tiny headless script
    script = tmp_path / "add.py"
    script.write_text(
        "import bpy,sys\n"
        "bpy.ops.wm.open_mainfile(filepath=r'%s')\n"
        "me=bpy.data.meshes.new('agent_thing'); me.from_pydata([(0,0,1),(1,0,1),(0,1,1)],[],[(0,1,2)]); me.update()\n"
        "ob=bpy.data.objects.new('agent_thing',me); bpy.data.collections['atlas_out'].objects.link(ob)\n"
        "ob['atlas_source']='agent'\n"
        "bpy.ops.wm.save_as_mainfile(filepath=r'%s')\n" % (tmp_path / "scene.blend", tmp_path / "scene.blend"),
        encoding="utf-8")
    import subprocess
    from atlas_camera.blender import resolve_blender_exe
    subprocess.run([str(resolve_blender_exe("")), "--background", "--factory-startup",
                    "--python", str(script)], capture_output=True, timeout=240)
    rep = run_recipe("massing.py", tmp_path, timeout_s=240)        # the re-run
    assert rep.get("preserved_objects") == 1
    names = [m["name"] for m in read_meshes(tmp_path)["meshes"]]
    assert "agent_thing" in names and names.count("ground_plane") == 1


def test_measured_floor_follows_the_cloud_minimum():
    from atlas_camera.comfy.nodes_geometry import _measured_floor
    assert _measured_floor({"ground_y_m": 0.0}, -0.05) == pytest.approx(-0.05)
    # coastal plate: cloud bottoms out at -5.4 (water) -> floor -6.4, not -0.05
    assert _measured_floor({"ground_y_m": 0.0, "measured": {"bbox_min": [-10, -5.4, -300]}}, -0.05) == pytest.approx(-6.4)
    # a cloud that never goes below ground keeps the widget floor
    assert _measured_floor({"ground_y_m": 0.0, "measured": {"bbox_min": [-10, 0.3, -300]}}, -0.05) == pytest.approx(-0.7)


def test_paint_with_rides_the_primitive_and_the_viewport_honours_it():
    from pathlib import Path
    s = _solve()
    v, f = _box()
    prims, _ = meshes_to_primitives(
        s, [{"name": "water", "vertices": v, "faces": f},
            {"name": "facade", "vertices": v, "faces": f, "paint": "source_photo"}],
        paint_with="clean_plate")
    assert prims[0].metadata["paint_with"] == "clean_plate"
    assert prims[1].metadata["paint_with"] == "source_photo"      # per-mesh override
    js = (Path(__file__).resolve().parents[1] / "atlas_camera/comfy/web/atlas_blockout.js").read_text(encoding="utf-8")
    assert 'e.metadata?.paint_with === "clean_plate"' in js


def test_massing_node_imports_only_its_own_meshes(tmp_path, monkeypatch):
    """Preserved agent objects ride out_meshes.npz but are NOT appended by
    massing — the handoff/import nodes own them (and their paint_with)."""
    from atlas_camera.comfy.nodes_geometry import AtlasBlenderMassing
    import atlas_camera.blender as B
    v, f = _box()
    def fake(recipe_name, exdir, *, blender_path="", timeout_s=300, blend_file=""):
        _write_out(exdir, [("ground_plane", v, f, {"kind": "ground_plane", "source": "blender_massing"}),
                           ("agent_water", v, f, {"kind": "agent_surface", "source": "agent_blender_mcp"})])
        return {"blender_version": "5.2.0", "meshes_out": 2}
    monkeypatch.setattr(B, "run_recipe", fake)
    out, report, _ = AtlasBlenderMassing().massing(_solve(), exchange_dir=str(tmp_path))
    names = [p.name for p in out.projection_scene.proxy_geometry]
    assert "massing_ground_plane" in names and not any("agent_water" in n for n in names)
    assert "1 preserved non-massing mesh(es) left" in report
