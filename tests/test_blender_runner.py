"""Contract for the headless-Blender plumbing. Runs with NO Blender installed.

Mirrors tests/test_fixer_render_fix.py — the repo's existing "shell out to a big
external tool" test shape. Everything here is argv construction, path
resolution, and array round-tripping; nothing launches Blender.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.blender import (  # noqa: E402
    BLENDER_PATH_ENV,
    atlas_to_blender,
    blender_to_atlas,
    build_blender_command,
    read_result,
    recipes_dir,
    resolve_blender_exe,
    write_exchange,
)
from atlas_camera.blender import convert, runner  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def no_blender(monkeypatch):
    """Nothing on PATH, nothing installed — the CI machine."""
    monkeypatch.delenv(BLENDER_PATH_ENV, raising=False)
    monkeypatch.setattr(runner.shutil, "which", lambda _n: None)
    monkeypatch.setattr(runner, "_candidates", list)


class TestLocatingBlender:
    def test_absent_blender_names_the_widget_env_download_and_paths_probed(
            self, no_blender):
        """A user must never have to guess where it looked."""
        with pytest.raises(RuntimeError) as exc:
            resolve_blender_exe("")
        msg = str(exc.value)
        assert BLENDER_PATH_ENV in msg
        assert "blender.org" in msg
        assert "blender_path widget" in msg
        assert "Probed and found nothing at" in msg

    def test_an_explicit_path_wins_over_the_env_var(self, tmp_path, monkeypatch):
        want = tmp_path / "want.exe"
        want.write_text("", encoding="utf-8")
        other = tmp_path / "other.exe"
        other.write_text("", encoding="utf-8")
        monkeypatch.setenv(BLENDER_PATH_ENV, str(other))
        assert resolve_blender_exe(str(want)) == want

    def test_the_env_var_is_used_when_no_argument_is_given(self, tmp_path,
                                                           monkeypatch):
        exe = tmp_path / "blender.exe"
        exe.write_text("", encoding="utf-8")
        monkeypatch.setenv(BLENDER_PATH_ENV, str(exe))
        assert resolve_blender_exe("") == exe

    def test_a_configured_path_that_does_not_exist_says_which_source_set_it(
            self, tmp_path, monkeypatch):
        monkeypatch.delenv(BLENDER_PATH_ENV, raising=False)
        with pytest.raises(RuntimeError, match="points at nothing executable"):
            resolve_blender_exe(str(tmp_path / "nope.exe"))

    def test_the_newest_install_wins_when_several_are_present(self, tmp_path,
                                                              monkeypatch):
        """Two Blenders side by side is normal. Picking whichever the
        filesystem lists first would make behaviour depend on directory order.
        """
        monkeypatch.delenv(BLENDER_PATH_ENV, raising=False)
        monkeypatch.setattr(runner.shutil, "which", lambda _n: None)
        cands = []
        for name in ("Blender 4.2", "Blender 5.2", "Blender 5.10"):
            d = tmp_path / name
            d.mkdir()
            exe = d / "blender.exe"
            exe.write_text("", encoding="utf-8")
            cands.append(exe)
        monkeypatch.setattr(runner, "_candidates", lambda: list(cands))
        assert resolve_blender_exe("").parent.name == "Blender 5.10"


class TestCommandContract:
    def test_the_argv_has_everything_a_headless_recipe_needs(self, tmp_path):
        cmd = build_blender_command(Path("/x/blender"), tmp_path / "r.py",
                                    tmp_path / "ex")
        assert isinstance(cmd, list), "a list, never a shell string"
        assert "--background" in cmd
        assert "--factory-startup" in cmd, (
            "without it a user's addons are an unversioned input to a "
            "supposedly deterministic construction")
        assert "--python" in cmd
        assert "--exchange" in cmd

    def test_the_bare_ddash_precedes_the_script_arguments(self, tmp_path):
        """Blender consumes everything before `--`; omitting it is the classic
        silent failure where the recipe never sees its arguments."""
        cmd = build_blender_command(Path("/x/blender"), tmp_path / "r.py",
                                    tmp_path / "ex")
        assert "--" in cmd
        assert cmd.index("--") < cmd.index("--exchange")
        assert cmd.index("--python") < cmd.index("--")

    def test_noaudio_is_NOT_passed(self, tmp_path):
        """Verified live against Blender 5.2: it is not a valid flag there, and
        Blender silently treats it as a filename to open ("Cannot read file
        C:/.../--noaudio") instead of rejecting it."""
        cmd = build_blender_command(Path("/x/blender"), tmp_path / "r.py",
                                    tmp_path / "ex")
        assert "--noaudio" not in cmd
        assert "-noaudio" not in cmd


class TestRecipesAreDataNotCode:
    def test_the_probe_recipe_ships(self):
        assert (recipes_dir() / "probe_api.py").is_file()

    def test_recipes_is_not_an_importable_package(self):
        """`import bpy` at module top would break any import sweep and confuse
        pytest collection. It must ship as package DATA."""
        assert not (recipes_dir() / "__init__.py").exists()

    @pytest.mark.parametrize("name", ["probe_api.py"])
    def test_a_recipe_never_imports_atlas(self, name):
        """Recipes run inside Blender's interpreter, which has no Atlas on its
        path. An `import atlas_camera` there is a guaranteed runtime failure."""
        tree = ast.parse((recipes_dir() / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not a.name.startswith("atlas_camera")
                           for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("atlas_camera")


class TestCoordinateConvention:
    def test_the_round_trip_is_exact(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(64, 3)) * 25.0
        np.testing.assert_allclose(
            blender_to_atlas(atlas_to_blender(pts)), pts, rtol=0, atol=0)

    def test_T_is_a_proper_rotation(self):
        """det = +1, so face winding survives — which matters because
        shrinkwrap PROJECT follows vertex normals."""
        assert np.linalg.det(convert.transform_matrix(np)) == pytest.approx(1.0)

    def test_it_matches_the_matrix_the_scene_EXPORTER_emits(self):
        """docs/DCC_EXPORTS.md records the exporter's verification as "Script
        inspection" — the emitted matrix has never been executed. Parsing it and
        checking it against the transform actually applied to geometry closes
        that gap.

        Exporter rows: [x, -z, y] from the Atlas world matrix.
        """
        src = (ROOT / "atlas_camera" / "exporters"
               / "blender_exporter.py").read_text(encoding="utf-8")
        assert "-wm[2][0], -wm[2][1], -wm[2][2]" in src, (
            "exporter row 1 is no longer -Atlas Z; convert.T must move with it")
        assert "wm[1][0],  wm[1][1],  wm[1][2]" in src, (
            "exporter row 2 is no longer +Atlas Y")
        T = convert.transform_matrix(np)
        np.testing.assert_allclose(T[1], (0.0, 0.0, -1.0))
        np.testing.assert_allclose(T[2], (0.0, 1.0, 0.0))


class TestExchange:
    def _mesh(self):
        v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
        f = np.array([[0, 1, 2], [1, 3, 2]])
        return v, f

    def test_written_arrays_are_in_BLENDER_space(self, tmp_path):
        v, f = self._mesh()
        write_exchange(tmp_path, patch_vertices=v, patch_faces=f,
                       target_vertices=v, target_faces=f,
                       camera_position=[0., 1., 2.], params={"voxel_size_m": 0.1})
        with np.load(tmp_path / "in.npz") as d:
            np.testing.assert_allclose(d["patch_vertices"], atlas_to_blender(v))
            np.testing.assert_allclose(d["camera_position"],
                                       atlas_to_blender([[0., 1., 2.]])[0])
        assert json.loads((tmp_path / "params.json").read_text(
            encoding="utf-8"))["voxel_size_m"] == 0.1

    def test_results_come_back_in_ATLAS_space(self, tmp_path):
        v, f = self._mesh()
        np.savez(tmp_path / "out.npz", vertices=atlas_to_blender(v),
                 faces=f.astype(np.int32),
                 snapped=np.ones(len(v), dtype=bool))
        got = read_result(tmp_path)
        np.testing.assert_allclose(got["vertices"], v)
        assert got["snapped"].all()

    def test_a_missing_result_is_named_not_silently_empty(self, tmp_path):
        with pytest.raises(RuntimeError, match="produced no out.npz"):
            read_result(tmp_path)

    def test_zero_faces_is_refused_with_a_reason(self, tmp_path):
        np.savez(tmp_path / "out.npz", vertices=np.zeros((3, 3)),
                 faces=np.zeros((0, 3), dtype=np.int32))
        with pytest.raises(RuntimeError, match="no geometry"):
            read_result(tmp_path)

    def test_nan_vertices_are_refused(self, tmp_path):
        """They would be written into the solve and surface much later as
        geometry in the wrong place."""
        v = np.zeros((3, 3))
        v[1, 1] = np.nan
        np.savez(tmp_path / "out.npz", vertices=v,
                 faces=np.array([[0, 1, 2]], dtype=np.int32))
        with pytest.raises(RuntimeError, match="non-finite"):
            read_result(tmp_path)

    def test_out_of_range_face_indices_are_refused(self, tmp_path):
        np.savez(tmp_path / "out.npz", vertices=np.zeros((3, 3)),
                 faces=np.array([[0, 1, 9]], dtype=np.int32))
        with pytest.raises(RuntimeError, match="out of range"):
            read_result(tmp_path)

    def test_a_result_without_snapped_defaults_to_none_snapped(self, tmp_path):
        """Absent evidence must read as "nothing landed on measured surface",
        never as "everything did" — the drift gate keys off this."""
        np.savez(tmp_path / "out.npz", vertices=np.zeros((3, 3)),
                 faces=np.array([[0, 1, 2]], dtype=np.int32))
        assert not read_result(tmp_path)["snapped"].any()


class TestLayering:
    def test_the_package_never_imports_comfy(self):
        """core/ and blender/ are host-agnostic; comfy imports them, never back."""
        pkg = ROOT / "atlas_camera" / "blender"
        for path in pkg.glob("*.py"):
            src = path.read_text(encoding="utf-8")
            assert "atlas_camera.comfy" not in src, f"{path.name} imports comfy"


class TestShrinkwrapLimitIntegration:
    """Behavior contract for the distance limit — needs a real Blender.

    The NEAREST_SURFACEPOINT wrap (the node's default) has no native distance
    cap in Blender — ``project_limit`` is PROJECT-only — so the recipe must
    clamp displacements itself or the ``shrinkwrap_limit_scale`` widget is a
    no-op and fill vertices snap onto unrelated geometry metres away (found
    live: a 4.11 m max move against a sub-metre limit on ghosttown).
    """

    @pytest.fixture
    def blender_exe(self):
        try:
            from atlas_camera.blender import resolve_blender_exe as _r
            return str(_r(""))
        except RuntimeError:
            pytest.skip("no local Blender install")

    def test_nearest_wrap_honors_the_distance_limit(self, blender_exe):
        from atlas_camera.blender import shrinkwrap_patch

        patch_v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        patch_f = np.array([[0, 1, 2]])
        # Target 10 m away with a 1 m median edge: limit_scale=1 -> 1 m limit,
        # nearest surface at 10 m -> nothing may snap.
        target_v = np.array([[0.0, 0.0, -10.0], [1.0, 0.0, -10.0],
                             [0.0, 1.0, -10.0]])
        target_f = np.array([[0, 1, 2]])
        got = shrinkwrap_patch(patch_v, patch_f, target_v, target_f,
                               blender_path=blender_exe, limit_scale=1.0)
        out_v = np.asarray(got["vertices"], dtype=np.float64)
        assert np.allclose(out_v, patch_v, atol=1e-6),             f"verts snapped past the limit: {out_v}"
