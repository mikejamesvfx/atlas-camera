"""The plate's SDR->HDR tone curve, carried onto generated vertex colour."""

import numpy as np
import pytest

from atlas_camera.core.hdr_transfer import apply_curve, fit_sdr_to_hdr_curve, srgb_to_acescg


def _model(x_acescg):
    """A stand-in conversion: linear below 0.4, highlights expanded above."""
    return np.where(x_acescg < 0.4, x_acescg, 0.4 + (x_acescg - 0.4) * 6.0)


def _plate_pair(h=300, w=400):
    rng = np.random.default_rng(0)
    sdr = rng.random((h, w, 3)).astype(np.float32)
    return sdr, _model(srgb_to_acescg(sdr)).astype(np.float32)


def test_curve_recovers_the_conversion_and_applies_to_unseen_colours():
    sdr, hdr = _plate_pair()
    curve = fit_sdr_to_hdr_curve(sdr, hdr, bins=96)
    assert curve["residual_stops"] < 0.02
    vc = np.array([[0.2, 0.3, 0.4], [0.95, 0.9, 0.85], [0.5, 0.5, 0.5]], np.float32)
    got = apply_curve(curve, vc)
    want = _model(srgb_to_acescg(vc))
    assert np.max(np.abs(np.log2(got / want))) < 0.08
    assert got.max() > 1.0                       # highlights really expanded


def test_curve_is_monotone_and_extrapolates_upward():
    sdr, hdr = _plate_pair()
    curve = fit_sdr_to_hdr_curve(sdr * 0.8, hdr)  # plate never reaches white
    ys = np.asarray(curve["log2_y"])
    assert (np.diff(ys, axis=1) >= -1e-9).all()
    lo, hi = apply_curve(curve, np.array([[0.8, 0.8, 0.8], [1.0, 1.0, 1.0]], np.float32))
    assert (hi >= lo).all()


def test_misaligned_plates_refused():
    with pytest.raises(ValueError, match="pixel-aligned"):
        fit_sdr_to_hdr_curve(np.zeros((4, 4, 3)), np.zeros((4, 5, 3)))


def test_node_stores_hdr_vertex_colours_and_scene_writes_ply(tmp_path):
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes_object_mesh import AtlasHDRVertexTransfer
    from atlas_camera.core.proxy_geometry import PROXY_ROLE
    from atlas_camera.core.schema import (AtlasExtrinsics, AtlasIntrinsics,
                                          AtlasProxyPrimitive, AtlasSolve, LatentCamera)
    from atlas_camera.exporters.scene_glb import build_scene_layers

    solve = AtlasSolve(camera=LatentCamera(
        intrinsics=AtlasIntrinsics(image_width=400, image_height=300, focal_length_mm=35.0,
                                   sensor_width_mm=36.0, fx_px=300.0, fy_px=300.0),
        extrinsics=AtlasExtrinsics(camera_view_matrix=tuple(map(tuple, np.eye(4))))))
    verts = [[0, 0, -5], [1, 0, -5], [0, 1, -5]]
    solve.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
        name="pixal3d_object", primitive_type="mesh", dimensions=(0.0, 0.0, 0.0),
        material="m", metadata={"role": PROXY_ROLE, "source": "pixal3d",
                                "vertices": np.ravel(verts).tolist(), "faces": [0, 1, 2],
                                "uvs": [0, 0, 1, 0, 0, 1],
                                "vertex_colors": [0.95, 0.9, 0.85] * 3,
                                "photo_weight": [0.0, 0.0, 0.0]}))
    sdr, hdr = _plate_pair()
    out, report = AtlasHDRVertexTransfer().transfer(
        solve, torch.from_numpy(sdr)[None], hdr_image=torch.from_numpy(hdr)[None])
    meta = out.projection_scene.proxy_geometry[0].metadata
    assert len(meta["vertex_colors_hdr"]) == 9 and max(meta["vertex_colors_hdr"]) > 1.0
    assert meta["vertex_colors_hdr_space"] == "ACEScg" and "reconstruction" in report
    assert "vertex_colors_hdr" not in solve.projection_scene.proxy_geometry[0].metadata

    from PIL import Image
    layers, sidecars, _ = build_scene_layers(out, Image.new("RGB", (400, 300)), exr_dir=tmp_path,
                                             exr_prefix="t")
    ply = tmp_path / "t_pixal3d_object_vertex_hdr.ply"
    assert ply.is_file() and ply.read_bytes().startswith(b"ply\nformat binary_little_endian")
    assert layers[0].extras["vertex_colors_hdr_ply"] == ply.name

    # The float colours read back equal the ACEScg values written.
    data = ply.read_bytes()
    body = data[data.index(b"end_header\n") + len(b"end_header\n"):]
    rows = np.frombuffer(body[:3 * 24], dtype=[("p", "<f4", 3), ("c", "<f4", 3)])
    assert np.allclose(rows["p"], np.asarray(verts, dtype=np.float32))
    want = np.asarray(meta["vertex_colors_hdr"], dtype=np.float32).reshape(-1, 3)
    assert np.array_equal(rows["c"], want) and rows["c"].max() > 1.0
    tri = np.frombuffer(body[3 * 24:], dtype=[("n", "u1"), ("i", "<i4", 3)])
    assert tri["n"].tolist() == [3] and tri["i"].tolist() == [[0, 1, 2]]


# --- hdr_exr_path resolution and the IS_CHANGED fingerprint ---------------

def _fake_comfy_dirs(monkeypatch, tmp_path):
    import sys
    import types

    out, inp = tmp_path / "output", tmp_path / "input"
    out.mkdir()
    inp.mkdir()
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: str(out)
    fp.get_input_directory = lambda: str(inp)
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    return out, inp


def test_is_changed_tracks_the_file_not_the_string(tmp_path, monkeypatch):
    from atlas_camera.comfy.nodes_object_mesh import AtlasHDRVertexTransfer

    out, _ = _fake_comfy_dirs(monkeypatch, tmp_path)
    exr = out / "hdr.exr"
    missing = AtlasHDRVertexTransfer.IS_CHANGED(hdr_exr_path=str(exr))
    assert "missing" in missing
    exr.write_bytes(b"one")
    present = AtlasHDRVertexTransfer.IS_CHANGED(hdr_exr_path=str(exr))
    assert present != missing and str(exr) in present
    assert AtlasHDRVertexTransfer.IS_CHANGED(hdr_exr_path=str(exr)) == present  # stable
    exr.write_bytes(b"replaced at the same path")
    import os
    st = os.stat(exr)
    os.utime(exr, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    replaced = AtlasHDRVertexTransfer.IS_CHANGED(hdr_exr_path=str(exr))
    assert replaced not in (present, missing)
    assert AtlasHDRVertexTransfer.IS_CHANGED(hdr_exr_path="") != missing


def test_relative_path_resolves_under_output_then_input_only(tmp_path, monkeypatch):
    from atlas_camera.comfy.nodes_object_mesh import _resolve_read_path

    out, inp = _fake_comfy_dirs(monkeypatch, tmp_path)
    (inp / "plates").mkdir()
    (inp / "plates" / "a.exr").write_bytes(b"x")
    assert _resolve_read_path("plates/a.exr")[0] == str((inp / "plates" / "a.exr").resolve())
    (out / "plates").mkdir()
    (out / "plates" / "a.exr").write_bytes(b"y")              # output wins
    assert _resolve_read_path("plates/a.exr")[0] == str((out / "plates" / "a.exr").resolve())
    # `..` cannot climb out of either directory, and the cwd is never used.
    (tmp_path / "secret.exr").write_bytes(b"z")
    got, looked = _resolve_read_path("../secret.exr")
    assert got == "" and "escapes" in looked
    monkeypatch.chdir(tmp_path)
    assert _resolve_read_path("secret.exr")[0] == ""
    # Absolute paths inside output/input resolve only when the file exists.
    assert _resolve_read_path(str(out / "plates" / "a.exr"))[0] == str(out / "plates" / "a.exr")
    assert _resolve_read_path(str(out / "nope.exr"))[0] == ""


# --- absolute reads are confined (no file-existence oracle on --listen) ------

def test_absolute_inside_output_or_input_is_read(tmp_path, monkeypatch):
    from atlas_camera.comfy.nodes_object_mesh import _resolve_read_path

    monkeypatch.delenv("ATLAS_ALLOW_ABSOLUTE_READS", raising=False)
    out, inp = _fake_comfy_dirs(monkeypatch, tmp_path)
    for root in (out, inp):
        f = root / "atlas" / "hdr.exr"
        f.parent.mkdir()
        f.write_bytes(b"x")
        assert _resolve_read_path(str(f)) == (str(f), "")
    # Not found INSIDE an allowed root may still say so.
    got, looked = _resolve_read_path(str(out / "atlas" / "nope.exr"))
    assert got == "" and "does not exist" in looked


def test_absolute_outside_is_refused_without_revealing_existence(tmp_path, monkeypatch):
    from atlas_camera.comfy import nodes_object_mesh as nom

    monkeypatch.delenv("ATLAS_ALLOW_ABSOLUTE_READS", raising=False)
    monkeypatch.delenv("ATLAS_PROJECT_ROOT", raising=False)
    _fake_comfy_dirs(monkeypatch, tmp_path)
    present = tmp_path / "secret.exr"
    present.write_bytes(b"z")
    absent = tmp_path / "nope.exr"
    r_present = nom._resolve_read_path(str(present))
    r_absent = nom._resolve_read_path(str(absent))
    assert r_present == ("", nom.ABSOLUTE_READ_REFUSED) == r_absent
    assert "output/input" in nom.ABSOLUTE_READ_REFUSED
    assert "ATLAS_ALLOW_ABSOLUTE_READS=1" in nom.ABSOLUTE_READ_REFUSED
    assert str(tmp_path) not in nom.ABSOLUTE_READ_REFUSED
    # `..` through an allowed root cannot climb out either.
    climb = tmp_path / "output" / ".." / "secret.exr"
    assert nom._resolve_read_path(str(climb)) == ("", nom.ABSOLUTE_READ_REFUSED)
    # IS_CHANGED: one constant token, never a stat of the refused file.
    tok = nom.AtlasHDRVertexTransfer.IS_CHANGED(hdr_exr_path=str(present))
    assert tok == "atlas-hdr:refused"
    assert nom.AtlasHDRVertexTransfer.IS_CHANGED(hdr_exr_path=str(absent)) == tok


def test_refused_report_is_identical_whether_or_not_the_file_exists(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy import nodes_object_mesh as nom
    from atlas_camera.core.proxy_geometry import PROXY_ROLE
    from atlas_camera.core.schema import (AtlasIntrinsics, AtlasProxyPrimitive, AtlasSolve,
                                          LatentCamera)

    monkeypatch.delenv("ATLAS_ALLOW_ABSOLUTE_READS", raising=False)
    monkeypatch.delenv("ATLAS_PROJECT_ROOT", raising=False)
    _fake_comfy_dirs(monkeypatch, tmp_path)
    solve = AtlasSolve(camera=LatentCamera(intrinsics=AtlasIntrinsics(
        image_width=4, image_height=4, focal_length_mm=35.0, sensor_width_mm=36.0)))
    solve.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
        name="pixal3d_object", primitive_type="mesh", dimensions=(0.0, 0.0, 0.0),
        material="m", metadata={"role": PROXY_ROLE, "source": "pixal3d",
                                "vertex_colors": [0.5, 0.5, 0.5]}))
    (tmp_path / "secret.exr").write_bytes(b"z")
    reports = [nom.AtlasHDRVertexTransfer().transfer(
        solve, torch.zeros(1, 4, 4, 3), hdr_exr_path=str(tmp_path / name))[1]
        for name in ("secret.exr", "nope.exr")]
    assert reports[0] == reports[1]
    assert "refused" in reports[0] and "passed through" in reports[0]
    assert str(tmp_path) not in reports[0] and "not found" not in reports[0]


def test_absolute_read_opt_ins(tmp_path, monkeypatch):
    from atlas_camera.comfy import nodes_object_mesh as nom

    monkeypatch.delenv("ATLAS_ALLOW_ABSOLUTE_READS", raising=False)
    monkeypatch.delenv("ATLAS_PROJECT_ROOT", raising=False)
    _fake_comfy_dirs(monkeypatch, tmp_path)
    show = tmp_path / "show"
    show.mkdir()
    f = show / "hdr.exr"
    f.write_bytes(b"z")
    assert nom._resolve_read_path(str(f))[1] == nom.ABSOLUTE_READ_REFUSED
    monkeypatch.setenv("ATLAS_PROJECT_ROOT", str(show))       # the project root is allowed
    assert nom._resolve_read_path(str(f)) == (str(f), "")
    monkeypatch.delenv("ATLAS_PROJECT_ROOT")
    monkeypatch.setenv("ATLAS_ALLOW_ABSOLUTE_READS", "1")     # explicit opt-in: anywhere
    assert nom._resolve_read_path(str(f)) == (str(f), "")
    assert str(f) in nom.AtlasHDRVertexTransfer.IS_CHANGED(hdr_exr_path=str(f))
    missing, looked = nom._resolve_read_path(str(tmp_path / "nope.exr"))
    assert missing == "" and "does not exist" in looked


def test_report_names_the_chosen_path_and_where_it_looked(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy import nodes_object_mesh as nom
    from atlas_camera.core.proxy_geometry import PROXY_ROLE
    from atlas_camera.core.schema import (AtlasIntrinsics, AtlasProxyPrimitive, AtlasSolve,
                                          LatentCamera)

    _fake_comfy_dirs(monkeypatch, tmp_path)
    solve = AtlasSolve(camera=LatentCamera(intrinsics=AtlasIntrinsics(
        image_width=40, image_height=30, focal_length_mm=35.0, sensor_width_mm=36.0)))
    solve.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
        name="pixal3d_object", primitive_type="mesh", dimensions=(0.0, 0.0, 0.0),
        material="m", metadata={"role": PROXY_ROLE, "source": "pixal3d",
                                "vertex_colors": [0.5, 0.5, 0.5]}))
    _, report = nom.AtlasHDRVertexTransfer().transfer(
        solve, torch.zeros(1, 30, 40, 3), hdr_exr_path="atlas/missing.exr")
    assert "passed through" in report and "not found" in report and "output:" in report

    sdr, hdr = _plate_pair(30, 40)
    target = tmp_path / "output" / "atlas" / "hdr.exr"
    target.parent.mkdir()
    target.write_bytes(b"stub")

    class _Plate:
        pixels = hdr

    import atlas_camera.plate.oiio_io as oiio_io
    monkeypatch.setattr(oiio_io, "read_plate", lambda path, output_colorspace=None: _Plate())
    _, report = nom.AtlasHDRVertexTransfer().transfer(
        solve, torch.from_numpy(sdr)[None], hdr_exr_path="atlas/hdr.exr")
    assert f"hdr_exr_path -> {target.resolve()}" in report


def test_node_passes_through_without_hdr():
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes_object_mesh import AtlasHDRVertexTransfer
    from atlas_camera.core.schema import AtlasSolve, LatentCamera, AtlasIntrinsics
    solve = AtlasSolve(camera=LatentCamera(intrinsics=AtlasIntrinsics(
        image_width=4, image_height=4, focal_length_mm=35.0, sensor_width_mm=36.0)))
    _, report = AtlasHDRVertexTransfer().transfer(solve, torch.zeros(1, 4, 4, 3))
    assert "passed through" in report
