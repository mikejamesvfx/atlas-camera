"""AtlasLoadRAW node + raw_meta hint precedence (mocked import_raw — no rawpy)."""

import sys
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from atlas_camera.comfy.nodes import (
    NODE_CLASS_MAPPINGS,
    AtlasLoadRAW,
    _resolve_raw_hints,
)
from atlas_camera.raw.pipeline import RawImportResult


def _fake_result(**overrides):
    h, w = 8, 12
    linear = np.full((h, w, 3), 0.25, dtype=np.float32)
    defaults = dict(
        linear_rgb=linear,
        display_srgb=np.clip(linear * 2.0, 0.0, 1.0),
        width=w, height=h,
        focal_length_mm=20.0,
        sensor_width_mm=35.9,
        sensor_height_mm=24.0,
        sensor_source="camera_db",
        camera_make="NIKON CORPORATION",
        camera_model="NIKON D810",
        lens_model="AF-S NIKKOR 20mm f/1.8G ED",
        undistort_applied=True,
        undistort_status="applied",
        source_path="D:/shots/DSC_0001.NEF",
        orientation=1,
    )
    defaults.update(overrides)
    return RawImportResult(**defaults)


@pytest.fixture
def raw_file(tmp_path, monkeypatch):
    path = tmp_path / "DSC_0001.NEF"
    path.write_bytes(b"not really a nef")
    return str(path)


def _patch_import(monkeypatch, result):
    import atlas_camera.raw.pipeline as pipeline
    monkeypatch.setattr(pipeline, "import_raw", lambda *a, **k: result)


def test_node_registered():
    assert NODE_CLASS_MAPPINGS["AtlasLoadRAW"] is AtlasLoadRAW


def test_load_outputs(monkeypatch, raw_file):
    _patch_import(monkeypatch, _fake_result())
    monkeypatch.setattr(AtlasLoadRAW, "_write_exr_sidecar",
                        staticmethod(lambda *a, **k: ("out/DSC_0001_linear.exr", None)))
    image, plate_ref, raw_meta, focal, sensor_w, report = AtlasLoadRAW().load(raw_file)
    assert tuple(image.shape) == (1, 8, 12, 3)
    assert image.dtype == torch.float32
    assert focal == pytest.approx(20.0)
    assert sensor_w == pytest.approx(35.9)
    assert raw_meta.camera_model == "NIKON D810"
    assert plate_ref.image_path == "out/DSC_0001_linear.exr"
    assert plate_ref.is_proxy is False
    assert plate_ref.bit_depth == "16f"
    assert plate_ref.metadata["undistort_status"] == "applied"
    assert "NIKON D810" in report and "camera_db" in report and "applied" in report


def test_exr_failure_degrades_to_proxy(monkeypatch, raw_file):
    _patch_import(monkeypatch, _fake_result())
    monkeypatch.setattr(AtlasLoadRAW, "_write_exr_sidecar",
                        staticmethod(lambda *a, **k: (None, "EXR sidecar FAILED: codec")))
    _, plate_ref, _, _, _, report = AtlasLoadRAW().load(raw_file)
    assert plate_ref.image_path is None
    assert plate_ref.is_proxy is True
    assert "FAILED" in report


def test_missing_metadata_uses_sentinels(monkeypatch, raw_file):
    _patch_import(monkeypatch, _fake_result(
        focal_length_mm=None, sensor_width_mm=36.0, sensor_height_mm=None,
        sensor_source="assumed_default", camera_make=None, camera_model=None,
        lens_model=None, undistort_status="no_lens_metadata",
        undistort_applied=False))
    _, _, _, focal, sensor_w, report = AtlasLoadRAW().load(
        raw_file, write_exr=False)
    assert focal == 0.0
    assert sensor_w == pytest.approx(36.0)
    assert "no_lens_metadata" in report


def test_missing_file_raises():
    with pytest.raises(RuntimeError, match="not found"):
        AtlasLoadRAW().load("Z:/nowhere/missing.arw")


def test_relative_raw_path_resolves_from_comfy_input(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    raw_path = input_dir / "atlas_Input_Examples" / "DSC_2328.NEF"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(b"raw")
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(
        get_input_directory=lambda: str(input_dir)))

    resolved = AtlasLoadRAW._resolve_input_path(
        r"atlas_Input_Examples\DSC_2328.NEF")

    assert resolved == raw_path
    assert AtlasLoadRAW.IS_CHANGED(
        r"atlas_Input_Examples\DSC_2328.NEF").startswith(str(raw_path))


def test_hint_precedence_widget_beats_raw_meta():
    meta = _fake_result()
    focal, sensor_w, sensor_h = _resolve_raw_hints(24.0, 36.0, meta)
    assert focal == pytest.approx(24.0)  # widget wins
    # widget sensor untouched (36.0 default) -> raw_meta sensor adopted
    assert sensor_w == pytest.approx(35.9)
    assert sensor_h == pytest.approx(24.0)


def test_hint_precedence_raw_meta_fills_zero_widget():
    focal, sensor_w, sensor_h = _resolve_raw_hints(0.0, 36.0, _fake_result())
    assert focal == pytest.approx(20.0)
    assert sensor_w == pytest.approx(35.9)


def test_hint_precedence_explicit_sensor_widget_wins():
    focal, sensor_w, sensor_h = _resolve_raw_hints(0.0, 23.5, _fake_result())
    assert sensor_w == pytest.approx(23.5)
    assert sensor_h is None


def test_no_raw_meta_no_hints():
    focal, sensor_w, sensor_h = _resolve_raw_hints(0.0, 36.0, None)
    assert focal is None and sensor_w == 36.0 and sensor_h is None


def test_widget_order_pins():
    """Positional widget serialization: these orders are frozen forever."""
    from atlas_camera.mcp.comfy_http import is_widget
    input_types = AtlasLoadRAW.INPUT_TYPES()
    # WIDGETS only — `project` (ATLAS_PROJECT, appended 2026-08-07) is a link
    # input and occupies no positional slot, so it cannot shift these.
    load_raw_optional = [n for n, spec in input_types["optional"].items()
                         if is_widget(spec)]
    assert load_raw_optional == ["undistort", "half_size", "white_balance",
                                 "exposure_ev", "write_exr", "output_dir",
                                 "colorspace", "headroom"]
    exr_tooltip = input_types["optional"]["write_exr"][1]["tooltip"]
    assert "OpenImageIO" in exr_tooltip
    assert "OpenCV is not used" in exr_tooltip
    from atlas_camera.comfy.nodes import AtlasLearnedSolveFromImage
    learned_optional = list(AtlasLearnedSolveFromImage.INPUT_TYPES()["optional"].keys())
    # focal_length_mm was APPENDED 2026-07-18; raw_meta is a link input (last).
    assert learned_optional[-2:] == ["focal_length_mm", "raw_meta"]
    assert learned_optional[:3] == ["height_mode", "camera_height_m", "depth_model"]


def test_intrinsics_hint_shape():
    hint = _fake_result().intrinsics_hint()
    assert hint == {"focal_length_mm": 20.0, "sensor_width_mm": 35.9,
                    "sensor_height_mm": 24.0}
    assert _fake_result(focal_length_mm=None, sensor_width_mm=None,
                        sensor_height_mm=None).intrinsics_hint() == {}


# --- delivery-project routing -------------------------------------------------
#
# Found live 2026-08-07: on the VFX lane every DCC export lands under
# <root>/<project>/<shot>/, but AtlasLoadRAW's developed EXR went to its own
# loose output_dir, so the ONE file the colour story is about — the undistorted
# scene-linear plate — was the only thing outside the delivery tree.
# AtlasExportPlateEXR already routed to the `plates` lane; this brings the
# loader to the same place by the same helper.


def test_load_raw_routes_the_exr_into_the_project_plates_lane(monkeypatch, raw_file,
                                                              tmp_path):
    from atlas_camera.core.project import build_project
    _patch_import(monkeypatch, _fake_result())
    monkeypatch.setattr(AtlasLoadRAW, "_write_exr_sidecar",
                        staticmethod(lambda *a, **k: ("out/DSC_0001_linear.exr", None)))
    project = build_project(str(tmp_path / "deliveries"), "p", "s010",
                            "VFX (ACEScg / float)")
    seen = {}

    import atlas_camera.comfy.nodes_solve as solve_mod
    real = solve_mod._project_routed_dir

    def spy(proj, output_dir, lane):
        seen["lane"] = lane
        seen["result"] = real(proj, output_dir, lane)
        return seen["result"]

    solve_mod._project_routed_dir = spy
    try:
        AtlasLoadRAW().load(raw_file, project=project)
    finally:
        solve_mod._project_routed_dir = real

    assert seen["lane"] == "plates"
    assert seen["result"] == str(project.subdir("plates"))


def test_load_raw_without_a_project_keeps_its_own_output_dir(raw_file):
    """The no-project fallback every existing workflow relies on."""
    import atlas_camera.comfy.nodes_solve as solve_mod
    assert solve_mod._project_routed_dir(None, "atlas_exports/raw_plates",
                                         "plates") == "atlas_exports/raw_plates"


def test_load_raw_project_is_a_link_input_appended_after_the_frozen_widgets():
    """The widget order above is frozen; project must be a LINK input, so it
    takes no positional slot and cannot shift a saved workflow."""
    optional = AtlasLoadRAW.INPUT_TYPES()["optional"]
    assert list(optional)[-1] == "project"
    assert optional["project"][0] == "ATLAS_PROJECT"


# --- headroom widget (docs/dev/raw_aces_headroom_and_colorspace_fix.md) -------


def test_headroom_widget_is_appended_last_among_the_frozen_widgets():
    from atlas_camera.mcp.comfy_http import is_widget
    optional = AtlasLoadRAW.INPUT_TYPES()["optional"]
    widgets = [n for n, spec in optional.items() if is_widget(spec)]
    assert widgets[-1] == "headroom", (
        "widgets_values is positional — headroom must APPEND, never insert")
    spec = optional["headroom"]
    assert spec[0] == "FLOAT"
    assert spec[1]["default"] == 6.0


def test_load_passes_headroom_through_to_import_raw(monkeypatch, raw_file):
    import atlas_camera.raw.pipeline as pipeline
    seen = {}

    def spy(*a, **kw):
        seen.update(kw)
        return _fake_result(headroom=kw.get("headroom", 6.0))

    monkeypatch.setattr(pipeline, "import_raw", spy)
    monkeypatch.setattr(AtlasLoadRAW, "_write_exr_sidecar",
                        staticmethod(lambda *a, **k: ("out/x.exr", None)))
    AtlasLoadRAW().load(raw_file, headroom=6.0)
    assert seen["headroom"] == 6.0


def test_report_names_the_headroom_factor(monkeypatch, raw_file):
    """The default changes the delivered plate's exposure by 2.6 stops, so a
    graph saved before this landed silently gets brighter EXRs. That is the
    fix, not a regression — but it has to be unmissable in the report."""
    _patch_import(monkeypatch, _fake_result(headroom=6.0))
    monkeypatch.setattr(AtlasLoadRAW, "_write_exr_sidecar",
                        staticmethod(lambda *a, **k: ("out/x.exr", None)))
    report = AtlasLoadRAW().load(raw_file, headroom=6.0)[5]
    assert "headroom: 6x" in report


def test_report_flags_a_headroom_of_one_as_not_aces_referred(monkeypatch, raw_file):
    _patch_import(monkeypatch, _fake_result(headroom=1.0))
    monkeypatch.setattr(AtlasLoadRAW, "_write_exr_sidecar",
                        staticmethod(lambda *a, **k: ("out/x.exr", None)))
    report = AtlasLoadRAW().load(raw_file, headroom=1.0)[5]
    assert "NOT ACES-referred" in report


def test_the_exr_records_the_headroom_it_was_written_with(monkeypatch, raw_file):
    """atlas:headroom comes from the import RESULT, never from the widget, so
    the attribute cannot drift from the pixels it describes."""
    seen = {}

    def fake_sidecar(linear_rgb, raw_path, output_dir, *, headroom=6.0):
        seen["headroom"] = headroom
        return ("out/x.exr", None)

    _patch_import(monkeypatch, _fake_result(headroom=6.0))
    monkeypatch.setattr(AtlasLoadRAW, "_write_exr_sidecar",
                        staticmethod(fake_sidecar))
    AtlasLoadRAW().load(raw_file, headroom=6.0)
    assert seen["headroom"] == 6.0
