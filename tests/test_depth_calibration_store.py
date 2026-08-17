"""The calibration store, and the two nodes that write and read it.

Step (2)+(4) of docs/ROADMAP.md. The behaviour these pin is mostly about what
must NOT happen: no fallback on a lookup miss, no silent passthrough, no
coefficient saved unless the artist asked for it, and no change to
`AtlasDepthMap` — a stored number must never reach shared depth through a file
the graph does not name.
"""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.depth_calibration import (  # noqa: E402
    DepthCorrection, fit_depth_correction,
)
from atlas_camera.core.depth_calibration_store import (  # noqa: E402
    SCENE_TYPES, STORE_SCHEMA_VERSION, CalibrationStore, store_key,
)

MODEL_A = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
MODEL_B = "Ruicheng/moge-2-vitl-normal"


def _correction(scale=2.0):
    truth = np.linspace(1.5, 30.0, 4096)
    return fit_depth_correction(truth / scale, truth, model="scale")


# ------------------------------------------------------------------- store


def test_a_missing_file_is_an_empty_store_not_an_error(tmp_path):
    """A fresh clone has no calibrations; that is the expected state."""
    store = CalibrationStore.load(tmp_path / "nothing.json")
    assert len(store) == 0
    assert store.lookup(MODEL_A, "outdoor") is None
    assert store.describe() == "no calibrations stored"


def test_round_trips_through_disk(tmp_path):
    path = tmp_path / "cal.json"
    store = CalibrationStore()
    corr = _correction()
    store.put(MODEL_A, "outdoor", corr, note="D810, coastal, 2026-08-17")
    store.save(path)

    back = CalibrationStore.load(path)
    got = back.lookup(MODEL_A, "outdoor")
    assert got is not None
    assert got.model == corr.model
    assert got.a == pytest.approx(corr.a)
    assert got.predicted_range == pytest.approx(corr.predicted_range)
    assert back.note_for(MODEL_A, "outdoor") == "D810, coastal, 2026-08-17"


def test_the_file_is_readable_and_versioned(tmp_path):
    """An artist must be able to open it and delete a bad coefficient."""
    path = tmp_path / "cal.json"
    store = CalibrationStore()
    store.put(MODEL_A, "outdoor", _correction())
    store.save(path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == STORE_SCHEMA_VERSION
    assert raw["entries"][0]["model_id"] == MODEL_A
    assert raw["entries"][0]["correction"]["schema_version"] == 1


def test_a_future_store_version_is_refused(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(
        {"schema_version": STORE_SCHEMA_VERSION + 1, "entries": []}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="newer than this build"):
        CalibrationStore.load(path)


def test_lookup_is_exact_and_never_falls_back():
    """The whole failure mode this module exists to prevent.

    A near-miss fallback is how a coefficient fitted on a 1.2 m interior wall
    ends up rescaling a 200 m exterior.
    """
    store = CalibrationStore()
    store.put(MODEL_A, "indoor", _correction())

    assert store.lookup(MODEL_A, "indoor") is not None
    assert store.lookup(MODEL_A, "outdoor") is None      # wrong scene
    assert store.lookup(MODEL_B, "indoor") is None       # wrong model
    assert store.lookup(MODEL_B, "outdoor") is None


def test_the_same_model_holds_independent_scene_entries():
    store = CalibrationStore()
    store.put(MODEL_A, "indoor", _correction(scale=2.0))
    store.put(MODEL_A, "outdoor", _correction(scale=5.0))
    assert len(store) == 2
    assert store.lookup(MODEL_A, "indoor").a != store.lookup(MODEL_A, "outdoor").a


def test_an_unknown_scene_type_is_refused():
    store = CalibrationStore()
    with pytest.raises(ValueError, match="unknown scene_type"):
        store.put(MODEL_A, "underwater", _correction())


def test_scene_types_mirror_the_derive_node_vocabulary():
    """Combo values are append-only across this repo; the store must not
    invent a vocabulary the graph that produced it does not use."""
    from atlas_camera.comfy.nodes_geometry import AtlasDeriveProjectionGeometry
    presets = set(AtlasDeriveProjectionGeometry._SCENE_TYPE_PRESETS)
    assert presets <= set(SCENE_TYPES)
    assert "manual" in SCENE_TYPES


def test_store_key_is_stable():
    assert store_key(MODEL_A, "outdoor") == f"{MODEL_A}::outdoor"


# -------------------------------------------------------------------- nodes


class _Depth:
    """Minimal stand-in for a DepthResult — the nodes only touch these."""

    def __init__(self, depth, model_id, is_metric=True):
        self.depth = depth
        self.model_id = model_id
        self.is_metric = is_metric
        self.metadata = {}


def _pair(scale=2.5):
    truth = np.linspace(1.5, 30.0, 64 * 64).reshape(64, 64)
    return (_Depth(truth, "record3d/arkit", is_metric=True),
            _Depth(truth / scale, MODEL_A, is_metric=True))


def test_fit_node_does_not_save_unless_asked(tmp_path):
    from atlas_camera.comfy.nodes_depth import AtlasFitDepthCalibration
    path = tmp_path / "cal.json"
    measured, predicted = _pair()

    _, report = AtlasFitDepthCalibration().fit(
        measured, predicted, scene_type="outdoor", store_path=str(path))
    assert not path.exists(), "save defaults to False — propose, never apply"
    assert "not saved" in report


def test_fit_node_saves_and_the_apply_node_reads_it_back(tmp_path):
    from atlas_camera.comfy.nodes_depth import (
        AtlasApplyDepthCalibration, AtlasFitDepthCalibration)
    path = tmp_path / "cal.json"
    measured, predicted = _pair(scale=2.5)

    payload, report = AtlasFitDepthCalibration().fit(
        measured, predicted, scene_type="outdoor", store_path=str(path),
        save=True, note="fixture")
    assert path.exists() and "SAVED" in report
    assert json.loads(payload)["model"] in ("scale", "affine", "affine_disparity")

    out, apply_report = AtlasApplyDepthCalibration().apply(
        _Depth(predicted.depth.copy(), MODEL_A), scene_type="outdoor",
        store_path=str(path))
    # the correction should recover the measured depth it was fitted against
    assert np.nanmedian(np.abs(out.depth - measured.depth)) < 0.05
    assert "applied" in apply_report and "fixture" in apply_report
    assert "depth_calibration" in out.metadata


def test_apply_node_says_so_when_there_is_no_calibration(tmp_path):
    """A silent passthrough is indistinguishable from a no-op correction."""
    from atlas_camera.comfy.nodes_depth import AtlasApplyDepthCalibration
    depth = _Depth(np.linspace(1.0, 40.0, 4096).reshape(64, 64), MODEL_A)

    out, report = AtlasApplyDepthCalibration().apply(
        depth, scene_type="outdoor", store_path=str(tmp_path / "absent.json"))
    assert out is depth
    assert "no calibration for" in report
    assert "UNCALIBRATED" in report


def test_apply_node_reports_a_wrong_scene_type_rather_than_guessing(tmp_path):
    from atlas_camera.comfy.nodes_depth import (
        AtlasApplyDepthCalibration, AtlasFitDepthCalibration)
    path = tmp_path / "cal.json"
    measured, predicted = _pair()
    AtlasFitDepthCalibration().fit(measured, predicted, scene_type="indoor",
                                   store_path=str(path), save=True)

    out, report = AtlasApplyDepthCalibration().apply(
        _Depth(predicted.depth, MODEL_A), scene_type="outdoor",
        store_path=str(path))
    assert out.depth is predicted.depth, "must not apply the indoor coefficient"
    assert "no calibration for" in report


def test_apply_node_disabled_is_a_stated_passthrough(tmp_path):
    from atlas_camera.comfy.nodes_depth import AtlasApplyDepthCalibration
    depth = _Depth(np.linspace(1.0, 40.0, 4096).reshape(64, 64), MODEL_A)
    out, report = AtlasApplyDepthCalibration().apply(depth, enabled=False)
    assert out is depth
    assert "disabled" in report


def test_fit_node_refuses_mismatched_resolutions(tmp_path):
    from atlas_camera.comfy.nodes_depth import AtlasFitDepthCalibration
    measured = _Depth(np.ones((64, 64)) * 5.0, "record3d/arkit")
    predicted = _Depth(np.ones((32, 32)) * 2.0, MODEL_A)
    payload, report = AtlasFitDepthCalibration().fit(measured, predicted)
    assert payload == ""
    assert "different resolutions" in report


def test_fit_node_flags_a_non_metric_truth_side():
    from atlas_camera.comfy.nodes_depth import AtlasFitDepthCalibration
    measured, predicted = _pair()
    measured.is_metric = False
    _, report = AtlasFitDepthCalibration().fit(measured, predicted)
    assert "NOT flagged metric" in report


def test_apply_node_carries_the_extrapolation_warning_through(tmp_path):
    """The range guard must survive the trip through the store and the node."""
    from atlas_camera.comfy.nodes_depth import AtlasApplyDepthCalibration
    path = tmp_path / "cal.json"
    store = CalibrationStore()
    store.put(MODEL_A, "outdoor",
              DepthCorrection(model="scale", a=2.0, b=0.0, n_samples=4096,
                              mae_before=1.0, mae_after=0.1,
                              predicted_range=(1.0, 5.0)))
    store.save(path)

    far = _Depth(np.linspace(100.0, 400.0, 4096).reshape(64, 64), MODEL_A)
    _, report = AtlasApplyDepthCalibration().apply(
        far, scene_type="outdoor", store_path=str(path))
    assert "extrapolated" in report
    assert "outside the fitted range" in report


# ------------------------------------------------------- nothing auto-applies


def test_atlas_depth_map_did_not_grow_a_calibration_input():
    """`ATLAS_DEPTH_MAP` feeds nine node modules. A stored coefficient must
    reach it only because an artist wired the apply node, never from a file
    the graph does not name."""
    from atlas_camera.comfy.nodes_depth import AtlasDepthMap
    spec = AtlasDepthMap.INPUT_TYPES()
    names = set(spec.get("required", {})) | set(spec.get("optional", {}))
    assert not any("calibrat" in n for n in names), sorted(names)
