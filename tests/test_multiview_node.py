"""Public ComfyUI contract for deterministic photographed multi-view solves."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from atlas_camera.comfy.nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)
from atlas_camera.core.multiview_types import (
    RegistrationDiagnostics,
    RegistrationOutcome,
)
from atlas_camera.core.schema import AtlasPlateRef
from atlas_camera.raw.pipeline import RawImportResult


ROOT = Path(__file__).resolve().parents[1]


class _Tensor:
    """Small torch-shaped test double; node imports must stay torch-optional."""

    def __init__(self, values):
        self._values = np.asarray(values)

    @property
    def shape(self):
        return self._values.shape

    @property
    def dtype(self):
        return self._values.dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self._values


class _Torch:
    @staticmethod
    def from_numpy(values):
        return _Tensor(values)


def _image(value: float = 0.0, *, batch: int = 1) -> _Tensor:
    return _Tensor(np.full((batch, 3, 4, 3), value, dtype=np.float64))


def _image_from_raw(raw: RawImportResult, *, batch: int = 1) -> _Tensor:
    """Model the single BHWC IMAGE emitted beside this AtlasLoadRAW metadata."""
    return _Tensor(np.repeat(raw.display_srgb[None, ...], batch, axis=0))


def _raw(
    index: int,
    *,
    model: str = "Atlas RAW",
    display_srgb: np.ndarray | None = None,
    linear_rgb: np.ndarray | None = None,
) -> RawImportResult:
    pixels = np.full((3, 4, 3), index / 10.0, dtype=np.float32)
    display = pixels if display_srgb is None else np.asarray(display_srgb, dtype=np.float32)
    linear = pixels if linear_rgb is None else np.asarray(linear_rgb, dtype=np.float32)
    return RawImportResult(
        linear_rgb=linear,
        display_srgb=display,
        width=4,
        height=3,
        focal_length_mm=35.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        sensor_source="camera_db",
        camera_make="Atlas",
        camera_model=model,
        lens_model="35mm",
        undistort_applied=True,
        undistort_status="applied",
        source_path=f"C:/RAW/photo_{index}.nef",
        orientation=1,
        metadata_source="raw_exif",
    )


def _plate(raw: RawImportResult, *, registered_from: str = "AtlasLoadRAW") -> AtlasPlateRef:
    return AtlasPlateRef(
        image_path=raw.source_path.replace(".nef", ".exr"),
        preview_b64="data:image/png;base64,photographed",
        colorspace="ACEScg",
        bit_depth="16f",
        role="source",
        is_proxy=False,
        metadata={
            "registered_from": registered_from,
            "raw_source": raw.source_path,
        },
    )


def _call_args():
    raw_1, raw_2, raw_3 = _raw(1), _raw(2), _raw(3)
    return {
        "image_1": _image_from_raw(raw_1),
        "image_2": _image_from_raw(raw_2),
        "image_3": None,
        "raw_meta_1": raw_1,
        "raw_meta_2": raw_2,
        "raw_meta_3": None,
        "plate_ref_1": _plate(raw_1),
        "plate_ref_2": _plate(raw_2),
        "plate_ref_3": None,
        "capture_mode": "auto",
        "camera_height_m": 0.0,
        "match_quality": "balanced",
        "seed": 0,
        "_raw_3": raw_3,
    }


def _node_class():
    return NODE_CLASS_MAPPINGS["AtlasMultiViewSolve"]


def test_node_contract_and_widget_order():
    cls = _node_class()
    assert NODE_DISPLAY_NAME_MAPPINGS["AtlasMultiViewSolve"] == "Atlas Multi-View RAW Solve 📷📷"
    assert cls.RETURN_TYPES == ("ATLAS_SOLVE", "STRING", "STRING", "IMAGE")
    assert cls.RETURN_NAMES == ("solve", "report", "registration_json", "match_overlays")
    spec = cls.INPUT_TYPES()
    assert list(spec["required"]) == ["image_1", "image_2"]
    assert list(spec["optional"])[:7] == [
        "image_3", "raw_meta_1", "raw_meta_2", "raw_meta_3",
        "plate_ref_1", "plate_ref_2", "plate_ref_3",
    ]
    assert all(spec["optional"][name][1]["forceInput"] for name in list(spec["optional"])[:7])
    widgets = [
        key for key, value in spec["optional"].items()
        if not value[1].get("forceInput", False)
    ]
    assert widgets == [
        "capture_mode", "camera_height_m", "match_quality", "seed",
        "learned_anchor_fallback", "baseline_m", "learned_scale_fallback",
    ]


def test_node_fingerprint_changes_when_any_link_or_widget_changes():
    cls = _node_class()
    args = _call_args()
    args.pop("_raw_3")
    first = cls.IS_CHANGED(**args)
    assert cls.IS_CHANGED(**args) == first

    changed_image_1 = args["raw_meta_1"].display_srgb.copy()
    changed_image_1[0, 0, 0] = 0.11
    changed_raw_1 = replace(args["raw_meta_1"], display_srgb=changed_image_1)
    changed_image_2 = args["raw_meta_2"].display_srgb.copy()
    changed_image_2[0, 0, 0] = 0.21
    changed_raw_2 = replace(args["raw_meta_2"], display_srgb=changed_image_2)
    changed = [
        cls.IS_CHANGED(**dict(
            args, image_1=_image_from_raw(changed_raw_1), raw_meta_1=changed_raw_1,
        )),
        cls.IS_CHANGED(**dict(
            args, image_2=_image_from_raw(changed_raw_2), raw_meta_2=changed_raw_2,
        )),
    ]

    raw_3 = _raw(3)
    third = dict(args, image_3=_image_from_raw(raw_3), raw_meta_3=raw_3, plate_ref_3=_plate(raw_3))
    changed.append(cls.IS_CHANGED(**third))
    changed.append(cls.IS_CHANGED(**dict(args, raw_meta_1=_raw(1, model="Changed body"))))
    changed.append(cls.IS_CHANGED(**dict(args, plate_ref_2=AtlasPlateRef(
        image_path="C:/RAW/other.exr", preview_b64="data:image/png;base64,photographed",
        role="source", is_proxy=False,
        metadata={"registered_from": "AtlasLoadRAW", "raw_source": "C:/RAW/photo_2.nef"},
    ))))
    changed.append(cls.IS_CHANGED(**dict(
        args,
        image_1=args["image_2"], image_2=args["image_1"],
        raw_meta_1=args["raw_meta_2"], raw_meta_2=args["raw_meta_1"],
        plate_ref_1=args["plate_ref_2"], plate_ref_2=args["plate_ref_1"],
    )))
    for name, value in (("capture_mode", "rotation_only"), ("camera_height_m", 1.65),
                        ("match_quality", "conservative"), ("seed", 9),
                        ("learned_anchor_fallback", True), ("baseline_m", 0.8),
                        ("learned_scale_fallback", True)):
        changed.append(cls.IS_CHANGED(**dict(args, **{name: value})))

    assert all(value != first for value in changed)
    assert len(set(changed)) == len(changed)


def test_node_cache_tracks_raw_display_binding_but_not_solver_irrelevant_linear_pixels():
    cls = _node_class()
    args = _call_args()
    args.pop("_raw_3")
    first = cls.IS_CHANGED(**args)

    changed_display = args["raw_meta_1"].display_srgb.copy()
    changed_display[0, 0, 0] = 0.77
    changed_raw = replace(args["raw_meta_1"], display_srgb=changed_display)
    mismatched = cls.IS_CHANGED(**dict(args, raw_meta_1=changed_raw))
    accepted = cls.IS_CHANGED(**dict(
        args, raw_meta_1=changed_raw, image_1=_image_from_raw(changed_raw),
    ))
    assert mismatched != first
    assert accepted != first

    changed_linear = args["raw_meta_1"].linear_rgb.copy()
    changed_linear[0, 0, 0] = 0.77
    linear_only = replace(args["raw_meta_1"], linear_rgb=changed_linear)
    assert cls.IS_CHANGED(**dict(args, raw_meta_1=linear_only)) == first


def test_node_rejects_multi_photo_image_batches():
    args = _call_args()
    args.pop("_raw_3")
    args["image_1"] = _image(batch=2)
    with pytest.raises(RuntimeError, match="image_1 must contain exactly one photograph \\(batch size 1\\); got 2"):
        _node_class()().solve(**args)


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (_Tensor(np.zeros((1, 3, 4), dtype=np.float32)), "must be a BHWC IMAGE tensor"),
        (_Tensor(np.zeros((1, 3, 4, 5), dtype=np.float32)), "must have exactly 3 channels"),
        (_Tensor(np.zeros((1, 3, 4, 3), dtype=np.uint8)), "must contain floating-point values"),
    ],
    ids=("rank", "nchw", "integer"),
)
def test_node_rejects_non_bhwc_or_nonfloating_image_inputs(image, message, monkeypatch):
    module = pytest.importorskip("atlas_camera.comfy.nodes_multiview")
    monkeypatch.setattr(module, "solve_multiview", lambda *_: pytest.fail("solver must not run"))
    args = _call_args()
    args.pop("_raw_3")
    args["image_1"] = image
    with pytest.raises(RuntimeError, match=message):
        _node_class()().solve(**args)


@pytest.mark.parametrize("missing", ["raw_meta_1", "raw_meta_2", "plate_ref_1", "plate_ref_2"])
def test_node_requires_trusted_raw_metadata_and_photographed_plate_for_every_photo(missing):
    args = _call_args()
    args.pop("_raw_3")
    args[missing] = None
    with pytest.raises(RuntimeError, match="AtlasMultiViewSolve: .*photographed RAW frame"):
        _node_class()().solve(**args)


def test_node_rejects_generated_projection_source_as_registration_frame():
    args = _call_args()
    args.pop("_raw_3")
    args["plate_ref_1"] = _plate(args["raw_meta_1"], registered_from="AtlasAddPatchView")
    with pytest.raises(RuntimeError, match="generated or proxy projection source"):
        _node_class()().solve(**args)


def test_node_requires_photographed_plate_preview_for_public_output():
    args = _call_args()
    args.pop("_raw_3")
    args["plate_ref_1"].preview_b64 = None
    with pytest.raises(RuntimeError, match="photographed preview"):
        _node_class()().solve(**args)


def test_node_rejects_image_that_does_not_bind_to_trusted_raw_display(monkeypatch):
    module = pytest.importorskip("atlas_camera.comfy.nodes_multiview")
    monkeypatch.setattr(module, "solve_multiview", lambda *_: pytest.fail("solver must not run"))
    args = _call_args()
    args.pop("_raw_3")
    args["image_1"] = _image(0.99)
    with pytest.raises(RuntimeError, match="pixels do not match trusted RAW display_srgb"):
        _node_class()().solve(**args)


def test_node_requires_complete_third_photo_before_solver_runs(monkeypatch):
    module = pytest.importorskip("atlas_camera.comfy.nodes_multiview")
    monkeypatch.setattr(module, "solve_multiview", lambda *_: pytest.fail("solver must not run"))
    args = _call_args()
    args["image_3"] = _image_from_raw(args["_raw_3"])
    args["raw_meta_3"] = None
    args["plate_ref_3"] = None
    args.pop("_raw_3")
    with pytest.raises(RuntimeError, match="image_3 requires a complete photographed RAW frame"):
        _node_class()().solve(**args)


def test_node_rejects_unphotographed_third_photo_before_solver_runs(monkeypatch):
    module = pytest.importorskip("atlas_camera.comfy.nodes_multiview")
    monkeypatch.setattr(module, "solve_multiview", lambda *_: pytest.fail("solver must not run"))
    args = _call_args()
    raw_3 = args.pop("_raw_3")
    args.update(
        image_3=_image_from_raw(raw_3),
        raw_meta_3=raw_3,
        plate_ref_3=_plate(raw_3, registered_from="AtlasAddPatchView"),
    )
    with pytest.raises(RuntimeError, match="generated or proxy projection source"):
        _node_class()().solve(**args)


def test_node_is_thin_orchestrator_and_batches_overlays(monkeypatch):
    module = pytest.importorskip("atlas_camera.comfy.nodes_multiview")
    captured = {}
    solve = object()
    diagnostics = RegistrationDiagnostics("rotation_only", "stable panorama", warnings=["no scale"])
    outcome = RegistrationOutcome(
        solve=solve,
        diagnostics=diagnostics,
        overlays=(
            np.full((2, 5, 3), 0.25, dtype=np.float64),
            np.full((2, 5, 3), 191, dtype=np.uint8),
        ),
    )

    def fake_solve(frames, settings):
        captured["frames"] = frames
        captured["settings"] = settings
        return outcome

    monkeypatch.setattr(module, "solve_multiview", fake_solve)
    monkeypatch.setattr(module, "_require_torch", lambda: _Torch)
    args = _call_args()
    args.pop("_raw_3")
    result = _node_class()().solve(**args)

    assert result[0] is solve
    assert result[1] == "rotation_only: stable panorama"
    assert result[2] == json.dumps(diagnostics.to_dict(), sort_keys=True)
    expected_overlays = np.stack((
        outcome.overlays[0].astype(np.float32),
        outcome.overlays[1].astype(np.float32) / 255.0,
    ))
    np.testing.assert_array_equal(result[3].numpy(), expected_overlays)
    assert [frame.label for frame in captured["frames"]] == ["photo_1", "photo_2"]
    assert captured["frames"][0].image.dtype == np.float32
    assert captured["frames"][0].image.flags.c_contiguous
    assert captured["settings"].capture_mode == "auto"
    assert captured["settings"].camera_height_m == 0.0
    assert captured["settings"].match_quality == "balanced"
    assert captured["settings"].seed == 0


def test_node_passes_three_photographs_to_solver_in_authoritative_order(monkeypatch):
    module = pytest.importorskip("atlas_camera.comfy.nodes_multiview")
    captured = {}
    diagnostics = RegistrationDiagnostics("rotation_only", "stable panorama")
    outcome = RegistrationOutcome(solve=object(), diagnostics=diagnostics, overlays=())

    def fake_solve(frames, settings):
        captured["frames"] = frames
        return outcome

    monkeypatch.setattr(module, "solve_multiview", fake_solve)
    monkeypatch.setattr(module, "_require_torch", lambda: _Torch)
    args = _call_args()
    raw_3 = args.pop("_raw_3")
    args.update(
        image_3=_image_from_raw(raw_3), raw_meta_3=raw_3, plate_ref_3=_plate(raw_3),
    )
    _node_class()().solve(**args)

    assert [frame.label for frame in captured["frames"]] == ["photo_1", "photo_2", "photo_3"]
    np.testing.assert_array_equal(captured["frames"][0].image, args["raw_meta_1"].display_srgb)
    np.testing.assert_array_equal(captured["frames"][1].image, args["raw_meta_2"].display_srgb)
    np.testing.assert_array_equal(captured["frames"][2].image, raw_3.display_srgb)


def test_node_raises_exact_sorted_diagnostics_when_registration_fails(monkeypatch, tmp_path):
    module = pytest.importorskip("atlas_camera.comfy.nodes_multiview")
    monkeypatch.chdir(tmp_path)
    diagnostics = RegistrationDiagnostics(
        "insufficient_overlap", "only 12 matches",
        scale={"z": 2, "a": 1},
    )
    overlay = np.full((4, 6, 3), 0.5, dtype=np.float32)
    monkeypatch.setattr(
        module, "solve_multiview",
        lambda *_: RegistrationOutcome(None, diagnostics, overlays=(overlay,)),
    )
    args = _call_args()
    args.pop("_raw_3")
    details = diagnostics.to_dict()
    debug_path = tmp_path / "atlas_debug" / "multiview_failure.json"
    expected = (
        "AtlasMultiViewSolve [insufficient_overlap]: only 12 matches\n"
        f"registration diagnostics: {json.dumps(details, sort_keys=True)}\n"
        f"failure diagnostics and overlays written to: {debug_path}"
    )
    with pytest.raises(RuntimeError, match="^" + re.escape(expected) + "$"):
        _node_class()().solve(**args)

    assert json.loads(debug_path.read_text(encoding="utf-8")) == json.loads(
        json.dumps(details, sort_keys=True)
    )


def test_failure_debug_write_errors_never_mask_the_registration_error(monkeypatch, tmp_path):
    module = pytest.importorskip("atlas_camera.comfy.nodes_multiview")
    monkeypatch.chdir(tmp_path)
    diagnostics = RegistrationDiagnostics("degenerate_geometry", "planar scene")
    monkeypatch.setattr(
        module, "solve_multiview",
        lambda *_: RegistrationOutcome(None, diagnostics),
    )
    monkeypatch.setattr(
        module.os, "makedirs",
        lambda *a, **k: (_ for _ in ()).throw(PermissionError("locked")),
    )
    args = _call_args()
    args.pop("_raw_3")
    with pytest.raises(RuntimeError, match=r"AtlasMultiViewSolve \[degenerate_geometry\]: planar scene"):
        _node_class()().solve(**args)
    assert not (tmp_path / "atlas_debug").exists()


def test_facade_import_and_execution_dependency_boundary_in_a_fresh_process():
    script = """
import builtins

original_import = builtins.__import__
missing = {"numpy", "cv2", "torch"}

def blocked_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in missing:
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
import atlas_camera.comfy.nodes as nodes
assert nodes.AtlasMultiViewSolve.__name__ == "AtlasMultiViewSolve"
try:
    nodes.AtlasMultiViewSolve().solve(None, None)
except RuntimeError as exc:
    assert str(exc) == (
        "AtlasMultiViewSolve requires NumPy. Install with: pip install -e .[vision]"
    )
else:
    raise AssertionError("execution must report the missing optional dependency")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
