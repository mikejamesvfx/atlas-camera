"""Public ComfyUI contract for deterministic photographed multi-view solves."""

from __future__ import annotations

import json
import re

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


def _raw(index: int, *, model: str = "Atlas RAW") -> RawImportResult:
    pixels = np.zeros((3, 4, 3), dtype=np.float32)
    return RawImportResult(
        linear_rgb=pixels,
        display_srgb=pixels,
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
        "image_1": _image(0.1),
        "image_2": _image(0.2),
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
    assert widgets == ["capture_mode", "camera_height_m", "match_quality", "seed"]


def test_node_fingerprint_changes_when_any_link_or_widget_changes():
    cls = _node_class()
    args = _call_args()
    args.pop("_raw_3")
    first = cls.IS_CHANGED(**args)
    assert cls.IS_CHANGED(**args) == first

    changed = []
    for name, value in (("image_1", _image(0.11)), ("image_2", _image(0.21))):
        candidate = dict(args, **{name: value})
        changed.append(cls.IS_CHANGED(**candidate))

    raw_3 = _raw(3)
    third = dict(args, image_3=_image(0.3), raw_meta_3=raw_3, plate_ref_3=_plate(raw_3))
    changed.append(cls.IS_CHANGED(**third))
    changed.append(cls.IS_CHANGED(**dict(args, raw_meta_1=_raw(1, model="Changed body"))))
    changed.append(cls.IS_CHANGED(**dict(args, plate_ref_2=AtlasPlateRef(
        image_path="C:/RAW/other.exr", role="source", is_proxy=False,
        metadata={"registered_from": "AtlasLoadRAW", "raw_source": "C:/RAW/photo_2.nef"},
    ))))
    changed.append(cls.IS_CHANGED(**dict(
        args,
        image_1=args["image_2"], image_2=args["image_1"],
        raw_meta_1=args["raw_meta_2"], raw_meta_2=args["raw_meta_1"],
        plate_ref_1=args["plate_ref_2"], plate_ref_2=args["plate_ref_1"],
    )))
    for name, value in (("capture_mode", "rotation_only"), ("camera_height_m", 1.65),
                        ("match_quality", "conservative"), ("seed", 9)):
        changed.append(cls.IS_CHANGED(**dict(args, **{name: value})))

    assert all(value != first for value in changed)
    assert len(set(changed)) == len(changed)


def test_node_rejects_multi_photo_image_batches():
    args = _call_args()
    args.pop("_raw_3")
    args["image_1"] = _image(batch=2)
    with pytest.raises(RuntimeError, match="image_1 must contain exactly one photograph \\(batch size 1\\); got 2"):
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


def test_node_requires_complete_third_photo_before_solver_runs(monkeypatch):
    module = pytest.importorskip("atlas_camera.comfy.nodes_multiview")
    monkeypatch.setattr(module, "solve_multiview", lambda *_: pytest.fail("solver must not run"))
    args = _call_args()
    args["image_3"] = _image(0.3)
    args["raw_meta_3"] = None
    args["plate_ref_3"] = None
    args.pop("_raw_3")
    with pytest.raises(RuntimeError, match="image_3 requires a complete photographed RAW frame"):
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
            np.full((2, 5, 3), 0.75, dtype=np.float64),
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
    np.testing.assert_array_equal(result[3].numpy(), np.stack(outcome.overlays).astype(np.float32))
    assert [frame.label for frame in captured["frames"]] == ["photo_1", "photo_2"]
    assert captured["frames"][0].image.dtype == np.float32
    assert captured["frames"][0].image.flags.c_contiguous
    assert captured["settings"].capture_mode == "auto"
    assert captured["settings"].camera_height_m == 0.0
    assert captured["settings"].match_quality == "balanced"
    assert captured["settings"].seed == 0


def test_node_raises_exact_sorted_diagnostics_when_registration_fails(monkeypatch):
    module = pytest.importorskip("atlas_camera.comfy.nodes_multiview")
    diagnostics = RegistrationDiagnostics(
        "insufficient_overlap", "only 12 matches",
        scale={"z": 2, "a": 1},
    )
    monkeypatch.setattr(
        module, "solve_multiview",
        lambda *_: RegistrationOutcome(None, diagnostics),
    )
    args = _call_args()
    args.pop("_raw_3")
    details = diagnostics.to_dict()
    expected = (
        "AtlasMultiViewSolve [insufficient_overlap]: only 12 matches\n"
        f"registration diagnostics: {json.dumps(details, sort_keys=True)}"
    )
    with pytest.raises(RuntimeError, match="^" + re.escape(expected) + "$"):
        _node_class()().solve(**args)
