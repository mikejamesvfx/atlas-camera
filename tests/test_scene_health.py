"""scale_health() — scale provenance -> safe-to-export mapping (pure Python)."""

import pytest

from atlas_camera.core.scene_health import (
    SCALE_STATUS_ASSUMED,
    SCALE_STATUS_MANUAL,
    SCALE_STATUS_MEASURED,
    SCALE_STATUS_UNKNOWN,
    ScaleHealth,
    scale_health,
)
from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasSolve,
    LatentCamera,
    LatentComponent,
)


def _solve(height=1.6, scale_source=None, **debug):
    intr = AtlasIntrinsics(image_width=256, image_height=256,
                           fx_px=250.0, fy_px=250.0, cx_px=128.0, cy_px=128.0)
    extr = AtlasExtrinsics(camera_position=(0.0, height, 0.0))
    s = AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))
    if scale_source is not None:
        s.debug_metadata["scale_source"] = scale_source
    s.debug_metadata.update(debug)
    return s


def test_reference_object_is_measured_and_safe():
    s = _solve(scale_source="reference_object",
               reference_scale={"camera_height_m": 17.5, "confidence": 0.8,
                                "adopted": True, "references": []})
    sh = scale_health(s)
    assert sh.status == SCALE_STATUS_MEASURED
    assert sh.safe_to_export is True
    assert sh.confidence == pytest.approx(0.8)
    assert sh.camera_height_m == pytest.approx(1.6)


def test_depth_ground_plane_metric_is_safe():
    s = _solve(scale_source="depth_ground_plane")
    s.depth = LatentComponent(value={"is_metric": True}, confidence=0.6)
    sh = scale_health(s)
    assert sh.status == SCALE_STATUS_MEASURED
    assert sh.safe_to_export is True
    assert sh.confidence == pytest.approx(0.6)


def test_depth_ground_plane_relative_depth_is_unsafe():
    s = _solve(scale_source="depth_ground_plane")
    s.depth = LatentComponent(value={"is_metric": False}, confidence=0.6)
    sh = scale_health(s)
    assert sh.status == SCALE_STATUS_MEASURED
    assert sh.safe_to_export is False
    assert "up-to-scale" in sh.detail


def test_manual_override_is_safe():
    sh = scale_health(_solve(height=45.0, scale_source="manual_override"))
    assert sh.status == SCALE_STATUS_MANUAL
    assert sh.safe_to_export is True
    assert sh.confidence == pytest.approx(1.0)


def test_assumed_default_is_unsafe():
    sh = scale_health(_solve(scale_source="assumed_default"))
    assert sh.status == SCALE_STATUS_ASSUMED
    assert sh.safe_to_export is False
    assert "NOT measured" in sh.detail
    assert "AtlasScaleOverride" in sh.detail


def test_missing_provenance_is_unknown_and_unsafe():
    sh = scale_health(_solve())
    assert sh.status == SCALE_STATUS_UNKNOWN
    assert sh.safe_to_export is False
    assert sh.scale_source is None


def test_never_raises_on_garbage():
    # Non-dict reference_scale degrades to confidence 0.0, no raise.
    s = _solve(scale_source="reference_object", reference_scale="not a dict")
    sh = scale_health(s)
    assert sh.status == SCALE_STATUS_MEASURED and sh.confidence == 0.0

    class Weird:  # noqa: B903 — missing every attribute
        pass
    sh = scale_health(Weird())
    assert sh.status == SCALE_STATUS_UNKNOWN
    assert sh.safe_to_export is False


def test_round_trip():
    sh = scale_health(_solve(scale_source="assumed_default"))
    again = ScaleHealth.from_dict(sh.to_dict())
    assert again == sh
    assert ScaleHealth.from_dict(None) is None


def test_solve_json_carries_scale_health():
    s = _solve(scale_source="assumed_default")
    data = s.to_dict()
    assert data["scale_health"]["status"] == SCALE_STATUS_ASSUMED
    assert data["scale_health"]["safe_to_export"] is False
    # Round-trips, and the derived key never breaks from_dict.
    again = AtlasSolve.from_dict(data)
    assert again.debug_metadata["scale_source"] == "assumed_default"


def test_old_json_without_scale_health_loads():
    data = _solve(scale_source="manual_override").to_dict()
    data.pop("scale_health")
    again = AtlasSolve.from_dict(data)
    assert scale_health(again).status == SCALE_STATUS_MANUAL
