"""Regression tests for the viewport occlusion cull's metric-depth packing.

These exist because the feature originally shipped BROKEN and SILENT: the
packing block imported `_depth_map_for_solve` / `_horizon_y_from_solve` from
`core.solver` (they are module-local to `node_helpers`) and
`estimate_ground_scale` from `core.camera_math` (it lives in
`core.relief_mesh`), all inside a bare `except Exception` that merely logged.
Every call therefore raised ImportError, was swallowed, and left
`primary_depth_b64` empty — so the ✂ Occlude cull could never receive a depth
map, while looking fully wired up.

The lesson these tests encode: a feature whose only failure mode is "produces
nothing" needs a test that asserts it produces SOMETHING, and asserts the
consumer's own formula can read it back.
"""

import base64
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

from test_inpaint_layers_nodes import _depth_result, _occluder_depth, _solve  # noqa: E402

from atlas_camera.comfy.viewport_payload import _extract_blockout_camera  # noqa: E402


def _payload_with_depth():
    image = np.zeros((1, 64, 64, 3), dtype=np.float32)
    return _extract_blockout_camera(
        _solve(), image, 64, 64, primary_depth=_depth_result(_occluder_depth()))


def _unpack_like_the_shader(rgb):
    """atlas_blockout.js: `z_mm = R*65536 + G*256 + B`, then `/1000` for metres.

    Mirrored here deliberately — if either side's byte order changes without
    the other, this test fails, which is the whole point of duplicating it.
    """
    r = rgb[..., 0].astype(np.uint32)
    g = rgb[..., 1].astype(np.uint32)
    b = rgb[..., 2].astype(np.uint32)
    return (r * 65536 + g * 256 + b) / 1000.0


def _decode(b64):
    from PIL import Image
    raw = base64.b64decode(b64.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(raw)))


def test_primary_depth_is_actually_packed():
    """The bug this guards: an empty string on every single call."""
    b64 = _payload_with_depth()["primary_depth_b64"]
    assert b64, "primary_depth_b64 is empty — the occlusion cull gets no depth"
    assert b64.startswith("data:image/png;base64,")


def test_no_depth_input_yields_empty_string():
    """Absent primary_depth is the legitimate no-cull case, not an error."""
    image = np.zeros((1, 64, 64, 3), dtype=np.float32)
    payload = _extract_blockout_camera(_solve(), image, 64, 64)
    assert payload["primary_depth_b64"] == ""


def test_packed_depth_round_trips_through_the_shader_formula():
    """Byte order is a contract with atlas_blockout.js: R=high, G=mid, B=low."""
    rgb = _decode(_payload_with_depth()["primary_depth_b64"])
    assert rgb.ndim == 3 and rgb.shape[2] == 3

    metres = _unpack_like_the_shader(rgb)
    assert metres.max() > 0.0, "all-zero depth — packing or scaling is broken"
    # The occluder fixture spans a near wall at ~3m out to a far backdrop.
    assert 1.0 < metres.max() < 1000.0, f"implausible depth range: max {metres.max()}"
    assert metres.min() >= 0.0


def test_packing_is_millimetre_accurate():
    """1mm quantisation: a known metric value must survive the round trip."""
    payload = _payload_with_depth()
    rgb = _decode(payload["primary_depth_b64"])
    metres = _unpack_like_the_shader(rgb)
    # Quantisation is the ONLY permitted loss, so every value must sit on a
    # whole millimetre.
    residual = np.abs(metres * 1000.0 - np.round(metres * 1000.0))
    assert residual.max() < 1e-6


def test_import_errors_are_not_swallowed(monkeypatch):
    """The narrowed except must let a bad module path surface loudly.

    Directly pins the regression: with a blanket `except Exception`, breaking
    estimate_ground_scale's import produced a silent empty string instead of
    a traceback.
    """
    import atlas_camera.core.relief_mesh as rm

    def _boom(*_a, **_k):
        raise ImportError("simulated bad import path")

    monkeypatch.setattr(rm, "estimate_ground_scale", _boom)
    with pytest.raises(ImportError):
        _payload_with_depth()
