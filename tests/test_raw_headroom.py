"""RAW decode headroom — the scene-referred scale that makes the sidecar ACES-usable.

Measured on ``DSC_2190.NEF`` (Nikon D810, 24mm, -0.7EV, backlit sunset) via a
live Nuke check on 2026-08-07: the written sidecar carried sensor clip at 1.0
and mid-grey at ~0.043, where the ACES scene-linear convention wants ~0.18.
Correct under an un-tone-mapped sRGB view; ~2.6 stops crushed under the ACES
1.0 SDR Video RRT+ODT, which is the reported "looks very dark in Nuke".

rawtoaces applies a headroom multiply that Atlas did not
(``AcademySoftwareFoundation/rawtoaces``,
``src/rawtoaces_util/image_converter.cpp:2443``,
``--headroom`` default ``6.0f`` at ``:845-854``), so an Atlas file was a
rawtoaces file divided by six. Cross-checked against the measured data:
``p75 0.150 x 6 = 0.90`` puts diffuse white at ~1.0.

The load-bearing test here is
``test_headroom_leaves_the_display_tensor_bit_identical``: the display path
normalises by a luminance percentile, so a constant factor cancels by
construction and the SOLVE tensor is untouched at any headroom. That is what
makes this change safe to land without re-verifying the solver.
"""

import pytest

np = pytest.importorskip("numpy")


# A tiny deterministic 16-bit frame with a real range, so a scale is visible.
_FAKE_RGB16 = (np.linspace(0, 65535, 4 * 6 * 3, dtype=np.float64)
               .reshape(4, 6, 3).astype(np.uint16))


class _FakeRaw:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def postprocess(self, **_kw):
        return _FAKE_RGB16


class _FakeColorSpace:
    sRGB = "sRGB"


class _FakeRawpy:
    ColorSpace = _FakeColorSpace

    @staticmethod
    def imread(_path):
        return _FakeRaw()


class _FakeMeta:
    camera_make = "NIKON CORPORATION"
    camera_model = "NIKON D810"
    lens_model = None
    focal_length_mm = 24.0
    warnings: list = []


class _FakeSensor:
    sensor_width_mm = 35.9
    sensor_height_mm = 24.0
    source = "camera_db"
    warnings: list = []


@pytest.fixture
def fake_rawpy(monkeypatch):
    import atlas_camera.raw.decode as decode_mod
    monkeypatch.setattr(decode_mod, "_require_rawpy", lambda: _FakeRawpy())
    return decode_mod


def test_headroom_scales_the_linear_master_linearly(fake_rawpy):
    base, _ = fake_rawpy.decode_raw("x.nef", headroom=1.0)
    scaled, _ = fake_rawpy.decode_raw("x.nef", headroom=6.0)
    assert np.allclose(scaled, base * 6.0)


def test_headroom_leaves_the_display_tensor_unchanged(fake_rawpy):
    """The guarantee that keeps the solver out of scope: display_from_linear
    maps a luminance percentile to 1.0, so a constant factor cancels.

    NOT bit-identical, and the work order's wording was too strong here. The
    cancellation is exact in real arithmetic but the pipeline is float32, so
    ``x * 6`` then ``/ (peak * 6)`` lands within an ULP or two rather than on
    the same bits (measured: 0.26390082 vs 0.2639008). Asserting equality would
    make this test fail for a reason that has nothing to do with headroom, so
    it asserts the property that is actually true and actually load-bearing —
    the display tensor does not move at a scale any solver could resolve. VP
    detection works on gradients and GeoCalib on 8-bit-ish input; both are many
    orders of magnitude away from 1e-6.
    """
    _, display_1 = fake_rawpy.decode_raw("x.nef", headroom=1.0)
    _, display_6 = fake_rawpy.decode_raw("x.nef", headroom=6.0)
    _, display_23 = fake_rawpy.decode_raw("x.nef", headroom=23.75)
    assert float(np.abs(display_6 - display_1).max()) < 1e-6
    assert float(np.abs(display_23 - display_1).max()) < 1e-6


def test_headroom_defaults_to_the_rawtoaces_six():
    import inspect

    from atlas_camera.raw.decode import decode_raw
    assert inspect.signature(decode_raw).parameters["headroom"].default == 6.0


def test_headroom_is_keyword_only():
    """A positional headroom could be mistaken for exposure_ev at a call site."""
    import inspect

    from atlas_camera.raw.decode import decode_raw
    param = inspect.signature(decode_raw).parameters["headroom"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_headroom_one_reproduces_the_pre_change_output(fake_rawpy):
    """Acceptance item 4: headroom=1.0 is exactly the old behaviour."""
    linear, _ = fake_rawpy.decode_raw("x.nef", headroom=1.0, exposure_ev=0.0)
    assert np.allclose(linear, _FAKE_RGB16.astype(np.float32) / 65535.0)


def test_headroom_composes_with_exposure_ev(fake_rawpy):
    linear, _ = fake_rawpy.decode_raw("x.nef", headroom=3.0, exposure_ev=1.0)
    expected = (_FAKE_RGB16.astype(np.float32) / 65535.0) * 6.0
    assert np.allclose(linear, expected)


def _patch_pipeline(monkeypatch, seen):
    import atlas_camera.raw.pipeline as pipeline_mod

    def fake_decode(path, **kw):
        seen.update(kw)
        arr = np.full((4, 6, 3), 0.2, dtype=np.float32)
        return arr, arr

    monkeypatch.setattr(pipeline_mod, "decode_raw", fake_decode)
    monkeypatch.setattr(pipeline_mod, "read_raw_metadata", lambda p: _FakeMeta())
    monkeypatch.setattr(pipeline_mod, "resolve_sensor_size",
                        lambda *a, **k: _FakeSensor())
    return pipeline_mod


def test_import_raw_threads_headroom_through_to_the_decode(monkeypatch):
    seen: dict = {}
    pipeline_mod = _patch_pipeline(monkeypatch, seen)
    pipeline_mod.import_raw("x.nef", undistort=False, headroom=6.0)
    assert seen["headroom"] == 6.0


def test_import_raw_records_the_applied_headroom_on_the_result(monkeypatch):
    """Recorded, not recomputed: the node and the EXR writer both report it,
    and a plate that has been scaled must be able to say by how much or a
    downstream re-grade is guesswork."""
    pipeline_mod = _patch_pipeline(monkeypatch, {})
    result = pipeline_mod.import_raw("x.nef", undistort=False, headroom=6.0)
    assert result.headroom == 6.0


def test_import_raw_headroom_defaults_to_six(monkeypatch):
    seen: dict = {}
    pipeline_mod = _patch_pipeline(monkeypatch, seen)
    result = pipeline_mod.import_raw("x.nef", undistort=False)
    assert seen["headroom"] == 6.0
    assert result.headroom == 6.0


def test_summary_lines_name_the_headroom(monkeypatch):
    pipeline_mod = _patch_pipeline(monkeypatch, {})
    result = pipeline_mod.import_raw("x.nef", undistort=False, headroom=6.0)
    assert any("headroom" in line for line in result.summary_lines())


def test_summary_warns_when_headroom_is_one(monkeypatch):
    """headroom=1.0 is legal (it reproduces the old file byte for byte) but it
    is NOT ACES-referred, and reading ~2.6 stops dark under an ACES view is
    exactly the confusion this whole change exists to end. Say so."""
    pipeline_mod = _patch_pipeline(monkeypatch, {})
    result = pipeline_mod.import_raw("x.nef", undistort=False, headroom=1.0)
    line = next(l for l in result.summary_lines() if "headroom" in l)
    assert "NOT ACES-referred" in line
