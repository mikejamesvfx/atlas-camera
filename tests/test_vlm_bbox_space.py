"""A VLM bbox is in SENT-image pixels, not plate pixels (2026-08-08).

`_image_data_url` downscales the long edge to `_VLM_MAX_IMAGE_SIDE` (1280)
before encoding, so every bbox a model returns is in that reduced space. The
node hands `scale_references_from_observation` the FULL plate size, so the two
were compared and emitted without ever being reconciled.

Two things broke, both silently, on every plate wider than 1280 px:

  * `_facade_is_truncated` never fired — a 1280-space box always looks far
    from the edges of a 7380x4928 frame, which is precisely the check that
    exists to stop a cut-off facade reporting `scale_source=reference_object`
    at high confidence (its own docstring records the 34 m vs 63.7 m miss).
  * the emitted `bbox_px` was ~5.8x too small, so any consumer applying it to
    the real plate measures the wrong pixels.

The module already refuses `bbox_px_relative` for exactly this reason — "it
arrives with pixel-looking values in an unstated coordinate space, and
guessing that space would silently misplace the anchor". These tests hold
`bbox_px` to the same standard.
"""
import pytest

from atlas_camera.inference.multimodal_helper import (
    _VLM_MAX_IMAGE_SIDE,
    MultimodalSceneObservation,
    SceneScaleCue,
    scale_references_from_observation,
)

PLATE = (7380, 4928)                      # a D810 frame
SENT_W = _VLM_MAX_IMAGE_SIDE              # 1280
SENT_H = round(PLATE[1] * SENT_W / PLATE[0])
FACTOR = PLATE[0] / SENT_W                # 5.766


def _observation(bbox, *, storeys=5, conf=0.9):
    cue = SceneScaleCue(
        label="pre-war walk-up",
        confidence=conf,
        bbox_px=bbox,
        suggested_reference_ids=["building_story_3m"],
        storey_count=storeys,
    )
    return MultimodalSceneObservation(
        image_path="DSC_2289_skyrise.png",
        summary="street-level urban canyon",
        scale_cues=[cue],
    )


def test_bbox_is_reported_in_plate_pixels_not_sent_pixels():
    """A box drawn on the 1280 px copy must come back scaled to the plate."""
    sent = (370.0, 495.0, 580.0, 750.0)          # measured live from gemma4
    refs = scale_references_from_observation(
        _observation(sent), image_size=PLATE)

    assert refs, "the cue was dropped instead of rescaled"
    got = refs[0]["bbox_px"]
    for value, expected in zip(got, (v * FACTOR for v in sent)):
        assert value == pytest.approx(expected, rel=1e-3), (
            f"bbox still in sent space: {got} — a consumer applying this to "
            f"the {PLATE[0]}x{PLATE[1]} plate measures the wrong pixels")


def test_a_facade_running_off_the_plate_bottom_is_still_rejected():
    """The truncation guard must see the same space the box was drawn in.

    This facade touches the BOTTOM of the image the model was shown, so its
    "base" is where the building leaves frame, not ground contact — the exact
    case the guard exists to refuse. Unscaled, y=SENT_H reads as thousands of
    pixels clear of a 4928 px frame and sails through.
    """
    flush_to_bottom = (370.0, 200.0, 580.0, float(SENT_H))
    obs = _observation(flush_to_bottom)
    refs = scale_references_from_observation(obs, image_size=PLATE)

    assert refs == [], (
        "a bottom-truncated facade was accepted as a scale reference — it "
        "would report a confident but wrong camera height")
    assert any("truncated" in w or "base and roofline" in w
               for w in obs.warnings), obs.warnings


def test_a_fully_visible_facade_still_survives():
    """The guard must not become so eager it refuses good references."""
    inset = (370.0, 120.0, 580.0, SENT_H - 120.0)
    refs = scale_references_from_observation(
        _observation(inset, storeys=6), image_size=PLATE)

    assert len(refs) == 1
    assert refs[0]["storey_count"] == 6
    assert refs[0]["height_m"] == pytest.approx(6 * 3.0)


def test_a_plate_smaller_than_the_cap_is_left_alone():
    """No downscale happened, so no rescale may happen either."""
    small = (1000, 700)
    sent = (100.0, 120.0, 300.0, 500.0)
    refs = scale_references_from_observation(
        _observation(sent), image_size=small)

    assert refs, "cue dropped on a small plate"
    assert refs[0]["bbox_px"] == pytest.approx(list(sent))
