"""AtlasCardMask — card alpha -> object_mask/depth-band bridge.

The node is pure decode + arithmetic over a solve's card ProjectionSources:
no plate read, no depth read, no ComfyUI import. Sources are stubbed with
plain namespaces because the node reads only ``name`` / ``metadata`` /
``mask_b64`` — the same duck-typing the real schema objects satisfy.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from atlas_camera.comfy.nodes_inpaint import AtlasCardMask
from atlas_camera.core.matte_codec import encode_matte_png


def _card(name="CARD_CAR", concept="car", distance=14.51, alpha=None):
    if alpha is None:
        alpha = np.zeros((8, 12), dtype=np.float32)
        alpha[2:6, 3:9] = 1.0
    return SimpleNamespace(
        name=name,
        mask_b64=encode_matte_png(alpha),
        metadata={
            "evidence_type": "observed_card",
            "concept": concept,
            "camera_distance_m": distance,
        },
    )


def _solve(*sources):
    return SimpleNamespace(projection_sources=list(sources))


def test_schema_pin():
    # Saved-workflow contract: names, order, and types are append-only.
    assert AtlasCardMask.RETURN_TYPES == ("MASK", "FLOAT", "FLOAT", "STRING")
    assert AtlasCardMask.RETURN_NAMES == ("mask", "near_m", "far_m", "report")
    assert AtlasCardMask.FUNCTION == "extract"
    assert AtlasCardMask.CATEGORY == "Atlas/11 · Evidence Plate"
    spec = AtlasCardMask.INPUT_TYPES()
    assert list(spec["required"]) == ["solve", "card"]
    assert list(spec["optional"]) == ["band_margin_m", "invert"]
    assert spec["required"]["card"][0] == "STRING"  # dynamic names: never a combo


def test_mask_roundtrips_exactly():
    alpha = np.zeros((8, 12), dtype=np.float32)
    alpha[2:6, 3:9] = 1.0
    mask, near, far, report = AtlasCardMask().extract(
        _solve(_card(alpha=alpha)), "CARD_CAR")
    assert mask.shape == (1, 8, 12)
    assert torch.equal(mask[0], torch.from_numpy(alpha))
    assert "CARD_CAR" in report and '"coverage_px": 24' in report


def test_band_arithmetic_and_clamp():
    _, near, far, _ = AtlasCardMask().extract(
        _solve(_card(distance=14.51)), "CARD_CAR", band_margin_m=3.0)
    assert near == pytest.approx(11.51)
    assert far == pytest.approx(17.51)
    # A margin wider than the distance clamps at 0, never negative.
    _, near, _, _ = AtlasCardMask().extract(
        _solve(_card(distance=1.0)), "CARD_CAR", band_margin_m=5.0)
    assert near == 0.0


def test_concept_lookup():
    mask, *_ = AtlasCardMask().extract(_solve(_card()), "car")
    assert float(mask.sum()) > 0


def test_invert():
    alpha = np.zeros((4, 4), dtype=np.float32)
    alpha[0, 0] = 1.0
    mask, *_ = AtlasCardMask().extract(
        _solve(_card(alpha=alpha)), "CARD_CAR", invert=True)
    assert float(mask[0, 0, 0]) == pytest.approx(0.0)
    assert float(mask[0, 3, 3]) == pytest.approx(1.0)


def test_unknown_card_errors_loudly_listing_cards():
    solve = _solve(_card(), _card(name="CARD_DINER_SIGN", concept="diner sign"))
    with pytest.raises(ValueError) as err:
        AtlasCardMask().extract(solve, "CARD_TREE")
    msg = str(err.value)
    assert "CARD_TREE" in msg
    assert "CARD_CAR (car)" in msg and "CARD_DINER_SIGN (diner sign)" in msg


def test_non_card_sources_are_invisible():
    # A plane source (no observed_card evidence) must neither match nor be
    # offered as a candidate.
    plane = SimpleNamespace(name="projection_plane_02", mask_b64="",
                            metadata={"evidence_type": "ransac_plane"})
    with pytest.raises(ValueError) as err:
        AtlasCardMask().extract(_solve(plane), "projection_plane_02")
    assert "Available cards: none" in str(err.value)


def test_empty_card_name_rejected():
    with pytest.raises(ValueError):
        AtlasCardMask().extract(_solve(_card()), "  ")
