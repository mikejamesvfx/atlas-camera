"""Tests for wiring VLM scale cues into solver scale references.

The cue -> reference-spec conversion is pure dataclasses (no model/server), so it
is fully testable. It then feeds resolve_reference_scale (single-view geometry).
"""

import numpy as np
import pytest

from atlas_camera.inference.multimodal_helper import (
    MultimodalSceneObservation,
    SceneScaleCue,
    scale_references_from_observation,
)
from atlas_camera.core.solver import resolve_reference_scale


def _obs(cues):
    return MultimodalSceneObservation(image_path="img.png", summary="test", scale_cues=cues)


def test_maps_suggested_reference_id_and_label():
    obs = _obs([
        SceneScaleCue(label="person", confidence=0.9, bbox_px=(470, 250, 560, 820),
                      suggested_reference_ids=["person_175cm"]),
        SceneScaleCue(label="door", confidence=0.7, bbox_px=(100, 200, 180, 600),
                      suggested_reference_ids=[]),                       # resolved by label
    ])
    refs = scale_references_from_observation(obs)
    assert len(refs) == 2
    assert refs[0]["reference_id"] == "person_175cm"
    assert refs[0]["bbox_px"] == [470.0, 250.0, 560.0, 820.0]
    assert refs[0]["source"] == "vlm_scale_cue"
    assert refs[1]["reference_id"] == "door_210cm"


def test_skips_cues_without_bbox_or_resolvable_height():
    obs = _obs([
        SceneScaleCue(label="person", confidence=0.9, bbox_px=None,
                      suggested_reference_ids=["person_175cm"]),        # no bbox
        SceneScaleCue(label="mysterious blob", confidence=0.8, bbox_px=(0, 0, 10, 10),
                      suggested_reference_ids=[]),                       # unresolvable
    ])
    assert scale_references_from_observation(obs) == []


def test_min_confidence_filter():
    obs = _obs([
        SceneScaleCue(label="person", confidence=0.9, bbox_px=(470, 250, 560, 820),
                      suggested_reference_ids=["person_175cm"]),
        SceneScaleCue(label="car", confidence=0.4, bbox_px=(600, 400, 900, 620),
                      suggested_reference_ids=["sedan_car"]),
    ])
    refs = scale_references_from_observation(obs, min_confidence=0.75)
    assert len(refs) == 1
    assert refs[0]["reference_id"] == "person_175cm"


def test_specs_feed_resolve_reference_scale():
    obs = _obs([
        SceneScaleCue(label="person", confidence=0.9, bbox_px=(470, 250, 560, 820),
                      suggested_reference_ids=["person_175cm"]),
    ])
    refs = scale_references_from_observation(obs)
    result = resolve_reference_scale(
        refs, rotation=np.eye(3), fx=800, fy=800, cx=512, cy=512
    )
    # A person standing in the lower frame with a level camera yields a plausible height.
    assert result["camera_height"] is not None
    assert result["camera_height"] > 0
