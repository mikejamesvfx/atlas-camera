"""Counting storeys is the scale reference an elevated city plate actually has.

Measured live 2026-08-01 on DSC_2328.NEF (Nikon D810, Manhattan birdseye): the
VLM described the scene correctly — "the high vantage point creates significant
foreshortening and requires precise scale anchoring to the street level" — and
then returned ZERO scale references, so the solve fell back to
`assumed_default` 1.6 m on a camera roughly 19 storeys up.

The cause was not the model. It was never asked. `_LABEL_TO_REFERENCE_ID`
mapped person / door / car / bus / container / hoop, and the prompt asked for
"a known-size object (person, car, door)". From 60 m up those are unreadable
specks. The one thing legible from a birdseye — building storeys — was the one
thing not on the list, despite `building_story_3m` already existing in the
registry and the MCP sky-rise doctrine telling operators to count them by hand.
"""
import json

import pytest

from atlas_camera.inference.multimodal_helper import (
    STOREY_HEIGHT_M,
    MultimodalSceneObservation,
    SceneScaleCue,
    _resolve_reference_id,
    _scene_observation_json_schema,
    _user_prompt,
    scale_references_from_observation,
)

BBOX = (3820.0, 1775.0, 4470.0, 2480.0)   # the doctrine's counted tenement


def _observation(*cues):
    return MultimodalSceneObservation(
        image_path="plate.png", summary="urban", scale_cues=list(cues))


def _cue(**kw):
    base = dict(label="building", confidence=0.9, bbox_px=BBOX)
    base.update(kw)
    return SceneScaleCue(**base)


# ------------------------------------------------------- label resolution

@pytest.mark.parametrize("label", [
    "building", "apartment building", "tenement", "facade", "storey",
    "story", "floor", "tower block", "high-rise",
])
def test_building_words_resolve_to_the_storey_reference(label):
    assert _resolve_reference_id(_cue(label=label)) == "building_story_3m"


def test_an_explicit_suggested_id_still_wins():
    cue = _cue(label="building", suggested_reference_ids=["person_175cm"])
    assert _resolve_reference_id(cue) == "person_175cm"


def test_a_non_building_label_is_untouched():
    assert _resolve_reference_id(_cue(label="car")) == "sedan_car"


# ------------------------------------------------------------- the count

def test_a_counted_storey_cue_becomes_its_full_height():
    """19 storeys is 19 x the storey height, not one storey. Emitting the bare
    reference_id would hand the solver a 3 m building."""
    (spec,) = scale_references_from_observation(_observation(_cue(storey_count=19)))
    assert spec["height_m"] == pytest.approx(19 * STOREY_HEIGHT_M)
    assert spec["storey_count"] == 19
    assert "reference_id" not in spec


def test_the_doctrines_worked_example_reproduces():
    """A 5-storey tenement, the case the sky-rise doctrine recorded."""
    (spec,) = scale_references_from_observation(_observation(_cue(storey_count=5)))
    assert spec["height_m"] == pytest.approx(15.0)


def test_the_storey_height_is_dialable():
    """The registry says 3.0 m; the doctrine's measured prewar tenement implied
    3.5 m. Callers must be able to say which, rather than inherit a guess."""
    (spec,) = scale_references_from_observation(
        _observation(_cue(storey_count=5)), storey_height_m=3.5)
    assert spec["height_m"] == pytest.approx(17.5)


def test_no_count_falls_back_to_the_single_storey_reference():
    """Absent a count, the cue is still a legitimate one-storey reference —
    it must not become a 0 m building."""
    (spec,) = scale_references_from_observation(_observation(_cue()))
    assert spec["reference_id"] == "building_story_3m"
    assert "height_m" not in spec


@pytest.mark.parametrize("bad", [0, -3])
def test_a_nonsense_count_does_not_produce_a_nonsense_height(bad):
    (spec,) = scale_references_from_observation(_observation(_cue(storey_count=bad)))
    assert spec.get("height_m") != 0.0
    assert spec["reference_id"] == "building_story_3m"


def test_a_count_on_a_non_building_cue_is_ignored():
    """A person does not have storeys; a stray count must not rescale them."""
    (spec,) = scale_references_from_observation(
        _observation(_cue(label="person", storey_count=19)))
    assert spec["reference_id"] == "person_175cm"
    assert "height_m" not in spec


# ------------------------------------------------------ facade selection

IMAGE = (1920, 1282)


def test_a_facade_running_off_the_top_of_frame_is_refused():
    """Measured live on the NYC birdseye: the model picked a tower whose
    roofline leaves the frame (bbox y0=0) and counted only its VISIBLE storeys,
    so its "base" was where it disappears behind a nearer building rather than
    ground contact. That solved the camera at 34 m against a hand-measured
    63.7 m. A truncated facade is not a usable reference."""
    obs = _observation(_cue(bbox_px=(405.0, 0.0, 575.0, 550.0), storey_count=15))
    assert scale_references_from_observation(obs, image_size=IMAGE) == []


def test_the_refusal_is_explained_not_silent():
    obs = _observation(_cue(bbox_px=(405.0, 0.0, 575.0, 550.0), storey_count=15))
    scale_references_from_observation(obs, image_size=IMAGE)
    joined = " ".join(obs.warnings).lower()
    assert "frame" in joined and "building" in joined


def test_a_facade_touching_the_bottom_edge_is_refused_too():
    """Base cut off at the bottom is the same failure from the other end."""
    obs = _observation(_cue(bbox_px=(994.0, 462.0, 1163.0, 1281.0), storey_count=5))
    assert scale_references_from_observation(obs, image_size=IMAGE) == []


def test_a_fully_visible_facade_is_kept():
    """The doctrine's own anchor on this plate — a 5-storey tenement fully
    inside the frame — must survive the check."""
    obs = _observation(_cue(bbox_px=(994.0, 462.0, 1163.0, 645.0), storey_count=5))
    (spec,) = scale_references_from_observation(obs, image_size=IMAGE)
    assert spec["height_m"] == pytest.approx(15.0)
    assert obs.warnings == []


def test_without_an_image_size_nothing_is_refused():
    """Back-compat: callers that cannot say how big the plate is keep the old
    behaviour rather than losing every reference."""
    obs = _observation(_cue(bbox_px=(405.0, 0.0, 575.0, 550.0), storey_count=15))
    assert len(scale_references_from_observation(obs)) == 1


def test_the_edge_margin_tolerates_near_misses():
    """A box one pixel off the edge is still truncated."""
    obs = _observation(_cue(bbox_px=(405.0, 1.0, 575.0, 550.0), storey_count=15))
    assert scale_references_from_observation(obs, image_size=IMAGE) == []


def test_a_non_facade_reference_is_not_edge_checked():
    """Scoped deliberately: the measured evidence is about counted facades. A
    person or car keeps whatever behaviour it had until there is evidence."""
    obs = _observation(_cue(label="car", bbox_px=(0.0, 1000.0, 90.0, 1281.0)))
    (spec,) = scale_references_from_observation(obs, image_size=IMAGE)
    assert spec["reference_id"] == "sedan_car"


def test_the_node_reports_why_it_dropped_a_facade():
    """Gate doctrine: a silent branch-skip needs a visible explanation. The node
    knows the plate size, so it is the one place that CAN run the frame check —
    and it must say when it fires, or the artist just sees 'no references'."""
    torch = pytest.importorskip("torch")

    from atlas_camera.inference import multimodal_helper as mh
    from atlas_camera.comfy.nodes_solve import AtlasVLMScaleCues

    class _Provider:
        def analyze_image(self, path, **kw):
            return _observation(_cue(bbox_px=(405.0, 0.0, 575.0, 550.0),
                                     storey_count=15))

    original = mh.create_multimodal_provider
    mh.create_multimodal_provider = lambda *a, **k: _Provider()
    try:
        image = torch.zeros((1, IMAGE[1], IMAGE[0], 3), dtype=torch.float32)
        refs, report = AtlasVLMScaleCues().analyze(image, provider="lmstudio")
    finally:
        mh.create_multimodal_provider = original

    assert json.loads(refs) == []
    assert "roofline runs off the top" in report
    assert "fully inside the frame" in report


def test_the_prompt_demands_a_fully_visible_facade():
    low = _user_prompt(None, None).lower()
    assert "frame" in low


# ------------------------------------------------- existing behaviour kept

def test_a_cue_without_a_bbox_is_still_skipped():
    assert scale_references_from_observation(
        _observation(_cue(bbox_px=None, storey_count=19))) == []


def test_the_confidence_floor_still_applies():
    assert scale_references_from_observation(
        _observation(_cue(confidence=0.2, storey_count=19)),
        min_confidence=0.5) == []


# --------------------------------------------------- the model's contract

def test_the_prompt_asks_the_model_to_count_storeys():
    prompt = _user_prompt(None, None)
    low = prompt.lower()
    assert "storey" in low or "story" in low
    assert "count" in low


def test_the_schema_exposes_storey_count():
    schema = _scene_observation_json_schema()
    cue_props = schema["properties"]["scale_cues"]["items"]["properties"]
    assert cue_props["storey_count"]["type"] == "integer"


def test_a_payload_carries_the_count_through():
    from atlas_camera.inference.multimodal_helper import _scene_scale_cue_from_payload

    cue = _scene_scale_cue_from_payload({
        "label": "apartment building", "confidence": 0.8,
        "bbox_px": list(BBOX), "storey_count": 19,
    })
    assert cue.storey_count == 19


def test_the_prompt_shows_the_expected_json_shape():
    """Measured live 2026-08-01, gemma-4-12b-qat via LM Studio: given only a
    PROSE description of the fields, the model invented its own key names —
    `anchor_id`, `bbox_px_relative`, `description`, `confidence_score` — and all
    seven of its cues were silently discarded. Showing the shape fixed it."""
    prompt = _user_prompt(["person_175cm"], None)
    assert '"bbox_px"' in prompt
    assert '"storey_count"' in prompt
    assert '"label"' in prompt


def test_the_prompt_stays_short_enough_for_a_local_model():
    """A 1560-char prompt drove gemma-4-12b into a repetition loop on 4/4
    attempts; the 415-char version answered cleanly every time. This is a
    regression guard on prompt bloat, not an arbitrary limit."""
    assert len(_user_prompt(["person_175cm"] * 17, None)) < 2600


@pytest.mark.parametrize("payload,expected_label", [
    ({"anchor_id": "sedan_car"}, "sedan_car"),
    ({"label": "car", "anchor_id": "sedan_car"}, "car"),
])
def test_anchor_id_is_accepted_as_a_label_alias(payload, expected_label):
    from atlas_camera.inference.multimodal_helper import _scene_scale_cue_from_payload

    assert _scene_scale_cue_from_payload(payload).label == expected_label


def test_confidence_score_is_accepted_as_a_confidence_alias():
    from atlas_camera.inference.multimodal_helper import _scene_scale_cue_from_payload

    assert _scene_scale_cue_from_payload(
        {"anchor_id": "car", "confidence_score": 0.85}).confidence == 0.85


def test_description_is_accepted_as_a_notes_alias():
    from atlas_camera.inference.multimodal_helper import _scene_scale_cue_from_payload

    cue = _scene_scale_cue_from_payload(
        {"anchor_id": "car", "description": "dark sedan on the street"})
    assert cue.notes == "dark sedan on the street"


def test_a_bare_bbox_key_is_accepted():
    from atlas_camera.inference.multimodal_helper import _scene_scale_cue_from_payload

    cue = _scene_scale_cue_from_payload(
        {"anchor_id": "car", "bbox": [1.0, 2.0, 3.0, 4.0]})
    assert cue.bbox_px == (1.0, 2.0, 3.0, 4.0)


def test_a_registry_anchor_id_is_offered_as_a_suggested_reference():
    """`anchor_id` is usually a registry id, which is strictly better evidence
    than keyword-matching the label."""
    from atlas_camera.inference.multimodal_helper import _scene_scale_cue_from_payload

    cue = _scene_scale_cue_from_payload({"anchor_id": "building_story_3m"})
    assert "building_story_3m" in cue.suggested_reference_ids
    assert _resolve_reference_id(cue) == "building_story_3m"


def test_an_ambiguous_relative_bbox_is_refused_not_guessed():
    """`bbox_px_relative` arrives with pixel-looking values in an unknown
    coordinate space. Guessing the space would silently misplace the anchor,
    so the cue is dropped instead."""
    from atlas_camera.inference.multimodal_helper import _scene_scale_cue_from_payload

    cue = _scene_scale_cue_from_payload(
        {"anchor_id": "car", "bbox_px_relative": [300, 710, 360, 740]})
    assert cue.bbox_px is None


def test_a_payload_without_a_count_is_none_not_zero():
    from atlas_camera.inference.multimodal_helper import _scene_scale_cue_from_payload

    cue = _scene_scale_cue_from_payload({"label": "car", "confidence": 0.5})
    assert cue.storey_count is None
