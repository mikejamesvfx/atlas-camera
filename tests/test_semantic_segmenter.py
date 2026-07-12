"""AtlasSemanticMask 🧩 — pure-function label matching + node registration.

The SegFormer inference path needs downloaded weights, so it is exercised
live (not here); the class-query resolution is pure and pinned below.
"""

import pytest

from atlas_camera.inference.semantic_segmenter import (
    DEFAULT_SEGFORMER_MODEL,
    SEGFORMER_MODELS,
    match_class_ids,
)

# Representative slice of the real ADE20K id2label (string keys — HF configs
# deserialize JSON object keys as str; synonym lists and trailing spaces both
# occur in the shipped configs).
ADE = {
    "0": "wall", "1": "building;edifice", "2": "sky", "3": "floor;flooring",
    "4": "tree", "5": "ceiling", "6": "road;route", "9": "grass",
    "12": "person;individual", "13": "earth;ground", "16": "mountain;mount",
    "21": "water", "26": "sea", "8": "windowpane;window ",
    "48": "skyscraper",
}


def test_exact_and_synonym_matching():
    # Exact-first: "sky" must NOT bleed into "skyscraper" via substring
    # (found live on the b0 model's real id2label).
    ids, matched = match_class_ids(ADE, "sky")
    assert ids == {2} and matched == ["sky"]
    # Synonym-part exact match: "ground" is a part of "earth;ground".
    ids, _ = match_class_ids(ADE, "ground")
    assert ids == {13}
    # Multi-class union.
    ids, _ = match_class_ids(ADE, "sky, floor, person")
    assert ids == {2, 3, 12}


def test_substring_fallback_and_trailing_space():
    # "window" isn't a whole ADE part but substring-matches "windowpane".
    ids, matched = match_class_ids(ADE, "window")
    assert ids == {8}
    assert matched == ["windowpane;window"]  # stripped in the report


def test_no_match_and_empty_query():
    assert match_class_ids(ADE, "spaceship") == (set(), [])
    assert match_class_ids(ADE, "") == (set(), [])
    assert match_class_ids(ADE, " , ,") == (set(), [])


def test_registered_in_node_pack():
    torch = pytest.importorskip("torch")  # noqa: F841 — nodes.py needs it at import
    from atlas_camera.comfy.nodes import (NODE_CLASS_MAPPINGS,
                                          AtlasSemanticMask)
    assert NODE_CLASS_MAPPINGS["AtlasSemanticMask"] is AtlasSemanticMask
    inputs = AtlasSemanticMask.INPUT_TYPES()
    assert inputs["required"]["classes"][1]["default"] == "sky"
    assert list(inputs["optional"]["model"][0]) == list(SEGFORMER_MODELS)
    assert SEGFORMER_MODELS[0] == DEFAULT_SEGFORMER_MODEL
