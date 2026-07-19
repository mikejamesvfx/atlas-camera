"""The OCIO showcase workflows must hand the DCC a colour-managed background.

Both tiers register a *durable* plate for the generated background layer, but
for different reasons:

* marketing  — the approved cleanplate is a real file, registered with its true
  colorspace (sRGB - Display).
* canonical  — the background is generated in-graph, so ``OCIOWrite`` writes it
  back to ACEScg and *that* file is registered.

The canonical case is the fragile one. ``exporters/_layers.py`` copies
``plate_ref.plate_path`` into the .nk/.ma **verbatim, with no existence check**,
and setting it *suppresses* the PNG fallback — so a registered path that does
not match what ``OCIOWrite`` actually writes yields a broken Read with no
image behind it. That is strictly worse than leaving ``plate_ref`` unwired.

These tests pin the two halves against each other. They are pure JSON +
generator-helper checks: no torch, no live ComfyUI, safe in CI.

The naming rule below is a deliberate hand-mirror of ComfyUI-OCIO's own
``_cs_tag``/``write`` in ``io_nodes.py`` — the same accepted-duplication pattern
this repo already uses for the frontend palette and Catmull-Rom mirrors. It was
verified live by queueing a one-node OCIOWrite and reading the result off disk
(``probe_plate_acescg.exr``). Re-verify the same way if that pack changes.

The generator itself is deliberately NOT imported: it runs its WF-helper codegen
at import time, and the invariant worth pinning is internal to each shipped
workflow anyway — the writer's widgets and the registered path must agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ComfyUI-OCIO: still image -> <folder>/<filename>[_<cs-tag>].<ext>
_STILL_EXT = {"exr": "exr", "tiff": "tif", "png": "png", "jpeg": "jpg"}
_CS_TAGS = ("acescct", "acescc", "acescg", "aces2065", "rec709", "srgb", "linear")


def _cs_tag(colorspace: str) -> str:
    low = (colorspace or "").lower()
    for tag in _CS_TAGS:
        if tag in low:
            return tag
    return low


def _ociowrite_output(widgets: list) -> str:
    """The exact path OCIOWrite will write, from its positional widgets."""
    out_colorspace, container, still_format = widgets[1], widgets[2], widgets[3]
    output_folder, filename, colorspace_in_name = widgets[12], widgets[13], widgets[14]
    assert container == "still image", (
        "a numbered sequence appends .0001 and breaks the registered path"
    )
    stem = f"{filename}_{_cs_tag(out_colorspace)}" if colorspace_in_name else filename
    return f"{output_folder}/{stem}.{_STILL_EXT[still_format]}"

REPO = Path(__file__).resolve().parents[1]
CANONICAL = sorted(
    (REPO / "examples" / "showcase").glob("atlas_canonical_ocio_*_dcc_workflow.json")
)
MARKETING = sorted(
    (REPO / "examples" / "showcase" / "marketing" / "workflows").glob(
        "atlas_marketing_ocio_*_dcc_workflow.json"
    )
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _nodes(doc: dict, node_type: str) -> list[dict]:
    return [n for n in doc["nodes"] if n.get("type") == node_type]


def _widgets(node: dict) -> list:
    return node.get("widgets_values") or []


def _background_layer(doc: dict) -> dict:
    hits = [
        n
        for n in _nodes(doc, "AtlasCleanPlateLayer")
        if "background" in (n.get("title") or "").lower()
    ]
    assert len(hits) == 1, "expected exactly one generated background layer"
    return hits[0]


def _clean_plate_ref(doc: dict) -> dict:
    """The AtlasRegisterPlate whose ``role`` widget is ``clean_plate``."""
    hits = [
        n
        for n in _nodes(doc, "AtlasRegisterPlate")
        if len(_widgets(n)) > 3 and _widgets(n)[3] == "clean_plate"
    ]
    assert len(hits) == 1, "expected exactly one clean_plate AtlasRegisterPlate"
    return hits[0]


def _plate_ref_link(layer: dict):
    slot = [i for i in layer["inputs"] if i["name"] == "plate_ref"]
    assert slot, "background layer has no plate_ref input"
    return slot[0].get("link")


assert CANONICAL, "no canonical OCIO workflows found"
assert MARKETING, "no marketing OCIO workflows found"


@pytest.mark.parametrize("path", CANONICAL, ids=lambda p: p.stem)
def test_canonical_registers_exactly_what_ociowrite_produces(path: Path):
    """The registered path must equal the file OCIOWrite writes — no guessing."""
    doc = _load(path)

    writers = _nodes(doc, "OCIOWrite")
    assert len(writers) == 1, "expected one OCIOWrite for the generated cleanplate"
    w = _widgets(writers[0])
    # Widget order is positional; these indices follow OCIOWrite's INPUT_TYPES
    # (``images`` is a link, not a widget).
    assert w[1] == "ACEScg", "generated content must go back to ACEScg"
    assert w[3] == "exr", "the DCC handoff must be float EXR, not 8-bit"
    assert w[5] in ("16f", "32f"), "EXR bit depth must be float"

    written = _ociowrite_output(w)
    registered = _widgets(_clean_plate_ref(doc))
    assert registered[0] == written, (
        "registered plate_path does not match the file OCIOWrite writes — the "
        "exporters would bake a broken Read into the .nk/.ma with no fallback"
    )
    assert registered[1] == "ACEScg"


@pytest.mark.parametrize("path", CANONICAL + MARKETING, ids=lambda p: p.stem)
def test_background_layer_has_a_non_proxy_plate_ref(path: Path):
    """A blank plate_path makes the ref a proxy, which the exporters ignore."""
    doc = _load(path)
    assert _plate_ref_link(_background_layer(doc)) is not None, (
        "generated background layer has no plate_ref — the layer exporters "
        "would fall back to an 8-bit sRGB PNG carrying no colorspace"
    )
    plate_path = _widgets(_clean_plate_ref(doc))[0]
    assert plate_path.strip(), (
        "blank plate_path sets is_proxy=True, so exporters silently skip the ref"
    )
