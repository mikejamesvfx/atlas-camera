"""Structural contracts for the three generated canonical OCIO/DCC graphs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "examples" / "showcase"
SLUGS = ("oceancastle", "spacehangar", "ghosttown")


def _workflow(slug: str) -> dict:
    path = SHOWCASE / f"atlas_canonical_ocio_{slug}_dcc_workflow.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _marketing_workflow(slug: str) -> dict:
    path = SHOWCASE / "marketing" / "workflows" / f"atlas_marketing_ocio_{slug}_dcc_workflow.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _node(wf: dict, type_name: str) -> dict:
    found = [n for n in wf["nodes"] if n["type"] == type_name]
    assert len(found) == 1, (type_name, len(found))
    return found[0]


@pytest.mark.parametrize("slug", SLUGS)
def test_canonical_ocio_dcc_contract(slug):
    wf = _workflow(slug)
    assert len(wf["groups"]) == 3
    assert wf["extra"]["source_colorspace"] == "ACEScg"
    assert wf["extra"]["neural_working_colorspace"] == "sRGB - Display"

    ocio = _node(wf, "OCIORead")
    assert "ACEScg" in ocio["widgets_values"]
    assert "sRGB - Display" in ocio["widgets_values"]

    if slug in {"oceancastle", "ghosttown"}:
        assert _node(wf, "INPAINT_InpaintWithModel")
    else:
        sdxl = _node(wf, "AtlasSDXLInpaint")
        assert True in sdxl["widgets_values"]
        assert 1024 in sdxl["widgets_values"]
    clean_depth = next(n for n in wf["nodes"] if n.get("title") == "Cleanplate depth — continuous hidden support")
    assert clean_depth["type"] == "AtlasDepthMap"
    assert wf["extra"]["cleanplate_depth_geometry"] is True
    cleanplates = [n for n in wf["nodes"] if n["type"] == "AtlasCleanPlateLayer"]
    assert len(cleanplates) == 2
    for layer in cleanplates:
        assert "manual" in layer["widgets_values"]

    for type_name in ("AtlasExportNukeLayers", "AtlasExportMayaLayers"):
        exporter = _node(wf, type_name)
        assert "decimate" in exporter["widgets_values"]
        assert wf["extra"]["retopo_target_vertex_count_per_layer"] in exporter["widgets_values"]
        profile = next(i for i in exporter["inputs"] if i["name"] == "output_profile")
        assert profile["link"] is not None


def test_hangar_uses_interior_recipe_and_outdoors_use_sky_cards():
    hangar = next(n for n in _workflow("spacehangar")["nodes"]
                  if n.get("title") == "Shared metric depth")["widgets_values"]
    assert "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf" in hangar

    for slug in ("oceancastle", "ghosttown"):
        outdoor = next(n for n in _workflow(slug)["nodes"]
                       if n.get("title") == "Shared metric depth")["widgets_values"]
        assert "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf" in outdoor


@pytest.mark.parametrize("slug", SLUGS)
def test_marketing_variant_exports_the_approved_4k_cleanplate(slug):
    wf = _marketing_workflow(slug)
    assert wf["extra"]["approved_marketing_cleanplate"] is True
    approved = next(n for n in wf["nodes"] if n.get("title") == "APPROVED 4K MARKETING CLEANPLATE")
    assert approved["type"] == "OCIORead"
    assert any(f"{slug}_marketing_cleanplate_4k.png" in str(v) for v in approved["widgets_values"])
    assert "sRGB - Display" in approved["widgets_values"]
    assert any(n.get("title") == "HERO OUTPUT — screenshot this preview" for n in wf["nodes"])

    background = next(n for n in wf["nodes"] if n.get("title") == "Generated background layer")
    plate_link = next(i["link"] for i in background["inputs"] if i["name"] == "plate_image")
    origin_id = next(link[1] for link in wf["links"] if link[0] == plate_link)
    assert origin_id == approved["id"]
    depth_link = next(i["link"] for i in background["inputs"] if i["name"] == "depth")
    depth_origin = next(link[1] for link in wf["links"] if link[0] == depth_link)
    clean_depth = next(n for n in wf["nodes"] if n.get("title") == "Cleanplate depth — continuous hidden support")
    assert depth_origin == clean_depth["id"]
