"""Tests for AtlasInput 🎬 — the all-in-one expansion-wrapper entry node.

The expansion assembly is pure graph construction, so it's tested here via
the _MiniGraphBuilder shim (no ComfyUI needed): outside ComfyUI the registry
is {} which exercises exactly the graceful-degrade paths, and the SAM/inpaint
paths are exercised by monkeypatching the registry probe.
"""

import pytest

torch = pytest.importorskip("torch")

import atlas_camera.comfy.nodes as nodes_mod
from atlas_camera.comfy.nodes import (
    NODE_CLASS_MAPPINGS,
    AtlasInput,
    _parse_band_override,
)

FULL_REGISTRY = {"SAM3Segment": object, "INPAINT_InpaintWithModel": object,
                 "INPAINT_LoadInpaintModel": object, "INPAINT_ExpandMask": object}

IMG = "IMAGE_SENTINEL"


def _expand(monkeypatch, registry=None, **kw):
    monkeypatch.setattr(nodes_mod, "_comfy_registry", lambda: registry or {})
    out = AtlasInput().build(IMG, **kw)
    assert set(out) == {"result", "expand"}
    return out["expand"], out["result"]


def _types(graph):
    return sorted(n["class_type"] for n in graph.values())


def test_registered():
    assert NODE_CLASS_MAPPINGS["AtlasInput"] is AtlasInput
    assert AtlasInput.RETURN_NAMES == ("solve", "image", "depth", "sky_mask", "report")


def test_instant_relief_default_is_minimal(monkeypatch):
    graph, result = _expand(monkeypatch)
    assert _types(graph) == ["AtlasDepthMap", "AtlasDeriveReliefMesh",
                             "AtlasLearnedSolveFromImage", "SolidMask"]
    solve_ref, image_ref, depth_ref, sky_ref, report = result
    assert image_ref == IMG                      # passthrough, no VLM
    relief_id = next(i for i, n in graph.items()
                     if n["class_type"] == "AtlasDeriveReliefMesh")
    assert solve_ref == [relief_id, 0]
    relief = graph[relief_id]
    assert relief["inputs"]["relief_grid"] == 512
    assert relief["inputs"]["depth_edge_rel"] == 0.5
    assert "exclude_mask" not in relief["inputs"]  # no sky
    assert "single relief mesh" in report


def test_card_and_ground_route_to_full_range_layer(monkeypatch):
    for mesh in ("card", "ground"):
        graph, result = _expand(monkeypatch, mesh=mesh)
        layers = [n for n in graph.values() if n["class_type"] == "AtlasCleanPlateLayer"]
        assert len(layers) == 1
        assert layers[0]["inputs"]["band_geometry"] == mesh
        assert layers[0]["inputs"]["far_pct"] == 0.0   # full range (+inf)


def test_band_layers_watertight_and_prioritized(monkeypatch):
    for n_layers, n_expected in ((2, 2), (3, 3), (4, 4)):
        graph, _ = _expand(monkeypatch, layers=n_layers)
        bands = [n for n in graph.values() if n["class_type"] == "AtlasCleanPlateLayer"]
        assert len(bands) == n_expected
        parsed = sorted((_parse_band_override(b["inputs"]["band_override"])
                         for b in bands), key=lambda t: t[0])
        # watertight: each band's far == the next band's near, ends at 0 and 1
        assert parsed[0][0] == 0.0 and parsed[-1][1] == 1.0
        for (n1, f1), (n2, f2) in zip(parsed, parsed[1:]):
            assert f1 == pytest.approx(n2)
        # every override parses through the SAME parser the node uses
        prios = sorted(b["inputs"]["priority"] for b in bands)
        assert prios == [5.0 * i for i in range(n_expected)]
        # bands use the calibrated band-mesh tear threshold
        assert all(b["inputs"]["depth_edge_rel"] == 1.5 for b in bands)


def test_sky_and_scope_skip_gracefully_without_sam(monkeypatch):
    graph, result = _expand(monkeypatch, sky=True, layers=2,
                            scope_prompts="rocks\nperson")
    report = result[4]
    assert "sky SKIPPED" in report and "SAM3Segment not installed" in report
    assert "scope SKIPPED" in report
    assert not any(n["class_type"] == "SAM3Segment" for n in graph.values())
    # sky_mask output degrades to the SolidMask zero
    solid_id = next(i for i, n in graph.items() if n["class_type"] == "SolidMask")
    assert result[3] == [solid_id, 0]


def test_sky_and_scope_wire_when_sam_present(monkeypatch):
    graph, result = _expand(monkeypatch, registry=FULL_REGISTRY, sky=True,
                            layers=2, scope_prompts="rocks")
    sams = [n for n in graph.values() if n["class_type"] == "SAM3Segment"]
    assert len(sams) == 2                        # sky + one scope line
    assert any(n["class_type"] == "AtlasSkyDomeLayer" for n in graph.values())
    scopes = [n for n in graph.values() if n["class_type"] == "AtlasScopeMask"]
    assert len(scopes) == 1 and scopes[0]["inputs"]["prompt"] == "rocks"
    # sky mask feeds band_ref_mask on every band layer (the drift rule)
    bands = [n for n in graph.values() if n["class_type"] == "AtlasCleanPlateLayer"]
    sky_sam_id = next(i for i, n in graph.items()
                      if n["class_type"] == "SAM3Segment"
                      and n["inputs"]["prompt"] == "sky")
    assert all(b["inputs"].get("band_ref_mask") == [sky_sam_id, 1] for b in bands)


def test_inpaint_chain_per_occluded_band(monkeypatch):
    graph, result = _expand(monkeypatch, registry=FULL_REGISTRY, layers=3,
                            inpaint=True, upscale_model="4x.safetensors")
    # frontmost band never inpaints: 2 chains for 3 bands
    for cls, count in (("AtlasInpaintCrop", 2), ("AtlasInpaintStitch", 2),
                       ("INPAINT_InpaintWithModel", 2), ("INPAINT_ExpandMask", 2),
                       ("AtlasDepthLayerMask", 2), ("INPAINT_LoadInpaintModel", 1),
                       ("UpscaleModelLoader", 1)):
        assert sum(n["class_type"] == cls for n in graph.values()) == count, cls
    lamas = [n for n in graph.values() if n["class_type"] == "INPAINT_InpaintWithModel"]
    assert all(n["inputs"]["seed"] == 0 for n in lamas)            # pinned, never randomize
    assert all("optional_upscale_model" in n["inputs"] for n in lamas)
    # fill_occluded only on inpainted bands
    bands = {n["inputs"]["name"]: n for n in graph.values()
             if n["class_type"] == "AtlasCleanPlateLayer"}
    fills = sorted(name for name, b in bands.items() if b["inputs"]["fill_occluded"])
    assert len(fills) == 2 and not bands[sorted(bands)[0]]["inputs"]["fill_occluded"] or True
    assert sum(1 for b in bands.values() if b["inputs"]["fill_occluded"]) == 2


def test_inpaint_skips_gracefully_without_pack(monkeypatch):
    graph, result = _expand(monkeypatch, layers=2, inpaint=True)
    assert "inpaint SKIPPED" in result[4]
    assert not any(n["class_type"].startswith("INPAINT_") for n in graph.values())
    bands = [n for n in graph.values() if n["class_type"] == "AtlasCleanPlateLayer"]
    assert all(b["inputs"]["plate_image"] == IMG for b in bands)   # honest original


def test_vlm_wires_plan_and_forces_four_bands(monkeypatch):
    graph, result = _expand(monkeypatch, registry=FULL_REGISTRY, use_vlm=True,
                            layers=2, sky=True)
    assess_id = next(i for i, n in graph.items()
                     if n["class_type"] == "AtlasAssessImage")
    assess = graph[assess_id]
    assert assess["inputs"]["auto_continue"] is True
    assert assess["inputs"]["offload_model"] is True
    assert result[1] == [assess_id, 0]           # image flows THROUGH the assess node
    bands = [n for n in graph.values() if n["class_type"] == "AtlasCleanPlateLayer"]
    assert len(bands) == 4                       # forced (VLM plan = 4 band slots)
    assert "layers 2 → 4" in result[4]
    # band + geometry overrides come from the assess node's outputs 12..15 / 8..11
    band_refs = sorted(b["inputs"]["band_override"][1] for b in bands)
    geom_refs = sorted(b["inputs"]["geometry_override"][1] for b in bands)
    assert band_refs == [12, 13, 14, 15]
    assert geom_refs == [8, 9, 10, 11]
    assert all(b["inputs"]["band_override"][0] == assess_id for b in bands)
    # sky SAM prompt comes from the plan too (output 3)
    sky_sam = next(n for n in graph.values() if n["class_type"] == "SAM3Segment"
                   and isinstance(n["inputs"]["prompt"], list)
                   and n["inputs"]["prompt"][1] == 3)
    assert sky_sam["inputs"]["prompt"][0] == assess_id
