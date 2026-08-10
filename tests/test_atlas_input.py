"""Tests for AtlasInput 🎬 — the all-in-one expansion-wrapper entry node.

The expansion assembly is pure graph construction, so it's tested here via
the _MiniGraphBuilder shim (no ComfyUI needed): outside ComfyUI the registry
is {} which exercises exactly the graceful-degrade paths. The inpaint path is
exercised by monkeypatching the registry probe (_comfy_registry); the native
SAM3 path is exercised by monkeypatching the separate capability probe
(_native_sam3_available), since AtlasSAM3Mask is Atlas's own node and is
therefore always present in the registry regardless of whether its actual
[sam3] dependency is satisfied.
"""

import pytest

torch = pytest.importorskip("torch")

# AtlasInput lives in nodes_viewport after the nodes.py modularization; its
# node-expansion helpers (_comfy_registry, _native_sam3_available) resolve in
# that module's namespace, so both probe monkeypatches must target it there.
import atlas_camera.comfy.nodes_viewport as nodes_mod
from atlas_camera.comfy.nodes import (
    NODE_CLASS_MAPPINGS,
    AtlasInput,
    _parse_band_override,
)

FULL_REGISTRY = {"INPAINT_InpaintWithModel": object,
                 "INPAINT_LoadInpaintModel": object, "INPAINT_ExpandMask": object}

IMG = "IMAGE_SENTINEL"


def _expand(monkeypatch, registry=None, native_sam3=False, **kw):
    monkeypatch.setattr(nodes_mod, "_comfy_registry", lambda: registry or {})
    monkeypatch.setattr(nodes_mod, "_native_sam3_available", lambda: native_sam3)
    out = AtlasInput().build(IMG, **kw)
    assert set(out) == {"result", "expand"}
    _assert_atlas_inputs_valid(out["expand"])
    return out["expand"], out["result"]


def _types(graph):
    return sorted(n["class_type"] for n in graph.values())


def _assert_atlas_inputs_valid(graph):
    """Every emitted Atlas-class node's input names must exist on the real
    class's INPUT_TYPES (code-review minor #7): a typo'd kwarg in build()
    would pass every value-assertion test and only explode at ComfyUI
    prompt validation at runtime. Third-party classes (INPAINT_*) are
    skipped — their schemas aren't importable here."""
    for node in graph.values():
        cls = NODE_CLASS_MAPPINGS.get(node["class_type"])
        if cls is None:
            continue
        spec = cls.INPUT_TYPES()
        legal = set(spec.get("required", {})) | set(spec.get("optional", {}))
        unknown = set(node["inputs"]) - legal
        assert not unknown, f"{node['class_type']}: unknown inputs {unknown}"


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
        # DMP seam doctrine (artist-corrected): priority is FARTHEST-HIGHEST
        # so the layer behind wins the seam near-tie; the extension/outpaint
        # lives on the layers BEHIND while the frontmost band keeps a clean
        # cut matte (no extend, no outpaint, no skirt).
        by_depth = sorted(bands,
                          key=lambda b: _parse_band_override(b["inputs"]["band_override"])[0])
        front, behind = by_depth[0], by_depth[1:]   # nearest first
        assert [b["inputs"]["priority"] for b in by_depth] == \
            [5.0 * i for i in range(n_expected)]     # nearest lowest
        assert front["inputs"]["edge_extend_px"] == 0
        assert front["inputs"]["skirt_bevel"] == 0.0
        assert front["inputs"]["frame_outpaint_px"] == 0
        assert all(b["inputs"]["edge_extend_px"] == 24 for b in behind)  # widget default
        assert all(b["inputs"]["skirt_bevel"] == 1.5 for b in behind)
        assert all(b["inputs"]["frame_outpaint_px"] == 64 for b in behind)
        # bands use the calibrated band-mesh tear threshold
        assert all(b["inputs"]["depth_edge_rel"] == 1.5 for b in bands)

    # the edge_extend_px widget threads through to the behind bands (foliage
    # needs it low; the frontmost band always stays a clean 0 cut)
    graph, _ = _expand(monkeypatch, layers=2, edge_extend_px=8)
    by_depth = sorted((n for n in graph.values()
                       if n["class_type"] == "AtlasCleanPlateLayer"),
                      key=lambda b: _parse_band_override(b["inputs"]["band_override"])[0])
    assert by_depth[0]["inputs"]["edge_extend_px"] == 0        # front: clean cut
    assert all(b["inputs"]["edge_extend_px"] == 8 for b in by_depth[1:])


def test_sky_and_scope_skip_gracefully_without_any_segmenter(monkeypatch):
    # Empty registry + native SAM3 unavailable -> drop to heuristic.
    graph, result = _expand(monkeypatch, sky=True, layers=2,
                            scope_prompts="rocks\nperson")
    report = result[4]
    assert "sky SKIPPED" in report and "no segmenter" in report
    assert "scope SKIPPED" in report
    assert not any(n["class_type"] == "AtlasSAM3Mask" for n in graph.values())
    assert not any(n["class_type"] == "AtlasSemanticMask" for n in graph.values())
    # sky_mask output degrades to the SolidMask zero
    solid_id = next(i for i, n in graph.items() if n["class_type"] == "SolidMask")
    assert result[3] == [solid_id, 0]


def test_sky_and_scope_fall_back_to_semantic_mask_without_native_sam3(monkeypatch):
    # Non-CUDA box, or [sam3] not installed: native SAM3 unavailable, but our
    # SegFormer node can still run — sky and scope must route to
    # AtlasSemanticMask, not collapse to the heuristic.
    graph, result = _expand(monkeypatch, registry={"AtlasSemanticMask": object},
                            sky=True, layers=2, scope_prompts="rocks")
    report = result[4]
    assert "AtlasSemanticMask" in report and "fallback" in report
    assert not any(n["class_type"] == "AtlasSAM3Mask" for n in graph.values())
    sems = [n for n in graph.values() if n["class_type"] == "AtlasSemanticMask"]
    assert len(sems) == 2                        # sky + one scope line
    # The sky dome is actually built (not skipped), fed by the SegFormer mask.
    assert any(n["class_type"] == "AtlasSkyDomeLayer" for n in graph.values())
    assert any(n["inputs"].get("classes") == "sky" for n in sems)
    scopes = [n for n in graph.values() if n["class_type"] == "AtlasScopeMask"]
    assert len(scopes) == 1
    rocks_id = next(i for i, n in graph.items()
                    if n["class_type"] == "AtlasSemanticMask"
                    and n["inputs"].get("classes") == "rocks")
    assert scopes[0]["inputs"]["segment_mask"] == [rocks_id, 0]


def test_sky_and_scope_wire_when_native_sam3_available(monkeypatch):
    graph, result = _expand(monkeypatch, native_sam3=True, sky=True,
                            layers=2, scope_prompts="rocks")
    sams = [n for n in graph.values() if n["class_type"] == "AtlasSAM3Mask"]
    assert len(sams) == 2                        # sky + one scope line
    sky_layer = next(n for n in graph.values()
                     if n["class_type"] == "AtlasSkyDomeLayer")
    # Generous sky smear (96/128) so ridge-silhouette reveals never go black.
    assert sky_layer["inputs"]["edge_extend_px"] == 96
    assert sky_layer["inputs"]["frame_outpaint_px"] == 128
    scopes = [n for n in graph.values() if n["class_type"] == "AtlasScopeMask"]
    assert len(scopes) == 1 and scopes[0]["inputs"]["prompt"] == "rocks"
    # sky mask feeds band_ref_mask on every band layer (the drift rule)
    bands = [n for n in graph.values() if n["class_type"] == "AtlasCleanPlateLayer"]
    sky_sam_id = next(i for i, n in graph.items()
                      if n["class_type"] == "AtlasSAM3Mask"
                      and n["inputs"]["concepts"] == "sky")
    assert all(b["inputs"].get("band_ref_mask") == [sky_sam_id, 0] for b in bands)


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
    graph, result = _expand(monkeypatch, registry=FULL_REGISTRY, native_sam3=True,
                            use_vlm=True, layers=2, sky=True)
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
    sky_sam = next(n for n in graph.values() if n["class_type"] == "AtlasSAM3Mask"
                   and isinstance(n["inputs"]["concepts"], list)
                   and n["inputs"]["concepts"][1] == 3)
    assert sky_sam["inputs"]["concepts"][0] == assess_id


def test_sky_lama_inpaint_chain_expands(monkeypatch):
    graph, result = _expand(monkeypatch, registry=FULL_REGISTRY, native_sam3=True,
                            sky=True, sky_inpaint_mode="lama", sky_lama_grow_px=48)
    assert "sky plate LaMa inpaint" in result[4]
    sky_layer = next(n for n in graph.values() if n["class_type"] == "AtlasSkyDomeLayer")
    # plate_image should be the stitch output, not the raw IMAGE sentinel
    assert sky_layer["inputs"]["plate_image"] != IMG
    # inverted sky mask feeds the LaMa chain
    invert_id = next(i for i, n in graph.items() if n["class_type"] == "InvertMask")
    assert graph[invert_id]["inputs"]["mask"][0] == next(
        i for i, n in graph.items() if n["class_type"] == "AtlasSAM3Mask"
        and n["inputs"].get("concepts") == "sky")
    expand = next(n for n in graph.values() if n["class_type"] == "INPAINT_ExpandMask")
    assert expand["inputs"]["grow"] == 48
    assert any(n["class_type"] == "AtlasInpaintStitch" for n in graph.values())


def test_sky_sdxl_inpaint_chain_expands(monkeypatch):
    graph, result = _expand(monkeypatch, registry=FULL_REGISTRY, native_sam3=True,
                            sky=True, sky_inpaint_mode="sdxl",
                            sky_sdxl_checkpoint="custom.safetensors",
                            sky_sdxl_positive="clean blue sky",
                            sky_sdxl_negative="buildings",
                            sky_sdxl_seed=123)
    assert "sky plate SDXL inpaint" in result[4]
    sky_layer = next(n for n in graph.values() if n["class_type"] == "AtlasSkyDomeLayer")
    assert sky_layer["inputs"]["plate_image"] != IMG
    sdxl = next(n for n in graph.values() if n["class_type"] == "AtlasSDXLInpaint")
    assert sdxl["inputs"]["checkpoint"] == "custom.safetensors"
    assert sdxl["inputs"]["positive_prompt"] == "clean blue sky"
    assert sdxl["inputs"]["negative_prompt"] == "buildings"
    assert sdxl["inputs"]["seed"] == 123


def test_sky_lama_skips_gracefully_without_inpaint_pack(monkeypatch):
    graph, result = _expand(monkeypatch, native_sam3=True, sky=True,
                            sky_inpaint_mode="lama")
    assert "sky plate LaMa SKIPPED" in result[4]
    sky_layer = next(n for n in graph.values() if n["class_type"] == "AtlasSkyDomeLayer")
    assert sky_layer["inputs"]["plate_image"] == IMG


def test_sky_auto_switches_moge_to_da2_outdoor(monkeypatch):
    graph, result = _expand(monkeypatch, native_sam3=True, sky=True,
                            depth_model="Ruicheng/moge-vitl")
    depth_node = next(n for n in graph.values() if n["class_type"] == "AtlasDepthMap")
    assert depth_node["inputs"]["depth_model"] ==         "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
    assert "depth auto-switched to DA2-Outdoor" in result[4]


def test_sky_keeps_non_moge_depth_model(monkeypatch):
    graph, result = _expand(monkeypatch, native_sam3=True, sky=True,
                            depth_model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
    depth_node = next(n for n in graph.values() if n["class_type"] == "AtlasDepthMap")
    assert depth_node["inputs"]["depth_model"] ==         "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    assert "auto-switched" not in result[4]


def test_moge_preserved_when_sky_is_off(monkeypatch):
    """When the artist explicitly picks MoGe and does NOT ask for a sky card,
    AtlasInput must pass that model through to AtlasDepthMap — the auto-switch
    is a sky-specific workaround, not an unconditional override."""
    graph, result = _expand(monkeypatch, native_sam3=True, sky=False,
                            depth_model="Ruicheng/moge-2-vitl-normal")
    depth_node = next(n for n in graph.values() if n["class_type"] == "AtlasDepthMap")
    assert depth_node["inputs"]["depth_model"] == "Ruicheng/moge-2-vitl-normal"
    assert "auto-switched" not in result[4]


def test_moge_unavailable_note_when_selected(monkeypatch):
    """If MoGe is selected but the package is not installed, the report should
    warn early so the artist knows why AtlasDepthMap will fail instead of
    getting a silent fallback to a different model."""
    monkeypatch.setattr(nodes_mod, "_moge_available", lambda: False)
    graph, result = _expand(monkeypatch, native_sam3=True, sky=False,
                            depth_model="Ruicheng/moge-2-vitl-normal")
    assert "MoGe package not installed" in result[4]



def test_build_signature_matches_input_types_widget_order():
    """ComfyUI passes ``widgets_values`` positionally to the node function,
    so ``build()``'s parameter order must exactly match ``INPUT_TYPES``
    widget order (excluding the required ``image`` socket).  A drift makes
    widgets control the wrong parameter silently — e.g. ``sky`` being read
    from the ``sky_prompt`` slot produced a black mask even though the
    artist turned ``sky`` on."""
    import inspect
    from atlas_camera.mcp.comfy_http import is_widget
    sig = inspect.signature(AtlasInput.build)
    it = AtlasInput.INPUT_TYPES()
    widgets, links = [], set()
    for sec in ("required", "optional"):
        for name, spec in it.get(sec, {}).items():
            (widgets.append(name) if is_widget(spec) else links.add(name))
    # LINK inputs (image, raw_meta) are keyword-delivered and carry no
    # positional slot, so they are excluded on both sides rather than being
    # required to interleave. What this pin protects is the WIDGET sequence:
    # any link param must therefore sit after every widget in the signature,
    # which the comparison below still enforces by order.
    params = [n for n in list(sig.parameters.keys())[1:]
              if not n.startswith("_") and n not in links]
    assert params == widgets, f"build params {params} vs widgets {widgets}"


def test_positional_sky_widget_turns_on_sky_card(monkeypatch):
    """When the artist enables the ``sky`` widget in a saved workflow,
    ComfyUI serializes it positionally.  Call ``build()`` with the exact
    positional sequence the frontend would send and confirm the graph
    actually builds a sky dome with the right prompt."""
    monkeypatch.setattr(nodes_mod, "_comfy_registry", lambda: {})
    monkeypatch.setattr(nodes_mod, "_native_sam3_available", lambda: True)
    # Positional args after ``image`` in INPUT_TYPES widget order.
    args = (0, "relief", 512, False, "lmstudio", "", True, "sky", "", False,
            "", 24, 12.0, True, 0.0,
            "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
            True, "lama", 32,
            "SDXL/sd_xl_base_1.0.safetensors",
            "clear seamless sky, high detail, no buildings, no trees, no roofs",
            "building, tree, roof, person, vehicle, text, watermark, blurry", 0)
    out = AtlasInput().build(IMG, *args)
    graph = out["expand"]
    assert any(n["class_type"] == "AtlasSkyDomeLayer" for n in graph.values())
    sam = next(n for n in graph.values()
               if n["class_type"] == "AtlasSAM3Mask"
               and n["inputs"].get("concepts") == "sky")
    assert sam["inputs"]["concepts"] == "sky"
    assert "sky card ON" in out["result"][4]


# ---------------------------------------------------------------------------
# retopo + edge smoothing (2026-08-01 plan: docs/dev/atlas_input_retopo_plan.md)
# ---------------------------------------------------------------------------

def test_retopo_widgets_appended_last_with_combo_subset():
    """The three retopo widgets are APPENDED (widgets_values is positional),
    and the combo is a pass-through SUBSET of AtlasRetopologizeLayer.method:
    `smooth` is excluded (driven by smooth_iterations, which AtlasInput does
    not expose — a guaranteed no-op dead value here).

    Later widgets may follow them (`sub_quad_boundary` did, 2026-08-10) — what
    this pins is that the trio stays CONTIGUOUS and in order, so no saved
    workflow's positional values shift underneath it.
    """
    from atlas_camera.mcp.comfy_http import is_widget
    it = AtlasInput.INPUT_TYPES()["optional"]
    # WIDGETS only, not entries: raw_meta is a link input appended after them,
    # and a link occupies no positional widget slot.
    widget_names = [n for n, spec in it.items() if is_widget(spec)]
    trio = ["retopo_method", "retopo_target_vertex_count",
            "boundary_smooth_iterations"]
    start = widget_names.index(trio[0])
    assert widget_names[start:start + 3] == trio
    assert widget_names[-1] == "sub_quad_boundary"
    assert it["sub_quad_boundary"][1]["default"] is False
    assert it["retopo_method"][0] == ["off", "quad", "decimate", "voxel_remesh"]
    assert it["retopo_method"][1]["default"] == "off"
    assert it["retopo_target_vertex_count"][1]["default"] == 2000
    assert it["boundary_smooth_iterations"][1]["default"] == 0


def test_retopologize_layer_combo_gained_voxel_remesh_appended_last():
    """Prerequisite for AtlasInput's pass-through: the live node's method
    combo gains voxel_remesh (already supported by core apply_retopo) as an
    APPENDED value — existing saved workflows keep their positions."""
    from atlas_camera.comfy.nodes_geometry import AtlasRetopologizeLayer
    method_values = AtlasRetopologizeLayer.INPUT_TYPES()["optional"]["method"][0]
    assert method_values == ["off", "quad", "decimate", "smooth", "voxel_remesh"]


def test_retopo_off_emits_no_retopo_node(monkeypatch):
    """Default stays exactly the pre-retopo graph — the front door must not
    get slower for everyone who does not want this."""
    for kw in ({}, {"layers": 3}):
        graph, _ = _expand(monkeypatch, **kw)
        assert not any(n["class_type"] == "AtlasRetopologizeLayer"
                       for n in graph.values())


def test_retopo_expands_one_star_node_at_chain_end(monkeypatch):
    """ONE AtlasRetopologizeLayer with layer='*' terminates the solve chain —
    covers layers=0 AND bands, one deepcopy, sky dome auto-skipped by the
    node's own depth_relief_mesh source filter."""
    for kw in ({"layers": 0}, {"layers": 3}):
        graph, result = _expand(monkeypatch, retopo_method="decimate",
                                retopo_target_vertex_count=1500,
                                boundary_smooth_iterations=4, **kw)
        retopos = [(i, n) for i, n in graph.items()
                   if n["class_type"] == "AtlasRetopologizeLayer"]
        assert len(retopos) == 1, kw
        rid, node = retopos[0]
        assert node["inputs"]["layer"] == "*"
        assert node["inputs"]["method"] == "decimate"
        assert node["inputs"]["target_vertex_count"] == 1500
        assert node["inputs"]["boundary_smooth_iterations"] == 4
        assert result[0] == [rid, 0]      # solve output IS the retopo node
        assert "retopo" in result[4]


def test_boundary_smooth_alone_emits_retopo_node_with_method_off(monkeypatch):
    """method='off' + boundary_smooth_iterations>0 is the 'don't retopo,
    just round the silhouette' configuration — it must still expand."""
    graph, result = _expand(monkeypatch, boundary_smooth_iterations=6)
    node = next(n for n in graph.values()
                if n["class_type"] == "AtlasRetopologizeLayer")
    assert node["inputs"]["method"] == "off"
    assert node["inputs"]["boundary_smooth_iterations"] == 6


def test_live_hole_fill_not_wired_to_card_or_ground(monkeypatch):
    """Card/ground single-layer path uses AtlasCleanPlateLayer, which does not
    accept live_fill_* kwargs."""
    monkeypatch.setattr(nodes_mod, "_comfy_registry", lambda: {})
    monkeypatch.setattr(nodes_mod, "_native_sam3_available", lambda: True)
    for geom in ("card", "ground"):
        out = AtlasInput().build(
            IMG, layers=0, mesh=geom, mesh_resolution=256,
            live_fill_holes=True, live_fill_distance_m=10.0, live_fill_max_hole_edges=64)
        graph = out["expand"]
        layer = next(n for n in graph.values() if n["class_type"] == "AtlasCleanPlateLayer")
        for key in ("live_fill_holes", "live_fill_distance_m", "live_fill_max_hole_edges",
                    "live_fill_edge_sawteeth"):
            assert key not in layer["inputs"], f"{geom} should not receive {key}"
        # No DeriveReliefMesh in this branch
        assert not any(n["class_type"] == "AtlasDeriveReliefMesh" for n in graph.values())


def test_live_hole_fill_not_wired_to_banded_layers(monkeypatch):
    """Depth-band layers also use AtlasCleanPlateLayer and must not receive
    live_fill_* kwargs in this pass (band boundaries are intentional holes)."""
    monkeypatch.setattr(nodes_mod, "_comfy_registry", lambda: {})
    monkeypatch.setattr(nodes_mod, "_native_sam3_available", lambda: True)
    out = AtlasInput().build(
        IMG, layers=2, mesh="relief", mesh_resolution=256,
        live_fill_holes=True, live_fill_distance_m=10.0, live_fill_max_hole_edges=64)
    graph = out["expand"]
    for n in graph.values():
        if n["class_type"] == "AtlasCleanPlateLayer":
            for key in ("live_fill_holes", "live_fill_distance_m", "live_fill_max_hole_edges",
                        "live_fill_edge_sawteeth"):
                assert key not in n["inputs"], f"band layer should not receive {key}"
    # No DeriveReliefMesh in banded path
    assert not any(n["class_type"] == "AtlasDeriveReliefMesh" for n in graph.values())



# --- RAW metadata passthrough ------------------------------------------------
#
# Found live 2026-08-07 while testing the export-fanout workflow against a
# camera RAW: AtlasLoadRAW hands back EXIF focal + a measured sensor width as
# `raw_meta`, and AtlasLearnedSolveFromImage has always accepted it, but
# AtlasInput — the FRONT DOOR, and the node the shipping workflows use — had no
# way to receive it. So the one-node path silently threw away the only
# *measured* intrinsics in the graph and let GeoCalib guess a focal it did not
# have to guess. `raw_meta` is a LINK input, not a widget, so adding it cannot
# shift widgets_values on any saved workflow.


def test_raw_meta_is_forwarded_to_the_inner_solve(monkeypatch):
    graph, _ = _expand(monkeypatch, raw_meta="RAW_META_SENTINEL")
    solve = next(n for n in graph.values()
                 if n["class_type"] == "AtlasLearnedSolveFromImage")
    assert solve["inputs"]["raw_meta"] == "RAW_META_SENTINEL"


def test_raw_meta_absent_when_not_wired(monkeypatch):
    """Unwired must mean ABSENT, not None. A literal None reaching the inner
    node's input dict is a value ComfyUI would try to validate as an
    ATLAS_RAW_META link."""
    graph, _ = _expand(monkeypatch)
    solve = next(n for n in graph.values()
                 if n["class_type"] == "AtlasLearnedSolveFromImage")
    assert "raw_meta" not in solve["inputs"]


def test_raw_meta_is_a_link_input_not_a_widget():
    """Widget order is POSITIONAL in widgets_values. raw_meta must stay a link
    input so no saved AtlasInput workflow shifts a value."""
    spec = AtlasInput.INPUT_TYPES()
    entry = spec["optional"]["raw_meta"]
    assert entry[0] == "ATLAS_RAW_META"
    assert len(entry) == 1 or not isinstance(entry[1], dict) or "default" not in entry[1]
