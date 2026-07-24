"""Graph-integrity regression tests for every shipped example workflow.

ComfyUI's UI-format JSON is redundantly linked (the links array AND each
node's inputs[].link / outputs[].links must agree), and `widgets_values` is
positional — both have caused real shipped bugs (the 2026-07-06 widget-order
corruption; a hand-edited origin-links omission in the shot-cam workflow that
this test's first run caught). The staged master alone was hand-edited eight
times on 2026-07-11; every edit was validated by a throwaway copy of exactly
this checker. Promoted to a permanent test per the beta-0.3 spec-panel review
("highest value-per-line item"): no live ComfyUI needed, runs in milliseconds.
"""

import glob
import json
import os
import re

import pytest

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")

STAGED_MASTER = "atlas_camera_staged_master_workflow.json"
STAGED_AGENTIC = "atlas_camera_staged_master_agentic_assessment_workflow.json"
QUICKSTART_PAIRS = (
    ("atlas_input_quickstart_workflow.json",
     "atlas_input_quickstart_agentic_assessment_workflow.json"),
    ("atlas_occlusion_cull_quickstart_workflow.json",
     "atlas_occlusion_cull_quickstart_agentic_assessment_workflow.json"),
)

# ComfyUI frontend zod schema: workflow id is z.string().uuid().optional()
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _ui_workflows():
    out = []
    for path in sorted(glob.glob(os.path.join(EXAMPLES_DIR, "*.json"))):
        try:
            wf = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # UI format only — api-format exports and non-workflow JSON are skipped
        if isinstance(wf, dict) and isinstance(wf.get("nodes"), list) and "links" in wf:
            out.append((os.path.basename(path), wf))
    return out


_WORKFLOWS = _ui_workflows()


def test_examples_directory_has_ui_workflows():
    # Shipping catalog: the three artist-facing workflows + three dedicated
    # agentic/headless variants (each preserves its source graph and appends one
    # enabled AtlasAssessOutput terminal + evidence preview), the AtlasInput
    # walkthrough, plus the 2026-07-23/24 additions: the live-mesh-repair test
    # pair (3-layer MoGe machine/foreground split; 4-layer bounded-band +
    # background-card gap fill), the KJ-rail staged master (portal plate,
    # Juggernaut SDXL), the RAW->ACEScg 3-layer OCIO workflow, and the
    # AtlasLiveMeshRepair band-box test graph. Deletion or an unreviewed
    # addition still fails loudly — review = add the name here.
    names = sorted(n for n, _ in _WORKFLOWS)
    assert names == ["atlas_3layer_sky_machine_foreground_moge_workflow.json",
                     "atlas_4layer_boundedband_bgcard_moge_workflow.json",
                     "atlas_camera_staged_master_agentic_assessment_workflow.json",
                     "atlas_camera_staged_master_workflow.json",
                     "atlas_input_quickstart_agentic_assessment_workflow.json",
                     "atlas_input_quickstart_workflow.json",
                     "atlas_input_walkthrough_switches_workflow.json",
                     "atlas_live_repair_and_bands_test_workflow.json",
                     "atlas_occlusion_cull_quickstart_agentic_assessment_workflow.json",
                     "atlas_occlusion_cull_quickstart_workflow.json",
                     "atlas_raw_3layer_ocio_workflow.json",
                     "atlas_staged_master_portal_neat_workflow.json"]


@pytest.mark.parametrize("name,wf", _WORKFLOWS, ids=[n for n, _ in _WORKFLOWS])
def test_workflow_link_graph_is_bidirectionally_consistent(name, wf):
    """Every link must be listed by its origin output AND referenced by its
    target input, and every node-side link id must exist in the links array —
    the invariant every hand/script edit of a workflow must preserve."""
    nodes = {n["id"]: n for n in wf["nodes"]}
    errs = []
    links = wf.get("links") or []
    for l in links:
        assert isinstance(l, list) and len(l) >= 6, f"malformed link entry {l!r}"
        lid, oid, oslot, tid, tslot = l[:5]
        if oid not in nodes or tid not in nodes:
            errs.append(f"link {lid}: references missing node ({oid}->{tid})")
            continue
        outs = nodes[oid].get("outputs") or []
        ins = nodes[tid].get("inputs") or []
        if oslot >= len(outs) or lid not in (outs[oslot].get("links") or []):
            errs.append(f"link {lid}: origin {oid}:{oslot} does not list it")
        if tslot >= len(ins) or ins[tslot].get("link") != lid:
            errs.append(f"link {lid}: target {tid}:{tslot} does not reference it")
    link_ids = {l[0] for l in links}
    for n in wf["nodes"]:
        for inp in n.get("inputs") or []:
            if inp.get("link") is not None and inp["link"] not in link_ids:
                errs.append(f"node {n['id']}: dangling input link {inp['link']}")
        for out in n.get("outputs") or []:
            for lid in out.get("links") or []:
                if lid not in link_ids:
                    errs.append(f"node {n['id']}: dangling output link {lid}")
    assert not errs, f"{name}: " + "; ".join(errs[:10])


@pytest.mark.parametrize("name,wf", _WORKFLOWS, ids=[n for n, _ in _WORKFLOWS])
def test_workflow_id_counters_cover_contents(name, wf):
    """last_node_id / last_link_id must be >= every id in use — a stale
    counter makes the frontend mint DUPLICATE ids for newly added nodes."""
    node_ids = [n["id"] for n in wf["nodes"] if isinstance(n["id"], int)]
    link_ids = [l[0] for l in (wf.get("links") or []) if isinstance(l[0], int)]
    if node_ids and isinstance(wf.get("last_node_id"), int):
        assert wf["last_node_id"] >= max(node_ids), name
    if link_ids and isinstance(wf.get("last_link_id"), int):
        assert wf["last_link_id"] >= max(link_ids), name


def _staged():
    match = [wf for n, wf in _WORKFLOWS if n == STAGED_MASTER]
    assert match, "staged master workflow missing from examples/"
    return match[0]


def _staged_agentic():
    return _workflow(STAGED_AGENTIC)


def _workflow(name):
    match = [wf for workflow_name, wf in _WORKFLOWS if workflow_name == name]
    assert match, f"{name} missing from examples/"
    return match[0]


def _all_nodes(wf):
    nodes = list(wf.get("nodes") or [])
    for definition in (wf.get("definitions") or {}).get("subgraphs") or []:
        nodes.extend(definition.get("nodes") or [])
    return nodes


def test_staged_master_uses_five_native_subgraphs_without_legacy_rails():
    """The staged master uses five real subgraphs, with no legacy rails."""
    wf = _staged()
    definitions = (wf.get("definitions") or {}).get("subgraphs") or []
    assert [item["name"] for item in definitions] == [
        "1 · SKY LAYER", "2 · FAR LAYER", "3 · BACKGROUND LAYER",
        "4 · MIDGROUND LAYER", "5 · FOREGROUND LAYER"]
    forbidden = {"SetNode", "GetNode", "Fast Groups Bypasser (rgthree)",
                 "INPAINT_LoadInpaintModel", "INPAINT_InpaintWithModel",
                 "INPAINT_ExpandMask", "UpscaleModelLoader"}
    assert not forbidden.intersection(node["type"] for node in _all_nodes(wf))
    assert sum(node["type"] == "AtlasSDXLInpaint" for node in _all_nodes(wf)) == 4


def test_staged_master_subgraph_links_are_consistent():
    wf = _staged()
    for definition in wf["definitions"]["subgraphs"]:
        nodes = {node["id"]: node for node in definition["nodes"]}
        inputs = definition["inputs"]
        outputs = definition["outputs"]
        link_ids = {link["id"] for link in definition["links"]}
        for link in definition["links"]:
            lid = link["id"]
            if link["origin_id"] == -10:
                assert lid in inputs[link["origin_slot"]]["linkIds"]
            else:
                assert lid in (nodes[link["origin_id"]]["outputs"]
                               [link["origin_slot"]].get("links") or [])
            if link["target_id"] == -20:
                assert lid in outputs[link["target_slot"]]["linkIds"]
            else:
                assert (nodes[link["target_id"]]["inputs"]
                        [link["target_slot"]].get("link") == lid)
        assert all(lid in link_ids for item in inputs + outputs
                   for lid in item.get("linkIds") or [])


def test_staged_master_band_priorities_are_farthest_highest():
    """DMP seam doctrine (2026-07-12, from the quickstart's striped-seam
    fix): at a watertight band seam the two surfaces are depth-adjacent and
    the priority near-tie bias decides which paints — farthest-highest makes
    the layer BEHIND win the seam ribbon, so a band's edge smear can never
    render in front of the layer behind it. Keeps the staged master and
    AtlasInput on the same convention."""
    wf = _staged()
    prios = {n["widgets_values"][4]: n["widgets_values"][5]
             for n in _all_nodes(wf) if n["type"] == "AtlasCleanPlateLayer"}
    assert prios == {"band_far": 15, "band_bg": 10, "band_mid": 5, "band_fg": 0}


def test_staged_master_scope_rows_are_always_active():
    """v7 doctrine: 🎯 AtlasScopeMask rows self-disarm — none may ship
    bypassed (mode 4) or muted (mode 2), and there must be one per band."""
    wf = _staged()
    scopes = [n for n in _all_nodes(wf) if n["type"] == "AtlasScopeMask"]
    assert len(scopes) == 4
    assert all(n.get("mode", 0) == 0 for n in scopes)


def _typed_edges(wf):
    nodes = {node["id"]: node for node in wf["nodes"]}
    edges = set()
    for _, origin_id, origin_slot, target_id, target_slot, *_ in wf["links"]:
        origin = nodes[origin_id]
        target = nodes[target_id]
        edges.add((origin["type"], origin["outputs"][origin_slot]["name"],
                   target["type"], target["inputs"][target_slot]["name"]))
    return edges


def test_shipping_quickstarts_use_current_outputs_and_guidance():
    """The lightweight workflows must not regress to the old Nuke/Maya/USD-
    only handoff or teach third-party SAM as AtlasInput's primary path."""
    required = {
        "LoadImage", "AtlasInput", "AtlasViewportControls",
        "AtlasBlockoutViewport", "Note", "AtlasExportSolveJSON",
        "AtlasExportNukeLayers", "AtlasExportMayaLayers",
        "AtlasExportNuke", "AtlasExportMayaReviewScene",
        "AtlasExportBlender", "AtlasExportUSD", "AtlasExportReliefMesh",
    }
    for base_name, agentic_name in QUICKSTART_PAIRS:
        base = _workflow(base_name)
        agentic = _workflow(agentic_name)
        assert {node["type"] for node in base["nodes"]} == required
        assert {node["type"] for node in agentic["nodes"]} == (
            required | {"AtlasAssessOutput", "PreviewImage"})
        assert not any(node["type"] == "AtlasAssessOutput"
                       for node in base["nodes"])
        assessor = next(node for node in agentic["nodes"]
                        if node["type"] == "AtlasAssessOutput")
        assert assessor["widgets_values"][0] is True
        assert agentic["extra"]["atlas_agentic_assessment"] is True
        agentic_load = next(node for node in agentic["nodes"]
                            if node["type"] == "LoadImage")
        assert agentic_load["widgets_values"][0] == (
            "moge_hangar_proj.jpg" if "occlusion" in agentic_name
            else "ghosttown.jpg")

        for wf in (base, agentic):
            note = next(node for node in wf["nodes"] if node["type"] == "Note")
            text = note["widgets_values"][0]
            assert "native" in text.casefold() and "AtlasSAM3Mask" in text
            assert "cropped" in text and "SDXL" in text
            atlas_input = next(node for node in wf["nodes"]
                               if node["type"] == "AtlasInput")
            # Positional index 13 is the append-stable sky_heuristic widget.
            assert atlas_input["widgets_values"][13] is False
            edges = _typed_edges(wf)
            for exporter in ("AtlasExportNukeLayers", "AtlasExportMayaLayers",
                             "AtlasExportNuke", "AtlasExportMayaReviewScene",
                             "AtlasExportBlender"):
                assert ("AtlasViewportControls", "output_profile", exporter,
                        "output_profile") in edges
            for exporter in ("AtlasExportNuke", "AtlasExportMayaReviewScene"):
                assert ("AtlasExportReliefMesh", "obj_path", exporter,
                        "relief_mesh_obj_path") in edges
            assert ("AtlasViewportControls", "output_profile",
                    "AtlasBlockoutViewport", "output_profile") in edges
            assert ("AtlasViewportControls", "controls",
                    "AtlasBlockoutViewport", "controls") in edges

        agentic_edges = _typed_edges(agentic)
        assert ("AtlasBlockoutViewport", "shaded",
                "AtlasAssessOutput", "camera_view") in agentic_edges
        assert ("AtlasInput", "solve", "AtlasAssessOutput", "solve") in agentic_edges
        assert ("AtlasInput", "image", "AtlasAssessOutput",
                "source_image") in agentic_edges
        assert ("AtlasInput", "depth", "AtlasAssessOutput", "depth") in agentic_edges
        assert ("AtlasAssessOutput", "assessed_image",
                "PreviewImage", "images") in agentic_edges


def test_quickstart_agentic_variants_only_append_terminal_assessment():
    expected_paths = {
        "atlas_input_quickstart_agentic_assessment_workflow.json":
            "atlas_debug/atlas_input_quickstart_agentic_output_assessment.json",
        "atlas_occlusion_cull_quickstart_agentic_assessment_workflow.json":
            "atlas_debug/atlas_occlusion_quickstart_agentic_output_assessment.json",
    }
    assessment_edges = {
        ("AtlasBlockoutViewport", "shaded", "AtlasAssessOutput", "camera_view"),
        ("AtlasInput", "solve", "AtlasAssessOutput", "solve"),
        ("AtlasInput", "image", "AtlasAssessOutput", "source_image"),
        ("AtlasInput", "depth", "AtlasAssessOutput", "depth"),
        ("AtlasAssessOutput", "assessed_image", "PreviewImage", "images"),
    }
    for base_name, agentic_name in QUICKSTART_PAIRS:
        base = _workflow(base_name)
        agentic = _workflow(agentic_name)
        assert len(agentic["nodes"]) == len(base["nodes"]) + 2
        assert _typed_edges(agentic) - _typed_edges(base) == assessment_edges
        assert _typed_edges(base) < _typed_edges(agentic)
        assessor = next(node for node in agentic["nodes"]
                        if node["type"] == "AtlasAssessOutput")
        assert assessor["widgets_values"][5] == expected_paths[agentic_name]
        note = next(node for node in agentic["nodes"] if node["type"] == "Note")
        assert "AGENTIC TERMINAL QA" in note["widgets_values"][0]
        assert "retained" in note["widgets_values"][0].casefold()
        assert any(node["type"] == "PreviewImage" and
                   "ASSESSED EVIDENCE" in node.get("title", "")
                   for node in agentic["nodes"])


def test_staged_master_terminal_assessment_consumes_view_and_debug_summary():
    base = _staged()
    agentic = _staged_agentic()
    assert not any(node["type"] == "AtlasAssessOutput" for node in base["nodes"])
    assert len(agentic["nodes"]) == len(base["nodes"]) + 2
    assessor = next(node for node in agentic["nodes"]
                    if node["type"] == "AtlasAssessOutput")
    assert assessor["widgets_values"][0] is True
    assert assessor["widgets_values"][5] == (
        "atlas_debug/staged_master_agentic_output_assessment.json")
    assert next(node for node in agentic["nodes"]
                if node["type"] == "LoadImage")["widgets_values"][0] == (
                    "ghosttown.jpg")
    edges = _typed_edges(agentic)
    assert ("AtlasBlockoutViewport", "shaded",
            "AtlasAssessOutput", "camera_view") in edges
    assert ("AtlasDebugReport", "report",
            "AtlasAssessOutput", "solve_summary") in edges
    assert _typed_edges(agentic) - _typed_edges(base) == {
        ("AtlasBlockoutViewport", "shaded", "AtlasAssessOutput", "camera_view"),
        ("AtlasDebugReport", "report", "AtlasAssessOutput", "solve_summary"),
        ("dd9ef001-2246-5d12-88fa-4b305feb60d4", "solve",
         "AtlasAssessOutput", "solve"),
        ("AtlasRegisterPlate", "image", "AtlasAssessOutput", "source_image"),
        ("AtlasDepthMap", "depth", "AtlasAssessOutput", "depth"),
        ("AtlasAssessOutput", "assessed_image", "PreviewImage", "images"),
    }
    assert any(node["type"] == "PreviewImage" and
               "ASSESSED EVIDENCE" in node.get("title", "")
               for node in agentic["nodes"])


def test_occlusion_quickstart_is_a_one_wire_matched_depth_ab_test():
    pairs = (
        ("atlas_input_quickstart_workflow.json",
         "atlas_occlusion_cull_quickstart_workflow.json"),
        ("atlas_input_quickstart_agentic_assessment_workflow.json",
         "atlas_occlusion_cull_quickstart_agentic_assessment_workflow.json"),
    )
    for standard_name, occlusion_name in pairs:
        standard = _workflow(standard_name)
        occlusion = _workflow(occlusion_name)
        standard_edges = _typed_edges(standard)
        occlusion_edges = _typed_edges(occlusion)
        assert standard_edges < occlusion_edges
        assert occlusion_edges - standard_edges == {
            ("AtlasInput", "depth", "AtlasBlockoutViewport", "primary_depth")}
        assert not standard["extra"]["atlas_occlusion_primary_depth"]
        assert occlusion["extra"]["atlas_occlusion_primary_depth"]


@pytest.mark.parametrize("name,wf", _WORKFLOWS, ids=[n for n, _ in _WORKFLOWS])
def test_workflow_id_is_a_uuid(name, wf):
    """The ComfyUI frontend validates loaded workflows against a zod schema
    whose top-level `id` is `z.string().uuid().optional()` — a human-readable
    slug raises the yellow toast "Invalid workflow against zod schema:
    Validation error: Invalid uuid at \"id\"". The graph still loads (the
    frontend falls back to the unvalidated data, `validatedGraphData ??
    graphData`) but it silently SKIPS the `tryFixLinks` repair pass, so a
    rejected workflow also loses the link-integrity fixups every other
    workflow gets. Three shipped quickstarts carried hand-written slugs
    ("atlas-input-quickstart"), two of them identical — found live 2026-07-21
    against the frontend bundled with ComfyUI 0.3.49, reproduced with a
    positive control. The key is optional — omitting it lets the frontend
    generate one — but every workflow here ships a stable id, so it must be a
    real UUID."""
    wf_id = wf.get("id")
    if wf_id is None:
        return  # omitted is valid: the frontend auto-generates
    assert UUID_RE.match(str(wf_id)), (
        f"{name}: workflow id {wf_id!r} is not a UUID — the frontend's zod "
        f"schema will reject this workflow on load")


def test_workflow_ids_are_unique():
    """Two quickstarts shipped the SAME id ("atlas-input-quickstart"), which
    would collide in the frontend's workflow store once both became valid
    UUIDs. Distinct ids are what make a stable id worth having at all."""
    ids = [wf["id"] for _, wf in _WORKFLOWS if wf.get("id") is not None]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate workflow ids: {sorted(dupes)}"
