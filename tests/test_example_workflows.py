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
    # Shipping catalog: the quickstart + the staged master (2026-07-12), the
    # OCIO/ACEScg EXR-handoff quickstart (2026-07-13), and the occlusion-cull
    # quickstart (2026-07-20 - the ✂ Occlude / primary_depth demo). Pins exactly
    # these, so an accidental deletion OR an unreviewed addition both fail loudly.
    # The trimmed-out examples live in git history (< 10e600b).
    names = sorted(n for n, _ in _WORKFLOWS)
    assert names == ["atlas_camera_staged_master_workflow.json",
                     "atlas_input_ocio_quickstart_workflow.json",
                     "atlas_input_quickstart_workflow.json",
                     "atlas_occlusion_cull_quickstart_workflow.json"]


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


def test_staged_master_rails_have_setters():
    """Every KJ GetNode's rail name must have exactly one SetNode — a renamed
    or deleted Set silently starves every Get on that rail."""
    wf = _staged()
    sets = {}
    for n in wf["nodes"]:
        if n["type"] == "SetNode":
            rail = n["widgets_values"][0]
            sets.setdefault(rail, 0)
            sets[rail] += 1
    dupes = [r for r, c in sets.items() if c > 1]
    assert not dupes, f"duplicate SetNodes for rails: {dupes}"
    orphans = [n["widgets_values"][0] for n in wf["nodes"]
               if n["type"] == "GetNode" and n["widgets_values"][0] not in sets]
    assert not orphans, f"GetNodes with no SetNode: {orphans}"


def test_staged_master_debug_strip_is_group_free():
    """The per-layer previews + the 🔍 debug node live OUTSIDE every group
    bounding on purpose: a node fully inside a stage group's bounds gets
    claimed by the rgthree bypasser and dies with that group — the exact
    trap the first preview placement fell into (inside the ASSEMBLE group)."""
    wf = _staged()

    def inside(n, g):
        x, y = n["pos"]
        w, h = n.get("size") or [210, 246]
        gx, gy, gw, gh = g["bounding"]
        return gx <= x and gy <= y and x + w <= gx + gw and y + h <= gy + gh

    strip_types = {"AtlasLayerPreview", "AtlasDebugReport"}
    claimed = []
    for n in wf["nodes"]:
        if n["type"] in strip_types and n["pos"][0] < 2010:  # the x1760 strip
            for g in wf["groups"]:
                if inside(n, g):
                    claimed.append((n["id"], n["type"], g["title"][:30]))
    assert not claimed, f"strip nodes claimed by groups: {claimed}"


def test_staged_master_band_priorities_are_farthest_highest():
    """DMP seam doctrine (2026-07-12, from the quickstart's striped-seam
    fix): at a watertight band seam the two surfaces are depth-adjacent and
    the priority near-tie bias decides which paints — farthest-highest makes
    the layer BEHIND win the seam ribbon, so a band's edge smear can never
    render in front of the layer behind it. Keeps the staged master and
    AtlasInput on the same convention."""
    wf = _staged()
    prios = {n["widgets_values"][4]: n["widgets_values"][5]
             for n in wf["nodes"] if n["type"] == "AtlasCleanPlateLayer"}
    assert prios == {"band_far": 15, "band_bg": 10, "band_mid": 5, "band_fg": 0}


def test_staged_master_scope_rows_are_always_active():
    """v7 doctrine: 🎯 AtlasScopeMask rows self-disarm — none may ship
    bypassed (mode 4) or muted (mode 2), and there must be one per band."""
    wf = _staged()
    scopes = [n for n in wf["nodes"] if n["type"] == "AtlasScopeMask"]
    assert len(scopes) == 4
    assert all(n.get("mode", 0) == 0 for n in scopes)


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
