"""Shipping a gate-hidden experimental node in a workflow is a PINNED choice.

Experimental nodes (AtlasPredictHiddenGeometry, AtlasRenderFix, the angle-patch
pair) are hidden unless ``ATLAS_EXPERIMENTAL=1``. A workflow that uses one loads
with a red/missing node on a DEFAULT install — a Mac reviewer hit exactly this on
the SAM3-port debug graph and could not tell whether the port was broken.

That is sometimes intended (the X-ray / hidden-geometry showcases genuinely
demonstrate those nodes), so the rule is not "never ship one" — it is "every
such workflow is on this pinned list, and each MUST document that it needs
``ATLAS_EXPERIMENTAL=1`` (see examples/showcase/README.md's requirements
matrix)." Adding an experimental node to any other workflow fails this test,
forcing the question: is it intended, and is the flag documented?
"""
from __future__ import annotations

import json
from pathlib import Path

from atlas_camera.comfy import node_registry as reg

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTAL = set(reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS)

# Every shipped workflow that intentionally uses an experimental node. Each is
# documented as needing ATLAS_EXPERIMENTAL=1. Keep in sync when a showcase is
# added/removed — that is the point of pinning it.
WORKFLOWS_USING_EXPERIMENTAL = {
    "experimental/atlas_jungle_xray_cameramove.json",
    "showcase/atlas_canonical_cleanplate_ghosttown_workflow.json",
    "showcase/atlas_canonical_research_newyork_lari_workflow.json",
    "showcase/atlas_dmp_angle_xray_newyork_workflow.json",
    "showcase/atlas_segmented_sdxl_hidden_d810raw_workflow.json",
    "showcase/atlas_segmented_sdxl_manual_debug_workflow.json",
    "showcase/atlas_xray_wreck_workflow.json",
}


def _actual() -> set[str]:
    found = set()
    for p in sorted((ROOT / "examples").rglob("*.json")):
        try:
            g = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not (isinstance(g, dict) and isinstance(g.get("nodes"), list)):
            continue
        if any(n.get("type") in EXPERIMENTAL for n in g["nodes"]):
            found.add(p.relative_to(ROOT / "examples").as_posix())
    return found


def test_experimental_node_usage_is_pinned():
    actual = _actual()
    added = actual - WORKFLOWS_USING_EXPERIMENTAL
    removed = WORKFLOWS_USING_EXPERIMENTAL - actual
    assert not added, (
        "workflow(s) now use a gate-hidden experimental node — intended? then "
        "document ATLAS_EXPERIMENTAL=1 in examples/showcase/README.md and add "
        f"them to this test's pinned set:\n  {sorted(added)}")
    assert not removed, (
        "pinned experimental-using workflow(s) no longer use one (or were "
        f"removed); update the pinned set:\n  {sorted(removed)}")
