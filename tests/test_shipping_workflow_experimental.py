"""Shipping a gate-hidden experimental node in a workflow is a PINNED choice.

Experimental nodes (AtlasPredictHiddenGeometry, AtlasRenderFix, the angle-patch
pair) are hidden unless ``ATLAS_EXPERIMENTAL=1``. A workflow that uses one loads
with a red/missing node on a DEFAULT install — a Mac reviewer hit exactly this on
the SAM3-port debug graph and could not tell whether the port was broken.

That is sometimes intended (an X-ray / hidden-geometry demo genuinely uses
those nodes), so the rule is not "never ship one" — it is "every such workflow
is on this pinned list, and each MUST document that it needs
``ATLAS_EXPERIMENTAL=1`` (in INSTALL.md / the workflow's own README)." Adding
an experimental node to any shipped workflow fails this test, forcing the
question: is it intended, and is the flag documented?
"""
from __future__ import annotations

import json
from pathlib import Path

from atlas_camera.comfy import node_registry as reg

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTAL = set(reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS)

# Every shipped workflow that intentionally uses an experimental node. Each is
# documented as needing ATLAS_EXPERIMENTAL=1. Keep in sync when a workflow is
# added/removed — that is the point of pinning it.
#
# EMPTY since the 0.8.1 trim: the repo's three base/agentic example.png pairs
# do not touch an experimental node. All the X-ray /
# hidden-geometry / angle-patch showcases that did were removed (they need
# downloaded plates and live as website-distributed demos now). The guard
# stays so that re-introducing an experimental node into a SHIPPED workflow is
# a conscious, reviewed choice rather than an accidental red node on a default
# install.
WORKFLOWS_USING_EXPERIMENTAL: set[str] = set()


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
        "document ATLAS_EXPERIMENTAL=1 (INSTALL.md / the workflow's README) and "
        f"add them to this test's pinned set:\n  {sorted(added)}")
    assert not removed, (
        "pinned experimental-using workflow(s) no longer use one (or were "
        f"removed); update the pinned set:\n  {sorted(removed)}")
