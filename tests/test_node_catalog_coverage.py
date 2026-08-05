"""Every registered node must have a row in docs/NODE_CATALOG.md.

The catalog is hand-authored (rich per-node prose in a table), which is exactly
why it drifts: a node gets registered and nobody adds its row. This pins the
COMPLETENESS half mechanically without touching the prose — register a node and
its key has to show up in the catalog, in any tier. It does NOT check the reverse
(a stale row for a removed node) or the row's content; it is the cheap guard that
stops the gap this file was added to close from silently reopening.

Node names are read from the registry at runtime and never appear as literals
here, so the usage audit's text scan attributes this file to no node.
"""
from __future__ import annotations

from pathlib import Path

from atlas_camera.comfy import node_registry as reg

CATALOG = Path(__file__).resolve().parents[1] / "docs" / "NODE_CATALOG.md"


def _registered_keys() -> set[str]:
    keys = set(reg.NODE_CLASS_MAPPINGS)
    keys |= set(reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS)
    keys |= set(getattr(reg, "LEGACY_NODE_CLASS_MAPPINGS", {}))
    keys |= set(getattr(reg, "IOS_NODE_CLASS_MAPPINGS", {}))
    return keys


def test_every_registered_node_is_in_the_catalog():
    text = CATALOG.read_text(encoding="utf-8")
    missing = sorted(k for k in _registered_keys() if f"`{k}`" not in text)
    assert missing == [], (
        "registered nodes with no row in docs/NODE_CATALOG.md "
        f"(add one each): {missing}")
