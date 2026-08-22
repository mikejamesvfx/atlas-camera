"""The node counts written in prose must match the registry.

They are maintained by hand across CLAUDE.md, three docs and two pin tests, so
adding one node means editing six files. The registry is the only authority;
this catches the file someone forgets.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from atlas_camera.comfy import node_registry as registry

ROOT = Path(__file__).parents[1]

DOCS = (
    "CLAUDE.md",
    "docs/NODE_CATALOG.md",
    "docs/ECOSYSTEM_GUIDE.md",
)

_COUNT = re.compile(
    r"(\d+) standard \+ (\d+) experimental \+ (\d+) legacy \+ (\d+) iOS"
)


def _registry_counts() -> tuple[int, int, int, int]:
    return (
        len(registry.NODE_CLASS_MAPPINGS),
        len(registry.EXPERIMENTAL_NODE_CLASS_MAPPINGS),
        len(getattr(registry, "LEGACY_NODE_CLASS_MAPPINGS", {})),
        len(getattr(registry, "IOS_NODE_CLASS_MAPPINGS", {})),
    )


@pytest.mark.parametrize("relative", DOCS)
def test_documented_tier_counts_match_the_registry(relative):
    text = (ROOT / relative).read_text(encoding="utf-8", errors="replace")
    found = _COUNT.findall(text)
    assert found, f"{relative} states no node tier counts; the pattern moved"
    expected = _registry_counts()
    for match in found:
        assert tuple(int(value) for value in match) == expected, (
            f"{relative} is stale: says {match}, registry has {expected}"
        )


@pytest.mark.parametrize("relative", DOCS)
def test_documented_totals_match_the_sum(relative):
    text = (ROOT / relative).read_text(encoding="utf-8", errors="replace")
    total = sum(_registry_counts())
    for match in re.findall(r"= (\d+)\s*\n?registered", text):
        assert int(match) == total, f"{relative} totals {match}, registry sums to {total}"
