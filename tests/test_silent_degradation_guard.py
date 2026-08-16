"""A node that degrades must be able to SAY it degraded.

Four nodes were fixed by hand on 2026-08-17 for the same defect, all with the
same trigger: `_metric_depth_and_validity` returns None when the solve carries
no focal length, and the node returned all-ZERO masks with no report. An
all-zero `hole_mask` asserts "no holes anywhere", which is exactly what a
FLAWLESS layer produces — so a node that built nothing was indistinguishable
downstream (AtlasPlanarHolePatch reads that mask; so does the inpaint router)
from the best possible one.

AtlasBoundedBand, the smallest of the callers, had it right from the start.
The convention existed; nothing enforced it. This is the enforcement, so the
fifth instance fails in CI instead of being found by reading 1,200 lines.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO / "atlas_camera" / "comfy"

#: Helpers that signal "I could not measure this scene" by returning None.
#: A caller that ignores that and returns data-shaped-like-success is the bug.
DEGRADING_HELPERS = {"_metric_depth_and_validity", "_solve_camera_params"}


def _classes_calling_a_degrading_helper():
    """(module, class, helper, RETURN_TYPES) for every node that can degrade."""
    out = []
    for py in sorted(COMFY.glob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            called = {
                n.func.id for n in ast.walk(cls)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in DEGRADING_HELPERS
            }
            if not called:
                continue
            rt = None
            for s in cls.body:
                if isinstance(s, ast.Assign) and any(
                        getattr(t, "id", "") == "RETURN_TYPES" for t in s.targets):
                    try:
                        rt = ast.literal_eval(s.value)
                    except ValueError:  # pragma: no cover
                        rt = None
            out.append((py.name, cls.name, sorted(called), rt))
    return out


def test_the_guard_has_something_to_check():
    """A rename must not silently turn this file into a no-op."""
    found = _classes_calling_a_degrading_helper()
    assert found, (
        f"no node calls any of {sorted(DEGRADING_HELPERS)} — were they renamed? "
        "Update DEGRADING_HELPERS rather than leaving this test vacuous."
    )


@pytest.mark.parametrize(
    "module,cls,helpers,return_types",
    _classes_calling_a_degrading_helper(),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_a_node_that_can_degrade_can_report(module, cls, helpers, return_types):
    assert return_types is not None, f"{module}:{cls} has no RETURN_TYPES"
    assert "STRING" in return_types, (
        f"{module}:{cls} calls {', '.join(helpers)}, which returns None when the "
        f"solve has no focal length — but its outputs are {return_types}, with no "
        "STRING to say so. Append a `report` output (LAST, so existing slots keep "
        "their index) and name the reason. If a mask is among the outputs, check "
        "what all-zero CLAIMS: for a hole/coverage mask that is 'perfect result', "
        "so a failed run must return ones, not zeros."
    )
