"""Regression test: every registered ComfyUI node's function signature must match
its INPUT_TYPES declaration order.

ComfyUI serializes widget values positionally in saved workflows and passes
inputs to the node's FUNCTION method in the order they appear in INPUT_TYPES
(required -> optional -> hidden). A mismatch means saved workflows load wrong
values into wrong parameters. This test catches that class of bug centrally.
"""
from __future__ import annotations

import inspect
from inspect import Parameter

import pytest

from atlas_camera.comfy.node_registry import (
    EXPERIMENTAL_NODE_CLASS_MAPPINGS,
    NODE_CLASS_MAPPINGS,
)


def _input_names_in_order(input_types):
    """Return input names in the order ComfyUI serializes and passes them."""
    names = []
    for section in ("required", "optional", "hidden"):
        sec = input_types.get(section)
        if sec:
            names.extend(sec.keys())
    return names


def _function_param_names(func):
    """Return non-variadic parameter names in signature order, excluding `self`."""
    sig = inspect.signature(func)
    out = []
    for name, p in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if p.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue
        out.append(name)
    return out


def _normalize_for_compare(names):
    """Normalize names so laziness metadata does not mask real mismatches."""
    return [n for n in names]


@pytest.mark.parametrize("name,cls", list(NODE_CLASS_MAPPINGS.items()) + list(EXPERIMENTAL_NODE_CLASS_MAPPINGS.items()))
def test_node_signature_matches_input_types_order(name, cls):
    input_types = cls.INPUT_TYPES()
    input_names = _input_names_in_order(input_types)
    func = getattr(cls, cls.FUNCTION)
    param_names = _function_param_names(func)

    # Trim to the shorter length for ordered comparison; trailing **kwargs are fine.
    compare_len = min(len(input_names), len(param_names))
    input_compare = input_names[:compare_len]
    param_compare = param_names[:compare_len]

    if input_compare != param_compare:
        # Build a detailed diff so the failure points exactly at the drift.
        lines = [f"{name}: INPUT_TYPES order != {cls.FUNCTION}() parameter order"]
        lines.append("  INPUT_TYPES names:")
        for i, n in enumerate(input_names):
            marker = "  <<<" if i < len(param_names) and n != param_names[i] else ""
            lines.append(f"    [{i:2d}] {n}{marker}")
        lines.append(f"  {cls.FUNCTION}() parameter names:")
        for i, n in enumerate(param_names):
            marker = "  <<<" if i < len(input_names) and n != input_names[i] else ""
            lines.append(f"    [{i:2d}] {n}{marker}")
        if len(input_names) != len(param_names):
            lines.append(f"  lengths differ: INPUT_TYPES={len(input_names)} params={len(param_names)}")
        pytest.fail("\n".join(lines))

    # If function accepts **kwargs, any extra declared inputs are fine; otherwise
    # the counts must match (ComfyUI will pass every declared input).
    sig = inspect.signature(func)
    has_kwargs = any(p.kind == Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not has_kwargs and len(input_names) != len(param_names):
        pytest.fail(
            f"{name}: {len(input_names)} INPUT_TYPES inputs but {len(param_names)} "
            f"{cls.FUNCTION}() parameters (no **kwargs to absorb the difference). "
            f"Extra inputs: {input_names[len(param_names):] or param_names[len(input_names):]}"
        )
