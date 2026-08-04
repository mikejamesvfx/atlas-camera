"""ATLAS_LEGACY_NODES — the superseded tier.

A legacy node is not deleted: its key and display name stay byte-identical and
the class stays importable, so a saved graph resolves for one migration cycle
when the operator opts in. What changes is that the default node menu offers
one obvious way to do each job.

Modelled on the ATLAS_EXPERIMENTAL gate test, including its reload teardown —
the gate is evaluated at import time, so a test that reloads the registry must
put it back or every later test sees a polluted mapping.
"""
from __future__ import annotations

import importlib

import pytest

from atlas_camera.comfy import node_registry as registry
from atlas_camera.comfy import nodes

LEGACY_KEYS = {"AtlasLiveMeshRepair", "AtlasGroundMask", "AtlasAddPlanePolygon"}


def test_legacy_tier_is_closed_by_default():
    assert nodes.ATLAS_LEGACY_DEFAULT == "0"
    assert set(nodes.LEGACY_NODE_CLASS_MAPPINGS) == LEGACY_KEYS
    assert not (LEGACY_KEYS & set(nodes.NODE_CLASS_MAPPINGS))
    assert not (LEGACY_KEYS & set(nodes.NODE_DISPLAY_NAME_MAPPINGS))


def test_legacy_classes_stay_importable_while_gated_out():
    """Gated out of the MENU, not out of the package — an import-time removal
    would break any downstream code holding a reference."""
    from atlas_camera.comfy.nodes import AtlasGroundMask, AtlasLiveMeshRepair  # noqa: F401


def test_setting_the_env_var_registers_them(monkeypatch):
    monkeypatch.setenv("ATLAS_LEGACY_NODES", "1")
    try:
        importlib.reload(registry)
        assert LEGACY_KEYS <= set(registry.NODE_CLASS_MAPPINGS)
        assert LEGACY_KEYS <= set(registry.NODE_DISPLAY_NAME_MAPPINGS)
        # Keys AND display names must be unchanged — the append-only contract
        # covers renames, so a saved graph must match byte for byte.
        assert registry.NODE_DISPLAY_NAME_MAPPINGS["AtlasLiveMeshRepair"] == \
            "Atlas Live Mesh Repair 🔧"
        assert registry.NODE_DISPLAY_NAME_MAPPINGS["AtlasGroundMask"] == \
            "Atlas Ground Mask"
    finally:
        monkeypatch.delenv("ATLAS_LEGACY_NODES", raising=False)
        importlib.reload(registry)
        importlib.reload(nodes)


def test_tiers_are_disjoint():
    assert not (set(nodes.LEGACY_NODE_CLASS_MAPPINGS)
                & set(nodes.EXPERIMENTAL_NODE_CLASS_MAPPINGS))
    assert not (set(nodes.LEGACY_NODE_CLASS_MAPPINGS)
                & set(nodes.NODE_CLASS_MAPPINGS))


def test_every_legacy_node_names_a_registered_replacement():
    """A replacement that was itself culled would strand the user."""
    assert set(nodes.LEGACY_REPLACEMENTS) == LEGACY_KEYS
    for key, target in nodes.LEGACY_REPLACEMENTS.items():
        head = target.split()[0].strip("(),")
        assert head in nodes.NODE_CLASS_MAPPINGS, (
            f"{key}: replacement {head!r} is not a registered standard node")


def test_deprecation_is_visible_on_the_solve(monkeypatch):
    """No STRING output exists and adding one would break the positional
    output contract, so the notice rides on the solve where AtlasDebugReport
    can surface it."""
    pytest.importorskip("torch")
    import numpy as np

    from atlas_camera.comfy.nodes import AtlasLiveMeshRepair
    from atlas_camera.core.camera_math import look_at_view_matrix
    from atlas_camera.core.schema import (
        AtlasExtrinsics, AtlasIntrinsics, AtlasSolve, LatentCamera)

    view, world, rot3 = look_at_view_matrix((0.0, 1.6, 0.0), (0.0, 0.5, -10.0))
    solve = AtlasSolve(camera=LatentCamera(
        intrinsics=AtlasIntrinsics(image_width=64, image_height=48, fx_px=60.0,
                                   fy_px=60.0, cx_px=32.0, cy_px=24.0),
        extrinsics=AtlasExtrinsics(camera_position=(0.0, 1.6, 0.0),
                                   camera_rotation_matrix=rot3,
                                   camera_world_matrix=world,
                                   camera_view_matrix=view)))
    out, = AtlasLiveMeshRepair().repair(solve)
    notices = out.debug_metadata.get("atlas_deprecations") or []
    assert notices and "SUPERSEDED" in notices[0]
    assert "AtlasPlanarHolePatch" in notices[0]
    assert all(isinstance(n, str) for n in notices)   # solve JSON is a contract
