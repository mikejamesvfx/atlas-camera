"""Characterization tests pinning the ComfyUI node registry surface.

These lock the public contract of ``atlas_camera.comfy.nodes`` so the
mechanical split of ``nodes.py`` into responsibility modules cannot silently
change a registered node key, a display name, the experimental gate, or a
public import. They assert the *current* behavior verbatim — if any of these
fail during the refactor, the split changed the contract and must be fixed.
"""
from __future__ import annotations

import importlib
import os

import atlas_camera.comfy.nodes as nodes


# The exact registered node keys at the time of the nodes.py modularization
# (91 standard + 6 experimental + 2 legacy). Grown since the nodes.py
# modularization; the SET below is the contract, the count just guards it. ComfyUI serializes these keys into saved
# workflows, so this set is a compatibility contract, not an implementation
# detail.
NORMAL_KEYS = {
    "AtlasAddPatchView", "AtlasApplyScaleReferences", "AtlasAssessImage",
    "AtlasAssessOutput",
    "AtlasLayerPlan",
    "AtlasMoveBudget",
    "AtlasOcclusionGraph",
    "AtlasAttachSourcePlate", "AtlasBlockoutViewport", "AtlasBoundedBand",
    "AtlasCleanPlateLayer", "AtlasCleanPlateStack", "AtlasConstrainedSolve",
    "AtlasDeband", "AtlasDisocclusionGuide", "AtlasSolvePatchViews",
    "AtlasApplyLUT", "AtlasDefocus", "AtlasGrade",
    "AtlasDebugReport", "AtlasDecomposeCamera", "AtlasDecomposeSolve",
    "AtlasDefineShotCam", "AtlasDepthAnything", "AtlasDepthBandSplit",
    "AtlasDepthCombine", "AtlasDepthDetailEnhance",
    "AtlasDepthLayerMask", "AtlasDepthMap", "AtlasDepthOutlierMask",
    "AtlasDeriveInteriorRoom", "AtlasDeriveProjectionGeometry",
    "AtlasDeriveReliefMesh", "AtlasDeriveRoofsFacades", "AtlasDeriveTowersSpires",
    "AtlasDeriveWalls", "AtlasExportBlender", "AtlasExportCameraPathUSD",
    "AtlasExportPlateEXR",
    "AtlasExportMayaLayers", "AtlasExportMayaReviewScene", "AtlasExportNuke",
    "AtlasExportNukeLayers", "AtlasExportReliefMesh", "AtlasExportReviewPackage",
    "AtlasExportSolveJSON", "AtlasExportUSD", "AtlasGravityOverride",
    "AtlasGravityCompass",
    "AtlasGroundDepthMap", "AtlasHorizonMask",
    "AtlasInpaintCrop", "AtlasInpaintStitch", "AtlasInput", "AtlasInstanceMask",
    "AtlasLayerPreview", "AtlasLearnedSolveFromImage",     "AtlasFaceScaleReference", "AtlasLoadRAW", "AtlasLoadRecord3D", "AtlasSplitEquirect", "AtlasEquirectMultiView", "AtlasLoadSolveJSON", "AtlasMergeGeometry", "AtlasMogeNormals",
    "AtlasOcclusionMask", "AtlasPathGuidedHoleRepair", "AtlasPlanarHolePatch",
    "AtlasReferenceScaleSolve",
    "AtlasLoadPlate",
    "AtlasRegisterPlate", "AtlasRetopologizeLayer", "AtlasRollTrim", "AtlasSAM3Mask", "AtlasSDXLInpaint",
    "AtlasScaleOverride",
    "AtlasSceneHealthGate", "AtlasScopeMask", "AtlasSegmentedSDXLInpaint",
    "AtlasSemanticMask", "AtlasSkyDomeLayer", "AtlasSolveFromImage", "AtlasStereoRender",
    "AtlasSolveGate", "AtlasStreamRecord3D", "AtlasOutpaintDepth", "AtlasShootList",
    "AtlasUSDCameraLoader", "AtlasVLMScaleCues",
    "AtlasVPVisualization", "AtlasViewportControls",
}

EXPERIMENTAL_KEYS = {
    "AtlasBlockoutMassing",
    "AtlasMaskedSurfaceReconstruct",
    "AtlasRefineOcclusionSeams",
    "AtlasCompleteDepth",
    "AtlasExtractAnglePatch", "AtlasImportAnglePatch",
}

# Public helper/constant names some tests import directly from the module; the
# compatibility façade must keep re-exporting them.
FACADE_HELPER_NAMES = (
    "_ATLAS_BLOCKOUT_CACHE", "_image_fingerprint", "_solve_fingerprint",
    "_b64_png_to_mask", "_parse_view_prompt", "_parse_exact_view",
    "_parse_band_override", "_flood_mask_to_frame_borders", "_resolve_depth_band",
    "_relief_mesh_from_solve", "_resize_normal_field", "_write_export_manifest",
    "_scale_summary_suffix", "_health_summary_suffix",
)


def test_normal_registry_keys_exact():
    assert set(nodes.NODE_CLASS_MAPPINGS) == NORMAL_KEYS
    assert len(nodes.NODE_CLASS_MAPPINGS) == 91


def test_experimental_registry_keys_exact():
    assert set(nodes.EXPERIMENTAL_NODE_CLASS_MAPPINGS) == EXPERIMENTAL_KEYS
    assert len(nodes.EXPERIMENTAL_NODE_CLASS_MAPPINGS) == 6


def test_display_name_mapping_covers_registry():
    # Every registered normal node has a display name, and no extras.
    assert set(nodes.NODE_DISPLAY_NAME_MAPPINGS) == NORMAL_KEYS


def test_mapping_values_are_the_registered_classes():
    # The class object under each key must expose the ComfyUI contract.
    for key, cls in nodes.NODE_CLASS_MAPPINGS.items():
        assert hasattr(cls, "INPUT_TYPES"), key
        assert hasattr(cls, "RETURN_TYPES"), key
        assert hasattr(cls, "FUNCTION"), key
        assert hasattr(cls, "CATEGORY"), key


def test_experimental_gate_off_by_default():
    # Default install ships the gate closed: experimental keys are NOT merged
    # into the standard registry, and stay in their own mapping.
    assert nodes.ATLAS_EXPERIMENTAL_DEFAULT == "0"
    assert not (EXPERIMENTAL_KEYS & set(nodes.NODE_CLASS_MAPPINGS))


def test_experimental_gate_merges_when_enabled(monkeypatch):
    # With ATLAS_EXPERIMENTAL truthy, a fresh import of the registration module
    # (where the gate + dict literals live post-modularization) merges the 4
    # experimental nodes into the standard registry.
    import atlas_camera.comfy.node_registry as registry
    monkeypatch.setenv("ATLAS_EXPERIMENTAL", "1")
    importlib.reload(registry)
    try:
        assert EXPERIMENTAL_KEYS <= set(registry.NODE_CLASS_MAPPINGS)
        assert EXPERIMENTAL_KEYS <= set(registry.NODE_DISPLAY_NAME_MAPPINGS)
    finally:
        monkeypatch.delenv("ATLAS_EXPERIMENTAL", raising=False)
        importlib.reload(registry)  # rebuild the default (gate-off) dicts
        importlib.reload(nodes)     # rebind the façade to the restored mappings


def test_representative_public_class_imports():
    from atlas_camera.comfy.nodes import (  # noqa: F401
        AtlasDepthMap, AtlasLearnedSolveFromImage, AtlasBlockoutViewport,
        AtlasExportNukeLayers, AtlasCleanPlateLayer, AtlasMergeGeometry,
        AtlasRollTrim, AtlasInput, AtlasAssessOutput,
    )
    # Experimental classes are importable as symbols even when gated out.
    from atlas_camera.comfy.nodes import (  # noqa: F401
        AtlasExtractAnglePatch, AtlasImportAnglePatch,
    )


def test_facade_reexports_public_helpers():
    for name in FACADE_HELPER_NAMES:
        assert hasattr(nodes, name), name


def test_signature_defaults_match_declared_defaults():
    """A node's Python default must equal the default it advertises.

    These diverge silently and the symptom depends on the CALLER. ComfyUI's UI
    serialises every widget, so a graph built in the browser always sends a
    value and the signature default never fires. An API caller that omits an
    optional input gets the SIGNATURE default instead — so the same node quietly
    behaves differently depending on how it was invoked.

    Found live 2026-07-28: `AtlasEquirectMultiView.n_views` advertised 4 but its
    signature still said 12, because only INPUT_TYPES was updated when the
    measured default changed. A headless run did 12 depth passes instead of 4 —
    three times the work, with nothing reporting anything wrong.
    """
    import inspect

    from atlas_camera.comfy import nodes

    mismatches = []
    for key, cls in nodes.NODE_CLASS_MAPPINGS.items():
        fn = getattr(cls, getattr(cls, "FUNCTION", ""), None)
        if fn is None:
            continue
        try:
            params = inspect.signature(fn).parameters
            spec = cls.INPUT_TYPES()
        except Exception:                      # noqa: BLE001 - not introspectable
            continue

        declared = {}
        for section in ("required", "optional"):
            for name, entry in (spec.get(section) or {}).items():
                if isinstance(entry, (list, tuple)) and len(entry) > 1 \
                        and isinstance(entry[1], dict) and "default" in entry[1]:
                    declared[name] = entry[1]["default"]

        for name, want in declared.items():
            p = params.get(name)
            if p is None or p.default is inspect.Parameter.empty:
                continue
            # `None` is a deliberate sentinel in several nodes: the body does
            # `if x is None: x = THE_DEFAULT`, which resolves to the SAME
            # advertised value. Flagging it would force a worse signature for
            # no behavioural gain (e.g. AtlasFaceScaleReference.stature_m).
            if p.default is None:
                continue
            if p.default != want:
                mismatches.append(f"{key}.{name}: signature={p.default!r} declared={want!r}")

    assert not mismatches, "signature/INPUT_TYPES default mismatch:\n  " + "\n  ".join(mismatches)


def test_every_node_sits_in_the_two_tier_atlas_menu():
    """CATEGORY is exactly "Atlas" or "Atlas/advanced" — the 0.8.0 front door.

    Before 0.8.0 the pack spread 91 registered nodes across 13
    "Atlas Camera/<area>" sub-menus. Searching "Atlas" returned all of them
    with nothing marking where to start, and 53 of the 91 appeared in no
    shipped example workflow at all. The rule now: a node a shipped workflow
    actually touches is front-door ("Atlas"); everything else, including every
    experimental- and legacy-tier node, is "Atlas/advanced".

    This is metadata only. ComfyUI serializes the registry KEY into a saved
    graph, never the category, so recategorizing cannot break a saved
    workflow — which is what makes it the one restructure available under the
    append-only rule. The keys and display names themselves stay pinned by
    the tests above.
    """
    import atlas_camera.comfy.nodes as nodes
    from atlas_camera.comfy import node_registry as reg

    everything = {**reg.NODE_CLASS_MAPPINGS,
                  **reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS,
                  **(getattr(reg, "LEGACY_NODE_CLASS_MAPPINGS", {}) or {})}
    allowed = {"Atlas", "Atlas/advanced"}

    wrong = {key: getattr(cls, "CATEGORY", None)
             for key, cls in everything.items()
             if getattr(cls, "CATEGORY", None) not in allowed}
    assert not wrong, f"nodes outside the two-tier menu: {wrong}"

    # A front door with nothing behind it, or everything behind it, is not a
    # front door. Guard both directions rather than pinning an exact count,
    # which would go stale the moment a workflow lands.
    front = [k for k, c in everything.items() if c.CATEGORY == "Atlas"]
    assert 10 < len(front) < len(everything) / 2, (
        f"front door holds {len(front)} of {len(everything)} nodes")
    assert "AtlasInput" in front, "AtlasInput is the front door and must be in it"

    # Tier gating and menu tier are independent axes, and an experimental node
    # must never present as supported no matter which workflow uses it.
    for key in reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS:
        assert everything[key].CATEGORY == "Atlas/advanced", key
