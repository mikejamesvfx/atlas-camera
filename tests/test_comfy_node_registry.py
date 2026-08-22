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
# 93 standard + 7 experimental + 2 legacy + 2 iOS. The set below is the
# modularization; the SET below is the contract, the count just guards it. ComfyUI serializes these keys into saved
# workflows, so this set is a compatibility contract, not an implementation
# detail.
NORMAL_KEYS = {
    "AtlasProject",
    "AtlasMultiViewSolve",
    "AtlasMultiViewSolveBurst",
    "AtlasSolveBurstPatchCrops",
    "AtlasOpenRealPlate", "AtlasReadLockedPlatePlan", "AtlasRecordPlateAttempt",
    "AtlasExportPlateHandoff", "AtlasRealPlateToScene",
    "AtlasAddPatchView", "AtlasApplyScaleReferences", "AtlasAssessImage",
    "AtlasAssessOutput",
    "AtlasLayerPlan",
    "AtlasMoveBudget",
    "AtlasOcclusionGraph",
    "AtlasAttachSourcePlate", "AtlasBlockoutViewport", "AtlasBoundedBand",
    "AtlasCleanPlateLayer", "AtlasCleanPlateStack", "AtlasPlateLayer", "AtlasConstrainedSolve",
    "AtlasDeband", "AtlasDisocclusionGuide", "AtlasSolvePatchViews",
    "AtlasApplyLUT", "AtlasDefocus", "AtlasGrade",
    "AtlasDebugReport", "AtlasDecomposeCamera", "AtlasDecomposeSolve",
    "AtlasDefineShotCam", "AtlasDepthAnything", "AtlasDepthBandSplit",
    "AtlasDepthCombine", "AtlasDepthDetailEnhance",
    "AtlasDepthLayerMask", "AtlasDepthMap", "AtlasDepthOutlierMask",
    "AtlasFitDepthCalibration", "AtlasApplyDepthCalibration",
    "AtlasDeriveInteriorRoom", "AtlasDeriveProjectionGeometry",
    "AtlasDeriveReliefMesh", "AtlasDeriveRoofsFacades", "AtlasDeriveTowersSpires",
    "AtlasDeriveWalls", "AtlasExportBlender", "AtlasExportCameraPathUSD",
    "AtlasExportPlateEXR",
    "AtlasExportMayaLayers", "AtlasExportMayaReviewScene", "AtlasExportNuke",
    "AtlasExportNukeLayers", "AtlasExportReliefMesh", "AtlasExportReviewPackage",
    "AtlasExportScenePackage",
    "AtlasExportSolveJSON", "AtlasExportUSD", "AtlasGravityOverride",
    "AtlasGravityCompass",
    "AtlasGroundDepthMap", "AtlasHorizonMask",
    "AtlasCardMask", "AtlasPlaneMattes", "AtlasInpaintCrop", "AtlasInpaintStitch", "AtlasInput",
    "AtlasInstanceMask",
    "AtlasLayerPreview", "AtlasLearnedSolveFromImage",     "AtlasFaceScaleReference", "AtlasLoadRAW", "AtlasSplitEquirect", "AtlasEquirectMultiView", "AtlasLoadSolveJSON", "AtlasMergeGeometry", "AtlasGroundPlane", "AtlasMogeNormals",
    "AtlasOcclusionMask", "AtlasPathGuidedHoleRepair", "AtlasPlanarHolePatch",
    "AtlasReferenceScaleSolve",
    "AtlasLoadPlate",
    "AtlasRegisterPlate", "AtlasRetopologizeLayer", "AtlasRollTrim", "AtlasSAM3Mask", "AtlasSDXLInpaint",
    "AtlasScaleOverride",
    "AtlasSceneHealthGate", "AtlasScopeMask", "AtlasSegmentedSDXLInpaint",
    "AtlasSemanticMask", "AtlasSkyDomeLayer", "AtlasSolveFromImage", "AtlasStereoRender",
    "AtlasSolveGate", "AtlasOutpaintDepth", "AtlasShootList",
    "AtlasUSDCameraLoader", "AtlasVLMScaleCues",
    "AtlasVPVisualization", "AtlasViewportControls",
    # Promoted from the experimental tier 2026-08-14 (two-pass fill engine):
    "AtlasInterpassGate", "AtlasMembraneComposite", "AtlasCropROI",
    "AtlasCompositeCrop", "AtlasCameraMovePreset",
    # 2026-08-16: photo crop for the Qwen ROI loop.
    "AtlasCropSourcePhoto",
    # Promoted 2026-08-14 (Dynamic Plates: the CLI half was never gated):
    "AtlasLoadDynamicPlate",
    # Promoted 2026-08-14 (serves the standard two-pass fill engine):
    "AtlasPathFrameIndex",
}

EXPERIMENTAL_KEYS = {
    "AtlasBlockoutMassing",
    "AtlasMaskedSurfaceReconstruct",
    "AtlasRefineOcclusionSeams",
    "AtlasCompleteDepth",
    "AtlasExtractAnglePatch", "AtlasImportAnglePatch",
    # Research bridge for the VolFill hidden-geometry evaluation (2026-08-15):
    # meshes an external amodal VOLUME into the solve so the viewport can show
    # it. Experimental-tier because the volume->layered-rays adapter that would
    # feed core/hidden_geometry.py is not measured yet.
    "AtlasLoadHiddenVolume",
    # Measured-primitives Blender bridge (2026-08-16): headless massing +
    # mesh import. Experimental: needs an external Blender install.
    "AtlasBlenderMassing", "AtlasBlenderImportMeshes",
    # 2026-08-16: pause/brief/resume contract for an external agent.
    "AtlasAgentHandoff",
}

# iOS / Record3D capture tier — gated behind ATLAS_IOS, held out of the v1
# default menu (a v2 capability). Keys stay byte-identical to when they were in
# the standard tier so a saved graph still resolves once the flag is set.
IOS_KEYS = {
    "AtlasLoadRecord3D",
    "AtlasStreamRecord3D",
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
    assert len(nodes.NODE_CLASS_MAPPINGS) == 113


def test_experimental_registry_keys_exact():
    assert set(nodes.EXPERIMENTAL_NODE_CLASS_MAPPINGS) == EXPERIMENTAL_KEYS
    assert len(nodes.EXPERIMENTAL_NODE_CLASS_MAPPINGS) == 10


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


def test_ios_registry_keys_exact():
    assert set(nodes.IOS_NODE_CLASS_MAPPINGS) == IOS_KEYS
    assert len(nodes.IOS_NODE_CLASS_MAPPINGS) == 2


def test_ios_gate_off_by_default():
    # v1 ships the iOS/Record3D capture tier gated closed: the keys are NOT in
    # the standard registry and stay in their own mapping.
    assert nodes.ATLAS_IOS_DEFAULT == "0"
    assert not (IOS_KEYS & set(nodes.NODE_CLASS_MAPPINGS))


def test_ios_gate_merges_when_enabled(monkeypatch):
    # With ATLAS_IOS truthy, a fresh import merges the 2 capture nodes into the
    # standard registry — same shape as the experimental/legacy gates.
    import atlas_camera.comfy.node_registry as registry
    monkeypatch.setenv("ATLAS_IOS", "1")
    importlib.reload(registry)
    try:
        assert IOS_KEYS <= set(registry.NODE_CLASS_MAPPINGS)
        assert IOS_KEYS <= set(registry.NODE_DISPLAY_NAME_MAPPINGS)
    finally:
        monkeypatch.delenv("ATLAS_IOS", raising=False)
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
        AtlasLoadDynamicPlate, AtlasInterpassGate,
        AtlasMembraneComposite, AtlasPathFrameIndex,
        AtlasCropROI, AtlasCompositeCrop, AtlasCameraMovePreset,
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


def test_every_node_sits_in_a_numbered_pipeline_folder():
    """CATEGORY places every node in a pipeline-stage folder from one central map.

    0.8.0 collapsed 13 scattered "Atlas Camera/<area>" sub-menus into two flat
    tiers ("Atlas" / "Atlas/advanced"). That gave a starting point but replaced
    it with two unscannable, unordered walls of ~40 and ~55 nodes. The menu now
    groups nodes by pipeline stage across ten numbered folders (01 Input &
    Camera ... 10 Export), with every gated tier (experimental, legacy, iOS) in
    "Atlas/advanced". The zero-padded numeric prefixes force workflow order,
    since ComfyUI sorts folders as strings.

    node_registry.MENU_CATEGORY is the single source of truth: one map, stamped
    onto every class at import. This pins that the map covers exactly the
    registered nodes, that each class carries its mapped folder, and that gated
    tiers stay in "advanced".

    CATEGORY is metadata only — ComfyUI serializes the registry KEY into a saved
    graph, never the category — so re-foldering cannot break a saved workflow,
    which is what makes it the one restructure the append-only key/display-name
    contract permits. Keys and display names stay pinned by the tests above.
    """
    from atlas_camera.comfy import node_registry as reg

    everything = {**reg.NODE_CLASS_MAPPINGS,
                  **reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS,
                  **(getattr(reg, "LEGACY_NODE_CLASS_MAPPINGS", {}) or {}),
                  **(getattr(reg, "IOS_NODE_CLASS_MAPPINGS", {}) or {})}

    # The map is the single source of truth: it must cover exactly the
    # registered nodes — no node without a folder, no folder entry for a node
    # that no longer exists.
    assert set(reg.MENU_CATEGORY) == set(everything), {
        "unmapped": sorted(set(everything) - set(reg.MENU_CATEGORY)),
        "stale": sorted(set(reg.MENU_CATEGORY) - set(everything))}

    # Every class actually carries the folder the map assigns it — the
    # import-time stamp ran and nothing overrode it afterwards.
    wrong = {key: getattr(cls, "CATEGORY", None)
             for key, cls in everything.items()
             if getattr(cls, "CATEGORY", None) != reg.MENU_CATEGORY[key]}
    assert not wrong, f"classes whose CATEGORY != MENU_CATEGORY: {wrong}"

    # Folders are exactly the eleven numbered pipeline stages plus the gated
    # bucket, and each one holds at least one node (no empty menu entries).
    allowed = {
        "Atlas/01 \u00b7 Input & Camera", "Atlas/02 \u00b7 Orient & Scale",
        "Atlas/03 \u00b7 Depth", "Atlas/04 \u00b7 Masks",
        "Atlas/05 \u00b7 Geometry", "Atlas/06 \u00b7 Patch & Repair",
        "Atlas/07 \u00b7 Clean Plate & Inpaint", "Atlas/08 \u00b7 Look & Render",
        "Atlas/09 \u00b7 QA & Gates", "Atlas/10 \u00b7 Export",
        "Atlas/11 \u00b7 Evidence Plate", "Atlas/advanced"}
    used = set(reg.MENU_CATEGORY.values())
    assert used == allowed, {"unexpected": sorted(used - allowed),
                             "empty": sorted(allowed - used)}

    # Gated nodes (experimental, legacy, iOS) all sit in "advanced". Tier gating
    # and menu placement are independent axes, but a not-for-v1 node must never
    # surface among the standard pipeline folders.
    gated = {**reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS,
             **(getattr(reg, "LEGACY_NODE_CLASS_MAPPINGS", {}) or {}),
             **(getattr(reg, "IOS_NODE_CLASS_MAPPINGS", {}) or {})}
    for key in gated:
        assert reg.MENU_CATEGORY[key] == "Atlas/advanced", key


def test_registry_surface_hash_is_stable_and_tracks_the_registered_surface():
    """registry_surface_hash() lets an artifact record WHICH node surface
    produced it (2026-08-08 hygiene pass) — an agent can then ask "was this
    debug report written by the registry I'm talking to?" without git
    archaeology. It must be deterministic across calls and change iff the
    registered keys change."""
    import atlas_camera.comfy.node_registry as reg

    h1 = reg.registry_surface_hash()
    h2 = reg.registry_surface_hash()
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 12, (
        "short stable hex digest — long enough to be unambiguous, short "
        "enough to eyeball in a JSON artifact")

    # Sensitive to the surface: a hypothetical extra key changes the hash.
    import hashlib
    keys = sorted({**reg.NODE_CLASS_MAPPINGS,
                   **reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS,
                   **(getattr(reg, "LEGACY_NODE_CLASS_MAPPINGS", {}) or {}),
                   **(getattr(reg, "IOS_NODE_CLASS_MAPPINGS", {}) or {})})
    expected = hashlib.sha256("\n".join(keys).encode()).hexdigest()[:12]
    assert h1 == expected, (
        "hash must cover the FULL registered surface (standard + gated) in "
        "sorted key order")


def test_debug_report_and_assessment_stamp_the_registry_hash():
    """Text pin: both comfy-layer artifact writers include registry_hash.
    (Runtime tests for these writers need torch; the wiring is one call each,
    pinned here the same way frontend mirrors are pinned.)"""
    base = os.path.join(os.path.dirname(__file__), "..", "atlas_camera", "comfy")
    for name in ("nodes_viewport.py", "nodes_qa.py"):
        src = open(os.path.join(base, name), encoding="utf-8").read()
        assert "registry_surface_hash" in src, f"{name} does not stamp registry_hash"
        assert '"registry_hash"' in src, f"{name} does not emit a registry_hash key"
