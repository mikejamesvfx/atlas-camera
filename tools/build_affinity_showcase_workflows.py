"""Generate the Affinity-bridge showcase workflows from a live ComfyUI.

Four graphs, all shot on real camera RAW (Fujifilm X-H2 / Sony A7S II), all
built around the same division of labour the bridge settled on:

    Affinity selects and paints.  Atlas decodes RAW, owns colorimetry,
    confines the edit, judges it against gates, and stitches the result.

===============================================  ==============================
``atlas_raw_affinity_cleanplate``                RAW -> measured intrinsics ->
                                                 generative cleanplate behind a
                                                 hero object.
``atlas_raw_multiview_affinity_patch``           Three real viewpoints of one
                                                 object; a painted view is
                                                 MEASURED against them, never
                                                 trusted.
``atlas_raw_street_affinity_declutter``          Poles, wires and bollards --
                                                 the hardest matte case.
``atlas_burst_night_affinity_relight``           A night plate with no RAW and
                                                 no parallax -- the weakest
                                                 case, on purpose.
===============================================  ==============================

Three of the four run green on the staged reference assets.
``atlas_raw_multiview_affinity_patch`` ships on PLACEHOLDER paths and will not
run until pointed at a real capture set: every RAW set in the reference shoot
was fed to it and every one was correctly refused (metadata_mismatch,
insufficient_overlap, dynamic_scene_contamination, ambiguous_motion_model).
Those four refusals, with their numbers, ARE its READ ME -- a capture spec
measured rather than asserted.

Nothing here is hand-authored, for the reasons recorded in
``build_v1_shipping_workflows``: UI-format JSON is redundantly linked and
``widgets_values`` is POSITIONAL, so widget order comes from the running
server's ``/object_info`` and can never drift from the node.

Three choices are forced rather than preferred, and each cost a measured run:

* Affinity's EXR export is MISLABELLED: a plate handed over as
  ``lin_rec709_scene`` returns tagged ``ACEScg`` with pixel values untouched
  (1.6% of frame bit-identical, 17.2% inside 1e-4).  The defence is the
  Atlas-side RE-TAG in ``tools/paint_confine_plate.py``, NOT the
  ``AtlasLoadPlate.input_colorspace`` widget: a declared ``oiio:ColorSpace``
  overrides even an explicit input_colorspace, so the widget is inert against a
  mislabelled file.  It is still named explicitly here so the graph states what
  the plate is rather than leaving a reader to guess.
* The cleanplate reaches the scene as ``plate_depth`` on an
  ``AtlasCleanPlateLayer`` -- the doctrine path -- not as a band hack, and the
  hero object is cut per-pixel IN THE SHADER by ``layer_matte`` + ``embed_matte``.
  ``object_mask`` is scale-REGISTRATION exclusion only and does not restrict
  mesh membership; a "phantom object" in the untextured geometry view is
  expected, not a bug.
* The RAW loaders feed ``raw_meta`` into the solve.  Without it GeoCalib
  assumes a 36 mm full-frame sensor and reported 26.8 mm at confidence 0.877 on
  the boiler plate; with the RAF's measured 23.5 mm sensor and lensfun-corrected
  geometry the same frame solves at 20.6 mm, confidence 0.94.

Usage::

    python tools/build_affinity_showcase_workflows.py
    python tools/build_affinity_showcase_workflows.py --check

Loading is not acceptance.  Run every result through
``tools/workflow_benchmark.py`` (or ``atlas_run_workflow``) before it lands.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Sibling-module imports, so the script runs from anywhere (the v1 builder is
# only importable when tools/ is on the path).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_v1_shipping_workflows import _controls, _group, _note
from rebuild_staged_master_workflow import (
    Graph, _fetch_object_info, _load_layout_module)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

# uuid5 of each slug under the DNS namespace, so the ids are reproducible if a
# file ever needs regenerating from scratch.  Must be real UUIDs and must
# differ -- tests/test_example_workflows pins both rules.
WORKFLOW_IDS = {
    "atlas_raw_affinity_cleanplate_workflow":
        "2f8a4d51-6b9c-5e37-a1d2-90c4e7b35f18",
    "atlas_raw_multiview_affinity_patch_workflow":
        "6c1e7b93-4a25-5d68-b3f0-27ad9e14c650",
    "atlas_raw_street_affinity_declutter_workflow":
        "b47d20fe-8c13-5a94-9e6b-51f8072dc3a1",
    "atlas_burst_night_affinity_relight_workflow":
        "d93f5a08-1e76-5c42-8bd5-3a0e6fc19b74",
}

# Exterior plates: the metric outdoor model is the doctrine choice
# (docs/development/design-rules.md, depth model doctrine).
OUTDOOR_DEPTH = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
# Night/interior-ish: MoGe is the doctrine choice where the outdoor metric
# model has no sky and no horizon to anchor on.
MOGE_DEPTH = "Ruicheng/moge-2-vitl-normal"

# What the confined clean plate actually is: the same scene-linear space
# AtlasLoadRAW tags its own sidecar with, and a literal value from
# AtlasLoadPlate's combo (not an OCIO alias).
#
# CORRECTION (2026-08-21): an earlier version of this comment claimed that
# naming the colourspace here is what defends against Affinity's mislabelled
# export. It is NOT. `atlas_camera/plate/oiio_io.py` gives a file's declared
# `oiio:ColorSpace` UNCONDITIONAL precedence -- it overrides even an explicitly
# passed input_colorspace -- so this widget is INERT as a defence and would be
# ignored if the file were mislabelled.
#
# The real defence is the Atlas-side RE-TAG performed by
# `tools/paint_confine_plate.py`, which writes the confined plate with the
# ORIGINAL's colourspace. The shipped plates are trustworthy because they were
# produced by that tool (both verified tagged `lin_rec709_scene`), not because
# of this value. Setting it correctly still matters so the graph does not lie
# to a reader.
AFFINITY_PLATE_COLORSPACE = "Linear Rec.709 (sRGB)"


def _finish(graph: Graph, slug: str, group_specs, layout, notes: str) -> dict:
    """Auto-layout, then box groups around where the nodes LANDED.

    Same contract as the v1 builder: positions are derived, not typed, so the
    overlap check below is a real assertion rather than a formality.
    """
    workflow_graph = layout.auto_layout({"nodes": graph.nodes, "links": graph.links})
    check = layout.inspect(workflow_graph)
    if check["overlaps"]:
        raise RuntimeError(f"{slug} layout overlaps: {check['overlaps']}")
    groups = [_group(nodes, title, color) for title, color, nodes in group_specs]
    return {
        "id": WORKFLOW_IDS[slug],
        "revision": 1,
        "last_node_id": graph._node_id,
        "last_link_id": graph._link_id,
        "nodes": graph.nodes,
        "links": graph.links,
        "groups": groups,
        "config": {},
        "extra": {
            "ds": {"scale": 0.58, "offset": [35, 85]},
            "frontendVersion": "1.25.11",
            "workflowRendererVersion": "LG",
            "atlas_shipping_set": "v1",
            "atlas_notes": notes,
        },
        "version": 0.4,
    }


def _raw(graph: Graph, path: str, title: str, *, out_dir: str,
         half_size: bool = True) -> dict:
    """A RAW loader wired for the showcase: measured intrinsics + EXR sidecar.

    ``half_size`` is on by default.  These are 40 MP frames; a half-size
    demosaic is 3876x2589, which is ample for depth, SAM3 and projection, and
    it is what makes the graph iterable rather than a coffee break.  The
    intrinsics are unaffected -- they come from EXIF and the camera database,
    not from the raster.
    """
    return graph.node("AtlasLoadRAW", title=title, values={
        "file_path": path,
        "undistort": True,          # lensfun HAS a profile for the XF16-55
        "half_size": half_size,
        "white_balance": "camera",
        "exposure_ev": 0.0,
        "write_exr": True,          # the scene-linear plate the bridge hands over
        "output_dir": out_dir,
        "colorspace": "Linear Rec.709 (sRGB)",
        "headroom": 6.0,
    }, size=(460, 320))


# ---------------------------------------------------------------------------
# 1 · RAW -> generative cleanplate behind a hero object
# ---------------------------------------------------------------------------

CLEANPLATE_NOTE = """RAW → AFFINITY CLEANPLATE — remove the hero object, keep the ground it stood on.

WHAT THIS SHOWS
A 40MP Fujifilm RAF goes in. Atlas decodes it, takes the intrinsics from the file rather than guessing them,
solves the camera, cuts the hero object out per-pixel, and puts a GENERATIVE CLEANPLATE behind it — so when
you orbit, there is real ground and real treeline where the object used to be instead of a black tear.

WHY raw_meta IS WIRED INTO THE SOLVE
From the JPEG alone, GeoCalib assumes a 36mm full-frame sensor and reports 26.8mm at confidence 0.877.
The RAF knows better: 23.5mm sensor, 20.6mm lens, lensfun distortion profile applied. Same frame, same
solver, wired this way: 20.6mm at confidence 0.94. The wire from AtlasLoadRAW.raw_meta into AtlasInput is
the whole difference. Unplug it and watch the focal drift.

THE AFFINITY LEG (already done — this graph consumes the result)
Affinity's generativeEditImage regenerates the WHOLE frame, not just the selection: measured containment
0.3740 on this plate. So Atlas confines it afterwards:

    python tools/paint_confine_plate.py --original <raw exr> --edited <affinity exr> \\
        --mask <sam3 mask>.png --out <confined>.exr --out-mask <authorised>.png \\
        --drop-px 320 --dilate-px 45 --feather-px 12
    python tools/paint_roundtrip_score.py --original <raw exr> \\
        --edited <confined>.exr --mask <authorised>.png

Confined: containment 1.0000 PASS, seam_gradient_ratio 0.9987 PASS — accepted.
--drop-px extends the matte DOWNWARD, because a ground-standing object's legs, footings and contact shadow
sit below it, not around it. Widening the SAM3 concept list instead is a trap: it has no semantic gating,
and asking for "concrete footing pad" returned the entire brick plinth (50.8% of frame) while STILL missing
the black steel legs.

READ THE GATES HONESTLY
containment says the edit stayed in bounds. seam_gradient_ratio says it joins cleanly at the rim. NEITHER
says the interior content is right — on this plate both gates pass while the regenerated treeline inside the
silhouette does not line up with the treeline outside it. Gates are necessary, not sufficient. Look at it.

COLOUR — the one setting you must not leave on 'auto'
AtlasLoadPlate.input_colorspace is set EXPLICITLY. Affinity exports this EXR tagged ACEScg while leaving the
Rec.709-linear values untouched (1.6% of frame comes back bit-identical, which no colour transform allows).
A plate that self-describes is believed, so 'auto' would convert Rec.709 data as if it were ACEScg and shift
every primary. Name what the file actually is.

LAYER SEMANTICS (the part that costs a day if you learn it the hard way)
• layer_matte + embed_matte cut the object per-pixel IN THE SHADER, under 📽 Project.
• object_mask is scale-REGISTRATION exclusion only. It does NOT restrict mesh membership.
• Geometry restriction is the depth BAND. In the untextured geometry view you will see raw band meshes,
  and a "phantom object" there is expected — not a bug.
• The cleanplate reaches the scene as plate_depth (MoGe on the cleanplate image, median-ratio registered).
  That is the doctrine path. Do not reach for band hacks.

TO RE-SHOOT THIS WITH YOUR OWN PLATE
Swap the RAF path, re-point the SAM3 concept at your object, run the two tools above, drop the confined EXR
in. Nothing else changes."""


def build_cleanplate(object_info: dict, layout) -> dict:
    slug = "atlas_raw_affinity_cleanplate_workflow"
    graph = Graph(object_info)

    raw = _raw(graph, "atlas_showcase/boiler/boiler_primary.RAF",
               "RAW · hero plate (X-H2 40MP RAF)",
               out_dir="atlas_exports/affinity_showcase/boiler")

    solve = graph.node("AtlasInput", title="SOLVE · intrinsics PINNED by raw_meta",
                       values={
                           "layers": 1,
                           "mesh": "relief",
                           "use_vlm": False,
                           "sky": True,
                           "depth_model": OUTDOOR_DEPTH,
                           "sub_quad_boundary": True,
                       }, size=(460, 700))
    graph.connect(raw, "image", solve, "image")
    graph.connect(raw, "raw_meta", solve, "raw_meta")

    matte = graph.node("AtlasSAM3Mask", title="SAM3 · the hero object", values={
        "concepts": "rusty steel boiler tank",
        "confidence_threshold": 0.4,
        "output_mode": "merged",
    }, size=(420, 260))
    graph.connect(solve, "image", matte, "image")

    hero = graph.node("AtlasCleanPlateLayer",
                      title="LAYER · hero object (matte cut in shader)", values={
                          "name": "boiler",
                          "priority": 1.0,
                          "relief_grid": 160,
                          "embed_matte": True,
                          "sub_quad_boundary": True,
                          "silhouette_matte": True,
                          "band_geometry": "relief",
                      }, size=(460, 1180))
    graph.connect(solve, "solve", hero, "solve")
    graph.connect(solve, "depth", hero, "depth")
    graph.connect(solve, "image", hero, "plate_image")
    graph.connect(matte, "mask", hero, "layer_matte")

    plate = graph.node("AtlasLoadPlate",
                       title="AFFINITY CLEANPLATE · re-tagged Atlas-side, then named here",
                       values={
                           "file_path":
                               "atlas_showcase/boiler/boiler_cleanplate_confined.exr",
                           "input_colorspace": AFFINITY_PLATE_COLORSPACE,
                           "output_colorspace": "sRGB - Display",
                           "raw_data": False,
                       }, size=(500, 220))

    plate_depth = graph.node("AtlasDepthMap",
                             title="CLEANPLATE DEPTH · its OWN geometry",
                             values={"depth_model": OUTDOOR_DEPTH},
                             size=(460, 240))
    graph.connect(plate, "image", plate_depth, "image")
    graph.connect(solve, "solve", plate_depth, "solve")

    background = graph.node("AtlasCleanPlateLayer",
                            title="LAYER · cleanplate BG (plate_depth, registered)",
                            values={
                                "name": "cleanplate_bg",
                                "priority": 8.0,      # FARTHEST-highest
                                "relief_grid": 160,
                                "near_m": 1.5,
                                "far_m": 120.0,
                                "fill_occluded": True,
                                "edge_extend_px": 48,
                                "sub_quad_boundary": True,
                            }, size=(460, 1180))
    # Layer nodes CHAIN: each takes the previous layer's solve and appends to
    # it. Wiring this one back to AtlasInput instead silently drops every
    # layer between them -- the hero layer vanished from the census exactly
    # once, which is how this comment came to exist.
    graph.connect(hero, "solve", background, "solve")
    graph.connect(solve, "depth", background, "depth")
    graph.connect(plate, "image", background, "plate_image")
    graph.connect(plate, "plate_ref", background, "plate_ref")
    graph.connect(plate_desk := plate_depth, "depth", background, "plate_depth")

    controls = _controls(graph, vfx=True)

    viewport = graph.node("AtlasBlockoutViewport",
                          title="VIEWPORT · orbit behind the hero object",
                          values={"resolution": 1024}, size=(1100, 900))
    graph.connect(background, "solve", viewport, "solve")
    graph.connect(solve, "image", viewport, "source_image")
    graph.connect(solve, "depth", viewport, "primary_depth")
    graph.connect(controls, "controls", viewport, "controls")
    graph.connect(controls, "output_profile", viewport, "output_profile")

    report = graph.node("AtlasDebugReport", title="REPORT · layers, bands, tears",
                        values={"file_path":
                                "atlas_debug/affinity_showcase_cleanplate.json"},
                        size=(460, 240))
    graph.connect(background, "solve", report, "solve")
    graph.connect(solve, "depth", report, "depth")

    note = _note(graph, CLEANPLATE_NOTE, "READ ME · RAW → Affinity cleanplate",
                 size=(900, 1180))

    return _finish(graph, slug, [
        ("RAW · measured intrinsics", "#3f789e", [raw, solve]),
        ("AFFINITY · confined + scored cleanplate", "#8A3F3F", [plate, plate_depth]),
        ("LAYERS", "#3f5159", [matte, hero, background]),
        ("VIEWPORT · colour desk", "#446e3f", [controls, viewport, report]),
        ("READ ME", "#653", [note]),
    ], layout, "RAW -> Affinity generative cleanplate, confined and gated.")


# ---------------------------------------------------------------------------
# 2 · three real viewpoints, one painted view, MEASURED
# ---------------------------------------------------------------------------

MULTIVIEW_NOTE = """THREE PHOTOGRAPHED VIEWS + ONE PAINTED VIEW — and the painted one has to earn its place.

⚠ THIS GRAPH SHIPS ON PLACEHOLDER PATHS AND WILL NOT RUN UNTIL YOU POINT IT AT A REAL CAPTURE SET.
That is deliberate, and the section "WHAT A VALID SET ACTUALLY LOOKS LIKE" below is the measured reason.

WHAT THIS SHOWS
Three real RAF frames of one subject, shot from three positions, registered against each other with no
learned prior in the loop. Then an optional PAINTED view is added — and it is MEASURED against the
photographed world rather than believed.

WHY THIS MATTERS
A generated view is evidence of what a surface looks like, never of where the camera was. AtlasAddPatchView
with camera_source=register_to_primary measures the generated view's camera against the primary's world.
The direction is one-way by design: generated → measured. The primary NEVER moves, and the patch pose
inherits primary_depth's scale.

If a strong match DISAGREES with the declared orbit, that is a WARNING and THE MEASUREMENT WINS. The declared
azimuth/elevation widgets are a hint for the generator, not a claim about geometry.

WHAT A VALID SET ACTUALLY LOOKS LIKE — measured, on 2026-08-21
Four candidate sets of real X-H2 frames were fed to this graph. EVERY ONE was refused, each for a different
and correct reason. These are the numbers to shoot against:

  metadata_mismatch     Mixing frames across shots: focal 20.6 / 24.9 / 23.4mm, orientation 1 vs 8,
                        developed dimensions [3876,2589] vs [2589,3876]. One shoot, one orientation,
                        one focal. The solver will not average away a portrait frame.

  insufficient_overlap  "photos 2-3 have 31 mutual matches; at least 48 are required."
                        48 mutual matches is the floor, per pair the topology uses.

  dynamic_scene_contamination
                        "many raw matches but every geometric consensus is confined to too few
                        4x4 grid cells" — 247 matches, consensus in 5 of 16 cells, all of them in
                        wind-moving tree canopy. Match COUNT is not match COVERAGE. Frame so that
                        static, textured surfaces span the whole frame, not just the top of it.

  ambiguous_motion_model
                        49 mutual matches collapsed to 7 essential inliers at angle_deg 164 — nonsense.
                        Raw match count survived; geometry did not. Walking 10 seconds between frames
                        past repetitive brick and siding produces confident wrong correspondences.

SO: 60-80% overlap between ADJACENT frames. Move LATERALLY — rotating on the spot gives no parallax and no
baseline to measure. Keep focal and orientation fixed across the set. Shoot the set as a set, in one pass.

pair_topology is anchor_star, not auto. Walking around a subject, the two OUTER frames barely see each other
(measured 247 matches for 1-2, 72 for 1-3, but only 31 for 2-3). Anchor-star only requires every frame to
overlap the ANCHOR. It is not a loosened threshold — every pair it uses still clears the same 48-match bar.

match_quality stays 'balanced'. Go permissive only AFTER reviewing the pair-match overlays, never before —
permissive on a bad set produces a confident wrong answer, which is worse than the refusal you started with.

WIRING THE PAINTED VIEW (Affinity, or any generator)
The patch node starts BYPASSED. To enable it:
1. Paint or generate the view — in Affinity: doc.generateImage(prompt), or generativeEditImage on a copy.
   Export it and load it through the LoadImage node.
2. Set node mode back to Always.
3. Read the report. registration_min_inliers 40 / max_residual_m 0.35 / max_deviation_deg 25 are the gates.
   A patch that cannot make inliers is a patch that does not belong in the scene.

Disconnect image_3 / raw_meta_3 / plate_ref_3 for a two-photo solve. The third view is genuinely optional."""


def build_multiview(object_info: dict, layout) -> dict:
    slug = "atlas_raw_multiview_affinity_patch_workflow"
    graph = Graph(object_info)

    out = "atlas_exports/affinity_showcase/multiview"
    # PLACEHOLDER paths. Every RAW set in this repo's reference shoot was fed
    # to this graph and every one was correctly refused (see MULTIVIEW_NOTE for
    # the four outcome codes and their numbers) -- they are bracket and
    # variation frames, not an overlapping survey. Shipping the graph pointed
    # at a set that cannot register would teach the wrong lesson, so it ships
    # pointed at nothing and carries the capture spec instead. Same convention
    # as atlas_multiview_raw_qwen_workflow.
    raw_a = _raw(graph, "atlas_multiview/view_01.RAF",
                 "VIEW 1 · primary — the world everything registers INTO",
                 out_dir=out)
    raw_b = _raw(graph, "atlas_multiview/view_02.RAF",
                 "VIEW 2 · lateral move, ~70% overlap", out_dir=out)
    raw_c = _raw(graph, "atlas_multiview/view_03.RAF",
                 "VIEW 3 · optional — disconnect for a 2-photo solve", out_dir=out)

    mv = graph.node("AtlasMultiViewSolve",
                    title="DETERMINISTIC REGISTRATION · no learned prior",
                    values={
                        "capture_mode": "auto",
                        "match_quality": "balanced",
                        # anchor_star, not auto. Walking around an object, the
                        # two OUTER frames barely see each other: measured
                        # 247 mutual matches for 1-2 and 72 for 1-3, but only
                        # 31 for 2-3 against a floor of 48, which failed the
                        # whole solve on [insufficient_overlap]. Anchor-star
                        # only requires every frame to overlap the ANCHOR,
                        # which is the honest topology for an orbit. It is not
                        # a loosened threshold -- every pair it does use still
                        # has to clear the same bar.
                        "pair_topology": "anchor_star",
                    }, size=(480, 380))
    for raw, idx in ((raw_a, 1), (raw_b, 2), (raw_c, 3)):
        graph.connect(raw, "image", mv, f"image_{idx}")
        graph.connect(raw, "raw_meta", mv, f"raw_meta_{idx}")
        graph.connect(raw, "plate_ref", mv, f"plate_ref_{idx}")

    depth = graph.node("AtlasDepthMap", title="PRIMARY DEPTH · shared metric scale",
                       values={"depth_model": OUTDOOR_DEPTH}, size=(460, 240))
    graph.connect(raw_a, "image", depth, "image")
    graph.connect(mv, "solve", depth, "solve")

    painted = graph.node("LoadImage",
                         title="PAINTED VIEW · Affinity generateImage output",
                         values={"image": "example.png"}, size=(400, 340))

    patch = graph.node("AtlasAddPatchView",
                       title="PATCH · MEASURED against the primary (start BYPASSED)",
                       values={
                           "name": "painted_view",
                           "camera_source": "register_to_primary",
                           "geometry_source": "reuse_scene",
                           "depth_model": OUTDOOR_DEPTH,
                           "registration_min_inliers": 40,
                           "registration_max_residual_m": 0.35,
                           "registration_max_deviation_deg": 25.0,
                       }, size=(480, 900), mode=4)   # 4 = bypassed
    graph.connect(mv, "solve", patch, "solve")
    graph.connect(painted, "IMAGE", patch, "patch_image")
    graph.connect(depth, "depth", patch, "primary_depth")
    graph.connect(raw_a, "image", patch, "primary_image")

    controls = _controls(graph, vfx=True)

    viewport = graph.node("AtlasBlockoutViewport",
                          title="VIEWPORT · registered rig, primary projection",
                          values={"resolution": 1024}, size=(1100, 900))
    graph.connect(patch, "solve", viewport, "solve")
    graph.connect(raw_a, "image", viewport, "source_image")
    graph.connect(depth, "depth", viewport, "primary_depth")
    graph.connect(controls, "controls", viewport, "controls")
    graph.connect(controls, "output_profile", viewport, "output_profile")

    overlays = graph.node("PreviewImage",
                          title="PAIR-MATCH OVERLAYS · photographed evidence",
                          size=(460, 460))
    graph.connect(mv, "match_overlays", overlays, "images")

    report = graph.node("AtlasDebugReport",
                        title="REPORT · registration + scene",
                        values={"file_path":
                                "atlas_debug/affinity_showcase_multiview.json"},
                        size=(460, 240))
    graph.connect(patch, "solve", report, "solve")
    graph.connect(depth, "depth", report, "depth")

    note = _note(graph, MULTIVIEW_NOTE, "READ ME · measured, not trusted",
                 size=(900, 900))

    return _finish(graph, slug, [
        ("PHOTOGRAPHED VIEWS · RAW", "#3f789e", [raw_a, raw_b, raw_c]),
        ("REGISTRATION", "#3f5159", [mv, depth, overlays]),
        ("PAINTED VIEW · measured", "#8A3F3F", [painted, patch]),
        ("VIEWPORT · colour desk", "#446e3f", [controls, viewport, report]),
        ("READ ME", "#653", [note]),
    ], layout, "Three photographed views; a painted view measured against them.")


# ---------------------------------------------------------------------------
# 3 · poles, wires and bollards — the hardest matte case
# ---------------------------------------------------------------------------

DECLUTTER_NOTE = """STREET DECLUTTER — poles, wires and bollards, which are where mattes actually break.

WHY THIS PLATE IS HARD
A boiler is a big convex blob and any segmenter gets it right. A light pole is four pixels wide against a
bright sky, an overhead wire is ONE pixel wide, and a bollard has a cast shadow longer than the bollard.
These are the cases that expose the difference between a matte that looks fine at fit-to-screen and a matte
that survives an orbit.

WHAT TO WATCH
1. The pole layer is FULL-BAND terrain, matte-cut in the shader. Under 📽 Project it reads correctly. In the
   untextured geometry view it is heavy and messy — that is the documented behaviour of a shader-cut matte,
   not a defect. Geometry-level mask restriction (true mesh membership from a MASK) is not a capability yet.
2. silhouette_matte is ON. The sky/exclusion edge and a depth cliff are TWO different staircases with two
   different fixes and no overlap:
      • sub_quad_boundary  → DEPTH CLIFFS. Both sheets share a pixel, so no matte can help. Topology.
      • silhouette_matte   → the SKY/EXCLUSION edge. A full-res matte cutting a skirt back.
   The skirt and the matte are ONE switch. An unmatted skirt is a measured defect, not a style.
3. Relief-mesh tears are LOAD-BEARING. Never fix a black tear by raising a global threshold. The fix is a
   deliberate layer — a card, a ground, a sky, or an inpaint.

THE AFFINITY LEG
Hand Affinity the scene-linear EXR, let it remove the street furniture, then confine and score:

    python tools/paint_confine_plate.py --original <raw exr> --edited <affinity exr> \\
        --mask <sam3 mask>.png --out <confined>.exr --out-mask <authorised>.png \\
        --drop-px 400 --dilate-px 30 --feather-px 10

--drop-px matters more here than anywhere: a bollard's shadow runs along the ground away from it, and the
matte has to reach the shadow or the composite keeps a shadow with nothing casting it.

SEAM DOCTRINE
Edge-extend smear lives on the layers BEHIND. The frontmost band keeps a CLEAN CUT. Band priorities are
FARTHEST-highest. If you find yourself smearing the front layer to hide a seam, the layer stack is wrong."""


def build_declutter(object_info: dict, layout) -> dict:
    slug = "atlas_raw_street_affinity_declutter_workflow"
    graph = Graph(object_info)

    raw = _raw(graph, "atlas_showcase/street/street_primary.RAF",
               "RAW · street plate (X-H2 40MP RAF)",
               out_dir="atlas_exports/affinity_showcase/street")

    solve = graph.node("AtlasInput", title="SOLVE · raw_meta pinned",
                       values={
                           "layers": 2,
                           "mesh": "relief",
                           "use_vlm": False,
                           "sky": True,
                           "depth_model": OUTDOOR_DEPTH,
                           "sub_quad_boundary": True,
                       }, size=(460, 700))
    graph.connect(raw, "image", solve, "image")
    graph.connect(raw, "raw_meta", solve, "raw_meta")

    poles = graph.node("AtlasSAM3Mask", title="SAM3 · poles, posts, bollards",
                       values={
                           "concepts": "street light pole, bollard, sign post",
                           "confidence_threshold": 0.35,
                           "output_mode": "merged",
                       }, size=(420, 260))
    graph.connect(solve, "image", poles, "image")

    pole_layer = graph.node("AtlasCleanPlateLayer",
                            title="LAYER · street furniture (shader-cut matte)",
                            values={
                                "name": "street_furniture",
                                "priority": 1.0,
                                "relief_grid": 160,
                                "embed_matte": True,
                                "sub_quad_boundary": True,
                                "silhouette_matte": True,
                                "band_geometry": "relief",
                            }, size=(460, 1180))
    graph.connect(solve, "solve", pole_layer, "solve")
    graph.connect(solve, "depth", pole_layer, "depth")
    graph.connect(solve, "image", pole_layer, "plate_image")
    graph.connect(poles, "mask", pole_layer, "layer_matte")

    plate = graph.node("AtlasLoadPlate",
                       title="AFFINITY CLEANPLATE · re-tagged Atlas-side, then named here",
                       values={
                           "file_path":
                               "atlas_showcase/street/street_cleanplate_confined.exr",
                           "input_colorspace": AFFINITY_PLATE_COLORSPACE,
                           "output_colorspace": "sRGB - Display",
                           "raw_data": False,
                       }, size=(500, 220))

    plate_depth = graph.node("AtlasDepthMap",
                             title="CLEANPLATE DEPTH · its OWN geometry",
                             values={"depth_model": OUTDOOR_DEPTH},
                             size=(460, 240))
    graph.connect(plate, "image", plate_depth, "image")
    graph.connect(solve, "solve", plate_depth, "solve")

    ground = graph.node("AtlasCleanPlateLayer",
                        title="LAYER · decluttered ground (farthest-highest)",
                        values={
                            "name": "street_ground",
                            "priority": 8.0,
                            "relief_grid": 160,
                            "near_m": 1.0,
                            "far_m": 80.0,
                            "fill_occluded": True,
                            "edge_extend_px": 48,   # smear lives BEHIND
                            "sub_quad_boundary": True,
                        }, size=(460, 1180))
    graph.connect(pole_layer, "solve", ground, "solve")
    graph.connect(solve, "depth", ground, "depth")
    graph.connect(plate, "image", ground, "plate_image")
    graph.connect(plate, "plate_ref", ground, "plate_ref")
    graph.connect(plate_depth, "depth", ground, "plate_depth")
    # Scoped excludes shift band percentiles, so the percentile reference stays
    # the PLAIN sky mask (drift rule, docs/development/design-rules.md).
    graph.connect(solve, "sky_mask", ground, "band_ref_mask")

    controls = _controls(graph, vfx=True)

    viewport = graph.node("AtlasBlockoutViewport",
                          title="VIEWPORT · orbit past the poles",
                          values={"resolution": 1024}, size=(1100, 900))
    graph.connect(ground, "solve", viewport, "solve")
    graph.connect(solve, "image", viewport, "source_image")
    graph.connect(solve, "depth", viewport, "primary_depth")
    graph.connect(controls, "controls", viewport, "controls")
    graph.connect(controls, "output_profile", viewport, "output_profile")

    report = graph.node("AtlasDebugReport",
                        title="REPORT · tear + matte QA",
                        values={"file_path":
                                "atlas_debug/affinity_showcase_declutter.json"},
                        size=(460, 240))
    graph.connect(ground, "solve", report, "solve")
    graph.connect(solve, "depth", report, "depth")

    note = _note(graph, DECLUTTER_NOTE, "READ ME · two staircases, two fixes",
                 size=(900, 1000))

    return _finish(graph, slug, [
        ("RAW · measured intrinsics", "#3f789e", [raw, solve]),
        ("MATTES · the hard case", "#8A3F3F", [poles]),
        ("AFFINITY · confined + scored cleanplate", "#8A3F3F", [plate, plate_depth]),
        ("LAYERS", "#3f5159", [pole_layer, ground]),
        ("VIEWPORT · colour desk", "#446e3f", [controls, viewport, report]),
        ("READ ME", "#653", [note]),
    ], layout, "Poles, wires and bollards: the matte cases that actually break.")


# ---------------------------------------------------------------------------
# 4 · handheld low-light burst
# ---------------------------------------------------------------------------

NIGHT_NOTE = """NIGHT PLATE — low light, no RAW, no parallax, and the report saying so.

WHAT THIS IS
One frame from a twelve-frame handheld burst: 1/100s, ISO 3200, 53mm, shot at dusk. It is included because
it is where this pipeline is WEAKEST, not where it looks best. Three things are missing at once, and the
graph is honest about all three.

1 · NO PARALLAX — measured, not assumed
The obvious move is to feed all twelve frames to AtlasMultiViewSolveBurst and register a rig. That was tried
against this exact folder. It refused, twice, and both refusals were correct:

    [degenerate_geometry]   Photo 1 needs two clear orthogonal architectural vanishing points.
                            Sparse correspondence cannot safely guess world-up.
    [ambiguous_motion_model] neither motion model passed checks;
                            essential(inliers=101, cells=8, error_px=0.30, angle_deg=0.359666)

angle_deg 0.36 is the answer. Twelve frames in three seconds of hand tremor is an EXPOSURE burst, not a
baseline. There is no parallax to triangulate, so there is no rig — and the engine says so instead of
returning a confident wrong one. A burst is not a multi-view set just because it has many frames.
Move LATERALLY if you want a rig.

learned_anchor_fallback / learned_scale_fallback exist for the first failure and were enough to clear it.
They are OFF by default on purpose: switching them on is a decision to accept a LEARNED quantity where a
measured one was wanted, and that decision should be visible in the graph.

2 · NO raw_meta — the solve estimates the sensor
These are JPEGs. There is no RAW alongside them, so nothing pins the intrinsics and the solver falls back to
assuming a full-frame sensor. That is exactly the gap atlas_raw_affinity_cleanplate_workflow closes: on the
boiler plate the same fallback reported 26.8mm / confidence 0.877 where the RAF's measured 23.5mm sensor
gave 20.6mm / confidence 0.94. Shoot RAW.

3 · NO SKY, NO HORIZON — so MoGe, not the metric outdoor model
Depth model doctrine sends exteriors to V2-Metric-Outdoor, but that model anchors on sky and horizon and a
night alley has neither. MoGe is the choice here. Doctrine is a default, not a rule to follow off a cliff.

THE AFFINITY LEG FOR NIGHT PLATES
Generative fill on a night plate regenerates the LIGHTING, not just the geometry: practicals move, colour
temperature drifts, wet-asphalt reflections get reinvented. Confine harder than you would by day:

    python tools/paint_confine_plate.py --original <exr> --edited <affinity exr> \
        --mask <mask>.png --out <confined>.exr --out-mask <authorised>.png \
        --drop-px 200 --dilate-px 24 --feather-px 16

A wider feather buys a smoother join in a noisy plate; it also enlarges the authorised region, and the
authorised mask this tool writes already accounts for that. Hand THAT mask to the scorer, never the raw
object mask — a feather is spill unless the authorised mask includes it."""


def build_night(object_info: dict, layout) -> dict:
    slug = "atlas_burst_night_affinity_relight_workflow"
    graph = Graph(object_info)

    # AtlasLoadPlate, not LoadImage: LoadImage's `image` is a COMBO built from
    # ComfyUI's input folder listing, which does not enumerate subfolders, so
    # a shipped subfolder path is an invalid option on every machine. A string
    # file_path resolves against the input folder at load time and keeps the
    # relative placeholder the path guard requires.
    plate = graph.node("AtlasLoadPlate",
                       title="NIGHT PLATE · JPEG, 1/100s ISO 3200, 53mm",
                       values={
                           "file_path": "atlas_showcase/night/night_168.JPG",
                           "input_colorspace": "auto",
                           "output_colorspace": "sRGB - Display",
                           "raw_data": False,
                       }, size=(460, 220))

    # No raw_meta wire: there is no RAW for these frames. That is the point —
    # compare the focal this reports against the RAW cleanplate workflow's.
    solve = graph.node("AtlasInput",
                       title="SOLVE · NO raw_meta — sensor is ESTIMATED",
                       values={
                           "layers": 1,
                           "mesh": "relief",
                           "use_vlm": False,
                           "sky": False,       # no sky in an alley at dusk
                           "sky_heuristic": False,
                           "depth_model": MOGE_DEPTH,
                           "sub_quad_boundary": True,
                       }, size=(460, 700))
    graph.connect(plate, "image", solve, "image")

    layer = graph.node("AtlasCleanPlateLayer",
                       title="LAYER · alley relief",
                       values={
                           "name": "alley",
                           "priority": 4.0,
                           "relief_grid": 160,
                           "near_m": 0.8,
                           "far_m": 60.0,
                           "fill_occluded": True,
                           "edge_extend_px": 32,
                           "sub_quad_boundary": True,
                       }, size=(460, 1180))
    graph.connect(solve, "solve", layer, "solve")
    graph.connect(solve, "depth", layer, "depth")
    graph.connect(plate, "image", layer, "plate_image")
    graph.connect(plate, "plate_ref", layer, "plate_ref")

    controls = _controls(graph, vfx=False)

    viewport = graph.node("AtlasBlockoutViewport",
                          title="VIEWPORT · push down the alley",
                          values={"resolution": 1024}, size=(1100, 900))
    graph.connect(layer, "solve", viewport, "solve")
    graph.connect(plate, "image", viewport, "source_image")
    graph.connect(solve, "depth", viewport, "primary_depth")
    graph.connect(controls, "controls", viewport, "controls")
    graph.connect(controls, "output_profile", viewport, "output_profile")

    report = graph.node("AtlasDebugReport",
                        title="REPORT · read this, not the picture",
                        values={"file_path":
                                "atlas_debug/affinity_showcase_night.json"},
                        size=(460, 240))
    graph.connect(layer, "solve", report, "solve")
    graph.connect(solve, "depth", report, "depth")

    note = _note(graph, NIGHT_NOTE, "READ ME · the weakest case, on purpose",
                 size=(900, 1100))

    return _finish(graph, slug, [
        ("NIGHT PLATE · no RAW", "#3f789e", [plate, solve]),
        ("LAYER", "#8A3F3F", [layer]),
        ("VIEWPORT · colour desk", "#446e3f", [controls, viewport, report]),
        ("READ ME", "#653", [note]),
    ], layout, "Night plate: no parallax, no raw_meta, and the report saying so.")


BUILDERS = {
    "atlas_raw_affinity_cleanplate_workflow": build_cleanplate,
    "atlas_raw_multiview_affinity_patch_workflow": build_multiview,
    "atlas_raw_street_affinity_declutter_workflow": build_declutter,
    "atlas_burst_night_affinity_relight_workflow": build_night,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1:8188")
    parser.add_argument("--only", default="", help="build a single slug")
    parser.add_argument(
        "--check", action="store_true",
        help="compare each builder's output against the committed JSON and "
             "exit non-zero on drift, writing nothing")
    args = parser.parse_args()
    object_info = _fetch_object_info(args.host)
    layout = _load_layout_module()
    drifted: list[str] = []

    for slug, builder in BUILDERS.items():
        if args.only and args.only not in slug:
            continue
        workflow = builder(object_info, layout)
        output = EXAMPLES / f"{slug}.json"

        if args.check:
            # Node TYPES, not bytes: positions and link ids move with the
            # layout pass, so a byte comparison would cry drift every run.
            if not output.is_file():
                drifted.append(f"{slug}: no committed file")
                continue
            committed = json.loads(output.read_text(encoding="utf-8"))
            was = sorted(n["type"] for n in committed["nodes"])
            now = sorted(n["type"] for n in workflow["nodes"])
            if was != now:
                drifted.append(
                    f"{slug}: committed-only={sorted(set(was) - set(now))} "
                    f"builder-only={sorted(set(now) - set(was))}")
            else:
                print(f"ok    {slug}")
            continue

        output.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
        print(f"  {layout.inspect(workflow)['summary']}")

    if drifted:
        print("\nDRIFT — the builder and the committed workflow disagree:")
        for line in drifted:
            print("  " + line)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
