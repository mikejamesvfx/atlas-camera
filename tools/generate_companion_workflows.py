"""Generate the two companion marketing workflows:
  2) atlas_jungle_xray_cameramove.json  — 🩻 X-ray hidden geometry + camera move
  3) atlas_castle_dcc_handoff.json      — 🎞 Nuke + Maya + USD + OBJ handoff

Shares the castle generator's helpers: every widget list is derived from the
LIVE /object_info so positional widgets_values cannot drift.

Human-readable guide (purpose, usage, architecture, gotchas):
docs/dev/generate_companion_workflows.md
"""
import importlib.util, json, sys, pathlib

GEN = pathlib.Path(sys.argv[3])          # path to generate_castle_dmp_workflow.py
OI_PATH, OUTDIR = sys.argv[1], pathlib.Path(sys.argv[2])

# Reuse the proven builder by importing it with argv spoofed (it reads argv at
# import time and writes on import; point it at a throwaway file).
_tmp = OUTDIR / "_scratch_ignore.json"
sys.argv = ["gen", OI_PATH, str(_tmp)]
spec = importlib.util.spec_from_file_location("castlegen", GEN)
cg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cg)
_tmp.unlink(missing_ok=True)

WF, RAIL = cg.WF, ("#2a363b", "#3f5159")


class Builder:
    def __init__(self):
        self.w = WF()
        self.sets = {}

    def rset(self, name, src, sslot, pos):
        t = src["outputs"][sslot]["type"]
        n = self.w.raw("SetNode", pos, [210, 60], f"Set_{name}", [name],
                       [{"name": t, "type": t, "link": None}],
                       [{"name": t, "type": t, "links": None}], color=RAIL,
                       props={"aux_id": "kijai/ComfyUI-KJNodes", "previousName": name})
        self.w.link(src, sslot, n, t)
        self.sets[name] = t
        return n

    def rget(self, name, pos):
        t = self.sets[name]
        return self.w.raw("GetNode", pos, [210, 60], f"Get_{name}", [name], [],
                          [{"name": t, "type": t, "links": [], "slot_index": 0}],
                          color=RAIL, props={"aux_id": "kijai/ComfyUI-KJNodes"})


# ══════════════════════════════════════════════════════════════════════════
# 2 · X-RAY HIDDEN GEOMETRY + CAMERA MOVE  (jungle temple)
# ══════════════════════════════════════════════════════════════════════════
def build_xray():
    b = Builder(); w = b.w
    PLATE = "atlas_jungle_temple.png"

    w.group("0 · PLATE + SOLVE + ✅ GATE — cheap solve first; approve before the X-ray model runs", [-40, -40, 1500, 860], "#355")
    load = w.raw("LoadImage", [0, 40], [320, 320], "Jungle temple plate", [PLATE, "image"], [],
                 [{"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0},
                  {"name": "MASK", "type": "MASK", "links": [], "slot_index": 1}])
    b.rset("plate", load, 0, [0, 400])
    g1 = b.rget("plate", [360, 40])
    solve = w.node("AtlasLearnedSolveFromImage", [360, 140], [400, 240], "Learned solve",
                   {"height_mode": "measure_from_depth", "camera_height_m": 1.6,
                    "depth_model": "Ruicheng/moge-2-vitl-normal"})
    w.link(g1, 0, solve, "image")
    g2 = b.rget("plate", [360, 420])
    gate = w.node("AtlasSolveGate", [360, 520], [400, 200], "✅ Solve gate")
    w.link(solve, 0, gate, "solve"); w.link(g2, 0, gate, "source_image")
    b.rset("solve", gate, 0, [800, 520])
    w.note([800, 40], [620, 440],
           "🩻 X-RAY CAMERA MOVE — the single-photo → Nuke dolly\n\n"
           "The problem a camera move exposes: everything the camera could not\n"
           "see is a HOLE. Orbit or dolly and the occluded background reads as\n"
           "black.\n\n"
           "This build predicts what is BEHIND the occluders (LaRI layered ray\n"
           "intersections), then paints it with a LaMa clean plate:\n"
           "  · the X-ray gives GEOMETRY\n"
           "  · the inpaint gives PIXELS\n"
           "  · the base mesh keeps projecting the ORIGINAL photo\n\n"
           "EXPERIMENTAL + research-only. LaRI has NO upstream license — you\n"
           "clone it yourself and point ATLAS_LARI_PATH at it. Needs\n"
           "ATLAS_EXPERIMENTAL=1 set BEFORE python starts (see run_nvidia_gpu.bat).")

    w.group("1 · DEPTH + 🩻 X-RAY — LaRI predicts the surfaces hidden behind the temple", [1540, -40, 1500, 860], "#535")
    g3 = b.rget("plate", [1580, 40]); g4 = b.rget("solve", [1580, 120])
    dm = w.node("AtlasDepthMap", [1580, 220], [400, 130], "Depth (MoGe)",
                {"depth_model": "Ruicheng/moge-2-vitl-normal"})
    w.link(g3, 0, dm, "image"); w.link(g4, 0, dm, "solve")
    b.rset("depth", dm, 0, [1580, 380])
    g5 = b.rget("plate", [2020, 40])
    xray = w.node("AtlasPredictHiddenGeometry", [2020, 140], [420, 380], "🩻 X-ray (LaRI)",
                  {"model": "lari-scene", "clear_rel": 0.15, "min_clear_m": 0.05,
                   "smooth_px": 31, "fill_gaps": True, "seed": 0})
    w.link(dm, 0, xray, "depth"); w.link(g5, 0, xray, "image")
    b.rset("depth_xray", xray, 0, [2480, 140])
    b.rset("hidden_mask", xray, 1, [2480, 220])
    b.rset("paint_matte", xray, 3, [2480, 300])
    w.note([2020, 560], [1000, 250],
           "CALIBRATION — six measurement rounds, all load-bearing:\n"
           "· fill_gaps ON + smooth_px=31 GAUSSIAN. Fragmented predictions shred the layer mesh via the world-edge check,\n"
           "  and that is IMMUNE to depth_edge_rel / grid / dilation (all measured no-ops). A MEDIAN filter is edge-preserving\n"
           "  and keeps the very steps that tear — gaussian measured 0.260 hole-in-paint vs median 0.455 on this exact plate.\n"
           "· seed PINNED. ComfyUI auto-adds 'randomize' to any widget named seed — it silently re-rolls generative geometry.\n"
           "· Domain-bounded: strong on architecture (cathedral: 76% of occluder pixels get plausible continuation), can\n"
           "  collapse to ~0 coverage on open terrain — then the depth passes through nearly unchanged (graceful).")

    w.group("2 · CLEAN PLATE — LaMa paints the revealed geometry. ✂ crop spends LaMa's fixed 256² on the reveal, not the 4K frame", [3080, -40, 1560, 860], "#453")
    lama = w.node("INPAINT_LoadInpaintModel", [3120, 40], [300, 60], "LaMa", {"model_name": "big-lama.pt"})
    g6 = b.rget("hidden_mask", [3120, 140])
    ex = w.node("INPAINT_ExpandMask", [3120, 240], [280, 110], "Expand reveal",
                {"grow": 48, "blur": 16, "blur_type": "gaussian"})
    w.link(g6, 0, ex, "mask")
    g7 = b.rget("plate", [3120, 400])
    cr = w.node("AtlasInpaintCrop", [3440, 240], [300, 100], "✂ Crop", {"context_pad_px": 128})
    w.link(g7, 0, cr, "image"); w.link(ex, 0, cr, "mask")
    pa = w.node("INPAINT_InpaintWithModel", [3780, 240], [300, 130], "LaMa (seed PINNED)", {"seed": 0})
    w.link(lama, 0, pa, "inpaint_model"); w.link(cr, 0, pa, "image"); w.link(cr, 1, pa, "mask")
    g8 = b.rget("plate", [3780, 420])
    st = w.node("AtlasInpaintStitch", [4120, 240], [300, 110], "✂ Stitch")
    w.link(g8, 0, st, "original_image"); w.link(pa, 0, st, "inpainted_crop")
    w.link(cr, 2, st, "crop_region")
    b.rset("xray_plate", st, 0, [4120, 400])

    w.group("3 · BASE MESH + 🩻 X-RAY LAYER — base projects the ORIGINAL photo; the layer uses MASK MEMBERSHIP, not a depth band", [4680, -40, 1560, 860], "#534")
    g9 = b.rget("solve", [4720, 40]); g10 = b.rget("depth", [4720, 120])
    base = w.node("AtlasDeriveReliefMesh", [4720, 220], [400, 260], "Base relief (original photo)",
                  {"relief_grid": 512, "depth_edge_rel": 1.0, "max_edge_factor": 40.0,
                   "sky_heuristic": True})
    w.link(g9, 0, base, "solve"); w.link(g10, 0, base, "depth")
    b.rset("solve_base", base, 0, [5160, 220])
    g11 = b.rget("hidden_mask", [4720, 520])
    grow = w.node("GrowMask", [4720, 620], [280, 90], "Grow 32", {"expand": 32, "tapered_corners": True})
    w.link(g11, 0, grow, "mask")
    inv = w.node("InvertMask", [5020, 620], [260, 60], "NOT(hidden) → exclude")
    w.link(grow, 0, inv, "mask")
    b.rset("xray_exclude", inv, 0, [5300, 620])
    g12 = b.rget("solve_base", [5460, 40]); g13 = b.rget("depth_xray", [5460, 120])
    g14 = b.rget("xray_plate", [5460, 200]); g15 = b.rget("xray_exclude", [5460, 280])
    g16 = b.rget("paint_matte", [5460, 360])
    xl = w.node("AtlasCleanPlateLayer", [5760, 220], [400, 420], "🩻 X-ray layer",
                {"name": "xray", "priority": 5.0, "band_side": "manual", "band_geometry": "relief",
                 "relief_grid": 384, "depth_edge_rel": 1.5, "far_pct": 1.0,
                 "fill_occluded": False, "embed_matte": True, "skirt_bevel": 1.5})
    w.link(g12, 0, xl, "solve"); w.link(g13, 0, xl, "depth")
    w.link(g14, 0, xl, "plate_image"); w.link(g15, 0, xl, "exclude_mask")
    w.link(g16, 0, xl, "layer_matte")
    b.rset("solve_xray", xl, 0, [6200, 220])
    w.note([4720, 720], [1480, 110],
           "A DEPTH BAND CANNOT HOLD THIS LAYER — measured: every split 0.30–0.55 lost 76–97% of the predictions, because surfaces\n"
           "hidden behind NEAR occluders are themselves near. So the layer uses MASK MEMBERSHIP: hidden_mask→Grow→Invert→exclude_mask\n"
           "(the geometry region) and paint_matte→layer_matte (the paint). band_side=manual, far uncapped. No split node at all.")

    w.group("4 · 🎥 CAMERA MOVE + EXPORT — author the move in the viewport (one-click move buttons), ⏺ Bake, then ship the .nk / .usda", [6280, -40, 1700, 900], "#446")
    g17 = b.rget("solve_xray", [6320, 40]); g18 = b.rget("plate", [6320, 120])
    vp = w.node("AtlasBlockoutViewport", [6320, 220], [960, 640], "🎥 Viewport — author + ⏺ Bake the move",
                {"resolution": 1024, "preview_expand": 1.0})
    w.link(g17, 0, vp, "solve"); w.link(g18, 0, vp, "source_image")
    g19 = b.rget("solve_xray", [7340, 40])
    cpu = w.node("AtlasExportCameraPathUSD", [7340, 140], [340, 130], "Camera path → .usda",
                 {"output_dir": "atlas_exports/jungle_xray"})
    w.link(g19, 0, cpu, "solve"); w.link(vp, 5, cpu, "camera_path")
    cpu["mode"] = 4   # MUTED — errors by design until ⏺ Bake has produced a path
    g20 = b.rget("solve_xray", [7340, 320])
    nk = w.node("AtlasExportNukeLayers", [7340, 420], [340, 130], "🎞 Nuke layers (.nk)",
                {"output_dir": "atlas_exports/jungle_xray"})
    w.link(g20, 0, nk, "solve")
    w.note([7340, 580], [640, 280],
           "THE MOVE\n\n"
           "1. 📽 Project ON (default). Orbit to check coverage.\n"
           "2. 🧭 Safe Zone probe-renders around the recovered camera and CLAMPS\n"
           "   the orbit to the measured hole-free arc — your move stays inside\n"
           "   what actually exists.\n"
           "3. 🎥 Camera Path → click a move (Orbit L/R, Pan L/R, Dolly In\n"
           "   — 24fps, 100 frames, auto-previews) → ⏺ Bake Path.\n"
           "4. Bake fills path_frames (→ a Video Combine node, NOT installed\n"
           "   here) and camera_path (→ the .usda export, MUTED until baked —\n"
           "   un-mute it after ⏺ Bake or it errors by design).\n\n"
           "The .nk carries every layer with its OWN camera; the move can also\n"
           "be authored in Nuke against the same geometry.")
    return b, "atlas_jungle_xray_cameramove"


# ══════════════════════════════════════════════════════════════════════════
# 3 · DCC / VFX HANDOFF  (castle)
# ══════════════════════════════════════════════════════════════════════════
def build_dcc():
    b = Builder(); w = b.w
    PLATE = "atlas_seacliff_castle.png"

    w.group("0 · PLATE REGISTRATION (Output Desk) — records the DURABLE file path + colorspace so exporters never mistake the 8-bit preview for final data", [-40, -40, 1500, 800], "#355")
    load = w.raw("LoadImage", [0, 40], [320, 320], "Castle plate", [PLATE, "image"], [],
                 [{"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0},
                  {"name": "MASK", "type": "MASK", "links": [], "slot_index": 1}])
    reg = w.node("AtlasRegisterPlate", [360, 40], [420, 220], "Register plate",
                 {"plate_path": "", "colorspace": "sRGB", "bit_depth": "auto", "role": "source"})
    w.link(load, 0, reg, "image")
    b.rset("plate", reg, 0, [360, 300])
    b.rset("plate_ref", reg, 1, [360, 380])
    w.note([820, 40], [620, 400],
           "🎨 OUTPUT DESK — why register the plate\n\n"
           "plate_path is BLANK here, so the ref is marked is_proxy=True and the\n"
           "exporters correctly refuse to treat this browser/8-bit preview as\n"
           "final data — they author a PNG instead.\n\n"
           "For a real VFX handoff: put the ORIGINAL .exr path in plate_path and\n"
           "set colorspace=ACEScg. The Nuke Read / Maya file nodes then point at\n"
           "that float original instead of the preview, and bit_depth=auto infers\n"
           "16f/32f from the .exr extension.\n\n"
           "For the full float path use OCIORead (ComfyUI-OCIO) + the shipping\n"
           "atlas_input_ocio_quickstart_workflow.json. Needs OPENCV_IO_ENABLE_\n"
           "OPENEXR=1 set BEFORE python starts, and opencv-python 4.x (5.x\n"
           "dropped the EXR codec).")

    w.group("1 · SOLVE + ✅ GATE + attach the plate", [1540, -40, 1200, 800], "#353")
    g1 = b.rget("plate", [1580, 40])
    solve = w.node("AtlasLearnedSolveFromImage", [1580, 140], [400, 240], "Learned solve",
                   {"height_mode": "assume", "camera_height_m": 40.0})
    w.link(g1, 0, solve, "image")
    g2 = b.rget("plate", [1580, 420])
    gate = w.node("AtlasSolveGate", [1580, 520], [400, 200], "✅ Solve gate")
    w.link(solve, 0, gate, "solve"); w.link(g2, 0, gate, "source_image")
    g3 = b.rget("plate_ref", [2020, 40])
    att = w.node("AtlasAttachSourcePlate", [2020, 140], [340, 100], "Attach source plate")
    w.link(gate, 0, att, "solve"); w.link(g3, 0, att, "plate_ref")
    b.rset("solve", att, 0, [2400, 140])

    w.group("2 · DEPTH + LAYERS — a sky card and the castle relief. The DCC layer exports need at least ONE ProjectionSource", [2780, -40, 1800, 800], "#335")
    g4 = b.rget("plate", [2820, 40]); g5 = b.rget("solve", [2820, 120])
    dm = w.node("AtlasDepthMap", [2820, 220], [400, 130], "Depth (V2-Outdoor)",
                {"depth_model": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"})
    w.link(g4, 0, dm, "image"); w.link(g5, 0, dm, "solve")
    b.rset("depth", dm, 0, [2820, 380])
    g6 = b.rget("plate", [3260, 40])
    sam = w.node("SAM3Segment", [3260, 140], [340, 300], "SAM3 · sky", {"prompt": "sky"})
    w.link(g6, 0, sam, "image")
    b.rset("sky_matte", sam, 1, [3260, 470])
    g7 = b.rget("solve", [3640, 40]); g8 = b.rget("depth", [3640, 120])
    g9 = b.rget("sky_matte", [3640, 200]); g10 = b.rget("plate", [3640, 280])
    sky = w.node("AtlasSkyDomeLayer", [3940, 140], [400, 300], "☁ Sky card",
                 {"radius_m": 800.0, "distance_m": 1200.0, "relief_grid": 96, "name": "sky",
                  "priority": -10.0, "edge_extend_px": 48, "frame_outpaint_px": 96})
    w.link(g7, 0, sky, "solve"); w.link(g8, 0, sky, "depth")
    w.link(g9, 0, sky, "sky_mask"); w.link(g10, 0, sky, "plate_image")
    g11 = b.rget("plate", [3940, 480])
    body = w.node("AtlasCleanPlateLayer", [4180, 480], [400, 300], "🏰 Castle body layer",
                  {"name": "body", "priority": 0.0, "band_side": "manual", "band_geometry": "relief",
                   "relief_grid": 384, "depth_edge_rel": 1.5, "near_pct": 0.0, "far_pct": 1.0,
                   "embed_matte": True})
    w.link(sky, 0, body, "solve"); w.link(g8, 0, body, "depth")
    w.link(g11, 0, body, "plate_image")
    b.rset("solve_layers", body, 0, [4180, 800 - 100])

    w.group("3 · 🎞 DCC EXPORTS — the same layer collection feeds Nuke and Maya, so the two DCCs can never drift", [4620, -40, 1500, 800], "#443")
    g12 = b.rget("solve_layers", [4660, 40])
    nk = w.node("AtlasExportNukeLayers", [4660, 140], [360, 130], "🎞 Nuke layers (.nk)",
                {"output_dir": "atlas_exports/castle_dcc"})
    w.link(g12, 0, nk, "solve")
    g13 = b.rget("solve_layers", [4660, 320])
    ma = w.node("AtlasExportMayaLayers", [4660, 420], [360, 130], "🧊 Maya layers (.ma)",
                {"output_dir": "atlas_exports/castle_dcc"})
    w.link(g13, 0, ma, "solve")
    g14 = b.rget("solve_layers", [4660, 600])
    us = w.node("AtlasExportUSD", [4660, 690], [360, 90], "USD camera (.usda)",
                {"output_dir": "atlas_exports/castle_dcc"})
    w.link(g14, 0, us, "solve")
    w.note([5060, 140], [1020, 620],
           "WHAT EACH EXPORT ACTUALLY GIVES YOU\n\n"
           "🎞 .nk — EVERY ProjectionSource as one native script: per-layer Read +\n"
           "its OWN Camera2 (patches orbit, outpainted skies widen) + Project3D2 +\n"
           "ReadGeo2, merged through one Scene into a ScanlineRender from the\n"
           "primary camera. Layer overlap resolves by real z-depth. Drag-and-drop;\n"
           "no Script Editor. The render cam is wired by a Root onScriptLoad\n"
           "callback — .nk's push/pop stack cannot re-resolve a Camera2 already\n"
           "consumed by Project3D2 (reverse-engineered by round-tripping, not read\n"
           "in a manual).\n\n"
           "🧊 .ma — the Maya twin: per-layer projector cameras as NATIVE nodes +\n"
           "an on-open scriptNode that imports the OBJs and builds the projection\n"
           "networks. Verified live in Maya 2027 via mayapy (37 checks), which\n"
           "caught two real bugs: the `projection` node has NO focalLength/aperture\n"
           "(the frustum comes solely from cameraShape.message → linkedCamera), and\n"
           "Maya's OBJ importer lands raw values as internal CENTIMETRES whatever\n"
           "the scene unit — so imported groups get a ×100 that MUST scale about\n"
           "the WORLD ORIGIN, not the import pivot.\n\n"
           "USD — camera only (camera.usda). Needs the [usd] extra (usd-core).\n\n"
           "Both layer exports need at least ONE ProjectionSource — they degrade\n"
           "gracefully at zero rather than crashing the queue.")

    w.group("4 · 🔧 EXPORT MESH + HOLE FILL + PREVIEW — the OBJ/GLB for Maya/ZBrush retopo", [6160, -40, 1800, 900], "#453")
    g15 = b.rget("solve", [6200, 40]); g16 = b.rget("depth", [6200, 120])
    der = w.node("AtlasDeriveReliefMesh", [6200, 220], [400, 260], "Relief mesh",
                 {"relief_grid": 1024, "depth_edge_rel": 1.5, "max_edge_factor": 12.0})
    w.link(g15, 0, der, "solve"); w.link(g16, 0, der, "depth")
    g17 = b.rget("plate", [6200, 520])
    exp = w.node("AtlasExportReliefMesh", [6640, 220], [400, 480], "🔧 Export + hole fill",
                 {"output_dir": "atlas_exports/castle_dcc", "use_solve_mesh": True,
                  "format": "both", "fill_interior_holes": True, "max_hole_edges": 128})
    w.link(der, 0, exp, "solve"); w.link(g17, 0, exp, "image")
    g18 = b.rget("plate", [7080, 40])
    vp = w.node("AtlasBlockoutViewport", [7080, 140], [840, 700], "🔍 Fill preview (what ships)",
                {"resolution": 768, "preview_expand": 1.0})
    w.link(exp, 2, vp, "solve"); w.link(g18, 0, vp, "source_image")
    w.note([6200, 560], [400, 280],
           "🔧 The OBJ is a DIFFERENT product from the .nk/.ma layers.\n\n"
           "AtlasExportReliefMesh reads projection_scene.proxy_geometry (what\n"
           "AtlasDeriveReliefMesh writes). The layer exports read\n"
           "projection_sources (what the clean-plate/sky nodes append). Hence\n"
           "two branches off one solve.\n\n"
           "use_solve_mesh=true inherits the derive node's grid/edge tuning\n"
           "verbatim, so the OBJ matches the viewport and no second depth\n"
           "inference runs.\n\n"
           "Wire preview_solve → the viewport to SEE the fill; it is invisible\n"
           "in a normal viewport by design.")
    return b, "atlas_castle_dcc_handoff"


for fn in (build_xray, build_dcc):
    b, name = fn()
    out = OUTDIR / f"{name}.json"
    d = b.w.dump(); d["id"] = name
    json.dump(d, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"wrote {out.name}: nodes={len(b.w.nodes)} links={len(b.w.links)} "
          f"groups={len(b.w.groups)} rails={len(b.sets)}")
