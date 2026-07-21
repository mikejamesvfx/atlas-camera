"""Generate the Atlas v1 SHOWCASE workflow set (examples/showcase/).

Nine plates, nine stories — collectively exercising EVERY Atlas node (the
2026-07-16 coverage audit found 33 of 60 classes in no workflow). Shares the
castle generator's helpers so every positional widgets_values list is derived
from the LIVE /object_info and cannot drift.

Usage:
    python tools/generate_showcase_workflows.py <object_info.json> <outdir> \
        <path-to-generate_castle_dmp_workflow.py> [only_names...]

The X-ray workflow (atlas_xray_wreck) references experimental nodes — generate
it against an object_info fetched from a server started with
ATLAS_EXPERIMENTAL=1, e.g.:
    python ... oi_experimental.json examples/showcase gen.py atlas_xray_wreck
"""
import importlib.util
import json
import pathlib
import sys

GEN = pathlib.Path(sys.argv[3])
OI_PATH, OUTDIR = sys.argv[1], pathlib.Path(sys.argv[2])
ONLY = set(sys.argv[4:])

_tmp = OUTDIR / "_scratch_ignore.json"
_argv = list(sys.argv)
sys.argv = ["gen", OI_PATH, str(_tmp)]
spec = importlib.util.spec_from_file_location("castlegen", GEN)
cg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cg)
sys.argv = _argv
_tmp.unlink(missing_ok=True)

WF, RAIL = cg.WF, ("#2a363b", "#3f5159")

# Repo-relative EXR location — matches generate_canonical_ocio_dcc_workflows.py
# (`_REL_EXR_DIR = examples/images`) so a fresh clone / any machine resolves it.
# Was an absolute authoring-machine path (C:\Users\miike\…) which baked into
# every generated showcase and broke on every other box — a Mac reviewer had to
# repoint it by hand. Forward slashes: portable, and ComfyUI accepts them on
# Windows. Users drop the separately-distributed float plates into examples/images/.
EXR_DIR = "examples/images"
VLM = {"provider": "lmstudio", "model": "google/gemma-4-12b-qat",
       "offload_model": True, "auto_continue": True}
V2_OUT = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
V2_IN = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
MOGE = "Ruicheng/moge-2-vitl-normal"


class Builder:
    def __init__(self, wf_id):
        self.w = WF()
        self.wf_id = wf_id
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

    def load_image(self, filename, pos, title):
        return self.w.raw("LoadImage", pos, [320, 320], title, [filename, "image"], [],
                          [{"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0},
                           {"name": "MASK", "type": "MASK", "links": [], "slot_index": 1}])

    def dump(self):
        d = self.w.dump()
        d["id"] = self.wf_id
        return d


# ══════════════════════════════════════════════════════════════════════════
# W1 · SOLVE LAB — coastal alley (VP solve, diagnostics, scale tiers, legacy)
# ══════════════════════════════════════════════════════════════════════════
def build_solve_lab():
    b = Builder("atlas-showcase-solve-lab")
    w = b.w
    w.group("1 · PLATE + GEOMETRIC VP SOLVE — strong alley lines are the classical solver's home turf", [-40, -40, 800, 560], "#355")
    load = b.load_image("coastal_alley_for_vlm.png", [0, 40], "Coastal alley (1408×768)")
    vp = w.node("AtlasSolveFromImage", [360, 40], [340, 170], "Geometric VP solve (detect ON)")
    w.link(load, 0, vp, "image")
    b.rset("plate", load, 0, [0, 400])
    b.rset("solve", vp, 0, [360, 260])
    w.note([360, 360], [340, 120],
           "SOLVE LAB — the teaching workflow.\nEverything a solve carries, made visible:\n"
           "VP fan-lines, horizon, ground, metric scale tiers,\nJSON round-trip, and the legacy/artist-guided paths.")

    w.group("2 · DIAGNOSTICS — see what was recovered", [800, -40, 1180, 700], "#345")
    g_img = b.rget("plate", [840, 0])
    g_s1 = b.rget("solve", [840, 80])
    viz = w.node("AtlasVPVisualization", [1100, 0], [300, 130], "VP fan-lines + horizon")
    w.link(g_img, 0, viz, "image")
    w.link(g_s1, 0, viz, "solve")
    pv1 = w.node("PreviewImage", [1440, 0], [260, 250], "VP overlay")
    w.link(viz, 0, pv1, "images")
    g_s2 = b.rget("solve", [840, 180])
    hz = w.node("AtlasHorizonMask", [1100, 180], [300, 120], "Horizon mask (0=ground)")
    w.link(g_s2, 0, hz, "solve")
    m2i = w.node("MaskToImage", [1440, 260], [220, 60], None)
    w.link(hz, 0, m2i, "mask")
    pv2 = w.node("PreviewImage", [1700, 180], [240, 230], "Sky / ground split")
    w.link(m2i, 0, pv2, "images")
    g_s3 = b.rget("solve", [840, 340])
    gd = w.node("AtlasGroundDepthMap", [1100, 340], [300, 140], "Analytic ground depth",
                {"near_m": 1.0, "far_m": 30.0})
    w.link(g_s3, 0, gd, "solve")
    pv3 = w.node("PreviewImage", [1440, 340], [240, 230], "Ground depth (Y=0 plane)")
    w.link(gd, 0, pv3, "images")
    g_s3b = b.rget("solve", [840, 440])
    gm = w.node("AtlasGroundMask", [1100, 520], [300, 100], "Ground mask")
    w.link(g_s3b, 0, gm, "solve")
    m2i2 = w.node("MaskToImage", [1440, 590], [220, 60], None)
    w.link(gm, 0, m2i2, "mask")
    g_img2 = b.rget("plate", [840, 540])
    g_s4 = b.rget("solve", [840, 620])
    da = w.node("AtlasDepthAnything", [1100, 640], [300, 130],
                "Depth PREVIEW (lossy IMAGE — see note)")
    w.link(g_img2, 0, da, "image")
    w.link(g_s4, 0, da, "solve")
    pv4 = w.node("PreviewImage", [1700, 470], [240, 230], "Depth preview")
    w.link(da, 0, pv4, "images")
    w.note([1100, 800], [560, 130],
           "AtlasDepthAnything outputs a NORMALIZED IMAGE preview — real near/far\n"
           "distances are discarded. For metric geometry always use AtlasDepthMap\n"
           "(ATLAS_DEPTH_MAP carrier). This node is QA eyes only.")

    w.group("3 · DECOMPOSE + JSON ROUND-TRIP — the viewport shows the RELOADED solve", [-40, 560, 800, 900], "#435")
    g_s5 = b.rget("solve", [0, 620])
    dec = w.node("AtlasDecomposeSolve", [240, 620], [300, 190], "Decompose solve")
    w.link(g_s5, 0, dec, "solve")
    deccam = w.node("AtlasDecomposeCamera", [240, 860], [300, 210], "Decompose camera")
    w.link(dec, 0, deccam, "camera")
    show_f = w.node("Display Any (rgthree)", [580, 860], [180, 90], "focal px")
    w.link(deccam, 0, show_f, "source")
    show_fov = w.node("Display Any (rgthree)", [580, 990], [180, 90], "fov h°")
    w.link(deccam, 8, show_fov, "source")
    g_s6 = b.rget("solve", [0, 720])
    expj = w.node("AtlasExportSolveJSON", [240, 1120], [320, 100], "Save solve JSON",
                  {"output_path": "atlas_exports/showcase_alley/alley_solve.json"})
    w.link(g_s6, 0, expj, "solve")
    loadj = w.node("AtlasLoadSolveJSON", [600, 1120], [320, 100],
                   "Reload it (path wired = ordered)")
    w.link(expj, 0, loadj, "json_path")
    vpn = w.node("AtlasBlockoutViewport", [0, 1280], [760, 170], "Viewport — RELOADED solve",
                 {"resolution": 1024})
    w.link(loadj, 0, vpn, "solve")
    g_img3 = b.rget("plate", [0, 1200])
    w.link(g_img3, 0, vpn, "source_image")

    w.group("4 · METRIC SCALE TIERS — reference object beats VLM cues beats assumed default", [2020, -40, 1060, 900], "#453")
    g_img4 = b.rget("plate", [2060, 0])
    cues = w.node("AtlasVLMScaleCues", [2320, 0], [340, 170], "🔎 VLM scale cues (tier 1 feed)",
                  {"provider": "lmstudio", "model": "google/gemma-4-12b-qat"})
    w.link(g_img4, 0, cues, "image")
    show_sum = w.node("ShowText|pysssss", [2700, 0], [340, 160], "Cue summary")
    w.link(cues, 1, show_sum, "text")
    g_s7 = b.rget("solve", [2060, 220])
    appl = w.node("AtlasApplyScaleReferences", [2320, 220], [340, 140],
                  "Apply cues (confirm ON)", {"confirm": True})
    w.link(g_s7, 0, appl, "solve")
    w.link(cues, 0, appl, "scale_references")
    show_rep = w.node("ShowText|pysssss", [2700, 220], [340, 160], "Scale report")
    w.link(appl, 2, show_rep, "text")
    g_s8 = b.rget("solve", [2060, 440])
    ref = w.node("AtlasReferenceScaleSolve", [2320, 440], [340, 220],
                 "Door = 2.1m reference (recalibrate bbox by eye)",
                 {"reference_id": "door_210cm", "bbox_x0": 96.0, "bbox_y0": 300.0,
                  "bbox_x1": 214.0, "bbox_y1": 640.0})
    w.link(g_s8, 0, ref, "solve")
    show_h = w.node("Display Any (rgthree)", [2700, 440], [180, 90], "camera height m")
    w.link(ref, 1, show_h, "source")
    w.note([2320, 720], [620, 130],
           "Tiered scale doctrine: a known-size REFERENCE (door) is tier 1;\n"
           "VLM cues feed the same tier but only apply with confirm=ON;\n"
           "depth-measured ground is tier 2; the 1.6m default is a flagged last resort.")

    w.group("5 · ARTIST-GUIDED + LEGACY corner", [2020, 900, 1060, 560], "#533")
    g_img5 = b.rget("plate", [2060, 960])
    cons = w.node("AtlasConstrainedSolve", [2320, 960], [380, 150], "Artist-guided (explicit VPs)",
                  {"constraints_json": '{"vanishing_points": {"left": [704, 400], "right": [60000, 400]}, "camera_height": 1.6}'})
    w.link(g_img5, 0, cons, "image")
    dec2 = w.node("AtlasDecomposeSolve", [2740, 960], [300, 190], "Inspect guided solve")
    w.link(cons, 0, dec2, "solve")
    show_j = w.node("Display Any (rgthree)", [2740, 1190], [180, 90], "guided confidence")
    w.link(dec2, 1, show_j, "source")
    # (The legacy AtlasLoadImageSolveCamera station was removed in 0.8.1 — the
    # deprecated file-path solve node is gone; AtlasSolveFromImage /
    # AtlasLearnedSolveFromImage are the tensor-based replacements shown above.)
    return b.dump()


# ══════════════════════════════════════════════════════════════════════════
# W2 · CITY BLOCKS — New York birdseye (aerial preset, ShotCam, Maya review)
# ══════════════════════════════════════════════════════════════════════════
def build_city_blocks():
    b = Builder("atlas-showcase-city-blocks")
    w = b.w
    w.group("1 · REAL PHOTO + LEARNED SOLVE — a 7360×4912 DSLR frame, not an AI plate", [-40, -40, 820, 620], "#355")
    load = b.load_image("newyork_Birdseye.png", [0, 40], "NYC birdseye (REAL photo, 177MB)")
    solve = w.node("AtlasLearnedSolveFromImage", [360, 40], [400, 230], "Learned solve (GeoCalib)",
                   {"depth_model": V2_OUT})
    w.link(load, 0, solve, "image")
    scale = w.node("AtlasReferenceScaleSolve", [360, 330], [400, 280],
                   "📐 COUNT THE STOREYS → tier-1 metric scale",
                   {"reference_id": "building_story_3m",
                    "bbox_x0": 3820.0, "bbox_y0": 1775.0,
                    "bbox_x1": 4470.0, "bbox_y1": 2480.0,
                    "height_override_m": 17.5})
    w.link(solve, 0, scale, "solve")
    b.rset("plate", load, 0, [0, 400])
    b.rset("solve", scale, 0, [360, 660])
    w.note([0, 480], [320, 240],
           "Heads-up: this plate is HUGE (177MB PNG). The browser preview is heavy —\n"
           "server-side solve/derive are fine. Real photography: GeoCalib gravity is\n"
           "trustworthy here (no roll trim needed) — but the street-level ground fit\n"
           "fails from this vantage (cars break it) → 1.6m fallback. THE FIX, and the\n"
           "doctrine for any sky-rise plate: COUNT THE LEVELS. The white tenement mid-\n"
           "frame ('PALM TOO') is 5 storeys × 3.5m = 17.5m; its bbox + that height in\n"
           "AtlasReferenceScaleSolve locks tier-1 metric scale by single-view geometry\n"
           "→ camera ≈ 60-64m up (photographer-confirmed high vantage). Every metre\n"
           "downstream is now MEASURED, not dialed.")

    w.group("2 · AERIAL PRESET — one node, buildings-as-boxes over a relief ground", [820, -40, 760, 620], "#345")
    g_i = b.rget("plate", [860, 0])
    g_s = b.rget("solve", [860, 80])
    dpg = w.node("AtlasDeriveProjectionGeometry", [1120, 0], [400, 420],
                 "scene_type=aerial (boxes + relief)",
                 {"scene_type": "aerial", "relief_grid": 256, "depth_model": V2_OUT})
    w.link(g_s, 0, dpg, "solve")
    w.link(g_i, 0, dpg, "image")
    b.rset("geo", dpg, 0, [1120, 480])

    w.group("3 · SHOTCAM CONFORM + VIEWPORT", [1580, -40, 1240, 620], "#435")
    shot = w.node("AtlasDefineShotCam", [1620, 0], [320, 170], "Project format (S35, 32mm, 1920)",
                  {"focal_length_mm": 32.0})
    g_g1 = b.rget("geo", [1620, 220])
    g_i2 = b.rget("plate", [1620, 300])
    vpn = w.node("AtlasBlockoutViewport", [1980, 0], [800, 500], "Viewport (shot-cam conformed)",
                 {"resolution": 1280})
    w.link(g_g1, 0, vpn, "solve")
    w.link(g_i2, 0, vpn, "source_image")
    w.link(shot, 0, vpn, "shot_cam")

    w.group("4 · MAYA REVIEW + REVIEW PACKAGE", [820, 620, 2000, 520], "#453")
    g_g2 = b.rget("geo", [860, 680])
    g_i3 = b.rget("plate", [860, 760])
    relief = w.node("AtlasExportReliefMesh", [1120, 680], [380, 420], "Relief OBJ (for the Maya card)",
                    {"output_dir": "atlas_exports/showcase_newyork"})
    w.link(g_g2, 0, relief, "solve")
    w.link(g_i3, 0, relief, "image")
    g_g3 = b.rget("geo", [1540, 680])
    maya = w.node("AtlasExportMayaReviewScene", [1800, 680], [380, 160],
                  "Maya review scene (+ real relief mesh)",
                  {"output_dir": "atlas_exports/showcase_newyork"})
    w.link(g_g3, 0, maya, "solve")
    w.link(relief, 0, maya, "relief_mesh_obj_path")
    g_g4 = b.rget("geo", [1540, 880])
    pkg = w.node("AtlasExportReviewPackage", [1800, 900], [380, 120], "Full review package",
                 {"output_dir": "atlas_exports/showcase_newyork_review"})
    w.link(g_g4, 0, pkg, "solve")
    w.note([2220, 680], [560, 300],
           "CITY BLOCKS — the aerial one-node preset.\n\n"
           "scene_type=aerial resolves to geometry_mode=both + azimuth_walls +\n"
           "max_objects=6: buildings become boxes over a relief ground/treeline.\n"
           "The ShotCam conforms the viewport render to a 1920 S35 project format\n"
           "without touching how the photo projects onto geometry.\n\n"
           "Exports: Maya REVIEW scene (image card + proxies + the real relief\n"
           "OBJ) and the full review package (report + JSON + overlays).")
    return b.dump()


# ══════════════════════════════════════════════════════════════════════════
# W3 · COMPOSABLE GEOMETRY — temple city (walls/towers/roofs/relief merged)
# ══════════════════════════════════════════════════════════════════════════
def build_composable():
    b = Builder("atlas-showcase-composable-geometry")
    w = b.w
    w.group("1 · PLATE + SOLVE + 📐 SCALE OVERRIDE — AI vista ⇒ assumed 1.6m is ~10× small", [-40, -40, 820, 700], "#355")
    load = b.load_image("atlas_00022_templecity.png", [0, 40], "Temple city (9216×3840)")
    solve = w.node("AtlasLearnedSolveFromImage", [360, 40], [400, 230], "Learned solve",
                   {"height_mode": "assume", "depth_model": V2_OUT})
    w.link(load, 0, solve, "image")
    scale = w.node("AtlasScaleOverride", [360, 330], [400, 150], "📐 Camera height → 16m",
                   {"camera_height_m": 16.0})
    w.link(solve, 0, scale, "solve")
    b.rset("plate", load, 0, [0, 400])
    b.rset("solve", scale, 0, [360, 540])
    w.note([0, 480], [320, 180],
           "AI cityscape vistas solve to the assumed 1.6m eye height\n"
           "(scale_source=assumed_default) — roughly 10× small for this\n"
           "elevated view. The 📐 dial sets 16m (calibrated for this plate)\n"
           "and EVERY downstream metric follows.")

    w.group("2 · ONE DEPTH, FOUR STRATEGIES — estimate once, derive four ways", [820, -40, 900, 1000], "#345")
    g_i = b.rget("plate", [860, 0])
    g_s0 = b.rget("solve", [860, 80])
    dm = w.node("AtlasDepthMap", [1120, 0], [340, 150], "Shared metric depth", {"depth_model": V2_OUT})
    w.link(g_i, 0, dm, "image")
    w.link(g_s0, 0, dm, "solve")
    b.rset("depth", dm, 0, [1120, 190])
    walls = w.node("AtlasDeriveWalls", [1120, 300], [380, 220], "Walls (3 depth modes/azimuth)",
                   {"max_walls": 12, "max_objects": 6, "distance_modes": 3})
    towers = w.node("AtlasDeriveTowersSpires", [1120, 560], [380, 240], "Towers + spires (roofline split)",
                    {"max_walls": 12, "roofline_split": True})
    roofs = w.node("AtlasDeriveRoofsFacades", [1120, 840], [380, 130], "Sloped roofs (RANSAC planes)",
                   {"max_planes": 10})
    for i, n in enumerate((walls, towers, roofs)):
        gs = b.rget("solve", [860, 300 + i * 260])
        gd = b.rget("depth", [860, 380 + i * 260])
        w.link(gs, 0, n, "solve")
        w.link(gd, 0, n, "depth")

    w.group("3 · RELIEF + MERGE CHAIN — the Nuke-Merge-node equivalent, ×3", [1720, -40, 900, 1000], "#435")
    g_s4 = b.rget("solve", [1760, 0])
    g_d4 = b.rget("depth", [1760, 80])
    relief = w.node("AtlasDeriveReliefMesh", [2020, 0], [380, 200], "Relief background",
                    {"relief_grid": 384, "depth_edge_rel": 1.0})
    w.link(g_s4, 0, relief, "solve")
    w.link(g_d4, 0, relief, "depth")
    m1 = w.node("AtlasMergeGeometry", [2020, 260], [300, 110], "Merge 1: walls + towers")
    w.link(walls, 0, m1, "solve_a")
    w.link(towers, 0, m1, "solve_b")
    m2 = w.node("AtlasMergeGeometry", [2020, 420], [300, 110], "Merge 2: + roofs")
    w.link(m1, 0, m2, "solve_a")
    w.link(roofs, 0, m2, "solve_b")
    m3 = w.node("AtlasMergeGeometry", [2020, 580], [300, 110], "Merge 3: + relief")
    w.link(m2, 0, m3, "solve_a")
    w.link(relief, 0, m3, "solve_b")
    b.rset("merged", m3, 0, [2020, 730])
    w.note([2020, 830], [560, 130],
           "Chaining derive nodes DIRECTLY would erase each other's geometry\n"
           "(every derive strips prior PROXY_ROLE geometry by design).\n"
           "AtlasMergeGeometry is the one explicit place branches combine;\n"
           "solve_a's camera wins and the backdrop plane is deduped.")

    w.group("4 · VIEWPORT + SINGLE-PROJECTION NUKE", [2620, -40, 1160, 1000], "#453")
    g_m1 = b.rget("merged", [2660, 0])
    g_i2 = b.rget("plate", [2660, 80])
    vpn = w.node("AtlasBlockoutViewport", [2920, 0], [820, 500], "Merged geometry viewport",
                 {"resolution": 1280})
    w.link(g_m1, 0, vpn, "solve")
    w.link(g_i2, 0, vpn, "source_image")
    g_m2 = b.rget("merged", [2660, 560])
    g_i3 = b.rget("plate", [2660, 640])
    reliefx = w.node("AtlasExportReliefMesh", [2920, 560], [380, 400], "Relief OBJ for Nuke ReadGeo",
                     {"output_dir": "atlas_exports/showcase_templecity"})
    w.link(g_m2, 0, reliefx, "solve")
    w.link(g_i3, 0, reliefx, "image")
    g_m3 = b.rget("merged", [3340, 560])
    nuke = w.node("AtlasExportNuke", [3600, 560], [340, 160], "Single-projection .nk (drag-drop)",
                  {"output_dir": "atlas_exports/showcase_templecity"})
    w.link(g_m3, 0, nuke, "solve")
    w.link(reliefx, 0, nuke, "relief_mesh_obj_path")
    return b.dump()


# ══════════════════════════════════════════════════════════════════════════
# W4 · OCIO FLOAT DMP — ocean castle (ACEScg EXR → layers → Nuke/USD)
# ══════════════════════════════════════════════════════════════════════════
def build_ocio_dmp():
    b = Builder("atlas-showcase-ocio-dmp")
    w = b.w
    exr = EXR_DIR + "/oceancastle_32bit_acescg.exr"
    w.group("0 · ACEScg EXR IN + 🧭 VLM + ✅ GATE — the float plate is the source of truth", [-40, -40, 1240, 760], "#355")
    ocio = w.node("OCIORead", [0, 40], [380, 330], "OCIORead — ACEScg float EXR",
                  {"source": exr, "input_colorspace": "ACEScg"})
    reg = w.node("AtlasRegisterPlate", [420, 40], [360, 200], "Register the FLOAT plate",
                 {"plate_path": exr, "colorspace": "ACEScg"})
    w.link(ocio, 0, reg, "image")
    assess = w.node("AtlasAssessImage", [820, 40], [400, 420], "🧭 VLM pre-flight", VLM)
    w.link(reg, 0, assess, "image")
    solve = w.node("AtlasLearnedSolveFromImage", [420, 300], [360, 230], "Learned solve",
                   {"depth_model": V2_OUT})
    w.link(assess, 0, solve, "image")
    gate = w.node("AtlasSolveGate", [420, 580], [360, 130], "✅ Approve solve, then run the stack")
    w.link(solve, 0, gate, "solve")
    w.link(assess, 0, gate, "source_image")
    att = w.node("AtlasAttachSourcePlate", [820, 580], [300, 100], "Attach float plate ref")
    w.link(gate, 0, att, "solve")
    w.link(reg, 1, att, "plate_ref")
    b.rset("plate_img", assess, 0, [0, 420])
    b.rset("plate_ref", reg, 1, [0, 500])
    b.rset("solve", att, 0, [820, 700])

    w.group("1 · ☁ SKY + SHARED DEPTH", [1240, -40, 900, 760], "#345")
    g_i = b.rget("plate_img", [1280, 0])
    sam = w.node("SAM3Segment", [1540, 0], [360, 330], "SAM3 sky", {"prompt": "sky"})
    w.link(g_i, 0, sam, "image")
    g_i2 = b.rget("plate_img", [1280, 380])
    g_s = b.rget("solve", [1280, 460])
    dm = w.node("AtlasDepthMap", [1540, 380], [340, 150], "Shared depth", {"depth_model": V2_OUT})
    w.link(g_i2, 0, dm, "image")
    w.link(g_s, 0, dm, "solve")
    b.rset("depth", dm, 0, [1540, 570])
    g_s2 = b.rget("solve", [1280, 560])
    g_i3 = b.rget("plate_img", [1280, 640])
    g_pr = b.rget("plate_ref", [1540, 660])
    dome = w.node("AtlasSkyDomeLayer", [1900, 380], [360, 300], "☁ Sky card (outpainted)",
                  {"radius_m": 900.0, "frame_outpaint_px": 96})
    w.link(g_s2, 0, dome, "solve")
    gd0 = b.rget("depth", [1280, 720])
    w.link(gd0, 0, dome, "depth")
    w.link(sam, 1, dome, "sky_mask")
    w.link(g_i3, 0, dome, "plate_image")
    w.link(g_pr, 0, dome, "plate_ref")

    w.group("2 · ONE SPLIT, TWO LAYERS — AtlasDepthBandSplit drives fg AND bg", [2140, -40, 1420, 760], "#435")
    split = w.node("AtlasDepthBandSplit", [2180, 0], [280, 110], "Band split @ P55")
    g_d1 = b.rget("depth", [2180, 160])
    g_i4 = b.rget("plate_img", [2180, 240])
    fg = w.node("AtlasCleanPlateLayer", [2500, 0], [380, 520], "🏰 Castle (foreground of the split)",
                {"name": "castle_fg", "priority": 0.0, "band_side": "foreground",
                 "embed_matte": True})
    w.link(dome, 0, fg, "solve")
    w.link(g_d1, 0, fg, "depth")
    w.link(g_i4, 0, fg, "plate_image")
    w.link(split, 0, fg, "band_split")
    # bg plate: LaMa-inpaint the fg band's footprint out of the plate
    g_d2 = b.rget("depth", [2180, 320])
    g_s3 = b.rget("solve", [2180, 400])
    dlm = w.node("AtlasDepthLayerMask", [2180, 480], [320, 220], "Occluder mask for the bg plate",
                 {"near_pct": 0.0, "far_pct": 0.55})
    w.link(g_s3, 0, dlm, "solve")
    w.link(g_d2, 0, dlm, "depth")
    lam = w.node("INPAINT_LoadInpaintModel", [2180, 740], [300, 90], "LaMa", {"model_name": "big-lama.pt"})
    exp = w.node("INPAINT_ExpandMask", [2540, 560], [280, 130], None, {"grow": 32, "blur": 8})
    w.link(dlm, 0, exp, "mask")
    g_i5 = b.rget("plate_img", [2540, 730])
    crop = w.node("AtlasInpaintCrop", [2860, 560], [300, 130], "✂ crop")
    w.link(g_i5, 0, crop, "image")
    w.link(exp, 0, crop, "mask")
    paint = w.node("INPAINT_InpaintWithModel", [2860, 740], [300, 150], "LaMa fill")
    w.link(lam, 0, paint, "inpaint_model")
    w.link(crop, 0, paint, "image")
    w.link(crop, 1, paint, "mask")
    g_i6 = b.rget("plate_img", [3200, 560])
    stitch = w.node("AtlasInpaintStitch", [3200, 660], [300, 150], "✂ stitch")
    w.link(g_i6, 0, stitch, "original_image")
    w.link(paint, 0, stitch, "inpainted_crop")
    w.link(crop, 2, stitch, "crop_region")
    g_d3 = b.rget("depth", [2920, 0])
    bg = w.node("AtlasCleanPlateLayer", [3200, 0], [380, 520], "🌊 Sea + far cliffs (background)",
                {"name": "sea_bg", "priority": 10.0, "band_side": "background",
                 "fill_occluded": True, "embed_matte": True, "edge_extend_px": 32})
    w.link(fg, 0, bg, "solve")
    w.link(g_d3, 0, bg, "depth")
    w.link(stitch, 0, bg, "plate_image")
    w.link(split, 0, bg, "band_split")

    w.group("3 · VIEWPORT + FLOAT-AWARE EXPORTS", [-40, 760, 2100, 620], "#453")
    g_f = b.rget("plate_img", [0, 820])
    vpn = w.node("AtlasBlockoutViewport", [260, 820], [820, 500], "Layered master", {"resolution": 1280})
    w.link(bg, 0, vpn, "solve")
    w.link(g_f, 0, vpn, "source_image")
    nkl = w.node("AtlasExportNukeLayers", [1140, 820], [340, 130], "🎞 Per-layer .nk (float Read paths)",
                 {"output_dir": "atlas_exports/showcase_oceancastle"})
    w.link(bg, 0, nkl, "solve")
    usd = w.node("AtlasExportUSD", [1140, 1010], [340, 110], "camera.usda",
                 {"output_dir": "atlas_exports/showcase_oceancastle"})
    w.link(bg, 0, usd, "solve")
    w.note([1520, 820], [520, 300],
           "OCIO FLOAT DMP.\n\n"
           "The registered ACEScg EXR rides the solve (AtlasAttachSourcePlate),\n"
           "so the Nuke layer export references the ORIGINAL float plate with\n"
           "colorspace='ACEScg' — the 8-bit browser preview is never mistaken\n"
           "for final data. Repoint OCIORead.source + RegisterPlate.plate_path\n"
           "at your own copy of the EXR.\n\n"
           "One AtlasDepthBandSplit feeds BOTH clean-plate layers (band_side\n"
           "foreground/background): an absolute split can't drift between them.\n"
           "Priorities are farthest-highest (sea 10 > castle 0) per the seam doctrine.")
    return b.dump()


# ══════════════════════════════════════════════════════════════════════════
# W5 · INTERIOR HANGAR — room cuboid, MoGe relight normals, patch loop
# ══════════════════════════════════════════════════════════════════════════
def build_interior_hangar():
    b = Builder("atlas-showcase-interior-hangar")
    w = b.w
    exr = EXR_DIR + "/spacehangar_32bit_acescg.exr"
    w.group("0 · ACEScg EXR + 🧭 VLM + ✅ GATE (interior: MoGe is the depth specialist)", [-40, -40, 1240, 760], "#355")
    ocio = w.node("OCIORead", [0, 40], [380, 330], "OCIORead — hangar EXR",
                  {"source": exr, "input_colorspace": "ACEScg"})
    reg = w.node("AtlasRegisterPlate", [420, 40], [360, 200], "Register float plate",
                 {"plate_path": exr, "colorspace": "ACEScg"})
    w.link(ocio, 0, reg, "image")
    assess = w.node("AtlasAssessImage", [820, 40], [400, 420], "🧭 VLM pre-flight", VLM)
    w.link(reg, 0, assess, "image")
    solve = w.node("AtlasLearnedSolveFromImage", [420, 300], [360, 230], "Learned solve (MoGe height)",
                   {"depth_model": MOGE})
    w.link(assess, 0, solve, "image")
    gate = w.node("AtlasSolveGate", [420, 580], [360, 130], "✅ Approve solve")
    w.link(solve, 0, gate, "solve")
    w.link(assess, 0, gate, "source_image")
    att = w.node("AtlasAttachSourcePlate", [820, 580], [300, 100], "Attach plate ref")
    w.link(gate, 0, att, "solve")
    w.link(reg, 1, att, "plate_ref")
    b.rset("plate_img", assess, 0, [0, 420])
    b.rset("solve", att, 0, [820, 700])

    w.group("1 · DEPTH + 🧭 DECOUPLED MoGe NORMALS + 🧩 SEMANTIC SELF-CHECK", [1240, -40, 900, 760], "#345")
    g_i = b.rget("plate_img", [1280, 0])
    g_s = b.rget("solve", [1280, 80])
    dm = w.node("AtlasDepthMap", [1540, 0], [340, 150], "V2-INDOOR depth", {"depth_model": V2_IN})
    w.link(g_i, 0, dm, "image")
    w.link(g_s, 0, dm, "solve")
    g_i2 = b.rget("plate_img", [1280, 200])
    mn = w.node("AtlasMogeNormals", [1540, 200], [340, 170], "🧭 MoGe relight normals ONTO V2 depth")
    w.link(dm, 0, mn, "depth")
    w.link(g_i2, 0, mn, "image")
    b.rset("depth", mn, 0, [1540, 420])
    show_mn = w.node("ShowText|pysssss", [1540, 520], [340, 130], "Normals report")
    w.link(mn, 1, show_mn, "text")
    g_i3 = b.rget("plate_img", [1280, 300])
    sem = w.node("AtlasSemanticMask", [1920, 0], [340, 170], "🧩 'sky' on an interior (expect ~0%)")
    w.link(g_i3, 0, sem, "image")
    show_sem = w.node("ShowText|pysssss", [1920, 220], [340, 130], "SegFormer report")
    w.link(sem, 1, show_sem, "text")

    w.group("2 · ROOM CUBOID + RELIEF + RELIGHT LAYER", [2140, -40, 980, 760], "#435")
    g_s2 = b.rget("solve", [2180, 0])
    g_d1 = b.rget("depth", [2180, 80])
    room = w.node("AtlasDeriveInteriorRoom", [2440, 0], [320, 110], "Manhattan room cuboid")
    w.link(g_s2, 0, room, "solve")
    w.link(g_d1, 0, room, "depth")
    g_s3 = b.rget("solve", [2180, 160])
    g_d2 = b.rget("depth", [2180, 240])
    relief = w.node("AtlasDeriveReliefMesh", [2440, 160], [320, 200], "Relief (interior tuning)",
                    {"relief_grid": 384, "depth_edge_rel": 1.5, "max_edge_factor": 40.0,
                     "sky_heuristic": False})
    w.link(g_s3, 0, relief, "solve")
    w.link(g_d2, 0, relief, "depth")
    merge = w.node("AtlasMergeGeometry", [2800, 80], [300, 110], "Relief wins camera; + room")
    w.link(relief, 0, merge, "solve_a")
    w.link(room, 0, merge, "solve_b")
    g_d3 = b.rget("depth", [2180, 420])
    g_i4 = b.rget("plate_img", [2180, 500])
    cpl = w.node("AtlasCleanPlateLayer", [2440, 420], [380, 300], "💡 Relight layer (normal map rides in)",
                 {"name": "hangar_relight", "relief_grid": 384, "depth_edge_rel": 1.5,
                  "embed_matte": True})
    w.link(merge, 0, cpl, "solve")
    w.link(g_d3, 0, cpl, "depth")
    w.link(g_i4, 0, cpl, "plate_image")
    b.rset("scene", cpl, 0, [2800, 640])

    w.group("3 · VIEWPORT + 📐-GATED PATCH LOOP + MAYA LAYERS", [-40, 760, 3160, 720], "#453")
    g_sc = b.rget("scene", [0, 820])
    g_i5 = b.rget("plate_img", [0, 900])
    vpn = w.node("AtlasBlockoutViewport", [260, 820], [860, 560], "Viewport — 💡 lights relight via MoGe normals",
                 {"resolution": 1280})
    w.link(g_sc, 0, vpn, "solve")
    w.link(g_i5, 0, vpn, "source_image")
    occ = w.node("AtlasOcclusionMask", [1180, 820], [380, 420], "Where the primary CAN'T cover (📐-gated)")
    g_sc2 = b.rget("scene", [1180, 1280])
    w.link(g_sc2, 0, occ, "solve")
    w.link(vpn, 0, occ, "target_image")
    w.link(vpn, 10, occ, "exact_view_override")
    m2i = w.node("MaskToImage", [1600, 820], [220, 60], None)
    w.link(occ, 0, m2i, "mask")
    pv = w.node("PreviewImage", [1600, 920], [240, 230], "Occlusion mask")
    w.link(m2i, 0, pv, "images")
    patch = w.node("AtlasAddPatchView", [1880, 820], [400, 460], "Render-conditioned patch (reuse_scene)",
                   {"name": "render_patch"})
    g_sc3 = b.rget("scene", [1880, 1320])
    w.link(g_sc3, 0, patch, "solve")
    w.link(vpn, 0, patch, "patch_image")
    w.link(vpn, 10, patch, "exact_view_override")
    pj = w.node("AtlasExportSolveJSON", [2320, 820], [340, 100], "Patched solve JSON (runs post-📐)",
                {"output_path": "atlas_exports/showcase_hangar/patched_solve.json"})
    w.link(patch, 0, pj, "solve")
    g_sc4 = b.rget("scene", [2320, 1000])
    ml = w.node("AtlasExportMayaLayers", [2600, 1000], [340, 130], "🧊 Maya per-layer scene",
                {"output_dir": "atlas_exports/showcase_hangar"})
    w.link(g_sc4, 0, ml, "solve")
    w.note([2320, 1180], [700, 260],
           "THE PATCH LOOP IS PAUSED BY DESIGN until you 📐 Extract Angle:\n"
           "orbit the viewport to a reveal, click ⬛ Render Passes then 📐 —\n"
           "the re-queue feeds the projected render (shaded) back as a PATCH at\n"
           "the EXACT measured pose (patch_exact → exact_view_override), painting\n"
           "only where the primary can't see (mask_unseen_only). AtlasOcclusionMask\n"
           "previews exactly that region. With a Qwen Multiple-Angles LoRA you'd\n"
           "generate the patch image instead — same wiring.")
    return b.dump()


# ══════════════════════════════════════════════════════════════════════════
# W6 · GHOST TOWN — AtlasInput + Output Desk + camera move + JPEG/mp4/USD
# ══════════════════════════════════════════════════════════════════════════
def build_ghosttown():
    b = Builder("atlas-showcase-ghosttown-move")
    w = b.w
    exr = EXR_DIR + "/ghosttown_32bit_acescg.exr"
    w.group("0 · ACEScg EXR → 🎬 ONE NODE — AtlasInput expands to the whole layered build", [-40, -40, 1260, 700], "#355")
    ocio = w.node("OCIORead", [0, 40], [380, 330], "OCIORead — ghost town EXR",
                  {"source": exr, "input_colorspace": "ACEScg"})
    reg = w.node("AtlasRegisterPlate", [420, 40], [360, 200], "Register float plate",
                 {"plate_path": exr, "colorspace": "ACEScg"})
    w.link(ocio, 0, reg, "image")
    ai = w.node("AtlasInput", [820, 40], [400, 480], "🎬 AtlasInput — VLM + 4 bands + LaMa",
                {"layers": 4, "sky": True, "use_vlm": True, "vlm_scope": False,
                 "vlm_provider": "lmstudio", "vlm_model": "google/gemma-4-12b-qat",
                 "inpaint": True, "mesh_resolution": 512})
    w.link(reg, 0, ai, "image")
    att = w.node("AtlasAttachSourcePlate", [820, 560], [300, 100], "Attach plate ref")
    w.link(ai, 0, att, "solve")
    w.link(reg, 1, att, "plate_ref")
    b.rset("solve", att, 0, [420, 300])
    b.rset("image", ai, 1, [420, 400])
    b.rset("depth", ai, 2, [420, 500])

    w.group("1 · OUTPUT DESK + VIEWPORT — author the move here", [1260, -40, 1240, 700], "#345")
    desk = w.node("AtlasViewportControls", [1300, 0], [340, 280], "🎛 Output Desk (detached controls)")
    g_s = b.rget("solve", [1300, 320])
    g_i = b.rget("image", [1300, 400])
    vpn = w.node("AtlasBlockoutViewport", [1680, 0], [780, 560], "Viewport — Orbit/Pan/Dolly + 🔭 lens",
                 {"resolution": 1280})
    w.link(g_s, 0, vpn, "solve")
    w.link(g_i, 0, vpn, "source_image")
    w.link(desk, 0, vpn, "controls")
    w.note([1300, 460], [340, 200],
           "THE MARKETING MOVE:\n1. Open 🎥 Camera Path (on the Output Desk)\n"
           "2. Click a move (⟳ Orbit R / ⭢ Dolly In)\n3. Set the 🔭 lens to taste\n"
           "4. ⏺ Bake Proxy Path → the graph re-queues and\n"
           "   the sequence/video/USD nodes below fill in.")

    w.group("2 · BAKED MOVE → JPEG SEQ + MP4 + USD CAMERA", [-40, 700, 1700, 560], "#435")
    save = w.node("SaveImageExtended", [0, 760], [420, 460], "JPEG sequence (marketing)",
                  {"filename_prefix": "ghosttown_move", "filename_keys": "",
                   "foldername_prefix": "showcase_ghosttown_move", "foldername_keys": "",
                   "output_ext": ".jpg", "quality": 92, "image_preview": False,
                   # save_metadata would embed the whole prompt (incl. the ~21MB
                   # baked client_data) into JPEG EXIF → "EXIF data is too long".
                   "save_metadata": False})
    w.link(vpn, 4, save, "images")
    vc = w.node("VideoCombinePlus", [460, 760], [380, 300], "MP4 (24 fps)",
                {"frame_rate": 24.0, "filename_prefix": "showcase_ghosttown_move"})
    w.link(vpn, 4, vc, "images")
    g_s2 = b.rget("solve", [460, 1100])
    usd = w.node("AtlasExportCameraPathUSD", [880, 760], [340, 130],
                 "camera_path.usda (BYPASSED until baked)",
                 {"output_dir": "atlas_exports/showcase_ghosttown"})
    usd["mode"] = 4
    w.link(g_s2, 0, usd, "solve")
    w.link(vpn, 5, usd, "camera_path")
    w.note([880, 940], [520, 220],
           "path_frames is a 1-frame placeholder until ⏺ Bake runs — the JPEG\n"
           "saver and MP4 write a single black frame on the first queue, then the\n"
           "real 100-frame 24fps move after baking. The USD export ships BYPASSED\n"
           "(it errors pre-bake by design): un-bypass after ⏺ Bake.")

    w.group("3 · 🔍 DEBUG REPORT — one JSON for tooling/AI assistants", [1700, 700, 900, 560], "#453")
    g_s3 = b.rget("solve", [1740, 760])
    g_d = b.rget("depth", [1740, 840])
    dbg = w.node("AtlasDebugReport", [2000, 760], [400, 300], "🔍 Full-stack diagnostic",
                 {"file_path": "atlas_debug/showcase_ghosttown.json"})
    w.link(g_s3, 0, dbg, "solve")
    w.link(g_d, 0, dbg, "depth")
    w.link(ai, 4, dbg, "vlm_report")
    return b.dump()


# ══════════════════════════════════════════════════════════════════════════
# W7 · JUNGLE RUINS — organic relief, quad retopo, Blender + OBJ + GLB
# ══════════════════════════════════════════════════════════════════════════
def build_jungleruins():
    b = Builder("atlas-showcase-jungleruins")
    w = b.w
    exr = EXR_DIR + "/jungleruins_32bit_acescg.exr"
    w.group("0 · ACEScg EXR IN + SOLVE", [-40, -40, 1240, 620], "#355")
    ocio = w.node("OCIORead", [0, 40], [380, 330], "OCIORead — jungle ruins EXR",
                  {"source": exr, "input_colorspace": "ACEScg"})
    reg = w.node("AtlasRegisterPlate", [420, 40], [360, 200], "Register float plate",
                 {"plate_path": exr, "colorspace": "ACEScg"})
    w.link(ocio, 0, reg, "image")
    solve = w.node("AtlasLearnedSolveFromImage", [820, 40], [380, 230], "Learned solve",
                   {"depth_model": V2_OUT})
    w.link(reg, 0, solve, "image")
    att = w.node("AtlasAttachSourcePlate", [820, 330], [300, 100], "Attach plate ref")
    w.link(solve, 0, att, "solve")
    w.link(reg, 1, att, "plate_ref")
    b.rset("plate_img", reg, 0, [420, 300])
    b.rset("solve", att, 0, [420, 400])

    w.group("1 · ORGANIC RELIEF — dense canopy wants a loose edge threshold", [1240, -40, 900, 620], "#345")
    g_i = b.rget("plate_img", [1280, 0])
    g_s = b.rget("solve", [1280, 80])
    dm = w.node("AtlasDepthMap", [1540, 0], [340, 150], "Shared depth", {"depth_model": V2_OUT})
    w.link(g_i, 0, dm, "image")
    w.link(g_s, 0, dm, "solve")
    g_s2 = b.rget("solve", [1280, 200])
    relief = w.node("AtlasDeriveReliefMesh", [1540, 200], [340, 200], "Relief (grid 512, edge 1.0)",
                    {"relief_grid": 512, "depth_edge_rel": 1.0})
    w.link(g_s2, 0, relief, "solve")
    w.link(dm, 0, relief, "depth")
    b.rset("scene", relief, 0, [1540, 460])

    w.group("2 · VIEWPORT + RETOPO EXPORT + BLENDER", [2140, -40, 1500, 900], "#453")
    g_sc = b.rget("scene", [2180, 0])
    g_i2 = b.rget("plate_img", [2180, 80])
    vpn = w.node("AtlasBlockoutViewport", [2440, 0], [820, 500], "Organic relief viewport",
                 {"resolution": 1280})
    w.link(g_sc, 0, vpn, "solve")
    w.link(g_i2, 0, vpn, "source_image")
    g_sc2 = b.rget("scene", [2180, 560])
    g_i3 = b.rget("plate_img", [2180, 640])
    exp = w.node("AtlasExportReliefMesh", [2440, 560], [400, 560],
                 "OBJ + GLB — hole-fill then QUAD retopo",
                 {"output_dir": "atlas_exports/showcase_jungleruins",
                  "fill_interior_holes": True, "retopo_method": "quad",
                  "retopo_target_vertex_count": 4000})
    w.link(g_sc2, 0, exp, "solve")
    w.link(g_i3, 0, exp, "image")
    g_sc3 = b.rget("scene", [2880, 0])
    bl = w.node("AtlasExportBlender", [3140, 80], [340, 120], "Blender build_scene.py",
                {"output_dir": "atlas_exports/showcase_jungleruins"})
    w.link(g_sc3, 0, bl, "solve")
    show = w.node("ShowText|pysssss", [2880, 560], [380, 200], "🔧/🔻 fill + retopo report")
    w.link(exp, 3, show, "text")
    w.note([3140, 240], [500, 260],
           "ORGANIC PIPELINE.\n\n"
           "Dense canopy depth is genuinely noisy at small scale — edge_rel 1.0\n"
           "keeps it continuous where the default 0.5 shreds it into holes.\n"
           "The export caps interior tears then quad-remeshes (~4000 verts,\n"
           "projective UVs regenerated) → OBJ+MTL+texture AND a self-contained\n"
           "GLB. Blender: run build_scene.py from Scripting; the OBJ imports\n"
           "textured into any DCC.")
    return b.dump()


# ══════════════════════════════════════════════════════════════════════════
# W8 · PORTAL — roll trim, shot cam, one-click moves, USD round-trip
# ══════════════════════════════════════════════════════════════════════════
def build_portal():
    b = Builder("atlas-showcase-portal-rolltrim")
    w = b.w
    w.group("0 · PLATE + SOLVE + 🎚 ROLL TRIM — level a drifting AI-plate solve by eye", [-40, -40, 820, 640], "#355")
    load = b.load_image("atlas_00024_portal.png", [0, 40], "Portal chamber (7680×4512)")
    solve = w.node("AtlasLearnedSolveFromImage", [360, 40], [400, 230], "Learned solve (MoGe)",
                   {"depth_model": MOGE})
    w.link(load, 0, solve, "image")
    trim = w.node("AtlasRollTrim", [360, 330], [400, 130], "🎚 Roll −3° (measured for this plate)",
                  {"roll_deg": -3.0})
    w.link(solve, 0, trim, "solve")
    b.rset("plate", load, 0, [0, 400])
    b.rset("solve", trim, 0, [360, 500])
    w.note([0, 480], [320, 160],
           "GeoCalib solved this plate's roll at −5.6°; the architecture's\n"
           "verticals imply ~−2.6° (measured against 196 detected edges).\n"
           "The 🎚 trim rotates the camera about its own view axis —\n"
           "position and view direction never move. Dial by eye.")

    w.group("1 · INTERIOR RELIEF + 2.39:1 SHOTCAM + MOVES", [820, -40, 1300, 640], "#345")
    g_i = b.rget("plate", [860, 0])
    g_s = b.rget("solve", [860, 80])
    dm = w.node("AtlasDepthMap", [1120, 0], [340, 150], "MoGe depth", {"depth_model": MOGE})
    w.link(g_i, 0, dm, "image")
    w.link(g_s, 0, dm, "solve")
    g_s2 = b.rget("solve", [860, 200])
    relief = w.node("AtlasDeriveReliefMesh", [1120, 200], [340, 200], "Relief (interior tuning)",
                    {"relief_grid": 384, "depth_edge_rel": 1.0, "max_edge_factor": 40.0,
                     "sky_heuristic": False})
    w.link(g_s2, 0, relief, "solve")
    w.link(dm, 0, relief, "depth")
    shot = w.node("AtlasDefineShotCam", [1120, 460], [320, 170], "2.39:1 anamorphic (36×15.1, 40mm)",
                  {"sensor_height_mm": 15.06, "focal_length_mm": 40.0, "resolution": 2048})
    g_i2 = b.rget("plate", [1500, 0])
    vpn = w.node("AtlasBlockoutViewport", [1500, 100], [600, 480],
                 "Viewport — try ⟲⟳⇠⇢⭢ + 🔭 lens", {"resolution": 1280})
    w.link(relief, 0, vpn, "solve")
    w.link(g_i2, 0, vpn, "source_image")
    w.link(shot, 0, vpn, "shot_cam")
    b.rset("scene", relief, 0, [1120, 660])

    w.group("2 · USD CAMERA ROUND-TRIP — export, reload, decompose: same lens", [-40, 760, 1700, 480], "#435")
    g_sc = b.rget("scene", [0, 820])
    usd = w.node("AtlasExportUSD", [260, 820], [340, 110], "camera.usda",
                 {"output_dir": "atlas_exports/showcase_portal"})
    w.link(g_sc, 0, usd, "solve")
    loader = w.node("AtlasUSDCameraLoader", [640, 820], [340, 160], "Reload the USD camera",
                    {"image_width": 7680, "image_height": 4512})
    w.link(usd, 0, loader, "usd_path")
    dec = w.node("AtlasDecomposeCamera", [1020, 820], [300, 210], "Decompose reloaded camera")
    w.link(loader, 0, dec, "camera")
    show = w.node("Display Any (rgthree)", [1360, 820], [200, 90], "focal mm (round-tripped)")
    w.link(dec, 7, show, "source")
    w.note([1360, 960], [300, 200],
           "The reloaded USD camera's focal must\n"
           "match the trimmed solve's — proving the\n"
           "camera survives the DCC handoff intact.\n"
           "(USD needs the [usd] extra: usd-core.)")
    return b.dump()


# ══════════════════════════════════════════════════════════════════════════
# W9 · X-RAY WRECK (EXPERIMENTAL) — LaRI hidden geometry behind the machine
# ══════════════════════════════════════════════════════════════════════════
def build_xray_wreck():
    b = Builder("atlas-showcase-xray-wreck")
    w = b.w
    w.group("0 · PLATE + SOLVE (EXPERIMENTAL — needs ATLAS_EXPERIMENTAL=1 + a LaRI clone)", [-40, -40, 820, 620], "#533")
    load = b.load_image("atlas_00025.png", [0, 40], "Coastal wreck (7680×4512)")
    solve = w.node("AtlasLearnedSolveFromImage", [360, 40], [400, 230], "Learned solve",
                   {"depth_model": V2_OUT})
    w.link(load, 0, solve, "image")
    b.rset("plate", load, 0, [0, 400])
    b.rset("solve", solve, 0, [360, 320])
    w.note([360, 420], [400, 160],
           "🩻 RESEARCH-ONLY. LaRI predicts the surface stack each camera ray\n"
           "pierces — strong on architecture, and this plate is the honest\n"
           "STRESS TEST: an isolated machine on OPEN MOOR, LaRI's documented\n"
           "weak domain. Expect partial coverage; the report says how much.")

    w.group("1 · SHARED DEPTH + 🩻 HIDDEN GEOMETRY", [820, -40, 940, 620], "#345")
    g_i = b.rget("plate", [860, 0])
    g_s = b.rget("solve", [860, 80])
    dm = w.node("AtlasDepthMap", [1120, 0], [340, 150], "Shared depth", {"depth_model": V2_OUT})
    w.link(g_i, 0, dm, "image")
    w.link(g_s, 0, dm, "solve")
    b.rset("depth", dm, 0, [1120, 190])
    g_i0 = b.rget("plate", [860, 180])
    sam = w.node("SAM3Segment", [1120, 290], [340, 330], "SAM3 — the occluder (restrict scope)",
                 {"prompt": "rusty derelict machine"})
    w.link(g_i0, 0, sam, "image")
    g_i2 = b.rget("plate", [860, 280])
    xr = w.node("AtlasPredictHiddenGeometry", [1500, 290], [400, 300],
                "🩻 LaRI — restricted to the machine")
    w.link(dm, 0, xr, "depth")
    w.link(g_i2, 0, xr, "image")
    w.link(sam, 1, xr, "restrict_mask")
    show = w.node("ShowText|pysssss", [1940, 290], [340, 200], "Registration + coverage report")
    w.link(xr, 2, show, "text")

    w.group("2 · BASE MESH + X-RAY LAYER (mask membership, NOT a depth band)", [1760, -40, 1400, 900], "#435")
    g_s2 = b.rget("solve", [1800, 0])
    g_d = b.rget("depth", [1800, 80])
    base = w.node("AtlasDeriveReliefMesh", [2060, 0], [340, 200], "Base relief (ORIGINAL depth)",
                  {"relief_grid": 384, "depth_edge_rel": 1.0, "max_edge_factor": 40.0})
    w.link(g_s2, 0, base, "solve")
    w.link(g_d, 0, base, "depth")
    grow = w.node("GrowMask", [2060, 260], [280, 130], "Grow hidden 32", {"expand": 32})
    w.link(xr, 1, grow, "mask")
    inv = w.node("InvertMask", [2380, 260], [240, 60], "Invert → exclude")
    w.link(grow, 0, inv, "mask")
    lam = w.node("INPAINT_LoadInpaintModel", [1800, 440], [300, 90], "LaMa", {"model_name": "big-lama.pt"})
    exp = w.node("INPAINT_ExpandMask", [2060, 440], [280, 130], None, {"grow": 48, "blur": 16})
    w.link(xr, 1, exp, "mask")
    g_i3 = b.rget("plate", [1800, 620])
    crop = w.node("AtlasInpaintCrop", [2380, 440], [300, 130], "✂ crop")
    w.link(g_i3, 0, crop, "image")
    w.link(exp, 0, crop, "mask")
    paint = w.node("INPAINT_InpaintWithModel", [2380, 620], [300, 150], "LaMa fill")
    w.link(lam, 0, paint, "inpaint_model")
    w.link(crop, 0, paint, "image")
    w.link(crop, 1, paint, "mask")
    g_i4 = b.rget("plate", [2720, 620])
    stitch = w.node("AtlasInpaintStitch", [2720, 700], [300, 150], "✂ stitch")
    w.link(g_i4, 0, stitch, "original_image")
    w.link(paint, 0, stitch, "inpainted_crop")
    w.link(crop, 2, stitch, "crop_region")
    cpl = w.node("AtlasCleanPlateLayer", [2720, 0], [400, 560], "🩻 X-ray layer (patched depth)",
                 {"name": "xray", "priority": 5.0, "relief_grid": 384, "depth_edge_rel": 1.5,
                  "band_side": "manual", "skirt_bevel": 1.5, "embed_matte": True})
    w.link(base, 0, cpl, "solve")
    w.link(xr, 0, cpl, "depth")
    w.link(stitch, 0, cpl, "plate_image")
    w.link(inv, 0, cpl, "exclude_mask")
    w.link(xr, 3, cpl, "layer_matte")
    b.rset("scene", cpl, 0, [3160, 600])

    w.group("3 · VIEWPORT + MOVE (🔬 RenderFix bypassed)", [-40, 900, 2400, 700], "#453")
    g_sc = b.rget("scene", [0, 960])
    g_i5 = b.rget("plate", [0, 1040])
    vpn = w.node("AtlasBlockoutViewport", [260, 960], [860, 560], "Dolly in — reveals show PREDICTED geometry",
                 {"resolution": 1024})
    w.link(g_sc, 0, vpn, "solve")
    w.link(g_i5, 0, vpn, "source_image")
    fix = w.node("AtlasRenderFix", [1180, 960], [380, 240], "🔬 Fixer repair (BYPASSED — needs Docker env)")
    fix["mode"] = 4
    w.link(vpn, 4, fix, "images")
    vc = w.node("VideoCombinePlus", [1620, 960], [380, 300], "MP4 of the baked move",
                {"frame_rate": 24.0, "filename_prefix": "showcase_xray_wreck_move"})
    w.link(fix, 0, vc, "images")
    w.note([1180, 1280], [700, 200],
           "AtlasRenderFix ships BYPASSED: it needs the fixer-spike-env Docker\n"
           "image + a user clone of nv-tlabs/Fixer (see docker/fixer/Dockerfile).\n"
           "Un-bypass to run each baked frame through the single-step repair\n"
           "diffusion before the MP4. Without it frames pass straight through.")
    return b.dump()


# ══════════════════════════════════════════════════════════════════════════
# W10 · DMP ANGLE, ANCHORED — NYC: ground-anchored facades vs floating relief
# ══════════════════════════════════════════════════════════════════════════
def build_dmp_angle_anchored():
    b = Builder("atlas-showcase-dmp-angle-anchored")
    w = b.w
    w.group("0 · THE CLASSIC DMP ANGLE — high vantage, street = ground plane", [-40, -40, 820, 620], "#355")
    load = b.load_image("newyork_Birdseye.png", [0, 40], "NYC birdseye (REAL photo)")
    solve = w.node("AtlasLearnedSolveFromImage", [360, 40], [400, 230], "Learned solve",
                   {"depth_model": V2_OUT})
    w.link(load, 0, solve, "image")
    scale = w.node("AtlasReferenceScaleSolve", [360, 330], [400, 280],
                   "📐 Counted storeys (5 × 3.5m tenement) → tier-1 scale",
                   {"reference_id": "building_story_3m",
                    "bbox_x0": 3820.0, "bbox_y0": 1775.0,
                    "bbox_x1": 4470.0, "bbox_y1": 2480.0,
                    "height_override_m": 17.5})
    w.link(solve, 0, scale, "solve")
    b.rset("plate", load, 0, [0, 400])
    b.rset("solve", scale, 0, [360, 660])
    w.note([0, 480], [320, 220],
           "From a high vantage, FOREGROUND buildings meet the street IN-FRAME —\n"
           "their geometry runs to Y=0. BACKGROUND buildings' bases are occluded\n"
           "behind nearer rooftops: the relief tears at the roofline and their\n"
           "meshes FLOAT. This workflow closes that gap with anchored facades.\n"
           "The ground fit fails up here (cars break it) → 1.6m fallback; the 📐\n"
           "reference node fixes it by COUNTED STOREYS (the 'PALM TOO' tenement:\n"
           "5 × 3.5m) → camera ≈ 60-64m, so anchored heights come out in real\n"
           "storeys (× 3.5m ≈ the floor count you can verify by eye).")

    w.group("1 · GROUND-ANCHORED WALLS + RELIEF MERGE", [820, -40, 1000, 620], "#345")
    g_i = b.rget("plate", [860, 0])
    g_s = b.rget("solve", [860, 80])
    dm = w.node("AtlasDepthMap", [1120, 0], [340, 150], "Shared depth", {"depth_model": V2_OUT})
    w.link(g_i, 0, dm, "image")
    w.link(g_s, 0, dm, "solve")
    g_s2 = b.rget("solve", [860, 200])
    walls = w.node("AtlasDeriveWalls", [1120, 200], [380, 240],
                   "⚓ Anchored facades (footprint = ray∩ground)",
                   {"max_walls": 24, "max_objects": 8, "distance_modes": 3,
                    "ground_anchor": True})
    w.link(g_s2, 0, walls, "solve")
    w.link(dm, 0, walls, "depth")
    g_s3 = b.rget("solve", [860, 320])
    relief = w.node("AtlasDeriveReliefMesh", [1120, 500], [340, 200], "Relief (visible surfaces)",
                    {"relief_grid": 384, "depth_edge_rel": 1.0})
    w.link(g_s3, 0, relief, "solve")
    w.link(dm, 0, relief, "depth")
    merge = w.node("AtlasMergeGeometry", [1540, 300], [300, 110], "Walls + relief")
    w.link(walls, 0, merge, "solve_a")
    w.link(relief, 0, merge, "solve_b")
    b.rset("scene", merge, 0, [1540, 460])

    w.group("2 · VIEWPORT — orbit: anchored facades SIT on the street", [1820, -40, 1300, 700], "#453")
    g_sc = b.rget("scene", [1860, 0])
    g_i2 = b.rget("plate", [1860, 80])
    vpn = w.node("AtlasBlockoutViewport", [2120, 0], [860, 560], "Compare with the aerial-preset build",
                 {"resolution": 1280})
    w.link(g_sc, 0, vpn, "solve")
    w.link(g_i2, 0, vpn, "source_image")
    w.note([2120, 580], [860, 200],
           "AtlasDeriveWalls ground_anchor=True: where a facade VISIBLY meets the\n"
           "street, its footprint comes from pure ray∩ground geometry (the depth\n"
           "model is demoted to grouping pixels) and the wall extrudes FROM Y=0 —\n"
           "so anchored buildings sit on the ground. The anchor deliberately\n"
           "REFUSES occluded bases (the contamination/occlusion gates), so the\n"
           "deepest rows still float — that residual gap is what the X-ray\n"
           "variant (atlas_dmp_angle_xray_newyork, experimental) fills with\n"
           "PREDICTED structure. distance_modes=3 fits one wall per depth mode\n"
           "per azimuth — the street-grid skyline case.")
    return b.dump()


# ══════════════════════════════════════════════════════════════════════════
# W11 · DMP ANGLE, X-RAY — NYC: LaRI predicts the occluded building bases
# ══════════════════════════════════════════════════════════════════════════
def build_dmp_angle_xray():
    b = Builder("atlas-showcase-dmp-angle-xray")
    w = b.w
    w.group("0 · PLATE + SOLVE (EXPERIMENTAL — ATLAS_EXPERIMENTAL=1 + a LaRI clone)", [-40, -40, 820, 620], "#533")
    load = b.load_image("newyork_Birdseye.png", [0, 40], "NYC birdseye (REAL photo)")
    solve = w.node("AtlasLearnedSolveFromImage", [360, 40], [400, 230], "Learned solve",
                   {"depth_model": V2_OUT})
    w.link(load, 0, solve, "image")
    scale = w.node("AtlasReferenceScaleSolve", [360, 330], [400, 280],
                   "📐 Counted storeys (5 × 3.5m tenement) → tier-1 scale",
                   {"reference_id": "building_story_3m",
                    "bbox_x0": 3820.0, "bbox_y0": 1775.0,
                    "bbox_x1": 4470.0, "bbox_y1": 2480.0,
                    "height_override_m": 17.5})
    w.link(solve, 0, scale, "solve")
    b.rset("plate", load, 0, [0, 400])
    b.rset("solve", scale, 0, [360, 660])
    w.note([0, 480], [320, 200],
           "🩻 Dense architecture is LaRI's STRONG domain — and 'predict the far\n"
           "building's occluded lower floors behind the near roofline' is the\n"
           "layered-ray-intersection use case. The restrict mask is the FOREGROUND\n"
           "DEPTH BAND (the node's own report recommends exactly this).\n"
           "📐 Metric scale comes from COUNTED STOREYS (the 'PALM TOO' tenement,\n"
           "5 × 3.5m → camera ≈ 60-64m) — the 1.6m fallback fires on this vantage.")

    w.group("1 · DEPTH + FG-BAND RESTRICT + 🩻 PREDICTION", [820, -40, 1140, 620], "#345")
    g_i = b.rget("plate", [860, 0])
    g_s = b.rget("solve", [860, 80])
    dm = w.node("AtlasDepthMap", [1120, 0], [340, 150], "Shared depth", {"depth_model": V2_OUT})
    w.link(g_i, 0, dm, "image")
    w.link(g_s, 0, dm, "solve")
    b.rset("depth", dm, 0, [1120, 190])
    g_s2 = b.rget("solve", [860, 200])
    dlm = w.node("AtlasDepthLayerMask", [1120, 300], [340, 240],
                 "FG band [0–45%] = the occluders",
                 {"near_pct": 0.0, "far_pct": 0.45})
    w.link(g_s2, 0, dlm, "solve")
    w.link(dm, 0, dlm, "depth")
    g_i2 = b.rget("plate", [860, 420])
    xr = w.node("AtlasPredictHiddenGeometry", [1500, 0], [400, 300],
                "🩻 LaRI — restricted to the fg band")
    w.link(dm, 0, xr, "depth")
    w.link(g_i2, 0, xr, "image")
    w.link(dlm, 0, xr, "restrict_mask")
    show = w.node("ShowText|pysssss", [1500, 360], [400, 200], "Registration + coverage report")
    w.link(xr, 2, show, "text")

    w.group("2 · BASE MESH + X-RAY LAYER (mask membership)", [1960, -40, 1400, 900], "#435")
    g_s3 = b.rget("solve", [2000, 0])
    g_d = b.rget("depth", [2000, 80])
    base = w.node("AtlasDeriveReliefMesh", [2260, 0], [340, 200], "Base relief (ORIGINAL depth)",
                  {"relief_grid": 384, "depth_edge_rel": 1.0, "max_edge_factor": 40.0})
    w.link(g_s3, 0, base, "solve")
    w.link(g_d, 0, base, "depth")
    grow = w.node("GrowMask", [2260, 260], [280, 130], "Grow hidden 32", {"expand": 32})
    w.link(xr, 1, grow, "mask")
    inv = w.node("InvertMask", [2580, 260], [240, 60], "Invert → exclude")
    w.link(grow, 0, inv, "mask")
    lam = w.node("INPAINT_LoadInpaintModel", [2000, 440], [300, 90], "LaMa", {"model_name": "big-lama.pt"})
    exp = w.node("INPAINT_ExpandMask", [2260, 440], [280, 130], None, {"grow": 48, "blur": 16})
    w.link(xr, 1, exp, "mask")
    g_i3 = b.rget("plate", [2000, 620])
    crop = w.node("AtlasInpaintCrop", [2580, 440], [300, 130], "✂ crop")
    w.link(g_i3, 0, crop, "image")
    w.link(exp, 0, crop, "mask")
    paint = w.node("INPAINT_InpaintWithModel", [2580, 620], [300, 150], "LaMa fill")
    w.link(lam, 0, paint, "inpaint_model")
    w.link(crop, 0, paint, "image")
    w.link(crop, 1, paint, "mask")
    g_i4 = b.rget("plate", [2920, 620])
    stitch = w.node("AtlasInpaintStitch", [2920, 700], [300, 150], "✂ stitch")
    w.link(g_i4, 0, stitch, "original_image")
    w.link(paint, 0, stitch, "inpainted_crop")
    w.link(crop, 2, stitch, "crop_region")
    cpl = w.node("AtlasCleanPlateLayer", [2920, 0], [400, 560], "🩻 X-ray layer (patched depth)",
                 {"name": "xray", "priority": 5.0, "relief_grid": 384, "depth_edge_rel": 1.5,
                  "band_side": "manual", "skirt_bevel": 1.5, "embed_matte": True})
    w.link(base, 0, cpl, "solve")
    w.link(xr, 0, cpl, "depth")
    w.link(stitch, 0, cpl, "plate_image")
    w.link(inv, 0, cpl, "exclude_mask")
    w.link(xr, 3, cpl, "layer_matte")
    b.rset("scene", cpl, 0, [3360, 600])

    w.group("3 · VIEWPORT + MOVE — dolly past the roofline", [-40, 900, 2200, 700], "#453")
    g_sc = b.rget("scene", [0, 960])
    g_i5 = b.rget("plate", [0, 1040])
    vpn = w.node("AtlasBlockoutViewport", [260, 960], [860, 560],
                 "Reveals show PREDICTED lower floors", {"resolution": 1280})
    w.link(g_sc, 0, vpn, "solve")
    w.link(g_i5, 0, vpn, "source_image")
    vc = w.node("VideoCombinePlus", [1180, 960], [380, 300], "MP4 of the baked move",
                {"frame_rate": 24.0, "filename_prefix": "showcase_nyc_xray_move"})
    w.link(vpn, 4, vc, "images")
    w.note([1180, 1300], [700, 180],
           "The anchored-walls variant (atlas_dmp_angle_anchored_newyork) closes\n"
           "the gap with GEOMETRY where a base is visible; this one INVENTS the\n"
           "occluded structure — LaRI hidden depth + LaMa pixels — so a dolly\n"
           "past the near roofline reveals plausible lower floors instead of a\n"
           "tear. Restricting to the fg band keeps visible surfaces untouched.")
    return b.dump()


BUILDERS = {
    "atlas_solve_lab_coastal_alley": build_solve_lab,
    "atlas_city_blocks_newyork": build_city_blocks,
    "atlas_composable_geometry_templecity": build_composable,
    "atlas_ocio_dmp_oceancastle": build_ocio_dmp,
    "atlas_interior_hangar": build_interior_hangar,
    "atlas_input_cameramove_ghosttown": build_ghosttown,
    "atlas_organic_relief_jungleruins": build_jungleruins,
    "atlas_rolltrim_shotcam_portal": build_portal,
    "atlas_xray_wreck": build_xray_wreck,
    "atlas_dmp_angle_anchored_newyork": build_dmp_angle_anchored,
    "atlas_dmp_angle_xray_newyork": build_dmp_angle_xray,
}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, fn in BUILDERS.items():
        if ONLY and name not in ONLY:
            continue
        wf = fn()
        path = OUTDIR / f"{name}_workflow.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wf, f, indent=1, ensure_ascii=False)
        # bidirectional link self-check (same invariant the validator pins)
        nodes = {n["id"]: n for n in wf["nodes"]}
        errs = []
        for l in wf["links"]:
            lid, oid, oslot, tid, tslot = l[:5]
            if lid not in (nodes[oid]["outputs"][oslot].get("links") or []):
                errs.append(f"link {lid} origin missing")
            if nodes[tid]["inputs"][tslot].get("link") != lid:
                errs.append(f"link {lid} target missing")
        assert not errs, f"{name}: {errs}"
        print(f"wrote {path.name}  ({len(wf['nodes'])} nodes, {len(wf['links'])} links, {len(wf['groups'])} groups)")


if __name__ == "__main__":
    main()
