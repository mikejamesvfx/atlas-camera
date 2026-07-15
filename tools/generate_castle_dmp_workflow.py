"""Generate the castle DMP marketing workflow (UI format).

Widget lists are derived from the LIVE /object_info so positional
widgets_values can never drift from the node definitions — hand-listing them is
how saved workflows silently corrupt.
"""
import json, sys

OI = json.load(open(sys.argv[1], encoding="utf-8"))
OUT = sys.argv[2]

PRIMS = {"INT", "FLOAT", "STRING", "BOOLEAN"}


def _spec_items(name):
    n = OI[name]["input"]
    items = []
    for sec in ("required", "optional"):
        for k, v in (n.get(sec) or {}).items():
            items.append((k, v))
    return items


def is_widget(spec):
    """A combo serializes EITHER as a legacy list of options OR (V3 schema) as
    the literal string "COMBO" with the options in its config. Missing the
    second form silently drops the widget from widgets_values."""
    t = spec[0]
    cfg = spec[1] if len(spec) > 1 else {}
    if isinstance(t, list):
        return True
    if t == "COMBO" or t in PRIMS:
        return not cfg.get("forceInput")
    return False


def widget_default(spec):
    t = spec[0]
    cfg = spec[1] if len(spec) > 1 else {}
    if isinstance(t, list):
        return cfg.get("default", t[0] if t else "")
    if t == "COMBO":
        opts = cfg.get("options") or []
        return cfg.get("default", opts[0] if opts else "")
    return cfg.get("default", {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}[t])


def widget_names(name):
    """Names of the WIDGET inputs, in order (link-only types excluded)."""
    return [k for k, spec in _spec_items(name) if is_widget(spec)]


def widget_defaults(name):
    return {k: widget_default(spec) for k, spec in _spec_items(name) if is_widget(spec)}


def input_names(name):
    """Every input that can take a LINK (widgets are convertible too)."""
    return [k for k, _ in _spec_items(name)]


def out_specs(name):
    n = OI[name]
    return list(zip(n.get("output_name", []), n.get("output", [])))


class WF:
    def __init__(self):
        self.nodes, self.links, self.groups = [], [], []
        self.nid, self.lid, self.gid = 0, 0, 0

    def node(self, type_, pos, size=None, title=None, overrides=None, color=None):
        self.nid += 1
        ov = overrides or {}
        wnames = widget_names(type_)
        defs = widget_defaults(type_)
        wv = []
        for w in wnames:
            wv.append(ov.get(w, defs[w]))
            if w in ("seed", "noise_seed"):
                wv.append("fixed")          # never ship 'randomize' (known trap)
        n = {
            "id": self.nid, "type": type_, "pos": list(pos),
            "size": list(size or [300, 120]), "flags": {}, "order": self.nid,
            "mode": 0,
            "inputs": [{"name": k, "type": (OI[type_]["input"].get("required", {}).get(k)
                                            or OI[type_]["input"].get("optional", {}).get(k))[0],
                        "link": None}
                       for k in input_names(type_)],
            "outputs": [{"name": a, "type": b, "links": [], "slot_index": i}
                        for i, (a, b) in enumerate(out_specs(type_))],
            "properties": {"Node name for S&R": type_},
            "widgets_values": wv,
        }
        # inputs whose type is a python list (combo) must serialize as "COMBO"
        for i in n["inputs"]:
            if isinstance(i["type"], list):
                i["type"] = "COMBO"
        if title: n["title"] = title
        if color: n["color"], n["bgcolor"] = color
        self.nodes.append(n)
        return n

    def raw(self, type_, pos, size, title, wv, inputs, outputs, color=None, props=None):
        self.nid += 1
        n = {"id": self.nid, "type": type_, "pos": list(pos), "size": list(size),
             "flags": {}, "order": self.nid, "mode": 0,
             "inputs": inputs, "outputs": outputs,
             "properties": {"Node name for S&R": type_, **(props or {})},
             "widgets_values": wv}
        if title: n["title"] = title
        if color: n["color"], n["bgcolor"] = color
        self.nodes.append(n)
        return n

    def link(self, src, sslot, dst, dname):
        self.lid += 1
        st = src["outputs"][sslot]["type"]
        src["outputs"][sslot]["links"].append(self.lid)
        di = next(i for i in dst["inputs"] if i["name"] == dname)
        di["link"] = self.lid
        self.links.append([self.lid, src["id"], sslot, dst["id"],
                           dst["inputs"].index(di), st])

    def group(self, title, bounding, color="#3f789e"):
        self.gid += 1
        self.groups.append({"id": self.gid, "title": title,
                            "bounding": list(bounding), "color": color, "flags": {}})

    def note(self, pos, size, text):
        return self.raw("Note", pos, size, None, [text], [], [],
                        color=("#432", "#653"))

    def dump(self):
        return {"id": "atlas-castle-dmp-marketing", "revision": 0,
                "last_node_id": self.nid, "last_link_id": self.lid,
                "nodes": self.nodes, "links": self.links, "groups": self.groups,
                "config": {}, "extra": {}, "version": 0.4}


RAIL = ("#2a363b", "#3f5159")
w = WF()
sets, gets = {}, []


def rail_set(name, src, sslot, pos):
    t = src["outputs"][sslot]["type"]
    n = w.raw("SetNode", pos, [210, 60], f"Set_{name}", [name],
              [{"name": t, "type": t, "link": None}],
              [{"name": t, "type": t, "links": None}],
              color=RAIL, props={"aux_id": "kijai/ComfyUI-KJNodes", "previousName": name})
    w.link(src, sslot, n, t)
    sets[name] = t
    return n


def rail_get(name, pos):
    t = sets[name]
    n = w.raw("GetNode", pos, [210, 60], f"Get_{name}", [name], [],
              [{"name": t, "type": t, "links": [], "slot_index": 0}],
              color=RAIL, props={"aux_id": "kijai/ComfyUI-KJNodes"})
    gets.append(n)
    return n


PLATE = "atlas_seacliff_castle.png"

# ── 0 · PLATE + VLM PRE-FLIGHT ─────────────────────────────────────────────
w.group("0 · PLATE + 🧭 VLM PRE-FLIGHT — Queue 1 costs only the assessment; auto_continue lets it flow straight on to the ✅ solve gate", [-40, -40, 1000, 900], "#355")
load = w.raw("LoadImage", [0, 40], [320, 320], "Castle plate (4K)", [PLATE, "image"],
             [], [{"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0},
                  {"name": "MASK", "type": "MASK", "links": [], "slot_index": 1}])
assess = w.node("AtlasAssessImage", [360, 40], [420, 460], "🧭 Assess (VLM pre-flight)",
                {"provider": "lmstudio", "model": "google/gemma-4-12b-qat",
                 "offload_model": True, "auto_continue": True})
w.link(load, 0, assess, "image")
rail_set("plate", assess, 0, [360, 560])
rail_set("sam_sky", assess, 3, [360, 640])
rail_set("sam_fg", assess, 7, [360, 720])
w.note([0, 400], [320, 380],
       "CASTLE DMP — MARKETING BUILD\n\n"
       "Sea-cliff castle: sky · ocean · castle+rocks.\n\n"
       "Queue 1: the 🧭 VLM reads the plate and fills the SAM3 prompts\n"
       "below (sam_sky / sam_fg rails). auto_continue is ON, so the same\n"
       "queue runs the solve and stops at the ✅ gate.\n\n"
       "Read the 🧭 report, then ✅ Approve Solve and queue again.\n\n"
       "Needs lmstudio on :1234 with a VISION model. No VLM? Set\n"
       "auto_continue ON and type the prompts into the SAM3 nodes by hand\n"
       "— everything downstream still works.")

# ── 1 · SOLVE + GATE ───────────────────────────────────────────────────────
w.group("1 · SOLVE + ✅ GATE — a cheap solve first; approve it before paying for depth/SAM/inpaint", [1080, -40, 900, 900], "#353")
g_plate1 = rail_get("plate", [1120, 40])
solve = w.node("AtlasLearnedSolveFromImage", [1120, 140], [400, 240], "Learned solve (GeoCalib)",
               {"height_mode": "assume", "camera_height_m": 40.0})
w.link(g_plate1, 0, solve, "image")
g_plate2 = rail_get("plate", [1120, 420])
gate = w.node("AtlasSolveGate", [1120, 520], [400, 200], "✅ Solve gate")
w.link(solve, 0, gate, "solve")
w.link(g_plate2, 0, gate, "source_image")
rail_set("solve", gate, 0, [1560, 520])
w.note([1560, 40], [380, 440],
       "SCALE — READ THIS\n\n"
       "height_mode=assume, camera_height_m=40.\n\n"
       "Single-image scale is ambiguous. On an elevated vista the default\n"
       "1.6 m eye height is ~10x too small, and EVERY metric downstream\n"
       "follows it — 📏 band cutoffs, the hole-fill band box, DCC cameras.\n\n"
       "40 m is a sighting-in guess for a clifftop camera. Dial it until\n"
       "the ℹ Info HUD's scene depth looks right, or swap in\n"
       "AtlasScaleOverride / a reference-scale node.\n\n"
       "Nothing here is 'wrong' at 1.6 — it is just 25x small.")

# ── 2 · DEPTH RAILS (MoGe near / DA3 far) ──────────────────────────────────
w.group("2 · DEPTH RAILS — MoGe for the near field, DA3 for the far field. One AtlasDepthMap per band, NOT one global default", [2020, -40, 900, 900], "#335")
g_p3 = rail_get("plate", [2060, 40]);  g_s3 = rail_get("solve", [2060, 120])
d_fg = w.node("AtlasDepthMap", [2060, 220], [400, 130], "Depth · MoGe (near/castle)",
              {"depth_model": "Ruicheng/moge-2-vitl-normal"})
w.link(g_p3, 0, d_fg, "image"); w.link(g_s3, 0, d_fg, "solve")
rail_set("depth_fg", d_fg, 0, [2500, 220])
g_p4 = rail_get("plate", [2060, 400]); g_s4 = rail_get("solve", [2060, 480])
d_bg = w.node("AtlasDepthMap", [2060, 580], [400, 130], "Depth · DA3 (far/sky+water)",
              {"depth_model": "depth-anything/DA3METRIC-LARGE"})
w.link(g_p4, 0, d_bg, "image"); w.link(g_s4, 0, d_bg, "solve")
rail_set("depth_bg", d_bg, 0, [2500, 580])
w.note([2500, 300], [380, 260],
       "WHY TWO DEPTH MODELS\n\n"
       "MoGe is the near-field specialist but its far field runs away\n"
       "(>1000 m) and it CULLS SKY — disqualifying for a sky dome.\n"
       "DA3/V2-Outdoor behave in the far field.\n\n"
       "So: MoGe -> castle+rocks. DA3 -> sky dome + water.\n"
       "The depth model is a PER-BAND choice; a node default is just the\n"
       "starting value of a widget you override per band anyway.\n\n"
       "DA3 needs the [neural-da3] extra (--no-deps, see INSTALL.md).")

# ── 3 · SAM3 MATTES ────────────────────────────────────────────────────────
w.group("3 · 🎯 SAM3 MATTES — sky · castle+rocks · water. The FG matte drives BOTH the layer scope and the exported mesh cut", [2960, -40, 1180, 900], "#535")
g_p5 = rail_get("plate", [3000, 40]); g_sk = rail_get("sam_sky", [3000, 120])
sam_sky = w.node("SAM3Segment", [3000, 220], [340, 300], "SAM3 · sky", {"prompt": "sky"})
w.link(g_p5, 0, sam_sky, "image"); w.link(g_sk, 0, sam_sky, "prompt")
rail_set("sky_matte", sam_sky, 1, [3000, 560])

g_p6 = rail_get("plate", [3380, 40]); g_fgp = rail_get("sam_fg", [3380, 120])
sam_fg = w.node("SAM3Segment", [3380, 220], [340, 300], "SAM3 · castle + rocks",
                {"prompt": "castle and rocks"})
w.link(g_p6, 0, sam_fg, "image"); w.link(g_fgp, 0, sam_fg, "prompt")
rail_set("fg_matte", sam_fg, 1, [3380, 560])
inv_fg = w.node("InvertMask", [3380, 660], [280, 60], "NOT(castle) → exclude")
w.link(sam_fg, 1, inv_fg, "mask")
rail_set("not_fg", inv_fg, 0, [3380, 750])

g_p7 = rail_get("plate", [3760, 40])
sam_w = w.node("SAM3Segment", [3760, 220], [340, 300], "SAM3 · ocean/water",
               {"prompt": "ocean sea water"})
w.link(g_p7, 0, sam_w, "image")
rail_set("water_matte", sam_w, 1, [3760, 560])
inv_w = w.node("InvertMask", [3760, 660], [280, 60], "NOT(water) → exclude")
w.link(sam_w, 1, inv_w, "mask")
rail_set("not_water", inv_w, 0, [3760, 750])

# ── 4 · SKY DOME + INPAINT ─────────────────────────────────────────────────
w.group("4 · ☁ SKY DOME + LaMa INPAINT — the classic DMP sky separation. ✂ crop spends LaMa's fixed 256² on the hole, not the whole 4K frame", [4180, -40, 1500, 900], "#453")
lama = w.node("INPAINT_LoadInpaintModel", [4220, 40], [300, 60], "LaMa", {"model_name": "big-lama.pt"})
g_skm = rail_get("sky_matte", [4220, 140])
exp = w.node("INPAINT_ExpandMask", [4220, 240], [280, 110], "Expand sky hole",
             {"grow": 32, "blur": 8, "blur_type": "gaussian"})
w.link(g_skm, 0, exp, "mask")
g_p8 = rail_get("plate", [4220, 400])
crop = w.node("AtlasInpaintCrop", [4540, 240], [300, 100], "✂ Crop to hole", {"context_pad_px": 128})
w.link(g_p8, 0, crop, "image"); w.link(exp, 0, crop, "mask")
paint = w.node("INPAINT_InpaintWithModel", [4880, 240], [300, 130], "LaMa inpaint (seed PINNED)",
               {"seed": 0})
w.link(lama, 0, paint, "inpaint_model"); w.link(crop, 0, paint, "image"); w.link(crop, 1, paint, "mask")
g_p9 = rail_get("plate", [4880, 420])
stitch = w.node("AtlasInpaintStitch", [5220, 240], [300, 110], "✂ Stitch back")
w.link(g_p9, 0, stitch, "original_image"); w.link(paint, 0, stitch, "inpainted_crop")
w.link(crop, 2, stitch, "crop_region")
g_s5 = rail_get("solve", [4220, 500]); g_db = rail_get("depth_bg", [4220, 580])
g_skm2 = rail_get("sky_matte", [4220, 660])
sky = w.node("AtlasSkyDomeLayer", [4540, 480], [400, 300], "☁ Sky dome (DA3 depth)",
             {"radius_m": 800.0, "distance_m": 1200.0, "relief_grid": 96, "name": "sky",
              "priority": -10.0, "edge_extend_px": 48, "frame_outpaint_px": 96})
w.link(g_s5, 0, sky, "solve"); w.link(g_db, 0, sky, "depth")
w.link(g_skm2, 0, sky, "sky_mask"); w.link(stitch, 0, sky, "plate_image")
rail_set("solve_sky", sky, 0, [4980, 480])
w.note([5220, 420], [420, 360],
       "SKY DOME\n\n"
       "distance_m=1200 puts the card far back (parallax); radius_m=800 is\n"
       "its MINIMUM half-extent (size), grown by outpaint — distance and\n"
       "size are decoupled.\n\n"
       "edge_extend_px smears sky colour past the silhouette (deterministic\n"
       "Nuke-style edge-extend, NOT an inpaint). frame_outpaint_px=96 pads\n"
       "past the FRAME so small orbits never hit the plate boundary.\n\n"
       "LaMa's seed is PINNED to 0 — ComfyUI auto-adds 'randomize' to any\n"
       "widget named seed, which silently re-rolls the plate every queue.")

# ── 5 · BOUNDED BAND ───────────────────────────────────────────────────────
w.group("5 · 📏 BOUNDED BAND — measures the castle's OWN depth extent and emits ONE cutoff that drives BOTH layers", [5720, -40, 900, 900], "#544")
g_s6 = rail_get("solve", [5760, 40]); g_dfg = rail_get("depth_fg", [5760, 120])
g_fgm = rail_get("fg_matte", [5760, 200])
band = w.node("AtlasBoundedBand", [5760, 300], [400, 200], "📏 Bounded band (castle)",
              {"extrude_multiplier": 2.0, "near_pct": 5.0, "far_pct": 95.0})
w.link(g_s6, 0, band, "solve"); w.link(g_dfg, 0, band, "depth")
w.link(g_fgm, 0, band, "foreground_mask")
rail_set("band", band, 0, [6200, 300])
w.note([5760, 540], [560, 300],
       "ONE SPLIT, BOTH LAYERS\n\n"
       "W = P95-P5 of the castle's depth over its matte.\n"
       "cutoff = near + 2·W  ->  'the relief may extrude back at most twice\n"
       "its own width'.\n\n"
       "The SAME band_split feeds both layers:\n"
       "  castle  band_side=foreground -> [0, cutoff]   (relief clipped)\n"
       "  water   band_side=background -> [cutoff, inf] (pushed back)\n\n"
       "Because the split is an ABSOLUTE distance (not a percentile over\n"
       "each layer's own pixels), both resolve the identical boundary —\n"
       "no band drift, no band_ref_mask needed.")

# ── 6 · WATER GROUND PLANE ─────────────────────────────────────────────────
w.group("6 · 🌊 WATER — band_geometry=ground: the ocean is analytically the Y=0 plane, so don't let depth noise make it lumpy", [6660, -40, 900, 900], "#345")
g_ssky = rail_get("solve_sky", [6700, 40]); g_db2 = rail_get("depth_bg", [6700, 120])
g_p10 = rail_get("plate", [6700, 200]); g_nw = rail_get("not_water", [6700, 280])
g_b1 = rail_get("band", [6700, 360])
water = w.node("AtlasCleanPlateLayer", [6700, 460], [400, 420], "🌊 Water (ground plane)",
               {"name": "water", "priority": -5.0, "band_side": "background",
                "band_geometry": "ground", "relief_grid": 384, "depth_edge_rel": 1.5,
                "embed_matte": True, "edge_extend_px": 32, "frame_outpaint_px": 64})
w.link(g_ssky, 0, water, "solve"); w.link(g_db2, 0, water, "depth")
w.link(g_p10, 0, water, "plate_image"); w.link(g_nw, 0, water, "exclude_mask")
w.link(g_b1, 0, water, "band_split")
rail_set("solve_water", water, 0, [7140, 460])

# ── 7 · CASTLE LAYER ───────────────────────────────────────────────────────
w.group("7 · 🏰 CASTLE + ROCKS — relief, clipped at the band cutoff, scoped to the SAM3 matte", [7220, -40, 900, 900], "#534")
g_swat = rail_get("solve_water", [7260, 40]); g_dfg2 = rail_get("depth_fg", [7260, 120])
g_p11 = rail_get("plate", [7260, 200]); g_nfg = rail_get("not_fg", [7260, 280])
g_b2 = rail_get("band", [7260, 360])
castle = w.node("AtlasCleanPlateLayer", [7260, 460], [400, 420], "🏰 Castle + rocks (relief)",
                {"name": "castle", "priority": 0.0, "band_side": "foreground",
                 "band_geometry": "relief", "relief_grid": 512, "depth_edge_rel": 1.5,
                 "fill_occluded": True, "embed_matte": True, "edge_extend_px": 0,
                 "skirt_bevel": 1.5, "max_edge_factor": 12.0})
w.link(g_swat, 0, castle, "solve"); w.link(g_dfg2, 0, castle, "depth")
w.link(g_p11, 0, castle, "plate_image"); w.link(g_nfg, 0, castle, "exclude_mask")
w.link(g_b2, 0, castle, "band_split")
rail_set("solve_dmp", castle, 0, [7700, 460])
w.note([7260, 900], [820, 170],
       "SEAM DOCTRINE — the smear lives on the layers BEHIND.\n"
       "The FRONTMOST band keeps a CLEAN cut matte (edge_extend_px=0); every band behind it gets edge_extend.\n"
       "Priorities are FARTHEST-HIGHEST (sky -10 < water -5 < castle 0) so at a watertight seam the layer BEHIND wins\n"
       "the depth near-tie — nearest-highest makes each band's smear render IN FRONT of the layer behind it (striped seams).")

# ── 8 · MASTER VIEWPORT ────────────────────────────────────────────────────
w.group("8 · 🖼 MASTER DMP VIEWPORT — sky + water + castle. 📽 Project is the product; grey mesh is the diagnostic", [8180, -40, 1000, 940], "#446")
g_sdmp = rail_get("solve_dmp", [8220, 40]); g_p12 = rail_get("plate", [8220, 120])
vp = w.node("AtlasBlockoutViewport", [8220, 220], [960, 720], "🖼 Master viewport",
            {"resolution": 1024, "preview_expand": 1.0})
w.link(g_sdmp, 0, vp, "solve"); w.link(g_p12, 0, vp, "source_image")

# ── 9 · EXPORT + HOLE FILL + PREVIEW ───────────────────────────────────────
w.group("9 · 🔧 EXPORT MESH + HOLE FILL + PREVIEW — a SEPARATE branch off the gated solve: clean-plate layers live in projection_sources, but the exporter reads proxy_geometry", [9220, -40, 1800, 940], "#443")
g_s7 = rail_get("solve", [9260, 40]); g_dfg3 = rail_get("depth_fg", [9260, 120])
g_nfg2 = rail_get("not_fg", [9260, 200])
derive = w.node("AtlasDeriveReliefMesh", [9260, 300], [400, 260], "Castle relief (ocean/sky CUT)",
                {"relief_grid": 1024, "depth_edge_rel": 1.5, "max_edge_factor": 12.0,
                 "sky_heuristic": True})
w.link(g_s7, 0, derive, "solve"); w.link(g_dfg3, 0, derive, "depth")
w.link(g_nfg2, 0, derive, "exclude_mask")
g_p13 = rail_get("plate", [9260, 600])
exp_n = w.node("AtlasExportReliefMesh", [9700, 300], [400, 480], "🔧 Export + interior hole fill",
               {"output_dir": "atlas_exports/castle", "use_solve_mesh": True, "format": "both",
                "fill_interior_holes": True, "max_hole_edges": 128,
                "fill_depth_near_m": 0.0, "fill_depth_far_m": 0.0})
w.link(derive, 0, exp_n, "solve"); w.link(g_p13, 0, exp_n, "image")
g_p14 = rail_get("plate", [10140, 40])
vp2 = w.node("AtlasBlockoutViewport", [10140, 140], [840, 700], "🔍 FILL PREVIEW (what ships to Maya)",
             {"resolution": 768, "preview_expand": 1.0})
w.link(exp_n, 2, vp2, "solve")      # preview_solve
w.link(g_p14, 0, vp2, "source_image")
w.note([9260, 700], [400, 220],
       "WHY A SEPARATE BRANCH\n\n"
       "AtlasExportReliefMesh reads projection_scene.proxy_geometry —\n"
       "which AtlasDeriveReliefMesh writes. Clean-plate layers append to\n"
       "projection_sources instead, so a castle built as a LAYER is not\n"
       "exportable by this node at all.\n\n"
       "So: the DMP scene (group 8) and the exportable mesh (here) are two\n"
       "products off the same gated solve. exclude_mask = NOT(castle) is\n"
       "what cuts the ocean and sky out of the OBJ.")
w.note([10140, 860], [840, 200],
       "🔧 HOLE FILL — export-only, and invisible in the MASTER viewport by design.\n"
       "A tear IS a depth discontinuity; filling one in the live mesh would bridge exactly what the tearing prevents.\n"
       "So read the report ON the node, and wire preview_solve into THIS viewport to see what actually ships.\n\n"
       "max_hole_edges counts GRID edges — it scales with relief_grid. 128 @ grid 1024 ≈ 64 @ grid 512.\n"
       "Measured on this plate: 128 -> 97 holes/+1316 faces; 1024 -> 101 holes/+2669 faces. The 4 extra are 128-617 edge\n"
       "loops = big flat caps over real silhouettes. 128 buys 97 of 101 holes for HALF the geometry.\n\n"
       "Band box (fill_depth_near_m/far_m) is OFF: both must be > 0, it needs a TRUSTWORTHY metric scale, and window mode\n"
       "bypasses the largest-loop guard (the one way to accidentally cap the outer frame). Set the camera height first.")

json.dump(w.dump(), open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"wrote {OUT}")
print(f"  nodes={len(w.nodes)} links={len(w.links)} groups={len(w.groups)} "
      f"rails={len(sets)} gets={len(gets)}")
