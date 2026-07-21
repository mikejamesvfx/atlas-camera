"""Generate the 🔬 segmented-SDXL hidden-geometry showcase workflow (UI format).

The organized, grouped version of the portable session's run-verified
experiment (examples/experimental/..._hidden_segmented_sdxl.json): RAW-native
solve → quality-guarded visible relief → LaRI hidden geometry restricted to
the foreground band → 🏢 per-instance SAM3+SDXL inpaint of the occluded
buildings → merged scene through the 🩺 health gate → viewport projecting the
INPAINTED clean plate + USD export with manifest.

Widget values are the live-calibrated ones from that experiment (band 0–50%,
grow 32, visible relief ultra/mef24/normal60, hidden ultra/mef8/normal45,
SDXL denoise 0.50 seed 48192037 — seeds ship PINNED). Requires
ATLAS_EXPERIMENTAL=1 + a LaRI clone + SAM3 (comfyui-rmbg) + an SDXL
checkpoint. Usage:
    python tools/generate_segmented_sdxl_showcase_workflow.py <object_info.json> \
        <out.json> <path-to-generate_castle_dmp_workflow.py>
"""
import importlib.util
import json
import pathlib
import sys

OI_PATH = sys.argv[1]
OUT = pathlib.Path(sys.argv[2])
GEN = pathlib.Path(sys.argv[3])

_tmp = OUT.parent / "_scratch_ignore.json"
_argv = list(sys.argv)
sys.argv = ["gen", OI_PATH, str(_tmp)]
spec = importlib.util.spec_from_file_location("castlegen", GEN)
cg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cg)
sys.argv = _argv
_tmp.unlink(missing_ok=True)

NEF = "input/CameraRaw/DSC_2327.NEF"
EXPORT_ROOT = "atlas_exports/segmented_sdxl"


def out_index(node_type, name):
    for i, (out_name, _) in enumerate(cg.out_specs(node_type)):
        if out_name == name:
            return i
    raise SystemExit(f"{node_type} has no output named {name!r}")


LARI = "AtlasPredictHiddenGeometry"
LARI_DEPTH = out_index(LARI, "depth")
LARI_HIDDEN = out_index(LARI, "hidden_mask")
LARI_PAINT = out_index(LARI, "paint_matte")
LARI_REPORT = out_index(LARI, "report")

w = cg.WF()

# ── 1 · RAW IN ─────────────────────────────────────────────────────────────
w.group("1 · 📷 RAW IN", [-40, -40, 480, 660], "#355")
raw = w.node("AtlasLoadRAW", [0, 40], [420, 280], "📷 Load RAW",
             {"file_path": NEF, "half_size": True,
              "output_dir": f"{EXPORT_ROOT}/raw_plate"})
raw_rep = w.node("PreviewAny", [0, 380], [420, 180], "RAW report")
w.link(raw, 5, raw_rep, "source")

# ── 2 · SOLVE + SCALE ──────────────────────────────────────────────────────
w.group("2 · SOLVE + 📐🎚", [440, -40, 480, 780], "#345")
solve = w.node("AtlasLearnedSolveFromImage", [480, 40], [400, 280],
               "Learned solve (EXIF via raw_meta)")
w.link(raw, 0, solve, "image")
w.link(raw, 2, solve, "raw_meta")
scale = w.node("AtlasScaleOverride", [480, 380], [400, 140],
               "📐 Camera height 45m", {"camera_height_m": 45.0})
w.link(solve, 0, scale, "solve")
pitch = w.node("AtlasGravityOverride", [480, 560], [400, 140],
               "🎚 Gravity override — TRUE angles for this plate",
               {"pitch_deg": 32.0, "roll_deg": 0.9})
w.link(scale, 0, pitch, "solve")

# ── 3 · VISIBLE GEOMETRY ───────────────────────────────────────────────────
w.group("3 · 🛡 VISIBLE GEOMETRY", [920, -40, 500, 660], "#435")
dm = w.node("AtlasDepthMap", [960, 40], [420, 160], "Shared metric depth")
w.link(raw, 0, dm, "image")
w.link(pitch, 0, dm, "solve")
outlier = w.node("AtlasDepthOutlierMask", [960, 240], [420, 150],
                 "🛡 Outlier shield",
                 {"relative_threshold": 0.35, "mad_threshold": 6.0, "dilate_px": 2})
w.link(dm, 0, outlier, "depth")
relief = w.node("AtlasDeriveReliefMesh", [960, 430], [420, 200],
                "Visible relief — measured geometry (ultra for finals)",
                {"relief_quality": "high", "depth_edge_rel": 1.5,
                 "max_edge_factor": 24.0, "sky_heuristic": False,
                 "normal_edge_deg": 60.0})
w.link(pitch, 0, relief, "solve")
w.link(dm, 0, relief, "depth")
w.link(outlier, 0, relief, "outlier_mask")

# ── 4 · HIDDEN GEOMETRY (LaRI, band-restricted) ────────────────────────────
w.group("4 · 🩻 HIDDEN GEOMETRY — LaRI, fg-band restricted", [1420, -40, 980, 660], "#453")
band = w.node("AtlasDepthLayerMask", [1460, 40], [420, 240], "FG band restrict [0–50%]",
              {"near_pct": 0.0, "far_pct": 0.5, "feather_px": 4,
               "compute_hole_mask": True, "relief_grid": 384,
               "depth_edge_rel": 1.5})
w.link(pitch, 0, band, "solve")
w.link(dm, 0, band, "depth")
lari = w.node(LARI, [1460, 330], [420, 300], "🩻 LaRI — predict occluded surfaces",
              {"lari_path": "", "model": "lari-scene",
               "clear_rel": 0.02, "min_clear_m": 2.0, "smooth_px": 31,
               "fill_gaps": True, "seed": 845740604})
w.link(dm, 0, lari, "depth")
w.link(raw, 0, lari, "image")
w.link(band, 0, lari, "restrict_mask")
grow = w.raw("GrowMask", [1930, 40], [260, 110], "Grow hidden 32px", [32, True],
             [{"name": "mask", "type": "MASK", "link": None}],
             [{"name": "MASK", "type": "MASK", "links": [], "slot_index": 0}])
w.link(lari, LARI_HIDDEN, grow, "mask")
inv = w.raw("InvertMask", [1930, 190], [260, 80], "Invert → exclude", [],
            [{"name": "mask", "type": "MASK", "link": None}],
            [{"name": "MASK", "type": "MASK", "links": [], "slot_index": 0}])
w.link(grow, 0, inv, "mask")
hidden = w.node("AtlasDeriveReliefMesh", [1930, 310], [420, 200],
                "Hidden relief — synthesized, tighter budget",
                {"relief_quality": "high", "depth_edge_rel": 0.5,
                 "max_edge_factor": 8.0, "sky_heuristic": False,
                 "normal_edge_deg": 45.0})
w.link(relief, 0, hidden, "solve")
w.link(lari, LARI_DEPTH, hidden, "depth")
w.link(inv, 0, hidden, "exclude_mask")
w.link(outlier, 0, hidden, "outlier_mask")
lari_rep = w.node("PreviewAny", [2230, 40], [160, 220], "LaRI report")
w.link(lari, LARI_REPORT, lari_rep, "source")

# ── 5 · SEGMENTED SDXL ─────────────────────────────────────────────────────
w.group("5 · 🏢 SEGMENTED SDXL — per-building disocclusion inpaint", [2400, -40, 520, 900], "#533")
sdxl = w.node("AtlasSegmentedSDXLInpaint", [2440, 40], [440, 320],
              "🏢 SAM3 instances ∩ LaRI matte → SDXL per crop",
              {"prompt": "photorealistic continuation of the existing Manhattan "
                         "apartment buildings, matching brick, concrete, glass "
                         "windows and cloudy daylight",
               "checkpoint": "SDXL\\sd_xl_base_1.0.safetensors",
               "max_instances": 4, "steps": 30, "cfg": 4.0,
               "denoise": 0.5, "seed": 48192037})
w.link(raw, 0, sdxl, "image")
w.link(lari, LARI_PAINT, sdxl, "restrict_mask")
sdxl_prev = w.raw("PreviewImage", [2440, 420], [440, 300], "Segmented clean plate", [],
                  [{"name": "images", "type": "IMAGE", "link": None}], [])
w.link(sdxl, 0, sdxl_prev, "images")
sdxl_rep = w.node("PreviewAny", [2440, 760], [440, 130], "Inpaint report")
w.link(sdxl, 1, sdxl_rep, "source")
w.note([2930, 40], [340, 320],
       "Why segmented: one giant SDXL crop\n"
       "invents a single connected mega-\n"
       "structure across buildings. SAM3\n"
       "Separate-mode instances ∩ the LaRI\n"
       "paint matte are inpainted in their\n"
       "OWN crops and stitched — each\n"
       "building stays individually\n"
       "plausible. Seeds ship PINNED.\n\n"
       "Needs: ATLAS_EXPERIMENTAL=1, a LaRI\n"
       "clone (lari_path), comfyui-rmbg\n"
       "(SAM3), an SDXL checkpoint.")

# ── 6 · MERGE + HEALTH + REVIEW ────────────────────────────────────────────
w.group("6 · 🩺 MERGE + HEALTH + REVIEW", [2940, 380, 1180, 900], "#353")
merge = w.node("AtlasMergeGeometry", [2980, 440], [360, 110], "Merge visible + hidden")
w.link(relief, 0, merge, "solve_a")
w.link(hidden, 0, merge, "solve_b")
gate = w.node("AtlasSceneHealthGate", [2980, 590], [360, 240], "🩺 Health gate")
w.link(merge, 0, gate, "solve")
w.link(raw, 0, gate, "source_image")
w.link(dm, 0, gate, "depth")
attach = w.node("AtlasAttachSourcePlate", [2980, 880], [360, 100], "Attach EXR plate")
w.link(gate, 0, attach, "solve")
w.link(raw, 1, attach, "plate_ref")
dbg = w.node("AtlasDebugReport", [2980, 1020], [360, 200], "🔍 Debug report",
             {"file_path": f"{EXPORT_ROOT}/master_debug.json"})
w.link(attach, 0, dbg, "solve")
w.link(dm, 0, dbg, "depth")
vp = w.node("AtlasBlockoutViewport", [3380, 440], [700, 560],
            "Viewport — projects the INPAINTED plate")
w.link(attach, 0, vp, "solve")
w.link(sdxl, 0, vp, "source_image")
usd = w.node("AtlasExportUSD", [3380, 1040], [340, 110], "USD + manifest",
             {"output_dir": f"{EXPORT_ROOT}/usd"})
w.link(attach, 0, usd, "solve")

d = w.dump()
d["id"] = "atlas-segmented-sdxl-hidden-d810raw"
d.setdefault("extra", {})["workflow_name"] = "ATLAS 🔬 SEGMENTED SDXL — HIDDEN GEOMETRY DISOCCLUSION"
d["extra"]["description"] = ("LaRI hidden geometry restricted to the fg band, per-building "
                             "SAM3+SDXL inpaint, merged and health-gated, viewport projecting "
                             "the inpainted clean plate.")
OUT.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT)
