"""Generate the MANUAL-DEBUG segmented-SDXL workflow — the 🏢 node unrolled.

`AtlasSegmentedSDXLInpaint` is a node-expansion wrapper; this workflow lays
its internal chain out as REAL nodes with a preview at every stage, so an
artist can debug per instance: SAM3 Separate stack → 🎭 instance select →
mask grow → ✂ crop → ✨ SDXL fill → ✂ stitch. Two instance rows are unrolled
(duplicate a row for more); the front of the graph is the same RAW → solve →
🎚 gravity override → LaRI chain as the packaged showcase, so the paint
matte matches exactly. Usage:
    python tools/generate_segmented_sdxl_manual_debug_workflow.py \
        <object_info.json> <out.json> <path-to-generate_castle_dmp_workflow.py>
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
PROMPT = ("photorealistic continuation of the existing Manhattan apartment "
          "buildings, matching brick, concrete, glass windows and cloudy daylight")


def out_index(node_type, name):
    for i, (out_name, _) in enumerate(cg.out_specs(node_type)):
        if out_name == name:
            return i
    raise SystemExit(f"{node_type} has no output named {name!r}")


LARI = "AtlasPredictHiddenGeometry"
LARI_PAINT = out_index(LARI, "paint_matte")

w = cg.WF()


def preview_img(src, slot, pos, title, size=(300, 260)):
    n = w.raw("PreviewImage", pos, list(size), title, [],
              [{"name": "images", "type": "IMAGE", "link": None}], [])
    w.link(src, slot, n, "images")
    return n


def mask_preview(src, slot, pos, title):
    m2i = w.raw("MaskToImage", pos, [220, 60], None, [],
                [{"name": "mask", "type": "MASK", "link": None}],
                [{"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0}])
    w.link(src, slot, m2i, "mask")
    return preview_img(m2i, 0, [pos[0] + 240, pos[1]], title)


# ── 1 · SOURCE CHAIN (matches the packaged showcase) ───────────────────────
w.group("1 · 📷 SOURCE — RAW → solve → 🎚 gravity → LaRI paint matte", [-40, -40, 1420, 700], "#355")
raw = w.node("AtlasLoadRAW", [0, 40], [400, 260], "📷 Load RAW",
             {"file_path": NEF, "half_size": True, "write_exr": False})
solve = w.node("AtlasLearnedSolveFromImage", [440, 40], [380, 260], "Solve (raw_meta)")
w.link(raw, 0, solve, "image")
w.link(raw, 2, solve, "raw_meta")
grav = w.node("AtlasGravityOverride", [440, 360], [380, 140],
              "🎚 TRUE gravity (32° down, +0.9° roll)",
              {"pitch_deg": 32.0, "roll_deg": 0.9})
w.link(solve, 0, grav, "solve")
scale = w.node("AtlasScaleOverride", [440, 540], [380, 120], "📐 45m",
               {"camera_height_m": 45.0})
w.link(grav, 0, scale, "solve")
dm = w.node("AtlasDepthMap", [860, 40], [380, 150], "Depth")
w.link(raw, 0, dm, "image")
w.link(scale, 0, dm, "solve")
band = w.node("AtlasDepthLayerMask", [860, 230], [380, 220], "FG band [0–50%]",
              {"near_pct": 0.0, "far_pct": 0.5, "feather_px": 4,
               "relief_grid": 384, "depth_edge_rel": 1.5})
w.link(scale, 0, band, "solve")
w.link(dm, 0, band, "depth")
lari = w.node(LARI, [860, 490], [380, 170], "🩻 LaRI (paint matte source)",
              {"lari_path": "", "model": "lari-scene",
               "clear_rel": 0.02, "min_clear_m": 2.0, "smooth_px": 31,
               "fill_gaps": True, "seed": 845740604})
w.link(dm, 0, lari, "depth")
w.link(raw, 0, lari, "image")
w.link(band, 0, lari, "restrict_mask")
paint_prev = mask_preview(lari, LARI_PAINT, [1280, 40], "Paint matte (∩ target)")

# ── 2 · SAM3 SEPARATE STACK ────────────────────────────────────────────────
w.group("2 · SAM3 Separate — the building instance stack", [-40, 700, 1420, 420], "#345")
sam = w.node("SAM3Segment", [0, 760], [380, 300], "SAM3 'building' (Separate)",
             {"prompt": "building", "output_mode": "Separate",
              "confidence_threshold": 0.5, "max_segments": 4})
w.link(raw, 0, sam, "image")
sam_prev = mask_preview(sam, 1, [420, 760], "Full instance stack")
w.note([1000, 760], [380, 300],
       "MANUAL DEBUG of AtlasSegmentedSDXLInpaint 🏢\n\n"
       "The packaged node runs exactly the rows below\n"
       "×N via node expansion. Here every stage is a\n"
       "real node with a preview: inspect the instance\n"
       "mask, the grown inpaint region, the crop, the\n"
       "SDXL fill, and the stitched plate per building.\n\n"
       "Row 2 consumes row 1's stitched plate — chain\n"
       "more rows for instances 2/3. Seeds pinned\n"
       "(48192037 + instance index).")

# ── 3/4 · TWO UNROLLED INSTANCE ROWS ───────────────────────────────────────
plate_src, plate_slot = raw, 0
for i in (0, 1):
    y = 1120 + i * 460
    w.group(f"{3 + i} · 🎭 instance {i} — select → grow → ✂ crop → ✨ SDXL → ✂ stitch",
            [-40, y - 20, 2560, 440], "#435" if i == 0 else "#453")
    inst = w.node("AtlasInstanceMask", [0, y + 40], [340, 160], f"🎭 Instance {i}",
                  {"instance_index": i, "min_coverage": 0.001})
    w.link(sam, 1, inst, "mask")
    w.link(lari, LARI_PAINT, inst, "restrict_mask")
    mask_preview(inst, 0, [380, y + 40], f"i{i} mask ∩ matte")
    grown = w.node("INPAINT_ExpandMask", [960, y + 40], [280, 130], "Grow 32 / blur 16",
                   {"grow": 32, "blur": 16, "blur_type": "gaussian"})
    w.link(inst, 0, grown, "mask")
    crop = w.node("AtlasInpaintCrop", [960, y + 220], [280, 140], "✂ Crop (pad 128)",
                  {"context_pad_px": 128})
    w.link(plate_src, plate_slot, crop, "image")
    w.link(grown, 0, crop, "mask")
    preview_img(crop, 0, [1280, y + 40], f"i{i} crop")
    fill = w.node("AtlasSDXLInpaint", [1620, y + 40], [340, 300], f"✨ SDXL fill {i}",
                  {"checkpoint": "SDXL\\sd_xl_base_1.0.safetensors",
                   "positive_prompt": PROMPT,
                   "negative_prompt": "fantasy, sci-fi, warped, duplicate, text, seams",
                   "seed": 48192037 + i, "steps": 30, "cfg": 4.0,
                   "denoise": 0.5, "grow_mask_by": 8, "max_side": 1024,
                   "preserve_perspective": True})
    w.link(crop, 0, fill, "image")
    w.link(crop, 1, fill, "mask")
    preview_img(fill, 0, [2000, y + 40], f"i{i} fill")
    stitch = w.node("AtlasInpaintStitch", [2340, y + 40], [180, 180], f"✂ Stitch {i}",
                    {"feather_px": 24})
    w.link(plate_src, plate_slot, stitch, "original_image")
    w.link(fill, 0, stitch, "inpainted_crop")
    w.link(crop, 2, stitch, "crop_region")
    w.link(grown, 0, stitch, "mask")
    plate_src, plate_slot = stitch, 0

# ── 5 · RESULT ─────────────────────────────────────────────────────────────
w.group("5 · RESULT — stitched clean plate", [-40, 2040, 800, 420], "#533")
preview_img(plate_src, plate_slot, [0, 2100], "Clean plate after 2 instances",
            size=(700, 340))

d = w.dump()
d["id"] = "atlas-segmented-sdxl-manual-debug"
d.setdefault("extra", {})["workflow_name"] = "ATLAS 🔬 SEGMENTED SDXL — MANUAL DEBUG (unrolled)"
d["extra"]["description"] = ("AtlasSegmentedSDXLInpaint's internal chain as real nodes with "
                             "per-stage previews — instance mask, grown region, crop, SDXL "
                             "fill, stitch — for two instances.")
OUT.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT)
