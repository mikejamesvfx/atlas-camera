"""Generate the TRUST canonical workflow — every 2026-07-18 feature in one graph.

RAW in (AtlasLoadRAW 📷 + EXIF raw_meta) → learned solve → 📐 scale dial →
🛡 outlier mask → quad-coherent relief → 🩺 scene-health gate → Output Desk
plate attach → 🔍 debug report + viewport + USD/Nuke exports (each writing
atlas_project.json + identity comments).

House rules: widgets_values derive from a LIVE /object_info (never hand-listed);
edit this generator, never the JSON. Usage:
    python tools/generate_canonical_trust_workflow.py <object_info.json> <out.json> \
        <path-to-generate_castle_dmp_workflow.py>
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

NEF = r"C:\Users\miike\ComfyUI_V91\ComfyUI\input\CameraRaw\DSC_2327.NEF"
EXPORT_ROOT = "atlas_exports/canonical_trust"

w = cg.WF()

# ── 1 · RAW IN ─────────────────────────────────────────────────────────────
w.group("1 · 📷 RAW IN — EXIF intrinsics + linear EXR sidecar", [-40, -40, 940, 660], "#355")
raw = w.node("AtlasLoadRAW", [0, 40], [430, 280], "📷 Load RAW (repoint at your NEF/CR2/CR3/RAF/ARW)",
             {"file_path": NEF, "half_size": True,
              "output_dir": f"{EXPORT_ROOT}/raw_plate"})
raw_rep = w.node("PreviewAny", [0, 380], [430, 200], "RAW report — camera · focal · sensor · lensfun profile")
w.link(raw, 5, raw_rep, "source")
w.note([470, 40], [410, 540],
       "TRUST CANONICAL — every 2026-07-18 feature.\n\n"
       "RAW decode replaces the ACR round-trip: ONE\n"
       "demosaic feeds the solve tensor AND a scene-\n"
       "linear EXR sidecar (Linear Rec.709 tag — OCIO\n"
       "converts downstream). EXIF focal + the camera-\n"
       "body registry ride raw_meta into the solve, so\n"
       "focal/sensor are MEASURED, not guessed; lensfun\n"
       "undistorts from MakerNote-derived lens specs\n"
       "(the report names the matched profile).\n\n"
       "half_size ON = fast iteration; turn OFF for\n"
       "the full-res final pass.\n\n"
       "This plate is deliberately imperfect: shot\n"
       "through glass, window haze at frame bottom.\n"
       "Watch what the trust tier does with it. →")

# ── 2 · SOLVE + SCALE TRUST ────────────────────────────────────────────────
w.group("2 · SOLVE + 📐 SCALE TRUST", [940, -40, 480, 660], "#345")
solve = w.node("AtlasLearnedSolveFromImage", [980, 40], [400, 280],
               "Learned solve — EXIF focal WINS via raw_meta")
w.link(raw, 0, solve, "image")
w.link(raw, 2, solve, "raw_meta")
scale = w.node("AtlasScaleOverride", [980, 380], [400, 140],
               "📐 Camera height = YOUR altitude (floors × ~3.2m)",
               {"camera_height_m": 45.0})
w.link(solve, 0, scale, "solve")
w.note([980, 560], [400, 60],
       "Single-image scale is ambiguous from altitude —\n"
       "the dial IS the doctrine. HUD shows ⚠ if assumed.")

# ── 3 · GEOMETRY QUALITY ───────────────────────────────────────────────────
w.group("3 · 🛡 GEOMETRY QUALITY — outliers out, quads coherent", [1420, -40, 500, 660], "#435")
dm = w.node("AtlasDepthMap", [1460, 40], [420, 170], "Shared metric depth (solved focal)")
w.link(raw, 0, dm, "image")
w.link(scale, 0, dm, "solve")
outlier = w.node("AtlasDepthOutlierMask", [1460, 250], [420, 150],
                 "🛡 Depth outliers → explicit holes")
w.link(dm, 0, outlier, "depth")
relief = w.node("AtlasDeriveReliefMesh", [1460, 440], [420, 200],
                "Relief — quad-coherent, outlier-masked")
w.link(scale, 0, relief, "solve")
w.link(dm, 0, relief, "depth")
w.link(outlier, 0, relief, "outlier_mask")

# ── 4 · SCENE HEALTH GATE ──────────────────────────────────────────────────
w.group("4 · 🩺 SCENE HEALTH — override a warning, never lose it", [1920, -40, 500, 660], "#453")
gate = w.node("AtlasSceneHealthGate", [1960, 40], [420, 260], "🩺 Health gate (ships closed on warn/fail)")
w.link(relief, 0, gate, "solve")
w.link(raw, 0, gate, "source_image")
w.link(dm, 0, gate, "depth")
w.note([1960, 360], [420, 280],
       "Runs the shared red-flag engine: scale trust,\n"
       "zero-vertex layers, band gaps, torn/stretched\n"
       "mesh QA, negative depth — and the gravity-flip\n"
       "guard. ON THIS PLATE it will WARN:\n"
       "'camera solved looking UP' — the bottom window\n"
       "haze reads as sky and flips GeoCalib's gravity\n"
       "(found live on this exact shot). Crop/re-render\n"
       "the haze for a correct solve, or ✅ Acknowledge\n"
       "to continue with the warning stamped into every\n"
       "export. A clean scene flows with zero clicks.")

# ── 5 · REVIEW ─────────────────────────────────────────────────────────────
w.group("5 · 🔍 REVIEW — debug JSON + viewport", [2420, -40, 1160, 900], "#353")
attach = w.node("AtlasAttachSourcePlate", [2460, 40], [400, 110], "Attach EXR plate (Output Desk)")
w.link(gate, 0, attach, "solve")
w.link(raw, 1, attach, "plate_ref")
dbg = w.node("AtlasDebugReport", [2460, 200], [400, 220],
             "🔍 Debug report — confidence vector + mesh QA",
             {"file_path": f"{EXPORT_ROOT}/master_debug.json"})
w.link(attach, 0, dbg, "solve")
w.link(dm, 0, dbg, "depth")
vp = w.node("AtlasBlockoutViewport", [2900, 40], [660, 560], "Viewport — ℹ HUD shows scale trust")
w.link(attach, 0, vp, "solve")
w.link(raw, 0, vp, "source_image")

# ── 6 · EXPORTS + MANIFEST ─────────────────────────────────────────────────
w.group("6 · 📦 EXPORTS — every artifact carries atlas_project.json", [2420, 900, 1160, 360], "#533")
usd = w.node("AtlasExportUSD", [2460, 960], [360, 110], "USD camera",
             {"output_dir": f"{EXPORT_ROOT}/usd"})
w.link(attach, 0, usd, "solve")
nuke = w.node("AtlasExportNuke", [2860, 960], [360, 140], "Nuke .py + .nk",
              {"output_dir": f"{EXPORT_ROOT}/nuke"})
w.link(attach, 0, nuke, "solve")
w.note([3260, 960], [300, 260],
       "Each export writes/merges\n"
       "atlas_project.json next to it\n"
       "(plate md5, solve fingerprint,\n"
       "models, seeds, scale + health\n"
       "verdicts, artifact list) and\n"
       "embeds its identity hash as a\n"
       "comment in .nk/.py — artifacts\n"
       "trace back to what made them.\n\n"
       "Acceptance: tools/\n"
       "orbit_stress_test.py scores\n"
       "±3°/±6° hole/stretch coverage.")

d = w.dump()
d["id"] = "atlas-canonical-trust-d810raw"
d.setdefault("extra", {})["workflow_name"] = "ATLAS TRUST — D810 RAW / QUALITY-GATED RELIEF"
d["extra"]["atlas_tier"] = "ATLAS TRUST"
d["extra"]["description"] = ("RAW-native solve with EXIF intrinsics, scale dial, outlier-masked "
                             "quad-coherent relief, the scene-health gate, and manifest-carrying exports.")
OUT.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT)
