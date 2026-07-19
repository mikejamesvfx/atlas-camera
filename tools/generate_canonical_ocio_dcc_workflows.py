"""Generate the three canonical ACEScg clean-plate/DCC workflows.

The UI-format widget arrays are derived from a live ``/object_info`` snapshot;
never hand-edit the generated JSON because ComfyUI widgets are positional.

The repository root, the ``WF`` helper, and the output directory are all
derived from ``__file__`` so the committed workflows carry NO machine-specific
paths. By default the ACEScg plates and marketing cleanplates are serialized
as portable, repo-relative POSIX paths. Pass ``--asset-root`` to bake absolute
paths for a local run instead (see ``keeping locally runnable`` below).

Usage::

    # Regenerate the committed, portable workflows (repo-relative asset paths):
    python tools/generate_canonical_ocio_dcc_workflows.py object_info.json

    # Bake absolute paths for a local run against real plates on disk:
    python tools/generate_canonical_ocio_dcc_workflows.py object_info.json \
        --asset-root "D:/plates/acescg" --output-root /tmp/atlas_ocio_local
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("object_info", type=Path,
                    help="Path to a live /object_info snapshot (JSON).")
    ap.add_argument("--gen", type=Path,
                    default=REPO_ROOT / "tools" / "generate_castle_dmp_workflow.py",
                    help="WF-helper generator that owns the positional widget layout.")
    ap.add_argument("--output-root", type=Path,
                    default=REPO_ROOT / "examples" / "showcase",
                    help="Directory the canonical/marketing JSON is written to.")
    ap.add_argument("--asset-root", type=Path, default=None,
                    help="Directory holding the ACEScg EXR plates. Omit (default) to "
                         "serialize portable repo-relative asset paths for committing; "
                         "pass an absolute directory to bake runnable absolute paths.")
    return ap.parse_args()


_ARGS = _parse_args()
OI_PATH = _ARGS.object_info
OUTDIR = _ARGS.output_root
GEN = _ARGS.gen
ASSET_ROOT = _ARGS.asset_root

OUTDIR.mkdir(parents=True, exist_ok=True)
_scratch = OUTDIR / "_canonical_ocio_scratch.json"
_argv = list(sys.argv)
sys.argv = ["gen", str(OI_PATH), str(_scratch)]
spec = importlib.util.spec_from_file_location("canonical_ocio_gen_helpers", GEN)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
sys.argv = _argv
_scratch.unlink(missing_ok=True)

WF = helpers.WF

# Portable-by-default asset serialization. With no --asset-root the paths are
# repo-relative POSIX strings (committed, machine-agnostic); with an explicit
# --asset-root they become absolute paths a local ComfyUI can read directly.
_REL_EXR_DIR = Path("examples/images")
_REL_CLEANPLATE_DIR = Path("examples/showcase/marketing/cleanplates")


def _exr_source(exr: str) -> str:
    if ASSET_ROOT is None:
        return (_REL_EXR_DIR / exr).as_posix()
    return str(ASSET_ROOT / exr)


def _cleanplate_source(slug: str) -> str:
    name = f"{slug}_marketing_cleanplate_4k.png"
    if ASSET_ROOT is None:
        return (_REL_CLEANPLATE_DIR / name).as_posix()
    return str((OUTDIR / "marketing" / "cleanplates" / name).resolve())


def _generated_plate_ref(slug: str) -> tuple[str, str, str]:
    """OCIOWrite's ``output_folder``/``filename`` plus the EXACT file it writes.

    ComfyUI-OCIO names a still as ``<folder>/<filename>_<cs-tag>.<ext>`` — for
    ACEScg EXR that is ``<filename>_acescg.exr`` (``_cs_tag`` in the pack's
    ``io_nodes.py``; verified live by queueing a write and reading the result
    off disk). Because the name is fully determined by those two widgets, the
    companion ``AtlasRegisterPlate`` can point at the written file without
    guessing — and both nodes are fed from HERE so they cannot drift apart.

    The path stays relative for portability. OCIOWrite resolves it against
    ComfyUI's output directory; the DCC exporters copy it verbatim into the
    .nk/.ma, so repoint the Read there if the DCC's working directory differs.
    """
    folder = f"atlas_exports/{slug}_canonical/cleanplate"
    name = f"{slug}_cleanplate_generated"
    return folder, name, f"{folder}/{name}_acescg.exr"


@dataclass(frozen=True)
class Scene:
    slug: str
    label: str
    exr: str
    depth_model: str
    sky: bool
    max_edge_factor: float
    normal_edge_deg: float
    edge_extend_px: int
    retopo_target: int
    thumbnail_note: str
    occluder_prompt: str
    positive_prompt: str
    negative_prompt: str
    seed: int
    fill_backend: str


SCENES = (
    Scene(
        slug="oceancastle",
        label="OCEAN CASTLE",
        exr="oceancastle_32bit_acescg.exr",
        depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        sky=True,
        max_edge_factor=50.0,
        normal_edge_deg=50.0,
        edge_extend_px=32,
        retopo_target=3000,
        thumbnail_note="Outdoor sea/castle plate: sky card plus cleanplate-derived headland support geometry.",
        occluder_prompt="castle",
        positive_prompt="empty open ocean, rolling waves and distant horizon continuing through the entire masked region, photorealistic coastal water, matching sunlight and aerial perspective, no structures",
        negative_prompt="castle, building, tower, roof, island, cliff, rock structure, front elevation, orthographic, warped perspective, duplicate objects, text, seams, blurry",
        seed=41001,
        fill_backend="lama",
    ),
    Scene(
        slug="spacehangar",
        label="SPACE HANGAR",
        exr="spacehangar_32bit_acescg.exr",
        depth_model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        sky=False,
        max_edge_factor=80.0,
        normal_edge_deg=55.0,
        edge_extend_px=24,
        retopo_target=3500,
        thumbnail_note="Enclosed interior: no sky heuristic/card; a semantic ship matte preserves the original foreground layer.",
        occluder_prompt="spaceship",
        positive_prompt="completely empty futuristic hangar interior through the entire masked region, continuous reflective floor, receding wall panels, matching cyan lights and strong central perspective, no vehicle",
        negative_prompt="spaceship, spacecraft, aircraft, vehicle, machinery, front elevation, orthographic, warped architecture, duplicate objects, text, seams, blurry",
        seed=41002,
        fill_backend="sdxl",
    ),
    Scene(
        slug="ghosttown",
        label="GHOST TOWN",
        exr="ghosttown_32bit_acescg.exr",
        depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        sky=True,
        max_edge_factor=50.0,
        normal_edge_deg=50.0,
        edge_extend_px=16,
        retopo_target=3000,
        thumbnail_note="Outdoor street: sky card plus a semantic car/sign matte and localized cleanplate fill.",
        occluder_prompt="rusty car and fallen sign",
        positive_prompt="completely empty uninterrupted old west dirt road through the entire masked region, continuous road ruts, gravel and desert ground, matching low warm sunlight and source perspective, no objects",
        negative_prompt="car, automobile, vehicle, wagon, sign, timber beam, debris pile, object, front elevation, orthographic, warped architecture, duplicate objects, text, seams, blurry",
        seed=41003,
        fill_backend="lama",
    ),
)


def build(scene: Scene, *, marketing: bool = False) -> dict:
    w = WF()
    tier = "marketing" if marketing else "canonical"
    exr = _exr_source(scene.exr)
    export_root = f"atlas_exports/{tier}_ocio_{scene.slug}"

    w.group(f"0 · {scene.label} — ACEScg solve + semantic foreground", [-40, -40, 1660, 760], "#35546b")
    ocio = w.node(
        "OCIORead", [0, 40], [390, 360], "OCIO Read — source ACEScg → sRGB neural working space",
        {"source": exr, "frame_mode": "single", "input_colorspace": "ACEScg",
         "output_colorspace": "sRGB - Display", "raw_data": False},
    )
    register = w.node(
        "AtlasRegisterPlate", [430, 40], [360, 210], "Register original 32-bit ACEScg plate",
        {"plate_path": exr, "colorspace": "ACEScg", "is_proxy": False},
    )
    w.link(ocio, 0, register, "image")
    solve = w.node("AtlasLearnedSolveFromImage", [830, 40], [360, 230], "Learned camera solve",
                   {"depth_model": scene.depth_model})
    w.link(register, 0, solve, "image")
    depth = w.node("AtlasDepthMap", [830, 320], [360, 160], "Shared metric depth",
                   {"depth_model": scene.depth_model})
    w.link(register, 0, depth, "image")
    w.link(solve, 0, depth, "solve")
    occ = w.node("SAM3Segment", [1230, 40], [380, 340], f"Foreground mask — {scene.occluder_prompt}",
                 {"prompt": scene.occluder_prompt, "output_mode": "Merged"})
    w.link(register, 0, occ, "image")
    mask_image = w.node("MaskToImage", [1230, 390], [180, 60], "Mask QA")
    w.link(occ, 1, mask_image, "mask")
    mask_preview = w.node("PreviewImage", [1420, 390], [190, 190], "Foreground mask — inspect")
    w.link(mask_image, 0, mask_preview, "images")
    w.group("1 · LOCALIZED PERSPECTIVE-PRESERVING CLEANPLATE", [-40, 760, 1660, 780], "#3b5d4c")
    grow = w.node("INPAINT_ExpandMask", [0, 820], [280, 140], "Grow object cut", {"grow": 24, "blur": 8})
    w.link(occ, 1, grow, "mask")
    crop = w.node("AtlasInpaintCrop", [320, 820], [300, 140], "Crop object + context", {"context_pad_px": 192})
    w.link(register, 0, crop, "image")
    w.link(grow, 0, crop, "mask")
    if scene.fill_backend == "lama":
        loader = w.node("INPAINT_LoadInpaintModel", [660, 800], [300, 100], "LaMa — localized texture continuation",
                        {"model_name": "big-lama.pt"})
        fill = w.node("INPAINT_InpaintWithModel", [660, 940], [360, 180], "LaMa object removal",
                      {"seed": scene.seed})
        w.link(loader, 0, fill, "inpaint_model")
        w.link(crop, 0, fill, "image")
        w.link(crop, 1, fill, "mask")
    else:
        fill = w.node("AtlasSDXLInpaint", [660, 800], [430, 520], "SDXL cleanplate — preserve perspective",
                      {"positive_prompt": scene.positive_prompt,
                       "negative_prompt": scene.negative_prompt,
                       "seed": scene.seed, "steps": 35, "cfg": 5.0, "denoise": 0.90,
                       "grow_mask_by": 4, "max_side": 1024, "preserve_perspective": True})
        w.link(crop, 0, fill, "image")
        w.link(crop, 1, fill, "mask")
    stitch = w.node("AtlasInpaintStitch", [1130, 800], [330, 180], "Feather fill back into full plate",
                    {"feather_px": 24})
    w.link(register, 0, stitch, "original_image")
    w.link(fill, 0, stitch, "inpainted_crop")
    w.link(crop, 2, stitch, "crop_region")
    w.link(grow, 0, stitch, "mask")
    if not marketing:
        filled_preview = w.node("PreviewImage", [1130, 1030], [430, 350], "Inspect cleanplate before export")
        w.link(stitch, 0, filled_preview, "images")
    w.note(
        [660, 1330], [800, 250],
        "CLEANPLATE APPROVAL GATE\n"
        "Large removals are model- and plate-dependent. Inspect both mask and cleanplate previews. "
        "For final/hero work, replace the stitched image feeding Generated background layer with "
        "an artist-painted cleanplate when the neural fill changes perspective, repeats the object, or smears texture.\n\n"
        + (
            "PLATE REF: the approved cleanplate is a real file, so it is registered with its true "
            "colorspace (sRGB - Display) and wired into the background layer's plate_ref — Nuke/Maya "
            "Read nodes point straight at that 4K plate instead of a re-encoded preview tensor."
            if marketing else
            "PLATE REF: the generated cleanplate is written back to ACEScg (OCIOWrite, 16f EXR) and that "
            "file is registered into the background layer's plate_ref. Without it the layer exporters "
            "fall back to an 8-bit sRGB PNG carrying no colorspace — a mismatch in an ACEScg pipeline. "
            "The EXR lands under ComfyUI's output directory; repoint the DCC Read if it runs elsewhere."
        ),
    )

    # Build the scene far-to-near: optional sky, full-frame generated
    # background with depth re-estimated FROM THE CLEANPLATE, then the
    # untouched original foreground constrained by the actual object matte.
    # The cleanplate depth is essential: band-filling the original depth puts
    # the hidden road/headland at the object's far cutoff, which creates a
    # vertical cliff and makes the car/castle float on an off-axis move.
    scene_solve = solve
    if scene.sky:
        sky_seg = w.node("SAM3Segment", [0, 1010], [300, 300], "Sky mask", {"prompt": "sky", "output_mode": "Merged"})
        w.link(register, 0, sky_seg, "image")
        dome = w.node("AtlasSkyDomeLayer", [320, 1030], [320, 300], "Sky card",
                      {"radius_m": 900.0, "edge_extend_px": 96, "frame_outpaint_px": 128})
        w.link(solve, 0, dome, "solve")
        w.link(depth, 0, dome, "depth")
        w.link(sky_seg, 1, dome, "sky_mask")
        w.link(register, 0, dome, "plate_image")
        w.link(register, 1, dome, "plate_ref")
        scene_solve = dome
    bg = w.node("AtlasCleanPlateLayer", [0, 1390], [390, 500], "Generated background layer",
                {"name": "background_clean", "priority": 10.0, "band_side": "manual",
                 "near_pct": 0.0, "far_pct": 0.0,
                 "relief_grid": 384, "depth_edge_rel": 1.5, "fill_occluded": False,
                 "embed_matte": True, "edge_extend_px": scene.edge_extend_px,
                 "skirt_bevel": 1.5, "frame_outpaint_px": 64,
                 "max_edge_factor": scene.max_edge_factor, "normal_edge_deg": scene.normal_edge_deg})
    w.link(scene_solve, 0, bg, "solve")
    approved_plate = stitch
    clean_ref = None
    if marketing:
        approved_plate = w.node(
            "OCIORead", [2780, 40], [500, 330], "APPROVED 4K MARKETING CLEANPLATE",
            {"source": _cleanplate_source(scene.slug), "frame_mode": "single",
             "input_colorspace": "sRGB - Display", "output_colorspace": "sRGB - Display",
             "raw_data": False},
        )
        approved_preview = w.node(
            "PreviewImage", [2780, 410], [500, 520], "HERO OUTPUT — screenshot this preview"
        )
        w.link(approved_plate, 0, approved_preview, "images")
        # The approved cleanplate is a real file on disk, so the background
        # layer can carry a durable, non-proxy plate_ref: the DCC exporters
        # then point Nuke's Read / Maya's file node straight at that 4K plate
        # instead of re-encoding the preview tensor to PNG.
        clean_ref = w.node(
            "AtlasRegisterPlate", [2780, 960], [500, 210],
            "Register approved cleanplate — durable ref for the DCC exports",
            {"plate_path": _cleanplate_source(scene.slug),
             "colorspace": "sRGB - Display", "bit_depth": "auto",
             "role": "clean_plate"},
        )
        w.link(approved_plate, 0, clean_ref, "image")
    else:
        # The canonical background is generated in-graph, so no file exists to
        # register. Write one: sRGB - Display (ComfyUI's working space) back to
        # ACEScg as a 16f EXR, then register THAT file. Without this the layer
        # exporters fall back to an 8-bit sRGB PNG with no colour metadata, in a
        # pipeline where every other plate is ACEScg.
        ocio_folder, ocio_name, generated_exr = _generated_plate_ref(scene.slug)
        writer = w.node(
            "OCIOWrite", [2780, 40], [500, 430],
            "Generated cleanplate → ACEScg EXR (float DCC handoff)",
            {"from_colorspace": "sRGB - Display", "output_colorspace": "ACEScg",
             "container": "still image", "still_format": "exr",
             "video_codec": "prores_4444", "bit_depth": "16f", "auto_range": True,
             "first_frame": 1, "last_frame": 0, "start_number": 1,
             "source_start": 1, "raw_data": False,
             "output_folder": ocio_folder, "filename": ocio_name,
             "colorspace_in_name": True, "auto_colorspace": True,
             "compression": "zip"},
        )
        w.link(stitch, 0, writer, "images")
        clean_ref = w.node(
            "AtlasRegisterPlate", [2780, 500], [500, 210],
            "Register the written ACEScg cleanplate",
            {"plate_path": generated_exr, "colorspace": "ACEScg",
             "bit_depth": "auto", "role": "clean_plate"},
        )
        w.link(stitch, 0, clean_ref, "image")
    clean_depth = w.node(
        "AtlasDepthMap", [1200, 1390], [400, 190],
        "Cleanplate depth — continuous hidden support",
        {"depth_model": scene.depth_model},
    )
    w.link(approved_plate, 0, clean_depth, "image")
    w.link(solve, 0, clean_depth, "solve")
    w.link(clean_depth, 0, bg, "depth")
    w.link(approved_plate, 0, bg, "plate_image")
    if clean_ref is not None:
        # Marketing tier only. The canonical tier's background is generated
        # in-graph (LaMa/SDXL) and has no durable file on disk, so it is left
        # WITHOUT a plate_ref on purpose: the layer exporters then author a
        # real PNG from the tensor. Wiring a guessed path here would be worse
        # than nothing — exporters/_layers.py uses plate_path verbatim with no
        # existence check AND skips the PNG fallback once it is set.
        w.link(clean_ref, 1, bg, "plate_ref")
    if scene.sky:
        w.link(sky_seg, 1, bg, "exclude_mask")
    fg = w.node("AtlasCleanPlateLayer", [430, 1390], [390, 500], "Untouched foreground occluder",
                {"name": "foreground_original", "priority": 0.0, "band_side": "manual",
                 "near_pct": 0.0, "far_pct": 0.0,
                 "relief_grid": 384, "depth_edge_rel": 1.5, "embed_matte": True,
                 "edge_extend_px": 0, "max_edge_factor": scene.max_edge_factor,
                 "normal_edge_deg": scene.normal_edge_deg})
    w.link(bg, 0, fg, "solve")
    w.link(depth, 0, fg, "depth")
    w.link(register, 0, fg, "plate_image")
    w.link(register, 1, fg, "plate_ref")
    w.link(occ, 1, fg, "layer_matte")
    attach = w.node("AtlasAttachSourcePlate", [860, 1390], [300, 100], "Attach float source identity")
    w.link(fg, 0, attach, "solve")
    w.link(register, 1, attach, "plate_ref")

    # Both tiers now place plate-registration nodes at x=2780, so the group is
    # the same width either way (marketing: OCIORead + preview + register;
    # canonical: OCIOWrite + register).
    w.group("2 · OUTPUT DESK + MATCHED NUKE/MAYA RETOPOLOGY", [1660, -40, 1680, 1580], "#614b38")
    desk = w.node(
        "AtlasViewportControls", [1710, 40], [360, 280], "ACES 2.0 Output Desk",
        {"config_label": "ACES 2.0 / Studio", "working_colorspace": "ACEScg",
         "output_colorspace": "ACES - ACEScg", "display": "sRGB - Display",
         "view": "ACES 2.0 SDR-video"},
    )
    viewport = w.node(
        "AtlasBlockoutViewport", [1710, 370], [820, 500], "Layered projection — inspect seams before export",
        {"resolution": 1280},
    )
    w.link(attach, 0, viewport, "solve")
    w.link(register, 0, viewport, "source_image")
    w.link(desk, 0, viewport, "controls")
    w.link(desk, 1, viewport, "output_profile")
    nuke = w.node(
        "AtlasExportNukeLayers", [1710, 920], [450, 300], "Nuke layered projection — quadric retopo",
        {"output_dir": f"{export_root}/nuke", "retopo_method": "decimate",
         "retopo_target_vertex_count": scene.retopo_target,
         "retopo_smooth_iterations": 1, "retopo_crease_angle": 45.0,
         "retopo_pure_quad": False},
    )
    w.link(attach, 0, nuke, "solve")
    w.link(desk, 1, nuke, "output_profile")
    maya = w.node(
        "AtlasExportMayaLayers", [2200, 920], [450, 300], "Maya layered projection — identical quadric retopo",
        {"output_dir": f"{export_root}/maya", "retopo_method": "decimate",
         "retopo_target_vertex_count": scene.retopo_target,
         "retopo_smooth_iterations": 1, "retopo_crease_angle": 45.0,
         "retopo_pure_quad": False},
    )
    w.link(attach, 0, maya, "solve")
    w.link(desk, 1, maya, "output_profile")
    usd = w.node(
        "AtlasExportUSD", [1710, 1270], [400, 120], "Static camera USD",
        {"output_dir": f"{export_root}/camera"},
    )
    w.link(attach, 0, usd, "solve")
    debug = w.node(
        "AtlasDebugReport", [2150, 1270], [450, 260], "Layer/seam diagnostic JSON",
        {"file_path": f"atlas_debug/{tier}_ocio_{scene.slug}.json"},
    )
    w.link(attach, 0, debug, "solve")
    w.link(depth, 0, debug, "depth")

    wf = w.dump()
    wf["id"] = f"atlas-{tier}-ocio-{scene.slug}-dcc"
    wf.setdefault("extra", {}).update({
        "workflow_name": f"ATLAS {tier.upper()} OCIO DCC — {scene.label}",
        "atlas_tier": f"{tier.title()} OCIO DCC",
        "description": scene.thumbnail_note + f" Localized SAM3 + {scene.fill_backend.upper()} cleanplate.",
        "source_colorspace": "ACEScg",
        "neural_working_colorspace": "sRGB - Display",
        "retopo_method": "decimate",
        "retopo_target_vertex_count_per_layer": scene.retopo_target,
        "approved_marketing_cleanplate": marketing,
        "cleanplate_depth_geometry": True,
    })
    return wf


def validate_links(wf: dict) -> None:
    nodes = {n["id"]: n for n in wf["nodes"]}
    for link in wf["links"]:
        lid, oid, oslot, tid, tslot = link[:5]
        assert lid in (nodes[oid]["outputs"][oslot].get("links") or [])
        assert nodes[tid]["inputs"][tslot].get("link") == lid


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    marketing_out = OUTDIR / "marketing" / "workflows"
    marketing_out.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        wf = build(scene)
        validate_links(wf)
        path = OUTDIR / f"atlas_canonical_ocio_{scene.slug}_dcc_workflow.json"
        path.write_text(json.dumps(wf, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {path.name}: {len(wf['nodes'])} nodes, {len(wf['links'])} links")
        marketing_wf = build(scene, marketing=True)
        validate_links(marketing_wf)
        marketing_path = marketing_out / f"atlas_marketing_ocio_{scene.slug}_dcc_workflow.json"
        marketing_path.write_text(json.dumps(marketing_wf, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {marketing_path.name}: {len(marketing_wf['nodes'])} nodes, {len(marketing_wf['links'])} links")


if __name__ == "__main__":
    main()
