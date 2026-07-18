"""Add crop/stitch inpainting and colored region previews to the RAW X-ray test."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "examples" / "2026-07-18_atlas_raw_quickstart_workflow_hidden_diagnostic.json"
dst = ROOT / "examples" / "2026-07-18_atlas_raw_quickstart_workflow_hidden_inpaint_tagged.json"
wf = json.loads(src.read_text(encoding="utf-8"))

# Paint only the genuinely hidden region. The crop keeps LaMa's fixed internal
# resolution focused on the repair instead of squashing the entire RAW frame.
wf["26"] = {
    "inputs": {"mask": ["11", 3], "grow": 72, "blur": 24, "blur_type": "gaussian"},
    "class_type": "INPAINT_ExpandMask",
    "_meta": {"title": "HIDDEN PAINT MATTE — grow/feather"},
}
wf["27"] = {
    "inputs": {"image": ["1", 0], "mask": ["26", 0], "context_pad_px": 192},
    "class_type": "AtlasInpaintCrop",
    "_meta": {"title": "HIDDEN REGION — crop for inference"},
}
wf["28"] = {
    "inputs": {"model_name": "big-lama.pt"},
    "class_type": "INPAINT_LoadInpaintModel",
    "_meta": {"title": "LaMa clean-plate model"},
}
wf["29"] = {
    "inputs": {
        "inpaint_model": ["28", 0],
        "image": ["27", 0],
        "mask": ["27", 1],
        "seed": 0,
    },
    "class_type": "INPAINT_InpaintWithModel",
    "_meta": {"title": "Inpaint hidden region"},
}
wf["30"] = {
    "inputs": {
        "original_image": ["1", 0],
        "inpainted_crop": ["29", 0],
        "crop_region": ["27", 2],
        "mask": ["26", 0],
        "feather_px": 32,
    },
    "class_type": "AtlasInpaintStitch",
    "_meta": {"title": "Stitch clean plate back into RAW frame"},
}
wf["19"]["inputs"]["source_image"] = ["30", 0]

# Colored review tags: orange = foreground/occluder, purple = inferred hidden
# surface, green = paint matte. AtlasLayerPreview uses the same legend palette
# as the browser viewport and makes these masks inspectable as IMAGE outputs.
for key, mask_slot, color, title in (
    ("31", ["16", 0], "ff6a3d", "TAG — foreground occluder (orange)"),
    ("32", ["11", 1], "c95aff", "TAG — inferred hidden geometry (purple)"),
    ("33", ["11", 3], "6aff5a", "TAG — inpaint paint matte (green)"),
):
    wf[key] = {
        "inputs": {"image": ["30", 0], "mask": mask_slot, "layer_index": 0, "color_hex": color},
        "class_type": "AtlasLayerPreview",
        "_meta": {"title": title},
    }
    wf[str(int(key) + 10)] = {
        "inputs": {"images": [key, 0]},
        "class_type": "PreviewImage",
        "_meta": {"title": title},
    }

# Keep the stitched clean plate visible as a normal ComfyUI preview too.
wf["34"] = {
    "inputs": {"images": ["30", 0]},
    "class_type": "PreviewImage",
    "_meta": {"title": "STITCHED CLEAN PLATE — inspect before projection"},
}

dst.write_text(json.dumps(wf, indent=2, ensure_ascii=False), encoding="utf-8")
print(dst)
