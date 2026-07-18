"""Build the RAW hidden-geometry workflow using native SDXL inpaint."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "examples" / "2026-07-18_atlas_raw_quickstart_workflow_hidden_inpaint_tagged.json"
dst = ROOT / "examples" / "2026-07-18_atlas_raw_quickstart_workflow_hidden_inpaint_sdxl.json"
wf = json.loads(src.read_text(encoding="utf-8"))

# Replace the LaMa loader/inference pair with Atlas' native SDXL adapter. The
# adapter expands to CheckpointLoaderSimple -> CLIPTextEncode ->
# VAEEncodeForInpaint -> KSampler -> VAEDecode at execution time.
wf["28"] = {
    "inputs": {
        "image": ["27", 0],
        "mask": ["27", 1],
        "checkpoint": "SDXL\\sd_xl_base_1.0.safetensors",
        "positive_prompt": "high detail, coherent architecture, realistic texture continuation",
        "negative_prompt": "blurry, warped geometry, duplicate structures, text, seams",
        "seed": 0,
        "steps": 30,
        "cfg": 5.5,
        "denoise": 0.85,
        "grow_mask_by": 8,
    },
    "class_type": "AtlasSDXLInpaint",
    "_meta": {"title": "SDXL inpaint — native ComfyUI graph"},
}
wf["29"] = {
    "inputs": {"images": ["28", 0]},
    "class_type": "PreviewImage",
    "_meta": {"title": "SDXL inpaint crop preview"},
}
wf["30"]["inputs"]["inpainted_crop"] = ["28", 0]
wf["35"] = {
    "inputs": {"source": ["28", 1]},
    "class_type": "PreviewAny",
    "_meta": {"title": "SDXL inpaint report"},
}

dst.write_text(json.dumps(wf, indent=2, ensure_ascii=False), encoding="utf-8")
print(dst)
