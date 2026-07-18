"""Build an API-format test graph using stock ComfyUI SDXL inpaint nodes."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "examples" / "2026-07-18_atlas_raw_quickstart_workflow_hidden_inpaint_sdxl.json"
dst = ROOT / "examples" / "2026-07-18_atlas_raw_quickstart_workflow_hidden_sdxl_native_api.json"
wf = json.loads(src.read_text(encoding="utf-8"))
wf.pop("28", None); wf.pop("29", None)
wf["36"] = {"inputs": {"ckpt_name": r"SDXL\sd_xl_base_1.0.safetensors"}, "class_type": "CheckpointLoaderSimple"}
wf["37"] = {"inputs": {"text": "high detail, coherent architecture, realistic texture continuation", "clip": ["36", 1]}, "class_type": "CLIPTextEncode"}
wf["38"] = {"inputs": {"text": "blurry, warped geometry, duplicate structures, text, seams", "clip": ["36", 1]}, "class_type": "CLIPTextEncode"}
wf["39"] = {"inputs": {"positive": ["37", 0], "negative": ["38", 0], "pixels": ["27", 0], "vae": ["36", 2], "mask": ["27", 1], "noise_mask": True}, "class_type": "InpaintModelConditioning"}
wf["40"] = {"inputs": {"model": ["36", 0], "seed": 1045806427285297, "steps": 30, "cfg": 5.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "positive": ["39", 0], "negative": ["39", 1], "latent_image": ["39", 2], "denoise": 0.85}, "class_type": "KSampler"}
wf["41"] = {"inputs": {"samples": ["40", 0], "vae": ["36", 2]}, "class_type": "VAEDecode"}
wf["30"]["inputs"]["inpainted_crop"] = ["41", 0]
wf["34"]["inputs"]["images"] = ["30", 0]
wf["35"]["inputs"]["source"] = ["2", 0]
dst.write_text(json.dumps(wf, indent=2), encoding="utf-8")
print(dst)
