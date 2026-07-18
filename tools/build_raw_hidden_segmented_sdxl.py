"""Build the SAM3-per-building SDXL inpaint workflow."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "examples" / "2026-07-18_atlas_raw_quickstart_workflow_hidden_diagnostic.json"
dst = ROOT / "examples" / "2026-07-18_atlas_raw_quickstart_workflow_hidden_segmented_sdxl.json"
wf = json.loads(src.read_text(encoding="utf-8"))
# v1 edge-safe defaults: the median-depth edge guard in core relief_mesh is
# complemented by a tighter hidden branch. The visible branch can stay more
# permissive because it is measured geometry; LaRI's synthesized branch cannot.
wf["8"]["inputs"].update({"relief_quality": "ultra", "max_edge_factor": 24.0,
                            "normal_edge_deg": 60.0})
wf["17"]["inputs"].update({"relief_quality": "ultra", "max_edge_factor": 8.0,
                             "normal_edge_deg": 45.0})
wf["47"] = {
    "inputs": {"depth": ["7", 0], "solve": ["5", 0],
                "relative_threshold": 0.35, "mad_threshold": 6.0, "dilate_px": 2},
    "class_type": "AtlasDepthOutlierMask",
    "_meta": {"title": "Depth outlier shield — explicit holes, no stretched shards"},
}
wf["8"]["inputs"]["outlier_mask"] = ["47", 0]
wf["17"]["inputs"]["outlier_mask"] = ["47", 0]
wf["8"]["inputs"]["quad_coherence"] = True
wf["17"]["inputs"]["quad_coherence"] = True
wf["44"] = {
    "inputs": {
        "image": ["1", 0],
        "restrict_mask": ["11", 3],
        "prompt": "photorealistic continuation of the existing Manhattan apartment buildings, matching brick, concrete, glass windows and cloudy daylight",
        "checkpoint": "SDXL\\sd_xl_base_1.0.safetensors",
        "max_instances": 4,
        "steps": 30,
        "cfg": 4.0,
        "denoise": 0.65,
        "seed": 48192037,
    },
    "class_type": "AtlasSegmentedSDXLInpaint",
    "_meta": {"title": "SAM3 buildings → per-instance SDXL inpaint"},
}
wf["19"]["inputs"]["source_image"] = ["44", 0]
wf["45"] = {"inputs": {"images": ["44", 0]}, "class_type": "PreviewImage", "_meta": {"title": "Segmented SDXL clean plate"}}
wf["46"] = {"inputs": {"source": ["44", 1]}, "class_type": "PreviewAny", "_meta": {"title": "Segmented inpaint report"}}
dst.write_text(json.dumps(wf, indent=2, ensure_ascii=False), encoding="utf-8")
print(dst)
