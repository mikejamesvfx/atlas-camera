"""Build an API-format LaRI diagnostic workflow with correct mask/merge wiring."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "examples" / "2026-07-18_atlas_raw_quickstart_workflow_hidden.json"
dst = ROOT / "examples" / "2026-07-18_atlas_raw_quickstart_workflow_hidden_diagnostic.json"
wf = json.loads(src.read_text(encoding="utf-8"))
wf["1"]["inputs"]["file_path"] = "input/CameraRaw/DSC_2330.NEF"

# The source graph used an optional Mask Preview+ pack. PreviewAny is already
# present in the Atlas test environment and is sufficient for this diagnostic.
for key, source in (("14", ["16", 0]), ("15", ["16", 2])):
    wf[key]["class_type"] = "PreviewAny"
    wf[key]["inputs"] = {"source": source}

# Keep the original visible relief branch (node 8). Turn the LaRI branch into
# a hidden-only mesh: grow the predicted region, invert it for DeriveRelief's
# exclusion semantics, then explicitly merge it back with the visible solve.
wf["18"] = {
    "inputs": {"mask": ["11", 1], "expand": 32, "tapered_corners": True},
    "class_type": "GrowMask",
    "_meta": {"title": "Grow predicted hidden region (geometry support)"},
}
wf["20"] = {
    "inputs": {"mask": ["18", 0]},
    "class_type": "InvertMask",
    "_meta": {"title": "Keep only predicted hidden region"},
}
wf["17"]["inputs"]["exclude_mask"] = ["20", 0]
wf["21"] = {
    "inputs": {"solve_a": ["8", 0], "solve_b": ["17", 0]},
    "class_type": "AtlasMergeGeometry",
    "_meta": {"title": "Merge visible relief + hidden relief"},
}
wf["19"]["inputs"]["solve"] = ["21", 0]

# Expose the report that the original graph left unconnected.
report_preview = copy.deepcopy(wf["2"])
report_preview["inputs"] = {"source": ["11", 2]}
report_preview["_meta"] = {"title": "LaRI diagnostic report — registration / coverage / layer histogram"}
wf["22"] = report_preview

# Keep the paint matte as a separate visible diagnostic.
wf["23"] = {
    "inputs": {"source": ["11", 3]},
    "class_type": "PreviewAny",
    "_meta": {"title": "LaRI paint matte — only genuinely behind FG"},
}

# Emit a machine-readable post-merge geometry report so the A/B is measurable.
wf["24"] = {
    "inputs": {
        "solve": ["21", 0],
        "depth": ["7", 0],
        "file_path": "atlas_debug/raw_hidden_diagnostic.json",
    },
    "class_type": "AtlasDebugReport",
    "_meta": {"title": "Merged geometry report — vertex counts / red flags"},
}
wf["25"] = {
    "inputs": {"source": ["24", 0]},
    "class_type": "PreviewAny",
    "_meta": {"title": "Merged geometry report text"},
}

dst.write_text(json.dumps(wf, indent=2, ensure_ascii=False), encoding="utf-8")
print(dst)
