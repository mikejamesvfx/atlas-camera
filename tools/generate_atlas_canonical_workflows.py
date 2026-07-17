"""Historical bootstrap that produced the first canonical Atlas workflows.

PROVENANCE, NOT A LIVE SOURCE OF TRUTH. This script generated the initial
``examples/showcase/atlas_canonical_*`` workflows on 2026-07-17 by rebranding
three hero/v3 source graphs. Two things have changed since:

1. The committed canonical workflows have been HAND-CALIBRATED after generation
   (SAM3 prompt fix ``bf515e3``, quickstart/cleanplate split ``533d76e``,
   cleanplate grow 24->48 ``975dfb5``, ...). Re-running this script would
   clobber that tuning.
2. The source hero/v3 workflows it reads are NOT in the repository (they were
   local working files in the ComfyUI-install clone this script was written in).

It is committed for provenance — to record exactly how the canonical tier was
derived — and guarded accordingly: it exits if a source workflow is missing and
refuses to overwrite an existing output unless ``--force`` is passed.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "examples" / "showcase"

CONFIGS = (
    (
        "atlas_input_cameramove_ghosttown_v3_workflow.json",
        "atlas_canonical_quickstart_ghosttown_workflow.json",
        "ATLAS QUICKSTART — GHOST TOWN / CAMERA-AWARE RELIEF",
        "atlas_canonical_quickstart_ghosttown",
        "Start here: one plate, MoGe-2 metric depth, masked 1024 relief, and an optional camera move.",
    ),
    (
        "atlas_hero_templecity_workflow.json",
        "atlas_canonical_production_templecity_workflow.json",
        "ATLAS PRODUCTION — TEMPLE CITY / LAYERED SKYDOME",
        "atlas_canonical_production_templecity",
        "Production path: measured elevated camera, sky dome, edge masks, retopology, Maya and Nuke review.",
    ),
    (
        "atlas_hero_newyork_lari_workflow.json",
        "atlas_canonical_research_newyork_lari_workflow.json",
        "ATLAS RESEARCH — NYC / MASKED LARI COMPLETION",
        "atlas_canonical_research_newyork_lari",
        "Research path: counted building scale, restricted LaRI hidden geometry, inpaint crop/stitch, and DCC exports.",
    ),
)


def build(source: str, output: str, label: str, export_name: str, description: str,
          force: bool = False) -> None:
    source_path = SHOWCASE / source
    output_path = SHOWCASE / output
    if not source_path.exists():
        sys.exit(f"Source workflow not in the repo (historical bootstrap input): {source_path}\n"
                 "See this script's docstring - the canonical workflows are maintained "
                 "by hand now, not regenerated.")
    if output_path.exists() and not force:
        sys.exit(f"Refusing to overwrite hand-calibrated canonical workflow: {output_path}\n"
                 "Pass --force only if you really intend to regenerate it from the "
                 "hero/v3 source (this discards post-generation calibration).")
    wf = copy.deepcopy(json.loads(source_path.read_text(encoding="utf-8")))
    wf["id"] = f"{output.removesuffix('.json')}-canonical"
    wf.setdefault("extra", {})["workflow_name"] = label
    wf["extra"]["atlas_tier"] = label.split("—", 1)[0].strip()
    wf["extra"]["description"] = description
    for node in wf["nodes"]:
        if node.get("title"):
            node["title"] = node["title"].replace("V3", "CANONICAL").replace("HERO", "CANONICAL")
        if node["type"] == "AtlasBlockoutViewport":
            node["title"] = f"{label} — PROJECT / GRAY REVIEW"
        if node["type"] in {"AtlasExportMayaReviewScene", "AtlasExportNuke"}:
            node["widgets_values"][0] = f"atlas_exports/{export_name}/{ 'maya' if node['type'].endswith('MayaReviewScene') else 'nuke' }"
        if node["type"] == "AtlasExportReliefMesh":
            node["widgets_values"][0] = f"atlas_exports/{export_name}/relief"
    output_path.write_text(json.dumps(wf, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    force = "--force" in sys.argv[1:]
    for config in CONFIGS:
        build(*config, force=force)
