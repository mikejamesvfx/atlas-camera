"""Wire a saved foreground/background mask into path-guided hole repair.

This preserves baked Camera Path proxy frames embedded in a UI workflow while
adding the new ``exclude_mask`` socket link.  The matching API export is
updated by input name, so both files remain equivalent and agent-runnable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _wire_ui(
    workflow: dict,
    *,
    mask_node_id: int,
    mask_output_slot: int,
    min_visible_pixels: int,
) -> None:
    nodes = {int(node["id"]): node for node in workflow["nodes"]}
    selector = next(
        node for node in workflow["nodes"]
        if node["type"] == "AtlasPathGuidedHoleRepair"
    )
    selector_id = int(selector["id"])
    selector["widgets_values"][6] = int(min_visible_pixels)

    existing = next(
        (item for item in selector["inputs"]
         if item.get("name") == "exclude_mask"),
        None,
    )
    if existing is not None and existing.get("link") is not None:
        link_id = int(existing["link"])
        for link in workflow["links"]:
            if int(link[0]) == link_id:
                link[1], link[2] = int(mask_node_id), int(mask_output_slot)
                return

    link_id = int(workflow.get("last_link_id", 0)) + 1
    input_slot = len(selector["inputs"])
    selector["inputs"].append({
        "name": "exclude_mask",
        "shape": 7,
        "type": "MASK",
        "link": link_id,
    })
    workflow["links"].append([
        link_id,
        int(mask_node_id),
        int(mask_output_slot),
        selector_id,
        input_slot,
        "MASK",
    ])
    source_output = nodes[int(mask_node_id)]["outputs"][int(mask_output_slot)]
    source_links = source_output.get("links")
    if source_links is None:
        source_output["links"] = [link_id]
    elif link_id not in source_links:
        source_links.append(link_id)
    workflow["last_link_id"] = link_id


def _wire_api(
    prompt: dict,
    *,
    mask_node_id: int,
    mask_output_slot: int,
    min_visible_pixels: int,
) -> None:
    selector_id = next(
        node_id for node_id, node in prompt.items()
        if node.get("class_type") == "AtlasPathGuidedHoleRepair"
    )
    inputs = prompt[selector_id]["inputs"]
    inputs["exclude_mask"] = [str(mask_node_id), int(mask_output_slot)]
    inputs["min_visible_pixels"] = int(min_visible_pixels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui", type=Path, required=True)
    parser.add_argument("--api", type=Path, required=True)
    parser.add_argument("--mask-node", type=int, required=True)
    parser.add_argument("--mask-output", type=int, default=1)
    parser.add_argument("--min-visible-pixels", type=int, default=32)
    args = parser.parse_args()

    ui = _load(args.ui)
    api = _load(args.api)
    _wire_ui(
        ui,
        mask_node_id=args.mask_node,
        mask_output_slot=args.mask_output,
        min_visible_pixels=args.min_visible_pixels,
    )
    _wire_api(
        api,
        mask_node_id=args.mask_node,
        mask_output_slot=args.mask_output,
        min_visible_pixels=args.min_visible_pixels,
    )
    _write(args.ui, ui)
    _write(args.api, api)
    print(args.ui)
    print(args.api)


if __name__ == "__main__":
    main()
