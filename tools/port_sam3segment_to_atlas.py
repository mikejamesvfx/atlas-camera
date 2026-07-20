"""Port shipped workflows from the third-party SAM3Segment to AtlasSAM3Mask.

WHY: `SAM3Segment` (comfyui-rmbg) hard-requires `triton`, which has no build for
Mac (MPS), CPU-only, or AMD. `AtlasSAM3Mask` loads the same SAM3 model straight
from `transformers`, so it runs everywhere. Every shipped workflow that segments
by text prompt is therefore arm64-blocked purely by the node it happens to use.

WHY A SCRIPT AND NOT HAND EDITS: the port rewrites 47 links across 15 files. A
link whose origin slot moves without its `links` array following is not a syntax
error — it loads with a silently missing connection. (Exactly that bug shipped in
the occlusion-cull quickstart.) A script makes the transform uniform and lets
`tests/test_example_workflows.py`'s bidirectional consistency check gate it.

MAPPING (verified against a real workflow and the live /object_info):

    SAM3Segment                          AtlasSAM3Mask
    inputs  image, prompt, +11 widgets   image, concepts, confidence_threshold, device
    outputs IMAGE(0), MASK(1), MASK_IMAGE(2)   mask(0), report(1)
    widgets [prompt, output_mode, confidence, max_segments, segment_pick,
             mask_blur, mask_offset, device, invert_output, unload_model,
             background, background_color]
          -> [concepts, confidence, device, output_mode, max_instances]

Only slot 1 (MASK) is ever wired in the shipped set, so links move 1 -> 0.

BOTH output modes port (2026-07-21). `Separate` used to be skipped because
AtlasSAM3Mask only ever returned a union; it now has its own
`output_mode="separate"` returning the (N,H,W) stack that
`post_process_instance_segmentation` was already producing, so instance
separation survives the port and `max_segments` carries over as
`max_instances`. Atlas orders instances LARGEST FIRST (SAM3's own score order
is unstable between runs), so a saved `AtlasInstanceMask` index may select a
different instance than it did under SAM3Segment — re-check the index after
porting a Separate graph.

    python tools/port_sam3segment_to_atlas.py --dry-run examples/**/*.json
    python tools/port_sam3segment_to_atlas.py examples/showcase/foo.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TARGET_TYPE = "AtlasSAM3Mask"
SOURCE_TYPE = "SAM3Segment"

# SAM3Segment widget order, as serialized by ComfyUI.
W_PROMPT, W_OUTPUT_MODE, W_CONFIDENCE, W_MAX_SEGMENTS, W_DEVICE = 0, 1, 2, 3, 7


def _device(value: Any) -> str:
    """SAM3Segment ships 'Auto'; AtlasSAM3Mask's combo is lowercase."""
    text = str(value or "auto").strip().lower()
    return text if text in ("auto", "cuda", "mps", "cpu") else "auto"


def port_node(node: dict, links: list) -> str | None:
    """Rewrite one SAM3Segment node in place. Returns a skip reason, or None."""
    widgets = node.get("widgets_values") or []
    mode = str(widgets[W_OUTPUT_MODE] if len(widgets) > W_OUTPUT_MODE else "Merged")
    if mode.lower() not in ("merged", "separate"):
        return f"unknown output_mode={mode!r}"
    separate = mode.lower() == "separate"

    outputs = node.get("outputs") or []
    for i, out in enumerate(outputs):
        if i != 1 and (out.get("links") or []):
            return f"slot {i} ({out.get('name')}) is wired; AtlasSAM3Mask has no such output"

    old_inputs = {i.get("name"): i for i in (node.get("inputs") or [])}
    mask_links = list((outputs[1].get("links") or []) if len(outputs) > 1 else [])

    node["type"] = TARGET_TYPE
    node.setdefault("properties", {})["Node name for S&R"] = TARGET_TYPE
    # `prompt` may be a LINKED input (the staged master feeds prompts through KJ
    # rails), so carry its link onto `concepts` rather than assuming a widget.
    node["inputs"] = [
        {"name": "image", "type": "IMAGE", "link": old_inputs.get("image", {}).get("link")},
        {"name": "concepts", "type": "STRING", "link": old_inputs.get("prompt", {}).get("link")},
        {"name": "confidence_threshold", "type": "FLOAT", "link": None},
        {"name": "device", "type": "COMBO", "link": None},
        {"name": "output_mode", "type": "COMBO", "link": None},
        {"name": "max_instances", "type": "INT", "link": None},
    ]
    node["outputs"] = [
        {"name": "mask", "type": "MASK", "links": mask_links, "slot_index": 0},
        {"name": "report", "type": "STRING", "links": [], "slot_index": 1},
    ]
    node["widgets_values"] = [
        widgets[W_PROMPT] if len(widgets) > W_PROMPT else "sky",
        widgets[W_CONFIDENCE] if len(widgets) > W_CONFIDENCE else 0.5,
        _device(widgets[W_DEVICE] if len(widgets) > W_DEVICE else "auto"),
        "separate" if separate else "merged",
        # max_instances only applies to the separated view; 0 = unlimited.
        (int(widgets[W_MAX_SEGMENTS]) if separate and len(widgets) > W_MAX_SEGMENTS
         else 0),
    ]
    # The node box was sized for 12 widgets; let ComfyUI recompute for 5.
    node.pop("size", None)

    for link in links:
        if link[1] == node["id"] and link[2] == 1:
            link[2] = 0
    return None


def check_links(graph: dict) -> list[str]:
    """Mirror of tests/test_example_workflows.py's bidirectional check, so a
    broken port is caught here rather than at review time."""
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    errs: list[str] = []
    for link in graph.get("links", []):
        if not (isinstance(link, list) and len(link) >= 6):
            errs.append(f"malformed link entry {link!r}")
            continue
        lid, on, os_, tn, ts = link[0], link[1], link[2], link[3], link[4]
        origin, target = nodes.get(on), nodes.get(tn)
        if origin is None:
            errs.append(f"link {lid}: origin node {on} missing")
        else:
            outs = origin.get("outputs") or []
            if os_ >= len(outs):
                errs.append(f"link {lid}: origin slot {os_} out of range")
            elif lid not in (outs[os_].get("links") or []):
                errs.append(f"link {lid}: origin {on}:{os_} does not list it")
        if target is None:
            errs.append(f"link {lid}: target node {tn} missing")
        else:
            ins = target.get("inputs") or []
            if ts >= len(ins):
                errs.append(f"link {lid}: target slot {ts} out of range")
            elif ins[ts].get("link") != lid:
                errs.append(f"link {lid}: target {tn}:{ts} has link={ins[ts].get('link')}")
    return errs


def detect_format(raw: str, graph: dict) -> tuple[int | None, bool, bool]:
    """Find the serialisation that reproduces `raw` BYTE-FOR-BYTE.

    Returns ``(indent, ensure_ascii, trailing_newline)``.

    Without this the writer reformats the WHOLE file and a two-node change
    lands as a ~2700-line diff — unreviewable, and impossible to eyeball for
    an unintended edit.

    The shipped set is NOT consistently formatted, which is the trap: some
    files are indent=1 with literal UTF-8, others indent=2 with
    escaped-unicode emoji in their node titles. Assuming ensure_ascii=False
    silently rewrites every title line in the escaped files. Found twice
    independently — here on the OCIO quickstart, and on a Mac while fixing the
    workflow ids — so both axes are probed rather than guessed.
    """
    trailing = raw.endswith("\n")
    body = raw[:-1] if trailing else raw
    for indent in (1, 2, 4, None):
        for ensure_ascii in (False, True):
            if json.dumps(graph, indent=indent, ensure_ascii=ensure_ascii) == body:
                return indent, ensure_ascii, trailing
    return 1, False, trailing  # unknown style; matches most of the shipped set


def port_file(path: Path, *, dry_run: bool) -> tuple[int, list[str], list[str]]:
    raw = path.read_text(encoding="utf-8")
    graph = json.loads(raw)
    indent, ensure_ascii, trailing = detect_format(raw, graph)
    before = check_links(graph)
    if before:
        return 0, [], [f"PRE-EXISTING link errors, refusing to touch: {before[:3]}"]

    ported, skipped = 0, []
    for node in graph.get("nodes", []):
        if node.get("type") != SOURCE_TYPE:
            continue
        reason = port_node(node, graph.setdefault("links", []))
        if reason:
            skipped.append(f"node {node['id']}: {reason}")
        else:
            ported += 1

    errs = check_links(graph)
    if errs:
        return 0, skipped, [f"PORT BROKE THE GRAPH, not written: {errs[:3]}"]
    if ported and not dry_run:
        out = json.dumps(graph, indent=indent, ensure_ascii=ensure_ascii)
        path.write_text(out + ("\n" if trailing else ""), encoding="utf-8")
    return ported, skipped, []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args(argv)

    total, failures = 0, 0
    for path in args.paths:
        if not path.is_file():
            continue
        try:
            ported, skipped, errs = port_file(path, dry_run=args.dry_run)
        except json.JSONDecodeError as exc:
            print(f"  SKIP  {path.name}: not JSON ({exc})")
            continue
        total += ported
        if errs:
            failures += 1
            for e in errs:
                print(f"  FAIL  {path.name}: {e}")
            continue
        if ported or skipped:
            verb = "would port" if args.dry_run else "ported"
            print(f"  {'OK  ' if ported else '--  '}{path.name}: {verb} {ported}")
            for s in skipped:
                print(f"          SKIPPED {s}")
    print(f"\n{'would port' if args.dry_run else 'ported'} {total} node(s); {failures} file(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
