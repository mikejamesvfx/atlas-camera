"""Strip authoring-machine absolute paths out of shipped ComfyUI workflows.

Every OCIO/RAW/hidden-geometry showcase had the author's own filesystem paths
baked into node widgets — ``C:\\Users\\miike\\…`` / ``/Users/…``. Those resolve
nowhere else, so a fresh clone or another OS loads a workflow that points at
files that do not exist (a Mac reviewer had to repoint one by hand). This
rewrites each absolute path to a portable relative one by CONTENT:

    *.exr/.dpx/.tif/.tiff  -> examples/images/<basename>   (the shipped plate
                                                             convention; users
                                                             drop the separately
                                                             distributed float
                                                             plates there)
    RAW (.nef/.cr2/.cr3/     -> input/CameraRaw/<basename>  (ComfyUI-launch cwd
         .raf/.arw/.dng)                                     relative)
    a lari clone directory   -> ""                          (AtlasPredictHidden-
                                                             Geometry: blank =
                                                             the ATLAS_LARI_PATH
                                                             env fallback)
    anything else            -> <basename>

Relative paths are left untouched. Format is preserved byte-for-byte
(detect_format), so this lands as a tiny diff. The invariant is enforced by
tests/test_shipping_workflow_paths.py.

    python tools/normalize_workflow_paths.py --check examples/**/*.json
    python tools/normalize_workflow_paths.py examples/showcase/foo.json
"""

from __future__ import annotations

import argparse
import json
import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from port_sam3segment_to_atlas import detect_format  # noqa: E402

_RAW_EXT = {".nef", ".cr2", ".cr3", ".raf", ".arw", ".dng"}
_PLATE_EXT = {".exr", ".dpx", ".tif", ".tiff"}


def is_absolute_machine_path(v: object) -> bool:
    """A Windows drive path (X:\\ or X:/) or a unix home path — not a URL, not
    a relative path."""
    if not isinstance(v, str) or len(v) < 3:
        return False
    if v[0].isalpha() and v[1] == ":" and v[2] in "\\/":
        return True
    return v.startswith("/Users/") or v.startswith("/home/")


def normalize(value: str, node_type: str) -> str:
    base = value.replace("\\", "/").rstrip("/").split("/")[-1]
    ext = posixpath.splitext(base)[1].lower()
    if node_type == "AtlasPredictHiddenGeometry" or not ext:
        return ""                       # a directory (lari clone) -> env fallback
    if ext in _PLATE_EXT:
        return f"examples/images/{base}"
    if ext in _RAW_EXT:
        return f"input/CameraRaw/{base}"
    return base


def fix_graph(graph: dict) -> int:
    fixed = 0
    for node in graph.get("nodes", []):
        wv = node.get("widgets_values")
        if not isinstance(wv, list):
            continue
        for i, v in enumerate(wv):
            if is_absolute_machine_path(v):
                wv[i] = normalize(v, node.get("type", ""))
                fixed += 1
    return fixed


def process(path: Path, *, check: bool) -> int:
    raw = path.read_text(encoding="utf-8")
    graph = json.loads(raw)
    if not isinstance(graph, dict) or "nodes" not in graph:
        return 0
    indent, ensure_ascii, trailing = detect_format(raw, graph)
    fixed = fix_graph(graph)
    if fixed and not check:
        out = json.dumps(graph, indent=indent, ensure_ascii=ensure_ascii)
        path.write_text(out + ("\n" if trailing else ""), encoding="utf-8")
    return fixed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="report without writing (exit 1 if any remain)")
    args = ap.parse_args(argv)

    total, files = 0, 0
    for path in args.paths:
        if not path.is_file():
            continue
        try:
            fixed = process(path, check=args.check)
        except json.JSONDecodeError as exc:
            print(f"  SKIP  {path.name}: not JSON ({exc})")
            continue
        if fixed:
            files += 1
            total += fixed
            verb = "would normalize" if args.check else "normalized"
            print(f"  {'--' if args.check else 'OK'}  {path.name}: {verb} {fixed} path(s)")
    verb = "would normalize" if args.check else "normalized"
    print(f"\n{verb} {total} path(s) across {files} file(s)")
    return 1 if (args.check and total) else 0


if __name__ == "__main__":
    sys.exit(main())
