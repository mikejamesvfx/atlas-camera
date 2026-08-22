"""Hand a plate to Photoshop and take it back with the colour intact.

The MANUAL lane: Atlas converts and hands over, a human paints, Atlas takes it
back correctly tagged and lifts any hand-painted mattes out as masks. Nothing
here needs scripted generative fill.

Send a plate (converts to ACES2065-1, which is what Photoshop assumes an OCIO
EXR contains, and optionally opens it as a 32-bit OCIO document)::

    python tools/photoshop_handoff.py send --plate <plate>.exr \
        --out <handoff>.exr [--open]

Paint in Photoshop. To hand a matte back, store a selection as an alpha channel
(Select > Save Selection, or the Channels panel), then
**File > Save As > TIFF**, uncompressed, with "Alpha Channels" ticked.

Take it back — the plate re-tagged, plus one PNG per painted matte::

    python tools/photoshop_handoff.py receive --tiff <saved>.tif \
        --out <returned>.exr --matte-dir <mattes/> \
        [--matte-names boiler,sky] [--target-space "Linear Rec.709 (sRGB)"]

Why the colourspace flags matter: Photoshop's TIFF carries NO colourspace tag,
so reading it on 'auto' makes OIIO guess `sRGB - Display` from the extension
and silently mis-convert scene-linear data. `receive` states what it is
(the OCIO working space) and re-tags the output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    from atlas_camera.paint import ocio
    from atlas_camera.paint.photoshop.handoff import (
        PHOTOSHOP_ASSUMED_INPUT, PHOTOSHOP_DEFAULT_WORKING_SPACE, receive, send)

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("send", help="convert a plate and hand it to Photoshop")
    s.add_argument("--plate", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--assumed-space", default=PHOTOSHOP_ASSUMED_INPUT,
                   help="What Photoshop will take the file to be. It does NOT "
                        "read the colourspace tag; it assumes ACES2065-1.")
    s.add_argument("--open", action="store_true",
                   help="Also open it in Photoshop as a 32-bit OCIO document")
    s.add_argument("--bit-depth", default="float", choices=("half", "float"))

    r = sub.add_parser("receive", help="take a painted TIFF back from Photoshop")
    r.add_argument("--tiff", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--working-space", default=PHOTOSHOP_DEFAULT_WORKING_SPACE,
                   help="Photoshop's OCIO working space — what its export is "
                        "actually in. The file itself is untagged.")
    r.add_argument("--target-space", default=None,
                   help="Convert to this on the way in (default: leave in the "
                        "working space and tag it honestly)")
    r.add_argument("--matte-dir", type=Path, default=None,
                   help="Write one PNG per painted matte channel here")
    r.add_argument("--matte-names", default="",
                   help="Comma-separated labels, in document order — TIFF does "
                        "not preserve Photoshop's channel names")
    r.add_argument("--keep-empty-mattes", action="store_true")
    r.add_argument("--bit-depth", default="float", choices=("half", "float"))

    for parser in (s, r):
        parser.add_argument("--ocio-config", type=Path, default=None)
        parser.add_argument("--report", type=Path, default=None)

    args = p.parse_args(argv)

    with ocio.scoped_config(args.ocio_config) as identity:
        print(ocio.describe(identity), file=sys.stderr)

        if args.mode == "send":
            result = send(plate_path=args.plate, out_path=args.out,
                          assumed_space=args.assumed_space,
                          bit_depth=args.bit_depth)
            print(f"source     {result['source']}")
            print(f"  was      {result['source_colorspace']!r}")
            print(f"  sent as  {result['sent_as']!r}   "
                  f"(Photoshop assumes this regardless of the tag)")
            print(f"wrote      {result['out']}")
            if args.open:
                from atlas_camera.paint.photoshop.com_client import PhotoshopClient

                client = PhotoshopClient()
                client.connect()
                opened = client.open_plate(args.out, as_ocio=True)
                result["opened"] = opened
                print(f"opened     {opened.get('bitsPerChannel')} "
                      f"{opened.get('mode')}")
                if "THIRTYTWO" not in str(opened.get("bitsPerChannel", "")):
                    print("WARNING: not a 32-bit document — check Edit > "
                          "OpenColorIO Settings (Document Depth must be "
                          "32-bit, and OCIO features enabled).",
                          file=sys.stderr)
        else:
            names = [n.strip() for n in args.matte_names.split(",") if n.strip()]
            result = receive(tiff_path=args.tiff, out_path=args.out,
                             target_space=args.target_space,
                             working_space=args.working_space,
                             matte_dir=args.matte_dir, matte_names=names,
                             keep_empty_mattes=args.keep_empty_mattes,
                             bit_depth=args.bit_depth)
            print(f"read       {result['source']}")
            print(f"  channels {result['channels']}")
            print(f"  assumed  {result['working_space']!r} (Photoshop's "
                  f"working space; the file is untagged)")
            print(f"wrote      {result['out']}  tagged "
                  f"{result['retagged_as']!r}")
            if result["mattes"]:
                print("mattes:")
                for m in result["mattes"]:
                    if m.get("skipped"):
                        print(f"  - {m['name']:14s} {m['file_channel']:10s} "
                              f"SKIPPED ({m['skipped']})")
                    else:
                        print(f"  - {m['name']:14s} {m['file_channel']:10s} "
                              f"coverage {m['coverage']:6.2%}  "
                              f"{m.get('path', '(not written)')}")
            else:
                print("mattes:    none (no channels beyond RGB)")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True),
                               encoding="utf-8")
        print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
