"""Score an externally-edited plate against its original, against the gates.

Thin CLI over ``atlas_camera.paint.score``. Formerly
``tools/affinity_roundtrip_score.py``; that path still works as a deprecation
shim, because ``reports/affinity_bridge_demo.md`` publishes it.

Usage::

    python tools/paint_roundtrip_score.py \
        --original <plate>.exr --edited <confined>.exr --mask <authorised>.png \
        [--out report.json] [--rim-px 2] [--change-eps 1e-4]

Exit 0 = accepted (every available gated metric passes), 2 = rejected (a gate
failed), 3 = inconclusive (no gated metric had evidence).

Hand this the AUTHORISED mask that ``paint_confine_plate`` wrote, not the raw
object mask: a feather is spill unless the authorised mask includes it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    from atlas_camera.paint import ocio, vendors
    from atlas_camera.paint.score import (exit_code, format_table, score,
                                          write_report)

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--original", required=True, type=Path,
                   help="The pre-edit plate")
    p.add_argument("--edited", required=True, type=Path,
                   help="The post-edit plate saved by the paint package")
    p.add_argument("--mask", required=True, type=Path,
                   help="The region the edit was AUTHORISED to touch (PNG)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--rim-px", type=int, default=2)
    p.add_argument("--vendor", default=None, choices=sorted(vendors.PROFILES),
                   help="Take the measured change-eps from this profile")
    p.add_argument("--change-eps", type=float, default=None,
                   help="Per-channel threshold for 'this pixel changed'. 1e-4 "
                        "suits a float round trip; a 16-bit display-referred "
                        "leg quantises further and needs its own measured value")
    p.add_argument("--ocio-config", type=Path, default=None)
    args = p.parse_args(argv)

    profile = vendors.get(args.vendor) if args.vendor else None
    change_eps = args.change_eps if args.change_eps is not None else (
        profile.change_eps if profile else 1e-4)

    with ocio.scoped_config(args.ocio_config) as identity:
        print(ocio.describe(identity), file=sys.stderr)
        payload = score(original_path=args.original, edited_path=args.edited,
                        mask_path=args.mask, rim_px=args.rim_px,
                        change_eps=change_eps)

    print(format_table(payload))
    if args.out:
        write_report(args.out, payload)
        print(f"report written to {args.out}")
    return exit_code(payload)


if __name__ == "__main__":
    sys.exit(main())
