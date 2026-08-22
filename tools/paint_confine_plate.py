"""Confine an external generative edit to the region Atlas authorised.

Thin CLI over ``atlas_camera.paint.confine`` -- see that module for the
measured hazards (vendor EXR mislabelling, the lossy dwab default, and the fact
that passing gates does not mean the picture is right).

Usage::

    python tools/paint_confine_plate.py \
        --original <plate>.exr --edited <edited>.exr --mask <object>.png \
        --out <confined>.exr --out-mask <authorised>.png \
        [--vendor affinity|photoshop_beta] \
        [--roi-manifest <roi>.json] [--drop-px 320] [--dilate-px 45] \
        [--feather-px 12] [--bit-depth float]

Then score it -- with the AUTHORISED mask this writes, never the object mask::

    python tools/paint_roundtrip_score.py --original <plate>.exr \
        --edited <confined>.exr --mask <authorised>.png
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
    from atlas_camera.paint import ocio, vendors
    from atlas_camera.paint.confine import confine

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--original", required=True, type=Path)
    p.add_argument("--edited", required=True, type=Path)
    p.add_argument("--mask", required=True, type=Path,
                   help="The object mask the edit was briefed with (PNG)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--out-mask", type=Path, default=None,
                   help="Authorised-mask PNG (the blend ramp's full support) -- "
                        "hand THIS to the scorer, not --mask")
    p.add_argument("--vendor", default=None, choices=sorted(vendors.PROFILES),
                   help="Take measured dilate/feather defaults from this profile")
    p.add_argument("--dilate-px", type=int, default=None)
    p.add_argument("--drop-px", type=int, default=0,
                   help="Extend the mask straight DOWN per column before "
                        "dilation: a ground-standing object's legs, footings "
                        "and contact shadow sit below it, not around it")
    p.add_argument("--feather-px", type=int, default=None)
    p.add_argument("--roi-manifest", type=Path, default=None,
                   help="Manifest from paint_roi_export.py: --edited is then an "
                        "edited CROP, pasted back at the recorded offset with "
                        "everything outside the ROI left bit-for-bit identical")
    p.add_argument("--bit-depth", default="float", choices=("half", "float"),
                   help="'float' (zip, LOSSLESS) by default because this plate "
                        "gets gated: 'half' selects the dwab DCT codec, which "
                        "moves every pixel past the scorer's 1e-4 threshold and "
                        "drowns the edit being measured")
    p.add_argument("--ocio-config", type=Path, default=None)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)

    profile = vendors.get(args.vendor) if args.vendor else None
    dilate_px = args.dilate_px if args.dilate_px is not None else (
        profile.dilate_px if profile else 45)
    feather_px = args.feather_px if args.feather_px is not None else (
        profile.feather_px if profile else 12)
    if profile:
        for caveat in profile.caveats:
            print(f"note [{profile.key}]: {caveat}", file=sys.stderr)
        if not profile.measured:
            print(f"note [{profile.key}]: dilate/feather are PROVISIONAL for "
                  f"this vendor -- nothing has been scored yet.", file=sys.stderr)

    with ocio.scoped_config(args.ocio_config) as identity:
        print(ocio.describe(identity), file=sys.stderr)
        stats = confine(
            original_path=args.original, edited_path=args.edited,
            mask_path=args.mask, out_path=args.out,
            out_mask_path=args.out_mask, roi_manifest=args.roi_manifest,
            drop_px=args.drop_px, dilate_px=dilate_px,
            feather_px=feather_px, bit_depth=args.bit_depth)

    print(f"object px             {stats['object_px']:>12,}")
    print(f"authorised px         {stats['authorised_px']:>12,}"
          f"  (drop {stats['drop_px']} + dilate {stats['dilate_px']}"
          f" + feather {stats['feather_px']})")
    print(f"edit changed px       {stats['edit_changed_px']:>12,}")
    print(f"  discarded (outside) {stats['edit_changed_outside_authorised_px']:>12,}")
    print(f"kept fraction of edit {stats['kept_fraction_of_edit']:>12.4f}")
    print(f"re-tagged as          {stats['source_colorspace']!r}")
    print(f"wrote {args.out}")
    if args.out_mask:
        print(f"wrote {args.out_mask}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(stats, indent=2, sort_keys=True),
                               encoding="utf-8")
        print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
