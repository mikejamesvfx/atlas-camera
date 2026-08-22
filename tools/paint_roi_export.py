"""Cut the ROI crop that makes an external generative edit survivable.

Thin CLI over ``atlas_camera.paint.roi`` — see that module for why the crop is
required for some vendors and a measurement for others.

Usage::

    python tools/paint_roi_export.py --plate <master>.exr --mask <object>.png \
        --out <roi>.exr --out-mask <roi_mask>.png --manifest <roi>.json \
        [--vendor affinity|photoshop_beta] [--margin-px 240] [--drop-px 320]

Then: edit ``<roi>.exr`` in the paint package, export the crop, and run
``paint_confine_plate.py --roi-manifest <roi>.json``.
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
    from atlas_camera.paint.roi import export_roi

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plate", required=True, type=Path)
    p.add_argument("--mask", required=True, type=Path,
                   help="Object mask at the plate raster (PNG)")
    p.add_argument("--out", required=True, type=Path, help="ROI crop EXR")
    p.add_argument("--out-mask", type=Path, default=None)
    p.add_argument("--manifest", required=True, type=Path,
                   help="Crop rectangle JSON, for the paste-back step")
    p.add_argument("--vendor", default=None, choices=sorted(vendors.PROFILES),
                   help="Take drop/margin hints and caveats from this profile")
    p.add_argument("--margin-px", type=int, default=240,
                   help="Context the model needs around the object to continue "
                        "paving, kerbs and shadow into the hole")
    p.add_argument("--drop-px", type=int, default=0,
                   help="Extend the object bbox DOWNWARD before adding margin, "
                        "so legs, footings and contact shadow fall inside")
    p.add_argument("--bit-depth", default="float", choices=("half", "float"))
    p.add_argument("--ocio-config", type=Path, default=None,
                   help="Scope $OCIO to this config for this run only")
    args = p.parse_args(argv)

    if args.vendor:
        profile = vendors.get(args.vendor)
        for caveat in profile.caveats:
            print(f"note [{profile.key}]: {caveat}", file=sys.stderr)

    with ocio.scoped_config(args.ocio_config) as identity:
        print(ocio.describe(identity), file=sys.stderr)
        manifest = export_roi(
            plate_path=args.plate, mask_path=args.mask, out_path=args.out,
            manifest_path=args.manifest, out_mask_path=args.out_mask,
            margin_px=args.margin_px, drop_px=args.drop_px,
            bit_depth=args.bit_depth)

    roi = manifest["roi"]
    bbox = manifest["object_bbox"]
    print(f"plate            {manifest['plate_width']}x{manifest['plate_height']}")
    print(f"object bbox      x{bbox['x']} y{bbox['y']} "
          f"{bbox['width']}x{bbox['height']}"
          + (f"  (+drop {args.drop_px})" if args.drop_px else ""))
    print(f"ROI              x{roi['x']} y{roi['y']} {roi['width']}x{roi['height']}"
          f"   {manifest['roi_fraction_of_frame'] * 100:.1f}% of frame")
    print(f"wrote {args.out}")
    if args.out_mask:
        print(f"wrote {args.out_mask}")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
