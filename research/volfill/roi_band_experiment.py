"""ROI crop x depth band -> effective VolFill resolution (RESEARCH).

Answers a specific question: can an Atlas ROI crop camera plus a depth band
constrain a deep exterior plate enough that a 256^3 TUDF resolves usable
geometry?

The trap this measures around: VolFill's canonical bbox is ISOTROPIC
(`half_scale = max(half_extent)` in `estimate_isotropic_bounds`), so the cube is
sized by the LONGEST axis. Constraining one axis alone buys nothing — on a street
plate the long axis is depth, not width, so an ROI crop by itself leaves the cube
sized by how far down the street the camera can see. The two constraints have to
multiply:

    ROI crop   -> bounds the ANGULAR (lateral) extent
    depth band -> bounds the RADIAL extent
    together   -> a genuinely bounded box

Uniform rescaling (e.g. a VFX 1/10 working scale) does NOT help: the bbox is fit
to the data, so voxel size and feature size shrink together and voxels-per-feature
is invariant. Atlas's own scale is applied on the return leg (the `s` scalar in
tudf_to_atlas) and never reaches VolFill at all.

Runs in the ATLAS env to build crops (needs atlas_camera), then shells out to the
isolated venv for inference — the same two-env split as run_volfill/roundtrip_eval.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
VENV_PY = HERE / ".venv" / "Scripts" / "python.exe"


def build_crop(image_path: Path, roi_frac: tuple[float, float, float, float],
               out_png: Path) -> dict:
    """Crop with a REAL Atlas crop camera so intrinsics stay exact.

    An image crop would silently shift the principal point and poison the
    round-trip; `crop_intrinsics` tracks it.
    """
    from atlas_camera.core.camera_crop import CropTransform, RegionROI, crop_intrinsics
    from atlas_camera.core.schema import AtlasIntrinsics

    with Image.open(image_path) as im:
        W, H = im.size
        fx, fy = float(W), float(W)  # placeholder if no solve is supplied
        x = int(round(roi_frac[0] * W))
        y = int(round(roi_frac[1] * H))
        w = int(round(roi_frac[2] * W))
        h = int(round(roi_frac[3] * H))
        roi = RegionROI(x=x, y=y, width=w, height=h)
        crop = im.crop((x, y, x + w, y + h))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_png)

    intr = AtlasIntrinsics(image_width=W, image_height=H, fx_px=fx, fy_px=fy,
                           cx_px=W / 2.0, cy_px=H / 2.0)
    cropped = crop_intrinsics(intr, roi)
    xform = CropTransform(source_width=W, source_height=H, roi=roi,
                          output_width=w, output_height=h)
    return {
        "roi": {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height},
        "source_size": [W, H],
        "crop_size": [w, h],
        "cropped_intrinsics": {
            "fx_px": cropped.fx_px, "fy_px": cropped.fy_px,
            "cx_px": cropped.cx_px, "cy_px": cropped.cy_px,
            "image_width": cropped.image_width, "image_height": cropped.image_height,
        },
        "crop_transform": {
            "source_width": xform.source_width, "source_height": xform.source_height,
            "output_width": xform.output_width, "output_height": xform.output_height,
        },
    }


def run_volfill(image: Path, out: Path, max_depth: float | None,
                steps: int = 50) -> dict:
    cmd = [str(VENV_PY), str(HERE / "run_volfill.py"), "--image", str(image),
           "--out", str(out), "--steps", str(steps)]
    if max_depth is not None:
        cmd += ["--max-depth", str(max_depth)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": (p.stderr or p.stdout).strip()[-400:]}
    return json.loads((out / "metadata.json").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--roi", nargs=4, type=float, default=[0.30, 0.35, 0.34, 0.45],
                    metavar=("X", "Y", "W", "H"),
                    help="ROI as fractions of the source frame.")
    ap.add_argument("--band", type=float, default=25.0,
                    help="Depth band far limit in metres (the radial constraint).")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    name = args.name or Path(args.image).stem
    src = Path(args.image)
    outdir = HERE / "out"
    rows = []

    # The 2x2: each constraint alone, neither, and both. The point is that only
    # the product moves the number, because the cube tracks the longest axis.
    arms = [
        ("neither", False, None),
        ("roi_only", True, None),
        ("band_only", False, args.band),
        ("roi_x_band", True, args.band),
    ]

    crop_png = outdir / f"roi_{name}" / "crop.png"
    crop_info = build_crop(src, tuple(args.roi), crop_png)
    (outdir / f"roi_{name}").mkdir(parents=True, exist_ok=True)
    (outdir / f"roi_{name}" / "crop_camera.json").write_text(
        json.dumps(crop_info, indent=2), encoding="utf-8")

    for arm, use_roi, band in arms:
        img = crop_png if use_roi else src
        out = outdir / f"roiband_{name}_{arm}"
        meta = run_volfill(img, out, band, steps=args.steps)
        if "error" in meta:
            print(f"{arm:<12} FAILED: {meta['error'][:140]}")
            rows.append({"arm": arm, "error": meta["error"]})
            continue
        edge = meta["voxel_edge_m"]
        extent = max(meta["extent_xyz"])
        rows.append({"arm": arm, "roi": use_roi, "band_m": band,
                     "extent_m": extent, "voxel_edge_m": edge,
                     "sampling_s": meta["timings_s"]["sampling"]})
        print(f"{arm:<12} extent {extent:8.1f} m   voxel {edge*100:7.2f} cm   "
              f"sample {meta['timings_s']['sampling']:.1f}s")

    dest = outdir / f"roiband_{name}.json"
    dest.write_text(json.dumps({"roi_frac": args.roi, "band_m": args.band,
                                "crop": crop_info, "arms": rows}, indent=2),
                    encoding="utf-8")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
