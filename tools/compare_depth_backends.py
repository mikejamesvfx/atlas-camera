"""A/B-compare depth backends (Depth Anything V2 vs DA3) on real images.

For each image the learned solve (GeoCalib) runs ONCE and is shared across all
backends, so every metric difference below is attributable to the depth model
alone. Per backend it reports the numbers the Atlas pipeline actually consumes:

  - ground-fit scale + inlier count      (relief_mesh.estimate_ground_scale)
  - relief-mesh torn fraction + faces    (relief_mesh.build_relief_mesh)
  - measured camera height + confidence  (solver.estimate_ground_height_from_depth)
  - near/far range, focal_source, runtime

Usage (venv with [neural] installed; DA3 models additionally need
the depth_anything_3 package — see INSTALL.md):

    python tools/compare_depth_backends.py examples/atlas_4k_testimages/*.png
    python tools/compare_depth_backends.py img.png --models default --json out.json

See docs/dev/da3_backend_test_plan.md for how to read the results.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

DEFAULT_MODELS = [
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "depth-anything/DA3METRIC-LARGE",
]


def _expand_images(args: list[str]) -> list[str]:
    """Expand globs ourselves — PowerShell/cmd don't."""
    out: list[str] = []
    for a in args:
        matches = sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a]
        out.extend(matches)
    return [p for p in out if Path(p).is_file()]


def _fmt(v, spec=".3f"):
    if v is None:
        return "-"
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return str(v)


def compare_image(image_path: str, models: list[str], device: str | None,
                  grid_long_edge: int, depth_edge_rel: float) -> dict:
    import numpy as np

    from atlas_camera.core.relief_mesh import build_relief_mesh, estimate_ground_scale
    from atlas_camera.core.solver import (
        _resize_depth,
        estimate_ground_height_from_depth,
        solve_from_learned_prior,
    )
    from atlas_camera.inference.depth_estimator import estimate_depth
    from atlas_camera.inference.learned_prior import estimate_camera_prior

    prior = estimate_camera_prior(image_path, device=device)
    solve = solve_from_learned_prior(prior, image_path=image_path)
    intr = solve.camera.intrinsics
    extr = solve.camera.extrinsics
    width, height = int(intr.image_width), int(intr.image_height)
    fx = float(intr.fx_px)
    fy = float(intr.fy_px or fx)
    cx = float(intr.cx_px if intr.cx_px is not None else width / 2.0)
    cy = float(intr.cy_px if intr.cy_px is not None else height / 2.0)
    vm = np.asarray(extr.camera_view_matrix, dtype=np.float64)
    # Same horizon estimate solve_still_image_learned uses for its ground fit.
    horizon_y = height / 2.0 + fy * math.tan(math.radians(prior.pitch_deg))

    report: dict = {
        "image": image_path,
        "solve": {
            "width": width, "height": height,
            "focal_px": float(prior.focal_px),
            "pitch_deg": float(prior.pitch_deg),
        },
        "backends": {},
    }

    for model_id in models:
        entry: dict = {"model_id": model_id}
        try:
            t0 = time.perf_counter()
            result = estimate_depth(
                image_path, model_id=model_id,
                device=device, focal_px=prior.focal_px,
            )
            entry["runtime_s"] = round(time.perf_counter() - t0, 2)

            depth = result.depth
            if depth.shape != (height, width):
                depth = _resize_depth(depth, width, height)

            entry["is_metric"] = result.is_metric
            entry["near_m"] = float(result.near)
            entry["far_m"] = float(result.far)
            entry["focal_source"] = result.metadata.get("focal_source")

            scale, scale_info = estimate_ground_scale(
                depth, view_matrix=vm, fx=fx, fy=fy, cx=cx, cy=cy,
                horizon_y=horizon_y,
            )
            entry["ground_scale"] = float(scale)
            entry["ground_inliers"] = int(
                scale_info.get("n_ground", scale_info.get("inliers", 0)) or 0
            )

            mesh = build_relief_mesh(
                depth, view_matrix=vm, fx=fx, fy=fy, cx=cx, cy=cy,
                grid_long_edge=grid_long_edge, depth_edge_rel=depth_edge_rel,
                scale=scale, horizon_y=horizon_y,
            )
            stats = getattr(mesh, "stats", {}) or {}
            entry["torn_fraction"] = stats.get("torn_fraction")
            entry["n_faces"] = stats.get("n_faces")

            ground = estimate_ground_height_from_depth(
                depth, rotation=vm[:3, :3], fx=fx, fy=fy, cx=cx, cy=cy,
                horizon_y=horizon_y,
            )
            entry["camera_height_m"] = ground.get("camera_height")
            entry["height_confidence"] = ground.get("confidence")
        except Exception as exc:  # keep comparing the other backends
            import traceback
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc().splitlines()[-3:]
        report["backends"][model_id] = entry

    return report


def print_report(report: dict) -> None:
    s = report["solve"]
    print(f"\n=== {report['image']} ===")
    print(f"    solve: {s['width']}x{s['height']}  focal {s['focal_px']:.1f}px  "
          f"pitch {s['pitch_deg']:.1f} deg")
    cols = ["backend", "time_s", "metric", "near_m", "far_m", "focal_src",
            "gnd_scale", "inliers", "torn_frac", "faces", "cam_h_m", "h_conf"]
    print("    " + " | ".join(f"{c:>10}" for c in cols))
    for model_id, e in report["backends"].items():
        short = model_id.split("/")[-1].replace("Depth-Anything-", "")[:18]
        if "error" in e:
            print(f"    {short:>10} | ERROR: {e['error']}")
            continue
        row = [short, _fmt(e.get("runtime_s"), ".2f"),
               str(e.get("is_metric")), _fmt(e.get("near_m"), ".2f"),
               _fmt(e.get("far_m"), ".1f"), str(e.get("focal_source") or "-"),
               _fmt(e.get("ground_scale")), str(e.get("ground_inliers", "-")),
               _fmt(e.get("torn_fraction"), ".4f"), str(e.get("n_faces", "-")),
               _fmt(e.get("camera_height_m"), ".2f"),
               _fmt(e.get("height_confidence"))]
        print("    " + " | ".join(f"{c:>10}" for c in row))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="+", help="image paths (globs OK)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help=f"model ids to compare (default: {DEFAULT_MODELS})")
    ap.add_argument("--device", default=None, help="cuda/mps/cpu (default: auto)")
    ap.add_argument("--grid", type=int, default=128, help="relief grid long edge")
    ap.add_argument("--edge", type=float, default=0.5, help="depth_edge_rel")
    ap.add_argument("--json", dest="json_out", default=None, help="dump raw JSON here")
    args = ap.parse_args(argv)

    images = _expand_images(args.images)
    if not images:
        print("No images matched.", file=sys.stderr)
        return 2

    reports = []
    for path in images:
        report = compare_image(path, args.models, args.device, args.grid, args.edge)
        print_report(report)
        reports.append(report)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2, default=str))
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
