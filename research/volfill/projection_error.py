"""Score hidden geometry the way Atlas actually USES it — as a projection surface.

Metric Chamfer answers "is this an accurate reconstruction". Atlas is not doing
reconstruction: it projects a photographed or inferred texture onto approximate
geometry and moves the camera a modest amount. That is far more forgiving, and in
a specific, computable way.

Decompose the geometry error at each point into

    RADIAL   along the source camera's view ray
    LATERAL  perpendicular to it

Under projection from the SOURCE camera, a radial error is invisible: the texture
travels down the same ray, so sliding a surface nearer or further along that ray
lands the same pixel on it. Only when the camera MOVES does radial error start to
show, and then only in proportion to the move. For a small offset angle t:

    screen_error_px  ~=  ( radial * sin(t) + lateral * cos(t) ) / depth * focal

So lateral error costs you immediately, radial error costs you `sin(t)` — at 5
degrees that is a 0.087 discount, at 2 degrees 0.035. A metric-Chamfer number
therefore over-penalises a projection surface by roughly an order of magnitude at
the move sizes Atlas ships.

This scores the G5 hidden-geometry result that way: real photographed truth,
error decomposed, expressed in SCREEN PIXELS at the offsets an artist would use.

Usage (Atlas env):
    python projection_error.py --volume out/fix_sh001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SP = Path(r"C:\Users\miike\AppData\Local\Temp\claude"
          r"\C--Users-miike-Desktop-AtlasCamera-Claude"
          r"\95656247-c586-4fdf-91ef-055be69d66cc\scratchpad")
OFFSETS_DEG = [2.0, 5.0, 10.0, 20.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default="out/fix_sh001")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--out", default="out/projection_error.json")
    args = ap.parse_args()

    from scipy.spatial import cKDTree
    from atlas_camera.core.io import load_solve_json
    from atlas_camera.core.camera_spec import CameraSpec
    from atlas_camera.core.hidden_geometry import register_layers_to_depth
    from tudf_to_atlas import (load_volume, surface_points_canonical,
                               moge_to_atlas_camera, atlas_camera_to_world)
    from g5_geometric_truth import backproject, occluded_from, _cached_depth

    solve = load_solve_json(str(SP / "sh001_rig.json"))
    intr = solve.camera.intrinsics
    spec = CameraSpec.from_intrinsics(intr)
    W, H = int(intr.image_width), int(intr.image_height)
    view1 = np.asarray(solve.camera.extrinsics.camera_view_matrix, dtype=np.float64)
    view2 = np.asarray(solve.projection_sources[0].camera.extrinsics.camera_view_matrix,
                       dtype=np.float64)

    cache = Path("out/_g5_depth")
    d1 = _cached_depth(SP / "DSCF3915.png", cache / "d1.npz", H, W)
    d2 = _cached_depth(SP / "DSCF3916.png", cache / "d2.npz", H, W)

    P2 = backproject(d2, view2, spec.fx, spec.fy, spec.cx, spec.cy, args.stride)
    occ2, _ = occluded_from(P2, view1, d1, spec.fx, spec.fy, spec.cx, spec.cy)
    T = P2[occ2]                                   # hidden truth, really photographed

    # --- prediction into world ---
    vol = load_volume(Path(args.volume))
    with np.load(Path(args.volume) / "pred_tudf_256.npz") as dd:
        vis_tudf = np.asarray(dd["visible_tudf"], dtype=np.float32)
        moge_pts = np.asarray(dd["moge_points"], dtype=np.float64)
        moge_mask = np.asarray(dd["moge_mask"], dtype=bool)
    mz = moge_pts[..., 2].copy(); mz[~moge_mask] = 0.0
    mh, mw = mz.shape
    d_small = np.asarray(Image.fromarray(d1.astype(np.float32)).resize(
        (mw, mh), Image.BILINEAR), dtype=np.float64)
    s, rel_mad, _ = register_layers_to_depth(mz[..., None], d_small)
    pts_c, _ = surface_points_canonical(vol, threshold=args.threshold)
    idx = np.clip(np.rint((pts_c - vol.bbox_min) / vol.voxel_size - 0.5).astype(int),
                  0, vol.resolution - 1)
    supported = vis_tudf[idx[:, 2], idx[:, 1], idx[:, 0]] <= (args.threshold + 1.0)
    V = atlas_camera_to_world(moge_to_atlas_camera(pts_c, scale=s), view1)
    V_inv = V[~supported]

    # Baseline: what Atlas has today, the frame-1 relief surface.
    P1 = backproject(d1, view1, spec.fx, spec.fy, spec.cx, spec.cy, args.stride)

    cam1 = np.linalg.inv(view1)[:3, 3]

    def decompose(truth, cand):
        """Nearest-candidate error at each truth point, split radial/lateral."""
        rng = np.random.default_rng(0)
        t = truth if len(truth) <= 120000 else truth[
            rng.choice(len(truth), 120000, replace=False)]
        c = cand if len(cand) <= 200000 else cand[
            rng.choice(len(cand), 200000, replace=False)]
        d, i = cKDTree(c).query(t, k=1)
        err = c[i] - t
        ray = t - cam1
        depth = np.linalg.norm(ray, axis=1)
        u = ray / np.maximum(depth, 1e-9)[:, None]
        radial = np.abs(np.einsum("ij,ij->i", err, u))
        lateral = np.linalg.norm(err - radial[:, None] * u * np.sign(
            np.einsum("ij,ij->i", err, u))[:, None], axis=1)
        return radial, lateral, depth, d

    out = {"registration": {"scale": s, "rel_mad": rel_mad},
           "truth_points": int(len(T)), "focal_px": spec.fx,
           "image": [W, H], "arms": {}}

    for name, cand in (("volfill_invented", V_inv), ("baseline_relief", P1)):
        radial, lateral, depth, total = decompose(T, cand)
        arm = {"total_m_median": float(np.median(total)),
               "radial_m_median": float(np.median(radial)),
               "lateral_m_median": float(np.median(lateral)),
               "depth_m_median": float(np.median(depth)),
               "screen_px": {}}
        for deg in OFFSETS_DEG:
            t = np.radians(deg)
            px = (radial * np.sin(t) + lateral * np.cos(t)) / np.maximum(depth, 1e-9) * spec.fx
            arm["screen_px"][f"{deg:g}deg"] = {
                "median": float(np.median(px)),
                "p90": float(np.percentile(px, 90)),
                "median_pct_of_width": float(np.median(px) / W * 100.0),
            }
        out["arms"][name] = arm

    print(f"hidden truth points {len(T)}   focal {spec.fx:.0f}px   frame {W}x{H}")
    for name, arm in out["arms"].items():
        print(f"\n--- {name} ---")
        print(f"  error   total {arm['total_m_median']:.3f} m   "
              f"radial {arm['radial_m_median']:.3f} m   "
              f"lateral {arm['lateral_m_median']:.3f} m   "
              f"(at {arm['depth_m_median']:.1f} m depth)")
        for k, v in arm["screen_px"].items():
            print(f"  {k:>6}  median {v['median']:8.1f} px  "
                  f"({v['median_pct_of_width']:.2f}% of width)  p90 {v['p90']:8.1f} px")

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
