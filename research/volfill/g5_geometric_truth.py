"""G5 GEOMETRIC truth: score invented geometry against structure really photographed.

The sh001 rig gives something rare — the same scene from two MEASURED poses
14.578 m apart. Surfaces hidden behind occluders in frame 1 are directly visible
in frame 2. So frame 2's reconstruction is an answer key for exactly the geometry
frame 1 could not see, and a hidden-geometry predictor can be scored on it
without any synthetic scene.

    frame 1 -> depth -> relief mesh          (what Atlas has today)
    frame 1 -> VolFill -> invented surface   (the hypothesis under test)
    frame 2 -> depth -> world points         (the answer key)
              |
              +-> keep only points that were OCCLUDED from camera 1
                  = genuinely hidden structure, really photographed

Then the question is falsifiable: is the invented surface CLOSER to that hidden
structure than the baseline relief mesh is? The baseline cannot represent hidden
geometry at all, so any real recovery must beat it.

HONEST LIMIT ON "TRUTH": frame 2's points come from monocular depth
(DA-V2-Metric-Outdoor), not a laser scan. They are anchored by a real photograph
at a surveyed 14.578 m baseline and a solved pose, which is far better than a
synthetic prior, but the answer key carries the depth model's own error. Treat
the numbers as RELATIVE (baseline vs VolFill on identical truth), never as
absolute metric accuracy. The registration check below reports the noise floor
so the comparison can be read against it.

Usage (Atlas env):
    python g5_geometric_truth.py --volume out/fix_sh001 --out out/g5_geometric.json
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
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"


def _cached_depth(image_path: Path, cache: Path, height: int, width: int):
    """DA-V2 metric depth, cached — a 5178x7752 plate is minutes of GPU."""
    if cache.exists():
        with np.load(cache) as d:
            return np.asarray(d["depth"], dtype=np.float64)
    from atlas_camera.inference import depth_estimator as de
    res = de.estimate_depth(str(image_path), model_id=DEPTH_MODEL)
    depth = np.asarray(getattr(res, "depth", res), dtype=np.float64)
    if depth.shape != (height, width):
        depth = np.asarray(
            Image.fromarray(depth.astype(np.float32)).resize(
                (width, height), Image.BILINEAR), dtype=np.float64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, depth=depth.astype(np.float32))
    return depth


def backproject(depth, view_matrix, fx, fy, cx, cy, stride=1):
    """Depth -> world points, Atlas convention (x right, y up, -z forward)."""
    d = np.asarray(depth, dtype=np.float64)[::stride, ::stride]
    h, w = d.shape
    uu, vv = np.meshgrid(np.arange(w) * stride, np.arange(h) * stride)
    x = (uu - cx) / fx * d
    y = -(vv - cy) / fy * d
    z = -d
    pts = np.stack([x, y, z], axis=-1)
    c2w = np.linalg.inv(np.asarray(view_matrix, dtype=np.float64))
    world = pts.reshape(-1, 3) @ c2w[:3, :3].T + c2w[:3, 3]
    good = np.isfinite(world).all(axis=1) & (d.reshape(-1) > 1e-6)
    return world[good]


def occluded_from(world_pts, view_matrix, depth_ref, fx, fy, cx, cy,
                  *, rel_margin=0.06):
    """Which world points were HIDDEN from the reference camera?

    A point is occluded when it projects inside the reference frame but sits
    measurably BEHIND whatever that camera actually saw along the same ray. The
    margin is relative because depth error grows with range; a fixed metric
    margin would call the whole far field occluded.

    Returns (occluded_mask, in_frustum_mask).
    """
    vm = np.asarray(view_matrix, dtype=np.float64)
    cam = world_pts @ vm[:3, :3].T + vm[:3, 3]
    z = -cam[:, 2]                                   # forward distance
    H, W = depth_ref.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam[:, 0] / z * fx + cx
        v = -cam[:, 1] / z * fy + cy
    inside = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    occ = np.zeros(len(world_pts), bool)
    ui = np.clip(u[inside].astype(int), 0, W - 1)
    vi = np.clip(v[inside].astype(int), 0, H - 1)
    dref = depth_ref[vi, ui]
    occ[inside] = (dref > 1e-6) & (z[inside] > dref * (1.0 + rel_margin))
    return occ, inside


def _nn_stats(query, target, sample=120000, seed=0):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    if len(query) == 0 or len(target) == 0:
        return {"n_query": int(len(query)), "mean": float("nan"),
                "median": float("nan"), "p90": float("nan")}
    if len(query) > sample:
        query = query[rng.choice(len(query), sample, replace=False)]
    if len(target) > sample:
        target = target[rng.choice(len(target), sample, replace=False)]
    d, _ = cKDTree(target).query(query, k=1)
    return {"n_query": int(len(query)), "mean": float(d.mean()),
            "median": float(np.median(d)), "p90": float(np.percentile(d, 90))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default="out/fix_sh001")
    ap.add_argument("--out", default="out/g5_geometric.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=6,
                    help="Pixel stride when back-projecting (memory).")
    args = ap.parse_args()

    from atlas_camera.core.io import load_solve_json
    from atlas_camera.core.camera_spec import CameraSpec
    from atlas_camera.core.hidden_geometry import register_layers_to_depth
    from tudf_to_atlas import (load_volume, surface_points_canonical,
                               moge_to_atlas_camera, atlas_camera_to_world)

    solve = load_solve_json(str(SP / "sh001_rig.json"))
    intr = solve.camera.intrinsics
    spec = CameraSpec.from_intrinsics(intr)
    W, H = int(intr.image_width), int(intr.image_height)
    view1 = np.asarray(solve.camera.extrinsics.camera_view_matrix, dtype=np.float64)
    src_cam = solve.projection_sources[0].camera
    view2 = np.asarray(src_cam.extrinsics.camera_view_matrix, dtype=np.float64)

    c1 = np.linalg.inv(view1)[:3, 3]
    c2 = np.linalg.inv(view2)[:3, 3]
    baseline_m = float(np.linalg.norm(c2 - c1))
    print(f"rig {W}x{H} fx={spec.fx:.1f}  baseline {baseline_m:.3f} m")

    cache = Path("out/_g5_depth")
    d1 = _cached_depth(SP / "DSCF3915.png", cache / "d1.npz", H, W)
    d2 = _cached_depth(SP / "DSCF3916.png", cache / "d2.npz", H, W)
    print(f"depth1 {d1.shape} {np.nanmin(d1):.2f}..{np.nanmax(d1):.2f} m")
    print(f"depth2 {d2.shape} {np.nanmin(d2):.2f}..{np.nanmax(d2):.2f} m")

    # --- the two frames' reconstructions, in one world ---
    P1 = backproject(d1, view1, spec.fx, spec.fy, spec.cx, spec.cy, args.stride)
    P2 = backproject(d2, view2, spec.fx, spec.fy, spec.cx, spec.cy, args.stride)
    print(f"P1 {len(P1)} pts   P2 {len(P2)} pts")

    # --- split the answer key: hidden from camera 1 vs already visible ---
    occ2, in2 = occluded_from(P2, view1, d1, spec.fx, spec.fy, spec.cx, spec.cy)
    P2_hidden = P2[occ2]
    P2_seen = P2[in2 & ~occ2]
    print(f"answer key: {len(P2_hidden)} HIDDEN pts, {len(P2_seen)} already-visible, "
          f"{int((~in2).sum())} out of frustum")

    # --- noise floor: how well do the two reconstructions agree where BOTH saw
    # --- the surface? Everything below must be read against this number.
    floor = _nn_stats(P2_seen, P1)
    print(f"NOISE FLOOR (already-visible P2 -> P1): median {floor['median']:.3f} m")

    # --- VolFill hidden geometry into the same world ---
    vol = load_volume(Path(args.volume))
    with np.load(Path(args.volume) / "pred_tudf_256.npz") as dd:
        vis_tudf = np.asarray(dd["visible_tudf"], dtype=np.float32)
        moge_pts = np.asarray(dd["moge_points"], dtype=np.float64)
        moge_mask = np.asarray(dd["moge_mask"], dtype=bool)

    # MoGe metres -> this solve's metres. Same median-ratio registration Atlas
    # already uses; rel_mad is the quality signal.
    mz = moge_pts[..., 2].copy()
    mz[~moge_mask] = 0.0
    mh, mw = mz.shape
    d1_small = np.asarray(Image.fromarray(d1.astype(np.float32)).resize(
        (mw, mh), Image.BILINEAR), dtype=np.float64)
    s, rel_mad, _ = register_layers_to_depth(mz[..., None], d1_small)
    print(f"MoGe->solve depth scale {s:.4f} (rel_mad {rel_mad:.3f})")

    pts_c, tud = surface_points_canonical(vol, threshold=args.threshold)
    idx = np.clip(np.rint(
        (pts_c - vol.bbox_min) / vol.voxel_size - 0.5).astype(int), 0, vol.resolution - 1)
    supported = vis_tudf[idx[:, 2], idx[:, 1], idx[:, 0]] <= (args.threshold + 1.0)
    V = atlas_camera_to_world(moge_to_atlas_camera(pts_c, scale=s), view1)
    V_inv = V[~supported]
    print(f"VolFill: {len(V)} surface pts, {len(V_inv)} invented "
          f"({100*len(V_inv)/max(len(V),1):.1f}%)")

    # Restrict invented geometry to what frame 2 could actually adjudicate:
    # inside its frustum, and not behind the surface frame 2 itself saw (a point
    # hidden from BOTH cameras has no answer key either way).
    _, in_f2 = occluded_from(V_inv, view2, d2, spec.fx, spec.fy, spec.cx, spec.cy)
    V_inv_seen2 = V_inv[in_f2]
    print(f"invented inside frame-2 frustum: {len(V_inv_seen2)} of {len(V_inv)} "
          f"({100*len(V_inv_seen2)/max(len(V_inv),1):.1f}%)")

    # --- the comparison ---
    report = {
        "rig": {"baseline_m": baseline_m, "image": [W, H], "fx": spec.fx,
                "stride": args.stride, "depth_model": DEPTH_MODEL},
        "answer_key": {"hidden_pts": int(len(P2_hidden)),
                       "already_visible_pts": int(len(P2_seen)),
                       "out_of_frustum": int((~in2).sum())},
        "noise_floor_visible_m": floor,
        "moge_registration": {"scale": s, "rel_mad": rel_mad},
        "volfill": {"surface_pts": int(len(V)), "invented_pts": int(len(V_inv))},
        # Does each candidate EXPLAIN the hidden truth? (truth -> candidate)
        "recall_hidden": {
            "baseline_relief": _nn_stats(P2_hidden, P1),
            "volfill_all": _nn_stats(P2_hidden, V),
            "volfill_invented": _nn_stats(P2_hidden, V_inv),
            "baseline_plus_volfill": _nn_stats(
                P2_hidden, np.vstack([P1, V_inv]) if len(V_inv) else P1),
        },
        # Is what it invented actually THERE? (candidate -> truth)
        #
        # FAIRNESS: VolFill predicts the whole isotropic cube, much of which
        # frame 2 never observed either. Scoring every invented point against
        # the answer key punishes it for regions the key cannot adjudicate, so
        # the headline number is restricted to invented geometry that actually
        # lands inside frame 2's view — where the key has something to say.
        "precision_invented": {
            "volfill_invented_to_hidden_ALL": _nn_stats(V_inv, P2_hidden),
            "volfill_invented_to_all_p2_ALL": _nn_stats(V_inv, P2),
            "volfill_invented_in_frame2_to_p2": _nn_stats(V_inv_seen2, P2),
            "volfill_invented_in_frame2_to_hidden": _nn_stats(V_inv_seen2, P2_hidden),
            "invented_inside_frame2_fraction":
                float(len(V_inv_seen2) / max(len(V_inv), 1)),
        },
    }

    print("\n--- RECALL of hidden truth (truth -> candidate, metres) ---")
    for k, v in report["recall_hidden"].items():
        print(f"  {k:26} median {v['median']:7.3f}  mean {v['mean']:7.3f}  "
              f"p90 {v['p90']:7.3f}")
    print("--- PRECISION of invented geometry (invented -> truth, metres) ---")
    for k, v in report["precision_invented"].items():
        if not isinstance(v, dict):
            print(f"  {k:38} {v:.3f}")
            continue
        print(f"  {k:38} median {v['median']:7.3f}  mean {v['mean']:7.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
