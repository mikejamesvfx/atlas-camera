"""Score a hidden-geometry prediction against EXACT Blender truth (RESEARCH).

Complements `g5_geometric_truth.py`. G5 scores against a real photograph but its
answer key is monocular depth, so it carries the depth model's error. Here the
scene is synthetic, so the truth is exact and the visibility classes the brief
asks for are decidable rather than estimated:

    VISIBLE        truth point the render camera actually saw
    OCCLUDED       truth point inside the frustum but behind a nearer surface
    OUT_OF_FRUSTUM truth point the camera could never have seen
    UNSUPPORTED    PREDICTED point with no truth anywhere near it — pure invention

Note the asymmetry: the first three classify TRUTH, the last classifies
PREDICTION. Conflating them is how a model gets credit for reproducing what it
was shown.

The depth buffer is rasterized from the truth samples themselves (min z per
pixel), so visibility and the MoGe->world scale registration are both exact —
no depth model anywhere in the loop.

Usage (Atlas env):
    python blender_truth_eval.py --scene out/truth_scene --volume out/truth_volfill
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def project(points, view_matrix, fx, fy, cx, cy):
    vm = np.asarray(view_matrix, dtype=np.float64)
    cam = points @ vm[:3, :3].T + vm[:3, 3]
    z = -cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam[:, 0] / z * fx + cx
        v = -cam[:, 1] / z * fy + cy
    return u, v, z


def depth_buffer(points, view_matrix, fx, fy, cx, cy, W, H):
    """Min-z buffer rasterized from the truth samples."""
    u, v, z = project(points, view_matrix, fx, fy, cx, cy)
    ok = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & np.isfinite(z)
    ui = u[ok].astype(int); vi = v[ok].astype(int); zz = z[ok]
    buf = np.full((H, W), np.inf)
    np.minimum.at(buf, (vi, ui), zz)
    return buf, (u, v, z, ok)


def classify_truth(points, view_matrix, fx, fy, cx, cy, W, H, *, rel_margin=0.02):
    buf, (u, v, z, inside) = depth_buffer(points, view_matrix, fx, fy, cx, cy, W, H)
    cls = np.full(len(points), "OUT_OF_FRUSTUM", dtype=object)
    ui = np.clip(u[inside].astype(int), 0, W - 1)
    vi = np.clip(v[inside].astype(int), 0, H - 1)
    front = buf[vi, ui]
    zin = z[inside]
    occ = zin > front * (1.0 + rel_margin)
    sub = np.where(inside)[0]
    cls[sub[~occ]] = "VISIBLE"
    cls[sub[occ]] = "OCCLUDED"
    return cls, buf


def _nn(query, target, sample=150000, seed=0):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    if len(query) == 0 or len(target) == 0:
        return {"n": int(len(query)), "median": float("nan"), "mean": float("nan"),
                "p90": float("nan")}
    if len(query) > sample:
        query = query[rng.choice(len(query), sample, replace=False)]
    if len(target) > sample:
        target = target[rng.choice(len(target), sample, replace=False)]
    d, _ = cKDTree(target).query(query, k=1)
    return {"n": int(len(query)), "median": float(np.median(d)),
            "mean": float(d.mean()), "p90": float(np.percentile(d, 90))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="out/truth_scene")
    ap.add_argument("--volume", default="out/truth_volfill")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="out/blender_truth_score.json")
    args = ap.parse_args()

    from atlas_camera.core.hidden_geometry import register_layers_to_depth
    from tudf_to_atlas import (load_volume, surface_points_canonical,
                               moge_to_atlas_camera, atlas_camera_to_world)

    scene = Path(args.scene)
    meta = json.loads((scene / "camera.json").read_text(encoding="utf-8"))
    with np.load(scene / "truth_points.npz") as d:
        T = np.asarray(d["points"], dtype=np.float64)
        tid = np.asarray(d["object_id"], dtype=np.int32)
    W, H = meta["image_width"], meta["image_height"]
    fx, fy = meta["fx_px"], meta["fy_px"]
    cx, cy = meta["cx_px"], meta["cy_px"]
    vm = np.asarray(meta["camera_view_matrix"], dtype=np.float64)

    cls, buf = classify_truth(T, vm, fx, fy, cx, cy, W, H)
    counts = {c: int((cls == c).sum()) for c in
              ("VISIBLE", "OCCLUDED", "OUT_OF_FRUSTUM")}
    print(f"truth points {len(T)}: " +
          "  ".join(f"{k} {v}" for k, v in counts.items()))
    T_vis = T[cls == "VISIBLE"]
    T_occ = T[cls == "OCCLUDED"]

    # Per-object occlusion, so the per-case story is visible not averaged away.
    names = meta["objects"]
    per_obj = {}
    for i, nm in enumerate(names):
        m = tid == i
        if not m.any():
            continue
        per_obj[nm] = {
            "total": int(m.sum()),
            "occluded": int((cls[m] == "OCCLUDED").sum()),
            "visible": int((cls[m] == "VISIBLE").sum()),
        }

    # --- prediction into the same world ---
    vol = load_volume(Path(args.volume))
    with np.load(Path(args.volume) / "pred_tudf_256.npz") as dd:
        vis_tudf = np.asarray(dd["visible_tudf"], dtype=np.float32)
        moge_pts = np.asarray(dd["moge_points"], dtype=np.float64)
        moge_mask = np.asarray(dd["moge_mask"], dtype=bool)

    # EXACT registration: MoGe z against the truth depth buffer, no depth model.
    finite = np.isfinite(buf)
    ref = np.where(finite, buf, 0.0)
    mh, mw = moge_pts.shape[:2]
    yi = (np.linspace(0, H - 1, mh)).astype(int)
    xi = (np.linspace(0, W - 1, mw)).astype(int)
    ref_small = ref[np.ix_(yi, xi)]
    mz = moge_pts[..., 2].copy(); mz[~moge_mask] = 0.0
    s, rel_mad, _ = register_layers_to_depth(mz[..., None], ref_small)
    print(f"MoGe->truth scale {s:.4f} (rel_mad {rel_mad:.3f})")

    pts_c, _ = surface_points_canonical(vol, threshold=args.threshold)
    if len(pts_c) == 0:
        print("PREDICTION EMPTY — nothing to score.")
        Path(args.out).write_text(json.dumps(
            {"truth_counts": counts, "per_object": per_obj,
             "prediction": "EMPTY"}, indent=2), encoding="utf-8")
        return
    idx = np.clip(np.rint((pts_c - vol.bbox_min) / vol.voxel_size - 0.5).astype(int),
                  0, vol.resolution - 1)
    supported = vis_tudf[idx[:, 2], idx[:, 1], idx[:, 0]] <= (args.threshold + 1.0)
    V = atlas_camera_to_world(moge_to_atlas_camera(pts_c, scale=s), vm)
    V_inv = V[~supported]
    print(f"prediction {len(V)} pts, invented {len(V_inv)} "
          f"({100*len(V_inv)/max(len(V),1):.1f}%)")

    # UNSUPPORTED: predicted points far from ANY truth surface — pure invention.
    from scipy.spatial import cKDTree
    tree = cKDTree(T[np.random.default_rng(0).choice(
        len(T), min(len(T), 400000), replace=False)])
    dpred, _ = tree.query(V, k=1)
    voxel = vol.voxel_edge_m
    tol = max(3.0 * voxel, 0.15)
    unsupported = dpred > tol
    print(f"UNSUPPORTED (>{tol:.2f} m from any truth): "
          f"{int(unsupported.sum())} of {len(V)} "
          f"({100*unsupported.mean():.1f}%)")

    report = {
        "scene": str(scene), "volume": str(args.volume),
        "voxel_edge_m": voxel, "unsupported_tolerance_m": tol,
        "truth_counts": counts, "per_object": per_obj,
        "registration": {"scale": s, "rel_mad": rel_mad},
        "prediction": {"points": int(len(V)), "invented": int(len(V_inv)),
                       "unsupported": int(unsupported.sum()),
                       "unsupported_fraction": float(unsupported.mean())},
        "recall_occluded": {
            "volfill_all": _nn(T_occ, V),
            "volfill_invented": _nn(T_occ, V_inv),
        },
        "recall_visible": {"volfill_all": _nn(T_vis, V)},
        "precision": {
            "prediction_to_truth": _nn(V, T),
            "invented_to_occluded_truth": _nn(V_inv, T_occ),
        },
    }
    print("\n--- RECALL (truth -> prediction, metres) ---")
    for grp in ("recall_occluded", "recall_visible"):
        for k, v in report[grp].items():
            print(f"  {grp}/{k:18} median {v['median']:7.3f} p90 {v['p90']:7.3f} (n={v['n']})")
    print("--- PRECISION (prediction -> truth, metres) ---")
    for k, v in report["precision"].items():
        print(f"  {k:28} median {v['median']:7.3f} p90 {v['p90']:7.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
