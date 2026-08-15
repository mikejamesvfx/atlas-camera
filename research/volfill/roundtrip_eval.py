"""Round-trip proof + visible/occluded split for a VolFill run (RESEARCH).

Runs in the ATLAS env (needs numpy + atlas_camera; does NOT need torch, VolFill,
or the isolated venv). Reads only what `run_volfill.py` wrote.

THE ROUND-TRIP PROOF, DECOMPOSED
--------------------------------
The chain canonical -> MoGe camera -> Atlas camera -> Atlas world has exactly one
estimated quantity (the depth scale ``s``); everything else is arithmetic. So it
is verified in three independent pieces rather than one end-to-end eyeball:

1. canonical -> MoGe camera: compare the VISIBLE TUDF's surface voxels against the
   raw MoGe pointmap they were voxelized from. Ground truth is exact and needs no
   Atlas solve. A correct mapping lands at the quantisation floor (~voxel/2, the
   mean offset from a point to its own voxel centre); an axis transposition or a
   missing half-voxel shifts it by whole voxels.
2. MoGe camera -> Atlas world: exact linear algebra, pinned by
   ``tests/test_tudf_to_atlas.py`` (camera origin, forward ray, handedness).
3. scale ``s``: measured by ``core.hidden_geometry.register_layers_to_depth``,
   reported with its ``rel_mad`` quality signal. Never assumed to be 1.

VISIBLE vs OCCLUDED
-------------------
VolFill builds its own visible TUDF from the MoGe points — that is the definition
of "what was observable from this camera". Predicted surface voxels near the
visible surface are VISIBLE; the rest are OCCLUDED (genuinely inferred). This
splits the metrics with no external visibility truth, which is the whole point:
a model scores well overall by reproducing what it was already shown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tudf_to_atlas import load_volume, surface_points_canonical, volfill_to_atlas_world


def _chamfer(a: np.ndarray, b: np.ndarray, *, sample: int = 60000,
             seed: int = 0) -> dict[str, float]:
    """Symmetric Chamfer via a KD-tree, subsampled for tractability."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    if a.shape[0] > sample:
        a = a[rng.choice(a.shape[0], sample, replace=False)]
    if b.shape[0] > sample:
        b = b[rng.choice(b.shape[0], sample, replace=False)]
    if a.size == 0 or b.size == 0:
        return {"a_to_b": float("nan"), "b_to_a": float("nan"),
                "chamfer": float("nan"), "median_a_to_b": float("nan")}
    da, _ = cKDTree(b).query(a, k=1)
    db, _ = cKDTree(a).query(b, k=1)
    return {
        "a_to_b": float(da.mean()),
        "b_to_a": float(db.mean()),
        "chamfer": float(da.mean() + db.mean()),
        "median_a_to_b": float(np.median(da)),
    }


def _fscore(pred: np.ndarray, gt: np.ndarray, tau: float,
            *, sample: int = 60000, seed: int = 0) -> dict[str, float]:
    """Precision / recall / F at distance threshold ``tau`` (metres)."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    if pred.shape[0] > sample:
        pred = pred[rng.choice(pred.shape[0], sample, replace=False)]
    if gt.shape[0] > sample:
        gt = gt[rng.choice(gt.shape[0], sample, replace=False)]
    if pred.size == 0 or gt.size == 0:
        return {"precision": 0.0, "recall": 0.0, "fscore": 0.0, "tau_m": tau}
    dp, _ = cKDTree(gt).query(pred, k=1)
    dg, _ = cKDTree(pred).query(gt, k=1)
    p = float((dp < tau).mean())
    r = float((dg < tau).mean())
    return {"precision": p, "recall": r,
            "fscore": float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0,
            "tau_m": tau}


def evaluate(sample_dir: str | Path, *, threshold: float = 0.5,
             view_matrix: Any = None, atlas_depth: Any = None) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    vol = load_volume(sample_dir)
    with np.load(sample_dir / "pred_tudf_256.npz") as data:
        vis_tudf = np.asarray(data["visible_tudf"], dtype=np.float32)
        moge_points = np.asarray(data["moge_points"], dtype=np.float64)
        moge_mask = np.asarray(data["moge_mask"], dtype=bool)

    voxel = vol.voxel_edge_m
    report: dict[str, Any] = {
        "sample": sample_dir.name,
        "voxel_edge_m": voxel,
        "resolution": vol.resolution,
        "threshold_voxels": threshold,
        "metadata": vol.metadata,
    }

    # --- (1) canonical -> MoGe camera, against the points it was built from ---
    vis_vol = type(vol)(tudf=vis_tudf, bbox_min=vol.bbox_min, extent=vol.extent,
                        truncation_voxels=vol.truncation_voxels)
    vis_pts, _ = surface_points_canonical(vis_vol, threshold=threshold)
    moge_pts = moge_points[moge_mask]
    inside = np.all((moge_pts >= vol.bbox_min) & (moge_pts <= vol.bbox_min + vol.extent),
                    axis=-1)
    moge_pts = moge_pts[inside]           # VolFill clips to the bbox too
    rt = _chamfer(vis_pts, moge_pts)
    # A point sits on average ~0.38*edge from its own voxel centre in 3D; anything
    # near or below one voxel edge means the index->metre mapping is correct.
    rt["quantisation_floor_m"] = 0.5 * voxel
    rt["ratio_to_floor"] = rt["median_a_to_b"] / max(0.5 * voxel, 1e-12)
    rt["PASS"] = bool(rt["median_a_to_b"] <= voxel)
    report["roundtrip_canonical_to_camera"] = rt

    # --- predicted surface, split VISIBLE vs OCCLUDED ---
    pred_mask = vol.tudf <= threshold
    vis_mask = vis_tudf <= threshold
    # "Near the visible surface" = within one voxel of it, i.e. inside the visible
    # TUDF's own truncation band rather than an arbitrary dilation.
    seen = vis_tudf <= (threshold + 1.0)
    n_pred = int(pred_mask.sum())
    report["surface_voxels"] = {
        "predicted": n_pred,
        "visible_reference": int(vis_mask.sum()),
        "predicted_visible": int((pred_mask & seen).sum()),
        "predicted_occluded": int((pred_mask & ~seen).sum()),
        "occluded_fraction": float((pred_mask & ~seen).sum() / max(n_pred, 1)),
    }

    # --- visible-geometry preservation: does it keep what it was shown? ---
    if n_pred:
        pred_pts, _ = surface_points_canonical(vol, threshold=threshold)
        report["visible_preservation"] = _chamfer(vis_pts, pred_pts)
        report["visible_fscore"] = _fscore(pred_pts, vis_pts, tau=2.0 * voxel)

    # --- (2)(3) Atlas world mapping, when a solve is supplied ---
    if view_matrix is not None:
        scale, rel_mad = 1.0, None
        if atlas_depth is not None:
            from tudf_to_atlas import estimate_depth_scale
            moge_depth = moge_points[..., 2].copy()
            moge_depth[~moge_mask] = 0.0
            scale, rel_mad = estimate_depth_scale(moge_depth, atlas_depth)
        out = volfill_to_atlas_world(vol, view_matrix, threshold=threshold,
                                     scale=scale)
        report["atlas_world"] = {
            "depth_scale": float(scale),
            "registration_rel_mad": (None if rel_mad is None else float(rel_mad)),
            "n_points": out["metadata"]["n_points"],
            "bounds_min": out["bounds"][0].tolist(),
            "bounds_max": out["bounds"][1].tolist(),
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dirs", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--solve", default=None, help="Atlas solve JSON (optional).")
    ap.add_argument("--out", default=None, help="Write the scorecard JSON here.")
    args = ap.parse_args()

    view_matrix = None
    if args.solve:
        from atlas_camera.core.io import load_solve_json
        solve = load_solve_json(args.solve)
        view_matrix = np.asarray(solve.camera.extrinsics.camera_view_matrix,
                                 dtype=np.float64)

    reports = [evaluate(d, threshold=args.threshold, view_matrix=view_matrix)
               for d in args.sample_dirs]
    for r in reports:
        rt = r["roundtrip_canonical_to_camera"]
        sv = r["surface_voxels"]
        print(f"{r['sample']:<28} voxel {r['voxel_edge_m']*100:6.1f} cm  "
              f"round-trip median {rt['median_a_to_b']*100:6.2f} cm "
              f"({rt['ratio_to_floor']:.2f}x floor) "
              f"{'PASS' if rt['PASS'] else 'FAIL'}  "
              f"occluded {sv['occluded_fraction']*100:5.1f}%")
    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
