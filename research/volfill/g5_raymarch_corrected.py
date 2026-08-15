"""G5 re-scored with the ray-march adapter — the double-wall bias removed.

Every Chamfer reported earlier came from marching cubes on an UNSIGNED field,
which returns a shell offset +/-threshold either side of the true surface. That
puts samples ~1 voxel (6.1 cm on sh001) off where the surface actually is, and
doubles the point count with a mirror copy. The numbers were therefore
pessimistic by an unknown amount, which this measures.

Same answer key as ``g5_geometric_truth.py`` — frame 2 of the sh001 rig,
photographed from a surveyed 14.6 m away, restricted to structure that was
OCCLUDED from frame 1 — so the two runs are directly comparable.

Usage (Atlas env):
    python g5_raymarch_corrected.py --volume out/fix_sh001
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default="out/fix_sh001")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--raster", type=int, default=384)
    ap.add_argument("--out", default="out/g5_raymarch_corrected.json")
    args = ap.parse_args()

    from atlas_camera.core.io import load_solve_json
    from atlas_camera.core.camera_spec import CameraSpec
    from atlas_camera.core.hidden_geometry import register_layers_to_depth
    from atlas_camera.core.volume_raymarch import march_layers
    from tudf_to_atlas import (load_volume, surface_points_canonical,
                               moge_to_atlas_camera, atlas_camera_to_world)
    from g5_geometric_truth import backproject, occluded_from, _cached_depth, _nn_stats

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

    P1 = backproject(d1, view1, spec.fx, spec.fy, spec.cx, spec.cy, args.stride)
    P2 = backproject(d2, view2, spec.fx, spec.fy, spec.cx, spec.cy, args.stride)
    occ2, in2 = occluded_from(P2, view1, d1, spec.fx, spec.fy, spec.cx, spec.cy)
    T_hidden = P2[occ2]
    T_seen = P2[in2 & ~occ2]
    floor = _nn_stats(T_seen, P1)
    print(f"answer key: {len(T_hidden)} hidden pts   noise floor "
          f"{floor['median']:.3f} m")

    vol = load_volume(Path(args.volume))
    with np.load(Path(args.volume) / "pred_tudf_256.npz") as dd:
        moge_pts = np.asarray(dd["moge_points"], dtype=np.float64)
        moge_mask = np.asarray(dd["moge_mask"], dtype=bool)
    mz = moge_pts[..., 2].copy(); mz[~moge_mask] = 0.0
    mh, mw = mz.shape
    d_small = np.asarray(Image.fromarray(d1.astype(np.float32)).resize(
        (mw, mh), Image.BILINEAR), dtype=np.float64)
    s, rel_mad, _ = register_layers_to_depth(mz[..., None], d_small)
    print(f"MoGe->solve scale {s:.4f} (rel_mad {rel_mad:.3f})   "
          f"voxel {vol.voxel_edge_m*100:.1f} cm")

    # --- ARM A: marching cubes (what every earlier number used) ---
    pts_c, _ = surface_points_canonical(vol, threshold=args.threshold)
    A = atlas_camera_to_world(moge_to_atlas_camera(pts_c, scale=s), view1)

    # --- ARM B: ray-march, crossings paired to midpoints ---
    R = args.raster
    aspect = H / float(W)
    rw, rh = R, int(round(R * aspect))
    # Intrinsics for the marching raster, from the same camera.
    fxm = spec.fx * (rw / float(W))
    fym = spec.fy * (rh / float(H))
    cxm = spec.cx * (rw / float(W))
    cym = spec.cy * (rh / float(H))
    layers, mstats = march_layers(
        vol.tudf, vol.bbox_min, vol.extent,
        fx=fxm, fy=fym, cx=cxm, cy=cym, width=rw, height=rh,
        threshold=args.threshold, max_layers=6)
    print(f"ray-march: {mstats['rays_with_surface']} rays hit, "
          f"{mstats['mean_layers_per_hit']:.2f} layers/hit, "
          f"odd-crossing {mstats['odd_crossing_fraction']*100:.1f}%")

    # Layered depths -> world points, through the same camera the rays used.
    uu, vv = np.meshgrid(np.arange(rw, dtype=np.float64),
                         np.arange(rh, dtype=np.float64))
    pts = []
    for li in range(layers.shape[2]):
        z = layers[..., li].astype(np.float64)
        m = z > 1e-6
        if not m.any():
            continue
        pz = z[m]
        px = (uu[m] - cxm) / fxm * pz
        py = (vv[m] - cym) / fym * pz
        pts.append(np.stack([px, py, pz], axis=-1))
    Bc = np.vstack(pts) if pts else np.zeros((0, 3))
    B = atlas_camera_to_world(moge_to_atlas_camera(Bc, scale=s), view1)

    print(f"\narm A (marching cubes): {len(A)} pts")
    print(f"arm B (ray-march, paired): {len(B)} pts "
          f"({len(B)/max(len(A),1)*100:.0f}% of A)")

    res = {
        "noise_floor_m": floor,
        "voxel_edge_m": vol.voxel_edge_m,
        "registration": {"scale": s, "rel_mad": rel_mad},
        "raymarch_stats": mstats,
        "points": {"marching_cubes": int(len(A)), "raymarch": int(len(B))},
        "recall_hidden": {
            "baseline_relief": _nn_stats(T_hidden, P1),
            "marching_cubes": _nn_stats(T_hidden, A),
            "raymarch_paired": _nn_stats(T_hidden, B),
        },
    }
    print("\n--- RECALL of photographed hidden truth (truth -> candidate) ---")
    for k, v in res["recall_hidden"].items():
        print(f"  {k:20} median {v['median']:7.3f} m   mean {v['mean']:7.3f}   "
              f"p90 {v['p90']:7.3f}")
    mc = res["recall_hidden"]["marching_cubes"]["median"]
    rm = res["recall_hidden"]["raymarch_paired"]["median"]
    res["bias_removed_m"] = mc - rm
    res["bias_removed_voxels"] = (mc - rm) / vol.voxel_edge_m
    print(f"\n  bias removed: {mc - rm:+.3f} m "
          f"({(mc - rm)/vol.voxel_edge_m:+.2f} voxels)")
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
