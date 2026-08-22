"""Measure the gravity-locked ground hypothesis against the shipping estimator.

Run by hand from a repo checkout; not imported by the package. Companion to
``docs/development/gravity-locked-ground-experiment.md`` and
``atlas_camera/core/ground_consensus.py``.

    PYTHONPATH=. python tools/ground_consensus_probe.py \
        --raw C:/Users/miike/Pictures/atlas_raws/atlas_raws/DSC_2245.NEF \
        --contacts contacts.json

The structure mirrors the graph under test
(``research/atlas_hero_02_photo_to_editable_scene_workflow_ground.json``), which
runs TWO depth models and chains two independent ground estimates: the camera
height is measured from Depth-Anything-V2-Metric-Outdoor, while the geometry is
built on MoGe, and ``estimate_ground_scale`` then rescales the MoGe world about
the camera so MoGe's own ground lands on Y=0 -- anchored to the V2 height. The
ratio between the two models is never measured anywhere in the pipeline, so a
disagreement rescales the whole world while the orientation stays perfect. That
is why stage A exists and why it runs first.

Stages, each independently skippable, with depth cached to disk between runs:

    A  cross-model ground disagreement (V2 vs MoGe)
    B  the shipping estimator, instrumented, per depth map
    C  the candidate estimator over the weighting x estimator x mask grid
    D  contact-point validation against hand-marked image positions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from atlas_camera.core.ground_consensus import (  # noqa: E402
    ESTIMATORS,
    WEIGHTINGS,
    estimate_ground_height_consensus,
)
from atlas_camera.core.relief_mesh import estimate_ground_scale  # noqa: E402
from atlas_camera.core.solver import (  # noqa: E402
    estimate_ground_height_from_depth,
    solve_still_image_learned,
)

V2_OUTDOOR = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
MOGE = "Ruicheng/moge-2-vitl-normal"

# The graph's own near-field ROI candidates. Fractions, not doctrine -- the
# point of the sweep is to find out whether the ROI matters at all.
ROIS = {
    "full": None,
    "bottom45_centre70": (0.55, 1.00, 0.15, 0.85),
    "bottom30_centre50": (0.70, 1.00, 0.25, 0.75),
}


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def prepare_plate(raw_path: Path, out_dir: Path, long_side: int) -> dict:
    """Decode the NEF exactly as the graph's AtlasLoadRAW does, then downscale.

    Intrinsics come from EXIF, so focal length is not a free variable in any
    comparison below -- the one thing this plate gives us that an AI-generated
    image never could.
    """
    from PIL import Image

    from atlas_camera.raw.pipeline import import_raw

    res = import_raw(str(raw_path), undistort=True, half_size=False)
    disp = np.asarray(res.display_srgb)
    full_h, full_w = disp.shape[:2]

    scale = float(long_side) / float(max(full_w, full_h))
    if scale < 1.0:
        new_w, new_h = int(round(full_w * scale)), int(round(full_h * scale))
        img = Image.fromarray((np.clip(disp, 0.0, 1.0) * 255.0).astype(np.uint8))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    else:
        scale = 1.0
        new_w, new_h = full_w, full_h
        img = Image.fromarray((np.clip(disp, 0.0, 1.0) * 255.0).astype(np.uint8))

    out_dir.mkdir(parents=True, exist_ok=True)
    plate = out_dir / f"{raw_path.stem}_{new_w}x{new_h}.png"
    if not plate.exists():
        img.save(plate)

    return {
        "path": plate,
        "width": new_w,
        "height": new_h,
        "full_width": full_w,
        "full_height": full_h,
        "scale": scale,
        "focal_length_mm": res.focal_length_mm,
        "sensor_width_mm": res.sensor_width_mm,
        "camera_model": res.camera_model,
        "lens_model": res.lens_model,
        "undistort": res.undistort_status,
        "warnings": list(res.warnings),
    }


def cached_depth(plate: Path, model_id: str, cache_dir: Path, tag: str):
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz = cache_dir / f"{plate.stem}_{tag}.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=False)
        return {"depth": z["depth"], "is_metric": bool(z["is_metric"]),
                "cached": True, "model_id": model_id}

    from atlas_camera.inference.depth_estimator import estimate_depth

    res = estimate_depth(str(plate), model_id=model_id)
    depth = np.asarray(res.depth, dtype=np.float32)
    np.savez_compressed(npz, depth=depth, is_metric=np.array(bool(res.is_metric)))
    return {"depth": depth, "is_metric": bool(res.is_metric),
            "cached": False, "model_id": model_id}


def concept_mask(plate: Path, cache_dir: Path, concepts: str, tag: str):
    """SAM3 concept mask, cached.

    ``sky`` is the mask the graph already computes at node 17 and wires to
    nothing. ``car`` etc. is the one that matters here: a car roof is a flat
    horizontal surface at the wrong height, so ``|n_y| > 0.90`` cannot reject
    it and the shipping estimator has no mask input to reject it with.
    """
    npz = cache_dir / f"{plate.stem}_sam3_{tag}.npz"
    if npz.exists():
        return np.load(npz)["mask"].astype(bool)
    try:
        from PIL import Image

        from atlas_camera.inference.sam3_segmenter import sam3_concept_mask
    except Exception as exc:
        print(f"       sam3 {tag}: unavailable ({exc})")
        return None
    try:
        img = Image.open(plate).convert("RGB")
        mask, matched, coverage = sam3_concept_mask(
            img, concepts=concepts, confidence_threshold=0.5)
        mask = np.asarray(mask).astype(bool)
        print(f"       sam3 {tag}: matched={matched} coverage={coverage:.4f}")
    except Exception as exc:
        print(f"       sam3 {tag}: failed ({exc})")
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, mask=mask)
    return mask


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def _rotation_and_horizon(solve, height: int):
    cam = solve.camera
    vm = np.asarray(cam.extrinsics.camera_view_matrix, dtype=np.float64)
    rotation = vm[:3, :3]                      # world -> cam
    intr = cam.intrinsics
    fx, fy = float(intr.fx_px), float(intr.fy_px)
    cx, cy = float(intr.cx_px), float(intr.cy_px)
    # Same analytic horizon the learned solve uses.
    r = rotation
    horizon = cy - fy * float(r[1, 2]) / float(r[1, 1]) if abs(r[1, 1]) > 1e-9 \
        else height * 0.45
    return vm, rotation, fx, fy, cx, cy, float(horizon)


def stage_a(depths: dict, solve, height: int) -> dict:
    """Cross-model ground disagreement. The suspected root cause."""
    vm, rot, fx, fy, cx, cy, horizon = _rotation_and_horizon(solve, height)
    out: dict = {"horizon_y": horizon}
    for tag, d in depths.items():
        r = estimate_ground_height_from_depth(
            d["depth"], rotation=rot, fx=fx, fy=fy, cx=cx, cy=cy,
            horizon_y=horizon)
        out[tag] = {
            "camera_height": r.get("camera_height"),
            "rejected_height": r.get("rejected_height"),
            "confidence": r.get("confidence"),
            "plane_y": r.get("plane_y"),
            "ground_pixels": r.get("ground_pixels"),
            "is_metric": d["is_metric"],
        }
    h_v2 = out.get("v2", {}).get("camera_height")
    h_moge = out.get("moge", {}).get("camera_height")
    if h_v2 and h_moge:
        out["ratio_v2_over_moge"] = float(h_v2) / float(h_moge)

    # The scale the chain actually applies to the MoGe world, computed the way
    # the derive nodes compute it (via _ground_scale_cached -> this function).
    if "moge" in depths and h_v2:
        vm_adopted = np.array(vm, dtype=np.float64)
        scale, info = estimate_ground_scale(
            depths["moge"]["depth"], view_matrix=vm_adopted,
            fx=fx, fy=fy, cx=cx, cy=cy, horizon_y=horizon)
        out["moge_ground_scale"] = {"scale": float(scale), "info": info}
    return out


def stage_b(depths: dict, solve, height: int) -> dict:
    """The shipping estimator, instrumented: where does its support live?"""
    _, rot, fx, fy, cx, cy, horizon = _rotation_and_horizon(solve, height)
    out: dict = {}
    for tag, d in depths.items():
        depth = np.asarray(d["depth"], dtype=np.float64)
        r = estimate_ground_height_from_depth(
            depth, rotation=rot, fx=fx, fy=fy, cx=cx, cy=cy, horizon_y=horizon)
        mask = np.asarray(r["ground_mask"], dtype=bool)
        entry = {
            "camera_height": r.get("camera_height"),
            "rejected_height": r.get("rejected_height"),
            "confidence": r.get("confidence"),
            "plane_y": r.get("plane_y"),
            "plane_tolerance": r.get("plane_tolerance"),
            "ground_pixels": int(mask.sum()),
            # The ground normal is +Y by construction in every single-image
            # path -- recorded so the report states it as a fact, not a claim.
            "plane_normal": [0.0, 1.0, 0.0],
            "plane_normal_source": "hard-coded world +Y (never fitted)",
        }
        if mask.any():
            dm = depth[mask]
            entry["support_depth"] = {
                "min": float(dm.min()), "p25": float(np.percentile(dm, 25)),
                "median": float(np.median(dm)),
                "p75": float(np.percentile(dm, 75)), "max": float(dm.max()),
            }
            rows = np.nonzero(mask)[0]
            entry["support_rows"] = {
                "lower_third": float((rows > height * 2 / 3).mean()),
                "middle_third": float(((rows > height / 3)
                                       & (rows <= height * 2 / 3)).mean()),
                "upper_third": float((rows <= height / 3).mean()),
            }
            entry["residual_profile"] = residual_profile(
                depth, rot, fx, fy, cx, cy, horizon,
                plane_y=float(r["plane_y"]))
        out[tag] = entry
    return out


def residual_profile(depth, rot, fx, fy, cx, cy, horizon, *, plane_y,
                     bands=((0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9))):
    """Signed vertical residual of the CANDIDATE ground against the fitted plane,
    binned by range.

    The accepted-inlier statistics cannot show this: a region that disagrees
    with the plane by more than the tolerance is simply absent from them, so a
    plane that misses the near road entirely still reports clean support. This
    reads the residual over every candidate pixel, including the ones the fit
    threw away, which is the only way to see WHERE the plane stops matching the
    road.
    """
    from atlas_camera.core.depth_geometry import back_project_normals

    vm = np.eye(4, dtype=np.float64)
    vm[:3, :3] = rot
    bp = back_project_normals(np.asarray(depth, dtype=np.float64),
                              view_matrix=vm, fx=fx, fy=fy, cx=cx, cy=cy)
    cand = (bp.valid_normal & (bp.vv > horizon)
            & (np.abs(bp.normals[..., 1]) > 0.90))
    if not cand.any():
        return {}
    d = np.asarray(depth, dtype=np.float64)[cand]
    resid = bp.pts_world[..., 1][cand] - float(plane_y)
    out = {}
    for lo, hi in bands:
        sel = (d >= lo) & (d < hi)
        n = int(sel.sum())
        if not n:
            continue
        rr = resid[sel]
        out[f"{lo:g}-{hi:g}m" if hi < 1e8 else f"{lo:g}m+"] = {
            "count": n,
            "median_residual_m": float(np.median(rr)),
            "mean_abs_residual_m": float(np.abs(rr).mean()),
            "p90_abs_residual_m": float(np.percentile(np.abs(rr), 90)),
        }
    return out


def stage_e(refs: list, solve, height: int) -> dict:
    """Independent metric scale from known-size vertical objects (tier 1).

    A parked car of a known model is the arbiter this plate happens to provide.
    ``metric_height_from_reference`` recovers camera height from one vertical
    object without assuming an eye height at all, so it can adjudicate between
    two depth models that disagree and are both confident.
    """
    from atlas_camera.core.solver import metric_height_from_reference

    _, rot, fx, fy, cx, cy, _ = _rotation_and_horizon(solve, height)
    out = {}
    for ref in refs:
        try:
            res = metric_height_from_reference(
                tuple(ref["base"]), tuple(ref["top"]), float(ref["height_m"]),
                rotation=rot, fx=fx, fy=fy, cx=cx, cy=cy)
        except Exception as exc:
            out[ref["name"]] = {"error": str(exc)}
            continue
        out[ref["name"]] = {
            "camera_height": res.get("camera_height"),
            "object_height_m": float(ref["height_m"]),
            "base": ref["base"], "top": ref["top"],
            "detail": {k: v for k, v in res.items() if k != "camera_height"},
        }
    return out


def stage_c(depths: dict, solve, height: int, exclude=None) -> dict:
    """The candidate estimator across the full knob grid."""
    _, rot, fx, fy, cx, cy, horizon = _rotation_and_horizon(solve, height)
    out: dict = {}
    for tag, d in depths.items():
        per_depth: dict = {}
        for roi_name, roi in ROIS.items():
            for mask_name, mask in (("none", None), ("sam3_sky_clutter", exclude)):
                if mask_name == "sam3_sky_clutter" and mask is None:
                    continue
                res = estimate_ground_height_consensus(
                    d["depth"], rotation=rot, fx=fx, fy=fy, cx=cx, cy=cy,
                    horizon_y=horizon, exclude_mask=mask, roi=roi,
                    weighting="uniform", estimator="mad_median",
                    height_prior=(1.0, 2.2))
                cell = {
                    "candidates": res.candidates,
                    "accepted": res.accepted,
                    "tolerance": res.tolerance,
                    "confidence": res.confidence,
                    "confidences": res.confidences,
                    "estimators_uniform": res.estimators,
                    "band_support": res.band_support,
                    "distribution": res.distribution,
                    "normal_probe": res.normal_probe,
                    "rejections": res.rejections,
                    "notes": res.notes,
                }
                # Weighting only changes the votes' authority, so one extra
                # pass per weighting gives the whole table.
                by_weight = {}
                for wt in WEIGHTINGS:
                    if wt == "uniform":
                        by_weight[wt] = res.estimators
                        continue
                    rw = estimate_ground_height_consensus(
                        d["depth"], rotation=rot, fx=fx, fy=fy, cx=cx, cy=cy,
                        horizon_y=horizon, exclude_mask=mask, roi=roi,
                        weighting=wt, estimator="mad_median")
                    by_weight[wt] = rw.estimators
                cell["estimators_by_weighting"] = by_weight
                per_depth[f"{roi_name}|mask={mask_name}"] = cell
        out[tag] = per_depth
    return out


def stage_d(contacts: list, solve, height: int, heights: dict) -> dict:
    """Do the candidate planes pass under the things touching the ground?

    Each contact is ``{"name": ..., "u": px, "v": px}`` in PLATE pixels: the
    bottom of a tyre, a foot, a kerb line. The camera ray through that pixel is
    intersected with each candidate ground plane and the miss reported. No
    automatic detection -- a handful of hand-marked points is the honest way to
    judge a height on a plate with no ground truth.
    """
    _, rot, fx, fy, cx, cy, _ = _rotation_and_horizon(solve, height)
    c2w = np.asarray(rot, dtype=np.float64).T
    out: dict = {}
    for c in contacts:
        u, v = float(c["u"]), float(c["v"])
        ray = np.array([(u - cx) / fx, -(v - cy) / fy, -1.0])
        world = c2w @ ray
        entry = {}
        for label, h in heights.items():
            if h is None or not np.isfinite(h):
                entry[label] = {"error": "no height"}
                continue
            # Camera at origin, ground at Y = -h.
            if world[1] >= -1e-9:
                entry[label] = {"error": "ray never descends to the ground"}
                continue
            t = (-float(h)) / float(world[1])
            p = world * t
            entry[label] = {
                "range_m": float(np.linalg.norm(p)),
                "ground_depth_m": float(-p[2]),
                "world": [float(x) for x in p],
            }
        # A contact point IS on the ground by definition, so any pair of planes
        # differing in height puts the same pixel at two different ranges --
        # that spread is the measurement.
        ranges = [e["range_m"] for e in entry.values() if "range_m" in e]
        if len(ranges) > 1:
            entry["range_spread_m"] = float(max(ranges) - min(ranges))
        out[c.get("name", f"{u:.0f},{v:.0f}")] = entry
    return out


# --------------------------------------------------------------------------

def _solve(plate: dict):
    """Reproduce the graph's AtlasLearnedSolveFromImage node exactly."""
    return solve_still_image_learned(
        str(plate["path"]),
        camera_height="auto",
        sensor_width_mm=float(plate["sensor_width_mm"] or 36.0),
        focal_length_mm_hint=float(plate["focal_length_mm"] or 0.0) or None,
        weights="pinhole",
        depth_model=V2_OUTDOOR,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="camera RAW plate (NEF/CR2/...)")
    ap.add_argument("--contacts", default="", help="JSON list of {name,u,v}")
    ap.add_argument("--references", default="",
                    help="JSON list of {name,base:[u,v],top:[u,v],height_m}")
    ap.add_argument("--out", default="docs/dev/ground_consensus", help="output dir")
    ap.add_argument("--long-side", type=int, default=2048)
    ap.add_argument("--stages", default="ABCD")
    args = ap.parse_args()

    out_dir = Path(args.out)
    cache = out_dir / "cache"
    plate = prepare_plate(Path(args.raw), out_dir, args.long_side)
    print(f"plate  {plate['path'].name}  {plate['width']}x{plate['height']}"
          f"  (from {plate['full_width']}x{plate['full_height']})")
    print(f"lens   {plate['camera_model']} {plate['lens_model']}"
          f"  f={plate['focal_length_mm']}mm  sensor={plate['sensor_width_mm']}mm"
          f"  undistort={plate['undistort']}")

    solve_cache = cache / f"{plate['path'].stem}_solve.json"
    if solve_cache.exists():
        from atlas_camera.core.schema import AtlasSolve
        solve = AtlasSolve.from_json(solve_cache.read_text())
        print("solve  (cached)")
    else:
        solve = _solve(plate)
        cache.mkdir(parents=True, exist_ok=True)
        solve_cache.write_text(solve.to_json())

    depths = {
        "v2": cached_depth(plate["path"], V2_OUTDOOR, cache, "v2"),
        "moge": cached_depth(plate["path"], MOGE, cache, "moge"),
    }
    for tag, d in depths.items():
        print(f"depth  {tag:5s} {d['model_id']}  metric={d['is_metric']}"
              f"  cached={d['cached']}  shape={np.asarray(d['depth']).shape}")

    sky = concept_mask(plate["path"], cache, "sky", "sky")
    clutter = concept_mask(
        plate["path"], cache,
        "car, truck, bus, van, person, bicycle", "clutter")
    parts = [m for m in (sky, clutter) if m is not None]
    exclude = None
    if parts:
        exclude = parts[0].copy()
        for m in parts[1:]:
            exclude |= m
    print(f"mask   sky={sky is not None} clutter={clutter is not None}"
          f" combined={None if exclude is None else float(exclude.mean())}")

    report: dict = {"plate": {k: str(v) for k, v in plate.items()}}
    h = int(plate["height"])

    if "A" in args.stages:
        report["stage_a_cross_model"] = stage_a(depths, solve, h)
    if "B" in args.stages:
        report["stage_b_shipping"] = stage_b(depths, solve, h)
    if "C" in args.stages:
        report["stage_c_candidate"] = stage_c(depths, solve, h, exclude=exclude)
    if "E" in args.stages and args.references:
        refs = json.loads(Path(args.references).read_text())
        report["stage_e_reference_scale"] = stage_e(refs, solve, h)
    if "D" in args.stages and args.contacts:
        contacts = json.loads(Path(args.contacts).read_text())
        heights = {}
        a = report.get("stage_a_cross_model", {})
        for tag in ("v2", "moge"):
            heights[f"shipping_{tag}"] = a.get(tag, {}).get("camera_height")
        c = report.get("stage_c_candidate", {})
        for tag, cells in c.items():
            for cell_name, cell in cells.items():
                est = cell.get("estimators_uniform", {}).get("mad_median")
                heights[f"candidate_{tag}_{cell_name}"] = est
        report["stage_d_contacts"] = stage_d(contacts, solve, h, heights)

    dest = out_dir / "report.json"
    dest.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {dest}")

    ref = report.get("stage_e_reference_scale")
    if ref:
        print("\nSTAGE E  independent metric scale (known-size objects)")
        for name, rr in ref.items():
            print(f"  {name:24s} h={rr.get('camera_height')}"
                  f"  {rr.get('error', '')}")

    a = report.get("stage_a_cross_model")
    if a:
        print("\nSTAGE A  cross-model ground disagreement")
        for tag in ("v2", "moge"):
            e = a.get(tag, {})
            print(f"  {tag:5s} h={e.get('camera_height')}"
                  f"  conf={e.get('confidence')}"
                  f"  rejected={e.get('rejected_height')}")
        if "ratio_v2_over_moge" in a:
            print(f"  ratio v2/moge = {a['ratio_v2_over_moge']:.4f}")
        if "moge_ground_scale" in a:
            print(f"  MoGe world rescaled by {a['moge_ground_scale']['scale']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
