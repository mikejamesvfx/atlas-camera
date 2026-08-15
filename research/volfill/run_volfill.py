"""Run VolFill inference and dump everything the Atlas side needs (RESEARCH).

Runs INSIDE the isolated venv (`research/volfill/.venv`) — it imports torch and
the VolFill package, and must never be run from the Atlas or ComfyUI env. The
Atlas-side consumer (`roundtrip_eval.py`) runs in the normal Atlas env and only
reads the .npz this writes. Two processes, two environments, no dependency
collision.

Beyond VolFill's own `pred_tudf_*.npz` + `metadata.json`, this also dumps:

  visible_tudf  — the TUDF VolFill built from the MoGe visible points. This is the
                  definition of "what was observable", so it splits predicted
                  geometry into VISIBLE vs OCCLUDED *without* needing external
                  visibility truth. Occluded-only metrics come from this mask.
  moge_points   — (H, W, 3) MoGe camera-frame metric pointmap + validity mask,
                  needed to register MoGe scale against the Atlas depth map.

Usage (from research/volfill):
    .venv/Scripts/python.exe run_volfill.py --image path/to.jpg --out out/<name> \
        [--steps 50] [--cfg 3.0] [--seed 0] [--max-depth 40]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
REPO = Path(__file__).resolve().parent / "repos" / "volfill"
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-repo", default="TuanNgo/VolFill")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Surface threshold in voxel units (visualize.py semantics).")
    ap.add_argument("--max-depth", type=float, default=None,
                    help="Clip MoGe points beyond this depth (metres) before the "
                         "bbox fit — the depth-slab mitigation for the 256^3 "
                         "resolution gate on deep exterior plates.")
    ap.add_argument("--fov-x-deg", type=float, default=None,
                    help="MEASURED horizontal FOV. Without it MoGe estimates the "
                         "camera itself, and the bbox, scale and every derived "
                         "surface inherit that guess — measured 15%% low on a "
                         "plate whose true focal was known from EXIF.")
    ap.add_argument("--raw", default=None,
                    help="RAW file to take focal + ORIENTED sensor width from, "
                         "instead of passing --fov-x-deg by hand.")
    ap.add_argument("--max-side", type=int, default=2048,
                    help="Downscale the long edge before MoGe (big plates OOM).")
    args = ap.parse_args()

    from volfill.amodal.inference_latent_visible import (  # noqa: E402
        LatentTUDFVisibleInference,
    )

    # MEASURED intrinsics beat an estimate. MoGe accepts a known horizontal FOV;
    # without one it predicts its own camera and everything downstream — the
    # canonical bbox, the metric scale, the extracted surface — is built on that
    # guess. Sensor width must be ORIENTED to the frame: a portrait plate needs
    # the sensor's short edge, or fx lands 34% out.
    fov_x = args.fov_x_deg
    if fov_x is None and args.raw:
        import math
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from atlas_camera.raw.pipeline import import_raw
        r = import_raw(args.raw, undistort=False, half_size=True)
        if r.focal_length_mm and r.sensor_width_mm:
            fov_x = math.degrees(
                2.0 * math.atan(r.sensor_width_mm / (2.0 * r.focal_length_mm)))
            print(f"[intrinsics] {Path(args.raw).name}: focal {r.focal_length_mm}mm "
                  f"sensor {r.sensor_width_mm}mm -> fov_x {fov_x:.2f} deg")
        else:
            print("[intrinsics] RAW carried no focal/sensor — MoGe will estimate.")
    if fov_x is None:
        print("[intrinsics] WARNING: no measured FOV supplied; MoGe estimates "
              "the camera and the volume inherits that guess.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(args.image).convert("RGB")
    if args.max_side and max(img.size) > args.max_side:
        s = args.max_side / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    infer = LatentTUDFVisibleInference.from_pretrained(
        args.hf_repo, steps=args.steps, cfg_strength=args.cfg,
    )
    t_load = time.perf_counter() - t0
    vram_load = torch.cuda.max_memory_allocated() / 2**30

    # --- Capture the EXACT visible volume that conditions the prediction.
    #
    # Two traps in `__call__`, both found the hard way:
    #   1. it hardcodes `max_depth=None`, so a depth-slab clip passed to a
    #      separate call never reaches the prediction; and
    #   2. it runs MoGe internally at `max_size=518`, so a side-call at full
    #      resolution yields a DIFFERENT pointmap -> different bbox -> a visible
    #      TUDF in a different canonical frame from the predicted one.
    # Recomputing the visible volume ourselves is therefore wrong on both
    # counts. Wrap the method instead: force max_depth through, and keep what
    # the real run produced.
    captured: dict[str, Any] = {}
    orig_visible = infer._compute_visible_tudf_online
    orig_moge_infer = infer.moge_for_visible.infer

    def _capture_moge(image_t, **kw):
        if fov_x is not None:
            kw["fov_x"] = fov_x          # measured camera, not MoGe's guess
        out = orig_moge_infer(image_t, **kw)
        # The LAST call inside __call__ is the conditioning one; keep overwriting.
        captured["moge_points"] = out["points"].squeeze(0).float().cpu().numpy()
        captured["moge_mask"] = out["mask"].squeeze(0).cpu().numpy().astype(bool)
        if "intrinsics" in out:
            captured["moge_intrinsics"] = out["intrinsics"].float().cpu().numpy()
        return out

    def _capture_visible(image, **kw):
        if args.max_depth is not None:
            kw["max_depth"] = args.max_depth      # override __call__'s None
        t1 = time.perf_counter()
        vis, bmin, ext = orig_visible(image, **kw)
        captured["visible_tudf"] = np.asarray(vis, dtype=np.float32)
        captured["t_visible"] = time.perf_counter() - t1
        return vis, bmin, ext

    infer.moge_for_visible.infer = _capture_moge
    infer._compute_visible_tudf_online = _capture_visible

    torch.cuda.reset_peak_memory_stats()
    t2 = time.perf_counter()
    result = infer(img)
    torch.cuda.synchronize()
    t_infer = time.perf_counter() - t2
    vram_peak = torch.cuda.max_memory_allocated() / 2**30

    vis_tudf = captured["visible_tudf"]
    moge_points = captured["moge_points"]
    moge_mask = captured["moge_mask"]
    t_visible = captured.get("t_visible", 0.0)
    t_infer = max(t_infer - t_visible, 0.0)   # report sampling net of MoGe

    trunc = infer.truncation_voxels
    tudf_norm = result["tudf"].squeeze().numpy()
    tudf_raw = np.clip((tudf_norm + 1.0) * trunc / 2.0, 0.0, trunc).astype(np.float32)

    np.savez_compressed(
        out_dir / "pred_tudf_256.npz",
        tudf=tudf_raw,
        visible_tudf=vis_tudf.astype(np.float32),
        moge_points=moge_points.astype(np.float32),
        moge_mask=moge_mask,
    )

    res = int(tudf_raw.shape[0])
    bbox_min = np.asarray(result["bbox_min"], dtype=np.float64)
    extent = np.asarray(result["extent"], dtype=np.float64)
    meta = {
        "representation": "tudf",
        "truncation_voxels": float(trunc),
        "field_range": [0.0, float(trunc)],
        "field_units": "voxel_units",
        "bbox_min": bbox_min.tolist(),
        "extent_xyz": extent.tolist(),
        "pred_resolution": [res, res, res],
        # --- Atlas-side additions ---
        "source_image": str(Path(args.image).resolve()),
        "image_size": [img.width, img.height],
        "voxel_edge_m": float(np.max(extent) / res),
        "surface_threshold_voxels": args.threshold,
        "steps": args.steps,
        "cfg_strength": args.cfg,
        "seed": args.seed,
        "max_depth": args.max_depth,
        "fov_x_deg": fov_x,
        "intrinsics_source": ("measured_raw" if args.raw else
                              ("measured_arg" if args.fov_x_deg else "moge_estimate")),
        "timings_s": {
            "model_load": round(t_load, 3),
            "moge_visible": round(t_visible, 3),
            "sampling": round(t_infer, 3),
        },
        "vram_gib": {"after_load": round(vram_load, 3), "peak": round(vram_peak, 3)},
        "moge_intrinsics_norm": (captured["moge_intrinsics"].tolist()
                                 if "moge_intrinsics" in captured else None),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "research_only": True,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    img.save(out_dir / "image.jpg", quality=95)

    n_surf = int((tudf_raw <= args.threshold).sum())
    print(f"[volfill] {out_dir.name}: {n_surf} surface voxels, "
          f"voxel {meta['voxel_edge_m']*100:.1f} cm, "
          f"load {t_load:.1f}s moge {t_visible:.1f}s sample {t_infer:.1f}s, "
          f"peak {vram_peak:.1f} GiB")


if __name__ == "__main__":
    main()
