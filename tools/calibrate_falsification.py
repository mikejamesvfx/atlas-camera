"""Measure what BROKEN geometry reads, so the falsification gates are not guesses.

``atlas_camera/core/plate_falsification.py`` ships two thresholds unset —
``MAX_SEAM_GRADIENT_RATIO`` and ``MIN_SILHOUETTE_IOU`` — because an empirical
threshold invented at the keyboard is as unfalsifiable as no threshold. This
sweep supplies them the way ``dynamic/fill_metrics.py`` got its 17.6 / 2.2 / 5.8
from the DSC_2289 plate: by deliberately breaking geometry in named ways and
recording what each metric reports.

The scene is analytic — a ground plane and two boxes in front of a known camera
— so the sweep is deterministic, needs no Blender render, and cannot drift with
a dataset. Pass ``--plate`` to take the seam statistics from a real photograph
instead of the synthetic texture; a flat synthetic plate has no gradient scale
to measure a rim against, and the ratio is correctly infinite there.

Adopts nothing. Writes no file unless ``--json PATH`` is given, following
``tools/tear_sweep.py``.

    python tools/calibrate_falsification.py
    python tools/calibrate_falsification.py --plate some.jpg --json cal.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from atlas_camera.core.plate_falsification import (  # noqa: E402
    falsification_report,
    rasterize_candidate,
    score_geometry_against_plate,
)

W, H = 320, 240
FX = FY = 260.0
CX, CY = W / 2.0, H / 2.0
VIEW = np.eye(4, dtype=np.float64)   # camera at origin, -Z forward, Y up

_BOX_FACES = np.array([
    [0, 1, 2], [0, 2, 3],  # -Z
    [4, 6, 5], [4, 7, 6],  # +Z
    [0, 4, 5], [0, 5, 1],  # -Y
    [3, 2, 6], [3, 6, 7],  # +Y
    [0, 3, 7], [0, 7, 4],  # -X
    [1, 5, 6], [1, 6, 2],  # +X
], dtype=np.int64)


def _box(center, size, *, yaw_deg: float = 0.0):
    cx, cy, cz = center
    sx, sy, sz = (s / 2.0 for s in size)
    corners = np.array([
        [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
        [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
    ], dtype=np.float64)
    t = math.radians(yaw_deg)
    rot = np.array([[math.cos(t), 0.0, math.sin(t)],
                    [0.0, 1.0, 0.0],
                    [-math.sin(t), 0.0, math.cos(t)]], dtype=np.float64)
    return corners @ rot.T + np.array([cx, cy, cz], dtype=np.float64), _BOX_FACES


def _merge(*meshes):
    verts, faces, offset = [], [], 0
    for v, f in meshes:
        verts.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.concatenate(verts), np.concatenate(faces)


def _ground(y=-1.6, extent=40.0):
    v = np.array([[-extent, y, -0.5], [extent, y, -0.5],
                  [extent, y, -extent], [-extent, y, -extent]],
                 dtype=np.float64)
    return v, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def _truth_scene():
    """Two boxes on a ground plane. The near box occludes part of the far one.

    Must be assembled from exactly the same parts as ``_perturbations()``'s
    ``truth`` row: a truth render that differs from the unperturbed candidate
    puts a floor under every score and quietly hides the real spread.
    """
    near = _box((-0.9, -0.7, -5.0), (1.6, 1.8, 1.6))
    far = _box((0.9, -0.4, -9.0), (2.4, 2.4, 2.4))
    return _merge(_ground(), near, far), near, far


def _render(mesh):
    return rasterize_candidate(
        mesh[0], mesh[1], view_matrix=VIEW, fx=FX, fy=FY, cx=CX, cy=CY,
        width=W, height=H)


def _fractal_plate(seed: int = 0):
    """1/f noise: a synthetic plate with a photograph-like gradient spectrum.

    A flat plate has no gradient scale, so every rim reads as an infinite
    seam. This is the stand-in when no real photograph is supplied.
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros((H, W), dtype=np.float64)
    for octave in range(1, 6):
        step = 2 ** octave
        coarse = rng.normal(0.0, 1.0, size=(H // step + 2, W // step + 2))
        up = np.kron(coarse, np.ones((step, step)))[:H, :W]
        acc += up / octave
    acc -= acc.min()
    acc /= max(acc.max(), 1e-9)
    return np.repeat(acc[..., None], 3, axis=2)


def _load_plate(path: Path):
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB").resize((W, H))
        return np.asarray(im, dtype=np.float64) / 255.0


def _evidence(truth_alpha, truth_depth):
    """The observed evidence a real plate would supply, derived from truth."""
    sky = ~truth_alpha
    # Only what is above the horizon counts as sky: uncovered ground beyond the
    # plane's extent is not sky, and calling it sky would falsify honest
    # geometry that legitimately extends past the fixture's ground quad.
    rows = np.arange(H)[:, None].repeat(W, axis=1)
    sky &= rows < CY
    return {
        "sky_mask": sky,
        "observed_mask": truth_alpha.copy(),
        "authorised_mask": truth_alpha.copy(),
        "reference_depth": np.where(np.isfinite(truth_depth), truth_depth,
                                    np.nanmax(truth_depth[np.isfinite(truth_depth)]) * 3.0),
    }


def _perturbations():
    """Each entry returns a candidate mesh. Names are the report's row labels."""
    near = (-0.9, -0.7, -5.0)
    far = (0.9, -0.4, -9.0)
    g = _ground()

    def build(near_c=near, far_c=far, near_s=(1.6, 1.8, 1.6),
              far_s=(2.4, 2.4, 2.4), yaw=0.0, extra=None):
        parts = [g, _box(near_c, near_s, yaw_deg=yaw), _box(far_c, far_s)]
        if extra is not None:
            parts.append(extra)
        return _merge(*parts)

    return [
        ("truth", build()),
        ("translate_0.1m", build(near_c=(-0.8, -0.7, -5.0))),
        ("translate_0.5m", build(near_c=(-0.4, -0.7, -5.0))),
        ("translate_1.0m", build(near_c=(0.1, -0.7, -5.0))),
        ("scale_10pct", build(near_s=(1.76, 1.98, 1.76))),
        ("scale_30pct", build(near_s=(2.08, 2.34, 2.08))),
        ("yaw_5deg", build(yaw=5.0)),
        ("depth_swap", build(near_c=(-0.9, -0.7, -9.0), far_c=(0.9, -0.4, -5.0))),
        ("box_in_known_sky", build(extra=_box((0.0, 3.2, -7.0), (2.0, 2.0, 2.0)))),
        ("ground_only", _merge(g)),
    ]


def _seam_rows(plate, truth_alpha):
    """Composites with a known seam and a known clean join, on one plate."""
    rows = []
    clean = plate.copy()
    rows.append(("seam_none", clean))

    hard = plate.copy()
    hard[truth_alpha] = np.clip(plate[truth_alpha] + 0.45, 0.0, 1.0)
    rows.append(("seam_exposure_offset", hard))

    shifted = plate.copy()
    rolled = np.roll(plate, 37, axis=1)
    shifted[truth_alpha] = rolled[truth_alpha]
    rows.append(("seam_wrong_content", shifted))

    smear = plate.copy()
    flat = float(plate[truth_alpha].mean())
    smear[truth_alpha] = flat
    rows.append(("seam_flat_smear", smear))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--plate", type=Path, default=None,
                    help="photograph to take the seam gradient statistics from")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the table here; nothing is written without it")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    truth_mesh, _near, _far = _truth_scene()
    truth_alpha, truth_depth = _render(truth_mesh)
    evidence = _evidence(truth_alpha, truth_depth)

    plate = _load_plate(args.plate) if args.plate else _fractal_plate(args.seed)
    plate_source = str(args.plate) if args.plate else f"fractal_noise(seed={args.seed})"

    geometry_rows = []
    baseline_kwargs = dict(alpha=truth_alpha, render_depth=truth_depth, **evidence)
    for name, mesh in _perturbations():
        alpha, depth = _render(mesh)
        if not alpha.any():
            geometry_rows.append({"case": name, "error": "nothing rasterized"})
            continue
        scored = score_geometry_against_plate(
            alpha=alpha, render_depth=depth, seed=args.seed, **evidence)
        geometry_rows.append({
            "case": name,
            "alpha_px": scored["alpha_px"],
            "sky_violation": scored["sky_violation"]["value"],
            "containment": scored["containment"]["value"],
            "spill_px": scored["containment"]["spill_px"],
            "silhouette_iou": scored["silhouette_iou"]["value"],
            "depth_order_agreement": scored["depth_order_agreement"]["value"],
        })

    # Depth-order robustness: the reference is monocular in practice, so record
    # what scale+shift and noise actually cost before any gate reads it.
    finite = np.isfinite(truth_depth)
    depth_rows = []
    rng = np.random.default_rng(args.seed)
    for name, ref in (
        ("reference_exact", evidence["reference_depth"]),
        ("reference_scale_shift", 6.0 * evidence["reference_depth"] + 90.0),
        ("reference_noise_0.5m", evidence["reference_depth"] + rng.normal(0, 0.5, (H, W))),
        ("reference_noise_2.0m", evidence["reference_depth"] + rng.normal(0, 2.0, (H, W))),
        ("reference_inverted", -evidence["reference_depth"]),
    ):
        d = score_geometry_against_plate(
            alpha=truth_alpha, render_depth=truth_depth, reference_depth=ref,
            seed=args.seed)["depth_order_agreement"]
        depth_rows.append({"case": name, "value": d["value"],
                           "n_pairs": d.get("n_pairs")})

    seam_rows = []
    for name, composite in _seam_rows(plate, truth_alpha):
        s = score_geometry_against_plate(
            alpha=truth_alpha, plate=plate, composite=composite,
        )["seam_gradient_ratio"]
        seam_rows.append({"case": name, "ratio": s["value"],
                          "rim_gradient": s["rim_gradient"],
                          "plate_rim_gradient": s["plate_rim_gradient"],
                          "plate_gradient": s["plate_gradient"]})

    # A worked example of the report contract: a broken candidate beside the
    # do-nothing render, which is how every real caller must publish.
    broken_alpha, broken_depth = _render(dict(_perturbations())["translate_1.0m"])
    example = falsification_report(
        candidate=dict(alpha=broken_alpha, render_depth=broken_depth, **evidence),
        baseline=baseline_kwargs,
    )

    payload = {
        "scene": {"width": W, "height": H, "fx": FX, "fy": FY,
                  "cx": CX, "cy": CY, "truth_alpha_px": int(truth_alpha.sum()),
                  "finite_depth_px": int(finite.sum())},
        "plate_source": plate_source,
        "geometry_perturbations": geometry_rows,
        "depth_reference_perturbations": depth_rows,
        "seam_perturbations": seam_rows,
        "example_report_beats_baseline": example.beats_baseline,
        "example_report_deltas": example.deltas,
    }

    def _fmt(v):
        return "n/a" if v is None else (f"{v:.4f}" if abs(v) < 1e6 else f"{v:.3e}")

    print(f"plate: {plate_source}   truth alpha {int(truth_alpha.sum())} px\n")
    print(f"{'case':<22}{'sky':>9}{'contain':>10}{'spill_px':>10}{'iou':>9}{'depth':>9}")
    for r in geometry_rows:
        if "error" in r:
            print(f"{r['case']:<22}{r['error']:>47}")
            continue
        print(f"{r['case']:<22}{_fmt(r['sky_violation']):>9}"
              f"{_fmt(r['containment']):>10}{r['spill_px']:>10}"
              f"{_fmt(r['silhouette_iou']):>9}{_fmt(r['depth_order_agreement']):>9}")
    print(f"\n{'depth reference':<26}{'agreement':>11}{'pairs':>9}")
    for r in depth_rows:
        print(f"{r['case']:<26}{_fmt(r['value']):>11}{str(r['n_pairs']):>9}")
    print(f"\n{'seam case':<24}{'ratio':>11}{'rim':>10}{'plate_rim':>11}")
    for r in seam_rows:
        print(f"{r['case']:<24}{_fmt(r['ratio']):>11}"
              f"{_fmt(r['rim_gradient']):>10}{_fmt(r['plate_rim_gradient']):>11}")
    print(f"\nexample report (translate_1.0m vs truth) beats baseline: "
          f"{example.beats_baseline}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
