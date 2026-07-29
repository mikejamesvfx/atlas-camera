"""Sweep relief-mesh tear settings against ground truth and report the trade.

Runs every combination of tear knobs over the ray-cast fixtures, scores each
against KNOWN occlusion edges, and prints the Pareto front.

    python tools/tear_sweep.py                       # default grid
    python tools/tear_sweep.py --mef 3,6,12 --nbd 0,45
    python tools/tear_sweep.py --json out.json

IT ADOPTS NOTHING. No default is changed, no file is written unless --json is
passed. That is deliberate: the fixtures are ray-cast, with perfect hard edges,
and a setting tuned to score well on box corners can be actively wrong on real
MoGe depth, which is smooth and soft at boundaries. Treat the output as evidence
for a decision, not as the decision.

Cost is one relief-mesh build per (config x scene) — pure numpy, no GPU, no
model, a few seconds for a default grid.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def _floats(s: str) -> list:
    return [float(v) for v in str(s).split(",") if str(v).strip()]


def evaluate(scene: str, *, max_edge_factor=None, normal_edge_deg=None,
             depth_edge_rel=None):
    """Build the mesh for one scene at one config and score it."""
    import numpy as np

    import golden_scenes as g
    from atlas_camera.core.camera_math import look_at_view_matrix
    from atlas_camera.core.projection_render import render_scene
    from atlas_camera.core.relief_mesh import build_relief_mesh
    from atlas_camera.core.tear_metrics import score_tears

    depth, sid = g.SCENES[scene]()
    h, w = depth.shape
    fx = fy = (w / 2.0) / np.tan(np.radians(g.FOV_DEG) / 2.0)
    view, _world, _rot = look_at_view_matrix(
        eye=(0.0, g.CAM_Y, 0.0), target=(0.0, g.CAM_Y, -1.0), up=(0.0, 1.0, 0.0))

    kw = {}
    if max_edge_factor is not None:
        kw["max_edge_factor"] = float(max_edge_factor)
    if normal_edge_deg is not None:
        kw["normal_edge_deg"] = float(normal_edge_deg)
    if depth_edge_rel is not None:
        kw["depth_edge_rel"] = float(depth_edge_rel)

    mesh = build_relief_mesh(
        np.asarray(depth, dtype=np.float64), view_matrix=view,
        fx=fx, fy=fy, cx=w / 2.0, cy=h / 2.0,
        grid_long_edge=g.GRID, apply_sky_heuristic=False, **kw)

    meshes = [("m", np.asarray(mesh.vertices), np.asarray(mesh.faces),
               np.asarray(mesh.uvs), "primary", {})]
    _rgb, alpha, _stats = render_scene(
        meshes, {"primary": g.checker_texture()},
        view, fx, fy, w / 2.0, h / 2.0, w, h)

    return score_tears(truth_depth=depth, surface_ids=sid, coverage_alpha=alpha)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mef", default="1.5,3,6,12,24",
                    help="max_edge_factor values (lower tears more readily)")
    ap.add_argument("--nbd", default="0,30,45,60",
                    help="normal_edge_deg values (0 = the current default, off)")
    ap.add_argument("--der", default="", help="depth_edge_rel values (blank = leave alone)")
    ap.add_argument("--scenes", default="", help="comma-separated subset")
    ap.add_argument("--json", default="", help="write the full table here")
    args = ap.parse_args()

    import golden_scenes as g
    from atlas_camera.core.tear_metrics import pareto_front

    scenes = ([s.strip() for s in args.scenes.split(",") if s.strip()]
              or sorted(g.SCENES))
    grid = list(itertools.product(_floats(args.mef), _floats(args.nbd),
                                  _floats(args.der) or [None]))

    print(f"{len(grid)} configs x {len(scenes)} scenes "
          f"= {len(grid) * len(scenes)} mesh builds\n")

    rows, aggregate = [], []
    for mef, nbd, der in grid:
        label = f"mef={mef:g} nbd={nbd:g}" + (f" der={der:g}" if der is not None else "")
        per_scene = {}
        for scene in scenes:
            s = evaluate(scene, max_edge_factor=mef, normal_edge_deg=nbd,
                         depth_edge_rel=der)
            per_scene[scene] = s.to_dict()
        # Aggregate by the WORST scene, not the mean: a config that ruins one
        # kind of geometry while flattering another is not a good default, and
        # an average hides exactly that.
        worst = max(per_scene.values(), key=lambda d: d["false_tear_fraction"])

        class _Agg:
            false_tear_fraction = worst["false_tear_fraction"]
            missed_edge_fraction = max(d["missed_edge_fraction"] for d in per_scene.values())
            coverage = min(d["coverage"] for d in per_scene.values())

        aggregate.append((label, _Agg))
        rows.append({"config": {"max_edge_factor": mef, "normal_edge_deg": nbd,
                                "depth_edge_rel": der},
                     "label": label, "per_scene": per_scene,
                     "worst_false_tear": _Agg.false_tear_fraction,
                     "worst_missed_edge": _Agg.missed_edge_fraction,
                     "min_coverage": _Agg.coverage})
        print(f"  {label:32} false-tear {_Agg.false_tear_fraction * 100:6.2f}%  "
              f"missed-edge {_Agg.missed_edge_fraction * 100:6.2f}%  "
              f"coverage {_Agg.coverage * 100:6.2f}%")

    front = pareto_front(aggregate)
    print(f"\nPARETO FRONT ({len(front)} of {len(aggregate)} configs) — "
          "nothing here dominates anything else; pick by what your shot needs:\n")
    for label, agg in sorted(front, key=lambda kv: kv[1].false_tear_fraction):
        print(f"  {label:32} false-tear {agg.false_tear_fraction * 100:6.2f}%  "
              f"missed-edge {agg.missed_edge_fraction * 100:6.2f}%  "
              f"coverage {agg.coverage * 100:6.2f}%")

    print("\nNo defaults were changed. These fixtures are ray-cast with perfect hard\n"
          "edges; real MoGe depth is smooth and soft at boundaries, so validate any\n"
          "candidate on a real plate before adopting it.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"rows": rows, "pareto": [l for l, _ in front]}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
