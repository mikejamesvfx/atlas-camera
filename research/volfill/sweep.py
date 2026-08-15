"""Step-count and extent-reduction sweeps for VolFill (RESEARCH).

Two questions, one driver:

STEPS  — VolFill is generative, so sampling cost is a dial. Sweep it against the
         quality of what it produces. Reference points from the existing Atlas
         slate (docs/dev/occlusion_arms_2026-08-14): Wan VACE CausVid 4-step is
         40 s cold / 25.5 s warm; WT-DiT ~17 s at 20 steps.

EXTENT — the 256^3 grid gives 256 voxels across the LONGEST axis, whatever the
         units, so voxel size tracks scene extent. Uniform rescaling of the
         Atlas solve cannot help (the bbox is fit to the data, and Atlas's scale
         is applied on the return leg via `s`). Reducing EXTENT is the only
         lever: `--max-depth` slabbing, or cropping to a hole ROI.

Run from research/volfill with the isolated venv:
    .venv/Scripts/python.exe sweep.py --image <plate> --mode steps
    .venv/Scripts/python.exe sweep.py --image <plate> --mode extent
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE / ".venv" / "Scripts" / "python.exe"

STEP_LADDER = [50, 25, 16, 8, 4]
DEPTH_LADDER = [None, 80.0, 40.0, 20.0, 10.0]


def run(image: str, out: Path, steps: int, max_depth: float | None,
        seed: int = 0) -> dict:
    cmd = [str(PY), str(HERE / "run_volfill.py"), "--image", image,
           "--out", str(out), "--steps", str(steps), "--seed", str(seed)]
    if max_depth is not None:
        cmd += ["--max-depth", str(max_depth)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()[-500:]}
    return json.loads((out / "metadata.json").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--mode", choices=["steps", "extent"], default="steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    name = args.name or Path(args.image).stem
    rows = []

    if args.mode == "steps":
        # Fixed seed across the ladder: the only variable is step count, so any
        # difference is the sampler's, not the noise draw's.
        for s in STEP_LADDER:
            out = HERE / "out" / f"sweep_{name}_s{s}"
            meta = run(args.image, out, s, None, seed=args.seed)
            if "error" in meta:
                print(f"steps={s:>3}  FAILED: {meta['error'][:120]}")
                rows.append({"steps": s, "error": meta["error"]})
                continue
            t = meta["timings_s"]
            rows.append({"steps": s, "sampling_s": t["sampling"],
                         "moge_s": t["moge_visible"],
                         "peak_gib": meta["vram_gib"]["peak"],
                         "voxel_edge_m": meta["voxel_edge_m"]})
            print(f"steps={s:>3}  sample {t['sampling']:6.2f}s  "
                  f"peak {meta['vram_gib']['peak']:.1f} GiB")
    else:
        for d in DEPTH_LADDER:
            tag = "full" if d is None else f"{int(d)}m"
            out = HERE / "out" / f"extent_{name}_{tag}"
            meta = run(args.image, out, 50, d, seed=args.seed)
            if "error" in meta:
                print(f"max_depth={tag:>5}  FAILED: {meta['error'][:120]}")
                rows.append({"max_depth": d, "error": meta["error"]})
                continue
            edge = meta["voxel_edge_m"]
            rows.append({"max_depth": d, "voxel_edge_m": edge,
                         "extent_m": max(meta["extent_xyz"]),
                         "sampling_s": meta["timings_s"]["sampling"]})
            print(f"max_depth={tag:>5}  extent {max(meta['extent_xyz']):7.1f} m  "
                  f"voxel {edge*100:6.1f} cm")

    dest = HERE / "out" / f"sweep_{name}_{args.mode}.json"
    dest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
