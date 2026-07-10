"""Package {degraded render, ground truth} image pairs into Fixer's training
dataset format.

NVIDIA Fixer (docker/fixer/Dockerfile, the AtlasRenderFix node's model) ships
full fine-tuning code that trains from a simple JSON manifest of aligned image
pairs at 576×1024:

    {"train": {"<id>": {"image": ..., "target_image": ...,
                        "prompt": "remove degradation"}}, "test": {...}}

This tool builds that manifest — and optionally letterboxes every frame to
Fixer's native 576×1024 (aspect preserved, edges padded; both sides of a pair
get the IDENTICAL transform, which is the whole game: Fixer pairs must stay
pixel-aligned).

Pair sources it accepts:
  --degraded-dir/--target-dir   two folders matched by filename (the natural
                                output of "bake the projected render at a pose
                                where a real photo exists")
  --manifest pairs.json         explicit [{"image": ..., "target_image": ...}]
                                list for pairs that live anywhere

Where do PAIRS come from? A degraded render needs a real photo at the SAME
camera — single-photo solves have no ground truth off the recovered view, so
pairs come from multi-view sources (video clips, RealEstate10K-style
sequences): solve view A with Atlas, project onto camera B, ground truth = the
real frame at B. That pipeline is designed (not yet built) in
docs/dev/fixer_finetune_data_plan.md; this tool is its final packaging stage,
usable today with pairs from any source.

Usage:
    python tools/generate_fixer_training_pairs.py \
        --degraded-dir renders/ --target-dir photos/ \
        --output-dir dataset/ --test-fraction 0.1 --resize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FIXER_TRAIN_W = 1024
FIXER_TRAIN_H = 576
FIXER_PROMPT = "remove degradation"
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def collect_pairs_from_dirs(degraded_dir: Path, target_dir: Path) -> list[dict]:
    """Match frames across two folders by filename stem; error on orphans so a
    silently half-matched dataset can't train."""
    def frames(d):
        return {p.stem: p for p in sorted(d.iterdir())
                if p.suffix.lower() in _IMAGE_EXTS}
    deg, tgt = frames(degraded_dir), frames(target_dir)
    missing_t = sorted(set(deg) - set(tgt))
    missing_d = sorted(set(tgt) - set(deg))
    if missing_t or missing_d:
        raise SystemExit(
            f"unpaired frames — in {degraded_dir.name} only: {missing_t[:5]}"
            f"{'...' if len(missing_t) > 5 else ''}; in {target_dir.name} "
            f"only: {missing_d[:5]}{'...' if len(missing_d) > 5 else ''}")
    if not deg:
        raise SystemExit(f"no images found in {degraded_dir}")
    return [{"image": str(deg[k]), "target_image": str(tgt[k])}
            for k in sorted(deg)]


def load_pairs_manifest(path: Path) -> list[dict]:
    pairs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pairs, list):
        raise SystemExit("--manifest must be a JSON list of "
                         '{"image": ..., "target_image": ...} objects')
    for i, p in enumerate(pairs):
        for key in ("image", "target_image"):
            if key not in p:
                raise SystemExit(f"--manifest entry {i} missing {key!r}")
            if not Path(p[key]).is_file():
                raise SystemExit(f"--manifest entry {i}: {p[key]} not found")
    return pairs


def letterbox_to_fixer(src: Path, dst: Path) -> None:
    """Aspect-preserving resize onto a 1024×576 canvas, edges black-padded.
    Deterministic per input SIZE, so a pair whose two frames share dimensions
    (they must — they're the same camera) gets the identical transform."""
    from PIL import Image

    im = Image.open(src).convert("RGB")
    scale = min(FIXER_TRAIN_W / im.width, FIXER_TRAIN_H / im.height)
    new = im.resize((max(1, round(im.width * scale)),
                     max(1, round(im.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (FIXER_TRAIN_W, FIXER_TRAIN_H))
    canvas.paste(new, ((FIXER_TRAIN_W - new.width) // 2,
                       (FIXER_TRAIN_H - new.height) // 2))
    canvas.save(dst, format="PNG")


def build_dataset(pairs: list[dict], output_dir: Path,
                  test_fraction: float = 0.1, resize: bool = False) -> Path:
    """Write data.json (+ optionally the letterboxed copies) under output_dir.

    The test split takes every Nth pair rather than the tail: consecutive
    frames from one baked sequence are near-duplicates, so a tail split would
    make test scores meaninglessly easy for every sequence but the last one.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    every_n = max(2, round(1.0 / test_fraction)) if test_fraction > 0 else 0
    manifest: dict = {"train": {}, "test": {}}
    for i, pair in enumerate(pairs):
        pid = f"pair_{i:06d}"
        image, target = Path(pair["image"]), Path(pair["target_image"])
        if resize:
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            img_out = frames_dir / f"{pid}_image.png"
            tgt_out = frames_dir / f"{pid}_target.png"
            letterbox_to_fixer(image, img_out)
            letterbox_to_fixer(target, tgt_out)
            image, target = img_out, tgt_out
        split = "test" if (every_n and i % every_n == every_n - 1) else "train"
        manifest[split][pid] = {
            "image": str(image.resolve()).replace("\\", "/"),
            "target_image": str(target.resolve()).replace("\\", "/"),
            "prompt": FIXER_PROMPT,
        }
    out = output_dir / "data.json"
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path,
                     help='JSON list of {"image","target_image"} pairs')
    src.add_argument("--degraded-dir", type=Path,
                     help="folder of degraded renders (paired with "
                          "--target-dir by filename)")
    ap.add_argument("--target-dir", type=Path,
                    help="folder of ground-truth frames (with --degraded-dir)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--resize", action="store_true",
                    help=f"letterbox all frames to Fixer's native "
                         f"{FIXER_TRAIN_W}x{FIXER_TRAIN_H} (copies written "
                         f"under output-dir/frames/)")
    args = ap.parse_args(argv)

    if args.degraded_dir is not None:
        if args.target_dir is None:
            ap.error("--degraded-dir requires --target-dir")
        pairs = collect_pairs_from_dirs(args.degraded_dir, args.target_dir)
    else:
        pairs = load_pairs_manifest(args.manifest)

    out = build_dataset(pairs, args.output_dir,
                        test_fraction=args.test_fraction, resize=args.resize)
    data = json.loads(out.read_text(encoding="utf-8"))
    print(f"{out}  train={len(data['train'])}  test={len(data['test'])}")
    print("Train with Fixer's own trainer (see its README):")
    print(f"  DATASET_FOLDER={out} + --pretrained_path models/pretrained/"
          "pretrained_fixer.pkl to fine-tune from the shipped checkpoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
