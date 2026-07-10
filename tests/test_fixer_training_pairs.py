"""Tests for tools/generate_fixer_training_pairs.py (the Fixer fine-tune
packaging stage — see docs/dev/fixer_finetune_data_plan.md)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from generate_fixer_training_pairs import (  # noqa: E402
    FIXER_PROMPT,
    FIXER_TRAIN_H,
    FIXER_TRAIN_W,
    build_dataset,
    collect_pairs_from_dirs,
    letterbox_to_fixer,
)


def _write_png(path, w=64, h=32, color=(200, 30, 30)):
    PIL = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL.new("RGB", (w, h), color).save(path)


def _pair_dirs(tmp_path, n=10):
    deg, tgt = tmp_path / "deg", tmp_path / "tgt"
    for i in range(n):
        _write_png(deg / f"frame_{i:03d}.png")
        _write_png(tgt / f"frame_{i:03d}.png", color=(30, 200, 30))
    return deg, tgt


def test_collect_pairs_matches_by_stem(tmp_path):
    deg, tgt = _pair_dirs(tmp_path, n=4)
    pairs = collect_pairs_from_dirs(deg, tgt)
    assert len(pairs) == 4
    assert all(Path(p["image"]).stem == Path(p["target_image"]).stem
               for p in pairs)


def test_collect_pairs_errors_on_orphans(tmp_path):
    deg, tgt = _pair_dirs(tmp_path, n=3)
    _write_png(deg / "frame_999.png")
    with pytest.raises(SystemExit, match="unpaired"):
        collect_pairs_from_dirs(deg, tgt)


def test_build_dataset_writes_fixer_manifest_with_interleaved_split(tmp_path):
    deg, tgt = _pair_dirs(tmp_path, n=10)
    pairs = collect_pairs_from_dirs(deg, tgt)
    out = build_dataset(pairs, tmp_path / "ds", test_fraction=0.2)
    data = json.loads(out.read_text())
    assert set(data) == {"train", "test"}
    assert len(data["test"]) == 2 and len(data["train"]) == 8
    # interleaved, not tail: pair_000004 / pair_000009 (every 5th)
    assert set(data["test"]) == {"pair_000004", "pair_000009"}
    entry = data["train"]["pair_000000"]
    assert entry["prompt"] == FIXER_PROMPT
    assert "\\" not in entry["image"]  # forward slashes for the linux container


def test_build_dataset_zero_test_fraction(tmp_path):
    deg, tgt = _pair_dirs(tmp_path, n=5)
    out = build_dataset(collect_pairs_from_dirs(deg, tgt), tmp_path / "ds",
                        test_fraction=0.0)
    data = json.loads(out.read_text())
    assert len(data["train"]) == 5 and not data["test"]


def test_letterbox_preserves_aspect_and_pads_to_native(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    src = tmp_path / "in.png"
    _write_png(src, w=512, h=512)  # square into 16:9 -> side pillarbox
    dst = tmp_path / "out.png"
    letterbox_to_fixer(src, dst)
    im = PIL.open(dst)
    assert im.size == (FIXER_TRAIN_W, FIXER_TRAIN_H)
    px = im.load()
    assert px[0, FIXER_TRAIN_H // 2] == (0, 0, 0)          # padded edge
    assert px[FIXER_TRAIN_W // 2, FIXER_TRAIN_H // 2] != (0, 0, 0)  # content


def test_resize_mode_writes_aligned_copies(tmp_path):
    deg, tgt = _pair_dirs(tmp_path, n=2)
    out = build_dataset(collect_pairs_from_dirs(deg, tgt), tmp_path / "ds",
                        test_fraction=0.0, resize=True)
    data = json.loads(out.read_text())
    for entry in data["train"].values():
        assert Path(entry["image"]).exists()
        assert Path(entry["target_image"]).exists()
        assert "frames" in entry["image"]
