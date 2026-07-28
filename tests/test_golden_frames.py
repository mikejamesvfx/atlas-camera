"""Golden-frame gate: the rasterised picture itself, pinned.

Atlas is a visual pipeline with an almost entirely non-visual test suite. Node
keys, math conventions, JSON shapes and coverage fractions are all pinned; what
the image actually LOOKS like was not, anywhere. Every visual regression found
on 2026-07-28 passed the full suite and was caught only by a human running
something by hand:

  * AtlasEquirectMultiView silently ran 12 depth passes instead of 4
  * EXR panoramas classified as `unknown` and were skipped entirely
  * a solve that reached its ProjectionSource but not `projection_scene`
  * headless viewport renders returning black frames

This gate closes that class.

WHY THE FENCE SITS WHERE IT DOES
Golden frames only work on deterministic output. Depth models are GPU- and
version-sensitive, so a baseline taken through MoGe would flap and be disabled
within a month — worse than no gate, because a gate people rubber-stamp
launders regressions as intentional.

So the scenes are ray-cast in pure numpy (`tests/golden_scenes.py`) and the gate
covers depth -> relief mesh -> rasterised RGB: back-projection, tearing,
triangulation, projection, z-buffer and UV interpolation. All of that is exactly
reproducible; model inference is deliberately left outside.

Verified bit-identical across repeated runs on this machine before the baselines
were taken.

TO UPDATE A BASELINE, deliberately:

    ATLAS_UPDATE_GOLDEN=1 python -m pytest tests/test_golden_frames.py

Review the diff image the failure points you at FIRST. Regenerating without
looking is the one way this suite becomes worthless.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import golden_scenes as scenes

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

GOLDEN_DIR = Path(__file__).parent / "golden"

#: A pixel counts as different past this 0-255 delta. Not zero: numpy's
#: reductions may reassociate across builds, and a 1-LSB rounding wobble is not
#: a regression. Anything a human would notice is far above this.
PIXEL_TOL = 2

#: Fraction of pixels allowed to exceed PIXEL_TOL. Deliberately tiny — a real
#: geometry change moves whole regions, not a scattering of isolated pixels.
MAX_DIFF_FRACTION = 0.001


def _golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.png"


def _write(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, optimize=True)


def _load(path: Path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _report_failure(name: str, actual, golden, diff_mask) -> str:
    """Write actual/diff next to the baseline and name the paths.

    A pixel gate that only says "images differ" is unactionable, so the failure
    always leaves something to look at.
    """
    out = GOLDEN_DIR / "_failures"
    out.mkdir(parents=True, exist_ok=True)
    _write(out / f"{name}.actual.png", actual)

    # Red where it changed, dimmed original underneath for context.
    vis = (actual.astype(np.float64) * 0.35).astype(np.uint8)
    vis[diff_mask] = [255, 0, 0]
    _write(out / f"{name}.diff.png", vis)

    delta = np.abs(actual.astype(np.int16) - golden.astype(np.int16))
    return (
        f"golden frame '{name}' changed: "
        f"{diff_mask.mean() * 100:.3f}% of pixels differ by >{PIXEL_TOL} "
        f"(max delta {int(delta.max())}/255)\n"
        f"  baseline : {_golden_path(name)}\n"
        f"  actual   : {out / (name + '.actual.png')}\n"
        f"  diff     : {out / (name + '.diff.png')}  (red = changed)\n"
        f"  If the change is intended, LOOK AT THE DIFF, then re-run with "
        f"ATLAS_UPDATE_GOLDEN=1"
    )


@pytest.mark.parametrize("name", sorted(scenes.SCENES))
def test_golden_frame(name):
    actual = scenes.render_scene_golden(name)
    path = _golden_path(name)

    if os.environ.get("ATLAS_UPDATE_GOLDEN") == "1":
        _write(path, actual)
        pytest.skip(f"baseline rewritten: {path}")

    if not path.exists():
        _write(path, actual)
        pytest.fail(
            f"no baseline for '{name}' — one was written to {path}. "
            "Inspect it, then commit it if it looks right.")

    golden = _load(path)
    assert actual.shape == golden.shape, (
        f"'{name}' render is {actual.shape}, baseline is {golden.shape}")

    delta = np.abs(actual.astype(np.int16) - golden.astype(np.int16)).max(axis=2)
    diff_mask = delta > PIXEL_TOL
    if diff_mask.mean() > MAX_DIFF_FRACTION:
        pytest.fail(_report_failure(name, actual, golden, diff_mask))


@pytest.mark.parametrize("name", sorted(scenes.SCENES))
def test_render_is_not_blank(name):
    """A gate that passes on an all-black frame would be worse than none.

    This is not hypothetical: queuing AtlasBlockoutViewport headless returns
    black frames, because it renders in the browser via three.js. If the golden
    baselines were ever regenerated from a broken render, every frame would
    match a black baseline forever.
    """
    actual = scenes.render_scene_golden(name)
    lit = (actual.max(axis=2) > 6).mean()
    assert lit > 0.30, f"'{name}' is {lit * 100:.1f}% lit — the render is broken, not the baseline"
    assert actual.std() > 12, f"'{name}' has no tonal variation (std {actual.std():.1f})"


def test_scenes_are_deterministic():
    """The premise of the whole gate, asserted rather than assumed.

    If this fails, every other test in this file is noise and the baselines
    should be deleted rather than chased.
    """
    for name in sorted(scenes.SCENES):
        a = scenes.render_scene_golden(name)
        b = scenes.render_scene_golden(name)
        assert np.array_equal(a, b), f"'{name}' is not reproducible within one process"


@pytest.mark.parametrize("name,expected_kept", [
    ("corridor", 0.990),
    ("ramp", 0.742),
    ("steps", 0.831),
])
def test_tear_rate_baseline(name, expected_kept):
    """Tearing pinned numerically as well as visually.

    The picture catches a change; this says by how much and in which direction,
    which is what you actually need when deciding whether a tear-threshold
    change was an improvement.

    These are MEASURED values, not targets. `ramp` tears most (a floor seen
    nearly edge-on is the grazing case `max_edge_factor` trades against), which
    makes it the first scene to move when thresholds change.
    """
    stats = scenes.tear_stats(name)
    gw, gh = stats["grid"]
    kept = stats["n_faces"] / (2 * (gw - 1) * (gh - 1))
    assert kept == pytest.approx(expected_kept, abs=0.02), (
        f"'{name}' kept {kept:.3f} of grid faces, baseline {expected_kept:.3f} — "
        "tearing behaviour changed")


def test_baselines_are_committed():
    """A missing baseline must fail loudly rather than silently self-heal.

    `test_golden_frame` writes one when absent so a first run is not a dead end,
    but if that file never gets committed the gate quietly protects nothing on
    every other machine.
    """
    missing = [n for n in scenes.SCENES if not _golden_path(n).exists()]
    assert not missing, f"no committed baseline for: {', '.join(sorted(missing))}"
