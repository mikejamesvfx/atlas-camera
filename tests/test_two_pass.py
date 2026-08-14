"""Two-pass fill: the inter-pass gate and the template guard-rails.

The gate is the load-bearing piece — pass 2's job is making pixels look
convincing, so a broken pass 1 must be refused BEFORE it is laundered into
confident fiction. Every guard here encodes a failure that happened live on
2026-08-14 (see docs/dev/occlusion_arms_2026-08-14/README.md).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from atlas_camera.dynamic.fill_metrics import edge_extend
from atlas_camera.dynamic.two_pass import (
    MAX_SENTINEL_BLEED_FRAC,
    check_wan_template,
    interpass_gate,
    load_template_guarded,
    wan_generation_raster,
)


def _plate(h=96, w=96, seed=0):
    rng = np.random.default_rng(seed)
    lum = rng.integers(90, 140, size=(h, w)).astype(np.int16)
    lum[:, ::8] = lum[:, ::8] // 2 + 60
    base = np.stack([lum + 4, lum, lum - 4], axis=-1)
    return np.clip(base, 0, 255).astype(np.uint8)


def _hole(h=96, w=96):
    m = np.zeros((h, w), dtype=bool)
    m[32:64, 32:64] = True
    return m


# ------------------------------------------------------------------ gate

def test_gate_passes_a_real_fill():
    guide = _plate()
    hole = _hole()
    rng = np.random.default_rng(1)
    fill = guide.copy()
    fill[hole] = rng.integers(60, 200, size=(int(hole.sum()), 3))
    verdict = interpass_gate(fill, guide, hole)
    assert verdict.ok, verdict.reasons
    assert verdict.sentinel_bleed_frac == 0.0


def test_gate_refuses_an_edge_extend_noop():
    """A structure pass indistinguishable from the deterministic smear is not
    worth a texture pass."""
    guide = _plate()
    hole = _hole()
    smear = edge_extend(guide, hole)
    verdict = interpass_gate(smear, guide, hole)
    assert not verdict.ok
    assert any("edge-extend" in r for r in verdict.reasons)


def test_gate_refuses_sentinel_bleed():
    """WAN preserved chroma green as CONTENT at cfg 6 (measured live)."""
    guide = _plate()
    hole = _hole()
    rng = np.random.default_rng(2)
    fill = guide.copy()
    fill[hole] = rng.integers(60, 200, size=(int(hole.sum()), 3))
    ys, xs = np.where(hole)
    n = max(1, int(len(ys) * (MAX_SENTINEL_BLEED_FRAC * 4)))
    fill[ys[:n], xs[:n]] = (102, 255, 0)
    verdict = interpass_gate(fill, guide, hole)
    assert not verdict.ok
    assert any("sentinel" in r for r in verdict.reasons)


def test_gate_refuses_a_shifted_fill():
    """E6 lesson: a displaced fill composites misregistered regardless of how
    plausible it looks — measure, never eyeball."""
    pytest.importorskip("cv2")
    guide = _plate(128, 128, seed=3)
    hole = np.zeros((128, 128), dtype=bool)
    hole[48:80, 48:80] = True
    shifted = np.roll(guide, 6, axis=1)          # whole content 6px sideways
    shifted[hole] = 200
    verdict = interpass_gate(shifted, guide, hole)
    assert not verdict.ok
    assert any("shift" in r for r in verdict.reasons)


def test_run_two_pass_returns_empty_hole_without_touching_a_generator():
    """The artist mis-click case: a holeless region must short-circuit BEFORE
    any template or generator work (the gate crashed on the empty mask when
    this guard did not exist)."""
    from atlas_camera.dynamic.two_pass import run_two_pass_fill

    def boom():
        raise AssertionError("generator must not be constructed")

    out = run_two_pass_fill(
        boom, "unused_dir", _plate(), np.zeros((96, 96), np.uint8),
        wan_template="does_not_exist.json",
        sdxl_template="does_not_exist.json", prompt="x")
    assert out == {"status": "empty_hole"}


# ------------------------------------------------------ template guards

def test_bel_control_character_is_named(tmp_path):
    """The adapter unicode-escapes template text: SDXL\\a... became BEL and
    cost a live run. The loader must refuse and explain."""
    bad = {"1": {"class_type": "CheckpointLoaderSimple",
                 "inputs": {"ckpt_name": "SDXL\x07lbedobase.safetensors"}}}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="DOUBLE-backslash"):
        load_template_guarded(p)


def test_clean_template_loads(tmp_path):
    good = {"1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "SDXL\\\\albedobase.safetensors",
                             "note": "line\nbreaks\tare fine"}}}
    p = tmp_path / "good.json"
    p.write_text(json.dumps(good), encoding="utf-8")
    assert load_template_guarded(p)["1"]["class_type"] == \
        "CheckpointLoaderSimple"


def test_wan_cfg_and_length_guards():
    tmpl = {
        "22": {"class_type": "KSampler", "inputs": {"cfg": 6.0}},
        "21": {"class_type": "WanVaceToVideo",
               "inputs": {"width": 1280, "height": 528, "length": 4}},
    }
    problems = check_wan_template(tmpl)
    assert any("cfg=6.0" in p for p in problems)
    assert any("4k+1" in p for p in problems)
    tmpl["22"]["inputs"]["cfg"] = 1.0
    tmpl["21"]["inputs"]["length"] = 5
    assert check_wan_template(tmpl) == []


def test_wan_raster_is_read_from_the_template():
    """WanVaceToVideo carries bare width/height ints that the adapter's
    config-resize would stomp — the guard is to read and echo them."""
    tmpl = {"21": {"class_type": "WanVaceToVideo",
                   "inputs": {"width": 1280, "height": 528, "length": 5}}}
    assert wan_generation_raster(tmpl) == (1280, 528)
    with pytest.raises(ValueError, match="WanVaceToVideo"):
        wan_generation_raster({"1": {"class_type": "KSampler", "inputs": {}}})


# --------------------------------------------------- kit template pins

KIT = r"C:\Users\miike\comfyui-agent-kit-data\workflow_templates"


@pytest.mark.skipif(not __import__("pathlib").Path(KIT).is_dir(),
                    reason="kit templates not on this machine")
def test_shipped_two_pass_templates_carry_the_guarded_values():
    from pathlib import Path

    wan = load_template_guarded(Path(KIT) / "atlas_wan21_vace_fill.json")
    assert check_wan_template(wan) == []
    assert wan_generation_raster(wan) == (1280, 528)
    # mask must reach the sampler as a latent noise mask in the SDXL pass
    sdxl = load_template_guarded(Path(KIT) / "atlas_sdxl_retexture_fill.json")
    kinds = {n.get("class_type") for n in sdxl.values()}
    assert "SetLatentNoiseMask" in kinds
    ks = [n for n in sdxl.values() if n.get("class_type") == "KSampler"]
    assert ks and 0.3 <= float(ks[0]["inputs"]["denoise"]) <= 0.6
