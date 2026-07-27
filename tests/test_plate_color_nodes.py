"""plate colour trio — AtlasGrade / AtlasDefocus / AtlasApplyLUT.

Math layer (`atlas_camera/plate/{grade,defocus,lut}.py`) is numpy-only and
tested without torch; the node layer is tested behind `importorskip("torch")`.

Two pins here exist because a measurement caught a defect before the nodes
shipped, and the defect would be invisible at test-fixture sizes:

* `test_defocus_band_accumulation_is_energy_preserving` — the render loop
  accumulates blur bands with hat weights instead of materializing an
  (L, H, W, C) stack. Measured on the original gather: 1510 MB peak at
  1920x1080 with levels=12 (2.39 GB of stack alone at 4K, and Atlas plates
  are 4K); the accumulation form measures 365 MB for a bit-identical result.
* `test_defocus_node_reports_relative_depth_loudly` — `focus_distance_m` is a
  METRIC knob, so a relative ATLAS_DEPTH_MAP silently means something else.
  The gate doctrine requires that be visible, not guessed at.
"""

import json

import numpy as np
import pytest

from atlas_camera.plate.defocus import coc_field, defocus_plate
from atlas_camera.plate.grade import grade_plate
from atlas_camera.plate.lut import apply_lut, parse_cube

# ---------------------------------------------------------------------------
# grade
# ---------------------------------------------------------------------------

def _img(h=16, w=24, c=3, seed=0):
    return np.random.default_rng(seed).random((h, w, c)).astype(np.float32)


def test_grade_identity_is_bit_exact():
    img = _img()
    assert np.array_equal(grade_plate(img), img)                     # default knobs
    assert np.array_equal(grade_plate(img, lift=0.5, gain=2.0, mix=0.0), img)


def test_grade_gain_is_linear_and_hdr_safe():
    img = np.array([[[0.5, 4.0, 12.0]]], dtype=np.float32)
    out = grade_plate(img, gain=2.0)
    assert out[0, 0].tolist() == pytest.approx([1.0, 8.0, 24.0])      # never clamps at 1.0


def test_grade_negative_linear_skips_pow():
    img = np.array([[[-0.3, 0.25, 0.5]]], dtype=np.float32)
    out = grade_plate(img, gamma=2.2)
    assert np.isfinite(out).all()                                    # no NaN from pow(<0)
    assert out[0, 0, 0] < 0                                          # stayed on the linear branch


def test_grade_saturation_zero_is_rec709_luma():
    from atlas_camera.plate.grade import _LUMA
    img = _img(4, 4)
    out = grade_plate(img, saturation=0.0)
    luma = img[..., 0] * _LUMA[0] + img[..., 1] * _LUMA[1] + img[..., 2] * _LUMA[2]
    for c in range(3):
        assert out[..., c] == pytest.approx(luma, abs=1e-6)


def test_grade_alpha_passes_through_ungraded():
    img = _img(c=4)
    out = grade_plate(img, lift=0.1, gamma=1.8, gain=1.5, saturation=2.0)
    assert out.shape == img.shape
    assert np.array_equal(out[..., 3], img[..., 3])                  # matte never graded


def test_grade_mix_lerps():
    img = _img()
    full = grade_plate(img, gain=2.0)
    half = grade_plate(img, gain=2.0, mix=0.5)
    assert half == pytest.approx((img + full) / 2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# defocus
# ---------------------------------------------------------------------------

def test_coc_field_thin_lens_shape():
    z = np.array([[2.0, 5.0, 10.0, 1000.0]])
    coc = coc_field(z, focus_distance_m=5.0, strength_px=8.0)
    assert coc[0, 1] == pytest.approx(0.0, abs=1e-9)                 # in focus
    assert coc[0, 2] > 0 and coc[0, 3] > coc[0, 2]                   # grows with distance
    assert coc[0, 3] <= 8.0 + 1e-6                                   # background saturates
    near = coc_field(np.array([[0.01]]), focus_distance_m=5.0, strength_px=8.0)
    assert near[0, 0] <= 4.0 * 8.0 + 1e-6                            # foreground capped


def test_coc_field_nan_depth_takes_far_blur():
    coc = coc_field(np.array([[np.nan]]), focus_distance_m=5.0, strength_px=8.0)
    assert coc[0, 0] == pytest.approx(8.0)                           # sky blurs, not sharp


def test_defocus_strength_zero_is_no_op():
    img = _img(20, 30)
    z = np.full((20, 30), 8.0)
    out, _ = defocus_plate(img, z, focus_distance_m=5.0, strength_px=0.0)
    assert np.array_equal(out, img)


def test_defocus_blurs_background_not_focal_plane():
    rng = np.random.default_rng(3)
    img = rng.random((40, 60, 3)).astype(np.float32)
    z = np.empty((40, 60))
    z[:, :30] = 5.0          # in focus
    z[:, 30:] = 80.0         # far
    out, _ = defocus_plate(img, z, focus_distance_m=5.0, strength_px=10.0, levels=6)
    assert out[:, :30].var() == pytest.approx(img[:, :30].var(), rel=0.05)
    assert out[:, 35:55].var() < img[:, 35:55].var() * 0.6


def test_defocus_band_accumulation_is_energy_preserving():
    """The hat weights must sum to 1 per pixel — a flat plate stays flat.

    This is the regression pin for the memory rewrite: any weighting bug
    shows up immediately as a brightness shift on a constant image, which
    the previous two-band gather could not produce.
    """
    img = np.full((32, 48, 3), 0.42, dtype=np.float32)
    z = np.linspace(2.0, 90.0, 48)[None, :] * np.ones((32, 1))
    out, _ = defocus_plate(img, z, focus_distance_m=6.0, strength_px=12.0, levels=12)
    assert out == pytest.approx(0.42, abs=1e-6)


def test_defocus_shape_mismatch_raises():
    with pytest.raises(ValueError):
        defocus_plate(_img(10, 10), np.zeros((5, 5)), focus_distance_m=5.0)


# ---------------------------------------------------------------------------
# LUT
# ---------------------------------------------------------------------------

def _write_identity_cube(path, n=2):
    lines = [f"LUT_3D_SIZE {n}"]
    for b in range(n):
        for g in range(n):
            for r in range(n):
                lines.append(f"{r/(n-1):.6f} {g/(n-1):.6f} {b/(n-1):.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_parse_cube_3d_identity_roundtrips(tmp_path):
    lut = parse_cube(_write_identity_cube(tmp_path / "id.cube"))
    assert lut.is_3d and lut.size == 2
    img = _img(8, 8)
    assert apply_lut(img, lut) == pytest.approx(img, abs=1e-5)


def test_parse_cube_1d_ramp(tmp_path):
    p = tmp_path / "ramp.cube"
    p.write_text("LUT_1D_SIZE 3\n0 0 0\n0.5 0.5 0.5\n1 1 1\n", encoding="utf-8")
    lut = parse_cube(p)
    assert not lut.is_3d and lut.size == 3
    img = np.full((2, 2, 3), 0.5, np.float32)
    assert apply_lut(img, lut) == pytest.approx(0.5, abs=1e-6)


def test_parse_cube_domain_min_max_honored(tmp_path):
    p = tmp_path / "dom.cube"
    p.write_text("LUT_1D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 2 2 2\n0 0 0\n1 1 1\n",
                 encoding="utf-8")
    lut = parse_cube(p)
    assert lut.domain_max == (2.0, 2.0, 2.0)
    out = apply_lut(np.full((1, 1, 3), 2.0, np.float32), lut)
    assert out[0, 0] == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)


def test_parse_cube_rejects_non_cube_suffix(tmp_path):
    p = tmp_path / "x.3dl"
    p.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        parse_cube(p)
    assert "convert" in str(exc.value)


def test_parse_cube_rejects_wrong_entry_count(tmp_path):
    p = tmp_path / "short.cube"
    p.write_text("LUT_3D_SIZE 2\n0 0 0\n1 1 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_cube(p)


def test_apply_lut_intensity_bypass_and_extrapolate(tmp_path):
    p = tmp_path / "half.cube"
    p.write_text("LUT_1D_SIZE 2\n0 0 0\n0.5 0.5 0.5\n", encoding="utf-8")
    lut = parse_cube(p)
    img = np.full((1, 1, 3), 1.0, np.float32)
    assert np.array_equal(apply_lut(img, lut, intensity=0.0), img)      # exact bypass
    full = apply_lut(img, lut, intensity=1.0)[0, 0, 0]
    two = apply_lut(img, lut, intensity=2.0)[0, 0, 0]
    assert full == pytest.approx(0.5, abs=1e-6)
    assert two == pytest.approx(1.0 + 2.0 * (0.5 - 1.0), abs=1e-6)      # extrapolates


def test_apply_lut_alpha_passes_through(tmp_path):
    lut = parse_cube(_write_identity_cube(tmp_path / "id.cube"))
    img = _img(4, 4, c=4)
    out = apply_lut(img, lut)
    assert out.shape == img.shape
    assert np.array_equal(out[..., 3], img[..., 3])


# ---------------------------------------------------------------------------
# node layer
# ---------------------------------------------------------------------------

def _depth_result(depth, is_metric=True):
    from atlas_camera.inference.depth_estimator import DepthResult
    d = np.asarray(depth, np.float32)
    return DepthResult(depth=d, is_metric=is_metric, model_id="test/model",
                       image_width=d.shape[1], image_height=d.shape[0],
                       near=float(d.min()), far=float(d.max()))


def test_color_nodes_are_registered_with_color_category():
    from atlas_camera.comfy import nodes as N
    for key in ("AtlasGrade", "AtlasDefocus", "AtlasApplyLUT"):
        assert N.NODE_CLASS_MAPPINGS[key].CATEGORY == "Atlas Camera/Color"
        assert key in N.NODE_DISPLAY_NAME_MAPPINGS


def test_color_node_widget_order_is_frozen():
    """Widgets are positional in saved workflows — appends only, never inserts."""
    from atlas_camera.comfy import nodes as N
    expected = {
        "AtlasGrade": (["image"], ["lift", "gamma", "gain", "saturation", "mix"]),
        "AtlasDefocus": (["image", "depth"], ["focus_distance_m", "strength_px", "levels"]),
        "AtlasApplyLUT": (["image", "lut_path"], ["intensity"]),
    }
    for key, (req, opt) in expected.items():
        it = N.NODE_CLASS_MAPPINGS[key].INPUT_TYPES()
        assert list(it["required"]) == req
        assert list(it.get("optional", {})) == opt


def test_grade_node_batch_roundtrip():
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import AtlasGrade
    img = torch.rand(2, 12, 16, 3)
    out, report = AtlasGrade().grade(img, lift=0.02, gamma=1.1, gain=1.2)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert "lift" in report and "gamma" in report and "gain" in report


def test_defocus_node_auto_focus_uses_median_depth():
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import AtlasDefocus
    h, w = 16, 24
    z = np.concatenate([np.full((h, w // 2), 3.0), np.full((h, w - w // 2), 9.0)], axis=1)
    img = torch.rand(1, h, w, 3)
    out, coc_preview, report = AtlasDefocus().defocus(img, _depth_result(z))
    assert out.shape == img.shape
    assert coc_preview.shape == (1, h, w, 3)
    assert "focus" in report and "metric" in report


def test_defocus_node_reports_relative_depth_loudly():
    """A metres knob fed unitless depth must SAY so — gate doctrine."""
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import AtlasDefocus
    z = np.full((8, 8), 0.5)
    _, _, report = AtlasDefocus().defocus(torch.rand(1, 8, 8, 3),
                                          _depth_result(z, is_metric=False))
    assert "RELATIVE" in report


def test_apply_lut_node_missing_file_passes_through():
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import AtlasApplyLUT
    img = torch.rand(1, 8, 8, 3)
    out, report = AtlasApplyLUT().apply(img, "")
    assert out is img and "not found" in report


def test_apply_lut_node_bad_format_passes_through(tmp_path):
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import AtlasApplyLUT
    bad = tmp_path / "x.3dl"
    bad.write_text("nope", encoding="utf-8")
    img = torch.rand(1, 8, 8, 3)
    out, report = AtlasApplyLUT().apply(img, str(bad))
    assert out is img and "convert" in report          # reported, never raised


def test_apply_lut_node_applies_identity(tmp_path):
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import AtlasApplyLUT
    cube = _write_identity_cube(tmp_path / "id.cube")
    img = torch.rand(1, 8, 8, 3)
    out, report = AtlasApplyLUT().apply(img, str(cube))
    assert out.numpy() == pytest.approx(img.numpy(), abs=1e-5)
    assert "3D 2^3" in report


def test_plate_layer_has_no_comfy_imports():
    """Layering rule: plate/ is host-agnostic — nothing there imports comfy."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "atlas_camera" / "plate"
    for mod in ("grade.py", "defocus.py", "lut.py"):
        src = (root / mod).read_text(encoding="utf-8")
        assert "atlas_camera.comfy" not in src
        assert "import torch" not in src
