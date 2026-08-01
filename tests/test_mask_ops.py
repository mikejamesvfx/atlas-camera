"""One dilation, with defined border behaviour.

Six separate binary dilations existed across the node layer — three inside a
single method — and they did not agree at the frame border: the `np.roll`
version in `AtlasOcclusionMask` wrapped, so a mask touching the left edge grew
onto the right edge. The wraparound was a known defect, recorded in a comment
next to the code rather than in a test.
"""
import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.mask_ops import box_blur, dilate  # noqa: E402


def test_zero_iterations_changes_nothing():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    assert np.array_equal(dilate(mask, 0), mask)


def test_one_iteration_grows_a_point_into_a_plus():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    got = dilate(mask, 1)
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = expected[1, 2] = expected[3, 2] = True
    expected[2, 1] = expected[2, 3] = True
    assert np.array_equal(got, expected)


def test_dilation_does_not_wrap_around_the_frame():
    """The defect this module exists to remove: a left-edge mask must not
    appear on the right edge."""
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 0] = True
    got = dilate(mask, 1)
    assert not got[2, -1], "dilation wrapped across the frame border"
    assert got[2, 1] and got[1, 0] and got[3, 0]


def test_iterations_accumulate():
    mask = np.zeros((9, 9), dtype=bool)
    mask[4, 4] = True
    # 4-connected growth reaches Manhattan distance == iterations
    assert dilate(mask, 3)[4, 1]
    assert not dilate(mask, 2)[4, 1]


def test_the_input_is_never_mutated():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    before = mask.copy()
    dilate(mask, 2)
    assert np.array_equal(mask, before)


def test_eight_connectivity_reaches_diagonals():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    assert dilate(mask, 1, connectivity=8)[1, 1]
    assert not dilate(mask, 1, connectivity=4)[1, 1]


def test_an_empty_mask_stays_empty():
    mask = np.zeros((4, 4), dtype=bool)
    assert not dilate(mask, 5).any()


def test_a_full_mask_stays_full():
    mask = np.ones((4, 4), dtype=bool)
    assert dilate(mask, 3).all()


def test_box_blur_preserves_a_constant_field():
    field = np.full((7, 7), 3.0)
    assert np.allclose(box_blur(field, 2), 3.0)


def test_box_blur_spreads_a_spike_and_conserves_mass():
    field = np.zeros((9, 9))
    field[4, 4] = 1.0
    out = box_blur(field, 1)
    assert out[4, 4] < 1.0
    assert out[3, 4] > 0.0
    assert np.isclose(out.sum(), 1.0, atol=1e-9)


def test_box_blur_with_zero_radius_is_the_identity():
    field = np.arange(16, dtype=np.float64).reshape(4, 4)
    assert np.allclose(box_blur(field, 0), field)
