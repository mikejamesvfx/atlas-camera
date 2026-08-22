"""Mask geometry for the external-paint bridges.

These four functions decide which pixels an external edit is ALLOWED to move,
so a silent change here changes what every containment score means. They were
moved verbatim out of ``tools/affinity_confine_plate.py`` when the bridge
gained a second vendor; these tests pin the behaviour they had when the
Affinity numbers were measured, so any later tuning shows up as a test diff
rather than as drift in results that are already published.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.paint import masks  # noqa: E402


# --- disc -------------------------------------------------------------------

@pytest.mark.parametrize("radius", [0, 1, 2, 5, 9])
def test_disc_is_square_and_symmetric(radius):
    d = masks.disc(np, radius)
    assert d.shape == (2 * radius + 1, 2 * radius + 1)
    assert d[radius, radius]                       # centre is always set
    assert np.array_equal(d, d[::-1, :])           # symmetric vertically
    assert np.array_equal(d, d[:, ::-1])           # and horizontally


def test_disc_radius_zero_is_a_single_pixel():
    assert masks.disc(np, 0).shape == (1, 1)
    assert masks.disc(np, 0).all()


@pytest.mark.parametrize("radius", [3, 6, 12])
def test_disc_area_approximates_pi_r_squared(radius):
    """A disc, not a square: the area must track pi*r^2, which is what makes
    dilation isotropic rather than boxy."""
    area = int(masks.disc(np, radius).sum())
    assert area == pytest.approx(np.pi * radius ** 2, rel=0.25)


# --- dilate -----------------------------------------------------------------

def _shapes():
    single = np.zeros((21, 21), bool)
    single[10, 10] = True

    diagonal = np.zeros((21, 21), bool)
    for i in range(4, 17):
        diagonal[i, i] = True

    border = np.zeros((15, 17), bool)
    border[0, :] = True          # touches the top edge
    border[:, -1] = True         # and the right edge
    border[-1, 0] = True         # and a corner

    rng = np.random.default_rng(1234)
    speckle = rng.random((23, 19)) > 0.75
    return {"single": single, "diagonal": diagonal, "border": border,
            "speckle": speckle}


@pytest.mark.parametrize("name", sorted(_shapes()))
@pytest.mark.parametrize("radius", [0, 1, 3, 6])
def test_dilate_scipy_and_fallback_agree_exactly(monkeypatch, name, radius):
    """The SciPy path and the dependency-free fallback must be interchangeable.

    Border handling is where a separable fallback silently diverges from a true
    disc dilation, so the fixtures deliberately include a mask touching the top
    edge, the right edge and a corner. If these two ever disagree, containment
    silently depends on whether SciPy happens to be installed.
    """
    scipy_ndimage = pytest.importorskip("scipy.ndimage")
    assert scipy_ndimage is not None

    mask = _shapes()[name]
    with_scipy = masks.dilate(np, mask.copy(), radius)

    # Force the fallback by making the in-function `from scipy.ndimage import
    # binary_dilation` raise, exactly as it would on a machine without SciPy.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def _no_scipy(name_, *args, **kwargs):
        if name_.startswith("scipy"):
            raise ImportError("scipy disabled for this test")
        return real_import(name_, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _no_scipy)
    without_scipy = masks.dilate(np, mask.copy(), radius)

    assert np.array_equal(with_scipy, without_scipy), (
        f"{name} r={radius}: SciPy and fallback dilation disagree on "
        f"{int((with_scipy ^ without_scipy).sum())} pixels")


@pytest.mark.parametrize("name", sorted(_shapes()))
def test_dilate_radius_zero_is_identity(name):
    mask = _shapes()[name]
    assert np.array_equal(masks.dilate(np, mask.copy(), 0), mask)


def test_dilate_only_grows(monkeypatch):
    mask = _shapes()["speckle"]
    grown = masks.dilate(np, mask.copy(), 3)
    assert (grown & mask).sum() == mask.sum()      # nothing was removed
    assert grown.sum() >= mask.sum()


# --- drop -------------------------------------------------------------------

def test_drop_extends_downward_only_and_by_exactly_the_distance():
    """Gravity-directed growth: a ground-standing object's legs, footings and
    contact shadow sit BELOW it, not around it. Growing sideways would bloat
    the authorised region for no reason."""
    mask = np.zeros((40, 30), bool)
    mask[10:15, 8:20] = True                        # a block
    dropped = masks.drop(np, mask.copy(), 7)

    ys, xs = np.where(dropped)
    assert ys.min() == 10                           # top unchanged
    assert xs.min() == 8 and xs.max() == 19         # sides unchanged
    assert ys.max() == 14 + 7                       # bottom extended by exactly 7


@pytest.mark.parametrize("distance", [0, 1, 2, 3, 5, 8, 13, 16])
def test_drop_column_run_length_is_exact(distance):
    """Doubling shifts are an easy place to be off by one, and an off-by-one
    here quietly changes the authorised area on every plate."""
    mask = np.zeros((60, 3), bool)
    mask[5, 1] = True
    rows = np.where(masks.drop(np, mask.copy(), distance)[:, 1])[0]
    assert rows.min() == 5
    assert rows.max() == 5 + distance
    assert len(rows) == distance + 1                # contiguous run, no gaps


def test_drop_zero_is_identity():
    mask = _shapes()["speckle"]
    assert np.array_equal(masks.drop(np, mask.copy(), 0), mask)


def test_drop_never_grows_upward():
    mask = np.zeros((30, 10), bool)
    mask[20, 5] = True
    dropped = masks.drop(np, mask.copy(), 6)
    assert not dropped[:20].any()


# --- feather ----------------------------------------------------------------

def test_feather_stays_in_unit_range_and_keeps_its_core():
    mask = np.zeros((60, 60), np.float32)
    mask[15:45, 15:45] = 1.0
    ramp = masks.feather(np, mask.copy(), 5)

    assert ramp.min() >= 0.0 and ramp.max() <= 1.0
    # Well inside the block the ramp must still be fully opaque, or the confine
    # composite would dilute the edit in its own interior.
    assert ramp[30, 30] == pytest.approx(1.0)


def test_feather_support_covers_the_mask():
    """The authorised mask is the ramp's SUPPORT, so the ramp must never be
    narrower than the mask it came from -- that would authorise less than was
    actually painted."""
    mask = np.zeros((40, 40), np.float32)
    mask[10:30, 10:30] = 1.0
    ramp = masks.feather(np, mask.copy(), 4)
    assert ((ramp > 0.0) | (mask <= 0)).all()
    assert (ramp[mask > 0] > 0).all()


def test_feather_falls_off_monotonically_across_an_edge():
    mask = np.zeros((20, 60), np.float32)
    mask[:, :30] = 1.0
    row = masks.feather(np, mask.copy(), 6)[10]
    outward = row[30:44]
    assert np.all(np.diff(outward) <= 1e-6), "ramp must not rise going outward"


def test_feather_radius_zero_is_identity():
    mask = np.zeros((10, 10), np.float32)
    mask[3:7, 3:7] = 1.0
    assert np.array_equal(masks.feather(np, mask.copy(), 0), mask)


def test_feather_widens_the_support_beyond_the_binary_mask():
    """This is the whole reason the authorised mask is the ramp support and not
    the object mask: a feather paints OUTSIDE the binary mask, and the
    containment gate correctly rejected a 'clean' feathered edit at 0.9329 for
    exactly this reason."""
    mask = np.zeros((40, 40), np.float32)
    mask[18:22, 18:22] = 1.0
    ramp = masks.feather(np, mask.copy(), 5)
    assert (ramp > 0).sum() > (mask > 0).sum()
