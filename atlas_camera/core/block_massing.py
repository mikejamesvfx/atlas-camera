"""Placeholder building mass for ground that was never photographed.

A plate shows one side of one block. Everything behind the foreground, off the
frame edges, or beyond the visible buildings is absent — and for a camera move
that absence reads as a hole in the world, not as distance.

This module fills that space with **grid-aligned cuboids**, and the whole point
is that it invents as little as possible. Every number a box is built from is
measured off the plate:

* the ground plane and metric scale come from the solve,
* the street-grid azimuth is fitted to back-projected ground lines,
* heights are RESAMPLED from roof heights actually observed in frame.

What is invented is only *that a building exists there at all*. That is the one
claim a placeholder is entitled to make, and it is why these are boxes rather
than architecture: a cuboid at the right scale, on the right grid, at a
plausible height reads as "a building stands here" and nothing more. Give it
windows and it starts asserting things nobody measured.

TRUST. Nothing here is measurement and the output says so — every box carries
``provenance="placeholder"``. It must never be promoted to a measured tier, and
a consumer that cannot tell the difference should not be consuming it.

Layering: numpy only, no ComfyUI and no OpenCV. Line DETECTION belongs to the
caller (it needs cv2, which lives in the vision extra); this module takes
segments that are already on the ground plane.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: A grid fit looser than this is not a grid. Streets in a real city are
#: straight to a fraction of a degree; a "grid" fitted to noise wanders, and
#: placing boxes on it produces mass that visibly disagrees with the plate.
MIN_GRID_COHERENCE = 0.35

#: Segments shorter than this (metres, on the ground) are mostly kerb stones,
#: road markings and vehicle edges — they carry a direction but it is not the
#: street's, and they outnumber the real lines.
MIN_SEGMENT_M = 1.0


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:                      # pragma: no cover
        raise RuntimeError(
            "Block massing requires numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


@dataclass(frozen=True)
class GridFit:
    """A street grid fitted to ground evidence."""

    azimuth_deg: float
    #: Fraction of total segment LENGTH lying within tolerance of the fit.
    #: Length-weighted on purpose: one 40 m kerb is better evidence than
    #: twenty 2 m fragments, and counting segments would say the opposite.
    coherence: float
    n_segments: int
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.coherence >= MIN_GRID_COHERENCE and self.n_segments >= 4


@dataclass(frozen=True)
class MassingBox:
    """One placeholder mass, in grid coordinates and in world space."""

    u0: float
    u1: float
    v0: float
    v1: float
    height_m: float
    #: 8 world-space corners, base first then top, matching `primitive_mesh`'s
    #: unit-box winding so a consumer can build faces without re-deriving them.
    corners: tuple[tuple[float, float, float], ...]
    provenance: str = "placeholder"
    zone: str = ""

    @property
    def footprint_m2(self) -> float:
        return abs(self.u1 - self.u0) * abs(self.v1 - self.v0)


def estimate_grid_azimuth(segments: Any, *, tolerance_deg: float = 5.0) -> GridFit:
    """Fit ONE street azimuth to segments already projected onto the ground.

    ``segments`` is (N, 2, 3) world-space endpoints. A city grid has two
    perpendicular families, so the fit is taken **modulo 90 degrees** — both
    families then reinforce the same estimate instead of averaging to something
    between them, which is what a modulo-180 fit would do.

    The estimator is a length-weighted circular mean. Circular, because 89 deg
    and 1 deg are 2 deg apart on a mod-90 space and an arithmetic mean of those
    returns 45 — the exact wrong answer, and one that looks plausible.
    """
    np = _require_numpy()
    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 3)
    if len(seg) == 0:
        return GridFit(0.0, 0.0, 0, "no ground segments were supplied")

    delta = seg[:, 1, :] - seg[:, 0, :]
    length = np.hypot(delta[:, 0], delta[:, 2])          # ground-plane length
    keep = length >= MIN_SEGMENT_M
    if not keep.any():
        return GridFit(0.0, 0.0, 0,
                       f"every segment is shorter than {MIN_SEGMENT_M} m on the "
                       "ground — too short to carry a street direction")
    delta, length = delta[keep], length[keep]

    theta = np.degrees(np.arctan2(delta[:, 0], -delta[:, 2])) % 90.0
    # mod-90 data lives on a quarter circle, so the angle must be quadrupled
    # before averaging and quartered after.
    quad = np.radians(4.0 * theta)
    azimuth = (np.degrees(np.arctan2((length * np.sin(quad)).sum(),
                                     (length * np.cos(quad)).sum())) / 4.0) % 90.0

    dev = np.abs(((theta - azimuth + 45.0) % 90.0) - 45.0)
    coherence = float(length[dev < tolerance_deg].sum() / length.sum())
    fit = GridFit(float(azimuth), coherence, int(len(theta)))
    if not fit.usable:
        return GridFit(
            fit.azimuth_deg, fit.coherence, fit.n_segments,
            f"only {100 * coherence:.0f}% of ground-line length agrees on one "
            f"azimuth (need {100 * MIN_GRID_COHERENCE:.0f}%) — this scene has no "
            "usable street grid, so boxes would be placed on a direction fitted "
            "to noise")
    return fit


def grid_basis(azimuth_deg: float) -> tuple[Any, Any]:
    """Unit vectors along and across the grid, both on the ground plane."""
    np = _require_numpy()
    a = np.radians(float(azimuth_deg))
    return (np.array([np.sin(a), 0.0, -np.cos(a)]),
            np.array([np.cos(a), 0.0, np.sin(a)]))


def sample_heights(observed_m: Any, count: int, *, seed: int = 0) -> Any:
    """Draw ``count`` heights by RESAMPLING the observed ones.

    Deliberately not a fitted distribution. A normal fitted to a handful of
    roofs has tails, and its tails produce buildings taller and shorter than
    anything in the plate — invented values wearing the authority of a
    measurement. Resampling can only ever return a height that was actually
    seen, so the worst case is a real building in the wrong place rather than
    a building that could not exist here.
    """
    np = _require_numpy()
    obs = np.asarray(observed_m, dtype=np.float64).reshape(-1)
    obs = obs[np.isfinite(obs) & (obs > 0)]
    if obs.size == 0:
        raise ValueError(
            "no observed heights to resample — placeholder masses must take "
            "their height from the plate, and there is nothing to take it from")
    rng = np.random.default_rng(int(seed))
    return rng.choice(obs, size=int(max(0, count)), replace=True)


def _overlaps(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float], *, margin: float) -> bool:
    """Axis-aligned overlap test in GRID coords (both rects share the basis)."""
    au0, au1, av0, av1 = a
    bu0, bu1, bv0, bv1 = b
    return not (au1 + margin <= bu0 or bu1 + margin <= au0
                or av1 + margin <= bv0 or bv1 + margin <= av0)


def place_massing(
    *,
    azimuth_deg: float,
    ground_y: float,
    observed_heights_m: Any,
    region_uv: tuple[float, float, float, float],
    occupied_uv: list[tuple[float, float, float, float]] | None = None,
    street_bands_v: list[tuple[float, float]] | None = None,
    block_depth_m: float = 62.0,
    frontage_range_m: tuple[float, float] = (16.0, 40.0),
    depth_range_m: tuple[float, float] = (26.0, 56.0),
    gap_m: float = 1.5,
    seed: int = 0,
    zone: str = "",
) -> list[MassingBox]:
    """Fill ``region_uv`` with grid-aligned placeholder masses.

    ``region_uv`` is ``(u0, u1, v0, v1)`` in grid coordinates — the caller
    decides *where* placeholders are wanted (behind the foreground, off-frame,
    into the distance), because that judgement needs the occlusion analysis and
    does not belong in a geometry helper.

    ``occupied_uv`` are footprints already MEASURED from the plate. Placeholders
    never overlap them: a box driven through a photographed building is the one
    failure that cannot be explained away as "it is only a blockout".

    ``street_bands_v`` are the roadway bands to keep clear. Without them the
    lattice cheerfully paves the street, which reads as wrong instantly because
    the street is the one part of the ground the viewer can see is empty.
    """
    np = _require_numpy()
    u0, u1, v0, v1 = (float(x) for x in region_uv)
    if u1 <= u0 or v1 <= v0:
        return []
    U, V = grid_basis(azimuth_deg)
    rng = np.random.default_rng(int(seed))
    occupied = list(occupied_uv or [])
    bands = list(street_bands_v or [])

    boxes: list[MassingBox] = []
    # Walk the region in rows one block deep, then along the row in frontages.
    # Both dimensions are jittered inside measured ranges so the result reads as
    # a street rather than a colonnade of identical blocks.
    row_v = v0
    while row_v < v1 - 1.0:
        row_top = min(row_v + float(block_depth_m), v1)
        u = u0
        while u < u1 - 1.0:
            frontage = float(rng.uniform(*frontage_range_m))
            depth = min(float(rng.uniform(*depth_range_m)), row_top - row_v)
            bu0, bu1 = u, min(u + frontage, u1)
            bv1 = row_top
            bv0 = max(bv1 - depth, row_v)
            u = bu1 + float(gap_m)
            if bu1 - bu0 < 6.0 or bv1 - bv0 < 6.0:
                continue
            rect = (bu0, bu1, bv0, bv1)
            if any(lo < bv1 and bv0 < hi for lo, hi in bands):
                continue                       # would sit in the roadway
            if any(_overlaps(rect, o, margin=gap_m) for o in occupied):
                continue                       # would sit inside measured mass
            boxes.append(rect)                 # placeholder; height added below
        row_v = row_top

    if not boxes:
        return []
    heights = sample_heights(observed_heights_m, len(boxes), seed=seed)
    out: list[MassingBox] = []
    for rect, h in zip(boxes, heights):
        bu0, bu1, bv0, bv1 = rect
        base = [bu * U + bv * V + np.array([0.0, float(ground_y), 0.0])
                for bu, bv in ((bu0, bv0), (bu1, bv0), (bu1, bv1), (bu0, bv1))]
        top = [p + np.array([0.0, float(h), 0.0]) for p in base]
        out.append(MassingBox(
            u0=bu0, u1=bu1, v0=bv0, v1=bv1, height_m=float(h),
            corners=tuple(tuple(float(c) for c in p) for p in base + top),
            zone=zone))
    return out


def box_transform(box: MassingBox, azimuth_deg: float, ground_y: float
                  ) -> tuple[tuple[tuple[float, ...], ...], tuple[float, float, float]]:
    """``MassingBox`` -> the ``(transform_matrix, dimensions)`` a box primitive wants.

    `AtlasProxyPrimitive` stores a box as a centre transform plus extents, not
    as corners, so the grid rotation has to live in the MATRIX. Baking the
    rotation into the corners and shipping an identity transform would render
    identically in the viewport and then export axis-aligned to every DCC,
    because the exporters read the transform rather than re-deriving it.
    """
    np = _require_numpy()
    U, V = grid_basis(azimuth_deg)
    up = np.array([0.0, 1.0, 0.0])
    centre = (0.5 * (box.u0 + box.u1) * U + 0.5 * (box.v0 + box.v1) * V
              + np.array([0.0, float(ground_y) + 0.5 * box.height_m, 0.0]))
    m = [[U[0], up[0], V[0], centre[0]],
         [U[1], up[1], V[1], centre[1]],
         [U[2], up[2], V[2], centre[2]],
         [0.0, 0.0, 0.0, 1.0]]
    dims = (abs(box.u1 - box.u0), box.height_m, abs(box.v1 - box.v0))
    return (tuple(tuple(float(c) for c in row) for row in m),
            (float(dims[0]), float(dims[1]), float(dims[2])))


def massing_report(boxes: list[MassingBox], fit: GridFit) -> str:
    """One human-readable line per decision, for the node's text output."""
    if not boxes:
        return "no placeholder masses placed"
    hs = sorted(b.height_m for b in boxes)
    area = sum(b.footprint_m2 for b in boxes)
    return (f"{len(boxes)} placeholder masses on a {fit.azimuth_deg:.2f}deg grid "
            f"({100 * fit.coherence:.0f}% of ground-line length agrees); "
            f"footprint {area:,.0f} m2; heights {hs[0]:.1f}-{hs[-1]:.1f} m "
            f"(median {hs[len(hs) // 2]:.1f} m), all resampled from observed "
            "roofs; provenance=placeholder, never a measured tier")
