"""Stable identity for the things an artist attaches decisions to.

**The problem this fixes.** `extract_planes_ransac` names planes after sorting
them by inlier count (`plane_extraction.py:283-287`), so a plane's name is its
RANK. Re-extract the same evidence with one more inlier on a neighbour and every
name shifts. An artist who classified `projection_plane_03` as a facade, or an
editor that recorded an edit against it, has attached that decision to a
position in a sorted list rather than to a surface.

**Content-derived, not a UUID.** The same plane re-fitted from the same plate
keeps its id with no lookup table to maintain, and a genuinely different plane
gets a different one. A UUID would need a mapping that itself has to survive
re-extraction, which is the same problem one level down.

**The quantisation is the contract.** A re-fit moves a normal and an offset
slightly, so the id is minted from ROUNDED values: without that, identity would
be exact-float equality and nothing would ever match itself. The tolerance is
stated here rather than left implicit, because a producer that rounds
differently mints different ids for the same surface:

- normal components to 4 decimal places — about 0.006 degrees of tilt
- offset to 3 decimal places — a millimetre, in metres

A re-fit that moves a plane further than that IS a different plane, and gets a
different id. That is a judgement, and it is the one this module is stating.

`observation_id` is part of the input because the same wall photographed from
two positions is two pieces of evidence, and a scene holding both must be able
to tell them apart.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from atlas_camera.format.digest import digest_json

#: Stated tolerance. Changing either of these changes every id this build mints,
#: which orphans every artist decision attached to an existing one — so it is a
#: format change, not a tuning knob.
PLANE_NORMAL_PLACES = 4
PLANE_OFFSET_PLACES = 3

#: How many hex characters of the digest an id carries. Twelve is 48 bits; a
#: scene has tens of planes, not billions, and the id is read by humans in
#: diffs and error messages.
ID_LENGTH = 12


def plane_id_for(
    normal: Sequence[float],
    offset_m: float,
    observation_id: str | None = None,
) -> str:
    """A `plane_id` that survives re-extraction of the same evidence."""

    unit, flipped = _canonical(normal)
    # The offset flips WITH the normal. A plane is {n, d} with d = n.c, so
    # {-n, -d} is the same surface — canonicalising the normal alone made the
    # same wall mint two different ids depending on which way the fitter
    # happened to report it, which is precisely the failure this module exists
    # to prevent.
    offset = -float(offset_m) if flipped else float(offset_m)
    payload = {
        "normal": [_quantise(value, PLANE_NORMAL_PLACES) for value in unit],
        "offset_m": _quantise(offset, PLANE_OFFSET_PLACES),
        "observation_id": observation_id or "",
    }
    return "plane_" + digest_json(payload)[:ID_LENGTH]


def _quantise(value: float, places: int) -> float:
    """Round, and collapse negative zero.

    `-0.0` and `0.0` are equal as floats and DIFFERENT as JSON — `json.dumps`
    writes "-0.0" — so a normal whose sign was flipped serialised to a different
    digest than the same normal that never needed flipping, and the same wall
    minted two ids. Adding zero is the collapse: IEEE says -0.0 + 0.0 is +0.0.
    """

    return round(float(value), places) + 0.0


def _canonical(normal: Sequence[float]) -> tuple[list[float], bool]:
    """Unit length and sign-canonical, plus whether the sign was flipped.

    A plane's normal and its negation describe the same surface, and different
    fitters disagree about which way round to report it. Left alone, the same
    wall would mint two ids depending on which side the fit happened to pick.
    The first non-zero component is forced positive — arbitrary, but stable,
    and stability is the whole requirement. The caller needs to know whether a
    flip happened so the offset can flip with it.
    """

    values = [float(value) for value in normal]
    if len(values) != 3:
        raise ValueError("a plane normal has three components")
    length = sum(value * value for value in values) ** 0.5
    if length < 1e-9:
        raise ValueError("a plane normal of zero length describes no plane")
    values = [value / length for value in values]

    for value in values:
        if abs(value) > 1e-9:
            if value < 0:
                return [-item for item in values], True
            break
    return values, False


def plane_from_transform(matrix: Any) -> tuple[list[float], float] | None:
    """(normal, offset) from a proxy primitive's own transform, or None.

    Atlas planes use the THREE.PlaneGeometry frame — local X=u, Y=v, Z=normal —
    so the normal is the transform's THIRD column, not its second. The same
    convention trips up anything that assumes a Y-up ground quad, which is why
    this is written once here and mirrored from `occlusion_graph`'s own note
    rather than re-derived at each call site.

    Pure Python on purpose: this package is standard library only, and a 4x4
    dot product is not worth a numpy import in a format reader.
    """

    rows = _rows(matrix)
    if rows is None:
        return None
    normal = [rows[0][2], rows[1][2], rows[2][2]]
    length = sum(value * value for value in normal) ** 0.5
    if length < 1e-9:
        return None
    normal = [value / length for value in normal]
    centre = [rows[0][3], rows[1][3], rows[2][3]]
    offset = sum(a * b for a, b in zip(normal, centre))
    return normal, offset


def _rows(matrix: Any) -> list[list[float]] | None:
    try:
        rows = [[float(value) for value in row] for row in _iterable(matrix)]
    except (TypeError, ValueError):
        return None
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        return None
    return rows


def _iterable(value: Any) -> Iterable[Any]:
    #: `tolist` first, so a numpy array from a solve converts cleanly without
    #: this module importing numpy to recognise one.
    return value.tolist() if hasattr(value, "tolist") else value
