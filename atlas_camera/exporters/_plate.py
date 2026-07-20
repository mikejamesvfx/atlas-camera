"""Shared source-plate resolution helpers for the DCC/review exporters.

A leaf module (no imports from sibling exporter modules) so maya_exporter.py,
nuke_exporter.py, and review_package.py can all depend on it without a
circular import through exporters/__init__.py.
"""

from __future__ import annotations

import os

from atlas_camera.core.schema import AtlasSolve


def primary_plate_path(solve: AtlasSolve, must_exist: bool = False) -> str | None:
    """The best available source-plate path: a registered non-proxy plate_ref
    if present, else the solve's own image_path. Always a str (or None) —
    callers that need a Path should wrap the result themselves.

    `must_exist=True` additionally drops an auto-recorded `solve.image_path`
    that is not on disk. Pass it from any exporter that bakes the path into an
    artifact expected to LOAD (a .nk Read, a .ma file node); leave it False for
    provenance records (the project manifest deliberately keeps a declared but
    unreachable path, with a null md5).

    Why it matters: for every tensor-based solve in ComfyUI, `image_path` is a
    NamedTemporaryFile that the solve node itself unlinks in its own `finally`
    block, so the recorded path is already dangling by the time an exporter
    runs — which baked a dead Read path into every .nk/.py produced by the
    quickstart (found by the Linux beta test). Dropping it lets each exporter
    take its existing packaged-source fallback, which produces a script that
    actually opens; wire an AtlasRegisterPlate to get the real plate in.

    A registered plate_ref is always returned VERBATIM, existence unchecked: it
    is an explicit artist declaration of where the final plate lives, and may
    legitimately resolve only on the DCC machine (a different mount of the same
    share).
    """
    plate = getattr(solve, "source_plate", None)
    if plate and plate.image_path and not plate.is_proxy:
        return str(plate.image_path)
    path = solve.image_path
    if must_exist and path and not os.path.exists(path):
        return None
    return path


def primary_plate_colorspace(solve: AtlasSolve) -> str | None:
    """The best available source-plate colorspace: a registered plate_ref's
    colorspace, else the solve's output_profile working colorspace, else None."""
    plate = getattr(solve, "source_plate", None)
    if plate and plate.colorspace:
        return str(plate.colorspace)
    profile = getattr(solve, "output_profile", None)
    if profile and profile.working_colorspace:
        return str(profile.working_colorspace)
    return None
