"""Shared source-plate resolution helpers for the DCC/review exporters.

A leaf module (no imports from sibling exporter modules) so maya_exporter.py,
nuke_exporter.py, and review_package.py can all depend on it without a
circular import through exporters/__init__.py.
"""

from __future__ import annotations

from atlas_camera.core.schema import AtlasSolve


def primary_plate_path(solve: AtlasSolve) -> str | None:
    """The best available source-plate path: a registered non-proxy plate_ref
    if present, else the solve's own image_path. Always a str (or None) —
    callers that need a Path should wrap the result themselves."""
    plate = getattr(solve, "source_plate", None)
    if plate and plate.image_path and not plate.is_proxy:
        return str(plate.image_path)
    return solve.image_path


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
