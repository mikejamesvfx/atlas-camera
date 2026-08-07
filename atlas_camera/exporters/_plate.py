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
    """The registered plate_ref's colorspace, or None. Never a guess.

    This used to fall back to ``output_profile.working_colorspace`` (default
    ``ACEScg``), which conflates two genuinely different things: the COMP
    WORKING SPACE the artist wants to end up in, and the SOURCE FILE's own
    space. They coincide often enough for the mistake to hide, and when they
    diverge the export states something false with full confidence.

    Found live 2026-08-07 on a D810 NEF: the delivered .nk tagged its Read
    ``ACEScg`` while the EXR on disk was ``lin_rec709_scene``, so Rec.709
    primaries were read as AP1 and nobody was told.

    An absent tag beats a wrong one. Absent makes the DCC apply its own default
    and leaves the artist looking at an unspecified colourspace they can
    notice and fix; wrong looks authoritative and gets believed. Every caller
    (Nuke's two writers, Maya, the manifest) already handles None by omitting
    the knob and recording "unspecified".
    """
    plate = getattr(solve, "source_plate", None)
    if plate and plate.colorspace:
        return str(plate.colorspace)
    return None


def _resolved_or_verbatim(name: str | None) -> str | None:
    """Resolve a colourspace name against the active OCIO config, softly.

    A name written verbatim into a .nk or .ma only works if the DCC's config
    spells it the same way. When it does not, Nuke does not complain — it
    quietly falls back to its ``scene_linear`` default, which is how a
    delivered script showed ``Read1 (scene_linear)`` despite carrying a name
    (found live 2026-08-07). Routing through OCIO roles and known cross-config
    aliases first removes most of that class of failure.

    Two honest limits:

    * This resolves against the **Atlas-side** config, which is not necessarily
      the DCC's. It is a strict improvement — a role or alias names the SAME
      space, never an approximation — but it is not a guarantee.
    * Anything unresolvable degrades to the artist's own literal. No OIIO, no
      config, or a name this config cannot place must never fail an export;
      a colour NAME is not worth losing a delivery over.

    The import stays inside the function. ``atlas_camera.plate`` is verified
    not to pull the OpenImageIO binding at module level, but ``core`` and
    ``exporters`` have to keep loading in a zero-dependency install, and a
    local import makes that independent of what the package does later.
    """
    if not name:
        return None
    try:
        from atlas_camera.plate import resolve_colorspace
        return resolve_colorspace(name)
    except Exception:  # noqa: BLE001 — unresolvable is not an export failure
        return name


def plate_file_colorspace(path: str | None) -> str | None:
    """The ``oiio:ColorSpace`` a plate file declares about ITSELF, or None.

    ``plate.write_exr`` has always stamped this attribute, and until now no
    consumer read it back — so a plate_ref claiming one space while the file on
    disk declared another produced no signal anywhere. Comparing the two is the
    last chance to catch that before it reaches a comp.

    Every failure path returns None: no OIIO, missing file, unreadable header,
    no attribute. This is provenance, and provenance must never be able to fail
    an export (the same rule the manifest lives under).
    """
    if not path:
        return None
    try:
        from atlas_camera.plate import oiio_available
        if not oiio_available():
            return None
        import OpenImageIO as oiio
        src = oiio.ImageInput.open(str(path))
        if src is None:
            # DRAIN the error. OIIO holds a per-thread pending message and
            # prints "error message that was never retrieved" to stderr if
            # nobody collects it — so a plate that simply is not readable here
            # (the normal case for a path that only resolves on the DCC box)
            # would spew a scary block on every single export.
            oiio.geterror()
            return None
        try:
            declared = src.spec().getattribute("oiio:ColorSpace")
        finally:
            src.close()
        return str(declared) if declared else None
    except Exception:  # noqa: BLE001 — provenance never fails an export
        return None
