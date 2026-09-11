"""The ``lens`` block of a matrixZone shot manifest (``matrixZone/1.2``).

Atlas Camera owns the lens. The Unreal side (Atlas Bridge, Prep Shot) and the
generator never compute distortion; they carry this record and apply the maps
it names. So the block is written HERE, from the import that undistorted the
plate, and everything in it is either verbatim from that import or measured
from its own remap grid. Nothing is reasoned from a lens model downstream.

Contract (``Docs/matrixzone.schema.json`` in ``atlas-unreal``, ``lens``):

- ``undistortStatus``: the importer's own word for what happened.
- ``space``: ``rectilinear`` when the plate is safe to treat as a pinhole
  image (``applied``, ``camera_processed``); ``distorted-uncorrected`` for
  every other status, which Prep Shot refuses without a human acknowledging.
- ``excursionPx``: how far the rectilinear frame must extend beyond the plate,
  per edge, for every distorted pixel to have a source. Measured with the
  lens's own correction (:func:`undistort.measure_excursion`). This is what
  sizes ``render.overscan`` on the Bridge side.
- ``maps``: the ST maps for the two trips through the lens, in the shipping
  conventions of :mod:`redistort` (float32, ``u,v,0,alpha``, normalised,
  bottom-left origin), each saying which frame it lives on and which it
  samples.

Two ways to call it. Without a render window the maps are plate-sized, which
is what the Nuke export ships today. With one — the padded, overscanned frame
Prep Shot renders — the redistort map samples the render, the undistort map
lives on it, and ``outsideFraction`` on the redistort map is the check that the
overscan covered the lens.
"""

from __future__ import annotations

import math
import os
from typing import Any

LENS_BLOCK_SCHEMA = "matrixZone/1.2"

STATUSES = ("applied", "camera_processed", "disabled", "no_lens_metadata",
            "no_profile_camera", "no_profile_lens", "lensfunpy_missing")
RECTILINEAR = ("applied", "camera_processed")


def lens_space(status: str) -> str:
    """``rectilinear`` or ``distorted-uncorrected`` for an importer status."""
    return "rectilinear" if status in RECTILINEAR else "distorted-uncorrected"


def plan_render(plate_width: int, plate_height: int,
                excursion: dict[str, float] | None, *,
                margin: tuple[int, int, int, int] = (0, 0, 0, 0),
                clean: int = 128) -> dict[str, Any]:
    """Size the padded render around a plate: the larger of the lens excursion
    and the pipeline margin on each edge, then padded so width and height are
    multiples of ``clean`` (128: so a quarter-linear base and a half-linear
    doubling are both whole pixels and 32-clean).

    Returns ``{"width", "height", "plateOrigin", "overscan"}`` with
    ``overscan = [left, top, right, bottom]`` as the FINAL extension per edge,
    padding included — so ``plateOrigin == overscan[:2]`` and
    ``width == plate_width + overscan[0] + overscan[2]`` are invariants a
    validator can check rather than conventions someone has to remember.
    Extra padding is split between the two edges of an axis, the odd pixel
    going to the right/bottom.
    """
    exc = excursion or {}
    need = [int(math.ceil(float(exc.get(k, 0.0)))) for k in
            ("left", "top", "right", "bottom")]
    edges = [max(n, int(m)) for n, m in zip(need, margin)]
    left, top, right, bottom = edges
    width = plate_width + left + right
    height = plate_height + top + bottom
    padw = (-width) % clean
    padh = (-height) % clean
    left += padw // 2
    right += padw - padw // 2
    top += padh // 2
    bottom += padh - padh // 2
    return {
        "width": plate_width + left + right,
        "height": plate_height + top + bottom,
        "plateOrigin": [left, top],
        "overscan": [left, top, right, bottom],
    }


def _map_entry(filename: str, info: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "direction": info["direction"],
        "file": filename,
        "domain": info["domain"],
        "samples": info["samples"],
        "size": [int(info["width"]), int(info["height"])],
        "origin": "bottom-left",
        "channels": info["channels"],
        "outsideFraction": float(info["outside_fraction"]),
    }
    if info["channels"].endswith("alpha"):
        entry["alphaMeans"] = "source-inside"
    if info["direction"] == "redistort":
        entry["inversionResidualPx"] = float(info["inversion_residual_px"])
        entry["inversionResidualP999Px"] = float(
            info["inversion_residual_p999_px"])
        entry["converged"] = bool(info["converged"])
    return entry


def build_lens_block(result: Any, out_dir: str | None, *,
                     render: dict[str, Any] | None = None,
                     write_undistort: bool = True,
                     iterations: int = 12,
                     redistort_precomputed: tuple[Any, dict[str, Any], str] | None = None,
                     ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build (and write) the ``lens`` block for a :class:`RawImportResult`.

    ``render`` is ``{"width", "height", "plateOrigin": [x, y]}`` — the padded
    frame the shot is rendered at — or None for plate-sized maps.
    ``redistort_precomputed`` lets an exporter that has already built and
    written the plate-sized redistort map hand it in as
    ``(stmap, info, filename)`` instead of paying for the inversion twice; it
    is ignored when ``render`` is given, because that map samples a different
    frame.

    Returns ``(block, info)``. ``info`` carries the per-map builder dicts and
    the measured excursion for the caller's report; the block is the record.
    Maps are only written when ``out_dir`` is given and the plate was actually
    undistorted; the other statuses carry no maps by definition.
    """
    status = str(getattr(result, "undistort_status", "disabled"))
    if status not in STATUSES:
        raise ValueError(f"unknown undistort_status {status!r}")
    block: dict[str, Any] = {
        "source": "atlas-camera",
        "undistortStatus": status,
        "space": lens_space(status),
        "maps": [],
    }
    dist = dict(getattr(result, "distortion", {}) or {})
    if dist:
        block["distortion"] = dist
    info: dict[str, Any] = {"status": status, "maps": {}}

    if status == "camera_processed":
        block["excursionPx"] = {"left": 0.0, "top": 0.0,
                                "right": 0.0, "bottom": 0.0}
        return block, info
    if status != "applied" or not getattr(result, "undistort_applied", False):
        return block, info

    src = getattr(result, "source_path", None)
    if not src:
        raise RuntimeError(
            "cannot build the lens block without the RAW: the lens profile is "
            "re-read from source_path")
    from atlas_camera.raw.metadata import read_raw_metadata
    from atlas_camera.raw.undistort import (
        build_undistort_map, extend_undistort_map, measure_excursion)
    from atlas_camera.raw.redistort import (
        build_redistort_stmap, build_undistort_stmap, write_stmap_exr)

    width, height = int(result.width), int(result.height)
    meta = read_raw_metadata(str(src))
    und = build_undistort_map(meta, width, height)
    if und.coords is None:
        raise RuntimeError(f"lens profile no longer resolves ({und.status})")
    if und.lens_name:
        block["profile"] = (f"{und.lens_name} on {und.cam_name}"
                            if und.cam_name else str(und.lens_name))

    excursion = measure_excursion(meta, width, height)
    block["excursionPx"] = {k: float(excursion[k])
                            for k in ("left", "top", "right", "bottom")}
    info["excursion_px"] = block["excursionPx"]

    if render is not None:
        rw, rh = int(render["width"]), int(render["height"])
        ox, oy = (int(v) for v in render["plateOrigin"])
        overscan = (ox, oy, rw - width - ox, rh - height - oy)
        if min(overscan) < 0:
            raise ValueError(
                f"plate {width}x{height} at {render['plateOrigin']} does not "
                f"fit in render {rw}x{rh}")
        short = [k for k, o in zip(("left", "top", "right", "bottom"), overscan)
                 if o < excursion[k]]
        if short:
            raise ValueError(
                f"render overscan is smaller than the lens excursion on "
                f"{', '.join(short)}: {overscan} vs {block['excursionPx']}")
        grid = extend_undistort_map(meta, width, height, und.coords, overscan)
        plate_size = (width, height)
        plate_origin = (ox, oy)
    else:
        grid = und.coords
        plate_size = None
        plate_origin = (0, 0)

    if out_dir is None:
        return block, info
    os.makedirs(out_dir, exist_ok=True)

    if render is None and redistort_precomputed is not None:
        stmap, rinfo, fname = redistort_precomputed
        rinfo = dict(rinfo)
        rinfo.setdefault("direction", "redistort")
        rinfo.setdefault("domain", "plate")
        rinfo.setdefault("samples", "plate")
    else:
        stmap, rinfo = build_redistort_stmap(
            grid, iterations=iterations, plate_size=plate_size,
            plate_origin=plate_origin)
        fname = "redistort_stmap.exr"
        write_stmap_exr(stmap, os.path.join(out_dir, fname),
                        content="redistort_stmap")
    block["maps"].append(_map_entry(fname, rinfo))
    info["maps"]["redistort"] = rinfo

    if write_undistort:
        ustmap, uinfo = build_undistort_stmap(grid, plate_size=plate_size)
        ufname = "undistort_stmap.exr"
        write_stmap_exr(ustmap, os.path.join(out_dir, ufname),
                        content="undistort_stmap")
        block["maps"].append(_map_entry(ufname, uinfo))
        info["maps"]["undistort"] = uinfo

    return block, info


def write_lens_block_json(block: dict[str, Any], path: str) -> str:
    """Write the block on its own, for pipelines that assemble the shot
    manifest elsewhere (Prep Shot reads this file and inlines it)."""
    import json
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"schema": LENS_BLOCK_SCHEMA, "lens": block}, fh, indent=2)
    return path
