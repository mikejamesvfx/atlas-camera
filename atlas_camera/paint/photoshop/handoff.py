"""Colour-exact plate handoff to and from Photoshop, plus hand-painted mattes.

This is the MANUAL lane, and it is the one that works today: Atlas hands over a
plate, a human paints in Photoshop, Atlas takes it back with the colour intact
and any hand-drawn mattes lifted out as masks. No scripted generative fill is
involved (``syntheticFill`` is not reachable from Action Manager — see
``reports/photoshop_bridge_probe.md``).

Two measured facts define the contract. Both were established on 2026-08-22
against Photoshop 27.11.0, and both are silent failures if ignored:

**Going out — Photoshop ASSUMES an OCIO EXR is ACES2065-1.** It does not read
the file's colourspace tag. Handed a plate tagged ``lin_rec709_scene`` it
applied an exact ``AP0 -> AP1`` matrix (best fit matched the canonical
ACES2065-1 -> ACEScg to 4.6e-5 with residual 0.0). So Atlas converts to
ACES2065-1 before handing over; then Photoshop's assumption is *correct* and
the conversion it performs is the one we wanted. Verified end to end at
**max abs 3e-8** against Atlas's own conversion.

**Coming back — Photoshop's TIFF carries NO colourspace tag.** Read on
``auto``, OIIO infers from the ``.tif`` extension and calls it
``sRGB - Display``, so scene-linear ACEScg data would be treated as display
sRGB and be quietly, badly wrong. The pixels are in the OCIO **working space**
(``Edit > OpenColorIO Settings > Working space``, ACEScg by default), so that
is what ``receive`` assumes and re-tags. Never let ``auto`` near it.

Mattes ride home as extra TIFF channels. Photoshop's channel NAMES do not
survive (``matte_boiler`` becomes ``channel4``), so they are returned in
document order and can be labelled with ``matte_names``. Photoshop also injects
its own transparency alpha as the first extra channel, which is normally empty;
empty channels are skipped by default rather than shipped as blank masks.
"""
from __future__ import annotations

from pathlib import Path

#: What Photoshop assumes an OCIO EXR contains, regardless of its tag.
PHOTOSHOP_ASSUMED_INPUT = "ACES2065-1"
#: Default OCIO working space, and therefore what its export is in.
PHOTOSHOP_DEFAULT_WORKING_SPACE = "ACEScg"


def send(*, plate_path, out_path, assumed_space: str = PHOTOSHOP_ASSUMED_INPUT,
         bit_depth: str = "float") -> dict:
    """Write a plate in the space Photoshop will assume it is in.

    ``bit_depth`` is ``float`` (zip, lossless) because this file is the input
    to a measured round trip; ``half`` selects the lossy dwab codec.
    """
    from atlas_camera.paint.ocio import config_identity
    from atlas_camera.plate.oiio_io import read_plate, write_exr

    out_path = Path(out_path)
    source = read_plate(str(plate_path), raw_data=True)
    converted = read_plate(str(plate_path), output_colorspace=assumed_space)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_exr(str(out_path), converted.pixels, bit_depth=bit_depth,
              source_colorspace=assumed_space,
              extra_attribs={"atlas:handoff_source": str(plate_path),
                             "atlas:handoff_from_space": source.input_colorspace})
    return {
        "out": str(out_path),
        "source": str(plate_path),
        "source_colorspace": source.input_colorspace,
        "sent_as": assumed_space,
        "width": source.width,
        "height": source.height,
        "ocio": config_identity(),
    }


def receive(*, tiff_path, out_path, target_space: str | None = None,
            working_space: str = PHOTOSHOP_DEFAULT_WORKING_SPACE,
            matte_dir=None, matte_names=None, keep_empty_mattes: bool = False,
            bit_depth: str = "float") -> dict:
    """Read Photoshop's export back with the colour intact, and lift the mattes.

    ``target_space`` defaults to ``working_space`` (no conversion, just an
    honest tag). Pass e.g. the original plate's colourspace to land back where
    the plate started.
    """
    import numpy as np
    import OpenImageIO as oiio
    from PIL import Image

    from atlas_camera.paint.ocio import config_identity
    from atlas_camera.plate.oiio_io import (read_plate, resolve_colorspace,
                                            write_exr)

    tiff_path, out_path = Path(tiff_path), Path(out_path)
    target_space = target_space or working_space

    src = oiio.ImageInput.open(str(tiff_path))
    if src is None:
        raise FileNotFoundError(f"could not open {tiff_path}: {oiio.geterror()}")
    spec = src.spec()
    pixels = src.read_image(format="float")
    channel_names = tuple(spec.channelnames)
    src.close()

    if spec.nchannels < 3:
        raise ValueError(
            f"{tiff_path}: expected at least RGB, got {spec.nchannels} channels")

    rgb = np.ascontiguousarray(pixels[..., :3])

    # The file is untagged, so state what it is rather than letting `auto`
    # infer `sRGB - Display` from the .tif extension and mis-convert.
    if resolve_colorspace(working_space) != resolve_colorspace(target_space):
        from OpenImageIO import ImageBuf, ImageBufAlgo, ImageSpec

        buf = ImageBuf(ImageSpec(spec.width, spec.height, 3, "float"))
        buf.set_pixels(oiio.ROI(), rgb)
        conv = ImageBufAlgo.colorconvert(buf, resolve_colorspace(working_space),
                                         resolve_colorspace(target_space))
        if conv.has_error:
            raise RuntimeError(
                f"colour conversion {working_space!r} -> {target_space!r} "
                f"failed: {conv.geterror()}")
        rgb = conv.get_pixels(oiio.FLOAT)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_exr(str(out_path), rgb, bit_depth=bit_depth,
              source_colorspace=target_space,
              extra_attribs={"atlas:received_from": str(tiff_path),
                             "atlas:photoshop_working_space": working_space})

    mattes = extract_mattes(
        pixels, channel_names, matte_dir=matte_dir, matte_names=matte_names,
        keep_empty=keep_empty_mattes)

    return {
        "out": str(out_path),
        "source": str(tiff_path),
        "channels": list(channel_names),
        "working_space": working_space,
        "target_space": target_space,
        "retagged_as": read_plate(str(out_path), raw_data=True).input_colorspace,
        "mattes": mattes,
        "ocio": config_identity(),
    }


def extract_mattes(pixels, channel_names, *, matte_dir=None, matte_names=None,
                   keep_empty: bool = False) -> list:
    """Every channel past RGB, as a mask.

    Photoshop's channel names do not survive into TIFF — a channel the artist
    called ``matte_boiler`` arrives as ``channel4`` — so they come back in
    DOCUMENT ORDER and ``matte_names`` labels them positionally.

    The first extra channel is usually Photoshop's own transparency alpha and
    is empty; empty channels are skipped unless ``keep_empty`` is set, because
    shipping a blank mask downstream reads as "the artist matted nothing"
    rather than "there was nothing here".
    """
    import numpy as np
    from PIL import Image

    names = list(matte_names or [])
    channels = []
    for index in range(3, len(channel_names)):
        data = np.clip(np.ascontiguousarray(pixels[..., index]), 0.0, 1.0)
        channels.append({
            "index": index,
            "file_channel": channel_names[index],
            "coverage": float((data > 0.5).mean()),
            "empty": bool(data.max() <= 0.0),
            "_data": data,
        })

    # Label the KEPT channels in order, not the raw channel index. The artist
    # names what they painted; Photoshop's empty transparency alpha usually
    # sits first and must not consume the first label — it did once, which
    # silently shifted every matte name by one.
    kept = [c for c in channels if keep_empty or not c["empty"]]
    for position, channel in enumerate(kept):
        channel["name"] = (names[position] if position < len(names)
                           else f"matte_{position}")

    out = []
    for channel in channels:
        data = channel.pop("_data")
        record = dict(channel)
        if channel.get("name") is None or "name" not in channel:
            record["name"] = f"(unlabelled {channel['file_channel']})"
        if channel["empty"] and not keep_empty:
            record["skipped"] = ("channel is empty (usually Photoshop's own "
                                 "transparency alpha, not a painted matte)")
            out.append(record)
            continue
        if matte_dir is not None:
            path = Path(matte_dir) / f"{record['name']}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((data * 255.0 + 0.5).astype("uint8")).save(path)
            record["path"] = str(path)
        out.append(record)
    return out
