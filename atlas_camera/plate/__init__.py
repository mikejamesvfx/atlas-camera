"""Colour-managed plate I/O — Atlas's own float image layer.

Atlas is a film tool, so EXR is not an optional extra: it is the format the
whole professional pipeline speaks. Until now that capability came from
opencv's image codec, which turned out to be the least dependable thing in the
dependency tree:

- disabled at RUNTIME by default since OpenCV 4.5.5 (security), needing
  ``OPENCV_IO_ENABLE_OPENEXR=1`` set before the first ``import cv2``;
- absent from the opencv-python 5.x wheels entirely — measured, the wheel
  reports ``OpenEXR: NO`` while every other codec is built in (opencv 5 itself
  did NOT drop EXR; the wheel build did, cf. opencv#26673);
- and shipped by three distributions (``opencv-python``,
  ``opencv-contrib-python``, ``opencv-python-headless``) that all own the same
  ``cv2`` namespace and overwrite each other. Upstream is explicit that
  installing more than one is an undefined error state — and they arrive as
  TRANSITIVE dependencies of unrelated node packs, so no pin of ours can stop
  it.

The consequence: ``OPENCV_IO_ENABLE_OPENEXR=1`` cannot help on a 5.x wheel,
because the codec was never compiled in. Advice to set it sends users down a
dead end.

This package moves float I/O onto **OpenImageIO** — the library the VFX
industry actually uses, from the same foundation as OpenEXR and OpenColorIO —
and leaves opencv to the computer-vision work it is genuinely good at (Hough
transform, Canny, ``remap``), where it has no codec problem.

Two things this buys beyond reliability:

- **Real colour management with no config file.** OIIO ships a built-in ACES
  OCIO config (``ocio://default``), so ACEScg/ACEScct/ACES2065-1 and the display
  spaces are available with no ``opencolorio`` dependency and nothing for the
  user to install. Verified: 0.18 scene-linear through ``sRGB - Display``
  gives 0.4614, the correct sRGB transfer.
- **Proper EXR.** Half/float control, compression, channel names and metadata —
  things opencv's codec does poorly or not at all.

Wheels cover every platform Atlas targets, including Apple Silicon
(``macosx_11_0_arm64``, cp39–cp314), which matters because the arm64 story is
the whole point of the 0.8 line.
"""

from atlas_camera.plate.oiio_io import (  # noqa: F401
    PlateRead,
    auto_colorspace_for_path,
    list_colorspaces,
    oiio_available,
    oiio_diagnostics,
    read_plate,
    write_exr,
)

__all__ = [
    "PlateRead",
    "auto_colorspace_for_path",
    "list_colorspaces",
    "oiio_available",
    "oiio_diagnostics",
    "read_plate",
    "write_exr",
]
