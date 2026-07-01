"""DCC and package exporters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path
    from atlas_camera.core.schema import AtlasSolve

from atlas_camera.exporters.review_package import build_review_package

__all__ = ["build_review_package", "DccExporter"]


class DccExporter(Protocol):
    """Common interface for single-output DCC script exporters.

    Implemented by MayaExporter, BlenderExporter, and NukeExporter.
    USDExporter produces multiple output files and is not part of this protocol.
    """

    def write_scene(self, solve: "AtlasSolve", output_path: "str | Path") -> "Path": ...
