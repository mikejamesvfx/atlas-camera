"""Camera RAW import (NEF / CR2 / CR3 / RAF / ARW) — optional [raw] extra.

Import of this package is dependency-free; the heavy pieces (rawpy decode,
lensfunpy undistort) import lazily inside their functions with actionable
errors, matching the repo-wide guarded-import doctrine.
"""

from __future__ import annotations

from atlas_camera.raw.metadata import (
    RawMetadata,
    SensorResolution,
    read_raw_metadata,
    resolve_sensor_size,
)

__all__ = [
    "RawImportResult",
    "RawMetadata",
    "SensorResolution",
    "import_raw",
    "read_raw_metadata",
    "resolve_sensor_size",
]


def __getattr__(name):  # lazy: pipeline pulls numpy, keep bare import light
    if name in ("RawImportResult", "import_raw"):
        from atlas_camera.raw import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
