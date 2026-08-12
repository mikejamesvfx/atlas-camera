"""DynamicPlate artifact package writer.

Package layout (Dynamic Plates v0.1 spec §24)::

    dynamic/
    └── WATER_0001/
        ├── manifest.json           # DynamicPlate.to_dict() + created_at etc.
        ├── source/
        │   ├── crop.png            # ROI crop of the source plate
        │   ├── matte.png           # ROI crop of the matte (when supplied)
        │   └── context.png         # wider crop for generator context
        ├── camera/
        │   ├── source_camera.json
        │   └── crop_camera.json
        ├── geometry/
        │   └── receiver.obj
        ├── generated/              # frame_0000.png ... (generator output)
        └── preview/                # optional preview.mp4 (derivative only)

The IMAGE SEQUENCE in ``generated/`` is authoritative; any preview video is a
derivative artifact. ``manifest.json`` IS the artifact contract here, so it
must succeed — unlike the side-car ``atlas_project.json`` convention where a
manifest failure never fails an export.

Needs Pillow + numpy (``pip install -e .[image,vision]``).
"""
from __future__ import annotations

import datetime as _dt
import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas_camera.core.dynamic_plate import (
    DynamicPlate,
    crop_image_region,
    write_plane_obj,
)

_SUBDIRS = ("source", "camera", "geometry", "generated", "preview")


def _require_pil():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Dynamic-plate packaging requires Pillow. Install with:\n"
            "    pip install -e .[image]") from exc
    return Image


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Dynamic-plate packaging requires numpy. Install with:\n"
            "    pip install -e .[vision]") from exc
    return np


@dataclass(slots=True)
class DynamicPlateResult:
    package_dir: Path
    files: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def build_dynamic_plate_package(plate: DynamicPlate, output_dir: Any, *,
                                source_image_path: Any,
                                matte: Any = None,
                                context_pad_frac: float = 0.15,
                                ) -> DynamicPlateResult:
    """Write the on-disk artifact package for a DynamicPlate.

    ``matte`` is a full-resolution HxW array (float [0,1] or uint8); it is
    cropped to the ROI alongside the plate. ``context.png`` is a wider crop
    (ROI expanded by ``context_pad_frac``) so the generator sees pixels past
    the hard matte boundary (spec §14).
    """
    Image = _require_pil()
    np = _require_numpy()

    if plate.source_roi is None:
        raise ValueError("Plate needs a source_roi before packaging")
    roi = plate.source_roi

    package_dir = Path(output_dir) / plate.plate_id
    for sub in _SUBDIRS:
        (package_dir / sub).mkdir(parents=True, exist_ok=True)

    result = DynamicPlateResult(package_dir=package_dir)

    with Image.open(source_image_path) as im:
        source = np.asarray(im.convert("RGB"))
    if source.shape[1] != plate.source_width or \
            source.shape[0] != plate.source_height:
        raise ValueError(
            f"Source image is {source.shape[1]}x{source.shape[0]} but the "
            f"plate records {plate.source_width}x{plate.source_height}")

    crop_path = package_dir / "source" / "crop.png"
    Image.fromarray(crop_image_region(source, roi)).save(crop_path)
    result.files["crop"] = crop_path

    context_roi = roi.expanded(pad_frac=float(context_pad_frac),
                               image_width=plate.source_width,
                               image_height=plate.source_height)
    context_path = package_dir / "source" / "context.png"
    Image.fromarray(crop_image_region(source, context_roi)).save(context_path)
    result.files["context"] = context_path
    plate.metadata.setdefault("context_roi", context_roi.to_dict())
    plate.metadata.setdefault("source_image_path",
                              str(Path(source_image_path).resolve()))

    if matte is not None:
        m = np.asarray(matte)
        if m.ndim == 3:
            m = m[..., 0]
        if m.dtype != np.uint8:
            m = (np.clip(m.astype(np.float32), 0.0, 1.0) * 255).astype(np.uint8)
        matte_path = package_dir / "source" / "matte.png"
        Image.fromarray(crop_image_region(m, roi), mode="L").save(matte_path)
        result.files["matte"] = matte_path
        plate.matte_path = "source/matte.png"

    if plate.source_camera is not None:
        p = package_dir / "camera" / "source_camera.json"
        p.write_text(plate.source_camera.to_json() + "\n", encoding="utf-8")
        result.files["source_camera"] = p
    if plate.crop_camera is not None:
        p = package_dir / "camera" / "crop_camera.json"
        p.write_text(plate.crop_camera.to_json() + "\n", encoding="utf-8")
        result.files["crop_camera"] = p

    if plate.receiver is not None and plate.receiver.primitive is not None:
        rec_path = package_dir / "geometry" / "receiver.obj"
        write_plane_obj(plate.receiver, rec_path)
        plate.receiver.path = "geometry/receiver.obj"
        result.files["receiver"] = rec_path

    manifest = plate.to_dict()
    manifest["created_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    try:
        from atlas_camera import __version__ as _atlas_version
    except Exception:  # pragma: no cover - defensive
        _atlas_version = "unknown"
    manifest["atlas_version"] = _atlas_version
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    result.files["manifest"] = manifest_path
    return result


def load_dynamic_plate(package_dir: Any) -> DynamicPlate:
    """Rehydrate a DynamicPlate from its package's manifest.json."""
    manifest_path = Path(package_dir) / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return DynamicPlate.from_dict(data)
