"""Temporal generator abstraction for Dynamic Plates (spec §16).

Atlas is never hard-wired to one video model. A generator receives a
DynamicPlate's package (crop + context + matte + prompt + seed) and returns a
frame sequence; its job is to ADD TEMPORAL MOTION — never to change the
camera, reconstruct the world, or reinterpret static geometry (spec §18).

The frame sequence is the contract (spec §21): an MP4 is at most a derivative
preview. Generator absence must degrade to ``status="not_available"`` without
breaking any normal Atlas import (spec §32).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from atlas_camera.core.camera_crop import RegionROI
from atlas_camera.core.dynamic_plate import GENERATOR_NOT_AVAILABLE
from atlas_camera.core.schema import LatentCamera, _json_ready

RESULT_OK = "ok"
RESULT_FAILED = "failed"
RESULT_NOT_AVAILABLE = GENERATOR_NOT_AVAILABLE

# Shipped input mode in v0.1 (spec §19): single cropped source image ->
# image-to-video. Video-to-video over an Atlas-rendered crop sequence is the
# designed future mode, not implemented here.
MODE_IMAGE_TO_VIDEO = "image_to_video"
MODE_VIDEO_TO_VIDEO = "video_to_video"


@dataclass(slots=True)
class TemporalGenerationConfig:
    prompt: str = ""
    seed: int | None = None
    fps: float = 24.0
    frame_count: int = 96
    width: int | None = None      # inference resize; None = native crop size
    height: int | None = None
    mode: str = MODE_IMAGE_TO_VIDEO
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TemporalGenerationResult:
    status: str
    frame_paths: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    generator: str = ""
    model: str = ""
    method: str = ""
    seed: int | None = None
    source_roi: RegionROI | None = None
    crop_camera: LatentCamera | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemporalGenerationResult":
        crop_cam = data.get("crop_camera")
        return cls(
            status=data.get("status", RESULT_FAILED),
            frame_paths=[str(p) for p in data.get("frame_paths", [])],
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            fps=float(data.get("fps", 0.0)),
            frame_count=int(data.get("frame_count", 0)),
            generator=data.get("generator", ""),
            model=data.get("model", ""),
            method=data.get("method", ""),
            seed=data.get("seed"),
            source_roi=RegionROI.from_dict(data.get("source_roi")),
            crop_camera=LatentCamera.from_dict(crop_cam) if crop_cam else None,
            metadata=dict(data.get("metadata", {})),
            warnings=list(data.get("warnings", [])),
        )


@runtime_checkable
class TemporalGenerator(Protocol):
    """A backend that animates a dynamic plate's crop."""

    name: str

    def available(self) -> tuple[bool, str]:
        """(usable, reason). Must never raise, never import heavy deps."""
        ...

    def generate(self, plate: Any, package_dir: Any,
                 config: TemporalGenerationConfig) -> TemporalGenerationResult:
        """Write frames into <package_dir>/generated/ and describe them."""
        ...


class NullGenerator:
    """The explicit no-generator path: every stage before generation still
    runs, and the plate honestly reports ``not_available`` (spec §34)."""

    name = "none"

    def available(self) -> tuple[bool, str]:
        return False, "no generator selected"

    def generate(self, plate: Any, package_dir: Any,
                 config: TemporalGenerationConfig) -> TemporalGenerationResult:
        return TemporalGenerationResult(
            status=RESULT_NOT_AVAILABLE,
            generator=self.name,
            warnings=["No temporal generator selected; plate packaged "
                      "without generated frames."],
        )


def resolve_generator(name: str) -> TemporalGenerator:
    """Look up a generator by registry name (``none`` | ``ltx``)."""
    key = (name or "none").strip().lower()
    if key == "none":
        return NullGenerator()
    if key == "ltx":
        # Lazy import: the adapter module stays stdlib-only, but keep the
        # abstraction module importable even if the adapter grows deps.
        from atlas_camera.dynamic.ltx_comfy import LTXComfyGenerator
        return LTXComfyGenerator()
    raise ValueError(
        f"Unknown temporal generator {name!r}; choices: none, ltx")
