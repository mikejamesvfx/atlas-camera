"""Learned single-image camera prior (GeoCalib).

This is an *optional* neural front-end for the solver. Classical vanishing-point
detection is fragile on AI-generated imagery, whose perspective is only locally
consistent — multi-line RANSAC latches onto contradictory edges and returns a
plausible-looking but wrong camera (often pitched the wrong way). A learned prior
predicts the camera's focal length and gravity (up-vector) directly from image
content and degrades gracefully instead of failing, which is exactly what AI
renders need.

The heavy dependencies (torch + geocalib) are imported lazily so the core package
stays dependency-free. Install with:  pip install -e .[neural]

The public result, :class:`CameraPrior`, is a pure-Python dataclass (no torch
objects) so it can cross the boundary into `atlas_camera.core` without dragging
torch into the DCC-agnostic layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _require_geocalib() -> tuple[Any, Any]:
    """Import torch + geocalib lazily with an informative error."""
    try:
        import torch
        from geocalib import GeoCalib
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "The learned camera prior requires torch and geocalib. Install with:\n"
            "    pip install -e .[neural]\n"
            "(geocalib is fetched from GitHub: "
            "pip install 'git+https://github.com/cvg/GeoCalib.git')"
        ) from exc
    return torch, GeoCalib


@dataclass(slots=True)
class CameraPrior:
    """Camera parameters predicted from a single image by a learned model.

    Angles are degrees. ``up_cam`` is the world *up* direction expressed in Atlas
    camera coordinates (x-right, y-up, z-back) — already converted out of the
    model's native (gravity/down, OpenCV) convention. ``*_uncertainty`` fields
    are the model's own predicted standard deviations (degrees / pixels) and
    drive real confidence.
    """

    focal_px: float
    fov_h_deg: float
    fov_v_deg: float
    roll_deg: float
    pitch_deg: float
    up_cam: tuple[float, float, float]
    principal_point_px: tuple[float, float]
    image_width: int
    image_height: int
    roll_uncertainty_deg: float | None = None
    pitch_uncertainty_deg: float | None = None
    focal_uncertainty_px: float | None = None
    source_model: str = "geocalib"
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Module-level model cache: GeoCalib weights load once and are reused across calls
# (the ComfyUI node solves many images per session).
_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _get_model(weights: str, device: str) -> Any:
    key = (weights, device)
    model = _MODEL_CACHE.get(key)
    if model is None:
        _, GeoCalib = _require_geocalib()
        model = GeoCalib(weights=weights).to(device)
        _MODEL_CACHE[key] = model
    return model


def _gravity_to_atlas_up(gravity_vec: Any) -> tuple[float, float, float]:
    """Convert GeoCalib gravity (native cam coords) to Atlas world-up in cam coords.

    GeoCalib uses an OpenCV-style camera frame (x-right, y-down, z-forward). Atlas
    uses x-right, y-up, z-back, so we flip Y and Z. World *up* is the negation of
    the (down-pointing) gravity vector; we orient it so +Y is up in the image.
    """
    gx, gy, gz = (float(v) for v in gravity_vec)
    # native gravity -> Atlas camera frame (flip Y and Z)
    g_atlas = (gx, -gy, -gz)
    up = [-g_atlas[0], -g_atlas[1], -g_atlas[2]]
    norm = (up[0] ** 2 + up[1] ** 2 + up[2] ** 2) ** 0.5 or 1.0
    up = [c / norm for c in up]
    if up[1] < 0:  # ensure +Y points up in the image
        up = [-c for c in up]
    return (up[0], up[1], up[2])


def estimate_camera_prior(
    image_path: str | Path,
    *,
    device: str | None = None,
    weights: str = "pinhole",
) -> CameraPrior:
    """Predict a :class:`CameraPrior` from a single image with GeoCalib.

    ``weights`` is ``"pinhole"`` (no lens distortion, best for clean AI renders)
    or one of GeoCalib's distortion models. Focal is reported in pixels at the
    image's *native* resolution (GeoCalib works at a fixed size internally; we
    rescale via the resolution-independent field of view).
    """
    import math

    torch, _ = _require_geocalib()
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = _get_model(weights, device)
    image = model.load_image(str(image_path)).to(device)
    with torch.no_grad():
        result = model.calibrate(image)

    cam = result["camera"]
    grav = result["gravity"]

    # Native size the model produced its estimate at (w, h).
    proc_w, proc_h = (float(v) for v in cam.size.detach().cpu().numpy()[0])
    vfov_deg = float(torch.rad2deg(cam.vfov).detach().cpu().numpy().reshape(-1)[0])
    roll_deg, pitch_deg = (
        float(v) for v in torch.rad2deg(grav.rp).detach().cpu().numpy().reshape(-1)
    )
    gravity_vec = grav.vec3d.detach().cpu().numpy().reshape(-1)[:3]

    # Focal is resolution-dependent; carry it to native resolution using vfov,
    # which is not. (fx == fy for a pinhole model with square pixels.)
    width, height = int(round(proc_w)), int(round(proc_h))
    focal_px = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    fov_h_deg = 2.0 * math.degrees(math.atan((width / 2.0) / focal_px))

    def _scalar(name: str) -> float | None:
        t = result.get(name)
        if t is None:
            return None
        return float(t.detach().cpu().numpy().reshape(-1)[0])

    return CameraPrior(
        focal_px=focal_px,
        fov_h_deg=fov_h_deg,
        fov_v_deg=vfov_deg,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        up_cam=_gravity_to_atlas_up(gravity_vec),
        principal_point_px=(width / 2.0, height / 2.0),
        image_width=width,
        image_height=height,
        roll_uncertainty_deg=_scalar("roll_uncertainty"),
        pitch_uncertainty_deg=_scalar("pitch_uncertainty"),
        focal_uncertainty_px=_scalar("focal_uncertainty"),
        source_model=f"geocalib:{weights}",
        raw={
            "vfov_deg": vfov_deg,
            "processed_size": [width, height],
            "gravity_native": [float(v) for v in gravity_vec],
        },
    )
