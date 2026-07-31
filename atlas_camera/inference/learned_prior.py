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

from atlas_camera.inference._common import bounded_cache_set, resolve_device


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
    #: First radial distortion coefficient, in the normalised-coordinate
    #: convention ``x_distorted = x_undistorted * (1 + k1 * r_u**2)`` where
    #: ``r_u`` is measured in focal-length units from the principal point.
    #: ``None`` for the pinhole model, which does not estimate one.
    #:
    #: Found live 2026-07-31: GeoCalib's ``distorted`` weights DO return a k1
    #: (a screen-grabbed plate measured -0.006633), and this class had nowhere
    #: to put it — so asking for the distorted model got you a differently
    #: solved camera with its distortion term silently discarded. For a plate
    #: with no EXIF, lensfun cannot help and this estimate is the only one
    #: available.
    k1: float | None = None
    source_model: str = "geocalib"
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Module-level model cache: GeoCalib weights load once and are reused across calls
# (the ComfyUI node solves many images per session). Bounded to avoid
# unbounded VRAM growth across a long session cycling through weights/devices.
_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_CACHE_MAX = 4


# Fixed so that solving the same image twice returns the same camera. The exact
# value is arbitrary; only its stability matters.
_CALIBRATE_SEED = 0


def _fork_devices(torch: Any, device: str) -> list:
    """CUDA devices whose RNG fork_rng must also save/restore (empty on CPU —
    passing a CUDA index there would force a needless CUDA init)."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return []
    index = torch.device(device).index
    return [torch.cuda.current_device() if index is None else index]


def _get_model(weights: str, device: str) -> Any:
    key = (weights, device)
    model = _MODEL_CACHE.get(key)
    if model is None:
        _, GeoCalib = _require_geocalib()
        model = GeoCalib(weights=weights).to(device)
        # Inference posture. GeoCalib ships the module in its default
        # training=True state (21 Dropout + 47 BatchNorm submodules); nothing
        # here ever trains it.
        model.eval()
        bounded_cache_set(_MODEL_CACHE, key, model, _MODEL_CACHE_MAX)
    return model


#: GeoCalib weight set -> the camera model `calibrate()` should fit with it.
#: Loading distortion-aware weights while fitting a pinhole camera throws the
#: distortion away, so these two must be chosen together. Keys not listed fall
#: back to "pinhole"; GeoCalib also ships `radial` and `simple_divisional`,
#: which Atlas has no consumer for yet — `undistort_pixel` implements
#: simple_radial only, so fitting one of the others would produce a coefficient
#: nothing could apply.
_CAMERA_MODEL_FOR_WEIGHTS: dict[str, str] = {
    "pinhole": "pinhole",
    "distorted": "simple_radial",
}


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
    device = resolve_device(device, torch)

    model = _get_model(weights, device)
    image = model.load_image(str(image_path)).to(device)

    # WEIGHTS AND CAMERA MODEL ARE TWO SEPARATE CHOICES, and only setting the
    # first is a silent no-op. `GeoCalib(weights="distorted")` loads a network
    # trained to see distortion, but `calibrate()` defaults to
    # `camera_model="pinhole"` — which has no k1 to fit, so the fitted camera
    # comes back distortion-free. Found live 2026-07-31: asking for the
    # distorted model produced a differently-solved camera (focal 1458.7 vs
    # 1442.7, pitch -32.70 vs -33.22) with NO distortion term, which reads
    # exactly like "this lens has no distortion" rather than "you never asked
    # for one".
    camera_model = _CAMERA_MODEL_FOR_WEIGHTS.get(weights, "pinhole")
    # GeoCalib's calibrate() draws from torch's global RNG, so the SAME image
    # solved twice used to return a different camera: measured on one plate,
    # focal spread ~2.5% (1613-1653 px) and roll ~0.7 deg (-1.50 to -2.22) over
    # four consecutive runs, on both CPU and CUDA. For a camera-solve tool that
    # is a correctness problem, not noise: roll is exactly what AtlasRollTrim
    # exists to dial in by hand, and every downstream metric rides the solve.
    #
    # fork_rng, NOT a bare torch.manual_seed(): this runs inside ComfyUI, where
    # clobbering the global RNG would silently change any sampler seeded later
    # in the same queue. Forking restores the caller's RNG state on exit, so the
    # determinism is scoped strictly to this call.
    with torch.no_grad(), torch.random.fork_rng(devices=_fork_devices(torch, device)):
        torch.manual_seed(_CALIBRATE_SEED)
        result = model.calibrate(image, camera_model=camera_model)

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

    # k1 lives on the camera object, not in `result`, and only the distortion
    # models carry it — a pinhole camera has no such attribute at all.
    #
    # NOT rescaled to native resolution, unlike focal. k1 multiplies r**2 in
    # FOCAL-LENGTH units, so it is already resolution-independent; scaling it
    # alongside the focal would apply the correction twice.
    k1_value: float | None = None
    raw_k1 = getattr(cam, "k1", None)
    if raw_k1 is not None:
        try:
            k1_value = float(raw_k1.detach().cpu().numpy().reshape(-1)[0])
        except AttributeError:
            k1_value = float(raw_k1)
        if not math.isfinite(k1_value):
            k1_value = None

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
        k1=k1_value,
        source_model=f"geocalib:{weights}",
        raw={
            "vfov_deg": vfov_deg,
            "processed_size": [width, height],
            "gravity_native": [float(v) for v in gravity_vec],
        },
    )
