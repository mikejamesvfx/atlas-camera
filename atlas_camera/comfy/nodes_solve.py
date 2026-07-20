"""Atlas ComfyUI nodes — solve group.

Extracted verbatim from nodes.py during modularization; no behavior
change. Registered/exported via atlas_camera.comfy.node_registry.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, NamedTuple
from atlas_camera.core.io import load_solve_json, save_solve_json
from atlas_camera.core.solver import solve_from_constraints, solve_still_image
from atlas_camera.importers.usd_camera_loader import USDCameraLoader

from atlas_camera.comfy.node_helpers import (
    _ATLAS_ASSESS_CACHE,
    _DEPTH_MODEL_CHOICES,
    _clone_solve_with_metadata,
    _execution_blocker,
    _extrinsics_from_view,
    _image_fingerprint,
    _image_tensor_to_preview_b64,
    _recompute_horizon_line,
    _reference_id_choices,
    _require_numpy,
    _require_torch,
    _resolve_raw_hints,
    _save_image_tensor_to_tmp,
    _solve_fingerprint,
    _stamp_raw_provenance,
)
from atlas_camera.comfy.nodes_viewport import AtlasDebugReport



# ---------------------------------------------------------------------------
# Existing nodes (unchanged)
# ---------------------------------------------------------------------------

class AtlasLoadImageSolveCamera:
    """DEPRECATED — file-path-based solve kept only so saved workflows load.

    Prefer AtlasSolveFromImage (geometric VP solve) or AtlasLearnedSolveFromImage
    (GeoCalib prior) — both take an IMAGE tensor and sit in a normal image chain.
    """

    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "solve"
    CATEGORY = "Atlas Camera"
    DEPRECATED = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": ""}),
                "image_width": ("INT", {"default": 0, "min": 0,
                                        "tooltip": "0 = auto (read from the image file)"}),
                "image_height": ("INT", {"default": 0, "min": 0,
                                         "tooltip": "0 = auto (read from the image file)"}),
            },
            "optional": {
                "focal_length_mm": ("FLOAT", {"default": 35.0, "min": 0.0}),
                "sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 0.01}),
            },
        }

    def solve(self, image_path, image_width, image_height,
              focal_length_mm=None, sensor_width_mm=36.0):
        import logging
        logging.warning(
            "AtlasLoadImageSolveCamera is deprecated — use AtlasSolveFromImage "
            "or AtlasLearnedSolveFromImage (IMAGE-tensor inputs) instead.")
        hints = {}
        if focal_length_mm:
            hints["focal_length_mm"] = focal_length_mm
            hints["sensor_width_mm"] = sensor_width_mm
        # 0×0 → let the solver read the image's true dimensions from the file.
        image_size = (image_width, image_height) if (image_width and image_height) else None
        return (solve_still_image(image_path,
                                  image_size=image_size,
                                  intrinsics_hint=hints,
                                  detect_vanishing_points=True),)


class AtlasUSDCameraLoader:
    RETURN_TYPES = ("ATLAS_CAMERA",)
    FUNCTION = "load"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "usd_path": ("STRING", {"default": ""}),
                "image_width": ("INT", {"default": 1920, "min": 1}),
                "image_height": ("INT", {"default": 1080, "min": 1}),
            }
        }

    def load(self, usd_path, image_width, image_height):
        return (USDCameraLoader().load(usd_path, image_size=(image_width, image_height)),)


class AtlasRegisterPlate:
    """Register a projection plate for float-safe final handoff.

    The IMAGE passes through unchanged. The ATLAS_PLATE_REF carries a durable
    file path/colorspace when supplied; if no path is supplied it is explicitly
    marked proxy-only so exporters do not mistake a browser/JPEG preview for
    final EXR data.
    """

    RETURN_TYPES = ("IMAGE", "ATLAS_PLATE_REF")
    RETURN_NAMES = ("image", "plate_ref")
    FUNCTION = "register"
    CATEGORY = "Atlas Camera/Color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "plate_path": ("STRING", {"default": "",
                    "tooltip": "Original/final plate path, ideally EXR 16f/32f. Leave blank to mark this as proxy-only."}),
                "colorspace": ("STRING", {"default": "ACEScg",
                    "tooltip": "Source plate colorspace for Nuke/Maya/OCIO handoff."}),
                "bit_depth": ("STRING", {"default": "auto",
                    "tooltip": "Descriptive bit depth, e.g. 16f, 32f, 10-bit, 8-bit, or auto."}),
                "role": (["source", "patch", "clean_plate", "matte", "proxy"], {"default": "source"}),
                "lut_path": ("STRING", {"default": ""}),
            },
        }

    def register(self, image, plate_path="", colorspace="ACEScg", bit_depth="auto", role="source", lut_path=""):
        from atlas_camera.core.schema import AtlasPlateRef

        path = str(plate_path or "").strip() or None
        suffix = Path(path).suffix.lower() if path else ""
        inferred_depth = bit_depth if bit_depth and bit_depth != "auto" else (
            "16f/32f" if suffix == ".exr" else "8-bit/proxy"
        )
        plate_ref = AtlasPlateRef(
            image_path=path,
            preview_b64=_image_tensor_to_preview_b64(image, quality=85),
            colorspace=colorspace or "ACEScg",
            bit_depth=inferred_depth,
            role=role or "source",
            is_proxy=not bool(path),
            lut_path=(str(lut_path).strip() or None),
            metadata={
                "path_exists": bool(path and Path(path).is_file()),
                "registered_from": "AtlasRegisterPlate",
            },
        )
        return (image, plate_ref)


def _plate_colorspace_choices(include_auto: bool = True) -> list:
    """Colourspace combo contents, safe to call at NODE REGISTRATION time.

    INPUT_TYPES runs when ComfyUI imports the pack, long before anyone has
    decided to use this node, so it must never raise or import OpenImageIO
    eagerly — a missing [oiio] extra has to degrade to a usable node that
    explains itself, not break the whole pack's registration.
    """
    head = ["auto"] if include_auto else []
    try:
        from atlas_camera.plate import list_colorspaces, oiio_available

        if oiio_available():
            spaces = list_colorspaces()
            if spaces:
                return head + spaces
    except Exception:  # noqa: BLE001 — registration must survive anything
        pass
    # Names from OIIO's built-in ACES config, so a graph authored without the
    # extra installed still carries valid values once it IS installed.
    return head + ["ACEScg", "ACES2065-1", "ACEScct", "sRGB - Display",
                   "Rec.1886 Rec.709 - Display", "Linear Rec.709 (sRGB)"]


class AtlasLoadPlate:
    """🎞 Colour-managed plate loader — Atlas's own float reader (OpenImageIO).

    Reads EXR/DPX/TIFF/PNG/JPEG as float and converts colour through OCIO,
    replacing the third-party OCIORead in the colour-managed path so Atlas owns
    the float pipeline end to end.

    Why not opencv: its EXR codec is disabled at runtime by default, is absent
    from the opencv-python 5.x wheels entirely, and is shipped by three
    distributions that overwrite each other and arrive as transitive deps of
    unrelated node packs. OpenImageIO is the VFX-industry library and carries a
    BUILT-IN ACES OCIO config, so ACEScg/ACEScct/ACES2065-1 work with nothing
    else installed and no $OCIO to configure ($OCIO is honoured when set).

    Outputs a ready `plate_ref`, so this one node replaces OCIORead +
    AtlasRegisterPlate — and the ref carries the colourspace and bit depth read
    from the FILE rather than typed by hand, which is what the DCC exporters
    want.

    `raw_data` is Nuke's "raw data": pass the file's values through untouched.
    Use it for DATA passes (depth, normals, mattes, UV) — colour-converting
    those corrupts them silently.

    ASSOCIATED ALPHA: EXR alpha is premultiplied, and the conversion correctly
    unpremultiplies, converts, then re-premultiplies. So 0.18 at alpha 0.5
    reads back as 0.317, not 0.461 — that is right, not a bug. Applying a
    transfer function straight to premultiplied values looks merely "a bit
    dark" rather than obviously broken, which is why it is worth stating.

    Needs `[oiio]`.
    """

    RETURN_TYPES = ("IMAGE", "MASK", "ATLAS_PLATE_REF", "STRING")
    RETURN_NAMES = ("image", "alpha", "plate_ref", "report")
    FUNCTION = "load"
    CATEGORY = "Atlas Camera/Color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "",
                    "tooltip": "Path to an EXR/DPX/TIFF/PNG/JPEG plate."}),
            },
            "optional": {
                "input_colorspace": (_plate_colorspace_choices(), {"default": "auto",
                    "tooltip": "What the FILE holds. 'auto' infers from the extension "
                               "(EXR/DPX -> ACEScg, 8-bit -> sRGB - Display). A file that "
                               "records its own colourspace always wins over this."}),
                "output_colorspace": (_plate_colorspace_choices(include_auto=False),
                    {"default": "sRGB - Display",
                     "tooltip": "What the IMAGE output should be in. ComfyUI's working "
                                "space is display-referred sRGB."}),
                "raw_data": ("BOOLEAN", {"default": False,
                    "tooltip": "Nuke 'raw data': skip colour conversion entirely. REQUIRED "
                               "for data passes (depth/normals/mattes) — converting data as "
                               "if it were colour corrupts it silently."}),
            },
        }

    def load(self, file_path, input_colorspace="auto",
             output_colorspace="sRGB - Display", raw_data=False, **_extra):
        from atlas_camera.core.schema import AtlasPlateRef
        from atlas_camera.plate import oiio_diagnostics, read_plate

        torch = _require_torch()
        np = _require_numpy()

        path = str(file_path or "").strip()
        if not path:
            raise RuntimeError("AtlasLoadPlate needs a file_path.")
        if not Path(path).is_file():
            raise RuntimeError(f"AtlasLoadPlate: no such file: {path}")

        result = read_plate(path, input_colorspace=input_colorspace,
                            output_colorspace=output_colorspace, raw_data=bool(raw_data))

        image = torch.from_numpy(np.ascontiguousarray(result.pixels)).unsqueeze(0)
        if result.alpha is not None:
            alpha = torch.from_numpy(np.ascontiguousarray(result.alpha)).unsqueeze(0)
        else:
            alpha = torch.ones((1, result.height, result.width), dtype=torch.float32)

        # The ref records what the FILE is, not what the preview tensor became —
        # the exporters need the original for the DCC handoff.
        plate_ref = AtlasPlateRef(
            image_path=path,
            preview_b64=_image_tensor_to_preview_b64(image, quality=85),
            colorspace=result.input_colorspace or "unknown",
            bit_depth=result.file_bit_depth or "unknown",
            role="source",
            is_proxy=not result.is_float,
            lut_path=None,
            metadata={"reader": "openimageio", "file_format": result.file_format,
                      "channels": list(result.channel_names)},
        )

        report = (f"{Path(path).name} · {result.summary()}\n"
                  f"backend: {oiio_diagnostics()}")
        if not result.is_float:
            report += ("\nNOTE: this is an integer file, so plate_ref is marked proxy — "
                       "wire a float EXR/DPX for a true colour-managed DCC handoff.")
        return (image, alpha, plate_ref, report)


class AtlasAttachSourcePlate:
    """Attach a registered source plate to an Atlas solve."""

    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "attach"
    CATEGORY = "Atlas Camera/Color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "plate_ref": ("ATLAS_PLATE_REF",),
            },
        }

    def attach(self, solve, plate_ref):
        return (_clone_solve_with_metadata(solve, source_plate=plate_ref),)


class AtlasLoadRAW:
    """📷 Camera RAW loader (NEF / CR2 / CR3 / RAF / ARW) — [raw] extra.

    One node replaces the ACR round-trip: rawpy demosaic -> IMAGE tensor for
    solve/preview, EXIF focal + camera-model->sensor lookup -> `raw_meta`
    (wire into a solve node's raw_meta input so the solve stops guessing
    intrinsics), optional lensfun undistort ([raw-lens]), and a scene-linear
    EXR sidecar + ATLAS_PLATE_REF so RAW slots into the OCIO Output Desk path
    exactly where OCIORead does. The EXR and the tensor share one demosaic
    and one undistort grid — geometrically identical by construction.
    """

    RETURN_TYPES = ("IMAGE", "ATLAS_PLATE_REF", "ATLAS_RAW_META", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "plate_ref", "raw_meta", "focal_length_mm",
                    "sensor_width_mm", "report")
    FUNCTION = "load"
    CATEGORY = "Atlas Camera/Color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "",
                    "tooltip": "Path to a camera RAW file (.nef .cr2 .cr3 .raf .arw .dng)."}),
            },
            "optional": {
                # Widget order below is FROZEN (positional serialization) —
                # new widgets append at the end only.
                "undistort": ("BOOLEAN", {"default": True,
                    "tooltip": "Lensfun geometry correction from the EXIF lens model "
                               "([raw-lens] extra). Skipped with a report line when no "
                               "profile matches (common for Fuji X — in-body corrections)."}),
                "half_size": ("BOOLEAN", {"default": False,
                    "tooltip": "Half-resolution demosaic for fast iteration on 36-100MP files."}),
                "white_balance": (["camera", "auto"], {"default": "camera"}),
                "exposure_ev": ("FLOAT", {"default": 0.0, "min": -6.0, "max": 6.0,
                                          "step": 0.1}),
                "write_exr": ("BOOLEAN", {"default": True,
                    "tooltip": "Write a scene-linear EXR sidecar and reference it in "
                               "plate_ref (needs opencv 4.x + OPENCV_IO_ENABLE_OPENEXR=1 "
                               "set before ComfyUI starts — same constraint as the OCIO "
                               "path). On failure the plate_ref degrades to proxy."}),
                "output_dir": ("STRING", {"default": "atlas_exports/raw_plates"}),
                "colorspace": ("STRING", {"default": "Linear Rec.709 (sRGB)",
                    "tooltip": "Colorspace TAG for the sidecar. rawpy's linear output has "
                               "sRGB/Rec.709 primaries — NOT ACEScg; convert downstream "
                               "via OCIO. Retag only if your config names it differently."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, file_path, **kwargs):
        try:
            stat = os.stat(str(file_path))
            return f"{file_path}:{stat.st_mtime_ns}:{stat.st_size}:{sorted(kwargs.items())}"
        except OSError:
            return f"{file_path}:missing:{sorted(kwargs.items())}"

    def load(self, file_path, undistort=True, half_size=False, white_balance="camera",
             exposure_ev=0.0, write_exr=True, output_dir="atlas_exports/raw_plates",
             colorspace="Linear Rec.709 (sRGB)"):
        np = _require_numpy()
        torch = _require_torch()
        from atlas_camera.core.schema import AtlasPlateRef
        try:
            from atlas_camera.raw import import_raw
        except ImportError as exc:
            raise RuntimeError(
                "AtlasLoadRAW requires the [raw] extra. "
                "Install with: pip install -e .[raw]") from exc

        path = str(file_path or "").strip()
        if not path or not Path(path).is_file():
            raise RuntimeError(f"AtlasLoadRAW: RAW file not found: {path!r}")

        result = import_raw(path, undistort=bool(undistort),
                            half_size=bool(half_size),
                            white_balance=white_balance,
                            exposure_ev=float(exposure_ev))

        image = torch.from_numpy(
            np.ascontiguousarray(result.display_srgb)).unsqueeze(0)

        exr_path, exr_warning = (None, None)
        if write_exr:
            exr_path, exr_warning = self._write_exr_sidecar(
                result.linear_rgb, path, output_dir)

        report_lines = result.summary_lines()
        if exr_path:
            report_lines.append(f"linear EXR: {exr_path} ({colorspace})")
        elif exr_warning:
            report_lines.append(exr_warning)

        plate_ref = AtlasPlateRef(
            image_path=exr_path,
            preview_b64=_image_tensor_to_preview_b64(image, quality=85),
            colorspace=colorspace or "Linear Rec.709 (sRGB)",
            bit_depth="16f" if exr_path else "8-bit/proxy",
            role="source",
            is_proxy=exr_path is None,
            metadata={
                "registered_from": "AtlasLoadRAW",
                "raw_source": path,
                "camera_model": result.camera_model,
                "undistort_status": result.undistort_status,
            },
        )
        return (image, plate_ref, result,
                float(result.focal_length_mm or 0.0),
                float(result.sensor_width_mm or 36.0),
                "\n".join(report_lines))

    @staticmethod
    def _write_exr_sidecar(linear_rgb, raw_path, output_dir):
        """Write the scene-linear half-float EXR. Returns (path, warning)."""
        try:
            import cv2
        except ImportError:
            return None, ("EXR sidecar skipped: opencv-python missing "
                          "(pip install -e .[raw]).")
        out_dir = Path(str(output_dir or "atlas_exports/raw_plates"))
        out_dir.mkdir(parents=True, exist_ok=True)
        exr_path = out_dir / (Path(raw_path).stem + "_linear.exr")
        bgr = linear_rgb[..., ::-1].astype("float32")
        try:
            ok = cv2.imwrite(str(exr_path),
                             bgr, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        except Exception:  # noqa: BLE001 — codec-disabled builds raise
            ok = False
        if not ok or not exr_path.is_file():
            return None, ("EXR sidecar FAILED: opencv needs the OpenEXR codec — "
                          "use opencv-python 4.x and set OPENCV_IO_ENABLE_OPENEXR=1 "
                          "before ComfyUI starts (same requirement as the OCIO path). "
                          "plate_ref downgraded to proxy.")
        return str(exr_path), None


# ---------------------------------------------------------------------------
# Track 1 — New Python-only nodes
# ---------------------------------------------------------------------------

class AtlasSolveFromImage:
    """Solve camera from a ComfyUI IMAGE tensor (no file path needed)."""
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "solve"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "focal_length_mm": ("FLOAT", {"default": 0.0, "min": 0.0,
                    "tooltip": "0 = auto-detect, or EXIF via a wired raw_meta"}),
                "sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 0.01}),
                "detect_vanishing_points": ("BOOLEAN", {"default": True,
                    "tooltip": "Run line/VP detection. Off = metadata-only solve "
                               "(no fx, cam_y=0 -> black depth/blockout)."}),
                # Link input (not a widget — saved-workflow-safe): AtlasLoadRAW's
                # metadata; supplies EXIF focal + measured sensor unless the
                # widgets above are explicitly set.
                "raw_meta": ("ATLAS_RAW_META",),
            },
        }

    def solve(self, image, focal_length_mm=0.0, sensor_width_mm=36.0,
              detect_vanishing_points=True, raw_meta=None):
        tmp = _save_image_tensor_to_tmp(image)
        try:
            focal_hint, sensor_w, sensor_h = _resolve_raw_hints(
                focal_length_mm, sensor_width_mm, raw_meta)
            hints: dict[str, Any] = {}
            if focal_hint:
                hints["focal_length_mm"] = focal_hint
                hints["sensor_width_mm"] = sensor_w
                if sensor_h:
                    hints["sensor_height_mm"] = sensor_h
            solve = solve_still_image(tmp, intrinsics_hint=hints or None,
                                      detect_vanishing_points=detect_vanishing_points)
            _stamp_raw_provenance(solve, raw_meta)
            return (solve,)
        finally:
            os.unlink(tmp)


class AtlasConstrainedSolve:
    """Guided solve using line constraints JSON (from Atlas UI or hand-crafted)."""
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "solve"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "constraints_json": ("STRING", {"default": "{}", "multiline": True,
                                                "tooltip": "Atlas constraints dict with line_groups and scale_constraints"}),
            },
            "optional": {
                "focal_length_mm": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 0.01}),
            },
        }

    def solve(self, image, constraints_json, focal_length_mm=0.0, sensor_width_mm=36.0):
        constraints = json.loads(constraints_json)
        tmp = _save_image_tensor_to_tmp(image)
        try:
            hint: dict[str, Any] | None = None
            if focal_length_mm and focal_length_mm > 0:
                hint = {"focal_length_mm": focal_length_mm, "sensor_width_mm": sensor_width_mm}
            return (solve_from_constraints(tmp, constraints, intrinsics_hint=hint),)
        finally:
            os.unlink(tmp)


class AtlasLearnedSolveFromImage:
    """Solve a camera from a ComfyUI IMAGE using the learned GeoCalib prior.

    Robust alternative to vanishing-point detection for AI-generated images:
    predicts focal length and gravity (up-vector) directly from image content, so
    it does not depend on clean straight edges converging to consistent VPs.
    Requires the [neural] extra (torch + geocalib) in ComfyUI's venv.
    """
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "solve"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "height_mode": (["measure_from_depth", "assume"], {"default": "measure_from_depth",
                    "tooltip": "measure_from_depth = fit the ground plane with Depth Anything V2 "
                               "(no assumed eye height); assume = use camera_height_m."}),
                "camera_height_m": ("FLOAT", {"default": 1.6, "min": 0.01, "max": 1000.0,
                    "tooltip": "Fallback / assumed camera height when not measured or low-confidence."}),
                "depth_model": (list(_DEPTH_MODEL_CHOICES),
                    {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                    "tooltip": "Metric depth backend (fed the solved focal). V2-Metric-Outdoor "
                               "(DEFAULT) / V2-Metric-Indoor: Apache, transformers-only (NO extra "
                               "install), best all-round; Outdoor wins on sky/exterior scenes. "
                               "MoGe-2 (Ruicheng/moge-*): MIT, cleanest on ENCLOSED/INTERIOR shots "
                               "but masks sky (poor outdoors) — needs [moge]. DA3* (EXPERIMENTAL): "
                               "strong metric, heavy deps, DA3NESTED is non-commercial CC BY-NC — "
                               "needs [neural-da3]. (4-scene A/B 2026-07-13.)"}),
                "sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 0.01}),
                "weights": (["pinhole", "simple_radial"], {"default": "pinhole",
                    "tooltip": "pinhole = no lens distortion (best for clean AI renders)."}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                # APPENDED 2026-07-18 (positional widget rule: new widgets go last).
                "focal_length_mm": ("FLOAT", {"default": 0.0, "min": 0.0,
                    "tooltip": "0 = GeoCalib predicts the focal. >0 (or a wired AtlasLoadRAW "
                               "raw_meta) = trusted focal (e.g. EXIF) wins; GeoCalib still "
                               "supplies gravity/roll."}),
                # Link input (not a widget — saved-workflow-safe).
                "raw_meta": ("ATLAS_RAW_META",),
            },
        }

    def solve(self, image, height_mode="measure_from_depth", camera_height_m=1.6,
              depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
              sensor_width_mm=36.0, weights="pinhole", device="auto",
              focal_length_mm=0.0, raw_meta=None):
        from atlas_camera.core.solver import solve_still_image_learned
        focal_hint, sensor_w, sensor_h = _resolve_raw_hints(
            focal_length_mm, sensor_width_mm, raw_meta)
        tmp = _save_image_tensor_to_tmp(image)
        try:
            h, w = int(image.shape[1]), int(image.shape[2])
            camera_height = "auto" if height_mode == "measure_from_depth" else camera_height_m
            solve = solve_still_image_learned(
                tmp,
                image_size=(w, h),
                camera_height=camera_height,
                sensor_width_mm=sensor_w,
                sensor_height_mm=sensor_h,
                focal_length_mm_hint=focal_hint,
                weights=weights,
                depth_model=depth_model,
                device=None if device == "auto" else device,
            )
            _stamp_raw_provenance(solve, raw_meta)
            return (solve,)
        finally:
            os.unlink(tmp)


class AtlasScaleOverride:
    """📐 Manual metric-scale dial for a solve — the artist's scale override.

    Single-image camera recovery has an inherent SCALE ambiguity: with no ground
    plane to fit and no known-size reference, the solve falls back to an assumed
    ~1.6 m eye height (`scale_source=assumed_default`), which is often far off for
    elevated vistas (a cityscape overlook can read ~10× too small). Metric scale
    is PROPORTIONAL to camera height (`scale = −cam_y/g`, with g fixed by the
    depth), so this node rescales the solve by a single factor — multiplying the
    camera position and both extrinsics matrices' translation columns — and EVERY
    downstream metric follows: geometry distances, the 📏 Band Box cutoffs, and
    the DCC-export camera positions. The projection is purely angular, so the
    view/texture mapping is pixel-identical — only the metric numbers move.

    `scale` is a plain multiplier (10.0 = ten times as far/big — the "1:10" case).
    `camera_height_m` (0 = off) instead SETS an absolute camera height when you
    know the real vantage, and the node computes the factor for you. Composable
    companion node (works after ANY solve); stamps `scale_source="manual_override"`.
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "override"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"solve": ("ATLAS_SOLVE",)},
            "optional": {
                "scale": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 100000.0, "step": 0.1,
                    "tooltip": "Metric scale multiplier — 10.0 = the whole scene is 10× as far/big "
                               "(the single-image '1:10' case). Metric scale ∝ camera height, so this "
                               "uniformly rescales every downstream distance (geometry, 📏 cutoffs, "
                               "DCC-export cameras); the projected view is unchanged. Ignored when "
                               "camera_height_m > 0."}),
                "camera_height_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000000.0, "step": 0.1,
                    "tooltip": "Absolute override: SET the camera height in metres when you know the "
                               "real vantage (the node computes the factor). 0 = use the scale "
                               "multiplier instead."}),
            },
        }

    def override(self, solve, scale=1.0, camera_height_m=0.0):
        import copy
        out = copy.deepcopy(solve)
        extr = out.camera.extrinsics
        vm = [list(r) for r in extr.camera_view_matrix]   # world->cam
        # Camera WORLD position p = -R_wc^T @ t_wc (robust: some solves leave the
        # camera_position field at 0 but always populate the view matrix). The
        # translation column scales by the same factor, so p_new = p * factor.
        t = [vm[0][3], vm[1][3], vm[2][3]]
        p = [-(vm[0][k] * t[0] + vm[1][k] * t[1] + vm[2][k] * t[2]) for k in range(3)]
        cur_h = p[1]
        if float(camera_height_m) > 0.0 and abs(cur_h) > 1e-6:
            factor = float(camera_height_m) / abs(cur_h)
        else:
            factor = float(scale)
        if not (factor > 0.0):
            factor = 1.0

        for r in range(3):
            vm[r][3] = vm[r][3] * factor
        extr.camera_view_matrix = tuple(tuple(r) for r in vm)
        extr.camera_position = tuple(c * factor for c in p)
        wm = [list(r) for r in extr.camera_world_matrix]
        for r in range(3):
            wm[r][3] = wm[r][3] * factor
        extr.camera_world_matrix = tuple(tuple(r) for r in wm)
        meta = dict(getattr(out, "debug_metadata", None) or {})
        meta["scale_override"] = factor
        meta["scale_source"] = "manual_override"
        out.debug_metadata = meta
        try:
            out.camera.confidence = out.camera.confidence.with_metric("scale", 1.0)
        except Exception:  # noqa: BLE001 — hand-built solves without a model
            pass

        new_h = extr.camera_position[1]
        report = (
            f"AtlasScaleOverride: ×{factor:.4g}  |  camera height {cur_h:.2f} m → {new_h:.2f} m\n"
            "  Rescales ALL downstream metric — geometry distances, 📏 Band Box cutoffs, and DCC "
            "export cameras — uniformly; the projection/view is unchanged (angular). Insert between "
            "the solve and the geometry/viewport nodes.")
        return (out, report)


class AtlasRollTrim:
    """🎚 Manual roll trim for a solve — level a leaning solve by eye.

    GeoCalib's gravity estimate can drift a few degrees on AI-generated plates
    with no true horizon (measured live: −5.6° solved vs ~−2.6° implied by the
    architecture's verticals on a sci-fi interior), and the classical VP
    cross-check often finds nothing on greebled/non-rectilinear scenes. This is
    the roll counterpart of `AtlasScaleOverride`'s scale dial: rotate the
    recovered camera about its own VIEW AXIS by `roll_deg` and let everything
    downstream follow. The camera position and view direction are INVARIANT —
    only the camera's up/right spin — so framing is preserved and the fix is
    purely orientational.

    Wire it between the solve and the depth/derive nodes (like the scale
    dial): geometry back-projects through the view matrix, so a trim applied
    AFTER derivation leaves already-built geometry in the old frame (the
    report warns if the incoming solve carries proxy geometry). Pure Python,
    zero deps; composable after any solve; stamps
    `debug_metadata["roll_trim_deg"]` (accumulates across chained trims).
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "trim"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"solve": ("ATLAS_SOLVE",)},
            "optional": {
                "roll_deg": ("FLOAT", {"default": 0.0, "min": -45.0, "max": 45.0, "step": 0.05,
                    "tooltip": "Extra roll (degrees) about the recovered camera's view axis. "
                               "0 = no-op. Positive rotates the projected scene counter-clockwise "
                               "on screen (the horizon's right end rises); negative clockwise. "
                               "Dial until verticals/horizon read level. Position and view "
                               "direction never move."}),
            },
        }

    def trim(self, solve, roll_deg=0.0):
        import copy
        import math
        out = copy.deepcopy(solve)
        d = float(roll_deg)
        if abs(d) < 1e-9:
            return (out, "AtlasRollTrim: 0.00° — no-op (dial roll_deg to level the solve)")
        extr = out.camera.extrinsics
        c, s = math.cos(math.radians(d)), math.sin(math.radians(d))

        # V' = Rz(d) @ V — an extra roll in the CAMERA frame, left-multiplied
        # onto the world→cam view matrix. Rz preserves the camera z axis, so
        # the view direction is untouched; the rigid inverse below shows the
        # position is too (Rz's translation is zero).
        vm = [list(r) for r in extr.camera_view_matrix]
        rz = ((c, -s, 0.0, 0.0), (s, c, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        vm2 = [[sum(rz[r][k] * vm[k][col] for k in range(4)) for col in range(4)] for r in range(4)]
        extr.camera_view_matrix = tuple(tuple(row) for row in vm2)

        # Rigid inverse → world matrix; R_cw (columns = camera axes in world)
        # is the transpose of the view rotation block.
        r_wc = [[vm2[r][k] for k in range(3)] for r in range(3)]
        t_wc = [vm2[r][3] for r in range(3)]
        r_cw = [[r_wc[k][r] for k in range(3)] for r in range(3)]
        pos = [-sum(r_cw[r][k] * t_wc[k] for k in range(3)) for r in range(3)]
        extr.camera_world_matrix = tuple(
            tuple([*r_cw[r], pos[r]]) for r in range(3)
        ) + ((0.0, 0.0, 0.0, 1.0),)
        extr.camera_rotation_matrix = tuple(tuple(row) for row in r_cw)
        extr.camera_position = tuple(pos)

        # Recompute the stored horizon LINE for the rolled camera (no longer a
        # single image row): the vanishing line of world-horizontal planes is
        # the set of pixels whose backprojected rays have zero world-Y
        # direction — linear in (u, v). Ray(u,v) ∝ ((u-cx)/fx, -(v-cy)/fy, -1)
        # in the camera frame; world-Y component = R_cw row 1 · ray = 0.
        horizon_note = ""
        intr = out.camera.intrinsics
        if out.horizon_line is not None and intr.fx_px and intr.image_width:
            fx = float(intr.fx_px)
            fy = float(intr.fy_px or intr.fx_px)
            cx = float(intr.cx_px if intr.cx_px is not None else intr.image_width / 2.0)
            cy = float(intr.cy_px if intr.cy_px is not None else (intr.image_height or 0) / 2.0)
            w = float(intr.image_width)
            a = r_cw[1][0] / fx
            b = -r_cw[1][1] / fy
            cc = -r_cw[1][0] * cx / fx + r_cw[1][1] * cy / fy - r_cw[1][2]
            if abs(b) > 1e-12:
                y_at = lambda u: (-cc - a * u) / b  # noqa: E731
                y0, y1 = y_at(0.0), y_at(w)
                out.horizon_line.endpoints_px = ((0.0, y0), (w, y1))
                out.horizon_line.line_coefficients = (a, b, cc)
                tilt = math.degrees(math.atan2(y1 - y0, w))
                horizon_note = f"  |  horizon tilt now {tilt:+.2f}°"
                meta_ce = dict((out.debug_metadata or {}).get("camera_estimation") or {})
                meta_ce["horizon_angle"] = tilt
                meta = dict(out.debug_metadata or {})
                meta["camera_estimation"] = meta_ce
                out.debug_metadata = meta

        meta = dict(out.debug_metadata or {})
        meta["roll_trim_deg"] = float(meta.get("roll_trim_deg", 0.0)) + d
        out.debug_metadata = meta

        geom_warn = ""
        scene = getattr(out, "projection_scene", None)
        if scene is not None and getattr(scene, "proxy_geometry", None):
            geom_warn = ("\n  ⚠ this solve already carries derived geometry, built in the UN-trimmed "
                         "frame — wire AtlasRollTrim BEFORE the depth/derive nodes instead.")
        report = (
            f"AtlasRollTrim: {d:+.2f}° about the view axis{horizon_note}\n"
            "  Camera position and view direction unchanged — only up/right rotate; every "
            "downstream derive/export follows. Composable after any solve." + geom_warn)
        return (out, report)


class AtlasGravityOverride:
    """🎚 ABSOLUTE gravity override — set the solve's pitch and roll directly.

    The trims (`AtlasRollTrim`/`AtlasPitchTrim`) are RELATIVE dials; this is
    the absolute version, born from the D810 haze incident's second act: the
    gravity MIRROR repaired the flip's sign, but the flipped estimate itself
    was ~7° off in pitch and ~9° in roll, so the mirrored scene still leaned.
    When you know the true angles (crop-probe, level references, or by eye),
    set them here: ``pitch_deg`` (positive = looking DOWN) and ``roll_deg``
    re-pose the camera about its own position, preserving the horizontal
    HEADING (yaw stays unobservable/canonical). Position never moves.

    Wire it between the solve and the depth/derive nodes, like the other
    dials. Stamps `debug_metadata["gravity_override"]`.
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "override"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"solve": ("ATLAS_SOLVE",)},
            "optional": {
                "pitch_deg": ("FLOAT", {"default": 0.0, "min": -89.0, "max": 89.0, "step": 0.05,
                    "tooltip": "ABSOLUTE camera pitch: positive = looking DOWN that many "
                               "degrees below the horizon, negative = up, 0 = level."}),
                "roll_deg": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.05,
                    "tooltip": "ABSOLUTE roll about the view axis (0 = level horizon). "
                               "Same screen direction as AtlasRollTrim: positive turns "
                               "the projected scene counter-clockwise."}),
            },
        }

    def override(self, solve, pitch_deg=0.0, roll_deg=0.0):
        import copy
        import math

        from atlas_camera.core.camera_math import look_at_view_matrix

        out = copy.deepcopy(solve)
        extr = out.camera.extrinsics

        wm = extr.camera_world_matrix
        fwd = (-float(wm[0][2]), -float(wm[1][2]), -float(wm[2][2]))
        old_pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd[1]))))
        # Preserve heading: the horizontal component of the current forward.
        hx, hz = fwd[0], fwd[2]
        norm = math.hypot(hx, hz)
        if norm < 1e-9:
            hx, hz = 0.0, -1.0  # straight up/down: canonical -Z heading
            norm = 1.0
        hx, hz = hx / norm, hz / norm

        p = math.radians(float(pitch_deg))  # positive = down
        new_fwd = (hx * math.cos(p), -math.sin(p), hz * math.cos(p))
        eye = tuple(float(v) for v in extr.camera_position)
        target = (eye[0] + new_fwd[0], eye[1] + new_fwd[1], eye[2] + new_fwd[2])
        view, _world, _rot3 = look_at_view_matrix(eye, target)
        vm = [list(r) for r in view]

        d = float(roll_deg)
        if abs(d) > 1e-9:
            c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
            rz = ((c, -s, 0.0, 0.0), (s, c, 0.0, 0.0),
                  (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
            vm = [[sum(rz[r][k] * vm[k][col] for k in range(4)) for col in range(4)]
                  for r in range(4)]

        r_cw = _extrinsics_from_view(extr, vm)
        _recompute_horizon_line(out, r_cw)

        meta = dict(out.debug_metadata or {})
        meta["gravity_override"] = {"pitch_deg": float(pitch_deg),
                                    "roll_deg": float(roll_deg),
                                    "previous_pitch_deg": round(old_pitch, 2)}
        out.debug_metadata = meta

        geom_warn = ""
        scene = getattr(out, "projection_scene", None)
        if scene is not None and getattr(scene, "proxy_geometry", None):
            geom_warn = ("\n  ⚠ this solve already carries derived geometry, built in the "
                         "old frame — wire AtlasGravityOverride BEFORE the derive nodes.")
        report = (
            f"AtlasGravityOverride: pitch {old_pitch:+.1f}° → {-float(pitch_deg):+.1f}° "
            f"(looking {'down' if pitch_deg > 0 else 'up' if pitch_deg < 0 else 'level'}), "
            f"roll set to {float(roll_deg):+.1f}°\n"
            "  Absolute orientation; heading and position preserved. Every downstream "
            "derive/export follows." + geom_warn)
        return (out, report)


class AtlasPitchTrim:
    """🎚 Manual pitch trim / gravity-mirror for a solve — RollTrim's sibling.

    Motivated by the live-found GeoCalib gravity FLIP (a D810 window shot:
    bright reflection haze at the frame bottom read as sky and the solve came
    out looking UP 39° on an obvious bird's-eye — the `camera_looks_up`
    health flag's exact failure mode). `mirror_gravity` reflects the camera's
    pitch about the horizon (new forward.y = −forward.y, heading and roll
    preserved) — the one-click repair for a flipped solve; `pitch_deg` then
    fine-tunes (positive tilts the view DOWN). Rotation happens about the
    camera's own RIGHT axis, so position is invariant and roll never changes.

    Wire it between the solve and the depth/derive nodes (the RollTrim /
    ScaleOverride slot); the report warns if the solve already carries
    derived geometry. Pure Python, zero deps; stamps
    `debug_metadata["pitch_trim_deg"]` (+ `gravity_mirrored`).
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "trim"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"solve": ("ATLAS_SOLVE",)},
            "optional": {
                "mirror_gravity": ("BOOLEAN", {"default": False,
                    "tooltip": "Reflect the camera's pitch about the horizon (forward.y "
                               "flips sign; heading + roll preserved). THE repair for a "
                               "flipped GeoCalib gravity — e.g. bottom-of-frame haze read "
                               "as sky turning a bird's-eye into an up-shot."}),
                "pitch_deg": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.05,
                    "tooltip": "Extra pitch (degrees) about the camera's RIGHT axis, applied "
                               "after any mirror. Positive tilts the view DOWN. 0 = no-op."}),
            },
        }

    def trim(self, solve, mirror_gravity=False, pitch_deg=0.0):
        import copy
        import math
        out = copy.deepcopy(solve)
        extr = out.camera.extrinsics

        wm = extr.camera_world_matrix
        fwd_y = -float(wm[1][2])
        old_pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd_y))))
        # Mirror = rotate view DOWN by 2×(current up-pitch): new forward.y
        # becomes exactly −forward.y while the horizontal heading (and roll,
        # since we rotate about the camera's own right axis) is untouched.
        d = (2.0 * old_pitch if mirror_gravity else 0.0) + float(pitch_deg)
        if abs(d) < 1e-9:
            return (out, "AtlasPitchTrim: 0.00° — no-op (mirror_gravity repairs a "
                         "flipped solve; pitch_deg fine-tunes)")
        c, s = math.cos(math.radians(d)), math.sin(math.radians(d))

        # V' = Rx(d) @ V — extra rotation about the CAMERA's x (right) axis,
        # left-multiplied onto the world→cam view matrix. Rx preserves the
        # camera x axis (roll untouched) and has zero translation, so the
        # rigid inverse below shows the position is invariant too. With the
        # camera frame y-up/z-back, positive d pitches the view DOWN.
        vm = [list(r) for r in extr.camera_view_matrix]
        rx = ((1.0, 0.0, 0.0, 0.0), (0.0, c, -s, 0.0), (0.0, s, c, 0.0), (0.0, 0.0, 0.0, 1.0))
        vm2 = [[sum(rx[r][k] * vm[k][col] for k in range(4)) for col in range(4)] for r in range(4)]
        extr.camera_view_matrix = tuple(tuple(row) for row in vm2)

        r_wc = [[vm2[r][k] for k in range(3)] for r in range(3)]
        t_wc = [vm2[r][3] for r in range(3)]
        r_cw = [[r_wc[k][r] for k in range(3)] for r in range(3)]
        pos = [-sum(r_cw[r][k] * t_wc[k] for k in range(3)) for r in range(3)]
        extr.camera_world_matrix = tuple(
            tuple([*r_cw[r], pos[r]]) for r in range(3)
        ) + ((0.0, 0.0, 0.0, 1.0),)
        extr.camera_rotation_matrix = tuple(tuple(row) for row in r_cw)
        extr.camera_position = tuple(pos)

        new_fwd_y = -float(extr.camera_world_matrix[1][2])
        new_pitch = math.degrees(math.asin(max(-1.0, min(1.0, new_fwd_y))))

        # Recompute the stored horizon line (same vanishing-line math as
        # AtlasRollTrim — world-Y ray component zero, linear in (u, v)).
        horizon_note = ""
        intr = out.camera.intrinsics
        if out.horizon_line is not None and intr.fx_px and intr.image_width:
            fx = float(intr.fx_px)
            fy = float(intr.fy_px or intr.fx_px)
            cx = float(intr.cx_px if intr.cx_px is not None else intr.image_width / 2.0)
            cy = float(intr.cy_px if intr.cy_px is not None else (intr.image_height or 0) / 2.0)
            w = float(intr.image_width)
            a = r_cw[1][0] / fx
            b = -r_cw[1][1] / fy
            cc = -r_cw[1][0] * cx / fx + r_cw[1][1] * cy / fy - r_cw[1][2]
            if abs(b) > 1e-12:
                y_at = lambda u: (-cc - a * u) / b  # noqa: E731
                y0, y1 = y_at(0.0), y_at(w)
                out.horizon_line.endpoints_px = ((0.0, y0), (w, y1))
                out.horizon_line.line_coefficients = (a, b, cc)

        meta = dict(out.debug_metadata or {})
        meta["pitch_trim_deg"] = float(meta.get("pitch_trim_deg", 0.0)) + d
        if mirror_gravity:
            meta["gravity_mirrored"] = True
        out.debug_metadata = meta

        geom_warn = ""
        scene = getattr(out, "projection_scene", None)
        if scene is not None and getattr(scene, "proxy_geometry", None):
            geom_warn = ("\n  ⚠ this solve already carries derived geometry, built in the "
                         "UN-trimmed frame — wire AtlasPitchTrim BEFORE the depth/derive nodes.")
        mirror_note = "gravity MIRRORED, " if mirror_gravity else ""
        report = (
            f"AtlasPitchTrim: {mirror_note}pitch {old_pitch:+.1f}° → {new_pitch:+.1f}° "
            f"(rotated {d:+.2f}° about the camera's right axis)\n"
            "  Position, heading and roll unchanged — every downstream derive/export "
            "follows. Composable after any solve." + geom_warn)
        return (out, report)


class AtlasReferenceScaleSolve:
    """Fix a solve's metric scale from a known-size reference object.

    The most reliable way to set absolute camera height: mark the pixel box of a
    known object (person, door, car, …) and Atlas solves the metric camera height
    by single-view geometry using the solve's orientation + focal — no assumed
    eye height. Composable after any solve node (e.g. the learned GeoCalib solve).
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "FLOAT")
    RETURN_NAMES = ("solve", "camera_height_m")
    FUNCTION = "apply"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "reference_id": (_reference_id_choices(), ),
                "bbox_x0": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 1.0}),
                "bbox_y0": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 1.0,
                                      "tooltip": "Top edge (smaller y) of the object box."}),
                "bbox_x1": ("FLOAT", {"default": 100.0, "min": 0.0, "step": 1.0}),
                "bbox_y1": ("FLOAT", {"default": 400.0, "min": 0.0, "step": 1.0,
                                      "tooltip": "Bottom edge (larger y) — the object's base on the ground."}),
            },
            "optional": {
                "height_override_m": ("FLOAT", {"default": 0.0, "min": 0.0,
                    "tooltip": "0 = use the reference's registry height; else override in metres."}),
            },
        }

    def apply(self, solve, reference_id, bbox_x0, bbox_y0, bbox_x1, bbox_y1,
              height_override_m=0.0):
        from atlas_camera.core.solver import apply_reference_scale
        ref: dict[str, Any] = {
            "reference_id": reference_id,
            "bbox_px": [bbox_x0, bbox_y0, bbox_x1, bbox_y1],
        }
        if height_override_m and height_override_m > 0:
            ref["height_m"] = height_override_m
        apply_reference_scale(solve, [ref])
        return (solve, float(solve.camera.extrinsics.camera_position[1]))


class AtlasAssessImage:
    """VLM pre-flight for the whole DMP pipeline — wire it directly after
    LoadImage, BEFORE anything else consumes the photo.

    A vision-language model (Ollama / LM Studio / llama.cpp locally, or the
    `openai` provider — any OpenAI-compatible cloud endpoint + api_key, for
    users without local models; the same provider layer as
    `AtlasVLMScaleCues`) analyzes the photo against an
    expert instruction prompt encoding Atlas Camera's full settings knowledge
    (`inference.assessor.ATLAS_ASSESSMENT_SYSTEM_PROMPT`): scene type /
    depth-model choice, sky separation, depth-band layer design, disocclusion
    fill, edge mattes, relief tuning, scale-reference opportunities, and an
    honest camera-move viability rubric (score + max orbit degrees + what
    breaks first). The `report` output is human-readable (wire to a
    Show Text node); `settings_json` is the machine-readable
    recommended_settings block.

    STAGED 5-LAYER PLAN: the assessment also divides the photo into the
    staged master workflow's five fixed layers (sky + far/bg/mid/fg depth
    bands) and emits one SAM3 prompt STRING output per layer
    (`sam_prompt_*`) — wire them into the sky SAM3Segment and the four SAM
    SCOPE rows' prompt inputs so each row's segmentation prompt comes from
    the assessment instead of hand-typing. Not every image has every layer:
    an absent layer (no sky, empty mid band, ...) yields "" and the report
    says to leave that stage bypassed; only sky falls back to the literal
    "sky" (a no-match prompt there returns an empty mask, which IS the
    correct sky mask for a skyless photo).

    EXECUTION PAUSE — opt-in since 2026-07-11 (`auto_continue`, default ON):
    by default the node is ADVISORY: the assessment runs, its staged
    prompts/geometry flow downstream, and the same queue continues — the ✅
    solve gate (and the 📐 patch gate) are the workflow's checkpoints. With
    `auto_continue` OFF the original hard gate returns: while `proceed` is
    False the `image` output returns ExecutionBlocker — everything
    downstream of the photo is silently skipped, so the first Queue costs
    only the assessment; ▶ Continue Workflow approves THIS image (the
    assessment is cached per image+provider, so continuing never re-runs the
    VLM). Same native pause mechanism as 📐 Extract Angle gating.

    Advisory only, per the LLM-confirm principle: the VLM never changes a
    setting itself — it recommends, the artist decides. Fails soft to a
    "provider unreachable" report; `proceed` still works without an
    assessment.
    """
    RETURN_TYPES = ("IMAGE", "STRING", "STRING",
                    "STRING", "STRING", "STRING", "STRING", "STRING",
                    "STRING", "STRING", "STRING", "STRING",
                    "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "report", "settings_json",
                    "sam_prompt_sky", "sam_prompt_far", "sam_prompt_bg",
                    "sam_prompt_mid", "sam_prompt_fg",
                    "geom_far", "geom_bg", "geom_mid", "geom_fg",
                    "band_far", "band_bg", "band_mid", "band_fg")
    FUNCTION = "assess"
    CATEGORY = "Atlas Camera"
    # OUTPUT_NODE so the assessment ALWAYS runs and shows its report on the
    # node itself (ui.text, rendered by atlas_assess.js) — without this, a
    # graph where nothing consumed `report` gave zero visible output (found
    # live: "the VLM did nothing").
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "provider": (["ollama", "lmstudio", "llamacpp", "openai"], {"default": "ollama",
                    "tooltip": "VLM backend. ollama/lmstudio/llamacpp are local servers; "
                               "'openai' is any OpenAI-compatible CLOUD endpoint (api.openai.com "
                               "by default, OpenRouter etc. via base_url) for users without local "
                               "models — needs api_key. Blank model/base_url use each provider's "
                               "own defaults (same conventions as AtlasVLMScaleCues)."}),
                "model": ("STRING", {"default": "",
                    "tooltip": "Vision model id; blank = provider default (ollama: gemma3:4b)."}),
                "base_url": ("STRING", {"default": ""}),
                "extra_instructions": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Optional artist notes appended to the assessment request — e.g. "
                               "'the camera move is a slow dolly-in on the tower'. The VLM tailors "
                               "band/patch advice to the intended move."}),
                "proceed": ("BOOLEAN", {"default": False,
                    "tooltip": "OFF = the image output is paused (downstream skipped) so you can "
                               "read the report and apply settings first. Turn ON (or click "
                               "▶ Continue Workflow) and re-queue to run the full pipeline. "
                               "A ▶ Continue click approves THIS image only (see approved_for); "
                               "a manual toggle here is an unconditional override."}),
                "approved_for": ("STRING", {"default": "",
                    "tooltip": "Managed by ▶ Continue Workflow: the fingerprint of the image the "
                               "current proceed=True was approved for. When the input image "
                               "changes, the gate re-arms automatically instead of running a "
                               "stale approval. Leave empty when toggling proceed by hand "
                               "(empty = unconditional)."}),
                # APPENDED last (widgets_values is positional — never insert).
                "api_key": ("STRING", {"default": "",
                    "tooltip": "API key for the 'openai' cloud provider (ignored by local "
                               "providers). SAVED INTO THE WORKFLOW FILE — prefer leaving this "
                               "blank and setting the OPENAI_API_KEY environment variable so "
                               "shared workflows never carry your key."}),
                "offload_model": ("BOOLEAN", {"default": False,
                    "tooltip": "Free the VLM's VRAM after a SUCCESSFUL assessment so the heavy "
                               "pipeline (depth/SAM/LaMa) doesn't fight it for memory — the "
                               "assessment is cached per image, so ▶ Continue never reloads the "
                               "model. Per provider: ollama = keep_alive:0 (clean unload); "
                               "lmstudio = request ttl for JIT loads + the 'lms' CLI when on "
                               "PATH for GUI-loaded models; llamacpp = NOT possible (the server "
                               "owns its model — restart it to free VRAM); openai = nothing "
                               "local. A failed assessment keeps the model warm for the retry. "
                               "The report shows what actually happened."}),
                "auto_continue": ("BOOLEAN", {"default": True,
                    "tooltip": "ON (default): advisory mode — the assessment runs, its SAM "
                               "prompts/geometry flow downstream, and the SAME queue continues "
                               "without a ▶ Continue click; the ✅ solve gate (and the later 📐 "
                               "patch gate) become the workflow's checkpoints. Turn OFF to "
                               "restore the hard per-image gate: the image output blocks until "
                               "▶ Continue Workflow approves THIS image."}),
            },
        }

    def assess(self, image, provider="ollama", model="", base_url="",
               extra_instructions="", proceed=False, approved_for="",
               api_key="", offload_model=False, auto_continue=True, **_extra):
        # **_extra: API-format exports can serialize the ▶ Continue Workflow
        # BUTTON widget as a bogus input key — tolerate unknown kwargs.
        import hashlib

        from atlas_camera.inference.assessor import (
            assess_image,
            staged_layer_bands,
            staged_layer_geometry,
            staged_layer_prompts,
        )

        # Cache per image+provider settings so flipping `proceed` (which
        # re-executes this node) doesn't re-run a 30-120s VLM call.
        key_src = image.cpu().numpy().tobytes()
        key = hashlib.md5(key_src).hexdigest() + f"|{provider}|{model}|{base_url}|{extra_instructions}"
        cached = _ATLAS_ASSESS_CACHE.get(key)
        if cached is None:
            tmp = _save_image_tensor_to_tmp(image)
            try:
                cached = assess_image(
                    tmp, provider=provider, model=model,
                    base_url=base_url.strip() or None,
                    api_key=api_key.strip() or None,
                    extra_instructions=extra_instructions,
                    offload_model=bool(offload_model))
            finally:
                os.unlink(tmp)
            # Never cache FAILED assessments: the user typically starts the
            # provider after seeing the failure report — the retry must
            # actually retry.
            if cached.ok:
                if len(_ATLAS_ASSESS_CACHE) >= 8:
                    _ATLAS_ASSESS_CACHE.pop(next(iter(_ATLAS_ASSESS_CACHE)))
                _ATLAS_ASSESS_CACHE[key] = cached

        settings_json = json.dumps(
            (cached.payload or {}).get("recommended_settings", {}), indent=1) if cached.ok else "{}"

        # ▶ Continue approvals are per-image: a non-empty approved_for that
        # doesn't match the CURRENT image re-arms the gate (found live — the
        # proceed widget persists, so a new image sailed through the previous
        # image's approval). An empty approved_for with proceed=True is the
        # manual unconditional override.
        img_fp = _image_fingerprint(image)
        report = cached.report
        # auto_continue (default ON): advisory mode — never block; the solve
        # gate downstream is the first checkpoint. OFF restores the hard
        # per-image ▶ Continue gate with its stale-approval re-arming.
        effective_proceed = bool(auto_continue) or (
            bool(proceed) and (not approved_for or approved_for == img_fp))
        if not auto_continue and proceed and approved_for and approved_for != img_fp:
            report = ("*** GATE RE-ARMED: the input image changed since ▶ Continue was "
                      "clicked — review the fresh assessment below, then ▶ Continue "
                      "again for this image. ***\n\n" + report)

        if effective_proceed:
            img_out = image
        else:
            blocker = _execution_blocker()
            img_out = blocker if blocker is not None else image
        # Staged 5-layer SAM prompts + per-band geometry recommendations —
        # plain strings, NOT gated: everything they feed (SAM3 nodes /
        # AtlasCleanPlateLayer) also consumes the gated image via the plate
        # rail, so the image blocker already pauses it. geom_* wires into
        # AtlasCleanPlateLayer.geometry_override ("" = no recommendation,
        # the layer node's own band_geometry combo applies).
        sam = staged_layer_prompts(cached.payload if cached.ok else {})
        geom = staged_layer_geometry(cached.payload if cached.ok else {})
        # Watertight band boundaries (jointly derived — adjacent bands share
        # edges by construction); "" when no assessment = nodes keep widgets.
        band = staged_layer_bands(cached.payload if cached.ok else {})

        # ui.text renders the report directly on the node (atlas_assess.js);
        # ui.fingerprint is what the ▶ button stamps into approved_for.
        # ui.sam_prompts / ui.sam_geometry let the frontend mirror the
        # resolved values into LINKED widgets — a widget converted to a
        # linked input keeps displaying its stale typed text otherwise
        # (found live: values flowed at execution but were invisible).
        return {"ui": {"text": [report], "fingerprint": [img_fp],
                       "sam_prompts": [sam["sky"], sam["far"], sam["bg"],
                                       sam["mid"], sam["fg"]],
                       "sam_geometry": [geom["far"], geom["bg"],
                                        geom["mid"], geom["fg"]],
                       "sam_bands": [band["far"], band["bg"],
                                     band["mid"], band["fg"]]},
                "result": (img_out, report, settings_json,
                           sam["sky"], sam["far"], sam["bg"], sam["mid"], sam["fg"],
                           geom["far"], geom["bg"], geom["mid"], geom["fg"],
                           band["far"], band["bg"], band["mid"], band["fg"])}


class AtlasSolveGate:
    """✅ Solve-confirm checkpoint — pause the heavy graph until the artist
    approves the camera solve.

    The third gate in the established family (AtlasAssessImage gates the
    whole graph on VLM pre-flight; 📐 pauses the patch branch): wire
    `solve → viewport` UNGATED for a cheap preview (a low-grid relief costs
    seconds) and `solve → this gate → the heavy stack` (grid-1024 band
    layers, sky dome, Fixer, exports — the minutes). The first Queue costs a
    solve and a thumbnail-grade preview; check the camera in ℹ/📊 (or the
    report rendered on this node), click ✅ Approve Solve, and the re-queue
    runs the expensive graph exactly once, on a solve you signed off.

    Approval is fingerprint-scoped to (solve camera + source image): a new
    photo OR a re-solve with different settings re-arms the gate instead of
    sailing through a stale approval (the persisted-gating rule every gate
    here follows). Empty `approved_for` with proceed=True stays the manual
    unconditional override. Outside a ComfyUI runtime the gate degrades to
    pass-through (no ExecutionBlocker available).
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "gate"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "source_image": ("IMAGE", {"tooltip":
                    "The photo this solve came from — part of the approval "
                    "identity, so swapping the image re-arms the gate."}),
            },
            "optional": {
                "proceed": ("BOOLEAN", {"default": False, "tooltip":
                    "While off, the solve output returns ExecutionBlocker and "
                    "everything downstream of the gate is silently paused. "
                    "The ✅ Approve Solve button sets this and re-queues."}),
                "approved_for": ("STRING", {"default": "", "tooltip":
                    "Fingerprint of the solve+image the current approval was "
                    "given for (stamped by ✅). Mismatch re-arms the gate. "
                    "Leave empty when toggling proceed by hand to approve "
                    "unconditionally."}),
            },
        }

    def gate(self, solve, source_image, proceed=False, approved_for="", **_extra):
        # **_extra: API-format exports can serialize the button widget as a
        # bogus input key — tolerate unknown kwargs (AssessImage precedent).
        import math as _math

        np = _require_numpy()

        fp = _solve_fingerprint(solve, source_image)
        intr = solve.camera.intrinsics
        extr = solve.camera.extrinsics
        try:
            vm = np.array(extr.camera_view_matrix, dtype=np.float64)
            fwd = np.linalg.inv(vm)[:3, :3] @ np.array([0.0, 0.0, -1.0])
            pitch = _math.degrees(_math.asin(max(-1.0, min(1.0, float(fwd[1])))))
        except Exception:
            pitch = float("nan")
        fov = (2 * _math.degrees(_math.atan((intr.image_width or 0) /
               (2 * intr.fx_px))) if intr.fx_px else float("nan"))
        cam_h = (extr.camera_position or (0, float("nan"), 0))[1]
        meta = solve.debug_metadata or {}
        from atlas_camera.core.scene_health import scale_health
        sh = scale_health(solve)
        effective = bool(proceed) and (not approved_for or approved_for == fp)
        lines = [
            "✅ SOLVE APPROVED — heavy graph running." if effective else
            "⏸ SOLVE GATE — downstream paused. Review, then ✅ Approve Solve.",
            (f"focal: {intr.focal_length_mm:.1f}mm ({fov:.1f}° hFOV) on "
             f"{intr.sensor_width_mm}mm") if intr.focal_length_mm else "focal: n/a",
            f"camera height: {cam_h:.2f}m  (scale: {sh.status} / "
            f"{meta.get('scale_source', 'n/a')})",
            f"pitch: {pitch:+.1f}°",
            (f"confidence: {solve.confidence:.2f}  ({solve.source_method})"
             if getattr(solve, "confidence", None) is not None else ""),
        ]
        if not sh.safe_to_export:
            lines.insert(1, f"⚠ SCALE NOT VERIFIED — {sh.detail}")
        if proceed and approved_for and approved_for != fp:
            lines.insert(0, "*** GATE RE-ARMED: the solve or image changed since "
                            "approval — review and ✅ Approve again. ***")
        report = "\n".join(l for l in lines if l)

        if effective:
            out = solve
        else:
            blocker = _execution_blocker()
            out = blocker if blocker is not None else solve
        return {"ui": {"text": [report], "fingerprint": [fp]},
                "result": (out, report)}


class AtlasSceneHealthGate:
    """🩺 Scene-health checkpoint before the exporters — Gate 4 of the family.

    Runs core.scene_health.evaluate_scene_health (the SAME red-flag engine
    AtlasDebugReport renders) and, when the level is warn/fail, holds the
    solve until the artist clicks ✅ Acknowledge & Continue. Ship-closed
    approval-gate semantics (the SolveGate pattern, `_solve_fingerprint`
    identity) rather than a hard fail — per the engineering-recommendations
    doctrine: the user may OVERRIDE a warning but never LOSE it. Every
    execution (blocked or flowing) stamps
    ``debug_metadata["scene_health"] = {**report, acknowledged, evaluated_at}``
    onto the solve (in place — additive metadata, geometry untouched), so an
    acknowledged warning rides into every DCC export, review report, and
    manifest downstream. ``pass_through_on_pass`` (default ON) means a clean
    scene flows with zero clicks — the gate only costs friction when
    something is actually wrong.
    """

    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "gate"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "source_image": ("IMAGE",),
            },
            "optional": {
                "depth": ("ATLAS_DEPTH_MAP", {
                    "tooltip": "Shared AtlasDepthMap — enables the negative-depth check."}),
                "status_1": ("STRING", {"forceInput": True, "default": ""}),
                "status_2": ("STRING", {"forceInput": True, "default": ""}),
                "status_3": ("STRING", {"forceInput": True, "default": ""}),
                "status_4": ("STRING", {"forceInput": True, "default": ""}),
                "pass_through_on_pass": ("BOOLEAN", {"default": True,
                    "tooltip": "A PASS-level report flows without a click; warn/fail "
                               "always needs ✅ Acknowledge & Continue."}),
                "proceed": ("BOOLEAN", {"default": False,
                    "tooltip": "Acknowledge the current flags and let the solve flow. "
                               "Ships OFF; the ✅ button sets it + the fingerprint."}),
                "approved_for": ("STRING", {"default": "",
                    "tooltip": "Fingerprint of the acknowledged solve+image; a re-solve "
                               "or swapped photo re-arms the gate."}),
            },
        }

    def gate(self, solve, source_image, depth=None,
             status_1="", status_2="", status_3="", status_4="",
             pass_through_on_pass=True, proceed=False, approved_for="", **_extra):
        import datetime

        from atlas_camera.core.scene_health import evaluate_scene_health

        statuses = {f"status_{i}": s for i, s in
                    enumerate((status_1, status_2, status_3, status_4), 1) if s}
        health = evaluate_scene_health(
            solve, depth, scope_statuses=statuses,
            matte_coverage_fn=AtlasDebugReport._matte_coverage)
        fp = _solve_fingerprint(solve, source_image)
        approval = bool(proceed) and (not approved_for or approved_for == fp)
        effective = (health.level == "pass" and bool(pass_through_on_pass)) or approval

        # The indelible stamp — acknowledged=True only for an explicit
        # override of a warn/fail report, never for a clean pass-through.
        meta = getattr(solve, "debug_metadata", None)
        if isinstance(meta, dict):
            meta["scene_health"] = {
                **health.to_dict(),
                "acknowledged": bool(approval and health.level != "pass"),
                "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }

        marks = {"fail": "✖", "warn": "⚠"}
        lines = []
        if health.level == "pass":
            lines.append("🩺 SCENE HEALTH: PASS — no flags."
                         + ("" if effective else "  (pass_through_on_pass is OFF — ✅ to continue.)"))
        else:
            state = ("acknowledged — exporting WITH warnings recorded" if effective
                     else "downstream paused. Review, fix, or ✅ Acknowledge & Continue.")
            lines.append(f"🩺 SCENE HEALTH: {health.level.upper()} "
                         f"({len(health.flags)} flag(s)) — {state}")
            for f in health.flags:
                lines.append(f"  {marks.get(f.severity, '•')} {f.message}")
        if proceed and approved_for and approved_for != fp:
            lines.insert(0, "*** GATE RE-ARMED: the solve or image changed since the "
                            "acknowledgement — review and ✅ again. ***")
        report = "\n".join(lines)

        if effective:
            out = solve
        else:
            blocker = _execution_blocker()
            out = blocker if blocker is not None else solve
        return {"ui": {"text": [report], "fingerprint": [fp]},
                "result": (out, report)}


class AtlasVLMScaleCues:
    """Detect scale-reference objects with a local vision-language model.

    Runs a local VLM (LM Studio / llama.cpp / Ollama) to find known-size objects
    (people, doors, cars, …) and emits ``scale_references`` JSON for
    AtlasApplyScaleReferences. Requires a running local VLM server — or, for
    users without local models, the ``openai`` provider: any OpenAI-compatible
    cloud endpoint via ``base_url`` + ``api_key``. The model must return pixel
    bounding boxes. Advisory only — nothing is applied without the artist
    confirming in AtlasApplyScaleReferences.
    """
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("scale_references", "summary")
    FUNCTION = "analyze"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "provider": (["ollama", "lmstudio", "llamacpp", "openai"], {"default": "ollama",
                    "tooltip": "ollama/lmstudio/llamacpp are local; 'openai' is any "
                               "OpenAI-compatible cloud endpoint (needs api_key)."}),
                "model": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": "", "tooltip": "Blank = provider default URL"}),
                "min_confidence": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                # APPENDED last (widgets_values is positional — never insert).
                "api_key": ("STRING", {"default": "",
                    "tooltip": "API key for the 'openai' cloud provider (ignored by local "
                               "providers). SAVED INTO THE WORKFLOW FILE — prefer the "
                               "OPENAI_API_KEY environment variable for shared workflows."}),
            },
        }

    def analyze(self, image, provider="ollama", model="", base_url="", min_confidence=0.0,
                api_key=""):
        from atlas_camera.inference.multimodal_helper import (
            create_multimodal_provider,
            scale_references_from_observation,
        )
        from atlas_camera.reference_data import load_scale_references

        tmp = _save_image_tensor_to_tmp(image)
        try:
            candidate_ids = [r.id for r in load_scale_references()]
            prov = create_multimodal_provider(provider, model=model, base_url=base_url or None,
                                              api_key=api_key.strip() or None)
            obs = prov.analyze_image(tmp, candidate_reference_ids=candidate_ids)
            refs = scale_references_from_observation(obs, min_confidence=min_confidence)
            lines = [obs.summary or "VLM analysis complete."]
            for r in refs:
                target = r.get("reference_id") or f"{r.get('height_m')} m"
                lines.append(f"• {r.get('label')} → {target}  bbox={r['bbox_px']}  conf={r['confidence']:.2f}")
            if not refs:
                lines.append("(no usable scale references detected)")
            return (json.dumps(refs), "\n".join(str(s) for s in lines if s))
        except Exception as exc:  # provider offline / model missing — fail soft
            return ("[]", f"VLM scale cues unavailable: {exc}")
        finally:
            os.unlink(tmp)


class AtlasApplyScaleReferences:
    """Apply VLM/JSON scale references to a solve — only when the artist confirms.

    Takes ``scale_references`` JSON (from AtlasVLMScaleCues or hand-written) and,
    when ``confirm`` is on, rescales the solve's metric camera height via single-view
    geometry. With ``confirm`` off the references are recorded as candidates only
    (LLM cues are never auto-promoted; the toggle is the one-click confirmation).
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "FLOAT", "STRING")
    RETURN_NAMES = ("solve", "camera_height_m", "report")
    FUNCTION = "apply"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "scale_references": ("STRING", {"default": "[]", "multiline": True,
                    "tooltip": "JSON list of scale references (from AtlasVLMScaleCues)."}),
            },
            "optional": {
                "confirm": ("BOOLEAN", {"default": False,
                    "tooltip": "Confirm to actually rescale the camera. Off = record candidates only."}),
                "min_confidence": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    def apply(self, solve, scale_references, confirm=False, min_confidence=0.0):
        from atlas_camera.core.solver import apply_reference_scale
        try:
            refs = json.loads(scale_references) if scale_references.strip() else []
        except json.JSONDecodeError:
            refs = []
        if not isinstance(refs, list):
            refs = []
        if min_confidence > 0:
            refs = [r for r in refs if float(r.get("confidence", 1.0)) >= min_confidence]

        apply_reference_scale(solve, refs, adopt=bool(confirm))
        rs = solve.debug_metadata.get("reference_scale", {})
        report = json.dumps({
            "confirmed": bool(confirm),
            "adopted": rs.get("adopted"),
            "scale_source": solve.debug_metadata.get("scale_source"),
            "camera_height_m": rs.get("camera_height_m"),
            "confidence": rs.get("confidence"),
            "references_in": len(refs),
        }, indent=2)
        return (solve, float(solve.camera.extrinsics.camera_position[1]), report)


class AtlasLoadSolveJSON:
    """Load a previously saved AtlasSolve from a JSON file."""
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "load"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_path": ("STRING", {"default": "atlas_solve.json"}),
            }
        }

    def load(self, json_path):
        return (load_solve_json(json_path),)


class AtlasDecomposeSolve:
    """Break an ATLAS_SOLVE into its typed component outputs."""
    RETURN_TYPES = ("ATLAS_CAMERA", "FLOAT", "STRING", "INT", "INT", "STRING", "FLOAT")
    RETURN_NAMES = ("camera", "confidence", "source_method", "image_width", "image_height", "solve_json", "horizon_angle_deg")
    FUNCTION = "decompose"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"solve": ("ATLAS_SOLVE",)}}

    def decompose(self, solve):
        horizon_deg = float(
            (solve.debug_metadata or {})
            .get("camera_estimation", {})
            .get("horizon_angle", 0.0)
        )
        return (
            solve.camera,
            float(solve.confidence),
            str(solve.source_method),
            int(solve.camera.intrinsics.image_width),
            int(solve.camera.intrinsics.image_height),
            solve.to_json(),
            horizon_deg,
        )


class AtlasDecomposeCamera:
    """Extract intrinsic and extrinsic floats from an ATLAS_CAMERA for downstream routing."""
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT",
                    "FLOAT", "FLOAT", "FLOAT",
                    "FLOAT", "FLOAT")
    RETURN_NAMES = ("fx", "fy", "cx", "cy",
                    "cam_x", "cam_y", "cam_z",
                    "focal_mm", "fov_h_deg")
    FUNCTION = "decompose"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"camera": ("ATLAS_CAMERA",)}}

    def decompose(self, camera):
        intr = camera.intrinsics
        extr = camera.extrinsics
        fx = intr.fx_px or 0.0
        fy = intr.fy_px or fx
        cx = intr.cx_px if intr.cx_px is not None else intr.image_width / 2.0
        cy = intr.cy_px if intr.cy_px is not None else intr.image_height / 2.0
        pos = extr.camera_position
        focal_mm = intr.focal_length_mm or 0.0
        fov_h = 0.0
        if fx > 0 and intr.image_width > 0:
            fov_h = math.degrees(2 * math.atan(intr.image_width / (2.0 * fx)))
        return (
            float(fx), float(fy), float(cx), float(cy),
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(focal_mm), float(fov_h),
        )
