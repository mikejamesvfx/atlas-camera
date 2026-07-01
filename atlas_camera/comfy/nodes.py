"""Initial ComfyUI node scaffolds for Atlas Camera."""

from __future__ import annotations

from atlas_camera.core.io import load_solve_json, save_solve_json
from atlas_camera.core.solver import solve_still_image
from atlas_camera.exporters.review_package import build_review_package
from atlas_camera.importers.usd_camera_loader import USDCameraLoader


class AtlasLoadImageSolveCamera:
    RETURN_TYPES = ("ATLAS_SOLVE",)
    FUNCTION = "solve"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": ""}),
                "image_width": ("INT", {"default": 1024, "min": 1}),
                "image_height": ("INT", {"default": 1024, "min": 1}),
            },
            "optional": {
                "focal_length_mm": ("FLOAT", {"default": 35.0, "min": 0.0}),
                "sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 0.01}),
            },
        }

    def solve(
        self,
        image_path: str,
        image_width: int,
        image_height: int,
        focal_length_mm: float | None = None,
        sensor_width_mm: float = 36.0,
    ):
        hints = {}
        if focal_length_mm:
            hints["focal_length_mm"] = focal_length_mm
            hints["sensor_width_mm"] = sensor_width_mm
        return (
            solve_still_image(
                image_path,
                image_size=(image_width, image_height),
                intrinsics_hint=hints,
            ),
        )


class AtlasExportReviewPackage:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "export"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "review_packages"}),
            }
        }

    def export(self, solve, output_dir: str):
        result = build_review_package(solve, output_dir)
        return (str(result.package_dir),)


class AtlasExportSolveJSON:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "export"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_path": ("STRING", {"default": "atlas_solve.json"}),
            }
        }

    def export(self, solve, output_path: str):
        return (str(save_solve_json(solve, output_path)),)


class AtlasExportMayaReviewScene:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "export"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "output_dir": ("STRING", {"default": "review_packages"}),
            }
        }

    def export(self, solve, output_dir: str):
        result = build_review_package(solve, output_dir, include_usd=False)
        return (str(result.files["maya_open_scene"]),)


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

    def load(self, usd_path: str, image_width: int, image_height: int):
        return (USDCameraLoader().load(usd_path, image_size=(image_width, image_height)),)


NODE_CLASS_MAPPINGS = {
    "AtlasLoadImageSolveCamera": AtlasLoadImageSolveCamera,
    "AtlasExportReviewPackage": AtlasExportReviewPackage,
    "AtlasExportSolveJSON": AtlasExportSolveJSON,
    "AtlasExportMayaReviewScene": AtlasExportMayaReviewScene,
    "AtlasUSDCameraLoader": AtlasUSDCameraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AtlasLoadImageSolveCamera": "Atlas Load Image / Solve Camera",
    "AtlasExportReviewPackage": "Atlas Export Review Package",
    "AtlasExportSolveJSON": "Atlas Export Solve JSON",
    "AtlasExportMayaReviewScene": "Atlas Export Maya Review Scene",
    "AtlasUSDCameraLoader": "Atlas USD Camera Loader",
}

