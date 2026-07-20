"""Atlas ComfyUI nodes — experimental tier.

These nodes present completely new paradigms or heavy abstractions that bypass
standard ComfyUI graph execution. They sit behind the ATLAS_EXPERIMENTAL gate.
"""
from __future__ import annotations

import logging
from typing import Any

from atlas_camera.comfy.nodes_solve import AtlasLearnedSolveFromImage
from atlas_camera.comfy.nodes_depth import AtlasDepthMap
from atlas_camera.comfy.nodes_geometry import AtlasDeriveProjectionGeometry
from atlas_camera.comfy.nodes_export import AtlasExportMayaReviewScene

class AtlasMegaPipeline:
    """A mega-node that abstracts the entire Atlas Camera pipeline into a single node.
    Takes an image, runs learned solve, extracts depth, builds geometry, and exports to Maya.
    
    Subverts the normal ComfyUI graph execution by instantiating and calling downstream
    nodes directly in Python.
    """
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("maya_scene_path",)
    FUNCTION = "execute_pipeline"
    CATEGORY = "Atlas Camera"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_dir": ("STRING", {"default": "atlas_exports"}),
            },
            "optional": {
                "camera_height_m": ("FLOAT", {"default": 1.6, "min": 0.01, "max": 1000.0}),
                "scene_type": (["manual", "organic", "indoor", "outdoor"], {"default": "outdoor"}),
            },
        }

    def execute_pipeline(self, image: Any, output_dir: str, camera_height_m: float = 1.6, scene_type: str = "outdoor"):
        logging.info("[AtlasMegaPipeline] Starting monolithic execution...")
        
        # 1. Solve Camera
        logging.info("  -> Solving Camera (GeoCalib)...")
        solver = AtlasLearnedSolveFromImage()
        solve_result = solver.solve(
            image=image, 
            height_mode="measure_from_depth", 
            camera_height_m=camera_height_m, 
            depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", 
            sensor_width_mm=36.0, 
            weights="pinhole", 
            device="auto", 
            focal_length_mm=0.0
        )[0]
        
        # 2. Extract Depth
        logging.info("  -> Extracting Metric Depth...")
        depth_extractor = AtlasDepthMap()
        depth_result = depth_extractor.estimate(
            image=image, 
            depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", 
            device="auto",
            solve=solve_result
        )[0]
        
        # 3. Derive Geometry
        logging.info("  -> Deriving Projection Geometry...")
        geom_deriver = AtlasDeriveProjectionGeometry()
        geom_solve = geom_deriver.derive(
            solve=solve_result, 
            image=image, 
            depth=depth_result, 
            geometry_mode="both", 
            primitive_method="ransac_planes", 
            scene_type=scene_type
        )[0]
        
        # 4. Export to Maya
        logging.info("  -> Exporting Maya Review Scene...")
        exporter = AtlasExportMayaReviewScene()
        maya_path = exporter.export(
            solve=geom_solve, 
            output_dir=output_dir, 
            relief_mesh_obj_path=""
        )[0]
        
        logging.info(f"[AtlasMegaPipeline] Done! Exported to: {maya_path}")
        return (maya_path,)
