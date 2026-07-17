"""Scale reference + camera-body registries for artist-guided solves."""

from atlas_camera.reference_data.camera_registry import (
    CameraBody,
    find_camera_body,
    load_camera_bodies,
)
from atlas_camera.reference_data.registry import (
    ScaleReference,
    get_scale_reference,
    list_categories,
    load_scale_references,
    search_scale_references,
)

__all__ = [
    "CameraBody",
    "ScaleReference",
    "find_camera_body",
    "get_scale_reference",
    "list_categories",
    "load_camera_bodies",
    "load_scale_references",
    "search_scale_references",
]
