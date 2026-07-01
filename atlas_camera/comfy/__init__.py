"""ComfyUI adapter package.

Importing this package must not require ComfyUI. Nodes wrap Atlas core behavior.
"""

from atlas_camera.comfy.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

