"""ComfyUI clone-and-go entry point for the Atlas Camera node pack.

`git clone` this repository straight into `ComfyUI/custom_nodes/` (or install
it via ComfyUI-Manager / the Comfy Registry) and it loads with NO pip install:
this file puts the checkout on sys.path so `import atlas_camera` resolves
here, then re-exports the node mappings from `atlas_camera.comfy`. Because the
whole checkout is present, the pieces a wheel install lacks (the 🧍/🚗 proxy
OBJ meshes and the example workflows under `examples/`) work too.

The development setup — editable pip install + a `custom_nodes/AtlasCamera`
symlink pointing directly at `atlas_camera/comfy` (see INSTALL.md) — keeps
working unchanged and never loads this file.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    # Prefer this checkout for `import atlas_camera`. Harmless when the
    # package is also pip-installed: the documented dev install is editable
    # and points at this same directory anyway.
    sys.path.insert(0, _HERE)

from atlas_camera.comfy import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

# Relative to THIS module's directory, per ComfyUI convention.
WEB_DIRECTORY = "./atlas_camera/comfy/web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
