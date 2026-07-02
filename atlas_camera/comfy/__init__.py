"""ComfyUI adapter package.

Importing this package must not require ComfyUI. Nodes wrap Atlas core behaviour.
When ComfyUI loads this package it will also discover WEB_DIRECTORY and register
the Atlas Blockout frontend extension.
"""

from __future__ import annotations

import os
from pathlib import Path

from atlas_camera.comfy.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Proxy scale-reference meshes (examples/models/*.obj) served to the blockout viewport.
_ATLAS_MODELS_DIR = Path(__file__).resolve().parents[2] / "examples" / "models"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# ComfyUI reads this to auto-register any *.js files as frontend extensions.
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

# ---------------------------------------------------------------------------
# Optional: register API routes if PromptServer is available (ComfyUI context).
# This is a no-op when the package is imported outside ComfyUI.
# ---------------------------------------------------------------------------
try:
    from aiohttp import web as aiohttp_web
    from server import PromptServer  # type: ignore[import]

    from atlas_camera.comfy.nodes import _ATLAS_BLOCKOUT_CACHE

    _routes = PromptServer.instance.routes
    _ATLAS_ROUTE_PATH = "/atlas/camera_data/{node_id}"

    # Guard against double-registration: this __init__.py is loaded twice when
    # ComfyUI loads it as a custom node (AtlasCamera) AND Python imports it as
    # atlas_camera.comfy — both under different sys.modules keys so the cache
    # doesn't deduplicate them.
    if not any(getattr(r, "path", None) == _ATLAS_ROUTE_PATH for r in _routes):

        @_routes.get(_ATLAS_ROUTE_PATH)
        async def _atlas_get_camera_data(request: aiohttp_web.Request) -> aiohttp_web.Response:
            node_id = request.match_info["node_id"]
            data = _ATLAS_BLOCKOUT_CACHE.get(node_id, {})
            return aiohttp_web.json_response(data)

    _ATLAS_MODEL_ROUTE_PATH = "/atlas/proxy_model/{name}"
    if not any(getattr(r, "path", None) == _ATLAS_MODEL_ROUTE_PATH for r in _routes):

        @_routes.get(_ATLAS_MODEL_ROUTE_PATH)
        async def _atlas_get_proxy_model(request: aiohttp_web.Request) -> aiohttp_web.Response:
            # Serve scale-reference OBJ/MTL meshes to the blockout viewport. Restrict
            # to basenames inside the models dir (no path traversal).
            name = os.path.basename(request.match_info["name"])
            if not name.lower().endswith((".obj", ".mtl")):
                return aiohttp_web.Response(status=400, text="Only .obj/.mtl are served.")
            path = _ATLAS_MODELS_DIR / name
            if not path.is_file():
                return aiohttp_web.Response(status=404, text=f"No such model: {name}")
            return aiohttp_web.FileResponse(path)

except Exception:
    # Running outside ComfyUI (tests, standalone import) — routes not needed.
    pass
