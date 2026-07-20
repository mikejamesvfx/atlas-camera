"""ComfyUI adapter package.

Importing this package must not require ComfyUI. Nodes wrap Atlas core behaviour.
When ComfyUI loads this package it will also discover WEB_DIRECTORY and register
the Atlas Blockout frontend extension.
"""

from __future__ import annotations

import os

from atlas_camera.comfy.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# ComfyUI reads this to auto-register any *.js files as frontend extensions.
# RELATIVE per convention (ComfyUI joins it onto the custom node's own dir;
# registry/manager tooling assumes relative) — was an absolute path, which
# only worked because os.path.join ignores the left side when the right side
# is absolute.
WEB_DIRECTORY = "./web"

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

        @_routes.get("/atlas/recipes/{recipe_name}")
        async def _atlas_get_recipe(request: aiohttp_web.Request) -> aiohttp_web.Response:
            recipe_name = request.match_info["recipe_name"]
            # A simple dynamic graph for the requested recipe
            graph = {
                "nodes": [
                    {"id": 1, "type": "AtlasInput", "pos": [100, 100], "flags": {}, "order": 0, "mode": 0, "inputs": [], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}], "properties": {}, "widgets_values": []},
                    {"id": 2, "type": "AtlasMegaPipeline", "pos": [500, 100], "flags": {}, "order": 1, "mode": 0, "inputs": [{"name": "image", "type": "IMAGE", "link": 1}], "outputs": [{"name": "maya_scene_path", "type": "STRING", "links": []}], "properties": {}, "widgets_values": ["atlas_exports", 1.6, "outdoor"]}
                ],
                "links": [
                    [1, 1, 0, 2, 0, "IMAGE"]
                ],
                "groups": [],
                "config": {},
                "extra": {},
                "version": 0.4
            }
            return aiohttp_web.json_response(graph)

    # (The /atlas/proxy_model route was removed 2026-07-12 for the public
    # release: its only frontend callers — the viewport's OBJ scale-proxy
    # buttons — were removed 2026-07-09, and examples/models no longer ships.)

except Exception:
    # Running outside ComfyUI (tests, standalone import) — routes not needed.
    pass
