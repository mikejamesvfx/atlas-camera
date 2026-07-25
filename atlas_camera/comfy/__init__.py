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
# Opt-in workaround: ATLAS_DISABLE_CUDNN_SDPA=1 disables torch's cuDNN SDPA
# backend for the whole process. On torch 2.12+cu130 / RTX 5090 that backend
# intermittently fails VAE-encode attention with "RuntimeError: query is not
# correctly aligned (strideM)" (seen live killing AtlasSDXLInpaint's
# InpaintModelConditioning; NOT reproducible with isolated same-shape tensors,
# so it is allocator-state dependent, and ComfyUI exposes no switch of its
# own). Flash/mem-efficient backends take over; VAE cost is negligible.
# OPT-IN by env var precisely because flipping a global torch backend from a
# node pack silently would be bad citizenship. Harmless if run twice (this
# module loads twice at ComfyUI startup — see the route guard below).
if os.environ.get("ATLAS_DISABLE_CUDNN_SDPA", "").strip().lower() not in ("", "0", "false", "off", "no"):
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.backends.cuda.enable_cudnn_sdp(False)
            import logging as _logging
            _logging.getLogger(__name__).info(
                "[AtlasCamera] cuDNN SDPA disabled (ATLAS_DISABLE_CUDNN_SDPA) — "
                "strideM workaround; flash/mem-efficient backends remain active.")
    except Exception:
        pass  # torch absent or too old — nothing to disable

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

    # (The /atlas/proxy_model route was removed 2026-07-12 for the public
    # release: its only frontend callers — the viewport's OBJ scale-proxy
    # buttons — were removed 2026-07-09, and examples/models no longer ships.)

except Exception:
    # Running outside ComfyUI (tests, standalone import) — routes not needed.
    pass
