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

# Reactive rescue for the same bug, ALWAYS on (unlike the proactive opt-in
# above) because it is a no-op until the failure actually happens: wrap
# comfy.ops.scaled_dot_product_attention and, only when the cuDNN/cutlass
# alignment error fires, retry with freshly-allocated clones (a fresh
# allocation is pointer-aligned; the live failures are allocator-state
# dependent) and finally with the math backend. Without this the queue dies
# mid-graph, so a retry cannot make anything worse. Guarded against the
# double import.
try:
    import comfy.ops as _comfy_ops  # only resolvable inside ComfyUI

    if not getattr(_comfy_ops, "_atlas_sdpa_rescue", False):
        _orig_sdpa = _comfy_ops.scaled_dot_product_attention

        def _atlas_sdpa(q, k, v, *args, **kwargs):
            try:
                return _orig_sdpa(q, k, v, *args, **kwargs)
            except RuntimeError as exc:
                if "aligned" not in str(exc):
                    raise
                import logging
                import torch as _t
                log = logging.getLogger("atlas_camera")
                try:
                    out = _orig_sdpa(q.contiguous().clone(), k.contiguous().clone(),
                                     v.contiguous().clone(), *args, **kwargs)
                    log.warning("[AtlasCamera] SDPA alignment error rescued via "
                                "fresh-allocation retry (%s)", str(exc)[:60])
                    return out
                except RuntimeError:
                    from torch.nn.attention import SDPBackend, sdpa_kernel
                    with sdpa_kernel([SDPBackend.MATH]):
                        out = _t.nn.functional.scaled_dot_product_attention(
                            q, k, v, *args, **kwargs)
                    log.warning("[AtlasCamera] SDPA alignment error rescued via "
                                "math backend (%s)", str(exc)[:60])
                    return out

        _comfy_ops.scaled_dot_product_attention = _atlas_sdpa
        _comfy_ops._atlas_sdpa_rescue = True
except Exception:
    pass  # outside ComfyUI, or comfy internals moved — never block import

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

    _ATLAS_DYNAMIC_ROUTE_PATH = "/atlas/dynamic_plate/{key}/{index}"
    if not any(getattr(r, "path", None) == _ATLAS_DYNAMIC_ROUTE_PATH
               for r in _routes):

        @_routes.get(_ATLAS_DYNAMIC_ROUTE_PATH)
        async def _atlas_get_dynamic_frame(request: aiohttp_web.Request) -> aiohttp_web.Response:
            # Frames are served only for packages AtlasLoadDynamicPlate has
            # registered (opaque tokens — never raw filesystem paths).
            from atlas_camera.comfy.nodes_dynamic import registered_plate_dir

            package_dir = registered_plate_dir(request.match_info["key"])
            if package_dir is None:
                return aiohttp_web.Response(status=404)
            try:
                index = int(request.match_info["index"])
            except ValueError:
                return aiohttp_web.Response(status=400)
            want_matte = request.query.get("matte") == "1"
            stem = "matte" if want_matte else "frame"
            frame = package_dir / "generated" / f"{stem}_{index:04d}.png"
            if not frame.exists():
                frame = package_dir / "source" / "crop.png"
                if want_matte or index != 0 or not frame.exists():
                    return aiohttp_web.Response(status=404)
            return aiohttp_web.FileResponse(frame)

    # Automatic end-of-run viewport snapshots (2026-08-16): the frontend POSTs
    # two PNGs (📽 on / off, recovered camera, long edge 1280) after every
    # viewport execution; they land under <output>/atlas_viewport/ and their
    # paths ride the camera_data payload so agents can find them via MCP.
    _ATLAS_SNAPSHOT_ROUTE_PATH = "/atlas/viewport_snapshot"
    if not any(getattr(r, "path", None) == _ATLAS_SNAPSHOT_ROUTE_PATH
               for r in _routes):

        @_routes.post(_ATLAS_SNAPSHOT_ROUTE_PATH)
        async def _atlas_post_viewport_snapshot(request: aiohttp_web.Request) -> aiohttp_web.Response:
            from atlas_camera.comfy.viewport_snapshot import (
                attach_snapshot_to_cache, save_viewport_snapshot,
            )
            try:
                import folder_paths  # type: ignore[import]
                out_dir = folder_paths.get_output_directory()
            except Exception:  # noqa: BLE001
                out_dir = "output"
            try:
                payload = await request.json()
                record = save_viewport_snapshot(payload, output_dir=out_dir)
            except ValueError as exc:
                return aiohttp_web.json_response({"error": str(exc)}, status=400)
            except Exception as exc:  # noqa: BLE001
                return aiohttp_web.json_response({"error": str(exc)}, status=500)
            attach_snapshot_to_cache(_ATLAS_BLOCKOUT_CACHE, record)
            return aiohttp_web.json_response(record)

    # Agent handoff (2026-08-16): AtlasAgentHandoff pauses a graph and waits
    # for resume.json; these routes let any agent read the brief and resume
    # over HTTP (the MCP tools write the same files).
    _ATLAS_AGENT_BRIEF_ROUTE = "/atlas/agent/brief/{node_id}"
    _ATLAS_AGENT_RESUME_ROUTE = "/atlas/agent/resume/{node_id}"
    if not any(getattr(r, "path", None) == _ATLAS_AGENT_BRIEF_ROUTE for r in _routes):

        @_routes.get(_ATLAS_AGENT_BRIEF_ROUTE)
        async def _atlas_get_agent_brief(request: aiohttp_web.Request) -> aiohttp_web.Response:
            from atlas_camera.comfy.agent_handoff import read_brief
            brief = read_brief(request.match_info["node_id"])
            if brief is None:
                return aiohttp_web.json_response({"error": "no brief for this node"}, status=404)
            return aiohttp_web.json_response(brief)

        @_routes.post(_ATLAS_AGENT_RESUME_ROUTE)
        async def _atlas_post_agent_resume(request: aiohttp_web.Request) -> aiohttp_web.Response:
            from atlas_camera.comfy.agent_handoff import write_resume
            try:
                payload = await request.json()
                rec = write_resume(request.match_info["node_id"], payload)
            except ValueError as exc:
                return aiohttp_web.json_response({"error": str(exc)}, status=400)
            except Exception as exc:  # noqa: BLE001
                return aiohttp_web.json_response({"error": str(exc)}, status=500)
            return aiohttp_web.json_response(rec)

    # Atlas Director launch/delivery (2026-08-30): starts Director on a
    # session package and remembers the take it pushes back. See
    # atlas_camera/comfy/director_session.py for the security posture -- the
    # executable is configuration-only, the request carries no argv, the
    # session path is checked against the output root, and the session id is
    # refused rather than sanitised when it isn't a plain slug.
    _ATLAS_DIRECTOR_LAUNCH = "/atlas/director/launch"
    if not any(getattr(r, "path", None) == _ATLAS_DIRECTOR_LAUNCH for r in _routes):

        @_routes.post(_ATLAS_DIRECTOR_LAUNCH)
        async def _atlas_director_launch(request: aiohttp_web.Request):
            from atlas_camera.comfy.director_session import launch_session

            try:
                session = launch_session(await request.json())
            except (ValueError, KeyError) as error:
                return aiohttp_web.json_response({"error": str(error)}, status=400)
            except RuntimeError as error:
                return aiohttp_web.json_response({"error": str(error)}, status=503)
            return aiohttp_web.json_response(session)

    _ATLAS_DIRECTOR_TAKE = "/atlas/director/take"
    if not any(getattr(r, "path", None) == _ATLAS_DIRECTOR_TAKE for r in _routes):

        @_routes.post(_ATLAS_DIRECTOR_TAKE)
        async def _atlas_director_take(request: aiohttp_web.Request):
            from atlas_camera.comfy.director_session import record_delivery

            body = await request.json()
            try:
                session = record_delivery(
                    body["session_id"], body["slate"], body["take_dir"]
                )
            except KeyError:
                return aiohttp_web.json_response(
                    {"error": "unknown session; re-launch or read the slate directly"},
                    status=404,
                )
            return aiohttp_web.json_response(session)

except Exception:
    # Running outside ComfyUI (tests, standalone import) — routes not needed.
    pass
