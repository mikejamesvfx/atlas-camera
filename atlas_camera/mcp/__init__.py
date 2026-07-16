"""Atlas Camera MCP server — expose a running ComfyUI + Atlas node pack to
MCP-capable assistants. Run with ``python -m atlas_camera.mcp``.

The heavy lifting lives in :mod:`atlas_camera.mcp.comfy_http` (stdlib-only
UI→API flattening / validation / queueing — importable without the ``mcp``
SDK) and :mod:`atlas_camera.mcp.server` (the FastMCP tool surface; needs the
``[mcp]`` extra).
"""
