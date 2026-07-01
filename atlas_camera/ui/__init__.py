"""Optional local web UI for Atlas Camera.

Importing :mod:`atlas_camera.ui` does not import FastAPI. The ASGI app lives in
``atlas_camera.ui.api`` so the core package remains lightweight unless the UI is
explicitly used.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
