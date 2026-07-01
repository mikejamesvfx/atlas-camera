"""Run the Atlas Camera local UI backend."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Atlas Camera local UI backend.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8787, help="Port to bind.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload for development.")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Atlas UI serving requires uvicorn. Install atlas-camera[ui].") from exc

    uvicorn.run("atlas_camera.ui.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
