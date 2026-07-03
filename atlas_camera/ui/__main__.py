"""Run the Atlas Camera local UI backend."""

from __future__ import annotations

import argparse


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Atlas Camera local UI backend.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8787, help="Port to bind.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload for development.")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help=(
            "Required alongside a non-loopback --host. The API accepts a client-supplied "
            "project_dir on nearly every endpoint (by design, as a local file picker) and "
            "reads/writes files under it with no further access control — fine for a single "
            "local user, but binding this off loopback exposes that filesystem access to "
            "anyone who can reach the port."
        ),
    )
    args = parser.parse_args()

    if args.host not in _LOOPBACK_HOSTS and not args.allow_remote:
        raise SystemExit(
            f"Refusing to bind --host {args.host!r} without --allow-remote: this server grants "
            "filesystem read/write under whatever project_dir a client supplies. Pass "
            "--allow-remote only if you understand and accept that on this network."
        )

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Atlas UI serving requires uvicorn. Install atlas-camera[ui].") from exc

    uvicorn.run("atlas_camera.ui.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
