#!/usr/bin/env python
"""Atlas Camera cross-machine smoke check.

Run this on each test machine to produce ONE copy-pasteable status line for the
README "Tested on" matrix, plus a short per-tier report. It deliberately walks
the real dependency tiers, so a no-GPU / no-torch machine proves the
"pure NumPy basics, no GPU" claim *by itself*:

    python tools/smoke_check.py

Tiers
-----
  core     zero-dependency: ``import atlas`` -> metadata-only camera solve ->
           export a Nuke .nk script. No numpy, no torch, no image file.
  nodes    import the ComfyUI node mappings and count them. Proves all nodes
           register with NO heavy deps present (torch is imported lazily inside
           the neural nodes, never at module load).
  vision   [vision] extra (numpy + opencv): run the geometric vanishing-point
           solve on a synthetic image. SKIPPED (not failed) if cv2 is absent.
  neural   [neural] extra (torch): report torch version + the compute device
           (cuda / mps / cpu). SKIPPED if torch is absent. Does NOT download
           GeoCalib weights — availability only.

Exit code is 0 when the two load-bearing tiers (core + nodes) pass; the optional
tiers never fail the run when their deps are missing — that is the entire point.
"""
from __future__ import annotations

import platform
import sys
import tempfile
from pathlib import Path

# Prefer an installed atlas (so this validates the packed archive / pip install);
# fall back to the checkout when run straight from a source tree.
try:  # noqa: SIM105
    import atlas  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _ver(mod_name: str) -> str | None:
    try:
        mod = __import__(mod_name)
    except Exception:
        return None
    return getattr(mod, "__version__", "?")


def _fmt(ok: bool | None) -> str:
    return {True: "PASS", False: "FAIL", None: "skip"}[ok]


def tier_core() -> tuple[bool, str]:
    """Zero-dep solve + export. No numpy / torch / image file required."""
    try:
        import atlas
        import atlas_camera
        from atlas_camera.exporters.nuke_exporter import write_nuke_native_script

        # detect_vanishing_points=False + explicit image_size never reads a file.
        solve = atlas.recover("synthetic.jpg", image_size=(1920, 1080))
        with tempfile.TemporaryDirectory() as td:
            nk = write_nuke_native_script(solve, Path(td) / "smoke.nk")
            wrote = nk.exists() and nk.stat().st_size > 0
        ver = getattr(atlas_camera, "__version__", "?")
        return (bool(wrote), f"solve ok (atlas {ver}), .nk written={wrote}")
    except Exception as exc:  # pragma: no cover - reported, not raised
        return (False, f"{type(exc).__name__}: {exc}")


def tier_nodes() -> tuple[bool, str, int]:
    """All ComfyUI nodes must register with no heavy deps present."""
    try:
        from atlas_camera.comfy import NODE_CLASS_MAPPINGS

        n = len(NODE_CLASS_MAPPINGS)
        return (n > 0, f"{n} nodes registered", n)
    except Exception as exc:  # pragma: no cover
        return (False, f"{type(exc).__name__}: {exc}", 0)


def tier_vision() -> tuple[bool | None, str]:
    """Geometric VP solve — numpy + opencv. Skipped if cv2 absent."""
    try:
        import cv2  # noqa: F401
        import numpy as np
    except Exception:
        return (None, "cv2/numpy not installed (expected on a core-only box)")
    try:
        import atlas

        # Synthetic frame with converging lines -> a plausible VP scene.
        img = np.full((720, 1280, 3), 255, np.uint8)
        for x in range(-1280, 1281, 120):
            cv2.line(img, (640, 240), (x, 720), (30, 30, 30), 2)
        for x in range(0, 1281, 160):
            cv2.line(img, (x, 300), (640, 240), (30, 30, 30), 1)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "synthetic_vp.png"
            cv2.imwrite(str(p), img)
            solve = atlas.recover(
                str(p), method="vanishing_points", detect_vanishing_points=True
            )
        vps = getattr(solve, "vanishing_points", None) or []
        return (True, f"VP pipeline ran (numpy {np.__version__}, cv2 {cv2.__version__}), vps_found={len(vps)}")
    except Exception as exc:  # pragma: no cover
        return (False, f"{type(exc).__name__}: {exc}")


def tier_neural() -> tuple[bool | None, str, str]:
    """torch availability + compute device. Skipped if torch absent."""
    try:
        import torch
    except Exception:
        return (None, "torch not installed (core-only / no-GPU box)", "none")
    device = "cpu"
    try:
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.get_device_name(0)}"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass
    return (True, f"torch {torch.__version__}, device={device}", device.split(":")[0])


def main() -> int:
    os_line = f"{platform.system()} {platform.release()} ({platform.machine()})"
    py = platform.python_version()

    core_ok, core_msg = tier_core()
    nodes_ok, nodes_msg, n_nodes = tier_nodes()
    vis_ok, vis_msg = tier_vision()
    neu_ok, neu_msg, device = tier_neural()

    print("=" * 70)
    print("Atlas Camera smoke check")
    print("=" * 70)
    print(f"  OS       : {os_line}")
    print(f"  Python   : {py}")
    print(f"  numpy    : {_ver('numpy')}")
    print(f"  opencv   : {_ver('cv2')}")
    print(f"  torch    : {_ver('torch')}")
    print("-" * 70)
    print(f"  [core  ] {_fmt(core_ok)}  {core_msg}")
    print(f"  [nodes ] {_fmt(nodes_ok)}  {nodes_msg}")
    print(f"  [vision] {_fmt(vis_ok)}  {vis_msg}")
    print(f"  [neural] {_fmt(neu_ok)}  {neu_msg}")
    print("-" * 70)

    # One copy-pasteable line for the README "Tested on" matrix.
    print(
        "TESTED-ON | "
        f"{os_line} | py{py} | "
        f"numpy={_ver('numpy') or '-'} | torch={_ver('torch') or '-'} | "
        f"device={device} | nodes={n_nodes} | "
        f"core={_fmt(core_ok)} vision={_fmt(vis_ok)} neural={_fmt(neu_ok)}"
    )
    print("=" * 70)

    return 0 if (core_ok and nodes_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
