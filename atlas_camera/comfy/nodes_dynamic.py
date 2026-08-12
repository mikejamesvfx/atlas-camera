"""Dynamic-plate ComfyUI nodes: bring a generated water plate into the viewport.

`AtlasLoadDynamicPlate` (experimental) reads a dynamic-plate artifact package
(built by ``python -m atlas_camera.dynamic``) and appends its receiver plane +
temporal projection to an ATLAS_SOLVE as a `ProjectionSource`, so the Atlas
viewport shows the animated water projected through the FIXED crop camera
while the viewport camera moves freely — the in-Comfy mirror of the Blender
handoff.

Frame streaming: the payload embeds only frame 0 (base texture). The full
sequence is served per-frame over HTTP (`/atlas/dynamic_plate/{key}/{index}`,
registered in ``comfy/__init__``) from a module-level registry of packages
this node has loaded — the frontend fetches frames lazily and swaps the
projection texture on its existing render ticker. The registry is keyed by
opaque tokens so the route can never serve arbitrary paths.
"""
from __future__ import annotations

import base64
import copy
import uuid

from pathlib import Path
from typing import Any

from atlas_camera.core.schema import ProjectionSource

# Opaque token -> package dir. Populated by node execution; read by the
# /atlas/dynamic_plate route. Deliberately process-global (the route lives in
# a different module instance thanks to the double-import quirk).
_DYNAMIC_PLATE_DIRS: dict[str, Path] = {}
_MAX_REGISTERED = 32


def registered_plate_dir(key: str) -> Path | None:
    return _DYNAMIC_PLATE_DIRS.get(key)


def _register_plate_dir(package_dir: Path) -> str:
    for key, existing in _DYNAMIC_PLATE_DIRS.items():
        if existing == package_dir:
            return key
    if len(_DYNAMIC_PLATE_DIRS) >= _MAX_REGISTERED:
        _DYNAMIC_PLATE_DIRS.pop(next(iter(_DYNAMIC_PLATE_DIRS)))
    key = uuid.uuid4().hex
    _DYNAMIC_PLATE_DIRS[key] = package_dir
    return key


def _image_data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


class AtlasLoadDynamicPlate:
    """Append a dynamic-plate package as an animated projection layer.

    The plate's crop camera stays the projector; the sequence plays in the
    viewport on the render ticker. Static scene layers in the solve occlude
    the receiver through normal depth testing.
    """

    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "load"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE", {"tooltip": "Solve to receive the dynamic plate layer."}),
                "package_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Dynamic-plate package directory "
                               "(dynamic/WATER_0001 — contains manifest.json)."}),
            },
            "optional": {
                "priority": ("FLOAT", {
                    "default": 5.0, "min": -100.0, "max": 100.0, "step": 0.5,
                    "tooltip": "Layer priority among projection sources "
                               "(farthest-highest seam doctrine)."}),
                "fps_override": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 120.0, "step": 1.0,
                    "tooltip": "Playback fps in the viewport; 0 = the "
                               "plate's own frame rate."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, solve, package_dir, priority=5.0, fps_override=0.0):
        # Content fingerprint (gate doctrine): a path widget that gates what
        # gets loaded must re-execute when the package content changes.
        try:
            manifest = Path(package_dir) / "manifest.json"
            stat = manifest.stat()
            generated = sorted(Path(package_dir).glob("generated/frame_*.png"))
            return f"{stat.st_mtime_ns}:{len(generated)}:{priority}:{fps_override}"
        except OSError:
            return f"missing:{package_dir}"

    def load(self, solve, package_dir, priority=5.0, fps_override=0.0):
        from atlas_camera.exporters.dynamic_plate_package import (
            load_dynamic_plate,
        )

        pkg = Path(package_dir)
        if not (pkg / "manifest.json").exists():
            return (solve, f"SKIPPED — no manifest.json in {package_dir!r}; "
                           f"point package_dir at a dynamic/<PLATE_ID> "
                           f"package (see docs/DYNAMIC_PLATES.md)")
        try:
            plate = load_dynamic_plate(pkg)
        except Exception as exc:  # noqa: BLE001 - report, never crash the graph
            return (solve, f"SKIPPED — manifest unreadable: {exc}")
        if plate.crop_camera is None or plate.receiver is None or \
                plate.receiver.primitive is None:
            return (solve, "SKIPPED — plate has no crop camera or receiver "
                           "plane (rebuild with the dynamic CLI)")

        frames = sorted((pkg / "generated").glob("frame_*.png"))
        still = pkg / "source" / "crop.png"
        base_image = frames[0] if frames else still
        if not base_image.exists():
            return (solve, "SKIPPED — package has neither generated frames "
                           "nor source/crop.png")

        key = _register_plate_dir(pkg)
        fps = float(fps_override) if fps_override else float(plate.frame_rate)
        receiver_prim = copy.deepcopy(plate.receiver.primitive)
        # Only role == PROXY_ROLE primitives reach the viewport
        # (projection_scene doctrine); keep the plate identity alongside.
        from atlas_camera.core.proxy_geometry import PROXY_ROLE
        receiver_prim.metadata = dict(receiver_prim.metadata or {})
        receiver_prim.metadata["role"] = PROXY_ROLE
        receiver_prim.metadata["dynamic_plate_role"] = "receiver"
        source = ProjectionSource(
            camera=copy.deepcopy(plate.crop_camera),
            name=f"dynamic_plate_{plate.plate_id.lower()}",
            image_b64=_image_data_uri(base_image),
            proxy_geometry=[receiver_prim],
            priority=float(priority),
            metadata={
                "projection_mode": "clean_plate",
                "evidence_type": "generated",
                "dynamic_plate": {
                    "key": key,
                    "frame_count": len(frames),
                    "fps": fps,
                    "plate_id": plate.plate_id,
                    "semantic_type": plate.semantic_type,
                },
            },
        )
        out = copy.deepcopy(solve)
        out.projection_sources.append(source)
        status = (f"{plate.plate_id}: {len(frames)} frame(s) @ {fps:g} fps"
                  if frames else
                  f"{plate.plate_id}: still crop only (no generated frames "
                  f"yet — run the generate stage)")
        return (out, f"Dynamic plate layer added — {status}; projector fixed "
                     f"at the crop camera, priority {priority:g}.")
