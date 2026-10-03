"""AtlasSceneTo3D: the layered Atlas scene as ComfyUI's native 3D sockets.

Core ComfyUI's Load3D / Preview 3D / Save 3D (Advanced) take ``model_3d``
(FILE_3D_GLB, a ``File3D``), ``model_3d_info`` (LOAD3D_MODEL_INFO) and
``camera_info`` (LOAD3D_CAMERA). This node fills all three from a solve:

* ``model_3d`` -- one GLB holding every projection layer as its own textured
  mesh (``exporters.scene_glb``), full-resolution PNG plates, with one float
  EXR sidecar per plate named in each material's ``extras`` (glTF has no EXR
  image format, so the floats cannot live inside the GLB);
* ``model_3d_info`` -- identity: Atlas geometry is already world-space;
* ``camera_info`` -- the recovered solve camera (``core.load3d_camera``),
  same Y-up / -Z convention as Load3D, so nothing is converted.

``File3D`` and the LOAD3D_* socket types are core ComfyUI (V135+). Without
them the node still writes the GLB and sidecars and returns a duck-typed file
object, so ``glb_path`` and ``report`` stay usable on an older install.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atlas_camera.comfy.node_helpers import _image_tensor_to_pil, _require_numpy, output_paths


class _FileRef:
    """Stand-in for comfy_api File3D when core ComfyUI predates it."""

    def __init__(self, source: str, file_format: str = "glb"):
        self._source, self.format = source, file_format

    def get_source(self) -> str:
        return self._source

    def save_to(self, path: str) -> str:
        import shutil
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(self._source).resolve() != dest.resolve():
            shutil.copy2(self._source, dest)
        return str(dest)

    def __repr__(self) -> str:
        return f"File3D(source={self._source!r}, format={self.format!r})"


def _file3d(path: str) -> tuple[Any, str]:
    """``(file, note)``: core ComfyUI's File3D, or the duck-typed stand-in with
    a note saying so (the fallback is never silent: an install that moved
    File3D would otherwise hand Load3D an object it may not accept)."""
    try:
        from comfy_api.latest._util.geometry_types import File3D  # type: ignore[import-not-found]
        return File3D(path, "glb"), ""
    except Exception as exc:  # noqa: BLE001 - older ComfyUI / tests
        return _FileRef(path, "glb"), (
            f"note: core File3D unavailable ({type(exc).__name__}: {exc}) - model_3d is a "
            "duck-typed file reference; glb_path is the GLB itself")


#: Default GLB size budget and the size above which the report warns (MB).
GLB_BUDGET_MB = 1024
GLB_WARN_MB = 200
#: A GLB stores its length as uint32: the budget widget cannot exceed it.
GLB_MAX_BUDGET_MB = 4095


class AtlasSceneTo3D:
    """🧊 Layered Atlas scene -> model_3d + model_3d_info + camera_info.

    Wire the three outputs into Save 3D (Advanced) / Preview 3D (Advanced).
    The GLB carries every layer (relief, generated objects, clean-plate bands,
    sky domes) with its own full-resolution plate; each plate is also written
    as a float EXR beside it (scene-linear when the layer has a float
    plate_ref, otherwise a linearised display plate, and the sidecar says
    which). The camera is the recovered solve camera.
    """

    RETURN_TYPES = ("FILE_3D_GLB", "LOAD3D_MODEL_INFO", "LOAD3D_CAMERA", "STRING", "STRING")
    RETURN_NAMES = ("model_3d", "model_3d_info", "camera_info", "glb_path", "report")
    FUNCTION = "export"
    CATEGORY = "Atlas Camera"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "source_image": ("IMAGE", {"tooltip": "Full-res primary plate (paints the "
                                                      "primary relief and generated objects)."}),
            },
            "optional": {
                "write_exr": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Write one float EXR per layer plate beside the GLB "
                               "(named in each material's extras). GLB itself cannot "
                               "hold EXR textures."}),
                "filename_prefix": ("STRING", {"default": "atlas/scene"}),
                # APPENDED: size budget. Full-res plates x layers grow fast (8K,
                # 6 layers = ~235 MB); past this the GLB is refused, not handed
                # to a browser viewer that cannot load it.
                "max_glb_mb": ("INT", {
                    "default": GLB_BUDGET_MB, "min": 0, "max": GLB_MAX_BUDGET_MB, "step": 64,
                    "tooltip": "Refuse a GLB larger than this (MB), naming its size, "
                               "BEFORE it is built. 0 = no budget (the GLB format's "
                               "4 GiB limit still applies)."}),
            },
        }

    def export(self, solve, source_image, write_exr=True, filename_prefix="atlas/scene",
               max_glb_mb=GLB_BUDGET_MB):
        np = _require_numpy()
        from atlas_camera.exporters.scene_glb import build_scene_layers

        intr, extr = _usable_camera(solve)
        folder, stem = output_paths(filename_prefix)
        folder.mkdir(parents=True, exist_ok=True)
        primary = _image_tensor_to_pil(source_image)

        layers, sidecars, notes = build_scene_layers(
            solve, primary, exr_dir=folder if write_exr else None, exr_prefix=stem)
        glb_path = folder / f"{stem}.glb"
        written, mb = _write_glb_within_budget(layers, glb_path, max_glb_mb, folder, sidecars)

        camera_info, model_info = _load3d_sockets(np, layers, intr, extr)
        manifest_note = _scene_manifest(solve, folder, glb_path, sidecars, written)
        model_3d, file_note = _file3d(str(glb_path))
        if file_note:
            notes = [*notes, file_note]

        report = _export_report(glb_path, mb, written, sidecars, notes, write_exr,
                                camera_info, intr, manifest_note)
        return {"ui": {"text": [report]},
                "result": (model_3d, model_info, camera_info, str(glb_path), report)}


def _usable_camera(solve):
    """Gate: ``(intrinsics, extrinsics)``, refused without a view matrix and focal."""
    cam = solve.camera
    intr, extr = cam.intrinsics, cam.extrinsics
    if extr is None or extr.camera_view_matrix is None or not intr.fy_px and not intr.fx_px:
        raise ValueError("AtlasSceneTo3D: the solve has no usable camera")
    return intr, extr


def _write_glb_within_budget(layers, glb_path, max_glb_mb, folder=None, sidecars=()):
    """Write the GLB; refuse past the size budget BEFORE building it.

    The size is planned from the layers' buffers (``scene_glb.plan_scene_glb``)
    so an over-budget scene allocates nothing. On refusal the sidecars this
    export already wrote are deleted (a refused export leaves no orphans) and
    the error names them and breaks the size down per layer / per plate.
    Returns ``(written, mb)``.
    """
    from atlas_camera.exporters import scene_glb

    budget = min(int(max_glb_mb or 0), GLB_MAX_BUDGET_MB)
    try:
        written = scene_glb.write_scene_glb(
            layers, glb_path, max_bytes=budget * 1_000_000 if budget > 0 else None)
    except scene_glb.GLBBudgetError as exc:
        glb_path.unlink(missing_ok=True)
        removed = _remove_sidecars(folder, sidecars)
        plan = exc.plan
        raise ValueError(
            f"AtlasSceneTo3D: the GLB would be {plan['bytes'] / 1e6:.0f} MB, over the "
            f"{budget} MB budget ({len(plan['layers'])} layers at full plate resolution, "
            f"{len(plan['images'])} embedded plate(s)) - feed a smaller plate or fewer "
            "layers, or raise max_glb_mb (0 = no budget). Per layer: "
            f"{scene_glb.describe_glb_plan(plan)}."
            + (f" Removed the sidecars already written: {', '.join(removed)}." if removed
               else "")) from None
    mb = written["bytes"] / 1e6
    return written, mb


def _remove_sidecars(folder, sidecars) -> list[str]:
    """Delete the EXR/PLY sidecars an export wrote; returns their names."""
    removed = []
    if folder is None:
        return removed
    for s in sidecars or ():
        name = s.get("exr")
        if not name:
            continue
        path = Path(folder) / name
        try:
            if path.is_file():
                path.unlink()
                removed.append(name)
        except OSError:
            continue
    return removed


def _load3d_sockets(np, layers, intr, extr):
    """Compute: ``(camera_info, model_info)`` for Load3D, far plane past every layer."""
    from atlas_camera.core.camera_math import ground_lookat_pivot
    from atlas_camera.core.load3d_camera import identity_model_info, load3d_camera_info

    view = np.asarray(extr.camera_view_matrix, dtype=np.float64)
    c2w = np.linalg.inv(view)
    far = 10.0
    for layer in layers:
        v = np.asarray(layer.vertices, dtype=np.float64).reshape(-1, 3)
        if len(v):
            far = max(far, float(np.linalg.norm(v - c2w[:3, 3], axis=1).max()))
    fy = float(intr.fy_px or intr.fx_px)
    camera_info = load3d_camera_info(
        view_matrix=view, fy=fy, image_width=int(intr.image_width),
        image_height=int(intr.image_height),
        target=list(ground_lookat_pivot(extr)), near=0.05, far=far * 1.5)
    return camera_info, identity_model_info(1)


def _scene_manifest(solve, folder, glb_path, sidecars, written) -> str:
    """Export manifest note; a manifest never fails an export."""
    manifest_note = ""
    try:
        from atlas_camera.comfy.node_reports import _write_export_manifest
        manifest_note = _write_export_manifest(
            solve, folder,
            [("scene_glb", str(glb_path))]
            + [("plate_exr", str(folder / s["exr"])) for s in sidecars],
            "AtlasSceneTo3D", extra={"scene_glb_layers": written["layers"]})
    except Exception as exc:  # noqa: BLE001 - a manifest never fails an export
        manifest_note = f"manifest skipped: {exc}"
    return manifest_note


def _export_report(glb_path, mb, written, sidecars, notes, write_exr, camera_info, intr,
                   manifest_note) -> str:
    """The scene export node's multi-line report."""
    lines = [f"AtlasSceneTo3D: {len(written['layers'])} layer mesh(es) -> {glb_path} "
             f"({mb:.1f} MB, {written.get('images', 0)} embedded plate image(s), one per "
             "source)"]
    for s in written["layers"]:
        lines.append(f"- {s['name']}: {s['vertices']} verts / {s['faces']} faces"
                     + (", textured" if s["textured"] else ", UNTEXTURED")
                     + (f" ({s['untextured_reason']})" if s.get("untextured_reason") else "")
                     + (", vertex-colour hidden side" if s["vertex_colour"] else ""))
    for d in written.get("dropped") or []:
        lines.append(f"- DROPPED {d['name']}: {d['reason']}")
    for s in sidecars:
        if s["exr"].endswith(".ply"):
            lines.append(f"- PLY {s['exr']}: HDR vertex colour, {s['exr_colorspace']}")
            continue
        lines.append(f"- EXR {s['exr']}: {s['exr_colorspace']}"
                     + ("" if s["scene_referred"] else
                        " (linearised display plate, NOT scene-referred)"))
    if not write_exr:
        lines.append("- EXR sidecars off")
    lines += [f"- {n}" for n in notes]
    lines.append(f"camera: fov {camera_info['fov']:.2f} deg vertical, aspect "
                 f"{camera_info['aspect']:.3f}, target = ground pivot")
    cx, cy = float(intr.cx_px or intr.image_width / 2), float(intr.cy_px or intr.image_height / 2)
    off = max(abs(cx - intr.image_width / 2), abs(cy - intr.image_height / 2))
    if off > 1.0:
        lines.append(f"note: principal point is {off:.1f} px off-centre; a Load3D "
                     "perspective camera is centred, so its view is approximate")
    if manifest_note:
        lines.append(manifest_note)
    if mb > GLB_WARN_MB:
        lines.append("warning: large GLB - the browser 3D viewer may be slow to load it")
    return "\n".join(lines)


def camera_info_json(camera_info: dict) -> str:
    """Pretty JSON of a camera_info payload (reports / tests)."""
    return json.dumps(camera_info, indent=1)
