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

from atlas_camera.comfy.node_helpers import (
    _image_tensor_to_pil,
    _require_numpy,
    output_paths,
    output_root,
    project_output_paths,
)


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
#: The budget widget's max. It stays 16384 (its original range) so saved
#: graphs above the format limit keep validating; the runtime clamps instead.
GLB_BUDGET_WIDGET_MAX_MB = 16384
#: A GLB stores its length as uint32: budgets above this are clamped (MB).
GLB_FORMAT_MAX_MB = 0xFFFFFFFF // 1_000_000


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
                    "default": GLB_BUDGET_MB, "min": 0, "max": GLB_BUDGET_WIDGET_MAX_MB,
                    "step": 64,
                    "tooltip": "Refuse a GLB larger than this (MB), naming its size, "
                               "BEFORE any file is written. 0 = no budget. Values past "
                               "the GLB format's 4 GiB limit are clamped to it."}),
                # APPENDED: delivery project. Save 3D copies only the GLB, so its
                # EXR/PLY sidecars are left behind; with a project the GLB, every
                # sidecar and the manifest land TOGETHER in the shot's geo lane.
                "project": ("ATLAS_PROJECT", {
                    "tooltip": "Optional delivery project from AtlasProject: writes the "
                               "GLB + EXR/PLY sidecars + manifest together into the "
                               "shot's geo/ lane (supersedes filename_prefix's folder)."}),
            },
        }

    def export(self, solve, source_image, write_exr=True, filename_prefix="atlas/scene",
               max_glb_mb=GLB_BUDGET_MB, project=None):
        np = _require_numpy()
        from atlas_camera.exporters.scene_glb import collect_scene_layers

        intr, extr = _usable_camera(solve)
        if project is not None:
            folder, stem = project_output_paths(project, "geo", filename_prefix)
        else:
            folder, stem = output_paths(filename_prefix)
        folder.mkdir(parents=True, exist_ok=True)
        primary = _image_tensor_to_pil(source_image)

        # Plan first, write second: the layers are built and every sidecar
        # NAMED, the GLB is sized (exact JSON + headroom for the sidecar
        # references), and an over-budget scene is refused before ANY file
        # (EXR, PLY or GLB) exists. Only then are the sidecars written.
        scene = collect_scene_layers(
            solve, primary, exr_dir=folder if write_exr else None, exr_prefix=stem,
            output_root=output_root())
        budget, clamp_note = _effective_budget_mb(max_glb_mb)
        _refuse_over_budget_before_writing(scene, budget)
        sidecars = scene.write_sidecars()
        layers, notes = scene.layers, list(scene.notes)
        if clamp_note:
            notes.append(clamp_note)
        if write_exr and sidecars:
            notes = [*notes, _sidecar_location_note(folder, project)]
        glb_path = folder / f"{stem}.glb"
        written, mb = _write_glb_within_budget(layers, glb_path, budget, folder, sidecars)

        camera_info, model_info = _load3d_sockets(np, layers, intr, extr)
        manifest_note = _scene_manifest(solve, folder, glb_path, sidecars, written)
        model_3d, file_note = _file3d(str(glb_path))
        if file_note:
            notes = [*notes, file_note]

        report = _export_report(glb_path, mb, written, sidecars, notes, write_exr,
                                camera_info, intr, manifest_note)
        return {"ui": {"text": [report]},
                "result": (model_3d, model_info, camera_info, str(glb_path), report)}


def _sidecar_location_note(folder, project) -> str:
    """Where the sidecars live and what travels with a copied GLB."""
    if project is not None:
        return (f"sidecars: in the project lane {folder} beside the GLB -- "
                "deliver the folder, not the GLB alone (a lane inside ComfyUI's output "
                "folder also records exr_output_path, so a Save 3D copy finds them)")
    return (f"sidecars: in {folder}; each reference also records its path relative "
            "to ComfyUI's output folder (exr_output_path), so a Save 3D copy in "
            "output/3d still finds them. Outside ComfyUI, copy them with the GLB "
            "(or connect an AtlasProject to deliver them together)")


def _usable_camera(solve):
    """Gate: ``(intrinsics, extrinsics)``, refused without a view matrix and focal."""
    cam = solve.camera
    intr, extr = cam.intrinsics, cam.extrinsics
    if extr is None or extr.camera_view_matrix is None or not intr.fy_px and not intr.fx_px:
        raise ValueError("AtlasSceneTo3D: the solve has no usable camera")
    return intr, extr


def _effective_budget_mb(max_glb_mb) -> tuple[int, str]:
    """``(budget_mb, note)``: the widget value clamped to the GLB format's
    4 GiB ceiling, with a report note when it was clamped (0 = no budget)."""
    requested = int(max_glb_mb or 0)
    if requested > GLB_FORMAT_MAX_MB:
        return GLB_FORMAT_MAX_MB, (
            f"note: max_glb_mb {requested} is past the GLB format's 4 GiB length limit - "
            f"clamped to {GLB_FORMAT_MAX_MB} MB")
    return max(requested, 0), ""


def _budget_bytes(budget_mb: int):
    return budget_mb * 1_000_000 if budget_mb > 0 else None


def _budget_refusal(plan, budget_mb, removed=()) -> ValueError:
    """The node's refusal: size, limit, per-layer / per-plate MB breakdown."""
    from atlas_camera.exporters import scene_glb

    limit = (f"the {budget_mb} MB budget" if budget_mb > 0
             else "the GLB format's 4 GiB length limit")
    return ValueError(
        f"AtlasSceneTo3D: the GLB would be {scene_glb.format_mb(plan['bytes'])} MB, "
        f"over {limit} "
        f"({len(plan['layers'])} layers at full plate resolution, "
        f"{len(plan['images'])} embedded plate(s)) - feed a smaller plate or fewer "
        "layers, or raise max_glb_mb (0 = no budget). Per layer: "
        f"{scene_glb.describe_glb_plan(plan)}."
        + (f" Removed the sidecars already written: {', '.join(removed)}." if removed
           else ""))


def _refuse_over_budget_before_writing(scene, budget_mb: int) -> None:
    """Size the GLB from the collected layers and refuse BEFORE any file is
    written. The plan's JSON is exact; ``extras_headroom`` bounds what the
    sidecar references (not written yet) will add to it."""
    from atlas_camera.exporters import scene_glb

    plan = scene_glb.plan_scene_glb(scene.layers)
    if not plan["prepared"]:
        return  # write_scene_glb raises its own "no layer had geometry" error
    try:
        scene_glb.check_glb_budget(plan, _budget_bytes(budget_mb),
                                   headroom=scene.extras_headroom())
    except scene_glb.GLBBudgetError as exc:
        raise _budget_refusal(exc.plan, budget_mb) from None


def _write_glb_within_budget(layers, glb_path, budget_mb, folder=None, sidecars=()):
    """Write the GLB, enforcing ``budget_mb`` (already clamped) again.

    ``write_scene_glb`` checks the exact plan before writing and the FINAL
    file size after. The node already refused over-budget scenes before any
    sidecar existed, so this only fires if the sidecar references outgrew
    their headroom; then the sidecars are deleted too (a refused export leaves
    no orphans) and named. Returns ``(written, mb)``.
    """
    from atlas_camera.exporters import scene_glb

    try:
        written = scene_glb.write_scene_glb(layers, glb_path,
                                            max_bytes=_budget_bytes(budget_mb))
    except scene_glb.GLBBudgetError as exc:
        glb_path.unlink(missing_ok=True)
        removed = _remove_sidecars(folder, sidecars)
        raise _budget_refusal(exc.plan, budget_mb, removed) from None
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
