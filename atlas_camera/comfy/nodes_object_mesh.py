"""Generated object meshes: give a foreground object its hidden sides.

Two nodes bracket a pixel-aligned image-to-3D model (core ComfyUI Pixal3D):

``AtlasObjectCrop`` 🎯
    Re-renders the photo through a virtual camera aimed at one masked object
    (:mod:`atlas_camera.core.object_crop`), so the model's centred-pinhole
    assumption holds exactly and its FOV comes from the SOLVE -- wire ``fov``
    into ``Pixal3DConditioning``'s fov, ``crop`` into its image.

``AtlasImportGeneratedMesh`` 🧩
    Maps the model's MESH back through that camera, measures its one unknown
    (scale along the rays) against the shared metric depth, scores the
    placement against the photograph, and APPENDS it as a PROXY_ROLE mesh.
    The photo paints what the solved camera saw; the model's vertex colour
    paints only the hidden side (``photo_weight`` per vertex).

The MESH input is duck-typed (``.vertices``, ``.faces``, optional
``.vertex_colors`` / ``.vertex_counts`` / ``.face_counts``) so this module
needs no ``comfy_api`` import.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from atlas_camera.comfy.node_helpers import (
    _metric_depth_and_validity,
    _require_numpy,
    _require_torch,
    _resolve_exclude_mask,
)

GENERATED_SOURCE = "pixal3d"
_BACKGROUNDS = ("black", "white", "gray", "photo")
_BG_RGB = {"black": (0.0, 0.0, 0.0), "white": (1.0, 1.0, 1.0), "gray": (0.5, 0.5, 0.5)}


def _solve_camera(solve):
    from atlas_camera.blender.measured import camera_params
    cp = camera_params(solve)
    if cp is None:
        raise ValueError("solve has no usable camera (focal / view matrix / image size)")
    return cp


def _image_at_solve(torch, image, width: int, height: int):
    """(1, H, W, C) float tensor at the solve's raster."""
    img = image[:1].float()
    if img.shape[1] != height or img.shape[2] != width:
        img = torch.nn.functional.interpolate(
            img.permute(0, 3, 1, 2), size=(height, width), mode="bilinear",
            align_corners=False).permute(0, 2, 3, 1)
    return img


class AtlasObjectCrop:
    """🎯 Square crop of ONE object, re-rendered for a centred-pinhole model.

    A plain rectangle cut from the photo keeps the solve's off-centre
    principal point; Pixal3D assumes a centred one, so an object near the
    frame edge would come back sheared. This node instead rotates a virtual
    camera (same centre as the solve) to look at the object and resamples the
    photo through the exact rotation homography. Outputs the square crop and
    its matte, the crop's horizontal FOV in degrees FROM THE SOLVE (wire into
    Pixal3DConditioning.fov -- do not use MoGeGeometryToFOV, which guesses a
    different camera), and the ATLAS_OBJECT_CROP handle
    AtlasImportGeneratedMesh needs to map the mesh back.
    """

    RETURN_TYPES = ("IMAGE", "MASK", "FLOAT", "ATLAS_OBJECT_CROP", "STRING")
    RETURN_NAMES = ("crop", "crop_mask", "fov", "object_crop", "report")
    FUNCTION = "crop"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "image": ("IMAGE", {"tooltip": "The source photo the solve was made from."}),
                "mask": ("MASK", {"tooltip": "The ONE object to lift (AtlasSAM3Mask -> "
                                             "AtlasInstanceMask). Visible pixels only."}),
            },
            "optional": {
                "pad": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 3.0, "step": 0.01,
                                  "tooltip": "Margin around the object. 1.1 matches the "
                                             "framing Pixal3D was published with."}),
                "size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64,
                                 "tooltip": "Crop resolution (Pixal3D expects 1024)."}),
                "roll": (["gravity", "source_camera"], {
                    "default": "gravity",
                    "tooltip": "gravity: world-up is up in the crop (the model's prior is "
                               "upright objects). source_camera: keep the photo's own roll."}),
                "background": (list(_BACKGROUNDS), {
                    "default": "black",
                    "tooltip": "What fills the crop outside the mask. black matches "
                               "ImageCropToMask; photo keeps the surrounding context."}),
                "grow_px": ("INT", {"default": 0, "min": 0, "max": 64,
                                    "tooltip": "Dilate the mask before cropping (source px)."}),
            },
        }

    def crop(self, solve, image, mask, pad=1.1, size=1024, roll="gravity",
             background="black", grow_px=0):
        torch = _require_torch()
        np = _require_numpy()
        from atlas_camera.core.mask_ops import dilate
        from atlas_camera.core.object_crop import crop_sample_grid, virtual_object_camera

        cp = _solve_camera(solve)
        w, h = int(cp["image_width"]), int(cp["image_height"])
        obj = _resolve_exclude_mask(mask, h, w)
        if obj is None or not bool(obj.any()):
            raise ValueError("AtlasObjectCrop: the object mask is empty")
        if int(grow_px) > 0:
            obj = dilate(obj, int(grow_px))
        cam = virtual_object_camera(
            view_matrix=cp["view_matrix"], fx=cp["fx"], fy=cp["fy"], cx=cp["cx"],
            cy=cp["cy"], image_width=w, image_height=h, mask=obj, pad=float(pad),
            size=int(size), roll=str(roll))

        sx, sy, valid = crop_sample_grid(cam)
        img = _image_at_solve(torch, image, w, h)
        # grid_sample with align_corners=True maps -1..1 onto pixel INDICES
        # 0..W-1 -- the index convention object_crop uses throughout.
        gx = torch.from_numpy((sx / max(w - 1, 1)) * 2.0 - 1.0).float()
        gy = torch.from_numpy((sy / max(h - 1, 1)) * 2.0 - 1.0).float()
        grid = torch.stack([gx, gy], dim=-1)[None].to(img.device)
        rgb = torch.nn.functional.grid_sample(
            img.permute(0, 3, 1, 2), grid, mode="bilinear", padding_mode="border",
            align_corners=True).permute(0, 2, 3, 1)[..., :3]
        m_src = torch.from_numpy(obj.astype(np.float32))[None, None].to(img.device)
        m = torch.nn.functional.grid_sample(
            m_src, grid, mode="bilinear", padding_mode="zeros",
            align_corners=True)[:, 0]
        vmask = torch.from_numpy(valid.astype(np.float32))[None].to(img.device)
        m = m * vmask
        if background != "photo":
            bg = torch.tensor(_BG_RGB.get(str(background), (0.0, 0.0, 0.0)),
                              dtype=rgb.dtype, device=rgb.device)
            rgb = rgb * m[..., None] + bg * (1.0 - m[..., None])
        else:
            rgb = rgb * vmask[..., None]

        handle = cam.to_dict()
        report = (f"AtlasObjectCrop: {int(obj.sum())} object px -> {cam.size}^2 crop, "
                  f"fov {cam.fov_deg:.3f} deg (from the solve, fx {cp['fx']:.1f} px), "
                  f"pad {cam.pad:.2f}, roll {cam.roll}"
                  + (" (gravity undefined here -- fell back to source up)"
                     if cam.roll_fallback else "")
                  + f"; {float((~valid).mean()) * 100:.1f}% of the crop lies outside the photo")
        return (rgb.cpu().float(), m.cpu().float(), float(cam.fov_deg), handle, report)


def _mesh_item(np, mesh):
    """Item 0 of a (possibly zero-padded) MESH batch as numpy arrays."""
    def arr(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            x = x.detach().cpu().float().numpy() if x.dtype.is_floating_point \
                else x.detach().cpu().numpy()
        return np.asarray(x)

    verts = arr(getattr(mesh, "vertices", None))
    faces = arr(getattr(mesh, "faces", None))
    if verts is None or faces is None:
        raise ValueError("MESH input has no vertices/faces")
    if verts.ndim == 3:
        verts = verts[0]
    if faces.ndim == 3:
        faces = faces[0]
    vc = arr(getattr(mesh, "vertex_counts", None))
    fc = arr(getattr(mesh, "face_counts", None))
    if vc is not None and np.ndim(vc) >= 1 and len(vc):
        verts = verts[: int(vc[0])]
    if fc is not None and np.ndim(fc) >= 1 and len(fc):
        faces = faces[: int(fc[0])]
    cols = arr(getattr(mesh, "vertex_colors", None))
    if cols is not None:
        if cols.ndim == 3:
            cols = cols[0]
        cols = cols[: len(verts), :3].astype(np.float64)
        if cols.max(initial=0.0) > 1.5:  # 0..255 storage
            cols = cols / 255.0
        # MESH.vertex_colors are LINEAR (the glTF convention): core PaintMesh
        # writes pow(srgb, 2.2). Everything downstream here is sRGB like the
        # plate, so undo that exact gamma. Found live 2026-10-02: read as sRGB,
        # the colours came in so dark the plate grade hit its 2.0 clamp.
        cols = np.power(np.clip(cols, 0.0, 1.0), 1.0 / 2.2)
    return verts.astype(np.float64), faces.astype(np.int64), cols


class AtlasImportGeneratedMesh:
    """🧩 Place a Pixal3D mesh in the solved scene -- the object's hidden sides.

    Maps the model's mesh back through the ATLAS_OBJECT_CROP camera (exact up
    to scale), measures the scale as ONE median ratio of the shared metric
    depth over the mesh's own z-buffer on the object's visible pixels, and
    APPENDS a PROXY_ROLE mesh (never clobbers). Painting: projective photo
    where the solved camera saw the surface, the model's vertex colour
    (graded onto the photo) everywhere else.

    Gates, in the scale-gate doctrine's terms: REFUSE (definitional) when the
    rendered depth ordering is no better than chance against the scene depth,
    when the mesh stands in observed sky, or when no scale could be measured;
    rel_mad thresholds and silhouette IoU are reported and grade INSPECT
    (uncalibrated / fixture-calibrated). Nothing here can verify the hidden
    side itself -- it is a hypothesis and the report says so.

    Wire ``object_alpha`` (OR'd with the sky mask) into the main relief's
    exclude_mask so the object is not built twice, and put an
    AtlasCleanPlateLayer behind it for the background it reveals.
    """

    RETURN_TYPES = ("ATLAS_SOLVE", "STRING", "MASK")
    RETURN_NAMES = ("solve", "report", "object_alpha")
    FUNCTION = "import_mesh"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "mesh": ("MESH", {"tooltip": "Pixal3D mesh: VaeDecodeShapeTrellis -> "
                                             "(Remesh/Decimate) -> PaintMesh for vertex colours."}),
                "object_crop": ("ATLAS_OBJECT_CROP",),
                "depth": ("ATLAS_DEPTH_MAP", {"tooltip": "The SHARED scene depth (same one "
                                                         "the relief is built from)."}),
                "image": ("IMAGE", {"tooltip": "Source photo (colour grading of the "
                                               "vertex colours onto the plate)."}),
                "object_mask": ("MASK", {"tooltip": "Same object mask AtlasObjectCrop got."}),
            },
            "optional": {
                "sky_mask": ("MASK", {"tooltip": "Observed sky. Enables the sky-violation "
                                                 "gate and replaces the sky heuristic."}),
                "rests_on_ground": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Cross-check the depth scale against the scale that puts "
                               "the object's lowest point on the ground (Y=0)."}),
                "max_faces": ("INT", {"default": 60000, "min": 1000, "max": 500000,
                                      "step": 1000,
                                      "tooltip": "Payload budget; decimate upstream first."}),
                "name": ("STRING", {"default": "object"}),
                "on_gate_fail": (["refuse", "inspect"], {
                    "default": "refuse",
                    "tooltip": "refuse: a REFUSE grade passes the solve through untouched. "
                               "inspect: append anyway, flagged in scene health."}),
                "match_colour": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Grade the vertex colours onto the photo (per-channel "
                               "median gain on seen vertices) so the seam does not show."}),
            },
        }

    def import_mesh(self, solve, mesh, object_crop, depth, image, object_mask,
                    sky_mask=None, rests_on_ground=False, max_faces=60000,
                    name="object", on_gate_fail="refuse", match_colour=True):
        torch = _require_torch()
        np = _require_numpy()
        from atlas_camera.blender.measured import meshes_to_primitives
        from atlas_camera.core.generated_mesh import (
            cluster_decimate,
            ground_contact_scale,
            match_vertex_colours,
            photo_visibility_weights,
            pixal_to_source_camera,
            register_object_scale,
            scale_verdict,
            source_camera_to_world,
        )
        from atlas_camera.core.object_crop import ObjectCropCamera
        from atlas_camera.core.plate_falsification import score_geometry_against_plate

        solve_out = copy.deepcopy(solve)
        name = str(name or "object").strip() or "object"
        lines: list[str] = []
        blank = None

        setup = _metric_depth_and_validity(solve_out, depth, exclude_mask=sky_mask)
        if setup is None:
            raise ValueError("AtlasImportGeneratedMesh: solve has no usable focal length")
        w, h = int(setup.width), int(setup.height)
        blank = torch.zeros(1, h, w, dtype=torch.float32)
        view = np.asarray(setup.extr.camera_view_matrix, dtype=np.float64)
        crop = ObjectCropCamera.from_dict(object_crop)
        if (crop.source_width, crop.source_height) != (w, h):
            lines.append(f"warning: crop was made at {crop.source_width}x{crop.source_height}, "
                         f"solve raster is {w}x{h}")

        verts, faces, cols = _mesh_item(np, mesh)
        n_in = len(faces)
        verts, faces, cols = cluster_decimate(verts, faces, max_faces=int(max_faces),
                                              colours=cols)
        if len(faces) < n_in:
            lines.append(f"decimated {n_in} -> {len(faces)} faces (max_faces {int(max_faces)}; "
                         "decimate upstream with DecimateMesh for better quality)")

        obj = _resolve_exclude_mask(object_mask, h, w)
        if obj is None or not bool(obj.any()):
            return (solve_out, "REFUSED — object_mask is empty; solve passed through", blank)
        sky = _resolve_exclude_mask(sky_mask, h, w) if sky_mask is not None else None

        cam_pts = pixal_to_source_camera(verts, rotation=crop.rotation, fov_deg=crop.fov_deg)
        reg = register_object_scale(
            cam_pts, faces, view_matrix=view, fx=setup.fx, fy=setup.fy, cx=setup.cx,
            cy=setup.cy, width=w, height=h, metric_depth=setup.metric,
            object_mask=obj, depth_valid=setup.valid)
        s = reg.get("scale")
        s_ground = ground_contact_scale(cam_pts, view_matrix=view) if rests_on_ground else None
        verdict = scale_verdict(rel_mad=reg["rel_mad"], depth_scale=s, ground_scale=s_ground)
        issues = list(verdict["issues"])
        if reg.get("reason"):
            issues.append(reg["reason"])

        alpha = reg["unit_alpha"]
        score: dict[str, Any] = {}
        if bool(alpha.any()) and s is not None:
            score = score_geometry_against_plate(
                alpha=alpha, render_depth=reg["unit_depth"] * s, sky_mask=sky,
                observed_mask=obj,
                reference_depth=np.where(setup.valid, setup.metric, np.nan))
            for key in ("depth_order_agreement", "sky_violation"):
                m = score.get(key) or {}
                if m.get("available") and m.get("pass") is False:
                    verdict["grade"] = "refuse"
                    issues.append(f"{key} {m['value']:.3f} fails its definitional gate "
                                  f"({m['threshold']})")
            iou = score.get("silhouette_iou") or {}
            if iou.get("available") and iou.get("pass") is False:
                issues.append(f"silhouette IoU {iou['value']:.3f} < {iou['threshold']} "
                              "(inspect; calibrated on another fixture)")
                if verdict["grade"] == "ok":
                    verdict["grade"] = "inspect"
        elif s is not None:
            verdict["grade"] = "refuse"
            issues.append("the mesh covers no pixel of the solve camera")
        grade = verdict["grade"]

        head = (f"AtlasImportGeneratedMesh '{name}': {len(verts)} verts / {len(faces)} faces; "
                f"scale {s if s is None else round(s, 4)} "
                f"(rel_mad {reg['rel_mad']:.4f} over {reg['registration_px']} px, "
                "thresholds uncalibrated)")
        if s_ground is not None:
            head += f"; ground-contact scale {s_ground:.4f}"
        report_tail = [f"grade: {grade.upper()}"] + [f"- {i}" for i in issues] + [
            "hidden side: a model hypothesis — no reprojection test can verify it."]
        if score:
            report_tail.append("scores: " + json.dumps({
                k: (round(v["value"], 4) if isinstance(v, dict) and v.get("available") else None)
                for k, v in score.items() if isinstance(v, dict) and "available" in v}))

        if grade == "refuse" and on_gate_fail == "refuse":
            return (solve_out, "\n".join([head, "REFUSED — solve passed through", *lines,
                                          *report_tail]), blank)

        world = source_camera_to_world(cam_pts, view_matrix=view, scale=float(s))
        weight, wstats = photo_visibility_weights(
            world, faces, view_matrix=view, fx=setup.fx, fy=setup.fy, cx=setup.cx,
            cy=setup.cy, width=w, height=h, object_mask=obj, metric_depth=setup.metric,
            mesh_depth=reg["unit_depth"] * float(s))

        colour_note = "no vertex colours on the MESH — hidden side painted neutral grey " \
                      "(wire PaintMesh upstream)"
        if cols is None:
            cols = np.full((len(world), 3), 0.5)
        elif match_colour:
            img = _image_at_solve(torch, image, w, h)[0, ..., :3].cpu().numpy()
            vmc = world @ view[:3, :3].T + view[:3, 3]
            fwd = np.maximum(-vmc[:, 2], 1e-9)
            ix = np.clip(np.rint(setup.cx + setup.fx * vmc[:, 0] / fwd), 0, w - 1).astype(int)
            iy = np.clip(np.rint(setup.cy - setup.fy * vmc[:, 1] / fwd), 0, h - 1).astype(int)
            cols, crep = match_vertex_colours(cols, weight, img[iy, ix])
            colour_note = (f"vertex colours graded onto the plate: gain "
                           f"{[round(g, 3) for g in crep['gain']]} over "
                           f"{crep['seen_vertices']} seen vertices"
                           + (" (CLAMPED)" if any(crep["clamped"]) else "")
                           + ("" if crep["applied"] else " — too few seen vertices, not applied"))
        else:
            colour_note = "vertex colours used as generated (match_colour off)"

        accepted, rejected = meshes_to_primitives(
            solve_out, [{"name": name, "vertices": world, "faces": faces}],
            source=GENERATED_SOURCE, name_prefix=GENERATED_SOURCE, min_y_m=-1e3)
        if not accepted:
            why = rejected[0]["reason"] if rejected else "unknown"
            return (solve_out, "\n".join([head, f"REFUSED — mesh rejected: {why}", *lines,
                                          *report_tail]), blank)
        prim = accepted[0]
        meta = prim.metadata
        meta["vertex_colors"] = np.round(np.clip(cols, 0.0, 1.0).reshape(-1), 3).tolist()
        meta["photo_weight"] = np.round(weight, 3).tolist()
        meta["generated_grade"] = grade
        meta["generated_issues"] = "; ".join(issues)
        meta["generated_scale"] = float(s)
        meta["generated_rel_mad"] = float(reg["rel_mad"])
        meta["generated_fov_deg"] = float(crop.fov_deg)
        meta["photo_fraction"] = float(wstats["photo_fraction"])
        solve_out.projection_scene.proxy_geometry.append(prim)
        dbg = solve_out.projection_scene.debug_metadata
        dbg.setdefault("generated_objects", []).append({
            "name": prim.name, "grade": grade, "scale": float(s),
            "rel_mad": float(reg["rel_mad"]), "issues": issues,
            "photo_visibility": wstats,
        })

        coverage = torch.from_numpy(alpha.astype(np.float32))[None]
        body = [head, f"APPENDED as PROXY_ROLE mesh '{prim.name}' "
                      f"(photo paints {wstats['photo_fraction'] * 100:.0f}% of vertices)",
                colour_note, *lines, *report_tail]
        return (solve_out, "\n".join(body), coverage)


def _resolve_output_path(path: str) -> str:
    """Absolute, or relative to ComfyUI's output directory (where the HDR
    workflow writes), or relative to the current directory."""
    from pathlib import Path
    p = Path(str(path or "").strip().strip('"'))
    if not str(p):
        return ""
    if p.is_absolute() and p.is_file():
        return str(p)
    try:
        import folder_paths  # type: ignore[import-not-found]
        cand = Path(folder_paths.get_output_directory()) / p
        if cand.is_file():
            return str(cand)
    except Exception:  # noqa: BLE001 - not inside ComfyUI
        pass
    return str(p) if p.is_file() else ""


class AtlasHDRVertexTransfer:
    """🌗 Give generated objects' hidden sides HDR colour, from the plate's own conversion.

    Run SDR->HDR on the plate in its own workflow (matrixZone still), then wire
    that EXR's path here. The model never saw a generated object's hidden side
    (it is vertex colour, not an image), so the plate's SDR/HDR pixel pairs are
    used to measure the model's own per-channel tone curve, which is applied to
    every generated object's vertex colours. Stores ACEScg linear
    ``vertex_colors_hdr`` on each primitive (AtlasSceneTo3D writes them as a
    float PLY sidecar). The SDR ``vertex_colors`` are left as they are.
    """

    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "transfer"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "source_image": ("IMAGE", {"tooltip": "The SDR plate the HDR was converted from."}),
            },
            "optional": {
                "hdr_exr_path": ("STRING", {
                    "default": "",
                    "tooltip": "The HDR plate EXR from the matrixZone still workflow "
                               "(absolute, or relative to ComfyUI's output folder, e.g. "
                               "atlas/hdr_plate_00001.exr). Empty = pass through."}),
                "hdr_image": ("IMAGE", {"tooltip": "Alternative to the path: a float HDR "
                                                   "IMAGE (ACEScg linear)."}),
            },
        }

    def transfer(self, solve, source_image, hdr_exr_path="", hdr_image=None):
        np = _require_numpy()
        from atlas_camera.core.hdr_transfer import apply_curve, fit_sdr_to_hdr_curve
        from atlas_camera.core.matrixzone import resize_bilinear

        solve_out = copy.deepcopy(solve)
        gens = [p for p in solve_out.projection_scene.proxy_geometry
                if (p.metadata or {}).get("source") == GENERATED_SOURCE
                and (p.metadata or {}).get("vertex_colors")]
        if not gens:
            return (solve_out, "AtlasHDRVertexTransfer: no generated object with vertex "
                               "colours in the solve - passed through")
        hdr, origin = None, ""
        if hdr_image is not None:
            hdr = hdr_image[0].detach().cpu().float().numpy()[..., :3]
            origin = "hdr_image input"
        else:
            path = _resolve_output_path(hdr_exr_path)
            if not path:
                return (solve_out, "AtlasHDRVertexTransfer: no HDR plate (set hdr_exr_path to "
                                   "the matrixZone still workflow's EXR) - passed through")
            from atlas_camera.plate.oiio_io import read_plate
            hdr = np.asarray(read_plate(path, output_colorspace=None).pixels,
                             dtype=np.float32)[..., :3]
            origin = path
        sdr = source_image[0].detach().cpu().float().numpy()[..., :3]
        if sdr.shape[:2] != hdr.shape[:2]:
            sdr = resize_bilinear(sdr, hdr.shape[0], hdr.shape[1])
        curve = fit_sdr_to_hdr_curve(sdr, hdr)
        lines = [f"AtlasHDRVertexTransfer: tone curve fitted from {origin} "
                 f"({curve['samples']} px, {curve['bins']} bins, ACEScg); one global curve "
                 f"explains the plate's conversion to {curve['residual_stops']:.3f} stops median"]
        for p in gens:
            vc = np.asarray(p.metadata["vertex_colors"], dtype=np.float32).reshape(-1, 3)
            hdr_vc = apply_curve(curve, vc)
            p.metadata["vertex_colors_hdr"] = np.round(hdr_vc.reshape(-1), 4).tolist()
            p.metadata["vertex_colors_hdr_space"] = "ACEScg"
            p.metadata["vertex_colors_hdr_residual_stops"] = round(curve["residual_stops"], 4)
            lines.append(f"- {p.name}: {len(vc)} vertex colours -> ACEScg linear, max "
                         f"{float(hdr_vc.max()):.2f}, {int((hdr_vc.max(-1) > 1).sum())} above 1.0")
        lines.append("the hidden side's HDR is the plate's conversion curve applied to a "
                     "model-guessed colour: a reconstruction of a reconstruction")
        return (solve_out, "\n".join(lines))
