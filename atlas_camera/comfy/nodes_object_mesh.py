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
from atlas_camera.core.generated_mesh import GENERATED_SOURCE

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


def _mesh_item(np, mesh, validate=False):
    """Item 0 of a (possibly zero-padded) MESH batch as numpy arrays.

    ``validate``: keep the face dtype (so non-integer indices can be refused
    by the caller) instead of casting to int64 here.
    """
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
    return (verts.astype(np.float64), faces if validate else faces.astype(np.int64),
            cols)


def _mesh_texture(np, mesh, n_verts):
    """The MESH's own UV layout + baked base-colour texture, or ``(None, None, why)``.

    Core ApplyTextureToMesh sets ``mesh.uvs`` (N, 2), top-left / glTF
    convention, and ``mesh.texture`` (H, W, 3) display-sRGB 0..1 (the voxel
    colours BakeTextureFromVoxel sampled -- glTF baseColor is sRGB). ``why`` is
    "" when there is simply no texture, else the reason one was rejected.
    """
    def arr(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            x = x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=np.float32)

    uvs, tex = arr(getattr(mesh, "uvs", None)), arr(getattr(mesh, "texture", None))
    if uvs is None or tex is None:
        return None, None, ""
    if uvs.ndim == 3:
        uvs = uvs[0]
    if tex.ndim == 4:
        tex = tex[0]
    uvs = uvs[:n_verts]
    if uvs.ndim != 2 or uvs.shape != (n_verts, 2) or not np.isfinite(uvs).all():
        return None, None, f"UVs {tuple(uvs.shape)} do not match {n_verts} vertices"
    if tex.ndim != 3 or tex.shape[-1] < 3 or min(tex.shape[:2]) < 2:
        return None, None, f"texture shape {tuple(tex.shape)} is not (H, W, 3)"
    if tex.max(initial=0.0) > 1.5:      # 0..255 storage
        tex = tex / 255.0
    return uvs.astype(np.float64), np.clip(tex[..., :3], 0.0, 1.0), ""


def _texture_data_uri(np, tex_srgb) -> str:
    """(H, W, 3) sRGB 0..1 -> ``data:image/png;base64,...`` (8-bit)."""
    import base64
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.round(np.clip(tex_srgb, 0.0, 1.0) * 255.0).astype(np.uint8)).save(
        buf, format="PNG", optimize=False, compress_level=6)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


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
                                             "(Remesh/Decimate) -> PaintMesh for vertex colours, "
                                             "OR -> UnwrapMesh -> BakeTextureFromVoxel -> "
                                             "ApplyTextureToMesh for a UV texture (hidden side "
                                             "painted per texel, not per vertex; not decimated "
                                             "here)."}),
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

        solve_out = copy.deepcopy(solve)
        name = str(name or "object").strip() or "object"
        lines: list[str] = []

        setup, crop, view, blank = _import_setup(torch, np, solve_out, depth, sky_mask,
                                                 object_crop, lines)
        w, h = int(setup.width), int(setup.height)
        # Validate the MESH before any decimation / rasterisation: a malformed
        # mesh is a REFUSED report, never an IndexError from deep inside.
        arrays, why = _import_mesh_arrays(np, mesh, max_faces, lines)
        if arrays is None:
            return (solve_out, "\n".join([
                f"AtlasImportGeneratedMesh '{name}': REFUSED — malformed MESH: {why}; "
                "solve passed through", *lines]), blank)
        verts, faces, cols, texture = arrays

        obj = _resolve_exclude_mask(object_mask, h, w)
        if obj is None or not bool(obj.any()):
            return (solve_out, "REFUSED — object_mask is empty; solve passed through", blank)
        sky = _resolve_exclude_mask(sky_mask, h, w) if sky_mask is not None else None

        cam_pts, reg, s, s_ground = _register_scale(
            verts, faces, crop, view, setup, obj, rests_on_ground)
        alpha = reg["unit_alpha"]
        score = _score_placement(np, reg, s, sky, obj, setup)
        grade, issues = _placement_grade(reg, s, s_ground, score)

        head, report_tail = _import_report_parts(name, verts, faces, s, s_ground, reg,
                                                 grade, issues, score)
        if s is None:
            # No measurable scale means no world placement exists to append:
            # refused whatever on_gate_fail says (inspect cannot place it).
            return (solve_out, "\n".join([
                head, "REFUSED — scale unmeasurable (no usable scale registration: "
                      f"{reg.get('reason') or 'unknown'}); refused regardless of "
                      "on_gate_fail; solve passed through", *lines, *report_tail]), blank)
        if grade == "refuse" and on_gate_fail == "refuse":
            return (solve_out, "\n".join([head, "REFUSED — solve passed through", *lines,
                                          *report_tail]), blank)

        world, weight, wstats = _photo_weights(np, cam_pts, faces, view, s, setup, obj, reg)
        cols, colour_note, gains = _grade_vertex_colours(torch, np, cols, weight, world, view,
                                                         setup, image, match_colour, faces=faces)
        if texture is not None and gains is not None:
            # The SAME linear gain the vertex colours got, so the texture and
            # the per-vertex fallback agree: per vertex -> interpolated per texel.
            texture = (texture[0], _grade_texture(np, texture[0], texture[1], faces, gains))

        prim, why = _append_generated_prim(np, solve_out, name, world, faces, cols, weight,
                                           grade=grade, issues=issues, s=s, reg=reg,
                                           crop=crop, wstats=wstats, texture=texture)
        if prim is None:
            return (solve_out, "\n".join([head, f"REFUSED — mesh rejected: {why}", *lines,
                                          *report_tail]), blank)

        coverage = torch.from_numpy(alpha.astype(np.float32))[None]
        body = [head, f"APPENDED as PROXY_ROLE mesh '{prim.name}' "
                      f"(photo paints {wstats['photo_fraction'] * 100:.0f}% of vertices)",
                colour_note, *lines, *report_tail]
        return (solve_out, "\n".join(body), coverage)


def _import_setup(torch, np, solve_out, depth, sky_mask, object_crop, lines):
    """Gate: metric depth on the solve raster + the crop camera.

    Returns ``(setup, crop, view, blank)``; a crop made at another raster is
    a warning appended to ``lines``.
    """
    from atlas_camera.core.object_crop import ObjectCropCamera

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
    return setup, crop, view, blank


def _validate_mesh_arrays(np, verts, faces):
    """Reason the MESH arrays cannot be placed, or ``""`` when they can."""
    if verts.ndim != 2 or verts.shape[-1] != 3:
        return f"vertices must be (N, 3), got shape {tuple(verts.shape)}"
    if faces.ndim != 2 or faces.shape[-1] != 3:
        return f"faces must be (M, 3) triangles, got shape {tuple(faces.shape)}"
    if not len(verts) or not len(faces):
        return f"{len(verts)} vertices / {len(faces)} faces — nothing to place"
    bad_v = int((~np.isfinite(verts).all(axis=1)).sum())
    if bad_v:
        return f"{bad_v} of {len(verts)} vertices are non-finite (NaN/inf)"
    if not np.issubdtype(faces.dtype, np.integer):
        if not np.isfinite(faces).all() or not np.all(faces == np.round(faces)):
            return "face indices are not integers"
    lo, hi = int(faces.min()), int(faces.max())
    if lo < 0 or hi >= len(verts):
        return (f"face indices span {lo}..{hi} but the mesh has {len(verts)} vertices "
                f"(valid 0..{len(verts) - 1})")
    return ""


def _import_mesh_arrays(np, mesh, max_faces, lines):
    """The MESH as numpy, VALIDATED, then decimated to the face budget.

    Returns ``((verts, faces, cols, texture), "")`` or ``(None, reason)`` for
    a MESH that cannot be placed; decimation (and a missed budget) is noted in
    ``lines``. ``texture`` is ``(uvs, tex_srgb)`` for a UV-textured MESH (core
    UnwrapMesh -> BakeTextureFromVoxel -> ApplyTextureToMesh), else None. A
    textured mesh is NEVER cluster-decimated here: merging vertices would
    scramble its UV layout, so the face budget is reported instead.
    """
    from atlas_camera.core.generated_mesh import cluster_decimate
    from atlas_camera.core.uv_bake import sample_texture_at_uvs

    try:
        verts, faces, cols = _mesh_item(np, mesh, validate=True)
    except (ValueError, IndexError, TypeError) as exc:
        return None, (str(exc) if isinstance(exc, ValueError)
                      else f"{type(exc).__name__}: {exc}")
    why = _validate_mesh_arrays(np, verts, faces)
    if why:
        return None, why
    faces = faces.astype(np.int64)
    if cols is not None and (cols.ndim != 2 or len(cols) != len(verts)):
        lines.append(f"warning: {len(cols)} vertex colours for {len(verts)} vertices — "
                     "colours dropped (hidden side painted neutral grey)")
        cols = None
    uvs, tex, tex_why = _mesh_texture(np, mesh, len(verts))
    if tex_why:
        lines.append(f"warning: MESH texture ignored — {tex_why}; using vertex colours")
    if tex is not None:
        if cols is None:
            # The per-vertex fallback (exporters without a texture path, the
            # colour grade, old viewports) comes from the texture itself.
            cols = sample_texture_at_uvs(tex, uvs).astype(np.float64)
        lines.append(f"UV texture {tex.shape[1]}x{tex.shape[0]} kept: hidden side painted "
                     "from the model's baked texture")
        if len(faces) > int(max_faces):
            lines.append(f"warning: {len(faces)} faces > max_faces {int(max_faces)} — a UV-"
                         "textured mesh is not decimated here (it would scramble the UVs); "
                         "lower DecimateMesh BEFORE UnwrapMesh to shrink it")
        return (verts, faces, cols, (uvs, tex)), ""
    n_in = len(faces)
    verts, faces, cols, dstats = cluster_decimate(
        verts, faces, max_faces=int(max_faces), colours=cols, return_stats=True)
    if len(faces) < n_in:
        lines.append(f"decimated {n_in} -> {len(faces)} faces (max_faces {int(max_faces)}; "
                     "decimate upstream with DecimateMesh for better quality)")
    if not dstats["met_budget"]:
        lines.append(f"warning: decimation MISSED the face budget — {len(faces)} faces > "
                     f"max_faces {int(max_faces)} after {dstats['rounds']} rounds; the "
                     "payload is larger than asked (decimate upstream with DecimateMesh)")
    return (verts, faces, cols, None), ""


def _register_scale(verts, faces, crop, view, setup, obj, rests_on_ground):
    """Compute: the mesh in the source camera and its one scale along the rays.

    Returns ``(cam_pts, reg, scale, ground_scale)``.
    """
    from atlas_camera.core.generated_mesh import (
        ground_contact_scale,
        pixal_to_source_camera,
        register_object_scale,
    )

    w, h = int(setup.width), int(setup.height)
    cam_pts = pixal_to_source_camera(verts, rotation=crop.rotation, fov_deg=crop.fov_deg)
    reg = register_object_scale(
        cam_pts, faces, view_matrix=view, fx=setup.fx, fy=setup.fy, cx=setup.cx,
        cy=setup.cy, width=w, height=h, metric_depth=setup.metric,
        object_mask=obj, depth_valid=setup.valid)
    s = reg.get("scale")
    s_ground = ground_contact_scale(cam_pts, view_matrix=view) if rests_on_ground else None
    return cam_pts, reg, s, s_ground


def _score_placement(np, reg, s, sky, obj, setup):
    """Measure: score the placement against the plate ({} when unscored)."""
    from atlas_camera.core.plate_falsification import score_geometry_against_plate

    alpha = reg["unit_alpha"]
    if not (bool(alpha.any()) and s is not None):
        return {}
    return score_geometry_against_plate(
        alpha=alpha, render_depth=reg["unit_depth"] * s, sky_mask=sky,
        observed_mask=obj,
        reference_depth=np.where(setup.valid, setup.metric, np.nan))


def _placement_grade(reg, s, s_ground, score):
    """Judge: the verdict comes from core.scene_health, never from here."""
    from atlas_camera.core.scene_health import generated_object_grade

    return generated_object_grade(
        s, reg["rel_mad"], score, ground_scale=s_ground,
        registration_note=reg.get("reason") or "",
        coverage_px=int(reg.get("coverage_px", 0)))


def _import_report_parts(name, verts, faces, s, s_ground, reg, grade, issues, score):
    """Report: ``(head, tail_lines)`` shared by every import outcome."""
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
    return head, report_tail


def _photo_weights(np, cam_pts, faces, view, s, setup, obj, reg):
    """Compute: world vertices + per-vertex photo weight ``(world, weight, wstats)``."""
    from atlas_camera.core.generated_mesh import photo_visibility_weights, source_camera_to_world

    w, h = int(setup.width), int(setup.height)
    world = source_camera_to_world(cam_pts, view_matrix=view, scale=float(s))
    weight, wstats = photo_visibility_weights(
        world, faces, view_matrix=view, fx=setup.fx, fy=setup.fy, cx=setup.cx,
        cy=setup.cy, width=w, height=h, object_mask=obj, metric_depth=setup.metric,
        mesh_depth=reg["unit_depth"] * float(s))
    return world, weight, wstats


def _grade_texture(np, uvs, tex, faces, gains):
    """Apply a LINEAR gain to an sRGB texture: 3 global values, or (N, 3) per
    vertex interpolated per texel through the UV layout (core.uv_bake); texels
    no triangle covers (the gutter) take the mean gain."""
    from atlas_camera.core.srgb import linear_to_srgb, srgb_to_linear
    from atlas_camera.core.uv_bake import rasterize_uv

    g = np.asarray(gains, dtype=np.float64)
    lin = srgb_to_linear(np.asarray(tex, dtype=np.float64))
    if g.ndim == 1:
        return np.clip(linear_to_srgb(lin * g[None, None, :]), 0.0, 1.0)
    h, w = lin.shape[:2]
    fi, bary = rasterize_uv(faces, uvs, (w, h))
    per = np.broadcast_to(g.mean(axis=0), (h, w, 3)).copy()
    cov = fi >= 0
    tri = np.asarray(faces, dtype=np.int64)[fi[cov]]
    per[cov] = (bary[cov].astype(np.float64)[:, :, None] * g[tri]).sum(axis=1)
    return np.clip(linear_to_srgb(lin * per), 0.0, 1.0)


def _grade_vertex_colours(torch, np, cols, weight, world, view, setup, image, match_colour,
                          faces=None):
    """Hidden-side colour: grey without vertex colours, else optionally graded
    onto the plate. Returns ``(colours, note, gains)``; ``gains`` is the applied
    LINEAR gain -- (N, 3) per vertex when ``faces`` allow the local match
    (core.generated_mesh.local_colour_gains), else 3 global values -- or None
    when no grade was applied."""
    gains = None
    from atlas_camera.core.generated_mesh import local_colour_gains, match_vertex_colours
    from atlas_camera.core.srgb import linear_to_srgb, srgb_to_linear

    w, h = int(setup.width), int(setup.height)
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
        raw = np.asarray(cols, dtype=np.float64)
        graded, crep = match_vertex_colours(raw, weight, img[iy, ix])
        colour_note = (f"vertex colours graded onto the plate: gain "
                       f"{[round(g, 3) for g in crep['gain']]} over "
                       f"{crep['seen_vertices']} seen vertices"
                       + (" (CLAMPED)" if any(crep["clamped"]) else "")
                       + ("" if crep["applied"] else " — too few seen vertices, not applied"))
        cols = graded
        if crep["applied"]:
            gains = [float(g) for g in crep["gain"]]
            if faces is not None:
                # LOCAL match on top: the global gain leaves a seam wherever
                # the model drifts in hue/exposure; this one fades to it.
                local, lrep = local_colour_gains(world, faces, raw, weight, img[iy, ix],
                                                 global_gain=gains)
                cols = np.clip(linear_to_srgb(srgb_to_linear(raw) * local), 0.0, 1.0)
                gains = local
                colour_note += (f"; LOCAL match near the seam (gain p5..p95 "
                                f"{lrep['gain_p05']}..{lrep['gain_p95']}, fades to the "
                                "global gain away from the photo)")
    else:
        colour_note = "vertex colours used as generated (match_colour off)"
    return cols, colour_note, gains


def _append_generated_prim(np, solve_out, name, world, faces, cols, weight, *, grade, issues,
                           s, reg, crop, wstats, texture=None):
    """APPEND the mesh as a PROXY_ROLE primitive on ``solve_out``.

    ``texture`` ``(uvs, tex_srgb)`` adds the model's own UV layout
    (``texture_uvs``, top-left convention) and its graded base-colour texture
    (``texture_b64``, PNG data URI) beside the per-vertex colours.
    Returns ``(prim, None)``, or ``(None, reason)`` when the mesh is rejected.
    """
    from atlas_camera.blender.measured import meshes_to_primitives

    accepted, rejected = meshes_to_primitives(
        solve_out, [{"name": name, "vertices": world, "faces": faces}],
        source=GENERATED_SOURCE, name_prefix=GENERATED_SOURCE, min_y_m=-1e3)
    if not accepted:
        return None, (rejected[0]["reason"] if rejected else "unknown")
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
    if texture is not None:
        uvs_t, tex_t = texture
        if len(uvs_t) == len(world):
            meta["texture_uvs"] = np.round(np.asarray(uvs_t).reshape(-1), 5).tolist()
            meta["texture_b64"] = _texture_data_uri(np, tex_t)
            meta["texture_size"] = [int(tex_t.shape[1]), int(tex_t.shape[0])]
    solve_out.projection_scene.proxy_geometry.append(prim)
    dbg = solve_out.projection_scene.debug_metadata
    dbg.setdefault("generated_objects", []).append({
        "name": prim.name, "grade": grade, "scale": float(s),
        "rel_mad": float(reg["rel_mad"]), "issues": issues,
        "photo_visibility": wstats,
    })
    return prim, None


def _comfy_read_roots() -> list[tuple[str, Any]]:
    """``[(label, Path)]`` of ComfyUI's output and input directories (empty
    outside ComfyUI)."""
    from pathlib import Path
    roots = []
    try:
        import folder_paths  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - not inside ComfyUI
        return roots
    for label, getter in (("output", "get_output_directory"),
                          ("input", "get_input_directory")):
        try:
            roots.append((label, Path(getattr(folder_paths, getter)())))
        except Exception:  # noqa: BLE001
            continue
    return roots


ABSOLUTE_READ_REFUSED = (
    "hdr_exr_path must be inside ComfyUI's output/input folder or $ATLAS_PROJECT_ROOT "
    "(or set ATLAS_ALLOW_ABSOLUTE_READS=1 to read absolute paths anywhere)")


def _absolute_read_roots() -> list[Any]:
    """Roots an ABSOLUTE read path may resolve inside: ComfyUI's output and
    input directories, plus ``$ATLAS_PROJECT_ROOT`` when set."""
    import os
    from pathlib import Path
    roots = [root for _label, root in _comfy_read_roots()]
    proj = (os.environ.get("ATLAS_PROJECT_ROOT") or "").strip()
    if proj:
        roots.append(Path(proj).expanduser())
    return roots


def _absolute_read_allowed(p: Any) -> bool:
    """Is absolute path ``p`` readable? Decided WITHOUT touching the file, so
    the answer (and the refusal message) never reveals whether it exists."""
    import os
    if os.environ.get("ATLAS_ALLOW_ABSOLUTE_READS", "").strip() == "1":
        return True
    try:
        cand = p.resolve()
    except Exception:  # noqa: BLE001
        return False
    for root in _absolute_read_roots():
        try:
            base = root.resolve()
        except Exception:  # noqa: BLE001
            continue
        if cand == base or base in cand.parents:
            return True
    return False


def _resolve_read_path(path: str) -> tuple[str, str]:
    """Resolve a READ path: ``(resolved, where_looked)``.

    Only two kinds of path resolve, so a graph cannot read arbitrary files:
    an ABSOLUTE path inside ComfyUI's output/input directory or
    ``$ATLAS_PROJECT_ROOT`` (anywhere only with
    ``ATLAS_ALLOW_ABSOLUTE_READS=1``), or a path RELATIVE to ComfyUI's output
    then input directory that stays inside that directory (``..`` cannot climb
    out). Never the process working directory. ``resolved`` is ``""`` when
    nothing matched; ``where_looked`` names the candidates for the report, or
    is exactly :data:`ABSOLUTE_READ_REFUSED` for an absolute path outside the
    allowed roots -- identical whether or not that file exists, so a shared
    workflow on a ``--listen`` server is not a file-existence oracle.
    """
    from pathlib import Path
    raw = str(path or "").strip().strip('"').strip()
    if not raw:
        return "", ""
    p = Path(raw)
    if p.is_absolute():
        if not _absolute_read_allowed(p):
            return "", ABSOLUTE_READ_REFUSED
        return (str(p), "") if p.is_file() else ("", f"absolute path {p} does not exist")
    looked = []
    for label, root in _comfy_read_roots():
        try:
            base = root.resolve()
            cand = (base / p).resolve()
        except Exception:  # noqa: BLE001
            continue
        if cand != base and base not in cand.parents:
            looked.append(f"{label}: escapes the {label} directory, ignored")
            continue
        if cand.is_file():
            return str(cand), ""
        looked.append(f"{label}: {cand}")
    if not looked:
        looked.append("relative paths resolve only under ComfyUI's output/input "
                      "directories (not running inside ComfyUI)")
    return "", "; ".join(looked)


def hdr_path_fingerprint(hdr_exr_path: str) -> str:
    """IS_CHANGED token for a path-gated read: resolved path + mtime_ns +
    size (stat only, never a content hash). A missing file gets its own
    token, so the file appearing later re-runs the node. A REFUSED absolute
    path gets one constant token (never stat'ed, never echoed)."""
    import os
    raw = str(hdr_exr_path or "").strip()
    if not raw:
        return "atlas-hdr:none"
    resolved, looked = _resolve_read_path(raw)
    if looked == ABSOLUTE_READ_REFUSED:
        return "atlas-hdr:refused"
    if not resolved:
        return f"atlas-hdr:missing:{raw}"
    try:
        st = os.stat(resolved)
    except OSError:
        return f"atlas-hdr:missing:{raw}"
    return f"atlas-hdr:{resolved}|{st.st_mtime_ns}|{st.st_size}"


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
                    "tooltip": "The HDR plate EXR from the matrixZone still workflow, "
                               "relative to ComfyUI's output or input folder (e.g. "
                               "atlas/hdr_plate_00001.exr), or absolute INSIDE the output/input "
                               "folder or $ATLAS_PROJECT_ROOT; other absolute paths are refused "
                               "unless ATLAS_ALLOW_ABSOLUTE_READS=1. Empty = pass through. A TYPED path "
                               "re-runs the node when the file changes on disk; a LINKED path "
                               "(e.g. from the stitch's exr_path) re-runs only when the upstream "
                               "node does -- ComfyUI does not pass linked values to IS_CHANGED."}),
                "hdr_image": ("IMAGE", {"tooltip": "Alternative to the path: a float HDR "
                                                   "IMAGE (ACEScg linear)."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, hdr_exr_path="", **_kwargs):
        """Gate doctrine: the EXR behind ``hdr_exr_path`` can be replaced at the
        same path, so the cache key is its stat fingerprint, not the string."""
        return hdr_path_fingerprint(hdr_exr_path)

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
            path, looked = _resolve_read_path(hdr_exr_path)
            if looked == ABSOLUTE_READ_REFUSED:
                return (solve_out, "AtlasHDRVertexTransfer: hdr_exr_path refused - "
                                   + ABSOLUTE_READ_REFUSED + " - passed through")
            if not path:
                return (solve_out, "AtlasHDRVertexTransfer: no HDR plate (set hdr_exr_path to "
                                   "the matrixZone still workflow's EXR) - passed through"
                        + (f"\nhdr_exr_path {str(hdr_exr_path).strip()!r} not found ({looked})"
                           if looked else ""))
            from atlas_camera.plate.oiio_io import read_plate
            hdr = np.asarray(read_plate(path, output_colorspace=None).pixels,
                             dtype=np.float32)[..., :3]
            origin = f"hdr_exr_path -> {path}"
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
            if p.metadata.get("texture_b64"):
                # A UV-textured object: the exporter bakes its texture and runs
                # it through this SAME curve (AtlasSceneTo3D's texture_hdr EXR).
                p.metadata["hdr_curve"] = {k: curve[k] for k in
                                           ("log2_x", "log2_y", "residual_stops", "space")}
            lines.append(f"- {p.name}: {len(vc)} vertex colours -> ACEScg linear, max "
                         f"{float(hdr_vc.max()):.2f}, {int((hdr_vc.max(-1) > 1).sum())} above 1.0")
        lines.append("the hidden side's HDR is the plate's conversion curve applied to a "
                     "model-guessed colour: a reconstruction of a reconstruction")
        return (solve_out, "\n".join(lines))
