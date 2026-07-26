"""Atlas ComfyUI nodes — planar unwarp/rewarp group.

Matte-paint round trip on a solved plane: AtlasPlanarUnwarp flattens the
ground (or any named proxy plane) into an orthographic texture via the
homography the recovered camera implies; the artist edits/inpaints the flat
image; AtlasPlanarRewarp composites the edit back into the plate,
perspective-correct by construction. The ATLAS_WARP_SPEC rides BY REFERENCE
between the two (the ATLAS_DEPTH_MAP pattern — never serialized).

Math lives in atlas_camera.core.planar_projection (pure numpy, host-agnostic);
this module is only the tensor/report adapter.
"""
from __future__ import annotations

from atlas_camera.comfy.node_helpers import (
    _require_numpy,
    _require_torch,
)


def _plane_candidates(solve):
    """(name, primitive) pairs for every plane-typed proxy on the solve."""
    seen = []
    scene = getattr(solve, "projection_scene", None)
    for prim in (getattr(scene, "proxy_geometry", None) or []):
        seen.append(prim)
    for src in (getattr(solve, "projection_sources", None) or []):
        for prim in (getattr(src, "proxy_geometry", None) or []):
            seen.append(prim)
    return [(p.name, p) for p in seen
            if str(getattr(p, "primitive_type", "")).lower() == "plane"]


def _scaled_intrinsics(solve, width, height):
    """(fx, fy, cx, cy) rescaled from the solve's stored resolution to the
    wired IMAGE tensor's — aspect-preserving per-axis scale."""
    intr = solve.camera.intrinsics
    sx = float(width) / float(intr.image_width or width)
    sy = float(height) / float(intr.image_height or height)
    return (float(intr.fx_px) * sx, float(intr.fy_px) * sy,
            float(intr.cx_px) * sx, float(intr.cy_px) * sy)


class AtlasPlanarUnwarp:
    """▱ Flatten a solved plane into an editable orthographic texture.

    Default plane is the solved GROUND (Y=0 — always exists). ``plane_name``
    (STRING, the *_override pattern) selects a named proxy plane instead — a
    facade from AtlasDeriveRoofsFacades, a wall, the backdrop; a name that
    matches nothing fails SOFT to the ground and lists what was available in
    the report. The flat_mask marks where a real plate pixel landed (0 =
    out-of-frame / behind camera / grazing incidence) — inpaint the zeros,
    edit the ones, then feed the result to AtlasPlanarRewarp with this node's
    warp_spec.
    """

    RETURN_TYPES = ("IMAGE", "MASK", "ATLAS_WARP_SPEC", "STRING")
    RETURN_NAMES = ("flat_image", "flat_mask", "warp_spec", "report")
    FUNCTION = "unwarp"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "image": ("IMAGE",),
            },
            "optional": {
                "plane_name": ("STRING", {"default": "", "tooltip":
                    "Named proxy plane to flatten (facade/wall/backdrop). "
                    "Blank = the solved ground plane."}),
                "px_per_meter": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0,
                    "tooltip": "0 = auto (match the plate's own pixel density "
                    "at the rect centre)."}),
                "center_u_m": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0,
                    "tooltip": "Manual rect centre (plane-space metres). Used only "
                    "when extent_u_m/extent_v_m > 0."}),
                "center_v_m": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0}),
                "extent_u_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20000.0,
                    "tooltip": "0 = auto-fit the visible footprint."}),
                "extent_v_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20000.0}),
                "max_resolution": ("INT", {"default": 4096, "min": 256, "max": 16384}),
            },
        }

    def unwarp(self, solve, image, plane_name="", px_per_meter=0.0,
               center_u_m=0.0, center_v_m=0.0, extent_u_m=0.0, extent_v_m=0.0,
               max_resolution=4096):
        from atlas_camera.core.planar_projection import (
            build_warp_spec,
            ground_plane_basis,
            plane_basis_from_primitive,
            unwarp_plate,
        )
        np = _require_numpy()
        torch = _require_torch()

        height, width = int(image.shape[1]), int(image.shape[2])
        notes = []
        basis = ground_plane_basis()
        wanted = str(plane_name or "").strip()
        if wanted:
            candidates = _plane_candidates(solve)
            match = next((p for n, p in candidates if n == wanted), None)
            if match is not None:
                basis = plane_basis_from_primitive(match)
            else:
                names = sorted({n for n, _ in candidates})
                notes.append(
                    f"plane '{wanted}' not found — fell back to the ground plane. "
                    f"Available plane proxies: {names or '(none on this solve)'}")

        view = solve.camera.extrinsics.camera_view_matrix
        fx, fy, cx, cy = _scaled_intrinsics(solve, width, height)
        rect = None
        if float(extent_u_m) > 0 and float(extent_v_m) > 0:
            rect = (float(center_u_m) - float(extent_u_m) / 2,
                    float(center_v_m) - float(extent_v_m) / 2,
                    float(center_u_m) + float(extent_u_m) / 2,
                    float(center_v_m) + float(extent_v_m) / 2)
        spec = build_warp_spec(
            view, fx, fy, cx, cy, basis, width, height,
            px_per_meter=float(px_per_meter), rect=rect,
            max_resolution=int(max_resolution))
        if spec is None:
            blank = torch.zeros(1, 8, 8, 3, dtype=torch.float32)
            blank_m = torch.zeros(1, 8, 8, dtype=torch.float32)
            return (blank, blank_m, None,
                    f"AtlasPlanarUnwarp: plane '{basis.name}' is not visible from "
                    "the recovered camera — nothing to flatten. " + " ".join(notes))

        arr = image[0].detach().cpu().numpy().astype(np.float32)
        flat, alpha = unwarp_plate(arr, spec)
        u_min, v_min, u_max, v_max = spec.metadata["rect"]
        report = (f"AtlasPlanarUnwarp: '{basis.name}' -> {spec.flat_width}x{spec.flat_height} px "
                  f"at {spec.px_per_meter:.1f} px/m; rect u [{u_min:.2f}, {u_max:.2f}] m, "
                  f"v [{v_min:.2f}, {v_max:.2f}] m; coverage {float(alpha.mean()):.1%}. "
                  "Edit the flat image (mask 1 = real plate pixels), then AtlasPlanarRewarp "
                  "with this warp_spec composites it back perspective-correct.")
        if notes:
            report += "\nWARNING: " + " ".join(notes)
        return (torch.from_numpy(flat).unsqueeze(0),
                torch.from_numpy(alpha).unsqueeze(0), spec, report)


class AtlasPlanarRewarp:
    """▱ Composite an edited flat image back into the plate.

    Inverse of AtlasPlanarUnwarp via the same warp_spec: pixels that map into
    the flat rect (and, when supplied, into ``edit_mask``) are replaced by the
    warped edit, feathered over ``feather_px`` plate pixels; everything
    off-plane passes through untouched.
    """

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "coverage_mask")
    FUNCTION = "rewarp"
    CATEGORY = "Atlas Camera/Derive Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "warp_spec": ("ATLAS_WARP_SPEC",),
                "edited_flat": ("IMAGE",),
                "original_image": ("IMAGE",),
            },
            "optional": {
                "edit_mask": ("MASK", {"tooltip": "Flat-space mask: restrict the "
                    "composite to the edited region (1 = take the edit)."}),
                "feather_px": ("INT", {"default": 4, "min": 0, "max": 64}),
            },
        }

    def rewarp(self, warp_spec, edited_flat, original_image, edit_mask=None,
               feather_px=4):
        from atlas_camera.core.planar_projection import rewarp_into_plate
        np = _require_numpy()
        torch = _require_torch()
        if warp_spec is None:
            h, w = int(original_image.shape[1]), int(original_image.shape[2])
            return (original_image, torch.zeros(1, h, w, dtype=torch.float32))

        flat = edited_flat[0].detach().cpu().numpy().astype(np.float32)
        orig = original_image[0].detach().cpu().numpy().astype(np.float32)
        mask = None
        if edit_mask is not None:
            mask = edit_mask[0].detach().cpu().numpy().astype(np.float32)
        out, cov = rewarp_into_plate(flat, orig, warp_spec,
                                     edit_mask=mask, feather_px=int(feather_px))
        return (torch.from_numpy(out).unsqueeze(0),
                torch.from_numpy(cov).unsqueeze(0))
