"""Conditioning bundle nodes: measured geometry as generative-model input.

These are the graph face of ``core.conditioning``. The computation lives in
core, unit-tested against closed forms with no ComfyUI and no torch; these
classes only convert at the boundary and explain themselves in a report.

WHY A CUSTOM SOCKET AND NOT IMAGE. ComfyUI's IMAGE is float32 but is
conventionally [0, 1], and every preview, save and composite hop in the
ecosystem clamps or encodes on that assumption. Metric depth in metres and
optical flow in pixels both leave that range immediately. Labelling them IMAGE
would therefore quantise measured data silently, which is precisely the failure
``ATLAS_RAYS`` was introduced to avoid for Plücker maps (see
``nodes_director.AtlasDirectorTake``). ComfyUI refusing to wire
``ATLAS_CONDITIONING`` into an image node is the protection, not an
inconvenience -- so every human-readable pass is offered SEPARATELY as an
explicitly display-only preview.
"""
from __future__ import annotations

import json

from pathlib import Path

from atlas_camera.comfy.node_helpers import _require_numpy, _require_torch

#: Per-pass EXR channel naming, recorded in the sidecar manifest rather than
#: re-derived from pixel order by a reader. Same doctrine as the take's
#: ``rayChannels``: a channel called "Z" that actually holds a normal is a
#: silent, plausible-looking error, and the only defence is writing the naming
#: down next to the data.
EXR_CHANNELS = {
    "depth": ["Z"],
    "normal": ["N.X", "N.Y", "N.Z"],
    "position": ["P.X", "P.Y", "P.Z"],
    "flow_fwd": ["FWD.X", "FWD.Y"],
    "flow_bwd": ["BWD.X", "BWD.Y"],
    "class": ["CLASS"],
}

#: Preview normalisation. These constants exist ONLY so a human can look at a
#: data pass; they are reported with every run so a preview can never be
#: mistaken for the measurement it was made from.
PREVIEW_FLOW_SCALE_PX = 32.0


class AtlasConditioningBundle:
    """Every per-frame conditioning signal along a camera path, measured.

    Renders each view of the move ONCE and reads depth, surface normals, world
    position, forward/backward optical flow and the ghost class map off that
    same z-buffer -- so no two passes can disagree with each other, or with the
    RGB, about what the geometry is.

    WHAT THIS IS FOR. A generative video model can be aimed at a camera move
    semantically ("slow dolly left, 35 mm") or geometrically. This node is the
    geometric route: it hands a model the exact depth, the exact displacement of
    every pixel, and -- through the ghost class map -- the only pixels it is
    entitled to invent, because they are the only ones the photograph never saw.
    Different models consume different subsets; the bundle emits all of them and
    lets the graph pick.

    FLOW IS DERIVED, NEVER ESTIMATED. With depth and two calibrated cameras the
    displacement is closed-form, so running a flow network over the render would
    be estimating something already known exactly. The occlusion flag that comes
    with it tests the target frame's own depth rather than inferring occlusion
    from flow inconsistency.

    A ZOOM IS NOT A DOLLY. Per-frame focal comes from the path's own ``fov_deg``
    channel, so a keyed zoom renders as a zoom -- with purely radial flow, a
    stationary eye, and (correctly) no disocclusion at all.

    The `*_preview` outputs are DISPLAY ONLY. Their normalisation constants are
    printed in the report; never wire one into anything but a preview.
    """

    CATEGORY = "Atlas/08 · Look & Render"
    FUNCTION = "build"
    RETURN_TYPES = ("ATLAS_CONDITIONING", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
                    "IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("bundle", "projected", "depth_preview", "normal_preview",
                    "flow_preview", "position_preview", "valid", "ghost",
                    "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE", {
                    "tooltip": "A solve carrying serialized projection meshes — "
                               "run AtlasDeriveProjectionGeometry or clean-plate "
                               "layers upstream. Without geometry there is "
                               "nothing to measure and this node refuses."}),
                "source_image": ("IMAGE", {
                    "tooltip": "The plate. Textures the primary geometry, and "
                               "sets the baseline the class map is measured "
                               "against."}),
            },
            "optional": {
                "camera_path": ("ATLAS_CAMERA_PATH", {
                    "tooltip": "The move to measure along. Without one only the "
                               "solved camera is rendered, which has no flow and "
                               "no disocclusion by definition."}),
                "resolution": ("INT", {
                    "default": 1024, "min": 256, "max": 4096,
                    "tooltip": "Long edge. The rasterizer is pure numpy and "
                               "O(faces x pixels) PER FRAME, and this renders "
                               "every frame — keep it moderate on long moves "
                               "over dense relief meshes, or use stride."}),
                "hole_dilate_px": ("INT", {
                    "default": 0, "min": 0, "max": 64,
                    "tooltip": "Grow the uncovered region before classifying. A "
                               "z-buffered render leaves one-pixel seams along "
                               "silhouettes; a few px turns speckle into "
                               "coherent blobs. Only ever marks MORE as unseen, "
                               "so it cannot invent coverage."}),
                "exclude_mask": ("MASK", {
                    "tooltip": "Region already handled by something other than "
                               "geometry — sky on a matte or a SkyDome. Excluded "
                               "pixels are classified LOW_CONFIDENCE and left "
                               "out of the fractions entirely."}),
                "last_n": ("INT", {
                    "default": 0, "min": 0, "max": 4096,
                    "tooltip": "Measure only the move's last N frames (0 = "
                               "every frame). The end of a move is where "
                               "disocclusion peaks."}),
                "stride": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": "Measure every Nth frame. Flow is then computed "
                               "between the KEPT frames, so its magnitudes are "
                               "per-stride, not per-source-frame — the report "
                               "says so."}),
                "emit_flow": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Forward and backward flow, derived from "
                               "geometry. Costs one back-projection and two "
                               "projections per frame."}),
                "emit_normals": ("BOOLEAN", {"default": True}),
                "emit_position": ("BOOLEAN", {"default": True}),
            },
        }

    def build(self, solve, source_image, camera_path=None, resolution=1024,
              hole_dilate_px=0, exclude_mask=None, last_n=0, stride=1,
              emit_flow=True, emit_normals=True, emit_position=True):
        from atlas_camera.comfy.headless_evidence import (
            _decode_mask, _decode_rgba, _fit_long_edge, _tensor_rgba,
        )
        from atlas_camera.comfy.nodes_viewport import AtlasDisocclusionGuide
        from atlas_camera.core.camera_path import sample_camera_path_intrinsics
        from atlas_camera.core.conditioning import render_conditioning_sequence
        from atlas_camera.core.ghost_pixels import GhostClass, class_masks
        from atlas_camera.core.projection_render import gather_scene_meshes

        np = _require_numpy()
        torch = _require_torch()

        sh, sw = int(source_image.shape[1]), int(source_image.shape[2])
        meshes = gather_scene_meshes(solve, with_uvs=True)
        if not meshes:
            # Refuse the same way AtlasGhostPixelMap and AtlasDisocclusionGuide
            # do: an empty bundle read as "nothing to condition on" would send a
            # model off inventing a whole frame with no geometric brief at all.
            report = json.dumps({
                "node": "AtlasConditioningBundle",
                "error": "no serialized projection meshes on this solve",
                "remedy": "run AtlasDeriveProjectionGeometry (or clean-plate "
                          "layers) upstream first",
                "warning": "NOTHING was measured; do not read the empty masks "
                           "as 'fully covered' or the absent flow as 'static'",
            }, indent=2)
            empty = torch.zeros(1, sh, sw, dtype=torch.float32)
            return (None, source_image, source_image, source_image,
                    source_image, source_image, empty, empty, report)

        width, height = _fit_long_edge(sw, sh, int(resolution))
        intr = solve.camera.intrinsics
        sx = float(width) / float(intr.image_width or width)
        sy = float(height) / float(intr.image_height or height)
        fx, fy = float(intr.fx_px) * sx, float(intr.fy_px) * sy
        cx, cy = float(intr.cx_px) * sx, float(intr.cy_px) * sy

        textures = {"primary": _tensor_rgba(source_image, width, height)}
        for source in getattr(solve, "projection_sources", None) or []:
            name = str(getattr(source, "name", "") or "layer")
            rgba = _decode_rgba(getattr(source, "image_b64", None) or "",
                                width, height)
            if rgba is None:
                continue
            mask = _decode_mask(getattr(source, "mask_b64", None), width, height)
            rgba = np.array(rgba, dtype=np.float64, copy=True)
            rgba[..., 3] = rgba[..., 3] * mask
            textures[name] = rgba

        exclude = AtlasDisocclusionGuide._exclude(exclude_mask, width, height, np)
        views, view_source = AtlasDisocclusionGuide._views(solve, camera_path)
        total = len(views)

        # Per-frame K from the path's own fov channel. One K per view, always:
        # letting the core function fall back to the solved focal would render a
        # keyed zoom as no change at all.
        if camera_path is not None:
            intrinsics = sample_camera_path_intrinsics(
                camera_path, fx=fx, fy=fy, cx=cx, cy=cy, height=height)
        else:
            intrinsics = []
        if len(intrinsics) != total:
            intrinsics = [[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
                          for _ in range(total)]

        kept = list(range(total))
        if int(stride) > 1:
            kept = kept[::int(stride)]
        if int(last_n) > 0 and len(kept) > int(last_n):
            kept = kept[-int(last_n):]
        views = [views[i] for i in kept]
        intrinsics = [intrinsics[i] for i in kept]
        if len(kept) != total:
            view_source += (f", {len(kept)} of {total} frames "
                            f"(stride={int(stride)}, last_n={int(last_n)})")

        plate_k = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        seq = render_conditioning_sequence(
            meshes, textures, views=views, intrinsics=intrinsics,
            plate_view=solve.camera.extrinsics.camera_view_matrix,
            plate_k=plate_k, width=width, height=height, exclude=exclude,
            hole_dilate_px=int(hole_dilate_px),
            want_flow=bool(emit_flow), want_normals=bool(emit_normals),
            want_position=bool(emit_position))
        seq.meta["source_frame_indices"] = kept
        seq.meta["view_source"] = view_source
        seq.meta["stride"] = int(stride)

        def batch(arrays):
            return torch.from_numpy(np.stack(arrays)).float()

        valid, ghost = [], []
        for index in range(seq.frames):
            masks = class_masks(seq.class_map[index])
            valid.append(masks["valid"].astype(np.float32))
            ghost.append(masks["ghost"].astype(np.float32))

        previews, preview_meta = self._previews(np, seq, height, width)
        report = json.dumps({
            "node": "AtlasConditioningBundle",
            "frames": seq.frames,
            "raster": [width, height],
            "view_source": view_source,
            "source_frame_indices": kept,
            "class_values": {c.name: int(c) for c in GhostClass},
            "emitted": {
                "depth_m": seq.depth_m is not None,
                "normal_world": seq.normal_world is not None,
                "position_world": seq.position_world is not None,
                "flow": seq.flow_fwd is not None,
            },
            "units": {
                "depth": seq.meta["depth_units"],
                "flow": seq.meta["flow_units"] + (
                    f" between KEPT frames (stride {int(stride)})"
                    if int(stride) > 1 else ""),
                "position": "metres, world space, Atlas Y-up",
                "normal": "unit vectors, world space",
            },
            "principal_point": seq.meta["principal_point"],
            "focal_varies": len({round(k[1][1], 6) for k in seq.k}) > 1,
            "parallax_px": seq.meta["parallax_px"],
            "frames_without_parallax": seq.meta["frames_without_parallax"],
            "per_frame_class_stats": seq.per_frame,
            "skipped_meshes": seq.meta["skipped_meshes"],
            "preview_normalisation": preview_meta,
            "note": "The bundle socket carries the MEASUREMENT. Every *_preview "
                    "output is display-only and normalised by the constants "
                    "above — never wire one into anything but a preview. Ghost "
                    "is the only class a generator should be aimed at.",
        }, indent=2)

        return (seq, batch(seq.rgb), previews["depth"], previews["normal"],
                previews["flow"], previews["position"], batch(valid),
                batch(ghost), report)

    @staticmethod
    def _previews(np, seq, height, width):
        """Human-readable versions of the data passes, and how they were made.

        Depth is normalised per SEQUENCE, not per frame: a per-frame stretch
        makes a dolly look like a static shot because every frame re-normalises
        to its own near/far. Flow is scaled by a FIXED constant for the same
        reason -- an auto-scaled flow preview hides exactly the magnitude change
        a reader is looking for.
        """
        import torch

        def grey(stack):
            return torch.from_numpy(
                np.repeat(np.nan_to_num(stack)[..., None], 3, axis=-1)).float()

        meta = {}
        blank = torch.zeros(max(seq.frames, 1), height, width, 3,
                            dtype=torch.float32)

        if seq.depth_m is not None:
            finite = seq.depth_m[np.isfinite(seq.depth_m)]
            near = float(finite.min()) if finite.size else 0.0
            far = float(finite.max()) if finite.size else 1.0
            span = max(far - near, 1e-6)
            depth = np.clip((seq.depth_m - near) / span, 0.0, 1.0)
            depth_preview = grey(depth)
            meta["depth"] = {"near_m": near, "far_m": far,
                             "mapping": "linear near->0, far->1, per SEQUENCE",
                             "holes": "rendered as 0 (NaN in the bundle)"}
        else:
            depth_preview = blank

        if seq.normal_world is not None:
            normal_preview = torch.from_numpy(
                np.nan_to_num(seq.normal_world) * 0.5 + 0.5).float()
            meta["normal"] = {"mapping": "n*0.5+0.5, world space"}
        else:
            normal_preview = blank

        if seq.flow_fwd is not None:
            f = np.nan_to_num(seq.flow_fwd) / PREVIEW_FLOW_SCALE_PX
            rgb = np.stack([np.clip(f[..., 0] * 0.5 + 0.5, 0, 1),
                            np.clip(f[..., 1] * 0.5 + 0.5, 0, 1),
                            np.zeros(f.shape[:3], dtype=np.float32)], axis=-1)
            flow_preview = torch.from_numpy(rgb).float()
            meta["flow"] = {"scale_px": PREVIEW_FLOW_SCALE_PX,
                            "mapping": "R=u, G=v, (f/scale)*0.5+0.5, clipped",
                            "note": "FIXED scale, not auto — an auto-scaled "
                                    "preview hides the magnitude change"}
        else:
            flow_preview = blank

        if seq.position_world is not None:
            p = np.nan_to_num(seq.position_world)
            lo, hi = float(p.min()), float(p.max())
            span = max(hi - lo, 1e-6)
            position_preview = torch.from_numpy((p - lo) / span).float()
            meta["position"] = {"min_m": lo, "max_m": hi,
                               "mapping": "linear per SEQUENCE over all axes"}
        else:
            position_preview = blank

        return ({"depth": depth_preview, "normal": normal_preview,
                 "flow": flow_preview, "position": position_preview}, meta)


class AtlasWriteConditioningEXR:
    """Write a conditioning bundle to float32 EXR sequences on disk.

    Separate from the bundle node because writing to disk is a side effect that
    must be opted into, and because OpenImageIO is an optional dependency while
    the sockets are not -- a missing OIIO SKIPS this node rather than failing
    the graph.

    ``bit_depth="float"`` is mandatory and never half. Half's mantissa step is
    ample for imagery and is exactly wrong for data: at depths of tens of metres
    it quantises to decimetres, and flow in pixels loses sub-pixel precision
    that the measurement has and a consumer needs. No colour conversion is
    applied either -- these are numbers, not colour.

    One EXR per pass per frame rather than one fat multi-channel file, because
    ``write_exr`` names channels positionally; the naming is recorded in a
    sidecar manifest so a reader never has to infer which channel is which.
    """

    CATEGORY = "Atlas/10 · Export"
    FUNCTION = "write"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("directory", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bundle": ("ATLAS_CONDITIONING", {
                    "tooltip": "From AtlasConditioningBundle."}),
                "directory": ("STRING", {
                    "default": "atlas_exports/conditioning",
                    "tooltip": "Written relative to the ComfyUI process cwd "
                               "unless absolute."}),
            },
            "optional": {
                "basename": ("STRING", {"default": "atlas"}),
                "write_flow": ("BOOLEAN", {"default": True}),
                "write_position": ("BOOLEAN", {"default": True}),
                "write_normals": ("BOOLEAN", {"default": True}),
            },
        }

    def write(self, bundle, directory, basename="atlas", write_flow=True,
              write_position=True, write_normals=True):
        from atlas_camera.plate.oiio_io import oiio_available, write_exr

        np = _require_numpy()
        if bundle is None:
            return ("", json.dumps({
                "node": "AtlasWriteConditioningEXR",
                "error": "no bundle — the upstream node refused or was bypassed",
            }, indent=2))
        if not oiio_available():
            return ("", json.dumps({
                "node": "AtlasWriteConditioningEXR",
                "skipped": "OpenImageIO is not installed",
                "remedy": "pip install -e .[oiio]",
                "note": "SKIPPED, not failed: the bundle socket is the primary "
                        "product and OIIO is optional in this package.",
            }, indent=2))

        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        passes: dict = {"depth": bundle.depth_m, "class": bundle.class_map}
        if write_normals:
            passes["normal"] = bundle.normal_world
        if write_position:
            passes["position"] = bundle.position_world
        if write_flow:
            passes["flow_fwd"] = bundle.flow_fwd
            passes["flow_bwd"] = bundle.flow_bwd

        written: dict = {}
        for name, stack in passes.items():
            if stack is None:
                continue
            out_dir = root / name
            out_dir.mkdir(parents=True, exist_ok=True)
            for index in range(stack.shape[0]):
                frame = np.asarray(stack[index], dtype=np.float32)
                if frame.ndim == 2:
                    frame = frame[..., None]
                path = out_dir / f"{basename}_{name}.{index:04d}.exr"
                write_exr(str(path), frame, bit_depth="float")
            written[name] = {"frames": int(stack.shape[0]),
                             "channels": EXR_CHANNELS[name],
                             "directory": str(out_dir)}

        manifest = {
            "node": "AtlasWriteConditioningEXR",
            "bit_depth": "float32",
            "colour_conversion": "none — these are data passes, not colour",
            "raster": bundle.meta.get("raster"),
            "frames": bundle.frames,
            "units": {"depth": bundle.meta.get("depth_units"),
                      "flow": bundle.meta.get("flow_units"),
                      "position": "metres, world space, Atlas Y-up",
                      "class": "GhostClass integer codes"},
            "passes": written,
            "camera": {"k_per_frame": bundle.k,
                       "principal_point": bundle.meta.get("principal_point"),
                       "view_matrices": [
                           np.asarray(v, dtype=float).tolist()
                           for v in bundle.views]},
            "note": "NaN marks a hole in every metric pass. Channel names are "
                    "recorded here rather than inferred from pixel order.",
        }
        (root / f"{basename}_conditioning_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        return (str(root), json.dumps(manifest, indent=2))
