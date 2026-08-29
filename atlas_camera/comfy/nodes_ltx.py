"""Atlas ComfyUI nodes — LTX camera path group.

Reads the `camera_ltx.json` an Atlas Director take exports and hands the
CrossView-Warp node the `keyframes` string it wants.

The node lives HERE rather than in the fork of ComfyUI-CrossViewWarp on
purpose. Atlas owns the Atlas format, so a change to what Director writes is a
change in one repo; the fork stays as close to upstream as it can, which is
what keeps it mergeable. The two packs sit side by side in `custom_nodes/`, so
wiring one into the other costs nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

#: What Director can write, and whether the file is the take or a stand-in.
#: Keys are the format ids `app/director/export/ltx.ts` stamps; a file carrying
#: anything else is refused rather than guessed at, because every one of these
#: means something different about how far you can trust the render.
FORMATS = {
    "atlas.ltx.crossview_warp": "single path",
    "atlas.ltx.crossview_warp.chain": "chain of segments",
    "atlas.ltx.crossview_warp.compressed": "compressed (previz)",
    "atlas.ltx.crossview_warp.pose": "per-frame pose",
}


def _fmt(pair) -> str:
    return f"max {pair['max']:.3f}, p95 {pair['p95']:.3f}"


def _report(document: dict, segment: dict | None, segment_count: int) -> str:
    """What the artist needs to know before trusting the render.

    Front-loads the node settings, because those are the failure that looks
    like success: `pivot_override` off makes the animated pivot inert and
    `keep_source_aim` on overrides the aim the knots encode. Either renders a
    DIFFERENT move with no error anywhere. Then the losses, then the warnings.
    """
    body = segment if segment is not None else document
    residual = body.get("residual", {})
    meta = body.get("meta", {})
    requires = meta.get("requires", {})
    lines = [
        f"Atlas camera path — {FORMATS.get(document.get('format'), 'unknown')}",
        "",
        "SET ON CrossView Warp (silently renders a different move otherwise):",
        f"  use_keyframes   = {requires.get('use_keyframes')}",
        f"  pivot_override  = {requires.get('pivot_override')}",
        f"  keep_source_aim = {requires.get('keep_source_aim')}",
        "",
        f"frames {meta.get('frameCount')}, pivot at {meta.get('pivotDepthM')}m "
        f"(the take's focus distance)",
    ]

    if segment is not None:
        source = segment.get("sourceFrames", {})
        lines += [
            "",
            f"SEGMENT {segment.get('index')} of {segment_count} — "
            f"take frames {source.get('first')}-{source.get('last')}, "
            f"{segment.get('travelM', 0):.2f}m of travel.",
        ]
        if segment.get("index", 1) > 1:
            lines.append(
                "  Its source clip is the PREVIOUS segment's final generated frame, "
                "not the plate. Re-derive geometry from that frame, and expect this "
                "segment's error to ride on the one before it."
            )

    if document.get("format") == "atlas.ltx.crossview_warp.pose":
        focal = document.get("focalLengthMm", {})
        return "\n".join([
            "Atlas camera path — per-frame pose",
            "",
            "Wire into CrossView Warp's `camera_path`. It outranks camera_info "
            "and the keyframes, so no orbit widget applies and none needs setting.",
            "",
            f"frames {document.get('frameCount')}, "
            f"focal {focal.get('min')}-{focal.get('max')}mm",
            f"frame: {document.get('frame')}",
            "",
            "Carries: " + ", ".join(document.get("carries", [])) + ".",
            "Nothing is approximated here — but the LoRA still only saw its "
            "training range, so a pose outside it renders as extrapolation.",
        ])

    compression = document.get("compression")
    if compression:
        travel = compression.get("travelM", {})
        lines += [
            "",
            f"COMPRESSED — previz only. Travel scaled by {compression.get('scale'):.3f}: "
            f"{travel.get('original', 0):.2f}m performed, {travel.get('compressed', 0):.2f}m "
            f"rendered. Timing and aim are the take's; the parallax is not, so do not "
            f"judge depth from this.",
        ]

    if residual:
        offset = residual.get("orbitalOffsetDeg", {})
        focal = residual.get("focalLengthMm", {})
        lines += [
            "",
            "Carried exactly:",
            f"  position  {_fmt(residual.get('positionErrorM', {'max': 0, 'p95': 0}))} m",
            f"  aim       {_fmt(residual.get('aimErrorDeg', {'max': 0, 'p95': 0}))} deg",
            "Not carried:",
            f"  roll      {_fmt(residual.get('discardedRollDeg', {'max': 0, 'p95': 0}))} deg "
            f"(the orbit builds every pose level)",
        ]
        if not focal.get("constant", True):
            lines.append(
                f"  focal     {focal.get('min')}-{focal.get('max')}mm varies; the node has "
                f"one static hfov, so breathing is dropped"
            )
        lines.append(
            f"Asks the LoRA to orbit {offset.get('azimuthMax', 0):.1f}deg azimuth, "
            f"{offset.get('elevationMin', 0):.1f} to {offset.get('elevationMax', 0):.1f}deg "
            f"elevation from frame 1."
        )

    lines += [
        "",
        "The clip must be at least as long as the path: the warp node errors on a "
        "keyframe past the end, which is the check this node cannot make for you.",
    ]
    return "\n".join(lines)


class AtlasLoadCameraPath:
    """🎥 A Director take as a CrossView-Warp camera path.

    Wire `keyframes` into the CrossView Warp node's `keyframes` input. Doing so
    makes its orbit sphere read-only, which is correct: the path is a recording
    of a camera that was operated, and a marker dragged in the graph would show
    a move that is not the one being rendered.

    Reads all four things Director can write, and says which it got. A take too
    long for one generation is exported as a CHAIN — pick the segment with
    `segment`, generate it, then feed its final frame back as the source clip
    for the next. A COMPRESSED file is previz: the timing is the take's and the
    parallax is not, and the report says so rather than letting a third-depth
    push pass for the shot.

    A POSE path goes into `camera_path` instead, on the extended node. That
    input outranks both `camera_info` and the keyframes and carries the pose per
    frame — roll and a breathing lens included — so none of the orbit widgets
    apply and none needs setting. It is the same output slot either way; the
    report says which input the file belongs in.

    The report output is worth wiring to a PreviewText at least once per graph.
    It carries the three node settings the knots depend on, and getting those
    wrong does not raise anything — it just renders a different move.
    """

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("keyframes", "report", "segment_count")
    FUNCTION = "load"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "",
                    "tooltip": "The camera_ltx.json Director wrote beside the "
                               "take, in its takes/<slate>/ directory."}),
            },
            "optional": {
                "segment": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": "Which segment of a chained path to load. "
                               "Ignored for a single path. Segment N's source "
                               "clip is segment N-1's final generated frame."}),
            },
        }

    def load(self, path: str, segment: int = 1):
        source = Path(path.strip().strip('"'))
        if not source.is_file():
            raise ValueError(
                f"Atlas camera path not found: {source}. Export a take with the 'ltx' "
                f"target and point this at the camera_ltx.json beside it."
            )
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"{source.name} is not readable JSON: {error}") from None

        fmt = document.get("format")
        if fmt not in FORMATS:
            # Refusing an unknown id is the point of stamping one. A compressed
            # path read as an ordinary one would render a scaled-down move with
            # nothing anywhere saying so.
            raise ValueError(
                f"{source.name} says format {fmt!r}, which this node does not know. "
                f"It reads: {', '.join(sorted(FORMATS))}."
            )

        if fmt == "atlas.ltx.crossview_warp.chain":
            segments = document.get("segments") or []
            count = len(segments)
            if not 1 <= segment <= count:
                raise ValueError(
                    f"{source.name} is a chain of {count} segment(s); segment {segment} "
                    f"does not exist."
                )
            chosen = segments[segment - 1]
            return (chosen["keyframes"], _report(document, chosen, count), count)

        if fmt == "atlas.ltx.crossview_warp.pose":
            # The whole document, not a knot list: `camera_path` parses the
            # poses itself. Sent on the same slot so one node serves both, and
            # the report says which input to wire it into.
            return (json.dumps(document), _report(document, None, 1), 1)

        return (document["keyframes"], _report(document, None, 1), 1)


class AtlasReliefGeometry:
    """🗺 The solve's own geometry, as MoGe-shaped metric depth.

    CrossView Warp wants `moge_geometry`: metric depth per pixel, a validity
    mask, and intrinsics. Running MoGe inside the graph produces all three, and
    for a pose path that is a hazard rather than a convenience — the camera
    arrives in METRES from an Atlas solve while the depth is a separate estimate
    of the same scene, made by a different model with its own scale and its own
    recovered lens. Nothing checks that the two agree, and when they disagree
    the move renders at the wrong depth with no error anywhere.

    This takes the geometry from the solve the camera came from. The relief mesh
    is already in Atlas world metres, already tuned by whatever edge settings
    built it, and it is rasterised through the solve's own intrinsics — so depth
    and pose cannot disagree, because they are the same measurement.

    The holes are the mesh's real tears, not a model's guess at sky. Where the
    relief has no triangle the depth is NaN and the mask is False, which the
    warp renders as magenta for the LoRA to fill. That is the honest answer to
    "what was never photographed", and it is the same answer 📽 Project draws.
    """

    RETURN_TYPES = ("MOGE_GEOMETRY", "STRING")
    RETURN_NAMES = ("moge_geometry", "report")
    FUNCTION = "build"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE", {}),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Raster width. Match the frames you feed the warp."}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Raster height. Match the frames you feed the warp."}),
            },
            "optional": {
                "frames": ("INT", {"default": 1, "min": 1, "max": 2048,
                    "tooltip": "How many frames to emit. A still plate warped by "
                               "a camera path needs one per frame of the clip."}),
            },
        }

    def build(self, solve, width: int, height: int, frames: int = 1):
        import numpy as np
        import torch

        from atlas_camera.core.camera_spec import CameraSpec
        from atlas_camera.core.mesh_voxel import render_depth_grid
        from atlas_camera.core.relief_mesh import _relief_mesh_from_solve

        mesh = _relief_mesh_from_solve(solve)
        if mesh is None:
            raise ValueError(
                "AtlasReliefGeometry: this solve carries no relief mesh. Derive one "
                "first (AtlasDeriveReliefMesh / AtlasInput) — there is no geometry "
                "here to rasterise, and inventing one would defeat the point of "
                "using the solve's own.")

        # Intrinsics scale with the raster, exactly as voxel_remesh does: the
        # solve's fx is in source-image pixels and means nothing at another size.
        cam = CameraSpec.from_solve(solve)
        src_w = int(getattr(cam, "width", 0) or solve.camera.intrinsics.width)
        src_h = int(getattr(cam, "height", 0) or solve.camera.intrinsics.height)
        sx, sy = width / float(src_w), height / float(src_h)

        depth = render_depth_grid(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
            cam.view_matrix,
            float(cam.fx) * sx, float(cam.fy) * sy,
            float(cam.cx) * sx, float(cam.cy) * sy,
            int(width), int(height),
        )
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any():
            raise ValueError(
                "AtlasReliefGeometry: the relief covers no pixel from the recovered "
                "camera. The mesh and the camera are not in the same space.")

        # The warp reads depth[i] per frame and reprojects it, so a still plate
        # needs the same geometry repeated rather than broadcast — matching how
        # its frames were repeated.
        d = np.where(valid, depth, np.nan).astype(np.float32)
        depth_t = torch.from_numpy(np.repeat(d[None], int(frames), axis=0))
        mask_t = torch.from_numpy(np.repeat(valid[None], int(frames), axis=0))
        # Normalised, which is how the warp reads it back: fx = K[0][0,0] * W.
        k = np.eye(3, dtype=np.float32)
        k[0, 0] = float(cam.fx) * sx / width
        k[1, 1] = float(cam.fy) * sy / height
        k[0, 2] = float(cam.cx) * sx / width
        k[1, 2] = float(cam.cy) * sy / height
        intrinsics = torch.from_numpy(np.repeat(k[None], int(frames), axis=0))

        near, far = float(np.nanmin(d)), float(np.nanmax(d))
        report = "\n".join([
            "Atlas relief geometry — from the solve, not inferred",
            f"  raster      {width}x{height} from a {src_w}x{src_h} plate",
            f"  depth       {near:.2f}m to {far:.2f}m (metric, Atlas world)",
            f"  covered     {valid.mean()*100:.1f}% of frame; the rest is the "
            f"mesh's own tears and renders as magenta",
            f"  lens        fx {float(cam.fx)*sx:.1f}px "
            f"({np.degrees(2*np.arctan(width/(2*float(cam.fx)*sx))):.1f} deg horizontal)",
            f"  frames      {frames}",
            "",
            "Depth and camera come from one measurement, so a metric camera_path "
            "cannot disagree with the geometry it moves through.",
        ])
        return ({"depth": depth_t, "mask": mask_t, "intrinsics": intrinsics}, report)


class AtlasUnrealDepthGeometry:
    """🗺 A rendered Unreal depth pass, as MoGe-shaped metric depth.

    The sibling of `AtlasReliefGeometry`, for the case that node cannot cover.
    The relief mesh is the SOLVE's geometry — the terrain that was photographed.
    A dressed frame carries what was built on top of it in Unreal, and none of
    that is in the relief: a castle added in the build rasterises at the hill's
    depth, so it warps as paint on the hillside rather than a thing standing on
    it. This reads the depth Unreal itself rendered, which has both.

    The pairing argument is the same one, and it is the whole point. Depth
    rendered through the Unreal camera and a pose path recorded in the same
    scene are one measurement, so they cannot disagree about scale or lens. A
    monocular estimate of the same frame is a second opinion with its own scale
    and its own recovered focal, and nothing downstream checks the two against
    each other.

    Feed it Movie Render Queue's `SceneDepthWorldUnits` post-process pass, as
    32-bit EXR. That material divides Unreal's centimetres down, so the values
    arrive in metres — `depth_scale` is there for the day that stops being true,
    not because it is expected to be needed. Verify once against something whose
    distance you know rather than trusting the name of the channel.

    `far_clip_m` exists for backplates. An ImagePlate standing in for sky is real
    geometry to the renderer and comes back at the card's distance, which the
    warp will happily parallax like a wall. Clipping beyond it puts the sky back
    where it belongs: masked out, magenta, for the LoRA to fill.
    """

    RETURN_TYPES = ("MOGE_GEOMETRY", "STRING")
    RETURN_NAMES = ("moge_geometry", "report")
    FUNCTION = "build"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "exr_path": ("STRING", {"default": "",
                    "tooltip": "The EXR Movie Render Queue wrote. Point at the file, "
                               "not the directory."}),
                "focal_mm": ("FLOAT", {"default": 35.0, "min": 1.0, "max": 1000.0, "step": 0.1,
                    "tooltip": "The RENDERING camera's focal length, off the CineCamera "
                               "component. Not the take's lens unless they are the same "
                               "camera."}),
                "sensor_width_mm": ("FLOAT", {"default": 23.5, "min": 1.0, "max": 200.0, "step": 0.1,
                    "tooltip": "That camera's filmback width. Focal and filmback together "
                               "give the horizontal field of view; either one alone does "
                               "not."}),
                "width": ("INT", {"default": 960, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Raster width. Match the frames you feed the warp."}),
                "height": ("INT", {"default": 640, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Raster height. Match the frames you feed the warp."}),
            },
            "optional": {
                "frames": ("INT", {"default": 1, "min": 1, "max": 2048,
                    "tooltip": "How many frames to emit. A still plate warped by a "
                               "camera path needs one per frame of the clip."}),
                "channel": ("STRING", {"default": "FinalImageSceneDepthWorldUnits",
                    "tooltip": "Layer name inside the EXR. MRQ names it after the "
                               "post-process material, so renaming the material "
                               "renames this."}),
                "depth_scale": ("FLOAT", {"default": 1.0, "min": 1e-6, "max": 1e6, "step": 0.01,
                    "tooltip": "Multiplied onto the channel to reach metres. 1.0 for "
                               "SceneDepthWorldUnits; 0.01 if you ever hand it raw "
                               "centimetres."}),
                "far_clip_m": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 1.0,
                    "tooltip": "Mask out anything at or beyond this depth, in metres. "
                               "0 keeps everything. Use it to drop a backplate card "
                               "standing in for sky."}),
            },
        }

    def build(self, exr_path: str, focal_mm: float, sensor_width_mm: float,
              width: int, height: int, frames: int = 1,
              channel: str = "FinalImageSceneDepthWorldUnits",
              depth_scale: float = 1.0, far_clip_m: float = 0.0):
        import numpy as np
        import torch

        src = Path(str(exr_path).strip().strip('"'))
        if not src.is_file():
            raise ValueError(
                f"Atlas Unreal depth: no EXR at {src}. Render Movie Render Queue's "
                "SceneDepthWorldUnits post-process pass and point this at the file."
            )
        try:
            import OpenEXR
        except ImportError as exc:
            raise RuntimeError(
                "Reading a multipart EXR needs the OpenEXR package: pip install OpenEXR"
            ) from exc

        with OpenEXR.File(str(src)) as handle:
            channels = handle.parts[0].channels
            if channel not in channels:
                raise ValueError(
                    f"Atlas Unreal depth: {src.name} has no channel {channel!r}. "
                    f"It carries {sorted(channels)}. MRQ names the layer after the "
                    "post-process material, so a renamed material renames this."
                )
            raw = np.asarray(channels[channel].pixels, dtype=np.float64)

        d = raw[..., 0] if raw.ndim == 3 else raw
        src_h, src_w = d.shape
        d = d * float(depth_scale)

        # Nearest, never bilinear: averaging across a depth discontinuity invents
        # a surface at a distance nothing in the scene occupies, and the warp
        # would smear the silhouette across it.
        yi = np.clip((np.arange(height) + 0.5) * src_h / height, 0, src_h - 1).astype(int)
        xi = np.clip((np.arange(width) + 0.5) * src_w / width, 0, src_w - 1).astype(int)
        z = d[np.ix_(yi, xi)]

        valid = np.isfinite(z) & (z > 0.0)
        clipped = 0
        if far_clip_m > 0.0:
            beyond = valid & (z >= float(far_clip_m))
            clipped = int(beyond.sum())
            valid &= ~beyond
        z = np.where(valid, z, np.nan)

        depth_t = torch.from_numpy(np.repeat(z[None].astype(np.float32), int(frames), axis=0))
        mask_t = torch.from_numpy(np.repeat(valid[None], int(frames), axis=0))

        # Normalised by width, which is how the warp reads it back:
        # `fx = float(K[0][0, 0]) * W`.
        hfov = _hfov_deg(focal_mm, sensor_width_mm)
        fx_px = width / (2.0 * np.tan(np.radians(hfov) / 2.0))
        k = np.eye(3, dtype=np.float32)
        k[0, 0] = fx_px / width
        k[1, 1] = fx_px / height
        k[0, 2] = 0.5
        k[1, 2] = 0.5
        intrinsics = torch.from_numpy(np.repeat(k[None], int(frames), axis=0))

        finite = z[np.isfinite(z)]
        near = float(finite.min()) if finite.size else float("nan")
        far = float(finite.max()) if finite.size else float("nan")
        src_aspect, dst_aspect = src_w / src_h, width / height
        lines = [
            "Atlas Unreal depth — rendered, not inferred",
            f"  source      {src.name}, channel {channel}",
            f"  raster      {width}x{height} from {src_w}x{src_h}",
            f"  depth       {near:.2f}m to {far:.2f}m (metric, Unreal world)",
            f"  covered     {valid.mean() * 100:.1f}% of frame; the rest is NaN and "
            f"renders as magenta",
            f"  lens        {focal_mm:.1f}mm on {sensor_width_mm:.1f}mm "
            f"= {hfov:.2f} deg horizontal, fx {fx_px:.1f}px",
            f"  frames      {frames}",
        ]
        if clipped:
            lines.append(
                f"  far_clip    {clipped} px at or beyond {far_clip_m:.0f}m masked out")
        if abs(src_aspect - dst_aspect) > 1e-3:
            lines.append(
                f"  MISMATCH    EXR is {src_aspect:.4f}, raster is {dst_aspect:.4f}. "
                "The depth was rendered for a different frame shape than the one being "
                "warped, so it does not line up with the image."
            )
        lines += [
            "",
            "Depth and camera come from one render, so a metric camera_path cannot "
            "disagree with the geometry it moves through — provided this EXR was "
            "rendered through the camera that made the source frame.",
        ]
        return ({"depth": depth_t, "mask": mask_t, "intrinsics": intrinsics},
                "\n".join(lines))


def _hfov_deg(focal_mm: float, sensor_width_mm: float) -> float:
    """Horizontal field of view, in degrees, from a lens and a filmback."""
    import numpy as np

    return float(np.degrees(
        2.0 * np.arctan(float(sensor_width_mm) / (2.0 * float(focal_mm)))))


class AtlasAnchorDepth:
    """🪢 A per-frame depth estimate, put on the render's scale.

    Neither depth source covers a dressed shot on its own, and the split is not
    a matter of quality. A rendered depth pass is exact for everything the build
    contains and has no opinion at all about anything it does not — a figure the
    generator invented, walking, is simply absent from it, so a still render
    repeated across a clip warps that figure with the depth of whatever it has
    walked away from. A monocular estimate tracks the figure because it looks at
    each frame, and pays for that by re-deriving scale and focal length every
    time, which is what puts a move at the wrong depth with nothing raising.

    This takes the tracking from one and the scale from the other. The estimate
    supplies per-frame depth and its own mask; the render supplies the scale,
    the shift and the intrinsics. A single least-squares fit of `z_render ≈
    s·z_est + t` on the frame the two share, trimmed once against its own
    residual, carries onto every frame of the estimate.

    The trim is what makes it work rather than a refinement. The two sources
    disagree in exactly two places: where the render is missing something the
    estimate can see (the invented figures), and where the render is a backplate
    card standing in for distance. Both are minorities of the frame and both are
    large residuals, so trimming removes them from the FIT while leaving them in
    the OUTPUT — which is the whole point, since those pixels are the reason the
    estimate is here.

    `s` and `t` are reported, not hidden. A scale far from 1 means the estimate
    disagreed with the render about the size of the scene, and that number is
    the honest measure of how much a run driven by the estimate alone was out.
    """

    RETURN_TYPES = ("MOGE_GEOMETRY", "STRING")
    RETURN_NAMES = ("moge_geometry", "report")
    FUNCTION = "anchor"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "estimated": ("MOGE_GEOMETRY", {
                    "tooltip": "Per-frame geometry from Run MoGe Inference. Supplies the "
                               "depth that tracks what moves, and its own mask."}),
                "anchor": ("MOGE_GEOMETRY", {
                    "tooltip": "Rendered geometry — Atlas Unreal Depth Geometry, or the "
                               "relief. Supplies scale, shift and the true intrinsics; "
                               "its depth is not carried through."}),
            },
            "optional": {
                "anchor_frame": ("INT", {"default": 0, "min": 0, "max": 4096,
                    "tooltip": "Which frame of the anchor to fit against. A still render "
                               "repeated across the clip has only frame 0."}),
                "estimated_frame": ("INT", {"default": 0, "min": 0, "max": 4096,
                    "tooltip": "The frame of the estimate that lines up with it. Normally "
                               "0: the frame the clip and the render share."}),
                "fit_space": (["depth", "disparity"], {"default": "depth",
                    "tooltip": "Where the fit is solved. DEPTH by measurement, not by "
                               "theory: disparity (1/z) is the textbook alignment for a "
                               "RELATIVE estimate, but MoGe v2 returns metric depth, so a "
                               "linear relation in metres holds and fits better. Measured "
                               "against a rendered pass on a 5m-1.2km coastal scene, "
                               "depth-space median error was 3.6% mid-field and 9.1% far "
                               "against disparity's 8.2% and 25.6%; disparity won only "
                               "under 15m, 6.6% to 7.5%. Try disparity if your estimator "
                               "is relative rather than metric."}),
                "fit_shift": ("BOOLEAN", {"default": True,
                    "tooltip": "Fit an offset as well as a scale. Off forces the fit "
                               "through the origin, which is right when the estimate is "
                               "truly metric and wrong when it carries a near-plane bias."}),
                "trim_sigma": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 20.0, "step": 0.5,
                    "tooltip": "Residuals beyond this many robust deviations are dropped "
                               "and the fit repeated once. Lower excludes more of the "
                               "disagreement; 0.5-2 if invented content fills the frame."}),
            },
        }

    def anchor(self, estimated, anchor, anchor_frame: int = 0, estimated_frame: int = 0,
               fit_space: str = "disparity", fit_shift: bool = True, trim_sigma: float = 3.0):
        import numpy as np
        import torch

        def _np(x):
            return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

        est_d = _np(estimated.get("depth"))
        anc_d = _np(anchor.get("depth"))
        if est_d is None or anc_d is None:
            raise ValueError("Atlas anchor depth: both inputs need a `depth` array.")
        if est_d.ndim == 2:
            est_d = est_d[None]
        if anc_d.ndim == 2:
            anc_d = anc_d[None]
        if est_d.shape[1:] != anc_d.shape[1:]:
            raise ValueError(
                f"Atlas anchor depth: the estimate is {est_d.shape[2]}x{est_d.shape[1]} and "
                f"the anchor is {anc_d.shape[2]}x{anc_d.shape[1]}. Fitting one onto the other "
                "pixel by pixel needs them rasterised at the same size."
            )
        ai = min(int(anchor_frame), anc_d.shape[0] - 1)
        ei = min(int(estimated_frame), est_d.shape[0] - 1)

        def _mask(g, arr, i):
            m = g.get("mask")
            m = _np(m).astype(bool) if m is not None else np.ones(arr.shape, bool)
            if m.ndim == 2:
                m = m[None]
            return m[min(i, m.shape[0] - 1)]

        e0, a0 = est_d[ei], anc_d[ai]
        both = (_mask(estimated, est_d, ei) & _mask(anchor, anc_d, ai)
                & np.isfinite(e0) & np.isfinite(a0) & (e0 > 0) & (a0 > 0))
        if int(both.sum()) < 64:
            raise ValueError(
                "Atlas anchor depth: the two sources overlap on fewer than 64 valid pixels, "
                "which is not enough to fit anything. Check they are the same view."
            )

        disparity = str(fit_space) == "disparity"

        def _to(v):
            """Into the space the fit is solved in."""
            return 1.0 / np.maximum(v, 1e-6) if disparity else v

        def _from(v):
            """And back to metres."""
            return 1.0 / np.maximum(v, 1e-9) if disparity else v

        def _fit(sel):
            x, y = _to(e0[sel]), _to(a0[sel])
            if fit_shift:
                A = np.stack([x, np.ones_like(x)], 1)
                s, t = np.linalg.lstsq(A, y, rcond=None)[0]
            else:
                s, t = float((x * y).sum() / max((x * x).sum(), 1e-12)), 0.0
            return float(s), float(t)

        def _apply(v):
            return _from(s * _to(v) + t)

        s, t = _fit(both)
        resid = np.abs(_apply(e0) - a0)
        mad = float(np.median(resid[both])) or 1e-6
        keep = both & (resid <= trim_sigma * 1.4826 * mad)
        trimmed = int(both.sum() - keep.sum())
        if int(keep.sum()) >= 64:
            s, t = _fit(keep)
            resid = np.abs(_apply(e0) - a0)

        out = _apply(est_d)
        out = np.where(np.isfinite(out) & (out > 0), out, np.nan).astype(np.float32)
        mask_src = estimated.get("mask")
        mask_t = (torch.as_tensor(_np(mask_src)).bool() if mask_src is not None
                  else torch.from_numpy(np.isfinite(out)))
        # The anchor's intrinsics, always: a recovered focal is the other half of
        # what the estimate gets wrong, and the render's is measured.
        K = anchor.get("intrinsics")
        if K is None:
            raise ValueError("Atlas anchor depth: the anchor carries no intrinsics to adopt.")
        K = torch.as_tensor(_np(K)).float()
        if K.ndim == 2:
            K = K[None]
        if K.shape[0] != out.shape[0]:
            K = K[:1].repeat(out.shape[0], 1, 1)

        keep_r = resid[keep] if int(keep.sum()) else resid[both]
        before = float(np.nanmedian(e0[both]))
        after = float(np.nanmedian(_apply(e0)[both]))
        target = float(np.nanmedian(a0[both]))
        lines = [
            "Atlas anchor depth — the estimate's tracking, the render's scale",
            (f"  fit         1/z_render = {s:.4f} x 1/z_est {t:+.6f}   (disparity)"
             if disparity else
             f"  fit         z_render = {s:.4f} x z_est {t:+.3f} m   (depth)")
            + ("" if fit_shift else "   [shift held at 0]"),
            f"  fitted on   {int(keep.sum()):,} px "
            f"({100.0 * keep.sum() / keep.size:.1f}% of frame), "
            f"{trimmed:,} trimmed beyond {trim_sigma:.1f} sigma",
            f"  residual    median {np.median(keep_r):.3f} m, p95 {np.percentile(keep_r, 95):.3f} m",
            f"  median depth  estimate {before:.2f} m -> {after:.2f} m, render says {target:.2f} m",
            f"  frames      {out.shape[0]} (estimate), anchored to frame {ai} of the render",
            f"  intrinsics  taken from the anchor, not the estimate",
        ]
        off = abs(after - before) / max(before, 1e-6) * 100.0
        if off > 5.0:
            lines.append(
                f"  NOTE        the estimate sat {off:.0f}% off the render at the median. "
                "A run driven by it alone put the move at that error."
            )
        lines += [
            "",
            "The trimmed pixels stay in the output. They are where the render has nothing "
            "to say — invented content, and any backplate standing in for distance — which "
            "is the reason the estimate is in this graph at all.",
        ]
        return ({"depth": torch.from_numpy(out), "mask": mask_t, "intrinsics": K},
                "\n".join(lines))
