"""Per-frame conditioning signals along a camera path, from measured geometry.

WHY THIS MODULE EXISTS. A generative video model can be pointed at a camera
move in two ways. The semantic way names the move ("slow dolly in, 35 mm") and
hopes the model's training data supplies the rest. The geometric way hands it
the answer: this is the depth, this is the surface orientation, this is exactly
how far every pixel travels, and these are the only pixels you are allowed to
invent because they are the only ones the photograph never saw. Atlas recovers a
real projection from a single still, so it can do the second -- and every signal
here is READ OFF THE SAME Z-BUFFER that chose the render's colours, so depth,
position, normals and flow cannot disagree with each other or with the RGB.

FLOW IS DERIVED, NOT ESTIMATED. Given depth and two calibrated cameras, the
displacement of every pixel is closed-form; running a learned flow network over
rendered frames would be estimating a quantity we already know exactly, and
would weaken the whole claim. The occlusion flag that comes with it is likewise
better than a learned flow's forward/backward consistency check, because it does
not infer which pixels became occluded -- it tests the target frame's own depth
and KNOWS.

``core.reprojection.reproject_at_infinity`` is NOT flow and must never be used
as it: it back-projects at infinity, where the inter-camera translation cancels
by construction. That cancellation is exactly right for the sky test it performs
and exactly wrong for a motion field.

Host-agnostic: numpy only, no torch, no ComfyUI, no Pillow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: A point is occluded in the target frame when that frame's own visible
#: surface sits closer than the point by more than this RELATIVE margin. The
#: test is directional -- an occluder must be IN FRONT -- so a slightly noisy
#: z-buffer cannot manufacture occlusion behind the surface it belongs to.
DEFAULT_OCCLUSION_TOL = 0.02

#: Below this median displacement a frame carries no usable parallax. Sized to
#: match the scorer's default ``min_parallax_px``; a zoom sits here by
#: definition (a focal change moves pixels without moving the eye, so it
#: reveals nothing).
PARALLAX_FLOOR_PX = 1.5


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded import
        raise RuntimeError(
            "atlas_camera.core.conditioning requires numpy. Install with: "
            "pip install -e .[vision]") from exc
    return np


@dataclass
class ConditioningSequence:
    """Per-frame conditioning, stacked on axis 0. ``None`` where not requested.

    Rasters are (F, H, W[, C]) float32 unless stated. Holes are NaN in the
    metric fields, never 0 and never a sentinel distance: zero is a legitimate
    coordinate and a large number is a legitimate depth, so either would be
    indistinguishable from data downstream.
    """

    views: list = field(default_factory=list)          # F x 4x4, as given
    k: list = field(default_factory=list)              # F x 3x3, per frame
    rgb: Any = None                                    # (F,H,W,3) in [0,1]
    alpha: Any = None                                  # (F,H,W)
    depth_m: Any = None                                # (F,H,W) m, NaN holes
    class_map: Any = None                              # (F,H,W) uint8 GhostClass
    normal_world: Any = None                           # (F,H,W,3) unit, NaN holes
    normal_valid: Any = None                           # (F,H,W) bool
    position_world: Any = None                         # (F,H,W,3) m, NaN holes
    flow_fwd: Any = None                               # (F,H,W,2) px, -> i+1
    flow_bwd: Any = None                               # (F,H,W,2) px, -> i-1
    flow_valid_fwd: Any = None                          # (F,H,W) bool
    flow_valid_bwd: Any = None                          # (F,H,W) bool
    flow_occluded_fwd: Any = None                       # (F,H,W) bool
    flow_occluded_bwd: Any = None                       # (F,H,W) bool
    per_frame: list = field(default_factory=list)      # ghost_stats per frame
    meta: dict = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return len(self.views)

    def parallax_px(self) -> list[float]:
        """Median |flow_fwd| over valid, unoccluded pixels, per frame.

        The scorer's parallax precondition reads this. A frame below
        ``PARALLAX_FLOOR_PX`` is reported but must not enter an adherence
        aggregate: with no parallax there is nothing for a camera to get right.
        """
        np = _require_numpy()
        if self.flow_fwd is None:
            return [0.0] * self.frames
        out: list[float] = []
        for i in range(self.frames):
            sel = self.flow_valid_fwd[i] & ~self.flow_occluded_fwd[i]
            if not sel.any():
                out.append(0.0)
                continue
            mag = np.linalg.norm(self.flow_fwd[i][sel], axis=-1)
            out.append(float(np.median(mag)))
        return out


def _clean_depth(np: Any, depth: Any) -> Any:
    """+inf (nothing rasterized) -> NaN, so arithmetic cannot launder a hole."""
    d = np.asarray(depth, dtype=np.float64)
    return np.where(np.isfinite(d) & (d > 1e-6), d, np.nan)


def _pixel_grid(np: Any, width: int, height: int) -> Any:
    """Integer pixel centres, matching ``back_project_normals``' own meshgrid."""
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    return np.stack([uu, vv], axis=-1)


def _flow_between(np: Any, *, depth_a: Any, view_a: Any, k_a: Any,
                  depth_b: Any, view_b: Any, k_b: Any,
                  occlusion_tol: float) -> tuple:
    """Pixel displacement a -> b for every pixel of a, plus validity/occlusion.

    Closed form: back-project a's depth to world, project into b, subtract the
    original pixel. Exact for any pair of calibrated cameras, including a focal
    change (a zoom produces purely radial flow with a stationary eye -- which is
    also the proof that a zoom cannot disocclude).
    """
    from atlas_camera.core.depth_geometry import back_project_normals
    from atlas_camera.core.projection_render import project_points

    height, width = depth_a.shape
    fx_a, fy_a, cx_a, cy_a = k_a[0][0], k_a[1][1], k_a[0][2], k_a[1][2]
    fx_b, fy_b, cx_b, cy_b = k_b[0][0], k_b[1][1], k_b[0][2], k_b[1][2]

    bp = back_project_normals(depth_a, view_matrix=view_a,
                              fx=fx_a, fy=fy_a, cx=cx_a, cy=cy_a)
    px_b, fwd_b = project_points(bp.pts_world.reshape(-1, 3), view_b,
                                 fx_b, fy_b, cx_b, cy_b)
    px_b = px_b.reshape(height, width, 2)
    fwd_b = fwd_b.reshape(height, width)

    grid = _pixel_grid(np, width, height)
    flow = px_b - grid

    valid = (np.isfinite(depth_a) & np.isfinite(px_b).all(axis=-1)
             & (fwd_b > 1e-6)
             & (px_b[..., 0] >= 0) & (px_b[..., 0] <= width - 1)
             & (px_b[..., 1] >= 0) & (px_b[..., 1] <= height - 1))

    # Occlusion: does b's OWN visible surface sit in front of where this point
    # landed? Nearest-neighbour is the right sampler -- an interpolated depth
    # across a silhouette is a value no surface has.
    ui = np.clip(np.rint(np.nan_to_num(px_b[..., 0])), 0,
                 width - 1).astype(np.int64)
    vi = np.clip(np.rint(np.nan_to_num(px_b[..., 1])), 0,
                 height - 1).astype(np.int64)
    d_at = depth_b[vi, ui]
    occluded = valid & np.isfinite(d_at) & (d_at < fwd_b * (1.0 - occlusion_tol))

    flow = np.where(valid[..., None], flow, np.nan)
    return flow.astype(np.float32), valid, occluded


def render_conditioning_sequence(
    meshes: list,
    textures: dict,
    *,
    views: list,
    intrinsics: list,
    plate_view: Any,
    plate_k: Any,
    width: int,
    height: int,
    exclude: Any = None,
    hole_dilate_px: int = 0,
    occlusion_tol: float = DEFAULT_OCCLUSION_TOL,
    want_flow: bool = True,
    want_normals: bool = True,
    want_position: bool = True,
) -> ConditioningSequence:
    """Render every view once and read every conditioning signal off it.

    ``intrinsics`` is one 3x3 K per view (see
    ``camera_path.sample_camera_path_intrinsics``); ``plate_view`` / ``plate_k``
    are the SOLVED camera, rendered at the same raster as the targets so the two
    can never disagree about where a hole is. ``exclude`` is an optional (H, W)
    boolean artist declaration, applied unchanged to every frame.

    The pass structure is forced by the occlusion test: flow out of frame i
    needs frame j's depth, so all frames are rendered first and flow is a
    second, much cheaper pass over the stored z-buffers.
    """
    np = _require_numpy()
    from atlas_camera.core.depth_geometry import back_project_normals
    from atlas_camera.core.ghost_pixels import (
        classify_target_coverage, ghost_stats,
    )
    from atlas_camera.core.mask_ops import dilate
    from atlas_camera.core.projection_render import render_scene
    from atlas_camera.core.reprojection import reproject_at_infinity

    if len(intrinsics) != len(views):
        raise ValueError(
            f"intrinsics carries {len(intrinsics)} entries for {len(views)} "
            "views; sample one K per view (a zoom needs its own focal per "
            "frame, and a silent fallback to the solved focal would render a "
            "zoom as no change at all)")
    width, height = int(width), int(height)

    p_fx, p_fy = float(plate_k[0][0]), float(plate_k[1][1])
    p_cx, p_cy = float(plate_k[0][2]), float(plate_k[1][2])
    _, base_alpha, _ = render_scene(meshes, textures, plate_view,
                                    p_fx, p_fy, p_cx, p_cy, width, height)
    plate = {"mask": base_alpha <= 0.0, "view": plate_view,
             "fx": p_fx, "fy": p_fy, "cx": p_cx, "cy": p_cy,
             "height": height, "width": width}

    if exclude is not None:
        exclude = np.asarray(exclude, dtype=bool)
        if exclude.shape != (height, width):
            raise ValueError(
                f"exclude shape {exclude.shape} is not the render raster "
                f"{(height, width)}")

    rgbs, alphas, depths, classes, stats_out = [], [], [], [], []
    normals, normal_valids, positions = [], [], []
    skipped: set = set()

    for index, view in enumerate(views):
        k = intrinsics[index]
        fx, fy = float(k[0][0]), float(k[1][1])
        cx, cy = float(k[0][2]), float(k[1][2])

        rgb, alpha, stats = render_scene(meshes, textures, view,
                                         fx, fy, cx, cy, width, height)
        skipped.update(stats["meshes_skipped"])
        depth = _clean_depth(np, stats["depth"])

        covered = alpha > 0.0
        if int(hole_dilate_px) > 0:
            # Shrink the HOLE, i.e. grow coverage's complement then invert --
            # the same direction AtlasGhostPixelMap dilates in, so the two
            # classify identically.
            covered = ~dilate(~covered, iterations=int(hole_dilate_px),
                              connectivity=8)

        r = reproject_at_infinity(plate, view=view, fx=fx, fy=fy, cx=cx, cy=cy,
                                  width=width, height=height)
        inside = r["inside"]
        ui = np.where(inside, r["u"], 0).astype(np.int64)
        vi = np.where(inside, r["v"], 0).astype(np.int64)
        plate_hole_at_target = inside & plate["mask"][vi, ui]

        class_map = classify_target_coverage(
            covered=covered, inside_plate_frame=inside,
            plate_hole_at_target=plate_hole_at_target, exclude=exclude)

        rgbs.append(np.clip(rgb, 0.0, 1.0).astype(np.float32))
        alphas.append(np.asarray(alpha, dtype=np.float32))
        depths.append(depth)
        classes.append(class_map)
        stats_out.append(ghost_stats(class_map))

        if want_normals or want_position:
            bp = back_project_normals(depth, view_matrix=view,
                                      fx=fx, fy=fy, cx=cx, cy=cy)
            hole = ~np.isfinite(depth)
            if want_position:
                pos = np.where(hole[..., None], np.nan, bp.pts_world)
                positions.append(pos.astype(np.float32))
            if want_normals:
                nrm = np.where(bp.valid_normal[..., None], bp.normals, np.nan)
                normals.append(nrm.astype(np.float32))
                normal_valids.append(np.asarray(bp.valid_normal, dtype=bool))

    seq = ConditioningSequence(
        views=list(views),
        k=[[[float(v) for v in row] for row in k] for k in intrinsics],
        rgb=np.stack(rgbs) if rgbs else None,
        alpha=np.stack(alphas) if alphas else None,
        depth_m=np.stack(depths).astype(np.float32) if depths else None,
        class_map=np.stack(classes) if classes else None,
        normal_world=np.stack(normals) if normals else None,
        normal_valid=np.stack(normal_valids) if normal_valids else None,
        position_world=np.stack(positions) if positions else None,
        per_frame=stats_out,
    )

    if want_flow and depths:
        f_fwd, f_bwd, v_fwd, v_bwd, o_fwd, o_bwd = [], [], [], [], [], []
        nan2 = np.full((height, width, 2), np.nan, dtype=np.float32)
        false1 = np.zeros((height, width), dtype=bool)
        count = len(views)
        for i in range(count):
            # First and last frame have no neighbour on one side. NaN flow with
            # a False validity, never a zero vector: zero is what a static
            # pixel legitimately reads, so the two must not look alike.
            if i + 1 < count:
                fl, va, oc = _flow_between(
                    np, depth_a=depths[i], view_a=views[i], k_a=seq.k[i],
                    depth_b=depths[i + 1], view_b=views[i + 1],
                    k_b=seq.k[i + 1], occlusion_tol=occlusion_tol)
            else:
                fl, va, oc = nan2.copy(), false1.copy(), false1.copy()
            f_fwd.append(fl)
            v_fwd.append(va)
            o_fwd.append(oc)

            # Backward flow is computed from THIS frame's own depth against the
            # previous view, never by negating the forward field: the negation
            # is only correct for pixels visible in both, which is precisely the
            # set the occlusion flag exists to identify.
            if i - 1 >= 0:
                fl, va, oc = _flow_between(
                    np, depth_a=depths[i], view_a=views[i], k_a=seq.k[i],
                    depth_b=depths[i - 1], view_b=views[i - 1],
                    k_b=seq.k[i - 1], occlusion_tol=occlusion_tol)
            else:
                fl, va, oc = nan2.copy(), false1.copy(), false1.copy()
            f_bwd.append(fl)
            v_bwd.append(va)
            o_bwd.append(oc)

        seq.flow_fwd = np.stack(f_fwd)
        seq.flow_bwd = np.stack(f_bwd)
        seq.flow_valid_fwd = np.stack(v_fwd)
        seq.flow_valid_bwd = np.stack(v_bwd)
        seq.flow_occluded_fwd = np.stack(o_fwd)
        seq.flow_occluded_bwd = np.stack(o_bwd)

    parallax = seq.parallax_px() if seq.flow_fwd is not None else []
    seq.meta = {
        "raster": [width, height],
        "frames": len(views),
        "hole_dilate_px": int(hole_dilate_px),
        "occlusion_tol": float(occlusion_tol),
        "principal_point": "static",
        "flow_units": "pixels at the render raster",
        "depth_units": "metres, NaN where nothing was rasterized",
        "skipped_meshes": sorted(skipped),
        "parallax_px": parallax,
        "frames_without_parallax": [i for i, p in enumerate(parallax)
                                    if p < PARALLAX_FLOOR_PX],
    }
    return seq
