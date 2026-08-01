"""The camera seam: one bundle of intrinsics, extrinsics and image size.

WHY THIS MODULE EXISTS. The principal-point fallback

    cx = intr.cx_px if intr.cx_px is not None else width / 2.0

was written by hand at 27 sites across 20 files — node bodies, exporters, the
UI service, the solver, tools — while the one bundle that already did it
correctly, `ReliefMeshCameraSpec`, sat inside `relief_mesh.py` where no caller
outside relief-mesh code discovered it. Every site agreed on the arithmetic and
disagreed on what `width` meant: the intrinsics' own `image_width`, a local
variable, or a depth estimate's resolution.

So the seam is here, not inside an engine, and it is where the fallback ladder
is stated once:

    cx_px  ->  principal_point_px  ->  image centre

The middle rung is not decoration. `build_intrinsics` always writes both, but a
solve rehydrated from JSON can carry `principal_point_px` without `cx_px`, and
dropping straight to the image centre there discards a measured value.

Host-agnostic by construction: no numpy, no torch, no ComfyUI. It reads a solve
and returns numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CameraSpec:
    """Camera intrinsic/extrinsic and scale bundle.

    Field order is load-bearing: `ReliefMeshCameraSpec` is an alias for this
    class and callers construct it by keyword, but the first seven fields keep
    that bundle's original order so nothing positional can break.
    """

    view_matrix: Any
    fx: float
    fy: float
    cx: float
    cy: float
    scale: float = 1.0
    horizon_y: float | None = None
    width: int = 0
    height: int = 0

    @property
    def has_focal(self) -> bool:
        """False when the solve has no usable focal length.

        Callers that previously returned `None` from `_solve_camera_params` and
        reported ``SKIPPED — solve has no usable focal`` branch on this.
        """
        return self.fx > 0

    def as_params(self) -> tuple[int, int, float, float, float, float]:
        """`(width, height, fx, fy, cx, cy)` — the tuple order the 15 hand-rolled
        unpack sites in `comfy/` already destructure."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                "CameraSpec has no image size; build it with from_solve(), or "
                "pass width=/height= when the intrinsics carry none")
        return (self.width, self.height, self.fx, self.fy, self.cx, self.cy)

    @classmethod
    def from_intrinsics(cls, intrinsics: Any, *, view_matrix: Any = None,
                        width: int | None = None, height: int | None = None,
                        scale: float = 1.0,
                        horizon_y: float | None = None) -> CameraSpec:
        """Build from an `AtlasIntrinsics`, resolving the fallback ladder.

        `width`/`height` stand in when the intrinsics carry no image size — the
        ATLAS_DEPTH_MAP path, where the depth estimate's own resolution is the
        authority.
        """
        w = int(intrinsics.image_width or (width or 0))
        h = int(intrinsics.image_height or (height or 0))
        fx = float(intrinsics.fx_px or 0.0)
        fy = float(intrinsics.fy_px or fx)
        pp = getattr(intrinsics, "principal_point_px", None)
        if intrinsics.cx_px is not None:
            cx = float(intrinsics.cx_px)
        elif pp is not None:
            cx = float(pp[0])
        else:
            cx = w / 2.0
        if intrinsics.cy_px is not None:
            cy = float(intrinsics.cy_px)
        elif pp is not None:
            cy = float(pp[1])
        else:
            cy = h / 2.0
        return cls(view_matrix=view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
                   scale=float(scale), horizon_y=horizon_y, width=w, height=h)

    @classmethod
    def for_image(cls, intrinsics: Any, width: int, height: int, *,
                  view_matrix: Any = None, scale: float = 1.0,
                  horizon_y: float | None = None) -> CameraSpec:
        """Build for an image of a KNOWN size, overriding the intrinsics'.

        The distinction from `from_intrinsics` is the fallback base. A caller
        holding a depth array at a different resolution than the recorded plate
        falls back to the centre of the array it has, not of the plate. Both
        behaviours existed by hand at the 27 sites; naming them is what stops
        the next reader guessing which one a site meant.

        `principal_point_px` is deliberately NOT consulted here: it is recorded
        in plate pixels, and this call is being made about a different raster.
        """
        w, h = int(width), int(height)
        fx = float(intrinsics.fx_px or 0.0)
        fy = float(intrinsics.fy_px or fx)
        cx = float(intrinsics.cx_px) if intrinsics.cx_px is not None else w / 2.0
        cy = float(intrinsics.cy_px) if intrinsics.cy_px is not None else h / 2.0
        return cls(view_matrix=view_matrix, fx=fx, fy=fy, cx=cx, cy=cy,
                   scale=float(scale), horizon_y=horizon_y, width=w, height=h)

    @classmethod
    def from_solve(cls, solve: Any, scale: float = 1.0, *,
                   width: int | None = None,
                   height: int | None = None) -> CameraSpec:
        """Build from an ATLAS_SOLVE, including the solved horizon row.

        `scale` stays positional: `ReliefMeshCameraSpec.from_solve(solve, 2.0)`
        was already a supported call.
        """
        return cls.from_intrinsics(
            solve.camera.intrinsics,
            view_matrix=solve.camera.extrinsics.camera_view_matrix,
            width=width,
            height=height,
            scale=scale,
            horizon_y=horizon_row(solve),
        )


def horizon_row(solve: Any) -> float | None:
    """Image row of the solved horizon, or None."""
    if solve.horizon_line and solve.horizon_line.endpoints_px:
        p1, p2 = solve.horizon_line.endpoints_px
        return 0.5 * (float(p1[1]) + float(p2[1]))
    return None
