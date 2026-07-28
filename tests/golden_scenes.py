"""Deterministic scenes for the golden-frame gate.

The fixtures are CODE, not committed depth buffers. A ray-cast scene is exactly
reproducible, reviewable in a diff, and cannot drift the way a binary blob can —
and it keeps the neural half of the pipeline outside the fence entirely. No depth
model runs here, so nothing in this file depends on a GPU, a checkpoint, or a
torch version.

Three scenes, chosen because they sit in DIFFERENT tear regimes. The measured
kept-face rates are the interesting part, and they are not what you would guess:

  corridor  box occluder, mostly face-on surfaces   99.0% faces kept
  ramp      a floor viewed nearly along its plane   74.2% faces kept
  steps     hard horizontal creases                 83.1% faces kept

The ramp tears MOST, not least. A floor seen from 1.45 m looking horizontally is
extreme grazing, which is precisely the "comb-tearing a continuous surface" case
that `max_edge_factor` exists to trade against — see AtlasDeriveReliefMesh's
`normal_edge_deg` tooltip, which recommends raising mef and using normal-bend
instead. That makes `ramp` the sensitive canary: any change to tear thresholds
moves it first and by the most.

These numbers are a BASELINE, not a judgement about what tearing should be. The
gate's job is to make a change in that behaviour visible and deliberate, not to
assert that today's value is ideal.
"""
from __future__ import annotations

from typing import Any

# Camera + framing shared by every scene, so the goldens stay small and quick.
WIDTH, HEIGHT = 256, 192
FOV_DEG = 60.0
CAM_Y = 1.45          # eye height, metres
GRID = 96             # relief grid long edge


def _np():
    import numpy as np
    return np


def _rays(w: int, h: int, fov_deg: float):
    """Camera-space rays: x-right, y-up, -Z forward (Atlas + ARKit convention)."""
    np = _np()
    f = (w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    j, i = np.meshgrid(np.arange(w, dtype=np.float64),
                       np.arange(h, dtype=np.float64), indexing="xy")
    d = np.stack([(j + 0.5 - w / 2.0) / f,
                  -(i + 0.5 - h / 2.0) / f,
                  -np.ones_like(j)], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True), f


def _trace(planes, w: int = WIDTH, h: int = HEIGHT):
    """Nearest hit per ray over a list of (axis, value, predicate) planes.

    Returns (forward_z_depth HxW float32, surface_id HxW int32). Depth is
    forward-Z, which is what `build_relief_mesh` expects — not ray distance.
    """
    np = _np()
    d, _f = _rays(w, h, FOV_DEG)
    o = np.array([0.0, CAM_Y, 0.0])
    best = np.full((h, w), np.inf)
    sid = np.zeros((h, w), dtype=np.int32)

    for axis, value, ident, cond in planes:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (value - o[axis]) / d[..., axis]
        hit = np.isfinite(t) & (t > 1e-6) & (t < best)
        if cond is not None:
            p = o + d * t[..., None]
            hit &= cond(p)
        best = np.where(hit, t, best)
        sid = np.where(hit, ident, sid)

    pts = o + d * np.where(np.isfinite(best), best, 0.0)[..., None]
    fwd = np.where(np.isfinite(best), -pts[..., 2], 0.0)
    return fwd.astype(np.float32), sid


# ------------------------------------------------------------------- scenes


def scene_corridor():
    """Corridor with a box occluder — mostly face-on surfaces, so it tears LEAST.

    Measured 99.0% of grid faces kept. Exercises occlusion silhouettes, a
    z-buffer contest between the box and the wall behind it, and UV
    interpolation across a near/far boundary.
    """
    np = _np()
    WALL_X, BACK_Z, CEIL = 2.0, -9.0, 3.0
    box = {"x": (-0.7, 0.7), "y": (0.0, 1.1), "z": (-5.6, -4.4)}

    def inside(ax):
        def f(p):
            m = np.ones(p.shape[:-1], dtype=bool)
            for a, key in ((0, "x"), (1, "y"), (2, "z")):
                if a == ax:
                    continue
                m &= (p[..., a] >= box[key][0] - 1e-6) & (p[..., a] <= box[key][1] + 1e-6)
            return m
        return f

    planes = [
        (1, 0.0, 1, lambda p: (np.abs(p[..., 0]) <= WALL_X) & (p[..., 2] > BACK_Z)),
        (1, CEIL, 2, lambda p: (np.abs(p[..., 0]) <= WALL_X) & (p[..., 2] > BACK_Z)),
        (0, -WALL_X, 3, lambda p: (p[..., 1] >= 0) & (p[..., 1] <= CEIL) & (p[..., 2] > BACK_Z)),
        (0, WALL_X, 4, lambda p: (p[..., 1] >= 0) & (p[..., 1] <= CEIL) & (p[..., 2] > BACK_Z)),
        (2, BACK_Z, 5, lambda p: (np.abs(p[..., 0]) <= WALL_X) & (p[..., 1] >= 0) & (p[..., 1] <= CEIL)),
    ]
    for ax, key, ident in ((0, "x", 6), (1, "y", 7), (2, "z", 8)):
        for v in box[key]:
            planes.append((ax, v, ident, inside(ax)))
    return _trace(planes)


def scene_ramp():
    """A floor viewed almost along its own plane — the GRAZING case.

    Measured 74.2% of faces kept, the most tearing of the three, because extreme
    foreshortening stretches each quad past `max_edge_factor`. Every surface here
    is continuous, so that tearing is the documented trade rather than a defect.

    This makes it the sensitive scene: tear-threshold changes show up here first
    and largest, which is exactly what you want a canary to do.
    """
    np = _np()
    planes = [
        (1, 0.0, 1, lambda p: (np.abs(p[..., 0]) <= 6.0) & (p[..., 2] > -14.0)),
        (2, -14.0, 5, lambda p: (np.abs(p[..., 0]) <= 6.0) & (p[..., 1] >= 0) & (p[..., 1] <= 6.0)),
    ]
    return _trace(planes)


def scene_steps():
    """Three risers — alternating tread and riser, so real creases dominate.

    Measured 83.1% of faces kept, between the other two. This is the scene where
    `normal_edge_deg` (off by default) would change the picture most, since every
    tread/riser join is a genuine orientation change rather than a stretch.
    """
    np = _np()
    planes = [
        (2, -12.0, 5, lambda p: (np.abs(p[..., 0]) <= 6.0) & (p[..., 1] >= 0) & (p[..., 1] <= 6.0)),
    ]
    # tread (horizontal) + riser (vertical) per step, marching away from camera
    for k, (y, z0, z1) in enumerate([(0.0, -4.0, -2.0), (0.45, -6.0, -4.0), (0.9, -8.0, -6.0)]):
        planes.append((1, y, 10 + k,
                       lambda p, z0=z0, z1=z1: (np.abs(p[..., 0]) <= 3.0)
                       & (p[..., 2] >= z0) & (p[..., 2] <= z1)))
        planes.append((2, z0, 20 + k,
                       lambda p, y=y: (np.abs(p[..., 0]) <= 3.0)
                       & (p[..., 1] >= y) & (p[..., 1] <= y + 0.45)))
    planes.append((1, 0.0, 1, lambda p: (np.abs(p[..., 0]) <= 6.0) & (p[..., 2] > -2.0)))
    return _trace(planes)


SCENES = {
    "corridor": scene_corridor,
    "ramp": scene_ramp,
    "steps": scene_steps,
}


# ------------------------------------------------------------------ texture


def checker_texture(w: int = 128, h: int = 128):
    """Procedural checker with a colour ramp.

    Deliberately high-frequency and directional: a UV flip, a transposed axis or
    an off-by-one in the interpolation shows up immediately, where a flat or
    smooth texture would hide all three.
    """
    np = _np()
    j, i = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    check = ((i // 8 + j // 8) % 2).astype(np.float64)
    tex = np.zeros((h, w, 3), dtype=np.float64)
    tex[..., 0] = 0.25 + 0.55 * check          # red carries the checker
    tex[..., 1] = j / max(1, w - 1)            # green ramps along U
    tex[..., 2] = 1.0 - i / max(1, h - 1)      # blue ramps along V
    return tex


# ------------------------------------------------------------------- render


def render_scene_golden(name: str) -> Any:
    """Depth -> relief mesh -> rasterised RGB, as uint8 HxWx3.

    Pure numpy end to end: no depth model, no torch, no ComfyUI, no GPU. That is
    the whole point — this gate covers the geometry half of the pipeline, which
    IS reproducible, and deliberately leaves model inference outside the fence
    where a golden frame could never hold still.
    """
    np = _np()
    from atlas_camera.core.camera_math import look_at_view_matrix
    from atlas_camera.core.projection_render import render_scene
    from atlas_camera.core.relief_mesh import build_relief_mesh

    depth, _sid = SCENES[name]()
    h, w = depth.shape
    fx = fy = (w / 2.0) / float(np.tan(np.radians(FOV_DEG) / 2.0))
    cx, cy = w / 2.0, h / 2.0

    view, _world, _rot = look_at_view_matrix(
        eye=(0.0, CAM_Y, 0.0), target=(0.0, CAM_Y, -1.0), up=(0.0, 1.0, 0.0))

    mesh = build_relief_mesh(
        np.asarray(depth, dtype=np.float64),
        view_matrix=view, fx=fx, fy=fy, cx=cx, cy=cy,
        grid_long_edge=GRID,
        apply_sky_heuristic=False,   # no sky in these scenes; keep the gate on geometry
    )

    meshes = [("golden", np.asarray(mesh.vertices), np.asarray(mesh.faces),
               np.asarray(mesh.uvs), "primary", {})]
    rgb, _alpha, _stats = render_scene(
        meshes, {"primary": checker_texture()},
        view, fx, fy, cx, cy, w, h)

    return (np.clip(np.asarray(rgb), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def tear_stats(name: str) -> dict:
    """Face count kept vs. the untorn grid — the numeric side of the same claim."""
    np = _np()
    from atlas_camera.core.camera_math import look_at_view_matrix
    from atlas_camera.core.relief_mesh import build_relief_mesh

    depth, _ = SCENES[name]()
    h, w = depth.shape
    fx = fy = (w / 2.0) / float(np.tan(np.radians(FOV_DEG) / 2.0))
    view, _world, _rot = look_at_view_matrix(
        eye=(0.0, CAM_Y, 0.0), target=(0.0, CAM_Y, -1.0), up=(0.0, 1.0, 0.0))
    mesh = build_relief_mesh(
        np.asarray(depth, dtype=np.float64), view_matrix=view,
        fx=fx, fy=fy, cx=w / 2.0, cy=h / 2.0,
        grid_long_edge=GRID, apply_sky_heuristic=False)
    return dict(mesh.stats or {})
