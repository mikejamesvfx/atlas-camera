"""Hidden-surface depth, computed rather than predicted.

A silhouette tear is a region where the camera ray hit an occluder and the
surface behind it was never photographed. Layered-ray models (LaRI and
relatives) answer that by predicting several ray intersections per pixel; Atlas
consumes such a stack in :mod:`atlas_camera.core.hidden_geometry`. This module
answers the same question **deterministically**, for the cases where the answer
is already implied by geometry Atlas has fitted.

The insight is that the solve's own back-projection

    x = (u - cx) / fx * depth        y = -(v - cy) / fy * depth

is a ray parametrisation: the pixel fixes a ray, depth picks a point on it.
Hidden geometry is picking a *different* point on the same ray. Where the
occlusion graph says a fitted plane lies behind the occluder, that point is the
exact ray-plane intersection — not an estimate, not an interpolation:

    t = (D - n·cam) / (n·dir)        P = cam + t·dir

That covers background continuation, which is the case that actually produces
visible tearing under a modest camera move. Two weaker tiers handle the rest:
first-order tangential extension using surface normals, then isotropic
diffusion. Every pixel records WHICH tier produced it, because a ray-plane
result and a diffusion guess deserve very different trust.

**Position in the pipeline.** This runs BEFORE the relief mesh is built. Repair
the depth map and a mesh with no holes follows by construction; repair the mesh
and you are reconstructing information that was discarded a step earlier.
Tearing then becomes purely the deliberate ``depth_edge_rel`` decision it is
supposed to be, rather than a consequence of missing data.

**Nothing here overwrites measured depth.** Completion only ever writes where
the input was invalid or explicitly marked as a tear, and the returned
``synthesized_mask`` / ``method_map`` make every invented pixel identifiable
downstream — the same discipline as ``ReliefMesh.filled_mask``.

Output deliberately matches :func:`hidden_geometry.select_hidden_surface`'s
shape — ``(hidden_depth, hidden_valid, stats)`` in the pipeline's depth units —
so this module and a layered model are interchangeable producers for the same
consumers, and a licensable layered model could be added later without
reworking anything downstream.

Numpy-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Per-pixel provenance. Ordered best-to-worst; the numbering is stored in
# `method_map`, so these are effectively APPEND-ONLY once serialized.
METHOD_NONE = 0
METHOD_MEASURED = 1
METHOD_RAY_PLANE = 2
METHOD_TANGENT = 3
METHOD_DIFFUSION = 4

METHOD_NAMES = {
    METHOD_NONE: "none",
    METHOD_MEASURED: "measured",
    METHOD_RAY_PLANE: "ray_plane",
    METHOD_TANGENT: "tangent",
    METHOD_DIFFUSION: "diffusion",
}

# How much each tier is trusted, used to weight the reported confidence.
# Ray-plane is exact given the plane fit, so it inherits the fit's own
# confidence rather than being discounted further.
_METHOD_TRUST = {
    METHOD_MEASURED: 1.0,
    METHOD_RAY_PLANE: 0.9,
    METHOD_TANGENT: 0.5,
    METHOD_DIFFUSION: 0.2,
}


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Depth completion requires numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


@dataclass(slots=True)
class DepthCompletion:
    """A repaired depth map plus a full account of what was invented."""

    depth: Any                       # (H, W) float64, pipeline units
    synthesized_mask: Any            # (H, W) bool — True where invented
    method_map: Any                  # (H, W) uint8 — one of METHOD_*
    stats: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def synthesized_fraction(self) -> float:
        return float(self.synthesized_mask.mean()) if self.synthesized_mask.size else 0.0

    def method_histogram(self) -> dict[str, int]:
        np = _require_numpy()
        counts = np.bincount(self.method_map.ravel(), minlength=len(METHOD_NAMES))
        return {name: int(counts[code]) for code, name in METHOD_NAMES.items()
                if code < len(counts) and counts[code]}

    def confidence(self) -> float:
        """Trust-weighted mean over the frame.

        A frame that is 90% measured and 10% ray-plane reads near 1.0; one
        rescued mostly by diffusion reads low, which is the honest answer and
        the number ``scene_health`` should degrade a verdict against.
        """
        np = _require_numpy()
        weights = np.zeros(self.method_map.shape, dtype=np.float64)
        for code, trust in _METHOD_TRUST.items():
            weights[self.method_map == code] = trust
        return float(weights.mean()) if weights.size else 0.0

    def describe(self) -> str:
        parts = [f"{k}={v}" for k, v in self.method_histogram().items()]
        lines = [
            f"Depth completion: {self.synthesized_fraction:.1%} of frame synthesized, "
            f"confidence {self.confidence():.2f}",
            f"  pixels by method: {', '.join(parts) if parts else 'none'}",
        ]
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)

    def as_hidden_surface(self) -> tuple[Any, Any, dict[str, Any]]:
        """``(hidden_depth, hidden_valid, stats)`` — the layered-model contract.

        Lets a caller treat this module and
        :func:`hidden_geometry.select_hidden_surface` as interchangeable
        producers of hidden-surface depth.
        """
        np = _require_numpy()
        hidden = np.where(self.synthesized_mask, self.depth, 0.0)
        stats = dict(self.stats)
        stats.update(coverage=self.synthesized_fraction,
                     n_hidden_pixels=int(self.synthesized_mask.sum()),
                     confidence=self.confidence(),
                     source="depth_completion")
        return hidden, self.synthesized_mask.copy(), stats


def pixel_rays(np: Any, height: int, width: int, *, view_matrix: Any,
               fx: float, fy: float, cx: float, cy: float) -> tuple[Any, Any]:
    """World-space ``(camera_position, directions)`` for every pixel.

    Directions are the exact rays the solve's back-projection uses, so a point
    at distance ``t`` along ray ``(v, u)`` reprojects to pixel ``(v, u)`` by
    construction. Uses the full 4x4 view matrix and its inverse — never the 3x3
    rotation, whose transpose ambiguity is the standing trap in this codebase.
    """
    vm = np.asarray(view_matrix, dtype=np.float64)
    cam_to_world = np.linalg.inv(vm)
    R_cw = cam_to_world[:3, :3]
    cam_pos = cam_to_world[:3, 3]

    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    dirs_cam = np.stack([(uu - cx) / fx, -(vv - cy) / fy,
                         -np.ones_like(uu)], axis=-1)
    dirs_world = dirs_cam @ R_cw.T
    return cam_pos, dirs_world


def ray_plane_depth(np: Any, cam_pos: Any, dirs: Any, normal: Any,
                    d: float, *, min_depth: float = 1e-3) -> tuple[Any, Any]:
    """Forward depth where each ray meets the plane ``n·x = d``.

    Returns ``(depth, valid)``. A ray is invalid when it is parallel to the
    plane, or meets it behind the camera — both mean this plane genuinely does
    not explain that pixel, and inventing a value would be worse than leaving
    the hole for a later tier.
    """
    normal = np.asarray(normal, dtype=np.float64)
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-12:
        shape = dirs.shape[:2]
        return np.zeros(shape), np.zeros(shape, dtype=bool)
    normal = normal / nrm
    d = float(d) / nrm

    denom = dirs @ normal
    parallel = np.abs(denom) < 1e-9
    numer = d - float(np.dot(normal, cam_pos))
    t = np.divide(numer, np.where(parallel, 1.0, denom))

    # `dirs` has -1 in camera z, so |t| scaled by the ray's forward component
    # gives forward distance; t > 0 means in front of the camera.
    forward = t
    valid = (~parallel) & np.isfinite(forward) & (forward > min_depth)
    return np.where(valid, forward, 0.0), valid


def complete_depth(
    depth: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    holes: Any = None,
    planes: list[dict[str, Any]] | None = None,
    normals: Any = None,
    use_diffusion: bool = True,
    diffusion_iterations: int = 64,
    max_plane_depth_ratio: float = 50.0,
    grazing_min_cos: float = 0.087,   # ~85 degrees incidence
) -> DepthCompletion:
    """Fill every hole in ``depth``, best tier first, recording which was used.

    ``holes`` (H, W) bool, optional — pixels to complete. Defaults to wherever
    ``depth`` is non-finite or non-positive.

    ``planes`` is a list of ``{"normal": [x,y,z], "d": float}`` in world space,
    ordered NEAREST-FIRST in the sense that earlier entries win. The occlusion
    graph produces exactly this from its occludee nodes, so the caller normally
    passes ``[n.plane for n in graph.nodes if n.plane]``.

    ``normals`` (H, W, 3), optional — enables the tangential tier, a first-order
    continuation of the surface at the hole's rim. Normals carry no information
    about what is *behind* a surface (they are the visible field's derivative),
    but they do say which way it was heading, which beats extending it flat.

    Tiers are applied in order and never overwrite an earlier, better one.
    """
    np = _require_numpy()
    depth = np.array(depth, dtype=np.float64, copy=True)
    if depth.ndim != 2:
        raise ValueError("complete_depth expects a 2D (H, W) depth map")
    height, width = depth.shape

    measured = np.isfinite(depth) & (depth > 1e-6)
    if holes is None:
        holes = ~measured
    else:
        holes = np.asarray(holes, dtype=bool) | ~measured

    method_map = np.where(measured & ~holes, METHOD_MEASURED,
                          METHOD_NONE).astype(np.uint8)
    out = np.where(measured, depth, 0.0)
    remaining = holes.copy()
    notes: list[str] = []
    stats: dict[str, Any] = {"n_holes_in": int(holes.sum())}

    if not remaining.any():
        return DepthCompletion(depth=out, synthesized_mask=np.zeros_like(holes),
                               method_map=method_map, stats=stats,
                               notes=["nothing to complete — depth had no holes."])

    cam_pos, dirs = pixel_rays(np, height, width, view_matrix=view_matrix,
                               fx=fx, fy=fy, cx=cx, cy=cy)

    # --- tier 1: exact ray-plane -------------------------------------------
    #
    # Every candidate plane is evaluated and the NEAREST valid intersection
    # wins per pixel. That is the physical answer — the closest surface behind
    # the occluder is the one you would see — and it needs no plane ordering.
    #
    # Ordering the planes globally and taking first-wins does not work, by any
    # ordering: a ground plane's perpendicular distance from the camera is just
    # the camera height, so "nearest plane" ranks it ahead of a wall it is
    # nowhere near, and it then claims near-horizon pixels at absurd distances
    # where its ray intersection is technically valid and practically useless.
    n_plane = 0
    if planes:
        scale_ref = float(np.median(depth[measured])) if measured.any() else 1.0
        limit = max_plane_depth_ratio * max(scale_ref, 1e-6)
        best = np.full(depth.shape, np.inf, dtype=np.float64)
        for plane in planes:
            normal = plane.get("normal")
            if normal is None or plane.get("d") is None:
                continue
            candidate, ok = ray_plane_depth(np, cam_pos, dirs, normal,
                                            float(plane["d"]))
            # A near-grazing hit is geometrically valid and physically
            # meaningless — the ray skims the surface, so a millimetre of fit
            # error moves the intersection by metres.
            cos_incidence = np.abs(dirs @ (np.asarray(normal, dtype=np.float64)
                                           / max(float(np.linalg.norm(normal)), 1e-12)))
            ok &= (candidate < limit) & (cos_incidence > grazing_min_cos)
            best = np.where(ok & (candidate < best), candidate, best)
        take = remaining & np.isfinite(best)
        if take.any():
            out[take] = best[take]
            method_map[take] = METHOD_RAY_PLANE
            remaining &= ~take
            n_plane = int(take.sum())
    stats["n_ray_plane"] = n_plane

    # --- tier 2: first-order tangential extension ---------------------------
    n_tangent = 0
    if normals is not None and remaining.any():
        filled, took = _tangent_extend(np, out, remaining, measured, normals,
                                       cam_pos, dirs)
        if took.any():
            out[took] = filled[took]
            method_map[took] = METHOD_TANGENT
            remaining &= ~took
            n_tangent = int(took.sum())
    stats["n_tangent"] = n_tangent

    # --- tier 3: isotropic diffusion ---------------------------------------
    n_diffusion = 0
    if use_diffusion and remaining.any():
        filled, took = _diffuse(np, out, remaining, method_map > METHOD_NONE,
                                iterations=diffusion_iterations)
        if took.any():
            out[took] = filled[took]
            method_map[took] = METHOD_DIFFUSION
            remaining &= ~took
            n_diffusion = int(took.sum())
    stats["n_diffusion"] = n_diffusion

    if remaining.any():
        notes.append(
            f"{int(remaining.sum())} pixels could not be completed by any tier "
            "and remain holes — no plane explained them and they touch no "
            "measured depth to diffuse from."
        )

    synthesized = holes & ~remaining
    return DepthCompletion(depth=out, synthesized_mask=synthesized,
                           method_map=method_map, stats=stats, notes=notes)


def _tangent_extend(np: Any, depth: Any, remaining: Any, measured: Any,
                    normals: Any, cam_pos: Any, dirs: Any) -> tuple[Any, Any]:
    """Extend the rim surface along its own tangent plane into the hole.

    Each hole pixel adjacent to measured depth adopts the local plane of its
    nearest measured neighbour — that neighbour's 3D point and normal define a
    plane, and this pixel's ray is intersected with it. That is a first-order
    continuation: flat fill assumes the surface stops, this assumes it keeps
    going the way it was going.

    One dilation ring per call keeps it local; the caller's diffusion tier
    handles anything deeper, where a first-order guess would not be credible
    anyway.
    """
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape[:2] != depth.shape or normals.shape[-1] != 3:
        return depth, np.zeros_like(remaining)

    pts = cam_pos + dirs * depth[..., None]
    took = np.zeros_like(remaining)
    filled = depth.copy()

    for shift_axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        src = np.roll(measured, shift, axis=shift_axis)
        src_pts = np.roll(pts, shift, axis=shift_axis)
        src_nrm = np.roll(normals, shift, axis=shift_axis)
        cand = remaining & src & ~took
        if not cand.any():
            continue
        n = src_nrm
        d = np.einsum("ijk,ijk->ij", n, src_pts)
        denom = np.einsum("ijk,ijk->ij", dirs, n)
        good = cand & (np.abs(denom) > 1e-9)
        t = np.divide(d - np.einsum("ijk,k->ij", n, cam_pos),
                      np.where(np.abs(denom) > 1e-9, denom, 1.0))
        good &= np.isfinite(t) & (t > 1e-3)
        if not good.any():
            continue
        filled = np.where(good, t, filled)
        took |= good
    return filled, took


def _diffuse(np: Any, depth: Any, remaining: Any, known: Any, *,
             iterations: int) -> tuple[Any, Any]:
    """Jacobi diffusion of known depth into the remaining holes.

    The last resort, and labelled as such: it produces a smooth interpolation
    of the surrounding surface, which is plausible-looking and carries no
    evidence whatsoever. Only pixels actually reached by the diffusion front
    are reported as filled, so an enclosed region with no known neighbour stays
    an honest hole.
    """
    work = np.where(known, depth, 0.0)
    reached = known.copy()
    for _ in range(max(int(iterations), 0)):
        if not (remaining & ~reached).any():
            break
        acc = np.zeros_like(work)
        cnt = np.zeros(work.shape, dtype=np.float64)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            nb_v = np.roll(reached, shift, axis=axis)
            nb = np.roll(work, shift, axis=axis)
            # Rolling wraps; a wrapped edge row must not seed the far side.
            nb_v = _clear_wrapped(np, nb_v, axis, shift)
            acc += np.where(nb_v, nb, 0.0)
            cnt += nb_v
        has = (cnt > 0) & remaining
        work = np.where(has & ~reached, acc / np.maximum(cnt, 1.0), work)
        reached |= has
    took = remaining & reached
    return work, took


def _clear_wrapped(np: Any, arr: Any, axis: int, shift: int) -> Any:
    out = arr.copy()
    if axis == 0:
        if shift > 0:
            out[0, :] = False
        else:
            out[-1, :] = False
    else:
        if shift > 0:
            out[:, 0] = False
        else:
            out[:, -1] = False
    return out


def complete_depth_from_graph(
    solve: Any,
    depth: Any,
    graph: Any,
    *,
    holes: Any = None,
    normals: Any = None,
    **kwargs: Any,
) -> DepthCompletion:
    """:func:`complete_depth` driven by an occlusion graph's fitted planes.

    Only nodes whose ``completion_policy`` actually licenses building are used.
    A node the graph marked ``none`` — an unclassifiable tear — contributes no
    plane, so the hole it guards stays open. That refusal is the whole point of
    the graph, and it has to survive into the thing that does the filling.
    """
    from atlas_camera.core.occlusion_graph import POLICY_NONE
    from atlas_camera.core.relief_mesh import ReliefMeshCameraSpec

    spec = ReliefMeshCameraSpec.from_solve(solve)

    # The backdrop is excluded for the same reason the move budget excludes it:
    # a cyclorama spans the whole frustum, so it "explains" every pixel and
    # would win tears belonging to a real surface metres in front of it.
    # Filling from it looks successful and puts the geometry at the far plane.
    # No ordering is needed beyond that — complete_depth takes the nearest
    # valid intersection per pixel.
    planes = [n.plane for n in getattr(graph, "nodes", [])
              if n.plane and n.completion_policy != POLICY_NONE
              and n.kind != "backdrop"]

    result = complete_depth(
        depth, view_matrix=spec.view_matrix, fx=spec.fx, fy=spec.fy,
        cx=spec.cx, cy=spec.cy, holes=holes, planes=planes, normals=normals,
        **kwargs,
    )
    blocked = [n.id for n in getattr(graph, "nodes", [])
               if n.completion_policy == POLICY_NONE and n.plane]
    if blocked:
        result.notes.append(
            "left unfilled by graph policy (unclassifiable tears): "
            + ", ".join(blocked)
        )
    return result
