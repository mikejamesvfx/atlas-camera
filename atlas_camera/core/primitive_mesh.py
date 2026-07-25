"""Tessellate :class:`AtlasProxyPrimitive` entries into triangle soup.

Atlas stores proxy geometry declaratively — a type, a 4x4 transform, and
dimensions — and every consumer so far has tessellated it in its own idiom:
the viewport builds three.js geometry in JS, the USD exporter defines native
``UsdGeom`` prims. Neither is reusable from Python, so anything that needs to
reason about the scene as actual triangles (the move budget's rasterizer today;
the v2 DCC scene packet later) had no route to them.

This is that route. It is deliberately minimal — the primitives Atlas actually
emits, at the lowest tessellation that is geometrically exact for planes and
boxes and visually sufficient for cylinders. It is not a modelling kernel.

Convention (matching ``proxy_geometry``): ``transform_matrix`` is row-major with
translation in column 3 and maps unit-primitive local space to world. Local
space is centred on the origin; ``dimensions`` are FULL extents, so a plane of
dimensions (w, _, d) spans local x in [-w/2, +w/2] and z in [-d/2, +d/2].
Numpy-only.
"""

from __future__ import annotations

import math
from typing import Any

# Radial segments for a cylinder. 24 keeps the silhouette smooth enough that a
# coverage rasterization does not report phantom holes along the curve, without
# making the face count meaningful next to a relief mesh.
_CYLINDER_SEGMENTS = 24


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Primitive tessellation requires numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


def _unit_plane(np: Any):
    """Unit quad in the local XY plane with +Z as the normal.

    This is the THREE.PlaneGeometry frame that ``proxy_geometry._plane_transform``
    builds against (local X=u, Y=v, Z=plane normal) — NOT an XZ ground quad.
    Tessellating a plane in the wrong local frame stands every plane in the
    scene up perpendicular to where it belongs, which reads as plausible
    geometry right up until a measurement depends on it.
    """
    verts = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0],
                      [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return verts, faces


def _unit_box(np: Any):
    verts = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ], dtype=np.float64)
    faces = np.array([
        [0, 2, 1], [0, 3, 2],      # -Z
        [4, 5, 6], [4, 6, 7],      # +Z
        [0, 4, 7], [0, 7, 3],      # -X
        [1, 2, 6], [1, 6, 5],      # +X
        [3, 7, 6], [3, 6, 2],      # +Y
        [0, 1, 5], [0, 5, 4],      # -Y
    ], dtype=np.int64)
    return verts, faces


def _unit_cylinder(np: Any, segments: int = _CYLINDER_SEGMENTS):
    """Unit cylinder about the local Y axis, radius 0.5, height 1, capped."""
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    ring_x = 0.5 * np.cos(angles)
    ring_z = 0.5 * np.sin(angles)
    bottom = np.stack([ring_x, np.full(segments, -0.5), ring_z], axis=1)
    top = np.stack([ring_x, np.full(segments, 0.5), ring_z], axis=1)
    centres = np.array([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]], dtype=np.float64)
    verts = np.concatenate([bottom, top, centres], axis=0)

    lo = np.arange(segments)
    hi = (lo + 1) % segments
    side = np.concatenate([
        np.stack([lo, hi, hi + segments], axis=1),
        np.stack([lo, hi + segments, lo + segments], axis=1),
    ], axis=0)
    bottom_centre, top_centre = 2 * segments, 2 * segments + 1
    caps = np.concatenate([
        np.stack([np.full(segments, bottom_centre), hi, lo], axis=1),
        np.stack([np.full(segments, top_centre), lo + segments, hi + segments], axis=1),
    ], axis=0)
    return verts, np.concatenate([side, caps], axis=0).astype(np.int64)


_UNIT_BUILDERS = {
    "plane": _unit_plane,
    "box": _unit_box,
    "cube": _unit_box,
    "cylinder": _unit_cylinder,
}


def tessellate_primitive(prim: Any) -> tuple[Any, Any] | None:
    """``(vertices, faces)`` in world space, or None if not tessellatable.

    ``mesh`` primitives carry their own triangles and are returned verbatim via
    :func:`atlas_camera.exporters._layers.mesh_from_primitive`. Unknown
    primitive types return None rather than guessing — a consumer that silently
    substituted a box for something it did not recognise would corrupt exactly
    the measurements this module exists to feed.
    """
    np = _require_numpy()
    kind = getattr(prim, "primitive_type", None)

    if kind == "mesh":
        from atlas_camera.exporters._layers import mesh_from_primitive
        mesh = mesh_from_primitive(prim)
        if mesh is None:
            return None
        return (np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.int64))

    builder = _UNIT_BUILDERS.get(kind or "")
    if builder is None:
        return None

    verts, faces = builder(np)
    dims = np.asarray(getattr(prim, "dimensions", (1.0, 1.0, 1.0)), dtype=np.float64)
    if dims.shape != (3,):
        dims = np.ones(3, dtype=np.float64)
    # A plane has no local thickness; a zero Y scale would collapse nothing but
    # must not be allowed to introduce NaNs elsewhere.
    scaled = verts * np.where(np.abs(dims) > 1e-12, dims, 1.0)

    mat = np.asarray(getattr(prim, "transform_matrix", None), dtype=np.float64)
    if mat.shape != (4, 4):
        mat = np.eye(4, dtype=np.float64)
    world = scaled @ mat[:3, :3].T + mat[:3, 3]
    return world, faces


def collect_scene_triangles(
    solve: Any,
    *,
    include_mesh: bool = True,
    include_primitives: bool = True,
    exclude_roles: tuple[str, ...] = (),
    exclude_names: tuple[str, ...] = (),
) -> tuple[Any, Any, list[str]]:
    """Every triangle in a solve's projection scene, merged.

    Returns ``(vertices, faces, sources)`` where ``sources`` names what was
    included, so a caller can report what its measurement was actually taken
    against instead of leaving the artist to guess.

    ``exclude_names`` drops primitives by name. The move budget uses it to leave
    out the backdrop cyclorama, which spans the whole frustum and would
    otherwise make every measurement report full coverage.
    """
    np = _require_numpy()
    scene = getattr(solve, "projection_scene", None)
    prims = list(getattr(scene, "proxy_geometry", None) or []) if scene is not None else []

    chunks: list[tuple[Any, Any]] = []
    sources: list[str] = []
    for prim in prims:
        kind = getattr(prim, "primitive_type", None)
        if kind == "mesh" and not include_mesh:
            continue
        if kind != "mesh" and not include_primitives:
            continue
        role = (getattr(prim, "metadata", None) or {}).get("role")
        if role and role in exclude_roles:
            continue
        if (getattr(prim, "name", None) or "") in exclude_names:
            continue
        built = tessellate_primitive(prim)
        if built is None:
            continue
        chunks.append(built)
        sources.append(getattr(prim, "name", None) or str(kind))

    if not chunks:
        return (np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.int64), sources)

    offset = 0
    all_verts, all_faces = [], []
    for verts, faces in chunks:
        all_verts.append(verts)
        all_faces.append(faces + offset)
        offset += len(verts)
    return (np.concatenate(all_verts, axis=0),
            np.concatenate(all_faces, axis=0), sources)
