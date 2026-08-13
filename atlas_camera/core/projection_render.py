"""Headless projection-scene rendering (pure numpy, host-agnostic).

Renders the layered projection scene from an ARBITRARY camera — the piece the
screen-space evidence composite in ``comfy/headless_evidence.py`` deliberately
does not do (it re-composites stored UV coverage at the recovered pose only).
Promoted here so geometry-true novel views (stereo eyes, QA offsets) share one
implementation:

  * ``gather_scene_meshes`` — the solve's serialized relief/proxy meshes with
    their per-vertex projective UVs and owning-layer label (the generalization
    of headless_evidence's ``_mesh_arrays``, which now delegates here);
  * ``project_points`` — the Atlas pixel convention (u = cx + fx*x/w,
    v = cy - fy*y/w, w = -z from the 4x4 view matrix);
  * ``render_scene`` — z-buffered, perspective-correct-UV triangle rasterizer.
    Each mesh samples its OWN layer's texture through the UVs its projector
    camera baked — the matte-painting property (texture stays glued to
    geometry) survives any viewing camera.

Stereo eye construction (``stereo_eye_view_matrices``) lives here too: eyes
translate along the camera's own right axis from the 4x4 view matrix (never
the bare 3x3 — the world-math rule); convergence is SHIFTED-SENSOR (cx
offsets), never toe-in, so verticals stay vertical in both eyes.

Textures arrive as plain float arrays (decode stays host-side); tears and
mattes are already baked into the serialized meshes / texture alphas, so a
disocclusion beyond the layered coverage renders as a hole — the honest
geometry-true answer, not a defect of this renderer.
"""

from __future__ import annotations

from typing import Any


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "projection rendering requires numpy. Install with:\n"
            "    pip install -e .[vision]"
        ) from exc
    return np


#: Solid proxy VOLUMES the viewport already draws as projectable geometry
#: (`atlas_blockout.js` builds BoxGeometry / CylinderGeometry and stamps
#: `atlasDerived = true`). Explicit ALLOWLIST, so a new primitive type cannot
#: leak into the rasterizer by default.
#:
#: `plane` is deliberately EXCLUDED even though the viewport draws it. Planes
#: in Atlas are COVERAGE CARDS — `projection_ground`, the projection backdrop,
#: dynamic-plate receivers — whose whole job is to guarantee camera-view
#: coverage. Rasterising them headless reports zero disocclusion everywhere,
#: which does not widen anything; it just blinds AtlasDisocclusionGuide and the
#: move-budget measurement to the holes they exist to find. Found by
#: tests/test_disocclusion_guide.py, which went from 7 failures to green when
#: planes came out.
#:
#: Guide affordances (`height_guide`, `axis_guide`) are absent for the simpler
#: reason that UI furniture must never occlude a render.
_TESSELLATED_PRIMITIVES = ("box", "cylinder")

#: THREE.BoxGeometry is CENTRED on the origin, and `block_massing.box_transform`
#: matches it — the transform carries `ground_y + 0.5 * height`, with
#: `dimensions` as FULL extents. A unit cube spanning y in [0, 1] would float
#: every mass half its own height above the ground.
_UNIT_BOX_V = (
    (-.5, -.5, -.5), (.5, -.5, -.5), (.5, -.5, .5), (-.5, -.5, .5),
    (-.5, .5, -.5), (.5, .5, -.5), (.5, .5, .5), (-.5, .5, .5))
_UNIT_BOX_F = (
    (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
    (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))

#: THREE.PlaneGeometry lies in XY, centred, facing +Z.
_UNIT_PLANE_V = ((-.5, -.5, 0.), (.5, -.5, 0.), (.5, .5, 0.), (-.5, .5, 0.))
_UNIT_PLANE_F = ((0, 1, 2), (0, 2, 3))

_CYLINDER_SEGMENTS = 24


def _unit_cylinder(np):
    """Y-axis cylinder, unit diameter and height, centred — THREE's convention
    (`CylinderGeometry(d0/2, d0/2, d1, 24)`)."""
    n = _CYLINDER_SEGMENTS
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x, z = 0.5 * np.cos(ang), 0.5 * np.sin(ang)
    verts = [(0.0, -0.5, 0.0), (0.0, 0.5, 0.0)]
    verts += [(float(x[i]), -0.5, float(z[i])) for i in range(n)]
    verts += [(float(x[i]), 0.5, float(z[i])) for i in range(n)]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        b0, b1, t0, t1 = 2 + i, 2 + j, 2 + n + i, 2 + n + j
        faces += [(0, b1, b0), (1, t0, t1),          # caps
                  (b0, b1, t1), (b0, t1, t0)]        # wall
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def tessellate_primitive(primitive: Any) -> tuple | None:
    """Solid proxy primitive -> ``(vertices Nx3, faces Mx3)`` in WORLD space.

    Returns ``None`` for anything not in ``_TESSELLATED_PRIMITIVES`` so guides
    and unknown types are skipped rather than guessed at.
    """
    np = _require_numpy()
    kind = getattr(primitive, "primitive_type", None)
    if kind not in _TESSELLATED_PRIMITIVES:
        return None
    dims = np.asarray(getattr(primitive, "dimensions", (1.0, 1.0, 1.0)),
                      dtype=np.float64).reshape(3)
    if kind == "box":
        v = np.asarray(_UNIT_BOX_V, dtype=np.float64) * dims[None, :]
        f = np.asarray(_UNIT_BOX_F, dtype=np.int64)
    elif kind == "plane":
        v = np.asarray(_UNIT_PLANE_V, dtype=np.float64) * dims[None, :]
        f = np.asarray(_UNIT_PLANE_F, dtype=np.int64)
    else:
        v, f = _unit_cylinder(np)
        v = v * np.array([dims[0], dims[1], dims[0]], dtype=np.float64)[None, :]
    matrix = np.asarray(getattr(primitive, "transform_matrix", None),
                        dtype=np.float64)
    if matrix.shape != (4, 4):
        return None
    # Column-vector convention (translation in column 3, as box_transform
    # writes it): world = M @ p, which for row-stored points is p @ M.T.
    hom = np.concatenate([v, np.ones((len(v), 1), dtype=np.float64)], axis=1)
    return (hom @ matrix.T)[:, :3], f


def merge_volume_primitives(solve: Any, *, name: str = "massing_mesh") -> int:
    """Fold every solid volume primitive into ONE serialized mesh primitive.

    Massing emits one primitive per building — 97 on a city plate — and
    ``render_scene`` loops per mesh, so the rasterizer slows roughly in
    proportion. Merging gives it a single mesh, and (the real reason this
    exists) turns placeholder mass into a normal relief-mesh-shaped layer that
    ``AtlasRetopologizeLayer`` can decimate/remesh like any other.

    TRUST IS PRESERVED. The merged primitive keeps ``provenance="placeholder"``
    and the placeholder trust tier: merging is a topology operation, and no
    topology operation may promote a guess to a measurement.

    Returns the number of primitives merged (0 = nothing to do, solve
    untouched).
    """
    np = _require_numpy()
    from atlas_camera.core.schema import AtlasProxyPrimitive

    scene = getattr(solve, "projection_scene", None)
    prims = list(getattr(scene, "proxy_geometry", None) or []) if scene else []
    volumes = [p for p in prims
               if getattr(p, "primitive_type", None) in _TESSELLATED_PRIMITIVES]
    if not volumes:
        return 0

    verts_all, faces_all, offset = [], [], 0
    for p in volumes:
        tess = tessellate_primitive(p)
        if tess is None:
            continue
        v, f = tess
        verts_all.append(v)
        faces_all.append(f + offset)
        offset += len(v)
    if not verts_all:
        return 0
    verts = np.concatenate(verts_all, axis=0)
    faces = np.concatenate(faces_all, axis=0)

    camera = getattr(solve, "camera", None)
    uvs = (_projective_uvs(np, verts, camera) if camera is not None
           else np.zeros((len(verts), 2)))

    # Placeholder-ness survives the merge, and is recorded as a count so a
    # consumer can still say "97 guesses" rather than "one mesh".
    merged = AtlasProxyPrimitive(
        name=name,
        primitive_type="mesh",
        material="atlas_projection_proxy",
        metadata={
            "role": (volumes[0].metadata or {}).get("role", "projection_proxy"),
            "source": "merged_volume_primitives",
            "provenance": "placeholder",
            "trust": "placeholder",
            "merged_primitive_count": len(volumes),
            "vertices": np.round(verts.reshape(-1), 3).tolist(),
            "faces": faces.reshape(-1).tolist(),
            "uvs": np.round(uvs.reshape(-1), 4).tolist(),
        },
    )
    keep = [p for p in prims
            if getattr(p, "primitive_type", None) not in _TESSELLATED_PRIMITIVES]
    scene.proxy_geometry = keep + [merged]
    return len(volumes)


def _projective_uvs(np, verts, camera) -> Any:
    """Per-vertex UVs from the solve camera — what the VIEWPORT does when it
    projects the primary plate onto derived geometry.

    V is BOTTOM-ORIGIN, matching the serialized-mesh convention `render_scene`
    documents and `relief_mesh_primitive`'s own baked UVs.
    """
    from atlas_camera.core.camera_spec import CameraSpec
    spec = CameraSpec.from_intrinsics(camera.intrinsics)
    w = float(camera.intrinsics.image_width)
    h = float(camera.intrinsics.image_height)
    px, _forward = project_points(
        verts, camera.extrinsics.camera_view_matrix,
        spec.fx, spec.fy, spec.cx, spec.cy)
    return np.stack([px[:, 0] / w, 1.0 - px[:, 1] / h], axis=1)


def gather_scene_meshes(solve: Any, *, with_uvs: bool = False) -> list:
    """Every serialized mesh on the solve as numpy arrays.

    Returns ``(label, vertices Nx3, faces Mx3, metadata)`` tuples — or, with
    ``with_uvs=True``, ``(label, vertices, faces, uvs Nx2 | None, texture_label,
    metadata)`` where ``texture_label`` is ``"primary"`` for the primary
    scene's meshes and the source's name for patch/clean-plate layers.
    """
    np = _require_numpy()
    out = []

    camera = getattr(solve, "camera", None)

    def add(primitives, prefix, tex_label):
        for primitive in primitives or []:
            kind = getattr(primitive, "primitive_type", None)
            if kind != "mesh":
                # Solid proxies (massing boxes, receiver planes, cylinders) are
                # real occluders the VIEWPORT already draws, but this renderer
                # skipped them entirely — so AtlasDisocclusionGuide, the
                # occlusion-fill path and every move-budget measurement were
                # blind to placeholder mass. Tessellate them so headless
                # matches the viewport.
                tess = tessellate_primitive(primitive)
                if tess is None or camera is None:
                    continue
                verts, tris = tess
                label = f"{prefix}{getattr(primitive, 'name', kind)}"
                meta = dict(getattr(primitive, "metadata", None) or {})
                meta["tessellated_from"] = kind
                if with_uvs:
                    out.append((label, verts, tris,
                                _projective_uvs(np, verts, camera),
                                tex_label, meta))
                else:
                    out.append((label, verts, tris, meta))
                continue
            meta = getattr(primitive, "metadata", None) or {}
            vertices = meta.get("vertices") or []
            faces = meta.get("faces") or []
            if len(vertices) < 9 or len(faces) < 3:
                continue
            verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
            tris = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
            label = f"{prefix}{getattr(primitive, 'name', 'mesh')}"
            if with_uvs:
                raw_uv = meta.get("uvs") or []
                uvs = (np.asarray(raw_uv, dtype=np.float64).reshape(-1, 2)
                       if len(raw_uv) >= 2 * len(verts) else None)
                out.append((label, verts, tris, uvs, tex_label, meta))
            else:
                out.append((label, verts, tris, meta))

    scene = getattr(solve, "projection_scene", None)
    add(getattr(scene, "proxy_geometry", None) or [], "", "primary")
    for source in getattr(solve, "projection_sources", None) or []:
        name = str(getattr(source, "name", "layer") or "layer")
        add(getattr(source, "proxy_geometry", None) or [], f"{name}/", name)
    return out


def project_points(vertices: Any, view: Any, fx: float, fy: float,
                   cx: float, cy: float) -> tuple:
    """(pixels Nx2, forward-depth N) under the Atlas projection convention."""
    np = _require_numpy()
    hom = np.concatenate(
        [vertices, np.ones((len(vertices), 1), dtype=np.float64)], axis=1)
    camera = hom @ np.asarray(view, dtype=np.float64).T
    forward = -camera[:, 2]
    safe = np.where(np.abs(forward) < 1e-9, 1e-9, forward)
    pixels = np.stack([
        cx + fx * camera[:, 0] / safe,
        cy - fy * camera[:, 1] / safe,
    ], axis=1)
    return pixels, forward


def stereo_eye_view_matrices(view_matrix: Any, interocular_m: float) -> tuple:
    """(left_view, right_view) 4x4s: eyes offset ±io/2 along CAMERA-right.

    Built strictly from the full 4x4 (world-math rule): right axis is the
    camera world matrix's first column; orientation is untouched — convergence
    belongs to the intrinsics (shifted sensor), never to a toe-in rotation.
    """
    np = _require_numpy()
    view = np.asarray(view_matrix, dtype=np.float64)
    if view.shape != (4, 4):
        raise ValueError("camera_view_matrix must be 4x4")
    cam_to_world = np.linalg.inv(view)
    right = cam_to_world[:3, 0]
    right = right / (np.linalg.norm(right) or 1.0)
    half = 0.5 * float(interocular_m)

    def eye_view(offset):
        w = cam_to_world.copy()
        w[:3, 3] = w[:3, 3] + right * offset
        return np.linalg.inv(w)

    return eye_view(-half), eye_view(+half)


def converged_cx(fx: float, cx: float, interocular_m: float,
                 convergence_m: float) -> tuple:
    """(cx_left, cx_right) for a shifted-sensor convergence at ``convergence_m``.

    0 (or negative) convergence means parallel eyes: both keep ``cx``. The
    point at the convergence distance lands on the SAME image x in both eyes.
    """
    if convergence_m is None or float(convergence_m) <= 0:
        return float(cx), float(cx)
    off = float(fx) * (0.5 * float(interocular_m)) / float(convergence_m)
    return float(cx) - off, float(cx) + off


def render_scene(
    meshes: list,
    textures: dict,
    view: Any, fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
) -> tuple:
    """Z-buffered render of ``meshes`` (from ``gather_scene_meshes(with_uvs=True)``).

    ``textures`` maps texture_label -> HxWx3/4 float array in [0, 1]; a mesh
    whose label or UVs are missing is skipped (reported in stats). UV origin
    is bottom-left (the serialized-mesh convention). Per-pixel nearest forward
    depth wins across ALL meshes — layer priority is irrelevant under a true
    z-test. Returns ``(rgb HxWx3 float32, alpha HxW float32, stats dict)``.
    """
    np = _require_numpy()
    zbuf = np.full((height, width), np.inf, dtype=np.float64)
    rgb = np.zeros((height, width, 3), dtype=np.float64)
    alpha = np.zeros((height, width), dtype=np.float64)
    stats = {"meshes_rendered": 0, "meshes_skipped": [], "faces_rasterized": 0}

    for label, verts, faces, uvs, tex_label, _meta in meshes:
        tex = textures.get(tex_label)
        if tex is None or uvs is None:
            stats["meshes_skipped"].append(label)
            continue
        tex = np.asarray(tex, dtype=np.float64)
        if tex.ndim == 2:
            tex = tex[..., None].repeat(3, axis=-1)
        th, tw = tex.shape[:2]
        tex_rgb = tex[..., :3]
        tex_a = tex[..., 3] if tex.shape[-1] >= 4 else np.ones((th, tw))

        px, fwd = project_points(verts, view, fx, fy, cx, cy)
        inv_w = 1.0 / np.maximum(fwd, 1e-9)
        uv_over_w = uvs * inv_w[:, None]

        tri_px = px[faces]              # (M, 3, 2)
        tri_fwd = fwd[faces]            # (M, 3)
        tri_uvw = uv_over_w[faces]      # (M, 3, 2)
        tri_invw = inv_w[faces]         # (M, 3)

        # Cull: any vertex behind the eye, or bbox fully outside the frame.
        front = (tri_fwd > 1e-6).all(axis=1)
        mins = tri_px.min(axis=1)
        maxs = tri_px.max(axis=1)
        onscreen = ((maxs[:, 0] >= 0) & (mins[:, 0] <= width - 1)
                    & (maxs[:, 1] >= 0) & (mins[:, 1] <= height - 1))
        keep = front & onscreen
        for t in np.nonzero(keep)[0]:
            p = tri_px[t]
            x0 = max(int(np.floor(p[:, 0].min())), 0)
            x1 = min(int(np.ceil(p[:, 0].max())), width - 1)
            y0 = max(int(np.floor(p[:, 1].min())), 0)
            y1 = min(int(np.ceil(p[:, 1].max())), height - 1)
            if x1 < x0 or y1 < y0:
                continue
            xs = np.arange(x0, x1 + 1, dtype=np.float64) + 0.5
            ys = np.arange(y0, y1 + 1, dtype=np.float64) + 0.5
            gx, gy = np.meshgrid(xs, ys)
            (ax, ay), (bx, by), (ccx, ccy) = p
            det = (bx - ax) * (ccy - ay) - (ccx - ax) * (by - ay)
            if abs(det) < 1e-12:
                continue
            w1 = ((gx - ax) * (ccy - ay) - (ccx - ax) * (gy - ay)) / det
            w2 = ((bx - ax) * (gy - ay) - (gx - ax) * (by - ay)) / det
            w0 = 1.0 - w1 - w2
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            if not inside.any():
                continue
            # Perspective-correct interpolation via 1/w.
            invw = (w0 * tri_invw[t, 0] + w1 * tri_invw[t, 1] + w2 * tri_invw[t, 2])
            invw = np.maximum(invw, 1e-12)
            depth = 1.0 / invw
            u = (w0 * tri_uvw[t, 0, 0] + w1 * tri_uvw[t, 1, 0] + w2 * tri_uvw[t, 2, 0]) / invw
            v = (w0 * tri_uvw[t, 0, 1] + w1 * tri_uvw[t, 1, 1] + w2 * tri_uvw[t, 2, 1]) / invw

            sub = (slice(y0, y1 + 1), slice(x0, x1 + 1))
            visible = inside & (depth < zbuf[sub])
            if not visible.any():
                continue
            # UV origin bottom-left; nearest-texel sample.
            ti = np.clip(np.rint((1.0 - v) * (th - 1)), 0, th - 1).astype(np.int64)
            tj = np.clip(np.rint(u * (tw - 1)), 0, tw - 1).astype(np.int64)
            a_smp = tex_a[ti, tj]
            visible &= a_smp > 0.5
            if not visible.any():
                continue
            zb = zbuf[sub]
            rb = rgb[sub]
            ab = alpha[sub]
            zb[visible] = depth[visible]
            rb[visible] = tex_rgb[ti[visible], tj[visible]]
            ab[visible] = a_smp[visible]
            stats["faces_rasterized"] += 1
        stats["meshes_rendered"] += 1

    # The z-buffer is already exact here and comes from the SAME depth test
    # that chose every colour, so a depth pass taken from it is guaranteed
    # consistent with the RGB pass — which is what a depth-conditioned
    # generator needs (Atlas can hand a model measured scene depth instead of
    # the monocular guess every other consumer feeds one). +inf where nothing
    # was rasterized; callers decide how to encode holes.
    stats["depth"] = zbuf
    return rgb.astype(np.float32), alpha.astype(np.float32), stats
