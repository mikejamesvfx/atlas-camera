"""Export-only retopology for Atlas relief meshes — CPU, M2-safe, no CUDA.

Atlas relief meshes are dense, irregular, and torn at depth discontinuities —
fine for live 📽 projection (texels assigned by ray), but a poor handoff for a
DCC retopo / boolean / 3D-print pass: 100k+ triangles, skinny slivers at tear
edges, and open boundary loops. This module caps / retopologizes the
**exported** mesh (never the live viewport projection mesh, never
``solve.proxy_geometry``) so the OBJ/GLB the artist hands to Maya / ZBrush /
Blender is clean and light.

It is deliberately **export-only** and **CPU-only**:

* Export-only — retopology changes vertex count, which breaks the 1:1
  vertex-UV invariant the live projection depends on. Running it on the live
  mesh would force a UV regen + full payload re-serialize through the browser
  every execution (the documented viewport-payload risk at high grids), and
  quad remeshing / decimation smooths the *deliberate* silhouette tears the
  matte-painting projection relies on. So it runs once, on the resolved
  ``ReliefMesh``, before the OBJ/GLB writers — the exact pattern
  ``mesh_repair.apply_interior_hole_fill`` already follows.
* CPU-only — retopology is geometry processing, not a GPU workload, so it
  needs no torch / CUDA / MPS / Docker. The only machine filter is the pip
  wheel matrix, and every path below is M2-safe (macOS arm64 wheels exist OR
  it is pure numpy).

Three paths, each guarded so the node degrades gracefully when its optional
dep is absent (the same discipline every optional import in ``atlas_camera``
follows):

1. **Quad retopology** (lead path) — ``pyinstantmeshes`` (BSD,
   greenbrettmichael/pyinstantmeshes), a Python wheel wrapping Instant Meshes
   (orientation-field quad remeshing). M2-safe: ``macosx_11_0_arm64`` wheels
   cp311-cp314. Outputs N×4 quad faces → triangulated before writing.
2. **Quadric decimation** — ``fast-simplification`` (BSD) backing
   ``trimesh.simplify_quadric_decimation``. M2-safe: ``macosx_11_0_arm64``
   cp39-cp313. Pure decimation (no remeshing) — keeps the original topology
   class, just fewer faces.
3. **Smooth / relax** — ``trimesh`` pure-numpy Taubin smoothing (MIT, the
   lightest install, so the practical baseline on every machine). Topology is
   *unchanged* (same faces, same vertex count) — only vertex positions move,
   so the 1:1 vertex-UV mapping is preserved and UVs are NOT regenerated.

The UV-loss problem and its fix
-------------------------------
Quad remeshing and decimation change the vertex count, which breaks the 1:1
vertex-UV mapping the OBJ/GLB writers depend on (``f {a+1}/{a+1} ...``).
The fix is to **regenerate the same projective UVs for the new vertices** —
project each new vertex through the recovered camera → image pixel → UV,
mirroring the bake in ``relief_mesh.build_relief_mesh`` (lines 452-496). This
is pure numpy → M2-safe and CI-safe, and it keeps the retopologized mesh
textured with the source photo with no writer change. The smooth path keeps
the existing UVs (topology unchanged → 1:1 preserved).
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Projective UV regeneration (pure numpy, dep-free, M2 + CI safe)
# ---------------------------------------------------------------------------

def regenerate_projective_uvs(
    vertices: np.ndarray,
    *,
    view_matrix: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Regenerate the recovered-camera projection UVs for ``vertices``.

    The inverse of ``relief_mesh.build_relief_mesh``'s forward back-projection
    (lines 454-496): each world vertex is transformed into the recovered
    camera's frame, projected to an image pixel, then mapped to an OBJ
    bottom-left UV — exactly reproducing the bake so a retopologized mesh
    stays textured with the source photo.

    Forward bake (for reference)::

        c2w = inv(view_matrix); R_cw = c2w[:3,:3]; cam = c2w[:3,3]
        x = (uu - cx)/fx * d ; y = -(vv - cy)/fy * d ; z = -d
        world = stack([x,y,z], -1) @ R_cw.T + cam
        u = uu/(W-1) ; v = 1 - vv/(H-1)

    Inverse (here)::

        p_cam = (world - cam) @ R_cw            # world → camera frame
        d = -p_cam.z                            # forward depth (camera faces -Z)
        px = cx - fx * p_cam.x / p_cam.z
        py = cy + fy * p_cam.y / p_cam.z
        u = px/(W-1) ; v = 1 - py/(H-1)

    Vertices behind the camera (``p_cam.z >= 0``, undefined projection) are
    clamped to the image boundary so the writer never sees NaN UVs — the
    retopologized mesh is a convex-ish combination of in-front-of-camera
    verts so this is a defensive guard, not a hot path.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3); got {v.shape}")
    vm = np.asarray(view_matrix, dtype=np.float64)
    if vm.shape != (4, 4):
        raise ValueError("view_matrix must be a 4x4 row-major matrix")

    c2w = np.linalg.inv(vm)
    R_cw = c2w[:3, :3]          # camera → world rotation
    cam = c2w[:3, 3]            # camera position (world)

    # World → camera frame. Forward bake did `world = p_local @ R_cw.T + cam`,
    # so `p_local = (world - cam) @ R_cw`.
    p_cam = (v - cam) @ R_cw
    x_c = p_cam[:, 0]
    y_c = p_cam[:, 1]
    z_c = p_cam[:, 2]

    # Camera faces world -Z → in-front vertices have z_c < 0. Guard against
    # division by zero / behind-camera verts (defensive; retopo'd verts are
    # combinations of in-front originals).
    eps = 1e-9
    safe_z = np.where(z_c < -eps, z_c, -eps)

    px = cx - fx * x_c / safe_z
    py = cy + fy * y_c / safe_z

    # Clamp to the image boundary so degenerate projections land on the edge
    # rather than far outside the texture (NaNs would corrupt the writer).
    W = max(int(image_width), 1)
    H = max(int(image_height), 1)
    px = np.clip(px, 0.0, float(W - 1))
    py = np.clip(py, 0.0, float(H - 1))

    u = px / max(W - 1, 1)
    v_uv = 1.0 - py / max(H - 1, 1)
    return np.stack([u, v_uv], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Quad → triangle conversion (pyinstantmeshes returns N×4 for quads)
# ---------------------------------------------------------------------------

def _triangulate_quads(faces: np.ndarray) -> np.ndarray:
    """Convert a quad / quad-dominant face array to triangles.

    ``pyinstantmeshes.remesh`` with ``posy=4`` returns an (M,4) quad array; a
    quad-dominant result (``pure_quad=False``) is still (M,4) but some rows are
    *degenerate quads* encoding a single triangle (``d == a`` / ``d == b`` /
    ``d == c`` / ``d < 0``). This splits each real quad into two triangles and
    each degenerate row into one, returning a clean (K,3) int array. An (M,3)
    input passes through unchanged.
    """
    f = np.asarray(faces)
    if f.ndim != 2 or f.shape[1] == 3:
        return np.asarray(f, dtype=np.int64).reshape(-1, 3)
    if f.shape[1] != 4:
        raise ValueError(f"faces must be (M,3) or (M,4); got {f.shape}")

    a = f[:, 0]
    b = f[:, 1]
    c = f[:, 2]
    d = f[:, 3]
    # A degenerate quad encodes one triangle (d repeats an existing index or
    # is negative). Real quad → two triangles [a,b,c],[a,c,d].
    degenerate = (d == a) | (d == b) | (d == c) | (d < 0)

    tri1 = np.stack([a, b, c], axis=1)
    tri2 = np.stack([a, c, d], axis=1)
    # Degenerate rows: keep tri1, drop tri2 (its d is invalid).
    tri2_mask = ~degenerate
    tris = np.concatenate([tri1, tri2[tri2_mask]], axis=0)
    return tris.astype(np.int64)


# ---------------------------------------------------------------------------
# Quad retopology (pyinstantmeshes — lead path, guarded)
# ---------------------------------------------------------------------------

def retopo_quad(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    target_vertex_count: int = 2000,
    posy: int = 4,
    rosy: int = 4,
    pure_quad: bool = False,
    crease_angle: float = 30.0,
    smooth_iterations: int = 0,
    deterministic: bool = True,
    align_to_boundaries: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Quad-retopologize via ``pyinstantmeshes.remesh``.

    Returns ``(out_vertices, out_faces)`` where ``out_faces`` is **triangulated**
    ((K,3)) so the OBJ/GLB writers (triangles-only) consume it directly.
    Raises an informative ``ImportError`` with the install hint when
    ``pyinstantmeshes`` is not installed.
    """
    try:
        import pyinstantmeshes
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "Quad retopology needs the 'pyinstantmeshes' package "
            "(BSD, CPU-only, macOS arm64 wheels available).\n"
            "Install it with:  pip install pyinstantmeshes"
        ) from exc

    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)
    out_v, out_f = pyinstantmeshes.remesh(
        v,
        f,
        target_vertex_count=int(target_vertex_count),
        posy=int(posy),
        rosy=int(rosy),
        pure_quad=bool(pure_quad),
        crease_angle=float(crease_angle),
        smooth_iterations=int(smooth_iterations),
        deterministic=bool(deterministic),
        align_to_boundaries=bool(align_to_boundaries),
    )
    out_v = np.asarray(out_v, dtype=np.float64)
    out_faces = _triangulate_quads(np.asarray(out_f))
    return out_v, out_faces


# ---------------------------------------------------------------------------
# Quadric decimation (fast-simplification via trimesh — guarded)
# ---------------------------------------------------------------------------

def decimate_quadric(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    target_face_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Quadric-decimate via ``trimesh.simplify_quadric_decimation``.

    ``trimesh`` (MIT, guarded optional import); its quadric decimation backs onto
    ``fast-simplification`` (BSD, macOS arm64 wheels). We guard the
    ``fast-simplification`` import explicitly first so the user gets a clean
    install hint rather than a trimesh-internal traceback. Decimation keeps
    the original topology class (no remeshing) — just fewer faces.
    """
    try:
        import fast_simplification  # noqa: F401 — presence check
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "Quadric decimation needs the 'fast-simplification' package "
            "(BSD, CPU-only, macOS arm64 wheels).\n"
            "Install it with:  pip install fast-simplification"
        ) from exc
    import trimesh

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    out = tm.simplify_quadric_decimation(face_count=int(target_face_count))
    out_v = np.asarray(out.vertices, dtype=np.float64)
    out_f = np.asarray(out.faces, dtype=np.int64).reshape(-1, 3)
    return out_v, out_f


# ---------------------------------------------------------------------------
# Smooth / relax (trimesh pure-numpy — guaranteed baseline, topology-preserving)
# ---------------------------------------------------------------------------

def smooth_relax(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Taubin-smooth vertex positions via ``trimesh`` (pure numpy).

    Topology is **unchanged** — same faces, same vertex count — so the 1:1
    vertex-UV mapping is preserved and the caller should NOT regenerate UVs
    after this. This is the runnable baseline on every machine (no CUDA, no
    native remesher), and raises an informative ``ImportError`` when a piece
    of it is missing.

    Needs **both** trimesh and scipy. trimesh's only required dependency is
    numpy — scipy is an optional ``trimesh[easy]`` extra that it imports
    lazily, and Taubin smoothing's laplacian goes through
    ``trimesh.graph`` -> scipy. Without the explicit check below, a
    trimesh-but-no-scipy environment fails deep inside trimesh with a bare
    ``ModuleNotFoundError`` instead of an actionable message.
    """
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - trimesh is a dev/test dep
        raise ImportError(
            "Smoothing needs 'trimesh' (MIT).\n"
            "Install it with:  pip install trimesh scipy"
        ) from exc
    try:
        import scipy  # noqa: F401  (trimesh imports it lazily, inside the filter)
    except ImportError as exc:
        raise ImportError(
            "Smoothing needs 'scipy' as well as trimesh — Taubin smoothing's "
            "laplacian is scipy-backed, and trimesh does not require scipy "
            "itself.\n"
            "Install it with:  pip install scipy"
        ) from exc

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    # filter_taubin alternates lamb/nu shrink/grow — net volume-preserving.
    # Kept faces identical; only tm.vertices move. (Its laplacian is
    # scipy-backed, hence the scipy guard above.)
    trimesh.smoothing.filter_taubin(
        tm, lamb=0.5, nu=-0.53, iterations=int(max(iterations, 0))
    )
    out_v = np.asarray(tm.vertices, dtype=np.float64)
    return out_v, np.asarray(faces, dtype=np.int64).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Node-facing wrapper (mirrors apply_interior_hole_fill's discipline)
# ---------------------------------------------------------------------------

_RETOPO_METHODS = ("off", "quad", "decimate", "smooth")


def apply_retopo(
    mesh: Any,
    *,
    method: str = "off",
    target_vertex_count: int = 2000,
    view_matrix: np.ndarray | None = None,
    fx: float = 0.0,
    fy: float = 0.0,
    cx: float | None = None,
    cy: float | None = None,
    image_width: int = 0,
    image_height: int = 0,
    pure_quad: bool = False,
    crease_angle: float = 30.0,
    smooth_iterations: int = 0,
    deterministic: bool = True,
    align_to_boundaries: bool = True,
) -> dict[str, Any]:
    """Apply a retopology pass to a ``ReliefMesh`` in place (export-only).

    ``method`` selects the path: ``"off"`` (no-op, the default so every saved
    workflow keeps working), ``"quad"`` (pyinstantmeshes), ``"decimate``
    (fast-simplification), or ``"smooth"`` (trimesh Taubin).

    For ``quad`` / ``decimate`` the vertex count changes → the 1:1 vertex-UV
    mapping breaks → UVs are **regenerated** via :func:`regenerate_projective_uvs`
    (needs ``view_matrix`` + the solved intrinsics + image size). For
    ``smooth`` the topology is unchanged → the existing UVs are kept (1:1
    preserved), and no intrinsics are needed.

    Returns a report dict (``method``, ``changed``, ``in_verts``, ``out_verts``,
    ``in_faces``, ``out_faces``, ``note``) for the node's status line. No-op
    report when ``method == "off"`` or the mesh has no faces.
    """
    if method not in _RETOPO_METHODS:
        raise ValueError(
            f"method must be one of {_RETOPO_METHODS}; got {method!r}"
        )
    if method == "off":
        return {"method": "off", "changed": False, "note": "retopology off"}

    faces = getattr(mesh, "faces", None)
    if faces is None or len(faces) == 0:
        return {"method": method, "changed": False, "note": "no faces to retopo"}

    vertices = np.asarray(getattr(mesh, "vertices"), dtype=np.float64)
    in_v, in_f = int(len(vertices)), int(len(faces))

    if method == "smooth":
        out_v, out_f = smooth_relax(vertices, faces, iterations=int(smooth_iterations))
        mesh.vertices = out_v
        # Topology unchanged → 1:1 vertex-UV preserved → keep existing UVs.
        return {
            "method": "smooth", "changed": True,
            "in_verts": in_v, "out_verts": int(len(out_v)),
            "in_faces": in_f, "out_faces": int(len(out_f)),
            "note": f"trimesh Taubin smooth ×{smooth_iterations} (UVs preserved)",
        }

    # quad / decimate change the vertex count → regenerate projective UVs.
    if view_matrix is None or fx <= 0 or image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"Retopology method '{method}' changes the vertex count, so the "
            "1:1 vertex-UV mapping must be regenerated — needs view_matrix + "
            "fx/fy/cx/cy + image_width/image_height (the solved intrinsics). "
            "Wire the solve's camera (or use method='smooth', which preserves UVs)."
        )
    cxv = float(cx) if cx is not None else image_width / 2.0
    cyv = float(cy) if cy is not None else image_height / 2.0

    if method == "quad":
        out_v, out_f = retopo_quad(
            vertices, faces,
            target_vertex_count=int(target_vertex_count),
            posy=4, rosy=4,
            pure_quad=bool(pure_quad),
            crease_angle=float(crease_angle),
            smooth_iterations=int(smooth_iterations),
            deterministic=bool(deterministic),
            align_to_boundaries=bool(align_to_boundaries),
        )
        lib = "pyinstantmeshes"
    elif method == "decimate":
        # Decimation is face-count-driven; derive a face target from the vertex
        # target (~2 faces per vert for a triangle mesh).
        target_faces = max(4, int(target_vertex_count) * 2)
        if target_faces >= in_f:
            return {
                "method": "decimate", "changed": False,
                "in_verts": in_v, "out_verts": in_v,
                "in_faces": in_f, "out_faces": in_f,
                "note": f"already below target ({in_f} faces <= {target_faces})",
            }
        out_v, out_f = decimate_quadric(vertices, faces, target_face_count=target_faces)
        lib = "fast-simplification"

    mesh.vertices = out_v
    mesh.faces = out_f
    mesh.uvs = regenerate_projective_uvs(
        out_v,
        view_matrix=view_matrix,
        fx=float(fx), fy=float(fy),
        cx=cxv, cy=cyv,
        image_width=int(image_width), image_height=int(image_height),
    )
    return {
        "method": method, "changed": True,
        "in_verts": in_v, "out_verts": int(len(out_v)),
        "in_faces": in_f, "out_faces": int(len(out_f)),
        "note": f"{lib} retopo → {len(out_v)} verts / {len(out_f)} faces "
                f"(UVs regenerated)",
    }
