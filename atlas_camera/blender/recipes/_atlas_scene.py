"""Shared helpers for the measured-primitives recipes (massing / export_meshes).

Runs inside Blender's interpreter — imports bpy. Recipes pull it in with a
sys.path insert of their own directory (see the top of massing.py); Atlas never
imports it. Everything here was written against the same Blender 5.2 API the
organic_fill recipe probed live.

Wire format it owns (mirrors atlas_camera/blender/exchange.py, which is the
Atlas-side reader — keep the two in step):

  seed.json / seed.npz          Atlas -> Blender (camera, reference prims, shapes)
  out_meshes.npz / .json        Blender -> Atlas (mesh_{i}_vertices/faces + tags)

Collections:
  atlas_reference   seed geometry, hidden from render, NEVER exported back
  atlas_out         everything the recipe (or the artist) wants Atlas to import
"""
import json
import sys
import time
import traceback

import bpy  # noqa: E402
import numpy as np  # noqa: E402

REFERENCE_COLLECTION = "atlas_reference"
OUT_COLLECTION = "atlas_out"
BLENDER_SENSOR_MM = 36.0


def exchange_dir():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--exchange" not in argv:
        raise RuntimeError("missing --exchange <dir>")
    return argv[argv.index("--exchange") + 1]


def collection(name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def object_from_arrays(name, verts, faces, col):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(map(float, v)) for v in verts], [],
                     [tuple(map(int, f)) for f in faces])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    col.objects.link(obj)
    return obj


def camera_from_seed(cam):
    """Place a Blender camera matching the seed's recovered camera.

    Blender's camera looks down its LOCAL -Z with +Y up, exactly like Atlas's,
    so `matrix_world_blender` (already T-rotated by the writer) is used as-is.
    Intrinsics: sensor_fit HORIZONTAL, lens from fx; principal-point offset as
    Blender shift (fraction of the horizontal sensor size; the sign convention
    follows BlenderProc's set_intrinsics_from_K_matrix). The camera exists for
    the artist / GUI round-trip and for renders — Atlas never reads it back.
    """
    W = int(cam["image_width"]); H = int(cam["image_height"])
    fx = float(cam["fx"]); fy = float(cam["fy"])
    cx = float(cam["cx"]); cy = float(cam["cy"])
    data = bpy.data.cameras.new("atlas_camera")
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = BLENDER_SENSOR_MM
    data.lens = fx * BLENDER_SENSOR_MM / float(W)
    data.shift_x = -(cx - W / 2.0) / float(W)
    data.shift_y = (cy - H / 2.0) / float(W)
    obj = bpy.data.objects.new("atlas_camera", data)
    bpy.context.scene.collection.objects.link(obj)
    m = cam["matrix_world_blender"]
    # mathutils.Matrix, NOT a list of lists: assigning a nested list to
    # matrix_world is silently ignored (found live 2026-08-16 — the camera sat
    # at the origin in every saved scene).
    from mathutils import Matrix
    obj.matrix_world = Matrix([[float(x) for x in row] for row in m])
    bpy.context.view_layer.update()
    scene = bpy.context.scene
    scene.camera = obj
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = float(fx / fy) if fy > 0 else 1.0
    return obj


#: MIRRORS atlas_camera.blender.exchange.EXCHANGE_VERSION — the recipes run
#: inside Blender and cannot import atlas_camera. Pinned by
#: tests/test_blender_measured_bridge.py so the two cannot drift.
EXCHANGE_VERSION = 1


def load_seed(ex):
    """Returns (seed_json, prims, cloud) — cloud is (N,3) Blender-space or None."""
    with open(f"{ex}/seed.json", encoding="utf-8") as fh:
        seed = json.load(fh)
    # Format identity, checked HERE too: this runs inside Blender, where the
    # exchange dir is the only contract with the pack that wrote it, and a
    # stale lane from an older pack must refuse rather than be misread.
    got = seed.get("atlas_exchange_version")
    if got != EXCHANGE_VERSION:
        raise RuntimeError(
            "seed.json: exchange format version %r, this recipe reads %d. "
            "Re-run the massing node to rewrite the exchange directory."
            % (got, EXCHANGE_VERSION))
    prims = []
    cloud = None
    with np.load(f"{ex}/seed.npz") as data:
        if "cloud_points" in data:
            cloud = np.asarray(data["cloud_points"], dtype=np.float64).reshape(-1, 3)
        for tags in seed.get("primitives") or []:
            i = int(tags["index"])
            vk, fk = f"prim_{i}_vertices", f"prim_{i}_faces"
            if vk not in data or fk not in data:
                continue
            prims.append((dict(tags),
                          np.asarray(data[vk], dtype=np.float64).reshape(-1, 3),
                          np.asarray(data[fk], dtype=np.int64).reshape(-1, 3)))
    return seed, prims, cloud


def cloud_object(name, points, col):
    """Vertex-only mesh from an (N,3) array — the measured point cloud, for
    snapping and reference. No faces, so it never renders or exports."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(map(float, p)) for p in points], [], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    col.objects.link(obj)
    obj.hide_render = True
    return obj


def boundary_loop(faces):
    """Ordered outer boundary loop of a triangle mesh (edges used once).

    Returns a list of vertex indices, or None if the mesh has no single loop.
    """
    from collections import defaultdict
    count = defaultdict(int)
    for a, b, c in faces:
        for e in ((a, b), (b, c), (c, a)):
            count[tuple(sorted(e))] += 1
    edges = [e for e, n in count.items() if n == 1]
    if not edges:
        return None
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    if any(len(v) != 2 for v in adj.values()):
        return None
    start = edges[0][0]
    loop = [start]
    prev, cur = None, start
    while True:
        nxt = [n for n in adj[cur] if n != prev]
        if not nxt:
            return None
        nxt = nxt[0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
        if len(loop) > len(edges):
            return None
    return loop


def prism_from_outline(outline_pts, cap_faces, cap_verts, offset):
    """Extrude a planar polygon mesh along ``offset`` into a closed prism.

    ``cap_verts``/``cap_faces``: the polygon's own triangulation (bottom cap).
    ``outline_pts``: ordered indices into cap_verts along the boundary.
    Returns (vertices, faces) as numpy arrays; winding puts the bottom cap
    facing -offset and the top facing +offset.
    """
    v0 = np.asarray(cap_verts, dtype=np.float64)
    n = len(v0)
    top = v0 + np.asarray(offset, dtype=np.float64)[None, :]
    verts = np.vstack([v0, top])
    faces = []
    for a, b, c in cap_faces:
        faces.append((a, c, b))                  # bottom, flipped
        faces.append((a + n, b + n, c + n))      # top
    m = len(outline_pts)
    for i in range(m):
        a = int(outline_pts[i]); b = int(outline_pts[(i + 1) % m])
        faces.append((a, b, b + n))
        faces.append((a, b + n, a + n))
    return verts, np.asarray(faces, dtype=np.int64)


def evaluated_triangles(obj):
    """World-space (evaluated) vertices + triangle indices of a mesh object."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    nv = len(me.vertices)
    vbuf = np.empty(nv * 3, dtype=np.float64)
    me.vertices.foreach_get("co", vbuf)
    verts = vbuf.reshape(-1, 3)
    nt = len(me.loop_triangles)
    fbuf = np.empty(nt * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", fbuf)
    faces = fbuf.reshape(-1, 3)
    mw = np.asarray([[float(x) for x in row] for row in obj.matrix_world],
                    dtype=np.float64)
    verts_w = verts @ mw[:3, :3].T + mw[:3, 3]
    ev.to_mesh_clear()
    return verts_w, faces


def write_out_meshes(ex, objects, *, recipe, extra_report=None):
    """Write out_meshes.npz / out_meshes.json / report.json from mesh objects."""
    arrays = {}
    meta = []
    for i, obj in enumerate(objects):
        v, f = evaluated_triangles(obj)
        arrays[f"mesh_{i}_vertices"] = v
        arrays[f"mesh_{i}_faces"] = f.astype(np.int32)
        tags = {"index": i, "name": obj.name,
                "n_vertices": int(len(v)), "n_faces": int(len(f))}
        for k in ("atlas_kind", "atlas_source", "atlas_height_m",
                  "atlas_from_primitive", "atlas_paint"):
            if k in obj:
                val = obj[k]
                tags[k.replace("atlas_", "")] = (
                    val if isinstance(val, (str, int, float, bool)) else str(val))
        meta.append(tags)
    if not arrays:
        arrays["_empty"] = np.zeros((0,), dtype=np.float64)
    np.savez(f"{ex}/out_meshes.npz", **arrays)
    with open(f"{ex}/out_meshes.json", "w", encoding="utf-8") as fh:
        json.dump({"recipe": recipe, "meshes": meta,
                   "blender_version": ".".join(str(v) for v in bpy.app.version)},
                  fh, indent=1)
    rep = {"recipe": recipe, "meshes_out": len(meta),
           "blender_version": ".".join(str(v) for v in bpy.app.version)}
    rep.update(extra_report or {})
    with open(f"{ex}/report.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    return meta


def fail(ex, stage, exc):
    try:
        with open(f"{ex}/error.json", "w", encoding="utf-8") as fh:
            json.dump({"stage": stage, "type": type(exc).__name__,
                       "message": str(exc),
                       "traceback": traceback.format_exc()}, fh, indent=1)
    except Exception:  # noqa: BLE001
        pass
    sys.exit(3)


def now():
    return time.time()
