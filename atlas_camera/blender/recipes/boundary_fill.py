"""Fill explicitly selected mesh boundary loops without remeshing.

Runs inside Blender only.  The native path uses Blender's public BMesh fill
operator.  The optional comparison path invokes the separately installed
``fill_mesh`` extension by its public operator; it neither imports Atlas code
nor copies the extension's GPL implementation.
"""
import importlib
import json
import sys
import time
import traceback

_T0 = time.time()

import bmesh  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402


def _exchange_dir():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--exchange" not in argv:
        raise RuntimeError("missing --exchange <dir>")
    return argv[argv.index("--exchange") + 1]


def _object_from_arrays(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _triangles(obj):
    me = obj.data
    me.calc_loop_triangles()
    buf = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", buf)
    return buf.reshape(-1, 3)


def _edges_for_loops(bm, loops):
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    out = []
    seen = set()
    for loop in loops:
        for a, b in zip(loop, loop[1:] + loop[:1]):
            edge = bm.edges.get((bm.verts[int(a)], bm.verts[int(b)]))
            if edge is None:
                raise RuntimeError(
                    f"selected loop edge ({a}, {b}) is not a mesh edge")
            key = tuple(sorted((int(a), int(b))))
            if key not in seen:
                seen.add(key)
                out.append(edge)
    return out


def _native_fill(obj, loops):
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        edges = _edges_for_loops(bm, loops)
        before = len(bm.faces)
        bmesh.ops.triangle_fill(bm, edges=edges, use_beauty=True)
        bm.to_mesh(obj.data)
        obj.data.update()
        return len(bm.faces) - before
    finally:
        bm.free()


def _addon_fill(obj, loops):
    """Run the user-installed Fill Mesh operator on only the selected rims."""
    try:
        addon = importlib.import_module("bl_ext.user_default.fill_mesh")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Fill Mesh add-on is not installed for this Blender user. Install "
            "it from Extensions, or select backend=native.") from exc
    if "fill_mesh_select" not in dir(bpy.ops.fillmesh):
        addon.register()

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    for edge in bm.edges:
        edge.select = False
    for edge in _edges_for_loops(bm, loops):
        edge.select = True
    bmesh.update_edit_mesh(obj.data)
    before = len(bm.faces)
    result = bpy.ops.fillmesh.fill_mesh_select()
    if "FINISHED" not in result:
        raise RuntimeError(f"Fill Mesh operator returned {result}")
    bm = bmesh.from_edit_mesh(obj.data)
    created = len(bm.faces) - before
    bpy.ops.object.mode_set(mode="OBJECT")
    return created


def main():
    ex = _exchange_dir()
    stage = "load"
    try:
        with np.load(f"{ex}/in.npz") as data:
            vertices = np.asarray(data["patch_vertices"], dtype=np.float64)
            faces = np.asarray(data["patch_faces"], dtype=np.int64)
        with open(f"{ex}/params.json", encoding="utf-8") as fh:
            params = json.load(fh)
        loops = [[int(v) for v in loop] for loop in params["selected_loops"]]
        backend = str(params.get("backend", "native"))

        bpy.ops.wm.read_factory_settings(use_empty=True)
        obj = _object_from_arrays("atlas_boundary_fill", vertices, faces)
        stage = "fill"
        if backend == "native":
            created = _native_fill(obj, loops)
        elif backend == "fill_mesh_addon":
            created = _addon_fill(obj, loops)
        else:
            raise ValueError(f"unknown backend {backend!r}")

        stage = "write"
        np.savez(f"{ex}/out.npz", vertices=np.asarray([v.co[:] for v in obj.data.vertices]),
                 faces=_triangles(obj).astype(np.int32))
        with open(f"{ex}/report.json", "w", encoding="utf-8") as fh:
            json.dump({
                "recipe": "boundary_fill", "backend": backend,
                "blender_version": ".".join(str(v) for v in bpy.app.version),
                "selected_loops": len(loops), "faces_created": int(created),
                "faces_in": int(len(faces)), "faces_out": int(len(_triangles(obj))),
                "seconds": round(time.time() - _T0, 3),
            }, fh, indent=1)
        print("ATLAS_RECIPE_OK boundary_fill")
    except Exception as exc:  # noqa: BLE001
        try:
            with open(f"{ex}/error.json", "w", encoding="utf-8") as fh:
                json.dump({"stage": stage, "type": type(exc).__name__,
                           "message": str(exc),
                           "traceback": traceback.format_exc()}, fh, indent=1)
        except Exception:  # noqa: BLE001
            pass
        sys.exit(3)


main()
