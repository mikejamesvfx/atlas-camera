"""Close a torn collar: solidify -> voxel remesh -> shrinkwrap -> cull.

The division of labour this whole design rests on: **voxel remesh CLOSES the
tear, shrinkwrap PLACES the result back on measured surface.** Measured on three
organic plates, Atlas's own numpy voxel_remesh already closes every tear in
~3.4s — and then sits 4-5x the median edge off the measured geometry (p95
0.86-0.93m against median edges of 0.17-0.23m). Closing was never the problem.
Snapping back is what Blender is here for.

Every attribute used below was probed live against Blender 5.2.0 LTS before
being written (recipes/probe_api.py) rather than taken from documentation.

Runs inside Blender's interpreter. Never imported by Atlas — it imports bpy.
"""
import json
import sys
import time
import traceback

_T0 = time.time()

import bmesh  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402

_TIMES = {}


def _exchange_dir():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--exchange" not in argv:
        raise RuntimeError("missing --exchange <dir>")
    return argv[argv.index("--exchange") + 1]


def _mark(name, t0):
    _TIMES[name] = round(time.time() - t0, 3)
    return time.time()


def _object_from_arrays(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _positions(obj):
    n = len(obj.data.vertices)
    buf = np.empty(n * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", buf)
    return buf.reshape(-1, 3)


def _evaluated_positions(obj):
    """Vertex positions WITH modifiers applied, without applying them.

    Needed to measure what shrinkwrap actually moved before committing it —
    after apply, the before-state is gone.
    """
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    n = len(me.vertices)
    buf = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", buf)
    out = buf.reshape(-1, 3).copy()
    ev.to_mesh_clear()
    return out


def _apply(obj, modifier_name):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier_name)


def _triangles(obj):
    me = obj.data
    me.calc_loop_triangles()
    n = len(me.loop_triangles)
    buf = np.empty(n * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", buf)
    return buf.reshape(-1, 3)


def main():
    ex = _exchange_dir()
    stage = "load"
    try:
        with np.load(f"{ex}/in.npz") as data:
            pv = np.asarray(data["patch_vertices"], dtype=np.float64)
            pf = np.asarray(data["patch_faces"], dtype=np.int64)
            tv = np.asarray(data["target_vertices"], dtype=np.float64)
            tf = np.asarray(data["target_faces"], dtype=np.int64)
            cam = np.asarray(data["camera_position"], dtype=np.float64).ravel()
        with open(f"{ex}/params.json", encoding="utf-8") as fh:
            p = json.load(fh)
        voxel = float(p["voxel_size_m"])
        adaptivity = float(p.get("adaptivity", 0.0))
        solidify_vox = float(p.get("solidify_voxels", 3.0))
        limit = float(p.get("shrinkwrap_limit_m", 2.0 * voxel))

        t = time.time()
        bpy.ops.wm.read_factory_settings(use_empty=True)
        # OpenVDB tiling is thread-count sensitive; pinning makes two runs of
        # the same input byte-identical, which is what lets determinism be a
        # test rather than a hope.
        bpy.context.scene.render.threads_mode = "FIXED"
        bpy.context.scene.render.threads = 1
        target = _object_from_arrays("atlas_target", tv, tf)
        patch = _object_from_arrays("atlas_patch", pv, pf)
        t = _mark("build", t)

        # --- 1 SOLIDIFY -------------------------------------------------
        # Voxel remesh needs a volume; an open sheet remeshes to a thin shell
        # anyway, so thicken deliberately and controllably. Same reasoning
        # core/mesh_voxel.voxel_remesh encodes as `thickness_vox`.
        stage = "solidify"
        m = patch.modifiers.new(name="atlas_solidify", type="SOLIDIFY")
        m.thickness = solidify_vox * voxel
        m.offset = 0.0
        m.use_even_offset = True
        _apply(patch, "atlas_solidify")
        t = _mark("solidify", t)

        # --- 2 VOXEL REMESH ---------------------------------------------
        # Closes the tear: once thickened, its two lips lie within one voxel.
        stage = "voxel_remesh"
        patch.data.remesh_mode = "VOXEL"
        patch.data.remesh_voxel_size = voxel
        patch.data.remesh_voxel_adaptivity = adaptivity
        patch.data.use_remesh_fix_poles = True
        bpy.context.view_layer.objects.active = patch
        bpy.ops.object.voxel_remesh()
        t = _mark("voxel_remesh", t)
        remeshed = _positions(patch)

        # --- 3 SHRINKWRAP onto MEASURED surface -------------------------
        # The step that earns Blender its place. PROJECT along vertex normals
        # with a limit, so a vertex over the tear (with nothing to hit) is left
        # where the remesh put it, while a vertex over real surface snaps back.
        stage = "shrinkwrap"
        sw = patch.modifiers.new(name="atlas_shrinkwrap", type="SHRINKWRAP")
        sw.target = target
        sw.wrap_method = "PROJECT"
        sw.wrap_mode = "ON_SURFACE"
        sw.use_project_x = False
        sw.use_project_y = False
        sw.use_project_z = False
        sw.use_negative_direction = True
        sw.use_positive_direction = True
        sw.project_limit = limit
        wrapped = _evaluated_positions(patch)
        # Measured BEFORE apply — afterwards the before-state is gone. A vertex
        # that moved found measured surface; one that did not is invented fill,
        # and the caller needs that distinction for its drift gate.
        snapped = (np.linalg.norm(wrapped - remeshed, axis=1) > 1e-9
                   if len(wrapped) == len(remeshed)
                   else np.zeros(len(wrapped), dtype=bool))
        _apply(patch, "atlas_shrinkwrap")
        t = _mark("shrinkwrap", t)

        # --- 4 CULL THE BACK SHEET --------------------------------------
        # Solidify made two surfaces; only the camera-facing one is wanted.
        # Deterministic ray-cast, no bpy.ops, no normal-direction guessing.
        stage = "cull"
        verts = _positions(patch)
        tris = _triangles(patch)
        kept = tris
        if len(tris) and np.isfinite(cam).all():
            bvh = BVHTree.FromPolygons([tuple(v) for v in verts],
                                       [tuple(f) for f in tris],
                                       all_triangles=True)
            centroids = verts[tris].mean(axis=1)
            keep = np.zeros(len(tris), dtype=bool)
            for i, c in enumerate(centroids):
                d = c - cam
                n = np.linalg.norm(d)
                if n < 1e-9:
                    continue
                hit = bvh.ray_cast(tuple(cam), tuple(d / n))
                if hit[2] is not None and int(hit[2]) == i:
                    keep[i] = True
            if keep.any():
                kept = tris[keep]
        t = _mark("cull", t)

        # Re-pack: keep only vertices the surviving faces use.
        stage = "pack"
        used = np.unique(kept)
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        out_v = verts[used]
        out_f = remap[kept]
        out_snapped = (snapped[used] if len(snapped) == len(verts)
                       else np.zeros(len(out_v), dtype=bool))

        stage = "write"
        np.savez(f"{ex}/out.npz", vertices=out_v,
                 faces=out_f.astype(np.int32), snapped=out_snapped)
        with open(f"{ex}/report.json", "w", encoding="utf-8") as fh:
            json.dump({
                "recipe": "organic_fill",
                "blender_version": ".".join(str(v) for v in bpy.app.version),
                "voxel_size_m": voxel, "shrinkwrap_limit_m": limit,
                "verts_in": int(len(pv)), "faces_in": int(len(pf)),
                "verts_remeshed": int(len(remeshed)),
                "verts_out": int(len(out_v)), "faces_out": int(len(out_f)),
                "faces_culled": int(len(tris) - len(kept)),
                "snapped_fraction": round(float(out_snapped.mean()), 4)
                if len(out_snapped) else 0.0,
                "stage_seconds": _TIMES,
                "seconds": round(time.time() - _T0, 3),
            }, fh, indent=1)
        if p.get("keep_debug_blend"):
            bpy.ops.wm.save_as_mainfile(filepath=f"{ex}/debug.blend")
        print("ATLAS_RECIPE_OK organic_fill")
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
