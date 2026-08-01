"""Snap an already-closed patch back onto measured surface. That is all.

WHY THIS IS THE WHOLE RECIPE. Phase 0 measured the division of labour and I
initially ignored my own result. Atlas's numpy `core.mesh_voxel.voxel_remesh`
closes every interior tear on real organic plates in ~3.4s; what it cannot do is
PLACE the result, drifting 4-5x the median edge off measured surface. So closing
was never the missing piece.

The first recipe tried to make Blender close the tear too, with solidify + voxel
remesh. Measured: it does not work, and cannot. Solidify thickens PERPENDICULAR
to the surface and walls the hole's rim, turning a hole in a sheet into a tube
through a slab; volumetric remesh then faithfully preserves the tube. Sweeping
voxel size on a 4-unit hole, the hole only vanished at voxel=4.0 — where the
patch collapsed from 66 vertices to 16. That is obliteration, not repair.
numpy succeeds at the same job only because it works in DEPTH-IMAGE space,
where a hole is a 2D region to flood-fill rather than a topological feature.

So Blender does exactly one thing here: BVH shrinkwrap, which numpy has no
answer for without shipping its own acceleration structure.

Runs inside Blender's interpreter. Never imported by Atlas — it imports bpy.
"""
import json
import sys
import time
import traceback

_T0 = time.time()

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


def _positions(obj):
    buf = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", buf)
    return buf.reshape(-1, 3)


def _evaluated_positions(obj):
    """Positions WITH modifiers, without applying them — so the before-state
    survives long enough to measure what moved."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    buf = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", buf)
    out = buf.reshape(-1, 3).copy()
    ev.to_mesh_clear()
    return out


def main():
    ex = _exchange_dir()
    stage = "load"
    times = {}
    try:
        with np.load(f"{ex}/in.npz") as data:
            pv = np.asarray(data["patch_vertices"], dtype=np.float64)
            pf = np.asarray(data["patch_faces"], dtype=np.int64)
            tv = np.asarray(data["target_vertices"], dtype=np.float64)
            tf = np.asarray(data["target_faces"], dtype=np.int64)
        with open(f"{ex}/params.json", encoding="utf-8") as fh:
            p = json.load(fh)
        limit = float(p["shrinkwrap_limit_m"])
        method = str(p.get("wrap_method", "NEAREST_SURFACEPOINT"))

        t = time.time()
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.context.scene.render.threads_mode = "FIXED"
        bpy.context.scene.render.threads = 1
        target = _object_from_arrays("atlas_target", tv, tf)
        patch = _object_from_arrays("atlas_patch", pv, pf)
        times["build"] = round(time.time() - t, 3)

        stage = "shrinkwrap"
        t = time.time()
        before = _positions(patch)
        sw = patch.modifiers.new(name="atlas_shrinkwrap", type="SHRINKWRAP")
        sw.target = target
        sw.wrap_method = method
        sw.wrap_mode = "ON_SURFACE"
        if method == "PROJECT":
            # No axis flags set -> project along the vertex normal, which is
            # what lets a vertex over a filled tear (nothing to hit within
            # `limit`) stay where the fill put it.
            sw.use_project_x = False
            sw.use_project_y = False
            sw.use_project_z = False
            sw.use_negative_direction = True
            sw.use_positive_direction = True
            sw.project_limit = limit
        after = _evaluated_positions(patch)
        times["shrinkwrap"] = round(time.time() - t, 3)

        if len(after) != len(before):
            raise RuntimeError(
                f"shrinkwrap changed the vertex count ({len(before)} -> "
                f"{len(after)}); it must only MOVE vertices")

        # NEAREST has no native distance cap — ``project_limit`` is
        # PROJECT-only in Blender's API — so enforce the limit here or the
        # caller's shrinkwrap_limit_m is a NO-OP for the default wrap method
        # and fill vertices snap onto unrelated geometry metres away (found
        # live: a 4.11 m max move against a sub-metre limit). Same semantic
        # as PROJECT's cap: a vertex with no surface within the limit stays
        # where the remesh put it, and reads as un-snapped fill below.
        if method != "PROJECT" and limit > 0:
            over = np.linalg.norm(after - before, axis=1) > limit
            after[over] = before[over]

        moved = np.linalg.norm(after - before, axis=1)
        # A vertex that moved found measured surface. One that did not is
        # invented fill left where numpy put it — the caller's drift gate keys
        # off exactly this distinction.
        snapped = moved > 1e-9

        stage = "write"
        np.savez(f"{ex}/out.npz", vertices=after,
                 faces=np.asarray(pf, dtype=np.int32), snapped=snapped)
        with open(f"{ex}/report.json", "w", encoding="utf-8") as fh:
            json.dump({
                "recipe": "shrinkwrap",
                "blender_version": ".".join(str(v) for v in bpy.app.version),
                "wrap_method": method, "shrinkwrap_limit_m": limit,
                "verts": int(len(after)), "faces": int(len(pf)),
                "snapped_fraction": round(float(snapped.mean()), 4)
                if len(snapped) else 0.0,
                "moved_median_m": round(float(np.median(moved[snapped])), 6)
                if snapped.any() else 0.0,
                "moved_max_m": round(float(moved.max()), 6) if len(moved) else 0.0,
                "stage_seconds": times,
                "seconds": round(time.time() - _T0, 3),
            }, fh, indent=1)
        if p.get("keep_debug_blend"):
            bpy.ops.wm.save_as_mainfile(filepath=f"{ex}/debug.blend")
        print("ATLAS_RECIPE_OK shrinkwrap")
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
