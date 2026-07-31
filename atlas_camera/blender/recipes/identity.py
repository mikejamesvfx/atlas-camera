"""Round-trip probe: hand a mesh to Blender and get it back untouched.

No modifiers, no bpy.ops on geometry. The only question is whether Atlas can
build a mesh in Blender, read it straight back, and still own the vertex order.
Everything in Phase 2 stacks on top of this, so if it drifts here it drifts
everywhere.

EXPECTED TOLERANCE: Blender stores vertex coordinates as float32. At a 100 m
scene scale that is ~1e-5 m of representation error, NOT float64 equality.
Budgeting for exact equality fails this probe for entirely the wrong reason.

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


def main():
    ex = _exchange_dir()
    stage = "load"
    try:
        with np.load(f"{ex}/in.npz") as data:
            verts = np.asarray(data["patch_vertices"], dtype=np.float64)
            faces = np.asarray(data["patch_faces"], dtype=np.int64)

        stage = "build"
        bpy.ops.wm.read_factory_settings(use_empty=True)
        mesh = bpy.data.meshes.new("atlas_identity")
        mesh.from_pydata([tuple(v) for v in verts], [],
                         [tuple(f) for f in faces])
        mesh.update()

        stage = "readback"
        # foreach_get, not a Python loop: it is the only route that preserves
        # Blender's internal ordering exactly, which is the whole point of not
        # using an importer.
        out_v = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", out_v)
        out_v = out_v.reshape(-1, 3)

        n_tris = len(mesh.loop_triangles)
        if n_tris == 0:
            mesh.calc_loop_triangles()
            n_tris = len(mesh.loop_triangles)
        out_f = np.empty(n_tris * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get("vertices", out_f)
        out_f = out_f.reshape(-1, 3)

        stage = "write"
        np.savez(f"{ex}/out.npz", vertices=out_v, faces=out_f.astype(np.int32),
                 snapped=np.ones(len(out_v), dtype=bool))
        with open(f"{ex}/report.json", "w", encoding="utf-8") as fh:
            json.dump({
                "recipe": "identity",
                "blender_version": ".".join(str(v) for v in bpy.app.version),
                "numpy": np.__version__,
                "verts_in": int(len(verts)), "verts_out": int(len(out_v)),
                "faces_in": int(len(faces)), "faces_out": int(len(out_f)),
                "seconds": round(time.time() - _T0, 3),
            }, fh, indent=1)
        print("ATLAS_RECIPE_OK identity")
    except Exception as exc:  # noqa: BLE001
        with open(f"{ex}/error.json", "w", encoding="utf-8") as fh:
            json.dump({"stage": stage, "type": type(exc).__name__,
                       "message": str(exc),
                       "traceback": traceback.format_exc()}, fh, indent=1)
        sys.exit(3)


main()
