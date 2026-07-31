"""Phase 0 probe: what does THIS Blender actually offer?

Runs inside Blender's own interpreter. Settles the questions that would
otherwise be discovered halfway through writing the real recipe:

* is numpy available? (the NPZ exchange format depends on it — without it the
  design falls back to raw .bin + a JSON header, which is a two-hour change now
  and a two-day change later)
* do the voxel-remesh and shrinkwrap attribute names still exist? The recipe was
  designed from 4.x knowledge against a 5.2 install, and a silently renamed
  attribute is the single largest correctness risk in the plan.
* how long does headless startup actually cost?

NEVER imported by Atlas. It imports bpy, which only exists inside Blender —
which is why recipes/ is package DATA and deliberately not a package.

    blender --background --factory-startup --python probe_api.py -- --out report.json
"""
import json
import sys
import time

_T0 = time.time()

import bpy  # noqa: E402  (only importable inside Blender)


def _argv_after_ddash():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def main():
    args = _argv_after_ddash()
    out = args[args.index("--out") + 1] if "--out" in args else "probe_report.json"

    report = {
        "blender_version": ".".join(str(v) for v in bpy.app.version),
        "blender_version_string": bpy.app.version_string,
        "python_version": sys.version.split()[0],
        "startup_seconds": round(time.time() - _T0, 3),
    }

    # --- numpy: does the exchange format survive? --------------------------
    try:
        import numpy as np
        report["numpy"] = np.__version__
        report["numpy_savez"] = hasattr(np, "savez")
    except Exception as exc:  # noqa: BLE001
        report["numpy"] = None
        report["numpy_error"] = f"{type(exc).__name__}: {exc}"

    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = bpy.data.meshes.new("probe")
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    mesh.update()
    obj = bpy.data.objects.new("probe", mesh)
    bpy.context.collection.objects.link(obj)

    # --- the attribute names the recipe depends on -------------------------
    wanted_mesh_attrs = ("remesh_mode", "remesh_voxel_size",
                         "remesh_voxel_adaptivity", "use_remesh_fix_poles")
    report["mesh_attrs"] = {a: hasattr(mesh, a) for a in wanted_mesh_attrs}
    if hasattr(mesh, "remesh_mode"):
        try:
            report["remesh_modes"] = [
                i.identifier for i in
                mesh.bl_rna.properties["remesh_mode"].enum_items]
        except Exception as exc:  # noqa: BLE001
            report["remesh_modes_error"] = str(exc)

    for mod_type, attrs in (
        ("SHRINKWRAP", ("target", "wrap_method", "wrap_mode", "project_limit",
                        "use_negative_direction", "use_positive_direction",
                        "use_project_x", "use_project_y", "use_project_z")),
        ("SOLIDIFY", ("thickness", "offset", "use_even_offset")),
    ):
        try:
            m = obj.modifiers.new(name=f"probe_{mod_type}", type=mod_type)
            report[f"{mod_type.lower()}_attrs"] = {a: hasattr(m, a) for a in attrs}
            if mod_type == "SHRINKWRAP":
                for enum_prop in ("wrap_method", "wrap_mode"):
                    try:
                        report[f"shrinkwrap_{enum_prop}_values"] = [
                            i.identifier for i in
                            m.bl_rna.properties[enum_prop].enum_items]
                    except Exception:  # noqa: BLE001
                        pass
            obj.modifiers.remove(m)
        except Exception as exc:  # noqa: BLE001
            report[f"{mod_type.lower()}_error"] = f"{type(exc).__name__}: {exc}"

    # --- the ops the recipe calls ------------------------------------------
    report["ops"] = {
        "object.voxel_remesh": hasattr(bpy.ops.object, "voxel_remesh"),
        "object.modifier_apply": hasattr(bpy.ops.object, "modifier_apply"),
        "wm.save_as_mainfile": hasattr(bpy.ops.wm, "save_as_mainfile"),
    }
    try:
        from mathutils.bvhtree import BVHTree
        report["bvhtree"] = hasattr(BVHTree, "FromPolygons")
    except Exception as exc:  # noqa: BLE001
        report["bvhtree_error"] = f"{type(exc).__name__}: {exc}"

    report["total_seconds"] = round(time.time() - _T0, 3)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("ATLAS_PROBE_OK " + out)


main()
