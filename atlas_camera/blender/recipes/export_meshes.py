"""Hand every mesh under the `atlas_out` collection back to Atlas.

The second half of the GUI round-trip: Blender is launched WITH the .blend the
artist edited (`runner.run_recipe(..., blend_file=...)`), this script runs on
it headless and writes out_meshes.npz / out_meshes.json for
`AtlasBlenderImportMeshes`.

Selection rule: mesh objects in `atlas_out` (recursively). If that collection
does not exist — a .blend the artist built from scratch — every visible mesh
object NOT under `atlas_reference` is exported, and the report says so.
Cameras, lights, empties are ignored. Modifiers are applied in the evaluated
copy only (the .blend is not touched); world transforms are baked in.

Runs inside Blender's interpreter. Never imported by Atlas — it imports bpy.
"""
import os
import sys

# APPEND, never insert(0): this runs inside Blender's interpreter, and
# prepending would let a recipe filename shadow a stdlib or site-packages
# module for the whole process. `_atlas_scene` is the only name we need
# from here, and appending still finds it.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402

import _atlas_scene as S  # noqa: E402


def _meshes_in(col):
    seen = []
    def walk(c):
        for o in c.objects:
            if o.type == "MESH" and o not in seen:
                seen.append(o)
        for ch in c.children:
            walk(ch)
    walk(col)
    return seen


def main():
    ex = S.exchange_dir()
    stage = "select"
    try:
        out_col = bpy.data.collections.get(S.OUT_COLLECTION)
        rule = "atlas_out"
        if out_col is not None:
            objs = _meshes_in(out_col)
        else:
            rule = "all_meshes_except_reference"
            ref = bpy.data.collections.get(S.REFERENCE_COLLECTION)
            excluded = set(o.name for o in _meshes_in(ref)) if ref else set()
            objs = [o for o in bpy.data.objects
                    if o.type == "MESH" and o.name not in excluded
                    and not o.hide_render]
        stage = "write"
        S.write_out_meshes(ex, objs, recipe="export_meshes",
                           extra_report={"selection_rule": rule,
                                         "blend_file": bpy.data.filepath})
        print("ATLAS_RECIPE_OK export_meshes")
    except Exception as exc:  # noqa: BLE001
        S.fail(ex, stage, exc)


main()
