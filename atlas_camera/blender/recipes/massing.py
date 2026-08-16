"""Measured primitives: seed a Blender scene from the metric solve, extrude
footprints, save a .blend for the artist, hand the meshes back to Atlas.

What this recipe MODELS (all in metres, in the recovered camera's world):
  * a ground plane at Atlas Y=0 (Blender Z=0), `ground_extent_m` square,
    centred under the camera;
  * every viewport-drawn polygon that lies FLAT (a footprint) extruded UP by
    its `height_m` tag or `default_height_m`;
  * every viewport-drawn polygon that stands VERTICAL (a facade) thickened
    into a slab of `wall_thickness_m` AWAY from the camera (its front face
    stays exactly where the artist drew it against the photo);
  * block-massing boxes copied through as volumes.
Reference primitives (relief mesh, derived planes) are linked under
`atlas_reference`, hidden from render, and never exported.

Extrusion is done in numpy inside Blender rather than with bmesh operators so
the exact same code path builds the meshes whether this runs headless or the
artist re-runs it in the GUI — and so the result is deterministic.

The scene is saved as `scene.blend` (unless params.save_blend is false). That
file IS the GUI round-trip: open it, model more under `atlas_out`, then run
`export_meshes.py` on it (AtlasBlenderImportMeshes does that for you) and the
new meshes come back into the solve.

Runs inside Blender's interpreter. Never imported by Atlas — it imports bpy.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import _atlas_scene as S  # noqa: E402

FLAT_NZ = 0.7   # |normal.z| above this in Blender = a footprint (ground-lying)


def _polygon_normal(pts):
    """Newell's method — robust for any simple planar polygon."""
    n = np.zeros(3)
    m = len(pts)
    for i in range(m):
        a, b = pts[i], pts[(i + 1) % m]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])


def main():
    ex = S.exchange_dir()
    stage = "load"
    try:
        seed, prims, cloud = S.load_seed(ex)
        p = seed.get("params") or {}
        # Ground level in ATLAS Y (metres) — measured from the MoGe pointmap
        # when the seed is a measured one, else 0. Blender Z = Atlas Y.
        ground_z = float(p.get("ground_y_m", 0.0) or 0.0)
        default_h = float(p.get("default_height_m", 3.0))
        ground_extent = float(p.get("ground_extent_m", 60.0))
        wall_t = float(p.get("wall_thickness_m", 0.3))
        source_mode = str(p.get("footprint_source", "both"))
        save_blend = bool(p.get("save_blend", True))
        make_ground = bool(p.get("ground_plane", True))

        stage = "scene"
        # RE-RUNS PRESERVE THE ARTIST'S / AGENT'S WORK. If scene.blend already
        # exists it is opened and only the recipe-owned parts are rebuilt: the
        # atlas_reference collection, the camera, and atlas_out objects tagged
        # atlas_source == blender_massing. Anything else under atlas_out (an
        # agent's or artist's models) survives. Found live 2026-08-16: a
        # re-queue rebuilt the seed and silently threw away fitted building
        # volumes; only Blender's .blend1 backup still had them.
        blend_path = f"{ex}/scene.blend"
        preserved = 0
        fp_now = str(p.get("solve_fingerprint") or "")
        if os.path.isfile(blend_path):
            bpy.ops.wm.open_mainfile(filepath=blend_path)
            fp_old = str(bpy.context.scene.get("atlas_seed_fingerprint", ""))
            if fp_old and fp_now and fp_old != fp_now:
                # A DIFFERENT solve landed in this exchange folder: nothing in
                # the old scene belongs to it. Archive, start empty (found live
                # 2026-08-16: Brooklyn building blocks rode into a coastal
                # plate because two graphs shared one folder).
                import shutil
                shutil.copyfile(blend_path, f"{ex}/scene_prev_{fp_old}.blend")
                bpy.ops.wm.read_factory_settings(use_empty=True)
                stale = True
            else:
                stale = False
        else:
            stale = False
        if os.path.isfile(blend_path) and not stale:
            ref = bpy.data.collections.get(S.REFERENCE_COLLECTION)
            if ref is not None:
                for o in list(ref.objects):
                    bpy.data.objects.remove(o, do_unlink=True)
            out_col = bpy.data.collections.get(S.OUT_COLLECTION)
            if out_col is not None:
                for o in list(out_col.objects):
                    if str(o.get("atlas_source", "")) == "blender_massing":
                        bpy.data.objects.remove(o, do_unlink=True)
                    else:
                        preserved += 1
            for o in list(bpy.data.objects):
                if o.type == "CAMERA" and o.name.startswith("atlas_camera"):
                    bpy.data.objects.remove(o, do_unlink=True)
            for m in list(bpy.data.meshes):
                if m.users == 0:
                    bpy.data.meshes.remove(m)
        elif not stale:
            bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.context.scene["atlas_seed_fingerprint"] = fp_now
        ref = S.collection(S.REFERENCE_COLLECTION)
        out = S.collection(S.OUT_COLLECTION)
        ref.hide_render = True
        cam_obj = S.camera_from_seed(seed["camera"])
        cam_pos = np.asarray(cam_obj.matrix_world.translation, dtype=np.float64)

        stage = "reference"
        n_cloud = 0
        if cloud is not None and len(cloud):
            S.cloud_object("atlas_cloud", cloud, ref)
            n_cloud = int(len(cloud))
        for tags, v, f in prims:
            name = "ref_%s" % (tags.get("name") or "prim_%d" % tags["index"])
            obj = S.object_from_arrays(name, v, f, ref)
            obj.hide_render = True
            for k in ("source", "name"):
                if tags.get(k) is not None:
                    obj["atlas_" + k] = tags[k]

        stage = "ground"
        made = []
        if make_ground:
            half = ground_extent / 2.0
            cxg, cyg = float(cam_pos[0]), float(cam_pos[1])
            gv = np.array([[cxg - half, cyg - half, ground_z], [cxg + half, cyg - half, ground_z],
                           [cxg + half, cyg + half, ground_z], [cxg - half, cyg + half, ground_z]])
            gf = np.array([[0, 1, 2], [0, 2, 3]])
            g = S.object_from_arrays("ground_plane", gv, gf, out)
            g["atlas_kind"] = "ground_plane"
            g["atlas_source"] = "blender_massing"
            made.append(g)

        stage = "footprints"
        n_foot = n_wall = n_box = n_skip = 0
        for tags, v, f in prims:
            src = str(tags.get("source") or "")
            if src == "viewport_polygon" and source_mode in ("drawn_polygons", "both", "all"):
                loop = S.boundary_loop(f)
                if loop is None or len(loop) < 3:
                    n_skip += 1
                    continue
                outline = v[loop]
                nrm = _polygon_normal(outline)
                h = float(tags.get("height_m") or default_h)
                if abs(nrm[2]) >= FLAT_NZ:
                    # Footprint: extrude straight up from the drawn plane.
                    pv, pf = S.prism_from_outline(loop, f, v, (0.0, 0.0, h))
                    obj = S.object_from_arrays(
                        "mass_%s" % (tags.get("name") or n_foot), pv, pf, out)
                    obj["atlas_kind"] = "footprint_extrusion"
                    obj["atlas_height_m"] = h
                    n_foot += 1
                else:
                    # Facade: thicken away from the camera so the drawn face
                    # stays on the photo.
                    to_cam = cam_pos - outline.mean(axis=0)
                    away = -nrm if float(np.dot(nrm, to_cam)) > 0 else nrm
                    away = away / max(np.linalg.norm(away), 1e-12)
                    pv, pf = S.prism_from_outline(loop, f, v, away * wall_t)
                    obj = S.object_from_arrays(
                        "wall_%s" % (tags.get("name") or n_wall), pv, pf, out)
                    obj["atlas_kind"] = "facade_slab"
                    obj["atlas_height_m"] = wall_t
                    n_wall += 1
                obj["atlas_source"] = "blender_massing"
                obj["atlas_from_primitive"] = str(tags.get("name") or "")
                made.append(obj)
            elif src == "measured_plane" and source_mode in ("measured_planes", "all"):
                # A MEASURED vertical plane (RANSAC on the MoGe cloud) becomes an
                # oriented facade slab, thickened AWAY from the camera — correctly
                # rotated starting geometry, so nobody has to eyeball axis-aligned
                # boxes against a street that runs 3° off the camera axis (found
                # live 2026-08-16). Ground/backdrop planes are skipped.
                loop = S.boundary_loop(f)
                name = str(tags.get("name") or "")
                if loop is None or len(loop) < 3 or "ground" in name or "backdrop" in name:
                    n_skip += 1
                    continue
                outline = v[loop]
                nrm = _polygon_normal(outline)
                if abs(nrm[2]) >= FLAT_NZ:
                    n_skip += 1
                    continue
                to_cam = cam_pos - outline.mean(axis=0)
                away = -nrm if float(np.dot(nrm, to_cam)) > 0 else nrm
                away = away / max(np.linalg.norm(away), 1e-12)
                pv, pf = S.prism_from_outline(loop, f, v, away * wall_t)
                obj = S.object_from_arrays("facade_%s" % (name or n_wall), pv, pf, out)
                obj["atlas_kind"] = "measured_facade_slab"
                obj["atlas_height_m"] = wall_t
                obj["atlas_source"] = "blender_massing"
                obj["atlas_from_primitive"] = name
                n_wall += 1
                made.append(obj)
            elif src == "block_massing" and source_mode in ("massing_boxes", "both", "all"):
                obj = S.object_from_arrays(
                    "box_%s" % (tags.get("name") or n_box), v, f, out)
                obj["atlas_kind"] = "massing_box"
                obj["atlas_source"] = "blender_massing"
                obj["atlas_from_primitive"] = str(tags.get("name") or "")
                n_box += 1
                made.append(obj)

        stage = "write"
        # Preserved (non-recipe) atlas_out objects are exported too, so a
        # re-run never drops an agent's meshes from out_meshes.npz.
        for o in out.objects:
            if o.type == "MESH" and o not in made:
                made.append(o)
        S.write_out_meshes(ex, made, recipe="massing", extra_report={
            "ground_plane": bool(make_ground), "ground_z": ground_z,
            "cloud_points": n_cloud, "footprints": n_foot,
            "preserved_objects": preserved,
            "facades": n_wall, "massing_boxes": n_box, "skipped_polygons": n_skip,
            "default_height_m": default_h, "wall_thickness_m": wall_t,
        })
        if save_blend:
            bpy.ops.wm.save_as_mainfile(filepath=f"{ex}/scene.blend")
        print("ATLAS_RECIPE_OK massing")
    except Exception as exc:  # noqa: BLE001
        S.fail(ex, stage, exc)


main()
