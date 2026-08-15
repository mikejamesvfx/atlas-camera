"""Build a labelled-visibility truth scene in Blender (RESEARCH).

Runs INSIDE Blender:
    blender --background --python blender_truth_scene.py -- --out out/truth_scene

Produces exact ground truth for hidden-geometry scoring — no depth model in the
loop, unlike the G5 photograph rig:

    render.png        the RGB a predictor gets as input
    truth_points.npz  dense surface samples of the WHOLE scene, in Atlas world
                      coordinates, with a per-point object id
    camera.json       intrinsics + Atlas camera_view_matrix for the render camera

The scene is deliberately built around occlusion: a foreground slab hides part
of a back wall, a box hides part of a cylinder, and a thin post casts a narrow
occlusion — the thin-occluder case from the brief. What is behind each occluder
is REAL geometry with known coordinates, so a prediction can be scored on
structure no camera in the scene ever saw from the render pose.

Blender is Z-up; Atlas is Y-up with the camera looking down -Z. The conversion
happens once, here, at the boundary — never in Atlas core.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

# Blender Z-up world -> Atlas Y-up world. (x, y, z)_blender -> (x, z, -y)_atlas.
BLENDER_TO_ATLAS = np.array([[1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [0.0, -1.0, 0.0]], dtype=np.float64)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def add_box(name, size, location, rotation=(0, 0, 0), colour=(0.5, 0.5, 0.5, 1)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location, rotation=rotation)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = Vector(size)
    _paint(ob, colour)
    return ob


def add_cylinder(name, radius, depth, location, colour=(0.5, 0.5, 0.5, 1)):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth,
                                        location=location, vertices=48)
    ob = bpy.context.active_object
    ob.name = name
    _paint(ob, colour)
    return ob


def _paint(ob, rgba, tex_scale=6.0):
    """Textured material.

    Texture is not cosmetic here. A flat untextured render is out of
    distribution for MoGe: measured 2026-08-15, the first version of this scene
    registered at scale 2.73 (a metric predictor on a metric scene should be
    ~1.0) and 94.8% of the prediction landed clear of any real surface — the
    same failure class as the untextured `golden_corridor` plate, which produced
    zero surface voxels. Monocular geometry needs surface detail to key off, so
    the truth scene must carry it or it measures the depth model's collapse
    instead of the thing under test.
    """
    mat = bpy.data.materials.new(ob.name + "_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None:
        ob.data.materials.append(mat)
        return

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (tex_scale, tex_scale, tex_scale)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 9.0
    noise.inputs["Detail"].default_value = 8.0
    if "Roughness" in noise.inputs:
        noise.inputs["Roughness"].default_value = 0.6
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    r, g, b, _ = rgba
    ramp.color_ramp.elements[0].color = (r * 0.55, g * 0.55, b * 0.55, 1.0)
    ramp.color_ramp.elements[1].color = (min(r * 1.35, 1.0), min(g * 1.35, 1.0),
                                         min(b * 1.35, 1.0), 1.0)
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35

    nt.links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.75
    ob.data.materials.append(mat)


def build_scene():
    """A courtyard with four distinct occlusion cases."""
    objs = []
    # Ground.
    objs.append(add_box("ground", (40, 40, 0.2), (0, 0, -0.1), colour=(0.35, 0.33, 0.30, 1)))
    # Back wall — the thing most often hidden.
    objs.append(add_box("back_wall", (24, 0.6, 9), (0, 11, 4.5), colour=(0.62, 0.58, 0.50, 1)))
    # Side wall, oblique to the camera.
    objs.append(add_box("side_wall", (0.6, 20, 7), (-10, 3, 3.5), colour=(0.55, 0.52, 0.48, 1)))
    # CASE 1 — a broad foreground slab hiding a chunk of the back wall.
    objs.append(add_box("occluder_slab", (7, 0.8, 5), (-1.5, 3.5, 2.5), colour=(0.30, 0.35, 0.42, 1)))
    # CASE 2 — a box hiding part of a cylinder (object behind object).
    objs.append(add_cylinder("cylinder", 1.4, 6, (6, 8, 3), colour=(0.65, 0.45, 0.35, 1)))
    objs.append(add_box("occluder_box", (3, 3, 3.2), (6, 4.5, 1.6), colour=(0.40, 0.45, 0.38, 1)))
    # CASE 3 — a THIN post: narrow occlusion, the brief's hard case.
    objs.append(add_cylinder("thin_post", 0.18, 6, (-5.5, 2.0, 3), colour=(0.25, 0.25, 0.28, 1)))
    # CASE 4 — a low step the ground plane hides behind (grazing occlusion).
    objs.append(add_box("low_step", (6, 1.2, 0.9), (3, 6.5, 0.45), colour=(0.48, 0.50, 0.45, 1)))
    # Scale cues. A monocular predictor has no metric anchor in an abstract box
    # scene; door- and crate-sized objects give it one, which is what keeps the
    # MoGe registration near 1.0 instead of the 2.73 the first build measured.
    objs.append(add_box("door_panel", (1.0, 0.12, 2.1), (2.5, 10.65, 1.05), colour=(0.30, 0.22, 0.18, 1)))
    objs.append(add_box("crate_a", (0.9, 0.9, 0.9), (-3.0, 6.0, 0.45), colour=(0.52, 0.40, 0.26, 1)))
    objs.append(add_box("crate_b", (0.9, 0.9, 0.9), (-3.0, 6.0, 1.35), colour=(0.50, 0.38, 0.25, 1)))
    objs.append(add_box("crate_c", (0.9, 0.9, 0.9), (-2.0, 6.4, 0.45), colour=(0.54, 0.42, 0.28, 1)))
    return objs


def setup_camera(width, height, focal_mm=35.0, sensor_mm=36.0):
    cam_data = bpy.data.cameras.new("truth_cam")
    cam_data.lens = focal_mm
    cam_data.sensor_width = sensor_mm
    cam = bpy.data.objects.new("truth_cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (0.0, -9.0, 1.7)              # eye height, looking +Y
    cam.rotation_euler = (math.radians(88.0), 0.0, 0.0)
    bpy.context.scene.camera = cam

    sc = bpy.context.scene
    sc.render.resolution_x = width
    sc.render.resolution_y = height
    sc.render.resolution_percentage = 100
    return cam


def setup_light():
    sun_data = bpy.data.lights.new("sun", type="SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("sun", sun_data)
    sun.rotation_euler = (math.radians(50), 0, math.radians(35))
    bpy.context.collection.objects.link(sun)
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.55, 0.62, 0.75, 1.0)
        bg.inputs["Strength"].default_value = 1.0
    bpy.context.scene.world = world


def sample_surfaces(objs, per_area=400, seed=0):
    """Dense area-weighted point samples of every triangle, in Atlas world."""
    rng = np.random.default_rng(seed)
    dg = bpy.context.evaluated_depsgraph_get()
    pts, ids = [], []
    for oid, ob in enumerate(objs):
        eval_ob = ob.evaluated_get(dg)
        mesh = eval_ob.to_mesh()
        mesh.calc_loop_triangles()
        mw = np.array(eval_ob.matrix_world)
        verts = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
        verts = verts @ mw[:3, :3].T + mw[:3, 3]
        tris = np.array([t.vertices[:] for t in mesh.loop_triangles], dtype=np.int64)
        if len(tris) == 0:
            eval_ob.to_mesh_clear(); continue
        a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        n = np.maximum(1, (area * per_area).astype(int))
        for i in range(len(tris)):
            k = int(n[i])
            u = rng.random(k); v = rng.random(k)
            over = u + v > 1
            u[over], v[over] = 1 - u[over], 1 - v[over]
            p = a[i] + np.outer(u, b[i] - a[i]) + np.outer(v, c[i] - a[i])
            pts.append(p); ids.append(np.full(k, oid))
        eval_ob.to_mesh_clear()
    P = np.vstack(pts) @ BLENDER_TO_ATLAS.T
    return P, np.concatenate(ids), [o.name for o in objs]


def atlas_view_matrix(cam):
    """Blender camera -> Atlas world->camera 4x4 (Atlas cam: x right, y up, -z fwd).

    Blender's camera also looks down its local -Z with +Y up, so the camera-local
    axes already agree; only the WORLD basis differs.
    """
    mw = np.array(cam.matrix_world, dtype=np.float64)
    R_b, t_b = mw[:3, :3], mw[:3, 3]
    R = BLENDER_TO_ATLAS @ R_b            # camera axes expressed in Atlas world
    t = BLENDER_TO_ATLAS @ t_b
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = t
    return np.linalg.inv(c2w)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=64)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reset_scene()
    objs = build_scene()
    cam = setup_camera(args.width, args.height)
    setup_light()

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    try:
        sc.cycles.samples = args.samples
        sc.cycles.use_denoising = True
    except AttributeError:
        pass
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = str(out / "render.png")
    bpy.ops.render.render(write_still=True)

    P, ids, names = sample_surfaces(objs)
    np.savez_compressed(out / "truth_points.npz", points=P.astype(np.float32),
                        object_id=ids.astype(np.int32))

    vm = atlas_view_matrix(cam)
    fx = cam.data.lens / cam.data.sensor_width * args.width
    meta = {
        "image_width": args.width, "image_height": args.height,
        "fx_px": fx, "fy_px": fx,
        "cx_px": args.width / 2.0, "cy_px": args.height / 2.0,
        "focal_mm": cam.data.lens, "sensor_mm": cam.data.sensor_width,
        "camera_view_matrix": vm.tolist(),
        "camera_position_atlas": (BLENDER_TO_ATLAS @ np.array(cam.matrix_world)[:3, 3]).tolist(),
        "objects": names,
        "n_truth_points": int(len(P)),
        "coordinate_system": "atlas_right_handed_y_up_minus_z_forward",
    }
    (out / "camera.json").write_text(json.dumps(meta, indent=2))
    print(f"[truth] {len(P)} points, {len(names)} objects -> {out}")


if __name__ == "__main__":
    main()
