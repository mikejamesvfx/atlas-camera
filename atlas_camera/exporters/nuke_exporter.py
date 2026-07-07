"""Nuke camera-projection script writer.

Generates a Nuke Python script that builds:
  Read (source plate) ─┐
                        ├─ Project3D2 ─ Card or ReadGeo2 ─ ScanlineRender
  Camera2 (solved) ─────┘                (ground card,      ↑
                                          or the real       Camera2 (same node)
                                          relief mesh OBJ)

Project3D2 is a 2D node (img + cam inputs only — it has no geometry input;
"projecting" here means baking camera-space UVs into the 2D stream) whose
OUTPUT feeds the geometry node's own image input — the geometry (a flat Card
by default, or the actual derived relief mesh via ReadGeo2 when
`relief_mesh_obj_path` is given) is what carries that LIVE-projected image
into 3D space; this is the same matte-painting behaviour as the ComfyUI
viewport's own 📽 Project (texels assigned by ray from the recovered camera,
so anything outside its frustum stays black/undefined) — deliberately not a
static UV-baked texture, even though the relief mesh's own UVs already
happen to bake that exact projection (feeding the raw, un-projected photo
directly into ReadGeo2 would look identical from the original camera's own
viewpoint, but wouldn't behave like a real projection rig: it wouldn't black
out correctly, and swapping in a different plate wouldn't re-project).
ScanlineRender's three inputs are bg=0 (unused here), obj=1 (the Card or
ReadGeo2), cam=2 (the same Camera2) — NOT obj=0/cam=1 as a naive reading of
the node's "3 inputs" might suggest. This topology and every knob name below
(Card's `scaling`, not a Card3D-only `xsize`/`ysize` which doesn't exist; the
render format living on `Root`, not on ScanlineRender itself) were confirmed
by actually building and rendering this graph in Nuke (16.1v3, Indie
license) — not just read from documentation — see the "Nuke
camera-projection topology" note in CLAUDE.md for how the earlier,
un-runnable version of this exporter was found and fixed.

Atlas core is right-handed Y-up. Nuke 3D space is also Y-up, so positions
and the world matrix pass through unchanged.
"""

from __future__ import annotations

from pathlib import Path

from atlas_camera.core.camera_math import derive_sensor_height_mm
from atlas_camera.core.schema import AtlasSolve
from atlas_camera.exporters._plate import primary_plate_colorspace, primary_plate_path

_SOURCE_IMAGE_NAME = "source_image.png"


def write_nuke_projection_script(
    solve: AtlasSolve,
    output_path: str | Path,
    *,
    source_image_name: str = _SOURCE_IMAGE_NAME,
    use_package_source: bool = False,
    relief_mesh_obj_path: str | Path | None = None,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    mesh_path = str(relief_mesh_obj_path).replace("\\", "/") if relief_mesh_obj_path else None

    intrinsics = solve.camera.intrinsics
    focal = intrinsics.focal_length_mm or 35.0
    sensor_w_mm = intrinsics.sensor_width_mm or 36.0
    sensor_h_mm = derive_sensor_height_mm(intrinsics)
    image_w = intrinsics.image_width
    image_h = intrinsics.image_height

    tx, ty, tz = solve.camera.extrinsics.camera_position

    # Principal-point offset as Nuke win_translate (normalised by aperture).
    # Nuke Y is up, image Y is down — flip the vertical component.
    cx = intrinsics.cx_px if intrinsics.cx_px is not None else image_w / 2.0
    cy = intrinsics.cy_px if intrinsics.cy_px is not None else image_h / 2.0
    win_tx = (cx - image_w / 2.0) / image_w
    win_ty = -((cy - image_h / 2.0) / image_h)

    # Camera world matrix, row-major flat list (Nuke Camera2 matrix knob format).
    flat_world = [v for row in solve.camera.extrinsics.camera_world_matrix for v in row]
    source_plate_path = None if use_package_source else primary_plate_path(solve)
    if source_plate_path:
        # Nuke's live knob-setting API (Node["file"].setValue(...)) runs the
        # value through TCL escape interpretation regardless of how the string
        # was built in Python — a Windows path's backslash-letter sequences
        # (e.g. "\Users", "\AtlasCamera") get silently eaten, corrupting the
        # path (confirmed by actually running the generated script in Nuke
        # 16.1v3). Forward slashes are accepted identically on Windows and
        # sidestep the whole TCL-escaping question, so normalise here rather
        # than trying to double-escape backslashes through two parsers.
        source_plate_path = source_plate_path.replace("\\", "/")
    source_colorspace = primary_plate_colorspace(solve)
    output_profile = getattr(solve, "output_profile", None)
    ocio_summary = (
        output_profile.to_dict() if output_profile and hasattr(output_profile, "to_dict") else None
    )

    inferred_warning = ""
    if solve.camera.focal_length_inferred:
        inferred_warning = (
            "\n    nuke.message("
            '"Atlas focal_length_mm was inferred from a fallback assumption; '
            'review atlas_solve.json before final handoff.")'
        )

    if mesh_path:
        geo_comment = (
            '    # The real derived relief mesh (ReadGeo2), textured by the projected\n'
            "    # image above via its own image input — same live matte-painting\n"
            "    # behaviour as the ground-card default, just on the actual geometry\n"
            "    # instead of a flat plane. World-space vertices are already correctly\n"
            "    # scaled/positioned (no rotate/scaling knobs needed, unlike Card)."
        )
        geo_creation = (
            f'    geo = nuke.createNode("ReadGeo2", inpanel=False)\n'
            f'    geo["file"].setValue({mesh_path!r})'
        )
    else:
        geo_comment = (
            "    # Ground-plane card (40 x 40 m), textured by the projected image\n"
            "    # above. Card's default 1x1 unit quad is sized via `scaling` (there is\n"
            "    # no xsize/ysize knob); rotate -90 degrees around X so it lies flat in\n"
            "    # the XZ plane (Y-up world ground)."
        )
        geo_creation = (
            '    geo = nuke.createNode("Card", inpanel=False)\n'
            '    geo["scaling"].setValue([40.0, 40.0, 1.0])\n'
            '    geo["rotate"].setValue([-90.0, 0.0, 0.0])'
        )

    script = f'''"""Atlas Camera Nuke projection setup.

Generated from an Atlas core solve. Builds a camera-projection node graph
for the solved camera onto {"the derived relief mesh" if mesh_path else "a 40 x 40 m ground-plane card"}.

Node graph:
  Read ─┐
        ├─ Project3D2 ─ {"ReadGeo2 (relief mesh)" if mesh_path else "Card"} ─ ScanlineRender
  Camera2 ┘              ↑          ↑
                    (same Camera2, reused as both the projector and the
                     render camera)

Project3D2 has only two inputs (img, cam) — it bakes the camera projection
into a 2D stream; the geometry node then carries that image into 3D space
via its own image input. ScanlineRender's inputs are bg=0 (unused),
obj=1=geo, cam=2=Camera2.

Run via Nuke's Script Editor: exec(open("nuke_cards.py").read()); build_projection()
"""

from __future__ import annotations

import os
import nuke


def build_projection(package_dir=None):
    package_dir = package_dir or os.path.dirname(os.path.abspath(__file__))
    nuke.scriptClear()
    {inferred_warning}

    # Source plate. Browser/viewport previews may be JPEG/PNG proxies; this
    # Read points at the registered float plate when Atlas has one.
    source_path = {source_plate_path!r} or os.path.join(package_dir, {source_image_name!r})
    # Nuke's knob-setting API runs string values through TCL escape
    # interpretation, which silently eats backslash-letter sequences in a
    # Windows path (confirmed by running this exact script in Nuke) -
    # forward slashes work identically on Windows and sidestep it entirely.
    source_path = source_path.replace("\\\\", "/")
    read = nuke.createNode("Read", inpanel=False)
    read["file"].setValue(source_path)
    read["first"].setValue(1)
    read["last"].setValue(1)
    source_colorspace = {source_colorspace!r}
    if source_colorspace and "colorspace" in read.knobs():
        try:
            read["colorspace"].setValue(source_colorspace)
        except Exception:
            nuke.tprint("Atlas: could not set Read colorspace to " + source_colorspace)

    ocio_note = nuke.createNode("StickyNote", inpanel=False)
    ocio_note["label"].setValue(
        "Atlas color handoff\\n"
        "Source colorspace: " + str(source_colorspace or "unspecified") + "\\n"
        "Output profile: " + {str(ocio_summary)!r}
    )

    # Solved camera — world matrix in row-major order
    cam = nuke.createNode("Camera2", inpanel=False)
    cam["focal"].setValue({focal!r})
    cam["haperture"].setValue({sensor_w_mm!r})
    cam["vaperture"].setValue({sensor_h_mm!r})
    cam["win_translate"].setValue([{win_tx!r}, {win_ty!r}])
    cam["translate"].setValue([{tx!r}, {ty!r}, {tz!r}])
    cam["useMatrix"].setValue(True)
    cam["matrix"].setValue({flat_world!r})

    # Project3D2: bakes the solved camera's projection into the 2D plate.
    # Only two real inputs exist on this node — img (0) = plate, cam (1) =
    # projection camera; it has no geometry input of its own.
    proj = nuke.createNode("Project3D2", inpanel=False)
    proj.setInput(0, read)
    proj.setInput(1, cam)

{geo_comment}
{geo_creation}
    geo.setInput(0, proj)

    # ScanlineRender: renders the geometry (with Camera2 as both projector and
    # render camera) to 2D. Real input mapping is bg=0 (left unconnected),
    # obj=1, cam=2 — NOT obj=0/cam=1.
    render = nuke.createNode("ScanlineRender", inpanel=False)
    render.setInput(1, geo)
    render.setInput(2, cam)

    # Render/output resolution is a Root (project) setting, not a knob on
    # ScanlineRender itself — ScanlineRender has no "format" knob.
    fmt_name = "atlas_{image_w}x{image_h}"
    if fmt_name not in [f.name() for f in nuke.formats()]:
        nuke.addFormat(f"{image_w} {image_h} 1.0 {{fmt_name}}")
    nuke.root()["format"].setValue(fmt_name)

    return render


ATLAS_CAMERA_NAME = {solve.camera.name!r}

if __name__ == "__main__":
    build_projection()
'''
    destination.write_text(script, encoding="utf-8")
    return destination


def write_nuke_native_script(
    solve: AtlasSolve,
    output_path: str | Path,
    *,
    use_package_source: bool = False,
    relief_mesh_obj_path: str | Path | None = None,
) -> Path:
    """Write the same camera-projection graph as `write_nuke_projection_script`
    as a native, plain-text `.nk` scene — drag-and-drop / File > Open ready,
    no Script Editor step required.

    The `.nk` push/pop stack model resolves connections in LIFO order (last
    `push` is considered first) and, for each candidate, walks a node's input
    slots from 0 upward looking for the first still-empty, type-compatible
    one — confirmed empirically (not from documentation, which doesn't spell
    this out) by round-tripping small graphs through Nuke's own
    `scriptReadFile` and inspecting the resulting connections. Two
    consequences that shape the layout below:
      - Project3D2's img/cam both accept index 0, so ordinary reversed-order
        pushes place them correctly (push cam, then push read -> img=0,
        cam=1).
      - ScanlineRender's bg/obj/cam slots are NOT type-symmetric (bg accepts
        neither a Card nor a Camera2), and reusing the same Camera2 node a
        SECOND time as a push target (it was already consumed once by
        Project3D2) does not reliably re-resolve through the stack — confirmed
        by direct testing, including with the correct push order and pairing.
        Rather than fight that specific case in text form, the Camera2->
        ScanlineRender(cam) link is completed by a one-line Python callback
        on Root's `onScriptLoad` knob, which Nuke runs automatically the
        moment the script opens — still a single self-contained `.nk` file,
        no companion script needed.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    intrinsics = solve.camera.intrinsics
    focal = intrinsics.focal_length_mm or 35.0
    sensor_w_mm = intrinsics.sensor_width_mm or 36.0
    sensor_h_mm = derive_sensor_height_mm(intrinsics)
    image_w = intrinsics.image_width
    image_h = intrinsics.image_height

    tx, ty, tz = solve.camera.extrinsics.camera_position

    cx = intrinsics.cx_px if intrinsics.cx_px is not None else image_w / 2.0
    cy = intrinsics.cy_px if intrinsics.cy_px is not None else image_h / 2.0
    win_tx = (cx - image_w / 2.0) / image_w
    win_ty = -((cy - image_h / 2.0) / image_h)

    m = solve.camera.extrinsics.camera_world_matrix
    matrix_block = "\n".join(
        "     {" + " ".join(repr(v) for v in row) + "}" for row in m
    )

    source_plate_path = None if use_package_source else primary_plate_path(solve)
    if source_plate_path:
        source_plate_path = source_plate_path.replace("\\", "/")
    else:
        # No registered float plate - fall back to whatever preview path the
        # solve itself carries (e.g. a browser-uploaded proxy), same
        # forward-slash normalisation as the .py writer.
        source_plate_path = str(solve.image_path or "").replace("\\", "/")
    source_colorspace = primary_plate_colorspace(solve)
    suffix = Path(source_plate_path).suffix.lower().lstrip(".") or "exr"

    output_profile = getattr(solve, "output_profile", None)
    ocio_summary = (
        output_profile.to_dict() if output_profile and hasattr(output_profile, "to_dict") else None
    )
    note_label = (
        "Atlas color handoff\\n"
        f"Source colorspace: {source_colorspace or 'unspecified'}\\n"
        f"Output profile: {ocio_summary}"
    ).replace('"', '\\"')

    colorspace_line = f' colorspace {source_colorspace}\n' if source_colorspace else ""
    fmt_name = f"atlas_{image_w}x{image_h}"

    mesh_path = str(relief_mesh_obj_path).replace("\\", "/") if relief_mesh_obj_path else None
    if mesh_path:
        # The real derived relief mesh, textured by the live camera projection
        # via its own image input (index 0, same as Card) — not a static
        # UV-baked texture, even though the mesh's own UVs already happen to
        # bake that exact projection: feeding the raw photo directly would
        # look identical from the original camera but wouldn't behave like a
        # real projection rig (no correct black-out outside the frustum, no
        # re-projection if a different plate is swapped in).
        geo_block = f'''ReadGeo2 {{
 file "{mesh_path}"
 name Geo1
 xpos 0
 ypos 200
}}'''
    else:
        geo_block = '''Card {
 rotate {-90 0 0}
 scaling {40 40 1}
 name Geo1
 xpos 0
 ypos 200
}'''

    script = f'''Root {{
 format "{image_w} {image_h} 0 0 {image_w} {image_h} 1 {fmt_name}"
 onScriptLoad "nuke.toNode('ScanlineRender1').setInput(2, nuke.toNode('Camera1'))"
 name "{destination.name}"
}}
StickyNote {{
 inputs 0
 name StickyNote1
 label "{note_label}"
 xpos 0
 ypos -120
}}
Read {{
 inputs 0
 file_type {suffix}
 file "{source_plate_path}"
 first 1
 last 1
{colorspace_line} name Read1
 xpos 0
 ypos 0
}}
set N_read [stack 0]
Camera2 {{
 inputs 0
 translate {{{tx!r} {ty!r} {tz!r}}}
 useMatrix true
 matrix {{
{matrix_block}
 }}
 focal {focal!r}
 haperture {sensor_w_mm!r}
 vaperture {sensor_h_mm!r}
 win_translate {{{win_tx!r} {win_ty!r}}}
 name Camera1
 xpos 200
 ypos 0
}}
set N_cam [stack 0]
push $N_cam
push $N_read
Project3D2 {{
 inputs 2
 name Project3D1
 xpos 0
 ypos 100
}}
{geo_block}
set N_geo [stack 0]
push $N_geo
ScanlineRender {{
 inputs 3
 name ScanlineRender1
 xpos 0
 ypos 300
}}
'''
    destination.write_text(script, encoding="utf-8")
    return destination


class NukeExporter:
    def write_scene(
        self,
        solve: AtlasSolve,
        output_path: str | Path,
        *,
        source_image_name: str = _SOURCE_IMAGE_NAME,
        use_package_source: bool = False,
    ) -> Path:
        return write_nuke_projection_script(
            solve,
            output_path,
            source_image_name=source_image_name,
            use_package_source=use_package_source,
        )
