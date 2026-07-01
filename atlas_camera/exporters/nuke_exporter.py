"""Nuke camera-projection script writer.

Generates a Nuke Python script that builds:
  Read (source plate) → Project3D2 ← Camera2 (solved params)
                         Card3D (40×40 m ground, XZ plane) ↗
                         ScanlineRender ← Camera2

Atlas core is right-handed Y-up. Nuke 3D space is also Y-up, so positions
and the world matrix pass through unchanged.
"""

from __future__ import annotations

from pathlib import Path

from atlas_camera.core.camera_math import derive_sensor_height_mm
from atlas_camera.core.schema import AtlasSolve

_SOURCE_IMAGE_NAME = "source_image.png"


def write_nuke_projection_script(solve: AtlasSolve, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

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

    inferred_warning = ""
    if solve.camera.focal_length_inferred:
        inferred_warning = (
            "\n    nuke.message("
            '"Atlas focal_length_mm was inferred from a fallback assumption; '
            'review atlas_solve.json before final handoff.")'
        )

    script = f'''"""Atlas Camera Nuke projection setup.

Generated from an Atlas core solve. Builds a camera-projection node graph
for the solved camera onto a 40 x 40 m ground-plane card.

Node graph:
  Read ──────────────────┐
                          ├─ Project3D2 ─ ScanlineRender
  Camera2 ───────────────┤               │
                          │  Card3D ──────┘
                          └─(projection camera also used as render camera)

Run via Nuke's Script Editor: exec(open("nuke_cards.py").read()); build_projection()
"""

from __future__ import annotations

import os
import nuke


def build_projection(package_dir=None):
    package_dir = package_dir or os.path.dirname(os.path.abspath(__file__))
    nuke.scriptClear()
    {inferred_warning}

    # Source plate
    read = nuke.createNode("Read", inpanel=False)
    read["file"].setValue(os.path.join(package_dir, "{_SOURCE_IMAGE_NAME}"))
    read["first"].setValue(1)
    read["last"].setValue(1)

    # Solved camera — world matrix in row-major order
    cam = nuke.createNode("Camera2", inpanel=False)
    cam["focal"].setValue({focal!r})
    cam["haperture"].setValue({sensor_w_mm!r})
    cam["vaperture"].setValue({sensor_h_mm!r})
    cam["win_translate"].setValue([{win_tx!r}, {win_ty!r}])
    cam["translate"].setValue([{tx!r}, {ty!r}, {tz!r}])
    cam["useMatrix"].setValue(True)
    cam["matrix"].setValue({flat_world!r})

    # Ground-plane card (40 x 40 m). Card3D default orientation is XY (vertical);
    # rotate -90° around X so it lies flat in the XZ plane (Y-up world ground).
    card = nuke.createNode("Card3D", inpanel=False)
    card["xsize"].setValue(40.0)
    card["ysize"].setValue(40.0)
    card["rotate"].setValue([-90.0, 0.0, 0.0])

    # Project3D2: stamps the source plate onto the card from the solved camera.
    # Inputs: img (0) = plate, cam (1) = projection camera, geo (2+) = geometry.
    proj = nuke.createNode("Project3D2", inpanel=False)
    proj.setInput(0, read)
    proj.setInput(1, cam)
    proj.setInput(2, card)

    # ScanlineRender: renders the projected 3D scene to 2D.
    render = nuke.createNode("ScanlineRender", inpanel=False)
    render.setInput(0, proj)
    render.setInput(1, cam)
    fmt_name = "atlas_{image_w}x{image_h}"
    if fmt_name not in [f.name() for f in nuke.formats()]:
        nuke.addFormat(f"{image_w} {image_h} 1.0 {{fmt_name}}")
    render["format"].setValue(fmt_name)

    return render


ATLAS_CAMERA_NAME = {solve.camera.name!r}

if __name__ == "__main__":
    build_projection()
'''
    destination.write_text(script, encoding="utf-8")
    return destination


class NukeExporter:
    def write_scene(self, solve: AtlasSolve, output_path: str | Path) -> Path:
        return write_nuke_projection_script(solve, output_path)
