"""Relief-mesh OBJ exporter for Maya / Nuke / ZBrush handoff.

Writes a Y-up OBJ with per-vertex UVs (the camera projection is baked into the
UVs by relief_mesh.py) plus an MTL referencing the source image — so the mesh
imports into Maya (File > Import), Nuke (ReadGeo), or ZBrush already textured
with the projected photo, ready to retopologize / reproject UVs.

No dependencies beyond the standard library (Pillow only if a texture image is
passed for saving).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atlas_camera.core.relief_mesh import ReliefMesh


def export_relief_mesh(
    mesh: ReliefMesh,
    output_dir: str | Path,
    *,
    texture: Any | None = None,
    name: str = "atlas_relief_mesh",
) -> dict[str, str]:
    """Write ``{name}.obj`` + ``{name}.mtl`` (+ texture PNG) to ``output_dir``.

    ``texture`` is an optional PIL Image (the source photo); when given it is
    saved next to the OBJ and referenced as ``map_Kd``. Returns the written
    paths. Coordinates are Atlas world (right-handed, Y-up, metres) — Maya and
    Nuke default conventions.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    obj_path = out / f"{name}.obj"
    mtl_path = out / f"{name}.mtl"
    material = "atlas_relief_projection"

    tex_path: Path | None = None
    if texture is not None:
        tex_path = out / f"{name}_diffuse.png"
        texture.save(tex_path)

    lines: list[str] = [
        "# Atlas Camera relief mesh — Y-up, metres.",
        "# UVs bake the recovered-camera projection: the referenced texture is",
        "# already correctly projected; retopo/reproject as needed.",
        f"mtllib {mtl_path.name}",
        f"o {name}",
    ]
    lines.extend(
        f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in mesh.vertices
    )
    lines.extend(
        f"vt {t[0]:.6f} {t[1]:.6f}" for t in mesh.uvs
    )
    lines.append(f"usemtl {material}")
    # Vertex and UV lists are 1:1, so face indices serve both (v/vt).
    lines.extend(
        f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}"
        for a, b, c in mesh.faces
    )
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    mtl_lines = [
        f"newmtl {material}",
        "Kd 1.000 1.000 1.000",
        "Ka 0.000 0.000 0.000",
        "Ks 0.000 0.000 0.000",
        "illum 1",
    ]
    if tex_path is not None:
        mtl_lines.append(f"map_Kd {tex_path.name}")
    mtl_path.write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")

    result = {"obj": str(obj_path), "mtl": str(mtl_path)}
    if tex_path is not None:
        result["texture"] = str(tex_path)
    return result


def export_relief_mesh_glb(
    mesh: ReliefMesh,
    output_dir: str | Path,
    *,
    texture: Any | None = None,
    name: str = "atlas_relief_mesh",
) -> dict[str, str]:
    """Write a self-contained ``{name}.glb`` (glTF 2.0 binary, texture embedded).

    glTF is right-handed Y-up like Atlas — coordinates pass through unchanged;
    only the UV origin flips (glTF is top-left, the mesh stores OBJ bottom-left).
    The material is tagged ``KHR_materials_unlit`` so the projected photo renders
    exactly as-is (no lighting), with a PBR fallback for viewers without the
    extension. Zero dependencies beyond numpy (+ Pillow when embedding texture).
    """
    import io
    import json as _json
    import struct

    import numpy as np

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    glb_path = out / f"{name}.glb"

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    uvs = np.asarray(mesh.uvs, dtype=np.float32).copy()
    uvs[:, 1] = 1.0 - uvs[:, 1]  # OBJ bottom-left → glTF top-left

    png_bytes = b""
    if texture is not None:
        buf = io.BytesIO()
        texture.save(buf, format="PNG")
        png_bytes = buf.getvalue()

    def _pad4(data: bytes, pad: bytes = b"\x00") -> bytes:
        return data + pad * ((4 - len(data) % 4) % 4)

    # Binary buffer layout: positions | uvs | indices | (image)
    parts = [_pad4(verts.tobytes()), _pad4(uvs.tobytes()), _pad4(faces.tobytes())]
    if png_bytes:
        parts.append(_pad4(png_bytes))
    offsets = []
    off = 0
    for part in parts:
        offsets.append(off)
        off += len(part)
    bin_chunk = b"".join(parts)

    buffer_views = [
        {"buffer": 0, "byteOffset": offsets[0], "byteLength": verts.nbytes, "target": 34962},
        {"buffer": 0, "byteOffset": offsets[1], "byteLength": uvs.nbytes, "target": 34962},
        {"buffer": 0, "byteOffset": offsets[2], "byteLength": faces.nbytes, "target": 34963},
    ]
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": int(len(verts)), "type": "VEC3",
         "min": [float(v) for v in verts.min(axis=0)],
         "max": [float(v) for v in verts.max(axis=0)]},
        {"bufferView": 1, "componentType": 5126, "count": int(len(uvs)), "type": "VEC2"},
        {"bufferView": 2, "componentType": 5125, "count": int(faces.size), "type": "SCALAR"},
    ]

    material: dict[str, Any] = {
        "name": "atlas_relief_projection",
        "doubleSided": True,
        "extensions": {"KHR_materials_unlit": {}},
        "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0},
    }
    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "AtlasCamera relief mesh"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{
            "name": name,
            "primitives": [{
                "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                "indices": 2,
                "material": 0,
            }],
        }],
        "materials": [material],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_chunk)}],
    }
    if png_bytes:
        buffer_views.append({"buffer": 0, "byteOffset": offsets[3], "byteLength": len(png_bytes)})
        gltf["images"] = [{"bufferView": 3, "mimeType": "image/png"}]
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987,
                             "wrapS": 33071, "wrapT": 33071}]
        gltf["textures"] = [{"source": 0, "sampler": 0}]
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    else:
        material["pbrMetallicRoughness"]["baseColorFactor"] = [0.6, 0.6, 0.6, 1.0]

    json_chunk = _pad4(_json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    with open(glb_path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))          # glTF header
        fh.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))    # JSON chunk
        fh.write(json_chunk)
        fh.write(struct.pack("<II", len(bin_chunk), 0x004E4942))     # BIN chunk
        fh.write(bin_chunk)

    return {"glb": str(glb_path)}
