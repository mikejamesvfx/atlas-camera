"""Multi-layer GLB: every Atlas projection layer as its own textured mesh.

``relief_mesh_exporter.export_relief_mesh_glb`` writes ONE mesh with ONE
texture. A layered Atlas scene is several surfaces each projected from its own
plate (photo on the relief, clean plate on the bands, sky on the domes, model
vertex colour on a generated object's hidden side), so this writer emits one
glTF node + mesh + material per layer into a single self-contained GLB.

Conventions follow the single-mesh exporter so the two agree: glTF is Y-up
like Atlas (positions pass through), UVs are stored OBJ bottom-left and
flipped here, materials are ``KHR_materials_unlit`` (a projection is lighting
baked), and a generated object's hidden side is a second, untextured primitive
coloured by linear ``COLOR_0`` (``_generated_split``).

glTF images are PNG/JPEG only -- there is no EXR image format -- so a layer's
float plate rides beside the GLB and is named in its material ``extras``.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PNG_MIME = "image/png"
_JPEG_MIME = "image/jpeg"


@dataclass
class SceneLayer:
    """One surface of the scene and the plate that paints it."""

    name: str
    vertices: Any                  # (N, 3) world-space metres, Y-up
    faces: Any                     # (M, 3)
    uvs: Any = None                # (N, 2) OBJ bottom-left, or None
    image_bytes: bytes | None = None
    image_mime: str = _PNG_MIME
    vertex_colors: Any = None      # (N, 3) sRGB 0..1 (generated objects)
    photo_weight: Any = None       # (N,) -- see core.generated_mesh
    extras: dict[str, Any] = field(default_factory=dict)


def _pad4(data: bytes, pad: bytes = b"\x00") -> bytes:
    return data + pad * ((4 - len(data) % 4) % 4)


def write_scene_glb(layers: list[SceneLayer], path: str | Path, *,
                    generator: str = "AtlasCamera scene") -> dict[str, Any]:
    """Write ``layers`` into one GLB at ``path``; returns a summary dict."""
    import numpy as np

    from atlas_camera.core.generated_mesh import srgb_to_linear
    from atlas_camera.core.relief_mesh import ReliefMesh
    from atlas_camera.exporters.relief_mesh_exporter import (
        _generated_split,
        _topology_safe_faces,
    )

    blob = bytearray()
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    materials: list[dict] = []
    meshes: list[dict] = []
    nodes: list[dict] = []
    images: list[dict] = []
    textures: list[dict] = []
    summary: list[dict] = []

    def add_view(data: bytes, target: int | None = None) -> int:
        offset = len(blob)
        blob.extend(_pad4(data))
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return len(buffer_views) - 1

    def add_accessor(arr: Any, kind: str, component: int, *, minmax: bool = False,
                     target: int | None = 34962) -> int:
        acc = {"bufferView": add_view(arr.tobytes(), target), "componentType": component,
               "count": int(arr.shape[0]) if kind != "SCALAR" else int(arr.size), "type": kind}
        if minmax:
            acc["min"] = [float(v) for v in arr.min(axis=0)]
            acc["max"] = [float(v) for v in arr.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    def unlit(name: str, *, texture: int | None, extras: dict, blend: bool = False) -> int:
        pbr: dict[str, Any] = {"metallicFactor": 0.0, "roughnessFactor": 1.0}
        if texture is not None:
            pbr["baseColorTexture"] = {"index": texture}
        else:
            pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
        mat = {"name": name, "doubleSided": True, "pbrMetallicRoughness": pbr,
               "extensions": {"KHR_materials_unlit": {}}}
        if extras:
            mat["extras"] = extras
        materials.append(mat)
        return len(materials) - 1

    for layer in layers:
        verts = np.asarray(layer.vertices, dtype=np.float32).reshape(-1, 3)
        if not len(verts):
            continue
        faces = _topology_safe_faces(verts, np.asarray(layer.faces).reshape(-1, 3))
        if not len(faces):
            continue
        n = len(verts)
        uvs = None if layer.uvs is None else np.asarray(layer.uvs, dtype=np.float32).reshape(-1, 2)
        if uvs is not None and len(uvs) != n:
            uvs = None

        gen = None
        if layer.vertex_colors is not None and uvs is not None:
            gen = _generated_split(ReliefMesh(
                vertices=verts, faces=faces, uvs=uvs,
                vertex_colors=np.asarray(layer.vertex_colors, dtype=np.float32).reshape(-1, 3),
                photo_weight=None if layer.photo_weight is None
                else np.asarray(layer.photo_weight, dtype=np.float32).reshape(-1)), faces)
        colors = None
        if gen is not None:
            verts = gen["vertices"].astype(np.float32)
            uvs = gen["uvs"].astype(np.float32)
            colors = np.ones((len(verts), 4), dtype=np.float32)
            colors[:, :3] = np.asarray(srgb_to_linear(gen["colors_srgb"]), dtype=np.float32)
            groups = [("photo", gen["photo_faces"]), ("generated", gen["vc_faces"])]
        else:
            groups = [("photo", faces)]

        attrs = {"POSITION": add_accessor(np.ascontiguousarray(verts), "VEC3", 5126, minmax=True)}
        if uvs is not None:
            st = uvs.copy()
            st[:, 1] = 1.0 - st[:, 1]  # OBJ bottom-left -> glTF top-left
            attrs["TEXCOORD_0"] = add_accessor(np.ascontiguousarray(st), "VEC2", 5126)
        if colors is not None:
            attrs["COLOR_0"] = add_accessor(np.ascontiguousarray(colors), "VEC4", 5126)

        texture = None
        if layer.image_bytes and uvs is not None:
            images.append({"bufferView": add_view(layer.image_bytes),
                           "mimeType": layer.image_mime, "name": layer.name})
            textures.append({"source": len(images) - 1, "sampler": 0})
            texture = len(textures) - 1

        primitives = []
        for kind, group in groups:
            idx = np.ascontiguousarray(np.asarray(group, dtype=np.uint32).reshape(-1))
            if not idx.size:
                continue
            if kind == "photo":
                mat = unlit(layer.name, texture=texture, extras=dict(layer.extras))
            else:
                mat = unlit(f"{layer.name}_generated_vertex_colour", texture=None,
                            extras={"atlas_role": "generated_hidden_side"})
            primitives.append({"attributes": attrs,
                               "indices": add_accessor(idx, "SCALAR", 5125, target=34963),
                               "material": mat})
        if not primitives:
            continue
        meshes.append({"name": layer.name, "primitives": primitives})
        nodes.append({"mesh": len(meshes) - 1, "name": layer.name})
        summary.append({"name": layer.name, "vertices": int(len(verts)),
                        "faces": int(sum(len(np.asarray(g).reshape(-1, 3)) for _, g in groups)),
                        "textured": texture is not None, "vertex_colour": gen is not None})

    if not nodes:
        raise ValueError("no layer had geometry to write")

    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": generator},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes, "meshes": meshes, "materials": materials,
        "accessors": accessors, "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob)}],
    }
    if images:
        gltf["images"] = images
        gltf["textures"] = textures
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987,
                             "wrapS": 33071, "wrapT": 33071}]

    json_chunk = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_chunk = bytes(blob)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
        fh.write(json_chunk)
        fh.write(struct.pack("<II", len(bin_chunk), 0x004E4942))
        fh.write(bin_chunk)
    return {"glb": str(out), "bytes": total, "layers": summary}


LINEAR_FROM_DISPLAY = "Linear Rec.709 (sRGB)"


def _png_bytes(pil: Any) -> bytes:
    import io
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def _decode_data_uri(uri: str | None) -> Any:
    """A ``data:image/...;base64,`` URI (ProjectionSource.image_b64) as PIL."""
    if not uri:
        return None
    import base64
    import io

    from PIL import Image
    payload = uri.split(",", 1)[1] if uri.startswith("data:") else uri
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def _write_exr_sidecar(path: Path, pil: Any, plate_ref: Any) -> dict[str, Any]:
    """One layer plate as float EXR. A float plate_ref on disk is preferred
    (its own values, its own colourspace tag); otherwise the 8-bit display
    plate is linearised with the sRGB EOTF and SAID to be that."""
    import numpy as np

    from atlas_camera.plate.oiio_io import read_plate, write_exr

    ref_path = getattr(plate_ref, "image_path", None) if plate_ref is not None else None
    if ref_path and Path(ref_path).is_file() and not getattr(plate_ref, "is_proxy", True):
        plate = read_plate(str(ref_path), output_colorspace=None)
        if (plate.width, plate.height) == pil.size:
            cs = plate.input_colorspace or getattr(plate_ref, "colorspace", "")
            write_exr(str(path), plate.pixels, bit_depth="half", source_colorspace=cs or None)
            return {"exr": path.name, "exr_colorspace": cs, "exr_origin": "plate_ref",
                    "scene_referred": True}
    from atlas_camera.core.generated_mesh import srgb_to_linear
    lin = srgb_to_linear(np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0)
    write_exr(str(path), lin.astype(np.float32), bit_depth="half",
              source_colorspace=LINEAR_FROM_DISPLAY)
    return {"exr": path.name, "exr_colorspace": LINEAR_FROM_DISPLAY,
            "exr_origin": "linearised_display_plate", "scene_referred": False}


def write_float_ply(path: Path, vertices: Any, faces: Any, colors: Any) -> str:
    """Binary little-endian PLY with FLOAT vertex colour (r, g, b may exceed 1).

    glTF COLOR_0 is a 0..1 quantity, so HDR vertex colour (ACEScg linear from
    AtlasHDRVertexTransfer) rides beside the GLB in a format DCCs read with
    float colour attributes (Houdini, Blender's PLY importer, Open3D).
    """
    import numpy as np

    v = np.asarray(vertices, dtype="<f4").reshape(-1, 3)
    c = np.asarray(colors, dtype="<f4").reshape(-1, 3)
    f = np.asarray(faces, dtype="<i4").reshape(-1, 3)
    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        "comment Atlas Camera generated object - vertex colour is ACEScg linear",
        f"element vertex {len(v)}",
        "property float x", "property float y", "property float z",
        "property float red", "property float green", "property float blue",
        f"element face {len(f)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]) + "\n"
    rows = np.empty(len(v), dtype=[("p", "<f4", 3), ("c", "<f4", 3)])
    rows["p"], rows["c"] = v, c
    tris = np.empty(len(f), dtype=[("n", "u1"), ("i", "<i4", 3)])
    tris["n"], tris["i"] = 3, f
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(rows.tobytes())
        fh.write(tris.tobytes())
    return str(path)


def build_scene_layers(
    solve: Any,
    primary_plate: Any,
    *,
    exr_dir: str | Path | None = None,
    exr_prefix: str = "atlas_scene",
) -> tuple[list[SceneLayer], list[dict[str, Any]], list[str]]:
    """Every projection layer of ``solve`` as a ``SceneLayer``.

    ``primary_plate`` (PIL) paints the primary scene's meshes; each
    ProjectionSource paints its own meshes with its ``image_b64`` -- the
    full-resolution plate exactly as it projects (frame-outpainted and
    edge-extended where the layer asked for it, so its baked UVs line up).
    Analytic primitives (the backdrop plane) carry no mesh and are skipped.
    ``exr_dir`` set -> one EXR sidecar per plate, named in each material's
    ``extras``. Returns ``(layers, sidecars, notes)``.
    """
    from atlas_camera.core.proxy_geometry import PROXY_ROLE
    from atlas_camera.exporters._layers import mesh_from_primitive

    layers: list[SceneLayer] = []
    sidecars: list[dict[str, Any]] = []
    notes: list[str] = []
    exr_root = Path(exr_dir) if exr_dir else None

    def plate_entry(key: str, pil: Any, plate_ref: Any) -> dict[str, Any]:
        extras: dict[str, Any] = {"atlas_plate": key,
                                  "plate_size": [int(pil.size[0]), int(pil.size[1])]}
        if exr_root is None:
            return extras
        try:
            info = _write_exr_sidecar(exr_root / f"{exr_prefix}_{key}.exr", pil, plate_ref)
            extras.update(info)
            sidecars.append({"plate": key, **info})
        except Exception as exc:  # noqa: BLE001 - a sidecar never fails the GLB
            notes.append(f"EXR sidecar for {key} skipped: {type(exc).__name__}: {exc}")
        return extras

    def add_meshes(prims: list, pil: Any, extras: dict, prefix: str) -> None:
        png = _png_bytes(pil) if pil is not None else None
        for prim in prims:
            if prim.primitive_type != "mesh":
                continue
            mesh = mesh_from_primitive(prim)
            if mesh is None:
                continue
            uvs = mesh.uvs if getattr(mesh.uvs, "size", 0) else None
            layer_extras = {**extras, "atlas_layer": f"{prefix}{prim.name}"}
            hdr_vc = (prim.metadata or {}).get("vertex_colors_hdr")
            if hdr_vc and exr_root is not None and len(hdr_vc) == 3 * len(mesh.vertices):
                try:
                    ply = write_float_ply(
                        exr_root / f"{exr_prefix}_{prim.name}_vertex_hdr.ply",
                        mesh.vertices, mesh.faces, hdr_vc)
                    layer_extras.update(vertex_colors_hdr_ply=Path(ply).name,
                                        vertex_colors_hdr_space=(prim.metadata or {}).get(
                                            "vertex_colors_hdr_space", "ACEScg"))
                    sidecars.append({"plate": f"{prim.name} vertex colour",
                                     "exr": Path(ply).name,
                                     "exr_colorspace": "ACEScg (float PLY vertex colour)",
                                     "exr_origin": "AtlasHDRVertexTransfer",
                                     "scene_referred": False})
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"HDR vertex-colour PLY for {prim.name} skipped: {exc}")
            layers.append(SceneLayer(
                name=f"{prefix}{prim.name}", vertices=mesh.vertices, faces=mesh.faces,
                uvs=uvs, image_bytes=png, vertex_colors=mesh.vertex_colors,
                photo_weight=mesh.photo_weight,
                extras=layer_extras))

    scene = solve.projection_scene
    primary = [p for p in (scene.proxy_geometry or [])
               if (p.metadata or {}).get("role") == PROXY_ROLE]
    if primary:
        extras = plate_entry("primary", primary_plate, getattr(scene, "plate_ref", None))
        add_meshes(primary, primary_plate, extras, "")
    for src in getattr(solve, "projection_sources", None) or []:
        pil = _decode_data_uri(src.image_b64)
        if pil is None:
            notes.append(f"layer {src.name}: no plate image - written untextured")
            add_meshes(src.proxy_geometry or [], None, {"atlas_plate": None}, f"{src.name}/")
            continue
        extras = plate_entry(src.name, pil, src.plate_ref)
        pad = (src.metadata or {}).get("frame_outpaint_px")
        if pad:
            extras["frame_outpaint_px"] = int(pad)
        add_meshes(src.proxy_geometry or [], pil, extras, f"{src.name}/")
    return layers, sidecars, notes


def read_glb_json(path: str | Path) -> dict[str, Any]:
    """The JSON chunk of a GLB (tests / inspection)."""
    data = Path(path).read_bytes()
    (json_len,) = struct.unpack_from("<I", data, 12)
    return json.loads(data[20:20 + json_len])
