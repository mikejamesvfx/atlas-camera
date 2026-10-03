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

One glTF image + texture per SOURCE plate: every layer a plate paints shares
it (layers carrying the same ``image_bytes`` object are deduplicated), so a
scene of N primary meshes embeds the primary plate once, not N times. The
GLB's size is computed from the planned buffer views BEFORE the binary blob is
assembled, so an over-budget scene is refused without allocating it
(:class:`GLBBudgetError`); a GLB's length field is uint32, so 4 GiB is a hard
ceiling regardless of budget.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PNG_MIME = "image/png"
_JPEG_MIME = "image/jpeg"

#: A GLB header stores its total length as uint32.
GLB_MAX_BYTES = 0xFFFFFFFF


class GLBBudgetError(ValueError):
    """The planned GLB exceeds a byte budget; nothing was assembled or written.

    ``plan`` carries ``bytes`` (estimated file size), ``layers`` (per-layer
    geometry bytes) and ``images`` (per-plate embedded bytes, each counted
    once) so a caller can name what is large.
    """

    def __init__(self, message: str, plan: dict[str, Any]):
        super().__init__(message)
        self.plan = plan


def sanitize_name(name: Any, default: str = "layer") -> str:
    """A file-name-safe token: runs outside ``[A-Za-z0-9_-]`` become ``_``."""
    out = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name or "")).strip("_")
    return out or default


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


def _pad_len(n: int) -> int:
    return n + (4 - n % 4) % 4


def plan_scene_glb(layers: list[SceneLayer]) -> dict[str, Any]:
    """Prepare every layer's arrays and size the GLB, without assembling it.

    Returns ``{"prepared", "dropped", "layers", "images", "bin_bytes",
    "bytes"}``: ``prepared`` feeds :func:`write_scene_glb`; ``dropped`` lists
    layers that carry no geometry (``{"name", "reason"}``); ``layers`` /
    ``images`` are the per-layer geometry and per-plate image byte counts
    (an image shared by several layers is counted ONCE, against its plate);
    ``bytes`` is the estimated file size (binary chunk + headers; the JSON
    chunk is small and estimated).
    """
    import numpy as np

    from atlas_camera.core.generated_mesh import srgb_to_linear
    from atlas_camera.core.relief_mesh import ReliefMesh
    from atlas_camera.exporters.relief_mesh_exporter import (
        _generated_split,
        _topology_safe_faces,
    )

    prepared: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    layer_sizes: list[dict[str, Any]] = []
    images: dict[int, dict[str, Any]] = {}
    for layer in layers:
        verts = np.asarray(layer.vertices, dtype=np.float32).reshape(-1, 3)
        if not len(verts):
            dropped.append({"name": layer.name, "reason": "empty geometry (0 vertices)"})
            continue
        faces = _topology_safe_faces(verts, np.asarray(layer.faces).reshape(-1, 3))
        if not len(faces):
            dropped.append({"name": layer.name,
                            "reason": "zero faces after topology cleanup"})
            continue
        n = len(verts)
        untextured_reason = ""
        uvs = None if layer.uvs is None else np.asarray(layer.uvs, dtype=np.float32).reshape(-1, 2)
        if uvs is not None and len(uvs) != n:
            untextured_reason = (f"UV count {len(uvs)} != vertex count {n} - "
                                 "written untextured")
            uvs = None
        elif uvs is None and layer.image_bytes:
            untextured_reason = "no UVs - written untextured"

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
        groups = [(k, np.ascontiguousarray(np.asarray(g, dtype=np.uint32).reshape(-1)))
                  for k, g in groups]
        if not any(g.size for _, g in groups):
            dropped.append({"name": layer.name, "reason": "zero faces"})
            continue

        geo = _pad_len(verts.astype(np.float32).nbytes)
        if uvs is not None:
            geo += _pad_len(uvs.astype(np.float32).nbytes)
        if colors is not None:
            geo += _pad_len(colors.nbytes)
        geo += sum(_pad_len(g.nbytes) for _, g in groups if g.size)

        image_key = None
        if layer.image_bytes and uvs is not None:
            image_key = id(layer.image_bytes)
            if image_key not in images:
                images[image_key] = {
                    "name": str(layer.extras.get("atlas_plate") or layer.name),
                    "bytes": _pad_len(len(layer.image_bytes)), "layers": 0,
                    "data": layer.image_bytes, "mime": layer.image_mime}
            images[image_key]["layers"] += 1
        prepared.append({"layer": layer, "verts": verts, "uvs": uvs, "colors": colors,
                         "groups": groups, "image_key": image_key, "gen": gen is not None,
                         "untextured_reason": untextured_reason})
        layer_sizes.append({"name": layer.name, "bytes": int(geo)})

    bin_bytes = sum(x["bytes"] for x in layer_sizes) + sum(i["bytes"] for i in images.values())
    # JSON chunk: a generous per-view estimate keeps the pre-check honest.
    json_est = 2048 + 512 * len(prepared) + 256 * len(images)
    return {"prepared": prepared, "dropped": dropped, "layers": layer_sizes,
            "images": [{k: v for k, v in i.items() if k != "data"} for i in images.values()],
            "_images": images, "bin_bytes": int(bin_bytes),
            "bytes": int(12 + 8 + json_est + 8 + bin_bytes)}


def describe_glb_plan(plan: dict[str, Any]) -> str:
    """Per-layer and per-plate MB, largest first (for a refusal message)."""
    parts = [f"{x['name']} {x['bytes'] / 1e6:.1f} MB geometry"
             for x in sorted(plan["layers"], key=lambda x: -x["bytes"])]
    parts += [f"plate {i['name']} {i['bytes'] / 1e6:.1f} MB "
              f"(shared by {i['layers']} layer{'s' if i['layers'] != 1 else ''})"
              for i in sorted(plan["images"], key=lambda x: -x["bytes"])]
    return "; ".join(parts)


def write_scene_glb(layers: list[SceneLayer], path: str | Path, *,
                    generator: str = "AtlasCamera scene",
                    max_bytes: int | None = None) -> dict[str, Any]:
    """Write ``layers`` into one GLB at ``path``; returns a summary dict.

    ``max_bytes`` (None/0 = no budget) is checked against the PLANNED size
    before the binary blob is built; over it raises :class:`GLBBudgetError`
    and nothing is written. The summary's ``dropped`` lists layers with no
    geometry, and each layer's ``untextured_reason`` says why a layer that
    had a plate was written untextured.
    """
    import numpy as np

    plan = plan_scene_glb(layers)
    if not plan["prepared"]:
        raise ValueError("no layer had geometry to write"
                         + (" (dropped: " + "; ".join(f"{d['name']}: {d['reason']}"
                                                      for d in plan["dropped"]) + ")"
                            if plan["dropped"] else ""))
    limit = GLB_MAX_BYTES if not max_bytes else min(int(max_bytes), GLB_MAX_BYTES)
    if plan["bytes"] > limit:
        why = ("the GLB format's 4 GiB length limit" if limit == GLB_MAX_BYTES
               else f"the {limit / 1e6:.0f} MB budget")
        raise GLBBudgetError(
            f"the GLB would be ~{plan['bytes'] / 1e6:.0f} MB, over {why} "
            f"({len(plan['prepared'])} layers, {len(plan['images'])} embedded plate(s)): "
            + describe_glb_plan(plan), plan)

    blob = bytearray()
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    materials: list[dict] = []
    meshes: list[dict] = []
    nodes: list[dict] = []
    images: list[dict] = []
    textures: list[dict] = []
    summary: list[dict] = []
    texture_of: dict[int, int] = {}

    def add_view(data: Any, target: int | None = None) -> int:
        offset = len(blob)
        blob.extend(data)
        n = len(blob) - offset
        blob.extend(b"\x00" * ((4 - n % 4) % 4))
        view = {"buffer": 0, "byteOffset": offset, "byteLength": n}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return len(buffer_views) - 1

    def add_accessor(arr: Any, kind: str, component: int, *, minmax: bool = False,
                     target: int | None = 34962) -> int:
        raw = np.ascontiguousarray(arr).reshape(-1).view(np.uint8)  # no tobytes() copy
        acc = {"bufferView": add_view(raw, target),
               "componentType": component,
               "count": int(arr.shape[0]) if kind != "SCALAR" else int(arr.size), "type": kind}
        if minmax:
            acc["min"] = [float(v) for v in arr.min(axis=0)]
            acc["max"] = [float(v) for v in arr.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    def unlit(name: str, *, texture: int | None, extras: dict) -> int:
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

    for item in plan["prepared"]:
        layer, verts, uvs, colors = item["layer"], item["verts"], item["uvs"], item["colors"]
        attrs = {"POSITION": add_accessor(verts, "VEC3", 5126, minmax=True)}
        if uvs is not None:
            st = uvs.copy()
            st[:, 1] = 1.0 - st[:, 1]  # OBJ bottom-left -> glTF top-left
            attrs["TEXCOORD_0"] = add_accessor(st, "VEC2", 5126)
        if colors is not None:
            attrs["COLOR_0"] = add_accessor(colors, "VEC4", 5126)

        texture = None
        key = item["image_key"]
        if key is not None:
            if key not in texture_of:
                img = plan["_images"][key]
                images.append({"bufferView": add_view(img["data"]),
                               "mimeType": img["mime"], "name": img["name"]})
                textures.append({"source": len(images) - 1, "sampler": 0})
                texture_of[key] = len(textures) - 1
            texture = texture_of[key]

        primitives = []
        for kind, idx in item["groups"]:
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
        meshes.append({"name": layer.name, "primitives": primitives})
        nodes.append({"mesh": len(meshes) - 1, "name": layer.name})
        entry = {"name": layer.name, "vertices": int(len(verts)),
                 "faces": int(sum(g.size // 3 for _, g in item["groups"])),
                 "textured": texture is not None, "vertex_colour": item["gen"]}
        if item["untextured_reason"]:
            entry["untextured_reason"] = item["untextured_reason"]
        summary.append(entry)

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
    total = 12 + 8 + len(json_chunk) + 8 + len(blob)
    if total > GLB_MAX_BYTES:  # the plan's JSON estimate was short (pathological)
        raise GLBBudgetError(f"the GLB is {total / 1e6:.0f} MB, over the GLB format's "
                             "4 GiB length limit", plan)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
        fh.write(json_chunk)
        fh.write(struct.pack("<II", len(blob), 0x004E4942))
        fh.write(blob)  # the bytearray itself: no bytes(blob) second copy
    return {"glb": str(out), "bytes": total, "layers": summary,
            "dropped": plan["dropped"], "images": len(images)}


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
    output_root: str | Path | None = None,
) -> tuple[list[SceneLayer], list[dict[str, Any]], list[str]]:
    """Every projection layer of ``solve`` as a ``SceneLayer``.

    ``primary_plate`` (PIL) paints the primary scene's meshes; each
    ProjectionSource paints its own meshes with its ``image_b64`` -- the
    full-resolution plate exactly as it projects (frame-outpainted and
    edge-extended where the layer asked for it, so its baked UVs line up).
    Analytic primitives (the backdrop plane) carry no mesh and are skipped.
    ``exr_dir`` set -> one EXR sidecar per plate, named in each material's
    ``extras``. With ``output_root`` (ComfyUI's output directory) each
    sidecar reference also records ``*_output_path``: its path relative to that
    root, so a COPY of the GLB elsewhere in the output tree (Save 3D puts one in
    ``3d/``) can still find the plates; a bare name only resolves beside the
    original. Returns ``(layers, sidecars, notes)``.
    """
    from atlas_camera.core.proxy_geometry import PROXY_ROLE
    from atlas_camera.exporters._layers import mesh_from_primitive

    layers: list[SceneLayer] = []
    sidecars: list[dict[str, Any]] = []
    notes: list[str] = []
    exr_root = Path(exr_dir) if exr_dir else None
    used_names: set[str] = set()
    out_root = Path(output_root).resolve() if output_root else None

    def output_rel(path: Path) -> str | None:
        """``path`` relative to the output root (posix), or None outside it."""
        if out_root is None:
            return None
        try:
            return Path(path).resolve().relative_to(out_root).as_posix()
        except ValueError:
            return None

    def sidecar_name(key: str, suffix: str) -> str:
        """``<prefix>_<key><suffix>`` with ``key`` sanitised to [A-Za-z0-9_-]
        (layer names come from the solve, never trusted as path parts) and
        made unique (``_2``, ``_3``) so two layers never overwrite a file."""
        stem = f"{exr_prefix}_{sanitize_name(key)}"
        name, n = f"{stem}{suffix}", 2
        while name.lower() in used_names:
            name, n = f"{stem}_{n}{suffix}", n + 1
        used_names.add(name.lower())
        return name

    def plate_entry(key: str, pil: Any, plate_ref: Any) -> dict[str, Any]:
        extras: dict[str, Any] = {"atlas_plate": key,
                                  "plate_size": [int(pil.size[0]), int(pil.size[1])]}
        if exr_root is None:
            return extras
        try:
            exr_root.mkdir(parents=True, exist_ok=True)
            exr_path = exr_root / sidecar_name(key, ".exr")
            info = _write_exr_sidecar(exr_path, pil, plate_ref)
            rel = output_rel(exr_path)
            if rel:
                info["exr_output_path"] = rel
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
                notes.append(f"layer {prefix}{prim.name}: dropped - no mesh payload "
                             "(vertices/faces) on the primitive")
                continue
            uvs = mesh.uvs if getattr(mesh.uvs, "size", 0) else None
            layer_extras = {**extras, "atlas_layer": f"{prefix}{prim.name}"}
            hdr_vc = (prim.metadata or {}).get("vertex_colors_hdr")
            if hdr_vc and exr_root is not None and len(hdr_vc) == 3 * len(mesh.vertices):
                try:
                    ply = write_float_ply(
                        exr_root / sidecar_name(prim.name, "_vertex_hdr.ply"),
                        mesh.vertices, mesh.faces, hdr_vc)
                    rel = output_rel(Path(ply))
                    if rel:
                        layer_extras["vertex_colors_hdr_ply_output_path"] = rel
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
        try:
            pil = _decode_data_uri(src.image_b64)
        except Exception as exc:  # noqa: BLE001 - a bad plate never fails the scene
            notes.append(f"layer {src.name}: plate image_b64 could not be decoded "
                         f"({type(exc).__name__}: {exc}) - written untextured")
            pil = None
        else:
            if pil is None:
                notes.append(f"layer {src.name}: no plate image - written untextured")
        if pil is None:
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
