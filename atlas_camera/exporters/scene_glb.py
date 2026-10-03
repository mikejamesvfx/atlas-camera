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
GLB's size is computed EXACTLY (planned buffer views + the serialised glTF
JSON) before a byte is written, so an over-budget scene is refused without
writing it (:class:`GLBBudgetError`), and the written file is checked again.
A GLB's length field is uint32, so 4 GiB is a hard ceiling regardless of
budget.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas_camera.exporters._glb import pad4, write_glb_header

_PNG_MIME = "image/png"
_JPEG_MIME = "image/jpeg"

#: A GLB header stores its total length as uint32.
GLB_MAX_BYTES = 0xFFFFFFFF


class GLBBudgetError(ValueError):
    """The GLB exceeds a byte budget; no GLB is left on disk.

    ``plan`` carries ``bytes`` (the planned or written file size), ``layers`` (per-layer
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


_pad4 = pad4  # historical private name; the framing lives in exporters._glb


def _pad_len(n: int) -> int:
    return n + (4 - n % 4) % 4


def plan_scene_glb(layers: list[SceneLayer], *,
                   generator: str = "AtlasCamera scene") -> dict[str, Any]:
    """Prepare every layer's arrays and size the GLB, without writing it.

    Returns ``{"prepared", "dropped", "layers", "images", "json_bytes",
    "bin_bytes", "bytes"}``: ``prepared`` feeds :func:`write_scene_glb`;
    ``dropped`` lists layers that carry no geometry (``{"name", "reason"}``);
    ``layers`` / ``images`` are the per-layer geometry and per-plate image
    byte counts (an image shared by several layers is counted ONCE, against
    its plate). ``bytes`` is the EXACT file size: the plan lays out every
    buffer view and serialises the real glTF JSON (names, extras, accessor
    min/max), so a 400 kB layer name costs 400 kB here too (``json_bytes``).
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

        geo = _pad_len(verts.astype(np.float32, copy=False).nbytes)
        if uvs is not None:
            geo += _pad_len(uvs.astype(np.float32, copy=False).nbytes)
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

    layout = _layout_gltf(prepared, images, generator)
    json_chunk = _pad4(json.dumps(layout["gltf"], separators=(",", ":")).encode("utf-8"),
                       b" ")
    bin_bytes = layout["bin_bytes"]
    return {"prepared": prepared, "dropped": dropped, "layers": layer_sizes,
            "images": [{k: v for k, v in i.items() if k != "data"} for i in images.values()],
            "_images": images, "_layout": layout, "_json_chunk": json_chunk,
            "json_bytes": len(json_chunk), "bin_bytes": int(bin_bytes),
            # EXACT: the very JSON chunk and buffer layout write_scene_glb streams.
            "bytes": int(12 + 8 + len(json_chunk) + 8 + bin_bytes)}


def _layout_gltf(prepared: list[dict[str, Any]], images_by_key: dict[int, dict[str, Any]],
                 generator: str) -> dict[str, Any]:
    """The complete glTF JSON dict + the ordered binary chunks, WITHOUT the blob.

    Every buffer view's offset and length follows from the arrays' byte sizes,
    so the JSON (names, extras, accessor min/max and all) is final here and its
    serialised length is the GLB's real JSON chunk. ``chunks`` are
    ``(nbytes, producer)`` in buffer order; :func:`write_scene_glb` streams
    them, so the binary chunk is never assembled in memory either.
    """
    import numpy as np

    chunks: list[tuple[int, Any]] = []
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    materials: list[dict] = []
    meshes: list[dict] = []
    nodes: list[dict] = []
    images: list[dict] = []
    textures: list[dict] = []
    summary: list[dict] = []
    texture_of: dict[int, int] = {}
    cursor = 0

    def add_view(nbytes: int, producer: Any, target: int | None = None) -> int:
        nonlocal cursor
        view = {"buffer": 0, "byteOffset": cursor, "byteLength": int(nbytes)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        chunks.append((int(nbytes), producer))
        cursor += _pad_len(int(nbytes))
        return len(buffer_views) - 1

    def add_accessor(arr: Any, kind: str, component: int, *, minmax: bool = False,
                     target: int | None = 34962, producer: Any = None) -> int:
        acc = {"bufferView": add_view(arr.nbytes, producer or (lambda a=arr: a), target),
               "componentType": component,
               "count": int(arr.shape[0]) if kind != "SCALAR" else int(arr.size), "type": kind}
        if minmax:
            acc["min"] = [float(v) for v in arr.min(axis=0)]
            acc["max"] = [float(v) for v in arr.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    def flipped_uvs(uvs: Any) -> Any:
        st = uvs.copy()
        st[:, 1] = 1.0 - st[:, 1]  # OBJ bottom-left -> glTF top-left
        return st

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

    for item in prepared:
        layer, verts, uvs, colors = item["layer"], item["verts"], item["uvs"], item["colors"]
        verts = np.ascontiguousarray(verts, dtype=np.float32)
        attrs = {"POSITION": add_accessor(verts, "VEC3", 5126, minmax=True)}
        if uvs is not None:
            uvs = np.ascontiguousarray(uvs, dtype=np.float32)
            attrs["TEXCOORD_0"] = add_accessor(uvs, "VEC2", 5126,
                                               producer=lambda u=uvs: flipped_uvs(u))
        if colors is not None:
            attrs["COLOR_0"] = add_accessor(np.ascontiguousarray(colors, dtype=np.float32),
                                            "VEC4", 5126)

        texture = None
        key = item["image_key"]
        if key is not None:
            if key not in texture_of:
                img = images_by_key[key]
                images.append({"bufferView": add_view(len(img["data"]),
                                                      lambda d=img["data"]: d),
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
        "buffers": [{"byteLength": int(cursor)}],
    }
    if images:
        gltf["images"] = images
        gltf["textures"] = textures
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987,
                             "wrapS": 33071, "wrapT": 33071}]
    return {"gltf": gltf, "chunks": chunks, "bin_bytes": int(cursor), "summary": summary,
            "images": len(images)}


def describe_glb_plan(plan: dict[str, Any]) -> str:
    """Per-layer and per-plate MB, largest first, plus the glTF JSON chunk
    (for a refusal message; names are clipped so a pathological one cannot
    swamp it)."""
    parts = [f"{x['name'][:80]} {x['bytes'] / 1e6:.1f} MB geometry"
             for x in sorted(plan["layers"], key=lambda x: -x["bytes"])]
    parts += [f"plate {i['name'][:80]} {i['bytes'] / 1e6:.1f} MB "
              f"(shared by {i['layers']} layer{'s' if i['layers'] != 1 else ''})"
              for i in sorted(plan["images"], key=lambda x: -x["bytes"])]
    if plan.get("json_bytes"):
        parts.append(f"glTF JSON {plan['json_bytes'] / 1e6:.1f} MB")
    return "; ".join(parts)


def glb_byte_limit(max_bytes: int | None) -> int:
    """The effective byte limit: ``max_bytes`` (None/0 = none) capped at 4 GiB."""
    return GLB_MAX_BYTES if not max_bytes else min(int(max_bytes), GLB_MAX_BYTES)


def format_mb(nbytes: int) -> str:
    """MB for a message: one decimal below 10 MB (so 1.2 MB is not "1 MB")."""
    return f"{nbytes / 1e6:.0f}" if nbytes >= 10_000_000 else f"{nbytes / 1e6:.1f}"


def _limit_phrase(limit: int) -> str:
    if limit == GLB_MAX_BYTES:
        return "the GLB format's 4 GiB length limit"
    return (f"the {limit / 1e6:.0f} MB budget" if limit >= 1_000_000
            else f"the {limit} byte budget")


def check_glb_budget(plan: dict[str, Any], max_bytes: int | None, *,
                     headroom: int = 0) -> None:
    """Raise :class:`GLBBudgetError` when ``plan['bytes'] + headroom`` is over
    the limit. ``headroom`` is JSON the layers will still gain (e.g. sidecar
    references not yet written into their extras) -- a conservative margin."""
    limit = glb_byte_limit(max_bytes)
    total = int(plan["bytes"]) + int(headroom)
    if total > limit:
        raise GLBBudgetError(
            f"the GLB would be ~{format_mb(total)} MB, over {_limit_phrase(limit)} "
            f"({len(plan['prepared'])} layers, {len(plan['images'])} embedded plate(s)): "
            + describe_glb_plan(plan), {**plan, "bytes": total})


def write_scene_glb(layers: list[SceneLayer], path: str | Path, *,
                    generator: str = "AtlasCamera scene",
                    max_bytes: int | None = None) -> dict[str, Any]:
    """Write ``layers`` into one GLB at ``path``; returns a summary dict.

    ``max_bytes`` (None/0 = no budget; 4 GiB always applies) is enforced
    twice: against the PLANNED size -- exact, the plan serialises the real
    glTF JSON -- before a byte is written, and against the FINAL file size
    (over it, the written file is deleted). Both raise :class:`GLBBudgetError`
    naming the size. The binary chunk is streamed from the arrays, never
    assembled. The summary's ``dropped`` lists layers with no geometry, and
    each layer's ``untextured_reason`` says why a layer that had a plate was
    written untextured.
    """
    import numpy as np

    plan = plan_scene_glb(layers, generator=generator)
    if not plan["prepared"]:
        raise ValueError("no layer had geometry to write"
                         + (" (dropped: " + "; ".join(f"{d['name']}: {d['reason']}"
                                                      for d in plan["dropped"]) + ")"
                            if plan["dropped"] else ""))
    check_glb_budget(plan, max_bytes)

    layout, json_chunk = plan["_layout"], plan["_json_chunk"]
    bin_bytes = layout["bin_bytes"]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, "wb") as fh:
            write_glb_header(fh, json_chunk, bin_bytes)
            for nbytes, producer in layout["chunks"]:
                data = producer()
                if isinstance(data, np.ndarray):  # a view, no tobytes() copy
                    data = np.ascontiguousarray(data).reshape(-1).view(np.uint8)
                fh.write(data)
                fh.write(b"\x00" * ((4 - nbytes % 4) % 4))
    except BaseException:
        out.unlink(missing_ok=True)  # never leave a truncated GLB behind
        raise
    size = out.stat().st_size
    limit = glb_byte_limit(max_bytes)
    if size > limit:  # the plan is exact; this guards the plan itself
        out.unlink(missing_ok=True)
        raise GLBBudgetError(
            f"the written GLB is {size / 1e6:.1f} MB ({size} bytes), over "
            f"{_limit_phrase(limit)} - deleted: " + describe_glb_plan(plan),
            {**plan, "bytes": size})
    return {"glb": str(out), "bytes": size, "layers": layout["summary"],
            "dropped": plan["dropped"], "images": layout["images"]}


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


#: JSON allowance per sidecar reference for fields only known once the file is
#: written (``exr_colorspace`` read from a float plate_ref, ``exr_origin``,
#: ``scene_referred``) -- colourspace names are short; this is generous.
_SIDECAR_UNKNOWN_EXTRAS_BYTES = 256


@dataclass
class _SidecarJob:
    """One sidecar not yet written: its file, the extras it will add to every
    layer it describes, and how to write it."""

    path: Path
    targets: list[dict[str, Any]]
    known: dict[str, Any]
    write: Any                       # () -> (extras info, sidecars entry)
    fail_note: Any                   # (exc) -> note


@dataclass
class SceneCollection:
    """Scene layers with their sidecars PLANNED but not yet written.

    :func:`collect_scene_layers` builds the layers (plates PNG-encoded once)
    and names every sidecar without touching the disk, so a caller can size
    the GLB -- ``plan_scene_glb(layers)`` plus :meth:`extras_headroom` -- and
    refuse before any file exists. :meth:`write_sidecars` then writes them and
    adds their references to the layers' extras. A failed sidecar is a note,
    never an error (a sidecar never fails the GLB).
    """

    layers: list[SceneLayer]
    notes: list[str]
    sidecars: list[dict[str, Any]] = field(default_factory=list)
    jobs: list[_SidecarJob] = field(default_factory=list)

    def extras_headroom(self) -> int:
        """Upper bound on the JSON bytes :meth:`write_sidecars` adds to the GLB."""
        total = 0
        for job in self.jobs:
            per = (len(json.dumps(job.known, separators=(",", ":")))
                   + _SIDECAR_UNKNOWN_EXTRAS_BYTES)
            total += per * len(job.targets)
        return total

    def write_sidecars(self) -> list[dict[str, Any]]:
        """Write every planned sidecar (once); returns ``sidecars``."""
        jobs, self.jobs = self.jobs, []
        for job in jobs:
            try:
                info, entry = job.write()
            except Exception as exc:  # noqa: BLE001 - a sidecar never fails the GLB
                self.notes.append(job.fail_note(exc))
                continue
            for extras in job.targets:
                extras.update(info)
            self.sidecars.append(entry)
        return self.sidecars


def build_scene_layers(
    solve: Any,
    primary_plate: Any,
    *,
    exr_dir: str | Path | None = None,
    exr_prefix: str = "atlas_scene",
    output_root: str | Path | None = None,
) -> tuple[list[SceneLayer], list[dict[str, Any]], list[str]]:
    """Every projection layer of ``solve`` as a ``SceneLayer``, sidecars written.

    :func:`collect_scene_layers` + :meth:`SceneCollection.write_sidecars` in
    one call. Returns ``(layers, sidecars, notes)``.
    """
    scene = collect_scene_layers(solve, primary_plate, exr_dir=exr_dir,
                                 exr_prefix=exr_prefix, output_root=output_root)
    scene.write_sidecars()
    return scene.layers, scene.sidecars, scene.notes


def collect_scene_layers(
    solve: Any,
    primary_plate: Any,
    *,
    exr_dir: str | Path | None = None,
    exr_prefix: str = "atlas_scene",
    output_root: str | Path | None = None,
) -> SceneCollection:
    """Every projection layer of ``solve`` as a ``SceneLayer``; NO file written.

    ``primary_plate`` (PIL) paints the primary scene's meshes; each
    ProjectionSource paints its own meshes with its ``image_b64`` -- the
    full-resolution plate exactly as it projects (frame-outpainted and
    edge-extended where the layer asked for it, so its baked UVs line up).
    Analytic primitives (the backdrop plane) carry no mesh and are skipped.
    ``exr_dir`` set -> one EXR sidecar per plate (and a float PLY per HDR
    vertex-coloured mesh) is PLANNED, to be named in each material's
    ``extras`` when :meth:`SceneCollection.write_sidecars` writes it. With
    ``output_root`` (ComfyUI's output directory) each sidecar reference also
    records ``*_output_path``: its path relative to that root, so a COPY of
    the GLB elsewhere in the output tree (Save 3D puts one in ``3d/``) can
    still find the plates; a bare name only resolves beside the original.
    """
    from atlas_camera.core.proxy_geometry import PROXY_ROLE
    from atlas_camera.exporters._layers import mesh_from_primitive

    layers: list[SceneLayer] = []
    notes: list[str] = []
    scene = SceneCollection(layers=layers, notes=notes)
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

    def plate_entry(key: str, pil: Any, plate_ref: Any) -> tuple[dict[str, Any], Any]:
        """``(extras, job)``: the plate's base extras and its planned EXR (or None)."""
        extras: dict[str, Any] = {"atlas_plate": key,
                                  "plate_size": [int(pil.size[0]), int(pil.size[1])]}
        if exr_root is None:
            return extras, None
        exr_path = exr_root / sidecar_name(key, ".exr")
        rel = output_rel(exr_path)

        def write() -> tuple[dict[str, Any], dict[str, Any]]:
            exr_root.mkdir(parents=True, exist_ok=True)
            info = _write_exr_sidecar(exr_path, pil, plate_ref)
            if rel:
                info["exr_output_path"] = rel
            return info, {"plate": key, **info}

        known = {"exr": exr_path.name, **({"exr_output_path": rel} if rel else {})}
        job = _SidecarJob(path=exr_path, targets=[], known=known, write=write,
                          fail_note=lambda exc: (f"EXR sidecar for {key} skipped: "
                                                 f"{type(exc).__name__}: {exc}"))
        scene.jobs.append(job)
        return extras, job

    def add_meshes(prims: list, pil: Any, extras: dict, prefix: str, job: Any) -> None:
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
            if job is not None:
                job.targets.append(layer_extras)
            hdr_vc = (prim.metadata or {}).get("vertex_colors_hdr")
            if hdr_vc and exr_root is not None and len(hdr_vc) == 3 * len(mesh.vertices):
                scene.jobs.append(_ply_job(prim, mesh, hdr_vc, layer_extras))
            layers.append(SceneLayer(
                name=f"{prefix}{prim.name}", vertices=mesh.vertices, faces=mesh.faces,
                uvs=uvs, image_bytes=png, vertex_colors=mesh.vertex_colors,
                photo_weight=mesh.photo_weight,
                extras=layer_extras))

    def _ply_job(prim: Any, mesh: Any, hdr_vc: Any, layer_extras: dict) -> _SidecarJob:
        ply_path = exr_root / sidecar_name(prim.name, "_vertex_hdr.ply")
        rel = output_rel(ply_path)
        space = (prim.metadata or {}).get("vertex_colors_hdr_space", "ACEScg")
        known: dict[str, Any] = ({"vertex_colors_hdr_ply_output_path": rel} if rel else {})
        known.update(vertex_colors_hdr_ply=ply_path.name, vertex_colors_hdr_space=space)

        def write() -> tuple[dict[str, Any], dict[str, Any]]:
            write_float_ply(ply_path, mesh.vertices, mesh.faces, hdr_vc)
            return dict(known), {"plate": f"{prim.name} vertex colour",
                                 "exr": ply_path.name,
                                 "exr_colorspace": "ACEScg (float PLY vertex colour)",
                                 "exr_origin": "AtlasHDRVertexTransfer",
                                 "scene_referred": False}

        return _SidecarJob(path=ply_path, targets=[layer_extras], known=known, write=write,
                           fail_note=lambda exc: (f"HDR vertex-colour PLY for {prim.name} "
                                                  f"skipped: {exc}"))

    solve_scene = solve.projection_scene
    primary = [p for p in (solve_scene.proxy_geometry or [])
               if (p.metadata or {}).get("role") == PROXY_ROLE]
    if primary:
        extras, job = plate_entry("primary", primary_plate,
                                  getattr(solve_scene, "plate_ref", None))
        add_meshes(primary, primary_plate, extras, "", job)
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
            add_meshes(src.proxy_geometry or [], None, {"atlas_plate": None},
                       f"{src.name}/", None)
            continue
        extras, job = plate_entry(src.name, pil, src.plate_ref)
        pad = (src.metadata or {}).get("frame_outpaint_px")
        if pad:
            extras["frame_outpaint_px"] = int(pad)
        add_meshes(src.proxy_geometry or [], pil, extras, f"{src.name}/", job)
    return scene


def read_glb_json(path: str | Path) -> dict[str, Any]:
    """The JSON chunk of a GLB (tests / inspection)."""
    data = Path(path).read_bytes()
    (json_len,) = struct.unpack_from("<I", data, 12)
    return json.loads(data[20:20 + json_len])
