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


def _ribbon_vertex_flags(mesh: ReliefMesh) -> Any:
    """``ribbon_t > 0`` per vertex, or None when the mesh carries no ribbon.

    Ring 0 sits at ``ribbon_t == 0`` and is therefore invisible to this test on
    its own — which is exactly why the FACE test below is what the exporters
    use. A ring-0 vertex is only ever referenced by ribbon quads, so a face
    touching any ``t > 0`` vertex is a ribbon face and the classification is
    complete.
    """
    values = getattr(mesh, "ribbon_t", None)
    if values is None:
        return None
    import numpy as np

    flags = np.asarray(values).reshape(-1) > 0.0
    if len(flags) != len(mesh.vertices) or not flags.any():
        return None
    return flags


def _ribbon_face_flags(mesh: ReliefMesh, ribbon_vert: Any) -> Any:
    if ribbon_vert is None:
        return None
    import numpy as np

    return ribbon_vert[np.asarray(mesh.faces)].any(axis=1)


def _ribbon_smudged_colors(mesh: ReliefMesh, texture: Any, smudge_px: float) -> Any:
    """Per-ribbon-vertex colour: the plate averaged ALONG the rim, widening with t.

    The CPU twin of the viewport's smudge, so a DCC sees the same softening
    instead of the hard radial streaks a single frozen texel per column gives.
    Returns an (N,3) float32 array, white on non-ribbon vertices.

    Direction comes from the mesh itself: within a column the frozen UV is
    constant, so a ribbon neighbour on the SAME ring necessarily belongs to an
    adjacent column and the UV difference points along the silhouette. No
    tangent needs storing.
    """
    import numpy as np

    rt = np.asarray(getattr(mesh, "ribbon_t", None), dtype=np.float64).reshape(-1)
    verts = np.asarray(mesh.vertices)
    colors = np.ones((len(verts), 3), dtype=np.float32)
    if len(rt) != len(verts) or smudge_px <= 0.0 or texture is None:
        return colors
    is_rib = rt > 0.0
    if not is_rib.any():
        return colors

    img = np.asarray(texture.convert("RGB"), dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    uvs = np.asarray(mesh.uvs, dtype=np.float64)
    px = uvs[:, 0] * (w - 1)
    py = (1.0 - uvs[:, 1]) * (h - 1)

    faces = np.asarray(mesh.faces, dtype=np.int64)
    rib_faces = faces[is_rib[faces].any(axis=1)]
    # Same-ring neighbour offsets, accumulated per vertex.
    dir_u = np.zeros(len(verts), dtype=np.float64)
    dir_v = np.zeros(len(verts), dtype=np.float64)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        va, vb = rib_faces[:, a], rib_faces[:, b]
        same = (np.abs(rt[va] - rt[vb]) < 1e-6) & is_rib[va] & is_rib[vb]
        if not same.any():
            continue
        sa, sb = va[same], vb[same]
        np.add.at(dir_u, sa, px[sb] - px[sa])
        np.add.at(dir_v, sa, py[sb] - py[sa])
        np.add.at(dir_u, sb, px[sa] - px[sb])
        np.add.at(dir_v, sb, py[sa] - py[sb])

    length = np.hypot(dir_u, dir_v)
    ok = is_rib & (length > 1e-9)
    step_u = np.zeros(len(verts))
    step_v = np.zeros(len(verts))
    reach = float(smudge_px) * np.clip(rt, 0.0, 1.0)
    step_u[ok] = dir_u[ok] / length[ok] * reach[ok]
    step_v[ok] = dir_v[ok] / length[ok] * reach[ok]

    idx = np.nonzero(is_rib)[0]
    acc = np.zeros((len(idx), 3), dtype=np.float64)
    for k in (-2.0, -1.0, 0.0, 1.0, 2.0):
        sx = np.clip(np.rint(px[idx] + step_u[idx] * k), 0, w - 1).astype(np.int64)
        sy = np.clip(np.rint(py[idx] + step_v[idx] * k), 0, h - 1).astype(np.int64)
        acc += img[sy, sx]
    colors[idx] = (acc / 5.0).astype(np.float32)
    return colors


def _obj_ribbon_colors(mesh: ReliefMesh, texture: Any,
                       texture_path: str | Path | None) -> Any:
    """Per-vertex colours for the OBJ, or None when there is nothing to bake.

    Returns white everywhere except the transition ribbon. ``texture_path``
    (a file-backed plate, often EXR) is opened only as a fallback and only when
    Pillow can read it — a float EXR handoff must not turn a working OBJ export
    into a traceback just because the skirt would have looked nicer.
    """
    smudge = float(((mesh.stats or {}).get("transition_ribbon") or {})
                   .get("smudge_px", 0.0) or 0.0)
    if smudge <= 0.0 or _ribbon_vertex_flags(mesh) is None:
        return None
    image = texture
    if image is None and texture_path is not None:
        try:
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = None
            image = Image.open(texture_path)
        except Exception:
            return None
    if image is None:
        return None
    try:
        return _ribbon_smudged_colors(mesh, image, smudge)
    except Exception:
        return None


def export_relief_mesh(
    mesh: ReliefMesh,
    output_dir: str | Path,
    *,
    texture: Any | None = None,
    texture_path: str | Path | None = None,
    name: str = "atlas_relief_mesh",
) -> dict[str, str]:
    """Write ``{name}.obj`` + ``{name}.mtl`` (+ texture PNG) to ``output_dir``.

    ``texture_path`` references an existing source plate directly, preserving
    EXR/float file-backed workflows. If no path is supplied, ``texture`` is an
    optional PIL Image saved next to the OBJ as a PNG preview and referenced as
    ``map_Kd``. Returns the written paths. Coordinates are Atlas world
    (right-handed, Y-up, metres) — Maya and Nuke default conventions.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    obj_path = out / f"{name}.obj"
    mtl_path = out / f"{name}.mtl"
    material = "atlas_relief_projection"

    tex_path: Path | None = Path(texture_path) if texture_path else None
    tex_written = False
    if tex_path is None and texture is not None:
        tex_path = out / f"{name}_diffuse.png"
        texture.save(tex_path)
        tex_written = True

    lines: list[str] = [
        "# Atlas Camera relief mesh — Y-up, metres.",
        "# UVs bake the recovered-camera projection: the referenced texture is",
        "# already correctly projected; retopo/reproject as needed.",
        f"mtllib {mtl_path.name}",
        f"o {name}",
    ]
    # Smudged skirt colour, when there is one, as OBJ VERTEX COLOURS.
    #
    # OBJ carries a single UV set and the ribbon's UVs must keep pointing at the
    # plate (the viewport samples them there), so remapping the skirt into a
    # baked strip atlas — the tidiest option in the abstract — is not available.
    # The GLB route is not available either: that leans on COLOR_0 plus a second
    # primitive, and OBJ has no per-vertex colour channel in the base format.
    # What is left is the widely-implemented `v x y z r g b` extension.
    ribbon_colors = _obj_ribbon_colors(mesh, texture, texture_path)
    if ribbon_colors is None:
        lines.extend(
            f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in mesh.vertices
        )
    else:
        lines.insert(3, "# vertex colours (v x y z r g b): transition-ribbon smudge")
        lines.extend(
            f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
            for v, c in zip(mesh.vertices, ribbon_colors)
        )
    lines.extend(
        f"vt {t[0]:.6f} {t[1]:.6f}" for t in mesh.uvs
    )
    # Vertex and UV lists are 1:1, so face indices serve both (v/vt).
    def _face(tri: Any) -> str:
        a, b, c = tri
        return f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}"

    # Transition-ribbon faces go under their own material. OBJ has no per-vertex
    # alpha, so the gradient itself cannot survive here (the GLB export bakes it
    # into COLOR_0) — but a separate material is what lets a TD dial the skirt
    # back, isolate it, or delete it in one click instead of hunting for loose
    # triangles. Grouping only kicks in when a ribbon exists, so an ordinary
    # export keeps its historical single-group face order.
    ribbon_material = f"{material}_transition_ribbon"
    ribbon_vert = _ribbon_vertex_flags(mesh)
    ribbon_face = _ribbon_face_flags(mesh, ribbon_vert)
    if ribbon_face is not None and ribbon_face.any():
        lines.append(f"usemtl {material}")
        lines.extend(_face(t) for t in mesh.faces[~ribbon_face])
        lines.append(f"usemtl {ribbon_material}")
        lines.extend(_face(t) for t in mesh.faces[ribbon_face])
    else:
        ribbon_face = None
        lines.append(f"usemtl {material}")
        lines.extend(_face(t) for t in mesh.faces)
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _material_block(mat_name: str, *, textured: bool = True,
                        kd: Any = None) -> list[str]:
        base = "1.000 1.000 1.000" if kd is None else \
            f"{kd[0]:.3f} {kd[1]:.3f} {kd[2]:.3f}"
        block = [
            f"newmtl {mat_name}",
            f"Kd {base}",
            "Ka 0.000 0.000 0.000",
            "Ks 0.000 0.000 0.000",
            "illum 1",
        ]
        if textured and tex_path is not None:
            map_path = tex_path.name if tex_written else tex_path.as_posix()
            block.append(f"map_Kd {map_path}")
        return block

    mtl_lines = _material_block(material)
    if ribbon_face is not None:
        if ribbon_colors is None:
            # No smudge baked: the skirt keeps the plate, which samples its
            # frozen rim texel and gives the hard edge-extend.
            mtl_lines.extend(_material_block(ribbon_material))
        else:
            # With vertex colours present the skirt material must NOT also carry
            # map_Kd. OBJ does not define whether a reader multiplies vertex
            # colour by the map or replaces it, so shipping both means the
            # smudge is either right or doubled-and-darkened depending on the
            # importer. Dropping the map makes it unambiguous, and Kd carries
            # the mean skirt colour so a reader that ignores vertex colours
            # still gets a plausible flat edge tone rather than white.
            import numpy as np

            mean_kd = np.asarray(
                ribbon_colors[np.asarray(ribbon_vert)], dtype=np.float64
            ).mean(axis=0)
            mtl_lines.extend(
                _material_block(ribbon_material, textured=False, kd=mean_kd))
    mtl_path.write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")

    result = {"obj": str(obj_path), "mtl": str(mtl_path)}
    if tex_path is not None:
        result["texture"] = str(tex_path)
        if not tex_written:
            result["texture_external"] = "true"
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

    # Transition ribbon: bake the EVALUATED fade, not the raw parameter. The
    # viewport applies smoothstep(RIBBON_FADE_START, 1, ribbon_t) in GLSL, so
    # exporting ribbon_t itself would hand a DCC a linear ramp where the
    # viewport shows an S-curve — the two would disagree about the same mesh.
    # One curve, evaluated in `core.transition_ribbon.ribbon_alpha`.
    colors = None
    ribbon_face_mask = None
    ribbon_values = getattr(mesh, "ribbon_t", None)
    if ribbon_values is not None:
        from atlas_camera.core.transition_ribbon import ribbon_alpha

        t = np.asarray(ribbon_values, dtype=np.float32).reshape(-1)
        if len(t) == len(verts) and bool((t > 0.0).any()):
            colors = np.ones((len(verts), 4), dtype=np.float32)
            colors[:, 3] = np.asarray(ribbon_alpha(t), dtype=np.float32)
            # Bake the along-rim smudge as vertex COLOUR. It cannot live in the
            # texture: every ring of a column shares one frozen texel but wants
            # a different blur width, and one texel cannot hold a t-dependent
            # value. The skirt therefore becomes its own glTF primitive with an
            # untextured material, so COLOR_0 supplies the colour outright
            # rather than multiplying the plate and darkening it.
            smudge = float(((mesh.stats or {}).get("transition_ribbon") or {})
                           .get("smudge_px", 0.0) or 0.0)
            if smudge > 0.0 and texture is not None:
                colors[:, :3] = _ribbon_smudged_colors(mesh, texture, smudge)
                ribbon_face_mask = (t > 0.0)[faces].any(axis=1)

    # Group the ribbon's triangles at the END of the index buffer so the two
    # primitives are contiguous ranges of one accessor rather than two buffers.
    n_surface_faces = len(faces)
    if ribbon_face_mask is not None and ribbon_face_mask.any():
        order = np.concatenate([np.nonzero(~ribbon_face_mask)[0],
                                np.nonzero(ribbon_face_mask)[0]])
        faces = faces[order]
        n_surface_faces = int((~ribbon_face_mask).sum())
    else:
        ribbon_face_mask = None

    png_bytes = b""
    if texture is not None:
        buf = io.BytesIO()
        texture.save(buf, format="PNG")
        png_bytes = buf.getvalue()

    def _pad4(data: bytes, pad: bytes = b"\x00") -> bytes:
        return data + pad * ((4 - len(data) % 4) % 4)

    # Binary buffer layout: positions | uvs | indices | (colors) | (image)
    parts = [_pad4(verts.tobytes()), _pad4(uvs.tobytes()), _pad4(faces.tobytes())]
    if colors is not None:
        parts.append(_pad4(colors.tobytes()))
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
    attributes = {"POSITION": 0, "TEXCOORD_0": 1}
    if colors is not None:
        buffer_views.append({"buffer": 0, "byteOffset": offsets[3],
                             "byteLength": colors.nbytes, "target": 34962})
        accessors.append({"bufferView": 3, "componentType": 5126,
                          "count": int(len(colors)), "type": "VEC4"})
        attributes["COLOR_0"] = 3
    image_view = 4 if colors is not None else 3

    material: dict[str, Any] = {
        "name": "atlas_relief_projection",
        "doubleSided": True,
        "extensions": {"KHR_materials_unlit": {}},
        "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0},
    }
    if colors is not None:
        # Without BLEND the alpha channel is ignored and the skirt reads as an
        # opaque lip — the exact defect this whole feature exists to remove.
        material["alphaMode"] = "BLEND"

    materials = [material]
    primitives = [{"attributes": attributes, "indices": 2, "material": 0}]
    if ribbon_face_mask is not None:
        # The skirt gets its own primitive and an UNTEXTURED material. COLOR_0
        # multiplies the base colour in glTF, so leaving it on the textured
        # material would multiply the baked smudge by the plate and darken it;
        # with no baseColorTexture the vertex colour IS the colour, which is
        # what makes the bake match the viewport instead of merely tinting it.
        primitives[0]["indices"] = len(accessors)
        accessors.append({
            "bufferView": 2, "componentType": 5125,
            "count": int(n_surface_faces * 3), "type": "SCALAR"})
        primitives.append({
            "attributes": attributes, "indices": len(accessors), "material": 1})
        accessors.append({
            "bufferView": 2, "byteOffset": int(n_surface_faces * 3 * 4),
            "componentType": 5125,
            "count": int(faces.size - n_surface_faces * 3), "type": "SCALAR"})
        materials.append({
            "name": "atlas_relief_transition_ribbon",
            "doubleSided": True,
            "alphaMode": "BLEND",
            "extensions": {"KHR_materials_unlit": {}},
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0, "roughnessFactor": 1.0},
        })

    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "AtlasCamera relief mesh"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": primitives}],
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_chunk)}],
    }
    if png_bytes:
        buffer_views.append({"buffer": 0, "byteOffset": offsets[image_view - 1],
                             "byteLength": len(png_bytes)})
        gltf["images"] = [{"bufferView": image_view, "mimeType": "image/png"}]
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
