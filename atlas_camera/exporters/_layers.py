"""Shared per-ProjectionSource layer collection for the all-in-one DCC
exporters (`nuke_exporter.write_nuke_layers_script`,
`maya_exporter.write_maya_layers_scene`).

One function walks a solve's projection sources — sky dome, clean-plate
bands, multi-angle patches — and materializes each as on-disk assets
(plate PNG with the edge matte in its ALPHA when one is embedded, a
standalone matte PNG, and the layer mesh as OBJ+MTL) plus the camera it
projects from. Both DCC writers consume the identical list, so a layer that
exports to Nuke always exports to Maya the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import OrderedDict
import hashlib
import threading


_RETOPO_CACHE: OrderedDict[str, tuple[Any, Any, Any, dict[str, Any]]] = OrderedDict()
_RETOPO_CACHE_LOCK = threading.Lock()
_RETOPO_CACHE_MAX = 64


def _retopo_cache_key(mesh, camera, *, method, target_vertex_count,
                      smooth_iterations, crease_angle, pure_quad) -> str:
    """Content key shared by the Nuke/Maya exporter calls in one process."""
    import numpy as np

    intr = camera.intrinsics
    extr = camera.extrinsics
    h = hashlib.sha256()
    for arr in (mesh.vertices, mesh.faces, mesh.uvs,
                np.asarray(extr.camera_view_matrix, dtype=np.float64)):
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode("ascii"))
        h.update(str(a.shape).encode("ascii"))
        h.update(a.tobytes())
    h.update(repr((method, int(target_vertex_count), int(smooth_iterations),
                   float(crease_angle), bool(pure_quad),
                   intr.fx_px, intr.fy_px, intr.cx_px, intr.cy_px,
                   intr.image_width, intr.image_height)).encode("utf-8"))
    return h.hexdigest()


def _retopologize_layer_mesh(mesh, camera, *, method, target_vertex_count,
                             smooth_iterations, crease_angle, pure_quad):
    """Retopologize once per source mesh/config and reuse exact arrays.

    Instant Meshes' ``deterministic`` flag still showed small topology drift
    across two independent calls in a live Nuke/Maya export. A content cache
    makes the shared-collector promise literal: both DCC packages get the
    byte-identical retopology without writing into a shared output folder.
    """
    import copy

    key = _retopo_cache_key(
        mesh, camera, method=method, target_vertex_count=target_vertex_count,
        smooth_iterations=smooth_iterations, crease_angle=crease_angle,
        pure_quad=pure_quad,
    )
    with _RETOPO_CACHE_LOCK:
        cached = _RETOPO_CACHE.get(key)
        if cached is not None:
            _RETOPO_CACHE.move_to_end(key)
            vertices, faces, uvs, report = cached
            mesh.vertices = vertices.copy()
            mesh.faces = faces.copy()
            mesh.uvs = uvs.copy()
            return copy.deepcopy(report)

    from atlas_camera.core.mesh_retopo import apply_retopo

    intr = camera.intrinsics
    extr = camera.extrinsics
    width = int(intr.image_width or 0)
    height = int(intr.image_height or 0)
    fx = float(intr.fx_px or 0.0)
    fy = float(intr.fy_px or fx)
    report = apply_retopo(
        mesh, method=str(method), target_vertex_count=int(target_vertex_count),
        view_matrix=extr.camera_view_matrix, fx=fx, fy=fy,
        cx=(float(intr.cx_px) if intr.cx_px is not None else width / 2.0),
        cy=(float(intr.cy_px) if intr.cy_px is not None else height / 2.0),
        image_width=width, image_height=height, pure_quad=bool(pure_quad),
        crease_angle=float(crease_angle), smooth_iterations=int(smooth_iterations),
    )
    with _RETOPO_CACHE_LOCK:
        _RETOPO_CACHE[key] = (
            mesh.vertices.copy(), mesh.faces.copy(), mesh.uvs.copy(),
            copy.deepcopy(report),
        )
        _RETOPO_CACHE.move_to_end(key)
        while len(_RETOPO_CACHE) > _RETOPO_CACHE_MAX:
            _RETOPO_CACHE.popitem(last=False)
    return report


def layer_focal_mm(intr) -> float:
    """Focal length for a layer camera, with the standard pinhole fallback.

    Patch/outpainted cameras are CONSTRUCTED, so their intrinsics may carry
    fx_px without an explicit focal_length_mm — recover it from the pinhole
    relation instead of silently defaulting (an outpainted sky camera's
    wider canvas then yields the correct wider-FOV projector).
    """
    if intr.focal_length_mm:
        return float(intr.focal_length_mm)
    fx = intr.fx_px or 0.0
    if fx > 0 and intr.image_width:
        sensor_w = intr.sensor_width_mm or 36.0
        return float(fx * sensor_w / intr.image_width)
    return 35.0


def decode_plate_b64(image_b64: str):
    """data:image/...;base64 payload -> PIL Image (None when empty/undecodable)."""
    if not image_b64:
        return None
    try:
        import base64
        import io

        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Layer export needs Pillow to write plate previews. "
            "Install with: pip install -e .[image]"
        ) from exc
    try:
        payload = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    except Exception:
        return None


def mesh_from_primitive(prim):
    """Rebuild a ReliefMesh from a mesh proxy-primitive's flattened metadata
    (relief_mesh_primitive's exact inverse) so export_relief_mesh can write it."""
    import numpy as np

    from atlas_camera.core.relief_mesh import ReliefMesh

    meta = prim.metadata or {}
    verts = np.asarray(meta.get("vertices") or [], dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(meta.get("faces") or [], dtype=np.int32).reshape(-1, 3)
    uvs = np.asarray(meta.get("uvs") or [], dtype=np.float32).reshape(-1, 2)
    if not len(verts) or not len(faces):
        return None
    return ReliefMesh(vertices=verts, faces=faces, uvs=uvs)


def collect_projection_layers(
    solve,
    output_dir: str | Path,
    *,
    retopo_method: str = "off",
    retopo_target_vertex_count: int = 2000,
    retopo_smooth_iterations: int = 0,
    retopo_crease_angle: float = 30.0,
    retopo_pure_quad: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Materialize every exportable ProjectionSource into ``output_dir``.

    Returns ``(layers, skipped)`` where each layer dict carries:
    ``name`` (filesystem-safe), ``camera`` (the source's own LatentCamera —
    clean plates share the primary pose, patches orbit, outpainted skies
    widen), ``plate_path`` / ``colorspace`` (registered non-proxy plate_ref
    when present, else the browser preview authored as a PNG — with the edge
    matte embedded in its ALPHA), ``obj_path`` (mesh written via
    export_relief_mesh), and ``has_matte``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from atlas_camera.exporters.relief_mesh_exporter import export_relief_mesh

    layers: list[dict[str, Any]] = []
    skipped: list[str] = []
    for src in getattr(solve, "projection_sources", None) or []:
        mesh_prim = next(
            (p for p in (src.proxy_geometry or []) if p.primitive_type == "mesh"), None)
        mesh = mesh_from_primitive(mesh_prim) if mesh_prim is not None else None
        if mesh is None:
            skipped.append(f"{src.name}: no mesh geometry")
            continue

        retopo_report = {"method": "off", "changed": False,
                         "note": "retopology off"}
        if retopo_method and retopo_method != "off":
            retopo_report = _retopologize_layer_mesh(
                mesh, src.camera, method=str(retopo_method),
                target_vertex_count=int(retopo_target_vertex_count),
                smooth_iterations=int(retopo_smooth_iterations),
                crease_angle=float(retopo_crease_angle),
                pure_quad=bool(retopo_pure_quad),
            )

        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (src.name or "layer"))
        plate_ref = getattr(src, "plate_ref", None)
        matte = None
        mask_b64 = getattr(src, "mask_b64", None) or ""
        if mask_b64:
            matte_pil = decode_plate_b64(mask_b64)
            if matte_pil is not None:
                matte = matte_pil.convert("L")
                matte.save(out / f"{safe}_matte.png")
        # Invented-pixels mask (edge-extend smears / frame-outpaint ring):
        # written as its own file so compositors can regrain/process the
        # extension separately from photographed content.
        extend_matte_path = None
        extend_b64 = getattr(src, "extend_mask_b64", None) or ""
        if extend_b64:
            ext_pil = decode_plate_b64(extend_b64)
            if ext_pil is not None:
                ext_file = out / f"{safe}_extend_matte.png"
                ext_pil.convert("L").save(ext_file)
                extend_matte_path = str(ext_file.resolve()).replace("\\", "/")
        plate_path = None
        colorspace = None
        if plate_ref is not None and getattr(plate_ref, "plate_path", None) \
                and not getattr(plate_ref, "is_proxy", True):
            plate_path = str(plate_ref.plate_path)
            colorspace = getattr(plate_ref, "colorspace", None)
        else:
            pil = decode_plate_b64(src.image_b64 or "")
            if pil is None:
                skipped.append(f"{src.name}: no plate image")
                continue
            if matte is not None:
                if matte.size != pil.size:
                    matte = matte.resize(pil.size)
                pil = pil.convert("RGBA")
                pil.putalpha(matte)
            plate_file = out / f"{safe}_plate.png"
            pil.save(plate_file)
            plate_path = str(plate_file.resolve())
            # IMAGE tensors embedded by Atlas clean-plate/inpaint nodes are
            # display-referred previews (the neural/image graph works in the
            # same sRGB display space ComfyUI shows). Tag the authored PNG
            # explicitly so Nuke/Maya do not guess Raw or scene-linear ACEScg.
            colorspace = "sRGB - Display"

        written = export_relief_mesh(mesh, out, name=f"{safe}_mesh")
        layers.append({
            "name": safe,
            "camera": src.camera,
            "plate_path": plate_path.replace("\\", "/"),
            "colorspace": colorspace,
            "obj_path": str(Path(written["obj"]).resolve()).replace("\\", "/"),
            "has_matte": matte is not None,
            "extend_matte_path": extend_matte_path,
            "retopo": retopo_report,
        })
    return layers, skipped
