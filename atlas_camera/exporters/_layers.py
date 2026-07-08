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


def collect_projection_layers(solve, output_dir: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
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

        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (src.name or "layer"))
        plate_ref = getattr(src, "plate_ref", None)
        matte = None
        mask_b64 = getattr(src, "mask_b64", None) or ""
        if mask_b64:
            matte_pil = decode_plate_b64(mask_b64)
            if matte_pil is not None:
                matte = matte_pil.convert("L")
                matte.save(out / f"{safe}_matte.png")
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

        written = export_relief_mesh(mesh, out, name=f"{safe}_mesh")
        layers.append({
            "name": safe,
            "camera": src.camera,
            "plate_path": plate_path.replace("\\", "/"),
            "colorspace": colorspace,
            "obj_path": str(Path(written["obj"]).resolve()).replace("\\", "/"),
            "has_matte": matte is not None,
        })
    return layers, skipped
