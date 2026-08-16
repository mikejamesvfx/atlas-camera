"""Measured-primitives bridge: seed Blender from a metric solve, bring meshes back.

Host-agnostic (numpy + core only; nothing here imports comfy). The ComfyUI
nodes `AtlasBlenderMassing` / `AtlasBlenderImportMeshes` are thin wrappers.

Two directions, one contract:

  seed_from_solve()      -> what `exchange.write_scene_seed` needs: the recovered
                            camera, every proxy primitive tessellated as
                            REFERENCE, the measured quantities the solve knows.
  meshes_to_primitives() -> imported meshes become PROXY_ROLE `mesh` primitives
                            with PROJECTIVE UVs regenerated for the recovered
                            camera (`core.mesh_retopo.regenerate_projective_uvs`),
                            so nothing downstream sees an unmapped mesh. They
                            APPEND — an imported measured primitive is an
                            addition, like a viewport-drawn plane, never a
                            re-derivation of the PROXY_ROLE set. `AtlasMergeGeometry`
                            stays the one combiner.

Gates on import (reject-and-report per mesh, never raise): finite, indexed,
non-empty, not below ground by more than `min_y_m`, bbox within `max_radius_m`
of the camera. A stale seed (built against a different solve) is refused by
fingerprint unless the caller opts out.
"""
from __future__ import annotations

import hashlib
from typing import Any

from atlas_camera.core.proxy_geometry import PROXY_ROLE
from atlas_camera.core.schema import AtlasProxyPrimitive

IMPORT_SOURCE = "blender_import"
MASSING_SOURCE = "blender_massing"


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "atlas_camera.blender requires numpy. Install with: "
            "pip install -e .[vision]") from exc
    return np


def camera_params(solve: Any) -> dict[str, Any] | None:
    """The recovered camera as the flat dict the seed writer and UV regen use."""
    cam = getattr(solve, "camera", None)
    intr = getattr(cam, "intrinsics", None)
    extr = getattr(cam, "extrinsics", None)
    view = getattr(extr, "camera_view_matrix", None)
    width = int(getattr(intr, "image_width", 0) or 0)
    height = int(getattr(intr, "image_height", 0) or 0)
    fx = float(getattr(intr, "fx_px", 0.0) or 0.0)
    fy = float(getattr(intr, "fy_px", 0.0) or fx)
    if view is None or fx <= 0 or fy <= 0 or width <= 1 or height <= 1:
        return None
    cx = getattr(intr, "cx_px", None)
    cy = getattr(intr, "cy_px", None)
    return {
        "view_matrix": view, "fx": fx, "fy": fy,
        "cx": float(cx) if cx is not None else width / 2.0,
        "cy": float(cy) if cy is not None else height / 2.0,
        "image_width": width, "image_height": height,
    }


def solve_seed_fingerprint(solve: Any) -> str:
    """Identity of the SOLVE a seed was built from — camera + primitive roster.

    Distinct from `comfy.fingerprints._solve_fingerprint`, which needs the image
    tensor; the Blender round-trip may span sessions with no tensor in hand.
    """
    h = hashlib.md5()
    cp = camera_params(solve) or {}
    h.update(repr(cp.get("view_matrix")).encode())
    h.update(repr((cp.get("fx"), cp.get("fy"), cp.get("cx"), cp.get("cy"),
                   cp.get("image_width"), cp.get("image_height"))).encode())
    scene = getattr(solve, "projection_scene", None)
    for prim in (getattr(scene, "proxy_geometry", None) or []):
        meta = getattr(prim, "metadata", None) or {}
        # Meshes that came BACK from Blender are excluded on purpose: the seed
        # was built before they existed, and the import node is normally fed
        # the massing node's output — including them would make every honest
        # round-trip read as stale (found live 2026-08-16).
        if meta.get("source") in (IMPORT_SOURCE, MASSING_SOURCE):
            continue
        h.update(repr((getattr(prim, "name", ""), getattr(prim, "primitive_type", ""),
                       meta.get("source"), meta.get("n_vertices"))).encode())
    return h.hexdigest()[:16]


def measured_quantities(solve: Any) -> dict[str, Any]:
    """Whatever metric anchors the solve carries, for the seed's `measured` block."""
    dm = dict(getattr(solve, "debug_metadata", None) or {})
    out: dict[str, Any] = {}
    cp = camera_params(solve)
    if cp:
        out["fx_px"] = cp["fx"]
        out["fy_px"] = cp["fy"]
        try:
            import numpy as np
            c2w = np.linalg.inv(np.asarray(cp["view_matrix"], dtype=np.float64))
            out["camera_height_m"] = float(c2w[1, 3])
        except Exception:  # noqa: BLE001
            pass
    for key in ("scale_source", "baseline_m", "focal_source", "reference_scale"):
        if key in dm and isinstance(dm[key], (str, int, float, bool)):
            out[key] = dm[key]
    rs = dm.get("reference_scale")
    if isinstance(rs, dict):
        for k in ("source", "confidence", "scale_factor"):
            if k in rs and isinstance(rs[k], (str, int, float, bool)):
                out[f"reference_scale_{k}"] = rs[k]
    return out


#: Sources that are ARTIST/MEASURED intent and always ride the seed; the
#: heavy projection surfaces (relief mesh, derived proxies) only when asked.
_LIGHT_SOURCES = ("viewport_polygon", "block_massing", MASSING_SOURCE, IMPORT_SOURCE)


def seed_from_solve(solve: Any, *, include_sources: bool = True,
                    depth_result: Any = None, exclude_mask: Any = None,
                    include_relief: bool | None = None,
                    max_points: int = 200_000, max_planes: int = 8,
                    seed: int = 0) -> dict[str, Any]:
    """Build the `write_scene_seed` inputs from a solve.

    Two modes, decided by what is available:

    * MEASURED (preferred): ``depth_result`` carries a MoGe pointmap → the seed
      is a sky-free metric point cloud + measured ground/camera height/extents/
      dominant planes (`measure_scene_from_pointmap`), plus the artist's drawn
      polygons and massing boxes. The relief mesh and derived proxies are LEFT
      OUT unless ``include_relief`` — a projection surface at the solve's
      assumed scale is not a measurement, and it is 6+ MB Blender never needs.
    * FALLBACK: no pointmap → every proxy primitive tessellated as reference
      (the original behaviour; ``include_relief`` defaults True here).

    Returns ``{"camera", "primitives", "drawn_shapes", "cloud", "measured",
    "ground_y", "fingerprint"}`` or raises ValueError when the solve has no
    usable camera.
    """
    from atlas_camera.core.primitive_mesh import tessellate_primitive
    cp = camera_params(solve)
    if cp is None:
        raise ValueError("solve has no usable camera intrinsics/extrinsics — "
                         "the seed needs fx/fy and a 4x4 view matrix")
    scene_meas = None
    if depth_result is not None:
        scene_meas = measure_scene_from_pointmap(
            solve, depth_result, exclude_mask=exclude_mask,
            max_points=int(max_points), max_planes=int(max_planes), seed=int(seed))
    measured_mode = scene_meas is not None
    if include_relief is None:
        include_relief = not measured_mode

    prims_out: list[dict[str, Any]] = []
    scene = getattr(solve, "projection_scene", None)
    roster = [(p, "primary") for p in (getattr(scene, "proxy_geometry", None) or [])]
    if include_sources:
        for src in (getattr(solve, "projection_sources", None) or []):
            for p in (getattr(src, "proxy_geometry", None) or []):
                roster.append((p, getattr(src, "name", "") or "layer"))
    for prim, layer in roster:
        meta = getattr(prim, "metadata", None) or {}
        if not include_relief and meta.get("source") not in _LIGHT_SOURCES:
            continue
        tess = tessellate_primitive(prim)
        if tess is None:
            continue
        v, f = tess
        if not len(v) or not len(f):
            continue
        entry: dict[str, Any] = {
            "name": getattr(prim, "name", "") or "",
            "primitive_type": getattr(prim, "primitive_type", ""),
            "source": meta.get("source"),
            "role": meta.get("role"),
            "layer": layer,
            "vertices": v, "faces": f,
        }
        for k in ("height_m", "label", "polygon_id", "provenance", "trust"):
            if k in meta and isinstance(meta[k], (str, int, float, bool)):
                entry[k] = meta[k]
        prims_out.append(entry)
    measured = measured_quantities(solve)
    cloud = None
    ground_y = 0.0
    if scene_meas is not None:
        prims_out.extend(scene_meas["planes"])
        cloud = scene_meas["cloud"]
        ground_y = scene_meas["ground_y"] if scene_meas["ground_y"] is not None else 0.0
        measured.update(scene_meas["measured"])
        measured["seed_mode"] = "measured_pointmap"
    else:
        measured["seed_mode"] = "relief_reference"
        measured["scale_source"] = measured.get("scale_source", "solve")
    return {
        "camera": cp,
        "primitives": prims_out,
        "drawn_shapes": [],
        "cloud": cloud,
        "ground_y": float(ground_y),
        "measured": measured,
        "fingerprint": solve_seed_fingerprint(solve),
    }


def measure_scene_from_pointmap(
    solve: Any,
    depth_result: Any,
    *,
    exclude_mask: Any = None,
    max_points: int = 200_000,
    max_planes: int = 8,
    seed: int = 0,
) -> dict[str, Any] | None:
    """The MEASURED seed: MoGe's metric pointmap, sky excluded, scene measured.

    Why this and not the relief mesh: MoGe's ``points`` are the accurate metric
    measurement (its depth AND intrinsics derive from them), while the relief
    mesh is a projection surface at the solve's assumed scale — 272k faces and
    6.6 MB of NPZ that Blender does not need to model against. What Blender
    needs is small: a sky-free metric point cloud to snap to, the ground plane,
    the camera height above it, scene extents, and the dominant planes.

    World frame: the SOLVE camera pose (rotation + position) applied to MoGe's
    camera-frame points — so returned meshes need no translation to sit with
    the rest of the solve. Scale is MoGe's, so the measured ground lands at
    ``ground_y_m`` (= camera_y − MoGe camera height), which is reported and
    used by the recipe instead of assuming Y=0. Returns None when the depth
    carries no pointmap.
    """
    np = _require_numpy()
    from atlas_camera.core.depth_geometry import (
        detect_sky_mask, opencv_points_to_world,
    )
    from atlas_camera.core.plane_extraction import extract_planes_ransac
    from atlas_camera.core.primitive_mesh import tessellate_primitive
    from atlas_camera.core.solver import estimate_ground_height_from_depth

    pts = getattr(depth_result, "points", None)
    cp = camera_params(solve)
    if pts is None or cp is None:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    H, W = pts.shape[:2]
    depth = pts[..., 2].copy()
    valid = np.isfinite(depth) & (depth > 1e-4)

    # Intrinsics of the POINTMAP frame (it is at the depth image's size; the
    # solve intrinsics are rescaled if the two differ).
    sx = W / float(cp["image_width"]); sy = H / float(cp["image_height"])
    fx, fy, cx, cy = cp["fx"] * sx, cp["fy"] * sy, cp["cx"] * sx, cp["cy"] * sy
    view = np.asarray(cp["view_matrix"], dtype=np.float64)

    # Sky: the explicit mask REPLACES the heuristic (design rule); either way
    # the excluded pixels leave the measurement entirely.
    from atlas_camera.core.camera_math import horizon_row_from_extrinsics
    try:
        horizon_y = horizon_row_from_extrinsics(solve.camera.extrinsics, fy=fy, cy=cy)
    except Exception:  # noqa: BLE001
        horizon_y = H * 0.45
    if exclude_mask is not None:
        ex = np.asarray(exclude_mask, dtype=bool)
        if ex.shape != (H, W):
            yi = (np.arange(H) * (ex.shape[0] / H)).astype(int).clip(0, ex.shape[0] - 1)
            xi = (np.arange(W) * (ex.shape[1] / W)).astype(int).clip(0, ex.shape[1] - 1)
            ex = ex[yi][:, xi]
        sky = ex
        sky_source = "exclude_mask"
    else:
        try:
            sky = np.asarray(detect_sky_mask(np.where(valid, depth, np.nan),
                                             horizon_y=float(horizon_y)), dtype=bool)
        except Exception:  # noqa: BLE001
            sky = np.zeros((H, W), dtype=bool)
        sky_source = "detect_sky_mask"
    keep = valid & ~sky
    depth_m = np.where(keep, depth, np.nan)

    # Ground + camera height, MoGe scale.
    R_wc = view[:3, :3]
    try:
        g = estimate_ground_height_from_depth(depth_m, rotation=R_wc, fx=fx, fy=fy,
                                              cx=cx, cy=cy, horizon_y=float(horizon_y))
    except Exception as exc:  # noqa: BLE001
        g = {"camera_height": None, "confidence": 0.0, "reason": str(exc)}
    cam_h = g.get("camera_height")
    c2w = np.linalg.inv(view)
    cam_pos = c2w[:3, 3]
    ground_y = float(cam_pos[1] - cam_h) if cam_h else None

    # World points (solve pose, MoGe scale), sky-free, subsampled.
    world = opencv_points_to_world(pts, view_matrix=view)
    idx = np.flatnonzero(keep.ravel())
    if len(idx) > int(max_points):
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(idx, size=int(max_points), replace=False))
    cloud = world.reshape(-1, 3)[idx]
    bbox_min = cloud.min(axis=0).tolist() if len(cloud) else [0, 0, 0]
    bbox_max = cloud.max(axis=0).tolist() if len(cloud) else [0, 0, 0]

    # Dominant planes at MoGe scale. extract_planes_ransac pins the ground to
    # the VIEW's camera height, so hand it a view whose camera sits at the
    # measured height above the measured ground: scale ≈ 1, geometry stays
    # metric, and the planes come back in our world once shifted by ground_y.
    planes: list[dict[str, Any]] = []
    plane_stats: dict[str, Any] = {}
    if cam_h and cam_h > 0:
        try:
            c2w_m = c2w.copy(); c2w_m[1, 3] = float(cam_h)
            view_m = np.linalg.inv(c2w_m)
            prims, plane_stats = extract_planes_ransac(
                depth_m, view_matrix=view_m, fx=fx, fy=fy, cx=cx, cy=cy,
                max_planes=int(max_planes), horizon_y=float(horizon_y))
            for prim in prims:
                tess = tessellate_primitive(prim)
                if tess is None:
                    continue
                v, f = tess
                v = np.asarray(v, dtype=np.float64) + np.array([0.0, ground_y, 0.0])
                meta = getattr(prim, "metadata", None) or {}
                planes.append({
                    "name": getattr(prim, "name", "plane"),
                    "primitive_type": getattr(prim, "primitive_type", "plane"),
                    "source": "measured_plane", "role": meta.get("role"),
                    "kind": str(meta.get("kind") or meta.get("plane_kind") or ""),
                    "vertices": v, "faces": np.asarray(f, dtype=np.int64),
                })
        except Exception as exc:  # noqa: BLE001
            plane_stats = {"error": str(exc)}

    measured = {
        "scale_source": "moge_pointmap",
        "camera_height_m": float(cam_h) if cam_h else None,
        "ground_y_m": ground_y,
        "ground_confidence": float(g.get("confidence") or 0.0),
        "ground_reason": g.get("reason", ""),
        "sky_source": sky_source,
        "sky_fraction": float(sky.mean()),
        "valid_fraction": float(valid.mean()),
        # Everything that left the measurement: sky + the model's own invalid
        # (MoGe marks sky/undefined pixels invalid, so on real plates this is
        # the number to read).
        "excluded_fraction": float(1.0 - keep.mean()),
        "n_cloud_points": int(len(cloud)),
        "n_cloud_candidates": int(keep.sum()),
        "bbox_min": bbox_min, "bbox_max": bbox_max,
        "extent_m": [float(b - a) for a, b in zip(bbox_min, bbox_max)],
        "median_depth_m": float(np.nanmedian(depth_m)) if keep.any() else None,
        "n_measured_planes": len(planes),
        "planes_scale_applied": float(plane_stats.get("ground_scale", 1.0) or 1.0)
        if isinstance(plane_stats, dict) else 1.0,
        "depth_model": getattr(depth_result, "model_id", ""),
        "predicted_focal_px": (getattr(depth_result, "metadata", {}) or {}).get("predicted_focal_px"),
    }
    return {"cloud": cloud, "planes": planes, "measured": measured,
            "camera": cp, "ground_y": ground_y}


def meshes_to_primitives(
    solve: Any,
    meshes: list[dict[str, Any]],
    *,
    source: str = IMPORT_SOURCE,
    name_prefix: str = "blender",
    min_y_m: float = -0.05,
    max_radius_m: float = 0.0,
    extra_tags: dict[str, Any] | None = None,
    paint_with: str = "source_photo",
) -> tuple[list[AtlasProxyPrimitive], list[dict[str, Any]]]:
    """Turn imported meshes into PROXY_ROLE primitives with projective UVs.

    ``paint_with`` decides which projector paints the mesh in the viewport:
    ``source_photo`` (the primary plate — right for facades the photo shows)
    or ``clean_plate`` (the viewport's `clean_plate` input — right for OCCLUDED
    surfaces: the water/hill behind a foreground object). A per-mesh Blender
    custom property ``atlas_paint`` overrides it (exported as tag ``paint``).

    Returns ``(accepted_primitives, rejected)`` where each rejected entry is
    ``{"name", "reason"}``. Nothing is appended here — the caller appends, so
    the same helper serves both nodes and stays testable without a solve copy.
    """
    np = _require_numpy()
    from atlas_camera.core.mesh_retopo import regenerate_projective_uvs
    cp = camera_params(solve)
    if cp is None:
        raise ValueError("solve has no usable camera; cannot regenerate projective UVs")
    c2w = np.linalg.inv(np.asarray(cp["view_matrix"], dtype=np.float64))
    cam_pos = c2w[:3, 3]

    accepted: list[AtlasProxyPrimitive] = []
    rejected: list[dict[str, Any]] = []
    for i, m in enumerate(meshes):
        name = str(m.get("name") or f"{name_prefix}_{i:02d}")
        v = np.asarray(m["vertices"], dtype=np.float64).reshape(-1, 3)
        f = np.asarray(m["faces"], dtype=np.int64).reshape(-1, 3)
        if not len(v) or not len(f):
            rejected.append({"name": name, "reason": "empty mesh"})
            continue
        if not np.isfinite(v).all():
            rejected.append({"name": name, "reason": "non-finite vertices"})
            continue
        if f.min() < 0 or f.max() >= len(v):
            rejected.append({"name": name, "reason": "face index out of range"})
            continue
        ymin = float(v[:, 1].min())
        if ymin < float(min_y_m):
            rejected.append({"name": name,
                             "reason": f"min Y {ymin:.3f} m is below ground "
                                       f"(limit {float(min_y_m):.3f} m)"})
            continue
        if max_radius_m and max_radius_m > 0:
            r = float(np.linalg.norm(v - cam_pos[None, :], axis=1).max())
            if r > float(max_radius_m):
                rejected.append({"name": name,
                                 "reason": f"extends {r:.1f} m from the camera "
                                           f"(limit {float(max_radius_m):.1f} m)"})
                continue
        uv = regenerate_projective_uvs(
            v, view_matrix=cp["view_matrix"], fx=cp["fx"], fy=cp["fy"],
            cx=cp["cx"], cy=cp["cy"], image_width=cp["image_width"],
            image_height=cp["image_height"])
        tags = {k: val for k, val in m.items()
                if k not in ("vertices", "faces", "name")
                and isinstance(val, (str, int, float, bool))}
        mesh_paint = str(m.get("paint") or paint_with or "source_photo")
        if mesh_paint not in ("source_photo", "clean_plate"):
            mesh_paint = "source_photo"
        meta: dict[str, Any] = {
            "role": PROXY_ROLE,
            "source": source,
            "paint_with": mesh_paint,
            "n_vertices": int(len(v)),
            "n_faces": int(len(f)),
            "vertices": np.round(v.reshape(-1), 3).tolist(),
            "faces": f.reshape(-1).astype(np.int64).tolist(),
            "uvs": np.round(np.asarray(uv, dtype=np.float64).reshape(-1), 4).tolist(),
            "edge_risk": [],
            "ribbon_t": [],
            "uv_source": "projective_regenerated_on_import",
            **{f"blender_{k}": val for k, val in tags.items()},
            **{k: val for k, val in (extra_tags or {}).items()
               if isinstance(val, (str, int, float, bool))},
        }
        accepted.append(AtlasProxyPrimitive(
            name=f"{name_prefix}_{name}" if not name.startswith(name_prefix) else name,
            primitive_type="mesh",
            dimensions=(0.0, 0.0, 0.0),
            material="atlas_projection_proxy",
            metadata=meta,
        ))
    return accepted, rejected
