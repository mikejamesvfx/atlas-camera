"""Derived trust/health evaluation for Atlas solves — pure Python, zero deps.

The single home for "is this scene trustworthy?" logic (per the 2026-07-17
engineering-recommendations response): scale provenance -> safe-to-export
status here in M1; the full scene-health engine (the AtlasDebugReport red-flag
checks, consumed by both that node and AtlasSceneHealthGate) lands here in a
later milestone. Everything is DERIVED — this module never changes solver
behavior, only reads provenance the solvers already record.

Callers on the serialization hot path (LatentScene.to_dict) rely on
scale_health() being exception-proof: any surprise degrades to the
"unknown / not safe" answer, never a raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

SCALE_STATUS_MEASURED = "measured"
SCALE_STATUS_MANUAL = "manual"
SCALE_STATUS_ASSUMED = "assumed"
SCALE_STATUS_UNKNOWN = "unknown"

# Fixed nominal confidence for the assumed-eye-height fallback: it is a guess,
# not a measurement — kept > 0 only so downstream sorting/plotting behaves.
_ASSUMED_CONFIDENCE = 0.15


@dataclass(slots=True)
class ScaleHealth:
    status: str                     # measured | manual | assumed | unknown
    scale_source: str | None
    confidence: float | None
    camera_height_m: float | None
    safe_to_export: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "scale_source": self.scale_source,
            "confidence": self.confidence,
            "camera_height_m": self.camera_height_m,
            "safe_to_export": self.safe_to_export,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ScaleHealth | None":
        if not isinstance(data, dict):
            return None
        return cls(
            status=str(data.get("status", SCALE_STATUS_UNKNOWN)),
            scale_source=data.get("scale_source"),
            confidence=(float(data["confidence"])
                        if data.get("confidence") is not None else None),
            camera_height_m=(float(data["camera_height_m"])
                             if data.get("camera_height_m") is not None else None),
            safe_to_export=bool(data.get("safe_to_export", False)),
            detail=str(data.get("detail", "")),
        )


def _camera_height(solve: Any) -> float | None:
    try:
        pos = solve.camera.extrinsics.camera_position
        return float(pos[1])
    except Exception:  # noqa: BLE001 — hand-built/partial solves
        return None


def _depth_is_metric(solve: Any) -> bool | None:
    """True/False when the depth component records it; None when unknown."""
    try:
        depth = getattr(solve, "depth", None)
        value = getattr(depth, "value", None)
        if isinstance(value, dict) and "is_metric" in value:
            return bool(value["is_metric"])
    except Exception:  # noqa: BLE001
        pass
    return None


def scale_health(solve: Any) -> ScaleHealth:
    """Map the solve's recorded scale provenance to a trust status.

    Reads only what the solvers already stamp (``debug_metadata`` +
    ``solve.depth``); patch-view registration provenance (``reuse_scene`` /
    ``ground_fit`` / ``primary_registration`` on ProjectionSource metadata) is
    deliberately out of scope — layers inherit the scene's scale.
    """
    try:
        return _scale_health_inner(solve)
    except Exception:  # noqa: BLE001 — hot path (to_dict); never raise
        return ScaleHealth(
            status=SCALE_STATUS_UNKNOWN, scale_source=None, confidence=None,
            camera_height_m=None, safe_to_export=False,
            detail="Scale provenance could not be evaluated.")


def _scale_health_inner(solve: Any) -> ScaleHealth:
    meta = getattr(solve, "debug_metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    source = meta.get("scale_source")
    height = _camera_height(solve)

    if source == "reference_object":
        ref = meta.get("reference_scale") or {}
        conf = float(ref.get("confidence", 0.0)) if isinstance(ref, dict) else 0.0
        return ScaleHealth(
            SCALE_STATUS_MEASURED, source, conf, height, True,
            f"Metric scale measured from reference object(s), "
            f"consistency {conf:.2f}.")

    if source == "depth_ground_plane":
        depth = getattr(solve, "depth", None)
        conf = float(getattr(depth, "confidence", 0.0) or 0.0)
        is_metric = _depth_is_metric(solve)
        if is_metric is False:
            return ScaleHealth(
                SCALE_STATUS_MEASURED, source, conf, height, False,
                "Camera height measured from RELATIVE depth — up-to-scale "
                "only, not metric. Verify with a reference or manual height.")
        return ScaleHealth(
            SCALE_STATUS_MEASURED, source, conf, height, True,
            f"Camera height measured from the depth ground plane, "
            f"confidence {conf:.2f}.")

    if source == "manual_override":
        return ScaleHealth(
            SCALE_STATUS_MANUAL, source, 1.0, height, True,
            "Metric scale set manually (artist decision).")

    if source == "assumed_default":
        detail = "Assumed 1.6 m eye height — NOT measured."
        if height is not None:
            detail = (f"Assumed eye height ({height:g} m) — NOT measured. "
                      "Elevated/AI plates are typically far off; use "
                      "AtlasScaleOverride or a scale reference.")
        return ScaleHealth(
            SCALE_STATUS_ASSUMED, source, _ASSUMED_CONFIDENCE, height, False,
            detail)

    return ScaleHealth(
        SCALE_STATUS_UNKNOWN, source if isinstance(source, str) else None,
        None, height, False,
        "No metric-scale provenance recorded on this solve.")


# ---------------------------------------------------------------------------
# Scene-health engine (M4): THE single red-flag evaluator. AtlasDebugReport
# and AtlasSceneHealthGate both consume this — change a check here and both
# stay in lockstep (the parity test in tests/test_debug_report_parity.py pins
# the exact flag text). Logic moved VERBATIM from AtlasDebugReport.report();
# severity/codes are new metadata layered on top for the gate.
# ---------------------------------------------------------------------------

# fail = the scene is structurally broken for export; warn = reviewable.
_FLAG_SEVERITY = {
    "camera_below_ground": "fail",
    "zero_vertex_layer": "fail",
    "near_empty_matte": "warn",
    "band_gap": "warn",
    "band_overlap": "warn",
    "scope_fallback": "warn",
    "negative_depth": "warn",
    "scale_unverified": "warn",
    # Ported from the portable outlier/stretched-edge worklog (2026-07-18):
    # mesh quality metrics as quality-based fallback triggers (card / ground
    # / segmented inpaint instead of raising global relief thresholds).
    "torn_excessive": "warn",
    "stretch_excessive": "warn",
}


@dataclass(slots=True)
class HealthFlag:
    severity: str            # "warn" | "fail"
    code: str
    message: str
    layer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"severity": self.severity, "code": self.code,
                "message": self.message, "layer": self.layer}


@dataclass(slots=True)
class HealthReport:
    level: str                              # pass | warn | fail
    flags: list[HealthFlag] = field(default_factory=list)
    camera: dict[str, Any] = field(default_factory=dict)
    per_layer: list[dict[str, Any]] = field(default_factory=list)
    depth: dict[str, Any] | None = None
    scale: ScaleHealth | None = None

    @property
    def flag_messages(self) -> list[str]:
        return [f.message for f in self.flags]

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "flags": [f.to_dict() for f in self.flags],
            "camera": self.camera,
            "per_layer": self.per_layer,
            "depth": self.depth,
            "scale": self.scale.to_dict() if self.scale else None,
        }


def _flag(code: str, message: str, layer: str | None = None) -> HealthFlag:
    return HealthFlag(_FLAG_SEVERITY.get(code, "warn"), code, message, layer)


def evaluate_scene_health(
    solve: Any,
    depth: Any = None,
    *,
    scope_statuses: dict[str, str] | None = None,
    matte_coverage_fn: Callable[[Any], float | None] | None = None,
    include_scale_flag: bool = True,
) -> HealthReport:
    """Evaluate the layered scene's red flags (the AtlasDebugReport checks).

    ``depth`` is a live DepthResult-like object (model_id/is_metric/near/far/
    image_width/image_height/metadata[/depth]); ``matte_coverage_fn`` decodes
    a mask_b64 to a coverage fraction (PIL lives in the caller, never here).
    """
    flags: list[HealthFlag] = []
    cam = solve.camera
    intr, extr = cam.intrinsics, cam.extrinsics

    # Camera height from the full 4x4 (the view-matrix convention rule) —
    # extrinsics.camera_position can legitimately be an unset default.
    cam_y = None
    if extr is not None and extr.camera_view_matrix is not None:
        try:
            import numpy as np
            cam_y = round(float(np.linalg.inv(np.asarray(
                extr.camera_view_matrix, dtype=float))[1, 3]), 4)
        except Exception:  # noqa: BLE001
            cam_y = None
    camera = {
        "image_wh": [intr.image_width, intr.image_height],
        "focal_mm": intr.focal_length_mm, "sensor_mm": intr.sensor_width_mm,
        "fx_px": intr.fx_px, "camera_height_m": cam_y,
        "confidence": getattr(solve, "confidence", None),
        "confidence_detail": dict(getattr(cam.confidence, "individual_metrics", {}) or {}),
        "source_method": getattr(solve, "source_method", None),
        "scale_source": (getattr(solve, "debug_metadata", None) or {}).get("scale_source"),
    }
    if cam_y is not None and cam_y <= 0:
        flags.append(_flag(
            "camera_below_ground",
            "camera height <= 0 — ground-based features (ground depth, "
            "band_geometry=ground) will fail"))

    sources: list[dict[str, Any]] = []
    for src in getattr(solve, "projection_sources", None) or []:
        meta = src.metadata or {}
        n_verts = sum(int((g.metadata or {}).get("n_vertices") or 0)
                      for g in (src.proxy_geometry or []))
        n_faces = sum(int((g.metadata or {}).get("n_faces") or 0)
                      for g in (src.proxy_geometry or []))
        cov = matte_coverage_fn(getattr(src, "mask_b64", None)) \
            if matte_coverage_fn else None
        # Mesh QA metrics ride the relief-mesh primitive's metadata (the
        # outlier/stretched-edge tier): surface them per layer and use them
        # as quality-based fallback triggers.
        mesh_meta = next((g.metadata or {} for g in (src.proxy_geometry or [])
                          if (g.metadata or {}).get("source") == "depth_relief_mesh"), {})
        entry = {
            "name": src.name, "priority": src.priority,
            "projection_mode": meta.get("projection_mode"),
            "band_geometry": meta.get("band_geometry"),
            "near_m": meta.get("near_m"), "far_m": meta.get("far_m"),
            "n_vertices": n_verts, "n_faces": n_faces,
            "torn_fraction": mesh_meta.get("torn_fraction"),
            "quad_coherence": mesh_meta.get("quad_coherence"),
            "stretch_ratio_p95": mesh_meta.get("stretch_ratio_p95"),
            "stretch_fraction_gt12": mesh_meta.get("stretch_fraction_gt12"),
            "n_filled_cells": meta.get("n_filled_cells"),
            "source_camera_wh": [src.camera.intrinsics.image_width,
                                 src.camera.intrinsics.image_height]
                                if src.camera else None,
            "matte_coverage": cov,
            "has_extend_mask": bool(getattr(src, "extend_mask_b64", None)),
            "scale_source": meta.get("scale_source"),
        }
        sources.append(entry)
        if n_verts == 0:
            flags.append(_flag(
                "zero_vertex_layer",
                f"{src.name}: ZERO vertices — this layer contributes no "
                "geometry (empty band, exclude-everything scope, or a "
                "failed flat-mode region)", src.name))
        elif cov is not None and cov < 0.005:
            flags.append(_flag(
                "near_empty_matte",
                f"{src.name}: matte covers only {cov:.2%} of the frame — "
                "layer will paint almost nothing", src.name))
        torn = entry.get("torn_fraction")
        # torn_fraction is GLOBAL (1 - faces/n_quads over the whole grid), so
        # a deliberately band-clipped layer always reads high — found by the
        # healthy-stack fixture flagging a correct narrow band at 73.8%. The
        # worklog's 65% threshold was calibrated on full-frame/mask-membership
        # meshes, so the check only applies where band clipping isn't the
        # dominant tear cause (no finite far edge).
        if (torn is not None and float(torn) > 0.65
                and entry.get("far_m") is None):
            flags.append(_flag(
                "torn_excessive",
                f"{src.name}: {float(torn):.1%} of relief quads torn — "
                "expect local coverage gaps; add/increase clean-plate matte",
                src.name))
        stretch = entry.get("stretch_ratio_p95")
        if stretch is not None and float(stretch) > 12.0:
            flags.append(_flag(
                "stretch_excessive",
                f"{src.name}: p95 world/UV edge ratio {float(stretch):.1f} — "
                "likely stretched texels; prefer card or segmented inpaint",
                src.name))

    # Band continuity (clean-plate band layers only, sorted by near edge).
    # NOTE: the membership expression's operator precedence is preserved
    # verbatim from the original AtlasDebugReport implementation.
    bands = sorted((s for s in sources
                    if s["projection_mode"] == "clean_plate" and s["near_m"] is not None
                    or (s["near_m"] is None and s["far_m"] is not None)),
                   key=lambda s: s["near_m"] or 0.0)
    for a, b in zip(bands, bands[1:]):
        fa, nb = a.get("far_m"), b.get("near_m")
        if fa is not None and nb is not None and abs(fa - nb) > max(0.05, 0.02 * fa):
            kind = "GAP" if nb > fa else "OVERLAP"
            flags.append(_flag(
                "band_gap" if nb > fa else "band_overlap",
                f"band {kind} between {a['name']} (far {fa:.2f}m) and "
                f"{b['name']} (near {nb:.2f}m)"))

    for k, s in (scope_statuses or {}).items():
        if "FALLBACK" in s:
            flags.append(_flag("scope_fallback", f"scope {k}: {s}"))

    # DA3 watch-item made measurable (see AtlasDebugReport's original note).
    depth_info = None
    if depth is not None:
        depth_info = {"model_id": depth.model_id, "is_metric": depth.is_metric,
                      "near": depth.near, "far": depth.far,
                      "wh": [depth.image_width, depth.image_height]}
        try:
            import numpy as np
            recorded = (depth.metadata or {}).get("negative_fraction")
            if recorded is not None:
                neg = float(recorded)
            else:
                arr = np.asarray(depth.depth)
                neg = float((arr < 0).mean())
            depth_info["negative_fraction"] = round(neg, 4)
            if neg > 0.01:
                flags.append(_flag(
                    "negative_depth",
                    f"depth: {neg:.1%} of raw depth is NEGATIVE (DA3 watch-item) — "
                    "ground-pinning renormalizes it, but suspect this first if a "
                    "band's geometry misbehaves on this shot"))
        except Exception:  # noqa: BLE001
            pass

    scale = scale_health(solve)
    if include_scale_flag and not scale.safe_to_export:
        flags.append(_flag(
            "scale_unverified",
            f"scale {scale.status.upper()} — not verified: {scale.detail}"))

    level = "pass"
    if any(f.severity == "fail" for f in flags):
        level = "fail"
    elif flags:
        level = "warn"
    return HealthReport(level=level, flags=flags, camera=camera,
                        per_layer=sources, depth=depth_info, scale=scale)
