"""atlas_project.json — the versioned reproducibility manifest.

One JSON per export directory bundling everything needed to trace an artifact
back to what produced it: plate checksum, solve fingerprint, model
provenance, seeds, scale/health verdicts, settings, and the artifact list.
Paths + hashes only — image data stays external, never base64.

Doctrine:
- ``load_project_manifest`` is the SINGLE read entrypoint (future schema
  migrations live there and nowhere else).
- ``manifest_identity_hash`` hashes the IDENTITY CORE only (fields frozen in
  ``_IDENTITY_FIELDS`` — changing that list is a schema bump by definition),
  so it is computable before artifacts are written and exporters can embed it
  as a comment without self-invalidation.
- A manifest failure must never fail an export (callers swallow OSError).
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "atlas_project.json"

# The identity core: the manifest keys hashed into manifest_identity_hash.
# FROZEN for schema v1 — extending or reordering this list is a schema bump.
_IDENTITY_FIELDS = ("schema", "plate", "solve_fingerprint", "models", "scale")


@dataclass(slots=True)
class ManifestArtifact:
    kind: str        # "solve_json" | "nuke_scene" | "maya_scene" | "usd_camera" | ...
    path: str
    exporter: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": self.path, "exporter": self.exporter}


def file_md5(path: str | Path | None) -> str | None:
    """Streaming md5 of a file; None when the path is missing/unreadable."""
    if not path:
        return None
    try:
        digest = hashlib.md5()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _plate_info(solve: Any) -> dict[str, Any]:
    from atlas_camera.exporters._plate import primary_plate_colorspace, primary_plate_path

    try:
        path = primary_plate_path(solve)
    except Exception:  # noqa: BLE001
        path = getattr(solve, "image_path", None)
    plate_ref = getattr(solve, "source_plate", None)
    return {
        "path": str(path) if path else None,
        "md5": file_md5(path),
        "colorspace": (primary_plate_colorspace(solve)
                       if path else getattr(plate_ref, "colorspace", None)),
        "bit_depth": getattr(plate_ref, "bit_depth", None),
    }


def solve_content_fingerprint(solve: Any) -> str:
    """md5[:16] of camera identity + plate FILE hash (no IMAGE tensor needed —
    the file-based sibling of nodes.py's ``_solve_fingerprint``)."""
    intr = solve.camera.intrinsics
    extr = solve.camera.extrinsics
    digest = hashlib.md5()
    digest.update(repr(extr.camera_view_matrix).encode())
    digest.update(repr((intr.fx_px, intr.fy_px, intr.cx_px, intr.cy_px,
                        intr.image_width, intr.image_height)).encode())
    plate_hash = _plate_info(solve).get("md5")
    if plate_hash:
        digest.update(plate_hash.encode())
    return digest.hexdigest()[:16]


def _models_info(solve: Any) -> dict[str, Any]:
    depth = getattr(solve, "depth", None)
    depth_value = getattr(depth, "value", None)
    depth_value = depth_value if isinstance(depth_value, dict) else {}
    source_method = str(getattr(solve, "source_method", "") or "")
    learned_prior = (source_method.split(":", 1)[1]
                     if "learned_prior:" in source_method else None)
    meta = getattr(solve, "debug_metadata", None) or {}
    return {
        "depth_model_id": depth_value.get("model_id"),
        "depth_is_metric": depth_value.get("is_metric"),
        "learned_prior": learned_prior,
        "vlm_report_present": bool(meta.get("vlm_report")),
    }


def build_project_manifest(solve: Any, *, artifacts: Iterable[ManifestArtifact] = (),
                           extra: dict[str, Any] | None = None) -> dict[str, Any]:
    from atlas_camera.core.scene_health import scale_health

    try:
        from atlas_camera import __version__ as atlas_version
    except Exception:  # noqa: BLE001
        atlas_version = "unknown"
    meta = getattr(solve, "debug_metadata", None) or {}
    scene_meta = getattr(getattr(solve, "projection_scene", None),
                         "debug_metadata", None) or {}
    confidence = getattr(solve.camera, "confidence", None)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA_VERSION,
        "atlas_version": atlas_version,
        "solve_schema_version": getattr(type(solve), "schema_version", None),
        "generated_at": now,
        "updated_at": now,
        "plate": _plate_info(solve),
        "solve_fingerprint": solve_content_fingerprint(solve),
        "solve": {
            "source_method": getattr(solve, "source_method", None),
            "seed": meta.get("seed", getattr(solve.camera, "seed", None)),
            "confidence": getattr(solve, "confidence", None),
            "confidence_detail": dict(getattr(confidence, "individual_metrics", {}) or {}),
        },
        "models": _models_info(solve),
        "scale": scale_health(solve).to_dict(),
        "scene_health": meta.get("scene_health"),
        "settings": {
            "scale_source": meta.get("scale_source"),
            "proxy_derivation": scene_meta.get("proxy_derivation"),
            "solve_mode": scene_meta.get("solve_mode"),
            "roll_trim_deg": meta.get("roll_trim_deg"),
        },
        "artifacts": [a.to_dict() for a in artifacts],
    }
    if extra:
        manifest["extra"] = dict(extra)
    manifest["identity_hash"] = _identity_hash_of(manifest)
    return manifest


def _identity_hash_of(manifest: dict[str, Any]) -> str:
    core = {key: manifest.get(key) for key in _IDENTITY_FIELDS}
    core["solve_fingerprint"] = manifest.get("solve_fingerprint")
    canonical = json.dumps(core, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()[:16]


def manifest_identity_hash(solve: Any) -> str:
    """The identity hash for this solve — what exporters embed as a comment."""
    return build_project_manifest(solve)["identity_hash"]


def write_project_manifest(solve: Any, output_dir: str | Path, *,
                           artifacts: Iterable[ManifestArtifact] = (),
                           extra: dict[str, Any] | None = None) -> Path:
    """Write (or merge-append into) ``atlas_project.json`` in ``output_dir``.

    Same solve fingerprint in the existing manifest -> extend its artifact
    list (deduped by kind+path) and bump ``updated_at``; a different
    fingerprint means a new solve owns the directory -> overwrite.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_FILENAME
    manifest = build_project_manifest(solve, artifacts=artifacts, extra=extra)

    if path.is_file():
        try:
            existing = load_project_manifest(path)
        except Exception:  # noqa: BLE001 — corrupt manifest: overwrite
            existing = None
        if existing and existing.get("solve_fingerprint") == manifest["solve_fingerprint"]:
            seen = {(a.get("kind"), a.get("path"))
                    for a in existing.get("artifacts", [])}
            merged = list(existing.get("artifacts", []))
            merged.extend(a for a in manifest["artifacts"]
                          if (a["kind"], a["path"]) not in seen)
            manifest["artifacts"] = merged
            manifest["generated_at"] = existing.get("generated_at",
                                                    manifest["generated_at"])

    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def load_project_manifest(path: str | Path) -> dict[str, Any]:
    """The single read entrypoint — schema validation + future migrations."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = data.get("schema")
    if not isinstance(schema, int) or schema < 1:
        raise ValueError(f"Not an atlas_project manifest (schema={schema!r})")
    if schema > MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Manifest schema {schema} is newer than this Atlas "
            f"({MANIFEST_SCHEMA_VERSION}) — upgrade atlas_camera to read it.")
    # v1 is current; migration steps for future versions belong HERE.
    return data
