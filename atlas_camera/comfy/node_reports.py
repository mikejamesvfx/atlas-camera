"""On-node report text and the export provenance manifest.

Split out of `node_helpers.py` in phase 3 of
`docs/dev/node_helpers_layering_plan.md`.

Two related jobs: the human-readable suffixes a node appends to its report
(scale trust, scene health), and `atlas_project.json` — the versioned
reproducibility manifest every exporter writes beside its artifacts.

The manifest rule that matters: a manifest failure must NEVER fail an export.
An export that succeeded and wrote no manifest is a missing audit trail; an
export that failed BECAUSE of its audit trail is lost work.
"""

from __future__ import annotations

import logging
from pathlib import Path


_IDENTITY_COMMENT_PREFIX = {".nk": "# ", ".py": "# ", ".ma": "// "}
def _scale_summary_suffix(solve) -> str:
    """Export-summary warning when the solve's metric scale isn't verified.

    Single source of truth is core.scene_health — never re-derive from
    scale_source ad hoc. Empty string when the scale is trustworthy, so
    healthy summaries are unchanged.
    """
    from atlas_camera.core.scene_health import scale_health
    sh = scale_health(solve)
    if sh.safe_to_export:
        return ""
    return f" | ⚠ scale {sh.status.upper()} — not verified"
def _health_summary_suffix(solve) -> str:
    """Export-summary marker when a scene-health stamp records warn/fail.

    Reads only the AtlasSceneHealthGate stamp (debug_metadata["scene_health"])
    — an acknowledged warning must survive into every artifact's summary.
    """
    stamp = (getattr(solve, "debug_metadata", None) or {}).get("scene_health")
    if not isinstance(stamp, dict) or stamp.get("level") in (None, "pass"):
        return ""
    n = len(stamp.get("flags") or [])
    ack = "acknowledged" if stamp.get("acknowledged") else "UNACKNOWLEDGED"
    return f" | 🩺 health: {str(stamp['level']).upper()} ({n} flag(s) {ack})"
def _write_export_manifest(solve, output_dir, kind_paths, exporter: str) -> None:
    """Write/merge atlas_project.json beside an export + embed the identity
    hash as a leading comment in text artifacts that tolerate one (.nk/.py/.ma).

    A manifest failure must NEVER fail the export — everything degrades to a
    log line. Called with [(kind, path), ...]; empty paths are skipped.
    """
    import logging
    try:
        from atlas_camera.exporters.manifest import (
            ManifestArtifact,
            manifest_identity_hash,
            write_project_manifest,
        )
        pairs = [(k, str(p)) for k, p in kind_paths if p]
        if not pairs:
            return
        write_project_manifest(
            solve, output_dir,
            artifacts=[ManifestArtifact(k, p, exporter) for k, p in pairs])
        ident = manifest_identity_hash(solve)
        for _, p in pairs:
            prefix = _IDENTITY_COMMENT_PREFIX.get(Path(p).suffix.lower())
            if not prefix or not Path(p).is_file():
                continue
            try:
                text = Path(p).read_text(encoding="utf-8")
                marker = f"{prefix}atlas_project_identity: "
                if text.startswith(marker):
                    text = text.split("\n", 1)[-1]
                Path(p).write_text(f"{marker}{ident}\n{text}", encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
    except Exception as exc:  # noqa: BLE001
        logging.warning("atlas_project.json manifest skipped: %s", exc)
def _format_hole_fill_report(enabled, n_filled, filled, faces_added, loops_left,
                             max_hole_edges, near_m, far_m):
    """Human-readable summary of an interior hole fill, for the export node.

    The fill is export-only, so this and ``preview_solve`` are the ONLY ways an
    artist learns what it did without a DCC round-trip — state the scope that
    was actually applied, not just the counts, since a disappointing result is
    usually a too-tight scope rather than a failed fill.
    """
    if not enabled:
        return "🔧 interior hole fill: off"
    lines = ["🔧 interior hole fill: ON"]
    if n_filled:
        lo, hi = min(filled), max(filled)
        span = f"{lo} edges" if lo == hi else f"{lo}–{hi} edges"
        lines.append(f"  filled {n_filled} hole{'s' if n_filled != 1 else ''} "
                     f"({span}, +{faces_added} faces)")
    else:
        lines.append("  filled 0 holes — nothing matched the scope below")
    # The outer frame is always one of these and must stay open by design.
    lines.append(f"  still open: {loops_left} boundary loop"
                 f"{'s' if loops_left != 1 else ''} (the outer frame is one)")
    scope = [f"max_hole_edges={int(max_hole_edges)}"]
    if near_m > 0.0 and far_m > 0.0:
        scope.append(f"band box {near_m:g}–{far_m:g} m")
    else:
        scope.append("band box off (set BOTH bounds > 0)")
    lines.append("  scope: " + ", ".join(scope))
    return "\n".join(lines)

__all__ = [
    "_IDENTITY_COMMENT_PREFIX",
    "_scale_summary_suffix",
    "_health_summary_suffix",
    "_write_export_manifest",
    "_format_hole_fill_report",
]
