"""Persist fitted depth corrections, keyed by (model_id, scene_type).

Step (2) of the sequence in docs/ROADMAP.md: `depth_calibration.py` could fit
and serialize a correction, but nothing remembered one, so every fit died with
the graph that produced it.

WHY THE KEY IS (model_id, scene_type) AND LOOKUP IS EXACT
---------------------------------------------------------
A correction maps ONE model's characteristic error on ONE kind of scene. MoGe's
bias indoors is not V2-Outdoor's bias on a coastal vista, and the whole reason
the module exists is that a fit is only valid over the conditions it saw.

So lookup is an EXACT match and never falls back. No "close enough" model, no
"any" scene type, no nearest-neighbour. A near-miss fallback is precisely how a
coefficient fitted on a 1.2 m interior wall ends up rescaling a 200 m exterior
— the measured 67% error that put the range guard into `DepthCorrection` in the
first place. A miss returns None and the caller says so out loud.

The store holds NO shipped coefficients and never will: fitting them needs real
captures, and numbers fitted from this repo's synthetic fixtures must not be
distributed. An empty store is the correct state of a fresh clone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from atlas_camera.core.depth_calibration import DepthCorrection

#: Bumped independently of `DepthCorrection.SCHEMA_VERSION` — the envelope and
#: the entries version separately, so adding a field to one does not invalidate
#: files written by the other.
STORE_SCHEMA_VERSION = 1

#: Scene types a correction may be keyed by. This MIRRORS
#: `AtlasDeriveProjectionGeometry._SCENE_TYPE_PRESETS` plus "manual", which is
#: that node's default. Combo values are append-only across this repo, so the
#: vocabulary is reused rather than invented — a store keyed by words the rest
#: of the pack does not use would be unjoinable to the graph that produced it.
SCENE_TYPES = (
    "manual", "organic", "mountains", "forests", "aerial",
    "indoor", "outdoor", "simple_walls", "towers_spires",
)


def store_key(model_id: str, scene_type: str) -> str:
    return f"{model_id}::{scene_type}"


@dataclass
class CalibrationStore:
    """A flat, human-readable set of fitted corrections.

    Deliberately a plain JSON file rather than a database: it is small, an
    artist should be able to read it to see what their pipeline will apply, and
    a wrong coefficient must be deletable with a text editor.
    """

    entries: dict = field(default_factory=dict)
    #: Where it was loaded from, for reports. None for an in-memory store.
    path: str | None = None

    # --- io ------------------------------------------------------------------

    @classmethod
    def load(cls, path) -> "CalibrationStore":
        """Read a store. A MISSING file is an empty store, not an error.

        A fresh clone has no calibrations and that is the expected state; making
        the first lookup raise would push every caller into a try/except that
        cannot distinguish "nothing fitted yet" from "the file is corrupt".
        """
        p = Path(path)
        if not p.is_file():
            return cls(entries={}, path=str(p))

        raw = json.loads(p.read_text(encoding="utf-8"))
        version = int(raw.get("schema_version", 1))
        if version > STORE_SCHEMA_VERSION:
            raise ValueError(
                f"calibration store schema_version {version} is newer than this "
                f"build understands ({STORE_SCHEMA_VERSION}): {p}")

        entries = {}
        for row in raw.get("entries", []):
            model_id = str(row["model_id"])
            scene_type = str(row["scene_type"])
            entries[store_key(model_id, scene_type)] = {
                "model_id": model_id,
                "scene_type": scene_type,
                "correction": DepthCorrection.from_dict(row["correction"]),
                "note": str(row.get("note", "")),
            }
        return cls(entries=entries, path=str(p))

    def save(self, path=None) -> str:
        target = Path(path or self.path)
        if target.parent and not target.parent.is_dir():
            target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": STORE_SCHEMA_VERSION,
            "entries": [
                {"model_id": e["model_id"], "scene_type": e["scene_type"],
                 "note": e["note"], "correction": e["correction"].to_dict()}
                for _, e in sorted(self.entries.items())
            ],
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=False),
                          encoding="utf-8")
        self.path = str(target)
        return str(target)

    # --- access --------------------------------------------------------------

    def put(self, model_id: str, scene_type: str,
            correction: DepthCorrection, note: str = "") -> None:
        if scene_type not in SCENE_TYPES:
            raise ValueError(
                f"unknown scene_type {scene_type!r}, expected one of {SCENE_TYPES}")
        self.entries[store_key(model_id, scene_type)] = {
            "model_id": model_id, "scene_type": scene_type,
            "correction": correction, "note": note,
        }

    def lookup(self, model_id: str, scene_type: str):
        """Exact match or None. Never falls back to a different key."""
        entry = self.entries.get(store_key(model_id, scene_type))
        return entry["correction"] if entry else None

    def note_for(self, model_id: str, scene_type: str) -> str:
        entry = self.entries.get(store_key(model_id, scene_type))
        return entry["note"] if entry else ""

    def models(self) -> list:
        return sorted({e["model_id"] for e in self.entries.values()})

    def describe(self) -> str:
        """One line per stored correction, for a node report."""
        if not self.entries:
            return "no calibrations stored"
        rows = []
        for _, e in sorted(self.entries.items()):
            c = e["correction"]
            lo, hi = c.predicted_range
            rows.append(
                f"  {e['scene_type']:<14} {e['model_id']}  "
                f"[{c.model} a={c.a:.4f} b={c.b:.4f}, "
                f"fitted {lo:.2f}-{hi:.2f} m on {c.n_samples} samples, "
                f"improvement {c.improvement:.0%}]")
        return f"{len(rows)} calibration(s):\n" + "\n".join(rows)

    def __len__(self) -> int:
        return len(self.entries)
