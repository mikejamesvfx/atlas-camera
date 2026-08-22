"""DEPRECATED alias for ``tools/paint_roundtrip_score.py``.

The scorer was never Affinity-specific — it reads both plates raw and gates them
with ``core.plate_falsification``, which is identical for any paint package. It
was renamed when the bridge gained a second vendor (Photoshop Beta, 2026-08-21).

This shim exists because ``reports/affinity_bridge_demo.md`` publishes this
exact command line, and a committed report whose command no longer runs is a
lie. It forwards argv unchanged, including the exit code (0 accepted, 2
rejected, 3 inconclusive), so any recorded invocation still behaves identically.

New work should call ``tools/paint_roundtrip_score.py``. Remove this shim at v1,
in the same commit that updates the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paint_roundtrip_score import main  # noqa: E402


if __name__ == "__main__":
    print("tools/affinity_roundtrip_score.py is deprecated; "
          "use tools/paint_roundtrip_score.py (same arguments, same exit codes).",
          file=sys.stderr)
    sys.exit(main())
