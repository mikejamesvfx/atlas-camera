"""External paint-package bridges — the vendor-neutral core.

One doctrine, settled by measurement across two packages:

    The paint package selects and paints. Atlas decodes RAW, owns colorimetry,
    confines the edit, judges it against gates, and stitches the result.

The authorised mask stays Atlas-side because the judge must be independent of
the editor, and every vendor claim in here is a recorded measurement rather
than a capability sheet. See reports/paint_bridge_provenance.md.
"""
from __future__ import annotations

__all__ = ["masks", "roi", "confine", "score", "ocio", "vendors"]
