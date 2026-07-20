"""Content fingerprints — the identity behind every gate approval.

Split out of `node_helpers.py` in phase 3 of
`docs/dev/node_helpers_layering_plan.md`.

Gate widgets (`proceed`, `approved_for`) PERSIST in a saved workflow, so
approval has to be scoped to WHAT was approved rather than to the click. Without
a fingerprint a new image sails straight through the previous image's approval —
which happened, and read as "it ran and produced nothing" because the patch
branch correctly stayed paused while the gate did not.

Any persisted widget that gates execution needs an identity fingerprint; any
silent branch-skip needs a visible explanation. This module is the first half.
"""

from __future__ import annotations

import hashlib


def _image_fingerprint(image) -> str:
    """Short identity hash of an IMAGE tensor (16x-subsampled digest) — the
    approval token for AtlasAssessImage's ▶ Continue: proceed only applies to
    the image it was clicked FOR, so swapping the input photo re-arms the
    assessment gate instead of sailing through a stale approval (the same
    staleness class as 📐 Extract Angle's solve fingerprint)."""
    import hashlib

    arr = image[0, ::16, ::16].cpu().numpy()
    h = hashlib.md5()
    h.update(repr(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()[:16]
def _solve_fingerprint(solve, source_image) -> str:
    """Short identity hash of (recovered camera, source image) — stamped into
    📐 Extract Angle's client_data.patch_angle by the frontend and validated
    by AtlasBlockoutViewport.render(): an extraction from a DIFFERENT solve/
    image than the current one is treated as not-extracted, so swapping the
    input photo re-arms the patch-branch pause instead of silently running
    the previous image's stale angle. Image identity uses a 16x-subsampled
    tensor digest (full 4K hashing per execution is needless cost; a swapped
    photo always changes the subsample)."""
    import hashlib

    h = hashlib.md5()
    extr = solve.camera.extrinsics
    intr = solve.camera.intrinsics
    h.update(repr(extr.camera_view_matrix).encode())
    h.update(repr((intr.fx_px, intr.fy_px, intr.cx_px, intr.cy_px,
                   intr.image_width, intr.image_height)).encode())
    arr = source_image[0, ::16, ::16].cpu().numpy()
    h.update(repr(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()[:16]

__all__ = [
    "_image_fingerprint",
    "_solve_fingerprint",
]
