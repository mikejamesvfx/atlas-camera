"""The one place Atlas Y-up becomes Blender Z-up, and back.

An adapter boundary in the CLAUDE.md sense: `core` never learns Blender exists,
and no other module in this package flips an axis.

The transform is the SAME one `exporters/blender_exporter.py:54-65` bakes into
the generated scene script — ``T: (x, y, z) -> (x, -z, y)``. That script has
never actually been run (docs/DCC_EXPORTS.md records its verification as
"Script inspection"), so a test parses the emitted matrix rows and asserts they
agree with `T` here. That finally EXECUTES the convention on real geometry
instead of eyeballing it.

Why convert at all, when nothing in the recipe is axis-dependent: the `.blend`
the recipe dumps on failure is useless if it opens on its side, and one stated
convention beats "sometimes we convert". T is a proper rotation (det = +1), so
face winding survives — which matters, because shrinkwrap PROJECT mode follows
vertex normals.
"""
from __future__ import annotations

from typing import Any


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only without numpy
        raise RuntimeError(
            "atlas_camera.blender requires numpy. Install with: "
            "pip install -e .[vision]"
        ) from exc
    return np


#: Rows of T. Blender X = Atlas X, Blender Y = -Atlas Z, Blender Z = Atlas Y.
T_ROWS = ((1.0, 0.0, 0.0),
          (0.0, 0.0, -1.0),
          (0.0, 1.0, 0.0))


def transform_matrix(np: Any = None) -> Any:
    """T as a 3x3 array."""
    np = np or _require_numpy()
    return np.asarray(T_ROWS, dtype=np.float64)


def atlas_to_blender(points: Any) -> Any:
    """(N, 3) Atlas world (Y-up) -> Blender (Z-up). Metres throughout."""
    np = _require_numpy()
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)


def blender_to_atlas(points: Any) -> Any:
    """(N, 3) Blender (Z-up) -> Atlas world (Y-up). Exact inverse."""
    np = _require_numpy()
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return np.stack([p[:, 0], p[:, 2], -p[:, 1]], axis=1)
