"""VolFill canonical TUDF -> Atlas world space (RESEARCH, not production).

Nothing here imports from ``atlas_camera.comfy``; the only Atlas dependency is
``core.hidden_geometry.register_layers_to_depth`` for the one estimated
quantity (MoGe->Atlas depth scale). Per the Atlas layering rule this file is a
research consumer, and nothing in ``atlas_camera/`` imports it.

WHY THIS IS ARITHMETIC, NOT REGISTRATION
----------------------------------------
VolFill's "canonical frame" is not a learned canonicalization. Traced in
``volfill/preprocess/visible_tudf_prep.py::estimate_isotropic_bounds``: it takes
the MoGe-v2 visible point cloud in METRIC camera space, computes an axis-aligned
bbox, centres it, and inflates it to an isotropic cube

    half_scale = max(half_extent) * (1 + margin_ratio)      # margin_ratio 0.1
    bbox_min/max = center -/+ half_scale

Translation and uniform scale only -- no rotation, no reorientation. So the
inverse is closed form and exact, and the whole chain reduces to:

    canonical voxel index -> MoGe camera metres  (exact, from metadata.json)
    MoGe camera -> Atlas camera                  (exact, axis flip)
    Atlas camera -> Atlas world                  (exact, inv(camera_view_matrix))

with a single estimated scalar ``s`` absorbing MoGe-vs-Atlas depth scale.

ON-DISK CONTRACT (traced from ``inference_latent_visible.py::_save_result``)
---------------------------------------------------------------------------
``pred_tudf_256.npz['tudf']``  (256, 256, 256) float32, **array axes (z, y, x)**,
values in ``[0, truncation_voxels]`` (default 3.0) in VOXEL units -- the
in-memory tensor is [-1, 1] but ``_save_result`` denormalizes on the way out.

``metadata.json``: ``bbox_min`` (xyz metres), ``extent_xyz``, ``truncation_voxels``,
``field_range``, ``pred_resolution``.

Surface = ``tudf <= threshold`` in voxel units (``visualize.py`` default 0.5).
NOTE: upstream's ``_tudf_to_pointcloud`` uses ``tudf < self.tudf_threshold`` with
default 0.0, which selects nothing on a non-negative field -- it is dead code
(commented out at its only call site) and ``visualize.py`` is the live path. Do
not copy that predicate.

AXIS CONVENTIONS
----------------
MoGe camera space is OpenCV: x right, y DOWN, +z FORWARD.
Atlas camera space is OpenGL-style, per ``core/relief_mesh.py:182-184``
(``x = (u-cx)/fx*d``, ``y = -(v-cy)/fy*d``, ``z = -d``): x right, y UP,
-z forward.
So ``p_atlas_cam = diag(1, -1, -1) @ p_moge_cam``.

``camera_view_matrix`` is row-major WORLD->CAM; ``cam_to_world = inv(view_matrix)``
(``core/relief_mesh.py:11-12``). World math is built from the 4x4 only, never
from the 3x3 rotation -- Atlas hard rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# MoGe (OpenCV, y-down/+z-forward) -> Atlas camera (y-up/-z-forward).
MOGE_TO_ATLAS_CAM = np.diag([1.0, -1.0, -1.0])


@dataclass(slots=True)
class VolFillVolume:
    """A loaded VolFill prediction plus its canonical frame."""

    tudf: np.ndarray          # (R, R, R) float32, axes (z, y, x), [0, truncation]
    bbox_min: np.ndarray      # (3,) xyz metres, MoGe camera space
    extent: np.ndarray        # (3,) xyz metres
    truncation_voxels: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def resolution(self) -> int:
        return int(self.tudf.shape[0])

    @property
    def voxel_size(self) -> np.ndarray:
        """Per-axis voxel edge length in metres (xyz order)."""
        return np.asarray(self.extent, dtype=np.float64) / float(self.resolution)

    @property
    def voxel_edge_m(self) -> float:
        """Scalar voxel edge — the bbox is isotropic by construction."""
        return float(np.max(self.voxel_size))


def load_volume(sample_dir: str | Path) -> VolFillVolume:
    """Load ``pred_tudf_*.npz`` + ``metadata.json`` from a VolFill output dir."""
    sample_dir = Path(sample_dir)
    npzs = sorted(sample_dir.glob("pred_tudf_*.npz"))
    if not npzs:
        raise FileNotFoundError(f"no pred_tudf_*.npz in {sample_dir}")
    with np.load(npzs[0]) as data:
        tudf = np.asarray(data["tudf"], dtype=np.float32)
    meta = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    extent = np.asarray(meta["extent_xyz"], dtype=np.float64)
    if tudf.ndim != 3 or len(set(tudf.shape)) != 1:
        raise ValueError(f"expected a cubic (R,R,R) TUDF, got {tudf.shape}")
    return VolFillVolume(
        tudf=tudf,
        bbox_min=np.asarray(meta["bbox_min"], dtype=np.float64),
        extent=extent,
        truncation_voxels=float(meta.get("truncation_voxels", 3.0)),
        metadata=meta,
    )


def surface_points_canonical(
    vol: VolFillVolume,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Near-surface voxel centres in MoGe camera metres.

    ``threshold`` is in VOXEL units (``visualize.py`` semantics: ``tudf <= t``).
    Returns ``(points_xyz (N, 3), tudf_value (N,))`` — the TUDF value doubles as
    a distance-to-surface confidence for the Atlas result schema.
    """
    mask = vol.tudf <= float(threshold)
    # Array axes are (z, y, x) — matches upstream's `iz, iy, ix = where(...)`.
    iz, iy, ix = np.nonzero(mask)
    if ix.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float32)
    vs = vol.voxel_size
    pts = np.empty((ix.size, 3), dtype=np.float64)
    pts[:, 0] = vol.bbox_min[0] + (ix + 0.5) * vs[0]
    pts[:, 1] = vol.bbox_min[1] + (iy + 0.5) * vs[1]
    pts[:, 2] = vol.bbox_min[2] + (iz + 0.5) * vs[2]
    return pts, vol.tudf[iz, iy, ix]


def moge_to_atlas_camera(points_moge: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    """MoGe camera-space metres -> Atlas camera space, with the depth scale applied."""
    pts = np.asarray(points_moge, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")
    return (pts * float(scale)) @ MOGE_TO_ATLAS_CAM.T


def atlas_camera_to_world(points_cam: np.ndarray, view_matrix: Any) -> np.ndarray:
    """Atlas camera space -> Atlas world, via ``inv(camera_view_matrix)``."""
    pts = np.asarray(points_cam, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")
    vm = np.asarray(view_matrix, dtype=np.float64)
    if vm.shape != (4, 4):
        raise ValueError(f"view_matrix must be 4x4, got {vm.shape}")
    c2w = np.linalg.inv(vm)
    return pts @ c2w[:3, :3].T + c2w[:3, 3]


def estimate_depth_scale(
    moge_depth: np.ndarray,
    atlas_depth: np.ndarray,
) -> tuple[float, float]:
    """MoGe -> Atlas depth scale via Atlas's own layer-0 median registration.

    Reuses ``core.hidden_geometry.register_layers_to_depth`` rather than writing
    a second estimator: it already does exactly this median-ratio fit and returns
    ``rel_mad`` as the quality signal (~0.1 on architectural scenes).

    Returns ``(scale, rel_mad)``. ``rel_mad`` is ``inf`` when registration failed.
    """
    from atlas_camera.core.hidden_geometry import register_layers_to_depth

    moge = np.asarray(moge_depth, dtype=np.float64)
    scale, rel_mad, _ = register_layers_to_depth(moge[..., None], atlas_depth)
    return scale, rel_mad


def volfill_to_atlas_world(
    vol: VolFillVolume,
    view_matrix: Any,
    *,
    threshold: float = 0.5,
    scale: float = 1.0,
) -> dict[str, Any]:
    """Full chain: canonical TUDF -> Atlas world points + provenance.

    Shaped like the brief's ``AtlasHiddenGeometryResult`` so the eval and any
    later adapter agree on field names. ``confidence`` is ``1 - tudf/threshold``
    clipped to [0, 1] — 1 at the surface, 0 at the selection boundary.
    """
    pts_canon, tudf_vals = surface_points_canonical(vol, threshold=threshold)
    pts_cam = moge_to_atlas_camera(pts_canon, scale=scale)
    pts_world = atlas_camera_to_world(pts_cam, view_matrix)
    conf = np.clip(1.0 - tudf_vals / max(float(threshold), 1e-6), 0.0, 1.0)

    vm = np.asarray(view_matrix, dtype=np.float64)
    bounds = (
        np.stack([pts_world.min(axis=0), pts_world.max(axis=0)])
        if pts_world.size else np.zeros((2, 3))
    )
    return {
        "points_xyz": pts_world,
        "points_camera": pts_cam,
        "confidence": conf.astype(np.float32),
        "source": "volfill",
        "representation": "tudf",
        "world_to_camera": vm,
        "camera_to_world": np.linalg.inv(vm),
        "bounds": bounds,
        "metadata": {
            "threshold_voxels": float(threshold),
            "depth_scale": float(scale),
            "voxel_edge_m": vol.voxel_edge_m,
            "resolution": vol.resolution,
            "truncation_voxels": vol.truncation_voxels,
            "n_points": int(pts_world.shape[0]),
            "research_only": True,
        },
    }
