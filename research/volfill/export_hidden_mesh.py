"""VolFill TUDF -> mesh in ATLAS WORLD space, split visible vs invented (RESEARCH).

Runs in the Atlas env. Produces, per plate, a GLB whose vertices are coloured by
provenance:

    GREEN   = predicted surface that coincides with what was VISIBLE
    MAGENTA = predicted surface with no visible support -- INVENTED geometry

That colouring is the point. Novel-view prettiness is not the metric; the
question is "what did the model actually make up", and a flat provenance shader
answers it at a glance where a shaded render hides it.

Axis note: `marching_cubes` returns verts indexed (z, y, x) because the on-disk
TUDF is stored that way, so the columns are reversed into xyz before the metric
mapping -- the same reversal `volfill/visualize.py` does.

Usage:
    python export_hidden_mesh.py out/sh001_street [out/portal ...] \
        [--threshold 0.5] [--solve path.json] [--decimate 200000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tudf_to_atlas import load_volume, moge_to_atlas_camera, atlas_camera_to_world

VISIBLE_RGBA = np.array([70, 200, 110, 255], dtype=np.uint8)
INVENTED_RGBA = np.array([230, 60, 190, 255], dtype=np.uint8)


def mesh_from_volume(sample_dir: Path, *, threshold: float = 0.5,
                     view_matrix=None, scale: float = 1.0,
                     decimate: int | None = None):
    from skimage.measure import marching_cubes

    vol = load_volume(sample_dir)
    with np.load(sample_dir / "pred_tudf_256.npz") as d:
        vis_tudf = np.asarray(d["visible_tudf"], dtype=np.float32)

    field = vol.tudf
    lo, hi = float(field.min()), float(field.max())
    if not (lo < threshold < hi):
        return None, {"error": f"threshold {threshold} outside field range [{lo}, {hi}]"}

    verts_zyx, faces, _, _ = marching_cubes(field, level=float(threshold))

    # (z, y, x) index -> xyz metres in the MoGe camera frame.
    vs = vol.voxel_size
    idx_xyz = verts_zyx[:, ::-1]
    pts_cam_moge = vol.bbox_min + idx_xyz * vs

    # Provenance: sample the VISIBLE field at each vertex (nearest voxel). A
    # vertex is "visible-supported" if it sits inside the visible surface's own
    # truncation band, the same test the occluded split uses.
    ijk = np.clip(np.rint(verts_zyx).astype(int), 0, vol.resolution - 1)
    vis_at_vert = vis_tudf[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    supported = vis_at_vert <= (threshold + 1.0)

    pts = moge_to_atlas_camera(pts_cam_moge, scale=scale)
    if view_matrix is not None:
        pts = atlas_camera_to_world(pts, view_matrix)

    colors = np.where(supported[:, None], VISIBLE_RGBA[None, :], INVENTED_RGBA[None, :])

    import trimesh
    mesh = trimesh.Trimesh(vertices=pts, faces=faces, vertex_colors=colors,
                           process=False)
    if decimate and len(mesh.faces) > decimate:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=decimate)
        except Exception:
            pass  # decimation is cosmetic; a dense mesh is still viewable

    stats = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "invented_vertex_fraction": float(1.0 - supported.mean()),
        "voxel_edge_m": vol.voxel_edge_m,
        "threshold_voxels": threshold,
        "world_space": view_matrix is not None,
        "depth_scale": scale,
    }
    return mesh, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dirs", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--solve", default=None)
    ap.add_argument("--decimate", type=int, default=200000)
    ap.add_argument("--outdir", default="out/meshes")
    args = ap.parse_args()

    view_matrix = None
    if args.solve:
        from atlas_camera.core.io import load_solve_json
        solve = load_solve_json(args.solve)
        view_matrix = np.asarray(solve.camera.extrinsics.camera_view_matrix,
                                 dtype=np.float64)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = []
    for d in args.sample_dirs:
        d = Path(d)
        mesh, stats = mesh_from_volume(d, threshold=args.threshold,
                                       view_matrix=view_matrix,
                                       decimate=args.decimate)
        if mesh is None:
            print(f"{d.name:<20} SKIP: {stats['error']}")
            report.append({"sample": d.name, **stats})
            continue
        dest = outdir / f"{d.name}.glb"
        mesh.export(dest)
        stats.update(sample=d.name, path=str(dest))
        report.append(stats)
        print(f"{d.name:<20} {stats['faces']:>8} faces  "
              f"invented {stats['invented_vertex_fraction']*100:5.1f}%  -> {dest.name}")

    (outdir / "meshes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
