"""Orbit stress test — the outlier/stretched-edge worklog's final acceptance item.

Scores a solved scene's projection coverage under small camera moves WITHOUT a
browser: for a grid of orbit deltas (default az +/-3/+/-6 deg, el +/-3 deg) the
recovered camera is orbited about its ground look-at pivot (the same
``camera_math.orbit_camera`` used for patch cameras), every mesh layer's
triangles are projected into the orbited view, and coverage is rasterized
(cv2.fillPoly) to measure per pose:

- ``hole_pct``      — frame fraction NO geometry covers (black on orbit)
- ``stretch_pct``   — frame fraction covered ONLY by world-anisotropic
                      triangles (max/min world edge ratio > threshold; the
                      stretched-texel suspects from the mesh-QA tier)

Honest limitations (a floor, not the full 🧭 Safe Zone): geometry coverage
only — per-pixel mattes, facing-ratio discards and z-ordering are not applied
(the browser probe remains the exact oracle), so real holes can only be EQUAL
OR WORSE than reported here. Faces touching the camera plane are culled whole
(no near-plane clipping) — conservative for frustum-bounded relief meshes at
small orbits, but a wildly over-wide plane can cull entirely. Static per-layer QA (torn/stretch stats) comes
from the mesh metadata the relief builder records.

Usage:
    python tools/orbit_stress_test.py <atlas_solve.json>
        [--res 512] [--az 3,6] [--el 3] [--stretch-ratio 12] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _meshes_from_solve(solve):
    """Every mesh-type primitive (primary proxy geometry + all layers) as
    (name, verts Nx3, faces Mx3, static_meta)."""
    from atlas_camera.exporters._layers import mesh_from_primitive

    out = []
    def add(prims, prefix):
        for prim in prims or []:
            if prim.primitive_type != "mesh":
                continue
            mesh = mesh_from_primitive(prim)
            if mesh is None:
                continue
            out.append((f"{prefix}{prim.name}", mesh.vertices, mesh.faces,
                        dict(prim.metadata or {})))
    add(solve.projection_scene.proxy_geometry, "")
    for src in getattr(solve, "projection_sources", None) or []:
        add(src.proxy_geometry, f"{src.name}/")
    return out


def _project(verts, view, fx, fy, cx, cy):
    """World Nx3 -> (pixel Nx2, cam_z N). Camera faces -Z; z_cam < 0 = in front."""
    import numpy as np

    hom = np.concatenate([verts, np.ones((len(verts), 1), dtype=np.float64)], axis=1)
    cam = hom @ np.asarray(view, dtype=np.float64).T
    z = cam[:, 2]
    fwd = -z  # forward distance, positive in front
    fwd_safe = np.where(np.abs(fwd) < 1e-9, 1e-9, fwd)
    u = cx + fx * cam[:, 0] / fwd_safe
    v = cy - fy * cam[:, 1] / fwd_safe
    return np.stack([u, v], axis=1), fwd


def run_stress(solve, *, res=512, az_steps=(3.0, 6.0), el_steps=(3.0,),
               stretch_ratio=12.0):
    import cv2
    import numpy as np

    from atlas_camera.core.camera_math import ground_lookat_pivot, orbit_camera

    intr = solve.camera.intrinsics
    extr = solve.camera.extrinsics
    W, H = int(intr.image_width), int(intr.image_height)
    scale = res / max(W, H)
    ew, eh = max(8, int(round(W * scale))), max(8, int(round(H * scale)))
    fx = float(intr.fx_px) * scale
    fy = float(intr.fy_px or intr.fx_px) * scale
    ccx = float(intr.cx_px if intr.cx_px is not None else W / 2.0) * scale
    ccy = float(intr.cy_px if intr.cy_px is not None else H / 2.0) * scale

    meshes = _meshes_from_solve(solve)
    if not meshes:
        raise SystemExit("Solve carries no mesh geometry — derive a relief "
                         "mesh (AtlasDeriveReliefMesh) before stress-testing.")

    # Precompute per-face world anisotropy once (pose-independent).
    face_data = []
    for name, verts, faces, meta in meshes:
        v = verts.astype(np.float64)
        a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
        el_len = np.stack([np.linalg.norm(b - a, axis=1),
                           np.linalg.norm(c - b, axis=1),
                           np.linalg.norm(a - c, axis=1)], axis=1)
        ratios = np.max(el_len, axis=1) / np.maximum(np.min(el_len, axis=1), 1e-9)
        face_data.append((name, v, faces, ratios > float(stretch_ratio), meta))

    pivot = ground_lookat_pivot(extr)
    deltas = [(0.0, 0.0)]
    for a in az_steps:
        deltas += [(a, 0.0), (-a, 0.0)]
    for e in el_steps:
        deltas += [(0.0, e), (0.0, -e)]

    rows = []
    for d_az, d_el in deltas:
        pose = (extr if (d_az == 0.0 and d_el == 0.0) else
                orbit_camera(extr, pivot, d_azimuth_deg=d_az, d_elevation_deg=d_el))
        view = pose.camera_view_matrix
        cover = np.zeros((eh, ew), dtype=np.uint8)
        suspect = np.zeros((eh, ew), dtype=np.uint8)
        for _, v, faces, is_stretched, _ in face_data:
            px, fwd = _project(v, view, fx, fy, ccx, ccy)
            ok = fwd[faces].min(axis=1) > 1e-6           # fully in front
            if not ok.any():
                continue
            polys = px[faces[ok]].astype(np.int32)
            cv2.fillPoly(cover, list(polys), 1)
            s = ok & is_stretched
            if s.any():
                cv2.fillPoly(suspect, list(px[faces[s]].astype(np.int32)), 1)
        frame = float(eh * ew)
        covered = float(cover.sum())
        rows.append({
            "d_az": d_az, "d_el": d_el,
            "hole_pct": round(100.0 * (1.0 - covered / frame), 2),
            "stretch_pct": round(100.0 * float((suspect & cover).sum()) / frame, 2),
        })

    static = [{"layer": name,
               "torn_fraction": meta.get("torn_fraction"),
               "stretch_ratio_p95": meta.get("stretch_ratio_p95"),
               "quad_coherence": meta.get("quad_coherence"),
               "n_faces": int(len(faces))}
              for name, _, faces, _, meta in face_data]
    return {"eval_res": [ew, eh], "stretch_ratio_threshold": stretch_ratio,
            "pivot": [round(p, 3) for p in pivot], "poses": rows,
            "layers": static,
            "note": "geometry coverage only — mattes/facing/z-order not "
                    "applied; real holes >= reported"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("solve_json")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--az", default="3,6")
    ap.add_argument("--el", default="3")
    ap.add_argument("--stretch-ratio", type=float, default=12.0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from atlas_camera.core.schema import LatentScene
    solve = LatentScene.from_dict(
        json.loads(Path(args.solve_json).read_text(encoding="utf-8")))
    report = run_stress(
        solve, res=args.res,
        az_steps=tuple(float(x) for x in args.az.split(",") if x),
        el_steps=tuple(float(x) for x in args.el.split(",") if x),
        stretch_ratio=args.stretch_ratio)

    print(f"ORBIT STRESS — eval {report['eval_res'][0]}x{report['eval_res'][1]}, "
          f"pivot {report['pivot']}")
    print(f"{'pose':>14} {'hole%':>7} {'stretch%':>9}")
    for row in report["poses"]:
        pose = f"az{row['d_az']:+g} el{row['d_el']:+g}"
        print(f"{pose:>14} {row['hole_pct']:>7.2f} {row['stretch_pct']:>9.2f}")
    print("layers:")
    for lay in report["layers"]:
        torn = lay["torn_fraction"]
        p95 = lay["stretch_ratio_p95"]
        print(f"  {lay['layer']:<28} faces {lay['n_faces']:>7}  "
              f"torn {torn if torn is not None else '-'}  "
              f"stretch_p95 {p95 if p95 is not None else '-'}")
    print(f"note: {report['note']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("json:", args.json)


if __name__ == "__main__":
    main()
