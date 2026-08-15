"""Multi-angle diagnostic renders of predicted hidden geometry (RESEARCH).

The brief's Step 7. Renders each plate from controlled camera offsets
(+/-2, 5, 10, 20 degrees) in five passes:

  1 visible    Atlas relief mesh only, plate-textured  — what Atlas has today
  2 hidden     predicted invented geometry only, flat  — what the model made up
  3 combined   both, z-buffered                        — what the artist would get
  4 flat       invented geometry, flat diagnostic      — shape without texture flattery
  5 depth      invented geometry, coloured by distance BEHIND the visible surface

Passes 4 and 5 are the point. A textured novel view flatters a prediction: the
plate is projected onto whatever geometry exists, so wrong geometry still looks
like the photograph. A flat shader shows the invented SHAPE, and the
behind-distance ramp shows how far the model reached into the occluded volume.
Novel-view image quality is deliberately NOT a metric here.

The offsets ORBIT the scene pivot rather than spinning in place: rotating a
camera about its own centre reveals no parallax, so nothing behind an occluder
would ever be exposed and the whole exercise would show nothing.

A dedicated rasterizer is used rather than core.projection_render.render_scene
because that renders textured meshes via UVs and these passes need PER-VERTEX
colour (provenance, confidence, behind-distance). One code path for all five
passes keeps them directly comparable.

Usage (Atlas env):
    python diagnostic_renders.py --plate sh001 --volume out/fix_sh001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SP = Path(r"C:\Users\miike\AppData\Local\Temp\claude"
          r"\C--Users-miike-Desktop-AtlasCamera-Claude"
          r"\95656247-c586-4fdf-91ef-055be69d66cc\scratchpad")

OFFSETS_DEG = [-20, -10, -5, -2, 0, 2, 5, 10, 20]
BG = np.array([0.10, 0.11, 0.13], dtype=np.float32)


# ---------------------------------------------------------------------------
# rasterizer
# ---------------------------------------------------------------------------

def rasterize(verts, faces, colors, view, fx, fy, cx, cy, W, H):
    """Z-buffered triangle raster with per-vertex colour (flat-filled per face).

    Faces are filled with the mean of their vertex colours — adequate for a
    diagnostic at these mesh densities, and it keeps the depth test exact
    rather than interpolating a barycentric colour that would blur provenance
    across a boundary the render exists to show.
    """
    vm = np.asarray(view, dtype=np.float64)
    cam = verts @ vm[:3, :3].T + vm[:3, 3]
    z = -cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam[:, 0] / z * fx + cx
        v = -cam[:, 1] / z * fy + cy

    rgb = np.tile(BG, (H, W, 1)).astype(np.float32)
    zbuf = np.full((H, W), np.inf, dtype=np.float64)
    if len(faces) == 0:
        return rgb, zbuf

    ok = np.isfinite(z) & (z > 1e-6)
    fz = z[faces]
    fu = u[faces]
    fv = v[faces]
    valid = ok[faces].all(axis=1)
    # Cull faces entirely off-screen; keeps the per-face loop honest.
    valid &= (fu.max(1) >= 0) & (fu.min(1) < W) & (fv.max(1) >= 0) & (fv.min(1) < H)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return rgb, zbuf

    fcol = colors[faces].mean(axis=1)
    fdepth = fz.mean(axis=1)
    # Painter's order back-to-front, then a real z-test per pixel. Sorting first
    # means the z-test rarely rejects, which keeps the Python loop cheap.
    idx = idx[np.argsort(-fdepth[idx])]

    for i in idx:
        x0 = max(int(np.floor(fu[i].min())), 0)
        x1 = min(int(np.ceil(fu[i].max())) + 1, W)
        y0 = max(int(np.floor(fv[i].min())), 0)
        y1 = min(int(np.ceil(fv[i].max())) + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1)
        ys = np.arange(y0, y1)
        px, py = np.meshgrid(xs, ys)
        ax, ay = fu[i, 0], fv[i, 0]
        bx, by = fu[i, 1], fv[i, 1]
        cx_, cy_ = fu[i, 2], fv[i, 2]
        den = (by - cy_) * (ax - cx_) + (cx_ - bx) * (ay - cy_)
        if abs(den) < 1e-12:
            continue
        w0 = ((by - cy_) * (px - cx_) + (cx_ - bx) * (py - cy_)) / den
        w1 = ((cy_ - ay) * (px - cx_) + (ax - cx_) * (py - cy_)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zz = w0 * fz[i, 0] + w1 * fz[i, 1] + w2 * fz[i, 2]
        sub = zbuf[y0:y1, x0:x1]
        win = inside & (zz < sub)
        if not win.any():
            continue
        sub[win] = zz[win]
        rgb[y0:y1, x0:x1][win] = fcol[i]
    return rgb, zbuf


def composite(layers):
    """Z-composite (rgb, zbuf) layers, nearest wins."""
    rgb = np.tile(BG, layers[0][0].shape[:2] + (1,)).astype(np.float32)
    zb = np.full(layers[0][1].shape, np.inf)
    for lr, lz in layers:
        win = lz < zb
        zb[win] = lz[win]
        rgb[win] = lr[win]
    return rgb, zb


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------

def orbit_view(view, pivot, deg):
    """Rotate the camera about the world-up axis through ``pivot``.

    An orbit, not a pan: parallax is what exposes geometry behind an occluder,
    and a camera rotating about its own centre produces none.
    """
    if abs(deg) < 1e-9:
        return np.asarray(view, dtype=np.float64)
    c2w = np.linalg.inv(np.asarray(view, dtype=np.float64))
    t = np.radians(deg)
    ct, st = np.cos(t), np.sin(t)
    R = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = pivot - R @ pivot
    return np.linalg.inv(M @ c2w)


def ramp(t):
    """Viridis-ish ramp without a matplotlib dependency."""
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)[..., None]
    a = np.array([0.27, 0.00, 0.33], np.float32)
    b = np.array([0.13, 0.57, 0.55], np.float32)
    c = np.array([0.99, 0.91, 0.14], np.float32)
    return np.where(t < 0.5, a + (b - a) * (t / 0.5), b + (c - b) * ((t - 0.5) / 0.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", default="sh001")
    ap.add_argument("--image", default=None)
    ap.add_argument("--solve", default=None)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--grid", type=int, default=224,
                    help="Relief-mesh grid for the visible layer.")
    args = ap.parse_args()

    from atlas_camera.core.io import load_solve_json
    from atlas_camera.core.camera_spec import CameraSpec
    from atlas_camera.core import relief_mesh as rm
    from atlas_camera.core.hidden_geometry import register_layers_to_depth
    from tudf_to_atlas import (load_volume, moge_to_atlas_camera,
                               atlas_camera_to_world)

    solve_path = args.solve or str(SP / "sh001_rig.json")
    image_path = args.image or str(SP / "DSCF3915.png")
    out = Path(args.out or f"out/diag_{args.plate}")
    out.mkdir(parents=True, exist_ok=True)

    solve = load_solve_json(solve_path)
    intr = solve.camera.intrinsics
    spec = CameraSpec.from_intrinsics(intr)
    W0, H0 = int(intr.image_width), int(intr.image_height)
    view0 = np.asarray(solve.camera.extrinsics.camera_view_matrix, dtype=np.float64)

    # Render at a workable raster; intrinsics scale with it.
    S = args.size / max(W0, H0)
    W, H = int(round(W0 * S)), int(round(H0 * S))
    fx, fy = spec.fx * S, spec.fy * S
    cx, cy = spec.cx * S, spec.cy * S

    # --- visible geometry: the relief mesh Atlas already builds ---
    # Cache key MUST include the plate: an unconditional shared cache silently
    # built every plate's visible mesh from sh001's depth map (found 2026-08-15
    # when the machine plate reported rel_mad 0.421 and a flat in-front curve).
    cache = Path("out/_g5_depth") / (
        "d1.npz" if args.plate == "sh001" else f"depth_{args.plate}.npz")
    if cache.exists():
        with np.load(cache) as d:
            depth = np.asarray(d["depth"], dtype=np.float64)
    else:
        from atlas_camera.inference import depth_estimator as de
        r = de.estimate_depth(image_path,
                              model_id="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
        depth = np.asarray(getattr(r, "depth", r), dtype=np.float64)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, depth=depth.astype(np.float32))
    mesh = rm.build_relief_mesh(depth, view_matrix=view0, fx=spec.fx, fy=spec.fy,
                                cx=spec.cx, cy=spec.cy, grid_long_edge=args.grid)
    Vv = np.asarray(mesh.vertices, dtype=np.float64)
    Fv = np.asarray(mesh.faces, dtype=np.int64)
    print(f"visible mesh {len(Vv)} verts {len(Fv)} faces")

    # Plate colour per vertex, by projecting through the ORIGINAL camera.
    with Image.open(image_path) as im:
        plate = np.asarray(im.convert("RGB").resize((W, H), Image.LANCZOS),
                           dtype=np.float32) / 255.0
    camv = Vv @ view0[:3, :3].T + view0[:3, 3]
    zc = np.maximum(-camv[:, 2], 1e-9)
    uu = np.clip((camv[:, 0] / zc * fx + cx).astype(int), 0, W - 1)
    vv = np.clip((-camv[:, 1] / zc * fy + cy).astype(int), 0, H - 1)
    Cv = plate[vv, uu]

    # --- hidden geometry: VolFill invented surface ---
    from skimage.measure import marching_cubes
    vol = load_volume(Path(args.volume))
    with np.load(Path(args.volume) / "pred_tudf_256.npz") as dd:
        vis_tudf = np.asarray(dd["visible_tudf"], dtype=np.float32)
        moge_pts = np.asarray(dd["moge_points"], dtype=np.float64)
        moge_mask = np.asarray(dd["moge_mask"], dtype=bool)
    mz = moge_pts[..., 2].copy(); mz[~moge_mask] = 0.0
    mh, mw = mz.shape
    d_small = np.asarray(Image.fromarray(depth.astype(np.float32)).resize(
        (mw, mh), Image.BILINEAR), dtype=np.float64)
    s, rel_mad, _ = register_layers_to_depth(mz[..., None], d_small)
    print(f"MoGe->solve scale {s:.4f} (rel_mad {rel_mad:.3f})")

    vz, fh, _, _ = marching_cubes(vol.tudf, level=float(args.threshold))
    ijk = np.clip(np.rint(vz).astype(int), 0, vol.resolution - 1)
    supported = vis_tudf[ijk[:, 0], ijk[:, 1], ijk[:, 2]] <= (args.threshold + 1.0)
    pts = vol.bbox_min + vz[:, ::-1] * vol.voxel_size
    Vh = atlas_camera_to_world(moge_to_atlas_camera(pts, scale=s), view0)
    # SKY REJECTION. VolFill fills its whole cube, including the region MoGe
    # masked out as sky — and geometry invented in the sky is spurious by
    # construction (there is no surface there). Found visually at +20 deg on
    # sh001, where invented sky geometry smothered the frame. Atlas's own
    # doctrine is sky-aware throughout, so the diagnostic must be too, or it
    # shows an artist a defect that the pipeline would never ship.
    sky = ~moge_mask
    sky_img = np.asarray(Image.fromarray(sky.astype(np.uint8) * 255).resize(
        (W, H), Image.NEAREST)) > 127
    camh0 = Vh @ view0[:3, :3].T + view0[:3, 3]
    zh0 = np.maximum(-camh0[:, 2], 1e-9)
    uh0 = np.clip((camh0[:, 0] / zh0 * fx + cx).astype(int), 0, W - 1)
    vh0 = np.clip((-camh0[:, 1] / zh0 * fy + cy).astype(int), 0, H - 1)
    in_sky = sky_img[vh0, uh0]
    print(f"sky-rejected invented verts: {int((in_sky & ~supported).sum())} "
          f"of {int((~supported).sum())} "
          f"({100*(in_sky & ~supported).sum()/max((~supported).sum(),1):.1f}%)")

    # PROVENANCE, measured against what ATLAS ALREADY HAS.
    #
    # The first version labelled a vertex "invented" when it sat clear of
    # VolFill's own visible TUDF. That over-reports badly: the visible volume is
    # a sparse 6 cm voxelization, so a grazing road surface is barely occupied
    # and the whole road came back magenta even though Atlas's relief mesh
    # covers it perfectly (seen at +10 deg on sh001).
    #
    # The question an artist is actually asking is "what does this ADD to what I
    # already have", so the reference is the Atlas relief mesh itself — dense,
    # and exactly the geometry VolFill would have to beat to be worth running.
    from scipy.spatial import cKDTree
    novel_tol = max(3.0 * vol.voxel_edge_m, 0.25)
    drel, _ = cKDTree(Vv).query(Vh, k=1)
    novel = drel > novel_tol
    print(f"novel vs Atlas relief mesh (> {novel_tol:.2f} m): "
          f"{int((novel & ~in_sky).sum())} of {len(Vh)} "
          f"({100*(novel & ~in_sky).sum()/max(len(Vh),1):.1f}%)")

    keep_v = novel & ~in_sky
    fmask = keep_v[fh].all(axis=1)
    fh = fh[fmask]
    used = np.unique(fh)
    remap = np.full(len(Vh), -1, np.int64); remap[used] = np.arange(used.size)
    fh = remap[fh]; Vh = Vh[used]
    print(f"invented mesh {len(Vh)} verts {len(fh)} faces")

    # Pass-5 colour: how far BEHIND the visible surface each invented vertex sits.
    camh = Vh @ view0[:3, :3].T + view0[:3, 3]
    zh = np.maximum(-camh[:, 2], 1e-9)
    uh = np.clip((camh[:, 0] / zh * fx + cx).astype(int), 0, W - 1)
    vh = np.clip((-camh[:, 1] / zh * fy + cy).astype(int), 0, H - 1)
    dvis = np.asarray(Image.fromarray(depth.astype(np.float32)).resize(
        (W, H), Image.BILINEAR), dtype=np.float64)
    behind = zh - dvis[vh, uh]
    hi = float(np.percentile(behind[behind > 0], 95)) if (behind > 0).any() else 1.0
    Cbehind = ramp(behind / max(hi, 1e-6)).astype(np.float32)

    FLAT = np.tile(np.array([0.90, 0.24, 0.75], np.float32), (len(Vh), 1))

    pivot = np.median(Vv, axis=0)
    sheet, manifest = [], []
    for deg in OFFSETS_DEG:
        view = orbit_view(view0, pivot, deg)
        vis = rasterize(Vv, Fv, Cv, view, fx, fy, cx, cy, W, H)
        hid = rasterize(Vh, fh, FLAT, view, fx, fy, cx, cy, W, H)
        beh = rasterize(Vh, fh, Cbehind, view, fx, fy, cx, cy, W, H)
        comb = composite([vis, hid])
        row = [vis[0], hid[0], comb[0], beh[0]]
        sheet.append(np.concatenate(row, axis=1))
        for nm, img in zip(("visible", "hidden_flat", "combined", "hidden_behind"), row):
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
                out / f"{args.plate}_{deg:+03d}_{nm}.png")
        newly = float((hid[1] < vis[1]).mean())
        manifest.append({"offset_deg": deg, "hidden_in_front_fraction": newly})
        print(f"  {deg:+3d} deg  hidden-in-front {newly*100:5.2f}%")

    contact = np.concatenate(sheet, axis=0)
    Image.fromarray((np.clip(contact, 0, 1) * 255).astype(np.uint8)).save(
        out / f"{args.plate}_contact_sheet.png")
    (out / "manifest.json").write_text(json.dumps(
        {"plate": args.plate, "columns": ["visible", "hidden_flat", "combined",
                                          "hidden_behind"],
         "rows_offsets_deg": OFFSETS_DEG, "depth_scale": s,
         "registration_rel_mad": rel_mad, "frames": manifest}, indent=2),
        encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
