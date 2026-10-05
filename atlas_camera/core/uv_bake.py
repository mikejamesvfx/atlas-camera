"""Bake the solved photo into a generated object's own UV texture.

A generated object (Pixal3D / TRELLIS.2) can arrive UV-unwrapped with a baked
base-colour texture covering EVERY side, including the ones the photo never
saw. The viewport paints such an object per fragment (photo where the solved
camera saw the surface, the model's texture elsewhere); a DCC has no shader
for that, so the export bakes the same decision into ONE texture in the
model's UV layout: each texel's surface point is projected through the solved
camera, and where the per-vertex ``photo_weight`` says the camera SAW it, the
photo replaces the generated colour.

UV convention: the glTF / ComfyUI ``MESH.uvs`` one -- origin TOP-left, ``v``
down the rows, texel ``(col, row)`` centred at ``((col + .5) / W,
(row + .5) / H)``.

Layering: ``core`` only -- numpy.
"""

from __future__ import annotations

from typing import Any

#: Triangles whose texel bounding box exceeds this are rasterised one at a
#: time (a block of ``n x n`` candidates per triangle bounds the memory).
_MAX_BLOCK = 64
_EPS = 1e-12

#: Default ``photo_weight`` feather either side of the split, matching the
#: viewport's smoothstep closely enough at texture resolution.
BAKE_FEATHER = 0.05


def _require_numpy() -> Any:
    try:
        import numpy as np
        return np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("UV baking requires numpy.") from exc


def rasterize_uv(faces: Any, uvs: Any, size: int | tuple[int, int]) -> tuple[Any, Any]:
    """Which triangle covers each texel, and where inside it.

    Returns ``(face_index, bary)``: ``face_index`` (H, W) int32, -1 where no
    triangle covers the texel centre; ``bary`` (H, W, 3) float32 barycentric
    weights of the covering triangle's corners. UV charts are not supposed to
    overlap; where they do, the later triangle wins. Bucketed by bounding-box
    size so same-footprint triangles are evaluated as one vectorised block.
    """
    np = _require_numpy()
    w, h = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    uv = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    face_index = np.full(h * w, -1, dtype=np.int32)
    bary = np.zeros((h * w, 3), dtype=np.float32)
    if not len(f):
        return face_index.reshape(h, w), bary.reshape(h, w, 3)
    # texel space with centres on integers: x = u*W - 0.5
    x = uv[f, 0] * w - 0.5
    y = uv[f, 1] * h - 0.5
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x0 = np.clip(np.ceil(np.nan_to_num(x.min(axis=1), nan=0.0)), 0, w - 1).astype(np.int64)
    x1 = np.clip(np.floor(np.nan_to_num(x.max(axis=1), nan=-1.0)), -1, w - 1).astype(np.int64)
    y0 = np.clip(np.ceil(np.nan_to_num(y.min(axis=1), nan=0.0)), 0, h - 1).astype(np.int64)
    y1 = np.clip(np.floor(np.nan_to_num(y.max(axis=1), nan=-1.0)), -1, h - 1).astype(np.int64)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    live = ok & (bw > 0) & (bh > 0)
    block = np.maximum(bw, bh)
    ids = np.arange(len(f), dtype=np.int64)
    for step in np.unique(block[live]):
        sel = np.flatnonzero(live & (block == step))
        chunks = ([sel[i:i + 1] for i in range(len(sel))] if step > _MAX_BLOCK
                  else np.array_split(sel, max(1, len(sel) * int(step) ** 2 // 4_000_000 + 1)))
        for c in chunks:
            if len(c):
                _raster_block(np, face_index, bary, x[c], y[c], x0[c], y0[c], x1[c], y1[c],
                              ids[c], int(step), w)
    return face_index.reshape(h, w), bary.reshape(h, w, 3)


def _raster_block(np, face_index, bary, x, y, x0, y0, x1, y1, ids, step, width):
    off = np.arange(step, dtype=np.int64)
    px = x0[:, None, None] + off[None, None, :]
    py = y0[:, None, None] + off[None, :, None]
    in_box = (px <= x1[:, None, None]) & (py <= y1[:, None, None])
    pxf, pyf = px.astype(np.float64), py.astype(np.float64)
    ax, bx, cx = x[:, 0, None, None], x[:, 1, None, None], x[:, 2, None, None]
    ay, by, cy = y[:, 0, None, None], y[:, 1, None, None], y[:, 2, None, None]
    w0 = (cx - bx) * (pyf - by) - (cy - by) * (pxf - bx)
    w1 = (ax - cx) * (pyf - cy) - (ay - cy) * (pxf - cx)
    w2 = (bx - ax) * (pyf - ay) - (by - ay) * (pxf - ax)
    area = w0 + w1 + w2
    sign = np.where(area >= 0.0, 1.0, -1.0)
    inside = in_box & (np.abs(area) > _EPS) & (w0 * sign >= 0) & (w1 * sign >= 0) & (w2 * sign >= 0)
    if not inside.any():
        return
    safe = np.where(np.abs(area) > _EPS, area, 1.0)
    flat = np.broadcast_to(py * width + px, inside.shape)[inside]
    fid = np.broadcast_to(ids[:, None, None], inside.shape)[inside]
    face_index[flat] = fid
    bary[flat, 0] = (w0 / safe)[inside]
    bary[flat, 1] = (w1 / safe)[inside]
    bary[flat, 2] = (w2 / safe)[inside]


def _bilinear(np, img, px, py):
    """Sample ``img`` (h, w, C) at float pixel coords (pixel centres on integers)."""
    h, w = img.shape[:2]
    px = np.clip(px, 0.0, w - 1.0)
    py = np.clip(py, 0.0, h - 1.0)
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (px - x0)[:, None], (py - y0)[:, None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def bake_photo_into_uv(
    vertices: Any,
    faces: Any,
    uvs: Any,
    generated_srgb: Any,
    photo_srgb: Any,
    photo_weight: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    split: float = 0.5,
    feather: float = BAKE_FEATHER,
) -> tuple[Any, dict[str, Any]]:
    """The generated texture with the photo baked in where the camera saw it.

    ``vertices`` world metres (N, 3); ``uvs`` (N, 2) top-left convention;
    ``generated_srgb`` (H, W, 3) 0..1 display sRGB -- its size is the bake
    size; ``photo_srgb`` (h, w, 3) 0..1 the solved plate; ``photo_weight``
    (N,) as stored on the primitive. Mixing is in LINEAR light like the
    viewport shader, and a texel the photo cannot reach (behind the camera,
    off the frame) keeps the generated colour whatever its weight says. Texels
    no triangle covers (the UV gutter) keep the generated texture.

    Returns ``(texture_srgb (H, W, 3) float32, stats)``.
    """
    np = _require_numpy()
    from atlas_camera.core.srgb import linear_to_srgb, srgb_to_linear

    gen = np.asarray(generated_srgb, dtype=np.float32)[..., :3]
    photo = np.asarray(photo_srgb, dtype=np.float32)[..., :3]
    th, tw = gen.shape[:2]
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    pw = np.asarray(photo_weight, dtype=np.float64).reshape(-1)
    face_index, bary = rasterize_uv(f, uvs, (tw, th))
    out = gen.copy()
    cov = face_index >= 0
    stats = {"texels": int(th * tw), "covered": int(cov.sum()), "photo_texels": 0,
             "photo_fraction": 0.0}
    if not cov.any():
        return out, stats
    fi = face_index[cov]
    b = bary[cov].astype(np.float64)
    tri = f[fi]                                            # (K, 3)
    pos = (b[:, :, None] * v[tri]).sum(axis=1)             # (K, 3)
    weight = (b * pw[tri]).sum(axis=1)
    view = np.asarray(view_matrix, dtype=np.float64).reshape(4, 4)
    cam = pos @ view[:3, :3].T + view[:3, 3]
    fwd = -cam[:, 2]
    front = fwd > 1e-6
    safe = np.where(front, fwd, 1.0)
    ix = cx + fx * cam[:, 0] / safe
    iy = cy - fy * cam[:, 1] / safe
    ph, pwid = photo.shape[:2]
    reach = front & (ix >= -0.5) & (ix <= pwid - 0.5) & (iy >= -0.5) & (iy <= ph - 0.5)
    t = np.clip((weight - (split - feather)) / max(2.0 * feather, 1e-9), 0.0, 1.0)
    mix = (t * t * (3.0 - 2.0 * t)) * reach
    if not (mix > 0).any():
        return out, stats
    sampled = _bilinear(np, photo, ix, iy)
    gen_lin = srgb_to_linear(out[cov])
    pho_lin = srgb_to_linear(sampled)
    m = mix[:, None]
    out[cov] = np.clip(linear_to_srgb(gen_lin * (1.0 - m) + pho_lin * m), 0.0, 1.0)
    n_photo = int((mix > 0.5).sum())
    stats.update(photo_texels=n_photo,
                 photo_fraction=round(n_photo / max(int(cov.sum()), 1), 4))
    return out.astype(np.float32), stats


def sample_texture_at_uvs(texture_srgb: Any, uvs: Any) -> Any:
    """Bilinear colour of ``texture_srgb`` (H, W, 3) at each ``uvs`` (N, 2)
    (top-left convention) -- per-vertex colours from a texture."""
    np = _require_numpy()
    tex = np.asarray(texture_srgb, dtype=np.float32)[..., :3]
    uv = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    h, w = tex.shape[:2]
    return _bilinear(np, tex, uv[:, 0] * w - 0.5, uv[:, 1] * h - 0.5).astype(np.float32)
