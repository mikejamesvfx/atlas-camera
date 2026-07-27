"""Native .cube LUT parsing + application (pure numpy, zero deps).

Pairs with AtlasRegisterPlate's recorded ``lut_path``: the plate-ref carries
the path for the Nuke/Maya handoff, and this actually APPLIES it in-graph for
preview/bake. Only the Resolve/Iridas ``.cube`` format (1D and 3D, the
overwhelmingly common interchange) is parsed natively — other formats
(.3dl/.spi/.csp) raise with a pointer to convert, rather than dragging
OpenColorIO in as a dependency for a text-format parser's worth of work.

Application is trilinear for 3D LUTs, linear for 1D. ``intensity`` follows
the Nuke Vectorfield convention: 0 = bypass, 1 = full, up to 2 extrapolates
past the LUT (out = in + intensity * (lut(in) - in)). DOMAIN_MIN/MAX are
honored; inputs outside the domain are clamped for the LOOKUP only — the
extrapolated mix keeps the result float-continuous rather than hard-clipped.
"""

from __future__ import annotations

from pathlib import Path


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded like the rest of the package
        raise ImportError("plate LUT needs numpy — pip install -e .[vision]") from exc
    return np


class CubeLUT:
    """Parsed .cube file: ``table`` is (N,3) for 1D or (N,N,N,3) for 3D."""

    def __init__(self, table, domain_min, domain_max, *, title: str = "",
                 is_3d: bool = False):
        self.table = table
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.title = title
        self.is_3d = is_3d

    @property
    def size(self) -> int:
        return int(self.table.shape[0])


def parse_cube(path) -> CubeLUT:
    """Parse a Resolve/Iridas .cube (1D or 3D). Raises ValueError on junk."""
    np = _require_numpy()
    p = Path(path)
    if p.suffix.lower() != ".cube":
        raise ValueError(
            f"unsupported LUT format '{p.suffix}' — only .cube is parsed "
            "natively; convert .3dl/.spi/.csp to .cube (any LUT tool) first")
    size_1d = size_3d = None
    dmin = (0.0, 0.0, 0.0)
    dmax = (1.0, 1.0, 1.0)
    title = ""
    rows: list[tuple[float, float, float]] = []
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        u = line.upper()
        if u.startswith("TITLE"):
            title = line[5:].strip().strip('"')
        elif u.startswith("LUT_1D_SIZE"):
            size_1d = int(line.split()[1])
        elif u.startswith("LUT_3D_SIZE"):
            size_3d = int(line.split()[1])
        elif u.startswith("DOMAIN_MIN"):
            dmin = tuple(float(v) for v in line.split()[1:4])
        elif u.startswith("DOMAIN_MAX"):
            dmax = tuple(float(v) for v in line.split()[1:4])
        elif u.startswith(("LUT_1D_INPUT_RANGE", "LUT_3D_INPUT_RANGE")):
            lo, hi = (float(v) for v in line.split()[1:3])
            dmin, dmax = (lo,) * 3, (hi,) * 3
        else:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rows.append(tuple(float(v) for v in parts[:3]))
                except ValueError:
                    raise ValueError(f"unparseable .cube data line: {line!r}")
    if size_3d:
        n = size_3d
        if len(rows) != n ** 3:
            raise ValueError(f".cube declares {n}^3={n ** 3} entries, has {len(rows)}")
        # .cube order: RED fastest -> reshape to [b][g][r] then index (r,g,b).
        table = np.asarray(rows, dtype=np.float64).reshape(n, n, n, 3)
        return CubeLUT(table, dmin, dmax, title=title, is_3d=True)
    if size_1d:
        if len(rows) != size_1d:
            raise ValueError(f".cube declares {size_1d} entries, has {len(rows)}")
        return CubeLUT(np.asarray(rows, dtype=np.float64), dmin, dmax,
                       title=title, is_3d=False)
    raise ValueError("not a .cube file: no LUT_1D_SIZE / LUT_3D_SIZE header")


def _apply_3d(lut: CubeLUT, rgb):
    """Trilinear sample of a 3D cube. rgb is (...,3) already domain-normalized."""
    np = _require_numpy()
    n = lut.size
    idx = np.clip(rgb, 0.0, 1.0) * (n - 1)
    i0 = np.clip(np.floor(idx).astype(np.int64), 0, n - 2)
    f = idx - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    t = lut.table  # indexed [b][g][r]

    def s(db, dg, dr):
        return t[b0 + db, g0 + dg, r0 + dr]

    c00 = s(0, 0, 0) * (1 - fr) + s(0, 0, 1) * fr
    c01 = s(0, 1, 0) * (1 - fr) + s(0, 1, 1) * fr
    c10 = s(1, 0, 0) * (1 - fr) + s(1, 0, 1) * fr
    c11 = s(1, 1, 0) * (1 - fr) + s(1, 1, 1) * fr
    c0 = c00 * (1 - fg) + c01 * fg
    c1 = c10 * (1 - fg) + c11 * fg
    return c0 * (1 - fb) + c1 * fb


def _apply_1d(lut: CubeLUT, rgb):
    np = _require_numpy()
    n = lut.size
    xs = np.linspace(0.0, 1.0, n)
    out = np.empty_like(rgb)
    for c in range(3):
        out[..., c] = np.interp(np.clip(rgb[..., c], 0.0, 1.0), xs, lut.table[:, c])
    return out


def apply_lut(image, lut: CubeLUT, *, intensity: float = 1.0):
    """Apply a parsed CubeLUT to a HxWxC float plate. Returns float32."""
    np = _require_numpy()
    img = np.asarray(image, dtype=np.float64)
    k = float(intensity)
    if k == 0.0 or img.size == 0:
        return img.astype(np.float32)
    chan = img if img.ndim == 3 else img[..., None]
    rgb = chan[..., :3] if chan.shape[-1] >= 3 else np.repeat(chan, 3, axis=-1)

    dmin = np.asarray(lut.domain_min, dtype=np.float64)
    dmax = np.asarray(lut.domain_max, dtype=np.float64)
    norm = (rgb - dmin) / np.maximum(dmax - dmin, 1e-9)
    looked = _apply_3d(lut, norm) if lut.is_3d else _apply_1d(lut, norm)
    out_rgb = rgb + k * (looked - rgb)

    if chan.shape[-1] >= 3:
        out = np.concatenate([out_rgb, chan[..., 3:]], axis=-1) \
            if chan.shape[-1] > 3 else out_rgb
    else:
        out = out_rgb[..., :1]
    if img.ndim == 2:
        out = out[..., 0]
    return out.astype(np.float32)
