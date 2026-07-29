"""Score relief-mesh tearing against KNOWN geometry.

The golden-frame gate answers "did the picture change?". This answers "is the
tearing right?", which is a different and harder question — and the one you need
before an automated loop can tune tear thresholds without a human judging every
render.

It needs ground truth: a depth map plus a surface-id map saying which pixel
boundaries are genuine occlusion edges. Ray-cast fixtures supply that by
construction, and so does a LiDAR capture with segmentation — which is why this
lives in core rather than in the test fixtures. Nothing here is synthetic-only.

COVERAGE IS IN THE OBJECTIVE FROM THE START, deliberately.
Optimising tear placement alone has an obvious degenerate solution: tear
everything. Every real silhouette then gets torn, precision against edges looks
respectable, and the mesh is swiss cheese. `false_tear_fraction` and
`coverage` exist so that solution scores badly, and `combined_score` weights
them together — but read the WEIGHTS as an editorial choice, not a measurement.
The individual numbers are the honest output; the scalar is a convenience for
ranking, and `pareto_front` is there for when you would rather not collapse them
at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: A depth step (metres) big enough to count as a real occlusion rather than a
#: surface merely turning away from the camera. Creases below this are
#: orientation changes; tearing them is a choice, not a requirement.
OCCLUSION_JUMP_M = 0.25

#: How far (pixels) a tear may sit from a true edge and still count as placed
#: there. Rasterisation and grid quantisation both move a tear a cell or two.
EDGE_TOLERANCE_PX = 3


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("tear metrics require numpy") from exc
    return np


@dataclass
class TearScore:
    """Every component reported separately; the scalar is derived, not primary."""

    coverage: float = 0.0              # fraction of true surface the mesh covers
    gap_fraction: float = 0.0          # 1 - coverage
    justified_fraction: float = 0.0    # of the gaps, how many sit on a real edge
    false_tear_fraction: float = 0.0   # gaps on CONTINUOUS surface (swiss cheese)
    missed_edge_fraction: float = 0.0  # real edges with no tear (under-tearing)
    n_surface_px: int = 0
    n_edge_px: int = 0
    metadata: dict = field(default_factory=dict)

    def combined_score(self, *, w_false: float = 1.0, w_missed: float = 0.5) -> float:
        """One number for ranking. HIGHER is better; 1.0 is perfect.

        The weights are a judgement: a false tear (a hole in a continuous wall)
        is treated as worse than a missed edge (a silhouette left un-torn),
        because the first is visible as damage and the second usually just
        reads as slightly soft. Change them if your footage disagrees — that is
        a preference, and pretending otherwise would hide it inside a metric.

        KNOWN WEAKNESS, measured rather than suspected: ``missed_edge_fraction``
        is trivially satisfied by tearing EVERYWHERE, because scattered holes
        land within tolerance of every edge by luck. A regular grid of holes
        therefore scores 0.756 against 0.500 for a mesh that never tears at all,
        which is not a defensible ordering of those two renders.

        So do not optimise this scalar on its own. ``false_tear_fraction`` is
        what catches the degenerate solution (0.244 vs 0.000 for those two), and
        `pareto_front` exists precisely so the trade does not have to be
        collapsed into one number at all. The scalar is for coarse ranking.
        """
        return float(1.0
                     - w_false * self.false_tear_fraction
                     - w_missed * self.missed_edge_fraction)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("coverage", "gap_fraction", "justified_fraction",
              "false_tear_fraction", "missed_edge_fraction",
              "n_surface_px", "n_edge_px")}
        d["combined_score"] = self.combined_score()
        d["metadata"] = dict(self.metadata)
        return d


def _dilate(mask, radius: int, np):
    out = mask.copy()
    for _ in range(max(0, int(radius))):
        p = np.pad(out, 1, constant_values=False)
        out = (p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1]
               | p[1:-1, :-2] | p[1:-1, 2:])
    return out


def true_occlusion_edges(depth, surface_ids, *, jump_m: float = OCCLUSION_JUMP_M):
    """Pixels where the surface genuinely changes AND the depth genuinely steps.

    Both conditions matter. A surface-id change alone includes creases — a wall
    meeting a floor — where the geometry is continuous and tearing would punch a
    hole in a solid corner. A depth step alone fires on any steeply receding
    surface, which is the grazing case that has nothing to do with occlusion.
    """
    np = _require_numpy()
    sid = np.asarray(surface_ids)
    d = np.asarray(depth, dtype=np.float64)

    changed = np.zeros(sid.shape, dtype=bool)
    changed[:, 1:] |= sid[:, 1:] != sid[:, :-1]
    changed[1:, :] |= sid[1:, :] != sid[:-1, :]

    jump = np.zeros_like(d)
    jump[:, 1:] = np.maximum(jump[:, 1:], np.abs(d[:, 1:] - d[:, :-1]))
    jump[1:, :] = np.maximum(jump[1:, :], np.abs(d[1:, :] - d[:-1, :]))

    return changed & (jump > float(jump_m))


def score_tears(*, truth_depth, surface_ids, coverage_alpha,
                jump_m: float = OCCLUSION_JUMP_M,
                tolerance_px: int = EDGE_TOLERANCE_PX) -> TearScore:
    """Score one render.

    ``coverage_alpha`` is the rasteriser's alpha (or any mask that is nonzero
    where the mesh covered a pixel) — i.e. what actually survived tearing.
    """
    np = _require_numpy()
    d = np.asarray(truth_depth, dtype=np.float64)
    alpha = np.asarray(coverage_alpha, dtype=np.float64)
    if alpha.shape != d.shape:
        raise ValueError(f"alpha {alpha.shape} must match truth depth {d.shape}")

    surface = np.isfinite(d) & (d > 0)
    n_surface = int(surface.sum())
    if n_surface == 0:
        raise ValueError("ground-truth depth has no valid surface to score against")

    uncovered = surface & (alpha <= 1e-3)
    edges = true_occlusion_edges(d, surface_ids, jump_m=jump_m) & surface
    near_edge = _dilate(edges, tolerance_px, np)

    n_gap = int(uncovered.sum())
    justified = int((uncovered & near_edge).sum())
    false_tears = n_gap - justified

    # An edge counts as HANDLED if a tear landed within tolerance of it. Missed
    # edges are silhouettes the mesh stretched across instead of tearing.
    n_edge = int(edges.sum())
    handled = int((edges & _dilate(uncovered, tolerance_px, np)).sum())

    return TearScore(
        coverage=1.0 - n_gap / n_surface,
        gap_fraction=n_gap / n_surface,
        justified_fraction=(justified / n_gap) if n_gap else 1.0,
        false_tear_fraction=false_tears / n_surface,
        missed_edge_fraction=((n_edge - handled) / n_edge) if n_edge else 0.0,
        n_surface_px=n_surface,
        n_edge_px=n_edge,
        metadata={"jump_m": float(jump_m), "tolerance_px": int(tolerance_px)},
    )


def pareto_front(entries: list, objectives: tuple = ("false_tear_fraction",
                                                     "missed_edge_fraction")) -> list:
    """Non-dominated ``(label, TearScore)`` pairs, minimising every objective.

    Offered because collapsing tearing to one number throws away the actual
    shape of the trade — and the right point on that curve depends on the shot,
    not on a constant someone picked once. A config only drops out if another is
    at least as good on EVERY objective and strictly better on one.
    """
    def vec(s):
        return tuple(float(getattr(s, o)) for o in objectives)

    keep = []
    for label, score in entries:
        v = vec(score)
        dominated = any(
            all(vo <= vi for vo, vi in zip(vec(other), v))
            and any(vo < vi for vo, vi in zip(vec(other), v))
            for lbl, other in entries if lbl != label)
        if not dominated:
            keep.append((label, score))
    return keep
