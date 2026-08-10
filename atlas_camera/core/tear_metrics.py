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

#: How much of the TRUE depth step a render must reproduce between adjacent
#: pixels before it counts as having torn rather than smeared. An absolute
#: threshold cannot make this call and the first version of this check got it
#: wrong: a curtain stretched across a cliff still steps ~`truth_jump / cell`
#: per pixel, which sails past any fixed metre value and scored a mesh that
#: never tears as perfect. Measured on the diagonal-cliff fixture (8 m cliff,
#: 8 px cell): a curtain delivers ~1 m/px, a real cut delivers the whole 8 m in
#: one pixel, so anything from ~0.2 to ~0.9 separates them cleanly.
REPRODUCED_STEP_RATIO = 0.5


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
    #: Real edges the mesh COVERED continuously, with no depth step across them —
    #: the rubber-sheet curtain. Only measurable when ``render_depth`` is given;
    #: 0.0 otherwise, and ``metadata["tear_evidence"]`` says which.
    bridged_edge_fraction: float = 0.0
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

        SECOND KNOWN WEAKNESS, and the reason ``render_depth`` exists (measured
        2026-08-10): without it, ``missed_edge_fraction`` reads a coverage GAP as
        the only evidence of a tear, so a mesh that tears the topology WITHOUT
        leaving a gap — `sub_quad_boundary` cuts the cell at the cliff and lets
        both sheets reach it — scores 0.98 missed against 0.00 for the very mesh
        it improves on. Pass ``render_depth`` and the depth step counts as
        evidence too; the ordering then comes out right.
        """
        return float(1.0
                     - w_false * self.false_tear_fraction
                     - w_missed * self.missed_edge_fraction)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("coverage", "gap_fraction", "justified_fraction",
              "false_tear_fraction", "missed_edge_fraction",
              "bridged_edge_fraction", "n_surface_px", "n_edge_px")}
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


def _dilate_max(values, radius: int, np):
    """Grey dilation: each pixel takes the max over its (radius)-neighbourhood."""
    out = np.asarray(values, dtype=np.float64).copy()
    for _ in range(max(0, int(radius))):
        p = np.pad(out, 1, mode="edge")
        out = np.maximum.reduce([p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1],
                                 p[1:-1, :-2], p[1:-1, 2:]])
    return out


def _neighbour_jump(values, valid, np):
    """Max |difference| to the left/up neighbour, counting only valid pairs."""
    v = np.where(valid, np.asarray(values, dtype=np.float64), np.nan)
    jump = np.zeros(v.shape, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        left = np.abs(v[:, 1:] - v[:, :-1])
        up = np.abs(v[1:, :] - v[:-1, :])
        jump[:, 1:] = np.maximum(jump[:, 1:], np.nan_to_num(left, nan=0.0))
        jump[1:, :] = np.maximum(jump[1:, :], np.nan_to_num(up, nan=0.0))
    return jump


def score_tears(*, truth_depth, surface_ids, coverage_alpha, render_depth=None,
                jump_m: float = OCCLUSION_JUMP_M,
                tolerance_px: int = EDGE_TOLERANCE_PX,
                step_ratio: float = REPRODUCED_STEP_RATIO) -> TearScore:
    """Score one render.

    ``coverage_alpha`` is the rasteriser's alpha (or any mask that is nonzero
    where the mesh covered a pixel) — i.e. what actually survived tearing.

    ``render_depth`` is the MESH's own rendered depth, and supplying it changes
    what counts as evidence of a tear. Tearing is a property of the mesh's
    TOPOLOGY, but alpha can only show a coverage gap, and those are not the same
    thing: a mesh can separate two sheets correctly and still cover every pixel,
    because at a depth cliff both sheets are photographed right up to the edge —
    which is exactly what `sub_quad_boundary` does. Scored on alpha alone that
    mesh is indistinguishable from one that rubber-sheeted across the cliff, and
    it scores WORSE than the whole-cell tear it improves on (measured
    missed_edge_fraction 0.00 -> 0.98).

    With ``render_depth``, an edge counts as torn if the mesh either left a gap
    OR reproduced the depth step across it, and ``bridged_edge_fraction``
    isolates the real failure: covered, continuous, no step — the curtain.
    Omitting it keeps the historical gap-only behaviour byte-for-byte, so every
    existing caller and recorded score is unaffected.
    """
    np = _require_numpy()
    d = np.asarray(truth_depth, dtype=np.float64)
    alpha = np.asarray(coverage_alpha, dtype=np.float64)
    if alpha.shape != d.shape:
        raise ValueError(f"alpha {alpha.shape} must match truth depth {d.shape}")
    if render_depth is not None:
        render_depth = np.asarray(render_depth, dtype=np.float64)
        if render_depth.shape != d.shape:
            raise ValueError(
                f"render_depth {render_depth.shape} must match truth depth {d.shape}")

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
    torn_evidence = _dilate(uncovered, tolerance_px, np)
    evidence_kind = "coverage_gap"
    bridged_fraction = 0.0

    if render_depth is not None:
        # A depth step in the mesh's OWN render is tearing too: the two sheets
        # meet at the cliff without sharing a vertex, so adjacent pixels sit a
        # whole cliff apart. But the step must be REPRODUCED, not merely present
        # — a curtain stretched over one grid cell also steps every pixel, just
        # by a cell-sized fraction of the cliff. Compare against the truth jump
        # rather than an absolute metre threshold, which cannot tell them apart.
        covered = alpha > 1e-3
        render_jump = _neighbour_jump(render_depth, covered, np)
        # Compare against the truth step of the NEARBY edge, not of this pixel.
        # Off-edge pixels have truth_jump 0, so a bare ratio test is trivially
        # satisfied there and dilation then carries that free pass onto every
        # edge — which scored a curtain as perfect. Scoping to `near_edge` and
        # max-filtering the truth jump over the same radius closes both holes.
        truth_near = _dilate_max(_neighbour_jump(d, surface, np), tolerance_px, np)
        stepped = (near_edge
                   & (render_jump > float(jump_m))
                   & (render_jump >= float(step_ratio) * truth_near))
        near_step = _dilate(stepped, tolerance_px, np)
        torn_evidence = torn_evidence | near_step
        evidence_kind = "coverage_gap+depth_step"
        # The failure missed_edge_fraction was always reaching for: the mesh
        # covered the edge AND ran continuously across it.
        bridged = edges & covered & ~torn_evidence
        bridged_fraction = (int(bridged.sum()) / n_edge) if n_edge else 0.0

    handled = int((edges & torn_evidence).sum())

    return TearScore(
        coverage=1.0 - n_gap / n_surface,
        gap_fraction=n_gap / n_surface,
        justified_fraction=(justified / n_gap) if n_gap else 1.0,
        false_tear_fraction=false_tears / n_surface,
        missed_edge_fraction=((n_edge - handled) / n_edge) if n_edge else 0.0,
        bridged_edge_fraction=bridged_fraction,
        n_surface_px=n_surface,
        n_edge_px=n_edge,
        metadata={"jump_m": float(jump_m), "tolerance_px": int(tolerance_px),
                  "tear_evidence": evidence_kind,
                  "step_ratio": float(step_ratio)},
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
