"""Which camera angles actually see a hole — ranked, once, for every consumer.

Atlas has three ways to fill occluded geometry and until now each answered the
"from where?" question separately:

* the **Qwen multi-angle patch** asked the ARTIST to pick a named view in the
  browser, hoping it saw the hole;
* the **iPhone shoot list** derived its own shots;
* **path-guided repair** knew the answer exactly, but only for one hardwired
  frame (the camera path's end).

This module is that computation, extracted and generalised: given hole islands
and a set of candidate views, report how much of each island each view actually
reveals. What a consumer does with the ranking is its own business.

DELIBERATELY VOCABULARY-AGNOSTIC. It ranks whatever candidate views it is given
and knows nothing about Qwen. The named-view tables live in `comfy/view_prompts`
and core may not import `comfy/`, but the deeper reason is that the consumers
want genuinely different candidate sets: Qwen has 96 discrete named views, a
phone shoot is a continuous sphere, and a clean plate is "the same camera". A
solver that hard-coded one of those would serve one consumer and be worked
around by the other two.

Placement goes through `camera_math.orbit_camera`, the same helper
`AtlasAddPatchView` and `AtlasOcclusionMask` use. That is load-bearing: a view
ranked in one frame and rendered in another would recommend an angle that does
not show what was promised, and nothing downstream would notice.

Islands and candidate planes come from
`path_hole_repair.build_island_candidates`, so the solver ranks exactly the
islands that module repairs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from atlas_camera.core.camera_math import orbit_camera
from atlas_camera.core.path_hole_repair import (
    PathHoleRepairConfig,
    _project_vertices,
    _rasterize_triangles,
    build_island_candidates,
)


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only without numpy
        raise RuntimeError(
            "view_solver requires numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


@dataclass(frozen=True)
class CandidateView:
    """An orbit offset from the source camera, in degrees and a radius scale.

    Deltas rather than absolute angles because that is what `orbit_camera` takes
    and what the Qwen named-view vocabulary resolves to — `_named_view_orbit_delta`
    returns exactly this triple.
    """

    d_azimuth_deg: float
    d_elevation_deg: float
    distance_scale: float = 1.0
    #: Free-form, for whoever built the candidate — a Qwen prompt fragment, a
    #: shot name for the phone. The solver never parses it.
    label: str = ""


@dataclass(frozen=True)
class IslandVisibility:
    """How much of one hole island a view reveals."""

    island_id: int
    visible_px: int
    #: Cells in the island, i.e. its size in the mesh lattice. Lets a consumer
    #: tell "small island fully seen" from "big island barely clipped", which
    #: raw pixel counts cannot.
    island_cells: int


@dataclass(frozen=True)
class ViewScore:
    view: CandidateView
    visible_px: int
    islands_seen: int
    islands: tuple[IslandVisibility, ...] = field(default_factory=tuple)

    def sees(self, island_id: int) -> int:
        """Visible pixels for one island from this view; 0 if it is hidden."""
        for item in self.islands:
            if item.island_id == island_id:
                return item.visible_px
        return 0


def _pivot_from_mesh(mesh: Any, np: Any) -> tuple:
    """Mesh centroid, as the orbit pivot.

    The centroid, not the hole's own centre: orbiting about the hole would swing
    the camera wildly for a small island near frame edge and produce views that
    see the island against nothing recognisable. `AtlasAddPatchView` orbits the
    subject, and a ranked view has to be the view that will actually be rendered.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
    if not len(verts):
        return (0.0, 0.0, 0.0)
    return tuple(float(v) for v in verts.mean(axis=0))


def rank_views(
    mesh: Any,
    hole_mask: Any,
    *,
    source_camera: Any,
    candidates: Iterable[CandidateView],
    pivot: Sequence[float] | None = None,
    resolution: int = 512,
    min_visible_pixels: int = 8,
    config: PathHoleRepairConfig | None = None,
) -> list[ViewScore]:
    """Rank ``candidates`` by how much of the hole geometry each one reveals.

    Returns every candidate, best first — including views that see nothing, at
    ``visible_px = 0``. A caller that wants only useful views can filter, but the
    zeros are information: "no named view sees this island" is the answer that
    should send a hole to the phone or a clean plate instead of to Qwen, and
    dropping them would make that indistinguishable from "not evaluated".

    Ties break on the candidate's original order, so a caller that supplied its
    preferred views first keeps that preference.
    """
    np = _require_numpy()
    cfg = config or PathHoleRepairConfig()

    src_intr = source_camera.intrinsics
    src_width = int(src_intr.image_width or 0)
    src_height = int(src_intr.image_height or 0)
    selected = np.asarray(hole_mask, dtype=bool)
    if selected.shape != (src_height, src_width):
        raise ValueError(
            f"hole mask shape {selected.shape} does not match source camera "
            f"image {(src_height, src_width)}")

    candidates = list(candidates)
    if not candidates:
        return []

    built = build_island_candidates(
        mesh, selected, source_camera=source_camera, config=cfg)
    candidate_mesh = built["candidate_mesh"]
    candidate_faces = built["candidate_faces"]
    face_ids = built["face_ids"]
    component_by_id = built["component_by_id"]

    if not len(candidate_faces):
        # No island produced fillable geometry, so no view can reveal one. Report
        # every candidate at zero rather than an empty list: "nothing to see" and
        # "nothing evaluated" are different answers and the caller must be able to
        # tell them apart.
        return [ViewScore(view=v, visible_px=0, islands_seen=0) for v in candidates]

    scale = float(max(1, int(resolution))) / max(src_width, src_height, 1)
    out_width = max(1, int(round(src_width * scale)))
    out_height = max(1, int(round(src_height * scale)))
    sx = out_width / max(src_width, 1)
    sy = out_height / max(src_height, 1)
    fx = float(src_intr.fx_px or 1.0) * sx
    fy = float(src_intr.fy_px or src_intr.fx_px or 1.0) * sy
    cx = float(src_intr.cx_px if src_intr.cx_px is not None
               else src_width / 2.0) * sx
    cy = float(src_intr.cy_px if src_intr.cy_px is not None
               else src_height / 2.0) * sy

    piv = tuple(float(v) for v in pivot) if pivot is not None \
        else _pivot_from_mesh(mesh, np)
    floor = max(1, int(min_visible_pixels))

    scored: list[ViewScore] = []
    for order, view in enumerate(candidates):
        extrinsics = orbit_camera(
            source_camera.extrinsics, piv,
            d_azimuth_deg=float(view.d_azimuth_deg),
            d_elevation_deg=float(view.d_elevation_deg),
            distance_scale=float(view.distance_scale),
        )
        view_matrix = extrinsics.camera_view_matrix

        z_buffer = np.full((out_height, out_width), np.inf, dtype=np.float64)
        id_map = np.zeros((out_height, out_width), dtype=np.int32)
        # The support mesh first, WITHOUT ids: it is the occluder. Skipping it
        # would score every island as fully visible, which is the whole failure
        # this solver exists to prevent.
        base_xy, base_z = _project_vertices(
            mesh.vertices, view_matrix, fx, fy, cx, cy)
        _rasterize_triangles(
            base_xy, base_z, mesh.faces,
            np.zeros(len(mesh.faces), dtype=np.int32), z_buffer)
        cand_xy, cand_z = _project_vertices(
            candidate_mesh.vertices, view_matrix, fx, fy, cx, cy)
        # Matches build_path_hole_repair's tie-break: candidate planes and their
        # support mesh can land at numerically equal depth.
        z_buffer *= 1.0 + 1.0e-6
        _rasterize_triangles(
            cand_xy, cand_z, candidate_faces, face_ids, z_buffer, id_map)

        seen = id_map[id_map > 0]
        islands: list[IslandVisibility] = []
        total = 0
        if seen.size:
            for island_id, count in zip(*np.unique(seen, return_counts=True)):
                if int(count) < floor:
                    continue
                islands.append(IslandVisibility(
                    island_id=int(island_id),
                    visible_px=int(count),
                    island_cells=len(component_by_id.get(int(island_id), ())),
                ))
                total += int(count)
        islands.sort(key=lambda item: (-item.visible_px, item.island_id))
        scored.append(ViewScore(
            view=view, visible_px=total, islands_seen=len(islands),
            islands=tuple(islands),
        ))

    order_of = {id(s.view): i for i, s in enumerate(scored)}
    scored.sort(key=lambda s: (-s.visible_px, -s.islands_seen,
                               order_of[id(s.view)]))
    return scored


def best_view_per_island(scores: Sequence[ViewScore]) -> dict[int, ViewScore]:
    """The single best view for each island that ANY candidate can see.

    One view rarely covers every island — that is the point of a multi-angle
    patch. An island absent from the result is one no candidate revealed, which
    is a positive finding (send it to a real capture or a clean plate), not a
    gap in the data.
    """
    best: dict[int, tuple[int, ViewScore]] = {}
    for score in scores:
        for item in score.islands:
            current = best.get(item.island_id)
            if current is None or item.visible_px > current[0]:
                best[item.island_id] = (item.visible_px, score)
    return {island_id: score for island_id, (_px, score) in best.items()}
