"""Route a hole to the fill machinery its SCENE KIND actually suits.

Atlas has two unrelated ways to close occluded geometry and they are good at
opposite things:

* **planar** — fit a construction (wall, room envelope, extruded profile) and
  license it per occlusion node via `completion_policy`. Exact where surfaces
  really are planar; nonsense on a hillside.
* **organic** — do not fit anything. Widen the depth bands so a hole falls
  INSIDE one rather than straddling a boundary, let the relief mesh's own
  hole-fill close it, then retopologise. Approximate by construction, which is
  fine: organic geometry is forgiving, and a slightly lumpy fill on foliage
  reads as foliage. The same lump on a wall reads as broken.

This module picks between them from an `AtlasAssessImage` payload — the VLM has
already looked at the plate, and its `scene_type` vocabulary was written in
construction terms that line up with Atlas's own derive nodes.

WHAT THE VLM IS AND IS NOT TRUSTED WITH. It selects a ROUTE; it never supplies a
value. Every construction downstream keeps its own acceptance gate — a plane
that does not fit is still rejected regardless of what was inferred here — and
nothing in this module can raise a completion's trust tier. The failure being
designed against is the quiet one: a mislabel routes correctly-executed geometry
at the wrong thing (call a facade "forest" and you get a soft blobby wall), so
every decision carries its reason and `fill_occluded=false` is an absolute veto.

`geometry` from the staged plan is deliberately NOT consulted. It has three
values (ground/card/relief) and selects a LAYER TYPE for a depth band — how to
build the projection stack, not how to fill a hole. Reading it as a fill
instruction was the first design here and it was wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atlas_camera.core.occlusion_graph import (
    POLICY_BACKDROP,
    POLICY_EXTEND_PLANE,
    POLICY_EXTRUDE_PROFILE,
    POLICY_NONE,
    POLICY_ROOM_ENVELOPE,
)

#: The two families, kept disjoint on purpose. A scene gets one or the other,
#: never a blend — mixing a plane fit into organic geometry is how a hillside
#: acquires a flat facet nobody asked for.
ROUTE_PLANAR = "planar"
ROUTE_ORGANIC = "organic"
ROUTE_NONE = "none"

#: scene_type -> (route, policy). The VLM's own prompt describes these in
#: construction terms; each planar entry has a derive node behind it.
_SCENE_ROUTES: dict[str, tuple[str, str]] = {
    # planar: real surfaces to fit
    "indoor":        (ROUTE_PLANAR, POLICY_ROOM_ENVELOPE),   # AtlasDeriveInteriorRoom
    "simple_walls":  (ROUTE_PLANAR, POLICY_EXTEND_PLANE),    # AtlasDeriveWalls
    "outdoor":       (ROUTE_PLANAR, POLICY_EXTEND_PLANE),    # AtlasDeriveRoofsFacades
    "aerial":        (ROUTE_PLANAR, POLICY_EXTEND_PLANE),
    "towers_spires": (ROUTE_PLANAR, POLICY_EXTRUDE_PROFILE), # AtlasDeriveTowersSpires
    # organic: nothing to fit; widen, fill, retopologise
    "organic":       (ROUTE_ORGANIC, POLICY_NONE),
    "mountains":     (ROUTE_ORGANIC, POLICY_NONE),
    "forests":       (ROUTE_ORGANIC, POLICY_NONE),
}

#: Unknown scene types take the organic route, NOT the planar one. Both are
#: guesses, but a soft fill on architecture looks wrong while a plane forced
#: through foliage produces geometry that is confidently, structurally false.
#: The cheaper mistake is the recoverable one.
_UNKNOWN_ROUTE = (ROUTE_ORGANIC, POLICY_NONE)

#: Per-family organic settings. Ragged tears want more edges allowed and the
#: sawtooth pass on; forests are the most forgiving and the noisiest depth, so
#: they get the loosest budget.
_ORGANIC_TUNING: dict[str, dict[str, Any]] = {
    "organic":   {"max_hole_edges": 384, "band_widening": 1.25,
                  "boundary_smooth_iterations": 4},
    "mountains": {"max_hole_edges": 512, "band_widening": 1.4,
                  "boundary_smooth_iterations": 6},
    "forests":   {"max_hole_edges": 768, "band_widening": 1.5,
                  "boundary_smooth_iterations": 8},
}
_ORGANIC_DEFAULT = _ORGANIC_TUNING["organic"]


@dataclass(frozen=True)
class FillPlan:
    """How one layer's holes should be closed, and why."""

    route: str
    #: Meaningful on ROUTE_PLANAR only. Always POLICY_NONE elsewhere, so a
    #: consumer that ignores `route` still cannot accidentally build a plane
    #: through a hillside.
    policy: str
    scene_type: str
    layer: str = ""

    # --- organic route settings (all inert on the planar route) -------------
    live_fill_holes: bool = False
    live_fill_max_hole_edges: int = 0
    live_fill_edge_sawteeth: bool = False
    retopo_method: str = "off"
    boundary_smooth_iterations: int = 0
    #: Multiplier on band width. Wider bands mean a hole falls INSIDE one rather
    #: than straddling a boundary, which is what makes mesh hole-fill tractable.
    band_widening: float = 1.0

    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def builds_geometry(self) -> bool:
        return self.route != ROUTE_NONE

    def describe(self) -> str:
        head = f"{self.layer or 'scene'}: route={self.route}"
        if self.route == ROUTE_PLANAR:
            head += f" policy={self.policy}"
        elif self.route == ROUTE_ORGANIC:
            head += (f" fill<= {self.live_fill_max_hole_edges} edges, "
                     f"retopo={self.retopo_method}, "
                     f"bands x{self.band_widening:g}")
        return head + "".join(f"\n    - {r}" for r in self.reasons)


def _organic_plan(scene_type: str, layer: str, reasons: list[str]) -> FillPlan:
    tune = _ORGANIC_TUNING.get(scene_type, _ORGANIC_DEFAULT)
    return FillPlan(
        route=ROUTE_ORGANIC, policy=POLICY_NONE, scene_type=scene_type,
        layer=layer,
        live_fill_holes=True,
        live_fill_max_hole_edges=int(tune["max_hole_edges"]),
        live_fill_edge_sawteeth=True,
        retopo_method="decimate",
        boundary_smooth_iterations=int(tune["boundary_smooth_iterations"]),
        band_widening=float(tune["band_widening"]),
        reasons=tuple(reasons),
    )


def policy_from_assessment(payload: dict[str, Any], *,
                           layer: str = "") -> FillPlan:
    """Pick a fill route for one layer from an AtlasAssessImage payload.

    ``layer`` selects an entry from ``payload["layers"]`` by name; omitted, the
    plan describes the scene as a whole — which is also the PRIMARY, the solve's
    own relief mesh rather than one of the VLM's named projection sources.

    Two ways to veto, because the primary is not a named layer. Top-level
    ``fill_occluded: false`` vetoes everything; an explicit ``{"name": ""}``
    entry vetoes the primary alone. Before 2026-07-31 neither worked: the layer
    lookup carried an ``and layer`` guard that made every entry unmatchable for
    the primary, so a veto aimed at it did nothing AND said nothing.

    Precedence, strongest first — every step can only ever REDUCE what gets
    built, because each input is inferred rather than measured:

    1. ``fill_occluded == false`` on the payload -> ROUTE_NONE (whole scene)
    2. ``fill_occluded == false`` on the layer   -> ROUTE_NONE (absolute veto)
    3. ``role == "sky"``                         -> POLICY_BACKDROP
    4. ``scene_type``                            -> planar or organic family
    5. unknown / missing scene_type              -> organic (the softer mistake)
    """
    reasons: list[str] = []
    settings = (payload or {}).get("recommended_settings") or {}
    scene_type = str(settings.get("scene_type") or "").strip().lower()

    # Layer lookup. The `and layer` guard this used to carry made an entry
    # UNMATCHABLE for the primary (whose key is ""), so a fill_occluded veto
    # could never protect it — and did nothing silently rather than erroring.
    # Matching on the name alone lets an explicit `{"name": "", ...}` entry
    # reach the primary, which is the only way to veto the solve's own geometry.
    entries = [i for i in ((payload or {}).get("layers") or [])
               if isinstance(i, dict)]
    entry: dict[str, Any] = {}
    matched = False
    for item in entries:
        # `"name" in item` matters only for the primary: it separates a
        # DELIBERATE `{"name": "", ...}` primary entry from a malformed one that
        # simply lost its name, which would otherwise silently become the entry
        # governing the whole scene.
        if "name" in item and str(item.get("name") or "") == layer:
            entry = item
            matched = True
            break
    if layer and not matched and entries:
        # A name that was asked for and is not in the assessment is a wiring
        # mistake, not a default. Silently falling through to the scene plan is
        # how a layer ends up filled by a policy nobody chose for it.
        reasons.append(
            f"layer {layer!r} is not in the assessment "
            f"(has: {', '.join(sorted(str(i.get('name') or '') for i in entries))}) "
            "— using the scene-wide plan, which may not be right for it")

    # A scene-wide veto, for the case a per-layer one cannot express: the
    # primary is the solve's own geometry rather than one of the VLM's named
    # layers, so `{"fill_occluded": false}` at the top level vetoes everything.
    if (payload or {}).get("fill_occluded") is False:
        return FillPlan(
            route=ROUTE_NONE, policy=POLICY_NONE, scene_type=scene_type,
            layer=layer,
            reasons=(*reasons,
                     "assessment set fill_occluded=false for the SCENE — "
                     "vetoed for every layer including the primary; a veto can "
                     "only ever leave a tear, never invent geometry"))

    # 1 — the veto. An inferred signal is allowed to switch construction OFF and
    # never on, so a wrong "do not fill" costs a tear while a wrong "fill" costs
    # invented geometry that looks deliberate.
    if entry and entry.get("fill_occluded") is False:
        return FillPlan(
            route=ROUTE_NONE, policy=POLICY_NONE, scene_type=scene_type,
            layer=layer,
            reasons=("assessment set fill_occluded=false for this layer — "
                     "vetoed; a veto can only ever leave a tear, never invent "
                     "geometry",))

    # 2 — sky is unambiguous and is not a surface anyone fits.
    if str(entry.get("role") or "").strip().lower() == "sky":
        return FillPlan(
            route=ROUTE_PLANAR, policy=POLICY_BACKDROP, scene_type=scene_type,
            layer=layer,
            reasons=("layer role is sky — backdrop, not a fitted surface",))

    # 3/4 — the scene family.
    route, policy = _SCENE_ROUTES.get(scene_type, _UNKNOWN_ROUTE)
    if scene_type not in _SCENE_ROUTES:
        reasons.append(
            f"scene_type {scene_type!r} not in the known vocabulary — taking "
            "the organic route, because a soft fill on architecture is "
            "recoverable while a plane forced through foliage is confidently "
            "wrong")
    else:
        reasons.append(f"scene_type {scene_type!r} -> {route}")

    if route == ROUTE_ORGANIC:
        reasons.append(
            "organic geometry is forgiving: widen the bands so holes fall "
            "inside one, let the relief mesh close them, then retopologise — "
            "no plane is fitted at all")
        return _organic_plan(scene_type, layer, reasons)

    reasons.append(f"policy {policy} licenses one construction; its own fit "
                   "gate still decides whether geometry is built")
    return FillPlan(route=ROUTE_PLANAR, policy=policy, scene_type=scene_type,
                    layer=layer, reasons=tuple(reasons))


def plans_for_layers(payload: dict[str, Any]) -> dict[str, FillPlan]:
    """A plan per named layer in the assessment, plus ``""`` for the scene."""
    out = {"": policy_from_assessment(payload)}
    for item in (payload or {}).get("layers") or []:
        name = str(item.get("name") or "").strip()
        if name:
            out[name] = policy_from_assessment(payload, layer=name)
    return out
