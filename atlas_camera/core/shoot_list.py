"""Turn occluded surfaces into a photographic shooting brief.

Every hole-fill Atlas performs today is INVENTED — inpainting, LaRI's predicted
hidden geometry, edge-dilated smear. But a matte painter with a camera can
often just go and photograph the missing material: the paving, the brickwork,
the render, the tarmac. Not the specific occluded scene, which may be
unreachable, but the same KIND of surface at the same angle and light.

Atlas already knows what is missing. `core.occlusion_graph` enumerates one node
per fitted surface with its plane and depth range, one edge per silhouette tear
with its pixel count, and `AtlasLayerPlan` says which of those need a clean
plate. What has never existed is a way to say that in terms a photographer can
act on.

This converts the graph into shots. Per occluded surface it answers:

    what to photograph      the semantic concept, e.g. "pavement"
    at what angle           incidence to the surface, in degrees off face-on
    from what distance      derived from the surface's depth range
    at what resolution      pixels per metre, so the capture is not too soft
    how badly it is needed  torn pixel count, so the worst hole is shot first

WHAT IT DELIBERATELY DOES NOT DO
It invents no lighting numbers. Sun direction and hardness are the difference
between a patch that sits and a patch that reads as a sticker, and Atlas cannot
currently measure them from a single plate. Rather than emit a confident-looking
azimuth that is really a guess, each shot carries a REFERENCE REGION — the plate
pixels around the hole — for the photographer to match by eye. A human matching
a photograph against a reference is reliable; a fabricated sun angle is not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

#: Beyond this incidence the plate's own sampling of a surface is so foreshortened
#: that replacing it is pointless — a few pixels stretched across metres. Shots
#: are still emitted (the app can warn) but flagged.
EXTREME_INCIDENCE_DEG = 85.0

#: Below this torn area a hole is not worth a trip with a camera.
MIN_USEFUL_TEAR_PX = 400


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("shoot list requires numpy") from exc
    return np


@dataclass
class ShootSpec:
    """One photograph the artist needs to take."""

    node_id: str
    subject: str = ""                 # what to point the camera at
    hidden_by: list = field(default_factory=list)
    incidence_deg: float = 0.0        # 0 = face-on, 90 = edge-on
    distance_m: float = 0.0
    depth_range_m: tuple = (0.0, 0.0)
    px_per_m: float = 0.0             # sampling density to match or beat
    tear_px: int = 0
    priority: int = 0                 # 1 = shoot this first
    surface_normal: tuple = (0.0, 1.0, 0.0)
    kind: str = ""
    volumetric: bool = False          # not a plane: needs AR alignment, not a texture
    warnings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id, "subject": self.subject,
            "hidden_by": list(self.hidden_by),
            "incidence_deg": round(self.incidence_deg, 2),
            "distance_m": round(self.distance_m, 3),
            "depth_range_m": [round(float(v), 3) for v in self.depth_range_m],
            "px_per_m": round(self.px_per_m, 2),
            "tear_px": int(self.tear_px), "priority": int(self.priority),
            "surface_normal": [round(float(v), 5) for v in self.surface_normal],
            "kind": self.kind, "volumetric": self.volumetric,
            "warnings": list(self.warnings), "metadata": dict(self.metadata),
        }

    @property
    def guidance(self) -> str:
        """One line a person can actually follow."""
        if self.volumetric:
            return (f"{self.subject or self.node_id}: has interior depth — align to "
                    "the on-screen ghost rather than shooting a flat surface.")
        face = ("face-on" if self.incidence_deg < 20 else
                "at a slight angle" if self.incidence_deg < 50 else
                "at a shallow raking angle")
        return (f"{self.subject or self.node_id}: shoot {face} "
                f"({self.incidence_deg:.0f}° off square), close enough for "
                f"{self.px_per_m:.0f} px per metre of surface.")


def surface_incidence_deg(normal, camera_position, surface_point) -> float:
    """Angle between the surface normal and the direction the camera views it.

    0 means the camera looks straight down onto the surface; 90 means it skims
    along it. This is the number that decides whether replacement material will
    match — a paving slab photographed square looks nothing like the same slab
    raking away at 80 degrees, however good the texture.
    """
    np = _require_numpy()
    n = np.asarray(normal, dtype=np.float64)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return 0.0
    n = n / nn
    v = np.asarray(surface_point, dtype=np.float64) - np.asarray(camera_position, dtype=np.float64)
    vn = float(np.linalg.norm(v))
    if vn < 1e-9:
        return 0.0
    v = v / vn
    # Both orientations of the normal describe the same surface, so fold to the
    # acute angle: a plane fitted "backwards" must not read as 170 degrees.
    cos = abs(float(np.dot(n, -v)))
    return float(math.degrees(math.acos(max(0.0, min(1.0, cos)))))


def incidence_at_range(normal, plane_d: float, camera_position, distance_m: float) -> float:
    """Incidence where the camera meets a plane at ``distance_m`` away.

    This has to depend on distance, and that is the whole point: the same floor
    is steep underfoot and almost edge-on at the horizon, so a single "angle to
    the ground" is meaningless. Photograph pavement square when the plate sees it
    at 85 degrees and the replacement will not match anything.

    Geometry: the camera stands ``s`` off the plane along its normal, so a ray
    reaching the plane after travelling ``distance_m`` must close that gap over
    that length — giving ``cos(incidence) = s / distance``. Grazing falls out
    naturally as distance grows.
    """
    np = _require_numpy()
    n = np.asarray(normal, dtype=np.float64)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9 or distance_m <= 1e-6:
        return 0.0
    n = n / nn
    standoff = abs(float(np.dot(n, np.asarray(camera_position, dtype=np.float64))
                         - float(plane_d)))
    if standoff >= distance_m:
        # The plane is closer than the requested range in every direction, so no
        # ray of that length reaches it — treat as looking straight on rather
        # than returning a NaN from acos.
        return 0.0
    return float(math.degrees(math.acos(max(0.0, min(1.0, standoff / distance_m)))))


def sampling_px_per_m(focal_px: float, distance_m: float, incidence_deg: float) -> float:
    """How many plate pixels cover one metre of that surface.

    ``focal_px / distance`` is the head-on density; the cosine term is the
    foreshortening, which is why a road surface running to the horizon carries so
    little real detail however large the plate is. Replacement material shot
    below this density will read soft against its surroundings.
    """
    if distance_m <= 1e-6 or focal_px <= 0:
        return 0.0
    return float(focal_px / distance_m * max(0.0, math.cos(math.radians(incidence_deg))))


def build_shoot_list(solve, graph, *, layer_plan=None,
                     min_tear_px: int = MIN_USEFUL_TEAR_PX) -> list:
    """Occlusion graph -> ordered list of photographs to take.

    ``layer_plan`` is the LayerSpec list when present; it supplies the semantic
    concepts ("pavement", "brick wall") and the needs_clean_plate flag. Without
    it every occluded node is offered and the subject falls back to its kind.
    """
    np = _require_numpy()

    cam = solve.camera
    intr, extr = cam.intrinsics, cam.extrinsics
    focal_px = float(getattr(intr, "fx_px", 0.0) or 0.0)
    eye = np.asarray(extr.camera_position, dtype=np.float64)

    # Torn pixels per OCCLUDEE — the amount of that surface actually missing.
    torn: dict = {}
    for e in graph.edges or []:
        torn[e.occludee] = torn.get(e.occludee, 0) + int(e.tear_pixels or 0)
    hidden_by: dict = {}
    for e in graph.edges or []:
        hidden_by.setdefault(e.occludee, []).append(e.occluder)

    plans = {p.node_id: p for p in (layer_plan or [])}

    specs = []
    for node in graph.nodes or []:
        tear = int(torn.get(node.id, 0))
        plan = plans.get(node.id)
        if plan is not None and not getattr(plan, "needs_clean_plate", True):
            continue
        if tear < int(min_tear_px):
            continue

        lo, hi = node.depth_range_m or (0.0, 0.0)
        distance = float((lo + hi) / 2.0) if hi > 0 else float(lo)

        warnings = []
        plane = node.plane or None
        volumetric = plane is None
        if volumetric:
            # No fitted plane means there is no single surface to photograph —
            # an alleyway, a doorway, a recess. Those need the AR alignment path,
            # not a texture, and saying so is more useful than inventing an angle.
            normal = (0.0, 1.0, 0.0)
            incidence = 0.0
            warnings.append("no fitted plane — align to the ghosted geometry instead")
        else:
            normal = tuple(float(v) for v in plane.get("normal", (0.0, 1.0, 0.0)))
            # Evaluated AT THE SURFACE'S OWN RANGE, not at the plane's closest
            # point to the world origin — which for a ground plane sits directly
            # beneath the camera and would report every floor as face-on.
            incidence = incidence_at_range(
                normal, float(plane.get("d", 0.0)), eye, distance)
            if incidence >= EXTREME_INCIDENCE_DEG:
                warnings.append(
                    f"the plate sees this at {incidence:.0f}° — almost edge-on, so "
                    "there is very little real detail to match; consider rebuilding "
                    "rather than photographing")

        density = sampling_px_per_m(focal_px, distance, incidence)
        if density and density < 20.0:
            warnings.append(
                f"only {density:.0f} px per metre in the plate — a phone will "
                "comfortably beat this, so shoot for the surroundings, not this number")

        subject = ""
        if plan is not None and getattr(plan, "concepts", None):
            subject = ", ".join(str(c) for c in plan.concepts[:3])
        subject = subject or node.kind or node.id

        specs.append(ShootSpec(
            node_id=node.id, subject=subject,
            hidden_by=sorted(set(hidden_by.get(node.id, []))),
            incidence_deg=incidence, distance_m=distance,
            depth_range_m=(float(lo), float(hi)),
            px_per_m=density, tear_px=tear,
            surface_normal=normal, kind=node.kind,
            volumetric=volumetric, warnings=warnings,
            metadata={"completion_policy": node.completion_policy,
                      "texture_policy": node.texture_policy,
                      "confidence": float(node.confidence)},
        ))

    # Biggest hole first: that is where a missing patch is most visible, and an
    # artist with limited time should spend it there.
    specs.sort(key=lambda s: -s.tear_px)
    for i, s in enumerate(specs, start=1):
        s.priority = i
    return specs


def shoot_project(specs: list, *, plate_size=None, notes=None) -> dict:
    """The serialisable project an Atlas phone app would open."""
    return {
        "version": 1,
        "plate_size": list(plate_size) if plate_size else None,
        "shots": [s.to_dict() for s in specs],
        "guidance": [s.guidance for s in specs],
        # Stated in the payload, not just in documentation, so a client cannot
        # mistake the absence of lighting fields for "lighting does not matter".
        "lighting": {
            "measured": False,
            "note": ("Atlas does not measure sun direction or hardness from a "
                     "single plate. Match the reference crop by eye; flat or "
                     "overcast light is the safest capture because Atlas can "
                     "relight it from the surface normals it already has."),
        },
        "notes": list(notes or []),
    }
