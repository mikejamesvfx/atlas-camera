"""The occlusion graph — what occludes what, and what completion is permitted.

Atlas already knows everything needed to answer "why is there a hole here":
the relief mesh records where it tore, the depth map says which side of a tear
is nearer, the plane/room/object fitters say what kind of surface each region
is, and ``scene_health`` says how much any of it can be trusted. What has never
existed is a structure that puts those together and states, per region, *what
may be built there*.

That structure is this module's output. It is the contract between measurement
and construction:

* **Nodes** are surfaces and objects — a fitted plane, a room shell, an object
  proxy, the ground. Each carries where it is, how confident Atlas is about it,
  and a ``completion_policy`` naming the ONE construction permitted for it.
* **Edges** are ``occludes`` relations recovered from silhouette tears: the
  nearer side of a tear is the occluder, the farther side the occludee.

Nothing here builds geometry, and nothing here changes a solve's measurements.
It is pure derivation, in the same spirit as ``scene_health``: read what the
solvers already recorded, decide nothing that cannot be justified from it, and
leave every judgement visible and reversible.

Design commitments worth not re-litigating later:

* Confidence is never invented here — it comes from ``core.scene_health``, per
  the standing rule that trust verdicts have exactly one home.
* ``completion_policy`` values are APPEND-ONLY once a node exposes them as a
  combo widget, because they serialize into saved workflows.
* ``unknown`` maps to policy ``none``. A tear Atlas cannot classify stays a
  tear, and says why. Guessing would put invented geometry into a solve that
  claims to be measured, which is the one thing this whole design exists to
  prevent.

The graph is stored on ``LatentScene.semantics`` — the reserved component slot
that has been empty since the schema was written. Numpy-only; no ComfyUI, no
torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

GRAPH_VERSION = 1

# Completion policies. APPEND-ONLY — these serialize into saved workflows.
POLICY_NONE = "none"
POLICY_EXTEND_PLANE = "extend_plane"
POLICY_ROOM_ENVELOPE = "room_envelope"
POLICY_EXTRUDE_PROFILE = "extrude_profile"
POLICY_CONSERVATIVE_PROXY = "conservative_proxy"
POLICY_BRIDGE_DISCONTINUITY = "bridge_discontinuity"
POLICY_BACKDROP = "backdrop"

COMPLETION_POLICIES = (
    POLICY_NONE,
    POLICY_EXTEND_PLANE,
    POLICY_ROOM_ENVELOPE,
    POLICY_EXTRUDE_PROFILE,
    POLICY_CONSERVATIVE_PROXY,
    POLICY_BRIDGE_DISCONTINUITY,
    POLICY_BACKDROP,
)

# Tear classifications, as produced by a VLM in Phase 2b or by the deterministic
# fallback below. APPEND-ONLY for the same reason.
TEAR_WALL_CONTINUATION = "wall_continuation"
TEAR_OBJECT_COMPLETION = "object_completion"
TEAR_BACKGROUND_FILL = "background_fill"
TEAR_UNKNOWN = "unknown"

TEAR_CLASSES = (
    TEAR_WALL_CONTINUATION,
    TEAR_OBJECT_COMPLETION,
    TEAR_BACKGROUND_FILL,
    TEAR_UNKNOWN,
)

# Which construction each tear class licenses. `unknown` deliberately licenses
# nothing — see the module docstring.
_CLASS_TO_POLICY = {
    TEAR_WALL_CONTINUATION: POLICY_EXTEND_PLANE,
    TEAR_OBJECT_COMPLETION: POLICY_CONSERVATIVE_PROXY,
    TEAR_BACKGROUND_FILL: POLICY_BRIDGE_DISCONTINUITY,
    TEAR_UNKNOWN: POLICY_NONE,
}

# Primitive names/sources that identify a node kind without guessing.
_BACKDROP_NAMES = ("projection_backdrop",)
_GROUND_NAMES = ("projection_ground", "atlas_projection_plane")


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The occlusion graph requires numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


@dataclass(slots=True)
class OcclusionNode:
    """One surface or object, and the single construction permitted for it."""

    id: str
    kind: str                      # surface | object | ground | backdrop
    source: str = ""               # which fitter produced it
    confidence: float = 0.0
    completion_policy: str = POLICY_NONE
    texture_policy: str = "edge_dilate"
    plane: dict[str, Any] | None = None
    depth_range_m: tuple[float, float] | None = None
    contact_node: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "source": self.source,
            "confidence": round(float(self.confidence), 4),
            "completion_policy": self.completion_policy,
            "texture_policy": self.texture_policy,
        }
        if self.plane is not None:
            out["plane"] = self.plane
        if self.depth_range_m is not None:
            out["depth_range_m"] = [round(float(v), 4) for v in self.depth_range_m]
        if self.contact_node:
            out["contact_node"] = self.contact_node
        if self.notes:
            out["notes"] = list(self.notes)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OcclusionNode":
        depth_range = data.get("depth_range_m")
        return cls(
            id=str(data["id"]),
            kind=str(data.get("kind", "surface")),
            source=str(data.get("source", "")),
            confidence=float(data.get("confidence", 0.0)),
            completion_policy=str(data.get("completion_policy", POLICY_NONE)),
            texture_policy=str(data.get("texture_policy", "edge_dilate")),
            plane=data.get("plane"),
            depth_range_m=(tuple(float(v) for v in depth_range)  # type: ignore[arg-type]
                           if depth_range else None),
            contact_node=data.get("contact_node"),
            notes=list(data.get("notes", [])),
        )


@dataclass(slots=True)
class OcclusionEdge:
    """A silhouette tear: ``occluder`` hides part of ``occludee``."""

    occluder: str
    occludee: str
    tear_pixels: int = 0
    boundary_length_px: int = 0
    near_depth_m: float = 0.0
    far_depth_m: float = 0.0
    tear_class: str = TEAR_UNKNOWN
    classified_by: str = "depth_heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "occluder": self.occluder,
            "occludee": self.occludee,
            "tear_pixels": int(self.tear_pixels),
            "boundary_length_px": int(self.boundary_length_px),
            "near_depth_m": round(float(self.near_depth_m), 4),
            "far_depth_m": round(float(self.far_depth_m), 4),
            "tear_class": self.tear_class,
            "classified_by": self.classified_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OcclusionEdge":
        return cls(
            occluder=str(data["occluder"]),
            occludee=str(data["occludee"]),
            tear_pixels=int(data.get("tear_pixels", 0)),
            boundary_length_px=int(data.get("boundary_length_px", 0)),
            near_depth_m=float(data.get("near_depth_m", 0.0)),
            far_depth_m=float(data.get("far_depth_m", 0.0)),
            tear_class=str(data.get("tear_class", TEAR_UNKNOWN)),
            classified_by=str(data.get("classified_by", "depth_heuristic")),
        )


@dataclass(slots=True)
class AtlasOcclusionGraph:
    """Scene decomposition plus the completion licence for each part."""

    version: int = GRAPH_VERSION
    nodes: list[OcclusionNode] = field(default_factory=list)
    edges: list[OcclusionEdge] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def node(self, node_id: str) -> OcclusionNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "AtlasOcclusionGraph":
        if not isinstance(data, dict):
            return cls()
        return cls(
            version=int(data.get("version", GRAPH_VERSION)),
            nodes=[OcclusionNode.from_dict(n) for n in data.get("nodes", [])],
            edges=[OcclusionEdge.from_dict(e) for e in data.get("edges", [])],
            notes=list(data.get("notes", [])),
        )

    def describe(self) -> str:
        """Artist-facing summary: what was found, and what may be built."""
        lines = [f"Occlusion graph v{self.version}: "
                 f"{len(self.nodes)} nodes, {len(self.edges)} tears"]
        for node in self.nodes:
            policy = node.completion_policy
            suffix = "" if policy != POLICY_NONE else "  (nothing will be built)"
            lines.append(f"  {node.id:<28} {node.kind:<9} "
                         f"conf {node.confidence:.2f}  -> {policy}{suffix}")
        for edge in self.edges:
            lines.append(
                f"  tear {edge.occluder} over {edge.occludee}: "
                f"{edge.tear_pixels} px, {edge.tear_class} ({edge.classified_by})"
            )
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def _node_kind(prim: Any) -> str:
    name = (getattr(prim, "name", None) or "").lower()
    if name in _BACKDROP_NAMES:
        return "backdrop"
    if name in _GROUND_NAMES or "ground" in name:
        return "ground"
    if getattr(prim, "primitive_type", None) in ("box", "cube", "cylinder"):
        return "object"
    return "surface"


def _default_policy(kind: str, source: str) -> str:
    """The construction a node licenses before any tear evidence refines it."""
    if kind == "backdrop":
        return POLICY_BACKDROP
    if kind == "object":
        return POLICY_CONSERVATIVE_PROXY
    if kind == "ground":
        return POLICY_EXTEND_PLANE
    if source == "room_cuboid":
        return POLICY_ROOM_ENVELOPE
    return POLICY_EXTEND_PLANE


def _plane_from_primitive(prim: Any) -> dict[str, Any] | None:
    """Plane normal + offset recovered from the primitive's own transform.

    Atlas planes use the THREE.PlaneGeometry frame (local X=u, Y=v, Z=normal),
    so the plane normal is the transform's third column — NOT its second. The
    same convention trips up anything that assumes a Y-up ground quad.
    """
    if getattr(prim, "primitive_type", None) != "plane":
        return None
    np = _require_numpy()
    mat = np.asarray(getattr(prim, "transform_matrix", None), dtype=np.float64)
    if mat.shape != (4, 4):
        return None
    normal = mat[:3, 2]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return None
    normal = normal / norm
    centre = mat[:3, 3]
    return {"normal": [round(float(v), 5) for v in normal],
            "d": round(float(np.dot(normal, centre)), 5)}


def build_occlusion_graph(
    solve: Any,
    *,
    depth: Any = None,
    tear_classes: dict[str, str] | None = None,
) -> AtlasOcclusionGraph:
    """Derive the occlusion graph from a solved scene.

    ``depth`` (H, W) forward metres, optional — supplies the per-tear depth
    statistics and the deterministic tear classification. Without it the graph
    still describes the scene's parts and their policies; it simply cannot say
    anything about the tears between them, and records that it could not.

    ``tear_classes`` maps ``"<occluder>|<occludee>"`` to one of
    :data:`TEAR_CLASSES`, letting a VLM pass (Phase 2b) override the
    deterministic classification. Unknown keys are ignored; unknown values are
    rejected rather than trusted, since a bad class silently licenses the wrong
    construction.
    """
    np = _require_numpy()
    from atlas_camera.core.scene_health import scale_health

    graph = AtlasOcclusionGraph()

    health = scale_health(solve)
    base_confidence = float(getattr(health, "confidence", 0.0) or 0.0)

    scene = getattr(solve, "projection_scene", None)
    prims = list(getattr(scene, "proxy_geometry", None) or []) if scene is not None else []
    derivation = {}
    if scene is not None:
        derivation = (getattr(scene, "debug_metadata", None) or {}).get(
            "proxy_derivation", {}) or {}
    source_method = str(derivation.get("primitive_method") or "")

    seen: set[str] = set()
    for index, prim in enumerate(prims):
        if getattr(prim, "primitive_type", None) == "mesh":
            continue                                # the relief mesh is not a part
        kind = _node_kind(prim)
        raw_name = getattr(prim, "name", None) or f"{kind}_{index}"
        node_id = raw_name if raw_name not in seen else f"{raw_name}_{index}"
        seen.add(node_id)
        meta = getattr(prim, "metadata", None) or {}
        source = str(meta.get("source") or source_method or "proxy_geometry")
        graph.nodes.append(OcclusionNode(
            id=node_id,
            kind=kind,
            source=source,
            confidence=base_confidence,
            completion_policy=_default_policy(kind, source_method),
            plane=_plane_from_primitive(prim),
        ))

    if not graph.nodes:
        graph.notes.append(
            "no proxy primitives on this solve — run AtlasDeriveProjectionGeometry "
            "before building the graph, or it has nothing to describe."
        )

    if depth is None:
        graph.notes.append(
            "no depth supplied: tears were not analysed, so this graph describes "
            "the scene's parts but licenses no tear-driven completion."
        )
        return graph

    graph.edges = _tears_from_depth(np, solve, depth, graph)
    _apply_tear_classes(graph, tear_classes)
    return graph


def _tears_from_depth(np: Any, solve: Any, depth: Any,
                      graph: AtlasOcclusionGraph) -> list[OcclusionEdge]:
    """Recover occluder/occludee pairs from silhouette discontinuities.

    A depth discontinuity in the source image IS an occlusion boundary: the
    nearer side is in front, the farther side continues behind it. That is the
    whole recovery — no segmentation model, no learned prior, just the geometry
    the depth map already states.

    Each side of a tear is attributed to the fitted surface whose own distance
    from the camera is closest to the measured depth there. Attribution by
    quantile band was tried first and is wrong: equal-count bands have nothing
    to do with where surfaces are, and on the bimodal depth a real occlusion
    produces they collapse into a single band and report no tears at all.
    Nodes already know where they are — use that.

    This is coarse where two surfaces sit at similar distances, and honestly so:
    Atlas carries no instance segmentation, and the completion policies act at
    surface granularity anyway.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2 or depth.size == 0:
        graph.notes.append("depth was not a 2D array; tears were not analysed.")
        return []

    valid = np.isfinite(depth) & (depth > 1e-6)
    if not valid.any():
        graph.notes.append("depth had no valid samples; tears were not analysed.")
        return []

    # Relative depth step, the same silhouette test build_relief_mesh tears on.
    edge_rel = 0.05
    dx = np.zeros_like(depth, dtype=bool)
    dy = np.zeros_like(depth, dtype=bool)
    both_x = valid[:, 1:] & valid[:, :-1]
    both_y = valid[1:, :] & valid[:-1, :]
    step_x = np.abs(depth[:, 1:] - depth[:, :-1])
    step_y = np.abs(depth[1:, :] - depth[:-1, :])
    ref_x = np.minimum(depth[:, 1:], depth[:, :-1])
    ref_y = np.minimum(depth[1:, :], depth[:-1, :])
    dx[:, :-1] = both_x & (step_x > edge_rel * np.maximum(ref_x, 1e-6))
    dy[:-1, :] = both_y & (step_y > edge_rel * np.maximum(ref_y, 1e-6))
    tear = dx | dy
    if not tear.any():
        graph.notes.append("no silhouette discontinuities found — nothing occludes "
                           "anything at this depth threshold.")
        return []

    surfaces = [n for n in graph.nodes if n.kind in ("surface", "ground", "object")]
    if not surfaces:
        graph.notes.append("tears exist but no surface nodes to attribute them to.")
        return []

    node_depths = _node_forward_depths(np, solve, surfaces)
    if node_depths is None:
        graph.notes.append("proxy primitives carry no usable transforms; tears "
                           "could not be attributed to surfaces.")
        return []
    assign = _nearest_node(np, depth, node_depths)
    assign = np.where(valid, assign, -1)

    h, w = depth.shape
    rows, cols = np.nonzero(tear)
    nb_c = np.minimum(cols + 1, w - 1)
    nb_r = np.minimum(rows + 1, h - 1)

    here = assign[rows, cols]
    # Whichever neighbour actually crossed the discontinuity at this pixel.
    across_x = assign[rows, nb_c]
    across_y = assign[nb_r, cols]
    depth_here = depth[rows, cols]

    out: list[OcclusionEdge] = []
    for a in range(len(surfaces)):
        for b in range(len(surfaces)):
            if a == b or node_depths[a] >= node_depths[b]:
                continue        # a must be the NEARER surface to be the occluder
            touches = (here == a) & ((across_x == b) | (across_y == b))
            count = int(touches.sum())
            if count == 0:
                continue
            far_side = assign == b
            out.append(OcclusionEdge(
                occluder=surfaces[a].id,
                occludee=surfaces[b].id,
                tear_pixels=count,
                boundary_length_px=count,
                near_depth_m=float(np.median(depth_here[touches])),
                far_depth_m=float(np.median(depth[far_side])) if far_side.any() else 0.0,
                tear_class=_classify_tear(surfaces[a], surfaces[b]),
            ))
    if not out:
        graph.notes.append("discontinuities found but none straddled two fitted "
                           "surfaces; no completion is licensed from them.")
    return out


def _node_forward_depths(np: Any, solve: Any,
                         surfaces: list[OcclusionNode]) -> Any:
    """Each surface node's own forward distance from the recovered camera.

    Taken from the primitive's transform translation, pushed through the solve's
    4x4 view matrix — the one convention used everywhere in ``core``. Never the
    3x3 rotation.
    """
    scene = getattr(solve, "projection_scene", None)
    prims = {(getattr(p, "name", None) or ""): p
             for p in (getattr(scene, "proxy_geometry", None) or [])}
    vm = np.asarray(solve.camera.extrinsics.camera_view_matrix, dtype=np.float64)
    if vm.shape != (4, 4):
        return None

    depths = []
    for node in surfaces:
        prim = prims.get(node.id)
        if prim is None:                       # de-duplicated id (name_index)
            prim = prims.get(node.id.rsplit("_", 1)[0])
        mat = np.asarray(getattr(prim, "transform_matrix", None), dtype=np.float64) \
            if prim is not None else None
        if mat is None or mat.shape != (4, 4):
            return None
        centre = mat[:3, 3]
        cam = centre @ vm[:3, :3].T + vm[:3, 3]
        depths.append(float(-cam[2]))          # -Z forward
    return np.asarray(depths, dtype=np.float64)


def _nearest_node(np: Any, depth: Any, node_depths: Any) -> Any:
    """Index of the surface whose distance best explains each pixel's depth."""
    diff = np.abs(depth[..., None] - node_depths[None, None, :])
    return np.argmin(diff, axis=-1)


def _classify_tear(occluder: OcclusionNode, occludee: OcclusionNode) -> str:
    """Deterministic fallback classification, from node kinds alone.

    Intentionally conservative: it only claims a class when the pair of kinds
    makes one obvious, and returns ``unknown`` otherwise so the tear is left
    open rather than filled on a guess. The VLM pass exists to do better than
    this, not to rubber-stamp it.
    """
    if occludee.kind == "backdrop":
        return TEAR_BACKGROUND_FILL
    if occluder.kind == "object" and occludee.kind in ("surface", "ground"):
        return TEAR_WALL_CONTINUATION
    if occluder.kind in ("surface", "ground") and occludee.kind == "surface":
        return TEAR_WALL_CONTINUATION
    if occluder.kind == "object" and occludee.kind == "object":
        return TEAR_OBJECT_COMPLETION
    return TEAR_UNKNOWN


def _apply_tear_classes(graph: AtlasOcclusionGraph,
                        tear_classes: dict[str, str] | None) -> None:
    """Fold classifications into edges, then propagate policy onto occludees."""
    if tear_classes:
        for edge in graph.edges:
            override = tear_classes.get(f"{edge.occluder}|{edge.occludee}")
            if override is None:
                continue
            if override not in TEAR_CLASSES:
                graph.notes.append(
                    f"ignored unrecognised tear class {override!r} for "
                    f"{edge.occluder}|{edge.occludee} — left as {edge.tear_class}."
                )
                continue
            edge.tear_class = override
            edge.classified_by = "vlm"

    # An occludee's policy is set by the tear that exposes it: that is the
    # surface completion actually has to build into, and observed tear evidence
    # beats the kind-based default the node was seeded with.
    #
    # Resolution when several tears expose the same surface: POLICY_NONE always
    # wins. One unclassifiable tear means part of that surface is genuinely
    # unknown, and building over the rest of it would hide that fact behind
    # geometry that looks finished.
    decided: set[str] = set()
    for edge in graph.edges:
        node = graph.node(edge.occludee)
        if node is None or node.kind == "backdrop":
            continue
        policy = _CLASS_TO_POLICY.get(edge.tear_class, POLICY_NONE)
        if policy == POLICY_NONE:
            node.completion_policy = POLICY_NONE
            node.notes.append(
                f"tear from {edge.occluder} could not be classified; left open "
                "rather than filled on a guess."
            )
            decided.add(node.id)
        elif node.id not in decided:
            node.completion_policy = policy
            decided.add(node.id)


def attach_occlusion_graph(solve: Any, graph: AtlasOcclusionGraph,
                           move_budget: Any = None) -> Any:
    """Store the graph (and optionally the move budget) on ``solve.semantics``.

    ``LatentComponent.value`` is the payload slot; ``confidence`` mirrors the
    graph's weakest node so a consumer reading only the component still sees a
    trustworthy summary. The solve is mutated in place and returned.
    """
    payload: dict[str, Any] = {"occlusion_graph": graph.to_dict()}
    if move_budget is not None:
        payload["move_budget"] = (move_budget.to_dict()
                                  if hasattr(move_budget, "to_dict") else move_budget)
    component = getattr(solve, "semantics", None)
    if component is None:      # pragma: no cover - schema guarantees the slot
        return solve
    component.value = payload
    component.confidence = min((n.confidence for n in graph.nodes), default=0.0)
    component.exportable = True
    return solve


# --------------------------------------------------------------------------
# Layer plan — the graph as a clean-plate layer manifest
# --------------------------------------------------------------------------

@dataclass(slots=True)
class LayerSpec:
    """One projection layer derived from the graph.

    ``role`` is the whole point of the split, and it follows the standing
    cleanplate doctrine (DESIGN_RULES 2026-07-19):

    ``foreground``
        An occluder. Matted from the ORIGINAL plate and projected on the
        ORIGINAL depth — it was photographed, so nothing about it is invented.
    ``background``
        Something an occluder hides part of. Needs a clean plate (the occluder
        painted out) and, critically, its own depth solve on that clean plate
        rather than a far-band extension of the original — extending the band
        put the support footprint at the cutoff and produced a vertical cliff
        with floating foreground during orbit.

    ``concepts`` is the SAM3 prompt for this layer, which is why the VLM's job
    is naming rather than masking: it supplies the words, SAM3 supplies pixels.
    """

    node_id: str
    role: str                      # foreground | background
    order: int                     # 0 = nearest; projection priority is FARTHEST-first
    concepts: str = ""
    exposes: list[str] = field(default_factory=list)
    hidden_by: list[str] = field(default_factory=list)
    depth_range_m: tuple[float, float] | None = None
    needs_clean_plate: bool = False
    needs_own_depth_solve: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "node_id": self.node_id, "role": self.role, "order": int(self.order),
            "concepts": self.concepts,
            "needs_clean_plate": bool(self.needs_clean_plate),
            "needs_own_depth_solve": bool(self.needs_own_depth_solve),
        }
        if self.exposes:
            out["exposes"] = list(self.exposes)
        if self.hidden_by:
            out["hidden_by"] = list(self.hidden_by)
        if self.depth_range_m:
            out["depth_range_m"] = [round(float(v), 4) for v in self.depth_range_m]
        return out


def layer_plan(graph: AtlasOcclusionGraph, *,
               include_unoccluded: bool = False) -> list[LayerSpec]:
    """Turn the graph into an ordered clean-plate layer manifest.

    This is what makes per-segment layering possible instead of whole-scene
    hole filling. Patching a hole in one surface makes the patch abut measured
    depth and tear along the seam; splitting into layers builds a background
    surface that CONTINUES underneath the occluder, so the two overlap and
    there is no seam to blend.

    Ordering is near-to-far by the occlusion relation itself — an occluder is
    always in front of what it occludes — with depth as the tie-break. Note
    that band priorities elsewhere in Atlas are FARTHEST-highest, so a consumer
    assigning priorities reverses this list rather than using it directly.

    KNOWN LIMITATION (found live on a real plate, 2026-07-25): the role is
    binary, but a node can be BOTH. In an A-hides-B-hides-C chain, B is
    reported ``foreground`` because it occludes C, and so gets no clean plate —
    yet A conceals part of B, which a plate would have to supply. Mid-chain
    occluders therefore under-report their plate need. Fixing it means a layer
    can carry both a matte and a plate, which is a change to the layer model
    rather than to this ordering.

    Nodes the graph declined to license (``completion_policy == none``) still
    appear, with ``needs_clean_plate`` False: an unclassifiable tear must not
    quietly acquire a generated plate. ``include_unoccluded`` adds surfaces
    nothing hides, which need no plate but may still be wanted as layers.
    """
    occludes: dict[str, list[str]] = {}
    hidden_by: dict[str, list[str]] = {}
    for edge in graph.edges:
        occludes.setdefault(edge.occluder, []).append(edge.occludee)
        hidden_by.setdefault(edge.occludee, []).append(edge.occluder)

    def _depth_key(node: OcclusionNode) -> float:
        rng = node.depth_range_m
        return float(rng[0]) if rng else float("inf")

    involved = set(occludes) | set(hidden_by)
    nodes = [n for n in graph.nodes
             if n.kind != "backdrop"
             and (include_unoccluded or n.id in involved)]

    # Occluders first, then by distance. `occludes` membership is the primary
    # key because the relation is direct evidence of ordering, whereas a fitted
    # node's depth range is a summary that can straddle another node's.
    nodes.sort(key=lambda n: (0 if n.id in occludes else 1, _depth_key(n)))

    plan: list[LayerSpec] = []
    for order, node in enumerate(nodes):
        is_occluder = node.id in occludes
        licensed = node.completion_policy != POLICY_NONE
        needs_plate = (not is_occluder) and node.id in hidden_by and licensed
        plan.append(LayerSpec(
            node_id=node.id,
            role="foreground" if is_occluder else "background",
            order=order,
            concepts=_concepts_for(node),
            exposes=list(occludes.get(node.id, [])),
            hidden_by=list(hidden_by.get(node.id, [])),
            depth_range_m=node.depth_range_m,
            needs_clean_plate=needs_plate,
            # The clean plate is a different image, so its depth must be solved
            # on that image — never inherited from the original.
            needs_own_depth_solve=needs_plate,
        ))
    return plan


def _concepts_for(node: OcclusionNode) -> str:
    """SAM3 prompt text for a layer.

    Derived from the node id, which is the fitter's own name — a placeholder
    the VLM pass is meant to replace with what the thing actually is. SAM3
    segments concepts, and "projection_box_01" is not a concept, so a layer
    left with this default will segment poorly and should be treated as
    unnamed rather than as a working prompt.
    """
    stem = node.id.replace("projection_", "").rsplit("_", 1)[0]
    return stem.replace("_", " ").strip()
