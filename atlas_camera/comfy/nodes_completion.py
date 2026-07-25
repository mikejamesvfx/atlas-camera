"""ComfyUI nodes for the unseen-geometry track.

Three nodes, in the order an artist uses them:

``AtlasOcclusionGraph`` 🕸
    Decomposes the solved scene into surfaces/objects and the tears between
    them, and states what completion each region licenses. Analysis only —
    it attaches metadata and never touches geometry.

``AtlasMoveBudget`` 📐
    How far the camera can move before a tear opens. Reports a per-DOF
    envelope, and optionally checks an authored camera path frame by frame.

``AtlasCompleteDepth`` 🩹
    Fills the depth map's holes BEFORE the relief mesh is built, so no tear
    exists to repair afterwards. Every invented pixel is marked.

The ordering matters and is not arbitrary: the graph decides what may be
built, the budget says how much needs building, and completion does only
that. Running completion without a graph is supported but deliberately
weaker — it has no fitted planes to be exact against, so it falls back to
diffusion and says so in its report.

Per the layering rule this module may import anything; nothing outside
``comfy/`` imports it.
"""

from __future__ import annotations

from typing import Any

from atlas_camera.comfy.node_helpers import _DEPTH_MODEL_CHOICES  # noqa: F401


def _depth_array(depth: Any) -> Any:
    """The raw HxW forward-distance array out of an ATLAS_DEPTH_MAP."""
    import numpy as np
    arr = getattr(depth, "depth", depth)
    return np.asarray(arr, dtype=np.float64)


def _normals_array(depth: Any) -> Any:
    """Predicted normals when the depth model supplied them, else None.

    These are in the MODEL's camera frame, so they are only used for the
    tangential tier, which is itself local and first-order — a frame
    misalignment there degrades an already-approximate tier rather than
    corrupting the exact ray-plane one.
    """
    import numpy as np
    nrm = getattr(depth, "normals", None)
    if nrm is None:
        return None
    nrm = np.asarray(nrm, dtype=np.float64)
    return nrm if nrm.ndim == 3 and nrm.shape[-1] == 3 else None


class AtlasOcclusionGraph:
    """Decompose the scene into what occludes what, and what may be built there.

    Reads the proxy primitives already on the solve plus the shared depth map,
    and produces the occlusion graph: one node per fitted surface/object with a
    single permitted ``completion_policy``, and one edge per silhouette tear
    (the nearer side occludes the farther one).

    Nothing is built here and no measurement changes — the graph is attached to
    the solve's ``semantics`` slot so downstream nodes and the exported solve
    JSON carry it. A tear that cannot be classified licenses NOTHING, which is
    the point: an honest hole beats invented geometry in a solve that claims to
    be measured.

    Run ``AtlasDeriveProjectionGeometry`` first — without fitted primitives
    there is nothing to decompose, and the report says so rather than returning
    a confident-looking empty graph.
    """

    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "build"
    CATEGORY = "Atlas Camera/Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
            },
            "optional": {
                "depth": ("ATLAS_DEPTH_MAP", {
                    "tooltip": "The shared depth map. Without it the graph still lists "
                               "the scene's parts but cannot analyse tears, and licenses "
                               "no tear-driven completion."}),
            },
        }

    def build(self, solve, depth=None):
        import copy

        from atlas_camera.core.occlusion_graph import (
            attach_occlusion_graph, build_occlusion_graph,
        )

        out = copy.deepcopy(solve)
        graph = build_occlusion_graph(
            out, depth=_depth_array(depth) if depth is not None else None)
        attach_occlusion_graph(out, graph)
        return (out, graph.describe())


class AtlasMoveBudget:
    """How far can this camera move before a tear opens?

    Rasterizes the scene from candidate cameras and measures where the
    photographed surface has been torn away — sealed-minus-covered, so the
    answer is unaffected by the backdrop and grows as tears get filled.

    Reports a per-DOF envelope (dolly x/y/z, pan, tilt). Connect a camera path
    to additionally get a per-frame verdict on the move you actually authored,
    which is usually the question worth asking.

    Two results that look wrong but are correct: pan and tilt are often
    unbounded, because rotation about the optical centre produces no parallax
    and so cannot open a tear; and an axis is unbounded when nothing is hidden
    along it. ``saturated`` names any axis that hit the search cap.

    Needs a relief mesh (``AtlasDeriveReliefMesh`` or ``AtlasInput``). Runs on
    the GPU via torch; the numpy path is a reference implementation and is
    roughly 60x slower, so leave the backend on auto.
    """

    RETURN_TYPES = ("ATLAS_SOLVE", "STRING")
    RETURN_NAMES = ("solve", "report")
    FUNCTION = "measure"
    CATEGORY = "Atlas Camera/Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
            },
            "optional": {
                "camera_path": ("ATLAS_CAMERA_PATH", {
                    "tooltip": "Optional authored move. When connected, every frame is "
                               "checked and the report names the worst one."}),
                "threshold": ("FLOAT", {
                    "default": 0.02, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Fraction of frame allowed to tear open before a move "
                               "counts as unsafe. 0.02 = 2% of pixels."}),
                "bisect_steps": ("INT", {
                    "default": 8, "min": 2, "max": 16,
                    "tooltip": "Precision of the envelope search. Each step halves the "
                               "bracket; 8 is ample. Higher costs proportionally more."}),
                "max_dolly_m": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1,
                    "tooltip": "Search cap for translation, in metres. 0 = auto (a "
                               "quarter of the median scene distance)."}),
                "max_angle_deg": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Search cap for pan/tilt, in degrees."}),
                "backend": (["auto", "torch", "numpy"], {
                    "default": "auto",
                    "tooltip": "auto picks torch (cuda/mps/cpu). numpy is the reference "
                               "implementation — correct but far slower."}),
            },
        }

    def measure(self, solve, camera_path=None, threshold=0.02, bisect_steps=8,
                max_dolly_m=0.0, max_angle_deg=30.0, backend="auto"):
        import copy

        from atlas_camera.core.move_budget import estimate_move_budget

        out = copy.deepcopy(solve)
        try:
            budget = estimate_move_budget(
                out,
                threshold=float(threshold),
                backend=backend,
                bisect_steps=int(bisect_steps),
                max_dolly_m=(float(max_dolly_m) if max_dolly_m > 0 else None),
                max_angle_deg=float(max_angle_deg),
                camera_path=camera_path,
            )
        except ValueError as exc:
            # A missing relief mesh is an ordinary authoring mistake, not a
            # crash — say what to connect and pass the solve through unchanged.
            return (out, f"Move budget not computed: {exc}")

        component = getattr(out, "semantics", None)
        if component is not None:
            payload = dict(component.value or {})
            payload["move_budget"] = budget.to_dict()
            component.value = payload
            component.exportable = True
        return (out, budget.describe())


class AtlasCompleteDepth:
    """Fill the depth map's holes before the mesh is built.

    A tear is a hole in the DEPTH MAP long before it is a hole in geometry.
    Repairing it here means the relief mesh is built from a complete surface
    and never tears in the first place — as opposed to reconstructing, later,
    information that was discarded a step earlier.

    Three tiers, best first, and each pixel records which produced it:

    * **ray-plane** — exact intersection of the pixel's own ray with a plane
      the occlusion graph already fitted. Not an interpolation.
    * **tangent** — first-order continuation using predicted normals, where
      the depth model supplied them.
    * **diffusion** — smooth interpolation. Plausible-looking and evidence-free,
      which is why it is last and labelled.

    Connect ``graph`` to get the exact tier; the graph also carries the
    refusals, so a tear it could not classify contributes no plane and stays
    open. Without a graph this degrades to diffusion and says so.

    ``hidden_mask`` output marks every synthesized pixel, for QA and for
    compositing decisions downstream.
    """

    RETURN_TYPES = ("ATLAS_DEPTH_MAP", "MASK", "STRING")
    RETURN_NAMES = ("depth", "hidden_mask", "report")
    FUNCTION = "complete"
    CATEGORY = "Atlas Camera/Geometry"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "depth": ("ATLAS_DEPTH_MAP",),
            },
            "optional": {
                "holes": ("MASK", {
                    "tooltip": "Extra regions to treat as holes, on top of wherever "
                               "depth is already invalid. Use to remove an occluder "
                               "deliberately."}),
                "use_diffusion": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Allow the last-resort smooth fill. Turn OFF to leave "
                               "anything the exact tiers could not justify as an honest "
                               "hole — useful for seeing what is actually supported."}),
                "use_normals": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use the depth model's predicted normals (MoGe *-normal "
                               "variants) for first-order tangential extension. Ignored "
                               "when the model supplied none."}),
                "diffusion_iterations": ("INT", {
                    "default": 64, "min": 1, "max": 512,
                    "tooltip": "How far the diffusion front travels. Larger reaches "
                               "deeper holes; regions it never reaches stay holes."}),
            },
        }

    def complete(self, solve, depth, holes=None, use_diffusion=True,
                 use_normals=True, diffusion_iterations=64):
        import copy

        import numpy as np

        from atlas_camera.core.depth_completion import complete_depth_from_graph
        from atlas_camera.core.occlusion_graph import AtlasOcclusionGraph

        raw = _depth_array(depth)
        hole_mask = None
        if holes is not None:
            hm = np.asarray(holes, dtype=np.float64)
            hm = hm[0] if hm.ndim == 3 else hm
            if hm.shape == raw.shape:
                hole_mask = hm > 0.5

        semantics = getattr(getattr(solve, "semantics", None), "value", None) or {}
        graph = AtlasOcclusionGraph.from_dict(semantics.get("occlusion_graph"))

        result = complete_depth_from_graph(
            solve, raw, graph,
            holes=hole_mask,
            normals=_normals_array(depth) if use_normals else None,
            use_diffusion=bool(use_diffusion),
            diffusion_iterations=int(diffusion_iterations),
        )
        if not graph.nodes:
            result.notes.append(
                "no occlusion graph on this solve — run AtlasOcclusionGraph first "
                "for exact ray-plane fills instead of diffusion guesses."
            )

        out = copy.deepcopy(depth)
        out.depth = result.depth.astype(np.float32)
        meta = dict(getattr(out, "metadata", None) or {})
        meta["depth_completion"] = {
            "synthesized_fraction": result.synthesized_fraction,
            "confidence": result.confidence(),
            "methods": result.method_histogram(),
            **result.stats,
        }
        out.metadata = meta

        import torch
        mask = torch.from_numpy(
            result.synthesized_mask.astype(np.float32)).unsqueeze(0)
        return (out, mask, result.describe())
