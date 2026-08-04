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

from atlas_camera.comfy.node_helpers import (  # noqa: F401
    _DEPTH_MODEL_CHOICES,
    _image_tensor_to_pil,
)
from atlas_camera.core.camera_spec import CameraSpec


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


def _mask_array(mask, shape):
    """A ComfyUI MASK as a plain (H, W) bool array, or None if unusable."""
    if mask is None:
        return None
    import numpy as np
    arr = np.asarray(mask, dtype=np.float64)
    arr = arr[0] if arr.ndim == 3 else arr
    return (arr > 0.5) if arr.shape == shape else None


def _relief_tear_mask(solve, shape):
    """Where the relief mesh has no triangle, from the recovered camera.

    This is the tear region — literally "where 📽 Project shows black" — and it
    is the only reliable source of it. A monocular depth model returns valid
    depth almost everywhere, so a tear never appears as invalid depth; it is
    created downstream by the mesh's silhouette edge test.

    Computed by rasterizing rather than read off ``ReliefMesh.hole_mask``,
    because that field does NOT survive the solve round-trip:
    ``_layers.mesh_from_primitive`` rebuilds a mesh from vertices/faces/uvs
    only. That is the right call for the serialized form — the mask is one bool
    per source pixel, 8.3M of them at 4K — but it means reading the attribute
    back yields None, which quietly evaluates to "no tears anywhere" and makes
    this node do nothing. One raster from the source camera reconstructs it
    exactly, at whatever resolution the caller actually needs.
    """
    import numpy as np

    from atlas_camera.core.move_budget import rasterize_coverage
    from atlas_camera.core.primitive_mesh import collect_scene_triangles

    verts, faces, _ = collect_scene_triangles(
        solve, include_primitives=False)          # the relief mesh alone
    if len(faces) == 0:
        return None
    intr = solve.camera.intrinsics
    height, width = shape
    # for_image: the raster is `shape`, whatever the caller asked for — the
    # docstring's "at whatever resolution the caller actually needs" — so the
    # centre fallback is this raster's, not the recorded plate's.
    spec = CameraSpec.for_image(intr, width, height)
    coverage, _ = rasterize_coverage(
        verts, faces,
        view_matrix=np.asarray(solve.camera.extrinsics.camera_view_matrix,
                               dtype=np.float64),
        fx=float(intr.fx_px or width), fy=float(intr.fy_px or width),
        cx=spec.cx, cy=spec.cy,
        width=width, height=height,
    )
    return ~coverage


def _sky_from_depth(solve, depth):
    """Sky heuristic fallback, so an unconnected sky_mask does not mean
    'fill the sky with wall geometry'."""
    try:
        from atlas_camera.core.depth_geometry import detect_sky_mask
    except Exception:
        return None
    horizon = None
    line = getattr(solve, "horizon_line", None)
    if line is not None and getattr(line, "endpoints_px", None):
        p1, p2 = line.endpoints_px
        horizon = 0.5 * (float(p1[1]) + float(p2[1]))
    if horizon is None:
        horizon = depth.shape[0] * 0.5
    try:
        return detect_sky_mask(depth, horizon_y=horizon)
    except Exception:
        return None


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
    CATEGORY = "Atlas"

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
    CATEGORY = "Atlas"

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
    CATEGORY = "Atlas/advanced"

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
                "use_relief_tears": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Treat the relief mesh's own tears as holes to fill. "
                               "This is what makes the node do anything on real data: "
                               "a monocular depth model returns VALID depth everywhere "
                               "(sky included), so nothing reads as a hole — the tears "
                               "live in the mesh, created by the silhouette edge test, "
                               "not in the depth map."}),
                "sky_mask": ("MASK", {
                    "tooltip": "Region to leave torn. Sky is part of the relief mesh's "
                               "hole mask but must NOT be filled with surface geometry "
                               "— the backdrop covers it. Falls back to the depth-based "
                               "sky heuristic when not connected."}),
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

    def complete(self, solve, depth, holes=None, use_relief_tears=True,
                 sky_mask=None, use_diffusion=True, use_normals=True,
                 diffusion_iterations=64):
        import copy

        import numpy as np

        from atlas_camera.core.depth_completion import complete_depth_from_graph
        from atlas_camera.core.occlusion_graph import AtlasOcclusionGraph

        raw = _depth_array(depth)
        notes: list[str] = []
        hole_mask = _mask_array(holes, raw.shape)

        if use_relief_tears:
            tears = _relief_tear_mask(solve, raw.shape)
            if tears is None:
                notes.append(
                    "use_relief_tears is on but this solve carries no relief mesh — "
                    "run AtlasDeriveReliefMesh first, or the only holes filled are "
                    "wherever depth is already invalid."
                )
            else:
                sky = _mask_array(sky_mask, raw.shape)
                if sky is None:
                    sky = _sky_from_depth(solve, raw)
                if sky is not None:
                    kept = tears & ~sky
                    notes.append(
                        f"relief tears {int(tears.sum())} px, of which "
                        f"{int((tears & sky).sum())} px are sky and left torn "
                        "(the backdrop covers those, geometry must not)."
                    )
                    tears = kept
                hole_mask = tears if hole_mask is None else (hole_mask | tears)

        semantics = getattr(getattr(solve, "semantics", None), "value", None) or {}
        graph = AtlasOcclusionGraph.from_dict(semantics.get("occlusion_graph"))

        result = complete_depth_from_graph(
            solve, raw, graph,
            holes=hole_mask,
            normals=_normals_array(depth) if use_normals else None,
            use_diffusion=bool(use_diffusion),
            diffusion_iterations=int(diffusion_iterations),
        )
        result.notes.extend(notes)
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


class AtlasLayerPlan:
    """Turn the occlusion graph into a clean-plate layer manifest 🥞.

    Says which segments are occluders, which surfaces they hide, and therefore
    what needs a clean plate — the split that lets a scene be rebuilt as
    overlapping layers instead of one surface with holes patched into it.
    That distinction is the whole point: a patch abuts measured depth and tears
    along its seam, whereas a background layer continues UNDERNEATH the
    occluder, so the two overlap and there is no seam.

    Two roles, following the standing cleanplate doctrine:

    * ``foreground`` — an occluder. Matte it from the ORIGINAL plate and
      project it on the ORIGINAL depth. It was photographed; nothing is
      invented.
    * ``background`` — something an occluder hides. Needs a clean plate with
      the occluder painted out, and **its own depth solve on that plate**.
      Never extend the original's far band to cover it: that puts the support
      footprint at the cutoff and produces a vertical cliff with floating
      foreground under orbit.

    The two ``concepts`` outputs are SAM3 prompts, ready to wire into
    ``AtlasSAM3Mask``. They are derived from fitter ids as placeholders, so
    they segment poorly until a VLM pass replaces them with what the things
    actually are — the division of labour being that the model supplies words
    and SAM3 supplies pixels.

    Needs ``AtlasOcclusionGraph`` upstream. A tear the graph declined to
    classify produces NO clean plate, because generating content for a region
    Atlas just said it could not reason about is exactly the failure the graph
    exists to prevent.
    """

    RETURN_TYPES = ("ATLAS_SOLVE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("solve", "report", "foreground_concepts",
                    "background_concepts")
    FUNCTION = "plan"
    CATEGORY = "Atlas"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
            },
            "optional": {
                "include_unoccluded": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also list surfaces nothing hides. They need no clean "
                               "plate, but you may still want them as their own layers."}),
                # The `*_override` STRING-link pattern (ComfyUI rejects
                # STRING->combo links; STRING->STRING sockets are fine). Wire
                # AtlasAssessImage's sam_prompt_fg / sam_prompt_bg here — the
                # VLM supplies words, SAM3 supplies pixels. Without them the
                # concept outputs are fitter-id placeholders ("box"), which
                # collapse distinct objects into one useless prompt.
                "foreground_concepts_override": ("STRING", {
                    "forceInput": True,
                    "tooltip": "VLM-named foreground concepts (e.g. AtlasAssessImage's "
                               "sam_prompt_fg). Replaces the placeholder foreground "
                               "concepts derived from fitter ids."}),
                "background_concepts_override": ("STRING", {
                    "forceInput": True,
                    "tooltip": "VLM-named background concepts (e.g. AtlasAssessImage's "
                               "sam_prompt_bg). Replaces the placeholder background "
                               "concepts derived from fitter ids."}),
            },
        }

    def plan(self, solve, include_unoccluded=False,
             foreground_concepts_override="", background_concepts_override=""):
        import copy

        from atlas_camera.core.occlusion_graph import (
            AtlasOcclusionGraph, layer_plan,
        )

        out = copy.deepcopy(solve)
        semantics = getattr(getattr(out, "semantics", None), "value", None) or {}
        graph = AtlasOcclusionGraph.from_dict(semantics.get("occlusion_graph"))
        if not graph.nodes:
            return (out, "No occlusion graph on this solve — run AtlasOcclusionGraph "
                         "first; there is nothing to split into layers.", "", "")

        specs = layer_plan(graph, include_unoccluded=bool(include_unoccluded))

        fg_override = (foreground_concepts_override or "").strip()
        bg_override = (background_concepts_override or "").strip()
        component = getattr(out, "semantics", None)
        if component is not None:
            payload = dict(component.value or {})
            payload["layer_plan"] = [s.to_dict() for s in specs]
            # Record where the prompts came from — a placeholder that segments
            # poorly and a VLM name that segments well must be tellable apart
            # after the fact.
            payload["layer_plan_concepts"] = {
                "foreground": fg_override or None,
                "background": bg_override or None,
                "source": ("vlm_override" if (fg_override or bg_override)
                           else "fitter_id_placeholder"),
            }
            component.value = payload
            component.exportable = True

        fg = [s for s in specs if s.role == "foreground"]
        bg = [s for s in specs if s.role == "background"]
        lines = [f"Layer plan: {len(fg)} foreground, {len(bg)} background "
                 f"(near to far; projection priority is FARTHEST-first, so "
                 f"reverse this when assigning it)"]
        for s in specs:
            bits = [f"  {s.order}. {s.node_id:<28} {s.role:<10}"]
            if s.needs_clean_plate:
                bits.append("clean plate + OWN depth solve")
            elif s.role == "background":
                bits.append("no plate — graph licensed nothing here")
            if s.exposes:
                bits.append(f"hides {', '.join(s.exposes)}")
            lines.append("  ".join(bits))
        if not any(s.needs_clean_plate for s in bg):
            lines.append("  note: nothing needs a clean plate — either no tear was "
                         "classified, or no occluder hides a fitted surface.")
        fg_out = fg_override or ", ".join(
            dict.fromkeys(s.concepts for s in fg if s.concepts))
        bg_out = bg_override or ", ".join(
            dict.fromkeys(s.concepts for s in bg if s.concepts))
        if fg_override or bg_override:
            lines.append("  concepts: VLM-named (override connected)")
        elif fg_out or bg_out:
            lines.append("  concepts: fitter-id placeholders — connect "
                         "AtlasAssessImage's sam_prompt outputs for names SAM3 "
                         "can actually segment")
        return (out, "\n".join(lines), fg_out, bg_out)


class AtlasShootList:
    """📸 Turn the occlusion graph into a SHOOTING BRIEF — go photograph the holes.

    Every hole Atlas fills today is invented: inpainting, LaRI's predicted hidden
    geometry, edge-dilated smear. But a matte painter with a camera can often just
    go and photograph the missing material — the paving, the brick, the render,
    the tarmac. Not the specific occluded scene, which may be a continent away,
    but the same KIND of surface at the same angle and the same light.

    Atlas already knows exactly what is missing. `AtlasOcclusionGraph` enumerates
    one node per fitted surface with its plane and depth range, one edge per
    silhouette tear with its pixel count, and `AtlasLayerPlan` says which of those
    need a clean plate. What has never existed is a way to state that in terms a
    photographer can act on.

    This writes a project file answering, per hole:

        what to photograph   the semantic concept, from the layer plan
        at what angle        incidence to the surface, 0 = square, 90 = edge-on
        from what distance   the surface's own depth range
        at what resolution   pixels per metre, so the capture is not too soft
        how badly            torn pixel count, worst first

    THE ANGLE IS THE LOAD-BEARING NUMBER, and it depends on RANGE: the same floor
    is steep underfoot and almost edge-on at the horizon, so "the angle to the
    ground" is meaningless without a distance. A slab shot square will not sit
    into a plate that sees it at 85 degrees, however good the texture.

    LIGHTING IS DELIBERATELY NOT NUMBERED. Sun direction and hardness are what
    separate a patch that sits from one that reads as a sticker, and Atlas cannot
    measure them from a single plate. Rather than emit a confident-looking azimuth
    that is really a guess, the project ships the PLATE ITSELF as a reference to
    match by eye — and says so in the payload, so a client cannot mistake missing
    lighting fields for "lighting does not matter". Flat or overcast capture is
    the safest, because Atlas can relight it from normals it already has.

    Surfaces with no fitted plane — an alleyway, a doorway, a recess — are flagged
    `volumetric`. They are not textures and no angle describes them; those need
    alignment against ghosted geometry rather than a flat shot.

    Run `AtlasOcclusionGraph` (and ideally `AtlasLayerPlan`) upstream, or there is
    nothing to enumerate.
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "STRING", "STRING")
    RETURN_NAMES = ("solve", "project_path", "report")
    FUNCTION = "build"
    CATEGORY = "Atlas"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE", {"tooltip":
                    "Must carry an occlusion graph — run AtlasOcclusionGraph first."}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip":
                    "The plate. Written alongside the project as the lighting and "
                    "material reference to match by eye."}),
                "output_dir": ("STRING", {"default": "",
                    "tooltip": "Where to write the project. Blank = ComfyUI's output "
                               "directory under atlas_shoot/."}),
                "project_name": ("STRING", {"default": "shoot"}),
                "min_tear_px": ("INT", {"default": 400, "min": 0, "max": 1000000, "step": 100,
                    "tooltip": "Holes smaller than this are not worth a trip with a "
                               "camera and are left to the inpainter."}),
            },
        }

    def build(self, solve, reference_image=None, output_dir="", project_name="shoot",
              min_tear_px=400):
        import json as _json
        import os

        from atlas_camera.core.occlusion_graph import AtlasOcclusionGraph
        from atlas_camera.core.shoot_list import build_shoot_list, shoot_project

        semantics = getattr(getattr(solve, "semantics", None), "value", None) or {}
        raw_graph = semantics.get("occlusion_graph")
        if not raw_graph:
            raise ValueError(
                "AtlasShootList: this solve carries no occlusion graph, so there is "
                "nothing to enumerate. Run AtlasOcclusionGraph upstream (and "
                "AtlasLayerPlan, which supplies the subject names).")
        graph = AtlasOcclusionGraph.from_dict(raw_graph)

        # The layer plan is optional but does real work: it names the surfaces
        # ("pavement", "brick wall") and vetoes anything already handled another
        # way. Without it every occluded node is offered under its bare kind.
        plan_raw = semantics.get("layer_plan") or []
        plans = []
        for p in plan_raw:
            if isinstance(p, dict):
                plans.append(type("_P", (), {
                    "node_id": p.get("node_id", ""),
                    "concepts": p.get("concepts") or [],
                    "needs_clean_plate": bool(p.get("needs_clean_plate", True)),
                })())

        specs = build_shoot_list(solve, graph, layer_plan=plans or None,
                                 min_tear_px=int(min_tear_px))

        out_root = str(output_dir or "").strip()
        if not out_root:
            try:
                import folder_paths
                out_root = os.path.join(folder_paths.get_output_directory(), "atlas_shoot")
            except Exception:  # noqa: BLE001 - outside ComfyUI
                out_root = os.path.join(os.getcwd(), "atlas_shoot")
        target = os.path.join(out_root, str(project_name or "shoot"))
        os.makedirs(target, exist_ok=True)

        plate_size = None
        if reference_image is not None:
            plate_size = (int(reference_image.shape[2]), int(reference_image.shape[1]))
            try:
                pil = _image_tensor_to_pil(reference_image)
                pil.save(os.path.join(target, "reference_plate.png"))
            except Exception as exc:  # noqa: BLE001 - a missing reference must not
                specs and specs[0].warnings.append(   # kill an otherwise valid brief
                    f"reference plate could not be written ({exc})")

        project = shoot_project(specs, plate_size=plate_size)
        project_path = os.path.join(target, "atlas_shoot.json")
        with open(project_path, "w", encoding="utf-8") as fh:
            _json.dump(project, fh, indent=2)

        lines = [f"AtlasShootList: {len(specs)} shot(s) -> {project_path}"]
        if not specs:
            lines.append(
                "  nothing worth photographing — every occluded surface is below "
                f"min_tear_px ({int(min_tear_px)}), or the layer plan says they are "
                "handled elsewhere.")
        for s in specs:
            lines.append(f"  {s.priority}. {s.guidance}")
            lines.append(f"     {s.tear_px} px hidden by "
                         f"{', '.join(s.hidden_by) or 'unknown'}; "
                         f"{s.distance_m:.1f} m away")
            for w in s.warnings:
                lines.append(f"     NOTE {w}")
        if specs:
            lines.append("")
            lines.append("  Lighting is NOT specified: Atlas cannot measure sun "
                         "direction or hardness from one plate. Match "
                         "reference_plate.png by eye — flat or overcast light is "
                         "safest, since Atlas can relight from its own normals.")
        return (solve, project_path, "\n".join(lines))
