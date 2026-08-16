"""Load an external hidden-geometry VOLUME into an Atlas solve (EXPERIMENTAL).

Research bridge for the 2026-08-15 VolFill evaluation
(``docs/research/FLASH3D_VOLFILL_ATLAS_EVALUATION.md``). A volumetric amodal
predictor (VolFill: a 256^3 truncated UNSIGNED distance field) is turned into
Atlas geometry and APPENDED to the solve as ``PROXY_ROLE`` mesh primitives, so
the Blockout Viewport can render it beside the plate's own geometry.

EXTRACTION: ``raymarch`` (default) or ``marching_cubes`` (legacy).

The field is UNSIGNED, so its level set is a SHELL offset +/-threshold either
side of the true surface -- a front wall and a back wall about 2*threshold
apart. Measured: 78.3% of rays cross exactly twice, 98.6% cross an EVEN number,
band thickness exactly 2.0 voxels. Meshing that shell with marching cubes
therefore doubles the geometry, leaves no consistent orientation (half the
triangles face away, which reads as flipped normals), and misplaces the surface
by ~1 voxel.

``raymarch`` marches the field along the RECOVERED camera's rays and pairs each
entry/exit into its MIDPOINT -- one sample per real surface, at the right depth,
ordered front-to-back. That IS Atlas's layered-ray representation, so
``core.hidden_geometry.select_hidden_surface`` consumes it directly, and each
layer is meshed with ``build_relief_mesh`` into oriented, single-sided geometry
with projective UVs. ``double_sided`` is then unnecessary and IGNORED.

APPENDS, never clobbers — this is not a Derive node. Per the design rule, only
Derive nodes replace prior PROXY_ROLE geometry; anything additive appends and
``AtlasMergeGeometry`` stays the one explicit combiner.

WHY A SEPARATE NODE, AND WHY EXPERIMENTAL
Atlas already owns the hidden-geometry consumer path
(``core/hidden_geometry.py``), but that path eats LAYERED RAYS — a per-pixel
front-to-back depth stack — while this eats a VOLUME. The ``raymarch``
extraction above IS that adapter, and it is measured, so that is no longer the
reason for the gate.

What keeps it experimental is the INPUT: it reads a truncated distance field
produced by a research predictor, not by anything the pack ships or installs,
and the divergence gate exists because the prediction can rewrite the plate
rather than complete it. Promote it when a producer ships and the gate stops
firing on real plates — not merely because the adapter works.

The volume directory is whatever ``research/volfill/run_volfill.py`` wrote
(that tree is in the git checkout only — ``.comfyignore`` keeps the research
harness out of the published pack, so a Registry install supplies the directory
some other way):
``pred_tudf_*.npz`` (keys ``tudf``, ``visible_tudf``) plus ``metadata.json``
(``bbox_min``, ``extent_xyz``). Nothing about the format is VolFill-specific
beyond those key names.

RECOMMENDED CHAIN::

    AtlasLoadHiddenVolume(extraction="raymarch")
        -> AtlasRetopologizeLayer(method="decimate"|"smooth")
        -> AtlasBlockoutViewport

``max_faces`` applies to the marching-cubes path only, where it sub-samples the
FIELD (never the face list -- striding faces PERFORATES a mesh rather than
decimating it). Field sub-sampling permanently destroys sub-voxel detail, so
prefer ``max_faces=0`` and let the retopo node's quadric decimation reduce the
full-resolution surface: measured on sh001, field-stride 2 gives 47,957 verts
off a 12.1 cm grid while full-res reduced to a comparable 59,657 verts retains
the 6.1 cm surface.

PROVENANCE, and why BOTH numbers are reported. ``invented_fraction`` -- the
quantity the divergence gate acts on -- asks whether a point lies within the
visible surface's own truncation band anywhere in 3D. Its thresholds were
validated on 26 held-out volumes against exactly that measure, so both
extraction modes compute it identically; changing the definition would silently
invalidate the calibration. The ray path additionally reports ``ray_agreement``:
whether the prediction lands where the visible surface is along the SAME ray.
That is stricter and more physically meaningful (measured 23 cm median on a
5.7 cm voxel, ~4 voxels), and it is REPORTED, not gated on, until it has its own
held-out calibration.

PROVENANCE IS THE POINT. Every emitted vertex is tagged visible-supported or
INVENTED, and ``show`` can isolate the invented half. A hidden-geometry
hypothesis that cannot be told apart from measured geometry is a liability, so
the split is carried on the primitive rather than left to the eye.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atlas_camera.comfy.node_helpers import _require_numpy
from atlas_camera.core.proxy_geometry import PROXY_ROLE
from atlas_camera.core.schema import AtlasProxyPrimitive

HIDDEN_VOLUME_SOURCE = "hidden_volume"

#: MoGe/OpenCV camera space (x right, y DOWN, +z FORWARD) -> Atlas camera space
#: (x right, y UP, -z forward, per core/relief_mesh.py:182-184). det = +1, so a
#: rotation rather than a mirror — handedness is preserved.
_MOGE_TO_ATLAS = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))


def _load_volume(path: Path):
    np = _require_numpy()
    npzs = sorted(path.glob("pred_tudf_*.npz"))
    if not npzs:
        raise RuntimeError(
            f"No pred_tudf_*.npz in {path}. Point volume_path at a directory "
            "written by research/volfill/run_volfill.py."
        )
    with np.load(npzs[0]) as data:
        tudf = np.asarray(data["tudf"], dtype=np.float32)
        visible = (np.asarray(data["visible_tudf"], dtype=np.float32)
                   if "visible_tudf" in data else None)
        # MoGe's validity mask. Where it is FALSE the depth stage saw no
        # surface — sky, blown highlights — and any geometry the predictor puts
        # there is invented against nothing.
        sky = (~np.asarray(data["moge_mask"], dtype=bool)
               if "moge_mask" in data else None)
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        raise RuntimeError(f"No metadata.json beside {npzs[0].name} in {path}.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return tudf, visible, sky, meta


def _empty_matte():
    """A 1x1 zero MASK. ComfyUI wants a tensor on every output, even a refusal."""
    np = _require_numpy()
    try:
        import torch
        return torch.zeros((1, 1, 1), dtype=torch.float32)
    except ImportError:                       # pragma: no cover - torch-free tests
        return np.zeros((1, 1, 1), dtype=np.float32)


def _to_matte(mask):
    """(H, W) bool -> ComfyUI MASK tensor (1, H, W) float."""
    np = _require_numpy()
    arr = np.asarray(mask, dtype=np.float32)[None, ...]
    try:
        import torch
        return torch.from_numpy(np.ascontiguousarray(arr))
    except ImportError:                       # pragma: no cover
        return arr


class AtlasLoadHiddenVolume:
    """Turn an external hidden-geometry volume into Atlas geometry (EXPERIMENTAL).

    Ray-marches (or, legacy, marching-cubes) a 256^3 truncated distance field
    into the solve's world space and appends it as projection-proxy meshes. The
    canonical frame is a pure translate+uniform-scale of the metric camera frame
    (the predictor's bbox is axis-aligned and isotropic, fit to the visible
    points), so the inverse is closed form -- no registration, no hand alignment.
    """

    RETURN_TYPES = ("ATLAS_SOLVE", "STRING", "MASK")
    RETURN_NAMES = ("solve", "report", "occlusion_matte")
    FUNCTION = "load"
    CATEGORY = "Atlas Camera/Experimental"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "volume_path": ("STRING", {"default": "", "tooltip":
                    "Directory holding pred_tudf_*.npz + metadata.json, as "
                    "written by research/volfill/run_volfill.py."}),
            },
            "optional": {
                "depth": ("ATLAS_DEPTH_MAP", {"tooltip":
                    "Atlas's OWN depth for this plate. In 'combined' mode this is "
                    "the visible surface the hidden layers are selected against, "
                    "and it is the right reference — the predictor's own visible "
                    "volume is only a coarse voxelization of it. Without it the "
                    "node falls back to that voxelization."}),
                "restrict_mask": ("MASK", {"tooltip":
                    "Bounds where hidden depth may be substituted, and where "
                    "fill_gaps may diffuse. REQUIRED for fill_gaps: diffusing "
                    "across a whole frame turns a small real selection into "
                    "total replacement — measured on sh001, a 0.9% selection "
                    "became 100% substitution. Wire a foreground band's "
                    "layer_mask or a hole ROI."}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 2.9,
                    "step": 0.05, "tooltip":
                    "Surface level in VOXEL units (the field is unsigned, range "
                    "[0, truncation]). 0.5 matches VolFill's own visualizer. "
                    "Larger = a fatter, smoother shell."}),
                "depth_scale": ("FLOAT", {"default": 1.0, "min": 0.001,
                    "max": 1000.0, "step": 0.001, "tooltip":
                    "Predictor-metres -> Atlas-metres. 1.0 when the predictor is "
                    "metric and the solve is too (MoGe-v2 is metric). Use "
                    "core.hidden_geometry.register_layers_to_depth to measure it "
                    "when the solve carries its own scale."}),
                "show": (["all", "invented_only", "visible_only"],
                    {"default": "all", "tooltip":
                    "Which predicted surface to emit. 'invented_only' keeps just "
                    "the geometry with NO visible support — the diagnostic view, "
                    "because it shows exactly what the model made up."}),
                "max_faces": ("INT", {"default": 200000, "min": 0, "max": 2000000,
                    "step": 10000, "tooltip":
                    "Face budget before double-siding (0 = no limit). Honoured by "
                    "SUB-SAMPLING THE VOLUME, not by dropping faces — striding a "
                    "face list perforates the mesh instead of decimating it. The "
                    "emitted mesh carries both windings, so its final face count "
                    "is twice this."}),
                "name": ("STRING", {"default": "hidden_volume", "tooltip":
                    "Primitive name prefix. Distinct names let several volumes "
                    "coexist in one solve."}),
                # APPENDED LAST — widgets_values is positional, so a new widget
                # goes on the end or every saved graph shifts.
                "double_sided": ("BOOLEAN", {"default": True, "tooltip":
                    "MARCHING-CUBES ONLY; IGNORED by the raymarch path.\n"
                    "Emits each triangle in both windings, because an UNSIGNED "
                    "field's level set has no consistent orientation and half the "
                    "surface would face away under single-sided shading. That is "
                    "a workaround for meshing the shell; raymarch pairs the walls "
                    "instead and comes out single-sided already. Turn OFF when "
                    "chaining AtlasRetopologizeLayer on the legacy path — "
                    "remeshers choke on coincident opposite faces."}),
                "max_invented_fraction": ("FLOAT", {"default": 0.88, "min": 0.0,
                    "max": 1.0, "step": 0.01, "tooltip":
                    "DIVERGENCE GATE, refuse level. Above this share of the "
                    "emitted surface having NO visible support, the prediction "
                    "has stopped agreeing with the plate. Validated on 26 volumes "
                    "held out from the volumes that suggested it: 0.88 refuse / "
                    "0.82 inspect called every decisive case correctly (20/20). A "
                    "single 0.85 threshold managed 88.5% — the errors all sat "
                    "between them, which is why there is an inspect band. "
                    "1.0 disables."}),
                "on_divergence": (["refuse", "mark", "allow"],
                    {"default": "refuse", "tooltip":
                    "What to do when the gate trips. 'refuse' emits NOTHING and "
                    "says why — a warning in a report string is not a gate. "
                    "'mark' emits but tags the primitive divergent so the "
                    "viewport and exporters can treat it as suspect. 'allow' is "
                    "the old advisory behaviour."}),
                "extraction": (["raymarch", "marching_cubes"],
                    {"default": "raymarch", "tooltip":
                    "How the surface comes out of the volume. "
                    "'raymarch' (default, correct): march the field along the "
                    "camera's own rays and PAIR each entry/exit into its "
                    "midpoint, then mesh each layer with build_relief_mesh. "
                    "Single-sided, correctly oriented, no double-wall bias, and "
                    "the layers are Atlas's own layered-ray representation. "
                    "'marching_cubes' (legacy): meshes the level set directly. "
                    "An UNSIGNED field's level set is a SHELL offset +/-threshold "
                    "either side of the surface, so that yields double the "
                    "geometry, no consistent orientation, and a ~1 voxel "
                    "placement error. Kept only to reproduce earlier runs."}),
                "min_surface_coverage": ("FLOAT", {"default": 0.02,
                    "min": 0.0, "max": 1.0, "step": 0.005, "tooltip":
                    "EMPTY-VOLUME GATE. Minimum share of camera rays that must "
                    "hit any predicted surface. A volume with (almost) no "
                    "content scores 0% INVENTED and therefore sails through the "
                    "divergence gate looking sound — measured on the "
                    "golden_corridor plate, where the depth stage collapsed, "
                    "VolFill produced ZERO surface voxels, and the gate still "
                    "passed it. Nothing is not agreement. 0.0 disables."}),
                "reject_sky": ("BOOLEAN", {"default": True, "tooltip":
                    "Drop predicted geometry that projects into the depth's "
                    "own INVALID mask — sky, blown highlights. The predictor "
                    "fills its whole cube, including where MoGe saw nothing, "
                    "and geometry invented in the sky is spurious by "
                    "construction. Measured on sh001: removes 18.9% of the "
                    "invented surface and drops the zero-offset in-front "
                    "fraction from 3.25% to 1.06%. Atlas is sky-aware "
                    "throughout; a hidden-geometry backend must be too."}),
                "emit": (["combined", "layers"], {"default": "combined",
                    "tooltip":
                    "'combined' (default): run the marched layers through "
                    "core.hidden_geometry.select_hidden_surface -- per-pixel "
                    "FIRST-CLEARING-LAYER selection, then gap diffusion -- and "
                    "substitute the result into the visible depth, so ONE relief "
                    "surface carries photographed and inferred geometry together. "
                    "That selector is already calibrated: for a solid occluder "
                    "layer 1 is usually its own BACK face, so the choice must be "
                    "per-pixel and never a fixed layer index. "
                    "'layers': one mesh per marched layer, for diagnosis."}),
                "clear_rel": ("FLOAT", {"default": 0.15, "min": 0.01,
                    "max": 1.0, "step": 0.01, "tooltip":
                    "A hidden layer must sit at least this fraction of the "
                    "visible depth BEHIND it to count as a separate surface. "
                    "Occluder back-faces are nearer than that and get skipped."}),
                "fill_gaps": ("BOOLEAN", {"default": True, "tooltip":
                    "Diffuse the per-pixel selections into ONE coherent hidden "
                    "surface. Scattered predictions shred the relief mesh via its "
                    "world-edge check regardless of thresholds -- the same "
                    "calibrated fix the shelved AtlasPredictHiddenGeometry used."}),
                "invert_restrict_mask": ("BOOLEAN", {"default": False,
                    "tooltip":
                    "Flip restrict_mask before use. Which side you want is "
                    "NOT fixed: a band mask marks the occluder's own pixels, "
                    "but depending on the plate the recovered layers may sit "
                    "in its complement instead — measured live, the same "
                    "foreground band overlapped 60.7% of the boiler's "
                    "selection and 0% of sh001's. The report prints how much "
                    "of frame the mask covers AND how much of the cleared "
                    "selection it catches, so flip this if the second number "
                    "is near zero while the first is not."}),
                "inspect_invented_fraction": ("FLOAT", {"default": 0.82,
                    "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":
                    "Lower edge of the AMBIGUOUS band. Between this and the "
                    "refuse level the prediction is neither clearly sound nor "
                    "clearly diverged, so it is emitted and TAGGED for a human "
                    "look rather than silently passed or silently dropped. "
                    "Held-out data put every misclassification in this band."}),
            },
        }

    def load(self, solve, volume_path, depth=None, restrict_mask=None,
             threshold=0.5, depth_scale=1.0,
             show="all", max_faces=200000, name="hidden_volume",
             double_sided=True, max_invented_fraction=0.88,
             on_divergence="refuse", extraction="raymarch",
             min_surface_coverage=0.02, reject_sky=True, emit="combined",
             clear_rel=0.15, fill_gaps=True, invert_restrict_mask=False,
             inspect_invented_fraction=0.82):
        """Thin wrapper so every early return can stay a 2-tuple.

        RETURN_TYPES gained the occlusion matte; normalising here beats
        threading a third element through a dozen refusal paths, each of which
        would be a place to get it wrong.
        """
        result = self._load_impl(
            solve, volume_path, depth, restrict_mask, threshold, depth_scale,
            show, max_faces, name, double_sided, max_invented_fraction,
            on_divergence, extraction, min_surface_coverage, reject_sky,
            emit, clear_rel, fill_gaps, invert_restrict_mask,
            inspect_invented_fraction)
        if len(result) == 3:
            return result
        return (result[0], result[1], _empty_matte())

    def _load_impl(self, solve, volume_path, depth=None, restrict_mask=None,
             threshold=0.5, depth_scale=1.0,
             show="all", max_faces=200000, name="hidden_volume",
             double_sided=True, max_invented_fraction=0.88,
             on_divergence="refuse", extraction="raymarch",
             min_surface_coverage=0.02, reject_sky=True, emit="combined",
             clear_rel=0.15, fill_gaps=True, invert_restrict_mask=False,
             inspect_invented_fraction=0.82):
        np = _require_numpy()
        import copy

        out = copy.deepcopy(solve)
        path = Path(str(volume_path).strip().strip('"'))
        if not path.is_dir():
            return (out, f"AtlasLoadHiddenVolume: not a directory: {path}")

        try:
            from skimage.measure import marching_cubes
        except ImportError:
            return (out, "AtlasLoadHiddenVolume: needs scikit-image "
                         "(pip install scikit-image).")

        try:
            tudf, visible, sky, meta = _load_volume(path)
        except RuntimeError as exc:
            return (out, f"AtlasLoadHiddenVolume: {exc}")

        surf_frac = float((tudf <= float(threshold)).mean())
        if min_surface_coverage > 0 and surf_frac <= 0.0:
            return (out, (
                "AtlasLoadHiddenVolume: REFUSED - the volume is EMPTY (no voxel "
                f"below threshold {threshold}). Nothing is not agreement."))
        lo, hi = float(tudf.min()), float(tudf.max())
        if not (lo < float(threshold) < hi):
            return (out, f"AtlasLoadHiddenVolume: threshold {threshold} outside "
                         f"field range [{lo:.3f}, {hi:.3f}] — nothing to mesh.")

        bbox_min = np.asarray(meta["bbox_min"], dtype=np.float64)
        extent = np.asarray(meta["extent_xyz"], dtype=np.float64)
        res = int(tudf.shape[0])
        voxel = extent / res

        if extraction == "raymarch":
            return self._load_raymarch(
                out, path, tudf, visible, sky, meta, threshold, depth_scale,
                show, name, max_invented_fraction, on_divergence,
                inspect_invented_fraction, min_surface_coverage, reject_sky,
                emit, clear_rel, fill_gaps, depth, restrict_mask,
                invert_restrict_mask, double_sided_set=bool(double_sided))

        # DECIMATE THE VOLUME, NEVER THE FACE LIST.
        #
        # The first version honoured max_faces with `faces = faces[::step]`.
        # Striding a face list does not decimate a mesh, it PERFORATES it: at
        # 372k faces against a 150k budget it kept every third triangle and
        # deleted the rest, so the viewport showed a shredded lattice (found
        # live 2026-08-15, every lane of the review workflow).
        #
        # Marching-cubes face count scales with surface area over voxel area, so
        # sub-sampling the grid by k cuts faces by roughly k^2 while keeping the
        # surface CLOSED. Step the volume up until the budget is met.
        budget = int(max_faces) if max_faces else 0
        k = 1
        while True:
            sub = tudf[::k, ::k, ::k]
            if min(sub.shape) < 4:                    # never sub-sample to mush
                break
            lo_s, hi_s = float(sub.min()), float(sub.max())
            if not (lo_s < float(threshold) < hi_s):
                k = max(1, k - 1)
                sub = tudf[::k, ::k, ::k]
                break
            verts_zyx, faces, _, _ = marching_cubes(sub, level=float(threshold))
            if not budget or len(faces) <= budget or k >= 8:
                break
            k += 1
        vis_sub = visible[::k, ::k, ::k] if visible is not None else None
        sub_res = sub.shape[0]
        # A k-strided grid samples the SAME extent with fewer cells, so the
        # metric size of one cell grows by k.
        voxel = voxel * k
        res = sub_res

        # Provenance BEFORE any coordinate work — the visible field shares the
        # grid, so it is sampled in index space.
        if vis_sub is not None:
            ijk = np.clip(np.rint(verts_zyx).astype(int), 0, res - 1)
            supported = vis_sub[ijk[:, 0], ijk[:, 1], ijk[:, 2]] <= (threshold + 1.0)
        else:
            supported = np.zeros(len(verts_zyx), dtype=bool)

        # (z, y, x) index -> xyz metres. The on-disk field is stored (z, y, x),
        # so the columns reverse before scaling — the same reversal VolFill's own
        # visualizer does.
        pts = bbox_min + verts_zyx[:, ::-1] * voxel
        pts = pts * float(depth_scale)
        pts = pts @ np.asarray(_MOGE_TO_ATLAS, dtype=np.float64).T

        # Camera -> world via the 4x4 ONLY (never the 3x3 — transpose ambiguity).
        vm = np.asarray(out.camera.extrinsics.camera_view_matrix, dtype=np.float64)
        c2w = np.linalg.inv(vm)
        pts = pts @ c2w[:3, :3].T + c2w[:3, 3]

        keep_v = (np.ones(len(pts), bool) if show == "all"
                  else (~supported if show == "invented_only" else supported))
        # Same sky rejection as the ray path: geometry invented where the depth
        # stage saw no surface is spurious by construction.
        if reject_sky and sky is not None and sky.size:
            intr_s = out.camera.intrinsics
            Ws = float(intr_s.image_width or 1)
            Hs = float(intr_s.image_height or 1)
            fxs = float(intr_s.fx_px or Ws)
            fys = float(intr_s.fy_px or fxs)
            cxs = float(intr_s.cx_px if intr_s.cx_px is not None else Ws * 0.5)
            cys = float(intr_s.cy_px if intr_s.cy_px is not None else Hs * 0.5)
            cam_s = (pts - c2w[:3, 3]) @ c2w[:3, :3]
            zs = np.maximum(-cam_s[:, 2], 1e-9)
            us = np.clip((cam_s[:, 0] / zs * fxs + cxs) / max(Ws, 1e-9), 0, 1)
            vs_ = np.clip((-cam_s[:, 1] / zs * fys + cys) / max(Hs, 1e-9), 0, 1)
            iy = np.clip((vs_ * (sky.shape[0] - 1)).astype(np.int32),
                         0, sky.shape[0] - 1)
            ix = np.clip((us * (sky.shape[1] - 1)).astype(np.int32),
                         0, sky.shape[1] - 1)
            keep_v = keep_v & ~sky[iy, ix]
        if not keep_v.all():
            # Keep a face only when ALL of its vertices survive, then reindex.
            fmask = keep_v[faces].all(axis=1)
            faces = faces[fmask]
            used = np.unique(faces)
            if used.size == 0:
                return (out, f"AtlasLoadHiddenVolume: '{show}' selected no faces.")
            remap = np.full(len(pts), -1, dtype=np.int64)
            remap[used] = np.arange(used.size)
            faces = remap[faces]
            pts = pts[used]
            supported = supported[used]

        # DOUBLE-SIDED, and not by preference.
        #
        # The field is UNSIGNED: it stores distance-to-surface with no inside or
        # outside, so marching cubes has no orientation to be consistent with
        # and roughly half the emitted triangles face away from any given
        # viewpoint. Under single-sided shading that half vanishes, which reads
        # as "every other triangle has flipped normals" (reported live
        # 2026-08-15). A signed field would not need this; a TUDF always will.
        # Emitting each triangle in both windings costs 2x the faces and makes
        # the surface visible from either side.
        n_single = int(len(faces))
        if double_sided:
            faces = np.vstack([faces, faces[:, ::-1]])

        # Projective UVs from the RECOVERED camera, so the plate projects onto
        # the hidden surface exactly as it does onto the relief mesh.
        intr = out.camera.intrinsics
        W = float(intr.image_width or 1)
        H = float(intr.image_height or 1)
        fx = float(intr.fx_px or W)
        fy = float(intr.fy_px or fx)
        cx = float(intr.cx_px if intr.cx_px is not None else W * 0.5)
        cy = float(intr.cy_px if intr.cy_px is not None else H * 0.5)
        cam = (pts - c2w[:3, 3]) @ c2w[:3, :3]      # world -> camera
        z = np.maximum(-cam[:, 2], 1e-9)            # Atlas looks down -Z
        u = (cam[:, 0] / z) * fx + cx
        v = (-cam[:, 1] / z) * fy + cy
        uvs = np.stack([u / W, 1.0 - v / H], axis=-1)

        # THE GATE, enforced. Computed before the primitive is built so
        # "refuse" can actually refuse — the previous version printed a warning
        # and emitted the geometry anyway, which let a diverged volume reach the
        # viewport and any export indistinguishable from a sound one.
        invented_fraction = float(1.0 - supported.mean()) if len(supported) else 0.0
        diverged = invented_fraction > float(max_invented_fraction)
        needs_inspection = (not diverged and
                            invented_fraction >= float(inspect_invented_fraction))
        if diverged and on_divergence == "refuse":
            return (out, (
                f"AtlasLoadHiddenVolume: REFUSED — {invented_fraction*100:.1f}% of the "
                f"predicted surface has no visible support (gate "
                f"{max_invented_fraction*100:.0f}%).\n"
                f"  The prediction has diverged from the plate; emitting it would "
                f"put invented geometry in front of an artist as if it were sound.\n"
                f"  Fixes, in order: tune the depth band on the volume (measured "
                f"knee is ~2x subject distance, not at the subject); check the "
                f"plate is photographic (synthetic/CG collapses the depth stage); "
                f"or set on_divergence='mark' to inspect it anyway."))

        prim = AtlasProxyPrimitive(
            name=f"{name}_{len(out.projection_scene.proxy_geometry):02d}",
            primitive_type="mesh",
            dimensions=(0.0, 0.0, 0.0),
            material="atlas_projection_proxy",
            metadata={
                "role": PROXY_ROLE,
                "source": HIDDEN_VOLUME_SOURCE,
                "n_vertices": int(len(pts)),
                "n_faces": int(len(faces)),
                "vertices": np.round(pts.reshape(-1), 3).tolist(),
                "faces": faces.reshape(-1).astype(np.int64).tolist(),
                "uvs": np.round(uvs.reshape(-1), 4).tolist(),
                "edge_risk": [],
                "ribbon_t": [],
                # Research provenance — this geometry is a HYPOTHESIS.
                "hidden_volume": {
                    "path": str(path),
                    "threshold_voxels": float(threshold),
                    "voxel_edge_m": float(np.max(voxel)),
                    "grid_stride": int(k),
                    "double_sided": bool(double_sided),
                    "faces_single_sided": n_single,
                    "depth_scale": float(depth_scale),
                    "show": show,
                    "invented_fraction": invented_fraction,
                    "diverged": bool(diverged),
                    "needs_inspection": bool(needs_inspection),
                    "divergence_gate": float(max_invented_fraction),
                    "inspect_gate": float(inspect_invented_fraction),
                    "predictor": meta.get("backend", "volfill"),
                    "steps": meta.get("steps"),
                    "seed": meta.get("seed"),
                },
                "research_only": True,
            },
        )
        out.projection_scene.proxy_geometry.append(prim)

        inv = invented_fraction
        report = (
            f"AtlasLoadHiddenVolume: {prim.name} <- {path.name}\n"
            f"  {len(pts)} verts / {len(faces)} faces "
            f"({n_single} single-sided, grid stride {k}), show={show}\n"
            f"  voxel {np.max(voxel)*100:.1f} cm, threshold {threshold} voxels\n"
            f"  INVENTED {inv*100:.1f}% of emitted vertices"
            + (f"  ** DIVERGED (gate {max_invented_fraction*100:.0f}%) "
               f"— emitted but MARKED SUSPECT **\n" if diverged
               else (f"  ** AMBIGUOUS ({inspect_invented_fraction*100:.0f}–"
                     f"{max_invented_fraction*100:.0f}%) — INSPECT before "
                     f"trusting **\n" if needs_inspection else "\n"))
            + f"  research-only hypothesis geometry; appended, nothing clobbered"
        )
        return (out, report)

    def _load_raymarch(self, out, path, tudf, visible, sky, meta, threshold,
                       depth_scale, show, name, max_invented_fraction,
                       on_divergence, inspect_invented_fraction,
                       min_surface_coverage=0.02, reject_sky=True,
                       emit="combined", clear_rel=0.15, fill_gaps=True,
                       depth=None, restrict_mask=None,
                       invert_restrict_mask=False,
                       # keyword-only in practice: appended last so the
                       # positional call sites above keep their meaning
                       double_sided_set=False):
        """Ray-march extraction: one relief surface per marched layer.

        Marching the ray rather than the shell pairs each entry/exit into the
        MIDPOINT -- the surface's real position -- and yields layers
        front-to-back, which is Atlas's own layered-ray form. Meshing each layer
        with ``build_relief_mesh`` then gives oriented, single-sided geometry
        with projective UVs, so no ``double_sided`` workaround is needed and the
        +/-threshold placement error is gone.
        """
        np = _require_numpy()
        from atlas_camera.core.relief_mesh import build_relief_mesh
        from atlas_camera.core.volume_raymarch import march_layers

        bbox_min = np.asarray(meta["bbox_min"], dtype=np.float64)
        extent = np.asarray(meta["extent_xyz"], dtype=np.float64)
        res = int(tudf.shape[0])
        voxel_edge = float(np.max(extent) / res)

        intr = out.camera.intrinsics
        W = int(intr.image_width or 1024)
        H = int(intr.image_height or 1024)
        fx = float(intr.fx_px or W)
        fy = float(intr.fy_px or fx)
        cx = float(intr.cx_px if intr.cx_px is not None else W * 0.5)
        cy = float(intr.cy_px if intr.cy_px is not None else H * 0.5)
        # The volume is 256^3; marching at plate resolution buys nothing and
        # costs a lot. Scale intrinsics with the raster so the rays stay correct.
        k = min(1.0, 512.0 / float(max(W, H)))
        rw, rh = max(8, int(round(W * k))), max(8, int(round(H * k)))
        fxm, fym, cxm, cym = fx * k, fy * k, cx * k, cy * k

        layers, mstats = march_layers(
            tudf, bbox_min, extent, fx=fxm, fy=fym, cx=cxm, cy=cym,
            width=rw, height=rh, threshold=float(threshold), max_layers=6)
        coverage = float(mstats["rays_with_surface"]) / float(rw * rh)
        if not layers.any() or coverage < float(min_surface_coverage):
            return (out, (
                f"AtlasLoadHiddenVolume: REFUSED - the volume is EMPTY.\n"
                f"  Only {coverage*100:.2f}% of camera rays hit any predicted "
                f"surface (floor {min_surface_coverage*100:.1f}%).\n"
                f"  An empty volume scores 0% invented and would otherwise pass "
                f"the divergence gate looking sound; nothing is not agreement.\n"
                f"  Usually the depth stage collapsed on this plate - check it is "
                f"photographic and has texture to key off."))

        # Visible surface along the SAME rays, for provenance.
        vis_z0 = None
        if visible is not None:
            vlayers, _ = march_layers(
                visible, bbox_min, extent, fx=fxm, fy=fym, cx=cxm, cy=cym,
                width=rw, height=rh, threshold=float(threshold), max_layers=1)
            vis_z0 = vlayers[..., 0]

        # SKY. The predictor fills its whole cube, including the region the
        # depth stage masked out, where there is no surface at all. Seen live in
        # the viewport as shards floating above the roofline.
        sky_r = None
        n_sky = 0
        if reject_sky and sky is not None and sky.size:
            yi = np.linspace(0, sky.shape[0] - 1, rh).astype(np.int32)
            xi = np.linspace(0, sky.shape[1] - 1, rw).astype(np.int32)
            sky_r = sky[np.ix_(yi, xi)]

        if emit == "combined":
            return self._combine(
                out, path, layers, vis_z0, sky_r, meta, threshold, depth_scale,
                name, clear_rel, fill_gaps, max_invented_fraction,
                inspect_invented_fraction, on_divergence, depth,
                (rw, rh, fxm, fym, cxm, cym), voxel_edge, mstats, n_sky,
                restrict_mask, invert_restrict_mask)

        tol = 2.0 * voxel_edge
        n_tot = 0
        n_sup = 0
        n_ray_agree = 0
        n_ray_both = 0
        made = []
        vm = np.asarray(out.camera.extrinsics.camera_view_matrix, dtype=np.float64)
        # Ray directions for sampling the visible FIELD at each layer's 3D point.
        uu, vv = np.meshgrid(np.arange(rw, dtype=np.float64),
                             np.arange(rh, dtype=np.float64))
        ddx = (uu - cxm) / fxm
        ddy = (vv - cym) / fym
        inv_voxel = np.asarray(extent, dtype=np.float64) / res
        for li in range(layers.shape[2]):
            z = layers[..., li].astype(np.float64)
            hit = z > 1e-6
            if sky_r is not None:
                n_sky += int((hit & sky_r).sum())
                hit = hit & ~sky_r
            if not hit.any():
                continue
            # PROVENANCE, deliberately the SAME quantity the marching-cubes path
            # uses: is this point within the visible surface's own truncation
            # band anywhere in 3D. The divergence gate's thresholds (0.82/0.88)
            # were validated on 26 held-out volumes against THIS measure, so
            # changing the definition here would silently invalidate them.
            if visible is not None:
                gx = np.clip((ddx * z - bbox_min[0]) / inv_voxel[0],
                             0, visible.shape[2] - 1).astype(np.int32)
                gy = np.clip((ddy * z - bbox_min[1]) / inv_voxel[1],
                             0, visible.shape[1] - 1).astype(np.int32)
                gz = np.clip((z - bbox_min[2]) / inv_voxel[2],
                             0, visible.shape[0] - 1).astype(np.int32)
                supported = (visible[gz, gy, gx] <= (float(threshold) + 1.0)) & hit
            else:
                supported = np.zeros_like(hit)
            # A STRICTER, ray-wise agreement reported alongside — "does the
            # prediction land where the visible surface is along the SAME ray".
            # Measured 23 cm median on a 5.7 cm voxel, i.e. ~4 voxels: harsher
            # than the gate's measure and the more physically meaningful one.
            # Reported, not gated on, until it has its own held-out calibration.
            if vis_z0 is not None:
                both = hit & (vis_z0 > 1e-6)
                n_ray_both += int(both.sum())
                n_ray_agree += int(((np.abs(z - vis_z0) <= tol) & both).sum())
            n_tot += int(hit.sum())
            n_sup += int((supported & hit).sum())
            keep = hit
            if show == "invented_only":
                keep = hit & ~supported
            elif show == "visible_only":
                keep = hit & supported
            if not keep.any():
                continue
            depth = np.where(keep, z * float(depth_scale), 0.0)
            mesh = build_relief_mesh(depth, view_matrix=vm, fx=fxm, fy=fym,
                                     cx=cxm, cy=cym,
                                     grid_long_edge=min(rw, rh))
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            if len(faces) == 0:
                continue
            uvs = np.asarray(mesh.uvs, dtype=np.float64)
            sup_frac = float((supported & keep).sum() / max(int(keep.sum()), 1))
            made.append((li, verts, faces, uvs, sup_frac, int(keep.sum())))

        if not made:
            return (out, f"AtlasLoadHiddenVolume: '{show}' selected no surface.")

        invented_fraction = 1.0 - (n_sup / max(n_tot, 1))
        diverged = invented_fraction > float(max_invented_fraction)
        needs_inspection = (not diverged and
                            invented_fraction >= float(inspect_invented_fraction))
        if diverged and on_divergence == "refuse":
            return (out, (
                f"AtlasLoadHiddenVolume: REFUSED - {invented_fraction*100:.1f}% of "
                f"the marched surface has no visible support (gate "
                f"{max_invented_fraction*100:.0f}%).\n"
                f"  The prediction has diverged from the plate.\n"
                f"  Tune the depth band on the volume, or check the plate is "
                f"photographic; set on_divergence='mark' to inspect anyway."))

        base = len(out.projection_scene.proxy_geometry)
        for n_i, (li, verts, faces, uvs, sup_frac, px) in enumerate(made):
            out.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
                name=f"{name}_{base + n_i:02d}_L{li}",
                primitive_type="mesh",
                dimensions=(0.0, 0.0, 0.0),
                material="atlas_projection_proxy",
                metadata={
                    "role": PROXY_ROLE,
                    "source": HIDDEN_VOLUME_SOURCE,
                    "n_vertices": int(len(verts)),
                    "n_faces": int(len(faces)),
                    "vertices": np.round(verts.reshape(-1), 3).tolist(),
                    "faces": faces.reshape(-1).astype(np.int64).tolist(),
                    "uvs": np.round(uvs.reshape(-1), 4).tolist(),
                    "edge_risk": [],
                    "ribbon_t": [],
                    "hidden_volume": {
                        "path": str(path),
                        "extraction": "raymarch",
                        "layer": int(li),
                        "threshold_voxels": float(threshold),
                        "voxel_edge_m": voxel_edge,
                        "depth_scale": float(depth_scale),
                        "show": show,
                        "supported_fraction": sup_frac,
                        "ray_agreement_fraction": (
                            n_ray_agree / n_ray_both if n_ray_both else None),
                        "invented_fraction": invented_fraction,
                        "diverged": bool(diverged),
                        "needs_inspection": bool(needs_inspection),
                        "divergence_gate": float(max_invented_fraction),
                        "inspect_gate": float(inspect_invented_fraction),
                        "pixels": px,
                        "ray_coverage": coverage,
                        "sky_rejected_samples": int(n_sky),
                        "predictor": meta.get("backend", "volfill"),
                        "intrinsics_source": meta.get("intrinsics_source"),
                    },
                    "research_only": True,
                },
            ))

        layer_txt = ", ".join(
            f"L{li}({len(f)}f, {sf*100:.0f}% supported)"
            for li, _, f, _, sf, _ in made)
        state = ""
        if diverged:
            state = "  ** DIVERGED - MARKED SUSPECT **"
        elif needs_inspection:
            state = "  ** AMBIGUOUS - INSPECT **"
        report = (
            f"AtlasLoadHiddenVolume: ray-march -> {len(made)} layer mesh(es) "
            f"from {path.name}\n"
            f"  raster {rw}x{rh}, {mstats['rays_with_surface']} rays hit, "
            f"{mstats['mean_layers_per_hit']:.2f} layers/hit, "
            f"odd-crossing {mstats['odd_crossing_fraction']*100:.1f}%\n"
            f"  layers: {layer_txt}\n"
            f"  INVENTED {invented_fraction*100:.1f}% (gate measure){state}\n"
            + (f"  sky-rejected {n_sky} ray samples "
               f"(geometry invented where the depth stage saw nothing)\n"
               if n_sky else "")
            + (f"  ray-wise agreement with the visible surface: "
               f"{100.0 * n_ray_agree / max(n_ray_both, 1):.1f}% "
               f"(stricter; reported, not gated)\n" if n_ray_both else "")
            +
            f"  single-sided, midpoint-paired (no double-wall bias); appended"
            # A widget that does nothing must SAY it does nothing: the gate
            # doctrine wants a visible explanation for every silent skip, and
            # double_sided defaults True while raymarch is the default path.
            + ("\n  double_sided ignored: raymarch pairs the shell walls into "
               "midpoints, so the output is already single-sided and oriented"
               if double_sided_set else "")
        )
        return (out, report)

    def _combine(self, out, path, layers, vis_z0, sky_r, meta, threshold,
                 depth_scale, name, clear_rel, fill_gaps, max_invented_fraction,
                 inspect_invented_fraction, on_divergence, depth, raster,
                 voxel_edge, mstats, n_sky, restrict_mask=None,
                 invert_restrict_mask=False):
        """Layered rays -> ONE projection surface, via Atlas's own selector.

        This is the wiring the whole adapter existed for. The marched layers are
        exactly the per-pixel front-to-back stack ``core.hidden_geometry`` was
        written to consume, so the calibrated behaviour applies unchanged:

          select_hidden_surface  per-pixel FIRST-CLEARING-LAYER choice. Never a
                                 fixed layer index -- for a solid occluder,
                                 layer 1 is usually its own BACK face, and the
                                 background continuation is further back still.
          fill_hidden_gaps       diffuse scattered picks into one surface;
                                 fragmented depth shreds the relief mesh via its
                                 world-edge check whatever the thresholds say.

        The occlusion matte is then the pixels where the substituted surface is
        genuinely BEHIND a nearer occluder — geometry and paint are separate
        concerns, so the depth stays continuous (no clamping, which would
        re-create the metre-scale seams the fill just removed) while only the
        matte says where painting is legitimate.
        """
        np = _require_numpy()
        from atlas_camera.core.hidden_geometry import (
            fill_hidden_gaps, select_hidden_surface)
        from atlas_camera.core.relief_mesh import build_relief_mesh

        rw, rh, fxm, fym, cxm, cym = raster

        # VISIBLE reference. Atlas's own depth when wired, because the
        # predictor's visible volume is only a coarse voxelization of it.
        vis_src = "predictor_visible_volume"
        if depth is not None and getattr(depth, "depth", None) is not None:
            d = np.asarray(depth.depth, dtype=np.float64)
            if d.shape != (rh, rw):
                yi = np.linspace(0, d.shape[0] - 1, rh).astype(np.int32)
                xi = np.linspace(0, d.shape[1] - 1, rw).astype(np.int32)
                d = d[np.ix_(yi, xi)]
            visible_depth = np.nan_to_num(d, nan=0.0)
            vis_src = "atlas_depth_map"
        elif vis_z0 is not None:
            visible_depth = np.asarray(vis_z0, dtype=np.float64) * float(depth_scale)
        else:
            return (out, "AtlasLoadHiddenVolume: 'combined' needs a visible "
                         "surface — wire `depth`, or use a volume whose npz "
                         "carries visible_tudf.")

        stack = np.asarray(layers, dtype=np.float64) * float(depth_scale)
        if sky_r is not None:
            stack[sky_r] = 0.0            # never select a layer in the sky
        hidden, hidden_valid, hstats = select_hidden_surface(
            stack, visible_depth, clear_rel=float(clear_rel))

        if not hidden_valid.any():
            return (out, (
                "AtlasLoadHiddenVolume: no hidden surface cleared the occluder.\n"
                f"  registration rel_mad {hstats.get('registration_rel_mad')}, "
                f"clear_rel {clear_rel}.\n"
                "  Every marched layer sat within the clearance margin of the "
                "visible surface — i.e. the prediction found nothing BEHIND "
                "anything. Lower clear_rel, or the plate may simply have no "
                "occlusion to recover."))

        # The RAW selection is the real signal; everything below only spreads it.
        raw_fraction = float(hidden_valid.mean())

        # RESTRICT. fill_hidden_gaps treats the valid picks as Dirichlet samples
        # of ONE hidden surface and diffuses them across `region`. Handed the
        # whole frame it fills the whole frame: measured on sh001, a 0.9%
        # selection became 100% substitution. So the region must be bounded by an
        # explicit mask, and without one the fill is SKIPPED rather than silently
        # swallowing the plate.
        region = None
        if restrict_mask is not None:
            m = restrict_mask
            if hasattr(m, "detach"):
                m = m.detach().cpu().numpy()
            m = np.asarray(m, dtype=np.float64)
            # MASK arrives as (1, H, W) from ComfyUI — from torch OR numpy, so
            # squeezing only the torch case silently compared 1 against H and
            # built a garbage region (found live 2026-08-15).
            while m.ndim > 2 and m.shape[0] == 1:
                m = m[0]
            if m.ndim > 2:
                m = m[..., 0] if m.shape[-1] <= 4 else m.reshape(m.shape[-2:])
            if m.shape != hidden_valid.shape:
                yi = np.linspace(0, m.shape[0] - 1,
                                 hidden_valid.shape[0]).astype(np.int32)
                xi = np.linspace(0, m.shape[1] - 1,
                                 hidden_valid.shape[1]).astype(np.int32)
                m = m[np.ix_(yi, xi)]
            keep = m > 0.5
            if invert_restrict_mask:
                keep = ~keep
            region = keep & (visible_depth > 1e-6)
            caught = float((hidden_valid & region).sum()) / max(
                float(hidden_valid.sum()), 1.0)
            hidden_valid = hidden_valid & region

        fill_note = ""
        if fill_gaps and region is None:
            fill_note = (" (fill SKIPPED: no restrict_mask — diffusing across a "
                         "whole frame turns a small selection into total "
                         "replacement)")
        elif fill_gaps and not hidden_valid.any():
            # Distinguish this from the no-mask case: the mask WAS given, it
            # simply does not overlap where layers cleared the occluder. Saying
            # "no restrict_mask" here sent me looking for a plumbing bug that
            # did not exist.
            fill_note = (f" (fill SKIPPED: the restrict_mask covers "
                         f"{float(region.mean())*100:.1f}% of frame but catches "
                         f"NONE of the {raw_fraction*100:.2f}% that cleared the "
                         f"occluder — try invert_restrict_mask)")
        elif fill_gaps:
            hidden, hidden_valid = fill_hidden_gaps(hidden, hidden_valid, region)
            fill_note = (f" (diffused inside restrict_mask, which catches "
                         f"{caught*100:.0f}% of the cleared selection"
                         + (", INVERTED)" if invert_restrict_mask else ")"))

        # Geometry and paint are SEPARATE concerns. The surface stays continuous
        # so it meshes; the matte marks only where the hidden surface is truly
        # behind a nearer one, which is where painting is legitimate.
        matte = hidden_valid & (hidden > visible_depth * 1.02)

        merged = visible_depth.copy()
        merged[hidden_valid] = hidden[hidden_valid]

        substituted_fraction = float(hidden_valid.mean())
        # Gate on the RAW selection: the diffused fraction is an artifact of the
        # fill, so gating on it would flag a healthy small recovery as divergence
        # purely because the restrict mask was large.
        invented_fraction = raw_fraction
        diverged = invented_fraction > float(max_invented_fraction)
        needs_inspection = (not diverged and
                            invented_fraction >= float(inspect_invented_fraction))
        if diverged and on_divergence == "refuse":
            return (out, (
                f"AtlasLoadHiddenVolume: REFUSED - hidden geometry cleared the "
                f"occluder on {invented_fraction*100:.1f}% of pixels (gate "
                f"{max_invented_fraction*100:.0f}%). At that share the prediction "
                f"is rewriting the plate, not completing it."))

        vm = np.asarray(out.camera.extrinsics.camera_view_matrix, dtype=np.float64)
        mesh = build_relief_mesh(merged, view_matrix=vm, fx=fxm, fy=fym,
                                 cx=cxm, cy=cym, grid_long_edge=min(rw, rh))
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(faces) == 0:
            return (out, "AtlasLoadHiddenVolume: merged surface produced no faces.")
        uvs = np.asarray(mesh.uvs, dtype=np.float64)

        out.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
            name=f"{name}_{len(out.projection_scene.proxy_geometry):02d}_combined",
            primitive_type="mesh",
            dimensions=(0.0, 0.0, 0.0),
            material="atlas_projection_proxy",
            metadata={
                "role": PROXY_ROLE,
                "source": HIDDEN_VOLUME_SOURCE,
                "n_vertices": int(len(verts)),
                "n_faces": int(len(faces)),
                "vertices": np.round(verts.reshape(-1), 3).tolist(),
                "faces": faces.reshape(-1).astype(np.int64).tolist(),
                "uvs": np.round(uvs.reshape(-1), 4).tolist(),
                "edge_risk": [],
                "ribbon_t": [],
                "hidden_volume": {
                    "path": str(path),
                    "extraction": "raymarch",
                    "emit": "combined",
                    "visible_reference": vis_src,
                    "threshold_voxels": float(threshold),
                    "voxel_edge_m": voxel_edge,
                    "depth_scale": float(depth_scale),
                    "clear_rel": float(clear_rel),
                    "fill_gaps": bool(fill_gaps),
                    "substituted_fraction": substituted_fraction,
                    "raw_selection_fraction": raw_fraction,
                    "invented_fraction": invented_fraction,
                    "restricted": bool(restrict_mask is not None),
                    "restrict_inverted": bool(invert_restrict_mask),
                    "matte_fraction": float(matte.mean()),
                    "diverged": bool(diverged),
                    "needs_inspection": bool(needs_inspection),
                    "divergence_gate": float(max_invented_fraction),
                    "sky_rejected_samples": int(n_sky),
                    "selector_stats": {k: v for k, v in hstats.items()
                                       if isinstance(v, (int, float, str))},
                    "predictor": meta.get("backend", "volfill"),
                    "intrinsics_source": meta.get("intrinsics_source"),
                },
                "research_only": True,
            },
        ))

        state = ""
        if diverged:
            state = "  ** DIVERGED - MARKED SUSPECT **"
        elif needs_inspection:
            state = "  ** AMBIGUOUS - INSPECT **"
        report = (
            f"AtlasLoadHiddenVolume: COMBINED surface from {path.name}\n"
            f"  raster {rw}x{rh}, {mstats['rays_with_surface']} rays hit, "
            f"{mstats['mean_layers_per_hit']:.2f} layers/hit\n"
            f"  visible reference: {vis_src}\n"
            f"  layers clearing the occluder: {raw_fraction*100:.2f}% of pixels"
            f"{state}\n"
            f"  substituted into {substituted_fraction*100:.1f}% of the frame"
            f"{fill_note}\n"
            f"  occlusion matte covers {matte.mean()*100:.1f}% "
            f"(where the hidden surface is genuinely behind a nearer one)\n"
            f"  {len(verts)} verts / {len(faces)} faces — ONE surface carrying "
            f"photographed and inferred geometry together"
        )
        return (out, report, _to_matte(matte))
