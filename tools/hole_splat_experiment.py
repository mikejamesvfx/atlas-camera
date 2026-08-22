"""Phase-1 experiment: does masked splat fusion fill an occlusion hole?

    python tools/hole_splat_experiment.py --shot-dir <RAF folder> [--json out.json]

IT ADOPTS NOTHING. No default changes, no file is written unless ``--json`` is
passed — the doctrine ``tools/tear_sweep.py`` already states. This answers a
question; it does not wire a feature.

THE QUESTION. A camera move reveals pixels no photograph covered. The current
2.5D layer stack leaves those black. Can gaussians, trained ONLY inside that
region and seeded only inside the depth interval the rim measures, fill it
view-consistently?

THE BASELINE IT MUST BEAT IS ALREADY ON RECORD. `docs/development/design-rules.md`
(the "Patches are texture projectors, not geometry sources" rule) records that
per-pixel scale registration of patch-derived depth was confirmed insufficient
in real Nuke: "no scalar makes derived geometry sit in the primary's world."
That machinery is `core/patch_registration.solve_scale_from_primary`. Splat
fusion is a different mechanism — a joint photometric fit over several views
rather than a per-view monocular derivation — so it can plausibly succeed where
scale registration failed. The report carries both.

STAGES, and why each is separately reported. A stage that cannot run records a
structured failure and the report still writes. A phase-1 experiment whose
report says "we got as far as X" is evidence; one that dies with a traceback is
not.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    # Run as a script, sys.path[0] is tools/ — without this `import
    # atlas_camera` resolves to whatever copy is installed, which on a dev box
    # is usually a different checkout entirely (see build_feature_audit.py).
    sys.path.insert(0, str(REPO))

RAW_SUFFIXES = (".raf", ".nef", ".cr2", ".cr3", ".arw", ".dng")


class _MetaWithSensor:
    """RawMetadata plus the sensor millimetres the solver requires.

    Fujifilm RAF carries no ``sensor_width_mm`` EXIF tag, and its
    ``FocalPlaneXResolution`` reports ~41 mm for a 23.5 mm APS-C sensor, so
    that route is unusable. The solver needs the real figure: ``fx = focal *
    width / sensor_width``, so an error here is a proportional error in every
    recovered angle. Prefer ``--sensor-width-mm`` from the camera's spec sheet;
    the 35mm-equivalent fallback is only as good as an EXIF field rounded to
    whole millimetres (X-H2: 24.9 mm reported as 37 mm eq gives 24.2 mm, about
    3% off the true 23.5 mm).
    """

    def __init__(self, meta: Any, width_mm: float, height_mm: float) -> None:
        self.__dict__["_meta"] = meta
        self.__dict__["sensor_width_mm"] = float(width_mm)
        self.__dict__["sensor_height_mm"] = float(height_mm)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_meta"], name, None)


def _sensor_mm_from_jpeg(jpeg_path: Path) -> tuple[float, float] | None:
    """Sensor millimetres from a sidecar JPEG's focal-plane resolution.

    This is why the JPEGs are kept next to the RAWs. Fujifilm RAF carries no
    sensor size, and its own FocalPlaneXResolution is unusable (1879 px/cm
    implies a 41 mm sensor). The camera-written JPEG carries the real figure:
    7728 px / 3289 px-per-cm = 23.50 mm, the X-H2's actual APS-C width, with
    height 15.66 mm. Measured beats both an EXIF-rounded crop factor and a
    hand-typed spec-sheet number.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except ImportError:  # pragma: no cover - Pillow is a stated dependency
        return None
    try:
        with Image.open(jpeg_path) as image:
            exif = image.getexif()
            tags = {TAGS.get(k, k): v for k, v in exif.items()}
            tags.update({TAGS.get(k, k): v for k, v in exif.get_ifd(0x8769).items()})
            pixel_w, pixel_h = image.size
    except Exception:  # noqa: BLE001 - a missing sidecar is not an error
        return None

    res_x = float(tags.get("FocalPlaneXResolution") or 0.0)
    res_y = float(tags.get("FocalPlaneYResolution") or res_x)
    if res_x <= 0.0:
        return None
    unit = int(tags.get("FocalPlaneResolutionUnit") or 2)
    per_unit_mm = {2: 25.4, 3: 10.0, 4: 1.0}.get(unit)
    if per_unit_mm is None:
        return None
    width = int(tags.get("ExifImageWidth") or pixel_w)
    height = int(tags.get("ExifImageHeight") or pixel_h)
    return (width / res_x * per_unit_mm, height / max(res_y, 1e-9) * per_unit_mm)


def _sensor_mm(meta: Any, args: argparse.Namespace,
               jpeg_path: Path | None = None) -> tuple[float, float, str]:
    """(width_mm, height_mm, provenance)."""
    if float(args.sensor_width_mm) > 0.0:
        width = float(args.sensor_width_mm)
        height = (float(args.sensor_height_mm) if float(args.sensor_height_mm) > 0.0
                  else width * 2.0 / 3.0)
        return width, height, "explicit"
    if jpeg_path is not None and jpeg_path.is_file():
        measured = _sensor_mm_from_jpeg(jpeg_path)
        if measured is not None:
            return measured[0], measured[1], f"sidecar_jpeg({jpeg_path.name})"

    focal = float(getattr(meta, "focal_length_mm", 0.0) or 0.0)
    eq35 = float(getattr(meta, "focal_length_35mm", 0.0) or 0.0)
    if focal <= 0.0 or eq35 <= 0.0:
        raise ValueError(
            "cannot derive sensor width: RAW carries neither sensor_width_mm "
            "nor a 35mm-equivalent focal length — pass --sensor-width-mm")
    crop = eq35 / focal
    return 36.0 / crop, 24.0 / crop, f"derived_from_35mm_equiv(crop={crop:.4f})"


@dataclass
class Stage:
    name: str
    status: str = "pending"
    detail: str = ""
    seconds: float = 0.0
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"stage": self.name, "status": self.status, "detail": self.detail,
                "seconds": round(self.seconds, 2), **({"data": self.data} if self.data else {})}


class Experiment:
    """Runs the stages, records what happened, never hides a failure."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stages: list[Stage] = []
        self.state: dict[str, Any] = {}

    def run_stage(self, name: str, fn) -> Stage:
        stage = Stage(name)
        self.stages.append(stage)
        print(f"[{name}] ...", flush=True)
        start = time.time()
        try:
            detail = fn(stage)
            stage.status = "ok"
            stage.detail = detail or ""
        except Exception as exc:  # noqa: BLE001 - a stage failure IS the result
            stage.status = "failed"
            stage.detail = f"{type(exc).__name__}: {exc}"
            if self.args.traceback:
                traceback.print_exc()
        stage.seconds = time.time() - start
        print(f"[{name}] {stage.status} ({stage.seconds:.1f}s) {stage.detail}", flush=True)
        return stage

    # ---------------------------------------------------------------- stages

    def _sidecar_jpeg(self, raw_path: Path) -> Path | None:
        """The camera-written JPEG beside a RAW, wherever the card put it."""
        if self.args.jpeg_dir:
            candidates = [Path(self.args.jpeg_dir)]
        else:
            # Typical layout: <set>/RAF/<shot>/x.RAF next to <set>/JPG/x.JPG.
            candidates = [raw_path.parent, raw_path.parent.parent / "JPG",
                          raw_path.parent.parent.parent / "JPG"]
        for folder in candidates:
            for suffix in (".JPG", ".jpg", ".JPEG", ".jpeg"):
                candidate = folder / (raw_path.stem + suffix)
                if candidate.is_file():
                    return candidate
        return None

    def _write_temp_image(self, image: Any, name: str) -> Path:
        """Persist a decoded frame so file-only models can read it."""
        import numpy as np
        from PIL import Image

        out = Path(self.args.work_dir or ".") / "hole_splat_tmp"
        out.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(image, dtype=np.float32)
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        path = out / name
        Image.fromarray(arr.clip(0, 255).astype("uint8")).save(path)
        return path

    def stage_frames(self, stage: Stage) -> str:
        """Decode the burst. Constant intrinsics across a burst is a hard
        requirement of the rig, so it is checked rather than assumed."""

        from atlas_camera.raw.decode import decode_raw
        from atlas_camera.raw.metadata import read_raw_metadata
        from atlas_camera.core.multiview_types import MultiViewFrame

        shot_dir = Path(self.args.shot_dir)
        paths = sorted(p for p in shot_dir.iterdir()
                       if p.suffix.lower() in RAW_SUFFIXES)
        if len(paths) < 2:
            raise ValueError(f"need >= 2 RAW frames, found {len(paths)} in {shot_dir}")

        metas = [read_raw_metadata(str(p)) for p in paths]
        focals = {round(float(m.focal_length_mm or 0.0), 2) for m in metas}
        if len(focals) > 1:
            raise ValueError(
                f"focal length varies across the burst ({sorted(focals)} mm); the "
                "rig assumes one lens setting per burst")

        frames = []
        for path, meta in zip(paths, metas):
            # decode_raw returns (linear_rgb, display_srgb); the solver's
            # feature matching wants the display-encoded one.
            _linear, display_srgb = decode_raw(
                str(path), half_size=self.args.half_size)
            width_mm, height_mm, provenance = _sensor_mm(
                meta, self.args, self._sidecar_jpeg(path))
            frames.append(MultiViewFrame(
                image=display_srgb,
                raw_meta=_MetaWithSensor(meta, width_mm, height_mm),
                label=path.stem))
            sensor_note = (width_mm, height_mm, provenance)

        self.state["frames"] = frames
        self.state["frame_paths"] = paths
        height, width = frames[0].image.shape[:2]
        stage.data = {
            "shot": shot_dir.name,
            "n_frames": len(frames),
            "frame_names": [p.stem for p in paths],
            "focal_mm": sorted(focals)[0],
            "camera": f"{metas[0].camera_make} {metas[0].camera_model}",
            "iso": metas[0].iso,
            "aperture": metas[0].aperture,
            "decoded_size": [int(width), int(height)],
            "sensor_width_mm": round(sensor_note[0], 3),
            "sensor_height_mm": round(sensor_note[1], 3),
            "sensor_provenance": sensor_note[2],
        }
        return f"{len(frames)} frames at {width}x{height}, {sorted(focals)[0]}mm"

    #: The degradation ladder. Each rung relaxes exactly one sanctioned knob
    #: and carries the trust label that relaxation costs. Ordered STRICTEST
    #: FIRST, and the tier is outer to the frame set: a balanced solve without
    #: the held-out frame is worth more than a salvage solve with it, because
    #: a pose that is wrong cannot be redeemed by having more of them.
    #:
    #: Measured 2026-08-20 on atlas_raws/MultiShots: every three-frame burst
    #: refused at T1 (sh002 32 mutual matches, sh003 45 against balanced's
    #: floor of 48; sh001/sh004 ambiguous_motion_model). sh001's evidence at
    #: salvage reads planar_translated(decomposed=True, cells=2,
    #: positive_depth=1) — which passes salvage's min_grid_cells of 2. It
    #: failed only because `auto` refuses to pick between two models that both
    #: look marginal, and forcing "translated" is the sanctioned way to say
    #: "this is a translated capture, use the planar decomposition".
    SOLVE_LADDER = (
        ("balanced", "auto", "T1_balanced", "measured"),
        ("permissive", "auto", "T2_permissive", "measured_thin_overlap"),
        ("salvage", "auto", "T3_salvage", "salvage_repetitive_texture"),
        ("salvage", "translated", "T4_forced_translated", "salvage_forced_model"),
        ("salvage", "rotation_only", "T5_rotation_only", "rotation_only_no_parallax"),
    )

    _QUALITY_ORDER = ("conservative", "balanced", "permissive", "salvage")

    def _attempt_solve(self, frames, settings):
        """One rung: solve, applying the two in-tier fallbacks if core asks.

        The up-hint and learned-depth-scale fallbacks are not rungs of the
        ladder — they answer specific outcome codes rather than relaxing a
        threshold, so they belong inside every attempt.
        """
        from dataclasses import replace as _replace
        from atlas_camera.core.multiview_solver import solve_multiview

        anchor_route = "vanishing_points"
        scale_route = ("measured_baseline" if float(self.args.baseline_m) > 0
                       else "ground_plane")
        outcome = solve_multiview(frames, settings)

        # A boiler against a fronto-parallel brick wall has no two orthogonal
        # architectural VP directions, and the solver refuses rather than guess
        # world-up from sparse correspondence. `anchor_up_hint` is the
        # sanctioned way through: core stays torch-free and the ADAPTER (this
        # driver) supplies a learned up vector. Reported, never silently
        # preferred — reproducibility then follows GeoCalib rather than the
        # deterministic VP detector.
        code = getattr(outcome.diagnostics, "outcome_code", "")
        if outcome.solve is None and code == "degenerate_geometry" and self.args.up_hint:
            from atlas_camera.inference.learned_prior import estimate_camera_prior

            anchor = frames[0]
            png = self._write_temp_image(anchor.image, f"{anchor.label}_anchor.png")
            prior = estimate_camera_prior(str(png))
            settings = _replace(
                settings, anchor_up_hint=tuple(float(v) for v in prior.up_cam),
                anchor_up_hint_source=f"GeoCalib on {anchor.label}")
            outcome = solve_multiview(frames, settings)
            anchor_route = "geocalib_up_hint"

        # Scale is the last gate. Without a measured baseline the rig needs a
        # valid ground plane in the mutual overlap; a boiler filling the frame
        # has none. Learned METRIC depth for photo 1 is tier 2 and recorded as
        # such, because splats trained at the wrong scale cannot be rescaled.
        code = getattr(outcome.diagnostics, "outcome_code", "")
        if outcome.solve is None and code == "scale_unavailable" and self.args.depth_scale:
            from atlas_camera.inference.depth_estimator import estimate_depth

            anchor = frames[0]
            png = self._write_temp_image(anchor.image, f"{anchor.label}_scale.png")
            metric = estimate_depth(str(png), model_id=self.args.depth_model,
                                    max_side=int(self.args.depth_max_side))
            frames = [_replace(anchor, metric_depth=metric.depth)] + list(frames[1:])
            outcome = solve_multiview(frames, settings)
            scale_route = "learned_metric_depth"

        return outcome, frames, anchor_route, scale_route

    def stage_solve(self, stage: Stage) -> str:
        """Recover the rig, degrading down a recorded ladder rather than refusing.

        TWO THINGS THIS STAGE LEARNED THE HARD WAY.

        First, the held-out frame used to be excluded from the solve as well as
        from training, on the reasoning that it could then never leak. The cost
        was that it could never be USED: with no recovered pose there is no
        camera to render the fill into, and the scoring stage fell back to
        comparing the splat alpha against a threshold of itself. A held-out
        frame that cannot be scored against is not a control, it is a discarded
        photograph. So pose comes from all frames where the rig allows it, and
        the leak this admits is stated rather than hidden: the holdout's
        correspondences influence the rig, so it is a control on invented
        CONTENT, not on calibration.

        Second, one set of thresholds is not a system. Measured across five
        real bursts, the strict profile solved NONE of them with three frames.
        A tool that refuses on every capture it is handed has not been strict,
        it has been useless. So the rungs of ``SOLVE_LADDER`` are tried in
        order, the tier that succeeded is recorded beside every tier that
        refused and why, and the trust label travels with the solve into the
        falsification report — a salvage rig cannot quietly produce a
        confident claim.
        """

        from atlas_camera.core.multiview_types import MultiViewSettings

        frames = list(self.state["frames"])
        holdout_index = int(self.args.holdout)
        if not (0 <= holdout_index < len(frames)):
            raise ValueError(f"--holdout {holdout_index} outside 0..{len(frames) - 1}")
        holdout = frames[holdout_index]
        train_only = [f for i, f in enumerate(frames) if i != holdout_index]
        if len(train_only) < 2:
            raise ValueError("need >= 2 frames left after the holdout")

        floor = self._QUALITY_ORDER.index(self.args.match_quality)
        rungs = [r for r in self.SOLVE_LADDER
                 if self._QUALITY_ORDER.index(r[0]) >= floor]
        if not self.args.degrade:
            rungs = rungs[:1]

        frame_sets = [("all", list(frames))] if self.args.pose_holdout else []
        frame_sets.append(("train_only", train_only))

        attempts: list[dict] = []
        chosen = None
        for quality, mode, tier, trust in rungs:
            for set_name, candidate_frames in frame_sets:
                if set_name == "all" and len(candidate_frames) == len(train_only):
                    continue  # nothing to gain: the holdout is not a frame here
                settings = MultiViewSettings(
                    camera_height_m=float(self.args.camera_height),
                    match_quality=quality,
                    capture_mode=mode,
                    # Anchor tier 1: a tape measurement between photo 1 and
                    # photo 2 optical centres pins absolute scale directly on
                    # the rig and needs no ground plane at all. Nothing else
                    # here is as trustworthy.
                    baseline_m=float(self.args.baseline_m),
                    pair_topology="anchor_star",
                    seed=0,
                )
                try:
                    outcome, used, anchor_route, scale_route = self._attempt_solve(
                        candidate_frames, settings)
                except Exception as exc:  # a refusal raised rather than returned
                    attempts.append({"tier": tier, "frames": set_name,
                                     "outcome_code": type(exc).__name__,
                                     "summary": str(exc)[:400]})
                    continue
                if outcome.solve is None:
                    diag = outcome.diagnostics
                    attempts.append({
                        "tier": tier, "frames": set_name,
                        "outcome_code": str(getattr(diag, "outcome_code", "?")),
                        "summary": str(getattr(diag, "summary", diag))[:400]})
                    continue
                attempts.append({"tier": tier, "frames": set_name,
                                 "outcome_code": "ok"})
                chosen = (outcome, used, anchor_route, scale_route, tier, trust,
                          set_name)
                break
            if chosen is not None:
                break

        if chosen is None:
            lines = "; ".join(f"{a['tier']}/{a['frames']}: {a['outcome_code']}"
                              for a in attempts)
            raise RuntimeError(
                f"every rung of the solve ladder refused — {lines}. The last "
                f"summary was: {attempts[-1]['summary'] if attempts else 'none'}")

        outcome, used, anchor_route, scale_route, tier, trust, set_name = chosen
        posed_holdout = set_name == "all"
        # Identity, not index: `used` may have had photo 1 replaced by a copy
        # carrying a learned depth map, and filtering it by the holdout's
        # position in the ORIGINAL list drops a training frame whenever the
        # holdout is not last.
        train_frames = [f for f in used if f.label != holdout.label]

        self.state["solve"] = outcome.solve
        self.state["holdout"] = holdout
        self.state["holdout_index"] = holdout_index
        self.state["holdout_posed"] = posed_holdout
        self.state["train_frames"] = train_frames
        self.state["solve_tier"] = tier
        self.state["solve_trust"] = trust
        n_sources = len(outcome.solve.projection_sources)
        stage.data = {
            "holdout_frame": holdout.label,
            "solve_tier": tier,
            "solve_trust": trust,
            "tier_degraded": tier != rungs[0][2],
            "solve_attempts": attempts,
            "holdout_posed_by_solve": posed_holdout,
            "holdout_leak": (
                ("pose only: the holdout's correspondences enter the rig solve. "
                 "It contributes no depth, no colour and no splat gradient.")
                if posed_holdout else
                "none: no rung of the ladder solved a rig containing the "
                "holdout, so it has no recovered pose and cannot be scored."),
            "train_frames": [f.label for f in train_frames],
            "projection_sources": n_sources,
            "confidence": float(getattr(outcome.solve, "confidence", 0.0)),
            "source_method": outcome.solve.source_method,
            "anchor_route": anchor_route,
            "scale_route": scale_route,
            "baseline_m": float(self.args.baseline_m),
            "solver_warnings": list(getattr(outcome.diagnostics, "warnings", []) or []),
        }
        return (f"{tier} ({trust}), {len(train_frames)} train frames, "
                f"{n_sources} sources, holdout={holdout.label}"
                f"{'' if posed_holdout else ' UNPOSED'}")

    def stage_geometry(self, stage: Stage) -> str:
        """Depth -> metric scale -> relief mesh. Scale is settled BEFORE any
        seeding: splats trained at the wrong scale cannot be rescaled later."""

        import numpy as np
        from atlas_camera.core.camera_spec import CameraSpec
        from atlas_camera.core.relief_mesh import build_relief_mesh, estimate_ground_scale
        from atlas_camera.inference.depth_estimator import estimate_depth

        solve = self.state["solve"]
        spec = CameraSpec.from_solve(solve)
        # Depth models read image FILES; hand them the frame already decoded so
        # the depth belongs to the exact pixels the rig was solved from.
        primary_frame = self.state["train_frames"][0]
        primary_path = self._write_temp_image(
            primary_frame.image, f"{primary_frame.label}_primary.png")

        depth_result = estimate_depth(
            str(primary_path), model_id=self.args.depth_model,
            focal_px=float(spec.fx), max_side=int(self.args.depth_max_side))
        depth = np.asarray(depth_result.depth, dtype=np.float64)

        view = np.asarray(solve.camera.extrinsics.camera_view_matrix, dtype=np.float64)
        height, width = depth.shape[:2]
        sx = width / float(spec.width)
        sy = height / float(spec.height)
        fx, fy = spec.fx * sx, spec.fy * sy
        cx, cy = spec.cx * sx, spec.cy * sy

        scale, scale_info = estimate_ground_scale(
            depth, view_matrix=view, fx=fx, fy=fy, cx=cx, cy=cy)
        mesh = build_relief_mesh(
            depth, view_matrix=view, fx=fx, fy=fy, cx=cx, cy=cy,
            grid_long_edge=int(self.args.relief_grid), scale=float(scale))

        # Attach the mesh to the solve: `gather_scene_meshes` is what the
        # disocclusion renderer walks, and it only sees PROXY_ROLE geometry on
        # the solve — a mesh held in a local variable is invisible to it.
        from atlas_camera.core.proxy_geometry import relief_mesh_primitive

        solve.projection_scene.proxy_geometry = [relief_mesh_primitive(mesh)]

        self.state.update({
            "depth": depth, "depth_scale": float(scale), "mesh": mesh,
            "depth_camera": {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                             "width": width, "height": height},
            "primary_path": primary_path,
        })
        stage.data = {
            "depth_model": self.args.depth_model,
            "depth_size": [int(width), int(height)],
            "ground_scale": float(scale),
            "ground_scale_status": str(scale_info.get("status", "")),
            "mesh_vertices": int(len(mesh.vertices)),
            "mesh_faces": int(len(mesh.faces)),
            "torn_fraction": float(mesh.stats.get("torn_fraction", 0.0)),
        }
        return (f"scale {scale:.4f}, {len(mesh.vertices)} verts, "
                f"torn {mesh.stats.get('torn_fraction', 0.0):.4f}")

    def stage_move(self, stage: Stage) -> str:
        """Pick a move that actually opens the hole, and say by how much."""

        from atlas_camera.core.camera_path import (
            build_preset_camera_path, sample_camera_path, scene_median_depth_pivot,
        )

        solve = self.state["solve"]
        pivot = scene_median_depth_pivot(solve)
        path, delta = build_preset_camera_path(
            solve.camera.extrinsics, self.args.move,
            angle_deg=float(self.args.move_angle),
            frame_count=int(self.args.frames), pivot=pivot)
        views = sample_camera_path(path)

        self.state["views"] = views
        self.state["camera_path"] = path
        stage.data = {"move": self.args.move, "angle_deg": float(self.args.move_angle),
                      "frames": len(views), "pivot": [float(v) for v in pivot],
                      "orbit_delta": [float(v) for v in delta]}
        return f"{self.args.move} {self.args.move_angle}deg, {len(views)} frames"

    def stage_hole(self, stage: Stage) -> str:
        """The hole in the MOVED camera's raster — not the primary's.

        In the primary frame an occlusion hole has zero extent by construction;
        `ReliefMesh.hole_mask` there is the cliff band plus sky.
        """

        import numpy as np
        from atlas_camera.dynamic.occlusion_fill import (
            render_disocclusion_sequence, survey_hole_rois,
        )

        solve = self.state["solve"]
        source = self.state["train_frames"][0].image
        views = self.state["views"]
        target_views = [views[-1]] if self.args.last_frame_only else views
        # The renderer takes view MATRICES; sample_camera_path returns full
        # AtlasExtrinsics.
        view_matrices = [v.camera_view_matrix for v in target_views]

        sequence = render_disocclusion_sequence(
            solve, source, view_matrices,
            resolution=int(self.args.survey_resolution), hole_dilate_px=0)
        rois, roi_set, survey_masks, peak = survey_hole_rois(
            solve, source, view_matrices,
            survey_resolution=int(self.args.survey_resolution),
            move_revealed_only=True)

        if not rois:
            raise RuntimeError(
                "the move opened no clustered hole — try a larger --move-angle "
                f"(peak uncovered fraction {peak:.4f})")

        self.state["survey_masks"] = survey_masks
        self.state["rois"] = rois
        self.state["target_view"] = target_views[-1]
        self.state["hole_mask_survey"] = np.asarray(survey_masks[-1], dtype=bool)
        stage.data = {
            "n_rois": len(rois),
            "peak_uncovered_fraction": float(peak),
            "roi_0": {"x": rois[0].x, "y": rois[0].y,
                      "w": rois[0].width, "h": rois[0].height},
            "per_frame_uncovered": [float(cov) for _g, _m, cov in sequence],
        }
        return f"{len(rois)} ROIs, peak uncovered {peak:.4f}"

    def stage_seed(self, stage: Stage) -> str:
        """Ownership, the measured rim interval, and the volumetric seed."""

        import numpy as np
        from atlas_camera.core.camera_spec import CameraSpec
        from atlas_camera.core.hole_splat import (
            hole_ownership, rim_depth_interval, seed_hole_volume,
        )
        from atlas_camera.core.move_budget import rasterize_coverage

        solve = self.state["solve"]
        mesh = self.state["mesh"]
        spec = CameraSpec.from_solve(solve)
        hole = self.state["hole_mask_survey"]
        height, width = hole.shape
        sx = width / float(spec.width)
        sy = height / float(spec.height)
        fx, fy = spec.fx * sx, spec.fy * sy
        cx, cy = spec.cx * sx, spec.cy * sy
        view = np.asarray(self.state["target_view"].camera_view_matrix, dtype=np.float64)

        coverage, zbuffer = rasterize_coverage(
            mesh.vertices, mesh.faces, view_matrix=view,
            fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)

        own = hole_ownership(hole, coverage, overlap_px=int(self.args.overlap_px))
        if not own.splat_mask.any():
            raise RuntimeError("hole and coverage disagree: nothing left to fill")
        rim = rim_depth_interval(zbuffer, own.splat_mask,
                                 ring_px=int(self.args.overlap_px))
        seed = seed_hole_volume(
            own.splat_mask, rim, view_matrix=view, fx=fx, fy=fy, cx=cx, cy=cy,
            layers=int(self.args.slab_layers),
            pixel_stride=int(self.args.pixel_stride), seed=0)

        self.state.update({"ownership": own, "rim": rim, "seed": seed,
                           "target_camera": {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                                             "width": width, "height": height},
                           "target_view_matrix": view, "zbuffer": zbuffer,
                           "coverage": coverage})
        stage.data = {**own.report, "rim": {"near_m": rim.near_m, "far_m": rim.far_m,
                                            "n_ring_px": rim.n_ring_px},
                      "seed": seed.report, "n_gaussians": seed.count}
        return (f"{own.report['splat_px']} hole px, rim {rim.near_m:.2f}-"
                f"{rim.far_m:.2f} m, {seed.count} gaussians")

    def stage_train(self, stage: Stage) -> str:
        """Fit the gaussians against the training frames, masked to the hole."""

        import numpy as np
        from atlas_camera.inference.hole_splat_train import TrainView, train_hole_splats

        solve = self.state["solve"]
        cam = self.state["target_camera"]
        own = self.state["ownership"]
        views: list[TrainView] = []

        # The primary is always a supervision view: it constrains the splats not
        # to spill into what the plate already shows.
        primary = self.state["train_frames"][0].image
        primary_small = _resize(np, primary, cam["height"], cam["width"])
        views.append(TrainView(
            image=primary_small, loss_mask=own.splat_mask,
            view_matrix=np.asarray(solve.camera.extrinsics.camera_view_matrix),
            fx=cam["fx"], fy=cam["fy"], cx=cam["cx"], cy=cam["cy"],
            name="primary"))

        for source in solve.projection_sources:
            image = getattr(source, "_decoded_image", None)
            if image is None:
                continue
            views.append(TrainView(
                image=image, loss_mask=own.splat_mask,
                view_matrix=np.asarray(source.camera.extrinsics.camera_view_matrix),
                fx=cam["fx"], fy=cam["fy"], cx=cam["cx"], cy=cam["cy"],
                name=source.name))

        trained = train_hole_splats(
            self.state["seed"], views, iters=int(self.args.iters),
            lr=float(self.args.lr), device=self.args.device,
            log_every=int(self.args.log_every),
            pixel_chunk=int(self.args.pixel_chunk))

        self.state["trained"] = trained
        stage.data = trained.report
        return (f"{trained.count} gaussians kept, loss "
                f"{trained.report['loss_first']:.5f} -> {trained.report['loss_last']:.5f}")

    def _render_splats(self, view_matrix, cam):
        """Render the trained gaussians into one camera. Returns a SplatRender."""
        import torch
        from atlas_camera.inference.hole_splat_train import render_gaussians

        trained = self.state["trained"]
        view = torch.as_tensor(view_matrix, dtype=torch.float32)
        return render_gaussians(
            torch.as_tensor(trained.means, dtype=torch.float32),
            torch.as_tensor(trained.quats, dtype=torch.float32),
            torch.as_tensor(trained.scales, dtype=torch.float32),
            torch.as_tensor(trained.opacities, dtype=torch.float32),
            torch.as_tensor(trained.colors, dtype=torch.float32),
            view_matrix=view, fx=cam["fx"], fy=cam["fy"], cx=cam["cx"], cy=cam["cy"],
            width=cam["width"], height=cam["height"])

    def stage_score(self, stage: Stage) -> str:
        """Score the fill in the moved camera, beside the alternative it must beat.

        WHAT THIS REPLACED, and why. The previous version computed
        ``|alpha - (alpha > threshold)|`` and split it visible/occluded: an
        error metric whose reference is a THRESHOLD OF ITS OWN INPUT. It
        measures how binary the alpha is and nothing else, and it cannot fail.
        Beside it sat hole closure, which is coverage, not correctness. Together
        they reported the 2026-08-20 sh001 run as "closed 100.0% of hole" while
        it painted 143,465 pixels outside that hole.

        Everything now goes through ``core.plate_falsification``, which takes a
        baseline or refuses to produce a report. The baseline is the sealed
        relief mesh: bridging the tear is the real alternative to training
        splats, and ``move_budget.seal_relief_mesh`` already builds it.
        """

        import numpy as np
        from atlas_camera.core.move_budget import rasterize_coverage, seal_relief_mesh
        from atlas_camera.core.plate_falsification import (
            falsification_report, score_geometry_against_plate,
        )

        cam = self.state["target_camera"]
        own = self.state["ownership"]
        view = np.asarray(self.state["target_view_matrix"], dtype=np.float64)
        zbuffer = np.asarray(self.state["zbuffer"], dtype=np.float64)

        out = self._render_splats(view, cam)
        splat_alpha = out.alpha.detach().cpu().numpy()
        splat_depth = out.depth.detach().cpu().numpy()

        hole = own.splat_mask
        authorised = hole | own.overlap_mask
        closed = splat_alpha > float(self.args.alpha_threshold)
        if not closed.any():
            raise RuntimeError("the trained splats rasterize to nothing above "
                               f"alpha {self.args.alpha_threshold}")

        # Containment is scored on the UNRESTRICTED render: the splat layer has
        # no confinement mechanism of its own, and the pixel count of what it
        # paints outside its own hole is the finding, not a footnote.
        raw = score_geometry_against_plate(
            alpha=closed, authorised_mask=authorised, observed_mask=hole)

        # The head-to-head is between two HOLE FILLS, so both are restricted to
        # the same region before comparison. Scoring an unbounded splat layer
        # against a whole sealed mesh would be a contest between different
        # kinds of object.
        sealed = seal_relief_mesh(self.state["mesh"], self.state["solve"])
        sealed_cov, sealed_depth = rasterize_coverage(
            sealed.vertices, sealed.faces, view_matrix=view,
            fx=cam["fx"], fy=cam["fy"], cx=cam["cx"], cy=cam["cy"],
            width=cam["width"], height=cam["height"])
        base_alpha = sealed_cov & authorised
        cand_alpha = closed & authorised
        if not base_alpha.any() or not cand_alpha.any():
            raise RuntimeError("nothing to compare inside the authorised region")

        report = falsification_report(
            candidate=dict(alpha=cand_alpha, authorised_mask=authorised,
                           observed_mask=hole,
                           render_depth=np.where(cand_alpha, splat_depth, np.inf),
                           reference_depth=zbuffer, seed=0),
            baseline=dict(alpha=base_alpha, authorised_mask=authorised,
                          observed_mask=hole,
                          render_depth=np.where(base_alpha, sealed_depth, np.inf),
                          reference_depth=zbuffer, seed=0),
        )

        stage.data = {
            "hole_px": int(hole.sum()),
            "closure": raw["containment"]["closure"],
            "containment_unrestricted": raw["containment"]["value"],
            "spill_px_outside_authorised": raw["containment"]["spill_px"],
            "containment_gate_pass": raw["containment"]["pass"],
            "alpha_threshold": float(self.args.alpha_threshold),
            "head_to_head": report.to_dict(),
            "baseline": "sealed relief mesh (move_budget.seal_relief_mesh)",
            # The rig's trust travels with the score. A salvage-tier solve puts
            # the geometry somewhere approximate, so a metric computed against
            # it inherits that and must not read as a confident claim.
            "solve_tier": self.state.get("solve_tier"),
            "solve_trust": self.state.get("solve_trust"),
            "note": ("closure is COVERAGE, not correctness, and is reported here "
                     "only beside the spill it used to hide. The measurement "
                     "that decides content is stage holdout."),
        }
        self.state["splat_render"] = out
        spill = raw["containment"]["spill_px"]
        return (f"closure {raw['containment']['closure'] * 100:.1f}%, spill "
                f"{spill} px, beats sealed mesh: {report.beats_baseline}")

    def stage_holdout(self, stage: Stage) -> str:
        """The measurement that turns the experiment: a photograph never trained on.

        The splats are rendered into the held-out frame's recovered camera and
        compared against that frame's actual pixels, split VISIBLE (where the
        mesh already measured, so a good score proves nothing) versus OCCLUDED
        (where it did not, which is the entire hypothesis). A whole-frame
        average would let a fill score well by reproducing what it was shown.
        """

        import numpy as np
        from atlas_camera.core.hole_splat import split_visible_occluded
        from atlas_camera.core.move_budget import rasterize_coverage

        solve = self.state["solve"]
        holdout = self.state["holdout"]
        cam = dict(self.state["target_camera"])
        height, width = int(cam["height"]), int(cam["width"])

        source = next((src for src in solve.projection_sources
                       if src.name == holdout.label), None)
        if source is None:
            names = [src.name for src in solve.projection_sources]
            if not self.state.get("holdout_posed", False):
                tier = self.state.get("solve_tier", "?")
                raise RuntimeError(
                    "the held-out frame has no recovered camera: no rung of "
                    f"the solve ladder posed it (settled at {tier}), so the one "
                    "measurement that could falsify the fill cannot be made. "
                    "This is a capture limit, not a code path — the burst needs "
                    "real lateral baseline to the held-out frame.")
            raise RuntimeError(
                f"the solve carries no camera named {holdout.label!r}; have {names}")

        # CameraSpec is the seam that resolves the principal-point fallback
        # ladder (cx_px -> principal_point_px -> image centre) once. The
        # holdout is rendered at the target raster, so the intrinsics are
        # rescaled to it rather than the frame being resampled to them.
        from atlas_camera.core.camera_spec import CameraSpec

        spec = CameraSpec.from_intrinsics(
            source.camera.intrinsics,
            view_matrix=source.camera.extrinsics.camera_view_matrix)
        sx = width / float(spec.width or width)
        sy = height / float(spec.height or height)
        hcam = {"fx": spec.fx * sx, "fy": spec.fy * sy,
                "cx": spec.cx * sx, "cy": spec.cy * sy,
                "width": width, "height": height}
        hview = np.asarray(spec.view_matrix, dtype=np.float64)

        out = self._render_splats(hview, hcam)
        splat_rgb = out.rgb.detach().cpu().numpy()
        splat_alpha = out.alpha.detach().cpu().numpy()

        mesh = self.state["mesh"]
        coverage, _z = rasterize_coverage(
            mesh.vertices, mesh.faces, view_matrix=hview,
            fx=hcam["fx"], fy=hcam["fy"], cx=hcam["cx"], cy=hcam["cy"],
            width=width, height=height)

        truth = _resize(np, holdout.image, height, width).astype(np.float64)
        if truth.ndim == 2:
            truth = np.repeat(truth[..., None], 3, axis=2)
        truth = truth[..., :3]
        if truth.max() > 1.5:
            truth = truth / 255.0

        painted = splat_alpha > float(self.args.alpha_threshold)
        occluded = painted & ~coverage
        visible = painted & coverage
        if not occluded.any():
            raise RuntimeError(
                "the splats paint nothing the mesh had not already measured in "
                "the held-out view: there is no occluded region to score")

        error = np.abs(splat_rgb[..., :3] - truth).mean(axis=2)
        split = split_visible_occluded(error, occluded, coverage)

        # The null hypothesis, in the held-out view: what does the plate's own
        # mean colour score over the same occluded pixels? A fill that cannot
        # beat a flat patch has not learned the content.
        flat = float(truth[coverage].mean()) if coverage.any() else float(truth.mean())
        null_error = float(np.abs(flat - truth[occluded]).mean())
        occ_error = float(error[occluded].mean())

        stage.data = {
            "holdout_frame": holdout.label,
            "occluded_px": int(occluded.sum()),
            "visible_px": int(visible.sum()),
            "occluded_mean_abs_error": occ_error,
            "visible_mean_abs_error": (float(error[visible].mean())
                                       if visible.any() else None),
            "flat_patch_null_error": null_error,
            "beats_flat_patch": bool(occ_error < null_error),
            "solve_tier": self.state.get("solve_tier"),
            "solve_trust": self.state.get("solve_trust"),
            "error_split": split,
            "note": ("VISIBLE error is where the mesh already measured; a good "
                     "score there proves nothing. OCCLUDED is the hypothesis."),
        }
        return (f"holdout {holdout.label}: occluded err {occ_error:.4f} vs flat "
                f"{null_error:.4f} over {int(occluded.sum())} px")

    # ------------------------------------------------------------------ main

    def report(self) -> dict:
        ok = [s for s in self.stages if s.status == "ok"]
        return {
            "schema_version": 1,
            "experiment": "hole_splat_phase1",
            "stage_1a": "photographed_burst",
            "args": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(self.args).items()},
            "stages": [s.to_dict() for s in self.stages],
            "solve_tier": self.state.get("solve_tier"),
            "solve_trust": self.state.get("solve_trust"),
            "holdout_posed": self.state.get("holdout_posed"),
            "stages_ok": len(ok),
            "stages_total": len(self.stages),
            "completed": len(ok) == len(self.stages),
            "calibration_reference": {
                "note": ("score_tears reference values measured in "
                         "tests/test_tear_metrics.py: clean cut 1.000, "
                         "whole-cell tear 0.991, curtain 0.504"),
                "falsification": (
                    "plate_falsification gates measured by "
                    "tools/calibrate_falsification.py 2026-08-20: seam ratio "
                    "clean 1.000 vs weakest defect 1.426 (gate 1.25); "
                    "silhouette IoU truth 1.000, translate 0.5 m 0.821 "
                    "(gate 0.90)"),
            },
        }


def _resize(np, image, height, width):
    """Nearest-neighbour index remap — never blur an image into a metric."""
    src = np.asarray(image)
    rows = (np.linspace(0, src.shape[0] - 1, height)).astype(np.int64)
    cols = (np.linspace(0, src.shape[1] - 1, width)).astype(np.int64)
    return src[np.ix_(rows, cols)].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot-dir", required=True, help="folder of burst RAW frames")
    ap.add_argument("--json", default="", help="write the report here (default: nothing)")
    ap.add_argument("--no-degrade", dest="degrade", action="store_false",
                    help="try only the strictest rung of the solve ladder and "
                         "refuse rather than relax a threshold")
    ap.add_argument("--no-pose-holdout", dest="pose_holdout", action="store_false",
                    help="exclude the held-out frame from the rig solve too. It "
                         "then has no pose, so the holdout stage cannot run — "
                         "use when a three-view solve refuses.")
    ap.add_argument("--holdout", type=int, default=-1,
                    help="frame index held out entirely; default last")
    ap.add_argument("--camera-height", type=float, default=1.6)
    ap.add_argument("--baseline-m", type=float, default=0.0,
                    help="MEASURED metres between photo 1 and photo 2 optical "
                         "centres — anchor tier 1, better than any inference")
    ap.add_argument("--no-depth-scale", dest="depth_scale", action="store_false",
                    help="do not fall back to a learned metric depth for scale")
    ap.add_argument("--sensor-width-mm", type=float, default=0.0,
                    help="physical sensor width; 0 derives it from the 35mm "
                         "equivalent (X-H2 is 23.5)")
    ap.add_argument("--sensor-height-mm", type=float, default=0.0,
                    help="physical sensor height; 0 assumes 3:2 (X-H2 is 15.6)")
    ap.add_argument("--match-quality", default="balanced",
                    choices=["balanced", "conservative", "permissive", "salvage"])
    ap.add_argument("--half-size", action="store_true",
                    help="half-resolution RAW decode (much faster)")
    ap.add_argument("--depth-model",
                    default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
    ap.add_argument("--depth-max-side", type=int, default=1024)
    ap.add_argument("--relief-grid", type=int, default=96)
    ap.add_argument("--move", default="arc_left")
    ap.add_argument("--move-angle", type=float, default=15.0)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--last-frame-only", action="store_true", default=True)
    ap.add_argument("--survey-resolution", type=int, default=512)
    ap.add_argument("--overlap-px", type=int, default=8)
    ap.add_argument("--slab-layers", type=int, default=4)
    ap.add_argument("--pixel-stride", type=int, default=16,
                    help="every Nth hole pixel seeds a ray; the rasterizer "
                         "is dense, so this is the memory dial")
    ap.add_argument("--pixel-chunk", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--alpha-threshold", type=float, default=0.1)
    ap.add_argument("--no-up-hint", dest="up_hint", action="store_false",
                    help="do not fall back to a GeoCalib world-up when the "
                         "vanishing-point anchor fails")
    ap.add_argument("--jpeg-dir", default="",
                    help="folder of camera JPEGs; they carry the sensor size "
                         "the RAW does not (default: look beside the RAWs)")
    ap.add_argument("--work-dir", default="",
                    help="scratch dir for intermediate images (default: cwd)")
    ap.add_argument("--traceback", action="store_true")
    args = ap.parse_args()

    if args.holdout < 0:
        args.holdout = 10 ** 6  # resolved after the frame count is known

    exp = Experiment(args)
    exp.run_stage("frames", exp.stage_frames)
    if args.holdout == 10 ** 6 and "frames" in exp.state:
        args.holdout = len(exp.state["frames"]) - 1

    for name, fn in (("solve", exp.stage_solve), ("geometry", exp.stage_geometry),
                     ("move", exp.stage_move), ("hole", exp.stage_hole),
                     ("seed", exp.stage_seed), ("train", exp.stage_train),
                     ("score", exp.stage_score),
                     ("holdout", exp.stage_holdout)):
        if exp.stages[-1].status != "ok":
            print(f"[{name}] skipped — {exp.stages[-1].name} did not complete",
                  flush=True)
            exp.stages.append(Stage(name, status="skipped",
                                    detail=f"{exp.stages[-1].name} did not complete"))
            continue
        exp.run_stage(name, fn)

    report = exp.report()
    print("\n=== summary ===")
    for entry in report["stages"]:
        print(f"  {entry['stage']:9s} {entry['status']:8s} {entry['detail']}")
    print(f"  {report['stages_ok']}/{report['stages_total']} stages completed")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(f"  wrote {out}")
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
