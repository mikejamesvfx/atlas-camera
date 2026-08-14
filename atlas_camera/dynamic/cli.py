"""Headless Dynamic Plates CLI (spec §27).

Usage:
    python -m atlas_camera.dynamic create \\
        --image castle.png --matte ocean_mask.png --type water \\
        --solve atlas_solve.json --out shots/castle \\
        [--generator none|ltx] [--fps 24] [--frames 96] [--seed N] \\
        [--prompt "..."] [--overscan-frac 0.10] [--plane-height 0.0] \\
        [--feather-px 0] [--max-distance 500] [--blender] \\
        [--host 127.0.0.1:8188] [--template ltx_workflow.json] \\
        [--auto-matte "ocean, sea water"]

    python -m atlas_camera.dynamic validate --package shots/castle/dynamic/WATER_0001

Without ``--solve`` the camera is recovered from the image via
``atlas.recover`` (needs the ``[vision]`` extra). ``--auto-matte`` uses the
optional SAM3 assist; the artist ``--matte`` path never depends on it.
A missing/unavailable generator is NOT an error: the package is still built
and reports ``generator status = not_available`` (spec §32/§34).
"""
from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

from atlas_camera.core.dynamic_plate import (
    DYNAMIC_REGION_TYPES,
    GENERATOR_NOT_AVAILABLE,
    PLATE_STATUS_FAILED,
    PLATE_STATUS_GENERATED,
    PLATE_STATUS_READY,
    WATER_PROMPT_DEFAULT,
    DynamicPlate,
    build_receiver_plane,
    crop_intrinsics_for_plate,
    matte_bbox,
    validate_dynamic_plate,
    validate_matte_dimensions,
)


def _load_matte(path: str):
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("L")).astype(np.float32) / 255.0


def _auto_matte(image_path: str, concepts: str):
    try:
        from PIL import Image

        from atlas_camera.inference.sam3_segmenter import sam3_concept_mask
        with Image.open(image_path) as im:
            mask, matched, coverage = sam3_concept_mask(
                im.convert("RGB"), concepts)
    except (RuntimeError, ImportError) as exc:
        raise SystemExit(
            f"ERROR: --auto-matte needs the native SAM3 stack and failed "
            f"({exc}).\nSupply an artist matte with --matte instead — the "
            f"production workflow never depends on VLM success.") from None
    import numpy as np
    print(f"auto-matte: matched={matched} coverage={coverage:.3f}")
    return np.asarray(mask, dtype=np.float32)


def _resolve_camera(args, image_width: int, image_height: int):
    from atlas_camera.core.io import load_solve_json

    if args.solve:
        solve = load_solve_json(args.solve)
        return solve.camera
    print(f"no --solve given; recovering camera from the image "
          f"(atlas.recover, method={args.method})...")
    import atlas
    solve = atlas.recover(args.image, method=args.method)
    return solve.camera


def _next_plate_id(dynamic_dir: Path, semantic_type: str) -> str:
    prefix = semantic_type.upper()
    index = 1
    while (dynamic_dir / f"{prefix}_{index:04d}").exists():
        index += 1
    return f"{prefix}_{index:04d}"


def _print_issues(issues) -> int:
    fails = 0
    for issue in issues:
        print(f"{issue.severity}: {issue.code}: {issue.message}")
        if issue.severity == "fail":
            fails += 1
    return fails


def _render_pass(plate, package_dir, *, dolly_str, frames, gen_width,
                 gen_height) -> int:
    """Stage 2: render the crop along the dolly (v2v generator input)."""
    import numpy as np
    from PIL import Image

    from atlas_camera.core.dynamic_plate_render import (
        dolly_view_matrices,
        render_crop_sequence,
    )
    from atlas_camera.exporters.dynamic_plate_package import (
        write_plate_manifest,
    )

    try:
        dolly = tuple(float(v) for v in dolly_str.split(","))
        assert len(dolly) == 3
    except (ValueError, AssertionError):
        print(f"ERROR: --dolly must be 'dx,dy,dz', got {dolly_str!r}")
        return 1
    crop_path = package_dir / "source" / "crop.png"
    if not crop_path.exists():
        print(f"ERROR: projection_setup_failure: {crop_path} missing "
              f"(run create first)")
        return 1
    with Image.open(crop_path) as im:
        crop_rgb = np.asarray(im.convert("RGB"))
    # The plane homography is only VALID for pixels on the receiver: static
    # content (castle, cliffs) must stay static or the generator re-imagines
    # its slide as geometry. Composite per frame: warped pixels inside the
    # water matte, frame-0 crop outside (soft edge via the plate feather).
    matte_path = package_dir / "source" / "matte.png"
    water = None
    if matte_path.exists():
        from atlas_camera.core.dynamic_plate import feather_matte
        with Image.open(matte_path) as im:
            water = np.asarray(im.convert("L")).astype(np.float32) / 255.0
        water = feather_matte(water, max(4.0, plate.matte_feather_px))[..., None]
    views = dolly_view_matrices(plate.crop_camera, offset=dolly,
                                frame_count=frames)
    rendered_dir = package_dir / "rendered"
    rendered_dir.mkdir(exist_ok=True)
    for index, (rgb, alpha) in enumerate(
            render_crop_sequence(plate, views, crop_rgb)):
        # disocclusion at the frame edge falls back to the still crop so
        # the generator sees plausible pixels, never black
        mask = np.asarray(alpha)[..., None] > 0.5
        frame = np.where(mask, np.asarray(rgb), crop_rgb).astype(np.float32)
        if water is not None:
            frame = frame * water + crop_rgb.astype(np.float32) * (1.0 - water)
        frame = frame.astype(np.uint8)
        out_im = Image.fromarray(frame)
        if gen_width and gen_height:
            # generator runs at a reduced raster (same aspect): resize the
            # rendered input so the whole v2v pass stays at gen res
            out_im = out_im.resize((gen_width, gen_height), Image.LANCZOS)
        out_im.save(rendered_dir / f"frame_{index:04d}.png")
    plate.metadata["rendered_input"] = {
        "mode": "v2v", "dolly_m": list(dolly), "frame_count": frames,
        "gen_size": [gen_width, gen_height] if gen_width else None,
        "static_outside_matte": water is not None,
        "disocclusion_fill": "still_crop"}
    write_plate_manifest(plate, package_dir)
    print(f"rendered {frames} Atlas camera-move frames "
          f"(dolly {dolly} m) -> {rendered_dir}")
    return 0


def _generate_pass(plate, package_dir, args) -> tuple[int, list | None]:
    """Stage 3: temporal generation over an existing package."""
    from atlas_camera.dynamic.generators import (
        TemporalGenerationConfig,
        resolve_generator,
    )
    from atlas_camera.exporters.dynamic_plate_package import (
        write_plate_manifest,
    )

    generator = resolve_generator(args.generator)
    if args.host and hasattr(generator, "host"):
        generator.host = args.host
    if args.template and hasattr(generator, "template_path"):
        generator.template_path = args.template
    config = TemporalGenerationConfig(
        prompt=args.prompt, seed=args.seed, fps=args.fps,
        frame_count=args.frames,
        width=getattr(args, "gen_width", None),
        height=getattr(args, "gen_height", None),
        mode="video_to_video" if args.mode == "v2v" else "image_to_video")
    gen_result = generator.generate(plate, package_dir, config)
    print(f"generator {generator.name}: status = {gen_result.status}")
    for warning in gen_result.warnings:
        print(f"  warning: {warning}")
    (package_dir / "generated").mkdir(exist_ok=True)
    (package_dir / "generated" / "generation_result.json").write_text(
        json.dumps(gen_result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    frame_paths = None
    if gen_result.status == "ok":
        plate.status = PLATE_STATUS_GENERATED
        plate.frame_end = plate.frame_start + gen_result.frame_count - 1
        frame_paths = gen_result.frame_paths
    elif gen_result.status == GENERATOR_NOT_AVAILABLE:
        plate.metadata["generator_status"] = GENERATOR_NOT_AVAILABLE
    else:
        plate.status = PLATE_STATUS_FAILED
        plate.metadata["generator_status"] = gen_result.status
    write_plate_manifest(plate, package_dir)
    return (0 if gen_result.status != "failed" else 1), frame_paths


def cmd_render(args) -> int:
    from atlas_camera.exporters.dynamic_plate_package import load_dynamic_plate

    package_dir = Path(args.package)
    plate = load_dynamic_plate(package_dir)
    return _render_pass(plate, package_dir, dolly_str=args.dolly,
                        frames=args.frames, gen_width=args.gen_width,
                        gen_height=args.gen_height)


def cmd_generate(args) -> int:
    from atlas_camera.exporters.dynamic_plate_package import load_dynamic_plate

    package_dir = Path(args.package)
    plate = load_dynamic_plate(package_dir)
    plate.generator = args.generator
    plate.prompt = args.prompt
    plate.seed = args.seed
    rc, frame_paths = _generate_pass(plate, package_dir, args)
    if not rc and frame_paths and getattr(args, "matte_mode", "none") == "chroma":
        from atlas_camera.core.dynamic_plate import chroma_key_mattes
        mattes = chroma_key_mattes(frame_paths, package_dir / "generated")
        print(f"chroma-keyed {len(mattes)} matte(s)")
    issues = validate_dynamic_plate(plate, package_dir=package_dir,
                                    frame_paths=frame_paths)
    fails = _print_issues(issues)
    return 1 if (rc or fails) else 0


def cmd_create(args) -> int:
    import numpy as np
    from PIL import Image

    from atlas_camera.exporters.dynamic_plate_package import (
        build_dynamic_plate_package,
    )

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        return 1
    with Image.open(image_path) as im:
        image_width, image_height = im.size

    if args.auto_matte:
        matte = _auto_matte(str(image_path), args.auto_matte)
    else:
        matte = _load_matte(args.matte)
    try:
        validate_matte_dimensions(matte.shape, image_width, image_height)
    except ValueError as exc:
        print(f"ERROR: region_invalid: {exc}")
        return 1

    bbox = matte_bbox(matte)
    if bbox is None:
        print("ERROR: region_invalid: matte is empty")
        return 1
    roi = bbox.expanded(pad_frac=args.overscan_frac,
                        pad_px=args.overscan_px,
                        image_width=image_width, image_height=image_height)
    print(f"matte bbox: {bbox.to_dict()}")
    print(f"inference ROI (overscan): {roi.to_dict()}")

    camera = _resolve_camera(args, image_width, image_height)
    if camera.intrinsics.fx_px is None:
        print("ERROR: camera_crop_failure: the solve carries no usable focal "
              "length; supply a better --solve (or try --method learned)")
        return 1
    from atlas_camera.core.camera_crop import CropTransform

    crop_camera = crop_intrinsics_for_plate(camera, roi)
    ci = crop_camera.intrinsics
    print(f"crop camera: {ci.image_width}x{ci.image_height} "
          f"fx={ci.fx_px:.2f} fy={ci.fy_px:.2f} "
          f"cx={ci.cx_px:.2f} cy={ci.cy_px:.2f}")

    try:
        if args.card:
            from atlas_camera.core.dynamic_plate import build_receiver_card
            px, py, dist, width = (float(v) for v in args.card.split(","))
            receiver = build_receiver_card(camera, anchor_px=(px, py),
                                           distance_m=dist, width_m=width)
            print(f"card receiver: anchor ({px:g},{py:g}) px, "
                  f"{dist:g} m out, {width:g} m wide")
        else:
            receiver = build_receiver_plane(camera, roi,
                                            plane_height=args.plane_height,
                                            max_distance=args.max_distance)
    except ValueError as exc:
        print(f"ERROR: receiver_geometry_unavailable: {exc}")
        return 1

    dynamic_dir = Path(args.out) / "dynamic"
    plate = DynamicPlate(
        plate_id=_next_plate_id(dynamic_dir, args.type),
        semantic_type=args.type,
        source_image=image_path.name,
        source_width=image_width,
        source_height=image_height,
        matte_bbox=bbox,
        source_roi=roi,
        crop_transform=CropTransform(
            source_width=image_width, source_height=image_height, roi=roi,
            output_width=roi.width, output_height=roi.height),
        source_camera=camera,
        crop_camera=crop_camera,
        receiver=receiver,
        frame_rate=args.fps,
        frame_start=0,
        frame_end=max(0, args.frames - 1),
        generator=args.generator,
        prompt=args.prompt,
        seed=args.seed,
        matte_feather_px=args.feather_px,
        status=PLATE_STATUS_READY,
    )

    result = build_dynamic_plate_package(
        plate, dynamic_dir, source_image_path=image_path, matte=matte)
    print(f"package: {result.package_dir}")

    # STAGED BY DEFAULT: create only prepares the package (the crop camera
    # makes later passes self-contained). --render / --generate opt back into
    # running the follow-up stages inline.
    if args.mode == "v2v" and (args.render or args.generator != "none"):
        rc = _render_pass(plate, result.package_dir, dolly_str=args.dolly,
                          frames=args.frames, gen_width=args.gen_width,
                          gen_height=args.gen_height)
        if rc:
            return rc

    frame_paths = None
    if args.generator != "none":
        rc, frame_paths = _generate_pass(plate, result.package_dir, args)
        if rc:
            return rc

    if args.blender:
        from atlas_camera.exporters.dynamic_plate_blender import (
            write_dynamic_plate_blender_script,
        )
        script = write_dynamic_plate_blender_script(
            plate, result.package_dir,
            result.package_dir / "blender_open_scene.py")
        print(f"blender script: {script}")

    issues = validate_dynamic_plate(plate, package_dir=result.package_dir,
                                    matte_shape=matte.shape,
                                    frame_paths=frame_paths)
    fails = _print_issues(issues)
    if fails:
        print(f"FAILED: {fails} blocking issue(s)")
        return 1
    print("dynamic plate OK")
    return 0


def cmd_occlusion_fill(args) -> int:
    """Track A: render disocclusion guide+mask along a move, LTX-inpaint it."""
    import numpy as np
    from PIL import Image

    from atlas_camera.core.io import load_solve_json
    from atlas_camera.dynamic.generators import (
        TemporalGenerationConfig,
        resolve_generator,
    )
    from atlas_camera.dynamic.occlusion_fill import (
        render_disocclusion_sequence,
        write_exr_sequence,
        write_sequences,
    )

    solve = load_solve_json(args.solve)
    with Image.open(args.image) as im:
        source = np.asarray(im.convert("RGB"))

    views, parsed = _orbit_views(solve, args.orbit, args.frames)
    if views is None:
        print(f"ERROR: --orbit must be 'd_azimuth,d_elevation,distance_scale'"
              f", got {args.orbit!r}")
        return 1
    d_az, d_el, d_dist = parsed

    out_dir = Path(args.out)
    frames = render_disocclusion_sequence(
        solve, source, views, resolution=args.resolution,
        hole_dilate_px=args.hole_dilate_px)
    guide_paths, mask_paths = write_sequences(frames, out_dir)
    worst = max(range(len(frames)), key=lambda i: frames[i][2])
    print(f"rendered {len(frames)} guide+mask frames; peak disocclusion "
          f"{frames[worst][2]:.1%} at frame {worst}")
    # patch re-entry sidecar: the final orbit IS the patch camera
    (out_dir / "patch_exact.txt").write_text(
        f"azimuth_deg={d_az:.4f} elevation_deg={d_el:.4f} "
        f"distance_scale={d_dist:.4f}\n", encoding="utf-8")

    if args.generator == "none":
        print("generator none — guide/mask sequences ready for a manual "
              "inpaint run")
        return 0
    generator = resolve_generator(args.generator)
    if args.host and hasattr(generator, "host"):
        generator.host = args.host
    if args.template and hasattr(generator, "template_path"):
        generator.template_path = args.template
    gen = generator  # encode the two input videos next to the sequences
    err = gen._encode_rendered_mp4(out_dir / "guide", out_dir / "guide.mp4",
                                   args.fps)
    err = err or gen._encode_rendered_mp4(out_dir / "mask",
                                          out_dir / "mask.mp4", args.fps)
    if err:
        print(f"ERROR: {err}")
        return 1
    # generate() wants a package shape: use guide frame 0 as the still crop
    (out_dir / "source").mkdir(exist_ok=True)
    Image.open(guide_paths[0]).save(out_dir / "source" / "crop.png")

    class _PseudoPlate:
        source_roi = None
        crop_camera = None

    config = TemporalGenerationConfig(
        prompt=args.prompt, seed=args.seed, fps=args.fps,
        frame_count=len(frames),
        extra={"upload_markers": {"{GUIDE_VIDEO}": out_dir / "guide.mp4",
                                  "{MASK_VIDEO}": out_dir / "mask.mp4"}})
    result = generator.generate(_PseudoPlate(), out_dir, config)
    print(f"generator {generator.name}: status = {result.status}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    if result.status != "ok":
        return 0 if result.status == GENERATOR_NOT_AVAILABLE else 1
    if args.exr:
        written = write_exr_sequence(result.frame_paths, out_dir / "exr")
        print(f"EXR wrap: {len(written)} frames (display-referred float "
              f"container, NOT scene-linear)")
    print(f"filled sequence: {len(result.frame_paths)} frames in "
          f"{out_dir / 'generated'}; patch re-entry: feed frame_{worst:04d} "
          f"+ mask/frame_{worst:04d}.png through AtlasAddPatchView with "
          f"exact_view_override from patch_exact.txt")
    return 0


def _orbit_views(solve, orbit: str, frames: int):
    """Camera-move views from an 'd_az,d_el,dist_scale' spec (frame 0 = solved
    pose). Returns (views, parsed) or (None, None) on a malformed spec."""
    from atlas_camera.core.camera_math import ground_lookat_pivot, orbit_camera

    try:
        d_az, d_el, d_dist = (float(v) for v in orbit.split(","))
    except ValueError:
        return None, None
    pivot = ground_lookat_pivot(solve.camera.extrinsics)
    views = []
    frames = int(frames)
    # ONE frame means the DESTINATION, not the origin. The solved pose has no
    # disocclusion by construction, so a single-frame render at t=0 would ask
    # the generator to repair nothing; the end of the move carries the whole
    # move's holes, which is also why the crops union across time.
    offsets = [1.0] if frames <= 1 else [i / float(frames - 1)
                                         for i in range(frames)]
    for t in offsets:
        extr = orbit_camera(solve.camera.extrinsics, pivot,
                            d_azimuth_deg=d_az * t,
                            d_elevation_deg=d_el * t,
                            distance_scale=1.0 + (d_dist - 1.0) * t)
        views.append(extr.camera_view_matrix)
    return views, (d_az, d_el, d_dist)


def _artist_fill_regions(solve) -> list:
    """Fill regions the artist drew in the Atlas viewport, if any.

    Written by AtlasBlockoutViewport's ``fill_roi`` shapes — world-space
    corners, already budgeted there, so a shot carries its own repair
    selection instead of the pipeline ranking holes by area.
    """
    scene = getattr(solve, "projection_scene", None)
    meta = getattr(scene, "debug_metadata", None) or {}
    entry = meta.get("fill_rois") or {}
    return list(entry.get("regions") or [])


def cmd_hole_crop_fill(args) -> int:
    """Track A, cropped: cluster the move's holes, fill each ROI at 1:1.

    Disocclusion is SPARSE. Generating a whole frame to repair a fifth of it
    spends most of the compute re-imagining pixels Atlas already has, dilutes
    the conditioning (which is how the model escapes into inventing its own
    scene), and forces an 8K plate through a 1024px raster. A crop is a shifted
    principal point with a smaller raster, so cropping costs no new renderer
    and native resolution comes free.
    """
    import json

    import numpy as np
    from PIL import Image

    from atlas_camera.core.camera_crop import (
        RegionROI,
        hole_rois,
        rois_from_world_regions,
    )
    from atlas_camera.core.camera_spec import CameraSpec
    from atlas_camera.core.io import load_solve_json
    from atlas_camera.core.projection_render import gather_scene_meshes
    from atlas_camera.dynamic.generators import (
        TemporalGenerationConfig,
        resolve_generator,
    )
    from atlas_camera.dynamic.occlusion_fill import (
        build_scene_textures,
        crop_context_depth,
        render_crop_sequence,
        render_disocclusion_sequence,
        write_sequences,
    )

    if (args.frames - 1) % 8:
        print(f"ERROR: LTX wants 8n+1 frames (49, 97, ...), got {args.frames}")
        return 1

    solve = load_solve_json(args.solve)
    with Image.open(args.image) as im:
        source = np.asarray(im.convert("RGB"))
    intr = solve.camera.intrinsics
    plate_w, plate_h = int(intr.image_width), int(intr.image_height)

    views, parsed = _orbit_views(solve, args.orbit, args.frames)
    if views is None:
        print(f"ERROR: --orbit must be 'd_azimuth,d_elevation,distance_scale'"
              f", got {args.orbit!r}")
        return 1
    d_az, d_el, d_dist = parsed

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Survey pass: find WHERE the holes are cheaply. Rendering 8K x N frames
    # only to threshold alpha would cost more than the fills themselves; the
    # ROIs are then lifted back to plate resolution, which is conservative
    # because the padding and the /64 snap both only ever grow the crop.
    # UNDILATED for clustering. Dilation exists to give diffusion context past
    # the exact tear, and it is applied to the per-ROI masks below — but at
    # survey resolution a few pixels of growth BRIDGES separate tears, chaining
    # compact clusters into one sprawling component whose bounding box covers
    # the frame. Measured on DSC_2289: 4 px of survey dilation took the top-4
    # ROI coverage from 23% to 124% of the plate. The context the crop needs
    # comes from pad_frac, which grows the ROI without merging clusters.
    survey = render_disocclusion_sequence(
        solve, source, views, resolution=args.survey_resolution,
        hole_dilate_px=0)
    survey_masks = [mask for _guide, mask, _cov in survey]
    survey_h, survey_w = survey_masks[0].shape[:2]
    peak = max(cov for _g, _m, cov in survey)

    # ARTIST SELECTION WINS. Regions drawn in the viewport are a judgement
    # about which tears are worth inventing pixels for — automatic clustering
    # ranks by area, which was only ever a stand-in for that judgement. When
    # the artist has marked regions, cluster analysis still RUNS (its report is
    # what tells them what they left unrepaired) but does not choose.
    artist_regions = _artist_fill_regions(solve)
    artist_rois = None
    if artist_regions and not args.ignore_artist_rois:
        spec = CameraSpec.from_intrinsics(intr)
        picked = rois_from_world_regions(
            artist_regions, views[-1], fx=spec.fx, fy=spec.fy,
            cx=spec.cx, cy=spec.cy,
            image_width=plate_w, image_height=plate_h,
            pad_frac=args.pad_frac, snap=args.snap)
        artist_rois = picked.rois
        print(f"artist selection: {len(picked.rois)} viewport region(s) — "
              f"automatic clustering reports only")
        for drop in picked.dropped:
            print(f"  DROPPED {drop['reason']}")

    # Cluster WITHOUT the budget so the long-edge filter below chooses among
    # every candidate; the budget is applied after, on what can actually be
    # generated at 1:1.
    lores = hole_rois(survey_masks, pad_frac=args.pad_frac,
                      min_area_px=args.min_area_px, snap=1, max_rois=0)
    sx, sy = plate_w / float(survey_w), plate_h / float(survey_h)
    rois = []
    for roi in lores.rois:
        scaled = RegionROI(x=int(roi.x * sx), y=int(roi.y * sy),
                           width=max(1, int(round(roi.width * sx))),
                           height=max(1, int(round(roi.height * sy))))
        plate_roi = scaled.clamped(plate_w, plate_h).snapped(
            args.snap, image_width=plate_w, image_height=plate_h)
        # A crop RENDERS at plate resolution for free, but it must also be
        # GENERATED, and the model's raster budget is fixed (~0.6 MP/frame on
        # LTX-2.5). A cluster larger than that cannot be filled at 1:1 in one
        # pass, so by default it is declined by name rather than quietly
        # downscaled — downscaling is exactly the full-frame failure this
        # pipeline exists to avoid. `--oversize tiled` keeps it instead:
        # LTXVTiledSampler fills at NATIVE raster in spatial tiles (measured
        # 2026-08-14: a 1088x4928 strip with a 26% hole passed the seam gate
        # at 1.38 with registration intact; the fill runs soft) — it needs a
        # tiled template (--template atlas_ltx25_inpaint_tiled_v2v.json class)
        # and requires the pack's STGGuiderAdvanced, not CFGGuider.
        limit = int(args.max_roi_long_edge)
        if limit and max(plate_roi.width, plate_roi.height) > limit:
            if args.oversize == "tiled":
                print(f"  oversize cluster {plate_roi.width}x"
                      f"{plate_roi.height} kept for TILED generation "
                      f"(needs a tiled --template)")
            else:
                lores.dropped.append(
                    {**plate_roi.to_dict(),
                     "area_px": plate_roi.area_px,
                     "reason": f"long edge "
                               f"{max(plate_roi.width, plate_roi.height)}px "
                               f"> max_roi_long_edge {limit} — a native 1:1 "
                               f"fill would need the generator to downscale "
                               f"(--oversize tiled to fill it anyway)"})
                continue
        rois.append(plate_roi)
    # Budget last: the largest fillable clusters win, the rest are logged.
    rois.sort(key=lambda r: r.area_px, reverse=True)
    if artist_rois is None and args.max_rois > 0 and len(rois) > args.max_rois:
        for extra in rois[args.max_rois:]:
            lores.dropped.append(
                {**extra.to_dict(), "area_px": extra.area_px,
                 "reason": f"beyond max_rois {args.max_rois} (ranked by area)"})
        rois = rois[:args.max_rois]

    auto_rois = list(rois)
    if artist_rois is not None:
        # The survey's numbers stay in the report — that is how the artist
        # sees what they chose NOT to repair — but they no longer select.
        rois = list(artist_rois)

    generated_px = sum(r.area_px for r in rois) * args.frames
    full_frame_px = plate_w * plate_h * args.frames
    report = {
        "plate": [plate_w, plate_h],
        "frames": args.frames,
        "orbit": {"d_azimuth_deg": d_az, "d_elevation_deg": d_el,
                  "distance_scale": d_dist},
        "peak_hole_frac": peak,
        "survey_raster": [survey_w, survey_h],
        "components_found": lores.component_count,
        "selection": "artist" if artist_rois is not None else "automatic",
        "rois": [r.to_dict() for r in rois],
        "auto_candidate_rois": [r.to_dict() for r in auto_rois],
        "dropped": lores.dropped,
        "generated_px": generated_px,
        "full_frame_px": full_frame_px,
        "generated_vs_full_frame": (generated_px / float(full_frame_px)
                                    if full_frame_px else 0.0),
    }
    (out_dir / "hole_crop_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"peak disocclusion {peak:.1%}; {lores.component_count} hole "
          f"cluster(s) -> {len(rois)} ROI(s) at plate resolution")
    for index, roi in enumerate(rois):
        print(f"  roi {index}: {roi.width}x{roi.height} at "
              f"({roi.x},{roi.y})")
    # Silent truncation is forbidden: a dropped cluster still has holes in it,
    # and small scattered ones usually want deterministic edge-extend.
    for drop in lores.dropped:
        print(f"  DROPPED cluster {drop['width']}x{drop['height']} at "
              f"({drop['x']},{drop['y']}): {drop['reason']}")
    if not rois:
        print("no hole clusters survived the budget — nothing to fill")
        return 0
    ratio = report["generated_vs_full_frame"]
    print(f"G1 pixel budget: {generated_px/1e6:.1f}M generated vs "
          f"{full_frame_px/1e6:.1f}M full-frame ({ratio:.1%})")
    if ratio > 0.7:
        print("  WARNING: crops cover most of the frame — the hole-crop "
              "premise does not hold for this plate/move")

    # Gather the scene ONCE for every ROI (the meshes and textures do not
    # depend on the crop).
    meshes = gather_scene_meshes(solve, with_uvs=True)
    textures = build_scene_textures(solve, source)
    spec = CameraSpec.from_intrinsics(intr)
    print(f"plate camera: fx={spec.fx:.1f}px — crops render 1:1, no "
          f"long-edge normalisation")

    generator = None
    if args.generator != "none":
        generator = resolve_generator(args.generator)
        if args.host and hasattr(generator, "host"):
            generator.host = args.host
        if args.template and hasattr(generator, "template_path"):
            generator.template_path = args.template

    class _PseudoPlate:
        source_roi = None
        crop_camera = None

    for index, roi in enumerate(rois):
        roi_dir = out_dir / f"roi_{index:02d}"
        roi_dir.mkdir(parents=True, exist_ok=True)
        (roi_dir / "roi.json").write_text(json.dumps(roi.to_dict(), indent=2),
                                          encoding="utf-8")
        frames = render_crop_sequence(solve, source, views, roi,
                                      hole_dilate_px=args.hole_dilate_px,
                                      meshes=meshes, textures=textures)
        # Depth-proportional generation raster. A crop always RENDERS at 1:1,
        # but what it is worth GENERATING at scales with distance: plate detail
        # per world-metre falls as 1/depth, so a far crop generated at 1/k and
        # resampled back loses nothing that was recoverable. `depth_scale_ref`
        # is the depth at which 1:1 is demanded; nearer crops are never scaled
        # ABOVE 1:1 (there is no detail to invent), and the floor keeps a
        # distant crop from collapsing below what the model can condition on.
        gen_w, gen_h, scale = roi.width, roi.height, 1.0
        depth_m = crop_context_depth(frames)
        # A tiled oversize ROI generates at NATIVE raster by definition — the
        # tiles are what fit the model budget, so no depth scaling applies.
        tiled_roi = (args.oversize == "tiled" and args.max_roi_long_edge > 0
                     and max(roi.width, roi.height) > args.max_roi_long_edge)
        if not tiled_roi and args.depth_scale_ref > 0 and depth_m > 0:
            scale = min(1.0, max(args.min_gen_scale,
                                 args.depth_scale_ref / depth_m))
        if not tiled_roi and args.max_gen_long_edge > 0:
            longest = max(roi.width, roi.height) * scale
            if longest > args.max_gen_long_edge:
                scale *= args.max_gen_long_edge / longest
        if scale < 1.0:
            snap = max(32, int(args.snap))
            gen_w = max(snap, int(round(roi.width * scale / snap)) * snap)
            gen_h = max(snap, int(round(roi.height * scale / snap)) * snap)
        worst = max(range(len(frames)), key=lambda i: frames[i][2])
        if (gen_w, gen_h) != (roi.width, roi.height):
            # Write at the GENERATION raster. The mask resamples NEAREST: a
            # bilinear mask edge invents grey pixels that are neither "keep"
            # nor "invent", and the inpaint preprocessor thresholds them into
            # a ragged boundary.
            resized = []
            for guide, mask, cov, depth in frames:
                g = np.asarray(Image.fromarray(guide).resize(
                    (gen_w, gen_h), Image.LANCZOS))
                m = np.asarray(Image.fromarray(mask, mode="L").resize(
                    (gen_w, gen_h), Image.NEAREST))
                resized.append((g, m, cov, depth))
            frames = resized
        guide_paths, _mask_paths = write_sequences(frames, roi_dir)
        detail = ("1:1 native" if (gen_w, gen_h) == (roi.width, roi.height)
                  else f"generated {gen_w}x{gen_h} "
                       f"(1:{roi.width / float(gen_w):.2f}, depth-justified)")
        print(f"roi {index}: {len(frames)} frames at {roi.width}x{roi.height}"
              f", peak hole {frames[worst][2]:.1%} at frame {worst}"
              f", context depth {depth_m:.1f}m -> {detail}")
        if frames[worst][2] <= 0.0:
            # An artist-drawn region with no tear in it is a mis-click, not a
            # no-op to run anyway: generating there spends a fill re-inventing
            # pixels the plate already has, and the budget is 2-3.
            print("  WARNING: no disocclusion inside this region — nothing to "
                  "repair here; the fill would only re-render real pixels")
        report["rois"][index].update(
            {"context_depth_m": depth_m, "gen_width": gen_w,
             "gen_height": gen_h, "gen_scale": gen_w / float(roi.width),
             "peak_hole_frac": frames[worst][2]})
        (out_dir / "hole_crop_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        if generator is None:
            continue
        # A one-frame clip at 24 fps lasts 1/24 s and its silent AAC track
        # rounds to ZERO samples, which kills the graph in VAEEncodeAudio
        # ("cannot reshape tensor of 0 elements") — LTX's AV graphs require
        # audio. Stretching the single frame to a full second keeps the track
        # non-empty; frame_count is what the model actually reads.
        enc_fps = 1.0 if len(frames) == 1 else args.fps
        err = generator._encode_rendered_mp4(roi_dir / "guide",
                                             roi_dir / "guide.mp4", enc_fps)
        err = err or generator._encode_rendered_mp4(
            roi_dir / "mask", roi_dir / "mask.mp4", enc_fps)
        if err:
            print(f"ERROR: {err}")
            return 1
        (roi_dir / "source").mkdir(exist_ok=True)
        Image.open(guide_paths[0]).save(roi_dir / "source" / "crop.png")
        config = TemporalGenerationConfig(
            prompt=args.prompt, seed=args.seed, fps=args.fps,
            frame_count=len(frames), width=gen_w, height=gen_h,
            extra={"upload_markers": {
                "{GUIDE_VIDEO}": roi_dir / "guide.mp4",
                "{MASK_VIDEO}": roi_dir / "mask.mp4"}})
        result = generator.generate(_PseudoPlate(), roi_dir, config)
        print(f"  generator {generator.name}: status = {result.status}")
        for warning in result.warnings:
            print(f"    warning: {warning}")
        if result.status != "ok":
            return 0 if result.status == GENERATOR_NOT_AVAILABLE else 1

    if generator is not None and len(views) == 1:
        rc = _write_patch_view(args, out_dir, solve, source, views[0], rois,
                               meshes, textures, report,
                               (d_az, d_el, d_dist))
        if rc:
            return rc

    print(f"report: {out_dir / 'hole_crop_report.json'}")
    return 0


def _write_patch_view(args, out_dir, solve, source, view, rois, meshes,
                      textures, report, orbit) -> int:
    """Composite the ROI fills into ONE full-frame plate at the move's end.

    A single frame is all the generator should ever be asked for: the fill is a
    static texture, and 49 near-identical frames spend the whole token budget
    on temporal redundancy instead of resolution (LTX costs roughly
    (W/32)(H/32) per latent frame, so one frame at 4096 wide costs about what
    49 frames at 1024 do). The composited frame is written as a PATCH VIEW —
    fed back through AtlasAddPatchView with the exact view override, it becomes
    geometry-projected scene texture, so every frame of the move gets the
    repair by projection rather than by generating it again.
    """
    import json

    import numpy as np
    from PIL import Image

    from atlas_camera.core.camera_crop import (
        composite_crops,
        match_reference_colour,
        membrane_blend,
        neutralize_fill_cast,
    )
    from atlas_camera.dynamic.occlusion_fill import (
        render_crop_sequence,
        render_disocclusion_sequence,
    )

    d_az, d_el, d_dist = orbit
    crops, masks, kept = [], [], []
    for index, roi in enumerate(rois):
        generated = sorted((out_dir / f"roi_{index:02d}" / "generated")
                           .glob("frame_*.png"))
        if not generated:
            print(f"  roi {index}: no generated frame — skipped in composite")
            continue
        guide, mask, _cov, _depth = render_crop_sequence(
            solve, source, [view], roi, hole_dilate_px=args.hole_dilate_px,
            meshes=meshes, textures=textures)[0]
        with Image.open(generated[-1]) as im:
            fill = np.asarray(im.convert("RGB"))
        if fill.shape[:2] != (roi.height, roi.width):
            fill = np.asarray(Image.fromarray(fill).resize(
                (roi.width, roi.height), Image.LANCZOS))
        hole = mask > 127
        # The plate is the reference for ALL THREE corrections: the generator
        # returns the whole crop re-toned (colour pair), and its content does
        # not continue the plate's at the rim (membrane). Measured 2026-08-14:
        # no generation-side arm moved the rim gradient below 2x the plate's
        # own statistics; the membrane took it to ~1.0 with fill texture
        # preserved.
        fill = match_reference_colour(fill, guide, hole)
        fill = neutralize_fill_cast(fill, hole, reference=guide,
                                    band_px=args.cast_band_px)
        fill = membrane_blend(fill, guide, hole)
        crops.append(fill)
        masks.append(hole)
        kept.append(roi)
    if not crops:
        print("no generated fills to composite")
        return 0

    intr = solve.camera.intrinsics
    base = render_disocclusion_sequence(
        solve, source, [view], resolution=max(int(intr.image_width),
                                              int(intr.image_height)),
        hole_dilate_px=args.hole_dilate_px)[0][0]
    patched = composite_crops(base, crops, kept, masks=masks,
                              feather_px=args.feather_px)
    patch_dir = out_dir / "patch"
    patch_dir.mkdir(parents=True, exist_ok=True)
    frame_path = patch_dir / "patch_view.png"
    Image.fromarray(patched).save(frame_path)
    # Re-entry sidecar: this IS the patch camera (same convention as
    # cmd_occlusion_fill), so AtlasAddPatchView can place it without guessing.
    (patch_dir / "patch_exact.txt").write_text(
        f"azimuth_deg={d_az:.4f} elevation_deg={d_el:.4f} "
        f"distance_scale={d_dist:.4f}\n", encoding="utf-8")
    report["patch_view"] = {
        "path": str(frame_path),
        "rois_composited": len(crops),
        "azimuth_deg": d_az, "elevation_deg": d_el,
        "distance_scale": d_dist,
    }
    (out_dir / "hole_crop_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"patch view: {frame_path} ({len(crops)} ROI fill(s) composited)")
    print("  feed it through AtlasAddPatchView with exact_view_override from "
          "patch_exact.txt — the fill then projects onto the geometry and "
          "every frame of the move gets it without another generation")
    return 0


def cmd_validate(args) -> int:
    from atlas_camera.exporters.dynamic_plate_package import load_dynamic_plate

    package_dir = Path(args.package)
    plate = load_dynamic_plate(package_dir)
    generated = sorted((package_dir / "generated").glob("frame_*.png"))
    frame_paths = generated if generated else None
    issues = validate_dynamic_plate(plate, package_dir=package_dir,
                                    frame_paths=frame_paths)
    fails = _print_issues(issues)
    if fails:
        print(f"FAILED: {fails} blocking issue(s)")
        return 1
    print(f"{plate.plate_id}: valid ({len(generated)} generated frame(s))")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m atlas_camera.dynamic",
                                     description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Build a DynamicPlate package")
    create.add_argument("--image", required=True, help="Source still image")
    matte_group = create.add_mutually_exclusive_group(required=True)
    matte_group.add_argument("--matte", help="Artist matte (white = dynamic)")
    matte_group.add_argument(
        "--auto-matte", metavar="CONCEPTS",
        help="Optional SAM3 assist, e.g. 'ocean, sea water'")
    create.add_argument("--type", default="water",
                        choices=sorted(DYNAMIC_REGION_TYPES))
    create.add_argument("--solve", help="atlas_solve.json (else recover)")
    create.add_argument("--method", default="vanishing_points",
                        choices=["vanishing_points", "learned"],
                        help="atlas.recover method when no --solve is given")
    create.add_argument("--out", required=True, help="Shot output directory")
    create.add_argument("--generator", default="none",
                        choices=["none", "ltx"])
    create.add_argument("--fps", type=float, default=24.0)
    create.add_argument("--frames", type=int, default=96)
    create.add_argument("--seed", type=int, default=None)
    create.add_argument("--prompt", default=WATER_PROMPT_DEFAULT)
    create.add_argument("--overscan-frac", type=float, default=0.10)
    create.add_argument("--overscan-px", type=int, default=0)
    create.add_argument("--plane-height", type=float, default=0.0)
    create.add_argument("--max-distance", type=float, default=500.0)
    create.add_argument("--feather-px", type=float, default=0.0)
    create.add_argument("--blender", action="store_true",
                        help="Write blender_open_scene.py into the package")
    create.add_argument("--host", default=None, help="ComfyUI host:port")
    create.add_argument("--template", default=None,
                        help="LTX ComfyUI workflow template JSON")
    create.add_argument("--mode", default="i2v", choices=["i2v", "v2v"],
                        help="v2v renders the crop along an Atlas dolly first "
                             "so camera motion is geometrically correct")
    create.add_argument("--dolly", default="0.4,0.0,-0.4",
                        help="v2v world-space dolly 'dx,dy,dz' in metres")
    create.add_argument("--gen-width", type=int, default=None,
                        help="generator inference width (default: ROI width; "
                             "keep the ROI aspect and the model's /32 grid)")
    create.add_argument("--gen-height", type=int, default=None,
                        help="generator inference height (default: ROI height)")
    create.add_argument("--render", action="store_true",
                        help="run the v2v render pass inline (staged "
                             "'render' subcommand is the default workflow)")
    create.add_argument("--card", default=None, metavar="PX,PY,DIST,WIDTH",
                        help="OBJECT plate receiver: billboard card at "
                             "DIST m along the ray through pixel (PX,PY), "
                             "WIDTH m wide (use with --type actor)")
    create.set_defaults(func=cmd_create)

    render = sub.add_parser(
        "render", help="Stage 2: render the crop along a dolly (v2v input)")
    render.add_argument("--package", required=True)
    render.add_argument("--dolly", default="0.4,0.0,-0.4")
    render.add_argument("--frames", type=int, default=96)
    render.add_argument("--gen-width", type=int, default=None)
    render.add_argument("--gen-height", type=int, default=None)
    render.set_defaults(func=cmd_render)

    generate = sub.add_parser(
        "generate", help="Stage 3: temporal generation over a package")
    generate.add_argument("--package", required=True)
    generate.add_argument("--generator", default="ltx",
                          choices=["none", "ltx"])
    generate.add_argument("--mode", default="i2v", choices=["i2v", "v2v"])
    generate.add_argument("--fps", type=float, default=24.0)
    generate.add_argument("--frames", type=int, default=96)
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--prompt", default=WATER_PROMPT_DEFAULT)
    generate.add_argument("--host", default=None)
    generate.add_argument("--template", default=None)
    generate.add_argument("--gen-width", type=int, default=None)
    generate.add_argument("--gen-height", type=int, default=None)
    generate.add_argument("--matte-mode", default="none",
                          choices=["none", "chroma"],
                          help="chroma: key the backdrop into "
                               "generated/matte_*.png (actor plates)")
    generate.set_defaults(func=cmd_generate)

    occ = sub.add_parser(
        "occlusion-fill",
        help="Render disocclusion guide+mask along an orbit, LTX-inpaint it")
    occ.add_argument("--solve", required=True,
                     help="atlas_solve.json WITH scene geometry (relief mesh)")
    occ.add_argument("--image", required=True, help="Source plate image")
    occ.add_argument("--out", required=True)
    occ.add_argument("--orbit", default="12.0,0.0,1.0",
                     help="'d_azimuth_deg,d_elevation_deg,distance_scale' "
                          "final orbit (frame 0 = solved pose)")
    occ.add_argument("--frames", type=int, default=49)
    occ.add_argument("--fps", type=float, default=24.0)
    occ.add_argument("--resolution", type=int, default=1024)
    occ.add_argument("--hole-dilate-px", type=int, default=8)
    occ.add_argument("--seed", type=int, default=None)
    occ.add_argument("--prompt", default="")
    occ.add_argument("--generator", default="none", choices=["none", "ltx"])
    occ.add_argument("--host", default=None)
    occ.add_argument("--template", default=None)
    occ.add_argument("--exr", action="store_true",
                     help="also write filled frames as 32f EXR "
                          "(display-referred container)")
    occ.set_defaults(func=cmd_occlusion_fill)

    hcf = sub.add_parser(
        "hole-crop-fill",
        help="Cluster the move's holes and fill each ROI at plate resolution")
    hcf.add_argument("--solve", required=True,
                     help="atlas_solve.json WITH scene geometry (relief mesh)")
    hcf.add_argument("--image", required=True, help="Source plate image")
    hcf.add_argument("--out", required=True)
    hcf.add_argument("--orbit", default="12.0,0.0,1.0",
                     help="'d_azimuth_deg,d_elevation_deg,distance_scale' "
                          "final orbit (frame 0 = solved pose)")
    hcf.add_argument("--frames", type=int, default=49,
                     help="LTX wants 8n+1 (49, 97)")
    hcf.add_argument("--fps", type=float, default=24.0)
    hcf.add_argument("--survey-resolution", type=int, default=1024,
                     help="long edge of the cheap pass that LOCATES holes; "
                          "the fills themselves always run at plate "
                          "resolution")
    hcf.add_argument("--hole-dilate-px", type=int, default=8)
    hcf.add_argument("--pad-frac", type=float, default=0.15,
                     help="context margin around each hole cluster")
    hcf.add_argument("--min-area-px", type=int, default=64,
                     help="survey-raster hole area below which a cluster is "
                          "dropped (and reported)")
    hcf.add_argument("--max-rois", type=int, default=4,
                     help="largest N clusters by area; the rest are dropped "
                          "and LOGGED, never silently ignored")
    hcf.add_argument("--ignore-artist-rois", action="store_true",
                     help="ignore fill regions drawn in the Atlas viewport "
                          "and select clusters automatically instead")
    hcf.add_argument("--feather-px", type=int, default=6,
                     help="composite blend width, ramped OUTSIDE the hole "
                          "(inward would expose the inpaint sentinel)")
    hcf.add_argument("--cast-band-px", type=int, default=48,
                     help="annulus width used to measure the generator's "
                          "colour cast against real plate pixels")
    hcf.add_argument("--depth-scale-ref", type=float, default=0.0,
                     help="metres at which a fill is generated 1:1; ROIs whose "
                          "context sits FARTHER are generated proportionally "
                          "smaller and resampled back, because plate detail "
                          "per world-metre falls as 1/depth (0 = always 1:1)")
    hcf.add_argument("--min-gen-scale", type=float, default=0.25,
                     help="floor on the depth-derived generation scale")
    hcf.add_argument("--max-gen-long-edge", type=int, default=0,
                     help="hard cap on the GENERATION raster's long edge "
                          "(the model's budget, not the renderer's); 0 = off")
    hcf.add_argument("--oversize", default="decline",
                     choices=["decline", "tiled"],
                     help="what to do with clusters past --max-roi-long-edge: "
                          "decline by name (default) or keep them for tiled "
                          "native generation (pass a tiled template)")
    hcf.add_argument("--max-roi-long-edge", type=int, default=0,
                     help="decline clusters whose plate-resolution ROI is "
                          "longer than this (0 = off). The generator's raster "
                          "budget, not the renderer's, is what bounds a 1:1 "
                          "fill; oversized clusters are LOGGED, not downscaled")
    hcf.add_argument("--snap", type=int, default=64,
                     help="model raster grid: /32 always, /64 when an adapter "
                          "declares reference_downscale_factor 2.0")
    hcf.add_argument("--seed", type=int, default=None)
    hcf.add_argument("--prompt", default="",
                     help="CONTENT only — naming camera motion makes the "
                          "model obey the prompt and discard the guide")
    hcf.add_argument("--generator", default="none", choices=["none", "ltx"])
    hcf.add_argument("--host", default=None)
    hcf.add_argument("--template", default=None)
    hcf.set_defaults(func=cmd_hole_crop_fill)

    validate = sub.add_parser("validate", help="Validate a plate package")
    validate.add_argument("--package", required=True)
    validate.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
