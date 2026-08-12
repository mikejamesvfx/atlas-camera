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
    views = dolly_view_matrices(plate.crop_camera, offset=dolly,
                                frame_count=frames)
    rendered_dir = package_dir / "rendered"
    rendered_dir.mkdir(exist_ok=True)
    for index, (rgb, alpha) in enumerate(
            render_crop_sequence(plate, views, crop_rgb)):
        # disocclusion at the frame edge falls back to the still crop so
        # the generator sees plausible pixels, never black
        mask = np.asarray(alpha)[..., None] > 0.5
        frame = np.where(mask, np.asarray(rgb), crop_rgb).astype(np.uint8)
        out_im = Image.fromarray(frame)
        if gen_width and gen_height:
            # generator runs at a reduced raster (same aspect): resize the
            # rendered input so the whole v2v pass stays at gen res
            out_im = out_im.resize((gen_width, gen_height), Image.LANCZOS)
        out_im.save(rendered_dir / f"frame_{index:04d}.png")
    plate.metadata["rendered_input"] = {
        "mode": "v2v", "dolly_m": list(dolly), "frame_count": frames,
        "gen_size": [gen_width, gen_height] if gen_width else None,
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
    generate.set_defaults(func=cmd_generate)

    validate = sub.add_parser("validate", help="Validate a plate package")
    validate.add_argument("--package", required=True)
    validate.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
