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
    print("no --solve given; recovering camera from the image "
          "(atlas.recover)...")
    import atlas
    solve = atlas.recover(args.image)
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


def cmd_create(args) -> int:
    import numpy as np
    from PIL import Image

    from atlas_camera.dynamic.generators import (
        TemporalGenerationConfig,
        resolve_generator,
    )
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

    frame_paths = None
    if args.generator != "none":
        generator = resolve_generator(args.generator)
        if args.host and hasattr(generator, "host"):
            generator.host = args.host
        if args.template and hasattr(generator, "template_path"):
            generator.template_path = args.template
        config = TemporalGenerationConfig(
            prompt=args.prompt, seed=args.seed, fps=args.fps,
            frame_count=args.frames)
        gen_result = generator.generate(plate, result.package_dir, config)
        print(f"generator {generator.name}: status = {gen_result.status}")
        for warning in gen_result.warnings:
            print(f"  warning: {warning}")
        (result.package_dir / "generated" / "generation_result.json").write_text(
            json.dumps(gen_result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        if gen_result.status == "ok":
            plate.status = PLATE_STATUS_GENERATED
            plate.frame_end = plate.frame_start + gen_result.frame_count - 1
            frame_paths = gen_result.frame_paths
        elif gen_result.status == GENERATOR_NOT_AVAILABLE:
            plate.metadata["generator_status"] = GENERATOR_NOT_AVAILABLE
        else:
            plate.status = PLATE_STATUS_FAILED
            plate.metadata["generator_status"] = gen_result.status
        # refresh the manifest with the post-generation status
        build_dynamic_plate_package(
            plate, dynamic_dir, source_image_path=image_path, matte=matte)

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
    create.set_defaults(func=cmd_create)

    validate = sub.add_parser("validate", help="Validate a plate package")
    validate.add_argument("--package", required=True)
    validate.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
