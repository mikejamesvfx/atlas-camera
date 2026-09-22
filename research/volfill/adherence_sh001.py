"""The sh001 anchor arm: adherence measured against a real second photograph.

WHY THIS SCENE. sh001 is the only plate in `plates.json` carrying real
photographed hidden-geometry truth. Frame 1 (DSCF3915) is solved and meshed;
frame 2 (DSCF3916) sits a measured 14.578 m away at a pose registered to 0.1 px.
Rendering frame 1's scene from frame 2's TRUE pose puts render and photograph on
the same pixel grid, which is the whole trick -- and it is `g5_build.py`'s trick,
reused here rather than reinvented.

WHAT HAS TRUTH AND WHAT DOES NOT. Stating this precisely matters more than the
numbers:

- **VALID pixels have deterministic truth at EVERY frame.** They are Atlas's own
  reprojection, so a generated frame can be scored against them anywhere along
  the path without a photograph at all. That is what makes the whole method work
  on plates that have no second frame.
- **GHOST pixels have photographic truth at the LAST frame ONLY**, because that
  is the only pose a camera was actually placed at. Intermediate frames have NO
  ghost answer key, and this script reports them as such rather than
  interpolating one. Anything else would be inventing truth to score invention.

This is a SCRIPT, not a test: the plates live outside the repo. `plates.json`
pointed into a session scratchpad that has since expired and taken DSCF3915,
DSCF3916 and sh001_rig.json with it, so the root now points at a durable
directory (override with ATLAS_PLATES_ROOT). Run with the files restored there.

    python research/volfill/adherence_sh001.py [--resolution 1536] [--frames 8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).resolve().parent / "plates.json"


def resolve_plate(name: str = "sh001_street") -> dict:
    """Resolve a plate entry's paths through the registry's roots."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    roots = {key: value for key, value in registry.items()
             if isinstance(value, str) and not key.startswith("_")
             and key != "plates_root_env"}
    override = os.environ.get(registry.get("plates_root_env", ""), "")
    if override:
        roots["scratchpad"] = override

    entry = next((p for p in registry["plates"] if p["name"] == name), None)
    if entry is None:
        raise SystemExit(f"{name} is not in {REGISTRY}")

    def expand(value: str) -> str:
        for key, root in roots.items():
            value = value.replace("{" + key + "}", root)
        return value

    truth = entry.get("truth") or {}
    return {
        "frame1": expand(entry["path"]),
        "frame2": expand(truth.get("frame2", "")),
        "solve": expand(truth.get("solve", "")),
    }


def require(paths: dict) -> None:
    missing = [f"{key}: {value}" for key, value in paths.items()
               if not value or not Path(value).exists()]
    if not missing:
        return
    raise SystemExit(
        "The sh001 truth assets are not on disk:\n  "
        + "\n  ".join(missing)
        + "\n\nThe registry's root expired with a session temp directory. Put "
          "DSCF3915.png, DSCF3916.png and sh001_rig.json in that directory (or "
          "point ATLAS_PLATES_ROOT at wherever they live) and re-run.\n"
          "Without them this arm cannot run; the deterministic VALID-only "
          "scoring in tests/test_adherence.py still can, and does not need a "
          "photograph at all.")


def build_scene(solve_path: str, frame1_path: str, resolution: int):
    """Solve 1's depth -> relief mesh, attached to the rig solve. From g5_build."""
    from PIL import Image

    from atlas_camera.core import relief_mesh as rm
    from atlas_camera.core.camera_spec import CameraSpec
    from atlas_camera.core.io import load_solve_json
    from atlas_camera.core.schema import AtlasProjectionScene, AtlasProxyPrimitive
    from atlas_camera.inference import depth_estimator as de

    Image.MAX_IMAGE_PIXELS = None
    solve = load_solve_json(solve_path)
    intr = solve.camera.intrinsics
    spec = CameraSpec.from_intrinsics(intr)
    width, height = int(intr.image_width), int(intr.image_height)
    print(f"rig: {width}x{height} fx={spec.fx:.1f}", flush=True)

    # Exterior street scene -> V2-Metric-Outdoor, per the depth doctrine.
    result = de.estimate_depth(
        frame1_path,
        model_id="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
    depth = np.asarray(getattr(result, "depth", result), dtype=np.float64)
    if depth.shape != (height, width):
        depth = np.asarray(
            Image.fromarray(depth.astype(np.float32)).resize(
                (width, height), Image.BILINEAR), dtype=np.float64)
    print(f"depth {depth.shape} "
          f"{float(np.nanmin(depth)):.2f}..{float(np.nanmax(depth)):.2f} m",
          flush=True)

    view = np.asarray(solve.camera.extrinsics.camera_view_matrix)
    mesh = rm.build_relief_mesh(depth, view_matrix=view, fx=spec.fx,
                                fy=spec.fy, cx=spec.cx, cy=spec.cy)
    prim = AtlasProxyPrimitive(
        name="sh001_relief", primitive_type="mesh",
        metadata={
            "vertices": np.asarray(mesh.vertices, np.float64).reshape(-1).tolist(),
            "faces": np.asarray(mesh.faces, np.int64).reshape(-1).tolist(),
            "uvs": np.asarray(mesh.uvs, np.float64).reshape(-1).tolist()})
    if getattr(solve, "projection_scene", None) is None:
        solve.projection_scene = AtlasProjectionScene(proxy_geometry=[prim])
    else:
        solve.projection_scene.proxy_geometry = [prim]
    return solve, spec, width, height


def path_to_frame2(solve, frames: int):
    """A path whose LAST frame is exactly DSCF3916's solved pose.

    Interpolating toward the true pose rather than sampling a preset is the whole
    point: the end of the move is the one camera position a photograph exists
    for, so it must land on it exactly, not near it.
    """
    from atlas_camera.core.camera_math import look_at_view_matrix  # noqa: F401
    from atlas_camera.core.schema import AtlasCameraKeyframe, AtlasCameraPath

    def eye_and_target(view):
        vm = np.asarray(view, dtype=np.float64).reshape(4, 4)
        cam_to_world = np.linalg.inv(vm)
        eye = cam_to_world[:3, 3]
        forward = -cam_to_world[:3, 2]
        return eye, eye + forward * 10.0

    eye0, target0 = eye_and_target(solve.camera.extrinsics.camera_view_matrix)
    src = solve.projection_sources[0].camera
    eye1, target1 = eye_and_target(src.extrinsics.camera_view_matrix)
    baseline = float(np.linalg.norm(eye1 - eye0))
    print(f"baseline to frame 2: {baseline:.3f} m", flush=True)

    return AtlasCameraPath(
        keyframes=[
            AtlasCameraKeyframe(frame_index=0, position=tuple(eye0),
                                target=tuple(target0), easing="linear"),
            AtlasCameraKeyframe(frame_index=frames - 1, position=tuple(eye1),
                                target=tuple(target1), easing="linear"),
        ],
        frame_count=frames), baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=1536)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--out", default="research/volfill/out/adherence_sh001")
    args = parser.parse_args()

    from PIL import Image

    from atlas_camera.core.adherence import gradient_zncc, raw_error
    from atlas_camera.core.camera_path import sample_camera_path_intrinsics
    from atlas_camera.core.conditioning import render_conditioning_sequence
    from atlas_camera.core.ghost_pixels import GhostClass
    from atlas_camera.core.projection_render import gather_scene_meshes

    paths = resolve_plate()
    require(paths)

    solve, spec, width, height = build_scene(
        paths["solve"], paths["frame1"], args.resolution)
    camera_path, baseline = path_to_frame2(solve, args.frames)

    long_edge = int(args.resolution)
    scale = long_edge / float(max(width, height))
    rw, rh = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    sx, sy = rw / float(width), rh / float(height)
    fx, fy = spec.fx * sx, spec.fy * sy
    cx, cy = spec.cx * sx, spec.cy * sy

    with Image.open(paths["frame1"]) as im:
        plate = np.asarray(im.convert("RGB").resize((rw, rh), Image.BILINEAR),
                           dtype=np.float64) / 255.0
    rgba = np.concatenate([plate, np.ones((rh, rw, 1))], axis=-1)

    from atlas_camera.core.camera_path import sample_camera_path
    views = [np.asarray(e.camera_view_matrix)
             for e in sample_camera_path(camera_path)]
    intrinsics = sample_camera_path_intrinsics(
        camera_path, fx=fx, fy=fy, cx=cx, cy=cy, height=rh)

    seq = render_conditioning_sequence(
        gather_scene_meshes(solve, with_uvs=True), {"primary": rgba},
        views=views, intrinsics=intrinsics,
        plate_view=solve.camera.extrinsics.camera_view_matrix,
        plate_k=[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        width=rw, height=rh)

    # GHOST truth exists at the LAST frame only -- the one pose a camera was
    # actually placed at. Say so; do not interpolate an answer key.
    with Image.open(paths["frame2"]) as im:
        truth = np.asarray(im.convert("RGB").resize((rw, rh), Image.BILINEAR),
                           dtype=np.float64) / 255.0

    last = seq.frames - 1
    ghost = seq.class_map[last] == int(GhostClass.GHOST)
    valid = seq.class_map[last] == int(GhostClass.VALID)
    report = {
        "scene": "sh001_street",
        "baseline_m": baseline,
        "raster": [rw, rh],
        "frames": seq.frames,
        "parallax_px": seq.meta["parallax_px"],
        "truth": {
            "valid": "deterministic, EVERY frame (Atlas's own reprojection)",
            "ghost": "photographic, LAST FRAME ONLY (DSCF3916's pose); "
                     "intermediate frames have NO ghost answer key",
        },
        "last_frame": {
            "ghost_px": int(ghost.sum()),
            "valid_px": int(valid.sum()),
            "ghost_truth_gradient_zncc": gradient_zncc(
                np, seq.rgb[last], truth, ghost) if ghost.any() else None,
            "ghost_truth_raw": raw_error(np, seq.rgb[last], truth, ghost)
            if ghost.any() else None,
            "valid_vs_photograph_gradient_zncc": gradient_zncc(
                np, seq.rgb[last], truth, valid),
        },
        "per_frame_class_stats": seq.per_frame,
        "note": "The render's GHOST pixels are holes, so scoring them against "
                "the photograph measures how much real content the move "
                "revealed that Atlas cannot supply -- it is the SIZE of the "
                "generative problem, and the answer key a filled result is "
                "then scored against. Feed a generated arm and its control to "
                "core.adherence.score_arms with this bundle.",
    }

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "adherence_sh001.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(out / "bundle_last_frame.npz",
                        rgb=seq.rgb[last], depth=seq.depth_m[last],
                        class_map=seq.class_map[last], truth=truth)
    print(json.dumps(report["last_frame"], indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
