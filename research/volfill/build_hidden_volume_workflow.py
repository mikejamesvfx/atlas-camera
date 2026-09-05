"""Emit a multi-viewport ComfyUI workflow for eyeballing VolFill hidden geometry.

One lane per test plate:

    AtlasLoadRAW / LoadImage
        -> AtlasLearnedSolveFromImage      (MEASURED intrinsics on RAW lanes)
        -> AtlasLoadHiddenVolume           (raymarch extraction, gates on)
        -> AtlasRetopologizeLayer          (decimate + smooth)
        -> AtlasBlockoutViewport
        -> AtlasExportNuke                 (RAW lanes: + redistort ST map)

so every plate gets its OWN viewport and they can be compared side by side in
one graph, and the RAW lanes additionally demonstrate the delivery round trip.

Findings baked into the wiring, each measured (see
docs/research/FLASH3D_VOLFILL_ATLAS_EVALUATION.md):
  * RAW lanes carry MEASURED focal + ORIENTED sensor width. sh001 is PORTRAIT,
    so its sensor width is the 15.6 mm short edge; the unoriented 23.5 mm put
    fx 34% out (4120 px against a surveyed 6207 px).
  * extraction="raymarch": pairs the unsigned field's two shell walls into their
    midpoint. Single-sided, correctly placed, and the layers are Atlas's own
    layered-ray form. marching_cubes is legacy.
  * max_faces=0: extract at FULL field resolution and let the retopo node's
    quadric decimation reduce it. Grid striding destroys sub-voxel detail.
  * Gates: A2 divergence (0.82 inspect / 0.88 refuse, validated on 26 held-out
    volumes) and A6 empty-volume floor (an empty volume scores 0% invented and
    used to pass looking sound).

Reuses ``tools/build_example_workflow.Builder`` rather than hand-authoring JSON:
the UI format is redundantly linked (top-level ``links`` plus each node's
``inputs[].link``/``outputs[].links`` must agree) and ``widgets_values`` is
POSITIONAL, so hand-editing is how the shipped set acquired its drift bugs.

The Builder reads widget order from a LIVE server's ``/object_info``. The new
experimental node is not on the server until ComfyUI restarts with
ATLAS_EXPERIMENTAL=1, so its spec is SYNTHESIZED from the class here and injected
into that dict. Same source of truth (the class), no restart required to build.

Usage (Atlas env, ComfyUI running on 8188):
    python build_hidden_volume_workflow.py --out out/atlas_volfill_review.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from build_example_workflow import Builder, fetch_object_info  # noqa: E402

#: (lane label, source, volume dir, note). ``source`` is a RAW path when one
#: exists — the measured intrinsics beat an estimate, and the RAW importer also
#: applies the lensfun undistortion the geometry is solved in.
RAW_ROOT = "C:/Users/miike/ComfyUI_V91/ComfyUI/input/CameraRaw"

#: Measured on 2026-08-15 by AtlasLoadRAW from each file's own EXIF + lensfun.
#: Mirrored here so the solve node SHOWS the numbers rather than depending on a
#: silent hand-off; they must match what the RAW node reports at run time.
#: sensor_mm is ORIENTED to the frame: a portrait plate needs the sensor's
#: SHORT edge, because every consumer divides it by the image width. sh001 is
#: portrait, so 15.6 not 23.5 — the unoriented value put its fx 34% out
#: (4120 px against a measured 6207 px). `resolve_sensor_size` now transposes
#: automatically and warns; these mirror what it returns.
RAW_INTRINSICS = {
    f"{RAW_ROOT}/sh004/DSCF3931.RAF": {"focal_mm": 20.60, "sensor_mm": 23.50,
                                       "fov_x_deg": 59.40},   # landscape
    f"{RAW_ROOT}/sh001/DSCF3915.RAF": {"focal_mm": 18.70, "sensor_mm": 15.60,
                                       "fov_x_deg": 45.28},   # PORTRAIT
}
LANES = [
    ("rusty boiler 8m band", f"{RAW_ROOT}/sh004/DSCF3931.RAF", "out/fov_machine",
     "RAW+undistort | MEASURED fov 59.4deg | 8m band (knee) | 3.0cm voxel | "
     "visF 0.37 | 65% invented -> PASS"),
    ("sh001 street", f"{RAW_ROOT}/sh001/DSCF3915.RAF", "out/fov_sh001",
     "RAW+undistort | MEASURED fov 45.3deg (PORTRAIT sensor 15.6mm) | 5.7cm | "
     "visF 0.43 | 67% invented -> PASS | the REAL-TRUTH plate"),
    ("ghost town", "vf_ghosttown.png", "out/ghosttown",
     "no RAW (AI plate) | 20.5cm voxel | visF 0.38 | 66% invented -> PASS"),
    ("portal", "vf_portal.png", "out/portal",
     "no RAW | 4.8cm voxel | visF 0.13 | 88.5% invented -> REFUSE at the gate "
     "(set on_divergence=mark to see it)"),
    ("coastal alley CONTROL", "vf_coastal_alley.png", "out/coastal_alley",
     "THE CONTROL | visF 0.01 | 99.6% invented -> hard REFUSE. If this looks "
     "as good as lane 1, the gates are measuring nothing"),
]

LANE_DY = 1100

#: Which side of the foreground band holds the recovered layers, MEASURED per
#: plate — it is not a constant. The same band caught 100% of the boiler's
#: cleared selection un-inverted and 0% of sh001's; sh001 needs the complement.
#: The node's report prints both coverage numbers, so re-measure if a plate
#: changes rather than trusting this table.
INVERT_RESTRICT = {
    "out/fov_machine": False,
    "out/fov_sh001": True,
}
VIEWPORT_RES = 1024


def synth_hidden_volume_spec() -> dict:
    """Build an /object_info-shaped entry from the node class itself."""
    from atlas_camera.comfy.nodes_hidden_volume import AtlasLoadHiddenVolume as N

    t = N.INPUT_TYPES()
    return {
        "input": {"required": t.get("required", {}), "optional": t.get("optional", {})},
        "output": list(N.RETURN_TYPES),
        "output_name": list(N.RETURN_NAMES),
        "output_is_list": [False] * len(N.RETURN_TYPES),
        "name": "AtlasLoadHiddenVolume",
        "display_name": "Atlas Load Hidden Volume",
        "category": N.CATEGORY,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--out", default="out/atlas_volfill_review.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--show", default="all",
                    choices=["all", "invented_only", "visible_only"])
    ap.add_argument("--max-faces", type=int, default=0,
                    help="0 = extract at FULL field resolution and let retopo reduce.")
    ap.add_argument("--target-verts", type=int, default=90000)
    ap.add_argument("--smooth", type=int, default=2)
    ap.add_argument("--gate", type=float, default=0.88,
                    help="A2 divergence gate, REFUSE level. 0.88 is the "
                         "held-out-validated value (with 0.82 inspect); a single "
                         "0.85 threshold scored only 88.5%% on the same 26 "
                         "volumes, with every error between the two.")
    ap.add_argument("--min-coverage", type=float, default=0.02,
                    help="A6 empty-volume floor: min share of rays that must hit "
                         "a surface. An empty volume scores 0%% invented and would "
                         "otherwise pass the divergence gate looking sound.")
    ap.add_argument("--on-divergence", default="mark",
                    choices=["refuse", "mark", "allow"],
                    help="'mark' for a review graph so the failure lane still "
                         "renders and can be SEEN; 'refuse' for production.")
    args = ap.parse_args()

    oi = fetch_object_info(args.server)
    oi["AtlasLoadHiddenVolume"] = synth_hidden_volume_spec()

    here = Path(__file__).resolve().parent
    b = Builder(oi)

    for i, (label, image, vol_dir, note) in enumerate(LANES):
        y = i * LANE_DY
        vol_abs = str((here / vol_dir).resolve())

        is_raw = str(image).lower().endswith((".raf", ".nef", ".cr2", ".cr3",
                                              ".arw", ".dng"))
        img_out = "image" if is_raw else "IMAGE"
        if is_raw:
            # AtlasLoadRAW, not LoadImage: it decodes, applies the lensfun
            # geometry correction, and hands back the MEASURED focal and sensor
            # width. Letting the solver estimate intrinsics it could simply read
            # was the single biggest defect in the first build of this graph.
            b.add(f"img{i}", "AtlasLoadRAW",
                  widgets={"file_path": image, "undistort": True,
                           "write_exr": False},
                  pos=(0, y), title=f"{label} — RAW (undistort + measured intrinsics)")
        else:
            b.add(f"img{i}", "LoadImage", widgets={"image": image},
                  pos=(0, y), title=f"{label} — plate")
        # MoGe, deliberately: the volumes were conditioned on MoGe-v2 metric
        # geometry, so solving with the SAME depth model puts solve and volume
        # in one metric space and `depth_scale = 1.0` is correct. Solving with
        # DA-V2 instead would need a per-plate scale measured against it.
        solve_widgets = {"depth_model": "Ruicheng/moge-2-vitl-normal"}
        if is_raw:
            # MEASURED, not estimated. The X-H2 is APS-C: leaving the default
            # 36.0 mm full-frame sensor width in place was wrong by 1.53x.
            solve_widgets["sensor_width_mm"] = RAW_INTRINSICS[image]["sensor_mm"]
            solve_widgets["focal_length_mm"] = RAW_INTRINSICS[image]["focal_mm"]
        b.add(f"solve{i}", "AtlasLearnedSolveFromImage",
              widgets=solve_widgets,
              pos=(420, y),
              title=(f"{label} — solve (MoGe; intrinsics MEASURED from RAW)"
                     if is_raw else f"{label} — solve (MoGe, matches the volume)"))
        # RAYMARCH extraction: pairs the unsigned field's two shell walls into
        # their midpoint, so the surface is placed correctly and comes out
        # single-sided and oriented — no double_sided workaround, and the layers
        # are Atlas's own layered-ray form. max_faces/double_sided apply to the
        # legacy marching-cubes path only.
        # DEPTH: Atlas's own, and on the SAME model the volume was conditioned
        # on (MoGe-v2). Matching the model is what makes depth_scale=1.0 correct
        # — with DA-V2 here the two would sit in different metric spaces and the
        # hidden layers would be selected against a mis-scaled visible surface.
        b.add(f"depth{i}", "AtlasDepthMap",
              widgets={"depth_model": "Ruicheng/moge-2-vitl-normal"},
              pos=(840, y + 340),
              title=f"{label} — depth (MoGe, matches the volume)")

        # RESTRICT: the foreground band's layer_mask. Substitution and gap
        # diffusion are bounded to it; handed a whole frame, fill_hidden_gaps
        # turns a small real selection into total replacement (measured on
        # sh001: 0.9% selected became 100% substituted).
        b.add(f"band{i}", "AtlasDepthLayerMask",
              pos=(840, y + 560),
              title=f"{label} — foreground band (restrict region)")

        b.add(f"vol{i}", "AtlasLoadHiddenVolume",
              widgets={"volume_path": vol_abs, "threshold": args.threshold,
                       "show": args.show, "max_faces": args.max_faces,
                       "name": f"volfill_{i:02d}", "extraction": "raymarch",
                       # 'combined' now that depth + restrict_mask are wired:
                       # the marched layers go through select_hidden_surface and
                       # come back as ONE surface carrying photographed and
                       # inferred geometry together. Switch to "layers" to see
                       # the raw per-layer prediction instead.
                       "emit": "combined",
                       "max_invented_fraction": args.gate,
                       "on_divergence": args.on_divergence,
                       "min_surface_coverage": args.min_coverage,
                       "invert_restrict_mask": INVERT_RESTRICT.get(vol_dir, False)},
              pos=(840, y), title=f"{label} — hidden volume\n{note}")
        b.add(f"retopo{i}", "AtlasRetopologizeLayer",
              widgets={"method": "decimate",
                       "target_vertex_count": args.target_verts,
                       "smooth_iterations": args.smooth},
              pos=(1300, y),
              title=f"{label} — retopo {args.target_verts // 1000}k + smooth {args.smooth}")
        b.add(f"view{i}", "AtlasBlockoutViewport",
              widgets={"resolution": VIEWPORT_RES},
              pos=(1760, y), title=f"{label} — VIEWPORT")

        b.link(f"img{i}", img_out, f"solve{i}", "image")
        if is_raw:
            # raw_meta is the designed channel for measured intrinsics and also
            # carries the lens/sensor provenance the trust tier reports on.
            # focal/sensor are WIDGETS on the solve (not link inputs), so they
            # are set below from the same measurement — belt and braces, and
            # visible on the node face where an artist can check them.
            b.link(f"img{i}", "raw_meta", f"solve{i}", "raw_meta")
        # By OUTPUT NAME, not by type: AtlasLearnedSolveFromImage's first
        # output is named "solve". It was written here as "ATLAS_SOLVE" (the
        # type) and every run of this script died on the first link, which is
        # how a research builder rots quietly — nothing imports it, so no test
        # catches it. `Builder.link` prints the real names on a miss.
        b.link(f"solve{i}", "solve", f"vol{i}", "solve")
        b.link(f"img{i}", img_out, f"depth{i}", "image")
        b.link(f"solve{i}", "solve", f"depth{i}", "solve")
        b.link(f"solve{i}", "solve", f"band{i}", "solve")
        b.link(f"depth{i}", "depth", f"band{i}", "depth")
        b.link(f"depth{i}", "depth", f"vol{i}", "depth")
        b.link(f"band{i}", "layer_mask", f"vol{i}", "restrict_mask")
        b.link(f"vol{i}", "solve", f"retopo{i}", "solve")
        b.link(f"retopo{i}", "solve", f"view{i}", "solve")
        b.link(f"img{i}", img_out, f"view{i}", "source_image")

        # DELIVERY, on the RAW lanes only. Undistorting on import is a one-way
        # door: the solve and every derived mesh live in rectilinear space, so
        # the comp needs the inverse to land back on the ORIGINAL distorted
        # plate. Feeding raw_meta makes the export write redistort_stmap.exr
        # beside the .nk and record it in the manifest.
        if is_raw:
            b.add(f"nuke{i}", "AtlasExportNuke",
                  widgets={"output_dir": f"atlas_exports/volfill_lane{i}",
                           "write_redistort_stmap": True},
                  pos=(2220, y),
                  title=f"{label} — Nuke + redistort ST map")
            b.link(f"retopo{i}", "solve", f"nuke{i}", "solve")
            b.link(f"img{i}", "raw_meta", f"nuke{i}", "raw_meta")

    wf = b.build()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(wf, indent=2), encoding="utf-8")
    print(f"{len(LANES)} lanes -> {out}")
    print("Load in ComfyUI AFTER restarting with ATLAS_EXPERIMENTAL=1.")


if __name__ == "__main__":
    main()
