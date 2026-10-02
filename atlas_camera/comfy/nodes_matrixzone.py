"""matrixZone SDR->HDR: split an 8K plate into exact crops, restitch the HDR.

``AtlasMatrixZoneSplit`` turns one plate into a ComfyUI LIST of clips -- the
whole-frame global clip first, then the zones -- so the LTX-2.5 SDR->HDR chain
downstream runs once per element (ComfyUI maps a list through every node that
does not ask for the whole list). ``AtlasMatrixZoneStitch`` takes the chain's
``hdr_linear`` LIST back, anchors every zone's low-frequency radiance to the
global pass, feathers the zones together in log2 space and writes one
full-resolution half-float EXR. All the arithmetic is ``core.matrixzone``.

Two sequence modes (the proposal's gate 5 decides the default):

* ``per_zone_clip`` -- each zone repeated ``clip_frames`` times; one LTX run per
  zone. Cross-zone agreement comes only from the radiance anchor.
* ``zones_as_frames`` -- the zones ARE the frames, serpentine so consecutive
  frames are spatial neighbours, ping-ponged to 8k+1; one LTX run, whose
  temporal attention sees every zone.
"""

from __future__ import annotations

from typing import Any

from atlas_camera.comfy.node_helpers import _require_numpy, _require_torch

MODES = ("per_zone_clip", "zones_as_frames")


def _frames_8k1(n: int) -> int:
    from atlas_camera.core.matrixzone import frames_8k1
    return frames_8k1(max(1, int(n)))


class AtlasMatrixZoneSplit:
    """🔲 Split a plate into matrixZone crops (+ a global clip) for LTX SDR->HDR.

    Output ``clips`` is a LIST: element 0 is the whole padded render at a
    quarter linear (the radiance anchor), then the zones. Wire it straight
    into the LTX SDR->HDR chain; wire ``matrixzone`` into AtlasMatrixZoneStitch.
    """

    RETURN_TYPES = ("IMAGE", "ATLAS_MATRIXZONE", "STRING")
    RETURN_NAMES = ("clips", "matrixzone", "report")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "split"
    CATEGORY = "Atlas Camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The full-resolution SDR plate."}),
            },
            "optional": {
                "grid_cols": ("INT", {"default": 2, "min": 1, "max": 8}),
                "grid_rows": ("INT", {"default": 2, "min": 1, "max": 8}),
                "overlap_min_px": ("INT", {
                    "default": 64, "min": 0, "max": 512, "step": 8,
                    "tooltip": "Minimum interior overlap per seam; the 64-clean zone "
                               "size usually gives more (128 on UHD 8K 2x2)."}),
                "mode": (list(MODES), {"default": "per_zone_clip"}),
                "clip_frames": ("INT", {
                    "default": 9, "min": 1, "max": 97, "step": 8,
                    "tooltip": "per_zone_clip: frames per zone clip (8k+1). The global "
                               "clip uses the same count."}),
            },
        }

    def split(self, image, grid_cols=2, grid_rows=2, overlap_min_px=64,
              mode="per_zone_clip", clip_frames=9):
        torch = _require_torch()
        from atlas_camera.core.matrixzone import plan_still, zones_as_frames

        img = image[:1].float()
        _, h, w, _ = img.shape
        plan = plan_still(int(w), int(h), (int(grid_cols), int(grid_rows)),
                          overlap_min=(int(overlap_min_px), int(overlap_min_px)))
        L, T, R, B = plan["render"]["overscan"]
        chw = img.permute(0, 3, 1, 2)
        if any((L, T, R, B)):
            chw = torch.nn.functional.pad(chw, (L, R, T, B), mode="replicate")
        render = chw.permute(0, 2, 3, 1)
        gw, gh = plan["global"]["size"]
        glob = torch.nn.functional.interpolate(chw, size=(gh, gw), mode="area").permute(0, 2, 3, 1)
        n = _frames_8k1(clip_frames)
        clips = [glob.repeat(n, 1, 1, 1)]
        zones = []
        for z in plan["zones"]:
            x, y, zw, zh = z["renderRect"]
            zones.append(render[:, y:y + zh, x:x + zw, :])
        frame_of_zone = None
        if mode == "zones_as_frames":
            frames, frame_of_zone = zones_as_frames([zt[0].cpu().numpy() for zt in zones], plan)
            clips.append(torch.from_numpy(frames).to(render.dtype))
        else:
            clips.extend(zt.repeat(n, 1, 1, 1) for zt in zones)
        handle = {"kind": "atlas_matrixzone", "plan": plan, "mode": mode,
                  "clip_frames": n, "frame_of_zone": frame_of_zone, "n_global": 1}
        zw, zh = plan["zone_size"]
        report = (f"AtlasMatrixZoneSplit: plate {w}x{h} -> render "
                  f"{plan['render']['width']}x{plan['render']['height']} "
                  f"(plate at {plan['render']['plateOrigin']}), grid {grid_cols}x{grid_rows}, "
                  f"zones {zw}x{zh}, overlap {plan['overlap']['px']} px; global {gw}x{gh}; "
                  f"mode {mode}: {len(clips)} clip(s) -> "
                  + (f"{len(zones)} zones x {n} frames + global" if mode == "per_zone_clip"
                     else f"one {len(clips[1])}-frame zone sequence (scan {plan['scan']}) + global"))
        return (clips, handle, report)


def _tonemap_preview(np, lin):
    x = lin / (1.0 + lin)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


class AtlasMatrixZoneStitch:
    """🔲 Restitch the zones' HDR into one full-resolution EXR.

    ``hdr`` is the LTX chain's ``hdr_linear`` LIST (global first, then the
    zones, in AtlasMatrixZoneSplit's order). Each still zone's clip is reduced
    to one image (median over its frames). With ``anchor`` on, every zone's
    low-frequency log2 radiance is replaced by the global pass's, so zones
    cannot disagree about how bright a region is; the zones then only carry
    detail. Seams are feathered in log2 space and the padding cropped away.
    The EXR is a MODEL RECONSTRUCTION of highlight radiance from a display-
    referred plate, and the report says so.
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("preview", "exr_path", "report")
    INPUT_IS_LIST = True
    FUNCTION = "stitch"
    CATEGORY = "Atlas Camera"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hdr": ("IMAGE", {"tooltip": "LTX hdr_linear (ACEScg linear), as a list."}),
                "matrixzone": ("ATLAS_MATRIXZONE",),
            },
            "optional": {
                "anchor": ("BOOLEAN", {"default": True,
                                       "tooltip": "Take each zone's low frequencies from "
                                                  "the global pass."}),
                "split_px": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8,
                                     "tooltip": "Anchor low-pass scale in render px. "
                                                "0 = the seam overlap."}),
                "colorspace": ("STRING", {"default": "ACEScg",
                                          "tooltip": "Tag for the EXR (LTX hdr_linear is "
                                                     "ACEScg linear)."}),
                "filename_prefix": ("STRING", {"default": "atlas/hdr_plate"}),
            },
        }

    def stitch(self, hdr, matrixzone, anchor=True, split_px=0, colorspace="ACEScg",
               filename_prefix="atlas/hdr_plate"):
        # INPUT_IS_LIST: ComfyUI hands every input as a list; tests and direct
        # callers may pass scalars. first() accepts both.
        np = _require_numpy()
        torch = _require_torch()
        from atlas_camera.core.matrixzone import stitch as mz_stitch

        handle = matrixzone[0] if isinstance(matrixzone, list) else matrixzone
        first = lambda v: v[0] if isinstance(v, list) else v  # noqa: E731
        anchor, split_px = bool(first(anchor)), int(first(split_px))
        colorspace, filename_prefix = str(first(colorspace)), str(first(filename_prefix))
        plan, mode = handle["plan"], handle["mode"]
        clips = [np.asarray(c.detach().cpu().float().numpy()) if hasattr(c, "detach")
                 else np.asarray(c, dtype=np.float32) for c in (hdr if isinstance(hdr, list) else [hdr])]
        n_global = int(handle.get("n_global", 1))
        expect = n_global + (1 if mode == "zones_as_frames" else len(plan["zones"]))
        if len(clips) != expect:
            raise ValueError(f"AtlasMatrixZoneStitch: got {len(clips)} clip(s), the split "
                             f"made {expect} (is the LTX chain fed the split's clips list?)")
        glob = np.median(clips[0], axis=0) if anchor else None
        if mode == "zones_as_frames":
            seq = clips[1]
            frame_of_zone = handle["frame_of_zone"]
            zones = [seq[min(f, len(seq) - 1)] for f in frame_of_zone]
        else:
            zones = [np.median(c, axis=0) for c in clips[1:]]
        zones = [z[..., :3] for z in zones]
        plate, rep = mz_stitch(zones, plan, global_hdr=None if glob is None else glob[..., :3],
                               split_px=split_px or None)

        exr_path, exr_note = "", ""
        try:
            from atlas_camera.comfy.nodes_scene3d import _output_paths
            from atlas_camera.plate.oiio_io import write_exr
            folder, stem = _output_paths(filename_prefix)
            folder.mkdir(parents=True, exist_ok=True)
            exr_path = str(folder / f"{stem}.exr")
            write_exr(exr_path, plate.astype(np.float32), bit_depth="half",
                      source_colorspace=colorspace,
                      extra_attribs={"atlas:content": "matrixZone SDR->HDR model reconstruction"})
        except Exception as exc:  # noqa: BLE001 - still return the preview + report
            exr_note = f"EXR not written: {type(exc).__name__}: {exc}"

        def fmt(seams):
            return ", ".join(f"{s['seam']} {s['median_stops']:.3f}/{s['p95_stops']:.3f}"
                             for s in seams) or "(no interior seams)"
        lines = [f"AtlasMatrixZoneStitch: {len(zones)} zones ({mode}) -> "
                 f"{plate.shape[1]}x{plate.shape[0]} {colorspace} half EXR "
                 f"{exr_path or '(not written)'}",
                 f"seams before, log2 luminance median/p95 stops: {fmt(rep['seams_before'])}"]
        if rep.get("anchored"):
            lines.append(f"seams after radiance anchor (split {rep['split_px']} px): "
                         f"{fmt(rep['seams_after_anchor'])}")
        else:
            lines.append("radiance anchor OFF: zones keep their own low frequencies")
        if rep.get("resized_zones"):
            lines.append(f"warning: {rep['resized_zones']} zone result(s) came back at a "
                         "different size and were resampled to their renderRect")
        lines.append(f"range: max {float(plate.max()):.2f}, p99 "
                     f"{float(np.percentile(plate, 99)):.3f} (linear); a MODEL RECONSTRUCTION "
                     "of highlight radiance from a display-referred plate, not photographed HDR")
        if exr_note:
            lines.append(exr_note)
        report = "\n".join(lines)
        preview = torch.from_numpy(_tonemap_preview(np, plate).astype(np.float32))[None]
        return {"ui": {"text": [report]}, "result": (preview, exr_path, report)}
