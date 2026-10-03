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


from atlas_camera.comfy.node_helpers import (
    _require_numpy,
    _require_torch,
    output_paths,
    project_output_paths,
)

MODES = ("per_zone_clip", "zones_as_frames")


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
        from atlas_camera.core.matrixzone import (
            frames_8k1,
            narrowed_overlaps,
            plan_still,
            zones_as_frames,
        )

        img = image[:1].float()
        _, h, w, _ = img.shape
        dropped = int(image.shape[0]) - 1
        plan = plan_still(int(w), int(h), (int(grid_cols), int(grid_rows)),
                          overlap_min=(int(overlap_min_px), int(overlap_min_px)))
        L, T, R, B = plan["render"]["overscan"]
        chw = img.permute(0, 3, 1, 2)
        if any((L, T, R, B)):
            chw = torch.nn.functional.pad(chw, (L, R, T, B), mode="replicate")
        render = chw.permute(0, 2, 3, 1)
        gw, gh = plan["global"]["size"]
        glob = torch.nn.functional.interpolate(chw, size=(gh, gw), mode="area").permute(0, 2, 3, 1)
        n = frames_8k1(max(1, int(clip_frames)))
        # Materialised clips (repeat), NOT zero-copy expand(): a stride-0 batch
        # sends the LTX-2.5 VAE encode down a pathological path -- measured
        # 2026-10-03 on V135 at 1088x1920x9: repeat 1.46 s, expand 166.8 s, and
        # the 2x2 8K HDR still went 14.5 -> 30 min. The ~3.4 GB of host RAM a
        # 2x2 8K, 9-frame split costs is the price of a working encode.
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
        narrowed = narrowed_overlaps(plan)
        if narrowed:
            report += ("\nwarning: clamp narrowed overlap: "
                       + ", ".join(f"{zid} {ax} {a}/{r} px" for zid, ax, a, r in narrowed))
        if dropped > 0:
            report += (f"\nwarning: image batch has {dropped + 1} frames; only the first was "
                       f"split, {dropped} dropped (matrixZone converts one still plate)")
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
                # Kept for saved graphs; the stitch maths is fixed to ACEScg, so
                # anything else is refused (see _check_colorspace).
                "colorspace": ("STRING", {"default": "ACEScg",
                                          "tooltip": "EXR colorspace tag. Must be ACEScg "
                                                     "(aliases: lin_ap1, ACES - ACEScg, "
                                                     "lin_ap1_scene): LTX hdr_linear is "
                                                     "ACEScg and the stitch only produces "
                                                     "ACEScg, so any other value is "
                                                     "refused rather than mis-tagged."}),
                "filename_prefix": ("STRING", {"default": "atlas/hdr_plate"}),
                # APPENDED: destripe against the SDR input (wire the same plate
                # the split got). The conversion adds faint vertical stripes in
                # every zone; measured against its own input they can be divided
                # out without touching the highlights it reconstructed.
                "destripe": ("BOOLEAN", {"default": True,
                                         "tooltip": "Remove vertical stripes the conversion "
                                                    "added (needs sdr_plate)."}),
                "sdr_plate": ("IMAGE", {"tooltip": "The SDR plate the split was given."}),
                # APPENDED: keep the conversion's radiance, take structure from the
                # SDR (guided filter on the log ratio). Removes the model's
                # stripes at every width; needs sdr_plate. Clipped highlights
                # keep the HDR's own pixels.
                "detail_from_sdr": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Rebuild unclipped areas as SDR x the conversion's smoothed "
                               "(edge-aware) radiance ratio: no model stripes, no halos. "
                               "Clipped highlights keep the HDR's pixels. Needs sdr_plate."}),
                # APPENDED: delivery project -> the EXR lands in the shot's plates lane.
                "project": ("ATLAS_PROJECT", {
                    "tooltip": "Optional delivery project from AtlasProject: writes the "
                               "EXR into the shot's plates/ lane (supersedes "
                               "filename_prefix's folder)."}),
            },
        }

    def stitch(self, hdr, matrixzone, anchor=True, split_px=0, colorspace="ACEScg",
               filename_prefix="atlas/hdr_plate", destripe=True, sdr_plate=None,
               detail_from_sdr=True, project=None):
        # INPUT_IS_LIST: ComfyUI hands every input as a list; tests and direct
        # callers may pass scalars. _first() accepts both.
        np = _require_numpy()
        torch = _require_torch()
        from atlas_camera.core.matrixzone import stitch as mz_stitch

        handle = matrixzone[0] if isinstance(matrixzone, list) else matrixzone
        anchor, split_px = bool(_first(anchor)), int(_first(split_px))
        colorspace, filename_prefix = str(_first(colorspace)), str(_first(filename_prefix))
        # Everything that can refuse or fail about the OUTPUT is settled here,
        # before 14-30 min of 8K stitch work: a non-ACEScg tag and a prefix
        # refusal raise now (graph errors); an unwritable lane is recorded and
        # the EXR skipped, but preview + report still come back.
        _check_colorspace(colorspace)
        project = _first(project) if project is not None else None
        out_folder, out_stem, out_note = _resolve_output(filename_prefix, project)
        plan, mode = handle["plan"], handle["mode"]
        clips = _stitch_clips(np, hdr, handle)
        glob = np.median(clips[0], axis=0) if anchor else None
        zones = _zone_images(np, clips, handle)
        plate, rep = mz_stitch(zones, plan, global_hdr=None if glob is None else glob[..., :3],
                               split_px=split_px or None)
        n_zones = len(zones)
        del clips, glob, zones          # free the zone medians before the 8K post-passes
        destripe = bool(_first(destripe))
        detail_from_sdr = bool(_first(detail_from_sdr))
        sdr = _first(sdr_plate) if sdr_plate is not None else None
        sdr_lin, s_disp, sdr_note = _sdr_reference(np, sdr, plate.shape)
        plate, stripe_rep, local_rep, xfer_rep = _clean_plate(
            plate, plan, sdr_lin, s_disp, destripe=destripe, detail_from_sdr=detail_from_sdr)
        del s_disp                      # only sdr_lin is used past the clean-up passes

        from atlas_camera.core.matrixzone import seam_step_test
        step_mem = seam_step_test(plate, plan, sdr_linear=sdr_lin)
        params = _provenance_params(handle, rep, anchor=anchor, destripe=destripe,
                                    detail_from_sdr=detail_from_sdr, sdr_wired=sdr is not None,
                                    local_rep=local_rep, xfer_rep=xfer_rep, stripe_rep=stripe_rep)
        if out_note:
            exr_path, exr_note = "", out_note
        else:
            exr_path, exr_note = _write_plate_exr(np, plate, colorspace, out_folder, out_stem,
                                                  params=params, step_mem=step_mem)
        step, delivered, diff_note = _score_seams(np, plate, plan, sdr_lin, exr_path,
                                                  step_mem=step_mem)

        report = _stitch_report(
            np, plate, rep, step, n_zones=n_zones, mode=mode, colorspace=colorspace,
            exr_path=exr_path, exr_note=exr_note, delivered=delivered, diff_note=diff_note,
            stripe_rep=stripe_rep, local_rep=local_rep, xfer_rep=xfer_rep,
            destripe=destripe, detail_from_sdr=detail_from_sdr, sdr_note=sdr_note)
        preview = torch.from_numpy(_tonemap_preview(np, plate).astype(np.float32))[None]
        return {"ui": {"text": [report]}, "result": (preview, exr_path, report)}


def _first(v):
    """Element 0 of an INPUT_IS_LIST list, or the scalar itself."""
    return v[0] if isinstance(v, list) else v


# Spellings of ACEScg the colorspace widget accepts (compared case-insensitively).
_ACESCG_NAMES = frozenset({"acescg", "lin_ap1", "aces - acescg", "lin_ap1_scene"})


def _check_colorspace(colorspace):
    """Refuse a ``colorspace`` tag that is not ACEScg.

    The widget is only the EXR's tag; the stitch maths (and LTX hdr_linear) is
    ACEScg, so another value would label ACEScg pixels as something they are
    not. Kept as a widget for saved graphs; refused rather than honoured.
    """
    if str(colorspace).strip().lower() not in _ACESCG_NAMES:
        raise ValueError(
            f"AtlasMatrixZoneStitch: colorspace {colorspace!r} is not supported; the "
            "stitch only produces ACEScg (LTX hdr_linear is ACEScg linear). Set it to "
            "'ACEScg' and convert downstream if another space is needed.")


def _resolve_output(filename_prefix, project):
    """``(folder, stem, note)`` for the EXR, resolved BEFORE any stitch work.

    A prefix refusal (ValueError, or ComfyUI's own exception for a prefix that
    leaves the output dir) propagates: it is a graph error. An OSError creating
    the project lane / output folder does NOT raise: it comes back as the
    "EXR NOT WRITTEN" note (report line 1) and the write is skipped.
    """
    try:
        folder, stem = (project_output_paths(project, "plates", filename_prefix)
                        if project is not None else output_paths(filename_prefix))
    except OSError as exc:
        return None, None, f"EXR NOT WRITTEN: {type(exc).__name__}: {exc}"
    return folder, stem, ""


def _stitch_clips(np, hdr, handle):
    """Gate: the hdr LIST as float numpy clips, refused unless it matches the split."""
    plan, mode = handle["plan"], handle["mode"]
    clips = [np.asarray(c.detach().cpu().float().numpy()) if hasattr(c, "detach")
             else np.asarray(c, dtype=np.float32) for c in (hdr if isinstance(hdr, list) else [hdr])]
    n_global = int(handle.get("n_global", 1))
    expect = n_global + (1 if mode == "zones_as_frames" else len(plan["zones"]))
    if len(clips) != expect:
        raise ValueError(f"AtlasMatrixZoneStitch: got {len(clips)} clip(s), the split "
                         f"made {expect} (is the LTX chain fed the split's clips list?)")
    return clips


def _zone_images(np, clips, handle):
    """One RGB image per zone: the zone's frame, or the median over its clip."""
    if handle["mode"] == "zones_as_frames":
        seq = clips[1]
        frame_of_zone = handle["frame_of_zone"]
        need = max(frame_of_zone) + 1 if frame_of_zone else 0
        if len(seq) < need:
            # Never clamp: a clamped index silently stitches another zone's
            # pixels into this zone's place.
            raise ValueError(f"AtlasMatrixZoneStitch: the zone sequence came back with "
                             f"{len(seq)} frame(s); the split's {len(frame_of_zone)} zones need "
                             f"at least {need} (the split sent {handle.get('clip_frames', '?')}) "
                             "-- did the LTX chain trim or subsample frames?")
        zones = [seq[f] for f in frame_of_zone]
    else:
        zones = [np.median(c, axis=0) for c in clips[1:]]
    return [z[..., :3] for z in zones]


def _sdr_reference(np, sdr, plate_shape):
    """``(sdr_linear, sdr_display, note)`` at the plate's raster, or ``(None, None, "")``.

    ``note`` is a warning line when the SDR had to be resampled to the plate.

    ``sdr_linear`` is linear REC.709 on purpose: every core pass that compares
    it with the ACEScg HDR (destripe_columns/destripe_local/seam_step_test via
    AP1 luminance of the Rec.709 pixel, sdr_detail_transfer via a full
    Rec.709->ACEScg convert) moves it into ACEScg itself, and keeps the
    clipped-highlight test on the SDR's own channels. Converting here as well
    would apply the matrix twice.
    """
    if sdr is None:
        return None, None, ""
    from atlas_camera.core.matrixzone import resize_bilinear, srgb_to_linear_f32
    s_np = np.asarray(sdr[0].detach().cpu().float().numpy() if hasattr(sdr, "detach")
                      else sdr[0], dtype=np.float32)[..., :3]
    note = ""
    if s_np.shape[:2] != plate_shape[:2]:
        note = (f"warning: sdr_plate is {s_np.shape[1]}x{s_np.shape[0]}, the stitched plate "
                f"{plate_shape[1]}x{plate_shape[0]}: SDR resampled (bilinear) for destripe/"
                "detail/seam control -- wire the plate the split got")
        s_np = resize_bilinear(s_np, plate_shape[0], plate_shape[1])
    return srgb_to_linear_f32(s_np), s_np, note


def _clean_plate(plate, plan, sdr_lin, s_disp, *, destripe, detail_from_sdr):
    """Destripe and SDR detail transfer, each only when the SDR plate is wired.

    Returns ``(plate, stripe_rep, local_rep, xfer_rep)``; a skipped step's
    report is None.
    """
    stripe_rep, local_rep, xfer_rep = None, None, None
    if destripe and sdr_lin is not None:
        from atlas_camera.core.matrixzone import (
            destripe_columns,
            destripe_local,
            zone_row_bands,
        )
        plate, stripe_rep = destripe_columns(plate, sdr_lin, bands=zone_row_bands(plan))
        plate, local_rep = destripe_local(plate, sdr_lin)
    if detail_from_sdr and s_disp is not None:
        from atlas_camera.core.matrixzone import sdr_detail_transfer
        plate, xfer_rep = sdr_detail_transfer(plate, s_disp)
    return plate, stripe_rep, local_rep, xfer_rep


def _provenance_params(handle, rep, *, anchor, destripe, detail_from_sdr, sdr_wired,
                       local_rep, xfer_rep, stripe_rep):
    """Everything needed to say how this plate was made (EXR ``atlas:matrixzone_params``).

    Booleans record what RAN, not just the widget: destripe with no sdr_plate
    is ``false`` here, with the widget value beside it.
    """
    from atlas_camera.core.matrixzone import (
        DESTRIPE_SDR_CLIP,
        SDR_TRANSFER_CLIP,
        SDR_TRANSFER_EPS,
        SDR_TRANSFER_RADIUS,
    )
    plan = handle["plan"]
    zw, zh = plan["zone_size"]
    long_edge = max(zw, zh)
    tier = "1080p" if long_edge <= 2048 else "4K" if long_edge <= 4096 else "8K"
    return {
        "schema": "atlasMatrixZoneParams/1",
        "grid": list(plan["grid"]), "zone_size": [zw, zh], "zone_tier": tier,
        "render": [plan["render"]["width"], plan["render"]["height"]],
        "plate": [plan["plate"]["width"], plan["plate"]["height"]],
        "mode": handle["mode"], "clip_frames": handle.get("clip_frames"),
        "overlap_px": list(plan["overlap"]["px"]),
        "anchor": bool(rep.get("anchored")), "anchor_widget": bool(anchor),
        "split_px": rep.get("split_px"),
        "sdr_plate": bool(sdr_wired),
        "destripe": stripe_rep is not None, "destripe_widget": bool(destripe),
        "destripe_iterations": None if stripe_rep is None else stripe_rep.get("iterations"),
        "destripe_window_px": None if stripe_rep is None else stripe_rep.get("window_px"),
        "destripe_sdr_clip": DESTRIPE_SDR_CLIP,
        "destripe_local": local_rep is not None,
        "detail_from_sdr": xfer_rep is not None, "detail_from_sdr_widget": bool(detail_from_sdr),
        "sdr_clip_window": list(SDR_TRANSFER_CLIP),
        "guided_radius_px": SDR_TRANSFER_RADIUS if xfer_rep is None else xfer_rep["radius_px"],
        "guided_eps": SDR_TRANSFER_EPS if xfer_rep is None else xfer_rep["eps"],
    }


def _atlas_version():
    try:
        import atlas_camera
        return str(atlas_camera.__version__)
    except Exception:  # noqa: BLE001 - provenance must never fail the write
        try:
            from importlib.metadata import version
            return version("atlas-camera")
        except Exception:  # noqa: BLE001
            return "unknown"


def _seam_worst_attr(step_mem):
    """JSON for ``atlas:seam_worst``: the in-memory score (the EXR cannot carry
    a score of itself; the report has the delivered one)."""
    import json
    w = step_mem.get("worst") if step_mem else None
    if not w:
        return json.dumps({"seam": None, "pass": step_mem.get("pass") if step_mem else None})
    return json.dumps({"seam": w["seam"], "ratio": round(float(w["ratio"]), 4),
                       "step_stops": round(float(w["step_stops"]), 5),
                       "window": w.get("worst_window"), "pass": step_mem["pass"],
                       "sdr_controlled": step_mem["sdr_controlled"],
                       "window_px": step_mem.get("window_px"),
                       "scored_on": "in-memory plate before half/DWAB encode"})


def _write_plate_exr(np, plate, colorspace, folder, stem, *, params=None, step_mem=None):
    """``(exr_path, note)``; a failed write leaves the path empty and says why.

    Provenance attributes: ``atlas:content`` (the model-reconstruction label),
    ``atlas:matrixzone_params`` (JSON, see _provenance_params),
    ``atlas:seam_worst`` (JSON) and ``atlas:version``.

    ``(folder, stem)`` come from _resolve_output, called at the top of
    stitch() so a prefix refusal (a graph error, F-1) fails before the
    compute, not after it. The note becomes report line 1 (F-6): a 14-minute
    run must not bury "no file" at the bottom.
    """
    exr_path, exr_note = str(folder / f"{stem}.exr"), ""
    try:
        from atlas_camera.plate.oiio_io import write_exr
        folder.mkdir(parents=True, exist_ok=True)
        import json
        attrs = {"atlas:content": "matrixZone SDR->HDR model reconstruction",
                 "atlas:version": _atlas_version()}
        if params is not None:
            attrs["atlas:matrixzone_params"] = json.dumps(params, sort_keys=True)
        if step_mem is not None:
            attrs["atlas:seam_worst"] = _seam_worst_attr(step_mem)
        write_exr(exr_path, plate.astype(np.float32, copy=False), bit_depth="half",
                  source_colorspace=colorspace, extra_attribs=attrs)
    except Exception as exc:  # noqa: BLE001 - still return the preview + report
        exr_note = f"EXR NOT WRITTEN: {type(exc).__name__}: {exc}"
        exr_path = ""
    return exr_path, exr_note


def _score_seams(np, plate, plan, sdr_lin, exr_path, *, step_mem=None):
    """Seam step test on what SHIPS: ``(step, delivered, diff_note)``.

    Gate on the decoded EXR (half + lossy DWAB), not the float in memory.
    Found live 2026-10-02: the two scored the worst seam 3.78x vs 2.34x on
    one plate. When they disagree the report carries the diff so the cause
    can be bisected from the report alone.
    """
    from atlas_camera.core.matrixzone import seam_step_test
    if step_mem is None:
        step_mem = seam_step_test(plate, plan, sdr_linear=sdr_lin)
    step, delivered, diff_note = step_mem, None, ""
    if exr_path:
        try:
            from atlas_camera.plate.oiio_io import _require_oiio
            oiio = _require_oiio()
            delivered = np.asarray(oiio.ImageBuf(exr_path).get_pixels(oiio.FLOAT),
                                   dtype=np.float32)[..., :3]
            if delivered.shape != plate.shape[:2] + (3,):
                diff_note = (f"decoded EXR is {delivered.shape[:2]}, plate "
                             f"{plate.shape[:2]} - scored in memory")
                delivered = None
            else:
                step = seam_step_test(delivered, plan, sdr_linear=sdr_lin)
        except Exception as exc:  # noqa: BLE001 - fall back to the in-memory score
            diff_note = f"EXR read-back failed ({type(exc).__name__}) - scored in memory"
            delivered = None
    if delivered is not None:
        diff_note = _memory_vs_delivered_note(np, plate, delivered, step_mem, step) or diff_note
    return step, delivered, diff_note


def _memory_vs_delivered_note(np, plate, delivered, step_mem, step):
    """The diff note when the in-memory and delivered scores disagree, else ""."""
    wm, wd = step_mem["worst"], step["worst"]
    if not (wm and wd and (abs(wm["ratio"] - wd["ratio"]) > 0.25
                           or step_mem["flagged"] != step["flagged"])):
        return ""
    mem3 = np.asarray(plate, dtype=np.float32)[..., :3]
    lr = np.log2(np.maximum(np.abs(delivered), 1e-6)) - \
        np.log2(np.maximum(np.abs(mem3), 1e-6))
    x = wm["at_px"]
    band = (slice(None), slice(max(0, x - 100), x + 100)) if wm["orientation"] == "v" \
        else (slice(max(0, x - 100), x + 100), slice(None))
    return (
        f"in-memory vs delivered DISAGREE: worst {wm['seam']} {wm['ratio']:.2f}x in memory, "
        f"{wd['seam']} {wd['ratio']:.2f}x in the file; flags {step_mem['flagged']} vs "
        f"{step['flagged']}. diff: max |log2| {float(np.nanmax(np.abs(lr))):.3f}, "
        f"mean log2 {float(np.nanmean(lr)):+.4f} (band at {x} px: "
        f"{float(np.nanmean(lr[band])):+.4f}); memory nan {int(np.isnan(mem3).sum())} "
        f"neg {int((mem3 < 0).sum())} dtype {np.asarray(plate).dtype}; file nan "
        f"{int(np.isnan(delivered).sum())} neg {int((delivered < 0).sum())}")


def _fmt_seams(seams):
    return ", ".join(f"{s['seam']} {s['median_stops']:.3f}/{s['p95_stops']:.3f}"
                     for s in seams) or "(no interior seams)"


def _stitch_report(np, plate, rep, step, *, n_zones, mode, colorspace, exr_path, exr_note,
                   delivered, diff_note, stripe_rep, local_rep, xfer_rep, destripe,
                   detail_from_sdr, sdr_note=""):
    """The stitch node's multi-line report. A failed EXR write is line 1."""
    lines = [exr_note] if exr_note else []
    lines += [f"AtlasMatrixZoneStitch: {n_zones} zones ({mode}) -> "
             f"{plate.shape[1]}x{plate.shape[0]} {colorspace} half EXR "
             f"{exr_path or '(not written)'}",
             f"seams before, log2 luminance median/p95 stops: {_fmt_seams(rep['seams_before'])}"]
    if rep.get("anchored"):
        lines.append(f"seams after radiance anchor (split {rep['split_px']} px): "
                     f"{_fmt_seams(rep['seams_after_anchor'])}")
    else:
        lines.append("radiance anchor OFF: zones keep their own low frequencies")
    lines += _cleanup_report_lines(stripe_rep, local_rep, xfer_rep, destripe, detail_from_sdr)
    lines += _seam_step_report_lines(step, delivered)
    if diff_note:
        lines.append(f"  {diff_note}")
    if sdr_note:
        lines.append(sdr_note)
    if rep.get("resized_zones"):
        lines.append(f"warning: {rep['resized_zones']} zone result(s) came back at a "
                     "different size and were resampled to their renderRect")
    vmax, vp99, bad = _range_stats(np, plate)
    lines.append(f"range: max {vmax:.2f}, p99 {vp99:.3f} (linear, "
                 + (f"{bad} NON-FINITE pixel value(s) excluded -- inspect the plate" if bad
                    else "0 non-finite")
                 + "); a MODEL RECONSTRUCTION "
                 "of highlight radiance from a display-referred plate, not photographed HDR")
    return "\n".join(lines)


def _range_stats(np, plate, *, step=4, chunk_rows=256):
    """``(max, p99, non_finite_count)`` for the report's range line without an
    8K-sized bool mask or boolean-indexed copy: the non-finite count is exact
    (row-chunked), max/p99 come from a ``step``-strided subsample (finite only;
    NaN when the subsample has no finite value)."""
    bad = 0
    for r0 in range(0, plate.shape[0], chunk_rows):
        blk = plate[r0:r0 + chunk_rows]
        bad += int(blk.size - np.count_nonzero(np.isfinite(blk)))
    sub = plate[::step, ::step]
    sub = np.where(np.isfinite(sub), sub, np.nan)
    if not np.isfinite(sub).any():
        return float("nan"), float("nan"), bad
    return float(np.nanmax(sub)), float(np.nanpercentile(sub, 99)), bad


def _cleanup_report_lines(stripe_rep, local_rep, xfer_rep, destripe, detail_from_sdr):
    lines = []
    if stripe_rep:
        lines.append(f"destripe: vertical ripple the conversion added "
                     f"{stripe_rep['ripple_before_stops']:.4f} -> "
                     f"{stripe_rep['ripple_after_stops']:.4f} stops rms "
                     f"({stripe_rep['bands']} band(s) of <= 1/4 plate height, {stripe_rep['window_px']} px)")
    if local_rep:
        lines.append(f"local destripe (flat regions incl. clipped highlights, "
                     f"{local_rep['rows']}-row windows): correction p99 "
                     f"{local_rep['field_p99_stops']:.3f} stops, max "
                     f"{local_rep['field_max_stops']:.3f}; {local_rep['flat_fraction']:.0%} of "
                     "the plate flat enough to measure")
    elif destripe:
        lines.append("destripe skipped: wire sdr_plate (the plate the split got)")
    if xfer_rep:
        lines.append(f"detail from SDR (guided filter r{xfer_rep['radius_px']}): structure from "
                     f"the SDR, radiance from the conversion; HDR pixels kept on "
                     f"{xfer_rep['kept_hdr_fraction']:.0%} (clipped highlights); change p50 "
                     f"{xfer_rep['change_p50_stops']:.3f} / p99 {xfer_rep['change_p99_stops']:.3f} stops")
    elif detail_from_sdr:
        lines.append("detail from SDR skipped: wire sdr_plate")
    return lines


def _seam_step_report_lines(step, delivered):
    if not step["seams"]:
        return []
    base = sorted({f"{s['orientation']} {s['baseline_stops']:.3f}" for s in step["seams"]
                   if s["baseline_stops"] is not None})
    w = step["worst"]
    lines = [
        f"seam step test ({'delivered EXR' if delivered is not None else 'in-memory plate'}, "
        f"{step['strip_px']} px strips, worst of ~{step['window_px']} px windows, "
        + ("SDR-controlled" if step["sdr_controlled"] else
           "NOT SDR-controlled - wire sdr_plate, structure on a line scores too")
        + f", pass <= {step['ratio_max']:.1f}x random-line p{step['baseline_percentile']} "
        f"[{', '.join(base)} stops]): "
        + ("PASS" if step["pass"] else
           f"{len(step['flagged'])} seam(s) FLAGGED: {', '.join(step['flagged'])}")
        + (f"; worst {w['seam']} at {w['at_px']} px (window {w.get('worst_window')}), "
           f"{w['step_stops']:.3f} stops = {w['ratio']:.2f}x, window p95 "
           f"{w.get('window_p95_stops') or 0.0:.3f}" if w else "")]
    if step["flagged"]:
        lines.append("  a flagged seam is either a tonal seam or real structure that "
                     "happens to sit on the line -- look at it before trusting the plate")
    return lines
