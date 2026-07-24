"""Atlas ComfyUI nodes — viewport group.

Extracted verbatim from nodes.py during modularization; no behavior
change. Registered/exported via atlas_camera.comfy.node_registry.
"""
from __future__ import annotations

import base64
import io
import json
import os

from atlas_camera.comfy.viewport_payload import (
    _extract_blockout_camera,
    _fit_long_edge,
)
from atlas_camera.comfy.node_helpers import (
    _ASSESS_OUTPUT_SLOTS,

    _ATLAS_INPUT_BAND_NAMES,
    _ATLAS_INPUT_BOUNDARIES,
    _DEPTH_MODEL_CHOICES,
    _LAYER_DEBUG_PALETTE_HEX,
    _LAYER_DEBUG_PRIMARY_HEX,
    _blockout_cache_set,
    _mask_to_b64_png,
    _clone_solve_with_metadata,
    _comfy_registry,
    _native_sam3_available,
    _moge_available,
    _decode_b64_to_tensor,
    _execution_blocker,
    _graph_builder,
    _named_view_orbit_delta,
    _require_numpy,
    _require_pil,
    _require_torch,
    _solve_fingerprint,
    build_segmentation_cascade,
)
from atlas_camera.inference.depth_estimator import is_moge_model








# ---------------------------------------------------------------------------
# Track 2 — AtlasBlockout viewport node
# ---------------------------------------------------------------------------

class AtlasViewportControls:
    """Atlas Output Desk companion for AtlasBlockoutViewport.

    Output 0 remains the original controls link for compatibility. Output 1
    carries an OCIO-style output profile that exporters and the viewport can
    use as metadata. Browser preview is display-inferred/proxy; final color
    fidelity belongs to OCIO Write, Nuke, Maya, Resolve, etc.
    """
    RETURN_TYPES = ("ATLAS_VIEWPORT_LINK", "ATLAS_OUTPUT_PROFILE")
    RETURN_NAMES = ("controls", "output_profile")
    FUNCTION = "profile"
    CATEGORY = "Atlas Camera/Blockout"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "config_label": ("STRING", {"default": "ACES 2.0 / Studio"}),
                "config_path": ("STRING", {"default": ""}),
                "working_colorspace": ("STRING", {"default": "ACEScg"}),
                "output_colorspace": ("STRING", {"default": "ACES - ACEScg"}),
                "display": ("STRING", {"default": "sRGB - Display"}),
                "view": ("STRING", {"default": "ACES 2.0 SDR-video"}),
                # look/lut_path/exposure/gamma widgets removed 2026-07-10
                # (user: redundant — exposure duplicated the viewport's own ☀
                # control, gamma was a crude CSS filter, look/lut were inert
                # metadata) to make room on the Output Desk. profile() still
                # accepts them as kwargs and the AtlasOutputProfile schema
                # keeps the fields defaulted, so old prompts/API workflows and
                # every downstream consumer are unaffected. NOTE: this is the
                # one sanctioned widget REMOVAL — display_trim moved from
                # widgets_values index 10 to 6, and every shipped example
                # carrying this node was re-saved in the same commit.
                "display_trim": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
            },
        }

    def profile(self, config_label="ACES 2.0 / Studio", config_path="",
                working_colorspace="ACEScg", output_colorspace="ACES - ACEScg",
                display="sRGB - Display", view="ACES 2.0 SDR-video", display_trim=1.0,
                **legacy):
        from atlas_camera.core.schema import AtlasOutputProfile

        look = legacy.get("look", "None")
        lut_path = legacy.get("lut_path", "")
        exposure = legacy.get("exposure", 0.0)
        gamma = legacy.get("gamma", 1.0)

        return ("", AtlasOutputProfile(
            config_label=config_label or "ACES 2.0 / Studio",
            config_path=(config_path.strip() or None),
            working_colorspace=working_colorspace or "ACEScg",
            output_colorspace=output_colorspace or "ACES - ACEScg",
            display=display or "sRGB - Display",
            view=view or "ACES 2.0 SDR-video",
            look=look or "None",
            lut_path=(lut_path.strip() or None),
            exposure=float(exposure),
            gamma=float(gamma),
            display_trim=float(display_trim),
            preview_only=True,
            metadata={"source": "AtlasViewportControls"},
        ))


class AtlasBlockoutViewport:
    """
    Browser-based 3D blockout viewport initialized with the recovered camera.
    Pattern mirrors Yedp Blockout: browser renders → base64 JSON → Python decodes
    → four proxy/LDR IMAGE outputs (shaded, depth, normal, mask).

    Workflow:
    1. Connect an ATLAS_SOLVE and a source IMAGE, then queue the prompt.
    2. The Three.js viewport in the ComfyUI node opens, camera pre-aligned to the photo.
    3. Place primitive geometry (box, plane, person card, etc.).
    4. Click "Render Proxy Passes" in the viewport — fills client_data and re-queues.
    5. Four proxy/LDR IMAGE outputs are now available for ControlNet or compositing.
    6. Optional: use 🎥 Camera Path mode to author a keyframed camera move (fly
       nav, unclamped — leaving the recovered cone is expected here), then
       click "⏺ Bake Proxy Path" to fill client_data with a rendered frame sequence.
       `path_frames` (a proxy/LDR IMAGE batch) feeds a core Video Combine node directly;
       `camera_path` (the raw keyframes) feeds AtlasExportCameraPathUSD for a
       DCC-facing animated camera. Frames sampled outside the recovered
       camera's cone will show the same documented black/undefined regions as
       orbiting past it under 📽 Project — expected, not a bug.
    7. Optional: connect an AtlasViewportControls node to `controls` to move
       every button/panel (primitives, Project/Diagram/Info, Camera Path +
       presets + FBX import, Render Proxy Passes) OUT of this node — this node then
       shows the perspective render only, and can be freely resized by
       dragging its corner. `controls` carries no real data (a link exists
       purely so the two nodes' frontend JS can find each other); Python
       ignores it. With nothing connected, all controls still appear locally
       here, unchanged — fully backward-compatible with saved workflows that
       predate AtlasViewportControls.
    8. Optional: 📐 Extract Angle — orbit/fly to the view you want a patch
       generated at (e.g. the last frame of an intended camera move, MPTK
       style), click 📐, and the measured orbit delta from the RECOVERED
       camera is snapped to the Qwen Multiple-Angles LoRA's nearest named
       views (assuming the source photo is "front view"/"eye-level shot")
       and re-queued into four STRING outputs: `patch_azimuth_view`/
       `patch_elevation_view`/`patch_distance` (wire into AtlasAddPatchView/
       AtlasOcclusionMask's widgets converted-to-inputs) and `patch_prompt`
       (the ready-to-use "<sks> ..." LoRA prompt for generating the novel
       view). The delta is computed about `camera_math.ground_lookat_pivot`
       — the SAME pivot orbit_camera uses backend-side — so the snapped
       views round-trip exactly through those nodes' own camera
       construction. UNTIL an angle is extracted these outputs return
       ExecutionBlocker — downstream nodes wired to them (the Qwen
       generation, AtlasAddPatchView, exports) are silently PAUSED rather
       than running a wasted zero-orbit patch; clicking 📐 re-queues and the
       branch resumes automatically. Outside a ComfyUI runtime they fall
       back to the zero-orbit named-view defaults.
    """
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "ATLAS_CAMERA_PATH",
                    "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("shaded", "depth", "normal", "mask", "path_frames", "camera_path",
                    "patch_azimuth_view", "patch_elevation_view", "patch_distance", "patch_prompt",
                    "patch_exact")
    FUNCTION = "render"
    CATEGORY = "Atlas Camera/Blockout"
    OUTPUT_NODE = True  # kept alive even without downstream connections

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "source_image": ("IMAGE",),
                "resolution": ("INT", {"default": 768, "min": 128, "max": 4096, "step": 8,
                    "tooltip": "Long-edge render resolution; the short side auto-follows the "
                               "source image aspect (viewport inherits the image's aspect). "
                               "Also settable by dragging the node's own resize handle."}),
                "client_data": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "primary_depth": ("ATLAS_DEPTH_MAP", {
                    "tooltip": "Optional depth map matching the source image AND the depth object "
                               "used to derive the displayed relief mesh. Do not pair a DCC retopo "
                               "preview regenerated with another depth model. "
                               "Used by the viewport to power the ray-traced "
                               "occlusion culling feature (✂ Occlude)."}),
                "preview_expand": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Dilate derived geometry outward from the camera for wider orbit "
                               "coverage before it disappears into unreconstructed space. "
                               "1.0 = off (accurate geometry). Display only — never affects "
                               "DCC exports or measured geometry. CAUTION: values above 1.0 "
                               "dilate geometry beyond what the camera actually photographed, "
                               "so with 📽 Project active the dilated fringe has no real photo "
                               "data and renders as empty/black the moment you orbit off the "
                               "exact recovered viewpoint — leave at 1.0 if you plan to use "
                               "Project, raise it only for inspecting undressed blockout shapes."}),
                "controls": ("ATLAS_VIEWPORT_LINK", {
                    "tooltip": "Connect an AtlasViewportControls node here to move all buttons/"
                               "panels off this node (perspective-only, freely resizable). Carries "
                               "no real data — Python ignores it; the link only lets the two "
                               "nodes' frontend JS find each other."}),
                "shot_cam": ("ATLAS_SHOT_CAM", {
                    "tooltip": "Optional project/shot camera format (AtlasDefineShotCam) — conforms "
                               "the render resolution/aspect and viewing-camera FOV to this format "
                               "instead of auto-following source_image's own aspect. A direct wire "
                               "here wins over one attached to `solve` by AtlasMergeGeometry. Never "
                               "affects how the source photo is projected onto geometry."}),
                "output_profile": ("ATLAS_OUTPUT_PROFILE", {
                    "tooltip": "Optional OCIO-style output/profile metadata from AtlasViewportControls. "
                               "Browser preview remains display-inferred/proxy; final fidelity belongs "
                               "to OCIO Write, Nuke, Maya, or Resolve."}),
                "debug_matte": ("MASK", {
                    "tooltip": "🎭 Optional debug isolate (e.g. a layer's SAM3 mask, source-image "
                               "space): under 📽 Project the viewport dims everything whose "
                               "primary-camera projection falls OUTSIDE this matte (🎭 toolbar "
                               "toggle + dim slider; dim 0 = hard cull). Display-only — never "
                               "affects exports, geometry, or the projection itself."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def render(self, solve, source_image, resolution, client_data, primary_depth=None, preview_expand=1.0, controls=None,
               shot_cam=None, output_profile=None, debug_matte=None, unique_id=None):
        torch = _require_torch()
        if output_profile is not None:
            solve = _clone_solve_with_metadata(solve, output_profile=output_profile)

        # A direct shot_cam wire wins over one inherited from the solve (e.g.
        # attached earlier by AtlasMergeGeometry) — explicit beats inherited.
        resolved_shot_cam = shot_cam if shot_cam is not None else getattr(solve, "shot_cam", None)
        shot_intrinsics = None
        if resolved_shot_cam is not None:
            from atlas_camera.core.intrinsics import intrinsics_from_shot_cam
            shot_intrinsics = intrinsics_from_shot_cam(resolved_shot_cam)
            width, height = shot_intrinsics.image_width, shot_intrinsics.image_height
        else:
            # Auto-adopt the source image aspect: derive W×H from the incoming
            # image, scaled so the long edge is `resolution`.
            src_h, src_w = int(source_image.shape[1]), int(source_image.shape[2])
            width, height = _fit_long_edge(src_w, src_h, int(resolution))

        # Store camera data for the browser extension to fetch
        node_id = str(unique_id) if unique_id is not None else "0"
        solve_fingerprint = _solve_fingerprint(solve, source_image)
        debug_matte_b64 = ""
        if debug_matte is not None:
            try:
                m = debug_matte
                if hasattr(m, "dim") and m.dim() == 3:  # ComfyUI MASK (B,H,W)
                    m = m[0]
                debug_matte_b64 = _mask_to_b64_png(m.cpu().numpy() if hasattr(m, "cpu") else m)
            except Exception:
                debug_matte_b64 = ""  # a bad matte must never kill the viewport
        _blockout_cache_set(node_id, _extract_blockout_camera(
            solve, source_image, width, height, preview_expand=float(preview_expand),
            shot_intrinsics=shot_intrinsics, output_profile=output_profile,
            solve_fingerprint=solve_fingerprint, primary_depth=primary_depth,
            debug_matte_b64=debug_matte_b64))

        # IMPORTANT: return a "ui" payload. ComfyUI only emits the "executed"
        # websocket message (which triggers node.onExecuted / the frontend's
        # api "executed" event) for nodes whose result includes UI output —
        # without this the browser extension never learns the solve is ready
        # and never fetches the camera data / background / proxies.
        ui_payload = {"atlas_ready": [node_id]}
        # Filled in below once client_data is parsed — lets the frontend show
        # a "patch branch paused — 📐 Extract Angle" hint instead of the
        # paused branch silently looking like a failed run.
        _patch_state = {"paused": True}

        # 📐 Extract Angle results (written into client_data by the viewport's
        # Extract Angle button). UNTIL an angle has been extracted, the four
        # patch_* outputs return ComfyUI's ExecutionBlocker sentinel — any
        # downstream node wired to them (the Qwen generation, AtlasAddPatchView,
        # the Nuke layers export) is silently SKIPPED rather than running a
        # wasted zero-orbit no-op patch. Extract Angle already re-queues on
        # click, so the paused branch resumes by itself the moment the artist
        # picks an angle — an automatic pause/resume, no manual resume node.
        # Outside a ComfyUI runtime (unit tests, plain-python imports) the
        # blocker class doesn't exist, so the zero-orbit named-view defaults
        # are returned instead — those also remain the fallback semantics if
        # a future ComfyUI ever drops ExecutionBlocker.
        _pa_defaults = ("front view", "eye-level shot", "medium shot",
                        "<sks> front view eye-level shot medium shot",
                        "azimuth_deg=0.0 elevation_deg=0.0 distance_scale=1.0")

        def _patch_exact_string(pa):
            # 📐 stores the RAW measured orbit floats alongside the snapped
            # named views (client_data.patch_angle.raw) — the exact-angle
            # channel the render-conditioned patch loop needs (bake a frame
            # at the artist's real orbit, fix it, project it back from the
            # IDENTICAL pose via AtlasAddPatchView's exact_view_override; a
            # 45°-grid named view would misregister the projection). Records
            # without `raw` fall back to the snapped named views' own delta,
            # so the string is always a pose AddPatchView can reproduce.
            raw = pa.get("raw") or {}
            try:
                d_az = float(raw["d_azimuth_deg"])
                d_el = float(raw["d_elevation_deg"])
                dist = float(raw["distance_scale"])
            except (KeyError, TypeError, ValueError):
                try:
                    d_az, d_el, dist = _named_view_orbit_delta(
                        str(pa.get("azimuth_view") or _pa_defaults[0]),
                        str(pa.get("elevation_view") or _pa_defaults[1]),
                        str(pa.get("distance_view") or _pa_defaults[2]),
                        "front view", "eye-level shot", False)
                except KeyError:
                    return _pa_defaults[4]
            return (f"azimuth_deg={d_az:.4f} elevation_deg={d_el:.4f} "
                    f"distance_scale={dist:.4f}")

        def _patch_angle_strings(parsed):
            pa = (parsed or {}).get("patch_angle") or {}
            # A stale extraction — made from a DIFFERENT solve/image than the
            # one now wired in (or from before fingerprints existed) — must
            # re-arm the pause, not silently patch at the old image's angle.
            if pa and pa.get("fingerprint") != solve_fingerprint:
                pa = {}
            _patch_state["paused"] = not pa
            if not pa:
                blocker = _execution_blocker()
                if blocker is not None:
                    return (blocker,) * 5
                return _pa_defaults
            return (
                str(pa.get("azimuth_view") or _pa_defaults[0]),
                str(pa.get("elevation_view") or _pa_defaults[1]),
                str(pa.get("distance_view") or _pa_defaults[2]),
                str(pa.get("prompt") or _pa_defaults[3]),
                _patch_exact_string(pa),
            )

        if not client_data.strip():
            blank = torch.zeros(1, height, width, 3, dtype=torch.float32)
            pa_strings = _patch_angle_strings(None)
            ui_payload["atlas_patch_paused"] = [_patch_state["paused"]]
            return {"ui": ui_payload,
                    "result": (blank, blank, blank, blank, blank, None) + pa_strings}

        try:
            data = json.loads(client_data)
        except json.JSONDecodeError:
            blank = torch.zeros(1, height, width, 3, dtype=torch.float32)
            pa_strings = _patch_angle_strings(None)
            ui_payload["atlas_patch_paused"] = [_patch_state["paused"]]
            return {"ui": ui_payload,
                    "result": (blank, blank, blank, blank, blank, None) + pa_strings}

        shaded = _decode_b64_to_tensor(data.get("shaded", ""), width, height)
        depth  = _decode_b64_to_tensor(data.get("depth",  ""), width, height)
        normal = _decode_b64_to_tensor(data.get("normal", ""), width, height)
        mask   = _decode_b64_to_tensor(data.get("mask",   ""), width, height)

        path_frames_b64 = data.get("path_frames") or []
        if path_frames_b64:
            path_frames = torch.cat(
                [_decode_b64_to_tensor(b64, width, height) for b64 in path_frames_b64], dim=0
            )
        else:
            path_frames = torch.zeros(1, height, width, 3, dtype=torch.float32)

        camera_path_data = data.get("camera_path")
        camera_path = None
        if camera_path_data:
            from atlas_camera.core.schema import AtlasCameraPath
            camera_path = AtlasCameraPath.from_dict(camera_path_data)

        pa_strings = _patch_angle_strings(data)
        ui_payload["atlas_patch_paused"] = [_patch_state["paused"]]
        return {"ui": ui_payload,
                "result": (shaded, depth, normal, mask, path_frames, camera_path)
                + pa_strings}

    @classmethod
    def IS_CHANGED(cls, client_data="", **_):
        import hashlib
        return hashlib.md5(client_data.encode()).hexdigest()


class AtlasDebugReport:
    """🔍 One-stop machine-readable diagnostic of the layered master scene.

    Wire the FINAL solve (whatever feeds the master viewport) plus, optionally,
    the shared depth, the 🎯 scope-status strings, and the 🧭 VLM report. Every
    execution introspects the whole stack — camera summary, every
    ProjectionSource's geometry type / vertex count / band range / matte
    coverage — runs red-flag analysis (zero-vertex layers, band gaps/overlaps,
    near-empty mattes, no-match scope fallbacks), renders a human-readable
    report ON the node, and writes the full structured JSON to a STABLE path
    (`file_path`, default `atlas_debug/master_debug.json` under ComfyUI's CWD).

    The JSON exists specifically so an AI assistant (or any tool) can read one
    file and see the same facts that otherwise take a live payload/history
    autopsy to reconstruct — built after exactly such a session: two layers
    silently shipped zero-vertex meshes and only a viewport-payload dump
    revealed it. Zero heavy computation: everything is read off data the solve
    already carries (matte coverage decodes the embedded PNGs, the one
    non-trivial step).
    """
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "json_path")
    FUNCTION = "report"
    CATEGORY = "Atlas Camera/Gates & QA"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"solve": ("ATLAS_SOLVE",)},
            "optional": {
                "depth": ("ATLAS_DEPTH_MAP",),
                "file_path": ("STRING", {"default": "atlas_debug/master_debug.json",
                    "tooltip": "Stable JSON path (relative to ComfyUI's working directory) — "
                               "keep it constant so external tooling can always find the "
                               "latest diagnostic."}),
                "status_1": ("STRING", {"forceInput": True}),
                "status_2": ("STRING", {"forceInput": True}),
                "status_3": ("STRING", {"forceInput": True}),
                "status_4": ("STRING", {"forceInput": True}),
                "vlm_report": ("STRING", {"forceInput": True}),
            },
        }

    @staticmethod
    def _matte_coverage(mask_b64):
        if not mask_b64:
            return None
        try:
            np = _require_numpy()
            PILImage = _require_pil()
            raw = base64.b64decode(mask_b64.split(",", 1)[-1])
            arr = np.asarray(PILImage.open(io.BytesIO(raw)).convert("L"))
            return round(float((arr > 127).mean()), 4)
        except Exception:
            return None

    def report(self, solve, depth=None, file_path="atlas_debug/master_debug.json",
               status_1="", status_2="", status_3="", status_4="", vlm_report="",
               **_extra):
        import datetime

        from atlas_camera.core.scene_health import evaluate_scene_health

        # The check logic lives in core.scene_health (the single red-flag
        # engine, shared with AtlasSceneHealthGate 🩺) — this node is a thin
        # consumer that renders the identical JSON/text it always did.
        # test_debug_report_parity.py pins the exact flag strings.
        statuses = {f"status_{i}": s for i, s in
                    enumerate((status_1, status_2, status_3, status_4), 1) if s}
        health = evaluate_scene_health(
            solve, depth, scope_statuses=statuses,
            matte_coverage_fn=self._matte_coverage)
        camera = health.camera
        cam_y = camera.get("camera_height_m")
        sources = health.per_layer
        depth_info = health.depth
        flags = health.flag_messages

        try:
            from atlas_camera import __version__ as _atlas_version
        except Exception:
            _atlas_version = "unknown"
        data = {
            # Versioned for external consumers (this JSON is parsed by
            # tooling/AI assistants, not just eyeballed): bump "schema" on
            # any breaking key change.
            "schema": 1,
            "atlas_version": _atlas_version,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "camera": camera,
            "depth": depth_info,
            "shot_cam": (solve.shot_cam.to_dict()
                         if getattr(solve, "shot_cam", None) and
                         hasattr(solve.shot_cam, "to_dict") else None),
            "projection_sources": sources,
            "primary_proxy_geometry": [
                {"name": g.name, "type": g.primitive_type,
                 "n_vertices": (g.metadata or {}).get("n_vertices")}
                for g in (getattr(solve.projection_scene, "proxy_geometry", None) or [])],
            "scope_status": statuses,
            "vlm_report": vlm_report or None,
            "flags": flags,
        }
        path = os.path.abspath(file_path or "atlas_debug/master_debug.json")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, ensure_ascii=False)
        except OSError as exc:
            flags.append(f"could not write {path}: {exc}")
            path = ""

        lines = ["ATLAS MASTER DEBUG", "=" * 18, ""]
        lines.append(f"camera  {camera['image_wh'][0]}x{camera['image_wh'][1]}  "
                     f"{camera['focal_mm'] or '?'}mm  height {cam_y if cam_y is not None else '?'}m  "
                     f"conf {camera['confidence']}")
        lines.append("")
        lines.append("LAYERS (name / geometry / band / verts / matte)")
        for s in sources:
            band = (f"{s['near_m'] or 0:.1f}-" +
                    (f"{s['far_m']:.1f}m" if s["far_m"] is not None else "inf"))
            cov = f"{s['matte_coverage']:.1%}" if s["matte_coverage"] is not None else "-"
            lines.append(f"  {s['name']:10s} {s['band_geometry'] or '-':7s} {band:12s} "
                         f"{s['n_vertices']:>8d} verts  matte {cov}")
        if statuses:
            lines += ["", "SCOPE STATUS"] + [f"  {v}" for v in statuses.values()]
        lines += ["", f"FLAGS ({len(flags)})"] + ([f"  ! {f}" for f in flags] or ["  (none — stack looks healthy)"])
        if path:
            lines += ["", f"full JSON: {path}"]
        report = "\n".join(lines)
        return {"ui": {"text": [report]}, "result": (report, path)}


class AtlasLayerPreview:
    """🎨 Cut-out layer preview: the plate's pixels where this layer's matte
    is, and the layer's 🎨 Layers debug color (opaque) everywhere else — one
    image that shows WHAT the layer will project AND which layer it is, in
    the exact color the master viewport's 🎨 Layers legend uses for it.

    Replaces the mask-preview + plate-preview pairs in the staged master's
    per-stage debug strip (user feedback: "the preview should only show
    already cut-out layers, then we don't need the mask preview").
    `layer_index` is the layer's position in `projection_sources` (the 🎨
    legend order — staged master: sky=0, far=1, bg=2, mid=3, fg=4; -1 = the
    primary teal); `color_hex` overrides the palette when set.
    """
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preview"
    CATEGORY = "Atlas Camera/Inpaint Layers"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The layer's plate (post-inpaint)."}),
                "mask": ("MASK", {"tooltip": "The layer's matte (band layer_mask / sky mask)."}),
            },
            "optional": {
                "layer_index": ("INT", {"default": 0, "min": -1, "max": 15,
                    "tooltip": "Position in projection_sources = the 🎨 Layers legend order "
                               "(staged master: sky=0, far=1, bg=2, mid=3, fg=4). -1 = primary "
                               "teal. Sets the opaque surround color."}),
                "color_hex": ("STRING", {"default": "",
                    "tooltip": "Optional explicit hex color (e.g. ff6a3d) — overrides "
                               "layer_index's palette pick."}),
            },
        }

    def preview(self, image, mask, layer_index=0, color_hex=""):
        torch = _require_torch()
        import torch.nn.functional as F

        hexs = (color_hex or "").strip().lstrip("#")
        if not hexs:
            hexs = (_LAYER_DEBUG_PRIMARY_HEX if int(layer_index) < 0 else
                    _LAYER_DEBUG_PALETTE_HEX[int(layer_index) % len(_LAYER_DEBUG_PALETTE_HEX)])
        try:
            rgb = tuple(int(hexs[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except (ValueError, IndexError):
            rgb = (1.0, 0.0, 1.0)  # loud magenta for a malformed hex
        h, w = int(image.shape[1]), int(image.shape[2])
        m = mask if mask.dim() == 3 else mask.unsqueeze(0)
        if tuple(m.shape[1:]) != (h, w):
            m = F.interpolate(m.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1)
        mm = (m[:1] > 0.5).float().unsqueeze(-1)
        color = torch.tensor(rgb, dtype=image.dtype, device=image.device).view(1, 1, 1, 3)
        out = image[:, :, :, :3] * mm + color * (1.0 - mm)
        return (out,)


class AtlasInput:
    """🎬 The all-in-one entry point — one node between LoadImage and the
    viewport that wraps the staged master's logic via NODE EXPANSION: at
    execution it emits the real mini-graph (our nodes by class, third-party
    LaMa by registry name; native SAM3 via AtlasSAM3Mask, our own node) so
    every inner step keeps its own cache, and missing packs degrade
    gracefully (skipped + named in the `report` output) instead of erroring.

    Out of the box (instant relief): layers=0, VLM/SAM/inpaint off — the
    first queue costs solve + depth + ONE high-resolution relief mesh, and
    the `solve`/`image` outputs wire straight into AtlasBlockoutViewport.
    Turn knobs up from there:
    - `layers` 2–4 splits into depth-band clean-plate layers on the proven
      splits, watertight by construction (the band_override channel).
    - `use_vlm` puts AtlasAssessImage in front (advisory mode, VRAM
      offloaded after) and wires its prompts / per-band geometry / band
      boundaries into the inner nodes exactly like the staged master. With
      layers>0 this forces the 4-band plan (the VLM speaks 5 fixed slots).
    - `sky` + `sky_prompt` adds the SAM sky card; the mask also feeds every
      mesh's exclude_mask AND band_ref_mask (the band-drift rule).
    - `scope_prompts` (one line per band, far→near) adds self-disarming 🎯
      scope rows; `inpaint` adds the ✂crop→LaMa(+upscale)→✂stitch clean-
      plate chain per occluded band.

    When you outgrow the fast wrapper, the staged master is the production
    evolution of this idea: five native ComfyUI subgraphs, explicit solve
    gates and masks, four cropped SDXL clean plates, per-layer previews, and
    DCC outputs.  It deliberately has no Set/Get rails or shared LaMa chain —
    see examples/atlas_camera_staged_master_workflow.json.
    """
    RETURN_TYPES = ("ATLAS_SOLVE", "IMAGE", "ATLAS_DEPTH_MAP", "MASK", "STRING")
    RETURN_NAMES = ("solve", "image", "depth", "sky_mask", "report")
    FUNCTION = "build"
    CATEGORY = "Atlas Camera/Solve"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "layers": ("INT", {"default": 0, "min": 0, "max": 4,
                    "tooltip": "0 = one full-range mesh (instant relief). 2/3/4 = depth-band "
                               "clean-plate layers on the proven splits (2→0.55; 3→0.2/0.65; "
                               "4→0.3/0.6/0.8), watertight by construction. 1 = one full-range "
                               "clean-plate layer (useful with mesh=card/ground)."}),
                "mesh": (["relief", "card", "ground"], {"default": "relief",
                    "tooltip": "layers=0: relief = depth-following mesh; card/ground = ONE flat "
                               "plane (band-median card / analytic ground). layers>0: the "
                               "DEFAULT band geometry — the VLM's per-band call wins when "
                               "use_vlm is on."}),
                "mesh_resolution": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64,
                    "tooltip": "Relief grid (long-edge cells). Internal tear threshold pairs "
                               "automatically: 0.5 for the single full-range mesh, 1.5 for "
                               "band-clipped layers (the calibrated pairings)."}),
                "use_vlm": ("BOOLEAN", {"default": False,
                    "tooltip": "Run the 🧭 VLM assessment first (advisory — never blocks; VRAM "
                               "offloaded after) and wire its SAM prompts, per-band geometry, "
                               "and band boundaries into the inner nodes. With layers>0 this "
                               "forces the 4-band plan (the VLM's plan has 5 fixed slots)."}),
                "vlm_provider": (["ollama", "lmstudio", "llamacpp", "openai"],
                    {"default": "lmstudio"}),
                "vlm_model": ("STRING", {"default": "",
                    "tooltip": "Blank = the provider's default model."}),
                "sky": ("BOOLEAN", {"default": False,
                    "tooltip": "SAM-segment the sky onto its own flat card, and feed the mask "
                               "into every mesh's exclude_mask + band_ref_mask. Uses native "
                               "SAM3 (transformers>=5.5.4, [sam3] extra) or falls back to "
                               "AtlasSemanticMask — skipped + noted if neither is available."}),
                "sky_prompt": ("STRING", {"default": "sky",
                    "tooltip": "Manual sky segmentation prompt; the VLM's wins when use_vlm."}),
                "scope_prompts": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Manual per-band SAM scoping, ONE PROMPT PER LINE far→near "
                               "(line 1 = farthest band). Blank line = that band stays "
                               "band-only. Self-disarming: a no-match segment falls back to "
                               "band-only automatically. The VLM's prompts win when use_vlm. "
                               "Uses native SAM3 ([sam3] extra) or AtlasSemanticMask."}),
                "inpaint": ("BOOLEAN", {"default": False,
                    "tooltip": "Build each occluded band's clean plate: occlusion mask → "
                               "expand → ✂crop → LaMa → ✂stitch (the 256²-bottleneck fix). "
                               "Needs comfyui-inpaint-nodes + big-lama.pt — skipped + noted if "
                               "absent. Off = bands project the original photo (honest holes "
                               "on reveal)."}),
                "upscale_model": ("STRING", {"default": "",
                    "tooltip": "Optional upscale model FILENAME (models/upscale_models) fed to "
                               "the inner LaMa nodes — e.g. 4xRealWebPhoto_v4_dat2.safetensors. "
                               "Measured 6.5× fill detail vs legacy. Blank = off."}),
                "edge_extend_px": ("INT", {"default": 24, "min": 0, "max": 256, "step": 4,
                    "tooltip": "Behind-band edge-extend (layers>0): how far plate colours smear "
                               "PAST each silhouette to hide grid-step tears — the frontmost band "
                               "always keeps a clean 0 cut. Was baked at 64 (tuned for smooth "
                               "ridgelines); lower it for high-frequency content like foliage, "
                               "which 64 shreds into halos. 0 = clean cut on every band."}),
                "max_edge_factor": ("FLOAT", {"default": 12.0, "min": 2.0, "max": 200.0, "step": 1.0,
                    "tooltip": "World-space edge tear threshold (layers=0 relief AND layers>0 "
                               "bands), SEPARATE from depth_edge_rel and often the DOMINANT tear "
                               "cause on deep / narrow-FOV / interior scenes — grazing walls and "
                               "receding floors trip the default 12x even where continuous, "
                               "shredding the mesh into 'combs'. Raise to 40-80 to close them; "
                               ">80 rubber-sheets real foreground silhouettes onto the background."}),
                "sky_heuristic": ("BOOLEAN", {"default": True,
                    "tooltip": "layers=0 relief mesh: exclude above-horizon far/rough regions as "
                               "sky before triangulation. Correct OUTDOORS; turn OFF for INTERIORS "
                               "(it eats the ceiling / vault / far wall as 'sky', punching large "
                               "holes). Ignored when sky (the SAM card) is on — that mask governs. "
                               "(layers>0 bands: sky handled per-band via exclude/scope instead.)"}),
                "normal_edge_deg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 180.0, "step": 1.0,
                    "tooltip": "0 = off. A THIRD tear test on surface-normal BEND (layers=0 relief "
                               "AND layers>0 bands): tears real creases / occlusion silhouettes "
                               "while leaving smoothly-receding walls and floors intact (unlike "
                               "max_edge_factor, which trips on any grazing surface). Pair with a "
                               "HIGHER max_edge_factor: raise mef to stop comb-tearing continuous "
                               "grazing surfaces, then set ~40-70 here to keep genuine edges torn."}),
                "depth_model": (list(_DEPTH_MODEL_CHOICES),
                    {"default": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                     "tooltip": "Monocular depth backend (fed the solved focal). "
                                "V2-Metric-Outdoor (DEFAULT) = Apache, transformers-only (NO extra "
                                "install), best all-round on OUTDOOR/sky scenes; V2-Metric-Indoor "
                                "is its interior twin. MoGe-2 (Ruicheng/moge-*) = MIT, cleanest on "
                                "ENCLOSED/INTERIOR shots but masks sky (poor outdoors) — needs "
                                "[moge]. DA3* (EXPERIMENTAL) = strong metric, heavy deps, DA3NESTED "
                                "is non-commercial CC BY-NC — needs [neural-da3]. Pick per shot: "
                                "outdoor->V2-Outdoor, interior->MoGe or V2-Indoor. (A/B 2026-07-13.)"}),
                "vlm_scope": ("BOOLEAN", {"default": True,
                    "tooltip": "When use_vlm: also SCOPE each band by the VLM's SAM prompt "
                               "(band ∩ segment). A PARTIAL segment match legitimately keeps "
                               "the scope and can cut real band geometry — found live on the "
                               "ghost-town plate, where the mid band's 4.6% segment left the "
                               "rest of the band exposing the behind-band's fill smear. OFF = "
                               "VLM still drives bands/geometry, layers stay band-only "
                               "(robust full coverage — best for camera moves)."}),
                "sky_inpaint_mode": (["lama", "sdxl"], {"default": "lama",
                    "tooltip": "How to build the sky card's clean plate: lama = fast deterministic edge-fill (needs comfyui-inpaint-nodes), sdxl = generative SDXL inpaint (needs a checkpoint)."}),
                "sky_lama_grow_px": ("INT", {"default": 32, "min": 0, "max": 128,
                    "tooltip": "Mask dilation before LaMa sky inpaint. Larger = more aggressive removal of foreground silhouettes from the sky plate."}),
                "sky_sdxl_checkpoint": ("STRING", {"default": "SDXL/sd_xl_base_1.0.safetensors",
                    "tooltip": "SDXL checkpoint filename in models/checkpoints when sky_inpaint_mode=sdxl."}),
                "sky_sdxl_positive": ("STRING", {"default": "clear seamless sky, high detail, no buildings, no trees, no roofs",
                    "multiline": True, "tooltip": "Positive prompt for SDXL sky inpaint."}),
                "sky_sdxl_negative": ("STRING", {"default": "building, tree, roof, person, vehicle, text, watermark, blurry",
                    "multiline": True, "tooltip": "Negative prompt for SDXL sky inpaint."}),
                "sky_sdxl_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Seed for SDXL sky inpaint. 0 = deterministic default behavior of the sampler node."}),
                # Live mesh repair (interior hole-fill / boundary sawtooth) is no
                # longer configured here — wire the standalone AtlasLiveMeshRepair
                # 🔧 node onto this node's `solve` output instead. It repairs any
                # relief mesh on the solve (single mesh or per-layer) downstream.
            },
        }

    # --- assembly ---------------------------------------------------------
    def build(self, image, layers=0, mesh="relief", mesh_resolution=512,
              use_vlm=False, vlm_provider="lmstudio", vlm_model="",
              sky=False, sky_prompt="sky", scope_prompts="", inpaint=False,
              upscale_model="", edge_extend_px=24, max_edge_factor=12.0,
              sky_heuristic=True, normal_edge_deg=0.0,
              depth_model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
              vlm_scope=True,
              sky_inpaint_mode="lama", sky_lama_grow_px=32,
              sky_sdxl_checkpoint="SDXL/sd_xl_base_1.0.safetensors",
              sky_sdxl_positive="clear seamless sky, high detail, no buildings, no trees, no roofs",
              sky_sdxl_negative="building, tree, roof, person, vehicle, text, watermark, blurry",
              sky_sdxl_seed=0, **_extra):

        registry = _comfy_registry()
        # Native SAM3 (AtlasSAM3Mask, transformers>=5.5.4, no triton) fully
        # supersedes the third-party SAM3Segment (comfyui-rmbg) in Atlas's own
        # cascade — it works on CUDA/CPU/MPS alike, so there's no case where
        # preferring the triton-locked node is better. AtlasSemanticMask
        # (SegFormer/ADE20K, [neural], no triton) remains the learned fallback
        # for transformers<5.5.4 / [sam3] not installed.
        have_native_sam3 = _native_sam3_available()
        have_semantic = "AtlasSemanticMask" in registry
        have_inpaint = ("INPAINT_InpaintWithModel" in registry
                        and "INPAINT_LoadInpaintModel" in registry
                        and "INPAINT_ExpandMask" in registry)
        notes: list = []

        if is_moge_model(str(depth_model)) and not _moge_available():
            notes.append("MoGe package not installed — AtlasDepthMap will fail; "
                         "install the [moge] extra (see INSTALL.md)")
        g = _graph_builder()

        def segment(image_ref, prompt_value):
            """Text-prompt segmentation via centralized build_segmentation_cascade."""
            mask_ref, _ = build_segmentation_cascade(
                g, image_ref, prompt_value, policy="semantic",
                have_native_sam3=have_native_sam3, registry=registry,
            )
            return mask_ref



        if not have_native_sam3 and have_semantic:
            notes.append("native SAM3 absent -> AtlasSemanticMask (SegFormer, CPU/MPS) "
                         "fallback for sky/scope")

        # 0. optional VLM assessment (advisory: auto_continue never blocks).
        image_ref = image
        vlm = None
        if use_vlm:
            vlm = g.node("AtlasAssessImage", image=image,
                         provider=vlm_provider, model=vlm_model,
                         auto_continue=True, offload_model=True)
            image_ref = vlm.out(0)
            notes.append("VLM assessment ON — prompts/geometry/bands from the plan")
            if layers > 0 and layers != 4:
                notes.append(f"layers {layers} → 4 (the VLM plan has 4 band slots)")
                layers = 4

        # 1. solve + shared depth (always). depth_model is fed the solve so
        # DA3METRIC / MoGe get the recovered focal (metric scale / fov_x).
        # MoGe masks sky poorly; when the user explicitly asks for a sky card
        # we auto-switch to DA2-Metric-Outdoor so the sky gets real depth and
        # the skyline is preserved for the matte.
        effective_depth_model = depth_model
        if sky and is_moge_model(str(depth_model)):
            effective_depth_model = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
            notes.append("MoGe + sky requested → depth auto-switched to DA2-Outdoor")

        solve = g.node("AtlasLearnedSolveFromImage", image=image_ref,
                       depth_model=effective_depth_model)

        depth = g.node("AtlasDepthMap", image=image_ref, solve=solve.out(0),
                       depth_model=effective_depth_model)

        # 2. sky mask (SolidMask zero when off/unavailable — every consumer
        # nearest-resizes masks, so the 64px placeholder is fine).
        zero_mask = g.node("SolidMask", value=0.0, width=64, height=64)
        sky_mask_ref = zero_mask.out(0)
        sky_on = bool(sky)
        if sky_on:
            sky_prompt_ref = vlm.out(_ASSESS_OUTPUT_SLOTS["sam_prompt_sky"]) if vlm is not None else sky_prompt


            sky_mask = segment(image_ref, sky_prompt_ref)
            if sky_mask is None:
                notes.append("sky SKIPPED — no segmenter (native SAM3 / AtlasSemanticMask absent)")
                sky_on = False
            else:
                sky_mask_ref = sky_mask
                notes.append("sky card ON")

        # 2b. clean-plate sky (inpaint foreground occluders from the inverted
        # sky mask) so the sky card projects a real sky, not buildings/trees.
        sky_plate_ref = image_ref
        if sky_on:
            inverted_sky = g.node("InvertMask", mask=sky_mask_ref).out(0)
            mode = str(sky_inpaint_mode or "lama").lower()
            if mode == "sdxl":
                grown = g.node("INPAINT_ExpandMask", mask=inverted_sky,
                               grow=8, blur=8, blur_type="gaussian")
                crop = g.node("AtlasInpaintCrop", image=image_ref,
                              mask=grown.out(0), context_pad_px=128)
                sdxl = g.node("AtlasSDXLInpaint",
                              image=crop.out(0), mask=crop.out(1),
                              checkpoint=sky_sdxl_checkpoint,
                              positive_prompt=sky_sdxl_positive,
                              negative_prompt=sky_sdxl_negative,
                              seed=int(sky_sdxl_seed),
                              steps=30, cfg=5.5, denoise=0.85,
                              grow_mask_by=8)
                stitch = g.node("AtlasInpaintStitch", original_image=image_ref,
                                inpainted_crop=sdxl.out(0),
                                crop_region=crop.out(2))
                sky_plate_ref = stitch.out(0)
                notes.append("sky plate SDXL inpaint")
            elif mode == "lama" and have_inpaint:
                lama_loader = g.node("INPAINT_LoadInpaintModel", model_name="big-lama.pt")
                grown = g.node("INPAINT_ExpandMask", mask=inverted_sky,
                               grow=int(sky_lama_grow_px), blur=8, blur_type="gaussian")
                crop = g.node("AtlasInpaintCrop", image=image_ref,
                              mask=grown.out(0), context_pad_px=128)
                lama = g.node("INPAINT_InpaintWithModel",
                              inpaint_model=lama_loader.out(0),
                              image=crop.out(0), mask=crop.out(1), seed=0)
                stitch = g.node("AtlasInpaintStitch", original_image=image_ref,
                                inpainted_crop=lama.out(0),
                                crop_region=crop.out(2))
                sky_plate_ref = stitch.out(0)
                notes.append("sky plate LaMa inpaint")
            else:
                notes.append("sky plate LaMa SKIPPED — comfyui-inpaint-nodes not installed, using raw image")

        # 3. geometry.
        solve_chain = solve.out(0)
        if sky_on:
            # Generous smear (the ultra workflow's 96/128 calibration): the
            # sky card must reach well below every ridge silhouette so orbit
            # reveals show smeared sky, never black.
            sky_layer = g.node("AtlasSkyDomeLayer", solve=solve_chain,
                               depth=depth.out(0), sky_mask=sky_mask_ref,
                               plate_image=sky_plate_ref,
                               edge_extend_px=96, frame_outpaint_px=128)
            solve_chain = sky_layer.out(0)

        exclude_kw = {"exclude_mask": sky_mask_ref} if sky_on else {}
        band_ref_kw = {"band_ref_mask": sky_mask_ref} if sky_on else {}

        if layers == 0:
            if mesh == "relief":
                relief = g.node("AtlasDeriveReliefMesh", solve=solve_chain,
                                depth=depth.out(0),
                                relief_grid=int(mesh_resolution),
                                depth_edge_rel=0.5,
                                max_edge_factor=float(max_edge_factor),
                                sky_heuristic=bool(sky_heuristic),
                                normal_edge_deg=float(normal_edge_deg),
                                **exclude_kw)
                solve_chain = relief.out(0)
                notes.append(f"single relief mesh, grid {int(mesh_resolution)}")
            else:
                flat = g.node("AtlasCleanPlateLayer", solve=solve_chain,
                              depth=depth.out(0), plate_image=image_ref,
                              near_pct=0.0, far_pct=0.0,  # full range (+inf)
                              name="full_range", priority=0.0,
                              relief_grid=int(mesh_resolution), depth_edge_rel=1.5,
                              embed_matte=sky_on, band_geometry=mesh,
                              frame_outpaint_px=64,  # orbit slack past the frame edge
                              **exclude_kw, **band_ref_kw)
                solve_chain = flat.out(0)
                notes.append(f"single full-range {mesh} plane")
        else:
            bounds = _ATLAS_INPUT_BOUNDARIES.get(int(layers), ())
            edges = [0.0, *bounds, 1.0]
            n_bands = len(edges) - 1
            # far -> near, staged priorities 0/5/10/15
            scope_lines = [s.strip() for s in (scope_prompts or "").splitlines()]
            lama_loader = None
            upscaler = None
            if inpaint and not have_inpaint:
                notes.append("inpaint SKIPPED — comfyui-inpaint-nodes not installed")
            inpaint_on = bool(inpaint) and have_inpaint
            if inpaint_on:
                lama_loader = g.node("INPAINT_LoadInpaintModel", model_name="big-lama.pt")
                if (upscale_model or "").strip():
                    upscaler = g.node("UpscaleModelLoader",
                                      model_name=upscale_model.strip())
                notes.append("per-band LaMa inpaint ON"
                             + (" + upscale model" if upscaler else ""))

            for i in range(n_bands):  # i=0 farthest
                near, far = edges[n_bands - 1 - i], edges[n_bands - i]
                override = f"near_pct={near:.3f} far_pct={far:.3f}"
                if vlm is not None:
                    override = vlm.out(_ASSESS_OUTPUT_SLOTS["band_override"][i])
                name = (_ATLAS_INPUT_BAND_NAMES[i] if n_bands == 4
                        else f"band_{n_bands - i}")

                # scope: manual line (or VLM prompt) -> SAM -> AtlasScopeMask
                exclude_ref = sky_mask_ref if sky_on else zero_mask.out(0)
                prompt_val = scope_lines[i] if i < len(scope_lines) else ""
                if vlm is not None:
                    prompt_val = None  # replaced by the VLM output below
                wants_scope = (vlm is not None and vlm_scope) or bool(prompt_val)
                if wants_scope:
                    p_ref = vlm.out(_ASSESS_OUTPUT_SLOTS["sam_prompt_band"][i]) if vlm is not None else prompt_val
                    seg_ref = segment(image_ref, p_ref)
                    if seg_ref is None:
                        if prompt_val:
                            notes.append(f"{name} scope SKIPPED — no segmenter (native SAM3 / AtlasSemanticMask absent)")
                    else:
                        scope = g.node("AtlasScopeMask",
                                       sky_mask=(sky_mask_ref if sky_on else zero_mask.out(0)),
                                       prompt=p_ref, segment_mask=seg_ref)
                        exclude_ref = scope.out(0)

                # plate: original photo, or the inpainted clean plate
                plate_ref = image_ref
                if inpaint_on and i < n_bands - 1:  # frontmost band never needs it
                    band_mask = g.node("AtlasDepthLayerMask", solve=solve.out(0),
                                       depth=depth.out(0), band_override=override,
                                       exclude_mask=exclude_ref, **band_ref_kw)
                    grown = g.node("INPAINT_ExpandMask", mask=band_mask.out(1),
                                   grow=32, blur=8, blur_type="gaussian")
                    crop = g.node("AtlasInpaintCrop", image=image_ref,
                                  mask=grown.out(0), context_pad_px=128)
                    lama_kw = {"optional_upscale_model": upscaler.out(0)} if upscaler else {}
                    lama = g.node("INPAINT_InpaintWithModel",
                                  inpaint_model=lama_loader.out(0),
                                  image=crop.out(0), mask=crop.out(1), seed=0,
                                  **lama_kw)
                    stitch = g.node("AtlasInpaintStitch", original_image=image_ref,
                                    inpainted_crop=lama.out(0),
                                    crop_region=crop.out(2))
                    plate_ref = stitch.out(0)

                layer_kw = {}
                if vlm is not None:
                    layer_kw["geometry_override"] = vlm.out(_ASSESS_OUTPUT_SLOTS["geom_band"][i])

                # DMP seam doctrine (artist-corrected 2026-07-12): the
                # extension/outpaint belongs on the layer BEHIND — the front
                # layer keeps a clean cut matte. So the frontmost band gets
                # NO extend/outpaint/skirt, every band behind gets the
                # generous smear (64/1.5/64) that covers the seam on camera
                # move. Priorities are FARTHEST-HIGHEST for the same reason:
                # at a watertight seam the two bands' surfaces are depth-
                # adjacent, and the near-tie priority bias decides which
                # paints — with nearest-highest (the first attempt) each
                # band's smear rendered IN FRONT of the layer behind it
                # (striped columns at every seam, artist-reported); with
                # farthest-highest the behind layer's real pixels win the
                # seam ribbon and its extension shows only in true reveals,
                # while genuinely nearer real geometry still wins by plain
                # depth test.
                is_front = (i == n_bands - 1)
                layer = g.node("AtlasCleanPlateLayer", solve=solve_chain,
                               depth=depth.out(0), plate_image=plate_ref,
                               band_override=override, name=name,
                               priority=float(5 * (n_bands - 1 - i)),
                               relief_grid=int(mesh_resolution),
                               depth_edge_rel=1.5,
                               max_edge_factor=float(max_edge_factor),
                               normal_edge_deg=float(normal_edge_deg),
                               fill_occluded=(inpaint_on and i < n_bands - 1),
                               embed_matte=True,
                               edge_extend_px=0 if is_front else int(edge_extend_px),
                               skirt_bevel=0.0 if is_front else 1.5,
                               frame_outpaint_px=0 if is_front else 64,
                               band_geometry=mesh,
                               exclude_mask=exclude_ref, **band_ref_kw, **layer_kw)
                solve_chain = layer.out(0)
            notes.append(f"{n_bands} band layer(s), grid {int(mesh_resolution)}")

        report = "ATLAS INPUT — expanded graph\n" + "\n".join(f"  · {n}" for n in notes)
        return {
            "result": (solve_chain, image_ref, depth.out(0), sky_mask_ref, report),
            "expand": g.finalize(),
        }
