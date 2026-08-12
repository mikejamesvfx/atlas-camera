"""LTX temporal generation through a running ComfyUI (spec §17/§18).

Integration decision (spec §17 step 3): the cleanest supported local path on
this project is the EXISTING headless ComfyUI bridge
(`atlas_camera.mcp.comfy_http`, stdlib-only, polling HTTP) driving the user's
installed ComfyUI-LTXVideo pack — not a direct-Python LTX integration, which
would drag torch/diffusers into Atlas and duplicate an install the user
already maintains. No LTX function signatures are assumed: the adapter is
TEMPLATE-DRIVEN — it loads a ComfyUI workflow JSON (UI or API format) that the
user/site provides, verifies every class_type it needs exists on the live
server, and only overrides well-known inputs:

    LoadImage.image          -> the uploaded plate crop
    CLIPTextEncode.text      -> config.prompt   (marker ``{PROMPT}`` preferred,
                                otherwise the first text encoder)
    seed / noise_seed        -> config.seed
    length / frames / num_frames / batch_size on latent-video nodes
                             -> config.frame_count
    fps / frame_rate         -> config.fps
    width / height           -> config resize (when set)

Template resolution: explicit ``template_path`` arg, else the
``ATLAS_LTX_TEMPLATE`` environment variable. The template must end in
SaveImage-style frame outputs — the frame sequence is the contract; a video
file is at most a preview (spec §21/§22).

Camera preservation (spec §20): plain image-to-video cannot GUARANTEE zero
generator camera drift, so results carry
``metadata["camera_preservation"] = "unverified_i2v"`` and the prompt preset
asks for a locked camera. The Atlas camera itself is never derived from the
generated pixels — the plate stays registered to the crop camera regardless.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from pathlib import Path
from typing import Any

from atlas_camera.dynamic.generators import (
    MODE_IMAGE_TO_VIDEO,
    RESULT_FAILED,
    RESULT_NOT_AVAILABLE,
    RESULT_OK,
    TemporalGenerationConfig,
    TemporalGenerationResult,
)

ENV_TEMPLATE = "ATLAS_LTX_TEMPLATE"
ENV_HOST = "COMFY_HOST"
_FRAME_COUNT_KEYS = ("length", "frames", "num_frames")
_FPS_KEYS = ("fps", "frame_rate")
_SEED_KEYS = ("seed", "noise_seed")
_PROMPT_MARKER = "{PROMPT}"


class LTXComfyGenerator:
    """Template-driven LTX image-to-video adapter over ComfyUI HTTP."""

    name = "ltx"

    def __init__(self, *, host: str | None = None,
                 template_path: str | None = None,
                 timeout: int = 1800) -> None:
        from atlas_camera.mcp import comfy_http as C  # stdlib-only module
        self._C = C
        self.host = host or os.environ.get(ENV_HOST) or C.DEFAULT_HOST
        self.template_path = template_path or os.environ.get(ENV_TEMPLATE)
        self.timeout = int(timeout)

    # ------------------------------------------------------------- template

    def _load_template(self) -> tuple[dict | None, str]:
        if not self.template_path:
            return None, (
                f"no LTX workflow template configured (pass template_path or "
                f"set {ENV_TEMPLATE})")
        path = Path(self.template_path)
        if not path.exists():
            return None, f"LTX template not found: {path}"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"LTX template unreadable: {exc}"
        return data, ""

    def _to_api(self, template: dict) -> tuple[dict | None, str]:
        """UI-format templates are flattened via the existing bridge; API
        format passes through."""
        if "nodes" in template:
            try:
                oi = self._C.fetch_object_info(self.host)
                ui = self._C.expand_subgraphs(template, oi)
                return self._C.ui_to_api(ui, oi), ""
            except Exception as exc:  # noqa: BLE001 - report, never raise
                return None, f"template conversion failed: {exc}"
        return {str(k): v for k, v in template.items()}, ""

    @staticmethod
    def _api_class_types(api: dict) -> set[str]:
        return {str(node.get("class_type", ""))
                for node in api.values() if isinstance(node, dict)}

    # ---------------------------------------------------------- availability

    def available(self) -> tuple[bool, str]:
        template, reason = self._load_template()
        if template is None:
            return False, reason
        try:
            oi = self._C.fetch_object_info(self.host)
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            return False, f"ComfyUI not reachable at {self.host}: {exc}"
        api, reason = self._to_api(template)
        if api is None:
            return False, reason
        missing = sorted(t for t in self._api_class_types(api)
                         if t and t not in oi)
        if missing:
            return False, (
                f"ComfyUI at {self.host} lacks node types {missing[:6]} "
                f"(install the LTX-Video pack)")
        return True, f"ComfyUI at {self.host} serves the LTX template"

    # ------------------------------------------------------------- overrides

    def _apply_overrides(self, api: dict, *, image_name: str,
                         config: TemporalGenerationConfig) -> list[str]:
        notes: list[str] = []
        # The prompt lands wherever the template put a {PROMPT} marker — any
        # string input on any node (a CLIPTextEncode.text, a
        # PrimitiveStringMultiline.value rail, ...). Templates without a
        # marker fall back to the first CLIPTextEncode.
        marker_found = False
        for nid, node in api.items():
            if not isinstance(node, dict):
                continue
            for key, value in node.get("inputs", {}).items():
                if isinstance(value, str) and _PROMPT_MARKER in value:
                    node["inputs"][key] = value.replace(
                        _PROMPT_MARKER, config.prompt or "")
                    notes.append(f"{nid}.{key}=<prompt>")
                    marker_found = True
        fallback_target = None
        if config.prompt and not marker_found:
            for nid, node in api.items():
                if isinstance(node, dict) and \
                        node.get("class_type") == "CLIPTextEncode":
                    fallback_target = nid
                    break
        for nid, node in api.items():
            if not isinstance(node, dict):
                continue
            inputs = node.setdefault("inputs", {})
            ctype = node.get("class_type", "")
            if ctype == "LoadImage" and "image" in inputs:
                inputs["image"] = image_name
                notes.append(f"{nid}.image={image_name}")
            if nid == fallback_target:
                inputs["text"] = config.prompt
                notes.append(f"{nid}.text=<prompt>")
            for key in _SEED_KEYS:
                if config.seed is not None and key in inputs and \
                        isinstance(inputs[key], (int, float)):
                    inputs[key] = int(config.seed)
                    notes.append(f"{nid}.{key}={config.seed}")
            for key in _FRAME_COUNT_KEYS:
                if key in inputs and isinstance(inputs[key], (int, float)):
                    inputs[key] = int(config.frame_count)
                    notes.append(f"{nid}.{key}={config.frame_count}")
            for key in _FPS_KEYS:
                if key in inputs and isinstance(inputs[key], (int, float)):
                    inputs[key] = float(config.fps)
                    notes.append(f"{nid}.{key}={config.fps}")
            if config.width and "width" in inputs and \
                    isinstance(inputs["width"], (int, float)):
                inputs["width"] = int(config.width)
                notes.append(f"{nid}.width={config.width}")
            if config.height and "height" in inputs and \
                    isinstance(inputs["height"], (int, float)):
                inputs["height"] = int(config.height)
                notes.append(f"{nid}.height={config.height}")
        # Site-specific knobs the generic pass cannot know (e.g. the LTX-2.5
        # duration rail): config.extra["overrides"] = {"<nodeId>.<input>": v},
        # same key form as comfy_http.apply_overrides.
        extra = (config.extra or {}).get("overrides") or {}
        if extra:
            self._C.apply_overrides(api, dict(extra))
            notes.extend(f"{k}={v}" for k, v in extra.items())
        return notes

    # -------------------------------------------------------------- download

    def _history_images(self, prompt_id: str) -> list[dict]:
        hist = self._C.http_json(f"http://{self.host}/history/{prompt_id}",
                                 timeout=60)
        rec = hist.get(prompt_id, {})
        images: list[dict] = []
        for node_output in rec.get("outputs", {}).values():
            for image in node_output.get("images", []):
                if image.get("type") == "output":
                    images.append(image)
        return images

    def _download(self, image: dict, dest: Path) -> None:
        query = urllib.parse.urlencode({
            "filename": image.get("filename", ""),
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        })
        url = f"http://{self.host}/view?{query}"
        with urllib.request.urlopen(url, timeout=120) as response:
            dest.write_bytes(response.read())

    # -------------------------------------------------------------- generate

    def generate(self, plate: Any, package_dir: Any,
                 config: TemporalGenerationConfig) -> TemporalGenerationResult:
        ok, reason = self.available()
        base = TemporalGenerationResult(
            status=RESULT_NOT_AVAILABLE, generator=self.name,
            method=MODE_IMAGE_TO_VIDEO, seed=config.seed,
            source_roi=getattr(plate, "source_roi", None),
            crop_camera=getattr(plate, "crop_camera", None),
            metadata={"camera_preservation": "unverified_i2v",
                      "generator_output_color_space": "sRGB",
                      "host": self.host,
                      "template": str(self.template_path or "")})
        if not ok:
            base.warnings.append(reason)
            return base

        package_dir = Path(package_dir)
        crop_path = package_dir / "source" / "crop.png"
        if not crop_path.exists():
            base.status = RESULT_FAILED
            base.warnings.append(f"plate crop missing: {crop_path}")
            return base

        template, _ = self._load_template()
        api, reason = self._to_api(template)
        if api is None:
            base.status = RESULT_FAILED
            base.warnings.append(reason)
            return base

        try:
            image_name = self._C.upload_image(str(crop_path), self.host)
        except Exception as exc:  # noqa: BLE001
            base.status = RESULT_FAILED
            base.warnings.append(f"crop upload failed: {exc}")
            return base

        if config.width is None or config.height is None:
            # None = native crop size (docstring contract): frames that match
            # the crop raster keep the projection registration exact. Video
            # models usually want multiples of 32 — choose the plate overscan
            # so the ROI lands on one, or set explicit width/height.
            roi = getattr(plate, "source_roi", None)
            if roi is not None:
                config.width = config.width or roi.width
                config.height = config.height or roi.height
        overrides = self._apply_overrides(api, image_name=image_name,
                                          config=config)
        base.metadata["overrides"] = overrides

        run = self._C.queue_and_wait(api, self.host, timeout=self.timeout)
        if not run.get("completed"):
            base.status = RESULT_FAILED
            base.warnings.extend(run.get("errors") or ["generation failed"])
            return base

        try:
            images = self._history_images(run["prompt_id"])
        except Exception as exc:  # noqa: BLE001
            base.status = RESULT_FAILED
            base.warnings.append(f"could not read outputs: {exc}")
            return base
        if not images:
            base.status = RESULT_FAILED
            base.warnings.append(
                "workflow produced no image outputs; the LTX template must "
                "end in SaveImage frames (a video file is only a preview)")
            return base

        generated = package_dir / "generated"
        generated.mkdir(parents=True, exist_ok=True)
        frame_paths: list[str] = []
        for index, image in enumerate(images):
            dest = generated / f"frame_{index:04d}.png"
            try:
                self._download(image, dest)
            except Exception as exc:  # noqa: BLE001
                base.status = RESULT_FAILED
                base.warnings.append(
                    f"frame download failed at {index}: {exc}")
                return base
            frame_paths.append(str(dest))

        width = height = 0
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(frame_paths[0]) as im:
                width, height = im.size
        except Exception:  # noqa: BLE001 - dims are best-effort metadata
            base.warnings.append("frame dimensions unverified (no Pillow)")

        base.status = RESULT_OK
        base.frame_paths = frame_paths
        base.frame_count = len(frame_paths)
        base.fps = float(config.fps)
        base.width = width
        base.height = height
        base.model = "ltx-video"
        if len(frame_paths) != int(config.frame_count):
            base.warnings.append(
                f"generator returned {len(frame_paths)} frames, config asked "
                f"for {config.frame_count}")
        return base
