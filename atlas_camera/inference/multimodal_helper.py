"""Future multimodal helper interfaces.

This module intentionally contains no model dependency. A small multimodal LLM
can later suggest scene objects and scale references, but the artist and the
deterministic solver remain in control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import base64
import mimetypes
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib import error, request


ProviderName = str


@dataclass(slots=True)
class SceneScaleCue:
    label: str
    confidence: float
    bbox_px: tuple[float, float, float, float] | None = None
    suggested_reference_ids: list[str] = field(default_factory=list)
    notes: str | None = None
    source: str = "multimodal_helper"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MultimodalSceneObservation:
    image_path: str
    summary: str
    scale_cues: list[SceneScaleCue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_response: dict[str, Any] | None = None
    model: str | None = None
    provider: str | None = None
    base_url: str | None = None
    vision_capable: bool | None = None
    diagnostic_status: str | None = None
    scene_description: str | None = None
    scale_candidates: list[str] = field(default_factory=list)
    perspective_cues: list[str] = field(default_factory=list)
    lens_distortion_notes: list[str] = field(default_factory=list)
    occlusion_notes: list[str] = field(default_factory=list)
    recommended_guides: list[str] = field(default_factory=list)
    technical_guidance: list[str] = field(default_factory=list)
    solve_risk_notes: list[str] = field(default_factory=list)
    dataset_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "summary": self.summary,
            "scale_cues": [cue.to_dict() for cue in self.scale_cues],
            "warnings": list(self.warnings),
            "raw_response": self.raw_response,
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "vision_capable": self.vision_capable,
            "diagnostic_status": self.diagnostic_status,
            "scene_description": self.scene_description,
            "scale_candidates": list(self.scale_candidates),
            "perspective_cues": list(self.perspective_cues),
            "lens_distortion_notes": list(self.lens_distortion_notes),
            "occlusion_notes": list(self.occlusion_notes),
            "recommended_guides": list(self.recommended_guides),
            "technical_guidance": list(self.technical_guidance),
            "solve_risk_notes": list(self.solve_risk_notes),
            "dataset_evidence": list(self.dataset_evidence),
        }


@dataclass(slots=True)
class ProviderModelInfo:
    id: str
    name: str
    vision_capable: bool | None
    capabilities: list[str] = field(default_factory=list)
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "vision_capable": self.vision_capable,
            "capabilities": list(self.capabilities),
            "raw": self.raw,
        }


class MultimodalSceneHelper:
    """Interface for a future small vision-language scale assistant."""

    def analyze_image(
        self,
        image_path: str | Path,
        *,
        prompt: str | None = None,
        candidate_reference_ids: list[str] | None = None,
    ) -> MultimodalSceneObservation:
        raise NotImplementedError(
            "Multimodal scene helper is planned but not implemented. "
            "Use this interface for future local or hosted vision-language models."
        )


class MultimodalProvider(MultimodalSceneHelper):
    """Provider-neutral local vision-language assistant interface."""

    def __init__(
        self,
        *,
        model: str = "",
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @property
    def provider(self) -> str:
        raise NotImplementedError

    def list_models(self) -> list[ProviderModelInfo]:
        raise NotImplementedError

    def validate_vision_model(self) -> ProviderModelInfo:
        models = self.list_models()
        if not models:
            raise RuntimeError(f"{self.provider} returned no available models.")

        selected = _select_model(models, self.model)
        if selected is None:
            ids = ", ".join(model.id for model in models[:8])
            raise RuntimeError(
                f"{self.provider} model '{self.model or '(blank)'}' is not available. "
                f"Available models: {ids or 'none'}."
            )

        if selected.vision_capable is False:
            capabilities = ", ".join(selected.capabilities) or "none reported"
            raise RuntimeError(
                f"Selected {self.provider} model '{selected.id}' does not advertise "
                f"image/vision capability ({capabilities}). Choose a multimodal vision model."
            )
        return selected

    def _request_json(self, endpoint: str, payload: dict[str, Any] | None = None, *, base_url: str | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(
            f"{(base_url or self.base_url).rstrip('/')}{endpoint}",
            data=data,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            detail = _provider_error_detail(body) or exc.reason or "request rejected"
            raise RuntimeError(f"{self.provider} request failed ({exc.code}): {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Unable to reach {self.provider} at {self.base_url}. Start the local "
                "provider and load/select a multimodal vision model."
            ) from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{self.provider} returned invalid JSON.") from exc


class OpenAICompatibleVisionProvider(MultimodalProvider):
    """OpenAI-compatible local vision provider for LM Studio and llama.cpp."""

    def __init__(
        self,
        *,
        model: str = "",
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key or "not-needed",
            timeout_seconds=timeout_seconds,
        )

    def analyze_image(
        self,
        image_path: str | Path,
        *,
        prompt: str | None = None,
        candidate_reference_ids: list[str] | None = None,
        app_context: dict[str, Any] | None = None,
    ) -> MultimodalSceneObservation:
        model_info = self.validate_vision_model()
        path = Path(image_path)
        payload = {
            "model": model_info.id,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": _system_prompt(),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt or _user_prompt(candidate_reference_ids, app_context),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(path)},
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 1800,
        }
        response_format = self.response_format()
        if response_format is not None:
            payload["response_format"] = response_format
        response_format_fallback = False
        try:
            response = self._request_json("/chat/completions", payload)
        except RuntimeError as exc:
            if response_format is None or not _is_response_format_error(str(exc)):
                raise
            response_format_fallback = True
            text_payload = {**payload, "response_format": {"type": "text"}}
            try:
                response = self._request_json("/chat/completions", text_payload)
            except RuntimeError as text_exc:
                if not _is_response_format_error(str(text_exc)):
                    raise
                plain_payload = dict(payload)
                plain_payload.pop("response_format", None)
                response = self._request_json("/chat/completions", plain_payload)
        content = _openai_chat_content(response)
        parsed = _parse_model_json(content)
        observation = _observation_from_model_payload(
            path,
            parsed,
            raw_response=response,
            model=model_info.id,
            provider=self.provider,
            base_url=self.base_url,
            vision_capable=model_info.vision_capable,
            diagnostic_status="available",
        )
        if response_format_fallback:
            observation.warnings.append(
                f"{self.provider} rejected structured response_format; retried with text JSON prompting."
            )
        return observation

    def response_format(self) -> dict[str, Any] | None:
        return None


class LMStudioVisionProvider(OpenAICompatibleVisionProvider):
    @property
    def provider(self) -> str:
        return "lmstudio"

    def response_format(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "atlas_camera_guidance",
                "schema": _scene_observation_json_schema(),
                "strict": False,
            },
        }

    def list_models(self) -> list[ProviderModelInfo]:
        root_url = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        data = self._request_json("/api/v1/models", base_url=root_url)
        return [_model_info_from_lmstudio(item) for item in _model_items(data)]


class LlamaCppVisionProvider(OpenAICompatibleVisionProvider):
    @property
    def provider(self) -> str:
        return "llamacpp"

    def list_models(self) -> list[ProviderModelInfo]:
        data = self._request_json("/models")
        items = _model_items(data)
        models = [_model_info_from_openai_compatible(item, default_vision=True) for item in items]
        if not models and self.model:
            return [
                ProviderModelInfo(
                    id=self.model,
                    name=self.model,
                    vision_capable=True,
                    capabilities=["assumed_vision_capable"],
                )
            ]
        return models


class OpenAIVisionProvider(OpenAICompatibleVisionProvider):
    """OpenAI-compatible CLOUD provider — for users without local models.

    Default base_url is api.openai.com, but any OpenAI-compatible hosted
    endpoint works via base_url (OpenRouter, and the Gemini/xAI/Mistral
    compatibility endpoints). Auth is the Bearer `api_key` header
    `_request_json` already sends; the key can come from the node widget or
    the OPENAI_API_KEY environment variable (preferred — a widget value is
    saved into workflow JSON, which artists share).
    """

    def __init__(
        self,
        *,
        model: str = "",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        # Skip OpenAICompatibleVisionProvider's "not-needed" key placeholder:
        # a cloud endpoint genuinely needs a key, and validate_vision_model
        # checks for its absence to fail with an actionable message.
        MultimodalProvider.__init__(
            self,
            model=model,
            base_url=base_url,
            api_key=(api_key or "").strip() or os.environ.get("OPENAI_API_KEY") or None,
            timeout_seconds=timeout_seconds,
        )

    @property
    def provider(self) -> str:
        return "openai"

    def validate_vision_model(self) -> ProviderModelInfo:
        if not self.api_key:
            raise RuntimeError(
                "openai provider needs an API key — set the node's api_key "
                "(or the OPENAI_API_KEY environment variable, preferred so "
                "the key never lands in shared workflow JSON)."
            )
        if self.model:
            # Cloud catalogs are huge and /models doesn't advertise vision
            # capability — trust the explicit id; a typo fails loudly at the
            # chat call with the host's own error message.
            return ProviderModelInfo(
                id=self.model,
                name=self.model,
                vision_capable=True,
                capabilities=["assumed_vision_capable"],
            )
        return super().validate_vision_model()

    def list_models(self) -> list[ProviderModelInfo]:
        data = self._request_json("/models")
        return [
            _model_info_from_openai_compatible(item, default_vision=True)
            for item in _model_items(data)
        ]

    def response_format(self) -> dict[str, Any]:
        # Same guided-JSON request LM Studio uses; hosts that reject it fall
        # back through analyze_image's _is_response_format_error chain.
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "atlas_camera_guidance",
                "schema": _scene_observation_json_schema(),
                "strict": False,
            },
        }


class OllamaVisionProvider(MultimodalProvider):
    """Local Ollama vision-language helper for advisory scene analysis."""

    def __init__(
        self,
        *,
        model: str = "gemma3:4b",
        base_url: str = "http://127.0.0.1:11434",
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )

    @property
    def provider(self) -> str:
        return "ollama"

    def list_models(self) -> list[ProviderModelInfo]:
        tags = self._request_json("/api/tags")
        models = tags.get("models", [])
        return [_model_info_from_ollama(item) for item in models if isinstance(item, dict)]

    def analyze_image(
        self,
        image_path: str | Path,
        *,
        prompt: str | None = None,
        candidate_reference_ids: list[str] | None = None,
        app_context: dict[str, Any] | None = None,
    ) -> MultimodalSceneObservation:
        model_info = self.validate_vision_model()
        path = Path(image_path)
        payload = {
            "model": model_info.id,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": _system_prompt(),
                },
                {
                    "role": "user",
                    "content": prompt or _user_prompt(candidate_reference_ids, app_context),
                    "images": [_image_base64(path)],
                },
            ],
            "format": "json",
            "options": {"num_predict": 1800},
        }
        response = self._request_json("/api/chat", payload)
        content = str(response.get("message", {}).get("content", "")).strip()
        parsed = _parse_model_json(content)
        return _observation_from_model_payload(
            path,
            parsed,
            raw_response=response,
            model=model_info.id,
            provider=self.provider,
            base_url=self.base_url,
            vision_capable=model_info.vision_capable,
            diagnostic_status="available",
        )


class OllamaVisionSceneHelper(OllamaVisionProvider):
    """Backward-compatible name for existing imports."""


def create_multimodal_provider(
    provider: ProviderName,
    *,
    model: str = "",
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 120.0,
) -> MultimodalProvider:
    provider_id = provider.strip().lower() or "lmstudio"
    if provider_id == "lmstudio":
        return LMStudioVisionProvider(
            model=model,
            base_url=base_url or "http://127.0.0.1:1234/v1",
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    if provider_id == "llamacpp":
        return LlamaCppVisionProvider(
            model=model,
            base_url=base_url or "http://127.0.0.1:8080/v1",
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    if provider_id == "ollama":
        return OllamaVisionProvider(
            model=model or "gemma3:4b",
            base_url=base_url or "http://127.0.0.1:11434",
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    if provider_id == "openai":
        return OpenAIVisionProvider(
            model=model or "gpt-4o-mini",
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def provider_models_response(
    provider: ProviderName,
    *,
    model: str = "",
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    helper = create_multimodal_provider(
        provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    models = helper.list_models()
    selected = _select_model(models, helper.model)
    return {
        "provider": helper.provider,
        "base_url": helper.base_url,
        "model": selected.id if selected is not None else helper.model,
        "models": [item.to_dict() for item in models],
        "vision_capable": selected.vision_capable if selected is not None else None,
        "diagnostic_status": _provider_diagnostic_status(models, selected),
    }


def _system_prompt() -> str:
    return (
        "You are Atlas Camera's local vision assistant. You advise a deterministic "
        "still-image camera solver for matte painting and projection setup. Do not claim "
        "a camera is solved unless the provided matrices and guide counts support it. "
        "Return strict JSON with keys: summary, scene_description, scale_candidates, "
        "scale_cues, perspective_cues, lens_distortion_notes, occlusion_notes, "
        "recommended_guides, technical_guidance, solve_risk_notes, dataset_evidence, "
        "warnings. Pay attention to rough scale anchors such as people, boxes, "
        "corridor widths, wall heights, floor seams, clothing, equipment, and debris. "
        "Keep outputs concise and cite uncertainty. Treat ETH3D/DTU-style calibrated datasets as validation "
        "evidence, not as proof for this image."
    )


def _user_prompt(
    candidate_reference_ids: list[str] | None,
    app_context: dict[str, Any] | None,
) -> str:
    return (
        "Analyze the image for camera-solve guidance. Identify visible scale anchors, "
        "rough object-size candidates, perspective families, horizon/depth cues, lens "
        "distortion risks, occlusions, and which additional artist guides would improve "
        "the solve. For every scale_cue, report bbox_px as [x0, y0, x1, y1] in pixel "
        "coordinates (top-left corner, then bottom-right corner) tightly enclosing the "
        "object's full visible extent — for an upright object include its base (feet/"
        "tyres) and its top. Prefer scale anchors that stand on the ground. Use this "
        "Atlas app context:\n"
        f"{json.dumps(app_context or {}, indent=2, sort_keys=True)}\n"
        "Candidate scale reference IDs:\n"
        f"{json.dumps(candidate_reference_ids or [], indent=2)}\n"
        "Return JSON only."
    )


def _scene_observation_json_schema() -> dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "summary": {"type": "string"},
            "scene_description": {"type": "string"},
            "scale_candidates": string_array,
            "scale_cues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "label": {"type": "string"},
                        "confidence": {"type": "number"},
                        "bbox_px": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "suggested_reference_ids": string_array,
                        "notes": {"type": "string"},
                    },
                },
            },
            "perspective_cues": string_array,
            "lens_distortion_notes": string_array,
            "occlusion_notes": string_array,
            "recommended_guides": string_array,
            "technical_guidance": string_array,
            "solve_risk_notes": string_array,
            "dataset_evidence": string_array,
            "warnings": string_array,
        },
        "required": ["summary"],
    }


def _image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{_image_base64(path)}"


_MAX_RESPONSE_CHARS = 32_000
_LOOP_REPEAT_RE = re.compile(r"(.{4,24})\1{11,}", re.DOTALL)


def _truncate_looping_response(content: str) -> tuple[str, bool]:
    """Detect a repetition loop (common in local LLMs) and truncate before it starts."""
    m = _LOOP_REPEAT_RE.search(content)
    if m and m.start() > 0:
        return content[: m.start()], True
    if len(content) > _MAX_RESPONSE_CHARS:
        return content[:_MAX_RESPONSE_CHARS], True
    return content, False


def _close_partial_json(text: str) -> dict[str, Any] | None:
    """Synthesize closing brackets to parse truncated JSON from a looping model."""
    if not text.strip().startswith("{"):
        return None
    stack: list[str] = []
    in_str = False
    esc = False
    last_outer_comma_pos = -1
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch == "}" and stack and stack[-1] == "}":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "]":
            stack.pop()
        if ch == "," and len(stack) == 1:
            last_outer_comma_pos = i
    if not stack:
        return None
    closing = "".join(reversed(stack))
    stripped = text.rstrip().rstrip(",").rstrip()
    try:
        result = json.loads(stripped + closing)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    if last_outer_comma_pos > 0:
        try:
            result = json.loads(text[:last_outer_comma_pos].rstrip() + "}")
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    return None


def _extend_warnings(payload: dict[str, Any], extra: list[str]) -> None:
    if not extra:
        return
    existing = payload.get("warnings")
    if isinstance(existing, list):
        existing.extend(extra)
    else:
        payload["warnings"] = list(extra)


def _parse_model_json(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {"summary": str(parsed)}
    except json.JSONDecodeError:
        pass

    truncated, was_looping = _truncate_looping_response(content)
    loop_warnings: list[str] = (
        ["Model response contained a repetition loop; partial data was recovered."]
        if was_looping
        else []
    )
    working = truncated if was_looping else content

    start = working.find("{")
    end = working.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(working[start : end + 1])
            if isinstance(parsed, dict):
                _extend_warnings(parsed, loop_warnings)
                return parsed
        except json.JSONDecodeError:
            pass

    # Try bracket-closing recovery for ANY unparseable tail, not just loops —
    # a max_tokens truncation cuts JSON mid-object with no repetition to detect.
    if start >= 0:
        recovered = _close_partial_json(working[start:])
        if recovered is not None:
            _extend_warnings(recovered, loop_warnings or
                             ["Model response was truncated; partial data was recovered."])
            return recovered

    return {
        "summary": working or "Model returned an empty response.",
        "warnings": loop_warnings + ["Model response was not valid JSON."],
    }


def _provider_error_detail(body: str) -> str:
    if not body.strip():
        return ""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body.strip()
    if isinstance(payload, dict):
        for key in ("error", "detail", "message"):
            value = payload.get(key)
            if value:
                return str(value)
    return body.strip()


def _is_response_format_error(message: str) -> bool:
    lowered = message.lower()
    return "response_format" in lowered and (
        "json_schema" in lowered
        or "json_object" in lowered
        or "must be" in lowered
        or "unsupported" in lowered
        or "invalid" in lowered
    )


def _observation_from_model_payload(
    image_path: Path,
    payload: dict[str, Any],
    *,
    raw_response: dict[str, Any],
    model: str,
    provider: str,
    base_url: str | None = None,
    vision_capable: bool | None = None,
    diagnostic_status: str | None = None,
) -> MultimodalSceneObservation:
    return MultimodalSceneObservation(
        image_path=str(image_path),
        summary=str(payload.get("summary") or "No summary returned."),
        scale_cues=[
            _scene_scale_cue_from_payload(item, source=provider)
            for item in payload.get("scale_cues", [])
            if isinstance(item, dict)
        ],
        warnings=_model_string_list(payload.get("warnings")),
        raw_response=raw_response,
        model=model,
        provider=provider,
        base_url=base_url,
        vision_capable=vision_capable,
        diagnostic_status=diagnostic_status,
        scene_description=_optional_string(payload.get("scene_description")),
        scale_candidates=_model_string_list(payload.get("scale_candidates")),
        perspective_cues=_model_string_list(payload.get("perspective_cues")),
        lens_distortion_notes=_model_string_list(payload.get("lens_distortion_notes")),
        occlusion_notes=_model_string_list(payload.get("occlusion_notes")),
        recommended_guides=_model_string_list(payload.get("recommended_guides")),
        technical_guidance=_model_string_list(payload.get("technical_guidance")),
        solve_risk_notes=_model_string_list(payload.get("solve_risk_notes")),
        dataset_evidence=_model_string_list(payload.get("dataset_evidence")),
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _openai_chat_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value:
                    parts.append(str(value))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _model_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data", payload.get("models", payload))
    if isinstance(data, dict):
        data = data.get("models", data.get("data", []))
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _select_model(models: list[ProviderModelInfo], model: str) -> ProviderModelInfo | None:
    if not models:
        return None
    if not model:
        return next((item for item in models if item.vision_capable is True), models[0])
    return next(
        (
            item
            for item in models
            if model in {item.id, item.name}
        ),
        None,
    )


def _provider_diagnostic_status(models: list[ProviderModelInfo], selected: ProviderModelInfo | None) -> str:
    if selected is None:
        return "no matching model selected" if models else "no models returned"
    if selected.vision_capable is True:
        return "selected model advertises vision capability"
    if selected.vision_capable is False:
        return "selected model is not vision-capable"
    return "provider did not report vision capability"


def _model_info_from_lmstudio(payload: dict[str, Any]) -> ProviderModelInfo:
    # LM Studio's own /api/v1/models (confirmed live 2026-07-04, LM Studio serving
    # google/gemma-4-12b-qat) uses "key" as the model identifier and "display_name"
    # for the human label -- it has neither "id" nor "model" nor "name". Without
    # "key" first, every model here resolves to the "unknown" fallback, which then
    # fails validate_vision_model()'s lookup against any requested model name and
    # makes AtlasVLMScaleCues fail soft to an empty scale_references list even
    # with a real, running, vision-capable model loaded. The OpenAI-compatible
    # /v1/models endpoint (used by list_models() for other providers) does use
    # "id", so both are kept as fallbacks for older/other LM Studio API surfaces.
    model_id = str(
        payload.get("key") or payload.get("id") or payload.get("model") or payload.get("name") or "unknown"
    )
    capabilities = payload.get("capabilities")
    capability_names, vision_capable = _capability_info(capabilities)
    return ProviderModelInfo(
        id=model_id,
        name=str(payload.get("display_name") or payload.get("name") or model_id),
        vision_capable=vision_capable,
        capabilities=capability_names,
        raw=payload,
    )


def _model_info_from_openai_compatible(payload: dict[str, Any], *, default_vision: bool | None) -> ProviderModelInfo:
    model_id = str(payload.get("id") or payload.get("model") or payload.get("name") or "unknown")
    capabilities = payload.get("capabilities")
    capability_names, vision_capable = _capability_info(capabilities)
    return ProviderModelInfo(
        id=model_id,
        name=str(payload.get("name") or model_id),
        vision_capable=vision_capable if vision_capable is not None else default_vision,
        capabilities=capability_names or (["assumed_vision_capable"] if default_vision else []),
        raw=payload,
    )


def _model_info_from_ollama(payload: dict[str, Any]) -> ProviderModelInfo:
    model_id = str(payload.get("name") or payload.get("model") or "unknown")
    capability_names, vision_capable = _capability_info(payload.get("capabilities"))
    return ProviderModelInfo(
        id=model_id,
        name=str(payload.get("model") or model_id),
        vision_capable=vision_capable,
        capabilities=capability_names,
        raw=payload,
    )


def _capability_info(value: Any) -> tuple[list[str], bool | None]:
    if isinstance(value, dict):
        capabilities = [str(key).lower() for key, enabled in value.items() if bool(enabled)]
        if any(name in {"vision", "image", "images"} for name in capabilities):
            return capabilities, True
        return capabilities, False if capabilities else None
    if isinstance(value, list):
        capabilities = [str(item).lower() for item in value]
        if any(name in {"vision", "image", "images"} for name in capabilities):
            return capabilities, True
        return capabilities, False if capabilities else None
    return [], None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        lines = [
            line.strip(" -\t")
            for line in value.splitlines()
            if line.strip(" -\t")
        ]
        return lines or ([value.strip()] if value.strip() else [])
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if isinstance(item, str):
                items.extend(_string_list(item))
            elif item is not None:
                items.append(json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item))
        return items
    if isinstance(value, dict):
        return [json.dumps(value, sort_keys=True)]
    return [str(value)]


def _model_string_list(value: Any) -> list[str]:
    """Apply space-stripped-word repair to strings parsed from raw model output."""
    return [_humanize_guidance_text(s) for s in _string_list(value)]


_GUIDANCE_WORDS = {
    "a",
    "accuracy",
    "accurate",
    "accurately",
    "achieve",
    "add",
    "adequate",
    "anchor",
    "anchors",
    "and",
    "as",
    "at",
    "building",
    "calibration",
    "camera",
    "comparable",
    "consider",
    "considered",
    "constraints",
    "current",
    "data",
    "datasets",
    "deficit",
    "defined",
    "demonstrate",
    "depth",
    "does",
    "distortion",
    "dtu",
    "due",
    "elements",
    "establish",
    "estimation",
    "eth3d",
    "evidence",
    "exhibit",
    "explicit",
    "family",
    "for",
    "guide",
    "guidelines",
    "guides",
    "height",
    "high",
    "highly",
    "horizon",
    "image",
    "immediately",
    "incorrect",
    "is",
    "least",
    "left",
    "like",
    "line",
    "lines",
    "may",
    "measuring",
    "metadata",
    "minimum",
    "necessary",
    "not",
    "of",
    "on",
    "one",
    "only",
    "or",
    "parameter",
    "parameters",
    "perspective",
    "provide",
    "projected",
    "recommended",
    "reference",
    "references",
    "reflects",
    "relationships",
    "require",
    "risk",
    "right",
    "robust",
    "scale",
    "scenario",
    "scene",
    "scenes",
    "significant",
    "solve",
    "solved",
    "solver",
    "solely",
    "spatial",
    "solutions",
    "state",
    "subsequently",
    "substantial",
    "such",
    "the",
    "these",
    "that",
    "to",
    "two",
    "using",
    "vanishing",
    "vertical",
    "which",
    "without",
}


def _humanize_guidance_text(value: str) -> str:
    if not _looks_space_stripped(value):
        return value

    repaired: list[str] = []
    token = ""
    for char in value:
        if char.isalnum() or char in "_'":
            token += char
            continue
        if token:
            repaired.append(_segment_compound_token(token))
            token = ""
        repaired.append(char)
    if token:
        repaired.append(_segment_compound_token(token))

    text = "".join(repaired)
    text = text.replace(" ,", ",").replace(" .", ".").replace("( ", "(").replace(" )", ")")
    text = re.sub(r"([.,;:!?])(?=\S)", r"\1 ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return " ".join(text.split())


def _looks_space_stripped(value: str) -> bool:
    compact = "".join(char for char in value if char.isalnum())
    if len(compact) < 32:
        return False
    if " " in value:
        return False
    lowered = compact.lower()
    hints = ("guideline", "vanishing", "reference", "dataset", "metadata", "solver", "camera")
    return any(hint in lowered for hint in hints)


def _segment_compound_token(value: str) -> str:
    index = 0
    words: list[str] = []
    while index < len(value):
        match = _longest_guidance_word(value, index)
        if match is None:
            next_index = index + 1
            while next_index < len(value) and _longest_guidance_word(value, next_index) is None:
                next_index += 1
            words.append(value[index:next_index])
            index = next_index
        else:
            words.append(value[index : index + match])
            index += match
    return " ".join(words)


def _longest_guidance_word(value: str, index: int) -> int | None:
    lowered = value[index:].lower()
    if lowered.startswith("solvedue"):
        return len("solve")
    matches = [
        len(word)
        for word in _GUIDANCE_WORDS
        if lowered.startswith(word)
    ]
    return max(matches) if matches else None


# Keyword → scale-reference registry id, for cues whose suggested_reference_ids
# don't already name a registry entry.
_LABEL_TO_REFERENCE_ID = {
    "person": "person_175cm", "man": "person_175cm", "woman": "person_175cm",
    "human": "person_175cm", "pedestrian": "person_175cm", "figure": "person_175cm",
    "worker": "person_175cm", "people": "person_175cm",
    "door": "door_210cm", "doorway": "door_210cm", "entrance": "door_210cm",
    "car": "sedan_car", "sedan": "sedan_car", "vehicle": "sedan_car",
    "automobile": "sedan_car", "hatchback": "sedan_car",
    "bus": "city_bus",
    "container": "shipping_container_20ft", "shipping container": "shipping_container_20ft",
    "hoop": "basketball_hoop_rim", "basketball hoop": "basketball_hoop_rim",
}


def _resolve_reference_id(cue: "SceneScaleCue") -> str | None:
    """Best-effort map a scale cue to a scale-reference registry id."""
    from atlas_camera.reference_data import get_scale_reference, search_scale_references

    for rid in cue.suggested_reference_ids:
        try:
            get_scale_reference(str(rid))
            return str(rid)
        except KeyError:
            continue
    label = (cue.label or "").strip().lower()
    if not label:
        return None
    if label in _LABEL_TO_REFERENCE_ID:
        return _LABEL_TO_REFERENCE_ID[label]
    for key, rid in _LABEL_TO_REFERENCE_ID.items():
        if key in label:
            return rid
    matches = search_scale_references(label)
    return matches[0].id if matches else None


def scale_references_from_observation(
    observation: "MultimodalSceneObservation",
    *,
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """Convert a VLM scene observation into solver ``scale_references`` specs.

    Only cues with a pixel bounding box and a resolvable real height (via a
    registry ``reference_id`` or an explicit ``height`` on the cue) become specs.
    The result feeds ``solve_still_image_learned(scale_references=...)`` /
    ``apply_reference_scale`` — but adoption still requires explicit confirmation
    (LLM cues are never auto-promoted).
    """
    refs: list[dict[str, Any]] = []
    for cue in observation.scale_cues:
        if cue.bbox_px is None or float(cue.confidence) < min_confidence:
            continue
        spec: dict[str, Any] = {
            "bbox_px": [float(v) for v in cue.bbox_px],
            "confidence": float(cue.confidence),
            "label": cue.label,
            "source": "vlm_scale_cue",
        }
        reference_id = _resolve_reference_id(cue)
        if reference_id:
            spec["reference_id"] = reference_id
        else:
            # No registry match — only usable if the cue itself carries a height.
            raw = getattr(cue, "height_m", None)
            if not raw:
                continue
            spec["height_m"] = float(raw)
        refs.append(spec)
    return refs


def _scene_scale_cue_from_payload(payload: dict[str, Any], *, source: str = "multimodal_helper") -> SceneScaleCue:
    bbox = payload.get("bbox_px")
    return SceneScaleCue(
        label=str(payload.get("label", "unknown")),
        confidence=float(payload.get("confidence", 0.0)),
        bbox_px=tuple(float(value) for value in bbox) if isinstance(bbox, list) and len(bbox) == 4 else None,  # type: ignore[arg-type]
        suggested_reference_ids=[str(item) for item in payload.get("suggested_reference_ids", [])],
        notes=payload.get("notes"),
        source=source,
    )
