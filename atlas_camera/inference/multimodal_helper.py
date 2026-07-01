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
        "the solve. Use this Atlas app context:\n"
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


def _parse_model_json(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {"summary": str(parsed)}
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(content[start : end + 1])
                return parsed if isinstance(parsed, dict) else {"summary": content}
            except json.JSONDecodeError:
                pass
    return {
        "summary": content or "Ollama returned an empty response.",
        "warnings": ["Model response was not valid JSON."],
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
    model_id = str(payload.get("id") or payload.get("model") or payload.get("name") or "unknown")
    capabilities = payload.get("capabilities")
    capability_names, vision_capable = _capability_info(capabilities)
    return ProviderModelInfo(
        id=model_id,
        name=str(payload.get("name") or model_id),
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
