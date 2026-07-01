import pytest
from unittest.mock import patch

from atlas_camera.inference import (
    LMStudioVisionProvider,
    LlamaCppVisionProvider,
    MultimodalSceneHelper,
    MultimodalSceneObservation,
    OllamaVisionSceneHelper,
    ProviderModelInfo,
    SceneScaleCue,
)


def _provider_model(model="gemma3:4b", vision=True):
    return ProviderModelInfo(
        id=model,
        name=model,
        vision_capable=vision,
        capabilities=["completion", "vision"] if vision else ["completion"],
    )


def test_multimodal_observation_serializes_scale_cues_and_provider_metadata():
    observation = MultimodalSceneObservation(
        image_path="concept.png",
        summary="Street scene with a person and car.",
        scale_cues=[
            SceneScaleCue(
                label="person",
                confidence=0.8,
                bbox_px=(10.0, 20.0, 30.0, 80.0),
                suggested_reference_ids=["person_175cm"],
                notes="Standing adult candidate.",
            )
        ],
        scale_candidates=["adult person", "sedan"],
        provider="lmstudio",
        base_url="http://127.0.0.1:1234/v1",
        vision_capable=True,
        warnings=["artist confirmation required"],
    )

    data = observation.to_dict()

    assert data["scale_cues"][0]["suggested_reference_ids"] == ["person_175cm"]
    assert data["scale_candidates"] == ["adult person", "sedan"]
    assert data["provider"] == "lmstudio"
    assert data["vision_capable"] is True
    assert data["warnings"] == ["artist confirmation required"]


def test_multimodal_scene_helper_is_explicit_placeholder():
    helper = MultimodalSceneHelper()

    with pytest.raises(NotImplementedError):
        helper.analyze_image("concept.png")


def test_ollama_helper_parses_advisory_json(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"fake-image")
    response = {
        "message": {
            "content": """
            {
              "summary": "Interior scene with strong linear perspective.",
              "scale_cues": [
                {"label": "door", "confidence": 0.7, "suggested_reference_ids": ["door_210cm"]}
              ],
              "technical_guidance": ["Add vertical guides along door frames."],
              "solve_risk_notes": ["Wide lens distortion may bias vanishing points."],
              "dataset_evidence": ["ETH3D-style calibrated interiors are useful comparison cases."],
              "warnings": ["Confirm scale before export."]
            }
            """
        }
    }

    helper = OllamaVisionSceneHelper(model="gemma3:4b")
    with (
        patch.object(helper, "list_models", return_value=[_provider_model()]),
        patch.object(helper, "_request_json", return_value=response),
    ):
        observation = helper.analyze_image(image, app_context={"guide_counts": {"left": 2}})

    assert observation.model == "gemma3:4b"
    assert observation.provider == "ollama"
    assert observation.scale_cues[0].suggested_reference_ids == ["door_210cm"]
    assert observation.technical_guidance == ["Add vertical guides along door frames."]


def test_lmstudio_provider_parses_openai_compatible_vision_response(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"fake-image")
    models_response = {
        "data": [
            {
                "id": "qwen2.5-vl",
                "name": "Qwen2.5 VL",
                "capabilities": {"completion": True, "vision": True},
            }
        ]
    }
    chat_response = {
        "choices": [
            {
                "message": {
                    "content": """
                    {
                      "summary": "Underground corridor with boxes and a distant person.",
                      "scene_description": "A concrete tunnel with debris and strong receding wall edges.",
                      "scale_candidates": ["distant human silhouette", "cardboard boxes", "corridor wall height"],
                      "perspective_cues": ["floor-wall seams converge near the distant figure"],
                      "lens_distortion_notes": ["Check for wide-angle edge bowing."],
                      "occlusion_notes": ["Foreground boxes occlude floor seam continuity."],
                      "recommended_guides": ["Trace left and right floor-wall seams."]
                    }
                    """
                }
            }
        ]
    }
    helper = LMStudioVisionProvider(model="qwen2.5-vl", base_url="http://127.0.0.1:1234/v1")

    def fake_request(endpoint, payload=None, *, base_url=None):
        if endpoint == "/api/v1/models":
            assert base_url == "http://127.0.0.1:1234"
            return models_response
        if endpoint == "/chat/completions":
            assert payload["messages"][1]["content"][1]["type"] == "image_url"
            assert payload["response_format"]["type"] == "json_schema"
            assert payload["response_format"]["json_schema"]["name"] == "atlas_camera_guidance"
            assert payload["response_format"]["json_schema"]["schema"]["required"] == ["summary"]
            return chat_response
        raise AssertionError(endpoint)

    with patch.object(helper, "_request_json", side_effect=fake_request):
        observation = helper.analyze_image(image)

    assert observation.provider == "lmstudio"
    assert observation.vision_capable is True
    assert observation.scale_candidates == ["distant human silhouette", "cardboard boxes", "corridor wall height"]
    assert observation.recommended_guides == ["Trace left and right floor-wall seams."]


def test_lmstudio_provider_retries_with_text_when_response_format_is_rejected(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"fake-image")
    models_response = {
        "data": [
            {
                "id": "qwen2.5-vl",
                "name": "Qwen2.5 VL",
                "capabilities": {"completion": True, "vision": True},
            }
        ]
    }
    chat_response = {"choices": [{"message": {"content": "{\"summary\":\"retry ok\"}"}}]}
    helper = LMStudioVisionProvider(model="qwen2.5-vl", base_url="http://127.0.0.1:1234/v1")
    seen_formats = []

    def fake_request(endpoint, payload=None, *, base_url=None):
        if endpoint == "/api/v1/models":
            return models_response
        if endpoint == "/chat/completions":
            seen_formats.append(payload.get("response_format"))
            if len(seen_formats) == 1:
                raise RuntimeError(
                    "lmstudio request failed (400): 'response_format.type' must be 'json_schema' or 'text'"
                )
            return chat_response
        raise AssertionError(endpoint)

    with patch.object(helper, "_request_json", side_effect=fake_request):
        observation = helper.analyze_image(image)

    assert seen_formats[0]["type"] == "json_schema"
    assert seen_formats[1]["type"] == "text"
    assert observation.summary == "retry ok"
    assert observation.warnings == [
        "lmstudio rejected structured response_format; retried with text JSON prompting."
    ]


def test_llamacpp_provider_assumes_openai_compatible_model_can_be_vision(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"fake-image")
    helper = LlamaCppVisionProvider(model="gemma-3-4b-it", base_url="http://127.0.0.1:8080/v1")

    def fake_request(endpoint, payload=None, *, base_url=None):
        if endpoint == "/models":
            return {"data": [{"id": "gemma-3-4b-it"}]}
        if endpoint == "/chat/completions":
            return {"choices": [{"message": {"content": "{\"summary\":\"ok\"}"}}]}
        raise AssertionError(endpoint)

    with patch.object(helper, "_request_json", side_effect=fake_request):
        observation = helper.analyze_image(image)

    assert observation.provider == "llamacpp"
    assert observation.vision_capable is True
    assert observation.summary == "ok"


def test_provider_rejects_non_vision_model_before_upload(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"fake-image")
    helper = OllamaVisionSceneHelper(model="gemma3:4b")

    with (
        patch.object(helper, "list_models", return_value=[_provider_model(vision=False)]),
        patch.object(helper, "_request_json") as request_json,
    ):
        with pytest.raises(RuntimeError, match="does not advertise image/vision capability"):
            helper.analyze_image(image)

    request_json.assert_not_called()
