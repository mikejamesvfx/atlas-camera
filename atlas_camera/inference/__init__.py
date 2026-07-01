"""Optional inference helpers around the Atlas core."""

from atlas_camera.inference.multimodal_helper import (
    LMStudioVisionProvider,
    LlamaCppVisionProvider,
    MultimodalSceneHelper,
    MultimodalSceneObservation,
    MultimodalProvider,
    OllamaVisionSceneHelper,
    OllamaVisionProvider,
    ProviderModelInfo,
    SceneScaleCue,
    create_multimodal_provider,
    provider_models_response,
)

__all__ = [
    "LMStudioVisionProvider",
    "LlamaCppVisionProvider",
    "MultimodalSceneHelper",
    "MultimodalSceneObservation",
    "MultimodalProvider",
    "OllamaVisionSceneHelper",
    "OllamaVisionProvider",
    "ProviderModelInfo",
    "SceneScaleCue",
    "create_multimodal_provider",
    "provider_models_response",
]
