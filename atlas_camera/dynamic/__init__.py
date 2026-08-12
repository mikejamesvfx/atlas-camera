"""Dynamic Plates: temporal generation for dynamic regions of a solved still.

Import-cheap by contract: this package must load with the core install alone.
Generator backends (ComfyUI/LTX) are capability-probed at runtime, never at
import time.
"""
from atlas_camera.dynamic.generators import (
    NullGenerator,
    TemporalGenerationConfig,
    TemporalGenerationResult,
    TemporalGenerator,
    resolve_generator,
)

__all__ = [
    "NullGenerator",
    "TemporalGenerationConfig",
    "TemporalGenerationResult",
    "TemporalGenerator",
    "resolve_generator",
]
