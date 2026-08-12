"""Temporal generator abstraction + dependency isolation (spec §16/§32)."""
from __future__ import annotations

import subprocess
import sys

import pytest

from atlas_camera.dynamic.generators import (
    NullGenerator,
    TemporalGenerationConfig,
    TemporalGenerationResult,
    resolve_generator,
)


def test_resolve_none_generator():
    gen = resolve_generator("none")
    assert isinstance(gen, NullGenerator)
    ok, reason = gen.available()
    assert ok is False and reason


def test_resolve_unknown_raises_with_choices():
    with pytest.raises(ValueError) as exc:
        resolve_generator("sora")
    assert "none" in str(exc.value) and "ltx" in str(exc.value)


def test_null_generator_returns_not_available(tmp_path):
    gen = NullGenerator()
    result = gen.generate(None, tmp_path, TemporalGenerationConfig())
    assert result.status == "not_available"
    assert result.frame_paths == []
    assert result.warnings
    assert list(tmp_path.iterdir()) == []  # never touches disk


def test_resolve_ltx_importable_without_deps():
    # constructing the adapter must never require torch/comfy — availability
    # is a runtime probe, not an import-time crash
    gen = resolve_generator("ltx")
    assert gen.name == "ltx"


def test_result_round_trip():
    r = TemporalGenerationResult(status="ok", frame_paths=["a.png"],
                                 width=8, height=4, fps=24.0, frame_count=1,
                                 generator="ltx", model="ltx-video",
                                 method="image_to_video", seed=5)
    again = TemporalGenerationResult.from_dict(r.to_dict())
    assert again.status == "ok"
    assert again.frame_paths == ["a.png"]
    assert again.seed == 5


def test_base_imports_stay_clean():
    """`import atlas_camera` + dynamic-plate schema must work in a bare
    interpreter with no generator deps (spec §32)."""
    code = (
        "import atlas_camera, atlas_camera.core.dynamic_plate, "
        "atlas_camera.dynamic.generators; print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
