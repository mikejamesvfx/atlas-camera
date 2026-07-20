"""Native SAM3 (AtlasSAM3Mask) — version probe, gated-repo wrapper, and
concept-union logic.

The real SAM3 inference path needs downloaded weights + a gated HF repo, so
it is exercised live (not here) — same split as test_semantic_segmenter.py.
Everything pure/mockable is pinned below: the transformers version compare,
the capability probe, the gated-repo error translation, and the
comma-separated-concept union logic (via mocking the actual per-concept
detector call).
"""

import importlib.util
import sys
import types

import numpy as np
import pytest

# Not used directly by these tests, but kept module-level because Task 3
# appends tests to this same file that need a real torch module.
torch = pytest.importorskip("torch")

import atlas_camera.inference.sam3_segmenter as sam3_mod
from atlas_camera.inference.sam3_segmenter import (
    Sam3GatedRepoError,
    _meets_min_version,
    _wrap_if_gated_repo,
    native_sam3_available,
)


def test_meets_min_version_exact_and_above():
    assert _meets_min_version("5.5.4")
    assert _meets_min_version("5.5.5")
    assert _meets_min_version("5.6.0")
    assert _meets_min_version("6.0.0")


def test_meets_min_version_below():
    assert not _meets_min_version("5.5.3")
    assert not _meets_min_version("5.4.9")
    assert not _meets_min_version("4.40.0")


def test_meets_min_version_dev_suffix():
    # '5.5.4.dev0' -> numeric prefix (5, 5, 4) still compares correctly
    assert _meets_min_version("5.5.4.dev0")
    assert not _meets_min_version("5.5.3.dev999")


def test_native_sam3_available_true_with_new_transformers(monkeypatch):
    fake = types.SimpleNamespace(__version__="5.5.4")
    monkeypatch.setitem(sys.modules, "transformers", fake)
    assert native_sam3_available() is True


def test_native_sam3_available_false_with_old_transformers(monkeypatch):
    fake = types.SimpleNamespace(__version__="4.40.0")
    monkeypatch.setitem(sys.modules, "transformers", fake)
    assert native_sam3_available() is False


def test_native_sam3_available_false_when_transformers_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert native_sam3_available() is False


def test_native_sam3_available_false_when_torch_missing(monkeypatch):
    # transformers is present at a sufficient version, but torch is not
    # importable -- native_sam3_available() must still report False (its
    # docstring/_require_sam3's error message both promise a torch check).
    fake = types.SimpleNamespace(__version__="5.5.4")
    monkeypatch.setitem(sys.modules, "transformers", fake)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **kw: None if name == "torch" else real_find_spec(name, *a, **kw),
    )
    assert native_sam3_available() is False


def test_require_sam3_raises_actionable_error_when_unavailable(monkeypatch):
    monkeypatch.setattr(sam3_mod, "native_sam3_available", lambda: False)
    with pytest.raises(RuntimeError, match=r"\[sam3\]"):
        sam3_mod._require_sam3()


def test_wrap_if_gated_repo_detects_gated_shape():
    exc = OSError("You are trying to access a gated repo. 401 Client Error.")
    wrapped = _wrap_if_gated_repo("facebook/sam3", exc)
    assert isinstance(wrapped, Sam3GatedRepoError)
    assert "hf auth login" in str(wrapped)
    assert "facebook/sam3" in str(wrapped)


def test_wrap_if_gated_repo_passes_through_unrelated_errors():
    exc = ValueError("some unrelated failure")
    assert _wrap_if_gated_repo("facebook/sam3", exc) is None
