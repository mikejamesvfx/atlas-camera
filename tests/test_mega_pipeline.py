"""AtlasMegaPipeline (experimental monolith) — call-site signature guard.

The monolith bypasses ComfyUI graph execution and calls downstream node
methods directly in Python. Nothing about ComfyUI validates those in-Python
calls, so a kwarg that drifts from a downstream node's real signature is
invisible until the node is actually queued — which is exactly how it shipped
calling `AtlasDeriveProjectionGeometry.derive(depth=...)` when that method
estimates its own depth and has no `depth` parameter at all
(`TypeError: got an unexpected keyword argument 'depth'`, found live on the
portable install).

This test executes `execute_pipeline` with the three downstream methods
replaced by fakes that BIND their received kwargs against the REAL method
signatures before returning a stub. So it runs torch-free (no GeoCalib, no
depth model, no Maya) yet still fails the instant any call site passes a
kwarg the real node would reject.
"""

import inspect

import pytest

import atlas_camera.comfy.nodes_experimental as mp
from atlas_camera.comfy.nodes_solve import AtlasLearnedSolveFromImage
from atlas_camera.comfy.nodes_geometry import AtlasDeriveProjectionGeometry
from atlas_camera.comfy.nodes_export import AtlasExportMayaReviewScene


def _binding_fake(real_method, return_value, recorder):
    """A stand-in for `real_method` that validates the call the pipeline makes
    against `real_method`'s true signature, records it, then returns a stub —
    so no heavy dependency runs but a bad kwarg still raises TypeError."""
    sig = inspect.signature(real_method)

    def fake(self, *args, **kwargs):
        sig.bind(self, *args, **kwargs)      # TypeError on any unknown/missing kwarg
        recorder.append(kwargs)
        return return_value

    return fake


def test_execute_pipeline_calls_every_downstream_node_with_valid_kwargs(monkeypatch):
    solve_calls, derive_calls, export_calls = [], [], []

    monkeypatch.setattr(
        AtlasLearnedSolveFromImage, "solve",
        _binding_fake(AtlasLearnedSolveFromImage.solve, ("SOLVE",), solve_calls))
    monkeypatch.setattr(
        AtlasDeriveProjectionGeometry, "derive",
        _binding_fake(AtlasDeriveProjectionGeometry.derive, ("GEOM_SOLVE",), derive_calls))
    monkeypatch.setattr(
        AtlasExportMayaReviewScene, "export",
        _binding_fake(AtlasExportMayaReviewScene.export, ("C:/out/scene.ma",), export_calls))

    out = mp.AtlasMegaPipeline().execute_pipeline(
        image="IMAGE", output_dir="atlas_exports",
        camera_height_m=1.6, scene_type="outdoor")

    assert out == ("C:/out/scene.ma",)
    assert len(solve_calls) == len(derive_calls) == len(export_calls) == 1

    # The derive step must NOT receive a `depth` kwarg — the god-node estimates
    # depth internally. This is the exact regression that shipped.
    assert "depth" not in derive_calls[0]
    assert derive_calls[0]["depth_model"], "derive should be told which depth model to use"

    # The solve's output must flow into derive, and derive's into export.
    assert derive_calls[0]["solve"] == "SOLVE"
    assert export_calls[0]["solve"] == "GEOM_SOLVE"
    assert export_calls[0]["output_dir"] == "atlas_exports"


def test_derive_really_has_no_depth_parameter():
    # Belt-and-braces: pin the fact the shipped bug violated, so a future
    # signature change to derive() that reintroduces `depth` is a conscious
    # decision, not a silent reopening.
    params = inspect.signature(AtlasDeriveProjectionGeometry.derive).parameters
    assert "depth" not in params
    assert "image" in params and "depth_model" in params
