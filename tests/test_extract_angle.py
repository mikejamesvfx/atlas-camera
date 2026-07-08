"""Tests for the 📐 Extract Angle feature — the viewport button that measures
an artist's orbit delta from the recovered camera and snaps it to the Qwen
Multiple-Angles LoRA's named views (AtlasBlockoutViewport's four new STRING
outputs: patch_azimuth_view / patch_elevation_view / patch_distance /
patch_prompt).

The browser-side snap math lives in atlas_blockout.js (extractPatchAngle);
the Python tests here cover (a) the node's output plumbing and defaults,
(b) the payload's orbit_pivot key, and (c) the extraction math itself as an
exact inverse of camera_math.orbit_camera — ported line-for-line from the JS
so a future orbit_camera convention change fails THIS test instead of
silently misaligning extracted angles.
"""

import json
import math

import pytest

from atlas_camera.comfy.nodes import _ATLAS_BLOCKOUT_CACHE, AtlasBlockoutViewport
from atlas_camera.core.camera_math import (
    ground_lookat_pivot,
    look_at_view_matrix,
    orbit_camera,
)
from atlas_camera.core.schema import AtlasExtrinsics, AtlasIntrinsics, AtlasSolve, LatentCamera


def _solve(width=800, height=600, fx=700.0):
    eye = (1.5, 2.0, 3.0)
    target = (0.5, 0.0, -8.0)
    view, world, rot3 = look_at_view_matrix(eye, target)
    extr = AtlasExtrinsics(
        camera_position=eye, camera_rotation_matrix=rot3,
        camera_world_matrix=world, camera_view_matrix=view,
    )
    intr = AtlasIntrinsics(
        image_width=width, image_height=height, focal_length_mm=35.0,
        sensor_width_mm=36.0, fx_px=fx, fy_px=fx, cx_px=width / 2.0, cy_px=height / 2.0,
    )
    return AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))


_PA_SLOTS = slice(6, 10)  # (shaded, depth, normal, mask, path_frames, camera_path, *patch strings)


def test_viewport_gains_four_patch_string_outputs():
    assert AtlasBlockoutViewport.RETURN_TYPES[_PA_SLOTS] == ("STRING",) * 4
    assert AtlasBlockoutViewport.RETURN_NAMES[_PA_SLOTS] == (
        "patch_azimuth_view", "patch_elevation_view", "patch_distance", "patch_prompt")
    # Existing outputs keep their slot indices (saved workflows link by index).
    assert AtlasBlockoutViewport.RETURN_NAMES[:6] == (
        "shaded", "depth", "normal", "mask", "path_frames", "camera_path")


def test_render_defaults_to_zero_orbit_views():
    # Outside a ComfyUI runtime (no comfy_execution module importable, as in
    # this test env) the unextracted patch outputs fall back to the
    # zero-orbit named-view defaults instead of ExecutionBlocker.
    torch = pytest.importorskip("torch")
    _ATLAS_BLOCKOUT_CACHE.clear()
    out = AtlasBlockoutViewport().render(
        _solve(), torch.rand(1, 600, 800, 3), resolution=768, client_data="",
        unique_id="test_pa_default")
    az, el, dist, prompt = out["result"][_PA_SLOTS]
    assert (az, el, dist) == ("front view", "eye-level shot", "medium shot")
    assert prompt == "<sks> front view eye-level shot medium shot"


def test_render_blocks_patch_outputs_until_extracted(monkeypatch):
    """Inside a ComfyUI runtime, unextracted patch outputs return the
    ExecutionBlocker sentinel — downstream nodes (Qwen generation, patch,
    exports) pause silently until 📐 Extract Angle runs and re-queues."""
    import sys
    import types

    torch = pytest.importorskip("torch")

    class FakeBlocker:
        def __init__(self, message):
            self.message = message

    fake_graph = types.ModuleType("comfy_execution.graph")
    fake_graph.ExecutionBlocker = FakeBlocker
    fake_pkg = types.ModuleType("comfy_execution")
    fake_pkg.graph = fake_graph
    monkeypatch.setitem(sys.modules, "comfy_execution", fake_pkg)
    monkeypatch.setitem(sys.modules, "comfy_execution.graph", fake_graph)

    _ATLAS_BLOCKOUT_CACHE.clear()
    # No angle extracted -> all four patch outputs are blockers.
    out = AtlasBlockoutViewport().render(
        _solve(), torch.rand(1, 600, 800, 3), resolution=768, client_data="",
        unique_id="test_pa_block")
    for v in out["result"][_PA_SLOTS]:
        assert isinstance(v, FakeBlocker)
        assert v.message is None  # silent skip, not an error

    # Angle extracted (with a MATCHING fingerprint) -> strings flow, branch resumes.
    from atlas_camera.comfy.nodes import _solve_fingerprint
    solve = _solve()
    image = torch.rand(1, 600, 800, 3)
    client_data = json.dumps({"patch_angle": {
        "azimuth_view": "right side view", "elevation_view": "eye-level shot",
        "distance_view": "medium shot", "prompt": "<sks> right side view eye-level shot medium shot",
        "fingerprint": _solve_fingerprint(solve, image),
    }})
    out2 = AtlasBlockoutViewport().render(
        solve, image, resolution=768, client_data=client_data,
        unique_id="test_pa_unblock")
    assert out2["result"][_PA_SLOTS][0] == "right side view"
    assert not any(isinstance(v, FakeBlocker) for v in out2["result"][_PA_SLOTS])

    # Same extraction against a DIFFERENT image -> blockers again (stale).
    out3 = AtlasBlockoutViewport().render(
        solve, torch.rand(1, 600, 800, 3), resolution=768, client_data=client_data,
        unique_id="test_pa_stale_block")
    assert all(isinstance(v, FakeBlocker) for v in out3["result"][_PA_SLOTS])


def test_render_passes_through_extracted_patch_angle():
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import _solve_fingerprint
    _ATLAS_BLOCKOUT_CACHE.clear()
    solve = _solve()
    image = torch.rand(1, 600, 800, 3)
    client_data = json.dumps({"patch_angle": {
        "azimuth_view": "front-right quarter view",
        "elevation_view": "elevated shot",
        "distance_view": "wide shot",
        "prompt": "<sks> front-right quarter view elevated shot wide shot",
        "raw": {"d_azimuth_deg": 38.2, "d_elevation_deg": 24.0, "distance_scale": 1.6},
        "fingerprint": _solve_fingerprint(solve, image),
    }})
    out = AtlasBlockoutViewport().render(
        solve, image, resolution=768, client_data=client_data,
        unique_id="test_pa_pass")
    az, el, dist, prompt = out["result"][_PA_SLOTS]
    assert az == "front-right quarter view"
    assert el == "elevated shot"
    assert dist == "wide shot"
    assert prompt.startswith("<sks> front-right quarter view")


def test_stale_extraction_from_a_different_image_rearms_the_pause():
    """Swapping the input image must NOT run the previous image's extracted
    angle: a patch_angle whose fingerprint doesn't match the current
    solve+image is treated as not-extracted (found live — the persisted
    client_data widget kept the old extraction across an image swap)."""
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import _solve_fingerprint
    _ATLAS_BLOCKOUT_CACHE.clear()
    solve = _solve()
    old_image = torch.rand(1, 600, 800, 3)
    new_image = torch.rand(1, 600, 800, 3)
    stale = json.dumps({"patch_angle": {
        "azimuth_view": "right side view", "elevation_view": "eye-level shot",
        "distance_view": "medium shot", "prompt": "<sks> right side view eye-level shot medium shot",
        "fingerprint": _solve_fingerprint(solve, old_image),
    }})
    out = AtlasBlockoutViewport().render(
        solve, new_image, resolution=768, client_data=stale,
        unique_id="test_pa_stale")
    az, el, dist, prompt = out["result"][_PA_SLOTS]
    # Falls back to the not-extracted state (defaults here, since no comfy
    # ExecutionBlocker is importable in the test env — inside ComfyUI these
    # would be blockers, pausing the patch branch again).
    assert (az, el, dist) == ("front view", "eye-level shot", "medium shot")
    # Legacy extraction with NO fingerprint is stale too, by design.
    legacy = json.dumps({"patch_angle": {"azimuth_view": "right side view"}})
    out2 = AtlasBlockoutViewport().render(
        solve, new_image, resolution=768, client_data=legacy,
        unique_id="test_pa_legacy")
    assert out2["result"][_PA_SLOTS][0] == "front view"


def test_payload_carries_solve_fingerprint():
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import _solve_fingerprint
    _ATLAS_BLOCKOUT_CACHE.clear()
    solve = _solve()
    image = torch.rand(1, 600, 800, 3)
    AtlasBlockoutViewport().render(solve, image, resolution=768, client_data="",
                                   unique_id="test_pa_fp")
    payload = _ATLAS_BLOCKOUT_CACHE["test_pa_fp"]
    assert payload["solve_fingerprint"] == _solve_fingerprint(solve, image)


def test_payload_carries_the_backend_orbit_pivot():
    torch = pytest.importorskip("torch")
    _ATLAS_BLOCKOUT_CACHE.clear()
    solve = _solve()
    AtlasBlockoutViewport().render(
        solve, torch.rand(1, 600, 800, 3), resolution=768, client_data="",
        unique_id="test_pa_pivot")
    payload = _ATLAS_BLOCKOUT_CACHE["test_pa_pivot"]
    expected = ground_lookat_pivot(solve.camera.extrinsics)
    assert payload["orbit_pivot"] == pytest.approx(list(expected))


def _extract_delta_js_mirror(eye0, eye1, pivot):
    """Line-for-line port of atlas_blockout.js's extractPatchAngle math."""
    o0 = [eye0[i] - pivot[i] for i in range(3)]
    o1 = [eye1[i] - pivot[i] for i in range(3)]
    r0 = max(math.sqrt(sum(v * v for v in o0)), 1e-9)
    r1 = max(math.sqrt(sum(v * v for v in o1)), 1e-9)
    az0, az1 = math.atan2(o0[0], o0[2]), math.atan2(o1[0], o1[2])
    el0 = math.asin(max(-1.0, min(1.0, o0[1] / r0)))
    el1 = math.asin(max(-1.0, min(1.0, o1[1] / r1)))
    wrap = lambda d: ((d + 180.0) % 360.0 + 360.0) % 360.0 - 180.0
    return (wrap(math.degrees(az1 - az0)), math.degrees(el1 - el0), r1 / r0)


def test_parse_view_prompt_round_trips_every_named_view_combination():
    from atlas_camera.comfy.nodes import (
        _AZIMUTH_VIEWS,
        _DISTANCE_VIEWS,
        _ELEVATION_VIEWS,
        _parse_view_prompt,
    )
    for az in _AZIMUTH_VIEWS:
        for el in _ELEVATION_VIEWS:
            for dist in _DISTANCE_VIEWS:
                assert _parse_view_prompt(f"<sks> {az} {el} {dist}") == (az, el, dist)
    # Also tolerated without the <sks> token.
    assert _parse_view_prompt("back view low-angle shot close-up") == (
        "back view", "low-angle shot", "close-up")


def test_parse_view_prompt_rejects_garbage():
    from atlas_camera.comfy.nodes import _parse_view_prompt
    assert _parse_view_prompt("") is None
    assert _parse_view_prompt("<sks> sideways glance dutch tilt macro") is None
    assert _parse_view_prompt("<sks> front view eye-level shot") is None  # missing distance
    assert _parse_view_prompt("<sks> front view eye-level shot medium shot extra") is None


def test_patch_view_override_errors_loudly_on_unparseable_string():
    from atlas_camera.comfy.nodes import AtlasAddPatchView
    with pytest.raises(ValueError, match="does not parse"):
        AtlasAddPatchView().add_patch(
            _solve(), None, patch_view_override="not a real view prompt")


@pytest.mark.parametrize("d_az,d_el,ds", [
    (50.0, 20.0, 1.3), (-95.0, -25.0, 0.6), (170.0, 5.0, 1.8), (0.0, 0.0, 1.0),
])
def test_extraction_math_is_exact_inverse_of_orbit_camera(d_az, d_el, ds):
    solve = _solve()
    extr = solve.camera.extrinsics
    pivot = ground_lookat_pivot(extr)
    patch = orbit_camera(extr, pivot, d_azimuth_deg=d_az, d_elevation_deg=d_el,
                         distance_scale=ds)
    got_az, got_el, got_ds = _extract_delta_js_mirror(
        extr.camera_position, patch.camera_position, pivot)
    wrap = lambda d: ((d + 180.0) % 360.0) - 180.0
    assert abs(wrap(got_az - d_az)) < 1e-6
    assert abs(got_el - d_el) < 1e-6
    assert abs(got_ds - ds) < 1e-9
