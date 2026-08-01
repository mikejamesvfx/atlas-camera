"""Counting storeys on AtlasReferenceScaleSolve, instead of by hand.

The sky-rise doctrine's own workflow (commit 8e62eb0) wires this node with
``height_override_m=17.5`` and explains in a note that the number came from
"5 storeys x 3.5 m". The artist does the multiplication in their head and types
the product, so the count — the thing they actually verified by eye, and the
thing the VLM now reports as `storey_count` — is nowhere in the graph.

Re-verified live 2026-08-01 at full res: that preset recovers 63.40 m against
the 63.7 m recorded, so the mechanism is sound; only its input is awkward.

Widgets are APPENDED, never inserted — `widgets_values` is positional and
serialized into every saved workflow.
"""
import pytest

from atlas_camera.comfy.nodes_solve import AtlasReferenceScaleSolve

pytest.importorskip("numpy")

BBOX = (3820.0, 1775.0, 4470.0, 2480.0)   # the doctrine's counted tenement


def _widget_order():
    spec = AtlasReferenceScaleSolve.INPUT_TYPES()
    return list(spec.get("required", {})) + list(spec.get("optional", {}))


def test_the_new_widgets_are_appended_after_the_existing_ones():
    order = _widget_order()
    assert order[:7] == ["solve", "reference_id", "bbox_x0", "bbox_y0",
                         "bbox_x1", "bbox_y1", "height_override_m"]
    assert order[7:] == ["storey_count", "storey_height_m"]


def test_the_defaults_are_inert():
    """A storey_count of 0 must leave every existing workflow untouched."""
    opt = AtlasReferenceScaleSolve.INPUT_TYPES()["optional"]
    assert opt["storey_count"][1]["default"] == 0
    assert opt["storey_height_m"][1]["default"] == 3.0


def test_the_signature_defaults_match_the_declared_defaults():
    """Pinned repo-wide, but assert it here too: an API caller omitting the
    input must get the same value the UI shows."""
    import inspect

    sig = inspect.signature(
        getattr(AtlasReferenceScaleSolve, AtlasReferenceScaleSolve.FUNCTION))
    assert sig.parameters["storey_count"].default == 0
    assert sig.parameters["storey_height_m"].default == 3.0


def _height_used(monkeypatch, **kwargs):
    """Capture the real height the node hands to the geometry."""
    import atlas_camera.core.solver as solver

    seen = {}
    real = solver.resolve_reference_scale

    def spy(references, **kw):
        seen["height_m"] = references[0].get("height_m")
        seen["reference_id"] = references[0].get("reference_id")
        return real(references, **kw)

    monkeypatch.setattr(solver, "resolve_reference_scale", spy)
    solve = _solve()
    AtlasReferenceScaleSolve().apply(solve, "building_story_3m", *BBOX, **kwargs)
    return seen


def _solve():
    from atlas_camera.core.intrinsics import build_intrinsics
    from atlas_camera.core.schema import (
        AtlasCamera, AtlasExtrinsics, AtlasSolve)

    intr = build_intrinsics(image_width=7360, image_height=4912,
                            focal_length_mm=24.9, sensor_width_mm=36.0)
    view = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.93, -0.36, 0.0),
            (0.0, 0.36, 0.93, 0.0), (0.0, 0.0, 0.0, 1.0))
    return AtlasSolve(
        camera=AtlasCamera(intrinsics=intr,
                           extrinsics=AtlasExtrinsics(camera_view_matrix=view)),
        image_width=7360, image_height=4912)


def test_a_counted_storey_becomes_the_reference_height(monkeypatch):
    """5 storeys x 3.5 m is the doctrine's 17.5 m, without typing 17.5."""
    seen = _height_used(monkeypatch, storey_count=5, storey_height_m=3.5)
    assert seen["height_m"] == pytest.approx(17.5)


def test_the_storey_height_dial_changes_the_answer(monkeypatch):
    """3.0 m is the registry's number, 3.5 m the doctrine's measured prewar
    tenement. The node must not bury that choice."""
    seen = _height_used(monkeypatch, storey_count=5, storey_height_m=3.0)
    assert seen["height_m"] == pytest.approx(15.0)


def test_a_count_beats_a_stale_hand_typed_override(monkeypatch):
    """If both are set the COUNT wins — it is the thing the artist verified,
    and leaving a stale product to override it silently is the bug this
    widget exists to remove."""
    seen = _height_used(monkeypatch, height_override_m=99.0,
                        storey_count=5, storey_height_m=3.5)
    assert seen["height_m"] == pytest.approx(17.5)


def test_no_count_still_honours_the_hand_typed_override(monkeypatch):
    seen = _height_used(monkeypatch, height_override_m=17.5)
    assert seen["height_m"] == pytest.approx(17.5)


def test_no_count_and_no_override_falls_back_to_the_registry_entry(monkeypatch):
    """`building_story_3m` is one storey; the node must pass the id through
    rather than inventing a height."""
    seen = _height_used(monkeypatch)
    assert seen["reference_id"] == "building_story_3m"


@pytest.mark.parametrize("bad", [0, -2])
def test_a_nonsense_count_is_ignored_not_multiplied(monkeypatch, bad):
    seen = _height_used(monkeypatch, height_override_m=17.5, storey_count=bad)
    assert seen["height_m"] == pytest.approx(17.5)
