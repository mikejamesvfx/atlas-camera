"""Contract for attaching a converted plate to the RIGHT camera.

A two-photo rig has an anchor plate on `solve.source_plate` and every other
photograph's plate on its own `ProjectionSource`. `AtlasAttachSourcePlate` could
only reach the first, so an ACEScg delivery converted both files and then shipped
a solve whose second camera still pointed at the Linear Rec.709 original — Nuke
renders the anchor so the picture looked right, and the wrong plate travelled
anyway. Found live on the sh004 two-RAW delivery, 2026-08-10.

The companion half is in test_export_plate_exr_is_terminal: converting the file
at all requires the export node to RUN, and with its outputs unconsumed it did
not.
"""
from __future__ import annotations

import pytest

from atlas_camera.comfy.nodes_solve import AtlasAttachSourcePlate
from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasPlateRef,
    AtlasSolve,
    LatentCamera,
    ProjectionSource,
)

np = pytest.importorskip("numpy")


def _camera():
    return LatentCamera(
        intrinsics=AtlasIntrinsics(image_width=64, image_height=48, fx_px=50.0,
                                   fy_px=50.0, cx_px=32.0, cy_px=24.0),
        extrinsics=AtlasExtrinsics(camera_position=(0.0, 0.0, 0.0),
                                   camera_view_matrix=np.eye(4),
                                   camera_world_matrix=np.eye(4)),
    )


def _plate(path, colorspace, preview=""):
    return AtlasPlateRef(image_path=path, colorspace=colorspace,
                         preview_b64=preview, role="source", is_proxy=False)


@pytest.fixture
def rig():
    return AtlasSolve(
        camera=_camera(),
        source_plate=_plate("anchor_linear.exr", "Linear Rec.709 (sRGB)"),
        projection_sources=[
            ProjectionSource(camera=_camera(), name="photo_2",
                             image_b64="data:image/jpeg;base64,OLD",
                             plate_ref=_plate("photo2_linear.exr",
                                              "Linear Rec.709 (sRGB)"),
                             metadata={"evidence_type": "photographed"}),
        ],
    )


def test_blank_source_name_still_attaches_the_primary_plate(rig):
    """The original behaviour is the default — every saved workflow relies on it."""
    acescg = _plate("anchor_acescg.exr", "ACEScg")
    (out,) = AtlasAttachSourcePlate().attach(rig, acescg)
    assert out.source_plate.image_path == "anchor_acescg.exr"
    assert out.source_plate.colorspace == "ACEScg"
    # ...and the non-anchor source is untouched, which is the whole reason the
    # named form had to be added rather than the primary form being widened.
    assert out.projection_sources[0].plate_ref.image_path == "photo2_linear.exr"


def test_a_named_source_gets_the_converted_plate(rig):
    acescg = _plate("photo2_acescg.exr", "ACEScg", preview="data:image/jpeg;base64,NEW")
    (out,) = AtlasAttachSourcePlate().attach(rig, acescg, source_name="photo_2")
    source = out.projection_sources[0]
    assert source.plate_ref.image_path == "photo2_acescg.exr"
    assert source.plate_ref.colorspace == "ACEScg"
    # The primary must NOT be hijacked by a named attach.
    assert out.source_plate.image_path == "anchor_linear.exr"


def test_the_stale_browser_preview_is_replaced(rig):
    """image_b64 is built from the OLD plate; leaving it would show
    pre-conversion pixels in the viewport while the exporters reference the new
    file — the two disagreeing is exactly the confusion this node prevents."""
    acescg = _plate("photo2_acescg.exr", "ACEScg", preview="data:image/jpeg;base64,NEW")
    (out,) = AtlasAttachSourcePlate().attach(rig, acescg, source_name="photo_2")
    assert out.projection_sources[0].image_b64 == "data:image/jpeg;base64,NEW"


def test_a_plate_with_no_preview_keeps_the_existing_one(rig):
    acescg = _plate("photo2_acescg.exr", "ACEScg", preview="")
    (out,) = AtlasAttachSourcePlate().attach(rig, acescg, source_name="photo_2")
    assert out.projection_sources[0].image_b64 == "data:image/jpeg;base64,OLD"


def test_the_input_solve_is_never_mutated(rig):
    acescg = _plate("photo2_acescg.exr", "ACEScg")
    AtlasAttachSourcePlate().attach(rig, acescg, source_name="photo_2")
    assert rig.projection_sources[0].plate_ref.image_path == "photo2_linear.exr"


def test_an_unknown_source_name_names_what_it_knows(rig):
    """Silently doing nothing would ship the unconverted plate and report
    success — the failure mode this whole fix exists to remove."""
    acescg = _plate("x.exr", "ACEScg")
    with pytest.raises(RuntimeError, match="photo_2"):
        AtlasAttachSourcePlate().attach(rig, acescg, source_name="photo_9")


def test_source_name_is_appended_last(rig):
    """Positional widgets_values: a new widget may only append."""
    from atlas_camera.mcp.comfy_http import is_widget

    spec = AtlasAttachSourcePlate.INPUT_TYPES()
    widgets = [n for sec in ("required", "optional")
               for n, s in (spec.get(sec) or {}).items() if is_widget(s)]
    assert widgets == ["source_name"]


def test_export_plate_exr_is_terminal():
    """A node whose entire purpose is writing a file must execute when queued.

    AtlasExportPlateEXR was not an OUTPUT_NODE, so ComfyUI ran it only when
    something consumed an output — and a graph that converts a plate to ACEScg
    purely for delivery has no consumer. The conversion was skipped in silence
    and the workflow reported success with Linear Rec.709 plates on disk.
    """
    from atlas_camera.comfy.nodes_export import (
        AtlasExportNuke, AtlasExportPlateEXR, AtlasExportReliefMesh,
    )

    assert AtlasExportPlateEXR.OUTPUT_NODE is True
    # Consistent with every sibling exporter, which is why the omission read as
    # an oversight rather than a design choice.
    assert AtlasExportNuke.OUTPUT_NODE is True
    assert AtlasExportReliefMesh.OUTPUT_NODE is True
