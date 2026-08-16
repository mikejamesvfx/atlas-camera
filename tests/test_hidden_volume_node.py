"""AtlasLoadHiddenVolume: the properties that were learned the hard way.

Every test here pins a defect found live on 2026-08-15 rather than a
hypothetical. The node is experimental, but these are the invariants that make
its output safe to put in front of an artist.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from atlas_camera.comfy.nodes_hidden_volume import AtlasLoadHiddenVolume
from atlas_camera.core.schema import (
    AtlasCamera,
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasProjectionScene,
    AtlasSolve,
)

pytest.importorskip("skimage", reason="marching cubes needs scikit-image")


def _solve():
    intr = AtlasIntrinsics(image_width=64, image_height=64, fx_px=64.0, fy_px=64.0,
                           cx_px=32.0, cy_px=32.0)
    return AtlasSolve(
        camera=AtlasCamera(intrinsics=intr, extrinsics=AtlasExtrinsics()),
        projection_scene=AtlasProjectionScene(),
    )


def _write_volume(tmp_path, *, invented: bool, res: int = 32, trunc: float = 3.0):
    """A slab volume. ``invented`` controls whether the VISIBLE field supports it.

    The predicted surface is the same either way; only its visible support
    changes, which is exactly the axis the divergence gate measures.
    """
    tudf = np.full((res, res, res), trunc, dtype=np.float32)
    tudf[res // 2 - 1: res // 2 + 1, :, :] = 0.0          # one slab, always
    visible = np.full((res, res, res), trunc, dtype=np.float32)
    if not invented:
        visible[res // 2 - 2: res // 2 + 2, :, :] = 0.0   # visible supports it
    # moge_mask all-valid by default: no sky to reject.
    np.savez_compressed(tmp_path / f"pred_tudf_{res}.npz",
                        tudf=tudf, visible_tudf=visible,
                        moge_mask=np.ones((res, res), dtype=bool))
    (tmp_path / "metadata.json").write_text(json.dumps({
        "representation": "tudf", "truncation_voxels": trunc,
        "field_range": [0.0, trunc], "field_units": "voxel_units",
        "bbox_min": [-1.0, -1.0, 2.0], "extent_xyz": [4.0, 4.0, 4.0],
        "pred_resolution": [res, res, res],
    }), encoding="utf-8")
    return str(tmp_path)


def _prim(solve):
    return solve.projection_scene.proxy_geometry[-1]


def _mesh(prim):
    v = np.asarray(prim.metadata["vertices"], dtype=np.float64).reshape(-1, 3)
    f = np.asarray(prim.metadata["faces"], dtype=np.int64).reshape(-1, 3)
    return v, f


# --- the decimation defect: striding a face list PERFORATES the mesh ---------

def test_decimation_leaves_no_orphan_vertices(tmp_path):
    """Shipped bug: `faces = faces[::step]` kept every 3rd triangle and deleted
    the rest, which the viewport showed as a shredded lattice. Decimation now
    sub-samples the FIELD, so every emitted vertex must still be referenced."""
    d = _write_volume(tmp_path, invented=False)
    out, report, _ = AtlasLoadHiddenVolume().load(_solve(), d, max_faces=200,
                                          double_sided=False,
                                          extraction="marching_cubes")
    v, f = _mesh(_prim(out))
    assert len(f) > 0
    assert len(np.unique(f)) == len(v), "orphan vertices — mesh was perforated"
    assert f.max() == len(v) - 1


def test_face_budget_shrinks_the_mesh_via_grid_stride(tmp_path):
    d = _write_volume(tmp_path, invented=False, res=64)
    node = AtlasLoadHiddenVolume()
    big, report, _ = node.load(_solve(), d, max_faces=0, double_sided=False,
                       extraction="marching_cubes")
    small, report, _ = node.load(_solve(), d, max_faces=300, double_sided=False,
                         extraction="marching_cubes")
    _, fb = _mesh(_prim(big))
    _, fs = _mesh(_prim(small))
    assert len(fs) < len(fb)
    assert _prim(small).metadata["hidden_volume"]["grid_stride"] > 1
    # A coarser grid means a physically larger voxel — the cost of the budget.
    assert (_prim(small).metadata["hidden_volume"]["voxel_edge_m"] >
            _prim(big).metadata["hidden_volume"]["voxel_edge_m"])


# --- double-siding is opt-out, and it is exactly a doubling ------------------

def test_double_sided_emits_both_windings(tmp_path):
    d = _write_volume(tmp_path, invented=False)
    node = AtlasLoadHiddenVolume()
    one, report, _ = node.load(_solve(), d, max_faces=0, double_sided=False,
                       extraction="marching_cubes")
    two, report, _ = node.load(_solve(), d, max_faces=0, double_sided=True,
                       extraction="marching_cubes")
    _, f1 = _mesh(_prim(one))
    _, f2 = _mesh(_prim(two))
    assert len(f2) == 2 * len(f1)
    np.testing.assert_array_equal(f2[len(f1):], f1[:, ::-1])


# --- the divergence gate must actually GATE ---------------------------------

def test_gate_refuses_a_diverged_volume(tmp_path):
    """A warning string is not a gate: the previous version emitted the geometry
    anyway, so a diverged volume reached the viewport looking sound."""
    d = _write_volume(tmp_path, invented=True)
    out, report, _ = AtlasLoadHiddenVolume().load(
        _solve(), d, max_faces=0, double_sided=False,
        max_invented_fraction=0.85, on_divergence="refuse", emit="layers")
    assert len(out.projection_scene.proxy_geometry) == 0, "refused but still emitted"
    assert "REFUSED" in report
    assert "%" in report


def test_gate_marks_when_asked_instead_of_refusing(tmp_path):
    d = _write_volume(tmp_path, invented=True)
    out, report, _ = AtlasLoadHiddenVolume().load(
        _solve(), d, max_faces=0, double_sided=False,
        max_invented_fraction=0.85, on_divergence="mark", emit="layers")
    assert len(out.projection_scene.proxy_geometry) == 1
    hv = _prim(out).metadata["hidden_volume"]
    assert hv["diverged"] is True
    assert hv["invented_fraction"] > 0.85
    assert "DIVERGED" in report


def test_supported_volume_passes_the_gate(tmp_path):
    d = _write_volume(tmp_path, invented=False)
    out, report, _ = AtlasLoadHiddenVolume().load(
        _solve(), d, max_faces=0, double_sided=False, on_divergence="refuse", emit="layers")
    assert len(out.projection_scene.proxy_geometry) == 1
    assert _prim(out).metadata["hidden_volume"]["diverged"] is False


def test_ambiguous_band_is_emitted_and_tagged_for_inspection(tmp_path):
    """Three states, not two. Validated on 26 held-out volumes: a single 85%
    threshold agreed with visible-surface quality 88.5% of the time, while
    0.82 inspect / 0.88 refuse called every decisive case correctly (20/20) —
    because every misclassification sat between those two numbers. A volume in
    that band is neither passed silently nor dropped silently."""
    d = _write_volume(tmp_path, invented=True)
    out, report, _ = AtlasLoadHiddenVolume().load(
        _solve(), d, max_faces=0, double_sided=False,
        inspect_invented_fraction=0.5, max_invented_fraction=1.0,
        on_divergence="refuse", emit="layers")
    assert len(out.projection_scene.proxy_geometry) == 1, "ambiguous must emit"
    hv = _prim(out).metadata["hidden_volume"]
    assert hv["diverged"] is False
    assert hv["needs_inspection"] is True
    assert "AMBIGUOUS" in report


def test_gate_can_be_disabled(tmp_path):
    d = _write_volume(tmp_path, invented=True)
    out, report, _ = AtlasLoadHiddenVolume().load(
        _solve(), d, max_faces=0, double_sided=False,
        max_invented_fraction=1.0, on_divergence="refuse", emit="layers")
    assert len(out.projection_scene.proxy_geometry) == 1


# --- contracts the rest of Atlas relies on ----------------------------------

def test_appends_and_never_clobbers(tmp_path):
    """Not a Derive node: prior PROXY_ROLE geometry must survive."""
    from atlas_camera.core.schema import AtlasProxyPrimitive
    solve = _solve()
    solve.projection_scene.proxy_geometry.append(
        AtlasProxyPrimitive(name="pre_existing", primitive_type="box"))
    d = _write_volume(tmp_path, invented=False)
    out, report, _ = AtlasLoadHiddenVolume().load(solve, d, max_faces=0, double_sided=False, emit="layers")
    names = [g.name for g in out.projection_scene.proxy_geometry]
    assert "pre_existing" in names
    assert len(names) == 2
    # And the input solve is untouched.
    assert len(solve.projection_scene.proxy_geometry) == 1


def test_retopo_can_see_this_primitive(tmp_path):
    """The retopo allowlist keys on metadata['source']; if this drifts, the
    full-res -> decimate chain silently does nothing."""
    from atlas_camera.comfy.nodes_hidden_volume import HIDDEN_VOLUME_SOURCE
    d = _write_volume(tmp_path, invented=False)
    out, report, _ = AtlasLoadHiddenVolume().load(_solve(), d, max_faces=0,
                                          double_sided=False, emit="layers")
    assert _prim(out).metadata["source"] == HIDDEN_VOLUME_SOURCE
    assert _prim(out).primitive_type == "mesh"


def test_missing_volume_reports_instead_of_raising(tmp_path):
    out, report, _ = AtlasLoadHiddenVolume().load(_solve(), str(tmp_path / "nope"))
    assert len(out.projection_scene.proxy_geometry) == 0
    assert "not a directory" in report.lower()


def test_volume_with_no_surface_reports_instead_of_crashing(tmp_path):
    """An all-truncation field has no level set to mesh — the real case being
    the `golden_corridor` plate, where the depth stage collapsed and VolFill
    produced zero surface voxels. Marching cubes would raise; the node must
    explain instead."""
    res, trunc = 16, 3.0
    np.savez_compressed(
        tmp_path / f"pred_tudf_{res}.npz",
        tudf=np.full((res, res, res), trunc, dtype=np.float32),
        visible_tudf=np.full((res, res, res), trunc, dtype=np.float32))
    (tmp_path / "metadata.json").write_text(json.dumps({
        "truncation_voxels": trunc, "bbox_min": [0.0, 0.0, 1.0],
        "extent_xyz": [2.0, 2.0, 2.0], "pred_resolution": [res, res, res],
    }), encoding="utf-8")

    out, report, _ = AtlasLoadHiddenVolume().load(_solve(), str(tmp_path))

    assert len(out.projection_scene.proxy_geometry) == 0
    # The empty-volume gate now fires first and names the condition directly.
    assert "empty" in report.lower()


# --- the empty-volume hole in the divergence gate ----------------------------

def _empty_volume(tmp_path, res=24, trunc=3.0):
    """A field with NO surface — what the golden_corridor plate produced when
    the depth stage collapsed."""
    np.savez_compressed(tmp_path / f"pred_tudf_{res}.npz",
                        tudf=np.full((res,) * 3, trunc, dtype=np.float32),
                        visible_tudf=np.full((res,) * 3, trunc, dtype=np.float32))
    (tmp_path / "metadata.json").write_text(json.dumps({
        "truncation_voxels": trunc, "bbox_min": [-1.0, -1.0, 2.0],
        "extent_xyz": [4.0, 4.0, 4.0], "pred_resolution": [res, res, res],
    }), encoding="utf-8")
    return str(tmp_path)


@pytest.mark.parametrize("extraction", ["raymarch", "marching_cubes"])
def test_empty_volume_is_refused_not_passed(tmp_path, extraction):
    """A volume with no content scores 0% INVENTED, so the divergence gate read
    it as SOUND — measured on golden_corridor, which produced zero surface
    voxels and sailed through. Nothing is not agreement."""
    d = _empty_volume(tmp_path)
    out, report, _ = AtlasLoadHiddenVolume().load(
        _solve(), d, extraction=extraction, on_divergence="mark")
    assert len(out.projection_scene.proxy_geometry) == 0
    assert "EMPTY" in report


def test_empty_gate_can_be_disabled(tmp_path):
    """Still refuses, because there is genuinely nothing to mesh — the gate
    being off must not turn an empty field into geometry."""
    d = _empty_volume(tmp_path)
    out, report, _ = AtlasLoadHiddenVolume().load(
        _solve(), d, extraction="marching_cubes", min_surface_coverage=0.0)
    assert len(out.projection_scene.proxy_geometry) == 0


def test_healthy_volume_clears_the_coverage_floor(tmp_path):
    d = _write_volume(tmp_path, invented=False, res=48)
    out, report, _ = AtlasLoadHiddenVolume().load(
        _solve(), d, extraction="raymarch", min_surface_coverage=0.02,
        on_divergence="mark", emit="layers")
    assert out.projection_scene.proxy_geometry
    hv = out.projection_scene.proxy_geometry[-1].metadata["hidden_volume"]
    assert hv["ray_coverage"] > 0.02


# --- sky: geometry invented where the depth stage saw nothing ----------------

def test_sky_geometry_is_rejected(tmp_path):
    """The predictor fills its whole cube, including the region MoGe masked out
    as sky — where there is no surface at all. Seen live in the viewport as
    shards floating above a roofline. Measured on sh001: 18.9% of the invented
    surface, and rejecting it dropped the zero-offset in-front fraction from
    3.25% to 1.06%."""
    res = 48
    trunc = 3.0
    tudf = np.full((res,) * 3, trunc, dtype=np.float32)
    tudf[res // 2 - 1: res // 2 + 1, :, :] = 0.0
    visible = np.full((res,) * 3, trunc, dtype=np.float32)
    visible[res // 2 - 2: res // 2 + 2, :, :] = 0.0
    # Top half of the frame is sky.
    mask = np.ones((res, res), dtype=bool)
    mask[: res // 2, :] = False
    np.savez_compressed(tmp_path / f"pred_tudf_{res}.npz", tudf=tudf,
                        visible_tudf=visible, moge_mask=mask)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "truncation_voxels": trunc, "bbox_min": [-1.0, -1.0, 2.0],
        "extent_xyz": [4.0, 4.0, 4.0], "pred_resolution": [res, res, res],
    }), encoding="utf-8")

    node = AtlasLoadHiddenVolume()
    off, report, _ = node.load(_solve(), str(tmp_path), extraction="raymarch",
                       reject_sky=False, on_divergence="mark", emit="layers")
    on, report, _ = node.load(_solve(), str(tmp_path), extraction="raymarch",
                           reject_sky=True, on_divergence="mark", emit="layers")

    f_off = sum(len(_mesh(p)[1]) for p in off.projection_scene.proxy_geometry)
    f_on = sum(len(_mesh(p)[1]) for p in on.projection_scene.proxy_geometry)
    assert f_on < f_off, "sky rejection removed nothing"
    assert "sky-rejected" in report
    assert on.projection_scene.proxy_geometry[-1].metadata[
        "hidden_volume"]["sky_rejected_samples"] > 0


def test_sky_rejection_survives_a_volume_with_no_mask(tmp_path):
    """Older volumes predate the mask; the node must not fail on them."""
    res, trunc = 32, 3.0
    tudf = np.full((res,) * 3, trunc, dtype=np.float32)
    tudf[res // 2 - 1: res // 2 + 1, :, :] = 0.0
    np.savez_compressed(tmp_path / f"pred_tudf_{res}.npz", tudf=tudf,
                        visible_tudf=tudf.copy())
    (tmp_path / "metadata.json").write_text(json.dumps({
        "truncation_voxels": trunc, "bbox_min": [-1.0, -1.0, 2.0],
        "extent_xyz": [4.0, 4.0, 4.0], "pred_resolution": [res, res, res],
    }), encoding="utf-8")
    out, report, _ = AtlasLoadHiddenVolume().load(
        _solve(), str(tmp_path), extraction="raymarch", reject_sky=True,
        on_divergence="mark", emit="layers")
    assert out.projection_scene.proxy_geometry


# --- combined: layered rays -> select_hidden_surface -> ONE surface ----------

def _two_slab_volume(tmp_path, res=48, trunc=3.0):
    """A near slab and a far slab: an occluder with something behind it, which
    is the only situation hidden-geometry selection has anything to do."""
    tudf = np.full((res,) * 3, trunc, dtype=np.float32)
    for iz in (res // 4, (3 * res) // 4):
        tudf[iz - 1: iz + 1, :, :] = 0.0
    visible = np.full((res,) * 3, trunc, dtype=np.float32)
    visible[res // 4 - 1: res // 4 + 1, :, :] = 0.0      # only the NEAR slab
    np.savez_compressed(tmp_path / f"pred_tudf_{res}.npz", tudf=tudf,
                        visible_tudf=visible,
                        moge_mask=np.ones((res, res), dtype=bool))
    (tmp_path / "metadata.json").write_text(json.dumps({
        "truncation_voxels": trunc, "bbox_min": [-2.0, -2.0, 2.0],
        "extent_xyz": [8.0, 8.0, 8.0], "pred_resolution": [res, res, res],
    }), encoding="utf-8")
    return str(tmp_path)


def test_combined_is_the_default_and_emits_one_surface(tmp_path):
    assert AtlasLoadHiddenVolume.INPUT_TYPES()["optional"]["emit"][1]["default"] \
        == "combined"
    d = _two_slab_volume(tmp_path)
    out, report, matte = AtlasLoadHiddenVolume().load(
        _solve(), d, emit="combined", on_divergence="mark")
    prims = out.projection_scene.proxy_geometry
    assert len(prims) == 1, "combined must merge into ONE surface, not per-layer"
    assert prims[0].name.endswith("_combined")
    assert "COMBINED" in report
    v, f = _mesh(prims[0])
    assert len(f) > 0 and f.max() < len(v)


def test_combined_returns_an_occlusion_matte(tmp_path):
    """Geometry and paint are separate concerns: the surface stays continuous so
    it meshes, and the matte says where painting is legitimate."""
    d = _two_slab_volume(tmp_path)
    out, _, matte = AtlasLoadHiddenVolume().load(
        _solve(), d, emit="combined", on_divergence="mark")
    assert matte.shape[0] == 1 and matte.ndim == 3
    arr = np.asarray(matte[0] if not hasattr(matte, "numpy") else matte.numpy()[0])
    assert arr.max() <= 1.0 and arr.min() >= 0.0
    assert arr.any(), "a slab behind an occluder must produce SOME matte"


def test_fill_is_skipped_without_a_restrict_mask(tmp_path):
    """Diffusing across a whole frame turns a small real selection into total
    replacement — measured on sh001, 0.9% became 100%."""
    d = _two_slab_volume(tmp_path)
    out, report, _ = AtlasLoadHiddenVolume().load(
        _solve(), d, emit="combined", fill_gaps=True, on_divergence="mark")
    assert "fill SKIPPED" in report
    hv = out.projection_scene.proxy_geometry[-1].metadata["hidden_volume"]
    assert hv["restricted"] is False
    assert hv["substituted_fraction"] == pytest.approx(
        hv["raw_selection_fraction"], abs=1e-6)


def test_restrict_mask_bounds_the_diffusion(tmp_path):
    d = _two_slab_volume(tmp_path)
    node = AtlasLoadHiddenVolume()
    free, _, _ = node.load(_solve(), d, emit="combined", fill_gaps=True,
                           on_divergence="mark")
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[32:, :] = 1.0
    bound, report, _ = node.load(_solve(), d, emit="combined", fill_gaps=True,
                                 restrict_mask=mask, on_divergence="mark")
    hf = free.projection_scene.proxy_geometry[-1].metadata["hidden_volume"]
    hb = bound.projection_scene.proxy_geometry[-1].metadata["hidden_volume"]
    assert "diffused inside restrict_mask" in report
    assert hb["restricted"] is True and hf["restricted"] is False
    # Containment is the property that matters: the mask covers the lower half,
    # so substitution must not exceed it (plus a little for the mask's own edge).
    assert hb["substituted_fraction"] <= 0.55, "diffusion escaped the mask"


def test_gate_reads_the_raw_selection_not_the_diffusion(tmp_path):
    """Gating on the diffused fraction would flag a healthy small recovery as
    divergence purely because the restrict mask was large."""
    d = _two_slab_volume(tmp_path)
    mask = np.ones((64, 64), dtype=np.float32)
    out, _, _ = AtlasLoadHiddenVolume().load(
        _solve(), d, emit="combined", fill_gaps=True, restrict_mask=mask,
        on_divergence="mark")
    hv = out.projection_scene.proxy_geometry[-1].metadata["hidden_volume"]
    assert hv["invented_fraction"] == pytest.approx(hv["raw_selection_fraction"])
    assert hv["invented_fraction"] < hv["substituted_fraction"]


def test_layers_mode_still_emits_per_layer(tmp_path):
    d = _two_slab_volume(tmp_path)
    out, _, _ = AtlasLoadHiddenVolume().load(
        _solve(), d, emit="layers", on_divergence="mark")
    assert len(out.projection_scene.proxy_geometry) >= 2


def test_invert_restrict_mask_flips_the_region(tmp_path):
    """Which side of a band mask holds the recovered layers is NOT fixed:
    measured live, the same foreground band caught 100% of the boiler's
    selection and 0% of sh001's, so the orientation has to be switchable."""
    d = _two_slab_volume(tmp_path)
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[:32, :] = 1.0
    node = AtlasLoadHiddenVolume()
    a, _, _ = node.load(_solve(), d, emit="combined", restrict_mask=mask,
                        invert_restrict_mask=False, on_divergence="mark")
    b, _, _ = node.load(_solve(), d, emit="combined", restrict_mask=mask,
                        invert_restrict_mask=True, on_divergence="mark")
    ha = a.projection_scene.proxy_geometry[-1].metadata["hidden_volume"]
    hb = b.projection_scene.proxy_geometry[-1].metadata["hidden_volume"]
    assert ha["restrict_inverted"] is False
    assert hb["restrict_inverted"] is True
    # Complementary halves: the substituted regions must not be the same.
    assert ha["substituted_fraction"] != pytest.approx(
        hb["substituted_fraction"], abs=1e-3)


def test_report_says_when_the_mask_catches_nothing(tmp_path):
    """The diagnostic that matters: a mask covering plenty of frame but none of
    the cleared selection means it is the wrong way round, and the report has to
    say so rather than just showing 0%."""
    d = _two_slab_volume(tmp_path)
    mask = np.zeros((64, 64), dtype=np.float32)   # empty region
    _, report, _ = AtlasLoadHiddenVolume().load(
        _solve(), d, emit="combined", restrict_mask=mask, on_divergence="mark")
    assert "catches NONE" in report or "fill SKIPPED" in report


def test_raymarch_says_double_sided_was_ignored(tmp_path):
    """A widget that does nothing must SAY it does nothing.

    `double_sided` defaults True and raymarch is the default extraction, so out
    of the box the widget is inert. The gate doctrine wants a visible
    explanation for a silent skip — a tooltip is documentation, the report is
    the explanation.
    """
    d = _two_slab_volume(tmp_path)
    _, report, _ = AtlasLoadHiddenVolume().load(
        _solve(), d, emit="layers", double_sided=True, on_divergence="mark")
    assert "double_sided ignored" in report

    _, off, _ = AtlasLoadHiddenVolume().load(
        _solve(), d, emit="layers", double_sided=False, on_divergence="mark")
    assert "double_sided ignored" not in off
