"""Review regressions with CI coverage: numpy-only, no torch, no OpenImageIO.

CI (.github/workflows/tests.yml) installs ``.[dev,ui,image,mcp,raw]`` -- no
torch and no OpenImageIO -- so the node-level regression tests that
``importorskip`` either never run there. This file re-asserts the important
ones against the pure logic underneath (some duplicate assertions that also
live in torch/OIIO-gated files, deliberately).

Must stay importable with ``sys.modules['torch'] = None`` and
``sys.modules['OpenImageIO'] = None``. ComfyUI's ``folder_paths`` is
simulated: ``None`` in ``sys.modules`` makes ``import folder_paths`` raise
ImportError (the "outside ComfyUI" branch); a fake module stands in for it
where the code reads checkpoints.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from atlas_camera.core import matrixzone as mz
from atlas_camera.core.scene_health import generated_object_grade


# ---------------------------------------------------------------------------
# core.matrixzone
# ---------------------------------------------------------------------------

_SWEEP_SIZES = [(3840, 2160), (7680, 4320), (1920, 1080), (1100, 620), (2048, 858),
                (1000, 3000), (4096, 4096), (777, 1333)]
_SWEEP_GRIDS = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (4, 4), (1, 8), (8, 1), (3, 5)]
_SWEEP_OVERLAPS = [(64, 64), (128, 128), (512, 512), (256, 64)]


def test_planner_clamp_keeps_every_render_rect_inside_the_render():
    checked = 0
    for w, h in _SWEEP_SIZES:
        for grid in _SWEEP_GRIDS:
            for ov in _SWEEP_OVERLAPS:
                try:
                    p = mz.plan_still(w, h, grid, overlap_min=ov)
                except ValueError:
                    continue          # "use a smaller grid" -- a refusal, not a bug
                rw, rh = p["render"]["width"], p["render"]["height"]
                for z in p["zones"]:
                    x, y, zw, zh = z["renderRect"]
                    assert x >= 0 and y >= 0 and x + zw <= rw and y + zh <= rh, (
                        w, h, grid, ov, z)
                    # the zone still covers the cell it owns
                    px, py, pw, ph = z["plateRect"]
                    assert x <= px and y <= py and px + pw <= x + zw and py + ph <= y + zh
                checked += 1
    assert checked > 100


def test_planner_tall_grid_case_from_the_review():
    # 3840x2160, 1x8, overlap 512: z10 started at y=-3 and z60 ended past 2176.
    p = mz.plan_still(3840, 2160, (1, 8), overlap_min=(512, 512))
    rh = p["render"]["height"]
    assert all(z["renderRect"][1] >= 0 and z["renderRect"][1] + z["renderRect"][3] <= rh
               for z in p["zones"])


def _texture(h, w, seed=0):
    """Smooth positive Rec.709-linear texture (structure on every line)."""
    rng = np.random.default_rng(seed)
    small = rng.random((h // 16 + 2, w // 16 + 2, 3)).astype(np.float32)
    big = mz.resize_bilinear(small, h + 32, w + 32)[:h, :w]
    return (0.05 + 0.6 * big).astype(np.float32)


def test_seam_step_test_fails_a_plus2_minus2_split_seam():
    w, h = 1024, 512
    p = mz.plan_still(w, h, (2, 1))
    (seam_x,) = {z["plateRect"][0] + z["plateRect"][2] - p["render"]["plateOrigin"][0]
                 for z in p["zones"] if z["index"] == [0, 0]}
    sdr = _texture(h, w)
    clean = mz.rec709_linear_to_acescg(sdr).astype(np.float32)
    seamed = clean.copy()
    seamed[: h // 2, seam_x:] *= 4.0      # +2 stops on the upper half of the seam
    seamed[h // 2:, seam_x:] *= 0.25      # -2 stops on the lower half

    # Whole-line median of the SDR-controlled step cancels to ~0 -- the bug.
    k = mz.SEAM_STRIP_PX
    lg = np.log2(seamed.mean(-1))
    ls = np.log2(clean.mean(-1))
    d = (lg[:, seam_x - k:seam_x].mean(1) - lg[:, seam_x:seam_x + k].mean(1)) - (
        ls[:, seam_x - k:seam_x].mean(1) - ls[:, seam_x:seam_x + k].mean(1))
    assert abs(np.median(d)) < 0.5

    rep = mz.seam_step_test(seamed, p, sdr_linear=sdr)
    assert rep["sdr_controlled"] is True
    (s,) = rep["seams"]
    assert s["windows"] >= 2
    assert s["step_stops"] == pytest.approx(2.0, abs=0.1)
    assert s["flagged"] and rep["pass"] is False

    ok = mz.seam_step_test(clean, p, sdr_linear=sdr)
    assert ok["pass"] is True and not ok["flagged"]


def _ramp(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([0.1 + xx / w, 0.2 + yy / h, 0.3 + 0.5 * xx / w * yy / h],
                    -1).astype(np.float32)


def test_stitch_never_writes_into_the_callers_global_pass():
    w, h = 1100, 620
    p = mz.plan_still(w, h, (2, 2))
    render = mz.pad_to_render(_ramp(w, h), p)
    zones = mz.crop_zones(render, p)
    glob = np.ascontiguousarray(render, dtype=np.float32) * 4.0   # render-size float32
    before = glob.copy()
    out1, _ = mz.stitch(zones, p, global_hdr=glob)
    np.testing.assert_array_equal(glob, before)
    out2, _ = mz.stitch(zones, p, global_hdr=glob)                 # 4.0 -> 2.0 -> 1.0 before
    np.testing.assert_array_equal(glob, before)
    np.testing.assert_allclose(out1, out2, rtol=1e-6)


def test_resize_bilinear_returns_a_fresh_array():
    a = np.random.default_rng(1).random((24, 36, 3)).astype(np.float32)
    out = mz.resize_bilinear(a, 24, 36)
    assert out is not a and not np.shares_memory(out, a)
    np.testing.assert_array_equal(out, a)
    out += 1.0
    assert a.max() <= 1.0
    small = mz.resize_bilinear(a, 12, 18)
    assert small.dtype == np.float32 and small.shape == (12, 18, 3)


def test_sdr_detail_transfer_pure_colour_ramp_has_no_hue_shift():
    H, W = 96, 144
    xx = np.mgrid[0:H, 0:W][1].astype(np.float32)
    pal = np.eye(3, dtype=np.float32)
    disp = ((0.1 + 0.6 * (xx / W))[..., None] * pal[xx.astype(int) % 3]).astype(np.float32)
    hdr = (mz.rec709_linear_to_acescg(mz.srgb_to_linear_f32(disp)) * 2.0).astype(np.float32)
    out, _ = mz.sdr_detail_transfer(hdr, disp, radius=16)

    def chroma(a):
        a = np.maximum(a, 1e-6)
        return a / a.sum(-1, keepdims=True)

    assert np.abs(chroma(out) - chroma(hdr)).max() < 1e-4
    assert np.abs(np.log2(np.maximum(out, 1e-6) / np.maximum(hdr, 1e-6))).max() < 1e-3


# ---------------------------------------------------------------------------
# core.scene_health.generated_object_grade
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [None, float("nan"), float("inf"), -float("inf"),
                                   0.0, -1.0, "not-a-number"])
def test_generated_object_grade_refuses_unusable_scale(scale):
    grade, issues = generated_object_grade(scale, 0.01, registration_note="why")
    assert grade == "refuse"
    assert "no usable scale registration" in issues and "why" in issues


def test_generated_object_grade_refuses_non_finite_rel_mad():
    assert generated_object_grade(1.0, float("nan"))[0] == "refuse"


def test_generated_object_grade_accepts_a_good_registration():
    grade, _ = generated_object_grade(1.0, 0.01)
    assert grade != "refuse"


# ---------------------------------------------------------------------------
# comfy.node_helpers output paths
# ---------------------------------------------------------------------------

@pytest.fixture
def no_comfy(monkeypatch, tmp_path):
    """Outside ComfyUI: ``import folder_paths`` raises ImportError; cwd = tmp."""
    monkeypatch.setitem(sys.modules, "folder_paths", None)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("prefix", ["../evil", "atlas/../../evil", "..", "a/..",
                                    "/abs/evil", "C:/abs/evil", "C:\\abs\\evil",
                                    "\\\\server\\share\\evil"])
def test_output_paths_rejects_escaping_prefixes(no_comfy, prefix):
    from atlas_camera.comfy.node_helpers import output_paths
    with pytest.raises(ValueError, match="output"):
        output_paths(prefix)
    assert not (no_comfy.parent / "evil").exists()


def test_output_paths_fallback_counts_up(no_comfy):
    from atlas_camera.comfy.node_helpers import output_paths
    folder, stem = output_paths("atlas/scene")
    assert Path(folder).resolve() == (no_comfy / "output" / "atlas").resolve()
    assert stem == "scene_00001"
    (Path(folder) / f"{stem}.exr").write_bytes(b"")
    assert output_paths("atlas/scene")[1] == "scene_00002"


def test_next_counter_any_extension_and_past_99999(tmp_path):
    from atlas_camera.comfy.node_helpers import _next_counter
    for n in ("hero_00007.exr", "hero_00003_vertex_hdr.ply", "hero_00010.glb",
              "heroic_00500.glb", "other_99999.glb", "hero_notanumber.glb"):
        (tmp_path / n).write_bytes(b"")
    assert _next_counter(tmp_path, "hero") == 11
    (tmp_path / "hero_123456.glb").write_bytes(b"")
    assert _next_counter(tmp_path, "hero") == 123457
    assert _next_counter(tmp_path, "fresh") == 1
    if os.name == "nt":
        (tmp_path / "Hero_200000.exr").write_bytes(b"")
        assert _next_counter(tmp_path, "hero") == 200001


def test_project_output_paths_routes_into_the_lane(tmp_path):
    from atlas_camera.comfy.node_helpers import project_output_paths
    from atlas_camera.core.project import build_project

    project = build_project(str(tmp_path), "proj", "sh010", "standard")
    folder, stem = project_output_paths(project, "scenes", "atlas/../../evil/hero")
    lane = Path(project.subdir("scenes")).resolve()
    assert Path(folder).resolve() == lane and lane.is_dir()
    assert lane.is_relative_to(tmp_path.resolve())
    assert stem == "hero_00001"
    (Path(folder) / f"{stem}.glb").write_bytes(b"")
    assert project_output_paths(project, "scenes", "hero")[1] == "hero_00002"
    with pytest.raises(ValueError, match="unknown shot subfolder"):
        project_output_paths(project, "../outside", "hero")


# ---------------------------------------------------------------------------
# comfy.sam3_core_backend
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_folder_paths(monkeypatch, tmp_path):
    files = {"sam3.1_multiplex_fp16.safetensors": tmp_path / "sam3_a.safetensors",
             "legacy_sam3_v0.safetensors": tmp_path / "legacy.safetensors",
             "sub/my_sam3.safetensors": tmp_path / "sub_sam3.safetensors",
             "sam3d_objects.safetensors": tmp_path / "sam3d.safetensors",
             "sdxl.safetensors": tmp_path / "sdxl.safetensors"}
    for p in files.values():
        p.write_bytes(b"x")
    mod = types.ModuleType("folder_paths")
    mod.get_filename_list = lambda kind: list(files) if kind == "checkpoints" else []
    mod.get_full_path = lambda kind, name: str(files[name]) if name in files else None
    monkeypatch.setitem(sys.modules, "folder_paths", mod)
    return files


def test_sam3_combo_values_are_fixed():
    from atlas_camera.comfy import sam3_core_backend as S
    assert S.sam3_checkpoint_choices() == ["hf:facebook/sam3", "core:auto"]
    assert S.SAM3_CHECKPOINT_CHOICES[:2] == (S.HF_BACKEND, S.CORE_AUTO)


def test_sam3_combo_values_do_not_depend_on_disk(fake_folder_paths):
    from atlas_camera.comfy import sam3_core_backend as S
    assert S.sam3_checkpoint_choices() == ["hf:facebook/sam3", "core:auto"]


def test_validate_checkpoint_choice(fake_folder_paths):
    from atlas_camera.comfy import sam3_core_backend as S
    assert S.validate_checkpoint_choice(None) is True
    assert S.validate_checkpoint_choice(S.HF_BACKEND) is True
    assert S.validate_checkpoint_choice(S.CORE_AUTO) is True
    assert S.validate_checkpoint_choice("legacy_sam3_v0.safetensors") is True
    assert S.validate_checkpoint_choice("sub\\my_sam3.safetensors") is True
    err = S.validate_checkpoint_choice("gone.safetensors")
    assert isinstance(err, str) and "gone.safetensors" in err


def test_validate_checkpoint_choice_outside_comfy(monkeypatch):
    from atlas_camera.comfy import sam3_core_backend as S
    monkeypatch.setitem(sys.modules, "folder_paths", None)
    assert S.validate_checkpoint_choice(S.CORE_AUTO) is True
    assert isinstance(S.validate_checkpoint_choice("legacy_sam3_v0.safetensors"), str)


def test_resolve_core_checkpoint_branches(fake_folder_paths):
    from atlas_camera.comfy import sam3_core_backend as S
    # core:auto -> first sorted *sam3* (sam3d excluded)
    assert S.resolve_core_checkpoint(S.CORE_AUTO) == "legacy_sam3_v0.safetensors"
    # legacy: a bare filename saved in the combo before the values were fixed
    assert S.resolve_core_checkpoint("sam3.1_multiplex_fp16.safetensors") == \
        "sam3.1_multiplex_fp16.safetensors"
    with pytest.raises(S.Sam3CheckpointMissing, match="not_here"):
        S.resolve_core_checkpoint("not_here.safetensors")
    # override wins, separators normalised
    assert S.resolve_core_checkpoint(S.HF_BACKEND, "sub\\my_sam3.safetensors") == \
        "sub/my_sam3.safetensors"
    with pytest.raises(S.Sam3CheckpointMissing, match="missing.safetensors"):
        S.resolve_core_checkpoint(S.CORE_AUTO, "missing.safetensors")
    assert S.wants_core(S.HF_BACKEND) is False
    assert S.wants_core(S.HF_BACKEND, "x") is True
    assert S.wants_core("legacy_sam3_v0.safetensors") is True


def test_core_auto_with_no_checkpoint_is_missing(monkeypatch):
    from atlas_camera.comfy import sam3_core_backend as S
    monkeypatch.setitem(sys.modules, "folder_paths", None)
    with pytest.raises(S.Sam3CheckpointMissing, match="core:auto"):
        S.resolve_core_checkpoint(S.CORE_AUTO)
    assert S.checkpoint_cache_token(S.CORE_AUTO) == "missing:core:auto"
    assert S.checkpoint_cache_token(S.CORE_AUTO, " x.safetensors ") == "missing:x.safetensors"


def test_core_checkpoint_label():
    from atlas_camera.comfy import sam3_core_backend as S
    assert S.core_checkpoint_label(S.CORE_AUTO, "", "a.st") == "core:auto -> a.st"
    assert S.core_checkpoint_label(S.CORE_AUTO, "a.st", "a.st") == "a.st"
    assert S.core_checkpoint_label("legacy.st", None, "legacy.st") == "legacy.st"


def test_checkpoint_cache_token_tracks_the_file(fake_folder_paths):
    from atlas_camera.comfy import sam3_core_backend as S
    assert S.checkpoint_cache_token(S.HF_BACKEND) == "hf"
    path = fake_folder_paths["legacy_sam3_v0.safetensors"]
    tok = S.checkpoint_cache_token(S.CORE_AUTO)
    assert tok == f"legacy_sam3_v0.safetensors:{os.stat(path).st_mtime_ns}"
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert S.checkpoint_cache_token(S.CORE_AUTO) != tok
    assert S.checkpoint_cache_token("legacy_sam3_v0.safetensors").startswith(
        "legacy_sam3_v0.safetensors:")


# ---------------------------------------------------------------------------
# exporters.scene_glb (numpy + Pillow; the EXR write itself needs OIIO)
# ---------------------------------------------------------------------------

_QV = np.array([[0, 0, -5], [1, 0, -5], [0, 1, -5], [1, 1, -6]], dtype=np.float32)
_QF = np.array([[0, 1, 2], [1, 3, 2]])
_QUV = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)


def test_plan_and_write_dedup_one_source_image(tmp_path):
    from atlas_camera.exporters.scene_glb import (
        SceneLayer,
        plan_scene_glb,
        read_glb_json,
        write_scene_glb,
    )
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 300     # one shared object
    layers = [SceneLayer("a", _QV, _QF, _QUV, png, extras={"atlas_plate": "primary"}),
              SceneLayer("b", _QV - [0, 0, 2], _QF, _QUV, png,
                         extras={"atlas_plate": "primary"})]
    plan = plan_scene_glb(layers)
    assert len(plan["images"]) == 1 and plan["images"][0]["layers"] == 2
    out = write_scene_glb(layers, tmp_path / "s.glb")
    assert out["bytes"] == plan["bytes"]
    g = read_glb_json(out["glb"])
    assert len(g["images"]) == 1 and len(g["nodes"]) == 2
    # a distinct bytes object is a distinct plate
    other = SceneLayer("c", _QV, _QF, _QUV, bytes(bytearray(png)))
    assert len(plan_scene_glb(layers + [other])["images"]) == 2


def _solve_two_primary_meshes():
    from atlas_camera.core.proxy_geometry import PROXY_ROLE
    from atlas_camera.core.schema import (
        AtlasExtrinsics,
        AtlasIntrinsics,
        AtlasProxyPrimitive,
        AtlasSolve,
        LatentCamera,
    )
    intr = AtlasIntrinsics(image_width=64, image_height=40, focal_length_mm=35.0,
                           sensor_width_mm=36.0, fx_px=50.0, fy_px=50.0, cx_px=32, cy_px=20)
    solve = AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=AtlasExtrinsics()))
    for name, dz in (("relief_a", 0.0), ("relief_b", -2.0)):
        solve.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
            name=name, primitive_type="mesh", dimensions=(0.0, 0.0, 0.0),
            material="atlas_projection_proxy",
            metadata={"role": PROXY_ROLE,
                      "vertices": (_QV + [0, 0, dz]).reshape(-1).tolist(),
                      "faces": _QF.reshape(-1).tolist(),
                      "uvs": _QUV.reshape(-1).tolist()}))
    return solve


def test_collect_scene_layers_records_exr_output_path_inside_output_root(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    from atlas_camera.exporters.scene_glb import collect_scene_layers, plan_scene_glb

    out_root = tmp_path / "output"
    lane = out_root / "proj" / "sh010" / "scenes"
    plate = Image.new("RGB", (64, 40), (200, 120, 60))
    scene = collect_scene_layers(_solve_two_primary_meshes(), plate, exr_dir=lane,
                                 exr_prefix="hero_00001", output_root=out_root)
    assert [l.name for l in scene.layers] == ["relief_a", "relief_b"]
    (job,) = scene.jobs
    assert job.known["exr"] == "hero_00001_primary.exr"
    assert job.known["exr_output_path"] == "proj/sh010/scenes/hero_00001_primary.exr"
    assert len(job.targets) == 2 and not lane.exists()    # planned, nothing written
    # both primary meshes share the ONE encoded plate -> embedded once
    plan = plan_scene_glb(scene.layers)
    assert len(plan["images"]) == 1 and plan["images"][0]["layers"] == 2

    # outside the output root: no output-relative path is invented
    elsewhere = collect_scene_layers(_solve_two_primary_meshes(), plate,
                                     exr_dir=tmp_path / "elsewhere", output_root=out_root)
    assert "exr_output_path" not in elsewhere.jobs[0].known


def test_build_scene_layers_with_output_root_records_exr_output_path(tmp_path):
    pytest.importorskip("PIL.Image")
    pytest.importorskip("OpenImageIO")     # the sidecar write itself needs OIIO
    from PIL import Image

    from atlas_camera.exporters.scene_glb import build_scene_layers
    out_root = tmp_path / "output"
    layers, sidecars, notes = build_scene_layers(
        _solve_two_primary_meshes(), Image.new("RGB", (64, 40)),
        exr_dir=out_root / "scenes", exr_prefix="t", output_root=out_root)
    assert all(l.extras["exr_output_path"] == "scenes/t_primary.exr" for l in layers)


# ---------------------------------------------------------------------------
# mcp.comfy_http dynamic-combo string options
# ---------------------------------------------------------------------------

def test_dynamic_combo_bare_string_options():
    from atlas_camera.mcp import comfy_http as C
    oi = {"Remesh": {"input": {"required": {
        "mesh": ["MESH", {}],
        "mode": ["COMFY_DYNAMICCOMBO_V3", {"options": [
            "uniform",
            {"key": "adaptive", "inputs": {"required": {
                "level": ["INT", {"min": 1, "max": 8}],
                "guide": ["IMAGE", {}]}}},
        ]}],
        "iterations": ["INT", {"min": 1, "max": 10}],
    }}}}
    assert C._dyn_option_key("uniform") == "uniform"
    assert C._dyn_chosen(oi["Remesh"]["input"]["required"]["mode"][1], "uniform") is None
    assert C.widget_inputs(oi, "Remesh", ["uniform", 3]) == {"mode": "uniform",
                                                             "iterations": 3}
    assert C.widget_inputs(oi, "Remesh", ["adaptive", 4, 2]) == {
        "mode": "adaptive", "mode.level": 4, "iterations": 2}
    items, want = C._widget_walk(oi, "Remesh", ["uniform", 3])
    assert want == 2
    assert items[0] == ("mode", ["COMBO", {"options": ["uniform", "adaptive"]}], "uniform")
    items, want = C._widget_walk(oi, "Remesh", ["adaptive", 4, 2])
    assert want == 3 and [k for k, _, _ in items] == ["mode", "mode.level", "iterations"]
    ui = {"nodes": [{"id": 1, "type": "Remesh", "inputs": [], "outputs": [],
                     "widgets_values": ["adaptive", 99, 2]}], "links": []}
    errs, _ = C.validate_ui(ui, oi)
    assert any("mode.level=99" in e for e in errs)
    ui["nodes"][0]["widgets_values"] = ["uniform", 3]
    errs, _ = C.validate_ui(ui, oi)
    assert not any("mode" in e or "widgets_values" in e for e in errs)


# ---------------------------------------------------------------------------
# the file's own contract
# ---------------------------------------------------------------------------

def test_this_file_does_not_need_torch_or_oiio():
    src = Path(__file__).read_text(encoding="utf-8")
    for mod in ("torch", "OpenImageIO"):
        assert f"import {mod}" not in src.replace(f'importorskip("{mod}")', "")
    assert "torch" not in {m.split(".")[0] for m in sys.modules if sys.modules[m] is not None} \
        or True   # informational: other test files may have imported it in-process
