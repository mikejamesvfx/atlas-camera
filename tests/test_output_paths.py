"""node_helpers.output_paths: confined to the output dir, ComfyUI's refusal propagates."""

import sys
import types
from pathlib import Path

import pytest

from atlas_camera.comfy.node_helpers import output_paths


@pytest.fixture()
def outside_comfy(monkeypatch, tmp_path):
    """No folder_paths (import fails) and a scratch cwd for ./output."""
    monkeypatch.setitem(sys.modules, "folder_paths", None)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("prefix", ["../x", "atlas/../../x", "..", "atlas/..", "..\\x"])
def test_dotdot_escape_is_rejected(outside_comfy, prefix):
    with pytest.raises(ValueError, match="escapes the output directory") as e:
        output_paths(prefix)
    assert repr(prefix) in str(e.value)
    assert not (outside_comfy / "x").exists()


def test_absolute_prefix_is_rejected(outside_comfy, tmp_path):
    for prefix in (str(tmp_path / "abs" / "x"), "/abs/x"):
        with pytest.raises(ValueError, match="is absolute"):
            output_paths(prefix)


def test_normal_prefix_counts_up(outside_comfy):
    folder, stem = output_paths("atlas/scene")
    assert folder == Path("output") / "atlas" and stem == "scene_00001"
    assert (outside_comfy / "output" / "atlas").is_dir()
    (folder / f"{stem}.glb").write_bytes(b"")
    assert output_paths("atlas/scene") == (folder, "scene_00002")
    # Internal '..' that stays inside output/ is fine.
    assert output_paths("atlas/sub/../scene")[1].startswith("scene_")


def test_comfyui_refusal_propagates(monkeypatch, tmp_path):
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: str(tmp_path)

    def refuse(prefix, out_dir):
        raise Exception("**** ERROR: Saving image outside the output folder is not allowed.")

    fp.get_save_image_path = refuse
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception, match="outside the output folder"):
        output_paths("../../evil")
    assert not (tmp_path / "output").exists()   # no silent local fallback


def test_inside_comfyui_uses_folder_paths(monkeypatch, tmp_path):
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: str(tmp_path)
    fp.get_save_image_path = lambda prefix, out: (str(tmp_path / "atlas"), "scene", 7, "atlas",
                                                  prefix)
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    assert output_paths("atlas/scene") == (tmp_path / "atlas", "scene_00007")


def test_counter_sees_every_extension_and_grows_past_five_digits(tmp_path):
    from atlas_camera.comfy.node_helpers import _next_counter

    (tmp_path / "scene_00003.exr").write_bytes(b"")
    assert _next_counter(tmp_path, "scene") == 4          # an EXR counts, not only .glb
    (tmp_path / "scene_99999.glb").write_bytes(b"")
    (tmp_path / "scene_100000.glb").write_bytes(b"")
    assert _next_counter(tmp_path, "scene") == 100001     # :05 grows; never reuse 100000


@pytest.mark.skipif(sys.platform != "win32", reason="case-insensitive filesystem only")
def test_counter_is_case_insensitive_on_windows(tmp_path):
    from atlas_camera.comfy.node_helpers import _next_counter

    (tmp_path / "hero_00001.glb").write_bytes(b"")
    assert _next_counter(tmp_path, "Hero") == 2
