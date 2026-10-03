"""AtlasSAM3Mask on ComfyUI's own SAM3 (sam3.1_multiplex_fp16), via stubbed core modules."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from atlas_camera.comfy import sam3_core_backend as backend  # noqa: E402

H, W = 32, 48

#: What the fake models/checkpoints folder lists.
CHECKPOINTS = ["sam3.1_multiplex_fp16.safetensors", "sam3d_body.safetensors",
               "sdxl.safetensors"]


class _FakeClip:
    def __init__(self):
        self.prompts = []

    def tokenize(self, text):
        self.prompts.append(text)
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return [[tokens, {}]]


class _FakeDetect:
    """Core SAM3_Detect stand-in: 'machine' -> two blobs, anything else -> none."""

    calls = []

    @classmethod
    def execute(cls, model, image, conditioning=None, threshold=0.5,
                refine_iterations=2, individual_masks=False):
        cls.calls.append((conditioning[0][0], threshold, individual_masks))
        if conditioning[0][0].startswith("machine"):
            a = torch.zeros(H, W)
            a[2:10, 2:10] = 1          # small
            b = torch.zeros(H, W)
            b[10:30, 10:40] = 1        # large
            stack = torch.stack([a, b])
        else:
            stack = torch.zeros(0, H, W)
        return types.SimpleNamespace(result=(stack, []))


@pytest.fixture()
def fake_comfy(monkeypatch):
    clip = _FakeClip()
    fp = types.ModuleType("folder_paths")
    fp.get_filename_list = lambda kind: list(CHECKPOINTS)
    fp.get_full_path_or_raise = lambda kind, name: f"/models/{kind}/{name}"
    fp.get_folder_paths = lambda kind: []
    sd = types.ModuleType("comfy.sd")
    sd.load_checkpoint_guess_config = lambda path, **kw: ("MODEL", clip, None, None)
    comfy_pkg = types.ModuleType("comfy")
    comfy_pkg.sd = sd
    extras = types.ModuleType("comfy_extras")
    nodes_sam3 = types.ModuleType("comfy_extras.nodes_sam3")
    nodes_sam3.SAM3_Detect = _FakeDetect
    for name, mod in {"folder_paths": fp, "comfy": comfy_pkg, "comfy.sd": sd,
                      "comfy_extras": extras, "comfy_extras.nodes_sam3": nodes_sam3}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    # A real ComfyUI's model_management must not leak in through the cache.
    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    backend._LOADED.clear()
    _FakeDetect.calls.clear()
    yield clip
    backend._LOADED.clear()


def _set_checkpoints(monkeypatch, names):
    monkeypatch.setattr(sys.modules["folder_paths"], "get_filename_list",
                        lambda kind: list(names))


# --- F-8 / D30: fixed combo values + override --------------------------------

def test_choices_are_fixed_values_not_read_from_disk(fake_comfy):
    # A disk-listed combo failed validation on any machine without the same
    # file. The values are FIXED; HF stays first and default.
    assert backend.sam3_checkpoint_choices() == ["hf:facebook/sam3", "core:auto"]
    assert backend.core_sam3_checkpoints() == [
        "sam3.1_multiplex_fp16.safetensors"]                      # sam3d excluded
    assert backend.core_sam3_available()


def test_choices_outside_comfy_are_the_same_fixed_values(monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", None)
    assert backend.sam3_checkpoint_choices() == ["hf:facebook/sam3", "core:auto"]
    assert backend.core_sam3_checkpoints() == []


def test_core_auto_picks_first_sorted_sam3_file(fake_comfy, monkeypatch):
    _set_checkpoints(monkeypatch, ["zz_sam3_b.safetensors", "sam3d_body.safetensors",
                                   "a_sam3_a.safetensors", "sdxl.safetensors"])
    assert backend.resolve_core_checkpoint("core:auto") == "a_sam3_a.safetensors"


def test_override_resolves_the_exact_file(fake_comfy):
    assert backend.resolve_core_checkpoint(
        "hf:facebook/sam3", "sdxl.safetensors") == "sdxl.safetensors"
    assert backend.wants_core("hf:facebook/sam3", "x.safetensors")
    assert not backend.wants_core("hf:facebook/sam3", "  ")
    assert backend.wants_core("core:auto", "")


def test_missing_checkpoint_raises_a_named_lookup(fake_comfy, monkeypatch):
    with pytest.raises(backend.Sam3CheckpointMissing, match="nope.safetensors"):
        backend.resolve_core_checkpoint("core:auto", "nope.safetensors")
    _set_checkpoints(monkeypatch, ["sdxl.safetensors"])
    with pytest.raises(backend.Sam3CheckpointMissing, match="core:auto"):
        backend.resolve_core_checkpoint("core:auto")


def test_each_concept_asks_for_many_detections(fake_comfy):
    masks, matched = backend.core_sam3_instances(
        torch.rand(1, H, W, 3), "machine, rock:3",
        ckpt_name="sam3.1_multiplex_fp16.safetensors")
    # A bare core prompt returns ONE detection; Atlas wants every instance.
    assert fake_comfy.prompts == ["machine:64", "rock:64"]
    assert all(c[2] is True for c in _FakeDetect.calls)       # individual_masks
    assert matched == ["machine", "machine"]
    assert masks[0].sum() > masks[1].sum()                     # largest first


def _node():
    from atlas_camera.comfy.nodes_inpaint import AtlasSAM3Mask
    return AtlasSAM3Mask()


def test_node_separate_mode_via_core(fake_comfy):
    stack, report = _node().segment(
        torch.rand(1, H, W, 3), "machine", output_mode="separate", max_instances=1,
        sam3_checkpoint="core:auto")
    assert stack.shape == (1, H, W) and float(stack.sum()) == 20 * 30
    assert "sam3.1_multiplex_fp16" in report and "largest first" in report


def test_node_merged_mode_via_core_names_the_auto_pick(fake_comfy):
    mask, report = _node().segment(
        torch.rand(1, H, W, 3), "machine", sam3_checkpoint="core:auto")
    assert mask.shape == (1, H, W) and float(mask.sum()) == 8 * 8 + 20 * 30
    assert "of frame" in report
    assert "core:auto -> sam3.1_multiplex_fp16.safetensors" in report


def test_node_no_match_is_empty_not_error(fake_comfy):
    mask, report = _node().segment(
        torch.rand(1, H, W, 3), "giraffe", sam3_checkpoint="core:auto")
    assert float(mask.sum()) == 0 and "NO MATCH" in report


def test_node_override_selects_the_exact_file(fake_comfy, monkeypatch):
    loaded = []
    sd = sys.modules["comfy.sd"]
    real = sd.load_checkpoint_guess_config

    def load(path, **kw):
        loaded.append(path)
        return real(path, **kw)
    monkeypatch.setattr(sd, "load_checkpoint_guess_config", load)
    _, report = _node().segment(
        torch.rand(1, H, W, 3), "machine", sam3_checkpoint="hf:facebook/sam3",
        sam3_checkpoint_override="sam3.1_multiplex_fp16.safetensors")
    assert loaded == ["/models/checkpoints/sam3.1_multiplex_fp16.safetensors"]
    assert "(sam3.1_multiplex_fp16.safetensors)" in report


def test_node_missing_override_is_empty_mask_with_named_report(fake_comfy):
    mask, report = _node().segment(
        torch.rand(1, H, W, 3), "machine", sam3_checkpoint="hf:facebook/sam3",
        sam3_checkpoint_override="sam3_elsewhere.safetensors")
    assert mask.shape == (1, H, W) and float(mask.sum()) == 0
    assert "MISSING" in report and "sam3_elsewhere.safetensors" in report


def test_node_core_auto_with_no_sam3_file_is_empty_with_report(fake_comfy, monkeypatch):
    _set_checkpoints(monkeypatch, ["sdxl.safetensors"])
    mask, report = _node().segment(
        torch.rand(1, H, W, 3), "machine", sam3_checkpoint="core:auto")
    assert float(mask.sum()) == 0 and "MISSING" in report and "core:auto" in report


# --- F-6 item 6 / D19: any non-missing core failure RAISES -------------------

def test_node_core_load_failure_raises_naming_checkpoint(fake_comfy, monkeypatch):
    # Only a MISSING file degrades; a load error / API drift raises (as the HF
    # path does for anything but the gated repo) -- never a silent empty mask
    # that a downstream sky card would build on.
    def boom(*a, **k):
        raise ValueError("bad checkpoint")
    monkeypatch.setattr(sys.modules["comfy.sd"], "load_checkpoint_guess_config", boom)
    with pytest.raises(RuntimeError, match=r"sam3\.1_multiplex_fp16.*bad checkpoint"):
        _node().segment(torch.rand(1, H, W, 3), "machine", sam3_checkpoint="core:auto")


def test_node_core_oom_keeps_its_type(fake_comfy, monkeypatch):
    # ComfyUI recognises an OOM by type and unloads models; wrapping it in a
    # RuntimeError would hide that. The checkpoint rides along as a note.
    class OutOfMemoryError(RuntimeError):
        pass

    def boom(*a, **k):
        raise OutOfMemoryError("CUDA out of memory")
    monkeypatch.setattr(sys.modules["comfy.sd"], "load_checkpoint_guess_config", boom)
    with pytest.raises(OutOfMemoryError) as info:
        _node().segment(torch.rand(1, H, W, 3), "machine", sam3_checkpoint="core:auto")
    notes = getattr(info.value, "__notes__", None)
    if notes is not None:                                  # py>=3.11
        assert any("sam3.1_multiplex_fp16" in n for n in notes)


# --- P-1(d): the module-global cache must be releasable ----------------------

def test_switching_checkpoint_releases_the_previous_model_first(fake_comfy, monkeypatch):
    _set_checkpoints(monkeypatch, ["a_sam3.safetensors", "b_sam3.safetensors"])
    held_during_load = []
    sd = sys.modules["comfy.sd"]
    real = sd.load_checkpoint_guess_config

    def load(path, **kw):
        held_during_load.append(sorted(backend._LOADED))
        return real(path, **kw)
    monkeypatch.setattr(sd, "load_checkpoint_guess_config", load)
    backend._load("a_sam3.safetensors")
    backend._load("a_sam3.safetensors")                 # cached: no reload
    backend._load("b_sam3.safetensors")
    # the old model is dropped BEFORE the new one loads: never two resident
    assert held_during_load == [[], []]
    assert list(backend._LOADED) == ["b_sam3.safetensors"]


def test_release_function_drops_the_cache(fake_comfy):
    backend._load("sam3.1_multiplex_fp16.safetensors")
    assert backend.release_sam3() is True
    assert backend._LOADED == {}
    assert backend.release_sam3() is False


def test_without_model_management_the_fallback_still_caches(fake_comfy):
    # The fake 'comfy' package has no model_management: the hook is a no-op
    # and clear-on-switch + release_sam3() are what free memory.
    assert backend._install_unload_hook() is False
    backend._load("sam3.1_multiplex_fp16.safetensors")
    assert "sam3.1_multiplex_fp16.safetensors" in backend._LOADED


def test_comfy_unload_all_models_releases_sam3(fake_comfy, monkeypatch):
    calls = []
    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = lambda: calls.append("orig")
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(sys.modules["comfy"], "model_management", mm, raising=False)
    backend._load("sam3.1_multiplex_fp16.safetensors")    # installs the hook
    assert backend._LOADED
    hooked = mm.unload_all_models
    assert hooked is not calls.append
    assert backend._install_unload_hook() is True        # idempotent
    assert mm.unload_all_models is hooked
    mm.unload_all_models()                               # /free, OOM handler, ...
    assert backend._LOADED == {} and calls == ["orig"]


# --- widget + cascade contract -------------------------------------------------

def test_widgets_are_appended_last_with_hf_default():
    from atlas_camera.comfy.nodes_inpaint import AtlasSAM3Mask
    from atlas_camera.comfy.nodes_viewport import AtlasInput
    for cls in (AtlasSAM3Mask, AtlasInput):
        opt = cls.INPUT_TYPES()["optional"]
        assert list(opt)[-2:] == ["sam3_checkpoint", "sam3_checkpoint_override"]
        assert opt["sam3_checkpoint"][1]["default"] == "hf:facebook/sam3"
        assert opt["sam3_checkpoint"][0] == ["hf:facebook/sam3", "core:auto"]
        assert opt["sam3_checkpoint_override"][0] == "STRING"
        assert opt["sam3_checkpoint_override"][1]["default"] == ""


def _cascade_sams(**kw):
    from atlas_camera.comfy.node_helpers import _MiniGraphBuilder, build_segmentation_cascade

    g = _MiniGraphBuilder()
    ref, path = build_segmentation_cascade(
        g, "img", "sky", policy="semantic", have_native_sam3=False, registry={}, **kw)
    sams = [n for n in g.finalize().values() if n["class_type"] == "AtlasSAM3Mask"]
    return ref, path, sams


def test_cascade_passes_checkpoint_and_counts_core_as_native():
    ref, path, sams = _cascade_sams(sam3_checkpoint="core:auto")
    assert ref is not None and "core core:auto" in path
    assert len(sams) == 1 and sams[0]["inputs"]["sam3_checkpoint"] == "core:auto"
    assert "sam3_checkpoint_override" not in sams[0]["inputs"]


def test_cascade_threads_the_override():
    ref, path, sams = _cascade_sams(sam3_checkpoint="hf:facebook/sam3",
                                    sam3_checkpoint_override="my_sam3.safetensors")
    assert ref is not None and "my_sam3.safetensors" in path
    assert sams[0]["inputs"]["sam3_checkpoint_override"] == "my_sam3.safetensors"


def test_cascade_hf_default_emits_no_checkpoint_inputs():
    ref, path, sams = _cascade_sams(sam3_checkpoint="hf:facebook/sam3")
    assert ref is None and path == "none"                # no native, no SegFormer


def test_legacy_exact_file_choice_still_resolves(fake_comfy, monkeypatch):
    """Graphs saved before the combo values were fixed carry a bare filename;
    programmatic callers may too. The resolver accepts it when installed and
    names it when not."""
    _set_checkpoints(monkeypatch, ["sam3.1_multiplex_fp16.safetensors"])
    assert backend.resolve_core_checkpoint(
        "sam3.1_multiplex_fp16.safetensors") == "sam3.1_multiplex_fp16.safetensors"
    with pytest.raises(backend.Sam3CheckpointMissing, match="gone.safetensors"):
        backend.resolve_core_checkpoint("gone.safetensors")


# --- review fixes: legacy filename validation, cache token, shared label -----

def _sam3_node_classes():
    from atlas_camera.comfy.nodes_inpaint import AtlasSAM3Mask
    from atlas_camera.comfy.nodes_viewport import AtlasInput
    return (AtlasSAM3Mask, AtlasInput)


def test_validate_inputs_names_only_sam3_checkpoint():
    """ComfyUI skips its built-in list/min/max checks for every input
    VALIDATE_INPUTS names, and for ALL inputs if it takes **kwargs -- so the
    signature must name sam3_checkpoint and nothing else."""
    import inspect
    for cls in _sam3_node_classes():
        spec = inspect.getfullargspec(cls.VALIDATE_INPUTS)
        assert spec.args == ["cls", "sam3_checkpoint"], cls.__name__
        assert spec.varkw is None, cls.__name__


@pytest.mark.parametrize("value", ["hf:facebook/sam3", "core:auto", None])
def test_validate_inputs_fixed_values_pass(fake_comfy, monkeypatch, value):
    _set_checkpoints(monkeypatch, [])          # even with nothing installed
    for cls in _sam3_node_classes():
        assert cls.VALIDATE_INPUTS(sam3_checkpoint=value) is True


def test_validate_inputs_legacy_installed_filename_validates_and_resolves(
        fake_comfy, monkeypatch):
    legacy = "sam3.1_multiplex_fp16.safetensors"
    _set_checkpoints(monkeypatch, [legacy])
    for cls in _sam3_node_classes():
        assert cls.VALIDATE_INPUTS(sam3_checkpoint=legacy) is True
    assert backend.resolve_core_checkpoint(legacy) == legacy
    mask, report = _node().segment(torch.rand(1, H, W, 3), "machine",
                                   sam3_checkpoint=legacy)
    assert float(mask.sum()) > 0 and legacy in report and "core:auto" not in report


def test_validate_inputs_unknown_filename_is_a_named_error(fake_comfy, monkeypatch):
    _set_checkpoints(monkeypatch, ["sam3.1_multiplex_fp16.safetensors"])
    for cls in _sam3_node_classes():
        res = cls.VALIDATE_INPUTS(sam3_checkpoint="gone.safetensors")
        assert isinstance(res, str) and "gone.safetensors" in res


def test_is_changed_token_changes_when_checkpoint_appears(fake_comfy, monkeypatch,
                                                          tmp_path):
    name = "sam3.1_multiplex_fp16.safetensors"
    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "get_full_path",
                        lambda kind, n: str(tmp_path / n), raising=False)
    _set_checkpoints(monkeypatch, [])
    for cls in _sam3_node_classes():
        before = cls.IS_CHANGED(sam3_checkpoint="core:auto")
        before_ov = cls.IS_CHANGED(sam3_checkpoint="hf:facebook/sam3",
                                   sam3_checkpoint_override=name)
        assert before.startswith("missing:") and before_ov == f"missing:{name}"
        (tmp_path / name).write_bytes(b"x")
        _set_checkpoints(monkeypatch, [name])
        after = cls.IS_CHANGED(sam3_checkpoint="core:auto", image=None,
                               concepts="sky")            # other widgets ignored
        assert after != before and after.startswith(f"{name}:")
        assert after == f"{name}:{(tmp_path / name).stat().st_mtime_ns}"
        assert cls.IS_CHANGED(sam3_checkpoint="hf:facebook/sam3",
                              sam3_checkpoint_override=name) != before_ov
        (tmp_path / name).unlink()
        _set_checkpoints(monkeypatch, [])


def test_is_changed_is_constant_on_the_hf_path(fake_comfy, monkeypatch):
    for cls in _sam3_node_classes():
        a = cls.IS_CHANGED(sam3_checkpoint="hf:facebook/sam3")
        _set_checkpoints(monkeypatch, ["sam3_new.safetensors"])
        b = cls.IS_CHANGED(sam3_checkpoint="hf:facebook/sam3",
                           sam3_checkpoint_override="", concepts="tree")
        assert a == b == backend.checkpoint_cache_token(None)
        _set_checkpoints(monkeypatch, CHECKPOINTS)


def test_core_checkpoint_label():
    assert backend.core_checkpoint_label("core:auto", "", "a.st") == "core:auto -> a.st"
    assert backend.core_checkpoint_label("core:auto", "a.st", "a.st") == "a.st"
    assert backend.core_checkpoint_label("a.st", None, "a.st") == "a.st"
