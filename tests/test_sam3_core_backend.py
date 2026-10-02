"""AtlasSAM3Mask on ComfyUI's own SAM3 (sam3.1_multiplex_fp16), via stubbed core modules."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from atlas_camera.comfy import sam3_core_backend as backend  # noqa: E402

H, W = 32, 48


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
    fp.get_filename_list = lambda kind: ["sam3.1_multiplex_fp16.safetensors",
                                         "sam3d_body.safetensors", "sdxl.safetensors"]
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
    backend._LOADED.clear()
    _FakeDetect.calls.clear()
    yield clip
    backend._LOADED.clear()


def test_choices_offer_hf_default_then_sam3_checkpoints(fake_comfy):
    assert backend.sam3_checkpoint_choices() == [
        "hf:facebook/sam3", "sam3.1_multiplex_fp16.safetensors"]   # sam3d excluded
    assert backend.core_sam3_available()


def test_choices_outside_comfy_are_hf_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", None)
    assert backend.sam3_checkpoint_choices() == ["hf:facebook/sam3"]


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
        sam3_checkpoint="sam3.1_multiplex_fp16.safetensors")
    assert stack.shape == (1, H, W) and float(stack.sum()) == 20 * 30
    assert "sam3.1_multiplex_fp16" in report and "largest first" in report


def test_node_merged_mode_via_core(fake_comfy):
    mask, report = _node().segment(
        torch.rand(1, H, W, 3), "machine",
        sam3_checkpoint="sam3.1_multiplex_fp16.safetensors")
    assert mask.shape == (1, H, W) and float(mask.sum()) == 8 * 8 + 20 * 30
    assert "of frame" in report


def test_node_no_match_is_empty_not_error(fake_comfy):
    mask, report = _node().segment(
        torch.rand(1, H, W, 3), "giraffe",
        sam3_checkpoint="sam3.1_multiplex_fp16.safetensors")
    assert float(mask.sum()) == 0 and "NO MATCH" in report


def test_node_core_failure_is_reported(fake_comfy, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad checkpoint")
    monkeypatch.setattr(sys.modules["comfy.sd"], "load_checkpoint_guess_config", boom)
    mask, report = _node().segment(
        torch.rand(1, H, W, 3), "machine",
        sam3_checkpoint="sam3.1_multiplex_fp16.safetensors")
    assert float(mask.sum()) == 0 and "FAILED" in report and "bad checkpoint" in report


def test_widget_is_appended_last_with_hf_default():
    from atlas_camera.comfy.nodes_inpaint import AtlasSAM3Mask
    from atlas_camera.comfy.nodes_viewport import AtlasInput
    for cls in (AtlasSAM3Mask, AtlasInput):
        opt = cls.INPUT_TYPES()["optional"]
        assert list(opt)[-1] == "sam3_checkpoint"
        assert opt["sam3_checkpoint"][1]["default"] == "hf:facebook/sam3"
        assert opt["sam3_checkpoint"][0][0] == "hf:facebook/sam3"


def test_cascade_passes_checkpoint_and_counts_core_as_native():
    from atlas_camera.comfy.node_helpers import _MiniGraphBuilder, build_segmentation_cascade

    g = _MiniGraphBuilder()
    ref, path = build_segmentation_cascade(
        g, "img", "sky", policy="semantic", have_native_sam3=False, registry={},
        sam3_checkpoint="sam3.1_multiplex_fp16.safetensors")
    assert ref is not None and "core sam3.1_multiplex_fp16" in path
    node = g.finalize() if hasattr(g, "finalize") else None
    assert node is None or any(
        v.get("inputs", {}).get("sam3_checkpoint") == "sam3.1_multiplex_fp16.safetensors"
        for v in node.values())
