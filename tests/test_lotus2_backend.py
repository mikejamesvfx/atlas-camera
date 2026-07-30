"""Contract for the Lotus-2 depth backend.

Lotus-2 (arXiv 2512.01030) is a FLUX.1-dev backbone LoRA-finetuned for dense
geometry. See docs/dev/flux_depth_finetune_analysis.md for why it is here and why
FLUX.1-Depth-dev (a depth-CONDITIONED generator) is not the same thing.

WHAT IS AND IS NOT VERIFIED. Everything here runs without model weights:
routing, the licence-aware error, cache keying, and the disparity conversion. The
backend has NOT been run against real Lotus-2 weights — that needs
black-forest-labs/FLUX.1-dev in diffusers layout (~24 GB, gated, non-commercial),
which is not present on the dev machine. The one thing weights would settle is
the POLARITY assumption, pinned below so a live run can confirm or flip it in one
place.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

import atlas_camera.inference.depth_estimator as de  # noqa: E402
from atlas_camera.comfy.node_helpers import _DEPTH_MODEL_CHOICES  # noqa: E402


class TestRouting:
    def test_lotus2_id_routes_to_the_lotus2_backend(self):
        assert de._is_lotus2_model("jingheya/Lotus-2")

    @pytest.mark.parametrize("model_id", [
        "jingheya/lotus-depth-g-v2-1-disparity",
        "jingheya/lotus-depth-d-v1-1",
        "jingheya/lotus-normal-g-v1-1",
    ])
    def test_the_older_sd_based_lotus_family_does_NOT_route_here(self, model_id):
        """A loose `"lotus" in id` test would silently send these to a loader
        built for a different architecture. Lotus v1/v2 are SD-based; Lotus-2 is
        FLUX-based, with a different pipeline entirely."""
        assert not de._is_lotus2_model(model_id)

    @pytest.mark.parametrize("model_id", [
        "apple/DepthPro-hf", "Ruicheng/moge-2-vitl-normal",
        "depth-anything/DA3METRIC-LARGE",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    ])
    def test_no_other_backend_is_captured(self, model_id):
        assert not de._is_lotus2_model(model_id)

    def test_matching_is_case_insensitive(self):
        assert de._is_lotus2_model("JINGHEYA/LOTUS-2")


class TestComboContract:
    def test_it_is_appended_LAST(self):
        """Combo values serialize into saved workflows — append only, never
        insert or reorder."""
        assert _DEPTH_MODEL_CHOICES[-1] == "jingheya/Lotus-2"

    def test_the_previous_tail_is_undisturbed(self):
        assert _DEPTH_MODEL_CHOICES[-2] == "apple/DepthPro-hf"

    def test_the_id_matches_the_real_published_repo(self):
        """Verified against both the Lotus-2 README and the demo Space's linked
        models. A wrong id here can never be removed — only appended past."""
        assert "jingheya/Lotus-2" in _DEPTH_MODEL_CHOICES


class TestMissingCloneIsExplained:
    def test_it_names_the_repo_the_env_var_and_the_licence(self, monkeypatch):
        monkeypatch.delenv(de.LOTUS2_PATH_ENV, raising=False)
        with pytest.raises(RuntimeError) as exc:
            de._resolve_lotus2_root("")
        msg = str(exc.value)
        assert "EnVision-Research/Lotus-2" in msg
        assert de.LOTUS2_PATH_ENV in msg
        assert "NON-COMMERCIAL" in msg, (
            "FLUX.1-dev's licence is the reason this backend is opt-in; a user "
            "must not discover it after a 24 GB download")
        assert de.LOTUS2_FLUX_BASE in msg

    def test_the_env_var_is_honoured(self, tmp_path, monkeypatch):
        (tmp_path / "pipeline.py").write_text("", encoding="utf-8")
        (tmp_path / "infer.py").write_text("", encoding="utf-8")
        monkeypatch.setenv(de.LOTUS2_PATH_ENV, str(tmp_path))
        assert de._resolve_lotus2_root("") == tmp_path

    def test_an_explicit_path_beats_the_env_var(self, tmp_path, monkeypatch):
        good = tmp_path / "explicit"
        good.mkdir()
        for n in ("pipeline.py", "infer.py"):
            (good / n).write_text("", encoding="utf-8")
        monkeypatch.setenv(de.LOTUS2_PATH_ENV, str(tmp_path / "nonexistent"))
        assert de._resolve_lotus2_root(str(good)) == good

    def test_a_clone_missing_its_repo_local_modules_is_rejected(self, tmp_path):
        """`from pipeline import Lotus2Pipeline` is a REPO-LOCAL import, so a
        directory that merely exists is not a usable clone."""
        (tmp_path / "pipeline.py").write_text("", encoding="utf-8")   # no infer.py
        with pytest.raises(RuntimeError, match="incomplete.*infer.py"):
            de._resolve_lotus2_root(str(tmp_path))


class TestPolarity:
    """The one thing that cannot be settled without weights, so it is pinned.

    Lotus-2 emits an affine-invariant map where LARGER = NEARER: its v1 siblings
    are named `*-disparity`, and upstream colourises with `reverse_color=True`.
    Atlas's DepthResult contract is the opposite — forward distance, larger =
    farther. If a live Lotus-2 render ever comes back inside-out, the assumption
    recorded here and in metadata["polarity_assumption"] is the single place to
    flip.
    """

    def test_the_shared_reciprocal_conversion_is_reused_not_reimplemented(self):
        """A linear `1 - d` flip is rank-preserving and looks fine while warping
        near/far spacing — the exact bug _disparity_to_depth was extracted to
        prevent. Lotus-2 must go through it, so its behaviour is pinned here."""
        disparity = np.array([[1.0, 0.5], [0.25, 0.0]], dtype=np.float32)
        depth, meta = de._disparity_to_depth(disparity, {})
        # Nearest (max disparity) must map to the SMALLEST distance.
        assert depth[0, 0] == pytest.approx(0.0)
        assert depth[1, 1] == pytest.approx(1.0)
        assert depth[0, 1] < depth[1, 0], "monotonic: less disparity = farther"
        assert meta["disparity_floor"] == de._DISPARITY_FLOOR

    def test_spacing_is_reciprocal_not_linear(self):
        """Guards against someone 'simplifying' the conversion to `1 - d`."""
        disparity = np.array([[1.0, 0.5, 0.0]], dtype=np.float32)
        depth, _ = de._disparity_to_depth(disparity, {})
        linear_midpoint = 0.5
        assert depth[0, 1] != pytest.approx(linear_midpoint, abs=0.05), (
            "a reciprocal conversion must NOT place half-disparity at half-depth")


class TestCacheKeying:
    def test_the_clone_path_joins_the_result_cache_key(self):
        """Two clones (e.g. a patched one) must not collide.

        Same failure the moge_key comment records: a knob that appears to do
        nothing because the first call's map is returned.
        """
        import inspect
        src = inspect.getsource(de.estimate_depth)
        assert "lotus_key" in src
        assert "lotus_key" in src.split("cache_key = ")[1].split("\n")[0]


class TestDeclaredMetadata:
    def test_the_module_declares_relative_not_metric(self):
        """Lotus-2 is affine-invariant. Declaring it metric would let downstream
        consumers trust an arbitrary unit as metres."""
        import inspect
        src = inspect.getsource(de._estimate_depth_lotus2)
        assert "is_metric=False" in src

    def test_it_takes_the_raw_array_not_the_colourised_visualisation(self):
        """process_single_image returns (image, output_vis, output_npy). The
        SECOND is a colourmap; using it would look plausible and be meaningless.
        """
        import inspect
        src = inspect.getsource(de._estimate_depth_lotus2)
        assert "_, _, output_npy = process_single_image" in src
