"""The benchmark harness must not report a run it did not measure.

Every bug pinned here produced a WRONG ANSWER rather than a crash, which is
the failure mode a scoreboard can least afford — a green row nobody re-checks.
All four were found by chasing three shipped workflows that the scoreboard
called "unscoreable":

  1. ComfyUI does not reject a graph whose node fails validation. It PRUNES
     that node and everything downstream, runs the rest, and reports
     status_str "success". atlas_layered_segmentation_workflow sat green with
     its viewport pruned by a missing cleanplate.png.
  2. PreviewImage writes to temp/, not output/. Resolving every reported image
     against output/ found nothing and scored the run as unmeasurable.
  3. Shipped workflows ship with their solve gates CLOSED (correct for a human
     opening one). Headless that pauses everything downstream, so both
     camera_staged_master workflows ran, succeeded, and measured nothing.
  4. The "why is this metric missing" messages were guesses, and the guesses
     were wrong: AtlasMoveBudget says "needs a relief mesh to seal" while the
     guard looked for "relief mesh is empty".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

wb = pytest.importorskip("workflow_benchmark")


class TestPrunedGraphs:
    """Fault 1 — the one that made every green row untrustworthy."""

    #: Verbatim from ComfyUI for atlas_layered_segmentation_workflow. Node 16 is
    #: the workflow's own viewport; 1017/1019 are the injected measurement tail.
    LIVE = {
        "2": {
            "class_type": "LoadImage",
            "errors": [{"type": "custom_validation_failed",
                        "message": "Custom validation failed for node",
                        "details": "image - Invalid image file: cleanplate.png"}],
            "dependent_outputs": ["1017", "1019", "16"],
        }
    }

    def test_a_pruned_graph_is_recorded_as_an_error_not_a_pass(self):
        got = wb.summarise_node_errors(self.LIVE)
        assert "error" in got, (
            "ComfyUI reports status_str 'success' for this run; if the harness "
            "does not treat it as a failure the scoreboard goes green on a "
            "graph whose output never executed")

    def test_it_names_the_outputs_that_were_pruned(self):
        assert wb.summarise_node_errors(self.LIVE)["pruned_outputs"] == \
            ["16", "1017", "1019"]

    def test_the_error_quotes_the_validation_detail(self):
        err = wb.summarise_node_errors(self.LIVE)["error"]
        assert "cleanplate.png" in err and "LoadImage" in err

    @pytest.mark.parametrize("empty", [None, {}])
    def test_a_clean_graph_is_not_flagged(self, empty):
        assert wb.summarise_node_errors(empty) == {}


class TestImageResolution:
    """Fault 2 — output/ and temp/ are siblings, not the same directory."""

    def test_a_preview_image_resolves_into_the_temp_directory(self, tmp_path):
        out = tmp_path / "output"
        got = wb._resolve_image(
            {"filename": "x.png", "subfolder": "", "type": "temp"}, out)
        assert got == tmp_path / "temp" / "x.png"

    def test_a_saved_image_resolves_into_the_output_directory(self, tmp_path):
        out = tmp_path / "output"
        got = wb._resolve_image(
            {"filename": "bench_1.png", "subfolder": "", "type": "output"}, out)
        assert got == out / "bench_1.png"

    def test_a_missing_type_is_treated_as_output(self, tmp_path):
        out = tmp_path / "output"
        assert wb._resolve_image({"filename": "x.png"}, out) == out / "x.png"

    def test_subfolders_are_honoured(self, tmp_path):
        out = tmp_path / "output"
        got = wb._resolve_image(
            {"filename": "x.png", "subfolder": "runs/a", "type": "output"}, out)
        assert got == out / "runs" / "a" / "x.png"

    def test_a_record_with_no_filename_resolves_to_nothing(self, tmp_path):
        assert wb._resolve_image({"type": "output"}, tmp_path) is None


class TestScoring:
    """score_images takes ComfyUI's records, not bare filename strings."""

    @staticmethod
    def _png(path: Path, value: int, size=(4, 4)):
        Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (value, value, value)).save(path)

    def test_it_scores_an_image_that_only_exists_in_temp(self, tmp_path):
        pytest.importorskip("numpy")
        out = tmp_path / "output"
        out.mkdir()
        self._png(tmp_path / "temp" / "prev.png", 255)
        scored = wb.score_images(
            [{"filename": "prev.png", "subfolder": "", "type": "temp"}], out)
        assert scored.get("non_black_frac") == 1.0, (
            "an image in temp/ must be scored, not silently skipped — this is "
            "the bug that made three workflows look unmeasurable")

    def test_the_injected_bench_render_wins_over_a_graphs_own_output(self, tmp_path):
        pytest.importorskip("numpy")
        out = tmp_path / "output"
        self._png(out / "bench_00001_.png", 0)      # injected: all black
        self._png(out / "artists_own.png", 255)     # graph's own: all white
        scored = wb.score_images([
            {"filename": "artists_own.png", "type": "output"},
            {"filename": "bench_00001_.png", "type": "output"},
        ], out)
        assert scored["scored_image"] == "bench_00001_.png", (
            "every workflow must be scored by the SAME instrument or the "
            "numbers are not comparable between workflows")

    def test_an_unreadable_record_scores_nothing_rather_than_guessing(self, tmp_path):
        pytest.importorskip("numpy")
        out = tmp_path / "output"
        out.mkdir()
        assert wb.score_images([{"filename": "gone.png", "type": "output"}], out) == {}


class TestKnownMissingAssets:
    def test_layered_segmentation_is_registered_as_needing_a_clean_plate(self):
        """It is not broken; it needs an asset that cannot ship in the repo.

        Without this it runs, gets partially pruned, and reports success.
        """
        assert (wb.KNOWN_MISSING_ASSETS.get("atlas_layered_segmentation_workflow")
                == "cleanplate.png")


class TestBudgetReporting:
    """Fault 4 — quote the node, do not guess at its wording."""

    ACTUAL = ("Move budget not computed: estimate_move_budget needs a relief "
              "mesh to seal — this solve has none.\nsecond line")

    def test_the_nodes_own_explanation_is_what_gets_reported(self):
        said = next((t for t in [self.ACTUAL] if "Move budget not computed" in t),
                    None)
        assert said is not None
        assert said.strip().splitlines()[0].startswith("Move budget not computed")

    def test_the_superseded_guard_would_have_missed_this_text(self):
        """Pins WHY the old guard was replaced, so it is not reinstated.

        It searched for "relief mesh is empty". The node says "needs a relief
        mesh to seal". Near-miss substring matching on another component's
        prose is the defect, not the particular string.
        """
        assert "relief mesh is empty" not in self.ACTUAL
        assert "relief mesh" in self.ACTUAL

    def test_a_real_envelope_report_still_parses(self):
        parsed = wb.parse_move_budget(
            "Safe camera envelope (disocclusion <= 2.0%):\n"
            "  dolly x +/-0.725 m\n"
            "  3.4% of frame already tears\n"
            "  46 faces fell behind the camera\n")
        assert parsed["dolly_m"] == 0.725
        assert parsed["already_tears_pct"] == 3.4
        assert parsed["dropped_faces"] == 46

    def test_a_search_capped_dolly_is_not_recorded_as_a_measurement(self):
        parsed = wb.parse_move_budget(
            "Safe camera envelope (disocclusion <= 2.0%):\n"
            "  dolly x +/-1359.373 m (unbounded within search cap)\n")
        assert "dolly_m" not in parsed
        assert parsed["dolly_unbounded"] is True
