"""Contract for the Atlas-side driver. Runs with NO Blender — subprocess faked.

The gate here measures something different from what the plan specified, and the
reason is the most useful thing this file records: after calibration,
distance-to-measured-surface is ZERO BY CONSTRUCTION for NEAREST_SURFACEPOINT
shrinkwrap (measured p95 0.0000m against a floor of exactly 0.0). A gate on that
number passes everything and catches nothing.

What it cannot see is a vertex slid ALONG the surface to the wrong place. So the
gate is on how far shrinkwrap had to DRAG the fill — a closure that landed near
the truth needs a small correction; one needing several edge lengths of hauling
was badly placed, and ending up on the surface does not redeem it.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.blender import organic_fill as of  # noqa: E402


class TestMedianEdge:
    def test_it_measures_the_scene_ruler(self):
        v = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        f = np.array([[0, 1, 2]])
        assert of.median_edge_length(v, f) == pytest.approx(2.0)

    def test_an_empty_mesh_has_no_ruler(self):
        assert of.median_edge_length(np.zeros((0, 3)), np.zeros((0, 3))) == 0.0


class TestMovementGate:
    def test_a_small_correction_is_accepted(self):
        ok, why = of.gate_movement(0.05, 0.2, median_edge_m=0.25)
        assert ok
        assert "0.20x median edge" in why, "the reason must carry the number"

    def test_a_large_drag_is_REJECTED_not_raised(self):
        """Rejection is a third outcome. A raise kills a 20-minute graph; a
        silent accept ships bad geometry."""
        ok, why = of.gate_movement(1.5, 3.0, median_edge_m=0.25)
        assert not ok
        assert "REJECTED" in why
        assert "6.0x" in why
        assert "Layer left unchanged" in why

    def test_the_reason_explains_why_on_surface_is_not_enough(self):
        """The subtlety a reader will otherwise miss: the result IS on the
        measured surface and is still being rejected."""
        _ok, why = of.gate_movement(1.5, 3.0, median_edge_m=0.25)
        assert "ON the measured surface" in why
        assert "slid" in why

    def test_acceptance_also_states_its_numbers(self):
        """A silent pass tells a reader nothing about how close the call was."""
        _ok, why = of.gate_movement(0.01, 0.02, median_edge_m=0.25)
        assert why and "accepted" in why

    def test_no_measurable_scale_refuses_rather_than_guessing(self):
        ok, why = of.gate_movement(0.1, 0.2, median_edge_m=0.0)
        assert not ok
        assert "no measurable edge" in why

    def test_the_limit_is_configurable_and_respected(self):
        assert of.gate_movement(1.0, 1.0, 0.25, max_move_scale=10.0)[0]
        assert not of.gate_movement(1.0, 1.0, 0.25, max_move_scale=1.0)[0]

    def test_the_gate_is_scale_free(self):
        """A tolerance tuned on a 10m interior must still mean something on a
        2km vista, so everything is expressed in median edge lengths."""
        small = of.gate_movement(0.05, 0.1, median_edge_m=0.25)[0]
        large = of.gate_movement(50.0, 100.0, median_edge_m=250.0)[0]
        assert small == large


class TestWeld:
    def test_rim_vertices_within_tolerance_are_paired(self):
        anchor = np.array([[0., 0., 0.], [1., 0., 0.]])
        new = np.array([[0.01, 0., 0.], [1.01, 0., 0.], [5., 5., 5.]])
        got = of.weld_to_anchor(new, anchor, tolerance_m=0.1)
        assert got["welded"] == 2
        assert got["unwelded"] == 0

    def test_an_unwelded_anchor_is_COUNTED_not_swallowed(self):
        """An unwelded rim vertex is a residual seam that boundary_edges will
        read as a fresh tear next pass. Silence would look like success."""
        anchor = np.array([[0., 0., 0.], [10., 0., 0.]])
        new = np.array([[0.01, 0., 0.]])
        got = of.weld_to_anchor(new, anchor, tolerance_m=0.1)
        assert got["welded"] == 1
        assert got["unwelded"] == 1

    def test_no_anchor_means_everything_is_unwelded(self):
        got = of.weld_to_anchor(np.zeros((3, 3)), np.zeros((0, 3)),
                                tolerance_m=1.0)
        assert got["welded"] == 0

    def test_pairs_index_into_both_arrays(self):
        anchor = np.array([[0., 0., 0.]])
        new = np.array([[9., 9., 9.], [0.01, 0., 0.]])
        pairs = of.weld_to_anchor(new, anchor, tolerance_m=0.1)["pairs"]
        assert pairs.shape == (1, 2)
        assert pairs[0, 0] == 1, "index into the NEW vertices"
        assert pairs[0, 1] == 0, "index into the ANCHOR vertices"


class TestFakeBlenderEndToEnd:
    """Drives shrinkwrap_patch with the subprocess stubbed — no Blender."""

    @staticmethod
    def _stub(monkeypatch, *, displace=0.0, moved_median=0.0):
        """Stand in for run_recipe: read in.npz, write a plausible out.npz."""
        def fake(recipe_name, exchange_dir, **kw):
            from pathlib import Path
            ex = Path(exchange_dir)
            with np.load(ex / "in.npz") as d:
                pv = np.asarray(d["patch_vertices"], dtype=np.float64)
                pf = np.asarray(d["patch_faces"], dtype=np.int64)
            out = pv + displace
            np.savez(ex / "out.npz", vertices=out,
                     faces=pf.astype(np.int32),
                     snapped=np.ones(len(out), dtype=bool))
            return {"recipe": recipe_name, "moved_median_m": moved_median,
                    "moved_max_m": moved_median * 2, "snapped_fraction": 1.0}
        monkeypatch.setattr(of, "run_recipe", fake)

    def _mesh(self):
        v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
        f = np.array([[0, 1, 2], [1, 3, 2]])
        return v, f

    def test_an_identity_result_round_trips_through_atlas_space(
            self, monkeypatch, tmp_path):
        """The conversion happens twice — in write_exchange, out in read_result
        — so an untouched patch must come back bit-identical."""
        self._stub(monkeypatch)
        v, f = self._mesh()
        got = of.shrinkwrap_patch(v, f, v, f, exchange_dir=tmp_path)
        np.testing.assert_allclose(got["vertices"], v, atol=1e-12)
        assert got["median_edge_m"] > 0

    def test_the_recipe_report_reaches_the_caller(self, monkeypatch, tmp_path):
        self._stub(monkeypatch, moved_median=0.02)
        v, f = self._mesh()
        got = of.shrinkwrap_patch(v, f, v, f, exchange_dir=tmp_path)
        assert got["report"]["moved_median_m"] == 0.02

    def test_a_wildly_displaced_result_is_gated_out(self, monkeypatch, tmp_path):
        """End to end: the stub returns geometry that needed a huge drag, and
        the gate refuses it."""
        self._stub(monkeypatch, displace=5.0, moved_median=5.0)
        v, f = self._mesh()
        got = of.shrinkwrap_patch(v, f, v, f, exchange_dir=tmp_path)
        ok, why = of.gate_movement(got["report"]["moved_median_m"],
                                   got["report"]["moved_max_m"],
                                   got["median_edge_m"])
        assert not ok and "REJECTED" in why

    def test_a_recipe_failure_propagates_rather_than_returning_junk(
            self, monkeypatch, tmp_path):
        def boom(*_a, **_k):
            raise RuntimeError("Blender exited 3")
        monkeypatch.setattr(of, "run_recipe", boom)
        v, f = self._mesh()
        with pytest.raises(RuntimeError, match="exited 3"):
            of.shrinkwrap_patch(v, f, v, f, exchange_dir=tmp_path)

    def test_the_limit_is_expressed_in_median_edges(self, monkeypatch, tmp_path):
        """A fixed metre limit would be meaningless across scene scales."""
        seen = {}

        def fake(recipe_name, exchange_dir, **kw):
            import json
            from pathlib import Path
            ex = Path(exchange_dir)
            seen.update(json.loads((ex / "params.json").read_text(
                encoding="utf-8")))
            with np.load(ex / "in.npz") as d:
                pv = np.asarray(d["patch_vertices"], dtype=np.float64)
                pf = np.asarray(d["patch_faces"], dtype=np.int64)
            np.savez(ex / "out.npz", vertices=pv, faces=pf.astype(np.int32),
                     snapped=np.ones(len(pv), dtype=bool))
            return {}
        monkeypatch.setattr(of, "run_recipe", fake)
        v, f = self._mesh()
        of.shrinkwrap_patch(v, f, v, f, limit_scale=4.0, exchange_dir=tmp_path)
        assert seen["shrinkwrap_limit_m"] == pytest.approx(
            4.0 * of.median_edge_length(v, f))
