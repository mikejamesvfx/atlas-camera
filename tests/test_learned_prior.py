"""Tests for the learned single-image camera prior integration.

The numpy solve path (CameraPrior -> AtlasSolve) is tested without torch. The
end-to-end GeoCalib inference is tested only when the [neural] extra is present.
"""

import math

import pytest

np = pytest.importorskip("numpy")

from atlas_camera.core.solver import _rotation_from_up_vector, solve_from_learned_prior
from atlas_camera.inference.learned_prior import CameraPrior


def _prior(pitch_deg=0.0, roll_deg=0.0, focal_px=700.0, size=(1024, 1024)):
    """A CameraPrior with a gravity/up vector consistent with the given pitch/roll."""
    p = math.radians(pitch_deg)
    # World up in an Atlas camera pitched by `p` about camera X. Matches GeoCalib's
    # observed convention (looking down, pitch<0, gives up_z>0): up = [0, cos p, -sin p].
    up = (0.0, math.cos(p), -math.sin(p))
    return CameraPrior(
        focal_px=focal_px,
        fov_h_deg=2 * math.degrees(math.atan(size[0] / 2 / focal_px)),
        fov_v_deg=2 * math.degrees(math.atan(size[1] / 2 / focal_px)),
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        up_cam=up,
        principal_point_px=(size[0] / 2, size[1] / 2),
        image_width=size[0],
        image_height=size[1],
        roll_uncertainty_deg=1.0,
        pitch_uncertainty_deg=1.5,
        focal_uncertainty_px=20.0,
    )


def test_rotation_from_up_vector_is_right_handed_and_y_up():
    R = _rotation_from_up_vector((0.0, 1.0, 0.0))
    assert abs(np.linalg.det(R) - 1.0) < 1e-6
    # World +Y maps to camera +Y (the up column).
    assert R[:, 1] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)


def test_learned_solve_is_y_up_with_positive_camera_height():
    solve = solve_from_learned_prior(_prior(pitch_deg=-5.0), camera_height=1.6)
    extr = solve.camera.extrinsics
    assert extr.up_axis == "Y"
    assert extr.camera_position[1] == pytest.approx(1.6)
    assert solve.camera.intrinsics.fx_px and solve.camera.intrinsics.fx_px > 0
    assert solve.source_method.startswith("automatic_still_image_learned_prior")


def test_learned_solve_confidence_reflects_uncertainty():
    confident = solve_from_learned_prior(_prior())  # low uncertainty
    prior_bad = _prior()
    prior_bad.pitch_uncertainty_deg = 14.0
    prior_bad.focal_uncertainty_px = 600.0
    shaky = solve_from_learned_prior(prior_bad)
    assert confident.confidence > shaky.confidence


def test_learned_solve_ground_plane_faces_camera_when_looking_down():
    # Camera looking slightly down should intersect the Y=0 ground plane below center.
    solve = solve_from_learned_prior(_prior(pitch_deg=-8.0), camera_height=1.6)
    vm = np.array(solve.camera.extrinsics.camera_view_matrix, dtype=np.float64)
    forward_world = np.linalg.inv(vm)[:3, :3] @ np.array([0.0, 0.0, -1.0])
    assert forward_world[1] < 0.0  # forward ray points downward


def test_estimate_camera_prior_end_to_end_if_available(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("geocalib")
    from PIL import Image

    from atlas_camera.inference.learned_prior import estimate_camera_prior

    # A plain gradient image is enough to exercise the code path end-to-end.
    Image.new("RGB", (640, 480), (120, 120, 120)).save(tmp_path / "img.png")
    prior = estimate_camera_prior(tmp_path / "img.png", device="cpu")
    assert prior.focal_px > 0
    assert len(prior.up_cam) == 3
    solve = solve_from_learned_prior(prior, image_size=(640, 480))
    assert solve.camera.intrinsics.fx_px > 0


# --- reproducibility of the GeoCalib call -----------------------------------
# GeoCalib's calibrate() draws from torch's global RNG, so the same image used
# to solve to a different camera each run (measured: ~2.5% focal spread, ~0.7deg
# roll, on both CPU and CUDA). estimate_camera_prior seeds inside a fork_rng so
# the solve is reproducible WITHOUT clobbering the caller's RNG - it runs inside
# ComfyUI, where a global reseed would silently change a later sampler.

def test_calibrate_is_seeded_and_leaves_the_callers_rng_untouched(monkeypatch):
    torch = pytest.importorskip("torch")
    from atlas_camera.inference import learned_prior as lp

    seen = []

    class _FakeModel:
        def __init__(self):
            self.training = True
            self.eval_calls = 0

        def eval(self):
            self.eval_calls += 1
            self.training = False
            return self

        def to(self, _device):
            return self

        def load_image(self, _path):
            return torch.zeros(1)

        def calibrate(self, _image):
            # Record what the RNG produces INSIDE the call: a seeded fork means
            # every invocation observes the same draw.
            seen.append(torch.randn(1).item())
            raise _Stop

    class _Stop(Exception):
        pass

    model = _FakeModel()
    monkeypatch.setattr(lp, "_get_model", lambda *_a, **_k: model)
    monkeypatch.setattr(lp, "_require_geocalib", lambda: (torch, object))
    monkeypatch.setattr(lp, "resolve_device", lambda *_a, **_k: "cpu")

    torch.manual_seed(1234)
    caller_before = torch.randn(3).tolist()

    torch.manual_seed(1234)
    for _ in range(3):
        with pytest.raises(_Stop):
            lp.estimate_camera_prior("unused.png")
    caller_after = torch.randn(3).tolist()

    assert len(set(seen)) == 1, f"calibrate saw a different RNG draw per run: {seen}"
    assert caller_before == caller_after, "estimate_camera_prior clobbered the caller's RNG"


def test_model_is_put_in_eval_mode():
    """GeoCalib ships the module in its default training=True state (21 Dropout
    + 47 BatchNorm submodules) and nothing here ever trains it."""
    pytest.importorskip("torch")
    pytest.importorskip("geocalib")
    from atlas_camera.inference import learned_prior as lp

    model = lp._get_model("pinhole", "cpu")
    assert model.training is False
