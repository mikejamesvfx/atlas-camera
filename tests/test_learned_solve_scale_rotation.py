"""The learned solve fed both metric-scale tiers a TRANSPOSED rotation.

`solve_still_image_learned` binds `rotation = _rotation_from_up_vector(...)`,
and builds the final camera as
``camera_view_matrix = _matrix4_with_rotation_translation(rotation.T, ...)`` —
so that variable is CAM->WORLD. But both scale tiers document the opposite:

    estimate_ground_height_from_depth: "``rotation`` is the world->cam matrix"
    metric_height_from_reference:      transposes internally to get cam->world

so both were handed the inverse of what they asked for.

A transpose is INVISIBLE for a level camera — R ≈ Rᵀ when the pitch is near
zero — which is why the suite never caught it. It bites in proportion to pitch,
which is exactly the long-standing symptom: `scale_source=assumed_default`
firing on elevated vantages and vistas while level shots solve fine.

Measured on DSC_2328.NEF (Manhattan birdseye, 21.3° down): with the
view-matrix block a counted-facade reference solves the camera at 34.0 m; with
the transpose it is rejected as "reference base is above the horizon".

These tests use an exact analytic ground plane, so the expected height is known
in closed form rather than asserted against a previous run.
"""
import numpy as np
import pytest

from atlas_camera.core import solver

W, H_PX = 640, 480
FX = FY = 500.0
CX, CY = W / 2.0, H_PX / 2.0
CAMERA_HEIGHT = 10.0
PITCH_DEG = 25.0          # pitched DOWN; a level camera cannot see this bug


def _theta():
    return np.deg2rad(PITCH_DEG)


def _cam_to_world():
    """Camera pitched down by theta about world X."""
    c, s = np.cos(_theta()), np.sin(_theta())
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, s],
                     [0.0, -s, c]])


def _up_cam():
    """World +Y expressed in camera coords = column 1 of the world->cam matrix.

    Sanity: the real NYC plate measured up_cam = [0.0004, 0.9318, 0.3631],
    i.e. [0, cos(21.3°), sin(21.3°)] — the same form this produces.
    """
    return _cam_to_world().T @ np.array([0.0, 1.0, 0.0])


def _ground_depth_map():
    """Exact forward-distance map of a flat ground plane CAMERA_HEIGHT below.

    Back-projection is x=(u-cx)/fx*d, y=-(v-cy)/fy*d, z=-d, so requiring the
    world Y of that point to equal -CAMERA_HEIGHT gives
        d(v) = H / (cos(theta) * (v - cy) / fy + sin(theta))
    which is finite only below the horizon. Above it the ray never hits the
    ground and the depth is invalid.
    """
    c, s = np.cos(_theta()), np.sin(_theta())
    v = np.arange(H_PX, dtype=np.float64)[:, None]
    denom = c * (v - CY) / FY + s
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(denom > 1e-6, CAMERA_HEIGHT / denom, np.nan)
    return np.repeat(d, W, axis=1).astype(np.float32)


class _FakePrior:
    image_width, image_height = W, H_PX
    focal_px = FX
    pitch_deg = -PITCH_DEG
    roll_deg = 0.0
    principal_point_px = (CX, CY)
    roll_uncertainty_deg = 0.5
    pitch_uncertainty_deg = 0.5
    focal_uncertainty_px = 5.0
    k1 = 0.0
    fov_h_deg = 60.0
    fov_v_deg = 45.0
    source_model = "fake"
    raw = None

    def __init__(self):
        self.up_cam = tuple(float(x) for x in _up_cam())


class _FakeDepth:
    is_metric = True
    model_id = "fake"
    image_width, image_height = W, H_PX

    def __init__(self):
        self.depth = _ground_depth_map()
        finite = self.depth[np.isfinite(self.depth)]
        self.near, self.far = float(finite.min()), float(finite.max())
        self.normals = None
        self.metadata = {}

    def summary(self):
        return {"model_id": self.model_id, "is_metric": self.is_metric,
                "near": self.near, "far": self.far,
                "image_width": self.image_width,
                "image_height": self.image_height}


@pytest.fixture()
def learned(monkeypatch, tmp_path):
    """A learned solve whose prior and depth model are known exactly."""
    from PIL import Image

    from atlas_camera.inference import depth_estimator as de
    from atlas_camera.inference import learned_prior as lp

    monkeypatch.setattr(lp, "estimate_camera_prior", lambda *a, **k: _FakePrior())
    monkeypatch.setattr(de, "estimate_depth", lambda *a, **k: _FakeDepth())
    path = tmp_path / "plate.png"
    Image.new("RGB", (W, H_PX), (128, 128, 128)).save(path)
    return str(path)


def test_the_ground_fit_recovers_a_pitched_cameras_height(learned):
    """Tier 2. The depth map IS the ground plane, so a correct fit must return
    CAMERA_HEIGHT. Fed a transposed rotation the plane is unfindable and the
    solve silently falls back to the 1.6 m assumption."""
    solve = solver.solve_still_image_learned(learned, camera_height="auto")

    assert solve.camera.extrinsics.camera_position[1] == pytest.approx(
        CAMERA_HEIGHT, rel=0.05)
    assert (solve.debug_metadata or {}).get("scale_source") != "assumed_default"


def test_a_vertical_reference_is_not_rejected_above_the_horizon(learned):
    """Tier 1. A 3 m post standing on the ground, well below the horizon.

    Under the transpose its base ray points UP, so it is refused with
    "reference base is above the horizon" — the exact rejection measured on the
    real plate.
    """
    c, s = np.cos(_theta()), np.sin(_theta())
    v_base = 400.0
    d = CAMERA_HEIGHT / (c * (v_base - CY) / FY + s)
    base_cam = np.array([0.0, -(v_base - CY) / FY * d, -d])
    base_world = _cam_to_world() @ base_cam
    top_world = base_world + np.array([0.0, 3.0, 0.0])
    top_cam = _cam_to_world().T @ top_world
    v_top = CY - FY * top_cam[1] / (-top_cam[2])

    spec = {"bbox_px": [CX - 5.0, v_top, CX + 5.0, v_base],
            "height_m": 3.0, "confidence": 0.95, "label": "post"}
    solve = solver.solve_still_image_learned(
        learned, camera_height="auto", scale_references=[spec])

    detail = (solve.debug_metadata or {}).get("reference_scale") or {}
    reasons = [r.get("reason") for r in (detail.get("references") or [])]
    assert "reference base is above the horizon" not in reasons, reasons
    assert detail.get("camera_height_m") == pytest.approx(CAMERA_HEIGHT, rel=0.1)
    assert detail.get("adopted") is True
    assert (solve.debug_metadata or {}).get("scale_source") == "reference_object"
    assert solve.camera.extrinsics.camera_position[1] == pytest.approx(
        CAMERA_HEIGHT, rel=0.05)


def test_the_helper_and_the_view_matrix_stay_mutually_consistent():
    """Whatever convention `_rotation_from_up_vector` uses, the solve builds the
    view matrix as its TRANSPOSE. Pinning that relationship is what stops the
    two drifting apart again and silently re-inverting the scale tiers."""
    rot = np.asarray(solver._rotation_from_up_vector(_up_cam()), dtype=np.float64)
    world_to_cam = _cam_to_world().T
    assert np.allclose(rot, world_to_cam.T, atol=1e-6), (
        "the helper no longer returns cam->world; the scale-tier call sites "
        "transpose it and must be revisited")
