"""Patch-camera registration (generated -> measured), synthetic truth.

Umeyama + RANSAC are pinned on random clouds with known (s,R,t) and outliers;
the end-to-end `register_patch_camera` is pinned by RENDERING the same textured
scene from two known cameras (so SIFT has something real to match), building
the primary's world pointmap and the patch's OpenCV pointmap from the truth
depth, and checking that the recovered pose lands on the truth camera.
"""
from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from atlas_camera.core.camera_math import look_at_view_matrix  # noqa: E402
from atlas_camera.core.depth_geometry import back_project_normals  # noqa: E402
from atlas_camera.core.patch_camera_registration import (  # noqa: E402
    RegistrationConfig, ransac_similarity, register_patch_camera, umeyama_similarity,
)


def _rot(axis, deg):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    x, y, z = a
    return np.array([[c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
                     [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
                     [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]])


class TestUmeyama:
    def test_recovers_known_similarity(self):
        rng = np.random.default_rng(1)
        src = rng.normal(size=(50, 3))
        R = _rot([0.3, 1.0, 0.2], 37.0); s = 2.4; t = np.array([1.0, -2.0, 0.5])
        dst = s * src @ R.T + t + rng.normal(scale=1e-4, size=src.shape)
        s2, R2, t2 = umeyama_similarity(src, dst)
        assert s2 == pytest.approx(s, rel=1e-3)
        assert np.allclose(R2, R, atol=1e-3)
        assert np.allclose(t2, t, atol=2e-3)
        assert np.linalg.det(R2) == pytest.approx(1.0)

    def test_reflection_is_never_returned(self):
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
        dst = src.copy(); dst[:, 0] *= -1        # mirrored
        _, R, _ = umeyama_similarity(src, dst)
        assert np.linalg.det(R) == pytest.approx(1.0)


class TestRansac:
    def test_survives_forty_percent_outliers_deterministically(self):
        rng = np.random.default_rng(7)
        src = rng.uniform(-5, 5, size=(200, 3))
        R = _rot([0, 1, 0], -25.0); s = 1.3; t = np.array([3.0, 0.2, -1.0])
        dst = s * src @ R.T + t
        bad = rng.choice(200, size=80, replace=False)
        dst[bad] += rng.uniform(-8, 8, size=(80, 3))
        r1 = ransac_similarity(src, dst, threshold_m=0.05, iters=400, seed=3)
        r2 = ransac_similarity(src, dst, threshold_m=0.05, iters=400, seed=3)
        assert r1 is not None and r1.n_inliers >= 115
        assert r1.scale == pytest.approx(s, rel=1e-3)
        assert np.allclose(r1.rotation, R, atol=1e-3)
        assert r1.rms_m < 1e-6
        assert np.array_equal(r1.inlier_mask, r2.inlier_mask)   # deterministic

    def test_no_consensus_returns_none(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(30, 3)); dst = rng.normal(size=(30, 3)) * 50
        assert ransac_similarity(src, dst, threshold_m=0.01, iters=100, min_inliers=10) is None


# ---------------------------------------------------------------------------
# End-to-end on a rendered synthetic scene

W, H = 640, 480
FX = 600.0


def _texture(rng, n=512):
    """A busy random-blob texture (uint8, n x n) so SIFT finds plenty."""
    img = np.zeros((n, n), np.float32)
    yy, xx = np.mgrid[0:n, 0:n]
    for _ in range(1500):
        cx, cy = rng.uniform(0, n, 2); r = rng.uniform(2, 14); v = rng.uniform(-1.0, 1.0)
        img += v * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    # Sharp corners too: random rectangles.
    for _ in range(300):
        x0, y0 = rng.integers(0, n - 20, 2); w_, h_ = rng.integers(4, 20, 2)
        img[y0:y0 + h_, x0:x0 + w_] += rng.uniform(-1.0, 1.0)
    img -= img.min(); img /= img.max()
    return (img * 255).astype(np.uint8)


def _render(view, tex, *, plane_z=-8.0, plane_half=6.0):
    """Ray-cast a textured vertical wall (z = plane_z, |x|,|y| <= plane_half)
    plus a textured ground (y = 0) from `view`. Returns (rgb HxWx3 uint8,
    depth HxW forward metres, NaN off-geometry)."""
    c2w = np.linalg.inv(view)
    R = c2w[:3, :3]; pos = c2w[:3, 3]
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    d_cam = np.stack([(uu - W / 2) / FX, -(vv - H / 2) / FX, -np.ones_like(uu)], -1)
    d = d_cam @ R.T
    depth = np.full((H, W), np.nan)
    rgb = np.zeros((H, W), np.uint8)
    n = tex.shape[0]
    with np.errstate(all="ignore"):
        # wall
        t = (plane_z - pos[2]) / d[..., 2]
        p = pos + t[..., None] * d
        hit = (t > 0) & (np.abs(p[..., 0]) <= plane_half) & (p[..., 1] >= 0) & (p[..., 1] <= 2 * plane_half)
        u = ((p[..., 0] + plane_half) / (2 * plane_half) * (n - 1)).astype(int)
        v = ((p[..., 1]) / (2 * plane_half) * (n - 1)).astype(int)
        depth = np.where(hit, t, depth)                       # t == forward metres (|d_cam.z|=1)
        rgb = np.where(hit, tex[np.clip(v, 0, n - 1), np.clip(u, 0, n - 1)], rgb)
        # ground (only where the wall was not hit closer)
        tg = (0.0 - pos[1]) / d[..., 1]
        pg = pos + tg[..., None] * d
        hitg = (tg > 0) & (np.abs(pg[..., 0]) <= plane_half) & (pg[..., 2] > plane_z) & (pg[..., 2] < 0)
        closer = hitg & (~hit | (tg < np.nan_to_num(t, nan=np.inf)))
        ug = ((pg[..., 0] + plane_half) / (2 * plane_half) * (n - 1)).astype(int)
        vg = ((-pg[..., 2] + plane_z) / (-plane_z) * (n - 1)).astype(int)
        depth = np.where(closer, tg, depth)
        rgb = np.where(closer, tex[np.clip(vg, 0, n - 1)[::-1] if False else np.clip(vg, 0, n - 1),
                                   np.clip(ug, 0, n - 1)], rgb)
    return np.repeat(rgb[..., None], 3, axis=-1), depth


def _cam(eye, target):
    view, _, _ = look_at_view_matrix(tuple(eye), tuple(target))
    return np.asarray(view, dtype=np.float64)


def _opencv_pointmap(depth):
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    return np.stack([(uu - W / 2) / FX * depth, (vv - H / 2) / FX * depth, depth], -1)


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(42)
    tex = _texture(rng)
    v_primary = _cam([0.0, 1.6, 0.0], [0.0, 1.5, -8.0])
    v_patch = _cam([3.5, 2.2, -1.5], [0.0, 1.5, -8.0])      # ~24° to the right, higher
    img_p, dep_p = _render(v_primary, tex)
    img_q, dep_q = _render(v_patch, tex)
    bp = back_project_normals(dep_p, view_matrix=v_primary, fx=FX, fy=FX, cx=W / 2, cy=H / 2)
    world_p = np.where(np.isfinite(dep_p)[..., None], bp.pts_world, np.nan)
    return {"v_primary": v_primary, "v_patch": v_patch, "img_p": img_p, "img_q": img_q,
            "dep_q": dep_q, "world_p": world_p}


def test_registers_a_known_second_camera(scene):
    K = {"fx": FX, "fy": FX, "cx": W / 2, "cy": H / 2}
    # MoGe-style pointmap for the patch, deliberately mis-scaled by 0.6 to
    # prove the similarity absorbs the monocular scale error.
    pts_q = _opencv_pointmap(scene["dep_q"] * 0.6)
    truth = scene["v_patch"]
    wrong = _cam([-3.5, 2.2, -1.5], [0.0, 1.5, -8.0])        # the mirrored guess
    reg = register_patch_camera(
        patch_image=scene["img_q"], primary_image=scene["img_p"],
        patch_points_cam=pts_q, primary_points_world=scene["world_p"],
        patch_intrinsics=K,
        declared_view_matrices={"noflip": wrong, "flip": truth},
        config=RegistrationConfig(min_inliers=20, max_residual_m=0.2,
                                  max_deviation_deg=15.0, seed=1))
    assert reg.accepted, reg.reason
    assert reg.scale == pytest.approx(1.0 / 0.6, rel=0.03)
    c2w_t = np.linalg.inv(truth); c2w_r = np.linalg.inv(reg.view_matrix)
    assert np.linalg.norm(c2w_t[:3, 3] - c2w_r[:3, 3]) < 0.15         # metres
    ang = math.degrees(math.acos(np.clip((np.trace(c2w_t[:3, :3].T @ c2w_r[:3, :3]) - 1) / 2, -1, 1)))
    assert ang < 2.0
    assert reg.flip_resolved is True and reg.deviation_deg < 2.0
    assert reg.reproj_rms_px < 3.0
    assert reg.summary()["registration_accepted"] is True


def test_deviation_from_declared_orbit_warns_on_a_strong_match_and_refuses_on_a_weak_one(scene):
    """The generator ignoring the requested angle is NOT a registration
    failure when the match is strong: the measurement wins and the
    disagreement is reported. A weak match still refuses (found live)."""
    K = {"fx": FX, "fy": FX, "cx": W / 2, "cy": H / 2}
    pts_q = _opencv_pointmap(scene["dep_q"])
    far_off = _cam([0.0, 30.0, -4.0], [0.0, 0.0, -8.0])       # top-down: 60°+ away
    reg = register_patch_camera(
        patch_image=scene["img_q"], primary_image=scene["img_p"],
        patch_points_cam=pts_q, primary_points_world=scene["world_p"],
        patch_intrinsics=K, declared_view_matrices={"noflip": far_off},
        config=RegistrationConfig(min_inliers=20, max_deviation_deg=15.0, seed=1))
    assert reg.accepted                             # strong: >= 2x inliers, small rms
    assert "deviates" in reg.reason and "measurement kept" in reg.reason
    assert reg.deviation_deg > 15.0
    # weak match (inlier floor set just under what we have): deviation refuses
    weak = register_patch_camera(
        patch_image=scene["img_q"], primary_image=scene["img_p"],
        patch_points_cam=pts_q, primary_points_world=scene["world_p"],
        patch_intrinsics=K, declared_view_matrices={"noflip": far_off},
        config=RegistrationConfig(min_inliers=max(20, reg.n_inliers - 5), max_deviation_deg=15.0, seed=1))
    assert not weak.accepted and "deviates" in weak.reason
    assert weak.view_matrix is not None          # diagnostics still carried
    assert weak.summary()["registration_accepted"] is False


def test_rejects_when_there_is_nothing_to_match(scene):
    K = {"fx": FX, "fy": FX, "cx": W / 2, "cy": H / 2}
    flat = np.full_like(scene["img_q"], 128)
    reg = register_patch_camera(
        patch_image=flat, primary_image=scene["img_p"],
        patch_points_cam=_opencv_pointmap(scene["dep_q"]),
        primary_points_world=scene["world_p"], patch_intrinsics=K)
    assert not reg.accepted and reg.n_inliers == 0


# --- planar consensus sets: what IS and IS NOT degenerate ------------------
# Atlas patches facades, so a near-planar inlier set is the common case, not a
# corner case. These pin both halves of the MIRROR note in ransac_similarity.

def _planar_pair(s_true=1.7, theta=0.6, noise=0.002, n=200, seed=0):
    rng = np.random.default_rng(seed)
    src = np.column_stack([rng.uniform(-5, 5, n), rng.uniform(0, 8, n), np.zeros(n)])
    R = np.array([[math.cos(theta), 0.0, math.sin(theta)],
                  [0.0, 1.0, 0.0],
                  [-math.sin(theta), 0.0, math.cos(theta)]])
    dst = s_true * src @ R.T + np.array([2.0, 0.5, -30.0])
    return src, dst + rng.normal(scale=noise, size=dst.shape)


def test_coplanar_correspondences_do_not_lose_scale():
    """The degeneracy people expect here does NOT exist: 3D<->3D similarity
    fixes scale from in-plane distances, so a facade registers cleanly."""
    src, dst = _planar_pair(s_true=1.7)
    fit = ransac_similarity(src, dst, threshold_m=0.05, min_inliers=12, seed=0)
    assert fit is not None
    assert fit.scale == pytest.approx(1.7, rel=1e-3)
    assert fit.n_inliers == len(src)


def test_a_mirrored_planar_pointmap_is_indistinguishable_here():
    """CHARACTERIZATION, not a wish: a patch pointmap reflected about the
    facade plane fits with the same scale, RMS and inlier count, via a PROPER
    rotation — so Umeyama's reflection handling cannot flag it and neither can
    this function. `RegistrationConfig.max_deviation_deg` is what refuses it
    downstream. If this test ever starts failing, the fit gained the ability
    to tell them apart and the MIRROR note should be revised.
    """
    src, dst = _planar_pair()
    mirrored = src.copy()
    mirrored[:, 2] *= -1.0

    direct = ransac_similarity(src, dst, threshold_m=0.05, min_inliers=12, seed=0)
    mirror = ransac_similarity(mirrored, dst, threshold_m=0.05, min_inliers=12, seed=0)
    assert direct is not None and mirror is not None
    assert mirror.scale == pytest.approx(direct.scale, rel=1e-6)
    assert mirror.rms_m == pytest.approx(direct.rms_m, rel=1e-6)
    assert mirror.n_inliers == direct.n_inliers
    assert np.linalg.det(mirror.rotation) == pytest.approx(1.0, abs=1e-6)
