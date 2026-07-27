import os
from pathlib import Path

import pytest

from atlas_camera.core.intrinsics import build_intrinsics
from atlas_camera.core.schema import AtlasCamera, AtlasExtrinsics, AtlasSolve


ROOT = Path(__file__).resolve().parents[1]

_PYTEST_TEMPROOT = Path(__file__).resolve().parents[1] / ".pytest_tmp"
_PYTEST_TEMPROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(_PYTEST_TEMPROOT))


@pytest.fixture()
def make_atlas_solve():
    def _factory(**kwargs):
        image_w = kwargs.get("image_width", 1920)
        image_h = kwargs.get("image_height", 1080)
        pos = kwargs.get("position", (0.0, 5.0, 10.0))
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        # Build a consistent world matrix (identity rotation + given position)
        # so exporters that read camera_world_matrix get the right data.
        world_matrix = (
            (1.0, 0.0, 0.0, px),
            (0.0, 1.0, 0.0, py),
            (0.0, 0.0, 1.0, pz),
            (0.0, 0.0, 0.0, 1.0),
        )
        return AtlasSolve(
            camera=AtlasCamera(
                intrinsics=build_intrinsics(
                    image_width=image_w,
                    image_height=image_h,
                    focal_length_mm=kwargs.get("focal", 35.0),
                    sensor_width_mm=kwargs.get("sensor_w", 36.0),
                    principal_point_px=kwargs.get("principal_point_px"),
                ),
                extrinsics=AtlasExtrinsics(
                    camera_position=pos,
                    camera_world_matrix=world_matrix,
                ),
            ),
            image_width=image_w,
            image_height=image_h,
        )
    return _factory


@pytest.fixture()
def synthetic_perspective_image(tmp_path):
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")

    width = 160
    height = 96
    image = np.zeros((height, width, 3), dtype=np.uint8)

    left_vp = (-80.0, 48.0)
    right_vp = (240.0, 48.0)

    for target_y in (12.0, 26.0, 38.0):
        slope = (target_y - left_vp[1]) / ((width - 1) - left_vp[0])
        y_at_0 = int(round(left_vp[1] + slope * (0 - left_vp[0])))
        cv2.line(image, (0, y_at_0), (width - 1, int(target_y)), (255, 255, 255), 2)

    for start_y in (12.0, 26.0, 38.0):
        slope = (right_vp[1] - start_y) / (right_vp[0] - 0)
        y_at_width = int(round(start_y + slope * (width - 1)))
        cv2.line(image, (0, int(start_y)), (width - 1, y_at_width), (255, 255, 255), 2)

    path = tmp_path / "synthetic_perspective.png"
    assert cv2.imwrite(str(path), image)
    return path, image


# ---------------------------------------------------------------------------
# Personal working copies in examples/
# ---------------------------------------------------------------------------
# Artists keep hand-tuned variants of a shipped graph next to it — the working
# copy of atlas_path_guided_hole_repair_workflow.json, a "_02" retry, a
# "_Neat" cleanup. They are untracked and must NEVER be graded as shipping
# workflows: they are authored against whatever branch was checked out, so
# they carry widget counts and node sets that do not match this one, and they
# would fail the pinned-name, unique-id and widget-drift sweeps for reasons
# that say nothing about the repo.
#
# Convention: a stem containing "-edit" (or living under examples/local/) is a
# personal file. tools/audit_node_usage.py and tools/migrate_legacy_nodes.py
# apply the same rule, so evidence-gathering and migration skip them too.
LOCAL_WORKFLOW_MARKERS = ("-edit",)


def is_local_workflow(path) -> bool:
    """True for an artist's personal working copy, not a shipping workflow.

    The "local" directory test is applied to the path RELATIVE TO THE REPO,
    never to the absolute path. Matching absolute parts made the checkout's
    own location part of the predicate: a worktree under
    ``C:\\Users\\...\\AppData\\Local\\Temp`` — or any checkout beneath
    ``~/.local/share`` — classified EVERY shipping workflow as an artist copy,
    so the workflow suites silently discovered nothing and passed on an empty
    set. Found 2026-07-28 running the suite from a temp worktree: 18 tests
    "failed" for reasons that had nothing to do with the change under test.
    """
    p = Path(path)
    if any(m in p.stem for m in LOCAL_WORKFLOW_MARKERS):
        return True
    try:
        parts = Path(p).resolve().relative_to(ROOT.resolve()).parts
    except ValueError:      # outside the repo (a tmp_path fixture) — judge by
        parts = p.parts     # the whole path, since there is no repo anchor
    return "local" in {part.lower() for part in parts}
