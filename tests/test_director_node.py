import json

import numpy as np
import pytest

from atlas_camera.comfy.director_session import SESSIONS
from atlas_camera.comfy.nodes_director import AtlasDirectorTake, StaleTakeError


@pytest.fixture()
def take_dir(tmp_path):
    """A committed take with four frames, marked to two.

    Non-square resolution (16x9) throughout: the ray-map shape assertions use
    it so a width/height transposition -- the most likely defect in this
    code path -- cannot hide behind a square fixture (addendum Ruling P2).
    """

    directory = tmp_path / "takes" / "sc" / "sh" / "a_take01"
    (directory / "playblast").mkdir(parents=True)
    for index in range(4):
        (directory / "playblast" / f"frame.{index:04d}.png").write_bytes(b"")
    samples = [
        {"position": [0.0, 1.6, float(index)], "rotation": [0, 0, 0, 1],
         "focalLengthMm": 35.0, "focusDistanceM": 4.0, "tStop": 2.8,
         "filmback": {"name": "S35", "widthMm": 24.89, "heightMm": 18.66}}
        for index in range(4)
    ]
    (directory / "samples.json").write_text(json.dumps(samples))
    (directory / "manifest.json").write_text(json.dumps({
        "schemaVersion": 1, "slate": "sc/sh/a_take01", "frameCount": 999,
    }))
    return directory


@pytest.fixture()
def package_path(tmp_path):
    """A stand-in .atlas session package, whose bytes `check_fresh` digests."""

    path = tmp_path / "scenes" / "shot_012.atlas"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fake atlas package bytes")
    return path


def test_frame_count_comes_from_the_directory_not_the_manifest(take_dir):
    # manifest.frameCount says 999 on purpose. A playblast is a marked range;
    # reading the manifest would claim frames that were never rendered.
    node = AtlasDirectorTake()
    rays = node.read_rays(str(take_dir), width=16, height=9)
    assert rays.shape[0] == 4


def test_rays_have_the_ray_map_shape(take_dir):
    node = AtlasDirectorTake()
    rays = node.read_rays(str(take_dir), width=16, height=9)
    assert rays.shape == (4, 9, 16, 6)


def test_plucker_embedding_has_6_channels(take_dir):
    node = AtlasDirectorTake()
    rays = node.read_rays(str(take_dir), width=16, height=9)
    embedded = node.embed_rays(rays)
    assert embedded.shape == (4, 9, 16, 6)


def test_rays_preview_is_a_3_channel_image_in_0_1(take_dir):
    node = AtlasDirectorTake()
    rays = node.read_rays(str(take_dir), width=16, height=9)
    preview = node.rays_to_preview(rays)
    assert preview.shape == (4, 9, 16, 3)
    assert float(preview.min()) >= 0.0
    assert float(preview.max()) <= 1.0


def test_a_stale_playblast_refuses_and_names_the_fix(package_path):
    # Addendum Ruling P8: freshness compares the PACKAGE digest recorded at
    # launch, never a samples digest -- the node has no scene document.
    node = AtlasDirectorTake()
    with pytest.raises(StaleTakeError) as raised:
        node.check_fresh(str(package_path), recorded_digest="not-the-digest",
                          slate="sc/sh/a_take01")
    message = str(raised.value)
    assert "sc/sh/a_take01" in message
    assert "re-push" in message.lower()


def test_a_matching_digest_passes(package_path):
    node = AtlasDirectorTake()
    digest = node.package_digest(str(package_path))
    assert node.check_fresh(str(package_path), recorded_digest=digest,
                             slate="sc/sh/a_take01") is True


def test_an_unpushed_session_says_so_rather_than_rendering_something_else():
    SESSIONS.clear()
    SESSIONS["shot_012"] = {"session_id": "shot_012", "slate": None, "take_dir": None}
    node = AtlasDirectorTake()
    with pytest.raises(ValueError, match="no take pushed"):
        node.resolve("shot_012")
    SESSIONS.clear()


def test_allowed_frames_is_a_single_entry_tuple_for_now():
    from atlas_camera.comfy.nodes_director import ALLOWED_FRAMES

    assert ALLOWED_FRAMES == ("121",)


# --- the ray-map EXR sidecar (Ruling P7) ------------------------------------


def test_write_ray_exr_uses_float_bit_depth_and_records_channel_naming(take_dir, monkeypatch):
    node = AtlasDirectorTake()
    rays = node.read_rays(str(take_dir), width=16, height=9)
    embedded = node.embed_rays(rays)

    calls = []

    def fake_write_exr(path, pixels, *, bit_depth, **kwargs):
        calls.append((path, bit_depth))

    import atlas_camera.plate.oiio_io as oiio_io
    monkeypatch.setattr(oiio_io, "oiio_available", lambda: True)
    monkeypatch.setattr(oiio_io, "write_exr", fake_write_exr)

    out = node.write_ray_exr(str(take_dir), embedded)
    assert out
    assert len(calls) == embedded.shape[0]
    assert all(bit_depth == "float" for _path, bit_depth in calls)

    manifest = json.loads((take_dir / "manifest.json").read_text())
    assert manifest["rayChannels"] == ["O.X", "O.Y", "O.Z", "D.X", "D.Y", "D.Z"]


def test_write_ray_exr_skips_without_failing_when_oiio_unavailable(take_dir, monkeypatch):
    node = AtlasDirectorTake()
    rays = node.read_rays(str(take_dir), width=16, height=9)
    embedded = node.embed_rays(rays)

    import atlas_camera.plate.oiio_io as oiio_io
    monkeypatch.setattr(oiio_io, "oiio_available", lambda: False)

    out = node.write_ray_exr(str(take_dir), embedded)
    assert out == ""
    manifest = json.loads((take_dir / "manifest.json").read_text())
    assert "rayChannels" not in manifest


# --- full read() -------------------------------------------------------------


@pytest.fixture()
def real_playblast_take(tmp_path):
    """Same shape as `take_dir`, but with real (openable) PNG frames."""

    from PIL import Image

    directory = tmp_path / "takes" / "sc" / "sh" / "a_take01"
    (directory / "playblast").mkdir(parents=True)
    for index in range(4):
        img = Image.new("RGB", (16, 9), color=(index * 10, 0, 0))
        img.save(directory / "playblast" / f"frame.{index:04d}.png")
    samples = [
        {"position": [0.0, 1.6, float(index)], "rotation": [0, 0, 0, 1],
         "focalLengthMm": 35.0, "focusDistanceM": 4.0, "tStop": 2.8,
         "filmback": {"name": "S35", "widthMm": 24.89, "heightMm": 18.66}}
        for index in range(4)
    ]
    (directory / "samples.json").write_text(json.dumps(samples))
    (directory / "manifest.json").write_text(json.dumps({
        "schemaVersion": 1, "slate": "sc/sh/a_take01", "frameCount": 999,
    }))
    return directory


def test_read_returns_all_four_outputs_and_never_mutates_sessions(
        real_playblast_take, package_path, monkeypatch):
    import atlas_camera.plate.oiio_io as oiio_io
    monkeypatch.setattr(oiio_io, "oiio_available", lambda: False)

    SESSIONS.clear()
    SESSIONS["shot_012"] = {
        "session_id": "shot_012",
        "package": str(package_path),
        "package_digest": AtlasDirectorTake().package_digest(str(package_path)),
        "slate": "sc/sh/a_take01",
        "take_dir": str(real_playblast_take),
    }
    before = dict(SESSIONS["shot_012"])

    node = AtlasDirectorTake()
    out = node.read(
        session_id="shot_012", width=16, height=9, frames="121", fps=24,
        colour_lane="png",
    )
    result = out["result"] if isinstance(out, dict) else out
    playblast, rays, rays_preview, samples = result

    assert tuple(playblast.shape) == (4, 9, 16, 3)
    assert tuple(rays.shape) == (4, 9, 16, 6)
    assert tuple(rays_preview.shape) == (4, 9, 16, 3)
    assert len(samples) == 4
    assert SESSIONS["shot_012"] == before
    SESSIONS.clear()


def test_read_refuses_a_stale_package(real_playblast_take, package_path):
    SESSIONS.clear()
    SESSIONS["shot_012"] = {
        "session_id": "shot_012",
        "package": str(package_path),
        "package_digest": "stale-digest",
        "slate": "sc/sh/a_take01",
        "take_dir": str(real_playblast_take),
    }
    node = AtlasDirectorTake()
    with pytest.raises(StaleTakeError):
        node.read(session_id="shot_012", width=16, height=9, frames="121",
                   fps=24, colour_lane="png")
    SESSIONS.clear()
