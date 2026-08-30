import json
from pathlib import Path

import numpy as np
import pytest

from atlas_camera.comfy.director_session import SESSIONS
from atlas_camera.comfy.nodes_director import AtlasDirectorTake, StaleTakeError


def _write_take(directory: Path, *, n_frames: int, n_samples: int,
                 frame_count: int = 999, pretty: bool = False) -> None:
    (directory / "playblast").mkdir(parents=True)
    for index in range(n_frames):
        (directory / "playblast" / f"frame.{index:04d}.png").write_bytes(b"")
    samples = [
        {"position": [0.0, 1.6, float(index)], "rotation": [0, 0, 0, 1],
         "focalLengthMm": 35.0, "focusDistanceM": 4.0, "tStop": 2.8,
         "filmback": {"name": "S35", "widthMm": 24.89, "heightMm": 18.66}}
        for index in range(n_samples)
    ]
    (directory / "samples.json").write_text(json.dumps(samples))
    manifest = {"schemaVersion": 1, "slate": "sc/sh/a_take01", "frameCount": frame_count}
    text = json.dumps(manifest, indent=2) if pretty else json.dumps(manifest)
    (directory / "manifest.json").write_text(text)


@pytest.fixture()
def take_dir(tmp_path):
    """A committed take: 4 rendered frames, 8 samples on the FULL take.

    "marked to two" (of the take's real length) is not observable with 4
    samples / 4 frames -- `samples[:rendered]` and
    `samples[:manifest['frameCount']]` both land on 4 either way, which is
    exactly the defect this fixture exists to catch (review finding C1). 8
    samples / 4 frames makes the two diverge: reading the directory gives 4,
    reading the (deliberately wrong) manifest.frameCount=999 would give all
    8 (clamped by Python slicing, but 8 != 4) -- a real regression back to
    the manifest is now visible.

    Non-square resolution (16x9) throughout: the ray-map shape assertions
    use it so a width/height transposition -- the most likely defect in
    this code path -- cannot hide behind a square fixture (addendum
    Ruling P2).
    """

    directory = tmp_path / "takes" / "sc" / "sh" / "a_take01"
    _write_take(directory, n_frames=4, n_samples=8)
    return directory


@pytest.fixture()
def package_path(tmp_path):
    """A stand-in .atlas session package, whose bytes `check_fresh` digests."""

    path = tmp_path / "scenes" / "shot_012.atlas"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fake atlas package bytes")
    return path


def test_frame_count_comes_from_the_directory_not_the_manifest(take_dir):
    # manifest.frameCount says 999 on purpose, and there are 8 samples on
    # the full take but only 4 rendered frames. Reading the manifest would
    # claim frames that were never rendered.
    node = AtlasDirectorTake()
    rays = node.read_rays(str(take_dir), width=16, height=9)
    assert rays.shape[0] == 4


def test_frame_count_defect_is_actually_caught_by_the_fixture(take_dir, monkeypatch):
    """Meta-test for review finding C1: prove the fixture can fail.

    Temporarily makes `read_rays` use `manifest.json['frameCount']` instead
    of the rendered directory count -- the exact regression this node
    exists to prevent -- and confirms the shape assertion above would then
    FAIL. This is what was manually verified (and is asserted here so the
    guarantee survives future edits): see the task-6 report for the
    transcript of running it stand-alone with the bug live.
    """
    original_frame_files = AtlasDirectorTake.frame_files

    def buggy_read_rays(self, take_dir_str, width, height):
        from atlas_camera.comfy.plucker import ray_map
        samples = self._load_samples(take_dir_str)
        manifest = json.loads((Path(take_dir_str) / "manifest.json").read_text())
        rendered = manifest["frameCount"]  # the bug: manifest, not directory
        return ray_map(samples[:rendered], width, height)

    monkeypatch.setattr(AtlasDirectorTake, "read_rays", buggy_read_rays)
    node = AtlasDirectorTake()
    rays = node.read_rays(str(take_dir), width=16, height=9)
    # 8 samples, sliced to frameCount=999 -> all 8 survive. Real code gives 4.
    assert rays.shape[0] == 8
    assert rays.shape[0] != 4
    # AtlasDirectorTake.frame_files itself is untouched by the monkeypatch.
    assert original_frame_files is AtlasDirectorTake.frame_files


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


def test_return_types_and_names_pin_the_atlas_rays_socket():
    # Regression guard for review finding I3: reverting `rays` to plain
    # IMAGE (the brief's original, non-implementable version) must fail
    # this test even though every other test still passes unchanged.
    assert AtlasDirectorTake.RETURN_TYPES == ("IMAGE", "ATLAS_RAYS", "IMAGE", "ATLAS_CAMERA")
    assert AtlasDirectorTake.RETURN_NAMES == ("playblast", "rays", "rays_preview", "samples")


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


def test_a_pushed_session_with_no_recorded_digest_refuses_rather_than_skipping(tmp_path):
    # Review finding I2: a session with a take pushed but no package_digest
    # (e.g. launched before this field existed, or by a path that never
    # records one) must REFUSE, not silently perform no freshness check.
    SESSIONS.clear()
    SESSIONS["shot_012"] = {
        "session_id": "shot_012",
        "slate": "sc/sh/a_take01",
        "take_dir": str(tmp_path),
        "package": None,
        "package_digest": None,
    }
    node = AtlasDirectorTake()
    with pytest.raises(StaleTakeError, match="package"):
        node._ensure_fresh("shot_012", SESSIONS["shot_012"])
    SESSIONS.clear()


def test_a_pushed_session_with_a_package_but_no_digest_also_refuses(tmp_path, package_path):
    SESSIONS.clear()
    SESSIONS["shot_012"] = {
        "session_id": "shot_012",
        "slate": "sc/sh/a_take01",
        "take_dir": str(tmp_path),
        "package": str(package_path),
        "package_digest": None,
    }
    node = AtlasDirectorTake()
    with pytest.raises(StaleTakeError, match="package_digest"):
        node._ensure_fresh("shot_012", SESSIONS["shot_012"])
    SESSIONS.clear()


def test_allowed_frames_is_a_single_entry_tuple_for_now():
    from atlas_camera.comfy.nodes_director import ALLOWED_FRAMES

    assert ALLOWED_FRAMES == ("121",)


# --- the ray-map EXR sidecar (Ruling P7) ------------------------------------


def test_write_ray_exr_receives_ray_map_not_the_plucker_embedding(take_dir, monkeypatch):
    # Review finding C2: write_ray_exr must be called with `rays` (the
    # ray_map output: origin, direction) so its channel naming is honest --
    # never with `embedded` (the Plucker embedding: moment, direction).
    node = AtlasDirectorTake()
    rays = node.read_rays(str(take_dir), width=16, height=9)
    embedded = node.embed_rays(rays)
    assert not np.allclose(rays, embedded)  # sanity: the two really differ

    captured = []

    def fake_write_exr(path, pixels, *, bit_depth, **kwargs):
        captured.append((path, np.asarray(pixels).copy(), bit_depth, kwargs))

    import atlas_camera.plate.oiio_io as oiio_io
    monkeypatch.setattr(oiio_io, "oiio_available", lambda: True)
    monkeypatch.setattr(oiio_io, "write_exr", fake_write_exr)

    node.write_ray_exr(str(take_dir), rays)

    assert len(captured) == rays.shape[0]
    for index, (_path, pixels, bit_depth, kwargs) in enumerate(captured):
        assert bit_depth == "float"
        np.testing.assert_allclose(pixels, rays[index])
        # No colourspace arguments at all -- these are numbers, not colour.
        assert "source_colorspace" not in kwargs
        assert "output_colorspace" not in kwargs

    manifest = json.loads((take_dir / "manifest.json").read_text())
    assert manifest["rayChannels"] == ["O.X", "O.Y", "O.Z", "D.X", "D.Y", "D.Z"]


def test_write_ray_exr_preserves_existing_manifest_keys_and_pretty_formatting(tmp_path):
    directory = tmp_path / "takes" / "sc" / "sh" / "a_take01"
    _write_take(directory, n_frames=2, n_samples=2, pretty=True)
    node = AtlasDirectorTake()
    rays = node.read_rays(str(directory), width=16, height=9)

    import atlas_camera.plate.oiio_io as oiio_io
    if not oiio_io.oiio_available():
        pytest.skip("OpenImageIO unavailable")

    node.write_ray_exr(str(directory), rays)
    raw = (directory / "manifest.json").read_text()
    manifest = json.loads(raw)
    assert manifest["schemaVersion"] == 1
    assert manifest["slate"] == "sc/sh/a_take01"
    assert manifest["frameCount"] == 999
    assert manifest["rayChannels"] == ["O.X", "O.Y", "O.Z", "D.X", "D.Y", "D.Z"]
    assert list(manifest.keys()) == ["schemaVersion", "slate", "frameCount", "rayChannels"]
    assert "\n" in raw  # pretty formatting preserved, not collapsed to one line


def test_write_ray_exr_skips_without_failing_when_oiio_unavailable(take_dir, monkeypatch):
    node = AtlasDirectorTake()
    rays = node.read_rays(str(take_dir), width=16, height=9)

    import atlas_camera.plate.oiio_io as oiio_io
    monkeypatch.setattr(oiio_io, "oiio_available", lambda: False)

    out = node.write_ray_exr(str(take_dir), rays)
    assert out == ""
    manifest = json.loads((take_dir / "manifest.json").read_text())
    assert "rayChannels" not in manifest


def test_write_ray_exr_round_trips_through_real_oiio_with_correct_channels(take_dir):
    """Un-mocked EXR write, independently catching C2 too: if
    `write_ray_exr` were ever fed the Plucker embedding instead of the raw
    ray map, the values read back would be moments, not the recorded
    origins, and this would fail.
    """
    OpenImageIO = pytest.importorskip("OpenImageIO")
    node = AtlasDirectorTake()
    rays = node.read_rays(str(take_dir), width=16, height=9)

    rays_dir = node.write_ray_exr(str(take_dir), rays)
    assert rays_dir

    exr_path = Path(rays_dir) / "rays.0000.exr"
    buf = OpenImageIO.ImageBuf(str(exr_path))
    buf.read(0, 0, True, OpenImageIO.FLOAT)
    assert not buf.has_error
    pixels = np.asarray(buf.get_pixels(OpenImageIO.FLOAT), dtype=np.float32)
    assert pixels.shape == (9, 16, 6)
    np.testing.assert_allclose(pixels, rays[0].astype(np.float32), atol=1e-4)


# --- colour_lane / frame-format honesty (review finding I4) ----------------


def test_exr_colour_lane_refuses_when_frames_are_actually_png(take_dir):
    node = AtlasDirectorTake()
    with pytest.raises(ValueError) as raised:
        node.load_playblast(str(take_dir), "exr")
    message = str(raised.value).lower()
    assert "8-bit srgb" in message or "sRGB".lower() in message
    assert "colour_lane='png'" in str(raised.value)


def test_png_colour_lane_refuses_when_frames_are_actually_exr(tmp_path):
    directory = tmp_path / "takes" / "sc" / "sh" / "a_take01"
    (directory / "playblast").mkdir(parents=True)
    (directory / "playblast" / "frame.0000.exr").write_bytes(b"")
    (directory / "samples.json").write_text(json.dumps([]))
    node = AtlasDirectorTake()
    with pytest.raises(ValueError, match="colour_lane='exr'"):
        node.load_playblast(str(directory), "png")


def test_no_playblast_directory_names_the_mp4_case(tmp_path):
    directory = tmp_path / "takes" / "sc" / "sh" / "a_take01"
    directory.mkdir(parents=True)
    (directory / "playblast.mp4").write_bytes(b"")
    node = AtlasDirectorTake()
    with pytest.raises(ValueError, match="playblast.mp4"):
        node.frame_files(str(directory))


def test_no_playblast_directory_and_no_mp4_refuses_generically(tmp_path):
    directory = tmp_path / "takes" / "sc" / "sh" / "a_take01"
    directory.mkdir(parents=True)
    node = AtlasDirectorTake()
    with pytest.raises(ValueError, match="no playblast/ directory"):
        node.frame_files(str(directory))


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
        for index in range(8)
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
        "timebase": {"width": 16, "height": 9, "frames": 121, "fps": 24},
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
    assert len(samples) == 4  # 8 samples on the take, 4 frames rendered
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


def test_read_refuses_when_no_digest_was_ever_recorded(real_playblast_take, package_path):
    SESSIONS.clear()
    SESSIONS["shot_012"] = {
        "session_id": "shot_012",
        "package": str(package_path),
        "package_digest": None,
        "slate": "sc/sh/a_take01",
        "take_dir": str(real_playblast_take),
    }
    node = AtlasDirectorTake()
    with pytest.raises(StaleTakeError, match="package_digest"):
        node.read(session_id="shot_012", width=16, height=9, frames="121",
                   fps=24, colour_lane="png")
    SESSIONS.clear()


def test_read_refuses_when_the_nodes_timebase_does_not_match_the_session(
        real_playblast_take, package_path, monkeypatch):
    import atlas_camera.plate.oiio_io as oiio_io
    monkeypatch.setattr(oiio_io, "oiio_available", lambda: False)

    SESSIONS.clear()
    SESSIONS["shot_012"] = {
        "session_id": "shot_012",
        "package": str(package_path),
        "package_digest": AtlasDirectorTake().package_digest(str(package_path)),
        # Launched at a different width/height than the node below reads with.
        "timebase": {"width": 768, "height": 512, "frames": 121, "fps": 24},
        "slate": "sc/sh/a_take01",
        "take_dir": str(real_playblast_take),
    }
    node = AtlasDirectorTake()
    with pytest.raises(ValueError, match="timebase"):
        node.read(session_id="shot_012", width=16, height=9, frames="121",
                   fps=24, colour_lane="png")
    SESSIONS.clear()
