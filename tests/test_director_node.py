import json
from pathlib import Path

import numpy as np
import pytest

from atlas_camera.comfy.director_session import SESSIONS
from atlas_camera.comfy.nodes_director import AtlasDirectorTake, StaleTakeError
from atlas_camera.comfy.plucker import ray_map


def _write_marks(directory: Path, *, frame_count: int, marked: bool = False,
                  mark_in: "int | None" = None, mark_out: "int | None" = None) -> None:
    """`<take_dir>/playblast/playblast.json` -- same schema `capturePlayblast`
    writes (Finding 2): `in`/`out` always present, null when unmarked.
    """
    marks = {
        "frame_count": frame_count,
        "in": mark_in if marked else None,
        "out": mark_out if marked else None,
        "marked": marked,
    }
    (directory / "playblast" / "playblast.json").write_text(
        json.dumps(marks, sort_keys=True, separators=(",", ":"))
    )


def _write_take(directory: Path, *, n_frames: int, n_samples: int,
                 frame_count: int = 999, pretty: bool = False,
                 write_marks: bool = True, marked: bool = False,
                 mark_in: "int | None" = None,
                 mark_out: "int | None" = None) -> None:
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
    if write_marks:
        # Default: unmarked, frame_count equal to what was actually
        # rendered -- matches every existing fixture's real behaviour
        # (frames staged from index 0 of the take).
        _write_marks(directory, frame_count=n_frames, marked=marked,
                     mark_in=mark_in, mark_out=mark_out)


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
    #
    # Finding 5: re-pushing the same take cannot fix this -- the recorded
    # digest was taken when Director was launched, so re-pushing the same
    # slate re-delivers it against the same stale digest. Only relaunching
    # Director re-records the digest, so that is the advice the message
    # must give, and it must say why re-pushing will not work.
    node = AtlasDirectorTake()
    with pytest.raises(StaleTakeError) as raised:
        node.check_fresh(str(package_path), recorded_digest="not-the-digest",
                          slate="sc/sh/a_take01")
    message = str(raised.value)
    assert "sc/sh/a_take01" in message
    assert "relaunch director" in message.lower()
    assert "re-pushing" in message.lower()
    assert "will not fix" in message.lower() or "will not" in message.lower()


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


def test_ltx_frame_count_rule():
    from atlas_camera.comfy.nodes_director import (
        is_ltx_valid_frame_count,
        nearest_ltx_valid_frame_counts,
    )

    # The real case that used to be broken: a take marked to 73 frames.
    assert is_ltx_valid_frame_count(73)
    assert is_ltx_valid_frame_count(121)
    assert is_ltx_valid_frame_count(1)  # smallest valid value
    assert not is_ltx_valid_frame_count(74)
    assert nearest_ltx_valid_frame_counts(74) == (73, 81)


def test_frames_widget_value_failing_the_rule_is_refused():
    node = AtlasDirectorTake()
    with pytest.raises(ValueError, match="frames=74"):
        node._ensure_frames_widget_is_valid(74)
    with pytest.raises(ValueError, match="% 8 == 1"):
        node._ensure_frames_widget_is_valid(74)


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
    """Same shape as `take_dir`, but with real (openable) PNG frames.

    73 frames, not 4 -- 73 % 8 == 1, a valid LTX length, and it is the real
    case from the field this rule fixes: a take marked to 73 frames while
    the `frames` widget (launched at the shot's own timebase) sits at a
    different, also-valid value. The rendered count no longer has to equal
    the widget, so these deliberately differ in the tests below.
    """

    from PIL import Image

    directory = tmp_path / "takes" / "sc" / "sh" / "a_take01"
    (directory / "playblast").mkdir(parents=True)
    for index in range(73):
        img = Image.new("RGB", (16, 9), color=(index % 256, 0, 0))
        img.save(directory / "playblast" / f"frame.{index:04d}.png")
    samples = [
        {"position": [0.0, 1.6, float(index)], "rotation": [0, 0, 0, 1],
         "focalLengthMm": 35.0, "focusDistanceM": 4.0, "tStop": 2.8,
         "filmback": {"name": "S35", "widthMm": 24.89, "heightMm": 18.66}}
        for index in range(73)
    ]
    (directory / "samples.json").write_text(json.dumps(samples))
    (directory / "manifest.json").write_text(json.dumps({
        "schemaVersion": 1, "slate": "sc/sh/a_take01", "frameCount": 999,
    }))
    _write_marks(directory, frame_count=73, marked=False)
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
        # frames=121 -- the shot's launch timebase -- deliberately does NOT
        # match real_playblast_take's 73 rendered frames. The rendered
        # count no longer has to equal the frames widget (see
        # `_ensure_frame_count_is_ltx_valid`); both are independently valid
        # LTX lengths (121 % 8 == 1, 73 % 8 == 1), so this must succeed.
        "timebase": {"width": 16, "height": 9, "frames": 121, "fps": 24},
        "slate": "sc/sh/a_take01",
        "take_dir": str(real_playblast_take),
    }
    before = dict(SESSIONS["shot_012"])

    node = AtlasDirectorTake()
    out = node.read(
        session_id="shot_012", width=16, height=9, frames=121, fps=24,
        colour_lane="png",
    )
    result = out["result"] if isinstance(out, dict) else out
    playblast, rays, rays_preview, samples = result

    assert tuple(playblast.shape) == (73, 9, 16, 3)
    assert tuple(rays.shape) == (73, 9, 16, 6)
    assert tuple(rays_preview.shape) == (73, 9, 16, 3)
    assert len(samples) == 73  # 73 samples on the take, 73 frames rendered
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
        node.read(session_id="shot_012", width=16, height=9, frames=121,
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
        node.read(session_id="shot_012", width=16, height=9, frames=121,
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
        node.read(session_id="shot_012", width=16, height=9, frames=121,
                   fps=24, colour_lane="png")
    SESSIONS.clear()


# --- Finding 3: playblast pixel size / frame count vs the node's widgets ----


def test_matching_dimensions_pass(real_playblast_take, package_path, monkeypatch):
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
    node = AtlasDirectorTake()
    # real_playblast_take renders real 16x9 PNGs, 73 of them -- must not
    # raise, even though the frames widget (121) disagrees with that count.
    node.read(session_id="shot_012", width=16, height=9, frames=121,
               fps=24, colour_lane="png")
    SESSIONS.clear()


def test_a_mismatched_frame_width_refuses_and_names_both_sizes(
        real_playblast_take, package_path, monkeypatch):
    import atlas_camera.plate.oiio_io as oiio_io
    monkeypatch.setattr(oiio_io, "oiio_available", lambda: False)

    SESSIONS.clear()
    SESSIONS["shot_012"] = {
        "session_id": "shot_012",
        "package": str(package_path),
        "package_digest": AtlasDirectorTake().package_digest(str(package_path)),
        # The node's own widgets must match the session's recorded timebase
        # (`_ensure_timebase_matches`), so the timebase is set to the SAME
        # (wrong) width the widgets below use -- what's under test here is
        # that the *rendered pixels* (still 16x9) disagree with widget/
        # timebase width (32), not a timebase mismatch.
        "timebase": {"width": 32, "height": 9, "frames": 121, "fps": 24},
        "slate": "sc/sh/a_take01",
        "take_dir": str(real_playblast_take),
    }
    node = AtlasDirectorTake()
    with pytest.raises(StaleTakeError) as raised:
        node.read(session_id="shot_012", width=32, height=9, frames="121",
                   fps=24, colour_lane="png")
    message = str(raised.value)
    assert "16x9" in message  # the frame's real size
    assert "32x9" in message  # the widget values
    SESSIONS.clear()


# --- Finding 2: aligning samples to a marked (not head) rendered range -----


@pytest.fixture()
def marked_range_take(tmp_path):
    """8 samples on the full take; the rendered playblast is a MARKED
    sub-range (frames 3..6), not the head. `capturePlayblast` always stages
    rendered frames starting at file index 0 regardless of where in the
    take they came from, so a head-slice (`samples[:4]`) would silently
    describe take frames 0..3 instead of the actual 3..6 -- exactly the
    defect Finding 2 exists to catch.
    """
    directory = tmp_path / "takes" / "sc" / "sh" / "a_take01"
    _write_take(directory, n_frames=4, n_samples=8,
                marked=True, mark_in=3, mark_out=6)
    return directory


def test_marked_range_aligns_to_the_marked_samples_not_the_head(marked_range_take):
    node = AtlasDirectorTake()
    frame_paths = node.frame_files(str(marked_range_take))
    start, stop = node._aligned_sample_range(
        str(marked_range_take), frame_paths, sample_count=8,
    )
    assert (start, stop) == (3, 7)

    all_samples = json.loads((marked_range_take / "samples.json").read_text())
    rays = node.read_rays(str(marked_range_take), width=16, height=9)

    correct = ray_map(all_samples[3:7], 16, 9)
    head_slice_wrong = ray_map(all_samples[0:4], 16, 9)

    np.testing.assert_allclose(rays, correct)
    # The point of this test: a regression back to a plain head-slice must
    # fail it, because the two ranges give different rays here.
    assert not np.allclose(rays, head_slice_wrong)


def test_unmarked_range_still_reads_the_head(take_dir):
    # `take_dir` is unmarked (playblast.json written with marked=False by
    # `_write_take`'s default), 4 rendered frames of 8 samples -- the head.
    node = AtlasDirectorTake()
    frame_paths = node.frame_files(str(take_dir))
    start, stop = node._aligned_sample_range(
        str(take_dir), frame_paths, sample_count=8,
    )
    assert (start, stop) == (0, 4)


def test_sidecar_disagreeing_with_disk_refuses_and_names_both_counts(tmp_path):
    directory = tmp_path / "takes" / "sc" / "sh" / "a_take01"
    # 4 real frames on disk, but the sidecar claims 6.
    _write_take(directory, n_frames=4, n_samples=8, write_marks=False)
    _write_marks(directory, frame_count=6, marked=False)
    node = AtlasDirectorTake()
    frame_paths = node.frame_files(str(directory))
    with pytest.raises(StaleTakeError) as raised:
        node._aligned_sample_range(str(directory), frame_paths, sample_count=8)
    message = str(raised.value)
    assert "6" in message
    assert "4" in message


def test_missing_sidecar_refuses(tmp_path):
    directory = tmp_path / "takes" / "sc" / "sh" / "a_take01"
    _write_take(directory, n_frames=4, n_samples=8, write_marks=False)
    node = AtlasDirectorTake()
    frame_paths = node.frame_files(str(directory))
    with pytest.raises(StaleTakeError, match="predates the marks record"):
        node._aligned_sample_range(str(directory), frame_paths, sample_count=8)


# --- coordinator fix-round-1: bounds, negative marks, contradictions,
# --- frame count vs the frames widget -----------------------------------


def test_marked_range_beyond_the_takes_samples_refuses(tmp_path):
    directory = tmp_path / "takes" / "sc" / "sh" / "a_take01"
    # 8 samples on the take, 4 real frames on disk -- but the sidecar
    # claims those 4 frames are marked in=6..out=9, which the width check
    # alone cannot catch (9+1-6 == 4, matching the on-disk count) even
    # though frame 9 does not exist in an 8-sample take.
    _write_take(directory, n_frames=4, n_samples=8, write_marks=False)
    _write_marks(directory, frame_count=4, marked=True, mark_in=6, mark_out=9)
    node = AtlasDirectorTake()
    frame_paths = node.frame_files(str(directory))
    with pytest.raises(StaleTakeError) as raised:
        node._aligned_sample_range(str(directory), frame_paths, sample_count=8)
    message = str(raised.value)
    assert "[6, 10)" in message
    assert "8" in message


def test_negative_mark_in_refuses(tmp_path):
    directory = tmp_path / "takes" / "sc" / "sh" / "a_take01"
    # in=-5, out=-2 satisfies the width check on its own (-2+1-(-5) == 4,
    # matching the 4 on-disk frames) via Python negative-index arithmetic
    # -- must be refused by name, not left to the bounds check alone.
    _write_take(directory, n_frames=4, n_samples=8, write_marks=False)
    _write_marks(directory, frame_count=4, marked=True, mark_in=-5, mark_out=-2)
    node = AtlasDirectorTake()
    frame_paths = node.frame_files(str(directory))
    with pytest.raises(StaleTakeError, match="negative mark"):
        node._aligned_sample_range(str(directory), frame_paths, sample_count=8)


def test_marked_false_with_non_null_marks_refuses(tmp_path):
    directory = tmp_path / "takes" / "sc" / "sh" / "a_take01"
    _write_take(directory, n_frames=4, n_samples=8, write_marks=False)
    # Two contradictory claims in one sidecar: marked=false, but in/out
    # are present anyway. Neither claim should be trusted.
    (directory / "playblast" / "playblast.json").write_text(json.dumps({
        "frame_count": 4, "marked": False, "in": 0, "out": 3,
    }))
    node = AtlasDirectorTake()
    frame_paths = node.frame_files(str(directory))
    with pytest.raises(StaleTakeError, match="marked=false"):
        node._aligned_sample_range(str(directory), frame_paths, sample_count=8)


def test_rendered_count_need_not_equal_frames_widget(
        real_playblast_take, package_path, monkeypatch):
    """The real case this rule fixes: a take marked to 73 frames, with the
    `frames` widget left at the shot's launch timebase (121). Both are
    independently valid LTX lengths (73 % 8 == 1, 121 % 8 == 1) -- forcing
    them to be equal bought no safety and blocked exactly this take.
    """
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
    node = AtlasDirectorTake()
    # Must NOT raise: real_playblast_take rendered 73 frames, the widget
    # says 121, and that mismatch is no longer a refusal.
    out = node.read(session_id="shot_012", width=16, height=9, frames=121,
                     fps=24, colour_lane="png")
    result = out["result"] if isinstance(out, dict) else out
    playblast = result[0]
    assert tuple(playblast.shape)[0] == 73
    SESSIONS.clear()


# --- the real LTX rule: n % 8 == 1, replacing the old equality-with-widget
# --- proxy ------------------------------------------------------------------


def test_rendered_count_73_is_ltx_valid(tmp_path):
    node = AtlasDirectorTake()
    frame_paths = [tmp_path / f"frame.{i:04d}.png" for i in range(73)]
    node._ensure_frame_count_is_ltx_valid(str(tmp_path), frame_paths)  # no raise


def test_rendered_count_121_is_ltx_valid(tmp_path):
    node = AtlasDirectorTake()
    frame_paths = [tmp_path / f"frame.{i:04d}.png" for i in range(121)]
    node._ensure_frame_count_is_ltx_valid(str(tmp_path), frame_paths)  # no raise


def test_rendered_count_1_is_ltx_valid(tmp_path):
    # The smallest valid value.
    node = AtlasDirectorTake()
    frame_paths = [tmp_path / "frame.0000.png"]
    node._ensure_frame_count_is_ltx_valid(str(tmp_path), frame_paths)  # no raise


def test_rendered_count_74_refuses_and_names_nearest_valid_counts(tmp_path):
    node = AtlasDirectorTake()
    frame_paths = [tmp_path / f"frame.{i:04d}.png" for i in range(74)]
    with pytest.raises(StaleTakeError) as raised:
        node._ensure_frame_count_is_ltx_valid(str(tmp_path), frame_paths)
    message = str(raised.value)
    assert "74" in message  # the rendered count
    assert "% 8 == 1" in message  # the rule
    assert "73" in message  # nearest valid count below
    assert "81" in message  # nearest valid count above
