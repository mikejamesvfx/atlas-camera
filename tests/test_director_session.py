import json
from pathlib import Path

import pytest

from atlas_camera.comfy.director_session import (
    SESSIONS,
    launch_session,
    record_delivery,
    session_package_path,
    validate_session_id,
)


@pytest.fixture(autouse=True)
def clear_sessions():
    SESSIONS.clear()
    yield
    SESSIONS.clear()


def test_a_slug_session_id_is_accepted():
    assert validate_session_id("shot_012-a") == "shot_012-a"


@pytest.mark.parametrize("value", ["../escape", "with space", "a/b", "", "drive:evil", "..\\b"])
def test_a_non_slug_session_id_is_refused_not_sanitised(value):
    # take_ops.py's SLATE_PART allowlist, mirrored. Refused, never repaired:
    # a sanitiser turns an attack into a valid path with a plausible name.
    with pytest.raises(ValueError):
        validate_session_id(value)


def test_the_package_lands_in_the_scenes_lane(tmp_path):
    path = session_package_path(None, str(tmp_path), "shot_012")
    assert path.parent.name == "scenes"
    assert path.name == "shot_012.atlas"


def test_a_path_outside_the_root_is_refused(tmp_path):
    with pytest.raises(ValueError):
        session_package_path(None, str(tmp_path), "..")


def test_launch_spawns_with_a_server_composed_argv(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_DIRECTOR_BIN", "C:/fake/electron.exe")
    spawned = {}

    def spawn(argv):
        spawned["argv"] = argv

    result = launch_session(
        {"session_id": "shot_012", "output_dir": str(tmp_path),
         "width": 768, "height": 512, "frames": 121, "fps": 24},
        spawn=spawn,
    )
    assert spawned["argv"][0] == "C:/fake/electron.exe"
    assert "--director-session" in spawned["argv"]
    assert result["session_id"] == "shot_012"


def test_a_request_cannot_name_the_executable(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_DIRECTOR_BIN", "C:/fake/electron.exe")
    spawned = {}

    def spawn(argv):
        spawned["argv"] = argv

    launch_session(
        {"session_id": "shot_012", "output_dir": str(tmp_path),
         "executable": "C:/evil.exe", "argv": ["--do-harm"],
         "width": 768, "height": 512, "frames": 121, "fps": 24},
        spawn=spawn,
    )
    assert spawned["argv"][0] == "C:/fake/electron.exe"
    assert "--do-harm" not in spawned["argv"]


def test_launch_refuses_when_no_executable_is_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("ATLAS_DIRECTOR_BIN", raising=False)
    with pytest.raises(RuntimeError):
        launch_session(
            {"session_id": "shot_012", "output_dir": str(tmp_path),
             "width": 768, "height": 512, "frames": 121, "fps": 24},
            spawn=lambda argv: None,
        )


def test_the_session_json_carries_the_timebase(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_DIRECTOR_BIN", "C:/fake/electron.exe")
    launch_session(
        {"session_id": "shot_012", "output_dir": str(tmp_path),
         "width": 768, "height": 512, "frames": 121, "fps": 24},
        spawn=lambda argv: None,
    )
    written = json.loads((Path(tmp_path) / "scenes" / "shot_012.session.json").read_text())
    assert written["timebase"] == {"width": 768, "height": 512, "frames": 121, "fps": 24}


def test_a_delivery_is_idempotent():
    SESSIONS["shot_012"] = {"session_id": "shot_012"}
    first = record_delivery("shot_012", "sc/sh/a_take01", "C:/p/takes/sc/sh/a_take01")
    second = record_delivery("shot_012", "sc/sh/a_take01", "C:/p/takes/sc/sh/a_take01")
    assert first == second
    assert SESSIONS["shot_012"]["slate"] == "sc/sh/a_take01"


def test_a_second_delivery_wins():
    SESSIONS["shot_012"] = {"session_id": "shot_012"}
    record_delivery("shot_012", "sc/sh/a_take01", "C:/p/takes/sc/sh/a_take01")
    record_delivery("shot_012", "sc/sh/a_take02", "C:/p/takes/sc/sh/a_take02")
    assert SESSIONS["shot_012"]["slate"] == "sc/sh/a_take02"


def test_a_delivery_to_an_unknown_session_is_refused():
    with pytest.raises(KeyError):
        record_delivery("never_launched", "sc/sh/a_take01", "C:/p")
