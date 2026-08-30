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


@pytest.fixture()
def director_root(tmp_path, monkeypatch):
    """Configure the root the way the module requires: env var, not a request."""
    monkeypatch.setenv("ATLAS_DIRECTOR_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture()
def director_take_root(tmp_path, monkeypatch):
    """Configure the SECOND root take_dir validates against.

    Deliberately a different directory from `director_root` -- takes live
    under Atlas Scene's workspace cache, not under ComfyUI's output
    directory, and this fixture exists to prove the two are validated
    independently.
    """
    root = tmp_path / "workspace_cache"
    root.mkdir()
    monkeypatch.setenv("ATLAS_DIRECTOR_TAKE_ROOT", str(root))
    return root


def _touch_package(root: Path, session_id: str, output_dir: str | None = None) -> Path:
    """Create the .atlas package launch_session now insists must pre-exist."""
    base = root / output_dir / "scenes" if output_dir else root / "scenes"
    base.mkdir(parents=True, exist_ok=True)
    package = base / f"{session_id}.atlas"
    package.write_bytes(b"")
    return package


def test_a_slug_session_id_is_accepted():
    assert validate_session_id("shot_012-a") == "shot_012-a"


@pytest.mark.parametrize("value", ["../escape", "with space", "a/b", "", "drive:evil", "..\\b"])
def test_a_non_slug_session_id_is_refused_not_sanitised(value):
    # take_ops.py's SLATE_PART allowlist, mirrored. Refused, never repaired:
    # a sanitiser turns an attack into a valid path with a plausible name.
    with pytest.raises(ValueError):
        validate_session_id(value)


def test_the_package_lands_in_the_scenes_lane(director_root):
    path = session_package_path(None, None, "shot_012")
    assert path.parent.name == "scenes"
    assert path.name == "shot_012.atlas"
    assert path.is_relative_to(director_root)


# --- CRITICAL: the root comes from configuration, never the request -------


def test_an_output_dir_outside_the_configured_root_is_refused(director_root):
    with pytest.raises(ValueError):
        session_package_path(None, "../escape", "shot_012")


def test_an_absolute_output_dir_is_refused(director_root, tmp_path):
    other = tmp_path.parent / "elsewhere"
    with pytest.raises(ValueError):
        session_package_path(None, str(other), "shot_012")


def test_an_output_dir_containing_dotdot_is_refused(director_root):
    with pytest.raises(ValueError):
        session_package_path(None, "shots/../../escape", "shot_012")


def test_session_package_path_requires_a_configured_root(monkeypatch):
    monkeypatch.delenv("ATLAS_DIRECTOR_ROOT", raising=False)
    with pytest.raises(RuntimeError):
        session_package_path(None, None, "shot_012")


# --- IMPORTANT: prefix-unsafe containment check ----------------------------


def test_containment_uses_is_relative_to_not_a_string_prefix(director_root):
    # A sibling directory that merely starts with the root's name must not
    # be treated as contained. Regression guard for the startswith() bug:
    # session_package_path must never resolve into `sibling` when asked to
    # resolve inside `director_root`.
    sibling = director_root.parent / (director_root.name + "-evil")
    sibling.mkdir(parents=True, exist_ok=True)
    path = session_package_path(None, None, "shot_012")
    assert path.is_relative_to(director_root)
    assert not path.is_relative_to(sibling)


# --- IMPORTANT: the package must already exist before Director launches ---


def test_launch_refuses_when_the_package_does_not_exist(director_root, monkeypatch):
    monkeypatch.setenv("ATLAS_DIRECTOR_BIN", "C:/fake/electron.exe")
    spawned = {}

    def spawn(argv):
        spawned["called"] = True

    with pytest.raises(ValueError):
        launch_session(
            {"session_id": "shot_012",
             "width": 768, "height": 512, "frames": 121, "fps": 24},
            spawn=spawn,
        )
    assert "called" not in spawned


def test_launch_spawns_with_a_server_composed_argv(director_root, monkeypatch):
    monkeypatch.setenv("ATLAS_DIRECTOR_BIN", "C:/fake/electron.exe")
    _touch_package(director_root, "shot_012")
    spawned = {}

    def spawn(argv):
        spawned["argv"] = argv

    result = launch_session(
        {"session_id": "shot_012",
         "width": 768, "height": 512, "frames": 121, "fps": 24},
        spawn=spawn,
    )
    assert spawned["argv"][0] == "C:/fake/electron.exe"
    assert "--director-session" in spawned["argv"]
    # Exactly the server-composed argv -- nothing appended from the request.
    assert len(spawned["argv"]) == 3
    assert result["session_id"] == "shot_012"


def test_a_request_cannot_name_the_executable(director_root, monkeypatch):
    monkeypatch.setenv("ATLAS_DIRECTOR_BIN", "C:/fake/electron.exe")
    _touch_package(director_root, "shot_012")
    spawned = {}

    def spawn(argv):
        spawned["argv"] = argv

    launch_session(
        {"session_id": "shot_012",
         "executable": "C:/evil.exe", "argv": ["--do-harm"],
         "width": 768, "height": 512, "frames": 121, "fps": 24},
        spawn=spawn,
    )
    assert spawned["argv"][0] == "C:/fake/electron.exe"
    assert "--do-harm" not in spawned["argv"]
    assert len(spawned["argv"]) == 3


def test_launch_refuses_when_no_executable_is_configured(director_root, monkeypatch):
    monkeypatch.delenv("ATLAS_DIRECTOR_BIN", raising=False)
    _touch_package(director_root, "shot_012")
    with pytest.raises(RuntimeError):
        launch_session(
            {"session_id": "shot_012",
             "width": 768, "height": 512, "frames": 121, "fps": 24},
            spawn=lambda argv: None,
        )


def test_the_session_json_carries_the_timebase(director_root, monkeypatch):
    monkeypatch.setenv("ATLAS_DIRECTOR_BIN", "C:/fake/electron.exe")
    _touch_package(director_root, "shot_012")
    launch_session(
        {"session_id": "shot_012",
         "width": 768, "height": 512, "frames": 121, "fps": 24},
        spawn=lambda argv: None,
    )
    written = json.loads((director_root / "scenes" / "shot_012.session.json").read_text())
    assert written["timebase"] == {"width": 768, "height": 512, "frames": 121, "fps": 24}


def test_a_request_output_dir_cannot_relocate_the_root(director_root, monkeypatch):
    # Even a request output_dir that is a real relative subpath only ever
    # nests under the configured root -- it can never become the root.
    monkeypatch.setenv("ATLAS_DIRECTOR_BIN", "C:/fake/electron.exe")
    _touch_package(director_root, "shot_012", output_dir="proj_a")
    session = launch_session(
        {"session_id": "shot_012", "output_dir": "proj_a",
         "width": 768, "height": 512, "frames": 121, "fps": 24},
        spawn=lambda argv: None,
    )
    package = Path(session["package"])
    assert package.is_relative_to(director_root)
    assert package == (director_root / "proj_a" / "scenes" / "shot_012.atlas").resolve()


# --- record_delivery --------------------------------------------------------


def test_a_delivery_is_idempotent(director_take_root):
    take = director_take_root / "takes" / "sc" / "sh" / "a_take01"
    take.mkdir(parents=True)
    SESSIONS["shot_012"] = {"session_id": "shot_012"}
    first = record_delivery("shot_012", "sc/sh/a_take01", str(take))
    second = record_delivery("shot_012", "sc/sh/a_take01", str(take))
    assert first == second
    assert SESSIONS["shot_012"]["slate"] == "sc/sh/a_take01"


def test_a_second_delivery_wins(director_take_root):
    take01 = director_take_root / "takes" / "sc" / "sh" / "a_take01"
    take02 = director_take_root / "takes" / "sc" / "sh" / "a_take02"
    take01.mkdir(parents=True)
    take02.mkdir(parents=True)
    SESSIONS["shot_012"] = {"session_id": "shot_012"}
    record_delivery("shot_012", "sc/sh/a_take01", str(take01))
    record_delivery("shot_012", "sc/sh/a_take02", str(take02))
    assert SESSIONS["shot_012"]["slate"] == "sc/sh/a_take02"


def test_a_delivery_to_an_unknown_session_is_refused():
    with pytest.raises(KeyError):
        record_delivery("never_launched", "sc/sh/a_take01", "C:/p")


def test_a_delivery_with_a_non_slug_slate_part_is_refused(director_take_root):
    take = director_take_root / "takes" / "evil"
    take.mkdir(parents=True)
    SESSIONS["shot_012"] = {"session_id": "shot_012"}
    with pytest.raises(ValueError):
        record_delivery("shot_012", "../escape/a_take01", str(take))


def test_a_delivery_with_a_take_dir_outside_the_root_is_refused(director_take_root):
    SESSIONS["shot_012"] = {"session_id": "shot_012"}
    with pytest.raises(ValueError):
        record_delivery("shot_012", "sc/sh/a_take01", "C:/somewhere/else")


# --- CRITICAL: take_dir validates against ATLAS_DIRECTOR_TAKE_ROOT, a ------
# --- SECOND root distinct from ATLAS_DIRECTOR_ROOT -------------------------


def test_a_take_dir_under_the_workspace_cache_root_is_the_real_shape(
        director_root, director_take_root):
    """The realistic case this finding exists for: Atlas Scene's workspace
    cache is validated only against the take root, never against
    director_root -- a real push must be accepted even when the two roots
    are unrelated directories (both fixtures happen to nest under the same
    tmp_path here only because that is pytest's isolation mechanism)."""
    take = director_take_root / "workspaces" / "abc123" / "package" / "takes" / "sc" / "sh" / "a_take01"
    take.mkdir(parents=True)
    SESSIONS["shot_012"] = {"session_id": "shot_012"}
    session = record_delivery("shot_012", "sc/sh/a_take01", str(take))
    assert session["take_dir"] == str(take)


def test_a_take_dir_outside_the_take_root_is_refused(director_take_root):
    SESSIONS["shot_012"] = {"session_id": "shot_012"}
    with pytest.raises(ValueError):
        record_delivery("shot_012", "sc/sh/a_take01", str(director_take_root.parent / "elsewhere"))


def test_take_dir_validation_fails_closed_when_the_take_root_is_unset(monkeypatch):
    monkeypatch.delenv("ATLAS_DIRECTOR_TAKE_ROOT", raising=False)
    SESSIONS["shot_012"] = {"session_id": "shot_012"}
    with pytest.raises(RuntimeError, match="ATLAS_DIRECTOR_TAKE_ROOT"):
        record_delivery("shot_012", "sc/sh/a_take01", "C:/anywhere")


# --- MINOR: SESSIONS is capped -----------------------------------------------


def test_sessions_dict_is_capped(director_root, monkeypatch):
    from atlas_camera.comfy import director_session as mod

    monkeypatch.setenv("ATLAS_DIRECTOR_BIN", "C:/fake/electron.exe")
    monkeypatch.setattr(mod, "_SESSION_LIMIT", 3)
    for i in range(5):
        sid = f"shot_{i:03d}"
        _touch_package(director_root, sid)
        launch_session(
            {"session_id": sid,
             "width": 768, "height": 512, "frames": 121, "fps": 24},
            spawn=lambda argv: None,
        )
    assert len(SESSIONS) <= 3
    # the most recent launches survive
    assert "shot_004" in SESSIONS
