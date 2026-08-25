"""The portable single-file container around an Atlas package tree."""

from __future__ import annotations

import stat
import zipfile
from pathlib import Path

import pytest

from atlas_camera.format.container import (
    ATLAS_MIMETYPE,
    ContainerError,
    pack_archive,
    unpack_archive,
)


def package_tree(root: Path) -> Path:
    (root / "geometry").mkdir(parents=True)
    (root / "imagery").mkdir()
    (root / "scene.json").write_text('{"schema_version":"0.7"}\n', encoding="utf-8")
    (root / "geometry" / "relief.obj").write_text("v 0 0 0\n", encoding="utf-8")
    (root / "imagery" / "plate.exr").write_bytes(b"EXR pixels stay exact")
    (root / "history").mkdir()
    return root


def test_archive_has_marker_at_root_and_round_trips_bytes(tmp_path):
    source = package_tree(tmp_path / "tree")
    archive = pack_archive(source, tmp_path / "shot.atlas")

    assert archive.is_file()
    with zipfile.ZipFile(archive) as handle:
        assert handle.namelist()[0] == "mimetype"
        assert handle.read("mimetype").decode("ascii") == ATLAS_MIMETYPE
        assert handle.getinfo("imagery/plate.exr").compress_type == zipfile.ZIP_STORED
        assert handle.getinfo("geometry/relief.obj").compress_type == zipfile.ZIP_DEFLATED
        assert handle.getinfo("history/").is_dir()

    restored = unpack_archive(archive, tmp_path / "restored")
    assert (restored / "scene.json").read_bytes() == (source / "scene.json").read_bytes()
    assert (restored / "imagery" / "plate.exr").read_bytes() == b"EXR pixels stay exact"


@pytest.mark.parametrize("member", ["../escape", "/absolute", "C:/drive"])
def test_unpack_refuses_paths_outside_the_workspace(tmp_path, member):
    archive = tmp_path / "bad.atlas"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("mimetype", ATLAS_MIMETYPE)
        handle.writestr("scene.json", "{}")
        handle.writestr(member, "bad")

    with pytest.raises(ContainerError, match="unsafe archive member"):
        unpack_archive(archive, tmp_path / "out")


def test_unpack_refuses_case_collisions(tmp_path):
    archive = tmp_path / "bad.atlas"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("mimetype", ATLAS_MIMETYPE)
        handle.writestr("scene.json", "{}")
        handle.writestr("Geometry/a.obj", "a")
        handle.writestr("geometry/A.obj", "b")

    with pytest.raises(ContainerError, match="duplicate or case-colliding"):
        unpack_archive(archive, tmp_path / "out")


def test_unpack_refuses_symlink_entries(tmp_path):
    archive = tmp_path / "bad.atlas"
    info = zipfile.ZipInfo("geometry/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("mimetype", ATLAS_MIMETYPE)
        handle.writestr("scene.json", "{}")
        handle.writestr(info, "../../outside")

    with pytest.raises(ContainerError, match="symbolic link"):
        unpack_archive(archive, tmp_path / "out")


def test_failed_pack_does_not_replace_an_existing_archive(tmp_path):
    destination = tmp_path / "shot.atlas"
    destination.write_bytes(b"previous archive")
    source = tmp_path / "incomplete"
    source.mkdir()

    with pytest.raises(ContainerError, match="scene.json"):
        pack_archive(source, destination)

    assert destination.read_bytes() == b"previous archive"
