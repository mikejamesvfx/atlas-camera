"""Portable single-file containers for an Atlas package tree.

The document schema deliberately knows nothing about ZIP.  Inside the archive
the package is the same ordinary tree every existing reader understands; this
module only moves that tree safely between disk and one user-facing file.
"""

from __future__ import annotations

import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

ATLAS_MIMETYPE = "application/vnd.atlas.scene+zip"

_STORED_SUFFIXES = {
    ".exr",
    ".hdr",
    ".rgbe",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".glb",
    ".usd",
    ".usda",
    ".usdc",
    ".usdz",
    ".zip",
    ".gz",
}


class ContainerError(ValueError):
    """An archive is malformed, unsafe, or cannot be written intact."""


def _member_name(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _compression(path: Path) -> int:
    return zipfile.ZIP_STORED if path.suffix.lower() in _STORED_SUFFIXES else zipfile.ZIP_DEFLATED


def _validate_member(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename
    if not name or "\\" in name or "\x00" in name:
        raise ContainerError(f"unsafe archive member {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ContainerError(f"unsafe archive member {name!r}")
    if any(":" in part for part in path.parts):
        raise ContainerError(f"unsafe archive member {name!r}")
    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise ContainerError(f"archive member {name!r} is a symbolic link")
    if info.flag_bits & 0x1:
        raise ContainerError(f"archive member {name!r} is encrypted")
    return path


def inspect_archive(path: str | Path) -> list[zipfile.ZipInfo]:
    """Validate container structure and return its member inventory."""

    archive = Path(path)
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        raise ContainerError(f"{archive} is not an Atlas archive")
    try:
        with zipfile.ZipFile(archive) as handle:
            infos = handle.infolist()
            if not infos or infos[0].filename != "mimetype":
                raise ContainerError("Atlas archive must begin with the root mimetype member")
            try:
                marker = handle.read("mimetype").decode("ascii")
            except (KeyError, UnicodeDecodeError, RuntimeError, zipfile.BadZipFile) as error:
                raise ContainerError("Atlas archive has an unreadable mimetype marker") from error
            if marker != ATLAS_MIMETYPE:
                raise ContainerError(f"unsupported Atlas archive mimetype {marker!r}")

            seen: set[str] = set()
            has_document = False
            for info in infos:
                member = _validate_member(info)
                folded = member.as_posix().casefold().rstrip("/")
                if folded in seen:
                    raise ContainerError(
                        f"duplicate or case-colliding archive member {info.filename!r}"
                    )
                seen.add(folded)
                has_document = has_document or member.as_posix() == "scene.json"
            if not has_document:
                raise ContainerError("Atlas archive has no root scene.json")
            corrupt = handle.testzip()
            if corrupt is not None:
                raise ContainerError(f"Atlas archive member {corrupt!r} failed its CRC check")
            return infos
    except zipfile.BadZipFile as error:
        raise ContainerError(f"{archive} is a corrupt Atlas archive") from error


def pack_archive(source_root: str | Path, destination: str | Path) -> Path:
    """Atomically pack an ordinary Atlas package tree into one ``.atlas`` file."""

    source = Path(source_root)
    target = Path(destination)
    if not (source / "scene.json").is_file():
        raise ContainerError(f"Atlas package {source} has no scene.json")
    if target.exists() and target.is_dir():
        raise ContainerError(f"archive destination {target} is a directory")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as handle:
            marker = zipfile.ZipInfo("mimetype")
            marker.compress_type = zipfile.ZIP_STORED
            marker.external_attr = 0o100644 << 16
            handle.writestr(marker, ATLAS_MIMETYPE)
            for path in sorted(source.rglob("*"), key=lambda item: item.as_posix().casefold()):
                if path.is_symlink():
                    raise ContainerError(f"package source contains symbolic link {path}")
                if path.is_dir():
                    directory = zipfile.ZipInfo(_member_name(path, source).rstrip("/") + "/")
                    directory.compress_type = zipfile.ZIP_STORED
                    directory.external_attr = (stat.S_IFDIR | 0o755) << 16
                    handle.writestr(directory, b"")
                    continue
                if not path.is_file() or _member_name(path, source) == "mimetype":
                    continue
                handle.write(path, _member_name(path, source), compress_type=_compression(path))
        inspect_archive(temporary)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def unpack_archive(path: str | Path, destination: str | Path) -> Path:
    """Safely extract an Atlas archive into a new or empty workspace."""

    archive = Path(path)
    infos = inspect_archive(archive)
    target = Path(destination)
    if target.exists() and any(target.iterdir()):
        raise ContainerError(f"workspace destination {target} is not empty")

    required = sum(info.file_size for info in infos if not info.is_dir())
    probe = target if target.exists() else target.parent
    probe.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(probe).free
    if required > available:
        raise ContainerError(
            f"Atlas archive needs {required} bytes of cache space; only {available} are free"
        )

    target.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as handle:
            for info in infos:
                member = _validate_member(info)
                if member.as_posix() == "mimetype":
                    continue
                output = target.joinpath(*member.parts)
                if info.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(info, "r") as source, output.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=1024 * 1024)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return target
