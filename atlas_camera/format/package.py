"""Open an existing `.atlas` package, edit it, and write it back intact.

**Why this module exists.** Until now this build could only CREATE packages.
`atlas_camera.exporters.atlas_package` builds a document from a solve, from
scratch, every time — which is right for an import and catastrophic for an
edit, because a document built from scratch contains exactly what this build
knows how to write and nothing else. Atlas Scene writes schema 0.7 with a
first-class `environment` block and provenance-bearing capture metadata that
this build does not interpret. Rebuilt rather than carried, those fields do not
come back.

**The rule.** A reader that does not own a field must return it unchanged. Not
"best effort", not "the parts we recognise": the document read off disk is a
plain `dict`, it stays a plain `dict`, and writing it back writes the same keys
in the same order with the same values. Nothing here routes a loaded document
through `scene_document()`, which would rebuild it from this build's literal and
silently drop every key 0.7 added.

    Given a canonical Scene-produced .atlas 0.7 package containing Scene-owned
    fields and artifacts unknown to Atlas Camera, when Atlas Camera reads it,
    modifies only Camera-owned state, and writes the document back, then all
    unrelated valid 0.7 content remains byte-identical where possible, or
    semantically identical where repacking necessarily changes representation.

That is ADR-001 step 3, and `tests/test_atlas_preservation.py` is where it is
checked against a real Scene-authored package rather than asserted here.

**Create versus edit.** These are different operations and this build now says
so. `write_atlas_archive` CREATES and refuses a destination that already exists.
`open_package` EDITS and is the only supported way to change a package that
somebody else wrote. Reaching for the writer to update a package is the mistake
this split exists to make impossible to commit by accident.

**Opening does not write.** Reading a package is free of side effects, including
on the archive's bytes. A change reaches disk when you call `commit()` and at no
other moment. Forgetting to is not silently tolerated: if the document was
modified and neither committed nor explicitly discarded, closing the package
raises, because an edit that evaporates is worse than one that fails loudly.
"""

from __future__ import annotations

import copy
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from atlas_camera.format.container import pack_archive, unpack_archive
from atlas_camera.format.layout import SCENE_DOCUMENT
from atlas_camera.format.version import check_readable

__all__ = [
    "AtlasPackage",
    "PackageError",
    "UncommittedChanges",
    "open_package",
    "read_document",
    "write_document",
]


class PackageError(RuntimeError):
    """Raised when a package cannot be opened, read or written."""


class UncommittedChanges(PackageError):
    """Raised when an edited package is closed without commit() or discard()."""


# --------------------------------------------------------------------- primitives
# Low-level, and deliberately dull. They read and write one file in an already
# extracted tree. `open_package` is the API to reach for; these are what it is
# built out of, and what a caller who is managing the tree themselves can use.


def read_document(root: str | Path) -> dict[str, Any]:
    """Return `scene.json` from an extracted package tree, exactly as written.

    "Exactly as written" is the whole contract. `json.load` preserves key order
    and round-trips floats through `repr`, so the mapping returned here can be
    handed straight back to `write_document` and produce the same bytes. No
    normalisation, no defaults filled in, no unknown keys stripped.

    The version is checked but the document is not otherwise validated: a
    caller that wants `validate_document` can run it, and a caller that only
    wants to read one field out of a package should not be forced to.
    """

    target = Path(root) / SCENE_DOCUMENT
    if not target.is_file():
        raise PackageError(f"no {SCENE_DOCUMENT} in package tree {root}")

    try:
        with target.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as error:
        raise PackageError(f"{target} is not valid JSON: {error}") from error

    if not isinstance(document, dict):
        raise PackageError(f"{target} is not a JSON object")

    check_readable(str(document.get("schema_version", "")))
    return document


def write_document(root: str | Path, document: dict[str, Any]) -> None:
    """Write `scene.json` back into an extracted tree, atomically.

    Byte-for-byte the same serialisation the package writer uses — `indent=2`,
    `sort_keys=False`, trailing newline — so a document read and written back
    untouched is unchanged on disk. `sort_keys=False` is not a style choice:
    key order is part of the format, and the producer spec's round-trip check
    is byte-identical.

    The version is checked on the way out as well as the way in. Writing a
    version this build cannot read would produce a package it could not open
    again, and the failure would surface later, somewhere else.
    """

    check_readable(str(document.get("schema_version", "")))

    target = Path(root) / SCENE_DOCUMENT
    if not target.parent.is_dir():
        raise PackageError(f"no package tree at {root}")

    temporary = target.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=False)
        handle.write("\n")
    temporary.replace(target)


# ----------------------------------------------------------------------- handle


class AtlasPackage:
    """An opened package: a mutable document and the tree it came from.

    Mutate `document` in place. Call `commit()` to write. `root` is the
    extracted tree, so a caller adding or replacing a side file (a matte, a
    mesh) writes it there and it is packed with everything else.
    """

    def __init__(self, root: Path, document: dict[str, Any], source: Path, *, archived: bool):
        self.root = root
        self.document = document
        #: The path the package was opened from, archive or directory.
        self.source = source
        self._archived = archived
        self._opened_as = copy.deepcopy(document)
        self._committed = False
        self._discarded = False
        self._closed = False

    # -- state ---------------------------------------------------------------

    @property
    def dirty(self) -> bool:
        """Has the document changed since it was opened?

        A deep compare rather than a mutation flag, so a caller that edits a
        nested list and a caller that assigns a top-level key are treated the
        same, and so that setting a field back to its original value correctly
        reads as no change at all.
        """

        return self.document != self._opened_as

    @property
    def committed(self) -> bool:
        return self._committed

    def discard(self) -> None:
        """Abandon the edits. Closing after this writes nothing."""

        self._discarded = True

    # -- writing -------------------------------------------------------------

    def commit(self) -> None:
        """Write the document back, and repack if this was an archive.

        Side files are carried by the container, not by this method:
        `pack_archive` walks the whole tree, so anything Scene left in
        `environments/`, `takes/` or `director/` is packed whether or not this
        build has ever heard of it.
        """

        if self._closed:
            raise PackageError("package is closed")

        write_document(self.root, self.document)

        if self._archived:
            # Pack beside the target and replace, so an interrupted commit
            # leaves the original package intact rather than half a zip.
            staging = self.source.with_name(f".{self.source.name}.tmp")
            try:
                pack_archive(self.root, staging)
                staging.replace(self.source)
            finally:
                if staging.exists():
                    staging.unlink()

        self._committed = True
        self._opened_as = copy.deepcopy(self.document)

    # -- lifecycle -----------------------------------------------------------

    def _close(self, *, failed: bool) -> None:
        if self._closed:
            return
        self._closed = True

        if failed or self._discarded or self._committed or not self.dirty:
            return

        raise UncommittedChanges(
            f"{self.source} was modified but never committed. Call commit() to "
            f"write the changes, or discard() if they were not meant to be kept."
        )


@contextmanager
def open_package(path: str | Path) -> Iterator[AtlasPackage]:
    """Open a `.atlas` package for inspection or editing.

    Accepts either a packed archive or an already-extracted tree. An archive is
    unpacked into a temporary directory and repacked on commit; a directory is
    edited where it lies.

    Inspection:

        with open_package("street.atlas") as package:
            print(package.document["schema_version"])

    Editing — note that nothing is written until `commit()`:

        with open_package("street.atlas") as package:
            package.document["scale"]["status"] = "measured"
            package.commit()

    Raises `UncommittedChanges` if the document was modified and neither
    committed nor discarded, so an edit is never silently lost. An exception
    raised inside the block propagates untouched; the package is not written
    and the check is skipped, because the interesting failure is the original
    one.
    """

    source = Path(path)
    if not source.exists():
        raise PackageError(f"no package at {source}")

    archived = source.is_file()
    staging: tempfile.TemporaryDirectory | None = None

    if archived:
        staging = tempfile.TemporaryDirectory(prefix=f".{source.stem}-")
        root = Path(staging.name) / "package"
        unpack_archive(source, root)
    else:
        root = source

    failed = False
    try:
        document = read_document(root)
        package = AtlasPackage(root, document, source, archived=archived)
        try:
            yield package
        except BaseException:
            failed = True
            raise
        finally:
            package._close(failed=failed)
    finally:
        if staging is not None:
            staging.cleanup()
