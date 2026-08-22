"""`atlas.format` — the `.atlas` scene package, shared by every producer.

**What this is.** A `.atlas` package is a directory with `scene.json` at its
root and the files that document points at beside it. Atlas Camera writes one;
Atlas Scene opens, edits and writes one back. This package owns the parts both
sides must agree on — the schema, its version, the digests, and validation — so
that agreement lives in one place instead of being re-derived on each side and
drifting.

**Why here and not in the editor.** The editor currently carries a mirror of
these structures with a drift alarm pointed at this repository, described in its
own rules as a transitional arrangement to be *deleted* rather than hardened
into a second permanent validator. Every producer needs the same answers —
"is this well-formed", "may this reader open it", "does this digest match" — and
a format with two validators has no validator.

**Zero dependencies, deliberately.** Nothing here imports numpy, OpenImageIO,
torch or ComfyUI. The whole point of a shared format library is that anything
can import it, including a headless consumer with no vision stack, so it stays
standard library only. The layer that writes pixels and meshes to disk is
`atlas_camera.exporters.atlas_package`, and that one may have opinions about
dependencies.

**The five rules the format exists to keep.** Everything below follows from
them; where this package is silent, they still apply:

> Evidence is immutable. History is append-only. Derived state is replaceable.
> Scene state is editable. Every replacement preserves lineage.
"""

from atlas_camera.format.digest import digest_bytes, digest_json
from atlas_camera.format.document import (
    camera_document,
    plane_documents,
    scene_document,
)
from atlas_camera.format.identity import PLANE_NORMAL_PLACES, PLANE_OFFSET_PLACES, plane_id_for
from atlas_camera.format.layout import (
    ATLAS_DIR,
    GEOMETRY_DIR,
    HISTORY_DIR,
    IMAGERY_DIR,
    MATTES_DIR,
    SCENE_DOCUMENT,
    TEXTURES_DIR,
    VISIBILITY_DIR,
)
from atlas_camera.format.validate import FormatError, validate_document
from atlas_camera.format.version import (
    READABLE_SCHEMA_VERSIONS,
    SCHEMA_VERSION,
    check_readable,
)

__all__ = [
    "ATLAS_DIR",
    "FormatError",
    "GEOMETRY_DIR",
    "HISTORY_DIR",
    "IMAGERY_DIR",
    "MATTES_DIR",
    "PLANE_NORMAL_PLACES",
    "PLANE_OFFSET_PLACES",
    "READABLE_SCHEMA_VERSIONS",
    "SCENE_DOCUMENT",
    "SCHEMA_VERSION",
    "TEXTURES_DIR",
    "VISIBILITY_DIR",
    "camera_document",
    "check_readable",
    "digest_bytes",
    "digest_json",
    "plane_documents",
    "plane_id_for",
    "scene_document",
    "validate_document",
]
