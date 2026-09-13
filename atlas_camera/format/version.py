"""Which `.atlas` versions this build writes, and which it may read.

Version negotiation belongs to the shared library rather than to each
application: two producers that disagree about whether they can read a document
have already failed, and the disagreement shows up as a misread field rather
than as an error.

**Unknown is refused, never guessed.** A version this build does not know is a
document whose fields may have changed meaning. Reading it anyway is how a field
that was redefined gets interpreted under the old rules — silently, and with
every downstream number wrong in a way nothing reports.
"""

from __future__ import annotations

#: What this build WRITES. Bumped by a change to the document's shape.
#:
#: 0.7 since 2026-09-13, ADR-001 step 4.4, and last of the four: the reader
#: widened first, preservation was proved in both directions, the shared
#: vocabulary was made tolerant, and the Camera<->Scene conformance check was
#: made unable to skip itself. Only then the writer, because a build that wrote
#: a version before it could read one back safely would have no way to discover
#: it was wrong.
#:
#: What this does NOT mean is that every document written here carries 0.7's
#: additions. `environment` and the camera's `capture_location` /
#: `capture_time` are optional and default correctly by their absence -- an
#: Atlas Camera package is plate-based, which is exactly what a missing
#: `environment` block asserts. The version says which rules the document is to
#: be read under, not which optional constructs it happens to use. Writing the
#: defaults explicitly would be new serialisation behaviour, and 4.4 is a
#: version promotion, not a format change.
SCHEMA_VERSION = "0.7"

#: What this build may READ. Every version listed here only ever ADDED fields,
#: and the defaults for those fields are the correct reading of a document that
#: predates them — `none` for a completion policy, `null` for a confidence —
#: not a fallback standing in for a value somebody forgot to write.
#:
#: 0.7 is now what BOTH producers write, and the canonical version (ADR-001).
#: 0.6 stays readable: retired as a producer version, not as a readable one, so
#: every package written before 2026-09-13 still opens. It adds a
#: first-class top-level `environment` and provenance-bearing capture metadata
#: on the camera (`capture_location`, `capture_time`). Atlas Camera interprets
#: none of the three, which is precisely why reading them is safe: nothing here
#: assumes ownership of Scene-authored content, and the editing path in
#: `atlas_camera.format.package` carries every field it does not understand
#: through a read -> write unchanged rather than rebuilding the document from
#: what this build happens to know about.
READABLE_SCHEMA_VERSIONS = frozenset({"0.2", "0.3", "0.4", "0.5", "0.6", "0.7"})


class UnsupportedVersion(RuntimeError):
    """Raised when a document's version is not one this build understands."""


def check_readable(version: str) -> str:
    """Return the version, or refuse loudly."""

    text = str(version or "")
    if text not in READABLE_SCHEMA_VERSIONS:
        raise UnsupportedVersion(
            f"unsupported .atlas schema_version {text!r}; "
            f"this build reads {sorted(READABLE_SCHEMA_VERSIONS)}"
        )
    return text
