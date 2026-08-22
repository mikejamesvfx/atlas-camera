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
SCHEMA_VERSION = "0.6"

#: What this build may READ. Every version listed here only ever ADDED fields,
#: and the defaults for those fields are the correct reading of a document that
#: predates them — `none` for a completion policy, `null` for a confidence —
#: not a fallback standing in for a value somebody forgot to write.
READABLE_SCHEMA_VERSIONS = frozenset({"0.2", "0.3", "0.4", "0.5", "0.6"})


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
