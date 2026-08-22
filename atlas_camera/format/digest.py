"""Content digests, spelled the same way on both sides of the format.

A digest is how a reader knows the file it opened is the file the document
described. Two implementations that serialise a dict differently produce
different digests for the same content and every check fails for a reason
nobody can see, so the canonical form is stated here once: sorted keys, no
inserted whitespace, UTF-8.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def digest_json(value: Any) -> str:
    """sha256 of a value in canonical JSON form."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def digest_bytes(payload: bytes) -> str:
    """sha256 of a file's contents, as written."""

    return hashlib.sha256(payload).hexdigest()
