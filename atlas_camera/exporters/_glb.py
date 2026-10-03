"""GLB (binary glTF 2.0) container framing, shared by the two GLB writers.

``relief_mesh_exporter.export_relief_mesh_glb`` (one mesh, buffer built in
memory) and ``scene_glb.write_scene_glb`` (many layers, chunks streamed) build
their glTF JSON and binary buffers differently; only the container framing --
4-byte padding, the 12-byte header and the two chunk headers -- is the same,
and it lives here once.
"""

from __future__ import annotations

import struct
from typing import BinaryIO

_GLB_MAGIC = 0x46546C67       # b"glTF"
_GLB_VERSION = 2
_CHUNK_JSON = 0x4E4F534A      # b"JSON"
_CHUNK_BIN = 0x004E4942       # b"BIN\0"


def pad4(data: bytes, pad: bytes = b"\x00") -> bytes:
    """``data`` padded to a 4-byte boundary (glTF: JSON with spaces, BIN with zeros)."""
    return data + pad * ((4 - len(data) % 4) % 4)


def write_glb_header(fh: BinaryIO, json_chunk: bytes, bin_bytes: int) -> None:
    """Write the GLB header, the whole JSON chunk and the BIN chunk header.

    The caller writes exactly ``bin_bytes`` of binary payload next.
    ``json_chunk`` and ``bin_bytes`` must already be 4-byte padded.
    """
    total = 12 + 8 + len(json_chunk) + 8 + bin_bytes
    fh.write(struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, total))     # glTF header
    fh.write(struct.pack("<II", len(json_chunk), _CHUNK_JSON))         # JSON chunk
    fh.write(json_chunk)
    fh.write(struct.pack("<II", bin_bytes, _CHUNK_BIN))                # BIN chunk
