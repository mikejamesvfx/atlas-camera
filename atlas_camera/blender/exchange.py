"""The Atlas <-> Blender wire format: NPZ arrays plus a JSON parameter block.

WHY NOT OBJ OR GLB, both of which Atlas can already write. Atlas has writers and
no readers for either, so one would have to be written — and worse, Blender's
importers for both MERGE duplicate positions and SPLIT vertices at UV seams.
That destroys the 1:1 vertex<->UV index mapping `ReliefMesh` treats as a
contract and `regenerate_projective_uvs` assumes. `from_pydata` + `foreach_get`
keeps Atlas owning vertex order end to end, which no importer can promise.

WHY NOT A GENERATED .py, which is what `blender_exporter` does today: at ultra
relief quality that is a repr of ~780K vertex tuples in one source file for
Blender's parser to tokenize. Fine for a one-off review scene, pathological per
execution.

NO UVs CROSS THE PIPE. The recipe does not need them and Atlas regenerates them
for the returned vertices anyway, which deletes the largest remaining convention
hazard (glTF's top-left origin vs OBJ's bottom-left) from the design entirely.

Verified live: Blender 5.2's bundled Python is 3.13 with numpy 2.3.4, so both
ends of this pipe are numpy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atlas_camera.blender.convert import atlas_to_blender, blender_to_atlas


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "atlas_camera.blender requires numpy. Install with: "
            "pip install -e .[vision]") from exc
    return np


def write_exchange(exchange_dir: str | Path, *, patch_vertices: Any,
                   patch_faces: Any, target_vertices: Any, target_faces: Any,
                   camera_position: Any, params: dict[str, Any]) -> Path:
    """Write `in.npz` + `params.json`. Vertices go in ATLAS space; converted here.

    Conversion happens at this boundary, once, so a caller never has to remember
    which space it is holding.
    """
    np = _require_numpy()
    out = Path(exchange_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "in.npz",
        patch_vertices=atlas_to_blender(patch_vertices),
        patch_faces=np.asarray(patch_faces, dtype=np.int32).reshape(-1, 3),
        target_vertices=atlas_to_blender(target_vertices),
        target_faces=np.asarray(target_faces, dtype=np.int32).reshape(-1, 3),
        camera_position=atlas_to_blender(
            np.asarray(camera_position, dtype=np.float64).reshape(1, 3))[0],
    )
    (out / "params.json").write_text(
        json.dumps(dict(params), indent=1), encoding="utf-8")
    return out


def read_result(exchange_dir: str | Path) -> dict[str, Any]:
    """Read `out.npz` back into ATLAS space, with the shape checks up front.

    Raises rather than returning something half-formed: a caller that receives
    NaN vertices will write them into the solve, and the failure surfaces much
    later as geometry in the wrong place.
    """
    np = _require_numpy()
    src = Path(exchange_dir) / "out.npz"
    if not src.is_file():
        raise RuntimeError(
            f"recipe produced no {src.name} — it exited cleanly but wrote "
            "nothing. Check report.json for the stage it reached.")
    with np.load(src) as data:
        missing = [k for k in ("vertices", "faces") if k not in data]
        if missing:
            raise RuntimeError(f"{src.name} is missing {missing}")
        verts = np.asarray(data["vertices"], dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(data["faces"], dtype=np.int64).reshape(-1, 3)
        snapped = (np.asarray(data["snapped"], dtype=bool).ravel()
                   if "snapped" in data else np.zeros(len(verts), dtype=bool))

    if not len(faces):
        raise RuntimeError(
            "recipe produced no geometry (0 faces). The tear is probably "
            "smaller than voxel_size_m — lower it, or widen collar_rings.")
    if not np.isfinite(verts).all():
        raise RuntimeError(
            f"recipe returned {int((~np.isfinite(verts)).any(axis=1).sum())} "
            "non-finite vertices; refusing to write them into the solve.")
    if faces.max() >= len(verts) or faces.min() < 0:
        raise RuntimeError(
            f"face indices out of range: [{faces.min()}, {faces.max()}] for "
            f"{len(verts)} vertices")
    if len(snapped) != len(verts):
        snapped = np.zeros(len(verts), dtype=bool)

    return {"vertices": blender_to_atlas(verts), "faces": faces,
            "snapped": snapped}
