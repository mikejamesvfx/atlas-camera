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


# ---------------------------------------------------------------------------
# Scene seed (Atlas -> Blender) and multi-mesh result (Blender -> Atlas)
# ---------------------------------------------------------------------------
# The measured-primitives bridge (2026-08-16). A SEED is everything Blender
# needs to model against the metric solve: the recovered camera, every existing
# proxy primitive tessellated (reference only), the viewport-drawn shapes, and
# the measured quantities. Meshes come BACK as `out_meshes.npz`, several per
# run, no UVs — Atlas regenerates projective UVs for the recovered camera on
# import, so glTF-vs-OBJ UV-origin hazards never enter the pipe (same reasoning
# as `write_exchange` above).

SEED_NPZ = "seed.npz"
SEED_JSON = "seed.json"
OUT_MESHES_NPZ = "out_meshes.npz"
OUT_MESHES_JSON = "out_meshes.json"

#: Format identity of the exchange directory, independent of the SOLVE
#: identity that `solve_seed_fingerprint` covers. The two catch different
#: failures: the fingerprint refuses a seed built from a different camera,
#: this refuses a seed written by a pack that laid the files out differently.
#: The round trip is explicitly allowed to span sessions and the directory
#: lives in a shot's `blender/` lane, so an exchange written weeks ago is a
#: normal input — and a silent misread of one is exactly the failure
#: `MIN_BLENDER` refuses on the Blender side.
#:
#: Bump when a key changes meaning, moves, or disappears. Adding an OPTIONAL
#: key readers already tolerate is not a bump.
EXCHANGE_VERSION = 1
EXCHANGE_VERSION_KEY = "atlas_exchange_version"


def check_exchange_version(seed: dict[str, Any], *, where: str = "seed.json") -> None:
    """Raise unless `seed` carries a format version this build understands.

    A seed with NO version predates versioning (pre-2026-08-17) — refused with
    the same message, because that is precisely the layout drift this guards.
    """
    got = seed.get(EXCHANGE_VERSION_KEY)
    if got == EXCHANGE_VERSION:
        return
    raise RuntimeError(
        f"{where}: exchange format version {got!r}, this build writes and reads "
        f"{EXCHANGE_VERSION}. Re-run the massing node to rewrite the exchange "
        "directory; do not hand-edit it."
    )


def _check_mesh(verts: Any, faces: Any, *, label: str) -> None:
    """Shape/finite/index checks shared by every reader. Raises RuntimeError."""
    np = _require_numpy()
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise RuntimeError(f"{label}: vertices must be (N,3); got {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"{label}: faces must be (M,3) triangles; got {faces.shape}")
    if not len(faces):
        raise RuntimeError(f"{label}: 0 faces")
    if not np.isfinite(verts).all():
        raise RuntimeError(
            f"{label}: {int((~np.isfinite(verts)).any(axis=1).sum())} "
            "non-finite vertices; refusing to write them into the solve.")
    if faces.max() >= len(verts) or faces.min() < 0:
        raise RuntimeError(
            f"{label}: face indices out of range [{faces.min()}, {faces.max()}] "
            f"for {len(verts)} vertices")


def write_scene_seed(exchange_dir: str | Path, *, camera: dict[str, Any],
                     primitives: list[dict[str, Any]],
                     drawn_shapes: list[dict[str, Any]] | None = None,
                     params: dict[str, Any] | None = None,
                     cloud: Any = None) -> Path:
    """Write `seed.npz` + `seed.json`. Everything arrives in ATLAS space.

    ``camera``: ``{"view_matrix": 4x4 world->cam (row-major), "fx", "fy",
    "cx", "cy", "image_width", "image_height"}``. Converted here to a Blender
    world matrix (Z-up) so the recipe places a camera object with no axis
    knowledge of its own.

    ``primitives``: ``[{"name", "source", "vertices": (N,3), "faces": (M,3),
    ...scalar tags}]`` — tessellated in world space by the caller (use
    ``core.primitive_mesh.tessellate_primitive``). Arrays land in the NPZ as
    ``prim_{i}_vertices`` / ``prim_{i}_faces``; the tags land in JSON.

    ``drawn_shapes``: the viewport records ``{id, label, kind, points_world}``,
    converted to Blender axes in JSON (they are small).

    ``cloud``: optional (N,3) Atlas-world metric points (the sky-free MoGe
    measurement) → ``cloud_points`` in the NPZ, Blender axes; the recipe
    builds a vertex-only object from it for snapping/reference.
    """
    np = _require_numpy()
    out = Path(exchange_dir)
    out.mkdir(parents=True, exist_ok=True)

    vm = np.asarray(camera["view_matrix"], dtype=np.float64).reshape(4, 4)
    c2w = np.linalg.inv(vm)
    T = np.asarray(((1, 0, 0), (0, 0, -1), (0, 1, 0)), dtype=np.float64)
    # Atlas camera looks down its own -Z with +Y up. Blender's camera object
    # ALSO looks down its local -Z with +Y up, so the camera-local frame needs
    # no change: only the WORLD frame rotates by T. matrix_world = T . c2w.
    T4 = np.eye(4)
    T4[:3, :3] = T
    matrix_world = T4 @ c2w

    arrays: dict[str, Any] = {}
    prim_json: list[dict[str, Any]] = []
    for i, prim in enumerate(primitives or []):
        v = np.asarray(prim["vertices"], dtype=np.float64).reshape(-1, 3)
        f = np.asarray(prim["faces"], dtype=np.int64).reshape(-1, 3)
        arrays[f"prim_{i}_vertices"] = atlas_to_blender(v)
        arrays[f"prim_{i}_faces"] = f.astype(np.int32)
        tags = {k: val for k, val in prim.items()
                if k not in ("vertices", "faces")
                and (val is None or isinstance(val, (str, int, float, bool)))}
        tags["index"] = i
        tags["n_vertices"] = int(len(v))
        tags["n_faces"] = int(len(f))
        prim_json.append(tags)
    n_cloud = 0
    if cloud is not None:
        c = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
        c = c[np.isfinite(c).all(axis=1)]
        n_cloud = int(len(c))
        arrays["cloud_points"] = atlas_to_blender(c).astype(np.float32)
    if not arrays:
        # np.savez with no arrays still writes a valid (empty) archive.
        arrays["_empty"] = np.zeros((0,), dtype=np.float64)
    np.savez(out / SEED_NPZ, **arrays)

    shapes_json: list[dict[str, Any]] = []
    for rec in (drawn_shapes or []):
        pts = np.asarray(rec.get("points_world") or [], dtype=np.float64).reshape(-1, 3)
        shapes_json.append({
            "id": rec.get("id"), "label": rec.get("label"),
            "kind": rec.get("kind"), "enabled": bool(rec.get("enabled", True)),
            "points_blender": atlas_to_blender(pts).tolist() if len(pts) else [],
        })

    seed = {
        "camera": {
            "matrix_world_blender": matrix_world.tolist(),
            "fx": float(camera["fx"]), "fy": float(camera["fy"]),
            "cx": float(camera["cx"]), "cy": float(camera["cy"]),
            "image_width": int(camera["image_width"]),
            "image_height": int(camera["image_height"]),
        },
        EXCHANGE_VERSION_KEY: EXCHANGE_VERSION,
        "primitives": prim_json,
        "drawn_shapes": shapes_json,
        "n_cloud_points": n_cloud,
        "params": dict(params or {}),
        "axes": "Blender Z-up; T rows (1,0,0),(0,0,-1),(0,1,0) applied to Atlas Y-up",
    }
    (out / SEED_JSON).write_text(json.dumps(seed, indent=1), encoding="utf-8")
    return out


def read_meshes(exchange_dir: str | Path) -> dict[str, Any]:
    """Read `out_meshes.npz` (+ `out_meshes.json`) back into ATLAS space.

    Returns ``{"meshes": [{"name", "vertices", "faces", **tags}], "info": {...}}``.
    Malformed archives raise; a mesh that individually fails the checks is
    returned under ``"rejected"`` with its reason so the node can report rather
    than silently drop it.
    """
    np = _require_numpy()
    base = Path(exchange_dir)
    src = base / OUT_MESHES_NPZ
    if not src.is_file():
        raise RuntimeError(
            f"no {src.name} in {base} — the recipe (or the in-Blender export "
            "script) wrote nothing. Check report.json / error.json.")
    # The meshes came out of a scene this build seeded, so the seed's format
    # version covers them. Checked here rather than only at write time because
    # a shot's blender/ lane can hold an exchange dir from an older pack.
    seed_path = base / SEED_JSON
    if seed_path.is_file():
        try:
            seed_head = json.loads(seed_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            seed_head = {}
        if isinstance(seed_head, dict):
            check_exchange_version(seed_head, where=str(seed_path))
    info: dict[str, Any] = {}
    meta_path = base / OUT_MESHES_JSON
    if meta_path.is_file():
        try:
            info = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            info = {}
    tags_by_index = {int(m.get("index", i)): m
                     for i, m in enumerate(info.get("meshes") or [])}

    meshes: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    with np.load(src) as data:
        keys = set(data.keys())
        indices = sorted({int(k.split("_")[1]) for k in keys
                          if k.startswith("mesh_") and k.endswith("_vertices")})
        for i in indices:
            vk, fk = f"mesh_{i}_vertices", f"mesh_{i}_faces"
            tags = dict(tags_by_index.get(i, {}))
            name = str(tags.pop("name", None) or f"blender_mesh_{i:02d}")
            if fk not in keys:
                rejected.append({"name": name, "reason": f"missing {fk}"})
                continue
            verts = np.asarray(data[vk], dtype=np.float64).reshape(-1, 3)
            faces = np.asarray(data[fk], dtype=np.int64).reshape(-1, 3)
            try:
                _check_mesh(verts, faces, label=name)
            except RuntimeError as exc:
                rejected.append({"name": name, "reason": str(exc)})
                continue
            tags.pop("index", None)
            meshes.append({"name": name, "vertices": blender_to_atlas(verts),
                           "faces": faces, **tags})
    return {"meshes": meshes, "rejected": rejected, "info": info}
