"""Headless-Blender geometry recipes for Atlas.

Layering: this package may import `core`; NOTHING here imports `comfy`, and
`comfy` imports this. Same position as `atlas_camera/inference/`.

`recipes/` is package DATA, not a package — those files `import bpy`, which only
exists inside Blender. Keeping them un-importable is what stops a module sweep
or pytest collection from choking on them.

Blender is an OPTIONAL external install. Nothing here is imported at Atlas
startup; a node calls in and gets an instructive RuntimeError if Blender is
absent.
"""
from atlas_camera.blender.convert import atlas_to_blender, blender_to_atlas
from atlas_camera.blender.exchange import read_result, write_exchange
from atlas_camera.blender.organic_fill import (
    gate_movement,
    median_edge_length,
    shrinkwrap_patch,
    weld_to_anchor,
)
from atlas_camera.blender.region import compact, select_torn_collar
from atlas_camera.blender.runner import (
    BLENDER_PATH_ENV,
    MIN_BLENDER,
    build_blender_command,
    recipes_dir,
    require_blender,
    resolve_blender_exe,
    run_recipe,
)

__all__ = [
    "BLENDER_PATH_ENV",
    "MIN_BLENDER",
    "atlas_to_blender",
    "blender_to_atlas",
    "build_blender_command",
    "compact",
    "gate_movement",
    "median_edge_length",
    "read_result",
    "recipes_dir",
    "require_blender",
    "resolve_blender_exe",
    "run_recipe",
    "shrinkwrap_patch",
    "select_torn_collar",
    "weld_to_anchor",
    "write_exchange",
]
