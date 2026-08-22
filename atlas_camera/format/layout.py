"""The directory a `.atlas` package is.

`scene.json` is the only authoritative document; everything else is content it
points at, by package-relative path. A package that names a file outside itself
is not portable, and a scene that cannot be moved is a note about a scene.
"""

from __future__ import annotations

SCENE_DOCUMENT = "scene.json"

GEOMETRY_DIR = "geometry"      # meshes an entity's geometry record names
IMAGERY_DIR = "imagery"        # source plate, undistorted plate, previews
MATTES_DIR = "mattes"          # one file per layer
TEXTURES_DIR = "textures"      # baked UDIM tiles
VISIBILITY_DIR = "visibility"  # samples.npz and its summary
ATLAS_DIR = "atlas"            # the solve this package was produced from
HISTORY_DIR = "history"        # the append-only ledger

PACKAGE_DIRS = (
    GEOMETRY_DIR,
    IMAGERY_DIR,
    MATTES_DIR,
    TEXTURES_DIR,
    VISIBILITY_DIR,
    ATLAS_DIR,
    HISTORY_DIR,
)
