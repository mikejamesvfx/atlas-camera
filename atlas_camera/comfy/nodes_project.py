"""ComfyUI node: set the Atlas delivery project once, thread it into exports.

Emits an ``ATLAS_PROJECT`` context (``atlas_camera.core.project.AtlasProject``)
that the export nodes read, so a user names the project/shot and colour lane in
one place instead of wiring an output path into every export.
"""
from __future__ import annotations

from atlas_camera.core import project as _project

# Dropdown labels deliberately keep OCIO / ACES vocabulary out of the default
# lane. RAW is the input that forces the choice (the file can't say whether it's
# a VFX plate or a print job), so the mode is stated, never inferred.
_MODE_LABELS = {
    "Standard (sRGB)": _project.MODE_STANDARD,
    "VFX (ACEScg / float)": _project.MODE_VFX,
}
_MODE_CHOICES = list(_MODE_LABELS)


def _default_output_root():
    """ComfyUI's output dir when running inside Comfy, else None so the core
    falls back to a clear location. Imported lazily so tests need no Comfy."""
    try:
        import folder_paths  # type: ignore

        return folder_paths.get_output_directory()
    except Exception:
        return None


class AtlasProject:
    """Name the project, shot and colour lane once; exports route into it.

    The graph face of ``atlas_camera.core.project.AtlasProject``; the two share a
    name across the comfy/core layers on purpose. Colour mode is the gate that
    keeps OCIO/ACES away from users who don't want it: Standard delivers sRGB and
    never mentions colour management, VFX opens the managed ACEScg lane.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project": ("STRING", {"default": "untitled_project"}),
                "shot": ("STRING", {"default": "shot010"}),
                "colour_mode": (_MODE_CHOICES, {"default": _MODE_CHOICES[0]}),
            },
            "optional": {
                "project_root": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Empty uses ComfyUI's output folder (or "
                        "$ATLAS_PROJECT_ROOT). Set an absolute path to route "
                        "the project elsewhere.",
                    },
                ),
                "create_tree": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("ATLAS_PROJECT",)
    RETURN_NAMES = ("project",)
    FUNCTION = "build"
    # Stamped by the central MENU_CATEGORY map at import; placeholder only.
    CATEGORY = "Atlas"

    def build(self, project, shot, colour_mode, project_root="", create_tree=True):
        mode = _MODE_LABELS.get(colour_mode, _project.MODE_STANDARD)
        proj = _project.build_project(
            project_root,
            project,
            shot,
            mode,
            default_root=_default_output_root(),
        )
        if create_tree:
            proj.ensure_tree()
            proj.write_manifest()
        return (proj,)
