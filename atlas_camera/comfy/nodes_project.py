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

    # `project` stays slot 0 (saved links are by slot). The STRING outputs are
    # APPENDED for custom pipes: wire the names into any node, `shot_dir` into
    # path inputs, `shot_prefix` as a filename_prefix base for core Save nodes.
    RETURN_TYPES = ("ATLAS_PROJECT", "STRING", "STRING", "STRING", "STRING", "STRING",
                    "STRING")
    RETURN_NAMES = ("project", "project_name", "shot", "colour_mode", "project_root",
                    "shot_dir", "shot_prefix")
    OUTPUT_TOOLTIPS = (
        "The delivery project; exporters with a `project` socket route into its lanes.",
        "Project name as entered.",
        "Shot name as entered.",
        "Colour mode label as chosen.",
        "Resolved project root (absolute).",
        "Absolute path of <root>/<project>/<shot>.",
        "Shot folder relative to ComfyUI's output folder (forward slashes), for a "
        "core Save node's filename_prefix, e.g. <shot_prefix>/plates/beauty; empty "
        "when the project lives outside the output folder.",
    )
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
            try:
                proj.write_manifest()
            except _project.ForeignProjectFileError as exc:
                # Refused, not overwritten (ADR-005). The project context is still valid, so
                # exports keep routing; the artist is told why the record was not updated.
                import logging

                logging.warning("atlas_project.json not updated: %s", exc)
                return {"ui": {"text": [f"atlas_project.json not updated: {exc}"]},
                        "result": _outputs(proj, colour_mode)}
        return _outputs(proj, colour_mode)


def _outputs(proj, colour_mode):
    """The node's result tuple: the project, then its pipe-friendly strings."""
    from pathlib import Path

    from atlas_camera.comfy.node_helpers import output_root

    shot_dir = Path(proj.shot_dir).resolve()
    try:
        prefix = shot_dir.relative_to(output_root()).as_posix()
    except ValueError:
        prefix = ""
    return (proj, str(proj.project), str(proj.shot), str(colour_mode),
            str(Path(proj.root).resolve()), str(shot_dir), prefix)
