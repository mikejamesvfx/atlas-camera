"""
LatentScene — the scene-level container for recovered objects.

v0.1 holds exactly one component: `camera`. LatentDepth, LatentGeometry,
etc. are roadmap (DECISIONS.md, scope ruling) and will attach to this
same scene later without changing the pattern below.

`scene.export` is a thin orchestrator (DECISIONS.md §7): it does not know
about file formats. It walks the scene's RecoveredObject components and
calls each one's own `to_<format>()`. A new RecoveredObject subclass that
defines `to_maya()` is picked up automatically — nothing here needs to
change to support it.
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas.core.latent_camera import LatentCamera


class _ExportOrchestrator:
    """`scene.export.maya()` etc. — delegates to each component's own
    export method rather than implementing format logic itself."""

    def __init__(self, scene: "LatentScene") -> None:
        self._scene = scene

    def maya(self) -> str:
        # v0.1: camera is the only exportable component. Future
        # components contribute additional .ma fragments here without
        # this method needing format-specific knowledge of them.
        return self._scene.camera.to_maya()

    def json(self) -> str:
        return self._scene.camera.to_json()


@dataclass
class LatentScene:
    camera: LatentCamera

    def __post_init__(self) -> None:
        self.export = _ExportOrchestrator(self)
