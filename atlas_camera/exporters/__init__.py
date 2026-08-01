"""DCC and package exporters.

There is deliberately NO common ``DccExporter`` Protocol here. One existed
until 2026-08-01, documented as "implemented by MayaExporter, BlenderExporter
and NukeExporter" — but only Blender was ever dispatched through it (a
one-element list in ``review_package``), while Maya and Nuke were called
directly with exporter-specific kwargs (``source_image_name``,
``relief_mesh_obj_path``, ``use_package_source``) that the Protocol's
signature could not express, and nothing outside this package ever imported
it. The writers genuinely differ in shape — USD alone emits three files — so
each is called directly by name. Add an interface back when a second
implementor actually needs to be swapped in, not before.
"""

from __future__ import annotations

from atlas_camera.exporters.review_package import build_review_package

__all__ = ["build_review_package"]
