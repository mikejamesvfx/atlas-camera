# DCC Exports

Atlas Camera treats DCC applications as adapters around the core solve.

## Maya

First pass writes a Maya Python scene-builder script:

- Estimated camera.
- Source image as a camera image plane when present.
- Y-up ground grid.
- X/Y/Z guide geometry.
- Simple proxy geometry hooks.

Raw `.ma` export is intentionally deferred. A Python scene builder is easier to
inspect, version, and adapt.

The local UI's 3D proxy objects are preview/layout state until they are promoted
to supported solve or projection-scene primitives. Exporters should continue to
prefer `solve.projection_scene.proxy_geometry` and explicit landmarks over
frontend-only `constraints.viewport3d.proxy_objects`.

## Blender

Blender is Z-up. Atlas core is Y-up.

The Blender exporter must convert coordinate conventions at the boundary. The
current script writer creates a camera and ground plane placeholder.

## Nuke

Nuke export is planned around:

- Camera node.
- Read node for source image.
- Card nodes.
- Projection setup.
- Axis/debug cards.

The current file is a placeholder script that documents the target.

## Houdini

Houdini export is not scaffolded yet. The likely path is USD-first, followed by a
native helper script if artist workflow needs it.

## USD

USD is the neutral interchange target:

- Camera.
- Image plane metadata.
- Ground plane.
- Proxy cubes / planes.
- Landmarks.
- Debug axes.

The USD dependency is optional and lazy. Missing `usd-core` should never break a
plain Atlas import.
