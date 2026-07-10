# ATLAS

> **Vintage note (2026-07-09):** written at project start and kept as the
> north star. For what has actually shipped since, see CHANGELOG.md and the
> dated addenda in ECOSYSTEM_GUIDE.md.


## Recover the Latent World

Atlas is an open-source platform for recovering the hidden 3D structure implied
by a single image. It begins with the `LatentCamera` and grows toward a reusable
`LatentScene`: camera, depth, proxy geometry, lighting cues, semantic anchors,
projection workspaces, confidence, and production-ready exports.

Atlas does not replace artists. It accelerates the technical reconstruction
work so artists can spend more time creating.

## Philosophy

- Recover, do not invent. Infer only what the image and supplied constraints can
  reasonably support.
- Artist first. Every inferred value should be inspectable, editable, and
  exportable.
- Production ready. Outputs should fit Maya, Blender, Houdini, Nuke, USD,
  OpenCV, JSON, and other pipeline targets.
- Deterministic core. Identical inputs should produce identical outputs whenever
  practical.
- Open platform. Each subsystem should remain reusable outside the full app.

## Product Position

Atlas is not an image generator, sequence camera tracker, photogrammetry system,
depth model, or ComfyUI-only node set. Atlas can use those technologies, but its
larger purpose is to recover and package a latent scene representation from a
single image.

## Core Concepts

```text
Image
  -> LatentScene
       -> LatentCamera
       -> LatentDepth
       -> LatentGeometry
       -> LatentLighting
       -> LatentSemantics
       -> ProjectionWorkspace
```

Version 0.1 focuses on the first recoverable object: the `LatentCamera`, exposed
in code today as both `AtlasCamera` and `LatentCamera`. The current
`AtlasSolve` object is also exported as `LatentScene` so the public API can grow
without breaking existing callers.

## LatentCamera

A `LatentCamera` is the plausible virtual camera implied by an image. It may
come from a photograph, diffusion image, matte painting, concept frame, game
screenshot, historical painting, or production plate.

The supported model includes:

- Image width and height.
- Intrinsics such as focal length, film back, principal point, pixel focal
  length, lens model, and distortion metadata.
- Extrinsics such as position, orientation, world matrix, and view matrix.
- Projection evidence such as horizon and vanishing points.
- Confidence, warnings, notes, landmarks, and debug metadata.

## Recovery Pipeline

```text
input image
  -> image analysis
  -> perspective detection
  -> vanishing-point detection
  -> horizon estimation
  -> lens inference
  -> camera optimization
  -> depth and proxy-geometry hooks
  -> LatentCamera
  -> LatentScene
  -> artist workspace
  -> export
```

## Workspace Philosophy

Atlas should feel like an inspection workspace, not a black box. The user should
see confidence, assumptions, uncertainty, and editable controls for the recovered
parameters. The UI should remain minimal, dark, technical, and
confidence-driven.

The local workbench now treats 3D inspection as the primary loaded-image
surface. The source image is shown as an image plate inside a Three.js camera
lineup scene with optional grid, axes, frustum, guide lines, horizon, and proxy
objects. Artist guide tools temporarily bring the 2D draw layer forward, while
Select mode returns to orbit and inspection. This keeps drawing, solving, and
projection-prep review in one workspace instead of splitting them into separate
screens.

Proxy objects are intentionally editable but advisory. They help artists reason
about scale, floor contact, and rough projection surfaces; deterministic camera
evidence still comes from explicit line groups, scale constraints, metadata, and
future supported geometry constraints.

## Guiding Statement

Atlas is an artist-first platform for recovering hidden structure encoded within
images and translating that structure into production-ready assets for visual
effects, animation, games, and virtual production. Every recovered parameter
should be inspectable. Every inference should be measurable. Every export should
be immediately useful.
