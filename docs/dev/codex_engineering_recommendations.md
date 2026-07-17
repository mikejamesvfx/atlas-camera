# Atlas Engineering Recommendations

**Audience:** Atlas core, ComfyUI integration, DCC/export, and workflow authors  
**Date:** 2026-07-17  
**Scope:** Engineering assessment of the current Atlas ecosystem and recommended next steps

## Executive summary

Atlas has the foundations of a serious single-image projection-scene platform. Its value is the complete chain: camera recovery, metric depth, masked relief, clean-plate layers, hidden-geometry completion, viewport review, and Maya/Nuke export.

The main risk is not lack of capability. It is lack of legibility and measurable trust. Atlas can produce a convincing image while still being metrically wrong, geometrically overconfident, or difficult to reproduce. The next phase should prioritize confidence reporting, diagnostics, canonical workflows, reproducibility, and export robustness over adding more nodes.

## Current strengths

1. **Strong end-to-end architecture.** The system crosses the boundary from image inference into editable scene data and DCC handoff.
2. **Good calibration doctrine.** The distinction between assumed, depth-measured, and reference-locked scale is exactly the right conceptual model.
3. **Useful layer model.** Separating sky, clean plates, relief, and hidden geometry is more production-appropriate than treating the result as one monolithic mesh.
4. **LaRI is well-matched to architecture.** Restricted-mask hidden-geometry completion is a meaningful differentiator for urban and architectural plates.
5. **Validation tooling is strategically important.** Positional widget drift, broken links, and gate state are real ComfyUI failure modes; Atlas is correctly making them testable.
6. **DCC export is already part of the product shape.** Maya and Nuke packages make Atlas useful beyond a node-graph demo.

## Primary engineering risks

### 1. Metric scale can be visually plausible and numerically wrong

Projection is angular, so an incorrect camera height can look acceptable in a still while breaking parallax, layer spacing, and exports. Elevated viewpoints and AI-generated plates are particularly dangerous.

**Recommendation:** Make scale provenance a first-class object with:

- source: `assumed`, `depth_ground_plane`, `manual_override`, or `reference_object`
- numeric uncertainty/range
- reference ID and bounding box when applicable
- a visible “safe to export” status

### 2. Confidence is too compressed

A single confidence value cannot distinguish camera confidence from depth confidence, scale confidence, mask confidence, or hidden-geometry confidence.

**Recommendation:** expose a confidence vector:

```text
camera / focal / horizon / metric scale / depth / mask / hidden geometry / reprojection
```

Exports should carry the vector and the thresholds used for approval.

### 3. Inpainting and hidden geometry need spatial diagnostics

An inpaint can complete successfully while producing implausible geometry or covering too much of the frame.

**Recommendation:** generate debug layers for:

- observed pixels
- inferred pixels
- restricted mask
- depth confidence
- extrapolated relief
- reprojection error
- geometry outside the reliable camera frustum

These should be viewable in the Atlas viewport and included in review packages.

### 4. Workflow sprawl is becoming a usability problem

Many variants are valuable during research but make it unclear which graph a new user should trust.

**Recommendation:** maintain three canonical workflows:

- **Quickstart:** one plate to basic masked relief
- **Production:** layered relief, sky separation, scale reference, and DCC export
- **Research:** LaRI, inpainting, and advanced diagnostics

All other graphs should be labeled experimental, legacy, or implementation fixtures.

### 5. Defaults are powerful but opaque

Sky heuristics, edge extension, band priorities, clipping, and mesh smoothing materially change output quality.

**Recommendation:** every heuristic should expose:

- selected value
- reason selected
- expected failure mode
- override control

The workflow UI should show “why this value” without requiring users to inspect JSON.

### 6. Reproducibility is incomplete

Inference models, seeds, masks, thresholds, and post-processing choices need to be recoverable from an export.

**Recommendation:** introduce a versioned `Atlas Project` manifest bundling:

- source plate checksum
- model IDs and revisions
- workflow ID/version
- seeds
- scale references
- masks
- depth/geometry settings
- Atlas and ComfyUI versions
- export timestamps

## Recommended architecture work

### A. Introduce a stable project manifest

Create a small schema, for example `atlas_project.json`, that references the source image, solve, layer graph, masks, depth, and exports. Keep image-heavy data external, but make the manifest self-describing and portable.

Acceptance criteria:

- A project can be reopened from a new machine with paths remapped.
- Every exported Maya/Nuke package points back to the manifest hash.
- Schema versioning is explicit and migration tests exist.

### B. Separate inference, geometry, and presentation states

Treat these as distinct layers in the data model:

1. **Inference state:** model outputs and confidence.
2. **Geometry state:** relief, bands, masks, hidden geometry, retopology.
3. **Presentation state:** viewport, camera path, DCC exports, marketing renders.

This will reduce accidental coupling and make it possible to re-export without rerunning expensive inference.

### C. Make provenance immutable and inspectable

Every solve and export should record the exact model ID, node version, input checksum, settings, and upstream artifact IDs. Avoid relying on mutable filenames or current UI state.

### D. Add a scene-health gate before export

The gate should report pass/warn/fail for:

- camera reprojection error
- scale provenance
- mask coverage and leakage
- depth confidence
- layer overlap/ordering
- mesh holes and non-manifold edges
- hidden-geometry fraction
- DCC package completeness

The user should be able to override a warning, but not lose the warning in the exported report.

## Workflow and UX recommendations

1. Ship the three canonical workflows prominently and hide the long tail behind an “advanced/examples” section.
2. Add a first-run diagnostic card explaining scale provenance and model selection.
3. Make project-on and gray-mesh comparison a standard viewport mode, not a marketing-only convention.
4. Add a “plate suitability” preflight for sky, texture, horizon, and architectural structure.
5. Use plain language in node titles: “Measured scale,” “Assumed scale,” “Sky excluded,” and “Inferred geometry.”
6. Show the active mask and band priority directly over the viewport image.
7. Make expensive stages cacheable by artifact hash, not by workflow position alone.

## Export and DCC recommendations

Standardize every review package around the same contents:

```text
source plate
camera solve
depth map
confidence maps
layer mattes
relief mesh (OBJ + GLB)
Maya scene/scripts
Nuke script
camera path (if present)
atlas_project.json
report.md
```

Add automated import smoke tests for Maya and Nuke script syntax, path normalization, camera units, image format, and frame range. USD should be treated as a first-class export, not only an optional secondary path.

## Testing and benchmark plan

Create a small, fixed benchmark set covering:

- street-level exterior
- elevated urban birdseye
- interior architecture
- sky-dominant temple/castle
- organic ruins/canopy
- low-texture or reflective scene
- severe occlusion requiring LaRI

Track quantitative metrics:

- camera reprojection error
- focal/FOV error where ground truth exists
- scale error
- silhouette overlap
- depth discontinuity error
- mask leakage
- mesh hole count/non-manifold count
- hidden-geometry area ratio
- export/import success rate
- runtime and VRAM peak

Regression tests should include at least one intentionally bad scale reference, one comma-vs-`and` SAM prompt case, one unrestricted LaRI mask case, and one workflow widget-order mutation.

The current test suite is broad and valuable, but the example-workflow catalog guard should be made explicit about legacy fixtures or moved to a dedicated fixture directory. Temp-root setup should also avoid depending on a repository directory with fragile Windows ACLs.

## Prioritized roadmap

### P0 — reliability and trust

- Add confidence vectors and scale provenance to all solves.
- Add scene-health preflight and export report.
- Fix the example-workflow catalog/fixture boundary.
- Make temp-root handling robust on Windows.
- Add model/version/input checksums to exports.

### P1 — production usability

- Publish the three canonical workflows as the official onboarding path.
- Add project manifest and artifact caching.
- Standardize Maya/Nuke/USD review package contents.
- Add viewport overlays for masks, inferred pixels, and reprojection error.

### P2 — quality and scale

- Benchmark model selection by scene type.
- Improve mask-guided edge handling and layer priority visualization.
- Add automated mesh quality repair and reporting.
- Add camera-path baking and review renders to the canonical production workflow.

### P3 — ecosystem expansion

- Formal plugin interfaces for depth, hidden geometry, and DCC exporters.
- Remote/batch execution with reproducible manifests.
- A lightweight standalone review viewer for clients and artists without ComfyUI.

## Product position

Atlas should be positioned as a **projection-scene authoring and review system**, not merely an image-to-depth node pack. That positioning matches the actual strengths of the project and encourages the right engineering priorities: trustworthy scale, inspectable inference, editable geometry, reproducible exports, and clear handoff to production tools.

The most important strategic decision is to stop measuring progress primarily by node count or mesh resolution. The next quality milestone should be: “An artist can understand why the scene is trustworthy, correct it when it is not, and deliver the same result to Maya or Nuke without guesswork.”
