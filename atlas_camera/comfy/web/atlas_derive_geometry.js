/**
 * Atlas Derive Projection Geometry — scene_type preset widget visibility.
 *
 * The original complaint this fixes: `primitive_method` (and several other
 * widgets) are silently ignored depending on `geometry_mode`/`scene_type`,
 * with no visual sign of it — you had to read a tooltip to learn that. This
 * hides whichever widgets a chosen scene_type preset has already decided
 * (or made moot), so what's left visible is always genuinely adjustable.
 *
 * Python (nodes.py's derive()) remains the single source of truth for what
 * actually happens — this is a pure UI convenience layered on top, never a
 * new decision path. If a widget IS visible, changing it has a real effect;
 * if it's hidden, Python is already overriding or ignoring it.
 */
import { app } from "../../scripts/app.js";

// Mirrors nodes.py's AtlasDeriveProjectionGeometry._SCENE_TYPE_PRESETS —
// keep the two in sync. Only used to decide which widgets a given preset
// makes moot; the actual override logic still lives entirely in Python.
const SCENE_TYPE_PRESETS = {
  organic: { geometry_mode: "relief_mesh" },
  mountains: { geometry_mode: "relief_mesh", relief_quality: "high" },
  forests: { geometry_mode: "relief_mesh", relief_quality: "high", depth_edge_rel: 1.0 },
  aerial: {
    geometry_mode: "both", primitive_method: "azimuth_walls",
    relief_quality: "medium", max_objects: 6,
  },
  indoor: {
    geometry_mode: "primitives", primitive_method: "room_cuboid",
    depth_model: "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
  },
  outdoor: {
    geometry_mode: "primitives", primitive_method: "ransac_planes",
    depth_model: "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
  },
  simple_walls: { geometry_mode: "primitives", primitive_method: "azimuth_walls" },
  towers_spires: { geometry_mode: "primitives", primitive_method: "vertical_extrusion" },
};

const ALL_PRESET_CONTROLLED_WIDGETS = [
  "geometry_mode", "primitive_method", "relief_grid", "relief_quality",
  "depth_edge_rel", "max_objects", "max_walls", "depth_model",
];

// Which widgets become moot once `sceneType` is picked — always the two at
// the root of the original bug (geometry_mode/primitive_method, for every
// non-manual preset) plus whatever that specific preset explicitly sets,
// plus whatever a *resolved* geometry_mode structurally rules out even when
// the preset itself doesn't mention it: relief_grid/relief_quality/
// depth_edge_rel do nothing when geometry_mode=primitives (no relief mesh
// ever builds); max_objects/max_walls do nothing when geometry_mode=
// relief_mesh (foreground objects/walls are derived but then discarded,
// only the backdrop survives — see derive()'s `keep` filtering). Also,
// relief_quality always overrides relief_grid whenever it's set to anything
// but "custom", so locking relief_quality locks relief_grid too.
function computeHiddenWidgets(sceneType) {
  const preset = SCENE_TYPE_PRESETS[sceneType];
  if (!preset) return new Set(); // "manual" (or an unknown value) — nothing hidden

  const hidden = new Set(["geometry_mode", "primitive_method", ...Object.keys(preset)]);
  if (hidden.has("relief_quality")) hidden.add("relief_grid");
  if (preset.geometry_mode === "primitives") {
    hidden.add("relief_grid");
    hidden.add("relief_quality");
    hidden.add("depth_edge_rel");
  }
  if (preset.geometry_mode === "relief_mesh") {
    hidden.add("max_objects");
    hidden.add("max_walls");
  }
  return hidden;
}

// Standard ComfyUI/litegraph widget-hide trick: swap `.type` to something the
// canvas draw loop doesn't recognise (so it's skipped) and make
// `computeSize()` report a collapsed [width, -4] — the same convention
// ComfyUI's own core widgets use for zero-height. Restores the original type/
// computeSize on show, so the widget behaves exactly as before once visible
// again. Does not touch `.value` — serialization/queueing are unaffected.
function setWidgetHidden(node, widgetName, hide) {
  const widget = node.widgets?.find((w) => w.name === widgetName);
  if (!widget) return;
  if (hide) {
    if (!widget._atlasHidden) {
      widget._atlasOrigType = widget.type;
      widget._atlasOrigComputeSize = widget.computeSize;
      widget.type = "atlas_hidden";
      widget.computeSize = () => [0, -4];
      widget._atlasHidden = true;
    }
  } else if (widget._atlasHidden) {
    widget.type = widget._atlasOrigType;
    widget.computeSize = widget._atlasOrigComputeSize;
    delete widget._atlasOrigType;
    delete widget._atlasOrigComputeSize;
    delete widget._atlasHidden;
  }
}

function applySceneTypeVisibility(node) {
  const sceneTypeWidget = node.widgets?.find((w) => w.name === "scene_type");
  if (!sceneTypeWidget) return;
  const hidden = computeHiddenWidgets(sceneTypeWidget.value);
  ALL_PRESET_CONTROLLED_WIDGETS.forEach((name) => setWidgetHidden(node, name, hidden.has(name)));
  node.setSize(node.computeSize());
  node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "AtlasCamera.DeriveGeometryPresets",

  async nodeCreated(node) {
    if (node.comfyClass !== "AtlasDeriveProjectionGeometry") return;
    // Wait one tick for ComfyUI to finish building the node's widgets.
    await new Promise((r) => setTimeout(r, 0));

    const sceneTypeWidget = node.widgets?.find((w) => w.name === "scene_type");
    if (!sceneTypeWidget) return;

    const prevCallback = sceneTypeWidget.callback;
    sceneTypeWidget.callback = function (...args) {
      prevCallback?.apply(this, args);
      applySceneTypeVisibility(node);
    };

    // Initial state — also handles a saved workflow loading with a non-
    // "manual" scene_type already selected.
    applySceneTypeVisibility(node);
  },
});
