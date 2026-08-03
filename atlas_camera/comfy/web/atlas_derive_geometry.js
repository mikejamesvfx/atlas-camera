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
  // indoor/outdoor use the zero-extra-install V2 metric models (A/B 2026-07-13
  // reverted the 2026-07-09 DA3 default) — mirrors nodes.py's _SCENE_TYPE_PRESETS
  // (keep in sync by hand).
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

// ComfyUI currently has two widget renderers: LiteGraph reads `widget.hidden`
// while the Vue renderer reads `widget.options.hidden`.  Set both flags as
// well as using the legacy zero-height/type convention.  Setting only the
// latter collapses layout but can leave the value painted on top of the next
// row (the apparent "widget drift" seen in saved workflows).  Restore every
// original property exactly when showing the widget again.  `.value` is never
// touched, so serialization and API queueing remain positional and stable.
function setWidgetHidden(node, widgetName, hide) {
  const widget = node.widgets?.find((w) => w.name === widgetName);
  if (!widget) return;
  if (hide) {
    if (!widget._atlasHidden) {
      widget._atlasOrigType = widget.type;
      widget._atlasOrigComputeSize = widget.computeSize;
      widget._atlasHadHidden = Object.prototype.hasOwnProperty.call(widget, "hidden");
      widget._atlasOrigHidden = widget.hidden;
      widget._atlasHadOptionsHidden = !!widget.options
        && Object.prototype.hasOwnProperty.call(widget.options, "hidden");
      widget._atlasOrigOptionsHidden = widget.options?.hidden;
      widget.type = "atlas_hidden";
      widget.computeSize = () => [0, -4];
      widget.hidden = true;
      if (widget.options) widget.options.hidden = true;
      widget._atlasHidden = true;
    }
  } else if (widget._atlasHidden) {
    widget.type = widget._atlasOrigType;
    widget.computeSize = widget._atlasOrigComputeSize;
    if (widget._atlasHadHidden) widget.hidden = widget._atlasOrigHidden;
    else delete widget.hidden;
    if (widget.options) {
      if (widget._atlasHadOptionsHidden) {
        widget.options.hidden = widget._atlasOrigOptionsHidden;
      } else {
        delete widget.options.hidden;
      }
    }
    delete widget._atlasOrigType;
    delete widget._atlasOrigComputeSize;
    delete widget._atlasHadHidden;
    delete widget._atlasOrigHidden;
    delete widget._atlasHadOptionsHidden;
    delete widget._atlasOrigOptionsHidden;
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
