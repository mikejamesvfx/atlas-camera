import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "AtlasCamera.Snippets",
  setup() {
    // Hook into the canvas right-click menu
    const origGetExtraMenuOptions = LGraphCanvas.prototype.getExtraMenuOptions;
    
    LGraphCanvas.prototype.getExtraMenuOptions = function(canvas, options) {
      if (origGetExtraMenuOptions) {
        origGetExtraMenuOptions.apply(this, arguments);
      }
      
      const pushSnippet = (name, nodeConfigs) => {
          return {
              content: name,
              callback: () => {
                  const mouse_x = canvas.graph_mouse[0];
                  const mouse_y = canvas.graph_mouse[1];
                  
                  const createdNodes = {};
                  
                  // Create all nodes
                  nodeConfigs.forEach((cfg) => {
                      const node = LiteGraph.createNode(cfg.type);
                      if (node) {
                          node.pos = [mouse_x + (cfg.offset?.[0] || 0), mouse_y + (cfg.offset?.[1] || 0)];
                          app.graph.add(node);
                          createdNodes[cfg.id] = node;
                      } else {
                          console.warn(`[Atlas Snippets] Could not create node type: ${cfg.type}`);
                      }
                  });
                  
                  // Wire them together
                  nodeConfigs.forEach((cfg) => {
                      if (cfg.links && createdNodes[cfg.id]) {
                          cfg.links.forEach((link) => {
                              const targetNode = createdNodes[link.targetId];
                              if (targetNode) {
                                  // Find the output slot index on source
                                  let outSlot = link.outSlot;
                                  if (typeof link.outSlotName === "string" && createdNodes[cfg.id].outputs) {
                                      outSlot = createdNodes[cfg.id].outputs.findIndex(o => o.name === link.outSlotName);
                                  }
                                  
                                  // Find the input slot index on target
                                  let inSlot = link.inSlot;
                                  if (typeof link.inSlotName === "string" && targetNode.inputs) {
                                      inSlot = targetNode.inputs.findIndex(i => i.name === link.inSlotName);
                                  }
                                  
                                  if (outSlot >= 0 && inSlot >= 0) {
                                      createdNodes[cfg.id].connect(outSlot, targetNode, inSlot);
                                  }
                              }
                          });
                      }
                  });
              }
          };
      };

      options.push({
          content: "Atlas Snippets 🧩",
          has_submenu: true,
          submenu: {
              options: [
                  pushSnippet("Clean Plate Stack (Layers)", [
                      { id: "stack", type: "AtlasCleanPlateStack", offset: [0, 0], links: [
                          { outSlotName: "clean_plate_layer", targetId: "band", inSlotName: "clean_plate_layer" }
                      ]},
                      { id: "band", type: "AtlasBoundedBand", offset: [400, 0] },
                      { id: "sky", type: "AtlasSkyDomeLayer", offset: [400, 200] },
                      { id: "export", type: "AtlasExportNukeLayers", offset: [800, 0] }
                  ]),
                  pushSnippet("Geometry Derivation (Exterior)", [
                      { id: "depth", type: "AtlasDepthMap", offset: [0, -200], links: [
                          { outSlotName: "metric_depth", targetId: "derive", inSlotName: "metric_depth" }
                      ]},
                      { id: "derive", type: "AtlasDeriveProjectionGeometry", offset: [0, 0], links: [
                          { outSlotName: "solve", targetId: "export", inSlotName: "solve" }
                      ]},
                      { id: "export", type: "AtlasExportMayaReviewScene", offset: [400, 0] }
                  ]),
                  pushSnippet("Hidden Geometry (X-Ray)", [
                      { id: "predict", type: "AtlasPredictHiddenGeometry", offset: [0, 0], links: [
                          { outSlotName: "hidden_depth", targetId: "mesh", inSlotName: "metric_depth" }
                      ]},
                      { id: "mesh", type: "AtlasDeriveReliefMesh", offset: [400, 0] }
                  ])
              ]
          }
      });
    };
  }
});
