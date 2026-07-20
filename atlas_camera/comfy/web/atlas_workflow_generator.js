import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "AtlasCamera.WorkflowGenerator",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AtlasWorkflowGenerator") return;
    
    const orig = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      orig?.apply(this, arguments);
      
      const recipeCombo = this.addWidget("combo", "Recipe", "Default", () => {}, { values: ["Default", "Mega-Node"] });
      
      const btn = this.addWidget("button", "Generate Workflow", null, async () => {
        try {
            const recipe = recipeCombo.value || "default";
            const res = await fetch(`/atlas/recipes/${recipe}`);
            if (!res.ok) {
                console.error("[Atlas Generator] Failed to fetch recipe:", res.status);
                return;
            }
            const graphJson = await res.json();
            
            // Clear current graph and load the new one
            app.graph.clear();
            app.loadGraphData(graphJson);
            
            // Adjust zoom to fit the new graph
            app.canvas.ds.offset[0] = 0;
            app.canvas.ds.offset[1] = 0;
            app.canvas.setDirty(true, true);
        } catch (err) {
            console.error("[Atlas Generator] Error:", err);
        }
      });
      btn.serialize = false;
      recipeCombo.serialize = false;
      
      this.setSize([300, 150]);
    };
  },
});
