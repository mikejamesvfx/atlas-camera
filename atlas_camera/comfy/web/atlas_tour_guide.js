import { app } from "../../scripts/app.js";

const TOUR_STEPS = [
    {
        title: "Step 1: Input",
        description: "Start by providing an image to the pipeline.",
        nodeType: "AtlasInput"
    },
    {
        title: "Step 2: Solve",
        description: "The solver recovers the physical camera properties (focal length, height).",
        nodeType: "AtlasLearnedSolveFromImage" // Or AtlasMegaPipeline
    },
    {
        title: "Step 3: Depth",
        description: "Metric depth is extracted to build geometry.",
        nodeType: "AtlasDepthMap"
    },
    {
        title: "Step 4: Geometry",
        description: "The pipeline derives 3D geometry from the depth and camera.",
        nodeType: "AtlasDeriveProjectionGeometry"
    },
    {
        title: "Step 5: Export",
        description: "Finally, the scene is exported to a Maya/USD package.",
        nodeType: "AtlasExportMayaReviewScene"
    }
];

class AtlasTourGuide {
    constructor() {
        this.currentStep = 0;
        this.activeNode = null;
        this.originalColors = {};
        
        this.createPanel();
    }
    
    createPanel() {
        this.panel = document.createElement("div");
        Object.assign(this.panel.style, {
            position: "fixed",
            bottom: "20px",
            right: "20px",
            width: "300px",
            background: "rgba(20, 20, 25, 0.95)",
            border: "1px solid #445",
            borderRadius: "8px",
            padding: "16px",
            color: "#fff",
            fontFamily: "sans-serif",
            zIndex: 9999,
            display: "none",
            boxShadow: "0 4px 12px rgba(0,0,0,0.5)"
        });
        
        this.titleEl = document.createElement("h3");
        this.titleEl.style.margin = "0 0 10px 0";
        this.panel.appendChild(this.titleEl);
        
        this.descEl = document.createElement("p");
        this.descEl.style.fontSize = "14px";
        this.descEl.style.margin = "0 0 15px 0";
        this.panel.appendChild(this.descEl);
        
        const btnRow = document.createElement("div");
        btnRow.style.display = "flex";
        btnRow.style.justifyContent = "space-between";
        
        this.prevBtn = document.createElement("button");
        this.prevBtn.innerText = "← Prev";
        this.prevBtn.onclick = () => this.prevStep();
        
        this.nextBtn = document.createElement("button");
        this.nextBtn.innerText = "Next →";
        this.nextBtn.onclick = () => this.nextStep();
        
        this.closeBtn = document.createElement("button");
        this.closeBtn.innerText = "Close Tour";
        this.closeBtn.onclick = () => this.stopTour();
        
        btnRow.appendChild(this.prevBtn);
        btnRow.appendChild(this.closeBtn);
        btnRow.appendChild(this.nextBtn);
        
        this.panel.appendChild(btnRow);
        document.body.appendChild(this.panel);
    }
    
    startTour() {
        if (!app.graph) return;
        this.currentStep = 0;
        this.panel.style.display = "block";
        this.updateStep();
    }
    
    stopTour() {
        this.panel.style.display = "none";
        this.clearHighlight();
    }
    
    nextStep() {
        if (this.currentStep < TOUR_STEPS.length - 1) {
            this.currentStep++;
            this.updateStep();
        }
    }
    
    prevStep() {
        if (this.currentStep > 0) {
            this.currentStep--;
            this.updateStep();
        }
    }
    
    clearHighlight() {
        if (this.activeNode && this.originalColors.color !== undefined) {
            this.activeNode.color = this.originalColors.color;
            this.activeNode.bgcolor = this.originalColors.bgcolor;
            this.activeNode = null;
            app.canvas.setDirty(true, true);
        }
    }
    
    highlightNode(nodeType) {
        this.clearHighlight();
        
        // Find the first node of this type
        const node = app.graph.findNodesByType(nodeType)[0];
        if (node) {
            this.activeNode = node;
            this.originalColors = { color: node.color, bgcolor: node.bgcolor };
            
            // Set to a glowing yellow
            node.color = "#cc9900";
            node.bgcolor = "#554400";
            
            // Center the canvas on this node
            const canvas = app.canvas;
            if (canvas && canvas.ds) {
                // Approximate centering
                canvas.ds.offset[0] = -node.pos[0] + (canvas.canvas.width / 2 / canvas.ds.scale);
                canvas.ds.offset[1] = -node.pos[1] + (canvas.canvas.height / 2 / canvas.ds.scale);
            }
            
            app.canvas.setDirty(true, true);
        }
    }
    
    updateStep() {
        const step = TOUR_STEPS[this.currentStep];
        this.titleEl.innerText = step.title;
        this.descEl.innerText = step.description;
        
        this.prevBtn.disabled = this.currentStep === 0;
        this.nextBtn.disabled = this.currentStep === TOUR_STEPS.length - 1;
        
        // Also check if MegaPipeline is being used to adapt the tour
        let targetType = step.nodeType;
        if (app.graph.findNodesByType("AtlasMegaPipeline").length > 0) {
            targetType = "AtlasMegaPipeline";
        }
        
        this.highlightNode(targetType);
    }
}

app.registerExtension({
  name: "AtlasCamera.TourGuide",
  setup() {
      const guide = new AtlasTourGuide();
      
      // Inject a "Start Tour" button into the main menu
      const origMenu = LGraphCanvas.prototype.getExtraMenuOptions;
      LGraphCanvas.prototype.getExtraMenuOptions = function(canvas, options) {
        if (origMenu) {
            origMenu.apply(this, arguments);
        }
        options.push({
            content: "🧭 Start Atlas Tour",
            callback: () => guide.startTour()
        });
      };
  }
});
