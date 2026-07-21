/** Render AtlasAssessOutput's ui.text report directly on the terminal node. */
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
  name: "AtlasCamera.AssessOutput",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AtlasAssessOutput") return;
    const original = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      original?.apply(this, arguments);
      const text = Array.isArray(message?.text) ? message.text.join("\n") : message?.text;
      if (!text) return;
      let widget = this.widgets?.find((item) => item.name === "output_assessment_report");
      if (!widget) {
        widget = ComfyWidgets.STRING(
          this, "output_assessment_report", ["STRING", { multiline: true }], app
        ).widget;
        widget.inputEl.readOnly = true;
        widget.inputEl.style.opacity = 0.88;
        widget.inputEl.style.fontFamily = "monospace";
        widget.serialize = false;
      }
      widget.value = text;
      this.setSize([Math.max(this.size[0], 470), Math.max(this.size[1], 430)]);
      this.graph?.setDirtyCanvas(true, true);
    };
  },
});
