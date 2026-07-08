/**
 * Atlas Assess Image — the ▶ Continue Workflow button.
 *
 * AtlasAssessImage pauses everything downstream of the photo (its image
 * output returns ExecutionBlocker) until `proceed` is turned on, so the
 * first Queue costs only the VLM assessment. This button is the one-click
 * resume: it sets the `proceed` widget and re-queues, mirroring how the
 * viewport's 📐 Extract Angle button resumes the patch branch. Pure UI
 * convenience — toggling the widget by hand and hitting Queue does exactly
 * the same thing (this file failing to load never blocks the pipeline).
 */
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
  name: "AtlasCamera.AssessImage",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AtlasAssessImage") return;
    const orig = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      orig?.apply(this, arguments);
      const btn = this.addWidget("button", "▶ Continue Workflow", null, () => {
        const w = this.widgets?.find((x) => x.name === "proceed");
        if (w) {
          w.value = true;
          w.callback?.(true);
        }
        // Approvals are per-image: stamp the fingerprint of the LAST-ASSESSED
        // image (delivered via ui.fingerprint in onExecuted) so a swapped
        // input photo re-arms the gate instead of running a stale approval.
        const af = this.widgets?.find((x) => x.name === "approved_for");
        if (af && this._atlasAssessFingerprint) {
          af.value = this._atlasAssessFingerprint;
          af.callback?.(af.value);
        }
        app.queuePrompt(0, 1);
      });
      // Buttons must never serialize — an API-format export otherwise turns
      // this into a bogus "▶ Continue Workflow" input on the prompt.
      btn.serialize = false;
    };

    // Render the assessment report directly on the node (the backend sends
    // it as ui.text) — no separate Show Text node required.
    const origExec = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      origExec?.apply(this, arguments);
      const fp = Array.isArray(message?.fingerprint) ? message.fingerprint[0] : message?.fingerprint;
      if (fp) this._atlasAssessFingerprint = fp;
      const text = Array.isArray(message?.text) ? message.text.join("\n") : message?.text;
      if (!text) return;
      let w = this.widgets?.find((x) => x.name === "assessment_report");
      if (!w) {
        w = ComfyWidgets.STRING(this, "assessment_report", ["STRING", { multiline: true }], app).widget;
        w.inputEl.readOnly = true;
        w.inputEl.style.opacity = 0.85;
        w.inputEl.style.fontFamily = "monospace";
        w.serialize = false;
      }
      w.value = text;
      this.setSize([Math.max(this.size[0], 420), Math.max(this.size[1], 380)]);
      this.graph?.setDirtyCanvas(true, true);
    };
  },
});
