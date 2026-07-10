/**
 * Atlas Solve Gate — the ✅ Approve Solve button.
 *
 * AtlasSolveGate pauses everything downstream of the solve (its solve
 * output returns ExecutionBlocker) until `proceed` is turned on, so the
 * first Queue costs only the solve + whatever cheap preview is wired
 * UNGATED. This button is the one-click approval: it sets `proceed`,
 * stamps the approval fingerprint of the LAST-GATED solve+image (delivered
 * via ui.fingerprint), and re-queues — the exact mechanics of
 * AtlasAssessImage's ▶ Continue (atlas_assess.js). Pure UI convenience:
 * toggling the widgets by hand does the same thing, and this file failing
 * to load never blocks the pipeline.
 */
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
  name: "AtlasCamera.SolveGate",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AtlasSolveGate") return;
    const orig = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      orig?.apply(this, arguments);
      const btn = this.addWidget("button", "✅ Approve Solve", null, () => {
        const w = this.widgets?.find((x) => x.name === "proceed");
        if (w) {
          w.value = true;
          w.callback?.(true);
        }
        // Approvals are per-solve+image: stamp the fingerprint so a swapped
        // photo or a re-solve re-arms the gate instead of running a stale
        // approval.
        const af = this.widgets?.find((x) => x.name === "approved_for");
        if (af && this._atlasSolveGateFingerprint) {
          af.value = this._atlasSolveGateFingerprint;
          af.callback?.(af.value);
        }
        app.queuePrompt(0, 1);
      });
      // Buttons must never serialize — an API-format export otherwise turns
      // this into a bogus input on the prompt.
      btn.serialize = false;
    };

    // Render the solve summary directly on the node (sent as ui.text).
    const origExec = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      origExec?.apply(this, arguments);
      const fp = Array.isArray(message?.fingerprint) ? message.fingerprint[0] : message?.fingerprint;
      if (fp) this._atlasSolveGateFingerprint = fp;
      const text = Array.isArray(message?.text) ? message.text.join("\n") : message?.text;
      if (!text) return;
      let w = this.widgets?.find((x) => x.name === "solve_summary");
      if (!w) {
        w = ComfyWidgets.STRING(this, "solve_summary", ["STRING", { multiline: true }], app).widget;
        w.inputEl.readOnly = true;
        w.inputEl.style.opacity = 0.85;
        w.inputEl.style.fontFamily = "monospace";
        w.serialize = false;
      }
      w.value = text;
      this.setDirtyCanvas(true, true);
    };
  },
});
