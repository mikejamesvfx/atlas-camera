/**
 * Atlas Scene Health Gate — the ✅ Acknowledge & Continue button.
 *
 * AtlasSceneHealthGate holds the solve when the scene-health report is
 * warn/fail (a PASS-level report flows automatically). This button is the
 * one-click acknowledgement: it sets `proceed`, stamps the fingerprint of
 * the LAST-GATED solve+image (delivered via ui.fingerprint), and re-queues —
 * the exact mechanics of AtlasSolveGate's ✅ (atlas_solve_gate.js). Pure UI
 * convenience: toggling the widgets by hand does the same thing, and this
 * file failing to load never blocks the pipeline.
 */
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
  name: "AtlasCamera.SceneHealthGate",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AtlasSceneHealthGate") return;
    const orig = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      orig?.apply(this, arguments);
      const btn = this.addWidget("button", "✅ Acknowledge & Continue", null, () => {
        const w = this.widgets?.find((x) => x.name === "proceed");
        if (w) {
          w.value = true;
          w.callback?.(true);
        }
        // Acknowledgements are per-solve+image: stamp the fingerprint so a
        // swapped photo or a re-solve re-arms the gate instead of running a
        // stale acknowledgement.
        const af = this.widgets?.find((x) => x.name === "approved_for");
        if (af && this._atlasHealthGateFingerprint) {
          af.value = this._atlasHealthGateFingerprint;
          af.callback?.(af.value);
        }
        app.queuePrompt(0, 1);
      });
      // Buttons must never serialize — an API-format export otherwise turns
      // this into a bogus input on the prompt.
      btn.serialize = false;
    };

    // Render the health report directly on the node (sent as ui.text).
    const origExec = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      origExec?.apply(this, arguments);
      const fp = Array.isArray(message?.fingerprint) ? message.fingerprint[0] : message?.fingerprint;
      if (fp) this._atlasHealthGateFingerprint = fp;
      const text = Array.isArray(message?.text) ? message.text.join("\n") : message?.text;
      if (!text) return;
      let w = this.widgets?.find((x) => x.name === "health_summary");
      if (!w) {
        w = ComfyWidgets.STRING(this, "health_summary", ["STRING", { multiline: true }], app).widget;
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
