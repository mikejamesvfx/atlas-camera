/**
 * Render AtlasExportReliefMesh's interior-hole-fill report on the node.
 *
 * The fill is export-only — it never touches the solve's own geometry, so it
 * is invisible in the viewport by design (filling a tear in the LIVE mesh
 * would bridge the very depth discontinuity the tearing exists to preserve).
 * That leaves the artist with no way to tell a working fill from a no-op
 * without a DCC round-trip. The backend sends the summary as ui.text; this
 * puts it where they are already looking.
 *
 * Same mechanism as atlas_assess.js — a read-only multiline STRING widget,
 * refreshed on every execution. Purely cosmetic: this file failing to load
 * never blocks the export (and the `report` output still carries the text).
 */
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
  name: "AtlasCamera.ExportReliefMesh",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AtlasExportReliefMesh") return;

    const origExec = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      origExec?.apply(this, arguments);
      const text = Array.isArray(message?.text) ? message.text.join("\n") : message?.text;
      if (!text) return;
      let w = this.widgets?.find((x) => x.name === "fill_report");
      if (!w) {
        w = ComfyWidgets.STRING(this, "fill_report", ["STRING", { multiline: true }], app).widget;
        w.inputEl.readOnly = true;
        w.inputEl.style.opacity = 0.85;
        w.inputEl.style.fontFamily = "monospace";
        // Never serialize: this is derived output, not artist input — a saved
        // workflow must not carry a stale report back as a widget value.
        w.serialize = false;
      }
      w.value = text;
      this.setSize([Math.max(this.size[0], 380), Math.max(this.size[1], 300)]);
      this.graph?.setDirtyCanvas(true, true);
    };
  },
});
