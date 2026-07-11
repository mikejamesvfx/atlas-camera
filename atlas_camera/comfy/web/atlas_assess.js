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

      // Mirror the resolved staged SAM prompts into LINKED prompt widgets.
      // A widget converted to a linked input keeps displaying its stale
      // typed text even though the linked value flows at execution (found
      // live: the scope rows ran with the VLM's prompts while their boxes
      // still showed the old hand-typed strings). Display-sync only — the
      // link stays authoritative; the written value doubles as a durable
      // fallback if the artist later removes the link. Output order matches
      // RETURN_NAMES: sam_prompt_sky/far/bg/mid/fg start at output slot 3.
      // Two mirrored groups, matching RETURN_NAMES output order:
      // sam_prompt_sky/far/bg/mid/fg at slots 3..7, geom_far/bg/mid/fg at
      // slots 8..11 (→ AtlasCleanPlateLayer.geometry_override).
      // Resolve a link to its REAL consumers, following one KJ Set/Get rail
      // hop: an output wired into a SetNode fans out to every GetNode with
      // the same rail name (staged master v6 routes all assess outputs
      // through rails). Never writes into Set/Get nodes themselves — their
      // only widget is the rail NAME.
      const railValue = (n) => n.widgets?.[0]?.value ?? n.widgets_values?.[0];
      const resolveConsumers = (graph, link) => {
        const target = link ? graph.getNodeById(link.target_id) : null;
        if (!target) return [];
        if (target.type !== "SetNode") return [{ node: target, slot: link.target_slot }];
        const rail = railValue(target);
        const out = [];
        for (const n of graph._nodes ?? []) {
          if (n.type !== "GetNode" || railValue(n) !== rail) continue;
          for (const lid of n.outputs?.[0]?.links ?? []) {
            const l2 = graph.links?.[lid];
            const t2 = l2 ? graph.getNodeById(l2.target_id) : null;
            if (t2 && t2.type !== "SetNode") out.push({ node: t2, slot: l2.target_slot });
          }
        }
        return out;
      };

      const groups = [
        { values: message?.sam_prompts, base: 3, count: 5, fallback: "prompt" },
        { values: message?.sam_geometry, base: 8, count: 4, fallback: "geometry_override" },
      ];
      let mirrored = false;
      for (const g of groups) {
        if (!Array.isArray(g.values) || g.values.length !== g.count || !this.graph) continue;
        for (let i = 0; i < g.count; i++) {
          for (const linkId of this.outputs?.[g.base + i]?.links ?? []) {
            for (const { node: target, slot } of resolveConsumers(this.graph, this.graph.links?.[linkId])) {
              // The linked input names which widget it feeds.
              const widgetName = target.inputs?.[slot]?.widget?.name || g.fallback;
              const w = target.widgets?.find((x) => x.name === widgetName);
              if (w && w.value !== g.values[i]) {
                w.value = g.values[i];
                w.callback?.(w.value);
                mirrored = true;
              }
            }
          }
        }
      }
      if (mirrored) this.graph.setDirtyCanvas(true, true);
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
