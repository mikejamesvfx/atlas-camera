/**
 * Atlas Director launch — the 🎬 Launch Director button on AtlasDirectorTake.
 *
 * Director is an external application: this button POSTs to
 * /atlas/director/launch and starts it on the artist's machine. It is not an
 * approval gate and it never re-queues the prompt — queueing here would run
 * the graph, including this node's own read, before any take exists to read.
 *
 * The request carries only session_id, width, height, frames and fps, read
 * off this node's own widgets. It deliberately never sends an executable,
 * an argv, or an output/root path -- the server takes those from its own
 * configuration (ATLAS_DIRECTOR_BIN, ATLAS_DIRECTOR_ROOT). That is a security
 * property established after a Critical finding, not an oversight to work
 * around from the browser.
 *
 * Button widget is serialize=false, same reason as every other Atlas button:
 * an API-format export must never see a bogus extra input on the prompt.
 * This file failing to load must never block the pipeline -- widget lookups
 * are optional-chained and the fetch is wrapped in try/catch.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LAUNCH_PATH = "/atlas/director/launch";
const BUTTON_LABEL = "🎬 Launch Director";
const STATUS_HOLD_MS = 6000;

function widgetValue(node, name) {
  return node.widgets?.find((w) => w.name === name)?.value;
}

app.registerExtension({
  name: "AtlasCamera.DirectorLaunch",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AtlasDirectorTake") return;
    const orig = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      orig?.apply(this, arguments);

      const showStatus = (text) => {
        btn.name = text;
        this.setDirtyCanvas(true, false);
        clearTimeout(this._atlasDirectorStatusTimer);
        this._atlasDirectorStatusTimer = setTimeout(() => {
          btn.name = BUTTON_LABEL;
          this.setDirtyCanvas(true, false);
        }, STATUS_HOLD_MS);
      };

      const btn = this.addWidget("button", BUTTON_LABEL, null, async () => {
        const session_id = widgetValue(this, "session_id");
        const width = widgetValue(this, "width");
        const height = widgetValue(this, "height");
        const frames = widgetValue(this, "frames");
        const fps = widgetValue(this, "fps");

        showStatus("launching…");

        let resp;
        try {
          resp = await api.fetchApi(LAUNCH_PATH, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id, width, height, frames, fps }),
          });
        } catch (error) {
          console.error("[AtlasCamera.DirectorLaunch]", error);
          showStatus("⚠ network error — see console");
          return;
        }

        let data = null;
        try {
          data = await resp.json();
        } catch {
          // Non-JSON body; fall through to the status-code handling below.
        }

        if (resp.status === 200) {
          const sid = data?.session_id ?? session_id;
          console.log(`[AtlasCamera.DirectorLaunch] Director opening on session ${sid}`);
          showStatus(`🎬 Director opening (${sid})`);
          return;
        }

        if (resp.status === 400) {
          const msg = data?.error || "request refused";
          console.error("[AtlasCamera.DirectorLaunch] 400:", msg);
          showStatus(`⚠ ${msg}`);
          return;
        }

        if (resp.status === 404) {
          const msg =
            data?.error ||
            "no session package found — export it first with " +
              "AtlasExportScenePackage (scene_id must match session_id, " +
              "output_dir must be the configured Director root)";
          console.error("[AtlasCamera.DirectorLaunch] 404:", msg);
          showStatus(`⚠ ${msg}`);
          return;
        }

        if (resp.status === 503) {
          const msg = data?.error || "no Director executable configured — set ATLAS_DIRECTOR_BIN";
          console.error("[AtlasCamera.DirectorLaunch] 503:", msg);
          showStatus(`⚠ ${msg}`);
          return;
        }

        const msg = data?.error || `${resp.status} ${resp.statusText}`;
        console.error("[AtlasCamera.DirectorLaunch]", resp.status, msg);
        showStatus(`⚠ launch failed: ${msg}`);
      });
      // Buttons must never serialize — an API-format export otherwise turns
      // this into a bogus input on the prompt.
      btn.serialize = false;
    };
  },
});
