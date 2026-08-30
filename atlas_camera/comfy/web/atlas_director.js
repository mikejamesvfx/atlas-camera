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
 *
 * Delivery address: once Director opens, it pushes the finished take back
 * by calling `deliverTake` against `ATLAS_COMFY_URL` or, absent that,
 * `http://127.0.0.1:8188`. This launch endpoint never tells Director where
 * THIS ComfyUI actually is -- doing that would put a request-influenced
 * value on Director's spawn command line, which director_session.py
 * deliberately never does (see its module docstring). If this ComfyUI is
 * not on the default host/port, set ATLAS_COMFY_URL in ComfyUI's own
 * process environment before launching -- Director inherits it from the
 * process that spawns it. Nothing below can detect or warn about this
 * misconfiguration: a launch here can succeed (200) while the take still
 * has nowhere to land.
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
          // This is a failure to reach THIS ComfyUI's own launch endpoint,
          // not the separate delivery leg Director runs later (see the
          // file header's "Delivery address" note) -- but it's the natural
          // place an operator looks first, so say both: check this
          // ComfyUI's address, and remember that a Director launched
          // against a non-default ComfyUI still needs ATLAS_COMFY_URL set
          // in ComfyUI's own environment or its take has nowhere to land.
          console.error("[AtlasCamera.DirectorLaunch]", error);
          showStatus("⚠ network error reaching ComfyUI — see console");
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
          // Every launch failure this server can raise for a bad request --
          // a missing session package included -- lands here as 400, never
          // 404 (launch_session raises ValueError/KeyError; the route maps
          // both to 400; nothing in the launch path returns 404). Assemble
          // the guidance once: keep the server's own message, which is what
          // distinguishes a bad session id from a genuinely missing
          // package, and append the one fact it doesn't say -- where the
          // package has to come from and go.
          const guidance =
            "the session package must exist before launching — export it " +
            "first with AtlasExportScenePackage. That node's scene_id must " +
            "equal this node's session_id, and its output_dir must be the " +
            "'scenes' subdirectory of the configured ATLAS_DIRECTOR_ROOT, " +
            "given as an absolute path (the export node's default " +
            "'atlas_scenes' is relative to ComfyUI's working directory and " +
            "will not do).";
          const msg = data?.error ? `${data.error} — ${guidance}` : guidance;
          console.error("[AtlasCamera.DirectorLaunch] 400:", msg);
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
