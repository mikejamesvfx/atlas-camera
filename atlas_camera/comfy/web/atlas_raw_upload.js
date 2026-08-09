/**
 * Atlas RAW upload — a "choose RAW file to upload" button on AtlasLoadRAW.
 *
 * file_path stays a plain STRING widget (a saved-workflow contract — paths
 * like "CameraRaw/sh001/DSCF3915.RAF" must keep loading verbatim), so this is
 * NOT the LoadImage combo-upload pattern: the button uploads the picked file
 * through ComfyUI's /upload/image endpoint (which stores arbitrary files in
 * the input directory; video nodes use it the same way) into a CameraRaw/
 * subfolder, then writes the returned relative path into file_path. The
 * button widget is serialize=false, so positional widgets_values are
 * untouched and this file failing to load never blocks the pipeline.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const RAW_ACCEPT = [
  ".raf", ".nef", ".nrw", ".cr2", ".cr3", ".crw", ".arw", ".srf", ".sr2",
  ".dng", ".orf", ".rw2", ".pef", ".srw", ".raw", ".rwl", ".3fr", ".fff",
  ".iiq", ".x3f",
].join(",");

const UPLOAD_SUBFOLDER = "CameraRaw";

async function uploadRawFile(file) {
  const body = new FormData();
  body.append("image", file);
  body.append("subfolder", UPLOAD_SUBFOLDER);
  body.append("type", "input");
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (resp.status !== 200) {
    throw new Error(`upload failed: ${resp.status} ${resp.statusText}`);
  }
  const data = await resp.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

app.registerExtension({
  name: "AtlasCamera.RawUpload",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AtlasLoadRAW") return;
    const orig = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      orig?.apply(this, arguments);
      const pathWidget = this.widgets?.find((w) => w.name === "file_path");
      if (!pathWidget) return;

      const fileInput = document.createElement("input");
      fileInput.type = "file";
      fileInput.accept = RAW_ACCEPT;
      fileInput.style.display = "none";
      document.body.appendChild(fileInput);

      fileInput.addEventListener("change", async () => {
        const file = fileInput.files?.[0];
        fileInput.value = "";
        if (!file) return;
        btn.name = `uploading ${file.name}…`;
        this.setDirtyCanvas(true, false);
        try {
          const relativePath = await uploadRawFile(file);
          pathWidget.value = relativePath;
          pathWidget.callback?.(relativePath);
        } catch (error) {
          console.error("[AtlasCamera.RawUpload]", error);
          btn.name = "⚠ upload failed — see console";
          this.setDirtyCanvas(true, false);
          setTimeout(() => {
            btn.name = BUTTON_LABEL;
            this.setDirtyCanvas(true, false);
          }, 4000);
          return;
        }
        btn.name = BUTTON_LABEL;
        this.setDirtyCanvas(true, false);
      });

      const BUTTON_LABEL = "choose RAW file to upload";
      const btn = this.addWidget("button", BUTTON_LABEL, null, () => {
        fileInput.click();
      });
      btn.serialize = false;

      // Lifecycle callbacks are CHAINED, never assigned (viewport doctrine):
      // an orphaned hidden <input> per removed node is a slow DOM leak.
      const origRemoved = this.onRemoved;
      this.onRemoved = function () {
        fileInput.remove();
        return origRemoved?.apply(this, arguments);
      };
    };
  },
});
