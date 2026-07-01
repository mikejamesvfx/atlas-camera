import { afterEach, describe, expect, it, vi } from "vitest";
import { analyzeProject, loadLlmModels } from "./api";

describe("api", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sends provider settings with analyze requests", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        project: { project_dir: "C:/tmp/atlas", source_image: "source.png" },
        summary: {
          source_method: "artist_guided_constraints",
          confidence: 0.8,
          vanishing_points: 2,
          guided_lines: 4,
          warnings: []
        },
        solve: {},
        analysis: {},
        preanalysis: null,
        preanalysis_status: "skipped",
        preanalysis_warning: null
      })
    });
    vi.stubGlobal("fetch", fetchMock);

    await analyzeProject("C:/tmp/atlas", "lmstudio", "qwen2.5-vl", "http://127.0.0.1:1234/v1", true);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/analyze",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_dir: "C:/tmp/atlas",
          provider: "lmstudio",
          model: "qwen2.5-vl",
          base_url: "http://127.0.0.1:1234/v1",
          enable_preanalysis: true
        })
      })
    );
  });

  it("loads provider model diagnostics", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        provider: "lmstudio",
        base_url: "http://127.0.0.1:1234/v1",
        model: "qwen2.5-vl",
        models: [],
        vision_capable: true,
        diagnostic_status: "selected model advertises vision capability"
      })
    });
    vi.stubGlobal("fetch", fetchMock);

    await loadLlmModels("lmstudio", "qwen2.5-vl", "http://127.0.0.1:1234/v1");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/llm/models",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: "lmstudio",
          model: "qwen2.5-vl",
          base_url: "http://127.0.0.1:1234/v1"
        })
      })
    );
  });
});
