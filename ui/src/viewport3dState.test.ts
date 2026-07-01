import { describe, expect, it } from "vitest";
import type { Constraints, LlmGuidance } from "./types";
import {
  addLlmScaleCandidates,
  addProxyObject,
  defaultViewport3DState,
  normalizeViewport3DState,
  selectProxyObject,
  selectedProxy,
  updateProxyObject,
  withViewport3DState
} from "./viewport3dState";

const baseConstraints: Constraints = {
  image_width: 1280,
  image_height: 720,
  line_groups: {
    left: [[[0, 0], [100, 100]]],
    right: [],
    vertical: []
  },
  scale_constraints: [],
  intrinsics_hint: { sensor_width_mm: 36 }
};

const guidance: LlmGuidance = {
  image_path: "scene.jpg",
  summary: "Tunnel scene with person and boxes.",
  scale_cues: [],
  warnings: [],
  raw_response: {},
  model: "local-vision",
  provider: "lmstudio",
  base_url: "http://127.0.0.1:1234/v1",
  vision_capable: true,
  diagnostic_status: "ok",
  scene_description: "A tunnel with rough scale anchors.",
  scale_candidates: ["human silhouette", "cardboard boxes", "corridor width"],
  perspective_cues: [],
  lens_distortion_notes: [],
  occlusion_notes: [],
  recommended_guides: [],
  technical_guidance: [],
  solve_risk_notes: [],
  dataset_evidence: []
};

describe("viewport3dState", () => {
  it("normalizes missing viewport state without requiring old constraints to change", () => {
    const state = normalizeViewport3DState(undefined);

    expect(state.schema_version).toBe(1);
    expect(state.display.active_mode).toBe("image_match");
    expect(state.proxy_objects).toEqual([]);
  });

  it("round-trips viewport3d through constraints without mutating guide lines", () => {
    const state = addProxyObject(defaultViewport3DState, "person_card");
    const next = withViewport3DState(baseConstraints, state);

    expect(next.viewport3d?.proxy_objects).toHaveLength(1);
    expect(next.line_groups).toBe(baseConstraints.line_groups);
    expect(next.line_groups.left).toEqual([[[0, 0], [100, 100]]]);
  });

  it("adds LLM scale candidates as advisory proxies only", () => {
    const state = addLlmScaleCandidates(defaultViewport3DState, guidance);

    expect(state.proxy_objects).toHaveLength(3);
    expect(state.proxy_objects.map((proxy) => proxy.source)).toEqual([
      "llm_suggestion",
      "llm_suggestion",
      "llm_suggestion"
    ]);
    expect(state.proxy_objects[0].type).toBe("person_card");
    expect(state.proxy_objects[2].type).toBe("corridor");
  });

  it("selects and updates proxy transforms independently", () => {
    const state = addProxyObject(defaultViewport3DState, "box");
    const id = state.proxy_objects[0].id;
    const selected = selectProxyObject(state, id);
    const updated = updateProxyObject(selected, id, { position: [1, 2, 3], locked: true });

    expect(selectedProxy(updated)?.position).toEqual([1, 2, 3]);
    expect(selectedProxy(updated)?.locked).toBe(true);
  });
});
