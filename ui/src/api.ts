import type {
  AnalyzeResponse,
  CameraAnalysis,
  Constraints,
  LlmGuidance,
  LlmProvider,
  LlmProviderModelsResponse,
  ProjectInfo,
  ScaleReference,
  SolveSummary
} from "./types";

type ProjectPayload = {
  project: ProjectInfo;
  constraints: Constraints;
};

export async function createProject(projectDir: string, image?: File): Promise<ProjectPayload> {
  const form = new FormData();
  if (projectDir) form.append("project_dir", projectDir);
  if (image) form.append("image", image);

  const response = await fetch("/api/projects", { method: "POST", body: form });
  return readJson(response);
}

export async function saveConstraints(
  projectDir: string,
  constraints: Constraints
): Promise<{ constraints: Constraints }> {
  const response = await fetch("/api/constraints", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_dir: projectDir, constraints })
  });
  return readJson(response);
}

export async function solveProject(
  projectDir: string
): Promise<{ project: ProjectInfo; summary: SolveSummary; solve: any }> {
  const response = await fetch("/api/solve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_dir: projectDir })
  });
  return readJson(response);
}

export async function analyzeProject(
  projectDir: string,
  provider: LlmProvider = "lmstudio",
  model = "",
  baseUrl = "http://127.0.0.1:1234/v1",
  enablePreanalysis = true,
  apiKey = ""
): Promise<AnalyzeResponse> {
  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      project_dir: projectDir,
      provider,
      model,
      base_url: baseUrl,
      api_key: apiKey || undefined,
      enable_preanalysis: enablePreanalysis
    })
  });
  return readJson(response);
}

export async function exportCameraUsd(projectDir: string): Promise<{ path: string; filename: string }> {
  const response = await fetch("/api/export/camera-usd", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_dir: projectDir })
  });
  return readJson(response);
}

export async function exportReviewPackage(projectDir: string): Promise<{ package_dir: string; warnings: string[] }> {
  const response = await fetch("/api/export/review-package", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_dir: projectDir, include_usd: false })
  });
  return readJson(response);
}

export async function requestLlmGuidance(
  projectDir: string,
  provider: LlmProvider,
  model: string,
  baseUrl: string,
  apiKey = ""
): Promise<{ guidance: LlmGuidance; analysis: CameraAnalysis; summary: SolveSummary; guidance_path: string }> {
  const response = await fetch("/api/llm/guidance", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_dir: projectDir, provider, model, base_url: baseUrl, api_key: apiKey || undefined })
  });
  return readJson(response);
}

export async function loadLlmModels(
  provider: LlmProvider,
  model: string,
  baseUrl: string,
  apiKey = ""
): Promise<LlmProviderModelsResponse> {
  const response = await fetch("/api/llm/models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, model, base_url: baseUrl, api_key: apiKey || undefined })
  });
  return readJson(response);
}

export async function promoteScaleCue(
  projectDir: string,
  referenceId: string,
  bboxPx: [number, number, number, number]
): Promise<{ constraints: Constraints; promoted: boolean }> {
  const response = await fetch("/api/promote_scale_cue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_dir: projectDir, reference_id: referenceId, bbox_px: bboxPx })
  });
  return readJson(response);
}

export async function loadReferences(query = ""): Promise<{ references: ScaleReference[]; categories: string[] }> {
  const params = new URLSearchParams();
  if (query) params.set("query", query);
  const response = await fetch(`/api/references?${params.toString()}`);
  return readJson(response);
}

async function readJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Atlas UI request failed.");
  }
  return payload as T;
}
