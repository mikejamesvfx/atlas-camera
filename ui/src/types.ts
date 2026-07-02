export type Point = [number, number];
export type Segment = [Point, Point];
export type Tool = "select" | "left" | "right" | "vertical" | "scale" | "horizon";
export type Viewport3DMode = "image_match" | "perspective" | "top" | "front" | "side";
export type Viewport3DProxyType = "person_card" | "box" | "floor_plane" | "wall_plane" | "corridor" | "unit_box" | "custom_box";

export type LineGroups = {
  left: Segment[];
  right: Segment[];
  vertical: Segment[];
};

export type ScaleConstraint = {
  reference_id?: string;
  image_points?: Segment;
  name?: string;
};

export type Constraints = {
  image_width: number;
  image_height: number;
  line_groups: LineGroups;
  scale_constraints: ScaleConstraint[];
  intrinsics_hint?: Record<string, number | string | number[]>;
  viewport3d?: Viewport3DState;
};

export type Viewport3DDisplay = {
  active_mode: Viewport3DMode;
  show_image: boolean;
  show_grid: boolean;
  show_axes: boolean;
  show_frustum: boolean;
  show_guides: boolean;
  show_proxies: boolean;
  show_horizon: boolean;
  show_projection: boolean;
  show_depth: boolean;
  image_opacity: number;
  grid_scale: number;
  lock_camera_to_view: boolean;
};

export type Viewport3DProxyObject = {
  id: string;
  type: Viewport3DProxyType;
  label: string;
  source: "user" | "llm_suggestion" | "preset";
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];
  locked: boolean;
};

export type Viewport3DCameraOverrides = {
  focal_length_mm?: number;
  sensor_width_mm?: number;
  camera_position?: [number, number, number];
  camera_rotation?: [number, number, number];
  principal_point_px?: Point;
  user_preview?: boolean;
};

export type Viewport3DState = {
  schema_version: 1;
  display: Viewport3DDisplay;
  proxy_objects: Viewport3DProxyObject[];
  camera_overrides: Viewport3DCameraOverrides;
  selected_proxy_id?: string | null;
};

export type ProjectInfo = {
  project_dir: string;
  source_image: string | null;
  has_solve: boolean;
  has_overlay: boolean;
};

export type ScaleReference = {
  id: string;
  label: string;
  category: string;
  height: number;
  units: string;
  confidence: string;
};

export type SolveSummary = {
  source_method: string;
  confidence: number;
  vanishing_points: number;
  guided_lines: number;
  focal_length_mm?: number;
  horizon_angle_deg?: number;
  warnings: string[];
};

export type ReadinessItem = {
  label: string;
  status: "ok" | "needs_input";
  detail: string;
};

export type CameraAnalysis = {
  mode: string;
  coordinate_system: string;
  up_axis: string;
  intrinsic_matrix: number[][];
  view_matrix: number[][];
  projection_matrix: number[][];
  camera_position: number[];
  focal_px: { fx: number; fy: number };
  principal_point_px: { cx: number; cy: number };
  fov_deg: { horizontal?: number; vertical?: number };
  rotation_quality: { determinant: number; orthogonality_residual: number };
  vanishing_point_support: {
    detected: number;
    left_lines: number;
    right_lines: number;
    vertical_lines: number;
    scale_guides: number;
    horizon_angle_deg?: number;
    focal_source: string;
  };
  readiness: ReadinessItem[];
  notes: string[];
};

export type SceneScaleCue = {
  label: string;
  confidence: number;
  bbox_px?: [number, number, number, number] | null;
  suggested_reference_ids: string[];
  notes?: string | null;
  source: string;
};

export type LlmProvider = "lmstudio" | "llamacpp" | "ollama";

export type LlmProviderModel = {
  id: string;
  name: string;
  vision_capable?: boolean | null;
  capabilities: string[];
};

export type LlmProviderModelsResponse = {
  provider: LlmProvider;
  base_url: string;
  model: string;
  models: LlmProviderModel[];
  vision_capable?: boolean | null;
  diagnostic_status: string;
};

export type LlmGuidance = {
  image_path: string;
  summary: string;
  scale_cues: SceneScaleCue[];
  warnings: string[];
  raw_response?: Record<string, unknown>;
  model?: string | null;
  provider?: string | null;
  base_url?: string | null;
  vision_capable?: boolean | null;
  diagnostic_status?: string | null;
  scene_description?: string | null;
  scale_candidates: string[];
  perspective_cues: string[];
  lens_distortion_notes: string[];
  occlusion_notes: string[];
  recommended_guides: string[];
  technical_guidance: string[];
  solve_risk_notes: string[];
  dataset_evidence: string[];
};

export type AnalyzePreanalysisStatus = "available" | "skipped" | "failed";

export type AnalyzeResponse = {
  project: ProjectInfo;
  summary: SolveSummary;
  solve: unknown;
  analysis: CameraAnalysis;
  preanalysis?: LlmGuidance | null;
  preanalysis_status: AnalyzePreanalysisStatus;
  preanalysis_warning?: string | null;
};
