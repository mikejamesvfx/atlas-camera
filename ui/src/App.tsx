import {
  AlertTriangle,
  ArrowUpRight,
  Box,
  Calculator,
  CheckCircle2,
  Crosshair,
  Download,
  FolderOpen,
  ImagePlus,
  Lightbulb,
  Minus,
  Move,
  Ruler,
  Save,
  ScanLine,
  Undo2,
  Upload
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  analyzeProject,
  createProject,
  exportReviewPackage,
  loadLlmModels,
  loadReferences,
  promoteScaleCue,
  requestLlmGuidance,
  saveConstraints,
  solveProject
} from "./api";
import { buildXyzGrid } from "./gridGeometry";
import { Viewport3D } from "./Viewport3D";
import {
  addLlmScaleCandidates,
  addProxyObject,
  normalizeViewport3DState,
  selectedProxy,
  selectProxyObject,
  updateProxyObject,
  withViewport3DState
} from "./viewport3dState";
import type {
  AnalyzePreanalysisStatus,
  CameraAnalysis,
  Constraints,
  LlmGuidance,
  LlmProvider,
  LlmProviderModel,
  Point,
  ProjectInfo,
  ScaleReference,
  SceneScaleCue,
  Segment,
  SolveSummary,
  Tool,
  Viewport3DDisplay,
  Viewport3DProxyObject,
  Viewport3DProxyType,
  Viewport3DState
} from "./types";

const tools: Array<{ id: Tool; label: string; icon: typeof Move }> = [
  { id: "select", label: "Select", icon: Move },
  { id: "left", label: "Left guides", icon: ArrowUpRight },
  { id: "right", label: "Right guides", icon: Minus },
  { id: "vertical", label: "Vertical guides", icon: ScanLine },
  { id: "scale", label: "Scale guide", icon: Ruler },
  { id: "horizon", label: "Horizon", icon: Crosshair }
];

const emptyConstraints: Constraints = {
  image_width: 0,
  image_height: 0,
  line_groups: { left: [], right: [], vertical: [] },
  scale_constraints: [],
  intrinsics_hint: { sensor_width_mm: 36 }
};

const providerDefaults: Record<LlmProvider, { label: string; baseUrl: string; model: string }> = {
  lmstudio: { label: "LM Studio", baseUrl: "http://127.0.0.1:1234/v1", model: "" },
  llamacpp: { label: "llama.cpp", baseUrl: "http://127.0.0.1:8080/v1", model: "" },
  ollama: { label: "Ollama", baseUrl: "http://127.0.0.1:11434", model: "gemma3:4b" }
};

export function App() {
  const [projectDir, setProjectDir] = useState("");
  const [project, setProject] = useState<ProjectInfo | null>(null);
  const [constraints, setConstraints] = useState<Constraints>(emptyConstraints);
  const [tool, setTool] = useState<Tool>("select");
  const [status, setStatus] = useState("Create a project and load an image.");
  const [error, setError] = useState("");
  const [summary, setSummary] = useState<SolveSummary | null>(null);
  const [solvePayload, setSolvePayload] = useState<any>(null);
  const [analysis, setAnalysis] = useState<CameraAnalysis | null>(null);
  const [preanalysis, setPreanalysis] = useState<LlmGuidance | null>(null);
  const [preanalysisStatus, setPreanalysisStatus] = useState<AnalyzePreanalysisStatus>("skipped");
  const [preanalysisWarning, setPreanalysisWarning] = useState<string | null>(null);
  const [llmGuidance, setLlmGuidance] = useState<LlmGuidance | null>(null);
  const [llmProvider, setLlmProvider] = useState<LlmProvider>("lmstudio");
  const [llmModel, setLlmModel] = useState("");
  const [llmBaseUrl, setLlmBaseUrl] = useState("http://127.0.0.1:1234/v1");
  const [llmApiKey, setLlmApiKey] = useState("");
  const [llmModels, setLlmModels] = useState<LlmProviderModel[]>([]);
  const [llmProviderStatus, setLlmProviderStatus] = useState("Refresh models to check local vision capability.");
  const [references, setReferences] = useState<ScaleReference[]>([]);
  const [referenceQuery, setReferenceQuery] = useState("person");
  const [selectedReference, setSelectedReference] = useState("person_175cm");
  const [draftStart, setDraftStart] = useState<Point | null>(null);
  const [draftEnd, setDraftEnd] = useState<Point | null>(null);
  const [history, setHistory] = useState<Constraints[]>([]);
  const [future, setFuture] = useState<Constraints[]>([]);
  const [assetRevision, setAssetRevision] = useState(() => Date.now());
  const canvasRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadReferences(referenceQuery).then((data) => setReferences(data.references)).catch(() => undefined);
  }, [referenceQuery]);

  const sourceUrl = useMemo(() => {
    if (!project?.source_image) return "";
    return `/api/files/source?project_dir=${encodeURIComponent(project.project_dir)}&v=${project.has_solve ? "solved" : "base"}-${assetRevision}`;
  }, [assetRevision, project]);

  const overlayUrl = useMemo(() => {
    if (!project?.has_overlay) return "";
    return `/api/files/overlay?project_dir=${encodeURIComponent(project.project_dir)}&v=${assetRevision}`;
  }, [assetRevision, project]);

  const xyzGrid = useMemo(
    () => buildXyzGrid(constraints, solvePayload),
    [constraints, solvePayload]
  );

  const spec = useMemo(
    () => buildCameraSpec({ analysis, constraints, solvePayload, summary }),
    [analysis, constraints, solvePayload, summary]
  );
  const viewport3d = useMemo(
    () => normalizeViewport3DState(constraints.viewport3d),
    [constraints.viewport3d]
  );
  const selectedViewportProxy = useMemo(() => selectedProxy(viewport3d), [viewport3d]);
  const guideOverlayActive = tool !== "select" && tool !== "horizon";

  const pushConstraints = (next: Constraints) => {
    setHistory((items) => [...items.slice(-24), constraints]);
    setFuture([]);
    setConstraints(next);
  };

  const commitViewport3D = (next: Viewport3DState) => {
    pushConstraints(withViewport3DState(constraints, next));
  };

  const handleViewportDisplayChange = (display: Viewport3DDisplay) => {
    commitViewport3D({ ...viewport3d, display });
  };

  const handleAddProxy = (type: Viewport3DProxyType) => {
    commitViewport3D(addProxyObject(viewport3d, type));
  };

  const handleAddLlmProxyCandidates = () => {
    commitViewport3D(addLlmScaleCandidates(viewport3d, preanalysis));
  };

  const handleSelectProxy = (id: string | null) => {
    commitViewport3D(selectProxyObject(viewport3d, id));
  };

  const handleUpdateSelectedProxy = (
    patch: Partial<Pick<Viewport3DProxyObject, "label" | "position" | "rotation" | "scale" | "locked">>
  ) => {
    if (!selectedViewportProxy) return;
    commitViewport3D(updateProxyObject(viewport3d, selectedViewportProxy.id, patch));
  };

  const handleCreateProject = async (file?: File) => {
    setError("");
    setStatus("Preparing project folder.");
    try {
      const payload = await createProject(projectDir, file);
      setProject(payload.project);
      setProjectDir(payload.project.project_dir);
      setConstraints(payload.constraints);
      setAssetRevision(Date.now());
      setSummary(null);
      setSolvePayload(null);
      setAnalysis(null);
      setPreanalysis(null);
      setPreanalysisStatus("skipped");
      setPreanalysisWarning(null);
      setLlmGuidance(null);
      setStatus("Image loaded. Draw left and right guides.");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  };

  const handleSave = async () => {
    if (!project) return;
    setStatus("Saving constraints.");
    const saved = await saveConstraints(project.project_dir, constraints);
    setConstraints(saved.constraints);
    setStatus("Constraints saved.");
  };

  const handleProviderChange = (provider: LlmProvider) => {
    const defaults = providerDefaults[provider];
    setLlmProvider(provider);
    setLlmBaseUrl(defaults.baseUrl);
    setLlmModel(defaults.model);
    setLlmModels([]);
    setLlmProviderStatus(`Using ${defaults.label}. Refresh models to check vision capability.`);
  };

  const handleRefreshLlmModels = async () => {
    setError("");
    setLlmProviderStatus(`Checking ${providerDefaults[llmProvider].label}.`);
    try {
      const payload = await loadLlmModels(llmProvider, llmModel, llmBaseUrl, llmApiKey);
      setLlmModels(payload.models);
      if (!llmModel && payload.model) setLlmModel(payload.model);
      const selected = payload.models.find((item) => item.id === payload.model || item.name === payload.model);
      const modelLabel = selected?.name ?? (payload.model || "no model");
      setLlmProviderStatus(`${providerDefaults[llmProvider].label}: ${modelLabel} - ${payload.diagnostic_status}.`);
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : String(exc);
      setLlmProviderStatus(`${providerDefaults[llmProvider].label}: ${message}`);
    }
  };

  const handleSolve = async () => {
    if (!project) return;
    setError("");
    setStatus("Solving camera.");
    try {
      await saveConstraints(project.project_dir, constraints);
      const solved = await solveProject(project.project_dir);
      setSummary(solved.summary);
      setSolvePayload(solved.solve);
      setAnalysis(null);
      setProject(solved.project);
      setAssetRevision(Date.now());
      setStatus("Solve complete.");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  };

  const handleAnalyze = async () => {
    if (!project) return;
    setError("");
    setStatus("Reading image with local vision model.");
    try {
      await saveConstraints(project.project_dir, constraints);
      setStatus("Analyzing camera geometry.");
      const analyzed = await analyzeProject(project.project_dir, llmProvider, llmModel, llmBaseUrl, true, llmApiKey);
      setSummary(analyzed.summary);
      setSolvePayload(analyzed.solve);
      setAnalysis(analyzed.analysis);
      setPreanalysis(analyzed.preanalysis ?? null);
      setPreanalysisStatus(analyzed.preanalysis_status);
      setPreanalysisWarning(analyzed.preanalysis_warning ?? null);
      setLlmGuidance(null);
      setProject(analyzed.project);
      setAssetRevision(Date.now());
      const imageReadingWarning = analyzed.preanalysis_warning?.replace(/^Image reading skipped:\s*/, "");
      setStatus(
        analyzed.preanalysis_status === "available"
          ? "Image reading and camera matrix analysis complete."
          : `Camera matrix analysis complete; image reading skipped${imageReadingWarning ? `: ${imageReadingWarning}` : "."}`
      );
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  };

  const handleLlmGuidance = async () => {
    if (!project) return;
    setError("");
    setStatus("Asking local vision model for solve guidance.");
    try {
      await saveConstraints(project.project_dir, constraints);
      const guided = await requestLlmGuidance(project.project_dir, llmProvider, llmModel, llmBaseUrl, llmApiKey);
      setSummary(guided.summary);
      setAnalysis(guided.analysis);
      setLlmGuidance(guided.guidance);
      setStatus(`Local LLM guidance saved: ${guided.guidance_path}`);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  };

  const handlePromoteScaleCue = async (cue: SceneScaleCue) => {
    if (!project || !cue.bbox_px || !cue.suggested_reference_ids.length) return;
    const referenceId = cue.suggested_reference_ids[0];
    setError("");
    setStatus(`Adding "${cue.label}" as scale constraint.`);
    try {
      const result = await promoteScaleCue(
        project.project_dir,
        referenceId,
        cue.bbox_px as [number, number, number, number]
      );
      if (!result.promoted) {
        setStatus(`Scale constraint "${referenceId}" already exists.`);
        return;
      }
      pushConstraints({ ...constraints, scale_constraints: result.constraints.scale_constraints });
      setStatus(`"${cue.label}" added as scale constraint. Run Solve to update.`);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  };

  const handleExport = async () => {
    if (!project) return;
    setError("");
    setStatus("Exporting review package.");
    try {
      const exported = await exportReviewPackage(project.project_dir);
      setStatus(`Review package exported: ${exported.package_dir}`);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  };

  const imagePointFromEvent = (event: React.PointerEvent): Point | null => {
    const box = canvasRef.current?.getBoundingClientRect();
    if (!box || constraints.image_width === 0 || constraints.image_height === 0) return null;
    const x = ((event.clientX - box.left) / box.width) * constraints.image_width;
    const y = ((event.clientY - box.top) / box.height) * constraints.image_height;
    return [
      Math.max(0, Math.min(constraints.image_width, x)),
      Math.max(0, Math.min(constraints.image_height, y))
    ];
  };

  const commitSegment = (segment: Segment) => {
    if (tool === "select" || tool === "horizon") return;
    if (tool === "scale") {
      pushConstraints({
        ...constraints,
        scale_constraints: [
          ...constraints.scale_constraints,
          { reference_id: selectedReference || undefined, image_points: segment }
        ]
      });
      return;
    }
    pushConstraints({
      ...constraints,
      line_groups: {
        ...constraints.line_groups,
        [tool]: [...constraints.line_groups[tool], segment]
      }
    });
  };

  const undo = () => {
    const previous = history[history.length - 1];
    if (!previous) return;
    setFuture((items) => [constraints, ...items]);
    setConstraints(previous);
    setHistory((items) => items.slice(0, -1));
  };

  const redo = () => {
    const next = future[0];
    if (!next) return;
    setHistory((items) => [...items, constraints]);
    setConstraints(next);
    setFuture((items) => items.slice(1));
  };

  return (
    <main className="recovery-shell">
      <header className="recovery-topbar">
        <div className="wordmark">
          <span>Atlas</span>
          <strong>Latent Recovery</strong>
        </div>
        <div className="session-trail">
          <span>{project ? shortProjectName(project.project_dir) : "ATL-new"}</span>
          <i />
          <span>{project?.source_image ? fileName(project.source_image) : "no image loaded"}</span>
          <i />
          <span>{constraints.image_width && constraints.image_height ? `${constraints.image_width} x ${constraints.image_height}` : "awaiting frame"}</span>
        </div>
        <div className={`confidence-badge ${confidenceTone(summary?.confidence ?? 0)}`}>
          <b />
          <span>{summary ? `${Math.round(summary.confidence * 100)}% recovered` : "unsolved"}</span>
          <em>{summary?.source_method ? formatModeLabel(summary.source_method) : "manual lineup"}</em>
        </div>
      </header>

      <div className="recovery-body">
        <section className="recovery-main">
          <div className="stage-wrap">
            <div className="stage-frame">
              {!sourceUrl ? (
                <div className="empty-state">
                  <ArchitecturalPlaceholder />
                  <Box size={28} />
                  <span>Load a still image to begin camera lineup.</span>
                </div>
              ) : (
                <div className="lineup-workbench">
                  <Viewport3D
                    analysis={analysis}
                    sourceUrl={sourceUrl}
                    constraints={constraints}
                    solvePayload={solvePayload}
                    state={viewport3d}
                    selectedProxy={selectedViewportProxy}
                    onDisplayChange={handleViewportDisplayChange}
                    onSelectProxy={handleSelectProxy}
                  />
                  <div
                    ref={canvasRef}
                    className={guideOverlayActive ? "lineup-guide-overlay active" : "lineup-guide-overlay"}
                    style={{ aspectRatio: `${constraints.image_width} / ${constraints.image_height}` }}
                    onPointerDown={(event) => {
                      if (!guideOverlayActive) return;
                      const point = imagePointFromEvent(event);
                      setDraftStart(point);
                      setDraftEnd(point);
                    }}
                    onPointerMove={(event) => {
                      if (guideOverlayActive && draftStart) setDraftEnd(imagePointFromEvent(event));
                    }}
                    onPointerUp={(event) => {
                      if (!guideOverlayActive) return;
                      const end = imagePointFromEvent(event);
                      if (draftStart && end) commitSegment([draftStart, end]);
                      setDraftStart(null);
                      setDraftEnd(null);
                    }}
                  >
                    {guideOverlayActive && <img key={sourceUrl} src={sourceUrl} alt="Source still guide overlay" draggable={false} />}
                    {overlayUrl && <img key={overlayUrl} className="debug-overlay" src={overlayUrl} alt="Debug overlay" draggable={false} />}
                    <svg className="solve-scope" viewBox={`0 0 ${constraints.image_width} ${constraints.image_height}`}>
                      <XyzGridOverlay grid={xyzGrid} />
                      <Guides constraints={constraints} />
                      {solvePayload?.horizon_line?.endpoints_px && (
                        <line
                          className="horizon-line"
                          x1={solvePayload.horizon_line.endpoints_px[0][0]}
                          y1={solvePayload.horizon_line.endpoints_px[0][1]}
                          x2={solvePayload.horizon_line.endpoints_px[1][0]}
                          y2={solvePayload.horizon_line.endpoints_px[1][1]}
                        />
                      )}
                      <SceneAnnotations spec={spec} width={constraints.image_width} height={constraints.image_height} />
                      {draftStart && draftEnd && (
                        <line
                          className={`guide-line ${tool}`}
                          x1={draftStart[0]}
                          y1={draftStart[1]}
                          x2={draftEnd[0]}
                          y2={draftEnd[1]}
                        />
                      )}
                    </svg>
                    <div className="image-caption">
                      {guideOverlayActive ? "2D guide draw layer" : "3D camera lineup"} / {constraints.image_width || "-"} x {constraints.image_height || "-"}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="operation-strip">
            <div className="tool-cluster" aria-label="Atlas tools">
              {tools.map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    key={item.id}
                    className={tool === item.id ? "tool-button active" : "tool-button"}
                    title={item.label}
                    aria-label={item.label}
                    onClick={() => setTool(item.id)}
                  >
                    <Icon size={17} />
                  </button>
                );
              })}
              <button className="tool-button" title="Undo" aria-label="Undo" onClick={undo} disabled={!history.length}>
                <Undo2 size={17} />
              </button>
              <button className="tool-button redo" title="Redo" aria-label="Redo" onClick={redo} disabled={!future.length}>
                <Undo2 size={17} />
              </button>
            </div>

            <label className="project-input">
              <FolderOpen size={15} />
              <input
                value={projectDir}
                onChange={(event) => setProjectDir(event.target.value)}
                placeholder="Project folder"
              />
            </label>

            <label className="action-button file-action">
              <ImagePlus size={15} />
              <span>Load image</span>
              <input
                type="file"
                accept="image/*"
                onChange={(event) => handleCreateProject(event.target.files?.[0])}
              />
            </label>
            <button className="action-button" onClick={() => handleCreateProject()} disabled={!projectDir}>
              <FolderOpen size={15} />
              Open
            </button>
            <button className="action-button" onClick={handleSave} disabled={!project}>
              <Save size={15} />
              Save
            </button>
            <button className="action-button technical" onClick={handleAnalyze} disabled={!project}>
              <Calculator size={15} />
              Analyze
            </button>
            <button className="action-button advisory" onClick={handleLlmGuidance} disabled={!project}>
              <Lightbulb size={15} />
              Guide
            </button>
            <button className="action-button primary" onClick={handleSolve} disabled={!project}>
              <Upload size={15} />
              Solve
            </button>
            <button className="action-button" onClick={handleExport} disabled={!project?.has_solve}>
              <Download size={15} />
              Export
            </button>
          </div>

          <footer className="status-strip">
            <span>{status}</span>
            <Metric label="Focal" value={formatNumber(spec.focalLength, "mm")} />
            <Metric label="Horizon" value={formatNumber(summary?.horizon_angle_deg, "deg")} />
            <Metric label="Guides" value={String(summary?.guided_lines ?? countGuides(constraints))} />
            <Metric label="Matrix" value={analysis ? formatModeLabel(analysis.mode) : "--"} />
            <Metric label="LLM" value={llmGuidance?.model ?? (preanalysisStatus === "available" ? preanalysis?.model ?? "ready" : "--")} />
            {error && <span className="error-text">{error}</span>}
          </footer>
        </section>

        <aside className="spec-drawer">
          <section className="spec-section">
            <SectionHeader title="Lens" />
            <SpecRow label="Focal length" value={formatNumber(spec.focalLength, "mm")} confidence={spec.metricConfidence.focal} />
            <SpecRow label="Field of view" value={formatNumber(spec.fovHorizontal, "deg")} confidence={spec.metricConfidence.focal} />
            <SpecRow label="Sensor width" value={formatNumber(spec.sensorWidth, "mm")} confidence={spec.metricConfidence.sensor} />
            <SpecRow label="Focal source" value={spec.focalSource} confidence={spec.metricConfidence.focal} last />
          </section>

          <section className="spec-section">
            <SectionHeader title="Frame" />
            <SpecRow label="Principal point" value={spec.principalPoint} confidence={spec.metricConfidence.extrinsics} />
            <SpecRow label="Horizon y" value={spec.horizonY} confidence={spec.metricConfidence.horizon} />
            <SpecRow label="Coordinate core" value={analysis ? `${analysis.coordinate_system}, ${analysis.up_axis}-up` : "right_handed, Y-up"} confidence={1} last />
          </section>

          <section className="spec-section">
            <SectionHeader title="Vanishing Points" />
            <VPTable vps={spec.vanishingPoints} metrics={spec.metricConfidence} />
          </section>

          <section className="spec-section">
            <SectionHeader title="Intrinsic Matrix K" />
            {analysis ? (
              <MatrixReadout matrix={analysis.intrinsic_matrix} />
            ) : (
              <div className="spec-empty">Run Analyze to inspect K.</div>
            )}
          </section>

          <section className="spec-section">
            <SectionHeader title="Scale Reference" />
            <input
              className="search-input"
              value={referenceQuery}
              onChange={(event) => setReferenceQuery(event.target.value)}
              placeholder="Search references"
            />
            <select value={selectedReference} onChange={(event) => setSelectedReference(event.target.value)}>
              {references.map((reference) => (
                <option key={reference.id} value={reference.id}>
                  {reference.label}
                </option>
              ))}
            </select>
          </section>

          <section className="spec-section">
            <SectionHeader title="3D Lineup" />
            <div className="proxy-actions">
              <button type="button" onClick={() => handleAddProxy("person_card")}>Person</button>
              <button type="button" onClick={() => handleAddProxy("box")}>Box</button>
              <button type="button" onClick={() => handleAddProxy("floor_plane")}>Floor</button>
              <button type="button" onClick={() => handleAddProxy("wall_plane")}>Wall</button>
              <button type="button" onClick={() => handleAddProxy("corridor")}>Corridor</button>
              <button type="button" onClick={handleAddLlmProxyCandidates} disabled={!preanalysis?.scale_candidates?.length}>LLM cues</button>
            </div>
            <ProxyInspector proxy={selectedViewportProxy} onChange={handleUpdateSelectedProxy} />
            <div className="readout technical-readout">
              <Metric label="Mode" value={formatViewportMode(viewport3d.display.active_mode)} />
              <Metric label="Proxies" value={String(viewport3d.proxy_objects.length)} />
              <Metric label="Reprojection" value={analysis ? "Matrix ready" : "Analyze first"} />
              <Metric label="Guide layer" value={guideOverlayActive ? "Drawing" : "Orbit"} />
            </div>
          </section>

          <section className="spec-section">
            <SectionHeader title="Analysis" />
            <section className="nested-section">
              <SectionHeader title="Image Reading" />
              {preanalysis ? (
                <PreanalysisPanel guidance={preanalysis} warning={preanalysisWarning} onPromote={handlePromoteScaleCue} />
              ) : (
                <EmptyPanel
                  icon={<Lightbulb size={17} />}
                  text={preanalysisWarning ?? "Analyze reads the image first when a local vision provider is available."}
                />
              )}
            </section>
            {analysis ? <AnalysisPanel analysis={analysis} /> : <EmptyPanel icon={<Calculator size={17} />} text="Run Analyze to inspect P and readiness." />}
          </section>

          <section className="spec-section">
            <SectionHeader title="Local LLM" />
            <div className="llm-controls">
              <label>
                <span>Provider</span>
                <select value={llmProvider} onChange={(event) => handleProviderChange(event.target.value as LlmProvider)}>
                  {Object.entries(providerDefaults).map(([id, config]) => (
                    <option key={id} value={id}>
                      {config.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Model</span>
                <input
                  list="llm-models"
                  value={llmModel}
                  onChange={(event) => setLlmModel(event.target.value)}
                  placeholder={llmProvider === "lmstudio" ? "Refresh to select loaded vision model" : "Model name"}
                />
                <datalist id="llm-models">
                  {llmModels.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
                </datalist>
              </label>
              <label>
                <span>Base URL</span>
                <input value={llmBaseUrl} onChange={(event) => setLlmBaseUrl(event.target.value)} />
              </label>
              <label>
                <span>API Key</span>
                <input value={llmApiKey} onChange={(event) => setLlmApiKey(event.target.value)} placeholder="optional local token" />
              </label>
              <button className="action-button" onClick={handleRefreshLlmModels} type="button">
                Refresh models
              </button>
              <div className="provider-status">{llmProviderStatus}</div>
            </div>
            {llmGuidance ? <GuidancePanel guidance={llmGuidance} onPromote={handlePromoteScaleCue} /> : <EmptyPanel icon={<Lightbulb size={17} />} text="Run Guide after Analyze for solve-context advice." />}
          </section>

          <section className="quality-card">
            <div>
              <span>Recovery Quality</span>
              <strong>{summary ? `${Math.round(summary.confidence * 100)}%` : "--"}</strong>
            </div>
            <QualityLegend />
            {!!summary?.warnings.length && (
              <ul className="warnings">
                {summary.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            )}
          </section>
        </aside>
      </div>
    </main>
  );
}

function SceneAnnotations({ spec, width, height }: { spec: CameraSpec; width: number; height: number }) {
  if (!width || !height) return null;
  const pp = spec.principalPointPx ?? [width / 2, height / 2];
  const horizonY = spec.horizonYPx ?? height * 0.45;
  return (
    <g className="scene-annotations">
      <AnnotationPin x={pp[0]} y={pp[1]} label="Principal" value={spec.principalPoint} side="right" highlight />
      <AnnotationPin x={width * 0.5} y={horizonY} label="Horizon" value={spec.horizonY} side="left" />
      <AnnotationPin x={width * 0.82} y={height * 0.28} label="Focal" value={formatNumber(spec.focalLength, "mm")} side="right" />
      <AnnotationPin x={width * 0.16} y={height * 0.56} label="FOV" value={formatNumber(spec.fovHorizontal, "deg")} side="left" />
    </g>
  );
}

function AnnotationPin({
  x,
  y,
  label,
  value,
  side,
  highlight = false
}: {
  x: number;
  y: number;
  label: string;
  value: string;
  side: "left" | "right";
  highlight?: boolean;
}) {
  const direction = side === "right" ? 1 : -1;
  const boxX = x + direction * 36;
  return (
    <g className={highlight ? "annotation-pin highlight" : "annotation-pin"}>
      <line x1={x} y1={y} x2={boxX} y2={y} />
      <circle cx={x} cy={y} r="3" />
      <text x={boxX + direction * 5} y={y - 7} textAnchor={side === "right" ? "start" : "end"}>
        {label}
      </text>
      <text x={boxX + direction * 5} y={y + 8} textAnchor={side === "right" ? "start" : "end"} className="pin-value">
        {value}
      </text>
    </g>
  );
}

function SectionHeader({ title }: { title: string }) {
  return (
    <div className="section-header">
      <span>{title}</span>
      <i />
    </div>
  );
}

function SpecRow({ label, value, confidence, last = false }: { label: string; value: string; confidence: number; last?: boolean }) {
  const tone = confidenceTone(confidence);
  return (
    <div className={last ? "spec-row last" : "spec-row"}>
      <span>
        <b className={tone} />
        {label}
      </span>
      <strong>{value}</strong>
    </div>
  );
}

function VPTable({ vps, metrics }: { vps: DisplayVanishingPoint[]; metrics: Record<string, number> }) {
  if (!vps.length) return <div className="spec-empty">No solved vanishing points.</div>;
  return (
    <table className="vp-table">
      <thead>
        <tr>
          <th>VP</th>
          <th>X</th>
          <th>Y</th>
          <th>Conf</th>
        </tr>
      </thead>
      <tbody>
        {vps.map((vp, index) => {
          const metric = metrics[`vp${index + 1}`] ?? vp.confidence;
          return (
            <tr key={`${vp.label}-${index}`}>
              <td>{vp.label}</td>
              <td>{formatSigned(vp.x)}</td>
              <td>{formatSigned(vp.y)}</td>
              <td><span className={confidenceTone(metric)}>{Math.round(metric * 100)}%</span></td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function AnalysisPanel({ analysis }: { analysis: CameraAnalysis }) {
  return (
    <div className="analysis-panel">
      <div className="analysis-mode">{formatModeLabel(analysis.mode)}</div>
      <div className="readiness-list">
        {analysis.readiness.map((item) => {
          const Icon = item.status === "ok" ? CheckCircle2 : AlertTriangle;
          return (
            <div className={`readiness-item ${item.status}`} key={item.label}>
              <Icon size={15} />
              <span>{item.label}</span>
              <strong>{item.detail}</strong>
            </div>
          );
        })}
      </div>
      <MatrixReadout matrix={analysis.projection_matrix} compact />
      <div className="readout technical-readout">
        <Metric label="fx / fy" value={`${analysis.focal_px.fx.toFixed(1)} / ${analysis.focal_px.fy.toFixed(1)}`} />
        <Metric label="det R" value={analysis.rotation_quality.determinant.toFixed(4)} />
      </div>
    </div>
  );
}

function ProxyInspector({
  proxy,
  onChange
}: {
  proxy: Viewport3DProxyObject | null;
  onChange: (patch: Partial<Pick<Viewport3DProxyObject, "label" | "position" | "rotation" | "scale" | "locked">>) => void;
}) {
  if (!proxy) {
    return <EmptyPanel icon={<Box size={17} />} text="Select or add a proxy object for 3D lineup." />;
  }
  return (
    <div className="proxy-inspector">
      <label>
        <span>Label</span>
        <input value={proxy.label} onChange={(event) => onChange({ label: event.target.value })} />
      </label>
      <VectorInput label="Position" value={proxy.position} onChange={(position) => onChange({ position })} />
      <VectorInput label="Rotation" value={proxy.rotation} onChange={(rotation) => onChange({ rotation })} />
      <VectorInput label="Scale" value={proxy.scale} min={0.01} onChange={(scale) => onChange({ scale })} />
      <label className="proxy-lock">
        <input type="checkbox" checked={proxy.locked} onChange={(event) => onChange({ locked: event.target.checked })} />
        <span>Locked proxy</span>
      </label>
    </div>
  );
}

function VectorInput({
  label,
  value,
  min,
  onChange
}: {
  label: string;
  value: [number, number, number];
  min?: number;
  onChange: (value: [number, number, number]) => void;
}) {
  return (
    <label>
      <span>{label}</span>
      <div className="vector-input">
        {value.map((component, index) => (
          <input
            key={index}
            type="number"
            step="0.05"
            min={min}
            value={Number(component.toFixed(3))}
            onChange={(event) => {
              const next = [...value] as [number, number, number];
              next[index] = Number(event.target.value);
              onChange(next);
            }}
          />
        ))}
      </div>
    </label>
  );
}

function MatrixReadout({ matrix, compact = false }: { matrix: number[][]; compact?: boolean }) {
  return (
    <div className={compact ? "matrix-readout compact" : "matrix-readout"}>
      <table>
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((value, colIndex) => (
                <td key={colIndex}>{formatMatrixValue(value)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PreanalysisPanel({ guidance, warning, onPromote }: { guidance: LlmGuidance; warning?: string | null; onPromote?: (cue: SceneScaleCue) => void }) {
  return (
    <div className="guidance-panel image-reading-panel">
      <div className="guidance-summary">
        <span>{formatProviderLabel(guidance.provider)} / {guidance.model ?? "vision model"}</span>
        <p>{guidance.summary}</p>
      </div>
      {guidance.scene_description && <GuidanceList title="Scene description" items={guidance.scene_description} />}
      <GuidanceList title="Scale candidates" items={guidance.scale_candidates ?? []} />
      <GuidanceList title="Perspective cues" items={(guidance.perspective_cues ?? []).length ? guidance.perspective_cues : guidance.technical_guidance} />
      <GuidanceList title="Recommended guides" items={guidance.recommended_guides ?? []} />
      <GuidanceList title="Lens notes" items={guidance.lens_distortion_notes ?? []} />
      <GuidanceList title="Occlusion notes" items={guidance.occlusion_notes ?? []} />
      <GuidanceList title="Lens and recovery risks" items={guidance.solve_risk_notes} />
      {!!guidance.scale_cues.length && (
        <div className="cue-list">
          <span>Scale indicators</span>
          {guidance.scale_cues.map((cue, index) => (
            <div className="cue-item" key={`${cue.label}-${index}`}>
              <strong>{cue.label}</strong>
              <span>{Math.round(cue.confidence * 100)}%</span>
              {!!cue.suggested_reference_ids.length && <em>{cue.suggested_reference_ids.join(", ")}</em>}
              {cue.notes && <p>{cue.notes}</p>}
              {onPromote && cue.bbox_px && !!cue.suggested_reference_ids.length && (
                <button type="button" className="cue-promote" onClick={() => onPromote(cue)} title={`Add ${cue.suggested_reference_ids[0]} as scale constraint`}>
                  Add constraint
                </button>
              )}
            </div>
          ))}
        </div>
      )}
      <GuidanceList title="Warnings" items={[...guidance.warnings, ...(warning ? [warning] : [])]} warning />
    </div>
  );
}

function GuidancePanel({ guidance, onPromote }: { guidance: LlmGuidance; onPromote?: (cue: SceneScaleCue) => void }) {
  return (
    <div className="guidance-panel">
      <div className="guidance-summary">
        <span>{formatProviderLabel(guidance.provider)} / {guidance.model ?? "model"}</span>
        <p>{guidance.summary}</p>
      </div>
      <GuidanceList title="Technical guidance" items={guidance.technical_guidance} />
      <GuidanceList title="Solve risks" items={guidance.solve_risk_notes} />
      {!!guidance.scale_cues.length && (
        <div className="cue-list">
          <span>Scale cues</span>
          {guidance.scale_cues.map((cue, index) => (
            <div className="cue-item" key={`${cue.label}-${index}`}>
              <strong>{cue.label}</strong>
              <span>{Math.round(cue.confidence * 100)}%</span>
              {!!cue.suggested_reference_ids.length && <em>{cue.suggested_reference_ids.join(", ")}</em>}
              {onPromote && cue.bbox_px && !!cue.suggested_reference_ids.length && (
                <button type="button" className="cue-promote" onClick={() => onPromote(cue)} title={`Add ${cue.suggested_reference_ids[0]} as scale constraint`}>
                  Add constraint
                </button>
              )}
            </div>
          ))}
        </div>
      )}
      <GuidanceList title="Warnings" items={guidance.warnings} warning />
    </div>
  );
}

function formatProviderLabel(provider?: string | null) {
  if (provider === "lmstudio") return "LM Studio";
  if (provider === "llamacpp") return "llama.cpp";
  if (provider === "ollama") return "Ollama";
  return provider ?? "local";
}

function GuidanceList({ title, items, warning = false }: { title: string; items: string[] | string; warning?: boolean }) {
  const normalizedItems = normalizeGuidanceItems(items);
  if (!normalizedItems.length) return null;
  return (
    <div className={warning ? "guidance-list warning" : "guidance-list"}>
      <span>{title}</span>
      <ul>
        {normalizedItems.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

function EmptyPanel({ icon, text }: { icon: React.ReactNode; text: string }) {
  return (
    <div className="analysis-empty">
      {icon}
      <span>{text}</span>
    </div>
  );
}

function QualityLegend() {
  return (
    <div className="quality-legend">
      <span><b className="confident" /> Confident &gt;= 88%</span>
      <span><b className="uncertain" /> Uncertain 72-87%</span>
      <span><b className="weak" /> Weak &lt; 72%</span>
    </div>
  );
}

function Guides({ constraints }: { constraints: Constraints }) {
  return (
    <>
      {(["left", "right", "vertical"] as const).flatMap((group) =>
        constraints.line_groups[group].map((line, index) => (
          <line
            key={`${group}-${index}`}
            className={`guide-line ${group}`}
            x1={line[0][0]}
            y1={line[0][1]}
            x2={line[1][0]}
            y2={line[1][1]}
          />
        ))
      )}
      {constraints.scale_constraints.map((scale, index) => {
        const line = scale.image_points;
        if (!line) return null;
        return (
          <line
            key={`scale-${index}`}
            className="guide-line scale"
            x1={line[0][0]}
            y1={line[0][1]}
            x2={line[1][0]}
            y2={line[1][1]}
          />
        );
      })}
    </>
  );
}

function XyzGridOverlay({ grid }: { grid: ReturnType<typeof buildXyzGrid> }) {
  return (
    <g className="xyz-grid" aria-hidden="true">
      {(["x", "y", "z"] as const).map((axis) => (
        <g key={axis} className={`xyz-grid-axis ${axis}`}>
          {grid[axis].lines.map((line, index) => (
            <line
              key={`${axis}-${index}`}
              x1={line[0][0]}
              y1={line[0][1]}
              x2={line[1][0]}
              y2={line[1][1]}
            />
          ))}
          {grid[axis].vanishingPoint && (
            <circle
              className="axis-vanishing-point"
              cx={grid[axis].vanishingPoint[0]}
              cy={grid[axis].vanishingPoint[1]}
              r="4"
            />
          )}
        </g>
      ))}
    </g>
  );
}

function ArchitecturalPlaceholder() {
  return (
    <svg className="placeholder-architecture" viewBox="0 0 800 450" aria-hidden="true">
      <rect x="0" y="0" width="800" height="198" />
      <rect x="0" y="198" width="800" height="252" />
      <polygon points="0,198 144,20 304,198" />
      <polygon points="224,198 400,0 576,198" />
      <polygon points="496,198 656,44 800,198" />
      {[0.55, 0.66, 0.78, 0.9].map((value) => (
        <line key={value} x1="0" y1={value * 450} x2="800" y2={value * 450} />
      ))}
    </svg>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <span className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </span>
  );
}

type DisplayVanishingPoint = {
  label: string;
  x: number;
  y: number;
  confidence: number;
};

type CameraSpec = {
  focalLength?: number;
  fovHorizontal?: number;
  sensorWidth?: number;
  principalPoint: string;
  principalPointPx?: Point;
  horizonY: string;
  horizonYPx?: number;
  focalSource: string;
  metricConfidence: Record<string, number>;
  vanishingPoints: DisplayVanishingPoint[];
};

function buildCameraSpec({
  analysis,
  constraints,
  solvePayload,
  summary
}: {
  analysis: CameraAnalysis | null;
  constraints: Constraints;
  solvePayload: any;
  summary: SolveSummary | null;
}): CameraSpec {
  const intrinsics = solvePayload?.camera?.intrinsics ?? {};
  const cameraEstimation = solvePayload?.debug_metadata?.camera_estimation ?? {};
  const confidence = solvePayload?.camera?.confidence ?? solvePayload?.confidence_detail ?? {};
  const metricConfidence = {
    horizon: 0,
    vp1: 0,
    vp2: 0,
    vp3: 0,
    focal: 0,
    extrinsics: 0,
    sensor: 0,
    ...(confidence.individual_metrics ?? {})
  };
  const principal = analysis?.principal_point_px
    ? [analysis.principal_point_px.cx, analysis.principal_point_px.cy] as Point
    : intrinsics.cx_px !== undefined && intrinsics.cy_px !== undefined
      ? [intrinsics.cx_px, intrinsics.cy_px] as Point
      : undefined;
  const horizonLine = solvePayload?.horizon_line?.endpoints_px;
  const horizonYPx = Array.isArray(horizonLine)
    ? (Number(horizonLine[0][1]) + Number(horizonLine[1][1])) / 2
    : undefined;
  const horizonY = horizonYPx !== undefined && constraints.image_height
    ? `${((horizonYPx / constraints.image_height) * 100).toFixed(1)}%`
    : "--";

  return {
    focalLength: summary?.focal_length_mm ?? intrinsics.focal_length_mm ?? cameraEstimation.focal_length_mm,
    fovHorizontal: analysis?.fov_deg.horizontal ?? cameraEstimation.fov_horizontal_deg,
    sensorWidth: intrinsics.sensor_width_mm,
    principalPoint: principal ? `${principal[0].toFixed(1)}, ${principal[1].toFixed(1)}` : "--",
    principalPointPx: principal,
    horizonY,
    horizonYPx,
    focalSource: cameraEstimation.focal_source ?? "metadata_or_hint",
    metricConfidence,
    vanishingPoints: (solvePayload?.vanishing_points ?? []).map((vp: any, index: number) => ({
      label: vp.direction_label ?? `VP${index + 1}`,
      x: Number(vp.position_px?.[0] ?? 0),
      y: Number(vp.position_px?.[1] ?? 0),
      confidence: Number(vp.confidence ?? 0)
    }))
  };
}

function countGuides(constraints: Constraints) {
  return (
    constraints.line_groups.left.length +
    constraints.line_groups.right.length +
    constraints.line_groups.vertical.length
  );
}

function formatNumber(value: number | undefined, unit: string) {
  return typeof value === "number" ? `${value.toFixed(2)} ${unit}` : "--";
}

function formatSigned(value: number) {
  const rounded = Math.round(value);
  return `${rounded > 0 ? "+" : ""}${rounded.toLocaleString()}`;
}

function formatMatrixValue(value: number) {
  if (Math.abs(value) >= 1000) return value.toExponential(2);
  if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(1);
  return value.toFixed(3);
}

function formatModeLabel(value: string) {
  return value.replace(/_/g, " ");
}

function formatViewportMode(value: Viewport3DState["display"]["active_mode"]) {
  if (value === "image_match") return "Image Match";
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function confidenceTone(value: number) {
  if (value >= 0.88) return "confident";
  if (value >= 0.72) return "uncertain";
  return "weak";
}

function shortProjectName(value: string) {
  return fileName(value) || "ATL-project";
}

function fileName(value: string) {
  const parts = value.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] ?? value;
}

function normalizeGuidanceItems(items: string[] | string) {
  if (typeof items === "string") {
    return items.trim() ? [humanizeGuidanceText(items.trim())] : [];
  }
  const cleanItems = items.filter((item) => item.trim().length > 0);
  if (cleanItems.length > 8 && cleanItems.every((item) => item.trim().length === 1)) {
    return [humanizeGuidanceText(cleanItems.join("").replace(/\s+/g, " ").trim())];
  }
  return cleanItems.map(humanizeGuidanceText);
}

const guidanceWords = [
  "immediately", "subsequently", "accurately", "recommended", "significant", "distortion",
  "relationships", "guidelines", "calibration", "demonstrate", "substantial", "constraints",
  "references", "reference", "solutions", "vanishing", "vertical", "metadata", "projected",
  "incorrect", "datasets", "dataset", "evidence", "guides", "family", "horizon", "provide",
  "establish", "current", "state", "represents", "solve", "solved", "risk", "due", "deficit",
  "without", "explicit", "scale", "anchors", "image", "exhibit", "spatial", "solver", "solely",
  "relying", "which", "does", "adequate", "accurate", "camera", "parameter", "eth3d", "dtu",
  "robust", "require", "minimum", "least", "left", "right", "line", "lines", "two", "one",
  "the", "and", "to", "of", "at", "as", "such", "for", "high", "accuracy", "achieve", "scene",
  "perspective", "depth", "estimation", "using", "like", "person_175cm", "building", "height",
  "is", "a", "not", "or", "on", "may", "that", "these", "necessary", "elements", "considered",
  "comparable", "only"
].sort((first, second) => second.length - first.length);

function humanizeGuidanceText(value: string) {
  const compact = value.replace(/[^a-zA-Z0-9]/g, "");
  if (value.includes(" ") || compact.length < 32 || !/(guideline|vanishing|reference|dataset|metadata|solver|camera)/i.test(value)) {
    return value;
  }
  let output = "";
  let token = "";
  for (const char of value) {
    if (/[\w']/u.test(char)) {
      token += char;
    } else {
      if (token) output += segmentGuidanceToken(token);
      token = "";
      output += char;
    }
  }
  if (token) output += segmentGuidanceToken(token);
  return output
    .replace(/\s+([.,;:!?])/g, "$1")
    .replace(/([.,;:!?])(?=\S)/g, "$1 ")
    .replace(/\s*-\s*/g, "-")
    .replace(/\s+/g, " ")
    .trim();
}

function segmentGuidanceToken(value: string) {
  let index = 0;
  const parts: string[] = [];
  while (index < value.length) {
    const lower = value.slice(index).toLowerCase();
    const word = lower.startsWith("solvedue")
      ? "solve"
      : guidanceWords.find((candidate) => lower.startsWith(candidate));
    if (word) {
      parts.push(value.slice(index, index + word.length));
      index += word.length;
    } else {
      parts.push(value[index]);
      index += 1;
    }
  }
  return parts.join(" ");
}
