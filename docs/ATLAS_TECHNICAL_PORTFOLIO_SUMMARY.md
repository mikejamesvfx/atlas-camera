# Atlas Camera — Technical Portfolio Summary

*A self-directed R&D project framed for AI / ML / software engineering roles. Written to
translate the work into transferable value for any technical field, not only film/VFX.*

---

## Positioning statement (the 30-second version)

I designed and built **Atlas Camera**, an end-to-end system that reconstructs a usable 3D
camera and scene geometry from a **single 2D image** and turns it into an interactive,
exportable 3D asset. It spans the full stack of an applied-AI product: integrating
pretrained deep-learning models (monocular depth, learned camera calibration, vision-language
models), a dependency-light numerical core implementing the classical computer-vision and 3D
geometry math around them, a real-time WebGL/GLSL rendering layer, a Python web backend, and a
plugin architecture that embeds the whole thing inside a host application. It is ~14,000 lines
of Python across 106 modules, backed by 312 automated tests, plus a custom Three.js/GLSL
viewport and a React/TypeScript workbench.

The core problem — **recovering 3D spatial structure from limited 2D observation** — is the
same problem that sits under robotics, autonomous vehicles, AR/VR, spatial computing, mapping,
and digital-twin systems. The film use-case is just where I applied it.

---

## What the system does (in plain terms)

Given one photograph or AI-generated image, Atlas Camera:

1. **Recovers the camera** — focal length, orientation, horizon, and metric height — using a
   learned prior (GeoCalib) with a classical vanishing-point solver as an alternative path.
2. **Derives 3D geometry** — from monocular depth (Depth Anything V2) it builds either a
   continuous triangulated relief mesh or fitted primitives (walls, planes via RANSAC, room
   cuboids), each an artist-selectable strategy.
3. **Establishes real-world scale** — via a tiered cascade (known-size reference objects, a
   depth-fitted ground plane, or a flagged default), so measurements are *measured, not
   assumed*.
4. **Projects and renders** — casts the source image back onto the geometry in a real-time
   WebGL viewport with custom GLSL projection shaders, enabling parallax camera moves from a
   single still.
5. **Exports to production tools** — Maya, Blender, Nuke, and USD, with correct coordinate-
   system conversions and time-sampled animated cameras.

---

## By the numbers (concrete scope)

| Dimension | Evidence |
|---|---|
| Python | ~14,150 LOC across 106 modules (`atlas_camera` package) |
| Tests | 312 test functions across 45 files |
| Integration surface | 41 ComfyUI node classes; 19 example workflows |
| Real-time graphics | ~2,700 LOC custom Three.js + GLSL viewport (orbit/fly controllers, projection material, multi-pass render) |
| Web app | ~3,400 LOC React/Vite/TypeScript workbench + FastAPI backend |
| Model integrations | GeoCalib (calibration), Depth Anything V2 (depth), local VLMs (scale cues) |
| Export targets | Maya, Blender, Nuke, USD, glTF/GLB (zero-dependency writer) |
| Eval infrastructure | COLMAP / DTU / ETH3D dataset loaders + benchmark harness |

---

## Technical competencies demonstrated

Grouped by transferable domain, each with concrete evidence and the portable claim.

### 1. Applied ML / model integration engineering
- Integrated three classes of pretrained models into one pipeline: **learned camera
  calibration** (GeoCalib), **monocular depth estimation** (Depth Anything V2, metric and
  relative), and **vision-language models** (Ollama / LM Studio / llama.cpp) for semantic
  scale cues.
- Built **device-aware inference** (CUDA / Apple-MPS / CPU auto-selection) and **fail-soft
  integration** (a missing local VLM server degrades gracefully to an empty result rather than
  crashing the pipeline).
- Kept the ML dependencies **optional and isolated** behind guarded imports so the core stays
  installable and testable without a GPU or torch — a discipline directly relevant to shipping
  ML features into constrained or heterogeneous production environments.
> *Portable claim:* I can take research models off the shelf and turn them into robust,
> deployable pipeline components with sane fallbacks and hardware portability.

### 2. Computer vision & 3D geometry
- Implemented the classical math end-to-end: **pinhole intrinsics/extrinsics**, focal-length↔
  pixel conversions, **vanishing-point and horizon estimation**, 4×4 view-matrix conventions
  (row-major, column-vector), and rigorous **coordinate-system conversions** at adapter
  boundaries (Y-up vs Z-up, image vs world).
- Built geometry derivation from depth: **back-projection, normal estimation, RANSAC plane
  fitting, ground-plane estimation, metric scale recovery, mesh triangulation/decimation, and
  silhouette-aware tearing**.
- Solved real signal-quality problems: a **Laplacian-roughness sky detector** that
  distinguishes noisy sky depth from genuinely sloped surfaces (plain variance
  misclassifies real roofs — roughness doesn't), and plausibility guards that reject
  physically impossible fits.
> *Portable claim:* This is the exact toolbox behind visual SLAM, structure-from-motion, AR
> anchoring, and 3D scene understanding — camera pose, depth, and metric reconstruction.

### 3. Software architecture & systems design
- Designed a **dependency-light pure-Python core** (zero required runtime deps) with all
  heavyweight libraries (torch, OpenCV, USD, FastAPI) behind **guarded optional dependency
  groups** with actionable install hints — a clean separation between algorithmic core and
  environment-specific adapters.
- Authored a **versioned dataclass schema** with full JSON round-tripping and backward-compatible
  loaders, plus **in-process custom types** passed by reference to avoid serialization overhead.
- Enforced **adapter-boundary discipline**: coordinate conversions and host-specific logic never
  leak into the core, making the same solver reusable across five DCC exporters and two UIs.
> *Portable claim:* I design systems where a stable, testable core is insulated from volatile
> integrations — the property that keeps ML products maintainable as models and hosts churn.

### 4. Full-stack & real-time graphics
- Wrote a **custom WebGL/Three.js viewport** including a self-contained orbit controller and an
  unclamped fly controller, **custom GLSL shaders** for image-to-geometry projection (world
  position → recovered-camera pixel → texture sample, with per-fragment frustum/facing
  discards), and a **multi-pass renderer** (shaded/depth/normal/mask).
- Built a **zero-dependency glTF 2.0 / GLB writer** (binary, embedded texture) and a FastAPI +
  React/TypeScript project service.
- Debugged non-obvious **CSS/flexbox layout and browser-replaced-element sizing** issues in an
  embedded canvas, and reverse-engineered a host framework's internal layout code to find the
  root cause — evidence of deep, persistent debugging across the stack.
> *Portable claim:* I'm comfortable from GLSL and render loops up through TypeScript UI and
> Python APIs — valuable for any team shipping interactive ML or 3D/spatial products.

### 5. Data, evaluation & tooling
- Built **dataset loaders and a benchmark harness** for standard CV datasets (COLMAP, DTU,
  ETH3D) and command-line tools for batch solving, review-package generation, and validation.
- Created a **curated reference-data registry** for metric scale grounding.
> *Portable claim:* I build the evaluation and tooling scaffolding that lets a team measure
> whether an ML system is actually working, not just demo it.

### 6. Developer-platform / extensibility engineering
- Designed **41 composable, single-responsibility nodes** for a host plugin system, including
  an explicit merge/composition model (rather than fragile implicit chaining), and solved
  real integration hazards: a **double-import route-registration guard**, an LRU cache, and
  static-file serving.
- Reasoned carefully about **open-source license boundaries** (keeping GPL-licensed
  dependencies as graph-level runtime components rather than linked code, to protect the
  project's own licensing).
> *Portable claim:* I can extend a platform with a clean, well-factored extension API and think
> through the legal/architectural implications of dependencies.

---

## Engineering practices & judgment

- **Test-backed**: 312 automated tests; algorithmic changes land with unit tests, and
  frontend/rendering changes are verified with live end-to-end reproduction on real inputs
  (not just synthetic scenes — a distinction I learned the hard way when synthetic tests missed
  a real-image failure).
- **Decisions are documented**: architecture, roadmap, and per-feature design notes with
  explicit tradeoffs, known limitations, and "verify-this-assumption-first" go/no-go gates.
- **Regression-aware**: shared math is factored into single sources of truth; where duplication
  is unavoidable (e.g. a Python sampler mirrored in JS for 60fps playback), it's documented as
  deliberate with a keep-in-sync note.
- **Honest about limits**: features ship with their failure modes written down (parallax
  budget, disocclusion smearing, orbit-cone coverage) rather than oversold.

---

## What I learned building this

- How to **wrap research models into production-grade pipelines** — the gap between "runs in a
  notebook" and "degrades gracefully, picks its device, and stays optional" is most of the
  real work.
- That **classical geometry and modern deep learning are complementary, not competing** — the
  best results came from a learned prior *plus* hand-implemented calibration/scale math that
  keeps the output physically meaningful and measurable.
- **Cross-stack debugging discipline** — chasing a bug from a GLSL shader through a JS layout
  chain into a host framework's internals, and reading the actual installed source rather than
  guessing.
- **Architectural restraint** — the most valuable design decisions were about what *not* to
  build: reuse an existing schema type instead of inventing one, keep a dependency at arm's
  length, add a plausibility guard instead of rewriting a clustering algorithm.

---

## Where this maps outside film

| Industry / role | Directly transferable from Atlas |
|---|---|
| **Robotics / autonomous systems** | Camera calibration, pose estimation, monocular depth → metric 3D, ground-plane fitting, coordinate frames |
| **AR / VR / spatial computing** | Single-image scene reconstruction, real-time projection/rendering, camera tracking, 3D asset export |
| **Autonomous vehicles / mapping / digital twins** | Depth-to-geometry, RANSAC plane extraction, metric scale, multi-source geometry fusion |
| **Applied ML product engineering** | End-to-end model→pipeline→interactive-UI delivery, device-aware inference, fail-soft integration |
| **Computer vision engineering (general)** | The full classical CV toolbox implemented and tested, integrated with learned models |
| **Developer tools / ML platform** | Composable plugin/node API design, extensibility, evaluation harnesses, licensing judgment |

---

## Resume-ready bullet points (quantified, ATS-friendly)

- Designed and built a single-image 3D scene reconstruction system (~14K LOC Python, 312 tests,
  41 plugin nodes) integrating learned camera calibration, monocular depth, and vision-language
  models into one pipeline.
- Implemented an end-to-end computer-vision stack — pinhole camera intrinsics/extrinsics,
  vanishing-point calibration, RANSAC plane fitting, ground-plane metric scaling, and depth-to-
  mesh reconstruction — with rigorous coordinate-system handling.
- Engineered device-aware (CUDA/MPS/CPU), fail-soft ML inference behind guarded optional
  dependencies, keeping a zero-dependency core installable and testable without a GPU.
- Authored a real-time WebGL/GLSL rendering pipeline (custom projection shaders, orbit/fly
  camera controllers, multi-pass render) plus a zero-dependency glTF/GLB exporter.
- Built dataset loaders and a benchmarking harness (COLMAP, DTU, ETH3D) to evaluate
  reconstruction quality against standard CV datasets.
- Delivered production integrations exporting to Maya, Blender, Nuke, and USD with correct
  cross-application coordinate conversions and time-sampled animated cameras.
- Designed a composable plugin architecture with explicit geometry-composition semantics,
  solving concrete integration hazards (route double-registration, caching, license isolation).

---

## Skills keyword bank (for ATS / profiles)

Python · PyTorch · Computer Vision · 3D Geometry · Camera Calibration · Monocular Depth
Estimation · Structure-from-Motion concepts · RANSAC · Linear Algebra · NumPy · OpenCV ·
Machine Learning Integration · Vision-Language Models · Inference Pipelines · GPU/CUDA/MPS ·
WebGL · GLSL · Three.js · Real-time Rendering · TypeScript · React · FastAPI · REST APIs ·
USD / OpenUSD · glTF · Software Architecture · API Design · Plugin/Extension Systems ·
Automated Testing · Data Pipeline Engineering · Benchmarking / Evaluation

---

## Framing note for the career transition

The strongest angle for a non-film AI role is **not** "VFX artist who codes a bit." It's:
*a builder who independently shipped a full applied-AI system — research-model integration,
classical CV/geometry, real-time graphics, a web stack, and production tooling — solving the
core problem of recovering 3D structure from 2D observation.* The domain (film) demonstrates
the ability to identify a real problem and build the complete solution; the **skills** are
squarely those of an applied-AI / computer-vision / full-stack ML engineer.

*Next step:* this summary can be tailored into a targeted résumé and cover letter for a specific
job listing — matching the posting's keywords and emphasizing the most relevant of the six
competency areas above.
