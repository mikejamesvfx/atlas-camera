# Response to the project-wide stale-code report (2026-07-17)

**Date:** 2026-07-18
**Refers to:** `ATLAS_PROJECT_WIDE_ENGINEERING_REPORT.md` (lives in the ComfyUI-install
clone at `ComfyUI_windows_portable\ComfyUI\custom_nodes\atlas-camera`, not in this repo).

The report was written against that **portable-install clone**, whose working tree
carries a large pile of local, never-committed files. Verification against the actual
repository showed most of the report's P0 findings describe that local clutter, not
tracked repo content. This note records the claim-by-claim verification, what was done
about the findings that were real, and where the clone-only assets live.

## Claim-by-claim verification

| Report claim | Verified reality |
|---|---|
| Example catalog test fails; ~20 extra legacy JSONs at `examples/` top level | **Clone-only.** The repo has exactly the 3 shipping workflows; `tests/test_example_workflows.py` passes. The ~20 extras are *untracked* working files in the clone (`atlas_moge_*`, `atlas_00022_*`, …) |
| Runtime artifacts (`.pytest_cache`, `.pytest_tmp`, `__pycache__`, `.venv`, screenshots, videos) in the repository | **Nothing is tracked in git** in either checkout; `.gitignore` already covers all of it. They exist on-disk in the clone only |
| V2/V3/hero/canonical generated-workflow duplication | Only the canonical workflows are committed (intentional, hand-calibrated since). The 26 v2/v3/hero variants and 3 of the 4 generator scripts are untracked clone-only files |
| `atlas_camera/mcp` is an implicit second API, plan-stage | **Wrong** — it is a shipped, tested (`tests/test_mcp_comfy_http.py`), `.mcp.json`-registered server with a `[mcp]` extra and `docs/MCP_SERVER.md` |
| `atlas_camera/inference` possibly unused | Every module in it is imported from production paths and has dedicated tests |
| `atlas_camera/reference_data` possibly unused | Used by `core/solver.py`, `comfy/nodes.py`, `ui/project.py`, `inference/multimodal_helper.py` |
| Older `atlas/` compatibility package | It is the intentional thin `import atlas` facade (a 6-line re-export), not stale duplication |
| `AtlasLoadImageSolveCamera` is legacy | **Confirmed** — zero tests, one showcase workflow that itself labels it a deprecation candidate |
| `atlas_camera/gaussian` needs evidence | **Confirmed unused** — a `NotImplementedError` placeholder with only a placeholder test |
| `comfy/nodes.py` monolith (8,177 lines) | Confirmed, but it is a refactor concern, not stale code — deferred to its own task |

## Actions taken in this pass (2026-07-18)

1. **`AtlasLoadImageSolveCamera` deprecated** — `DEPRECATED = True`, "(Deprecated)"
   display name, one-line log warning naming `AtlasSolveFromImage` /
   `AtlasLearnedSolveFromImage`. Kept registered so saved workflows load; removal in a
   later release.
2. **`atlas_camera/gaussian` removed** — the 3DGS placeholder package, its placeholder
   test, and its doc mentions. One `git revert` away if 3DGS work starts.
3. **`tools/generate_atlas_canonical_workflows.py` committed as guarded provenance** —
   the script that bootstrapped the canonical workflows. Its hero/v3 source graphs are
   not in the repo and the committed canonicals have since been hand-calibrated, so it
   now refuses to overwrite outputs without `--force` and exits clearly on missing
   sources.
4. **`docs/dev/archive/` created** — seven fully-superseded plan/research docs moved
   there with `> ARCHIVED` headers; all repo references updated to the new paths.

## Deliberately NOT done

- **The portable clone was not touched** (user decision: repo-only cleanup). Its
  untracked working files, v2/v3/hero variants, and runtime caches remain as they were.
- **`nodes.py` domain split** (report P2) — real, but a regression-risk refactor that
  deserves its own planned task, not a cleanup side effect.
- **Compat surfaces kept**, per the report's own P2 guidance: the `atlas/` facade, the
  `projection_workspace` schema alias, positional-widget compatibility branches, and
  the experimental backends (LaRI/WT, Fixer, Angle Patch).

## Clone-only assets worth knowing about

All untracked, in `ComfyUI_windows_portable\ComfyUI\custom_nodes\atlas-camera`:

- `examples/*.json` — ~20 working session files (`atlas_moge_atlas_*`,
  `atlas_moge_flux_*`, `atlas_00022_*`, `atlas_scope_seacliff_castle.json`, …).
  These are the user's live working workflows; do not delete blindly.
- `examples/showcase/` — ~26 untracked `*_v2_workflow.json` / `*_v3_workflow.json` /
  `atlas_hero_*_workflow.json` variants (the canonical workflows' source graphs among
  them).
- `tools/` — `generate_atlas_hero_workflows.py`, `generate_showcase_workflows_v2.py`,
  `generate_showcase_workflows_v3.py` (never committed; the canonical generator was
  brought into this repo, these three were not).
