# `tools/` — developer scripts

Not part of the installed package and not imported by it. Each is run by hand
from a repo checkout. Nothing in `atlas_camera/` depends on anything here, so a
user installing from the ComfyUI Registry never sees these.

They have no importer by design, which makes them invisible to dependency
analysis. This file records the ones that are **load-bearing** — the scripts
that regenerate a shipped artifact, or that a documented workflow depends on.

It is not a full inventory. Most of the rest are covered by a test in `tests/`,
which is what keeps them honest. A script that is neither listed here nor
referenced by a test, a doc or CI has nothing establishing it is still wanted —
check its git history before assuming either way.

## Shipping artifacts — these regenerate what the repo publishes

| script | what it maintains |
|---|---|
| `rebuild_shipping_quickstarts.py` | Rebuilds `examples/atlas_input_quickstart_workflow.json` (the README's front door) and its agentic twin from live node schemas. Run it after changing any widget on a node the quickstart uses, or the shipped graph drifts from the schema. |
| `build_example_workflow.py` | The workflow-authoring library the generators build on. Emits the redundantly-linked UI format correctly (top-level `links` **and** each node's own slot links) — hand-authoring that is how the shipped set acquired its drift bugs. |
| `build_feature_audit.py` | Regenerates `reports/feature_audit.json` + `docs/FEATURE_AUDIT.md`. **Run it last**, after every other edit, and always as `PYTHONPATH=. python tools/build_feature_audit.py` — plain `python tools/…` puts `tools/` on `sys.path[0]`, so `atlas_camera` resolves through the editable install to whichever checkout pip points at, and you silently audit the wrong tree. `tests/test_feature_audit.py` fails when the committed artifact is stale. |
| `audit_node_usage.py` | Evidence collector behind the feature audit: workflow, test, MCP, tool and doc references per registered node. |

## Measurement and validation

| script | what it is for |
|---|---|
| `tear_sweep.py` | Runs every combination of tear knobs over the ray-cast fixtures, scores each against known occlusion edges, prints the Pareto front. The measured curve behind the tearing defaults. |
| `validate_review_package.py` | Checks a written review package against the format contract before it is handed to a client or a DCC. |
| `add_debug_tail.py` | Converts a workflow that ends in `AtlasBlockoutViewport` (renders in the browser via three.js) into one that also writes files, so it can be run headlessly. |
| `build_v1_shipping_workflows.py --check` | Regenerates each shipping workflow in memory and compares node types against the committed JSON, writing nothing and exiting non-zero on drift. Run it before regenerating: a builder that has fallen behind the file it owns will *silently rewrite* that file. Found four drifted workflows on 2026-08-17. |
| `smoke_hero_workflows.py` | **Release acceptance.** Queues each hero workflow against a live server and asserts the artifacts it should have written are on disk and non-trivial. `--validate-only` checks the graphs against `/object_info` without queueing. Needs `COMFY_DIR`. This is the only check that exercises what CI structurally cannot — a real queue with real models — and it exists because Hero 02's first real run found a `NameError` in **both** solver nodes that 3150 green tests had missed for a day. |

## Solvers and capture

| script | what it is for |
|---|---|
| `solve_image.py` | CLI: auto vanishing-point detection + debug overlay + review package. |
| `solve_constraints.py` | CLI: JSON line constraints → review package. |
| `validate_multiview_capture.py` | Pre-flight check on a multi-view capture set before solving. |
| `record3d_bridge.py` | Streams a Record3D capture into Atlas. |

Everything else here is a one-off: a benchmark harness, a migration that has
already run, or a generator for a workflow that is no longer distributed from
this repository. Ten such generators were removed on 2026-08-14 — every
workflow they wrote had left `examples/` in the 0.8.1 trim, so they generated
files nothing shipped. Git history has them if one is ever wanted back.
