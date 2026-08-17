---
name: atlas-v1-audit
description: Evidence-first v1 cleanup audit for Atlas Camera — repository archaeology that produces a disposition manifest, never a deletion. Use when preparing a public v1 release, deciding what belongs in the shipped tree, hunting duplicate or obsolete workflows, or resolving contradictory documentation. Read-only on the first run.
---

# /atlas-v1-audit

Turn an experimental-development repository into a clear, minimal, trustworthy
v1 product repository — **without deleting obscure but essential capabilities.**

    Find evidence first. Classify second. Delete last.

## Why this is not `vulture` plus a shell loop

Static "unused" is close to meaningless in Atlas. Code here is routinely
reached with no import anywhere:

| reached via | invisible to |
|---|---|
| `NODE_CLASS_MAPPINGS` | every import-graph tool |
| a serialized ComfyUI workflow | every import-graph tool |
| an MCP tool name in a resource string | every import-graph tool |
| pytest collection | every import-graph tool |
| `pkgutil` / `importlib` | every import-graph tool |
| Maya / Nuke / Blender executing an exported script | everything, including this audit |

So a single tool finding can **never** reach `CERTAIN`, and whole categories
are capped below it structurally. The confidence ceiling — not the scan — is
what stands between a false positive and a deleted feature.

## Running it

```bash
python .claude/skills/atlas-v1-audit/scripts/run_audit.py --bootstrap-venv
```

`--bootstrap-venv` installs vulture / deptry / ruff into `.v1-audit/.tools-venv`
(~50 MB). It never installs into the project environment; an audit must not
change what the project depends on. Drop the flag on later runs.

| flag | effect |
|---|---|
| *(none)* | full audit, ~30 s |
| `--quick` | stub the static-analysis phase, ~4 s — for iterating on a later phase |
| `--only <phase>` | run one phase and stop |
| `--check` | CI mode: exit non-zero on a broken workflow, a deregistered node class, or a live doc with a dead path or a wrong node count |

Phases run in dependency order and each writes one JSON into `.v1-audit/raw/`.
A failed phase stops the run rather than producing a manifest built on missing
evidence.

## Destructive boundary

The first run is **READ-ONLY with respect to the project**. The only writes are
under `.v1-audit/`.

Never, in audit mode: delete a project file · edit project source · change a
dependency · rewrite a workflow · update a README · run any tool's autofix.

**There is no `apply.py`.** `--apply` is a documented human procedure, not
code, because a script that can delete is a script that can delete by accident.
The procedure is at the bottom of this file.

## What it produces

```
.v1-audit/
├── summary.md              decision-oriented top level — read this first
├── unknown-review.md       needs a human; never automate these
├── delete-candidates.md    tiered; only CERTAIN is eligible for --apply
├── capability-surface.md   the v1 product surface, derived from the repo
├── node-audit.md           registration, orphan nodes, duplicate display names
├── workflow-audit.md       node-GRAPH duplicate clustering, broken workflows
├── docs-audit.md           canonical doc per topic, stale claims, overlap
├── experiment-audit.md     deregistered / unwired / gated / dormant
├── dcc-audit.md            Blender · Maya · Nuke · USD layers, kept separate
├── dependency-audit.md     deptry + knip, as candidates only
├── setup-audit.md          contradictory install paths
├── merge-candidates.md     archive-candidates.md
├── disposition.json        the authoritative machine-readable artifact
├── inventory.json          every tracked file with its git history
└── raw/                    one JSON per phase
```

## Dispositions

`KEEP` · `KEEP_PUBLIC` · `KEEP_INTERNAL` · `CANONICAL` · `MERGE` · `ARCHIVE` ·
`FEATURE_FLAG` · `DELETE_CANDIDATE` · `BROKEN` · `GENERATED` · `UNKNOWN`

Not DELETE/KEEP. Atlas needs the nuance: a node can be legitimate with no
shipping example, a module can be implemented-and-tested but not yet wired up,
and a workflow can be superseded without being wrong.

## Confidence

| level | meaning |
|---|---|
| `CERTAIN` | every dimension checked, all negative, and a superseding implementation exists |
| `HIGH` | strong evidence, one uncertainty remains |
| `MEDIUM` | likely redundant, but dynamic references or historical intent are unclear |
| `LOW` | sparse evidence |
| `UNKNOWN` | cannot determine safely |

**`UNKNOWN` means the evidence was insufficient — not that the file is dead.**
`tools/build_*.py` scripts land there by design: they are CLI entry points with
no importer, and that is the correct answer, not a scanner gap.

Ceilings applied before any verdict:

- protected categories (`TEST`, `DCC`, `MODEL_ADAPTER`, `SETUP`, `CI`,
  `CONFIG`, `ASSET`) → at most `MEDIUM`
- any directory containing a dynamic-loading marker → at most `MEDIUM`
- entry-point directories (`tools/`, `scripts/`) → at most `HIGH`
- generated / vendored artifacts → at most `HIGH`

On a healthy tree the CERTAIN set is empty. That is the expected result.

## Reading the output without re-litigating it

- `FEATURE_FLAG` from a `TEST_ONLY_REACHABLE` signal means implemented **and
  tested** but **unwired**. Read the last-commit date to tell new-work-not-yet-
  hooked-up from abandoned prototype — the audit cannot, and does not guess.
- "Registered but demonstrated by no workflow" is a **documentation and
  example-coverage gap**, not a deletion list.
- Provenance is not drift. `CHANGELOG.md` and `docs/development/design-rules.md`
  name files that were later removed and counts that were right when written;
  `config.json` exempts them, and "fixing" them destroys the evidence the
  design rules are cited from.
- Workflow duplicate clusters compare **node graphs**, never filenames.

## config.json

Every entry is a judgement the scanners cannot make from evidence, and each
carries the reason it was made — provenance docs, foreign path prefixes,
symbols that are absent on purpose, protected categories.

Adding a path there to silence a finding you have not understood defeats the
audit. If a finding is wrong, prefer fixing the scanner: an earlier revision of
`scan_nodes.py` read only dict literals and reported twelve live, registered
nodes as deregistered leftovers, and `scan_references.py` counted the
package-wide private helper `_require_numpy` as a reference, which hid the one
genuinely unwired module in the tree. Both were scanner defects wearing the
costume of findings.

## Applying the manifest (human procedure — there is no script)

1. Working tree clean. `git status` must be empty.
2. Dedicated cleanup branch, not `main`.
3. Recommend gstack `/guard` for the session.
4. Read `delete-candidates.md`. Only the `CERTAIN` block is eligible without
   per-file confirmation; everything else needs a decision.
5. Present the list before touching anything, and get explicit confirmation.
6. Delete in small batches, in this order:
   generated debris → certain dead scripts → broken and duplicate workflows →
   superseded experiments → duplicate documentation → legacy setup files.
7. After **every** batch: run the targeted tests, re-run
   `run_audit.py --check`, commit atomically. Never one commit of hundreds of
   deletions.

Prefer git history as the archive. Move something into `docs/archive/`,
`examples/archive/` or `experimental/` only when it still needs to be
*discoverable* — otherwise delete it and let history hold it. The public v1
tree should show what Atlas Camera is now, not every route used to build it.

## Release sequence

```
/health → /atlas-v1-audit → review the manifest → /guard →
apply in small batches → /review → /cso --comprehensive →
/document-release → /document-generate → /plan-devex-review → /devex-review → /ship
```

Do not invoke any of those automatically.

## Known limitations

- A script executed by a DCC host out-of-process leaves no trace this audit can
  read. That is why `DCC` is a protected category.
- Doc topic classification is keyword-based; the canonical-doc table is a
  starting point for `/document-release`, not a verdict.
- `knip` needs `npx`; without it the JS/TS half reports `NOT_AVAILABLE` rather
  than guessing.
- Near-duplicate workflow detection uses node-type Jaccard overlap, so two
  workflows with the same node types but different wiring read as similar.

## Recommended next command

**Review the audit.** Not "delete everything."
