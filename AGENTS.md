# AGENTS.md

Guidance for AI coding agents (Codex, and any tool that reads AGENTS.md)
working in this repository.

**The single source of truth is [CLAUDE.md](CLAUDE.md)** — commands, the
architecture map, the ComfyUI node catalog, coordinate conventions, and the
key design rules (with the reasoning and live-verification history behind
them). Read it before changing code; this file used to carry a fork of that
content and the two had already drifted, so it now just points there.

Repo-wide expectations, agent or human:

- Run `python -m pytest -q` before and after changes; the suite is designed
  to pass without torch (heavy tests skip via `importorskip`).
- The core package (`atlas_camera/core`) takes no third-party runtime
  dependencies; optional features guard their imports with informative
  errors.
- Coordinate conversions happen only at adapter boundaries — never in core.
- ComfyUI widget lists are positional in saved workflows: **append new
  widgets, never insert** (see CLAUDE.md's example-workflow refresh note).
- Experimental (🔬) nodes register behind the `ATLAS_EXPERIMENTAL` gate; the
  `experimental` branch differs from `main` only by that default.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
