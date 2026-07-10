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
