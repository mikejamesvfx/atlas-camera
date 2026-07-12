# Published artifact sources

Source HTML for the Claude-published documentation pages. Committed so future
sessions can update and republish them without reconstructing from a fetch —
the lesson of 2026-07-11, when two pages' sources had died with an old
session scratchpad.

| Page | Source | Published URL |
|---|---|---|
| 🏗 Staged Master — Operator's Guide | `atlas_staged_master_guide.html` | https://claude.ai/code/artifact/174fccee-3200-43f9-83c5-7875ada0b8cf |
| 🥞 Layered Projection — Build-Up Guide | `atlas_layered_guide.html` | https://claude.ai/code/artifact/77b10784-a6d5-4def-89bd-84cbfaabc21e |
| 🎞 Examples Catalog | `atlas_examples_catalog.html` *(rebuilt 2026-07-12 for the v0.4.0 two-workflow catalog + 4-plate sample pack)* | https://claude.ai/code/artifact/186c3a6a-a778-40f0-8f39-fe29cfa6aace |
| 📊 Technical Details | *(source not recovered — regenerate charts by script when next updated)* | https://claude.ai/code/artifact/4781289c-50dd-47fc-8571-1ef67513b7ba |

To republish after editing a source: pass the file plus the page's URL as
`url` to the Artifact tool (any session). Keep each page's favicon stable.
The files are page BODIES (title/style/main) — the publisher wraps them in
the document skeleton; don't add doctype/html/head/body.
