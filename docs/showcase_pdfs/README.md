# Showcase documentation — PDF exports

Print-to-PDF snapshots (Chrome headless, backgrounds + embedded imagery
included) of the illustrated showcase documentation. The live pages are
Claude artifacts; these PDFs are the offline/shareable copies, exported
2026-07-17 against the v0.6.0 beta.

| PDF | Live page |
|---|---|
| `atlas_showcase_field_guide.pdf` | The master field guide — all 11 workflows, gates callout, dolly filmstrip, per-plate tweak lists |
| `atlas_wf_solve_lab.pdf` | 01 · The Solve Lab (coastal alley, VP diagnostics, scale tiers) |
| `atlas_wf_city_blocks.pdf` | 02 · City Blocks — aerial preset + counted-storey scale (NYC) |
| `atlas_wf_dmp_anchored.pdf` | 03 · DMP angle pt 1 — ground-anchored walls (NYC) |
| `atlas_wf_dmp_xray.pdf` | 04 · DMP angle pt 2 — LaRI X-ray, fg-band restrict (NYC) 🔬 |
| `atlas_wf_composable.pdf` | 05 · Composable geometry — the merge chain (temple city) |
| `atlas_wf_oceancastle.pdf` | 06 · OCIO layered DMP (ACEScg float pipeline) |
| `atlas_wf_hangar.pdf` | 07 · Interior doctrine (MoGe normals, room fit, patch loop) |
| `atlas_wf_ghosttown.pdf` | 08 · One node → a camera move (AtlasInput + bake) |
| `atlas_wf_jungleruins.pdf` | 09 · Organic relief + quad retopo |
| `atlas_wf_portal.pdf` | 10 · Roll trim + 2.39:1 conform + USD round-trip |
| `atlas_wf_wreck.pdf` | 11 · X-ray with a restrict mask (the honest stress test) 🔬 |

Every deep-dive carries the same structure: the story, live viewport
captures (📽 projected vs the grey mesh at multiple orbit angles, plus that
workflow's special overlay — 📊 solve diagram, 🩻 X-ray provenance, 📏 band
box, 🎨 layer debug, ℹ HUD), the nodes on stage, per-plate retune knobs, the
run-verified result, and requirements.

Workflows themselves: `examples/showcase/`. Findings log:
`examples/showcase/README.md`.
