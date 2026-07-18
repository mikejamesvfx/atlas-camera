# Atlas canonical workflows

These are the recommended entry points. Use them in order of ambition, not image quality:

| Tier | Workflow | Plate | What it teaches |
|---|---|---|---|
| Quickstart | `atlas_canonical_quickstart_ghosttown_workflow.json` | `ghosttown_32bit_acescg.exr` | One plate → MoGe-2 metric depth → masked 1024 relief, with an optional camera move. Deliberately SIMPLE — the Photoshop/cleanplate material lives in its own tier below. |
| Cleanplate | `atlas_canonical_cleanplate_ghosttown_workflow.json` | `ghosttown_32bit_acescg.exr` | The 🧽 CLEANPLATE DOCTRINE end-to-end, laid out as a 6-group story: (0) paint the fg occluders out in Photoshop FIRST and inject at the port — the base scene solves/meshes/projects/EXPORTS the clean source-quality pixels; (1) the original car+sign ride an FG OCCLUDER layer (SAM3 'rusty car and fallen sign' — simple noun phrases + 'and'; or the parked 🎨 artist-matte override); (2) optional 🧽 CleanPlateStack for up-to-4 painted strata; (3) Output Desk + viewport; (4) move/exports/debug; (5) 📐 Angle Patch for RESIDUAL stretching only, gated until Extract Angle. Ships runnable out-of-box (port bypassed, stack unwired). |
| Production | `atlas_canonical_production_templecity_workflow.json` | `atlas_00022_templecity.png` | Elevated scale override, sky dome, clean masks, retopology, and Maya/Nuke review. |
| Trust | `atlas_canonical_trust_d810raw_workflow.json` | `CameraRaw/DSC_2327.NEF` (repoint at any RAW) | The 2026-07-18 trust tier in one graph: 📷 RAW-native solve (EXIF focal + camera-body sensor via `raw_meta`, lensfun undistort, linear EXR sidecar) → 📐 scale dial → 🛡 depth-outlier mask + quad-coherent relief → 🩺 scene-health gate (ON THIS PLATE it deliberately WARNS — the bottom window haze flips GeoCalib's gravity; acknowledge and the warning is stamped into every artifact) → 🔍 debug report + viewport (ℹ HUD scale trust) → USD/Nuke exports each carrying `atlas_project.json` + identity comments. Needs `[raw]`(+`[raw-lens]`) extras. |
| Research | `atlas_canonical_research_newyork_lari_workflow.json` | `newyork_Birdseye.png` | Counted building reference, restricted LaRI hidden geometry, inpaint crop/stitch, and DCC export. |

All three have been validated against the live Atlas node catalog and executed successfully in ComfyUI. The workflow titles are intentionally explicit about the intended user journey and the viewport title calls out the project/gray review state.
