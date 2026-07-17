# Atlas canonical workflows

These are the three recommended entry points. Use them in order of ambition, not image quality:

| Tier | Workflow | Plate | What it teaches |
|---|---|---|---|
| Quickstart | `atlas_canonical_quickstart_ghosttown_workflow.json` | `ghosttown_32bit_acescg.exr` | One plate → MoGe-2 metric depth → masked 1024 relief, an optional camera move, and the 📐 Angle Patch → Photoshop round trip (extract the holes at an exact orbit pose, repair in Photoshop, reproject from that same pose — group 4, gated until you click 📐 Extract Angle). |
| Production | `atlas_canonical_production_templecity_workflow.json` | `atlas_00022_templecity.png` | Elevated scale override, sky dome, clean masks, retopology, and Maya/Nuke review. |
| Research | `atlas_canonical_research_newyork_lari_workflow.json` | `newyork_Birdseye.png` | Counted building reference, restricted LaRI hidden geometry, inpaint crop/stitch, and DCC export. |

All three have been validated against the live Atlas node catalog and executed successfully in ComfyUI. The workflow titles are intentionally explicit about the intended user journey and the viewport title calls out the project/gray review state.
