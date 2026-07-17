# Atlas canonical workflows

These are the three recommended entry points. Use them in order of ambition, not image quality:

| Tier | Workflow | Plate | What it teaches |
|---|---|---|---|
| Quickstart | `atlas_canonical_quickstart_ghosttown_workflow.json` | `ghosttown_32bit_acescg.exr` | The 🧽 CLEANPLATE DOCTRINE end-to-end: paint the fg occluders (car + sign) out in Photoshop FIRST and inject the cleanplate at the port — the base scene solves/meshes/projects the CLEAN pixels at source quality (and that is what the DCC exports carry), while an FG OCCLUDER layer keeps the original car+sign on their own depth geometry (deterministic SegFormer matte). 📐 Angle Patch (group 4, gated) is demoted to residual cleanup — small stretching at grazing angles, never whole-object paint-outs. Ships with the port BYPASSED so it runs out-of-box on the raw plate. |
| Production | `atlas_canonical_production_templecity_workflow.json` | `atlas_00022_templecity.png` | Elevated scale override, sky dome, clean masks, retopology, and Maya/Nuke review. |
| Research | `atlas_canonical_research_newyork_lari_workflow.json` | `newyork_Birdseye.png` | Counted building reference, restricted LaRI hidden geometry, inpaint crop/stitch, and DCC export. |

All three have been validated against the live Atlas node catalog and executed successfully in ComfyUI. The workflow titles are intentionally explicit about the intended user journey and the viewport title calls out the project/gray review state.
