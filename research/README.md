# research/ — experimental work, not part of the supported v1 API

Everything under this directory is **research**. It is not imported by
`atlas_camera`, not registered as a ComfyUI node, not covered by the public
API, and not subject to the compatibility promises the shipped package makes.
Treat it as a lab notebook that happens to run.

Nothing here is required to install, run, or use Atlas Camera. A user who
deletes this directory loses no product capability.

## What is in here

### `volfill/`

The hidden-volume / X-ray geometry investigation: whether a predicted
occluded-space volume can be turned into geometry Atlas can project onto, and
how far off the result is from measured truth.

It backs `AtlasLoadHiddenVolume` 🧊🔬 (experimental, `ATLAS_EXPERIMENTAL=1`) —
the node loads an externally-produced volume, and these scripts are how such a
volume gets produced and evaluated. The written-up conclusions live in
[docs/research/FLASH3D_VOLFILL_ATLAS_EVALUATION.md](../docs/research/FLASH3D_VOLFILL_ATLAS_EVALUATION.md).

Its dependencies are **not** declared in `pyproject.toml` — see
`volfill/requirements-win.txt`. Several scripts additionally need Blender, a
user-cloned upstream research repo, or plates that are not distributed.

## Rules for anything added here

1. **No product code may import `research/`.** The dependency direction is
   one-way: research may import `atlas_camera`, never the reverse.
2. **It carries no compatibility guarantee.** Rename, rewrite or delete freely.
3. **Conclusions belong in `docs/`, not here.** When an experiment settles
   something, write the finding into the design rules or a guide and let git
   history hold the scripts that produced it.
4. **If it stops being an active line of enquiry, delete it.** Git history is
   the archive; a dormant experiment in the public tree only costs the reader
   time working out whether it matters.
