# Export-Only Retopology (quad / decimate / smooth)

**Status:** ported onto main 2026-07-16 from the `atlas-ollama` research clone
(H:\ComfyUI_windows_GLM, local branch `claude/atlas-pytest-pil-error-f87721`,
research dated 2026-07-15). The research doc there also carried a fan-fill
hole-fill implementation — that half was **discarded**: main's ear-clipping
`mesh_repair.py` (see `docs/dev/archive/atlas_mesh_repair_solution.md` and the
CLAUDE.md design bullet) superseded fan-fill after it was measured emitting
triangles outside non-convex holes. Only the retopology half was new, and it
is what this document describes.

**Module:** `atlas_camera/core/mesh_retopo.py` · **Tests:**
`tests/test_mesh_retopo.py` (20: 18 always-on + 2 dep-contract inverses).

## What it is

The same export-only doctrine as the interior hole-fill, extended to
**retopology**: Atlas relief meshes are dense, irregular, and torn — fine for
live 📽 projection (texels assigned by ray) but a poor handoff for a DCC
retopo / boolean / 3D-print pass (100k+ triangles, skinny slivers at tear
edges, open boundary loops). `mesh_retopo.py` retopologizes the **exported**
mesh (never the live viewport projection mesh, never `solve.proxy_geometry`)
so the OBJ/GLB handed to Maya / ZBrush / Blender is clean and light.

* **Export-only** — retopology changes vertex count, which breaks the 1:1
  vertex-UV invariant the live projection depends on, and quad remeshing /
  decimation smooths the *deliberate* silhouette tears the matte-painting
  projection relies on. It runs once, on the resolved `ReliefMesh`, **after**
  the interior hole-fill (so it retopologizes the capped mesh) and before the
  OBJ/GLB writers.
* **CPU-only** — geometry processing, not a GPU workload. Every path is
  M2-safe (macOS arm64 wheels exist OR it is pure numpy).

## Three paths, guarded

1. **Quad retopology** (lead) — `pyinstantmeshes` (BSD), a Python wheel
   wrapping Instant Meshes (orientation-field quad remeshing). Outputs N×4
   quads → triangulated before writing (`_triangulate_quads` handles real
   quads, degenerate quad-encodes-one-triangle rows, and (M,3) passthrough).
2. **Quadric decimation** — `fast-simplification` (BSD) backing
   `trimesh.simplify_quadric_decimation`. Pure decimation (no remeshing) —
   keeps the original topology class, just fewer faces. Face target ≈ 2× the
   vertex-count widget.
3. **Smooth / relax** — `trimesh` pure-numpy Taubin smoothing (MIT, light
   install — the practical baseline). Topology **unchanged** (same faces,
   same vertex count) — only vertex positions move, so the 1:1 vertex-UV
   mapping is preserved and UVs are **not** regenerated.

Each path raises an informative `ImportError` (+ install hint) when its
optional dep is absent — the discipline every optional import in
`atlas_camera` follows. None of these are runtime deps of the package.

## The UV-loss problem and its fix (the load-bearing math)

Quad remeshing and decimation change the vertex count, which breaks the 1:1
vertex-UV mapping the OBJ/GLB writers depend on (`f {a+1}/{a+1} ...`). The fix
is `regenerate_projective_uvs` — regenerate the same projective UVs for the
new vertices by projecting each through the recovered camera → image pixel →
UV, mirroring the bake in `relief_mesh.build_relief_mesh`:

```
Forward bake:   p_cam = ((uu-cx)/fx·d, -(vv-cy)/fy·d, -d)
                world = p_cam @ R_cw.T + cam          # then ray-preserving rescale about cam
                u = uu/(W-1) ; v = 1 - vv/(H-1)

Inverse (here): p_cam = (world - cam) @ R_cw           # world → camera frame
                px = cx - fx·x_c/z_c   (z_c < 0 in front)   →  px == uu
                py = cy + fy·y_c/z_c                         →  py == vv
                u = px/(W-1) ; v = 1 - py/(H-1)
```

**Why the rescale-about-cam, `floor_clamp`, and band near-clip don't break the
inverse:** all three move vertices *along their own view rays*, so a vertex's
projected pixel — and therefore its UV — is invariant under them. The tests
build the forward bake and assert the inverse matches to 1e-5 under identity,
under `scale=3.5`, and under a rotated+translated camera. Behind-camera
vertices (`z_c >= 0`, undefined projection) clamp to the image boundary so the
writers never see NaN UVs. The smooth path keeps the existing UVs.

## Node wiring (`AtlasExportReliefMesh`)

Five optional widgets appended **after** the four hole-fill widgets (all
default to off/disabled, so every saved workflow keeps working):

| Widget | Type | Default | Meaning |
|---|---|---|---|
| `retopo_method` | combo `off/quad/decimate/smooth` | `off` | Master switch. |
| `retopo_target_vertex_count` | INT (4–2e6) | `2000` | Target verts (quad) / ~2× this in faces (decimate). Ignored by smooth. |
| `retopo_smooth_iterations` | INT (0–100) | `0` | quad: Instant Meshes post-smooth; smooth: Taubin strength; decimate: ignored. |
| `retopo_crease_angle` | FLOAT (0–180) | `30.0` | quad only: orientation-field crease angle (deg). |
| `retopo_pure_quad` | BOOLEAN | `False` | quad only: force pure-quad output. |

`export()` runs it after the hole-fill + open-loop census and **before**
`preview_solve` is built — so the fill report's "still open" counts describe
the pre-retopo mesh, a `🔻 retopo` line is appended to the same on-node
report, and `preview_solve` (and therefore the wired viewport) carries the
mesh ACTUALLY written. `apply_retopo` raises `ValueError` if a
vertex-count-changing method is selected without solved intrinsics, so a bare
solve can't silently produce an untextured mesh (use `smooth` there).

## Live verification (2026-07-16)

The three `examples/retopo/` demo workflows (generated by
`examples/retopo/_generate.py`, validated by `tools/validate_ui_workflow.py`,
run headless via `tools/run_ui_workflow.py`) were run to completion on the
real ComfyUI against `atlas_monument_valley.png` — quad (hole-fill 48 →
pure-quad 3000 target + USD camera), decimate (1500-vert budget + viewport),
and smooth A/B (off vs Taubin ×12) — see the run numbers in the examples'
README.

## Deps recap

```
pip install pyinstantmeshes        # quad (BSD, CPU, arm64 wheels cp311-314)
pip install fast-simplification   # decimate (BSD, CPU, arm64 wheels)
pip install trimesh                # smooth (MIT, pure python)
```
