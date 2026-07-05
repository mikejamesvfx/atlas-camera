# Atlas Viewport — DOM-widget sizing investigation

**Date:** 2026-07-05
**Component:** `AtlasBlockoutViewport` (ComfyUI frontend, `atlas_camera/comfy/web/atlas_blockout.js`)
**Status:** Unresolved for the unified-node redesign; **reverted to the last known-good architecture for the `v0.1.0-beta` release**. No functionality lost — the viewport works as it did in the previous shipped version.

## Summary

We attempted to redesign the Atlas Viewport from a two-node layout (viewport + a
separate detached controls node) into a single unified node with a Resolve/
Premiere-style bottom transport bar, per a UX request. During implementation we
hit a severe ComfyUI 1.45 frontend bug: **the node's DOM-widget width
intermittently collapses to ~300px on certain relayout triggers** (clicking a
toolbar button, orbiting the 3D view then releasing the mouse, clicking the
node's title bar to select it). After roughly fifteen distinct fix attempts —
several of which chased the wrong layer of the problem — we identified one
confirmed root cause and shipped a partial fix for it, but a second collapse
path proved unfixable from the extension side within the time available. We
reverted the whole redesign to the previously-shipped, fully-working two-node
architecture for the beta. **No user-facing capability was lost; the unified
transport bar is simply deferred.**

## Background

Prior state (commit `b23ff65`, working): `AtlasBlockoutViewport` (the 3D canvas
+ Three.js render loop) and `AtlasViewportControls` (all buttons/toolbar/panels)
were **separate ComfyUI nodes**, linked by a data-free graph edge purely so
their frontend JS instances could find each other and reparent DOM elements.
This kept the viewport node's DOM-widget container **canvas-only**. It resized
and orbited correctly.

Requested change: merge everything into one node with a bottom-docked control
bar (grouped buttons + popovers for sliders/panels), matching a reference
extension's video-editor-style UI. This is a legitimate, contained frontend
change — no core/Python logic involved, isolated to one JS file.

## What we tried, in order

1. **`getMaxHeight` alongside the existing `getMinHeight`.** ComfyUI's
   `DOMWidgetImpl.computeLayoutSize` (confirmed by extracting the frontend's
   own sourcemap, `scripts/domWidget.ts`) treats an *undefined* `getMaxHeight`
   as "not growable," which explained an early snap-back-on-release. Setting a
   large constant `getMaxHeight` fixed *that specific* symptom but not the
   later ones.
2. **`node.size[1]`-derived height locks / holds** (post-hoc mutation each
   animation frame, and a time-boxed "hold the pre-click size" window after a
   toolbar click). These either did nothing (the mutation ran *after* layout,
   so `node.size` reported one value while the actual DOM widget was a
   different, smaller size) or introduced genuine feedback oscillation
   (a value derived from `node.size` was fed back into a sizing hook that then
   changed `node.size` again, every frame).
3. **`node.computeSize` width floor** (`Math.max(natural, widestSeenSoFar)`).
   Correct in principle — `computeSize` runs *during* layout, unlike a
   post-hoc mutation — but ultimately masked rather than fixed anything once
   we found root cause #1 below, and on its own did not survive an
   orbit-release collapse (that path does not go through `computeSize` at
   all — see below).
4. **A live debug HUD** (temporary, later removed) overlaid on the viewport
   showing `node.size`, the DOM-widget container's own `offsetWidth/Height`,
   the canvas's intrinsic buffer size, and whether ComfyUI's canvas thought a
   resize-drag was active (`app.canvas.resizing_node`) or the pointer was down.
   This was the single most useful step — it let us **prove**, rather than
   guess, that:
   - `node.size` could read one (large, correct-looking) value while the
     *actual DOM-widget container* was a completely different, much smaller
     value — i.e. the width had genuinely decoupled from `node.size`, and any
     fix that only touched `node.size` was chasing the wrong variable.
   - Every problematic size change during a drag correctly showed
     `resizing_node === node`; the collapses we cared about did **not** — they
     fired with both signals false or, worse, *during* an orbit interaction
     that also reports `pointer down`, which ruled out a simple "only allow
     size changes while a drag is active" gate.
5. **Root cause found (git archaeology + frontend sourcemap read):**
   Diffing the last known-good commit (`b23ff65`) against the broken version
   showed the *only* structural difference in the DOM-widget container was
   that the new toolbar had become a **flex sibling of the canvas inside the
   widget's container**, where previously that container held the canvas
   alone (controls lived on the separate node). Reading ComfyUI's own
   `computeLayoutSize` confirmed it hardcodes `minWidth: 0` for a DOM widget —
   so nothing in ComfyUI's layout protects width the way `getMinHeight`
   protects height, and an extra flex sibling in that container was enough to
   perturb the width computation on certain relayouts.
   **Fix:** make the toolbar an absolutely-positioned overlay docked to the
   bottom of the canvas (CSS `position:absolute`, no longer in the flex flow),
   so the DOM-widget container goes back to being canvas-only, matching the
   working version's structure. **This fixed the resize-drag collapse** —
   confirmed live by the user; resizing the node by dragging its corner then
   worked correctly and held.
6. **Orbit-release collapse (the remaining bug) — multiple further attempts,
   all unsuccessful:**
   - A monotonic "widest-seen" width floor applied in `computeSize` +
     `node.setSize` overrides. Did not catch it: the orbit-release (and
     node-title-click) collapse writes to `node.size` **directly**, bypassing
     both `computeSize` and `setSize` entirely, so no override on either
     function can intercept it.
   - An active per-frame "restore the width if it dropped" check in the
     render loop, first via direct `node.size[0]` mutation (silently
     ineffective — same decoupling as attempt #2), then via the "proper"
     `node.setSize()` call. The `setSize()` version worked for the isolated
     title-click case but, when combined with the per-frame render-loop
     timing, actively fought the orbit-drag's own continuous size writes and
     **broke orbiting itself** (regression, immediately reverted).
   - Removing `min-width:0` from the container/canvasWrap CSS, on the theory
     that the canvas's own intrinsic width (a "replaced element" min-content
     floor, the same CSS mechanism that caused an *earlier*, different bug
     this viewport already has documented history with) would passively floor
     the container width with no JS involved. Verified live: **did not stop
     the orbit-release collapse.**
   - At this point we were no longer forming falsifiable hypotheses about the
     mechanism — every remaining idea was another guess at *a* fix without a
     proven *cause* for this specific path, unlike root cause #1 above which
     we could actually observe and confirm. We stopped rather than continue
     guessing.

## Decision

Given:
- The redesign's core value (bottom transport bar, nicer control grouping) is
  a UX nicety, not new capability — every control still worked correctly in
  the old two-node layout.
- The orbit-release collapse is a genuine app-blocking regression for daily
  use of the viewport (it collapses on ordinary interaction, not just an edge
  case).
- We were past the point of principled, falsifiable debugging and into blind
  trial-and-error against an undocumented internal ComfyUI 1.45 layout path.

**We reverted `atlas_camera/comfy/web/atlas_blockout.js` and the corresponding
`CLAUDE.md` section in full to the last known-good commit (`b23ff65`)** before
cutting the beta. This is a pure revert — no partial/half-migrated state was
shipped. The `v0.1.0-beta` tag and `main`/`release/beta-0.1` branches all carry
the working, two-node viewport.

## What would be needed to revisit this

The one root cause we *did* confirm (toolbar-in-flex-container breaking
width) is fixed and could be reused if we take this on again. The blocking
unknown is specifically **what internal ComfyUI code path writes `node.size`
directly on orbit-release / title-bar-click**, bypassing both `computeSize`
and `setSize`. Two credible next steps, neither attempted yet:
1. Instrument via a `Proxy` (or `Object.defineProperty` accessor) on
   `node.size` itself rather than on the functions that are supposed to
   compute it, to get a stack trace at the exact write that causes the
   collapse — this is the direct-observation step we didn't get to.
2. Check whether ComfyUI's newer **Vue-based node rendering** ("Vue nodes")
   mode sidesteps this legacy DOM-widget layout path entirely; if so, the
   fix might be "wait for/opt into that," not a workaround in the current
   layout system.

## Files touched by the (reverted) attempt

- `atlas_camera/comfy/web/atlas_blockout.js` — all changes reverted to
  `b23ff65`.
- `CLAUDE.md` — the "Detached viewport controls" / unified-node documentation
  reverted to `b23ff65`'s version.
- No Python/core changes were involved at any point; this was entirely
  contained to one frontend JS file.
