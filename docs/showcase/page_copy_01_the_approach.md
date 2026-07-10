# Showcase Page — Concept 01 · "The Approach"
*Full page copy, usable as the template for all ten. Section headers map to site
components; body text is final, not placeholder. Swap the bracketed [MEDIA] notes for
assets.*

---

**KICKER:** Showcase · Aerial
**TITLE:** The Approach
**LOGLINE:** A single AI still. A real camera push over a living forest to a castle in the
mist. No video model touched this shot.

[MEDIA: hero — the final baked move, autoplay/loop, muted, ~8s]

---

## Intro

This started as one image — a fantasy castle above a pine forest, generated in a text-to-image
model. There is no second frame, no video diffusion, no drone footage. Every bit of parallax
you see — the canopy sliding underneath, the towers separating from the ridgeline as we
descend — was reconstructed by Atlas Camera from that one still, and rendered as a genuine 3D
camera move you can hand straight to Nuke or Maya.

## The Shot

A high, slow drone push crests a wall of pines. The canopy rolls beneath us in depth, a valley
opens, and a spired castle lifts out of the morning fog. As we close the distance the camera
cranes gently down, and the towers part from the far ridge with the parallax of a real aerial
plate.

## How It's Made — five steps, one still

**1 · Solve the camera.**
Atlas's learned solver (GeoCalib prior) recovers focal length, tilt, and horizon directly from
the image — and it's built to be robust on AI-generated frames, where classic vanishing-point
math struggles. [MEDIA: still with VP/horizon diagram overlay]

**2 · Derive the geometry.**
The `forests` preset builds a dense relief mesh that follows the canopy's real depth contours,
while a `towers & spires` pass captures the castle's silhouette. `Merge Geometry` combines them
— foreground terrain and hero structure in one scene. [MEDIA: grey shaded geometry, no texture]

**3 · Project the plate.**
Hit **📽 Project** and the original image is cast back onto that geometry from the recovered
camera. From Camera View it's pixel-perfect — indistinguishable from the source still. [MEDIA:
projected view matching the still]

**4 · Move the camera.**
In Camera Path mode, a crane-down + push is keyed against the live geometry. Atmospheric fog on
the far backdrop hides the flatness of distant layers — the same trick a matte painter uses by
hand. [MEDIA: the move, with a small camera-path HUD]

**5 · Hand it off.**
Bake to a frame batch for the comp, or export the relief mesh (OBJ/GLB) and a time-sampled USD
camera into Maya, Blender, or Nuke — correctly scaled, ready to light and dress. [MEDIA: same
mesh open in a DCC viewport]

## What this shot demonstrates
`Learned solve` · `Forests relief mesh` · `Towers & spires` · `Merge geometry` · `Camera Path
(crane + push)` · `Atmospheric backdrop` · `USD / mesh export`

## Pull quote (USP band)
> One image in. A moving, projectable, exportable 3D shot out — without a single frame of video
> generation.

## Tech spec (sidebar)
- **Source:** 1× text-to-image still, 16:9
- **Solve:** AtlasLearnedSolveFromImage (GeoCalib)
- **Geometry:** forests relief mesh ⊕ towers/spires (AtlasMergeGeometry)
- **Move:** Camera Path — crane-down + dolly-in, ease-in-out
- **Output:** baked frame batch (Video Combine) + relief mesh (GLB) + USD camera
- **DCC:** opens textured in Maya / Blender / Nuke

## CTA
**[ Download this workflow ]** — the exact ComfyUI graph, ready to run.
**[ Read the User Guide ]** — from still to shot in ten minutes.

## Nav
← *Winter's Eve* | *Cathedral of Light* →

---

### Template notes (for the other nine pages)
- Keep the **five-step "How It's Made"** arc identical everywhere — it *is* the Atlas pitch,
  repeated. Only the feature chips and media change.
- Every page opens with the **"one still → moving shot, no video model"** promise in the intro.
- The **tech-spec sidebar** stays a fixed shape (Source / Solve / Geometry / Move / Output /
  DCC) so the pages read as a consistent body of work.
- Reserve one optional **"Without vs. With"** slot per page for the concepts where a black-hole
  failure is the compelling before (3 · orbit patch, 5 · street plates, 2 · interior columns).
