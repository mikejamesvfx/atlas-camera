# AtlasShoot — iOS design brief

For a design agent or designer. Everything here is drawn from the existing Atlas
identity, not invented: the mark is `assets/atlas_camera_icon.png` (also
`.svg`), and the palette is taken from the Atlas workbench UI and the user-guide
design concepts in `assets/user-guide-designs/`.

---

## What the app is

Atlas recovers a 3D camera from a single photograph and projects the plate onto
derived geometry. Whatever the original camera could not see becomes a hole, and
today those holes are filled by invention — inpainting, predicted geometry,
smeared edges.

AtlasShoot replaces invention with photography. Atlas hands the phone a
**shooting brief**: a prioritised list of missing surfaces, each with the angle,
distance and resolution needed. The user photographs that material — paving,
brick, render, tarmac — and sends it back.

**The user is a working matte painter or photographer, on location, one-handed,
often in sunlight.** Not a consumer taking snapshots.

## The mark already states the personality

An amber "A" built from a surveyor's tripod with a **registration reticle** at
its centre, on warm near-black. Survey instrument crossed with a camera
viewfinder target. Lean into that: this is a measuring device that happens to
take photographs.

It also anticipates the product — the crosshair is an alignment target, which is
literally what the app's second phase does. Keep the reticle motif available for
the AR alignment state rather than inventing new iconography for it.

Preserve the mark. Do not redraw, recolour or restyle it.

## Palette

Atlas amber is the brand colour. Everything else is a warm neutral.

| role | hex | notes |
|---|---|---|
| Atlas amber | `#eaa03a` | the accent. Actions, active state, the reticle |
| deep ground | `#171411` | darkest surface |
| ground | `#1a1714` | primary background — warm near-black, never pure `#000` |
| graphite | `#6b6560` | secondary text, rules |
| ash | `#a09890` | tertiary text, disabled |
| bone | `#f5f2ed` | primary text on dark |
| pale | `#d4cfc8` | dividers on dark, secondary surfaces |
| signal red | `#d42b2b` | errors and refusals ONLY |
| ochre | `#b87300` | amber's darker partner — pressed states, shadows |

The neutrals are deliberately **warm** (brown-biased). A cool grey will read as
generic-app immediately and break the family with the desktop tool.

**Dark by default.** It matches the mark, matches camera-app convention, and
does not wreck night vision on a dusk shoot. But this app is used **outdoors**,
so contrast must survive direct sunlight: bone on ground, amber never on ochre,
and no thin light-weight type for anything load-bearing.

## Type

A condensed or technical sans for headings — the user-guide direction "Cinematic
Control" uses condensed display type, and that reads as instrument labelling.
Body text stays plain and highly legible; this is read at arm's length in bad
light.

Numbers matter here — angles, distances, pixels per metre. Use **tabular
figures** wherever they appear so a list of shots does not jitter.

## Screens

### 1. Shot list — the home screen

A prioritised queue. Each row carries: **subject** (e.g. "pavement, kerb"), what
is hiding it, the guidance line, and the key metrics.

Two row states must be **unmistakably different at a glance**:

- **Planar** — a normal surface. Shows an angle and a resolution target.
- **Volumetric** — an alleyway, doorway, recess. **No angle exists**; showing
  one would be a lie. Needs its own visual treatment and its own affordance
  ("align on site" rather than "shoot at 79°").

That distinction is the single most important thing on this screen. Do not let
it become a subtle badge.

Rows should also carry completion state — shot / not shot / sent — because this
is a checklist someone works through over an afternoon.

### 2. Shot detail

Everything needed to take one photograph, sized for a glance while holding a
camera:

- the guidance sentence, large
- angle, distance, and required pixels-per-metre as instrument readouts
- **the reference crop from the original plate** — the lighting and material
  target, shown large enough to actually match against
- any warnings, verbatim

### 3. Capture

Viewfinder-dominant. Minimal chrome. A live indicator of whether the current
framing meets the brief's resolution requirement.

The reticle from the mark belongs here as the alignment/confirm motif.

### 4. Review and send

Confirm the capture, then hand off. Sending goes over the user's own private
network — no cloud account, nothing uploaded to a service. That is a genuine
product value and worth stating in the UI rather than hiding.

## A statement the design must carry

The brief tells the user **what angle and what resolution** — but deliberately
**not what lighting**, because Atlas cannot measure sun direction or hardness
from a single plate.

That absence is a designed statement, not an oversight, and it is the thing most
likely to ruin a patch. The reference crop and a plain note ("lighting not
specified — match the reference") must be visible, not buried. Do not let a
screen full of precise numbers imply that lighting is handled.

## Do not

- Invent a new logo, or restyle the existing mark.
- Use pure black or cool greys.
- Use signal red for anything except an error or a refusal.
- Display any value Atlas did not measure as though it did.
- Show an incidence angle on a volumetric shot.
- Make it look like a consumer camera app. No filters, no social affordances,
  no playful illustration. It is a field instrument.

## Reference

- Mark: `assets/atlas_camera_icon.png` / `.svg`
- Existing directions: `assets/user-guide-designs/README.md` — the three
  established Atlas visual directions with their palettes. **"01 Cinematic
  Control" is the closest relative to this app** (near-black, Atlas amber, bone
  white, graphite).

  The three concept IMAGES beside that README are ~6 MB and deliberately not
  committed — this repo publishes to the ComfyUI registry and they are user-guide
  artwork, not app assets. Send them over Taildrop if the designer wants them.
- Data the UI must render: `docs/shoot_project.example.json`
- Field meanings: `docs/SHOOT_PROJECT_FORMAT.md`
