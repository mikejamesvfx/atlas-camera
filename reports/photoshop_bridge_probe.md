# Photoshop (Beta) paint bridge — probe report

**Date:** 2026-08-21 · **App:** Adobe Photoshop 27.4.0 (Beta), Win32 ·
**OCIO config:** `C:\OCIO\fn-nuke_cg-config-v1.0.0_aces-v1.3_ocio-v2.1.ocio`
(sha256 `c4bbc97455a9d4e1…`, 14 colourspaces / 5 displays)

Every claim below is either a string extracted from the shipped binary or a
number from a live run. Nothing here is from documentation.

## Rung ladder — where we are

| Rung | What it needs | Status |
|---|---|---|
| **D** — vendor-neutral tooling, manual PS leg | Phase 1+2 suites green | ✅ **achieved** |
| **C** — COM handshake | version + verified install path | ✅ **achieved** |
| **B** — colour-managed file leg | 32-bit + OCIO engaged | ✅ **achieved 2026-08-22** |
| **M1** — PS colour math vs Atlas | max abs < 1e-3 | ✅ **PASS at 3e-8** |
| **M3** — lossless file round trip | max abs < 1e-4 | ✅ **PASS** (same 3e-8 run) |
| **B′** — Smart Object container | mechanism works | ⚠️ **partly** — see below |
| **MANUAL** — colour-exact handoff + hand-painted mattes | round trip < 1e-4 | ✅ **achieved at 1.4e-6** |
| **A** — scripted generative fill | containment ≥ 0.99 | ⛔ **blocked** — `syntheticFill` unavailable EVERYWHERE |

## The manual lane — delivered, and it is the one that works

Scripted generative fill is unreachable (below), so the shipped lane is: Atlas
hands over a plate, a human paints in Photoshop, Atlas takes it back with the
colour intact and lifts hand-drawn mattes out as masks. `tools/photoshop_handoff.py`.

**Colour, measured end to end on the boiler plate:**

```
Atlas -> ACES2065-1 -> Photoshop 32-bit OCIO -> float TIFF -> Atlas
  max abs delta : 0.00000143
  mean abs delta: 0.00000004
  frac < 1e-4   : 1.0000        GATE <1e-4: PASS
```

**Hand-painted mattes** ride home as extra TIFF channels (Select > Save
Selection, then Save As TIFF with "Alpha Channels" ticked) and come back as one
PNG mask per channel, coverage reported:

```
  - boiler   channel4  coverage 15.55%  mattes/boiler.png
  - sky      channel5  coverage  9.97%  mattes/sky.png
```

Two contract details that are silent failures if ignored:

* **TIFF does not preserve Photoshop's channel names** — a channel the artist
  called `matte_boiler` arrives as `channel4`. Mattes therefore come back in
  document order and `--matte-names` labels them positionally.
* **Photoshop injects its own transparency alpha as the FIRST extra channel**,
  normally empty. Labelling by raw channel index made it consume the first
  name and shifted every matte by one — the artist's `boiler` landed on
  nothing and `sky` landed on the boiler. Labels now apply to the KEPT
  channels, and empty ones are skipped with a reason rather than shipped as
  blank masks.

### The 2700x precision bug this uncovered

The first end-to-end round trip measured **2.9e-3**, and Photoshop was not the
cause — an Atlas-only control with Photoshop removed measured the same:

| path | max abs |
|---|---|
| Atlas only, Rec709→AP0→Rec709 | 0.00289917 |
| Atlas only, +AP1 hop | 0.00289929 |
| with Photoshop in the loop | 0.00289929 |

Photoshop contributed **1.2e-7**. The rest was `read_plate` decoding at the
file's NATIVE type and letting `ImageBufAlgo.colorconvert` run in **half
precision**. Forcing the read to float:

```
convert at NATIVE (half) : max 0.00292969
convert at FLOAT         : max 0.00000107      <- 2700x
```

This affected **every colour conversion of every half EXR** — which is most of
them, since `AtlasLoadRAW` writes half sidecars and `write_exr` defaults to
half. It silently set the noise floor for every colour-managed handoff Atlas
has ever done. Fixed in `plate/oiio_io.py` (`buf.read(0, 0, True, oiio.FLOAT)`)
with a regression test that round-trips a half plate through ACES2065-1 and
fails above 1e-4.

## Rung B′ — the Smart Object mechanism works; the fill does not run anywhere

**The container architecture holds.** Placing a 16-bit display-referred TIFF
into the 32-bit OCIO document:

```
container_bits       : THIRTYTWO        (before placing)
placed               : ok
container_bits_after : THIRTYTWO        (container never changed mode)
layer_kind           : LayerKind.SMARTOBJECT
so_opened            : boiler_srgb16.tif
so_bits              : SIXTEEN          DocumentMode.RGB
```

So the 32-bit OCIO container with a 16-bit Smart Object inside it is real, and
`placedLayerEditContents` opens that inner document at 16-bit as intended. The
whole-frame `Image > Mode` round trip — which would have shifted every unedited
pixel — is avoided.

One caveat: the placed layer's `smartObject` descriptor exposes only
`placed, documentID, compsList, linked, fileReference`. **No
`placedLayerOCIOSpace` / `placedLayerOCIOConversion`.** Those keys exist in the
action dictionary but are not populated on a layer placed this way, so the
input-space declaration is not (yet) reachable by this route.

### `syntheticFill` is unavailable in EVERY context tested

| context | result |
|---|---|
| 32-bit OCIO container | `The command "Generative Fill" is not currently available.` |
| 16-bit Smart Object contents | identical error |
| **plain 8-bit RGB document, selection verified** (bounds 300,300,700,700) | **identical error** |

The 8-bit case is the control, and it is decisive: 8-bit RGB with a live
selection is the configuration Adobe documents as *supported*. Failing there
means the blocker is **not** bit depth, **not** OCIO, **not** Smart Objects and
**not** a missing selection.

Two candidate causes remain, and they were NOT separated:

1. **Generative Fill is not exposed to ExtendScript / Action Manager at all.**
   Consistent with the research finding that OCIO has no documented UXP
   scripting surface and that a 2024 developer request for scripting access
   went unanswered.
2. **No Firefly entitlement / not signed in.** `app.featureEnabled(...)`
   returned false, but it returns false for unrecognised feature names too, so
   that is not evidence.

A menu-enabled-state probe was attempted to separate them and returned
`no-menu-entry` for **every** command including `openAsOCIO` — which demonstrably
works. The probe form is wrong; nothing should be concluded from it.

**The discriminator is a manual test:** open a document, make a selection, and
use Generative Fill from the UI. Works manually ⇒ the scripting surface is the
blocker. Fails manually ⇒ entitlement.

Until that is settled, the Smart Object architecture is **untested, not
refuted** — the mechanism carrying it works; the payload never ran.

## Rung B — achieved (2026-08-22, Photoshop 27.11.0)

After enabling **Edit ▸ OpenColorIO Settings ▸ Enable OpenColorIO Features** and
setting **Document Depth: 32-bit**:

```
connected: Adobe Photoshop 27.11.0
opened:    boiler_master-1 3876x2589 BitsPerChannelType.THIRTYTWO DocumentMode.RGB
```

Config: both sides on the **built-in** `cg-config-v4.0.0_aces-v2.0_ocio-v2.5`,
with `$OCIO` deliberately **unset** — see "Which config is authoritative" below.

### Photoshop ASSUMES an OCIO EXR is ACES2065-1

The load-bearing finding of this rung. Handed a plate tagged
`lin_rec709_scene`, Photoshop applied a pure linear 3×3 to the pixels. Fitting
that matrix from the data gave a **residual of exactly 0.0**, and the matrix is:

```
[ 1.4514 -0.2365 -0.2149]        canonical AP0 -> AP1 (ACES2065-1 -> ACEScg)
[-0.0766  1.1762 -0.0997]        matched to 4.6e-5
[ 0.0083 -0.0060  0.9977]
```

**Photoshop does not read the EXR's colourspace tag.** It assumes ACES2065-1
and converts into the working space. Feeding it Atlas's Rec.709-linear RAW
sidecar therefore mis-reads the primaries silently, with no error and a
plausible-looking picture — exactly the hazard `docs/USER_GUIDE.md` warns about.

**The fix is Atlas-side, and it is the doctrine one:** convert to ACES2065-1
before handing over. Then Photoshop's assumption is *correct* and the chain is
exact.

### M1 / M3 — measured

Handing over an ACES2065-1 plate and comparing Photoshop's output against
Atlas's own conversion of the same file:

```
max abs delta : 0.00000003        mean 0.00000001
GATE <1e-3 (M1 parity)  : PASS
GATE <1e-4 (M3 lossless): PASS
```

**Photoshop's colour math is numerically identical to Atlas's**, to float32
precision. This settles the doctrine objection to the Smart Object architecture
by MEASUREMENT rather than by prohibition: the gate is not measuring
Photoshop's colour math, because Photoshop's colour math agrees with ours.

(An earlier comparison read 2.1e-3 max. That was the *baseline*, not
Photoshop: one-hop `Rec709→AP1` in Atlas versus two-hop `Rec709→AP0` then
`AP0→AP1`, through a half-float source. Different routing, not a defect.)

### The EXR export trap

Both obvious routes to writing EXR from a script fail, one of them silently:

* `doc.saveAs(file)` with a **`.exr`** extension **reports success and writes a
  460 MB PSD.** Photoshop does not infer format from the extension.
* Driving ProEXR via Action Manager needs the plugin's registered class ID for
  the `as` descriptor. Every plausible identifier — `OpenEXR`, `EXRFormat`,
  `ProEXR`, `exr`, `openEXR`, `ProEXR EZ` — returned *"General Photoshop error
  occurred. This functionality may not be available in this version of
  Photoshop."* Not guessable, and a wrong guess is indistinguishable from the
  feature being absent.

The exchange format is therefore an **uncompressed 32-bit float TIFF**
(`TiffSaveOptions`, `TIFFEncoding.NONE`) — lossless, typed, and read directly
by OIIO. That is what carried the 3e-8 result above.

Note `Document.close()` **is** implemented in Photoshop (unlike Affinity), so a
scripted session can start from a clean slate rather than accumulating
documents and acting on the wrong `activeDocument`.

## Which config is authoritative

Measured, not assumed:

```
ocio://default  ==  Photoshop's Builtin cg-config-v4.0.0_aces-v2.0_ocio-v2.5 : True
fn-nuke_cg      ==  Photoshop's Builtin                                      : False
```

OIIO's `ocio://default` **is** the config Photoshop defaults to (25 colourspaces
/ 8 displays, identical listings). So with `$OCIO` unset and Photoshop on
Builtin, **both applications are already on the same config** and nothing needs
pinning.

This reverses the original plan. Pinning `$OCIO` to
`C:\OCIO\fn-nuke_cg-config-v1.0.0_aces-v1.3_ocio-v2.1.ocio` would *create* the
mismatch it was meant to prevent: it drops Atlas to a 14-space ACES 1.3 config
while Photoshop stays on the 25-space ACES 2.0 one, losing `sRGB - Display`,
the Display P3 / Rec.2100 displays and the encoded-Rec.709 family.

The `$OCIO` work still earned its place — it proved Atlas is config-portable
and found four real bugs — but it should not be *used* for this bridge.

## Rung C — achieved

```
connected: Adobe Photoshop 27.4.0
install:   C:\Program Files\Adobe\Adobe Photoshop (Beta)
OCIO seen by Photoshop: C:\OCIO\fn-nuke_cg-config-v1.0.0_aces-v1.3_ocio-v2.1.ocio
```

Two things this settles:

* **The Beta is addressable unambiguously.** `Photoshop.Application.BETA` is a
  registered ProgID pointing at
  `...\Adobe Photoshop (Beta)\Photoshop.exe /Automation`. Photoshop 2026 is
  also installed, so the specific ProgID matters; the client additionally
  verifies `app.path` before doing anything.
* **`$OCIO` propagates into Photoshop's process.** The per-process scoping in
  `atlas_camera/paint/ocio.py` put the config on Photoshop's environment, and
  Photoshop reports back the exact path. Both applications can therefore
  resolve *the same config file*, verified by sha256 rather than assumed from a
  matching colourspace name.

## Rung B — blocked: OpenColorIO is compiled in but not ENABLED

Opening the boiler plate with the `openAsOCIO` boolean key set:

```
opened: boiler_master-1 3876x2589 BitsPerChannelType.SIXTEEN DocumentMode.RGB
WARNING: asked for an OpenColorIO document but it did not open at 32
         bits/channel — the OCIO leg did not engage.
```

A 32-bit RGB document is a hard requirement for OCIO ("OpenColorIO disabled due
to document bit depth"). Opening at 16-bit means the request was ignored.

Diagnostics confirm it is not a scripting mistake:

* Querying the application's `ocioConfiguration` property returns
  *"General Photoshop error occurred. This functionality may not be available
  in this version of Photoshop. — The command 'Get' is not currently
  available."*
* The active document's full Action Manager key list contains **no OCIO keys at
  all**: `mode, depth, profile, manage, …` and nothing OCIO-related.
* No OCIO or native-canvas flag has ever been written to
  `…\Adobe Photoshop (Beta) Settings\Adobe Photoshop (Beta) Prefs.psp`.

The binary carries the error string that explains it:

> `$$$/OCIO/Error/MantaRequired=OpenColorIO requires that the native canvas is enabled.`

**So the OCIO surface is gated behind the native-canvas ("Manta") technology
preview, which has never been switched on in this install.** That is a
one-time UI action, not an engineering problem.

### To unblock rung B (user action, once)

**Corrected 2026-08-21 after a forum/documentation review** — the first version
of this section sent the reader to Technology Previews, which is where the
toggle used to live. Adobe moved it.

1. **Edit ▸ OpenColorIO Settings…** → tick **"Enable OpenColorIO Features"**.
   This is the real switch. Adobe employee *cody_cuellar*, 18 Sep 2024: *"It's
   been moved to 'Edit > OpenColorIO Settings'. From that dialog you can
   enable/disable the OCIO features."* Adobe's shipping help page
   ([opencolorio-transform](https://helpx.adobe.com/photoshop/using/opencolorio-transform.html))
   says the same. Technology Previews (as *"OpenColorIO Color Management"*) was
   the original location and stopped being current around build 25.13.0.
2. **Enable the GPU** in Preferences ▸ Performance. Adobe's announcement thread
   says the GPU *"should also be enabled"*. The binary's
   `OpenColorIO requires that the native canvas is enabled` is consistent with
   this — native canvas is the OS-native GPU render path (DirectX/Metal) and
   cannot run with the GPU off — but **no source explicitly names native canvas
   as an OCIO prerequisite**, so treat that link as unconfirmed inference.
3. **Consider updating the Beta.** This install is **27.4, built 2026-01-28**,
   roughly seven months old; 27.8 shipped June 2026 and 27.9.1 in August 2026.
   27.8 is also the release that moved Photoshop to **OCIO 2.5 with bundled
   ACES 2.0 configs** (CY2026 VFX Reference Platform). A single unconfirmed
   user report (KrisRivel, 2 Dec 2025) claims *"Open as OpenColorIO"* worked on
   v26.x and misbehaves on v27.x with identical steps — the closest thing found
   to our 16-bit-instead-of-32-bit symptom, but not established as the same
   issue.
4. Re-run `python tools/photoshop_bridge.py --probe`, then the `--open` leg;
   the document must report **32 bits/channel**.

OCIO is **not** build-, region-, platform- or account-gated: it shipped in
Photoshop 26.0 (October 2024) as a documented feature. It is simply off by
default.

### Config discovery, confirmed

Photoshop resolves an OCIO config from three sources, in priority order:

1. the **`OCIO` environment variable** (our probe already proved this reaches
   Photoshop's process),
2. a **built-in ACES** configuration,
3. `%APPDATA%\Adobe\Adobe Photoshop (Beta)\Presets\OCIO\Configurations`.

So the per-process `$OCIO` scoping in `atlas_camera/paint/ocio.py` is the
documented mechanism, not a trick.

## Two findings that constrain the architecture

**Generative Fill does not work in 32-bit documents.** Adobe documents 32-bit
alongside CMYK and Lab as unsupported modes for generative AI. A VFX user hit
this doing exactly our job — *"I'd like to take a frame into photoshop and use
some generative fill to create a clean plate, but Photoshop doesn't seem to
like 32 bpc"* (Justin Sarceno, 30 Oct 2024, ACEScg EXR clean plate); the
Community Expert reply pointed at OCIO Settings and did not dispute the limit.
**No verified workaround survived review** — the widely-repeated "duplicate the
document to 16-bit and drag the result back" recipe was refuted.

This is the constraint the 16-bit-Smart-Object-inside-a-32-bit-OCIO-container
architecture is betting against. Note the refuted recipe is *not* the same
mechanism as a placed Smart Object carrying `placedLayerOCIOSpace`, so the
architecture is not proven impossible — but it is now an open empirical
question with the burden of proof against it, and it is directly testable the
moment OCIO is enabled. Test it before building on it.

**OCIO has no documented UXP scripting surface.** Neither `OpenColorIO`,
`OCIO`, nor `ocioConfiguration` appears anywhere in Adobe's Photoshop UXP
changelog at any version through 27.4, and a 2024 developer request for exactly
that ("a way to find out if a document is of type OpenColorIO and to
access/change the corresponding color settings") has no Adobe reply. That
matches our observed `ocioConfiguration` error precisely.

Caveat worth keeping: this evidence is about **UXP**, and our bridge drives
**Action Manager via ExtendScript**, which is a different surface — the
`openAsOCIO` / `convertToOCIO` / `adjustmentLayerOCIOAdjustment` events were
read straight out of the shipped action dictionary, so they exist. Whether they
are *drivable* is untested and cannot be tested until OCIO is switched on.

## What the binary establishes (verified strings, not docs)

**OpenColorIO v2.5** with the full ACES 1.3 + 2.0 CLF transform library
embedded. Menu surface: `Open as OpenColorIO`, `Convert to OpenColorIO`,
`New OpenColorIO`, `OpenColorIO Settings`, an OCIO palette, and an OCIO
**adjustment layer** (ColorSpace / DisplayTransform / NamedTransform / CDL).

Config domains: `Builtin`, `Custom`, **`Environment`**, `Preset` — the
Environment domain is what reads `$OCIO`.

**Scriptable Action Manager events**, all present in the action dictionary:

| event | keys |
|---|---|
| `syntheticFill` ("Generative Fill") | `prompt` (string), `workflowType` (`genWorkflow`), `referenceImage` |
| `syntheticFillMode` (enum) | **`inpaint` / `variation` / `synthesize`** |
| `open` | boolean key `openAsOCIO` |
| `convertToOCIO` | `configuration` (`ocioConfiguration`), `working_space` |
| `adjustmentLayerOCIOAdjustment` | `transformType`, `colorSpace` |
| placed layers | `placedLayerOCIOConversion`, `placedLayerOCIOSpace` |

Also present: `syntheticGenHarmonize`, `syntheticGenerateBackground`,
`syntheticGenerateSimilar`, `enhanceGeneratedVariation`.

Generative models named in embedded tool schemas: Firefly 3, Firefly 4,
**Flux Kontext**, **Nano Banana**, **Nano Banana 2**.

**The central architectural constraint:** OCIO needs a 32-bit RGB document with
the native canvas enabled; Generative Fill needs *"RGB, Lab, or CMYK in either
8 or 16 bit"*. They cannot coexist in one document — hence the 16-bit Smart
Object inside a 32-bit OCIO container.

## Why this is worth finishing

Affinity's `generativeEditImage` is image-to-image **regeneration, not
inpainting** — no confined-inpaint call exists in its SDK, so a selection is
only a hint. Measured: containment **0.3740** on the boiler plate, and the
street plate came back as a *different building*.

Photoshop's `syntheticFill` has an explicit **`inpaint`** mode. Whether it is
genuinely bounded is the one question this bridge exists to answer, and it is
recorded as **unmeasured** in `atlas_camera/paint/vendors.py`
(`needs_roi_crop=None`), which makes the tools refuse to assume either way
until a containment measurement settles it.

## Open unknowns (cannot be retired without rung B)

* Whether `DoJavaScript` returns before an async, network-backed `syntheticFill`
  completes. The sentinel-file contract is built so the answer does not matter,
  but it is untested against a real fill.
* Whether `syntheticFill` requires a signed-in Firefly entitlement. A rung-A
  failure could be an account problem rather than a technical one, and the
  client reports timeouts distinctly from script errors so the two are not
  conflated.
* Whether model choice (Flux Kontext / Nano Banana 2) is reachable as an Action
  Manager key, or only through the internal agent tool schema.
* ProEXR export defaults — half or DWA would silently blow the M3 null
  round-trip gate and look like a colour bug.

## Bugs found and fixed on the way here

* **`np.roll` dilation wrapped around the frame.** The mask fallback was a
  separable BOX dilation built from `np.roll`, so a mask touching the left edge
  grew onto the right edge, and it produced a square rather than a disc (56
  extra corner pixels at r=6 on a single point). The authorised region
  therefore depended on whether SciPy happened to be installed. Replaced with
  an exact, non-wrapping disc decomposition; a parity test now compares both
  paths pixel-for-pixel on border-touching masks.
* **OIIO silently dropped the colourspace tag on every EXR written under a
  studio config.** OIIO only persists `oiio:ColorSpace` when the active config
  can supply a `colorInteropID` for the space. The built-in config can (it
  writes `colorInteropID = lin_ap1_scene` alongside the tag); fn-nuke_cg
  v1.0.0 cannot, and OIIO then wrote **no colourspace attribute at all**.

  The tail is what makes it serious. An untagged plate read on `auto` falls
  back to guessing from the extension, and `.exr` guesses ACEScg — so a RAW
  sidecar, which is Rec.709-linear, would come back read as ACEScg and the
  colour would be quietly wrong. That is precisely the failure
  `docs/USER_GUIDE.md` warns about, arriving by a route nobody was watching.
  It also silently disabled the paint bridges' re-tag defence, which is the
  only thing protecting the graph from a vendor's mislabelled export.

  Fixed by writing an `atlas:ColorSpace` attribute unconditionally and reading
  it as a fallback, so a plate self-describes under any config while a file
  written by another application still wins on the standard tag. The same
  drop broke `atlas/world/plate_artifacts.py` and
  `atlas_camera/exporters/real_plate_nuke.py`, whose own ACEScg validators
  then rejected the artifacts they had just written; both now read the
  fallback. (`real_plate_nuke` also allowed `""` for "absent", but `dict.get`
  returns `None`, so the allowance never matched.)

* **Two colourspace names broke under a studio config.** `lin_rec709_scene`
  (the tag on every confined clean plate) and `sRGB - Display` (which *is*
  `COMFY_WORKING_COLORSPACE`) both failed to resolve under fn-nuke_cg — the
  first because `list_colorspaces()` enumerates canonical names and not
  aliases, the second because OIIO does not count display colourspaces. Both
  were lookup bugs, not capability gaps: the config converts 0.18 ACEScg to
  0.46135 correctly, matching the recorded sanity value.
* **`configname` is a method, not a property** — read as an attribute it put a
  bound-method repr into the config identity where the config's name belongs.
* **ExtendScript's `app.path` is a Folder URI**
  (`/c/Program%20Files/…`), not an OS path. The install guard rejected the very
  install it had been asked for on the first live run. The probe now reports
  `.fsName`, and the comparison accepts both spellings.
