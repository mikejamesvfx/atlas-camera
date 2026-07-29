# Bootstrapping the Atlas iPhone app on the Mac

Getting the code onto the MacBook and building the first milestone. Written on
the Windows box, meant to be read on the Mac.

---

## Two decisions made up front

**The app gets its OWN repository.** Not a folder inside Atlas. Atlas publishes
to the ComfyUI registry, so an Xcode project living in it is dead weight in every
published package, needs `.comfyignore` entries, and drags a second language and
toolchain into a Python repo's CI. The app will also iterate far faster than the
nodes.

The coupling between them is not code — it is
[`SHOOT_PROJECT_FORMAT.md`](SHOOT_PROJECT_FORMAT.md), a versioned JSON contract.
Two repos, one format, pinned on both sides.

**Do not build over a network share.** Xcode on SMB is slow, its file-watching
misbehaves, `DerivedData` thrashes, and git over a share can corrupt index and
lock files when two machines touch one working tree. Clone properly.

---

## 1. Get Atlas onto the Mac (once)

`.git` is ~35 MB and bundles to ~11 MB, so it travels fine over Taildrop — no
SSH server, no GitHub round trip, nothing leaving your devices.

**Commit first.** A bundle carries COMMITS, not your working tree: anything
uncommitted simply is not in it, and you find that out on the Mac when the file
you came for is missing. Verified the hard way while writing this.

```bash
git status --porcelain     # must be clean for anything you need on the Mac
```

On Windows:

```bash
git bundle create ../atlas.bundle --all
```

Written OUTSIDE the repo on purpose — a bundle is a build artifact, and
an 11 MB binary committed by accident is painful to remove from history.
`*.bundle` is gitignored as a second line of defence.

Send it with the Tailscale share sheet, or:

```bash
tailscale file cp atlas.bundle <mac-hostname>:
```

`tailscale status` lists your machines. A bare MagicDNS short name works
(`my-macbook:`); the full `<host>.<tailnet>.ts.net` form is only needed if the
short name is ambiguous. Deliberately not hard-coded here — see the note at the
end of this section.

On the Mac:

```bash
tailscale file get ~/Downloads
git clone ~/Downloads/atlas.bundle atlas-camera
```

You now have full history. To refresh later, make a new bundle — or set up a
real remote (below) once you are iterating.

### Ongoing sync, if you want it

A bundle is fine for reference material you read occasionally. If you end up
editing Atlas from the Mac, either use a private GitHub remote, or install
**OpenSSH Server** on Windows (an optional feature, not currently installed) and
clone over the tailnet:

```bash
git clone ssh://<user>@<windows-hostname>/C:/path/to/AtlasCamera_Claude
```

Only reachable from your own tailnet either way.

**Why the placeholders.** This repo is public. Tailscale's `100.64.0.0/10`
addresses are CGNAT and unreachable without tailnet membership, so those are
harmless — but a tailnet ID, machine names, and a local user path are none of
them secret and all of them useful to someone targeting you. The one that
actually bites: enable `tailscale funnel` or `serve` on a machine and its
`<host>.<tailnet>.ts.net` name becomes publicly resolvable and reachable from
the internet. A hostname published in advance turns that config change into a
URL an attacker already has. Fill these in locally; don't commit them.

---

## 2. Start the app repo

```bash
mkdir atlas-ios && cd atlas-ios && git init
```

Copy in, from the Atlas clone:

- `docs/SHOOT_PROJECT_FORMAT.md` — the contract
- `docs/shoot_project.example.json` — **the fixture the app develops against**
- `docs/ios_app_CLAUDE.md` → save as **`CLAUDE.md` in the repo root**

```bash
cp ../atlas-camera/docs/SHOOT_PROJECT_FORMAT.md .
cp ../atlas-camera/docs/shoot_project.example.json .
cp ../atlas-camera/docs/ios_app_CLAUDE.md ./CLAUDE.md
```

That third file matters most for a session on the Mac: it carries the two
silently-failing conventions, the `volumetric` branch, and the build order. A
Claude session there has none of the context this was designed in, and without
it will invent a plausible-looking format.

The fixture matters more than it looks: with it, the app can be built and tested
with no Windows box, no ComfyUI, and no network. Do not skip it.

If you have them, bring `docs/dev/atlas_iphone_app_spec.md` and
`atlas_iphone_app_concept.md` too — but note `docs/dev/` is **gitignored**, so
they will NOT be in the clone. Copy those by hand or they are lost.

---

## 3. First milestone — the loop, without any AR

Resist starting with the AR overlay. Build the boring round trip first; if it
works the app has a job, and the overlay becomes a well-justified phase 2.

1. **Parse** `atlas_shoot.json`, list the shots sorted by `priority`. Show
   `subject`, `guidance`, and `warnings` verbatim.
2. **Branch on `volumetric`.** `true` means no plane was fitted and
   `incidence_deg` is a placeholder, not an angle — those shots are out of scope
   for phase 1.
3. **Capture** with `AVCapturePhotoOutput`. On a Pro, enable
   `isDepthDataDeliveryEnabled` and `isCameraCalibrationDataDeliveryEnabled` on
   the session configuration BEFORE the photo settings, or depth comes back nil.
   Convert depth with `AVDepthData.converting(toDepthDataType:
   kCVPixelFormatType_DepthFloat32)` — **depth, not disparity**; disparity is
   1/metres and would invert the scene while looking plausible.
4. **Rescale the intrinsics.** `intrinsicMatrixReferenceDimensions` is the
   resolution the matrix was measured at, and it usually differs from the
   delivered photo. Skip this and the principal point lands in the wrong place.
5. **Package** as a one-frame `.r3d` — see the format doc for the two
   conventions that silently produce wrong geometry (`K` column-major, pose
   quaternion scalar-last).
6. **Send back** via the Tailscale share sheet to the Windows machine running
   Atlas.
7. **Confirm** Atlas ingests it through `AtlasAddPatchView`.

A non-Pro device has no usable `AVDepthData`. Degrade to a plain photo and
**say so in the UI** — a capture without measured depth looks identical to one
with it, and that difference is the point of the app.

---

## 4. Phase 2 — the AR overlay

Only for `volumetric: true` shots, where no flat texture can work.

Overlay the **hole's geometry**, not the plate's pixels: off-location the plate
shows a building that is not in front of you, which is confusing and implies a
correspondence that does not exist. Show the plate crop alongside as the
appearance reference instead — alignment and look-matching are two jobs and one
translucent image does neither well.

Alignment does not need to be precise. Capture the ARKit pose and let Atlas
correct the residual; a measured pose beats a matched one. LiDAR also resolves
the depth ambiguity that eyeball alignment has on its own — closer-and-wider
looks identical to further-and-narrower until you can measure it.

---

## 5. Running Claude Code on the Mac

Point it at the `atlas-ios` clone, not a share. Give it
`SHOOT_PROJECT_FORMAT.md` and the example fixture early — most of the useful
context is in those two files, and without them it will invent a format.

Worth stating in the repo's `CLAUDE.md`: the JSON contract is **append-only and
versioned**, clients refuse unknown versions rather than guessing, and
`lighting.measured` is a flag to branch on rather than a field to look for.
