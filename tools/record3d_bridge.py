"""iPhone <-> ComfyUI bridge for Record3D captures, over Tailscale.

A round trip on your own tailnet: scan a room on the phone, share it to this
machine, and the rendered result comes back to the phone's camera roll. No cloud
account, no sync client, no third party holding the file — which matters more
than usual here, because `.r3d` captures run to hundreds of megabytes.

    python tools/record3d_bridge.py                 # receive, render, send back
    python tools/record3d_bridge.py --no-return     # receive and render only

On the phone: Record3D -> your capture -> Share -> Tailscale -> this machine.

WHY TAILDROP RATHER THAN A SYNCED FOLDER
Taildrop moves the file directly between two devices that already trust each
other, and only publishes it once it has fully arrived — so the classic folder-
watcher failure, grabbing a half-synced capture and choking on a truncated ZIP,
mostly cannot happen. A cloud folder additionally means a copy of every scan
sits on someone else's disk. `--watch <folder>` remains for anyone who wants the
old behaviour; the completeness check below runs either way, because "mostly
cannot happen" is not "cannot happen".

THE RETURN LEG
The render is sent back with `tailscale file cp`, so results land on the phone
without you walking back to the desk. Note the phone must accept the incoming
file — iOS does not auto-save Taildrop transfers.

Alternatively, view everything live: ComfyUI is reachable from the phone at
this machine's own tailnet address. Run `--url` to print it — it is read from
`tailscale status`, never hard-coded, because a tailnet name published in a
public repo is a hostname an attacker has in advance if Funnel is ever enabled.

Device captures are LZFSE-compressed and need the decoder:
    pip install pyliblzfse        (into ComfyUI's venv)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path

HOST = "127.0.0.1:8188"

#: Standard Tailscale install location on Windows. NOT machine-specific — a
#: `shutil.which` fallback covers macOS/Linux and any custom install.
TAILSCALE_EXE = Path(r"C:\Program Files\Tailscale\tailscale.exe")

#: Directories under which a ComfyUI install is commonly unpacked. Searched with
#: a `ComfyUI*` glob rather than a fixed list, because real installs carry a
#: version or distribution suffix — "ComfyUI_V91", "ComfyUI_windows_portable" —
#: and a fixed list finds none of them. Each hit is checked BOTH directly and one
#: level down, since the portable builds nest the app inside the download folder.
COMFYUI_SEARCH_PARENTS = (".", "Desktop", "Documents", "Downloads", "comfy")

#: Where received captures land. Deliberately NOT inside any synced directory —
#: the point of Taildrop here is that nothing leaves the tailnet, and quietly
#: dropping captures into a cloud folder would undo exactly that.
INBOX_DIR = Path.home() / "AtlasCaptures"


# --------------------------------------------------------------- comfyui paths


def comfyui_root(explicit: str | None = None) -> Path | None:
    """Locate a ComfyUI install: explicit wins, then COMFYUI_ROOT, then a guess.

    Returns None rather than raising — the caller decides whether a missing
    install is fatal, and ``--no-render`` does not need one at all.

    Discovery is last, and deliberately weak: guessing at someone's layout and
    being confidently wrong is worse than saying so, because the failure would
    surface much later as "capture delivered" followed by nothing happening.
    """
    for candidate in (explicit, os.environ.get("COMFYUI_ROOT")):
        if candidate and (root := Path(candidate).expanduser()).is_dir():
            return root
    # Running under ComfyUI's own interpreter? Then it can just tell us.
    try:
        import folder_paths  # type: ignore[import-not-found]
        return Path(folder_paths.base_path)
    except Exception:  # noqa: BLE001
        pass
    found = list(_search_for_comfyui())
    if len(found) > 1:
        # Do not choose between two installs in silence. Rendering into the
        # wrong ComfyUI looks exactly like rendering into the right one right
        # up until you go looking for the output.
        print(f"    NOTE: {len(found)} ComfyUI installs found; using {found[0]}")
        for other in found[1:]:
            print(f"          (also: {other} — pass --comfyui-root to choose)")
    return found[0] if found else None


def _looks_like_comfyui(path: Path) -> bool:
    return (path / "input").is_dir() and (path / "output").is_dir()


def _search_for_comfyui(home: Path | None = None):
    """Yield plausible ComfyUI roots under the usual parents, best-effort.

    Sorted so the result does not depend on filesystem enumeration order — two
    machines with the same layout should resolve the same install, and a bridge
    that silently picks a different ComfyUI between runs is a bad way to spend
    an afternoon.
    """
    home = home or Path.home()
    for parent_rel in COMFYUI_SEARCH_PARENTS:
        parent = home / parent_rel
        try:
            candidates = sorted(parent.glob("ComfyUI*"))
        except OSError:
            continue
        for cand in candidates:
            if not cand.is_dir():
                continue
            if _looks_like_comfyui(cand):
                yield cand
            # Portable/versioned builds nest the app one level down.
            elif _looks_like_comfyui(cand / "ComfyUI"):
                yield cand / "ComfyUI"


def resolve_io_dirs(input_dir: str | None, output_dir: str | None,
                    root: str | None = None) -> tuple:
    """(input, output) for this machine, or a message saying what to pass."""
    found = comfyui_root(root)
    inp = Path(input_dir).expanduser() if input_dir else (
        found / "input" if found else None)
    out = Path(output_dir).expanduser() if output_dir else (
        found / "output" if found else None)
    missing = [n for n, p in (("--input-dir", inp), ("--output-dir", out))
               if p is None or not p.is_dir()]
    if missing:
        return None, (
            f"Could not locate ComfyUI's {' and '.join(missing)}.\n"
            "Point at your install with one of:\n"
            "    set COMFYUI_ROOT=<path to your ComfyUI checkout>\n"
            "    python tools/record3d_bridge.py --comfyui-root <path>\n"
            "    python tools/record3d_bridge.py --input-dir <path> "
            "--output-dir <path>")
    return (inp, out), None


# --------------------------------------------------------------- tailscale


def tailscale() -> Path:
    if TAILSCALE_EXE.exists():
        return TAILSCALE_EXE
    found = shutil.which("tailscale")
    if found:
        return Path(found)
    raise SystemExit(
        "tailscale not found. Install Tailscale and sign in, or fall back to a\n"
        "synced folder with:  --watch \"D:\\some\\folder\"")


def _ts(exe: Path, *args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run([str(exe), *args], capture_output=True, text=True,
                          timeout=timeout)


def tailnet_self(exe: Path) -> dict:
    """This node's own tailnet identity, for printing a URL that actually works."""
    try:
        st = json.loads(_ts(exe, "status", "--json", timeout=30).stdout)
    except Exception:  # noqa: BLE001
        return {}
    return st.get("Self") or {}


def find_phone(exe: Path, explicit: str | None = None) -> str | None:
    """Pick the return target: an ONLINE iOS peer, unless one is named.

    Matching on OS rather than hostname on purpose — this phone reports its
    hostname as "localhost", which would make a name-based guess look sensible
    and then silently target nothing.
    """
    if explicit:
        return explicit
    try:
        st = json.loads(_ts(exe, "status", "--json", timeout=30).stdout)
    except Exception:  # noqa: BLE001
        return None
    for peer in (st.get("Peer") or {}).values():
        if peer.get("OS") == "iOS" and peer.get("Online"):
            dns = (peer.get("DNSName") or "").rstrip(".")
            return dns or (peer.get("TailscaleIPs") or [None])[0]
    return None


def drain_inbox(dest: Path, exe: Path) -> list[Path]:
    """Move anything waiting in the Taildrop inbox into `dest`; return new .r3d.

    `--conflict=rename` rather than the default `skip`: re-sending a capture
    under the same name is a normal thing to do (a second scan of the same room),
    and having it silently not arrive would be baffling to debug.
    """
    before = {p.name for p in dest.glob("*.r3d")}
    try:
        out = _ts(exe, "file", "get", "--conflict=rename", str(dest))
    except Exception as exc:  # noqa: BLE001
        print(f"    inbox check failed: {type(exc).__name__}: {exc}")
        return []
    err = (out.stderr or "").strip()
    # An empty inbox is the normal case and not worth printing every poll.
    if err and not any(q in err.lower() for q in ("no files", "empty")):
        print(f"    taildrop: {err[:200]}")
    return sorted(p for p in dest.glob("*.r3d") if p.name not in before)


def send_to_phone(paths: list[Path], target: str, exe: Path) -> None:
    """Return renders to the phone. Best-effort: a failed send must not lose work."""
    if not paths:
        return
    try:
        out = _ts(exe, "file", "cp", *[str(p) for p in paths], f"{target}:")
    except Exception as exc:  # noqa: BLE001
        print(f"    send failed: {type(exc).__name__}: {exc}")
        return
    if out.returncode == 0:
        print(f"    sent {len(paths)} file(s) -> {target}  (accept on the phone)")
    else:
        print(f"    send failed: {(out.stderr or out.stdout).strip()[:200]}")


# --------------------------------------------------------------- classifying

#: Everything the router will pick up. `.r3d` covers BOTH a Record3D room scan
#: and a single framed still packaged as a one-frame capture — they are the same
#: container and the importer does not care which it is given.
CAPTURE_EXTS = {".r3d"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".exr", ".tif", ".tiff", ".heic"}
ACCEPTED_EXTS = CAPTURE_EXTS | IMAGE_EXTS

#: An equirectangular panorama is 2:1 by definition. Real files miss slightly
#: (an 8K plate cropped by a pixel), so allow a little slack — but keep it TIGHT.
#: Widening this is how an ordinary wide photo starts getting solved as a 360,
#: which produces a confusing picture rather than an error.
PANO_ASPECT_LO, PANO_ASPECT_HI = 1.9, 2.1


def image_size(path: Path) -> tuple[int, int] | None:
    """(width, height), or None if it cannot be read as an image.

    Pillow first (cheap, covers jpg/png), then OpenImageIO — because **stock
    Pillow cannot read EXR**, and EXR is precisely the format the 8K panoramas
    arrive in. Without the fallback every equirect plate classified as `unknown`
    and was silently skipped, which is how a working panorama route would have
    shipped doing nothing at all.

    Both paths read the HEADER only. That is deliberate: decoding a 200 MB 8K
    EXR just to learn its aspect ratio would stall the poll loop on every pass.
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:  # noqa: BLE001 - unsupported format, or a broken file
        pass
    try:
        import OpenImageIO as oiio  # type: ignore

        src = oiio.ImageInput.open(str(path))
        if src is None:
            # Drain the error OIIO queued. Left unretrieved it prints
            # "exited with a pending error message that was never retrieved"
            # to stderr at interpreter shutdown — alarming output for the
            # entirely normal case of probing a file that is not an image.
            try:
                oiio.geterror()
            except Exception:  # noqa: BLE001
                pass
            return None
        try:
            spec = src.spec()
            return int(spec.width), int(spec.height)
        finally:
            src.close()
    except Exception:  # noqa: BLE001
        return None


def classify(path: Path) -> tuple[str, str]:
    """Route a file to a graph. Returns (kind, human-readable reason).

    Dimensions are READ, never inferred from the filename — a file called
    `pano.jpg` that is 16:9 is not a panorama, and treating it as one would tear
    the geometry in a way that looks like a bug in the solver.
    """
    ext = path.suffix.lower()
    if ext in CAPTURE_EXTS:
        return "record3d", "Record3D capture (scan or single-frame still)"
    if ext not in IMAGE_EXTS:
        return "unknown", f"unrecognised extension {ext!r}"

    size = image_size(path)
    if size is None:
        # Deliberately NOT defaulting to the still route: an unreadable image is
        # a fact worth surfacing, and guessing here would bury it.
        return "unknown", "could not read image dimensions"
    w, h = size
    if h <= 0:
        return "unknown", f"degenerate size {w}x{h}"
    aspect = w / h
    if PANO_ASPECT_LO <= aspect <= PANO_ASPECT_HI:
        return "panorama", f"{w}x{h}, aspect {aspect:.3f} — equirectangular"
    return "still", f"{w}x{h}, aspect {aspect:.3f}"


# ------------------------------------------------------------ capture gate


def is_complete(path: Path, settle: float) -> bool:
    """True once the file has stopped growing AND parses as a Record3D bundle.

    Two gates on purpose. Size-stability alone still lets a paused transfer
    through; the ZIP check proves the bytes are actually a capture rather than a
    placeholder or a partial write.

    ONLY meaningful for `.r3d`. An image is a single write with no container to
    validate, and running the ZIP check against one would reject every photo.
    """
    try:
        first = path.stat().st_size
    except OSError:
        return False
    if first == 0:
        return False
    time.sleep(settle)
    try:
        if path.stat().st_size != first:
            return False
    except OSError:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            return "metadata" in names and any(n.startswith("rgbd/") for n in names)
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------- comfyui


# EVERY graph below ends in AtlasStereoRender, never AtlasBlockoutViewport.
# The viewport renders in the BROWSER via three.js, so queuing it headless
# returns black frames — it looks like a broken solve and is not one.
# AtlasStereoRender rasterises server-side and actually returns pixels.
_STEREO = {"interocular_m": 0.30, "convergence_m": 5.0,
           "output_mode": "sbs", "resolution": 1024}


def graph_record3d(path: Path) -> dict:
    """Measured ARKit geometry — a room scan OR a single framed still.

    Both are the same container, and `AtlasLoadRecord3D` reads either without
    caring: `Record3DCapture.open()` needs only `w`/`h` and a pose, so a
    one-frame capture from a phone app works here with no special case.
    """
    return {
        "1": {"class_type": "AtlasLoadRecord3D",
              "inputs": {"capture_path": str(path), "frame_index": 0,
                         "depth_resolution": "colour_frame",
                         "min_confidence": "medium"}},
        "2": {"class_type": "AtlasDeriveReliefMesh",
              "inputs": {"solve": ["1", 1], "depth": ["1", 2]}},
        "3": {"class_type": "AtlasStereoRender",
              "inputs": {"solve": ["2", 0], "source_image": ["1", 0], **_STEREO}},
        "4": {"class_type": "SaveImage",
              "inputs": {"images": ["3", 0], "filename_prefix": f"r3d_{path.stem}"}},
        "5": {"class_type": "SaveImage",
              "inputs": {"images": ["1", 0],
                         "filename_prefix": f"r3d_{path.stem}_plate"}},
    }


def graph_panorama(path: Path) -> dict:
    """Equirectangular 360 -> multi-camera solve.

    `AtlasEquirectMultiView`'s own defaults are deliberately NOT overridden here.
    They are measured, not guessed: 4 views (2->4 lifts safe z dolly 4.2x, then
    plateaus), pitch 0 (tilting removes the horizon and the ground fit stops
    working entirely), and MoGe (V2-Metric-Outdoor mis-scales 90-degree panorama
    crops ~4.4x). Re-stating them here would let this file drift from the node.
    """
    return {
        "1": {"class_type": "AtlasLoadPlate", "inputs": {"file_path": str(path)}},
        "2": {"class_type": "AtlasEquirectMultiView", "inputs": {"equirect": ["1", 0]}},
        "3": {"class_type": "AtlasStereoRender",
              "inputs": {"solve": ["2", 0], "source_image": ["2", 1], **_STEREO}},
        "4": {"class_type": "SaveImage",
              "inputs": {"images": ["3", 0], "filename_prefix": f"pano_{path.stem}"}},
        "5": {"class_type": "SaveImage",
              "inputs": {"images": ["2", 1], "filename_prefix": f"pano_{path.stem}_views"}},
    }


def graph_still(path: Path) -> dict:
    """An ordinary photograph -> learned solve + relief geometry.

    `layers=0` with `mesh="relief"` is the SINGLE full-range relief mesh — the
    fast path, and the right trade for a phone round trip. The alternative
    (`layers=2/3/4`) builds depth-band clean-plate layers with inpainting, which
    is a much longer job for a result someone is waiting on with a phone in hand.
    Note `mesh` is a COMBO of mode names, not a boolean.
    """
    return {
        "1": {"class_type": "AtlasLoadPlate", "inputs": {"file_path": str(path)}},
        "2": {"class_type": "AtlasInput",
              "inputs": {"image": ["1", 0], "layers": 0, "mesh": "relief"}},
        "3": {"class_type": "AtlasStereoRender",
              "inputs": {"solve": ["2", 0], "source_image": ["2", 1], **_STEREO}},
        "4": {"class_type": "SaveImage",
              "inputs": {"images": ["3", 0], "filename_prefix": f"still_{path.stem}"}},
        "5": {"class_type": "SaveImage",
              "inputs": {"images": ["2", 1], "filename_prefix": f"still_{path.stem}_plate"}},
    }


GRAPH_BUILDERS = {
    "record3d": graph_record3d,
    "panorama": graph_panorama,
    "still": graph_still,
}


def queue_render(path: Path, host: str, kind: str) -> str | None:
    """Queue the graph matching `kind`."""
    builder = GRAPH_BUILDERS.get(kind)
    if builder is None:
        print(f"    no graph for kind {kind!r} — skipped")
        return None
    graph = builder(path)
    req = urllib.request.Request(
        f"http://{host}/prompt",
        data=json.dumps({"prompt": graph, "client_id": str(uuid.uuid4())}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt_id"]
    except Exception as exc:  # noqa: BLE001
        print(f"    queue failed: {type(exc).__name__}: {str(exc)[:160]}")
        return None


def wait_for(prompt_id: str, host: str, output_dir: Path,
             timeout: int = 900) -> list[Path]:
    """Block until the render finishes; return the files it produced."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            hist = json.loads(urllib.request.urlopen(
                f"http://{host}/history/{prompt_id}", timeout=30).read())
        except Exception:  # noqa: BLE001
            print("    server unreachable while waiting")
            return []
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            produced: list[Path] = []
            if status.get("status_str") == "success":
                for out in entry.get("outputs", {}).values():
                    for im in out.get("images", []):
                        print(f"    rendered: {im['filename']}")
                        p = output_dir / im.get("subfolder", "") / im["filename"]
                        if p.exists():
                            produced.append(p)
            else:
                for m in status.get("messages", []):
                    if m[0] == "execution_error":
                        print(f"    FAILED {m[1].get('node_type')}: "
                              f"{str(m[1].get('exception_message'))[:200]}")
            return produced
        time.sleep(3)
    print("    timed out waiting for the render")
    return []


# -------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", default=None,
                    help="use a synced folder instead of Taildrop")
    ap.add_argument("--inbox", default=str(INBOX_DIR),
                    help="where received captures land")
    ap.add_argument("--comfyui-root", default=None,
                    help="ComfyUI install (default: $COMFYUI_ROOT, then a guess)")
    ap.add_argument("--input-dir", default=None,
                    help="override ComfyUI's input directory")
    ap.add_argument("--output-dir", default=None,
                    help="override ComfyUI's output directory")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--peer", default=None,
                    help="return target (default: the online iOS peer)")
    ap.add_argument("--no-render", action="store_true", help="deliver only")
    ap.add_argument("--no-return", action="store_true",
                    help="do not send renders back to the phone")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="seconds a file must stop growing before it is accepted")
    ap.add_argument("--poll", type=float, default=5.0, help="seconds between checks")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--url", action="store_true",
                    help="print the tailnet URL for ComfyUI and exit")
    args = ap.parse_args()

    use_taildrop = args.watch is None
    exe = tailscale() if use_taildrop or args.url else None

    if args.url:
        me = tailnet_self(exe)
        name = (me.get("DNSName") or "").rstrip(".")
        ip = (me.get("TailscaleIPs") or [""])[0]
        port = args.host.rsplit(":", 1)[-1]
        print(f"ComfyUI on the tailnet:\n    http://{name}:{port}\n    http://{ip}:{port}")
        return

    watch = Path(args.watch) if args.watch else Path(args.inbox)
    dirs, problem = resolve_io_dirs(args.input_dir, args.output_dir,
                                    args.comfyui_root)
    if problem:
        raise SystemExit(problem)
    inp, outp = dirs
    watch.mkdir(parents=True, exist_ok=True)
    inp.mkdir(parents=True, exist_ok=True)

    peer = None
    if use_taildrop and not args.no_return:
        peer = find_phone(exe, args.peer)

    print(f"transport : {'Tailscale Taildrop' if use_taildrop else f'folder {watch}'}")
    if use_taildrop:
        me = tailnet_self(exe)
        print(f"this node : {(me.get('DNSName') or '?').rstrip('.')}")
        print(f"landing   : {watch}")
    print(f"delivering: {inp}")
    print(f"render    : {'no' if args.no_render else 'yes'}")
    print(f"return to : {peer or 'nobody (renders stay here)'}")
    print("\nOn the phone: Record3D -> capture -> Share -> Tailscale -> this machine."
          "\nCtrl-C to stop.\n" if use_taildrop else "\nCtrl-C to stop.\n")

    # Seed from what has already been DELIVERED, not from what is sitting in the
    # landing folder. Seeding from the folder looks equivalent and is not: a
    # capture sent while this script was down would be ignored forever, which is
    # precisely the case it exists to serve.
    seen: set[str] = {p.name for p in inp.iterdir()
                      if p.suffix.lower() in ACCEPTED_EXTS} if inp.is_dir() else set()
    if seen:
        print(f"{len(seen)} file(s) already delivered — skipping those\n")

    while True:
        if use_taildrop:
            arrived = drain_inbox(watch, exe)
            for p in arrived:
                print(f"[{time.strftime('%H:%M:%S')}] taildrop received: {p.name}")

        incoming = sorted(p for p in watch.iterdir()
                          if p.is_file() and p.suffix.lower() in ACCEPTED_EXTS)
        for src in incoming:
            if src.name in seen:
                continue
            size_mb = src.stat().st_size / 1024 / 1024
            kind, why = classify(src)
            print(f"[{time.strftime('%H:%M:%S')}] new {kind}: {src.name} "
                  f"({size_mb:.1f} MB) — {why}")
            if kind == "unknown":
                seen.add(src.name)          # do not re-report it every poll
                print("    skipped — nothing to route it to")
                continue
            # The completeness gate is a Record3D-container check, so it only
            # applies to captures. Images are a single write, and running the
            # ZIP check against one would reject every photo.
            if kind == "record3d" and not is_complete(src, args.settle):
                print("    still arriving — will retry")
                continue           # deliberately NOT marked seen; retry next pass
            seen.add(src.name)
            dst = inp / src.name
            shutil.copy2(src, dst)
            print(f"    delivered -> {dst}")

            if args.no_render:
                continue
            pid = queue_render(dst, args.host, kind)
            if not pid:
                continue
            produced = wait_for(pid, args.host, outp)
            if peer and produced:
                send_to_phone(produced, peer, exe)

        if args.once:
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
