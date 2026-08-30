"""AtlasDirectorTake -- the take a director shot, as graph inputs.

Launching Director happens on a widget button, outside execution, so a
queued prompt never waits on a human (spec 3.4). Execution here READS what
the session already holds and never mutates ``SESSIONS`` or any scene
document. It DOES write two things beside the take, both intended and
described where they happen: `write_ray_exr` creates `<take_dir>/rays/` and
records `manifest.json["rayChannels"]`. No coordinate conversion happens
anywhere in this file -- samples arrive in Atlas canonical space
(right-handed, Y-up, metres, camera down -Z) and are handed to
``plucker.ray_map`` / ``plucker.plucker_embedding`` exactly as they are.

The graph -- not this node -- writes the ``.atlas`` session package.
``AtlasExportScenePackage`` writes it; its ``scene_id`` must equal this
node's ``session_id``, and its ``output_dir`` must be the configured
``ATLAS_DIRECTOR_ROOT``. Nothing here enforces that mapping in code:
``director_session.launch_session`` already refuses to spawn Director when
the package is absent, naming the fix, so a mismatch fails loudly at launch
-- before anyone shoots a take.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from atlas_camera.comfy.director_session import SESSIONS
from atlas_camera.comfy.node_helpers import (
    _require_numpy,
    _require_pil,
    _require_torch,
)
from atlas_camera.comfy.plucker import plucker_embedding, ray_map


class StaleTakeError(RuntimeError):
    """The session package changed since Director was launched onto it,
    or freshness cannot be checked at all because launch never recorded a
    digest to check against.

    Always a hard refusal, never a warning that continues: a stale grey
    guide produces a plausible photoreal result built on the wrong world,
    and a console warning during a batch run is not read.
    """


#: LTX rejects arbitrary lengths, and rejects them AFTER a take is shot. One
#: value until the real set is confirmed against the model (spec open
#: question 7): refusing everything else is the correct failure direction.
ALLOWED_FRAMES = ("121",)

#: Whether `read()` loads the 8-bit playblast PNGs or the float colour lane
#: (spec 3.9). "exr" reads floats as-is -- no divide-by-255, no colour
#: convert; these are numbers a downstream model conditions on, not colour
#: for a human to look at. Whether a take actually HAS an exr lane is a
#: property of the take directory, not of this widget -- see
#: `load_playblast`, which refuses rather than silently substituting.
COLOUR_LANES = ("png", "exr")

#: Channel naming for the ray-map EXR sidecar, recorded into the take
#: manifest. No consumer exists yet, so the manifest is what a future
#: consumer checks its expectation against, instead of re-deriving it from
#: pixel order. Matches `ray_map`'s own channel order (origin, direction) --
#: NOT `plucker_embedding`'s (moment, direction). `write_ray_exr` must only
#: ever be called with a `ray_map`-shaped array, never an embedded one, or
#: this naming lies.
RAY_CHANNEL_NAMES = ["O.X", "O.Y", "O.Z", "D.X", "D.Y", "D.Z"]


class AtlasDirectorTake:
    """Reads a take a director already shot as ComfyUI graph inputs.

    Returns four sockets:

    * ``playblast`` -- the rendered frames as a standard ``IMAGE`` tensor.
    * ``rays`` -- the full-precision 6-channel Plücker embedding
      (``ATLAS_RAYS``, a custom socket type ComfyUI refuses to wire into an
      image node). Directions span [-1, 1] and moments are unbounded (around
      10 for a camera 10 m out); an ``IMAGE`` tensor is float32 clamped to
      [0, 1], so labelling this "IMAGE" would silently quantise it -- one
      careless 8-bit hop costs roughly 0.45 degrees of angular error against
      a measured Plücker fidelity of 0.171 degrees through a lossless codec.
    * ``rays_preview`` -- a DISPLAY-ONLY 3-channel ``IMAGE``: direction
      remapped to ``0.5 * (d + 1.0)``. Never wire this into anything but a
      preview widget; the moment is not represented here at all.
    * ``samples`` -- the raw per-frame camera samples for the rendered
      range, tagged ``ATLAS_CAMERA``.
    """

    RETURN_TYPES = ("IMAGE", "ATLAS_RAYS", "IMAGE", "ATLAS_CAMERA")
    RETURN_NAMES = ("playblast", "rays", "rays_preview", "samples")
    FUNCTION = "read"
    CATEGORY = "Atlas"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_id": ("STRING", {
                    "default": "shot_001",
                    "tooltip": "The session id Director was launched with. "
                               "Launching happens on a widget button outside "
                               "execution -- this node only reads what a "
                               "director already pushed for it.",
                }),
                "width": ("INT", {"default": 768, "min": 16, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 16, "max": 4096}),
                "frames": (ALLOWED_FRAMES, {
                    "tooltip": "LTX rejects arbitrary lengths, and only after "
                               "a take is shot. One value until the real set "
                               "is confirmed against the model (spec open "
                               "question 7); refusing everything else is the "
                               "correct failure direction. Not used to slice "
                               "this node's own outputs -- the rendered "
                               "playblast directory is what decides frame "
                               "count (see `frame_files`) -- it exists so a "
                               "downstream LTX node reads a value that "
                               "matches what was actually launched with.",
                }),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "colour_lane": (COLOUR_LANES, {
                    "tooltip": "png (default): the 8-bit playblast frames. "
                               "exr: the float colour lane, when the take "
                               "actually carries one -- read as data, no "
                               "divide-by-255, no colour convert. Refuses "
                               "loudly on a mismatch rather than silently "
                               "reading the wrong lane.",
                }),
            },
        }

    # -- session --------------------------------------------------------

    def _session(self, session_id: str) -> dict:
        session = SESSIONS.get(session_id)
        if session is None:
            raise ValueError(
                f"ComfyUI does not know session {session_id!r}. Launch "
                "Director, or point this node at a slate directly."
            )
        if not session.get("take_dir"):
            raise ValueError(
                f"no take pushed for session {session_id!r}. Shoot a take "
                "and press Send to Comfy in the deck."
            )
        return session

    def resolve(self, session_id: str) -> str:
        """The take directory a director pushed for `session_id`."""
        return self._session(session_id)["take_dir"]

    # -- freshness (addendum Ruling P8) ----------------------------------

    def package_digest(self, package_path: str) -> str:
        return hashlib.sha256(Path(package_path).read_bytes()).hexdigest()

    def check_fresh(self, package_path: str, recorded_digest: str,
                     slate: str = "") -> bool:
        """Refuse, loudly, when the session package changed since launch.

        Compares the PACKAGE digest `launch_session` recorded when Director
        was opened, never a samples digest: this node has no scene document
        and no channel to compute a real `scene_digest` over. "The package
        changed since Director opened it" is a wider window than "since the
        take was pushed" -- it refuses more, never less, which is the
        correct direction for spec 3.12's posture.
        """
        current = self.package_digest(package_path)
        if current != recorded_digest:
            name = slate or package_path
            raise StaleTakeError(
                f"the playblast for {name} no longer matches the session "
                "package it was launched from. Re-push the take from the "
                "deck to re-render it."
            )
        return True

    def _ensure_fresh(self, session_id: str, session: dict) -> None:
        """Refuse, never silently skip, when there is nothing to check freshness against.

        `package` / `package_digest` being absent is not "no session to
        worry about" -- a take is already pushed (`_session` guaranteed
        that) -- it is "this session was never launched through the path
        that records a digest to check". Skipping the check in that case is
        warn-and-continue wearing a guard's clothes; refuse instead, and
        name what is missing.
        """
        package = session.get("package")
        recorded_digest = session.get("package_digest")
        if not package or not recorded_digest:
            missing = "package" if not package else "package_digest"
            raise StaleTakeError(
                f"session {session_id!r} has a take pushed but no {missing} "
                "recorded from launch, so freshness cannot be checked. "
                "Re-launch Director from the graph (AtlasExportScenePackage "
                "-> launch) so a digest is recorded to check against."
            )
        self.check_fresh(package, recorded_digest, session.get("slate") or "")

    # -- frames -----------------------------------------------------------

    def frame_files(self, take_dir: str) -> list[Path]:
        """The frames that exist, in order.

        NOT `manifest.frameCount`. A playblast is a marked range: the deck's
        in/out points decide what was rendered, and the manifest counts the
        whole take. Reading the manifest would claim frames nobody rendered.

        Refuses, naming the cause, rather than raising a bare
        `FileNotFoundError` when there is no `playblast/` directory at all --
        which happens for real when a take was rendered to `playblast.mp4`
        instead of a frame sequence (this node needs a sequence; it does not
        decode video).
        """
        playblast = Path(take_dir) / "playblast"
        if not playblast.is_dir():
            mp4 = Path(take_dir) / "playblast.mp4"
            if mp4.exists():
                raise ValueError(
                    f"{take_dir} was rendered as playblast.mp4 (a video "
                    "file), not a frame sequence. AtlasDirectorTake needs a "
                    "rendered image sequence under playblast/ -- re-render "
                    "this take with a PNG or EXR sequence output."
                )
            raise ValueError(
                f"no playblast/ directory found under {take_dir!r}."
            )
        return sorted(p for p in playblast.iterdir() if p.is_file())

    def _load_samples(self, take_dir: str) -> list[dict]:
        return json.loads((Path(take_dir) / "samples.json").read_text())

    def read_rays(self, take_dir: str, width: int, height: int):
        """Ray origins/directions per rendered frame -- `ray_map`, untouched.

        NO coordinate conversion happens here, or anywhere downstream in
        this node: samples arrive in Atlas canonical space already, and
        `ray_map` encodes them exactly as given.
        """
        samples = self._load_samples(take_dir)
        rendered = len(self.frame_files(take_dir))
        return ray_map(samples[:rendered], width, height)

    def embed_rays(self, rays):
        """Full-precision Plücker embedding: `(o x d, d)`, 6 channels."""
        return plucker_embedding(rays)

    def rays_to_preview(self, rays):
        """Display-only 3-channel preview from the direction channels.

        Direction remapped to viewable range as `0.5 * (d + 1.0)`. The
        moment is unbounded and is NEVER remapped here -- this output never
        feeds anything but a preview widget.
        """
        np = _require_numpy()
        directions = np.asarray(rays)[..., 3:]
        return (0.5 * (directions + 1.0)).astype(np.float32)

    def _load_png_frames(self, frame_paths: list[Path]):
        np = _require_numpy()
        PILImage = _require_pil()
        torch = _require_torch()
        frames = []
        for path in frame_paths:
            image = PILImage.open(path).convert("RGB")
            frames.append(np.array(image, dtype=np.float32) / 255.0)
        batch = np.stack(frames, axis=0)  # (N, H, W, 3)
        return torch.from_numpy(batch)

    def _load_exr_frames(self, frame_paths: list[Path]):
        np = _require_numpy()
        torch = _require_torch()
        from atlas_camera.plate.oiio_io import read_plate

        frames = []
        for path in frame_paths:
            # raw_data=True: these are numbers a model conditions on, not
            # colour for a human to look at -- no colour conversion, and no
            # divide by 255 (the file is float already).
            plate = read_plate(str(path), raw_data=True)
            frames.append(np.asarray(plate.pixels, dtype=np.float32))
        batch = np.stack(frames, axis=0)
        return torch.from_numpy(batch)

    def load_playblast(self, take_dir: str, colour_lane: str):
        """Load the rendered frames, refusing rather than silently coercing
        when `colour_lane` does not match what is actually on disk.

        The upstream playblast operation
        (`atlas_scene.operations.playblast_ops.copy_exr_sequence`) delivers
        8-bit sRGB PNG today -- its own docstring says so -- there is no
        real scene-linear EXR lane yet. Detecting the real extension and
        refusing a mismatch keeps that honest instead of feeding PNG bytes
        through the float/data path (or vice versa) and calling it EXR.
        """
        frame_paths = self.frame_files(take_dir)
        extensions = {path.suffix.lower() for path in frame_paths}
        if len(extensions) != 1:
            raise ValueError(
                f"mixed frame formats in {take_dir}/playblast: "
                f"{sorted(extensions)}. A take must render one format."
            )
        extension = next(iter(extensions))

        if colour_lane == "exr":
            if extension != ".exr":
                raise ValueError(
                    "colour_lane='exr' was requested, but this take's "
                    f"playblast frames are {extension} -- the upstream "
                    "playblast lane (playblast_ops.copy_exr_sequence) "
                    "delivers 8-bit sRGB PNG today, not scene-linear EXR; a "
                    "real EXR lane does not exist yet. Use colour_lane="
                    "'png'."
                )
            return self._load_exr_frames(frame_paths)

        if extension == ".exr":
            raise ValueError(
                "colour_lane='png' was requested, but this take's "
                "playblast frames are EXR. Use colour_lane='exr'."
            )
        return self._load_png_frames(frame_paths)

    # -- optional ray-map EXR sidecar (addendum Ruling P7) -----------------

    def write_ray_exr(self, take_dir: str, rays) -> str:
        """Write the ray map as a float32 EXR sequence beside the take.

        `<take_dir>/rays/rays.####.exr`. `rays` must be `ray_map`'s output
        shape -- (origin, direction) -- to match `RAY_CHANNEL_NAMES`
        ("O.*", "D.*"). Passing the Plücker EMBEDDING here (moment,
        direction) would write moments under channel names that say
        "origin", which is exactly the silent, plausible-looking error the
        channel naming exists to prevent -- do not do that.

        `bit_depth="float"` is mandatory and explicit: `write_exr` defaults
        to `bit_depth="half"`, whose mantissa step near 1.0 (~0.056 degrees)
        already spends a third of the whole Plücker fidelity budget, and is
        16x coarser again at moment magnitudes around 10. No colour
        conversion is passed -- these are numbers, not colour.

        Skipped, not failed, when OpenImageIO is unavailable: the sockets
        are the primary product, and OIIO is an optional dependency in this
        package. Returns the rays directory on success, "" when skipped.
        """
        from atlas_camera.plate.oiio_io import oiio_available, write_exr

        if not oiio_available():
            return ""
        np = _require_numpy()
        rays_dir = Path(take_dir) / "rays"
        rays_dir.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(rays)
        for index in range(arr.shape[0]):
            path = rays_dir / f"rays.{index:04d}.exr"
            write_exr(str(path), arr[index], bit_depth="float")
        self._record_channel_naming(take_dir)
        return str(rays_dir)

    def _record_channel_naming(self, take_dir: str) -> None:
        """Add `rayChannels` to the take's manifest, preserving everything
        else about it: the deck owns this file's format and key order, and
        this node adds exactly one key rather than re-authoring the whole
        document in its own style. Python's `json.loads` already preserves
        key order as insertion order, so the only thing left to preserve is
        whether the file was pretty-printed or compact -- detected from the
        raw text rather than assumed.
        """
        manifest_path = Path(take_dir) / "manifest.json"
        raw = manifest_path.read_text()
        manifest = json.loads(raw)
        manifest["rayChannels"] = list(RAY_CHANNEL_NAMES)
        if "\n" in raw:
            manifest_path.write_text(json.dumps(manifest, indent=2))
        else:
            manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))

    # -- execution ----------------------------------------------------------

    def read(self, session_id: str, width: int, height: int, frames: str,
              fps: int, colour_lane: str) -> dict[str, Any]:
        session = self._session(session_id)
        take_dir = session["take_dir"]
        self._ensure_fresh(session_id, session)

        rendered = len(self.frame_files(take_dir))
        samples = self._load_samples(take_dir)[:rendered]

        playblast = self.load_playblast(take_dir, colour_lane)
        rays = ray_map(samples, width, height)
        embedded = self.embed_rays(rays)
        preview_np = self.rays_to_preview(rays)
        torch = _require_torch()
        rays_preview = torch.from_numpy(preview_np)

        rays_dir = self.write_ray_exr(take_dir, rays)
        note = (f"ray map EXR written to {rays_dir}" if rays_dir else
                "OpenImageIO unavailable -- ray-map EXR sidecar skipped "
                "(sockets are unaffected).")

        return {
            "result": (playblast, embedded, rays_preview, samples),
            "ui": {"text": [note]},
        }
