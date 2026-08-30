"""AtlasDirectorTake -- the take a director shot, as graph inputs.

Launching Director happens on a widget button, outside execution, so a
queued prompt never waits on a human (spec 3.4). Execution here is a pure
READ of what the session already holds: it never mutates ``SESSIONS``, the
take directory, or any scene document, and it performs no coordinate
conversion -- samples arrive in Atlas canonical space (right-handed, Y-up,
metres, camera down -Z) and are handed to ``plucker.ray_map`` /
``plucker.plucker_embedding`` exactly as they are.

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
    """The session package changed since Director was launched onto it.

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
#: for a human to look at.
COLOUR_LANES = ("png", "exr")

#: Channel naming for the ray-map EXR sidecar, recorded into the take
#: manifest. No consumer exists yet, so the manifest is what a future
#: consumer checks its expectation against, instead of re-deriving it from
#: pixel order.
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
                               "correct failure direction.",
                }),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "colour_lane": (COLOUR_LANES, {
                    "tooltip": "png (default): the 8-bit playblast frames. "
                               "exr: the float colour lane, when the take "
                               "carries one -- read as data, no divide-by-255, "
                               "no colour convert.",
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

    # -- frames -----------------------------------------------------------

    def frame_files(self, take_dir: str) -> list[Path]:
        """The frames that exist, in order.

        NOT `manifest.frameCount`. A playblast is a marked range: the deck's
        in/out points decide what was rendered, and the manifest counts the
        whole take. Reading the manifest would claim frames nobody rendered.
        """
        playblast = Path(take_dir) / "playblast"
        return sorted(p for p in playblast.iterdir() if p.is_file())

    def read_rays(self, take_dir: str, width: int, height: int):
        """Ray origins/directions per rendered frame -- `ray_map`, untouched.

        NO coordinate conversion happens here, or anywhere downstream in
        this node: samples arrive in Atlas canonical space already, and
        `ray_map` encodes them exactly as given.
        """
        samples = json.loads((Path(take_dir) / "samples.json").read_text())
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
        frame_paths = self.frame_files(take_dir)
        if colour_lane == "exr":
            return self._load_exr_frames(frame_paths)
        return self._load_png_frames(frame_paths)

    # -- optional ray-map EXR sidecar (addendum Ruling P7) -----------------

    def write_ray_exr(self, take_dir: str, embedded) -> str:
        """Write the ray map as a float32 EXR sequence beside the take.

        `<take_dir>/rays/rays.####.exr`. `bit_depth="float"` is mandatory
        and explicit: `write_exr` defaults to `bit_depth="half"`, whose
        mantissa step near 1.0 (~0.056 degrees) already spends a third of
        the whole Plücker fidelity budget, and is 16x coarser again at
        moment magnitudes around 10. No colour conversion -- these are
        numbers, not colour.

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
        arr = np.asarray(embedded)
        for index in range(arr.shape[0]):
            path = rays_dir / f"rays.{index:04d}.exr"
            write_exr(str(path), arr[index], bit_depth="float")
        self._record_channel_naming(take_dir)
        return str(rays_dir)

    def _record_channel_naming(self, take_dir: str) -> None:
        manifest_path = Path(take_dir) / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["rayChannels"] = list(RAY_CHANNEL_NAMES)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    # -- execution ----------------------------------------------------------

    def read(self, session_id: str, width: int, height: int, frames: str,
              fps: int, colour_lane: str) -> dict[str, Any]:
        session = self._session(session_id)
        take_dir = session["take_dir"]
        package = session.get("package")
        recorded_digest = session.get("package_digest")
        if package and recorded_digest:
            self.check_fresh(package, recorded_digest, session.get("slate") or "")

        # frames is one of ALLOWED_FRAMES already (a combo widget), int()
        # only converts it for any caller that wants the numeric value.
        int(frames)

        rendered = len(self.frame_files(take_dir))
        all_samples = json.loads((Path(take_dir) / "samples.json").read_text())
        samples = all_samples[:rendered]

        playblast = self.load_playblast(take_dir, colour_lane)
        rays = self.read_rays(take_dir, width, height)
        embedded = self.embed_rays(rays)
        preview_np = self.rays_to_preview(rays)
        torch = _require_torch()
        rays_preview = torch.from_numpy(preview_np)

        rays_dir = self.write_ray_exr(take_dir, embedded)
        note = (f"ray map EXR written to {rays_dir}" if rays_dir else
                "OpenImageIO unavailable -- ray-map EXR sidecar skipped "
                "(sockets are unaffected).")

        return {
            "result": (playblast, embedded, rays_preview, samples),
            "ui": {"text": [note]},
        }
