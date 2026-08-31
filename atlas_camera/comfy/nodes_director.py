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
node's ``session_id``, and its ``output_dir`` must be the ``scenes``
subdirectory of the configured ``ATLAS_DIRECTOR_ROOT``, given as an
absolute path (the export node's own default, ``atlas_scenes``, is
relative to ComfyUI's working directory and will not do --
``AtlasExportScenePackage`` writes to ``<output_dir>/<name>.atlas``
directly, while ``launch_session`` looks for
``<root>/scenes/<name>.atlas``). Nothing here enforces that mapping in
code: ``director_session.launch_session`` already refuses to spawn
Director when the package is absent, naming the fix, so a mismatch fails
loudly at launch -- before anyone shoots a take.

Launching Director itself needs ``ATLAS_DIRECTOR_BIN`` set to the Director
binary. A packaged install needs nothing more. A development install, where
the only binary is Electron itself and it requires the app directory as its
first argument, also needs ``ATLAS_DIRECTOR_ARGS`` set to that directory --
see ``director_session.director_extra_args()``. Both are configuration,
read from the environment only, never from this node or the launch request.
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


#: The official LTX frame-count constraint, confirmed by the user: LTX only
#: accepts a frame count `n` satisfying `n % LTX_FRAME_MODULUS ==
#: LTX_FRAME_REMAINDER` (1, 9, 17, 25, ..., 121, 129, ...). This replaces the
#: `ALLOWED_FRAMES = ("121",)` placeholder that stood in for the real rule
#: while it was unknown (spec open question 7) -- refusing everything but
#: one known-valid value was the safe direction until the rule itself could
#: be stated exactly.
LTX_FRAME_MODULUS = 8
LTX_FRAME_REMAINDER = 1

#: Default for the `frames` INT widget -- the same value the old
#: `ALLOWED_FRAMES` placeholder's single entry used, still a valid length.
DEFAULT_FRAMES = 121

#: width/height/frames/fps are LAUNCH TARGETS ONLY -- pushed to Director
#: before any take exists, so Director has a timebase to draw frame-lines
#: against. None of the four describes what actually got rendered: that is
#: a property of the take directory, read back through this node's
#: `frame_count`/`width`/`height` OUTPUTS instead (derived from the frame
#: files themselves -- see `_read_frame_dimensions`). A take marked to a
#: different length or shot at a different canvas size than these widgets
#: is normal operator behaviour, not an error; `read()` no longer refuses
#: on that disagreement, only notes it in its text output.
WIDTH_TOOLTIP = (
    "Launch target only: part of the timebase pushed to Director before "
    "any take exists, for drawing frame-lines. NOT the rendered "
    "playblast's actual pixel width -- read that off this node's `width` "
    "OUTPUT, derived from the first rendered frame. A mismatch between "
    "this widget and the rendered take is normal, not an error."
)
HEIGHT_TOOLTIP = (
    "Launch target only: part of the timebase pushed to Director before "
    "any take exists, for drawing frame-lines. NOT the rendered "
    "playblast's actual pixel height -- read that off this node's "
    "`height` OUTPUT, derived from the first rendered frame. A mismatch "
    "between this widget and the rendered take is normal, not an error."
)
FPS_TOOLTIP = (
    "Launch target only: part of the timebase pushed to Director before "
    "any take exists. This node has no per-frame timing to derive an "
    "actual fps from, so there is no `fps` output -- unlike width/height/"
    "frames, this widget has nothing on the take to disagree with."
)

#: Read at the widget so the operator sees the rule where they set the
#: value, not just when a take later gets refused for violating it.
FRAMES_TOOLTIP = (
    "Launch target only: part of the timebase pushed to Director before "
    "any take exists, for drawing frame-lines. NOT a description of the "
    "rendered take -- read the actual rendered length off this node's "
    "`frame_count` OUTPUT, derived from the frame files on disk (see "
    "`frame_files`). A take marked to a different length than this widget "
    "is normal, not an error. LTX only accepts frame counts n where "
    "n % 8 == 1 (1, 9, 17, 25, ..., 121, 129, ...) -- enforced on this "
    "widget too, so frame-lines land on valid marks; `read()` separately "
    "refuses, naming the nearest valid counts, when the RENDERED count "
    "violates this rule -- see `_ensure_frame_count_is_ltx_valid`."
)


def is_ltx_valid_frame_count(n: int) -> bool:
    """The official LTX constraint: `n % 8 == 1`."""
    return n % LTX_FRAME_MODULUS == LTX_FRAME_REMAINDER


def nearest_ltx_valid_frame_counts(n: int) -> tuple[int, int]:
    """The nearest valid frame counts either side of an invalid `n`.

    Valid counts are spaced `LTX_FRAME_MODULUS` apart starting at
    `LTX_FRAME_REMAINDER` (1, 9, 17, ...). Clamped at 1 -- there is no valid
    count below the smallest one.
    """
    lower = n - ((n - LTX_FRAME_REMAINDER) % LTX_FRAME_MODULUS)
    if lower < LTX_FRAME_REMAINDER:
        lower = LTX_FRAME_REMAINDER
    upper = lower + LTX_FRAME_MODULUS
    return lower, upper

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

    Returns seven sockets:

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
    * ``frame_count``, ``width``, ``height`` -- the batch's REAL shape, read
      off the playblast directory itself (frame count from `frame_files`,
      pixel size from the first rendered frame) rather than from the
      `frames`/`width`/`height` widgets, which are launch targets set
      before any take existed and can legitimately disagree with what got
      shot. `rays`/`rays_preview` are built at this derived resolution, not
      the widgets', so they cannot drift from `playblast`'s actual pixels --
      see `read()`.
    * ``projected`` -- the same marked span as `playblast`, but the plate
      projected onto the geometry instead of a grey flat-shaded render.
      Loaded through the exact same code path as `playblast`
      (`_load_frame_batch` / `_load_png_frames` / `_load_exr_frames`) so the
      two batches cannot diverge in dtype, channel order or normalisation.
    * ``first_frame`` -- `projected`'s frame 0 as a batch of size 1: the
      marked IN-point, photoreal. This is deliberately not left for the
      graph to slice out of `projected` itself -- the video model's
      contract wants exactly one frame, and an operator-wired "index 0"
      node is a mistake waiting to happen.

    `projected` is refused, naming the take as needing a re-push, when the
    take predates the second sequence (no `projected/` directory), or when
    its `playblast.json` sidecar or frame count disagrees with
    `playblast/`'s -- a projected batch that does not correspond
    frame-for-frame with the grey guide is worse than none: it looks usable
    and is silently misaligned. See `projected_frame_files`,
    `_ensure_projected_matches_playblast`.

    Delivery address: after the "Launch Director" button opens Director on
    a session, Director pushes the finished take back by calling
    ``deliverTake`` against ``ATLAS_COMFY_URL``, falling back to
    ``http://127.0.0.1:8188`` when that is unset. The launch request never
    carries this ComfyUI's own address to Director -- doing so would put a
    request-influenced value on Director's spawn command line, which
    ``director_session.py`` deliberately never does. If this ComfyUI is not
    on the default host/port, set ``ATLAS_COMFY_URL`` in ComfyUI's own
    process environment before launching: Director inherits it from the
    process that spawns it. A misconfigured address is silent -- the launch
    itself can still succeed with nowhere for the take to land.
    """

    RETURN_TYPES = ("IMAGE", "ATLAS_RAYS", "IMAGE", "ATLAS_CAMERA",
                     "INT", "INT", "INT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("playblast", "rays", "rays_preview", "samples",
                     "frame_count", "width", "height",
                     "projected", "first_frame")
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
                "width": ("INT", {
                    "default": 768, "min": 16, "max": 4096,
                    "tooltip": WIDTH_TOOLTIP,
                }),
                "height": ("INT", {
                    "default": 512, "min": 16, "max": 4096,
                    "tooltip": HEIGHT_TOOLTIP,
                }),
                "frames": ("INT", {
                    "default": DEFAULT_FRAMES,
                    "min": 1,
                    "max": 4096,
                    "step": LTX_FRAME_MODULUS,
                    "tooltip": FRAMES_TOOLTIP,
                }),
                "fps": ("INT", {
                    "default": 24, "min": 1, "max": 120,
                    "tooltip": FPS_TOOLTIP,
                }),
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

    @classmethod
    def VALIDATE_INPUTS(cls, frames: int) -> "bool | str":
        """ComfyUI's graph-validation hook: catches an invalid `frames`
        widget value before the graph is even queued, not just when
        `read()` runs. Mirrors `_ensure_frames_widget_is_valid`'s rule.
        """
        if is_ltx_valid_frame_count(int(frames)):
            return True
        return (
            f"frames={frames} is not a length LTX accepts -- LTX only "
            "takes frame counts n where n % 8 == 1 (1, 9, 17, 25, ..., "
            "121, 129, ...)."
        )

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
                "package it was launched from. Re-pushing the same take will "
                "not fix this -- the recorded digest was taken when Director "
                "was launched, so re-pushing re-delivers the same slate "
                "against the same stale digest. Relaunch Director from this "
                "node (the package changed since it was opened, for example "
                "from a Save in Director)."
            )
        return True

    def _ensure_timebase_matches(self, session_id: str, session: dict, *,
                                  width: int, height: int, frames: int,
                                  fps: int) -> None:
        """Refuse when this node's widgets do not match the session's timebase.

        `launch_session` records width/height/frames/fps as the timebase
        Director opened with -- the ray map and the playblast frames shipped
        beside it are only meaningful together at that resolution and frame
        count. A silently mismatched width/height still *runs*: it just
        produces a ray map that does not correspond to the rendered frames.
        Refused, not warned about, and naming both values so the fix is
        obvious.
        """
        requested = {
            "width": int(width), "height": int(height),
            "frames": int(frames), "fps": int(fps),
        }
        timebase = session.get("timebase")
        if not timebase:
            raise ValueError(
                f"session {session_id!r} has no timebase recorded from "
                "launch, so this node's width/height/frames/fps cannot be "
                "checked against it. Re-launch Director from the graph so a "
                "timebase is recorded to check against."
            )
        if timebase != requested:
            raise ValueError(
                f"this node's timebase {requested} does not match session "
                f"{session_id!r}'s recorded timebase {timebase} -- Director "
                "was launched with different width/height/frames/fps than "
                "this node is reading with, so the ray map would not "
                "correspond to the shipped playblast frames. Match this "
                "node's widgets to the session's timebase."
            )

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

    #: The marks sidecar `capturePlayblast` writes beside the frames
    #: (Finding 2). Lives inside `playblast/` alongside the frame files, so
    #: `frame_files` must exclude it by name -- it is not a rendered frame.
    PLAYBLAST_MARKS_NAME = "playblast.json"

    def frame_files(self, take_dir: str) -> list[Path]:
        """The frames that exist, in order.

        NOT `manifest.frameCount`. A playblast is a marked range: the deck's
        in/out points decide what was rendered, and the manifest counts the
        whole take. Reading the manifest would claim frames nobody rendered.

        This sorts lexically and accepts any file under `playblast/` other
        than `playblast.json` (the marks sidecar, not a frame), unlike
        `atlas_scene.operations.playblast_ops.frame_files`, which enforces
        the `frame_%06d.png` naming. Harmless only because the producer on
        the other side (`playblast_ops.copy_exr_sequence`) always zero-pads
        -- if that assumption ever stops holding, this function's lexical
        sort silently reorders frames instead of refusing.

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
        return sorted(
            p for p in playblast.iterdir()
            if p.is_file() and p.name != self.PLAYBLAST_MARKS_NAME
        )

    # -- the projected (photoreal) sequence --------------------------------

    #: `<take_dir>/projected/` -- the same marked span as `playblast/`, the
    #: plate projected onto the geometry instead of a grey flat-shaded
    #: render. Written alongside `playblast/` by the same take push; a take
    #: pushed before this existed has no such directory at all.
    PROJECTED_DIR_NAME = "projected"

    def projected_frame_files(self, take_dir: str) -> list[Path]:
        """The `projected/` frames, in the same order `frame_files` returns
        `playblast/`'s.

        A take pushed before the second sequence existed has no
        `projected/` directory -- refused by name, in the same posture
        already used for a missing marks sidecar (`_load_playblast_marks`):
        this is "predates the feature", not an error to warn past. Unlike
        `frame_files`, there is no `.mp4` fallback case to detect here --
        `projected/` either exists as a frame sequence or the take predates
        it.
        """
        projected = Path(take_dir) / self.PROJECTED_DIR_NAME
        if not projected.is_dir():
            raise StaleTakeError(
                f"no {self.PROJECTED_DIR_NAME}/ directory found under "
                f"{take_dir!r} -- this take predates the second (projected) "
                "sequence and must be re-pushed. Re-push this take from "
                "Director so both playblast/ and projected/ are written."
            )
        return sorted(
            p for p in projected.iterdir()
            if p.is_file() and p.name != self.PLAYBLAST_MARKS_NAME
        )

    def _load_projected_marks(self, take_dir: str) -> dict:
        """Read `<take_dir>/projected/playblast.json`, the same marks
        sidecar schema `_load_playblast_marks` reads from `playblast/`
        (identical contents, by contract -- see `_ensure_projected_matches_
        playblast`).
        """
        marks_path = (
            Path(take_dir) / self.PROJECTED_DIR_NAME / self.PLAYBLAST_MARKS_NAME
        )
        if not marks_path.exists():
            raise StaleTakeError(
                f"no {self.PLAYBLAST_MARKS_NAME} found under "
                f"{take_dir!r}/{self.PROJECTED_DIR_NAME} -- this take "
                "predates the second (projected) sequence and must be "
                "re-pushed."
            )
        return json.loads(marks_path.read_text())

    def _ensure_projected_matches_playblast(
        self, take_dir: str, *, playblast_frame_paths: list[Path],
        projected_frame_paths: list[Path],
    ) -> None:
        """Refuse when `projected/` does not correspond frame-for-frame with
        `playblast/`.

        A projected batch that does not line up with the grey guide is
        worse than none: it looks usable and is silently misaligned. Checks
        two independent things, either one enough to refuse on its own: the
        marks sidecars must be byte-identical in content (both directories
        cover the same marked span), and the two directories must hold the
        same number of frames.
        """
        playblast_marks = self._load_playblast_marks(take_dir)
        projected_marks = self._load_projected_marks(take_dir)
        if projected_marks != playblast_marks:
            raise StaleTakeError(
                f"projected/{self.PLAYBLAST_MARKS_NAME} under {take_dir!r} "
                f"disagrees with playblast/{self.PLAYBLAST_MARKS_NAME}: "
                f"{projected_marks!r} != {playblast_marks!r} -- the two "
                "sequences must describe the same marked range. Re-push "
                "this take."
            )
        n_playblast = len(playblast_frame_paths)
        n_projected = len(projected_frame_paths)
        if n_playblast != n_projected:
            raise StaleTakeError(
                f"projected/ under {take_dir!r} has {n_projected} frame(s) "
                f"but playblast/ has {n_playblast} -- the two sequences "
                "must cover the same marked span frame-for-frame. Re-push "
                "this take."
            )

    def _ensure_projected_dimensions_match(
        self, take_dir: str, *, playblast_size: tuple[int, int],
        projected_size: tuple[int, int],
    ) -> None:
        """Refuse when `projected/`'s pixel size disagrees with
        `playblast/`'s. Both come from the same canvas, so they should
        already agree -- verified rather than assumed.
        """
        if playblast_size != projected_size:
            pb_w, pb_h = playblast_size
            pj_w, pj_h = projected_size
            raise StaleTakeError(
                f"projected/ frames under {take_dir!r} are {pj_w}x{pj_h} "
                f"but playblast/ frames are {pb_w}x{pb_h} -- both come "
                "from the same canvas and must match. Re-push this take."
            )

    def _load_samples(self, take_dir: str) -> list[dict]:
        return json.loads((Path(take_dir) / "samples.json").read_text())

    # -- marked-range alignment (Finding 2) --------------------------------

    def _load_playblast_marks(self, take_dir: str) -> dict:
        """Read `<take_dir>/playblast/playblast.json`, the marked-range
        record the Director renderer writes beside the frames it staged.

        Required, never optional, and never a silent fallback to "assume
        the head": `capturePlayblast` stages a marked range starting at
        frame file index 0 regardless of where in the take those frames
        actually came from (`frame_000000.png` upward either way), so
        without this sidecar there is no way to tell "the whole take" from
        "marked in at frame 100" -- and guessing wrong produces a ray map
        that describes the wrong frames with no error at all (the exact
        failure this file exists to remove). Nothing shipped before this
        sidecar existed, so its absence means the playblast predates the
        marks record, not "this is an old-style unmarked take".
        """
        marks_path = Path(take_dir) / "playblast" / self.PLAYBLAST_MARKS_NAME
        if not marks_path.exists():
            raise StaleTakeError(
                f"no {self.PLAYBLAST_MARKS_NAME} found under "
                f"{take_dir!r}/playblast -- this playblast predates the "
                "marks record and cannot be safely aligned to samples.json "
                "(a marked range's pixels do not start at take frame 0, and "
                "there is no way to tell without this file). Re-render this "
                "take."
            )
        return json.loads(marks_path.read_text())

    def _aligned_sample_range(self, take_dir: str, frame_paths: list[Path],
                               *, sample_count: int) -> tuple[int, int]:
        """The `[start, stop)` slice into `samples.json` the rendered frames cover.

        `capturePlayblast` stages the deck's marked range from frame-file
        index 0, so `playblast/frame_000000.png` upward exists regardless
        of where in the take those frames came from -- `samples[:rendered]`
        alone would describe take frames `0..rendered-1` even when the
        pixels are actually frames `in..out` of a marked range (Finding 2).

        Reads `playblast.json`'s `marked` / `in` / `out` / `frame_count`.
        `marked=true` means the pixels are take frames `in..out` inclusive,
        so the sample slice is `[in, out+1)`; `marked=false` means the
        frames are the head of the take, so `[0, frame_count)` -- and a
        `marked=false` sidecar recording a non-null `in`/`out` is a
        contradiction, refused rather than one of the two claims being
        silently preferred. `frame_count` is checked against the number of
        frame files actually on disk unconditionally; the derived range's
        WIDTH (`stop - start`) is checked against that same on-disk count
        too, which is a real, non-redundant check for a marked range (a
        sidecar can misdescribe `in`/`out` while still reporting a correct
        `frame_count`) and is merely restating the same fact for an
        unmarked one (there `stop - start` reduces to `frame_count`
        itself).

        `caller-round-1 CRITICAL fix`: none of the above proves `start` and
        `stop` are valid indices into `samples.json` -- a sidecar naming
        frames past the end of the take (or a negative `in`, which can
        satisfy the width check on its own via Python's negative-index
        arithmetic, e.g. `in=-5, out=-2` on a 4-frame render) would
        otherwise slice `samples[start:stop]` into something shorter than
        the playblast, or the wrong tail of the list, with no error at
        all -- the exact silent-wrong-answer class this whole method
        exists to remove, reproduced inside its own fix. So `start` and
        `stop` are bounds-checked against `sample_count`
        (`len(samples.json)`, supplied by the caller so this function need
        not load the file itself) before being returned: `0 <= start <
        stop <= sample_count` is required, and a negative `in` is refused
        by name rather than relying on that bounds check alone to catch it.
        """
        marks = self._load_playblast_marks(take_dir)
        on_disk = len(frame_paths)
        frame_count = marks.get("frame_count")
        marked = bool(marks.get("marked"))
        mark_in = marks.get("in")
        mark_out = marks.get("out")

        if frame_count != on_disk:
            raise StaleTakeError(
                f"playblast.json under {take_dir!r} records frame_count="
                f"{frame_count!r}, but {on_disk} frame file(s) actually "
                "exist under playblast/ -- the sidecar and the directory "
                "describe different renders. Re-render this take."
            )

        if marked:
            if mark_in is None or mark_out is None:
                raise StaleTakeError(
                    f"playblast.json under {take_dir!r} says marked=true "
                    f"but in={mark_in!r}/out={mark_out!r} -- a marked range "
                    "must record both. Re-render this take."
                )
            mark_in, mark_out = int(mark_in), int(mark_out)
            if mark_in < 0 or mark_out < 0:
                raise StaleTakeError(
                    f"playblast.json under {take_dir!r} records a negative "
                    f"mark (in={mark_in}, out={mark_out}) -- a marked range "
                    "cannot start or end before take frame 0. Re-render "
                    "this take."
                )
            start, stop = mark_in, mark_out + 1
            range_desc = f"marked in={mark_in} out={mark_out}"
        else:
            if mark_in is not None or mark_out is not None:
                raise StaleTakeError(
                    f"playblast.json under {take_dir!r} says marked=false "
                    f"but records in={mark_in!r}/out={mark_out!r} -- an "
                    "unmarked sidecar must not also carry a mark range. "
                    "Re-render this take."
                )
            start, stop = 0, int(frame_count)
            range_desc = "unmarked head"

        if stop - start != on_disk:
            raise StaleTakeError(
                f"playblast.json under {take_dir!r} describes a "
                f"{range_desc} range of {stop - start} frame(s), but "
                f"{on_disk} frame file(s) actually exist under playblast/ "
                "-- the sidecar and the directory describe different "
                "renders. Re-render this take."
            )

        if start < 0 or stop <= start or stop > sample_count:
            raise StaleTakeError(
                f"playblast.json under {take_dir!r} describes take-sample "
                f"range [{start}, {stop}) ({range_desc}), but samples.json "
                f"only has {sample_count} sample(s) recorded -- the "
                "sidecar names frames the take does not have. Re-render "
                "this take."
            )
        return start, stop

    def _read_frame_dimensions(self, take_dir: str,
                                frame_paths: list[Path]) -> tuple[int, int]:
        """The playblast's actual pixel size, read off the first rendered
        frame (Finding 3, corrected).

        `width`/`height` widgets are launch targets only -- pushed to
        Director before any take exists, so nothing carries them into the
        Director canvas that actually rendered the frames, and the two can
        disagree freely. Previously that gap needed a check
        (`_ensure_frame_dimensions_match`, refusing on disagreement) because
        `read()` built the ray map at the WIDGET size regardless -- two
        sources of truth for one fact, one of them wrong whenever they
        disagreed. Deriving the ray map's resolution from these frames
        instead (see `read()`) removes the second source of truth rather
        than continuing to guard it: there is nothing left to mismatch, so
        this only reads, never refuses.
        """
        if not frame_paths:
            raise ValueError(
                f"no rendered frames under {take_dir!r}/playblast to read "
                "dimensions from."
            )
        PILImage = _require_pil()
        with PILImage.open(frame_paths[0]) as image:
            return image.size  # (width, height)

    def _ensure_frames_widget_is_valid(self, frames: int) -> None:
        """Refuse when the `frames` widget itself violates the LTX rule.

        The widget is an INT with `step=LTX_FRAME_MODULUS`, which keeps the
        UI slider on valid values, but a typed-in value can still bypass
        that -- and this method is also what tests exercise directly,
        without going through ComfyUI's own `VALIDATE_INPUTS` path. Checked
        independently of the rendered take: this is about the widget's own
        value, not about what got shot.
        """
        if not is_ltx_valid_frame_count(int(frames)):
            raise ValueError(
                f"frames={frames} is not a length LTX accepts -- LTX only "
                "takes frame counts n where n % 8 == 1 (1, 9, 17, 25, ..., "
                "121, 129, ...). Set frames to a value satisfying that "
                "rule."
            )

    def _ensure_frame_count_is_ltx_valid(self, take_dir: str,
                                          frame_paths: list[Path]) -> None:
        """Refuse when the rendered frame count is not a length LTX accepts
        (Finding 3, corrected: the real rule replaces the
        rendered-equals-`frames`-widget proxy).

        `_aligned_sample_range` proves the sidecar describes the directory
        it lives in -- it says nothing about whether that many frames is a
        length the model will accept. That question now has an exact
        answer (`is_ltx_valid_frame_count`), so it is checked directly
        against the rendered count instead of via equality with the
        `frames` widget: the widget records what Director was launched
        with, but a take marked to a different, still-valid length (say 73
        frames from a shot launched at 121) is a legitimate take, not an
        operator mistake, and forcing the widget to match it for every take
        bought no real safety. A rendered length that fails the LTX rule
        itself, however, is refused here -- naming the count and the
        nearest valid counts either side -- so the failure lands at the
        mark instead of surfacing later as a confusing model error inside
        the LTX call.
        """
        actual = len(frame_paths)
        if not is_ltx_valid_frame_count(actual):
            lower, upper = nearest_ltx_valid_frame_counts(actual)
            raise StaleTakeError(
                f"playblast under {take_dir!r} has {actual} rendered "
                f"frame(s), which LTX will not accept -- LTX only takes "
                "frame counts n where n % 8 == 1. "
                f"{actual} frames is not valid; nearest are {lower} and "
                f"{upper}. Re-mark the take to a valid length."
            )

    def read_rays(self, take_dir: str, width: int, height: int):
        """Ray origins/directions per rendered frame -- `ray_map`, untouched.

        NO coordinate conversion happens here, or anywhere downstream in
        this node: samples arrive in Atlas canonical space already, and
        `ray_map` encodes them exactly as given.

        Aligned to the rendered frames via `_aligned_sample_range`, NOT a
        plain `samples[:rendered]` head-slice: a playblast is frequently a
        marked sub-range (`in..out`), and the frame files always start at
        index 0 regardless of where in the take they came from (Finding 2).
        """
        frame_paths = self.frame_files(take_dir)
        all_samples = self._load_samples(take_dir)
        start, stop = self._aligned_sample_range(
            take_dir, frame_paths, sample_count=len(all_samples),
        )
        return ray_map(all_samples[start:stop], width, height)

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

    def _load_frame_batch(self, take_dir: str, frame_paths: list[Path],
                           colour_lane: str, *, directory_name: str):
        """The shared loading path: frames listed, format-checked, read and
        stacked into a tensor -- used by both `load_playblast` and
        `load_projected` so the two batches cannot diverge in dtype,
        channel order or normalisation. `directory_name` only changes the
        wording of a refusal message; it does not change behaviour.

        Refuses rather than silently coercing when `colour_lane` does not
        match what is actually on disk. The upstream playblast operation
        (`atlas_scene.operations.playblast_ops.copy_exr_sequence`) delivers
        8-bit sRGB PNG today -- its own docstring says so -- there is no
        real scene-linear EXR lane yet. Detecting the real extension and
        refusing a mismatch keeps that honest instead of feeding PNG bytes
        through the float/data path (or vice versa) and calling it EXR.
        """
        extensions = {path.suffix.lower() for path in frame_paths}
        if len(extensions) != 1:
            raise ValueError(
                f"mixed frame formats in {take_dir}/{directory_name}: "
                f"{sorted(extensions)}. A take must render one format."
            )
        extension = next(iter(extensions))

        if colour_lane == "exr":
            if extension != ".exr":
                raise ValueError(
                    "colour_lane='exr' was requested, but this take's "
                    f"{directory_name} frames are {extension} -- the "
                    "upstream playblast lane (playblast_ops."
                    "copy_exr_sequence) delivers 8-bit sRGB PNG today, not "
                    "scene-linear EXR; a real EXR lane does not exist yet. "
                    "Use colour_lane='png'."
                )
            return self._load_exr_frames(frame_paths)

        if extension == ".exr":
            raise ValueError(
                "colour_lane='png' was requested, but this take's "
                f"{directory_name} frames are EXR. Use colour_lane='exr'."
            )
        return self._load_png_frames(frame_paths)

    def load_playblast(self, take_dir: str, colour_lane: str):
        """Load the grey/flat-shaded rendered frames. See
        `_load_frame_batch` -- the shared loading path with `load_projected`.
        """
        frame_paths = self.frame_files(take_dir)
        return self._load_frame_batch(
            take_dir, frame_paths, colour_lane, directory_name="playblast",
        )

    def load_projected(self, take_dir: str, colour_lane: str):
        """Load the plate-projected rendered frames. See
        `_load_frame_batch` -- the shared loading path with `load_playblast`,
        so the two batches cannot diverge in dtype, channel order or
        normalisation.
        """
        frame_paths = self.projected_frame_files(take_dir)
        return self._load_frame_batch(
            take_dir, frame_paths, colour_lane, directory_name="projected",
        )

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

    def read(self, session_id: str, width: int, height: int, frames: int,
              fps: int, colour_lane: str) -> dict[str, Any]:
        self._ensure_frames_widget_is_valid(frames)
        session = self._session(session_id)
        take_dir = session["take_dir"]
        self._ensure_fresh(session_id, session)
        self._ensure_timebase_matches(
            session_id, session, width=width, height=height,
            frames=frames, fps=fps,
        )

        frame_paths = self.frame_files(take_dir)
        self._ensure_frame_count_is_ltx_valid(take_dir, frame_paths)
        actual_width, actual_height = self._read_frame_dimensions(
            take_dir, frame_paths,
        )
        actual_frame_count = len(frame_paths)

        # projected/ must correspond frame-for-frame with playblast/ --
        # refuses (naming what disagreed) rather than silently reading a
        # misaligned batch. See AtlasDirectorTake's class docstring.
        projected_frame_paths = self.projected_frame_files(take_dir)
        self._ensure_projected_matches_playblast(
            take_dir, playblast_frame_paths=frame_paths,
            projected_frame_paths=projected_frame_paths,
        )
        projected_width, projected_height = self._read_frame_dimensions(
            take_dir, projected_frame_paths,
        )
        self._ensure_projected_dimensions_match(
            take_dir,
            playblast_size=(actual_width, actual_height),
            projected_size=(projected_width, projected_height),
        )

        all_samples = self._load_samples(take_dir)
        start, stop = self._aligned_sample_range(
            take_dir, frame_paths, sample_count=len(all_samples),
        )
        samples = all_samples[start:stop]

        playblast = self.load_playblast(take_dir, colour_lane)
        projected = self.load_projected(take_dir, colour_lane)
        # `projected` frame 0 as a batch of size 1 -- the marked IN-point,
        # the model's photoreal first frame. Sliced here, not left for the
        # graph to derive from `projected` by index (see class docstring).
        first_frame = projected[0:1]
        # Built at the frames' ACTUAL resolution (`actual_width`/
        # `actual_height`, read off the first frame above), NOT the
        # width/height widgets: those are launch targets set before any
        # take existed (see WIDTH_TOOLTIP/HEIGHT_TOOLTIP) and can
        # legitimately disagree with what got rendered. Deriving the ray
        # map from the same frames the batch came from means the rays and
        # the pixels cannot disagree -- do not put a widget value back in
        # here, or the mismatch this replaces comes back with it.
        rays = ray_map(samples, actual_width, actual_height)
        embedded = self.embed_rays(rays)
        preview_np = self.rays_to_preview(rays)
        torch = _require_torch()
        rays_preview = torch.from_numpy(preview_np)

        rays_dir = self.write_ray_exr(take_dir, rays)
        notes = [f"ray map EXR written to {rays_dir}" if rays_dir else
                 "OpenImageIO unavailable -- ray-map EXR sidecar skipped "
                 "(sockets are unaffected)."]
        # A launch-target/rendered disagreement is normal -- the operator
        # marked a different range, or Director's canvas was a different
        # size, than the shot was launched at -- not an error. Noted in the
        # text output, cheap since the values are already in hand, rather
        # than added as a new failure path.
        launched = (int(width), int(height), int(frames))
        rendered = (actual_width, actual_height, actual_frame_count)
        if launched != rendered:
            notes.append(
                f"launch target was {width}x{height}, {frames} frame(s); "
                f"this take rendered {actual_width}x{actual_height}, "
                f"{actual_frame_count} frame(s)."
            )

        return {
            "result": (playblast, embedded, rays_preview, samples,
                       actual_frame_count, actual_width, actual_height,
                       projected, first_frame),
            "ui": {"text": [" ".join(notes)]},
        }
