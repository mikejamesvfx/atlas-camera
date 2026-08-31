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

#: Read at the widget so the operator sees the rule where they set the
#: value, not just when a take later gets refused for violating it.
FRAMES_TOOLTIP = (
    "LTX only accepts frame counts n where n % 8 == 1 (1, 9, 17, 25, ..., "
    "121, 129, ...). Pushed to Director at launch as the shot's timebase "
    "-- the operator marks take ranges against it. Not used to slice this "
    "node's own outputs; the rendered playblast directory decides frame "
    "count (see `frame_files`). `read()` separately refuses, naming the "
    "nearest valid counts, when the rendered frame count itself violates "
    "this rule -- see `_ensure_frame_count_is_ltx_valid`. The rendered "
    "count no longer has to equal this widget's value."
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
                "frames": ("INT", {
                    "default": DEFAULT_FRAMES,
                    "min": 1,
                    "max": 4096,
                    "step": LTX_FRAME_MODULUS,
                    "tooltip": FRAMES_TOOLTIP,
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

    def _ensure_frame_dimensions_match(self, take_dir: str,
                                        frame_paths: list[Path], *,
                                        width: int, height: int) -> None:
        """Refuse when the playblast's actual pixel size doesn't match the
        node's width/height widgets (Finding 3).

        `nodes_director.py` builds the ray map at the node's width/height
        widgets. The playblast frames, however, are whatever size the
        Director window's canvas happened to be when they were rendered --
        nothing carries this node's timebase into that renderer, so the two
        can disagree freely. `_ensure_timebase_matches` cannot catch it: it
        compares the node's widgets against the session body those same
        widgets produced, which is comparing a value to itself. Forcing the
        renderer to a target resolution is out of scope (deferred); this
        only verifies after the fact, against the first rendered frame's
        actual pixels, and refuses loudly rather than silently building a
        ray map that does not correspond to the shipped frames.
        """
        if not frame_paths:
            return
        PILImage = _require_pil()
        first = frame_paths[0]
        with PILImage.open(first) as image:
            actual_width, actual_height = image.size
        expected_width, expected_height = int(width), int(height)
        if (actual_width, actual_height) != (expected_width, expected_height):
            raise StaleTakeError(
                f"playblast frame {first.name!r} under {take_dir!r} is "
                f"{actual_width}x{actual_height} pixels, but this node's "
                f"width/height widgets are "
                f"{expected_width}x{expected_height}. The ray map is built "
                "at the widget size, so a mismatched playblast would not "
                "correspond to it -- match the width/height widgets to the "
                "actual rendered size, or re-render the take at that size."
            )

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
        self._ensure_frame_dimensions_match(
            take_dir, frame_paths, width=width, height=height,
        )
        self._ensure_frame_count_is_ltx_valid(take_dir, frame_paths)
        all_samples = self._load_samples(take_dir)
        start, stop = self._aligned_sample_range(
            take_dir, frame_paths, sample_count=len(all_samples),
        )
        samples = all_samples[start:stop]

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
