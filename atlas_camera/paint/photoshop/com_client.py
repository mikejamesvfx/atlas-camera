"""Drive Adobe Photoshop (Beta) over COM + ExtendScript.

Why COM rather than a UXP plugin: COM is already registered, so there is
nothing to install. A UXP plugin needs a manifest, developer mode enabled in
the Creative Cloud app, and Adobe signing for anything shippable — three user
actions before line one of Atlas code runs, which is exactly the class of
blocker that stalled the Affinity bridge at its manual rung. ExtendScript's
``app.executeAction`` reaches the identical Action Manager surface as UXP
``BatchPlay``, and nothing this bridge needs (``syntheticFill``,
``openAsOCIO``, ``convertToOCIO``, ``placedLayerOCIOSpace``) requires UXP.

Two things make this safe to automate:

**The install-path guard.** Photoshop 2026 is commonly installed alongside the
Beta. The ``Photoshop.Application.BETA`` ProgID is registered specifically to
the Beta executable, and the client verifies ``app.path`` against the expected
install directory before doing anything. The bare ``Photoshop.Application``
ProgID currently resolves to Beta on this machine, but that is an accident of
registration order — driving the wrong Photoshop would silently produce
results attributed to the wrong application.

**The sentinel contract.** Generative fill is asynchronous and network-backed,
so ``DoJavaScript`` may return before variations exist. Every generated script
writes a JSON sidecar as its final statement, and the client polls for that
file. Timeout is reported distinctly from a script error, which in turn is
reported distinctly from a gate failure — three different problems that would
otherwise all read as "the bridge didn't work".

The transport is deliberately thin. The part most likely to be wrong is the
descriptor building, and that lives in ``jsx.py`` as pure functions with unit
tests that need no Photoshop.
"""
from __future__ import annotations

import json
import ntpath
import os
import time
from pathlib import Path

from atlas_camera.paint.photoshop import jsx

#: Registered specifically to the Beta executable.
PROGID_BETA = "Photoshop.Application.BETA"
#: Default install directory the guard checks `app.path` against.
DEFAULT_INSTALL_DIR = r"C:\Program Files\Adobe\Adobe Photoshop (Beta)"


class PhotoshopBridgeError(RuntimeError):
    """Any failure driving Photoshop."""


class PhotoshopNotFound(PhotoshopBridgeError):
    """COM could not hand us the Beta application object."""


class WrongPhotoshop(PhotoshopBridgeError):
    """We got A Photoshop, but not the one we were asked for."""


class ScriptTimeout(PhotoshopBridgeError):
    """The script never wrote its sentinel. Distinct from a script ERROR and
    from a gate failure — do not collapse the three."""


class ScriptFailed(PhotoshopBridgeError):
    """The script ran and reported an error of its own."""


def _normalise(path: str) -> str:
    """Normalise an install path for comparison, accepting ExtendScript's URI form.

    ``String(app.path)`` in ExtendScript yields a Folder URI —
    ``/c/Program%20Files/Adobe/Adobe%20Photoshop%20(Beta)`` — not an OS path.
    The probe asks for ``.fsName`` instead, but older scripts, other hosts and
    a saved result may still carry the URI, and treating the two spellings as
    different applications would refuse the very install we asked for.
    """
    from urllib.parse import unquote

    text = unquote(str(path)).strip().rstrip("\\/")
    if text.startswith("/") and len(text) > 2 and text[2] in "/\\":
        # "/c/Program Files/..." -> "c:/Program Files/..."
        text = f"{text[1]}:{text[2:]}"
    # This is always a WINDOWS application path, even when its tests run on a
    # Linux CI host. ``os.path`` adopts the runner's semantics and therefore
    # leaves backslashes, drive-letter case and forward slashes incomparable.
    return ntpath.normcase(ntpath.normpath(text))


class PhotoshopClient:
    """A connected Photoshop (Beta), verified to be the right one."""

    def __init__(self, *, progid: str = PROGID_BETA,
                 install_dir: str = DEFAULT_INSTALL_DIR,
                 scratch_dir: str | os.PathLike | None = None,
                 timeout_s: float = 300.0, poll_s: float = 0.5):
        self.progid = progid
        self.install_dir = install_dir
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self.scratch = Path(scratch_dir) if scratch_dir else None
        self._app = None
        self._info: dict = {}

    # -- connection ---------------------------------------------------------

    def connect(self, dispatch=None) -> dict:
        """Attach to Photoshop and PROVE it is the expected install.

        ``dispatch`` is injectable so the guard logic can be tested without
        Photoshop or pywin32 present.
        """
        if dispatch is None:
            try:
                from win32com.client import Dispatch  # type: ignore
            except ImportError as exc:                # pragma: no cover - env
                raise PhotoshopNotFound(
                    "pywin32 is required to drive Photoshop over COM: "
                    "pip install pywin32") from exc
            dispatch = Dispatch

        try:
            self._app = dispatch(self.progid)
        except Exception as exc:                      # noqa: BLE001 - COM error
            raise PhotoshopNotFound(
                f"could not start {self.progid!r}. Adobe Photoshop (Beta) must "
                f"be installed; it registers this ProgID itself. Underlying "
                f"error: {exc}") from exc

        info = self.run(jsx.probe, label="probe")
        actual, expected = _normalise(info.get("path", "")), _normalise(
            self.install_dir)
        if expected and actual != expected:
            raise WrongPhotoshop(
                f"connected to a Photoshop at {info.get('path')!r}, but this "
                f"bridge was asked for {self.install_dir!r}. Refusing to "
                f"continue: another Photoshop (e.g. 2026) is probably "
                f"registered, and results would be attributed to the wrong "
                f"application. Pass install_dir= to override deliberately.")
        self._info = info
        return info

    @property
    def info(self) -> dict:
        return dict(self._info)

    # -- script execution ---------------------------------------------------

    def _sentinel_path(self, label: str) -> Path:
        base = self.scratch or Path(os.environ.get("TEMP", ".")) / "atlas_photoshop"
        base.mkdir(parents=True, exist_ok=True)
        # A monotonic counter rather than a timestamp: two calls in the same
        # second must not share a sentinel, or the second would read the first
        # one's stale result and report success for a script that never ran.
        self._seq = getattr(self, "_seq", 0) + 1
        return base / f"atlas_{label}_{self._seq}{jsx.SENTINEL_SUFFIX}"

    def run(self, script_builder, *, label: str = "job", timeout_s=None,
            **kwargs) -> dict:
        """Build a script, execute it, and wait for its sentinel.

        ``script_builder`` is one of the pure generators in ``jsx``; it is
        always called with ``sentinel_path=``.
        """
        if self._app is None:
            raise PhotoshopBridgeError("not connected: call connect() first")

        sentinel = self._sentinel_path(label)
        if sentinel.exists():
            sentinel.unlink()
        script = script_builder(sentinel_path=str(sentinel), **kwargs)

        try:
            self._app.DoJavaScript(script)
        except Exception as exc:                      # noqa: BLE001 - COM error
            # A modal dialog in Photoshop blocks scripting entirely, and the
            # COM error for it is not self-explanatory. Say so.
            raise ScriptFailed(
                f"DoJavaScript failed for {label!r}: {exc}. If Photoshop has a "
                f"modal dialog open, scripting is blocked until it is "
                f"dismissed.") from exc

        return self._await_sentinel(sentinel, label,
                                    self.timeout_s if timeout_s is None
                                    else float(timeout_s))

    def _await_sentinel(self, sentinel: Path, label: str,
                        timeout_s: float) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if sentinel.exists():
                try:
                    payload = json.loads(sentinel.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    # Written but not yet flushed — keep waiting rather than
                    # calling a partial file a failure.
                    time.sleep(self.poll_s)
                    continue
                if not payload.get("ok", False):
                    raise ScriptFailed(
                        f"{label!r} reported: {payload.get('error', 'unknown')}")
                return payload
            time.sleep(self.poll_s)

        raise ScriptTimeout(
            f"{label!r} did not finish within {timeout_s:.0f}s (no sentinel at "
            f"{sentinel}). Generative fill is network-backed and can be slow or "
            f"can require a signed-in Firefly entitlement; a timeout here is "
            f"NOT the same as a failed gate, and must not be recorded as one.")

    # -- convenience --------------------------------------------------------

    def open_plate(self, path, *, as_ocio: bool = True) -> dict:
        return self.run(jsx.open_as_ocio, label="open", path=str(path),
                        as_ocio=as_ocio)

    def convert_to_ocio(self, *, working_space: str,
                        configuration: str | None = "Environment") -> dict:
        return self.run(jsx.convert_to_ocio, label="convert",
                        working_space=working_space, configuration=configuration)

    def generative_fill(self, *, prompt: str, mode: str = "inpaint",
                        timeout_s: float | None = None) -> dict:
        return self.run(jsx.synthetic_fill, label="fill", prompt=prompt,
                        mode=mode, timeout_s=timeout_s)

    def export(self, path) -> dict:
        """Export the active document as lossless 32-bit float TIFF.

        Not EXR: see jsx.export_float_tiff -- a .exr saveAs silently writes a
        PSD, and ProEXR's Action Manager class ID is not discoverable.
        """
        return self.run(jsx.export_float_tiff, label="export", path=str(path))

    def close_all(self) -> dict:
        return self.run(jsx.close_all, label="close")
