"""The Photoshop bridge's script generation and transport guards.

No Photoshop, no COM, no GPU: the JSX generators are pure functions, and the
transport takes an injectable ``dispatch`` so the guards can be exercised
against a fake application object.

The guards under test are the ones whose failure would be SILENT — driving the
wrong Photoshop, or a mangled path reported as a permissions problem.
"""
from __future__ import annotations

import json

import pytest

from atlas_camera.paint.photoshop import jsx
from atlas_camera.paint.photoshop.com_client import (DEFAULT_INSTALL_DIR,
                                                     PROGID_BETA,
                                                     PhotoshopClient,
                                                     PhotoshopNotFound,
                                                     ScriptFailed,
                                                     ScriptTimeout,
                                                     WrongPhotoshop)


# --- JSX generation ---------------------------------------------------------

def test_windows_paths_survive_into_js_string_literals():
    """A raw backslash in a JS literal is an ESCAPE, so an unquoted Windows
    path silently becomes a different path — and Photoshop then reports
    PERMISSION_DENIED, which reads as a permissions problem rather than the
    mangled path it actually is."""
    literal = jsx.js_string(r"C:\Users\miike\Desktop\atlas run\plate.exr")
    assert literal.startswith('"') and literal.endswith('"')
    # Round-tripping through a JSON parser is exactly how ExtendScript will
    # read it, so this asserts the path a script would actually receive.
    assert json.loads(literal) == r"C:\Users\miike\Desktop\atlas run\plate.exr"


def test_js_string_escapes_quotes_and_newlines():
    assert json.loads(jsx.js_string('he said "no"\nthen left')) == (
        'he said "no"\nthen left')


def test_probe_asks_for_the_install_path():
    """`app.path` is the field the whole wrong-app guard rests on."""
    script = jsx.probe(sentinel_path=r"C:\t\s.json")
    assert "app.path" in script
    assert "app.version" in script


def test_every_script_writes_its_sentinel_last():
    """The sentinel is what makes an async, network-backed fill observable.
    If a script could return without writing it, a timeout and a success would
    be indistinguishable."""
    for script in (
        jsx.probe(sentinel_path=r"C:\t\s.json"),
        jsx.open_as_ocio(path=r"C:\p.exr", sentinel_path=r"C:\t\s.json"),
        jsx.convert_to_ocio(working_space="ACEScg", sentinel_path=r"C:\t\s.json"),
        jsx.synthetic_fill(prompt="remove", sentinel_path=r"C:\t\s.json"),
        jsx.export_float_tiff(path=r"C:\o.tif", sentinel_path=r"C:\t\s.json"),
        jsx.close_all(sentinel_path=r"C:\t\s.json"),
    ):
        assert "__atlasWrite(__atlas.result)" in script
        assert "catch (e)" in script, "a thrown error must reach the sentinel"


def test_synthetic_fill_uses_the_inpaint_mode_by_default():
    """`inpaint` is the enum value Affinity had no equivalent of, and the whole
    reason Photoshop is worth bridging to."""
    script = jsx.synthetic_fill(prompt="remove the boiler",
                                sentinel_path=r"C:\t\s.json")
    assert "syntheticFill" in script
    assert "inpaint" in script
    assert "remove the boiler" in script


@pytest.mark.parametrize("mode", ["inpaint", "variation", "synthesize"])
def test_synthetic_fill_accepts_every_documented_mode(mode):
    assert mode in jsx.synthetic_fill(prompt="x", sentinel_path=r"C:\t\s.json",
                                      mode=mode)


def test_synthetic_fill_rejects_an_invented_mode():
    """The enum came from the shipped action dictionary; a typo must fail here
    rather than inside Photoshop where the error is invisible."""
    with pytest.raises(ValueError, match="unknown syntheticFillMode"):
        jsx.synthetic_fill(prompt="x", sentinel_path=r"C:\t\s.json",
                           mode="inpainting")


def test_open_carries_the_open_as_ocio_key():
    assert "openAsOCIO" in jsx.open_as_ocio(path=r"C:\p.exr",
                                            sentinel_path=r"C:\t\s.json")


def test_convert_to_ocio_can_select_the_environment_domain():
    """The Environment domain is what makes Photoshop read $OCIO — i.e. what
    puts it on the SAME config as Atlas rather than one that merely shares
    colourspace names."""
    script = jsx.convert_to_ocio(working_space="ACEScg", configuration="Environment",
                                 sentinel_path=r"C:\t\s.json")
    assert "convertToOCIO" in script
    assert "Environment" in script
    assert "working_space" in script


# --- transport guards -------------------------------------------------------

class _FakeApp:
    def __init__(self, path=DEFAULT_INSTALL_DIR, ok=True, error=None,
                 write_sentinel=True):
        self.path = path
        self.ok = ok
        self.error = error
        self.write_sentinel = write_sentinel
        self.scripts = []

    def DoJavaScript(self, script):     # noqa: N802 - COM naming
        self.scripts.append(script)
        if not self.write_sentinel:
            return "{}"
        # Mimic ExtendScript: parse the sentinel path back out of the script,
        # which also asserts the script really carries one.
        marker = "var f = new File("
        idx = script.index(marker) + len(marker)
        path = json.loads(script[idx:script.index(")", idx)])
        payload = {"ok": self.ok, "path": self.path, "version": "27.0 (beta)",
                   "name": "Adobe Photoshop (Beta)", "documents": 0}
        if self.error:
            payload["error"] = self.error
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return json.dumps(payload)


def _client(tmp_path, **kwargs):
    return PhotoshopClient(scratch_dir=tmp_path, timeout_s=2.0, poll_s=0.01,
                           **kwargs)


def test_connect_targets_the_beta_progid_verbatim(tmp_path):
    """Photoshop 2026 may also be installed; the Beta-specific ProgID is the
    only unambiguous target."""
    seen = {}

    def dispatch(progid):
        seen["progid"] = progid
        return _FakeApp()

    _client(tmp_path).connect(dispatch=dispatch)
    assert seen["progid"] == PROGID_BETA == "Photoshop.Application.BETA"


def test_connect_refuses_a_different_photoshop_install(tmp_path):
    """The wrong-app failure is the dangerous one: everything would appear to
    work and the results would be attributed to the wrong application."""
    wrong = _FakeApp(path=r"C:\Program Files\Adobe\Adobe Photoshop 2026")
    with pytest.raises(WrongPhotoshop, match="Refusing to continue"):
        _client(tmp_path).connect(dispatch=lambda progid: wrong)


@pytest.mark.parametrize("reported", [
    DEFAULT_INSTALL_DIR,
    DEFAULT_INSTALL_DIR.lower() + "\\",
    DEFAULT_INSTALL_DIR.replace("\\", "/"),
    # ExtendScript's Folder URI, which is what String(app.path) actually
    # returns -- this exact form made the guard reject the very install it had
    # been asked for on the first live run.
    "/c/Program%20Files/Adobe/Adobe%20Photoshop%20(Beta)",
])
def test_connect_accepts_the_expected_install_however_it_is_spelled(
        tmp_path, reported):
    """Trailing separators, case, slash direction and ExtendScript's percent-
    encoded Folder URI are all the SAME install; none is a different app."""
    app = _FakeApp(path=reported)
    info = _client(tmp_path).connect(dispatch=lambda progid: app)
    assert info["version"]


def test_connect_reports_a_missing_photoshop_actionably(tmp_path):
    def dispatch(progid):
        raise OSError("Invalid class string")

    with pytest.raises(PhotoshopNotFound, match="Adobe Photoshop \\(Beta\\)"):
        _client(tmp_path).connect(dispatch=dispatch)


def test_a_script_error_is_raised_not_silently_passed(tmp_path):
    app = _FakeApp(ok=False, error="Error: NOT_IMPLEMENTED")
    client = _client(tmp_path)
    with pytest.raises(ScriptFailed, match="NOT_IMPLEMENTED"):
        client.connect(dispatch=lambda progid: app)


def test_a_missing_sentinel_times_out_distinctly_from_a_failure(tmp_path):
    """A timeout, a script error and a failed gate are three different
    problems. Generative fill is network-backed and may simply be slow, or may
    need a Firefly entitlement — neither is a gate result."""
    app = _FakeApp(write_sentinel=False)
    client = _client(tmp_path)
    with pytest.raises(ScriptTimeout, match="NOT the same as a failed gate"):
        client.connect(dispatch=lambda progid: app)


def test_each_run_uses_a_fresh_sentinel(tmp_path):
    """Two calls in the same second must not share a sentinel: the second would
    read the first one's stale result and report success for a script that
    never ran."""
    client = _client(tmp_path)
    first = client._sentinel_path("fill")
    second = client._sentinel_path("fill")
    assert first != second


def test_export_uses_lossless_float_tiff_not_exr():
    """EXR is the obvious choice and both routes to it are traps, measured
    2026-08-22: doc.saveAs with a .exr extension reports success and silently
    writes a PSD, and ProEXR's Action Manager class ID is not discoverable
    (every plausible identifier returned "may not be available in this version
    of Photoshop"). An uncompressed 32-bit float TIFF is lossless and works."""
    script = jsx.export_float_tiff(path=r"C:\out.tif", sentinel_path=r"C:	\s.json")
    assert "TiffSaveOptions" in script
    assert "TIFFEncoding.NONE" in script, "the exchange file gets GATED; no codec"
    assert ".exr" not in script
    assert not hasattr(jsx, "export_exr"), (
        "export_exr wrote save-for-web output and could never produce an EXR; "
        "it must not come back")


def test_open_as_ocio_documents_the_aces2065_assumption():
    """Photoshop does not read the EXR colourspace tag: it ASSUMES ACES2065-1
    and converts into the working space. Feeding it Atlas's Rec.709 sidecar
    mis-reads the primaries with no error, so the docstring carries the
    warning where someone wiring this will actually read it."""
    assert "ACES2065-1" in (jsx.open_as_ocio.__doc__ or "")
    assert "Rec.709" in (jsx.open_as_ocio.__doc__ or "")
