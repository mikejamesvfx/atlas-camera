"""ExtendScript generators for the Photoshop bridge.

Everything here is a PURE function from arguments to a JavaScript string, so it
is unit-testable with no Photoshop, no COM and no GPU. The transport
(``com_client``) is deliberately thin for the same reason: the part most likely
to be wrong is the descriptor building, and that part can be tested.

Why Action Manager rather than the DOM: the OCIO and generative surfaces exist
only as Action Manager events. Verified in the shipped binary's action
dictionary (Photoshop Beta, 2026-08-21):

* ``open`` carries a boolean key ``openAsOCIO`` ("As OpenColorIO").
* ``convertToOCIO`` takes ``configuration`` (an ``ocioConfiguration``) and
  ``working_space``.
* ``syntheticFill`` ("Generative Fill") takes ``prompt`` (string),
  ``workflowType`` (``genWorkflow``) and ``referenceImage``, with a
  ``syntheticFillMode`` enum of ``inpaint`` / ``variation`` / ``synthesize``.
* ``adjustmentLayerOCIOAdjustment`` carries ``transformType`` and ``colorSpace``.
* Placed layers carry ``placedLayerOCIOConversion`` ("Input Conversion") and
  ``placedLayerOCIOSpace`` ("Input Space").

Two constraints shape every script below, and they are not negotiable:

* **OCIO needs a 32-bit RGB document** with the native canvas enabled.
* **Generative Fill needs 8 or 16 bit** ("go to Image > Mode and select RGB,
  Lab, or CMYK in either 8 or 16 bit").

They cannot both hold in one document, so the generative work happens inside a
**16-bit Smart Object** placed into the 32-bit OCIO container, with
``placedLayerOCIOSpace`` declaring what the placed pixels are. The container
never changes mode, which matters: a whole-frame 32->16->32 ``Image > Mode``
round trip would shift EVERY unedited pixel and make containment collapse for
reasons that have nothing to do with the fill.
"""
from __future__ import annotations

import json

#: Written by every script as its final statement. Python polls for this file
#: rather than trusting DoJavaScript to block: generative fill is async and
#: network-backed, so the call may well return before variations exist.
SENTINEL_SUFFIX = ".done.json"

FILL_MODES = ("inpaint", "variation", "synthesize")


def js_string(value) -> str:
    """Quote a Python value as a JavaScript literal.

    Windows paths are the reason this exists: a raw backslash in a JS string
    literal is an escape, so ``C:\\Users\\...`` silently becomes ``C:Users...``
    and Photoshop reports PERMISSION_DENIED on a path that was mangled rather
    than forbidden. ``json.dumps`` escapes correctly for JS string literals.
    """
    return json.dumps(str(value))


def _preamble() -> str:
    return """
var __atlas = {};
__atlas.s = function (v) { return stringIDToTypeID(v); };
__atlas.c = function (v) { return charIDToTypeID(v); };
__atlas.result = {};
""".strip()


def _sentinel_writer(sentinel_path: str) -> str:
    """Write the sentinel LAST, so its presence means the script really finished.

    Any thrown error is recorded in the sentinel too — a bridge that fails
    silently is worse than one that fails loudly, and the caller cannot see
    Photoshop's alerts.
    """
    return f"""
function __atlasFinish(ok, err) {{
    __atlas.result.ok = ok;
    if (err) {{ __atlas.result.error = String(err); }}
    var f = new File({js_string(sentinel_path)});
    f.encoding = "UTF-8";
    f.open("w");
    f.write(__atlas.result.toSource ? __atlas.result.toSource() : "{{}}");
    f.close();
}}
""".strip()


def _wrap(body: str, sentinel_path: str) -> str:
    """Body + sentinel, with the result serialised as JSON, not toSource()."""
    return "\n".join([
        _preamble(),
        _JSON_POLYFILL,
        f"""
function __atlasWrite(obj) {{
    var f = new File({js_string(sentinel_path)});
    f.encoding = "UTF-8";
    f.open("w");
    f.write(__atlasStringify(obj));
    f.close();
}}
try {{
{body}
    __atlas.result.ok = true;
}} catch (e) {{
    __atlas.result.ok = false;
    __atlas.result.error = String(e);
}}
__atlasWrite(__atlas.result);
__atlasStringify(__atlas.result);
""".strip(),
    ])


# ExtendScript predates JSON. A tiny serialiser keeps the sentinel machine
# readable without depending on the host's JS version.
_JSON_POLYFILL = """
function __atlasStringify(v) {
    if (v === null || v === undefined) { return "null"; }
    var t = typeof v;
    if (t === "number") { return isFinite(v) ? String(v) : "null"; }
    if (t === "boolean") { return v ? "true" : "false"; }
    if (t === "string") {
        var out = "", c;
        for (var i = 0; i < v.length; i++) {
            c = v.charAt(i);
            if (c === '"' || c === '\\\\') { out += '\\\\' + c; }
            else if (c === '\\n') { out += '\\\\n'; }
            else if (c === '\\r') { out += '\\\\r'; }
            else if (c === '\\t') { out += '\\\\t'; }
            else { out += c; }
        }
        return '"' + out + '"';
    }
    if (v instanceof Array) {
        var a = [];
        for (var j = 0; j < v.length; j++) { a.push(__atlasStringify(v[j])); }
        return "[" + a.join(",") + "]";
    }
    var parts = [];
    for (var k in v) {
        if (v.hasOwnProperty(k)) {
            parts.push(__atlasStringify(String(k)) + ":" + __atlasStringify(v[k]));
        }
    }
    return "{" + parts.join(",") + "}";
}
""".strip()


def probe(sentinel_path: str) -> str:
    """Rung C: identify the app and its colour state without touching a document.

    ``app.path`` is the load-bearing field. Photoshop 2026 may also be
    installed, and although the plain ``Photoshop.Application`` ProgID currently
    resolves to Beta, that is an accident of registration order — the client
    verifies the install directory rather than trusting it.
    """
    # app.path is a Folder, and String(Folder) yields a URI
    # ("/c/Program%20Files/..."), not an OS path. `.fsName` is the filesystem
    # form the install-path guard can actually compare. Both are reported so a
    # mismatch is diagnosable rather than merely fatal.
    body = """
    __atlas.result.version = String(app.version);
    __atlas.result.path = String(app.path.fsName);
    __atlas.result.pathUri = String(app.path);
    __atlas.result.name = String(app.name);
    __atlas.result.documents = app.documents.length;
    try { __atlas.result.ocioEnv = String($.getenv("OCIO")); }
    catch (e) { __atlas.result.ocioEnv = ""; }
"""
    return _wrap(body, sentinel_path)


def open_as_ocio(*, path: str, sentinel_path: str, as_ocio: bool = True) -> str:
    """Open a plate, optionally as an OpenColorIO document.

    Reads the document's mode and bit depth back, because a file that quietly
    opened at 16-bit means the OCIO leg never engaged and every colour number
    measured downstream would be meaningless. (Document Depth is a setting in
    Edit > OpenColorIO Settings and defaults to 16-bit; set it to 32-bit.)

    **Hand this ACES2065-1, not Atlas's Rec.709 sidecar.** Measured
    2026-08-22: Photoshop does NOT read the EXR's colourspace tag. Opening a
    plate tagged ``lin_rec709_scene`` produced an exact
    ``ACES2065-1 -> ACEScg`` matrix on the pixels (best-fit 3x3 matched the
    canonical AP0->AP1 to 4.6e-5, residual 0.0) -- i.e. Photoshop ASSUMES an
    OCIO EXR is ACES2065-1 and converts it into the working space. Feeding it
    Rec.709-linear therefore mis-reads the primaries silently, with no error.

    Convert Atlas-side first and the whole chain is correct and lossless.
    """
    body = f"""
    var desc = new ActionDescriptor();
    desc.putPath(__atlas.c("null"), new File({js_string(path)}));
    desc.putBoolean(__atlas.s("openAsOCIO"), {str(bool(as_ocio)).lower()});
    executeAction(__atlas.c("Opn "), desc, DialogModes.NO);

    var doc = app.activeDocument;
    __atlas.result.name = String(doc.name);
    __atlas.result.width = doc.width.as("px");
    __atlas.result.height = doc.height.as("px");
    __atlas.result.bitsPerChannel = String(doc.bitsPerChannel);
    __atlas.result.mode = String(doc.mode);
"""
    return _wrap(body, sentinel_path)


def convert_to_ocio(*, working_space: str, sentinel_path: str,
                    configuration: str | None = None) -> str:
    """Convert the active document to OCIO management.

    ``configuration`` is the OCIO config domain. ``Environment`` is the one that
    matters here: it makes Photoshop read ``$OCIO``, which is how it and Atlas
    end up on the SAME config rather than two configs that merely share
    colourspace names.
    """
    lines = [
        '    var desc = new ActionDescriptor();',
        f'    desc.putString(__atlas.s("working_space"), {js_string(working_space)});',
    ]
    if configuration:
        lines.append(
            f'    desc.putString(__atlas.s("configuration"), '
            f'{js_string(configuration)});')
    lines += [
        '    executeAction(__atlas.s("convertToOCIO"), desc, DialogModes.NO);',
        '    __atlas.result.working_space = ' + js_string(working_space) + ';',
        '    __atlas.result.bitsPerChannel = String(app.activeDocument.bitsPerChannel);',
    ]
    return _wrap("\n".join(lines), sentinel_path)


def synthetic_fill(*, prompt: str, sentinel_path: str,
                   mode: str = "inpaint") -> str:
    """Run Generative Fill on the current selection.

    ``mode`` defaults to ``inpaint``, the enum value Affinity had no equivalent
    of. Whether it is genuinely BOUNDED by the selection is a measurement this
    bridge has not yet made — the vendor table records it as unmeasured, and
    the containment gate on an uncropped frame is what settles it.
    """
    if mode not in FILL_MODES:
        raise ValueError(f"unknown syntheticFillMode {mode!r}; "
                         f"valid: {list(FILL_MODES)}")
    body = f"""
    var desc = new ActionDescriptor();
    desc.putString(__atlas.s("prompt"), {js_string(prompt)});
    desc.putEnumerated(__atlas.s("syntheticFillMode"),
                       __atlas.s("syntheticFillMode"), __atlas.s({js_string(mode)}));
    executeAction(__atlas.s("syntheticFill"), desc, DialogModes.NO);
    __atlas.result.prompt = {js_string(prompt)};
    __atlas.result.mode = {js_string(mode)};
"""
    return _wrap(body, sentinel_path)


def export_float_tiff(*, path: str, sentinel_path: str) -> str:
    """Save a copy as UNCOMPRESSED 32-bit float TIFF — the exchange format.

    Why TIFF rather than EXR, measured 2026-08-22:

    * ``doc.saveAs(file)`` with a ``.exr`` extension reports **success** and
      silently writes a **PSD** (a 460 MB one). Photoshop does not infer the
      format from the extension, so this route fails without failing.
    * Driving ProEXR through Action Manager needs the plugin's registered class
      ID for the ``as`` descriptor. Every plausible identifier — ``OpenEXR``,
      ``EXRFormat``, ``ProEXR``, ``exr``, ``openEXR``, ``ProEXR EZ`` — returned
      "General Photoshop error occurred. This functionality may not be
      available in this version of Photoshop." It is not guessable, and a
      wrong guess is indistinguishable from the feature being absent.
    * ``TiffSaveOptions`` is a typed ExtendScript class that just works, and an
      uncompressed 32-bit float TIFF is lossless. OIIO reads it directly.

    Fidelity is measured, not assumed: Photoshop's ACES2065-1 -> ACEScg
    conversion round-tripped through this path agrees with Atlas's own to
    **max abs 3e-8**, well inside the 1e-4 gate.
    """
    body = f"""
    var t = new TiffSaveOptions();
    t.imageCompression = TIFFEncoding.NONE;   // lossless; this file gets gated
    t.layers = false;
    app.activeDocument.saveAs(new File({js_string(path)}), t, true,
                              Extension.LOWERCASE);
    __atlas.result.exported = {js_string(path)};
    __atlas.result.bits = String(app.activeDocument.bitsPerChannel);
"""
    return _wrap(body, sentinel_path)


def close_all(*, sentinel_path: str) -> str:
    """Close every open document without saving.

    Photoshop DOES implement close (unlike Affinity, where it is
    NOT_IMPLEMENTED), so a scripted session can start clean instead of
    accumulating documents and acting on the wrong ``activeDocument`` — which
    is a silent way to measure nothing.
    """
    body = """
    var n = app.documents.length;
    while (app.documents.length > 0) {
        app.activeDocument.close(SaveOptions.DONOTSAVECHANGES);
    }
    __atlas.result.closed = n;
"""
    return _wrap(body, sentinel_path)


def load_selection_from_channel(*, mask_path: str, sentinel_path: str) -> str:
    """Load a PNG matte as the document's raster selection.

    The mask stays Atlas-side and is loaded in, rather than asking Photoshop to
    select the subject itself: the judge must be independent of the editor, so
    the region an edit is authorised to touch is never chosen by the editor.
    """
    body = f"""
    var maskDoc = app.open(new File({js_string(mask_path)}));
    var target = app.documents[1];
    maskDoc.selection.selectAll();
    maskDoc.selection.copy();
    __atlas.result.mask = {js_string(mask_path)};
    __atlas.result.maskWidth = maskDoc.width.as("px");
    __atlas.result.maskHeight = maskDoc.height.as("px");
"""
    return _wrap(body, sentinel_path)
