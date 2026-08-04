"""Every shipped frontend extension must PARSE as an ES module.

Found live 2026-08-04: a patch put a real newline inside a JS string literal in
`atlas_blockout.js`, leaving it unterminated. The browser refused the whole
module, so `AtlasCamera.Blockout` never registered and the viewport node came
up with its widgets but no 3D canvas at all — a total loss of the viewport, not
a subtle glitch.

It shipped because the syntax check being used, plain ``node --check FILE.js``,
parses a `.js` file as CommonJS. The error sat past the point where a CommonJS
parse had already given up on the module syntax, so it reported OK. Copying to
`.mjs` forces the module grammar and catches it.

These files are served straight to the browser, so a parse error is fatal and
invisible to every Python test in the suite. This is the only thing standing
between a bad edit and a dead viewport.
"""

import os
import shutil
import subprocess

import pytest

WEB = os.path.join(os.path.dirname(__file__), "..", "atlas_camera", "comfy", "web")


def _extension_files():
    return sorted(f for f in os.listdir(WEB) if f.endswith(".js"))


def test_there_are_extension_files_to_check():
    """Guard the guard: a glob that silently matches nothing proves nothing."""
    assert len(_extension_files()) >= 5


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("name", _extension_files())
def test_extension_parses_as_an_es_module(name, tmp_path):
    source = os.path.join(WEB, name)
    # The .mjs extension is the point: it forces node's MODULE grammar, which
    # is what the browser uses. A .js copy would be parsed as CommonJS and can
    # miss real errors (see the module docstring).
    target = tmp_path / (name + ".mjs")
    target.write_bytes(open(source, "rb").read())

    result = subprocess.run(["node", "--check", str(target)],
                            capture_output=True, text=True)

    assert result.returncode == 0, (
        f"{name} does not parse as an ES module — the browser will refuse it "
        f"and the extension will not register:\n{result.stderr}")
