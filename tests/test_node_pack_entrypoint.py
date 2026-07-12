"""The clone-and-go ComfyUI entry point (repo-root __init__.py): loading the
repository root the way ComfyUI loads a custom_nodes folder must yield the
node mappings and a RELATIVE WEB_DIRECTORY (registry/Manager tooling assumes
relative paths). This is what `git clone <repo> custom_nodes/atlasCamera`
exercises — the dev symlink setup never loads this file.
"""

import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")  # nodes.py import chain expects it available

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_as_custom_node():
    spec = importlib.util.spec_from_file_location(
        "atlasCamera_clone_test", os.path.join(ROOT, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_repo_root_loads_like_a_custom_node():
    mod = _load_as_custom_node()
    assert len(mod.NODE_CLASS_MAPPINGS) > 50
    assert "AtlasBlockoutViewport" in mod.NODE_CLASS_MAPPINGS
    assert "AtlasAssessImage" in mod.NODE_DISPLAY_NAME_MAPPINGS
    # WEB_DIRECTORY must be relative and must exist relative to the root.
    assert not os.path.isabs(mod.WEB_DIRECTORY)
    web = os.path.join(ROOT, mod.WEB_DIRECTORY)
    assert os.path.isdir(web)
    assert os.path.isfile(os.path.join(web, "atlas_blockout.js"))
    assert os.path.isfile(os.path.join(web, "lib", "atlas-three.bundle.js"))


def test_comfy_subpackage_web_directory_is_relative():
    import atlas_camera.comfy as comfy_pkg
    assert comfy_pkg.WEB_DIRECTORY == "./web"
    web = os.path.join(os.path.dirname(comfy_pkg.__file__), comfy_pkg.WEB_DIRECTORY)
    assert os.path.isfile(os.path.join(web, "atlas_blockout.js"))


def test_package_discovery_cannot_swallow_runtime_dirs():
    """The beta-0.3 packaging audit found `include = ["atlas*"]` glob-matched
    atlas_exports/ and atlas_ui_projects/ — GITIGNORED runtime output dirs on
    any dev machine — and shipped 800+ files of local test exports inside the
    wheel. Exact package names + dotted-subpackage globs only: "atlas.*" can
    never match a sibling directory, because the dot means subpackage-of."""
    import re
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    m = re.search(r"\[tool\.setuptools\.packages\.find\].*?include\s*=\s*\[([^\]]*)\]",
                  text, re.DOTALL)
    assert m, "packages.find include list missing from pyproject.toml"
    entries = sorted(e.strip().strip('"') for e in m.group(1).split(",") if e.strip())
    assert entries == ["atlas", "atlas.*", "atlas_camera", "atlas_camera.*"], entries
