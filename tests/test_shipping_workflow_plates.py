"""A shipping workflow may not open red because its plate is missing.

Decision 2026-08-07: **no image assets ship in this repo.** The Atlas plate
pack is distributed from the project website instead. That makes ComfyUI's own
bundled ``example.png`` the only image guaranteed to exist on a fresh install,
so every shipping workflow starts there and queues green with nothing to
download.

This guard exists because the failure is silent and easy to reintroduce. The
export fan-out shipped pointing at ``oceancastle.jpg`` purely because that file
happened to sit in the author's local ComfyUI input folder — it was never
tracked, so the workflow was red on open for everyone else from the day it
landed, and nothing failed. Authoring against a live server makes this
specific mistake very easy to make: the plate resolves for you.

The rule is also the premise the v1 workflow cut was justified on. Two
workflows were dropped from the shipping set for loading red (a gated node, a
missing VLM); a shipping workflow that cannot find its plate is the same
failure wearing different clothes.
"""

import glob
import json
import os
import subprocess

import pytest

from conftest import is_local_workflow

ROOT = os.path.join(os.path.dirname(__file__), "..")
EXAMPLES_DIR = os.path.join(ROOT, "examples")

#: ComfyUI ships this in its own input folder, so it resolves on every install
#: without a download. It is the ONLY safe default while the repo ships no
#: images of its own.
BUNDLED_PLATE = "example.png"

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".dpx")


def _shipping_workflows():
    out = []
    for path in sorted(glob.glob(os.path.join(EXAMPLES_DIR, "*.json"))):
        if is_local_workflow(path):
            continue
        try:
            wf = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(wf, dict) and isinstance(wf.get("nodes"), list):
            out.append((os.path.basename(path), wf))
    return out


_WORKFLOWS = _shipping_workflows()


def test_there_are_shipping_workflows_to_check():
    """Guard the guard — an empty discovery would pass everything below."""
    assert _WORKFLOWS, "no shipping workflows discovered"


@pytest.mark.parametrize("name,wf", _WORKFLOWS, ids=[n for n, _ in _WORKFLOWS])
def test_every_load_image_uses_the_bundled_plate(name, wf):
    """Every ``LoadImage`` must name ``example.png``.

    If a workflow genuinely needs a real plate, the honest options are to ship
    that plate (reversing the no-images decision) or to keep the workflow in
    ``examples/local/`` — not to reference a file the user does not have.
    """
    offenders = [
        node["widgets_values"][0]
        for node in wf["nodes"]
        if node.get("type") == "LoadImage" and node.get("widgets_values")
        and node["widgets_values"][0] != BUNDLED_PLATE
    ]
    assert not offenders, (
        f"{name}: LoadImage references {offenders}, which this repo does not "
        f"ship — a fresh install opens this workflow with a red node. Use "
        f"{BUNDLED_PLATE!r}, or move the workflow to examples/local/.")


def test_the_repo_ships_no_plate_images():
    """The other half of the same decision, checked against GIT rather than the
    working tree: an untracked plate sitting in the author's checkout is
    exactly how the oceancastle reference survived review.

    Scoped to ``examples/``. Branding is a different kind of asset with a
    different reason to exist — ``assets/atlas_camera_icon.*`` is referenced by
    pyproject's ``Icon`` field and is what the ComfyUI registry renders, so it
    is not a download and must stay tracked. Only PLATES were moved to the
    website.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "examples"],
        cwd=ROOT, capture_output=True, text=True, check=False).stdout.split()
    images = [p for p in tracked if p.lower().endswith(_IMAGE_SUFFIXES)]
    assert not images, (
        "plate images are tracked under examples/, but the 2026-08-07 decision "
        "is that plates are distributed from the website instead:\n  "
        + "\n  ".join(images))


def test_the_bundled_plate_choice_is_recorded_in_the_generator():
    """The generator is the source of truth for these files, so the constraint
    has to live there too — a fix applied only to the JSON is overwritten by
    the next regeneration."""
    src = open(os.path.join(ROOT, "tools", "build_v1_shipping_workflows.py"),
               encoding="utf-8").read()
    assert '"image": "example.png"' in src
    for gone in ("ghosttown.jpg", "oceancastle.jpg", "spacehangar.jpg"):
        assert gone not in src, f"generator still references {gone}"
