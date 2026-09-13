"""Publication is an explicit act, never a side effect of merging (atlas-nexus ADR-006).

The publish workflow used to fire on any push to `main` whose `version =` line changed, so merging a
branch that carried a version bump published it to the ComfyUI Registry whether or not anyone meant to
release. These tests pin the decoupling by reading the workflow file itself: no merge, push or pull
request can publish, and a deliberate publish can only ship the version `pyproject.toml` declares, from a
commit on `main`.
"""
from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "publish-comfyui-registry.yml"


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _triggers() -> str:
    """The top-level `on:` block, as text: everything between `on:` and `jobs:`."""
    text = _text()
    start = text.index("\non:")
    end = text.index("\njobs:")
    return text[start:end]


def test_no_merge_push_or_pull_request_can_publish():
    triggers = _triggers()
    for event in ("push:", "pull_request:", "pull_request_target:", "schedule:"):
        assert event not in triggers, f"publication must not be triggered by {event}"


def test_publication_is_a_published_release_or_a_manual_run():
    triggers = _triggers()
    assert "release:" in triggers and "published" in triggers
    assert "workflow_dispatch:" in triggers


def test_a_manual_run_must_name_the_version_it_publishes():
    triggers = _triggers()
    assert "inputs:" in triggers and "version:" in triggers and "required: true" in triggers


def test_the_requested_version_must_equal_pyproject():
    text = _text()
    assert "pyproject.toml" in text and "tomllib" in text
    assert '"$REQUESTED" != "$DECLARED"' in text, "a mismatch must refuse, not warn"


def test_only_a_commit_on_main_can_be_published():
    text = _text()
    assert "fetch-depth: 0" in text
    assert "merge-base --is-ancestor" in text and "origin/main" in text


def test_the_guards_run_before_the_publish_step():
    text = _text()
    assert text.index('"$REQUESTED" != "$DECLARED"') < text.index("Comfy-Org/publish-node-action")
    assert text.index("merge-base --is-ancestor") < text.index("Comfy-Org/publish-node-action")
