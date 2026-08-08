"""Shipped docs may not point at things that do not exist.

Both guards here exist because of the same recurring failure: a doc states a
fact that WAS true, the code moves, and nothing fails. The 2026-07-31 workflow
replacement deleted 21 files and left 14 live references to
``atlas_camera_staged_master_workflow.json`` across INSTALL, USER_GUIDE,
ECOSYSTEM_GUIDE and NODE_CATALOG — instructions telling a reader to open a file
that is not there. Nobody noticed for a week, because prose has no test.

Scope is deliberately narrow. These check facts that are MECHANICALLY
checkable — does this filename exist, does this number match the registry —
and say nothing about whether the prose around them is any good.
"""

import os
import re

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
EXAMPLES_DIR = os.path.join(ROOT, "examples")

#: Docs a user or reviewer actually reads. CHANGELOG is EXCLUDED on purpose: a
#: changelog's job is to name what existed at the time, so a reference to a
#: since-deleted workflow is correct there and would be wrong to "fix".
#: docs/dev/ is excluded because it is gitignored working material, not shipped.
_SHIPPED_DOCS = ("README.md", "INSTALL.md", "THIRD_PARTY.md")
_SHIPPED_DOC_DIRS = ("docs",)
#: DESIGN_RULES joins CHANGELOG in the workflow-existence exclusion, for the
#: same reason: it cites workflows as PROVENANCE ("found live in X"), not as
#: instructions to go and open one. CLAUDE.md already records that its
#: citations resolve in a working checkout rather than a fresh clone. Rewriting
#: those to name surviving files would falsify the record of where a rule came
#: from, which is the one thing that document is for.
_PROVENANCE_DOCS = {"CHANGELOG.md", "DESIGN_RULES.md"}
_EXCLUDED_DOCS = {"CHANGELOG.md"}
_EXCLUDED_DIRS = {"dev", "artifacts", "superpowers", "showcase_pdfs"}

_WORKFLOW_RE = re.compile(r"(atlas[A-Za-z0-9_]*workflow[A-Za-z0-9_]*)\.json")


def _shipped_docs():
    out = []
    for name in _SHIPPED_DOCS:
        path = os.path.join(ROOT, name)
        if os.path.isfile(path):
            out.append(path)
    for directory in _SHIPPED_DOC_DIRS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, directory)):
            dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIRS]
            for filename in filenames:
                if filename.endswith(".md") and filename not in _EXCLUDED_DOCS:
                    out.append(os.path.join(dirpath, filename))
    return sorted(out)


def _shipped_workflow_names():
    return {f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".json")}


def test_there_are_shipped_docs_to_check():
    """Guard the guard. A walk that silently matches nothing proves nothing —
    the same trap that made the workflow suites pass on an empty set when
    run from a temp worktree (see conftest.is_local_workflow)."""
    docs = _shipped_docs()
    assert len(docs) >= 8, f"only found {len(docs)} shipped docs — walk is broken"
    assert any(d.endswith("NODE_CATALOG.md") for d in docs)
    assert any(d.endswith("README.md") for d in docs)


def test_docs_reference_existing_workflows():
    """Every ``atlas_*workflow*.json`` named in a shipped doc must exist.

    This is the test that would have caught the 2026-07-31 breakage on the day
    it happened, and the one that makes the v1 workflow cut safe to repeat:
    drop a workflow and the docs that still point at it fail immediately,
    naming the file and the line.
    """
    shipped = _shipped_workflow_names()
    dead = []
    for path in _shipped_docs():
        if os.path.basename(path) in _PROVENANCE_DOCS:
            continue
        text = open(path, encoding="utf-8").read()
        for line_no, line in enumerate(text.splitlines(), 1):
            for match in _WORKFLOW_RE.findall(line):
                name = f"{match}.json"
                if name in shipped:
                    continue
                # examples/local/ is gitignored artist material; a doc may
                # legitimately mention one as a local-only file.
                if "examples/local/" in line or "local/" in line:
                    continue
                rel = os.path.relpath(path, ROOT).replace("\\", "/")
                dead.append(f"{rel}:{line_no}: {name}")
    assert not dead, (
        "shipped docs reference workflow files that do not exist in examples/ "
        "— either restore the file or update the doc:\n  " + "\n  ".join(dead))


# --- node counts -------------------------------------------------------------
#
# Reality on 2026-08-07 was 90 standard + 6 experimental + 2 legacy + 2 iOS.
# Docs variously claimed 56 (three docs), 68 (two docs), 89, 91 and 99. Every
# one of those was correct when written. A number typed into prose goes stale
# the next time a node lands, so the durable fix is to let the registry be the
# single source and fail when a doc disagrees with it.


def _registry_counts():
    os.environ.pop("ATLAS_EXPERIMENTAL", None)
    os.environ.pop("ATLAS_IOS", None)
    from atlas_camera.comfy import node_registry as registry
    return {
        "standard": len(registry.NODE_CLASS_MAPPINGS),
        "experimental": len(registry.EXPERIMENTAL_NODE_CLASS_MAPPINGS),
        "legacy": len(registry.LEGACY_NODE_CLASS_MAPPINGS),
        "ios": len(registry.IOS_NODE_CLASS_MAPPINGS),
    }


#: Docs allowed to state a node count, and the phrasing each must use. Keeping
#: this list SHORT is the point — every entry is a number someone has to keep
#: true, so the answer to "where do I put the count?" should almost always be
#: "nowhere, link to NODE_CATALOG".
_COUNT_CLAIMS = (
    ("docs/NODE_CATALOG.md",
     "{standard} standard + {experimental} experimental + {legacy} legacy + {ios} iOS = {total} registered"),
    ("docs/NODE_CATALOG.md",
     "the {total}\n\nnode classes ({standard} standard + {experimental} experimental + {legacy} legacy + {ios} iOS)"),
)


@pytest.mark.parametrize("rel_path,template", _COUNT_CLAIMS,
                         ids=[f"{p}::{t[:28]}" for p, t in _COUNT_CLAIMS])
def test_documented_node_counts_match_the_registry(rel_path, template):
    counts = _registry_counts()
    counts["total"] = sum(counts.values())
    expected = template.format(**counts)
    text = open(os.path.join(ROOT, rel_path), encoding="utf-8").read()
    assert expected in text, (
        f"{rel_path} no longer states the real node counts. The registry says "
        f"{counts['standard']} standard + {counts['experimental']} experimental "
        f"+ {counts['legacy']} legacy + {counts['ios']} iOS = {counts['total']}. "
        f"Expected to find:\n  {expected!r}")


def test_no_other_shipped_doc_hardcodes_a_node_count():
    """Counts live in NODE_CATALOG and nowhere else.

    Any other doc that types one is a future lie with no test behind it. If a
    doc genuinely needs the number, add it to _COUNT_CLAIMS above so it is
    checked against the registry — do not just write it down.
    """
    allowed = {os.path.normpath(os.path.join(ROOT, p)) for p, _ in _COUNT_CLAIMS}
    pattern = re.compile(r"\b(\d{2,3})\s+(?:standard\s+)?nodes?\b", re.IGNORECASE)
    offenders = []
    for path in _shipped_docs():
        if os.path.normpath(path) in allowed:
            continue
        for line_no, line in enumerate(
                open(path, encoding="utf-8").read().splitlines(), 1):
            for match in pattern.finditer(line):
                rel = os.path.relpath(path, ROOT).replace("\\", "/")
                offenders.append(f"{rel}:{line_no}: '{match.group(0)}'")
    assert not offenders, (
        "shipped docs hardcode a node count. Counts go stale the next time a "
        "node lands — cite NODE_CATALOG instead, or add the doc to "
        "_COUNT_CLAIMS so the number is checked against the registry:\n  "
        + "\n  ".join(offenders))


# --- version drift -----------------------------------------------------------
#
# On 2026-08-07 pyproject said 0.8.1, atlas_camera.__version__ said 0.8.1, and
# CHANGELOG's newest entry was a dated 0.8.2 — a release documented but never
# versioned. Nothing failed, because no test compared them.
#
# Note pyproject's version is ALSO the registry publish trigger: the publish
# workflow fires on a push to main that touches pyproject.toml. So a bump is a
# release action, not a tidy-up, and these tests are deliberately written so
# that keeping the version put is a valid way to be consistent.

import re as _re


def _pyproject_version():
    text = open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read()
    match = _re.search(r'^version\s*=\s*"([^"]+)"', text, _re.MULTILINE)
    assert match, "pyproject.toml has no version"
    return match.group(1)


def test_the_package_version_matches_pyproject():
    """A wheel built from pyproject and the running package must not disagree
    about what they are."""
    import atlas_camera
    assert atlas_camera.__version__ == _pyproject_version(), (
        f"atlas_camera.__version__ is {atlas_camera.__version__!r} but "
        f"pyproject says {_pyproject_version()!r}")


def test_the_changelog_does_not_claim_an_unreleased_version_as_released():
    """CHANGELOG's newest entry must be the shipped version, or be explicitly
    marked unreleased.

    A dated heading is a claim that the version went out. When the newest one
    runs ahead of pyproject, the changelog is describing a release that does
    not exist — which is what happened with 0.8.2. Either bump (a release
    action, and it publishes) or say 'unreleased'.
    """
    text = open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8").read()
    headings = _re.findall(r"^##\s+(.+)$", text, _re.MULTILINE)
    assert headings, "CHANGELOG has no version headings"
    newest = headings[0].strip()
    version = _pyproject_version()
    if newest.lower().startswith("unreleased") or "unreleased" in newest.lower():
        return                                    # explicitly pending — honest
    assert newest.startswith(version), (
        f"CHANGELOG's newest entry is {newest!r} but the shipped version is "
        f"{version!r}. Either bump pyproject (a RELEASE — note this publishes "
        f"to the ComfyUI registry) or mark the entry unreleased.")


def test_readme_states_the_three_tier_dependency_contract():
    """"The core is pure NumPy with zero required dependencies" read as a
    contradiction in a clean environment: the numerical solver lazily
    REQUIRES NumPy and raises with an install hint when it is absent
    (deep-research report, 2026-08-08). The honest contract has three tiers —
    schema/JSON/export = dependency-free; numerical camera recovery = NumPy;
    automatic line detection = NumPy + OpenCV — and the README must state it
    rather than the old absolute."""
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    assert "zero required dependencies" not in readme, (
        "the old absolute claim is back — the solver requires NumPy")
    for phrase in ("dependency-free", "NumPy", "OpenCV"):
        assert phrase in readme, f"three-tier dependency contract lost: {phrase}"
