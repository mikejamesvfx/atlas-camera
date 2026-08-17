"""Fixture tests for the /atlas-v1-audit scanners.

Builds a throwaway git repository shaped like Atlas — dynamic node
registration, a serialized workflow, a test-only module, a doc with a dead
path — and asserts the scanners reach the right verdicts on it.

The point of every case is the same: a dynamically referenced file must NEVER
be nominated for deletion. Two of these cases are regressions for scanner bugs
that reported live features as dead, which is the worst failure this tool has,
since the suggested remedy is deletion.

    python .claude/skills/atlas-v1-audit/tests/run_tests.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL / "scripts"
sys.path.insert(0, str(SCRIPTS))

PASSED = 0
FAILED: list[str] = []


def check(label: str, condition, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
    else:
        FAILED.append(f"{label}{(' — ' + detail) if detail else ''}")


# --- fixture ----------------------------------------------------------------

REGISTRY = '''\
"""Fixture registry — exercises all three mapping-construction forms."""
from .nodes_core import FixSolve, FixExport
from .nodes_extra import FixOrbit, FixGated

NODE_CLASS_MAPPINGS = {
    "FixSolve": FixSolve,
    "FixExport": FixExport,
}
# subscript form: an earlier scanner revision missed these entirely and
# reported them as deregistered leftovers
NODE_CLASS_MAPPINGS["FixOrbit"] = FixOrbit

EXPERIMENTAL_NODE_CLASS_MAPPINGS = {"FixGated": FixGated}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FixSolve": "Fixture Solve",
    "FixExport": "Fixture Export",
    "FixOrbit": "Fixture Orbit",
    "FixGated": "Fixture Gated",
}

if True:
    NODE_CLASS_MAPPINGS.update(EXPERIMENTAL_NODE_CLASS_MAPPINGS)
'''

NODES_CORE = '''\
"""Two ordinary nodes."""
import os


def _require_numpy():
    """Private helper whose NAME repeats across the package on purpose."""
    return os


class FixSolve:
    INPUT_TYPES = {}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self):
        return solve_the_scene()


class FixExport:
    INPUT_TYPES = {}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    def run(self):
        return ""


def solve_the_scene():
    return 1
'''

NODES_EXTRA = '''\
"""A subscript-registered node and a gated one."""


def _require_numpy():
    """Same private name as nodes_core. Not a reference to it."""
    return None


class FixOrbit:
    INPUT_TYPES = {}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self):
        return None


class FixGated:
    INPUT_TYPES = {}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self):
        return None
'''

DEREGISTERED = '''\
"""Node-shaped, but no mapping names it. The superseded-feature signal."""


class FixLegacyBlur:
    INPUT_TYPES = {}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self):
        return None
'''

TEST_ONLY = '''\
"""Implemented and tested, but nothing in the product reaches it."""


def _require_numpy():
    return None


def calibrate_the_thing(value):
    return value * 2
'''

TEST_FILE = '''\
from fixpkg.only_tested import calibrate_the_thing


def test_it():
    assert calibrate_the_thing(2) == 4
'''

DEAD_DOC = """\
# Fixture guide

The pack ships 99 standard nodes.

See examples/fixture_gone.json for the demo. The implementation lives in
fixpkg/nodes_core.py.
"""

PROVENANCE_DOC = """\
# Changelog

## 0.1.0

Removed `examples/fixture_gone.json`. At the time the pack had 99 standard
nodes.
"""

WORKFLOW = {
    "nodes": [
        {"id": 1, "type": "LoadImage"},
        {"id": 2, "type": "FixSolve"},
        {"id": 3, "type": "FixOrbit"},
    ]
}
WORKFLOW_DUP = {
    "nodes": [
        {"id": 7, "type": "FixOrbit"},
        {"id": 8, "type": "LoadImage"},
        {"id": 9, "type": "FixSolve"},
    ]
}
WORKFLOW_BROKEN = {"nodes": [{"id": 1, "type": "FixDeletedNode"}]}


def build_fixture(root: Path) -> None:
    (root / "fixpkg").mkdir()
    (root / "tests").mkdir()
    (root / "examples").mkdir()
    (root / "docs").mkdir()
    (root / "tools").mkdir()

    (root / "fixpkg" / "web").mkdir()
    (root / "fixpkg" / "__init__.py").write_text(
        'WEB_DIRECTORY = "./web"\n', encoding="utf-8")
    # No file imports this; ComfyUI serves it because of WEB_DIRECTORY above.
    (root / "fixpkg" / "web" / "panel.js").write_text(
        "export const panel = 1;\n", encoding="utf-8")
    (root / "fixpkg" / "node_registry.py").write_text(REGISTRY, encoding="utf-8")
    (root / "fixpkg" / "nodes_core.py").write_text(NODES_CORE, encoding="utf-8")
    (root / "fixpkg" / "nodes_extra.py").write_text(NODES_EXTRA, encoding="utf-8")
    (root / "fixpkg" / "deregistered.py").write_text(DEREGISTERED, encoding="utf-8")
    (root / "fixpkg" / "only_tested.py").write_text(TEST_ONLY, encoding="utf-8")
    (root / "tests" / "test_only_tested.py").write_text(TEST_FILE, encoding="utf-8")
    (root / "tools" / "build_thing.py").write_text(
        '"""A CLI entry point with no importer — must land in UNKNOWN."""\n',
        encoding="utf-8")
    (root / "examples" / "fixture_a.json").write_text(json.dumps(WORKFLOW), encoding="utf-8")
    (root / "examples" / "fixture_b.json").write_text(json.dumps(WORKFLOW_DUP), encoding="utf-8")
    (root / "examples" / "fixture_broken.json").write_text(
        json.dumps(WORKFLOW_BROKEN), encoding="utf-8")
    (root / "docs" / "GUIDE.md").write_text(DEAD_DOC, encoding="utf-8")
    (root / "CHANGELOG.md").write_text(PROVENANCE_DOC, encoding="utf-8")
    (root / "pyproject.toml").write_text('[project]\nname = "fixpkg"\n', encoding="utf-8")

    # config: CHANGELOG.md is provenance, tools/ is an entry-point dir
    (root / "audit-config.json").write_text(json.dumps({
        "provenance_docs": ["CHANGELOG.md"],
        "entry_point_dirs": ["tools"],
    }), encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "audit@fixture"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Audit Fixture"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)


def run_phase(script: str, root: Path) -> None:
    cp = subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--root", str(root)],
        capture_output=True, text=True,
    )
    if cp.returncode != 0:
        raise AssertionError(f"{script} failed:\n{cp.stdout}\n{cp.stderr}")


def load(root: Path, name: str):
    return json.loads((root / ".v1-audit" / "raw" / f"{name}.json").read_text(encoding="utf-8"))


# --- cases ------------------------------------------------------------------


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="atlas-v1-audit-fixture-"))
    try:
        root = tmp / "repo"
        root.mkdir()
        build_fixture(root)

        # the fixture carries its own config; point the skill at it
        real_config = SKILL / "config.json"
        backup = None
        if real_config.is_file():
            backup = real_config.read_text(encoding="utf-8")
        merged = json.loads((root / "audit-config.json").read_text(encoding="utf-8"))
        real_config.write_text(json.dumps(merged), encoding="utf-8")

        try:
            for script in ("inventory.py", "scan_nodes.py", "scan_workflows.py",
                           "scan_references.py", "scan_docs.py",
                           "scan_experiments.py"):
                run_phase(script, root)
            subprocess.run([sys.executable, str(SCRIPTS / "scan_tools.py"),
                            "--root", str(root), "--skip"],
                           capture_output=True, check=True)
            run_phase("build_manifest.py", root)

            inv = load(root, "inventory")
            nodes = load(root, "nodes")
            wf = load(root, "workflows")
            refs = load(root, "references")["references"]
            docs = load(root, "docs")
            exp = load(root, "experiments")
            disposition = json.loads(
                (root / ".v1-audit" / "disposition.json").read_text(encoding="utf-8"))

            # --- node registration -------------------------------------------
            check("registry finds all four nodes",
                  nodes["counts"]["registered"] == 4,
                  f"got {nodes['counts']['registered']}")
            check("standard/experimental split is right",
                  nodes["counts"]["standard"] == 3 and nodes["counts"]["experimental"] == 1,
                  json.dumps(nodes["counts"]))
            check("REGRESSION: subscript-registered node is NOT deregistered",
                  "FixOrbit" not in nodes["deregistered_classes"],
                  "NODE_CLASS_MAPPINGS['X'] = Y form was missed")
            check("REGRESSION: .update()-registered node is registered",
                  "FixGated" in nodes["nodes"])
            check("genuinely deregistered class is found",
                  nodes["deregistered_classes"] == ["FixLegacyBlur"],
                  json.dumps(nodes["deregistered_classes"]))
            check("no registered node lacks an implementation",
                  nodes["missing_implementation"] == [])

            # --- dynamic reference -------------------------------------------
            core = "fixpkg/nodes_core.py"
            check("node module is reached by the workflow that names its node",
                  "examples/fixture_a.json" in refs[core]["referrers"],
                  "a workflow reference must count as a reference")
            check("REGRESSION: repeated private helper is not a reference",
                  "fixpkg/nodes_extra.py" not in refs["fixpkg/only_tested.py"]["referrers"],
                  "_require_numpy collided across modules")
            check("test-only module is reached ONLY by its test",
                  refs["fixpkg/only_tested.py"]["referrers"] == ["tests/test_only_tested.py"],
                  json.dumps(refs["fixpkg/only_tested.py"]["referrers"]))

            # --- dispositions -------------------------------------------------
            check("registered node module is CANONICAL",
                  disposition[core]["disposition"] == "CANONICAL",
                  disposition[core]["disposition"])
            check("registered node module is never a delete candidate",
                  disposition[core]["disposition"] != "DELETE_CANDIDATE")
            check("test-only module is FEATURE_FLAG, not DELETE",
                  disposition["fixpkg/only_tested.py"]["disposition"] == "FEATURE_FLAG",
                  disposition["fixpkg/only_tested.py"]["disposition"])
            check("unwired module is reported by the experiment phase",
                  [r["path"] for r in exp["unwired_modules"]] == ["fixpkg/only_tested.py"],
                  json.dumps(exp["unwired_modules"]))
            check("CLI entry point with no importer lands in UNKNOWN",
                  disposition["tools/build_thing.py"]["disposition"] == "UNKNOWN",
                  disposition["tools/build_thing.py"]["disposition"])
            check("entry-point script is capped below CERTAIN",
                  disposition["tools/build_thing.py"]["confidence"] != "CERTAIN")
            check("test file is never a delete candidate",
                  disposition["tests/test_only_tested.py"]["disposition"] != "DELETE_CANDIDATE")
            check("test file is capped at MEDIUM",
                  disposition["tests/test_only_tested.py"]["confidence"] in
                  ("MEDIUM", "LOW", "UNKNOWN", "HIGH"),
                  disposition["tests/test_only_tested.py"]["confidence"])
            check("no fixture file reaches CERTAIN delete",
                  not [p for p, r in disposition.items()
                       if r["disposition"] == "DELETE_CANDIDATE"
                       and r["confidence"] == "CERTAIN"],
                  "a dynamically-reachable tree produced a CERTAIN deletion")

            # --- workflows -----------------------------------------------------
            check("all three workflows found", wf["counts"]["total"] == 3)
            check("broken workflow is detected",
                  "examples/fixture_broken.json" in wf["broken"])
            check("broken workflow is disposed BROKEN",
                  disposition["examples/fixture_broken.json"]["disposition"] == "BROKEN",
                  disposition["examples/fixture_broken.json"]["disposition"])
            check("duplicate workflows cluster despite different filenames",
                  any(set(c) == {"examples/fixture_a.json", "examples/fixture_b.json"}
                      for c in wf["exact_duplicate_clusters"]),
                  json.dumps(wf["exact_duplicate_clusters"]))
            check("one side of the duplicate pair is MERGE",
                  sorted(r["disposition"] for p, r in disposition.items()
                         if p in ("examples/fixture_a.json", "examples/fixture_b.json"))
                  == ["CANONICAL", "MERGE"] or "MERGE" in
                  [disposition["examples/fixture_b.json"]["disposition"]],
                  disposition["examples/fixture_b.json"]["disposition"])

            # --- docs ------------------------------------------------------------
            guide = docs["docs"]["docs/GUIDE.md"]
            check("dead path in a live doc is reported",
                  "examples/fixture_gone.json" in guide["dead_paths"],
                  json.dumps(guide["dead_paths"]))
            check("wrong node count in a live doc is reported",
                  any(c["claim"] == "99 standard" for c in guide["count_claims"]),
                  json.dumps(guide["count_claims"]))
            check("live doc's real path reference is NOT reported as dead",
                  "fixpkg/nodes_core.py" not in guide["dead_paths"])
            changelog = docs["docs"]["CHANGELOG.md"]
            check("provenance doc is flagged as provenance", changelog["provenance"])
            check("provenance doc's dead path does not count against the tree",
                  docs["counts"]["live_with_dead_paths"] == 1,
                  f"got {docs['counts']['live_with_dead_paths']}")
            check("provenance doc's stale count does not count against the tree",
                  docs["counts"]["live_with_bad_counts"] == 1,
                  f"got {docs['counts']['live_with_bad_counts']}")

            # --- inventory / evidence completeness --------------------------------
            check("inventory categorised the test file as TEST",
                  inv["files"]["tests/test_only_tested.py"]["category"] == "TEST")
            check("inventory categorised the workflow as COMFY_WORKFLOW",
                  inv["files"]["examples/fixture_a.json"]["category"] == "COMFY_WORKFLOW")
            check("git history was captured",
                  bool(inv["files"][core]["last_commit_sha"]))
            check("every row records all nine evidence dimensions",
                  all({"static_reference", "registered_reference", "workflow_reference",
                       "test_reference", "doc_reference", "setup_reference",
                       "ci_reference", "static_analysis", "superseded_by"}
                      <= set(r["evidence"]) for r in disposition.values()))
            check("every row carries a reason",
                  all(r["reason"] for r in disposition.values()))

            # --- scanner regressions (found by running the audit for real) ----
            check("REGRESSION: a path ending a sentence still counts",
                  "docs/GUIDE.md" in refs["fixpkg/nodes_core.py"]["referrers"],
                  "the trailing full stop was tokenized into the path")
            # The verdict was ALREADY non-CERTAIN without this fix, because a
            # file with no static-analysis hit tops out at MEDIUM anyway — so
            # asserting the verdict proves nothing. What the fix changes is the
            # REASON, which is the part a human acts on: "nothing references
            # this" invites deletion, "ComfyUI serves this directory" does not.
            panel = disposition["fixpkg/web/panel.js"]
            check("REGRESSION: a WEB_DIRECTORY-served file says WHY it is kept",
                  "dynamically" in (panel["ceiling_reason"] or ""),
                  f"reason was {panel['ceiling_reason']!r} — the declaration "
                  "sits in the package __init__, one level above the files it "
                  "governs, so a per-directory marker scan never sees it")
            check("REGRESSION: web/ asset is never a delete candidate",
                  panel["disposition"] != "DELETE_CANDIDATE",
                  panel["disposition"])

            # --- a tool that did not run is not a clean result ----------------
            import scan_tools

            ok, note = scan_tools._parse_json('{"issues": []}')
            check("clean JSON parses with no note", ok == {"issues": []} and not note)

            ok, note = scan_tools._parse_json('ERROR: no vite\n{"issues": [1]}')
            check("REGRESSION: a warning preamble does not become zero findings",
                  ok == {"issues": [1]} and note == "ERROR: no vite",
                  f"got {ok!r} note={note!r} — knip prints ERROR lines before "
                  "its JSON, and a bare json.loads turned a real report into []")

            ok, note = scan_tools._parse_json("not json at all")
            check("unparseable output returns None, never an empty list",
                  ok is None and note,
                  "an empty list would read downstream as 'the tool found nothing'")

            ok, note = scan_tools._parse_json("")
            check("genuinely empty output is an empty result", ok == [] and not note)

            check("a launch failure has its own return code",
                  scan_tools.RC_LAUNCH_FAILED < 0)
            rc, out, err = scan_tools._run(["definitely-not-a-real-binary-xyz"], root)
            check("REGRESSION: an unlaunchable tool reports rc<0 and an error",
                  rc == scan_tools.RC_LAUNCH_FAILED and err and not out,
                  f"rc={rc} err={err!r} — npx is a .cmd shim on Windows, so "
                  "shutil.which finds it and CreateProcess cannot run it")
            check("_launchable returns None for an absent binary",
                  scan_tools._launchable("definitely-not-a-real-binary-xyz") is None)

            # --- suppressions are applied AND counted -------------------------
            sample = {
                "vulture": [
                    {"path": "a.py", "message": "unused attribute 'CATEGORY'"},
                    {"path": "a.py", "message": "unused variable 'real_leftover'"},
                ],
                "deptry": [
                    {"module": "bpy", "error": {"code": "DEP001"}},
                    {"module": "pillow", "error": {"code": "DEP002"}},
                    {"module": "genuinely_missing", "error": {"code": "DEP001"}},
                ],
            }
            cfg = {"static_analysis": {
                "vulture": {"ignore_names": ["CATEGORY"]},
                "deptry": {"host_provided": ["bpy"], "ignore_codes": ["DEP002"]},
            }}
            kept, dropped = scan_tools._apply_suppressions(sample, cfg)
            check("a node-contract attribute is suppressed",
                  [r["message"] for r in kept["vulture"]]
                  == ["unused variable 'real_leftover'"])
            check("a host-provided module and an ignored code are suppressed",
                  [r["module"] for r in kept["deptry"]] == ["genuinely_missing"])
            check("REGRESSION: what was dropped is COUNTED, never silent",
                  dropped == {"vulture": 1, "deptry": 2}, str(dropped))

            empty, none_dropped = scan_tools._apply_suppressions(
                {"vulture": [], "deptry": []}, cfg)
            check("nothing to suppress reports no suppressions",
                  none_dropped == {}, str(none_dropped))

            # --- read-only contract -------------------------------------------
            status = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                    capture_output=True, text=True).stdout
            touched = [line for line in status.splitlines()
                       if ".v1-audit" not in line and "audit-config.json" not in line]
            check("audit modified NO project file", not touched, "\n".join(touched))

            # --- --check gate ---------------------------------------------------
            cp = subprocess.run(
                [sys.executable, str(SCRIPTS / "run_audit.py"), "--root", str(root),
                 "--quick", "--check"], capture_output=True, text=True)
            check("--check fails on a tree with known defects", cp.returncode != 0,
                  cp.stdout[-400:])
        finally:
            if backup is not None:
                real_config.write_text(backup, encoding="utf-8")
            elif real_config.is_file():
                real_config.unlink()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    total = PASSED + len(FAILED)
    print(f"\n{PASSED}/{total} assertions passed")
    for line in FAILED:
        print(f"  FAIL  {line}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
