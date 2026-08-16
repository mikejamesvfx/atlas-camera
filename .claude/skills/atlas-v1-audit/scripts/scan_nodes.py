"""Phase 2 — the ComfyUI node registry, read from source, not from an import.

Importing the package would be easier and is wrong: the audit must work on a
checkout whose dependencies are not installed, and an import silently resolves
`atlas_camera` through the editable install — which on this repo points at the
MAIN checkout, so a worktree would audit a different tree. Parsing the mapping
literals with `ast` keeps the answer tied to the files on disk.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

#: name -> tier. A node is never in more than one gated tier.
TIER_DICTS = {
    "EXPERIMENTAL_NODE_CLASS_MAPPINGS": "experimental",
    "LEGACY_NODE_CLASS_MAPPINGS": "legacy",
    "IOS_NODE_CLASS_MAPPINGS": "ios",
}
BASE_DICT = "NODE_CLASS_MAPPINGS"
DISPLAY_DICT = "NODE_DISPLAY_NAME_MAPPINGS"


def _render(value: ast.AST) -> str | None:
    """The source-level name of a mapping's value, or None if not a name."""
    if isinstance(value, ast.Constant):
        return str(value.value)
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return None


def _dict_literals(tree: ast.AST) -> dict[str, dict[str, str]]:
    """Every module-level mapping, however it was built.

    THREE forms, and missing any one of them is a scanner defect rather than a
    finding. `node_registry.py` uses all three: a dict literal for the bulk,
    `MAPPINGS["Key"] = Class` for a dozen nodes appended after their imports,
    and `MAPPINGS.update(OTHER)` for the gated tiers. An earlier revision read
    only the literal and reported twelve live, registered nodes as
    deregistered leftovers — the worst kind of wrong, since the suggested
    remedy is deletion.

    `.update(NAME)` is recorded as an alias so the caller can splice the other
    mapping in once every file has been read; resolving it here would depend
    on file order.
    """
    found: dict[str, dict[str, str]] = {}
    aliases: dict[str, list[str]] = {}

    for node in ast.walk(tree):
        # NAME = {...}
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            name = getattr(node.targets[0], "id", None)
            if name:
                entries = found.setdefault(name, {})
                for key, value in zip(node.value.keys, node.value.values):
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        rendered = _render(value)
                        if rendered is not None:
                            entries[key.value] = rendered
            continue

        # NAME["Key"] = Value
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
            sub = node.targets[0]
            name = getattr(sub.value, "id", None)
            key = sub.slice
            if name and isinstance(key, ast.Constant) and isinstance(key.value, str):
                rendered = _render(node.value)
                if rendered is not None:
                    found.setdefault(name, {})[key.value] = rendered
            continue

        # NAME.update(OTHER) / NAME.update({...})
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update" and node.args):
            name = getattr(node.func.value, "id", None)
            if not name:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Name):
                aliases.setdefault(name, []).append(arg.id)
            elif isinstance(arg, ast.Dict):
                entries = found.setdefault(name, {})
                for key, value in zip(arg.keys, arg.values):
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        rendered = _render(value)
                        if rendered is not None:
                            entries[key.value] = rendered

    for name, sources in aliases.items():
        found.setdefault(name, {})
        found[f"{name}__updates_from"] = {src: src for src in sources}
    return found


def _implemented_classes(root: Path, pkg: Path) -> dict[str, str]:
    """class name -> defining file (REPO-RELATIVE), for node-shaped classes.

    Node-shaped means it declares the ComfyUI contract (`INPUT_TYPES` /
    `RETURN_TYPES` / `FUNCTION`), which is what makes a deregistered class
    detectable as a leftover rather than as an ordinary helper.

    The path must be repo-relative: the manifest matches these against
    inventory keys, and an absolute path never matches one, which silently
    demoted every registered node module from CANONICAL to plain KEEP.
    """
    out: dict[str, str] = {}
    for py in sorted(pkg.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            names = {
                t.id
                for stmt in cls.body if isinstance(stmt, ast.Assign)
                for t in stmt.targets if isinstance(t, ast.Name)
            }
            names |= {f.name for f in cls.body if isinstance(f, ast.FunctionDef)}
            if {"RETURN_TYPES", "INPUT_TYPES", "FUNCTION"} & names:
                out.setdefault(cls.name, common.rel_posix(root, py))
    return out


def _node_prefix(keys) -> str:
    """The registry's own naming prefix, DERIVED rather than hardcoded.

    Downstream phases need to tell "a node type this project owns" from "a
    ComfyUI builtin or another pack's node", and there is no way to enumerate
    the latter. The project's own prefix is the available evidence: on Atlas
    every key starts with `Atlas`. Hardcoding that string made the workflow
    scanner blind to any pack not called Atlas — including the audit's own
    fixture, whose broken workflow went undetected.

    Returns "" when the keys share no prefix of at least three characters, and
    callers then fall back to exact registry membership only.
    """
    keys = sorted(keys)
    if len(keys) < 2:
        return ""
    first, last = keys[0], keys[-1]
    i = 0
    while i < min(len(first), len(last)) and first[i] == last[i]:
        i += 1
    prefix = first[:i]
    return prefix if len(prefix) >= 3 else ""


def build(root: Path, cfg: dict) -> dict:
    pkg = root / "atlas_camera"
    registry_files = []
    base: dict[str, str] = {}
    tiers: dict[str, str] = {}
    display: dict[str, str] = {}

    search = list(pkg.rglob("*.py")) if pkg.is_dir() else list(root.rglob("*.py"))
    for py in sorted(search):
        if common.is_excluded(common.rel_posix(root, py), cfg):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
        literals = _dict_literals(tree)
        if not ({BASE_DICT, DISPLAY_DICT} | set(TIER_DICTS)) & set(literals):
            continue
        registry_files.append(common.rel_posix(root, py))
        base.update(literals.get(BASE_DICT, {}))
        display.update(literals.get(DISPLAY_DICT, {}))
        for dict_name, tier in TIER_DICTS.items():
            for key in literals.get(dict_name, {}):
                tiers[key] = tier
        for dict_name in ("EXPERIMENTAL_NODE_DISPLAY_NAME_MAPPINGS",
                          "LEGACY_NODE_DISPLAY_NAME_MAPPINGS",
                          "IOS_NODE_DISPLAY_NAME_MAPPINGS"):
            display.update(literals.get(dict_name, {}))
        for dict_name in TIER_DICTS:
            base.update(literals.get(dict_name, {}))

    implemented = _implemented_classes(root, pkg if pkg.is_dir() else root)
    prefix = _node_prefix(base)

    standard = [k for k in base if k not in tiers]
    counts = {
        "registered": len(base),
        "standard": len(standard),
        "experimental": sum(1 for t in tiers.values() if t == "experimental"),
        "legacy": sum(1 for t in tiers.values() if t == "legacy"),
        "ios": sum(1 for t in tiers.values() if t == "ios"),
    }

    # A registered key whose class we never found: the mapping points at
    # something the tree does not define. Always a defect, never a candidate.
    missing_impl = sorted(k for k, cls in base.items() if cls not in implemented)
    counts["missing_implementation"] = len(missing_impl)

    # A node-shaped class that no mapping names. For a node pack this is THE
    # superseded-feature signal: the implementation outlived its menu entry.
    # Node-SHAPE is the evidence; a name prefix is not required. Filtering on
    # `startswith("Atlas")` looked harmless and meant the check only worked on
    # one pack — it silently found nothing anywhere else, including in this
    # skill's own fixture.
    registered_classes = set(base.values())
    known_absent = set(cfg.get("known_absent_symbols", []))
    deregistered = sorted(
        name for name in implemented
        if name not in registered_classes and name not in known_absent
    )

    dupes: dict[str, list[str]] = {}
    for key, label in display.items():
        dupes.setdefault(label, []).append(key)

    return {
        "counts": counts,
        "node_prefix": prefix,
        "registry_files": sorted(set(registry_files)),
        "nodes": {
            key: {
                "key": key,
                "class": cls,
                "implementation": implemented.get(cls),
                "display_name": display.get(key),
                "tier": tiers.get(key, "standard"),
            }
            for key, cls in sorted(base.items())
        },
        "display_names": sorted(v for v in display.values() if v),
        "missing_implementation": missing_impl,
        "deregistered_classes": deregistered,
        "duplicate_display_names": {k: sorted(v) for k, v in sorted(dupes.items())
                                    if len(v) > 1},
    }


def main() -> int:
    common.force_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = common.repo_root(Path(args.root))
    cfg = common.load_config(Path(__file__).resolve().parents[1])
    payload = build(root, cfg)
    common.write_raw(root, "nodes", payload)
    c = payload["counts"]
    print(f"nodes: {c['registered']} registered "
          f"({c['standard']} standard, {c['experimental']} experimental, "
          f"{c['legacy']} legacy, {c['ios']} iOS); "
          f"{len(payload['deregistered_classes'])} deregistered class(es)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
