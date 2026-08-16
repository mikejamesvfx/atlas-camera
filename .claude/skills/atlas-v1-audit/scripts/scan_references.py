"""Phase 4 — who reaches what, beyond the import graph.

This phase is the reason the audit exists. In Atlas a module is routinely
reached with no Python import anywhere: through `NODE_CLASS_MAPPINGS`, a
serialized workflow, an MCP tool name, pytest collection, `pkgutil`, or a DCC
host executing an exported script out-of-process. Treating "no importer" as
"dead" would nominate live features for deletion.

So every tracked TEXT file is scanned for every other file's *handles*: its
path, its bare filename, its dotted module name, the class names it defines,
and — for node modules — the registry keys and display names those classes
carry. A hit in any file counts, and which KIND of file it was decides which
evidence dimension it lands in.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TEXT_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json", ".yaml", ".yml",
    ".toml", ".md", ".rst", ".txt", ".sh", ".ps1", ".bat", ".cmd", ".cfg", ".ini",
    "",  # Dockerfile, Makefile
}
MAX_BYTES = 4_000_000

#: One token = one identifier OR one path. Dots and slashes stay inside the
#: token so `atlas_camera.core.solver` and `examples/foo.json` survive whole.
TOKEN_RE = re.compile(r"[\w./-]+")


def _handles(root: Path, rel: str, node_index: dict) -> set[str]:
    """Every string by which some other file could name this one.

    PRIVATE symbols are excluded. A leading underscore means module-private by
    convention, and Atlas repeats the same private helper names across the
    package — `_require_numpy` alone is defined in most of `core/`. Treating
    those as handles made `core/depth_calibration.py` look like it had 75
    referrers when nothing imports it at all, which silently hid the one
    genuinely unwired module in the tree. A name that repeats by convention is
    evidence of convention, not of reference.
    """
    path = Path(rel)
    out = {rel, path.name}
    if path.suffix == ".py":
        out.add(path.stem)
        out.add(rel[:-3].replace("/", "."))
        if path.name == "__init__.py":
            out.add(str(path.parent).replace("\\", "/").replace("/", "."))
        try:
            tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        except (SyntaxError, OSError, UnicodeDecodeError):
            tree = None
        if tree is not None:
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    if not node.name.startswith("_"):
                        out.add(node.name)
    # a node module also answers to its registry keys and menu labels
    for info in node_index.values():
        if info.get("implementation", "").endswith(rel):
            out.add(info["key"])
            if info.get("display_name"):
                out.add(info["display_name"])
    return {h for h in out if h and len(h) > 3}


def _read(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def build(root: Path, cfg: dict) -> dict:
    inventory = common.read_raw(root, "inventory")["files"]
    nodes = common.read_raw(root, "nodes")["nodes"]
    node_index = {
        k: {**v, "implementation": (v.get("implementation") or "")}
        for k, v in nodes.items()
    }

    scannable = [rel for rel in inventory
                 if Path(rel).suffix in TEXT_EXTS or "." not in Path(rel).name]
    corpus: dict[str, str] = {}
    for rel in scannable:
        text = _read(root / rel)
        if text is not None:
            corpus[rel] = text

    handles = {rel: _handles(root, rel, node_index) for rel in inventory}

    # TOKENIZE each file once, then intersect with handle sets.
    #
    # The obvious implementation — one compiled alternation per target,
    # searched across every file — is 588 x 557 regex passes. With word-
    # boundary lookarounds that took the phase from 21 seconds to over six
    # minutes, so this inverts it: each file is split into identifier/path
    # tokens exactly once, and a handle "matches" when it IS one of those
    # tokens. That is linear in corpus size, and it is also STRICTER than the
    # regex was — a token is a whole identifier, so `calibrate` can no longer
    # match inside `recalibrate` for free.
    tokenized: dict[str, set[str]] = {}
    for rel, text in corpus.items():
        raw = TOKEN_RE.findall(text.replace("\\", "/"))
        tokens = set(raw)
        # a path token also answers to each of its trailing segments, so
        # "atlas_camera/core/solver.py" is found by "core/solver.py" too
        for token in raw:
            if "/" in token:
                parts = token.split("/")
                for i in range(1, len(parts)):
                    tokens.add("/".join(parts[i:]))
        tokenized[rel] = tokens

    # Display names carry spaces and emoji, so they are never tokens. They are
    # few, so a plain substring pass over the corpus is affordable.
    spaced = {rel: {h for h in hs if not TOKEN_RE.fullmatch(h)}
              for rel, hs in handles.items()}

    refs: dict[str, dict] = {}
    for target, target_handles in handles.items():
        plain = {h for h in target_handles if h not in spaced[target]}
        referrers = []
        for rel, tokens in tokenized.items():
            if rel == target:
                continue
            if plain & tokens:
                referrers.append(rel)
                continue
            if spaced[target] and any(h in corpus[rel] for h in spaced[target]):
                referrers.append(rel)
        by_category: dict[str, int] = {}
        for rel in referrers:
            cat = inventory[rel]["category"] if rel in inventory else common.categorize(rel)
            by_category[cat] = by_category.get(cat, 0) + 1
        refs[target] = {"referrers": sorted(referrers), "by_category": by_category}

    return {"references": refs, "scanned": len(corpus)}


def main() -> int:
    common.force_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = common.repo_root(Path(args.root))
    cfg = common.load_config(Path(__file__).resolve().parents[1])
    payload = build(root, cfg)
    common.write_raw(root, "references", payload)
    orphans = sum(1 for v in payload["references"].values() if not v["referrers"])
    print(f"references: scanned {payload['scanned']} text files; "
          f"{orphans} file(s) with no inbound reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
