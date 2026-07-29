"""No tracked file may carry a real tailnet identifier.

This repo is public. Tailscale's ``100.64.0.0/10`` addresses are CGNAT and
unreachable without tailnet membership, so an address alone is not a
credential — but a **tailnet ID** (``tail068f49``) and machine names are a
different matter. Enable ``tailscale funnel`` or ``tailscale serve`` on a
machine and its ``<host>.<tailnet>.ts.net`` name becomes publicly resolvable
and reachable from the internet; a hostname published in advance turns that
config change into a URL an attacker already has.

docs/IOS_APP_BOOTSTRAP.md carried exactly that — the author's tailnet ID, two
machine names, their Windows username and home path — and was caught with the
commit still unpushed. This keeps it caught, because the next one will arrive
in a doc nobody re-reads.

Placeholders are the fix and are allowed: ``<host>``, ``<tailnet>``,
``example.ts.net``, and the range ``100.64.0.0/10`` itself.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).name

#: 100.64.0.0/10 — second octet 64-127.
_CGNAT = re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}")
#: Tailscale mints tailnet IDs as "tail" + hex.
_TAILNET_ID = re.compile(r"\btail[0-9a-f]{6,}\b")
#: A MagicDNS name. Angle brackets are allowed IN the pattern so a placeholder
#: matches and can then be recognised as one, rather than slipping past.
_TS_HOST = re.compile(r"([A-Za-z0-9<>._-]+)\.ts\.net")


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True,
        check=True).stdout
    names = [n for n in out.split("\0") if n]
    # An empty listing must FAIL, never quietly pass: a broken git invocation
    # scanning zero files looks identical to a clean repo. (This exact shape of
    # bug already shipped once in the feature audit's tracked-file filter.)
    assert names, "git ls-files returned nothing — cannot verify the repo"
    return [ROOT / n for n in names if Path(n).name != SELF]


def _findings(text: str) -> list[str]:
    hits = []
    for m in _CGNAT.finditer(text):
        # "100.64.0.0/10" is the range in prose, not anyone's address.
        if text[m.end():m.end() + 1] != "/":
            hits.append(m.group(0))
    hits += _TAILNET_ID.findall(text)
    for host in _TS_HOST.findall(text):
        if "<" in host or ">" in host:
            continue                       # a placeholder, which is the fix
        if host.split(".")[-1] == "example":
            continue                       # documentation fixture
        hits.append(f"{host}.ts.net")
    return hits


def test_no_tracked_file_carries_a_tailnet_identifier():
    problems = []
    for path in _tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue                       # binary, or a stale index entry
        for hit in _findings(text):
            problems.append(f"{path.relative_to(ROOT).as_posix()}: {hit}")
    assert not problems, (
        "real tailnet identifiers in tracked files — replace with placeholders "
        "(<host>, <tailnet>) before pushing:\n  " + "\n  ".join(problems))


def test_the_guard_detects_what_it_is_written_for():
    """The regressions this exists to stop, and the forms it must not flag."""
    caught = _findings(
        "tailscale file cp x mikes-macbook-pro.tail068f49.ts.net:\n"
        "ssh://miike@mjomen-1.tail068f49.ts.net/C:/Users/miike\n"
        "the box is at 100.81.12.88\n")
    assert "tail068f49" in caught
    assert "mikes-macbook-pro.tail068f49.ts.net" in caught
    assert "100.81.12.88" in caught

    assert _findings(
        "clone ssh://<user>@<windows-hostname>/C:/path\n"
        "the full <host>.<tailnet>.ts.net form\n"
        "Tailscale's 100.64.0.0/10 addresses are CGNAT\n"
        "assert find_phone(...) == 'iphone-12-pro.example.ts.net'\n"
        "a plain 100.5.4.3 is not CGNAT\n") == []
