"""File-level derivation audit: VolFill sources vs LaRI sources.

Normalises whitespace/comments, then scores every VolFill file against every
LaRI file with difflib's ratio on the token stream. High ratio == likely copied
or lightly edited; we report the best match per VolFill file so a human can
judge, rather than asserting a threshold verdict.
"""
import difflib, io, json, re, sys, tokenize
from pathlib import Path

ROOT = Path(__file__).parent / "repos"
VOL = ROOT / "volfill"
LARI = ROOT / "lari"


def norm(path: Path) -> list[str]:
    """Source -> comment/docstring-free token list (names + ops, no layout)."""
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out: list[str] = []
    try:
        toks = tokenize.generate_tokens(io.StringIO(src).readline)
        prev_was_stmt_start = True
        for tok in toks:
            if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                            tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING,
                            tokenize.ENDMARKER):
                prev_was_stmt_start = tok.type in (tokenize.NEWLINE, tokenize.NL,
                                                   tokenize.INDENT, tokenize.DEDENT)
                continue
            if tok.type == tokenize.STRING and prev_was_stmt_start:
                continue  # docstring
            out.append(tok.string)
            prev_was_stmt_start = False
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return re.findall(r"\w+|\S", src)
    return out


lari_files = {p: norm(p) for p in LARI.rglob("*.py")}
lari_files = {p: t for p, t in lari_files.items() if len(t) > 40}

rows = []
for vp in sorted((VOL / "volfill").rglob("*.py")):
    vt = norm(vp)
    if len(vt) < 40:
        rows.append({"volfill_file": str(vp.relative_to(VOL)), "tokens": len(vt),
                     "best_lari_match": None, "ratio": 0.0, "note": "trivial/empty"})
        continue
    best, best_r = None, 0.0
    for lp, lt in lari_files.items():
        # cheap length prefilter: wildly different sizes can't be near-copies
        if not (0.4 <= len(lt) / len(vt) <= 2.5):
            continue
        sm = difflib.SequenceMatcher(None, vt, lt, autojunk=False)
        if sm.real_quick_ratio() < best_r or sm.quick_ratio() < best_r:
            continue
        r = sm.ratio()
        if r > best_r:
            best, best_r = lp, r
    rows.append({
        "volfill_file": str(vp.relative_to(VOL)).replace("\\", "/"),
        "tokens": len(vt),
        "best_lari_match": str(best.relative_to(LARI)).replace("\\", "/") if best else None,
        "ratio": round(best_r, 3),
    })

rows.sort(key=lambda r: -r["ratio"])
print(f"{'ratio':>6}  {'tok':>6}  volfill file -> best LaRI match")
for r in rows:
    print(f"{r['ratio']:>6.3f}  {r['tokens']:>6}  {r['volfill_file']} -> {r['best_lari_match']}")
Path(__file__).with_name("out").mkdir(exist_ok=True)
(Path(__file__).parent / "out" / "provenance_volfill_vs_lari.json").write_text(
    json.dumps(rows, indent=2), encoding="utf-8")
