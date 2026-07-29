"""Run shipped workflows and score their OUTPUT, not their process.

Not a golden-frame gate. That answers "did it change?" and cannot grade — which
matters, because on 2026-07-29 a depth completion reported 100% of tears filled
at confidence 1.00 while the render got visibly worse. Both numbers were true
and neither measured the thing that mattered.

So this records what came OUT:

    non_black_frac      how much of the render carries geometry
    already_tears_pct   AtlasMoveBudget's "% of frame already tears at the
                        recovered camera" — the number that caught it
    dropped_faces       faces that fell behind the candidate camera
    dolly_m             the safe camera envelope, i.e. what a user actually buys
    runtime_s

The process-side numbers (pixels synthesised, completion confidence) are
deliberately NOT scored. They were the ones that lied.

    python tools/workflow_benchmark.py --list
    python tools/workflow_benchmark.py --only atlas_input_quickstart_workflow
    python tools/workflow_benchmark.py --all --save

Deltas are reported as improved / regressed / unchanged rather than pass/fail.
These runs put depth models on a GPU, so the numbers WOBBLE between runs — a
threshold decides what counts as movement, and bit-identity is not on offer.
That is the whole reason this is separate from the golden-frame suite, which is
fenced at the deterministic geometry boundary precisely to avoid this.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SCOREBOARD = ROOT / "reports" / "workflow_benchmark.json"
HOST = "127.0.0.1:8188"

#: Below this fractional change, treat a metric as noise rather than movement.
#: Model inference is not deterministic run to run; calling a 0.3% wobble a
#: regression would train everyone to ignore the report.
NOISE_FRAC = 0.02

#: Workflows needing assets that are not in the repo. Skipped by default rather
#: than reported as failures — a missing plate is not a regression.
KNOWN_MISSING_ASSETS = {
    "atlas_auto_layered_inpaint_workflow": "cleanplate.png",
    # Same asset. Found because ComfyUI prunes the failing branch and still
    # reports success, so this ran "green" while its viewport never executed.
    "atlas_layered_segmentation_workflow": "cleanplate.png",
    "atlas_unseen_geometry_test_workflow": "moge_hangar_proj.jpg",
    # A camera RAW is the artist's own file, never shipped. The workflow is
    # correct; the asset simply cannot live in the repo.
    "atlas_raw_3layer_ocio_workflow": "input/CameraRaw/*.NEF",
}


def shipped_workflows() -> list:
    """Shipped graphs only — never an artist's personal '-edit' copy."""
    return sorted(p for p in EXAMPLES.glob("*.json")
                  if "-edit" not in p.stem and "local" not in p.parts)


def _post(path: str, payload: dict, host: str, timeout: int = 120):
    req = urllib.request.Request(
        f"http://{host}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as exc:
        # ComfyUI puts the REASON in the 400 body. Letting the bare HTTPError
        # escape reported "HTTP Error 400: Bad Request" for a workflow whose
        # actual problem was a one-line "Invalid image file: <name>".
        #
        # Note the two different shapes: when only SOME outputs fail validation
        # ComfyUI prunes them and returns 200 + success (see
        # summarise_node_errors); when ALL of them fail it returns this 400.
        try:
            body = json.loads(exc.read())
        except Exception:  # noqa: BLE001
            raise exc from None
        why = [f"{e.get('class_type', '?')}(id{nid}): "
               f"{d.get('details') or d.get('message')}"
               for nid, e in (body.get("node_errors") or {}).items()
               for d in e.get("errors", [])]
        msg = (body.get("error") or {}).get("message", "prompt rejected")
        raise RuntimeError(f"{msg}: {'; '.join(why[:3]) or 'no detail given'}") \
            from None


def _get(path: str, host: str, timeout: int = 120):
    return json.loads(urllib.request.urlopen(
        f"http://{host}{path}", timeout=timeout).read())


def parse_move_budget(text: str) -> dict:
    """Pull the numbers that actually track quality out of a budget report.

    ``dolly_m`` is dropped when the budget reports "unbounded within search cap".
    That value is the SEARCH LIMIT, not a measurement — the probe never found an
    edge — and recording it produces a scoreboard where 1359.373 sits beside
    0.725 and swamps every real delta. Measured live on two workflows before this
    guard existed.
    """
    out = {}
    unbounded = "unbounded within search cap" in text
    m = re.search(r"dolly\s+x\s*\+/-([\d.]+)\s*m", text)
    if m and not unbounded:
        out["dolly_m"] = float(m.group(1))
    elif m:
        out["dolly_unbounded"] = True
    m = re.search(r"([\d.]+)%\s+of frame already tears", text)
    if m:
        out["already_tears_pct"] = float(m.group(1))
    m = re.search(r"(\d+)\s+faces fell behind", text)
    if m:
        out["dropped_faces"] = int(m.group(1))
    return out


def summarise_node_errors(node_errors: dict | None) -> dict:
    """Turn ComfyUI's ``node_errors`` into a failed measurement, or ``{}``.

    ComfyUI does NOT reject a prompt whose node fails validation. It prunes
    that node and every output downstream of it, executes the remainder, and
    reports ``status_str: "success"``. A harness that reads only the status
    therefore records a pass for a run whose real output never executed —
    atlas_layered_segmentation_workflow sat green in the scoreboard with its
    viewport pruned by a missing cleanplate.png, and the run was described as
    merely "unscoreable".

    A partially pruned graph is a FAILED measurement, not a pass.
    """
    if not node_errors:
        return {}
    pruned, why = set(), []
    for nid, err in node_errors.items():
        pruned.update(str(x) for x in (err.get("dependent_outputs") or []))
        for e in err.get("errors", []):
            why.append(f"{err.get('class_type', '?')}(id{nid}): "
                       f"{e.get('details') or e.get('message')}")
    return {"pruned_outputs": sorted(pruned, key=lambda s: (len(s), s)),
            "error": "graph partially pruned by ComfyUI validation — "
                     + "; ".join(why[:3])}


def _resolve_image(rec: dict, output_dir: Path):
    """Where ComfyUI actually put an image it reported.

    ``type`` is one of output/temp/input and they are SIBLING directories.
    ``PreviewImage`` — which most Atlas debug tails use, because it does not
    litter the output folder — reports ``temp``.
    """
    name = rec.get("filename")
    if not name:
        return None
    kind = rec.get("type") or "output"
    root = output_dir if kind == "output" else output_dir.parent / kind
    return root / (rec.get("subfolder") or "") / name


def score_images(records: list, output_dir: Path) -> dict:
    """How much of the render carries geometry. The measure that caught it.

    Takes ComfyUI's full image records, not bare filenames. ``PreviewImage``
    writes to the **temp** directory and reports ``type: "temp"``; resolving
    every record against ``output/`` silently found nothing and scored the run
    as unmeasurable. Three shipped workflows were mis-diagnosed that way.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return {}
    # Prefer the INJECTED render: every workflow gets the same instrument, so
    # scoring a graph's own arbitrary SaveImage would make the numbers
    # incomparable between workflows.
    bench = [r for r in records if str(r.get("filename", "")).startswith("bench")]
    records = bench or records
    best = {}
    for rec in records:
        name = rec.get("filename", "")
        p = _resolve_image(rec, output_dir)
        if p is None or not p.exists():
            continue
        try:
            a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        except Exception:  # noqa: BLE001
            continue
        frac = float((a.max(axis=2) > 0.02).mean())
        # Several SaveImage nodes per graph; keep the WORST, because a graph is
        # only as good as its weakest visible output.
        if "non_black_frac" not in best or frac < best["non_black_frac"]:
            best = {"non_black_frac": round(frac, 5),
                    "scored_image": name,
                    "size": [a.shape[1], a.shape[0]]}
    return best


def apply_variant(api: dict, spec: str) -> tuple:
    """Patch ``NodeClass.input=value`` across a graph, for A/B runs.

    WHY THIS REPORTS AND DOES NOT ADOPT
    Every metric here is gameable in one direction: `non_black_frac` and
    `already_tears_pct` are both optimised by NEVER TEARING. Raise
    `max_edge_factor` far enough and the scoreboard goes green while every
    silhouette rubber-sheets — the exact artifact DESIGN_RULES calls
    load-bearing tearing.

    Grading that automatically needs ground truth, which exists on the synthetic
    scenes (`core.tear_metrics`) and does NOT exist on a real plate. So on real
    workflows the honest output is the trade, shown to a person. Anything that
    picked a winner here would be picking the degenerate solution and calling it
    an improvement.
    """
    if "=" not in spec or "." not in spec.split("=", 1)[0]:
        raise SystemExit(f"--variant wants NodeClass.input=value, got {spec!r}")
    target, raw = spec.split("=", 1)
    cls, field = target.split(".", 1)
    try:
        value = json.loads(raw)          # numbers, booleans, null
    except json.JSONDecodeError:
        value = raw                      # bare strings / combo values
    out, hits = {}, 0
    for nid, node in api.items():
        node = dict(node)
        if node.get("class_type") == cls:
            inputs = dict(node.get("inputs") or {})
            inputs[field] = value
            node["inputs"] = inputs
            hits += 1
        out[nid] = node
    if hits == 0:
        raise SystemExit(f"no {cls} node in this graph to apply {spec!r} to")
    return out, f"{cls}.{field}={value!r} on {hits} node(s)"


def attach_measurement_tail(api: dict, oi: dict) -> tuple:
    """Append a standard scoring tail to whatever solve the graph produces.

    Most shipped workflows END IN AtlasBlockoutViewport, which renders in the
    BROWSER via three.js — so run headlessly they emit nothing at all and there
    is nothing to score. Rather than restrict the benchmark to the handful of
    graphs that happen to save an image, every graph gets the SAME instrument
    bolted on: move budget for the geometry numbers, a server-side stereo render
    for the picture.

    That also makes results comparable BETWEEN workflows, which they would not be
    if each were scored by whatever tail its author happened to wire.

    Returns (api, note). The graph is left untouched when no solve is found.
    """
    def out_types(class_type):
        info = oi.get(class_type) or {}
        return list(info.get("output") or [])

    # A solve is "terminal" when nothing else in the graph consumes it.
    consumed = set()
    for node in api.values():
        for v in (node.get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2:
                consumed.add((str(v[0]), int(v[1])))

    solve_src = image_src = None
    for nid, node in api.items():
        types = out_types(node.get("class_type", ""))
        for slot, t in enumerate(types):
            if t == "ATLAS_SOLVE" and (str(nid), slot) not in consumed:
                solve_src = [str(nid), slot]
            if t == "IMAGE" and image_src is None:
                image_src = [str(nid), slot]
    if solve_src is None:
        # Fall back to ANY solve output, terminal or not — a graph whose solve is
        # consumed everywhere still has geometry worth measuring.
        for nid, node in api.items():
            for slot, t in enumerate(out_types(node.get("class_type", ""))):
                if t == "ATLAS_SOLVE":
                    solve_src = [str(nid), slot]
    if solve_src is None:
        return api, "no ATLAS_SOLVE in the graph — nothing to measure"

    base = max((int(k) for k in api if str(k).isdigit()), default=0) + 1000
    api = dict(api)
    api[str(base)] = {"class_type": "AtlasMoveBudget",
                      "inputs": {"solve": solve_src}}
    api[str(base + 1)] = {"class_type": "ShowText|pysssss",
                          "inputs": {"text": [str(base), 1]}}
    note = "tail: move budget"
    if image_src is not None:
        api[str(base + 2)] = {"class_type": "AtlasStereoRender", "inputs": {
            "solve": solve_src, "source_image": image_src,
            "interocular_m": 0.30, "convergence_m": 5.0,
            "output_mode": "sbs", "resolution": 1024}}
        api[str(base + 3)] = {"class_type": "SaveImage", "inputs": {
            "images": [str(base + 2), 0], "filename_prefix": "bench"}}
        note = "tail: move budget + stereo render"
    return api, note


def run_one(path: Path, host: str, output_dir: Path, timeout: int = 1800,
            variant: str = "") -> dict:
    """Queue one workflow and collect its output-side metrics."""
    from atlas_camera.mcp import comfy_http as ch

    rec = {"workflow": path.stem, "ok": False}
    stem = path.stem
    if stem in KNOWN_MISSING_ASSETS:
        rec.update(skipped=f"needs {KNOWN_MISSING_ASSETS[stem]}, not in the repo")
        return rec

    t0 = time.time()
    try:
        oi = _get("/object_info", host, timeout=180)
        ui = json.loads(path.read_text(encoding="utf-8"))
        api = ch.ui_to_api(ui, oi)
        # Shipped workflows ship with their solve gates CLOSED — that is the
        # gate doctrine, and correct for a human opening the workflow. Headless
        # it means AtlasSolveGate pauses everything downstream, so the graph
        # runs, succeeds, and measures nothing. Both camera_staged_master
        # workflows looked "unscoreable" for exactly this reason and are not
        # broken at all. atlas_run_workflow opens gates by default; so do we.
        # gate_overrides reads the UI graph, which still contains MUTED nodes;
        # ui_to_api drops them. Overriding a dropped node raises KeyError and
        # would turn a perfectly good workflow into a spurious harness error,
        # so keep only gates that survived the flatten.
        gates = {k: v for k, v in ch.gate_overrides(ui, oi).items()
                 if k.partition(".")[0] in api}
        if gates:
            ch.apply_overrides(api, gates)
            rec["gates_opened"] = len(gates)
        if variant:
            api, vnote = apply_variant(api, variant)
            rec["variant"] = vnote
        api, tail_note = attach_measurement_tail(api, oi)
        rec["tail"] = tail_note
        posted = _post("/prompt", {"prompt": api, "client_id": str(uuid.uuid4())},
                       host)
        # ComfyUI does NOT reject a graph whose node fails validation. It PRUNES
        # that node and everything downstream, runs the remainder, and reports
        # status_str "success". Ignoring this recorded ok=True for runs whose
        # real output never executed — atlas_layered_segmentation_workflow sat
        # green in the scoreboard with its viewport pruned by a missing
        # cleanplate.png. A pruned graph is a failed measurement, not a pass.
        pruned = summarise_node_errors(posted.get("node_errors"))
        if pruned:
            rec.update(pruned)
            return rec
        pid = posted["prompt_id"]
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return rec

    hist = {}
    while time.time() - t0 < timeout:
        try:
            hist = _get(f"/history/{pid}", host, timeout=60)
        except Exception:  # noqa: BLE001
            time.sleep(5)
            continue
        if hist:
            break
        time.sleep(5)
    rec["runtime_s"] = round(time.time() - t0, 1)
    if not hist:
        rec["error"] = f"no result within {timeout}s"
        return rec

    entry = hist.get(pid, {})
    status = entry.get("status", {})
    if status.get("status_str") != "success":
        for m in status.get("messages", []):
            if m[0] == "execution_error":
                rec["error"] = (f"{m[1].get('node_type')}: "
                                f"{str(m[1].get('exception_message'))[:250]}")
        rec.setdefault("error", "failed")
        return rec

    images, texts = [], []
    for o in (entry.get("outputs") or {}).values():
        images += [im for im in o.get("images", []) if isinstance(im, dict)]
        texts += [str(t) for t in (o.get("text") or [])]

    rec["ok"] = True
    rec["n_images"] = len(images)
    scored = score_images(images, output_dir)
    rec.update(scored)
    if not scored:
        # Say WHICH of these it was. The previous message guessed ("no IMAGE
        # source found?") and the guess was wrong every time it fired: the
        # images existed, they were just in temp/ or the branch was gated.
        if any(t.lstrip().startswith("⏸") for t in texts):
            rec["no_image_reason"] = (
                "halted at a closed AtlasSolveGate — nothing downstream ran")
        elif images:
            rec["no_image_reason"] = (
                f"{len(images)} image(s) reported but none readable on disk: "
                + ", ".join(f"{i.get('type', '?')}/{i.get('filename')}"
                            for i in images[:3]))
        else:
            rec["no_image_reason"] = "the graph produced no image at all"

    # WHY a metric is absent is as informative as the metric. A silently missing
    # number reads as "not measured" when it often means "could not be measured",
    # and those want different responses.
    budget = next((t for t in texts if "Safe camera envelope" in t), None)
    if budget:
        parsed = parse_move_budget(budget)
        rec.update(parsed)
        if not parsed:
            rec["no_budget_reason"] = "budget text present but nothing parsed from it"
    else:
        # AtlasMoveBudget already SAYS why it declined, in a line beginning
        # "Move budget not computed:". Quote it rather than pattern-matching a
        # guess at its wording — the previous guard looked for "relief mesh is
        # empty" while the node says "needs a relief mesh to seal", so it never
        # fired and three workflows reported the useless "produced no envelope
        # report" instead of the real reason sitting in the output.
        said = next((t for t in texts if "Move budget not computed" in t), None)
        rec["no_budget_reason"] = (
            said.strip().splitlines()[0] if said else
            "AtlasMoveBudget emitted no report at all")
    return rec


def compare(now: dict, before: dict) -> list:
    """Deltas, labelled by DIRECTION rather than pass/fail.

    Higher-is-better and lower-is-better are declared per metric, because a
    scoreboard that cannot say which way is up is just a wall of numbers.
    """
    higher_better = {"non_black_frac", "dolly_m"}
    lower_better = {"already_tears_pct", "dropped_faces", "runtime_s"}
    lines = []
    for name, cur in sorted(now.items()):
        old = before.get(name)
        if not old or not cur.get("ok") or not old.get("ok"):
            continue
        for key in sorted((higher_better | lower_better) & set(cur) & set(old)):
            a, b = float(old[key]), float(cur[key])
            if a == 0 and b == 0:
                continue
            rel = (b - a) / (abs(a) or 1.0)
            if abs(rel) < NOISE_FRAC:
                continue
            better = (b > a) if key in higher_better else (b < a)
            lines.append(
                f"  {'IMPROVED ' if better else 'REGRESSED'} {name}.{key}: "
                f"{a:g} -> {b:g} ({rel * 100:+.1f}%)")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--output-dir",
                    default=r"C:\Users\miike\ComfyUI_V91\ComfyUI\output")
    ap.add_argument("--only", action="append", default=[],
                    help="workflow stem; repeatable")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--save", action="store_true",
                    help="write the scoreboard (otherwise report only)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--variant", action="append", default=[],
                    help="NodeClass.input=value — runs an extra pass per variant "
                         "and REPORTS the trade. Never auto-adopted; see "
                         "apply_variant's docstring for why.")
    args = ap.parse_args()

    flows = shipped_workflows()
    if args.list:
        for p in flows:
            note = KNOWN_MISSING_ASSETS.get(p.stem)
            print(f"  {p.stem}{'  (skipped: ' + note + ')' if note else ''}")
        return

    if args.only:
        wanted = set(args.only)
        flows = [p for p in flows if p.stem in wanted]
        if not flows:
            raise SystemExit(f"no shipped workflow matched {sorted(wanted)}")
    elif not args.all:
        raise SystemExit("pass --only <stem> (repeatable), --all, or --list")

    before = {}
    if SCOREBOARD.exists():
        before = json.loads(SCOREBOARD.read_text(encoding="utf-8")).get("workflows", {})

    now = {}
    passes = [("", "base")] + [(v, v) for v in args.variant]
    for i, p in enumerate(flows, 1):
        for variant, label in passes:
            tag = p.stem if label == "base" else f"{p.stem} [{label}]"
            print(f"[{i}/{len(flows)}] {tag} ...", flush=True)
            rec = run_one(p, args.host, Path(args.output_dir),
                          timeout=args.timeout, variant=variant)
            now[p.stem if label == "base" else f"{p.stem}::{label}"] = rec
            # Report inside the variant loop — otherwise only the last pass
            # prints and the base run vanishes from the output.
            if rec.get("skipped"):
                print(f"    skipped: {rec['skipped']}")
            elif rec.get("ok"):
                bits = [f"{k}={rec[k]}" for k in
                        ("non_black_frac", "already_tears_pct", "dropped_faces",
                         "dolly_m", "runtime_s") if k in rec]
                print("    " + "  ".join(bits))
                for why in ("no_budget_reason", "no_image_reason"):
                    if rec.get(why):
                        print(f"      ({rec[why]})")
            else:
                print(f"    FAILED: {rec.get('error')}")

    if args.variant:
        print("\n=== variants vs base (REPORTED, not adopted) ===")
        for p in flows:
            base = now.get(p.stem)
            if not base or not base.get("ok"):
                continue
            for v in args.variant:
                alt = now.get(f"{p.stem}::{v}")
                if not alt or not alt.get("ok"):
                    continue
                print(f"  {p.stem}  {v}")
                for key in ("non_black_frac", "already_tears_pct",
                            "dropped_faces", "dolly_m", "runtime_s"):
                    if key in base and key in alt:
                        a, b = float(base[key]), float(alt[key])
                        arrow = "->" if a == b else ("UP  " if b > a else "DOWN")
                        print(f"      {key:20} {a:>10g} {arrow} {b:<10g}")
        print("  Read the trade; do not read a winner. Every metric above is\n"
              "  maximised by NEVER TEARING, which is the artifact Atlas exists\n"
              "  to avoid — grading this needs ground truth, which a real plate\n"
              "  does not have.")

    deltas = compare(now, before)
    if before:
        print("\n=== vs the committed scoreboard ===")
        print("\n".join(deltas) if deltas
              else f"  nothing moved by more than {NOISE_FRAC * 100:g}%")
    else:
        print("\n  no previous scoreboard — this run becomes the baseline once saved")

    if args.save:
        SCOREBOARD.parent.mkdir(parents=True, exist_ok=True)
        merged = dict(before)
        merged.update(now)
        SCOREBOARD.write_text(json.dumps(
            {"note": ("Output-side metrics for shipped workflows. Process-side "
                      "numbers (pixels synthesised, completion confidence) are "
                      "deliberately absent — they moved opposite to quality on "
                      "2026-07-29."),
             "noise_threshold": NOISE_FRAC,
             "workflows": merged}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"\n  wrote {SCOREBOARD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
