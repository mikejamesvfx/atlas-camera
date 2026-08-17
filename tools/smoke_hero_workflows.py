"""Release acceptance: run each hero workflow and prove it produced artifacts.

The hero workflows are not marketing assets. They are the shortest end-to-end
paths through the product, so they are also the only check that exercises what
CI structurally cannot: a real queue, real models, real files on disk.

That is not a theory. Hero 02's first real run raised
``NameError: _solve_summary is not defined`` from BOTH solver nodes — the two
most important nodes in the pack could not execute at all, and 3150 green tests
said nothing, because no test executes a node that needs torch and models. The
bug had been on main since 2026-08-16. This harness is what would have caught
it on the day.

Usage::

    python tools/smoke_hero_workflows.py --validate-only   # schema only, fast
    python tools/smoke_hero_workflows.py                   # queue them for real

`--validate-only` needs a running server for `/object_info` but queues nothing.
The full run needs the models each workflow's depth backend pulls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_camera.mcp import comfy_http as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: slug -> (workflow file, artifacts its export nodes must leave on disk).
#: Artifact names are RELATIVE to the workflow's own export directory under
#: ComfyUI's working dir. Listing them is the point: "completed with no errors"
#: is not proof, because a node that silently produced nothing also completes.
HEROES: dict[str, dict[str, Any]] = {
    "hero_02": {
        "workflow": "examples/atlas_hero_02_photo_to_editable_scene_workflow.json",
        "export_dir": "atlas_exports/atlas_hero_02_photo_to_editable_scene_workflow",
        "artifacts": [
            "atlas_relief_mesh.obj",
            "atlas_relief_mesh.mtl",
            "atlas_relief_mesh_diffuse.png",
            "build_scene.py",
        ],
        # AtlasSceneHealthGate holds downstream until acknowledged, and on the
        # bundled placeholder plate it always warns. Acknowledging is what an
        # artist does after reading it; the harness does the same so the
        # exporters actually run.
        "overrides": {"5.proceed": True},
        "min_bytes": {"atlas_relief_mesh.obj": 100_000, "build_scene.py": 10_000},
    },
    # hero_01 and hero_03 land here as they are built.
}


def assess_hero_result(result: dict[str, Any], spec: dict[str, Any],
                       *, sizes: dict[str, int]) -> dict[str, Any]:
    """Turn one queue result + on-disk sizes into a pass/fail summary.

    Pure so the offline contract test can drive it without a server.
    ``sizes`` maps artifact name -> byte size, with a MISSING artifact absent
    from the mapping rather than present with size 0 — the two mean different
    things and the caller should not have to guess which.
    """
    if not result.get("completed") or result.get("errors"):
        raise RuntimeError("workflow execution failed: " + "; ".join(
            str(e) for e in result.get("errors") or ["not completed"]))

    expected = list(spec["artifacts"])
    missing = [name for name in expected if name not in sizes]
    if missing:
        raise RuntimeError(
            "workflow completed but produced no " + ", ".join(missing)
            + " — a node that silently exports nothing still reports success")

    empty = [name for name in expected if sizes[name] == 0]
    if empty:
        raise RuntimeError("zero-byte artifact(s): " + ", ".join(empty))

    undersized = [
        f"{name} is {sizes[name]}B, expected >= {floor}B"
        for name, floor in (spec.get("min_bytes") or {}).items()
        if sizes.get(name, 0) < floor
    ]
    if undersized:
        raise RuntimeError("; ".join(undersized))

    return {
        "completed": True,
        "artifacts": {name: sizes[name] for name in expected},
        "reports": sorted((result.get("reports") or {})),
        "output_nodes": sorted(result.get("output_nodes") or []),
    }


def _artifact_sizes(base: Path, spec: dict[str, Any]) -> dict[str, int]:
    export_dir = base / spec["export_dir"]
    sizes: dict[str, int] = {}
    for name in spec["artifacts"]:
        path = export_dir / name
        if path.is_file():
            sizes[name] = path.stat().st_size
    return sizes


def _comfy_root(host: str) -> Path:
    """Where the server writes. Exports are relative to ComfyUI's cwd."""
    import os
    env = os.environ.get("COMFY_DIR")
    if env:
        return Path(env)
    raise SystemExit(
        "set COMFY_DIR to ComfyUI's root so the harness can find exported "
        "artifacts (the server writes them relative to its own cwd)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=C.DEFAULT_HOST)
    parser.add_argument("--hero", action="append", choices=sorted(HEROES),
                        help="run one hero (repeatable); default is all built ones")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--validate-only", action="store_true",
                        help="check each graph against live /object_info, queue nothing")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    oi = C.fetch_object_info(args.host)
    slugs = args.hero or sorted(HEROES)
    failures: list[str] = []

    for slug in slugs:
        spec = HEROES[slug]
        ui = json.loads((ROOT / spec["workflow"]).read_text(encoding="utf-8"))
        errs, warns = C.validate_ui(ui, oi)
        if errs:
            failures.append(f"{slug}: {len(errs)} validation error(s): {errs[:3]}")
            continue
        print(f"{slug}: schema OK ({len(ui['nodes'])} nodes, {len(warns)} warning(s))")
        if args.validate_only:
            continue

        api = C.ui_to_api(ui, oi)
        # Solve gates are closed in shipped workflows by design; the health
        # gate additionally needs its own acknowledgement (see spec overrides).
        overrides = dict(C.gate_overrides(ui, oi))
        overrides.update(spec.get("overrides") or {})
        C.apply_overrides(api, overrides)
        result = C.queue_and_wait(api, host=args.host, timeout=args.timeout)
        sizes = _artifact_sizes(_comfy_root(args.host), spec)
        try:
            summary = assess_hero_result(result, spec, sizes=sizes)
        except RuntimeError as exc:
            failures.append(f"{slug}: {exc}")
            continue
        print(f"{slug}: PASS — " + ", ".join(
            f"{n} {v:,}B" for n, v in summary["artifacts"].items()))

    if failures:
        print("\nFAILURES:")
        for line in failures:
            print("  " + line)
        return 1
    print("\nall heroes green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
