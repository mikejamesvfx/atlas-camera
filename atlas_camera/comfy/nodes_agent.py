"""🤝 AtlasAgentHandoff — pause the graph, brief an external agent, resume on its reply.

The contract half of the "Atlas agent runner" idea (2026-08-16). The node holds
NO model and NO MCP client: it publishes a brief (task, exchange dir, seed /
.blend paths, snapshot PNGs, measured numbers, allowed tools, resume
instructions), BLOCKS until an agent — Claude Code through the atlas MCP,
Hermes, OpenClaw, a human with curl — writes resume.json, then optionally runs
`export_meshes.py` on the agent's .blend and appends the meshes exactly like
AtlasBlenderImportMeshes. Timeout / skip / fail are third outcomes: reported,
never a raise (unless `on_timeout=fail`, which is the artist's explicit choice).

A later `operator=local_vlm` mode can run the loop in-process against the same
brief; the contract does not change.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from atlas_camera.comfy import agent_handoff as AH
from atlas_camera.comfy.node_helpers import _require_numpy  # noqa: F401  (import guard parity)

_TOOL_CHOICES = ("blender_mcp", "blender_headless", "atlas_mcp", "comfy_mcp", "filesystem")


class AtlasAgentHandoff:
    """🤝 Pause here; an agent operates Blender (or anything) and hands back.

    Feed it the solve after `AtlasBlenderMassing` (and that node's
    `exchange_dir`), write the task in plain words, queue. The graph WAITS
    (`timeout_s`). Meanwhile the brief is at
    <output>/atlas_agent/<node>/brief.json — an agent reads it (MCP
    `atlas_agent_brief`), looks at the attached viewport snapshots, opens
    <exchange_dir>/scene.blend (blender-mcp GUI, or headless recipes), models
    under the `atlas_out` collection, saves, and resumes (MCP
    `atlas_agent_resume` / `POST /atlas/agent/resume/{node}` / resume.json).
    With `auto_import` the node then runs `export_meshes.py` on the returned
    .blend and appends the meshes as PROXY_ROLE geometry with projective UVs.
    """

    RETURN_TYPES = ("ATLAS_SOLVE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("solve", "agent_reply", "brief_path", "report")
    FUNCTION = "handoff"
    CATEGORY = "Atlas Camera/Experimental"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solve": ("ATLAS_SOLVE",),
                "task": ("STRING", {
                    "multiline": True,
                    "default": "Look at the snapshots and the measured scene. In Blender, model "
                               "the building volumes and any ground/facade geometry the photo "
                               "shows but the massing missed, under the atlas_out collection, "
                               "at the measured scale. Save scene.blend and resume with a short "
                               "note of what you added.",
                    "tooltip": "Plain-language instruction for the agent. Goes into the brief verbatim."}),
            },
            "optional": {
                "exchange_dir": ("STRING", {
                    "default": "", "forceInput": False,
                    "tooltip": "AtlasBlenderMassing's exchange_dir output (seed + scene.blend). "
                               "Empty = brief carries no Blender scene."}),
                "depth": ("ATLAS_DEPTH_MAP", {
                    "tooltip": "Optional; its metadata (model, predicted focal) rides the brief."}),
                "snapshot_node_id": ("STRING", {
                    "default": "",
                    "tooltip": "Viewport node id whose automatic 1280 snapshots to attach "
                               "(output/atlas_viewport/viewport_<id>_*.png). Empty = attach any "
                               "found."}),
                "tools_allowed": ("STRING", {
                    "default": "blender_mcp, blender_headless, atlas_mcp",
                    "tooltip": "Comma list the agent is told it may use: " + ", ".join(_TOOL_CHOICES)}),
                "blender_path": ("STRING", {"default": "",
                    "tooltip": "For auto_import's export_meshes.py run. Empty = ATLAS_BLENDER_PATH/PATH."}),
                "timeout_s": ("INT", {"default": 300, "min": 10, "max": 86400,
                    "tooltip": "How long the graph waits for a resume. The queue is blocked "
                               "meanwhile — that IS the pause. ComfyUI runs one prompt at a "
                               "time, so this stalls EVERY queued job, not just this graph; "
                               "the 5-minute default keeps an absent agent from reading as a "
                               "dead server. Raise it for a long modelling session."}),
                "on_timeout": (["continue", "fail"], {"default": "continue"}),
                "auto_import": ("BOOLEAN", {"default": True,
                    "tooltip": "On resume with a blend_file (or when scene.blend exists in "
                               "exchange_dir), run export_meshes.py headless and append the "
                               "atlas_out meshes to the solve."}),
                "expect_fingerprint": ("BOOLEAN", {"default": True,
                    "tooltip": "Refuse imported meshes whose seed came from a different solve."}),
                "min_y_m": ("FLOAT", {"default": -0.05, "min": -100.0, "max": 100.0, "step": 0.01}),
                "poll_s": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 30.0, "step": 0.1}),
                "project": ("ATLAS_PROJECT", {
                    "tooltip": "Optional delivery project — resolves exchange_dir into the shot's "
                               "blender/ lane like AtlasBlenderMassing, and files the brief/resume "
                               "under <shot>/blender/agent/<node> as well as the ComfyUI output copy."}),
                # APPENDED 2026-08-16: the override switch (the artist re-queues
                # a graph a dozen times; not every run needs an agent).
                "mode": (["wait", "import_only", "skip"], {
                    "default": "wait",
                    "tooltip": "wait: publish the brief and PAUSE for an agent (default). "
                               "import_only: no pause — behave as if the agent replied 'done' "
                               "right away and import whatever scene.blend / out_meshes already "
                               "hold (re-runs after the agent has done its work). "
                               "skip: no pause, no import; solve passes through untouched. "
                               "ATLAS_AGENT_MODE env var overrides the widget for the whole server."}),
                "paint_with": (["clean_plate", "source_photo"], {
                    "default": "clean_plate",
                    "tooltip": "Which projector paints the agent's meshes in the viewport. "
                               "clean_plate (default): they are OCCLUDED surfaces — the water "
                               "and hillside behind the foreground — so the viewport's "
                               "clean_plate input paints them. source_photo: facades the plate "
                               "shows. Per-mesh Blender property atlas_paint overrides."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # A handoff is an interaction; never serve it from the execution cache.
        return float("nan")

    # ------------------------------------------------------------------
    def _snapshots(self, snapshot_node_id: str) -> list[dict[str, Any]]:
        from atlas_camera.comfy.viewport_snapshot import SNAPSHOT_DIRNAME
        root = AH.output_root() / SNAPSHOT_DIRNAME
        if not root.is_dir():
            return []
        want = str(snapshot_node_id or "").strip()
        out = []
        for side in sorted(root.glob("viewport_*.json")):
            try:
                rec = json.loads(side.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if want and str(rec.get("node_id")) != want:
                continue
            out.append({"node_id": rec.get("node_id"), "files": rec.get("files"),
                        "stamp": rec.get("stamp"), "solve_fingerprint": rec.get("solve_fingerprint")})
        return out

    def handoff(self, solve, task, exchange_dir="", depth=None, snapshot_node_id="",
                tools_allowed="blender_mcp, blender_headless, atlas_mcp", blender_path="",
                timeout_s=300, on_timeout="continue", auto_import=True,
                expect_fingerprint=True, min_y_m=-0.05, poll_s=1.0, project=None,
                mode="wait", paint_with="clean_plate", unique_id=None):
        import copy

        from atlas_camera.blender.measured import IMPORT_SOURCE, solve_seed_fingerprint
        from atlas_camera.comfy.nodes_geometry import _blender_exchange_dir

        node_id = str(unique_id or "agent")
        solve_out = copy.deepcopy(solve)
        lines: list[str] = []
        exdir = _blender_exchange_dir(str(exchange_dir), tag="agent", project=project, create=False)

        seed_meta: dict[str, Any] = {}
        if exdir is not None and (exdir / "seed.json").is_file():
            try:
                seed_meta = json.loads((exdir / "seed.json").read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                seed_meta = {}
        params = (seed_meta.get("params") or {}) if isinstance(seed_meta, dict) else {}
        depth_meta = {}
        if depth is not None:
            md = getattr(depth, "metadata", None) or {}
            depth_meta = {"model_id": getattr(depth, "model_id", ""),
                          "predicted_focal_px": md.get("predicted_focal_px"),
                          "focal_source": md.get("focal_source")}
        cam = solve_out.camera
        intr, extr = cam.intrinsics, cam.extrinsics
        brief = {
            "task": str(task),
            "tools_allowed": [t.strip() for t in str(tools_allowed).split(",") if t.strip()],
            "timeout_s": int(timeout_s),
            "exchange_dir": str(exdir) if exdir is not None else "",
            "seed_json": str(exdir / "seed.json") if exdir is not None else "",
            "scene_blend": (str(exdir / "scene.blend")
                            if exdir is not None and (exdir / "scene.blend").is_file() else ""),
            "out_meshes_npz": str(exdir / "out_meshes.npz") if exdir is not None else "",
            "collections": {"model_under": "atlas_out", "reference_only": "atlas_reference"},
            "measured": params.get("measured") or {},
            "ground_y_m": params.get("ground_y_m"),
            "camera": {
                "fx_px": intr.fx_px, "fy_px": intr.fy_px, "cx_px": intr.cx_px, "cy_px": intr.cy_px,
                "image_width": intr.image_width, "image_height": intr.image_height,
                "position": list(extr.camera_position or []),
                "view_matrix": [list(r) for r in (extr.camera_view_matrix or [])],
                "convention": "Atlas: right-handed Y-up, camera looks down -Z; Blender seed is Z-up (T rows (1,0,0),(0,0,-1),(0,1,0))",
            },
            "depth": depth_meta,
            "snapshots": self._snapshots(snapshot_node_id),
            "solve_fingerprint": solve_seed_fingerprint(solve_out),
            "expect_fingerprint": bool(expect_fingerprint),
            "return_contract": {
                "blend": "save the edited .blend (default: the scene_blend path) — meshes under "
                         "atlas_out are exported by export_meshes.py and appended to the solve",
                "npz": "or write out_meshes.npz/.json yourself (Blender axes) into exchange_dir",
                "resume": "then resume with status done|skip|fail, a short reply, and blend_file",
            },
        }
        # Override switch: env beats widget; neither pauses when not "wait".
        env_mode = os.environ.get("ATLAS_AGENT_MODE", "").strip().lower()
        eff_mode = env_mode if env_mode in ("wait", "import_only", "skip") else str(mode or "wait")
        if eff_mode == "skip":
            return (solve_out, "", "", f"mode=skip{' (ATLAS_AGENT_MODE)' if env_mode else ''}: "
                                       "no pause, no import; solve passed through")
        written = AH.write_brief(node_id, brief)
        brief_path = str(AH.agent_dir(node_id) / AH.BRIEF_NAME)
        t0 = time.time()
        if eff_mode == "import_only":
            lines.append(f"mode=import_only{' (ATLAS_AGENT_MODE)' if env_mode else ''}: brief "
                         f"written to {brief_path}, NOT waiting — importing the current scene")
            rec = {"status": "done", "reply": "", "blend_file": brief["scene_blend"]}
        else:
            lines.append(f"brief written: {brief_path} (token {written['token']}); waiting up to "
                         f"{int(timeout_s)} s for a resume")
            rec = AH.wait_for_resume(node_id, written["token"], timeout_s=float(timeout_s),
                                     poll_s=float(poll_s))
        waited = time.time() - t0
        status = str(rec.get("status") or "timeout")
        reply = str(rec.get("reply") or "")
        lines.append(f"resume: status={status} after {waited:.0f} s"
                     + (f"; reply: {reply}" if reply else ""))
        if status == "timeout":
            if str(on_timeout) == "fail":
                raise RuntimeError(f"AtlasAgentHandoff: no resume within {int(timeout_s)} s "
                                   f"(brief: {brief_path})")
            return (solve_out, "", brief_path, "\n".join([*lines, "TIMEOUT — continued without agent input"]))
        if status in ("skip", "fail"):
            return (solve_out, reply, brief_path, "\n".join([*lines, f"agent said {status}; solve passed through"]))

        # done → optional import of the agent's Blender work.
        n_added = 0
        if auto_import:
            imp_dir = exdir if exdir is not None else AH.agent_dir(node_id) / "exchange"
            imp_dir.mkdir(parents=True, exist_ok=True)
            # A resume can arrive over the unauthenticated HTTP route, and
            # blend_file is the field that turns it into a Blender subprocess.
            # Constrain it to the exchange the brief published before acting.
            claimed, refusal = AH.resolve_blend_file(
                rec.get("blend_file"), node_id, exchange_dir=exdir)
            if refusal:
                lines.append(refusal)
            blend = str(claimed) if claimed is not None else brief["scene_blend"]
            try:
                if blend and os.path.isfile(blend):
                    from atlas_camera.blender import run_recipe
                    rep = run_recipe("export_meshes.py", imp_dir, blender_path=str(blender_path),
                                     timeout_s=600, blend_file=blend)
                    lines.append(f"export_meshes.py: {rep.get('meshes_out', '?')} meshes from "
                                 f"{os.path.basename(blend)} ({rep.get('selection_rule', '?')})")
                elif not (imp_dir / "out_meshes.npz").is_file():
                    lines.append("auto_import: no blend_file and no out_meshes.npz — nothing to import")
                    return (solve_out, reply, brief_path, "\n".join(lines))
                from atlas_camera.blender import read_meshes
                got = read_meshes(imp_dir)
                seed_fp = params.get("solve_fingerprint")
                cur_fp = brief["solve_fingerprint"]
                if expect_fingerprint and seed_fp and seed_fp != cur_fp:
                    lines.append(
                        f"REFUSED import: seed fingerprint {seed_fp} != current {cur_fp}. "
                        "If the camera did not change, a node between the brief and "
                        "here moved geometry out of the primary scene — "
                        "AtlasPlateLayer's move_from_primary does that to drawn "
                        "planes (Blender-sourced meshes are already exempt).")
                    return (solve_out, reply, brief_path, "\n".join(lines))
                from atlas_camera.comfy.nodes_geometry import (
                    _append_blender_meshes, _measured_floor,
                )
                accepted, _ = _append_blender_meshes(
                    solve_out, got["meshes"], source=IMPORT_SOURCE, name_prefix="agent",
                    min_y_m=_measured_floor(params, min_y_m),
                    max_radius_m=0.0,
                    extra_tags={"exchange_dir": str(imp_dir), "agent_reply": reply[:200],
                                "agent_token": written["token"]},
                    lines=lines, paint_with=str(paint_with or "clean_plate"))
                n_added = len(accepted)
            except RuntimeError as exc:
                lines.append(f"auto_import FAILED — {exc}")
        solve_out.projection_scene.debug_metadata["agent_handoff"] = {
            "brief": brief_path, "token": written["token"], "status": status,
            "reply": reply, "waited_s": round(waited, 1), "meshes_added": n_added,
        }
        return (solve_out, reply, brief_path, "\n".join(lines))
