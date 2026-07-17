"""The Atlas Camera MCP server (v1) — stdio, thin, HTTP-only.

Exposes a running ComfyUI (with the Atlas node pack) to any MCP-capable
assistant. Design: docs/dev/archive/atlas_mcp_server_plan.md — every tool is an
operation the 2026-07 verification sessions actually performed. This process
never imports torch/numpy; ComfyUI stays the execution engine.

Run:      python -m atlas_camera.mcp
Config:   COMFY_HOST   (default 127.0.0.1:8188)
          ATLAS_REPO   (optional — repo checkout for doc-backed resources)

Deliberately NOT in v1: atlas_bake_camera_move (the one browser-bound
operation — ⏺ Bake runs in the viewport's WebGL context; drive it via a real
browser, e.g. Playwright, per the plan doc).
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        "The Atlas MCP server needs the 'mcp' package.\n"
        "Install it with:  pip install atlas-camera[mcp]   (or: pip install mcp)"
    ) from exc

from . import comfy_http as C

HOST = os.environ.get("COMFY_HOST", C.DEFAULT_HOST)
REPO = os.environ.get("ATLAS_REPO", "")

# Third-party packs the showcase workflows lean on (probed by atlas_health).
_KNOWN_DEPS = {
    "SAM3Segment": "ComfyUI-RMBG (SAM3 segmentation)",
    "INPAINT_InpaintWithModel": "comfyui-inpaint-nodes (+ big-lama.pt)",
    "OCIORead": "ComfyUI-OCIO (float EXR / ACEScg path)",
    "VideoCombinePlus": "Comfyui_VideoCombine_Plus (baked-move MP4s)",
    "ShowText|pysssss": "pysssss custom-scripts (report display)",
}

_EXPERIMENTAL = ("AtlasPredictHiddenGeometry", "AtlasRenderFix")

mcp = FastMCP(
    "atlas-camera",
    instructions=(
        "Tools for driving Atlas Camera (single-image camera recovery + "
        "matte-painting projection) on a local ComfyUI. Start with "
        "atlas_health; read the atlas://calibration resource before choosing "
        "depth models or band settings. Shipped workflows close their solve "
        "gates — atlas_run_workflow opens them by default."
    ),
)


def _load_ui(workflow_path: str) -> dict:
    p = pathlib.Path(workflow_path)
    if not p.is_file() and REPO:
        p = pathlib.Path(REPO) / workflow_path
    return json.loads(p.read_text(encoding="utf-8"))


@mcp.tool()
def atlas_health() -> str:
    """Probe the ComfyUI server: version/VRAM, which Atlas nodes are
    registered (catches a missing ATLAS_EXPERIMENTAL=1), and which known
    third-party packs are absent."""
    try:
        stats = C.http_json(f"http://{HOST}/system_stats", timeout=10)
    except Exception as exc:
        return json.dumps({"ok": False, "host": HOST,
                           "error": f"ComfyUI not reachable: {exc}"})
    oi = C.fetch_object_info(HOST)
    atlas = sorted(k for k in oi if k.startswith("Atlas"))
    dev = stats["devices"][0] if stats.get("devices") else {}
    return json.dumps({
        "ok": True,
        "host": HOST,
        "comfyui": stats["system"].get("comfyui_version"),
        "vram_free_gb": round(dev.get("vram_free", 0) / 2**30, 1),
        "atlas_nodes": len(atlas),
        "experimental_registered": all(n in oi for n in _EXPERIMENTAL),
        "missing_third_party": [f"{k} — {v}" for k, v in _KNOWN_DEPS.items()
                                if k not in oi],
    }, indent=1)


@mcp.tool()
def atlas_validate_workflow(workflow_path: str) -> str:
    """Validate a UI-format workflow against the live server's node
    definitions: positional-widget drift, link integrity, widget ranges,
    KJ-rail resolution. Returns the error/warning lists."""
    ui = _load_ui(workflow_path)
    oi = C.fetch_object_info(HOST)
    errs, warns = C.validate_ui(ui, oi)
    return json.dumps({"valid": not errs, "errors": errs, "warnings": warns}, indent=1)


@mcp.tool()
def atlas_run_workflow(workflow_path: str, overrides: dict | None = None,
                       open_gates: bool = True, timeout: int = 1800) -> str:
    """Flatten a UI-format workflow (KJ rails resolved, muted/bypassed nodes
    handled) and run it to completion on the server. `open_gates` (default
    True) sets proceed=True on every AtlasSolveGate — shipped workflows close
    them. `overrides` is {"<nodeId>.<input>": value}. Returns completion
    status, verbatim node errors, and which nodes produced outputs."""
    ui = _load_ui(workflow_path)
    oi = C.fetch_object_info(HOST)
    api = C.ui_to_api(ui, oi)
    ov = dict(overrides or {})
    if open_gates:
        for k, v in C.gate_overrides(ui).items():
            ov.setdefault(k, v)
    applied = C.apply_overrides(api, ov)
    result = C.queue_and_wait(api, HOST, timeout=timeout)
    result["overrides_applied"] = applied
    result["executing_nodes"] = len(api)
    return json.dumps(result, indent=1)


@mcp.tool()
def atlas_solve_image(image_path: str, method: str = "learned",
                      depth_model: str = "", camera_height_m: float = 0.0,
                      timeout: int = 900) -> str:
    """Solve the camera for a single image. Uploads the image to ComfyUI,
    runs LoadImage → solve (learned GeoCalib by default; method='vp' for the
    classical vanishing-point solve) → AtlasExportSolveJSON, and returns the
    solve summary (focal, FOV, height, confidence, scale_source).
    `camera_height_m` > 0 adds an AtlasScaleOverride (the elevated-vantage
    fix — see atlas://calibration)."""
    name = C.upload_image(image_path, HOST)
    out_rel = f"atlas_exports/mcp_solves/{pathlib.Path(name).stem}_solve.json"
    solve_cls = ("AtlasLearnedSolveFromImage" if method == "learned"
                 else "AtlasSolveFromImage")
    api = {
        "1": {"class_type": "LoadImage", "inputs": {"image": name}},
        "2": {"class_type": solve_cls, "inputs": {"image": ["1", 0]}},
    }
    if method == "learned" and depth_model:
        api["2"]["inputs"]["depth_model"] = depth_model
    solve_ref = ["2", 0]
    if camera_height_m > 0:
        api["3"] = {"class_type": "AtlasScaleOverride",
                    "inputs": {"solve": solve_ref,
                               "camera_height_m": float(camera_height_m)}}
        solve_ref = ["3", 0]
    api["9"] = {"class_type": "AtlasExportSolveJSON",
                "inputs": {"solve": solve_ref, "output_path": out_rel}}
    result = C.queue_and_wait(api, HOST, timeout=timeout)
    if not result["completed"]:
        return json.dumps(result, indent=1)
    summary = {"completed": True, "solve_json": out_rel}
    comfy_dir = os.environ.get("COMFY_DIR", "")
    p = pathlib.Path(comfy_dir) / out_rel if comfy_dir else pathlib.Path(out_rel)
    if p.is_file():
        s = json.loads(p.read_text(encoding="utf-8"))
        cam = s.get("camera", {})
        intr = cam.get("intrinsics", {})
        summary.update({
            "confidence": s.get("confidence"),
            "source_method": s.get("source_method"),
            "focal_mm": intr.get("focal_length_mm"),
            "image_wh": [intr.get("image_width"), intr.get("image_height")],
            "camera_position": cam.get("extrinsics", {}).get("camera_position"),
            "scale_source": (s.get("debug_metadata") or {}).get("scale_source"),
        })
    else:
        summary["note"] = ("solve JSON written server-side; set COMFY_DIR to "
                           "the ComfyUI folder so the summary can be read back")
    return json.dumps(summary, indent=1)


@mcp.tool()
def atlas_read_debug_report(json_path: str = "atlas_debug/master_debug.json") -> str:
    """Read a 🔍 AtlasDebugReport JSON (the stable-path full-stack diagnostic:
    camera, per-layer geometry/bands/mattes, red flags). Relative paths
    resolve against COMFY_DIR."""
    comfy_dir = os.environ.get("COMFY_DIR", "")
    p = pathlib.Path(json_path)
    if not p.is_file() and comfy_dir:
        p = pathlib.Path(comfy_dir) / json_path
    if not p.is_file():
        return json.dumps({"ok": False, "error": f"not found: {json_path} "
                           "(set COMFY_DIR, or pass an absolute path)"})
    return p.read_text(encoding="utf-8")


@mcp.tool()
def atlas_inspect_viewport(node_id: int) -> str:
    """Summarize a viewport's live payload (GET /atlas/camera_data/{id}):
    camera meta + every projection layer's name/priority/band/verts/mattes.
    The census that catches empty layers and band drift without a browser."""
    d = C.http_json(f"http://{HOST}/atlas/camera_data/{node_id}", timeout=60)
    layers = []
    for s in d.get("projection_sources") or []:
        verts = sum(len(p.get("vertices") or []) for p in s.get("proxy_geometry") or []) // 3
        layers.append({
            "name": s.get("name"), "priority": s.get("priority"),
            "band_m": [s.get("near_m"), s.get("far_m")],
            "band_geometry": s.get("band_geometry"), "verts": verts,
            "matte": bool(s.get("mask_b64")),
            "normal_map": bool(s.get("normal_map_b64")),
            "hidden_provenance": bool(s.get("hidden_mask_b64")),
        })
    prims = [{"name": p.get("name"), "type": p.get("type")}
             for p in d.get("proxy_geometry") or []]
    return json.dumps({
        "camera_meta": d.get("camera_meta"),
        "camera_position": d.get("camera_position"),
        "render_wh": [d.get("target_width"), d.get("target_height")],
        "primary_geometry": prims,
        "layers": layers,
    }, indent=1)


@mcp.tool()
def atlas_export_scene(solve_json_path: str, formats: list[str],
                       output_dir: str = "atlas_exports/mcp_export",
                       timeout: int = 1800) -> str:
    """Export a saved solve (AtlasExportSolveJSON output, server-relative
    path) to DCC formats. `formats` ⊆ ["nuke", "nuke_layers", "maya_layers",
    "maya_review", "blender", "usd", "review_package"]. Layer exports need a
    solve that carries projection_sources."""
    fmap = {
        "nuke": "AtlasExportNuke", "nuke_layers": "AtlasExportNukeLayers",
        "maya_layers": "AtlasExportMayaLayers",
        "maya_review": "AtlasExportMayaReviewScene",
        "blender": "AtlasExportBlender", "usd": "AtlasExportUSD",
        "review_package": "AtlasExportReviewPackage",
    }
    bad = [f for f in formats if f not in fmap]
    if bad:
        return json.dumps({"ok": False, "error": f"unknown formats {bad}; "
                           f"choose from {sorted(fmap)}"})
    api = {"1": {"class_type": "AtlasLoadSolveJSON",
                 "inputs": {"json_path": solve_json_path}}}
    for i, f in enumerate(formats, start=2):
        api[str(i)] = {"class_type": fmap[f],
                       "inputs": {"solve": ["1", 0], "output_dir": output_dir}}
        if f == "review_package":
            api[str(i)]["inputs"] = {"solve": ["1", 0], "output_dir": output_dir}
    result = C.queue_and_wait(api, HOST, timeout=timeout)
    result["output_dir"] = output_dir
    return json.dumps(result, indent=1)


@mcp.tool()
def atlas_node_catalog(name_filter: str = "") -> str:
    """List the Atlas nodes registered on the server with their input names
    (widgets vs links) and outputs. `name_filter` substring-matches."""
    oi = C.fetch_object_info(HOST)
    out = {}
    for k in sorted(oi):
        if not k.startswith("Atlas"):
            continue
        if name_filter and name_filter.lower() not in k.lower():
            continue
        widgets, links_ = [], []
        for name, spec in C.spec_items(oi, k):
            (widgets if C.is_widget(spec) else links_).append(name)
        out[k] = {"widgets": widgets, "link_inputs": links_,
                  "outputs": list(oi[k].get("output_name", []))}
    return json.dumps(out, indent=1)


# ── Resources ───────────────────────────────────────────────────────────────

_CALIBRATION = """\
Atlas per-scene calibration doctrine (distilled from the run-verified showcase):

DEPTH MODEL — per shot, never global:
  exterior → depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
  interior → Ruicheng/moge-2-vitl-normal (the specialist; also carries relight
             normals) or V2-Indoor. MoGe's far field runs away outdoors and it
             culls sky — never use it for a sky dome.

METRIC SCALE — measured, not assumed, and often WRONG on elevated vantages:
  scale_source=assumed_default (1.6 m) fires on AI vistas AND real photos shot
  from height (a street-level ground fit breaks over cars). Fix with
  AtlasScaleOverride camera_height_m (temple city ≈16 m) or a known-size
  reference (AtlasReferenceScaleSolve). THE SKY-RISE DOCTRINE (NYC birdseye,
  2026-07-17): on any plate with buildings, COUNT THE STOREYS — pick one fully
  visible base-to-roof, count levels x 3.5 m, and give
  AtlasReferenceScaleSolve its bbox with height_override_m
  (reference_id=building_story_3m). On the NYC plate the counted 5-storey
  tenement (17.5 m, bbox 3820,1775-4470,2480) locks the camera at 63.7 m,
  conf 0.96, scale_source=reference_object — the photographer recalled
  70-100 m and the original eyeballed 25 m dial was ~2.5x small. Lesson: an
  eyeballed dial can be off severalfold while looking plausible (projection
  is angular); counted geometry beats memory, and both beat a guess.

RELIEF / BANDS:
  generic relief: grid 128 / edge_rel 0.5 · organic canopy: grid 512 / 1.0
  band clean-plate layers: grid 384 / edge_rel 1.5 · interiors: sky_heuristic
  False + max_edge_factor 40-80. Band priorities are FARTHEST-HIGHEST; the
  edge-extend smear belongs on the layers BEHIND (frontmost keeps a clean cut).
  One AtlasDepthBandSplit (absolute metres) may feed both sides of a split —
  percentile bands with scoped excludes need band_ref_mask to avoid drift.

SAM3 PROMPTS: SIMPLE NOUN PHRASES joined with "and" — a comma-separated prompt
  silently returns an EMPTY mask, and a relational clause silently DROPS
  objects (measured 2026-07-17: "rusty car with its shadow and fallen sign"
  returned the car only; "rusty car and fallen sign" returned both). Shadows
  are not segmentable objects — cover a contact shadow with GrowMask
  (~24 px at 4K), never the prompt.

GATES: AtlasSolveGate ships closed (proceed=False); AtlasAssessImage
  auto_continue=True is advisory flow; approvals are fingerprinted per image.

ROLL: GeoCalib gravity can drift a few degrees on AI plates with no horizon —
  AtlasRollTrim (positive = scene turns counter-clockwise on screen).

X-RAY (experimental): always wire restrict_mask (a SAM3 segment of the
  occluder or the foreground band's layer_mask) — unrestricted substitution
  covered 50%+ of frame. Architecture is LaRI's strong domain.
"""


@mcp.resource("atlas://calibration")
def calibration() -> str:
    """The per-scene-type settings doctrine, distilled from live runs."""
    return _CALIBRATION


@mcp.resource("atlas://gates")
def gates() -> str:
    """The gate/approval state table (docs/dev/gate_state_table.md when
    ATLAS_REPO is set; a summary otherwise)."""
    if REPO:
        p = pathlib.Path(REPO) / "docs/dev/gate_state_table.md"
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return ("AtlasAssessImage: auto_continue=True → advisory (flows same "
            "queue); proceed persists but approved_for fingerprints the image "
            "— a new image re-arms the gate. AtlasSolveGate: proceed=False "
            "ships closed; ✅ approve stamps a solve+image fingerprint. 📐 "
            "patch outputs return ExecutionBlocker until an extraction exists "
            "for the CURRENT solve fingerprint.")


@mcp.resource("atlas://schemas/solve")
def solve_schema() -> str:
    """Shape of an Atlas solve JSON (AtlasExportSolveJSON output)."""
    return json.dumps({
        "camera": {"intrinsics": "fx/fy/cx/cy px, focal_length_mm, sensor mm, image_width/height",
                   "extrinsics": "camera_position, camera_rotation_matrix (3x3 cam→world), "
                                 "camera_view_matrix + camera_world_matrix (4x4 row-major; "
                                 "world math ALWAYS via the 4x4 view matrix)"},
        "confidence": "0..1", "source_method": "solver id string",
        "vanishing_points / horizon_line": "present on the VP path; VPs empty on the learned path",
        "projection_scene.proxy_geometry": "derived primitives + relief mesh (PROXY_ROLE)",
        "projection_sources": "layered ProjectionSource list (per-layer camera, plate, matte, geometry)",
        "debug_metadata": "scale_source, roll_trim_deg, proxy_derivation, ...",
    }, indent=1)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
