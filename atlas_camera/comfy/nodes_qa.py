"""Terminal QA nodes for agentic and headless Atlas Camera runs."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any


_OUTPUT_ASSESS_CACHE: dict[str, Any] = {}
_OUTPUT_ASSESS_CACHE_MAX = 8
_OUTPUT_ASSESS_CACHE_CONTRACT = "terminal_assessor_v2_multi_evidence_2"


def _matte_coverage(mask_b64: str | None) -> float | None:
    if not mask_b64:
        return None
    try:
        import numpy as np
        from PIL import Image
        raw = base64.b64decode(mask_b64.split(",", 1)[-1])
        arr = np.asarray(Image.open(io.BytesIO(raw)).convert("L"))
        return round(float((arr > 127).mean()), 4)
    except Exception:
        return None


def build_output_solve_summary(solve: Any, depth: Any = None) -> dict[str, Any]:
    """Build the deterministic half of the terminal report."""

    from atlas_camera.core.scene_health import evaluate_scene_health

    health = evaluate_scene_health(
        solve, depth, matte_coverage_fn=_matte_coverage)
    primary = []
    scene = getattr(solve, "projection_scene", None)
    for primitive in (getattr(scene, "proxy_geometry", None) or []):
        meta = getattr(primitive, "metadata", None) or {}
        primary.append({
            "name": getattr(primitive, "name", ""),
            "type": getattr(primitive, "primitive_type", ""),
            "n_vertices": meta.get("n_vertices"),
            "n_faces": meta.get("n_faces"),
            # Gate on the WHOLE-QUAD figure whenever a sub-quad cut ran: cut
            # cells emit faces from partial quads, which deflates the plain
            # ratio and would make `torn_excessive` quietly more permissive on
            # exactly the meshes doing something clever. The raw value is kept
            # beside it so the two are never confused for one another.
            "torn_fraction": (meta.get("torn_fraction_whole_quad")
                              if meta.get("torn_fraction_whole_quad") is not None
                              else meta.get("torn_fraction")),
            "torn_fraction_emitted": meta.get("torn_fraction"),
            "stretch_ratio_p95": meta.get("stretch_ratio_p95"),
        })
    result = {
        "scene_health": health.to_dict(),
        "primary_proxy_geometry": primary,
        "projection_source_count": len(getattr(solve, "projection_sources", None) or []),
        "has_shot_camera": bool(getattr(solve, "shot_cam", None)),
        "field_semantics": {
            "scene_health_flags": (
                "calibrated findings with authoritative severity; do not upgrade raw "
                "telemetry to a defect when no corresponding flag was emitted"),
            "per_layer.matte_coverage": (
                "fraction of the full frame intentionally assigned to that depth band; "
                "small foreground/midground fractions are normal, not missing final coverage; "
                "only near_empty_matte is a calibrated problem"),
            "per_layer.torn_fraction": (
                "raw grid-quad removal fraction; expected to be high on deliberately "
                "band-clipped layers; only torn_excessive is a calibrated problem"),
            "headless_evidence.coverage_fraction": (
                "union coverage of the retained canonical output; this is the field to use "
                "for canonical missing-output claims"),
            "orbit_coverage": (
                "geometry-only small-orbit census; not a rendered WebGL occlusion result"),
            "source_comparison": (
                "exposure-tolerant canonical output/reference structure comparison; severe "
                "drift is a release failure unless explicitly artist-approved"),
        },
    }
    try:
        from atlas_camera.comfy.headless_evidence import orbit_coverage_summary
        orbit = orbit_coverage_summary(solve)
    except (ImportError, ValueError, TypeError, AttributeError, IndexError,
            ArithmeticError) as exc:
        orbit = {"status": "unavailable", "reason": str(exc)}
    if orbit is not None:
        result["orbit_coverage"] = orbit
    return result


def _image_stats(image: Any) -> dict[str, Any]:
    """Small, JSON-safe telemetry used to reject an unbaked black pass."""

    import numpy as np

    arr = image.detach().cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[0] < 1 or arr.shape[1] < 1:
        return {"usable": False, "blank": True, "reason": "invalid_shape",
                "shape": list(arr.shape)}
    finite = np.isfinite(arr)
    if not finite.any():
        return {"usable": False, "blank": True, "reason": "no_finite_pixels",
                "shape": list(arr.shape)}
    values = arr[finite]
    lo, hi = float(values.min()), float(values.max())
    mean, std = float(values.mean()), float(values.std())
    blank = bool((hi - lo) < 1e-5 and abs(mean) < 1e-5)
    return {
        "usable": not blank,
        "blank": blank,
        "reason": "all_zero_unbaked_pass" if blank else "",
        "shape": list(arr.shape),
        "channels": int(arr.shape[2]),
        "min": round(lo, 6), "max": round(hi, 6),
        "mean": round(mean, 6), "std": round(std, 6),
        "alpha_policy": (
            "inspection only; graph pixels and alpha are not modified or color-transformed"),
    }


def _write_tensor_png(image: Any, path: str) -> str:
    """Write straight RGB(A) without colour-transforming or premultiplying."""

    import numpy as np
    from PIL import Image

    arr = image.detach().cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    channels = 4 if arr.shape[-1] >= 4 else 3
    pixels = np.clip(arr[..., :channels], 0.0, 1.0)
    u8 = np.rint(pixels * 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(u8, mode="RGBA" if channels == 4 else "RGB").save(path)
    return path


def _write_mask_png(mask: Any, path: str) -> str:
    import numpy as np
    from PIL import Image

    arr = mask.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    u8 = np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(u8, mode="L").save(path)
    return path


def _temporary_tensor_png(image: Any) -> str:
    handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    handle.close()
    return _write_tensor_png(image, handle.name)


def _evidence_paths(json_path: str) -> tuple[str, str, str]:
    path = Path(json_path)
    stem = path.with_suffix("")
    return (str(stem) + "_evidence.png", str(stem) + "_coverage.png",
            str(stem) + "_source_reference.png")


def _safe_external_summary(value: str) -> Any:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text[:12000]


def _enforce_provenance(payload: dict[str, Any], provenance: str) -> dict[str, Any]:
    """Constrain VLM claims to what the supplied evidence can actually prove."""

    result = json.loads(json.dumps(payload))
    if provenance == "camera_view":
        return result
    checks = result.setdefault("checks", {})
    if provenance == "headless_projection_reconstruction":
        # The canonical reconstruction is a real graph-output composite, so it
        # can prove framing, coverage at the recovered pose, clean-plate seams,
        # and colour continuity. It cannot prove interactive orbit occlusion.
        result["verdict"] = (
            "fail" if str(result.get("verdict", "")).lower() == "fail"
            else "inconclusive")
        names = ("occlusion_tearing",)
        prefix = (
            "Canonical headless projection evidence cannot prove orbit/grazing "
            "occlusion; use deterministic orbit_coverage or a browser/DCC render.")
    else:
        result["verdict"] = "inconclusive"
        names = ("camera_match", "projection_edges", "occlusion_tearing",
                 "layer_inpaint")
        prefix = (
            "Browser/WebGL camera-view pass was not baked; source plate fallback only.")
    for name in names:
        check = checks.setdefault(name, {})
        check["status"] = "inconclusive"
        evidence = str(check.get("evidence") or "").strip()
        check["evidence"] = f"{prefix} {evidence}".strip()
    return result


def _overall_verdict(*, health_level: str, vlm_ok: bool,
                     visual_verdict: str, provenance: str) -> str:
    if health_level == "fail":
        return "fail"
    if (not vlm_ok or provenance not in {
            "camera_view", "headless_projection_reconstruction"}):
        return "inconclusive" if health_level == "pass" else "warn"
    rank = {"pass": 0, "inconclusive": 1, "warn": 2, "fail": 3}
    return max((health_level, visual_verdict), key=lambda value: rank.get(value, 1))


def _deterministic_output_floor(provenance: str,
                                summary: dict[str, Any]) -> str:
    """Minimum terminal verdict proved without a VLM response."""

    if provenance != "headless_projection_reconstruction":
        return "pass"
    evidence = summary.get("headless_evidence") or {}
    coverage = evidence.get("coverage_fraction")
    result = "pass"
    if isinstance(coverage, (int, float)):
        if float(coverage) < 0.95:
            result = "fail"
        elif float(coverage) < 0.995:
            result = "warn"
    comparison = summary.get("source_comparison") or {}
    if int(summary.get("projection_source_count") or 0) > 0:
        if comparison.get("status") == "severe":
            result = "fail"
        elif comparison.get("status") == "warn" and result == "pass":
            result = "warn"
    return result


def _enforce_deterministic_evidence(payload: dict[str, Any], provenance: str,
                                    summary: dict[str, Any]) -> dict[str, Any]:
    """Ground model claims in calibrated flags and measured output coverage."""

    if provenance != "headless_projection_reconstruction":
        return payload
    result = json.loads(json.dumps(payload))
    checks = result.setdefault("checks", {})
    corrections = result.setdefault("grounding_corrections", [])
    flags = ((summary.get("scene_health") or {}).get("flags") or [])
    flag_codes = {str(item.get("code") or "") for item in flags
                  if isinstance(item, dict)}
    evidence = summary.get("headless_evidence") or {}
    coverage = evidence.get("coverage_fraction")
    grounded_failures = []
    if isinstance(coverage, (int, float)) and float(coverage) < 0.995:
        status = "fail" if float(coverage) < 0.95 else "warn"
        missing = 100.0 * (1.0 - float(coverage))
        check = checks.setdefault("projection_edges", {})
        ranks = {"pass": 0, "inconclusive": 0, "warn": 1, "fail": 2}
        if ranks.get(status, 0) >= ranks.get(str(check.get("status", "pass")), 0):
            check["status"] = status
            check["evidence"] = (
                f"Retained coverage matte measures {missing:.2f}% canonical output "
                "holes (white = covered, black = missing).")
        if status == "fail":
            result["verdict"] = "fail"
            grounded_failures.append(
                f"canonical projection coverage fails with {missing:.2f}% holes")
        issue_severity = "fail" if status == "fail" else "warn"
        result.setdefault("issues", []).append({
            "severity": issue_severity,
            "category": "measured_canonical_coverage",
            "evidence": f"Coverage matte reports {missing:.2f}% missing pixels.",
            "action": "Repair the relief/matte coverage, then rerun terminal QA.",
        })
        corrections.append(
            f"projection_edges constrained by measured union coverage {float(coverage):.6f}")

    # A single relief has no generated clean-plate layer to inspect. Keep a
    # projection defect in its proper check rather than mislabelling it inpaint.
    if int(summary.get("projection_source_count") or 0) == 0:
        check = checks.setdefault("layer_inpaint", {})
        check["status"] = "inconclusive"
        check["evidence"] = (
            "Not applicable: this solve has no generated projection/inpaint layers.")

    comparison = summary.get("source_comparison") or {}
    if (int(summary.get("projection_source_count") or 0) > 0
            and comparison.get("status") in {"warn", "severe"}):
        severe = comparison.get("status") == "severe"
        status = "fail" if severe else "warn"
        check = checks.setdefault("layer_inpaint", {})
        check["status"] = status
        check["evidence"] = (
            "Deterministic source comparison: "
            f"luma correlation {comparison.get('luma_correlation')}, edge correlation "
            f"{comparison.get('edge_correlation')}, RGB MAE "
            f"{comparison.get('rgb_mean_absolute_error')}, changed-pixel fraction "
            f"{comparison.get('changed_fraction_gt_0_15')}. "
            "The retained output/reference pair requires visual review for unintended "
            "scene or camera replacement.")
        result.setdefault("issues", []).append({
            "severity": status,
            "category": "source_structure_drift",
            "evidence": check["evidence"],
            "action": (
                "Reduce denoise/rewrite scope or approve the broad clean-plate change "
                "explicitly after comparing the retained source and output images."),
        })
        if severe:
            result["verdict"] = "fail"
            grounded_failures.append("severe source/output structural drift")

    unsupported_terms = {
        "stretch_ratio_p95": "stretch_excessive",
        "torn_fraction": "torn_excessive",
        "matte_coverage": "near_empty_matte",
    }
    kept_issues = []
    for issue in result.get("issues") or []:
        if not isinstance(issue, dict):
            kept_issues.append(issue)
            continue
        text = " ".join(str(issue.get(key) or "")
                        for key in ("category", "evidence", "action")).casefold()
        unsupported = [term for term, flag in unsupported_terms.items()
                       if term in text and flag not in flag_codes]
        if ("stretch" in text and ("p95" in text or "stretch ratio" in text)
                and "stretch_excessive" not in flag_codes):
            unsupported.append("uncalibrated stretch p95")
        if unsupported:
            corrections.append(
                "discarded uncalibrated raw-metric issue: " + ", ".join(unsupported))
            continue
        kept_issues.append(issue)
    result["issues"] = kept_issues
    if grounded_failures:
        raw_summary = str(result.get("summary") or "").strip()
        if raw_summary:
            result["model_summary_raw"] = raw_summary
        result["summary"] = (
            "Grounded terminal failure: " + "; ".join(grounded_failures)
            + ". Orbit/grazing occlusion remains visually inconclusive in the "
              "canonical headless reconstruction.")
    return result


class AtlasAssessOutput:
    """Terminal VLM + deterministic solve-health report for automation.

    Wire the final viewport's ``shaded`` image and final ``solve`` here. In a
    browser session, click **Render Proxy Passes** before queueing to provide
    the exact WebGL camera-view render. During a purely headless queue that
    socket is all-zero; the node reconstructs a canonical camera view from the
    solve's actual projected plates, mattes, and relief topology. If the solve
    cannot support that reconstruction, it falls back explicitly to the source
    plate and marks projection-specific checks *inconclusive*.

    The stable JSON is written even when the VLM is disabled or unavailable,
    alongside the exact assessed PNG and its coverage matte. The IMAGE output
    is the same evidence sent to the VLM. The node never gates or changes the
    upstream workflow image.
    """

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING",
                    "IMAGE", "STRING")
    RETURN_NAMES = ("report", "assessment_json", "json_path", "verdict",
                    "image_provenance", "assessed_image", "evidence_path")
    FUNCTION = "assess"
    CATEGORY = "Atlas"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_view": ("IMAGE", {"tooltip":
                    "Final camera-view image. AtlasBlockoutViewport.shaded is valid only "
                    "after Render Proxy Passes; the node detects its headless all-zero state."}),
                "solve": ("ATLAS_SOLVE",),
            },
            "optional": {
                "source_image": ("IMAGE", {"tooltip":
                    "Honest fallback for an unbaked/blank viewport pass. Projection checks "
                    "remain inconclusive when this image is used."}),
                "depth": ("ATLAS_DEPTH_MAP",),
                "solve_summary": ("STRING", {"forceInput": True,
                    "tooltip": "Optional upstream debug/export summary appended to the "
                               "internally derived scene-health JSON."}),
                "enabled": ("BOOLEAN", {"default": False,
                    "tooltip": "Run the VLM. OFF still writes deterministic solve health. "
                               "atlas_run_workflow(..., assess_output=True) enables every "
                               "terminal assessor for agentic runs."}),
                "provider": (["ollama", "lmstudio", "llamacpp", "openai"],
                    {"default": "lmstudio"}),
                "model": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": ""}),
                "extra_instructions": ("STRING", {"default": "", "multiline": True}),
                "file_path": ("STRING", {
                    "default": "atlas_debug/output_assessment.json",
                    "tooltip": "Stable path, relative to ComfyUI's working directory."}),
                "api_key": ("STRING", {"default": "",
                    "tooltip": "Saved in the workflow; prefer OPENAI_API_KEY."}),
                "offload_model": ("BOOLEAN", {"default": True}),
                "fallback_to_source": ("BOOLEAN", {"default": True}),
            },
        }

    def assess(self, camera_view, solve, source_image=None, depth=None,
               solve_summary="", enabled=False, provider="lmstudio", model="",
               base_url="", extra_instructions="",
               file_path="atlas_debug/output_assessment.json", api_key="",
               offload_model=True, fallback_to_source=True, **_extra):
        import datetime

        path = os.path.abspath(file_path or "atlas_debug/output_assessment.json")
        (requested_evidence_path, requested_coverage_path,
         requested_source_reference_path) = _evidence_paths(path)
        camera_stats = _image_stats(camera_view)
        selected = camera_view
        provenance = "camera_view"
        headless_evidence = None
        coverage_mask = None
        reconstruction_error = ""
        source_stats = _image_stats(source_image) if source_image is not None else None
        if camera_stats.get("blank"):
            try:
                from atlas_camera.comfy.headless_evidence import reconstruct_camera_view
                reconstructed = reconstruct_camera_view(solve, source_image)
            except (ImportError, ValueError, TypeError, AttributeError, IndexError,
                    ArithmeticError) as exc:
                reconstructed = None
                reconstruction_error = str(exc)
            if reconstructed is not None:
                selected = reconstructed.image
                coverage_mask = reconstructed.coverage_mask
                headless_evidence = reconstructed.metadata
                provenance = "headless_projection_reconstruction"
            elif (bool(fallback_to_source) and source_image is not None
                  and source_stats and source_stats.get("usable")):
                selected = source_image
                provenance = "source_image_fallback"
            else:
                provenance = "no_usable_image"

        summary = build_output_solve_summary(solve, depth)
        if headless_evidence is not None:
            summary["headless_evidence"] = headless_evidence
        elif reconstruction_error:
            summary["headless_evidence"] = {
                "status": "unavailable", "reason": reconstruction_error}
        external = _safe_external_summary(solve_summary)
        if external is not None:
            summary["upstream_summary"] = external
        if (provenance == "headless_projection_reconstruction"
                and source_image is not None and source_stats
                and source_stats.get("usable")):
            try:
                from atlas_camera.comfy.headless_evidence import compare_source_structure
                summary["source_comparison"] = compare_source_structure(
                    selected, source_image)
            except (ImportError, ValueError, TypeError, AttributeError, IndexError,
                    ArithmeticError) as exc:
                summary["source_comparison"] = {
                    "status": "unavailable", "reason": str(exc)}
        health = summary["scene_health"]

        selected_stats = (_image_stats(selected)
                          if provenance != "no_usable_image" else camera_stats)
        evidence_path = ""
        coverage_path = ""
        source_reference_path = ""
        evidence_sha256 = ""
        source_reference_sha256 = ""
        evidence_write_error = ""
        if provenance != "no_usable_image":
            try:
                evidence_path = _write_tensor_png(selected, requested_evidence_path)
                with open(evidence_path, "rb") as handle:
                    evidence_sha256 = hashlib.sha256(handle.read()).hexdigest()
                if coverage_mask is not None:
                    coverage_path = _write_mask_png(
                        coverage_mask, requested_coverage_path)
            except OSError as exc:
                evidence_write_error = str(exc)
                evidence_path = ""
                coverage_path = ""
        if source_image is not None and source_stats and source_stats.get("usable"):
            try:
                source_reference_path = _write_tensor_png(
                    source_image, requested_source_reference_path)
                with open(source_reference_path, "rb") as handle:
                    source_reference_sha256 = hashlib.sha256(handle.read()).hexdigest()
            except OSError as exc:
                evidence_write_error = (evidence_write_error + "; " + str(exc)).strip("; ")
                source_reference_path = ""

        vlm_result = None
        status = "disabled"
        if bool(enabled) and provenance != "no_usable_image":
            from atlas_camera.inference.output_assessor import assess_output

            selected_bytes = selected.detach().cpu().float().numpy().tobytes()
            key_src = (selected_bytes + json.dumps(summary, sort_keys=True, default=str).encode()
                       + (f"|{evidence_sha256}|{source_reference_sha256}|"
                          f"{_OUTPUT_ASSESS_CACHE_CONTRACT}|{provenance}|{provider}|"
                          f"{model}|{base_url}|{extra_instructions}").encode())
            key = hashlib.md5(key_src).hexdigest()
            vlm_result = _OUTPUT_ASSESS_CACHE.get(key)
            if vlm_result is None:
                temporary = not bool(evidence_path)
                vlm_image_path = (evidence_path if evidence_path
                                  else _temporary_tensor_png(selected))
                try:
                    vlm_result = assess_output(
                        vlm_image_path, solve_summary=summary,
                        image_provenance=provenance,
                        coverage_path=coverage_path or None,
                        source_reference_path=source_reference_path or None,
                        provider=provider, model=model,
                        base_url=base_url.strip() or None,
                        api_key=api_key.strip() or None,
                        extra_instructions=extra_instructions,
                        offload_model=bool(offload_model))
                finally:
                    if temporary:
                        try:
                            os.unlink(vlm_image_path)
                        except OSError:
                            pass
                if vlm_result.ok:
                    if len(_OUTPUT_ASSESS_CACHE) >= _OUTPUT_ASSESS_CACHE_MAX:
                        _OUTPUT_ASSESS_CACHE.pop(next(iter(_OUTPUT_ASSESS_CACHE)))
                    _OUTPUT_ASSESS_CACHE[key] = vlm_result
            status = "complete" if vlm_result.ok else "unavailable"
        elif bool(enabled):
            status = "no_usable_image"

        visual = (_enforce_provenance(vlm_result.payload, provenance)
                  if vlm_result and vlm_result.ok else {})
        if visual:
            visual = _enforce_deterministic_evidence(
                visual, provenance, summary)
        visual_verdict = str(visual.get("verdict", "inconclusive")).lower()
        deterministic_output_verdict = _deterministic_output_floor(
            provenance, summary)
        overall = _overall_verdict(
            health_level=str(health.get("level", "warn")),
            vlm_ok=bool(vlm_result and vlm_result.ok),
            visual_verdict=visual_verdict, provenance=provenance)
        rank = {"pass": 0, "inconclusive": 1, "warn": 2, "fail": 3}
        overall = max((overall, deterministic_output_verdict),
                      key=lambda value: rank.get(value, 1))

        try:
            from atlas_camera import __version__ as atlas_version
        except Exception:
            atlas_version = "unknown"
        try:
            from atlas_camera.comfy.node_registry import registry_surface_hash
            registry_hash = registry_surface_hash()
        except Exception:
            registry_hash = "unknown"
        data = {
            "schema": 2,
            "atlas_version": atlas_version,
            "registry_hash": registry_hash,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "verdict": overall,
            "deterministic_output_verdict": deterministic_output_verdict,
            "image": {
                "provenance": provenance,
                "camera_view_stats": camera_stats,
                "source_image_stats": source_stats,
                "assessed_image_stats": selected_stats,
                "evidence_path": evidence_path,
                "coverage_path": coverage_path,
                "source_reference_path": source_reference_path,
                "evidence_sha256": evidence_sha256,
                "source_reference_sha256": source_reference_sha256,
                "evidence_write_error": evidence_write_error or None,
                "headless_reconstruction": headless_evidence,
                "projection_checks": (
                    "inconclusive: source plate fallback is not a rendered output"
                    if provenance == "source_image_fallback" else
                    "canonical camera output reconstructed from projected plates, mattes, "
                    "and mesh UV coverage; orbit occlusion remains deterministic-only"
                    if provenance == "headless_projection_reconstruction" else
                    "unavailable: no usable image"
                    if provenance == "no_usable_image" else
                    "camera-view image supplied"),
                "ocio_alpha_contract": (
                    "no RGB colour transform; straight RGB(A) evidence is written without "
                    "premultiplication; alpha/mattes remain data and are used only as coverage "
                    "for the display-proxy reconstruction"),
            },
            "solve_summary": summary,
            "vlm": {
                "enabled": bool(enabled),
                "ok": bool(vlm_result and vlm_result.ok),
                "provider": (vlm_result.provider if vlm_result else provider),
                "model": (vlm_result.model if vlm_result else model),
                "assessment": visual or None,
                "error_report": (vlm_result.report
                                 if vlm_result and not vlm_result.ok else None),
            },
        }

        lines = [f"ATLAS TERMINAL OUTPUT QA — {overall.upper()}",
                 f"status {status}  · image {provenance}  · solve health "
                 f"{str(health.get('level', 'unknown')).upper()}", ""]
        if provenance == "source_image_fallback":
            lines += ["HEADLESS LIMIT: viewport.shaded was an unbaked all-zero pass. The VLM "
                      "saw the source plate; projection edges, occlusion, parallax, and final "
                      "inpaint appearance remain inconclusive. Bake Render Proxy Passes or "
                      "feed a DCC render for a true visual verdict.", ""]
        elif provenance == "headless_projection_reconstruction":
            coverage = (headless_evidence or {}).get("coverage_fraction")
            lines += [
                "HEADLESS EVIDENCE: viewport.shaded was blank, so Atlas reconstructed the "
                "canonical camera output from the solve's projected plates, mattes, and mesh "
                f"coverage ({coverage if coverage is not None else 'unknown'} frame coverage).",
                "Camera framing, canonical projection boundaries, layer inpaint, and colour are "
                "visually assessable. Orbit/grazing occlusion remains inconclusive visually; "
                "use the deterministic orbit_coverage block or a browser/DCC render.", ""]
        elif provenance == "no_usable_image":
            lines += ["NO USABLE IMAGE: connect a rendered camera view, or connect source_image "
                      "to permit the explicitly limited fallback.", ""]
        flags = health.get("flags") or []
        lines.append(f"DETERMINISTIC FLAGS ({len(flags)})")
        lines += ([f"  ! {flag.get('severity', 'warn').upper()} "
                   f"{flag.get('code', 'health')}: {flag.get('message', '')}"
                   for flag in flags] or ["  (none)"])
        if vlm_result and vlm_result.ok:
            from atlas_camera.inference.output_assessor import (
                format_output_assessment_report)
            lines += ["", format_output_assessment_report(
                visual, provider=vlm_result.provider, model=vlm_result.model)]
        elif vlm_result:
            lines += ["", vlm_result.report]
        elif not enabled:
            lines += ["", "VLM disabled — enable on the node, or run through MCP with "
                      "assess_output=true. Deterministic solve health was still recorded."]

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=1, ensure_ascii=False)
        except OSError as exc:
            lines += ["", f"could not write {path}: {exc}"]
            path = ""
        if path:
            lines += ["", f"full JSON: {path}"]
        if evidence_path:
            lines += [f"assessed image: {evidence_path}"]
        if coverage_path:
            lines += [f"coverage matte: {coverage_path}"]
        if source_reference_path:
            lines += [f"source reference: {source_reference_path}"]
        report = "\n".join(lines)
        serialized = json.dumps(data, indent=1, ensure_ascii=False)
        return {
            "ui": {
                "text": [report],
                "atlas_output_assessment": [serialized],
                "json_path": [path],
                "evidence_path": [evidence_path],
                "coverage_path": [coverage_path],
                "source_reference_path": [source_reference_path],
            },
            "result": (report, serialized, path, overall, provenance,
                       selected, evidence_path),
        }


class AtlasAdherenceScore:
    """Did the generator obey the camera? Measured, with a required control.

    Scores a generated frame sequence against the deterministic reprojection in
    an `ATLAS_CONDITIONING` bundle, in VALID pixels ONLY -- the pixels Atlas
    knows the true appearance of. That is what makes the measurement possible at
    all: a generated move cannot be solved (measured 2026-09-03, every frame of
    an LTX CrossView move collapses to 12-20 SIFT inliers against the plate's
    1022 and is refused), so any metric that needs registration on generated
    frames is dead on arrival. This one needs none.

    THE CONTROL ARM IS A REQUIRED INPUT, not an optional one. An adherence
    figure with nothing to compare it against cannot support a claim in either
    direction, so the socket enforces at graph level what a report could only
    ask for politely. Feed it the same prompt, seed, step count and frame count
    with the geometric conditioning removed.

    IT REFUSES RATHER THAN FLATTERING YOU. A clip that simply repeats its first
    frame scores near-perfect adherence in VALID, because a static plate agrees
    with Atlas's reprojection wherever the move is small. The scorer therefore
    measures `parallax_response` -- agreement against frame i minus agreement
    against frame 0 -- and raises when it is not positive. A frozen or
    near-frozen result gets no headline number, because the number would be
    measuring nothing.

    The headline is gradient ZNCC over VALID, chosen because generated frames
    come back sRGB and often graded: gradients remove the exposure offset,
    normalisation removes the gain, and what survives is whether the structure
    is where Atlas said it would be. The fitted grade is reported as a measured
    number beside it. Masked SSIM and raw MAE/PSNR are diagnostics only.
    """

    CATEGORY = "Atlas/09 · QA & Gates"
    FUNCTION = "score"
    RETURN_TYPES = ("FLOAT", "FLOAT", "IMAGE", "STRING")
    RETURN_NAMES = ("adherence", "margin_over_control", "drift_plot", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bundle": ("ATLAS_CONDITIONING", {
                    "tooltip": "From AtlasConditioningBundle — carries the "
                               "deterministic reprojection and the class map "
                               "this is scored against."}),
                "generated": ("IMAGE", {
                    "tooltip": "The arm under test: frames generated WITH Atlas "
                               "conditioning. One frame per bundle frame."}),
                "control": ("IMAGE", {
                    "tooltip": "The prompt-only arm: same prompt, seed, steps "
                               "and frame count, no geometric conditioning. "
                               "REQUIRED — without it the headline number "
                               "cannot support a claim either way."}),
            },
            "optional": {
                "min_parallax_px": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 64.0, "step": 0.1,
                    "tooltip": "Frames whose median measured parallax is below "
                               "this are reported but excluded from the "
                               "aggregate. A zoom sits here by definition: it "
                               "moves pixels without moving the eye, so there "
                               "is nothing for a camera to get right."}),
                "bootstrap_seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "refuse_degenerate": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Raise when parallax_response is not positive. "
                               "Turn this off ONLY to inspect why a run is "
                               "degenerate — the number it then reports is not "
                               "evidence of anything."}),
            },
        }

    def score(self, bundle, generated, control, min_parallax_px=1.5,
              bootstrap_seed=0, refuse_degenerate=True):
        from atlas_camera.comfy.node_helpers import _require_numpy, _require_torch
        from atlas_camera.core.adherence import (
            ATLAS_ARM, CONTROL_ARM, score_arms,
        )

        np = _require_numpy()
        torch = _require_torch()
        if bundle is None:
            raise ValueError(
                "AtlasAdherenceScore: no bundle. The upstream "
                "AtlasConditioningBundle refused (no projection geometry on the "
                "solve) or was bypassed; there is nothing to score against.")

        def to_numpy(frames):
            arr = (frames.detach().cpu().numpy()
                   if hasattr(frames, "detach") else np.asarray(frames))
            return np.asarray(arr, dtype=np.float64)[..., :3]

        result = score_arms(
            {ATLAS_ARM: to_numpy(generated), CONTROL_ARM: to_numpy(control)},
            bundle=bundle, min_parallax_px=float(min_parallax_px),
            bootstrap_seed=int(bootstrap_seed),
            refuse_degenerate=bool(refuse_degenerate))

        plot = self._drift_plot(np, result)
        adherence = result["adherence"]
        margin = result["margin_over_control"]
        return (float(adherence) if adherence is not None else 0.0,
                float(margin) if margin is not None else 0.0,
                torch.from_numpy(plot).float()[None], json.dumps(result, indent=2))

    @staticmethod
    def _drift_plot(np, result, width: int = 512, height: int = 256):
        """Per-frame headline for both arms, drawn without a plotting library.

        A number and a slope do not show a reader WHERE a generator lost the
        camera; the curve does. Deliberately dependency-free — matplotlib is not
        an Atlas dependency and a 30-line scatter does not justify making it one.
        """
        img = np.ones((height, width, 3), dtype=np.float32) * 0.08
        arms = result.get("arms", {})
        colours = {"atlas": (0.2, 1.0, 0.4), "prompt_only": (1.0, 0.5, 0.2),
                   "static": (0.45, 0.45, 0.5)}
        series = {label: [f["gradient_zncc"] for f in arm["per_frame"]]
                  for label, arm in arms.items()}
        finite = [v for vals in series.values() for v in vals
                  if v is not None and np.isfinite(v)]
        if not finite:
            return img
        lo, hi = min(finite), max(finite)
        span = max(hi - lo, 1e-6)

        # Gridlines first, so the curves draw over them.
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            row = int((1.0 - frac) * (height - 1))
            img[row, :, :] = 0.18

        for label, vals in series.items():
            colour = np.asarray(colours.get(label, (0.7, 0.7, 0.7)),
                                dtype=np.float32)
            n = max(len(vals) - 1, 1)
            for index, value in enumerate(vals):
                if value is None or not np.isfinite(value):
                    continue
                col = int(index * (width - 1) / n)
                row = int((1.0 - (value - lo) / span) * (height - 1))
                r0, r1 = max(row - 2, 0), min(row + 3, height)
                c0, c1 = max(col - 2, 0), min(col + 3, width)
                img[r0:r1, c0:c1] = colour
        return img
