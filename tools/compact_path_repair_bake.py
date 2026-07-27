"""Compact saved Atlas Camera paths to one indexed repair-preview frame.

The parametric camera path remains complete. Only the embedded JPEG batch is
reduced, so path-guided geometry still samples every pose while the default
final-frame repair preview avoids carrying a full video clip in UI/API JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compact_client_data(
    client_data: str,
    *,
    frame_offset_from_end: int = 0,
) -> tuple[str, int, int]:
    """Return compact JSON plus ``(old_count, selected_path_frame)``."""
    data = json.loads(client_data)
    frames = list(data.get("path_frames") or [])
    camera_path = data.get("camera_path") or {}
    if not frames or not camera_path:
        raise ValueError("client_data has no baked path frames/camera_path")

    frame_count = max(0, int(camera_path.get("frame_count", len(frames))))
    if frame_count <= 0:
        raise ValueError("camera_path frame_count must be positive")
    selected_frame = max(
        0,
        min(
            frame_count - 1,
            frame_count - 1 - max(0, int(frame_offset_from_end)),
        ),
    )

    explicit_indices = [
        int(value) for value in camera_path.get("baked_frame_indices", [])
    ]
    if explicit_indices:
        if len(explicit_indices) != len(frames):
            raise ValueError(
                "baked_frame_indices length does not match path_frames"
            )
        try:
            batch_index = explicit_indices.index(selected_frame)
        except ValueError as exc:
            raise ValueError(
                f"path frame {selected_frame} is not present in baked batch"
            ) from exc
    else:
        if selected_frame >= len(frames):
            raise ValueError(
                f"legacy baked batch has no path frame {selected_frame}"
            )
        batch_index = selected_frame

    old_count = len(frames)
    data["path_frames"] = [frames[batch_index]]
    camera_path["baked_frame_indices"] = [selected_frame]
    data["camera_path"] = camera_path

    provenance = dict(data.get("atlas_proxy_path") or {})
    provenance["frame_count"] = frame_count
    provenance["stored_frame_count"] = 1
    provenance["frame_indices"] = [selected_frame]
    data["atlas_proxy_path"] = provenance
    return (
        json.dumps(data, separators=(",", ":"), ensure_ascii=False),
        old_count,
        selected_frame,
    )


def _upgrade_ui_depth_gates(
    workflow: dict[str, Any],
    max_patch_depth_factor: float,
    max_pass2_edge_factor: float = 40.0,
) -> None:
    """Append the Z gate and clamp the unsafe relaxed Pass 2 edge budget."""
    for node in workflow.get("nodes", []):
        if node.get("type") != "AtlasPlanarHolePatch":
            continue
        values = node.setdefault("widgets_values", [])
        if len(values) == 9:
            values.append(float(max_patch_depth_factor))
        if len(values) >= 9 and float(values[8]) > max_pass2_edge_factor:
            values[8] = float(max_pass2_edge_factor)
            node["title"] = str(node.get("title", "")).replace("65×", "40×")


def _upgrade_api_depth_gates(
    prompt: dict[str, Any],
    max_patch_depth_factor: float,
    max_pass2_edge_factor: float = 40.0,
) -> None:
    for node in prompt.values():
        if node.get("class_type") == "AtlasPlanarHolePatch":
            inputs = node.setdefault("inputs", {})
            inputs.setdefault(
                "max_patch_depth_factor",
                float(max_patch_depth_factor),
            )
            current_edge = float(inputs.get("max_patch_edge_factor", 0.0))
            if current_edge > max_pass2_edge_factor:
                inputs["max_patch_edge_factor"] = float(max_pass2_edge_factor)
                meta = node.get("_meta") or {}
                if "title" in meta:
                    meta["title"] = str(meta["title"]).replace("65×", "40×")


def _compact_ui(workflow: dict[str, Any], offset: int) -> tuple[int, int]:
    result = None
    for node in workflow.get("nodes", []):
        values = node.get("widgets_values") or []
        for index, value in enumerate(values):
            if isinstance(value, str) and '"path_frames"' in value:
                compacted, old_count, selected = compact_client_data(
                    value, frame_offset_from_end=offset)
                values[index] = compacted
                result = (old_count, selected)
                break
        if result is not None:
            break
    if result is None:
        raise ValueError("UI workflow has no client_data with path_frames")
    _upgrade_ui_depth_gates(workflow, 2.0)
    return result


def _compact_api(prompt: dict[str, Any], offset: int) -> tuple[int, int]:
    result = None
    for node in prompt.values():
        inputs = node.get("inputs") or {}
        value = inputs.get("client_data")
        if isinstance(value, str) and '"path_frames"' in value:
            compacted, old_count, selected = compact_client_data(
                value, frame_offset_from_end=offset)
            inputs["client_data"] = compacted
            result = (old_count, selected)
            break
    if result is None:
        raise ValueError("API workflow has no client_data with path_frames")
    _upgrade_api_depth_gates(prompt, 2.0)
    return result


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui", type=Path)
    parser.add_argument("--api", type=Path)
    parser.add_argument("--frame-offset-from-end", type=int, default=0)
    args = parser.parse_args()
    if args.ui is None and args.api is None:
        parser.error("provide --ui and/or --api")

    for path, handler in (
        (args.ui, _compact_ui),
        (args.api, _compact_api),
    ):
        if path is None:
            continue
        payload = _load(path)
        old_count, selected = handler(
            payload, max(0, args.frame_offset_from_end))
        _write(path, payload)
        print(
            f"{path}: {old_count} frames -> 1 "
            f"(path frame {selected})"
        )


if __name__ == "__main__":
    main()
