"""Atlas ComfyUI nodes — LTX camera path group.

Reads the `camera_ltx.json` an Atlas Director take exports and hands the
CrossView-Warp node the `keyframes` string it wants.

The node lives HERE rather than in the fork of ComfyUI-CrossViewWarp on
purpose. Atlas owns the Atlas format, so a change to what Director writes is a
change in one repo; the fork stays as close to upstream as it can, which is
what keeps it mergeable. The two packs sit side by side in `custom_nodes/`, so
wiring one into the other costs nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

#: What Director can write, and whether the file is the take or a stand-in.
#: Keys are the format ids `app/director/export/ltx.ts` stamps; a file carrying
#: anything else is refused rather than guessed at, because every one of these
#: means something different about how far you can trust the render.
FORMATS = {
    "atlas.ltx.crossview_warp": "single path",
    "atlas.ltx.crossview_warp.chain": "chain of segments",
    "atlas.ltx.crossview_warp.compressed": "compressed (previz)",
}


def _fmt(pair) -> str:
    return f"max {pair['max']:.3f}, p95 {pair['p95']:.3f}"


def _report(document: dict, segment: dict | None, segment_count: int) -> str:
    """What the artist needs to know before trusting the render.

    Front-loads the node settings, because those are the failure that looks
    like success: `pivot_override` off makes the animated pivot inert and
    `keep_source_aim` on overrides the aim the knots encode. Either renders a
    DIFFERENT move with no error anywhere. Then the losses, then the warnings.
    """
    body = segment if segment is not None else document
    residual = body.get("residual", {})
    meta = body.get("meta", {})
    requires = meta.get("requires", {})
    lines = [
        f"Atlas camera path — {FORMATS.get(document.get('format'), 'unknown')}",
        "",
        "SET ON CrossView Warp (silently renders a different move otherwise):",
        f"  use_keyframes   = {requires.get('use_keyframes')}",
        f"  pivot_override  = {requires.get('pivot_override')}",
        f"  keep_source_aim = {requires.get('keep_source_aim')}",
        "",
        f"frames {meta.get('frameCount')}, pivot at {meta.get('pivotDepthM')}m "
        f"(the take's focus distance)",
    ]

    if segment is not None:
        source = segment.get("sourceFrames", {})
        lines += [
            "",
            f"SEGMENT {segment.get('index')} of {segment_count} — "
            f"take frames {source.get('first')}-{source.get('last')}, "
            f"{segment.get('travelM', 0):.2f}m of travel.",
        ]
        if segment.get("index", 1) > 1:
            lines.append(
                "  Its source clip is the PREVIOUS segment's final generated frame, "
                "not the plate. Re-derive geometry from that frame, and expect this "
                "segment's error to ride on the one before it."
            )

    compression = document.get("compression")
    if compression:
        travel = compression.get("travelM", {})
        lines += [
            "",
            f"COMPRESSED — previz only. Travel scaled by {compression.get('scale'):.3f}: "
            f"{travel.get('original', 0):.2f}m performed, {travel.get('compressed', 0):.2f}m "
            f"rendered. Timing and aim are the take's; the parallax is not, so do not "
            f"judge depth from this.",
        ]

    if residual:
        offset = residual.get("orbitalOffsetDeg", {})
        focal = residual.get("focalLengthMm", {})
        lines += [
            "",
            "Carried exactly:",
            f"  position  {_fmt(residual.get('positionErrorM', {'max': 0, 'p95': 0}))} m",
            f"  aim       {_fmt(residual.get('aimErrorDeg', {'max': 0, 'p95': 0}))} deg",
            "Not carried:",
            f"  roll      {_fmt(residual.get('discardedRollDeg', {'max': 0, 'p95': 0}))} deg "
            f"(the orbit builds every pose level)",
        ]
        if not focal.get("constant", True):
            lines.append(
                f"  focal     {focal.get('min')}-{focal.get('max')}mm varies; the node has "
                f"one static hfov, so breathing is dropped"
            )
        lines.append(
            f"Asks the LoRA to orbit {offset.get('azimuthMax', 0):.1f}deg azimuth, "
            f"{offset.get('elevationMin', 0):.1f} to {offset.get('elevationMax', 0):.1f}deg "
            f"elevation from frame 1."
        )

    lines += [
        "",
        "The clip must be at least as long as the path: the warp node errors on a "
        "keyframe past the end, which is the check this node cannot make for you.",
    ]
    return "\n".join(lines)


class AtlasLoadCameraPath:
    """🎥 A Director take as a CrossView-Warp camera path.

    Wire `keyframes` into the CrossView Warp node's `keyframes` input. Doing so
    makes its orbit sphere read-only, which is correct: the path is a recording
    of a camera that was operated, and a marker dragged in the graph would show
    a move that is not the one being rendered.

    Reads all three things Director can write, and says which it got. A take too
    long for one generation is exported as a CHAIN — pick the segment with
    `segment`, generate it, then feed its final frame back as the source clip
    for the next. A COMPRESSED file is previz: the timing is the take's and the
    parallax is not, and the report says so rather than letting a third-depth
    push pass for the shot.

    The report output is worth wiring to a PreviewText at least once per graph.
    It carries the three node settings the knots depend on, and getting those
    wrong does not raise anything — it just renders a different move.
    """

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("keyframes", "report", "segment_count")
    FUNCTION = "load"
    CATEGORY = "Atlas/advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "",
                    "tooltip": "The camera_ltx.json Director wrote beside the "
                               "take, in its takes/<slate>/ directory."}),
            },
            "optional": {
                "segment": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": "Which segment of a chained path to load. "
                               "Ignored for a single path. Segment N's source "
                               "clip is segment N-1's final generated frame."}),
            },
        }

    def load(self, path: str, segment: int = 1):
        source = Path(path.strip().strip('"'))
        if not source.is_file():
            raise ValueError(
                f"Atlas camera path not found: {source}. Export a take with the 'ltx' "
                f"target and point this at the camera_ltx.json beside it."
            )
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"{source.name} is not readable JSON: {error}") from None

        fmt = document.get("format")
        if fmt not in FORMATS:
            # Refusing an unknown id is the point of stamping one. A compressed
            # path read as an ordinary one would render a scaled-down move with
            # nothing anywhere saying so.
            raise ValueError(
                f"{source.name} says format {fmt!r}, which this node does not know. "
                f"It reads: {', '.join(sorted(FORMATS))}."
            )

        if fmt == "atlas.ltx.crossview_warp.chain":
            segments = document.get("segments") or []
            count = len(segments)
            if not 1 <= segment <= count:
                raise ValueError(
                    f"{source.name} is a chain of {count} segment(s); segment {segment} "
                    f"does not exist."
                )
            chosen = segments[segment - 1]
            return (chosen["keyframes"], _report(document, chosen, count), count)

        return (document["keyframes"], _report(document, None, 1), 1)
