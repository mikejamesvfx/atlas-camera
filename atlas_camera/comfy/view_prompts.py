"""The Qwen named-view vocabulary and the strings that carry it between nodes.

Split out of `node_helpers.py` in phase 3 of
`docs/dev/node_helpers_layering_plan.md`.

These tables are module-level ON PURPOSE, not private to `AtlasAddPatchView`:
`AtlasOcclusionMask` must place its target camera IDENTICALLY, and any drift
between the two would silently misalign a precomputed occlusion mask from the
patch geometry derived for the same image. Sharing one implementation is what
makes that impossible.

The parsers exist because ComfyUI's backend REJECTS STRING->combo links at
prompt validation even though the frontend lets you draw them — so an extracted
angle travels as one STRING and is parsed here (greedy longest-prefix over the
named-view vocabulary, erroring loudly on anything unrecognised).
"""

from __future__ import annotations

import re


# Exact named views from ComfyUI-qwenmultiangle / the Multiple-Angles LoRA, shared
# by every node that places a patch/target camera relative to a source photo
# (`AtlasAddPatchView`, `AtlasOcclusionMask`) so the same choice is picked
# everywhere and the two nodes' camera placement can never drift apart. Azimuth
# is absolute about the subject's front; distance scales the orbit radius
# (close-up pulls in).
_AZIMUTH_VIEWS = {
    "front view": 0.0, "front-right quarter view": 45.0, "right side view": 90.0,
    "back-right quarter view": 135.0, "back view": 180.0, "back-left quarter view": 225.0,
    "left side view": 270.0, "front-left quarter view": 315.0,
}
_ELEVATION_VIEWS = {
    "low-angle shot": -30.0, "eye-level shot": 0.0, "elevated shot": 30.0, "high-angle shot": 60.0,
}
_DISTANCE_VIEWS = {"close-up": 0.6, "medium shot": 1.0, "wide shot": 1.8}
def _named_view_orbit_delta(
    patch_azimuth_view, patch_elevation_view, patch_distance,
    source_azimuth_view, source_elevation_view, flip_azimuth,
):
    """Resolve absolute (subject-relative) LoRA named views into the actual
    orbit delta to apply to the recovered/source camera: ``patch - source``.

    Returns ``(d_azimuth_deg, d_elevation_deg, distance_scale)``.
    """
    d_azimuth = _AZIMUTH_VIEWS[patch_azimuth_view] - _AZIMUTH_VIEWS[source_azimuth_view]
    d_azimuth = ((d_azimuth + 180.0) % 360.0) - 180.0   # shortest way round
    if flip_azimuth:
        d_azimuth = -d_azimuth
    d_elevation = _ELEVATION_VIEWS[patch_elevation_view] - _ELEVATION_VIEWS[source_elevation_view]
    distance_scale = _DISTANCE_VIEWS[patch_distance]  # source assumed "medium shot"
    return float(d_azimuth), float(d_elevation), float(distance_scale)
def _parse_view_prompt(text):
    """Parse a Multiple-Angles LoRA prompt — "<sks> [azimuth] [elevation]
    [distance]", the exact string 📐 Extract Angle's `patch_prompt` output
    emits — back into the three named views. Returns (azimuth, elevation,
    distance) or None when the text doesn't match the vocabulary.

    Exists because ComfyUI's backend REJECTS a STRING link into a combo-list
    input ("received_type(STRING) mismatch input_type([...])" at prompt
    validation), so the viewport's per-view STRING outputs can't wire into
    the named-view dropdowns directly — instead one `patch_view_override`
    STRING socket takes the whole prompt and this parses it. The names
    contain spaces, so parsing is greedy prefix-matching against the known
    vocabularies (longest first), which is unambiguous because the LoRA's
    view names are a fixed, non-overlapping set.
    """
    rest = (text or "").strip()
    if rest.startswith("<sks>"):
        rest = rest[len("<sks>"):].strip()
    parsed = []
    for table in (_AZIMUTH_VIEWS, _ELEVATION_VIEWS, _DISTANCE_VIEWS):
        match = next((name for name in sorted(table, key=len, reverse=True)
                      if rest.startswith(name)), None)
        if match is None:
            return None
        parsed.append(match)
        rest = rest[len(match):].strip()
    if rest:
        return None
    return tuple(parsed)
def _parse_exact_view(text):
    """Parse an EXACT orbit delta — "azimuth_deg=<f> elevation_deg=<f>
    distance_scale=<f>", the string 📐 Extract Angle's `patch_exact` output
    emits (raw measured floats, BEFORE named-view snapping). Returns
    (d_azimuth_deg, d_elevation_deg, distance_scale) or None when the text
    doesn't carry all three keys.

    The render-conditioned patch loop needs this precision: a frame baked at
    the artist's real orbit must be projected back from the IDENTICAL pose —
    snapping to the LoRA's 45° azimuth grid would misregister the projection.
    Key=value format (any order, comma or space separated) so the string is
    self-documenting in Show Text nodes and export logs.
    """
    import re

    vals = dict(re.findall(
        r"(azimuth_deg|elevation_deg|distance_scale)\s*=\s*(-?\d+(?:\.\d+)?)",
        text or ""))
    if set(vals) != {"azimuth_deg", "elevation_deg", "distance_scale"}:
        return None
    return (float(vals["azimuth_deg"]), float(vals["elevation_deg"]),
            float(vals["distance_scale"]))

__all__ = [
    "_AZIMUTH_VIEWS",
    "_ELEVATION_VIEWS",
    "_DISTANCE_VIEWS",
    "_named_view_orbit_delta",
    "_parse_view_prompt",
    "_parse_exact_view",
]
