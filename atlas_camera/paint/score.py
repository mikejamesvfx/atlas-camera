"""Score an externally-edited plate against its original.

The acceptance gate for any paint-package bridge: an edited plate comes back,
and this answers "did the edit stay inside the region it was authorised to
touch, and does it join the photograph cleanly?" using
``core.plate_falsification`` — the same metrics that falsified the hole-splat
run, so an external fill is held to exactly the standard an internal one is.

Mapping onto the metrics:

* ``alpha`` (what the candidate painted) = the pixels that actually CHANGED
  between original and edited, not the request mask — an edit that strayed is
  measured by where it strayed.
* ``authorised_mask`` = the supplied mask: containment < 1.0 means the fill
  painted outside its brief.
* ``composite`` = the edited plate, ``plate`` = the original, so
  ``seam_gradient_ratio`` measures the join at the changed region's rim,
  self-referenced against the original plate's own rim busyness.

Both files are read with ``raw_data=True``: the comparison is in the files'
own shared colourspace, converting both by the same transform would only
rescale both sides of every ratio, and a vendor that MISLABELS its export
would otherwise poison the comparison.

Two findings that decide how the verdict is computed:

1. **A feather is spill unless the authorised mask includes it.** The
   containment gate caught a "clean" feathered edit painting outside a binary
   mask at 0.9329. Hand this the mask ``confine`` emitted, not the raw object
   mask.
2. **The do-nothing baseline is unbeatable on seam for 2D edits.** A
   do-nothing composite's rim IS the plate, so its seam ratio is exactly 1.0,
   and a clean real edit measured 0.9996 reads as infinitesimally "worse". The
   baseline stays in the JSON because it is what falsifies GEOMETRY candidates,
   but the decision comes from the candidate's own calibrated gates.

And the standing caveat: **gates are necessary, not sufficient.** A confined
plate has passed both while being visibly wrong, because a hazy blend has a low
rim gradient and a smooth wrong answer scores well.
"""
from __future__ import annotations

from pathlib import Path

#: Order matters only for display; every available gated metric must pass.
GATED_METRICS = ("containment", "seam_gradient_ratio", "sky_violation")

#: Exit codes, kept stable because callers and CI branch on them.
EXIT_ACCEPTED = 0
EXIT_REJECTED = 2
EXIT_INCONCLUSIVE = 3


def load_mask(path):
    import numpy as np
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("L"), dtype="float32") / 255.0
    return arr > 0.5


def changed_pixels(np, original, edited, eps: float):
    return np.abs(edited - original).max(axis=-1) > eps


def score(*, original_path, edited_path, mask_path, rim_px: int = 2,
          change_eps: float = 1e-4) -> dict:
    """Return the full falsification payload plus a decision.

    ``change_eps`` is not a constant of nature. 1e-4 is right for a float
    round trip; a 16-bit display-referred leg quantises further than that and
    would flag the entire frame as painted, so a bridge that goes through one
    must supply its own measured value.
    """
    import numpy as np

    from atlas_camera.core.plate_falsification import falsification_report
    from atlas_camera.paint.ocio import config_identity
    from atlas_camera.plate.oiio_io import read_plate

    original = read_plate(str(original_path), raw_data=True)
    edited = read_plate(str(edited_path), raw_data=True)
    if (original.height, original.width) != (edited.height, edited.width):
        raise ValueError(
            f"raster mismatch: original {original.width}x{original.height} vs "
            f"edited {edited.width}x{edited.height} — the edit must come back "
            f"at the plate raster")

    authorised = load_mask(mask_path)
    if authorised.shape != (original.height, original.width):
        raise ValueError(
            f"mask raster {authorised.shape[::-1]} does not match the plate "
            f"{original.width}x{original.height}")

    changed = changed_pixels(np, original.pixels, edited.pixels, change_eps)
    if not bool(changed.any()):
        raise ValueError(
            "the edited plate is pixel-identical to the original: nothing to "
            "score")

    report = falsification_report(
        candidate=dict(alpha=changed, plate=original.pixels,
                       composite=edited.pixels, authorised_mask=authorised,
                       rim_px=rim_px),
        baseline=dict(
            # The do-nothing edit: the authorised region, left as the plate.
            alpha=authorised, plate=original.pixels, composite=original.pixels,
            authorised_mask=authorised, rim_px=rim_px),
    )

    payload = report.to_dict()
    payload["inputs"] = {
        "original": str(original_path),
        "edited": str(edited_path),
        "mask": str(mask_path),
        "changed_px": int(changed.sum()),
        "authorised_px": int(authorised.sum()),
        "change_eps": change_eps,
        "ocio": config_identity(),
    }

    results = []
    for name in GATED_METRICS:
        metric = payload["candidate"][name]
        if metric.get("available") and metric.get("pass") is not None:
            results.append(bool(metric["pass"]))
    payload["decision"] = ("inconclusive" if not results
                           else "accepted" if all(results) else "rejected")
    return payload


def format_table(payload: dict) -> str:
    """The human-readable gate table the CLIs print."""
    inputs = payload["inputs"]
    lines = [
        f"changed px          {inputs['changed_px']:>10,}",
        f"authorised px       {inputs['authorised_px']:>10,}",
    ]
    for name in GATED_METRICS:
        metric = payload["candidate"][name]
        value = metric.get("value")
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        status = ("n/a" if not metric.get("available")
                  else {True: "PASS", False: "FAIL",
                        None: "ungated"}[metric.get("pass")])
        lines.append(f"{name:<19} {shown:>10}  {status}")
    lines.append(f"decision            {payload['decision']:>10}   "
                 f"(baseline verdict: {payload['verdict']})")
    return "\n".join(lines)


def exit_code(payload: dict) -> int:
    return {"accepted": EXIT_ACCEPTED,
            "rejected": EXIT_REJECTED}.get(payload["decision"],
                                           EXIT_INCONCLUSIVE)


def write_report(path, payload: dict) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")
    return path
