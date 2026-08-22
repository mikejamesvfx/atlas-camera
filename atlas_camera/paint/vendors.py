"""Per-paint-package defaults, and what is actually MEASURED about each.

The point of this table is not convenience — it is to stop a hypothesis from
becoming a fact by being typed into an ``argparse`` default. Every number here
carries the run it came from, and anything unmeasured is ``None`` rather than a
plausible guess, because a plausible guess is indistinguishable from a
measurement once it is in the code.

The load-bearing field is ``needs_roi_crop``:

* ``True``  — the package regenerates beyond its selection and must be handed a
  cropped ROI. Measured for Affinity.
* ``False`` — the package's fill is genuinely bounded; a full frame is safe.
  Only ever set from a passing containment measurement at full resolution.
* ``None``  — UNMEASURED. Tools must refuse to choose and make the caller say
  ``--roi`` or ``--no-roi`` explicitly.

Provenance for the Affinity numbers: ``reports/paint_bridge_provenance.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VendorProfile:
    """One paint package's measured behaviour and the defaults it implies."""

    key: str
    display_name: str
    #: Grow the object mask before blending — room for contact shadow and the
    #: object's own soft rim.
    dilate_px: int
    #: Blend ramp width, inside the dilated mask.
    feather_px: int
    #: Tri-state. See the module docstring; None means UNMEASURED.
    needs_roi_crop: bool | None
    #: Per-channel threshold for "this pixel changed" when scoring. A 16-bit
    #: display round trip cannot be judged at the float default.
    change_eps: float
    #: Free text: where these numbers came from, or why they are provisional.
    provenance: str
    #: Anything a caller should be warned about, printed by the CLIs.
    caveats: tuple[str, ...] = field(default_factory=tuple)

    @property
    def measured(self) -> bool:
        return self.needs_roi_crop is not None


AFFINITY = VendorProfile(
    key="affinity",
    display_name="Affinity (Canva) 3.2.3",
    dilate_px=45,
    feather_px=12,
    needs_roi_crop=True,
    change_eps=1e-4,
    provenance=(
        "Measured 2026-08-21 on two X-H2 plates. generativeEditImage is "
        "image-to-image REGENERATION, not inpainting, even after "
        "selectSubject(): the boiler plate scored containment 0.3740 and the "
        "street plate came back as a different building. The ROI crop is the "
        "only reliable confinement. See reports/paint_bridge_provenance.md."
    ),
    caveats=(
        "Mislabels its EXR export as ACEScg while leaving Rec.709-linear "
        "values untouched — read its output raw_data=True, and re-tag "
        "Atlas-side. Naming AtlasLoadPlate.input_colorspace does NOT protect: "
        "a file's declared oiio:ColorSpace wins unconditionally.",
        "Document.close() is NOT_IMPLEMENTED, so documents accumulate.",
    ),
)

PHOTOSHOP_BETA = VendorProfile(
    key="photoshop_beta",
    display_name="Adobe Photoshop (Beta)",
    # Carried over from Affinity as a STARTING POINT only. Photoshop's
    # syntheticFill has an explicit `inpaint` mode, so its spill profile may be
    # completely different; these two numbers are the first thing to re-measure
    # once rung A runs.
    dilate_px=45,
    feather_px=12,
    # UNMEASURED, deliberately. syntheticFillMode exposes an `inpaint` enum
    # value, which Affinity had no equivalent of, so the crop may well be
    # unnecessary — but "may well be" is not a measurement, and writing False
    # here would silently ship that assumption.
    needs_roi_crop=None,
    # Provisional: the generative rung runs through a 16-bit display-referred
    # Smart Object, and 1e-4 would flag the entire frame as painted. Replaced
    # by the M2 measurement.
    change_eps=1e-4,
    provenance=(
        "Capabilities established 2026-08-21 by inspecting the shipped binary: "
        "native OpenColorIO v2.5 with an Environment ($OCIO) config domain; "
        "syntheticFill is a scriptable Action Manager event with a "
        "syntheticFillMode enum of inpaint/variation/synthesize. NO fill has "
        "been run or scored yet — every number above is provisional."
    ),
    caveats=(
        "OCIO needs a 32-bit RGB document with the native canvas (Manta) "
        "enabled; Generative Fill needs 8 or 16 bit. They cannot coexist in "
        "one document — the bridge uses a 16-bit Smart Object inside a 32-bit "
        "OCIO container.",
        "Photoshop 2026 may also be installed: always Dispatch the "
        "'Photoshop.Application.BETA' ProgID and verify app.path.",
    ),
)

PROFILES = {p.key: p for p in (AFFINITY, PHOTOSHOP_BETA)}


class UnmeasuredVendorBehaviour(RuntimeError):
    """Raised when a tool would have to GUESS a vendor's behaviour."""


def get(key: str) -> VendorProfile:
    try:
        return PROFILES[key]
    except KeyError:
        raise KeyError(
            f"unknown paint vendor {key!r}; known: {sorted(PROFILES)}") from None


def require_roi_decision(profile: VendorProfile, explicit: bool | None) -> bool:
    """Resolve whether to crop an ROI, refusing to invent an answer.

    An explicit ``--roi/--no-roi`` always wins — that is how a measurement run
    overrides the table on its way to producing the number that updates it.
    """
    if explicit is not None:
        return explicit
    if profile.needs_roi_crop is None:
        raise UnmeasuredVendorBehaviour(
            f"{profile.display_name}: whether a cropped ROI is required has "
            f"not been measured, so this tool will not assume one way or the "
            f"other. Pass --roi or --no-roi explicitly. To settle it, run the "
            f"containment measurement on a full uncropped frame and record the "
            f"result in vendors.py.\n  provenance: {profile.provenance}")
    return profile.needs_roi_crop
