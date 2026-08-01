"""The Gate — one implementation of the approval checkpoint every gated node
shares.

Four nodes pause the graph on an artist's approval: ✅ AtlasSolveGate,
🩺 AtlasSceneHealthGate, AtlasAssessImage's ▶ Continue, and
AtlasBlockoutViewport's 📐 patch branch. Each used to hand-roll the same five
steps — compute a fingerprint, compare it against the persisted approval,
prepend a re-arm sentence, pick ComfyUI's ExecutionBlocker, wrap the
`{"ui": ..., "result": ...}` envelope — which is four places for the doctrine
to drift. This module owns the HOW; a node declares only WHAT it gates (what
it fingerprints, and what its re-arm sentence says).

The doctrine it encodes (docs/DESIGN_RULES.md, "gates"):

* Gate widgets PERSIST in a saved workflow, so an approval must be scoped to
  WHAT was approved, not to the click — hence the fingerprint. Without it a
  new image sails straight through the previous image's approval.
* A silent branch-skip needs a visible explanation — hence the re-arm
  sentence, which says the identity changed instead of leaving a paused
  branch looking like a failed run.
* The user may OVERRIDE a warning but never LOSE it: `bypass` widens what
  flows, never what is REPORTED (a stale approval still re-arms visibly).
* Outside a ComfyUI runtime there is no ExecutionBlocker, so every gate
  degrades to pass-through — that is what makes gated nodes unit-testable.

CONTRACT — do not change lightly:

* Fingerprint VALUES are saved-workflow state. A changed byte silently
  re-arms every gate in every workflow the user has already approved; the
  helpers here are RE-EXPORTS of `comfy.fingerprints`, never
  re-implementations.
* The envelope shape `{"ui": {"text": [report], "fingerprint": [fp]}, ...}`
  is read by three frontend extensions (`web/atlas_solve_gate.js`,
  `web/atlas_scene_health_gate.js`, `web/atlas_assess.js`).
* Each node's re-arm TAIL is its own wording — pass it in. The nodes do not
  share a sentence, only its `*** GATE RE-ARMED: … ***` frame.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from atlas_camera.comfy.fingerprints import _image_fingerprint, _solve_fingerprint
from atlas_camera.comfy.node_helpers import _execution_blocker

#: Public names for the identity hashes. Aliases — same function objects, so
#: no refactor here can move a fingerprint value.
solve_fingerprint = _solve_fingerprint
image_fingerprint = _image_fingerprint

#: The frame every gate's re-arm sentence is written in.
RE_ARM_PREFIX = "*** GATE RE-ARMED: "
RE_ARM_SUFFIX = " ***"


class Gate:
    """One gate's decision: is the persisted approval still valid for THIS
    input, and what flows out if it is not.

    Args:
        fingerprint: identity of what is being approved (see `for_solve` /
            `for_image`).
        proceed: the persisted approval widget.
        approved_for: the fingerprint the current `proceed` was approved FOR
            (stamped by the node's ✅/▶ button).
        bypass: a node-declared widening that lets the value flow with no
            approval at all — AtlasAssessImage's `auto_continue`,
            AtlasSceneHealthGate's clean `pass_through_on_pass`. It affects
            what FLOWS, never what is reported.
        blank_is_unconditional: an empty `approved_for` with `proceed=True`
            is the manual override (the artist flipped the widget by hand).
            The viewport's patch branch sets this False: an extraction from
            before fingerprints existed carries no identity and must re-arm
            the pause rather than read as a blanket approval.
    """

    __slots__ = ("fingerprint", "proceed", "approved_for", "bypass",
                 "blank_is_unconditional")

    def __init__(self, fingerprint: str, *, proceed: bool = False,
                 approved_for: str = "", bypass: bool = False,
                 blank_is_unconditional: bool = True) -> None:
        self.fingerprint = fingerprint
        self.proceed = bool(proceed)
        self.approved_for = approved_for or ""
        self.bypass = bool(bypass)
        self.blank_is_unconditional = bool(blank_is_unconditional)

    # -- construction ----------------------------------------------------

    @classmethod
    def for_solve(cls, solve, source_image, **kwargs) -> "Gate":
        """Gate on (recovered camera + source image): a re-solve with different
        settings OR a swapped photo re-arms."""
        return cls(solve_fingerprint(solve, source_image), **kwargs)

    @classmethod
    def for_image(cls, image, **kwargs) -> "Gate":
        """Gate on the image alone — the approval token for a per-photo
        checkpoint that runs before any solve exists."""
        return cls(image_fingerprint(image), **kwargs)

    # -- the decision ----------------------------------------------------

    @property
    def approved(self) -> bool:
        """The arming comparison alone: did the artist approve THIS input."""
        if not self.proceed:
            return False
        if self.approved_for:
            return self.approved_for == self.fingerprint
        return self.blank_is_unconditional

    @property
    def re_armed(self) -> bool:
        """A live approval that belongs to a DIFFERENT input — the case the
        artist has to be told about. Deliberately independent of `bypass`: a
        warning may be overridden, never lost."""
        return (self.proceed and bool(self.approved_for)
                and self.approved_for != self.fingerprint)

    @property
    def passed(self) -> bool:
        """What actually flows: the approval, or the node's own bypass."""
        return self.bypass or self.approved

    def __bool__(self) -> bool:
        return self.passed

    # -- the visible explanation -----------------------------------------

    def re_arm_banner(self, tail: str) -> str:
        """`tail` framed as the gate family's re-arm sentence. Each node
        supplies its own wording — the frame is all that is shared."""
        return f"{RE_ARM_PREFIX}{tail}{RE_ARM_SUFFIX}"

    def annotate(self, report: str, tail: str, sep: str = "\n") -> str:
        """`report` with the re-arm banner in front of it when the gate
        re-armed, unchanged otherwise."""
        if not self.re_armed:
            return report
        return f"{self.re_arm_banner(tail)}{sep}{report}"

    # -- what leaves the node --------------------------------------------

    def route(self, value: Any) -> Any:
        """`value` when the gate passed; ComfyUI's silent ExecutionBlocker
        when it did not — falling back to `value` outside a ComfyUI runtime,
        where no blocker class exists."""
        if self.passed:
            return value
        blocker = _execution_blocker()
        return blocker if blocker is not None else value

    def route_each(self, values: Sequence[Any] | Iterable[Any]) -> tuple:
        """`route` across a whole output fan (the viewport's five patch_*
        slots) — one blocker instance shared by every slot."""
        vals = tuple(values)
        if self.passed:
            return vals
        blocker = _execution_blocker()
        if blocker is None:
            return vals
        return (blocker,) * len(vals)

    def envelope(self, report: str, result: Sequence[Any] | Iterable[Any],
                 ui: Mapping[str, Any] | None = None) -> dict:
        """The node return value: the report on the node plus the fingerprint
        the ✅/▶ button stamps back into `approved_for`.

        `ui` merges extra frontend keys (AtlasAssessImage's mirrored SAM
        prompts) WITHOUT displacing `text`/`fingerprint` — three JS
        extensions read those two by name.
        """
        payload: dict[str, Any] = {"text": [report],
                                   "fingerprint": [self.fingerprint]}
        if ui:
            payload.update(ui)
        return {"ui": payload, "result": tuple(result)}


__all__ = [
    "Gate",
    "RE_ARM_PREFIX",
    "RE_ARM_SUFFIX",
    "image_fingerprint",
    "solve_fingerprint",
]
