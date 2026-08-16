"""Shared recovered-object contracts for Atlas core types."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from atlas_camera.core.confidence import ConfidenceModel

RecoveredObjectT = TypeVar("RecoveredObjectT", bound="RecoveredObject")


def check_schema_version(got: Any, *, expected: str, where: str) -> str | None:
    """Refuse an artifact whose MAJOR schema version this build cannot read.

    Every Atlas artifact stamps ``schema_version`` on write and, until
    2026-08-17, nothing compared one on read — the stamp was the hard half and
    the check was simply never added. This is that check, deliberately narrow:

    * a MISSING version is accepted. Solve JSON is a USER artifact — exported
      solves, review packages, saved projects — and refusing files written
      before the check existed would break loading them for no safety gain.
      Same reasoning as `.gitignore`-adjacent back-compat, not laziness.
    * a matching MAJOR is accepted, including a HIGHER minor. Minor bumps are
      additive by convention, and a reader that refuses 0.3 because it knows
      0.2 makes every additive field a breaking change.
    * a different MAJOR raises. That is the case where fields have changed
      meaning, and reading it silently is how a plate ends up in the wrong
      space with no error.

    Returns a warning string when the minor is ahead (the caller decides
    whether to surface it), else None.
    """
    if got in (None, ""):
        return None
    got_s, exp_s = str(got), str(expected)

    def major_minor(v: str) -> tuple[int, int]:
        head, _, tail = v.partition(".")
        try:
            return int(head), int(tail or 0)
        except ValueError:
            raise ValueError(
                f"{where}: unreadable schema_version {v!r} (expected MAJOR.MINOR)"
            ) from None

    got_mj, got_mn = major_minor(got_s)
    exp_mj, exp_mn = major_minor(exp_s)
    if got_mj != exp_mj:
        raise ValueError(
            f"{where}: schema_version {got_s} has major {got_mj}, this build reads "
            f"{exp_s} (major {exp_mj}). Fields have changed meaning across that "
            "boundary — re-export the artifact from a matching build rather than "
            "loading it."
        )
    if got_mn > exp_mn:
        return (f"{where}: written by a newer minor schema ({got_s} > {exp_s}); "
                "loading anyway — unknown fields are ignored")
    return None


@runtime_checkable
class RecoveredObject(Protocol):
    """Minimal shared surface for concrete recovered objects."""

    schema_version: str
    confidence: ConfidenceModel

    def to_dict(self) -> dict[str, Any]:
        ...

    @classmethod
    def from_dict(cls: type[RecoveredObjectT], data: dict[str, Any]) -> RecoveredObjectT:
        ...
