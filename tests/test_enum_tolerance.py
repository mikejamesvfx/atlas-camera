"""ADR-001 step 4.2: this build carries the editor's vocabulary untouched.

Atlas Scene writes three `observation_state` values Atlas World never produces
-- `USER_CREATED`, `AGENT_CREATED`, `UNKNOWN` -- and they sit inside `entities[]`
and `planes[]`, structures this build already parses and believes it
understands. An unknown top-level key is obviously not ours; an unknown token in
a field we do recognise is not, which makes this the sharper half of 0.7
compatibility.

This build passes them through today, and it does so because `validate.py` has
no rule enumerating `observation_state` at all. That is tolerance by omission.
An omission is one well-meaning commit away from becoming a refusal, so the
tests below pin it as a decision: the field is deliberately not enumerated, and
here is the reason.

What is NOT tolerated, equally deliberately: `completion_policy`, `alpha_mode`,
`scale.status` and `material.projection.role`. Those are instructions rather
than descriptions. This build refuses an unrecognised value in each, and the
message in `validate.py` says why -- treating an unevaluable policy as `none`
would look identical to a correctly classified surface. 4.2 did not loosen them
and this file proves it.

Not settled here: who owns the vocabulary, or whether the editor-only states
should have an upstream home. Carrying a token is not agreeing what it means.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from atlas_camera.format import SCENE_DOCUMENT, validate_document
from atlas_camera.format.container import pack_archive, unpack_archive
from atlas_camera.format.package import open_package, read_document
from atlas_camera.format.validate import (
    ALPHA_MODES,
    COMPLETION_POLICIES,
    PROJECTION_ROLES,
    SCALE_STATUSES,
    collect_problems,
)

FIXTURE = Path(__file__).parent / "fixtures" / "scene_0_7_street_min.atlas"

#: Every value Atlas Scene's ObservationState carries, upstream six first.
#: Restated rather than imported because this build must not depend on the
#: editor to run its own tests -- but `test_the_editor_vocabulary_has_not_moved`
#: checks the list against the real thing when the editor is present.
SCENE_OBSERVATION_STATES = [
    "GROUND_TRUTH",
    "OBSERVED",
    "SOLVED",
    "INFERRED",
    "GENERATED",
    "HUMAN_VERIFIED",
    "USER_CREATED",
    "AGENT_CREATED",
    "UNKNOWN",
]

EDITOR_ONLY = {"USER_CREATED", "AGENT_CREATED", "UNKNOWN"}

FOREIGN_TOKENS = ["FUTURE_STATE", "vendor_specific", "x-experimental.2"]


@pytest.fixture
def package(tmp_path: Path) -> Path:
    target = tmp_path / "street.atlas"
    shutil.copytree(FIXTURE, target)
    return target


def with_state(package: Path, state: str) -> dict:
    """Put `state` on an entity and a plane, then read the document back."""

    document = json.loads((package / SCENE_DOCUMENT).read_text(encoding="utf-8"))
    document.setdefault("entities", []).append({
        "entity_id": "probe",
        "kind": "proxy",
        "observation_state": state,
        "parent_id": None,
    })
    if document.get("planes"):
        document["planes"][0]["provenance"] = state
    (package / SCENE_DOCUMENT).write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    return document


# ------------------------------------------------- the editor's vocabulary


@pytest.mark.parametrize("state", SCENE_OBSERVATION_STATES)
def test_every_editor_state_validates(package: Path, state: str):
    with_state(package, state)
    validate_document(read_document(package))


@pytest.mark.parametrize("state", SCENE_OBSERVATION_STATES)
def test_every_editor_state_survives_an_edit(package: Path, state: str):
    with_state(package, state)

    with open_package(package) as opened:
        opened.document["scale"] = {"status": "measured", "confidence": 0.8}
        opened.commit()

    document = read_document(package)
    probe = [e for e in document["entities"] if e["entity_id"] == "probe"]
    assert probe and probe[0]["observation_state"] == state
    if document.get("planes"):
        assert document["planes"][0]["provenance"] == state


def test_the_editor_only_states_are_the_ones_that_matter(package: Path):
    """Named on their own because they have no upstream home at all."""

    for state in sorted(EDITOR_ONLY):
        with_state(package, state)
        validate_document(read_document(package))


@pytest.mark.parametrize("token", FOREIGN_TOKENS)
def test_a_token_no_build_knows_still_survives(package: Path, token: str):
    """Tolerance is not a list of the editor's current values.

    A vocabulary that only survives while both sides agree on it is not
    tolerance, it is luck with a test attached.
    """

    with_state(package, token)
    validate_document(read_document(package))

    with open_package(package) as opened:
        opened.document["scale"] = {"status": "manual", "confidence": None}
        opened.commit()

    probe = [e for e in read_document(package)["entities"] if e["entity_id"] == "probe"]
    assert probe and probe[0]["observation_state"] == token


def test_states_survive_the_container_too(package: Path, tmp_path: Path):
    with_state(package, "AGENT_CREATED")
    archive = tmp_path / "packed.atlas"
    pack_archive(package, archive)

    with open_package(archive) as opened:
        opened.document["scale"] = {"status": "measured", "confidence": 0.5}
        opened.commit()

    extracted = tmp_path / "out"
    unpack_archive(archive, extracted)
    probe = [e for e in read_document(extracted)["entities"] if e["entity_id"] == "probe"]
    assert probe and probe[0]["observation_state"] == "AGENT_CREATED"


# ------------------------------------------- tolerance pinned as a decision


def test_observation_state_is_deliberately_not_enumerated(package: Path):
    """Tolerance by omission, made explicit so nobody 'fixes' it.

    Adding a rule that enumerates this field would make every package carrying
    an editor-authored entity fail to validate here. If that is ever wanted, it
    needs a decision and an ADR, not a tidy-up.
    """

    with_state(package, "SOMETHING_NO_BUILD_KNOWS")
    problems = collect_problems(read_document(package))
    assert not [p for p in problems if "observation_state" in p or "provenance" in p]


def test_the_actionable_vocabularies_are_still_enforced(package: Path):
    """4.2 loosened descriptions, not instructions. Each of the four, by name."""

    document = json.loads((package / SCENE_DOCUMENT).read_text(encoding="utf-8"))

    if document.get("planes"):
        broken = json.loads(json.dumps(document))
        broken["planes"][0]["completion_policy"] = "SOMETHING_NEW"
        assert [p for p in collect_problems(broken) if "completion_policy" in p]

    if document.get("layers"):
        broken = json.loads(json.dumps(document))
        broken["layers"][0]["alpha_mode"] = "SOMETHING_NEW"
        assert [p for p in collect_problems(broken) if "alpha_mode" in p]

    broken = json.loads(json.dumps(document))
    broken["scale"] = {"status": "SOMETHING_NEW"}
    assert [p for p in collect_problems(broken) if "scale.status" in p]

    broken = json.loads(json.dumps(document))
    broken.setdefault("entities", []).append({
        "entity_id": "proj",
        "parent_id": None,
        "material": {"projection": {"role": "SOMETHING_NEW"}},
    })
    assert [p for p in collect_problems(broken) if "role" in p]


def test_the_four_actionable_vocabularies_have_not_quietly_grown():
    """Their contents are the contract; a new value is a format change."""

    assert set(COMPLETION_POLICIES) == {
        "none", "extend_plane", "room_envelope", "extrude_profile",
        "conservative_proxy", "bridge_discontinuity", "backdrop",
    }
    assert set(SCALE_STATUSES) == {"measured", "manual", "assumed", "unknown"}
    assert set(ALPHA_MODES) == {"straight", "associated"}
    assert set(PROJECTION_ROLES) == {"foreground", "cleanplate_background"}


# ------------------------------------------------- drift against the editor


def test_the_editor_vocabulary_has_not_moved():
    """Check the restated list against the real editor when it is present.

    Skips rather than fails without an Atlas Scene checkout, which is exactly
    the silent-skip problem ADR-001 step 4.3 exists to fix. Until then this is
    an alarm that works on a developer machine and nowhere else, and saying so
    here is better than implying coverage it does not have.
    """

    types = pytest.importorskip(
        "atlas_scene.scene.types",
        reason="no Atlas Scene checkout on sys.path (see ADR-001 step 4.3)",
    )
    assert [s.value for s in types.ObservationState] == SCENE_OBSERVATION_STATES
