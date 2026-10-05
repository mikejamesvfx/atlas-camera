"""ADR-001: Atlas Camera reads .atlas 0.7 without damaging what it does not own.

The property under test, in full:

    Given a canonical Scene-produced .atlas 0.7 package containing Scene-owned
    fields and artifacts unknown to Atlas Camera, when Atlas Camera reads it,
    modifies only Camera-owned state, and writes or repackages the document,
    then all unrelated valid 0.7 content remains byte-identical where possible,
    or semantically identical where repacking necessarily changes
    representation.

Two fixtures stand behind it. `scene_0_7_street_min.atlas` is a REAL package,
copied verbatim out of Atlas Scene's own test fixtures, so these tests are
anchored to what Scene actually writes rather than to this repository's idea of
it. The hard case is built on top of it, because no fixture anywhere exercises a
populated environment: Scene's own suite only constructs those in-test.

Why the hard case has to be synthesised, and what it carries: a `physical_sky`
atmosphere with its dataset digest, an HDRI, a sky mask, populated
`capture_location` and `capture_time`, a `director_take` entry in `derived`
whose `depends_on` keys are not digests of anything this build knows, the
editor-only observation states `USER_CREATED` and `AGENT_CREATED`, and side
directories `environments/`, `takes/` and `director/`. Every shape here is
copied from Atlas Scene's `environment/model.py`, `scene/camera.py` and
`operations/take_ops.py`; if Scene changes them this file goes stale, and the
honest answer to that is the conformance test in Scene's own suite, not a
second mirror here.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from atlas_camera.exporters.atlas_package import PackageExists, write_atlas_archive
from atlas_camera.format import (
    SCENE_DOCUMENT,
    SCHEMA_VERSION,
    validate_document,
)
from atlas_camera.format.container import pack_archive, unpack_archive
from atlas_camera.format.package import (
    UncommittedChanges,
    open_package,
    read_document,
    write_document,
)
from atlas_camera.format.version import UnsupportedVersion, check_readable

FIXTURE = Path(__file__).parent / "fixtures" / "scene_0_7_street_min.atlas"

#: 0.7 minus 0.6, as Atlas Scene's own conformance test enumerates it.
SCENE_ADDED_TOP_LEVEL = "environment"
SCENE_ADDED_CAMERA = ("capture_location", "capture_time")


# ----------------------------------------------------------------- fixtures


@pytest.fixture
def street(tmp_path: Path) -> Path:
    """A writable copy of the real Scene-authored 0.7 package."""

    target = tmp_path / "street.atlas"
    shutil.copytree(FIXTURE, target)
    return target


@pytest.fixture
def loaded(street: Path) -> Path:
    """The same package, with every 0.7 feature Scene can put in one.

    Populated rather than defaulted, because a block of nulls would pass a
    preservation test that a real environment fails.
    """

    document = json.loads((street / SCENE_DOCUMENT).read_text(encoding="utf-8"))

    document["environment"] = {
        "source": "physical_sky",
        "atmosphere": {
            "model": "prague_sky",
            "mode": "spectral",
            "dataset": {
                "kind": "prague",
                "path": "environments/prague_sky.dat",
                "version": "1.2",
                "sha256": "a" * 64,
                "source": "vendored",
            },
            "visibility_km": {"value": 23.5, "provenance": "measured",
                              "source": "exif", "confidence": 0.8},
            "ground_albedo": {"value": 0.21, "provenance": "inferred",
                              "source": "vlm", "confidence": 0.4},
            "observer_altitude_m": {"value": 18.0, "provenance": "measured",
                                    "source": "gps", "confidence": 0.9},
            "exposure_ev": {"value": -1.5, "provenance": "user_override",
                            "source": "artist", "confidence": None},
            "spectral_mode": "full",
            "wavelength_nm": [380, 480, 580, 680],
            "sky_radiance_enabled": True,
            "sun_radiance_enabled": True,
            "transmittance_enabled": True,
            "polarisation_enabled": False,
        },
        "sun": {
            "mode": "offset",
            "derived": {"azimuth_deg": 118.4, "elevation_deg": 37.2,
                        "provenance": "derived", "source": "capture_time",
                        "confidence": 0.75},
            "offset": {"azimuth_deg": -4.0, "elevation_deg": 1.5},
            "manual": None,
        },
        "hdri": {
            "path": "environments/dome.exr",
            "rotation_deg": 42.0,
            "exposure_ev": 0.5,
            "flip_horizontal": False,
            "vertical_correction_deg": -1.25,
            "alignment": "horizon",
            "provenance": "user_override",
        },
        "sky_mask": {
            "source": "segmentation",
            "path": "environments/sky_mask.png",
            "feather_px": 4,
            "erode_px": 1,
            "dilate_px": 0,
            "confidence": 0.92,
            "provenance": "derived",
        },
        "provenance": [
            {"at": "2026-09-12T04:05:06+00:00", "by": "editor", "what": "sun solved"},
        ],
    }

    document["camera"]["capture_location"] = {
        "latitude_deg": -34.1358,
        "longitude_deg": 150.9321,
        "altitude_m": 18.0,
        "provenance": "measured",
        "source": "exif_gps",
        "confidence": 0.95,
    }
    document["camera"]["capture_time"] = {
        "utc": "2026-04-02T23:41:18+00:00",
        "local": "2026-04-03T09:41:18",
        "timezone": "Australia/Sydney",
        "provenance": "measured",
        "source": "exif",
        "confidence": 1.0,
    }

    # An editor-authored entity, carrying observation states that have no
    # upstream home. These sit inside structures this build already parses,
    # which makes them the sharper hazard: an unknown top-level key is
    # obviously not ours, an unknown enum value in a known field is not.
    document["entities"].append({
        "entity_id": "editor_block",
        "kind": "proxy",
        "observation_state": "USER_CREATED",
        "parent_id": None,
        "provenance": [{"by": "artist"}],
    })
    document["entities"].append({
        "entity_id": "agent_block",
        "kind": "proxy",
        "observation_state": "AGENT_CREATED",
        "parent_id": None,
        "provenance": [{"by": "agent"}],
    })

    # A director take. Scene deliberately records these as derived artifacts
    # rather than a top-level field, so a reader walking `derived` meets a kind
    # it does not know and a depends_on whose keys digest nothing it can check.
    document["derived"].append({
        "kind": "director_take",
        "artifact_id": "STREET_010_A_take03",
        "path": "takes/street/010/A_take03/manifest.json",
        "depends_on": {"input": "b" * 64, "seed": "20260912"},
    })

    (street / SCENE_DOCUMENT).write_text(
        json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    # Side files in directories this build has never heard of.
    for relative, payload in (
        ("environments/dome.exr", b"\x76\x2f\x31\x01 not really an exr"),
        ("environments/sky_mask.png", b"\x89PNG\r\n\x1a\n not really a png"),
        ("environments/prague_sky.dat", b"\x00\x01\x02\x03"),
        ("takes/street/010/A_take03/manifest.json", b'{"schemaVersion": 1}\n'),
        ("takes/street/010/A_take03/samples.json", b'{"samples": []}\n'),
        ("director/session.json", b'{"schemaVersion": 1, "fps": 24.0}\n'),
    ):
        path = street / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    return street


# ----------------------------------------------------------------- helpers


def snapshot(root: Path) -> dict[str, bytes]:
    """Every file in the tree, by relative path, as raw bytes."""

    return {
        item.relative_to(root).as_posix(): item.read_bytes()
        for item in sorted(root.rglob("*"))
        if item.is_file()
    }


def edit_camera_owned_state(document: dict) -> None:
    """The 'modifies only Camera-owned state' half of the property.

    `scale` is this build's own concept -- it is what `core.scene_health`
    produces and what this build's writer emits -- so changing it is a genuine
    Camera edit and touches nothing Scene owns.
    """

    document["scale"] = {
        "status": "measured",
        "scale_source": "reference_object",
        "confidence": 0.8,
    }


# ------------------------------------------------- step 1: the version gate


def test_the_reader_accepts_0_7():
    assert check_readable("0.7") == "0.7"


def test_the_writer_has_been_promoted():
    """ADR-001 step 4.4, landed 2026-09-13 once 4.1 to 4.3 were green."""

    assert SCHEMA_VERSION == "0.7"


def test_historical_versions_are_still_readable():
    """Promotion retires 0.6 as a PRODUCER version, not as a readable one.

    Every package written before the promotion is still openable, which is the
    whole reason the readable set and the writer are separate constants.
    """

    for version in ("0.2", "0.3", "0.4", "0.5", "0.6"):
        assert check_readable(version) == version


def test_a_written_document_does_not_claim_content_it_lacks(loaded: Path, tmp_path: Path):
    """0.7 is a reading contract, not a promise to use every optional field.

    This build writes 0.7 and still does not emit `environment` or the camera's
    capture provenance, because those default correctly by their absence: an
    Atlas Camera package is plate-based, which is what a missing `environment`
    asserts. Writing them would be new serialisation behaviour. The test exists
    so that stays a decision rather than drift.
    """

    fresh = tmp_path / "fresh.atlas"
    write_atlas_archive(_minimal_solve(), fresh)

    extracted = tmp_path / "fresh_out"
    unpack_archive(fresh, extracted)
    document = read_document(extracted)

    assert document["schema_version"] == "0.7"
    assert "environment" not in document
    assert "capture_location" not in document["camera"]


def test_an_unknown_version_is_still_refused():
    """Widening the reader must not have turned the gate into a shrug."""

    with pytest.raises(UnsupportedVersion):
        check_readable("0.8")


# --------------------------------------- step 2: a real 0.7 package opens


def test_a_scene_authored_0_7_package_validates(street: Path):
    """The round-trip half of ADR-001 step 2, from Scene's side to ours."""

    validate_document(read_document(street))


def test_reading_does_not_rewrite_the_version_to_ours(street: Path):
    document = read_document(street)
    assert document["schema_version"] == "0.7"


def test_the_0_7_delta_is_the_three_fields_scene_says_it_is(street: Path):
    """Pin the delta, so a later version cannot widen it unnoticed.

    Atlas Scene's own conformance test enumerates 0.7 minus 0.6 by popping
    exactly these three. If that ever grows, this build's claim to read 0.7
    safely needs re-examining rather than extending by assumption.
    """

    document = read_document(street)
    assert SCENE_ADDED_TOP_LEVEL in document
    for field in SCENE_ADDED_CAMERA:
        assert field in document["camera"]


# ----------------------------------- step 3: the preservation property


def test_a_no_op_read_and_write_is_byte_identical(loaded: Path):
    """The floor. If this fails, nothing below it means anything."""

    before = (loaded / SCENE_DOCUMENT).read_bytes()
    write_document(loaded, read_document(loaded))
    assert (loaded / SCENE_DOCUMENT).read_bytes() == before


def test_editing_camera_owned_state_preserves_everything_else(loaded: Path):
    """The property, on an extracted tree."""

    before = json.loads((loaded / SCENE_DOCUMENT).read_text(encoding="utf-8"))

    with open_package(loaded) as package:
        edit_camera_owned_state(package.document)
        package.commit()

    after = json.loads((loaded / SCENE_DOCUMENT).read_text(encoding="utf-8"))

    assert after["scale"] != before["scale"], "the edit did not happen"
    for key in before:
        if key == "scale":
            continue
        assert after[key] == before[key], f"{key} was not preserved"
    assert list(after) == list(before), "key order is part of the format"


def test_the_hard_fixture_actually_carries_0_7_content():
    """Guard against the comparisons below passing on two absent fields.

    `after["environment"] == before["environment"]` is true when neither
    document has an environment at all, so the preservation assertions are only
    worth anything if the fixture is genuinely populated. This test is what
    makes them non-vacuous, and it takes the fixture through the same builder
    the other tests use rather than trusting the file on disk.
    """

    # Built here rather than via the fixture so this test stands alone.
    import tempfile

    with tempfile.TemporaryDirectory() as staging:
        target = Path(staging) / "street.atlas"
        shutil.copytree(FIXTURE, target)
        document = json.loads((target / SCENE_DOCUMENT).read_text(encoding="utf-8"))

    assert document["schema_version"] == "0.7"
    assert SCENE_ADDED_TOP_LEVEL in document
    for field in SCENE_ADDED_CAMERA:
        assert field in document["camera"]


def test_editing_preserves_the_environment_block_exactly(loaded: Path):
    """Named separately because it is the headline of 0.7."""

    before = json.loads((loaded / SCENE_DOCUMENT).read_text(encoding="utf-8"))

    # Non-vacuity, stated inline: these must be real content, not two Nones.
    assert before["environment"]["source"] == "physical_sky"
    assert before["environment"]["atmosphere"]["dataset"]["sha256"]
    assert before["camera"]["capture_location"]["latitude_deg"]
    assert before["camera"]["capture_time"]["timezone"]

    with open_package(loaded) as package:
        edit_camera_owned_state(package.document)
        package.commit()

    after = json.loads((loaded / SCENE_DOCUMENT).read_text(encoding="utf-8"))
    assert after["environment"] == before["environment"]
    assert after["camera"]["capture_location"] == before["camera"]["capture_location"]
    assert after["camera"]["capture_time"] == before["camera"]["capture_time"]


def test_editing_preserves_editor_only_observation_states(loaded: Path):
    """USER_CREATED and AGENT_CREATED have no upstream home. Carry them anyway."""

    with open_package(loaded) as package:
        edit_camera_owned_state(package.document)
        package.commit()

    document = read_document(loaded)
    states = {entity.get("observation_state") for entity in document["entities"]}
    assert {"USER_CREATED", "AGENT_CREATED"} <= states


def test_editing_does_not_downgrade_the_schema_version(loaded: Path):
    """Preserving means NOT stamping our own version on someone else's document.

    Atlas Scene re-emits every scene it saves at its own constant. This build
    must not do the mirror image of that, or a 0.7 package would come back from
    a Camera edit claiming to be 0.6 while still carrying 0.7 content.
    """

    with open_package(loaded) as package:
        edit_camera_owned_state(package.document)
        package.commit()

    assert read_document(loaded)["schema_version"] == "0.7"


def test_side_files_in_unknown_directories_survive(loaded: Path):
    """environments/, takes/ and director/ are none of this build's business."""

    before = snapshot(loaded)

    with open_package(loaded) as package:
        edit_camera_owned_state(package.document)
        package.commit()

    after = snapshot(loaded)
    assert set(after) == set(before), "a file appeared or vanished"
    for relative, payload in before.items():
        if relative == SCENE_DOCUMENT:
            continue
        assert after[relative] == payload, f"{relative} changed"


# ----------------------------------------- the same, through the container


def test_the_property_holds_through_a_packed_archive(loaded: Path, tmp_path: Path):
    """Repacking is where 'byte-identical where possible' earns its hedge.

    The zip's own bytes need not match -- timestamps and ordering are the
    container's business -- but every member's content must.
    """

    archive = tmp_path / "street_packed.atlas"
    pack_archive(loaded, archive)
    before = snapshot(loaded)

    with open_package(archive) as package:
        edit_camera_owned_state(package.document)
        package.commit()

    extracted = tmp_path / "extracted"
    unpack_archive(archive, extracted)
    after = snapshot(extracted)

    assert set(after) == set(before)
    for relative, payload in before.items():
        if relative == SCENE_DOCUMENT:
            continue
        assert after[relative] == payload, f"{relative} changed through the container"

    document = json.loads(after[SCENE_DOCUMENT])
    assert document["schema_version"] == "0.7"
    assert document["environment"] == json.loads(before[SCENE_DOCUMENT])["environment"]
    assert document["scale"]["status"] == "measured"


# ------------------------------------------------------- opening is safe


def test_opening_for_inspection_writes_nothing(loaded: Path):
    before = snapshot(loaded)

    with open_package(loaded) as package:
        assert package.document["schema_version"] == "0.7"
        assert not package.dirty

    assert snapshot(loaded) == before


def test_opening_an_archive_for_inspection_does_not_touch_it(loaded: Path, tmp_path: Path):
    archive = tmp_path / "street_packed.atlas"
    pack_archive(loaded, archive)
    before = archive.read_bytes()

    with open_package(archive) as package:
        assert package.document["environment"]["source"] == "physical_sky"

    assert archive.read_bytes() == before


# ------------------------------------------------------- dirty tracking


def test_an_uncommitted_edit_is_refused_not_silently_dropped(loaded: Path):
    with pytest.raises(UncommittedChanges):
        with open_package(loaded) as package:
            edit_camera_owned_state(package.document)

    assert read_document(loaded)["scale"] is None, "nothing should have been written"


def test_a_discarded_edit_closes_quietly(loaded: Path):
    with open_package(loaded) as package:
        edit_camera_owned_state(package.document)
        package.discard()

    assert read_document(loaded)["scale"] is None


def test_an_edit_reverted_by_hand_is_not_dirty(loaded: Path):
    with open_package(loaded) as package:
        original = package.document["scale"]
        edit_camera_owned_state(package.document)
        assert package.dirty
        package.document["scale"] = original
        assert not package.dirty


def test_an_exception_inside_the_block_propagates_unchanged(loaded: Path):
    """The interesting failure is the caller's, not ours."""

    with pytest.raises(ZeroDivisionError):
        with open_package(loaded) as package:
            edit_camera_owned_state(package.document)
            raise ZeroDivisionError("the caller's problem")


# --------------------------------------------------- create versus edit


def test_the_archive_writer_refuses_an_existing_package(loaded: Path, tmp_path: Path):
    """The footgun: this writer rebuilds, so aimed at a package it destroys it."""

    archive = tmp_path / "street_packed.atlas"
    pack_archive(loaded, archive)

    with pytest.raises(PackageExists):
        write_atlas_archive(_minimal_solve(), archive)


def test_explicit_overwrite_is_a_destructive_replace(loaded: Path, tmp_path: Path):
    """Allowed, but it replaces -- it does not merge. Say so in a test."""

    archive = tmp_path / "street_packed.atlas"
    pack_archive(loaded, archive)

    write_atlas_archive(_minimal_solve(), archive, overwrite=True)

    extracted = tmp_path / "after"
    unpack_archive(archive, extracted)
    document = read_document(extracted)
    assert document["schema_version"] == SCHEMA_VERSION
    assert "environment" not in document, (
        "overwrite=True is documented as destructive; if this starts passing "
        "Scene content through, the docstring is now a lie"
    )


def test_the_directory_writer_refuses_an_existing_package(loaded: Path):
    from atlas_camera.exporters.atlas_package import write_atlas_package

    with pytest.raises(PackageExists):
        write_atlas_package(_minimal_solve(), loaded)


def test_an_empty_directory_is_not_an_existing_package(tmp_path: Path):
    """The archive writer's temporary tree must still work."""

    from atlas_camera.exporters.atlas_package import write_atlas_package

    empty = tmp_path / "fresh"
    empty.mkdir()
    write_atlas_package(_minimal_solve(), empty)
    assert (empty / SCENE_DOCUMENT).is_file()


def _minimal_solve():
    """The smallest solve the package writer accepts."""

    from test_atlas_format import solve_with, wall

    return solve_with(wall())
