"""Every Atlas artifact stamped a schema_version and nothing ever read one.

The stamp is the hard half and it was already there — on AtlasSolve (0.2), on
DynamicPlate (0.1), with `atlas_version` alongside recording which build wrote
the file. What was missing was any reader comparing one, so a format bump
would have been absorbed silently. These pin the check.
"""
from __future__ import annotations

import pytest

from atlas_camera.core.base import check_schema_version


def test_missing_version_is_accepted():
    """Solve JSON is a USER artifact; files predate the check."""
    assert check_schema_version(None, expected="0.2", where="x") is None
    assert check_schema_version("", expected="0.2", where="x") is None


def test_same_version_is_silent():
    assert check_schema_version("0.2", expected="0.2", where="x") is None


def test_a_higher_minor_loads_with_a_warning():
    """Minor bumps are additive by convention — refusing them would make every
    new optional field a breaking change."""
    warn = check_schema_version("0.3", expected="0.2", where="solve JSON")
    assert warn and "newer minor" in warn


def test_a_lower_minor_is_silent():
    assert check_schema_version("0.1", expected="0.2", where="x") is None


def test_a_different_major_raises():
    """The case where fields changed meaning. Reading it silently is how a
    plate ends up in the wrong space with no error."""
    with pytest.raises(ValueError, match="major"):
        check_schema_version("1.0", expected="0.2", where="solve JSON")


def test_an_unparseable_version_raises():
    with pytest.raises(ValueError, match="unreadable schema_version"):
        check_schema_version("banana", expected="0.2", where="x")


def test_the_solve_loader_refuses_a_future_major():
    from atlas_camera.core.schema import LatentScene
    from test_blender_measured_bridge import _solve

    data = _solve().to_dict()
    assert data["schema_version"] == LatentScene.schema_version
    LatentScene.from_dict(data)                      # round-trips today

    data["schema_version"] = "9.0"
    with pytest.raises(ValueError, match="solve JSON"):
        LatentScene.from_dict(data)


def test_the_dynamic_plate_loader_refuses_a_future_major():
    from atlas_camera.core.dynamic_plate import DynamicPlate

    payload = {"plate_id": "WATER_0001", "semantic_type": "water",
               "schema_version": "9.0"}
    with pytest.raises(ValueError, match="dynamic plate manifest"):
        DynamicPlate.from_dict(payload)
