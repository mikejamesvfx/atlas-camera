"""Tests for atlas_camera.ui.api's exception-handling boundary.

Regression coverage for narrowing `except Exception` (which masked genuine
internal bugs as client 400s) down to the specific exception types
project.py/solver.py/multimodal_helper.py actually raise for expected,
client-input-shaped failures (ValueError, FileNotFoundError, RuntimeError).
Anything else must now propagate as an unhandled exception, which FastAPI's
TestClient re-raises by default (matching what a real deployment's ASGI
server would turn into a 500 response).
"""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from atlas_camera.ui import api as api_module
from atlas_camera.ui.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_expected_value_error_is_reported_as_400(client, monkeypatch):
    def fake_solve_project(project_dir):
        raise ValueError("bad project state")

    monkeypatch.setattr(api_module, "solve_project", fake_solve_project)

    response = client.post("/api/solve", json={"project_dir": "/some/project"})
    assert response.status_code == 400
    assert "bad project state" in response.json()["detail"]


def test_expected_file_not_found_is_reported_as_400(client, monkeypatch):
    def fake_solve_project(project_dir):
        raise FileNotFoundError("Project has no solved atlas_solve.json.")

    monkeypatch.setattr(api_module, "solve_project", fake_solve_project)

    response = client.post("/api/solve", json={"project_dir": "/some/project"})
    assert response.status_code == 400


def test_genuine_internal_bug_is_not_masked_as_400(client, monkeypatch):
    # A TypeError/AttributeError-shaped bug in the underlying implementation
    # must NOT come back as a 400 "client error" — that's exactly the
    # confusion the broad `except Exception` used to cause, misdirecting
    # debugging effort during an actual regression.
    def fake_solve_project(project_dir):
        raise TypeError("unsupported operand type(s) for +: 'NoneType' and 'int'")

    monkeypatch.setattr(api_module, "solve_project", fake_solve_project)

    with pytest.raises(TypeError):
        client.post("/api/solve", json={"project_dir": "/some/project"})
