"""Tests for the Fixer render-repair inference module (docker-free parts).

The container invocation itself is exercised live (see the AtlasRenderFix
node's spike/verification notes); these tests pin the pure-Python contract:
clone resolution/error messages, the exact docker argv, and the bundled
transformer-engine shim the mounts depend on.
"""

import os

import pytest

from atlas_camera.inference.fixer_render_fix import (
    FIXER_DEFAULT_IMAGE,
    FIXER_MODEL_RELPATH,
    FIXER_SCRIPT_RELPATH,
    build_docker_command,
    resolve_fixer_root,
    shim_dir,
)


def _make_fixer_clone(root):
    (root / FIXER_MODEL_RELPATH).parent.mkdir(parents=True)
    (root / FIXER_MODEL_RELPATH).write_bytes(b"pkl")
    (root / FIXER_SCRIPT_RELPATH).parent.mkdir(parents=True)
    (root / FIXER_SCRIPT_RELPATH).write_text("# stub")
    return root


def test_resolve_missing_path_errors_with_instructions(monkeypatch):
    monkeypatch.delenv("ATLAS_FIXER_PATH", raising=False)
    with pytest.raises(RuntimeError) as exc:
        resolve_fixer_root("")
    msg = str(exc.value)
    assert "nv-tlabs/Fixer" in msg and "ATLAS_FIXER_PATH" in msg


def test_resolve_incomplete_clone_names_missing_files(tmp_path, monkeypatch):
    monkeypatch.delenv("ATLAS_FIXER_PATH", raising=False)
    with pytest.raises(RuntimeError) as exc:
        resolve_fixer_root(str(tmp_path))  # dir exists, no weights/script
    msg = str(exc.value)
    assert str(FIXER_MODEL_RELPATH) in msg
    assert "hf download nvidia/Fixer" in msg


def test_resolve_widget_wins_over_env(tmp_path, monkeypatch):
    good = _make_fixer_clone(tmp_path / "good")
    monkeypatch.setenv("ATLAS_FIXER_PATH", str(tmp_path / "nonexistent"))
    assert resolve_fixer_root(str(good)) == good


def test_resolve_env_fallback(tmp_path, monkeypatch):
    good = _make_fixer_clone(tmp_path / "good")
    monkeypatch.setenv("ATLAS_FIXER_PATH", str(good))
    assert resolve_fixer_root("") == good


def test_build_docker_command_contract(tmp_path):
    root = _make_fixer_clone(tmp_path / "Fixer")
    exchange = tmp_path / "exchange"
    cmd = build_docker_command(root, exchange, timestep=199)
    # Spike-proven invocation shape (no shell string — argv list, so Windows
    # paths never pass through MSYS/TCL-style mangling).
    assert cmd[:2] == ["docker", "run"]
    for flag in ("--rm", "--gpus=all", "--ipc=host"):
        assert flag in cmd
    assert "PYTHONPATH=/atlas_shim" in cmd
    assert f"{root}:/work" in cmd
    assert f"{exchange}:/exchange" in cmd
    assert f"{shim_dir()}:/atlas_shim" in cmd
    assert FIXER_DEFAULT_IMAGE in cmd
    inner = cmd[-1]
    assert cmd[-2] == "-c"  # image ENTRYPOINT is /bin/bash
    assert "--timestep 199" in inner
    assert "--input /exchange/in" in inner and "--output /exchange/out" in inner
    # container paths must be POSIX regardless of host OS
    assert "\\" not in inner


def test_shim_ships_with_package():
    d = shim_dir()
    shim = d / "sitecustomize.py"
    assert shim.is_file(), "TE-compat shim must ship as package data"
    text = shim.read_text()
    assert "apply_rotary_pos_emb" in text
    # the shim must never raise at interpreter startup
    assert "except Exception" in text
