"""Contract for the phone -> ComfyUI bridge (tools/record3d_bridge.py).

The bridge had NO tests until now, and it shows in its history: a stale Dropbox
folder left behind by an uninstalled client was auto-selected as a destination,
and captures that arrived while the script was down were ignored forever. Both
were found by running it by hand. These pin the parts that fail silently.
"""
from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "record3d_bridge", ROOT / "tools" / "record3d_bridge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bridge = _load()
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")


def _png(path: Path, w: int, h: int) -> Path:
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(path)
    return path


def _r3d(path: Path, *, frames: int = 1, valid: bool = True) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        if valid:
            zf.writestr("metadata", json.dumps({
                "w": 64, "h": 48, "dw": 16, "dh": 12,
                "K": [50.0, 0, 0, 0, 50.0, 0, 32.0, 24.0, 1.0],
                "poses": [[0, 0, 0, 1, 0, 1.5, 0]] * frames}))
            for i in range(frames):
                zf.writestr(f"rgbd/{i}.jpg", b"notreallyajpeg")
        else:
            zf.writestr("something_else", b"x")
    return path


# ------------------------------------------------------------- classification


def test_r3d_routes_to_the_measured_graph(tmp_path):
    kind, _ = bridge.classify(_r3d(tmp_path / "scan.r3d"))
    assert kind == "record3d"


def test_single_frame_capture_is_not_special_cased(tmp_path):
    """A framed still from a phone app IS a one-frame .r3d.

    The whole reason no new payload format is needed: Record3DCapture.open()
    requires only w/h and a pose, so a single-frame capture takes the same route
    as a room scan. If this ever needs a branch, the format decision changed.
    """
    scan = bridge.classify(_r3d(tmp_path / "scan.r3d", frames=30))[0]
    still = bridge.classify(_r3d(tmp_path / "still.r3d", frames=1))[0]
    assert scan == still == "record3d"


@pytest.mark.parametrize("w,h", [(2048, 1024), (8192, 4096), (1000, 500)])
def test_two_to_one_images_route_to_the_panorama_graph(tmp_path, w, h):
    kind, why = bridge.classify(_png(tmp_path / f"p{w}.png", w, h))
    assert kind == "panorama", why


@pytest.mark.parametrize("w,h", [(1920, 1080), (4032, 3024), (1024, 1024), (1080, 1920)])
def test_ordinary_photos_route_to_the_still_graph(tmp_path, w, h):
    kind, why = bridge.classify(_png(tmp_path / f"s{w}x{h}.png", w, h))
    assert kind == "still", why


def test_panorama_band_is_tight_enough_to_exclude_wide_photos(tmp_path):
    """Widening the band is how a wide photo starts getting solved as a 360.

    2.35:1 is a normal cinematic crop and must NOT be treated as equirectangular
    — the result would be torn geometry that reads as a solver bug.
    """
    assert bridge.classify(_png(tmp_path / "scope.png", 2350, 1000))[0] == "still"
    assert bridge.classify(_png(tmp_path / "wide.png", 1800, 1000))[0] == "still"


def test_exr_panoramas_are_sized_without_pillow(tmp_path):
    """Stock Pillow cannot read EXR — and EXR is what the 8K panoramas arrive in.

    Regression: with Pillow alone every equirect plate returned None and
    classified as `unknown`, so the panorama route would have shipped silently
    doing nothing. OpenImageIO reads the header only; decoding a 200 MB plate
    just to learn its aspect would stall the poll loop on every pass.
    """
    oiio = pytest.importorskip("OpenImageIO")

    path = tmp_path / "pano.exr"
    spec = oiio.ImageSpec(2048, 1024, 3, "half")
    out = oiio.ImageOutput.create(str(path))
    assert out is not None, "OpenImageIO cannot write EXR here"
    out.open(str(path), spec)
    out.write_image(np.zeros((1024, 2048, 3), dtype=np.float16))
    out.close()

    from PIL import Image as _PIL
    with pytest.raises(Exception):
        _PIL.open(path).size          # the reason the fallback has to exist

    assert bridge.image_size(path) == (2048, 1024)
    assert bridge.classify(path)[0] == "panorama"


def test_filename_never_decides_the_route(tmp_path):
    """Dimensions are read, not guessed from the name."""
    liar = _png(tmp_path / "panorama_360_equirect.png", 1920, 1080)
    assert bridge.classify(liar)[0] == "still"


def test_unreadable_and_unknown_files_are_reported_not_guessed(tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("hello", encoding="utf-8")
    assert bridge.classify(junk)[0] == "unknown"

    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png at all")
    kind, why = bridge.classify(broken)
    assert kind == "unknown" and "dimensions" in why


# -------------------------------------------------------------- graph shapes


@pytest.mark.parametrize("kind", ["record3d", "panorama", "still"])
def test_every_graph_ends_in_a_server_side_render(tmp_path, kind):
    """Never AtlasBlockoutViewport.

    The viewport renders in the BROWSER via three.js, so a headless queue would
    save black frames — which looks like a broken solve and is not one. Only
    AtlasStereoRender rasterises server-side.
    """
    graph = bridge.GRAPH_BUILDERS[kind](tmp_path / "x.png")
    types = {n["class_type"] for n in graph.values()}
    assert "AtlasStereoRender" in types
    assert "AtlasBlockoutViewport" not in types
    assert "SaveImage" in types


@pytest.mark.parametrize("kind", ["record3d", "panorama", "still"])
def test_graphs_only_reference_registered_nodes(tmp_path, kind):
    """A renamed node must break here, not silently on the phone.

    Registered keys are a saved-workflow contract and rarely change, but this
    file builds graphs by hand with no workflow JSON to validate it — so nothing
    else would catch a typo until a capture arrived and failed.
    """
    from atlas_camera.comfy import node_registry as reg

    builtin = {"SaveImage", "LoadImage", "PreviewImage"}
    # Gated tiers count as registered here: a record3d graph legitimately uses
    # the ATLAS_IOS-gated capture nodes, which aren't in the default mapping.
    registered = {**reg.NODE_CLASS_MAPPINGS,
                  **reg.EXPERIMENTAL_NODE_CLASS_MAPPINGS,
                  **getattr(reg, "LEGACY_NODE_CLASS_MAPPINGS", {}),
                  **getattr(reg, "IOS_NODE_CLASS_MAPPINGS", {})}
    graph = bridge.GRAPH_BUILDERS[kind](tmp_path / "x.png")
    for node in graph.values():
        ct = node["class_type"]
        if ct in builtin:
            continue
        assert ct in registered, f"{ct} is not a registered node"


def test_still_graph_asks_for_geometry_by_a_valid_mesh_mode(tmp_path):
    """`mesh` is a COMBO of mode names, not a boolean.

    Passing `True` would be rejected by ComfyUI's validator at queue time — i.e.
    only when a real photo arrived from the phone. `layers=0` is deliberate: it
    is the single full-range relief mesh (the fast path), not "no geometry".
    """
    from atlas_camera.comfy import node_registry as reg

    graph = bridge.GRAPH_BUILDERS["still"](tmp_path / "x.png")
    inp = next(n for n in graph.values() if n["class_type"] == "AtlasInput")

    spec = reg.NODE_CLASS_MAPPINGS["AtlasInput"].INPUT_TYPES()
    modes = next(v[0] for sec in ("required", "optional")
                 for k, v in (spec.get(sec) or {}).items() if k == "mesh")
    assert inp["inputs"]["mesh"] in modes, f"mesh must be one of {modes}"
    assert isinstance(inp["inputs"]["layers"], int)


def test_panorama_graph_does_not_restate_measured_defaults(tmp_path):
    """n_views / pitch_deg / depth_model must come from the node.

    They are measured values (4 views, pitch 0, MoGe) and re-specifying them
    here lets this file drift from the node that owns them.
    """
    graph = bridge.GRAPH_BUILDERS["panorama"](tmp_path / "x.exr")
    mv = next(n for n in graph.values() if n["class_type"] == "AtlasEquirectMultiView")
    assert set(mv["inputs"]) == {"equirect"}


def test_graphs_are_json_serialisable(tmp_path):
    """They are POSTed to /prompt, so a stray Path would fail at the worst time."""
    for kind in bridge.GRAPH_BUILDERS:
        json.dumps(bridge.GRAPH_BUILDERS[kind](tmp_path / "x.png"))


# ------------------------------------------------------------ completeness


def test_partial_capture_is_refused(tmp_path):
    full = _r3d(tmp_path / "full.r3d")
    half = tmp_path / "half.r3d"
    half.write_bytes(full.read_bytes()[: len(full.read_bytes()) // 2])
    assert bridge.is_complete(half, settle=0.0) is False


def test_complete_capture_is_accepted(tmp_path):
    assert bridge.is_complete(_r3d(tmp_path / "ok.r3d"), settle=0.0) is True


def test_zip_without_capture_contents_is_refused(tmp_path):
    assert bridge.is_complete(_r3d(tmp_path / "bad.r3d", valid=False), settle=0.0) is False


def test_empty_file_is_refused(tmp_path):
    (tmp_path / "empty.r3d").write_bytes(b"")
    assert bridge.is_complete(tmp_path / "empty.r3d", settle=0.0) is False


# ----------------------------------------------------------------- transport


def test_accepted_extensions_cover_both_captures_and_images():
    assert ".r3d" in bridge.ACCEPTED_EXTS
    for ext in (".jpg", ".png", ".exr", ".heic"):
        assert ext in bridge.ACCEPTED_EXTS


def test_phone_is_matched_on_os_not_hostname(monkeypatch):
    """This phone reports its hostname as "localhost".

    Matching on a name would look perfectly sensible and then target nothing, so
    the return leg keys on OS and resolves the real DNS name.
    """
    payload = {"Peer": {"k": {"HostName": "localhost", "OS": "iOS", "Online": True,
                              "DNSName": "iphone-12-pro.example.ts.net.",
                              "TailscaleIPs": ["100.64.0.2"]}}}

    class _Out:
        stdout = json.dumps(payload)

    monkeypatch.setattr(bridge, "_ts", lambda *a, **k: _Out())
    assert bridge.find_phone(Path("tailscale")) == "iphone-12-pro.example.ts.net"


def test_offline_phone_is_not_selected(monkeypatch):
    payload = {"Peer": {"k": {"HostName": "localhost", "OS": "iOS", "Online": False,
                              "DNSName": "iphone.example.ts.net.",
                              "TailscaleIPs": ["100.64.0.2"]}}}

    class _Out:
        stdout = json.dumps(payload)

    monkeypatch.setattr(bridge, "_ts", lambda *a, **k: _Out())
    assert bridge.find_phone(Path("tailscale")) is None


def test_explicit_peer_overrides_detection(monkeypatch):
    monkeypatch.setattr(bridge, "_ts",
                        lambda *a, **k: pytest.fail("should not query tailscale"))
    assert bridge.find_phone(Path("tailscale"), "my-phone") == "my-phone"


def test_inbox_default_is_not_inside_a_synced_folder():
    """The point of Taildrop here is that nothing leaves the tailnet.

    Landing captures in a cloud-synced directory would undo that silently.
    """
    parts = {p.lower() for p in bridge.INBOX_DIR.parts}
    for provider in ("onedrive", "dropbox", "google drive", "my drive", "iclouddrive"):
        assert provider not in parts


# ------------------------------------------------------- install discovery
#
# The bridge used to hard-code C:\Users\<author>\ComfyUI_V91\..., which is why
# it could not be committed at all. These pin the replacement: it must find a
# real install with zero configuration, refuse to invent one, and never choose
# between two in silence.


def _fake_install(root: Path) -> Path:
    (root / "input").mkdir(parents=True)
    (root / "output").mkdir(parents=True)
    return root


class TestComfyUIDiscovery:
    def test_an_explicit_path_wins_over_everything(self, tmp_path, monkeypatch):
        chosen = _fake_install(tmp_path / "chosen")
        _fake_install(tmp_path / "ComfyUI")
        monkeypatch.setenv("COMFYUI_ROOT", str(tmp_path / "ComfyUI"))
        assert bridge.comfyui_root(str(chosen)) == chosen

    def test_the_env_var_is_used_when_no_path_is_given(self, tmp_path, monkeypatch):
        root = _fake_install(tmp_path / "somewhere")
        monkeypatch.setenv("COMFYUI_ROOT", str(root))
        assert bridge.comfyui_root(None) == root

    def test_a_versioned_install_is_found_by_glob(self, tmp_path):
        """The real-world case the old fixed-name list could not find.

        Installs carry a suffix — ComfyUI_V91, ComfyUI_windows_portable — and
        the portable builds nest the app one level down.
        """
        _fake_install(tmp_path / "ComfyUI_V91" / "ComfyUI")
        assert list(bridge._search_for_comfyui(home=tmp_path)) == \
            [tmp_path / "ComfyUI_V91" / "ComfyUI"]

    def test_a_directory_without_input_and_output_is_not_an_install(self, tmp_path):
        (tmp_path / "ComfyUI_notreally").mkdir()
        assert list(bridge._search_for_comfyui(home=tmp_path)) == []

    def test_discovery_order_is_stable(self, tmp_path):
        """Two machines with the same layout must resolve the same install."""
        for name in ("ComfyUI_b", "ComfyUI_a", "ComfyUI_c"):
            _fake_install(tmp_path / name)
        twice = [list(bridge._search_for_comfyui(home=tmp_path)) for _ in range(2)]
        assert twice[0] == twice[1] == sorted(twice[0])

    def test_two_installs_are_reported_not_silently_chosen(self, tmp_path, capsys,
                                                           monkeypatch):
        _fake_install(tmp_path / "ComfyUI_V91" / "ComfyUI")
        _fake_install(tmp_path / "ComfyUI_portable" / "ComfyUI")
        monkeypatch.delenv("COMFYUI_ROOT", raising=False)
        monkeypatch.setattr(bridge.Path, "home", staticmethod(lambda: tmp_path))
        bridge.comfyui_root(None)
        out = capsys.readouterr().out
        assert "2 ComfyUI installs found" in out and "--comfyui-root" in out, (
            "rendering into the wrong ComfyUI looks identical to rendering into "
            "the right one until you go looking for the output")


class TestResolveIODirs:
    def test_it_derives_both_directories_from_the_root(self, tmp_path, monkeypatch):
        root = _fake_install(tmp_path / "ComfyUI")
        monkeypatch.setenv("COMFYUI_ROOT", str(root))
        dirs, problem = bridge.resolve_io_dirs(None, None, None)
        assert problem is None
        assert dirs == (root / "input", root / "output")

    def test_an_explicit_override_beats_the_root(self, tmp_path, monkeypatch):
        root = _fake_install(tmp_path / "ComfyUI")
        other = tmp_path / "elsewhere"
        other.mkdir()
        monkeypatch.setenv("COMFYUI_ROOT", str(root))
        dirs, _ = bridge.resolve_io_dirs(str(other), None, None)
        assert dirs[0] == other and dirs[1] == root / "output"

    def test_it_says_what_to_do_rather_than_inventing_a_path(self, tmp_path,
                                                             monkeypatch):
        monkeypatch.delenv("COMFYUI_ROOT", raising=False)
        monkeypatch.setattr(bridge.Path, "home", staticmethod(lambda: tmp_path))
        dirs, problem = bridge.resolve_io_dirs(None, None, None)
        assert dirs is None
        assert "COMFYUI_ROOT" in problem and "--comfyui-root" in problem

    def test_a_nonexistent_override_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.delenv("COMFYUI_ROOT", raising=False)
        monkeypatch.setattr(bridge.Path, "home", staticmethod(lambda: tmp_path))
        dirs, problem = bridge.resolve_io_dirs(str(tmp_path / "nope"), None, None)
        assert dirs is None and "--input-dir" in problem
