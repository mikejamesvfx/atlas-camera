import ast

from atlas_camera.exporters.nuke_exporter import write_nuke_native_script, write_nuke_projection_script


def test_nuke_exporter_script_is_valid_python(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    script = write_nuke_projection_script(solve, tmp_path / "nuke_cards.py").read_text(encoding="utf-8")
    ast.parse(script)


def test_nuke_exporter_writes_script_file(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    path = write_nuke_projection_script(solve, tmp_path / "nuke_cards.py")
    assert path.is_file()
    script = path.read_text(encoding="utf-8")
    assert "import nuke" in script
    assert "build_projection" in script


def test_nuke_exporter_is_a_real_implementation(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    script = write_nuke_projection_script(solve, tmp_path / "nuke_cards.py").read_text(encoding="utf-8")
    assert "NotImplementedError" not in script
    assert "nuke.createNode" in script


def test_nuke_exporter_creates_required_nodes(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    script = write_nuke_projection_script(solve, tmp_path / "nuke_cards.py").read_text(encoding="utf-8")
    assert '"Read"' in script
    assert '"Camera2"' in script
    # Card, not Card3D: confirmed live in Nuke 16.1v3 that Card3D has no
    # xsize/ysize knobs at all (it's a lens/format-driven camera billboard,
    # not a manually-sized plane) - plain Card's default 1x1 unit quad, sized
    # via `scaling`, is the node that actually has the geometry we want.
    assert '"Card"' in script
    assert '"Card3D"' not in script
    assert '"Project3D2"' in script
    assert '"ScanlineRender"' in script


def test_nuke_exporter_sets_camera_film_back(tmp_path, make_atlas_solve):
    solve = make_atlas_solve(focal=50.0, sensor_w=36.0)
    script = write_nuke_projection_script(solve, tmp_path / "nuke_cards.py").read_text(encoding="utf-8")
    assert 'cam["focal"].setValue(50.0)' in script
    assert 'cam["haperture"].setValue(36.0)' in script


def test_nuke_exporter_sets_ground_plane_size(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    script = write_nuke_projection_script(solve, tmp_path / "nuke_cards.py").read_text(encoding="utf-8")
    # Card has no xsize/ysize knob (confirmed live) - it's a 1x1 unit quad
    # sized via the universal `scaling` transform knob instead.
    assert 'geo["scaling"].setValue([40.0, 40.0, 1.0])' in script
    assert '[-90.0' in script  # flat XZ ground plane rotation


def test_nuke_exporter_embeds_world_matrix(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    script = write_nuke_projection_script(solve, tmp_path / "nuke_cards.py").read_text(encoding="utf-8")
    assert "useMatrix" in script
    assert "matrix" in script
    # 16 floats from the identity world matrix
    assert "1.0" in script


def test_nuke_exporter_escapes_source_image_name_with_quote_character(tmp_path, make_atlas_solve):
    # Regression test: source_image_name is interpolated into the generated
    # script's source_path expression. It must use !r (repr) escaping like
    # every other interpolated value in this f-string — without it, a
    # filename containing a double-quote breaks out of the string literal
    # and injects arbitrary Python that Nuke would execute when the artist
    # opens the review scene.
    solve = make_atlas_solve()
    malicious_name = 'evil".os.system("calc")#.png'

    script = write_nuke_projection_script(
        solve, tmp_path / "nuke_cards.py", source_image_name=malicious_name,
    ).read_text(encoding="utf-8")

    ast.parse(script)  # must still be syntactically valid
    assert repr(malicious_name) in script
    without_safe_repr = script.replace(repr(malicious_name), "")
    assert "os.system(" not in without_safe_repr


def test_nuke_exporter_sets_camera_translate(tmp_path, make_atlas_solve):
    solve = make_atlas_solve(position=(1.0, 2.0, 3.0))
    script = write_nuke_projection_script(solve, tmp_path / "nuke_cards.py").read_text(encoding="utf-8")
    assert "1.0" in script
    assert "2.0" in script
    assert "3.0" in script


def test_nuke_exporter_wires_projection_inputs(tmp_path, make_atlas_solve):
    # This exact wiring was confirmed by actually rendering the graph in
    # Nuke 16.1v3 (nuke.execute() through a real Write node, producing real
    # pixels) - see nuke_exporter.py's module docstring. Project3D2 only has
    # two real inputs (img, cam); its output feeds Card's own image input,
    # not the other way around. ScanlineRender's real slots are bg=0
    # (unconnected), obj=1, cam=2 - not obj=0/cam=1.
    solve = make_atlas_solve()
    script = write_nuke_projection_script(solve, tmp_path / "nuke_cards.py").read_text(encoding="utf-8")
    assert "proj.setInput(0, read)" in script
    assert "proj.setInput(1, cam)" in script
    assert "geo.setInput(0, proj)" in script
    assert "render.setInput(1, geo)" in script
    assert "render.setInput(2, cam)" in script


def test_nuke_exporter_normalises_windows_paths_for_tcl_safety(tmp_path, make_atlas_solve):
    # Nuke's live knob-setting API runs string values through TCL escape
    # interpretation, which silently eats backslash-letter sequences in a
    # Windows path (confirmed by actually running the generated script in
    # Nuke - a raw "C:\Users\..." path arrived at the Read node with every
    # backslash stripped). Forward slashes sidestep this entirely.
    solve = make_atlas_solve()
    script = write_nuke_projection_script(solve, tmp_path / "nuke_cards.py").read_text(encoding="utf-8")
    assert "source_path.replace(" in script


def test_nuke_native_script_is_written(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    path = write_nuke_native_script(solve, tmp_path / "nuke_cards.nk")
    assert path.is_file()
    assert path.suffix == ".nk"


def test_nuke_native_script_creates_required_nodes(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    script = write_nuke_native_script(solve, tmp_path / "nuke_cards.nk").read_text(encoding="utf-8")
    assert "Root {" in script
    assert "Read {" in script
    assert "Camera2 {" in script
    assert "Card {" in script
    assert "Project3D2 {" in script
    assert "ScanlineRender {" in script
    assert "Card3D" not in script


def test_nuke_native_script_completes_camera_link_via_onscriptload(tmp_path, make_atlas_solve):
    # The one link .nk's push/pop stack model can't reliably re-resolve as
    # text (reusing the same Camera2 a second time, after Project3D2 already
    # consumed it once, as a push target for ScanlineRender's cam slot -
    # confirmed by testing every push-order/pairing permutation in real
    # Nuke) is completed by a one-line Python callback on Root's
    # onScriptLoad knob instead, which Nuke runs automatically on open.
    solve = make_atlas_solve()
    script = write_nuke_native_script(solve, tmp_path / "nuke_cards.nk").read_text(encoding="utf-8")
    assert "onScriptLoad" in script
    assert "setInput(2, nuke.toNode('Camera1'))" in script


def test_nuke_native_script_sets_ground_plane_scaling(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    script = write_nuke_native_script(solve, tmp_path / "nuke_cards.nk").read_text(encoding="utf-8")
    assert "scaling {40 40 1}" in script
    assert "rotate {-90 0 0}" in script


def test_nuke_native_script_sets_root_format(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    script = write_nuke_native_script(solve, tmp_path / "nuke_cards.nk").read_text(encoding="utf-8")
    assert 'format "' in script


def test_nuke_exporter_imports_relief_mesh_when_given(tmp_path, make_atlas_solve):
    # ReadGeo2 imports the real derived relief mesh instead of a flat Card
    # when relief_mesh_obj_path is supplied - still live-projected onto via
    # Project3D2 (not a static UV-baked texture), confirmed by actually
    # rendering this exact topology in Nuke 16.1v3.
    solve = make_atlas_solve()
    mesh_path = tmp_path / "atlas_relief_mesh.obj"
    mesh_path.write_text("# stub obj\n", encoding="utf-8")
    script = write_nuke_projection_script(
        solve, tmp_path / "nuke_cards.py", relief_mesh_obj_path=mesh_path,
    ).read_text(encoding="utf-8")
    assert '"ReadGeo2"' in script
    assert '"Card"' not in script
    assert "geo.setInput(0, proj)" in script
    assert "render.setInput(1, geo)" in script
    assert str(mesh_path).replace("\\", "/") in script


def test_nuke_exporter_defaults_to_flat_card_without_relief_mesh(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    script = write_nuke_projection_script(solve, tmp_path / "nuke_cards.py").read_text(encoding="utf-8")
    assert '"Card"' in script
    assert '"ReadGeo2"' not in script


def test_nuke_native_script_imports_relief_mesh_when_given(tmp_path, make_atlas_solve):
    solve = make_atlas_solve()
    mesh_path = tmp_path / "atlas_relief_mesh.obj"
    mesh_path.write_text("# stub obj\n", encoding="utf-8")
    script = write_nuke_native_script(
        solve, tmp_path / "nuke_cards.nk", relief_mesh_obj_path=mesh_path,
    ).read_text(encoding="utf-8")
    assert "ReadGeo2 {" in script
    assert "Card {" not in script
    assert str(mesh_path).replace("\\", "/") in script
    # The camera-fixup callback still applies regardless of geometry type.
    assert "onScriptLoad" in script


def test_dangling_temp_plate_is_not_baked_into_the_script(tmp_path, make_atlas_solve):
    """Every tensor-based solve records a NamedTemporaryFile as image_path and
    then unlinks it, so by export time the path is dangling. Baking it into the
    Read node produced a .nk/.py that could not resolve its plate (found by the
    Linux beta test on the shipped quickstart)."""
    solve = make_atlas_solve()
    missing = tmp_path / "deleted_temp_plate.png"
    solve.image_path = str(missing)
    assert not missing.exists()

    for writer, name in ((write_nuke_projection_script, "nuke_cards.py"),
                         (write_nuke_native_script, "nuke_cards.nk")):
        script = writer(solve, tmp_path / name).read_text(encoding="utf-8")
        assert "deleted_temp_plate.png" not in script


def test_existing_plate_is_still_baked_into_the_script(tmp_path, make_atlas_solve):
    """The existence check must not cost a real plate its path."""
    solve = make_atlas_solve()
    real = tmp_path / "real_plate.png"
    real.write_bytes(b"\x89PNG\r\n\x1a\n")
    solve.image_path = str(real)

    script = write_nuke_projection_script(solve, tmp_path / "nuke_cards.py").read_text(encoding="utf-8")
    assert "real_plate.png" in script
