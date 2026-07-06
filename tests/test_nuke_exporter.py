import ast

from atlas_camera.exporters.nuke_exporter import write_nuke_projection_script


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
    assert '"Card3D"' in script
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
    assert 'card["xsize"].setValue(40.0)' in script
    assert 'card["ysize"].setValue(40.0)' in script
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
    solve = make_atlas_solve()
    script = write_nuke_projection_script(solve, tmp_path / "nuke_cards.py").read_text(encoding="utf-8")
    assert "proj.setInput(0, read)" in script
    assert "proj.setInput(1, cam)" in script
    assert "proj.setInput(2, card)" in script
    assert "render.setInput(0, proj)" in script
