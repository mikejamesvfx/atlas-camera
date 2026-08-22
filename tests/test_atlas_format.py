"""The `.atlas` package a solve exports, and what it refuses to claim.

Atlas Scene opens what this writes. Most of these tests are about what does NOT
get emitted — an unearned licence, an invented confidence, a plane identity that
shifts when the fit is re-run — because a document that overstates what a solve
knows is worse than one that says less.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasProxyPrimitive,
    AtlasSolve,
    LatentCamera,
    ProjectionSource,
)
from atlas_camera.exporters.atlas_package import write_atlas_package
from atlas_camera.format import (
    SCHEMA_VERSION,
    FormatError,
    plane_id_for,
    scene_document,
    validate_document,
)
from atlas_camera.format.validate import COMPLETION_POLICIES, collect_problems


def solve_with(*primitives: AtlasProxyPrimitive, image_path: str | None = None) -> AtlasSolve:
    intrinsics = AtlasIntrinsics(
        image_width=1920, image_height=1080, focal_length_mm=35.0, sensor_width_mm=36.0
    )
    extrinsics = AtlasExtrinsics(
        camera_position=(0.0, 1.6, 0.0),
        camera_rotation_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )
    solve = AtlasSolve(
        camera=LatentCamera(intrinsics=intrinsics, extrinsics=extrinsics),
        image_path=image_path,
    )
    solve.projection_scene.proxy_geometry.extend(primitives)
    return solve


def wall(distance: float = 4.0, name: str = "projection_plane_01") -> AtlasProxyPrimitive:
    """A plane in the THREE.PlaneGeometry frame: local Z is the normal."""

    return AtlasProxyPrimitive(
        name=name,
        primitive_type="plane",
        transform_matrix=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, distance), (0, 0, 0, 1)),
        dimensions=(6.0, 3.0, 0.0),
        material="atlas_projection_proxy",
        metadata={"source": "ransac_plane_extraction", "inliers": 4210},
    )


# -- plane identity ----------------------------------------------------------


def test_the_same_plane_refitted_keeps_its_id():
    """A plane's NAME is its rank. An artist's decision must not be."""

    first = plane_id_for([0.0, 0.0, 1.0], 4.0, "observation_001")
    refitted = plane_id_for([0.00002, 0.0, 0.9999999], 4.0003, "observation_001")

    assert first == refitted


def test_a_normal_reported_the_other_way_round_is_the_same_plane():
    """{n, d} and {-n, -d} describe one surface; fitters disagree which to emit."""

    assert plane_id_for([0.0, 0.0, 1.0], 4.0, "o") == plane_id_for(
        [0.0, 0.0, -1.0], -4.0, "o"
    )
    assert plane_id_for([0.7071, 0.0, 0.7071], 3.0, "o") == plane_id_for(
        [-0.7071, 0.0, -0.7071], -3.0, "o"
    )


def test_a_different_surface_gets_a_different_id():
    assert plane_id_for([0.0, 0.0, 1.0], 4.0, "o") != plane_id_for(
        [0.0, 0.0, 1.0], 9.0, "o"
    )
    assert plane_id_for([0.0, 0.0, 1.0], 4.0, "o") != plane_id_for(
        [0.0, 1.0, 0.0], 4.0, "o"
    )


def test_the_same_wall_from_two_photographs_is_two_pieces_of_evidence():
    assert plane_id_for([0.0, 0.0, 1.0], 4.0, "observation_001") != plane_id_for(
        [0.0, 0.0, 1.0], 4.0, "observation_002"
    )


def test_a_zero_normal_describes_no_plane():
    with pytest.raises(ValueError):
        plane_id_for([0.0, 0.0, 0.0], 1.0)


def test_reordering_the_fit_does_not_reorder_identity():
    """The blocking failure: rank-named planes shift when inlier counts move."""

    document = scene_document(
        solve_with(wall(4.0, "projection_plane_01"), wall(9.0, "projection_plane_02")),
        scene_id="s",
        observation_id="o",
    )
    swapped = scene_document(
        solve_with(wall(9.0, "projection_plane_01"), wall(4.0, "projection_plane_02")),
        scene_id="s",
        observation_id="o",
    )

    assert [p["plane_id"] for p in document["planes"]] == [
        p["plane_id"] for p in reversed(swapped["planes"])
    ]
    # The label, meanwhile, stayed put while the SURFACE under it changed:
    # `projection_plane_01` names the 4 m wall in one run and the 9 m wall in
    # the other. That is exactly why nothing may key on it, and why an artist's
    # decision attached to a label is attached to a rank.
    assert document["planes"][0]["label"] == swapped["planes"][0]["label"]
    assert document["planes"][0]["plane_id"] != swapped["planes"][0]["plane_id"]


# -- what a plane licenses ---------------------------------------------------


def test_an_unclassified_plane_licenses_nothing():
    """`none` dominates. A tear nobody classified permits no construction."""

    document = scene_document(solve_with(wall()), scene_id="s")

    assert document["planes"][0]["completion_policy"] == "none"
    assert document["planes"][0]["confidence"] is None


def test_a_classified_plane_carries_the_graphs_verdict():
    solve = solve_with(wall())
    solve.semantics.value = {
        "occlusion_graph": {
            "nodes": [
                {
                    "id": "projection_plane_01",
                    "completion_policy": "extrude_profile",
                    "confidence": 0.78,
                    "source": "ransac_plane_extraction",
                }
            ]
        }
    }

    plane = scene_document(solve, scene_id="s")["planes"][0]

    assert plane["completion_policy"] == "extrude_profile"
    assert plane["confidence"] == pytest.approx(0.78)
    # How it was classified, so the judgement is reviewable rather than assumed.
    assert "classified by" in plane["method"]


def test_a_fitted_plane_is_inferred_not_measured():
    """Fitting a plane to a mesh is not measuring one off a photograph."""

    assert scene_document(solve_with(wall()), scene_id="s")["planes"][0][
        "provenance"
    ] == "INFERRED"


def test_the_policies_match_the_occlusion_graph():
    """The validator mirrors them so it stays import-light; drift is a bug."""

    from atlas_camera.core.occlusion_graph import COMPLETION_POLICIES as upstream

    assert set(COMPLETION_POLICIES) == set(upstream)


# -- what is never invented --------------------------------------------------


def test_a_solve_with_no_scale_verdict_says_unknown_rather_than_a_number():
    document = scene_document(solve_with(wall()), scene_id="s")

    scale = document["scale"]
    assert scale is None or scale["status"] in {"measured", "manual", "assumed", "unknown"}


def test_the_producer_never_sets_the_artists_scale():
    """What a scene unit is worth is established by measuring, in the editor."""

    assert scene_document(solve_with(wall()), scene_id="s")["scene_scale"] is None


def test_a_lens_with_no_recorded_distortion_says_null_not_zero():
    """A zeroed distortion block claims the lens is rectilinear."""

    assert scene_document(solve_with(wall()), scene_id="s")["camera"]["distortion"] is None


def test_pixel_intrinsics_are_derived_through_upstreams_own_helper():
    from atlas_camera.core.intrinsics import focal_length_mm_to_pixels

    camera = scene_document(solve_with(wall()), scene_id="s")["camera"]

    assert camera["fx"] == pytest.approx(focal_length_mm_to_pixels(35.0, 36.0, 1920))
    assert camera["cx"] == pytest.approx(960.0)


# -- validation --------------------------------------------------------------


def test_a_written_document_validates():
    validate_document(scene_document(solve_with(wall()), scene_id="s"))


def test_an_unknown_completion_policy_is_refused_not_downgraded():
    """Downgrading it to `none` looks identical to a correct classification."""

    document = scene_document(solve_with(wall()), scene_id="s")
    document["planes"][0]["completion_policy"] = "extrude_everything"

    with pytest.raises(FormatError, match="cannot evaluate"):
        validate_document(document)


def test_a_rotation_that_is_not_a_rotation_is_refused():
    document = scene_document(solve_with(wall()), scene_id="s")
    document["camera"]["rotation"] = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]

    with pytest.raises(FormatError, match="orthonormal"):
        validate_document(document)


def test_a_mirrored_frame_is_refused():
    """Determinant -1 is orthonormal and still mirrors the scene."""

    document = scene_document(solve_with(wall()), scene_id="s")
    document["camera"]["rotation"] = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]

    with pytest.raises(FormatError, match="orthonormal"):
        validate_document(document)


def test_an_absolute_asset_path_is_refused():
    """A package that names a file outside itself cannot be moved."""

    document = scene_document(solve_with(wall()), scene_id="s")
    document["camera"]["plate_path"] = "C:/somewhere/else/plate.exr"

    problems = collect_problems(document)
    assert any("absolute" in problem for problem in problems)


def test_a_matte_declaring_associated_alpha_is_refused():
    """Premultiplying a matte multiplies a coverage signal by itself."""

    document = scene_document(solve_with(wall()), scene_id="s")
    plane_id = document["planes"][0]["plane_id"]
    document["layers"] = [
        {"layer_id": "l", "plane_id": plane_id, "alpha_mode": "associated"}
    ]

    problems = collect_problems(document)
    assert any("straight" in problem for problem in problems)


def test_a_layer_naming_a_plane_that_is_not_there_is_refused():
    document = scene_document(solve_with(wall()), scene_id="s")
    document["layers"] = [{"layer_id": "l", "plane_id": "plane_ghost"}]

    problems = collect_problems(document)
    assert any("not in planes" in problem for problem in problems)


def test_a_confidence_outside_zero_to_one_is_refused():
    document = scene_document(solve_with(wall()), scene_id="s")
    document["planes"][0]["confidence"] = 1.4

    problems = collect_problems(document)
    assert any("outside [0, 1]" in problem for problem in problems)


def test_a_revision_deriving_from_nothing_that_exists_is_refused():
    document = scene_document(solve_with(wall()), scene_id="s")
    document["revisions"] = {
        "revisions": [
            {"revision_id": "r2", "digest": "a" * 64, "derived_from": "r1"}
        ],
        "selected": {},
    }

    problems = collect_problems(document)
    assert any("absent" in problem for problem in problems)


def test_an_unreadable_version_is_refused():
    document = scene_document(solve_with(wall()), scene_id="s")
    document["schema_version"] = "99.0"

    with pytest.raises(FormatError, match="schema_version"):
        validate_document(document)


# -- the package on disk -----------------------------------------------------


def test_a_package_is_self_contained(tmp_path):
    plate = tmp_path / "source.png"
    plate.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    mesh = tmp_path / "relief.glb"
    mesh.write_bytes(b"glTF")

    result = write_atlas_package(
        solve_with(wall(), image_path=str(plate)),
        tmp_path / "street.atlas",
        scene_id="street_001",
        geometry_path=str(mesh),
    )

    document = json.loads(result.document.read_text(encoding="utf-8"))
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["camera"]["plate_path"] == "imagery/source.png"
    assert (result.package_dir / "imagery" / "source.png").is_file()
    assert (result.package_dir / "geometry" / "relief.glb").is_file()
    assert result.complaints == []


def test_a_plate_that_has_expired_is_complained_about(tmp_path):
    """A solve's image_path is routinely a temp file that has since gone."""

    result = write_atlas_package(
        solve_with(wall(), image_path=str(tmp_path / "gone.png")),
        tmp_path / "street.atlas",
    )

    assert any("was named but is not" in complaint for complaint in result.complaints)


def test_a_plate_that_was_never_given_is_complained_about_too(tmp_path):
    """Absent and expired are both silent to the artist unless both are said.

    The complaint used to fire only when a path was NAMED and missing; supplying
    nothing returned early with no note at all. So a package could be written
    with no plate, no geometry and nothing on the node to explain why the scene
    it opens is inert — the exact failure the node's own comment forbids.
    Found live: a real export produced 6 planes, no imagery, no geometry, and a
    clean node.
    """

    result = write_atlas_package(
        solve_with(wall(), image_path=None), tmp_path / "street.atlas"
    )

    assert any("no plate" in complaint for complaint in result.complaints)
    assert any("no geometry" in complaint for complaint in result.complaints)


def test_the_solve_is_written_even_when_no_solve_path_is_given(tmp_path):
    """`atlas/` is advertised as the solve this package was produced from.

    It was filled only when the caller handed over a PATH, and the ComfyUI node
    has no such input — so every real package shipped that lane empty while the
    producer held the solve in memory the whole time. A package that cannot show
    the solve it came from cannot be audited, which is most of the point.
    """

    result = write_atlas_package(
        solve_with(wall()), tmp_path / "street.atlas", scene_id="street_001"
    )

    written = result.package_dir / "atlas" / "atlas_solve.json"
    assert written.is_file(), "the solve is in hand; the lane must not ship empty"

    document = json.loads(result.document.read_text(encoding="utf-8"))
    assert document["source"]["solve"] == "atlas/atlas_solve.json"


def test_a_supplied_solve_path_still_wins(tmp_path):
    """An explicit path is adopted as-is — the producer must not overwrite the
    file the caller nominated with its own re-serialisation."""

    nominated = tmp_path / "from_disk.json"
    nominated.write_text(solve_with(wall()).to_json(indent=2), encoding="utf-8")

    result = write_atlas_package(
        solve_with(wall()), tmp_path / "street.atlas", solve_path=str(nominated)
    )

    document = json.loads(result.document.read_text(encoding="utf-8"))
    assert document["source"]["solve"] == "atlas/from_disk.json"
    assert not (result.package_dir / "atlas" / "atlas_solve.json").exists()


def test_the_relief_mesh_is_observed_not_created(tmp_path):
    mesh = tmp_path / "relief.glb"
    mesh.write_bytes(b"glTF")

    result = write_atlas_package(
        solve_with(wall()), tmp_path / "s.atlas", geometry_path=str(mesh)
    )

    document = json.loads(result.document.read_text(encoding="utf-8"))
    entity = document["entities"][0]
    assert entity["observation_state"] == "OBSERVED"
    assert entity["confidence"] is None


def test_a_matte_becomes_a_file_with_its_digest(tmp_path):
    """Eleven 8K mattes inline as base64 is a document nothing opens twice."""

    payload = b"\x89PNG\r\n\x1a\n" + b"matte bytes"
    solve = solve_with(wall())
    solve.projection_sources.append(
        ProjectionSource(
            camera=solve.camera,
            name="wall",
            mask_b64="data:image/png;base64," + base64.b64encode(payload).decode(),
        )
    )

    result = write_atlas_package(solve, tmp_path / "s.atlas")
    document = json.loads(result.document.read_text(encoding="utf-8"))
    layer = document["layers"][0]

    written = (result.package_dir / layer["matte_path"]).read_bytes()
    # The producer's own bytes, not a re-encode: no image library in between to
    # turn a truncation into a round.
    assert written == payload
    assert layer["alpha_mode"] == "straight"
    assert len(layer["matte_digest"]) == 64


def test_the_package_records_that_it_was_produced(tmp_path):
    """History is append-only, and the first entry is the import itself."""

    result = write_atlas_package(solve_with(wall()), tmp_path / "s.atlas")

    ledger = (result.package_dir / "history" / "ledger.jsonl").read_text(encoding="utf-8")
    entry = json.loads(ledger.strip().splitlines()[0])
    assert entry["operation"] == "produce_package"
    assert entry["actor"] == "atlas-camera"


def test_a_document_that_would_not_validate_is_never_written(tmp_path):
    """By the time the editor refuses it, the artist already has it."""

    solve = solve_with(wall())
    solve.camera.extrinsics.camera_rotation_matrix = ((2, 0, 0), (0, 1, 0), (0, 0, 1))

    with pytest.raises(FormatError):
        write_atlas_package(solve, tmp_path / "s.atlas")

    assert not (tmp_path / "s.atlas" / "scene.json").exists()


# -- the node ----------------------------------------------------------------


def test_the_export_node_writes_a_package_an_editor_can_open(tmp_path):
    """The node is the only way an artist reaches any of this."""

    from atlas_camera.comfy.nodes_export import AtlasExportScenePackage

    plate = tmp_path / "source.png"
    plate.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    mesh = tmp_path / "relief.glb"
    mesh.write_bytes(b"glTF")

    outcome = AtlasExportScenePackage().export(
        solve_with(wall(), image_path=str(plate)),
        str(tmp_path),
        "street_001",
        relief_mesh_path=str(mesh),
        observation_id="observation_001",
    )

    package = Path(outcome["result"][0] if isinstance(outcome, dict) else outcome[0])
    document = json.loads((package / "scene.json").read_text(encoding="utf-8"))
    assert package.name == "street_001.atlas"
    assert document["camera"]["plate_path"] == "imagery/source.png"
    assert document["entities"][0]["geometry"]["path"] == "geometry/relief.glb"
    validate_document(document)


def test_the_node_says_what_it_could_not_do(tmp_path):
    """A missing plate on the node, where the artist is looking."""

    from atlas_camera.comfy.nodes_export import AtlasExportScenePackage

    outcome = AtlasExportScenePackage().export(
        solve_with(wall(), image_path=str(tmp_path / "gone.png")),
        str(tmp_path),
        "street_001",
    )

    assert isinstance(outcome, dict), "a complaint must reach the node's UI"
    assert "was named but is not" in outcome["ui"]["text"][0]
