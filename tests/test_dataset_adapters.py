import json
import subprocess
import sys

from atlas_camera.datasets.benchmark import BenchmarkOptions, benchmark_eth3d
from atlas_camera.datasets.colmap import read_colmap_cameras, read_colmap_images
from atlas_camera.datasets.dtu import load_dtu_projections, parse_projection_matrix
from atlas_camera.datasets.eth3d import load_eth3d_dataset


def _write_eth3d_fixture(root):
    (root / "cameras.txt").write_text(
        "\n".join(
            [
                "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]",
                "1 PINHOLE 160 120 100.0 101.0 80.0 60.0",
            ]
        ),
        encoding="utf-8",
    )
    (root / "images.txt").write_text(
        "\n".join(
            [
                "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME",
                "7 1 0 0 0 1 2 3 1 images/frame_0001.png",
                "",
                "8 1 0 0 0 0 0 0 1 images/frame_0002.png",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_colmap_camera_and_image_parsing(tmp_path):
    _write_eth3d_fixture(tmp_path)

    cameras = read_colmap_cameras(tmp_path / "cameras.txt")
    images = read_colmap_images(tmp_path / "images.txt")

    assert cameras[1].model == "PINHOLE"
    assert cameras[1].fx_px == 100.0
    assert cameras[1].fy_px == 101.0
    assert cameras[1].cx_px == 80.0
    assert cameras[1].cy_px == 60.0
    assert images[7].camera_id == 1
    assert images[7].camera_center_world == (-1.0, -2.0, -3.0)
    assert images[8].name == "images/frame_0002.png"


def test_eth3d_loader_discovers_colmap_files(tmp_path):
    scene = tmp_path / "scene"
    scene.mkdir()
    _write_eth3d_fixture(scene)

    dataset = load_eth3d_dataset(tmp_path)

    assert dataset.cameras_path == scene / "cameras.txt"
    assert dataset.images_path == scene / "images.txt"
    assert dataset.iter_images()[0].name == "images/frame_0001.png"


def test_dtu_projection_parser_reads_tiny_matrix_fixture(tmp_path):
    projection_path = tmp_path / "pos_001.txt"
    projection_path.write_text(
        """
        1 0 0 10
        0 1 0 20
        0 0 1 30
        """,
        encoding="utf-8",
    )

    assert parse_projection_matrix(projection_path) == (
        (1.0, 0.0, 0.0, 10.0),
        (0.0, 1.0, 0.0, 20.0),
        (0.0, 0.0, 1.0, 30.0),
    )
    assert len(load_dtu_projections(tmp_path)) == 1


def test_eth3d_benchmark_metadata_only_smoke(tmp_path):
    _write_eth3d_fixture(tmp_path)

    records = benchmark_eth3d(tmp_path, BenchmarkOptions(limit=1))

    assert len(records) == 1
    record = records[0]
    assert record.status == "ok"
    assert record.source_method == "automatic_still_image_metadata_only"
    assert record.fx_abs_error_px == 0.0
    assert record.fy_abs_error_px == 0.0
    assert record.principal_point_error_px == 0.0
    assert record.orientation_error_deg == 0.0


def test_benchmark_cli_writes_json_and_csv(tmp_path):
    _write_eth3d_fixture(tmp_path)
    output_json = tmp_path / "report.json"
    output_csv = tmp_path / "report.csv"

    completed = subprocess.run(
        [
            sys.executable,
            "tools/benchmark_datasets.py",
            "--dataset",
            "eth3d",
            "--root",
            str(tmp_path),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--limit",
            "1",
        ],
        check=True,
        cwd=".",
        capture_output=True,
        text=True,
    )

    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert "records: 1" in completed.stdout
    assert data[0]["gt_fx_px"] == 100.0
    assert output_csv.read_text(encoding="utf-8").startswith("dataset,image_name")
