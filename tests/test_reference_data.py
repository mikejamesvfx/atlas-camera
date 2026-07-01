import json
import subprocess
import sys

from atlas_camera.reference_data import (
    get_scale_reference,
    list_categories,
    search_scale_references,
)


def test_reference_registry_loads_and_searches_common_items():
    person = get_scale_reference("person_175cm")
    eiffel = search_scale_references("eiffel")

    assert person.height == 1.75
    assert person.units == "m"
    assert "human" in list_categories()
    assert {reference.id for reference in eiffel} >= {
        "eiffel_tower_tip_330m",
        "eiffel_tower_architectural_300m",
    }


def test_list_references_cli_outputs_json():
    completed = subprocess.run(
        [
            sys.executable,
            "tools/list_references.py",
            "--query",
            "container",
            "--json",
        ],
        check=True,
        cwd=".",
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert any(item["id"] == "shipping_container_20ft" for item in payload)
    assert all("height" in item for item in payload)

