"""Automate an Atlas Camera evaluation dataset through a running local ComfyUI instance.

Generates N Qwen-image photographs across Exterior/Interior/Nature categories
(each prompt written with deliberate, unambiguous scale cues: people, cars,
5-story buildings, standard doors, etc.), and for every photo runs all 5
documented projection-derivation variants — the "organic"/"indoor"/"outdoor"
scene_type presets, a manual primitive_method=azimuth_walls run, and a
geometry_mode="both" comparison run — exporting a relief mesh (OBJ+GLB) and a
real (non-placeholder) Maya review scene per variant, plus the solved
atlas_solve.json for manifest bookkeeping.

This talks directly to a local ComfyUI HTTP API (default
http://127.0.0.1:8188), not any cloud service: the workflow depends on
custom AtlasCamera nodes and local model files (Qwen checkpoint, LoRA,
GeoCalib, Depth Anything V2) that only exist in that local install.

Usage:
    python tools/generate_qwen_dataset.py
    python tools/generate_qwen_dataset.py --limit 3          # smoke-test a few images first
    python tools/generate_qwen_dataset.py --output-root D:/atlas_eval/qwen_dataset_01
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.request
import uuid

TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "atlas_camera_core_projection_qwen_dataset_example_01.json"
)
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "atlas_exports" / "qwen_dataset_01"

OUTDOOR_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
INDOOR_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

CATEGORY_DEPTH_MODEL = {
    "exterior": OUTDOOR_DEPTH_MODEL,
    "interior": INDOOR_DEPTH_MODEL,
    "nature": OUTDOOR_DEPTH_MODEL,
}

# Fixed per-image seed so re-runs are reproducible and every output is
# traceable to its exact prompt+seed. The template's KSampler defaults to
# "randomize" — deliberately overridden in build_graph() below.
BASE_SEED = 667_099_001_500_000

# Every prompt names at least one recognizable, describable-distance scale
# reference (an adult person, a car/bus, a 5-story building with countable
# floors, a standard door, etc.) so a human reviewer can judge the recovered
# camera/geometry scale at a glance without any measurement tools.
SCENES: list[dict[str, str]] = [
    # ---------------------------------------------------------------- Exterior
    {
        "category": "exterior",
        "id": "exterior_01_apartment_sedan",
        "prompt": "photorealistic street-level photograph of a 5-story brick apartment "
                   "building with five clearly countable rows of windows, a red sedan "
                   "parked directly at the curb in front of it, one adult pedestrian "
                   "walking past the car for scale, midday overcast light, straight-on "
                   "eye-level camera height around 1.6m.",
    },
    {
        "category": "exterior",
        "id": "exterior_02_bus_stop",
        "prompt": "photorealistic photograph of a city bus stop, a standard city transit "
                   "bus about 12 meters long parked at the curb, two adults waiting "
                   "at the stop beside it, a 4-story office building behind them, "
                   "late-afternoon sun casting long shadows.",
    },
    {
        "category": "exterior",
        "id": "exterior_03_highrise_construction",
        "prompt": "photorealistic photograph of a 5-story building under construction, "
                   "steel scaffolding visible at every floor level, a yellow shipping-"
                   "container-sized site office trailer beside it, one construction "
                   "worker in a hi-vis vest and hard hat standing near the trailer for "
                   "scale, overcast daylight.",
    },
    {
        "category": "exterior",
        "id": "exterior_04_suburban_house",
        "prompt": "photorealistic photograph of a two-story suburban house with a "
                   "standard front door and a single-car attached garage, a compact SUV "
                   "parked in the driveway, one adult retrieving mail from a mailbox at "
                   "the curb for scale, bright sunny afternoon.",
    },
    {
        "category": "exterior",
        "id": "exterior_05_rainy_night_crosswalk",
        "prompt": "photorealistic rainy night street-crossing photograph, reflective wet "
                   "asphalt, an adult pedestrian using a crosswalk beside a parked sedan, "
                   "a traffic light and street lamps for scale, neon storefront signs in "
                   "the background, cinematic dusk lighting.",
    },
    {
        "category": "exterior",
        "id": "exterior_06_double_decker",
        "prompt": "photorealistic photograph of a red double-decker tour bus, about 4.4 "
                   "meters tall, stopped at a curb, one adult standing beside its front "
                   "wheel for scale, a 5-story brick building facade in the background, "
                   "morning golden-hour light.",
    },
    {
        "category": "exterior",
        "id": "exterior_07_gas_station",
        "prompt": "photorealistic photograph of a gas station forecourt, a standard sedan "
                   "parked at a fuel pump, one adult filling the tank for scale, a "
                   "5-story hotel building visible behind the station, dusk lighting with "
                   "the station canopy lights switched on.",
    },
    {
        "category": "exterior",
        "id": "exterior_08_school",
        "prompt": "photorealistic photograph of the front entrance of a 3-story brick "
                   "schoolhouse, a standard flagpole about 6 meters tall beside the main "
                   "door, a yellow school bus parked out front, two children walking "
                   "toward the entrance for scale, midday sun.",
    },
    {
        "category": "exterior",
        "id": "exterior_09_parking_garage",
        "prompt": "photorealistic photograph of the entrance ramp to a multi-level "
                   "parking garage, a height-clearance sign reading 2.1m mounted above "
                   "the entrance, a mid-size sedan approaching the entrance, one adult "
                   "attendant standing beside the booth for scale, overcast afternoon.",
    },
    {
        "category": "exterior",
        "id": "exterior_10_church",
        "prompt": "photorealistic photograph of a stone church with a bell tower roughly "
                   "five stories tall, a standard arched wooden double door at the "
                   "entrance, one adult in a coat walking up the front steps, a parked "
                   "sedan at the curb, late-afternoon warm light.",
    },
    {
        "category": "exterior",
        "id": "exterior_11_alley_dumpster",
        "prompt": "photorealistic photograph of a narrow city alley between two 5-story "
                   "brick buildings, a standard commercial dumpster about 1.8 meters "
                   "long against one wall, one adult walking through the alley for scale, "
                   "overcast daylight, fire escapes visible on the walls.",
    },
    {
        "category": "exterior",
        "id": "exterior_12_bridge",
        "prompt": "photorealistic photograph of a pedestrian crossing a steel truss "
                   "bridge over a river, a standard guardrail about 1.1 meters tall along "
                   "the walkway, a car crossing the bridge roadway lane beside the "
                   "pedestrian path, midday clear sky.",
    },
    {
        "category": "exterior",
        "id": "exterior_13_storefront_row",
        "prompt": "photorealistic photograph of a row of ground-floor storefronts below "
                   "5 stories of apartments, awnings at a standard height above the "
                   "sidewalk doors, one adult exiting a shop doorway, a delivery van "
                   "double-parked at the curb, bright afternoon sun.",
    },
    {
        "category": "exterior",
        "id": "exterior_14_town_square",
        "prompt": "photorealistic photograph of a town square with a stone clock tower "
                   "approximately five stories tall at its center, an ornate lamp post "
                   "beside it, several adults walking across the square, a parked sedan "
                   "at the square's edge, early-evening golden-hour light.",
    },
    {
        "category": "exterior",
        "id": "exterior_15_overpass",
        "prompt": "photorealistic photograph of a highway overpass with a standard "
                   "passenger car driving beneath it, a pedestrian walking on the "
                   "sidewalk beside the on-ramp, a highway sign mounted on the overpass, "
                   "overcast midday light.",
    },
    # ---------------------------------------------------------------- Interior
    {
        "category": "interior",
        "id": "interior_01_living_room",
        "prompt": "photorealistic photograph of the interior of a furnished living room, "
                   "a standard 2032mm (6'8\") interior door open in the background, an "
                   "adult seated on a three-seat sofa in the foreground for scale, a "
                   "coffee table in front of the sofa, warm afternoon window light.",
    },
    {
        "category": "interior",
        "id": "interior_02_kitchen",
        "prompt": "photorealistic photograph of a modern kitchen with standard-height "
                   "countertops about 90cm tall, a full-size refrigerator beside the "
                   "counter, an adult standing at the stove for scale, overhead pendant "
                   "lighting, straight-on eye-level camera angle.",
    },
    {
        "category": "interior",
        "id": "interior_03_hotel_room",
        "prompt": "photorealistic photograph of a hotel room with a standard queen-size "
                   "bed 152cm wide, a full-length door-mounted mirror on the closet "
                   "door, one adult standing beside the bed for scale, warm lamp "
                   "lighting, evening ambience.",
    },
    {
        "category": "interior",
        "id": "interior_04_hallway",
        "prompt": "photorealistic photograph of a long apartment-building hallway lined "
                   "with standard 2032mm interior doors at regular intervals, one adult "
                   "walking down the hallway toward the camera for scale, fluorescent "
                   "ceiling lights, straight-on perspective view.",
    },
    {
        "category": "interior",
        "id": "interior_05_staircase",
        "prompt": "photorealistic photograph of a straight residential staircase with "
                   "standard 7-inch risers and a handrail at standard height, one adult "
                   "walking up the stairs for scale, daylight coming through a window "
                   "at the landing.",
    },
    {
        "category": "interior",
        "id": "interior_06_bathroom",
        "prompt": "photorealistic photograph of a standard bathroom with a full-size "
                   "bathtub about 1.5 meters long and a pedestal sink, one adult "
                   "standing beside the sink brushing their teeth for scale, tile "
                   "flooring, bright overhead lighting.",
    },
    {
        "category": "interior",
        "id": "interior_07_bedroom",
        "prompt": "photorealistic photograph of a bedroom with a standard full-size "
                   "2032mm door and a queen bed, one adult sitting on the edge of the "
                   "bed for scale, a dresser with a mirror on the far wall, soft morning "
                   "window light.",
    },
    {
        "category": "interior",
        "id": "interior_08_restaurant",
        "prompt": "photorealistic photograph of a restaurant dining room with "
                   "standard-height dining tables about 75cm tall and chairs, several "
                   "adults seated at tables for scale, a server standing beside one "
                   "table, warm ambient pendant lighting.",
    },
    {
        "category": "interior",
        "id": "interior_09_gym",
        "prompt": "photorealistic photograph of an indoor gym with a standard 2032mm "
                   "doorway and a rack of dumbbells, one adult standing beside a squat "
                   "rack for scale, rubber flooring, bright fluorescent lighting.",
    },
    {
        "category": "interior",
        "id": "interior_10_classroom",
        "prompt": "photorealistic photograph of a school classroom with rows of "
                   "standard student desks about 75cm tall and chairs, a whiteboard at "
                   "the front, a teacher standing beside the whiteboard for scale, "
                   "daylight through tall windows.",
    },
    {
        "category": "interior",
        "id": "interior_11_warehouse",
        "prompt": "photorealistic photograph of the interior of a warehouse with tall "
                   "steel shelving racks about 4 meters high holding cardboard boxes, a "
                   "forklift parked between the racks, one adult worker standing beside "
                   "the forklift for scale, overhead industrial lighting.",
    },
    {
        "category": "interior",
        "id": "interior_12_church_interior",
        "prompt": "photorealistic photograph of the interior of a church nave with tall "
                   "wooden pews, standard bench height about 45cm, and a high vaulted "
                   "ceiling, one adult seated in a pew for scale, stained-glass windows, "
                   "soft daylight streaming in.",
    },
    {
        "category": "interior",
        "id": "interior_13_retail_store",
        "prompt": "photorealistic photograph of a retail clothing store interior with "
                   "standard clothing racks about 1.5 meters tall and a checkout "
                   "counter, one adult customer browsing a rack for scale, a cashier at "
                   "the counter, bright retail lighting.",
    },
    {
        "category": "interior",
        "id": "interior_14_elevator_lobby",
        "prompt": "photorealistic photograph of a building elevator lobby with a "
                   "standard 2032mm elevator door and a directory sign on the wall, one "
                   "adult standing waiting for the elevator for scale, polished floor "
                   "reflecting overhead lighting.",
    },
    {
        "category": "interior",
        "id": "interior_15_office",
        "prompt": "photorealistic photograph of an open-plan office with standard "
                   "cubicle partitions about 1.5 meters tall and desks, one adult seated "
                   "at a desk typing for scale, a water cooler beside the wall, bright "
                   "overhead fluorescent lighting.",
    },
    # ------------------------------------------------------------------ Nature
    {
        "category": "nature",
        "id": "nature_01_forest_trail",
        "prompt": "photorealistic photograph of a forest trail, one adult hiker standing "
                   "beside a large tree trunk for scale, dense pine forest in the "
                   "background, dappled midday sunlight through the canopy.",
    },
    {
        "category": "nature",
        "id": "nature_02_mountain_vista",
        "prompt": "photorealistic photograph of a hiker standing on a rocky overlook "
                   "with a mountain range in the far distance, a standard backpack about "
                   "60cm tall on the ground beside them for scale, clear afternoon light.",
    },
    {
        "category": "nature",
        "id": "nature_03_beach",
        "prompt": "photorealistic photograph of a sandy beach, one adult standing near "
                   "the shoreline, a standard beach umbrella about 2 meters tall planted "
                   "in the sand beside them, gentle waves in the background, bright "
                   "midday sun.",
    },
    {
        "category": "nature",
        "id": "nature_04_desert_dunes",
        "prompt": "photorealistic photograph of desert sand dunes, one adult walking "
                   "along a dune ridge for scale, a parked SUV at the base of the dunes "
                   "in the distance, clear blue sky, late-afternoon warm light.",
    },
    {
        "category": "nature",
        "id": "nature_05_waterfall",
        "prompt": "photorealistic photograph of a waterfall cascading into a pool, one "
                   "adult standing on rocks at the base of the falls for scale, mist "
                   "rising from the pool, midday overcast light.",
    },
    {
        "category": "nature",
        "id": "nature_06_lake_shore",
        "prompt": "photorealistic photograph of a calm lake shoreline with a wooden "
                   "dock, one adult standing on the dock beside a canoe about 4.5 "
                   "meters long pulled up on the shore, mountains reflected in the "
                   "water, early-morning light.",
    },
    {
        "category": "nature",
        "id": "nature_07_canyon",
        "prompt": "photorealistic photograph of a hiker standing at the rim of a canyon "
                   "with layered rock walls descending below, a standard hiking pole "
                   "about 1.3 meters long in hand for scale, midday clear light.",
    },
    {
        "category": "nature",
        "id": "nature_08_meadow",
        "prompt": "photorealistic photograph of an open meadow with tall grass, one "
                   "adult standing in the field with a golden retriever dog about 60cm "
                   "tall at the shoulder beside them for scale, distant tree line, soft "
                   "late-afternoon light.",
    },
    {
        "category": "nature",
        "id": "nature_09_riverbank",
        "prompt": "photorealistic photograph of a riverbank with smooth stones, one "
                   "adult crouched beside the water fishing, a standard kayak about 3 "
                   "meters long resting on the bank beside them, midday light with "
                   "gentle reflections.",
    },
    {
        "category": "nature",
        "id": "nature_10_snowy_slope",
        "prompt": "photorealistic photograph of a snowy mountain slope, one skier "
                   "standing at the top for scale, evergreen trees dotting the slope "
                   "below, bright clear winter daylight.",
    },
    {
        "category": "nature",
        "id": "nature_11_rocky_coastline",
        "prompt": "photorealistic photograph of a rocky ocean coastline with crashing "
                   "waves, one adult standing on a flat rock outcrop for scale, a "
                   "lighthouse visible in the distance, overcast dramatic sky.",
    },
    {
        "category": "nature",
        "id": "nature_12_redwood_forest",
        "prompt": "photorealistic photograph of a towering redwood forest, one adult "
                   "standing at the base of a massive redwood trunk for scale, ferns "
                   "covering the forest floor, soft filtered light through the canopy.",
    },
    {
        "category": "nature",
        "id": "nature_13_farmland_silo",
        "prompt": "photorealistic photograph of open farmland with a grain silo about "
                   "five stories (15 meters) tall beside a red barn, a pickup truck "
                   "parked near the barn, one farmer standing beside the truck for "
                   "scale, golden late-afternoon light.",
    },
    {
        "category": "nature",
        "id": "nature_14_botanical_garden",
        "prompt": "photorealistic photograph of a botanical garden path lined with "
                   "flowering shrubs, one adult walking along the path beside a stone "
                   "garden bench about 1.5 meters long, midday bright sunlight.",
    },
    {
        "category": "nature",
        "id": "nature_15_cliffside_path",
        "prompt": "photorealistic photograph of a narrow cliffside hiking path along a "
                   "coastal bluff, one adult hiker walking the path with the ocean far "
                   "below for scale, a wooden trail marker post about 1.2 meters tall "
                   "beside the path, clear afternoon light.",
    },
]

assert len(SCENES) == 45, f"expected 45 scenes, got {len(SCENES)}"
assert sum(1 for s in SCENES if s["category"] == "exterior") == 15
assert sum(1 for s in SCENES if s["category"] == "interior") == 15
assert sum(1 for s in SCENES if s["category"] == "nature") == 15

# name, scene_type, geometry_mode, primitive_method — see the approved plan's
# variant table. geometry_mode/primitive_method are irrelevant (overridden
# internally) for the organic/indoor/outdoor presets, but ComfyUI still
# requires *some* valid enum value in each widget slot.
VARIANTS: list[tuple[str, str, str, str]] = [
    ("organic", "organic", "relief_mesh", "azimuth_walls"),
    ("indoor", "indoor", "primitives", "room_cuboid"),
    ("outdoor", "outdoor", "primitives", "ransac_planes"),
    ("manual_azimuth", "manual", "primitives", "azimuth_walls"),
    ("both", "manual", "both", "azimuth_walls"),
    # Same wall orientation/distance detection as azimuth_walls, but height
    # extruded to the real image-space silhouette top (reaches towers/spires/
    # sloped roofs azimuth_walls truncates) — see CLAUDE.md's "Sky-aware
    # depth" and "Multiple geometry-derivation strategies" notes.
    ("vertical_extrusion", "manual", "primitives", "vertical_extrusion"),
]


@dataclass
class ManifestRecord:
    scene_id: str
    category: str
    variant: str
    seed: int
    status: str
    runtime_seconds: float
    confidence: float | None = None
    source_method: str | None = None
    scale_source: str | None = None
    image_path: str | None = None
    obj_path: str | None = None
    glb_path: str | None = None
    maya_path: str | None = None
    solve_json_path: str | None = None
    error: str | None = None


def load_template() -> dict[str, Any]:
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


def build_graph(
    template: dict[str, Any],
    scene: dict[str, str],
    seed: int,
    scene_output_dir: Path,
) -> dict[str, Any]:
    """Clone the base Qwen-generation graph into a 5-branch evaluation graph.

    Nodes "3"/"6" (the single AtlasDeriveProjectionGeometry / AtlasExportReliefMesh
    pair in the saved template) are replaced entirely by 5 explicit clones —
    one per variant in VARIANTS — each with its own AtlasExportMayaReviewScene
    and AtlasExportSolveJSON. Node "4" (AtlasBlockoutViewport) and its
    PreviewImage "5" are dropped: headless automation has no browser to click
    "Render Passes", so their image outputs would just be black placeholders.
    """
    graph = copy.deepcopy(template)
    graph.pop("4", None)
    graph.pop("5", None)

    graph["12"]["inputs"]["text"] = scene["prompt"]
    graph["18"]["inputs"]["seed"] = seed
    graph["18"]["inputs"]["control_after_generate"] = "fixed"
    graph["7"]["inputs"]["filename_prefix"] = f"qwen_dataset_01/{scene['id']}"

    depth_model = CATEGORY_DEPTH_MODEL[scene["category"]]
    # AtlasLearnedSolveFromImage's own depth_model widget isn't reached by any
    # downstream scene_type preset — set it per-category so measure_from_depth
    # camera-height estimation uses the right model regardless of which
    # derive variant runs afterward.
    graph["2"]["inputs"]["depth_model"] = depth_model

    base_derive = copy.deepcopy(graph["3"])
    base_export = copy.deepcopy(graph["6"])
    del graph["3"]
    del graph["6"]

    for variant_name, scene_type, geometry_mode, primitive_method in VARIANTS:
        derive_id = f"derive_{variant_name}"
        maya_id = f"maya_{variant_name}"
        solve_json_id = f"solve_json_{variant_name}"
        variant_dir = scene_output_dir / variant_name

        derive_node = copy.deepcopy(base_derive)
        derive_node["inputs"].update({
            "scene_type": scene_type,
            "geometry_mode": geometry_mode,
            "primitive_method": primitive_method,
            "depth_model": depth_model,
        })
        graph[derive_id] = derive_node

        maya_inputs = {
            "solve": [derive_id, 0],
            "output_dir": str(variant_dir),
        }
        # AtlasExportReliefMesh runs entirely independently of primitive_method
        # (it re-derives depth and builds a mesh from scratch, never reading
        # the derive node's own proxy_geometry) — its output is IDENTICAL
        # across every "primitives"-only variant of the same photo. Wiring it
        # into every variant's Maya scene regardless of relevance made every
        # variant's scene contain the same dominant mesh chunk, burying the
        # actual wall-height difference this comparison exists to show. Only
        # export/import it for variants whose geometry_mode actually uses a
        # relief mesh (organic, both) — also saves ~4x redundant depth-model
        # inference per image for a mesh nothing else would ever reference.
        if geometry_mode in ("relief_mesh", "both"):
            export_id = f"export_{variant_name}"
            export_node = copy.deepcopy(base_export)
            export_node["inputs"].update({
                "solve": [derive_id, 0],
                "output_dir": str(variant_dir),
                "depth_model": depth_model,
            })
            graph[export_id] = export_node
            maya_inputs["relief_mesh_obj_path"] = [export_id, 0]

        graph[maya_id] = {
            "inputs": maya_inputs,
            "class_type": "AtlasExportMayaReviewScene",
            "_meta": {"title": f"Atlas Export Maya ({variant_name})"},
        }

        graph[solve_json_id] = {
            "inputs": {
                "solve": [derive_id, 0],
                "output_path": str(variant_dir / "atlas_solve.json"),
            },
            "class_type": "AtlasExportSolveJSON",
            "_meta": {"title": f"Atlas Export Solve JSON ({variant_name})"},
        }

    return graph


def queue_prompt(graph: dict[str, Any], comfy_url: str, client_id: str) -> str:
    payload = json.dumps({"prompt": graph, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(
        f"{comfy_url}/prompt", data=payload, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI rejected the prompt ({exc.code}): {detail}") from exc
    if body.get("node_errors"):
        raise RuntimeError(f"ComfyUI reported node errors before queuing: {body['node_errors']}")
    return body["prompt_id"]


def wait_for_history(
    prompt_id: str,
    comfy_url: str,
    *,
    timeout_s: float = 900.0,
    poll_interval_s: float = 2.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with urllib.request.urlopen(f"{comfy_url}/history/{prompt_id}", timeout=30) as resp:
            history = json.loads(resp.read())
        entry = history.get(prompt_id)
        if entry is not None:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                messages = status.get("messages", [])
                raise RuntimeError(f"ComfyUI execution failed for prompt {prompt_id}: {messages}")
            if entry.get("outputs") or status.get("completed"):
                return entry
        time.sleep(poll_interval_s)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id} after {timeout_s}s")


def fetch_generated_image(entry: dict[str, Any], comfy_url: str, destination: Path) -> str | None:
    """Copy the SaveImage (node "7") output into scene_output_dir/source_image.png."""
    images = entry.get("outputs", {}).get("7", {}).get("images", [])
    if not images:
        return None
    image = images[0]
    url = (
        f"{comfy_url}/view?filename={image['filename']}"
        f"&subfolder={image.get('subfolder', '')}&type={image.get('type', 'output')}"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        destination.write_bytes(resp.read())
    return str(destination)


def _read_solve_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def run_one_scene(
    scene: dict[str, str],
    seed: int,
    output_root: Path,
    template: dict[str, Any],
    comfy_url: str,
) -> list[ManifestRecord]:
    scene_dir = output_root / scene["id"]
    scene_dir.mkdir(parents=True, exist_ok=True)
    graph = build_graph(template, scene, seed, scene_dir)

    client_id = str(uuid.uuid4())
    started = time.perf_counter()
    prompt_id = queue_prompt(graph, comfy_url, client_id)
    entry = wait_for_history(prompt_id, comfy_url)
    runtime = time.perf_counter() - started

    image_path = fetch_generated_image(entry, comfy_url, scene_dir / "source_image.png")

    records = []
    for variant_name, *_ in VARIANTS:
        variant_dir = scene_dir / variant_name
        obj_path = variant_dir / "atlas_relief_mesh.obj"
        glb_path = variant_dir / "atlas_relief_mesh.glb"
        # AtlasExportMayaReviewScene calls build_review_package(), which always
        # nests its outputs (maya_open_scene.py, blender/nuke scripts, report.md,
        # its own atlas_solve.json copy) under a "atlas_review_001" package
        # subfolder rather than writing directly into output_dir.
        maya_path = variant_dir / "atlas_review_001" / "maya_open_scene.py"
        solve_json_path = variant_dir / "atlas_solve.json"
        solve_data = _read_solve_json(solve_json_path)
        debug_metadata = solve_data.get("debug_metadata", {})

        records.append(ManifestRecord(
            scene_id=scene["id"],
            category=scene["category"],
            variant=variant_name,
            seed=seed,
            status="ok" if solve_json_path.is_file() else "missing_outputs",
            runtime_seconds=runtime,
            confidence=solve_data.get("confidence"),
            source_method=solve_data.get("source_method"),
            scale_source=debug_metadata.get("scale_source"),
            image_path=image_path,
            obj_path=str(obj_path) if obj_path.is_file() else None,
            glb_path=str(glb_path) if glb_path.is_file() else None,
            maya_path=str(maya_path) if maya_path.is_file() else None,
            solve_json_path=str(solve_json_path) if solve_json_path.is_file() else None,
        ))
    return records


def write_manifest(records: list[ManifestRecord], output_root: Path) -> None:
    json_path = output_root / "manifest.json"
    csv_path = output_root / "manifest.csv"
    json_path.write_text(
        json.dumps([asdict(r) for r in records], indent=2, sort_keys=True), encoding="utf-8",
    )
    fieldnames = list(ManifestRecord.__dataclass_fields__)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--comfy-url", default=DEFAULT_COMFY_URL)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N scenes (smoke test).")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N scenes (resume a batch).")
    args = parser.parse_args()

    # Atlas export nodes resolve a relative output_dir/output_path against
    # ComfyUI's OWN process working directory, not this script's — always
    # resolve to absolute regardless of how --output-root was passed in.
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    template = load_template()

    scenes = SCENES[args.start_index:]
    if args.limit is not None:
        scenes = scenes[: args.limit]

    all_records: list[ManifestRecord] = []
    for index, scene in enumerate(scenes):
        seed = BASE_SEED + args.start_index + index
        print(f"[{index + 1}/{len(scenes)}] {scene['id']} (seed={seed})")
        try:
            records = run_one_scene(scene, seed, output_root, template, args.comfy_url)
            all_records.extend(records)
            print(f"  ok — {len(records)} variants")
        except Exception as exc:  # noqa: BLE001 — one bad image must not abort the batch
            print(f"  ERROR: {exc}", file=sys.stderr)
            all_records.append(ManifestRecord(
                scene_id=scene["id"],
                category=scene["category"],
                variant="(all)",
                seed=seed,
                status="error",
                runtime_seconds=0.0,
                error=str(exc),
            ))
        # Write the manifest after every scene, not just at the end, so a
        # crashed/interrupted batch still leaves a reviewable partial result.
        write_manifest(all_records, output_root)

    ok = sum(1 for r in all_records if r.status == "ok")
    print(f"\nDone: {ok}/{len(all_records)} records ok, {len(scenes)} scenes attempted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
