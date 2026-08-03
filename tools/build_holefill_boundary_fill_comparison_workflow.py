"""Build a local native-vs-Fill-Mesh comparison from a holefill lab graph.

Usage::

    python tools/build_holefill_boundary_fill_comparison_workflow.py \
        C:\\Users\\you\\Downloads\\2026-08-01_atlas_holefill_lab.json

The output is deliberately under ``examples/local/``: it is a working artist
lab, not a claimed portable showcase. It keeps A–G untouched and converts H
into four tests: Atlas's scoped BMesh fill, the installed Fill Mesh add-on,
mask-authoritative pure-NumPy surface reconstruction, and that reconstruction
followed by depth-safe dual-sheet seam refinement.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "local" / "atlas_holefill_boundary_fill_comparison.json"


def _by_name(items: list[dict], name: str) -> dict:
    for item in items:
        if item.get("name") == name:
            return item
    raise ValueError(f"missing socket {name!r}")


def _add_link(graph: dict, link_id: int, origin: dict, output: str,
              target: dict, input_name: str, kind: str) -> None:
    out = _by_name(origin["outputs"], output)
    out.setdefault("links", []).append(link_id)
    _by_name(target["inputs"], input_name)["link"] = link_id
    graph["links"].append([link_id, origin["id"], origin["outputs"].index(out),
                           target["id"], target["inputs"].index(
                               _by_name(target["inputs"], input_name)), kind])


def _link(graph: dict, link_id: int) -> list:
    for link in graph["links"]:
        if link[0] == link_id:
            return link
    raise ValueError(f"missing link {link_id}")


def _clear_input_links(node: dict) -> None:
    for inp in node["inputs"]:
        inp["link"] = None


def _clear_output_links(node: dict) -> None:
    for out in node["outputs"]:
        out["links"] = []


def _boundary_node(node: dict, *, backend: str, title: str) -> None:
    node["type"] = "AtlasBlenderBoundaryFill"
    node["title"] = title
    node["properties"]["Node name for S&R"] = "AtlasBlenderBoundaryFill"
    node["inputs"] = [
        {"name": "solve", "type": "ATLAS_SOLVE", "link": None},
        {"name": "hole_mask", "type": "MASK", "link": None},
    ]
    node["outputs"] = [
        {"name": "solve", "type": "ATLAS_SOLVE", "links": []},
        {"name": "report", "type": "STRING", "links": []},
    ]
    # New-node widgets are positional. This order must follow INPUT_TYPES.
    node["widgets_values"] = ["", backend, 256, "", 600]


def _reconstruct_node(node: dict) -> None:
    node["type"] = "AtlasMaskedSurfaceReconstruct"
    node["title"] = (
        "[H_numpy] masked surface reconstruction — manufactured local rim")
    node["properties"]["Node name for S&R"] = "AtlasMaskedSurfaceReconstruct"
    node["inputs"] = [
        {"name": "solve", "type": "ATLAS_SOLVE", "link": None},
        {"name": "hole_mask", "type": "MASK", "link": None},
    ]
    node["outputs"] = [
        {"name": "solve", "type": "ATLAS_SOLVE", "links": []},
        {"name": "remaining_holes", "type": "MASK", "links": []},
        {"name": "created_region", "type": "MASK", "links": []},
        {"name": "report", "type": "STRING", "links": []},
    ]
    # INPUT_TYPES optional widget order: layer, rim, budget, fraction,
    # enclosed-only, Jacobi iterations.
    node["widgets_values"] = ["", 1, 64, 0.20, True, 128]


def _seam_node(node: dict) -> None:
    node["type"] = "AtlasRefineOcclusionSeams"
    node["title"] = (
        "[H_seams] smooth sawtooth boundaries — independent depth sheets")
    node["properties"]["Node name for S&R"] = "AtlasRefineOcclusionSeams"
    node["inputs"] = [
        {"name": "solve", "type": "ATLAS_SOLVE", "link": None},
        {"name": "hole_mask", "type": "MASK", "link": None},
    ]
    node["outputs"] = [
        {"name": "solve", "type": "ATLAS_SOLVE", "links": []},
        {"name": "remaining_holes", "type": "MASK", "links": []},
        {"name": "created_region", "type": "MASK", "links": []},
        {"name": "report", "type": "STRING", "links": []},
    ]
    # INPUT_TYPES optional widget order. Never insert ahead of these values.
    node["widgets_values"] = [
        "", 2.0, 8, 0.35, 256, 0.08, 2, "away_from_camera"]


def build(source: Path) -> dict:
    graph = json.loads(source.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    patch = next(node for node in nodes.values()
                 if node["type"] == "AtlasPlanarHolePatch"
                 and node.get("title") == "[H] patch first")
    native = next(node for node in nodes.values()
                  if node["type"] == "AtlasBlenderOrganicFill"
                  and node.get("title") == "[H] organic on remainder")
    native_debug = next(node for node in nodes.values()
                        if node.get("title") == "[H_patch_organic] debug")
    native_viewport = next(node for node in nodes.values()
                           if node.get("title") == "[H] viewport")

    # Keep the original H solve connection; only its target class changes.
    native_solve_links = list(_by_name(native["outputs"], "solve").get("links") or [])
    native_report_links = list(_by_name(native["outputs"], "report").get("links") or [])
    _boundary_node(native, backend="native",
                   title="[H_native] scoped boundary fill — Atlas BMesh (no voxel remesh)")
    _by_name(native["inputs"], "solve")["link"] = 77
    _by_name(native["outputs"], "solve")["links"] = native_solve_links
    _by_name(native["outputs"], "report")["links"] = native_report_links
    _add_link(graph, 86, patch, "remaining_holes", native, "hole_mask", "MASK")
    native_debug["title"] = "[H_native] debug — planar + native BMesh"
    native_debug["widgets_values"] = ["atlas_debug/holefill_H_native_boundary.json"]
    native_viewport["title"] = "[H_native] viewport — original topology + closure faces"

    # A second branch starts from the exact same planar mesh and hole mask.
    addon = copy.deepcopy(native)
    addon["id"] = 43
    addon["order"] = max(node.get("order", 0) for node in graph["nodes"]) + 1
    addon["pos"] = [980, 4940]
    _boundary_node(addon, backend="fill_mesh_addon",
                   title="[H_addon] scoped boundary fill — installed Fill Mesh")
    graph["nodes"].append(addon)

    addon_debug = copy.deepcopy(native_debug)
    addon_debug["id"] = 44
    addon_debug["order"] = addon["order"] + 1
    addon_debug["pos"] = [1390, 4940]
    addon_debug["title"] = "[H_addon] debug — planar + installed Fill Mesh"
    addon_debug["widgets_values"] = ["atlas_debug/holefill_H_fill_mesh_addon.json"]
    _clear_input_links(addon_debug)
    _clear_output_links(addon_debug)
    graph["nodes"].append(addon_debug)

    addon_viewport = copy.deepcopy(native_viewport)
    addon_viewport["id"] = 45
    addon_viewport["order"] = addon_debug["order"] + 1
    addon_viewport["pos"] = [1800, 4940]
    addon_viewport["title"] = "[H_addon] viewport — installed Fill Mesh result"
    _clear_input_links(addon_viewport)
    _clear_output_links(addon_viewport)
    graph["nodes"].append(addon_viewport)

    # Existing H links carry the initial solve, debug source/depth, and viewport
    # source/depth. Derive their origins instead of hard-coding shared node IDs.
    patch_to_native = _link(graph, 77)
    native_debug_depth = _link(graph, 79)
    native_view_source = _link(graph, 83)
    native_view_depth = _link(graph, 84)
    link_nodes = {node["id"]: node for node in graph["nodes"]}
    _add_link(graph, 87, patch, "solve", addon, "solve", "ATLAS_SOLVE")
    _add_link(graph, 88, patch, "remaining_holes", addon, "hole_mask", "MASK")
    _add_link(graph, 89, addon, "solve", addon_debug, "solve", "ATLAS_SOLVE")
    _add_link(graph, 90, addon, "report", addon_debug, "status_2", "STRING")
    _add_link(graph, 91, patch, "report", addon_debug, "status_1", "STRING")
    _add_link(graph, 92, addon, "solve", addon_viewport, "solve", "ATLAS_SOLVE")
    _add_link(graph, 93, patch, "created_islands", addon_viewport, "patch_mask", "MASK")
    _add_link(graph, 94, link_nodes[native_view_source[1]],
              link_nodes[native_view_source[1]]["outputs"][native_view_source[2]]["name"],
              addon_viewport, "source_image", native_view_source[5])
    _add_link(graph, 95, link_nodes[native_view_depth[1]],
              link_nodes[native_view_depth[1]]["outputs"][native_view_depth[2]]["name"],
              addon_viewport, "primary_depth", native_view_depth[5])

    # Third branch: no Blender at all. The same planar remainder is treated as
    # an authoritative image-space region, expanded by one cell to manufacture
    # a clean support rim, then reconstructed along exact camera rays.
    numpy_reconstruct = copy.deepcopy(native)
    numpy_reconstruct["id"] = 46
    numpy_reconstruct["order"] = addon_viewport["order"] + 1
    numpy_reconstruct["pos"] = [980, 5580]
    _reconstruct_node(numpy_reconstruct)
    graph["nodes"].append(numpy_reconstruct)

    numpy_debug = copy.deepcopy(native_debug)
    numpy_debug["id"] = 47
    numpy_debug["order"] = numpy_reconstruct["order"] + 1
    numpy_debug["pos"] = [1390, 5580]
    numpy_debug["title"] = "[H_numpy] debug — manufactured rim reconstruction"
    numpy_debug["widgets_values"] = [
        "atlas_debug/holefill_H_numpy_surface_reconstruct.json"]
    _clear_input_links(numpy_debug)
    _clear_output_links(numpy_debug)
    graph["nodes"].append(numpy_debug)

    numpy_viewport = copy.deepcopy(native_viewport)
    numpy_viewport["id"] = 48
    numpy_viewport["order"] = numpy_debug["order"] + 1
    numpy_viewport["pos"] = [1800, 5580]
    numpy_viewport["title"] = "[H_numpy] viewport — camera-ray reconstruction"
    _clear_input_links(numpy_viewport)
    _clear_output_links(numpy_viewport)
    graph["nodes"].append(numpy_viewport)

    _add_link(graph, 96, patch, "solve", numpy_reconstruct, "solve", "ATLAS_SOLVE")
    _add_link(graph, 97, patch, "remaining_holes", numpy_reconstruct,
              "hole_mask", "MASK")
    _add_link(graph, 98, numpy_reconstruct, "solve", numpy_debug,
              "solve", "ATLAS_SOLVE")
    _add_link(graph, 99, numpy_reconstruct, "report", numpy_debug,
              "status_2", "STRING")
    _add_link(graph, 100, patch, "report", numpy_debug, "status_1", "STRING")
    _add_link(graph, 101, numpy_reconstruct, "solve", numpy_viewport,
              "solve", "ATLAS_SOLVE")
    _add_link(graph, 102, numpy_reconstruct, "created_region", numpy_viewport,
              "patch_mask", "MASK")
    _add_link(graph, 103, link_nodes[native_view_source[1]],
              link_nodes[native_view_source[1]]["outputs"][native_view_source[2]]["name"],
              numpy_viewport, "source_image", native_view_source[5])
    _add_link(graph, 104, link_nodes[native_view_depth[1]],
              link_nodes[native_view_depth[1]]["outputs"][native_view_depth[2]]["name"],
              numpy_viewport, "primary_depth", native_view_depth[5])

    # Fourth branch: preserve the raw NumPy result above for inspection, then
    # refine only the remaining camera-space tear boundaries.  Near and far
    # layers receive separate overlapping strips and are never bridged.
    seam_refine = copy.deepcopy(numpy_reconstruct)
    seam_refine["id"] = 49
    seam_refine["order"] = numpy_viewport["order"] + 1
    seam_refine["pos"] = [980, 6220]
    _seam_node(seam_refine)
    graph["nodes"].append(seam_refine)

    seam_debug = copy.deepcopy(native_debug)
    seam_debug["id"] = 50
    seam_debug["order"] = seam_refine["order"] + 1
    seam_debug["pos"] = [1390, 6220]
    seam_debug["title"] = "[H_seams] debug — NumPy reconstruction + seam underlap"
    seam_debug["widgets_values"] = [
        "atlas_debug/holefill_H_numpy_occlusion_seams.json"]
    _clear_input_links(seam_debug)
    _clear_output_links(seam_debug)
    graph["nodes"].append(seam_debug)

    seam_viewport = copy.deepcopy(native_viewport)
    seam_viewport["id"] = 51
    seam_viewport["order"] = seam_debug["order"] + 1
    seam_viewport["pos"] = [1800, 6220]
    seam_viewport["title"] = (
        "[H_seams] viewport — smoothed dual-sheet underlap")
    _clear_input_links(seam_viewport)
    _clear_output_links(seam_viewport)
    graph["nodes"].append(seam_viewport)

    _add_link(graph, 105, numpy_reconstruct, "solve", seam_refine,
              "solve", "ATLAS_SOLVE")
    _add_link(graph, 106, numpy_reconstruct, "remaining_holes", seam_refine,
              "hole_mask", "MASK")
    _add_link(graph, 107, seam_refine, "solve", seam_debug,
              "solve", "ATLAS_SOLVE")
    _add_link(graph, 108, numpy_reconstruct, "report", seam_debug,
              "status_1", "STRING")
    _add_link(graph, 109, seam_refine, "report", seam_debug,
              "status_2", "STRING")
    _add_link(graph, 110, seam_refine, "solve", seam_viewport,
              "solve", "ATLAS_SOLVE")
    _add_link(graph, 111, seam_refine, "created_region", seam_viewport,
              "patch_mask", "MASK")
    _add_link(graph, 112, link_nodes[native_view_source[1]],
              link_nodes[native_view_source[1]]["outputs"][native_view_source[2]]["name"],
              seam_viewport, "source_image", native_view_source[5])
    _add_link(graph, 113, link_nodes[native_view_depth[1]],
              link_nodes[native_view_depth[1]]["outputs"][native_view_depth[2]]["name"],
              seam_viewport, "primary_depth", native_view_depth[5])

    # The copied native link remains valid after the class replacement.
    assert patch_to_native[3] == native["id"]
    assert native_debug_depth[3] == native_debug["id"]
    graph["last_node_id"] = 51
    graph["last_link_id"] = 113
    for group in graph.get("groups", []):
        if group.get("id") == 9:
            group["title"] = (
                "H — boundary-fill A/B, NumPy reconstruction, and dual-sheet seam refinement")
            group["bounding"] = [540, 4304, 2450, 2520]
    return graph


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("pass the source holefill-lab workflow path")
    source = Path(sys.argv[1])
    if not source.is_file():
        raise SystemExit(f"workflow not found: {source}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(source), indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
