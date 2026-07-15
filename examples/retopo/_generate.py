"""Throwaway generator for the 3 retopology demo workflows.

Emits ComfyUI editor/graph save-format JSON with bidirectionally-consistent
links (the invariant tests/test_example_workflows.py checks). Run once, then
this file can be deleted; the three .json files are the artifact.
"""
import json
import os

V2 = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"

# AtlasExportReliefMesh widget order (from nodes.py export() signature):
#   output_dir, grid_long_edge, depth_edge_rel, depth_model, device, format,
#   use_solve_mesh, max_edge_factor, normal_edge_deg,
#   fill_interior_holes, max_hole_edges, fill_depth_near_m, fill_depth_far_m,
#   retopo_method, retopo_target_vertex_count, retopo_smooth_iterations,
#   retopo_crease_angle, retopo_pure_quad
def relief_widgets(out_dir, *, retopo="off", target=2000, smooth=0,
                   crease=30.0, pure_quad=False, fill=False, max_hole=64,
                   grid=128, edge=0.5, fmt="both", use_solve=True):
    return [out_dir, grid, edge, V2, "auto", fmt, use_solve, 12.0, 0.0,
            fill, max_hole, 0.0, 0.0, retopo, target, smooth, crease, pure_quad]


class Graph:
    def __init__(self):
        self.nodes = []
        self.links = []  # each: [lid, oid, oslot, tid, tslot, type]
        self._lid = 0

    def add_node(self, id, type, pos, size, *, inputs=None, outputs=None,
                 widgets=None, title=None, order=None):
        node = {
            "id": id, "type": type, "pos": pos, "size": size,
            "flags": {}, "order": order if order is not None else id - 1,
            "mode": 0,
            "inputs": inputs or [],
            "outputs": outputs or [],
            "properties": {"Node name for S&R": type},
            "widgets_values": widgets or [],
        }
        if title:
            node["title"] = title
        self.nodes.append(node)
        return node

    def out(self, name, type, links, slot_index=0):
        return {"name": name, "type": type,
                "links": links, "slot_index": slot_index}

    def inp(self, name, type, link):
        return {"name": name, "type": type, "link": link}

    def link(self, oid, oslot, tid, tslot, type):
        self._lid += 1
        lid = self._lid
        self.links.append([lid, oid, oslot, tid, tslot, type])
        return lid

    def finalize(self, name):
        return {
            "id": name, "revision": 0,
            "last_node_id": max(n["id"] for n in self.nodes),
            "last_link_id": self._lid,
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "config": {},
            "extra": {},
            "version": 0.4,
        }


def load_image_node(id, pos, filename="atlas_monument_valley.png"):
    return {
        "id": id, "type": "LoadImage", "pos": pos, "size": [280, 314],
        "flags": {}, "order": 0, "mode": 0,
        "inputs": [],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "MASK", "type": "MASK", "links": None},
        ],
        "properties": {"Node name for S&R": "LoadImage"},
        "widgets_values": [filename, "image"],
    }


def learned_solve_node(id, pos, image_link):
    return {
        "id": id, "type": "AtlasLearnedSolveFromImage", "pos": pos,
        "size": [340, 230], "flags": {}, "order": 1, "mode": 0,
        "inputs": [{"name": "image", "type": "IMAGE", "link": image_link}],
        "outputs": [
            {"name": "solve", "type": "ATLAS_SOLVE", "links": [], "slot_index": 0},
        ],
        "properties": {"Node name for S&R": "AtlasLearnedSolveFromImage"},
        "widgets_values": [
            "measure_from_depth", 1.6, V2, 36.0, "pinhole", "auto",
        ],
        "title": "Atlas Learned Solve (GeoCalib prior + depth-measured height)",
    }


def viewport_node(id, pos, solve_link, image_link):
    return {
        "id": id, "type": "AtlasBlockoutViewport", "pos": pos,
        "size": [960, 720], "flags": {}, "mode": 0,
        "inputs": [
            {"name": "solve", "type": "ATLAS_SOLVE", "link": solve_link},
            {"name": "source_image", "type": "IMAGE", "link": image_link},
        ],
        "outputs": [
            {"name": "shaded", "type": "IMAGE", "links": None, "slot_index": 0},
            {"name": "depth", "type": "IMAGE", "links": None},
            {"name": "normal", "type": "IMAGE", "links": None},
            {"name": "mask", "type": "IMAGE", "links": None},
            {"name": "path_frames", "type": "IMAGE", "links": None},
            {"name": "camera_path", "type": "ATLAS_CAMERA_PATH", "links": None},
            {"name": "patch_azimuth_view", "type": "STRING", "links": None},
            {"name": "patch_elevation_view", "type": "STRING", "links": None},
            {"name": "patch_distance", "type": "STRING", "links": None},
            {"name": "patch_prompt", "type": "STRING", "links": None},
            {"name": "patch_exact", "type": "STRING", "links": None},
        ],
        "properties": {"Node name for S&R": "AtlasBlockoutViewport"},
        "widgets_values": [1464, "", 1.0],
    }


def note_node(id, pos, size, text):
    return {
        "id": id, "type": "Note", "pos": pos, "size": size,
        "flags": {}, "mode": 0, "inputs": [], "outputs": [],
        "properties": {"Node name for S&R": "Note"},
        "widgets_values": [text],
        "title": "Note",
    }


# ---------------------------------------------------------------------------
# Workflow 1 — smooth A/B (retopo off baseline vs Taubin smooth)
# ---------------------------------------------------------------------------

def workflow_1():
    g = Graph()
    li = load_image_node(1, [40, 120])
    g.nodes.append(li)
    # links from LoadImage IMAGE -> solve(2), export-baseline image(3), export-smooth image(4)
    l1 = g.link(1, 0, 2, 0, "IMAGE")   # -> solve.image
    l2 = g.link(1, 0, 3, 1, "IMAGE")   # -> baseline.image
    l3 = g.link(1, 0, 4, 1, "IMAGE")   # -> smooth.image
    li["outputs"][0]["links"] = [l1, l2, l3]

    solve = learned_solve_node(2, [380, 120], l1)
    g.nodes.append(solve)
    l4 = g.link(2, 0, 3, 0, "ATLAS_SOLVE")  # -> baseline.solve
    l5 = g.link(2, 0, 4, 0, "ATLAS_SOLVE")  # -> smooth.solve
    solve["outputs"][0]["links"] = [l4, l5]

    baseline = {
        "id": 3, "type": "AtlasExportReliefMesh", "pos": [780, 120],
        "size": [360, 360], "flags": {}, "mode": 0,
        "inputs": [
            {"name": "solve", "type": "ATLAS_SOLVE", "link": l4},
            {"name": "image", "type": "IMAGE", "link": l2},
        ],
        "outputs": [
            {"name": "obj_path", "type": "STRING", "links": None, "slot_index": 0},
            {"name": "glb_path", "type": "STRING", "links": None, "slot_index": 1},
            {"name": "preview_solve", "type": "ATLAS_SOLVE", "links": None, "slot_index": 2},
            {"name": "report", "type": "STRING", "links": None, "slot_index": 3},
        ],
        "properties": {"Node name for S&R": "AtlasExportReliefMesh"},
        "widgets_values": relief_widgets(
            "atlas_exports/relief_off", retopo="off"),
        "title": "Export — baseline (retopo OFF)",
    }
    g.nodes.append(baseline)

    smooth = {
        "id": 4, "type": "AtlasExportReliefMesh", "pos": [780, 520],
        "size": [360, 360], "flags": {}, "mode": 0,
        "inputs": [
            {"name": "solve", "type": "ATLAS_SOLVE", "link": l5},
            {"name": "image", "type": "IMAGE", "link": l3},
        ],
        "outputs": [
            {"name": "obj_path", "type": "STRING", "links": None, "slot_index": 0},
            {"name": "glb_path", "type": "STRING", "links": None, "slot_index": 1},
            {"name": "preview_solve", "type": "ATLAS_SOLVE", "links": None, "slot_index": 2},
            {"name": "report", "type": "STRING", "links": None, "slot_index": 3},
        ],
        "properties": {"Node name for S&R": "AtlasExportReliefMesh"},
        "widgets_values": relief_widgets(
            "atlas_exports/relief_smooth", retopo="smooth", smooth=12),
        "title": "Export — trimesh Taubin SMOOTH (UVs preserved)",
    }
    g.nodes.append(smooth)

    g.nodes.append(note_node(5, [40, 470], [720, 360],
        "RETOPOLGY SMOOTH — A/B against the OFF baseline.\n\n"
        "Two AtlasExportReliefMesh nodes share ONE solve + image:\n"
        "· left = retopo_method OFF  (the default; torn silhouette, DMP-correct)\n"
        "· right = retopo_method SMOOTH (trimesh Taubin relax, 12 iterations)\n\n"
        "SMOOTH is topology-PRESERVING — same faces, same vertex count — so the\n"
        "1:1 vertex-UV mapping baked in build_relief_mesh stays valid and UVs are\n"
        "NOT regenerated (pure numpy, trimesh is already a dep — runs everywhere,\n"
        "no extra install). It only moves vertex positions: relaxes the noisy mono-\n"
        "depth ripple into a smoother surface for a cleaner DCC import.\n\n"
        "Export-only doctrine: retopology NEVER touches the live viewport projection\n"
        "mesh or solve.proxy_geometry — it runs once on the resolved ReliefMesh,\n"
        "after any hole-fill, before the OBJ/GLB writers.\n\n"
        "Compare the two OBJs in Maya/ZBrush/Blender: same vert/face count, smoother\n"
        "surface on the right, identical UVs/textures."))

    g.nodes.sort(key=lambda n: n["id"])
    return g.finalize("atlas-retopo-smooth-ab")


# ---------------------------------------------------------------------------
# Workflow 2 — quad retopo + interior hole-fill + USD camera (clean DCC handoff)
# ---------------------------------------------------------------------------

def workflow_2():
    g = Graph()
    li = load_image_node(1, [40, 120])
    g.nodes.append(li)
    l1 = g.link(1, 0, 2, 0, "IMAGE")   # -> solve.image
    l2 = g.link(1, 0, 3, 0, "IMAGE")   # -> depth.image
    l3 = g.link(1, 0, 5, 1, "IMAGE")   # -> export.image
    li["outputs"][0]["links"] = [l1, l2, l3]

    solve = learned_solve_node(2, [380, 120], l1)
    g.nodes.append(solve)
    l4 = g.link(2, 0, 3, 1, "ATLAS_SOLVE")  # -> depth.solve (optional focal)
    l5 = g.link(2, 0, 4, 0, "ATLAS_SOLVE")  # -> derive.solve
    solve["outputs"][0]["links"] = [l4, l5]

    depth = {
        "id": 3, "type": "AtlasDepthMap", "pos": [380, 400],
        "size": [340, 150], "flags": {}, "mode": 0,
        "inputs": [
            {"name": "image", "type": "IMAGE", "link": l2},
            {"name": "solve", "type": "ATLAS_SOLVE", "link": l4},
        ],
        "outputs": [
            {"name": "depth", "type": "ATLAS_DEPTH_MAP", "links": [], "slot_index": 0},
        ],
        "properties": {"Node name for S&R": "AtlasDepthMap"},
        "widgets_values": [V2, "auto"],
        "title": "AtlasDepthMap (run once, feed the derive + export)",
    }
    g.nodes.append(depth)
    l6 = g.link(3, 0, 4, 1, "ATLAS_DEPTH_MAP")  # -> derive.depth
    depth["outputs"][0]["links"] = [l6]

    derive = {
        "id": 4, "type": "AtlasDeriveReliefMesh", "pos": [760, 120],
        "size": [360, 230], "flags": {}, "mode": 0,
        "inputs": [
            {"name": "solve", "type": "ATLAS_SOLVE", "link": l5},
            {"name": "depth", "type": "ATLAS_DEPTH_MAP", "link": l6},
        ],
        "outputs": [
            {"name": "solve", "type": "ATLAS_SOLVE", "links": [], "slot_index": 0},
            {"name": "hole_mask", "type": "MASK", "links": None, "slot_index": 1},
        ],
        "properties": {"Node name for S&R": "AtlasDeriveReliefMesh"},
        "widgets_values": [256, "custom", 0.5, 12.0, True, 0.0],
        "title": "AtlasDeriveReliefMesh (grid 256, edge_rel 0.5)",
    }
    g.nodes.append(derive)
    l7 = g.link(4, 0, 5, 0, "ATLAS_SOLVE")  # -> export.solve
    l8 = g.link(4, 0, 6, 0, "ATLAS_SOLVE")  # -> USD.solve
    derive["outputs"][0]["links"] = [l7, l8]

    export = {
        "id": 5, "type": "AtlasExportReliefMesh", "pos": [1160, 120],
        "size": [360, 420], "flags": {}, "mode": 0,
        "inputs": [
            {"name": "solve", "type": "ATLAS_SOLVE", "link": l7},
            {"name": "image", "type": "IMAGE", "link": l3},
        ],
        "outputs": [
            {"name": "obj_path", "type": "STRING", "links": None, "slot_index": 0},
            {"name": "glb_path", "type": "STRING", "links": None, "slot_index": 1},
            {"name": "preview_solve", "type": "ATLAS_SOLVE", "links": None, "slot_index": 2},
            {"name": "report", "type": "STRING", "links": None, "slot_index": 3},
        ],
        "properties": {"Node name for S&R": "AtlasExportReliefMesh"},
        "widgets_values": relief_widgets(
            "atlas_exports/relief_quad",
            retopo="quad", target=3000, smooth=0, crease=25.0, pure_quad=True,
            fill=True, max_hole=48, fmt="both"),
        "title": "Export — hole-FILL then QUAD retopo (pure quad)",
    }
    g.nodes.append(export)

    usd = {
        "id": 6, "type": "AtlasExportUSD", "pos": [1160, 580],
        "size": [320, 90], "flags": {}, "mode": 0,
        "inputs": [{"name": "solve", "type": "ATLAS_SOLVE", "link": l8}],
        "outputs": [
            {"name": "usd_path", "type": "STRING", "links": None, "slot_index": 0},
        ],
        "properties": {"Node name for S&R": "AtlasExportUSD"},
        "widgets_values": ["atlas_exports"],
        "title": "AtlasExportUSD — camera.usda (needs [usd])",
    }
    g.nodes.append(usd)

    g.nodes.append(note_node(7, [40, 700], [1100, 260],
        "QUAD RETOPO + INTERIOR HOLE-FILL — the clean DCC handoff.\n\n"
        "Pipeline: Solve → DepthMap → DeriveReliefMesh → AtlasExportReliefMesh\n"
        "(fill_interior_holes ON, max_hole_edges 48  →  retopo_method QUAD,\n"
        "target 3000 verts, pure_quad True, crease 25°) + AtlasExportUSD (camera).\n\n"
        "The hole-fill runs FIRST (export-only: fans closed the small interior tear\n"
        "loops so the retopologizer gets a cleanable surface, never the outer\n"
        "silhouette/frame boundary). QUAD retopo runs AFTER on the capped mesh:\n"
        "pyinstantmeshes orientation-field quad remesh → N×4 quads → triangulated\n"
        "for the OBJ writer. Vertex count changes → the 1:1 vertex-UV mapping is\n"
        "REGENERATED from the recovered camera (pure numpy, no writer change) so the\n"
        "retopologized mesh stays textured with the source photo.\n\n"
        "Needs the pyinstantmeshes package (BSD, CPU-only, macOS arm64 wheels):\n    pip install pyinstantmeshes\n"
        "AtlasExportUSD needs the [usd] extra:  pip install atlas-camera[usd]\n"
        "Missing the quad dep → the node raises an actionable ImportError (the\n"
        "workflow still loads; swap retopo_method to 'decimate' or 'smooth' to run\n"
        "without it). AtlasDeriveReliefMesh's edge tuning (grid 256 / edge_rel 0.5)\n"
        "carries into the OBJ because use_solve_mesh defaults True."))

    g.nodes.sort(key=lambda n: n["id"])
    return g.finalize("atlas-retopo-quad-dcc-handoff")


# ---------------------------------------------------------------------------
# Workflow 3 — decimate to a budget + viewport preview
# ---------------------------------------------------------------------------

def workflow_3():
    g = Graph()
    li = load_image_node(1, [40, 120])
    g.nodes.append(li)
    l1 = g.link(1, 0, 2, 0, "IMAGE")   # -> solve.image
    l2 = g.link(1, 0, 3, 0, "IMAGE")   # -> depth.image
    l3 = g.link(1, 0, 5, 1, "IMAGE")   # -> viewport.source_image
    l4 = g.link(1, 0, 6, 1, "IMAGE")   # -> export.image
    li["outputs"][0]["links"] = [l1, l2, l3, l4]

    solve = learned_solve_node(2, [380, 120], l1)
    g.nodes.append(solve)
    l5 = g.link(2, 0, 3, 1, "ATLAS_SOLVE")  # -> depth.solve
    l6 = g.link(2, 0, 4, 0, "ATLAS_SOLVE")  # -> derive.solve
    solve["outputs"][0]["links"] = [l5, l6]

    depth = {
        "id": 3, "type": "AtlasDepthMap", "pos": [380, 400],
        "size": [340, 150], "flags": {}, "mode": 0,
        "inputs": [
            {"name": "image", "type": "IMAGE", "link": l2},
            {"name": "solve", "type": "ATLAS_SOLVE", "link": l5},
        ],
        "outputs": [
            {"name": "depth", "type": "ATLAS_DEPTH_MAP", "links": [], "slot_index": 0},
        ],
        "properties": {"Node name for S&R": "AtlasDepthMap"},
        "widgets_values": [V2, "auto"],
        "title": "AtlasDepthMap",
    }
    g.nodes.append(depth)
    l7 = g.link(3, 0, 4, 1, "ATLAS_DEPTH_MAP")  # -> derive.depth
    depth["outputs"][0]["links"] = [l7]

    derive = {
        "id": 4, "type": "AtlasDeriveReliefMesh", "pos": [760, 120],
        "size": [360, 230], "flags": {}, "mode": 0,
        "inputs": [
            {"name": "solve", "type": "ATLAS_SOLVE", "link": l6},
            {"name": "depth", "type": "ATLAS_DEPTH_MAP", "link": l7},
        ],
        "outputs": [
            {"name": "solve", "type": "ATLAS_SOLVE", "links": [], "slot_index": 0},
            {"name": "hole_mask", "type": "MASK", "links": None, "slot_index": 1},
        ],
        "properties": {"Node name for S&R": "AtlasDeriveReliefMesh"},
        "widgets_values": [256, "custom", 0.5, 12.0, True, 0.0],
        "title": "AtlasDeriveReliefMesh (grid 256)",
    }
    g.nodes.append(derive)
    l8 = g.link(4, 0, 5, 0, "ATLAS_SOLVE")  # -> viewport.solve
    l9 = g.link(4, 0, 6, 0, "ATLAS_SOLVE")  # -> export.solve
    derive["outputs"][0]["links"] = [l8, l9]

    g.nodes.append(viewport_node(5, [760, 400], l8, l3))

    export = {
        "id": 6, "type": "AtlasExportReliefMesh", "pos": [1160, 120],
        "size": [360, 420], "flags": {}, "mode": 0,
        "inputs": [
            {"name": "solve", "type": "ATLAS_SOLVE", "link": l9},
            {"name": "image", "type": "IMAGE", "link": l4},
        ],
        "outputs": [
            {"name": "obj_path", "type": "STRING", "links": None, "slot_index": 0},
            {"name": "glb_path", "type": "STRING", "links": None, "slot_index": 1},
            {"name": "preview_solve", "type": "ATLAS_SOLVE", "links": None, "slot_index": 2},
            {"name": "report", "type": "STRING", "links": None, "slot_index": 3},
        ],
        "properties": {"Node name for S&R": "AtlasExportReliefMesh"},
        "widgets_values": relief_widgets(
            "atlas_exports/relief_decimate",
            retopo="decimate", target=1500, fmt="both"),
        "title": "Export — QUADRIC DECIMATE to ~1500 verts (budget)",
    }
    g.nodes.append(export)

    g.nodes.append(note_node(7, [40, 760], [1100, 260],
        "DECIMATE TO A BUDGET — real-time / WebGL / file-size target.\n\n"
        "Pipeline: Solve → DepthMap → DeriveReliefMesh → (viewport preview +\n"
        "AtlasExportReliefMesh with retopo_method DECIMATE, target 1500).\n\n"
        "DECIMATE = quadric decimation via fast-simplification (backing\n"
        "trimesh.simplify_quadric_decimation). Pure decimation, no remeshing —\n"
        "keeps the original topology class, just fewer faces. The node targets\n"
        "~2× this in faces (decimate is face-count-driven). Vertex count changes\n"
        "→ projection-baked UVs are REGENERATED from the recovered camera so the\n"
        "light mesh stays textured (pure numpy).\n\n"
        "Needs the fast-simplification package (BSD, CPU-only, macOS arm64 wheels):\n    pip install fast-simplification\n"
        "Missing it → actionable ImportError (swap to 'smooth' to run without it).\n\n"
        "The AtlasBlockoutViewport shows the DENSE derived mesh (use_solve_mesh keeps\n"
        "the viewport on the un-decimated projection mesh); the export is the light,\n"
        "retopologized-for-budget OBJ. retopo_target_vertex_count is the lever —\n"
        "drop it for a real-time asset, raise it for a hero DCC handoff."))

    g.nodes.sort(key=lambda n: n["id"])
    return g.finalize("atlas-retopo-decimate-budget")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for name, wf in [
        ("atlas_retopo_smooth_ab_workflow.json", workflow_1()),
        ("atlas_retopo_quad_dcc_handoff_workflow.json", workflow_2()),
        ("atlas_retopo_decimate_budget_workflow.json", workflow_3()),
    ]:
        path = os.path.join(here, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wf, f, indent=2, ensure_ascii=False)
        # quick self-check: bidirectional link consistency
        nodes = {n["id"]: n for n in wf["nodes"]}
        errs = []
        for l in wf["links"]:
            lid, oid, oslot, tid, tslot = l[:5]
            outs = nodes[oid]["outputs"]
            ins = nodes[tid]["inputs"]
            if lid not in (outs[oslot].get("links") or []):
                errs.append(f"link {lid}: origin {oid}:{oslot} missing")
            if ins[tslot].get("link") != lid:
                errs.append(f"link {lid}: target {tid}:{tslot} missing")
        assert not errs, f"{name}: {errs}"
        print(f"wrote {name}  ({len(wf['nodes'])} nodes, {len(wf['links'])} links) OK")


if __name__ == "__main__":
    main()