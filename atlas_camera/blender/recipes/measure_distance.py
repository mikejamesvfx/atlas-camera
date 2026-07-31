"""Exact point-to-SURFACE distance via Blender's BVH. A measuring instrument.

Written after a drift metric convicted itself: nearest-VERTEX against a
subsampled reference read 2.48x the median edge for points that had not moved at
all, and an acceptance gate of 2.0x was therefore below its own noise floor.

Nearest-vertex is not nearest-surface. On a mesh with 0.23m edges a point dead
centre in a triangle is ~0.13m from any vertex before any real drift exists.
`BVHTree.find_nearest` gives the actual quantity, against every triangle, with
no subsampling — which is what "is this on the surface" has always meant.

Runs inside Blender's interpreter. Never imported by Atlas — it imports bpy.
"""
import json
import sys
import time
import traceback

_T0 = time.time()

import numpy as np  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402


def _exchange_dir():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--exchange" not in argv:
        raise RuntimeError("missing --exchange <dir>")
    return argv[argv.index("--exchange") + 1]


def main():
    ex = _exchange_dir()
    stage = "load"
    try:
        with np.load(f"{ex}/in.npz") as data:
            query = np.asarray(data["patch_vertices"], dtype=np.float64)
            rv = np.asarray(data["target_vertices"], dtype=np.float64)
            rf = np.asarray(data["target_faces"], dtype=np.int64)

        stage = "bvh"
        t = time.time()
        bvh = BVHTree.FromPolygons([tuple(v) for v in rv],
                                   [tuple(f) for f in rf], all_triangles=True)
        build_s = round(time.time() - t, 3)

        stage = "query"
        t = time.time()
        dist = np.empty(len(query), dtype=np.float64)
        for i, p in enumerate(query):
            hit = bvh.find_nearest(tuple(p))
            # (location, normal, index, distance); None when nothing is found,
            # which for a non-empty tree means the point is unreachable rather
            # than close — record inf so it cannot masquerade as a good result.
            dist[i] = hit[3] if hit is not None and hit[3] is not None else np.inf
        query_s = round(time.time() - t, 3)

        stage = "write"
        np.savez(f"{ex}/out.npz", vertices=query,
                 faces=np.zeros((1, 3), dtype=np.int32),
                 snapped=np.isfinite(dist), distance=dist)
        finite = dist[np.isfinite(dist)]
        with open(f"{ex}/report.json", "w", encoding="utf-8") as fh:
            json.dump({
                "recipe": "measure_distance",
                "n_query": int(len(query)), "n_ref_tris": int(len(rf)),
                "n_unreachable": int((~np.isfinite(dist)).sum()),
                "median_m": round(float(np.median(finite)), 6) if len(finite) else None,
                "p95_m": round(float(np.percentile(finite, 95)), 6) if len(finite) else None,
                "max_m": round(float(finite.max()), 6) if len(finite) else None,
                "bvh_build_s": build_s, "query_s": query_s,
                "seconds": round(time.time() - _T0, 3),
            }, fh, indent=1)
        print("ATLAS_RECIPE_OK measure_distance")
    except Exception as exc:  # noqa: BLE001
        try:
            with open(f"{ex}/error.json", "w", encoding="utf-8") as fh:
                json.dump({"stage": stage, "type": type(exc).__name__,
                           "message": str(exc),
                           "traceback": traceback.format_exc()}, fh, indent=1)
        except Exception:  # noqa: BLE001
            pass
        sys.exit(3)


main()
