"""List and search Atlas Camera scale references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_camera.reference_data import (
    get_scale_reference,
    list_categories,
    search_scale_references,
)


def _format_reference(reference) -> str:
    dimensions = [f"H {reference.height:g} {reference.units}"]
    if reference.width is not None:
        dimensions.append(f"W {reference.width:g} {reference.units}")
    if reference.depth is not None:
        dimensions.append(f"D {reference.depth:g} {reference.units}")
    return (
        f"{reference.id:34} {reference.category:13} "
        f"{', '.join(dimensions):28} {reference.label}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", "-q", help="Search text.")
    parser.add_argument("--category", help="Filter by category.")
    parser.add_argument("--id", help="Show one reference by id.")
    parser.add_argument("--categories", action="store_true", help="List categories.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args()

    if args.categories:
        categories = list_categories()
        if args.json:
            print(json.dumps(categories, indent=2))
        else:
            for category in categories:
                print(category)
        return 0

    if args.id:
        references = [get_scale_reference(args.id)]
    else:
        references = search_scale_references(args.query, category=args.category)

    if args.json:
        print(json.dumps([reference.to_dict() for reference in references], indent=2))
        return 0

    for reference in references:
        print(_format_reference(reference))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

