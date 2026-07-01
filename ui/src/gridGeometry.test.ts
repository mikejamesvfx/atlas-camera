import { describe, expect, it } from "vitest";
import { buildXyzGrid, fitVanishingPoint } from "./gridGeometry";
import type { Constraints } from "./types";

describe("gridGeometry", () => {
  it("fits a vanishing point from guide intersections", () => {
    const vp = fitVanishingPoint([
      [[0, 60], [100, 40]],
      [[0, 100], [100, 120]]
    ]);

    expect(vp?.[0]).toBeCloseTo(-100, 1);
    expect(vp?.[1]).toBeCloseTo(80, 1);
  });

  it("builds full-image axis grid lines from live constraints", () => {
    const constraints: Constraints = {
      image_width: 160,
      image_height: 96,
      line_groups: {
        left: [
          [[0, 62], [159, 12]],
          [[0, 57], [159, 26]]
        ],
        right: [
          [[0, 12], [159, 36]],
          [[0, 26], [159, 42]]
        ],
        vertical: []
      },
      scale_constraints: []
    };

    const grid = buildXyzGrid(constraints, null);

    expect(grid.x.lines.length).toBeGreaterThan(8);
    expect(grid.z.lines.length).toBeGreaterThan(8);
    expect(grid.y.lines.length).toBeGreaterThan(8);
    for (const axis of [grid.x, grid.y, grid.z]) {
      for (const line of axis.lines) {
        for (const point of line) {
          expect(point[0]).toBeGreaterThanOrEqual(0);
          expect(point[0]).toBeLessThanOrEqual(160);
          expect(point[1]).toBeGreaterThanOrEqual(0);
          expect(point[1]).toBeLessThanOrEqual(96);
        }
      }
    }
  });
});
