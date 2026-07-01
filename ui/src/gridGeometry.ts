import type { Constraints, Point, Segment } from "./types";

export type AxisName = "x" | "y" | "z";

export type AxisGrid = {
  axis: AxisName;
  vanishingPoint?: Point;
  lines: Segment[];
};

export type XyzGrid = {
  x: AxisGrid;
  y: AxisGrid;
  z: AxisGrid;
};

const EPSILON = 1e-6;

export function buildXyzGrid(constraints: Constraints, solvePayload: any): XyzGrid {
  const width = constraints.image_width;
  const height = constraints.image_height;
  const solved = solvedVanishingPoints(solvePayload);
  const x = fitVanishingPoint(constraints.line_groups.left) ?? solved.x;
  const y = fitVanishingPoint(constraints.line_groups.vertical) ?? solved.y;
  const z = fitVanishingPoint(constraints.line_groups.right) ?? solved.z;

  return {
    x: { axis: "x", vanishingPoint: x, lines: gridLinesForAxis(x, width, height, "x") },
    y: { axis: "y", vanishingPoint: y, lines: gridLinesForAxis(y, width, height, "y") },
    z: { axis: "z", vanishingPoint: z, lines: gridLinesForAxis(z, width, height, "z") }
  };
}

export function fitVanishingPoint(lines: Segment[]): Point | undefined {
  if (lines.length < 2) return undefined;

  const intersections: Point[] = [];
  for (let i = 0; i < lines.length - 1; i += 1) {
    for (let j = i + 1; j < lines.length; j += 1) {
      const point = intersectLines(lines[i], lines[j]);
      if (point && Number.isFinite(point[0]) && Number.isFinite(point[1])) {
        intersections.push(point);
      }
    }
  }

  if (!intersections.length) return undefined;
  return [median(intersections.map((point) => point[0])), median(intersections.map((point) => point[1]))];
}

function solvedVanishingPoints(solvePayload: any): Partial<Record<AxisName, Point>> {
  const points = Array.isArray(solvePayload?.vanishing_points) ? solvePayload.vanishing_points : [];
  const resolved: Partial<Record<AxisName, Point>> = {};

  for (const point of points) {
    const position = point?.position_px;
    if (!Array.isArray(position) || position.length < 2) continue;
    const axis = axisFromLabel(point?.direction_label);
    if (axis) resolved[axis] = [Number(position[0]), Number(position[1])];
  }
  return resolved;
}

function axisFromLabel(label: unknown): AxisName | undefined {
  const value = String(label ?? "").toLowerCase();
  if (["left", "vp1", "x", "horizontal_left"].includes(value)) return "x";
  if (["vertical", "vp3", "y"].includes(value)) return "y";
  if (["right", "vp2", "z", "horizontal_right"].includes(value)) return "z";
  return undefined;
}

function intersectLines(first: Segment, second: Segment): Point | undefined {
  const [a1, b1, c1] = lineCoefficients(first);
  const [a2, b2, c2] = lineCoefficients(second);
  const determinant = a1 * b2 - a2 * b1;
  if (Math.abs(determinant) < EPSILON) return undefined;
  return [(b1 * c2 - b2 * c1) / determinant, (c1 * a2 - c2 * a1) / determinant];
}

function lineCoefficients(line: Segment): [number, number, number] {
  const [[x1, y1], [x2, y2]] = line;
  const a = y1 - y2;
  const b = x2 - x1;
  const c = x1 * y2 - x2 * y1;
  return [a, b, c];
}

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2) return sorted[middle];
  return (sorted[middle - 1] + sorted[middle]) / 2;
}

function gridLinesForAxis(
  vanishingPoint: Point | undefined,
  width: number,
  height: number,
  axis: AxisName
): Segment[] {
  if (!vanishingPoint || width <= 0 || height <= 0) return fallbackGrid(width, height, axis);

  const anchors = boundaryAnchors(width, height, axis);
  return anchors
    .map((anchor) => clipLineToRect(vanishingPoint, anchor, width, height))
    .filter((line): line is Segment => Boolean(line));
}

function fallbackGrid(width: number, height: number, axis: AxisName): Segment[] {
  if (width <= 0 || height <= 0) return [];
  const count = 12;
  const lines: Segment[] = [];
  for (let index = 1; index < count; index += 1) {
    const t = index / count;
    if (axis === "y") {
      const x = width * t;
      lines.push([[x, 0], [x, height]]);
    } else {
      const y = height * t;
      lines.push([[0, y], [width, y]]);
    }
  }
  return lines;
}

function boundaryAnchors(width: number, height: number, axis: AxisName): Point[] {
  const count = 14;
  const anchors: Point[] = [];
  for (let index = 0; index <= count; index += 1) {
    const t = index / count;
    if (axis === "y") {
      anchors.push([width * t, 0], [width * t, height]);
    } else {
      anchors.push([0, height * t], [width, height * t]);
    }
  }
  return anchors;
}

function clipLineToRect(first: Point, second: Point, width: number, height: number): Segment | undefined {
  const [x1, y1] = first;
  const [x2, y2] = second;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const points: Point[] = [];

  if (Math.abs(dx) > EPSILON) {
    const tLeft = (0 - x1) / dx;
    const yLeft = y1 + tLeft * dy;
    if (yLeft >= -EPSILON && yLeft <= height + EPSILON) points.push([0, clamp(yLeft, 0, height)]);

    const tRight = (width - x1) / dx;
    const yRight = y1 + tRight * dy;
    if (yRight >= -EPSILON && yRight <= height + EPSILON) points.push([width, clamp(yRight, 0, height)]);
  }

  if (Math.abs(dy) > EPSILON) {
    const tTop = (0 - y1) / dy;
    const xTop = x1 + tTop * dx;
    if (xTop >= -EPSILON && xTop <= width + EPSILON) points.push([clamp(xTop, 0, width), 0]);

    const tBottom = (height - y1) / dy;
    const xBottom = x1 + tBottom * dx;
    if (xBottom >= -EPSILON && xBottom <= width + EPSILON) points.push([clamp(xBottom, 0, width), height]);
  }

  const unique = dedupePoints(points);
  if (unique.length < 2) return undefined;
  return [unique[0], unique[1]];
}

function dedupePoints(points: Point[]): Point[] {
  const unique: Point[] = [];
  for (const point of points) {
    if (!unique.some((item) => distanceSquared(item, point) < 0.25)) {
      unique.push(point);
    }
  }
  return unique;
}

function distanceSquared(first: Point, second: Point) {
  return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}
