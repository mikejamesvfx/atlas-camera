import { describe, expect, it } from "vitest";
import {
  atlasIntrinsicsToThreeCamera,
  atlasViewMatrixToWorldMatrix,
  projectPointToImage
} from "./viewport3dMath";

describe("viewport3dMath", () => {
  it("converts Atlas intrinsics to a Three.js perspective camera projection", () => {
    const projection = atlasIntrinsicsToThreeCamera({
      image_width: 1280,
      image_height: 720,
      fx_px: 800,
      fy_px: 800,
      cx_px: 641,
      cy_px: 361
    });

    expect(projection.aspect).toBeCloseTo(1280 / 720, 6);
    expect(projection.fov).toBeCloseTo(48.455, 3);
    expect(projection.focalPx).toBe(800);
    expect(projection.principalPoint).toEqual([641, 361]);
  });

  it("derives focal pixels from focal length and sensor width when pixel focal is absent", () => {
    const projection = atlasIntrinsicsToThreeCamera({
      image_width: 1920,
      image_height: 1080,
      focal_length_mm: 35,
      sensor_width_mm: 36
    });

    expect(projection.focalPx).toBeCloseTo(1866.667, 3);
    expect(projection.fov).toBeCloseTo(32.269, 3);
  });

  it("inverts row-major Atlas view matrices into world matrices", () => {
    const world = atlasViewMatrixToWorldMatrix([
      [1, 0, 0, -2],
      [0, 1, 0, -3],
      [0, 0, 1, -4],
      [0, 0, 0, 1]
    ]);

    expect(world).toEqual([
      1, 0, 0, 2,
      0, 1, 0, 3,
      0, 0, 1, 4,
      0, 0, 0, 1
    ]);
  });

  it("projects 3D points into image pixels with Atlas camera coordinates", () => {
    const pixel = projectPointToImage(
      [1, 2, -10],
      [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ],
      {
        image_width: 1280,
        image_height: 720,
        fx_px: 1000,
        fy_px: 1000,
        cx_px: 640,
        cy_px: 360
      }
    );

    expect(pixel).toEqual([740, 160]);
  });
});
