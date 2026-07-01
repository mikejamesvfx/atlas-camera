export type IntrinsicsLike = {
  image_width?: number;
  image_height?: number;
  focal_length_mm?: number | null;
  sensor_width_mm?: number | null;
  fx_px?: number | null;
  fy_px?: number | null;
  cx_px?: number | null;
  cy_px?: number | null;
};

export type ThreeCameraProjection = {
  fov: number;
  aspect: number;
  near: number;
  far: number;
  focalPx: number;
  principalPoint: [number, number];
};

export function atlasIntrinsicsToThreeCamera(intrinsics: IntrinsicsLike): ThreeCameraProjection {
  const width = Number(intrinsics.image_width || 1);
  const height = Number(intrinsics.image_height || 1);
  const fxFromMm = intrinsics.focal_length_mm && intrinsics.sensor_width_mm
    ? (Number(intrinsics.focal_length_mm) / Number(intrinsics.sensor_width_mm)) * width
    : undefined;
  const focalPx = Number(intrinsics.fy_px || intrinsics.fx_px || fxFromMm || width);
  const fov = radiansToDegrees(2 * Math.atan(height / (2 * Math.max(1e-6, focalPx))));
  return {
    fov,
    aspect: width / height,
    near: 0.01,
    far: 1000,
    focalPx,
    principalPoint: [
      Number(intrinsics.cx_px ?? width / 2),
      Number(intrinsics.cy_px ?? height / 2)
    ]
  };
}

export function atlasViewMatrixToWorldMatrix(viewMatrix: number[][]): number[] {
  const matrix = flattenMatrix4(viewMatrix);
  return invertMatrix4(matrix);
}

export function projectPointToImage(
  point: [number, number, number],
  viewMatrix: number[][],
  intrinsics: IntrinsicsLike
): [number, number] | null {
  const view = multiplyMatrix4Vector(flattenMatrix4(viewMatrix), [point[0], point[1], point[2], 1]);
  if (Math.abs(view[2]) < 1e-6) return null;
  const width = Number(intrinsics.image_width || 1);
  const height = Number(intrinsics.image_height || 1);
  const fx = Number(intrinsics.fx_px || width);
  const fy = Number(intrinsics.fy_px || fx);
  const cx = Number(intrinsics.cx_px ?? width / 2);
  const cy = Number(intrinsics.cy_px ?? height / 2);
  const z = -view[2];
  if (z <= 0) return null;
  return [
    cx + (view[0] * fx) / z,
    cy - (view[1] * fy) / z
  ];
}

export function proxyDimensionsLabel(scale: [number, number, number]) {
  return `${scale[0].toFixed(2)} x ${scale[1].toFixed(2)} x ${scale[2].toFixed(2)}m`;
}

function flattenMatrix4(matrix: number[][]): number[] {
  const fallback = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ];
  const rows = matrix.length >= 4 ? matrix : fallback;
  return rows.slice(0, 4).flatMap((row, rowIndex) => {
    const source = row.length >= 4 ? row : fallback[rowIndex];
    return source.slice(0, 4).map(Number);
  });
}

function multiplyMatrix4Vector(matrix: number[], vector: [number, number, number, number]) {
  return [
    matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2] + matrix[3] * vector[3],
    matrix[4] * vector[0] + matrix[5] * vector[1] + matrix[6] * vector[2] + matrix[7] * vector[3],
    matrix[8] * vector[0] + matrix[9] * vector[1] + matrix[10] * vector[2] + matrix[11] * vector[3],
    matrix[12] * vector[0] + matrix[13] * vector[1] + matrix[14] * vector[2] + matrix[15] * vector[3]
  ] as [number, number, number, number];
}

function invertMatrix4(matrix: number[]): number[] {
  const m = matrix;
  const inv = new Array(16);
  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
  const det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  if (Math.abs(det) < 1e-9) return [...matrix];
  return inv.map((value) => value / det);
}

function radiansToDegrees(value: number) {
  return (value * 180) / Math.PI;
}
