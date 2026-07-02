declare module "three" {
  const THREE: any;
  export = THREE;
}

declare module "three/examples/jsm/controls/OrbitControls.js" {
  export class OrbitControls {
    constructor(camera: any, domElement: HTMLElement | null);
    target: any;
    enabled: boolean;
    enableRotate: boolean;
    enablePan: boolean;
    enableZoom: boolean;
    enableDamping: boolean;
    dampingFactor: number;
    autoRotate: boolean;
    autoRotateSpeed: number;
    minDistance: number;
    maxDistance: number;
    screenSpacePanning: boolean;
    update(): void;
    reset(): void;
    dispose(): void;
  }
}
