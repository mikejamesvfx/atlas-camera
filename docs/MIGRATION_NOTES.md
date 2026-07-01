# Migration Notes From Maya2Comfy

Source prototype inspected:

```text
C:/Users/miike/Documents/Maya2Comfy
```

Candidate modules found:

- `camera_estimator.py`
  - `VanishingPointDetector`
  - `CameraFromVanishingPoints`
  - `CameraEstimatorViz`
  - `AutoCameraEstimator`
  - `ExportCameraJSON`
- `maya_camera_converter.py`
  - `MayaCameraData`
  - `MayaFilmFitCalculator`
  - `MayaCameraConverter`
  - `CameraMatrixValidator`
- `maya_usd_camera_loader.py`
  - `USDCameraReader`
  - `MayaUSDCameraLoader`
- `camera_export_formats.py`
  - `CameraExportData`
  - `USDExporter`
  - `BatchCameraExporter`
- `utils/validators.py`
  - path and JSON validation helpers.

The first Atlas pass copied no Maya2Comfy source code. This pass ported and
adapted the vanishing-point detection and camera-from-vanishing-points logic
from `camera_estimator.py` into Atlas core modules.

Concepts adapted into a cleaner structure:

- `MayaCameraData` -> `AtlasCamera`, `AtlasIntrinsics`, `AtlasExtrinsics`
- `AutoCameraEstimator` -> `StillImageCameraEstimator`
- `ExportCameraJSON` -> `atlas_camera.core.io`
- `MayaUSDCameraLoader` -> `USDCameraLoader`
- Maya scene output -> `atlas_camera.exporters.maya_exporter`
- Comfy custom nodes -> `atlas_camera.comfy.nodes`
- `VanishingPointDetector` -> `atlas_camera.core.vanishing_points.VanishingPointDetector`
- `CameraFromVanishingPoints` -> `atlas_camera.core.solver.CameraFromVanishingPoints`

Next migration target:

1. Improve line grouping and confidence scoring on real image fixtures.
2. Port validation tests without requiring generated Maya scenes.
3. Add metric depth fitting from recorded scale and object-height references.
4. Expand review-package DCC outputs beyond the Maya script builder.
