"""Record3D (`.r3d`) capture importer — measured iPhone/iPad ARKit geometry.

Record3D writes an ARKit session to a `.r3d` file: a ZIP holding a JSON
``metadata`` blob plus a per-frame ``rgbd/`` folder of JPEG colour, LZFSE-
compressed depth, and LZFSE-compressed confidence. That bundle carries the two
things Atlas otherwise has to *estimate* from a single photograph:

  * **Calibrated intrinsics** — Apple's factory per-lens calibration, not a
    GeoCalib/vanishing-point inference.
  * **A metric, gravity-aligned pose** — ``ARCamera.transform``, in metres.

so this importer is a THIRD solve source alongside vanishing points and the
learned prior, and the only one whose numbers are measured rather than guessed.

Why the axis conversion here is a no-op
---------------------------------------
ARKit's camera basis is x-right / y-up / z-back (the camera looks down its own
**-Z**), its world is right-handed Y-up with +Y opposed to gravity, and its
units are metres. That is, axis for axis, *already* Atlas's convention — see
``core.camera_math.look_at_view_matrix`` ("camera looks along -Z in camera
space, camera space is x-right / y-up / z-back") and the back-projection in
``core.relief_mesh`` (``x=(u-cx)/fx*d``, ``y=-(v-cy)/fy*d``, ``z=-d``). So the
quaternion becomes ``camera_rotation_matrix`` directly, with no basis change.
This is a genuine coincidence of conventions, not an accident of this module,
and it is pinned by ``tests/test_record3d_importer.py`` so a future edit that
"helpfully" inserts a flip fails loudly.

Why the -Z canonicalization is deliberately NOT applied
-------------------------------------------------------
``solver._face_camera_toward_negative_z`` exists because yaw is unobservable
from a single still, so Atlas picks a convention. Here yaw is **measured** by
the device's IMU and visual-inertial odometry, and every frame of a capture
shares one world origin. Applying the canonical flip would rotate that measured
yaw into fiction and de-register frames from each other. Measured poses pass
through untouched.

Scope / status
--------------
Prototype. The `.r3d` container is not a published spec; the layout below is
what Record3D writes in practice, and every field this module cannot verify is
reported through :attr:`Record3DCapture.warnings` rather than silently assumed.
Depth decode needs the optional ``[record3d]`` extra (``pyliblzfse``); an
extracted capture with raw ``.npy``/float32 depth reads with no extra at all.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas_camera.core.confidence import ConfidenceModel
from atlas_camera.core.projection_scene import create_default_projection_scene
from atlas_camera.core.schema import (
    AtlasCamera,
    AtlasIntrinsics,
    AtlasSolve,
    Matrix4,
)

# Shared Atlas-convention adapter math: turns a cam->world 4x4 into the full
# AtlasExtrinsics (position + rotation3 + world + view). Same job on the way in
# from ARKit as it does on the way in from USD, so it is imported rather than
# duplicated.
from atlas_camera.importers.usd_camera_loader import _extrinsics_from_world_matrix

__all__ = [
    "Record3DCapture",
    "Record3DFrame",
    "Record3DError",
    "SOURCE_METHOD",
    "quaternion_to_rotation_matrix",
]

#: ``AtlasSolve.source_method`` stamped on every solve this importer builds.
#: ``source_method`` is a free-form string (not a registered combo), so this is
#: additive and cannot disturb any saved workflow.
SOURCE_METHOD = "measured_arkit_record3d"

#: Record3D's confidence codes, straight from ARKit's ARConfidenceLevel.
CONFIDENCE_LOW, CONFIDENCE_MEDIUM, CONFIDENCE_HIGH = 0, 1, 2

_METADATA_NAMES = ("metadata", "metadata.json")
_DEPTH_SUFFIXES = (".depth", ".npy", ".bin")


class Record3DError(RuntimeError):
    """Raised when a capture cannot be read or is missing required fields."""


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only without numpy
        raise Record3DError(
            "Reading Record3D depth requires numpy. Install with: pip install -e .[record3d]"
        ) from exc
    return np


def quaternion_to_rotation_matrix(
    qx: float, qy: float, qz: float, qw: float
) -> tuple[tuple[float, float, float], ...]:
    """Unit quaternion -> right-handed 3x3 rotation, COLUMNS = rotated basis axes.

    Column-major-basis is Atlas's ``camera_rotation_matrix`` convention (see
    ``camera_math.look_at_view_matrix``, whose ``rotation3`` columns are the
    camera's x/y/z axes in world space). Record3D stores ARKit poses as
    ``[qx, qy, qz, qw, tx, ty, tz]`` — scalar LAST.
    """
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm < 1e-12:
        raise Record3DError("Degenerate (zero-norm) pose quaternion.")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    return (
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
        (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
        (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
    )


def _world_matrix_from_pose(pose: list[float]) -> Matrix4:
    """ARKit ``[qx,qy,qz,qw,tx,ty,tz]`` -> Atlas cam->world 4x4.

    Atlas's Matrix4 is row-major with the translation in the last COLUMN
    (column-vector convention, ``p' = M @ p``) — matching
    ``camera_math.look_at_view_matrix``'s ``world``. No axis remap: see the
    module docstring on why ARKit's basis is already Atlas's.
    """
    if len(pose) < 7:
        raise Record3DError(
            f"Pose entry needs 7 values [qx,qy,qz,qw,tx,ty,tz], got {len(pose)}."
        )
    qx, qy, qz, qw, tx, ty, tz = (float(v) for v in pose[:7])
    r = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    return (
        (r[0][0], r[0][1], r[0][2], tx),
        (r[1][0], r[1][1], r[1][2], ty),
        (r[2][0], r[2][1], r[2][2], tz),
        (0.0, 0.0, 0.0, 1.0),
    )


def _intrinsics_from_K(K: list[float]) -> tuple[float, float, float, float]:
    """Record3D's flat 9-value ``K`` -> (fx, fy, cx, cy).

    Record3D serializes K **column-major**, i.e.
    ``[fx, 0, 0,  0, fy, 0,  cx, cy, 1]``. Read row-major by mistake and you get
    cx/cy of 0 and a principal point in the image corner, which is why this is
    asserted rather than inferred: a column-major K has its last-but-two and
    last-but-one entries non-zero and its 2nd/3rd entries zero.
    """
    if len(K) != 9:
        raise Record3DError(f"Intrinsic matrix K needs 9 values, got {len(K)}.")
    vals = [float(v) for v in K]
    fx, fy, cx, cy = vals[0], vals[4], vals[6], vals[7]

    # Guard the transpose ambiguity explicitly instead of trusting the layout.
    if (abs(cx) < 1e-6 and abs(cy) < 1e-6) and (abs(vals[2]) > 1.0 or abs(vals[5]) > 1.0):
        # Looks row-major after all — principal point sits at [2] and [5].
        cx, cy = vals[2], vals[5]
    if fx <= 0 or fy <= 0:
        raise Record3DError(f"Non-positive focal length in K: fx={fx}, fy={fy}.")
    return fx, fy, cx, cy


def _decompress_lzfse(payload: bytes) -> bytes:
    try:
        import liblzfse  # type: ignore
    except ImportError as exc:
        raise Record3DError(
            "Record3D `.depth`/`.conf` frames are LZFSE-compressed. Install the "
            "decoder with: pip install -e .[record3d]  (provides pyliblzfse). "
            "Alternatively export the capture already extracted."
        ) from exc
    return liblzfse.decompress(payload)


def _decompress_lzfse_or_explain(payload: bytes, raw_failure: Record3DError) -> bytes:
    """LZFSE-decompress, or explain BOTH candidate causes when the decoder is absent.

    A buffer that failed a raw read is either compressed or malformed. If
    pyliblzfse is not installed we genuinely cannot distinguish the two, and
    blaming only the missing dependency sends anyone with a truncated capture
    on a long detour.
    """
    try:
        return _decompress_lzfse(payload)
    except Record3DError as lzfse_exc:
        raise Record3DError(f"{raw_failure} Nor could it be decompressed: {lzfse_exc}") from raw_failure


def _depth_from_buffer(raw: bytes, width: int, height: int, np: Any) -> Any:
    """Decode a raw depth buffer, inferring float32 vs float16 from its length.

    Record3D writes float32 for LiDAR captures and float16 for some compressed
    exports; the pixel count is known, so the element size is determined rather
    than guessed.
    """
    n = width * height
    if len(raw) == n * 4:
        arr = np.frombuffer(raw, dtype=np.float32)
    elif len(raw) == n * 2:
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        raise Record3DError(
            f"Depth buffer is {len(raw)} bytes — not {n}x4 (float32) or {n}x2 "
            f"(float16) for a {width}x{height} frame. Wrong dw/dh in metadata, "
            "or an export format this importer does not yet read."
        )
    return np.ascontiguousarray(arr.reshape(height, width))


@dataclass(slots=True)
class Record3DFrame:
    """One frame: colour bytes, metric depth, confidence, and a measured pose."""

    index: int
    rgb_jpeg: bytes | None
    depth: Any  # numpy (dh, dw) float32 metres, or None
    confidence: Any  # numpy (dh, dw) uint8 in {0,1,2}, or None
    camera_world_matrix: Matrix4
    intrinsics: AtlasIntrinsics
    depth_size: tuple[int, int]  # (width, height) of the NATIVE depth buffer


@dataclass(slots=True)
class Record3DCapture:
    """An opened `.r3d` capture (or an extracted capture directory)."""

    path: Path
    metadata: dict[str, Any]
    rgb_size: tuple[int, int]
    depth_size: tuple[int, int]
    poses: list[list[float]]
    fps: float
    warnings: list[str] = field(default_factory=list)
    _zip: Any = None
    _root: Path | None = None

    # ---------------------------------------------------------------- opening

    @classmethod
    def open(cls, path: str | Path) -> "Record3DCapture":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)

        warnings: list[str] = []
        archive = None
        root = None

        if source.is_dir():
            root = source
            meta_bytes = None
            for name in _METADATA_NAMES:
                candidate = source / name
                if candidate.is_file():
                    meta_bytes = candidate.read_bytes()
                    break
            if meta_bytes is None:
                raise Record3DError(
                    f"No `metadata` file in {source} — is this an extracted .r3d capture?"
                )
        else:
            if not zipfile.is_zipfile(source):
                raise Record3DError(
                    f"{source} is not a ZIP archive. A Record3D `.r3d` file is a ZIP; "
                    "point this at the .r3d itself or at an extracted capture folder."
                )
            archive = zipfile.ZipFile(source)
            names = set(archive.namelist())
            meta_bytes = None
            for name in _METADATA_NAMES:
                if name in names:
                    meta_bytes = archive.read(name)
                    break
            if meta_bytes is None:
                raise Record3DError(
                    f"No `metadata` entry inside {source}. Entries: "
                    f"{sorted(names)[:8]}{'...' if len(names) > 8 else ''}"
                )

        try:
            metadata = json.loads(meta_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Record3DError(f"Capture `metadata` is not valid JSON: {exc}") from exc

        rgb_w = int(metadata.get("w") or 0)
        rgb_h = int(metadata.get("h") or 0)
        depth_w = int(metadata.get("dw") or 0)
        depth_h = int(metadata.get("dh") or 0)
        if rgb_w <= 0 or rgb_h <= 0:
            raise Record3DError("Capture metadata has no usable RGB size (`w`/`h`).")
        if depth_w <= 0 or depth_h <= 0:
            warnings.append(
                "Capture metadata has no depth size (`dw`/`dh`) — depth frames "
                "cannot be decoded; camera solve only."
            )

        poses = [list(p) for p in (metadata.get("poses") or [])]
        init_pose = metadata.get("initPose")
        if not poses and init_pose:
            poses = [list(init_pose)]
            warnings.append("No `poses` array — falling back to the single `initPose`.")
        if not poses:
            raise Record3DError(
                "Capture metadata carries no camera poses (`poses`/`initPose`); "
                "there is no measured extrinsic to import."
            )

        # Depth resolution is the honest headline number: LiDAR sceneDepth is
        # 256x192 regardless of how large the colour frame is. Say so up front so
        # nobody reads a 4K projection as 4K geometry.
        if depth_w > 0 and rgb_w > 0:
            ratio = (rgb_w * rgb_h) / float(depth_w * depth_h)
            if ratio >= 4.0:
                warnings.append(
                    f"Depth is {depth_w}x{depth_h} against a {rgb_w}x{rgb_h} colour "
                    f"frame ({ratio:.0f}x fewer samples). This depth is a METRIC "
                    "anchor, not a high-resolution surface — pair it with a "
                    "monocular model via AtlasDepthCombine for detail."
                )

        if rgb_h > rgb_w:
            warnings.append(
                "Colour frame is portrait. Record3D buffers are natively landscape; "
                "verify the horizon in AtlasBlockoutViewport before trusting the roll."
            )

        return cls(
            path=source,
            metadata=metadata,
            rgb_size=(rgb_w, rgb_h),
            depth_size=(depth_w, depth_h),
            poses=poses,
            fps=float(metadata.get("fps") or 0.0),
            warnings=warnings,
            _zip=archive,
            _root=root,
        )

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __enter__(self) -> "Record3DCapture":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------- properties

    @property
    def n_frames(self) -> int:
        return len(self.poses)

    @property
    def device_hint(self) -> str:
        """Best-effort capture-device description for provenance."""
        dt = self.metadata.get("deviceType")
        depth_w, depth_h = self.depth_size
        if (depth_w, depth_h) == (256, 192):
            sensor = "rear LiDAR (ARKit sceneDepth)"
        elif depth_w and depth_h and depth_w * depth_h >= 640 * 480 * 0.5:
            sensor = "TrueDepth front camera"
        elif depth_w:
            sensor = f"{depth_w}x{depth_h} depth"
        else:
            sensor = "no depth"
        return f"{dt if dt is not None else 'iOS device'} / {sensor}"

    # ------------------------------------------------------------- frame read

    def _read_member(self, *candidates: str) -> bytes | None:
        for name in candidates:
            if self._zip is not None:
                try:
                    return self._zip.read(name)
                except KeyError:
                    continue
            elif self._root is not None:
                candidate = self._root / name
                if candidate.is_file():
                    return candidate.read_bytes()
        return None

    def _frame_intrinsics(self, index: int) -> AtlasIntrinsics:
        """Intrinsics for a frame, preferring per-frame coefficients when present."""
        rgb_w, rgb_h = self.rgb_size
        per_frame = self.metadata.get("perFrameIntrinsicCoeffs") or []
        if index < len(per_frame) and len(per_frame[index]) >= 4:
            fx, fy, cx, cy = (float(v) for v in per_frame[index][:4])
        else:
            K = self.metadata.get("K")
            if not K:
                raise Record3DError("Capture metadata has neither `K` nor per-frame intrinsics.")
            fx, fy, cx, cy = _intrinsics_from_K(list(K))

        # Some exports write K at the DEPTH buffer's scale rather than the colour
        # frame's. A principal point sitting in the left quarter of a frame whose
        # depth buffer is genuinely smaller is the tell — rescale, rather than
        # emit a solve whose principal point is near the image corner.
        depth_w = self.depth_size[0]
        if depth_w and depth_w < rgb_w and cx < rgb_w * 0.25:
            scale = rgb_w / float(depth_w)
            fx, fy, cx, cy = fx * scale, fy * scale, cx * scale, cy * scale

        return AtlasIntrinsics(
            image_width=rgb_w,
            image_height=rgb_h,
            focal_length_mm=None,  # measured in pixels; no sensor-mm round trip needed
            sensor_width_mm=36.0,
            sensor_height_mm=36.0 * (rgb_h / float(rgb_w)),
            principal_point_px=(cx, cy),
            fx_px=fx,
            fy_px=fy,
            cx_px=cx,
            cy_px=cy,
            lens_model="pinhole",
        )

    def frame(self, index: int = 0, *, load_depth: bool = True) -> Record3DFrame:
        """Read one frame. ``load_depth=False`` skips the LZFSE dependency."""
        if not 0 <= index < self.n_frames:
            raise IndexError(f"Frame {index} out of range (capture has {self.n_frames}).")

        rgb = self._read_member(f"rgbd/{index}.jpg", f"rgbd/{index}.jpeg", f"{index}.jpg")
        depth_arr = None
        conf_arr = None
        depth_w, depth_h = self.depth_size

        if load_depth and depth_w > 0 and depth_h > 0:
            np = _require_numpy()
            raw = self._read_member(*(f"rgbd/{index}{sfx}" for sfx in _DEPTH_SUFFIXES))
            if raw is not None:
                depth_arr = self._decode_depth(raw, depth_w, depth_h, np)
            conf_raw = self._read_member(f"rgbd/{index}.conf")
            if conf_raw is not None:
                conf_arr = self._decode_confidence(conf_raw, depth_w, depth_h, np)

        return Record3DFrame(
            index=index,
            rgb_jpeg=rgb,
            depth=depth_arr,
            confidence=conf_arr,
            camera_world_matrix=_world_matrix_from_pose(self.poses[index]),
            intrinsics=self._frame_intrinsics(index),
            depth_size=(depth_w, depth_h),
        )

    @staticmethod
    def _decode_depth(raw: bytes, width: int, height: int, np: Any) -> Any:
        # A bare .npy round-trips with no LZFSE dependency at all.
        if raw[:6] == b"\x93NUMPY":
            import io

            return np.ascontiguousarray(np.load(io.BytesIO(raw)).astype(np.float32))
        try:
            return _depth_from_buffer(raw, width, height, np)
        except Record3DError as raw_exc:
            # The buffer is either LZFSE-compressed (the normal case) or simply
            # the wrong size. Without the decoder installed we cannot tell which,
            # so report BOTH causes rather than blaming the missing dependency
            # for what may be a malformed capture.
            return _depth_from_buffer(
                _decompress_lzfse_or_explain(raw, raw_exc), width, height, np
            )

    @staticmethod
    def _decode_confidence(raw: bytes, width: int, height: int, np: Any) -> Any:
        n = width * height
        if len(raw) != n:
            size_exc = Record3DError(
                f"Confidence buffer is {len(raw)} bytes, expected {n} for {width}x{height}."
            )
            buf = _decompress_lzfse_or_explain(raw, size_exc)
            if len(buf) != n:
                raise size_exc
        else:
            buf = raw
        return np.frombuffer(buf, dtype=np.uint8).reshape(height, width).copy()

    # ------------------------------------------------------------ solve build

    def solve(self, index: int = 0, *, image_path: str | None = None) -> AtlasSolve:
        """Build an :class:`AtlasSolve` from a frame's MEASURED pose + intrinsics.

        Confidence is stamped high for ``focal``/``sensor``/``extrinsics``
        because those are read off the device rather than inferred — but
        ``scale`` is only as good as ARKit's drift, and ``horizon``/``vp*``
        stay at 0 because nothing here estimated them.
        """
        frame = self.frame(index, load_depth=False)
        extrinsics = _extrinsics_from_world_matrix(frame.camera_world_matrix)

        confidence = ConfidenceModel.for_latent_camera(
            global_score=0.9,
            overrides={
                "focal": 0.95,      # Apple factory per-lens calibration
                "sensor": 0.95,
                "extrinsics": 0.9,  # visual-inertial odometry, metres, gravity-aligned
                "scale": 0.85,      # metric by construction; degrades with VIO drift
                "depth": 0.7 if self.depth_size[0] else 0.0,
                "horizon": 0.0,     # not estimated — measured pose supersedes it
                "vp1": 0.0, "vp2": 0.0, "vp3": 0.0,
            },
        )

        camera = AtlasCamera(
            intrinsics=frame.intrinsics,
            extrinsics=extrinsics,
            name="atlas_record3d_camera",
            confidence=confidence,
            focal_length_inferred=False,
        )

        rgb_w, rgb_h = self.rgb_size
        return AtlasSolve(
            camera=camera,
            image_path=image_path or str(self.path),
            image_width=rgb_w,
            image_height=rgb_h,
            confidence=0.9,
            source_method=SOURCE_METHOD,
            known_intrinsics_used=True,
            projection_scene=create_default_projection_scene(),
            debug_metadata={
                "record3d_capture": str(self.path),
                "record3d_frame_index": index,
                "record3d_frame_count": self.n_frames,
                "record3d_fps": self.fps,
                "record3d_device": self.device_hint,
                "depth_width": self.depth_size[0],
                "depth_height": self.depth_size[1],
                "measured_pose": True,
                "canonical_negative_z_applied": False,
                "warnings": list(self.warnings),
            },
        )
