"""Validated packets shared by the offline synchronized camera-fusion pipeline.

Camera extrinsics map the common metric world to each camera. Capture timestamps
are camera clocks; capture + configured clock_offset_seconds is the local monotonic
timeline. Network arrival time is separate and never substitutes for capture time.
These packets contain detections, not ground-truth identities or inferred depths.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .camera_calibration import CameraCalibration


def validated_identifier(value, name):
    if (not isinstance(value, str) or not value.strip() or len(value) > 128
            or any(ord(character) < 32 for character in value)):
        raise ValueError(f"{name} must be a nonempty string of at most 128 characters")
    return value


def finite_scalar(value, name, *, minimum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}" if minimum is not None else f"{name} must be finite")
    return result


def immutable_array(value, name, shape=None, *, allow_nan=False):
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric coordinates") from exc
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if np.isinf(result).any() or (not allow_nan and not np.isfinite(result).all()):
        raise ValueError(f"{name} contains invalid coordinates")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CalibratedCamera:
    camera_id: str
    calibration: CameraCalibration
    rotation: np.ndarray
    translation_m: np.ndarray
    clock_offset_seconds: float = 0.0
    clock_uncertainty_seconds: float = 0.0

    def __post_init__(self):
        validated_identifier(self.camera_id, "camera_id")
        if not isinstance(self.calibration, CameraCalibration):
            raise ValueError("calibration must be a CameraCalibration")
        rotation = immutable_array(self.rotation, "rotation", (3, 3))
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7, rtol=0)):
            raise ValueError("rotation must be a proper orthonormal world-to-camera matrix")
        translation = np.asarray(self.translation_m)
        if translation.shape not in ((3,), (3, 1), (1, 3)):
            raise ValueError("translation_m must contain three coordinates")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation_m", immutable_array(translation.reshape(3), "translation_m", (3,)))
        object.__setattr__(self, "clock_offset_seconds", finite_scalar(self.clock_offset_seconds, "clock_offset_seconds"))
        object.__setattr__(self, "clock_uncertainty_seconds", finite_scalar(self.clock_uncertainty_seconds, "clock_uncertainty_seconds", minimum=0))


@dataclass(frozen=True)
class PersonObservation:
    local_track_id: str
    landmarks_px: np.ndarray
    confidence: np.ndarray
    identity_id: str | None = None
    identity_confidence: float = 0.0

    def __post_init__(self):
        validated_identifier(self.local_track_id, "local_track_id")
        points = immutable_array(self.landmarks_px, "landmarks_px", allow_nan=True)
        if points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 256:
            raise ValueError("landmarks_px must have shape (3..256, 2)")
        confidence = immutable_array(self.confidence, "confidence", (len(points),))
        if np.any(confidence < 0) or np.any(confidence > 1):
            raise ValueError("joint confidence must be in [0, 1]")
        if np.any(~np.isfinite(points).all(axis=1) & (confidence > 0)):
            raise ValueError("missing/NaN landmarks must have zero confidence")
        identity_confidence = finite_scalar(self.identity_confidence, "identity_confidence", minimum=0)
        if identity_confidence > 1:
            raise ValueError("identity_confidence must be in [0, 1]")
        if self.identity_id is not None:
            validated_identifier(self.identity_id, "identity_id")
        elif identity_confidence != 0:
            raise ValueError("identity confidence requires an identity_id")
        object.__setattr__(self, "landmarks_px", points)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "identity_confidence", identity_confidence)


@dataclass(frozen=True)
class CameraFrame:
    camera_id: str
    sequence: int
    captured_at_seconds: float
    received_at_seconds: float
    observations: tuple
    stream_epoch: str = "default"

    def __post_init__(self):
        validated_identifier(self.camera_id, "camera_id")
        validated_identifier(self.stream_epoch, "stream_epoch")
        if (not isinstance(self.sequence, Integral) or isinstance(self.sequence, (bool, np.bool_))
                or not 0 <= self.sequence <= 2**63 - 1):
            raise ValueError("sequence must be a nonnegative signed 64-bit integer")
        observations = tuple(self.observations)
        if len(observations) > 64 or any(not isinstance(item, PersonObservation) for item in observations):
            raise ValueError("a frame may contain at most 64 PersonObservation objects")
        if len({item.local_track_id for item in observations}) != len(observations):
            raise ValueError("local_track_id must be unique within each camera frame")
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "captured_at_seconds", finite_scalar(self.captured_at_seconds, "captured_at_seconds", minimum=0))
        object.__setattr__(self, "received_at_seconds", finite_scalar(self.received_at_seconds, "received_at_seconds", minimum=0))
        object.__setattr__(self, "observations", observations)


@dataclass(frozen=True)
class FrameBundle:
    bundle_id: str
    target_timestamp_seconds: float
    frames: tuple
    missing_cameras: tuple
    span_seconds: float

    def __post_init__(self):
        validated_identifier(self.bundle_id, "bundle_id")
        frames = tuple(self.frames)
        missing = tuple(self.missing_cameras)
        if not frames or any(not isinstance(frame, CameraFrame) for frame in frames):
            raise ValueError("a bundle requires at least one CameraFrame")
        if len({frame.camera_id for frame in frames}) != len(frames):
            raise ValueError("a bundle cannot reuse a camera")
        for camera_id in missing:
            validated_identifier(camera_id, "missing camera_id")
        if len(set(missing)) != len(missing) or set(missing) & {frame.camera_id for frame in frames}:
            raise ValueError("missing_cameras must be unique and absent from the bundle")
        object.__setattr__(self, "target_timestamp_seconds", finite_scalar(self.target_timestamp_seconds, "target_timestamp_seconds", minimum=0))
        object.__setattr__(self, "span_seconds", finite_scalar(self.span_seconds, "span_seconds", minimum=0))
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "missing_cameras", missing)
