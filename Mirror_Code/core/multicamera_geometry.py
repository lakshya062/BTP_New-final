"""Conservative, distortion-aware multi-view geometry for synchronized observations.

Extrinsics transform world metres into camera coordinates: ``R @ X + t``.
This module does not infer a missing depth from a single camera or extrapolate an
occluded joint. Each returned point requires two independent optical centres,
positive camera depth, adequate parallax, and a consistent reprojection consensus.
Geometry alone cannot establish a person's identity; epipolar-compatible people
can still be indistinguishable and association must handle that ambiguity.
"""

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral

import cv2
import numpy as np

from .camera_calibration import CalibrationError, CameraCalibration


MAX_CAMERAS = 32
MAX_JOINTS = 256


@dataclass(frozen=True)
class JointEstimate:
    position_m: np.ndarray | None = None
    supporting_cameras: tuple[str, ...] = ()
    reason: str = "insufficient_views"
    reprojection_error_px: float | None = None
    ray_angle_degrees: float | None = None

    def __post_init__(self):
        if self.position_m is not None:
            position = np.array(self.position_m, dtype=np.float64, copy=True)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise CalibrationError("joint position must contain three finite coordinates")
            position.setflags(write=False)
            object.__setattr__(self, "position_m", position)


@dataclass(frozen=True)
class PairCompatibility:
    compatible: bool
    shared_joints: int
    median_epipolar_error_px: float | None
    reason: str


@dataclass(frozen=True)
class _View:
    camera_id: str
    camera: object
    pixel: np.ndarray
    confidence: float
    normalized: np.ndarray
    ray: np.ndarray
    center: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray


def _finite(value, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise CalibrationError(f"{name} must contain finite numbers") from exc
    if not np.isfinite(array).all():
        raise CalibrationError(f"{name} must contain finite numbers")
    return array


def _scalar(value, name, lower, upper, inclusive_lower=True):
    if isinstance(value, (bool, np.bool_)):
        raise CalibrationError(f"{name} must be a finite scalar")
    result = _finite(value, name)
    if result.ndim != 0 or result > upper or (result < lower if inclusive_lower else result <= lower):
        raise CalibrationError(f"{name} is outside its supported range")
    return float(result)


def _limits(min_confidence, max_reprojection_error_px, min_ray_angle_degrees):
    angle = _scalar(min_ray_angle_degrees, "min_ray_angle_degrees", 0, 90, False)
    if angle >= 90:
        raise CalibrationError("min_ray_angle_degrees must be less than 90")
    return (
        _scalar(min_confidence, "min_confidence", 0, 1),
        _scalar(max_reprojection_error_px, "max_reprojection_error_px", 0, 1000, False),
        angle,
    )


def _camera(camera):
    try:
        calibration = camera.calibration
        rotation = _finite(camera.rotation, "rotation")
        translation = _finite(camera.translation_m, "translation_m")
    except AttributeError as exc:
        raise CalibrationError("camera requires calibration and world-to-camera extrinsics") from exc
    if not isinstance(calibration, CameraCalibration):
        raise CalibrationError("camera calibration must be CameraCalibration")
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7, rtol=0):
        raise CalibrationError("camera rotation must be a proper orthonormal 3x3 matrix")
    if translation.shape != (3,) or np.max(np.abs(translation)) > 1e6:
        raise CalibrationError("camera translation_m must be a finite three-vector within 1000 km")
    return rotation, translation


def _prepare(observations, cameras, minimum):
    if not hasattr(observations, "items") or not hasattr(cameras, "items"):
        raise CalibrationError("observations and cameras must be mappings")
    if len(cameras) > MAX_CAMERAS or len(observations) > MAX_CAMERAS:
        raise CalibrationError(f"at most {MAX_CAMERAS} cameras are supported per fusion")
    if any(not isinstance(key, str) or not key for key in observations):
        raise CalibrationError("camera ids must be nonempty strings")
    views = []
    for camera_id, observation in sorted(observations.items()):
        if camera_id not in cameras:
            raise CalibrationError(f"unknown camera: {camera_id}")
        try:
            pixel, confidence = observation
        except (ValueError, TypeError) as exc:
            raise CalibrationError("each joint observation must contain pixel and confidence") from exc
        confidence = _scalar(confidence, "confidence", 0, 1)
        if confidence < minimum or confidence <= 0:
            continue  # Missing/occluded coordinates are deliberately never used.
        rotation, translation = _camera(cameras[camera_id])
        pixel = _finite(pixel, "pixel")
        calibration = cameras[camera_id].calibration
        if pixel.shape != (2,) or np.any(pixel < 0) or np.any(pixel >= calibration.image_size):
            raise CalibrationError("visible pixel must lie inside its calibrated image")
        try:
            normalized = cv2.undistortPointsIter(
                pixel.reshape(1, 1, 2), calibration.camera_matrix,
                calibration.distortion_coefficients, None, None,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-12),
            )[0, 0]
        except cv2.error as exc:
            raise CalibrationError("OpenCV could not undistort the observation") from exc
        if not np.isfinite(normalized).all():
            raise CalibrationError("observation cannot be undistorted to a finite ray")
        ray = rotation.T @ np.r_[normalized, 1.0]
        ray /= np.linalg.norm(ray)
        views.append(_View(camera_id, cameras[camera_id], pixel, confidence,
                           normalized, ray, -rotation.T @ translation, rotation, translation))
    return views


def _angle(first, second):
    # Acute angle also treats antiparallel rays as degenerate for depth.
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(first.ray, second.ray)),
                                      abs(np.dot(first.ray, second.ray)))))


def _max_angle(views):
    return max((_angle(a, b) for a, b in combinations(views, 2)
                if np.linalg.norm(a.center - b.center) > 1e-6), default=0.0)


def _independent_centers(views):
    centers = []
    for view in views:
        if all(np.linalg.norm(view.center - center) > 1e-6 for center in centers):
            centers.append(view.center)
    return len(centers)


def _triangulate(views):
    rows = []
    for view in views:
        projection = np.column_stack((view.rotation, view.translation))
        x, y = view.normalized
        weight = np.sqrt(view.confidence)
        rows.extend((weight * (x * projection[2] - projection[0]),
                     weight * (y * projection[2] - projection[1])))
    try:
        homogeneous = np.linalg.svd(np.asarray(rows), full_matrices=False)[2][-1]
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(homogeneous).all() or abs(homogeneous[3]) <= 1e-10:
        return None
    point = homogeneous[:3] / homogeneous[3]
    if not np.isfinite(point).all() or np.max(np.abs(point)) > 1e6:
        return None
    return point


def _error(point, view):
    if (view.rotation @ point + view.translation)[2] <= 1e-8:
        return np.inf
    calibration = view.camera.calibration
    try:
        projected = cv2.projectPoints(point.reshape(1, 3), cv2.Rodrigues(view.rotation)[0],
                                      view.translation, calibration.camera_matrix,
                                      calibration.distortion_coefficients)[0][0, 0]
    except cv2.error:
        return np.inf
    return float(np.linalg.norm(projected - view.pixel)) if np.isfinite(projected).all() else np.inf


def fuse_joint(observations, cameras, min_confidence=0.5,
               max_reprojection_error_px=3.0, min_ray_angle_degrees=1.0):
    """Triangulate one joint using deterministic pair-hypothesis consensus.

    Observations map camera ids to ``(pixel_xy, confidence)``. Zero/low-confidence
    samples are ignored. Invalid visible inputs raise ``CalibrationError``;
    valid but unobservable/inconsistent inputs return an empty ``JointEstimate``.
    Equal-size consensus sets yielding materially different positions fail closed.
    """
    minimum, limit, min_angle = _limits(min_confidence, max_reprojection_error_px,
                                       min_ray_angle_degrees)
    views = _prepare(observations, cameras, minimum)
    if len(views) < 2 or _independent_centers(views) < 2:
        return JointEstimate(reason="insufficient_views")
    if _max_angle(views) < min_angle:
        return JointEstimate(reason="insufficient_parallax")
    candidates = {}
    for first, second in combinations(views, 2):
        if np.linalg.norm(first.center - second.center) <= 1e-6 or _angle(first, second) < min_angle:
            continue
        point = _triangulate((first, second))
        if point is None:
            continue
        inliers = tuple(view for view in views if _error(point, view) <= limit)
        # Fit the consensus, then recheck membership instead of blindly accepting
        # a low pair residual. This prevents a high-confidence bad view dominating.
        for _ in range(3):
            if _independent_centers(inliers) < 2 or _max_angle(inliers) < min_angle:
                break
            refined = _triangulate(inliers)
            if refined is None:
                break
            updated = tuple(view for view in views if _error(refined, view) <= limit)
            point = refined
            if tuple(view.camera_id for view in updated) == tuple(view.camera_id for view in inliers):
                inliers = updated
                break
            inliers = updated
        if _independent_centers(inliers) < 2 or _max_angle(inliers) < min_angle:
            continue
        errors = np.array([_error(point, view) for view in inliers])
        if not np.isfinite(errors).all() or np.any(errors > limit):
            continue
        support = tuple(view.camera_id for view in inliers)
        candidates[support] = (point, inliers, float(np.sqrt(np.mean(errors ** 2))))
    if not candidates:
        return JointEstimate(reason="inconsistent_geometry")
    best_count = max(_independent_centers(value[1]) for value in candidates.values())
    best = [value for _, value in sorted(candidates.items())
            if _independent_centers(value[1]) == best_count]
    best.sort(key=lambda value: value[2])
    point, inliers, error = best[0]
    tolerance_m = max(0.01, 0.005 * min(np.linalg.norm(point - view.center) for view in inliers))
    if any(np.linalg.norm(other[0] - point) > tolerance_m for other in best[1:]):
        return JointEstimate(reason="ambiguous_consensus")
    return JointEstimate(position_m=point, supporting_cameras=tuple(view.camera_id for view in inliers),
                         reason="ok", reprojection_error_px=error,
                         ray_angle_degrees=_max_angle(inliers))


def _pose(observation):
    try:
        pixels = np.asarray(observation.landmarks_px, dtype=np.float64)
        confidence = _finite(observation.confidence, "confidence")
    except (AttributeError, ValueError, TypeError, OverflowError) as exc:
        raise CalibrationError("person observation requires landmarks_px and confidence") from exc
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not 1 <= len(pixels) <= MAX_JOINTS:
        raise CalibrationError(f"landmarks_px must have shape (N, 2), 1 <= N <= {MAX_JOINTS}")
    if confidence.shape != (len(pixels),) or np.any(confidence < 0) or np.any(confidence > 1):
        raise CalibrationError("confidence must contain one value in [0, 1] per joint")
    return pixels, confidence


def fuse_pose(observations, cameras, min_confidence=0.5,
              max_reprojection_error_px=3.0, min_ray_angle_degrees=1.0):
    """Fuse already-associated observations independently for every joint."""
    _limits(min_confidence, max_reprojection_error_px, min_ray_angle_degrees)
    if not hasattr(observations, "items") or not observations:
        raise CalibrationError("pose fusion requires an observation mapping")
    if len(observations) > MAX_CAMERAS:
        raise CalibrationError(f"at most {MAX_CAMERAS} cameras are supported per fusion")
    poses = {key: _pose(value) for key, value in observations.items()}
    sizes = {len(value[0]) for value in poses.values()}
    if len(sizes) != 1:
        raise CalibrationError("all cameras must use the same landmark schema")
    return tuple(fuse_joint({key: (pixels[joint], confidence[joint])
                             for key, (pixels, confidence) in poses.items()}, cameras,
                            min_confidence, max_reprojection_error_px, min_ray_angle_degrees)
                 for joint in range(sizes.pop()))


def estimate_pair_compatibility(first, second, first_camera, second_camera,
                                min_shared_joints=4, min_confidence=0.5,
                                max_epipolar_error_px=3.0):
    """Check common visible landmarks using lens-corrected epipolar geometry.

    At least 75% of common landmarks must also triangulate with positive depth
    and adequate parallax. A successful result is a geometric candidate, not an
    identity guarantee: association must reject multiple compatible candidates.
    """
    minimum, limit, _ = _limits(min_confidence, max_epipolar_error_px, 1.0)
    if not isinstance(min_shared_joints, Integral) or isinstance(min_shared_joints, (bool, np.bool_)) or not 1 <= min_shared_joints <= MAX_JOINTS:
        raise CalibrationError(f"min_shared_joints must be an integer in [1, {MAX_JOINTS}]")
    left, left_confidence = _pose(first)
    right, right_confidence = _pose(second)
    if left.shape != right.shape:
        raise CalibrationError("all cameras must use the same landmark schema")
    shared = np.flatnonzero((left_confidence >= minimum) & (left_confidence > 0)
                            & (right_confidence >= minimum) & (right_confidence > 0))
    if len(shared) < min_shared_joints:
        return PairCompatibility(False, len(shared), None, "insufficient_shared_joints")
    r1, t1 = _camera(first_camera)
    r2, t2 = _camera(second_camera)
    relative_rotation = r2 @ r1.T
    relative_translation = t2 - relative_rotation @ t1
    if np.linalg.norm(relative_translation) <= 1e-6:
        return PairCompatibility(False, len(shared), None, "coincident_cameras")
    tx, ty, tz = relative_translation
    cross = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
    fundamental = (np.linalg.inv(second_camera.calibration.camera_matrix).T @ cross
                   @ relative_rotation @ np.linalg.inv(first_camera.calibration.camera_matrix))
    distances = []
    triangulated = 0
    cameras = {"first": first_camera, "second": second_camera}
    for index in shared:
        observations = {"first": (left[index], left_confidence[index]),
                        "second": (right[index], right_confidence[index])}
        views = _prepare(observations, cameras, minimum)
        p1 = first_camera.calibration.camera_matrix @ np.r_[views[0].normalized, 1]
        p2 = second_camera.calibration.camera_matrix @ np.r_[views[1].normalized, 1]
        line2, line1 = fundamental @ p1, fundamental.T @ p2
        denominator1, denominator2 = np.linalg.norm(line1[:2]), np.linalg.norm(line2[:2])
        if min(denominator1, denominator2) <= 1e-15:
            distances.append(np.inf)
            continue
        residual = abs(float(p2 @ fundamental @ p1))
        distances.append(residual * 0.5 * (1 / denominator1 + 1 / denominator2))
        if distances[-1] <= limit and fuse_joint(observations, cameras, minimum, limit).position_m is not None:
            triangulated += 1
    median = float(np.median(distances))
    compatible = (np.isfinite(median) and median <= limit
                  and triangulated >= max(min_shared_joints, int(np.ceil(len(shared) * 0.75))))
    return PairCompatibility(bool(compatible), len(shared), median if np.isfinite(median) else None,
                             "ok" if compatible else "inconsistent_geometry")
