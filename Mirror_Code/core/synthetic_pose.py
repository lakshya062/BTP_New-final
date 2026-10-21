"""Deterministic metric pose fixtures for testing camera calibration.

Coordinates are in metres with +X right, +Y down and +Z forward. Each arm
contains shoulder, elbow and wrist, in that order. The elbow is the origin;
the unrotated motion plane is XY. Yaw rotates that plane around Y, so a
zero-degree yaw is a side-on camera view and 90 degrees is edge-on.

These fixtures exercise geometry with *known landmark correspondences*.
They do not test whether a pose model detects the right landmarks in video.
Lens calibration recovers camera rays; a single ray still has unknown depth.
Consequently a 2D angle between calibrated rays need not equal a 3D angle.
"""

from math import atan2, cos, degrees, isfinite, radians, sin

import cv2
import numpy as np


def _finite_array(value, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def _finite_scalar(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def calculate_joint_angle_3d(a, b, c):
    """Return the 0–180 degree metric angle at b for three 3D points.

    Coincident and nonfinite landmarks are rejected instead of producing a
    plausible-looking angle. Coordinate units must match across all points.
    """
    points = [_finite_array(point, "3D point") for point in (a, b, c)]
    if any(point.shape != (3,) for point in points):
        raise ValueError("Each 3D point must contain exactly three coordinates")
    with np.errstate(over="ignore", invalid="ignore"):
        first, second = points[0] - points[1], points[2] - points[1]
    for vector in (first, second):
        if not np.isfinite(vector).all():
            raise ValueError("Joint angle requires finite limb vectors")
        scale = np.max(np.abs(vector))
        if scale <= 1e-12:
            raise ValueError("Joint angle requires two nonzero limb vectors")
        # Scale separately before the dot/cross products to avoid overflow.
        vector /= scale
        vector /= np.linalg.norm(vector)
    return degrees(atan2(float(np.linalg.norm(np.cross(first, second))),
                         float(np.dot(first, second))))


def generate_arm_pose(
    angle_degrees,
    plane_yaw_degrees=0.0,
    upper_arm_m=0.32,
    forearm_m=0.27,
):
    """Build shoulder/elbow/wrist at a known joint angle and viewing yaw.

    The default segment lengths are plausible adult arm dimensions, not an
    anatomical model. All angles from zero (folded) to 180 (straight) are
    supported. Camera placement is intentionally a separate transformation.
    """
    angle = _finite_scalar(angle_degrees, "angle_degrees")
    yaw = _finite_scalar(plane_yaw_degrees, "plane_yaw_degrees")
    upper = _finite_scalar(upper_arm_m, "upper_arm_m")
    forearm = _finite_scalar(forearm_m, "forearm_m")
    if not 0.0 <= angle <= 180.0:
        raise ValueError("angle_degrees must be between zero and 180")
    if upper <= 1e-12 or forearm <= 1e-12:
        raise ValueError("Arm segment lengths must be positive")
    theta, yaw = radians(angle), radians(yaw)
    # The shoulder points up from the elbow; the wrist rotates away from it.
    wrist_x = forearm * sin(theta)
    return np.array([
        [0.0, -upper, 0.0],
        [0.0, 0.0, 0.0],
        [wrist_x * cos(yaw), -forearm * cos(theta), -wrist_x * sin(yaw)],
    ], dtype=np.float64)


def generate_arm_trajectory(angles_degrees, plane_yaw_degrees=0.0):
    """Return an (N, 3, 3) sequence of known arm poses, in metres.

    Pass any nonempty 1D sequence of angles to model a curl or extension.
    Nothing is randomized, making projection/noise tests reproducible.
    """
    angles = _finite_array(angles_degrees, "angles_degrees")
    if angles.ndim != 1 or angles.size == 0:
        raise ValueError("angles_degrees must be a nonempty 1D sequence")
    return np.stack([generate_arm_pose(angle, plane_yaw_degrees) for angle in angles])


def _camera_parameters(camera_matrix, dist_coeffs):
    matrix = _finite_array(camera_matrix, "camera_matrix")
    if matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape (3, 3)")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    if (not np.allclose(matrix[2], (0.0, 0.0, 1.0), rtol=0, atol=1e-12)
            or matrix[0, 1] != 0 or matrix[1, 0] != 0):
        raise ValueError("Expected an OpenCV camera matrix with zero skew")
    if dist_coeffs is None:
        return matrix, None
    coefficients = _finite_array(dist_coeffs, "dist_coeffs")
    if (coefficients.ndim not in (1, 2)
            or (coefficients.ndim == 2 and 1 not in coefficients.shape)
            or coefficients.size not in (4, 5, 8, 12, 14)):
        raise ValueError("dist_coeffs must contain 4, 5, 8, 12 or 14 coefficients")
    return matrix, coefficients.reshape(-1)


def _camera_vector(value, name):
    if value is None:
        return np.zeros(3, dtype=np.float64)
    result = _finite_array(value, name)
    if result.shape not in ((3,), (3, 1), (1, 3)):
        raise ValueError(f"{name} must contain three coordinates")
    return result.reshape(3)


def project_pose(
    points_3d,
    camera_matrix,
    dist_coeffs=None,
    rotation_vector=None,
    translation_vector=None,
):
    """Project metric points through an OpenCV pinhole/distortion camera.

    The extrinsics map object coordinates to camera coordinates:
    ``X_camera = R @ X_object + translation_vector``. Points behind or on the
    camera plane are rejected. Leading dimensions are retained, so (N, 3, 3)
    arm trajectories produce (N, 3, 2) pixel coordinates. A camera at the
    origin looking at an arm three metres away can use translation [0, 0, 3].
    """
    points = _finite_array(points_3d, "points_3d")
    if points.ndim < 1 or points.shape[-1] != 3 or points.size == 0:
        raise ValueError("points_3d must be a nonempty array ending in dimension 3")
    matrix, coefficients = _camera_parameters(camera_matrix, dist_coeffs)
    rotation = _camera_vector(rotation_vector, "rotation_vector")
    translation = _camera_vector(translation_vector, "translation_vector")
    flat_points = np.ascontiguousarray(points.reshape(-1, 3))
    rotation_matrix, _ = cv2.Rodrigues(rotation)
    with np.errstate(over="ignore", invalid="ignore"):
        camera_points = flat_points @ rotation_matrix.T + translation
    if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 1e-9):
        raise ValueError("All projected points must have positive, finite camera depth")
    pixels, _ = cv2.projectPoints(flat_points, rotation, translation, matrix, coefficients)
    if not np.isfinite(pixels).all():
        raise ValueError("Camera projection produced nonfinite image coordinates")
    return pixels.reshape(points.shape[:-1] + (2,))


def undistort_pose(points_2d, camera_matrix, dist_coeffs=None):
    """Return distortion-free normalized camera coordinates, preserving shape.

    Results have unit focal length, not pixel units. This also removes unequal
    X/Y pixel scaling. Each (x, y) defines ray (x, y, 1); its depth is unknown.
    Supplying a calibrated matrix therefore does not recover a metric 3D pose.
    """
    points = _finite_array(points_2d, "points_2d")
    if points.ndim < 1 or points.shape[-1] != 2 or points.size == 0:
        raise ValueError("points_2d must be a nonempty array ending in dimension 2")
    matrix, coefficients = _camera_parameters(camera_matrix, dist_coeffs)
    normalized = cv2.undistortPointsIter(
        np.ascontiguousarray(points.reshape(-1, 1, 2)),
        matrix,
        coefficients,
        None,
        None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12),
    )
    if not np.isfinite(normalized).all():
        raise ValueError("Undistortion produced nonfinite camera coordinates")
    return normalized.reshape(points.shape)
