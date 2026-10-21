"""Offline pinhole calibration and metric stereo geometry, independent of live pose.

Board coordinates and translations use meters. Pixel coordinates use the original
calibration resolution (width, height). Stereo transforms follow OpenCV's
convention: ``point_right = rotation @ point_left + translation``. A low training
reprojection error does not establish real-world accuracy; validate held-out views
and physical measurements before using a calibration on a mirror.
"""

from dataclasses import dataclass
from numbers import Integral

import cv2
import numpy as np


class CalibrationError(ValueError):
    """Invalid, underconstrained, or poor-quality calibration/3D observations."""


def _array(value, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalibrationError(f"{name} must contain finite numbers") from exc
    if not np.isfinite(result).all():
        raise CalibrationError(f"{name} must contain finite numbers")
    return result


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise CalibrationError(f"{name} must be a positive finite number")
    result = _array(value, name)
    if result.ndim != 0 or result <= 0:
        raise CalibrationError(f"{name} must be a positive finite number")
    return float(result)


def _image_size(value):
    try:
        valid = len(value) == 2 and all(
            isinstance(item, Integral) and not isinstance(item, (bool, np.bool_))
            and 0 < item <= np.iinfo(np.int32).max for item in value
        )
    except TypeError:
        valid = False
    if not valid:
        raise CalibrationError("image_size must be positive integer (width, height)")
    return tuple(int(item) for item in value)


def _points(value, dimensions, name, minimum=1):
    result = _array(value, name)
    if result.ndim == 3 and result.shape[1:] == (1, dimensions):
        result = result[:, 0, :]
    if result.ndim != 2 or result.shape[1] != dimensions or len(result) < minimum:
        raise CalibrationError(f"{name} must have shape (N, {dimensions}), N >= {minimum}")
    return np.ascontiguousarray(result)


def _pixels(value, image_size, name, minimum=1):
    result = _points(value, 2, name, minimum)
    if np.any(result < 0) or np.any(result >= np.asarray(image_size)):
        raise CalibrationError(f"{name} contains pixels outside the calibration image")
    return result


def _readonly(value):
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _camera_parameters(matrix, distortion):
    matrix = _array(matrix, "camera_matrix")
    if matrix.shape != (3, 3) or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise CalibrationError("camera_matrix must be 3x3 with positive focal lengths")
    if not np.allclose(matrix[2], [0, 0, 1], atol=1e-12, rtol=0) or not np.allclose(
        [matrix[0, 1], matrix[1, 0]], 0, atol=1e-12, rtol=0
    ):
        raise CalibrationError("camera_matrix must use the OpenCV zero-skew pinhole form")
    distortion = _array(distortion, "distortion_coefficients")
    if distortion.ndim > 2 or (distortion.ndim == 2 and 1 not in distortion.shape):
        raise CalibrationError("distortion_coefficients must be a vector")
    if distortion.size not in (4, 5, 8, 12, 14):
        raise CalibrationError("distortion_coefficients must have 4, 5, 8, 12, or 14 values")
    return matrix, distortion.reshape(-1)


def _rigid_transform(rotation, translation):
    rotation = _array(rotation, "rotation")
    translation = _array(translation, "translation")
    if rotation.shape != (3, 3) or not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0
    ) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7, rtol=0):
        raise CalibrationError("rotation must be a proper orthonormal 3x3 matrix")
    if translation.shape not in ((3,), (3, 1), (1, 3)):
        raise CalibrationError("translation must contain three coordinates")
    translation = translation.reshape(3)
    if np.linalg.norm(translation) <= 1e-9:
        raise CalibrationError("stereo cameras must have a nonzero baseline")
    return rotation, translation


def _check_lens_model(matrix, distortion, image_size):
    """Reject folds in a fitted five-term lens across the nominal image field.

    OpenCV's unconstrained optimizer can return a non-monotonic radial polynomial
    despite a small corner residual. Radial extrema are checked analytically;
    tangential fold checks sample a grid and are a diagnostic, not a proof that
    a physical lens follows this model outside the measured board coverage.
    """
    if len(distortion) not in (4, 5):
        return  # This solver fits five terms; extended external models are not fitted here.
    k1, k2, p1, p2 = distortion[:4]
    k3 = distortion[4] if len(distortion) == 5 else 0.0
    width, height = image_size
    x_edges = (np.array([0, width - 1]) - matrix[0, 2]) / matrix[0, 0]
    y_edges = (np.array([0, height - 1]) - matrix[1, 2]) / matrix[1, 1]
    radius_squared = max(x_edges ** 2) + max(y_edges ** 2)
    # d(r * radial(r^2))/dr = 1 + 3*k1*s + 5*k2*s^2 + 7*k3*s^3.
    critical = np.polynomial.polynomial.polyroots([3*k1, 10*k2, 21*k3])
    candidates = [0.0, radius_squared, *(
        float(root.real) for root in critical
        if abs(root.imag) < 1e-10 and 0 < root.real < radius_squared
    )]
    with np.errstate(over="ignore", invalid="ignore"):
        radial_slope = np.polynomial.polynomial.polyval(candidates, [1, 3*k1, 5*k2, 7*k3])
        x, y = np.meshgrid(np.linspace(*x_edges, 25), np.linspace(*y_edges, 25))
        s = x*x + y*y
        radial = 1 + k1*s + k2*s*s + k3*s*s*s
        gradient = k1 + 2*k2*s + 3*k3*s*s
        xx = radial + 2*x*x*gradient + 2*p1*y + 6*p2*x
        xy = 2*x*y*gradient + 2*p1*x + 2*p2*y
        yy = radial + 2*y*y*gradient + 6*p1*y + 2*p2*x
        determinants = xx*yy - xy*xy
    if not np.isfinite(radial_slope).all() or np.any(radial_slope <= 0):
        raise CalibrationError("lens distortion is non-monotonic within the image field")
    if not np.isfinite(determinants).all() or np.any(determinants <= 0):
        raise CalibrationError("lens distortion folds within the image field")


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size: tuple
    rms_error_px: float = 0.0
    per_view_rms_px: tuple = ()
    rotation_vectors: tuple = ()
    translation_vectors: tuple = ()

    def __post_init__(self):
        matrix, distortion = _camera_parameters(self.camera_matrix, self.distortion_coefficients)
        size = _image_size(self.image_size)
        _check_lens_model(matrix, distortion, size)
        error = _array(self.rms_error_px, "rms_error_px")
        if error.ndim != 0 or error < 0:
            raise CalibrationError("rms_error_px must be nonnegative")
        object.__setattr__(self, "camera_matrix", _readonly(matrix))
        object.__setattr__(self, "distortion_coefficients", _readonly(distortion))
        object.__setattr__(self, "image_size", size)
        object.__setattr__(self, "rms_error_px", float(error))
        per_view = _array(self.per_view_rms_px, "per_view_rms_px")
        if per_view.ndim != 1 or np.any(per_view < 0):
            raise CalibrationError("per_view_rms_px must be a nonnegative vector")
        rotations = tuple(_readonly(_points([np.asarray(v).reshape(3)], 3, "rotation_vectors")[0])
                          for v in self.rotation_vectors)
        translations = tuple(_readonly(_points([np.asarray(v).reshape(3)], 3, "translation_vectors")[0])
                             for v in self.translation_vectors)
        if not (len(per_view) == len(rotations) == len(translations)):
            raise CalibrationError("per-view errors and poses must have the same length")
        object.__setattr__(self, "per_view_rms_px", tuple(float(v) for v in per_view))
        object.__setattr__(self, "rotation_vectors", rotations)
        object.__setattr__(self, "translation_vectors", translations)


@dataclass(frozen=True)
class StereoCalibration:
    rotation: np.ndarray
    translation: np.ndarray
    rms_error_px: float

    def __post_init__(self):
        rotation, translation = _rigid_transform(self.rotation, self.translation)
        error = _array(self.rms_error_px, "rms_error_px")
        if error.ndim != 0 or error < 0:
            raise CalibrationError("rms_error_px must be nonnegative")
        object.__setattr__(self, "rotation", _readonly(rotation))
        object.__setattr__(self, "translation", _readonly(translation))
        object.__setattr__(self, "rms_error_px", float(error))


def chessboard_object_points(columns, rows, square_size_m):
    """Return row-major inner-corner coordinates on a board's z=0 plane."""
    if any(not isinstance(v, Integral) or isinstance(v, (bool, np.bool_)) or v < 2
           for v in (columns, rows)):
        raise CalibrationError("a chessboard needs at least two inner corners on each axis")
    size = _positive(square_size_m, "square_size_m")
    points = np.zeros((rows * columns, 3), dtype=np.float64)
    points[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * size
    return points


def _view_collections(object_views, image_views, image_size, min_views):
    if not isinstance(min_views, Integral) or isinstance(min_views, (bool, np.bool_)) or min_views < 3:
        raise CalibrationError("min_views must be an integer >= 3; six or more are recommended")
    try:
        objects, images = list(object_views), list(image_views)
    except TypeError as exc:
        raise CalibrationError("calibration requires sequences of object and image point views") from exc
    if len(objects) != len(images) or len(objects) < min_views:
        raise CalibrationError(f"calibration requires at least {min_views} matching views")
    checked_objects, checked_images = [], []
    for index, (obj, img) in enumerate(zip(objects, images)):
        obj = _points(obj, 3, f"object view {index}", 6)
        img = _pixels(img, image_size, f"image view {index}", 6)
        if len(obj) != len(img):
            raise CalibrationError(f"point counts differ in view {index}")
        if not np.allclose(obj[:, 2], 0, atol=1e-9, rtol=0):
            raise CalibrationError("board object points must lie on the z=0 plane")
        if any(np.any(np.abs(points) > np.finfo(np.float32).max) for points in (obj, img)):
            raise CalibrationError("board observations exceed OpenCV coordinate precision")
        for points, label in ((obj[:, :2], "object"), (img, "image")):
            singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
            if singular[0] <= 1e-12 or singular[1] / singular[0] < 1e-3:
                raise CalibrationError(f"{label} points in view {index} are collinear or degenerate")
            if len(np.unique(points, axis=0)) != len(points):
                raise CalibrationError(f"{label} points in view {index} contain duplicates")
        for points, collection in ((obj, checked_objects), (img, checked_images)):
            converted = points.astype(np.float32)
            if len(np.unique(converted, axis=0)) != len(converted):
                raise CalibrationError("board observations collapse at OpenCV coordinate precision")
            collection.append(converted)
    return checked_objects, checked_images


def _check_view_diversity(objects, images, image_size):
    # Zhang's two intrinsic constraints per planar homography need rank five.
    # Normalize coordinates first so the condition check is resolution/board-unit independent.
    constraints = []
    for obj, img in zip(objects, images):
        xy = obj[:, :2].astype(np.float64)
        xy = (xy - xy.mean(axis=0)) / np.sqrt(np.mean(np.sum((xy - xy.mean(axis=0)) ** 2, axis=1)))
        uv = (img.astype(np.float64) - np.asarray(image_size) / 2) / max(image_size)
        homography, _ = cv2.findHomography(xy, uv, 0)
        if homography is None or not np.isfinite(homography).all():
            raise CalibrationError("cannot estimate a finite board homography")

        def constraint(i, j):
            a, b = homography[:, i], homography[:, j]
            return np.array([a[0]*b[0], a[0]*b[1]+a[1]*b[0], a[1]*b[1],
                             a[2]*b[0]+a[0]*b[2], a[2]*b[1]+a[1]*b[2], a[2]*b[2]])

        for row in (constraint(0, 1), constraint(0, 0) - constraint(1, 1)):
            constraints.append(row / max(np.linalg.norm(row), 1e-15))
    singular = np.linalg.svd(np.asarray(constraints), compute_uv=False)
    if singular[4] / singular[0] < 5e-4:
        raise CalibrationError("insufficient board tilt diversity; capture varied tilted views")


def _quality(errors, limit):
    if not np.isfinite(errors).all() or np.any(np.asarray(errors) > limit):
        raise CalibrationError(f"reprojection error exceeds {limit:g} pixel quality limit")


def _undistorted_views(images, matrix, distortion):
    return [cv2.undistortPointsIter(
        image[:, None].astype(np.float64), matrix, distortion, None, matrix,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-12),
    )[:, 0] for image in images]


def calibrate_camera(object_points_views, image_points_views, image_size, *, max_rms_px=1.0, min_views=6):
    """Recover intrinsics and five Brown-Conrady distortion terms from board views.

    Every view must contain at least six corresponding, non-collinear corners.
    Repeated or only front-facing views are underconstrained and rejected. Both
    overall RMS and each view's RMS must meet ``max_rms_px``. No corners are
    silently discarded; callers should diagnose poor captures and recollect them.
    """
    size = _image_size(image_size)
    limit = _positive(max_rms_px, "max_rms_px")
    objects, images = _view_collections(object_points_views, image_points_views, size, min_views)
    _check_view_diversity(objects, images, size)
    try:
        rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
            objects, images, size, None, None,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-10),
        )
        errors = []
        for obj, img, rvec, tvec in zip(objects, images, rotations, translations):
            predicted, _ = cv2.projectPoints(obj, rvec, tvec, matrix, distortion)
            errors.append(float(np.sqrt(np.mean(np.sum((predicted[:, 0] - img) ** 2, axis=1)))))
            rotation, _ = cv2.Rodrigues(rvec)
            if np.any((obj @ rotation.T + tvec.reshape(3))[:, 2] <= 0):
                raise CalibrationError("calibration placed board points behind the camera")
        _quality([rms, *errors], limit)
        # Lens distortion must not manufacture apparent homography tilt in a
        # capture set consisting entirely of parallel, translated boards.
        _check_lens_model(matrix, distortion.reshape(-1), size)
        _check_view_diversity(objects, _undistorted_views(images, matrix, distortion), size)
    except cv2.error as exc:
        raise CalibrationError("OpenCV could not calibrate these board observations") from exc
    return CameraCalibration(matrix, distortion, size, rms, tuple(errors), tuple(rotations), tuple(translations))


def calibrate_stereo(object_points_views, left_image_points_views, right_image_points_views,
                     left_calibration, right_calibration, *, max_rms_px=1.0, min_views=6):
    """Recover the right-from-left rigid transform from simultaneous board views.

    Intrinsics remain fixed. Point order, board poses, and camera timestamps must
    correspond between views. Different camera image resolutions are unsupported.
    """
    if not isinstance(left_calibration, CameraCalibration) or not isinstance(right_calibration, CameraCalibration):
        raise CalibrationError("both cameras need CameraCalibration intrinsics")
    if left_calibration.image_size != right_calibration.image_size:
        raise CalibrationError("stereo calibration requires matching image resolutions")
    limit = _positive(max_rms_px, "max_rms_px")
    objects, left = _view_collections(object_points_views, left_image_points_views,
                                     left_calibration.image_size, min_views)
    _, right = _view_collections(objects, right_image_points_views, right_calibration.image_size, min_views)
    try:
        _check_view_diversity(objects, _undistorted_views(
            left, left_calibration.camera_matrix, left_calibration.distortion_coefficients,
        ), left_calibration.image_size)
        _check_view_diversity(objects, _undistorted_views(
            right, right_calibration.camera_matrix, right_calibration.distortion_coefficients,
        ), right_calibration.image_size)
        result = cv2.stereoCalibrateExtended(
            objects, left, right, left_calibration.camera_matrix.copy(),
            left_calibration.distortion_coefficients.copy(), right_calibration.camera_matrix.copy(),
            right_calibration.distortion_coefficients.copy(), left_calibration.image_size, None, None,
            flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-10),
        )
    except cv2.error as exc:
        raise CalibrationError("OpenCV could not calibrate these stereo observations") from exc
    rms, rotation, translation = result[0], result[5], result[6]
    # A bad capture must not disappear in the average across many good views.
    _quality(np.r_[rms, np.asarray(result[-1]).reshape(-1)], limit)
    return StereoCalibration(rotation, translation, rms)


def triangulate_stereo_points(left_pixels, right_pixels, left_calibration, right_calibration,
                              right_rotation, right_translation, *, min_ray_angle_degrees=0.1,
                              max_reprojection_error_px=2.0):
    """Return corresponding 3D points in the left camera frame, in meters.

    Reject an entire observation batch if any pair is nonfinite, off-image,
    nearly parallel, behind either camera, or inconsistent with the calibration.
    This requires two actual views; MediaPipe's monocular z is not a stereo view.
    """
    if not isinstance(left_calibration, CameraCalibration) or not isinstance(right_calibration, CameraCalibration):
        raise CalibrationError("both cameras need CameraCalibration intrinsics")
    rotation, translation = _rigid_transform(right_rotation, right_translation)
    min_angle = _positive(min_ray_angle_degrees, "min_ray_angle_degrees")
    if min_angle >= 90:
        raise CalibrationError("min_ray_angle_degrees must be less than 90")
    limit = _positive(max_reprojection_error_px, "max_reprojection_error_px")
    left = _pixels(left_pixels, left_calibration.image_size, "left_pixels")
    right = _pixels(right_pixels, right_calibration.image_size, "right_pixels")
    if left.shape != right.shape:
        raise CalibrationError("left and right point counts must match")
    try:
        normalized = []
        for pixels, camera in ((left, left_calibration), (right, right_calibration)):
            normalized.append(cv2.undistortPointsIter(
                pixels[:, None], camera.camera_matrix, camera.distortion_coefficients, None, None,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-12),
            )[:, 0])
        left_rays = np.column_stack((normalized[0], np.ones(len(left))))
        right_rays = np.column_stack((normalized[1], np.ones(len(right)))) @ rotation
        angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(left_rays, right_rays), axis=1),
                                       np.abs(np.sum(left_rays * right_rays, axis=1))))
        if not np.isfinite(angles).all() or np.any(angles < min_angle):
            raise CalibrationError("stereo rays are nearly parallel; depth is unobservable")
        homogeneous = cv2.triangulatePoints(
            np.column_stack((np.eye(3), np.zeros(3))), np.column_stack((rotation, translation)),
            normalized[0].T, normalized[1].T,
        )
        if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[3]) <= 1e-10):
            raise CalibrationError("triangulated points are at infinity")
        points = (homogeneous[:3] / homogeneous[3]).T
        if np.any(points[:, 2] <= 0) or np.any((points @ rotation.T + translation)[:, 2] <= 0):
            raise CalibrationError("triangulated points are behind a camera")
        for pixels, camera, rvec, tvec in (
            (left, left_calibration, np.zeros(3), np.zeros(3)),
            (right, right_calibration, cv2.Rodrigues(rotation)[0], translation),
        ):
            projected, _ = cv2.projectPoints(points, rvec, tvec, camera.camera_matrix,
                                              camera.distortion_coefficients)
            _quality(np.linalg.norm(projected[:, 0] - pixels, axis=1), limit)
    except cv2.error as exc:
        raise CalibrationError("OpenCV could not triangulate these observations") from exc
    return points
