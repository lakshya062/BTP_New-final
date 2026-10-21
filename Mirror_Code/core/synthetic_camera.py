"""Deterministic virtual camera fixtures for calibration verification.

The rendered images contain an actual perspective chessboard with lens distortion,
antialiasing, blur and sensor noise. Corner detectors must recover pixels from the
image; ``image_points`` are separate ground truth and must not be passed as detected
corners when testing the image-to-calibration pipeline. All physical units are metres.
This module has no camera, GUI, machine-learning or filesystem side effects.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticCamera:
    image_size: Tuple[int, int]
    camera_matrix: np.ndarray
    distortion: np.ndarray


@dataclass(frozen=True)
class SyntheticView:
    image: np.ndarray
    object_points: np.ndarray
    image_points: np.ndarray
    rotation_vector: np.ndarray
    translation_vector: np.ndarray


def default_camera(image_size=(960, 720)):
    """Return a camera with known intrinsics and moderate radial distortion."""
    width, height = _image_size(image_size)
    return SyntheticCamera(
        (width, height),
        np.array(
            [[850.0 * width / 960, 0, width / 2.0 + 7.0 * width / 960],
             [0, 840.0 * height / 720, height / 2.0 - 5.0 * height / 720],
             [0, 0, 1]],
            dtype=np.float64,
        ),
        np.array([-0.17, 0.055, 0.001, -0.0007, 0.01], dtype=np.float64),
    )


def _image_size(image_size):
    if len(image_size) != 2 or any(
        not isinstance(value, (int, np.integer)) or value <= 0
        for value in image_size
    ):
        raise ValueError("image_size must contain two positive integers")
    return int(image_size[0]), int(image_size[1])


def chessboard_object_points(pattern_size=(9, 6), square_size=0.03):
    """Return row-major inner corners; pattern_size means (columns, rows)."""
    columns, rows = _image_size(pattern_size)
    if columns < 2 or rows < 2:
        raise ValueError("the chessboard must have at least two rows and columns")
    if not np.isfinite(square_size) or square_size <= 0:
        raise ValueError("square_size must be positive and finite")
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size
    return points


def _camera_parameters(camera):
    width, height = _image_size(camera.image_size)
    matrix = np.asarray(camera.camera_matrix, dtype=np.float64)
    distortion = np.asarray(camera.distortion, dtype=np.float64).reshape(-1)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[2], [0, 0, 1])
        or matrix[0, 1] != 0
        or matrix[1, 0] != 0
        or distortion.size not in (4, 5, 8, 12, 14)
        or not np.isfinite(distortion).all()
    ):
        raise ValueError("invalid pinhole camera matrix or distortion coefficients")
    return (width, height), matrix, distortion


@lru_cache(maxsize=4)
def _distorted_to_ideal_map(image_size, matrix_values, distortion_values):
    """For each distorted output pixel, find its ideal input pixel."""
    width, height = image_size
    matrix = np.array(matrix_values, dtype=np.float64).reshape(3, 3)
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack((xx, yy), axis=-1).astype(np.float32).reshape(-1, 1, 2)
    mapped = cv2.undistortPointsIter(
        pixels,
        matrix,
        np.array(distortion_values, dtype=np.float64),
        None,
        matrix,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-10),
    ).reshape(height, width, 2)
    return mapped[..., 0].copy(), mapped[..., 1].copy()


def render_chessboard(
    camera,
    rotation_vector,
    translation_vector,
    *,
    pattern_size=(9, 6),
    square_size=0.03,
    noise_std=0.0,
    blur_sigma=0.0,
    seed=0,
):
    """Render a physical board pose using supersampled inverse lens mapping.

    Rotation and translation follow OpenCV's object-to-camera convention. The
    board origin is the first inner corner. The finite white border is physical
    board material and is warped/distorted together with the black squares.
    """
    image_size, matrix, distortion = _camera_parameters(camera)
    width, height = image_size
    object_points = chessboard_object_points(pattern_size, square_size)
    columns, rows = pattern_size
    rotation_vector = np.asarray(rotation_vector, dtype=np.float64).reshape(3, 1)
    translation_vector = np.asarray(translation_vector, dtype=np.float64).reshape(3, 1)
    if not np.isfinite(rotation_vector).all() or not np.isfinite(translation_vector).all():
        raise ValueError("board pose must be finite")
    if any(not np.isfinite(value) or value < 0 for value in (noise_std, blur_sigma)):
        raise ValueError("noise_std and blur_sigma must be nonnegative and finite")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    # Checking only inner corners misses a nearby tilted board whose white
    # border crosses the camera plane. A planar board's depth extrema occur
    # at its physical outer vertices, so these cover every rendered pixel.
    outer_vertices = np.array(
        [[-1.5, -1.5, 0], [columns + 0.5, -1.5, 0],
         [columns + 0.5, rows + 0.5, 0], [-1.5, rows + 0.5, 0]],
        dtype=np.float64,
    ) * square_size
    camera_vertices = outer_vertices @ rotation.T + translation_vector.reshape(1, 3)
    if np.any(camera_vertices[:, 2] <= 0):
        raise ValueError("the entire board, including its border, must be in front of the camera")

    # A white half-square margin on every side allows real corner detection.
    pixels_per_square = 80
    texture = np.full(
        ((rows + 2) * pixels_per_square, (columns + 2) * pixels_per_square),
        245,
        dtype=np.uint8,
    )
    margin = pixels_per_square // 2
    for row in range(rows + 1):
        for column in range(columns + 1):
            if (row + column) % 2 == 0:
                x, y = margin + column * pixels_per_square, margin + row * pixels_per_square
                texture[y:y + pixels_per_square, x:x + pixels_per_square] = 12

    # Pixel centres, including supersampled ones, use the same convention as
    # cv2.projectPoints. Avoid the common half-pixel resize bias in ground truth.
    scale = 2
    sampling = np.array([[scale, 0, (scale - 1) / 2],
                         [0, scale, (scale - 1) / 2], [0, 0, 1]], dtype=np.float64)
    high_matrix = sampling @ matrix
    texture_to_board = np.array(
        [[square_size / pixels_per_square, 0, -1.5 * square_size],
         [0, square_size / pixels_per_square, -1.5 * square_size],
         [0, 0, 1]], dtype=np.float64,
    )
    # An integer checker square's edge lies half a texture pixel before its
    # first pixel centre. Align physical corners to that edge, not the centre.
    texture_to_board[0, 2] += square_size / (2 * pixels_per_square)
    texture_to_board[1, 2] += square_size / (2 * pixels_per_square)
    plane_to_camera = np.column_stack((rotation[:, :2], translation_vector))
    homography = high_matrix @ plane_to_camera @ texture_to_board
    ideal = cv2.warpPerspective(
        texture, homography, (width * scale, height * scale),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=155,
    )
    map_x, map_y = _distorted_to_ideal_map(
        (width * scale, height * scale),
        tuple(high_matrix.reshape(-1)), tuple(distortion),
    )
    distorted = cv2.remap(
        ideal, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=155,
    )
    image = cv2.resize(distorted, image_size, interpolation=cv2.INTER_AREA)
    if blur_sigma:
        image = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    if noise_std:
        noise = np.random.default_rng(seed).normal(0, noise_std, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    image_points, _ = cv2.projectPoints(
        object_points, rotation_vector, translation_vector, matrix, distortion,
    )
    return SyntheticView(
        image, object_points, image_points.astype(np.float32),
        rotation_vector.copy(), translation_vector.copy(),
    )


def generate_chessboard_views(
    count=16,
    *,
    camera: Optional[SyntheticCamera] = None,
    pattern_size=(9, 6),
    square_size=0.03,
    noise_std=1.0,
    blur_sigma=0.5,
    seed=42,
):
    """Generate tilted boards at varied distances and positions across the image."""
    if not isinstance(count, (int, np.integer)) or count <= 0:
        raise ValueError("count must be a positive integer")
    camera = default_camera() if camera is None else camera
    (width, height), matrix, distortion = _camera_parameters(camera)
    object_points = chessboard_object_points(pattern_size, square_size)
    board_center = object_points.mean(axis=0)
    columns, rows = pattern_size
    outer = np.array(
        [[-1.5, -1.5, 0], [columns + 0.5, -1.5, 0],
         [columns + 0.5, rows + 0.5, 0], [-1.5, rows + 0.5, 0]],
        dtype=np.float64,
    ) * square_size
    rng = np.random.default_rng(seed)
    views = []
    for _ in range(count):
        for attempt in range(100):
            angles = np.deg2rad(rng.uniform([-32, -32, -16], [32, 32, 16]))
            # Independent Euler rotations produce non-parallel physical views.
            rx, _ = cv2.Rodrigues(np.array([angles[0], 0, 0]))
            ry, _ = cv2.Rodrigues(np.array([0, angles[1], 0]))
            rz, _ = cv2.Rodrigues(np.array([0, 0, angles[2]]))
            rotation = rz @ ry @ rx
            rvec, _ = cv2.Rodrigues(rotation)
            distance = rng.uniform(0.62, 1.05) * square_size / 0.03
            center_pixel = rng.uniform([0.27 * width, 0.27 * height],
                                       [0.73 * width, 0.73 * height])
            center_ray = cv2.undistortPoints(
                center_pixel.reshape(1, 1, 2), matrix, distortion,
            ).reshape(2)
            center_camera = np.array([*center_ray, 1.0]) * distance
            tvec = (center_camera - rotation @ board_center).reshape(3, 1)
            boundary, _ = cv2.projectPoints(outer, rvec, tvec, matrix, distortion)
            boundary = boundary.reshape(-1, 2)
            if (
                np.all(boundary[:, 0] > 20) and np.all(boundary[:, 0] < width - 20)
                and np.all(boundary[:, 1] > 20) and np.all(boundary[:, 1] < height - 20)
            ):
                break
        else:
            raise ValueError("cannot fit a visible chessboard inside this camera's image")
        views.append(render_chessboard(
            camera, rvec, tvec, pattern_size=pattern_size, square_size=square_size,
            noise_std=noise_std, blur_sigma=blur_sigma,
            seed=int(rng.integers(0, 2 ** 32)),
        ))
    return views
