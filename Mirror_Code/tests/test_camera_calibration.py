"""Deterministic numerical controls; raster corner detection is tested separately."""

import unittest

import cv2
import numpy as np

from core.camera_calibration import (
    CalibrationError,
    CameraCalibration,
    StereoCalibration,
    calibrate_camera,
    calibrate_stereo,
    chessboard_object_points,
    triangulate_stereo_points,
)


IMAGE_SIZE = (1280, 960)
MATRIX = np.array([[980.0, 0, 640], [0, 1000.0, 480], [0, 0, 1]])
DISTORTION = np.array([-0.12, 0.025, 0.001, -0.0008, 0.002])
RIGHT_ROTATION = cv2.Rodrigues(np.array([0.005, 0.025, -0.003]))[0]
RIGHT_TRANSLATION = np.array([-0.24, 0.002, 0.004])


def observations(noise=0.0, count=18, front_only=False):
    rng = np.random.default_rng(173)
    board = chessboard_object_points(9, 6, 0.035)
    objects, left, right = [], [], []
    for index in range(count):
        rotation_vector = rng.uniform([-0.5, -0.5, -0.15], [0.5, 0.5, 0.15])
        if front_only:
            rotation_vector = np.zeros(3)
        translation = rng.uniform([-0.2, -0.16, 0.65], [0.12, 0.08, 1.25])
        rotation = cv2.Rodrigues(rotation_vector)[0]
        right_vector = cv2.Rodrigues(RIGHT_ROTATION @ rotation)[0]
        right_translation = RIGHT_ROTATION @ translation + RIGHT_TRANSLATION
        for views, rvec, tvec in ((left, rotation_vector, translation),
                                 (right, right_vector, right_translation)):
            projected = cv2.projectPoints(board, rvec, tvec, MATRIX, DISTORTION)[0][:, 0]
            views.append(projected + rng.normal(0, noise, projected.shape))
        objects.append(board.copy())
    return objects, left, right


class CameraCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.objects, cls.images, cls.right_images = observations()
        cls.camera = CameraCalibration(MATRIX, DISTORTION, IMAGE_SIZE)

    def test_recovers_known_intrinsics_and_distortion_from_tilted_boards(self):
        result = calibrate_camera(self.objects, self.images, IMAGE_SIZE)
        self.assertLess(result.rms_error_px, 0.001)
        np.testing.assert_allclose(result.camera_matrix, MATRIX, atol=0.02)
        np.testing.assert_allclose(result.distortion_coefficients, DISTORTION, atol=0.001)
        self.assertEqual(len(result.per_view_rms_px), len(self.objects))
        self.assertEqual(len(result.rotation_vectors), len(self.objects))
        self.assertEqual(len(result.translation_vectors), len(self.objects))
        with self.assertRaises(ValueError):
            result.camera_matrix[0, 0] = 1

    def test_localization_noise_retains_focal_accuracy(self):
        objects, images, _ = observations(noise=0.2)
        result = calibrate_camera(objects, images, IMAGE_SIZE)
        self.assertLess(result.rms_error_px, 0.4)
        self.assertLess(abs(result.camera_matrix[0, 0] / MATRIX[0, 0] - 1), 0.02)
        self.assertLess(abs(result.camera_matrix[1, 1] / MATRIX[1, 1] - 1), 0.02)

    def test_quality_gate_rejects_noisy_or_corrupted_views(self):
        objects, images, _ = observations(noise=2)
        with self.assertRaisesRegex(CalibrationError, "reprojection"):
            calibrate_camera(objects, images, IMAGE_SIZE)
        corrupted = [view.copy() for view in self.images]
        corrupted[-1][0] += [15, -15]
        with self.assertRaisesRegex(CalibrationError, "reprojection"):
            calibrate_camera(self.objects, corrupted, IMAGE_SIZE, max_rms_px=0.5)

    def test_rejects_insufficient_and_mismatched_views(self):
        for objects, images in ((self.objects[:5], self.images[:5]),
                                (self.objects, self.images[:-1]), (None, None)):
            with self.subTest(views=images), self.assertRaises(CalibrationError):
                calibrate_camera(objects, images, IMAGE_SIZE)

    def test_rejects_repeated_and_front_facing_degenerate_views(self):
        for objects, images in (([self.objects[0]] * 6, [self.images[0]] * 6),
                                observations(front_only=True)[:2]):
            with self.assertRaisesRegex(CalibrationError, "diversity"):
                calibrate_camera(objects, images, IMAGE_SIZE)

    def test_rejects_invalid_shapes_nonfinite_off_image_and_bad_board(self):
        for bad in (np.zeros((54, 2)), np.full((54, 3), np.nan),
                    np.tile([1, 1, 0], (54, 1)),
                    np.column_stack((np.arange(54), np.arange(54), np.zeros(54))),
                    self.objects[0] + [0, 0, 0.1]):
            objects = [bad, *self.objects[1:]]
            with self.subTest(object=bad.shape), self.assertRaises(CalibrationError):
                calibrate_camera(objects, self.images, IMAGE_SIZE)
        for bad in (np.zeros((54, 3)), np.full((54, 2), np.inf),
                    np.tile([12, 12], (54, 1)), self.images[0][:-1],
                    np.full((54, 2), -0.1), np.full((54, 2), [1280, 960])):
            with self.subTest(image=bad.shape), self.assertRaises(CalibrationError):
                calibrate_camera(self.objects, [bad, *self.images[1:]], IMAGE_SIZE)

    def test_invalid_sizes_thresholds_and_boards_are_rejected(self):
        for size in ((0, 960), (1280.0, 960), (1280,), None, (True, 960)):
            with self.subTest(size=size), self.assertRaises(CalibrationError):
                calibrate_camera(self.objects, self.images, size)
        for limit in (0, -1, np.nan, np.inf, [1], True):
            with self.subTest(limit=limit), self.assertRaises(CalibrationError):
                calibrate_camera(self.objects, self.images, IMAGE_SIZE, max_rms_px=limit)
        for minimum in (0, 2, 3.2, True):
            with self.subTest(minimum=minimum), self.assertRaises(CalibrationError):
                calibrate_camera(self.objects, self.images, IMAGE_SIZE, min_views=minimum)
        for args in ((1, 6, 0.03), (9.0, 6, 0.03), (9, 6, 0), (9, 6, np.nan)):
            with self.subTest(args=args), self.assertRaises(CalibrationError):
                chessboard_object_points(*args)

    def test_recovers_stereo_transform_using_board_correspondences(self):
        result = calibrate_stereo(self.objects, self.images, self.right_images,
                                  self.camera, self.camera)
        self.assertLess(result.rms_error_px, 0.001)
        np.testing.assert_allclose(result.rotation, RIGHT_ROTATION, atol=1e-5)
        np.testing.assert_allclose(result.translation, RIGHT_TRANSLATION, atol=1e-5)

    def test_noisy_stereo_recovers_baseline_and_rejects_mismatched_capture(self):
        objects, left, right = observations(noise=0.2)
        result = calibrate_stereo(objects, left, right, self.camera, self.camera)
        self.assertLess(np.linalg.norm(result.translation - RIGHT_TRANSLATION), 0.002)
        self.assertLess(result.rms_error_px, 0.5)
        with self.assertRaisesRegex(CalibrationError, "reprojection"):
            calibrate_stereo(self.objects, self.images, self.right_images[::-1],
                             self.camera, self.camera)
        with self.assertRaises(CalibrationError):
            calibrate_stereo(self.objects, self.images, self.right_images[:-1],
                             self.camera, self.camera)
        with self.assertRaises(CalibrationError):
            calibrate_stereo(self.objects, self.images, self.right_images, self.camera,
                             CameraCalibration(MATRIX, DISTORTION, (640, 480)))

    def test_stereo_rejects_single_corrupted_capture_even_when_overall_rms_is_low(self):
        corrupted = [image.copy() for image in self.right_images]
        corrupted[-1][0] += [12, -12]
        with self.assertRaisesRegex(CalibrationError, "reprojection"):
            calibrate_stereo(self.objects, self.images, corrupted, self.camera, self.camera)

    def test_rejects_nonmonotonic_and_folded_lens_models(self):
        # A negative radial slope maps different scene rays to the same pixels.
        for distortion, message in (([-1, 0, 0, 0, 0], "non-monotonic"),
                                    ([0, 0, 1, 1, 0], "folds")):
            with self.subTest(distortion=distortion), self.assertRaisesRegex(CalibrationError, message):
                CameraCalibration(MATRIX, distortion, IMAGE_SIZE)

    def test_rejects_unrepresentable_coordinates_without_runtime_warnings(self):
        import warnings
        for factor in (1e100, 1e308):
            objects = [view * factor for view in self.objects]
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                with self.assertRaises(CalibrationError):
                    calibrate_camera(objects, self.images, IMAGE_SIZE)

    def test_camera_model_rejects_bad_intrinsics_distortion_and_transforms(self):
        for matrix in (np.eye(4), np.zeros((3, 3)), MATRIX * np.nan,
                       MATRIX + [[0, 1, 0], [0, 0, 0], [0, 0, 0]]):
            with self.assertRaises(CalibrationError):
                CameraCalibration(matrix, DISTORTION, IMAGE_SIZE)
        for distortion in ([0, 0, 0], [0, 0, 0, np.inf], np.zeros((2, 2))):
            with self.assertRaises(CalibrationError):
                CameraCalibration(MATRIX, distortion, IMAGE_SIZE)
        for rotation, translation in ((np.zeros((3, 3)), RIGHT_TRANSLATION),
                                      (np.diag([1, 1, -1]), RIGHT_TRANSLATION),
                                      (np.eye(3), np.zeros(3)), (np.eye(3), [1, 2])):
            with self.assertRaises(CalibrationError):
                StereoCalibration(rotation, translation, 0)


class StereoTriangulationTests(unittest.TestCase):
    def setUp(self):
        self.camera = CameraCalibration(MATRIX, DISTORTION, IMAGE_SIZE)
        self.points = np.array([[-0.2, -0.3, 2.0], [0.15, 0.3, 3.2], [0.4, -0.1, 4.0]])
        self.left, self.right = self.project(self.points)

    def project(self, points):
        left = cv2.projectPoints(points, np.zeros(3), np.zeros(3), MATRIX, DISTORTION)[0]
        right = cv2.projectPoints(points, cv2.Rodrigues(RIGHT_ROTATION)[0],
                                  RIGHT_TRANSLATION, MATRIX, DISTORTION)[0]
        return left, right

    def triangulate(self, left=None, right=None, **kwargs):
        return triangulate_stereo_points(
            self.left if left is None else left, self.right if right is None else right,
            self.camera, self.camera, RIGHT_ROTATION, RIGHT_TRANSLATION, **kwargs,
        )

    def test_recovers_held_out_metric_points_with_distortion(self):
        np.testing.assert_allclose(self.triangulate(), self.points, atol=1e-8)

    def test_rejects_negative_disparity_behind_camera(self):
        left, right = self.project(np.array([[0.0, 0.0, -3.0]]))
        with self.assertRaisesRegex(CalibrationError, "behind"):
            self.triangulate(left, right)

    def test_rejects_parallel_and_nearly_parallel_rays(self):
        left, right = self.project(np.array([[0.0, 0.0, 10000.0]]))
        with self.assertRaisesRegex(CalibrationError, "parallel"):
            self.triangulate(left, right)
        identical_camera = CameraCalibration(MATRIX, np.zeros(5), IMAGE_SIZE)
        with self.assertRaisesRegex(CalibrationError, "parallel"):
            triangulate_stereo_points([[640, 480]], [[640, 480]], identical_camera,
                                      identical_camera, np.eye(3), [-0.24, 0, 0])

    def test_rejects_mismatched_correspondence_and_high_residual(self):
        right = self.right.copy()
        right[0, 0, 1] += 20
        with self.assertRaisesRegex(CalibrationError, "reprojection"):
            self.triangulate(right=right)
        for left, right in ((self.left[:-1], self.right), ([], []),
                            ([[np.nan, 0]], [[1, 2]]), ([[-1, 200]], [[10, 200]])):
            with self.assertRaises(CalibrationError):
                self.triangulate(left, right)

    def test_rejects_invalid_quality_thresholds(self):
        for angle in (0, 90, 180, np.nan):
            with self.assertRaises(CalibrationError):
                self.triangulate(min_ray_angle_degrees=angle)
        with self.assertRaises(CalibrationError):
            self.triangulate(max_reprojection_error_px=0)


if __name__ == '__main__':
    unittest.main()
