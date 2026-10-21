import unittest

import cv2
import numpy as np

from core.synthetic_camera import (
    SyntheticCamera,
    chessboard_object_points,
    default_camera,
    generate_chessboard_views,
    render_chessboard,
)


class SyntheticCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.camera = default_camera()
        cls.views = generate_chessboard_views(count=8)

    def test_chessboard_units_and_row_major_order(self):
        points = chessboard_object_points((9, 6), 0.03)
        self.assertEqual(points.shape, (54, 3))
        self.assertEqual(points.dtype, np.float32)
        np.testing.assert_allclose(points[0], (0, 0, 0))
        np.testing.assert_allclose(points[1], (0.03, 0, 0))
        np.testing.assert_allclose(points[9], (0, 0.03, 0))
        np.testing.assert_allclose(points[-1], (0.24, 0.15, 0))

    def test_real_detector_recovers_distorted_noisy_blurred_raster_corners(self):
        # Detection sees only uint8 camera images. Ground truth is consulted
        # afterwards to measure actual pixel recovery, never as solver input.
        for index, view in enumerate(self.views):
            with self.subTest(view=index):
                self.assertEqual(view.image.shape, (720, 960))
                self.assertEqual(view.image.dtype, np.uint8)
                found, detected = cv2.findChessboardCornersSB(
                    view.image, (9, 6), cv2.CALIB_CB_NORMALIZE_IMAGE,
                )
                self.assertTrue(found)
                actual = detected.reshape(-1, 2)
                expected = view.image_points.reshape(-1, 2)
                # A chessboard is ambiguous under a half turn. Either valid
                # row ordering describes the same physical camera calibration.
                errors = min(
                    (np.linalg.norm(actual - expected, axis=1),
                     np.linalg.norm(actual - expected[::-1], axis=1)),
                    key=np.mean,
                )
                self.assertLess(float(np.mean(errors)), 0.16)
                self.assertLess(float(np.max(errors)), 0.65)

    def test_noise_and_randomized_poses_are_reproducible(self):
        first = generate_chessboard_views(count=2, seed=1234)
        second = generate_chessboard_views(count=2, seed=1234)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.image, b.image)
            np.testing.assert_array_equal(a.rotation_vector, b.rotation_vector)
            np.testing.assert_array_equal(a.translation_vector, b.translation_vector)
        different = generate_chessboard_views(count=1, seed=1235)[0]
        self.assertFalse(np.array_equal(first[0].image, different.image))

    def test_views_cover_tilt_distance_and_image_position(self):
        rotations = np.array([view.rotation_vector.ravel() for view in self.views])
        centers = np.array([view.image_points.reshape(-1, 2).mean(axis=0)
                            for view in self.views])
        distances = np.array([view.translation_vector[2, 0] for view in self.views])
        self.assertGreater(float(np.ptp(rotations[:, 0])), 0.5)
        self.assertGreater(float(np.ptp(rotations[:, 1])), 0.4)
        self.assertGreater(float(np.ptp(distances)), 0.2)
        self.assertGreater(float(np.ptp(centers[:, 0])), 200)
        self.assertGreater(float(np.ptp(centers[:, 1])), 180)

    def test_lens_distortion_changes_actual_raster_images(self):
        view = self.views[0]
        ideal_camera = SyntheticCamera(
            self.camera.image_size, self.camera.camera_matrix,
            np.zeros(5, dtype=np.float64),
        )
        ideal = render_chessboard(
            ideal_camera, view.rotation_vector, view.translation_vector,
        )
        distorted = render_chessboard(
            self.camera, view.rotation_vector, view.translation_vector,
        )
        displacement = np.linalg.norm(
            ideal.image_points - distorted.image_points, axis=-1,
        )
        self.assertGreater(float(np.max(displacement)), 2.0)
        self.assertGreater(float(np.mean(np.abs(
            ideal.image.astype(float) - distorted.image.astype(float),
        ))), 0.5)

    def test_image_resolution_scales_intrinsics(self):
        camera = default_camera((480, 360))
        np.testing.assert_allclose(
            camera.camera_matrix[:2], self.camera.camera_matrix[:2] / 2,
        )
        view = generate_chessboard_views(count=1, camera=camera)[0]
        self.assertEqual(view.image.shape, (360, 480))
        found, _ = cv2.findChessboardCornersSB(view.image, (9, 6))
        self.assertTrue(found)

    def test_border_crossing_camera_plane_is_rejected_even_when_corners_are_in_front(self):
        rotation_vector = np.array([0, np.pi / 3, 0])
        translation = np.array([0, 0, 0.22])
        rotation, _ = cv2.Rodrigues(rotation_vector)
        inner_camera = chessboard_object_points() @ rotation.T + translation
        self.assertTrue(np.all(inner_camera[:, 2] > 0))
        with self.assertRaisesRegex(ValueError, "including its border"):
            render_chessboard(self.camera, rotation_vector, translation)

    def test_invalid_geometry_and_render_parameters_are_rejected(self):
        for pattern_size, square_size in [((1, 6), 0.03), ((9, 0), 0.03),
                                           ((9, 6), 0), ((9, 6), float("nan"))]:
            with self.subTest(pattern_size=pattern_size, square_size=square_size):
                with self.assertRaises(ValueError):
                    chessboard_object_points(pattern_size, square_size)
        with self.assertRaises(ValueError):
            default_camera((0, 720))
        with self.assertRaises(ValueError):
            generate_chessboard_views(count=0)
        with self.assertRaises(ValueError):
            render_chessboard(self.camera, [0, 0, 0], [0, 0, -1])
        with self.assertRaises(ValueError):
            render_chessboard(self.camera, [0, 0, 0], [0, 0, 1], noise_std=-1)


if __name__ == "__main__":
    unittest.main()
