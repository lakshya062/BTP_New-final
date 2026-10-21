"""Known 3D fixtures distinguish lens calibration from depth recovery."""

import math
import unittest

import cv2
import numpy as np

from core.synthetic_pose import (
    calculate_joint_angle_3d,
    generate_arm_pose,
    generate_arm_trajectory,
    project_pose,
    undistort_pose,
)
from core.utils import calculate_joint_angle


class SyntheticPoseTest(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[850.0, 0.0, 640.0], [0.0, 850.0, 360.0], [0.0, 0.0, 1.0]])
        self.translation = np.array([0.0, 0.0, 3.0])

    def test_generated_trajectories_keep_known_metric_angles_and_lengths(self):
        angles = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
        for yaw in (0, 30, 60, 80, 90, -45, 180):
            with self.subTest(yaw=yaw):
                poses = generate_arm_trajectory(angles, yaw)
                self.assertEqual(poses.shape, (7, 3, 3))
                np.testing.assert_allclose(
                    [calculate_joint_angle_3d(*pose) for pose in poses], angles, atol=1e-12
                )
                np.testing.assert_allclose(np.linalg.norm(poses[:, 0] - poses[:, 1], axis=1), 0.32)
                np.testing.assert_allclose(np.linalg.norm(poses[:, 2] - poses[:, 1], axis=1), 0.27)

    def test_metric_angle_is_invariant_to_rigid_camera_rotation_and_scale(self):
        pose = generate_arm_pose(123, 61)
        rotation, _ = cv2.Rodrigues(np.array([0.4, -0.6, 0.2]))
        transformed = (pose @ rotation.T) * 1000.0 + np.array([200, -123, 400])
        self.assertAlmostEqual(calculate_joint_angle_3d(*transformed), 123, places=11)
        self.assertAlmostEqual(calculate_joint_angle_3d(*pose[::-1]), 123, places=11)

    def test_camera_projection_matches_hand_computed_pinhole_coordinates(self):
        points = np.array([[1.0, 0.5, 0.0], [-0.5, -0.25, 1.0]])
        pixels = project_pose(points, self.matrix, translation_vector=self.translation)
        expected = np.array([[640 + 850 / 3, 360 + 425 / 3], [640 - 425 / 4, 360 - 212.5 / 4]])
        np.testing.assert_allclose(pixels, expected, atol=1e-12)

    def test_projection_respects_object_to_camera_rotation(self):
        point = np.array([[1.0, 0.0, 0.0]])
        pixels = project_pose(
            point, self.matrix, rotation_vector=[0, 0, math.pi / 2],
            translation_vector=self.translation,
        )
        np.testing.assert_allclose(pixels, [[640, 360 + 850 / 3]], atol=1e-12)

    def test_projection_and_undistortion_preserve_trajectory_and_single_point_shapes(self):
        for points in (generate_arm_trajectory([45, 60, 120]), generate_arm_pose(70), np.array([0, 0, 1])):
            with self.subTest(shape=points.shape):
                pixels = project_pose(points, self.matrix, translation_vector=self.translation)
                self.assertEqual(pixels.shape, points.shape[:-1] + (2,))
                normalized = undistort_pose(pixels, self.matrix)
                expected = points[..., :2] / (points[..., 2:] + 3)
                np.testing.assert_allclose(normalized, expected, atol=1e-12)

    def test_lens_and_pixel_aspect_correction_restore_camera_rays(self):
        matrix = self.matrix.copy()
        matrix[1, 1] = 1040.0
        distortion = np.array([-0.28, 0.08, 0.002, -0.001, 0.01])
        poses = generate_arm_trajectory([30, 60, 90, 120, 160], 55)
        translation = np.array([0.8, 0.5, 1.7])
        pixels = project_pose(poses, matrix, distortion, translation_vector=translation)
        normalized = undistort_pose(pixels, matrix, distortion)
        camera_points = poses + translation
        expected = camera_points[..., :2] / camera_points[..., 2:]
        np.testing.assert_allclose(normalized, expected, atol=1e-11)
        wrong_assumption = (pixels - matrix[:2, 2]) / np.diag(matrix)[:2]
        self.assertGreater(float(np.max(np.abs(wrong_assumption - expected))), 0.01)

    def test_side_on_lens_corrected_angles_match_truth(self):
        poses = generate_arm_trajectory([25, 60, 90, 120, 165])
        pixels = project_pose(poses, self.matrix, [-0.3, 0.07, 0.002, 0.001, 0],
                              translation_vector=self.translation)
        normalized = undistort_pose(pixels, self.matrix, [-0.3, 0.07, 0.002, 0.001, 0])
        np.testing.assert_allclose(
            [calculate_joint_angle(*pose) for pose in normalized],
            [calculate_joint_angle_3d(*pose) for pose in poses], atol=1e-9,
        )

    def test_calibration_alone_does_not_correct_out_of_plane_angle_bias(self):
        # Even exact intrinsics and perfect, noiseless correspondences cannot
        # recover lost depth. Bias increases as the motion plane turns away.
        errors = []
        for yaw in (0, 30, 60, 80, 90):
            pose = generate_arm_pose(60, yaw)
            pixels = project_pose(pose, self.matrix, translation_vector=self.translation)
            normalized = undistort_pose(pixels, self.matrix)
            errors.append(abs(calculate_joint_angle(*normalized) - calculate_joint_angle_3d(*pose)))
        self.assertLess(errors[0], 1e-9)
        self.assertGreater(errors[2], 19.0)
        self.assertGreater(errors[3], 40.0)
        self.assertAlmostEqual(errors[4], 60.0, places=10)
        self.assertTrue(all(next_error > error for error, next_error in zip(errors, errors[1:])))

    def test_distinct_3d_angles_can_have_identical_monocular_pixels(self):
        # Scale one landmark along its camera ray: the image is unchanged,
        # while its metric relationship to the other joints changes.
        first = np.array([[0, -1, 4], [0, 0, 4], [1, -1, 4]], dtype=float)
        second = first.copy()
        second[2] *= 2
        np.testing.assert_allclose(project_pose(first, self.matrix), project_pose(second, self.matrix))
        self.assertAlmostEqual(calculate_joint_angle_3d(*first), 45)
        self.assertGreater(calculate_joint_angle_3d(*second) - calculate_joint_angle_3d(*first), 20)

    def test_edge_on_right_angle_is_rejected_by_existing_2d_geometry(self):
        pose = generate_arm_pose(90, 90)
        pixels = project_pose(pose, self.matrix, translation_vector=self.translation)
        self.assertAlmostEqual(calculate_joint_angle_3d(*pose), 90)
        with self.assertRaises(ValueError):
            calculate_joint_angle(*pixels)

    def test_invalid_or_coincident_3d_points_are_rejected(self):
        for invalid in ([math.nan, 0, 1], [0, math.inf, 1], [0, 1], None, [0, 0, "x"]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                calculate_joint_angle_3d(invalid, [0, 0, 0], [1, 0, 0])
        for points in (([0, 0, 0], [0, 0, 0], [1, 0, 0]), ([1, 0, 0], [0, 0, 0], [0, 0, 0])):
            with self.assertRaises(ValueError):
                calculate_joint_angle_3d(*points)

    def test_invalid_pose_parameters_are_rejected(self):
        for kwargs in ({"angle_degrees": -1}, {"angle_degrees": 181}, {"angle_degrees": math.nan},
                       {"angle_degrees": 60, "plane_yaw_degrees": math.inf},
                       {"angle_degrees": 60, "upper_arm_m": 0},
                       {"angle_degrees": 60, "forearm_m": -0.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_arm_pose(**kwargs)
        for angles in ([], [[60, 90]], 90, [60, math.nan]):
            with self.subTest(angles=angles), self.assertRaises(ValueError):
                generate_arm_trajectory(angles)

    def test_projection_rejects_invalid_points_and_camera_geometry(self):
        pose = generate_arm_pose(60)
        for points in ([], [1, 2], [[1, 2, math.nan]], pose):
            # The valid pose here has its elbow on z=0 without translation.
            with self.subTest(points=points), self.assertRaises(ValueError):
                project_pose(points, self.matrix)
        with self.assertRaises(ValueError):
            project_pose(pose, self.matrix, translation_vector=[0, 0, -3])
        for matrix in (np.eye(2), np.diag([-1.0, 1.0, 1.0]), np.zeros((3, 3)),
                       [[1, 0.1, 0], [0, 1, 0], [0, 0, 1]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                project_pose(pose, matrix, translation_vector=self.translation)
        for coefficients in ([1, 2, 3], [[1, 2], [3, 4]], [0, 0, 0, math.nan]):
            with self.subTest(coefficients=coefficients), self.assertRaises(ValueError):
                project_pose(pose, self.matrix, coefficients, translation_vector=self.translation)
        for vector in ([0, 0], [0, 0, math.inf], np.ones((1, 1, 3))):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                project_pose(pose, self.matrix, translation_vector=vector)

    def test_undistortion_rejects_invalid_landmarks(self):
        for points in ([], [1, 2, 3], [[math.nan, 3]], [[math.inf, 3]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                undistort_pose(points, self.matrix)


if __name__ == "__main__":
    unittest.main()
