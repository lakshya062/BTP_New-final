"""World-space controls for occlusion, outliers, and unobservable depth."""

from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from core.camera_calibration import CalibrationError, CameraCalibration
from core.multicamera_geometry import (
    estimate_pair_compatibility, fuse_joint, fuse_pose,
)
from core.multicamera_types import CalibratedCamera, PersonObservation


MATRIX = np.array([[1000., 0, 640], [0, 980., 480], [0, 0, 1]])
DISTORTION = np.array([-0.15, 0.025, 0.001, -0.0008, 0.001])
CALIBRATION = CameraCalibration(MATRIX, DISTORTION, (1280, 960))
POINT = np.array([0.1, 0.15, 3.2])
POSE = np.array([[-0.2, -0.7, 3.0], [0.2, -0.7, 3.0], [-0.4, -0.3, 3.1],
                 [0.4, -0.3, 3.1], [-0.3, 0.1, 3.2], [0.3, 0.1, 3.2],
                 [-0.15, 0.8, 3.0], [0.15, 0.8, 3.0]])


def camera(camera_id, center, rvec=None):
    rotation = np.eye(3) if rvec is None else cv2.Rodrigues(np.asarray(rvec, dtype=float))[0]
    return CalibratedCamera(camera_id, CALIBRATION, rotation, -rotation @ np.asarray(center, dtype=float))


def rig():
    return {item.camera_id: item for item in (
        camera("a", [-0.7, 0, 0]), camera("b", [0.7, 0, 0]),
        camera("c", [0, -0.5, 0]), camera("d", [0, 0.5, 0]),
    )}


def pixels(points, cam):
    return cv2.projectPoints(np.asarray(points, dtype=float).reshape(-1, 3),
                             cv2.Rodrigues(cam.rotation)[0], cam.translation_m,
                             cam.calibration.camera_matrix,
                             cam.calibration.distortion_coefficients)[0][:, 0]


def joint_observations(cameras, point=POINT):
    return {key: (pixels(point, cam)[0], 0.95) for key, cam in cameras.items()}


def observation(cam, confidence=None, points=POSE):
    if confidence is None:
        confidence = np.full(len(points), .95)
    landmarks = pixels(points, cam)
    landmarks[np.asarray(confidence) == 0] = np.nan
    return PersonObservation("track", landmarks, confidence)


class MultiCameraGeometryTests(unittest.TestCase):
    def setUp(self):
        self.cameras = rig()

    def test_two_views_reconstruct_world_metres_with_lens_distortion(self):
        for first, second in (("a", "b"), ("a", "c"), ("b", "d")):
            cams = {key: self.cameras[key] for key in (first, second)}
            result = fuse_joint(joint_observations(cams), cams)
            self.assertEqual(result.reason, "ok")
            np.testing.assert_allclose(result.position_m, POINT, atol=1e-9)
            self.assertEqual(result.supporting_cameras, tuple(sorted(cams)))
            self.assertLess(result.reprojection_error_px, 1e-8)

    def test_rotated_extrinsics_return_common_world_coordinates(self):
        cams = {"left": camera("left", [-.7, .1, .1], [.05, .2, -.04]),
                "right": camera("right", [.7, -.1, .1], [-.03, -.2, .05])}
        result = fuse_joint(joint_observations(cams), cams)
        np.testing.assert_allclose(result.position_m, POINT, atol=1e-9)

    def test_four_views_use_every_consistent_camera(self):
        result = fuse_joint(joint_observations(self.cameras), self.cameras)
        np.testing.assert_allclose(result.position_m, POINT, atol=1e-9)
        self.assertEqual(result.supporting_cameras, ("a", "b", "c", "d"))
        with self.assertRaises(ValueError):
            result.position_m[0] = 100

    def test_one_view_and_total_occlusion_never_invent_depth(self):
        for observations in ({}, {"a": joint_observations(self.cameras)["a"]},
                             {key: (np.full(2, np.nan), 0) for key in self.cameras}):
            result = fuse_joint(observations, self.cameras)
            self.assertIsNone(result.position_m)
            self.assertEqual(result.supporting_cameras, ())
            self.assertEqual(result.reason, "insufficient_views")

    def test_low_confidence_hallucinations_do_not_corrupt_valid_views(self):
        observations = joint_observations(self.cameras)
        observations["d"] = ([5, 5], .49)
        observations["c"] = ([np.nan, np.nan], 0)
        result = fuse_joint(observations, self.cameras)
        np.testing.assert_allclose(result.position_m, POINT, atol=1e-9)
        self.assertEqual(result.supporting_cameras, ("a", "b"))

    def test_complementary_occlusion_is_fused_per_joint(self):
        cameras = {key: value for key, value in self.cameras.items() if key != "d"}
        confidence = {"a": np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=float),
                      "b": np.array([1, 0, 1, 1, 0, 1, 1, 0], dtype=float),
                      "c": np.array([1, 1, 0, 1, 1, 0, 0, 1], dtype=float)}
        result = fuse_pose({key: observation(cam, confidence[key]) for key, cam in cameras.items()}, cameras)
        self.assertEqual(len(result), len(POSE))
        for index in range(7):
            np.testing.assert_allclose(result[index].position_m, POSE[index], atol=1e-9)
            self.assertEqual(len(result[index].supporting_cameras), 2)
        self.assertIsNone(result[-1].position_m)

    def test_pose_can_be_partly_visible_in_every_camera(self):
        poses = {key: observation(cam) for key, cam in self.cameras.items()}
        result = fuse_pose(poses, self.cameras)
        np.testing.assert_allclose(np.array([item.position_m for item in result]), POSE, atol=1e-9)

    def test_one_confident_outlier_is_removed_by_three_view_consensus(self):
        observations = joint_observations(self.cameras)
        observations["d"] = (observations["d"][0] + [80, 50], 1)
        result = fuse_joint(observations, self.cameras)
        np.testing.assert_allclose(result.position_m, POINT, atol=1e-9)
        self.assertEqual(result.supporting_cameras, ("a", "b", "c"))

    def test_two_valid_views_can_recover_joint_with_one_off_epipolar_outlier(self):
        cameras = {key: value for key, value in self.cameras.items() if key != "d"}
        observations = joint_observations(cameras)
        observations["c"] = (observations["c"][0] + [40, 70], .99)
        result = fuse_joint(observations, cameras)
        np.testing.assert_allclose(result.position_m, POINT, atol=1e-9)
        self.assertEqual(result.supporting_cameras, ("a", "b"))

    def test_split_consensus_fails_closed_instead_of_guessing(self):
        cameras = {key: camera(key, [center, 0, 0])
                   for key, center in zip("abcd", [-.8, -.3, .3, .8])}
        observations = joint_observations(cameras)
        other = np.array([-.1, -.4, 3.7])
        for key in ("c", "d"):
            observations[key] = (pixels(other, cameras[key])[0], .95)
        result = fuse_joint(observations, cameras)
        self.assertIsNone(result.position_m)
        self.assertEqual(result.reason, "ambiguous_consensus")

    def test_mapping_order_does_not_change_consensus(self):
        observations = joint_observations(self.cameras)
        observations["d"] = (observations["d"][0] + [70, 35], 1)
        a = fuse_joint(observations, self.cameras)
        b = fuse_joint(dict(reversed(list(observations.items()))), dict(reversed(list(self.cameras.items()))))
        np.testing.assert_array_equal(a.position_m, b.position_m)
        self.assertEqual(a.supporting_cameras, b.supporting_cameras)

    def test_coincident_cameras_do_not_create_observable_depth(self):
        cameras = {"a": camera("a", [0, 0, 0]), "b": camera("b", [0, 0, 0], [0, .1, 0])}
        result = fuse_joint(joint_observations(cameras), cameras)
        self.assertIsNone(result.position_m)
        self.assertEqual(result.reason, "insufficient_views")

    def test_duplicate_optical_centres_do_not_outvote_other_cameras(self):
        cameras = {"a": camera("a", [-.7, 0, 0]), "b": camera("b", [-.7, 0, 0]),
                   "c": camera("c", [.7, 0, 0]), "d": camera("d", [.7, 0, 0])}
        observations = joint_observations(cameras)
        alternate = np.array([.5, .15, 3.2])
        observations["b"] = (pixels(alternate, cameras["b"])[0], .95)
        observations["d"] = (pixels(alternate, cameras["d"])[0], .95)
        self.assertIsNone(fuse_joint(observations, cameras).position_m)

    def test_negligible_baseline_and_parallel_rays_are_rejected(self):
        cameras = {"a": camera("a", [0, 0, 0]), "b": camera("b", [.001, 0, 0])}
        result = fuse_joint(joint_observations(cameras), cameras)
        self.assertEqual(result.reason, "insufficient_parallax")
        observations = {key: ([640, 480], .9) for key in self.cameras}
        self.assertEqual(fuse_joint(observations, self.cameras).reason, "insufficient_parallax")

    def test_points_behind_cameras_are_rejected(self):
        result = fuse_joint(joint_observations(self.cameras, [0, .1, -3.2]), self.cameras)
        self.assertIsNone(result.position_m)
        self.assertEqual(result.reason, "inconsistent_geometry")

    def test_gross_two_view_reprojection_mismatch_is_rejected(self):
        observations = joint_observations(self.cameras)
        observations = {"a": observations["a"], "b": (observations["b"][0] + [0, 80], .99)}
        result = fuse_joint(observations, self.cameras)
        self.assertIsNone(result.position_m)
        self.assertEqual(result.reason, "inconsistent_geometry")

    def test_subpixel_noise_has_bounded_world_error_across_multiple_seeds(self):
        errors = []
        for seed in range(40):
            rng = np.random.default_rng(seed)
            truth = rng.uniform([-.4, -.3, 2.3], [.4, .3, 4.2])
            observations = joint_observations(self.cameras, truth)
            observations = {key: (pixel + rng.normal(0, .5, 2), confidence)
                            for key, (pixel, confidence) in observations.items()}
            result = fuse_joint(observations, self.cameras)
            self.assertIsNotNone(result.position_m, (seed, result.reason))
            errors.append(np.linalg.norm(result.position_m - truth))
        self.assertLess(np.percentile(errors, 95), .01)

    def test_compatibility_accepts_correct_pose_under_distortion_and_occlusion(self):
        first = observation(self.cameras["a"], [0, 1, 1, 1, 1, 1, 1, 1])
        second = observation(self.cameras["b"], [1, 1, 1, 1, 1, 1, 0, 1])
        result = estimate_pair_compatibility(first, second, self.cameras["a"], self.cameras["b"])
        self.assertTrue(result.compatible, result)
        self.assertEqual(result.shared_joints, 6)
        self.assertLess(result.median_epipolar_error_px, 1e-8)

    def test_compatibility_requires_shared_joints(self):
        first = observation(self.cameras["a"], [1, 1, 1, 1, 1, 0, 0, 0])
        second = observation(self.cameras["b"], [0, 0, 0, 1, 1, 1, 1, 1])
        result = estimate_pair_compatibility(first, second, self.cameras["a"], self.cameras["b"])
        self.assertFalse(result.compatible)
        self.assertEqual(result.reason, "insufficient_shared_joints")

    def test_compatibility_rejects_wrong_person_and_bad_landmark_mapping(self):
        first = observation(self.cameras["a"])
        for points in (POSE + [0, .4, .2], POSE[::-1]):
            second = observation(self.cameras["b"], points=points)
            result = estimate_pair_compatibility(first, second, self.cameras["a"], self.cameras["b"])
            self.assertFalse(result.compatible, result)

    def test_compatibility_rejects_behind_camera_and_degenerate_baseline(self):
        first = observation(self.cameras["a"], points=POSE * [1, 1, -1])
        second = observation(self.cameras["b"], points=POSE * [1, 1, -1])
        self.assertFalse(estimate_pair_compatibility(first, second, self.cameras["a"], self.cameras["b"]).compatible)
        cam = camera("same", [-.7, 0, 0])
        result = estimate_pair_compatibility(observation(self.cameras["a"]), observation(cam), self.cameras["a"], cam)
        self.assertEqual(result.reason, "coincident_cameras")

    def test_epipolar_geometry_does_not_claim_identity_certainty(self):
        # Identical skeletons on the same epipolar lines can both fit. The caller
        # must reject ambiguous identities rather than treating this as identity.
        cameras = {"a": camera("a", [-.7, 0, 0]), "b": camera("b", [.7, 0, 0])}
        first = observation(cameras["a"])
        second = observation(cameras["b"], points=POSE + [.25, 0, 0])
        self.assertTrue(estimate_pair_compatibility(first, second, cameras["a"], cameras["b"]).compatible)

    def test_invalid_parameters_and_visible_inputs_raise(self):
        observations = joint_observations(self.cameras)
        for kwargs in ({"min_confidence": -1}, {"min_confidence": True},
                       {"max_reprojection_error_px": 0}, {"max_reprojection_error_px": np.inf},
                       {"min_ray_angle_degrees": 0}, {"min_ray_angle_degrees": 100}):
            with self.subTest(kwargs=kwargs), self.assertRaises(CalibrationError):
                fuse_joint(observations, self.cameras, **kwargs)
        for value in (([np.nan, 1], .9), ([1280, 500], .9), ([1, 1, 1], .9),
                      ([1, 1], np.nan), ([1, 1], -1), ([1, 1], 2)):
            with self.subTest(value=value), self.assertRaises(CalibrationError):
                fuse_joint({"a": value}, self.cameras)
        with self.assertRaises(CalibrationError):
            fuse_joint({"unknown": ([1, 1], .9)}, self.cameras)

    def test_mismatched_pose_schema_and_cardinality_raise(self):
        with self.assertRaises(CalibrationError):
            fuse_pose({"a": observation(self.cameras["a"]),
                       "b": observation(self.cameras["b"], points=POSE[:4])}, self.cameras)
        with self.assertRaises(CalibrationError):
            fuse_joint({}, {str(index): self.cameras["a"] for index in range(33)})
        with self.assertRaises(CalibrationError):
            fuse_pose({"a": SimpleNamespace(landmarks_px=np.zeros((257, 2)), confidence=np.ones(257))}, self.cameras)
        for minimum in (True, 0, 1.5, 257):
            with self.assertRaises(CalibrationError):
                estimate_pair_compatibility(observation(self.cameras["a"]), observation(self.cameras["b"]),
                                            self.cameras["a"], self.cameras["b"], min_shared_joints=minimum)


if __name__ == "__main__":
    unittest.main()
