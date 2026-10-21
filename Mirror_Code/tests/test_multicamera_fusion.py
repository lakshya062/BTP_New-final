"""Identity safety, occlusion recovery and packet contracts on projected poses.

The fixtures supply synthetic detections, not image-model identity ground truth.
Failures intentionally sacrifice availability to avoid merging different people.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import unittest

import numpy as np

from core.camera_calibration import CameraCalibration
from core.multicamera_fusion import FusedPacket, FusedPerson, MultiCameraFusion
from core.multicamera_types import CalibratedCamera, CameraFrame, FrameBundle, PersonObservation
from core.synthetic_pose import project_pose


class MultiCameraFusionTest(unittest.TestCase):
    def setUp(self):
        calibration = CameraCalibration([[800, 0, 640], [0, 800, 360], [0, 0, 1]], np.zeros(5), (1280, 720))
        self.cameras = {name: CalibratedCamera(name, calibration, np.eye(3), np.array([-x, 0, 0]))
                        for name, x in (("left", -0.6), ("middle", 0), ("right", 0.6))}
        self.pose = np.array([
            [0, -0.55, 4], [-0.2, -0.25, 4.05], [0.2, -0.25, 4.05],
            [-0.35, 0, 3.95], [0.3, 0.05, 3.85], [-0.12, 0.3, 4],
            [0.12, 0.3, 4], [0.13, 0.65, 4.15],
        ])
        self.fusion = MultiCameraFusion(self.cameras)

    def observation(self, camera_id, local_id="track", identity="alice", *, pose=None,
                    hidden=(), identity_confidence=0.99):
        camera = self.cameras[camera_id]
        pixels = project_pose(self.pose if pose is None else pose, camera.calibration.camera_matrix,
                              camera.calibration.distortion_coefficients,
                              translation_vector=camera.translation_m)
        confidence = np.ones(len(pixels))
        pixels[list(hidden)] = np.nan
        confidence[list(hidden)] = 0
        return PersonObservation(local_id, pixels, confidence, identity,
                                 identity_confidence if identity is not None else 0)

    def bundle(self, observations, *, timestamp=10.0, sequence=1, bundle_id="bundle-1",
               captures=None, arrivals=None, missing=None):
        frames = tuple(CameraFrame(camera, sequence,
                                   captures.get(camera, timestamp) if captures else timestamp,
                                   arrivals.get(camera, timestamp + 0.02) if arrivals else timestamp + 0.02,
                                   tuple(people)) for camera, people in observations.items())
        corrected = [frame.captured_at_seconds + self.cameras[frame.camera_id].clock_offset_seconds for frame in frames]
        return FrameBundle(bundle_id, timestamp, frames,
                           tuple(sorted(set(self.cameras) - set(observations))) if missing is None else tuple(missing),
                           max(corrected) - min(corrected))

    def test_three_views_merge_to_one_known_person_and_metric_joints(self):
        bundle = self.bundle({camera: [self.observation(camera)] for camera in self.cameras})
        packet = self.fusion.fuse(bundle)
        self.assertEqual(len(packet.persons), 1)
        person = packet.persons[0]
        self.assertEqual((person.identity_id, person.identity_status), ("alice", "confirmed"))
        self.assertTrue(person.valid_mask.all())
        np.testing.assert_allclose(person.world_points, self.pose, atol=1e-7)
        self.assertTrue(all(len(support) == 3 for support in person.joint_support))
        self.assertEqual(len(person.observation_keys), 3)

    def test_complementary_occlusion_recovers_only_joints_with_two_visible_views(self):
        packet = self.fusion.fuse(self.bundle({
            "left": [self.observation("left", hidden=(0, 1))],
            "middle": [self.observation("middle", hidden=(2,))],
            "right": [self.observation("right", hidden=(0, 3))],
        }))
        person = packet.persons[0]
        self.assertEqual(len(packet.persons), 1)
        self.assertFalse(person.valid_mask[0])  # Only the middle camera sees it.
        self.assertTrue(person.valid_mask[1:].all())
        self.assertEqual(set(person.joint_support[1]), {"middle", "right"})
        np.testing.assert_allclose(person.world_points[1:], self.pose[1:], atol=1e-7)

    def test_head_identity_occlusion_attaches_unknown_view_by_unique_body_geometry(self):
        packet = self.fusion.fuse(self.bundle({
            "left": [self.observation("left", hidden=(0,))],
            "middle": [self.observation("middle", identity=None, hidden=(0,))],
            "right": [self.observation("right", identity=None)],
        }))
        self.assertEqual(len(packet.persons), 1)
        self.assertEqual(packet.persons[0].identity_id, "alice")
        self.assertEqual(len(packet.persons[0].observation_keys), 3)
        self.assertFalse(packet.persons[0].valid_mask[0])

    def test_one_camera_blackout_reconstructs_from_two_remaining_views(self):
        packet = self.fusion.fuse(self.bundle({camera: [self.observation(camera)] for camera in ("left", "right")}))
        self.assertEqual(packet.missing_cameras, ("middle",))
        self.assertIn("incomplete_camera_bundle", packet.reasons)
        np.testing.assert_allclose(packet.persons[0].world_points, self.pose, atol=1e-7)

    def test_two_camera_blackout_reports_unavailable_depth_without_fabrication(self):
        packet = self.fusion.fuse(self.bundle({"left": [self.observation("left")]}))
        self.assertFalse(packet.persons[0].valid_mask.any())
        self.assertTrue(np.isnan(packet.persons[0].world_points).all())
        self.assertTrue(all(point is None for point in packet.to_dict()["persons"][0]["world_points_m"]))

    def test_two_people_are_separate_even_when_local_tracker_ids_are_equal(self):
        second = self.pose + np.array([0.05, 0.8, 0.2])
        packet = self.fusion.fuse(self.bundle({camera: [
            self.observation(camera, local_id="local-A", identity="alice"),
            self.observation(camera, local_id="local-B", identity="bob", pose=second),
        ] for camera in self.cameras}))
        self.assertEqual(len(packet.persons), 2)
        people = {person.identity_id: person for person in packet.persons}
        self.assertEqual(set(people), {"alice", "bob"})
        np.testing.assert_allclose(people["alice"].world_points, self.pose, atol=1e-7)
        np.testing.assert_allclose(people["bob"].world_points, second, atol=1e-7)

    def test_local_track_ids_may_differ_across_views(self):
        packet = self.fusion.fuse(self.bundle({camera: [self.observation(camera, local_id=f"{i}-unrelated")]
                                              for i, camera in enumerate(self.cameras)}))
        self.assertEqual(len(packet.persons), 1)
        self.assertEqual(len(packet.persons[0].observation_keys), 3)

    def test_anonymous_person_can_merge_without_becoming_a_known_user(self):
        packet = self.fusion.fuse(self.bundle({camera: [self.observation(camera, identity=None)] for camera in self.cameras}))
        self.assertEqual(len(packet.persons), 1)
        self.assertIsNone(packet.persons[0].identity_id)
        self.assertEqual(packet.persons[0].identity_status, "anonymous")
        self.assertTrue(packet.persons[0].valid_mask.all())

    def test_two_anonymous_people_merge_only_with_unique_geometric_correspondences(self):
        second = self.pose + [0, 0.9, 0.2]
        packet = self.fusion.fuse(self.bundle({camera: [
            self.observation(camera, "unknown-1", None),
            self.observation(camera, "unknown-2", None, pose=second),
        ] for camera in self.cameras}))
        self.assertEqual(len(packet.persons), 2)
        self.assertTrue(all(person.identity_id is None and person.valid_mask.all() for person in packet.persons))

    def test_different_high_confidence_face_ids_never_merge(self):
        packet = self.fusion.fuse(self.bundle({
            "left": [self.observation("left", identity="alice")],
            "right": [self.observation("right", identity="bob")],
        }))
        self.assertEqual(len(packet.persons), 2)
        self.assertTrue(all(not person.valid_mask.any() for person in packet.persons))

    def test_low_confidence_label_is_not_promoted_to_identity(self):
        packet = self.fusion.fuse(self.bundle({camera: [self.observation(camera, identity="guess", identity_confidence=0.4)]
                                              for camera in self.cameras}))
        self.assertEqual(len(packet.persons), 1)
        self.assertIsNone(packet.persons[0].identity_id)

    def test_duplicate_known_identity_in_one_camera_is_quarantined(self):
        packet = self.fusion.fuse(self.bundle({
            "left": [self.observation("left", "track-1"), self.observation("left", "track-2", pose=self.pose + [0, 0.7, 0])],
            "right": [self.observation("right")],
        }))
        self.assertEqual(len(packet.persons), 3)
        self.assertTrue(all(person.identity_status == "ambiguous" and person.identity_id is None for person in packet.persons))
        self.assertTrue(all("duplicate_identity_in_camera" in person.reasons for person in packet.persons))

    def test_same_identity_with_incompatible_geometry_is_quarantined(self):
        packet = self.fusion.fuse(self.bundle({
            "left": [self.observation("left")],
            "right": [self.observation("right", pose=self.pose + [0, 0.6, 0])],
        }))
        self.assertEqual(len(packet.persons), 2)
        self.assertTrue(all(person.identity_status == "ambiguous" for person in packet.persons))
        self.assertTrue(all("identity_geometry_conflict" in person.reasons for person in packet.persons))

    def test_swapped_labels_between_people_fail_closed(self):
        second = self.pose + [0, 0.8, 0.2]
        packet = self.fusion.fuse(self.bundle({
            "left": [self.observation("left", "l1", "alice"), self.observation("left", "l2", "bob", pose=second)],
            "right": [self.observation("right", "r1", "bob"), self.observation("right", "r2", "alice", pose=second)],
        }))
        self.assertEqual(len(packet.persons), 4)
        self.assertTrue(all(person.identity_id is None and not person.valid_mask.any() for person in packet.persons))

    def test_same_height_crossing_with_swapped_labels_rejects_equally_plausible_geometry(self):
        second = self.pose + [0.2, 0, 0]
        packet = self.fusion.fuse(self.bundle({
            "left": [self.observation("left", "l1", "alice"), self.observation("left", "l2", "bob", pose=second)],
            "right": [self.observation("right", "r1", "bob"), self.observation("right", "r2", "alice", pose=second)],
        }))
        self.assertEqual(len(packet.persons), 4)
        self.assertTrue(all(person.identity_id is None for person in packet.persons))
        self.assertTrue(all("ambiguous_identity_geometry" in person.reasons for person in packet.persons))

    def test_three_views_recover_unique_two_camera_consensus_when_third_label_swaps(self):
        second = self.pose + [0, 0.8, 0.2]
        observations = {camera: [
            self.observation(camera, f"{camera}-one", "alice"),
            self.observation(camera, f"{camera}-two", "bob", pose=second),
        ] for camera in self.cameras}
        observations["right"] = [
            self.observation("right", "right-one", "bob"),
            self.observation("right", "right-two", "alice", pose=second),
        ]
        packet = self.fusion.fuse(self.bundle(observations))
        known = {person.identity_id: person for person in packet.persons if person.identity_id is not None}
        self.assertEqual(set(known), {"alice", "bob"})
        self.assertTrue(all(len(person.observation_keys) == 2 for person in known.values()))
        np.testing.assert_allclose(known["alice"].world_points, self.pose, atol=1e-7)
        np.testing.assert_allclose(known["bob"].world_points, second, atol=1e-7)
        rejected = [person for person in packet.persons if person.identity_status == "ambiguous"]
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all("quarantined_inconsistent_camera" in person.reasons for person in rejected))

    def test_three_view_same_height_swap_remains_ambiguous_despite_majority_labels(self):
        second = self.pose + [0.2, 0, 0]
        observations = {camera: [self.observation(camera, "one", "alice"),
                                 self.observation(camera, "two", "bob", pose=second)]
                        for camera in self.cameras}
        observations["right"] = [self.observation("right", "one", "bob"),
                                 self.observation("right", "two", "alice", pose=second)]
        packet = self.fusion.fuse(self.bundle(observations))
        self.assertEqual(len(packet.persons), 6)
        self.assertTrue(all(person.identity_id is None and not person.valid_mask.any() for person in packet.persons))

    def test_one_bad_joint_in_one_view_does_not_contaminate_fused_pose(self):
        observations = {camera: [self.observation(camera)] for camera in self.cameras}
        corrupted = observations["right"][0]
        pixels = corrupted.landmarks_px.copy()
        pixels[0] += [80, -55]
        observations["right"] = [replace(corrupted, landmarks_px=pixels)]
        packet = self.fusion.fuse(self.bundle(observations))
        self.assertEqual(len(packet.persons), 1)
        np.testing.assert_allclose(packet.persons[0].world_points, self.pose, atol=1e-7)
        self.assertNotIn("right", packet.persons[0].joint_support[0])

    def test_unique_two_camera_consensus_survives_an_entire_bad_view(self):
        packet = self.fusion.fuse(self.bundle({
            "left": [self.observation("left")], "middle": [self.observation("middle")],
            "right": [self.observation("right", pose=self.pose + [0, 0.7, 0])],
        }))
        known = [person for person in packet.persons if person.identity_id == "alice"]
        self.assertEqual(len(known), 1)
        np.testing.assert_allclose(known[0].world_points, self.pose, atol=1e-7)
        self.assertEqual(len(known[0].observation_keys), 2)
        self.assertEqual(len(packet.persons), 2)
        rejected = next(person for person in packet.persons if person.identity_id is None)
        self.assertIn("quarantined_inconsistent_camera", rejected.reasons)

    def test_anonymous_same_height_crossing_is_not_arbitrarily_assigned(self):
        second = self.pose + [0.2, 0, 0]
        packet = self.fusion.fuse(self.bundle({camera: [
            self.observation(camera, "one", None), self.observation(camera, "two", None, pose=second),
        ] for camera in ("left", "right")}))
        self.assertEqual(len(packet.persons), 4)
        self.assertTrue(all(person.identity_status == "ambiguous" for person in packet.persons))

    def test_unknown_view_cannot_choose_between_two_plausible_known_people(self):
        packet = self.fusion.fuse(self.bundle({
            "left": [self.observation("left", "one", "alice"), self.observation("left", "two", "bob", pose=self.pose + [0.2, 0, 0])],
            "right": [self.observation("right", "unknown", None)],
        }))
        self.assertEqual(len(packet.persons), 3)
        unknown = next(person for person in packet.persons if person.identity_id is None)
        self.assertIn("ambiguous_identity_attachment", unknown.reasons)
        self.assertTrue(all(not person.valid_mask.any() for person in packet.persons))

    def test_two_unknown_observations_cannot_attach_to_one_known_person(self):
        packet = self.fusion.fuse(self.bundle({
            "left": [self.observation("left")],
            "right": [self.observation("right", "one", None), self.observation("right", "two", None, pose=self.pose + [0.2, 0, 0])],
        }))
        self.assertEqual(len(packet.persons), 3)
        self.assertTrue(all(len(person.observation_keys) == 1 for person in packet.persons))

    def test_local_track_reuse_does_not_inherit_prior_face_identity(self):
        first = self.fusion.fuse(self.bundle({camera: [self.observation(camera, "reused")] for camera in self.cameras}))
        second = self.fusion.fuse(self.bundle({camera: [self.observation(camera, "reused", None)] for camera in self.cameras},
                                              timestamp=10000, sequence=2, bundle_id="later"))
        self.assertEqual(first.persons[0].identity_id, "alice")
        self.assertIsNone(second.persons[0].identity_id)
        self.assertNotEqual(first.persons[0].person_id, second.persons[0].person_id)

    def test_clock_offsets_are_added_before_comparing_capture_times(self):
        self.cameras["right"] = replace(self.cameras["right"], clock_offset_seconds=0.3)
        fusion = MultiCameraFusion(self.cameras)
        bundle = self.bundle({camera: [self.observation(camera)] for camera in ("left", "right")},
                             captures={"left": 10, "right": 9.7})
        self.assertEqual(len(fusion.fuse(bundle).persons), 1)

    def test_arrival_jitter_does_not_replace_capture_timestamps(self):
        packet = self.fusion.fuse(self.bundle({camera: [self.observation(camera)] for camera in ("left", "right")},
                                               arrivals={"left": 10.01, "right": 20.0}))
        self.assertTrue(packet.persons[0].valid_mask.all())

    def test_forged_span_cannot_hide_unsynchronized_capture_times(self):
        bundle = self.bundle({camera: [self.observation(camera)] for camera in ("left", "right")},
                             captures={"left": 10, "right": 10.2})
        with self.assertRaisesRegex(ValueError, "synchronization window"):
            self.fusion.fuse(replace(bundle, span_seconds=0))

    def test_inconsistent_target_timestamp_is_rejected(self):
        bundle = self.bundle({"left": [self.observation("left")]})
        with self.assertRaisesRegex(ValueError, "target timestamp"):
            self.fusion.fuse(replace(bundle, target_timestamp_seconds=20))

    def test_large_clock_uncertainty_is_rejected(self):
        self.cameras["right"] = replace(self.cameras["right"], clock_uncertainty_seconds=0.05)
        fusion = MultiCameraFusion(self.cameras)
        bundle = self.bundle({camera: [self.observation(camera)] for camera in ("left", "right")})
        with self.assertRaisesRegex(ValueError, "synchronization window"):
            fusion.fuse(bundle)

    def test_empty_detection_frames_produce_empty_packet(self):
        packet = self.fusion.fuse(self.bundle({"left": [], "right": []}))
        self.assertEqual(packet.persons, ())
        self.assertEqual(json.loads(packet.to_json())["persons"], [])

    def test_unconfigured_camera_and_unknown_missing_camera_are_rejected(self):
        bundle = self.bundle({"left": [self.observation("left")]})
        for changed in (replace(bundle, missing_cameras=("unconfigured",)),
                        replace(bundle, frames=(replace(bundle.frames[0], camera_id="unconfigured"),), missing_cameras=())):
            with self.assertRaisesRegex(ValueError, "camera"):
                self.fusion.fuse(changed)

    def test_bounded_association_workload_rejects_oversized_bundle(self):
        fusion = MultiCameraFusion(self.cameras, max_observations=2)
        bundle = self.bundle({camera: [self.observation(camera)] for camera in self.cameras})
        with self.assertRaisesRegex(ValueError, "workload"):
            fusion.fuse(bundle)

    def test_packet_is_deterministic_for_camera_and_detection_input_order(self):
        observations = {camera: [self.observation(camera)] for camera in self.cameras}
        first = self.fusion.fuse(self.bundle(observations))
        second = self.fusion.fuse(self.bundle(dict(reversed(list(observations.items())))))
        self.assertEqual(first.to_json(), second.to_json())

    def test_each_input_observation_appears_exactly_once_in_output(self):
        second = self.pose + [0.2, 0, 0]
        bundle = self.bundle({camera: [self.observation(camera, "one", None), self.observation(camera, "two", None, pose=second)]
                              for camera in self.cameras})
        packet = self.fusion.fuse(bundle)
        keys = [key for person in packet.persons for key in person.observation_keys]
        self.assertEqual(len(keys), 6)
        self.assertEqual(len(keys), len(set(keys)))

    def test_shared_fusion_instance_is_stateless_under_concurrent_calls(self):
        bundle = self.bundle({camera: [self.observation(camera)] for camera in self.cameras})
        with ThreadPoolExecutor(max_workers=4) as pool:
            packets = list(pool.map(lambda _: self.fusion.fuse(bundle).to_json(), range(8)))
        self.assertEqual(len(set(packets)), 1)

    def test_packet_arrays_are_readonly_and_json_has_no_nonfinite_literals(self):
        person = self.fusion.fuse(self.bundle({"left": [self.observation("left")]})).persons[0]
        with self.assertRaises(ValueError):
            person.world_points[0, 0] = 3
        with self.assertRaises(ValueError):
            person.valid_mask[0] = True
        payload = FusedPacket("packet", 1.0, (person,)).to_json()
        self.assertNotIn("NaN", payload)
        self.assertNotIn("Infinity", payload)

    def test_invalid_fusion_configuration_is_rejected(self):
        for options in ({"identity_threshold": 0}, {"max_span_seconds": -1}, {"min_shared_joints": 2},
                        {"min_confidence": 1.1}, {"max_epipolar_error_px": 0}, {"max_observations": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                MultiCameraFusion(self.cameras, **options)

    def test_fused_output_rejects_claiming_valid_nonfinite_geometry(self):
        with self.assertRaises(ValueError):
            FusedPerson("test", "alice", "confirmed", np.full((3, 3), np.nan), np.ones(3, bool), ((), (), ()), (), ())
        with self.assertRaises(ValueError):
            FusedPerson("test", "alice", "anonymous", np.zeros((3, 3)), np.ones(3, bool), ((), (), ()), (), ())

    def test_fused_person_strictly_validates_external_output_contract(self):
        person = self.fusion.fuse(self.bundle({camera: [self.observation(camera)] for camera in self.cameras})).persons[0]
        invalid_changes = (
            {"person_id": ""}, {"identity_id": " "}, {"valid_mask": np.ones(8)},
            {"world_points": np.full((8, 3), np.inf)}, {"world_points": np.zeros((2, 3)), "valid_mask": np.ones(2, bool)},
            {"observation_keys": ()}, {"observation_keys": (("left", "epoch", True, "track"),)},
            {"observation_keys": (("left", "epoch", -1, "track"),)},
            {"observation_keys": (("left", "epoch", 2**63, "track"),)},
            {"observation_keys": (("left", "epoch", 1, ""),)},
            {"observation_keys": (("left", "epoch", 1, "one"), ("left", "epoch", 1, "two"))},
            {"joint_support": (("left",),) * 8}, {"joint_support": (("left", "unconfigured"),) * 8},
            {"joint_support": (("left", "left"),) * 8}, {"joint_support": None},
            {"reasons": ("duplicate", "duplicate")}, {"reasons": "a-string-is-not-a-sequence"},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(person, **changes)

    def test_fused_packet_strictly_validates_external_output_contract(self):
        packet = self.fusion.fuse(self.bundle({camera: [self.observation(camera)] for camera in self.cameras}))
        person = packet.persons[0]
        other = replace(person, person_id="different-person")
        invalid_changes = (
            {"bundle_id": ""}, {"timestamp_seconds": True}, {"timestamp_seconds": -1}, {"timestamp_seconds": np.inf},
            {"persons": (object(),)}, {"persons": None}, {"persons": (person, person)},
            {"persons": (person, other)}, {"missing_cameras": ("left",)},
            {"missing_cameras": ("missing", "missing")}, {"reasons": ("bad\nreason",)},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(packet, **changes)

    def test_bundle_cannot_silently_omit_a_configured_camera(self):
        bundle = self.bundle({"left": [self.observation("left")]})
        with self.assertRaisesRegex(ValueError, "cover the configured rig"):
            self.fusion.fuse(replace(bundle, missing_cameras=()))

    def test_packet_rejects_two_frame_sequences_or_epochs_for_one_camera(self):
        packet = self.fusion.fuse(self.bundle({"left": [self.observation("left")]}))
        person = packet.persons[0]
        for key in (("left", "default", 2, "different-track"), ("left", "different-epoch", 1, "different-track")):
            other = replace(person, person_id="other", identity_id=None, identity_status="anonymous", observation_keys=(key,))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "one frame"):
                replace(packet, persons=(person, other))

    def test_packet_rejects_more_than_eight_observed_and_missing_cameras(self):
        packet = self.fusion.fuse(self.bundle({"left": [self.observation("left")]}))
        with self.assertRaisesRegex(ValueError, "eight configured cameras"):
            replace(packet, missing_cameras=tuple(f"missing-{i}" for i in range(8)))


if __name__ == "__main__":
    unittest.main()
