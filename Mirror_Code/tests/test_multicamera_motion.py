"""Continuous-time motion and compound multi-person rig integration tests.

These fixtures project known 3D bodies into each camera at its *actual* exposure
time. They supply detections rather than exercising image pose/face models. The
true actor and exposure time are retained only by the fixture/evaluation layer;
the real synchronizer and fusion receive ordinary validated camera packets.
"""

from dataclasses import replace
import random
import unittest

import cv2
import numpy as np

from core.multicamera_fusion import MultiCameraFusion
from core.multicamera_sync import FrameSynchronizer
from core.multicamera_types import CameraFrame, PersonObservation
from core.synthetic_multicamera import body_pose, make_rig
from core.synthetic_pose import calculate_joint_angle_3d, project_pose


START = 1000.0


def moving_body(exposure_seconds, actor):
    # A four-hertz curl changes the wrist materially over a 20–40 ms clock error.
    return body_pose((exposure_seconds - START) * 480, actor, 121)


def fixed_body(exposure_seconds, actor):
    return body_pose(19, actor, 121)


class MotionFixture:
    def __init__(self, camera_ids=None, *, configured=None):
        truth = make_rig()
        self.truth_cameras = {key: value for key, value in truth.items() if camera_ids is None or key in camera_ids}
        self.cameras = dict(self.truth_cameras if configured is None else configured)
        self.truth = {}

    def frame(self, camera_id, sequence, exposure, actors=(0,), *, pose=moving_body,
              identified=True, epoch="motion-v1", delay=.01, clock_error=0.):
        camera = self.truth_cameras[camera_id]
        people = []
        for actor in actors:
            points = pose(exposure, actor)
            pixels = project_pose(points, camera.calibration.camera_matrix,
                                  camera.calibration.distortion_coefficients,
                                  cv2.Rodrigues(camera.rotation)[0], camera.translation_m)
            confidence = np.full(len(points), .99)
            hidden = np.any((pixels < 0) | (pixels >= camera.calibration.image_size), axis=1)
            pixels[hidden] = np.nan
            confidence[hidden] = 0
            # IDs deliberately differ between cameras; their actor mapping is
            # never made available to association or copied across cameras.
            local_id = f"{camera_id}-track-{(actor * 11 + len(camera_id)) % 37}"
            observation = PersonObservation(local_id, pixels, confidence,
                                            f"member-{actor}" if identified else None,
                                            .99 if identified else 0)
            people.append(observation)
            self.truth[camera_id, epoch, sequence, local_id] = (actor, exposure, points, observation)
        return CameraFrame(camera_id, sequence,
                           exposure - camera.clock_offset_seconds + clock_error,
                           exposure + delay, tuple(people), epoch)

    def engines(self):
        return (FrameSynchronizer(self.cameras.values(), max_skew_seconds=.012,
                                  reorder_horizon_seconds=.15, max_frame_age_seconds=.5,
                                  session_id="motion-test"),
                MultiCameraFusion(self.cameras, max_span_seconds=.012))

    def replay(self, frames):
        synchronizer, fusion = self.engines()
        bundles = []
        for frame in sorted(frames, key=lambda item: (item.received_at_seconds, item.camera_id, item.sequence)):
            bundles.extend(synchronizer.add_frame(frame))
        bundles.extend(synchronizer.flush())
        return synchronizer, tuple(bundles), tuple(fusion.fuse(bundle) for bundle in bundles)


class MultiCameraContinuousMotionTest(unittest.TestCase):
    def assert_safe(self, fixture, packets, *, exact_geometry=False):
        seen = set()
        for packet in packets:
            for person in packet.persons:
                sources = [fixture.truth[key] for key in person.observation_keys]
                self.assertEqual(len({source[0] for source in sources}), 1, "different people merged")
                actor = sources[0][0]
                if person.identity_id is not None:
                    self.assertEqual(person.identity_id, f"member-{actor}")
                for key in person.observation_keys:
                    self.assertNotIn(key, seen, "observation reused across fused people/packets")
                    seen.add(key)
                if person.identity_status == "ambiguous":
                    self.assertFalse(person.valid_mask.any(), "quarantine must not invent multiview depth")
                for joint in np.flatnonzero(person.valid_mask):
                    self.assertGreaterEqual(len(person.joint_support[joint]), 2)
                    available = {key[0] for key, source in zip(person.observation_keys, sources)
                                 if source[3].confidence[joint] >= .5}
                    self.assertLessEqual(set(person.joint_support[joint]), available)
                if exact_geometry and person.valid_mask.any():
                    self.assertLess(np.max(np.linalg.norm(person.world_points[person.valid_mask]
                                                         - sources[0][2][person.valid_mask], axis=1)), 1e-4)

    def assert_people_available(self, fixture, packet, expected_actors, *, minimum=25):
        recovered = set()
        for person in packet.persons:
            if int(person.valid_mask.sum()) >= minimum:
                recovered.add(fixture.truth[person.observation_keys[0]][0])
        self.assertEqual(recovered, set(expected_actors))

    def test_simultaneous_fast_motion_recovers_metric_pose_and_elbow_angle(self):
        fixture = MotionFixture()
        frames = [fixture.frame(camera, sequence, START + .017 + sequence / 30)
                  for sequence in range(5) for camera in fixture.cameras]
        sync, bundles, packets = fixture.replay(frames)
        self.assertEqual(len(bundles), 5)
        self.assertEqual(sync.stats["emitted_frames"], 20)
        self.assert_safe(fixture, packets, exact_geometry=True)
        for packet in packets:
            self.assert_people_available(fixture, packet, {0}, minimum=33)
            person, = packet.persons
            expected = moving_body(packet.timestamp_seconds, 0)
            self.assertLess(abs(calculate_joint_angle_3d(*person.world_points[[11, 13, 15]])
                                - calculate_joint_angle_3d(*expected[[11, 13, 15]])), 1e-3)

    def test_small_known_capture_skew_uses_distinct_exposure_poses(self):
        fixture = MotionFixture()
        shifts = dict(zip(fixture.cameras, (0., .0005, .001, .0015)))
        frames = [fixture.frame(camera, sequence, START + .017 + sequence / 30 + shifts[camera])
                  for sequence in range(4) for camera in fixture.cameras]
        _, bundles, packets = fixture.replay(frames)
        self.assertEqual(len(bundles), 4)
        self.assert_safe(fixture, packets)
        errors = []
        for packet in packets:
            self.assert_people_available(fixture, packet, {0}, minimum=30)
            person, = packet.persons
            self.assertTrue(person.valid_mask[15])
            expected = moving_body(packet.timestamp_seconds, 0)
            errors.append(np.linalg.norm(person.world_points[15] - expected[15]))
        self.assertGreater(max(errors), 1e-4, "fixture must expose real motion-induced time error")
        self.assertLess(max(errors), .020)

    def test_capture_separation_over_budget_never_creates_metric_depth(self):
        fixture = MotionFixture(("front", "right"))
        frames = [fixture.frame(camera, sequence, START + sequence * .08 + (.020 if camera == "right" else 0))
                  for sequence in range(3) for camera in fixture.cameras]
        _, bundles, packets = fixture.replay(frames)
        self.assertEqual(len(bundles), 6)
        self.assertTrue(all(len(bundle.frames) == 1 for bundle in bundles))
        self.assertTrue(all(not person.valid_mask.any() for packet in packets for person in packet.persons))
        self.assert_safe(fixture, packets)

    def test_uncertain_capture_clock_is_rejected_during_fast_motion(self):
        cameras = make_rig()
        configured = {key: replace(value, clock_uncertainty_seconds=.007) if key == "front" else value
                      for key, value in cameras.items() if key in ("front", "right")}
        fixture = MotionFixture(("front", "right"), configured=configured)
        frames = [fixture.frame(camera, sequence, START + .01 + sequence / 30)
                  for sequence in range(3) for camera in fixture.cameras]
        sync, bundles, packets = fixture.replay(frames)
        self.assertEqual(sync.stats["uncertain_clock_frames"], 3)
        self.assertEqual(len(bundles), 3)
        self.assertTrue(all(bundle.missing_cameras == ("front",) for bundle in bundles))
        self.assertTrue(all(not person.valid_mask.any() for packet in packets for person in packet.persons))

    def test_incorrect_static_clock_offset_can_destroy_otherwise_observable_depth(self):
        cameras = make_rig()
        configured = {key: replace(value, clock_offset_seconds=value.clock_offset_seconds + .020)
                      if key == "right" else value for key, value in cameras.items() if key in ("front", "right")}
        fixture = MotionFixture(("front", "right"), configured=configured)
        frames = [fixture.frame(camera, sequence, START + .02 + sequence * .08)
                  for sequence in range(3) for camera in fixture.cameras]
        _, bundles, packets = fixture.replay(frames)
        self.assertEqual(len(bundles), 6)
        self.assertTrue(all(not person.valid_mask.any() for packet in packets for person in packet.persons))

    def test_hidden_clock_error_exposes_fast_motion_degradation(self):
        cameras = make_rig()
        configured = {key: replace(value, clock_offset_seconds=value.clock_offset_seconds - .040)
                      if key == "right" else value for key, value in cameras.items() if key in ("front", "right")}
        fixture = MotionFixture(("front", "right"), configured=configured)
        exposure = START + .035
        frames = [fixture.frame("front", 0, exposure), fixture.frame("right", 0, exposure + .040)]
        _, bundles, packets = fixture.replay(frames)
        self.assertEqual(len(bundles), 1)
        self.assertLess(bundles[0].span_seconds, 1e-9, "wrong clock calibration conceals true exposure skew")
        self.assert_safe(fixture, packets)
        person, = packets[0].persons
        self.assertGreaterEqual(int(person.valid_mask.sum()), 25, "static torso still has observable geometry")
        wrist_error = (np.linalg.norm(person.world_points[15] - moving_body(exposure, 0)[15])
                       if person.valid_mask[15] else float("inf"))
        self.assertGreater(wrist_error, .020, "unsafe clock configuration must measurably degrade moving joint")

    def test_static_scene_cannot_reveal_a_concealed_exposure_offset(self):
        cameras = make_rig()
        configured = {key: replace(value, clock_offset_seconds=value.clock_offset_seconds - .040)
                      if key == "right" else value for key, value in cameras.items() if key in ("front", "right")}
        fixture = MotionFixture(("front", "right"), configured=configured)
        frames = [fixture.frame("front", 0, START, pose=fixed_body),
                  fixture.frame("right", 0, START + .040, pose=fixed_body)]
        _, bundles, packets = fixture.replay(frames)
        self.assertEqual(len(bundles), 1)
        self.assert_safe(fixture, packets, exact_geometry=True)
        self.assert_people_available(fixture, packets[0], {0}, minimum=33)

    def test_uncompensated_clock_drift_reduces_availability_instead_of_reusing_views(self):
        fixture = MotionFixture(("front", "right"))
        frames = [fixture.frame(camera, sequence, START + sequence * .08,
                                clock_error=sequence * .006 if camera == "right" else 0.)
                  for sequence in range(4) for camera in fixture.cameras]
        sync, bundles, packets = fixture.replay(frames)
        self.assertTrue(any(len(bundle.frames) == 2 for bundle in bundles))
        self.assertTrue(any(len(bundle.frames) == 1 for bundle in bundles))
        self.assertTrue(all(bundle.span_seconds <= .012 + 1e-12 for bundle in bundles))
        self.assertEqual(sync.stats["emitted_frames"], 8)
        self.assert_safe(fixture, packets, exact_geometry=True)

    def test_three_people_reconstruct_with_complementary_whole_person_view_misses(self):
        fixture = MotionFixture()
        misses = {"front": 0, "left": 1, "right": 2, "high": None}
        frames = [fixture.frame(camera, sequence, START + .015 + sequence / 30,
                                tuple(actor for actor in range(3) if actor != misses[camera]))
                  for sequence in range(3) for camera in fixture.cameras]
        _, _, packets = fixture.replay(frames)
        self.assertEqual(len(packets), 3)
        self.assert_safe(fixture, packets, exact_geometry=True)
        for packet in packets:
            self.assert_people_available(fixture, packet, {0, 1, 2}, minimum=33)

    def test_three_people_enter_and_leave_without_ghost_people_or_identity_transfer(self):
        fixture = MotionFixture()
        active = ((0,), (0, 1), (1, 2), (2,))
        frames = [fixture.frame(camera, sequence, START + .012 + sequence / 30, actors)
                  for sequence, actors in enumerate(active) for camera in fixture.cameras]
        _, _, packets = fixture.replay(frames)
        self.assertEqual(len(packets), len(active))
        self.assert_safe(fixture, packets, exact_geometry=True)
        for packet, actors in zip(packets, active):
            self.assertEqual(len(packet.persons), len(actors))
            self.assert_people_available(fixture, packet, actors, minimum=33)

    def test_whole_person_visible_in_one_camera_remains_without_invented_depth(self):
        fixture = MotionFixture()
        frames = [fixture.frame(camera, 0, START + .012, (0, 1) if camera == "front" else (0,))
                  for camera in fixture.cameras]
        _, _, packets = fixture.replay(frames)
        self.assert_safe(fixture, packets, exact_geometry=True)
        self.assert_people_available(fixture, packets[0], {0}, minimum=33)
        absent = next(person for person in packets[0].persons if person.identity_id == "member-1")
        self.assertFalse(absent.valid_mask.any())
        self.assertTrue(np.isnan(absent.world_points).all())

    def test_reappearing_person_with_hidden_face_does_not_inherit_prior_identity(self):
        fixture = MotionFixture()
        frames = []
        for sequence, actors in enumerate(((0, 1), (0,), (0, 1))):
            for camera in fixture.cameras:
                frames.append(fixture.frame(camera, sequence, START + .017 + sequence / 30,
                                            actors, identified=sequence != 2))
        _, _, packets = fixture.replay(frames)
        self.assert_safe(fixture, packets, exact_geometry=True)
        self.assert_people_available(fixture, packets[-1], {0, 1}, minimum=33)
        self.assertTrue(all(person.identity_id is None for person in packets[-1].persons))

    def test_compound_crossing_face_occlusion_dropout_reordering_and_stream_restart(self):
        fixture = MotionFixture()
        rng = random.Random(173)
        events = []
        count = 9
        def crossing_body(exposure, actor):
            return body_pose((exposure - START) * 30, actor, count, crossing=True)
        restart_time = START + 4 / 30 - .001
        for step in range(count):
            exposure = START + step / 30
            for camera in fixture.cameras:
                if camera == "front" and 2 <= step <= 5:
                    continue
                restarted = camera == "right" and step >= 4
                sequence = step - 4 if restarted else step
                epoch = "motion-v2" if restarted else "motion-v1"
                delay = .070 if camera == "right" and step == 3 else rng.uniform(.004, .055)
                frame = fixture.frame(camera, sequence, exposure, (0, 1), pose=crossing_body,
                                      identified=False, epoch=epoch, delay=delay)
                events.append((frame.received_at_seconds, "frame", frame))
                if (step + len(camera)) % 3 == 0:
                    duplicate = replace(frame, received_at_seconds=frame.received_at_seconds + .002)
                    events.append((duplicate.received_at_seconds, "frame", duplicate))
        events.append((restart_time, "restart", None))
        sync, fusion = fixture.engines()
        bundles = []
        for _, kind, frame in sorted(events, key=lambda item: (item[0], item[1])):
            if kind == "restart":
                bundles.extend(sync.disconnect_camera("right"))
                sync.connect_camera("right", stream_epoch="motion-v2")
            else:
                bundles.extend(sync.add_frame(frame))
        bundles.extend(sync.flush())
        packets = tuple(fusion.fuse(bundle) for bundle in bundles)
        self.assertEqual(len(bundles), count)
        self.assertGreater(sync.stats["duplicate_frames"], 0)
        self.assertGreater(sync.stats["epoch_mismatch_frames"], 0)
        self.assertTrue(any(bundle.missing_cameras for bundle in bundles))
        self.assert_safe(fixture, packets, exact_geometry=True)
        self.assertTrue(all(person.identity_id is None for packet in packets for person in packet.persons))
        self.assertTrue(any(person.identity_status == "ambiguous" for packet in packets for person in packet.persons))
        # During the exact overlap ambiguity is expected. At the separated ends,
        # at least three healthy cameras can independently recover both bodies.
        for packet in (packets[0], packets[1], packets[-2], packets[-1]):
            self.assert_people_available(fixture, packet, {0, 1}, minimum=25)


if __name__ == "__main__":
    unittest.main()
