import math
import os
import unittest
from copy import copy
from types import SimpleNamespace
from unittest.mock import patch

try:
    from core.pose_analysis import ExerciseAnalyzer, _project_landmarks_for_geometry, mp_pose
except ModuleNotFoundError:
    ExerciseAnalyzer = None
    mp_pose = None


def make_pose():
    points = [SimpleNamespace(x=0.5, y=0.2, z=0.0, visibility=1.0)
              for _ in range(len(mp_pose.PoseLandmark))]
    for side, x, depth in (('LEFT', 0.45, -0.1), ('RIGHT', 0.48, 0.1)):
        for joint, y in (('SHOULDER', 0.30), ('HIP', 0.65), ('ELBOW', 0.52),
                         ('WRIST', 0.625), ('KNEE', 0.81), ('ANKLE', 0.96)):
            points[getattr(mp_pose.PoseLandmark, f'{side}_{joint}').value] = SimpleNamespace(
                x=x + (0.038 if joint == 'WRIST' else 0), y=y, z=depth, visibility=1.0)
    return points


def curl_form(angle=160, side='left', form_ok=True):
    return {
        'form_ok': form_ok, 'startup_form_ok': form_ok,
        'front_facing': False, 'side_pose_detected': form_ok, 'side_pose_votes': 3 if form_ok else 0,
        'active_arm': side if form_ok else None, 'rep_angle': angle if form_ok else None,
        'startup_active_arm': side if form_ok else None,
        'startup_rep_angle': angle if form_ok else None,
        'active_wrist_shoulder_ratio': 0.8 if angle is not None and angle > 100 else 0.45,
        'startup_wrist_shoulder_ratio': 0.8,
        'top_position_ok': True, 'arm_metrics': {},
    }


@unittest.skipIf(ExerciseAnalyzer is None, 'mediapipe is not installed in this environment')
class PoseAnalysisReliabilityTest(unittest.TestCase):
    def analyzer(self, exercise='bicep_curl'):
        return ExerciseAnalyzer(exercise, None, None)

    def test_bad_environment_values_do_not_crash_startup(self):
        with patch.dict(os.environ, {'SMART_MIRROR_ARUCO_DETECT_EVERY_N_FRAMES': 'bad',
                                     'SMART_MIRROR_READY_FEEDBACK_FRAMES': ''}):
            analyzer = self.analyzer()
        self.assertEqual(analyzer.weight_detection_stride, 2)
        self.assertEqual(analyzer.ready_feedback_hold_frames, 8)

    def test_invalid_tracking_drops_incomplete_rep_and_preserves_completed_activity(self):
        analyzer = self.analyzer()
        analyzer.stable_start_detected = True
        analyzer.rep_state = 2
        analyzer.rep_started_ts = 20
        analyzer.rep_started_at_utc = '2026-01-01T00:00:00'
        analyzer.rep_active_arm = 'left'
        analyzer.rep_count = 3
        analyzer.set_count = 2
        analyzer.rep_data = [{'weight': 10}]
        analyzer.invalidate_tracking()
        self.assertEqual(analyzer.rep_state, 0)
        self.assertIsNone(analyzer.rep_started_ts)
        self.assertIsNone(analyzer.rep_started_at_utc)
        self.assertIsNone(analyzer.rep_active_arm)
        self.assertEqual((analyzer.rep_count, analyzer.set_count), (3, 2))
        self.assertEqual(analyzer.rep_data, [{'weight': 10}])

    def test_no_count_across_form_gate_tracking_loss(self):
        analyzer = self.analyzer()
        analyzer.stable_start_detected = True
        analyzer.side_pose_stable_frames = 12
        with patch.object(analyzer, '_bicep_curl_form_gate', side_effect=[
                curl_form(), curl_form(45), curl_form(None, form_ok=False), curl_form()]):
            for _ in range(4):
                analyzer.analyze_exercise_form(make_pose(), None)
        self.assertEqual(analyzer.rep_count, 0)
        self.assertEqual(analyzer.rep_state, 0)
        self.assertIsNone(analyzer.rep_started_ts)

    def test_an_arm_switch_cannot_supply_the_other_arms_up_phase(self):
        analyzer = self.analyzer()
        analyzer.stable_start_detected = True
        analyzer.side_pose_stable_frames = 12
        with patch.object(analyzer, '_bicep_curl_form_gate', side_effect=[
                curl_form(), curl_form(45, 'right'), curl_form(160, 'right')]):
            analyzer.analyze_exercise_form(make_pose(), None)
            analyzer.analyze_exercise_form(make_pose(), None)
            self.assertEqual(analyzer.rep_state, 0)
            analyzer.analyze_exercise_form(make_pose(), None)
        self.assertEqual(analyzer.rep_count, 0)
        self.assertEqual(analyzer.rep_state, 1)
        self.assertEqual(analyzer.rep_active_arm, 'right')

    def test_none_nonfinite_and_partial_landmarks_do_not_count_or_escape(self):
        for exercise in ('bicep_curl', 'deadlift', 'seated_shoulder_press', 'lateral_raises'):
            for landmarks in (None, [], make_pose()):
                with self.subTest(exercise=exercise, landmarks=type(landmarks).__name__):
                    if landmarks:
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x = math.nan
                    analyzer = self.analyzer(exercise)
                    analyzer.stable_start_detected = True
                    analyzer.rep_state = 2
                    analyzer.rep_started_ts = 1
                    feedback, metrics = analyzer.analyze_exercise_form(landmarks, None)
                    self.assertEqual(analyzer.rep_count, 0)
                    self.assertEqual(analyzer.rep_state, 0)
                    self.assertIsNone(metrics['rep_angle'])
                    self.assertIsNone(analyzer.rep_started_ts)
                    self.assertTrue(feedback)

    def test_generic_movement_requires_all_visible_joint_measurements(self):
        analyzer = self.analyzer('seated_shoulder_press')
        pose = make_pose()
        pose[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility = 0.1
        analyzer.stable_start_detected = True
        analyzer.rep_state = 2
        _, metrics = analyzer.analyze_exercise_form(pose, None)
        self.assertNotIn('right_elbow', metrics['angles'])
        self.assertIsNone(metrics['rep_angle'])
        self.assertEqual(analyzer.rep_state, 0)

    def test_generic_movement_cannot_count_with_unobservable_torso(self):
        analyzer = self.analyzer('seated_shoulder_press')
        analyzer.stable_start_detected = True
        analyzer.rep_state = 2
        pose = make_pose()
        pose[mp_pose.PoseLandmark.LEFT_HIP.value].visibility = 0.0
        pose[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility = 0.0
        with patch.object(analyzer, '_resolve_rep_angle', return_value=90):
            feedback, _ = analyzer.analyze_exercise_form(pose, None)
        self.assertEqual(analyzer.rep_count, 0)
        self.assertEqual(analyzer.rep_state, 0)
        self.assertTrue(any('posture' in text for text in feedback))

    def test_form_gates_reject_nonfinite_coordinates_and_confidence(self):
        for exercise in ('bicep_curl', 'deadlift'):
            analyzer = self.analyzer(exercise)
            gate = analyzer._bicep_curl_form_gate if exercise == 'bicep_curl' else analyzer._deadlift_form_gate
            for attribute, value in (('x', math.nan), ('y', math.inf), ('z', math.nan),
                                     ('visibility', math.nan), ('visibility', 2)):
                with self.subTest(exercise=exercise, attribute=attribute, value=value):
                    pose = make_pose()
                    setattr(pose[mp_pose.PoseLandmark.LEFT_SHOULDER.value], attribute, value)
                    result = gate(pose)
                    self.assertFalse(result['form_ok'])
                    self.assertFalse(result['startup_form_ok'])

    def test_same_pixel_skeleton_has_same_geometry_in_landscape_and_portrait_frames(self):
        for exercise in ('bicep_curl', 'deadlift', 'seated_shoulder_press', 'lateral_raises'):
            metrics_by_shape = []
            for height, width in ((480, 640), (640, 480)):
                pose = make_pose()
                for point in pose:
                    point.x *= 400.0 / width
                    point.y *= 400.0 / height
                originals = [(point.x, point.y, point.z, point.visibility) for point in pose]
                analyzer = self.analyzer(exercise)
                analyzer.side_pose_stable_frames = 12
                _, metrics = analyzer.analyze_exercise_form(pose, SimpleNamespace(shape=(height, width, 3)))
                metrics_by_shape.append(metrics)
                self.assertEqual(originals, [(point.x, point.y, point.z, point.visibility) for point in pose])
            with self.subTest(exercise=exercise):
                first, second = metrics_by_shape
                self.assertEqual(first['angles'].keys(), second['angles'].keys())
                self.assertTrue(first['angles'])
                for key in first['angles']:
                    self.assertAlmostEqual(first['angles'][key], second['angles'][key], places=6)
                self.assertAlmostEqual(first['rep_angle'], second['rep_angle'], places=6)

    def test_projection_preserves_native_protobuf_landmarks_and_confidence(self):
        from mediapipe.framework.formats import landmark_pb2

        point = landmark_pb2.NormalizedLandmark(x=0.6, y=0.4, z=-0.2, visibility=0.8, presence=0.7)
        original = copy(point)
        projected = _project_landmarks_for_geometry([point], SimpleNamespace(shape=(480, 640, 3)))
        self.assertEqual(point, original)
        self.assertIsNot(projected[0], point)
        self.assertAlmostEqual(projected[0].x, point.x * 640 / 480, places=6)
        self.assertEqual(projected[0].y, point.y)
        self.assertEqual(projected[0].z, point.z)
        self.assertEqual(projected[0].visibility, point.visibility)
        self.assertEqual(projected[0].presence, point.presence)

    def test_startup_requires_consecutive_valid_observations(self):
        analyzer = self.analyzer('seated_shoulder_press')
        pose = make_pose()
        with patch.object(analyzer, '_resolve_rep_angle', side_effect=[90] * 29 + [120] + [90] * 29):
            for _ in range(59):
                analyzer.analyze_exercise_form(pose, None)
        self.assertFalse(analyzer.stable_start_detected)
        self.assertEqual(analyzer.stable_frames, 29)

    def test_generic_reacquisition_requires_return_to_start_before_counting(self):
        analyzer = self.analyzer('seated_shoulder_press')
        analyzer.stable_start_detected = True
        analyzer.rep_state = 2
        analyzer.invalidate_tracking()
        with patch.object(analyzer, '_resolve_rep_angle', side_effect=[170, 90, 170, 90]):
            analyzer.analyze_exercise_form(make_pose(), None)
            self.assertEqual(analyzer.rep_state, 0)
            for _ in range(3):
                analyzer.analyze_exercise_form(make_pose(), None)
        self.assertEqual(analyzer.rep_count, 1)

    def test_increasing_angle_exercises_signal_partial_up_and_return_ranges(self):
        analyzer = self.analyzer('seated_shoulder_press')
        analyzer.stable_start_detected = True
        analyzer.rep_state = 1
        with patch.object(analyzer, '_resolve_rep_angle', side_effect=[90, 130, 90, 170, 130, 170]):
            for _ in range(3):
                _, metrics = analyzer.analyze_exercise_form(make_pose(), None)
            self.assertTrue(metrics['up_range_miss'])
            for _ in range(3):
                _, metrics = analyzer.analyze_exercise_form(make_pose(), None)
            self.assertTrue(metrics['down_range_miss'])
        self.assertEqual(analyzer.rep_count, 0)

    def test_real_curl_geometry_reaches_ready_and_counts_complete_rep(self):
        analyzer = self.analyzer()
        pose = make_pose()
        for _ in range(20):
            analyzer.analyze_exercise_form(pose, None)
        self.assertTrue(analyzer.stable_start_detected)
        self.assertEqual(analyzer.rep_active_arm, 'left')
        for side, x in (('LEFT', 0.45), ('RIGHT', 0.48)):
            wrist = pose[getattr(mp_pose.PoseLandmark, f'{side}_WRIST').value]
            wrist.x, wrist.y = x + 0.078, 0.442
        analyzer.analyze_exercise_form(pose, None)
        self.assertEqual(analyzer.rep_state, 2)
        for side, x in (('LEFT', 0.45), ('RIGHT', 0.48)):
            wrist = pose[getattr(mp_pose.PoseLandmark, f'{side}_WRIST').value]
            wrist.x, wrist.y = x + 0.038, 0.625
        analyzer.analyze_exercise_form(pose, None)
        self.assertEqual(analyzer.rep_count, 1)
        self.assertEqual(len(analyzer.rep_data), 1)

    def test_rep_duration_uses_monotonic_clock(self):
        analyzer = self.analyzer()
        analyzer.stable_start_detected = True
        analyzer.side_pose_stable_frames = 12
        with patch.object(analyzer, '_bicep_curl_form_gate', side_effect=[curl_form(), curl_form(45), curl_form()]), \
                patch('core.pose_analysis.time.monotonic', side_effect=[100.0, 102.5]), \
                patch('core.pose_analysis.time.time', side_effect=[1000.0, 1.0]) as wall_clock:
            for _ in range(3):
                analyzer.analyze_exercise_form(make_pose(), None)
        self.assertEqual(analyzer.rep_data[0]['duration_seconds'], 2.5)
        wall_clock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
