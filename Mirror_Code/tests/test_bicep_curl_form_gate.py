import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from core.pose_analysis import ExerciseAnalyzer, mp_pose
except ModuleNotFoundError:
    ExerciseAnalyzer = None
    mp_pose = None


def make_landmarks():
    if mp_pose is None:
        return []
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(len(mp_pose.PoseLandmark))]
    for idx in range(len(points)):
        setattr(points[idx], "visibility", 1.0)
    return points


def set_point(landmarks, pose_landmark, x, y, visibility=1.0, z=0.0):
    landmarks[pose_landmark.value] = SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


@unittest.skipIf(ExerciseAnalyzer is None, "mediapipe is not installed in this environment")
class BicepCurlFormGateTest(unittest.TestCase):
    def setUp(self):
        self.analyzer = ExerciseAnalyzer(
            exercise="bicep_curl",
            aruco_detector=None,
            database_handler=None,
            user_id=None,
        )

    def test_front_facing_pose_is_rejected(self):
        landmarks = make_landmarks()
        set_point(landmarks, mp_pose.PoseLandmark.NOSE, 0.50, 0.20)
        set_point(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, 0.34, 0.30)
        set_point(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, 0.66, 0.30)
        set_point(landmarks, mp_pose.PoseLandmark.LEFT_HIP, 0.38, 0.68)
        set_point(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, 0.62, 0.68)
        set_point(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW, 0.25, 0.48)
        set_point(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW, 0.75, 0.48)
        set_point(landmarks, mp_pose.PoseLandmark.LEFT_WRIST, 0.22, 0.62)
        set_point(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, 0.78, 0.62)

        form = self.analyzer._bicep_curl_form_gate(landmarks)

        self.assertTrue(form["front_facing"])
        self.assertFalse(form["form_ok"])

    def test_rep_not_counted_if_top_position_is_not_curl_like(self):
        landmarks = make_landmarks()
        self.analyzer.stable_start_detected = True
        self.analyzer.rep_state = 0
        self.analyzer.side_pose_stable_frames = 12

        invalid_form = {
            "form_ok": True,
            "front_facing": False,
            "side_pose_detected": True,
            "active_arm": "left",
            "rep_angle": 42,
            "active_wrist_shoulder_ratio": 0.52,
            "top_position_ok": False,
            "arm_metrics": {},
        }
        down_form = {
            "form_ok": True,
            "front_facing": False,
            "side_pose_detected": True,
            "active_arm": "left",
            "rep_angle": 160,
            "active_wrist_shoulder_ratio": 0.72,
            "top_position_ok": True,
            "arm_metrics": {},
        }
        up_form = {
            "form_ok": True,
            "front_facing": False,
            "side_pose_detected": True,
            "active_arm": "left",
            "rep_angle": 42,
            "active_wrist_shoulder_ratio": 0.50,
            "top_position_ok": True,
            "arm_metrics": {},
        }

        with patch.object(self.analyzer, "_bicep_curl_form_gate", return_value=down_form):
            self.analyzer.analyze_exercise_form(landmarks, frame=None)
        self.assertEqual(self.analyzer.rep_state, 1)

        with patch.object(self.analyzer, "_bicep_curl_form_gate", return_value=invalid_form):
            self.analyzer.analyze_exercise_form(landmarks, frame=None)
        self.assertEqual(self.analyzer.rep_state, 1)
        self.assertEqual(self.analyzer.rep_count, 0)

        with patch.object(self.analyzer, "_bicep_curl_form_gate", return_value=up_form):
            self.analyzer.analyze_exercise_form(landmarks, frame=None)
        self.assertEqual(self.analyzer.rep_state, 2)

        with patch.object(self.analyzer, "_bicep_curl_form_gate", return_value=down_form):
            self.analyzer.analyze_exercise_form(landmarks, frame=None)
        self.assertEqual(self.analyzer.rep_count, 1)
        self.assertEqual(self.analyzer.rep_state, 1)

    def test_startup_gate_allows_easier_ready_pose_before_strict_form(self):
        landmarks = make_landmarks()
        startup_only_form = {
            "form_ok": False,
            "startup_form_ok": True,
            "front_facing": False,
            "side_pose_detected": True,
            "side_pose_votes": 2,
            "active_arm": None,
            "rep_angle": None,
            "startup_active_arm": "left",
            # Outside strict down_range (150-170) but inside startup_down_range (140-178).
            "startup_rep_angle": 174,
            "active_wrist_shoulder_ratio": 0.75,
            "top_position_ok": True,
            "arm_metrics": {},
        }

        feedback = []
        with patch.object(self.analyzer, "_bicep_curl_form_gate", return_value=startup_only_form):
            for _ in range(17):
                feedback, _ = self.analyzer.analyze_exercise_form(landmarks, frame=None)

        self.assertTrue(self.analyzer.stable_start_detected)
        self.assertEqual(self.analyzer.rep_state, 1)
        self.assertIn("Ready to start!", feedback)

    def test_bicep_rep_not_counted_when_bend_warning_is_active(self):
        landmarks = make_landmarks()
        self.analyzer.stable_start_detected = True
        self.analyzer.rep_state = 1
        self.analyzer.side_pose_stable_frames = 12

        up_form = {
            "form_ok": True,
            "front_facing": False,
            "side_pose_detected": True,
            "active_arm": "left",
            "rep_angle": 42,
            "active_wrist_shoulder_ratio": 0.50,
            "top_position_ok": True,
            "arm_metrics": {},
        }
        down_form = {
            "form_ok": True,
            "front_facing": False,
            "side_pose_detected": True,
            "active_arm": "left",
            "rep_angle": 160,
            "active_wrist_shoulder_ratio": 0.72,
            "top_position_ok": True,
            "arm_metrics": {},
        }

        with patch.object(self.analyzer, "detect_bend", return_value=(False, None)):
            with patch.object(self.analyzer, "_bicep_curl_form_gate", return_value=up_form):
                self.analyzer.analyze_exercise_form(landmarks, frame=None)
        self.assertEqual(self.analyzer.rep_state, 2)

        with patch.object(self.analyzer, "detect_bend", return_value=(True, "back")):
            with patch.object(self.analyzer, "_bicep_curl_form_gate", return_value=down_form):
                feedback, _ = self.analyzer.analyze_exercise_form(landmarks, frame=None)

        self.assertEqual(self.analyzer.rep_count, 0)
        self.assertEqual(self.analyzer.rep_state, 0)
        self.assertTrue(any("Warning:" in text for text in feedback))

    def test_non_bicep_rep_not_counted_when_bend_warning_is_active(self):
        analyzer = ExerciseAnalyzer(
            exercise="seated_shoulder_press",
            aruco_detector=None,
            database_handler=None,
            user_id=None,
        )
        analyzer.stable_start_detected = True
        analyzer.rep_state = 1
        landmarks = make_landmarks()

        with patch.object(analyzer, "detect_bend", return_value=(False, None)):
            with patch.object(analyzer, "_resolve_rep_angle", return_value=170):
                analyzer.analyze_exercise_form(landmarks, frame=None)
        self.assertEqual(analyzer.rep_state, 2)

        with patch.object(analyzer, "detect_bend", return_value=(True, "back")):
            with patch.object(analyzer, "_resolve_rep_angle", return_value=90):
                feedback, _ = analyzer.analyze_exercise_form(landmarks, frame=None)

        self.assertEqual(analyzer.rep_count, 0)
        self.assertEqual(analyzer.rep_state, 1)
        self.assertTrue(any("Warning:" in text for text in feedback))

    def test_bicep_up_range_miss_overlay_flag_on_partial_rep(self):
        landmarks = make_landmarks()
        self.analyzer.stable_start_detected = True
        self.analyzer.rep_state = 1
        self.analyzer.side_pose_stable_frames = 12

        down_form = {
            "form_ok": True,
            "front_facing": False,
            "side_pose_detected": True,
            "active_arm": "left",
            "rep_angle": 160,
            "active_wrist_shoulder_ratio": 0.72,
            "top_position_ok": True,
            "arm_metrics": {},
        }
        partial_up_form = {
            "form_ok": True,
            "front_facing": False,
            "side_pose_detected": True,
            "active_arm": "left",
            "rep_angle": 132,
            "active_wrist_shoulder_ratio": 0.66,
            "top_position_ok": True,
            "arm_metrics": {},
        }

        with patch.object(self.analyzer, "detect_bend", return_value=(False, None)):
            with patch.object(
                self.analyzer,
                "_bicep_curl_form_gate",
                side_effect=[down_form, partial_up_form, down_form],
            ):
                self.analyzer.analyze_exercise_form(landmarks, frame=None)
                self.analyzer.analyze_exercise_form(landmarks, frame=None)
                _, metrics = self.analyzer.analyze_exercise_form(landmarks, frame=None)

        self.assertTrue(metrics.get("up_range_miss"))

    def test_bicep_down_range_miss_overlay_flag_on_partial_return(self):
        landmarks = make_landmarks()
        self.analyzer.stable_start_detected = True
        self.analyzer.rep_state = 2
        self.analyzer.rep_active_arm = "left"
        self.analyzer.side_pose_stable_frames = 12

        up_form = {
            "form_ok": True,
            "front_facing": False,
            "side_pose_detected": True,
            "active_arm": "left",
            "rep_angle": 45,
            "active_wrist_shoulder_ratio": 0.52,
            "top_position_ok": True,
            "arm_metrics": {},
        }
        partial_down_form = {
            "form_ok": True,
            "front_facing": False,
            "side_pose_detected": True,
            "active_arm": "left",
            "rep_angle": 98,
            "active_wrist_shoulder_ratio": 0.62,
            "top_position_ok": True,
            "arm_metrics": {},
        }

        with patch.object(self.analyzer, "detect_bend", return_value=(False, None)):
            with patch.object(
                self.analyzer,
                "_bicep_curl_form_gate",
                side_effect=[up_form, partial_down_form, up_form],
            ):
                self.analyzer.analyze_exercise_form(landmarks, frame=None)
                self.analyzer.analyze_exercise_form(landmarks, frame=None)
                _, metrics = self.analyzer.analyze_exercise_form(landmarks, frame=None)

        self.assertTrue(metrics.get("down_range_miss"))


if __name__ == "__main__":
    unittest.main()
