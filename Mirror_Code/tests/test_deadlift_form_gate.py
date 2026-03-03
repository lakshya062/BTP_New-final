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
class DeadliftFormGateTest(unittest.TestCase):
    def setUp(self):
        self.analyzer = ExerciseAnalyzer(
            exercise="deadlift",
            aruco_detector=None,
            database_handler=None,
            user_id=None,
        )

    def test_front_facing_pose_is_rejected(self):
        landmarks = make_landmarks()
        set_point(landmarks, mp_pose.PoseLandmark.NOSE, 0.50, 0.18)
        set_point(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, 0.34, 0.30)
        set_point(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, 0.66, 0.30)
        set_point(landmarks, mp_pose.PoseLandmark.LEFT_HIP, 0.38, 0.56)
        set_point(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, 0.62, 0.56)
        set_point(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, 0.40, 0.75)
        set_point(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, 0.60, 0.75)
        set_point(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, 0.41, 0.93)
        set_point(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, 0.59, 0.93)

        form = self.analyzer._deadlift_form_gate(landmarks)

        self.assertTrue(form["front_facing"])
        self.assertFalse(form["form_ok"])

    def test_startup_gate_allows_ready_pose_before_strict_deadlift_form(self):
        landmarks = make_landmarks()
        startup_only_form = {
            "form_ok": False,
            "startup_form_ok": True,
            "front_facing": False,
            "side_pose_detected": True,
            "side_pose_votes": 2,
            "active_side": None,
            "rep_angle": None,
            "startup_active_side": "left",
            "startup_rep_angle": 154,  # In startup_down_range (150-180).
            "top_position_ok": False,
            "startup_top_position_ok": True,
            "bottom_position_ok": False,
            "active_knee_angle": None,
            "startup_knee_angle": 153,
            "active_torso_lean_angle": None,
            "startup_torso_lean_angle": 18,
            "side_metrics": {},
        }

        feedback = []
        with patch.object(self.analyzer, "_deadlift_form_gate", return_value=startup_only_form):
            for _ in range(18):
                feedback, _ = self.analyzer.analyze_exercise_form(landmarks, frame=None)

        self.assertTrue(self.analyzer.stable_start_detected)
        self.assertEqual(self.analyzer.rep_state, 1)
        self.assertIn("Ready to start!", feedback)

    def test_rep_count_requires_valid_bottom_and_lockout_positions(self):
        landmarks = make_landmarks()
        self.analyzer.stable_start_detected = True
        self.analyzer.rep_state = 1
        self.analyzer.side_pose_stable_frames = 10

        top_form = {
            "form_ok": True,
            "startup_form_ok": True,
            "front_facing": False,
            "side_pose_detected": True,
            "side_pose_votes": 3,
            "active_side": "left",
            "rep_angle": 166,
            "startup_active_side": "left",
            "startup_rep_angle": 166,
            "top_position_ok": True,
            "startup_top_position_ok": True,
            "bottom_position_ok": False,
            "active_knee_angle": 168,
            "startup_knee_angle": 168,
            "active_torso_lean_angle": 16,
            "startup_torso_lean_angle": 16,
            "side_metrics": {},
        }
        invalid_bottom_form = {
            **top_form,
            "rep_angle": 102,
            "top_position_ok": False,
            "bottom_position_ok": False,
            "active_knee_angle": 150,
            "active_torso_lean_angle": 18,
        }
        valid_bottom_form = {
            **top_form,
            "rep_angle": 96,
            "top_position_ok": False,
            "bottom_position_ok": True,
            "active_knee_angle": 116,
            "active_torso_lean_angle": 46,
        }
        invalid_lockout_form = {
            **top_form,
            "rep_angle": 164,
            "top_position_ok": False,
            "bottom_position_ok": False,
            "active_knee_angle": 152,
            "active_torso_lean_angle": 34,
        }
        valid_lockout_form = {
            **top_form,
            "rep_angle": 168,
            "top_position_ok": True,
            "bottom_position_ok": False,
            "active_knee_angle": 170,
            "active_torso_lean_angle": 14,
        }

        with patch.object(self.analyzer, "_deadlift_form_gate", return_value=top_form):
            self.analyzer.analyze_exercise_form(landmarks, frame=None)
        self.assertEqual(self.analyzer.rep_state, 1)

        with patch.object(self.analyzer, "_deadlift_form_gate", return_value=invalid_bottom_form):
            _, metrics = self.analyzer.analyze_exercise_form(landmarks, frame=None)
        self.assertEqual(self.analyzer.rep_state, 1)
        self.assertEqual(self.analyzer.rep_count, 0)
        self.assertTrue(metrics.get("up_range_miss"))

        with patch.object(self.analyzer, "_deadlift_form_gate", return_value=valid_bottom_form):
            self.analyzer.analyze_exercise_form(landmarks, frame=None)
        self.assertEqual(self.analyzer.rep_state, 2)

        with patch.object(self.analyzer, "_deadlift_form_gate", return_value=invalid_lockout_form):
            _, metrics = self.analyzer.analyze_exercise_form(landmarks, frame=None)
        self.assertEqual(self.analyzer.rep_state, 2)
        self.assertEqual(self.analyzer.rep_count, 0)
        self.assertTrue(metrics.get("down_range_miss"))

        with patch.object(self.analyzer, "_deadlift_form_gate", return_value=valid_lockout_form):
            self.analyzer.analyze_exercise_form(landmarks, frame=None)
        self.assertEqual(self.analyzer.rep_state, 1)
        self.assertEqual(self.analyzer.rep_count, 1)


if __name__ == "__main__":
    unittest.main()
