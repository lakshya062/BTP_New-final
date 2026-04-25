import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from core.config import (
        exercise_config,
        get_exercise_display_name,
        selectable_exercises,
    )
    from core.pose_analysis import ExerciseAnalyzer, mp_pose
except ModuleNotFoundError:
    exercise_config = {}
    get_exercise_display_name = None
    selectable_exercises = None
    ExerciseAnalyzer = None
    mp_pose = None


def make_landmarks():
    if mp_pose is None:
        return []
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(len(mp_pose.PoseLandmark))]
    for point in points:
        setattr(point, "visibility", 1.0)
    return points


@unittest.skipIf(ExerciseAnalyzer is None, "mediapipe is not installed in this environment")
class AdditionalExercisesTest(unittest.TestCase):
    def test_new_exercises_are_selectable_and_alias_is_hidden(self):
        selectable = selectable_exercises()

        self.assertIn("shoulder_press", selectable)
        self.assertIn("squat", selectable)
        self.assertNotIn("seated_shoulder_press", selectable)
        self.assertEqual(get_exercise_display_name("shoulder_press"), "Shoulder Press")
        self.assertEqual(get_exercise_display_name("squat"), "Squat")

    def test_shoulder_press_counts_overhead_then_return_to_shoulders(self):
        analyzer = ExerciseAnalyzer(
            exercise="shoulder_press",
            aruco_detector=None,
            database_handler=None,
            user_id=None,
        )
        analyzer.stable_start_detected = True
        analyzer.rep_state = 1

        landmarks = make_landmarks()
        with patch.object(analyzer, "_resolve_rep_angle", return_value=170):
            analyzer.analyze_exercise_form(landmarks, frame=None)
        with patch.object(analyzer, "_resolve_rep_angle", return_value=90):
            analyzer.analyze_exercise_form(landmarks, frame=None)

        self.assertEqual(analyzer.rep_count, 1)

    def test_squat_counts_depth_then_return_to_standing(self):
        analyzer = ExerciseAnalyzer(
            exercise="squat",
            aruco_detector=None,
            database_handler=None,
            user_id=None,
        )
        analyzer.stable_start_detected = True
        analyzer.rep_state = 1

        landmarks = make_landmarks()
        with patch.object(analyzer, "_resolve_rep_angle", return_value=92):
            analyzer.analyze_exercise_form(landmarks, frame=None)
        with patch.object(analyzer, "_resolve_rep_angle", return_value=170):
            analyzer.analyze_exercise_form(landmarks, frame=None)

        self.assertEqual(analyzer.rep_count, 1)

    def test_future_exercise_template_is_not_selectable(self):
        self.assertNotIn("new_exercise_key", exercise_config)
        self.assertNotIn("new_exercise_key", selectable_exercises())
