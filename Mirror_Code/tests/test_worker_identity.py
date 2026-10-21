import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from ui.worker import ExerciseWorker


class WorkerIdentityTest(unittest.TestCase):
    def test_smoothed_label_cannot_start_previous_users_session_on_new_face(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        recognizer = MagicMock()
        recognizer.recognize_faces.return_value = (frame, [(10, 50, 50, 10)], ["Bob"])
        database = MagicMock()
        worker = ExerciseWorker(database, 0, face_recognizer=recognizer)
        worker.recognition_state.last_recognized_user = "Alice"
        worker.recognition_state.known_elapsed_seconds = worker.recognition_state.known_threshold_seconds
        tracker = worker.recognition_state.label_tracker
        tracker._scores = {"Alice": 100.0}
        tracker._stable_primary_label = "Alice"
        capture = MagicMock()
        capture.read.side_effect = [(True, frame), (False, None)]
        with patch("ui.worker.open_camera_capture", return_value=capture), patch(
            "ui.worker.mp_pose.Pose"
        ), patch("ui.worker.ExerciseAnalyzer") as analyzer, patch.object(
            worker, "_sleep_to_target_loop"
        ), patch.object(worker, "_emit_visual_outputs"):
            worker.run()
        database.get_member_info.assert_not_called()
        analyzer.assert_not_called()
        self.assertFalse(worker.exercise_analysis_active)
        self.assertEqual(worker.recognition_state.known_elapsed_seconds, 0.0)
        capture.release.assert_called_once()

    def test_unreliable_pose_discards_pending_rep_in_running_worker(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        worker = ExerciseWorker(MagicMock(), 0, face_recognizer=MagicMock())
        worker.exercise_analysis_active = True
        worker.face_recognition_active = False
        worker.exercise_analyzer = MagicMock()
        worker.exercise_analyzer.stable_start_detected = True
        worker.exercise_analyzer.rep_count = 0
        worker.exercise_analyzer.set_count = 0
        worker.exercise_analyzer.current_weight = 0
        worker.exercise_analyzer.no_pose_frames = 0
        capture = MagicMock()
        capture.read.side_effect = [(True, frame), (False, None)]
        pose = MagicMock()
        pose.process.return_value = SimpleNamespace(pose_landmarks=None)
        with patch("ui.worker.open_camera_capture", return_value=capture), patch(
            "ui.worker.mp_pose.Pose", return_value=pose
        ), patch.object(worker, "_sleep_to_target_loop"), patch.object(
            worker, "_emit_visual_outputs"
        ), patch.object(worker, "_should_run_target_face_recheck", return_value=False):
            worker.run()
        worker.exercise_analyzer.invalidate_tracking.assert_called_once()
        worker.exercise_analyzer.analyze_exercise_form.assert_not_called()

    def test_malformed_numeric_environment_does_not_crash_worker(self):
        with patch.dict(os.environ, {
            "SMART_MIRROR_EXIT_SAVE_RETRY_LIMIT": "invalid",
            "SMART_MIRROR_WORKER_ANALYSIS_FPS": "inf",
            "SMART_MIRROR_REGISTRATION_CAPTURE_SECONDS": "nan",
        }):
            worker = ExerciseWorker(MagicMock(), 0, face_recognizer=MagicMock())
        self.assertEqual(worker.exit_save_retry_limit, 2)
        self.assertEqual(worker._timing_reference_fps, 20)
        self.assertTrue(np.isfinite(worker.registration_state.target_capture_seconds))
