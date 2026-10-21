import unittest
from types import SimpleNamespace

import numpy as np

try:
    from ui.worker import ExerciseWorker, mp_pose
except ModuleNotFoundError:
    ExerciseWorker = None
    mp_pose = None


class DummyDbHandler:
    def get_daily_exercise_progress(self, user_id=None, exercise=None, date_str=None):
        return {"set_count": 0}


class DummyFaceRecognizer:
    def __init__(self, responses=None, frame_scale=0.25):
        self.responses = list(responses or [])
        self.calls = []
        self._inference_config = {"frame_scale": frame_scale}

    def recognize_faces(self, frame):
        self.calls.append(frame.shape)
        if self.responses:
            face_locations, face_names = self.responses.pop(0)
        else:
            face_locations, face_names = ([], [])
        return frame, face_locations, face_names


def make_pose_landmarks(center_x=0.30, center_y=0.54, width=0.18, height=0.58):
    if mp_pose is None:
        return []

    points = [
        SimpleNamespace(x=center_x, y=center_y, z=0.0, visibility=1.0)
        for _ in range(len(mp_pose.PoseLandmark))
    ]

    shoulder_y = center_y - (height * 0.30)
    elbow_y = center_y - (height * 0.06)
    wrist_y = center_y + (height * 0.12)
    hip_y = center_y
    knee_y = center_y + (height * 0.34)
    ankle_y = center_y + (height * 0.48)

    def set_point(pose_landmark, x, y, visibility=1.0):
        points[pose_landmark.value] = SimpleNamespace(
            x=x,
            y=y,
            z=0.0,
            visibility=visibility,
        )

    set_point(mp_pose.PoseLandmark.NOSE, center_x, center_y - (height * 0.52))
    set_point(mp_pose.PoseLandmark.LEFT_SHOULDER, center_x - (width / 2.0), shoulder_y)
    set_point(mp_pose.PoseLandmark.RIGHT_SHOULDER, center_x + (width / 2.0), shoulder_y)
    set_point(mp_pose.PoseLandmark.LEFT_ELBOW, center_x - (width * 0.55), elbow_y)
    set_point(mp_pose.PoseLandmark.RIGHT_ELBOW, center_x + (width * 0.55), elbow_y)
    set_point(mp_pose.PoseLandmark.LEFT_WRIST, center_x - (width * 0.65), wrist_y)
    set_point(mp_pose.PoseLandmark.RIGHT_WRIST, center_x + (width * 0.65), wrist_y)
    set_point(mp_pose.PoseLandmark.LEFT_HIP, center_x - (width * 0.30), hip_y)
    set_point(mp_pose.PoseLandmark.RIGHT_HIP, center_x + (width * 0.30), hip_y)
    set_point(mp_pose.PoseLandmark.LEFT_KNEE, center_x - (width * 0.28), knee_y)
    set_point(mp_pose.PoseLandmark.RIGHT_KNEE, center_x + (width * 0.28), knee_y)
    set_point(mp_pose.PoseLandmark.LEFT_ANKLE, center_x - (width * 0.22), ankle_y)
    set_point(mp_pose.PoseLandmark.RIGHT_ANKLE, center_x + (width * 0.22), ankle_y)
    return points


def make_pose_landmark_packet(center_x=0.30, center_y=0.54, width=0.18, height=0.58):
    return SimpleNamespace(
        landmark=make_pose_landmarks(
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
        )
    )


def paint_box(frame, box, color):
    left, top, right, bottom = box
    frame[top:bottom, left:right] = color
    return frame


@unittest.skipIf(ExerciseWorker is None, "worker dependencies are not installed in this environment")
class ExerciseWorkerTrackingLockTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.face_recognizer = DummyFaceRecognizer()
        self.worker = ExerciseWorker(
            db_handler=DummyDbHandler(),
            camera_index=0,
            face_recognizer=self.face_recognizer,
        )
        self.worker._seed_tracked_user_lock(
            "Alice",
            (60, 156, 72, 144),
            self.frame.shape,
            now=1.0,
        )
        self.worker.full_frame_pose_acquire = True

    def test_pose_lock_accepts_small_movement_for_same_user(self):
        initial_center = self.worker.locked_body_center

        accepted = self.worker._accept_pose_for_target_lock(
            make_pose_landmarks(center_x=0.31, center_y=0.55),
            self.frame.shape,
        )

        self.assertTrue(accepted)
        self.assertNotEqual(initial_center, self.worker.locked_body_center)

    def test_pose_lock_rejects_far_person_in_front(self):
        initial_center = self.worker.locked_body_center

        accepted = self.worker._accept_pose_for_target_lock(
            make_pose_landmarks(center_x=0.78, center_y=0.56),
            self.frame.shape,
        )

        self.assertFalse(accepted)
        self.assertEqual(initial_center, self.worker.locked_body_center)

    def test_pose_lock_rejects_overlapping_passerby_with_shifted_head_anchor(self):
        initial_center = self.worker.locked_body_center

        accepted = self.worker._accept_pose_for_target_lock(
            make_pose_landmarks(center_x=0.36, center_y=0.56, width=0.28, height=0.72),
            self.frame.shape,
        )

        self.assertFalse(accepted)
        self.assertEqual(initial_center, self.worker.locked_body_center)

    def test_locked_face_recheck_confirms_same_user_from_roi(self):
        self.worker.locked_body_center = (640.0, 540.0)
        self.worker.locked_body_size = (300.0, 760.0)
        self.worker.target_lost_since = 10.0
        self.face_recognizer.responses = [
            ([(14, 52, 44, 20)], ["Alice"]),
        ]

        result = self.worker._run_locked_face_recheck(
            self.frame,
            now=11.0,
            roi_box=(440, 100, 920, 1040),
        )

        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(self.worker.locked_username, "Alice")
        self.assertIsNone(self.worker.target_lost_since)
        self.assertEqual(self.worker.last_identity_confirmed_ts, 11.0)
        self.assertEqual(self.face_recognizer.calls[-1][:2], (940, 480))

    def test_locked_face_recheck_rejects_different_known_user(self):
        self.worker.locked_body_center = (640.0, 540.0)
        self.worker.locked_body_size = (300.0, 760.0)
        self.face_recognizer.responses = [
            ([(10, 44, 40, 18)], ["Bob"]),
        ]

        result = self.worker._run_locked_face_recheck(
            self.frame,
            now=11.0,
            roi_box=(440, 100, 920, 1040),
        )

        self.assertEqual(result["status"], "other")
        self.assertEqual(result["label"], "Bob")
        self.assertEqual(self.worker.locked_username, "Alice")
        self.assertEqual(self.worker.last_identity_confirmed_ts, 1.0)

    def test_locked_face_recheck_falls_back_to_full_frame_for_same_user(self):
        self.worker.locked_body_center = (640.0, 540.0)
        self.worker.locked_body_size = (300.0, 760.0)
        self.worker.face_recognizer.responses = [
            ([(10, 44, 40, 18)], ["Bob"]),
            ([(80, 260, 180, 160)], ["Alice"]),
        ]

        result = self.worker._run_locked_face_recheck(
            self.frame,
            now=11.0,
            roi_box=(440, 100, 920, 1040),
        )

        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(result["label"], "Alice")
        self.assertEqual(len(self.face_recognizer.calls), 2)
        self.assertEqual(self.face_recognizer.calls[0][:2], (940, 480))
        self.assertEqual(self.face_recognizer.calls[1][:2], (1080, 1920))

    def test_tracked_user_hold_requires_full_timeout_before_switching_back(self):
        self.worker.target_lost_since = 10.0
        self.worker.tracked_user_hold_seconds = 2.0
        self.worker.target_exit_started_since = 10.2

        self.assertFalse(self.worker._should_return_to_face_recognition(11.8, self.frame.shape))
        self.assertTrue(self.worker._should_return_to_face_recognition(12.3, self.frame.shape))

    def test_target_exit_requires_edge_evidence_or_long_fallback(self):
        self.worker.target_lost_since = 10.0
        self.worker.tracked_user_hold_seconds = 1.0
        self.worker.tracked_user_missing_timeout_seconds = 5.0
        self.worker.locked_body_box = (520, 180, 980, 960)

        self.assertFalse(self.worker._has_target_exited_view(12.1, self.frame.shape))
        self.assertTrue(self.worker._has_target_exited_view(15.1, self.frame.shape))

    def test_mark_target_missing_starts_exit_timer_when_target_reaches_edge(self):
        self.worker.tracked_user_hold_seconds = 0.5
        self.worker.target_exit_confirm_seconds = 0.4
        self.worker.locked_body_box = (8, 220, 520, 1010)

        self.worker._mark_target_missing(11.0, self.frame.shape)

        self.assertEqual(self.worker.target_exit_started_since, 11.0)
        self.assertFalse(self.worker._has_target_exited_view(11.35, self.frame.shape))
        self.assertTrue(self.worker._has_target_exited_view(11.95, self.frame.shape))

    def test_face_confirmation_clears_temporary_loss_before_timeout(self):
        self.worker.locked_body_center = (640.0, 540.0)
        self.worker.locked_body_size = (300.0, 760.0)
        self.worker.target_lost_since = 10.0
        self.face_recognizer.responses = [
            ([(14, 52, 44, 20)], ["Alice"]),
        ]

        result = self.worker._run_locked_face_recheck(
            self.frame,
            now=11.2,
            roi_box=(440, 100, 920, 1040),
        )

        self.assertEqual(result["status"], "confirmed")
        self.assertFalse(self.worker._should_return_to_face_recognition(11.9))

    def test_session_profile_rejects_same_place_different_appearance_after_lock(self):
        landmarks = make_pose_landmarks(center_x=0.31, center_y=0.55)
        pose_lock = self.worker._resolve_pose_lock(landmarks, self.frame.shape)
        self.worker.locked_body_box = pose_lock["box"]
        self.worker.locked_body_center = pose_lock["center"]
        self.worker.locked_body_size = pose_lock["size"]
        self.worker._pose_lock_established = True

        blue_frame = np.zeros_like(self.frame)
        paint_box(blue_frame, pose_lock["box"], (255, 0, 0))
        self.worker._update_session_identity_profile(
            frame=blue_frame,
            body_box=pose_lock["box"],
            face_box=pose_lock["face_box"],
            now=12.0,
            strong_identity=True,
        )

        self.assertTrue(
            self.worker._accept_pose_for_target_lock(
                landmarks,
                self.frame.shape,
                frame=blue_frame,
            )
        )

        red_frame = np.zeros_like(self.frame)
        paint_box(red_frame, pose_lock["box"], (0, 0, 255))

        self.assertFalse(
            self.worker._accept_pose_for_target_lock(
                landmarks,
                self.frame.shape,
                frame=red_frame,
            )
        )

    def test_missing_center_lock_uses_thirty_second_stale_timeout(self):
        self.worker.target_lost_since = 10.0
        self.worker.tracked_user_hold_seconds = 5.0
        self.worker.tracked_user_missing_timeout_seconds = 30.0
        self.worker.target_exit_started_since = None
        self.worker.locked_body_box = (520, 180, 980, 960)

        self.assertFalse(self.worker._should_return_to_face_recognition(39.9, self.frame.shape))
        self.assertTrue(self.worker._should_return_to_face_recognition(40.1, self.frame.shape))

    def test_missing_status_hint_escalates_to_countdown(self):
        self.worker.target_lost_since = 10.0
        self.worker.tracked_user_missing_warning_seconds = 5.0
        self.worker.tracked_user_stale_countdown_seconds = 15.0
        self.worker.tracked_user_missing_timeout_seconds = 30.0

        self.assertEqual(self.worker._tracked_user_missing_status_hint("Alice", 13.0), "Waiting for Alice.")
        self.assertIn("Step back in", self.worker._tracked_user_missing_status_hint("Alice", 17.0))
        self.assertIn("Session ending in", self.worker._tracked_user_missing_status_hint("Alice", 26.0))

    def test_lock_marker_draws_above_tracked_user(self):
        frame = np.zeros_like(self.frame)

        drawn = self.worker._draw_tracked_user_lock_marker(frame, now=12.0, paused=False)

        self.assertTrue(drawn)
        self.assertGreater(int(frame.sum()), 0)

    def test_end_session_palm_requires_hold_duration(self):
        landmarks = make_pose_landmarks(center_x=0.31, center_y=0.55)
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value] = SimpleNamespace(
            x=left_shoulder.x,
            y=left_shoulder.y - 0.08,
            z=0.0,
            visibility=1.0,
        )
        landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value] = SimpleNamespace(
            x=left_shoulder.x - 0.035,
            y=left_shoulder.y - 0.14,
            z=0.0,
            visibility=1.0,
        )
        landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value] = SimpleNamespace(
            x=left_shoulder.x + 0.035,
            y=left_shoulder.y - 0.14,
            z=0.0,
            visibility=1.0,
        )
        landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value] = SimpleNamespace(
            x=left_shoulder.x - 0.055,
            y=left_shoulder.y - 0.10,
            z=0.0,
            visibility=1.0,
        )

        complete, hint = self.worker._update_end_session_palm_hold(landmarks, now=20.0)
        self.assertFalse(complete)
        self.assertIn("Hold palm to end", hint)

        complete, _hint = self.worker._update_end_session_palm_hold(landmarks, now=23.1)
        self.assertTrue(complete)

    def test_analysis_roi_uses_full_frame_until_pose_lock_is_established(self):
        roi_box = self.worker._resolve_analysis_roi_box(self.frame.shape)

        self.assertEqual(roi_box, (0, 0, self.frame.shape[1], self.frame.shape[0]))

        self.worker._pose_lock_established = True
        locked_roi_box = self.worker._resolve_analysis_roi_box(self.frame.shape)

        self.assertNotEqual(locked_roi_box, roi_box)

    def test_initial_pose_acquire_relaxes_pose_size_thresholds(self):
        initial_landmarks = make_pose_landmarks(width=0.031, height=0.22)

        self.assertTrue(self.worker._has_reliable_pose(initial_landmarks))

        self.worker._pose_lock_established = True

        self.assertFalse(self.worker._has_reliable_pose(initial_landmarks))

    def test_cache_pose_overlay_smooths_landmark_motion(self):
        first_pose = make_pose_landmark_packet(center_x=0.30, center_y=0.54)
        second_pose = make_pose_landmark_packet(center_x=0.37, center_y=0.54)

        self.worker._cache_pose_overlay(first_pose, (118, 242, 130), (118, 242, 130), now=1.0)
        first_shoulder_x = self.worker._cached_pose_points[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0]

        self.worker._cache_pose_overlay(second_pose, (118, 242, 130), (118, 242, 130), now=1.1)
        smoothed_shoulder_x = self.worker._cached_pose_points[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0]
        raw_second_shoulder_x = second_pose.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x

        self.assertGreater(smoothed_shoulder_x, first_shoulder_x)
        self.assertLess(smoothed_shoulder_x, raw_second_shoulder_x)

    def test_draw_cached_pose_overlay_renders_styled_skeleton(self):
        frame = np.zeros_like(self.frame)
        pose_packet = make_pose_landmark_packet()

        self.worker._cache_pose_overlay(pose_packet, (70, 70, 255), (70, 70, 255), now=2.0)

        drawn = self.worker._draw_cached_pose_overlay_if_recent(frame, now=2.0)

        self.assertTrue(drawn)
        self.assertGreater(int(frame.sum()), 0)


if __name__ == "__main__":
    unittest.main()
