# ui/worker.py

import cv2
import face_recognition as fr_lib
import mediapipe as mp
import logging
import numpy as np
import os
import time
from datetime import datetime
from math import hypot, isfinite

from PySide6.QtCore import QThread, Signal

from core.aruco_detection import ArucoDetector
from core.pose_analysis import ExerciseAnalyzer
from core.config import exercise_config
from ui.camera_utils import open_camera_capture, transform_camera_frame
from ui.coaching import motivation_line, range_miss_message
from ui.worker_state import (
    FaceRecognitionState,
    RegistrationCaptureState,
    StablePrimaryLabelTracker,
)

mp_pose = mp.solutions.pose

POSE_RENDER_CONNECTIONS = (
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value),
    (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value),
    (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value),
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
    (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
    (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
    (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
    (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
    (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value),
)
POSE_RENDER_JOINTS = tuple(
    sorted({joint_idx for connection in POSE_RENDER_CONNECTIONS for joint_idx in connection})
)

class ExerciseWorker(QThread):
    frame_signal = Signal(object)
    thumbnail_frame_signal = Signal(object)
    display_state_signal = Signal(object)
    status_signal = Signal(str)
    counters_signal = Signal(int, int)
    user_recognized_signal = Signal(dict)
    unknown_user_detected = Signal()
    data_updated = Signal()
    audio_feedback_signal = Signal(str)
    new_user_registration_signal = Signal(str, bool)

    @staticmethod
    def _read_int_env(var_name, default, minimum=0):
        raw = os.getenv(var_name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        return max(minimum, value)

    @staticmethod
    def _read_float_env(var_name, default, minimum=0.0):
        raw = os.getenv(var_name)
        if raw is None:
            return default
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        return max(minimum, value) if isfinite(value) else default

    @staticmethod
    def _read_bool_env(var_name, default=False):
        raw = os.getenv(var_name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    def _read_duration_env(
        self,
        seconds_var_name,
        *,
        default_seconds=None,
        legacy_frame_default=None,
        legacy_frame_var_name=None,
        minimum=0.0,
    ):
        raw_seconds = os.getenv(seconds_var_name)
        if raw_seconds is not None:
            try:
                seconds = float(raw_seconds)
                if isfinite(seconds):
                    return max(minimum, seconds)
            except (TypeError, ValueError):
                pass

        if legacy_frame_var_name:
            raw_frames = os.getenv(legacy_frame_var_name)
            if raw_frames is not None:
                try:
                    legacy_frames = float(raw_frames)
                except (TypeError, ValueError):
                    legacy_frames = None
                if legacy_frames is not None and isfinite(legacy_frames):
                    return max(minimum, legacy_frames / self._timing_reference_fps)

        if default_seconds is not None:
            return max(minimum, float(default_seconds))
        if legacy_frame_default is not None:
            return max(minimum, float(legacy_frame_default) / self._timing_reference_fps)
        return max(minimum, 0.0)

    def __init__(
        self,
        db_handler,
        camera_index,
        exercise_choice="bicep_curl",
        face_recognizer=None,
        assigned_user_name=None,
        aruco_dict_type="DICT_5X5_100",
        parent=None,
    ):
        super().__init__(parent)
        self.db_handler = db_handler
        self.exercise_choice = exercise_choice
        self.face_recognizer = face_recognizer
        self.stop_requested = False
        self.camera_index = camera_index

        self.face_recognition_active = True
        self.exercise_analysis_active = False
        self.exercise_analyzer = None
        self.current_user_info = None
        self.exit_save_retry_limit = self._read_int_env(
            "SMART_MIRROR_EXIT_SAVE_RETRY_LIMIT", default=2, minimum=0,
        )
        self._exit_save_retry_count = 0

        self._timing_reference_fps = self._read_float_env(
            "SMART_MIRROR_WORKER_ANALYSIS_FPS",
            default=20.0,
            minimum=1.0,
        )
        self.UNKNOWN_FRAME_THRESHOLD = 15
        self.KNOWN_FRAME_THRESHOLD = 30
        self.KNOWN_FRAME_SWITCH_PENALTY = 12
        self.MIN_REGISTRATION_FACE_AREA = 36 * 36

        label_tracker = StablePrimaryLabelTracker(
            buildup_units_per_second=3.0 * self._timing_reference_fps,
            decay_units_per_second=1.0 * self._timing_reference_fps,
            min_score=6.0,
            switch_margin=4.0,
            forget_after_seconds=self._read_duration_env(
                "SMART_MIRROR_STABLE_LABEL_FORGET_SECONDS",
                legacy_frame_default=20,
                minimum=0.1,
            ),
        )
        self.recognition_state = FaceRecognitionState(
            unknown_threshold_seconds=self._read_duration_env(
                "SMART_MIRROR_UNKNOWN_STABLE_SECONDS",
                legacy_frame_default=self.UNKNOWN_FRAME_THRESHOLD,
                minimum=0.1,
            ),
            known_threshold_seconds=self._read_duration_env(
                "SMART_MIRROR_KNOWN_STABLE_SECONDS",
                legacy_frame_default=self.KNOWN_FRAME_THRESHOLD,
                minimum=0.1,
            ),
            switch_penalty_seconds=self._read_duration_env(
                "SMART_MIRROR_KNOWN_SWITCH_PENALTY_SECONDS",
                legacy_frame_default=self.KNOWN_FRAME_SWITCH_PENALTY,
                minimum=0.0,
            ),
            min_observation_seconds=max(0.01, 1.0 / self._timing_reference_fps),
            label_tracker=label_tracker,
        )
        registration_target_seconds = self._read_duration_env(
            "SMART_MIRROR_REGISTRATION_CAPTURE_SECONDS",
            legacy_frame_default=120,
            legacy_frame_var_name="SMART_MIRROR_REGISTRATION_CAPTURE_FRAMES",
            minimum=1.0,
        )
        registration_min_samples = self._read_int_env(
            "SMART_MIRROR_REGISTRATION_MIN_SAMPLES",
            default=max(12, int(round(registration_target_seconds * 4.0))),
            minimum=1,
        )
        registration_max_samples = self._read_int_env(
            "SMART_MIRROR_REGISTRATION_MAX_SAMPLES",
            default=max(registration_min_samples, int(round(registration_target_seconds * 20.0))),
            minimum=registration_min_samples,
        )
        self.registration_state = RegistrationCaptureState(
            target_capture_seconds=registration_target_seconds,
            minimum_samples=registration_min_samples,
            maximum_samples=registration_max_samples,
        )
        self._analysis_mode_last_timing_ts = 0.0
        self._face_mode_last_timing_ts = 0.0
        self._registration_mode_last_timing_ts = 0.0

        self.assigned_user_name = (assigned_user_name or "Athlete").strip() or "Athlete"
        self.display_user_name = self.assigned_user_name
        self.last_audio_message = ""
        self._last_audio_emit_ts = 0.0
        self._audio_min_gap_sec = 1.3
        self._audio_repeat_suppression_sec = 4.0
        self._audio_warning_active = False
        self._audio_ready_announced = False
        self._audio_last_set_announced = -1
        self._audio_last_rep_marker = (-1, -1)
        self._audio_last_range_marker = None
        self._rep_progress_key = (0, 0)
        self._rep_progress_ts = time.monotonic()
        self.aruco_dict_type = ArucoDetector.normalize_dict_type(aruco_dict_type)
        self.disable_aruco = os.environ.get("SMART_MIRROR_DISABLE_ARUCO", "0") == "1"
        self.exit_no_pose_frames = self._read_int_env(
            "SMART_MIRROR_EXIT_NO_POSE_FRAMES",
            default=10,
            minimum=3,
        )
        self.exit_reacquire_grace_seconds = self._read_float_env(
            "SMART_MIRROR_EXIT_REACQUIRE_GRACE_SECONDS",
            default=0.65,
            minimum=0.0,
        )
        self.pose_acquire_grace_seconds = self._read_duration_env(
            "SMART_MIRROR_POSE_ACQUIRE_GRACE_SECONDS",
            legacy_frame_default=14,
            legacy_frame_var_name="SMART_MIRROR_POSE_ACQUIRE_GRACE_FRAMES",
            minimum=0.0,
        )
        self.pose_visibility_threshold = self._read_float_env(
            "SMART_MIRROR_MIN_POSE_VISIBILITY",
            default=0.45,
            minimum=0.1,
        )
        self.min_pose_shoulder_width = self._read_float_env(
            "SMART_MIRROR_MIN_POSE_SHOULDER_WIDTH",
            default=0.05,
            minimum=0.01,
        )
        self.min_pose_torso_height = self._read_float_env(
            "SMART_MIRROR_MIN_POSE_TORSO_HEIGHT",
            default=0.08,
            minimum=0.02,
        )
        self._analysis_pose_grace_remaining_sec = 0.0
        self._last_reliable_pose_ts = 0.0
        self.loop_fps = self._read_float_env(
            "SMART_MIRROR_WORKER_LOOP_FPS",
            default=0.0,
            minimum=0.0,
        )
        self.analysis_fps = self._timing_reference_fps
        self.ui_fps = self._read_float_env(
            "SMART_MIRROR_WORKER_UI_FPS",
            default=0.0,
            minimum=0.0,
        )
        self.thumbnail_fps = self._read_float_env(
            "SMART_MIRROR_WORKER_THUMBNAIL_FPS",
            default=8.0,
            minimum=1.0,
        )
        self.counter_fps = self._read_float_env(
            "SMART_MIRROR_WORKER_COUNTER_FPS",
            default=10.0,
            minimum=1.0,
        )
        self._loop_interval_sec = 0.0 if self.loop_fps <= 0 else 1.0 / self.loop_fps
        self._analysis_interval_sec = 0.0 if self.analysis_fps <= 0 else 1.0 / self.analysis_fps
        self._ui_interval_sec = 0.0 if self.ui_fps <= 0 else 1.0 / self.ui_fps
        self._thumbnail_interval_sec = (
            0.0 if self.thumbnail_fps <= 0 else 1.0 / self.thumbnail_fps
        )
        self._counter_interval_sec = 0.0 if self.counter_fps <= 0 else 1.0 / self.counter_fps
        self._last_analysis_ts = 0.0
        self._last_ui_emit_ts = 0.0
        self._last_thumbnail_emit_ts = 0.0
        self._last_counter_emit_ts = 0.0
        self._last_counter_value = (-1, -1)
        self._overlay_timestamp_cache = ""
        self._overlay_timestamp_second = -1
        self._pose_overlay_hold_sec = self._read_float_env(
            "SMART_MIRROR_POSE_OVERLAY_HOLD_SEC",
            default=max(0.08, self._analysis_interval_sec * 2.5),
            minimum=0.05,
        )
        self._face_box_hold_sec = self._read_float_env(
            "SMART_MIRROR_FACE_BOX_HOLD_SEC",
            default=max(0.08, self._analysis_interval_sec * 2.5),
            minimum=0.05,
        )
        self.tracked_user_hold_seconds = self._read_float_env(
            "SMART_MIRROR_TRACKED_USER_HOLD_SECONDS",
            default=5.0,
            minimum=0.5,
        )
        self.tracked_user_missing_timeout_seconds = self._read_float_env(
            "SMART_MIRROR_TRACKED_USER_MISSING_TIMEOUT_SECONDS",
            default=max(30.0, self.tracked_user_hold_seconds * 6.0),
            minimum=self.tracked_user_hold_seconds,
        )
        self.tracked_user_missing_warning_seconds = self._read_float_env(
            "SMART_MIRROR_TRACKED_USER_MISSING_WARNING_SECONDS",
            default=5.0,
            minimum=0.5,
        )
        self.tracked_user_stale_countdown_seconds = self._read_float_env(
            "SMART_MIRROR_TRACKED_USER_STALE_COUNTDOWN_SECONDS",
            default=15.0,
            minimum=self.tracked_user_missing_warning_seconds,
        )
        self.target_exit_confirm_seconds = self._read_float_env(
            "SMART_MIRROR_TARGET_EXIT_CONFIRM_SECONDS",
            default=0.8,
            minimum=0.2,
        )
        self.target_exit_edge_margin_ratio = self._read_float_env(
            "SMART_MIRROR_TARGET_EXIT_EDGE_MARGIN_RATIO",
            default=0.07,
            minimum=0.01,
        )
        self.target_lock_roi_scale = self._read_float_env(
            "SMART_MIRROR_TARGET_LOCK_ROI_SCALE",
            default=1.65,
            minimum=1.1,
        )
        self.target_lock_face_recheck_seconds = self._read_duration_env(
            "SMART_MIRROR_TARGET_LOCK_FACE_RECHECK_SECONDS",
            default_seconds=max(0.25, float(self.analysis_fps) * 0.0 + 0.6),
            legacy_frame_var_name="SMART_MIRROR_TARGET_LOCK_FACE_RECHECK_FRAMES",
            minimum=0.1,
        )
        self.target_lock_max_center_drift = self._read_float_env(
            "SMART_MIRROR_TARGET_LOCK_MAX_CENTER_DRIFT",
            default=0.42,
            minimum=0.1,
        )
        self.target_lock_max_scale_drift = self._read_float_env(
            "SMART_MIRROR_TARGET_LOCK_MAX_SCALE_DRIFT",
            default=1.6,
            minimum=1.05,
        )
        self.target_lock_min_profile_score = self._read_float_env(
            "SMART_MIRROR_TARGET_LOCK_MIN_PROFILE_SCORE",
            default=0.42,
            minimum=0.0,
        )
        self.target_lock_min_appearance_similarity = self._read_float_env(
            "SMART_MIRROR_TARGET_LOCK_MIN_APPEARANCE_SIMILARITY",
            default=0.24,
            minimum=0.0,
        )
        self.target_lock_appearance_update_alpha = min(
            1.0,
            self._read_float_env(
                "SMART_MIRROR_TARGET_LOCK_APPEARANCE_UPDATE_ALPHA",
                default=0.12,
                minimum=0.01,
            ),
        )
        self.end_session_palm_hold_seconds = self._read_float_env(
            "SMART_MIRROR_END_SESSION_PALM_HOLD_SECONDS",
            default=3.0,
            minimum=0.5,
        )
        self.end_session_palm_enabled = self._read_bool_env(
            "SMART_MIRROR_END_SESSION_PALM_ENABLED",
            default=True,
        )
        self.full_frame_pose_acquire = self._read_bool_env(
            "SMART_MIRROR_FULL_FRAME_POSE_ACQUIRE",
            default=True,
        )
        self.pose_acquire_relaxed_threshold_factor = min(
            1.0,
            self._read_float_env(
                "SMART_MIRROR_POSE_ACQUIRE_RELAXED_THRESHOLD_FACTOR",
                default=0.48,
                minimum=0.35,
            ),
        )
        self.pose_overlay_smoothing_alpha = self._read_float_env(
            "SMART_MIRROR_POSE_OVERLAY_SMOOTHING_ALPHA",
            default=0.22,
            minimum=0.05,
        )
        self.pose_overlay_smoothing_max_alpha = self._read_float_env(
            "SMART_MIRROR_POSE_OVERLAY_SMOOTHING_MAX_ALPHA",
            default=0.55,
            minimum=0.1,
        )
        self.pose_overlay_motion_boost = self._read_float_env(
            "SMART_MIRROR_POSE_OVERLAY_MOTION_BOOST",
            default=3.0,
            minimum=0.0,
        )
        self.pose_overlay_visibility_floor = self._read_float_env(
            "SMART_MIRROR_POSE_OVERLAY_VISIBILITY_FLOOR",
            default=0.24,
            minimum=0.05,
        )
        self.pose_overlay_glow_strength = self._read_float_env(
            "SMART_MIRROR_POSE_OVERLAY_GLOW_STRENGTH",
            default=0.34,
            minimum=0.0,
        )
        self._cached_pose_points = None
        self._smoothed_pose_points = None
        self._cached_pose_colors = ((118, 242, 130), (118, 242, 130))
        self._cached_pose_ts = 0.0
        self._cached_face_boxes = []
        self._cached_face_boxes_ts = 0.0
        self.locked_username = None
        self.locked_face_box = None
        self.locked_body_box = None
        self.locked_body_center = None
        self.locked_body_size = None
        self.target_lost_since = None
        self.target_exit_started_since = None
        self.last_identity_confirmed_ts = 0.0
        self._target_face_recheck_elapsed_sec = 0.0
        self._pose_lock_established = False
        self.session_identity_profile = None
        self._end_session_palm_started_since = None

    def request_stop(self):
        self.stop_requested = True
        logging.info("Stop requested for ExerciseWorker.")

    def _mode_elapsed_seconds(self, timestamp_attr, now, default_step=None):
        default_step = (
            self._analysis_interval_sec
            if default_step is None and self._analysis_interval_sec > 0
            else default_step
        )
        if default_step is None or default_step <= 0.0:
            default_step = max(0.01, 1.0 / self._timing_reference_fps)

        previous = float(getattr(self, timestamp_attr, 0.0) or 0.0)
        setattr(self, timestamp_attr, float(now))
        if previous <= 0.0:
            return float(default_step)
        return min(1.0, max(0.0, float(now) - previous))

    def start_record_new_user(self, user_name):
        normalized_user_name = str(user_name or "").strip() or self.assigned_user_name
        self.registration_state.start(normalized_user_name)
        self.exercise_analysis_active = False
        self.face_recognition_active = False
        self.exercise_analyzer = None
        self.current_user_info = None
        self.display_user_name = normalized_user_name
        self._analysis_mode_last_timing_ts = 0.0
        self._registration_mode_last_timing_ts = 0.0
        self._reset_face_recognition_state()
        self._reset_tracked_user_lock()
        self.status_signal.emit(f"Capturing face data for {normalized_user_name}. Please wait...")
        logging.info(f"Started capturing face data for new user: {normalized_user_name}")

    def _reset_face_recognition_state(self):
        self.recognition_state.reset()
        self._face_mode_last_timing_ts = 0.0

    def _switch_to_face_recognition_mode(self, reason="Returning to Face Recognition Mode."):
        self.exercise_analyzer = None
        self.exercise_analysis_active = False
        self.face_recognition_active = True
        self.registration_state.reset()
        self.current_user_info = None
        self.display_user_name = self.assigned_user_name
        self._clear_pose_overlay_cache()
        self._exit_save_retry_count = 0
        self._analysis_pose_grace_remaining_sec = 0.0
        self._last_reliable_pose_ts = 0.0
        self._analysis_mode_last_timing_ts = 0.0
        self._registration_mode_last_timing_ts = 0.0
        self._reset_tracked_user_lock()
        self._reset_face_recognition_state()
        self.status_signal.emit(reason)
        logging.info(reason)

    def _reset_tracked_user_lock(self):
        self.locked_username = None
        self.locked_face_box = None
        self.locked_body_box = None
        self.locked_body_center = None
        self.locked_body_size = None
        self.target_lost_since = None
        self.target_exit_started_since = None
        self.last_identity_confirmed_ts = 0.0
        self._target_face_recheck_elapsed_sec = 0.0
        self._pose_lock_established = False
        self.session_identity_profile = None
        self._end_session_palm_started_since = None

    def _face_location_scale(self):
        inference_config = getattr(self.face_recognizer, "_inference_config", {}) or {}
        try:
            frame_scale = float(inference_config.get("frame_scale", 0.25))
        except (TypeError, ValueError):
            frame_scale = 0.25
        if frame_scale <= 0.0 or frame_scale > 1.0:
            return 0.25
        return frame_scale

    def _normalize_pixel_box(self, left, top, right, bottom, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        left = int(max(0, min(frame_w - 1, round(left))))
        top = int(max(0, min(frame_h - 1, round(top))))
        right = int(max(left + 1, min(frame_w, round(right))))
        bottom = int(max(top + 1, min(frame_h, round(bottom))))
        return (left, top, right, bottom)

    def _box_from_center_size(self, center, size, frame_shape):
        if not center or not size:
            return None
        center_x, center_y = center
        width, height = size
        half_w = max(1.0, float(width) / 2.0)
        half_h = max(1.0, float(height) / 2.0)
        return self._normalize_pixel_box(
            float(center_x) - half_w,
            float(center_y) - half_h,
            float(center_x) + half_w,
            float(center_y) + half_h,
            frame_shape,
        )

    @staticmethod
    def _box_center(box):
        if not box:
            return None
        left, top, right, bottom = box
        return (
            (float(left) + float(right)) / 2.0,
            (float(top) + float(bottom)) / 2.0,
        )

    @staticmethod
    def _box_size(box):
        if not box:
            return None
        left, top, right, bottom = box
        return (
            max(1.0, float(right - left)),
            max(1.0, float(bottom - top)),
        )

    @staticmethod
    def _box_iou(first_box, second_box):
        if not first_box or not second_box:
            return 0.0
        first_left, first_top, first_right, first_bottom = first_box
        second_left, second_top, second_right, second_bottom = second_box

        intersect_left = max(first_left, second_left)
        intersect_top = max(first_top, second_top)
        intersect_right = min(first_right, second_right)
        intersect_bottom = min(first_bottom, second_bottom)
        if intersect_right <= intersect_left or intersect_bottom <= intersect_top:
            return 0.0

        intersection_area = float(intersect_right - intersect_left) * float(
            intersect_bottom - intersect_top
        )
        first_area = max(1.0, float(first_right - first_left) * float(first_bottom - first_top))
        second_area = max(
            1.0,
            float(second_right - second_left) * float(second_bottom - second_top),
        )
        union_area = first_area + second_area - intersection_area
        if union_area <= 0.0:
            return 0.0
        return intersection_area / union_area

    def _box_near_frame_edge(self, box, frame_shape, margin_ratio=None):
        if not box or frame_shape is None:
            return False
        frame_h, frame_w = frame_shape[:2]
        margin_ratio = (
            self.target_exit_edge_margin_ratio
            if margin_ratio is None
            else max(0.0, float(margin_ratio))
        )
        margin_x = max(8.0, float(frame_w) * margin_ratio)
        margin_y = max(8.0, float(frame_h) * margin_ratio)
        left, top, right, bottom = box
        return (
            float(left) <= margin_x
            or float(top) <= margin_y
            or float(right) >= (float(frame_w) - margin_x)
            or float(bottom) >= (float(frame_h) - margin_y)
        )

    def _face_location_to_pixels(self, face_location, frame_shape, offset=(0, 0)):
        if not face_location:
            return None
        scale = self._face_location_scale()
        upscale = 1.0 / scale if scale > 0 else 4.0
        top, right, bottom, left = face_location
        return self._normalize_pixel_box(
            (float(left) * upscale) + float(offset[0]),
            (float(top) * upscale) + float(offset[1]),
            (float(right) * upscale) + float(offset[0]),
            (float(bottom) * upscale) + float(offset[1]),
            frame_shape,
        )

    def _estimate_body_lock_from_face_box(self, face_box, frame_shape):
        if not face_box:
            return None, None
        frame_h, frame_w = frame_shape[:2]
        left, top, right, bottom = face_box
        face_w = max(1.0, float(right - left))
        face_h = max(1.0, float(bottom - top))

        body_w = min(float(frame_w), max(face_w * 6.0, float(frame_w) * 0.22))
        body_h = min(float(frame_h), max(face_h * 13.0, float(frame_h) * 0.48))
        center_x = float(left) + (face_w / 2.0)
        center_y = float(top) + (face_h * 4.5)
        half_w = body_w / 2.0
        half_h = body_h / 2.0
        center_x = min(max(center_x, half_w), max(half_w, float(frame_w) - half_w))
        center_y = min(max(center_y, half_h), max(half_h, float(frame_h) - half_h))
        return (center_x, center_y), (body_w, body_h)

    def _estimate_face_box_from_pose(self, landmarks, frame_shape):
        if not landmarks:
            return None

        visibility_floor = max(0.12, self.pose_visibility_threshold - 0.24)
        face_indices = [
            mp_pose.PoseLandmark.NOSE.value,
            mp_pose.PoseLandmark.LEFT_EYE.value,
            mp_pose.PoseLandmark.RIGHT_EYE.value,
            mp_pose.PoseLandmark.LEFT_EAR.value,
            mp_pose.PoseLandmark.RIGHT_EAR.value,
        ]
        frame_h, frame_w = frame_shape[:2]
        face_coords = []
        for landmark_idx in face_indices:
            if landmark_idx >= len(landmarks):
                continue
            landmark = landmarks[landmark_idx]
            if getattr(landmark, "visibility", 1.0) < visibility_floor:
                continue
            face_coords.append(
                (
                    min(1.0, max(0.0, float(landmark.x))) * float(frame_w),
                    min(1.0, max(0.0, float(landmark.y))) * float(frame_h),
                )
            )

        shoulder_coords = []
        for landmark_idx in (
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        ):
            if landmark_idx >= len(landmarks):
                continue
            landmark = landmarks[landmark_idx]
            if getattr(landmark, "visibility", 1.0) < visibility_floor:
                continue
            shoulder_coords.append(
                (
                    min(1.0, max(0.0, float(landmark.x))) * float(frame_w),
                    min(1.0, max(0.0, float(landmark.y))) * float(frame_h),
                )
            )

        if len(face_coords) >= 2:
            xs = [coord[0] for coord in face_coords]
            ys = [coord[1] for coord in face_coords]
            center_x = sum(xs) / float(len(xs))
            top_y = min(ys)
            bottom_y = max(ys)
            face_span = max(xs) - min(xs)
            shoulder_span = (
                abs(float(shoulder_coords[1][0]) - float(shoulder_coords[0][0]))
                if len(shoulder_coords) == 2
                else 0.0
            )
            head_width = max(face_span * 1.9, shoulder_span * 0.42, 42.0)
            shoulder_y = (
                min(float(shoulder_coords[0][1]), float(shoulder_coords[1][1]))
                if len(shoulder_coords) == 2
                else None
            )
            lower_extent = shoulder_y if shoulder_y is not None else (bottom_y + (head_width * 0.45))
            head_height = max((float(lower_extent) - float(top_y)) * 1.05, head_width * 1.18, 52.0)
        elif len(shoulder_coords) == 2:
            shoulder_left, shoulder_right = shoulder_coords
            center_x = (float(shoulder_left[0]) + float(shoulder_right[0])) / 2.0
            shoulder_y = min(float(shoulder_left[1]), float(shoulder_right[1]))
            shoulder_span = abs(float(shoulder_right[0]) - float(shoulder_left[0]))
            head_width = max(42.0, shoulder_span * 0.40)
            head_height = max(52.0, head_width * 1.28)
            top_y = shoulder_y - (head_height * 0.92)
        else:
            return None

        return self._normalize_pixel_box(
            center_x - (head_width * 0.6),
            top_y - (head_height * 0.35),
            center_x + (head_width * 0.6),
            top_y + (head_height * 0.65),
            frame_shape,
        )

    def _body_appearance_box(self, body_box, frame_shape):
        if not body_box or frame_shape is None:
            return None
        frame_h, frame_w = frame_shape[:2]
        left, top, right, bottom = body_box
        body_w = max(1.0, float(right - left))
        body_h = max(1.0, float(bottom - top))
        crop_left = float(left) + (body_w * 0.14)
        crop_right = float(right) - (body_w * 0.14)
        crop_top = float(top) + (body_h * 0.20)
        crop_bottom = float(top) + (body_h * 0.66)
        if crop_right <= crop_left + 4 or crop_bottom <= crop_top + 4:
            crop_left, crop_top, crop_right, crop_bottom = left, top, right, bottom
        return self._normalize_pixel_box(
            crop_left,
            crop_top,
            crop_right,
            crop_bottom,
            (frame_h, frame_w),
        )

    def _extract_appearance_histogram(self, frame, body_box):
        if frame is None or body_box is None or getattr(frame, "size", 0) == 0:
            return None
        crop_box = self._body_appearance_box(body_box, frame.shape)
        crop = self._crop_frame_to_roi(frame, crop_box)
        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        try:
            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist(
                [hsv_crop],
                [0, 1],
                None,
                [24, 16],
                [0, 180, 0, 256],
            ).astype(np.float32)
            total = float(hist.sum())
            if total <= 1e-6:
                return None
            return (hist / total).reshape(-1)
        except Exception:
            return None

    @staticmethod
    def _appearance_similarity(reference_hist, candidate_hist):
        if reference_hist is None or candidate_hist is None:
            return None
        reference = np.asarray(reference_hist, dtype=np.float32).reshape(-1)
        candidate = np.asarray(candidate_hist, dtype=np.float32).reshape(-1)
        if reference.shape != candidate.shape or reference.size == 0:
            return None
        try:
            correlation = float(cv2.compareHist(reference, candidate, cv2.HISTCMP_CORREL))
        except Exception:
            return None
        return max(0.0, min(1.0, correlation))

    def _ensure_session_identity_profile(self):
        username = (str(self.locked_username or "").strip() or None)
        if not username:
            return None
        if (
            not isinstance(self.session_identity_profile, dict)
            or self.session_identity_profile.get("username") != username
        ):
            self.session_identity_profile = {
                "username": username,
                "appearance_hist": None,
                "appearance_samples": 0,
                "body_aspect": None,
                "body_area_ratio": None,
                "last_body_box": None,
                "last_face_box": None,
                "last_profile_update_ts": 0.0,
                "last_match_score": None,
                "last_appearance_similarity": None,
            }
        return self.session_identity_profile

    def _body_profile_metrics(self, body_box, frame_shape):
        if not body_box or frame_shape is None:
            return None
        frame_h, frame_w = frame_shape[:2]
        left, top, right, bottom = body_box
        width = max(1.0, float(right - left))
        height = max(1.0, float(bottom - top))
        frame_area = max(1.0, float(frame_w) * float(frame_h))
        return {
            "body_aspect": height / width,
            "body_area_ratio": (width * height) / frame_area,
        }

    def _update_session_identity_profile(
        self,
        frame=None,
        body_box=None,
        face_box=None,
        now=0.0,
        strong_identity=False,
    ):
        profile = self._ensure_session_identity_profile()
        if profile is None:
            return None

        body_box = body_box or self.locked_body_box
        face_box = face_box or self.locked_face_box
        if body_box is not None:
            profile["last_body_box"] = body_box
            metrics = self._body_profile_metrics(
                body_box,
                frame.shape if frame is not None else None,
            )
            if metrics:
                for key in ("body_aspect", "body_area_ratio"):
                    existing = profile.get(key)
                    value = metrics[key]
                    if existing is None or strong_identity:
                        profile[key] = float(value)
                    else:
                        profile[key] = (0.90 * float(existing)) + (0.10 * float(value))
        if face_box is not None:
            profile["last_face_box"] = face_box

        appearance_hist = self._extract_appearance_histogram(frame, body_box)
        if appearance_hist is not None:
            existing_hist = profile.get("appearance_hist")
            if existing_hist is None:
                profile["appearance_hist"] = appearance_hist
            else:
                alpha = 0.35 if strong_identity else self.target_lock_appearance_update_alpha
                blended = ((1.0 - alpha) * np.asarray(existing_hist, dtype=np.float32)) + (
                    alpha * appearance_hist
                )
                total = float(blended.sum())
                profile["appearance_hist"] = blended / total if total > 1e-6 else appearance_hist
            profile["appearance_samples"] = int(profile.get("appearance_samples") or 0) + 1

        if now:
            profile["last_profile_update_ts"] = float(now)
        return profile

    def _score_pose_against_session_profile(self, frame, pose_lock):
        profile = self._ensure_session_identity_profile()
        if profile is None or not pose_lock:
            return {"matches": True, "score": 1.0, "appearance_similarity": None}

        center = pose_lock.get("center")
        size = pose_lock.get("size")
        candidate_box = pose_lock.get("box")
        candidate_w, candidate_h = size if size else (0.0, 0.0)
        current_w, current_h = self.locked_body_size or (candidate_w, candidate_h)
        current_center = self.locked_body_center or center

        position_score = 1.0
        if center and current_center:
            center_reference = max(48.0, float(current_w), float(current_h))
            distance = hypot(
                float(center[0]) - float(current_center[0]),
                float(center[1]) - float(current_center[1]),
            )
            position_score = 1.0 - min(1.0, distance / max(center_reference, 1.0))

        scale_score = 1.0
        if candidate_w > 0 and candidate_h > 0 and current_w > 0 and current_h > 0:
            scale_ratio = max(
                float(candidate_w) / max(float(current_w), 1.0),
                float(current_w) / max(float(candidate_w), 1.0),
                float(candidate_h) / max(float(current_h), 1.0),
                float(current_h) / max(float(candidate_h), 1.0),
            )
            scale_score = 1.0 / max(1.0, scale_ratio)

        aspect_score = 0.5
        profile_aspect = profile.get("body_aspect")
        if profile_aspect and candidate_w > 0 and candidate_h > 0:
            candidate_aspect = float(candidate_h) / max(float(candidate_w), 1.0)
            aspect_delta = abs(np.log(max(candidate_aspect, 1e-6) / max(float(profile_aspect), 1e-6)))
            aspect_score = 1.0 - min(1.0, aspect_delta / max(np.log(2.2), 1e-6))

        candidate_hist = self._extract_appearance_histogram(frame, candidate_box)
        appearance_similarity = self._appearance_similarity(
            profile.get("appearance_hist"),
            candidate_hist,
        )
        appearance_score = 0.5 if appearance_similarity is None else appearance_similarity

        score = (
            (0.34 * position_score)
            + (0.20 * scale_score)
            + (0.16 * aspect_score)
            + (0.30 * appearance_score)
        )
        appearance_samples = int(profile.get("appearance_samples") or 0)
        relaxed_initial = not self._pose_lock_established
        min_profile_score = self.target_lock_min_profile_score
        min_appearance = self.target_lock_min_appearance_similarity
        if relaxed_initial:
            min_profile_score *= 0.72
            min_appearance *= 0.60

        matches = True
        if (
            appearance_samples > 0
            and appearance_similarity is not None
            and (self._pose_lock_established or appearance_samples >= 2)
        ):
            matches = appearance_similarity >= min_appearance
        if matches and appearance_samples > 0:
            matches = score >= min_profile_score

        profile["last_match_score"] = float(score)
        profile["last_appearance_similarity"] = appearance_similarity
        return {
            "matches": bool(matches),
            "score": float(score),
            "appearance_similarity": appearance_similarity,
            "position_score": float(position_score),
            "scale_score": float(scale_score),
            "aspect_score": float(aspect_score),
        }

    def _seed_tracked_user_lock(self, username, face_location, frame_shape, now, frame=None):
        self.locked_username = (str(username or "").strip() or None)
        self.locked_face_box = self._face_location_to_pixels(face_location, frame_shape)
        body_center, body_size = self._estimate_body_lock_from_face_box(
            self.locked_face_box,
            frame_shape,
        )
        self.locked_body_box = self._box_from_center_size(body_center, body_size, frame_shape)
        self.locked_body_center = body_center
        self.locked_body_size = body_size
        self.target_lost_since = None
        self.target_exit_started_since = None
        self.last_identity_confirmed_ts = float(now or 0.0)
        self._target_face_recheck_elapsed_sec = 0.0
        self.session_identity_profile = None
        self._update_session_identity_profile(
            frame=frame,
            body_box=self.locked_body_box,
            face_box=self.locked_face_box,
            now=now,
            strong_identity=True,
        )

    def _build_target_lock_roi(self, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        if self.locked_body_center and self.locked_body_size:
            center_x, center_y = self.locked_body_center
            body_w, body_h = self.locked_body_size
        elif self.locked_face_box:
            body_center, body_size = self._estimate_body_lock_from_face_box(
                self.locked_face_box,
                frame_shape,
            )
            if not body_center or not body_size:
                return (0, 0, frame_w, frame_h)
            center_x, center_y = body_center
            body_w, body_h = body_size
        else:
            return (0, 0, frame_w, frame_h)

        roi_w = min(float(frame_w), max(96.0, float(body_w) * self.target_lock_roi_scale))
        roi_h = min(float(frame_h), max(160.0, float(body_h) * self.target_lock_roi_scale))
        left = float(center_x) - (roi_w / 2.0)
        top = float(center_y) - (roi_h / 2.0)
        right = left + roi_w
        bottom = top + roi_h
        return self._normalize_pixel_box(left, top, right, bottom, frame_shape)

    def _full_frame_box(self, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        return (0, 0, frame_w, frame_h)

    def _resolve_analysis_roi_box(self, frame_shape):
        if self.full_frame_pose_acquire and not self._pose_lock_established:
            return self._full_frame_box(frame_shape)
        return self._build_target_lock_roi(frame_shape)

    def _crop_frame_to_roi(self, frame, roi_box):
        if frame is None or roi_box is None:
            return None
        left, top, right, bottom = roi_box
        if right <= left or bottom <= top:
            return None
        roi = frame[top:bottom, left:right]
        if roi is None or getattr(roi, "size", 0) == 0:
            return None
        return roi

    def _remap_pose_landmarks_to_frame(self, pose_landmarks, roi_box, frame_shape):
        if pose_landmarks is None:
            return
        left, top, right, bottom = roi_box
        roi_w = max(1, right - left)
        roi_h = max(1, bottom - top)
        frame_h, frame_w = frame_shape[:2]
        for landmark in pose_landmarks.landmark:
            x, y = float(landmark.x), float(landmark.y)
            z = float(getattr(landmark, "z", 0.0))
            visibility = float(getattr(landmark, "visibility", 0.0))
            if not all(isfinite(value) for value in (x, y, z, visibility)) or not 0 <= visibility <= 1:
                # min/max would turn NaN into an apparently valid border point.
                landmark.visibility = 0.0
                continue
            landmark.x = (float(left) + (x * roi_w)) / float(frame_w)
            landmark.y = (float(top) + (y * roi_h)) / float(frame_h)
            # MediaPipe z uses the same scale as x, relative to the input width.
            # Keep depth gates consistent when pose inference switches to a crop.
            landmark.z = z * float(roi_w) / float(frame_w)
            if not (0.0 <= landmark.x <= 1.0 and 0.0 <= landmark.y <= 1.0):
                # Preserve the geometry instead of clamping a limb into view.
                landmark.visibility = 0.0

    def _landmark_bounds(self, landmarks, frame_shape):
        if not landmarks:
            return None
        frame_h, frame_w = frame_shape[:2]
        visibility_floor = max(0.2, self.pose_visibility_threshold - 0.15)
        coords = []
        for landmark in landmarks:
            visibility = getattr(landmark, "visibility", 1.0)
            if visibility < visibility_floor:
                continue
            coords.append(
                (
                    min(1.0, max(0.0, float(landmark.x))) * float(frame_w),
                    min(1.0, max(0.0, float(landmark.y))) * float(frame_h),
                )
            )
        if len(coords) < 4:
            coords = [
                (
                    min(1.0, max(0.0, float(landmark.x))) * float(frame_w),
                    min(1.0, max(0.0, float(landmark.y))) * float(frame_h),
                )
                for landmark in landmarks
            ]
        if not coords:
            return None
        xs = [coord[0] for coord in coords]
        ys = [coord[1] for coord in coords]
        return self._normalize_pixel_box(min(xs), min(ys), max(xs), max(ys), frame_shape)

    def _resolve_pose_lock(self, landmarks, frame_shape):
        body_box = self._landmark_bounds(landmarks, frame_shape)
        if not body_box:
            return None
        left, top, right, bottom = body_box
        return {
            "box": body_box,
            "center": (
                (float(left) + float(right)) / 2.0,
                (float(top) + float(bottom)) / 2.0,
            ),
            "size": (
                max(48.0, float(right - left)),
                max(96.0, float(bottom - top)),
            ),
            "face_box": self._estimate_face_box_from_pose(landmarks, frame_shape),
        }

    def _pose_matches_target_lock(
        self,
        center,
        size,
        candidate_box=None,
        candidate_face_box=None,
    ):
        if not center or not size:
            return False
        if not self.locked_body_center or not self.locked_body_size:
            return True

        current_x, current_y = self.locked_body_center
        current_w, current_h = self.locked_body_size
        candidate_x, candidate_y = center
        candidate_w, candidate_h = size
        center_reference = max(48.0, float(current_w), float(current_h))
        center_distance = hypot(candidate_x - current_x, candidate_y - current_y)
        max_center_drift = self.target_lock_max_center_drift
        max_scale_drift = self.target_lock_max_scale_drift
        if not self._pose_lock_established:
            # The initial face box only approximates the athlete's body. Keep the
            # first pose acquisition permissive so portrait setups do not bounce
            # back into face-recognition mode before the skeleton appears.
            max_center_drift = max(max_center_drift, 0.72)
            max_scale_drift = max(max_scale_drift, 2.3)

        if center_distance > (max_center_drift * center_reference):
            return False

        scale_ratio = max(
            float(candidate_w) / max(float(current_w), 1.0),
            float(current_w) / max(float(candidate_w), 1.0),
            float(candidate_h) / max(float(current_h), 1.0),
            float(current_h) / max(float(candidate_h), 1.0),
        )
        if scale_ratio > max_scale_drift:
            return False

        if candidate_box and self.locked_body_box:
            body_iou = self._box_iou(candidate_box, self.locked_body_box)
            min_body_iou = 0.10 if not self._pose_lock_established else 0.20
            if body_iou < min_body_iou and center_distance > (max_center_drift * center_reference * 0.72):
                return False

        if candidate_face_box and self.locked_face_box:
            candidate_face_center = self._box_center(candidate_face_box)
            locked_face_center = self._box_center(self.locked_face_box)
            if candidate_face_center and locked_face_center:
                locked_face_size = self._box_size(self.locked_face_box) or (0.0, 0.0)
                candidate_face_size = self._box_size(candidate_face_box) or (0.0, 0.0)
                locked_face_area = max(
                    1.0,
                    float(locked_face_size[0]) * float(locked_face_size[1]),
                )
                candidate_face_area = max(
                    1.0,
                    float(candidate_face_size[0]) * float(candidate_face_size[1]),
                )
                face_reference = max(
                    48.0,
                    float(locked_face_size[0]),
                    float(locked_face_size[1]),
                    float(candidate_face_size[0]),
                    float(candidate_face_size[1]),
                )
                face_center_distance = hypot(
                    float(candidate_face_center[0]) - float(locked_face_center[0]),
                    float(candidate_face_center[1]) - float(locked_face_center[1]),
                )
                face_area_ratio = max(
                    candidate_face_area / locked_face_area,
                    locked_face_area / candidate_face_area,
                )
                large_new_head_limit = 4.0 if not self._pose_lock_established else 3.0
                if (
                    face_area_ratio >= large_new_head_limit
                    and face_center_distance > max(42.0, face_reference * 0.22)
                ):
                    return False
                face_drift_limit = 0.90 if not self._pose_lock_established else 0.70
                if face_center_distance > (face_reference * face_drift_limit):
                    return False

        return True

    def _accept_pose_for_target_lock(self, landmarks, frame_shape, frame=None):
        pose_lock = self._resolve_pose_lock(landmarks, frame_shape)
        if not pose_lock:
            return False
        if not self._pose_matches_target_lock(
            pose_lock["center"],
            pose_lock["size"],
            candidate_box=pose_lock.get("box"),
            candidate_face_box=pose_lock.get("face_box"),
        ):
            return False
        profile_score = self._score_pose_against_session_profile(frame, pose_lock)
        if not profile_score.get("matches", True):
            logging.debug(
                "Rejected pose candidate for %s by session profile score: %.3f "
                "(appearance=%s).",
                self.locked_username,
                profile_score.get("score", 0.0),
                profile_score.get("appearance_similarity"),
            )
            return False
        self.locked_body_box = pose_lock["box"]
        self.locked_body_center = pose_lock["center"]
        self.locked_body_size = pose_lock["size"]
        if pose_lock.get("face_box") is not None:
            self.locked_face_box = pose_lock["face_box"]
        self.target_lost_since = None
        self.target_exit_started_since = None
        self._pose_lock_established = True
        self._update_session_identity_profile(
            frame=frame,
            body_box=self.locked_body_box,
            face_box=self.locked_face_box,
            now=time.monotonic(),
        )
        return True

    def _mark_target_visible(self):
        self.target_lost_since = None
        self.target_exit_started_since = None

    def _mark_target_lost(self, now):
        if self.target_lost_since is None:
            self.target_lost_since = float(now)
            return 0.0
        return max(0.0, float(now) - self.target_lost_since)

    def _target_anchor_near_frame_edge(self, frame_shape):
        return any(
            self._box_near_frame_edge(box, frame_shape)
            for box in (self.locked_face_box, self.locked_body_box)
            if box is not None
        )

    def _mark_target_missing(self, now, frame_shape):
        lost_elapsed = self._mark_target_lost(now)
        if self._target_anchor_near_frame_edge(frame_shape):
            if self.target_exit_started_since is None:
                self.target_exit_started_since = float(now)
        return lost_elapsed

    def _tracked_user_lost_elapsed(self, now):
        if self.target_lost_since is None:
            return 0.0
        return max(0.0, float(now) - self.target_lost_since)

    def _has_target_exited_view(self, now, frame_shape):
        if self.target_lost_since is None:
            return False

        lost_elapsed = self._tracked_user_lost_elapsed(now)
        if lost_elapsed < self.tracked_user_hold_seconds:
            return False

        if self.target_exit_started_since is not None:
            return (float(now) - float(self.target_exit_started_since)) >= self.target_exit_confirm_seconds

        if frame_shape is not None and self._target_anchor_near_frame_edge(frame_shape):
            return lost_elapsed >= (self.tracked_user_hold_seconds + self.target_exit_confirm_seconds)

        return lost_elapsed >= self.tracked_user_missing_timeout_seconds

    def _should_return_to_face_recognition(self, now, frame_shape=None):
        return self._has_target_exited_view(now, frame_shape)

    def _tracked_user_missing_status_hint(self, tracked_name, now):
        name = tracked_name or self.locked_username or self.assigned_user_name or "Athlete"
        lost_elapsed = self._tracked_user_lost_elapsed(now)
        timeout = max(
            self.tracked_user_missing_timeout_seconds,
            self.tracked_user_stale_countdown_seconds,
        )
        if lost_elapsed >= self.tracked_user_stale_countdown_seconds:
            remaining = max(0, int(round(timeout - lost_elapsed)))
            return f"{name} not visible. Session ending in {remaining}s."
        if lost_elapsed >= self.tracked_user_missing_warning_seconds:
            return f"{name} not visible. Step back in to resume or hold palm when visible to end."
        return f"Waiting for {name}."

    def _lock_marker_anchor(self, frame_shape):
        anchor_box = self.locked_face_box or self.locked_body_box
        if not anchor_box or frame_shape is None:
            return None
        frame_h, frame_w = frame_shape[:2]
        left, top, right, bottom = anchor_box
        anchor_x = int(round((float(left) + float(right)) / 2.0))
        anchor_y = int(round(float(top)))
        marker_y = max(18, anchor_y - max(18, int(round(float(bottom - top) * 0.28))))
        return (
            max(0, min(frame_w - 1, anchor_x)),
            max(0, min(frame_h - 1, marker_y)),
        )

    def _draw_tracked_user_lock_marker(self, frame, now, paused=False):
        if frame is None or getattr(frame, "size", 0) == 0 or not self.locked_username:
            return False
        anchor = self._lock_marker_anchor(frame.shape)
        if anchor is None:
            return False

        lost_elapsed = self._tracked_user_lost_elapsed(now)
        is_waiting = paused or self.target_lost_since is not None
        marker_color = (80, 220, 255) if is_waiting else (118, 242, 130)
        outline_color = (9, 18, 28)
        text = (
            f"WAITING {self.locked_username}"
            if is_waiting and lost_elapsed >= self.tracked_user_missing_warning_seconds
            else f"LOCKED {self.locked_username}"
        )

        x, y = anchor
        pointer = np.array(
            [
                [x, y + 12],
                [x - 12, y - 8],
                [x + 12, y - 8],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, pointer, outline_color, cv2.LINE_AA)
        inner_pointer = np.array(
            [
                [x, y + 8],
                [x - 8, y - 6],
                [x + 8, y - 6],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, inner_pointer, marker_color, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.52, min(frame.shape[:2]) / 1350.0)
        thickness = max(1, int(round(min(frame.shape[:2]) / 720.0)))
        text_size, _baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size
        pad_x = 9
        pad_y = 7
        label_left = max(
            4,
            min(
                frame.shape[1] - text_w - (pad_x * 2) - 4,
                x - (text_w // 2) - pad_x,
            ),
        )
        label_top = max(4, y - text_h - 32)
        label_right = label_left + text_w + (pad_x * 2)
        label_bottom = label_top + text_h + (pad_y * 2)
        cv2.rectangle(
            frame,
            (label_left, label_top),
            (label_right, label_bottom),
            outline_color,
            -1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            frame,
            (label_left, label_top),
            (label_right, label_bottom),
            marker_color,
            max(1, thickness),
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (label_left + pad_x, label_bottom - pad_y - 1),
            font,
            font_scale,
            (236, 244, 255),
            thickness,
            cv2.LINE_AA,
        )
        return True

    def _should_run_target_face_recheck(self, now, force=False):
        if not self.locked_username or self.face_recognizer is None:
            return False
        if force:
            return True
        if self._target_face_recheck_elapsed_sec >= self.target_lock_face_recheck_seconds:
            return True
        if self.last_identity_confirmed_ts <= 0.0:
            return True
        return (float(now) - self.last_identity_confirmed_ts) >= self.tracked_user_hold_seconds

    def _select_face_index_for_label(self, face_locations, face_names, desired_label):
        matching = [
            idx
            for idx, name in enumerate(face_names)
            if idx < len(face_locations) and name == desired_label
        ]
        if not matching:
            return None
        return max(
            matching,
            key=lambda idx: max(0, face_locations[idx][2] - face_locations[idx][0])
            * max(0, face_locations[idx][1] - face_locations[idx][3]),
        )

    def _run_locked_face_recheck(self, frame, now, roi_box=None):
        if not self.locked_username or self.face_recognizer is None:
            return {"status": "unavailable", "label": None, "face_box": None}

        self._target_face_recheck_elapsed_sec = 0.0
        full_frame_box = self._full_frame_box(frame.shape)
        roi_box = roi_box or self._resolve_analysis_roi_box(frame.shape)
        search_boxes = [roi_box]
        if tuple(roi_box) != tuple(full_frame_box):
            search_boxes.append(full_frame_box)

        fallback_result = {"status": "missing", "label": None, "face_box": None}
        for search_box in search_boxes:
            roi_frame = self._crop_frame_to_roi(frame, search_box)
            if roi_frame is None:
                continue

            left, top, _right, _bottom = search_box
            _, face_locations, face_names = self.face_recognizer.recognize_faces(roi_frame)
            if not face_locations or not face_names:
                continue

            matching_index = self._select_face_index_for_label(
                face_locations,
                face_names,
                self.locked_username,
            )
            if matching_index is not None:
                face_box = self._face_location_to_pixels(
                    face_locations[matching_index],
                    frame.shape,
                    offset=(left, top),
                )
                self.locked_face_box = face_box
                body_center, body_size = self._estimate_body_lock_from_face_box(
                    face_box,
                    frame.shape,
                )
                if body_center and body_size:
                    self.locked_body_box = self._box_from_center_size(body_center, body_size, frame.shape)
                    self.locked_body_center = body_center
                    self.locked_body_size = body_size
                self.display_user_name = self.locked_username
                self.last_identity_confirmed_ts = float(now)
                self._mark_target_visible()
                self._update_session_identity_profile(
                    frame=frame,
                    body_box=self.locked_body_box,
                    face_box=self.locked_face_box,
                    now=now,
                    strong_identity=True,
                )
                return {
                    "status": "confirmed",
                    "label": self.locked_username,
                    "face_box": face_box,
                }

            primary_index = self._select_primary_face_index(face_locations)
            if primary_index is None or primary_index >= len(face_names):
                continue

            fallback_face_box = self._face_location_to_pixels(
                face_locations[primary_index],
                frame.shape,
                offset=(left, top),
            )
            fallback_label = face_names[primary_index]
            if fallback_label and fallback_label != "Unknown":
                fallback_result = {
                    "status": "other",
                    "label": fallback_label,
                    "face_box": fallback_face_box,
                }
            elif fallback_result["status"] == "missing":
                fallback_result = {
                    "status": "unknown",
                    "label": fallback_label or "Unknown",
                    "face_box": fallback_face_box,
                }

        return fallback_result

    def _finalize_tracked_user_exit(self, now, exit_reason):
        self.status_signal.emit(f"{exit_reason}, updating database...")
        logging.info("%s. Updating database.", exit_reason)
        had_activity = bool(
            self.exercise_analyzer
            and (
                self.exercise_analyzer.rep_data
                or self.exercise_analyzer.rep_count > 0
                or self.exercise_analyzer.sets_reps
            )
        )
        record = self.exercise_analyzer.update_data() if self.exercise_analyzer else None
        if record:
            self._exit_save_retry_count = 0
            self.status_signal.emit("Exercise data saved locally.")
            logging.info("Data saved locally.")
            self.data_updated.emit()
            next_mode_reason = f"{exit_reason}. Ready for next user."
        elif had_activity:
            self._exit_save_retry_count = 0
            logging.error(
                "Exercise session had activity but save returned no record. "
                "Resetting session state for next user."
            )
            self.status_signal.emit(
                "Failed to save exercise data. Resetting session state for next user."
            )
            self.exercise_analyzer.reset_counters()
            next_mode_reason = f"{exit_reason}. Reset complete. Waiting for recognition."
        else:
            self._exit_save_retry_count = 0
            self.status_signal.emit("No exercise data to save.")
            next_mode_reason = f"{exit_reason}. Waiting for recognition."

        self._switch_to_face_recognition_mode(reason=next_mode_reason)
        self._emit_counters_if_due(0, 0, now, force=True)
        return next_mode_reason

    def _select_primary_face_index(self, face_locations):
        if not face_locations:
            return None
        return max(
            range(len(face_locations)),
            key=lambda idx: max(0, face_locations[idx][2] - face_locations[idx][0])
            * max(0, face_locations[idx][1] - face_locations[idx][3]),
        )

    def _update_stable_primary_label(self, observed_label, elapsed_seconds):
        return self.recognition_state.update_primary_label(observed_label, elapsed_seconds)

    def _has_reliable_pose(self, landmarks):
        if not landmarks:
            return False
        try:
            initial_pose_acquire = not self._pose_lock_established
            threshold_factor = (
                1.0
                if self._pose_lock_established
                else self.pose_acquire_relaxed_threshold_factor
            )
            pose_visibility_threshold = max(
                0.14 if initial_pose_acquire else 0.18,
                self.pose_visibility_threshold * threshold_factor,
            )
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

            def visible(point, threshold):
                confidence = float(getattr(point, "visibility", 0.0))
                return (
                    isfinite(confidence)
                    and threshold <= confidence <= 1.0
                    and all(isfinite(float(getattr(point, axis, 0.0))) for axis in ("x", "y", "z"))
                )

            left_shoulder_visible = visible(left_shoulder, pose_visibility_threshold)
            right_shoulder_visible = visible(right_shoulder, pose_visibility_threshold)
            if not (left_shoulder_visible and right_shoulder_visible):
                return False

            arm_visibility_threshold = max(
                0.18,
                max(0.2, self.pose_visibility_threshold - 0.15) * threshold_factor,
            )
            left_elbow_visible = visible(left_elbow, arm_visibility_threshold)
            right_elbow_visible = visible(right_elbow, arm_visibility_threshold)
            if self.exercise_choice == "deadlift":
                leg_visibility_threshold = max(
                    0.18,
                    max(0.2, self.pose_visibility_threshold - 0.15) * threshold_factor,
                )
                left_knee_visible = visible(left_knee, leg_visibility_threshold)
                right_knee_visible = visible(right_knee, leg_visibility_threshold)
                if not (left_knee_visible or right_knee_visible):
                    return False
            elif not (left_elbow_visible or right_elbow_visible):
                return False

            shoulder_width = hypot(
                float(left_shoulder.x) - float(right_shoulder.x),
                float(left_shoulder.y) - float(right_shoulder.y),
            )
            shoulder_width_floor = 0.014 if initial_pose_acquire else 0.02
            if shoulder_width < max(shoulder_width_floor, self.min_pose_shoulder_width * threshold_factor):
                return False

            left_hip_visible = visible(left_hip, pose_visibility_threshold)
            right_hip_visible = visible(right_hip, pose_visibility_threshold)
            if left_hip_visible and right_hip_visible:
                shoulder_mid_y = (float(left_shoulder.y) + float(right_shoulder.y)) / 2.0
                hip_mid_y = (float(left_hip.y) + float(right_hip.y)) / 2.0
                torso_height = abs(hip_mid_y - shoulder_mid_y)
                torso_height_floor = 0.022 if initial_pose_acquire else 0.03
                if torso_height < max(torso_height_floor, self.min_pose_torso_height * threshold_factor):
                    return False

            return True
        except Exception:
            return False

    @staticmethod
    def _pose_landmark_xy(landmarks, pose_landmark):
        point = landmarks[pose_landmark.value]
        return float(point.x), float(point.y)

    def _hand_open_palm_like(self, landmarks, side):
        if not landmarks:
            return False
        try:
            if side == "left":
                shoulder_lmk = mp_pose.PoseLandmark.LEFT_SHOULDER
                wrist_lmk = mp_pose.PoseLandmark.LEFT_WRIST
                index_lmk = mp_pose.PoseLandmark.LEFT_INDEX
                pinky_lmk = mp_pose.PoseLandmark.LEFT_PINKY
                thumb_lmk = mp_pose.PoseLandmark.LEFT_THUMB
            else:
                shoulder_lmk = mp_pose.PoseLandmark.RIGHT_SHOULDER
                wrist_lmk = mp_pose.PoseLandmark.RIGHT_WRIST
                index_lmk = mp_pose.PoseLandmark.RIGHT_INDEX
                pinky_lmk = mp_pose.PoseLandmark.RIGHT_PINKY
                thumb_lmk = mp_pose.PoseLandmark.RIGHT_THUMB

            visibility_floor = max(0.30, self.pose_visibility_threshold - 0.10)
            required = [wrist_lmk, index_lmk, pinky_lmk, thumb_lmk]
            if any(
                getattr(landmarks[landmark.value], "visibility", 0.0) < visibility_floor
                for landmark in required
            ):
                return False

            wrist = self._pose_landmark_xy(landmarks, wrist_lmk)
            index = self._pose_landmark_xy(landmarks, index_lmk)
            pinky = self._pose_landmark_xy(landmarks, pinky_lmk)
            thumb = self._pose_landmark_xy(landmarks, thumb_lmk)
            shoulder = self._pose_landmark_xy(landmarks, shoulder_lmk)
            left_shoulder = self._pose_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
            right_shoulder = self._pose_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)

            shoulder_width = hypot(
                left_shoulder[0] - right_shoulder[0],
                left_shoulder[1] - right_shoulder[1],
            )
            hand_spread = hypot(index[0] - pinky[0], index[1] - pinky[1])
            hand_reach = max(
                hypot(index[0] - wrist[0], index[1] - wrist[1]),
                hypot(pinky[0] - wrist[0], pinky[1] - wrist[1]),
                hypot(thumb[0] - wrist[0], thumb[1] - wrist[1]),
            )
            hand_is_open = (
                hand_spread >= max(0.018, shoulder_width * 0.12)
                and hand_reach >= max(0.028, shoulder_width * 0.16)
            )
            hand_is_raised = wrist[1] <= (shoulder[1] + 0.05)
            return bool(hand_is_open and hand_is_raised)
        except Exception:
            return False

    def _detect_end_session_palm_gesture(self, landmarks):
        if not self.end_session_palm_enabled:
            return False
        return self._hand_open_palm_like(landmarks, "left") or self._hand_open_palm_like(
            landmarks,
            "right",
        )

    def _reset_end_session_palm_hold(self):
        self._end_session_palm_started_since = None

    def _update_end_session_palm_hold(self, landmarks, now):
        if not self._detect_end_session_palm_gesture(landmarks):
            self._reset_end_session_palm_hold()
            return False, None

        if self._end_session_palm_started_since is None:
            self._end_session_palm_started_since = float(now)
        elapsed = max(0.0, float(now) - float(self._end_session_palm_started_since))
        hint = (
            f"Hold palm to end: {min(elapsed, self.end_session_palm_hold_seconds):.1f}s / "
            f"{self.end_session_palm_hold_seconds:.1f}s"
        )
        return elapsed >= self.end_session_palm_hold_seconds, hint

    def _emit_audio_message(self, text, force=False):
        text = (text or "").strip()
        if not text:
            return False

        now = time.monotonic()
        if not force and (now - self._last_audio_emit_ts) < self._audio_min_gap_sec:
            return False
        if (
            text == self.last_audio_message
            and (now - self._last_audio_emit_ts) < self._audio_repeat_suppression_sec
        ):
            return False

        self.audio_feedback_signal.emit(text)
        self.last_audio_message = text
        self._last_audio_emit_ts = now
        return True

    def _emit_important_audio(self, feedback_texts, metrics=None):
        metrics = metrics or {}
        reps = int(
            metrics.get(
                "rep_count",
                self.exercise_analyzer.rep_count if self.exercise_analyzer else 0,
            )
        )
        sets_done = int(
            metrics.get(
                "set_count",
                self.exercise_analyzer.set_count if self.exercise_analyzer else 0,
            )
        )
        cfg = exercise_config.get(self.exercise_choice, {})
        reps_per_set = int(cfg.get("reps_per_set", 12))

        warning_detected = any("Warning" in text for text in feedback_texts)
        ready_detected = "Ready to start!" in feedback_texts
        set_complete_detected = "Set complete!" in feedback_texts
        up_range_miss = bool(metrics.get("up_range_miss", False))
        down_range_miss = bool(metrics.get("down_range_miss", False))

        if warning_detected:
            if not self._audio_warning_active and self._emit_audio_message(
                "Form warning. Reset posture and keep your torso stable."
            ):
                self._audio_warning_active = True
            return

        self._audio_warning_active = False

        if ready_detected and not self._audio_ready_announced:
            if self._emit_audio_message("Start position matched. Begin your reps."):
                self._audio_ready_announced = True
            return

        if set_complete_detected:
            if sets_done != self._audio_last_set_announced and self._emit_audio_message(
                f"Set {sets_done} complete. Great work.",
                force=True,
            ):
                self._audio_last_set_announced = sets_done
                self._audio_ready_announced = False
                self._audio_last_rep_marker = (sets_done, -1)
            return

        range_message, range_direction = range_miss_message(
            exercise_choice=self.exercise_choice,
            up_range_miss=up_range_miss,
            down_range_miss=down_range_miss,
        )
        range_marker = (
            (sets_done, reps, range_direction)
            if range_message and range_direction
            else None
        )

        if range_marker is not None:
            if range_marker != self._audio_last_range_marker and self._emit_audio_message(
                range_message
            ):
                self._audio_last_range_marker = range_marker
            return

        self._audio_last_range_marker = None

        if reps_per_set <= 0 or reps <= 0:
            return

        halfway_rep = max(1, reps_per_set // 2)
        final_push_rep = max(1, reps_per_set - 2)
        milestone_messages = {
            1: "First rep locked in. Keep your tempo.",
            halfway_rep: "Halfway there. Stay strict and controlled.",
            final_push_rep: "Final reps. Drive through with clean form.",
        }
        milestone_message = milestone_messages.get(reps)
        marker = (sets_done, reps)
        if (
            milestone_message
            and marker != self._audio_last_rep_marker
            and self._emit_audio_message(milestone_message)
        ):
            self._audio_last_rep_marker = marker

    def _rep_stagnation_seconds(self, reps, sets_done, stable_start_detected):
        now = time.monotonic()
        progress_key = (sets_done, reps)
        if progress_key != self._rep_progress_key:
            self._rep_progress_key = progress_key
            self._rep_progress_ts = now
            return 0.0
        if not stable_start_detected or reps <= 0:
            self._rep_progress_ts = now
            return 0.0
        return max(0.0, now - self._rep_progress_ts)

    def _resolve_user_name(self):
        if self.current_user_info and self.current_user_info.get("username"):
            return self.current_user_info.get("username")
        return self.display_user_name

    def _today_exercise_set_offset(self, user_info):
        if not isinstance(user_info, dict):
            return 0
        user_id = user_info.get("user_id")
        if not user_id:
            return 0
        try:
            progress = self.db_handler.get_daily_exercise_progress(
                user_id=user_id,
                exercise=self.exercise_choice,
                date_str=datetime.now().strftime("%Y-%m-%d"),
            )
            return max(0, int(progress.get("set_count") or 0))
        except Exception as exc:
            logging.error(
                "Unable to resolve daily exercise progress for user_id=%s exercise=%s: %s",
                user_id,
                self.exercise_choice,
                exc,
            )
            return 0

    def _motivation_line(
        self,
        reps,
        sets_done,
        reps_per_set,
        feedback_texts,
        status_hint="",
        rep_state=0,
        stable_start_detected=False,
        up_range_miss=False,
        down_range_miss=False,
        stagnant_seconds=0.0,
    ):
        return motivation_line(
            exercise_choice=self.exercise_choice,
            reps=reps,
            sets_done=sets_done,
            reps_per_set=reps_per_set,
            feedback_texts=feedback_texts,
            status_hint=status_hint,
            rep_state=rep_state,
            stable_start_detected=stable_start_detected,
            up_range_miss=up_range_miss,
            down_range_miss=down_range_miss,
            stagnant_seconds=stagnant_seconds,
        )

    def _format_angle_line(self, angles):
        if not angles:
            return "Angles: --"

        angle_items = list(angles.items())
        primary = []
        for key, value in angle_items[:2]:
            label = key.replace("_", " ").title()
            primary.append(f"{label}: {int(value)}deg")

        return "   |   ".join(primary)

    def _build_overlay_data(self, metrics=None, feedback_texts=None, status_hint=""):
        metrics = metrics or {}
        feedback_texts = feedback_texts or []

        angles = metrics.get("angles") or {}
        reps = int(
            metrics.get(
                "rep_count",
                self.exercise_analyzer.rep_count if self.exercise_analyzer else 0,
            )
        )
        sets_done = int(
            metrics.get(
                "set_count",
                self.exercise_analyzer.set_count if self.exercise_analyzer else 0,
            )
        )
        current_weight = int(
            metrics.get(
                "current_weight",
                self.exercise_analyzer.current_weight if self.exercise_analyzer else 0,
            )
        )

        cfg = exercise_config.get(self.exercise_choice, {})
        reps_per_set = int(cfg.get("reps_per_set", 12))
        rep_angle_raw = metrics.get("rep_angle")
        rep_angle = float(rep_angle_raw) if rep_angle_raw is not None else None

        up_range_raw = metrics.get("up_range")
        down_range_raw = metrics.get("down_range")
        up_range = (
            (float(up_range_raw[0]), float(up_range_raw[1]))
            if isinstance(up_range_raw, (list, tuple)) and len(up_range_raw) == 2
            else None
        )
        down_range = (
            (float(down_range_raw[0]), float(down_range_raw[1]))
            if isinstance(down_range_raw, (list, tuple)) and len(down_range_raw) == 2
            else None
        )
        rep_state = int(metrics.get("rep_state", self.exercise_analyzer.rep_state if self.exercise_analyzer else 0))
        stable_start_detected = bool(
            metrics.get(
                "stable_start_detected",
                self.exercise_analyzer.stable_start_detected if self.exercise_analyzer else False,
            )
        )
        up_range_miss = bool(metrics.get("up_range_miss", False))
        down_range_miss = bool(metrics.get("down_range_miss", False))
        stagnant_seconds = self._rep_stagnation_seconds(
            reps,
            sets_done,
            stable_start_detected,
        )

        return {
            "username": self._resolve_user_name(),
            "exercise": self.exercise_choice.replace("_", " ").title(),
            "rep_count": reps,
            "set_count": sets_done,
            "current_weight": current_weight,
            "reps_per_set": reps_per_set,
            "angles_line": self._format_angle_line(angles),
            "timestamp": self._overlay_timestamp(),
            "motivation": self._motivation_line(
                reps,
                sets_done,
                reps_per_set,
                feedback_texts,
                status_hint,
                rep_state=rep_state,
                stable_start_detected=stable_start_detected,
                up_range_miss=up_range_miss,
                down_range_miss=down_range_miss,
                stagnant_seconds=stagnant_seconds,
            ),
            "rep_angle": rep_angle,
            "up_range": up_range,
            "down_range": down_range,
            "rep_state": rep_state,
            "up_range_miss": up_range_miss,
            "down_range_miss": down_range_miss,
            "up_range_label": "BOTTOM" if self.exercise_choice == "deadlift" else "UP",
            "down_range_label": "LOCKOUT" if self.exercise_choice == "deadlift" else "DOWN",
            "show_range_bars": (
                self.exercise_choice in {"bicep_curl", "deadlift"}
                and stable_start_detected
                and rep_angle is not None
                and up_range is not None
                and down_range is not None
            ),
        }

    def _clear_pose_overlay_cache(self):
        self._cached_pose_points = None
        self._smoothed_pose_points = None
        self._cached_pose_ts = 0.0

    def _extract_pose_overlay_points(self, pose_landmarks):
        landmarks = getattr(pose_landmarks, "landmark", None)
        if not landmarks:
            return None
        points = []
        for landmark in landmarks:
            points.append(
                (
                    min(1.0, max(0.0, float(getattr(landmark, "x", 0.0)))),
                    min(1.0, max(0.0, float(getattr(landmark, "y", 0.0)))),
                    min(1.0, max(0.0, float(getattr(landmark, "visibility", 1.0)))),
                )
            )
        return points or None

    def _smooth_pose_overlay_points(self, current_points):
        if not current_points:
            return None

        previous_points = self._smoothed_pose_points
        if not previous_points or len(previous_points) != len(current_points):
            self._smoothed_pose_points = [tuple(point) for point in current_points]
            return self._smoothed_pose_points

        min_alpha = min(self.pose_overlay_smoothing_alpha, self.pose_overlay_smoothing_max_alpha)
        max_alpha = max(self.pose_overlay_smoothing_alpha, self.pose_overlay_smoothing_max_alpha)
        smoothed_points = []
        for (curr_x, curr_y, curr_visibility), (
            prev_x,
            prev_y,
            prev_visibility,
        ) in zip(current_points, previous_points):
            motion = hypot(curr_x - prev_x, curr_y - prev_y)
            alpha = min(max_alpha, max(min_alpha, min_alpha + (motion * self.pose_overlay_motion_boost)))
            if curr_visibility < self.pose_overlay_visibility_floor:
                alpha = min(alpha, 0.20)

            smoothed_points.append(
                (
                    prev_x + ((curr_x - prev_x) * alpha),
                    prev_y + ((curr_y - prev_y) * alpha),
                    prev_visibility + ((curr_visibility - prev_visibility) * min(0.75, alpha + 0.15)),
                )
            )

        self._smoothed_pose_points = smoothed_points
        return smoothed_points

    @staticmethod
    def _lighten_bgr(color, amount=0.25):
        amount = min(1.0, max(0.0, float(amount)))
        return tuple(
            int(round(channel + ((255 - channel) * amount)))
            for channel in color
        )

    @staticmethod
    def _darken_bgr(color, amount=0.30):
        amount = min(1.0, max(0.0, float(amount)))
        return tuple(
            int(round(channel * (1.0 - amount)))
            for channel in color
        )

    def _pose_point_to_pixels(self, pose_points, point_idx, frame_w, frame_h, visibility_floor):
        if not pose_points or point_idx >= len(pose_points):
            return None
        x, y, visibility = pose_points[point_idx]
        if visibility < visibility_floor:
            return None
        return (
            int(round(min(1.0, max(0.0, x)) * float(frame_w))),
            int(round(min(1.0, max(0.0, y)) * float(frame_h))),
        )

    def _draw_styled_pose_overlay(self, frame, pose_points, connection_color, landmark_color):
        if frame is None or getattr(frame, "size", 0) == 0 or not pose_points:
            return False

        frame_h, frame_w = frame.shape[:2]
        if frame_h <= 0 or frame_w <= 0:
            return False

        visibility_floor = self.pose_overlay_visibility_floor
        base_thickness = max(4, int(round(min(frame_h, frame_w) / 220.0)))
        outline_thickness = base_thickness + 4
        inner_thickness = max(2, base_thickness - 1)
        glow_thickness = outline_thickness + max(4, base_thickness * 2)
        joint_radius = max(5, base_thickness + 1)
        joint_outline_radius = joint_radius + 2
        joint_inner_radius = max(2, int(round(joint_radius * 0.52)))
        joint_glow_radius = joint_radius + max(5, base_thickness * 2)

        glow_layer = np.zeros_like(frame)
        line_glow_color = self._lighten_bgr(connection_color, 0.18)
        joint_glow_color = self._lighten_bgr(landmark_color, 0.24)
        line_outline_color = self._darken_bgr(connection_color, 0.58)
        joint_outline_color = self._darken_bgr(landmark_color, 0.56)
        line_inner_color = self._lighten_bgr(connection_color, 0.34)
        joint_inner_color = self._lighten_bgr(landmark_color, 0.42)

        for start_idx, end_idx in POSE_RENDER_CONNECTIONS:
            start_point = self._pose_point_to_pixels(
                pose_points,
                start_idx,
                frame_w,
                frame_h,
                visibility_floor,
            )
            end_point = self._pose_point_to_pixels(
                pose_points,
                end_idx,
                frame_w,
                frame_h,
                visibility_floor,
            )
            if start_point is None or end_point is None:
                continue

            cv2.line(
                glow_layer,
                start_point,
                end_point,
                line_glow_color,
                glow_thickness,
                cv2.LINE_AA,
            )

        for joint_idx in POSE_RENDER_JOINTS:
            point = self._pose_point_to_pixels(
                pose_points,
                joint_idx,
                frame_w,
                frame_h,
                visibility_floor,
            )
            if point is None:
                continue
            cv2.circle(
                glow_layer,
                point,
                joint_glow_radius,
                joint_glow_color,
                -1,
                cv2.LINE_AA,
            )

        if self.pose_overlay_glow_strength > 0.0:
            cv2.addWeighted(
                frame,
                1.0,
                glow_layer,
                self.pose_overlay_glow_strength,
                0.0,
                dst=frame,
            )

        for start_idx, end_idx in POSE_RENDER_CONNECTIONS:
            start_point = self._pose_point_to_pixels(
                pose_points,
                start_idx,
                frame_w,
                frame_h,
                visibility_floor,
            )
            end_point = self._pose_point_to_pixels(
                pose_points,
                end_idx,
                frame_w,
                frame_h,
                visibility_floor,
            )
            if start_point is None or end_point is None:
                continue

            cv2.line(
                frame,
                start_point,
                end_point,
                line_outline_color,
                outline_thickness,
                cv2.LINE_AA,
            )
            cv2.line(
                frame,
                start_point,
                end_point,
                connection_color,
                base_thickness,
                cv2.LINE_AA,
            )
            cv2.line(
                frame,
                start_point,
                end_point,
                line_inner_color,
                inner_thickness,
                cv2.LINE_AA,
            )

        for joint_idx in POSE_RENDER_JOINTS:
            point = self._pose_point_to_pixels(
                pose_points,
                joint_idx,
                frame_w,
                frame_h,
                visibility_floor,
            )
            if point is None:
                continue

            cv2.circle(
                frame,
                point,
                joint_outline_radius,
                joint_outline_color,
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                point,
                joint_radius,
                landmark_color,
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                point,
                joint_inner_radius,
                joint_inner_color,
                -1,
                cv2.LINE_AA,
            )

        return True

    def _cache_pose_overlay(self, pose_landmarks, connection_color, landmark_color, now):
        if pose_landmarks is None:
            return
        pose_points = self._extract_pose_overlay_points(pose_landmarks)
        if not pose_points:
            return
        self._cached_pose_points = self._smooth_pose_overlay_points(pose_points)
        self._cached_pose_colors = (connection_color, landmark_color)
        self._cached_pose_ts = now

    def _draw_cached_pose_overlay_if_recent(self, frame, now):
        if self._cached_pose_points is None:
            return False
        if (now - self._cached_pose_ts) > self._pose_overlay_hold_sec:
            self._clear_pose_overlay_cache()
            return False
        connection_color, landmark_color = self._cached_pose_colors
        return self._draw_styled_pose_overlay(
            frame,
            self._cached_pose_points,
            connection_color,
            landmark_color,
        )

    def _cache_face_boxes(self, face_locations, display_face_names, now):
        cached = []
        upscale = 1.0 / self._face_location_scale()
        for ((top, right, bottom, left), name) in zip(face_locations, display_face_names):
            cached.append(
                (
                    int(left * upscale),
                    int(top * upscale),
                    int(right * upscale),
                    int(bottom * upscale),
                    str(name),
                )
            )
        if cached:
            self._cached_face_boxes = cached
            self._cached_face_boxes_ts = now

    def _draw_cached_face_boxes_if_recent(self, frame, now):
        if not self._cached_face_boxes:
            return False
        if (now - self._cached_face_boxes_ts) > self._face_box_hold_sec:
            self._cached_face_boxes = []
            self._cached_face_boxes_ts = 0.0
            return False
        for left, top, right, bottom, name in self._cached_face_boxes:
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (110, 255, 180),
                2,
            )
            cv2.rectangle(
                frame,
                (left, bottom - 28),
                (right, bottom),
                (18, 32, 54),
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                name,
                (left + 8, bottom - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (224, 250, 248),
                2,
                cv2.LINE_AA,
            )
        return True

    def _overlay_timestamp(self):
        now = datetime.now()
        second = int(now.timestamp())
        if second != self._overlay_timestamp_second:
            self._overlay_timestamp_second = second
            self._overlay_timestamp_cache = now.strftime("%d %b %Y | %I:%M:%S %p")
        return self._overlay_timestamp_cache

    def _should_run_analysis(self, now):
        if self._analysis_interval_sec <= 0:
            return True
        return (now - self._last_analysis_ts) >= self._analysis_interval_sec

    def _visual_output_due(self, now):
        return (
            self._ui_interval_sec <= 0
            or self._thumbnail_interval_sec <= 0
            or (now - self._last_ui_emit_ts) >= self._ui_interval_sec
            or (now - self._last_thumbnail_emit_ts) >= self._thumbnail_interval_sec
        )

    def _emit_visual_outputs(self, frame, now, force=False, overlay_data=None, feedback_texts=None):
        emit_frame = (
            force
            or self._ui_interval_sec <= 0
            or (now - self._last_ui_emit_ts) >= self._ui_interval_sec
        )
        emit_thumb = (
            force
            or self._thumbnail_interval_sec <= 0
            or (now - self._last_thumbnail_emit_ts) >= self._thumbnail_interval_sec
        )
        if emit_frame:
            self.frame_signal.emit(frame)
            self.display_state_signal.emit(
                {
                    "overlay_data": dict(overlay_data or {}),
                    "feedback_texts": list(feedback_texts or []),
                }
            )
            self._last_ui_emit_ts = now
        if emit_thumb:
            thumb = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            self.thumbnail_frame_signal.emit(thumb)
            self._last_thumbnail_emit_ts = now
        return emit_frame or emit_thumb

    def _emit_counters_if_due(self, reps, sets_done, now, force=False):
        counters = (int(reps), int(sets_done))
        if not force:
            if counters == self._last_counter_value and (
                now - self._last_counter_emit_ts
            ) < self._counter_interval_sec:
                return
            if (
                now - self._last_counter_emit_ts
            ) < self._counter_interval_sec and counters != self._last_counter_value:
                return
        self.counters_signal.emit(*counters)
        self._last_counter_value = counters
        self._last_counter_emit_ts = now

    def _sleep_to_target_loop(self, loop_started_at):
        if self._loop_interval_sec <= 0:
            return
        elapsed = time.monotonic() - loop_started_at
        remaining = self._loop_interval_sec - elapsed
        if remaining <= 0:
            return
        self.msleep(max(1, int(remaining * 1000)))

    def run(self):
        cap = None
        mp_pose_solution = None
        try:
            cap = open_camera_capture(self.camera_index)
            if cap is None or not cap.isOpened():
                self.status_signal.emit(f"Error: Could not open camera {self.camera_index}.")
                logging.error(f"Could not open camera {self.camera_index}.")
                return
            logging.info(f"Camera {self.camera_index} initialized.")

            mp_pose_solution = mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1,
            )
            aruco_detector = None
            if self.disable_aruco:
                logging.info(
                    "ExerciseWorker cam_%s running with ArUco disabled (SMART_MIRROR_DISABLE_ARUCO=1).",
                    self.camera_index,
                )
            else:
                aruco_detector = ArucoDetector(dict_type=self.aruco_dict_type)
                logging.info(
                    "ExerciseWorker cam_%s using ArUco dictionary: %s",
                    self.camera_index,
                    aruco_detector.get_dict_type(),
                )

            if self.exercise_choice not in exercise_config:
                self.status_signal.emit(
                    f"Exercise '{self.exercise_choice}' not configured. Using default if available."
                )
                logging.warning(f"Exercise '{self.exercise_choice}' not found in config.")
                if exercise_config:
                    self.exercise_choice = list(exercise_config.keys())[0]
                else:
                    self.status_signal.emit("No exercises configured.")
                    logging.error("No exercises configured. Worker terminating.")
                    return

            if self.face_recognizer is None:
                self.status_signal.emit("Face recognizer is not initialized.")
                logging.error("Face recognizer is not initialized. Worker terminating.")
                return

            analysis_overlay_data = None
            analysis_feedback_texts = []
            registration_overlay_data = None
            recognition_overlay_data = None
            self._clear_pose_overlay_cache()
            self._cached_face_boxes = []
            self._cached_face_boxes_ts = 0.0

            while not self.stop_requested:
                loop_started_at = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    self.status_signal.emit("Failed to capture frame.")
                    logging.error("Failed to capture frame from camera.")
                    break

                frame = transform_camera_frame(frame, camera_source=self.camera_index)
                now = time.monotonic()
                analysis_due = self._should_run_analysis(now)
                if analysis_due:
                    self._last_analysis_ts = now

                # If exercise analysis is active, skip face recognition
                if self.exercise_analysis_active and not self.registration_state.active:
                    try:
                        frame_bgr = frame
                        if analysis_due:
                            analysis_elapsed_seconds = self._mode_elapsed_seconds(
                                "_analysis_mode_last_timing_ts",
                                now,
                            )
                            analysis_roi_box = self._resolve_analysis_roi_box(frame_bgr.shape)
                            analysis_roi_frame = self._crop_frame_to_roi(frame_bgr, analysis_roi_box)
                            if analysis_roi_frame is None:
                                analysis_roi_box = self._full_frame_box(frame_bgr.shape)
                                analysis_roi_frame = frame_bgr

                            frame_rgb = cv2.cvtColor(analysis_roi_frame, cv2.COLOR_BGR2RGB)
                            results = mp_pose_solution.process(frame_rgb)
                            if results.pose_landmarks:
                                self._remap_pose_landmarks_to_frame(
                                    results.pose_landmarks,
                                    analysis_roi_box,
                                    frame_bgr.shape,
                                )

                            self._target_face_recheck_elapsed_sec += analysis_elapsed_seconds
                            reliable_pose = bool(
                                results.pose_landmarks
                                and self._has_reliable_pose(results.pose_landmarks.landmark)
                            )
                            pose_matches_target = False
                            if reliable_pose:
                                pose_matches_target = self._accept_pose_for_target_lock(
                                    results.pose_landmarks.landmark,
                                    frame_bgr.shape,
                                    frame=frame_bgr,
                                )
                                if not pose_matches_target:
                                    reliable_pose = False
                                    logging.debug(
                                        "Rejected pose candidate outside tracked-user lock on cam_%s.",
                                        self.camera_index,
                                    )

                            face_recheck_result = None
                            force_face_recheck = not reliable_pose and self._pose_lock_established
                            if self._should_run_target_face_recheck(
                                now,
                                force=force_face_recheck,
                            ):
                                face_recheck_result = self._run_locked_face_recheck(
                                    frame_bgr,
                                    now,
                                    roi_box=analysis_roi_box,
                                )

                            face_recheck_status = (face_recheck_result or {}).get("status")
                            face_confirmed = face_recheck_status == "confirmed"
                            other_person_detected = face_recheck_status == "other"
                            start_position_locked = bool(
                                self.exercise_analyzer
                                and self.exercise_analyzer.stable_start_detected
                            )
                            grace_active = self._analysis_pose_grace_remaining_sec > 0.0
                            tracked_name = (
                                self.locked_username
                                or self.display_user_name
                                or self.assigned_user_name
                            )

                            if reliable_pose and pose_matches_target and not other_person_detected:
                                self.exercise_analyzer.no_pose_frames = 0
                                self._analysis_pose_grace_remaining_sec = 0.0
                                self._last_reliable_pose_ts = now
                                self._mark_target_visible()
                                landmarks = results.pose_landmarks.landmark
                                end_session_requested, end_session_hint = self._update_end_session_palm_hold(
                                    landmarks,
                                    now,
                                )
                                if end_session_requested:
                                    self._finalize_tracked_user_exit(
                                        now,
                                        exit_reason=f"{tracked_name} ended the session",
                                    )
                                    analysis_overlay_data = None
                                    analysis_feedback_texts = []
                                    recognition_overlay_data = None
                                    self._target_face_recheck_elapsed_sec = 0.0
                                    self._sleep_to_target_loop(loop_started_at)
                                    continue

                                feedback_texts, overlay_metrics = self.exercise_analyzer.analyze_exercise_form(
                                    landmarks,
                                    frame_bgr,
                                )
                                if end_session_hint:
                                    feedback_texts = [end_session_hint] + list(feedback_texts)

                                if not self.exercise_analyzer.stable_start_detected:
                                    connection_color = (255, 140, 90)
                                    landmark_color = (255, 140, 90)
                                elif self.exercise_analyzer.bend_warning_displayed:
                                    connection_color = (70, 70, 255)
                                    landmark_color = (70, 70, 255)
                                else:
                                    connection_color = (118, 242, 130)
                                    landmark_color = (118, 242, 130)

                                self._cache_pose_overlay(
                                    results.pose_landmarks,
                                    connection_color,
                                    landmark_color,
                                    now,
                                )

                                if self._visual_output_due(now):
                                    self._draw_cached_pose_overlay_if_recent(frame_bgr, now)
                                    self._draw_tracked_user_lock_marker(frame_bgr, now, paused=False)

                                self._emit_important_audio(feedback_texts, overlay_metrics)
                                analysis_overlay_data = self._build_overlay_data(
                                    overlay_metrics,
                                    feedback_texts,
                                )
                                analysis_feedback_texts = list(feedback_texts)

                                if self._visual_output_due(now):
                                    self._emit_visual_outputs(
                                        frame_bgr,
                                        now,
                                        overlay_data=analysis_overlay_data,
                                        feedback_texts=feedback_texts,
                                    )

                                self._emit_counters_if_due(
                                    self.exercise_analyzer.rep_count,
                                    self.exercise_analyzer.set_count,
                                    now,
                                )
                            else:
                                self.exercise_analyzer.invalidate_tracking()
                                self._reset_end_session_palm_hold()
                                self.exercise_analyzer.update_current_weight(frame_bgr)
                                if face_confirmed and not other_person_detected:
                                    self.exercise_analyzer.no_pose_frames = 0
                                    self._mark_target_visible()
                                    if grace_active:
                                        self._analysis_pose_grace_remaining_sec = max(
                                            0.0,
                                            self._analysis_pose_grace_remaining_sec - analysis_elapsed_seconds,
                                        )
                                        status_hint = "Face recognized. Align upper body to start tracking."
                                    elif not start_position_locked:
                                        status_hint = "Move to start position (green lines) to begin tracking."
                                    else:
                                        status_hint = "Face confirmed. Hold steady to resume tracking."
                                elif other_person_detected:
                                    if start_position_locked:
                                        self.exercise_analyzer.no_pose_frames += 1
                                    else:
                                        self.exercise_analyzer.no_pose_frames = 0
                                    self._mark_target_missing(now, frame_bgr.shape)
                                    status_hint = self._tracked_user_missing_status_hint(
                                        tracked_name,
                                        now,
                                    )
                                elif grace_active:
                                    self._analysis_pose_grace_remaining_sec = max(
                                        0.0,
                                        self._analysis_pose_grace_remaining_sec - analysis_elapsed_seconds,
                                    )
                                    self.exercise_analyzer.no_pose_frames = 0
                                    status_hint = "Face recognized. Align upper body to start tracking."
                                elif not start_position_locked:
                                    if (
                                        self._last_reliable_pose_ts > 0.0
                                        and (now - self._last_reliable_pose_ts)
                                        < self.exit_reacquire_grace_seconds
                                    ):
                                        self.exercise_analyzer.no_pose_frames = 0
                                        status_hint = "Move to start position (green lines) to begin tracking."
                                    else:
                                        self.exercise_analyzer.no_pose_frames = 0
                                        self._mark_target_missing(now, frame_bgr.shape)
                                        status_hint = self._tracked_user_missing_status_hint(
                                            tracked_name,
                                            now,
                                        )
                                else:
                                    if self._last_reliable_pose_ts <= 0.0:
                                        self._last_reliable_pose_ts = now
                                    seconds_since_reliable = now - self._last_reliable_pose_ts

                                    if (
                                        seconds_since_reliable < self.exit_reacquire_grace_seconds
                                        and not other_person_detected
                                    ):
                                        self.exercise_analyzer.no_pose_frames = 0
                                        status_hint = f"Tracking lost briefly. Waiting for {tracked_name}."
                                    else:
                                        self.exercise_analyzer.no_pose_frames += 1
                                        self._mark_target_missing(now, frame_bgr.shape)
                                        if results.pose_landmarks:
                                            logging.debug(
                                                "Ignoring unstable pose landmarks during active session "
                                                "(no_pose_frames=%s).",
                                                self.exercise_analyzer.no_pose_frames,
                                            )
                                            status_hint = (
                                                self._tracked_user_missing_status_hint(
                                                    tracked_name,
                                                    now,
                                                )
                                            )
                                        else:
                                            logging.debug(
                                                "No pose detected while holding tracked-user lock. "
                                                "no_pose_frames: %s",
                                                self.exercise_analyzer.no_pose_frames,
                                            )
                                            status_hint = (
                                                self._tracked_user_missing_status_hint(
                                                    tracked_name,
                                                    now,
                                                )
                                            )

                                analysis_overlay_data = self._build_overlay_data(
                                    metrics={
                                        "rep_count": self.exercise_analyzer.rep_count,
                                        "set_count": self.exercise_analyzer.set_count,
                                        "current_weight": self.exercise_analyzer.current_weight,
                                    },
                                    feedback_texts=[],
                                    status_hint=status_hint,
                                )
                                analysis_feedback_texts = []
                                if self._visual_output_due(now):
                                    self._draw_cached_pose_overlay_if_recent(frame_bgr, now)
                                    self._draw_tracked_user_lock_marker(frame_bgr, now, paused=True)
                                    self._emit_visual_outputs(
                                        frame_bgr,
                                        now,
                                        overlay_data=analysis_overlay_data,
                                        feedback_texts=[],
                                    )
                                self._emit_counters_if_due(
                                    self.exercise_analyzer.rep_count,
                                    self.exercise_analyzer.set_count,
                                    now,
                                )

                                if self._should_return_to_face_recognition(now, frame_bgr.shape):
                                    self._finalize_tracked_user_exit(
                                        now,
                                        exit_reason="Tracked athlete unavailable",
                                    )
                                    analysis_overlay_data = None
                                    analysis_feedback_texts = []
                                    recognition_overlay_data = None
                                    self._target_face_recheck_elapsed_sec = 0.0
                        elif self._visual_output_due(now):
                            overlay_data = analysis_overlay_data or self._build_overlay_data(
                                metrics={
                                    "rep_count": self.exercise_analyzer.rep_count,
                                    "set_count": self.exercise_analyzer.set_count,
                                    "current_weight": self.exercise_analyzer.current_weight,
                                },
                                feedback_texts=[],
                                status_hint="Tracking movement...",
                            )
                            self._draw_cached_pose_overlay_if_recent(frame_bgr, now)
                            self._draw_tracked_user_lock_marker(
                                frame_bgr,
                                now,
                                paused=self.target_lost_since is not None,
                            )
                            self._emit_visual_outputs(
                                frame_bgr,
                                now,
                                overlay_data=overlay_data,
                                feedback_texts=analysis_feedback_texts,
                            )

                        if self.exercise_analyzer:
                            self._emit_counters_if_due(
                                self.exercise_analyzer.rep_count,
                                self.exercise_analyzer.set_count,
                                now,
                            )
                        self._sleep_to_target_loop(loop_started_at)
                        continue

                    except Exception as e:
                        if self.exercise_analyzer:
                            self.exercise_analyzer.invalidate_tracking()
                        self.status_signal.emit("Exercise analysis error.")
                        logging.error(f"Exercise analysis error: {e}")
                        self._sleep_to_target_loop(loop_started_at)
                        continue

                # If capturing new user face data
                if self.registration_state.active:
                    try:
                        display_frame = frame.copy()
                        if analysis_due:
                            capture_elapsed_seconds = self._mode_elapsed_seconds(
                                "_registration_mode_last_timing_ts",
                                now,
                            )
                            small_frame = cv2.resize(display_frame, (0, 0), fx=0.25, fy=0.25)
                            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                            face_locations = fr_lib.face_locations(
                                rgb_small_frame,
                                number_of_times_to_upsample=0,
                                model="hog",
                            )
                            encodings = fr_lib.face_encodings(rgb_small_frame, face_locations)
                            capture_feedback = "Hold still and look straight at the camera."
                            if len(face_locations) == 1 and len(encodings) == 1:
                                top, right, bottom, left = face_locations[0]
                                face_area = max(0, bottom - top) * max(0, right - left)
                                if face_area >= self.MIN_REGISTRATION_FACE_AREA:
                                    self.registration_state.add_valid_encoding(
                                        encodings[0],
                                        capture_elapsed_seconds,
                                    )
                                else:
                                    capture_feedback = "Move closer so your face appears larger."
                            elif len(face_locations) > 1:
                                capture_feedback = "Multiple faces detected. Keep only one face in frame."
                            else:
                                capture_feedback = "No face detected. Center your face in frame."

                            progress = len(self.registration_state.encodings)
                            target_seconds = self.registration_state.target_capture_seconds
                            captured_seconds = self.registration_state.valid_capture_seconds
                            required_samples = self.registration_state.minimum_samples
                            registration_user_name = (
                                self.registration_state.user_name
                                or self.display_user_name
                                or self.assigned_user_name
                            )
                            hint = (
                                f"Registering {registration_user_name}: "
                                f"{captured_seconds:.1f}/{target_seconds:.1f}s, "
                                f"{progress}/{required_samples} samples | {capture_feedback}"
                            )
                            registration_overlay_data = self._build_overlay_data(
                                metrics={"rep_count": 0, "set_count": 0, "current_weight": 0},
                                feedback_texts=[],
                                status_hint=hint,
                            )

                            if self.registration_state.is_complete:
                                self.status_signal.emit(f"Registering user {registration_user_name}...")
                                logging.info(f"Registering new user: {registration_user_name}")
                                registered_name = registration_user_name
                                registration_success = self.face_recognizer.register_new_user(
                                    registered_name,
                                    self.registration_state.encodings,
                                )
                                if registration_success:
                                    self.status_signal.emit(
                                        f"User {registered_name} registered successfully."
                                    )
                                    logging.info(
                                        f"User {registered_name} registered successfully."
                                    )
                                    self.display_user_name = registered_name
                                else:
                                    self.status_signal.emit(
                                        f"Failed to register user {registered_name}."
                                    )
                                    logging.error(
                                        f"Failed to register user {registered_name}."
                                    )
                                self.new_user_registration_signal.emit(
                                    registered_name,
                                    registration_success,
                                )

                                self.registration_state.reset()
                                self._registration_mode_last_timing_ts = 0.0
                                self.face_recognition_active = True
                                registration_overlay_data = None
                                self._reset_face_recognition_state()

                        if self._visual_output_due(now):
                            overlay_data = registration_overlay_data or self._build_overlay_data(
                                metrics={"rep_count": 0, "set_count": 0, "current_weight": 0},
                                feedback_texts=[],
                                status_hint="Capturing face data...",
                            )
                            self._emit_visual_outputs(
                                display_frame,
                                now,
                                overlay_data=overlay_data,
                                feedback_texts=[],
                            )

                        self._sleep_to_target_loop(loop_started_at)
                        continue
                    except Exception as e:
                        self.status_signal.emit("New user registration error.")
                        logging.error(f"New user registration error: {e}")
                        self._sleep_to_target_loop(loop_started_at)
                        continue

                # Face Recognition Mode
                if self.face_recognition_active:
                    try:
                        display_frame = frame.copy()
                        if analysis_due:
                            recognition_elapsed_seconds = self._mode_elapsed_seconds(
                                "_face_mode_last_timing_ts",
                                now,
                            )
                            _, face_locations, face_names = self.face_recognizer.recognize_faces(frame)
                            status_hint = "Stand in front of the camera for recognition."
                            primary_face_index = self._select_primary_face_index(face_locations)
                            primary_raw_label = (
                                face_names[primary_face_index]
                                if primary_face_index is not None and primary_face_index < len(face_names)
                                else None
                            )
                            stable_primary_label = self._update_stable_primary_label(
                                primary_raw_label,
                                recognition_elapsed_seconds,
                            )
                            display_face_names = list(face_names)
                            if (
                                primary_face_index is not None
                                and stable_primary_label is not None
                                and stable_primary_label == primary_raw_label
                                and primary_face_index < len(display_face_names)
                            ):
                                display_face_names[primary_face_index] = stable_primary_label

                            self._cache_face_boxes(face_locations, display_face_names, now)
                            self._draw_cached_face_boxes_if_recent(display_frame, now)

                            if primary_face_index is not None and primary_face_index < len(display_face_names):
                                effective_label = primary_raw_label
                                if not effective_label or effective_label == "Unknown":
                                    status_hint = "Unknown user detected. Hold still for registration."
                                    should_prompt_unknown = self.recognition_state.observe_unknown(
                                        recognition_elapsed_seconds,
                                    )
                                    if should_prompt_unknown:
                                        self.status_signal.emit(
                                            "Stable unknown user detected. Prompting for user name..."
                                        )
                                        logging.info("Stable unknown user detected.")
                                        self.unknown_user_detected.emit()
                                elif effective_label != stable_primary_label:
                                    # Label smoothing is for presentation. Starting a
                                    # session also requires current evidence for the
                                    # identity whose face will seed the body lock.
                                    self.recognition_state.clear_known_progress()
                                    status_hint = "Verifying the person in front of the camera."
                                else:
                                    recognized_user = effective_label
                                    self.display_user_name = recognized_user
                                    status_hint = "User recognized. Preparing real-time feedback."
                                    recognition_confirmed = self.recognition_state.observe_recognized(
                                        recognized_user,
                                        recognition_elapsed_seconds,
                                    )

                                    if recognition_confirmed:
                                        logging.info(f"User recognized: {recognized_user}")
                                        user_info = self.db_handler.get_member_info(recognized_user)
                                        if not user_info:
                                            self.status_signal.emit(
                                                f"Face recognized as {recognized_user}, but no active member record was found."
                                            )
                                            logging.warning(
                                                "Recognized face %s has no active member record. "
                                                "Skipping analysis to avoid auto-recreating deleted members.",
                                                recognized_user,
                                            )
                                            status_hint = (
                                                f"{recognized_user} is recognized, but the member record is missing. "
                                                "Restore or re-register this member."
                                            )
                                            self.display_user_name = recognized_user
                                            self.recognition_state.clear_known_progress()
                                        else:
                                            self.user_recognized_signal.emit(user_info)
                                            self.status_signal.emit(
                                                f"Face recognized as {recognized_user}, starting exercise analysis."
                                            )
                                            logging.info(
                                                f"Starting exercise analysis for user: {recognized_user}"
                                            )
                                            self.current_user_info = user_info
                                            self.display_user_name = recognized_user
                                            if primary_face_index is not None and primary_face_index < len(face_locations):
                                                self._seed_tracked_user_lock(
                                                    recognized_user,
                                                    face_locations[primary_face_index],
                                                    frame.shape,
                                                    now,
                                                    frame=frame,
                                                )
                                            resume_set_offset = self._today_exercise_set_offset(user_info)
                                            self.exercise_analyzer = ExerciseAnalyzer(
                                                self.exercise_choice,
                                                aruco_detector,
                                                self.db_handler,
                                                user_id=user_info.get("user_id"),
                                                username_snapshot=recognized_user,
                                                starting_set_count=resume_set_offset,
                                            )
                                            self._clear_pose_overlay_cache()
                                            if resume_set_offset > 0:
                                                self.status_signal.emit(
                                                    "Welcome back. Continuing "
                                                    f"{self.exercise_choice.replace('_', ' ')} from set {resume_set_offset + 1}."
                                                )
                                            self._exit_save_retry_count = 0
                                            self._analysis_pose_grace_remaining_sec = self.pose_acquire_grace_seconds
                                            self._last_reliable_pose_ts = now
                                            self._analysis_mode_last_timing_ts = 0.0
                                            self.exercise_analysis_active = True
                                            self.face_recognition_active = False
                                            analysis_overlay_data = None
                                            analysis_feedback_texts = []
                                            self._reset_face_recognition_state()
                            else:
                                self.recognition_state.observe_no_face(recognition_elapsed_seconds)

                            recognition_overlay_data = self._build_overlay_data(
                                metrics={"rep_count": 0, "set_count": 0, "current_weight": 0},
                                feedback_texts=[],
                                status_hint=status_hint,
                            )

                        if self._visual_output_due(now):
                            self._draw_cached_face_boxes_if_recent(display_frame, now)
                            overlay_data = recognition_overlay_data or self._build_overlay_data(
                                metrics={"rep_count": 0, "set_count": 0, "current_weight": 0},
                                feedback_texts=[],
                                status_hint="Stand in front of the camera for recognition.",
                            )
                            self._emit_visual_outputs(
                                display_frame,
                                now,
                                overlay_data=overlay_data,
                                feedback_texts=[],
                            )
                    except Exception as e:
                        self.status_signal.emit("Face recognition error.")
                        logging.error(f"Face recognition error: {e}")

                self._sleep_to_target_loop(loop_started_at)
        finally:
            if cap is not None:
                cap.release()
            if mp_pose_solution is not None:
                mp_pose_solution.close()
            self.cleanup()

    def cleanup(self):
        """Handle cleanup operations when stopping the worker."""
        try:
            if self.exercise_analyzer:
                record = self.exercise_analyzer.update_data()
                if record:
                    logging.info("Exercise data saved locally during cleanup.")
                else:
                    logging.warning("No exercise data to save during cleanup.")

            self.data_updated.emit()
        except Exception as e:
            self.status_signal.emit("Error during cleanup.")
            logging.error(f"Error during cleanup: {e}")
        # No signal disconnect calls here
