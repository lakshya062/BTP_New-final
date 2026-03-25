# ui/worker.py

import cv2
import face_recognition as fr_lib
import mediapipe as mp
import logging
import os
import time
import uuid
from datetime import datetime
from math import hypot

from PySide6.QtCore import QThread, Signal

from core.aruco_detection import ArucoDetector
from core.pose_analysis import ExerciseAnalyzer
from core.config import exercise_config
from ui.coaching import motivation_line, range_miss_message
from ui.edge_utils import open_camera_capture
from ui.live_overlay import draw_live_overlay

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class ExerciseWorker(QThread):
    frame_signal = Signal(object)
    thumbnail_frame_signal = Signal(object)
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
        return max(minimum, value)

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

        self.unknown_frames = 0
        self.UNKNOWN_FRAME_THRESHOLD = 15
        self.KNOWN_FRAME_THRESHOLD = 30
        self.KNOWN_FRAME_DECAY = 1
        self.UNKNOWN_FRAME_DECAY = 1
        self.KNOWN_FRAME_SWITCH_PENALTY = 12

        self.new_user_name = None
        self.new_user_encodings = []
        self.capturing_new_user_data = False
        self.frames_to_capture = 120
        self.MIN_REGISTRATION_FACE_AREA = 36 * 36

        self.face_recognition_active = True
        self.exercise_analysis_active = False
        self.exercise_analyzer = None
        self.current_user_info = None
        self.exit_save_retry_limit = max(
            0,
            int(os.getenv("SMART_MIRROR_EXIT_SAVE_RETRY_LIMIT", "2")),
        )
        self._exit_save_retry_count = 0

        self.known_frames = 0
        self.last_recognized_user = None
        self.unknown_stable_detected = False

        self.STABLE_LABEL_BUILDUP = 3
        self.STABLE_LABEL_DECAY = 1
        self.STABLE_LABEL_MIN_SCORE = 6
        self.STABLE_LABEL_SWITCH_MARGIN = 4
        self.STABLE_LABEL_FORGET_FRAMES = 20
        self._label_scores = {}
        self._stable_primary_label = None
        self._frames_without_primary_face = 0

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
        self.is_edge_mode = os.environ.get("SMART_MIRROR_EDGE_MODE", "0") == "1"
        default_disable_aruco = "1" if self.is_edge_mode else "0"
        self.disable_aruco = os.environ.get("SMART_MIRROR_DISABLE_ARUCO", default_disable_aruco) == "1"
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
        self.pose_acquire_grace_frames = self._read_int_env(
            "SMART_MIRROR_POSE_ACQUIRE_GRACE_FRAMES",
            default=14,
            minimum=0,
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
        self._analysis_pose_grace_frames_remaining = 0
        self._last_reliable_pose_ts = 0.0
        self.loop_fps = self._read_float_env(
            "SMART_MIRROR_WORKER_LOOP_FPS",
            default=0.0,
            minimum=0.0,
        )
        self.analysis_fps = self._read_float_env(
            "SMART_MIRROR_WORKER_ANALYSIS_FPS",
            default=20.0,
            minimum=1.0,
        )
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
        self._cached_pose_landmarks = None
        self._cached_pose_colors = ((118, 242, 130), (118, 242, 130))
        self._cached_pose_ts = 0.0
        self._cached_face_boxes = []
        self._cached_face_boxes_ts = 0.0

    def request_stop(self):
        self.stop_requested = True
        logging.info("Stop requested for ExerciseWorker.")

    def start_record_new_user(self, user_name):
        self.new_user_name = user_name
        self.new_user_encodings = []
        self.capturing_new_user_data = True
        self._reset_face_recognition_state()
        self.status_signal.emit(f"Capturing face data for {user_name}. Please wait...")
        logging.info(f"Started capturing face data for new user: {user_name}")

    def _reset_face_recognition_state(self):
        self.unknown_frames = 0
        self.known_frames = 0
        self.last_recognized_user = None
        self.unknown_stable_detected = False
        self._label_scores = {}
        self._stable_primary_label = None
        self._frames_without_primary_face = 0

    def _switch_to_face_recognition_mode(self, reason="Returning to Face Recognition Mode."):
        self.exercise_analyzer = None
        self.exercise_analysis_active = False
        self.face_recognition_active = True
        self.current_user_info = None
        self.display_user_name = self.assigned_user_name
        self._clear_pose_overlay_cache()
        self._exit_save_retry_count = 0
        self._analysis_pose_grace_frames_remaining = 0
        self._last_reliable_pose_ts = 0.0
        self._reset_face_recognition_state()
        self.status_signal.emit(reason)
        logging.info(reason)

    def _select_primary_face_index(self, face_locations):
        if not face_locations:
            return None
        return max(
            range(len(face_locations)),
            key=lambda idx: max(0, face_locations[idx][2] - face_locations[idx][0])
            * max(0, face_locations[idx][1] - face_locations[idx][3]),
        )

    def _update_stable_primary_label(self, observed_label):
        if observed_label is None:
            self._frames_without_primary_face += 1
            for label in list(self._label_scores.keys()):
                updated_score = self._label_scores[label] - self.STABLE_LABEL_DECAY
                if updated_score <= 0:
                    del self._label_scores[label]
                else:
                    self._label_scores[label] = updated_score
            if self._frames_without_primary_face >= self.STABLE_LABEL_FORGET_FRAMES:
                self._stable_primary_label = None
            return self._stable_primary_label

        self._frames_without_primary_face = 0
        for label in list(self._label_scores.keys()):
            if label == observed_label:
                self._label_scores[label] = min(
                    100,
                    self._label_scores[label] + self.STABLE_LABEL_BUILDUP,
                )
            else:
                updated_score = self._label_scores[label] - self.STABLE_LABEL_DECAY
                if updated_score <= 0:
                    del self._label_scores[label]
                else:
                    self._label_scores[label] = updated_score

        if observed_label not in self._label_scores:
            self._label_scores[observed_label] = self.STABLE_LABEL_BUILDUP

        top_label, top_score = max(self._label_scores.items(), key=lambda item: item[1])
        if self._stable_primary_label is None:
            if top_score >= self.STABLE_LABEL_MIN_SCORE:
                self._stable_primary_label = top_label
        elif top_label != self._stable_primary_label:
            stable_score = self._label_scores.get(self._stable_primary_label, 0)
            if (
                top_score >= self.STABLE_LABEL_MIN_SCORE
                and top_score >= stable_score + self.STABLE_LABEL_SWITCH_MARGIN
            ):
                self._stable_primary_label = top_label

        if self._stable_primary_label is not None:
            return self._stable_primary_label
        return top_label

    def _has_reliable_pose(self, landmarks):
        if not landmarks:
            return False
        try:
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

            left_shoulder_visible = getattr(left_shoulder, "visibility", 0.0) >= self.pose_visibility_threshold
            right_shoulder_visible = getattr(right_shoulder, "visibility", 0.0) >= self.pose_visibility_threshold
            if not (left_shoulder_visible and right_shoulder_visible):
                return False

            arm_visibility_threshold = max(0.2, self.pose_visibility_threshold - 0.15)
            left_elbow_visible = getattr(left_elbow, "visibility", 0.0) >= arm_visibility_threshold
            right_elbow_visible = getattr(right_elbow, "visibility", 0.0) >= arm_visibility_threshold
            if self.exercise_choice == "deadlift":
                leg_visibility_threshold = max(0.2, self.pose_visibility_threshold - 0.15)
                left_knee_visible = getattr(left_knee, "visibility", 0.0) >= leg_visibility_threshold
                right_knee_visible = getattr(right_knee, "visibility", 0.0) >= leg_visibility_threshold
                if not (left_knee_visible or right_knee_visible):
                    return False
            elif not (left_elbow_visible or right_elbow_visible):
                return False

            shoulder_width = hypot(
                float(left_shoulder.x) - float(right_shoulder.x),
                float(left_shoulder.y) - float(right_shoulder.y),
            )
            if shoulder_width < self.min_pose_shoulder_width:
                return False

            left_hip_visible = getattr(left_hip, "visibility", 0.0) >= self.pose_visibility_threshold
            right_hip_visible = getattr(right_hip, "visibility", 0.0) >= self.pose_visibility_threshold
            if left_hip_visible and right_hip_visible:
                shoulder_mid_y = (float(left_shoulder.y) + float(right_shoulder.y)) / 2.0
                hip_mid_y = (float(left_hip.y) + float(right_hip.y)) / 2.0
                torso_height = abs(hip_mid_y - shoulder_mid_y)
                if torso_height < self.min_pose_torso_height:
                    return False

            return True
        except Exception:
            return False

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

    def _draw_live_overlay(self, frame, overlay_data, feedback_texts=None):
        draw_live_overlay(frame, overlay_data, feedback_texts)

    def _clear_pose_overlay_cache(self):
        self._cached_pose_landmarks = None
        self._cached_pose_ts = 0.0

    def _cache_pose_overlay(self, pose_landmarks, connection_color, landmark_color, now):
        if pose_landmarks is None:
            return
        self._cached_pose_landmarks = pose_landmarks
        self._cached_pose_colors = (connection_color, landmark_color)
        self._cached_pose_ts = now

    def _draw_cached_pose_overlay_if_recent(self, frame, now):
        if self._cached_pose_landmarks is None:
            return False
        if (now - self._cached_pose_ts) > self._pose_overlay_hold_sec:
            self._clear_pose_overlay_cache()
            return False
        connection_color, landmark_color = self._cached_pose_colors
        mp_drawing.draw_landmarks(
            frame,
            self._cached_pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(
                color=connection_color,
                thickness=2,
                circle_radius=2,
            ),
            mp_drawing.DrawingSpec(
                color=landmark_color,
                thickness=2,
                circle_radius=2,
            ),
        )
        return True

    def _cache_face_boxes(self, face_locations, display_face_names, now):
        cached = []
        for ((top, right, bottom, left), name) in zip(face_locations, display_face_names):
            cached.append(
                (
                    int(left * 4),
                    int(top * 4),
                    int(right * 4),
                    int(bottom * 4),
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

    def _emit_visual_outputs(self, frame, now, force=False):
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

            occlusion_timeout = self.exit_no_pose_frames
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

                now = time.monotonic()
                analysis_due = self._should_run_analysis(now)
                if analysis_due:
                    self._last_analysis_ts = now

                # If exercise analysis is active, skip face recognition
                if self.exercise_analysis_active and not self.capturing_new_user_data:
                    try:
                        frame_bgr = frame
                        if analysis_due:
                            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                            results = mp_pose_solution.process(frame_rgb)

                            reliable_pose = bool(
                                results.pose_landmarks
                                and self._has_reliable_pose(results.pose_landmarks.landmark)
                            )
                            if reliable_pose:
                                self.exercise_analyzer.no_pose_frames = 0
                                self._analysis_pose_grace_frames_remaining = 0
                                self._last_reliable_pose_ts = now
                                landmarks = results.pose_landmarks.landmark
                                feedback_texts, overlay_metrics = self.exercise_analyzer.analyze_exercise_form(
                                    landmarks,
                                    frame_bgr,
                                )

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

                                self._emit_important_audio(feedback_texts, overlay_metrics)
                                analysis_overlay_data = self._build_overlay_data(
                                    overlay_metrics,
                                    feedback_texts,
                                )
                                analysis_feedback_texts = list(feedback_texts)

                                if self._visual_output_due(now):
                                    self._draw_live_overlay(
                                        frame_bgr,
                                        analysis_overlay_data,
                                        feedback_texts,
                                    )
                                    self._emit_visual_outputs(frame_bgr, now)

                                self._emit_counters_if_due(
                                    self.exercise_analyzer.rep_count,
                                    self.exercise_analyzer.set_count,
                                    now,
                                )
                            else:
                                self.exercise_analyzer.update_current_weight(frame_bgr)
                                grace_active = self._analysis_pose_grace_frames_remaining > 0
                                start_position_locked = bool(
                                    self.exercise_analyzer
                                    and self.exercise_analyzer.stable_start_detected
                                )
                                if grace_active:
                                    self._analysis_pose_grace_frames_remaining -= 1
                                    status_hint = "Face recognized. Align upper body to start tracking."
                                elif not start_position_locked:
                                    # Do not run person-exit logic before start position is locked (green landmarks).
                                    self.exercise_analyzer.no_pose_frames = 0
                                    status_hint = "Move to start position (green lines) to begin tracking."
                                else:
                                    if self._last_reliable_pose_ts <= 0.0:
                                        self._last_reliable_pose_ts = now
                                    seconds_since_reliable = now - self._last_reliable_pose_ts

                                    if seconds_since_reliable < self.exit_reacquire_grace_seconds:
                                        self.exercise_analyzer.no_pose_frames = 0
                                        status_hint = "Tracking lost briefly. Step back in to continue."
                                    else:
                                        self.exercise_analyzer.no_pose_frames += 1
                                        if results.pose_landmarks:
                                            logging.debug(
                                                "Ignoring unstable pose landmarks during active session (no_pose_frames=%s).",
                                                self.exercise_analyzer.no_pose_frames,
                                            )
                                            status_hint = "Body landmarks unstable. Hold still in frame."
                                        else:
                                            logging.debug(
                                                "No pose detected. no_pose_frames: %s",
                                                self.exercise_analyzer.no_pose_frames,
                                            )
                                            status_hint = "Body not detected. Step back into frame."

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
                                    self._draw_live_overlay(frame_bgr, analysis_overlay_data, [])
                                    self._emit_visual_outputs(frame_bgr, now)
                                self._emit_counters_if_due(
                                    self.exercise_analyzer.rep_count,
                                    self.exercise_analyzer.set_count,
                                    now,
                                )

                                if (
                                    not grace_active
                                    and start_position_locked
                                    and self.exercise_analyzer.no_pose_frames >= occlusion_timeout
                                ):
                                    self.status_signal.emit("Person exited the frame, updating database...")
                                    logging.info("Person exited frame. Updating database.")
                                    had_activity = bool(
                                        self.exercise_analyzer.rep_data
                                        or self.exercise_analyzer.rep_count > 0
                                        or self.exercise_analyzer.sets_reps
                                    )
                                    record = self.exercise_analyzer.update_data()
                                    if record:
                                        self._exit_save_retry_count = 0
                                        self.status_signal.emit("Exercise data saved locally.")
                                        logging.info("Data saved locally.")
                                        self.data_updated.emit()
                                        next_mode_reason = "Person exited frame. Ready for next user."
                                    elif had_activity:
                                        self._exit_save_retry_count = 0
                                        logging.error(
                                            "Exercise session had activity but save returned no record. "
                                            "Resetting session state for next user."
                                        )
                                        self.status_signal.emit(
                                            "Failed to save exercise data. Resetting for next user."
                                        )
                                        self.exercise_analyzer.reset_counters()
                                        next_mode_reason = "Person exited frame. Reset complete. Waiting for recognition."
                                    else:
                                        self._exit_save_retry_count = 0
                                        self.status_signal.emit("No exercise data to save.")
                                        next_mode_reason = "Person exited frame. Waiting for recognition."

                                    self._switch_to_face_recognition_mode(reason=next_mode_reason)
                                    analysis_overlay_data = None
                                    analysis_feedback_texts = []
                                    recognition_overlay_data = None
                                    self._emit_counters_if_due(0, 0, now, force=True)
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
                            self._draw_live_overlay(frame_bgr, overlay_data, analysis_feedback_texts)
                            self._emit_visual_outputs(frame_bgr, now)

                        if self.exercise_analyzer:
                            self._emit_counters_if_due(
                                self.exercise_analyzer.rep_count,
                                self.exercise_analyzer.set_count,
                                now,
                            )
                        self._sleep_to_target_loop(loop_started_at)
                        continue

                    except Exception as e:
                        self.status_signal.emit("Exercise analysis error.")
                        logging.error(f"Exercise analysis error: {e}")
                        self._sleep_to_target_loop(loop_started_at)
                        continue

                # If capturing new user face data
                if self.capturing_new_user_data:
                    try:
                        display_frame = frame.copy()
                        if analysis_due:
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
                                    self.new_user_encodings.append(encodings[0])
                                else:
                                    capture_feedback = "Move closer so your face appears larger."
                            elif len(face_locations) > 1:
                                capture_feedback = "Multiple faces detected. Keep only one face in frame."
                            else:
                                capture_feedback = "No face detected. Center your face in frame."

                            progress = len(self.new_user_encodings)
                            hint = (
                                f"Registering {self.new_user_name}: "
                                f"{progress}/{self.frames_to_capture} | {capture_feedback}"
                            )
                            registration_overlay_data = self._build_overlay_data(
                                metrics={"rep_count": 0, "set_count": 0, "current_weight": 0},
                                feedback_texts=[],
                                status_hint=hint,
                            )

                            if len(self.new_user_encodings) >= self.frames_to_capture:
                                self.status_signal.emit(f"Registering user {self.new_user_name}...")
                                logging.info(f"Registering new user: {self.new_user_name}")
                                registered_name = self.new_user_name
                                registration_success = self.face_recognizer.register_new_user(
                                    registered_name,
                                    self.new_user_encodings,
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

                                self.new_user_name = None
                                self.new_user_encodings = []
                                self.capturing_new_user_data = False
                                self.face_recognition_active = True
                                registration_overlay_data = None
                                self._reset_face_recognition_state()

                        if self._visual_output_due(now):
                            overlay_data = registration_overlay_data or self._build_overlay_data(
                                metrics={"rep_count": 0, "set_count": 0, "current_weight": 0},
                                feedback_texts=[],
                                status_hint="Capturing face data...",
                            )
                            self._draw_live_overlay(display_frame, overlay_data, [])
                            self._emit_visual_outputs(display_frame, now)

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
                            _, face_locations, face_names = self.face_recognizer.recognize_faces(frame)
                            status_hint = "Stand in front of the camera for recognition."
                            primary_face_index = self._select_primary_face_index(face_locations)
                            primary_raw_label = (
                                face_names[primary_face_index]
                                if primary_face_index is not None and primary_face_index < len(face_names)
                                else None
                            )
                            stable_primary_label = self._update_stable_primary_label(primary_raw_label)
                            display_face_names = list(face_names)
                            if (
                                primary_face_index is not None
                                and stable_primary_label is not None
                                and primary_face_index < len(display_face_names)
                            ):
                                display_face_names[primary_face_index] = stable_primary_label

                            self._cache_face_boxes(face_locations, display_face_names, now)
                            self._draw_cached_face_boxes_if_recent(display_frame, now)

                            if primary_face_index is not None and primary_face_index < len(display_face_names):
                                effective_label = display_face_names[primary_face_index]
                                if effective_label == "Unknown":
                                    self.unknown_frames = min(
                                        self.UNKNOWN_FRAME_THRESHOLD,
                                        self.unknown_frames + 1,
                                    )
                                    self.known_frames = max(0, self.known_frames - self.KNOWN_FRAME_DECAY)
                                    if self.known_frames == 0:
                                        self.last_recognized_user = None
                                    status_hint = "Unknown user detected. Hold still for registration."
                                    if (
                                        self.unknown_frames >= self.UNKNOWN_FRAME_THRESHOLD
                                        and not self.unknown_stable_detected
                                    ):
                                        self.status_signal.emit(
                                            "Stable unknown user detected. Prompting for user name..."
                                        )
                                        logging.info("Stable unknown user detected.")
                                        self.unknown_user_detected.emit()
                                        self.unknown_stable_detected = True
                                else:
                                    recognized_user = effective_label
                                    self.unknown_frames = max(
                                        0,
                                        self.unknown_frames - self.UNKNOWN_FRAME_DECAY,
                                    )
                                    if self.unknown_frames == 0:
                                        self.unknown_stable_detected = False
                                    self.display_user_name = recognized_user
                                    status_hint = "User recognized. Preparing real-time feedback."
                                    if self.last_recognized_user is None:
                                        self.last_recognized_user = recognized_user
                                        self.known_frames = max(1, self.known_frames)
                                    elif recognized_user == self.last_recognized_user:
                                        self.known_frames = min(
                                            self.KNOWN_FRAME_THRESHOLD,
                                            self.known_frames + 1,
                                        )
                                    else:
                                        self.last_recognized_user = recognized_user
                                        self.known_frames = max(
                                            1,
                                            self.known_frames - self.KNOWN_FRAME_SWITCH_PENALTY,
                                        )

                                    if self.known_frames >= self.KNOWN_FRAME_THRESHOLD:
                                        logging.info(f"User recognized: {recognized_user}")
                                        user_info = self.db_handler.get_member_info(recognized_user)
                                        if not user_info:
                                            # Edge nodes can have face embeddings before member rows are synced.
                                            # Auto-create a lightweight local member so analysis can continue.
                                            provisional_member = {
                                                "user_id": str(uuid.uuid4()),
                                                "username": recognized_user,
                                                "email": "NA",
                                                "membership": "NA",
                                                "joined_on": datetime.utcnow().strftime("%Y-%m-%d"),
                                            }
                                            try:
                                                inserted = self.db_handler.insert_member_local(provisional_member)
                                                user_info = self.db_handler.get_member_info(recognized_user)
                                                logging.info(
                                                    "Auto-provisioned local member for %s (inserted=%s).",
                                                    recognized_user,
                                                    inserted,
                                                )
                                            except Exception as exc:
                                                logging.error(
                                                    "Failed to auto-provision member %s: %s",
                                                    recognized_user,
                                                    exc,
                                                )
                                                user_info = None

                                            if not user_info:
                                                user_info = provisional_member

                                        self.user_recognized_signal.emit(user_info)
                                        self.status_signal.emit(
                                            f"Face recognized as {recognized_user}, starting exercise analysis."
                                        )
                                        logging.info(
                                            f"Starting exercise analysis for user: {recognized_user}"
                                        )
                                        self.current_user_info = user_info
                                        self.display_user_name = recognized_user
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
                                        self._analysis_pose_grace_frames_remaining = self.pose_acquire_grace_frames
                                        self._last_reliable_pose_ts = now
                                        self.exercise_analysis_active = True
                                        self.face_recognition_active = False
                                        analysis_overlay_data = None
                                        analysis_feedback_texts = []
                                        self._reset_face_recognition_state()
                            else:
                                self.unknown_frames = max(0, self.unknown_frames - self.UNKNOWN_FRAME_DECAY)
                                self.known_frames = max(0, self.known_frames - self.KNOWN_FRAME_DECAY)
                                if self.known_frames == 0:
                                    self.last_recognized_user = None
                                if self.unknown_frames == 0:
                                    self.unknown_stable_detected = False

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
                            self._draw_live_overlay(display_frame, overlay_data, [])
                            self._emit_visual_outputs(display_frame, now)
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
