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

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def draw_translucent_panel(
    frame,
    x,
    y,
    width,
    height,
    fill_color=(20, 28, 38),
    border_color=(103, 188, 255),
    alpha=0.62,
    border_thickness=2,
):
    """Draw a translucent rectangular panel with a border."""
    x = max(0, int(x))
    y = max(0, int(y))
    width = max(1, int(width))
    height = max(1, int(height))

    x2 = min(frame.shape[1] - 1, x + width)
    y2 = min(frame.shape[0] - 1, y + height)
    if x >= x2 or y >= y2:
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), fill_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x2, y2), border_color, border_thickness)


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
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

            left_shoulder_visible = getattr(left_shoulder, "visibility", 0.0) >= self.pose_visibility_threshold
            right_shoulder_visible = getattr(right_shoulder, "visibility", 0.0) >= self.pose_visibility_threshold
            if not (left_shoulder_visible and right_shoulder_visible):
                return False

            arm_visibility_threshold = max(0.2, self.pose_visibility_threshold - 0.15)
            left_elbow_visible = getattr(left_elbow, "visibility", 0.0) >= arm_visibility_threshold
            right_elbow_visible = getattr(right_elbow, "visibility", 0.0) >= arm_visibility_threshold
            if not (left_elbow_visible or right_elbow_visible):
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

        range_marker = None
        range_message = None
        if up_range_miss:
            range_marker = (sets_done, reps, "up")
            range_message = "Go higher. Reach the top range before lowering."
        elif down_range_miss:
            range_marker = (sets_done, reps, "down")
            range_message = "Lower fully. Reach the down range before next rep."

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
        if any("Warning" in text for text in feedback_texts):
            return "Reset your posture first, then we go again with control."
        if up_range_miss:
            return "You are close. Lift a little higher and squeeze at the top."
        if down_range_miss:
            return "Nice effort. Lower all the way down, then start the next curl."
        if "Set complete!" in feedback_texts:
            return "Great set. Take two deep breaths and get ready for the next round."
        if "Ready to start!" in feedback_texts:
            return "Setup looks strong. Start smooth and own the first rep."

        if stable_start_detected and reps > 0:
            if stagnant_seconds >= 14:
                lines = [
                    "You are not done yet. One clean rep breaks this plateau.",
                    "Stay with me. Elbows tight, breathe out, and finish this rep.",
                    "Tough moment right now. Slow down, reset, and drive through.",
                ]
                return lines[int(stagnant_seconds // 4) % len(lines)]
            if stagnant_seconds >= 8:
                lines = [
                    "You are stuck on this rep. Small reset, then go again.",
                    "Keep calm and stay tight. You can get this next rep.",
                    "One solid rep right now. Controlled up, controlled down.",
                ]
                return lines[int(stagnant_seconds // 3) % len(lines)]

        if stable_start_detected and sets_done == 0 and reps == 0:
            return "Start position locked. Curl up with intent and return with control."
        if sets_done > 0 and reps == 0:
            return "Strong previous set. Stay composed and attack this next one."
        if reps_per_set > 0 and reps >= max(1, int(reps_per_set * 0.9)):
            return "Last push. Finish every inch of these final reps."
        if reps_per_set > 0 and reps >= max(1, int(reps_per_set * 0.75)):
            return "You are in the hard zone now. Keep your form crisp."
        if reps_per_set > 0 and reps >= max(1, reps_per_set // 2):
            return "Halfway there. Keep breathing and keep moving clean."
        if reps_per_set > 0 and reps >= max(1, int(reps_per_set * 0.25)):
            return "Rhythm looks good. Stack clean reps one by one."
        if reps > 0:
            if rep_state == 1:
                return "Drive up through the curl. Elbows stay pinned."
            if rep_state == 2:
                return "Control the way down. Earn the full range."
            return "Good tempo. Stay smooth and keep tension."
        if status_hint:
            return status_hint
        return "Stand centered and start when you are ready."

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
            "timestamp": datetime.now().strftime("%d %b %Y | %I:%M:%S %p"),
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
            "show_range_bars": (
                self.exercise_choice == "bicep_curl"
                and stable_start_detected
                and rep_angle is not None
                and up_range is not None
                and down_range is not None
            ),
        }

    def _draw_live_overlay(self, frame, overlay_data, feedback_texts=None):
        feedback_texts = feedback_texts or []
        h, w = frame.shape[:2]
        scale = max(0.72, min(w / 1280.0, h / 720.0))

        margin = int(20 * scale)
        panel_w = int(470 * scale)
        panel_h = int(250 * scale)

        panel_x = margin
        panel_y = margin

        draw_translucent_panel(
            frame,
            panel_x,
            panel_y,
            panel_w,
            panel_h,
            fill_color=(22, 30, 43),
            border_color=(108, 201, 255),
            alpha=0.66,
            border_thickness=max(1, int(2 * scale)),
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 0.88 * scale
        label_scale = 0.53 * scale
        value_scale = 0.82 * scale

        cv2.putText(
            frame,
            "LIVE PERFORMANCE",
            (panel_x + int(18 * scale), panel_y + int(34 * scale)),
            font,
            title_scale,
            (240, 248, 255),
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"User: {overlay_data['username']}",
            (panel_x + int(18 * scale), panel_y + int(64 * scale)),
            font,
            0.64 * scale,
            (186, 227, 255),
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Exercise: {overlay_data['exercise']}",
            (panel_x + int(18 * scale), panel_y + int(88 * scale)),
            font,
            label_scale,
            (212, 223, 240),
            max(1, int(1 * scale)),
            cv2.LINE_AA,
        )

        chip_w = int((panel_w - int(52 * scale)) / 3)
        chip_h = int(90 * scale)
        chip_y = panel_y + int(102 * scale)
        chip_gap = int(14 * scale)

        stats = [
            ("REPS", str(overlay_data["rep_count"])),
            ("SETS", str(overlay_data["set_count"])),
            (
                "WEIGHT",
                f"{overlay_data['current_weight']} lbs" if overlay_data["current_weight"] > 0 else "--",
            ),
        ]

        for idx, (label, value) in enumerate(stats):
            chip_x = panel_x + int(16 * scale) + idx * (chip_w + chip_gap)
            draw_translucent_panel(
                frame,
                chip_x,
                chip_y,
                chip_w,
                chip_h,
                fill_color=(30, 41, 57),
                border_color=(92, 166, 236),
                alpha=0.7,
                border_thickness=max(1, int(1 * scale)),
            )
            cv2.putText(
                frame,
                label,
                (chip_x + int(12 * scale), chip_y + int(30 * scale)),
                font,
                0.48 * scale,
                (176, 210, 246),
                max(1, int(1 * scale)),
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                value,
                (chip_x + int(12 * scale), chip_y + int(68 * scale)),
                font,
                value_scale,
                (245, 250, 255),
                max(1, int(2 * scale)),
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            overlay_data["angles_line"],
            (panel_x + int(18 * scale), panel_y + panel_h - int(20 * scale)),
            font,
            0.52 * scale,
            (178, 197, 218),
            max(1, int(1 * scale)),
            cv2.LINE_AA,
        )

        time_panel_w = int(390 * scale)
        time_panel_h = int(62 * scale)
        time_x = max(margin, w - time_panel_w - margin)
        time_y = margin
        draw_translucent_panel(
            frame,
            time_x,
            time_y,
            time_panel_w,
            time_panel_h,
            fill_color=(22, 30, 43),
            border_color=(108, 201, 255),
            alpha=0.64,
            border_thickness=max(1, int(2 * scale)),
        )
        cv2.putText(
            frame,
            overlay_data["timestamp"],
            (time_x + int(16 * scale), time_y + int(39 * scale)),
            font,
            0.60 * scale,
            (238, 246, 255),
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )

        if overlay_data.get("show_range_bars"):
            rep_angle = float(overlay_data.get("rep_angle") or 0.0)
            up_range = overlay_data.get("up_range") or (0.0, 0.0)
            down_range = overlay_data.get("down_range") or (0.0, 0.0)
            up_min, up_max = float(up_range[0]), float(up_range[1])
            down_min, down_max = float(down_range[0]), float(down_range[1])
            rep_state = int(overlay_data.get("rep_state", 0))
            blink_on = ((datetime.now().microsecond // 180000) % 2) == 0

            range_panel_w = int(390 * scale)
            range_panel_h = int(154 * scale)
            range_x = max(margin, w - range_panel_w - margin)
            range_y = time_y + time_panel_h + int(12 * scale)
            draw_translucent_panel(
                frame,
                range_x,
                range_y,
                range_panel_w,
                range_panel_h,
                fill_color=(22, 30, 43),
                border_color=(108, 201, 255),
                alpha=0.64,
                border_thickness=max(1, int(2 * scale)),
            )
            cv2.putText(
                frame,
                "RANGE CHECK",
                (range_x + int(16 * scale), range_y + int(28 * scale)),
                font,
                0.58 * scale,
                (232, 244, 255),
                max(1, int(2 * scale)),
                cv2.LINE_AA,
            )

            bar_left = range_x + int(16 * scale)
            bar_w = range_panel_w - int(32 * scale)
            bar_h = int(18 * scale)
            up_bar_y = range_y + int(48 * scale)
            down_bar_y = range_y + int(100 * scale)

            angle_span = max(1.0, down_max - up_min)
            up_progress = max(0.0, min(1.0, (down_max - rep_angle) / angle_span))
            down_progress = max(0.0, min(1.0, (rep_angle - up_min) / angle_span))
            up_hit = up_min <= rep_angle <= up_max
            down_hit = down_min <= rep_angle <= down_max
            up_miss = bool(overlay_data.get("up_range_miss", False))
            down_miss = bool(overlay_data.get("down_range_miss", False))

            def draw_range_bar(label, y, progress, target_min, target_max, in_target, expected_now, miss_now):
                draw_translucent_panel(
                    frame,
                    bar_left,
                    y,
                    bar_w,
                    bar_h,
                    fill_color=(26, 38, 54),
                    border_color=(76, 132, 188),
                    alpha=0.74,
                    border_thickness=max(1, int(1 * scale)),
                )

                fill_w = max(1, int(bar_w * progress))
                if in_target:
                    fill_color = (64, 188, 120)
                    border_color = (118, 242, 130)
                    state_text = "ON TARGET"
                elif miss_now and blink_on:
                    fill_color = (52, 46, 188)
                    border_color = (92, 92, 255)
                    state_text = "MISSED RANGE"
                elif expected_now:
                    fill_color = (82, 132, 214)
                    border_color = (116, 168, 248)
                    state_text = "IN PROGRESS"
                else:
                    fill_color = (66, 108, 164)
                    border_color = (100, 144, 206)
                    state_text = "READY"

                cv2.rectangle(
                    frame,
                    (bar_left, y),
                    (bar_left + fill_w, y + bar_h),
                    fill_color,
                    -1,
                )
                cv2.rectangle(
                    frame,
                    (bar_left, y),
                    (bar_left + bar_w, y + bar_h),
                    border_color,
                    max(1, int(2 * scale)),
                )

                cv2.putText(
                    frame,
                    f"{label} {int(target_min)}-{int(target_max)}deg",
                    (bar_left, y - int(7 * scale)),
                    font,
                    0.46 * scale,
                    (196, 220, 246),
                    max(1, int(1 * scale)),
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    state_text,
                    (bar_left + int(6 * scale), y + bar_h + int(16 * scale)),
                    font,
                    0.44 * scale,
                    (214, 232, 250),
                    max(1, int(1 * scale)),
                    cv2.LINE_AA,
                )

            draw_range_bar(
                "UP",
                up_bar_y,
                up_progress,
                up_min,
                up_max,
                up_hit,
                rep_state == 1,
                up_miss,
            )
            draw_range_bar(
                "DOWN",
                down_bar_y,
                down_progress,
                down_min,
                down_max,
                down_hit,
                rep_state == 2,
                down_miss,
            )

        motivation_h = int(72 * scale)
        motivation_x = margin
        motivation_y = h - motivation_h - margin
        motivation_w = max(100, w - (2 * margin))

        draw_translucent_panel(
            frame,
            motivation_x,
            motivation_y,
            motivation_w,
            motivation_h,
            fill_color=(17, 77, 88),
            border_color=(88, 223, 230),
            alpha=0.68,
            border_thickness=max(1, int(2 * scale)),
        )
        cv2.putText(
            frame,
            overlay_data["motivation"],
            (motivation_x + int(18 * scale), motivation_y + int(45 * scale)),
            font,
            0.68 * scale,
            (235, 255, 255),
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )

        warning_texts = [text for text in feedback_texts if "Warning" in text]
        if warning_texts:
            warning_text = warning_texts[0]
            warning_w = min(int(640 * scale), w - 2 * margin)
            warning_h = int(58 * scale)
            warning_x = max(margin, (w - warning_w) // 2)
            warning_y = margin + int(80 * scale)
            draw_translucent_panel(
                frame,
                warning_x,
                warning_y,
                warning_w,
                warning_h,
                fill_color=(61, 25, 32),
                border_color=(255, 124, 124),
                alpha=0.74,
                border_thickness=max(1, int(2 * scale)),
            )
            cv2.putText(
                frame,
                warning_text,
                (warning_x + int(16 * scale), warning_y + int(36 * scale)),
                font,
                0.62 * scale,
                (255, 230, 230),
                max(1, int(2 * scale)),
                cv2.LINE_AA,
            )

    def run(self):
        cap = None
        mp_pose_solution = None
        try:
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                self.status_signal.emit(f"Error: Could not open camera {self.camera_index}.")
                logging.error(f"Could not open camera {self.camera_index}.")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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

            while not self.stop_requested:
                ret, frame = cap.read()
                if not ret:
                    self.status_signal.emit("Failed to capture frame.")
                    logging.error("Failed to capture frame from camera.")
                    break

                display_frame = frame.copy()

                # If exercise analysis is active, skip face recognition
                if self.exercise_analysis_active and not self.capturing_new_user_data:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = mp_pose_solution.process(frame_rgb)
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                        reliable_pose = bool(
                            results.pose_landmarks
                            and self._has_reliable_pose(results.pose_landmarks.landmark)
                        )
                        if reliable_pose:
                            self.exercise_analyzer.no_pose_frames = 0
                            self._analysis_pose_grace_frames_remaining = 0
                            self._last_reliable_pose_ts = time.monotonic()
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

                            mp_drawing.draw_landmarks(
                                frame_bgr,
                                results.pose_landmarks,
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

                            self._emit_important_audio(feedback_texts, overlay_metrics)

                            overlay_data = self._build_overlay_data(overlay_metrics, feedback_texts)
                            self._draw_live_overlay(frame_bgr, overlay_data, feedback_texts)

                            final_frame = frame_bgr
                            self.frame_signal.emit(final_frame)
                            self.thumbnail_frame_signal.emit(
                                cv2.resize(final_frame, (160, 90), interpolation=cv2.INTER_AREA)
                            )
                            self.counters_signal.emit(
                                self.exercise_analyzer.rep_count,
                                self.exercise_analyzer.set_count,
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
                                now = time.monotonic()
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
                            overlay_data = self._build_overlay_data(
                                metrics={
                                    "rep_count": self.exercise_analyzer.rep_count,
                                    "set_count": self.exercise_analyzer.set_count,
                                    "current_weight": self.exercise_analyzer.current_weight,
                                },
                                feedback_texts=[],
                                status_hint=status_hint,
                            )
                            self._draw_live_overlay(frame_bgr, overlay_data, [])
                            self.frame_signal.emit(frame_bgr)
                            self.thumbnail_frame_signal.emit(
                                cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
                            )
                            self.counters_signal.emit(
                                self.exercise_analyzer.rep_count,
                                self.exercise_analyzer.set_count,
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
                                    or self.exercise_analyzer.set_count > 0
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
                                self.counters_signal.emit(0, 0)

                        self.msleep(10)
                        continue

                    except Exception as e:
                        self.status_signal.emit("Exercise analysis error.")
                        logging.error(f"Exercise analysis error: {e}")
                        self.msleep(10)
                        continue

                # If capturing new user face data
                if self.capturing_new_user_data:
                    try:
                        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
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
                        overlay_data = self._build_overlay_data(
                            metrics={"rep_count": 0, "set_count": 0, "current_weight": 0},
                            feedback_texts=[],
                            status_hint=hint,
                        )
                        self._draw_live_overlay(display_frame, overlay_data, [])

                        self.frame_signal.emit(display_frame)
                        self.thumbnail_frame_signal.emit(
                            cv2.resize(display_frame, (160, 90), interpolation=cv2.INTER_AREA)
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
                            self._reset_face_recognition_state()

                        self.msleep(10)
                        continue
                    except Exception as e:
                        self.status_signal.emit("New user registration error.")
                        logging.error(f"New user registration error: {e}")
                        self.msleep(10)
                        continue

                # Face Recognition Mode
                if self.face_recognition_active:
                    try:
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

                        for ((top, right, bottom, left), name) in zip(face_locations, display_face_names):
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4
                            cv2.rectangle(
                                display_frame,
                                (left, top),
                                (right, bottom),
                                (110, 255, 180),
                                2,
                            )
                            cv2.rectangle(
                                display_frame,
                                (left, bottom - 28),
                                (right, bottom),
                                (18, 32, 54),
                                cv2.FILLED,
                            )
                            cv2.putText(
                                display_frame,
                                name,
                                (left + 8, bottom - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.56,
                                (224, 250, 248),
                                2,
                                cv2.LINE_AA,
                            )

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
                                    self.exercise_analyzer = ExerciseAnalyzer(
                                        self.exercise_choice,
                                        aruco_detector,
                                        self.db_handler,
                                        user_id=user_info.get("user_id"),
                                        username_snapshot=recognized_user,
                                    )
                                    self._exit_save_retry_count = 0
                                    self._analysis_pose_grace_frames_remaining = self.pose_acquire_grace_frames
                                    self._last_reliable_pose_ts = time.monotonic()
                                    self.exercise_analysis_active = True
                                    self.face_recognition_active = False
                                    self._reset_face_recognition_state()
                        else:
                            self.unknown_frames = max(0, self.unknown_frames - self.UNKNOWN_FRAME_DECAY)
                            self.known_frames = max(0, self.known_frames - self.KNOWN_FRAME_DECAY)
                            if self.known_frames == 0:
                                self.last_recognized_user = None
                            if self.unknown_frames == 0:
                                self.unknown_stable_detected = False

                        overlay_data = self._build_overlay_data(
                            metrics={"rep_count": 0, "set_count": 0, "current_weight": 0},
                            feedback_texts=[],
                            status_hint=status_hint,
                        )
                        self._draw_live_overlay(display_frame, overlay_data, [])

                        self.frame_signal.emit(display_frame)
                        self.thumbnail_frame_signal.emit(
                            cv2.resize(display_frame, (160, 90), interpolation=cv2.INTER_AREA)
                        )
                    except Exception as e:
                        self.status_signal.emit("Face recognition error.")
                        logging.error(f"Face recognition error: {e}")

                self.msleep(10)
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
