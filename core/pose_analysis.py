# core/pose_analysis.py

import logging
from datetime import datetime
import uuid

import mediapipe as mp

from .config import exercise_config
from .utils import calculate_bend_angle, calculate_joint_angle

mp_pose = mp.solutions.pose
logger = logging.getLogger(__name__)


class ExerciseAnalyzer:
    def __init__(self, exercise, aruco_detector, database_handler, user_id=None):
        self.exercise = exercise
        self.aruco_detector = aruco_detector
        self.database_handler = database_handler
        self.user_id = user_id
        self.reset_counters()

    def reset_counters(self):
        self.rep_count = 0
        self.set_count = 0
        self.stable_start_detected = False
        self.stable_frames = 0
        self.sets_reps = []
        self.rep_data = []
        self.rep_start_angle = None
        self.rep_end_angle = None
        self.current_weight = 0
        self.current_marker_id = None
        self.last_logged_weight = None
        self.last_logged_marker_id = None
        self.bend_warning_displayed = False
        self.person_in_frame = False
        self.no_pose_frames = 0

        # 0 = waiting for stable down position
        # 1 = waiting for up phase
        # 2 = waiting to return down to count a rep
        self.rep_state = 0
        self.exercise_start_time = None

    def detect_bend(self, landmarks):
        try:
            left_shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            ]
            right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            ]
            left_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            ]
            right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
            ]

            angle_left = calculate_bend_angle(left_shoulder, left_hip)
            angle_right = calculate_bend_angle(right_shoulder, right_hip)
            angle = (angle_left + angle_right) / 2
            tolerance = 3
            if angle > tolerance:
                return True, "back"
            if angle < -tolerance:
                return True, "front"
            return False, None
        except Exception as exc:
            logger.error("Error in detect_bend: %s", exc)
            return False, None

    def _log_weight_update(self):
        if self.current_marker_id is not None:
            if (
                self.last_logged_marker_id != self.current_marker_id
                or self.last_logged_weight != self.current_weight
            ):
                dict_type = (
                    self.aruco_detector.get_dict_type()
                    if self.aruco_detector and hasattr(self.aruco_detector, "get_dict_type")
                    else "unknown"
                )
                logger.info(
                    "ArUco weight detected: marker_id=%s, current_weight=%s lbs, dict=%s",
                    self.current_marker_id,
                    self.current_weight,
                    dict_type,
                )
        else:
            if self.last_logged_marker_id is not None or self.last_logged_weight not in (None, 0):
                logger.info("ArUco marker not detected. current_weight=0 lbs")

        self.last_logged_marker_id = self.current_marker_id
        self.last_logged_weight = self.current_weight

    def update_current_weight(self, frame):
        self.current_weight = 0
        self.current_marker_id = None
        if not self.aruco_detector:
            self._log_weight_update()
            return self.current_weight

        try:
            _, ids, _ = self.aruco_detector.detect_markers(frame)
            if ids is not None and len(ids) > 0:
                marker_ids = ids.flatten() if hasattr(ids, "flatten") else [ids[0][0]]
                self.current_marker_id = int(marker_ids[0])
                self.current_weight = int(self.current_marker_id)
        except (AttributeError, IndexError, ValueError, TypeError):
            logger.warning("Incorrect ArUco marker detected.")
            self.current_weight = 0
            self.current_marker_id = None
        except Exception as exc:
            logger.error("ArUco detection failed: %s", exc)
            self.current_weight = 0
            self.current_marker_id = None

        self._log_weight_update()
        return self.current_weight

    def _resolve_rep_angle(self, angle_values):
        if not angle_values:
            return None

        cfg = exercise_config.get(self.exercise, {})
        rep_angle_keys = cfg.get("rep_angle_keys") or list(angle_values.keys())
        rep_values = [angle_values[key] for key in rep_angle_keys if key in angle_values]
        if not rep_values:
            return None
        return sum(rep_values) / len(rep_values)

    def analyze_exercise_form(self, landmarks, frame):
        feedback_texts = []
        overlay_metrics = {
            "angles": {},
            "current_weight": int(self.current_weight),
            "rep_count": int(self.rep_count),
            "set_count": int(self.set_count),
            "stable_start_detected": bool(self.stable_start_detected),
        }

        if self.exercise not in exercise_config:
            feedback_texts.append("Exercise config missing.")
            return feedback_texts, overlay_metrics

        cfg = exercise_config[self.exercise]

        if cfg.get("bend_detection"):
            bend_detected, bend_type = self.detect_bend(landmarks)
            if bend_detected:
                if bend_type == "back":
                    feedback_texts.append("Warning: Keep your back straight!")
                elif bend_type == "front":
                    feedback_texts.append("Warning: Do not bend forward!")
                self.bend_warning_displayed = True
            else:
                self.bend_warning_displayed = False

        angle_values = {}
        for angle_name, points in cfg["angles"].items():
            try:
                p1 = [landmarks[points[0].value].x, landmarks[points[0].value].y]
                p2 = [landmarks[points[1].value].x, landmarks[points[1].value].y]
                p3 = [landmarks[points[2].value].x, landmarks[points[2].value].y]
                angle_values[angle_name] = calculate_joint_angle(p1, p2, p3)
            except Exception as exc:
                logger.error("Error calculating angle for %s: %s", angle_name, exc)
                angle_values[angle_name] = 0.0

        self.update_current_weight(frame)

        rep_angle = self._resolve_rep_angle(angle_values)
        if rep_angle is not None:
            down_min, down_max = cfg["down_range"]
            up_min, up_max = cfg["up_range"]

            if not self.stable_start_detected:
                if down_min <= rep_angle <= down_max:
                    self.stable_frames += 1
                    if self.stable_frames > 30:
                        self.stable_start_detected = True
                        self.exercise_start_time = datetime.utcnow().isoformat()
                        feedback_texts.append("Ready to start!")
                        self.rep_state = 1
                else:
                    self.stable_frames = max(0, self.stable_frames - 1)
            else:
                if self.rep_state == 1 and up_min <= rep_angle <= up_max:
                    self.rep_state = 2
                    self.rep_start_angle = rep_angle
                elif self.rep_state == 2 and down_min <= rep_angle <= down_max:
                    self.rep_state = 1
                    self.rep_count += 1
                    self.rep_end_angle = rep_angle
                    self.rep_data.append(
                        {
                            "start_angle": self.rep_start_angle,
                            "end_angle": self.rep_end_angle,
                            "weight": int(self.current_weight),
                        }
                    )

                reps_per_set = cfg.get("reps_per_set", 12)
                if self.rep_count >= reps_per_set:
                    self.sets_reps.append(self.rep_count)
                    self.set_count += 1
                    self.rep_count = 0
                    feedback_texts.append("Set complete!")

        overlay_metrics["angles"] = angle_values
        overlay_metrics["current_weight"] = int(self.current_weight)
        overlay_metrics["rep_count"] = int(self.rep_count)
        overlay_metrics["set_count"] = int(self.set_count)
        overlay_metrics["stable_start_detected"] = bool(self.stable_start_detected)

        return feedback_texts, overlay_metrics

    def update_data(self):
        try:
            if not self.user_id:
                logger.warning("Skipping exercise data save because user_id is missing.")
                return None

            has_activity = bool(
                self.rep_data or self.rep_count > 0 or self.set_count > 0 or self.sets_reps
            )
            if not has_activity:
                return None

            sets_reps = [int(rep) for rep in self.sets_reps]
            if self.rep_count > 0:
                sets_reps.append(int(self.rep_count))

            record = {
                "id": str(uuid.uuid4()),
                "user_id": self.user_id,
                "exercise": self.exercise,
                "set_count": self.set_count,
                "sets_reps": sets_reps,
                "rep_data": [
                    {
                        "start_angle": rep["start_angle"],
                        "end_angle": rep["end_angle"],
                        "weight": int(rep["weight"]),
                    }
                    for rep in self.rep_data
                ],
                "timestamp": self.exercise_start_time or datetime.utcnow().isoformat(),
                "date": datetime.utcnow().strftime("%Y-%m-%d"),
            }

            if self.database_handler.insert_exercise_data_local(record):
                self.reset_counters()
                return record

            logger.error("Failed to insert exercise data into the database.")
            return None
        except Exception as exc:
            logger.error("Error updating exercise data: %s", exc)
            return None
