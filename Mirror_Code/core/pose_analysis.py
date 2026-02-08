# core/pose_analysis.py

import logging
from datetime import datetime
from math import hypot
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
        self.active_curl_arm = None
        self.rep_active_arm = None
        self.rep_down_wrist_to_shoulder_ratio = None
        self.side_pose_stable_frames = 0
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

    @staticmethod
    def _landmark_xy(landmarks, landmark):
        point = landmarks[landmark.value]
        return [point.x, point.y]

    @staticmethod
    def _landmark_xyz(landmarks, landmark):
        point = landmarks[landmark.value]
        return [point.x, point.y, float(getattr(point, "z", 0.0))]

    @staticmethod
    def _landmark_visibility(landmarks, landmark):
        point = landmarks[landmark.value]
        return float(getattr(point, "visibility", 1.0))

    @staticmethod
    def _distance(p1, p2):
        return hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _build_bicep_arm_metrics(
        self,
        landmarks,
        torso_height,
        shoulder_lmk,
        elbow_lmk,
        wrist_lmk,
        hip_lmk,
    ):
        shoulder = self._landmark_xy(landmarks, shoulder_lmk)
        elbow = self._landmark_xy(landmarks, elbow_lmk)
        wrist = self._landmark_xy(landmarks, wrist_lmk)
        hip = self._landmark_xy(landmarks, hip_lmk)

        return {
            "elbow_angle": calculate_joint_angle(shoulder, elbow, wrist),
            "underarm_angle": calculate_joint_angle(hip, shoulder, elbow),
            "elbow_hip_distance_ratio": self._distance(elbow, hip) / torso_height,
            "elbow_shoulder_x_ratio": abs(elbow[0] - shoulder[0]) / torso_height,
            "elbow_drop_ratio": (elbow[1] - shoulder[1]) / torso_height,
            "wrist_shoulder_ratio": self._distance(wrist, shoulder) / torso_height,
            "wrist_shoulder_x_ratio": abs(wrist[0] - shoulder[0]) / torso_height,
            "wrist_elbow_x_ratio": abs(wrist[0] - elbow[0]) / torso_height,
            "visibility": min(
                self._landmark_visibility(landmarks, shoulder_lmk),
                self._landmark_visibility(landmarks, elbow_lmk),
                self._landmark_visibility(landmarks, wrist_lmk),
                self._landmark_visibility(landmarks, hip_lmk),
            ),
        }

    def _bicep_curl_form_gate(self, landmarks):
        cfg = exercise_config.get("bicep_curl", {})
        gate_cfg = cfg.get("form_gate", {})

        front_shoulder_to_torso_ratio = float(
            gate_cfg.get("front_shoulder_to_torso_ratio", 0.78)
        )
        front_hip_to_torso_ratio = float(gate_cfg.get("front_hip_to_torso_ratio", 0.62))
        max_centered_nose_offset_ratio = float(
            gate_cfg.get("max_centered_nose_offset_ratio", 0.18)
        )
        max_front_shoulder_depth_delta = float(
            gate_cfg.get("max_front_shoulder_depth_delta", 0.025)
        )
        max_front_hip_depth_delta = float(gate_cfg.get("max_front_hip_depth_delta", 0.02))
        min_side_shoulder_depth_delta = float(
            gate_cfg.get("min_side_shoulder_depth_delta", 0.045)
        )
        min_side_hip_depth_delta = float(gate_cfg.get("min_side_hip_depth_delta", 0.035))
        max_side_shoulder_to_torso_ratio = float(
            gate_cfg.get("max_side_shoulder_to_torso_ratio", 0.72)
        )

        max_underarm_angle = float(gate_cfg.get("max_underarm_angle", 48))
        max_elbow_hip_distance_ratio = float(
            gate_cfg.get("max_elbow_hip_distance_ratio", 0.58)
        )
        max_elbow_shoulder_x_ratio = float(
            gate_cfg.get("max_elbow_shoulder_x_ratio", 0.32)
        )
        min_elbow_drop_ratio = float(gate_cfg.get("min_elbow_drop_ratio", 0.12))
        startup_max_underarm_angle = float(
            gate_cfg.get("startup_max_underarm_angle", max_underarm_angle)
        )
        startup_max_elbow_hip_distance_ratio = float(
            gate_cfg.get(
                "startup_max_elbow_hip_distance_ratio",
                max_elbow_hip_distance_ratio,
            )
        )
        startup_max_elbow_shoulder_x_ratio = float(
            gate_cfg.get(
                "startup_max_elbow_shoulder_x_ratio",
                max_elbow_shoulder_x_ratio,
            )
        )
        startup_min_elbow_drop_ratio = float(
            gate_cfg.get("startup_min_elbow_drop_ratio", min_elbow_drop_ratio)
        )
        max_wrist_to_shoulder_ratio_at_top = float(
            gate_cfg.get("max_wrist_to_shoulder_ratio_at_top", 0.62)
        )
        max_top_wrist_shoulder_x_ratio = float(
            gate_cfg.get("max_top_wrist_shoulder_x_ratio", 0.26)
        )
        min_arm_visibility = float(gate_cfg.get("min_arm_visibility", 0.35))
        min_valid_arms = int(gate_cfg.get("min_valid_arms", 1))

        try:
            left_shoulder_xyz = self._landmark_xyz(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
            right_shoulder_xyz = self._landmark_xyz(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
            left_hip_xyz = self._landmark_xyz(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
            right_hip_xyz = self._landmark_xyz(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
            nose = self._landmark_xy(landmarks, mp_pose.PoseLandmark.NOSE)

            left_shoulder = left_shoulder_xyz[:2]
            right_shoulder = right_shoulder_xyz[:2]
            left_hip = left_hip_xyz[:2]
            right_hip = right_hip_xyz[:2]

            shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_mid_y = (left_hip[1] + right_hip[1]) / 2
            torso_height = max(1e-6, abs(shoulder_mid_y - hip_mid_y))

            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            hip_width = abs(left_hip[0] - right_hip[0])
            shoulder_to_torso_ratio = shoulder_width / torso_height
            hip_to_torso_ratio = hip_width / torso_height

            shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
            nose_offset_ratio = abs(nose[0] - shoulder_mid_x) / max(1e-6, shoulder_width)

            shoulder_depth_delta = abs(left_shoulder_xyz[2] - right_shoulder_xyz[2])
            hip_depth_delta = abs(left_hip_xyz[2] - right_hip_xyz[2])

            front_facing = (
                shoulder_depth_delta <= max_front_shoulder_depth_delta
                and hip_depth_delta <= max_front_hip_depth_delta
                and shoulder_to_torso_ratio >= front_shoulder_to_torso_ratio
                and hip_to_torso_ratio >= front_hip_to_torso_ratio
                and nose_offset_ratio <= max_centered_nose_offset_ratio
            )
            side_pose_votes = 0
            if shoulder_depth_delta >= min_side_shoulder_depth_delta:
                side_pose_votes += 1
            if hip_depth_delta >= min_side_hip_depth_delta:
                side_pose_votes += 1
            if shoulder_to_torso_ratio <= max_side_shoulder_to_torso_ratio:
                side_pose_votes += 1
            side_pose_detected = side_pose_votes >= 2

            arm_metrics = {
                "left": self._build_bicep_arm_metrics(
                    landmarks,
                    torso_height,
                    mp_pose.PoseLandmark.LEFT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_ELBOW,
                    mp_pose.PoseLandmark.LEFT_WRIST,
                    mp_pose.PoseLandmark.LEFT_HIP,
                ),
                "right": self._build_bicep_arm_metrics(
                    landmarks,
                    torso_height,
                    mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.RIGHT_ELBOW,
                    mp_pose.PoseLandmark.RIGHT_WRIST,
                    mp_pose.PoseLandmark.RIGHT_HIP,
                ),
            }

            valid_arms = []
            startup_valid_arms = []
            for side, metrics in arm_metrics.items():
                arm_ok = (
                    metrics["underarm_angle"] <= max_underarm_angle
                    and metrics["elbow_hip_distance_ratio"] <= max_elbow_hip_distance_ratio
                    and metrics["elbow_shoulder_x_ratio"] <= max_elbow_shoulder_x_ratio
                    and metrics["elbow_drop_ratio"] >= min_elbow_drop_ratio
                    and metrics["visibility"] >= min_arm_visibility
                )
                startup_arm_ok = (
                    metrics["underarm_angle"] <= startup_max_underarm_angle
                    and metrics["elbow_hip_distance_ratio"] <= startup_max_elbow_hip_distance_ratio
                    and metrics["elbow_shoulder_x_ratio"] <= startup_max_elbow_shoulder_x_ratio
                    and metrics["elbow_drop_ratio"] >= startup_min_elbow_drop_ratio
                    and metrics["visibility"] >= min_arm_visibility
                )
                metrics["arm_ok"] = arm_ok
                metrics["startup_arm_ok"] = startup_arm_ok
                if arm_ok:
                    valid_arms.append(side)
                if startup_arm_ok:
                    startup_valid_arms.append(side)

            if self.active_curl_arm in valid_arms:
                active_arm = self.active_curl_arm
            elif valid_arms:
                active_arm = min(
                    valid_arms,
                    key=lambda side: (
                        arm_metrics[side]["elbow_hip_distance_ratio"],
                        -arm_metrics[side]["visibility"],
                    ),
                )
            else:
                active_arm = None

            if active_arm in startup_valid_arms:
                startup_active_arm = active_arm
            elif self.active_curl_arm in startup_valid_arms:
                startup_active_arm = self.active_curl_arm
            elif startup_valid_arms:
                startup_active_arm = min(
                    startup_valid_arms,
                    key=lambda side: (
                        arm_metrics[side]["elbow_hip_distance_ratio"],
                        -arm_metrics[side]["visibility"],
                    ),
                )
            else:
                startup_active_arm = None

            self.active_curl_arm = active_arm
            active_arm_metrics = arm_metrics.get(active_arm) if active_arm else None
            startup_active_arm_metrics = (
                arm_metrics.get(startup_active_arm) if startup_active_arm else None
            )

            top_position_ok = bool(
                active_arm_metrics
                and active_arm_metrics["wrist_shoulder_ratio"] <= max_wrist_to_shoulder_ratio_at_top
                and active_arm_metrics["wrist_shoulder_x_ratio"] <= max_top_wrist_shoulder_x_ratio
            )

            form_ok = (
                side_pose_detected
                and not front_facing
                and len(valid_arms) >= max(1, min_valid_arms)
                and active_arm_metrics is not None
            )
            startup_form_ok = (
                side_pose_detected
                and not front_facing
                and startup_active_arm_metrics is not None
            )

            return {
                "form_ok": form_ok,
                "startup_form_ok": startup_form_ok,
                "front_facing": front_facing,
                "side_pose_detected": side_pose_detected,
                "side_pose_votes": side_pose_votes,
                "active_arm": active_arm,
                "rep_angle": active_arm_metrics["elbow_angle"] if active_arm_metrics else None,
                "startup_active_arm": startup_active_arm,
                "startup_rep_angle": (
                    startup_active_arm_metrics["elbow_angle"]
                    if startup_active_arm_metrics
                    else None
                ),
                "active_wrist_shoulder_ratio": (
                    active_arm_metrics["wrist_shoulder_ratio"] if active_arm_metrics else None
                ),
                "top_position_ok": top_position_ok,
                "shoulder_to_torso_ratio": shoulder_to_torso_ratio,
                "hip_to_torso_ratio": hip_to_torso_ratio,
                "nose_offset_ratio": nose_offset_ratio,
                "shoulder_depth_delta": shoulder_depth_delta,
                "hip_depth_delta": hip_depth_delta,
                "valid_arms": valid_arms,
                "startup_valid_arms": startup_valid_arms,
                "arm_metrics": arm_metrics,
            }
        except Exception as exc:
            logger.debug("Unable to evaluate bicep-curl form gate: %s", exc)
            self.active_curl_arm = None
            return {
                "form_ok": False,
                "startup_form_ok": False,
                "front_facing": False,
                "side_pose_detected": False,
                "side_pose_votes": 0,
                "active_arm": None,
                "rep_angle": None,
                "startup_active_arm": None,
                "startup_rep_angle": None,
                "active_wrist_shoulder_ratio": None,
                "top_position_ok": False,
                "shoulder_to_torso_ratio": None,
                "hip_to_torso_ratio": None,
                "nose_offset_ratio": None,
                "shoulder_depth_delta": None,
                "hip_depth_delta": None,
                "valid_arms": [],
                "startup_valid_arms": [],
                "arm_metrics": {},
            }

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
        else:
            self.bend_warning_displayed = False
        posture_ok = not self.bend_warning_displayed

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
        bicep_form = {
            "form_ok": True,
            "startup_form_ok": True,
            "front_facing": False,
            "side_pose_detected": True,
            "side_pose_votes": 3,
            "active_arm": None,
            "rep_angle": None,
            "startup_active_arm": None,
            "startup_rep_angle": None,
            "active_wrist_shoulder_ratio": None,
            "top_position_ok": True,
            "shoulder_depth_delta": None,
            "hip_depth_delta": None,
            "arm_metrics": {},
        }
        if self.exercise == "bicep_curl":
            bicep_form = self._bicep_curl_form_gate(landmarks)
            overlay_metrics["bicep_form_ok"] = bool(bicep_form["form_ok"])
            overlay_metrics["bicep_front_facing"] = bool(bicep_form["front_facing"])
            overlay_metrics["bicep_side_pose_detected"] = bool(
                bicep_form.get("side_pose_detected", False)
            )
            overlay_metrics["bicep_side_pose_votes"] = int(bicep_form.get("side_pose_votes", 0))
            overlay_metrics["bicep_shoulder_depth_delta"] = bicep_form.get(
                "shoulder_depth_delta"
            )
            overlay_metrics["bicep_hip_depth_delta"] = bicep_form.get("hip_depth_delta")
            overlay_metrics["active_curl_arm"] = bicep_form.get("active_arm")

            for side in ("left", "right"):
                arm_metrics = (bicep_form.get("arm_metrics") or {}).get(side)
                if arm_metrics and arm_metrics.get("underarm_angle") is not None:
                    angle_values[f"{side}_underarm"] = arm_metrics["underarm_angle"]

            gate_cfg = cfg.get("form_gate", {})
            required_side_pose_frames = int(gate_cfg.get("required_side_pose_frames", 8))
            startup_required_side_pose_frames = int(
                gate_cfg.get(
                    "startup_required_side_pose_frames",
                    max(1, required_side_pose_frames),
                )
            )
            side_pose_detected = bool(bicep_form.get("side_pose_detected", False))
            if side_pose_detected and not bicep_form["front_facing"]:
                self.side_pose_stable_frames = min(
                    required_side_pose_frames + 10, self.side_pose_stable_frames + 1
                )
            else:
                self.side_pose_stable_frames = max(0, self.side_pose_stable_frames - 2)
            side_pose_ready = self.side_pose_stable_frames >= max(1, required_side_pose_frames)
            startup_side_pose_ready = self.side_pose_stable_frames >= max(
                1,
                startup_required_side_pose_frames,
            )
            overlay_metrics["bicep_side_pose_ready"] = bool(side_pose_ready)

            startup_form_ok = bool(bicep_form.get("startup_form_ok", bicep_form["form_ok"]))
            gate_ready = (
                side_pose_ready and bicep_form["form_ok"]
                if self.stable_start_detected
                else startup_side_pose_ready and startup_form_ok
            )

            if not gate_ready:
                if bicep_form["front_facing"]:
                    feedback_texts.append("Turn slightly sideways. Front-facing curls are not counted.")
                else:
                    feedback_texts.append(
                        "Face slightly sideways, keep elbow pinned to torso, and curl toward shoulder."
                    )
        else:
            side_pose_ready = True
            startup_side_pose_ready = True

        start_rep_angle = (
            bicep_form.get("startup_rep_angle")
            if self.exercise == "bicep_curl"
            else None
        )

        rep_angle = (
            (
                start_rep_angle
                if start_rep_angle is not None and not self.stable_start_detected
                else bicep_form["rep_angle"]
            )
            if self.exercise == "bicep_curl"
            else self._resolve_rep_angle(angle_values)
        )
        if rep_angle is not None:
            down_min, down_max = cfg["down_range"]
            up_min, up_max = cfg["up_range"]
            startup_down_min, startup_down_max = down_min, down_max
            stable_frames_required = 30

            if self.exercise == "bicep_curl":
                gate_cfg = cfg.get("form_gate", {})
                startup_form_ok = bool(bicep_form.get("startup_form_ok", bicep_form["form_ok"]))
                startup_down_range = gate_cfg.get("startup_down_range")
                if isinstance(startup_down_range, (list, tuple)) and len(startup_down_range) == 2:
                    startup_down_min = float(startup_down_range[0])
                    startup_down_max = float(startup_down_range[1])

                stable_frames_required = int(
                    gate_cfg.get("startup_stable_frames_required", 30)
                )
                start_form_ready = startup_form_ok and startup_side_pose_ready and posture_ok
                form_ready = bicep_form["form_ok"] and side_pose_ready and posture_ok
            else:
                start_form_ready = posture_ok
                form_ready = posture_ok

            if not self.stable_start_detected:
                if start_form_ready and startup_down_min <= rep_angle <= startup_down_max:
                    self.stable_frames += 1
                    if self.stable_frames >= max(1, stable_frames_required):
                        self.stable_start_detected = True
                        self.exercise_start_time = datetime.utcnow().isoformat()
                        feedback_texts.append("Ready to start!")
                        self.rep_state = 1
                        if self.exercise == "bicep_curl":
                            self.rep_down_wrist_to_shoulder_ratio = bicep_form[
                                "active_wrist_shoulder_ratio"
                            ]
                else:
                    self.stable_frames = max(0, self.stable_frames - 1)
            else:
                if self.exercise == "bicep_curl":
                    if not form_ready:
                        self.rep_state = 0
                        self.rep_start_angle = None
                        self.rep_active_arm = None
                        self.rep_down_wrist_to_shoulder_ratio = None
                    else:
                        active_arm = bicep_form["active_arm"]
                        if self.rep_state == 0 and down_min <= rep_angle <= down_max:
                            self.rep_state = 1
                            self.rep_active_arm = active_arm
                            self.rep_down_wrist_to_shoulder_ratio = bicep_form[
                                "active_wrist_shoulder_ratio"
                            ]
                        elif self.rep_state == 1 and up_min <= rep_angle <= up_max:
                            min_wrist_delta = float(
                                cfg.get("form_gate", {}).get(
                                    "min_top_wrist_to_shoulder_delta",
                                    0.08,
                                )
                            )
                            wrist_ratio_now = bicep_form["active_wrist_shoulder_ratio"]
                            wrist_ratio_down = self.rep_down_wrist_to_shoulder_ratio
                            moved_toward_shoulder = (
                                wrist_ratio_now is not None
                                and wrist_ratio_down is not None
                                and wrist_ratio_now <= (wrist_ratio_down - min_wrist_delta)
                            )

                            if (
                                bicep_form["top_position_ok"]
                                and moved_toward_shoulder
                                and active_arm is not None
                            ):
                                self.rep_state = 2
                                self.rep_start_angle = rep_angle
                                self.rep_active_arm = active_arm
                            else:
                                feedback_texts.append(
                                    "Drive wrist to shoulder without flaring elbow out to the side."
                                )
                        elif self.rep_state == 2:
                            same_arm = active_arm is not None and active_arm == self.rep_active_arm
                            if same_arm and down_min <= rep_angle <= down_max:
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
                                self.rep_down_wrist_to_shoulder_ratio = bicep_form[
                                    "active_wrist_shoulder_ratio"
                                ]
                            elif not same_arm:
                                self.rep_state = 0
                                self.rep_start_angle = None
                                self.rep_active_arm = None
                                self.rep_down_wrist_to_shoulder_ratio = None
                else:
                    if not form_ready:
                        self.rep_state = 1
                        self.rep_start_angle = None
                    elif self.rep_state == 1 and up_min <= rep_angle <= up_max:
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
