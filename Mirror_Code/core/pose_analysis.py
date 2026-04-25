# core/pose_analysis.py

import logging
from datetime import datetime
import os
import time
import uuid

import mediapipe as mp

from .config import exercise_config
from .form_gates import evaluate_bicep_curl_form_gate, evaluate_deadlift_form_gate
from .utils import calculate_bend_angle, calculate_joint_angle

mp_pose = mp.solutions.pose
logger = logging.getLogger(__name__)


class ExerciseAnalyzer:
    def __init__(
        self,
        exercise,
        aruco_detector,
        database_handler,
        user_id=None,
        username_snapshot=None,
        starting_set_count=0,
    ):
        self.exercise = exercise
        self.aruco_detector = aruco_detector
        self.database_handler = database_handler
        self.user_id = user_id
        self.username_snapshot = username_snapshot
        try:
            self.starting_set_count = max(0, int(starting_set_count or 0))
        except (TypeError, ValueError):
            self.starting_set_count = 0
        self.weight_detection_stride = max(
            1,
            int(os.getenv("SMART_MIRROR_ARUCO_DETECT_EVERY_N_FRAMES", "2")),
        )
        self.ready_feedback_hold_frames = max(
            1,
            int(os.getenv("SMART_MIRROR_READY_FEEDBACK_FRAMES", "8")),
        )
        self.reset_counters()

    def reset_counters(self):
        self.rep_count = 0
        self.set_count = int(self.starting_set_count)
        self.stable_start_detected = False
        self.stable_frames = 0
        self.sets_reps = []
        self.rep_data = []
        self.rep_start_angle = None
        self.rep_end_angle = None
        self.rep_started_ts = None
        self.rep_started_at_utc = None
        self.current_weight = 0
        self.current_marker_id = None
        self._weight_detect_frame_count = 0
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
        self.active_body_side = None
        self.rep_active_arm = None
        self.rep_down_wrist_to_shoulder_ratio = None
        self.side_pose_stable_frames = 0
        self.rep_left_down_zone = False
        self.rep_left_up_zone = False
        self.up_range_miss_flash_frames = 0
        self.down_range_miss_flash_frames = 0
        self.exercise_start_time = None
        self.ready_feedback_frames_remaining = 0

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

    def update_current_weight(self, frame, force=False):
        if not self.aruco_detector:
            self.current_weight = 0
            self.current_marker_id = None
            self._log_weight_update()
            return self.current_weight

        self._weight_detect_frame_count += 1
        if (
            not force
            and self._weight_detect_frame_count % self.weight_detection_stride != 0
        ):
            return self.current_weight

        detected_weight = 0
        detected_marker_id = None

        try:
            _, ids, _ = self.aruco_detector.detect_markers(frame)
            if ids is not None and len(ids) > 0:
                marker_ids = ids.flatten() if hasattr(ids, "flatten") else [ids[0][0]]
                detected_marker_id = int(marker_ids[0])
                detected_weight = int(detected_marker_id)
        except (AttributeError, IndexError, ValueError, TypeError):
            logger.warning("Incorrect ArUco marker detected.")
        except Exception as exc:
            logger.error("ArUco detection failed: %s", exc)

        self.current_marker_id = detected_marker_id
        self.current_weight = detected_weight

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

    def _bicep_curl_form_gate(self, landmarks):
        cfg = exercise_config.get("bicep_curl", {})
        form_result, active_arm = evaluate_bicep_curl_form_gate(
            landmarks=landmarks,
            cfg=cfg,
            active_curl_arm=self.active_curl_arm,
        )
        self.active_curl_arm = active_arm
        return form_result

    def _deadlift_form_gate(self, landmarks):
        cfg = exercise_config.get("deadlift", {})
        form_result, active_side = evaluate_deadlift_form_gate(
            landmarks=landmarks,
            cfg=cfg,
            active_body_side=self.active_body_side,
        )
        self.active_body_side = active_side
        return form_result

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
        down_min, down_max = cfg.get("down_range", (0.0, 0.0))
        up_min, up_max = cfg.get("up_range", (0.0, 0.0))
        count_range_margin = max(0.0, float(cfg.get("count_range_margin", 0.0)))
        count_down_min = float(down_min) - count_range_margin
        count_down_max = float(down_max) + count_range_margin
        count_up_min = float(up_min) - count_range_margin
        count_up_max = float(up_max) + count_range_margin
        phase_leave_margin = 12.0

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
        self.up_range_miss_flash_frames = max(0, self.up_range_miss_flash_frames - 1)
        self.down_range_miss_flash_frames = max(0, self.down_range_miss_flash_frames - 1)

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
        deadlift_form = {
            "form_ok": True,
            "startup_form_ok": True,
            "front_facing": False,
            "side_pose_detected": True,
            "side_pose_votes": 3,
            "active_side": None,
            "rep_angle": None,
            "startup_active_side": None,
            "startup_rep_angle": None,
            "top_position_ok": True,
            "startup_top_position_ok": True,
            "bottom_position_ok": True,
            "active_knee_angle": None,
            "active_torso_lean_angle": None,
            "shoulder_depth_delta": None,
            "hip_depth_delta": None,
            "side_metrics": {},
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
        elif self.exercise == "deadlift":
            deadlift_form = self._deadlift_form_gate(landmarks)
            overlay_metrics["deadlift_form_ok"] = bool(deadlift_form["form_ok"])
            overlay_metrics["deadlift_front_facing"] = bool(deadlift_form["front_facing"])
            overlay_metrics["deadlift_side_pose_detected"] = bool(
                deadlift_form.get("side_pose_detected", False)
            )
            overlay_metrics["deadlift_side_pose_votes"] = int(
                deadlift_form.get("side_pose_votes", 0)
            )
            overlay_metrics["deadlift_shoulder_depth_delta"] = deadlift_form.get(
                "shoulder_depth_delta"
            )
            overlay_metrics["deadlift_hip_depth_delta"] = deadlift_form.get("hip_depth_delta")
            overlay_metrics["deadlift_active_side"] = deadlift_form.get("active_side")
            overlay_metrics["deadlift_top_position_ok"] = bool(
                deadlift_form.get("top_position_ok", False)
            )
            overlay_metrics["deadlift_bottom_position_ok"] = bool(
                deadlift_form.get("bottom_position_ok", False)
            )

            active_side = deadlift_form.get("active_side")
            if active_side:
                knee_angle = deadlift_form.get("active_knee_angle")
                torso_lean_angle = deadlift_form.get("active_torso_lean_angle")
                if knee_angle is not None:
                    angle_values[f"{active_side}_knee_active"] = float(knee_angle)
                if torso_lean_angle is not None:
                    angle_values[f"{active_side}_torso_lean"] = float(torso_lean_angle)

            gate_cfg = cfg.get("form_gate", {})
            required_side_pose_frames = int(gate_cfg.get("required_side_pose_frames", 6))
            startup_required_side_pose_frames = int(
                gate_cfg.get(
                    "startup_required_side_pose_frames",
                    max(1, required_side_pose_frames),
                )
            )
            side_pose_detected = bool(deadlift_form.get("side_pose_detected", False))
            if side_pose_detected and not deadlift_form["front_facing"]:
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
            overlay_metrics["deadlift_side_pose_ready"] = bool(side_pose_ready)

            startup_form_ok = bool(deadlift_form.get("startup_form_ok", deadlift_form["form_ok"]))
            gate_ready = (
                side_pose_ready and deadlift_form["form_ok"]
                if self.stable_start_detected
                else startup_side_pose_ready and startup_form_ok
            )

            if not gate_ready:
                if deadlift_form["front_facing"]:
                    feedback_texts.append("Turn sideways. Deadlift reps count only from side view.")
                else:
                    feedback_texts.append(
                        "Stand side-on with shoulder, hip, knee, and ankle visible before lifting."
                    )
        else:
            side_pose_ready = True
            startup_side_pose_ready = True

        if self.exercise == "bicep_curl":
            start_rep_angle = bicep_form.get("startup_rep_angle")
        elif self.exercise == "deadlift":
            start_rep_angle = deadlift_form.get("startup_rep_angle")
        else:
            start_rep_angle = None

        rep_angle = (
            (
                start_rep_angle
                if start_rep_angle is not None and not self.stable_start_detected
                else bicep_form["rep_angle"]
            )
            if self.exercise == "bicep_curl"
            else (
                (
                    start_rep_angle
                    if start_rep_angle is not None and not self.stable_start_detected
                    else deadlift_form["rep_angle"]
                )
                if self.exercise == "deadlift"
                else self._resolve_rep_angle(angle_values)
            )
        )
        if rep_angle is not None:
            startup_down_min, startup_down_max = down_min, down_max
            startup_count_down_min, startup_count_down_max = count_down_min, count_down_max
            stable_frames_required = 30

            if self.exercise == "bicep_curl":
                gate_cfg = cfg.get("form_gate", {})
                startup_form_ok = bool(bicep_form.get("startup_form_ok", bicep_form["form_ok"]))
                startup_down_range = gate_cfg.get("startup_down_range")
                if isinstance(startup_down_range, (list, tuple)) and len(startup_down_range) == 2:
                    startup_down_min = float(startup_down_range[0])
                    startup_down_max = float(startup_down_range[1])
                    startup_count_down_min = startup_down_min - count_range_margin
                    startup_count_down_max = startup_down_max + count_range_margin

                stable_frames_required = int(
                    gate_cfg.get("startup_stable_frames_required", 30)
                )
                start_form_ready = startup_form_ok and startup_side_pose_ready and posture_ok
                form_ready = bicep_form["form_ok"] and side_pose_ready and posture_ok
            elif self.exercise == "deadlift":
                gate_cfg = cfg.get("form_gate", {})
                startup_form_ok = bool(
                    deadlift_form.get("startup_form_ok", deadlift_form["form_ok"])
                )
                startup_top_position_ok = bool(
                    deadlift_form.get(
                        "startup_top_position_ok",
                        deadlift_form.get("top_position_ok", False),
                    )
                )
                startup_down_range = gate_cfg.get("startup_down_range")
                if isinstance(startup_down_range, (list, tuple)) and len(startup_down_range) == 2:
                    startup_down_min = float(startup_down_range[0])
                    startup_down_max = float(startup_down_range[1])
                    startup_count_down_min = startup_down_min - count_range_margin
                    startup_count_down_max = startup_down_max + count_range_margin

                stable_frames_required = int(
                    gate_cfg.get("startup_stable_frames_required", 18)
                )
                start_form_ready = (
                    startup_form_ok
                    and startup_side_pose_ready
                    and startup_top_position_ok
                    and posture_ok
                )
                form_ready = deadlift_form["form_ok"] and side_pose_ready and posture_ok
            else:
                start_form_ready = posture_ok
                form_ready = posture_ok

            if not self.stable_start_detected:
                self.rep_left_down_zone = False
                self.rep_left_up_zone = False
                if start_form_ready and startup_count_down_min <= rep_angle <= startup_count_down_max:
                    self.stable_frames += 1
                    if self.stable_frames >= max(1, stable_frames_required):
                        self.stable_start_detected = True
                        self.exercise_start_time = datetime.utcnow().isoformat()
                        feedback_texts.append("Ready to start!")
                        self.ready_feedback_frames_remaining = self.ready_feedback_hold_frames
                        self.rep_state = 1
                        self.rep_left_down_zone = False
                        self.rep_left_up_zone = False
                        if self.exercise == "bicep_curl":
                            self.rep_down_wrist_to_shoulder_ratio = bicep_form[
                                "active_wrist_shoulder_ratio"
                            ]
                        elif self.exercise == "deadlift":
                            self.rep_active_arm = deadlift_form.get("active_side")
                else:
                    self.stable_frames = max(0, self.stable_frames - 1)
            else:
                if self.exercise == "bicep_curl":
                    if not form_ready:
                        self.rep_state = 0
                        self.rep_start_angle = None
                        self.rep_active_arm = None
                        self.rep_down_wrist_to_shoulder_ratio = None
                        self.rep_left_down_zone = False
                        self.rep_left_up_zone = False
                    else:
                        active_arm = bicep_form["active_arm"]
                        if self.rep_state == 0 and count_down_min <= rep_angle <= count_down_max:
                            self.rep_state = 1
                            self.rep_active_arm = active_arm
                            self.rep_down_wrist_to_shoulder_ratio = bicep_form[
                                "active_wrist_shoulder_ratio"
                            ]
                            self.rep_left_down_zone = False
                            self.rep_left_up_zone = False
                        elif self.rep_state == 1 and count_up_min <= rep_angle <= count_up_max:
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
                            if wrist_ratio_now is None or wrist_ratio_down is None:
                                moved_toward_shoulder = True

                            if (
                                bicep_form["top_position_ok"]
                                and moved_toward_shoulder
                                and active_arm is not None
                            ):
                                self.rep_state = 2
                                self.rep_start_angle = rep_angle
                                self.rep_started_ts = time.time()
                                self.rep_started_at_utc = datetime.utcnow().isoformat()
                                self.rep_active_arm = active_arm
                                self.rep_left_down_zone = False
                                self.rep_left_up_zone = False
                            else:
                                self.up_range_miss_flash_frames = max(
                                    self.up_range_miss_flash_frames,
                                    24,
                                )
                                feedback_texts.append(
                                    "Drive wrist to shoulder without flaring elbow out to the side."
                                )
                        elif self.rep_state == 2:
                            same_arm = active_arm is not None and active_arm == self.rep_active_arm
                            if same_arm and count_down_min <= rep_angle <= count_down_max:
                                self.rep_state = 1
                                self.rep_count += 1
                                self.rep_end_angle = rep_angle
                                rep_ended_at_utc = datetime.utcnow().isoformat()
                                duration_seconds = None
                                if self.rep_started_ts is not None:
                                    duration_seconds = max(0.0, time.time() - self.rep_started_ts)
                                self.rep_data.append(
                                    {
                                        "start_angle": self.rep_start_angle,
                                        "end_angle": self.rep_end_angle,
                                        "weight": int(self.current_weight),
                                        "duration_seconds": duration_seconds,
                                        "rep_started_at_utc": self.rep_started_at_utc,
                                        "rep_ended_at_utc": rep_ended_at_utc,
                                    }
                                )
                                self.rep_started_ts = None
                                self.rep_started_at_utc = None
                                self.rep_down_wrist_to_shoulder_ratio = bicep_form[
                                    "active_wrist_shoulder_ratio"
                                ]
                                self.rep_left_down_zone = False
                                self.rep_left_up_zone = False
                            elif not same_arm:
                                self.rep_state = 0
                                self.rep_start_angle = None
                                self.rep_started_ts = None
                                self.rep_started_at_utc = None
                                self.rep_active_arm = None
                                self.rep_down_wrist_to_shoulder_ratio = None
                                self.rep_left_down_zone = False
                                self.rep_left_up_zone = False

                        if self.rep_state == 1:
                            if rep_angle < (down_min - phase_leave_margin):
                                self.rep_left_down_zone = True
                            if self.rep_left_down_zone and down_min <= rep_angle <= down_max:
                                self.up_range_miss_flash_frames = max(
                                    self.up_range_miss_flash_frames,
                                    18,
                                )
                                self.rep_left_down_zone = False
                        elif self.rep_state == 2:
                            if rep_angle > (up_max + phase_leave_margin):
                                self.rep_left_up_zone = True
                            if self.rep_left_up_zone and up_min <= rep_angle <= up_max:
                                self.down_range_miss_flash_frames = max(
                                    self.down_range_miss_flash_frames,
                                    18,
                                )
                                self.rep_left_up_zone = False
                elif self.exercise == "deadlift":
                    if not form_ready:
                        self.rep_state = 0
                        self.rep_start_angle = None
                        self.rep_started_ts = None
                        self.rep_started_at_utc = None
                        self.rep_active_arm = None
                        self.rep_left_down_zone = False
                        self.rep_left_up_zone = False
                    else:
                        active_side = deadlift_form.get("active_side")
                        top_position_ok = bool(deadlift_form.get("top_position_ok", False))
                        bottom_position_ok = bool(deadlift_form.get("bottom_position_ok", False))

                        if (
                            self.rep_state == 0
                            and count_down_min <= rep_angle <= count_down_max
                            and top_position_ok
                        ):
                            self.rep_state = 1
                            self.rep_active_arm = active_side
                            self.rep_left_down_zone = False
                            self.rep_left_up_zone = False
                        elif self.rep_state == 1 and count_up_min <= rep_angle <= count_up_max:
                            same_side = active_side is not None and (
                                self.rep_active_arm is None or active_side == self.rep_active_arm
                            )
                            if bottom_position_ok and same_side:
                                self.rep_state = 2
                                self.rep_start_angle = rep_angle
                                self.rep_started_ts = time.time()
                                self.rep_started_at_utc = datetime.utcnow().isoformat()
                                self.rep_active_arm = active_side
                                self.rep_left_down_zone = False
                                self.rep_left_up_zone = False
                            else:
                                self.up_range_miss_flash_frames = max(
                                    self.up_range_miss_flash_frames,
                                    24,
                                )
                                feedback_texts.append(
                                    "Reach a controlled bottom hinge with neutral spine before standing up."
                                )
                        elif self.rep_state == 2:
                            same_side = active_side is not None and active_side == self.rep_active_arm
                            if (
                                same_side
                                and count_down_min <= rep_angle <= count_down_max
                                and top_position_ok
                            ):
                                self.rep_state = 1
                                self.rep_count += 1
                                self.rep_end_angle = rep_angle
                                rep_ended_at_utc = datetime.utcnow().isoformat()
                                duration_seconds = None
                                if self.rep_started_ts is not None:
                                    duration_seconds = max(0.0, time.time() - self.rep_started_ts)
                                self.rep_data.append(
                                    {
                                        "start_angle": self.rep_start_angle,
                                        "end_angle": self.rep_end_angle,
                                        "weight": int(self.current_weight),
                                        "duration_seconds": duration_seconds,
                                        "rep_started_at_utc": self.rep_started_at_utc,
                                        "rep_ended_at_utc": rep_ended_at_utc,
                                    }
                                )
                                self.rep_started_ts = None
                                self.rep_started_at_utc = None
                                self.rep_left_down_zone = False
                                self.rep_left_up_zone = False
                            elif same_side and count_down_min <= rep_angle <= count_down_max and not top_position_ok:
                                self.down_range_miss_flash_frames = max(
                                    self.down_range_miss_flash_frames,
                                    24,
                                )
                                feedback_texts.append(
                                    "Finish tall at lockout with hips and knees fully extended."
                                )
                            elif active_side is not None and not same_side:
                                self.rep_state = 0
                                self.rep_start_angle = None
                                self.rep_started_ts = None
                                self.rep_started_at_utc = None
                                self.rep_active_arm = None
                                self.rep_left_down_zone = False
                                self.rep_left_up_zone = False

                        if self.rep_state == 1:
                            if rep_angle < (down_min - phase_leave_margin):
                                self.rep_left_down_zone = True
                            if self.rep_left_down_zone and down_min <= rep_angle <= down_max:
                                self.up_range_miss_flash_frames = max(
                                    self.up_range_miss_flash_frames,
                                    18,
                                )
                                self.rep_left_down_zone = False
                        elif self.rep_state == 2:
                            if rep_angle > (up_max + phase_leave_margin):
                                self.rep_left_up_zone = True
                            if self.rep_left_up_zone and up_min <= rep_angle <= up_max:
                                self.down_range_miss_flash_frames = max(
                                    self.down_range_miss_flash_frames,
                                    18,
                                )
                                self.rep_left_up_zone = False
                else:
                    if not form_ready:
                        self.rep_state = 1
                        self.rep_start_angle = None
                        self.rep_left_down_zone = False
                        self.rep_left_up_zone = False
                    elif self.rep_state == 1 and count_up_min <= rep_angle <= count_up_max:
                        self.rep_state = 2
                        self.rep_start_angle = rep_angle
                        self.rep_started_ts = time.time()
                        self.rep_started_at_utc = datetime.utcnow().isoformat()
                        self.rep_left_down_zone = False
                        self.rep_left_up_zone = False
                    elif self.rep_state == 2 and count_down_min <= rep_angle <= count_down_max:
                        self.rep_state = 1
                        self.rep_count += 1
                        self.rep_end_angle = rep_angle
                        rep_ended_at_utc = datetime.utcnow().isoformat()
                        duration_seconds = None
                        if self.rep_started_ts is not None:
                            duration_seconds = max(0.0, time.time() - self.rep_started_ts)
                        self.rep_data.append(
                            {
                                "start_angle": self.rep_start_angle,
                                "end_angle": self.rep_end_angle,
                                "weight": int(self.current_weight),
                                "duration_seconds": duration_seconds,
                                "rep_started_at_utc": self.rep_started_at_utc,
                                "rep_ended_at_utc": rep_ended_at_utc,
                            }
                        )
                        self.rep_started_ts = None
                        self.rep_started_at_utc = None
                        self.rep_left_down_zone = False
                        self.rep_left_up_zone = False

                    if self.rep_state == 1:
                        if rep_angle < (down_min - phase_leave_margin):
                            self.rep_left_down_zone = True
                        if self.rep_left_down_zone and down_min <= rep_angle <= down_max:
                            self.up_range_miss_flash_frames = max(
                                self.up_range_miss_flash_frames,
                                18,
                            )
                            self.rep_left_down_zone = False
                    elif self.rep_state == 2:
                        if rep_angle > (up_max + phase_leave_margin):
                            self.rep_left_up_zone = True
                        if self.rep_left_up_zone and up_min <= rep_angle <= up_max:
                            self.down_range_miss_flash_frames = max(
                                self.down_range_miss_flash_frames,
                                18,
                            )
                            self.rep_left_up_zone = False

                reps_per_set = cfg.get("reps_per_set", 12)
                if self.rep_count >= reps_per_set:
                    self.sets_reps.append(self.rep_count)
                    self.set_count += 1
                    self.rep_count = 0
                    feedback_texts.append("Set complete!")

        if self.rep_state == 2 or self.rep_count > 0:
            self.ready_feedback_frames_remaining = 0
        if self.ready_feedback_frames_remaining > 0:
            if "Ready to start!" not in feedback_texts:
                feedback_texts.append("Ready to start!")
            self.ready_feedback_frames_remaining -= 1

        overlay_metrics["angles"] = angle_values
        overlay_metrics["current_weight"] = int(self.current_weight)
        overlay_metrics["rep_count"] = int(self.rep_count)
        overlay_metrics["set_count"] = int(self.set_count)
        overlay_metrics["stable_start_detected"] = bool(self.stable_start_detected)
        overlay_metrics["rep_state"] = int(self.rep_state)
        overlay_metrics["rep_angle"] = float(rep_angle) if rep_angle is not None else None
        overlay_metrics["up_range"] = (float(up_min), float(up_max))
        overlay_metrics["down_range"] = (float(down_min), float(down_max))
        overlay_metrics["up_range_miss"] = bool(self.up_range_miss_flash_frames > 0)
        overlay_metrics["down_range_miss"] = bool(self.down_range_miss_flash_frames > 0)

        return feedback_texts, overlay_metrics

    def update_data(self):
        try:
            if not self.user_id:
                logger.warning("Skipping exercise data save because user_id is missing.")
                return None

            has_activity = bool(
                self.rep_data or self.rep_count > 0 or self.sets_reps
            )
            if not has_activity:
                return None

            # Persist only completed sets here. Any in-progress work remains visible
            # through rep_data, but it should not be treated as another finished set.
            sets_reps = [int(rep) for rep in self.sets_reps]
            session_set_count = len(sets_reps)
            if session_set_count <= 0:
                session_set_count = max(0, int(self.set_count) - int(self.starting_set_count))

            mirror_id = os.getenv("SMART_MIRROR_MIRROR_ID")
            if not mirror_id and hasattr(self.database_handler, "_resolve_default_mirror_id"):
                mirror_id = self.database_handler._resolve_default_mirror_id()

            record = {
                "id": str(uuid.uuid4()),
                "user_id": self.user_id,
                "username_snapshot": self.username_snapshot,
                "exercise": self.exercise,
                "set_count": int(session_set_count),
                "sets_reps": sets_reps,
                "rep_data": [
                    {
                        "start_angle": rep["start_angle"],
                        "end_angle": rep["end_angle"],
                        "weight": int(rep["weight"]),
                        "duration_seconds": rep.get("duration_seconds"),
                        "rep_started_at_utc": rep.get("rep_started_at_utc"),
                        "rep_ended_at_utc": rep.get("rep_ended_at_utc"),
                    }
                    for rep in self.rep_data
                ],
                "timestamp": self.exercise_start_time or datetime.utcnow().isoformat(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "gym_id": os.getenv("SMART_MIRROR_GYM_ID"),
                "main_system_id": os.getenv("SMART_MIRROR_MAIN_SYSTEM_ID"),
                "mirror_id": mirror_id,
                "source_node": os.getenv("SMART_MIRROR_NODE_ROLE", "MAIN"),
                "equipment_type": "FREE_WEIGHT",
            }

            if self.database_handler.insert_exercise_data_local(record):
                self.reset_counters()
                return record

            logger.error("Failed to insert exercise data into the database.")
            return None
        except Exception as exc:
            logger.error("Error updating exercise data: %s", exc)
            return None
