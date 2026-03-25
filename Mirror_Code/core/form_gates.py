import logging
from math import atan2, degrees, hypot

import mediapipe as mp

from .utils import calculate_joint_angle

mp_pose = mp.solutions.pose
logger = logging.getLogger(__name__)


def _landmark_xy(landmarks, landmark):
    point = landmarks[landmark.value]
    return [point.x, point.y]


def _landmark_xyz(landmarks, landmark):
    point = landmarks[landmark.value]
    return [point.x, point.y, float(getattr(point, "z", 0.0))]


def _landmark_visibility(landmarks, landmark):
    point = landmarks[landmark.value]
    return float(getattr(point, "visibility", 1.0))


def _distance(p1, p2):
    return hypot(p1[0] - p2[0], p1[1] - p2[1])


def _torso_lean_angle(shoulder, hip):
    dx = float(shoulder[0]) - float(hip[0])
    dy = float(shoulder[1]) - float(hip[1])
    return degrees(atan2(abs(dx), max(1e-6, abs(dy))))


def _build_bicep_arm_metrics(
    landmarks,
    torso_height,
    shoulder_lmk,
    elbow_lmk,
    wrist_lmk,
    hip_lmk,
):
    shoulder = _landmark_xy(landmarks, shoulder_lmk)
    elbow = _landmark_xy(landmarks, elbow_lmk)
    wrist = _landmark_xy(landmarks, wrist_lmk)
    hip = _landmark_xy(landmarks, hip_lmk)

    return {
        "elbow_angle": calculate_joint_angle(shoulder, elbow, wrist),
        "underarm_angle": calculate_joint_angle(hip, shoulder, elbow),
        "elbow_hip_distance_ratio": _distance(elbow, hip) / torso_height,
        "elbow_shoulder_x_ratio": abs(elbow[0] - shoulder[0]) / torso_height,
        "elbow_drop_ratio": (elbow[1] - shoulder[1]) / torso_height,
        "wrist_shoulder_ratio": _distance(wrist, shoulder) / torso_height,
        "wrist_shoulder_x_ratio": abs(wrist[0] - shoulder[0]) / torso_height,
        "wrist_elbow_x_ratio": abs(wrist[0] - elbow[0]) / torso_height,
        "visibility": min(
            _landmark_visibility(landmarks, shoulder_lmk),
            _landmark_visibility(landmarks, elbow_lmk),
            _landmark_visibility(landmarks, wrist_lmk),
            _landmark_visibility(landmarks, hip_lmk),
        ),
    }


def _build_deadlift_side_metrics(
    landmarks,
    torso_height,
    shoulder_lmk,
    hip_lmk,
    knee_lmk,
    ankle_lmk,
):
    shoulder = _landmark_xy(landmarks, shoulder_lmk)
    hip = _landmark_xy(landmarks, hip_lmk)
    knee = _landmark_xy(landmarks, knee_lmk)
    ankle = _landmark_xy(landmarks, ankle_lmk)

    shoulder_depth = _landmark_xyz(landmarks, shoulder_lmk)[2]
    hip_depth = _landmark_xyz(landmarks, hip_lmk)[2]
    knee_depth = _landmark_xyz(landmarks, knee_lmk)[2]
    ankle_depth = _landmark_xyz(landmarks, ankle_lmk)[2]

    return {
        "hip_angle": calculate_joint_angle(shoulder, hip, knee),
        "knee_angle": calculate_joint_angle(hip, knee, ankle),
        "torso_lean_angle": _torso_lean_angle(shoulder, hip),
        "shoulder_hip_x_ratio": abs(shoulder[0] - hip[0]) / torso_height,
        "knee_ankle_x_ratio": abs(knee[0] - ankle[0]) / torso_height,
        "visibility": min(
            _landmark_visibility(landmarks, shoulder_lmk),
            _landmark_visibility(landmarks, hip_lmk),
            _landmark_visibility(landmarks, knee_lmk),
            _landmark_visibility(landmarks, ankle_lmk),
        ),
        "depth_mean": (shoulder_depth + hip_depth + knee_depth + ankle_depth) / 4.0,
    }


def evaluate_bicep_curl_form_gate(landmarks, cfg, active_curl_arm=None):
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
        left_shoulder_xyz = _landmark_xyz(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder_xyz = _landmark_xyz(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip_xyz = _landmark_xyz(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        right_hip_xyz = _landmark_xyz(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
        nose = _landmark_xy(landmarks, mp_pose.PoseLandmark.NOSE)

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
            "left": _build_bicep_arm_metrics(
                landmarks,
                torso_height,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.LEFT_HIP,
            ),
            "right": _build_bicep_arm_metrics(
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

        if active_curl_arm in valid_arms:
            active_arm = active_curl_arm
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
        elif active_curl_arm in startup_valid_arms:
            startup_active_arm = active_curl_arm
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
        }, active_arm
    except Exception as exc:
        logger.debug("Unable to evaluate bicep-curl form gate: %s", exc)
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
        }, None


def evaluate_deadlift_form_gate(landmarks, cfg, active_body_side=None):
    gate_cfg = cfg.get("form_gate", {})

    front_shoulder_to_torso_ratio = float(
        gate_cfg.get("front_shoulder_to_torso_ratio", 0.78)
    )
    front_hip_to_torso_ratio = float(gate_cfg.get("front_hip_to_torso_ratio", 0.62))
    max_centered_nose_offset_ratio = float(
        gate_cfg.get("max_centered_nose_offset_ratio", 0.20)
    )
    max_front_shoulder_depth_delta = float(
        gate_cfg.get("max_front_shoulder_depth_delta", 0.028)
    )
    max_front_hip_depth_delta = float(gate_cfg.get("max_front_hip_depth_delta", 0.024))
    min_side_shoulder_depth_delta = float(
        gate_cfg.get("min_side_shoulder_depth_delta", 0.040)
    )
    min_side_hip_depth_delta = float(gate_cfg.get("min_side_hip_depth_delta", 0.032))
    max_side_shoulder_to_torso_ratio = float(
        gate_cfg.get("max_side_shoulder_to_torso_ratio", 0.74)
    )
    min_side_visibility = float(gate_cfg.get("min_side_visibility", 0.42))
    max_tracking_torso_lean = float(gate_cfg.get("max_tracking_torso_lean", 78))
    min_tracking_hip_angle = float(gate_cfg.get("min_tracking_hip_angle", 60))
    min_tracking_knee_angle = float(gate_cfg.get("min_tracking_knee_angle", 65))

    top_hip_angle_min = float(gate_cfg.get("top_hip_angle_min", 158))
    top_knee_angle_min = float(gate_cfg.get("top_knee_angle_min", 154))
    max_top_torso_lean = float(gate_cfg.get("max_top_torso_lean", 30))
    startup_top_hip_angle_min = float(
        gate_cfg.get("startup_top_hip_angle_min", top_hip_angle_min)
    )
    startup_top_knee_angle_min = float(
        gate_cfg.get("startup_top_knee_angle_min", top_knee_angle_min)
    )
    startup_max_top_torso_lean = float(
        gate_cfg.get("startup_max_top_torso_lean", max_top_torso_lean)
    )

    bottom_hip_range = gate_cfg.get("bottom_hip_angle_range", (74, 126))
    if not (isinstance(bottom_hip_range, (list, tuple)) and len(bottom_hip_range) == 2):
        bottom_hip_range = (74, 126)
    bottom_knee_range = gate_cfg.get("bottom_knee_angle_range", (78, 145))
    if not (isinstance(bottom_knee_range, (list, tuple)) and len(bottom_knee_range) == 2):
        bottom_knee_range = (78, 145)
    bottom_torso_lean_range = gate_cfg.get("bottom_torso_lean_range", (28, 74))
    if not (
        isinstance(bottom_torso_lean_range, (list, tuple))
        and len(bottom_torso_lean_range) == 2
    ):
        bottom_torso_lean_range = (28, 74)
    bottom_hip_min, bottom_hip_max = float(bottom_hip_range[0]), float(bottom_hip_range[1])
    bottom_knee_min, bottom_knee_max = float(bottom_knee_range[0]), float(bottom_knee_range[1])
    bottom_torso_min, bottom_torso_max = float(bottom_torso_lean_range[0]), float(
        bottom_torso_lean_range[1]
    )

    try:
        left_shoulder_xyz = _landmark_xyz(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder_xyz = _landmark_xyz(
            landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER
        )
        left_hip_xyz = _landmark_xyz(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        right_hip_xyz = _landmark_xyz(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
        nose = _landmark_xy(landmarks, mp_pose.PoseLandmark.NOSE)

        left_shoulder = left_shoulder_xyz[:2]
        right_shoulder = right_shoulder_xyz[:2]
        left_hip = left_hip_xyz[:2]
        right_hip = right_hip_xyz[:2]

        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
        hip_mid_y = (left_hip[1] + right_hip[1]) / 2.0
        torso_height = max(1e-6, abs(shoulder_mid_y - hip_mid_y))

        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        hip_width = abs(left_hip[0] - right_hip[0])
        shoulder_to_torso_ratio = shoulder_width / torso_height
        hip_to_torso_ratio = hip_width / torso_height

        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
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

        side_metrics = {
            "left": _build_deadlift_side_metrics(
                landmarks,
                torso_height,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE,
            ),
            "right": _build_deadlift_side_metrics(
                landmarks,
                torso_height,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
            ),
        }

        valid_sides = []
        startup_valid_sides = []
        for side, metrics in side_metrics.items():
            side_ok = (
                metrics["visibility"] >= min_side_visibility
                and metrics["torso_lean_angle"] <= max_tracking_torso_lean
                and metrics["hip_angle"] >= min_tracking_hip_angle
                and metrics["knee_angle"] >= min_tracking_knee_angle
            )
            startup_side_ok = (
                metrics["visibility"] >= max(0.25, min_side_visibility - 0.08)
                and metrics["torso_lean_angle"] <= (max_tracking_torso_lean + 6)
                and metrics["hip_angle"] >= (min_tracking_hip_angle - 8)
                and metrics["knee_angle"] >= (min_tracking_knee_angle - 8)
            )
            metrics["side_ok"] = side_ok
            metrics["startup_side_ok"] = startup_side_ok
            if side_ok:
                valid_sides.append(side)
            if startup_side_ok:
                startup_valid_sides.append(side)

        if active_body_side in valid_sides:
            active_side = active_body_side
        elif valid_sides:
            active_side = max(
                valid_sides,
                key=lambda side: (
                    side_metrics[side]["visibility"],
                    -side_metrics[side]["depth_mean"],
                ),
            )
        else:
            active_side = None

        if active_side in startup_valid_sides:
            startup_active_side = active_side
        elif active_body_side in startup_valid_sides:
            startup_active_side = active_body_side
        elif startup_valid_sides:
            startup_active_side = max(
                startup_valid_sides,
                key=lambda side: (
                    side_metrics[side]["visibility"],
                    -side_metrics[side]["depth_mean"],
                ),
            )
        else:
            startup_active_side = None

        active_metrics = side_metrics.get(active_side) if active_side else None
        startup_active_metrics = (
            side_metrics.get(startup_active_side) if startup_active_side else None
        )

        top_position_ok = bool(
            active_metrics
            and active_metrics["hip_angle"] >= top_hip_angle_min
            and active_metrics["knee_angle"] >= top_knee_angle_min
            and active_metrics["torso_lean_angle"] <= max_top_torso_lean
        )
        startup_top_position_ok = bool(
            startup_active_metrics
            and startup_active_metrics["hip_angle"] >= startup_top_hip_angle_min
            and startup_active_metrics["knee_angle"] >= startup_top_knee_angle_min
            and startup_active_metrics["torso_lean_angle"] <= startup_max_top_torso_lean
        )
        bottom_position_ok = bool(
            active_metrics
            and bottom_hip_min <= active_metrics["hip_angle"] <= bottom_hip_max
            and bottom_knee_min <= active_metrics["knee_angle"] <= bottom_knee_max
            and bottom_torso_min
            <= active_metrics["torso_lean_angle"]
            <= bottom_torso_max
        )

        form_ok = (
            side_pose_detected
            and not front_facing
            and active_metrics is not None
            and active_metrics.get("side_ok", False)
        )
        startup_form_ok = (
            side_pose_detected
            and not front_facing
            and startup_active_metrics is not None
            and startup_active_metrics.get("startup_side_ok", False)
        )

        return {
            "form_ok": form_ok,
            "startup_form_ok": startup_form_ok,
            "front_facing": front_facing,
            "side_pose_detected": side_pose_detected,
            "side_pose_votes": side_pose_votes,
            "active_side": active_side,
            "rep_angle": active_metrics["hip_angle"] if active_metrics else None,
            "startup_active_side": startup_active_side,
            "startup_rep_angle": (
                startup_active_metrics["hip_angle"] if startup_active_metrics else None
            ),
            "top_position_ok": top_position_ok,
            "startup_top_position_ok": startup_top_position_ok,
            "bottom_position_ok": bottom_position_ok,
            "active_knee_angle": active_metrics["knee_angle"] if active_metrics else None,
            "startup_knee_angle": (
                startup_active_metrics["knee_angle"] if startup_active_metrics else None
            ),
            "active_torso_lean_angle": (
                active_metrics["torso_lean_angle"] if active_metrics else None
            ),
            "startup_torso_lean_angle": (
                startup_active_metrics["torso_lean_angle"]
                if startup_active_metrics
                else None
            ),
            "shoulder_to_torso_ratio": shoulder_to_torso_ratio,
            "hip_to_torso_ratio": hip_to_torso_ratio,
            "nose_offset_ratio": nose_offset_ratio,
            "shoulder_depth_delta": shoulder_depth_delta,
            "hip_depth_delta": hip_depth_delta,
            "valid_sides": valid_sides,
            "startup_valid_sides": startup_valid_sides,
            "side_metrics": side_metrics,
        }, active_side
    except Exception as exc:
        logger.debug("Unable to evaluate deadlift form gate: %s", exc)
        return {
            "form_ok": False,
            "startup_form_ok": False,
            "front_facing": False,
            "side_pose_detected": False,
            "side_pose_votes": 0,
            "active_side": None,
            "rep_angle": None,
            "startup_active_side": None,
            "startup_rep_angle": None,
            "top_position_ok": False,
            "startup_top_position_ok": False,
            "bottom_position_ok": False,
            "active_knee_angle": None,
            "startup_knee_angle": None,
            "active_torso_lean_angle": None,
            "startup_torso_lean_angle": None,
            "shoulder_to_torso_ratio": None,
            "hip_to_torso_ratio": None,
            "nose_offset_ratio": None,
            "shoulder_depth_delta": None,
            "hip_depth_delta": None,
            "valid_sides": [],
            "startup_valid_sides": [],
            "side_metrics": {},
        }, None
