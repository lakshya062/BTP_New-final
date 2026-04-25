# config.py

import mediapipe as mp

mp_pose = mp.solutions.pose

exercise_config = {
    'bicep_curl': {
        'display_name': 'Bicep Curl',
        'angles': {
            'left_elbow': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
            'right_elbow': [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST],
        },
        'rep_angle_keys': ['left_elbow', 'right_elbow'],
        'down_range': (150, 170),  # Fully extended
        'up_range': (35, 55),      # Fully contracted
        'reps_per_set': 12,
        'bend_detection': True,
        'form_gate': {
            # Enforce side-ish orientation (not front-facing) before counting curls.
            'front_shoulder_to_torso_ratio': 0.78,
            'front_hip_to_torso_ratio': 0.62,
            'max_centered_nose_offset_ratio': 0.18,
            'max_front_shoulder_depth_delta': 0.025,
            'max_front_hip_depth_delta': 0.02,
            'min_side_shoulder_depth_delta': 0.045,
            'min_side_hip_depth_delta': 0.035,
            'max_side_shoulder_to_torso_ratio': 0.72,
            'required_side_pose_frames': 8,
            # Keep upper arm close to torso (strict curl mechanics).
            'max_underarm_angle': 48,
            'max_elbow_hip_distance_ratio': 0.58,
            'max_elbow_shoulder_x_ratio': 0.32,
            'min_elbow_drop_ratio': 0.12,
            # Looser startup gate so "ready" is easier to achieve before first rep.
            'startup_required_side_pose_frames': 4,
            'startup_stable_frames_required': 14,
            'startup_down_range': (140, 178),
            'startup_max_underarm_angle': 62,
            'startup_max_elbow_hip_distance_ratio': 0.70,
            'startup_max_elbow_shoulder_x_ratio': 0.40,
            'startup_min_elbow_drop_ratio': 0.06,
            # At top of rep, wrist should approach shoulder (true curl path).
            'max_wrist_to_shoulder_ratio_at_top': 0.62,
            'max_top_wrist_shoulder_x_ratio': 0.26,
            'min_top_wrist_to_shoulder_delta': 0.08,
            'min_arm_visibility': 0.35,
            'min_valid_arms': 1,
        },
    },
    'deadlift': {
        'display_name': 'Deadlift',
        'angles': {
            'left_hip': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE],
            'right_hip': [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE],
            'left_knee': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE],
            'right_knee': [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE],
        },
        'rep_angle_keys': ['left_hip', 'right_hip'],
        'down_range': (158, 180),   # Standing lockout
        'up_range': (78, 122),      # Bottom hinge
        'reps_per_set': 8,
        'bend_detection': False,
        'form_gate': {
            # Side-angle enforcement.
            'front_shoulder_to_torso_ratio': 0.78,
            'front_hip_to_torso_ratio': 0.62,
            'max_centered_nose_offset_ratio': 0.20,
            'max_front_shoulder_depth_delta': 0.028,
            'max_front_hip_depth_delta': 0.024,
            'min_side_shoulder_depth_delta': 0.040,
            'min_side_hip_depth_delta': 0.032,
            'max_side_shoulder_to_torso_ratio': 0.74,
            'required_side_pose_frames': 6,
            'startup_required_side_pose_frames': 3,
            # Start position: tall lockout before first rep.
            'startup_stable_frames_required': 12,
            'startup_down_range': (150, 180),
            # Tracking quality and movement constraints.
            'min_side_visibility': 0.42,
            'max_tracking_torso_lean': 78,
            'min_tracking_hip_angle': 60,
            'min_tracking_knee_angle': 65,
            # Top lockout validation.
            'top_hip_angle_min': 158,
            'top_knee_angle_min': 154,
            'max_top_torso_lean': 30,
            'startup_top_hip_angle_min': 150,
            'startup_top_knee_angle_min': 148,
            'startup_max_top_torso_lean': 34,
            # Bottom hinge validation.
            'bottom_hip_angle_range': (74, 126),
            'bottom_knee_angle_range': (78, 145),
            'bottom_torso_lean_range': (28, 74),
        },
    },
    'shoulder_press': {
        'display_name': 'Shoulder Press',
        'angles': {
            'left_elbow': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
            'right_elbow': [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST],
            'left_underarm': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW],
            'right_underarm': [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW],
        },
        'rep_angle_keys': ['left_elbow', 'right_elbow'],
        'down_range': (75, 105),   # Dumbbells/bar at shoulder level
        'up_range': (158, 180),    # Arms locked out overhead
        'reps_per_set': 10,
        'bend_detection': True,
        'range_labels': {
            'up': 'OVERHEAD',
            'down': 'SHOULDERS',
        },
        'coaching': {
            'start': 'Start at shoulder height, brace your core, and press overhead.',
            'up_miss': 'Press fully overhead before lowering.',
            'down_miss': 'Return to shoulder height before the next rep.',
            'phase_up': 'Drive the weight overhead without leaning back.',
            'phase_down': 'Lower under control to shoulder height.',
        },
    },
    'seated_shoulder_press': {
        'display_name': 'Seated Shoulder Press',
        'alias_for': 'shoulder_press',
        'angles': {
            'left_elbow': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
            'right_elbow': [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST],
            'left_underarm': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW],
            'right_underarm': [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW],
        },
        'rep_angle_keys': ['left_elbow', 'right_elbow'],
        'down_range': (75, 105),
        'up_range': (158, 180),
        'reps_per_set': 10,
        'bend_detection': True,
        'range_labels': {
            'up': 'OVERHEAD',
            'down': 'SHOULDERS',
        },
        'coaching': {
            'start': 'Start at shoulder height, brace your core, and press overhead.',
            'up_miss': 'Press fully overhead before lowering.',
            'down_miss': 'Return to shoulder height before the next rep.',
            'phase_up': 'Drive the weight overhead without leaning back.',
            'phase_down': 'Lower under control to shoulder height.',
        },
    },
    'squat': {
        'display_name': 'Squat',
        'angles': {
            'left_hip': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE],
            'right_hip': [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE],
            'left_knee': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE],
            'right_knee': [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE],
        },
        'rep_angle_keys': ['left_knee', 'right_knee'],
        'down_range': (158, 180),  # Standing tall before/after each rep
        'up_range': (72, 112),     # Bottom squat depth
        'reps_per_set': 10,
        'bend_detection': False,
        'range_labels': {
            'up': 'DEPTH',
            'down': 'STAND',
        },
        'coaching': {
            'start': 'Stand tall, brace, then sit into the squat with control.',
            'up_miss': 'Reach squat depth before standing up.',
            'down_miss': 'Stand tall before starting the next rep.',
            'phase_up': 'Drive through the floor and stand tall.',
            'phase_down': 'Sit down and back, knees tracking over toes.',
        },
    },
    'lateral_raises': {
        'display_name': 'Lateral Raises',
        'angles': {
            'left_shoulder_elevation': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW],
            'right_shoulder_elevation': [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW],
        },
        'rep_angle_keys': ['left_shoulder_elevation', 'right_shoulder_elevation'],
        'down_range': (70, 90),    # Arms down
        'up_range': (130, 150),    # Arms lifted to side
        'reps_per_set': 15,
        'bend_detection': True,
        'range_labels': {
            'up': 'RAISED',
            'down': 'LOWERED',
        },
    }
}


exercise_template = {
    'new_exercise_key': {
        'display_name': 'New Exercise',
        'angles': {
            # angle_name: [first_landmark, center_joint_landmark, third_landmark]
            # Example:
            # 'left_knee': [
            #     mp_pose.PoseLandmark.LEFT_HIP,
            #     mp_pose.PoseLandmark.LEFT_KNEE,
            #     mp_pose.PoseLandmark.LEFT_ANKLE,
            # ],
        },
        'rep_angle_keys': [],
        'down_range': (150, 180),
        'up_range': (60, 110),
        'reps_per_set': 10,
        'bend_detection': False,
        'range_labels': {
            'up': 'TOP',
            'down': 'START',
        },
        'coaching': {
            'start': 'Set your start position and move with control.',
            'up_miss': 'Reach the target range before returning.',
            'down_miss': 'Return to the start range before the next rep.',
            'phase_up': 'Move into the working range.',
            'phase_down': 'Return with control.',
        },
    }
}


def get_exercise_display_name(exercise_key):
    cfg = exercise_config.get(exercise_key, {})
    return cfg.get('display_name') or exercise_key.replace('_', ' ').title()


def selectable_exercises():
    return [
        key
        for key, cfg in exercise_config.items()
        if not cfg.get('alias_for') and not cfg.get('hidden')
    ]
