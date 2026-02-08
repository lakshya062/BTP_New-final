# config.py

import mediapipe as mp

mp_pose = mp.solutions.pose

exercise_config = {
    'bicep_curl': {
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
    'seated_shoulder_press': {
        'angles': {
            'left_elbow': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
            'right_elbow': [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST],
            'left_underarm': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW],
            'right_underarm': [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW],
        },
        'rep_angle_keys': ['left_elbow', 'right_elbow'],
        'down_range': (80, 100),   # Arms down
        'up_range': (160, 180),    # Arms up
        'reps_per_set': 10,
        'bend_detection': True
    },
    'lateral_raises': {
        'angles': {
            'left_shoulder_elevation': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW],
            'right_shoulder_elevation': [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW],
        },
        'rep_angle_keys': ['left_shoulder_elevation', 'right_shoulder_elevation'],
        'down_range': (70, 90),    # Arms down
        'up_range': (130, 150),    # Arms lifted to side
        'reps_per_set': 15,
        'bend_detection': True
    }
}
