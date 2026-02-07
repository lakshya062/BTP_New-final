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
        'bend_detection': True
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
