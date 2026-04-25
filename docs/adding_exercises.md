# Adding Exercises

Exercise options shown in the app come from `exercise_config` in
`Mirror_Code/core/config.py`. Add a new key there and it will appear in the
Add Exercise dialog unless the config has `hidden: True` or `alias_for`.

Skeleton:

```python
'new_exercise_key': {
    'display_name': 'New Exercise',
    'angles': {
        'left_joint': [
            mp_pose.PoseLandmark.LEFT_FIRST_POINT,
            mp_pose.PoseLandmark.LEFT_CENTER_JOINT,
            mp_pose.PoseLandmark.LEFT_THIRD_POINT,
        ],
        'right_joint': [
            mp_pose.PoseLandmark.RIGHT_FIRST_POINT,
            mp_pose.PoseLandmark.RIGHT_CENTER_JOINT,
            mp_pose.PoseLandmark.RIGHT_THIRD_POINT,
        ],
    },
    'rep_angle_keys': ['left_joint', 'right_joint'],
    'down_range': (150, 180),
    'up_range': (60, 110),
    'reps_per_set': 10,
    'bend_detection': False,
    'range_labels': {
        'up': 'TARGET',
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
```

For stricter movement validation, add a dedicated form gate in
`Mirror_Code/core/form_gates.py` and call it from `ExerciseAnalyzer` in
`Mirror_Code/core/pose_analysis.py`, following the bicep curl or deadlift
pattern.
