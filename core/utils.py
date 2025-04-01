# core/utils.py

import numpy as np

def calculate_joint_angle(a, b, c):
    """
    Calculate the angle between three points for joint angles.
    a, b, c: Each is a list or array with two elements [x, y].
    """
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180:
            angle = 360 - angle
        return angle
    except Exception as e:
        print(f"Error calculating joint angle: {e}")
        return 0

def calculate_bend_angle(a, b):
    """
    Calculate the angle between the vector from b to a and the vertical axis.
    Positive angle indicates back bend, negative indicates front bend.
    """
    try:
        a = np.array(a)
        b = np.array(b)
        vector = a - b
        vertical = np.array([0, -1])  # Assuming y-axis points down

        # Normalize vectors
        if np.linalg.norm(vector) == 0:
            return 0
        vector_norm = vector / np.linalg.norm(vector)
        vertical_norm = vertical / np.linalg.norm(vertical)

        # Calculate dot product and angle
        dot_prod = np.dot(vector_norm, vertical_norm)
        angle_rad = np.arccos(np.clip(dot_prod, -1.0, 1.0))
        angle = np.degrees(angle_rad)

        # Determine direction (front or back bend)
        cross_prod = np.cross(vertical_norm, vector_norm)
        if cross_prod > 0:
            angle = -angle  # Front bend
        return angle
    except Exception as e:
        print(f"Error calculating bend angle: {e}")
        return 0

def is_within_range(value, target, tolerance):
    """
    Check if a value is within a specified tolerance of a target.
    """
    try:
        return target - tolerance <= value <= target + tolerance
    except Exception as e:
        print(f"Error in is_within_range: {e}")
        return False