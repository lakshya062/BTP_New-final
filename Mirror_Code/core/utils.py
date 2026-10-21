"""Geometry helpers for pose analysis.

Invalid or coincident landmarks must not look like valid zero-degree joints.
Callers can catch ValueError and reject the observation instead of counting it.
"""

from math import atan2, degrees, hypot, isfinite


def _point_xy(point):
    try:
        if len(point) != 2:
            raise ValueError("Expected a two-dimensional point")
        result = (float(point[0]), float(point[1]))
    except (TypeError, IndexError, OverflowError) as exc:
        raise ValueError("Expected a two-dimensional numeric point") from exc
    if not all(isfinite(value) for value in result):
        raise ValueError("Point coordinates must be finite")
    return result


def calculate_joint_angle(a, b, c):
    """Return the 0–180 degree angle at b, rejecting invalid/zero-length limbs."""
    a, b, c = _point_xy(a), _point_xy(b), _point_xy(c)
    first = (a[0] - b[0], a[1] - b[1])
    second = (c[0] - b[0], c[1] - b[1])
    first_length, second_length = hypot(*first), hypot(*second)
    if (
        not isfinite(first_length)
        or not isfinite(second_length)
        or first_length <= 1e-9
        or second_length <= 1e-9
    ):
        raise ValueError("Joint angle requires two nonzero, finite limb vectors")
    # atan2 avoids arccos rounding near a fully straight or contracted joint.
    angle = degrees(atan2(second[1], second[0]) - atan2(first[1], first[0]))
    return abs((angle + 180.0) % 360.0 - 180.0)


def calculate_bend_angle(a, b):
    """Return the signed angle from upright in image coordinates (y down)."""
    a, b = _point_xy(a), _point_xy(b)
    dx, dy = a[0] - b[0], a[1] - b[1]
    length = hypot(dx, dy)
    if not isfinite(length) or length <= 1e-9:
        raise ValueError("Bend angle requires distinct, finite shoulder and hip points")
    return degrees(atan2(-dx, -dy))


def is_within_range(value, target, tolerance):
    """Check a finite value against a finite target and nonnegative tolerance."""
    try:
        value, target, tolerance = float(value), float(target), float(tolerance)
        return (
            all(isfinite(item) for item in (value, target, tolerance))
            and tolerance >= 0.0
            and target - tolerance <= value <= target + tolerance
        )
    except (TypeError, ValueError, OverflowError):
        return False
