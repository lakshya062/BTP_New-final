import math
import unittest

from core.utils import calculate_bend_angle, calculate_joint_angle, is_within_range


class PoseGeometryTest(unittest.TestCase):
    def test_joint_angle_measures_straight_right_and_contracted_limbs(self):
        self.assertAlmostEqual(calculate_joint_angle([0, -1], [0, 0], [0, 1]), 180)
        self.assertAlmostEqual(calculate_joint_angle([0, -1], [0, 0], [1, 0]), 90)
        self.assertAlmostEqual(calculate_joint_angle([0, -1], [0, 0], [0, -2]), 0)

    def test_joint_angle_is_invariant_to_scale_translation_and_order(self):
        angle = calculate_joint_angle([0, -1], [0, 0], [1, -1])
        self.assertAlmostEqual(angle, 45)
        self.assertAlmostEqual(angle, calculate_joint_angle([14, 6], [10, 10], [10, 6]))

    def test_invalid_joints_raise_instead_of_fabricating_zero_angles(self):
        for invalid in ([math.nan, 0], [math.inf, 0], [0], [0, 1, 2], None, ['bad', 0]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                calculate_joint_angle(invalid, [0, 0], [1, 0])
        for first, middle, last in (([0, 0], [0, 0], [1, 0]), ([1, 0], [0, 0], [0, 0])):
            with self.assertRaises(ValueError):
                calculate_joint_angle(first, middle, last)

    def test_bend_angle_has_consistent_image_axis_sign(self):
        self.assertAlmostEqual(calculate_bend_angle([0, -1], [0, 0]), 0)
        self.assertAlmostEqual(calculate_bend_angle([1, -1], [0, 0]), -45)
        self.assertAlmostEqual(calculate_bend_angle([-1, -1], [0, 0]), 45)

    def test_unmeasurable_bend_is_rejected(self):
        for shoulder in ([0, 0], [math.nan, -1], [0, math.inf]):
            with self.subTest(shoulder=shoulder), self.assertRaises(ValueError):
                calculate_bend_angle(shoulder, [0, 0])

    def test_range_check_rejects_nonfinite_and_negative_tolerance(self):
        self.assertTrue(is_within_range(10, 12, 2))
        self.assertFalse(is_within_range(math.inf, 0, math.inf))
        self.assertFalse(is_within_range(0, 0, -1))
        self.assertFalse(is_within_range('invalid', 0, 1))


if __name__ == '__main__':
    unittest.main()
