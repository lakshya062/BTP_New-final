import unittest

from core.config import exercise_config
from core.utils import calculate_joint_angle


class ExerciseConfigurationGeometryTests(unittest.TestCase):
    def test_lateral_raise_targets_match_torso_and_horizontal_arm(self):
        config = exercise_config["lateral_raises"]
        hip = (0.5, 0.8)
        shoulder = (0.5, 0.4)
        for arm_x in (0.2, 0.8):
            elbow_down = (0.5, 0.65)
            elbow_raised = (arm_x, 0.4)
            for elbow, target in (
                (elbow_down, config["down_range"]),
                (elbow_raised, config["up_range"]),
            ):
                with self.subTest(arm_x=arm_x, target=target):
                    angle = calculate_joint_angle(hip, shoulder, elbow)
                    self.assertGreaterEqual(angle, target[0])
                    self.assertLessEqual(angle, target[1])


if __name__ == "__main__":
    unittest.main()
