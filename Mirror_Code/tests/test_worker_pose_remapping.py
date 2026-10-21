import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.pose_analysis import _project_landmarks_for_geometry
from ui.worker import ExerciseWorker, mp_pose


def make_pose():
    points = [SimpleNamespace(x=0.5, y=0.2, z=0.0, visibility=1.0)
              for _ in range(len(mp_pose.PoseLandmark))]
    for side, x in (('LEFT', 0.4), ('RIGHT', 0.6)):
        for joint, y in (('SHOULDER', 0.3), ('ELBOW', 0.5), ('WRIST', 0.65),
                         ('HIP', 0.6), ('KNEE', 0.8), ('ANKLE', 0.95)):
            points[getattr(mp_pose.PoseLandmark, f'{side}_{joint}').value] = SimpleNamespace(
                x=x, y=y, z=0.0, visibility=1.0)
    return points


class WorkerPoseRemappingTest(unittest.TestCase):
    def setUp(self):
        self.worker = ExerciseWorker(MagicMock(), 0, face_recognizer=MagicMock())

    def test_roi_coordinates_and_depth_return_to_full_frame_units(self):
        # The same point in an 800x600 frame and a 400x300 crop at (100, 50).
        point = SimpleNamespace(x=0.75, y=0.5, z=-0.3, visibility=0.9)
        packet = SimpleNamespace(landmark=[point])
        self.worker._remap_pose_landmarks_to_frame(packet, (100, 50, 500, 350), (600, 800, 3))
        self.assertAlmostEqual(point.x, 0.5)
        self.assertAlmostEqual(point.y, 1 / 3)
        self.assertAlmostEqual(point.z, -0.15)
        self.assertEqual(point.visibility, 0.9)
        projected = _project_landmarks_for_geometry(packet.landmark, SimpleNamespace(shape=(600, 800, 3)))
        self.assertAlmostEqual(projected[0].x, 400 / 600)
        self.assertAlmostEqual(projected[0].y, 200 / 600)

    def test_full_frame_remapping_preserves_all_coordinate_units(self):
        point = SimpleNamespace(x=0.4, y=0.6, z=-0.2, visibility=0.9)
        packet = SimpleNamespace(landmark=[point])
        self.worker._remap_pose_landmarks_to_frame(packet, (0, 0, 800, 600), (600, 800, 3))
        self.assertEqual((point.x, point.y, point.z, point.visibility), (0.4, 0.6, -0.2, 0.9))

    def test_nonfinite_points_cannot_become_valid_border_landmarks(self):
        for attribute, value in (('x', math.nan), ('y', math.inf), ('z', math.nan),
                                 ('visibility', math.inf), ('visibility', 2.0)):
            with self.subTest(attribute=attribute, value=value):
                point = SimpleNamespace(x=0.4, y=0.6, z=-0.2, visibility=0.9)
                setattr(point, attribute, value)
                packet = SimpleNamespace(landmark=[point])
                self.worker._remap_pose_landmarks_to_frame(packet, (0, 0, 800, 600), (600, 800, 3))
                self.assertEqual(point.visibility, 0.0)
                if attribute in ('x', 'y', 'z'):
                    self.assertFalse(math.isfinite(getattr(point, attribute)))

    def test_out_of_frame_joints_keep_geometry_and_are_unusable(self):
        point = SimpleNamespace(x=-0.2, y=0.6, z=-0.2, visibility=0.9)
        packet = SimpleNamespace(landmark=[point])
        self.worker._remap_pose_landmarks_to_frame(packet, (0, 0, 800, 600), (600, 800, 3))
        self.assertEqual(point.x, -0.2)
        self.assertEqual(point.visibility, 0.0)

    def test_pose_reliability_requires_finite_body_joints_and_confidence(self):
        self.assertTrue(self.worker._has_reliable_pose(make_pose()))
        for attribute, value in (('x', math.nan), ('y', math.inf), ('z', math.nan),
                                 ('visibility', math.inf), ('visibility', 2.0)):
            with self.subTest(attribute=attribute, value=value):
                pose = make_pose()
                setattr(pose[mp_pose.PoseLandmark.LEFT_SHOULDER.value], attribute, value)
                self.assertFalse(self.worker._has_reliable_pose(pose))


if __name__ == '__main__':
    unittest.main()
