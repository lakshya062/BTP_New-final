import os
import unittest
from unittest.mock import patch

import numpy as np

from ui import camera_utils


class FakeCapture:
    def __init__(self, opened=True):
        self._opened = opened
        self.released = False
        self.set_calls = []
        self.values = {
            camera_utils.cv2.CAP_PROP_FRAME_WIDTH: 640,
            camera_utils.cv2.CAP_PROP_FRAME_HEIGHT: 480,
            camera_utils.cv2.CAP_PROP_FPS: 0.0,
            camera_utils.cv2.CAP_PROP_FOURCC: 0.0,
        }

    def isOpened(self):
        return self._opened and not self.released

    def set(self, prop_id, value):
        self.set_calls.append((prop_id, value))
        self.values[prop_id] = value
        return True

    def get(self, prop_id):
        return self.values.get(prop_id, 0.0)

    def release(self):
        self.released = True


class CameraCaptureUtilsTest(unittest.TestCase):
    def setUp(self):
        camera_utils._camera_cache.clear()

    def test_open_camera_capture_prefers_mjpeg_and_does_not_lock_fps_by_default(self):
        cap = FakeCapture(opened=True)
        expected_fourcc = camera_utils.cv2.VideoWriter_fourcc(*"MJPG")

        with patch.dict(
            os.environ,
            {
                "SMART_MIRROR_CAMERA_PREFER_MJPEG": "1",
                "SMART_MIRROR_CAMERA_WIDTH": "1920",
                "SMART_MIRROR_CAMERA_HEIGHT": "1080",
            },
            clear=False,
        ):
            with patch(
                "ui.camera_utils._camera_backend_candidates",
                return_value=[("AVFoundation", 1200)],
            ):
                with patch("ui.camera_utils.cv2.VideoCapture", return_value=cap) as video_capture:
                    result = camera_utils.open_camera_capture(0)

        self.assertIs(result, cap)
        video_capture.assert_called_once_with(0, 1200)
        self.assertIn((camera_utils.cv2.CAP_PROP_FOURCC, expected_fourcc), cap.set_calls)
        self.assertIn((camera_utils.cv2.CAP_PROP_FRAME_WIDTH, 1920), cap.set_calls)
        self.assertIn((camera_utils.cv2.CAP_PROP_FRAME_HEIGHT, 1080), cap.set_calls)
        self.assertFalse(
            any(prop_id == camera_utils.cv2.CAP_PROP_FPS for prop_id, _ in cap.set_calls)
        )

    def test_open_camera_capture_requests_wide_angle_zoom_when_supported(self):
        cap = FakeCapture(opened=True)

        with patch.dict(
            os.environ,
            {"SMART_MIRROR_CAMERA_WIDE_ANGLE": "1"},
            clear=False,
        ):
            with patch(
                "ui.camera_utils._camera_backend_candidates",
                return_value=[("AVFoundation", 1200)],
            ):
                with patch.object(camera_utils.cv2, "CAP_PROP_ZOOM", 27, create=True):
                    with patch("ui.camera_utils.cv2.VideoCapture", return_value=cap):
                        result = camera_utils.open_camera_capture(0)

        self.assertIs(result, cap)
        self.assertIn((27, 0.0), cap.set_calls)

    def test_open_camera_capture_falls_back_when_preferred_backend_cannot_open(self):
        preferred = FakeCapture(opened=False)
        fallback = FakeCapture(opened=True)

        def fake_video_capture(source, backend_code=None):
            if backend_code == 900:
                return preferred
            return fallback

        with patch(
            "ui.camera_utils._camera_backend_candidates",
            return_value=[("Preferred", 900), ("Default", None)],
        ):
            with patch("ui.camera_utils.cv2.VideoCapture", side_effect=fake_video_capture):
                result = camera_utils.open_camera_capture(2, apply_preferences=False)

        self.assertIs(result, fallback)
        self.assertTrue(preferred.released)
        self.assertFalse(fallback.released)

    def test_transform_camera_frame_supports_per_camera_rotation(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        frame[0, 0] = (10, 20, 30)

        with patch.dict(
            os.environ,
            {"SMART_MIRROR_CAMERA_ROTATION_CAM_2": "90"},
            clear=False,
        ):
            rotated = camera_utils.transform_camera_frame(frame, camera_source=2)

        self.assertEqual(rotated.shape[:2], (3, 2))


if __name__ == "__main__":
    unittest.main()
