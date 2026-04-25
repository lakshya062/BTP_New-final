import unittest
from unittest.mock import patch

import numpy as np

from ui.display_layout import (
    build_display_lines,
    compose_mirror_frame,
    fit_frame_to_region,
    resolve_display_fit_mode,
)


class DisplayLayoutTest(unittest.TestCase):
    def test_resolve_display_fit_mode_defaults_to_contain(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_display_fit_mode(), "contain")

    def test_fit_frame_to_region_cover_preserves_target_shape(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fitted = fit_frame_to_region(frame, target_width=1080, target_height=1344, fit_mode="cover")
        self.assertEqual(fitted.shape, (1344, 1080, 3))

    def test_fit_frame_to_region_contain_fills_background_panel(self):
        frame = np.full((1920, 1080, 3), 120, dtype=np.uint8)
        fitted = fit_frame_to_region(frame, target_width=1080, target_height=1344, fit_mode="contain")

        self.assertEqual(fitted.shape, (1344, 1080, 3))
        self.assertTrue(np.any(fitted[:, 0] > 0))
        self.assertTrue(np.any(fitted[:, -1] > 0))

    def test_build_display_lines_prioritizes_feedback_and_metadata(self):
        top_lines, bottom_lines = build_display_lines(
            overlay_data={
                "username": "Lakshya",
                "exercise": "Bicep Curl",
                "rep_count": 4,
                "set_count": 1,
                "reps_per_set": 12,
                "current_weight": 20,
                "angles_line": "Elbow: 82deg",
                "timestamp": "14 Apr 2026 | 09:30:00 AM",
                "motivation": "Stay tall and keep the curl strict.",
            },
            feedback_texts=["Drive through the top range."],
        )
        self.assertEqual(top_lines[0], "Lakshya | Bicep Curl")
        self.assertIn("Drive through the top range.", bottom_lines[0])
        self.assertIn("Elbow: 82deg", bottom_lines[1])

    def test_compose_mirror_frame_returns_full_canvas(self):
        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
        canvas = compose_mirror_frame(
            frame,
            canvas_width=1080,
            canvas_height=1920,
            overlay_data={"username": "Lakshya", "exercise": "Deadlift"},
            feedback_texts=["Chest up and lock out cleanly."],
        )
        self.assertEqual(canvas.shape, (1920, 1080, 3))


if __name__ == "__main__":
    unittest.main()
