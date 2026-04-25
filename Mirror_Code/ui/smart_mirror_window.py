import logging
import os
import time

from core.config import get_exercise_display_name

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QMainWindow, QSizePolicy, QVBoxLayout, QWidget
from screeninfo import get_monitors


class SmartMirrorWindow(QMainWindow):
    """
    Window for the HDMI smart mirror display. Frames are centered on a styled
    background while preserving camera aspect ratio.
    """

    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.camera_labels = {}
        self.setStyleSheet("background-color: #04070d;")

        self.exercise_counters = {}

        self.screen_width = 0
        self.screen_height = 0
        self.screen_x = 0
        self.screen_y = 0
        self.max_feed_width = 0
        self.max_feed_height = 0
        self._background_template = None
        self._last_render_ts = {}
        self._max_render_fps = max(
            1.0,
            float(os.getenv("SMART_MIRROR_SMART_WINDOW_FPS", "30")),
        )
        self._min_render_interval_sec = 1.0 / self._max_render_fps

    def set_displays(self, displays):
        _ = displays
        monitors = get_monitors()
        if len(monitors) < 2:
            logging.warning("Only one monitor detected. Falling back to primary display.")
            chosen_monitor = monitors[0]
        else:
            chosen_monitor = monitors[1]

        self.move(chosen_monitor.x, chosen_monitor.y)
        self.showFullScreen()

        self.screen_width = chosen_monitor.width
        self.screen_height = chosen_monitor.height
        self.screen_x = chosen_monitor.x
        self.screen_y = chosen_monitor.y

        self.max_feed_width = int(self.screen_width * 0.9)
        self.max_feed_height = int(self.screen_height * 0.86)
        self._invalidate_background_cache()

    def add_camera_display(self, camera_index, exercise):
        key = (camera_index, exercise)
        if key not in self.camera_labels:
            label = QLabel(self)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background-color: black;")
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.camera_labels[key] = label
            self.main_layout.addWidget(label)

    def remove_camera_display(self, camera_index, exercise):
        key = (camera_index, exercise)
        if key in self.camera_labels:
            label = self.camera_labels[key]
            self.main_layout.removeWidget(label)
            label.deleteLater()
            del self.camera_labels[key]
            self._last_render_ts.pop(key, None)

    def update_counters_for_exercise(self, camera_index, exercise, reps, sets):
        self.exercise_counters[(camera_index, exercise)] = (reps, sets)

    def _invalidate_background_cache(self):
        self._background_template = None

    def _build_background_canvas(self):
        if self._background_template is not None:
            return self._background_template
        h = max(1, self.screen_height)
        w = max(1, self.screen_width)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        top_color = np.array([14, 24, 38], dtype=np.float32)
        bottom_color = np.array([4, 7, 13], dtype=np.float32)
        blend = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        gradient_row = (top_color * (1.0 - blend) + bottom_color * blend).astype(np.uint8)
        canvas[:, :, 0] = gradient_row[:, 0:1]
        canvas[:, :, 1] = gradient_row[:, 1:2]
        canvas[:, :, 2] = gradient_row[:, 2:3]
        self._background_template = canvas
        return self._background_template

    def update_thumbnail(self, camera_index, exercise, frame):
        key = (camera_index, exercise)
        if key not in self.camera_labels:
            return

        label = self.camera_labels[key]
        if frame is None or frame.size == 0:
            return

        now = time.monotonic()
        last_ts = self._last_render_ts.get(key, 0.0)
        if (now - last_ts) < self._min_render_interval_sec:
            return
        self._last_render_ts[key] = now

        background = self._build_background_canvas()
        canvas = background.copy()
        frame_h, frame_w = frame.shape[:2]
        if frame_h <= 0 or frame_w <= 0:
            return

        max_w = self.max_feed_width or int(canvas.shape[1] * 0.9)
        max_h = self.max_feed_height or int(canvas.shape[0] * 0.86)
        scale = min(max_w / float(frame_w), max_h / float(frame_h))
        display_w = max(1, int(frame_w * scale))
        display_h = max(1, int(frame_h * scale))
        resized_frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)

        x1 = (canvas.shape[1] - display_w) // 2
        y1 = (canvas.shape[0] - display_h) // 2
        x2 = x1 + display_w
        y2 = y1 + display_h

        canvas[y1:y2, x1:x2] = resized_frame
        cv2.rectangle(
            canvas,
            (x1 - 2, y1 - 2),
            (x2 + 2, y2 + 2),
            (98, 186, 255),
            2,
            cv2.LINE_AA,
        )

        reps, sets_done = self.exercise_counters.get((camera_index, exercise), (0, 0))
        header_text = (
            f"cam_{camera_index} | {get_exercise_display_name(exercise)} | "
            f"Reps {reps} | Sets {sets_done}"
        )
        banner_margin = 24
        banner_h = 56
        banner_w = min(canvas.shape[1] - 2 * banner_margin, int(canvas.shape[1] * 0.58))
        banner_x2 = banner_margin + banner_w
        banner_y2 = banner_margin + banner_h
        banner_roi = canvas[banner_margin:banner_y2, banner_margin:banner_x2]
        if banner_roi.size:
            banner_overlay = banner_roi.copy()
            cv2.rectangle(
                banner_overlay,
                (0, 0),
                (banner_overlay.shape[1] - 1, banner_overlay.shape[0] - 1),
                (18, 30, 45),
                -1,
            )
            cv2.addWeighted(banner_overlay, 0.68, banner_roi, 0.32, 0, banner_roi)
        cv2.rectangle(
            canvas,
            (banner_margin, banner_margin),
            (banner_margin + banner_w, banner_margin + banner_h),
            (92, 166, 236),
            2,
        )
        cv2.putText(
            canvas,
            header_text,
            (banner_margin + 16, banner_margin + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (232, 242, 252),
            2,
            cv2.LINE_AA,
        )

        h, w, ch = canvas.shape
        bytes_per_line = ch * w
        qimg = QImage(canvas.data, w, h, bytes_per_line, QImage.Format_BGR888)
        label.setPixmap(QPixmap.fromImage(qimg))
