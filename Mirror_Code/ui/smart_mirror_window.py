import os
import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QMainWindow

from ui.display_layout import compose_mirror_frame, resolve_display_fit_mode


class SmartMirrorWindow(QMainWindow):
    """
    Full-screen portrait-friendly output window for a single analyzed camera feed.
    """

    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setStyleSheet("background-color: #050912;")

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.video_label)

        self.camera_index = None
        self.exercise = None
        self.screen = None
        self.screen_width = 1080
        self.screen_height = 1920
        self.overlay_data = {}
        self.feedback_texts = []

        self._last_render_ts = 0.0
        self._max_render_fps = max(
            1.0,
            float(os.getenv("SMART_MIRROR_SMART_WINDOW_FPS", "60")),
        )
        self._min_render_interval_sec = 1.0 / self._max_render_fps

    def configure_display(self, screen):
        self.screen = screen
        if screen is not None:
            geometry = screen.geometry()
            self.screen_width = max(1, int(geometry.width()))
            self.screen_height = max(1, int(geometry.height()))
            self.setGeometry(geometry)

        self.show()
        if self.windowHandle() is not None and screen is not None:
            self.windowHandle().setScreen(screen)
        self.showFullScreen()

    def set_assignment(self, camera_index, exercise):
        self.camera_index = camera_index
        self.exercise = exercise
        self.overlay_data = {
            "username": "Athlete",
            "exercise": str(exercise or "").replace("_", " ").title() or "Exercise",
            "rep_count": 0,
            "set_count": 0,
            "current_weight": 0,
            "reps_per_set": 0,
            "angles_line": "",
            "timestamp": "",
            "motivation": "",
        }
        self.feedback_texts = []

    def update_counters(self, reps, sets_done):
        self.overlay_data["rep_count"] = int(reps)
        self.overlay_data["set_count"] = int(sets_done)

    def update_overlay_state(self, overlay_data=None, feedback_texts=None):
        if overlay_data:
            merged = dict(self.overlay_data)
            merged.update(dict(overlay_data))
            self.overlay_data = merged
        if feedback_texts is not None:
            self.feedback_texts = list(feedback_texts or [])

    def update_frame(self, frame):
        if frame is None or getattr(frame, "size", 0) == 0:
            return

        now = time.monotonic()
        if (now - self._last_render_ts) < self._min_render_interval_sec:
            return
        self._last_render_ts = now

        canvas = compose_mirror_frame(
            frame=frame,
            canvas_width=self.screen_width,
            canvas_height=self.screen_height,
            overlay_data=self.overlay_data,
            feedback_texts=self.feedback_texts,
            fit_mode=resolve_display_fit_mode(),
        )

        height, width, channels = canvas.shape
        bytes_per_line = channels * width
        qimg = QImage(canvas.data, width, height, bytes_per_line, QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))
