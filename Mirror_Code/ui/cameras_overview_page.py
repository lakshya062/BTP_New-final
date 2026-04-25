# ui/cameras_overview_page.py (unchanged except grid_mode logic)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QGridLayout, QLabel, QFrame
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QImage
import math
import logging
import os
import time

from core.config import get_exercise_display_name

class CamerasOverviewPage(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Header
        self.header = QLabel("Cameras Overview")
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        self.layout.addWidget(self.header)

        # Scroll Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.inner_widget = QWidget()
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(15)
        self.inner_widget.setLayout(self.grid_layout)
        self.scroll_area.setWidget(self.inner_widget)
        self.layout.addWidget(self.scroll_area)

        self.thumbnails = {}
        self.grid_mode = 4  # Default
        self._last_thumbnail_update_ts = {}
        self._thumbnail_max_fps = max(
            1.0,
            float(os.getenv("SMART_MIRROR_OVERVIEW_THUMBNAIL_FPS", "8")),
        )
        self._thumbnail_interval_sec = 1.0 / self._thumbnail_max_fps

    def set_grid_mode(self, screens):
        self.grid_mode = screens
        self.relayout_thumbnails()

    def compute_rows_cols(self, count):
        if self.grid_mode == 2:
            rows = 1
            cols = 2
        elif self.grid_mode == 4:
            rows = 2
            cols = 2
        elif self.grid_mode == 8:
            rows = 2
            cols = 4
        elif self.grid_mode == 16:
            rows = 4
            cols = 4
        else:
            rows = int(math.ceil(math.sqrt(count)))
            cols = rows
        return rows, cols

    def add_camera_display(self, camera_index, exercise):
        key = (camera_index, exercise)
        if key in self.thumbnails:
            logging.warning(f"Attempted to add duplicate thumbnail for cam_{camera_index}, exercise '{exercise}'.")
            return

        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("background-color: #2E2E2E; border: 2px solid #007ACC; border-radius: 8px;")
        v_layout = QVBoxLayout()
        frame.setLayout(v_layout)

        label_pixmap = QLabel()
        label_pixmap.setAlignment(Qt.AlignCenter)
        label_pixmap.setFixedSize(320, 180)
        label_pixmap.setStyleSheet("border: 1px solid #555555; border-radius: 4px;")
        placeholder = QPixmap(label_pixmap.size())
        placeholder.fill(Qt.darkGray)
        label_pixmap.setPixmap(placeholder)

        label = QLabel(f"{get_exercise_display_name(exercise)} (cam_{camera_index})")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #FFFFFF; font-size: 12pt; margin-top: 5px;")
        label.setFixedHeight(30)

        v_layout.addWidget(label_pixmap)
        v_layout.addWidget(label)

        self.thumbnails[key] = {
            "frame": frame,
            "label_pixmap": label_pixmap,
            "label_text": label,
        }
        logging.info(f"Added thumbnail for cam_{camera_index}, exercise '{exercise}'.")
        self.relayout_thumbnails()

    def remove_camera_display(self, camera_index, exercise):
        key = (camera_index, exercise)
        if key in self.thumbnails:
            frame = self.thumbnails[key]["frame"]
            self.grid_layout.removeWidget(frame)
            frame.deleteLater()
            del self.thumbnails[key]
            self._last_thumbnail_update_ts.pop(key, None)
            logging.info(f"Removed thumbnail for cam_{camera_index}, exercise '{exercise}'.")
            self.relayout_thumbnails()
        else:
            logging.warning(f"Attempted to remove non-existent thumbnail for cam_{camera_index}, exercise '{exercise}'.")

    def clear_thumbnails(self):
        for key, frame in list(self.thumbnails.items()):
            self.grid_layout.removeWidget(frame["frame"])
            frame["frame"].deleteLater()
            del self.thumbnails[key]
            self._last_thumbnail_update_ts.pop(key, None)
            logging.info(f"Cleared thumbnail for cam_{key[0]}, exercise '{key[1]}'.")
        self.relayout_thumbnails()

    def relayout_thumbnails(self):
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)

        count = len(self.thumbnails)
        if count == 0:
            return

        rows, cols = self.compute_rows_cols(count)
        logging.debug(f"Relayout thumbnails to {rows} rows and {cols} columns.")

        items = list(self.thumbnails.items())

        for idx, ((ci, ex), frame_data) in enumerate(items):
            row = idx // cols
            col = idx % cols
            self.grid_layout.addWidget(frame_data["frame"], row, col)
            logging.debug(f"Placed cam_{ci}, exercise '{ex}' at row {row}, column {col}.")

    @Slot(object, int, str)
    def update_thumbnail(self, frame, camera_index, exercise):
        key = (camera_index, exercise)
        if key not in self.thumbnails:
            logging.warning(f"Received frame for unknown camera/exercise: cam_{camera_index}, exercise '{exercise}'.")
            return

        now = time.monotonic()
        last_ts = self._last_thumbnail_update_ts.get(key, 0.0)
        if (now - last_ts) < self._thumbnail_interval_sec:
            return
        self._last_thumbnail_update_ts[key] = now

        label_pixmap = self.thumbnails[key]["label_pixmap"]
        try:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pix = QPixmap.fromImage(qimg)
            scaled_pix = pix.scaled(
                label_pixmap.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )
            label_pixmap.setPixmap(scaled_pix)
            logging.debug(f"Updated thumbnail for cam_{camera_index}, exercise '{exercise}'.")
        except Exception as e:
            logging.error(f"Error updating thumbnail for cam_{camera_index}, exercise '{exercise}': {e}")
