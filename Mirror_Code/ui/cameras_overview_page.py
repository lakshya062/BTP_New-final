# ui/cameras_overview_page.py (unchanged except grid_mode logic)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QGridLayout, QLabel, QFrame
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QImage
import cv2
import math
import logging

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

        label = QLabel(f"{exercise.replace('_', ' ').title()} (cam_{camera_index})")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #FFFFFF; font-size: 12pt; margin-top: 5px;")
        label.setFixedHeight(30)

        v_layout.addWidget(label_pixmap)
        v_layout.addWidget(label)

        self.thumbnails[key] = frame
        logging.info(f"Added thumbnail for cam_{camera_index}, exercise '{exercise}'.")
        self.relayout_thumbnails()

    def remove_camera_display(self, camera_index, exercise):
        key = (camera_index, exercise)
        if key in self.thumbnails:
            frame = self.thumbnails[key]
            self.grid_layout.removeWidget(frame)
            frame.deleteLater()
            del self.thumbnails[key]
            logging.info(f"Removed thumbnail for cam_{camera_index}, exercise '{exercise}'.")
            self.relayout_thumbnails()
        else:
            logging.warning(f"Attempted to remove non-existent thumbnail for cam_{camera_index}, exercise '{exercise}'.")

    def clear_thumbnails(self):
        for key, frame in list(self.thumbnails.items()):
            self.grid_layout.removeWidget(frame)
            frame.deleteLater()
            del self.thumbnails[key]
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

        for idx, ((ci, ex), frame) in enumerate(items):
            row = idx // cols
            col = idx % cols
            self.grid_layout.addWidget(frame, row, col)
            logging.debug(f"Placed cam_{ci}, exercise '{ex}' at row {row}, column {col}.")

    @Slot(object, int, str)
    def update_thumbnail(self, frame, camera_index, exercise):
        key = (camera_index, exercise)
        if key not in self.thumbnails:
            logging.warning(f"Received frame for unknown camera/exercise: cam_{camera_index}, exercise '{exercise}'.")
            return

        frame_widget = self.thumbnails[key]
        label_pixmap = frame_widget.findChild(QLabel)
        if label_pixmap:
            try:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg)
                scaled_pix = pix.scaled(label_pixmap.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label_pixmap.setPixmap(scaled_pix)
                logging.debug(f"Updated thumbnail for cam_{camera_index}, exercise '{exercise}'.")
            except Exception as e:
                logging.error(f"Error updating thumbnail for cam_{camera_index}, exercise '{exercise}': {e}")
        else:
            logging.error(f"Label pixmap not found in frame for cam_{camera_index}, exercise '{exercise}'.")