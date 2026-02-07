# ui/exercise_page.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QGroupBox
)
from PySide6.QtGui import QPixmap, QImage, QFont, QIcon
from PySide6.QtCore import Qt, Slot, Signal
import numpy as np
from ui.worker import ExerciseWorker
from core.paths import resource_path

class ExercisePage(QWidget):
    status_message = Signal(str)
    counters_update = Signal(int, int)
    user_recognized_signal = Signal(dict)
    unknown_user_detected = Signal(object)
    new_user_registration_signal = Signal(str, bool)
    worker_started = Signal()

    def __init__(
        self,
        db_handler,
        camera_index,
        exercise_choice,
        face_recognizer,
        assigned_user_name=None,
        aruco_dict_type="DICT_5X5_100",
        parent=None,
    ):
        super().__init__(parent)
        self.db_handler = db_handler
        self.camera_index = camera_index
        self.exercise_choice = exercise_choice
        self.face_recognizer = face_recognizer
        self.assigned_user_name = assigned_user_name
        self.aruco_dict_type = aruco_dict_type

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.title_label = QLabel(f"Exercise: {self.exercise_choice.replace('_', ' ').title()}")
        self.title_label.setObjectName("exerciseTitle")
        self.title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label, alignment=Qt.AlignCenter)

        self.video_label = QLabel("Video Feed")
        self.video_label.setObjectName("exerciseVideo")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(1280, 720)

        self.layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        self.controls_group = QGroupBox("Controls")
        self.controls_group.setObjectName("exerciseControlsGroup")
        self.controls_layout = QHBoxLayout()
        self.controls_group.setLayout(self.controls_layout)

        self.start_button = QPushButton(QIcon(resource_path("icons", "start.png")), "Start")
        self.start_button.setToolTip("Start monitoring exercise")
        self.start_button.setMinimumHeight(42)
        self.stop_button = QPushButton(QIcon(resource_path("icons", "stop.png")), "Stop")
        self.stop_button.setToolTip("Stop monitoring exercise")
        self.stop_button.setMinimumHeight(42)
        self.stop_button.setEnabled(False)

        self.controls_layout.addWidget(self.start_button)
        self.controls_layout.addWidget(self.stop_button)
        self.controls_layout.addStretch()

        self.rep_label = QLabel("Reps: 0")
        self.set_label = QLabel("Sets: 0")
        self.rep_label.setObjectName("counterLabel")
        self.set_label.setObjectName("counterLabel")
        counter_font = QFont("Segoe UI", 12, QFont.Bold)
        self.rep_label.setFont(counter_font)
        self.set_label.setFont(counter_font)

        self.counters_layout = QHBoxLayout()
        self.counters_layout.addWidget(self.rep_label)
        self.counters_layout.addWidget(self.set_label)
        self.counters_layout.addStretch()

        self.controls_layout.addLayout(self.counters_layout)

        self.layout.addWidget(self.controls_group)
        self.layout.addStretch()

        self.worker = None

        self.start_button.clicked.connect(self.start_exercise)
        self.stop_button.clicked.connect(self.stop_exercise)

    def start_exercise(self):
        if not self.worker:
            self.worker = ExerciseWorker(
                db_handler=self.db_handler,
                camera_index=self.camera_index,
                exercise_choice=self.exercise_choice,
                face_recognizer=self.face_recognizer,
                assigned_user_name=self.assigned_user_name,
                aruco_dict_type=self.aruco_dict_type,
            )
            self.worker.frame_signal.connect(self.update_frame)
            self.worker.status_signal.connect(self.emit_status_message)
            self.worker.counters_signal.connect(self.emit_counters_update)
            self.worker.user_recognized_signal.connect(self.handle_user_recognized)
            self.worker.unknown_user_detected.connect(self.prompt_new_user_name)
            self.worker.new_user_registration_signal.connect(self.handle_new_user_registration_result)
            self.worker.data_updated.connect(self.on_data_updated)
            self.worker.started.connect(self.on_worker_started)

            self.worker.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_message.emit("Exercise monitoring started.")

    def stop_exercise(self):
        if self.worker:
            try:
                self.worker.frame_signal.disconnect(self.update_frame)
                self.worker.status_signal.disconnect(self.emit_status_message)
                self.worker.counters_signal.disconnect(self.emit_counters_update)
                self.worker.user_recognized_signal.disconnect(self.handle_user_recognized)
                self.worker.unknown_user_detected.disconnect(self.prompt_new_user_name)
                self.worker.new_user_registration_signal.disconnect(self.handle_new_user_registration_result)
                self.worker.data_updated.disconnect(self.on_data_updated)
                self.worker.started.disconnect(self.on_worker_started)
            except TypeError as e:
                self.status_message.emit(f"Error disconnecting signals: {e}")

            self.worker.request_stop()
            self.worker.wait()
            self.worker = None

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_message.emit("Exercise monitoring stopped.")

        blank_pixmap = QPixmap(self.video_label.size())
        blank_pixmap.fill(Qt.black)
        self.video_label.setPixmap(blank_pixmap)

    def on_worker_started(self):
        self.worker_started.emit()
        self.status_message.emit("Worker thread started.")

    def is_exercise_running(self):
        return self.worker is not None

    @Slot(object)
    def update_frame(self, frame):
        try:
            rgb_image = frame[..., ::-1].copy()
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            scaled_pix = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pix)
        except Exception as e:
            self.emit_status_message(f"Error updating frame: {e}")

    @Slot(str)
    def emit_status_message(self, message):
        self.status_message.emit(message)

    @Slot(int, int)
    def emit_counters_update(self, reps, sets):
        """
        This slot receives the rep and set counters from the Worker.
        It updates the local labels and also emits a signal so that
        the MainWindow (and hence SmartMirrorWindow) can use them.
        """
        self.counters_update.emit(reps, sets)  # <-- forwards to any external connections
        self.rep_label.setText(f"Reps: {reps}")
        self.set_label.setText(f"Sets: {sets}")

    @Slot(dict)
    def handle_user_recognized(self, user_info):
        self.user_recognized_signal.emit(user_info)

    @Slot()
    def prompt_new_user_name(self):
        self.unknown_user_detected.emit(self)

    @Slot()
    def on_data_updated(self):
        self.status_message.emit("Exercise data updated.")

    @Slot(str, bool)
    def handle_new_user_registration_result(self, username, success):
        self.new_user_registration_signal.emit(username, success)

    def start_user_registration(self, user_name):
        if self.worker:
            self.worker.start_record_new_user(user_name)

    def closeEvent(self, event):
        self.stop_exercise()
        event.accept()
