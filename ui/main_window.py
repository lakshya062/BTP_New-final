# ui/main_window.py

import json
import os
import sys
import uuid
from PySide6.QtWidgets import (
    QApplication, QMessageBox, QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QStatusBar, QPushButton, QHBoxLayout, QComboBox, QInputDialog, QDialog, QLabel, QSizePolicy
)
from PySide6.QtGui import QFont, QIcon, QGuiApplication, QPixmap, QImage
from PySide6.QtCore import Qt, QTimer, Slot, QSize
from screeninfo import get_monitors

import cv2
import logging
from datetime import datetime
from functools import partial

from core.database import DatabaseHandler
from core.face_recognition import FaceRecognizer
from core.aruco_detection import ArucoDetector
from core.paths import project_path, resource_path
from ui.exercise_page import ExercisePage
from ui.profile_page import ProfilePage
from ui.member_list_page import MemberListPage
from ui.cameras_overview_page import CamerasOverviewPage
from ui.add_exercise_dialog import AddExerciseDialog
from ui.add_member_dialog import AddMemberDialog
from ui.home_page import HomePage

def detect_available_cameras(max_cameras=10):
    available_cams = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cams.append(i)
        cap.release()
    return available_cams

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

    def set_displays(self, displays):
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
            lbl = self.camera_labels[key]
            self.main_layout.removeWidget(lbl)
            lbl.deleteLater()
            del self.camera_labels[key]

    def update_counters_for_exercise(self, camera_index, exercise, reps, sets):
        self.exercise_counters[(camera_index, exercise)] = (reps, sets)

    def _build_background_canvas(self):
        import numpy as np

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
        return canvas

    def update_thumbnail(self, camera_index, exercise, frame):
        key = (camera_index, exercise)
        if key not in self.camera_labels:
            return

        label = self.camera_labels[key]

        if frame is None or frame.size == 0:
            return

        canvas = self._build_background_canvas()

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
            f"cam_{camera_index} | {exercise.replace('_', ' ').title()} | "
            f"Reps {reps} | Sets {sets_done}"
        )
        overlay = canvas.copy()
        banner_margin = 24
        banner_h = 56
        banner_w = min(canvas.shape[1] - 2 * banner_margin, int(canvas.shape[1] * 0.58))
        cv2.rectangle(
            overlay,
            (banner_margin, banner_margin),
            (banner_margin + banner_w, banner_margin + banner_h),
            (18, 30, 45),
            -1,
        )
        cv2.addWeighted(overlay, 0.68, canvas, 0.32, 0, canvas)
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

        rgb_image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        label.setPixmap(pix)


class MainWindow(QMainWindow):
    CONFIG_FILE = project_path("config.json")

    def __init__(self):
        super().__init__()
        self.smart_mirror_window = None

        self.setWindowTitle("Smart Gym Client System")
        self.setMinimumSize(1600, 1000)
        font = QFont("Avenir Next", 11)
        self.setFont(font)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.db_handler = DatabaseHandler()
        self.global_face_recognizer = FaceRecognizer()
        if not self.global_face_recognizer.known_face_encodings:
            QMessageBox.warning(
                self,
                "Face Recognition",
                "No known faces loaded. All detections will be unknown until users are registered."
            )

        self.sync_face_model_with_db()

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setTabsClosable(False)
        self.tabs.setMovable(True)
        self.tabs.setIconSize(QSize(26, 26))

        self.home_page = HomePage(self.db_handler)
        self.profile_page = ProfilePage()
        self.member_list_page = MemberListPage(self.db_handler, self.global_face_recognizer)
        self.cameras_overview_page = CamerasOverviewPage()
        self.home_page.add_member_requested.connect(self.add_member_dialog)
        self.home_page.add_exercise_requested.connect(self.add_exercise_dialog)
        self.home_page.view_reports_requested.connect(lambda: self.tabs.setCurrentWidget(self.member_list_page))

        self.tabs.addTab(self.home_page, QIcon(resource_path("icons", "home.png")), "Home")
        self.tabs.addTab(self.profile_page, QIcon(resource_path("icons", "profile.png")), "Profile")
        self.tabs.addTab(self.member_list_page, QIcon(resource_path("icons", "members.png")), "Members")
        self.tabs.addTab(self.cameras_overview_page, QIcon(resource_path("icons", "cameras.png")), "Cameras Overview")

        self.main_layout.addWidget(self.tabs)

        self.controls_layout = QHBoxLayout()
        self.controls_layout.setSpacing(10)
        self.controls_layout.setContentsMargins(10, 10, 10, 10)

        self.add_exercise_button = QPushButton(QIcon(resource_path("icons", "add.png")), "Add Exercise")
        self.add_exercise_button.setToolTip("Add a new exercise with a camera")
        self.delete_exercise_button = QPushButton(QIcon(resource_path("icons", "delete.png")), "Delete Exercise")
        self.delete_exercise_button.setToolTip("Delete the selected exercise")
        self.start_all_button = QPushButton(QIcon(resource_path("icons", "start.png")), "Start All")
        self.start_all_button.setToolTip("Start all exercises")
        self.layout_selector = QComboBox()
        self.layout_selector.addItems(["2 Screens", "4 Screens", "8 Screens", "16 Screens"])
        self.layout_selector.setCurrentIndex(1)
        self.layout_selector.setToolTip("Select Camera Layout")

        self.controls_layout.addWidget(self.add_exercise_button)
        self.controls_layout.addWidget(self.delete_exercise_button)
        self.controls_layout.addWidget(self.start_all_button)
        self.controls_layout.addWidget(QLabel("Set Layout:"))
        self.controls_layout.addWidget(self.layout_selector)
        self.controls_layout.addStretch()

        self.main_layout.addLayout(self.controls_layout)

        self.add_exercise_button.clicked.connect(self.add_exercise_dialog)
        self.delete_exercise_button.clicked.connect(self.delete_current_exercise)
        self.start_all_button.clicked.connect(self.start_all_exercises)
        self.layout_selector.currentIndexChanged.connect(self.change_camera_layout)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.exercise_pages = []
        self._mirror_slots = {}
        self._overview_slots = {}
        self.aruco_dict_type = "DICT_5X5_100"
        self.load_config()
        self.update_overview_tab()

        self.sync_timer = QTimer(self)
        self.sync_timer.timeout.connect(self.sync_local_data_to_sqlite)
        self.sync_timer.start(60000)

    def sync_face_model_with_db(self):
        for name in sorted(set(self.global_face_recognizer.known_face_names)):
            member = self.db_handler.get_member_info(name)
            if not member:
                joined_on = datetime.utcnow().strftime('%Y-%m-%d')
                member_info = {
                    "user_id": str(uuid.uuid4()),
                    "username": name,
                    "email": "NA",
                    "membership": "NA",
                    "joined_on": joined_on
                }
                success = self.db_handler.insert_member_local(member_info)
                if success:
                    logging.info(f"Added {name} to local db from face model.")
                else:
                    logging.error(f"Failed to add {name} to local db.")

    def _connect_mirror_feed(self, exercise_page, camera_index, exercise):
        if exercise_page in self._mirror_slots:
            existing_worker, existing_slot = self._mirror_slots[exercise_page]
            if existing_worker is exercise_page.worker:
                return
            try:
                existing_worker.frame_signal.disconnect(existing_slot)
            except (TypeError, RuntimeError):
                pass
            del self._mirror_slots[exercise_page]
        if not exercise_page.worker:
            return

        slot = lambda frame, ci=camera_index, ex=exercise: self.smart_mirror_window.update_thumbnail(ci, ex, frame)
        exercise_page.worker.frame_signal.connect(slot)
        self._mirror_slots[exercise_page] = (exercise_page.worker, slot)

    def _disconnect_mirror_feed(self, exercise_page):
        if exercise_page not in self._mirror_slots:
            return
        worker, slot = self._mirror_slots[exercise_page]
        try:
            worker.frame_signal.disconnect(slot)
        except (TypeError, RuntimeError):
            pass
        del self._mirror_slots[exercise_page]

    def _connect_overview_feed(self, exercise_page, camera_index, exercise):
        if exercise_page in self._overview_slots:
            existing_worker, existing_slot = self._overview_slots[exercise_page]
            if existing_worker is exercise_page.worker:
                return
            try:
                existing_worker.thumbnail_frame_signal.disconnect(existing_slot)
            except (TypeError, RuntimeError):
                pass
            del self._overview_slots[exercise_page]
        if not exercise_page.worker:
            return

        slot = lambda frame, ci=camera_index, ex=exercise: self.cameras_overview_page.update_thumbnail(frame, ci, ex)
        exercise_page.worker.thumbnail_frame_signal.connect(slot)
        self._overview_slots[exercise_page] = (exercise_page.worker, slot)

    def _disconnect_overview_feed(self, exercise_page):
        if exercise_page not in self._overview_slots:
            return
        worker, slot = self._overview_slots[exercise_page]
        try:
            worker.thumbnail_frame_signal.disconnect(slot)
        except (TypeError, RuntimeError):
            pass
        del self._overview_slots[exercise_page]

    def load_config(self):
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, "r") as f:
                data = json.load(f)
            self.aruco_dict_type = ArucoDetector.normalize_dict_type(
                data.get("aruco_dict_type", "DICT_5X5_100")
            )
            exercises = data.get("exercises", [])
            for ex in exercises:
                camera_index = ex["camera_index"]
                exercise = ex["exercise"]
                user_name = ex.get("user_name", None)
                self.add_exercise_page(
                    camera_index,
                    exercise,
                    user_name=user_name,
                    start_immediately=False,
                    aruco_dict_type=self.aruco_dict_type,
                )
        else:
            with open(self.CONFIG_FILE, "w") as f:
                json.dump({"aruco_dict_type": self.aruco_dict_type, "exercises": []}, f, indent=4)

    def save_config(self):
        exercises = []
        for (page, cam_idx, ex, user) in self.exercise_pages:
            exercises.append({"camera_index": cam_idx, "exercise": ex, "user_name": user})
        data = {"aruco_dict_type": self.aruco_dict_type, "exercises": exercises}
        with open(self.CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)

    def add_exercise_dialog(self):
        available_cams = detect_available_cameras(max_cameras=10)
        assigned_cams = [cam for (_, cam, _, _) in self.exercise_pages]
        available_cams = [cam for cam in available_cams if cam not in assigned_cams]

        if not available_cams:
            QMessageBox.warning(
                self,
                "No Available Cameras",
                "All available cameras are already assigned to exercises."
            )
            return

        dialog = AddExerciseDialog(available_cams, self)
        if dialog.exec() == QDialog.Accepted:
            cam_text, exercise, user_name = dialog.get_selection()
            camera_index = int(cam_text.replace("cam_", ""))
            self.add_exercise_page(camera_index, exercise, user_name=user_name)
            self.save_config()

    def add_member_dialog(self):
        dialog = AddMemberDialog(self.db_handler, self)
        if dialog.exec() == QDialog.Accepted:
            self.member_list_page.load_members()
            self.home_page.refresh_summary()

    def add_exercise_page(
        self,
        camera_index,
        exercise,
        user_name=None,
        start_immediately=True,
        aruco_dict_type=None,
    ):
        resolved_aruco_dict_type = ArucoDetector.normalize_dict_type(
            aruco_dict_type or self.aruco_dict_type
        )
        self.aruco_dict_type = resolved_aruco_dict_type
        if user_name and user_name.strip():
            member = self.db_handler.get_member_info(user_name)
            if not member:
                joined_on = datetime.utcnow().strftime('%Y-%m-%d')
                member_info = {
                    "user_id": str(uuid.uuid4()),
                    "username": user_name,
                    "email": "NA",
                    "membership": "NA",
                    "joined_on": joined_on
                }
                success = self.db_handler.insert_member_local(member_info)
                if success:
                    self.status_bar.showMessage(f"Added {user_name} to local database.", 5000)
                else:
                    self.status_bar.showMessage(f"Failed to add {user_name} to local database.", 5000)

        exercise_page = ExercisePage(
            self.db_handler,
            camera_index,
            exercise,
            self.global_face_recognizer,
            assigned_user_name=user_name,
            aruco_dict_type=resolved_aruco_dict_type,
        )
        exercise_page.status_message.connect(self.update_status)
        exercise_page.counters_update.connect(
            partial(self.handle_counters_for_exercise, camera_index, exercise)
        )  # <-- ADDED FOR BLACK CANVAS DRAWING
        exercise_page.user_recognized_signal.connect(self.handle_user_recognized)
        exercise_page.unknown_user_detected.connect(self.prompt_new_user_name)
        exercise_page.new_user_registration_signal.connect(self.handle_new_user_registration_result)
        exercise_page.worker_started.connect(lambda: self.connect_data_updated_signal(exercise_page))

        tab_label = f"{exercise.replace('_', ' ').title()} (cam_{camera_index})"
        self.tabs.addTab(exercise_page, QIcon(resource_path("icons", "exercise.png")), tab_label)
        self.exercise_pages.append((exercise_page, camera_index, exercise, user_name))

        if start_immediately:
            exercise_page.start_exercise()
            # If we haven't opened the smart mirror window yet, do so now
            if self.smart_mirror_window is None:
                screens = QGuiApplication.screens()
                if len(screens) > 1:
                    self.smart_mirror_window = SmartMirrorWindow()
                    self.smart_mirror_window.set_displays(screens)
                    self.smart_mirror_window.add_camera_display(camera_index, exercise)
                    self._connect_mirror_feed(exercise_page, camera_index, exercise)
            else:
                # If it already exists, just add the camera label
                self.smart_mirror_window.add_camera_display(camera_index, exercise)
                self._connect_mirror_feed(exercise_page, camera_index, exercise)

            if self.smart_mirror_window and not self.smart_mirror_window.isVisible():
                self.smart_mirror_window.show()

        self.update_overview_tab()

    @Slot(int, str, int, int)
    def handle_counters_for_exercise(self, camera_index, exercise, reps, sets_):
        if self.smart_mirror_window:
            self.smart_mirror_window.update_counters_for_exercise(camera_index, exercise, reps, sets_)

    @Slot()
    def connect_data_updated_signal(self, exercise_page):
        if exercise_page.worker and not getattr(exercise_page.worker, "_data_updated_connected", False):
            exercise_page.worker.data_updated.connect(self.sync_local_data_to_sqlite)
            exercise_page.worker._data_updated_connected = True

    def prompt_new_user_name(self, exercise_page):
        username, ok = QInputDialog.getText(
            self, "New User Registration", "Enter username for the new user:"
        )
        if ok and username.strip():
            username = username.strip()
            exercise_page.start_user_registration(username)
            self.status_bar.showMessage(f"Capturing face data for user: {username}", 5000)
        else:
            self.status_bar.showMessage("User registration canceled.", 5000)
            QMessageBox.warning(self, "Registration Canceled", "User was not registered.")

    @Slot(str, bool)
    def handle_new_user_registration_result(self, username, success):
        if not success:
            self.status_bar.showMessage(f"Failed to register user: {username}", 5000)
            QMessageBox.warning(self, "Registration Failed", f"Could not register user: {username}")
            return

        existing_member = self.db_handler.get_member_info(username)
        if not existing_member:
            member_info = {
                "user_id": str(uuid.uuid4()),
                "username": username,
                "email": "NA",
                "membership": "NA",
                "joined_on": datetime.utcnow().strftime('%Y-%m-%d')
            }
            success = self.db_handler.insert_member_local(member_info)
            if not success:
                self.status_bar.showMessage(f"Face registered but failed to save user: {username}", 5000)
                QMessageBox.warning(self, "Registration Partial", f"Face saved, but member record failed: {username}")
                return

        self.status_bar.showMessage(f"Registered new user: {username}", 5000)
        self.member_list_page.load_members()
        self.home_page.refresh_summary()

    def handle_user_recognized(self, user_info):
        username = user_info.get("username", "Unknown")
        if username != "Unknown":
            member = self.db_handler.get_member_info(username)
            if not member:
                member_info = {
                    "user_id": str(uuid.uuid4()),
                    "username": username,
                    "email": "NA",
                    "membership": "NA",
                    "joined_on": datetime.utcnow().strftime('%Y-%m-%d')
                }
                success = self.db_handler.insert_member_local(member_info)
                if success:
                    self.status_bar.showMessage(f"Added {username} to local database.", 5000)
                    self.global_face_recognizer.reload_model()
                    self.member_list_page.load_members()
                else:
                    self.status_bar.showMessage(f"Failed to add {username} to local database.", 5000)
                    QMessageBox.warning(self, "Registration Failed", f"Could not register user: {username}")
            self.profile_page.update_profile(user_info)

    def delete_current_exercise(self):
        current_idx = self.tabs.currentIndex()
        if current_idx < 4:
            QMessageBox.information(self, "Cannot Delete", "Cannot delete default tabs.")
            return

        reply = QMessageBox.question(
            self, "Delete Exercise",
            "Are you sure you want to delete this exercise?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            widget = self.tabs.widget(current_idx)
            for i, (page, cam_idx, ex, user) in enumerate(self.exercise_pages):
                if page == widget:
                    if self.smart_mirror_window:
                        self.smart_mirror_window.remove_camera_display(cam_idx, ex)
                    self._disconnect_mirror_feed(page)
                    self._disconnect_overview_feed(page)
                    page.stop_exercise()
                    self.exercise_pages.pop(i)
                    break
            self.tabs.removeTab(current_idx)
            self.save_config()
            self.update_overview_tab()

    def start_all_exercises(self):
        available_cams = detect_available_cameras(max_cameras=10)
        missing = []
        for (page, cam_idx, ex, _) in self.exercise_pages:
            if cam_idx not in available_cams:
                missing.append((cam_idx, ex))

        if missing:
            msg = "Some cameras are not connected:\n"
            for (ci, ex) in missing:
                msg += f"cam_{ci} for exercise {ex.replace('_', ' ').title()}\n"
            msg += "The remaining cameras will be started."
            QMessageBox.warning(self, "Missing Cameras", msg)

        started_any = False
        for (page, cam_idx, ex, _) in self.exercise_pages:
            if cam_idx in available_cams:
                if not page.is_exercise_running():
                    page.start_exercise()
                    started_any = True

        if started_any:
            if self.smart_mirror_window is None:
                screens = QGuiApplication.screens()
                if len(screens) > 1:
                    self.smart_mirror_window = SmartMirrorWindow()
                    self.smart_mirror_window.set_displays(screens)
                    for (p, ci, exx, _) in self.exercise_pages:
                        if p.is_exercise_running():
                            self.smart_mirror_window.add_camera_display(ci, exx)
                            self._connect_mirror_feed(p, ci, exx)
            else:
                for (p, ci, exx, _) in self.exercise_pages:
                    if p.is_exercise_running():
                        self.smart_mirror_window.add_camera_display(ci, exx)
                        self._connect_mirror_feed(p, ci, exx)

            if self.smart_mirror_window and not self.smart_mirror_window.isVisible():
                self.smart_mirror_window.show()

        self.update_overview_tab()

    def change_camera_layout(self):
        selected = self.layout_selector.currentText()
        if selected == "2 Screens":
            self.cameras_overview_page.set_grid_mode(2)
        elif selected == "4 Screens":
            self.cameras_overview_page.set_grid_mode(4)
        elif selected == "8 Screens":
            self.cameras_overview_page.set_grid_mode(8)
        elif selected == "16 Screens":
            self.cameras_overview_page.set_grid_mode(16)
        else:
            self.cameras_overview_page.set_grid_mode(4)

    def update_overview_tab(self):
        self.cameras_overview_page.clear_thumbnails()
        active_pages = {page for (page, _, _, _) in self.exercise_pages}
        for page in list(self._overview_slots.keys()):
            if page not in active_pages or not page.worker:
                self._disconnect_overview_feed(page)

        for (page, cam_idx, ex, _) in self.exercise_pages:
            self.cameras_overview_page.add_camera_display(cam_idx, ex)
            if page.worker:
                self._connect_overview_feed(page, cam_idx, ex)

    def update_status(self, message):
        self.status_bar.showMessage(message, 5000)

    def update_counters(self, reps, sets):
        if self.smart_mirror_window:
            logging.debug("Counters updated reps=%s sets=%s", reps, sets)

    def sync_local_data_to_sqlite(self):
        try:
            self.home_page.refresh_summary()
            self.home_page.load_recent_activities()
            self.member_list_page.load_members()
        except Exception as exc:
            logging.error("Dashboard refresh failed: %s", exc)

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, 'Confirm Exit',
            "Are you sure you want to exit the application?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            for (page, cam_idx, ex, _) in self.exercise_pages[:]:
                self._disconnect_mirror_feed(page)
                self._disconnect_overview_feed(page)
                page.stop_exercise()
            self.db_handler.close_connections()
            self.global_face_recognizer.close()
            self.sync_local_data_to_sqlite()
            if self.smart_mirror_window:
                self.smart_mirror_window.close()
            event.accept()
        else:
            event.ignore()
