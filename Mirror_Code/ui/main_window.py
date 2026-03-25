# ui/main_window.py

import json
import os
import socket
import time
import uuid
import getpass
from PySide6.QtWidgets import (
    QApplication, QMessageBox, QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QStatusBar, QPushButton, QHBoxLayout, QComboBox, QInputDialog, QDialog, QLabel
)
from PySide6.QtGui import QFont, QIcon, QGuiApplication
from PySide6.QtCore import QTimer, Slot, QSize

import logging
from datetime import datetime
from functools import partial

from core.database import DatabaseHandler
from core.face_recognition import FaceRecognizer
from core.aruco_detection import ArucoDetector
from core.config import exercise_config
from core.edge_device_manager import EdgeDeviceManager
from core.paths import project_path, resource_path
from core.server_sync import MainToServerSyncClient
from ui.exercise_page import ExercisePage
from ui.profile_page import ProfilePage
from ui.member_list_page import MemberListPage
from ui.cameras_overview_page import CamerasOverviewPage
from ui.add_exercise_dialog import AddExerciseDialog
from ui.add_device_dialog import AddDeviceDialog
from ui.add_member_dialog import AddMemberDialog
from ui.home_page import HomePage
from ui.register_user_dialog import RegisterUserDialog
from ui.background_workers import EdgeDeployWorker, EdgeStopWorker, SyncPipelineWorker
from ui.edge_utils import (
    detect_available_cameras,
    resolve_edge_camera_index,
    resolve_edge_pg_password,
    resolve_edge_pg_user,
)
from ui.smart_mirror_window import SmartMirrorWindow


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
        pg_default_user = os.getenv(
            "SMART_MIRROR_DEFAULT_PG_USER",
            os.getenv("PGUSER", getpass.getuser()),
        )
        pg_default_password = os.getenv("SMART_MIRROR_DEFAULT_PG_PASSWORD", "admin")
        runtime_edge_mode = os.environ.get("SMART_MIRROR_EDGE_MODE", "0") == "1"
        if runtime_edge_mode:
            os.environ.setdefault("SMART_MIRROR_DB_BACKEND", "postgres")
            os.environ.setdefault("SMART_MIRROR_NODE_ROLE", "EDGE")
            os.environ.setdefault("SMART_MIRROR_PG_DB", "gym_edge_db")
            os.environ.setdefault("SMART_MIRROR_PG_USER", pg_default_user)
            os.environ.setdefault("SMART_MIRROR_PG_PASSWORD", pg_default_password)
            os.environ.setdefault("SMART_MIRROR_PG_HOST", "localhost")
            os.environ.setdefault("SMART_MIRROR_PG_PORT", "5432")
            os.environ.setdefault("SMART_MIRROR_LOCAL_CAMERA_AS_EDGE", "0")
        else:
            os.environ.setdefault("SMART_MIRROR_DB_BACKEND", "postgres")
            os.environ.setdefault("SMART_MIRROR_NODE_ROLE", "MAIN")
            os.environ.setdefault("SMART_MIRROR_PG_DB", "gym_main_db")
            os.environ.setdefault("SMART_MIRROR_PG_USER", pg_default_user)
            os.environ.setdefault("SMART_MIRROR_PG_PASSWORD", pg_default_password)
            os.environ.setdefault("SMART_MIRROR_PG_HOST", "localhost")
            os.environ.setdefault("SMART_MIRROR_PG_PORT", "5432")
            os.environ.setdefault("SMART_MIRROR_LOCAL_CAMERA_AS_EDGE", "0")

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
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.main_layout.addWidget(self.tabs)

        self.controls_layout = QHBoxLayout()
        self.controls_layout.setSpacing(10)
        self.controls_layout.setContentsMargins(10, 10, 10, 10)

        self.add_device_button = QPushButton(QIcon(resource_path("icons", "add.png")), "Add Device")
        self.add_device_button.setToolTip("Discover and deploy code to an edge device")
        self.add_exercise_button = QPushButton(QIcon(resource_path("icons", "add.png")), "Add Exercise")
        self.add_exercise_button.setToolTip("Add a new exercise with a camera")
        self.delete_exercise_button = QPushButton(QIcon(resource_path("icons", "delete.png")), "Delete Exercise")
        self.delete_exercise_button.setToolTip("Delete the selected exercise")
        self.start_all_button = QPushButton(QIcon(resource_path("icons", "start.png")), "Start All")
        self.start_all_button.setToolTip("Start all exercises")
        self.stop_edge_button = QPushButton(QIcon(resource_path("icons", "stop.png")), "Stop")
        self.stop_edge_button.setToolTip("Stop runtime on a selected edge device")
        self.stop_all_edge_button = QPushButton(QIcon(resource_path("icons", "stop.png")), "Stop All")
        self.stop_all_edge_button.setToolTip("Stop runtime on all edge devices")
        self.layout_selector = QComboBox()
        self.layout_selector.addItems(["2 Screens", "4 Screens", "8 Screens", "16 Screens"])
        self.layout_selector.setCurrentIndex(1)
        self.layout_selector.setToolTip("Select Camera Layout")

        self.controls_layout.addWidget(self.add_device_button)
        self.controls_layout.addWidget(self.add_exercise_button)
        self.controls_layout.addWidget(self.delete_exercise_button)
        self.controls_layout.addWidget(self.start_all_button)
        self.controls_layout.addWidget(self.stop_edge_button)
        self.controls_layout.addWidget(self.stop_all_edge_button)
        self.controls_layout.addWidget(QLabel("Set Layout:"))
        self.controls_layout.addWidget(self.layout_selector)
        self.controls_layout.addStretch()

        self.main_layout.addLayout(self.controls_layout)

        self.add_device_button.clicked.connect(self.add_device_dialog)
        self.add_exercise_button.clicked.connect(self.add_exercise_dialog)
        self.delete_exercise_button.clicked.connect(self.delete_current_exercise)
        self.start_all_button.clicked.connect(self.start_all_exercises)
        self.stop_edge_button.clicked.connect(self.stop_selected_edge_runtime)
        self.stop_all_edge_button.clicked.connect(self.stop_all_edge_runtimes)
        self.layout_selector.currentIndexChanged.connect(self.change_camera_layout)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.exercise_pages = []
        self._mirror_slots = {}
        self._overview_slots = {}
        self.edge_device_manager = EdgeDeviceManager(project_path())
        self.edge_devices = []
        self._edge_deploy_worker = None
        self._edge_stop_worker = None
        self._sync_pipeline_worker = None
        self._sync_force_transfer_pending = False
        self.is_edge_mode = os.environ.get("SMART_MIRROR_EDGE_MODE", "0") == "1"
        self.edge_allow_close = os.environ.get("SMART_MIRROR_EDGE_ALLOW_CLOSE", "0") == "1"
        self._is_shutting_down = False
        self.aruco_dict_type = "DICT_5X5_100"
        self.system_config = {
            "gym_id": "",
            "main_system_id": "",
            "node_role": "EDGE" if self.is_edge_mode else "MAIN",
            "mirror_id": os.getenv("SMART_MIRROR_MIRROR_ID", ""),
        }
        self.server_sync_config = {
            "enabled": not self.is_edge_mode,
            "server_url": "http://127.0.0.1:8000",
            "ingest_endpoint": "/api/ingest/main-batch/",
            "api_key": "local-test-key",
            "gym_id": "",
            "main_system_id": "",
            "timeout_seconds": 20,
        }
        self._last_sync_transfer_at = 0.0
        self._sync_transfer_min_gap_sec = 15.0
        self._pending_registration_profiles = {}
        self._dashboard_dirty = {
            "summary": True,
            "activities": True,
            "members": True,
        }
        self._last_dashboard_refresh_at = 0.0
        self._dashboard_refresh_min_gap_sec = max(
            0.2,
            float(os.getenv("SMART_MIRROR_DASHBOARD_MIN_REFRESH_SEC", "1.0")),
        )
        self._last_status_message = ""
        self._last_status_ts = 0.0
        self.local_camera_as_edge = False
        self.load_config()
        if (self.system_config.get("node_role") or "MAIN").upper() == "MAIN":
            if not self.system_config.get("gym_id"):
                self.system_config["gym_id"] = str(uuid.uuid4())
            if not self.system_config.get("main_system_id"):
                self.system_config["main_system_id"] = str(uuid.uuid4())
            self.server_sync_config["gym_id"] = self.server_sync_config.get("gym_id") or self.system_config["gym_id"]
            self.server_sync_config["main_system_id"] = self.server_sync_config.get("main_system_id") or self.system_config["main_system_id"]
            auto_pipeline = os.getenv("SMART_GYM_AUTOMATIC_PIPELINE", "1") == "1"
            if auto_pipeline:
                self.server_sync_config["enabled"] = True
            else:
                self.server_sync_config["enabled"] = bool(self.server_sync_config.get("enabled", True))
        if not self.system_config.get("mirror_id"):
            self.system_config["mirror_id"] = str(uuid.uuid4())
        os.environ["SMART_MIRROR_NODE_ROLE"] = self.system_config.get("node_role", "MAIN")
        if self.system_config.get("gym_id"):
            os.environ["SMART_MIRROR_GYM_ID"] = self.system_config["gym_id"]
        if self.system_config.get("main_system_id"):
            os.environ["SMART_MIRROR_MAIN_SYSTEM_ID"] = self.system_config["main_system_id"]
        if self.system_config.get("mirror_id"):
            os.environ["SMART_MIRROR_MIRROR_ID"] = self.system_config["mirror_id"]
        self.local_camera_as_edge = (
            (self.system_config.get("node_role") or "MAIN").upper() == "MAIN"
            and os.getenv("SMART_MIRROR_LOCAL_CAMERA_AS_EDGE", "0") == "1"
        )
        self.server_sync_client = MainToServerSyncClient(self.db_handler, self.server_sync_config)
        self._ensure_edge_default_exercise()
        self.update_overview_tab()
        self.save_config()

        if self.is_edge_mode:
            self.add_device_button.setEnabled(False)
            self.add_device_button.setVisible(False)
            self.stop_edge_button.setEnabled(False)
            self.stop_edge_button.setVisible(False)
            self.stop_all_edge_button.setEnabled(False)
            self.stop_all_edge_button.setVisible(False)
            if os.environ.get("SMART_MIRROR_EDGE_AUTOSTART", "0") == "1":
                QTimer.singleShot(1800, self.start_all_exercises)

        self.sync_timer = QTimer(self)
        self.sync_timer.timeout.connect(lambda: self.sync_local_data_to_postgres(force_transfer=False))
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

    def _mark_dashboard_dirty(
        self,
        summary=False,
        activities=False,
        members=False,
        all_sections=False,
    ):
        if all_sections:
            self._dashboard_dirty["summary"] = True
            self._dashboard_dirty["activities"] = True
            self._dashboard_dirty["members"] = True
            return
        if summary:
            self._dashboard_dirty["summary"] = True
        if activities:
            self._dashboard_dirty["activities"] = True
        if members:
            self._dashboard_dirty["members"] = True

    def _is_members_tab_active(self):
        return self.tabs.currentWidget() == self.member_list_page

    @Slot(int)
    def _on_tab_changed(self, _index):
        if self._is_shutting_down:
            return
        if self._is_members_tab_active() and self._dashboard_dirty.get("members"):
            self.member_list_page.load_members(force=True)
            self._dashboard_dirty["members"] = False

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
            self.system_config = {
                "gym_id": data.get("system", {}).get("gym_id", ""),
                "main_system_id": data.get("system", {}).get("main_system_id", ""),
                "node_role": data.get("system", {}).get(
                    "node_role",
                    "EDGE" if self.is_edge_mode else "MAIN",
                ),
                "mirror_id": data.get("system", {}).get(
                    "mirror_id",
                    os.getenv("SMART_MIRROR_MIRROR_ID", ""),
                ),
            }
            self.server_sync_config = {
                "enabled": bool(data.get("server_sync", {}).get("enabled", True)),
                "server_url": data.get("server_sync", {}).get("server_url", "http://127.0.0.1:8000"),
                "ingest_endpoint": data.get("server_sync", {}).get(
                    "ingest_endpoint",
                    "/api/ingest/main-batch/",
                ),
                "api_key": data.get("server_sync", {}).get("api_key", "local-test-key"),
                "gym_id": data.get("server_sync", {}).get(
                    "gym_id",
                    data.get("system", {}).get("gym_id", ""),
                ),
                "main_system_id": data.get("server_sync", {}).get(
                    "main_system_id",
                    data.get("system", {}).get("main_system_id", ""),
                ),
                "timeout_seconds": float(data.get("server_sync", {}).get("timeout_seconds", 20)),
            }
            self.edge_devices = []
            for item in data.get("devices", []):
                cfg_username = item.get("username", "")
                cfg_pg_user = (item.get("pg_user") or "").strip()
                if not cfg_pg_user:
                    cfg_pg_user = (
                        cfg_username
                        or os.getenv("SMART_MIRROR_DEFAULT_PG_USER", os.getenv("PGUSER", "")).strip()
                        or getpass.getuser()
                    )
                if not cfg_pg_user:
                    cfg_pg_user = getpass.getuser()
                self.edge_devices.append(
                    {
                        "ip": item.get("ip", ""),
                        "hostname": item.get("hostname", ""),
                        "interface": item.get("interface", ""),
                        "linux_verified": bool(item.get("linux_verified", False)),
                        "username": item.get("username", ""),
                        "password": item.get("password", ""),
                        "remote_dir": item.get("remote_dir", "~/smart_mirror_edge"),
                        "display": item.get("display", ":0"),
                        "camera_index": resolve_edge_camera_index(item),
                        "install_deps": bool(item.get("install_deps", False)),
                        "setup_postgres": bool(item.get("setup_postgres", True)),
                        "db_backend": item.get("db_backend", "postgres"),
                        "pg_db": item.get("pg_db", "gym_edge_db"),
                        "pg_user": cfg_pg_user,
                        "pg_password": item.get("pg_password") or "admin",
                        "pg_host": item.get("pg_host", "localhost"),
                        "pg_port": item.get("pg_port", "5432"),
                        "mirror_id": item.get("mirror_id", ""),
                        "gym_id": item.get(
                            "gym_id",
                            data.get("system", {}).get("gym_id", ""),
                        ),
                        "main_system_id": item.get(
                            "main_system_id",
                            data.get("system", {}).get("main_system_id", ""),
                        ),
                        "last_deployed_at": item.get("last_deployed_at", ""),
                    }
                )
            self.aruco_dict_type = ArucoDetector.normalize_dict_type(
                data.get("aruco_dict_type", "DICT_5X5_100")
            )
            edge_camera_index_override = (
                self._edge_camera_index_override() if self.is_edge_mode else None
            )
            exercises = data.get("exercises", [])
            for ex in exercises:
                if edge_camera_index_override is not None:
                    camera_index = edge_camera_index_override
                else:
                    camera_index = int(ex.get("camera_index", 0))
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
                json.dump(
                    {
                        "aruco_dict_type": self.aruco_dict_type,
                        "system": self.system_config,
                        "server_sync": self.server_sync_config,
                        "exercises": [],
                        "devices": [],
                    },
                    f,
                    indent=4,
                )

    def save_config(self):
        exercises = []
        for (page, cam_idx, ex, user) in self.exercise_pages:
            exercises.append({"camera_index": cam_idx, "exercise": ex, "user_name": user})
        data = {
            "aruco_dict_type": self.aruco_dict_type,
            "system": self.system_config,
            "server_sync": self.server_sync_config,
            "exercises": exercises,
            "devices": self.edge_devices,
        }
        with open(self.CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)

    def _ensure_edge_default_exercise(self):
        if not self.is_edge_mode:
            return
        if self.exercise_pages:
            return
        if not exercise_config:
            return
        default_exercise = list(exercise_config.keys())[0]
        default_camera_index = 0
        edge_camera_index_override = self._edge_camera_index_override()
        if edge_camera_index_override is not None:
            default_camera_index = edge_camera_index_override
        self.add_exercise_page(
            camera_index=default_camera_index,
            exercise=default_exercise,
            user_name=None,
            start_immediately=False,
            aruco_dict_type=self.aruco_dict_type,
        )
        self.save_config()

    def _edge_camera_index_override(self):
        raw_value = (
            os.getenv("SMART_MIRROR_EDGE_CAMERA_INDEX", "")
            or os.getenv("SMART_MIRROR_CAMERA_INDEX", "")
        ).strip()
        if not raw_value:
            return None
        try:
            return max(0, int(raw_value))
        except ValueError:
            logging.warning("Invalid SMART_MIRROR_EDGE_CAMERA_INDEX value: %s", raw_value)
            return None

    def add_exercise_dialog(self):
        available_cams = detect_available_cameras(max_cameras=10)
        if not available_cams:
            available_cams = detect_available_cameras(max_cameras=10, force_refresh=True)
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

    def add_device_dialog(self):
        dialog = AddDeviceDialog(self.edge_device_manager, self)
        if dialog.exec() != QDialog.Accepted:
            return

        selection = dialog.get_selection()
        device = selection["device"]
        username = selection["username"]
        password = selection["password"]
        remote_dir = selection["remote_dir"]
        display = selection["display"]
        camera_index = int(selection.get("camera_index", 0))
        install_deps = selection["install_deps"]
        setup_postgres = True

        if not device.ip:
            QMessageBox.warning(self, "Add Device", "No edge device selected.")
            return

        self._upsert_edge_device(
            device,
            username=username,
            password=password,
            remote_dir=remote_dir,
            display=display,
            camera_index=camera_index,
            install_deps=install_deps,
            setup_postgres=setup_postgres,
        )
        self.save_config()
        message = (
            f"Edge device added.\n\n"
            f"Host: {device.hostname or 'unknown'}\n"
            f"IP: {device.ip}\n"
            f"Camera index: {camera_index}\n"
            f"Edge DB mode: postgres\n"
            f"Code deploy/start will run automatically on Start All."
        )
        self.status_bar.showMessage(f"Edge device {device.ip} added.", 8000)
        QMessageBox.information(self, "Add Device", message)

    def _upsert_edge_device(
        self,
        device,
        username,
        password,
        remote_dir,
        display,
        install_deps,
        camera_index=0,
        setup_postgres=False,
    ):
        setup_postgres = True
        existing_record = None
        mirror_id = ""
        for existing in self.edge_devices:
            if existing.get("ip") == device.ip:
                existing_record = existing
                mirror_id = existing.get("mirror_id", "")
                break
        if not mirror_id:
            mirror_id = str(uuid.uuid4())

        existing_pg_password = ""
        if existing_record is not None:
            existing_pg_password = (existing_record.get("pg_password") or "").strip()
        default_pg_user = (
            (username or "").strip()
            or os.getenv("SMART_MIRROR_DEFAULT_PG_USER", os.getenv("PGUSER", "")).strip()
            or getpass.getuser()
        )
        # Keep pg_password optional in config; runtime falls back to admin default.
        effective_pg_password = existing_pg_password or "admin"
        effective_camera_index = resolve_edge_camera_index(
            {"camera_index": camera_index},
            fallback=resolve_edge_camera_index(existing_record or {}, fallback=0),
        )

        record = {
            "ip": device.ip,
            "hostname": device.hostname,
            "interface": device.interface,
            "linux_verified": bool(device.linux_verified),
            "username": username,
            "password": password,
            "remote_dir": remote_dir,
            "display": display,
            "camera_index": effective_camera_index,
            "install_deps": bool(install_deps),
            "setup_postgres": True,
            "db_backend": "postgres",
            "pg_db": "gym_edge_db",
            "pg_user": default_pg_user,
            "pg_password": effective_pg_password,
            "pg_host": "localhost",
            "pg_port": "5432",
            "mirror_id": mirror_id,
            "gym_id": self.system_config.get("gym_id", ""),
            "main_system_id": self.system_config.get("main_system_id", ""),
            "last_deployed_at": datetime.utcnow().isoformat(),
        }
        for idx, existing in enumerate(self.edge_devices):
            if existing.get("ip") == device.ip:
                self.edge_devices[idx] = record
                return
        self.edge_devices.append(record)

    def _start_edge_deployments(self):
        if self.is_edge_mode:
            return True
        if self._edge_stop_worker and self._edge_stop_worker.isRunning():
            QMessageBox.information(
                self,
                "Edge Runtime",
                "Edge stop operation is currently running.",
            )
            return False
        if self._edge_deploy_worker and self._edge_deploy_worker.isRunning():
            QMessageBox.information(
                self,
                "Edge Deployment",
                "Edge deployment is already running.",
            )
            return False

        if not self.edge_devices:
            return True

        self.start_all_button.setEnabled(False)
        self.stop_edge_button.setEnabled(False)
        self.stop_all_edge_button.setEnabled(False)
        self.status_bar.showMessage("Deploying edge devices over local network...")

        worker_payload = [dict(edge) for edge in self.edge_devices]
        self._edge_deploy_worker = EdgeDeployWorker(
            self.edge_device_manager,
            worker_payload,
            self,
        )
        self._edge_deploy_worker.progress_signal.connect(self.update_status)
        self._edge_deploy_worker.finished_signal.connect(self._on_edge_deploy_finished)
        self._edge_deploy_worker.start()
        return False

    @Slot(object)
    def _on_edge_deploy_finished(self, payload):
        success_ips = payload.get("success_ips", [])
        failures = payload.get("failures", [])

        deployed_count = 0
        for edge in self.edge_devices:
            if edge.get("ip") in success_ips:
                edge["last_deployed_at"] = datetime.utcnow().isoformat()
                deployed_count += 1
        self.save_config()

        if failures:
            QMessageBox.warning(
                self,
                "Edge Deployment",
                "Some edge devices failed to deploy:\n\n" + "\n".join(failures),
            )
            self.status_bar.showMessage(
                f"Deployed {deployed_count} edge device(s); {len(failures)} failed.",
                10000,
            )
        else:
            self.status_bar.showMessage(
                f"Deployed and started {deployed_count} edge device(s).",
                8000,
            )

        if self._edge_deploy_worker:
            self._edge_deploy_worker.deleteLater()
            self._edge_deploy_worker = None
        self.start_all_button.setEnabled(True)
        self.stop_edge_button.setEnabled(True)
        self.stop_all_edge_button.setEnabled(True)
        self._start_local_exercises()

    def _edge_choices(self):
        choices = []
        for edge in self.edge_devices:
            ip = (edge.get("ip") or "").strip()
            if not ip:
                continue
            hostname = (edge.get("hostname") or "unknown-host").strip()
            label = f"{hostname} ({ip})"
            choices.append((label, edge))
        return choices

    def stop_selected_edge_runtime(self):
        if self.is_edge_mode:
            return
        if self._edge_deploy_worker and self._edge_deploy_worker.isRunning():
            QMessageBox.information(
                self,
                "Edge Runtime",
                "Wait for edge deployment to finish before stopping runtime.",
            )
            return
        if self._edge_stop_worker and self._edge_stop_worker.isRunning():
            QMessageBox.information(
                self,
                "Edge Runtime",
                "Edge stop operation is already running.",
            )
            return

        choices = self._edge_choices()
        if not choices:
            QMessageBox.information(self, "Edge Runtime", "No edge devices are configured.")
            return

        labels = [label for (label, _) in choices]
        selected_label, ok = QInputDialog.getItem(
            self,
            "Stop Edge Runtime",
            "Choose an edge device to stop:",
            labels,
            0,
            False,
        )
        if not ok:
            return

        selected_devices = [dict(edge) for (label, edge) in choices if label == selected_label]
        if not selected_devices:
            return
        self._start_edge_stop(selected_devices)

    def stop_all_edge_runtimes(self):
        if self.is_edge_mode:
            return
        if self._edge_deploy_worker and self._edge_deploy_worker.isRunning():
            QMessageBox.information(
                self,
                "Edge Runtime",
                "Wait for edge deployment to finish before stopping runtime.",
            )
            return
        if self._edge_stop_worker and self._edge_stop_worker.isRunning():
            QMessageBox.information(
                self,
                "Edge Runtime",
                "Edge stop operation is already running.",
            )
            return
        if not self.edge_devices:
            QMessageBox.information(self, "Edge Runtime", "No edge devices are configured.")
            return

        reply = QMessageBox.question(
            self,
            "Stop All Edge Devices",
            "Stop runtime on all configured edge devices?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._start_edge_stop([dict(edge) for edge in self.edge_devices if edge.get("ip")])

    def _start_edge_stop(self, edge_payload):
        if not edge_payload:
            return

        self.start_all_button.setEnabled(False)
        self.stop_edge_button.setEnabled(False)
        self.stop_all_edge_button.setEnabled(False)
        self.status_bar.showMessage("Stopping edge runtime...")

        self._edge_stop_worker = EdgeStopWorker(
            self.edge_device_manager,
            edge_payload,
            self,
        )
        self._edge_stop_worker.progress_signal.connect(self.update_status)
        self._edge_stop_worker.finished_signal.connect(self._on_edge_stop_finished)
        self._edge_stop_worker.start()

    @Slot(object)
    def _on_edge_stop_finished(self, payload):
        success_ips = payload.get("success_ips", [])
        failures = payload.get("failures", [])

        if failures:
            QMessageBox.warning(
                self,
                "Edge Runtime Stop",
                "Some edge devices failed to stop:\n\n" + "\n".join(failures),
            )
            self.status_bar.showMessage(
                f"Stopped {len(success_ips)} edge device(s); {len(failures)} failed.",
                10000,
            )
        else:
            self.status_bar.showMessage(
                f"Stopped {len(success_ips)} edge device(s).",
                8000,
            )

        if self._edge_stop_worker:
            self._edge_stop_worker.deleteLater()
            self._edge_stop_worker = None
        self.start_all_button.setEnabled(True)
        self.stop_edge_button.setEnabled(True)
        self.stop_all_edge_button.setEnabled(True)

        # Immediately pull any just-flushed edge data after stop operations.
        if not self._is_shutting_down and self._can_run_transfer_pipeline():
            self._start_background_transfer_sync(force_transfer=True)

    def add_member_dialog(self):
        dialog = AddMemberDialog(self.db_handler, self)
        if dialog.exec() == QDialog.Accepted:
            self._mark_dashboard_dirty(all_sections=True)
            self.sync_local_data_to_postgres(force_transfer=False)

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
            exercise_page.worker.data_updated.connect(self._on_worker_data_updated)
            exercise_page.worker._data_updated_connected = True

    @Slot()
    def _on_worker_data_updated(self):
        self._mark_dashboard_dirty(summary=True, activities=True, members=True)
        self.sync_local_data_to_postgres(force_transfer=self._can_run_transfer_pipeline())

    def _can_run_transfer_pipeline(self):
        if self.is_edge_mode:
            return False
        return (self.system_config.get("node_role") or "MAIN").upper() == "MAIN"

    def prompt_new_user_name(self, exercise_page):
        if self.is_edge_mode:
            # Edge runtime should not block camera processing with modal prompts.
            logging.info("Unknown user detected in edge mode; registration prompt suppressed.")
            self.status_bar.showMessage(
                "Unknown user detected on edge. Registration prompt is disabled in edge mode.",
                6000,
            )
            return

        dialog = RegisterUserDialog(self)
        if dialog.exec() != QDialog.Accepted:
            self.status_bar.showMessage("User registration canceled.", 5000)
            QMessageBox.warning(self, "Registration Canceled", "User was not registered.")
            return

        member_info = dialog.get_member_info()
        username = member_info["username"]

        existing_member = self.db_handler.get_member_info(username)
        if existing_member:
            QMessageBox.warning(
                self,
                "Username Exists",
                f"Username '{username}' already exists. Please choose another username.",
            )
            return

        self._pending_registration_profiles[username] = member_info
        exercise_page.start_user_registration(username)
        self.status_bar.showMessage(
            f"Capturing face data for user: {username}",
            5000,
        )

    @Slot(str, bool)
    def handle_new_user_registration_result(self, username, success):
        pending_profile = self._pending_registration_profiles.pop(username, None)
        if not success:
            self.status_bar.showMessage(f"Failed to register user: {username}", 5000)
            QMessageBox.warning(self, "Registration Failed", f"Could not register user: {username}")
            return

        existing_member = self.db_handler.get_member_info(username)
        member_info = dict(pending_profile or {})
        member_info.setdefault("username", username)
        member_info.setdefault("email", "NA")
        member_info.setdefault("membership", "Basic")
        member_info.setdefault("joined_on", datetime.utcnow().strftime('%Y-%m-%d'))
        member_info["user_id"] = existing_member.get("user_id") if existing_member else str(uuid.uuid4())

        upsert_success = self.db_handler.insert_member_local(member_info)
        if not upsert_success:
            self.status_bar.showMessage(f"Face registered but failed to save user: {username}", 5000)
            QMessageBox.warning(self, "Registration Partial", f"Face saved, but member record failed: {username}")
            return

        self.status_bar.showMessage(f"Registered new user: {username}", 5000)
        self._mark_dashboard_dirty(all_sections=True)
        self.sync_local_data_to_postgres(force_transfer=False)

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
                    self._mark_dashboard_dirty(summary=True, members=True)
                    member = self.db_handler.get_member_info(username) or member_info
                else:
                    self.status_bar.showMessage(f"Failed to add {username} to local database.", 5000)
                    QMessageBox.warning(self, "Registration Failed", f"Could not register user: {username}")
                    return
            self.profile_page.update_profile(member or user_info)

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
        deploy_complete_immediately = self._start_edge_deployments()
        if deploy_complete_immediately:
            self._start_local_exercises()

    def _start_local_exercises(self):
        available_cams = detect_available_cameras(max_cameras=10, force_refresh=True)
        missing = []
        for (page, cam_idx, ex, _) in self.exercise_pages:
            if cam_idx not in available_cams:
                missing.append((cam_idx, ex))

        if missing:
            msg = "Some cameras are not connected:\n"
            for (ci, ex) in missing:
                msg += f"cam_{ci} for exercise {ex.replace('_', ' ').title()}\n"
            msg += "The remaining cameras will be started."
            if self.is_edge_mode:
                logging.warning(msg)
                self.status_bar.showMessage(msg, 8000)
            else:
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
                if screens:
                    self.smart_mirror_window = SmartMirrorWindow()
                    self.smart_mirror_window.set_displays(screens)
                    for (p, ci, exx, _) in self.exercise_pages:
                        if p.is_exercise_running():
                            self.smart_mirror_window.add_camera_display(ci, exx)
                            self._connect_mirror_feed(p, ci, exx)
                else:
                    logging.warning("No display detected for camera feed window.")
            else:
                for (p, ci, exx, _) in self.exercise_pages:
                    if p.is_exercise_running():
                        self.smart_mirror_window.add_camera_display(ci, exx)
                        self._connect_mirror_feed(p, ci, exx)

            if self.smart_mirror_window and not self.smart_mirror_window.isVisible():
                self.smart_mirror_window.show()
            if self.is_edge_mode:
                self.hide()

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
        normalized = str(message or "").strip()
        if not normalized:
            return
        now = time.monotonic()
        if (
            normalized == self._last_status_message
            and (now - self._last_status_ts) < 0.9
        ):
            return
        self._last_status_message = normalized
        self._last_status_ts = now
        logging.info(normalized)
        self.status_bar.showMessage(normalized, 5000)

    def update_counters(self, reps, sets):
        if self.smart_mirror_window:
            logging.debug("Counters updated reps=%s sets=%s", reps, sets)

    def _should_run_transfer_sync(self):
        if not self._can_run_transfer_pipeline():
            return False
        now = time.monotonic()
        if (now - self._last_sync_transfer_at) < self._sync_transfer_min_gap_sec:
            return False
        self._last_sync_transfer_at = now
        return True

    @staticmethod
    def _is_local_machine_ip(ip):
        ip = (ip or "").strip()
        if not ip:
            return False
        if ip in {"127.0.0.1", "localhost"}:
            return True
        local_ips = {"127.0.0.1"}
        try:
            local_ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
        except Exception:
            pass
        return ip in local_ips

    def _pull_edge_data_to_main(self, progress_callback=None):
        progress_callback = progress_callback or self.update_status
        pulled_total = 0

        for edge in self.edge_devices:
            ip = (edge.get("ip") or "").strip()
            username = (edge.get("username") or "").strip()
            password = (edge.get("password") or "").strip()
            remote_dir = (edge.get("remote_dir") or "~/smart_mirror_edge").strip()
            if not ip or not username or not password:
                continue

            ok, records, details = self.edge_device_manager.fetch_pending_exercise_data(
                ip=ip,
                username=username,
                password=password,
                remote_dir=remote_dir,
                limit=500,
                db_backend=edge.get("db_backend", "postgres"),
                pg_db=edge.get("pg_db", "gym_edge_db"),
                pg_user=resolve_edge_pg_user(edge),
                pg_password=resolve_edge_pg_password(edge),
                pg_host=edge.get("pg_host", "localhost"),
                pg_port=edge.get("pg_port", "5432"),
                progress_callback=progress_callback,
            )
            if not ok:
                progress_callback(f"Edge pull failed for {ip}: {details}")
                continue
            if not records:
                continue

            applied, applied_record_ids = self.db_handler.upsert_edge_records(
                records,
                source_edge_ip=ip,
                return_record_ids=True,
            )
            pulled_total += applied
            if applied < len(records):
                progress_callback(
                    f"Edge {ip}: applied {applied}/{len(records)} record(s); unresolved records will retry."
                )
            if applied > 0 and applied_record_ids:
                ack_ok, ack_details = self.edge_device_manager.mark_remote_records_synced(
                    ip=ip,
                    username=username,
                    password=password,
                    record_ids=applied_record_ids,
                    remote_dir=remote_dir,
                    status="SYNCED",
                    db_backend=edge.get("db_backend", "postgres"),
                    pg_db=edge.get("pg_db", "gym_edge_db"),
                    pg_user=resolve_edge_pg_user(edge),
                    pg_password=resolve_edge_pg_password(edge),
                    pg_host=edge.get("pg_host", "localhost"),
                    pg_port=edge.get("pg_port", "5432"),
                    progress_callback=progress_callback,
                )
                if not ack_ok:
                    progress_callback(f"Edge ACK failed for {ip}: {ack_details}")
                else:
                    progress_callback(f"Edge {ip}: pulled and ACKed {applied} record(s).")
        return pulled_total

    def _pull_local_edge_data_to_main(self, limit=500):
        _ = limit
        return 0

    def _sync_main_to_server(self, status_callback=None):
        status_callback = status_callback or self.update_status
        synced = 0
        if not hasattr(self, "server_sync_client"):
            return synced
        ok, message, synced = self.server_sync_client.push_pending(limit=500)
        if message:
            status_callback(message)
        if not ok:
            logging.warning("Main->Server sync issue: %s", message)
        return synced

    def _start_background_transfer_sync(self, force_transfer=False):
        if self._is_shutting_down:
            return

        if not self._can_run_transfer_pipeline():
            return

        if not (force_transfer or self._should_run_transfer_sync()):
            return

        if self._sync_pipeline_worker and self._sync_pipeline_worker.isRunning():
            if force_transfer:
                self._sync_force_transfer_pending = True
            return

        self._sync_pipeline_worker = SyncPipelineWorker(
            pull_callable=lambda progress_cb: self._pull_edge_data_to_main(progress_callback=progress_cb),
            push_callable=lambda progress_cb: self._sync_main_to_server(status_callback=progress_cb),
            parent=self,
        )
        self._sync_pipeline_worker.progress_signal.connect(self.update_status)
        self._sync_pipeline_worker.finished_signal.connect(self._on_background_transfer_finished)
        self._sync_pipeline_worker.start()

    @Slot(object)
    def _on_background_transfer_finished(self, payload):
        payload = payload or {}
        error = (payload.get("error") or "").strip()
        if error:
            logging.warning("Background transfer sync failed: %s", error)
            self.update_status(f"Background sync error: {error}")
        pulled = int(payload.get("pulled") or 0)
        if pulled > 0:
            self._mark_dashboard_dirty(all_sections=True)
            self.sync_local_data_to_postgres(force_transfer=False)

        rerun_forced = self._sync_force_transfer_pending
        self._sync_force_transfer_pending = False

        if self._sync_pipeline_worker:
            self._sync_pipeline_worker.deleteLater()
            self._sync_pipeline_worker = None

        if rerun_forced and not self._is_shutting_down:
            QTimer.singleShot(0, lambda: self._start_background_transfer_sync(force_transfer=True))

    def sync_local_data_to_postgres(self, force_transfer=False):
        if self._is_shutting_down:
            return
        try:
            now = time.monotonic()
            if (now - self._last_dashboard_refresh_at) < self._dashboard_refresh_min_gap_sec:
                if force_transfer:
                    self._last_sync_transfer_at = now
                self._start_background_transfer_sync(force_transfer=force_transfer)
                return

            self._last_dashboard_refresh_at = now
            if self._dashboard_dirty.get("summary", False):
                self.home_page.refresh_summary()
                self._dashboard_dirty["summary"] = False
            if self._dashboard_dirty.get("activities", False):
                self.home_page.load_recent_activities()
                self._dashboard_dirty["activities"] = False
            if self._dashboard_dirty.get("members", False) and self._is_members_tab_active():
                self.member_list_page.load_members()
                self._dashboard_dirty["members"] = False

            if force_transfer:
                self._last_sync_transfer_at = time.monotonic()
            self._start_background_transfer_sync(force_transfer=force_transfer)
        except Exception as exc:
            logging.error("Dashboard refresh failed: %s", exc)

    def _shutdown_runtime(self):
        if self._is_shutting_down:
            return
        self._is_shutting_down = True

        def _stop_edges_for_shutdown():
            if self.is_edge_mode:
                return

            edge_payload = [dict(edge) for edge in self.edge_devices if (edge.get("ip") or "").strip()]
            if not edge_payload:
                return

            self.status_bar.showMessage("Stopping edge runtime(s) before exit...")
            failures = []

            for edge in edge_payload:
                ip = (edge.get("ip") or "").strip()
                username = (edge.get("username") or "").strip()
                password = (edge.get("password") or "").strip()
                remote_dir = (edge.get("remote_dir") or "~/smart_mirror_edge").strip()

                if not ip:
                    continue
                if not username or not password:
                    failures.append(f"{ip}: Missing SSH username/password.")
                    continue

                self.update_status(f"Stopping edge runtime on {ip}...")
                result = self.edge_device_manager.stop_runtime(
                    ip=ip,
                    username=username,
                    password=password,
                    remote_dir=remote_dir,
                    progress_callback=self.update_status,
                )
                if not result.success:
                    details = f" ({result.details})" if result.details else ""
                    failures.append(f"{ip}: {result.message}{details}")

            if failures:
                logging.warning("Edge shutdown failures: %s", " | ".join(failures))
                self.status_bar.showMessage(
                    f"App closed with {len(failures)} edge stop failure(s). Check logs.",
                    10000,
                )
            else:
                self.status_bar.showMessage("All edge runtimes stopped.", 5000)

        if self.sync_timer and self.sync_timer.isActive():
            self.sync_timer.stop()

        if self._edge_deploy_worker and self._edge_deploy_worker.isRunning():
            self._edge_deploy_worker.requestInterruption()
            self._edge_deploy_worker.wait(3000)
        if self._edge_stop_worker and self._edge_stop_worker.isRunning():
            self._edge_stop_worker.requestInterruption()
            self._edge_stop_worker.wait(3000)
        if self._sync_pipeline_worker and self._sync_pipeline_worker.isRunning():
            self._sync_pipeline_worker.requestInterruption()
            self._sync_pipeline_worker.wait(3000)

        _stop_edges_for_shutdown()

        for (page, _, _, _) in self.exercise_pages[:]:
            self._disconnect_mirror_feed(page)
            self._disconnect_overview_feed(page)
            page.stop_exercise()

        # Final dashboard sync while DB is still open.
        try:
            if self._can_run_transfer_pipeline():
                self._pull_edge_data_to_main()
                self._sync_main_to_server()
            self._mark_dashboard_dirty(all_sections=True)
            self.home_page.refresh_summary()
            self.home_page.load_recent_activities()
            self.member_list_page.load_members(force=True)
        except Exception as exc:
            logging.error("Final dashboard refresh failed during shutdown: %s", exc)

        self.db_handler.close_connections()
        self.global_face_recognizer.close()
        if hasattr(self, "server_sync_client") and self.server_sync_client:
            try:
                self.server_sync_client.close()
            except Exception:
                pass

        if self.smart_mirror_window:
            self.smart_mirror_window.close()

    def closeEvent(self, event):
        logging.info(
            "MainWindow closeEvent received (edge_mode=%s, edge_allow_close=%s).",
            self.is_edge_mode,
            self.edge_allow_close,
        )

        if self.is_edge_mode and not self.edge_allow_close:
            logging.warning("Ignoring close event in edge mode (SMART_MIRROR_EDGE_ALLOW_CLOSE != 1).")
            self.status_bar.showMessage(
                "Edge runtime close blocked. Set SMART_MIRROR_EDGE_ALLOW_CLOSE=1 to allow closing.",
                8000,
            )
            event.ignore()
            return

        if not self.is_edge_mode:
            reply = QMessageBox.question(
                self, 'Confirm Exit',
                "Are you sure you want to exit the application?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

        self._shutdown_runtime()
        event.accept()
