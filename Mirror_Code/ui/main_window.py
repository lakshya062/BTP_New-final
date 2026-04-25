# ui/main_window.py

import getpass
import json
import logging
import os
import time
import uuid
from datetime import datetime
from functools import partial

from PySide6.QtCore import QSize, QTimer, Slot
from PySide6.QtGui import QFont, QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from core.aruco_detection import ArucoDetector
from core.config import exercise_config
from core.database import DatabaseHandler
from core.face_recognition import FaceRecognizer
from core.paths import project_path, resource_path, state_path
from core.server_sync import MainToServerSyncClient
from ui.add_exercise_dialog import AddExerciseDialog
from ui.add_member_dialog import AddMemberDialog
from ui.background_workers import ServerSyncWorker
from ui.camera_utils import detect_available_cameras
from ui.cameras_overview_page import CamerasOverviewPage
from ui.exercise_page import ExercisePage
from ui.home_page import HomePage
from ui.member_list_page import MemberListPage
from ui.profile_page import ProfilePage
from ui.register_user_dialog import RegisterUserDialog
from ui.smart_mirror_window import SmartMirrorWindow


class MainWindow(QMainWindow):
    CONFIG_FILE = state_path("config.json")
    LEGACY_CONFIG_FILE = project_path("config.json")

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Smart Gym Client System")
        self.setMinimumSize(1600, 1000)
        self.setFont(QFont("Avenir Next", 11))

        self._configure_runtime_defaults()

        self.db_handler = DatabaseHandler()
        self.global_face_recognizer = FaceRecognizer()
        if not self.global_face_recognizer.known_face_encodings:
            QMessageBox.warning(
                self,
                "Face Recognition",
                "No known faces loaded. All detections will be unknown until users are registered.",
            )

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setTabsClosable(False)
        self.tabs.setMovable(True)
        self.tabs.setIconSize(QSize(26, 26))
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.home_page = HomePage(self.db_handler)
        self.profile_page = ProfilePage()
        self.member_list_page = MemberListPage(self.db_handler, self.global_face_recognizer)
        self.cameras_overview_page = CamerasOverviewPage()
        self.home_page.add_member_requested.connect(self.add_member_dialog)
        self.home_page.add_exercise_requested.connect(self.add_exercise_dialog)
        self.home_page.view_reports_requested.connect(
            lambda: self.tabs.setCurrentWidget(self.member_list_page)
        )

        self.tabs.addTab(self.home_page, QIcon(resource_path("icons", "home.png")), "Home")
        self.tabs.addTab(self.profile_page, QIcon(resource_path("icons", "profile.png")), "Profile")
        self.tabs.addTab(self.member_list_page, QIcon(resource_path("icons", "members.png")), "Members")
        self.tabs.addTab(
            self.cameras_overview_page,
            QIcon(resource_path("icons", "cameras.png")),
            "Cameras Overview",
        )
        self.main_layout.addWidget(self.tabs)

        self.controls_layout = QHBoxLayout()
        self.controls_layout.setSpacing(10)
        self.controls_layout.setContentsMargins(10, 10, 10, 10)

        self.add_exercise_button = QPushButton(QIcon(resource_path("icons", "add.png")), "Add Exercise")
        self.add_exercise_button.setToolTip("Add a new exercise with a camera")
        self.delete_exercise_button = QPushButton(
            QIcon(resource_path("icons", "delete.png")),
            "Delete Exercise",
        )
        self.delete_exercise_button.setToolTip("Delete the selected exercise")
        self.start_all_button = QPushButton(QIcon(resource_path("icons", "start.png")), "Start All")
        self.start_all_button.setToolTip("Start all exercises")
        self.stop_all_button = QPushButton(QIcon(resource_path("icons", "stop.png")), "Stop All")
        self.stop_all_button.setToolTip("Stop all exercise cameras")
        self.layout_selector = QComboBox()
        self.layout_selector.addItems(["2 Screens", "4 Screens", "8 Screens", "16 Screens"])
        self.layout_selector.setCurrentIndex(1)
        self.layout_selector.setToolTip("Select camera overview layout")
        self.layout_selector.currentIndexChanged.connect(self.change_camera_layout)

        self.controls_layout.addWidget(self.add_exercise_button)
        self.controls_layout.addWidget(self.delete_exercise_button)
        self.controls_layout.addWidget(self.start_all_button)
        self.controls_layout.addWidget(self.stop_all_button)
        self.controls_layout.addWidget(QLabel("Set Layout:"))
        self.controls_layout.addWidget(self.layout_selector)
        self.controls_layout.addStretch()
        self.main_layout.addLayout(self.controls_layout)

        self.add_exercise_button.clicked.connect(self.add_exercise_dialog)
        self.delete_exercise_button.clicked.connect(self.delete_current_exercise)
        self.start_all_button.clicked.connect(self.start_all_exercises)
        self.stop_all_button.clicked.connect(self.stop_all_exercises)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.exercise_pages = []
        self._mirror_slots = {}
        self._overview_slots = {}
        self._mirror_windows = {}
        self._latest_display_state = {}
        self._latest_counter_values = {}
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
        self._is_shutting_down = False
        self._sync_worker = None
        self._sync_force_pending = False
        self.aruco_dict_type = "DICT_5X5_100"
        self.system_config = {
            "gym_id": "",
            "main_system_id": "",
            "node_role": "MAIN",
            "mirror_id": os.getenv("SMART_MIRROR_MIRROR_ID", ""),
        }
        self.server_sync_config = {
            "enabled": True,
            "server_url": "http://127.0.0.1:8000",
            "ingest_endpoint": "/api/ingest/main-batch/",
            "api_key": "local-test-key",
            "gym_id": "",
            "main_system_id": "",
            "timeout_seconds": 20,
        }

        self._orphaned_face_identities = self.sync_face_model_with_db()
        self.load_config()
        self._ensure_system_ids()
        self._apply_server_sync_overrides()
        self._apply_system_env()
        self.server_sync_client = MainToServerSyncClient(self.db_handler, self.server_sync_config)
        self._report_orphaned_face_identities()

        self.update_overview_tab()
        self.save_config()

        self.sync_timer = QTimer(self)
        self.sync_timer.timeout.connect(lambda: self.sync_local_data_to_postgres(force_sync=False))
        self.sync_timer.start(60000)

    def _configure_runtime_defaults(self):
        pg_default_user = os.getenv(
            "SMART_MIRROR_DEFAULT_PG_USER",
            os.getenv("PGUSER", getpass.getuser()),
        )
        pg_default_password = os.getenv("SMART_MIRROR_DEFAULT_PG_PASSWORD", "admin")
        os.environ.setdefault("SMART_MIRROR_DB_BACKEND", "postgres")
        os.environ["SMART_MIRROR_NODE_ROLE"] = "MAIN"
        os.environ.setdefault("SMART_MIRROR_PG_DB", "gym_main_db")
        os.environ.setdefault("SMART_MIRROR_PG_USER", pg_default_user)
        os.environ.setdefault("SMART_MIRROR_PG_PASSWORD", pg_default_password)
        os.environ.setdefault("SMART_MIRROR_PG_HOST", "localhost")
        os.environ.setdefault("SMART_MIRROR_PG_PORT", "5432")

    def sync_face_model_with_db(self):
        auto_import_enabled = os.getenv("SMART_MIRROR_SYNC_FACE_MODEL_TO_DB", "0") == "1"
        missing_names = []
        for name in sorted(set(self.global_face_recognizer.known_face_names)):
            member = self.db_handler.get_member_info(name)
            if member:
                continue
            missing_names.append(name)
            if not auto_import_enabled:
                continue

            joined_on = datetime.utcnow().strftime("%Y-%m-%d")
            member_info = {
                "user_id": str(uuid.uuid4()),
                "username": name,
                "email": "NA",
                "membership": "NA",
                "joined_on": joined_on,
            }
            success = self.db_handler.insert_member_local(member_info)
            if success:
                logging.info("Added %s to local db from face model.", name)
            else:
                logging.error("Failed to add %s to local db.", name)

        if missing_names and not auto_import_enabled:
            logging.warning(
                "Found %s face-model profile(s) without matching members. "
                "They will not be auto-imported. Set SMART_MIRROR_SYNC_FACE_MODEL_TO_DB=1 "
                "to restore the legacy import behavior.",
                len(missing_names),
            )
            return missing_names

        return []

    def _report_orphaned_face_identities(self):
        if not self._orphaned_face_identities:
            return
        preview = ", ".join(self._orphaned_face_identities[:3])
        if len(self._orphaned_face_identities) > 3:
            preview = f"{preview}, ..."
        self.update_status(
            "Face profiles missing member records will not be auto-imported. "
            f"Review stale identities: {preview}"
        )

    @staticmethod
    def _read_json_file(path):
        with open(path, "r", encoding="utf-8") as config_file:
            return json.load(config_file)

    def _build_runtime_config_from_legacy(self, legacy_path):
        data = self._read_json_file(legacy_path)
        server_sync = dict(data.get("server_sync", {}) or {})
        runtime_config = {
            "aruco_dict_type": data.get("aruco_dict_type", self.aruco_dict_type),
            "system": {
                "gym_id": "",
                "main_system_id": "",
                "node_role": "MAIN",
                "mirror_id": "",
            },
            "server_sync": {
                "enabled": bool(server_sync.get("enabled", True)),
                "server_url": server_sync.get("server_url", "http://127.0.0.1:8000"),
                "ingest_endpoint": server_sync.get(
                    "ingest_endpoint",
                    "/api/ingest/main-batch/",
                ),
                # Device secrets should come from the environment or the new runtime file,
                # not from a repo-local config snapshot.
                "api_key": str(os.getenv("SMART_GYM_MAIN_API_KEY") or "").strip() or "local-test-key",
                "gym_id": "",
                "main_system_id": "",
                "timeout_seconds": float(server_sync.get("timeout_seconds", 20)),
            },
            "exercises": list(data.get("exercises", []) or []),
        }
        logging.info(
            "Migrated legacy repo config from %s into per-device runtime state at %s.",
            legacy_path,
            self.CONFIG_FILE,
        )
        return runtime_config

    def _ensure_system_ids(self):
        if not self.system_config.get("gym_id"):
            self.system_config["gym_id"] = str(uuid.uuid4())
        if not self.system_config.get("main_system_id"):
            self.system_config["main_system_id"] = str(uuid.uuid4())
        if not self.system_config.get("mirror_id"):
            self.system_config["mirror_id"] = str(uuid.uuid4())
        self.system_config["node_role"] = "MAIN"
        self.server_sync_config["enabled"] = bool(self.server_sync_config.get("enabled", True))
        self.server_sync_config["gym_id"] = (
            self.server_sync_config.get("gym_id") or self.system_config["gym_id"]
        )
        self.server_sync_config["main_system_id"] = (
            self.server_sync_config.get("main_system_id") or self.system_config["main_system_id"]
        )

    def _apply_system_env(self):
        os.environ["SMART_MIRROR_NODE_ROLE"] = "MAIN"
        os.environ["SMART_MIRROR_GYM_ID"] = self.system_config["gym_id"]
        os.environ["SMART_MIRROR_MAIN_SYSTEM_ID"] = self.system_config["main_system_id"]
        os.environ["SMART_MIRROR_MIRROR_ID"] = self.system_config["mirror_id"]

    def _apply_server_sync_overrides(self):
        server_url = os.getenv("SMART_GYM_SERVER_URL", "").strip()
        if server_url:
            self.server_sync_config["server_url"] = server_url

        ingest_endpoint = os.getenv("SMART_GYM_INGEST_ENDPOINT", "").strip()
        if ingest_endpoint:
            self.server_sync_config["ingest_endpoint"] = ingest_endpoint

        api_key = os.getenv("SMART_GYM_MAIN_API_KEY", "").strip()
        if api_key:
            self.server_sync_config["api_key"] = api_key

        timeout_override = os.getenv("SMART_GYM_SERVER_TIMEOUT_SECONDS", "").strip()
        if timeout_override:
            try:
                self.server_sync_config["timeout_seconds"] = float(timeout_override)
            except ValueError:
                logging.warning(
                    "Ignoring invalid SMART_GYM_SERVER_TIMEOUT_SECONDS value: %s",
                    timeout_override,
                )

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

    def _connect_mirror_feed(self, exercise_page):
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

        slot = lambda frame, page=exercise_page: self._update_mirror_frame(page, frame)
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

        slot = (
            lambda frame, ci=camera_index, ex=exercise:
            self.cameras_overview_page.update_thumbnail(frame, ci, ex)
        )
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

    def _update_mirror_frame(self, exercise_page, frame):
        window = self._mirror_windows.get(exercise_page)
        if window is not None:
            window.update_frame(frame)

    def _close_all_mirror_windows(self):
        for window in self._mirror_windows.values():
            try:
                window.close()
            except Exception:
                pass
        self._mirror_windows = {}

    def _available_output_screens(self):
        screens = list(QGuiApplication.screens())
        if not screens:
            return []
        if len(screens) > 1:
            return screens[1:]
        return screens[:1]

    def _rebuild_mirror_windows(self):
        self._close_all_mirror_windows()

        active_pages = [
            (page, cam_idx, exercise)
            for (page, cam_idx, exercise, _user_name) in self.exercise_pages
            if page.is_exercise_running()
        ]
        if not active_pages:
            return

        screens = self._available_output_screens()
        if not screens:
            logging.warning("No display outputs detected for smart mirror windows.")
            return

        hidden_count = max(0, len(active_pages) - len(screens))
        if hidden_count > 0:
            self.status_bar.showMessage(
                f"Only {len(screens)} display output(s) available for {len(active_pages)} active feed(s).",
                8000,
            )

        for (page, cam_idx, exercise), screen in zip(active_pages, screens):
            window = SmartMirrorWindow()
            window.set_assignment(cam_idx, exercise)
            counters = self._latest_counter_values.get(page)
            if counters:
                window.update_counters(*counters)
            state = self._latest_display_state.get(page) or {}
            window.update_overlay_state(
                overlay_data=state.get("overlay_data"),
                feedback_texts=state.get("feedback_texts"),
            )
            window.configure_display(screen)
            self._mirror_windows[page] = window

    def load_config(self):
        data = None
        if os.path.exists(self.CONFIG_FILE):
            try:
                data = self._read_json_file(self.CONFIG_FILE)
            except Exception as exc:
                logging.error("Failed to read runtime config %s: %s", self.CONFIG_FILE, exc)
                data = None
        elif os.path.exists(self.LEGACY_CONFIG_FILE):
            try:
                data = self._build_runtime_config_from_legacy(self.LEGACY_CONFIG_FILE)
            except Exception as exc:
                logging.error(
                    "Failed to migrate legacy config %s: %s",
                    self.LEGACY_CONFIG_FILE,
                    exc,
                )
                data = None

        if data is not None:
            system = data.get("system", {})
            server_sync = data.get("server_sync", {})
            self.system_config = {
                "gym_id": system.get("gym_id", ""),
                "main_system_id": system.get("main_system_id", ""),
                "node_role": "MAIN",
                "mirror_id": system.get("mirror_id", os.getenv("SMART_MIRROR_MIRROR_ID", "")),
            }
            self.server_sync_config = {
                "enabled": bool(server_sync.get("enabled", True)),
                "server_url": server_sync.get("server_url", "http://127.0.0.1:8000"),
                "ingest_endpoint": server_sync.get(
                    "ingest_endpoint",
                    "/api/ingest/main-batch/",
                ),
                "api_key": server_sync.get("api_key", "local-test-key"),
                "gym_id": server_sync.get("gym_id", system.get("gym_id", "")),
                "main_system_id": server_sync.get(
                    "main_system_id",
                    system.get("main_system_id", ""),
                ),
                "timeout_seconds": float(server_sync.get("timeout_seconds", 20)),
            }
            self.aruco_dict_type = ArucoDetector.normalize_dict_type(
                data.get("aruco_dict_type", "DICT_5X5_100")
            )
            for exercise_data in data.get("exercises", []):
                self.add_exercise_page(
                    camera_index=int(exercise_data.get("camera_index", 0)),
                    exercise=exercise_data["exercise"],
                    user_name=exercise_data.get("user_name"),
                    start_immediately=False,
                    aruco_dict_type=self.aruco_dict_type,
                )
            return

        with open(self.CONFIG_FILE, "w", encoding="utf-8") as config_file:
            json.dump(
                {
                    "aruco_dict_type": self.aruco_dict_type,
                    "system": self.system_config,
                    "server_sync": self.server_sync_config,
                    "exercises": [],
                },
                config_file,
                indent=4,
            )

    def save_config(self):
        exercises = []
        for page, cam_idx, exercise, user_name in self.exercise_pages:
            exercises.append(
                {
                    "camera_index": cam_idx,
                    "exercise": exercise,
                    "user_name": user_name,
                }
            )
        data = {
            "aruco_dict_type": self.aruco_dict_type,
            "system": self.system_config,
            "server_sync": self.server_sync_config,
            "exercises": exercises,
        }
        with open(self.CONFIG_FILE, "w", encoding="utf-8") as config_file:
            json.dump(data, config_file, indent=4)

    def add_exercise_dialog(self):
        available_cams = detect_available_cameras(max_cameras=10)
        if not available_cams:
            available_cams = detect_available_cameras(max_cameras=10, force_refresh=True)
        assigned_cams = [cam for (_page, cam, _exercise, _user) in self.exercise_pages]
        available_cams = [cam for cam in available_cams if cam not in assigned_cams]

        if not available_cams:
            QMessageBox.warning(
                self,
                "No Available Cameras",
                "All available cameras are already assigned to exercises.",
            )
            return

        dialog = AddExerciseDialog(available_cams, self)
        if dialog.exec() != QDialog.Accepted:
            return

        cam_text, exercise, user_name = dialog.get_selection()
        camera_index = int(cam_text.replace("cam_", ""))
        self.add_exercise_page(camera_index, exercise, user_name=user_name)
        self.save_config()

    def add_member_dialog(self):
        dialog = AddMemberDialog(self.db_handler, self)
        if dialog.exec() == QDialog.Accepted:
            self._mark_dashboard_dirty(all_sections=True)
            self.sync_local_data_to_postgres(force_sync=True)

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
        resolved_user_name = (user_name or "").strip() or None
        if resolved_user_name and not self.db_handler.get_member_info(resolved_user_name):
            logging.warning(
                "Skipping auto-provision for exercise assignment user '%s'; "
                "no matching member exists.",
                resolved_user_name,
            )
            self.status_bar.showMessage(
                f"Assigned user '{resolved_user_name}' is missing from members. "
                "The station will wait for a valid member.",
                7000,
            )
            resolved_user_name = None

        exercise_page = ExercisePage(
            self.db_handler,
            camera_index,
            exercise,
            self.global_face_recognizer,
            assigned_user_name=resolved_user_name,
            aruco_dict_type=resolved_aruco_dict_type,
        )
        exercise_page.status_message.connect(self.update_status)
        exercise_page.counters_update.connect(
            partial(self.handle_counters_for_exercise, exercise_page)
        )
        exercise_page.display_state_update.connect(
            partial(self.handle_display_state_for_exercise, exercise_page)
        )
        exercise_page.user_recognized_signal.connect(self.handle_user_recognized)
        exercise_page.unknown_user_detected.connect(self.prompt_new_user_name)
        exercise_page.new_user_registration_signal.connect(self.handle_new_user_registration_result)
        exercise_page.worker_started.connect(lambda page=exercise_page: self.connect_data_updated_signal(page))
        exercise_page.worker_started.connect(lambda page=exercise_page: self._connect_mirror_feed(page))
        exercise_page.worker_started.connect(
            lambda page=exercise_page, ci=camera_index, ex=exercise: self._connect_overview_feed(page, ci, ex)
        )
        exercise_page.worker_started.connect(self._rebuild_mirror_windows)

        tab_label = f"{exercise.replace('_', ' ').title()} (cam_{camera_index})"
        self.tabs.addTab(exercise_page, QIcon(resource_path("icons", "exercise.png")), tab_label)
        self.exercise_pages.append((exercise_page, camera_index, exercise, resolved_user_name))

        if start_immediately:
            exercise_page.start_exercise()

        self.update_overview_tab()

    @Slot(object, int, int)
    def handle_counters_for_exercise(self, exercise_page, reps, sets_done):
        counters = (int(reps), int(sets_done))
        self._latest_counter_values[exercise_page] = counters
        window = self._mirror_windows.get(exercise_page)
        if window is not None:
            window.update_counters(*counters)

    @Slot(object, object)
    def handle_display_state_for_exercise(self, exercise_page, payload):
        normalized_payload = dict(payload or {})
        normalized_payload["overlay_data"] = dict(normalized_payload.get("overlay_data") or {})
        normalized_payload["feedback_texts"] = list(normalized_payload.get("feedback_texts") or [])
        self._latest_display_state[exercise_page] = normalized_payload
        window = self._mirror_windows.get(exercise_page)
        if window is not None:
            window.update_overlay_state(
                overlay_data=normalized_payload.get("overlay_data"),
                feedback_texts=normalized_payload.get("feedback_texts"),
            )

    @Slot()
    def connect_data_updated_signal(self, exercise_page):
        if exercise_page.worker and not getattr(exercise_page.worker, "_data_updated_connected", False):
            exercise_page.worker.data_updated.connect(self._on_worker_data_updated)
            exercise_page.worker._data_updated_connected = True

    @Slot()
    def _on_worker_data_updated(self):
        self._mark_dashboard_dirty(summary=True, activities=True, members=True)
        self.sync_local_data_to_postgres(force_sync=True)

    def prompt_new_user_name(self, exercise_page):
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
        self.status_bar.showMessage(f"Capturing face data for user: {username}", 5000)

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
        member_info.setdefault("joined_on", datetime.utcnow().strftime("%Y-%m-%d"))
        member_info["user_id"] = (
            existing_member.get("user_id") if existing_member else str(uuid.uuid4())
        )

        upsert_success = self.db_handler.insert_member_local(member_info)
        if not upsert_success:
            self.status_bar.showMessage(
                f"Face registered but failed to save user: {username}",
                5000,
            )
            QMessageBox.warning(
                self,
                "Registration Partial",
                f"Face saved, but member record failed: {username}",
            )
            return

        self.status_bar.showMessage(f"Registered new user: {username}", 5000)
        self._mark_dashboard_dirty(all_sections=True)
        self.sync_local_data_to_postgres(force_sync=True)

    def handle_user_recognized(self, user_info):
        username = user_info.get("username", "Unknown")
        if username == "Unknown":
            return

        member = self.db_handler.get_member_info(username)
        if not member:
            logging.warning(
                "Recognized face '%s' has no active member record. "
                "Auto-provisioning is disabled to avoid ghost members.",
                username,
            )
            self.status_bar.showMessage(
                f"Recognized face '{username}' is not linked to an active member.",
                7000,
            )
            self.profile_page.update_profile(user_info)
            return

        self.profile_page.update_profile(member or user_info)

    def delete_current_exercise(self):
        current_idx = self.tabs.currentIndex()
        if current_idx < 4:
            QMessageBox.information(self, "Cannot Delete", "Cannot delete default tabs.")
            return

        reply = QMessageBox.question(
            self,
            "Delete Exercise",
            "Are you sure you want to delete this exercise?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        widget = self.tabs.widget(current_idx)
        for index, (page, _cam_idx, _exercise, _user_name) in enumerate(self.exercise_pages):
            if page != widget:
                continue
            self._disconnect_mirror_feed(page)
            self._disconnect_overview_feed(page)
            page.stop_exercise()
            self.exercise_pages.pop(index)
            self._latest_display_state.pop(page, None)
            self._latest_counter_values.pop(page, None)
            break

        self.tabs.removeTab(current_idx)
        self._rebuild_mirror_windows()
        self.save_config()
        self.update_overview_tab()

    def start_all_exercises(self):
        self._start_local_exercises()

    def stop_all_exercises(self):
        stopped_any = False
        for page, _cam_idx, _exercise, _user_name in self.exercise_pages:
            self._disconnect_mirror_feed(page)
            self._disconnect_overview_feed(page)
            if page.is_exercise_running():
                page.stop_exercise()
                stopped_any = True

        self._rebuild_mirror_windows()
        self.update_overview_tab()
        if stopped_any:
            self.update_status("Stopped all exercise cameras.")
        else:
            self.update_status("No exercise cameras were running.")

    def _start_local_exercises(self):
        available_cams = detect_available_cameras(max_cameras=10, force_refresh=True)
        missing = []
        for page, cam_idx, exercise, _user_name in self.exercise_pages:
            if cam_idx not in available_cams:
                missing.append((cam_idx, exercise))

        if missing:
            lines = ["Some cameras are not connected:"]
            for cam_idx, exercise in missing:
                lines.append(f"cam_{cam_idx} for exercise {exercise.replace('_', ' ').title()}")
            lines.append("The remaining cameras will be started.")
            QMessageBox.warning(self, "Missing Cameras", "\n".join(lines))

        started_any = False
        for page, cam_idx, exercise, _user_name in self.exercise_pages:
            if cam_idx not in available_cams:
                continue
            if not page.is_exercise_running():
                page.start_exercise()
                started_any = True
            self._connect_mirror_feed(page)
            self._connect_overview_feed(page, cam_idx, exercise)

        if started_any or any(page.is_exercise_running() for page, *_rest in self.exercise_pages):
            self._rebuild_mirror_windows()

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
        active_pages = {page for (page, _cam_idx, _exercise, _user_name) in self.exercise_pages}
        for page in list(self._overview_slots.keys()):
            if page not in active_pages or not page.worker:
                self._disconnect_overview_feed(page)

        for page, cam_idx, exercise, _user_name in self.exercise_pages:
            self.cameras_overview_page.add_camera_display(cam_idx, exercise)
            if page.worker:
                self._connect_overview_feed(page, cam_idx, exercise)

    def update_status(self, message):
        normalized = str(message or "").strip()
        if not normalized:
            return
        now = time.monotonic()
        if normalized == self._last_status_message and (now - self._last_status_ts) < 0.9:
            return
        self._last_status_message = normalized
        self._last_status_ts = now
        logging.info(normalized)
        self.status_bar.showMessage(normalized, 5000)

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

    def _start_background_server_sync(self, force=False):
        if self._is_shutting_down or not self.server_sync_client.is_enabled():
            return

        if self._sync_worker and self._sync_worker.isRunning():
            if force:
                self._sync_force_pending = True
            return

        self._sync_worker = ServerSyncWorker(
            push_callable=lambda progress_cb: self._sync_main_to_server(status_callback=progress_cb),
            parent=self,
        )
        self._sync_worker.progress_signal.connect(self.update_status)
        self._sync_worker.finished_signal.connect(self._on_background_sync_finished)
        self._sync_worker.start()

    @Slot(object)
    def _on_background_sync_finished(self, payload):
        payload = payload or {}
        error = str(payload.get("error") or "").strip()
        if error:
            self.update_status(f"Background sync error: {error}")
            logging.warning("Background sync failed: %s", error)

        rerun_forced = self._sync_force_pending
        self._sync_force_pending = False

        if self._sync_worker:
            self._sync_worker.deleteLater()
            self._sync_worker = None

        if rerun_forced and not self._is_shutting_down:
            QTimer.singleShot(0, lambda: self._start_background_server_sync(force=True))

    def sync_local_data_to_postgres(self, force_sync=False):
        if self._is_shutting_down:
            return

        try:
            now = time.monotonic()
            if (now - self._last_dashboard_refresh_at) >= self._dashboard_refresh_min_gap_sec:
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

            self._start_background_server_sync(force=force_sync)
        except Exception as exc:
            logging.error("Dashboard refresh failed: %s", exc)

    def _shutdown_runtime(self):
        if self._is_shutting_down:
            return
        self._is_shutting_down = True

        if self.sync_timer and self.sync_timer.isActive():
            self.sync_timer.stop()

        if self._sync_worker and self._sync_worker.isRunning():
            self._sync_worker.requestInterruption()
            self._sync_worker.wait(3000)

        for page, _cam_idx, _exercise, _user_name in self.exercise_pages[:]:
            self._disconnect_mirror_feed(page)
            self._disconnect_overview_feed(page)
            page.stop_exercise()

        try:
            self._mark_dashboard_dirty(all_sections=True)
            self.home_page.refresh_summary()
            self.home_page.load_recent_activities()
            self.member_list_page.load_members(force=True)
            self._sync_main_to_server()
        except Exception as exc:
            logging.error("Final dashboard refresh failed during shutdown: %s", exc)

        self._close_all_mirror_windows()
        self.db_handler.close_connections()
        self.global_face_recognizer.close()
        if self.server_sync_client:
            try:
                self.server_sync_client.close()
            except Exception:
                pass

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit the application?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            event.ignore()
            return

        self._shutdown_runtime()
        event.accept()
