import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox
from ui.main_window import MainWindow


class MainWindowLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_first_tab_signal_during_startup_does_not_raise(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"exercises": [], "server_sync": {"enabled": False}}))
            stack.enter_context(patch.dict(os.environ, {"SMART_GYM_SERVER_SYNC_ENABLED": "0"}))
            stack.enter_context(patch.object(MainWindow, "CONFIG_FILE", str(config)))
            database = MagicMock()
            database.get_dashboard_stats.return_value = {}
            database.get_recent_activities.return_value = []
            database.get_all_members.return_value = []
            stack.enter_context(patch("ui.main_window.DatabaseHandler", return_value=database))
            recognizer = MagicMock()
            recognizer.known_face_encodings = []
            recognizer.known_face_names = []
            stack.enter_context(patch("ui.main_window.FaceRecognizer", return_value=recognizer))
            stack.enter_context(patch.object(QMessageBox, "warning", return_value=QMessageBox.Ok))
            exception_hook = stack.enter_context(patch("sys.excepthook"))
            window = MainWindow()
            try:
                window.show()
                self.app.processEvents()
                self.assertEqual(window.tabs.count(), 4)
                exception_hook.assert_not_called()
                self.assertFalse(window.server_sync_client.is_enabled())
            finally:
                window._shutdown_runtime()
                window.deleteLater()
                self.app.processEvents()
            database.close_connections.assert_called_once()

    def test_shutdown_waits_for_sync_before_closing_shared_resources(self):
        class SlowSync:
            running = True

            def isRunning(self):
                return self.running

            def requestInterruption(self):
                pass

            def wait(self, timeout=None):
                if timeout is not None:
                    return False  # Simulate an HTTP request exceeding the old 3s join.
                self.running = False
                return True

        worker = SlowSync()
        database = MagicMock()
        database.close_connections.side_effect = lambda: self.assertFalse(worker.isRunning())
        client = MagicMock()
        client.close.side_effect = lambda: self.assertFalse(worker.isRunning())
        context = SimpleNamespace(
            _is_shutting_down=False, sync_timer=None, _sync_worker=worker,
            exercise_pages=[], _mark_dashboard_dirty=MagicMock(),
            home_page=MagicMock(), member_list_page=MagicMock(),
            _sync_main_to_server=MagicMock(), _close_all_mirror_windows=MagicMock(),
            db_handler=database, global_face_recognizer=MagicMock(), server_sync_client=client,
        )
        MainWindow._shutdown_runtime(context)
        database.close_connections.assert_called_once()
        client.close.assert_called_once()
