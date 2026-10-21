import os
import tempfile
import unittest
from unittest.mock import patch
import uuid

from core.database import DatabaseHandler


class DatabaseHandlerTest(unittest.TestCase):
    def setUp(self):
        DatabaseHandler._instance = None
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "test_local_members.db")
        # Exercise the retained SQLite SQL path in a disposable database; unit
        # tests must never connect to a developer's configured PostgreSQL DB.
        def connect_test_database(handler):
            handler.backend = "sqlite"
            handler._connect_sqlite()

        self.environment = patch.dict(os.environ, {"SMART_MIRROR_DB_BACKEND": "postgres"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        with patch.object(DatabaseHandler, "_connect_postgres", connect_test_database):
            self.db = DatabaseHandler(local_db_file=self.db_path)

    def tearDown(self):
        self.db.close_connections()
        DatabaseHandler._instance = None
        self.tmpdir.cleanup()

    def test_recent_activities_includes_sets_and_rep_count(self):
        user_id = str(uuid.uuid4())
        self.assertTrue(
            self.db.insert_member_local(
                {
                    "user_id": user_id,
                    "username": "alice",
                    "email": "alice@example.com",
                    "membership": "Basic",
                    "joined_on": "2026-01-01",
                }
            )
        )

        self.assertTrue(
            self.db.insert_exercise_data_local(
                {
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "exercise": "bicep_curl",
                    "set_count": 1,
                    "sets_reps": [1],
                    "rep_data": [{"start_angle": 40, "end_angle": 160, "weight": 10}],
                    "timestamp": "2026-01-01T10:00:00",
                    "date": "2026-01-01",
                }
            )
        )

        activities = self.db.get_recent_activities(limit=5)
        self.assertEqual(len(activities), 1)
        self.assertEqual(activities[0]["username"], "alice")
        self.assertEqual(activities[0]["set_count"], 1)
        self.assertEqual(activities[0]["rep_count"], 1)

    def test_failed_sync_does_not_claim_success_timestamp(self):
        user_id = str(uuid.uuid4())
        record_id = str(uuid.uuid4())
        self.assertTrue(self.db.insert_member_local({"user_id": user_id, "username": "test_sync", "joined_on": "2026-01-01"}))
        self.assertTrue(self.db.insert_exercise_data_local({"id": record_id, "user_id": user_id, "exercise": "bicep_curl"}))
        self.assertTrue(self.db.mark_exercise_data_synced([record_id], status="FAILED"))
        pending = self.db.get_pending_exercise_data()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["sync_status"], "FAILED")
        self.assertIsNone(pending[0]["synced_at"])
        self.assertTrue(self.db.mark_exercise_data_synced([record_id], status="SYNCED"))
        self.assertEqual(self.db.get_pending_exercise_data(), [])
        with self.db.lock:
            self.db.cursor.execute("SELECT synced_at FROM exercise_data WHERE id = ?", (record_id,))
            self.assertIsNotNone(self.db.cursor.fetchone()[0])

    def test_unknown_sync_status_is_rejected(self):
        with self.assertRaises(ValueError):
            self.db.mark_exercise_data_synced([str(uuid.uuid4())], status="TYPO")


class DatabaseInitializationTest(unittest.TestCase):
    def tearDown(self):
        DatabaseHandler._instance = None

    def test_failed_initialization_can_be_retried(self):
        DatabaseHandler._instance = None
        with patch.dict(os.environ, {"SMART_MIRROR_DB_BACKEND": "postgres"}), patch.object(
            DatabaseHandler, "_connect_postgres", side_effect=[RuntimeError("unavailable"), None]
        ) as connect, patch.object(DatabaseHandler, "setup_local_db"):
            with self.assertRaises(RuntimeError):
                DatabaseHandler()
            self.assertFalse(getattr(DatabaseHandler._instance, "_initialized", False))
            handler = DatabaseHandler()
            self.assertTrue(handler._initialized)
            self.assertEqual(connect.call_count, 2)


if __name__ == "__main__":
    unittest.main()
