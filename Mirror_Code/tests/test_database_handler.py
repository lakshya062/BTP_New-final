import os
import tempfile
import unittest
import uuid

from core.database import DatabaseHandler


class DatabaseHandlerTest(unittest.TestCase):
    def setUp(self):
        DatabaseHandler._instance = None
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "test_local_members.db")
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


if __name__ == "__main__":
    unittest.main()
