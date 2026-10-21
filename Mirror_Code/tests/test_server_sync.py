import json
import unittest
import uuid
from unittest.mock import Mock, patch

import requests

from core.server_sync import MainToServerSyncClient


class ServerSyncTest(unittest.TestCase):
    def setUp(self):
        self.db = Mock()
        self.record = {"id": str(uuid.uuid4()), "user_id": str(uuid.uuid4()), "exercise": "bicep_curl",
                       "timestamp": "2026-10-18T10:00:00Z", "sets_reps": [1], "rep_data": []}
        self.db.get_pending_exercise_data.return_value = [self.record]
        self.db.mark_exercise_data_synced.return_value = True
        self.config = {"enabled": True, "server_url": "https://example.invalid", "api_key": "controller-key",
                       "gym_id": str(uuid.uuid4()), "main_system_id": str(uuid.uuid4())}
        self.client = MainToServerSyncClient(self.db, self.config)
        self.client._session.close()
        self.client._session = Mock()

    def tearDown(self):
        self.client.close()

    def response(self, payload=None, code=201):
        response = Mock(status_code=code, content=b"json", text="body")
        response.json.return_value = payload
        return response

    def test_only_applied_matching_batch_is_acknowledged(self):
        def applied_response(*args, **kwargs):
            payload = json.loads(kwargs["data"])
            self.assertFalse(kwargs["allow_redirects"])
            self.assertEqual(kwargs["headers"]["X-Main-System-Key"], "controller-key")
            self.assertEqual(payload["records"][0]["id"], self.record["id"])
            return self.response({"success": True, "status": "APPLIED", "batch_id": payload["batch_id"]})

        self.client._session.post.side_effect = applied_response
        self.assertEqual(self.client.push_pending()[::2], (True, 1))
        self.db.mark_exercise_data_synced.assert_called_once_with([self.record["id"]], status="SYNCED")

    def test_redirect_empty_body_wrong_batch_or_missing_ack_stays_pending(self):
        for response in (self.response(code=302), self.response(code=204), self.response({}),
                         self.response({"success": True}), self.response([]),
                         self.response({"success": True, "status": "RECEIVED", "batch_id": "other"}),
                         self.response({"success": True, "status": "APPLIED", "batch_id": "other"})):
            self.client._session.post.side_effect = None
            self.client._session.post.return_value = response
            self.db.mark_exercise_data_synced.reset_mock()
            self.assertFalse(self.client.push_pending()[0])
            self.db.mark_exercise_data_synced.assert_called_once_with([self.record["id"]], status="FAILED")

    def test_network_or_json_error_does_not_acknowledge(self):
        for exception in (requests.Timeout("timed out"), ValueError("invalid json")):
            self.client._session.post.side_effect = exception
            self.db.mark_exercise_data_synced.reset_mock()
            self.assertFalse(self.client.push_pending()[0])
            self.db.mark_exercise_data_synced.assert_called_once_with([self.record["id"]], status="FAILED")

    def test_disabled_string_does_not_send(self):
        self.config["enabled"] = "false"
        self.assertTrue(self.client.push_pending()[0])
        self.client._session.post.assert_not_called()
        self.db.get_pending_exercise_data.assert_not_called()

    def test_invalid_config_does_not_mutate_queue(self):
        for change in ({"timeout_seconds": 0}, {"timeout_seconds": float("nan")},
                       {"timeout_seconds": -1}, {"gym_id": "invalid"},
                       {"server_url": "file:///tmp/test"}, {"server_url": "https://user:password@example.invalid"},
                       {"api_key": ""}):
            with self.subTest(change=change), patch.dict(self.config, change), patch.dict("os.environ", {"SMART_GYM_MAIN_API_KEY": ""}):
                self.assertFalse(self.client.push_pending()[0])
        self.client._session.post.assert_not_called()
        self.db.mark_exercise_data_synced.assert_not_called()

    def test_invalid_schema_record_is_quarantined(self):
        self.record.pop("exercise")
        self.assertTrue(self.client.push_pending()[0])
        self.db.mark_exercise_data_synced.assert_called_once_with([self.record["id"]], status="INVALID")
        self.client._session.post.assert_not_called()

    def test_date_fallback_accepts_sql_timestamp(self):
        normalized = self.client._normalize_record({**self.record, "timestamp": "2026-10-18 12:30:00"})
        self.assertEqual(normalized["date"], "2026-10-18")

    def test_failed_retry_mark_does_not_escape_client(self):
        self.client._session.post.side_effect = requests.Timeout("timed out")
        self.db.mark_exercise_data_synced.side_effect = RuntimeError("database unavailable")
        self.assertFalse(self.client.push_pending()[0])

    def test_malformed_local_record_returns_failure_without_losing_queue(self):
        self.db.get_pending_exercise_data.return_value = [42]
        self.assertFalse(self.client.push_pending()[0])
        self.client._session.post.assert_not_called()
        self.db.mark_exercise_data_synced.assert_not_called()
