import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)


class MainToServerSyncClient:
    def __init__(self, db_handler, sync_config):
        self.db_handler = db_handler
        self.sync_config = sync_config or {}

    def is_enabled(self):
        return bool(self.sync_config.get("enabled", False))

    def _server_url(self):
        return str(self.sync_config.get("server_url") or "http://127.0.0.1:8000").rstrip("/")

    def _endpoint(self):
        endpoint = str(self.sync_config.get("ingest_endpoint") or "/api/ingest/main-batch/")
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return endpoint

    def _api_key(self):
        return str(self.sync_config.get("api_key") or os.getenv("SMART_GYM_MAIN_API_KEY", "")).strip()

    @staticmethod
    def _build_hash(payload_obj):
        payload_str = json.dumps(payload_obj, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_record(record):
        rec = dict(record or {})
        now_iso = datetime.now(timezone.utc).isoformat()

        timestamp = rec.get("timestamp")
        if isinstance(timestamp, str) and timestamp.strip().lower() in {"", "none", "null"}:
            timestamp = None
        if not timestamp:
            timestamp = now_iso
        rec["timestamp"] = str(timestamp)

        date_val = rec.get("date")
        if isinstance(date_val, str) and date_val.strip().lower() in {"", "none", "null"}:
            date_val = None
        if not date_val:
            date_val = rec["timestamp"].split("T", 1)[0]
        rec["date"] = str(date_val)

        sets_reps = rec.get("sets_reps")
        if not isinstance(sets_reps, list):
            sets_reps = []
        rec["sets_reps"] = sets_reps

        rep_data = rec.get("rep_data")
        if not isinstance(rep_data, list):
            rep_data = []
        rec["rep_data"] = rep_data

        set_count = rec.get("set_count")
        if set_count is None:
            set_count = len(sets_reps)
        try:
            rec["set_count"] = int(set_count)
        except Exception:
            rec["set_count"] = 0

        rec["source_node"] = (rec.get("source_node") or "EDGE")
        rec["equipment_type"] = (rec.get("equipment_type") or "FREE_WEIGHT")
        if not rec.get("source_edge_ip") and rec.get("source_Edge_ip"):
            rec["source_edge_ip"] = rec.get("source_Edge_ip")
        rec.pop("source_Edge_ip", None)

        mirror_id = rec.get("mirror_id") or os.getenv("SMART_MIRROR_MIRROR_ID")
        try:
            mirror_id = str(uuid.UUID(str(mirror_id)))
        except Exception:
            seed = f"{rec.get('main_system_id') or ''}|{rec.get('gym_id') or ''}|{rec.get('source_node') or ''}"
            mirror_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"smart-mirror-sync:{seed}"))
        rec["mirror_id"] = mirror_id
        return rec

    def build_batch_payload(self, records):
        if not records:
            return None

        gym_id = self.sync_config.get("gym_id") or os.getenv("SMART_MIRROR_GYM_ID")
        main_system_id = self.sync_config.get("main_system_id") or os.getenv("SMART_MIRROR_MAIN_SYSTEM_ID")

        batch_id = str(uuid.uuid4())
        generated_at = datetime.now(timezone.utc).isoformat()

        payload = {
            "schema_version": 1,
            "batch_id": batch_id,
            "gym_id": gym_id,
            "main_system_id": main_system_id,
            "generated_at_utc": generated_at,
            "records": records,
        }
        payload["payload_hash"] = self._build_hash(payload)
        return payload

    def push_pending(self, limit=500):
        if not self.is_enabled():
            return True, "Server sync disabled", 0

        records = self.db_handler.get_pending_exercise_data(limit=limit)
        if not records:
            return True, "No pending records", 0

        normalized_records = []
        dropped_record_ids = []
        for rec in records:
            rec_id = rec.get("id")
            if not rec.get("id") or not rec.get("user_id") or not rec.get("exercise"):
                if rec_id:
                    dropped_record_ids.append(rec_id)
                continue
            normalized_records.append(self._normalize_record(rec))

        if dropped_record_ids:
            # Permanent schema-invalid records should not block retry queue forever.
            self.db_handler.mark_exercise_data_synced(dropped_record_ids, status="INVALID")

        if not normalized_records:
            return True, "No valid pending records", 0

        payload = self.build_batch_payload(normalized_records)
        if not payload:
            return True, "No payload generated", 0

        url = f"{self._server_url()}{self._endpoint()}"
        headers = {"Content-Type": "application/json"}
        api_key = self._api_key()
        if api_key:
            headers["X-Main-System-Key"] = api_key

        record_ids = [item.get("id") for item in normalized_records if item.get("id")]

        try:
            response = requests.post(
                url,
                data=json.dumps(payload),
                headers=headers,
                timeout=float(self.sync_config.get("timeout_seconds", 20)),
            )
            if response.status_code >= 400:
                self.db_handler.mark_exercise_data_synced(record_ids, status="FAILED")
                return (
                    False,
                    f"Server sync failed HTTP {response.status_code}: {response.text[:300]}",
                    0,
                )

            response_payload = response.json() if response.content else {}
            if not response_payload.get("success", True):
                self.db_handler.mark_exercise_data_synced(record_ids, status="FAILED")
                return False, f"Server rejected payload: {response_payload}", 0

            self.db_handler.mark_exercise_data_synced(record_ids, status="SYNCED")
            synced_count = len(record_ids)
            return True, f"Synced {synced_count} record(s) to server", synced_count
        except Exception as exc:
            logger.error("Main->Server sync error: %s", exc)
            self.db_handler.mark_exercise_data_synced(record_ids, status="FAILED")
            return False, str(exc), 0
