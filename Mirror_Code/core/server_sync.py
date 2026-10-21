import hashlib
import json
import logging
import math
import os
import uuid
from datetime import datetime, timezone
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)


class MainToServerSyncClient:
    def __init__(self, db_handler, sync_config):
        self.db_handler = db_handler
        self.sync_config = sync_config or {}
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def is_enabled(self):
        value = self.sync_config.get("enabled", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _server_url(self):
        return str(self.sync_config.get("server_url") or "http://127.0.0.1:8000").rstrip("/")

    def _endpoint(self):
        endpoint = str(self.sync_config.get("ingest_endpoint") or "/api/ingest/main-batch/")
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return endpoint

    def _api_key(self):
        return str(self.sync_config.get("api_key") or os.getenv("SMART_GYM_MAIN_API_KEY", "")).strip()

    def _validate_config(self):
        parsed = urlsplit(self._server_url())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("Server sync requires a valid HTTP(S) server_url without query or fragment.")
        if parsed.username or parsed.password:
            raise ValueError("Server URL must not contain credentials.")
        if not self._api_key():
            raise ValueError("Server sync requires a main system API key.")
        for name, value in (
            ("gym_id", self.sync_config.get("gym_id") or os.getenv("SMART_MIRROR_GYM_ID")),
            ("main_system_id", self.sync_config.get("main_system_id") or os.getenv("SMART_MIRROR_MAIN_SYSTEM_ID")),
        ):
            try:
                uuid.UUID(str(value))
            except (ValueError, TypeError, AttributeError):
                raise ValueError(f"Server sync requires a valid {name}.") from None
        timeout = float(self.sync_config.get("timeout_seconds", 20))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Server sync timeout_seconds must be finite and positive.")
        return timeout

    def _mark_failed(self, record_ids):
        try:
            self.db_handler.mark_exercise_data_synced(record_ids, status="FAILED")
        except Exception:
            logger.exception("Could not mark failed sync records for retry")

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
            date_val = rec["timestamp"].split("T", 1)[0].split(" ", 1)[0]
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

        rec["source_node"] = (rec.get("source_node") or "MAIN")
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

        try:
            timeout = self._validate_config()
            records = self.db_handler.get_pending_exercise_data(limit=limit)
        except Exception as exc:
            return False, str(exc), 0
        if not records:
            return True, "No pending records", 0

        try:
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

        except Exception as exc:
            logger.error("Could not prepare server sync batch: %s", exc)
            return False, str(exc), 0

        url = f"{self._server_url()}{self._endpoint()}"
        headers = {"Content-Type": "application/json"}
        api_key = self._api_key()
        if api_key:
            headers["X-Main-System-Key"] = api_key

        record_ids = [item.get("id") for item in normalized_records if item.get("id")]

        try:
            response = self._session.post(
                url,
                data=json.dumps(payload, allow_nan=False),
                headers=headers,
                timeout=timeout,
                allow_redirects=False,
            )
            if response.status_code not in {200, 201}:
                self._mark_failed(record_ids)
                return (
                    False,
                    f"Server sync failed HTTP {response.status_code}: {response.text[:300]}",
                    0,
                )

            response_payload = response.json() if response.content else {}
            if (not isinstance(response_payload, dict)
                    or response_payload.get("success") is not True
                    or response_payload.get("status") != "APPLIED"
                    or str(response_payload.get("batch_id")) != payload["batch_id"]):
                self._mark_failed(record_ids)
                return False, "Server did not acknowledge this batch as applied", 0

            if not self.db_handler.mark_exercise_data_synced(record_ids, status="SYNCED"):
                return False, "Server applied batch, but local acknowledgement could not be saved", 0
            synced_count = len(record_ids)
            return True, f"Synced {synced_count} record(s) to server", synced_count
        except Exception as exc:
            logger.error("Main->Server sync error: %s", exc)
            self._mark_failed(record_ids)
            return False, str(exc), 0

    def close(self):
        try:
            self._session.close()
        except Exception:
            pass
