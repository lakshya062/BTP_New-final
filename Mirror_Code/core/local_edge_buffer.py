import logging
import threading

from core.database import DatabaseHandler

logger = logging.getLogger(__name__)


class LocalEdgeBuffer:
    """
    Backward-compatible local edge buffer adapter.

    SQLite buffering is removed; this adapter now routes all operations to
    PostgreSQL through DatabaseHandler so sync state is managed in one store.
    """

    def __init__(self, db_path=None):
        _ = db_path
        self._lock = threading.Lock()
        self.db_handler = DatabaseHandler()

    def enqueue_record(self, record):
        with self._lock:
            return bool(self.db_handler.insert_exercise_data_local(record or {}))

    def fetch_pending_records(self, limit=500, mark_queued=True):
        limit = max(1, int(limit))
        with self._lock:
            records = self.db_handler.get_pending_exercise_data(limit=limit)
            if mark_queued and records:
                ids = [item.get("id") for item in records if item.get("id")]
                if ids:
                    self.db_handler.mark_exercise_data_synced(ids, status="QUEUED")
            return records

    def mark_records(self, record_ids, status="SYNCED"):
        with self._lock:
            return bool(self.db_handler.mark_exercise_data_synced(record_ids, status=status))

    def close(self):
        # Database lifecycle is shared; do not close global handler here.
        return


_LOCAL_EDGE_BUFFER = None
_LOCAL_EDGE_BUFFER_LOCK = threading.Lock()


def get_local_edge_buffer():
    global _LOCAL_EDGE_BUFFER
    with _LOCAL_EDGE_BUFFER_LOCK:
        if _LOCAL_EDGE_BUFFER is None:
            _LOCAL_EDGE_BUFFER = LocalEdgeBuffer()
        return _LOCAL_EDGE_BUFFER
