#!/usr/bin/env python3
import argparse
import json
import os
import sys

# Allow running as: python scripts/edge_mark_synced.py from project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.database import DatabaseHandler


def main():
    parser = argparse.ArgumentParser(description="Mark edge exercise records as synced")
    parser.add_argument("--ids", required=True, help="JSON array of UUID record IDs")
    parser.add_argument("--status", default="SYNCED")
    args = parser.parse_args()

    try:
        record_ids = json.loads(args.ids)
        if not isinstance(record_ids, list):
            raise ValueError("ids payload must be a list")
    except Exception as exc:
        json.dump({"ok": False, "error": f"invalid ids: {exc}"}, sys.stdout)
        sys.exit(1)

    db = DatabaseHandler()
    ok = db.mark_exercise_data_synced(record_ids, status=args.status)
    json.dump({"ok": bool(ok), "count": len(record_ids)}, sys.stdout)


if __name__ == "__main__":
    main()
