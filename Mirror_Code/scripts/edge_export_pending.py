#!/usr/bin/env python3
import argparse
import json
import os
import sys

# Allow running as: python scripts/edge_export_pending.py from project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.database import DatabaseHandler


def main():
    parser = argparse.ArgumentParser(description="Export pending edge exercise records as JSON")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--mark-queued", action="store_true")
    args = parser.parse_args()

    db = DatabaseHandler()
    records = db.get_pending_exercise_data(limit=args.limit)

    if args.mark_queued and records:
        ids = [item.get("id") for item in records if item.get("id")]
        db.mark_exercise_data_synced(ids, status="QUEUED")

    payload = {
        "count": len(records),
        "records": records,
    }
    json.dump(payload, sys.stdout)


if __name__ == "__main__":
    main()
